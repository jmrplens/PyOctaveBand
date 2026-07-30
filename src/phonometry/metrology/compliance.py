#  Copyright (c) 2026. Jose M. Requena-Plens
"""
IEC 61260-1:2014 filter and IEC 61672-1:2013 weighting class verification.

**Filters.** Acceptance limits on relative attenuation transcribed from the
official text (BS EN 61260-1:2014, **Table 1**, standard pages 15-16):
octave-band breakpoint frequencies with class 1 and class 2 minimum/maximum
limits. Fractional-octave-band breakpoints are derived with Formulas (9) and
(10) (subclauses 5.10.3-5.10.4) and limits between breakpoints are interpolated
linearly in lg(Omega) per Formula (11) (subclause 5.10.6). Relative attenuation
is ``deltaA(Omega) = A(Omega) - Aref`` (Formula 8) with ``A = Lin - Lout``
(Formula 7); here ``Aref`` is the attenuation at the exact mid-band frequency
(subclause 5.9: the pass-band reference attenuation).

IEC 61260-1:2014 defines only classes 1 and 2. **Class 0** (the tightest,
laboratory-grade class) lives only in the withdrawn **IEC 61260:1995 /
EN 61260:1995 Table 1** and its US twin **ANSI S1.11-2004 Table 1**, whose
class 1/2 masks differ numerically from the 2014 edition (e.g. the 2014
pass-band reference tolerance is ±0.4 dB for class 1 vs ±0.3 dB in 1995, and
the 2014 stop-band edge minimum is +1.2 dB vs +2.0 dB in 1995). The two editions
are therefore kept as separate mask tables selected by the ``edition`` argument
(``"2014"`` default -> classes 1/2; ``"1995"`` -> classes 0/1/2). The 1995 /
ANSI-2004 octave-band table was transcribed digit-for-digit and cross-checked
between the two standards (they agree exactly).

**Weightings.** A/C/Z frequency-weighting acceptance limits transcribed from
BS EN 61672-1:2013, **Table 3** (standard page 22): the design-goal responses
and the class 1 and class 2 upper/lower limits at the 34 nominal frequencies
from 10 Hz to 20 kHz. A lower limit of ``-inf`` means only the upper limit
applies (subclause 5.5.6 checks measured deviations at the nominal frequencies).

The historical **B weighting** is verified against ANSI S1.4-1983: design
goals from the B column of **Table IV** (whose A and C columns equal IEC
61672-1:2013 Table 3 digit for digit) and tolerance limits from **Table V**,
whose instrument Types 1 and 2 fill the class 1 / class 2 verdict slots (the
stricter laboratory Type 0 mask is exercised by the CI conformance report).
The **AU weighting** is verified against IEC 61012:1990: design goals are the
sum of the nominal A response and the **Table 1** nominal U response (with the
subclause 2.2 explicit AU values at 25/31.5/40 kHz), checked against the
Table 1 tolerances for the filter as a separate unit, the tighter of the two
tolerance readings the standard offers. IEC 61012 publishes a single
tolerance set, so both verdict slots carry the same margin for AU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

from .core import OctaveFilterBank
from .parametric_filters import WeightingFilter

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

__all__ = [
    "FilterComplianceResult",
    "class_limits",
    "filter_class_compliance",
    "verify_aircraft_noise_system",
    "verify_filter_class",
    "verify_weighting_class",
    "weighting_class_limits",
]

_INF = float("inf")

_G = 10 ** (3 / 10)

# BS EN 61260-1:2014 Table 1, high side (Omega >= 1), as exponents x of the
# octave-band normalized frequency G**x with (min, max) limits per class.
# The low side mirrors these at 1/Omega (Formula 10). The band-edge rows
# G**(1/2 - epsilon) and G**(1/2 + epsilon) encode the discontinuity at the
# edge: the pass-band segment carries the max limits, the stop-band segment
# the min limits.
#
# Pass-band max limits (min is constant -0.4 dB class 1 / -0.6 dB class 2):
_PASSBAND_MAX: list[tuple[float, float, float]] = [
    # (exponent, class 1 max, class 2 max)
    (0.0, 0.4, 0.6),      # Omega = 1
    (1 / 8, 0.5, 0.7),
    (1 / 4, 0.7, 0.9),
    (3 / 8, 1.4, 1.7),
    (1 / 2, 5.3, 5.8),    # G**(1/2) - epsilon
]
_PASSBAND_MIN = {1: -0.4, 2: -0.6}

# Stop-band min limits (max is +inf):
_STOPBAND_MIN: list[tuple[float, float, float]] = [
    # (exponent, class 1 min, class 2 min)
    (1 / 2, 1.2, 0.8),    # G**(1/2) + epsilon
    (1.0, 16.6, 15.6),
    (2.0, 40.5, 39.5),
    (3.0, 60.0, 54.0),
    (4.0, 70.0, 60.0),    # and >= G**4: constant
]

# EN 61260:1995 / IEC 61260:1995 Table 1 == ANSI S1.11-2004 Table 1 (verified
# identical digit-for-digit between both standards). Same layout as the 2014
# tables above, plus a class-0 column. Pass-band min is constant per class. The
# fractional-octave breakpoint mapping is the same as the 2014 edition: 1995
# Annex B equation (10) is identical to 2014 Formula (9), so _map_breakpoint is
# reused unchanged for both editions.
_PASSBAND_MAX_1995: list[tuple[float, float, float, float]] = [
    # (exponent, class 0 max, class 1 max, class 2 max)
    (0.0, 0.15, 0.3, 0.5),   # Omega = 1
    (1 / 8, 0.2, 0.4, 0.6),
    (1 / 4, 0.4, 0.6, 0.8),
    (3 / 8, 1.1, 1.3, 1.6),
    (1 / 2, 4.5, 5.0, 5.5),  # G**(1/2) - epsilon
]
_PASSBAND_MIN_1995 = {0: -0.15, 1: -0.3, 2: -0.5}
_STOPBAND_MIN_1995: list[tuple[float, float, float, float]] = [
    # (exponent, class 0 min, class 1 min, class 2 min)
    (1 / 2, 2.3, 2.0, 1.6),  # G**(1/2) + epsilon
    (1.0, 18.0, 17.5, 16.5),
    (2.0, 42.5, 42.0, 41.0),
    (3.0, 62.0, 61.0, 55.0),
    (4.0, 75.0, 70.0, 60.0),  # and >= G**4: constant
]

# Per-edition mask spec: ordered classes (best -> worst), the three limit tables
# and the column index of each class within the (exponent, ...) rows.
_FILTER_EDITIONS: dict[str, dict[str, Any]] = {
    "2014": {
        "classes": (1, 2),
        "passband_max": _PASSBAND_MAX,
        "passband_min": _PASSBAND_MIN,
        "stopband_min": _STOPBAND_MIN,
        "col": {1: 1, 2: 2},
    },
    "1995": {
        "classes": (0, 1, 2),
        "passband_max": _PASSBAND_MAX_1995,
        "passband_min": _PASSBAND_MIN_1995,
        "stopband_min": _STOPBAND_MIN_1995,
        "col": {0: 1, 1: 2, 2: 3},
    },
}


def _map_breakpoint(exponent: float, fraction: float) -> float:
    """
    Map an octave-band breakpoint G**x to a fractional-octave-band one.

    BS EN 61260-1:2014 Formula (9): the high-frequency breakpoint for
    bandwidth designator 1/b is
    ``1 + (G**(1/(2b)) - 1) / (G**(1/2) - 1) * (Omega_h(1/1) - 1)``.
    """
    omega_octave = _G ** exponent
    scale = (_G ** (1 / (2 * fraction)) - 1) / (_G ** 0.5 - 1)
    return float(1 + scale * (omega_octave - 1))


def class_limits(
    fraction: float, filter_class: int, omega: np.ndarray, *, edition: str = "2014"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Acceptance limits on relative attenuation at normalized frequencies.

    :param fraction: Bandwidth designator denominator b (1 for octave,
        3 for one-third octave, ...).
    :param filter_class: Performance class: 1 or 2 for ``edition="2014"``;
        0, 1 or 2 for ``edition="1995"``.
    :param omega: Normalized frequencies f/fm (> 0).
    :param edition: ``"2014"`` (IEC 61260-1:2014, classes 1/2) or ``"1995"``
        (IEC 61260:1995 / ANSI S1.11-2004, classes 0/1/2).
    :return: Tuple (minimum, maximum) relative attenuation in dB per point;
        the maximum is ``+inf`` outside the pass-band.

    .. note::
        The exact band-edge point ``Omega = G^(1/2)`` is treated as pass-band.
        The 1995 edition's Table 1 prints a dedicated minimum (+2.3/+2.0/
        +1.6 dB) *at* that single frequency, which this convention relaxes to
        the pass-band minimum; the discrepancy has measure zero -- any
        continuous response violating the edge row is caught at ``edge + eps``
        by the interpolated stop-band mask. The 2014 edition defines only the
        ``G^(1/2) - eps`` and ``G^(1/2) + eps`` rows, which the masks match
        exactly.
    """
    spec = _FILTER_EDITIONS.get(edition)
    if spec is None:
        raise ValueError("edition must be '2014' or '1995'.")
    if filter_class not in spec["classes"]:
        raise ValueError(
            f"filter_class must be one of {spec['classes']} for edition '{edition}'."
        )
    if fraction <= 0:
        raise ValueError("'fraction' must be positive.")
    col = spec["col"][filter_class]
    passband_max = spec["passband_max"]
    stopband_min = spec["stopband_min"]

    omega_arr = np.asarray(omega, dtype=np.float64)
    if np.any(omega_arr <= 0):
        raise ValueError("Normalized frequencies must be positive.")
    # Formula (10): low side mirrors the high side.
    omega_h = np.where(omega_arr < 1.0, 1.0 / omega_arr, omega_arr)

    pass_x = np.array([_map_breakpoint(row[0], fraction) for row in passband_max])
    pass_y = np.array([row[col] for row in passband_max])
    stop_x = np.array([_map_breakpoint(row[0], fraction) for row in stopband_min])
    stop_y = np.array([row[col] for row in stopband_min])

    edge = pass_x[-1]  # mapped G**(1/2): the band-edge frequency ratio
    in_pass = omega_h <= edge

    minimum = np.empty_like(omega_h)
    maximum = np.empty_like(omega_h)

    # Pass-band: constant min, interpolated max (linear in lg(Omega), Formula 11).
    minimum[in_pass] = spec["passband_min"][filter_class]
    maximum[in_pass] = np.interp(np.log10(omega_h[in_pass]), np.log10(pass_x), pass_y)

    # Stop-band: interpolated min (constant beyond the last breakpoint), max +inf.
    lg = np.log10(omega_h[~in_pass])
    minimum[~in_pass] = np.interp(lg, np.log10(stop_x), stop_y)
    maximum[~in_pass] = np.inf

    return minimum, maximum


def _verify_band(
    bank: OctaveFilterBank,
    idx: int,
    spec: dict[str, Any],
    classes_ordered: tuple[int, ...],
    breakpoint_omegas: np.ndarray,
    edition: str,
    num_points: int,
) -> tuple[dict[str, Any], float]:
    """Evaluate one band against every class; return its entry and Nyquist."""
    fm = float(bank.freq[idx])
    fsd = bank.fs / float(bank.factor[idx])
    w, h = signal.sosfreqz(bank.sos[idx], worN=num_points, fs=fsd)

    # Attenuation relative to the mid-band attenuation (Formulas 7-8),
    # with the reference evaluated exactly at the mid-band frequency.
    attenuation = -20.0 * np.log10(np.abs(h) + np.finfo(float).eps)
    _, h_ref = signal.sosfreqz(bank.sos[idx], worN=np.array([fm]), fs=fsd)
    a_ref = float(-20.0 * np.log10(np.abs(h_ref[0]) + np.finfo(float).eps))
    delta_all = attenuation - a_ref

    omega = w / fm
    valid = omega > 0
    omega, delta_a = omega[valid], delta_all[valid]

    # Guarantee the Table 1 breakpoints (pass-band included) are evaluated,
    # exactly (sosfreqz at the breakpoint frequencies, not interpolated
    # off the grid, so a coarse grid cannot smooth a dip across them).
    # Cut at the processing Nyquist, not omega.max(): the sosfreqz grid
    # stops short of Nyquist and must not exclude checkable breakpoints.
    omega_nyq = fsd / 2.0 / fm
    extra = breakpoint_omegas[(breakpoint_omegas > 0) & (breakpoint_omegas <= omega_nyq)]
    if extra.size:
        _, h_extra = signal.sosfreqz(bank.sos[idx], worN=extra * fm, fs=fsd)
        att_extra = -20.0 * np.log10(np.abs(h_extra) + np.finfo(float).eps)
        omega = np.concatenate([omega, extra])
        delta_a = np.concatenate([delta_a, att_extra - a_ref])

    margins: dict[int, float] = {}
    for cls in classes_ordered:
        minimum, maximum = class_limits(bank.fraction, cls, omega, edition=edition)
        low_margin = float(np.min(delta_a - minimum))
        finite = np.isfinite(maximum)
        high_margin = (
            float(np.min(maximum[finite] - delta_a[finite])) if np.any(finite) else np.inf
        )
        margins[cls] = min(low_margin, high_margin)

    band_class: int | None = next(
        (cls for cls in classes_ordered if margins[cls] >= 0), None
    )
    band_entry: dict[str, Any] = {
        "freq": fm,
        "class": band_class,
        "checked_to_omega": float(omega_nyq),
    }
    for cls in classes_ordered:
        band_entry[f"margin_class{cls}_db"] = margins[cls]
    return band_entry, omega_nyq


def verify_filter_class(
    bank: OctaveFilterBank, num_points: int = 2 ** 15, *, edition: str = "2014"
) -> dict[str, Any]:
    """
    Verify a filter bank against the IEC 61260 class limits.

    Each band's relative attenuation (referenced to the attenuation at its
    exact mid-band frequency) is checked against every acceptance-limit class of
    the selected edition's Table 1, evaluated on a dense frequency grid up to
    the band's processing Nyquist. The Table 1 breakpoint frequencies inside
    that range are always included in the evaluation, so the pass-band
    constraints are checked even if the grid were coarse. Frequencies beyond
    the processing Nyquist cannot carry signal energy at the band's decimated
    rate (the multirate anti-aliasing filter removes them), so they are
    treated as compliant; because the Table 1 limits there are nevertheless
    not demonstrated, the returned ``range_limited`` flag is set whenever a
    band's stop-band mask extends beyond its processing Nyquist, and the
    per-band ``checked_to_omega`` records how far the check reached.

    :param bank: The filter bank to verify (its designed SOS are analyzed;
        works for stateful and stateless banks alike).
    :param num_points: Number of frequency grid points per band (>= 16).
    :param edition: ``"2014"`` (IEC 61260-1:2014, classes 1/2) or ``"1995"``
        (IEC 61260:1995 / ANSI S1.11-2004, adds the stricter class 0).
    :return: Dict with ``overall_class`` (the strictest class every band meets,
        or ``None``), ``range_limited`` (``True`` when at least one band's
        stop-band mask extends beyond its processing Nyquist, so the returned
        class attests the verified frequency range rather than the full
        Table 1 mask; see above) and ``bands``: a list of ``{"freq", "class",
        "checked_to_omega", "margin_class<c>_db"}`` for each class ``c`` of
        the edition, where a positive margin means the limits are met with
        that much room and ``checked_to_omega`` is the highest normalized
        frequency the band's verification could reach (its processing Nyquist
        over ``f_m``).
    """
    if num_points < 16:
        raise ValueError("'num_points' must be at least 16.")
    spec = _FILTER_EDITIONS.get(edition)
    if spec is None:
        raise ValueError("edition must be '2014' or '1995'.")
    classes_ordered: tuple[int, ...] = spec["classes"]  # best -> worst

    bands: list[dict[str, Any]] = []

    # Table 1 breakpoints (both sides) that must always be evaluated.
    rows = list(spec["passband_max"]) + list(spec["stopband_min"])
    breakpoint_omegas = np.array([_map_breakpoint(row[0], bank.fraction) for row in rows])
    breakpoint_omegas = np.concatenate([1.0 / breakpoint_omegas, breakpoint_omegas])

    # The outermost stop-band breakpoint (G**4 mapped to the bandwidth
    # designator): a band whose processing Nyquist lies below it cannot have
    # its full high-side stop-band mask demonstrated, so the verdict is then
    # range-limited (the multirate anti-aliasing justifies treating the
    # unreachable region as compliant, but it is not verified).
    mask_top_omega = _map_breakpoint(spec["stopband_min"][-1][0], bank.fraction)
    range_limited = False

    for idx in range(bank.num_bands):
        band_entry, omega_nyq = _verify_band(
            bank, idx, spec, classes_ordered, breakpoint_omegas, edition, num_points
        )
        if omega_nyq < mask_top_omega:
            range_limited = True
        bands.append(band_entry)

    if not bands:
        # No bands to verify: never report compliance vacuously.
        return {"overall_class": None, "range_limited": False, "bands": []}

    classes = [band["class"] for band in bands]
    # The strictest class every band meets is the worst (largest) per-band class;
    # None if any band meets no class.
    overall: int | None = None if None in classes else max(classes)

    return {"overall_class": overall, "range_limited": range_limited, "bands": bands}


@dataclass(frozen=True)
class FilterComplianceResult:
    """IEC 61260-1 class-compliance verdict of an :class:`OctaveFilterBank`.

    Wraps the dictionary of :func:`verify_filter_class` together with the
    minimal filter-bank data needed to redraw the measured relative-attenuation
    curve, so the result exposes the standard ``plot`` / ``report`` pair without
    holding a reference to the (possibly stateful) bank.

    :ivar overall_class: The strictest class every band meets (0/1/2), or
        ``None`` when at least one band meets no class of the edition.
    :ivar bands: The per-band verdict dictionaries of
        :func:`verify_filter_class` (one ``{"freq", "class",
        "margin_class<c>_db", ...}`` per band), as an immutable tuple.
    :ivar fraction: Bandwidth designator ``b`` (1 for octave, 3 for
        one-third-octave).
    :ivar edition: ``"2014"`` (IEC 61260-1:2014, classes 1/2) or ``"1995"``
        (IEC 61260:1995 / ANSI S1.11-2004, classes 0/1/2).
    :ivar sos: Per-band second-order sections of the analysed bank (one array
        per band), kept so the relative attenuation can be recomputed with
        :func:`scipy.signal.sosfreqz` exactly as the verifier does.
    :ivar band_frequencies: The exact mid-band frequencies ``f_m`` in Hz.
    :ivar factors: Per-band decimation factor; the band's processing sample
        rate is ``fs / factor`` (the multirate rate the SOS were designed at).
        Stored because the response must be evaluated at that decimated rate,
        which the verifier's public return does not expose.
    :ivar fs: The bank's full sampling rate in Hz.
    :ivar num_points: Frequency grid points per band used by the verification,
        retained so the redrawn curve matches the analysed grid.
    :ivar range_limited: ``True`` when at least one band's stop-band mask
        extends beyond its processing Nyquist frequency, so the verification
        could not exercise the full Table 1 mask there (the multirate
        anti-aliasing removes signal energy beyond it, but the limits are not
        demonstrated); the stated class then attests the verified frequency
        range and the ``.report()`` fiche prints a qualifying note.
    """

    overall_class: int | None
    bands: tuple[dict[str, Any], ...]
    fraction: int
    edition: str
    sos: tuple[np.ndarray, ...]
    band_frequencies: np.ndarray
    factors: tuple[int, ...]
    fs: float
    num_points: int
    range_limited: bool = False

    def available_classes(self) -> list[int]:
        """The performance classes carried by the per-band verdict dictionaries.

        Reads the ``margin_class<n>_db`` keys of a band verdict, so it reflects
        the edition (the 1995 edition adds class 0; the 2014 edition keeps only
        classes 1 and 2). An empty result (a bank with no bands in range)
        carries no verdicts, so this returns an empty list.
        """
        if not self.bands:
            return []
        prefix, suffix = "margin_class", "_db"
        return sorted(
            int(key[len(prefix):-len(suffix)])
            for key in self.bands[0]
            if key.startswith(prefix) and key.endswith(suffix)
        )

    def reference_class(self) -> int:
        """The class whose corridor the fiche/plot overlays.

        The achieved overall class when the bank complies, else the loosest
        class of the edition (the one it comes closest to meeting).

        :raises ValueError: If the result carries no bands, so there is no
            reference class to report.
        """
        if self.overall_class is not None:
            return self.overall_class
        classes = self.available_classes()
        if not classes:
            raise ValueError(
                "This filter-compliance result has no bands, so it has no "
                "reference class; check the bank's frequency limits."
            )
        return max(classes)

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
        """Plot the worst-margin band against its class-limit corridor.

        Draws the measured relative attenuation of the binding band over the
        acceptance corridor of the achieved (or, when non-compliant, the
        loosest) class; see :func:`phonometry._plot.metrology.plot_filter_class`.
        Requires matplotlib (``pip install phonometry[plot]``) and returns the
        :class:`~matplotlib.axes.Axes`.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.metrology import plot_filter_class

        check_language(language)
        return plot_filter_class(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an IEC 61260-1 filter-class-compliance fiche to a PDF.

        Writes a one-page accredited report: the standard-basis line, an
        optional metadata header block, a per-band classification table beside
        the mask-overlay plot (the result's own :meth:`plot`), the boxed
        class-compliance result, an optional verdict row against a supplied
        ``required_class`` and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a
            prediction fiche (body, result and disclaimer only). A supplied
            ``required_class`` drives the verdict row.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform signature; it has no effect on
            the single-layout filter-compliance fiche.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iec61260 import render_iec61260_report

        return render_iec61260_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def filter_class_compliance(
    bank: OctaveFilterBank, *, num_points: int = 2 ** 15, edition: str = "2014"
) -> FilterComplianceResult:
    """Verify a filter bank and package the verdict as a reportable result.

    Runs :func:`verify_filter_class` and stores the outcome together with the
    bank's second-order sections, mid-band frequencies, per-band decimation
    factors and sampling rate, so the returned object can redraw the measured
    relative attenuation and render an accredited ``.report()`` fiche without
    keeping a reference to the bank.

    :param bank: The filter bank to verify.
    :param num_points: Frequency grid points per band (>= 16).
    :param edition: ``"2014"`` (IEC 61260-1:2014, classes 1/2) or ``"1995"``
        (IEC 61260:1995 / ANSI S1.11-2004, adds the stricter class 0).
    :return: A :class:`FilterComplianceResult`.
    """
    verdict = verify_filter_class(bank, num_points, edition=edition)
    return FilterComplianceResult(
        overall_class=verdict["overall_class"],
        bands=tuple(verdict["bands"]),
        fraction=int(bank.fraction),
        edition=edition,
        sos=tuple(np.asarray(s, dtype=np.float64) for s in bank.sos),
        band_frequencies=np.asarray(bank.freq, dtype=np.float64),
        factors=tuple(int(f) for f in bank.factor),
        fs=float(bank.fs),
        num_points=int(num_points),
        range_limited=bool(verdict["range_limited"]),
    )


# BS EN 61672-1:2013 Table 3 (standard page 22): design-goal frequency
# weightings and class 1 / class 2 acceptance limits at the 34 nominal
# frequencies. Columns: (nominal Hz, A dB, C dB, class1 upper, class1 lower,
# class2 upper, class2 lower); Z is 0.0 dB at every frequency. A lower limit
# of -inf means only the upper limit applies.
_WEIGHTING_TABLE3: list[tuple[float, float, float, float, float, float, float]] = [
    (10.0, -70.4, -14.3, 3.0, -_INF, 5.0, -_INF),
    (12.5, -63.4, -11.2, 2.5, -_INF, 5.0, -_INF),
    (16.0, -56.7, -8.5, 2.0, -4.0, 5.0, -_INF),
    (20.0, -50.5, -6.2, 2.0, -2.0, 3.0, -3.0),
    (25.0, -44.7, -4.4, 2.0, -1.5, 3.0, -3.0),
    (31.5, -39.4, -3.0, 1.5, -1.5, 3.0, -3.0),
    (40.0, -34.6, -2.0, 1.0, -1.0, 2.0, -2.0),
    (50.0, -30.2, -1.3, 1.0, -1.0, 2.0, -2.0),
    (63.0, -26.2, -0.8, 1.0, -1.0, 2.0, -2.0),
    (80.0, -22.5, -0.5, 1.0, -1.0, 2.0, -2.0),
    (100.0, -19.1, -0.3, 1.0, -1.0, 1.5, -1.5),
    (125.0, -16.1, -0.2, 1.0, -1.0, 1.5, -1.5),
    (160.0, -13.4, -0.1, 1.0, -1.0, 1.5, -1.5),
    (200.0, -10.9, 0.0, 1.0, -1.0, 1.5, -1.5),
    (250.0, -8.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (315.0, -6.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (400.0, -4.8, 0.0, 1.0, -1.0, 1.5, -1.5),
    (500.0, -3.2, 0.0, 1.0, -1.0, 1.5, -1.5),
    (630.0, -1.9, 0.0, 1.0, -1.0, 1.5, -1.5),
    (800.0, -0.8, 0.0, 1.0, -1.0, 1.5, -1.5),
    (1000.0, 0.0, 0.0, 0.7, -0.7, 1.0, -1.0),
    (1250.0, 0.6, 0.0, 1.0, -1.0, 1.5, -1.5),
    (1600.0, 1.0, -0.1, 1.0, -1.0, 2.0, -2.0),
    (2000.0, 1.2, -0.2, 1.0, -1.0, 2.0, -2.0),
    (2500.0, 1.3, -0.3, 1.0, -1.0, 2.5, -2.5),
    (3150.0, 1.2, -0.5, 1.0, -1.0, 2.5, -2.5),
    (4000.0, 1.0, -0.8, 1.0, -1.0, 3.0, -3.0),
    (5000.0, 0.5, -1.3, 1.5, -1.5, 3.5, -3.5),
    (6300.0, -0.1, -2.0, 1.5, -2.0, 4.5, -4.5),
    (8000.0, -1.1, -3.0, 1.5, -2.5, 5.0, -5.0),
    (10000.0, -2.5, -4.4, 2.0, -3.0, 5.0, -_INF),
    (12500.0, -4.3, -6.2, 2.0, -5.0, 5.0, -_INF),
    (16000.0, -6.6, -8.5, 2.5, -16.0, 5.0, -_INF),
    (20000.0, -9.3, -11.2, 3.0, -_INF, 5.0, -_INF),
]

_WEIGHTING_COL = {"A": 1, "C": 2, "Z": None}

# ---------------------------------------------------------------------------
# ANSI S1.4-1983 - historical B weighting (dropped when IEC 61672-1 replaced
# the older sound-level-meter standards).
# ---------------------------------------------------------------------------

# ANSI S1.4-1983 Table IV (standard page 6): random-incidence relative
# response level of the B weighting at the 34 nominal frequencies. The A and
# C columns of Table IV equal IEC 61672-1:2013 Table 3 digit for digit, so
# only the B column is transcribed here. Row = (nominal Hz, B dB).
_ANSI_S14_TABLE4_B: list[tuple[float, float]] = [
    (10.0, -38.2), (12.5, -33.2), (16.0, -28.5), (20.0, -24.2),
    (25.0, -20.4), (31.5, -17.1), (40.0, -14.2), (50.0, -11.6),
    (63.0, -9.3), (80.0, -7.4), (100.0, -5.6), (125.0, -4.2),
    (160.0, -3.0), (200.0, -2.0), (250.0, -1.3), (315.0, -0.8),
    (400.0, -0.5), (500.0, -0.3), (630.0, -0.1), (800.0, 0.0),
    (1000.0, 0.0), (1250.0, 0.0), (1600.0, 0.0), (2000.0, -0.1),
    (2500.0, -0.2), (3150.0, -0.4), (4000.0, -0.7), (5000.0, -1.2),
    (6300.0, -1.9), (8000.0, -2.9), (10000.0, -4.3), (12500.0, -6.1),
    (16000.0, -8.4), (20000.0, -11.1),
]

# ANSI S1.4-1983 Table V (standard page 6): tolerance limits on relative
# response levels for Type 1 and Type 2 instruments; the verifier maps them
# to the class 1 / class 2 verdict slots. (The stricter laboratory Type 0
# column lives in tests/reference_data.py and is pinned by the CI
# conformance report.) Row = (nominal Hz, type1 upper, type1 lower,
# type2 upper, type2 lower); a -inf lower limit means upper-only.
# Transcription note, 20 Hz Type 2: the standard prints a bare "+3" there,
# read here as +3/upper-only (matching the surrounding upper-only rows); a
# plausible alternative reading is +/-3. The B-weighting response at 20 Hz
# sits only 0.05 dB below nominal, so either reading yields the same
# verdict.
_ANSI_S14_TABLE5_12: list[tuple[float, float, float, float, float]] = [
    (10.0, 4.0, -4.0, 5.0, -_INF),
    (12.5, 3.5, -3.5, 5.0, -_INF),
    (16.0, 3.0, -3.0, 5.0, -_INF),
    (20.0, 2.5, -2.5, 3.0, -_INF),
    (25.0, 2.0, -2.0, 3.0, -3.0),
    (31.5, 1.5, -1.5, 3.0, -3.0),
    (40.0, 1.5, -1.5, 2.0, -2.0),
    (50.0, 1.0, -1.0, 2.0, -2.0),
    (63.0, 1.0, -1.0, 2.0, -2.0),
    (80.0, 1.0, -1.0, 2.0, -2.0),
    (100.0, 1.0, -1.0, 1.5, -1.5),
    (125.0, 1.0, -1.0, 1.5, -1.5),
    (160.0, 1.0, -1.0, 1.5, -1.5),
    (200.0, 1.0, -1.0, 1.5, -1.5),
    (250.0, 1.0, -1.0, 1.5, -1.5),
    (315.0, 1.0, -1.0, 1.5, -1.5),
    (400.0, 1.0, -1.0, 1.5, -1.5),
    (500.0, 1.0, -1.0, 1.5, -1.5),
    (630.0, 1.0, -1.0, 1.5, -1.5),
    (800.0, 1.0, -1.0, 1.5, -1.5),
    (1000.0, 1.0, -1.0, 1.5, -1.5),
    (1250.0, 1.0, -1.0, 1.5, -1.5),
    (1600.0, 1.0, -1.0, 2.0, -2.0),
    (2000.0, 1.0, -1.0, 2.0, -2.0),
    (2500.0, 1.0, -1.0, 2.5, -2.5),
    (3150.0, 1.0, -1.0, 2.5, -2.5),
    (4000.0, 1.0, -1.0, 3.0, -3.0),
    (5000.0, 1.5, -1.5, 3.5, -3.5),
    (6300.0, 1.5, -2.0, 4.5, -4.5),
    (8000.0, 1.5, -3.0, 5.0, -5.0),
    (10000.0, 2.0, -4.0, 5.0, -_INF),
    (12500.0, 3.0, -6.0, 5.0, -_INF),
    (16000.0, 3.0, -_INF, 5.0, -_INF),
    (20000.0, 3.0, -_INF, 5.0, -_INF),
]

# ANSI S1.4-1983 Appendix C: the B weighting is the C weighting with one
# extra zero at the origin and one extra real pole at f5 (Formula C2).
_F5 = 158.48932

# ---------------------------------------------------------------------------
# IEC 61012:1990 - AU weighting (audible sound in the presence of ultrasound)
# ---------------------------------------------------------------------------

# IEC 61012:1990 Table 1 (standard page 11): nominal relative response and
# tolerances of the U weighting as a separate filter unit, at the 37 nominal
# frequencies from 10 Hz to 40 kHz. The tolerance is zero at the 1 kHz
# reference frequency (Table 1 note; IEC 651 subclause 3.7) and the -inf
# lower limit at 40 kHz means upper-only. Row = (nominal Hz, U dB, upper
# tolerance, lower tolerance).
_IEC61012_TABLE1: list[tuple[float, float, float, float]] = [
    (10.0, 0.0, 3.0, -3.0), (12.5, 0.0, 3.0, -3.0), (16.0, 0.0, 3.0, -3.0),
    (20.0, 0.0, 3.0, -3.0), (25.0, 0.0, 2.0, -2.0), (31.5, 0.0, 1.0, -1.0),
    (40.0, 0.0, 1.0, -1.0), (50.0, 0.0, 1.0, -1.0), (63.0, 0.0, 1.0, -1.0),
    (80.0, 0.0, 1.0, -1.0), (100.0, 0.0, 1.0, -1.0), (125.0, 0.0, 1.0, -1.0),
    (160.0, 0.0, 1.0, -1.0), (200.0, 0.0, 1.0, -1.0), (250.0, 0.0, 1.0, -1.0),
    (315.0, 0.0, 1.0, -1.0), (400.0, 0.0, 1.0, -1.0), (500.0, 0.0, 1.0, -1.0),
    (630.0, 0.0, 1.0, -1.0), (800.0, 0.0, 1.0, -1.0), (1000.0, 0.0, 0.0, 0.0),
    (1250.0, 0.0, 1.0, -1.0), (1600.0, 0.0, 1.0, -1.0), (2000.0, 0.0, 1.0, -1.0),
    (2500.0, 0.0, 1.0, -1.0), (3150.0, 0.0, 1.0, -1.0), (4000.0, 0.0, 1.0, -1.0),
    (5000.0, 0.0, 1.0, -1.0), (6300.0, 0.0, 1.0, -1.0), (8000.0, 0.0, 1.0, -1.0),
    (10000.0, 0.0, 1.0, -1.0), (12500.0, -2.8, 2.0, -2.0),
    (16000.0, -13.0, 3.0, -3.0), (20000.0, -25.3, 3.0, -6.0),
    (25000.0, -37.6, 3.0, -6.0), (31500.0, -49.7, 3.0, -10.0),
    (40000.0, -61.8, 3.0, -_INF),
]

# IEC 61012:1990 subclause 2.2: explicit nominal AU values at the three
# frequencies above the last IEC 651 A-weighting row (20 kHz), prescribed
# directly because A + U cannot be summed from tabulated columns there.
_IEC61012_AU_HF = {25000.0: -50.0, 31500.0: -65.4, 40000.0: -81.1}

# IEC 61012:1990 Table 2: pole locations of the U weighting (Hz).
_U_POLES_HZ = np.array(
    [
        -12200.0,
        -12200.0,
        -7850.0 + 8800.0j,
        -7850.0 - 8800.0j,
        -2900.0 + 12150.0j,
        -2900.0 - 12150.0j,
    ]
)

# IEC 61672-1:2013 Annex E pole frequencies of the analytic A/C design goals
# (E.4.1-E.4.8); identical to the constants the WeightingFilter design uses.
_F1 = 20.598997
_F2 = 107.65265
_F3 = 737.86223
_F4 = 12194.217


def _exact_base10(frequencies: np.ndarray) -> np.ndarray:
    """Exact base-10 frequencies ``1000 * 10^(n/10)`` behind nominal labels.

    IEC 61672-1:2013 Table 3 NOTE: the tabulated weightings are computed at
    the exact frequencies ``f = 1000 * 10^(0.1 (n - 30))``, not at the nominal
    labels (e.g. 15 848.9 Hz behind "16 kHz").
    """
    return np.asarray(
        10.0 ** (np.round(10.0 * np.log10(frequencies)) / 10.0), dtype=np.float64
    )


def _analytic_weighting_db(curve: str, frequencies: np.ndarray) -> np.ndarray:
    """Analytic design-goal weighting, re 1 kHz.

    For A/C(/Z) this evaluates the exact transfer-function magnitudes of IEC
    61672-1:2013 Annex E (E.4.1/E.4.2) at ``frequencies`` and normalizes to
    the 1 kHz value, reproducing every Table 3 design goal after 0.1 dB
    rounding. ``B`` adds the ANSI S1.4-1983 Appendix C Formula (C2) factor to
    the C response, and ``AU`` cascades the A response with the U low-pass
    built from the IEC 61012:1990 Table 2 poles (which reproduces every
    Table 1 nominal value within 0.05 dB).
    """
    f = np.asarray(frequencies, dtype=np.float64)
    if curve == "Z":
        return np.zeros_like(f)

    def _c_gain(x: np.ndarray) -> np.ndarray:
        x2 = x**2
        return np.asarray(
            (_F4**2 * x2) / ((x2 + _F1**2) * (x2 + _F4**2)), dtype=np.float64
        )

    def _a_gain(x: np.ndarray) -> np.ndarray:
        x2 = x**2
        return np.asarray(
            _c_gain(x) * x2 / np.sqrt((x2 + _F2**2) * (x2 + _F3**2)),
            dtype=np.float64,
        )

    def _b_gain(x: np.ndarray) -> np.ndarray:
        # ANSI S1.4-1983 Formula (C2): W_B = 10 lg(K2 f^2/(f^2 + f5^2)) + W_C
        # (the constant K2 cancels in the 1 kHz normalization below).
        return np.asarray(
            _c_gain(x) * x / np.sqrt(x**2 + _F5**2), dtype=np.float64
        )

    def _u_gain(x: np.ndarray) -> np.ndarray:
        # Magnitude of the all-pole U weighting from the Table 2 pole
        # coordinates in Hz (the 2*pi scale cancels in the normalization).
        den = np.prod(1j * x[:, None] - _U_POLES_HZ[None, :], axis=1)
        return np.asarray(1.0 / np.abs(den), dtype=np.float64)

    def _au_gain(x: np.ndarray) -> np.ndarray:
        return np.asarray(_a_gain(x) * _u_gain(x), dtype=np.float64)

    gain = {"A": _a_gain, "B": _b_gain, "C": _c_gain, "AU": _au_gain}[curve]
    ref = gain(np.asarray([1000.0]))[0]
    return np.asarray(20.0 * np.log10(gain(f) / ref), dtype=np.float64)


def weighting_class_limits(
    weighting_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IEC 61672-1:2013 Table 3 acceptance limits for a performance class.

    The limits apply to every IEC 61672-1 weighting (A, C, Z); they qualify
    the deviation of the measured relative response from the design goal at
    each nominal frequency, not the response itself. (The B and AU masks that
    ``verify_weighting_class`` uses come from ANSI S1.4-1983 Table V and
    IEC 61012:1990 Table 1 instead and are not returned here.)

    :param weighting_class: 1 or 2 (IEC 61672-1:2013 performance class).
    :return: Tuple ``(frequencies, lower, upper)`` of the 34 nominal
        frequencies (Hz) and the lower/upper deviation limits in dB. A lower
        limit of ``-inf`` means only the upper limit applies.
    """
    if weighting_class not in (1, 2):
        raise ValueError("weighting_class must be 1 or 2.")
    up_col, lo_col = (3, 4) if weighting_class == 1 else (5, 6)
    freqs = np.array([row[0] for row in _WEIGHTING_TABLE3], dtype=np.float64)
    upper = np.array([row[up_col] for row in _WEIGHTING_TABLE3], dtype=np.float64)
    lower = np.array([row[lo_col] for row in _WEIGHTING_TABLE3], dtype=np.float64)
    return freqs, lower, upper


def _curve_design_and_limits(
    curve: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Nominal frequencies, design goals and the two acceptance masks.

    A/C/Z read IEC 61672-1:2013 Table 3 (classes 1 and 2). B reads ANSI
    S1.4-1983 Table IV (design goals) and Table V (Types 1 and 2 fill the
    two mask slots). AU reads IEC 61012:1990 Table 1: the design goal is
    nominal A + nominal U (with the subclause 2.2 explicit values above
    20 kHz) and the single Table 1 tolerance set fills both mask slots.
    """
    if curve in _WEIGHTING_COL:
        col = _WEIGHTING_COL[curve]
        nominal = np.array([row[0] for row in _WEIGHTING_TABLE3], dtype=np.float64)
        design = (
            np.zeros_like(nominal)
            if col is None
            else np.array([row[col] for row in _WEIGHTING_TABLE3], dtype=np.float64)
        )
        _, lower1, upper1 = weighting_class_limits(1)
        _, lower2, upper2 = weighting_class_limits(2)
    elif curve == "B":
        nominal = np.array([row[0] for row in _ANSI_S14_TABLE4_B], dtype=np.float64)
        design = np.array([row[1] for row in _ANSI_S14_TABLE4_B], dtype=np.float64)
        upper1 = np.array([row[1] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        lower1 = np.array([row[2] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        upper2 = np.array([row[3] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
        lower2 = np.array([row[4] for row in _ANSI_S14_TABLE5_12], dtype=np.float64)
    else:  # AU
        a_design = {row[0]: row[1] for row in _WEIGHTING_TABLE3}
        nominal = np.array([row[0] for row in _IEC61012_TABLE1], dtype=np.float64)
        design = np.array(
            [
                _IEC61012_AU_HF.get(row[0], a_design.get(row[0], 0.0) + row[1])
                for row in _IEC61012_TABLE1
            ],
            dtype=np.float64,
        )
        upper1 = np.array([row[2] for row in _IEC61012_TABLE1], dtype=np.float64)
        lower1 = np.array([row[3] for row in _IEC61012_TABLE1], dtype=np.float64)
        upper2, lower2 = upper1, lower1
    return nominal, design, (lower1, upper1), (lower2, upper2)


def _weighting_response_db(wf: WeightingFilter, frequencies: np.ndarray) -> np.ndarray:
    """Relative steady-state response of *wf* in dB, normalized to 1 kHz."""
    if wf.curve == "Z" or wf.sos.size == 0:
        return np.zeros_like(frequencies)
    # The SOS are designed at the (possibly oversampled) processing rate.
    fs_proc = wf.fs * wf._oversample
    worn = np.concatenate([frequencies, [1000.0]])
    _, h = signal.sosfreqz(wf.sos, worN=worn, fs=fs_proc)
    gain_db = 20.0 * np.log10(np.abs(h) + np.finfo(float).eps)
    return np.asarray(gain_db[:-1] - gain_db[-1], dtype=np.float64)  # relative to 1 kHz


def _weighting_band_verdicts(
    freqs_nom: np.ndarray,
    deviation: np.ndarray,
    limits1: tuple[np.ndarray, np.ndarray],
    limits2: tuple[np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    """Per-band class verdicts against the Table 3 acceptance limits.

    Margin = distance to the nearer limit; a -inf lower limit makes that side
    non-binding (its term is +inf), i.e. an upper-only limit.
    """
    lower1, upper1 = limits1
    lower2, upper2 = limits2
    bands: list[dict[str, Any]] = []
    for i, fm in enumerate(freqs_nom):
        m1 = min(upper1[i] - deviation[i], deviation[i] - lower1[i])
        m2 = min(upper2[i] - deviation[i], deviation[i] - lower2[i])
        band_class = 1 if m1 >= 0 else (2 if m2 >= 0 else None)
        bands.append(
            {
                "freq": float(fm),
                "class": band_class,
                "deviation_db": float(deviation[i]),
                "margin_class1_db": float(m1),
                "margin_class2_db": float(m2),
            }
        )
    return bands


def _between_nominals_sweep(
    wf: WeightingFilter,
    freqs_exact: np.ndarray,
    limits1: tuple[np.ndarray, np.ndarray],
    limits2: tuple[np.ndarray, np.ndarray],
    sweep_points: int,
) -> dict[str, float]:
    """Subclause 5.5.7 sweep between adjacent exact nominal frequencies.

    The acceptance limits between two adjacent nominal frequencies are the
    larger of the two adjacent Table 3 limits; the design goal there is the
    analytic Annex E response.
    """
    lower1, upper1 = limits1
    lower2, upper2 = limits2
    grid = np.geomspace(freqs_exact[0], freqs_exact[-1], sweep_points)
    sweep_dev = _weighting_response_db(wf, grid) - _analytic_weighting_db(
        wf.curve, grid
    )
    seg = np.clip(np.searchsorted(freqs_exact, grid, side="right") - 1, 0,
                  freqs_exact.size - 2)
    up1 = np.maximum(upper1[seg], upper1[seg + 1])
    lo1 = np.minimum(lower1[seg], lower1[seg + 1])
    up2 = np.maximum(upper2[seg], upper2[seg + 1])
    lo2 = np.minimum(lower2[seg], lower2[seg + 1])
    sweep_m1 = np.minimum(up1 - sweep_dev, sweep_dev - lo1)
    sweep_m2 = np.minimum(up2 - sweep_dev, sweep_dev - lo2)
    worst = int(np.argmin(sweep_m1))
    return {
        "worst_freq": float(grid[worst]),
        "margin_class1_db": float(np.min(sweep_m1)),
        "margin_class2_db": float(np.min(sweep_m2)),
    }


def verify_weighting_class(
    wf: WeightingFilter, *, sweep_points: int = 4096
) -> dict[str, Any]:
    """
    Verify a frequency-weighting filter against its standard's tolerances.

    ``A``/``C``/``Z`` are checked against IEC 61672-1:2013 Table 3 (classes 1
    and 2). The historical ``B`` weighting is checked against ANSI S1.4-1983:
    Table IV design goals with the Table V tolerance limits, whose instrument
    Types 1 and 2 fill the class 1 / class 2 verdict slots (an
    ``overall_class`` of 1 then reads "ANSI S1.4-1983 Type 1"). ``AU`` is
    checked against IEC 61012:1990: design goals are nominal A + nominal U
    (Table 1, plus the subclause 2.2 explicit AU values at 25/31.5/40 kHz)
    with the single Table 1 tolerance set for the filter as a separate unit,
    so both class slots carry the same margin and ``overall_class`` is 1
    (complies) or ``None``. ``G`` is not supported here (ISO 7196 defines one
    +/-1 dB instrumentation tolerance, no class structure; the CI conformance
    report pins it), nor is ``D`` (the tolerance tables of the withdrawn
    IEC 537 did not survive it; the conformance report pins the D response
    against its published transfer function and tabulated curve).

    The filter's relative response (normalized to its 1 kHz gain) is evaluated
    at the *exact* base-10 frequency behind each nominal label below
    the Nyquist frequency (IEC 61672-1 Table 3 NOTE: the design goals are
    computed at ``f = 1000 * 10^(0.1 (n - 30))``, e.g. 15 848.9 Hz for
    "16 kHz"; IEC 61672-3:2013 subclause 13.3 tests the deviation at the same
    exact frequencies, and IEC 61012 Table 1 lists the same exact
    frequencies). The deviation from the design-goal weighting is checked
    against the two acceptance masks.

    A dense logarithmic sweep between the checked frequencies additionally
    enforces IEC 61672-1 subclause 5.5.7: at any frequency between two
    adjacent nominal frequencies, the deviation of the response from the
    analytic design goal (Annex E for A/C/Z, the ANSI S1.4-1983 Appendix C
    formulas for B, the A response cascaded with the IEC 61012 Table 2 poles
    for AU) must stay within the *larger* of the two adjacent limits. Without
    it a resonance or notch between the nominal frequencies would go
    unnoticed (for B and AU the sweep is applied as the analogous engineering
    check). Both the per-frequency verdicts and the sweep must pass for
    ``overall_class``. The sweep samples ``sweep_points`` grid frequencies; a
    violation narrower than the grid spacing could in principle fall between
    samples, so raise ``sweep_points`` for higher-Q suspects (the verdict
    attests the sampled grid, not a continuous proof).

    The response is taken from the designed second-order sections (evaluated
    with ``sosfreqz`` at their design rate), so it is exact and deterministic;
    it does not model the runtime resampling stages that ``high_accuracy``
    adds around them, whose anti-alias response is flat across the audio band
    checked here. The ``Z`` weighting is a flat bypass and always complies.

    When rows that carry a *finite lower* acceptance limit fall at or
    above the Nyquist frequency (e.g. the 8-16 kHz class 1 rows of a 16 kHz
    sampled system, or the 25-40 kHz AU rows of a 48 kHz one), they cannot be
    checked and ``range_limited`` is ``True``: the returned class then
    attests conformance over the checked frequencies only, not conformance
    over the standard's full frequency range.

    :param wf: The weighting filter to verify (``A``, ``B``, ``C``, ``AU``
        or ``Z``).
    :param sweep_points: Number of points of the 5.5.7 between-nominals sweep
        (>= 64).
    :return: Dict with ``overall_class`` (1, 2 or None), ``range_limited``
        (see above), ``bands``: a list of ``{"freq", "class", "deviation_db",
        "margin_class1_db", "margin_class2_db"}`` where ``freq`` is the
        nominal label, a positive margin means the limits are met with that
        much room, and ``between_nominals``: ``{"worst_freq",
        "margin_class1_db", "margin_class2_db"}`` for the sweep.
    """
    if wf.curve not in _WEIGHTING_COL and wf.curve not in ("B", "AU"):
        raise ValueError("Weighting curve must be 'A', 'B', 'C', 'AU' or 'Z'.")
    if sweep_points < 64:
        raise ValueError("'sweep_points' must be at least 64.")

    nyquist = wf.fs / 2.0
    nominal, design_all, limits1_all, limits2_all = _curve_design_and_limits(
        wf.curve
    )
    lower1_all, upper1_all = limits1_all
    lower2_all, upper2_all = limits2_all
    exact = _exact_base10(nominal)
    in_range = exact < nyquist

    # Any row beyond Nyquist cannot be demonstrated (its acceptance
    # limits and the adjacent between-nominal interval go unchecked), so the
    # verdict is then range-limited, not full-range conformance.
    dropped = ~in_range
    range_limited = bool(np.any(dropped))

    freqs_nom = nominal[in_range]
    freqs_exact = exact[in_range]
    design = design_all[in_range]
    response = _weighting_response_db(wf, freqs_exact)
    deviation = response - design

    lower1, upper1 = lower1_all[in_range], upper1_all[in_range]
    lower2, upper2 = lower2_all[in_range], upper2_all[in_range]

    bands = _weighting_band_verdicts(
        freqs_nom, deviation, (lower1, upper1), (lower2, upper2)
    )

    if not bands:
        return {
            "overall_class": None,
            "range_limited": range_limited,
            "bands": [],
            "between_nominals": None,
        }

    between = _between_nominals_sweep(
        wf, freqs_exact, (lower1, upper1), (lower2, upper2), sweep_points
    )

    classes = [band["class"] for band in bands]
    if all(c == 1 for c in classes) and between["margin_class1_db"] >= 0.0:
        overall: int | None = 1
    elif (
        all(c in (1, 2) for c in classes)
        and between["margin_class2_db"] >= 0.0
    ):
        overall = 2
    else:
        overall = None

    return {
        "overall_class": overall,
        "range_limited": range_limited,
        "bands": bands,
        "between_nominals": between,
    }


# ---------------------------------------------------------------------------
# IEC 61265:1995 aircraft-noise measurement-system tolerances
# ---------------------------------------------------------------------------

#: Tabulated depression/incidence angles (degrees) of IEC 61265 Table 1.
_IEC61265_ANGLES: tuple[float, ...] = (30.0, 60.0, 90.0, 120.0, 150.0)

#: IEC 61265:1995 Table 1: maximum permitted |sensitivity(0°) − sensitivity(θ)|
#: (dB) per one-third-octave band. Rows: (f_low, f_high, limits at the angles).
_IEC61265_DIRECTIONAL: tuple[tuple[float, float, tuple[float, ...]], ...] = (
    (50.0, 1600.0, (0.5, 0.5, 1.0, 1.0, 1.0)),
    (2000.0, 2000.0, (0.5, 0.5, 1.0, 1.0, 1.0)),
    (2500.0, 2500.0, (0.5, 0.5, 1.0, 1.5, 1.5)),
    (3150.0, 3150.0, (0.5, 1.0, 1.5, 2.0, 2.0)),
    (4000.0, 4000.0, (0.5, 1.0, 2.0, 2.5, 2.5)),
    (5000.0, 5000.0, (0.5, 1.5, 2.5, 3.0, 3.0)),
    (6300.0, 6300.0, (1.0, 2.0, 3.0, 4.0, 4.0)),
    (8000.0, 8000.0, (1.5, 2.5, 4.0, 5.5, 5.5)),
    (10000.0, 10000.0, (2.0, 3.5, 5.5, 6.5, 7.5)),
)


def _iec61265_directional_limit(frequency: float, angle: float) -> float:
    """The IEC 61265 Table 1 directional tolerance for a frequency and angle.

    Per subclause 4.4.2, an incidence angle between two tabulated angles takes
    the limit of the greater tabulated angle.
    """
    row = next(
        (lims for lo, hi, lims in _IEC61265_DIRECTIONAL if lo <= frequency <= hi),
        None,
    )
    if row is None:
        raise ValueError(
            "'frequency' is not an IEC 61265 tabulated one-third-octave band "
            "(50 Hz-1.6 kHz, then 2, 2.5, 3.15, 4, 5, 6.3, 8, 10 kHz)."
        )
    if angle <= 0.0 or angle > _IEC61265_ANGLES[-1]:
        raise ValueError("'angle' must lie in (0, 150] degrees.")
    col = next(i for i, a in enumerate(_IEC61265_ANGLES) if a >= angle)
    return row[col]


def verify_aircraft_noise_system(
    *,
    directional: dict[float, dict[float, float]] | None = None,
    frequency_response: dict[float, float] | None = None,
    linearity: dict[str, float] | None = None,
    resolution: float | None = None,
) -> dict[str, Any]:
    """Verify measured performance against IEC 61265:1995 tolerances.

    Each supplied measurement is checked against the standard's limit; the
    one-third-octave filtering itself is covered by the IEC 61260 class-2
    verification (subclause 4.6) and is not repeated here.

    :param directional: Microphone directional response as
        ``{frequency_hz: {angle_deg: |Δsensitivity| dB}}`` (Table 1, §4.4.2).
    :param frequency_response: System response deviations
        ``{frequency_hz: deviation_db}`` against the ±1.5 dB limit (§4.5.1).
    :param linearity: Level non-linearity ``{"reference": dB, "other": dB}``
        against the ±0.4/±0.5 dB limits (§4.5.2).
    :param resolution: Readout resolution, in dB, against the 0.1 dB limit (§4.7).
    :return: ``{"passed": bool, "checks": [{"quantity", "limit", "value", "ok",
        ...}]}``; ``passed`` is the conjunction of every check.
    :raises ValueError: If a frequency or angle is out of the tabulated range.
    """
    checks: list[dict[str, Any]] = []

    if directional is not None:
        for freq, per_angle in directional.items():
            for angle, value in per_angle.items():
                limit = _iec61265_directional_limit(float(freq), float(angle))
                checks.append(
                    {
                        "quantity": "directional_response",
                        "frequency": float(freq),
                        "angle": float(angle),
                        "limit": limit,
                        "value": float(value),
                        "ok": abs(float(value)) <= limit,
                    }
                )

    if frequency_response is not None:
        for freq, dev in frequency_response.items():
            checks.append(
                {
                    "quantity": "frequency_response",
                    "frequency": float(freq),
                    "limit": 1.5,
                    "value": float(dev),
                    "ok": abs(float(dev)) <= 1.5,
                }
            )

    if linearity is not None:
        for kind, dev in linearity.items():
            if kind not in ("reference", "other"):
                raise ValueError("linearity keys must be 'reference' or 'other'.")
            limit = 0.4 if kind == "reference" else 0.5
            checks.append(
                {
                    "quantity": f"linearity_{kind}",
                    "limit": limit,
                    "value": float(dev),
                    "ok": abs(float(dev)) <= limit,
                }
            )

    if resolution is not None:
        res = float(resolution)
        checks.append(
            {
                "quantity": "resolution",
                "limit": 0.1,
                "value": res,
                "ok": bool(np.isfinite(res)) and 0.0 <= res <= 0.1,
            }
        )

    passed = bool(checks) and all(c["ok"] for c in checks)
    return {"passed": passed, "checks": checks}
