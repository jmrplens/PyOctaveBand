#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
IEC 61260-1:2014 band-filter class verification.

Acceptance limits on relative attenuation transcribed from the
official text (BS EN 61260-1:2014, **Table 1**, standard pages 15-16):
octave-band breakpoint frequencies with class 1 and class 2 minimum/maximum
limits. Fractional-octave-band breakpoints are derived with Formulas (9) and
(10) (subclauses 5.10.3-5.10.4) and limits between breakpoints are interpolated
linearly in :math:`\log_{10} \Omega` per Formula (11) (subclause 5.10.6).
Relative attenuation is
:math:`\Delta A(\Omega) = A(\Omega) - A_{\mathrm{ref}}` (Formula 8) with
:math:`A = L_{\mathrm{in}} - L_{\mathrm{out}}`
(Formula 7); here :math:`A_{\mathrm{ref}}` is the attenuation at the exact
mid-band frequency
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

One subject: the class limits of a band filter, a mask around each mid-band
frequency the filter's own relative attenuation is measured against. The
acceptance limits of the A/B/C/AU/Z frequency weightings, which qualify a
network applied to the whole signal against a design-goal response, live in
:mod:`phonometry.filters.weighting_compliance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

from .core import OctaveFilterBank

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

__all__ = [
    "FilterComplianceResult",
    "class_limits",
    "filter_class_compliance",
    "verify_filter_class",
]

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
    (0.0, 0.4, 0.6),  # band centre (Omega of 1)
    (1 / 8, 0.5, 0.7),
    (1 / 4, 0.7, 0.9),
    (3 / 8, 1.4, 1.7),
    (1 / 2, 5.3, 5.8),  # G**(1/2) - epsilon
]
_PASSBAND_MIN = {1: -0.4, 2: -0.6}

# Stop-band min limits (max is +inf):
_STOPBAND_MIN: list[tuple[float, float, float]] = [
    # (exponent, class 1 min, class 2 min)
    (1 / 2, 1.2, 0.8),  # G**(1/2) + epsilon
    (1.0, 16.6, 15.6),
    (2.0, 40.5, 39.5),
    (3.0, 60.0, 54.0),
    (4.0, 70.0, 60.0),  # and >= G**4: constant
]

# EN 61260:1995 / IEC 61260:1995 Table 1 == ANSI S1.11-2004 Table 1 (verified
# identical digit-for-digit between both standards). Same layout as the 2014
# tables above, plus a class-0 column. Pass-band min is constant per class. The
# fractional-octave breakpoint mapping is the same as the 2014 edition: 1995
# Annex B equation (10) is identical to 2014 Formula (9), so _map_breakpoint is
# reused unchanged for both editions.
_PASSBAND_MAX_1995: list[tuple[float, float, float, float]] = [
    # (exponent, class 0 max, class 1 max, class 2 max)
    (0.0, 0.15, 0.3, 0.5),  # band centre (Omega of 1)
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
    r"""
    Map an octave-band breakpoint :math:`G^x` to a fractional-octave one.

    BS EN 61260-1:2014 Formula (9): the high-frequency breakpoint for
    bandwidth designator 1/b is

    .. math::

       1 + \frac{G^{1/(2b)} - 1}{G^{1/2} - 1}
       \left( \Omega_\mathrm{h}(1/1) - 1 \right)
    """
    omega_octave = _G**exponent
    scale = (_G ** (1 / (2 * fraction)) - 1) / (_G**0.5 - 1)
    return float(1 + scale * (omega_octave - 1))


def class_limits(
    fraction: float, filter_class: int, omega: np.ndarray, *, edition: str = "2014"
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Acceptance limits on relative attenuation at normalized frequencies.

    :param fraction: Bandwidth designator denominator b (1 for octave,
        3 for one-third octave, ...).
    :param filter_class: Performance class: 1 or 2 for ``edition="2014"``;
        0, 1 or 2 for ``edition="1995"``.
    :param omega: Normalized frequencies :math:`f/f_\mathrm{m}` (> 0).
    :param edition: ``"2014"`` (IEC 61260-1:2014, classes 1/2) or ``"1995"``
        (IEC 61260:1995 / ANSI S1.11-2004, classes 0/1/2).
    :return: Tuple (minimum, maximum) relative attenuation in dB per point;
        the maximum is ``+inf`` outside the pass-band.

    .. note::
        The exact band-edge point :math:`\Omega = G^{1/2}` is treated as
        pass-band.
        The 1995 edition's Table 1 prints a dedicated minimum (+2.3/+2.0/
        +1.6 dB) *at* that single frequency, which this convention relaxes to
        the pass-band minimum; the discrepancy has measure zero -- any
        continuous response violating the edge row is caught at
        :math:`\text{edge} + \epsilon`
        by the interpolated stop-band mask. The 2014 edition defines only
        the :math:`G^{1/2} - \epsilon` and :math:`G^{1/2} + \epsilon`
        rows, which the masks match
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
    extra = breakpoint_omegas[
        (breakpoint_omegas > 0) & (breakpoint_omegas <= omega_nyq)
    ]
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
            float(np.min(maximum[finite] - delta_a[finite]))
            if np.any(finite)
            else np.inf
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
    bank: OctaveFilterBank, num_points: int = 2**15, *, edition: str = "2014"
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
    breakpoint_omegas = np.array(
        [_map_breakpoint(row[0], bank.fraction) for row in rows]
    )
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
            bank, idx, classes_ordered, breakpoint_omegas, edition, num_points
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
            int(key[len(prefix) : -len(suffix)])
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

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the worst-margin band against its class-limit corridor.

        Draws the measured relative attenuation of the binding band over the
        acceptance corridor of the achieved (or, when non-compliant, the
        loosest) class; see :func:`phonometry._plot.filters.plot_filter_class`.
        Requires matplotlib (``pip install phonometry[plot]``) and returns the
        :class:`~matplotlib.axes.Axes`.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.filters import plot_filter_class

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
    bank: OctaveFilterBank, *, num_points: int = 2**15, edition: str = "2014"
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


#: The frequency-weighting transcriptions this module used to carry went to
#: :mod:`phonometry.filters.weighting_compliance` with their subject. The
#: clean-room oracles pin the tables by this path, so the reads still resolve
#: to the module that holds them now.
_WEIGHTING_TRANSCRIPTIONS = frozenset(
    {
        "_ANSI_S14_TABLE4_B",
        "_ANSI_S14_TABLE5_12",
        "_IEC61012_AU_HF",
        "_IEC61012_TABLE1",
        "_U_POLES_HZ",
        "_WEIGHTING_TABLE3",
        "_analytic_weighting_db",
    }
)


def __getattr__(name: str) -> Any:
    """Serve the transcriptions this module used to carry from their module."""
    if name in _WEIGHTING_TRANSCRIPTIONS:
        from . import weighting_compliance

        return getattr(weighting_compliance, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
