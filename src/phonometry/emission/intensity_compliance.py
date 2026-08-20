#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""IEC 61043:1993 sound-intensity instrument class verification.

A two-microphone (p-p) intensity chain is graded by its **pressure-residual
intensity index** ``delta_pI0``: feed both measurement channels the same pink
noise (or expose both probe microphones to the same pressure) and the true
intensity is exactly zero, yet the residual phase mismatch between the channels
reports a small false intensity. The difference between the indicated sound
pressure level and that residual intensity level, evaluated for an air density
of 1.2048 kg/m3, is ``delta_pI0`` (IEC 61043:1993, definition 3.11).

**Table 2** of the standard (EN 61043:1994 standard page 14) prescribes the
*minimum* ``delta_pI0`` per one-third-octave band from 50 Hz to 6.3 kHz for a
probe, a processor and a complete instrument, in class 1 and class 2, at the
nominal microphone separation of 25 mm. The table is transcribed digit for
digit below. Its Note 1 gives the separation rule: for any other microphone
separation ``x`` in millimetres, add :math:`10 \log_{10}(x/25)` dB to every figure,
so a
wider spacer both earns and demands more low-frequency margin. Note 2 restricts
the requirement to the octave-band centre frequencies for processors that only
analyse in octave bands.

Two related requirements of the same standard are exposed here as well:

* **Clause 6.1** (frequency range of processors): a class 1 processor covers at
  least 45 Hz to 7.1 kHz in one-third-octave bands (the 22 tabulated bands from
  50 Hz to 6.3 kHz). A class 2 processor covers *either* that same
  one-third-octave range *or*, alternatively, 45 Hz to 5.6 kHz in octave bands
  (the 7 octave bands from 63 Hz to 4 kHz). A verdict computed over a narrower
  set of bands attests only the bands supplied, which
  :func:`verify_intensity_class` flags as ``range_limited``. Because the octave
  range is open to class 2 alone, a class 1 verdict reached over octave bands
  only is flagged too, and a probe (tested in one-third octaves by clause 12.4)
  cannot use the alternative at all.

  The Spanish translation UNE-EN 61043:1999 states only the octave alternative
  for class 2, dropping the one-third-octave one; this module follows the
  EN/IEC text (see ``docs/ERRATA.md``).
* **Clause 8** (instrument assembled from separate components): a class 1
  instrument consists of a class 1 processor and a class 1 probe; a class 2
  instrument of any other combination of class 1 and class 2 components. See
  :func:`instrument_class_from_components`.

The index is also the instrument's phase-error floor in disguise. In an axially
propagating plane progressive wave the true phase difference across the spacer
is :math:`k d`, so a residual intensity produced by a channel phase mismatch
:math:`\phi_s` gives :math:`\delta_{pI0} = 10 \log_{10}(k d / \phi_s)` (Fahy,
*Sound Intensity*
2nd ed., equation (7.16)); :func:`phase_mismatch_from_residual_index` and
:func:`residual_index_from_phase_mismatch` convert between the two. Fahy's
worked check in section 6.8 is the anchor: :math:`\delta_{pI0} = 20` dB means
a mismatch of one hundredth of :math:`k d`, about 0.26 degrees at 1 kHz over
a 25 mm
separation.

The measured ``delta_pI0`` this module classifies is a property of the whole
probe-spacer-analyser chain and must be determined with the spacer that will be
fitted in the field; the library does not measure it. Once classified, the
ISO 9614 dynamic capability :math:`L_\mathrm{d} = \delta_{pI0} - K` follows from
:func:`phonometry.emission.intensity.dynamic_capability_index`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

__all__ = [
    "IntensityInstrumentComplianceResult",
    "instrument_class_from_components",
    "intensity_class_compliance",
    "phase_mismatch_from_residual_index",
    "residual_index_from_phase_mismatch",
    "residual_index_limits",
    "verify_intensity_class",
]

#: Nominal microphone separation the IEC 61043 Table 2 figures apply to, in
#: metres (25 mm). Note 1 of the table rescales them by ``10 lg(x/25)``.
REFERENCE_SPACING = 0.025

#: IEC 61043:1993 Table 2 (EN 61043:1994 standard page 14): minimum
#: pressure-residual intensity index requirements, in decibels, for probes,
#: processors and instruments at the 25 mm nominal microphone separation.
#: Rows are ``(nominal one-third-octave centre frequency in Hz, probe class 1,
#: probe class 2, processor class 1, processor class 2, instrument class 1,
#: instrument class 2)``.
_TABLE2: tuple[tuple[float, float, float, float, float, float, float], ...] = (
    (50.0, 13.0, 7.0, 19.0, 13.0, 12.0, 6.0),
    (63.0, 14.0, 8.0, 20.0, 14.0, 13.0, 7.0),
    (80.0, 15.0, 9.0, 21.0, 15.0, 14.0, 8.0),
    (100.0, 16.0, 10.0, 22.0, 16.0, 15.0, 9.0),
    (125.0, 17.0, 11.0, 23.0, 17.0, 16.0, 10.0),
    (160.0, 18.0, 12.0, 24.0, 18.0, 17.0, 11.0),
    (200.0, 19.0, 13.0, 25.0, 19.0, 18.0, 12.0),
    (250.0, 20.0, 14.0, 26.0, 20.0, 19.0, 13.0),
    (315.0, 20.0, 15.0, 26.0, 20.0, 19.0, 14.0),
    (400.0, 20.0, 16.0, 26.0, 20.0, 19.0, 14.5),
    (500.0, 20.0, 17.0, 26.0, 20.0, 19.0, 15.0),
    (630.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (800.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (1000.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (1250.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (1600.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (2000.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (2500.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (3150.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (4000.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (5000.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
    (6300.0, 20.0, 18.0, 26.0, 20.0, 19.0, 16.0),
)

#: Column index of each device kind's (class 1, class 2) pair in ``_TABLE2``.
_DEVICE_COLUMNS: dict[str, tuple[int, int]] = {
    "probe": (1, 2),
    "processor": (3, 4),
    "instrument": (5, 6),
}

#: Every tabulated nominal one-third-octave centre frequency, in Hz. Clause 6.1
#: requires a class 1 processor to cover 45 Hz to 7.1 kHz in one-third-octave
#: bands, which is exactly this set of 22 bands.
_THIRD_OCTAVE_BANDS: tuple[float, ...] = tuple(row[0] for row in _TABLE2)

#: Octave-band centre frequencies of clause 6.1's alternative class 2 range
#: (45 Hz to 5.6 kHz in octave bands); Table 2 Note 2 applies the requirement at
#: these centres only for octave-only processors. A probe has no analysis
#: bands of its own and clause 12.4 determines its index "at one-third-octave
#: intervals in the frequency range 50 Hz to 6.3 kHz", so this alternative is
#: not open to the ``"probe"`` column.
_OCTAVE_BANDS: tuple[float, ...] = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)

#: Largest relative deviation accepted between a supplied band centre and a
#: tabulated nominal one. The exact base-ten centres (IEC 61260-1 Annex A) sit
#: within 0,8 % of their nominal labels, so 3 % accepts both conventions while
#: still rejecting a frequency that is not a tabulated band at all.
_BAND_MATCH_TOLERANCE = 0.03


def _match_band(frequency: float) -> float:
    """The tabulated nominal band centre a supplied frequency designates.

    Accepts both the nominal labels of Table 2 (1000 Hz) and the exact
    base-ten mid-band frequencies behind them (1000 * 10**(n/10)), snapping to
    the nearest tabulated row.

    :raises ValueError: If no tabulated band lies within
        :data:`_BAND_MATCH_TOLERANCE` of ``frequency``.
    """
    if not np.isfinite(frequency) or frequency <= 0.0:
        msg = "Band centre frequencies must be finite and positive."
        raise ValueError(msg)
    nearest = min(_THIRD_OCTAVE_BANDS, key=lambda f: abs(np.log(f / frequency)))
    if abs(nearest / frequency - 1.0) > _BAND_MATCH_TOLERANCE:
        msg = (
            f"{frequency:g} Hz is not an IEC 61043 Table 2 band; the table "
            "covers the one-third-octave bands from 50 Hz to 6.3 kHz."
        )
        raise ValueError(msg)
    return nearest


def _spacing_offset(spacing: float) -> float:
    r"""Table 2 Note 1 separation term :math:`10 \log_{10}(x/25)` in dB, ``x`` in
    mm.
    """
    if not np.isfinite(spacing) or spacing <= 0.0:
        msg = "'spacing' must be a positive, finite distance in metres."
        raise ValueError(msg)
    return float(10.0 * np.log10(spacing / REFERENCE_SPACING))


def _check_device(device: str) -> tuple[int, int]:
    """Validate a device kind and return its Table 2 column indices."""
    columns = _DEVICE_COLUMNS.get(device)
    if columns is None:
        msg = (
            "'device' must be 'probe', 'processor' or 'instrument' "
            f"(IEC 61043 Table 2 columns), got {device!r}."
        )
        raise ValueError(msg)
    return columns


def residual_index_limits(
    device: str = "instrument",
    *,
    spacing: float = REFERENCE_SPACING,
    frequencies: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""IEC 61043:1993 Table 2 minimum pressure-residual intensity index.

    Returns the class 1 and class 2 minima the standard requires of a device
    kind, already rescaled to the microphone separation in use with the Note 1
    rule :math:`+10 \log_{10}(x/25)` (``x`` in millimetres, i.e.
    :math:`10 \log_{10}(\text{spacing}/0.025)`
    for a spacing in metres).

    :param device: ``"probe"``, ``"processor"`` or ``"instrument"`` (the three
        column groups of Table 2).
    :param spacing: Microphone separation in metres (default 0.025, the
        nominal separation the table is printed for).
    :param frequencies: Band centre frequencies in Hz to report the limits at,
        as nominal labels or the exact base-ten centres behind them. ``None``
        (default) returns all 22 tabulated one-third-octave bands.
    :return: Tuple ``(frequencies, class1, class2)`` of the nominal band
        centres in Hz and the two minimum ``delta_pI0`` requirements in dB.
    :raises ValueError: If ``device`` is unknown, ``spacing`` is not positive
        or a frequency is not a tabulated band.
    """
    col1, col2 = _check_device(device)
    offset = _spacing_offset(spacing)
    rows = {row[0]: row for row in _TABLE2}
    if frequencies is None:
        wanted = list(_THIRD_OCTAVE_BANDS)
    else:
        wanted = [
            _match_band(float(f))
            for f in np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        ]
    freqs = np.asarray(wanted, dtype=np.float64)
    class1 = np.asarray([rows[f][col1] + offset for f in wanted], dtype=np.float64)
    class2 = np.asarray([rows[f][col2] + offset for f in wanted], dtype=np.float64)
    return freqs, class1, class2


def instrument_class_from_components(probe_class: int, processor_class: int) -> int:
    """Class of an instrument assembled from a separate probe and processor.

    IEC 61043:1993 clause 8: when a probe and a processor are supplied
    separately, a class 1 instrument consists of a class 1 processor and a
    class 1 probe, while a class 2 instrument consists of any other combination
    of class 1 and class 2 components (class 1 processor with class 2 probe,
    class 2 processor with class 1 probe, or both class 2). The rule therefore
    reduces to the looser of the two component classes.

    :param probe_class: Class of the probe (1 or 2).
    :param processor_class: Class of the processor (1 or 2).
    :return: The class of the assembled instrument (1 or 2).
    :raises ValueError: If either class is not 1 or 2. The non-real-time
        class 2X of clause 4 is not modelled here.
    """
    if probe_class not in (1, 2) or processor_class not in (1, 2):
        msg = (
            "'probe_class' and 'processor_class' must be 1 or 2; the "
            "non-real-time class 2X of IEC 61043 clause 4 is not modelled."
        )
        raise ValueError(msg)
    return max(int(probe_class), int(processor_class))


def _band_verdicts(
    freqs: np.ndarray,
    measured: np.ndarray,
    class1: np.ndarray,
    class2: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-band class verdicts and margins against the two Table 2 minima."""
    bands: list[dict[str, Any]] = []
    for i, freq in enumerate(freqs):
        m1 = float(measured[i] - class1[i])
        m2 = float(measured[i] - class2[i])
        band_class: int | None = None
        if m1 >= 0.0:
            band_class = 1
        elif m2 >= 0.0:
            band_class = 2
        bands.append(
            {
                "freq": float(freq),
                "class": band_class,
                "residual_index_db": float(measured[i]),
                "limit_class1_db": float(class1[i]),
                "limit_class2_db": float(class2[i]),
                "margin_class1_db": m1,
                "margin_class2_db": m2,
            }
        )
    return bands


def verify_intensity_class(
    residual_index: list[float] | np.ndarray,
    frequencies: list[float] | np.ndarray,
    *,
    device: str = "instrument",
    spacing: float = REFERENCE_SPACING,
) -> dict[str, Any]:
    r"""Verify a measured ``delta_pI0`` spectrum against IEC 61043:1993 Table 2.

    Each band's measured pressure-residual intensity index is compared with the
    class 1 and class 2 minima of Table 2 for the device kind, rescaled to the
    microphone separation in use (Note 1, :math:`+10 \log_{10}(x/25)`). A band meets
    a class when its measured index is greater than or equal to that class's
    minimum, so the margin is ``measured - minimum`` and a band exactly on the
    limit passes. The overall class is the loosest per-band class, or ``None``
    when any band meets neither class.

    Clause 6.1 fixes the frequency range the class attests: 45 Hz to 7.1 kHz in
    one-third-octave bands (the 22 tabulated bands, 50 Hz to 6.3 kHz), or, for
    an octave-band processor, 45 Hz to 5.6 kHz in octave bands (63 Hz to
    4 kHz). ``range_limited`` is ``True`` when the supplied bands cover neither
    of those sets, in which case the returned class attests the bands actually
    verified and not the standard's full frequency range. The one-third-octave
    range is required for class 1 and available to class 2, while the octave
    range is offered as a class 2 alternative only, so a verdict computed over
    the 7 octave bands attests a class 2 result but is still ``range_limited``
    when it reaches class 1. The octave
    alternative is not open to a ``"probe"`` at all: a probe has no analysis
    bands of its own and clause 12.4 determines its index at one-third-octave
    intervals across the whole 50 Hz to 6.3 kHz range.

    :param residual_index: Measured ``delta_pI0`` per band, in decibels.
    :param frequencies: Band centre frequencies in Hz, one per entry of
        ``residual_index``, as nominal Table 2 labels or the exact base-ten
        centres behind them.
    :param device: ``"probe"``, ``"processor"`` or ``"instrument"``.
    :param spacing: Microphone separation in metres (default 0.025).
    :return: Dict with ``overall_class`` (1, 2 or ``None``), ``range_limited``,
        ``bands`` (a list of ``{"freq", "class", "residual_index_db",
        "limit_class1_db", "limit_class2_db", "margin_class1_db",
        "margin_class2_db"}``), ``device``, ``spacing`` and
        ``spacing_offset_db`` (the Note 1 term applied to the table).
    :raises ValueError: If the inputs disagree in length, a frequency is not a
        tabulated band, a band is repeated, or ``device``/``spacing`` are
        invalid.
    """
    measured = np.atleast_1d(np.asarray(residual_index, dtype=np.float64))
    supplied = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if measured.ndim != 1 or supplied.ndim != 1:
        msg = "verify_intensity_class expects 1D per-band arrays."
        raise ValueError(msg)
    if measured.size != supplied.size:
        msg = (
            f"'residual_index' and 'frequencies' must have the same length, "
            f"got {measured.size} and {supplied.size}."
        )
        raise ValueError(msg)
    if measured.size == 0:
        msg = "At least one measurement band is required."
        raise ValueError(msg)
    if not np.all(np.isfinite(measured)):
        msg = "'residual_index' must be finite."
        raise ValueError(msg)

    freqs = np.asarray([_match_band(float(f)) for f in supplied], dtype=np.float64)
    if np.unique(freqs).size != freqs.size:
        msg = (
            "'frequencies' repeats an IEC 61043 Table 2 band; supply one "
            "measured value per band."
        )
        raise ValueError(msg)

    _, class1, class2 = residual_index_limits(
        device, spacing=spacing, frequencies=freqs
    )
    bands = _band_verdicts(freqs, measured, class1, class2)

    classes = [band["class"] for band in bands]
    overall: int | None = None if None in classes else max(classes)

    covered = set(freqs.tolist())
    # Clause 6.1 gives the one-third-octave range to class 1 and offers the
    # octave range only to class 2, so the octave set can attest a class 2
    # verdict but never a class 1 one. A probe has no analysis bands of its
    # own and clause 12.4 tests it in one-third octaves, so the alternative is
    # not open to it at all.
    full_range = covered.issuperset(_THIRD_OCTAVE_BANDS) or (
        device != "probe" and overall == 2 and covered.issuperset(_OCTAVE_BANDS)  # noqa: PLR2004
    )
    range_limited = not full_range

    return {
        "overall_class": overall,
        "range_limited": range_limited,
        "bands": bands,
        "device": device,
        "spacing": float(spacing),
        "spacing_offset_db": _spacing_offset(spacing),
    }


@dataclass(frozen=True)
class IntensityInstrumentComplianceResult:
    r"""IEC 61043:1993 class verdict of a p-p sound-intensity chain.

    Wraps the outcome of :func:`verify_intensity_class` together with the
    measured spectrum and the two Table 2 masks it was judged against, so the
    result can redraw itself and render an accredited fiche.

    :ivar overall_class: The loosest class every band meets (1 or 2), or
        ``None`` when at least one band meets neither.
    :ivar bands: The per-band verdict dictionaries of
        :func:`verify_intensity_class`, as an immutable tuple.
    :ivar frequency: Nominal band centre frequencies, in Hz.
    :ivar residual_index: Measured ``delta_pI0`` per band, in dB.
    :ivar limit_class1: Class 1 minimum ``delta_pI0`` per band, in dB, already
        rescaled to ``spacing``.
    :ivar limit_class2: Class 2 minimum per band, in dB, likewise rescaled.
    :ivar device: ``"probe"``, ``"processor"`` or ``"instrument"``.
    :ivar spacing: Microphone separation the verdict applies to, in metres.
    :ivar spacing_offset_db: The Table 2 Note 1 term :math:`10 \log_{10}(x/25)`
        added to the printed 25 mm figures, in dB.
    :ivar range_limited: ``True`` when the verified bands cover neither the 22
        one-third-octave bands nor the 7 octave bands of clause 6.1, so the
        stated class attests only the bands supplied.
    """

    overall_class: int | None
    bands: tuple[dict[str, Any], ...]
    frequency: np.ndarray
    residual_index: np.ndarray
    limit_class1: np.ndarray
    limit_class2: np.ndarray
    device: str
    spacing: float
    spacing_offset_db: float
    range_limited: bool = False

    def reference_class(self) -> int:
        """The class whose mask the fiche and the plot read margins against.

        The achieved class when the chain complies, else class 2 (the loosest
        class of the standard, the one it comes closest to meeting).
        """
        return 2 if self.overall_class is None else self.overall_class

    def binding_margin(self, device_class: int | None = None) -> float:
        """Smallest per-band margin to a class, in dB (the binding margin).

        :param device_class: 1 or 2; ``None`` (default) uses
            :meth:`reference_class`.
        :raises ValueError: If ``device_class`` is not 1 or 2.
        """
        cls = self.reference_class() if device_class is None else int(device_class)
        if cls not in (1, 2):
            msg = "'device_class' must be 1 or 2."
            raise ValueError(msg)
        return min(float(band[f"margin_class{cls}_db"]) for band in self.bands)

    def failing_bands(self, device_class: int | None = None) -> list[float]:
        """Nominal centre frequencies of the bands that miss a class, in Hz.

        :param device_class: 1 or 2; ``None`` (default) uses
            :meth:`reference_class`.
        """
        cls = self.reference_class() if device_class is None else int(device_class)
        if cls not in (1, 2):
            msg = "'device_class' must be 1 or 2."
            raise ValueError(msg)
        key = f"margin_class{cls}_db"
        return [float(b["freq"]) for b in self.bands if float(b[key]) < 0.0]

    def phase_mismatch(self, c: float = 343.0) -> np.ndarray:
        """Equivalent channel phase mismatch per band, in degrees.

        Converts the measured ``delta_pI0`` spectrum with
        :func:`phase_mismatch_from_residual_index` at the result's own
        microphone separation, so the verdict can be read as the phase-matching
        the chain achieves.

        :param c: Speed of sound in m/s (default 343.0).
        """
        return phase_mismatch_from_residual_index(
            self.residual_index, self.frequency, self.spacing, c=c
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the measured ``delta_pI0`` over the Table 2 class masks.

        See :func:`phonometry._plot.emission.plot_intensity_class`. Requires
        matplotlib (``pip install phonometry[plot]``) and returns the
        :class:`~matplotlib.axes.Axes`.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_intensity_class

        check_language(language)
        return plot_intensity_class(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an IEC 61043 residual-index verification fiche to a PDF.

        Writes a one-page accredited report: the standard-basis line, an
        optional metadata header block, the per-band table of measured index,
        Table 2 requirement and margin beside the mask-overlay plot, the boxed
        class result, an optional verdict row against a supplied
        ``required_class`` and the footer disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a bare fiche (body, result and disclaimer only). A
            supplied ``required_class`` drives the verdict row.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform signature; it has no effect on
            the single-layout intensity-compliance fiche.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            raise ValueError(msg)
        from .._report.iec61043 import render_iec61043_report

        return render_iec61043_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def intensity_class_compliance(
    residual_index: list[float] | np.ndarray,
    frequencies: list[float] | np.ndarray,
    *,
    device: str = "instrument",
    spacing: float = REFERENCE_SPACING,
) -> IntensityInstrumentComplianceResult:
    """Verify a ``delta_pI0`` spectrum and package the verdict as a result.

    Runs :func:`verify_intensity_class` and stores the outcome together with
    the measured spectrum and the two rescaled Table 2 masks, so the returned
    object exposes ``.plot()`` and an accredited ``.report()`` fiche.

    :param residual_index: Measured ``delta_pI0`` per band, in decibels.
    :param frequencies: Band centre frequencies in Hz, one per entry.
    :param device: ``"probe"``, ``"processor"`` or ``"instrument"``.
    :param spacing: Microphone separation in metres (default 0.025).
    :return: An :class:`IntensityInstrumentComplianceResult`.
    """
    verdict = verify_intensity_class(
        residual_index, frequencies, device=device, spacing=spacing
    )
    bands = verdict["bands"]
    return IntensityInstrumentComplianceResult(
        overall_class=verdict["overall_class"],
        bands=tuple(bands),
        frequency=np.asarray([b["freq"] for b in bands], dtype=np.float64),
        residual_index=np.asarray(
            [b["residual_index_db"] for b in bands], dtype=np.float64
        ),
        limit_class1=np.asarray(
            [b["limit_class1_db"] for b in bands], dtype=np.float64
        ),
        limit_class2=np.asarray(
            [b["limit_class2_db"] for b in bands], dtype=np.float64
        ),
        device=str(verdict["device"]),
        spacing=float(verdict["spacing"]),
        spacing_offset_db=float(verdict["spacing_offset_db"]),
        range_limited=bool(verdict["range_limited"]),
    )


# ---------------------------------------------------------------------------
# delta_pI0 <-> channel phase mismatch
# ---------------------------------------------------------------------------


def _plane_wave_phase_deg(
    frequency: np.ndarray, spacing: float, c: float
) -> np.ndarray:
    r"""Plane-wave phase difference :math:`k d` across the spacer, in
    degrees.
    """
    if not np.isfinite(spacing) or spacing <= 0.0:
        msg = "'spacing' must be a positive, finite distance in metres."
        raise ValueError(msg)
    if not np.isfinite(c) or c <= 0.0:
        msg = "'c' must be a positive, finite speed of sound."
        raise ValueError(msg)
    freq = np.asarray(frequency, dtype=np.float64)
    if np.any(~np.isfinite(freq)) or np.any(freq <= 0.0):
        msg = "'frequency' must be finite and positive."
        raise ValueError(msg)
    return np.asarray(360.0 * freq * spacing / c, dtype=np.float64)


def phase_mismatch_from_residual_index(
    residual_index: float | list[float] | np.ndarray,
    frequency: float | list[float] | np.ndarray,
    spacing: float,
    c: float = 343.0,
) -> np.ndarray:
    r"""Channel phase mismatch equivalent to a pressure-residual intensity index.

    In an axially propagating plane progressive wave the true phase difference
    between the two sensing points is :math:`k d = 2\pi f d / c`, and a
    residual
    intensity produced by a channel phase mismatch :math:`\phi_s` satisfies
    :math:`\delta_{pI0} = 10 \log_{10}(k d / \phi_s)` (Fahy, *Sound Intensity*
    2nd ed., equations (7.4) and (7.16)), so:

    .. math::

       \phi_s = k d \cdot 10^{-\delta_{pI0} / 10}

    The ratio is dimensionless, so :math:`\phi_s` is returned in the same
    angular unit :math:`k d` is expressed in; degrees are used here.

    :param residual_index: ``delta_pI0`` in decibels (scalar or array).
    :param frequency: Frequency in Hz (scalar or array, broadcast against
        ``residual_index``).
    :param spacing: Microphone separation in metres.
    :param c: Speed of sound in m/s (default 343.0).
    :return: The equivalent phase mismatch in degrees, as a
        :class:`numpy.ndarray` (0-d for scalar inputs).
    :raises ValueError: If ``spacing``, ``c`` or ``frequency`` are not positive
        and finite, or if ``residual_index`` is not finite.
    """
    index = np.asarray(residual_index, dtype=np.float64)
    if np.any(~np.isfinite(index)):
        msg = "'residual_index' must be finite."
        raise ValueError(msg)
    kd_deg = _plane_wave_phase_deg(np.asarray(frequency, dtype=np.float64), spacing, c)
    return np.asarray(kd_deg * 10.0 ** (-index / 10.0), dtype=np.float64)


def residual_index_from_phase_mismatch(
    phase_mismatch: float | list[float] | np.ndarray,
    frequency: float | list[float] | np.ndarray,
    spacing: float,
    c: float = 343.0,
) -> np.ndarray:
    r"""Pressure-residual intensity index of a given channel phase mismatch.

    The inverse of :func:`phase_mismatch_from_residual_index`:

    .. math::

       \delta_{pI0} = 10 \log_{10}\frac{k d}{\phi_s}, \qquad
       k d = \frac{2 \pi f d}{c}

    with :math:`k d` and :math:`\phi_s` both in degrees (the ratio is
    dimensionless).
    Because :math:`k d` grows with frequency while a mismatch that is constant
    in
    degrees does not, the index rises by 10 dB per decade of frequency: this is
    why a fixed phase-matching quality yields the falling low-frequency
    ``delta_pI0`` that IEC 61043 Table 2 grades band by band, and why a wider
    spacer buys :math:`10 \log_{10}(x/25)` dB of index exactly as Note 1 of that
    table requires.

    :param phase_mismatch: :math:`\phi_s` in degrees (scalar or array, > 0).
    :param frequency: Frequency in Hz (scalar or array, broadcast against
        ``phase_mismatch``).
    :param spacing: Microphone separation in metres.
    :param c: Speed of sound in m/s (default 343.0).
    :return: ``delta_pI0`` in decibels, as a :class:`numpy.ndarray` (0-d for
        scalar inputs).
    :raises ValueError: If ``phase_mismatch`` is not positive and finite, or
        if ``spacing``, ``c`` or ``frequency`` are not positive and finite.
    """
    phi = np.asarray(phase_mismatch, dtype=np.float64)
    if np.any(~np.isfinite(phi)) or np.any(phi <= 0.0):
        msg = "'phase_mismatch' must be finite and positive, in degrees."
        raise ValueError(msg)
    kd_deg = _plane_wave_phase_deg(np.asarray(frequency, dtype=np.float64), spacing, c)
    return np.asarray(10.0 * np.log10(kd_deg / phi), dtype=np.float64)
