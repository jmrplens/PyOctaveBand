#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Sound insulation measured with sound intensity (ISO 15186).

This is the sound-**intensity** counterpart of the sound-pressure methods in
:mod:`phonometry.lab_insulation` (ISO 10140) and :mod:`phonometry.insulation`
(ISO 16283). Instead of an equivalent absorption area in the receiving room,
the transmitted sound power is measured directly by scanning an intensity
probe over a measurement surface enclosing the specimen. The main use is when
the traditional pressure method fails because of high flanking transmission
(ISO 15186-1:2000, Clause 1): the intensity method only captures the power
radiated by the element itself.

**Intensity sound reduction index (ISO 15186-1:2000, Clause 3.8, Formula
(7)).** From the average source-room sound pressure level ``Lp1`` and the
average normal sound intensity level ``LIn`` over the measurement surface,

``RI = Lp1 - 6 - [LIn + 10 lg(Sm / S)]`` dB

with the measurement-surface area ``Sm`` and the specimen area ``S``. The
constant ``6`` dB is the diffuse-field relationship between the sound pressure
level and the sound intensity level incident on the specimen. The same formula
yields the *apparent* index ``R'I`` in the field (ISO 15186-2), the only
difference being the measurement condition (flanking is not suppressed), not
the arithmetic.

**Modified intensity sound reduction index (Clause 3.10, Formula (9)).**
``RI,M = RI + Kc`` corrects ``RI`` so that it reproduces the ISO 140-3 (now
ISO 10140-2) pressure result, which slightly overestimates ``R`` because the
power radiated into the receiving room is underestimated. The adaptation term
``Kc`` (Annex B) is ``10 lg(1 + Sb2 lambda / (8 V2))`` (Formula (B.1)) for a
well-defined receiving room of boundary area ``Sb2`` and volume ``V2``, or the
room-independent approximation ``10 lg(1 + 61,4 / f)`` (Formula (B.2)); both
use the speed of sound ``c = 340 m/s`` so that (B.1) with the reference room
``Sb2 = 117 m²``, ``V2 = 81 m³`` reduces to (B.2).

**Intensity element normalized level difference (Clause 3.9, Formula (8)).**
For small building elements, ``DI,n,e = Lp1 - 6 - (LIn + 10 lg(Sm / A0)) +
10 lg N`` dB with the reference absorption area ``A0 = 10 m²`` and the number
``N`` of element units in the measurement surface. The printed Formula (8)
subtracts its ``10 lg(N)`` term instead of adding it, which is physically
inconsistent with ISO 10140-2:2010 Formula (6) and ISO 15186-2:2010
Formula (12); the corrected per-unit form is implemented (see
``docs/ERRATA.md``).

**Surface pressure-intensity indicator (Clause 3.6 / 6.4.2, Formula (10)).**
``FpI = Lp - LIn`` qualifies the measurement surface: it must not exceed
10 dB for a sound-reflecting specimen (6 dB when the receiving side is
sound absorbing), and the probe's pressure-residual intensity index must
exceed ``FpI + 10`` dB (Clause 4.1) for the dynamic capability to be adequate.

**Frequency range (Clause 6.6).** Quantities are measured over the mandatory
one-third-octave range 100 Hz to 5000 Hz (18 bands), optionally extended down
to 50 Hz. The single-number weighted rating uses the ISO 717-1 core range, so
the automatic rating (``RI,w``, ``RI,M,w``, ``DI,n,e,w``) is formed via the
verified :func:`phonometry.weighted_rating` engine only when exactly 16
one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are supplied.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .insulation import (
    WeightedRatingResult,
    _as_band_levels,
    weighted_rating,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

#: Reference equivalent absorption area for the element-normalized level
#: difference (ISO 15186-1:2000, Clause 3.9): 10 m².
_A0 = 10.0

#: Diffuse-field level difference between the sound pressure level and the
#: incident sound intensity level (ISO 15186-1:2000, Formulas (7)-(8)): 6 dB.
_DIFFUSE_FIELD = 6.0

#: Speed of sound used in the adaptation term ``Kc`` (Annex B): 340 m/s, the
#: value for which Formula (B.1) with the reference room reduces to (B.2).
_SPEED_OF_SOUND = 340.0

#: Numerator of the room-independent adaptation term ``Kc`` (Formula (B.2)).
_KC_APPROX_COEFF = 61.4


def _validate_intensity_report(
    *,
    engine: str,
    language: str,
    rating: WeightedRatingResult | None,
    curve: np.ndarray,
    label: str,
) -> WeightedRatingResult:
    """Validate a shared ISO 15186-1 intensity-report request, or raise.

    Both intensity fiches accept only the ``"reportlab"`` engine and a
    translated language, and both need the ISO 717-1 single-number rating that
    is formed for exactly the 16 one-third-octave or 5 octave bands the fiche
    is defined over. A manually built result whose hand-crafted rating carries
    another band count is rejected too (``curve`` is the reported per-band
    array, ``label`` names the quantity in the messages). Returns the validated
    (non-``None``) rating so the caller can hand it straight to the renderer.

    :raises ValueError: If ``engine`` is unknown, ``language`` is unsupported,
        the single-number ``rating`` is absent, or the band count is neither 16
        nor 5.
    """
    from .._i18n import check_language

    check_language(language)
    if engine != "reportlab":
        raise ValueError(
            f"Unknown report engine {engine!r}; only 'reportlab' is supported."
        )
    if rating is None:
        raise ValueError(
            f"The {label} report needs the ISO 717-1 single-number rating; it "
            "is formed only for exactly 16 one-third-octave (100 Hz to "
            "3150 Hz) or 5 octave (125 Hz to 2000 Hz) bands. Build the result "
            "with that band count."
        )
    n_bands = int(np.asarray(curve).size)
    if n_bands not in (16, 5):
        raise ValueError(
            f"The {label} report supports only 16 one-third-octave (100 Hz to "
            "3150 Hz) or 5 octave (125 Hz to 2000 Hz) bands; the result "
            f"carries {n_bands}."
        )
    return rating


@dataclass(frozen=True)
class IntensityReductionResult:
    """Per-band intensity sound reduction index (ISO 15186-1:2000).

    :ivar r_i: Intensity sound reduction index ``RI = Lp1 - 6 -
        [LIn + 10 lg(Sm/S)]`` per band, in dB (Clause 3.8, Formula (7)). In
        the field (ISO 15186-2) this is the apparent index ``R'I``.
    :ivar r_i_modified: Modified index ``RI,M = RI + Kc`` per band, in dB
        (Clause 3.10, Formula (9)), or ``None`` when no adaptation term was
        supplied.
    :ivar rating: Single-number weighted rating ``RI,w`` with ``C`` / ``Ctr``
        (ISO 717-1), or ``None`` when the band count is neither 16
        (one-third octave) nor 5 (octave).
    :ivar rating_modified: Weighted rating ``RI,M,w`` of the modified index,
        or ``None`` when unavailable.
    :ivar area: Test-object area ``S``, in m², or ``None`` on a manually
        built result. Carried so the report can state it (Clause 8 g).
    :ivar measurement_area: Measurement-surface area ``Sm``, in m², or
        ``None`` on a manually built result (Clause 8 g).
    """

    r_i: np.ndarray
    r_i_modified: np.ndarray | None
    rating: WeightedRatingResult | None
    rating_modified: WeightedRatingResult | None
    area: float | None = None
    measurement_area: float | None = None

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot ``RI`` against the shifted ISO 717-1 reference curve.

        Delegates to the weighted-rating plot (measured ``RI`` versus the
        shifted reference, unfavourable deviations shaded). Requires the
        automatic rating to be available (16 or 5 bands) and matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        if self.rating is None:
            raise ValueError(
                "No single-number rating is available to plot (need 16 "
                "one-third-octave or 5 octave bands)."
            )
        return self.rating.plot(ax=ax, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
        fpi: Sequence[float] | np.ndarray | None = None,
        residual_index: Sequence[float] | np.ndarray | None = None,
    ) -> str:
        """Render an ISO 15186-1 intensity sound-insulation report to a PDF.

        Writes the one-page laboratory test report of ISO 15186-1:2000
        Clause 8 for sound insulation measured with sound intensity: the
        standard-basis line, an optional metadata header block, the band table
        (16 one-third-octave or 5 octave bands) beside the
        measured-versus-shifted-reference curve, the boxed rating ``RI,w
        (C; Ctr)`` (the intensity sound reduction index ``RI`` rated per
        ISO 717-1), the intensity-method statement with the test-object and
        measurement-surface areas ``S`` / ``Sm`` when the result carries them
        (Clause 8 g), an optional verdict row and a footer with the identity
        block and disclaimer. The report requires the single-number ``rating``
        to be present on the result; it is formed automatically only for
        exactly 16 one-third-octave (100 Hz to 3150 Hz) or 5 octave (125 Hz
        to 2000 Hz) bands, and a result carrying no rating (any other band
        count) is rejected.

        The applicable :class:`~phonometry.ReportMetadata` fields describe the
        intensity measurement: ``specimen`` (the tested element and its
        mounting, sealing and mass per unit area, Clause 8 e), ``client``,
        ``manufacturer``, ``test_room`` (the laboratory / facility),
        ``laboratory``, ``operator``, ``report_id`` and ``test_date``, plus
        the room/climate fields shared with the other insulation fiches. The
        measurement-surface shape, the measurement distance and the
        scanning-versus-discrete-point acquisition method (Clause 8 j-l) are
        not dedicated metadata fields; record them in ``notes`` (free text)
        and name the measurement standard in ``measurement_standard``
        (``"ISO 15186-1"``).

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating, statement and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` and the Kc-modified index ``RI,M`` is
            available, the table annexes ``RI,M`` beside the reported ``RI``.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :param fpi: Optional per-band surface pressure-intensity indicator
            ``FpI`` (Clause 8 i requires it as a function of frequency);
            annexed as a table column when supplied.
        :param residual_index: Optional per-band pressure-residual intensity
            index ``δpI0`` of the probe and analyser (Clause 8 i); annexed as
            a table column when supplied.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown, ``language`` is not one
            of the supported values, the result carries no single-number
            rating (its band count is neither 16 one-third-octave nor 5
            octave, so the ISO 717-1 rating the fiche needs was not formed),
            or ``fpi`` / ``residual_index`` do not match the band count.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded rating figure (``pip install phonometry[plot]``).
        """
        rating = _validate_intensity_report(
            engine=engine,
            language=language,
            rating=self.rating,
            curve=self.r_i,
            label="intensity",
        )

        from .._report.iso15186 import render_iso15186_report

        return render_iso15186_report(
            self, rating, path, metadata=metadata,
            verbose=verbose, language=language,
            fpi=fpi, residual_index=residual_index,
        )


@dataclass(frozen=True)
class IntensityElementNormalizedResult:
    """Per-band intensity element normalized level difference (ISO 15186-1).

    :ivar d_i_n_e: Intensity element normalized level difference
        ``DI,n,e = Lp1 - 6 - (LIn + 10 lg(Sm/A0)) + 10 lg N`` per band, in dB
        (Clause 3.9, Formula (8) with the corrected sign of its ``10 lg N``
        term; see ``docs/ERRATA.md``).
    :ivar rating: Single-number weighted rating ``DI,n,e,w`` with ``C`` /
        ``Ctr`` (ISO 717-1), or ``None`` when the band count is neither 16
        (one-third octave) nor 5 (octave).
    :ivar measurement_area: Measurement-surface area ``Sm``, in m², or
        ``None`` on a manually built result (Clause 8 g).
    :ivar n: Number ``N`` of element units within the measurement surface.
    """

    d_i_n_e: np.ndarray
    rating: WeightedRatingResult | None
    measurement_area: float | None = None
    n: int = 1

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot ``DI,n,e`` against the shifted ISO 717-1 reference curve.

        Delegates to the weighted-rating plot. Requires the automatic rating
        to be available (16 or 5 bands) and matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        if self.rating is None:
            raise ValueError(
                "No single-number rating is available to plot (need 16 "
                "one-third-octave or 5 octave bands)."
            )
        return self.rating.plot(ax=ax, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
        fpi: Sequence[float] | np.ndarray | None = None,
        residual_index: Sequence[float] | np.ndarray | None = None,
    ) -> str:
        """Render an ISO 15186-1 element-normalized insulation report to a PDF.

        Writes the one-page laboratory test report of ISO 15186-1:2000
        Clause 8 for the element-normalized level difference ``DI,n,e`` of a
        small building element measured with sound intensity: the
        standard-basis line, an optional metadata header block, the band table
        (16 one-third-octave or 5 octave bands) beside the
        measured-versus-shifted-reference curve, the boxed rating ``DI,n,e,w
        (C; Ctr)`` (the element-normalized level difference ``DI,n,e`` rated
        per ISO 717-1), the intensity-method statement with the
        measurement-surface area ``Sm`` and unit count ``N`` when the result
        carries them (Clause 8 g), an optional verdict row and a footer with
        the identity block and disclaimer. The report requires the
        single-number ``rating`` to be present on the result; it is formed
        automatically only for exactly 16 one-third-octave (100 Hz to 3150 Hz)
        or 5 octave (125 Hz to 2000 Hz) bands, and a result carrying no rating
        (any other band count) is rejected.

        The applicable :class:`~phonometry.ReportMetadata` fields describe the
        intensity measurement: ``specimen`` (the tested element and its
        mounting and sealing, Clause 8 e), ``area`` (the element area),
        ``client``, ``manufacturer``, ``test_room`` (the laboratory /
        facility), ``laboratory``, ``operator``, ``report_id`` and
        ``test_date``, plus the room/climate fields shared with the other
        insulation fiches. The measurement-surface shape, the measurement
        distance and the scanning-versus-discrete-point acquisition method
        (Clause 8 j-l) are not dedicated metadata fields; record them in
        ``notes`` (free text) and name the measurement standard in
        ``measurement_standard`` (``"ISO 15186-1"``). A ``requirement`` adds a
        PASS/FAIL verdict; the element insulation passes at or above the target.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating, statement and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the left table shows the ISO 717
            evaluation per band (the ``DI,n,e`` value, the shifted reference
            and the unfavourable deviation) instead of the two-column form.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :param fpi: Optional per-band surface pressure-intensity indicator
            ``FpI`` (Clause 8 i requires it as a function of frequency);
            annexed as a table column when supplied.
        :param residual_index: Optional per-band pressure-residual intensity
            index ``δpI0`` of the probe and analyser (Clause 8 i); annexed as
            a table column when supplied.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown, ``language`` is not one
            of the supported values, the result carries no single-number
            rating (its band count is neither 16 one-third-octave nor 5
            octave, so the ISO 717-1 rating the fiche needs was not formed),
            or ``fpi`` / ``residual_index`` do not match the band count.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded rating figure (``pip install phonometry[plot]``).
        """
        rating = _validate_intensity_report(
            engine=engine,
            language=language,
            rating=self.rating,
            curve=self.d_i_n_e,
            label="element-normalized",
        )

        from .._report.iso15186 import render_iso15186_element_report

        return render_iso15186_element_report(
            self, rating, path, metadata=metadata,
            verbose=verbose, language=language,
            fpi=fpi, residual_index=residual_index,
        )


def _positive_area(value: float, name: str) -> float:
    """Return ``value`` as a positive, finite area, or raise."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"'{name}' must be positive.")
    return v


def adaptation_term_kc(
    freq: Sequence[float] | np.ndarray,
    *,
    boundary_area: float | None = None,
    volume: float | None = None,
) -> np.ndarray:
    """Adaptation term ``Kc`` per ISO 15186-1:2000, Annex B.

    Returns, per one-third-octave midband frequency, the term ``Kc`` that
    turns the intensity sound reduction index ``RI`` into the modified index
    ``RI,M = RI + Kc`` (Clause 3.10). Two forms are available:

    - **Well-defined receiving room (Formula (B.1)):** when both
      ``boundary_area`` (``Sb2``) and ``volume`` (``V2``) are supplied,
      ``Kc = 10 lg(1 + Sb2 lambda / (8 V2))`` with the midband wavelength
      ``lambda = c / f`` and ``c = 340 m/s``.
    - **Room-independent approximation (Formula (B.2)):** when neither is
      supplied, ``Kc = 10 lg(1 + 61,4 / f)``, the exact reduction of (B.1)
      for the reference room ``Sb2 = 117 m²``, ``V2 = 81 m³``.

    :param freq: One-third-octave midband frequencies, in Hz.
    :param boundary_area: Total boundary-surface area ``Sb2`` of the
        receiving room, in m². Supply together with ``volume`` for (B.1).
    :param volume: Receiving-room volume ``V2``, in m³.
    :return: The adaptation term ``Kc`` per band, in dB.
    :raises ValueError: If ``freq`` is not positive/finite, if only one of
        ``boundary_area`` / ``volume`` is supplied, or if either is not
        positive.
    """
    f = np.asarray(freq, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError("'freq' must be one-dimensional.")
    if not np.all(np.isfinite(f)) or np.any(f <= 0.0):
        raise ValueError("'freq' must contain positive, finite values.")

    if boundary_area is None and volume is None:
        ratio = _KC_APPROX_COEFF / f
    elif boundary_area is not None and volume is not None:
        sb2 = _positive_area(boundary_area, "boundary_area")
        v2 = _positive_area(volume, "volume")
        wavelength = _SPEED_OF_SOUND / f
        ratio = sb2 * wavelength / (8.0 * v2)
    else:
        raise ValueError(
            "Supply both 'boundary_area' and 'volume' for Formula (B.1), or "
            "neither for the Formula (B.2) approximation."
        )
    return 10.0 * np.log10(1.0 + ratio)


def surface_pressure_intensity_indicator(
    lp: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Surface pressure-intensity indicator ``FpI`` (ISO 15186-1, Formula (10)).

    Returns ``FpI = Lp - LIn`` per band from the surface- and time-averaged
    sound pressure level ``Lp`` and normal sound intensity level ``LIn`` on
    the measurement surface (Clause 3.6 / 6.4.2). The measurement surface is
    adequately qualified when ``FpI`` does not exceed 10 dB for a
    sound-reflecting specimen, or 6 dB when the receiving side is sound
    absorbing (Clause 6.4.2 flags ``FpI > 10 dB`` / ``FpI > 6 dB`` as not
    satisfactory); in addition the probe's pressure-residual intensity index
    must exceed ``FpI + 10`` dB (Clause 4.1).

    :param lp: Surface-averaged sound pressure levels, in dB.
    :param l_in: Normal sound intensity levels on the surface, in dB.
    :return: The indicator ``FpI`` per band, in dB.
    :raises ValueError: If the shapes differ or contain non-finite values.
    """
    p = np.asarray(lp, dtype=np.float64)
    i = np.asarray(l_in, dtype=np.float64)
    if p.shape != i.shape:
        raise ValueError("'lp' and 'l_in' must share their shape.")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(i))):
        raise ValueError("Levels must contain only finite values.")
    return p - i


def combine_subareas(
    l_in: Sequence[Sequence[float]] | np.ndarray,
    measurement_area: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Combine per-subarea intensity levels (ISO 15186-1, Formulas (11)-(12)).

    When the measurement surface is divided into subareas ``Smi`` each scanned
    individually, the normal sound intensity level over the whole surface is
    the area-weighted energy average

    ``LIn = 10 lg[ (1/Sm) sum_i Smi 10^(0,1 LIni) ]`` dB

    with the total measured area ``Sm = sum_i |Smi|`` (Formula (12)).

    **Negative-direction subareas (Clause 6.4.6).** When the sound intensity
    of a subarea has a negative direction (net energy flowing back towards
    the specimen), the standard requires a minus sign before that ``Smi`` in
    Formula (11). Express this by passing the subarea's area as a *negative*
    number: its energy is subtracted in the numerator while ``Sm`` keeps the
    unsigned area sum.

    :param l_in: Per-subarea intensity levels as a ``(subareas, bands)``
        array (one row per subarea), in dB (magnitude of the intensity).
    :param measurement_area: Subarea areas ``Smi``, in m² (one per row).
        Negative values mark reverse-flow subareas per Clause 6.4.6; zero is
        invalid.
    :return: A tuple ``(LIn, Sm)`` with the combined level per band, in dB,
        and the total measured area ``Sm = sum |Smi|``, in m².
    :raises ValueError: If the shapes are inconsistent or values non-finite,
        if any subarea area is zero, or if the signed energy sum of
        Formula (11) is not positive in some band (the reverse flows cancel
        or exceed the forward flow, so no level exists).
    """
    levels = np.asarray(l_in, dtype=np.float64)
    if levels.ndim != 2:
        raise ValueError(
            "'l_in' must be a two-dimensional (subareas, bands) array."
        )
    areas = np.asarray(measurement_area, dtype=np.float64)
    if areas.ndim != 1 or areas.size != levels.shape[0]:
        raise ValueError(
            "'measurement_area' must give one area per subarea (row of "
            "'l_in')."
        )
    if not np.all(np.isfinite(levels)):
        raise ValueError("'l_in' must contain only finite values.")
    if not np.all(np.isfinite(areas)) or not np.all(np.abs(areas) > 0.0):
        raise ValueError(
            "'measurement_area' must contain non-zero, finite areas (negative "
            "marks a reverse-flow subarea, Clause 6.4.6)."
        )

    sm = float(np.sum(np.abs(areas)))
    energy = np.sum(areas[:, None] * 10.0 ** (0.1 * levels), axis=0)
    if np.any(energy <= 0.0):
        raise ValueError(
            "The signed subarea energy sum of Formula (11) is not positive in "
            "at least one band: the negative-direction subareas cancel or "
            "exceed the forward flow, so no combined intensity level exists."
        )
    l_in_total = 10.0 * np.log10(energy / sm)
    return l_in_total, sm


def intensity_sound_reduction(
    lp1: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
    *,
    measurement_area: float,
    area: float,
    kc: Sequence[float] | np.ndarray | None = None,
) -> IntensityReductionResult:
    """
    Intensity sound reduction index per ISO 15186-1:2000 (Formula (7)).

    Computes, per frequency band, the intensity sound reduction index

    ``RI = Lp1 - 6 - [LIn + 10 lg(Sm / S)]`` dB

    from the average source-room sound pressure level ``Lp1`` and the average
    normal sound intensity level ``LIn`` over the measurement surface of area
    ``Sm`` (``measurement_area``), for a specimen of area ``S`` (``area``).
    The same formula gives the apparent index ``R'I`` in the field
    (ISO 15186-2). When an adaptation term ``kc`` is supplied (see
    :func:`adaptation_term_kc`), the modified index ``RI,M = RI + Kc``
    (Formula (9)) is also formed. Weighted ratings ``RI,w`` (and ``RI,M,w``)
    are computed via :func:`phonometry.weighted_rating` (ISO 717-1) when
    exactly 16 one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz)
    values are supplied.

    ``lp1`` and ``l_in`` may be one value per band (already averaged) or a
    two-dimensional ``(positions, bands)`` array, in which case the positions
    are energy-averaged. Subareas scanned separately should first be combined
    with :func:`combine_subareas`.

    :param lp1: Source-room sound pressure levels, in dB.
    :param l_in: Normal sound intensity levels over the measurement surface,
        in dB.
    :param measurement_area: Measurement-surface area ``Sm``, in m².
    :param area: Specimen area ``S``, in m².
    :param kc: Adaptation term ``Kc`` per band (dB) for the modified index,
        or ``None`` to skip it.
    :return: :class:`IntensityReductionResult`.
    :raises ValueError: If the band counts differ, if ``measurement_area`` /
        ``area`` are not positive, or if inputs are non-finite.
    """
    lp1_bands = _as_band_levels(lp1, "lp1")
    l_in_bands = _as_band_levels(l_in, "l_in")
    if lp1_bands.shape != l_in_bands.shape:
        raise ValueError("'lp1' and 'l_in' must share the same band count.")
    sm = _positive_area(measurement_area, "measurement_area")
    s = _positive_area(area, "area")

    r_i = lp1_bands - _DIFFUSE_FIELD - (l_in_bands + 10.0 * np.log10(sm / s))

    r_i_modified: np.ndarray | None = None
    if kc is not None:
        kc_bands = np.asarray(kc, dtype=np.float64)
        if kc_bands.shape != r_i.shape:
            raise ValueError("'kc' must share the band count of the levels.")
        if not np.all(np.isfinite(kc_bands)):
            raise ValueError("'kc' must contain only finite values.")
        r_i_modified = r_i + kc_bands

    rating = weighted_rating(r_i) if r_i.size in (16, 5) else None
    rating_modified = (
        weighted_rating(r_i_modified)
        if r_i_modified is not None and r_i_modified.size in (16, 5)
        else None
    )
    return IntensityReductionResult(
        r_i=r_i,
        r_i_modified=r_i_modified,
        rating=rating,
        rating_modified=rating_modified,
        area=s,
        measurement_area=sm,
    )


def intensity_element_normalized_difference(
    lp1: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
    *,
    measurement_area: float,
    n: int = 1,
) -> IntensityElementNormalizedResult:
    """
    Intensity element normalized level difference per ISO 15186-1 (Formula (8)).

    Computes, per frequency band, the intensity element normalized level
    difference of a single element unit

    ``DI,n,e = Lp1 - 6 - (LIn + 10 lg(Sm / A0)) + 10 lg N`` dB

    from the average source-room sound pressure level ``Lp1``, the average
    normal sound intensity level ``LIn`` over the measurement surface of area
    ``Sm`` (``measurement_area``), the reference absorption area ``A0 = 10
    m²`` and the number ``N`` of element units installed within the surface.
    The weighted rating ``DI,n,e,w`` is computed via
    :func:`phonometry.weighted_rating` (ISO 717-1) when exactly 16 or 5 values
    are supplied.

    .. note::
        The printed Formula (8) *subtracts* its ``10 lg(N)`` term. That sign
        cannot be derived: measuring ``N`` identical units together raises
        the transmitted power by ``10 lg N``, so recovering the per-unit
        ``DI,n,e`` requires *adding* ``10 lg N``, exactly as the
        pressure-based ISO 10140-2:2010 Formula (6) does with its
        ``10 lg(nA0/A)`` term (and consistently with ISO 15186-2:2010
        Formula (12), which is Formula (8) without an ``N`` term). This
        function implements the corrected per-unit form and warns when
        ``n > 1`` deviates from the print (see ``docs/ERRATA.md``).

    :param lp1: Source-room sound pressure levels, in dB.
    :param l_in: Normal sound intensity levels over the measurement surface,
        in dB.
    :param measurement_area: Measurement-surface area ``Sm``, in m².
    :param n: Number ``N`` of small element units in the surface (Default: 1).
    :return: :class:`IntensityElementNormalizedResult`.
    :raises ValueError: If the band counts differ, if ``measurement_area`` is
        not positive, if ``n`` is not a positive integer, or if inputs are
        non-finite.
    """
    lp1_bands = _as_band_levels(lp1, "lp1")
    l_in_bands = _as_band_levels(l_in, "l_in")
    if lp1_bands.shape != l_in_bands.shape:
        raise ValueError("'lp1' and 'l_in' must share the same band count.")
    sm = _positive_area(measurement_area, "measurement_area")
    if int(n) != n or n < 1:
        raise ValueError("'n' must be a positive integer.")
    if n > 1:
        warnings.warn(
            "ISO 15186-1:2000 Formula (8) as printed subtracts 10 lg(N); "
            "that sign is physically inconsistent (see docs/ERRATA.md). The "
            "per-unit DI,n,e is returned with the corrected +10 lg(N) term, "
            "consistent with ISO 10140-2:2010 Formula (6).",
            UserWarning,
            stacklevel=2,
        )

    d_i_n_e = (
        lp1_bands
        - _DIFFUSE_FIELD
        - (l_in_bands + 10.0 * np.log10(sm / _A0))
        + 10.0 * np.log10(float(n))
    )
    rating = weighted_rating(d_i_n_e) if d_i_n_e.size in (16, 5) else None
    return IntensityElementNormalizedResult(
        d_i_n_e=d_i_n_e, rating=rating, measurement_area=sm, n=int(n)
    )
