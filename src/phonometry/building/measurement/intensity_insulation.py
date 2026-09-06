#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound insulation measured with sound intensity (ISO 15186).

This is the sound-**intensity** counterpart of the sound-pressure methods in
:mod:`phonometry.building.measurement.lab_insulation` (ISO 10140) and :mod:`phonometry.building.measurement.insulation`
(ISO 16283). Instead of an equivalent absorption area in the receiving room,
the transmitted sound power is measured directly by scanning an intensity
probe over a measurement surface enclosing the specimen. The main use is when
the traditional pressure method fails because of high flanking transmission
(ISO 15186-1:2000, Clause 1): the intensity method only captures the power
radiated by the element itself.

**Intensity sound reduction index (ISO 15186-1:2000, Clause 3.8, Formula
(7)).** From the average source-room sound pressure level ``Lp1`` and the
average normal sound intensity level ``LIn`` over the measurement surface,
in dB,

.. math::

   R_\mathrm{I} = L_{p1} - 6 - \left[ L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{S} \right]

with the measurement-surface area ``Sm`` and the specimen area ``S``. The
constant ``6`` dB is the diffuse-field relationship between the sound pressure
level and the sound intensity level incident on the specimen. The same formula
yields the *apparent* index ``R'I`` in the field (ISO 15186-2), the only
difference being the measurement condition (flanking is not suppressed), not
the arithmetic.

**Modified intensity sound reduction index (Clause 3.10, Formula (9)).**
:math:`R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}` corrects ``RI`` so that it reproduces the
ISO 140-3 (now ISO 10140-2) pressure result, which slightly overestimates
``R`` because the power radiated into the receiving room is underestimated.
The adaptation term ``Kc`` (Annex B) is
:math:`10 \log_{10}(1 + S_{\mathrm{b}2} \lambda / (8 V_2))` (Formula (B.1)) for a
well-defined receiving room of boundary area ``Sb2`` and volume ``V2``, or
the room-independent approximation :math:`10 \log_{10}(1 + 61.4 / f)`
(Formula (B.2)); both use the speed of sound :math:`c = 340` m/s so that
(B.1) with the reference room :math:`S_{\mathrm{b}2} = 117` m², :math:`V_2 = 81` m³
reduces to (B.2).

**Intensity element normalized level difference (Clause 3.9, Formula (8)).**
For small building elements, in dB,

.. math::

   D_\mathrm{I,n,e} = L_{p1} - 6 - \left( L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{A_0} \right)
   + 10 \log_{10} N

with the reference absorption area :math:`A_0 = 10` m² and the number ``N``
of element units in the measurement surface. The printed Formula (8)
subtracts its :math:`10 \log_{10} N` term instead of adding it, which is physically
inconsistent with ISO 10140-2:2010 Formula (6) and ISO 15186-2:2010
Formula (12); the corrected per-unit form is implemented (see
``docs/ERRATA.md``).

**Surface pressure-intensity indicator (Clause 3.6 / 6.4.2, Formula (10)).**
:math:`F_{pI} = L_p - L_{I\mathrm{n}}` qualifies the measurement surface: it must not
exceed 10 dB for a sound-reflecting specimen (6 dB when the receiving side is
sound absorbing), and the probe's pressure-residual intensity index must
exceed :math:`F_{pI} + 10` dB (Clause 4.1) for the dynamic capability to be
adequate.

**Frequency range (part 1, Clause 6.6).** The part 1 and part 2 quantities are
measured over the mandatory one-third-octave range 100 Hz to 5000 Hz (18
bands), optionally extended down to 50 Hz. The part 3 quantities at the end of
this module answer over 50 Hz to 160 Hz instead (its Clause 1.1), and results
from the two ranges are meant to be combined into one 50 Hz to 5000 Hz curve.
 The single-number weighted rating uses the ISO 717-1 core range, so
the automatic rating (``RI,w``, ``RI,M,w``, ``DI,n,e,w``) is formed via the
verified :func:`phonometry.building.weighted_rating` engine only when exactly
16 one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are
supplied.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ..._internal.validation import (
    check_engine,
    require_equal_counts,
    require_equal_shapes,
    require_finite_fields,
    require_positive,
    require_ranks,
    require_same_length,
)
from .insulation import (
    WeightedRatingResult,
    _as_band_levels,
    weighted_rating,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

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
    from ..._i18n import check_language

    check_language(language)
    check_engine(engine)
    if rating is None:
        msg = (
            f"The {label} report needs the ISO 717-1 single-number rating; it "
            "is formed only for exactly 16 one-third-octave (100 Hz to "
            "3150 Hz) or 5 octave (125 Hz to 2000 Hz) bands. Build the result "
            "with that band count."
        )
        raise ValueError(msg)
    n_bands = int(np.asarray(curve).size)
    if n_bands not in (16, 5):
        msg = (
            f"The {label} report supports only 16 one-third-octave (100 Hz to "
            "3150 Hz) or 5 octave (125 Hz to 2000 Hz) bands; the result "
            f"carries {n_bands}."
        )
        raise ValueError(msg)
    return rating


@dataclass(frozen=True)
class IntensityReductionResult:
    r"""Per-band intensity sound reduction index (ISO 15186-1:2000).

    :ivar r_i: Intensity sound reduction index
        :math:`R_\mathrm{I} = L_{p1} - 6 - [L_{I\mathrm{n}} + 10 \log_{10}(S_\mathrm{m}/S)]` per band, in dB
        (Clause 3.8, Formula (7)). In
        the field (ISO 15186-2) this is the apparent index ``R'I``.
    :ivar r_i_modified: Modified index :math:`R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}` per band, in dB
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

    def __post_init__(self) -> None:
        """Reject a result whose two indices were measured over different bands.

        The verbose ISO 15186-1 fiche annexes ``RI,M`` beside ``RI`` as a
        second column of one band table. It measures ``FpI`` and ``δpI0``
        against the reported band count before admitting them, because the
        caller passes those to :meth:`report`; the modified index arrives on
        the result itself and is admitted unmeasured. So the one column that
        invites a comparison band by band -- the same index before and after
        the ``Kc`` adaptation -- is the one nothing lines up, and the surplus
        entries of a longer array are dropped by the table without a word.

        The retained areas are pinned with the same predicate
        :func:`intensity_sound_reduction` applies to its own arguments,
        because a manually built result skips that entry point: ``S`` and
        ``Sm`` exist only to be printed in the Clause 8 g) statement, so a
        NaN admitted here surfaced as "S = nan m2" in the accredited
        sentence with no warning anywhere.

        The two indices are pinned finite for the same reason. ISO 15186 has
        no undeterminable band to flag: the only producer,
        :func:`intensity_sound_reduction`, forms ``RI`` from levels already
        refused non-finite by :func:`._as_band_levels` less a constant and a
        logarithm of two positive areas, and adds an equally checked ``Kc``
        for ``RI,M``. So a NaN band can only be a mistake, and admitting it
        printed a bare ``nan`` as a measured index in the accredited band
        table.

        :raises ValueError: if ``r_i`` and ``r_i_modified`` disagree or carry
            a non-finite band, or a supplied ``area``/``measurement_area`` is
            not a positive, finite number.
        """
        require_ranks(self, r_i=1, r_i_modified=1)
        require_same_length(self, "r_i", "r_i_modified")
        require_finite_fields(self, "r_i", "r_i_modified")
        for name in ("area", "measurement_area"):
            value = getattr(self, name)
            if value is not None:
                _positive_area(value, name)

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot ``RI`` against the shifted ISO 717-1 reference curve.

        Delegates to the weighted-rating plot (measured ``RI`` versus the
        shifted reference, unfavourable deviations shaded). Requires the
        automatic rating to be available (16 or 5 bands) and matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        if self.rating is None:
            msg = (
                "No single-number rating is available to plot (need 16 "
                "one-third-octave or 5 octave bands)."
            )
            raise ValueError(msg)
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

        from ..._report.iso15186 import render_iso15186_report

        return render_iso15186_report(
            self,
            rating,
            path,
            metadata=metadata,
            verbose=verbose,
            language=language,
            fpi=fpi,
            residual_index=residual_index,
        )


@dataclass(frozen=True)
class IntensityElementNormalizedResult:
    r"""Per-band intensity element normalized level difference (ISO 15186-1).

    :ivar d_i_n_e: Intensity element normalized level difference
        :math:`D_\mathrm{I,n,e} = L_{p1} - 6 - (L_{I\mathrm{n}} + 10 \log_{10}(S_\mathrm{m}/A_0)) +
        10 \log_{10} N` per band, in dB
        (Clause 3.9, Formula (8) with the corrected sign of its
        :math:`10 \log_{10} N` term; see ``docs/ERRATA.md``).
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
            msg = (
                "No single-number rating is available to plot (need 16 "
                "one-third-octave or 5 octave bands)."
            )
            raise ValueError(msg)
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

        from ..._report.iso15186 import render_iso15186_element_report

        return render_iso15186_element_report(
            self,
            rating,
            path,
            metadata=metadata,
            verbose=verbose,
            language=language,
            fpi=fpi,
            residual_index=residual_index,
        )


def _positive_area(value: float, name: str) -> float:
    """Return ``value`` as a positive, finite area, or raise."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        msg = f"'{name}' must be positive."
        raise ValueError(msg)
    return v


@overload
def adaptation_term_kc(
    freq: Sequence[float] | np.ndarray,
    *,
    boundary_area: float,
    volume: float,
) -> np.ndarray: ...


@overload
def adaptation_term_kc(freq: Sequence[float] | np.ndarray) -> np.ndarray: ...


def adaptation_term_kc(
    freq: Sequence[float] | np.ndarray,
    *,
    boundary_area: float | None = None,
    volume: float | None = None,
) -> np.ndarray:
    r"""Adaptation term ``Kc`` per ISO 15186-1:2000, Annex B.

    Returns, per one-third-octave midband frequency, the term ``Kc`` that
    turns the intensity sound reduction index ``RI`` into the modified index
    :math:`R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}` (Clause 3.10). Two forms are available:

    - **Well-defined receiving room (Formula (B.1)):** when both
      ``boundary_area`` (``Sb2``) and ``volume`` (``V2``) are supplied,
      :math:`K_\mathrm{c} = 10 \log_{10}(1 + S_{\mathrm{b}2} \lambda / (8 V_2))` with the midband
      wavelength :math:`\lambda = c / f` and :math:`c = 340` m/s.
    - **Room-independent approximation (Formula (B.2)):** when neither is
      supplied, :math:`K_\mathrm{c} = 10 \log_{10}(1 + 61.4 / f)`, the exact reduction of
      (B.1) for the reference room :math:`S_{\mathrm{b}2} = 117` m²,
      :math:`V_2 = 81` m³.

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
        msg = "'freq' must be one-dimensional."
        raise ValueError(msg)
    if not np.all(np.isfinite(f)) or np.any(f <= 0.0):
        msg = "'freq' must contain positive, finite values."
        raise ValueError(msg)

    if boundary_area is None and volume is None:
        ratio = _KC_APPROX_COEFF / f
    elif boundary_area is not None and volume is not None:
        sb2 = _positive_area(boundary_area, "boundary_area")
        v2 = _positive_area(volume, "volume")
        wavelength = _SPEED_OF_SOUND / f
        ratio = sb2 * wavelength / (8.0 * v2)
    else:
        msg = (
            "Supply both 'boundary_area' and 'volume' for Formula (B.1), or "
            "neither for the Formula (B.2) approximation."
        )
        raise ValueError(msg)
    return 10.0 * np.log10(1.0 + ratio)


def surface_pressure_intensity_indicator(
    lp: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
) -> np.ndarray:
    r"""Surface pressure-intensity indicator ``FpI`` (ISO 15186-1, Formula (10)).

    Returns :math:`F_{pI} = L_p - L_{I\mathrm{n}}` per band from the surface- and
    time-averaged
    sound pressure level ``Lp`` and normal sound intensity level ``LIn`` on
    the measurement surface (Clause 3.6 / 6.4.2). The measurement surface is
    adequately qualified when ``FpI`` does not exceed 10 dB for a
    sound-reflecting specimen, or 6 dB when the receiving side is sound
    absorbing (Clause 6.4.2 flags :math:`F_{pI} > 10` dB /
    :math:`F_{pI} > 6` dB as not
    satisfactory); in addition the probe's pressure-residual intensity index
    must exceed :math:`F_{pI} + 10` dB (Clause 4.1).

    :param lp: Surface-averaged sound pressure levels, in dB.
    :param l_in: Normal sound intensity levels on the surface, in dB.
    :return: The indicator ``FpI`` per band, in dB.
    :raises ValueError: If the shapes differ or contain non-finite values.
    """
    p = np.asarray(lp, dtype=np.float64)
    i = np.asarray(l_in, dtype=np.float64)
    require_equal_shapes(
        "surface_pressure_intensity_indicator",
        {"lp": p.shape, "l_in": i.shape},
        "band",
    )
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(i))):
        msg = "Levels must contain only finite values."
        raise ValueError(msg)
    return p - i


def combine_subareas(
    l_in: Sequence[Sequence[float]] | np.ndarray,
    measurement_area: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, float]:
    r"""Combine per-subarea intensity levels (ISO 15186-1, Formulas (11)-(12)).

    When the measurement surface is divided into subareas ``Smi`` each scanned
    individually, the normal sound intensity level over the whole surface is
    the area-weighted energy average, in dB,

    .. math::

       L_{I\mathrm{n}} = 10 \log_{10}\!\left[ \frac{1}{S_\mathrm{m}}
       \sum_i S_{\mathrm{m}i}\, 10^{0.1 L_{I\mathrm{n}i}} \right]

    with the total measured area :math:`S_\mathrm{m} = \sum_i |S_{\mathrm{m}i}|`
    (Formula (12)).

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
        and the total measured area
        :math:`S_\mathrm{m} = \sum \lvert S_{\mathrm{m}i} \rvert`, in m².
    :raises ValueError: If the shapes are inconsistent or values non-finite,
        if any subarea area is zero, or if the signed energy sum of
        Formula (11) is not positive in some band (the reverse flows cancel
        or exceed the forward flow, so no level exists).
    """
    levels = np.asarray(l_in, dtype=np.float64)
    if levels.ndim != 2:  # noqa: PLR2004
        msg = "'l_in' must be a two-dimensional (subareas, bands) array."
        raise ValueError(msg)
    areas = np.asarray(measurement_area, dtype=np.float64)
    if areas.ndim != 1:
        msg = "'measurement_area' must be a one-dimensional array of subarea areas."
        raise ValueError(msg)
    require_equal_counts(
        "combine_subareas",
        {"measurement_area": areas.size, "l_in rows": levels.shape[0]},
        "subarea",
    )
    if not np.all(np.isfinite(levels)):
        msg = "'l_in' must contain only finite values."
        raise ValueError(msg)
    if not np.all(np.isfinite(areas)) or not np.all(np.abs(areas) > 0.0):
        msg = (
            "'measurement_area' must contain non-zero, finite areas (negative "
            "marks a reverse-flow subarea, Clause 6.4.6)."
        )
        raise ValueError(msg)

    sm = float(np.sum(np.abs(areas)))
    energy = np.sum(areas[:, None] * 10.0 ** (0.1 * levels), axis=0)
    if np.any(energy <= 0.0):
        msg = (
            "The signed subarea energy sum of Formula (11) is not positive in "
            "at least one band: the negative-direction subareas cancel or "
            "exceed the forward flow, so no combined intensity level exists."
        )
        raise ValueError(msg)
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
    r"""Intensity sound reduction index per ISO 15186-1:2000 (Formula (7)).

    Computes, per frequency band, the intensity sound reduction index, in dB,

    .. math::

       R_\mathrm{I} = L_{p1} - 6 - \left[ L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{S} \right]

    from the average source-room sound pressure level ``Lp1`` and the average
    normal sound intensity level ``LIn`` over the measurement surface of area
    ``Sm`` (``measurement_area``), for a specimen of area ``S`` (``area``). The
    same formula gives the apparent index ``R'I`` in the field (ISO 15186-2).
    When an adaptation term ``kc`` is supplied (see
    :func:`adaptation_term_kc`), the modified index
    :math:`R_\mathrm{I,M} = R_\mathrm{I} + K_\mathrm{c}` (Formula (9)) is also
    formed. Weighted ratings ``RI,w`` (and ``RI,M,w``) are computed via
    :func:`phonometry.building.weighted_rating` (ISO 717-1) when exactly 16
    one-third-octave (100-3150 Hz) or 5 octave (125-2000 Hz) values are
    supplied.

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
    require_equal_shapes(
        "intensity_sound_reduction",
        {"lp1": lp1_bands.shape, "l_in": l_in_bands.shape},
        "band",
    )
    sm = _positive_area(measurement_area, "measurement_area")
    s = _positive_area(area, "area")

    r_i = lp1_bands - _DIFFUSE_FIELD - (l_in_bands + 10.0 * np.log10(sm / s))

    r_i_modified: np.ndarray | None = None
    if kc is not None:
        kc_bands = np.asarray(kc, dtype=np.float64)
        require_equal_shapes(
            "intensity_sound_reduction",
            {"lp1": lp1_bands.shape, "kc": kc_bands.shape},
            "band",
        )
        if not np.all(np.isfinite(kc_bands)):
            msg = "'kc' must contain only finite values."
            raise ValueError(msg)
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
    r"""Intensity element normalized level difference per ISO 15186-1 (Formula (8)).

    Computes, per frequency band, the intensity element normalized level
    difference of a single element unit, in dB,

    .. math::

       D_\mathrm{I,n,e} = L_{p1} - 6 - \left( L_{I\mathrm{n}} + 10 \log_{10}\frac{S_\mathrm{m}}{A_0}
       \right) + 10 \log_{10} N

    from the average source-room sound pressure level ``Lp1``, the average
    normal sound intensity level ``LIn`` over the measurement surface of area
    ``Sm`` (``measurement_area``), the reference absorption area
    :math:`A_0 = 10` m² and the number ``N`` of element units installed within
    the surface. The weighted rating ``DI,n,e,w`` is computed via
    :func:`phonometry.building.weighted_rating` (ISO 717-1) when exactly 16 or
    5 values are supplied.

    .. note::
        The printed Formula (8) *subtracts* its :math:`10 \log_{10} N` term. That
        sign cannot be derived: measuring ``N`` identical units together
        raises the transmitted power by :math:`10 \log_{10} N`, so recovering the
        per-unit ``DI,n,e`` requires *adding* :math:`10 \log_{10} N`, exactly as
        the pressure-based ISO 10140-2:2010 Formula (6) does with its
        :math:`10 \log_{10}(n A_0/A)` term (and consistently with ISO 15186-2:2010
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
    require_equal_shapes(
        "intensity_element_normalized_difference",
        {"lp1": lp1_bands.shape, "l_in": l_in_bands.shape},
        "band",
    )
    sm = _positive_area(measurement_area, "measurement_area")
    if int(n) != n or n < 1:
        msg = "'n' must be a positive integer."
        raise ValueError(msg)
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


# ---------------------------------------------------------------------------
# ISO 15186-3:2002 -- laboratory measurements at low frequencies
# ---------------------------------------------------------------------------

#: Level difference between the sound pressure level measured *on the surface
#: of the test specimen* and the incident sound intensity level
#: (ISO 15186-3:2002, Clause 3.8, Formula (7)): 9 dB.
#:
#: Part 1 subtracts 6 dB because its ``Lp1`` is the diffuse-field average of
#: the source room. Part 3 measures the pressure against the specimen itself,
#: and close to a rigid boundary a diffuse field carries twice the mean-square
#: pressure it carries away from one, which is 3 dB more level for the same
#: incident field; the 9 dB is that 6 plus that 3. Reading part 1's 6 dB into
#: a part 3 measurement would report a specimen 3 dB better than it is.
_SURFACE_FIELD = 9.0

#: The one-third-octave bands ISO 15186-3:2002 Clause 6.6 requires, in hertz.
_LOW_FREQUENCY_BANDS: tuple[int, ...] = (50, 63, 80)

#: The bands the same clause allows the measurement to be extended with.
_LOW_FREQUENCY_OPTIONAL_BANDS: tuple[int, ...] = (100, 125, 160)

#: Largest surface-pressure intensity indicator the measurement surface may
#: show, in dB, by the kind of test specimen (ISO 15186-3:2002, Clause 6.4.2):
#: 10 dB for a sound-reflecting one, 6 dB for one presenting a sound-absorbing
#: surface in the receiving room. The wall opposite the specimen absorbs in
#: either case, because Clause 5.1 requires it of the facility.
_FPI_LIMIT_REFLECTING = 10.0
_FPI_LIMIT_ABSORBING = 6.0

#: Reference static pressure of Annex A, in pascals (Formula (A.4)).
_ANNEX_A_B0 = 101300.0
#: Characteristic impedance of air at 0 degC and B0, in N s/m^3 (Formula (A.4)).
_ANNEX_A_RHO_C0 = 427.0
#: Celsius-to-kelvin offset as Formula (A.4) prints it: 273, not 273,15.
_ANNEX_A_KELVIN = 273.0
#: Speed of sound of Formula (A.5), c = 331 + 0,6 theta.
_ANNEX_A_C0 = 331.0
_ANNEX_A_C_PER_DEGREE = 0.6
#: The constant inside the radiation-efficiency approximation (Formula (A.3)).
_ANNEX_A_SIGMA_CONST = 0.20
#: Smallest specimen area Formula (A.3) is stated to be valid for, in m^2.
_ANNEX_A_MIN_AREA = 1.0


def limp_panel_reduction_index(
    frequencies: Sequence[float] | np.ndarray,
    *,
    surface_mass: float,
    area: float,
    temperature: float = 23.0,
    static_pressure: float = _ANNEX_A_B0,
) -> np.ndarray:
    r"""Sound reduction index of a limp panel (ISO 15186-3:2002, Annex A).

    Annex A is normative and is how a laboratory qualifies itself: measure a
    limp panel of area :math:`S > 1~\text{m}^2`, calculate what it should
    read, and require the two to agree within 4,0 dB from 50 Hz to 160 Hz.
    This is the calculated half.

    A.1 states two different things about the area, and this enforces the
    second: the *qualification panel* is required to be larger than 1 m², while
    Formula (A.3) is declared valid "if the area of the test specimen is at
    least 1 m²". A panel of exactly 1 m² is therefore refused as a
    qualification but accepted as an input, which is the boundary the code
    takes.

    .. math::

       R = R_0 - 10 \lg 2\sigma_\mathrm{d} \tag{A.1}

       R_0 = 20 \lg \frac{\pi f m}{\rho c} \tag{A.2}

       \sigma_\mathrm{d} = \frac{1}{2}
       \left[ 0{,}20 + \ln\left( 2\pi \frac{f}{c} \sqrt{S} \right) \right] \tag{A.3}

    with the characteristic impedance and the speed of sound taken from the
    climate of the test (Formulas (A.4) and (A.5)),

    .. math::

       \rho c = 427 \sqrt{\frac{273}{273 + \theta}} \cdot \frac{B}{B_0},
       \qquad c = 331 + 0{,}6\,\theta

    The panel is limp by assumption: :math:`R_0` is the mass law and
    :math:`\sigma_\mathrm{d}` the radiation efficiency of forced transmission
    alone. The 160 Hz ceiling is not this annex's own: Clause 1.1 applies the
    whole of this part over 50 Hz to 160 Hz, and the qualification inherits it.

    Any frequency is computed, and no range is imposed. A.1 declares Formula
    (A.3) valid "for the frequency range of this part of ISO 15186", so a
    result outside 50 Hz to 160 Hz is the model evaluated past its stated
    validity: useful for seeing where a real panel leaves it, not usable as a
    qualification. Restricting the input is the caller's to do, unlike
    :func:`low_frequency_intensity_reduction`, whose quantity is defined over
    that range and nowhere else.

    :param frequencies: One-third-octave mid-band frequencies, in hertz.
    :param surface_mass: Surface mass ``m`` of the panel, in kg/m².
    :param area: Panel area ``S``, in m². Formula (A.3) is stated valid for
        at least 1 m², so a smaller one is refused rather than extrapolated.
        A panel used to qualify a facility has to exceed 1 m² (A.1).
    :param temperature: Air temperature ``theta``, in degrees Celsius.
    :param static_pressure: Static pressure ``B``, in pascals.
    :return: The calculated sound reduction index per band, in dB.
    :raises ValueError: for a non-finite or non-positive frequency, surface
        mass, area below 1 m², or a climate the formulas cannot be evaluated
        in.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.ndim != 1:
        msg = (
            "'frequencies' must be one-dimensional: one mid-band frequency "
            "per band, and not a grid of measurement positions."
        )
        raise ValueError(msg)
    if freqs.size == 0 or not np.all(np.isfinite(freqs)) or np.any(freqs <= 0.0):
        msg = "'frequencies' must be positive and finite."
        raise ValueError(msg)
    m = require_positive(surface_mass, "surface_mass")
    s = _positive_area(area, "area")
    if s < _ANNEX_A_MIN_AREA:
        msg = (
            "'area' must be at least 1 m2: Formula (A.3) is stated only for a "
            f"panel that large (ISO 15186-3:2002, A.1); got {s:g} m2."
        )
        raise ValueError(msg)
    theta = float(temperature)
    b = require_positive(static_pressure, "static_pressure")
    if not np.isfinite(theta) or theta <= -_ANNEX_A_KELVIN:
        msg = "'temperature' must be finite and above -273 degC."
        raise ValueError(msg)
    c = _ANNEX_A_C0 + _ANNEX_A_C_PER_DEGREE * theta
    if c <= 0.0:
        msg = "'temperature' puts the speed of sound of Formula (A.5) at or below zero."
        raise ValueError(msg)
    rho_c = (
        _ANNEX_A_RHO_C0
        * np.sqrt(_ANNEX_A_KELVIN / (_ANNEX_A_KELVIN + theta))
        * (b / _ANNEX_A_B0)
    )
    sigma_d = 0.5 * (
        _ANNEX_A_SIGMA_CONST + np.log(2.0 * np.pi * (freqs / c) * np.sqrt(s))
    )
    if np.any(sigma_d <= 0.0):
        msg = (
            "Formula (A.3) gives a non-positive radiation efficiency for one of "
            "'frequencies', which Formula (A.1) cannot take the logarithm of."
        )
        raise ValueError(msg)
    r0 = 20.0 * np.log10(np.pi * freqs * m / rho_c)
    return np.asarray(r0 - 10.0 * np.log10(2.0 * sigma_d), dtype=np.float64)


#: Relative tolerance that matches a given mid-band frequency to the band it
#: names. A nominal label and its exact series value differ by 0,15 % and
#: neighbouring one-third octaves by 26 %, so this separates them cleanly.
_BAND_TOLERANCE = 0.03


def _validated_element_count(elements: object) -> int:
    """The number ``N`` of element units, as a positive integer.

    :param elements: The caller's ``elements`` argument, of any type.
    :return: ``N`` as a built-in ``int``.
    :raises ValueError: for a bool (``True`` counts no units), a value that is
        not a whole number, one below 1, or anything an integer cannot be
        taken from.
    """
    msg = "'elements' must be a positive integer."
    if isinstance(elements, (bool, np.bool_)):
        raise ValueError(msg)
    try:
        count = int(elements)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise ValueError(msg) from exc
    if count != elements or count < 1:
        raise ValueError(msg)
    return int(count)


def _validated_absorbing_flag(absorbing_specimen_surface: object) -> bool:
    """The Clause 6.4.2 case, as a bool.

    Rejected rather than coerced: every non-empty string is truthy, so
    ``"reflecting"`` would silently pick the 6 dB limit of the absorbing case
    and refuse a measurement the clause admits.

    :param absorbing_specimen_surface: The caller's argument, of any type.
    :return: The flag as a built-in ``bool``.
    :raises ValueError: for anything that is not a bool.
    """
    if not isinstance(absorbing_specimen_surface, (bool, np.bool_)):
        msg = (
            "'absorbing_specimen_surface' must be True or False: it selects "
            "between the two limits of Clause 6.4.2, and a truthy value of "
            "another type would silently pick the tighter one."
        )
        raise ValueError(msg)
    return bool(absorbing_specimen_surface)


def _low_frequency_limit(absorbing_specimen_surface: bool) -> float:
    """The Clause 6.4.2 limit on ``FpI`` for this kind of test specimen."""
    if absorbing_specimen_surface:
        return _FPI_LIMIT_ABSORBING
    return _FPI_LIMIT_REFLECTING


def _low_frequency_qualification(
    l_p: Sequence[float] | np.ndarray | None,
    l_in: np.ndarray,
    *,
    owner: str,
    absorbing_specimen_surface: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Formula (5) and the Clause 6.4.2 verdict it feeds, or ``(None, None)``.

    Clause 6.4.2 asks for the receiving-side pressure level "if possible", so
    an absent ``l_p`` leaves both halves unanswered rather than guessed.
    """
    if l_p is None:
        return None, None
    lp_receiving = _as_band_levels(l_p, "l_p")
    require_equal_shapes(owner, {"l_p": lp_receiving.shape, "l_in": l_in.shape}, "band")
    f_pi = surface_pressure_intensity_indicator(lp_receiving, l_in)
    limit = _low_frequency_limit(_validated_absorbing_flag(absorbing_specimen_surface))
    return f_pi, np.asarray(f_pi <= limit, dtype=bool)


def _check_low_frequency_bands(
    frequencies: Sequence[float] | np.ndarray | None,
    band_count: int,
    *,
    owner: str,
) -> np.ndarray | None:
    """Clause 1.1: this part applies from 50 Hz to 160 Hz and nowhere else.

    The six one-third octaves are matched by proximity rather than equality,
    because the same band is written 63 Hz on a filter and 63,096 Hz by the
    exact series; neighbouring bands sit 26 % apart, so nothing is ambiguous
    at this tolerance.
    """
    if frequencies is None:
        return None
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.ndim != 1:
        msg = (
            "'frequencies' must be one-dimensional: one mid-band frequency "
            "per band, and not a grid of measurement positions."
        )
        raise ValueError(msg)
    require_equal_counts(owner, {"frequencies": freqs.size, "l_in": band_count})
    bands = np.asarray(
        (*_LOW_FREQUENCY_BANDS, *_LOW_FREQUENCY_OPTIONAL_BANDS), dtype=np.float64
    )
    known = np.isclose(freqs[:, None], bands[None, :], rtol=_BAND_TOLERANCE).any(axis=1)
    if not known.all():
        stray = float(freqs[~known][0])
        msg = (
            "'frequencies' must be one-third-octave mid-band frequencies from "
            "50 Hz to 160 Hz, the range this part of ISO 15186 applies over "
            f"(Clause 1.1); got {stray:g} Hz."
        )
        raise ValueError(msg)
    return freqs


def _check_indicator_pair(owner: object) -> None:
    """Reject a result carrying only one half of the Clause 6.4.2 answer."""
    indicator = owner.surface_pressure_intensity  # type: ignore[attr-defined]
    qualified = owner.qualified  # type: ignore[attr-defined]
    if (indicator is None) != (qualified is None):
        msg = (
            "'surface_pressure_intensity' and 'qualified' are the two "
            "halves of one Clause 6.4.2 answer and are given together or "
            "not at all."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class LowFrequencyIntensityResult:
    r"""Per-band intensity sound reduction index at low frequencies.

    The result of ISO 15186-3:2002, Clause 3.8, Formula (7). It differs from
    its part 1 sibling in where the source-room pressure is measured and so in
    what has to be subtracted from it: 9 dB against the surface of the
    specimen, where part 1 subtracts 6 dB from a room average.

    :ivar r_i: Intensity sound reduction index
        :math:`R_\mathrm{I} = L_{p\mathrm{S}} - 9 - [L_{I\mathrm{n}} + 10\lg(S_\mathrm{m}/S)]`
        per band, in dB.
    :ivar surface_pressure_intensity: Surface-pressure intensity indicator
        :math:`F_{pI} = L_p - L_{I\mathrm{n}}` per band, in dB (Formula (5)),
        which Clause 7 requires to be reported beside the index, or ``None``
        where the receiving-side pressure level was not measured alongside
        the intensity. Clause 6.4.2 only asks for that measurement "if
        possible".
    :ivar qualified: ``True`` in each band whose ``FpI`` is within the limit
        Clause 6.4.2 sets, ``False`` where the measurement surface is not
        qualified and the index is not a result the standard admits, and
        ``None`` throughout when the indicator itself is ``None``.
    :ivar frequencies: Mid-band frequencies, in hertz, or ``None``.
    :ivar area: Test-object area ``S``, in m².
    :ivar measurement_area: Measurement-surface area ``Sm``, in m².
    :ivar absorbing_specimen_surface: Which of the two Clause 6.4.2 limits
        was applied, 6 dB when the specimen presents a sound-absorbing surface
        in the receiving room and 10 dB when it is sound-reflecting.
    """

    r_i: np.ndarray
    surface_pressure_intensity: np.ndarray | None
    qualified: np.ndarray | None
    frequencies: np.ndarray | None
    area: float
    measurement_area: float
    absorbing_specimen_surface: bool

    def __post_init__(self) -> None:
        """Reject a result whose per-band arrays do not index each other.

        Every reader of this result walks the four arrays together: the plot
        draws one bar per band and hatches it by ``qualified``, and a report
        prints the indicator beside the index. One array a band short raises
        an ``IndexError`` somewhere else entirely, so the shapes are pinned
        where they are built.

        :raises ValueError: if the per-band arrays disagree in length, if
            ``frequencies`` is given and does not match them, or if either
            area is not positive and finite.
        """
        require_ranks(self, r_i=1)
        _check_indicator_pair(self)
        if self.surface_pressure_intensity is not None:
            require_ranks(self, surface_pressure_intensity=1, qualified=1)
            require_same_length(self, "r_i", "surface_pressure_intensity", "qualified")
        if self.frequencies is not None:
            require_equal_counts(
                "LowFrequencyIntensityResult",
                {"frequencies": self.frequencies.size, "r_i": self.r_i.size},
            )
        if self.surface_pressure_intensity is None:
            require_finite_fields(self, "r_i")
        else:
            require_finite_fields(self, "r_i", "surface_pressure_intensity")
        _positive_area(self.area, "area")
        _positive_area(self.measurement_area, "measurement_area")

    @property
    def indicator_limit(self) -> float:
        """The Clause 6.4.2 limit on ``FpI`` that this result was judged by.

        Reported beside the indicator, and drawn by ``plot`` so the figure
        does not have to restate which of the two limits applies.

        :return: 6.0 dB when the specimen presents a sound-absorbing surface in
            the receiving room, 10.0 dB when it is sound-reflecting.
        """
        return _low_frequency_limit(self.absorbing_specimen_surface)

    def plot(self, ax: Axes | None = None, language: str = "en", **kwargs: Any) -> Axes:
        """Draw the index per band, hatching any band Clause 6.4.2 refuses.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the band bar.
        :return: The axes.
        """
        from ..._plot.building import plot_low_frequency_intensity

        return plot_low_frequency_intensity(self, ax=ax, language=language, **kwargs)


def low_frequency_intensity_reduction(
    lp_surface: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
    *,
    measurement_area: float,
    area: float,
    l_p: Sequence[float] | np.ndarray | None = None,
    frequencies: Sequence[float] | np.ndarray | None = None,
    absorbing_specimen_surface: bool = False,
) -> LowFrequencyIntensityResult:
    r"""Intensity sound reduction index at low frequencies (ISO 15186-3:2002).

    Below 100 Hz a source room has too few modes for its average pressure to
    describe what reaches the specimen, so this part measures the pressure
    **on the surface of the specimen** instead and the receiving side with an
    intensity probe as before (Clause 3.8, Formula (7)):

    .. math::

       R_\mathrm{I} = L_{p\mathrm{S}} - 9 -
       \left[ L_{I\mathrm{n}} + 10 \lg \frac{S_\mathrm{m}}{S} \right] \mathrm{dB}

    The 9 dB is the whole difference from :func:`intensity_sound_reduction`,
    which subtracts 6: close to a rigid boundary a diffuse field carries twice
    the mean-square pressure it carries away from one, so the surface average
    sits three decibels above the room average of the same field.

    Clause 7 requires the surface-pressure intensity indicator
    :math:`F_{pI} = L_p - L_{I\mathrm{n}}` (Formula (5)) to be reported beside
    the index, and Clause 6.4.2 refuses the measurement surface where it
    exceeds 10 dB for a sound-reflecting test specimen, or 6 dB for a specimen
    with a sound-absorbing surface in the receiving room. Those bands come
    back flagged rather than dropped, because the standard's answer to them is
    to improve the measurement environment, not to discard the band.

    Both levels of Formula (5) are read on the measurement surface in the
    **receiving** room, so the indicator needs ``l_p`` and not the source-room
    surface levels Formula (7) is built from. Clause 6.4.2 asks for that
    second measurement "if possible", so it is optional here and its absence
    leaves the indicator and the qualification unanswered rather than
    guessed.

    Clause 6.4.2 refuses a negative measured intensity for the same reason,
    which a level cannot carry: pass the signed sub-area intensities through
    :func:`combine_subareas` first, which is where the sign lives.

    ``lp_surface`` and ``l_in`` may be one value per band or a
    ``(positions, bands)`` array, in which case the positions are
    energy-averaged. Sub-areas scanned separately are combined first with
    :func:`combine_subareas`, which is Formulas (9) and (10) here.

    :param lp_surface: Sound pressure levels over the surface of the test
        specimen in the source room, ``LpS``, in dB.
    :param l_in: Normal sound intensity levels over the measurement surface
        in the receiving room, in dB.
    :param measurement_area: Measurement-surface area ``Sm``, in m².
    :param area: Test-object area ``S``, in m².
    :param l_p: Sound pressure levels on the measurement surface in the
        receiving room, measured alongside ``l_in`` (Clause 6.4.2), in dB, or
        ``None`` when they were not. They are what Formula (5) subtracts
        ``l_in`` from; without them no band can be qualified.
    :param frequencies: Mid-band frequencies, in hertz, or ``None``. Clause
        6.6 requires at least the 50 Hz, 63 Hz and 80 Hz one-third octaves
        and allows 100 Hz, 125 Hz and 160 Hz; a band outside 50 Hz to 160 Hz
        is refused, because this method is defined for the low-frequency
        range alone.
    :param absorbing_specimen_surface: ``True`` when the test specimen
        presents a sound-absorbing surface in the receiving room, which
        tightens the Clause 6.4.2 limit from 10 dB to 6 dB. A specimen
        absorbing on one side only is mounted with that side towards the
        source room (Clause 5.3), so this is the two-absorbing-sides case.
    :return: :class:`LowFrequencyIntensityResult`.
    :raises ValueError: if the band counts differ, if either area is not
        positive, if any level is non-finite, or if ``frequencies`` carries a
        band outside the range this part is defined for.
    """
    lp_bands = _as_band_levels(lp_surface, "lp_surface")
    l_in_bands = _as_band_levels(l_in, "l_in")
    require_equal_shapes(
        "low_frequency_intensity_reduction",
        {"lp_surface": lp_bands.shape, "l_in": l_in_bands.shape},
        "band",
    )
    sm = _positive_area(measurement_area, "measurement_area")
    s = _positive_area(area, "area")
    freqs = _check_low_frequency_bands(
        frequencies, l_in_bands.size, owner="low_frequency_intensity_reduction"
    )
    f_pi, qualified = _low_frequency_qualification(
        l_p,
        l_in_bands,
        owner="low_frequency_intensity_reduction",
        absorbing_specimen_surface=absorbing_specimen_surface,
    )
    absorbing = _validated_absorbing_flag(absorbing_specimen_surface)
    r_i = np.asarray(
        lp_bands - _SURFACE_FIELD - (l_in_bands + 10.0 * np.log10(sm / s)),
        dtype=np.float64,
    )
    return LowFrequencyIntensityResult(
        r_i=r_i,
        surface_pressure_intensity=f_pi,
        qualified=qualified,
        frequencies=freqs,
        area=s,
        measurement_area=sm,
        absorbing_specimen_surface=absorbing,
    )


@dataclass(frozen=True)
class LowFrequencyElementResult:
    r"""Per-band element normalized level difference at low frequencies.

    The result of ISO 15186-3:2002, Clause 3.9, Formula (8), the small-element
    counterpart of :class:`LowFrequencyIntensityResult`. No single-number
    rating accompanies it: Clause 6.6 stops at 160 Hz, six one-third octaves,
    and ISO 717-1 needs sixteen.

    :ivar d_i_n_e: Intensity element normalized level difference
        :math:`D_{I\mathrm{n,e}} = L_{p\mathrm{S}} - 9 -
        [L_{I\mathrm{n}} - 10\lg(A_0/S_\mathrm{m}) - 10\lg N]` per band, in dB.
    :ivar surface_pressure_intensity: Surface-pressure intensity indicator
        :math:`F_{pI}` per band, in dB (Formula (5)), or ``None`` where the
        receiving-side pressure level was not measured alongside the
        intensity.
    :ivar qualified: The Clause 6.4.2 verdict per band, or ``None`` throughout
        when the indicator itself is ``None``.
    :ivar frequencies: Mid-band frequencies, in hertz, or ``None``.
    :ivar measurement_area: Measurement-surface area ``Sm``, in m².
    :ivar elements: Number ``N`` of element units installed within the
        measurement surface.
    :ivar absorbing_specimen_surface: Which of the two Clause 6.4.2 limits
        was applied.
    """

    d_i_n_e: np.ndarray
    surface_pressure_intensity: np.ndarray | None
    qualified: np.ndarray | None
    frequencies: np.ndarray | None
    measurement_area: float
    elements: int
    absorbing_specimen_surface: bool

    def __post_init__(self) -> None:
        """Reject a result whose per-band arrays do not index each other.

        :raises ValueError: if the per-band arrays disagree in length, if
            ``frequencies`` is given and does not match them, or if the
            measurement area is not positive and finite.
        """
        require_ranks(self, d_i_n_e=1)
        _check_indicator_pair(self)
        if self.surface_pressure_intensity is not None:
            require_ranks(self, surface_pressure_intensity=1, qualified=1)
            require_same_length(
                self, "d_i_n_e", "surface_pressure_intensity", "qualified"
            )
        if self.frequencies is not None:
            require_equal_counts(
                "LowFrequencyElementResult",
                {"frequencies": self.frequencies.size, "d_i_n_e": self.d_i_n_e.size},
            )
        if self.surface_pressure_intensity is None:
            require_finite_fields(self, "d_i_n_e")
        else:
            require_finite_fields(self, "d_i_n_e", "surface_pressure_intensity")
        _positive_area(self.measurement_area, "measurement_area")

    @property
    def indicator_limit(self) -> float:
        """The Clause 6.4.2 limit on ``FpI`` that this result was judged by.

        :return: 6.0 dB when the specimen presents a sound-absorbing surface
            in the receiving room, 10.0 dB when it is sound-reflecting.
        """
        return _low_frequency_limit(self.absorbing_specimen_surface)

    def plot(self, ax: Axes | None = None, language: str = "en", **kwargs: Any) -> Axes:
        """Draw ``DI,n,e`` per band, hatching any band Clause 6.4.2 refuses.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the band bar.
        :return: The axes.
        """
        from ..._plot.building import plot_low_frequency_element

        return plot_low_frequency_element(self, ax=ax, language=language, **kwargs)


def low_frequency_element_normalized_difference(
    lp_surface: Sequence[float] | np.ndarray,
    l_in: Sequence[float] | np.ndarray,
    *,
    measurement_area: float,
    elements: int = 1,
    l_p: Sequence[float] | np.ndarray | None = None,
    frequencies: Sequence[float] | np.ndarray | None = None,
    absorbing_specimen_surface: bool = False,
) -> LowFrequencyElementResult:
    r"""Element normalized level difference at low frequencies (ISO 15186-3).

    Clause 3.9, Formula (8), for small building elements measured with the
    surface-pressure method of this part:

    .. math::

       D_{I\mathrm{n,e}} = L_{p\mathrm{S}} - 9 -
       \left[ L_{I\mathrm{n}} - 10 \lg \frac{A_0}{S_\mathrm{m}}
       - 10 \lg N \right] \mathrm{dB}

    with the reference absorption area :math:`A_0 = 10` m². Two things
    separate it from its part 1 sibling
    (:func:`intensity_element_normalized_difference`): the 9 dB of the
    surface measurement in place of 6, and the sign of the :math:`10\lg N`
    term, which this part prints as the derivable one. Part 1 prints it
    subtracted, which is registered in ``docs/ERRATA.md``; the two parts
    disagree on the page, and this is the one that agrees with the physics.

    :param lp_surface: Sound pressure levels over the surface of the test
        specimen in the source room, ``LpS``, in dB. One value per band, or a
        ``(positions, bands)`` array that is energy-averaged.
    :param l_in: Normal sound intensity levels over the measurement surface
        in the receiving room, in dB.
    :param measurement_area: Measurement-surface area ``Sm``, in m².
    :param elements: Number ``N`` of identical element units installed within
        the measurement surface, at least 1.
    :param l_p: Sound pressure levels on the measurement surface in the
        receiving room, measured alongside ``l_in`` (Clause 6.4.2), in dB, or
        ``None`` when they were not.
    :param frequencies: Mid-band frequencies, in hertz, or ``None``. Clause
        6.6 admits 50 Hz to 160 Hz and nothing else.
    :param absorbing_specimen_surface: ``True`` when the test specimen
        presents a sound-absorbing surface in the receiving room, which
        tightens the Clause 6.4.2 limit from 10 dB to 6 dB.
    :return: :class:`LowFrequencyElementResult`.
    :raises ValueError: if the band counts differ, if the measurement area is
        not positive, if ``elements`` is not a positive integer, if any level
        is non-finite, or if ``frequencies`` carries a band outside the range
        this part is defined for.
    """
    lp_bands = _as_band_levels(lp_surface, "lp_surface")
    l_in_bands = _as_band_levels(l_in, "l_in")
    require_equal_shapes(
        "low_frequency_element_normalized_difference",
        {"lp_surface": lp_bands.shape, "l_in": l_in_bands.shape},
        "band",
    )
    sm = _positive_area(measurement_area, "measurement_area")
    n = _validated_element_count(elements)
    freqs = _check_low_frequency_bands(
        frequencies,
        l_in_bands.size,
        owner="low_frequency_element_normalized_difference",
    )
    f_pi, qualified = _low_frequency_qualification(
        l_p,
        l_in_bands,
        owner="low_frequency_element_normalized_difference",
        absorbing_specimen_surface=absorbing_specimen_surface,
    )
    d_i_n_e = np.asarray(
        lp_bands
        - _SURFACE_FIELD
        - (l_in_bands + 10.0 * np.log10(sm / _A0))
        + 10.0 * np.log10(n),
        dtype=np.float64,
    )
    return LowFrequencyElementResult(
        d_i_n_e=d_i_n_e,
        surface_pressure_intensity=f_pi,
        qualified=qualified,
        frequencies=freqs,
        measurement_area=sm,
        elements=n,
        absorbing_specimen_surface=_validated_absorbing_flag(
            absorbing_specimen_surface
        ),
    )
