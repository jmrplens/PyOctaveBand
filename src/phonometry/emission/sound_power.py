#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound power level of a noise source from sound pressure measurements over an
enveloping measurement surface: ISO 3744:2010 (engineering, accuracy grade 2)
and ISO 3746:2010 (survey, accuracy grade 3).

The source stands on one (or more) reflecting plane(s). Sound pressure levels
are measured at an array of microphone positions on a hypothetical surface of
area ``S`` enveloping the source (a hemisphere or a right parallelepiped). The
sound power level follows from the energy-averaged pressure level, the
background correction :math:`K_1`, the environmental correction :math:`K_2`
and the surface area (ISO 3744:2010 clause 8.2, equations (12), (16)-(18)):

.. math::

   \overline{L_p} = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}}
   \sum_i 10^{0.1 L_{pi}} \right] \tag{Eq. 12}

   K_1 = -10 \log_{10}\!\left( 1 - 10^{-0.1 \Delta L_p} \right) \tag{Eq. 16}

   K_2 = 10 \log_{10}\!\left( 1 + \frac{4S}{A} \right) \tag{Eq. A.2}

   L_p = \overline{L_p} - K_1 - K_2 \tag{Eq. 17}

   L_W = L_p + 10 \log_{10}\frac{S}{S_0}, \qquad S_0 = 1~\text{m}^2 \tag{Eq. 18}

The measurement surface area is a closed form of the source geometry: a full
hemisphere :math:`S = 2\pi r^2` (half :math:`\pi r^2`, quarter
:math:`\pi r^2/2`) for one, two or three reflecting planes (ISO 3744 clause
7.2.3); a parallelepiped :math:`S = 4(ab+bc+ca)` with :math:`a = 0.5\,l_1+d`,
:math:`b = 0.5\,l_2+d`, :math:`c = l_3+d` for one plane (clause 7.2.4,
equations (9)-(11)).

The A-weighted sound power level is combined from band levels with the
A-weighting band corrections :math:`C_k` of ISO 3744 Annex E (Tables
E.1/E.2):

.. math::

   L_{W\mathrm{A}} = 10 \log_{10}\!\left[ \sum_k 10^{0.1 (L_{Wk} + C_k)} \right]
   \tag{Eq. E.1}

ISO 3746:2010 shares the surfaces, the energy average and the LW/K1/K2 forms
but is coarser: fewer microphone positions (clause 8.2.1), a background
criterion of 3 dB instead of 6 dB (clause 8.4.1) and validity up to
:math:`K_{2\mathrm{A}} \le 7` dB instead of 4 dB (clause 4.3).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._internal.levels_math import energy_mean, energy_sum
from .._internal.types import as_float_or_array
from ._shared import (
    _S0,
    Grade,
    SoundPowerWarning,
    _a_weighting_corrections,
    _check_grade,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata
    from .declaration import DeclarationForm, NoiseEmissionDeclaration

__all__ = [
    "Grade",
    "RoomEnvironment",
    "SoundPowerResult",
    # Defined in :mod:`._shared`, which every method module imports; re-exported
    # here because this is the documented path of the warning class.
    "SoundPowerWarning",
    "Surface",
    "background_noise_correction",
    "environmental_correction",
    "measurement_positions",
    "sound_power_pressure",
]

Surface = Literal["hemisphere", "box"]


@dataclass(frozen=True)
class RoomEnvironment:
    r"""Room data behind the environmental correction ``K2`` (ISO 3744 Annex A).

    The three routes the standard offers to the equivalent sound absorption area
    ``A`` of the test room, in the order :func:`environmental_correction` tries
    them: ``A`` itself, the Sabine reverberation time with the room volume
    (:math:`A = 0.16 V / T`, Eq. A.3) and the mean absorption coefficient with
    the area of the room boundaries (:math:`A = \alpha S_v`, Eq. A.7). Each
    route is a pair that must be given whole; the empty environment carries no
    room data at all, which is the free field (:math:`K_2 = 0`).

    Every field may also be a per-band array, in which case ``K2`` comes out
    per band with that shape.

    :param absorption_area: Equivalent absorption area ``A`` (m^2), scalar or
        per band.
    :param reverberation_time: Sabine ``T`` (s), scalar or per band, with
        ``volume`` (Eq. A.3).
    :param volume: Room volume ``V`` (m^3), with ``reverberation_time``.
    :param mean_absorption_coefficient: ``alpha`` in (0, 1], scalar or per band,
        with ``room_surface`` (Eq. A.7).
    :param room_surface: Room boundary area ``Sv`` (m^2), with ``alpha``.
    """

    absorption_area: float | np.ndarray | None = None
    reverberation_time: float | np.ndarray | None = None
    volume: float | None = None
    mean_absorption_coefficient: float | np.ndarray | None = None
    room_surface: float | None = None


# --- ISO 3744:2010 Annex B, normative microphone coordinates (x/r,y/r,z/r) ---
#: Table B.1 - preferred positions for all sources (including tones).
_TABLE_B1: np.ndarray = np.array(
    [
        [0.16, -0.96, 0.22],
        [0.78, -0.60, 0.20],
        [0.78, 0.55, 0.31],
        [0.16, 0.90, 0.41],
        [-0.83, 0.32, 0.45],
        [-0.83, -0.40, 0.38],
        [-0.26, -0.65, 0.71],
        [0.74, -0.07, 0.67],
        [-0.26, 0.50, 0.83],
        [0.10, -0.10, 0.99],
        [0.91, -0.34, 0.22],
        [0.91, 0.38, 0.20],
        [-0.09, 0.95, 0.31],
        [-0.70, 0.59, 0.41],
        [-0.69, -0.56, 0.45],
        [-0.07, -0.92, 0.38],
        [0.43, -0.55, 0.71],
        [0.43, 0.61, 0.67],
        [-0.56, 0.02, 0.83],
        [0.14, 0.04, 0.99],
    ]
)
#: Table B.2 - positions for a broadband (tone-free) source.
_TABLE_B2: np.ndarray = np.array(
    [
        [-0.99, 0.0, 0.15],
        [0.50, -0.86, 0.15],
        [0.50, 0.86, 0.15],
        [-0.45, 0.77, 0.45],
        [-0.45, -0.77, 0.45],
        [0.89, 0.0, 0.45],
        [0.33, 0.57, 0.75],
        [-0.66, 0.0, 0.75],
        [0.33, -0.57, 0.75],
        [0.0, 0.0, 1.00],
        [0.99, 0.0, 0.15],
        [-0.50, 0.86, 0.15],
        [-0.50, -0.86, 0.15],
        [0.45, -0.77, 0.45],
        [0.45, 0.77, 0.45],
        [-0.89, 0.0, 0.45],
        [-0.33, -0.57, 0.75],
        [0.66, 0.0, 0.75],
        [-0.33, 0.57, 0.75],
        [0.0, 0.0, 1.00],
    ]
)
#: Table B.3 - source adjacent to three reflecting planes.
_TABLE_B3: np.ndarray = np.array(
    [
        [0.86, -0.50, 0.15],
        [0.45, -0.77, 0.45],
        [0.47, -0.47, 0.75],
        [0.50, -0.86, 0.15],
        [0.77, -0.45, 0.45],
        [0.47, -0.47, 0.75],
    ]
)

#: Background-noise criteria (low, high) in dB: below ``low`` the correction is
#: clamped (upper bound, warn); above ``high`` it is set to zero. ISO 3744
#: clause 8.2.3 (6, 15); ISO 3746 clause 8.4.1 (3, 10).
_K1_CRITERIA: dict[str, tuple[float, float]] = {
    "engineering": (6.0, 15.0),
    "survey": (3.0, 10.0),
}
#: Environmental-correction validity limit K2A, in dB. ISO 3744 clause 4.3.2
#: (4 dB); ISO 3746 clause 4.3 (7 dB).
_K2_LIMIT: dict[str, float] = {"engineering": 4.0, "survey": 7.0}
#: Minimum microphone positions by grade and number of reflecting planes.
#: ISO 3744 clause 8.1.1 (10/5/3); ISO 3746 clause 8.2.1 (4/3/3).
_MIN_HEMI_POSITIONS: dict[str, dict[int, int]] = {
    "engineering": {1: 10, 2: 5, 3: 3},
    "survey": {1: 4, 2: 3, 3: 3},
}
#: Minimum microphone positions on a parallelepiped: ISO 3744 clause C.1 (9 key
#: positions for rectangular partial areas); ISO 3746 clause C.1, Figure C.7 (4
#: positions for a floor-standing source on one reflecting plane).
_MIN_BOX_POSITIONS: dict[str, int] = {"engineering": 9, "survey": 4}
#: Typical A-weighted reproducibility standard deviation sigma_R0, in dB.
#: ISO 3744 Table 2 (1,5); ISO 3746 Table 1 (3,0, tone-free).
_SIGMA_R0: dict[str, float] = {"engineering": 1.5, "survey": 3.0}


@dataclass(frozen=True)
class SoundPowerResult:
    r"""Result of a sound power determination from surface pressure levels.

    ``sound_power_level`` is the per-band ``LW`` (ISO 3744 Eq. 18);
    ``surface_pressure_level`` the surface SPL ``Lp`` after the K1/K2
    corrections (Eq. 17); ``mean_pressure_level`` the raw energy-averaged
    level ``Lp'(ST)`` (Eq. 12). ``background_correction`` (K1) and
    ``environmental_correction`` (K2) are per band. ``sound_power_level_a`` is
    the A-weighted total ``LWA`` (Eq. E.1), computed only when ``frequencies``
    are supplied; for a single band it equals ``LW``, and for several bands
    without ``frequencies`` it is ``NaN`` (A-weighting needs the band centres).
    ``directivity_index`` is the apparent directivity index ``DIi*`` per
    microphone position and frequency band, shape ``(NM, NB)`` (Eq. 7,
    evaluated per band per clause 8.4). ``uncertainty`` is the expanded
    uncertainty
    :math:`U = 2\sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}` (95 %, ISO 3744
    clause 9.5).
    """

    frequencies: np.ndarray | None
    sound_power_level: np.ndarray
    surface_pressure_level: np.ndarray
    mean_pressure_level: np.ndarray
    background_correction: np.ndarray
    environmental_correction: np.ndarray
    directivity_index: np.ndarray
    surface_area: float
    sound_power_level_a: float
    uncertainty: float
    grade: str

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the LW spectrum with the A-weighted total annotated.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_sound_power

        check_language(language)
        return plot_sound_power(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 3744/3746 sound-power determination fiche to a PDF.

        Writes a one-page sound-power test sheet: the standard-basis line naming
        the applied method and accuracy grade (ISO 3744:2010 engineering grade 2
        or ISO 3746:2010 survey grade 3), an optional metadata header (client,
        noise source, test environment, instrumentation, climate, date), a
        per-band table (nominal octave/one-third-octave frequency, the surface
        sound-pressure level ``Lp`` and the band sound-power level ``LW``), the
        sound-power spectrum ``LW(f)``, the boxed A-weighted sound power level
        ``LWA`` (dB re 1 pW) with the total ``LW``, the expanded uncertainty
        ``U`` and the measurement surface area ``S``, an optional verdict row
        against a declared limit, and a measurement-basis strip stating the
        applied background (``K1``) and environmental (``K2``) corrections.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the noise source, ``test_room``
            the test environment, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``), the footer
            identity (``laboratory``, ``operator``, ``report_id``, ``notes``)
            and, via ``requirement``, a declared A-weighted sound-power limit
            the fiche checks the result against (lower is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds the
            energy-averaged level ``Lp'`` and the background (``K1``) and
            environmental (``K2``) corrections.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            raise ValueError(msg)
        from .._report.iso3744 import render_sound_power_report

        return render_sound_power_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )

    def declare(
        self,
        *,
        uncertainty: float | None = None,
        mode: str = "Operating mode 1",
        emission_pressure_level: float | None = None,
        emission_pressure_uncertainty: float | None = None,
        verification_level: float | None = None,
        machine: str | None = None,
        operating_conditions: str | None = None,
        noise_test_code: str | None = None,
        basic_standards: str | Sequence[str] = (),
        form: DeclarationForm = "dual-number",
    ) -> NoiseEmissionDeclaration:
        r"""Build an ISO 4871:1996 noise-emission declaration from this result.

        Wraps the A-weighted sound power level ``LWA`` of this measurement as the
        declared measured value ``L_WA`` of a single operating mode, with the
        uncertainty ``K_WA`` defaulting to the result's own expanded uncertainty
        ``U`` (ISO 3744/3746 clause 9.5). The declared single-number value is
        :math:`L_{W\mathrm{Ad}} = L_{W\mathrm{A}} + K_{W\mathrm{A}}` (ISO 4871 clause 3.15).

        :param uncertainty: ``K_WA`` in decibels; defaults to this result's
            expanded uncertainty :attr:`uncertainty`.
        :param mode: Operating-mode label for the declaration column.
        :param emission_pressure_level: Optional A-weighted emission sound
            pressure level ``L_pA`` at a work station, in decibels re 20 uPa.
        :param emission_pressure_uncertainty: ``K_pA`` in decibels; required
            with ``emission_pressure_level``.
        :param verification_level: Optional verification measurement ``L_1`` of
            the A-weighted sound power level (ISO 4871 clause 6).
        :param machine: Machine identification (clause 5 a).
        :param operating_conditions: Operating/mounting conditions (clause 5 c).
        :param noise_test_code: Noise test code the values were determined to
            (clause 5 b).
        :param basic_standards: Basic emission standard(s) used (clause 5 b).
        :param form: ``"dual-number"`` (default) or ``"single-number"``.
        :return: A single-mode
            :class:`~phonometry.emission.declaration.NoiseEmissionDeclaration`.
        :raises ValueError: If the A-weighted sound power level is not finite
            (several bands were combined without ``frequencies``).
        """
        from .declaration import NoiseEmissionDeclaration, OperatingModeDeclaration

        lwa = float(self.sound_power_level_a)
        if not np.isfinite(lwa):
            msg = (
                "declare() needs a finite A-weighted sound power level; supply "
                "'frequencies' to sound_power_pressure(...) so LWA is defined."
            )
            raise ValueError(msg)
        k = float(self.uncertainty if uncertainty is None else uncertainty)
        return NoiseEmissionDeclaration(
            modes=(
                OperatingModeDeclaration(
                    mode=mode,
                    sound_power_level=lwa,
                    sound_power_uncertainty=k,
                    emission_pressure_level=emission_pressure_level,
                    emission_pressure_uncertainty=emission_pressure_uncertainty,
                    verification_level=verification_level,
                ),
            ),
            machine=machine,
            operating_conditions=operating_conditions,
            noise_test_code=noise_test_code,
            basic_standards=basic_standards,
            form=form,
        )


def background_noise_correction(
    source_levels: np.ndarray,
    background_levels: np.ndarray,
    grade: Grade = "engineering",
) -> np.ndarray:
    r"""Background-noise correction ``K1`` per band (ISO 3744:2010 Eq. 16).

    :math:`K_1 = -10 \log_{10}\left( 1 - 10^{-0.1 \Delta L_p} \right)` with
    :math:`\Delta L_p = L_{\text{source}} - L_{\text{background}}`. For
    :math:`\Delta L_p` strictly above the upper criterion (15 dB engineering,
    10 dB survey) the background is negligible and :math:`K_1 = 0`; at the
    criterion itself Eq. (16) still applies (ISO 3744:2010, 8.2.3:
    :math:`6 \le \Delta L_p \le 15` dB; ISO 3746:2010, 8.3.3:
    :math:`3 \le \Delta L_p \le 10` dB). For :math:`\Delta L_p` below the
    lower criterion (6 dB engineering, 3 dB survey) the accuracy is reduced:
    ``K1`` is clamped to its value at that criterion and a
    :class:`SoundPowerWarning` is emitted, the result then being an upper
    bound (clause 8.2.3).

    :param source_levels: Levels with the source operating, in decibels.
    :param background_levels: Background-noise levels, in decibels.
    :param grade: ``'engineering'`` (ISO 3744) or ``'survey'`` (ISO 3746).
    :return: ``K1`` per band, in decibels.
    """
    low, high = _K1_CRITERIA[_check_grade(grade)]
    src = np.asarray(source_levels, dtype=np.float64)
    bg = np.asarray(background_levels, dtype=np.float64)
    delta = src - bg
    clamped = np.maximum(delta, low)
    k1 = -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * clamped))
    k1 = np.where(delta > high, 0.0, k1)
    if np.any(delta < low):
        warnings.warn(
            f"Background margin below {low:g} dB in one or more bands; K1 "
            "clamped to the criterion value and levels are upper bounds "
            "(ISO 3744:2010, 8.2.3).",
            SoundPowerWarning,
            stacklevel=2,
        )
    return np.asarray(k1, dtype=np.float64)


def _require_room_pair(
    first: Any, second: Any, first_name: str, second_name: str, equation: str
) -> None:
    """Reject a half-specified room pair for the ``K2`` absorption area.

    A half-specified room pair must never be read as free field: naming only
    one member of a pair is a mistake, not a ``K2 = 0`` request.

    :param first: First member of the pair, or ``None``.
    :param second: Second member of the pair, or ``None``.
    :param first_name: Argument name of the first member.
    :param second_name: Argument name of the second member.
    :param equation: ISO 3744 equation the pair belongs to.
    :raises ValueError: If exactly one member of the pair is given.
    """
    if (first is None) == (second is None):
        return
    missing = second_name if second is None else first_name
    msg = (
        f"{first_name} and {second_name} must be given together "
        f"({equation}); '{missing}' is missing."
    )
    raise ValueError(msg)


def _room_absorption_area(
    reverberation_time: float | np.ndarray | None,
    volume: float | None,
    mean_absorption_coefficient: float | np.ndarray | None,
    room_surface: float | None,
) -> float | np.ndarray | None:
    """Equivalent absorption area ``A`` from the room data, or ``None``.

    ``None`` means no room data was supplied at all, i.e. a free field.

    :param reverberation_time: Sabine ``T`` (s), with ``volume`` (Eq. A.3).
    :param volume: Room volume ``V`` (m^3), with ``reverberation_time``.
    :param mean_absorption_coefficient: ``alpha``, with ``room_surface``
        (Eq. A.7).
    :param room_surface: Room boundary area ``Sv`` (m^2), with ``alpha``.
    :return: ``A`` in square metres (scalar or per band), or ``None``.
    :raises ValueError: For a half-specified or non-physical room pair.
    """
    _require_room_pair(
        reverberation_time, volume, "reverberation_time", "volume", "Eq. A.3"
    )
    _require_room_pair(
        mean_absorption_coefficient,
        room_surface,
        "mean_absorption_coefficient",
        "room_surface",
        "Eq. A.7",
    )
    if reverberation_time is not None and volume is not None:
        t = np.asarray(reverberation_time, dtype=np.float64)
        if volume <= 0 or np.any(t <= 0.0):
            msg = "reverberation_time and volume must be > 0."
            raise ValueError(msg)
        return 0.16 * volume / t
    if mean_absorption_coefficient is not None and room_surface is not None:
        alpha = np.asarray(mean_absorption_coefficient, dtype=np.float64)
        if room_surface <= 0 or np.any(alpha <= 0.0) or np.any(alpha > 1.0):
            msg = "mean_absorption_coefficient must be in (0, 1] and room_surface > 0."
            raise ValueError(msg)
        return alpha * room_surface
    return None


def environmental_correction(
    surface_area: float,
    *,
    absorption_area: float | np.ndarray | None = None,
    reverberation_time: float | np.ndarray | None = None,
    volume: float | None = None,
    mean_absorption_coefficient: float | np.ndarray | None = None,
    room_surface: float | None = None,
) -> float | np.ndarray:
    r"""Environmental correction ``K2`` (ISO 3744:2010 Eq. A.2).

    :math:`K_2 = 10 \log_{10}\left( 1 + 4 S / A \right)` where ``A`` is the
    equivalent sound absorption area of the room. ``A`` is taken directly
    from ``absorption_area``, or from
    the Sabine reverberation time :math:`A = 0.16 V / T` (Eq. A.3,
    ``reverberation_time`` + ``volume``), or from the mean absorption
    coefficient :math:`A = \alpha S_v` (Eq. A.7,
    ``mean_absorption_coefficient`` + ``room_surface``). With no room data the
    field is treated as free and :math:`K_2 = 0`;
    supplying only one member of a pair raises :class:`ValueError` rather than
    silently falling back to the free-field result.

    The room absorption is frequency dependent (``T``, ``alpha`` and hence ``A``
    vary with the band). Passing ``absorption_area``, ``reverberation_time`` or
    ``mean_absorption_coefficient`` as a per-band array returns ``K2`` per band
    with that shape; scalar inputs return a scalar, unchanged.

    :param surface_area: Measurement surface area ``S``, in square metres.
    :param absorption_area: Equivalent absorption area ``A`` (m^2), scalar or
        per band.
    :param reverberation_time: Sabine ``T`` (s), scalar or per band, with
        ``volume`` (Eq. A.3).
    :param volume: Room volume ``V`` (m^3), with ``reverberation_time``.
    :param mean_absorption_coefficient: ``alpha`` in (0, 1], scalar or per band,
        with ``room_surface`` (Eq. A.7).
    :param room_surface: Room boundary area ``Sv`` (m^2), with ``alpha``.
    :return: ``K2`` in decibels; a scalar for scalar inputs, otherwise an array
        per band.
    """
    if absorption_area is None:
        absorption_area = _room_absorption_area(
            reverberation_time, volume, mean_absorption_coefficient, room_surface
        )
        if absorption_area is None:
            return 0.0  # no room data at all: the field is treated as free
    a = np.asarray(absorption_area, dtype=np.float64)
    if np.any(a <= 0.0):
        msg = "absorption_area must be positive."
        raise ValueError(msg)
    k2 = 10.0 * np.log10(1.0 + 4.0 * surface_area / a)
    return as_float_or_array(k2)


def _hemisphere_position_table(
    grade: Grade, reflecting_planes: int, tones: bool
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Coordinate table and row selection for the hemisphere key positions.

    :param grade: ``'engineering'`` (ISO 3744) or ``'survey'`` (ISO 3746).
    :param reflecting_planes: Number of reflecting planes (1, 2 or 3).
    :param tones: If True use Table B.1, else Table B.2.
    :return: The unscaled coordinate table and the rows to take from it.
    :raises NotImplementedError: For the survey array on three planes.
    """
    if grade == "engineering":  # ISO 3744 clause 8.1.1
        if reflecting_planes == 1:  # positions 1-10, Table B.1 (or B.2 broadband)
            return (_TABLE_B1 if tones else _TABLE_B2), tuple(range(10))
        if reflecting_planes == 2:  # positions 2,3,6,7,9 of Table B.2
            return _TABLE_B2, (1, 2, 5, 6, 8)
        return _TABLE_B3, (0, 1, 2)  # positions 1,2,3 of Table B.3
    # survey, ISO 3746 clause 8.2.1
    if reflecting_planes == 1:  # positions 4,5,6,10 of Table B.1
        return _TABLE_B1, (3, 4, 5, 9)
    if reflecting_planes == 2:  # positions 14,15,18 of Table B.2
        return _TABLE_B2, (13, 14, 17)
    # positions 14,21,22 of Table B.2 (extended array not transcribed)
    msg = (
        "Survey coordinates for a source adjacent to three reflecting "
        "planes require the extended ISO 3746:2010 Table B.2 positions "
        "21 and 22 (see Figure B.4)."
    )
    raise NotImplementedError(msg)


def measurement_positions(
    surface: Surface,
    *,
    radius: float,
    reflecting_planes: int = 1,
    tones: bool = True,
    grade: Grade = "engineering",
) -> np.ndarray:
    """Normative microphone coordinates on the measurement surface.

    For a ``'hemisphere'`` the coordinates come from ISO 3744:2010 Annex B:
    Table B.1 for sources that may emit discrete tones (``tones=True``) and
    Table B.2 for broadband sources. The engineering grade uses the 10 key
    positions for one reflecting plane (5 for two, 3 for three); the survey
    grade uses the reduced arrays of ISO 3746:2010 clause 8.2.1 (positions
    4, 5, 6, 10 for one plane). Coordinates are scaled by ``radius`` and
    returned as an ``(N, 3)`` array of Cartesian ``(x, y, z)`` in metres.

    :param surface: ``'hemisphere'`` (only shape with a coordinate table).
    :param radius: Hemisphere radius ``r``, in metres.
    :param reflecting_planes: Number of reflecting planes (1, 2 or 3).
    :param tones: If True use Table B.1, else Table B.2.
    :param grade: ``'engineering'`` or ``'survey'``.
    :return: ``(N, 3)`` microphone coordinates, in metres.
    """
    if surface != "hemisphere":
        msg = (
            "measurement_positions provides coordinates for 'hemisphere' only; "
            "parallelepiped positions are defined by area subdivision "
            "(ISO 3744:2010 Annex C)."
        )
        raise ValueError(msg)
    if radius <= 0:
        msg = "A positive 'radius' is required for a hemisphere."
        raise ValueError(msg)
    if reflecting_planes not in (1, 2, 3):
        msg = "'reflecting_planes' must be 1, 2 or 3."
        raise ValueError(msg)
    grade = _check_grade(grade)
    table, index = _hemisphere_position_table(grade, reflecting_planes, tones)
    return np.asarray(table[list(index)] * radius, dtype=np.float64)


def _hemisphere_area(radius: float, reflecting_planes: int) -> float:
    """Hemisphere/half/quarter area (ISO 3744:2010 clause 7.2.3)."""
    factor = {1: 2.0, 2: 1.0, 3: 0.5}[reflecting_planes]
    return factor * np.pi * radius**2


def _box_area(
    dimensions: tuple[float, float, float], distance: float, reflecting_planes: int
) -> float:
    """Parallelepiped area (ISO 3744:2010 clause 7.2.4, equations (9)-(11))."""
    l1, l2, l3 = dimensions
    d = distance
    if reflecting_planes == 1:  # Eq. 9
        a, b, c = 0.5 * l1 + d, 0.5 * l2 + d, l3 + d
        return 4.0 * (a * b + b * c + c * a)
    if reflecting_planes == 2:  # Eq. 10 (against a wall)
        a, b, c = 0.5 * l2 + 0.5 * d, 0.5 * l1 + d, l3 + d
        return 2.0 * (2.0 * a * b + b * c + 2.0 * c * a)
    # Eq. 11 (in a corner)
    a, b, c = 0.5 * l1 + 0.5 * d, 0.5 * l2 + 0.5 * d, l3 + d
    return 2.0 * (2.0 * a * b + b * c + c * a)


def _measurement_surface(
    surface: Surface,
    radius: float | None,
    dimensions: tuple[float, float, float] | None,
    distance: float | None,
    reflecting_planes: int,
    grade: Grade,
) -> tuple[float, int]:
    """Measurement surface area ``S`` and the minimum number of positions.

    :param surface: ``'hemisphere'`` (clause 7.2.3) or ``'box'`` (clause 7.2.4).
    :param radius: Hemisphere radius ``r``, in metres.
    :param dimensions: Reference box ``(l1, l2, l3)``, in metres.
    :param distance: Measurement distance ``d``, in metres.
    :param reflecting_planes: Number of reflecting planes (1, 2 or 3).
    :param grade: ``'engineering'`` (ISO 3744) or ``'survey'`` (ISO 3746).
    :return: ``(S, minimum positions)``.
    :raises ValueError: For an unknown surface or missing/non-physical geometry.
    """
    if surface == "hemisphere":
        if radius is None or radius <= 0:
            msg = "A positive 'radius' is required for a hemisphere."
            raise ValueError(msg)
        return (
            _hemisphere_area(radius, reflecting_planes),
            _MIN_HEMI_POSITIONS[grade][reflecting_planes],
        )
    if surface == "box":
        if dimensions is None or distance is None:
            msg = "'dimensions' and 'distance' are required for a box."
            raise ValueError(msg)
        if len(dimensions) != 3 or any(v <= 0 for v in dimensions) or distance <= 0:
            msg = "'dimensions' must be 3 positive values and 'distance' > 0."
            raise ValueError(msg)
        return (
            _box_area(dimensions, distance, reflecting_planes),
            _MIN_BOX_POSITIONS[grade],
        )
    msg = "'surface' must be 'hemisphere' or 'box'."
    raise ValueError(msg)


def _surface_background_correction(
    background_levels: np.ndarray | None,
    levels: np.ndarray,
    mean_level: np.ndarray,
    grade: Grade,
) -> np.ndarray:
    """Per-band background correction ``K1`` of the surface energy mean.

    :param background_levels: ``(NM, NB)`` background levels, or a single
        spectrum ``(NB,)`` / ``(1, NB)`` broadcast to every position, or
        ``None`` for no correction.
    :param levels: ``(NM, NB)`` sound pressure levels with the source running.
    :param mean_level: Surface energy mean of ``levels``, per band.
    :param grade: ``'engineering'`` (ISO 3744) or ``'survey'`` (ISO 3746).
    :return: ``K1`` per band, in decibels (zeros when no background is given).
    :raises ValueError: If the background shape is neither of the two accepted.
    """
    n_positions, n_bands = levels.shape
    if background_levels is None:
        return np.zeros(n_bands, dtype=np.float64)
    bg = np.atleast_2d(np.asarray(background_levels, dtype=np.float64))
    # A single background spectrum (shape (NB,) or (1, NB)) is broadcast to
    # every microphone position; a full (NM, NB) array is used as given.
    if bg.shape == (1, n_bands) and n_positions != 1:
        bg = np.broadcast_to(bg, (n_positions, n_bands))
    if bg.shape != levels.shape:
        msg = (
            "'background_levels' must match 'levels_positions' shape, or be "
            "a single spectrum of shape (NB,) or (1, NB) broadcast to all "
            "positions."
        )
        raise ValueError(msg)
    return background_noise_correction(mean_level, energy_mean(bg, axis=0), grade)


def _surface_environmental_correction(
    area: float,
    n_bands: int,
    grade: Grade,
    room: RoomEnvironment,
) -> np.ndarray:
    """Per-band environmental correction ``K2`` broadcast over the bands.

    :param area: Measurement surface area ``S``, in square metres.
    :param n_bands: Number of frequency bands.
    :param grade: ``'engineering'`` (ISO 3744) or ``'survey'`` (ISO 3746).
    :param room: Room data behind ``K2`` (:class:`RoomEnvironment`).
    :return: ``K2`` per band, in decibels.
    :raises ValueError: If a per-band room input does not match the bands.
    """
    k2_value = environmental_correction(
        area,
        absorption_area=room.absorption_area,
        reverberation_time=room.reverberation_time,
        volume=room.volume,
        mean_absorption_coefficient=room.mean_absorption_coefficient,
        room_surface=room.room_surface,
    )
    k2_arr = np.atleast_1d(np.asarray(k2_value, dtype=np.float64))
    if k2_arr.shape not in ((1,), (n_bands,)):
        msg = (
            "per-band environmental inputs (absorption_area / reverberation_time"
            " / mean_absorption_coefficient) must match the number of bands."
        )
        raise ValueError(msg)
    if np.any(k2_arr > _K2_LIMIT[grade]):
        warnings.warn(
            f"K2 up to {float(np.max(k2_arr)):.1f} dB exceeds the {grade} "
            f"validity limit of {_K2_LIMIT[grade]:g} dB. ISO 3744:2010 clause "
            "4.3.2 states this limit for the A-weighted K2A, so this per-band "
            "check is conservative; the acoustic environment may not qualify "
            "for this method.",
            SoundPowerWarning,
            stacklevel=3,  # report at the caller of sound_power_pressure
        )
    return np.broadcast_to(k2_arr, (n_bands,)).astype(np.float64)


def _a_weighted_total(
    lw: np.ndarray, frequencies: np.ndarray | None, n_bands: int
) -> tuple[np.ndarray | None, float]:
    """Band frequencies and the A-weighted total ``LWA`` (ISO 3744 Annex E).

    :param lw: Band sound power levels, in decibels.
    :param frequencies: Band mid-band frequencies, in hertz, or ``None``.
    :param n_bands: Number of frequency bands.
    :return: ``(frequencies as an array or None, LWA)``.
    :raises ValueError: If the frequencies do not match the number of bands.
    """
    if frequencies is None:
        # A-weighting needs the band centre frequencies; with several bands and
        # none supplied the A-weighted total is undefined (NaN). A single band
        # carries no weighting, so LWA = LW.
        return None, (float(lw[0]) if n_bands == 1 else float("nan"))
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.shape[0] != n_bands:
        msg = "'frequencies' length must match the number of bands."
        raise ValueError(msg)
    return freqs, energy_sum(lw + _a_weighting_corrections(freqs))


def sound_power_pressure(
    levels_positions: np.ndarray,
    surface: Surface,
    *,
    radius: float | None = None,
    dimensions: tuple[float, float, float] | None = None,
    distance: float | None = None,
    reflecting_planes: int = 1,
    background_levels: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    room: RoomEnvironment | None = None,
    grade: Grade = "engineering",
    omc_uncertainty: float = 0.0,
) -> SoundPowerResult:
    r"""Sound power level from surface pressure levels (ISO 3744/3746:2010).

    ``levels_positions`` is an ``(NM, NB)`` array of time-averaged sound
    pressure levels: one row per microphone position, one column per frequency
    band (or a single column for a directly measured A-weighted level). The
    surface-averaged level is corrected for background noise (``K1``, from
    ``background_levels``) and for the test environment (``K2``, from the
    ``room`` absorption data) and combined with the measurement surface area:

    .. math::

       L_W = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}} \sum_i 10^{0.1 L_{pi}} \right]
       - K_1 - K_2 + 10 \log_{10}\frac{S}{S_0}

    The surface area ``S`` is computed from the geometry: ``radius`` for a
    ``'hemisphere'`` (clause 7.2.3) or ``dimensions`` + ``distance`` for a
    ``'box'`` (clause 7.2.4). When ``frequencies`` are given the A-weighted
    sound power level is combined via ISO 3744 Annex E.

    :param levels_positions: ``(NM, NB)`` sound pressure levels, in decibels.
    :param surface: ``'hemisphere'`` or ``'box'``.
    :param radius: Hemisphere radius ``r`` (metres), for ``surface='hemisphere'``.
    :param dimensions: Reference box ``(l1, l2, l3)`` (metres), for ``'box'``.
    :param distance: Measurement distance ``d`` (metres), for ``'box'``.
    :param reflecting_planes: Number of reflecting planes (1, 2 or 3).
    :param background_levels: ``(NM, NB)`` background levels for ``K1``, or a
        single spectrum ``(NB,)`` / ``(1, NB)`` broadcast to every position.
    :param frequencies: Band mid-band frequencies (Hz) for the A-weighted total.
    :param room: Room absorption data behind ``K2`` (:class:`RoomEnvironment`);
        ``None`` is a room with no data at all, i.e. a free field
        (:math:`K_2 = 0`).
    :param grade: ``'engineering'`` (ISO 3744) or ``'survey'`` (ISO 3746).
    :param omc_uncertainty: ``sigma_omc`` (dB), operating/mounting instability.
    :return: :class:`SoundPowerResult`.
    """
    grade = _check_grade(grade)
    levels = np.atleast_2d(np.asarray(levels_positions, dtype=np.float64))
    if levels.ndim != 2:
        msg = "'levels_positions' must be a 2D (positions, bands) array."
        raise ValueError(msg)
    n_positions = levels.shape[0]

    # --- measurement surface area -----------------------------------------
    if reflecting_planes not in (1, 2, 3):
        msg = "'reflecting_planes' must be 1, 2 or 3."
        raise ValueError(msg)
    area, min_positions = _measurement_surface(
        surface, radius, dimensions, distance, reflecting_planes, grade
    )

    if n_positions < min_positions:
        msg = (
            f"{grade} {surface} with {reflecting_planes} reflecting plane(s) "
            f"requires at least {min_positions} microphone positions, got "
            f"{n_positions} (ISO 3744/3746:2010 clause 8)."
        )
        raise ValueError(msg)

    # --- energy average and background correction K1 ----------------------
    mean_level = energy_mean(levels, axis=0)
    n_bands = mean_level.shape[0]
    k1 = _surface_background_correction(background_levels, levels, mean_level, grade)

    # --- environmental correction K2 (scalar or per band) -----------------
    k2 = _surface_environmental_correction(
        area, n_bands, grade, RoomEnvironment() if room is None else room
    )

    # --- surface SPL, sound power level and A-weighted total ---------------
    surface_spl = mean_level - k1 - k2
    lw = surface_spl + 10.0 * np.log10(area / _S0)
    freqs, lwa = _a_weighted_total(lw, frequencies, n_bands)

    # --- apparent directivity index per position AND band (Eq. 7) ---------
    # ISO 3744:2010 clause 8.4 requires the apparent directivity indices to be
    # calculated on the actual measurement surface (to validate the number of
    # measurement positions), per frequency band, so DIi* is a (NM, NB) array.
    # Per Eq. 7
    # DIi*(k) = Lpi(k) - (Lp'(k) - K1(k)): BOTH the per-position level Lpi(k)
    # and the surface energy mean Lp'(k) carry the same per-band background
    # correction K1(k), which cancels in the difference (3.24 DI definition).
    # The uniform per-band K1 therefore drops out and DIi*(k) reduces to the raw
    # per-band level minus the raw per-band surface mean (no residual +K1 bias).
    directivity = np.asarray(levels - mean_level[np.newaxis, :], dtype=np.float64)

    uncertainty = 2.0 * float(np.hypot(_SIGMA_R0[grade], omc_uncertainty))

    return SoundPowerResult(
        frequencies=freqs,
        sound_power_level=np.asarray(lw, dtype=np.float64),
        surface_pressure_level=np.asarray(surface_spl, dtype=np.float64),
        mean_pressure_level=mean_level,
        background_correction=k1,
        environmental_correction=k2,
        directivity_index=directivity,
        surface_area=float(area),
        sound_power_level_a=lwa,
        uncertainty=uncertainty,
        grade=grade,
    )
