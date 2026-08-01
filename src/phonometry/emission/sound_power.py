#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Sound power level of a noise source from sound pressure measurements over an
enveloping measurement surface: ISO 3744:2010 (engineering, accuracy grade 2)
and ISO 3746:2010 (survey, accuracy grade 3).

The source stands on one (or more) reflecting plane(s). Sound pressure levels
are measured at an array of microphone positions on a hypothetical surface of
area ``S`` enveloping the source (a hemisphere or a right parallelepiped). The
sound power level follows from the energy-averaged pressure level, the
background correction :math:`K_1`, the environmental correction :math:`K_2`
and the surface area (ISO 3744:2010 clause 8.2, equations (12), (16)-(18)):

.. math::

   \overline{L_p} = 10 \log_{10}\!\left[ \frac{1}{N_M}
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

   L_{WA} = 10 \log_{10}\!\left[ \sum_k 10^{0.1 (L_{Wk} + C_k)} \right]
   \tag{Eq. E.1}

ISO 3746:2010 shares the surfaces, the energy average and the LW/K1/K2 forms
but is coarser: fewer microphone positions (clause 8.2.1), a background
criterion of 3 dB instead of 6 dB (clause 8.4.1) and validity up to
:math:`K_{2A} \le 7` dB instead of 4 dB (clause 4.3).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .._internal.levels_math import energy_mean, energy_sum, weighted_energy_mean
from .._internal.types import as_float_or_array
from .._internal.warnings import PhonometryWarning, _warn_renamed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata
    from .declaration import DeclarationForm, NoiseEmissionDeclaration

_S0 = 1.0  #: Reference area, in square metres (ISO 3744, 8.2.5).

Grade = Literal["engineering", "survey"]
Surface = Literal["hemisphere", "box"]


class SoundPowerWarning(PhonometryWarning):
    """Non-fatal qualification issue in any of the sound-power methods.

    Emitted for ISO 3744/3746 background margin below the criterion and for
    ``K2`` beyond the method's validity limit (8.2.3, 4.3.2); for ISO 3741
    reverberation-room qualification (room volume vs Table 1, mean absorption)
    and microphone/source-position sampling; and for ISO 9614-2 negative total
    partial power and unmet field-indicator criteria. Where a lower criterion
    is only just met the returned levels represent upper bounds and must be
    reported as such."""


# --- ISO 3744:2010 Annex B, normative microphone coordinates (x/r,y/r,z/r) ---
#: Table B.1 - preferred positions for all sources (including tones).
_TABLE_B1: np.ndarray = np.array(
    [
        [0.16, -0.96, 0.22], [0.78, -0.60, 0.20], [0.78, 0.55, 0.31],
        [0.16, 0.90, 0.41], [-0.83, 0.32, 0.45], [-0.83, -0.40, 0.38],
        [-0.26, -0.65, 0.71], [0.74, -0.07, 0.67], [-0.26, 0.50, 0.83],
        [0.10, -0.10, 0.99], [0.91, -0.34, 0.22], [0.91, 0.38, 0.20],
        [-0.09, 0.95, 0.31], [-0.70, 0.59, 0.41], [-0.69, -0.56, 0.45],
        [-0.07, -0.92, 0.38], [0.43, -0.55, 0.71], [0.43, 0.61, 0.67],
        [-0.56, 0.02, 0.83], [0.14, 0.04, 0.99],
    ]
)
#: Table B.2 - positions for a broadband (tone-free) source.
_TABLE_B2: np.ndarray = np.array(
    [
        [-0.99, 0.0, 0.15], [0.50, -0.86, 0.15], [0.50, 0.86, 0.15],
        [-0.45, 0.77, 0.45], [-0.45, -0.77, 0.45], [0.89, 0.0, 0.45],
        [0.33, 0.57, 0.75], [-0.66, 0.0, 0.75], [0.33, -0.57, 0.75],
        [0.0, 0.0, 1.00], [0.99, 0.0, 0.15], [-0.50, 0.86, 0.15],
        [-0.50, -0.86, 0.15], [0.45, -0.77, 0.45], [0.45, 0.77, 0.45],
        [-0.89, 0.0, 0.45], [-0.33, -0.57, 0.75], [0.66, 0.0, 0.75],
        [-0.33, 0.57, 0.75], [0.0, 0.0, 1.00],
    ]
)
#: Table B.3 - source adjacent to three reflecting planes.
_TABLE_B3: np.ndarray = np.array(
    [
        [0.86, -0.50, 0.15], [0.45, -0.77, 0.45], [0.47, -0.47, 0.75],
        [0.50, -0.86, 0.15], [0.77, -0.45, 0.45], [0.47, -0.47, 0.75],
    ]
)

# --- ISO 3744:2010 Annex E, A-weighting band corrections Ck (dB) ------------
#: Table E.1 - one-third-octave mid-band frequencies.
_CK_THIRD: dict[int, float] = {
    50: -30.2, 63: -26.2, 80: -22.5, 100: -19.1, 125: -16.1, 160: -13.4,
    200: -10.9, 250: -8.6, 315: -6.6, 400: -4.8, 500: -3.2, 630: -1.9,
    800: -0.8, 1000: 0.0, 1250: 0.6, 1600: 1.0, 2000: 1.2, 2500: 1.3,
    3150: 1.2, 4000: 1.0, 5000: 0.5, 6300: -0.1, 8000: -1.1, 10000: -2.5,
}
#: Table E.2 - octave mid-band frequencies (subset of E.1 values).
_CK_OCTAVE: dict[int, float] = {
    63: -26.2, 125: -16.1, 250: -8.6, 500: -3.2,
    1000: 0.0, 2000: 1.2, 4000: 1.0, 8000: -1.1,
}

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
    :math:`U = 2\sqrt{\sigma_{R0}^2 + \sigma_{omc}^2}` (95 %, ISO 3744
    clause 9.5)."""

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

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
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
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
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
        :math:`L_{WAd} = L_{WA} + K_{WA}` (ISO 4871 clause 3.15).

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
            raise ValueError(
                "declare() needs a finite A-weighted sound power level; supply "
                "'frequencies' to sound_power_pressure(...) so LWA is defined."
            )
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


def environmental_correction(
    surface_area: float,
    *,
    absorption_area: float | np.ndarray | None = None,
    reverberation_time: float | np.ndarray | None = None,
    volume: float | None = None,
    mean_absorption_coefficient: float | np.ndarray | None = None,
    room_surface: float | None = None,
    room_volume: float | str | None = "deprecated",
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
    :param room_volume: Deprecated alias of ``volume`` (remove in 4.0).
    :return: ``K2`` in decibels; a scalar for scalar inputs, otherwise an array
        per band.
    """
    # An explicit None matches the old default and stays silent; only a real
    # value through the deprecated alias warns.
    if not isinstance(room_volume, str) and room_volume is not None:
        _warn_renamed(
            "the 'room_volume' keyword of environmental_correction()",
            "'volume'",
        )
        if volume is not None:
            raise ValueError(
                "environmental_correction() got both 'volume' and its "
                "deprecated alias 'room_volume'; pass only 'volume'."
            )
        volume = room_volume
    if absorption_area is None:
        # A half-specified room pair must never be read as free field: naming
        # only one member of a pair is a mistake, not a K2 = 0 request.
        if (reverberation_time is None) != (volume is None):
            missing = "volume" if volume is None else "reverberation_time"
            raise ValueError(
                "reverberation_time and volume must be given together "
                f"(Eq. A.3); '{missing}' is missing."
            )
        if (mean_absorption_coefficient is None) != (room_surface is None):
            missing = (
                "room_surface"
                if room_surface is None
                else "mean_absorption_coefficient"
            )
            raise ValueError(
                "mean_absorption_coefficient and room_surface must be given "
                f"together (Eq. A.7); '{missing}' is missing."
            )
        if reverberation_time is not None and volume is not None:
            t = np.asarray(reverberation_time, dtype=np.float64)
            if volume <= 0 or np.any(t <= 0.0):
                raise ValueError("reverberation_time and volume must be > 0.")
            absorption_area = 0.16 * volume / t
        elif mean_absorption_coefficient is not None and room_surface is not None:
            alpha = np.asarray(mean_absorption_coefficient, dtype=np.float64)
            if room_surface <= 0 or np.any(alpha <= 0.0) or np.any(alpha > 1.0):
                raise ValueError(
                    "mean_absorption_coefficient must be in (0, 1] and "
                    "room_surface > 0."
                )
            absorption_area = alpha * room_surface
        else:
            return 0.0
    a = np.asarray(absorption_area, dtype=np.float64)
    if np.any(a <= 0.0):
        raise ValueError("absorption_area must be positive.")
    k2 = 10.0 * np.log10(1.0 + 4.0 * surface_area / a)
    return as_float_or_array(k2)


def measurement_positions(
    surface: Surface,
    *,
    radius: float | None = None,
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
        raise ValueError(
            "measurement_positions provides coordinates for 'hemisphere' only; "
            "parallelepiped positions are defined by area subdivision "
            "(ISO 3744:2010 Annex C)."
        )
    if radius is None or radius <= 0:
        raise ValueError("A positive 'radius' is required for a hemisphere.")
    if reflecting_planes not in (1, 2, 3):
        raise ValueError("'reflecting_planes' must be 1, 2 or 3.")
    grade = _check_grade(grade)
    if grade == "engineering":  # ISO 3744 clause 8.1.1
        if reflecting_planes == 1:  # positions 1-10, Table B.1 (or B.2 broadband)
            table, index = (_TABLE_B1 if tones else _TABLE_B2), tuple(range(10))
        elif reflecting_planes == 2:  # positions 2,3,6,7,9 of Table B.2
            table, index = _TABLE_B2, (1, 2, 5, 6, 8)
        else:  # positions 1,2,3 of Table B.3
            table, index = _TABLE_B3, (0, 1, 2)
    else:  # survey, ISO 3746 clause 8.2.1
        if reflecting_planes == 1:  # positions 4,5,6,10 of Table B.1
            table, index = _TABLE_B1, (3, 4, 5, 9)
        elif reflecting_planes == 2:  # positions 14,15,18 of Table B.2
            table, index = _TABLE_B2, (13, 14, 17)
        else:  # positions 14,21,22 of Table B.2 (extended array not transcribed)
            raise NotImplementedError(
                "Survey coordinates for a source adjacent to three reflecting "
                "planes require the extended ISO 3746:2010 Table B.2 positions "
                "21 and 22 (see Figure B.4)."
            )
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


def _a_weighting_corrections(frequencies: np.ndarray) -> np.ndarray:
    """A-weighting band corrections Ck for the given nominal mid-band
    frequencies (ISO 3744:2010 Annex E, Tables E.1/E.2). The octave-band
    values (Table E.2) coincide with the one-third-octave values (Table E.1)
    at the shared mid-band frequencies, so a single lookup serves both."""
    nominal = [round(f) for f in frequencies]
    # Use the octave table (E.2) when every band is an octave mid-band centre,
    # otherwise the one-third-octave table (E.1); the two agree where they
    # overlap, so the result is identical either way.
    table = _CK_OCTAVE if set(nominal) <= set(_CK_OCTAVE) else _CK_THIRD
    ck = []
    for freq in nominal:
        if freq not in table:
            raise ValueError(
                f"No ISO 3744 Annex E A-weighting value for {freq} Hz; "
                "expected a nominal octave or one-third-octave mid-band "
                "frequency (50 Hz to 10 kHz)."
            )
        ck.append(table[freq])
    return np.asarray(ck, dtype=np.float64)


def _check_grade(grade: str) -> Grade:
    if grade not in ("engineering", "survey"):
        raise ValueError("'grade' must be 'engineering' or 'survey'.")
    return cast(Grade, grade)


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
    absorption_area: float | None = None,
    reverberation_time: float | None = None,
    volume: float | None = None,
    mean_absorption_coefficient: float | None = None,
    room_surface: float | None = None,
    grade: Grade = "engineering",
    omc_uncertainty: float = 0.0,
    room_volume: float | str | None = "deprecated",
) -> SoundPowerResult:
    r"""Sound power level from surface pressure levels (ISO 3744/3746:2010).

    ``levels_positions`` is an ``(NM, NB)`` array of time-averaged sound
    pressure levels: one row per microphone position, one column per frequency
    band (or a single column for a directly measured A-weighted level). The
    surface-averaged level is corrected for background noise (``K1``, from
    ``background_levels``) and for the test environment (``K2``, from the room
    absorption data) and combined with the measurement surface area:

    .. math::

       L_W = 10 \log_{10}\!\left[ \frac{1}{N_M} \sum_i 10^{0.1 L_{pi}} \right]
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
    :param absorption_area: Equivalent absorption area ``A`` (m^2) for ``K2``.
    :param reverberation_time: Sabine ``T`` (s), with ``volume``, for ``K2``.
    :param volume: Room volume ``V`` (m^3), with ``reverberation_time``.
    :param mean_absorption_coefficient: ``alpha``, with ``room_surface``, for ``K2``.
    :param room_surface: Room boundary area ``Sv`` (m^2), with ``alpha``.
    :param grade: ``'engineering'`` (ISO 3744) or ``'survey'`` (ISO 3746).
    :param omc_uncertainty: ``sigma_omc`` (dB), operating/mounting instability.
    :param room_volume: Deprecated alias of ``volume`` (remove in 4.0).
    :return: :class:`SoundPowerResult`.
    """
    if not isinstance(room_volume, str) and room_volume is not None:
        _warn_renamed(
            "the 'room_volume' keyword of sound_power_pressure()", "'volume'"
        )
        if volume is not None:
            raise ValueError(
                "sound_power_pressure() got both 'volume' and its deprecated "
                "alias 'room_volume'; pass only 'volume'."
            )
        volume = room_volume
    grade = _check_grade(grade)
    levels = np.atleast_2d(np.asarray(levels_positions, dtype=np.float64))
    if levels.ndim != 2:
        raise ValueError("'levels_positions' must be a 2D (positions, bands) array.")
    n_positions = levels.shape[0]

    # --- measurement surface area -----------------------------------------
    if reflecting_planes not in (1, 2, 3):
        raise ValueError("'reflecting_planes' must be 1, 2 or 3.")
    if surface == "hemisphere":
        if radius is None or radius <= 0:
            raise ValueError("A positive 'radius' is required for a hemisphere.")
        area = _hemisphere_area(radius, reflecting_planes)
        min_positions = _MIN_HEMI_POSITIONS[grade][reflecting_planes]
    elif surface == "box":
        if dimensions is None or distance is None:
            raise ValueError("'dimensions' and 'distance' are required for a box.")
        if len(dimensions) != 3 or any(v <= 0 for v in dimensions) or distance <= 0:
            raise ValueError("'dimensions' must be 3 positive values and 'distance' > 0.")
        area = _box_area(dimensions, distance, reflecting_planes)
        min_positions = _MIN_BOX_POSITIONS[grade]
    else:
        raise ValueError("'surface' must be 'hemisphere' or 'box'.")

    if n_positions < min_positions:
        raise ValueError(
            f"{grade} {surface} with {reflecting_planes} reflecting plane(s) "
            f"requires at least {min_positions} microphone positions, got "
            f"{n_positions} (ISO 3744/3746:2010 clause 8)."
        )

    # --- energy average and background correction K1 ----------------------
    mean_level = energy_mean(levels, axis=0)
    n_bands = mean_level.shape[0]
    if background_levels is not None:
        bg = np.atleast_2d(np.asarray(background_levels, dtype=np.float64))
        # A single background spectrum (shape (NB,) or (1, NB)) is broadcast to
        # every microphone position; a full (NM, NB) array is used as given.
        if bg.shape == (1, n_bands) and n_positions != 1:
            bg = np.broadcast_to(bg, (n_positions, n_bands))
        if bg.shape != levels.shape:
            raise ValueError(
                "'background_levels' must match 'levels_positions' shape, or be "
                "a single spectrum of shape (NB,) or (1, NB) broadcast to all "
                "positions."
            )
        k1 = background_noise_correction(mean_level, energy_mean(bg, axis=0), grade)
    else:
        k1 = np.zeros(n_bands, dtype=np.float64)

    # --- environmental correction K2 (scalar or per band) -----------------
    k2_value = environmental_correction(
        area,
        absorption_area=absorption_area,
        reverberation_time=reverberation_time,
        volume=volume,
        mean_absorption_coefficient=mean_absorption_coefficient,
        room_surface=room_surface,
    )
    k2_arr = np.atleast_1d(np.asarray(k2_value, dtype=np.float64))
    if k2_arr.shape not in ((1,), (n_bands,)):
        raise ValueError(
            "per-band environmental inputs (absorption_area / reverberation_time"
            " / mean_absorption_coefficient) must match the number of bands."
        )
    if np.any(k2_arr > _K2_LIMIT[grade]):
        warnings.warn(
            f"K2 up to {float(np.max(k2_arr)):.1f} dB exceeds the {grade} "
            f"validity limit of {_K2_LIMIT[grade]:g} dB. ISO 3744:2010 clause "
            "4.3.2 states this limit for the A-weighted K2A, so this per-band "
            "check is conservative; the acoustic environment may not qualify "
            "for this method.",
            SoundPowerWarning,
            stacklevel=2,
        )
    k2 = np.broadcast_to(k2_arr, (n_bands,)).astype(np.float64)

    # --- surface SPL, sound power level and A-weighted total ---------------
    surface_spl = mean_level - k1 - k2
    lw = surface_spl + 10.0 * np.log10(area / _S0)

    if frequencies is not None:
        freqs = np.asarray(frequencies, dtype=np.float64)
        if freqs.shape[0] != n_bands:
            raise ValueError("'frequencies' length must match the number of bands.")
        ck = _a_weighting_corrections(freqs)
        lwa = energy_sum(lw + ck)
    else:
        freqs = None
        # A-weighting needs the band centre frequencies; with several bands and
        # none supplied the A-weighted total is undefined (NaN). A single band
        # carries no weighting, so LWA = LW.
        lwa = float(lw[0]) if n_bands == 1 else float("nan")

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


# ===========================================================================
# ISO 3745:2012 - sound power in anechoic / hemi-anechoic rooms (precision,
# grade 1). Precision sibling of ISO 3744 (engineering) / ISO 3746 (survey).
# The surface-averaged pressure -> LW path is shared, but the room is a
# qualified (hemi-)free field: no K2 environmental term, a per-position and
# frequency-dependent background correction K1i (Eq. 11), fixed 40-position
# equal-area arrays (Annex D sphere, Annex E hemisphere), full-sphere area
# S1 = 4*pi*r^2, and three meteorological corrections C1/C2/C3 (Eq. 14).
# ===========================================================================

PrecisionSurface = Literal["sphere", "hemisphere"]
PrecisionArray = Literal["general", "broadband"]
PrecisionRoom = Literal["anechoic", "hemi-anechoic"]

# --- ISO 3745:2012 Annex D/E, normative microphone coordinates (x/r,y/r,z/r) -
# Digit-exact, image-verified transcriptions. Positions 1-20 are the primary
# array; positions 21-40 (the mirror set) are added when the band-SPL spread
# exceeds NM/2 (clause 9.3.2/9.3.3). Every position carries an equal surface
# area (S/40). z points up from the horizontal plane z = 0.

#: Table D.1 - sphere, anechoic room (Annex D). 40 positions.
_TABLE_D1: np.ndarray = np.array(
    [
        [-0.999, 0.0, 0.050], [0.494, -0.856, 0.150], [0.484, 0.839, 0.250],
        [-0.468, 0.811, 0.350], [-0.447, -0.773, 0.450], [0.835, 0.0, 0.550],
        [0.380, 0.658, 0.650], [-0.661, 0.0, 0.750], [0.263, -0.456, 0.850],
        [0.312, 0.0, 0.950], [0.999, 0.0, -0.050], [-0.494, 0.856, -0.150],
        [-0.484, -0.839, -0.250], [0.468, -0.811, -0.350], [0.447, 0.773, -0.450],
        [-0.835, 0.0, -0.550], [-0.380, -0.658, -0.650], [0.661, 0.0, -0.750],
        [-0.263, 0.456, -0.850], [-0.312, 0.0, -0.950], [0.999, 0.0, 0.050],
        [-0.494, -0.856, 0.150], [-0.484, 0.839, 0.250], [0.468, 0.811, 0.350],
        [0.447, -0.773, 0.450], [-0.835, 0.0, 0.550], [-0.380, 0.658, 0.650],
        [0.661, 0.0, 0.750], [-0.263, -0.456, 0.850], [-0.312, 0.0, 0.950],
        [-0.999, 0.0, -0.050], [0.494, 0.856, -0.150], [0.484, -0.839, -0.250],
        [-0.468, -0.811, -0.350], [-0.447, 0.773, -0.450], [0.835, 0.0, -0.550],
        [0.380, -0.658, -0.650], [-0.661, 0.0, -0.750], [0.263, 0.456, -0.850],
        [0.312, 0.0, -0.950],
    ]
)
#: Table E.1 - hemisphere, general case (Annex E). 40 positions. z/r at pos
#: 7/27 is 0.320 (not 0.325), verified against the source image.
_TABLE_E1: np.ndarray = np.array(
    [
        [-1.000, 0.000, 0.025], [0.499, -0.864, 0.075], [0.496, 0.859, 0.125],
        [-0.492, 0.853, 0.175], [-0.487, -0.844, 0.225], [0.961, 0.000, 0.275],
        [0.000, 0.947, 0.320], [-0.803, -0.464, 0.375], [0.784, -0.453, 0.425],
        [0.762, 0.440, 0.475], [-0.737, 0.426, 0.525], [0.000, -0.818, 0.575],
        [0.781, 0.000, 0.625], [-0.369, 0.639, 0.675], [-0.344, -0.596, 0.725],
        [0.316, -0.547, 0.775], [0.283, 0.489, 0.825], [-0.484, 0.000, 0.875],
        [0.000, -0.380, 0.925], [0.192, 0.111, 0.975], [1.000, 0.000, 0.025],
        [-0.499, 0.864, 0.075], [-0.496, -0.859, 0.125], [0.492, -0.853, 0.175],
        [0.487, 0.844, 0.225], [-0.961, 0.000, 0.275], [0.000, -0.947, 0.320],
        [0.803, 0.464, 0.375], [-0.784, 0.453, 0.425], [-0.762, -0.440, 0.475],
        [0.737, -0.426, 0.525], [0.000, 0.818, 0.575], [-0.781, 0.000, 0.625],
        [0.369, -0.639, 0.675], [0.344, 0.596, 0.725], [-0.316, 0.547, 0.775],
        [-0.283, -0.489, 0.825], [0.484, 0.000, 0.875], [0.000, 0.380, 0.925],
        [-0.192, -0.111, 0.975],
    ]
)
#: Table E.2 - hemisphere, broadband omnidirectional source (Annex E). 40
#: positions. Pos 19 x/r is -0.380 (a normal negative), verified.
_TABLE_E2: np.ndarray = np.array(
    [
        [-1.000, 0.000, 0.025], [0.499, -0.864, 0.075], [0.496, 0.859, 0.125],
        [-0.492, 0.853, 0.175], [-0.487, -0.844, 0.225], [0.961, 0.000, 0.275],
        [0.474, 0.820, 0.325], [-0.927, 0.000, 0.375], [0.453, -0.784, 0.425],
        [0.880, 0.000, 0.475], [-0.426, 0.737, 0.525], [-0.409, -0.709, 0.575],
        [0.390, -0.676, 0.625], [0.369, 0.639, 0.675], [-0.689, 0.000, 0.725],
        [-0.316, -0.547, 0.775], [0.565, 0.000, 0.825], [-0.242, 0.419, 0.875],
        [-0.380, 0.000, 0.925], [0.111, -0.192, 0.975], [1.000, 0.000, 0.025],
        [-0.499, 0.864, 0.075], [-0.496, -0.859, 0.125], [0.492, -0.853, 0.175],
        [0.487, 0.844, 0.225], [-0.961, 0.000, 0.275], [-0.474, -0.820, 0.325],
        [0.927, 0.000, 0.375], [-0.453, 0.784, 0.425], [-0.880, 0.000, 0.475],
        [0.426, -0.737, 0.525], [0.409, 0.709, 0.575], [-0.390, 0.676, 0.625],
        [-0.369, -0.639, 0.675], [0.689, 0.000, 0.725], [0.316, 0.547, 0.775],
        [-0.565, 0.000, 0.825], [0.242, -0.419, 0.875], [0.380, 0.000, 0.925],
        [-0.111, 0.192, 0.975],
    ]
)

#: Tolerance on the unit-vector self-check of the coordinate tables, in units
#: of the (dimensionless) coordinate norm. The tabulated coordinates are given
#: to three decimals, so the exact-unit-sphere residual is at most ~1.4e-3.
_UNIT_NORM_TOL = 2.0e-3

#: Meteorological reference constants (ISO 3745:2012 Eq. 14 block, clause 4).
_PS0_KPA = 101.325  #: Reference static pressure, in kilopascals.
_THETA0_K = 314.0  #: C1 reference temperature theta0, in kelvin.
_THETA1_K = 296.0  #: C2 reference temperature theta1, in kelvin.

#: Background-noise correction floor criteria, in dB (clause 9.4.2). The lower
#: criterion is 10 dB for one-third-octave mid-bands 250 Hz to 5000 Hz and
#: 6 dB for bands <= 200 Hz and >= 6300 Hz; the upper criterion is 15 dB.
_K1_UPPER_3745 = 15.0
_K1_LOW_MID = 10.0  #: 250-5000 Hz
_K1_LOW_EDGE = 6.0  #: <= 200 Hz and >= 6300 Hz

#: A-weighted reproducibility standard deviation sigma_R0, in dB (Tables 2/3).
_SIGMA_R0_3745_A = 0.5


@dataclass(frozen=True)
class MeteorologicalCorrection:
    r"""Meteorological corrections C1, C2, C3 (ISO 3745:2012 Eq. 14 block).

    ``c1`` is the reference-quantity (impedance) correction and ``c2`` the
    radiation-impedance correction, both scalars in decibels; ``c3`` is the
    air-absorption correction (scalar, or per band when the attenuation
    coefficient ``a(f)`` is supplied per band). All three are added to
    :math:`\overline{L_p} + 10 \log_{10}(S/S_0)` to obtain ``LW``.
    """

    c1: float
    c2: float
    c3: float | np.ndarray


@dataclass(frozen=True)
class PrecisionSoundPowerResult:
    r"""Result of an ISO 3745:2012 (precision) sound power determination.

    ``sound_power_level`` is the per-band
    :math:`L_W = \overline{L_p} + 10\log_{10}(S/S_0) + C_1 + C_2 + C_3`
    (Eq. 14/15). ``surface_pressure_level`` is the surface time-
    averaged level ``Lp_bar`` after the per-position background correction
    (Eq. 12/13); ``mean_pressure_level`` the same energy average of the raw
    (uncorrected) position levels. ``background_correction`` is the
    per-position per-band ``K1i`` (Eq. 11), shape ``(NM, NB)``.
    ``c1``/``c2``/``c3`` are the
    meteorological corrections (Eq. 14). ``directivity_index`` is
    :math:`DI_i = L_{pi} - \overline{L_p}`
    per position and band (Eq. 21); ``non_uniformity_index`` the
    per-band ``VIr`` sample standard deviation about the arithmetic mean
    (Eq. 22). ``uncertainty`` is the A-weighted expanded uncertainty
    :math:`U = k\sqrt{\sigma_{R0}^2 + \sigma_{omc}^2}`
    (Eq. 24/25) and ``uncertainty_bands`` the
    per-band value (``NaN`` without ``frequencies``).
    ``sound_power_level_a`` is the A-weighted total ``LWA`` (Eq. C.1)."""

    frequencies: np.ndarray | None
    sound_power_level: np.ndarray
    surface_pressure_level: np.ndarray
    mean_pressure_level: np.ndarray
    background_correction: np.ndarray
    c1: float
    c2: float
    c3: np.ndarray
    directivity_index: np.ndarray
    non_uniformity_index: np.ndarray
    surface_area: float
    surface: str
    sound_power_level_a: float
    uncertainty: float
    uncertainty_bands: np.ndarray
    coverage_factor: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the precision ``LW`` spectrum with the A-weighted total.

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
        """Render an ISO 3745:2012 precision sound-power fiche to a PDF.

        Writes the same one-page sound-power test sheet as
        :meth:`SoundPowerResult.report`, with the standard-basis line naming the
        precision method in an anechoic or hemi-anechoic room (ISO 3745:2012,
        accuracy grade 1) and the measurement-basis strip stating the applied
        meteorological corrections ``C1``/``C2``/``C3`` instead of the ISO 3744
        ``K1``/``K2``. The per-band table shows the surface time-averaged level
        ``Lp`` and the band sound-power level ``LW``; ``verbose`` adds the
        energy-averaged level ``Lp'``.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header and footer identity and, via ``requirement``, a declared
            A-weighted sound-power limit (lower is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds the
            energy-averaged level ``Lp'``.
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
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso3744 import render_sound_power_report

        return render_sound_power_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _precision_table(surface: PrecisionSurface, array: PrecisionArray) -> np.ndarray:
    """Select the normative coordinate table (Annex D/E) and self-check it."""
    if surface == "sphere":
        table = _TABLE_D1
    elif surface == "hemisphere":
        table = _TABLE_E1 if array == "general" else _TABLE_E2
    else:  # pragma: no cover - guarded by the public callers
        raise ValueError("'surface' must be 'sphere' or 'hemisphere'.")
    norms = np.linalg.norm(table, axis=1)
    if np.any(np.abs(norms - 1.0) > _UNIT_NORM_TOL):
        raise ValueError(
            "Microphone coordinate table is not a set of unit vectors within "
            f"{_UNIT_NORM_TOL:g}; a transcription error is present."
        )
    return table


def precision_positions(
    surface: PrecisionSurface,
    *,
    radius: float | None = None,
    array: PrecisionArray = "general",
    count: int = 40,
) -> np.ndarray:
    """Normative ISO 3745:2012 microphone coordinates, scaled by ``radius``.

    For a ``'sphere'`` (anechoic room) the coordinates come from Annex D
    Table D.1; for a ``'hemisphere'`` (hemi-anechoic room) from Annex E
    Table E.1 (``array='general'``) or Table E.2 (``array='broadband'``, an
    omnidirectional broadband source). Positions 1-20 are the primary array;
    the full 40 add the mirror set (positions 21-40), used when the band-SPL
    spread exceeds NM/2 (clause 9.3). Each row is a unit vector (self-checked)
    scaled to metres by ``radius``.

    :param surface: ``'sphere'`` or ``'hemisphere'``.
    :param radius: Measurement radius ``r``, in metres.
    :param array: ``'general'`` (Table E.1) or ``'broadband'`` (Table E.2);
        ignored for a sphere (only Table D.1 exists).
    :param count: ``20`` (primary array) or ``40`` (full array).
    :return: ``(count, 3)`` microphone coordinates, in metres.
    """
    if surface not in ("sphere", "hemisphere"):
        raise ValueError("'surface' must be 'sphere' or 'hemisphere'.")
    if array not in ("general", "broadband"):
        raise ValueError("'array' must be 'general' or 'broadband'.")
    if radius is None or radius <= 0:
        raise ValueError("A positive 'radius' is required.")
    if count not in (20, 40):
        raise ValueError("'count' must be 20 (primary array) or 40 (full array).")
    table = _precision_table(surface, array)
    return np.asarray(table[:count] * radius, dtype=np.float64)


def _k1_lower_criterion(frequencies: np.ndarray) -> np.ndarray:
    """Frequency-dependent lower K1 criterion, in dB (clause 9.4.2)."""
    freqs = np.asarray(frequencies, dtype=np.float64)
    return np.where(
        (freqs >= 250.0) & (freqs <= 5000.0), _K1_LOW_MID, _K1_LOW_EDGE
    ).astype(np.float64)


def precision_background_correction(
    source_levels: np.ndarray,
    background_levels: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    r"""Per-position background correction ``K1i`` (ISO 3745:2012 Eq. 11).

    :math:`K_{1i} = -10 \log_{10}\left( 1 - 10^{-0.1 \Delta L_{pi}} \right)` with
    :math:`\Delta L_{pi} = L'_{pi(\mathrm{ST})} - L_{pi(\mathrm{B})}`
    evaluated at each microphone position ``i`` and band. Above the upper
    criterion (:math:`\Delta L_{pi} \ge 15` dB) the background is negligible
    and :math:`K_{1i} = 0`. The lower criterion is frequency dependent:
    ``10 dB`` for one-third-octave mid-bands 250 Hz to 5000 Hz and ``6 dB``
    for bands :math:`\le 200` Hz and :math:`\ge 6300` Hz. Below it,
    :math:`K_{1i}` is clamped to its value
    at the criterion (``0.46 dB`` and ``1.26 dB`` respectively), a
    :class:`SoundPowerWarning` is emitted and those band results are upper
    bounds (clause 9.4.2).

    :param source_levels: ``L'pi(ST)`` per position and band, in decibels;
        shape ``(NM, NB)`` (or ``(NB,)`` for one position).
    :param background_levels: ``Lpi(B)`` in the same shape (or a single
        spectrum broadcast to every position).
    :param frequencies: ``(NB,)`` nominal mid-band frequencies (Hz), selecting
        the per-band lower criterion.
    :return: ``K1i`` per position and band, in decibels, matching the broadcast
        shape of the inputs.
    """
    src = np.asarray(source_levels, dtype=np.float64)
    bg = np.asarray(background_levels, dtype=np.float64)
    freqs = np.asarray(frequencies, dtype=np.float64)
    if src.shape[-1] != freqs.shape[0] or bg.shape[-1] != freqs.shape[0]:
        raise ValueError(
            "The last axis of 'source_levels'/'background_levels' must match "
            "the number of 'frequencies'."
        )
    low = _k1_lower_criterion(freqs)  # (NB,)
    delta = src - bg
    clamped = np.maximum(delta, low)
    k1 = -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * clamped))
    k1 = np.where(delta >= _K1_UPPER_3745, 0.0, k1)
    if np.any(delta < low):
        warnings.warn(
            "Background margin below the frequency-dependent criterion (6 dB "
            "edge bands / 10 dB mid bands) in one or more positions; K1 clamped "
            "and levels are upper bounds (ISO 3745:2012, 9.4.2).",
            SoundPowerWarning,
            stacklevel=2,
        )
    return np.asarray(k1, dtype=np.float64)


def meteorological_corrections(
    temperature: float = 23.0,
    static_pressure: float = _PS0_KPA,
    *,
    air_absorption_coefficient: float | np.ndarray | None = None,
    radius: float = 1.0,
) -> MeteorologicalCorrection:
    r"""Meteorological corrections C1, C2, C3 (ISO 3745:2012 Eq. 14 block).

    Using the measured static pressure :math:`p_s` (kPa) and air temperature
    :math:`\theta` (deg C) form:

    .. math::

       C_1 = -10 \log_{10}\frac{p_s}{p_{s,0}}
       + 5 \log_{10}\frac{273 + \theta}{\theta_0},
       \qquad \theta_0 = 314~\text{K}

       C_2 = -10 \log_{10}\frac{p_s}{p_{s,0}}
       + 15 \log_{10}\frac{273 + \theta}{\theta_1},
       \qquad \theta_1 = 296~\text{K}

       C_3 = A_0 \left( 1.0053 - 0.0012\,A_0 \right)^{1.6},
       \qquad A_0 = \alpha(f)\,r

    :math:`p_{s,0} = 101.325` kPa. This is the :math:`p_s`/:math:`\theta`
    form of C1 (not the characteristic-impedance form), chosen because it
    needs only the measured :math:`p_s` and :math:`\theta` and is consistent
    with C2. At the reference conditions (23 deg C, 101.325 kPa)
    :math:`C_2 = 0` exactly while :math:`C_1 = 5 \log_{10}(296/314) = -0.128` dB.
    C3 requires the atmospheric attenuation coefficient :math:`\alpha(f)`
    from ISO 9613-1 (not computed here); without it :math:`C_3 = 0`.

    :param temperature: Air temperature ``theta`` at the test, in degrees C.
    :param static_pressure: Static pressure ``ps`` at the test, in kilopascals.
    :param air_absorption_coefficient: ``a(f)`` (dB/m), scalar or per band, for
        C3; ``None`` leaves :math:`C_3 = 0`.
    :param radius: Measurement radius ``r`` (m), used only in
        :math:`A_0 = \alpha(f)\,r`.
    :return: :class:`MeteorologicalCorrection`.
    """
    if static_pressure <= 0.0:
        raise ValueError("'static_pressure' must be positive (kPa).")
    if temperature <= -273.0:
        raise ValueError("'temperature' must be above -273 degrees Celsius.")
    if radius <= 0.0:
        raise ValueError("'radius' must be positive.")
    theta_k = 273.0 + temperature
    p_term = -10.0 * np.log10(static_pressure / _PS0_KPA)
    c1 = float(p_term + 5.0 * np.log10(theta_k / _THETA0_K))
    c2 = float(p_term + 15.0 * np.log10(theta_k / _THETA1_K))
    if air_absorption_coefficient is None:
        c3: float | np.ndarray = 0.0
    else:
        a0 = np.asarray(air_absorption_coefficient, dtype=np.float64) * radius
        if np.any(a0 < 0.0):
            raise ValueError("'air_absorption_coefficient' must be non-negative.")
        c3_arr = a0 * (1.0053 - 0.0012 * a0) ** 1.6
        c3 = as_float_or_array(c3_arr)
    return MeteorologicalCorrection(c1=c1, c2=c2, c3=c3)


def _sigma_r0_3745(nominal: int, room: PrecisionRoom) -> float:
    """Per-band sigma_R0 (ISO 3745:2012 Table 2 hemi / Table 3 anechoic), dB."""
    if 50 <= nominal <= 80:
        return 2.0
    if 100 <= nominal <= 630:
        return 1.5 if room == "hemi-anechoic" else 1.0
    if 800 <= nominal <= 5000:
        return 1.0 if room == "hemi-anechoic" else 0.5
    if 6300 <= nominal <= 10000:
        return 1.5 if room == "hemi-anechoic" else 1.0
    if 12500 <= nominal <= 20000:
        return 2.0
    raise ValueError(
        f"No ISO 3745:2012 sigma_R0 for {nominal} Hz; expected a nominal "
        "one-third-octave mid-band from 50 Hz to 20 kHz."
    )


def precision_uncertainty(
    sigma_r0: float | np.ndarray,
    sigma_omc: float = 0.0,
    coverage_factor: float = 2.0,
) -> float | np.ndarray:
    r"""Expanded uncertainty :math:`U = k \sigma_{tot}` (ISO 3745:2012).

    ISO 3745:2012 Eq. 24/25:
    :math:`\sigma_{tot} = \sqrt{\sigma_{R0}^2 + \sigma_{omc}^2}` and
    :math:`U = k \sigma_{tot}`, with :math:`k = 2` (95 %, two-sided) or
    :math:`k = 1.6` (95 %, one-sided, when comparing to a limit).

    :param sigma_r0: Reproducibility standard deviation (Tables 2/3), dB.
    :param sigma_omc: Operating/mounting standard deviation ``sigma_omc``, dB.
    :param coverage_factor: ``k`` (typically 2 or 1.6).
    :return: ``U`` in decibels, scalar or per band matching ``sigma_r0``.
    """
    if coverage_factor <= 0.0:
        raise ValueError("'coverage_factor' must be positive.")
    if sigma_omc < 0.0:
        raise ValueError("'sigma_omc' must be non-negative.")
    sigma_tot = np.hypot(np.asarray(sigma_r0, dtype=np.float64), sigma_omc)
    u = coverage_factor * sigma_tot
    return as_float_or_array(u)


def sound_power_anechoic(
    levels_positions: np.ndarray,
    surface: PrecisionSurface,
    *,
    radius: float | None = None,
    background_levels: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    areas: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = _PS0_KPA,
    air_absorption_coefficient: float | np.ndarray | None = None,
    sigma_omc: float = 0.0,
    coverage_factor: float = 2.0,
) -> PrecisionSoundPowerResult:
    r"""Sound power level in an (hemi-)anechoic room (ISO 3745:2012, precision).

    ``levels_positions`` is an ``(NM, NB)`` array of time-averaged position
    levels :math:`L'_{pi(\mathrm{ST})}` (one row per microphone, one column
    per band). Each position is background-corrected by :math:`K_{1i}`
    (Eq. 11, from ``background_levels`` and ``frequencies``), the corrected
    levels are surface-averaged (equal-area Eq. 12, or area-weighted Eq. 13
    when ``areas`` are given) and combined with the surface area and the
    meteorological corrections:

    .. math::

       L_W = 10 \log_{10}\!\left[ \frac{1}{N_M}
       \sum_i 10^{0.1 (L'_{pi} - K_{1i})} \right]
       + 10 \log_{10}\frac{S}{S_0} + C_1 + C_2 + C_3

    :math:`S = 4\pi r^2` for a ``'sphere'`` (anechoic, Eq. 14) or
    :math:`2\pi r^2` for a ``'hemisphere'`` (hemi-anechoic, Eq. 15). There is
    no ISO 3744 ``K2``
    environmental term. The reproducibility ``sigma_R0`` is taken from Table 3
    (sphere/anechoic) or Table 2 (hemisphere/hemi-anechoic).

    :param levels_positions: ``(NM, NB)`` position levels, in decibels.
    :param surface: ``'sphere'`` or ``'hemisphere'``.
    :param radius: Measurement radius ``r``, in metres.
    :param background_levels: ``(NM, NB)`` (or single-spectrum) background
        levels for ``K1i``; requires ``frequencies``.
    :param frequencies: ``(NB,)`` nominal mid-band frequencies (Hz), for the
        K1 criterion, the A-weighted total and the per-band uncertainty.
    :param areas: ``(NM,)`` partial areas ``Si`` for the area-weighted average
        (Eq. 13); omit for the equal-area average (Eq. 12).
    :param temperature: Air temperature ``theta`` (deg C), for C1/C2.
    :param static_pressure: Static pressure ``ps`` (kPa), for C1/C2.
    :param air_absorption_coefficient: ``a(f)`` (dB/m) for C3, scalar or
        per band; ``None`` leaves :math:`C_3 = 0`.
    :param sigma_omc: Operating/mounting standard deviation, dB.
    :param coverage_factor: ``k`` (2 two-sided, 1.6 one-sided).
    :return: :class:`PrecisionSoundPowerResult`.
    """
    if surface not in ("sphere", "hemisphere"):
        raise ValueError("'surface' must be 'sphere' or 'hemisphere'.")
    if radius is None or radius <= 0:
        raise ValueError("A positive 'radius' is required.")
    levels = np.atleast_2d(np.asarray(levels_positions, dtype=np.float64))
    if levels.ndim != 2:
        raise ValueError("'levels_positions' must be a 2D (positions, bands) array.")
    n_positions, n_bands = levels.shape

    area = (4.0 if surface == "sphere" else 2.0) * np.pi * radius**2
    room: PrecisionRoom = "anechoic" if surface == "sphere" else "hemi-anechoic"

    freqs = None if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    if freqs is not None and freqs.shape[0] != n_bands:
        raise ValueError("'frequencies' length must match the number of bands.")

    # --- per-position background correction K1i (Eq. 11) ------------------
    if background_levels is not None:
        if freqs is None:
            raise ValueError(
                "'frequencies' are required with 'background_levels' to select "
                "the frequency-dependent K1 criterion (ISO 3745:2012 9.4.2)."
            )
        bg = np.atleast_2d(np.asarray(background_levels, dtype=np.float64))
        if bg.shape == (1, n_bands) and n_positions != 1:
            bg = np.broadcast_to(bg, (n_positions, n_bands))
        if bg.shape != levels.shape:
            raise ValueError(
                "'background_levels' must match 'levels_positions' shape, or be "
                "a single spectrum of shape (NB,) or (1, NB)."
            )
        k1 = precision_background_correction(levels, bg, freqs)
    else:
        k1 = np.zeros_like(levels)

    corrected = levels - k1  # Lpi = L'pi(ST) - K1i

    # --- surface time-averaged level Lp_bar (Eq. 12 equal / Eq. 13 area) --
    mean_level = energy_mean(levels, axis=0)
    if areas is None:
        lp_bar = energy_mean(corrected, axis=0)
    else:
        seg = np.asarray(areas, dtype=np.float64)
        if seg.shape != (n_positions,):
            raise ValueError("'areas' must have one value per microphone position.")
        if np.any(seg <= 0.0):
            raise ValueError("All 'areas' must be positive.")
        lp_bar = weighted_energy_mean(corrected, seg[:, None], axis=0)

    # --- meteorological corrections C1, C2, C3 (Eq. 14) -------------------
    mc = meteorological_corrections(
        temperature,
        static_pressure,
        air_absorption_coefficient=air_absorption_coefficient,
        radius=radius,
    )
    c3 = np.broadcast_to(np.asarray(mc.c3, dtype=np.float64), (n_bands,)).astype(
        np.float64
    )

    lw = lp_bar + 10.0 * np.log10(area / _S0) + mc.c1 + mc.c2 + c3

    # --- A-weighted total LWA (Eq. C.1) -----------------------------------
    if freqs is not None:
        try:
            ck = _a_weighting_corrections(freqs)
        except ValueError:
            # Bands outside the ISO 3744 Annex E table (e.g. the 12,5 kHz to
            # 20 kHz one-third octaves that ISO 3745 covers, Tables 2/3): the
            # per-band determination stands, only the A-weighted total is
            # undefined.
            lwa = float("nan")
        else:
            lwa = energy_sum(lw + ck)
    else:
        lwa = float(lw[0]) if n_bands == 1 else float("nan")

    # --- directivity (Eq. 21) and non-uniformity (Eq. 22) indices ---------
    directivity = np.asarray(corrected - lp_bar[np.newaxis, :], dtype=np.float64)
    if n_positions > 1:
        lp_av = np.mean(corrected, axis=0)  # arithmetic mean (Eq. 22)
        vir = np.sqrt(
            np.sum((corrected - lp_av[np.newaxis, :]) ** 2, axis=0)
            / (n_positions - 1)
        )
    else:
        vir = np.zeros(n_bands, dtype=np.float64)

    # --- uncertainty (Eq. 24/25) ------------------------------------------
    u_a = float(precision_uncertainty(_SIGMA_R0_3745_A, sigma_omc, coverage_factor))
    if freqs is not None:
        sigma_bands = np.array(
            [_sigma_r0_3745(round(float(f)), room) for f in freqs],
            dtype=np.float64,
        )
        u_bands = np.asarray(
            precision_uncertainty(sigma_bands, sigma_omc, coverage_factor),
            dtype=np.float64,
        )
    else:
        u_bands = np.full(n_bands, np.nan, dtype=np.float64)

    return PrecisionSoundPowerResult(
        frequencies=freqs,
        sound_power_level=np.asarray(lw, dtype=np.float64),
        surface_pressure_level=np.asarray(lp_bar, dtype=np.float64),
        mean_pressure_level=mean_level,
        background_correction=np.asarray(k1, dtype=np.float64),
        c1=mc.c1,
        c2=mc.c2,
        c3=c3,
        directivity_index=directivity,
        non_uniformity_index=np.asarray(vir, dtype=np.float64),
        surface_area=float(area),
        surface=surface,
        sound_power_level_a=lwa,
        uncertainty=u_a,
        uncertainty_bands=u_bands,
        coverage_factor=float(coverage_factor),
    )


# ===========================================================================
# ISO 9614-3:2002 - sound power by sound-intensity scanning (precision). The
# precision sibling of ISO 9614-2 (engineering). Single grade, bias-error
# factor K = 10 dB, five acceptance criteria, and a meteorologically
# normalized sound power level LW0 (Eq. 10).
# ===========================================================================

_P0_INTENSITY = 1.0e-12  #: Reference sound power, in watts (3.6.3).
_I0 = 1.0e-12  #: Reference sound intensity, in W/m^2 (3.5).
_K_9614_3 = 10.0  #: Bias-error factor K, in dB (def. 3.11).
_FS_LIMIT = 2.0  #: Criterion 4 field-non-uniformity limit (Eq. C.4).
_F_PI_DIFF_LIMIT = 3.0  #: Criterion 3 signed-minus-unsigned limit, dB (Eq. C.3).
_FS_RATIO_LOW = 0.83  #: Criterion 5 lower bound on FS(1)/FS(2) (Eq. C.5).
_FS_RATIO_HIGH = 1.2  #: Criterion 5 upper bound on FS(1)/FS(2) (Eq. C.5).


@dataclass(frozen=True)
class PrecisionFieldIndicators:
    r"""ISO 9614-3:2002 Annex B field indicators (per band).

    ``ft`` is the temporal-variability indicator (= F1 of ISO 9614-1, Eq. B.1),
    ``None`` unless time-window intensities are supplied. ``f_pi_unsigned`` is
    the unsigned pressure-intensity indicator (= F2, Eq. B.3, using the mean
    magnitude of the segment intensities) and ``f_pi_signed`` the signed one
    (= F3, Eq. B.6, using the algebraic mean); by construction
    :math:`F_{pI_n}^{\mathrm{signed}} \ge F_{pI_n}^{\mathrm{unsigned}}`.
    ``fs`` is the field-non-uniformity indicator (= F4, Eq. B.8)."""

    ft: np.ndarray | None
    f_pi_unsigned: np.ndarray
    f_pi_signed: np.ndarray
    fs: np.ndarray


@dataclass(frozen=True)
class PrecisionCriteria:
    r"""ISO 9614-3:2002 Annex C acceptance criteria (per band, pass/fail).

    Each attribute is a boolean array (True = satisfied) or ``None`` when its
    inputs are absent. ``criterion_1`` scan repeatability
    :math:`\lvert L_{I_n}(1) - L_{I_n}(2) \rvert \le s/2` (Eq. C.1);
    ``criterion_2`` dynamic-capability
    adequacy :math:`L_d \ge F_{pI_n}^{\mathrm{signed}}` (Eq. C.2);
    ``criterion_3``
    :math:`F_{pI_n}^{\mathrm{signed}} - F_{pI_n}^{\mathrm{unsigned}} \le 3` dB
    (Eq. C.3); ``criterion_4``
    :math:`F_S \le 2` (Eq. C.4); ``criterion_5`` scan-density convergence
    :math:`0.83 \le F_S(1)/F_S(2) \le 1.2` (Eq. C.5). ``qualified`` is the
    conjunction of criteria 1-3 with the field non-uniformity accepted
    through criterion 4
    or, where evaluated, criterion 5 (C.1.6.2: a band satisfying criterion 5
    is qualified as a final result even if :math:`F_S(2) \ge 2`); ``None``
    unless both criterion 1 and criterion 2 are evaluable."""

    criterion_1: np.ndarray | None
    criterion_2: np.ndarray | None
    criterion_3: np.ndarray
    criterion_4: np.ndarray
    criterion_5: np.ndarray | None
    qualified: np.ndarray | None


@dataclass(frozen=True)
class PrecisionIntensityResult:
    r"""Result of an ISO 9614-3:2002 sound-power-by-scanning determination.

    ``partial_power`` is the signed :math:`P_i = I_{n,i} S_i` per partial
    surface and band (Eq. 5); ``sound_power`` the signed band total
    :math:`P = \sum P_i` (Eq. 8) and ``sound_power_level`` its level
    :math:`L_W = 10 \log_{10}(P/P_0)` (Eq. 9), ``NaN``
    where :math:`P \le 0` (``not_applicable_band`` True, clause 9.2).
    ``sound_power_level_normalized`` is ``LW0`` normalized to 23 deg C /
    101 325 Pa (Eq. 10). ``sound_power_level_a`` is the A-weighted total over
    applicable bands (``NaN`` without ``frequencies`` and more than one band)."""

    frequencies: np.ndarray | None
    partial_power: np.ndarray
    sound_power: np.ndarray
    sound_power_level: np.ndarray
    sound_power_level_normalized: np.ndarray
    not_applicable_band: np.ndarray
    surface_area: float
    sound_power_level_a: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the ``LW`` spectrum; non-applicable bands are hatched/greyed.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_sound_power

        check_language(language)
        return plot_sound_power(self, ax=ax, language=language, **kwargs)


def precision_field_indicators(
    segment_intensity: np.ndarray,
    segment_pressure_levels: np.ndarray,
    *,
    time_window_intensity: np.ndarray | None = None,
) -> PrecisionFieldIndicators:
    r"""ISO 9614-3:2002 Annex B field indicators from segment data.

    Over the ``N`` segments of the whole measurement surface (per band):

    .. math::

       \overline{L_p} = 10 \log_{10}\!\left[ \frac{1}{N}
       \sum_j 10^{0.1 L_{pj}} \right] \tag{Eq. B.4}

       L_{|I_n|} = 10 \log_{10}\!\left[ \frac{1}{N}
       \sum_j \frac{|I_{nj}|}{I_0} \right] \tag{Eq. B.5}

       L_{I_n} = 10 \log_{10}\!\left[ \frac{1}{I_0} \left| \frac{1}{N}
       \sum_j I_{nj} \right| \right] \tag{Eq. B.7}

       F_{pI_n}^{\mathrm{unsigned}} = \overline{L_p} - L_{|I_n|}
       \tag{Eq. B.3}

       F_{pI_n}^{\mathrm{signed}} = \overline{L_p} - L_{I_n} \tag{Eq. B.6}

       F_S = \frac{1}{\overline{I_n}} \sqrt{ \frac{1}{N-1}
       \sum_j \left( I_{nj} - \overline{I_n} \right)^2 } \tag{Eq. B.8}

    With ``time_window_intensity`` (an ``(M, NB)`` array of window-averaged
    intensities) the temporal-variability indicator ``FT`` (Eq. B.1) is also
    returned.

    :param segment_intensity: ``(N, NB)`` signed segment normal intensity, W/m^2.
    :param segment_pressure_levels: ``(N, NB)`` segment pressure levels, dB.
    :param time_window_intensity: Optional ``(M, NB)`` window intensities for FT.
    :return: :class:`PrecisionFieldIndicators`.
    """
    i_n = np.atleast_2d(np.asarray(segment_intensity, dtype=np.float64))
    lp = np.atleast_2d(np.asarray(segment_pressure_levels, dtype=np.float64))
    if i_n.shape != lp.shape:
        raise ValueError(
            "'segment_intensity' and 'segment_pressure_levels' must have the "
            f"same shape, got {i_n.shape} and {lp.shape}."
        )
    n_seg = i_n.shape[0]
    if n_seg < 2:
        raise ValueError("At least two segments are required for the indicators.")

    lp_bar = energy_mean(lp, axis=0)  # Eq. B.4
    li_unsigned = 10.0 * np.log10(np.mean(np.abs(i_n), axis=0) / _I0)  # Eq. B.5
    mean_signed = np.mean(i_n, axis=0)
    li_signed = 10.0 * np.log10(
        np.maximum(np.abs(mean_signed), np.finfo(float).tiny) / _I0
    )  # Eq. B.7 (magnitude; sign carried separately by the P<0 rule)
    f_pi_unsigned = np.asarray(lp_bar - li_unsigned, dtype=np.float64)
    f_pi_signed = np.asarray(lp_bar - li_signed, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        fs = np.sqrt(
            np.sum((i_n - mean_signed[np.newaxis, :]) ** 2, axis=0) / (n_seg - 1)
        ) / mean_signed  # Eq. B.8
    fs = np.asarray(fs, dtype=np.float64)

    ft: np.ndarray | None = None
    if time_window_intensity is not None:
        win = np.atleast_2d(np.asarray(time_window_intensity, dtype=np.float64))
        if win.shape[-1] != i_n.shape[-1]:
            raise ValueError(
                "'time_window_intensity' last axis must match the number of bands."
            )
        m = win.shape[0]
        if m < 2:
            raise ValueError("At least two time windows are required for FT.")
        mean_t = np.mean(win, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ft = np.asarray(
                np.sqrt(np.sum((win - mean_t[np.newaxis, :]) ** 2, axis=0) / (m - 1))
                / mean_t,
                dtype=np.float64,
            )  # Eq. B.1

    return PrecisionFieldIndicators(
        ft=ft, f_pi_unsigned=f_pi_unsigned, f_pi_signed=f_pi_signed, fs=fs
    )


def _sigma_r0_9614_3(nominal: int) -> float:
    """Per-band sigma_R0 (ISO 9614-3:2002 Table 1), in dB; also criterion-1 s."""
    if 50 <= nominal <= 160:
        return 2.0
    if 200 <= nominal <= 315:
        return 1.5
    if 400 <= nominal <= 5000:
        return 1.0
    if nominal == 6300:
        return 2.0
    raise ValueError(
        f"No ISO 9614-3:2002 Table 1 sigma_R0 for {nominal} Hz; expected a "
        "nominal one-third-octave mid-band from 50 Hz to 6300 Hz."
    )


def precision_qualification(
    indicators: PrecisionFieldIndicators,
    *,
    scan_intensity_level_1: np.ndarray | None = None,
    scan_intensity_level_2: np.ndarray | None = None,
    pressure_residual_index: float | np.ndarray | None = None,
    field_nonuniformity_1: np.ndarray | None = None,
    field_nonuniformity_2: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    repeatability_limit: float | np.ndarray | None = None,
) -> PrecisionCriteria:
    r"""Evaluate the five ISO 9614-3:2002 Annex C acceptance criteria per band.

    :param indicators: The :class:`PrecisionFieldIndicators` (gives criteria 3
        and 4 directly).
    :param scan_intensity_level_1: ``LIn(1)`` per band (dB), first scan.
    :param scan_intensity_level_2: ``LIn(2)`` per band (dB), second scan; with
        the first scan and ``s`` this gives criterion 1
        (:math:`\lvert \Delta L \rvert \le s/2`).
    :param pressure_residual_index: ``delta_pI0`` (dB), scalar or per band; with
        :math:`K = 10` gives ``Ld`` for criterion 2
        (:math:`L_d \ge F_{pI_n}^{\mathrm{signed}}`).
    :param field_nonuniformity_1: ``FS(1)`` per band (initial scan density).
    :param field_nonuniformity_2: ``FS(2)`` per band (doubled density); with
        ``FS(1)`` gives criterion 5.
    :param frequencies: ``(NB,)`` nominal mid-band frequencies (Hz), selecting
        the criterion-1 limit ``s`` from Table 1.
    :param repeatability_limit: Override for ``s`` (dB), scalar or per band.
    :return: :class:`PrecisionCriteria`.
    """
    f_pi_signed = indicators.f_pi_signed
    n_bands = f_pi_signed.shape[0]

    # Criteria 3 and 4 are always available from the indicators.
    criterion_3 = np.asarray(
        (f_pi_signed - indicators.f_pi_unsigned) <= _F_PI_DIFF_LIMIT, dtype=bool
    )
    criterion_4 = np.asarray(indicators.fs <= _FS_LIMIT, dtype=bool)

    # Criterion 1: |LIn(1) - LIn(2)| <= s/2.
    criterion_1: np.ndarray | None = None
    if scan_intensity_level_1 is not None and scan_intensity_level_2 is not None:
        l1 = np.asarray(scan_intensity_level_1, dtype=np.float64)
        l2 = np.asarray(scan_intensity_level_2, dtype=np.float64)
        if repeatability_limit is not None:
            s = np.broadcast_to(
                np.asarray(repeatability_limit, dtype=np.float64), (n_bands,)
            ).astype(np.float64)
        elif frequencies is not None:
            nominal = [round(float(f)) for f in np.asarray(frequencies)]
            s = np.array([_sigma_r0_9614_3(f) for f in nominal], dtype=np.float64)
        else:
            raise ValueError(
                "Criterion 1 needs the limit s: provide 'frequencies' (Table 1) "
                "or 'repeatability_limit'."
            )
        criterion_1 = np.asarray(np.abs(l1 - l2) <= s / 2.0, dtype=bool)

    # Criterion 2: Ld >= F_pIn(signed), Ld = delta_pI0 - K.
    criterion_2: np.ndarray | None = None
    if pressure_residual_index is not None:
        dpi0 = np.broadcast_to(
            np.asarray(pressure_residual_index, dtype=np.float64), (n_bands,)
        ).astype(np.float64)
        ld = dpi0 - _K_9614_3
        criterion_2 = np.asarray(ld >= f_pi_signed, dtype=bool)

    # Criterion 5: 0,83 <= FS(1)/FS(2) <= 1,2.
    criterion_5: np.ndarray | None = None
    if field_nonuniformity_1 is not None and field_nonuniformity_2 is not None:
        fs1 = np.asarray(field_nonuniformity_1, dtype=np.float64)
        fs2 = np.asarray(field_nonuniformity_2, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = fs1 / fs2
        criterion_5 = np.asarray(
            (ratio >= _FS_RATIO_LOW) & (ratio <= _FS_RATIO_HIGH), dtype=bool
        )

    qualified: np.ndarray | None = None
    if criterion_1 is not None and criterion_2 is not None:
        # C.1.6.2: where the doubled-density scan satisfies criterion 5, the
        # band qualifies as a final result even if FS(2) >= 2 fails criterion 4.
        non_uniformity_ok = (
            criterion_4 if criterion_5 is None else (criterion_4 | criterion_5)
        )
        qualified = criterion_1 & criterion_2 & criterion_3 & non_uniformity_ok

    return PrecisionCriteria(
        criterion_1=criterion_1,
        criterion_2=criterion_2,
        criterion_3=criterion_3,
        criterion_4=criterion_4,
        criterion_5=criterion_5,
        qualified=qualified,
    )


def sound_power_intensity_precision(
    partial_intensity: np.ndarray,
    areas: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    temperature: float = 23.0,
    barometric_pressure: float = 101325.0,
) -> PrecisionIntensityResult:
    r"""Sound power by intensity scanning, precision (ISO 9614-3:2002).

    ``partial_intensity`` is an ``(N, NB)`` array (or ``(N,)`` for a single
    band) of the signed normal intensity :math:`I_{ni}` on each of the ``N``
    partial surfaces (already the two-scan result), and ``areas`` the ``(N,)``
    partial surface areas :math:`S_i`. The partial powers
    :math:`P_i = I_{ni} S_i` (Eq. 5) are summed to :math:`P` (Eq. 8) and
    :math:`L_W = 10 \log_{10}(P/P_0)` (Eq. 9); a band with net :math:`P \le 0` is
    flagged (``not_applicable_band``, clause 9.2) and reported as ``NaN``.
    :math:`L_{W0}` normalizes to reference meteorology:

    .. math::

       L_{W0} = L_W - 15 \log_{10}\!\left( \frac{B}{101325} \cdot
       \frac{296.15}{273.15 + \theta} \right) \tag{Eq. 10}

    :param partial_intensity: ``(N, NB)`` signed normal intensity, W/m^2.
    :param areas: ``(N,)`` partial surface areas ``Si``, m^2.
    :param frequencies: ``(NB,)`` nominal mid-band frequencies (Hz), for LWA.
    :param temperature: Air temperature ``theta`` (deg C), for LW0 (Eq. 10).
    :param barometric_pressure: Barometric pressure ``B`` (Pa), for LW0.
    :return: :class:`PrecisionIntensityResult`.
    """
    raw_intensity = np.asarray(partial_intensity, dtype=np.float64)
    seg = np.asarray(areas, dtype=np.float64)
    if seg.ndim != 1:
        raise ValueError("'areas' must be a 1D array of partial surface areas.")
    n_seg = seg.shape[0]
    # A 1-D input is unambiguously ``(N,)`` segments with one band -> ``(N, 1)``;
    # a 2-D input is taken as ``(segments, bands)`` as given. Keying off the
    # original ndim avoids misreading a genuine ``(1, N)`` single-segment,
    # N-band array as N segments when ``n_seg == N``.
    if raw_intensity.ndim == 1:
        intensity = raw_intensity.reshape(-1, 1)
    else:
        intensity = np.atleast_2d(raw_intensity)
    if intensity.shape[0] != n_seg:
        raise ValueError(
            f"'partial_intensity' first axis ({intensity.shape[0]}) must match "
            f"the number of 'areas' ({n_seg})."
        )
    if np.any(seg <= 0.0):
        raise ValueError("All 'areas' must be positive.")
    if temperature <= -273.15:
        raise ValueError("'temperature' must be above -273,15 degrees Celsius.")
    if barometric_pressure <= 0.0:
        raise ValueError("'barometric_pressure' must be positive (Pa).")
    n_bands = intensity.shape[1]
    if frequencies is not None and np.asarray(frequencies).shape != (n_bands,):
        raise ValueError("'frequencies' length must match the number of bands.")

    partial_power = intensity * seg[:, None]  # Eq. 5
    total_power = np.sum(partial_power, axis=0)  # Eq. 8
    not_applicable = total_power <= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lw = np.where(
            total_power > 0.0,
            10.0 * np.log10(np.maximum(total_power, np.finfo(float).tiny) / _P0_INTENSITY),
            np.nan,
        )

    # Eq. 10: meteorological normalization to 23 deg C / 101 325 Pa.
    norm = 15.0 * np.log10(
        (barometric_pressure / 101325.0) * (296.15 / (273.15 + temperature))
    )
    lw0 = lw - norm

    if np.any(not_applicable):
        warnings.warn(
            "Net sound power is non-positive in one or more bands; ISO "
            "9614-3:2002 is not applicable to those bands (clause 9.2).",
            SoundPowerWarning,
            stacklevel=2,
        )

    # A-weighted total over applicable bands (clause 9.2 / 4.3).
    if frequencies is not None:
        freqs = np.asarray(frequencies, dtype=np.float64)
        ck = _a_weighting_corrections(freqs)
        contrib = 10.0 ** (0.1 * (lw + ck))
        total = float(np.sum(contrib[~not_applicable]))
        lwa = 10.0 * np.log10(total) if total > 0.0 else float("nan")
    else:
        freqs = None
        lwa = float(lw[0]) if n_bands == 1 and not bool(not_applicable[0]) else float("nan")

    return PrecisionIntensityResult(
        frequencies=freqs,
        partial_power=np.asarray(partial_power, dtype=np.float64),
        sound_power=np.asarray(total_power, dtype=np.float64),
        sound_power_level=np.asarray(lw, dtype=np.float64),
        sound_power_level_normalized=np.asarray(lw0, dtype=np.float64),
        not_applicable_band=np.asarray(not_applicable, dtype=bool),
        surface_area=float(np.sum(seg)),
        sound_power_level_a=lwa,
    )
