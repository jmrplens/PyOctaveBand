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

A noise burst or a transient emission has no steady power to report, and both
standards describe it by the **sound energy level** :math:`L_J = 10 \log_{10}
(J/J_0)` instead, :math:`J = \int P(t)\,\mathrm{d}t` in joules and
:math:`J_0 = 1` pJ (ISO 3744:2010 clauses 3.22 and 3.23). Its determination
(clause 8.3; ISO 3746:2010 clause 8.4) is the chain above with the single
event time-integrated sound pressure level :math:`L_E = 10 \log_{10}\!\left[
\int p^2\,\mathrm{d}t / E_0\right]`, :math:`E_0 = (20\ \mu\mathrm{Pa})^2\,
\mathrm{s}` (clause 3.4), in place of the time-averaged :math:`L_p`: the
:math:`N_\mathrm{e}` events at each position are combined into the level of one
event (Eq. 19 or Eq. 20), the positions are averaged as in 8.2.2 (Eq. 12), the
background and the environment are corrected by the same :math:`K_1` and
:math:`K_2` (Eq. 21, 22) and the surface term closes it:

.. math::

   L_J = \overline{L_E} + 10 \log_{10}\frac{S}{S_0} \tag{Eq. 23}

For a source that is steady over the whole interval :math:`T`, clause 3.4
NOTE 1 gives :math:`L_E = L_{p,T} + 10 \log_{10}(T/T_0)` with :math:`T_0 = 1`
s, so :math:`L_J = L_W + 10 \log_{10}(T/T_0)`: the energy a steady source
radiates in :math:`T` seconds. Annex E carries the band levels to the
A-weighted :math:`L_{J\mathrm{A}}` with the same :math:`C_k` as :math:`L_{W\mathrm{A}}`
(Eq. E.2), and Annex G refers either level to the reference atmosphere with
the corrections :math:`C_1 + C_2` (Eq. G.1, G.3), required above 500 m of
altitude or below 10 degrees C (clauses 8.2.5 and 8.3.6).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._internal.levels_math import energy_mean, energy_sum
from .._internal.types import as_float_or_array
from .._internal.validation import (
    check_engine,
    require_choice,
    require_non_negative,
    require_per_band,
    require_positive,
    require_positive_array,
    require_ranks,
    require_same_length,
)
from ._shared import (
    _S0,
    Grade,
    SoundPowerWarning,
    _a_weighting_corrections,
    _background_exposure,
    _check_grade,
    _single_event_mean,
    _validate_event_count,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata
    from .declaration import DeclarationForm, NoiseEmissionDeclaration

__all__ = [
    "Grade",
    "ReferenceAtmosphereCorrection",
    "RoomEnvironment",
    "SoundEnergyResult",
    "SoundPowerResult",
    # Defined in :mod:`._shared`, which every method module imports; re-exported
    # here because this is the documented path of the warning class.
    "SoundPowerWarning",
    "Surface",
    "background_noise_correction",
    "environmental_correction",
    "mean_single_event_level",
    "measurement_positions",
    "reference_atmosphere_correction",
    "sound_energy_pressure",
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

#: Refusal for the number of reflecting planes bounding the measurement
#: surface. Named because the three entry points that take it must refuse it
#: in the same words: a caller who reads one message and then meets another
#: for the same mistake has to work out whether it is the same mistake.
_REFLECTING_PLANES_MSG = "'reflecting_planes' must be 1, 2 or 3."
#: Typical A-weighted reproducibility standard deviation sigma_R0, in dB.
#: ISO 3744 Table 2 (1,5); ISO 3746 Table 1 (3,0, tone-free). Table 2 states
#: its values "for sound power levels and sound energy levels" alike, and
#: Eq. (24) sets u(LJ) = u(LW), so the sound energy path reads the same table.
_SIGMA_R0: dict[str, float] = {"engineering": 1.5, "survey": 3.0}
#: Coverage factor of the expanded uncertainty U = k sigma_tot, 95 % two-sided
#: (ISO 3744:2010 clause 9.1, Eq. 26).
_COVERAGE_FACTOR_95 = 2.0

# --- ISO 3744:2010 Annex G, reference meteorological conditions ------------
#: Reference static pressure p_s,0, in kilopascals (Annex G, Eq. G.1).
_PS0_KPA = 101.325
#: theta_0 = 314 K, the C1 reference temperature that makes the characteristic
#: impedance of air 400 N s/m^3 at p_s,0 (Annex G NOTE); theta_1 = 296 K, the
#: C2 radiation-impedance reference.
_THETA0_K = 314.0
_THETA1_K = 296.0
#: Celsius to kelvin offset as Annex G prints it, (273,15 + theta).
_KELVIN_OFFSET = 273.15
#: Static pressure from the altitude of the test site (Eq. G.2):
#: p_s = p_s,0 (1 - a H_a)^b, a = 2,256 0 x 10^-5 m^-1, b = 5,255 3.
_ALTITUDE_A_PER_M = 2.2560e-5
_ALTITUDE_B = 5.2553


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

    def __post_init__(self) -> None:
        """Reject a determination whose per-band quantities disagree.

        The boxed A-weighted total is summed over every band of
        ``sound_power_level``, and the table beneath prints one row per band,
        so a spectrum of another length gives a sheet whose headline number
        sums bands its own table does not show.

        ``grade`` is pinned beside the shapes because the fiche names the
        applied standard from it: a tag that is not exactly ``'survey'``
        or ``'engineering'`` would print a survey (ISO 3746, grade 3)
        measurement as ISO 3744 engineering grade 2, a stricter standard and
        a better accuracy grade than the measurement had.

        ``surface_area`` must be finite: no constructor can compute a
        non-finite ``S`` (:func:`sound_power_pressure` derives it from a
        validated positive geometry), and the boxed statement prints it as a
        plain number, so a NaN here would reach the accredited sheet as a
        literal ``nan``.

        ``sound_power_level`` must be finite for the same reason, band by
        band. The one producer, :func:`sound_power_pressure`, now requires
        finite pressure levels and computes a finite ``K1`` (clamped at the
        criterion) and ``K2`` from them, so a NaN band can only be written in
        by hand -- and nothing downstream would refuse it: the spectrum
        figure passes the bands through ``nan_to_num`` and would draw the NaN
        as a fabricated 0 dB bar in the ordinary colour, while the A-weighted
        total quietly vanishes from the title. This determination carries no
        per-band validity flags; the intensity-scanning siblings, whose
        standard does let a band be undeterminable, flag those bands and are
        rendered hatched instead. ``sound_power_level_a`` stays unpinned on
        purpose: several bands without band frequencies leave it NaN by
        documented design.

        :raises ValueError: if any per-band quantity disagrees with the rest,
            ``grade`` is neither ``'engineering'`` nor ``'survey'``,
            ``surface_area`` is not finite, or ``sound_power_level`` carries
            a non-finite band.
        """
        require_choice(self.grade, "grade", ("engineering", "survey"))
        if not math.isfinite(self.surface_area):
            msg = (
                "SoundPowerResult: 'surface_area' must be finite; "
                f"got {self.surface_area!r}."
            )
            raise ValueError(msg)
        require_ranks(
            self,
            frequencies=1,
            sound_power_level=1,
            surface_pressure_level=1,
            mean_pressure_level=1,
            background_correction=1,
            environmental_correction=1,
            directivity_index=2,
        )
        require_same_length(
            self,
            "frequencies",
            "sound_power_level",
            "surface_pressure_level",
            "mean_pressure_level",
            "background_correction",
            "environmental_correction",
            ("directivity_index", 1),
        )
        if not np.all(np.isfinite(self.sound_power_level)):
            msg = "SoundPowerResult: 'sound_power_level' must be finite."
            raise ValueError(msg)

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
        check_engine(engine)
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
    bg = require_per_band(background_levels, "background_levels", src, "source_levels")
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
    first: float | np.ndarray | None,
    second: float | None,
    first_name: str,
    second_name: str,
    equation: str,
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
        if np.ndim(volume) != 0:
            msg = (
                "'volume' must be a scalar (m^3); only 'reverberation_time' "
                "may vary per band."
            )
            raise ValueError(msg)
        t = np.asarray(reverberation_time, dtype=np.float64)
        if volume <= 0 or np.any(t <= 0.0):
            msg = "reverberation_time and volume must be > 0."
            raise ValueError(msg)
        return 0.16 * volume / t
    if mean_absorption_coefficient is not None and room_surface is not None:
        if np.ndim(room_surface) != 0:
            msg = (
                "'room_surface' must be a scalar (m^2); only "
                "'mean_absorption_coefficient' may vary per band."
            )
            raise ValueError(msg)
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
    surface_area = require_positive(surface_area, "surface_area")
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
        if reflecting_planes == 2:  # positions 2,3,6,7,9 of Table B.2  # noqa: PLR2004
            return _TABLE_B2, (1, 2, 5, 6, 8)
        return _TABLE_B3, (0, 1, 2)  # positions 1,2,3 of Table B.3
    # survey, ISO 3746 clause 8.2.1
    if reflecting_planes == 1:  # positions 4,5,6,10 of Table B.1
        return _TABLE_B1, (3, 4, 5, 9)
    if reflecting_planes == 2:  # positions 14,15,18 of Table B.2  # noqa: PLR2004
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
        msg = _REFLECTING_PLANES_MSG
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
    if reflecting_planes == 2:  # Eq. 10 (against a wall)  # noqa: PLR2004
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
        dims = require_positive_array(dimensions, "dimensions")
        if dims.size != 3:  # noqa: PLR2004
            msg = "'dimensions' must be 3 positive values (l1, l2, l3)."
            raise ValueError(msg)
        distance = require_positive(distance, "distance")
        box = (float(dims[0]), float(dims[1]), float(dims[2]))
        return (
            _box_area(box, distance, reflecting_planes),
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
    # A scalar is one band's centre frequency, not a shape mistake: accept it
    # for a single band (as 1-D) and reject it, by name, for several.
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
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
    omc_uncertainty = require_non_negative(omc_uncertainty, "omc_uncertainty")
    levels = np.atleast_2d(np.asarray(levels_positions, dtype=np.float64))
    if levels.ndim != 2:  # noqa: PLR2004
        msg = "'levels_positions' must be a 2D (positions, bands) array."
        raise ValueError(msg)
    if levels.shape[1] == 0:
        msg = "'levels_positions' must contain at least one frequency band."
        raise ValueError(msg)
    if not np.all(np.isfinite(levels)):
        msg = "'levels_positions' must contain only finite values."
        raise ValueError(msg)
    n_positions = levels.shape[0]

    # --- measurement surface area -----------------------------------------
    if reflecting_planes not in (1, 2, 3):
        msg = _REFLECTING_PLANES_MSG
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

    uncertainty = _COVERAGE_FACTOR_95 * float(
        np.hypot(_SIGMA_R0[grade], omc_uncertainty)
    )

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


# ---------------------------------------------------------------------------
# Sound energy level of a noise burst or transient emission (ISO 3744 clause
# 8.3, ISO 3746 clause 8.4, Annexes E and G)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceAtmosphereCorrection:
    r"""The two Annex G corrections to reference meteorological conditions.

    ``c1`` is the reference-quantity correction and ``c2`` the
    radiation-impedance correction of ISO 3744:2010 Annex G, both in
    decibels; ``total`` is their sum, the whole of what Eq. (G.1) adds to
    :math:`L_W` and Eq. (G.3) to :math:`L_J`. ``static_pressure`` is the
    :math:`p_\mathrm{s}` the corrections were evaluated at, in kilopascals,
    whether it was measured or estimated from the altitude by Eq. (G.2), and
    ``temperature`` the air temperature :math:`\theta`, in degrees Celsius.
    """

    c1: float
    c2: float
    static_pressure: float
    temperature: float

    @property
    def total(self) -> float:
        """``c1 + c2``, in decibels: the correction Eq. (G.1)/(G.3) applies."""
        return self.c1 + self.c2


@dataclass(frozen=True)
class SoundEnergyResult:
    r"""Result of a sound energy level determination from surface single event
    levels (ISO 3744:2010 clause 8.3, ISO 3746:2010 clause 8.4).

    ``sound_energy_level`` is the per-band ``LJ`` (ISO 3744 Eq. 23);
    ``surface_event_level`` the surface single event time-integrated sound
    pressure level :math:`\overline{L_E}` after the K1/K2 corrections (Eq. 22);
    ``mean_event_level`` the raw energy-averaged level
    :math:`\overline{L'_E}(\mathrm{ST})` over the positions (clause 8.3.3, as
    Eq. 12). ``background_correction`` (K1, Eq. 21) and
    ``environmental_correction`` (K2) are per band. ``sound_energy_level_a`` is
    the A-weighted total ``LJA`` (Eq. E.2), computed only when ``frequencies``
    are supplied; for a single band it equals ``LJ``, and for several bands
    without ``frequencies`` it is ``NaN`` (A-weighting needs the band centres).
    ``directivity_index`` is the apparent directivity index per microphone
    position and band, shape ``(NM, NB)``, formed from the single event levels
    exactly as clause 3.24 allows. ``uncertainty`` is the expanded uncertainty
    :math:`U = 2\sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}` (95 %),
    which clause 9.1 Eq. (24) makes the same for ``LJ`` as for ``LW``.
    ``events`` is the number of single sound emission events :math:`N_\mathrm{e}`
    the levels were reduced from, or ``None`` when the caller supplied the
    per-position mean single event levels directly; ``integration_time`` is
    the interval :math:`T` of the single event levels, in seconds, or ``None``
    when no background correction needed it.
    """

    frequencies: np.ndarray | None
    sound_energy_level: np.ndarray
    surface_event_level: np.ndarray
    mean_event_level: np.ndarray
    background_correction: np.ndarray
    environmental_correction: np.ndarray
    directivity_index: np.ndarray
    surface_area: float
    sound_energy_level_a: float
    uncertainty: float
    grade: str
    events: int | None
    integration_time: float | None

    def __post_init__(self) -> None:
        """Reject a determination whose per-band quantities disagree.

        The pins are those of :class:`SoundPowerResult`, for the same reasons:
        the A-weighted total is summed over every band of
        ``sound_energy_level`` and the figure draws one bar per band, so every
        per-band column must be as long as it; ``grade`` names the applied
        standard; ``surface_area`` and every band of ``sound_energy_level``
        must be finite because the figure would draw a NaN band as a 0 dB bar.
        ``events`` and ``integration_time`` are pinned as well: a count of
        events below one or a non-positive interval describes no measurement
        the standard defines.

        :raises ValueError: if any per-band quantity disagrees with the rest,
            ``grade`` is neither ``'engineering'`` nor ``'survey'``,
            ``surface_area`` is not finite, ``sound_energy_level`` carries a
            non-finite band, ``events`` is below one or ``integration_time``
            is not positive.
        """
        require_choice(self.grade, "grade", ("engineering", "survey"))
        if not math.isfinite(self.surface_area):
            msg = (
                "SoundEnergyResult: 'surface_area' must be finite; "
                f"got {self.surface_area!r}."
            )
            raise ValueError(msg)
        _validate_event_count(self.events, "SoundEnergyResult")
        if self.integration_time is not None:
            require_positive(self.integration_time, "integration_time")
        require_ranks(
            self,
            frequencies=1,
            sound_energy_level=1,
            surface_event_level=1,
            mean_event_level=1,
            background_correction=1,
            environmental_correction=1,
            directivity_index=2,
        )
        require_same_length(
            self,
            "frequencies",
            "sound_energy_level",
            "surface_event_level",
            "mean_event_level",
            "background_correction",
            "environmental_correction",
            ("directivity_index", 1),
        )
        if not np.all(np.isfinite(self.sound_energy_level)):
            msg = "SoundEnergyResult: 'sound_energy_level' must be finite."
            raise ValueError(msg)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the LJ spectrum with the A-weighted total annotated.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_sound_energy

        check_language(language)
        return plot_sound_energy(self, ax=ax, language=language, **kwargs)


def mean_single_event_level(
    levels: np.ndarray, *, events: int | None = None
) -> np.ndarray:
    r"""Mean single event time-integrated level of one event (ISO 3744 Eq. 19/20).

    Two ways of measuring :math:`N_\mathrm{e}` sound emission events reach the
    same quantity, the single event time-integrated sound pressure level of
    *one* event at a microphone position. Measured one at a time
    (``events=None``), the first axis of ``levels`` holds one entry per event
    and the mean is their energy average, ISO 3744:2010 Eq. (19):

    .. math::

       L'_{Ei(\mathrm{ST})} = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{e}}
       \sum_{q=1}^{N_\mathrm{e}} 10^{0.1 L'_{Ei,q(\mathrm{ST})}} \right]

    Measured as one level that encompasses :math:`N_\mathrm{e}` successive
    events (``events`` :math:`= N_\mathrm{e}`), the level of one event is the
    measurement less :math:`10 \log_{10} N_\mathrm{e}`, Eq. (20):

    .. math::

       L'_{Ei(\mathrm{ST})} = L'_{Ei,N_\mathrm{e}(\mathrm{ST})}
       - 10 \log_{10} N_\mathrm{e}

    ISO 3746:2010 Eq. (16)/(17) and ISO 3741:2010 Eq. (22)/(23) print the
    same pair. Both modes require at least five events (clause 8.3.1); fewer
    are accepted with a :class:`SoundPowerWarning`.

    :param levels: With ``events=None``, the per-event levels in decibels with
        the events on the first axis (``(Ne,)``, ``(Ne, NB)`` or
        ``(Ne, NM, NB)``); otherwise the one measurement encompassing
        ``events`` events, of any shape.
    :param events: ``None`` when each entry of the first axis is one event,
        else the number of events the single measurement encompasses.
    :return: The level of one event, in decibels: ``levels`` less its first
        axis for ``events=None``, the shape of ``levels`` otherwise.
    :raises ValueError: for non-finite levels, an empty first axis or a
        non-positive ``events``.
    """
    return _single_event_mean(levels, events, name="levels", stacklevel=3)


def _static_pressure_at_altitude(altitude: float) -> float:
    r"""Static pressure :math:`p_\mathrm{s} = p_{\mathrm{s},0}(1 - aH_\mathrm{a})^b`
    (ISO 3744:2010 Eq. G.2), in kilopascals.

    :param altitude: Altitude of the test site ``Ha``, in metres.
    :return: ``ps`` in kilopascals.
    :raises ValueError: for a non-finite altitude, or one at which the
        printed power law has no positive base.
    """
    if not math.isfinite(altitude) or 1.0 - _ALTITUDE_A_PER_M * altitude <= 0.0:
        msg = (
            "'altitude' must be finite and below "
            f"{1.0 / _ALTITUDE_A_PER_M:.0f} m (ISO 3744:2010 Eq. G.2)."
        )
        raise ValueError(msg)
    return float(_PS0_KPA * (1.0 - _ALTITUDE_A_PER_M * altitude) ** _ALTITUDE_B)


def reference_atmosphere_correction(
    temperature: float,
    static_pressure: float | None = None,
    *,
    altitude: float | None = None,
) -> ReferenceAtmosphereCorrection:
    r"""Corrections to reference meteorological conditions (ISO 3744 Annex G).

    A sound power or sound energy level determined by Eq. (18) or Eq. (23)
    holds for the meteorological conditions at the time and place of the test.
    Above 500 m of altitude or below 10 degrees C the standard requires it to
    be carried to the reference static pressure 101.325 kPa and air
    temperature 23.0 degrees C (clauses 8.2.5 and 8.3.6) by adding

    .. math::

       C_1 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}}
       + 5 \log_{10}\frac{273.15 + \theta}{\theta_0}, \qquad
       C_2 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}}
       + 15 \log_{10}\frac{273.15 + \theta}{\theta_1}

    with :math:`\theta_0 = 314` K and :math:`\theta_1 = 296` K, so that
    :math:`L_{W\mathrm{ref,atm}} = L_W + C_1 + C_2` (Eq. G.1) and
    :math:`L_{J\mathrm{ref,atm}} = L_J + C_1 + C_2` (Eq. G.3). :math:`C_1`
    accounts for the different reference quantities of the pressure and power
    decibels through the characteristic impedance of the air, and is omitted
    when :math:`K_2` came from the absolute comparison test of A.2;
    :math:`C_2` is the radiation-impedance correction of a monopole, a mean
    value for other sources. Where the static pressure was not measured it is
    estimated from the altitude of the site by Eq. (G.2),
    :math:`p_\mathrm{s} = p_{\mathrm{s},0}(1 - aH_\mathrm{a})^b`, with
    :math:`a = 2.2560 \times 10^{-5}` m^-1 and :math:`b = 5.2553`.

    :param temperature: Air temperature ``theta`` at the test, in degrees C.
    :param static_pressure: Static pressure ``ps`` at the test, in
        kilopascals; give this or ``altitude``.
    :param altitude: Altitude ``Ha`` of the test site, in metres, from which
        ``ps`` is estimated by Eq. (G.2) when it was not measured.
    :return: :class:`ReferenceAtmosphereCorrection` with ``c1``, ``c2``, their
        ``total`` and the static pressure used.
    :raises ValueError: if neither or both of ``static_pressure`` and
        ``altitude`` are given, or either is out of range, or ``temperature``
        is not above absolute zero.
    """
    if not math.isfinite(temperature) or temperature <= -_KELVIN_OFFSET:
        msg = f"'temperature' must be finite and above {-_KELVIN_OFFSET} degrees C."
        raise ValueError(msg)
    if static_pressure is not None and altitude is not None:
        msg = (
            "Give either 'static_pressure' (kPa, measured) or 'altitude' (m, for "
            "Eq. G.2), not both."
        )
        raise ValueError(msg)
    if altitude is not None:
        ps = _static_pressure_at_altitude(float(altitude))
    elif static_pressure is not None:
        ps = require_positive(static_pressure, "static_pressure")
    else:
        msg = "Give one of 'static_pressure' (kPa) or 'altitude' (m)."
        raise ValueError(msg)
    p_term = -10.0 * math.log10(ps / _PS0_KPA)
    theta_k = _KELVIN_OFFSET + temperature
    c1 = p_term + 5.0 * math.log10(theta_k / _THETA0_K)
    c2 = p_term + 15.0 * math.log10(theta_k / _THETA1_K)
    return ReferenceAtmosphereCorrection(
        c1=float(c1), c2=float(c2), static_pressure=ps, temperature=float(temperature)
    )


def _event_position_levels(
    levels_positions: np.ndarray, events: int | None
) -> tuple[np.ndarray, int | None]:
    """The ``(NM, NB)`` mean single event levels behind the three input forms.

    A 3-D ``(Ne, NM, NB)`` array holds one event per entry of its first axis
    and is reduced by Eq. (19); a 2-D ``(NM, NB)`` array with ``events`` is
    one measurement encompassing that many events and is reduced by Eq. (20);
    a 2-D array without ``events`` is taken as the mean single event level of
    one event at each position, already formed.

    :param levels_positions: The levels, in decibels.
    :param events: ``None``, or the number of events one measurement holds.
    :return: The ``(NM, NB)`` levels and the event count they rest on
        (``None`` when the caller supplied the means).
    :raises ValueError: for a rank other than 2 or 3, non-finite levels, or a
        per-event array given together with ``events``.
    """
    arr = np.asarray(levels_positions, dtype=np.float64)
    if arr.ndim == 3:  # noqa: PLR2004
        if events is not None:
            msg = (
                "'levels_positions' already carries one entry per event on its "
                "first axis; 'events' applies to one measurement encompassing "
                "several events (ISO 3744:2010 Eq. 20), not to per-event levels."
            )
            raise ValueError(msg)
        count = int(arr.shape[0])
        arr = _single_event_mean(arr, None, name="levels_positions", stacklevel=4)
        return arr, count
    if arr.ndim != 2:  # noqa: PLR2004
        msg = (
            "'levels_positions' must be a 2D (positions, bands) array of single "
            "event levels or a 3D (events, positions, bands) array."
        )
        raise ValueError(msg)
    if events is None:
        if not np.all(np.isfinite(arr)):
            msg = "'levels_positions' must contain only finite values."
            raise ValueError(msg)
        return arr, None
    arr = _single_event_mean(arr, events, name="levels_positions", stacklevel=4)
    return arr, int(events)


def sound_energy_pressure(
    levels_positions: np.ndarray,
    surface: Surface,
    *,
    radius: float | None = None,
    dimensions: tuple[float, float, float] | None = None,
    distance: float | None = None,
    reflecting_planes: int = 1,
    events: int | None = None,
    background_levels: np.ndarray | None = None,
    integration_time: float | None = None,
    frequencies: np.ndarray | None = None,
    room: RoomEnvironment | None = None,
    grade: Grade = "engineering",
    omc_uncertainty: float = 0.0,
) -> SoundEnergyResult:
    r"""Sound energy level of a noise burst from surface single event levels
    (ISO 3744:2010 clause 8.3, ISO 3746:2010 clause 8.4).

    The single event time-integrated sound pressure levels
    :math:`L'_{Ei(\mathrm{ST})}` are measured simultaneously at every
    microphone position through a period that encompasses the full burst
    (clause 8.3.1; a traversing microphone is not permitted). ``levels_positions``
    is either the ``(NM, NB)`` mean single event level of one event at each
    position, a ``(Ne, NM, NB)`` array of the :math:`N_\mathrm{e}` events measured
    one at a time (reduced by Eq. 19), or the ``(NM, NB)`` level of one
    measurement encompassing ``events`` successive events (reduced by Eq. 20).
    The positions are then energy-averaged as in 8.2.2 (clause 8.3.3), the
    surface level is corrected for background noise and for the test
    environment and the surface term added:

    .. math::

       L_J = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}}
       \sum_i 10^{0.1 L'_{Ei(\mathrm{ST})}} \right] - K_1 - K_2
       + 10 \log_{10}\frac{S}{S_0} \tag{Eq. 22, 23}

    :math:`K_1` follows Eq. (21) with the same criteria as the time-averaged
    path (:func:`background_noise_correction`: 6 dB to 15 dB engineering, 3 dB
    to 10 dB survey, clamped below the lower criterion with a
    :class:`SoundPowerWarning`), and :math:`K_2` the room data in ``room``
    (Annex A). The background is the time-averaged level the standard has
    measured over the same integration time :math:`T` as the events, and it
    is compared as its exposure over that :math:`T`,
    :math:`L_{p(\mathrm{B})} + 10 \log_{10}(T/T_0)` (clause 3.4 NOTE 1), so
    that the energies Eq. (21) subtracts share one reference; this is why
    ``integration_time`` is required whenever ``background_levels`` is given.
    The surface area ``S`` comes from the geometry exactly as in
    :func:`sound_power_pressure`, and so do the minimum number of positions,
    the A-weighted total (Annex E, Eq. E.2) and the expanded uncertainty,
    which clause 9.1 Eq. (24) makes the same for :math:`L_J` as for
    :math:`L_W`.

    The level holds for the meteorological conditions of the test; above
    500 m of altitude or below 10 degrees C add the Annex G correction of
    :func:`reference_atmosphere_correction` (Eq. G.3).

    :param levels_positions: Single event levels, in decibels: ``(NM, NB)``
        means, ``(Ne, NM, NB)`` per-event levels, or ``(NM, NB)`` of one
        measurement of ``events`` events.
    :param surface: ``'hemisphere'`` or ``'box'``.
    :param radius: Hemisphere radius ``r`` (metres), for ``surface='hemisphere'``.
    :param dimensions: Reference box ``(l1, l2, l3)`` (metres), for ``'box'``.
    :param distance: Measurement distance ``d`` (metres), for ``'box'``.
    :param reflecting_planes: Number of reflecting planes (1, 2 or 3).
    :param events: The number of events ``Ne`` one measurement encompasses
        (Eq. 20); ``None`` when ``levels_positions`` is per event or already
        the mean of one event.
    :param background_levels: ``(NM, NB)`` time-averaged background levels for
        ``K1``, or a single spectrum ``(NB,)`` / ``(1, NB)`` broadcast to every
        position; measured over the same interval as the events.
    :param integration_time: The interval ``T`` of the single event levels, in
        seconds; required with ``background_levels``.
    :param frequencies: Band mid-band frequencies (Hz) for the A-weighted total.
    :param room: Room absorption data behind ``K2`` (:class:`RoomEnvironment`);
        ``None`` is a room with no data at all, i.e. a free field
        (:math:`K_2 = 0`).
    :param grade: ``'engineering'`` (ISO 3744) or ``'survey'`` (ISO 3746).
    :param omc_uncertainty: ``sigma_omc`` (dB), operating/mounting instability.
    :return: :class:`SoundEnergyResult`.
    :raises ValueError: for a malformed or non-finite level array, a
        geometry that does not describe the surface, too few positions, a
        background without its ``integration_time``, or a mismatched
        ``frequencies`` length.
    """
    grade = _check_grade(grade)
    omc_uncertainty = require_non_negative(omc_uncertainty, "omc_uncertainty")
    levels, event_count = _event_position_levels(levels_positions, events)
    if levels.shape[1] == 0:
        msg = "'levels_positions' must contain at least one frequency band."
        raise ValueError(msg)
    n_positions = levels.shape[0]
    if integration_time is not None:
        integration_time = require_positive(integration_time, "integration_time")

    # --- measurement surface area (clause 7.2, as for LW) -----------------
    if reflecting_planes not in (1, 2, 3):
        msg = _REFLECTING_PLANES_MSG
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

    # --- mean over the surface (8.3.3, as Eq. 12) and K1 (Eq. 21) ---------
    mean_level = energy_mean(levels, axis=0)
    n_bands = mean_level.shape[0]
    if background_levels is None:
        k1 = np.zeros(n_bands, dtype=np.float64)
    else:
        exposure = _background_exposure(background_levels, integration_time)
        k1 = _surface_background_correction(exposure, levels, mean_level, grade)

    # --- environmental correction K2 (Annex A, as for LW) -----------------
    k2 = _surface_environmental_correction(
        area, n_bands, grade, RoomEnvironment() if room is None else room
    )

    # --- surface single event level (Eq. 22), LJ (Eq. 23), LJA (Eq. E.2) --
    surface_level = mean_level - k1 - k2
    lj = surface_level + 10.0 * np.log10(area / _S0)
    freqs, lja = _a_weighted_total(lj, frequencies, n_bands)

    # The same per-band K1 corrects the position level and the surface mean,
    # so the apparent directivity index reduces to the raw difference exactly
    # as it does for the time-averaged levels (clause 3.24, 8.4).
    directivity = np.asarray(levels - mean_level[np.newaxis, :], dtype=np.float64)

    uncertainty = _COVERAGE_FACTOR_95 * float(
        np.hypot(_SIGMA_R0[grade], omc_uncertainty)
    )

    return SoundEnergyResult(
        frequencies=freqs,
        sound_energy_level=np.asarray(lj, dtype=np.float64),
        surface_event_level=np.asarray(surface_level, dtype=np.float64),
        mean_event_level=mean_level,
        background_correction=k1,
        environmental_correction=k2,
        directivity_index=directivity,
        surface_area=float(area),
        sound_energy_level_a=lja,
        uncertainty=uncertainty,
        grade=grade,
        events=event_count,
        integration_time=integration_time,
    )
