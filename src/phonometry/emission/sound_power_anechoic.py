#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound power level of a noise source in an anechoic or hemi-anechoic room, from
sound pressure measurements over an enveloping surface: ISO 3745:2012
(precision, accuracy grade 1).

This is the precision sibling of ISO 3744:2010 (engineering) and ISO 3746:2010
(survey), and the path from the surface-averaged pressure to the sound power
level is the same one. What sets it apart is the room. In a qualified
(hemi-)free field there is no reflected energy to correct for, so the
environmental correction :math:`K_2` disappears altogether; in its place the
standard asks for more of everything else. The background correction is
evaluated at every microphone position and every band against a
frequency-dependent criterion (clause 9.4.2), the microphone array is a fixed
40-position equal-area set tabulated for the sphere (Annex D) and the
hemisphere (Annex E), and three meteorological corrections refer the
determination to the reference atmosphere (clause 9.5):

.. math::

   K_{1i} = -10 \log_{10}\!\left( 1 - 10^{-0.1 \Delta L_{pi}} \right)
   \tag{Eq. 11}

   \overline{L_p} = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}}
   \sum_i 10^{0.1 (L'_{pi} - K_{1i})} \right] \tag{Eq. 12}

   L_W = \overline{L_p} + 10 \log_{10}\frac{S}{S_0} + C_1 + C_2 + C_3
   \tag{Eq. 14/15}

The measurement surface is a full sphere :math:`S = 4\pi r^2` in an anechoic
room (Eq. 14) or a hemisphere :math:`S = 2\pi r^2` over the reflecting plane
of a hemi-anechoic room (Eq. 15), and each tabulated position carries an equal
share of it; unequal partial areas :math:`S_i` are averaged by Eq. 13 instead.
The A-weighted total is combined with the ISO 3744 Annex E band corrections
(Annex C, Eq. C.1) and the expanded uncertainty
:math:`U = k\sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}` (Eq. 24/25) takes its
reproducibility standard deviation from Table 3 (anechoic) or Table 2
(hemi-anechoic).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from .._internal.levels_math import energy_mean, energy_sum, weighted_energy_mean
from .._internal.types import as_float_or_array
from ._shared import _S0, SoundPowerWarning, _a_weighting_corrections

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

PrecisionSurface = Literal["sphere", "hemisphere"]
PrecisionArray = Literal["general", "broadband"]
PrecisionRoom = Literal["anechoic", "hemi-anechoic"]

_SURFACE_CHOICE_MSG = "'surface' must be 'sphere' or 'hemisphere'."

# --- ISO 3745:2012 Annex D/E, normative microphone coordinates (x/r,y/r,z/r) -
# Digit-exact, image-verified transcriptions. Positions 1-20 are the primary
# array; positions 21-40 (the mirror set) are added when the band-SPL spread
# exceeds NM/2 (clause 9.3.2/9.3.3). Every position carries an equal surface
# area (S/40). z points up from the horizontal plane z = 0.

#: Table D.1 - sphere, anechoic room (Annex D). 40 positions.
_TABLE_D1: np.ndarray = np.array(
    [
        [-0.999, 0.0, 0.050],
        [0.494, -0.856, 0.150],
        [0.484, 0.839, 0.250],
        [-0.468, 0.811, 0.350],
        [-0.447, -0.773, 0.450],
        [0.835, 0.0, 0.550],
        [0.380, 0.658, 0.650],
        [-0.661, 0.0, 0.750],
        [0.263, -0.456, 0.850],
        [0.312, 0.0, 0.950],
        [0.999, 0.0, -0.050],
        [-0.494, 0.856, -0.150],
        [-0.484, -0.839, -0.250],
        [0.468, -0.811, -0.350],
        [0.447, 0.773, -0.450],
        [-0.835, 0.0, -0.550],
        [-0.380, -0.658, -0.650],
        [0.661, 0.0, -0.750],
        [-0.263, 0.456, -0.850],
        [-0.312, 0.0, -0.950],
        [0.999, 0.0, 0.050],
        [-0.494, -0.856, 0.150],
        [-0.484, 0.839, 0.250],
        [0.468, 0.811, 0.350],
        [0.447, -0.773, 0.450],
        [-0.835, 0.0, 0.550],
        [-0.380, 0.658, 0.650],
        [0.661, 0.0, 0.750],
        [-0.263, -0.456, 0.850],
        [-0.312, 0.0, 0.950],
        [-0.999, 0.0, -0.050],
        [0.494, 0.856, -0.150],
        [0.484, -0.839, -0.250],
        [-0.468, -0.811, -0.350],
        [-0.447, 0.773, -0.450],
        [0.835, 0.0, -0.550],
        [0.380, -0.658, -0.650],
        [-0.661, 0.0, -0.750],
        [0.263, 0.456, -0.850],
        [0.312, 0.0, -0.950],
    ]
)
#: Table E.1 - hemisphere, general case (Annex E). 40 positions. z/r at pos
#: 7/27 is 0.320 (not 0.325), verified against the source image.
_TABLE_E1: np.ndarray = np.array(
    [
        [-1.000, 0.000, 0.025],
        [0.499, -0.864, 0.075],
        [0.496, 0.859, 0.125],
        [-0.492, 0.853, 0.175],
        [-0.487, -0.844, 0.225],
        [0.961, 0.000, 0.275],
        [0.000, 0.947, 0.320],
        [-0.803, -0.464, 0.375],
        [0.784, -0.453, 0.425],
        [0.762, 0.440, 0.475],
        [-0.737, 0.426, 0.525],
        [0.000, -0.818, 0.575],
        [0.781, 0.000, 0.625],
        [-0.369, 0.639, 0.675],
        [-0.344, -0.596, 0.725],
        [0.316, -0.547, 0.775],
        [0.283, 0.489, 0.825],
        [-0.484, 0.000, 0.875],
        [0.000, -0.380, 0.925],
        [0.192, 0.111, 0.975],
        [1.000, 0.000, 0.025],
        [-0.499, 0.864, 0.075],
        [-0.496, -0.859, 0.125],
        [0.492, -0.853, 0.175],
        [0.487, 0.844, 0.225],
        [-0.961, 0.000, 0.275],
        [0.000, -0.947, 0.320],
        [0.803, 0.464, 0.375],
        [-0.784, 0.453, 0.425],
        [-0.762, -0.440, 0.475],
        [0.737, -0.426, 0.525],
        [0.000, 0.818, 0.575],
        [-0.781, 0.000, 0.625],
        [0.369, -0.639, 0.675],
        [0.344, 0.596, 0.725],
        [-0.316, 0.547, 0.775],
        [-0.283, -0.489, 0.825],
        [0.484, 0.000, 0.875],
        [0.000, 0.380, 0.925],
        [-0.192, -0.111, 0.975],
    ]
)
#: Table E.2 - hemisphere, broadband omnidirectional source (Annex E). 40
#: positions. Pos 19 x/r is -0.380 (a normal negative), verified.
_TABLE_E2: np.ndarray = np.array(
    [
        [-1.000, 0.000, 0.025],
        [0.499, -0.864, 0.075],
        [0.496, 0.859, 0.125],
        [-0.492, 0.853, 0.175],
        [-0.487, -0.844, 0.225],
        [0.961, 0.000, 0.275],
        [0.474, 0.820, 0.325],
        [-0.927, 0.000, 0.375],
        [0.453, -0.784, 0.425],
        [0.880, 0.000, 0.475],
        [-0.426, 0.737, 0.525],
        [-0.409, -0.709, 0.575],
        [0.390, -0.676, 0.625],
        [0.369, 0.639, 0.675],
        [-0.689, 0.000, 0.725],
        [-0.316, -0.547, 0.775],
        [0.565, 0.000, 0.825],
        [-0.242, 0.419, 0.875],
        [-0.380, 0.000, 0.925],
        [0.111, -0.192, 0.975],
        [1.000, 0.000, 0.025],
        [-0.499, 0.864, 0.075],
        [-0.496, -0.859, 0.125],
        [0.492, -0.853, 0.175],
        [0.487, 0.844, 0.225],
        [-0.961, 0.000, 0.275],
        [-0.474, -0.820, 0.325],
        [0.927, 0.000, 0.375],
        [-0.453, 0.784, 0.425],
        [-0.880, 0.000, 0.475],
        [0.426, -0.737, 0.525],
        [0.409, 0.709, 0.575],
        [-0.390, 0.676, 0.625],
        [-0.369, -0.639, 0.675],
        [0.689, 0.000, 0.725],
        [0.316, 0.547, 0.775],
        [-0.565, 0.000, 0.825],
        [0.242, -0.419, 0.875],
        [0.380, 0.000, 0.925],
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
    :math:`U = k\sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}`
    (Eq. 24/25) and ``uncertainty_bands`` the
    per-band value (``NaN`` without ``frequencies``).
    ``sound_power_level_a`` is the A-weighted total ``LWA`` (Eq. C.1).
    """

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

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
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
            msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            raise ValueError(msg)
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
        raise ValueError(_SURFACE_CHOICE_MSG)
    norms = np.linalg.norm(table, axis=1)
    if np.any(np.abs(norms - 1.0) > _UNIT_NORM_TOL):
        msg = (
            "Microphone coordinate table is not a set of unit vectors within "
            f"{_UNIT_NORM_TOL:g}; a transcription error is present."
        )
        raise ValueError(msg)
    return table


def precision_positions(
    surface: PrecisionSurface,
    *,
    radius: float,
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
        raise ValueError(_SURFACE_CHOICE_MSG)
    if array not in ("general", "broadband"):
        msg = "'array' must be 'general' or 'broadband'."
        raise ValueError(msg)
    if radius <= 0:
        msg = "A positive 'radius' is required."
        raise ValueError(msg)
    if count not in (20, 40):
        msg = "'count' must be 20 (primary array) or 40 (full array)."
        raise ValueError(msg)
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
        msg = (
            "The last axis of 'source_levels'/'background_levels' must match "
            "the number of 'frequencies'."
        )
        raise ValueError(msg)
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

    Using the measured static pressure :math:`p_\mathrm{s}` (kPa) and air temperature
    :math:`\theta` (deg C) form:

    .. math::

       C_1 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}}
       + 5 \log_{10}\frac{273 + \theta}{\theta_0},
       \qquad \theta_0 = 314~\text{K}

       C_2 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}}
       + 15 \log_{10}\frac{273 + \theta}{\theta_1},
       \qquad \theta_1 = 296~\text{K}

       C_3 = A_0 \left( 1.0053 - 0.0012\,A_0 \right)^{1.6},
       \qquad A_0 = \alpha(f)\,r

    :math:`p_{\mathrm{s},0} = 101.325` kPa. This is the :math:`p_\mathrm{s}`/:math:`\theta`
    form of C1 (not the characteristic-impedance form), chosen because it
    needs only the measured :math:`p_\mathrm{s}` and :math:`\theta` and is consistent
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
        msg = "'static_pressure' must be positive (kPa)."
        raise ValueError(msg)
    if temperature <= -273.0:
        msg = "'temperature' must be above -273 degrees Celsius."
        raise ValueError(msg)
    if radius <= 0.0:
        msg = "'radius' must be positive."
        raise ValueError(msg)
    theta_k = 273.0 + temperature
    p_term = -10.0 * np.log10(static_pressure / _PS0_KPA)
    c1 = float(p_term + 5.0 * np.log10(theta_k / _THETA0_K))
    c2 = float(p_term + 15.0 * np.log10(theta_k / _THETA1_K))
    if air_absorption_coefficient is None:
        c3: float | np.ndarray = 0.0
    else:
        a0 = np.asarray(air_absorption_coefficient, dtype=np.float64) * radius
        if np.any(a0 < 0.0):
            msg = "'air_absorption_coefficient' must be non-negative."
            raise ValueError(msg)
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
    msg = (
        f"No ISO 3745:2012 sigma_R0 for {nominal} Hz; expected a nominal "
        "one-third-octave mid-band from 50 Hz to 20 kHz."
    )
    raise ValueError(msg)


def precision_uncertainty(
    sigma_r0: float | np.ndarray,
    sigma_omc: float = 0.0,
    coverage_factor: float = 2.0,
) -> float | np.ndarray:
    r"""Expanded uncertainty :math:`U = k \sigma_\mathrm{tot}` (ISO 3745:2012).

    ISO 3745:2012 Eq. 24/25:
    :math:`\sigma_\mathrm{tot} = \sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}` and
    :math:`U = k \sigma_\mathrm{tot}`, with :math:`k = 2` (95 %, two-sided) or
    :math:`k = 1.6` (95 %, one-sided, when comparing to a limit).

    :param sigma_r0: Reproducibility standard deviation (Tables 2/3), dB.
    :param sigma_omc: Operating/mounting standard deviation ``sigma_omc``, dB.
    :param coverage_factor: ``k`` (typically 2 or 1.6).
    :return: ``U`` in decibels, scalar or per band matching ``sigma_r0``.
    """
    if coverage_factor <= 0.0:
        msg = "'coverage_factor' must be positive."
        raise ValueError(msg)
    if sigma_omc < 0.0:
        msg = "'sigma_omc' must be non-negative."
        raise ValueError(msg)
    sigma_tot = np.hypot(np.asarray(sigma_r0, dtype=np.float64), sigma_omc)
    u = coverage_factor * sigma_tot
    return as_float_or_array(u)


def _precision_k1(
    levels: np.ndarray,
    background_levels: np.ndarray | None,
    freqs: np.ndarray | None,
) -> np.ndarray:
    """Per-position background correction K1i (Eq. 11), zero without background.

    A single background spectrum is broadcast over the microphone positions;
    any other shape mismatch is rejected.
    """
    if background_levels is None:
        return np.zeros_like(levels)
    if freqs is None:
        msg = (
            "'frequencies' are required with 'background_levels' to select "
            "the frequency-dependent K1 criterion (ISO 3745:2012 9.4.2)."
        )
        raise ValueError(msg)
    n_positions, n_bands = levels.shape
    bg = np.atleast_2d(np.asarray(background_levels, dtype=np.float64))
    if bg.shape == (1, n_bands) and n_positions != 1:
        bg = np.broadcast_to(bg, (n_positions, n_bands))
    if bg.shape != levels.shape:
        msg = (
            "'background_levels' must match 'levels_positions' shape, or be "
            "a single spectrum of shape (NB,) or (1, NB)."
        )
        raise ValueError(msg)
    return precision_background_correction(levels, bg, freqs)


def _precision_surface_level(
    corrected: np.ndarray, areas: np.ndarray | None
) -> np.ndarray:
    """Surface time-averaged level Lp_bar: equal-area Eq. 12 or area-weighted
    Eq. 13 when partial areas ``Si`` are supplied.
    """
    if areas is None:
        return np.asarray(energy_mean(corrected, axis=0), dtype=np.float64)
    n_positions = corrected.shape[0]
    seg = np.asarray(areas, dtype=np.float64)
    if seg.shape != (n_positions,):
        msg = "'areas' must have one value per microphone position."
        raise ValueError(msg)
    if np.any(seg <= 0.0):
        msg = "All 'areas' must be positive."
        raise ValueError(msg)
    return np.asarray(
        weighted_energy_mean(corrected, seg[:, None], axis=0), dtype=np.float64
    )


def _precision_a_weighted_total(lw: np.ndarray, freqs: np.ndarray | None) -> float:
    """A-weighted total LWA (Eq. C.1), NaN where it is not defined."""
    if freqs is None:
        return float(lw[0]) if lw.shape[0] == 1 else float("nan")
    try:
        ck = _a_weighting_corrections(freqs)
    except ValueError:
        # Bands outside the ISO 3744 Annex E table (e.g. the 12,5 kHz to
        # 20 kHz one-third octaves that ISO 3745 covers, Tables 2/3): the
        # per-band determination stands, only the A-weighted total is
        # undefined.
        return float("nan")
    return float(energy_sum(lw + ck))


def _precision_uncertainty_bands(
    freqs: np.ndarray | None,
    n_bands: int,
    room: PrecisionRoom,
    sigma_omc: float,
    coverage_factor: float,
) -> np.ndarray:
    """Per-band expanded uncertainty (Eq. 24/25), NaN without band centres."""
    if freqs is None:
        return np.full(n_bands, np.nan, dtype=np.float64)
    sigma_bands = np.array(
        [_sigma_r0_3745(round(float(f)), room) for f in freqs],
        dtype=np.float64,
    )
    return np.asarray(
        precision_uncertainty(sigma_bands, sigma_omc, coverage_factor),
        dtype=np.float64,
    )


@overload
def sound_power_anechoic(
    levels_positions: np.ndarray,
    surface: PrecisionSurface,
    *,
    radius: float,
    background_levels: np.ndarray,
    frequencies: np.ndarray,
    areas: np.ndarray | None = ...,
    temperature: float = ...,
    static_pressure: float = ...,
    air_absorption_coefficient: float | np.ndarray | None = ...,
    sigma_omc: float = ...,
    coverage_factor: float = ...,
) -> PrecisionSoundPowerResult: ...


@overload
def sound_power_anechoic(
    levels_positions: np.ndarray,
    surface: PrecisionSurface,
    *,
    radius: float,
    frequencies: np.ndarray | None = ...,
    areas: np.ndarray | None = ...,
    temperature: float = ...,
    static_pressure: float = ...,
    air_absorption_coefficient: float | np.ndarray | None = ...,
    sigma_omc: float = ...,
    coverage_factor: float = ...,
) -> PrecisionSoundPowerResult: ...


def sound_power_anechoic(
    levels_positions: np.ndarray,
    surface: PrecisionSurface,
    *,
    radius: float,
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

       L_W = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}}
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
        raise ValueError(_SURFACE_CHOICE_MSG)
    if radius <= 0:
        msg = "A positive 'radius' is required."
        raise ValueError(msg)
    levels = np.atleast_2d(np.asarray(levels_positions, dtype=np.float64))
    if levels.ndim != 2:
        msg = "'levels_positions' must be a 2D (positions, bands) array."
        raise ValueError(msg)
    n_positions, n_bands = levels.shape

    area = (4.0 if surface == "sphere" else 2.0) * np.pi * radius**2
    room: PrecisionRoom = "anechoic" if surface == "sphere" else "hemi-anechoic"

    freqs = None if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    if freqs is not None and freqs.shape[0] != n_bands:
        msg = "'frequencies' length must match the number of bands."
        raise ValueError(msg)

    # --- per-position background correction K1i (Eq. 11) ------------------
    k1 = _precision_k1(levels, background_levels, freqs)

    corrected = levels - k1  # Lpi = L'pi(ST) - K1i

    # --- surface time-averaged level Lp_bar (Eq. 12 equal / Eq. 13 area) --
    mean_level = energy_mean(levels, axis=0)
    lp_bar = _precision_surface_level(corrected, areas)

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
    lwa = _precision_a_weighted_total(lw, freqs)

    # --- directivity (Eq. 21) and non-uniformity (Eq. 22) indices ---------
    directivity = np.asarray(corrected - lp_bar[np.newaxis, :], dtype=np.float64)
    if n_positions > 1:
        lp_av = np.mean(corrected, axis=0)  # arithmetic mean (Eq. 22)
        vir = np.sqrt(
            np.sum((corrected - lp_av[np.newaxis, :]) ** 2, axis=0) / (n_positions - 1)
        )
    else:
        vir = np.zeros(n_bands, dtype=np.float64)

    # --- uncertainty (Eq. 24/25) ------------------------------------------
    u_a = float(precision_uncertainty(_SIGMA_R0_3745_A, sigma_omc, coverage_factor))
    u_bands = _precision_uncertainty_bands(
        freqs, n_bands, room, sigma_omc, coverage_factor
    )

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
