#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound power level of a noise source measured in a reverberation test room:
ISO 3741:2010 (precision method, accuracy grade 1).

The source is placed in a hard-walled reverberation room whose reverberant
field is sampled by microphones. Two methods are provided.

The **direct method** derives the sound power from the mean corrected room
sound pressure level ``Lp(ST)`` and the equivalent absorption area ``A`` of the
room, with the Sabine absorption area and the speed of sound ``c`` in m/s
(ISO 3741:2010 clause 9.1.4, Eq. 20):

.. math::

   L_p(\text{ST}) = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}} \sum_i 10^{0.1 L_{pi}}
   \right] \tag{Eq. 16}

   A = \frac{55.26}{c} \, \frac{V}{T_{60}}

   c = 20.05 \sqrt{273 + \theta}

   L_W = L_p(\text{ST}) + 10 \log_{10}\frac{A}{A_0} + 4.34 \frac{A}{S}
   + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6
   \tag{Eq. 20}

:math:`10 \log_{10}(1 + Sc/(8Vf))` is the Waterhouse boundary correction (energy
stored near the room boundaries); it vanishes as the frequency grows. ``C1``
(Eq. 20, reference-quantity correction) and ``C2`` (radiation-impedance
correction) carry the result to the reference meteorological conditions of
clause 4 (23.0 C, 101.325 kPa, 50 %), per clause 9.1.4:

.. math::

   C_1 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s}0}} + 5 \log_{10}\frac{273.15 + \theta}{314}

   C_2 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s}0}} + 15 \log_{10}\frac{273.15 + \theta}{296}

The **comparison method** replaces the absorption-area terms by a reference
sound source (RSS) of known sound power ``LW(RSS)`` measured at the same
positions (ISO 3741:2010 clause 9.1.5, Eq. 21):

.. math::

   L_W = L_W(\text{RSS}) + \left( L_p(\text{ST}) - L_p(\text{RSS}) + C_2
   \right) \tag{Eq. 21}

Both methods cover the one-third-octave bands from 100 Hz to 10 kHz (clause
8.1). Octave-band, A-weighted and total levels follow ISO 3741 Annex F, which
reuses the ISO 3744 Annex E A-weighting band corrections.

A noise burst or a transient emission is described by the **sound energy
level** :math:`L_J = 10 \log_{10}(J/J_0)`, :math:`J_0 = 1` pJ (clause 3.18),
and clause 9.2 determines it by the same two methods with the single event
time-integrated sound pressure level :math:`L_E` (clause 3.4) in place of the
time-averaged :math:`L_p`: the :math:`N_\mathrm{e}` events at each position
are reduced to the level of one event (Eq. 22 or Eq. 23), each position is
corrected for its background by :math:`K_{1i}` (Eq. 25, 26) as in 9.1.2, the
positions are energy-averaged (Eq. 27), and the room enters through the same
bracket as Eq. (20) or the same reference source as Eq. (21):

.. math::

   L_J = \overline{L_E(\text{ST})} + \left[ 10 \log_{10}\frac{A}{A_0}
   + 4.34 \frac{A}{S} + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right)
   + C_1 + C_2 - 6 \right] \tag{Eq. 30}

   L_J = L_W(\text{RSS}) + \left( \overline{L_E(\text{ST})}
   - \overline{L_p(\text{RSS})} \right) + C_2 \tag{Eq. 31}

For a source steady over the whole interval :math:`T`, clause 3.4 NOTE 1
gives :math:`L_E = L_{p,T} + 10 \log_{10}(T/T_0)`, :math:`T_0 = 1` s, and
so :math:`L_J = L_W + 10 \log_{10}(T/T_0)`. Annex F sums the one-third-octave
levels into octave bands (Eq. F.1, F.4) and A-weights them (Eq. F.2, F.5)
alike for the two quantities.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from .._internal.levels_math import energy_mean, energy_sum
from .._internal.validation import (
    check_engine,
    require_choice,
    require_positive,
    require_ranks,
    require_same_length,
)
from ._shared import (
    _PS0,
    SoundPowerWarning,
    _a_weighting_corrections,
    _background_exposure,
    _c2_correction,
    _single_event_mean,
    _validate_meteorology,
)

_A0 = 1.0  #: Reference absorption area, in square metres (ISO 3741, Eq. 20).
_THETA0 = 314.0  #: Reference temperature for C1, in K (ISO 3741 clause 9.1.4).
#: K1 qualification edges: bands at or below the low edge or at or above the
#: high edge carry the relaxed 6 dB lower criterion (ISO 3741:2010, 9.1.2).
_K1_EDGE_LOW_HZ = 200.0
_K1_EDGE_HIGH_HZ = 6300.0
#: Background margin at or above which the background is negligible and K1 = 0
#: (ISO 3741:2010, 9.1.2).
_K1_UPPER_DB = 15.0
#: The T60 > V/S room-absorptivity criterion covers bands below 6,3 kHz
#: (ISO 3741:2010, 5.3, Eq. 7); distinct from the K1 edge above.
_ABSORPTION_CRITERION_MAX_HZ = 6300.0
#: Minimum microphone positions in an unqualified room (ISO 3741:2010, 8.3, 8.4.1).
_MIN_MIC_POSITIONS = 6
#: The inter-position sample deviation sM (ddof=1) needs at least two positions.
_MIN_POSITIONS_FOR_SM = 2
#: Maximum admissible inter-position deviation sM (ISO 3741:2010, 8.4.2.2, Eq. 10).
_SM_CRITERION_DB = 1.5
#: The diffuse-field constant of Eq. (20)/(30), in dB: the -6 dB that turns
#: the mean-square pressure and the Sabine absorption area into a power against
#: the two reference quantities (ISO 3741:2010, 9.1.4).
_DIFFUSE_FIELD_CONSTANT_DB = -6.0
#: The Sabine constant 55,26 of A = (55,26/c)(V/T60) and the 4,34 = 10 lg e of
#: the first-order Eyring term (ISO 3741:2010, 9.1.4).
_SABINE_CONSTANT = 55.26
_EYRING_TERM_DB = 4.34
#: Table F.1 of ISO 3741:2010 (= ISO 3744 Table E.1): the nominal one-third-
#: octave mid-band frequencies, k = 1 (50 Hz) to k = 24 (10 kHz), grouped
#: three per octave so that octave i = 1 (63 Hz) to 8 (8 kHz) is the sum over
#: k = 3i-2 to 3i (Eq. F.1/F.4).
_THIRD_OCTAVE_NOMINAL: tuple[int, ...] = (
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
)  # fmt: skip
_OCTAVE_NOMINAL: tuple[int, ...] = (63, 125, 250, 500, 1000, 2000, 4000, 8000)
_THIRDS_PER_OCTAVE = 3


@dataclass(frozen=True)
class ReverberationSoundPowerResult:
    r"""Result of an ISO 3741:2010 reverberation-room sound power determination.

    ``sound_power_level`` is the per-band ``LW`` (Eq. 20 direct method, Eq. 21
    comparison method). ``mean_pressure_level`` is the mean corrected room level
    ``Lp(ST)`` (Eq. 16). For the direct method ``absorption_area`` is the Sabine
    equivalent absorption area ``A`` per band and ``waterhouse_correction`` the
    boundary term :math:`10 \log_{10}(1 + Sc/(8Vf))`; both are ``NaN`` for the
    comparison method. ``background_correction`` is the effective per-band
    background correction ``K1``: with per-position input each position is
    corrected by its own ``K1i`` (Eq. 14/15) before the energy average
    (Eq. 16), and the reported value is the resulting per-band shift of the
    mean level (zero when no background is supplied).
    ``c1`` and ``c2`` are the reference-quantity and radiation-impedance
    corrections (``c1`` is ``NaN`` for the comparison method, which uses only
    ``c2``). ``speed_of_sound`` is ``c`` at the test temperature.
    ``sound_power_level_a`` is the A-weighted total ``LWA`` (Annex F Eq. F.2),
    computed only when ``frequencies`` are supplied (``NaN`` for several bands
    without them; equal to ``LW`` for a single band). ``method`` is ``'direct'``
    or ``'comparison'``.
    """

    frequencies: np.ndarray | None
    sound_power_level: np.ndarray
    mean_pressure_level: np.ndarray
    absorption_area: np.ndarray
    waterhouse_correction: np.ndarray
    background_correction: np.ndarray
    c1: float
    c2: float
    speed_of_sound: float
    sound_power_level_a: float
    method: str

    def __post_init__(self) -> None:
        """Reject a determination whose per-band quantities disagree.

        The fiche prints one row per band under a single header, and the row
        count comes from ``sound_power_level`` alone: every other column
        (``Lp``, and ``K1``, ``A``, ``Cw`` in the verbose table) is indexed
        alongside it, and ``frequencies`` names the bands those rows are
        labelled with. A quantity one entry short therefore cannot be read at
        all, raising a bare ``IndexError`` from inside the row loop, and one
        entry too long is dropped by the table without a word: the boxed
        ``LWA`` and the total beneath it come from ``LW``, so nothing on the
        sheet is summed over the surplus and the fiche renders whole.

        ``method`` is pinned beside the shapes because the whole fiche
        dispatches on it: a tag that is not exactly ``'direct'`` or
        ``'comparison'`` would render a comparison measurement under the
        direct method's basis line and equation, with the comparison result's
        NaN ``c1`` printed as a correction on the accredited sheet.

        :raises ValueError: if any per-band quantity disagrees with the rest,
            or ``method`` is neither ``'direct'`` nor ``'comparison'``.
        """
        require_choice(self.method, "method", ("direct", "comparison"))
        require_ranks(
            self,
            frequencies=1,
            sound_power_level=1,
            mean_pressure_level=1,
            absorption_area=1,
            waterhouse_correction=1,
            background_correction=1,
        )
        require_same_length(
            self,
            "frequencies",
            "sound_power_level",
            "mean_pressure_level",
            "absorption_area",
            "waterhouse_correction",
            "background_correction",
        )

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
        """Render an ISO 3741 reverberation-room sound-power determination fiche.

        Writes a one-page sound-power test sheet: the standard-basis line naming
        the reverberation-room method (the direct method using the room
        equivalent absorption area, or the comparison method using a reference
        sound source) and the precision accuracy grade (ISO 3741:2010, grade 1),
        an optional metadata header (client, noise source, test environment,
        instrumentation, climate, date), a per-band table (nominal octave/
        one-third-octave frequency, the mean room sound-pressure level ``Lp``
        and the band sound-power level ``LW``), the sound-power spectrum
        ``LW(f)`` with a nominal band axis, the boxed A-weighted sound power
        level ``LWA`` (dB re 1 pW) with the total ``LW`` and the determination
        method, an optional verdict row against a declared limit, and a
        measurement-basis strip stating the correction model (Eq. 20 direct or
        Eq. 21 comparison), the applied meteorological corrections ``C1``/``C2``
        and the speed of sound.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the noise source, ``test_room``
            the reverberation test room, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``), the footer
            identity (``laboratory``, ``operator``, ``report_id``, ``notes``)
            and, via ``requirement``, a declared A-weighted sound-power limit
            the fiche checks the result against (lower is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds the background
            correction ``K1`` and, for the direct method, the equivalent
            absorption area ``A`` and the Waterhouse boundary correction ``Cw``.
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
        from .._report.iso3741 import render_reverberation_power_report

        return render_reverberation_power_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _speed_of_sound(temperature: float) -> float:
    r"""Speed of sound :math:`c = 20.05 \sqrt{273 + \theta}` (ISO 3741,
    clause 9.1.4).
    """
    return float(20.05 * np.sqrt(273.0 + temperature))


def _c1_correction(temperature: float, static_pressure: float) -> float:
    """Reference-quantity correction ``C1`` (ISO 3741:2010 clause 9.1.4)."""
    return float(
        -10.0 * np.log10(static_pressure / _PS0)
        + 5.0 * np.log10((273.15 + temperature) / _THETA0)
    )


def _mean_level(levels: np.ndarray) -> np.ndarray:
    """Energy mean over microphone positions (rows), returning one value per
    band. A 1D input is treated as a single averaged spectrum (ISO 3741
    Eq. 16).
    """
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:  # noqa: PLR2004
        return energy_mean(arr, axis=0)
    msg = "'levels' must be a 1D spectrum or a 2D (positions, bands) array."
    raise ValueError(msg)


def _k1_eq14(delta: np.ndarray, frequencies: np.ndarray) -> tuple[np.ndarray, bool]:
    r"""Background-noise correction ``K1`` from ``dLp`` (ISO 3741:2010 Eq. 14).

    :math:`K_1 = -10 \log_{10}(1 - 10^{-0.1 \Delta L_p})`. The precision-grade
    qualification is frequency dependent (clause 9.1.2):
    :math:`\Delta L_p \ge 15` dB gives :math:`K_1 = 0`; below
    the lower criterion (6 dB for bands <= 200 Hz and >= 6 300 Hz, 10 dB for
    250 Hz to 5 000 Hz) ``K1`` is clamped to the criterion value (1.26 dB /
    0.46 dB) and the levels become upper bounds. ``delta`` may be per band
    ``(NB,)`` or per position and band ``(NM, NB)``; the second returned value
    flags whether any element fell below the lower criterion.
    """
    low = np.where(
        (frequencies <= _K1_EDGE_LOW_HZ) | (frequencies >= _K1_EDGE_HIGH_HZ),
        6.0,
        10.0,
    )
    clamped = np.maximum(delta, low)
    k1 = -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * clamped))
    k1 = np.where(delta >= _K1_UPPER_DB, 0.0, k1)
    return np.asarray(k1, dtype=np.float64), bool(np.any(delta < low))


def _background_corrected_mean(
    levels: np.ndarray,
    background_levels: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Background-corrected mean room level and effective per-band ``K1``.

    With per-position 2D ``levels`` the correction follows ISO 3741:2010
    clauses 9.1.2/9.1.3 exactly: the correction ``K1i`` is computed at each
    microphone position (Eq. 14), each position level is corrected first
    (:math:`L_{pi}(\text{ST}) = L'_{pi}(\text{ST}) - K_{1i}`, Eq. 15) and the
    corrected levels are then
    energy-averaged (Eq. 16). ``background_levels`` may be per position
    (matching shape) or a single ``(NB,)`` spectrum used at every position.
    The returned per-band ``K1`` is the effective correction (uncorrected mean
    minus corrected mean).

    With 1D pre-averaged ``levels`` the per-position information is gone, so a
    single ``K1`` is computed from the averaged spectra -- an approximation of
    the per-position procedure that is exact only when every position has the
    same source-to-background margin. Prefer per-position input.
    """
    arr = np.asarray(levels, dtype=np.float64)
    raw_mean = _mean_level(levels)
    if arr.ndim == 2:  # noqa: PLR2004
        bg = np.asarray(background_levels, dtype=np.float64)
        # Checked before the broadcast, not after it. A 1-D spectrum of the
        # wrong length used to die inside `np.broadcast_to`, whose message
        # reports two shapes from inside numpy and names neither the argument
        # nor the function, so the caller never saw the sentence below.
        expected = arr.shape[1:] if bg.ndim == 1 else arr.shape
        if bg.shape != expected:
            msg = (
                "'background_levels' must match the per-position 'levels' "
                "shape or be a single (bands,) spectrum."
            )
            raise ValueError(msg)
        if bg.ndim == 1:
            bg = np.broadcast_to(bg, arr.shape)
        k1i, clamped = _k1_eq14(arr - bg, frequencies)
        corrected = energy_mean(arr - k1i, axis=0)
    else:
        bg_mean = _mean_level(background_levels)
        # The per-position branch's check, one axis down: a spectrum of the
        # wrong band count used to die in the subtraction below with numpy's
        # two-shape message, which names neither the argument nor the function.
        if bg_mean.shape != raw_mean.shape:
            msg = (
                "'background_levels' must carry one value per band "
                f"({raw_mean.size} in 'levels'); got shape "
                f"{np.shape(background_levels)}."
            )
            raise ValueError(msg)
        k1, clamped = _k1_eq14(raw_mean - bg_mean, frequencies)
        corrected = raw_mean - k1
    if clamped:
        warnings.warn(
            "Background margin below the ISO 3741 criterion (6 dB / 10 dB) in "
            "one or more bands; K1 clamped to the criterion value and the "
            "levels are upper bounds (ISO 3741:2010, 9.1.2).",
            SoundPowerWarning,
            stacklevel=3,
        )
    return np.asarray(corrected, dtype=np.float64), np.asarray(
        raw_mean - corrected, dtype=np.float64
    )


#: Minimum room volume vs lowest 1/3-oct band of interest (ISO 3741 Table 1).
_TABLE1_MIN_VOLUME: tuple[tuple[float, float], ...] = (
    (100.0, 200.0),
    (125.0, 150.0),
    (160.0, 100.0),
)


def _min_room_volume(lowest_band: float) -> float:
    """Minimum room volume for the lowest band of interest (ISO 3741 Table 1).

    Bands at or below 160 Hz demand a progressively larger room; from 200 Hz
    upward the floor is 70 m^3.
    """
    for band, vmin in _TABLE1_MIN_VOLUME:
        if lowest_band <= band:
            return vmin
    return 70.0


def _room_qualification_warnings(
    levels: np.ndarray,
    t60: np.ndarray,
    volume: float,
    surface_area: float,
    frequencies: np.ndarray,
) -> None:
    r"""Emit advisory :class:`SoundPowerWarning`\ s when the room or the
    microphone sampling fails an ISO 3741 qualification criterion.

    The determination still proceeds and returns a result; the warnings flag
    that the room must be qualified per Annex C/D or that more microphone
    positions are needed (ISO 3741:2010, clauses 5.2, 5.3, 8.3, 8.4.2.2).
    """
    lowest = float(np.min(frequencies))
    vmin = _min_room_volume(lowest)
    if volume < vmin:
        warnings.warn(
            f"Room volume {volume:g} m^3 is below the ISO 3741 Table 1 minimum "
            f"({vmin:g} m^3) for the lowest band of interest ({lowest:g} Hz); "
            "the room must be qualified per Annex C/D (ISO 3741:2010, 5.2, "
            "Table 1).",
            SoundPowerWarning,
            stacklevel=3,
        )
    floor = volume / surface_area
    below_6k3 = frequencies < _ABSORPTION_CRITERION_MAX_HZ
    if np.any(below_6k3 & (t60 <= floor)):
        warnings.warn(
            f"Reverberation time falls to or below the V/S floor ({floor:g} s) "
            "in one or more bands below 6,3 kHz; the room is too absorptive and "
            "must be qualified per Annex C (ISO 3741:2010, 5.3, Eq. 7).",
            SoundPowerWarning,
            stacklevel=3,
        )
    _position_sampling_warnings(levels, stacklevel=4)


def _position_sampling_warnings(levels: np.ndarray, stacklevel: int) -> None:
    """Emit the microphone-sampling advisories for a per-position level array.

    Flags fewer than 6 positions (ISO 3741:2010, 8.3, 8.4.1) and an
    inter-position standard deviation above the sM criterion (1,5 dB;
    8.4.2.2, Eq. 10). These sampling criteria need no room geometry, so they
    apply to both the direct and the comparison methods. A 1D (already-averaged)
    spectrum carries no per-position information and is skipped.
    """
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim != 2:  # noqa: PLR2004
        return
    n_positions = arr.shape[0]
    if n_positions < _MIN_MIC_POSITIONS:
        warnings.warn(
            f"Only {n_positions} microphone position(s) were supplied; an "
            f"unqualified reverberation room requires at least {_MIN_MIC_POSITIONS} "
            "(ISO 3741:2010, 8.3, 8.4.1).",
            SoundPowerWarning,
            stacklevel=stacklevel,
        )
    if n_positions >= _MIN_POSITIONS_FOR_SM:
        s_m = np.std(arr, axis=0, ddof=1)
        if np.any(s_m > _SM_CRITERION_DB):
            warnings.warn(
                "Inter-position standard deviation exceeds the ISO 3741 sM "
                "criterion (1,5 dB) in one or more bands; the source may radiate "
                "significant discrete tones, requiring more microphone/source "
                "positions or room qualification per Annex D (ISO 3741:2010, "
                "8.4.2.2, Eq. 10).",
                SoundPowerWarning,
                stacklevel=stacklevel,
            )


@dataclass(frozen=True)
class _DirectTerms:
    """The level-free part of Eq. (20)/(30), and the quantities it is made of.

    ``bracket`` is what the mean room level is raised by to become ``LW``
    (Eq. 20) or ``LJ`` (Eq. 30); the other fields are the per-band Sabine
    absorption area ``A``, the Waterhouse term, the corrections ``C1``/``C2``
    and the speed of sound the result reports.
    """

    absorption_area: np.ndarray
    waterhouse_correction: np.ndarray
    c1: float
    c2: float
    speed_of_sound: float
    bracket: np.ndarray


def _room_inputs(
    volume: float, surface_area: float, temperature: float, static_pressure: float
) -> None:
    """Refuse a room or a climate the direct method cannot be evaluated in.

    Finiteness is tested first because a comparison against zero lets ``NaN``
    through: ``nan <= 0`` is ``False``, so the room would pass here and the
    determination would return ``NaN`` bands with nothing said about why.
    """
    if (
        not math.isfinite(volume)
        or not math.isfinite(surface_area)
        or volume <= 0
        or surface_area <= 0
    ):
        msg = "'volume' and 'surface_area' must be positive and finite."
        raise ValueError(msg)
    _validate_meteorology(temperature, static_pressure)


def _band_inputs(
    n_bands: int, t60: float | np.ndarray, frequencies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The per-band ``T60`` and mid-band frequencies, one value per band."""
    freqs = np.asarray(frequencies, dtype=np.float64)
    t60_arr = np.broadcast_to(np.asarray(t60, dtype=np.float64), (n_bands,)).copy()
    if freqs.shape != (n_bands,):
        msg = "'frequencies' length must match the number of bands."
        raise ValueError(msg)
    if not np.all(np.isfinite(t60_arr)) or np.any(t60_arr <= 0.0):
        # As above: NaN slips past the comparison, and an infinite T60 drives
        # the absorption area to zero and the level to -inf.
        msg = "'t60' values must be positive and finite."
        raise ValueError(msg)
    if not np.all(np.isfinite(freqs)):
        msg = "'frequencies' must contain only finite values."
        raise ValueError(msg)
    if np.any(freqs <= 0.0):
        msg = "'frequencies' must be positive."
        raise ValueError(msg)
    return t60_arr, freqs


def _direct_terms(
    t60_arr: np.ndarray,
    volume: float,
    surface_area: float,
    freqs: np.ndarray,
    temperature: float,
    static_pressure: float,
) -> _DirectTerms:
    r"""Evaluate the bracket of Eq. (20)/(30) for a room, a climate and its bands.

    :math:`A = (55.26/c)(V/T_{60})`, :math:`c = 20.05\sqrt{273 + \theta}`, the
    Waterhouse term :math:`10 \log_{10}(1 + Sc/(8Vf))` and the meteorological
    corrections ``C1``/``C2`` (ISO 3741:2010, clause 9.1.4).
    """
    c = _speed_of_sound(temperature)
    c1 = _c1_correction(temperature, static_pressure)
    c2 = _c2_correction(temperature, static_pressure)
    absorption = (_SABINE_CONSTANT / c) * (volume / t60_arr)
    waterhouse = 10.0 * np.log10(1.0 + surface_area * c / (8.0 * volume * freqs))
    bracket = (
        10.0 * np.log10(absorption / _A0)
        + _EYRING_TERM_DB * (absorption / surface_area)
        + waterhouse
        + c1
        + c2
        + _DIFFUSE_FIELD_CONSTANT_DB
    )
    return _DirectTerms(
        absorption_area=np.asarray(absorption, dtype=np.float64),
        waterhouse_correction=np.asarray(waterhouse, dtype=np.float64),
        c1=c1,
        c2=c2,
        speed_of_sound=c,
        bracket=np.asarray(bracket, dtype=np.float64),
    )


def _comparison_inputs(
    n_bands: int,
    levels_ref: np.ndarray,
    lw_ref: np.ndarray,
    frequencies: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The reference source's mean room level, its known ``LW`` and the bands.

    :return: ``(Lp(RSS), LW(RSS), frequencies or None)``, each spanning the
        ``n_bands`` of the source under test.
    :raises ValueError: if the reference levels, ``lw_ref`` or ``frequencies``
        do not span the same bands as the source under test.
    """
    lp_rss = _mean_level(levels_ref)
    lw_rss = np.asarray(lw_ref, dtype=np.float64)
    if lp_rss.shape != (n_bands,) or lw_rss.shape != (n_bands,):
        msg = "'levels', 'levels_ref' and 'lw_ref' must span the same bands."
        raise ValueError(msg)
    freqs = None if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    if freqs is not None and freqs.shape != (n_bands,):
        msg = "'frequencies' length must match the number of bands."
        raise ValueError(msg)
    return lp_rss, lw_rss, freqs


def _require_frequencies(freqs: np.ndarray | None, name: str) -> np.ndarray:
    """The band centres the frequency-dependent ``K1`` criterion needs."""
    if freqs is None:
        msg = f"'frequencies' are required to apply '{name}'."
        raise ValueError(msg)
    return freqs


def _a_weighted_total(
    sound_power_level: np.ndarray, frequencies: np.ndarray | None
) -> float:
    """A-weighted total ``LWA`` (ISO 3741 Annex F Eq. F.2). ``NaN`` for several
    bands without frequencies; equal to ``LW`` for a single band.
    """
    n_bands = sound_power_level.shape[0]
    if frequencies is not None:
        try:
            ck = _a_weighting_corrections(frequencies)
        except ValueError:
            # Frequencies are not nominal band centres (e.g. exact filter
            # centres from room_parameters); the A-weighted total is undefined.
            return float("nan")
        return energy_sum(sound_power_level + ck)
    return float(sound_power_level[0]) if n_bands == 1 else float("nan")


def sound_power_reverberation(
    levels: np.ndarray,
    t60: float | np.ndarray,
    volume: float,
    surface_area: float,
    frequencies: np.ndarray,
    *,
    background_levels: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundPowerResult:
    r"""Sound power level in a reverberation room, direct method (ISO 3741:2010).

    ``levels`` is either a 1D per-band spectrum of the mean room sound pressure
    level or a 2D ``(NM, NB)`` array (one row per microphone position, one
    column per band) that is energy-averaged over positions (Eq. 16). The sound
    power level in each band follows Eq. (20):

    .. math::

       L_W = L_p(\text{ST}) + 10 \log_{10}\frac{A}{A_0} + 4.34 \frac{A}{S}
       + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6

    with the Sabine equivalent absorption area
    :math:`A = (55.26/c)(V/T_{60})` and the speed of sound
    :math:`c = 20.05 \sqrt{273 + \theta}`. The Waterhouse term
    :math:`10 \log_{10}(1 + Sc/(8Vf))` needs the band mid-frequencies, so
    ``frequencies`` is required. ``C1`` and ``C2`` carry the result to the
    reference meteorological conditions (clause 4).

    :param levels: Mean room SPL per band (1D) or ``(NM, NB)`` per-position
        levels, in decibels.
    :param t60: Reverberation time ``T60`` per band, in seconds (scalar or
        one value per band).
    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param surface_area: Total room surface area ``S``, in square metres.
    :param frequencies: One-third-octave (or octave) band mid-frequencies, Hz.
    :param background_levels: Background levels for the ``K1`` correction:
        per-position ``(NM, NB)`` (or a single ``(NB,)`` spectrum used at every
        position) with per-position ``levels``, applied per position (Eq. 14/15)
        before the energy average (Eq. 16). With 1D pre-averaged ``levels`` a
        single ``K1`` from the averaged spectra approximates the per-position
        procedure of clause 9.1.2.
    :param temperature: Air temperature ``theta`` in the room, in degrees Celsius.
    :param static_pressure: Static pressure ``ps`` in the room, in kilopascals.
    :return: :class:`ReverberationSoundPowerResult`.
    """
    _room_inputs(volume, surface_area, temperature, static_pressure)
    mean_level = _mean_level(levels)
    n_bands = mean_level.shape[0]
    t60_arr, freqs = _band_inputs(n_bands, t60, frequencies)

    _room_qualification_warnings(levels, t60_arr, volume, surface_area, freqs)

    if background_levels is not None:
        mean_level, k1 = _background_corrected_mean(levels, background_levels, freqs)
    else:
        k1 = np.zeros(n_bands, dtype=np.float64)

    terms = _direct_terms(
        t60_arr, volume, surface_area, freqs, temperature, static_pressure
    )
    lw = np.asarray(mean_level + terms.bracket, dtype=np.float64)

    return ReverberationSoundPowerResult(
        frequencies=freqs,
        sound_power_level=lw,
        mean_pressure_level=mean_level,
        absorption_area=terms.absorption_area,
        waterhouse_correction=terms.waterhouse_correction,
        background_correction=k1,
        c1=terms.c1,
        c2=terms.c2,
        speed_of_sound=terms.speed_of_sound,
        sound_power_level_a=_a_weighted_total(lw, freqs),
        method="direct",
    )


def sound_power_comparison(
    levels: np.ndarray,
    levels_ref: np.ndarray,
    lw_ref: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    background_levels: np.ndarray | None = None,
    background_levels_ref: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundPowerResult:
    r"""Sound power level in a reverberation room, comparison method (ISO 3741).

    A reference sound source of known per-band sound power ``lw_ref`` is
    measured at the same microphone positions as the source under test. The
    sound power level in each band follows Eq. (21):

    .. math::

       L_W = L_W(\text{RSS}) + \left( L_p(\text{ST}) - L_p(\text{RSS}) + C_2
       \right)

    where ``Lp(ST)`` and ``Lp(RSS)`` are the mean room levels (Eq. 16/17) of the
    test source and the reference source and ``C2`` is the radiation-impedance
    correction. The absorption-area, Waterhouse and ``C1`` terms cancel between
    the two sources, so the room absorption need not be known.

    :param levels: Mean room SPL per band (1D) or ``(NM, NB)`` per-position
        levels of the source under test, in decibels.
    :param levels_ref: Same, for the reference sound source, in decibels.
    :param lw_ref: Known sound power level ``LW(RSS)`` per band, in decibels.
    :param frequencies: Band mid-frequencies (Hz) for the A-weighted total.
    :param background_levels: Background levels for the ``K1`` correction of
        ``levels`` (per position, or a single spectrum; applied per position
        per Eq. 14/15 before the Eq. 16 average when ``levels`` is 2D).
    :param background_levels_ref: Background levels matching ``levels_ref``.
    :param temperature: Air temperature ``theta`` in the room, in degrees Celsius.
    :param static_pressure: Static pressure ``ps`` in the room, in kilopascals.
    :return: :class:`ReverberationSoundPowerResult` (``method='comparison'``).
    """
    _validate_meteorology(temperature, static_pressure)
    lp_st = _mean_level(levels)
    n_bands = lp_st.shape[0]
    lp_rss, lw_rss, freqs = _comparison_inputs(n_bands, levels_ref, lw_ref, frequencies)

    # Microphone-sampling advisories (<6 positions, sM > 1,5 dB) apply to the
    # per-position measurement of the source under test, exactly as in the
    # direct method; there is no room geometry here, so no V/S check. Emitted
    # only after the shape validations so malformed input raises cleanly first.
    _position_sampling_warnings(levels, stacklevel=2)

    k1_st = np.zeros(n_bands, dtype=np.float64)
    if background_levels is not None:
        lp_st, k1_st = _background_corrected_mean(
            levels, background_levels, _require_frequencies(freqs, "background_levels")
        )
    lp_rss = _reference_source_level(lp_rss, levels_ref, background_levels_ref, freqs)

    c2 = _c2_correction(temperature, static_pressure)
    lw = np.asarray(lw_rss + (lp_st - lp_rss + c2), dtype=np.float64)

    nan_band = np.full(n_bands, np.nan, dtype=np.float64)
    return ReverberationSoundPowerResult(
        frequencies=freqs,
        sound_power_level=lw,
        mean_pressure_level=lp_st,
        absorption_area=nan_band,
        waterhouse_correction=nan_band,
        background_correction=k1_st,
        c1=float("nan"),
        c2=c2,
        speed_of_sound=_speed_of_sound(temperature),
        sound_power_level_a=_a_weighted_total(lw, freqs),
        method="comparison",
    )


def _reference_source_level(
    lp_rss: np.ndarray,
    levels_ref: np.ndarray,
    background_levels_ref: np.ndarray | None,
    freqs: np.ndarray | None,
) -> np.ndarray:
    """The reference source's mean room level, background-corrected if asked.

    The reference sound source runs steadily, so its correction is the
    time-averaged one of 9.1.2 (Eq. 14/15 before Eq. 17) in both the sound
    power and the sound energy comparison.
    """
    if background_levels_ref is None:
        return lp_rss
    corrected, _ = _background_corrected_mean(
        levels_ref,
        background_levels_ref,
        _require_frequencies(freqs, "background_levels_ref"),
    )
    return corrected


# ---------------------------------------------------------------------------
# Sound energy level of a noise burst or transient emission (ISO 3741 clause
# 9.2, Annex F)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReverberationSoundEnergyResult:
    r"""Result of an ISO 3741:2010 reverberation-room sound energy level
    determination (clause 9.2).

    ``sound_energy_level`` is the per-band ``LJ`` (Eq. 30 direct method, Eq. 31
    comparison method), under reference meteorological conditions as both
    equations state it. ``mean_event_level`` is the mean corrected single event
    time-integrated level in the room :math:`\overline{L_E(\text{ST})}`
    (Eq. 27). ``absorption_area``, ``waterhouse_correction``, ``c1``, ``c2``,
    ``speed_of_sound`` and ``background_correction`` are what they are in
    :class:`ReverberationSoundPowerResult`, the background correction being
    the per-band shift of the mean level after each position was corrected by
    its own ``K1i`` (Eq. 25/26). ``sound_energy_level_a`` is the A-weighted
    total ``LJA`` (Annex F Eq. F.5), computed only when ``frequencies`` are
    supplied (``NaN`` for several bands without them; equal to ``LJ`` for a
    single band). ``method`` is ``'direct'`` or ``'comparison'``. ``events``
    is the number of single sound emission events :math:`N_\mathrm{e}` the
    levels were reduced from, or ``None`` when the caller supplied the mean
    single event level of one event; ``integration_time`` is the interval
    :math:`T` of the single event levels, in seconds, or ``None`` when no
    background correction needed it.
    """

    frequencies: np.ndarray | None
    sound_energy_level: np.ndarray
    mean_event_level: np.ndarray
    absorption_area: np.ndarray
    waterhouse_correction: np.ndarray
    background_correction: np.ndarray
    c1: float
    c2: float
    speed_of_sound: float
    sound_energy_level_a: float
    method: str
    events: int | None
    integration_time: float | None

    def __post_init__(self) -> None:
        """Reject a determination whose per-band quantities disagree.

        The pins are those of :class:`ReverberationSoundPowerResult`: the
        figure draws one bar per band of ``sound_energy_level`` and every
        other per-band column is read beside it, and ``method`` names the
        equation the level came from. ``events`` and ``integration_time`` are
        pinned as well, because a count of events below one or a
        non-positive interval describes no measurement the standard defines.

        :raises ValueError: if any per-band quantity disagrees with the rest,
            ``method`` is neither ``'direct'`` nor ``'comparison'``,
            ``events`` is below one or ``integration_time`` is not positive.
        """
        require_choice(self.method, "method", ("direct", "comparison"))
        if self.events is not None and self.events < 1:
            msg = (
                "ReverberationSoundEnergyResult: 'events' must be at least 1; "
                f"got {self.events!r}."
            )
            raise ValueError(msg)
        if self.integration_time is not None:
            require_positive(self.integration_time, "integration_time")
        require_ranks(
            self,
            frequencies=1,
            sound_energy_level=1,
            mean_event_level=1,
            absorption_area=1,
            waterhouse_correction=1,
            background_correction=1,
        )
        require_same_length(
            self,
            "frequencies",
            "sound_energy_level",
            "mean_event_level",
            "absorption_area",
            "waterhouse_correction",
            "background_correction",
        )

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


def _room_event_levels(
    levels: np.ndarray, events: int | None
) -> tuple[np.ndarray, int | None]:
    """The mean single event level of one event behind the three input forms.

    A 3-D ``(Ne, NM, NB)`` array holds one event per entry of its first axis
    and is reduced by Eq. (22); a 1-D or 2-D array with ``events`` is one
    measurement encompassing that many events and is reduced by Eq. (23); a
    1-D or 2-D array without ``events`` is the mean single event level of one
    event, already formed (per position, or already averaged over the room).

    :param levels: The levels, in decibels.
    :param events: ``None``, or the number of events one measurement holds.
    :return: The 1-D or 2-D levels and the event count they rest on
        (``None`` when the caller supplied the means).
    :raises ValueError: for a rank other than 1 to 3, non-finite levels, or a
        per-event array given together with ``events``.
    """
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim == 3:  # noqa: PLR2004
        if events is not None:
            msg = (
                "'levels' already carries one entry per event on its first axis; "
                "'events' applies to one measurement encompassing several events "
                "(ISO 3741:2010 Eq. 23), not to per-event levels."
            )
            raise ValueError(msg)
        count = int(arr.shape[0])
        return _single_event_mean(arr, None, name="levels", stacklevel=4), count
    if arr.ndim not in (1, 2):
        msg = (
            "'levels' must be a 1D spectrum, a 2D (positions, bands) array or a "
            "3D (events, positions, bands) array of single event levels."
        )
        raise ValueError(msg)
    if events is None:
        if not np.all(np.isfinite(arr)):
            msg = "'levels' must contain only finite values."
            raise ValueError(msg)
        return arr, None
    return _single_event_mean(arr, events, name="levels", stacklevel=4), int(events)


def sound_energy_reverberation(
    levels: np.ndarray,
    t60: float | np.ndarray,
    volume: float,
    surface_area: float,
    frequencies: np.ndarray,
    *,
    events: int | None = None,
    background_levels: np.ndarray | None = None,
    integration_time: float | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundEnergyResult:
    r"""Sound energy level in a reverberation room, direct method (ISO 3741:2010
    clause 9.2.4).

    ``levels`` holds the single event time-integrated sound pressure levels
    :math:`L'_{Ei(\mathrm{ST})}` measured through a period that encompasses the
    whole of the event, its decay included (clause 8.5.1; a moving microphone
    is not permitted for non-repetitive impulsive noise): a 1D per-band
    spectrum already averaged over the room, a 2D ``(NM, NB)`` array of one
    event's level at each position, a 3D ``(Ne, NM, NB)`` array of the
    :math:`N_\mathrm{e}` events measured one at a time (reduced by Eq. 22), or
    a 1D/2D level of one measurement encompassing ``events`` successive events
    (reduced by Eq. 23). Each position is corrected for its background by
    :math:`K_{1i}` (Eq. 25/26, the frequency-dependent criterion of 9.1.2),
    the positions are energy-averaged (Eq. 27) and the sound energy level in
    each band follows Eq. (30):

    .. math::

       L_J = \overline{L_E(\text{ST})} + \left[ 10 \log_{10}\frac{A}{A_0}
       + 4.34 \frac{A}{S} + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right)
       + C_1 + C_2 - 6 \right]

    with every term of the bracket exactly as in :func:`sound_power_reverberation`
    (Eq. 20), so the level is stated under the reference meteorological
    conditions of clause 4. The background is the time-averaged level the
    standard has measured over the same integration time :math:`T` as the
    events (clause 9.2.2), and it is compared as its exposure over that
    :math:`T`, :math:`L_{pi(\mathrm{B})} + 10 \log_{10}(T/T_0)` (clause 3.4
    NOTE 1), so that the energies Eq. (25) subtracts share one reference;
    ``integration_time`` is therefore required with ``background_levels``.

    :param levels: Single event levels, in decibels, in one of the four forms
        above.
    :param t60: Reverberation time ``T60`` per band, in seconds (scalar or
        one value per band).
    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param surface_area: Total room surface area ``S``, in square metres.
    :param frequencies: One-third-octave band mid-frequencies, Hz (required:
        the Waterhouse term and the ``K1`` criterion need them).
    :param events: The number of events ``Ne`` one measurement encompasses
        (Eq. 23); ``None`` when ``levels`` is per event or already the mean
        of one event.
    :param background_levels: Time-averaged background levels for ``K1``:
        per-position ``(NM, NB)`` (or a single ``(NB,)`` spectrum used at every
        position) with per-position ``levels``, applied per position before
        the energy average; with 1D ``levels`` a single ``K1`` from the averaged
        spectra approximates the per-position procedure.
    :param integration_time: The interval ``T`` of the single event levels, in
        seconds; required with ``background_levels``.
    :param temperature: Air temperature ``theta`` in the room, in degrees Celsius.
    :param static_pressure: Static pressure ``ps`` in the room, in kilopascals.
    :return: :class:`ReverberationSoundEnergyResult` (``method='direct'``).
    :raises ValueError: for a malformed or non-finite level array, a
        non-physical room or climate, a background without its
        ``integration_time``, or mismatched band counts.
    """
    _room_inputs(volume, surface_area, temperature, static_pressure)
    event_levels, event_count = _room_event_levels(levels, events)
    if integration_time is not None:
        integration_time = require_positive(integration_time, "integration_time")
    mean_level = _mean_level(event_levels)
    n_bands = mean_level.shape[0]
    t60_arr, freqs = _band_inputs(n_bands, t60, frequencies)

    _room_qualification_warnings(event_levels, t60_arr, volume, surface_area, freqs)

    if background_levels is not None:
        exposure = _background_exposure(background_levels, integration_time)
        mean_level, k1 = _background_corrected_mean(event_levels, exposure, freqs)
    else:
        k1 = np.zeros(n_bands, dtype=np.float64)

    terms = _direct_terms(
        t60_arr, volume, surface_area, freqs, temperature, static_pressure
    )
    lj = np.asarray(mean_level + terms.bracket, dtype=np.float64)

    return ReverberationSoundEnergyResult(
        frequencies=freqs,
        sound_energy_level=lj,
        mean_event_level=mean_level,
        absorption_area=terms.absorption_area,
        waterhouse_correction=terms.waterhouse_correction,
        background_correction=k1,
        c1=terms.c1,
        c2=terms.c2,
        speed_of_sound=terms.speed_of_sound,
        sound_energy_level_a=_a_weighted_total(lj, freqs),
        method="direct",
        events=event_count,
        integration_time=integration_time,
    )


def sound_energy_comparison(
    levels: np.ndarray,
    levels_ref: np.ndarray,
    lw_ref: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    events: int | None = None,
    background_levels: np.ndarray | None = None,
    integration_time: float | None = None,
    background_levels_ref: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundEnergyResult:
    r"""Sound energy level in a reverberation room, comparison method
    (ISO 3741:2010 clause 9.2.5).

    A reference sound source of known per-band sound power ``lw_ref`` runs
    steadily at the same microphone positions as the source under test, whose
    single event levels ``levels`` take any of the forms
    :func:`sound_energy_reverberation` accepts. The sound energy level in each
    band follows Eq. (31):

    .. math::

       L_J = L_W(\text{RSS}) + \left( \overline{L_E(\text{ST})}
       - \overline{L_p(\text{RSS})} \right) + C_2

    where :math:`\overline{L_E(\text{ST})}` is the mean corrected single event
    level of the source under test (Eq. 27), :math:`\overline{L_p(\text{RSS})}`
    the mean corrected time-averaged level of the reference source (Eq. 17) and
    ``C2`` the radiation-impedance correction. The room terms and ``C1`` cancel
    between the two sources exactly as in :func:`sound_power_comparison`. The
    source under test is background-corrected as in
    :func:`sound_energy_reverberation` (its background compared as an exposure
    over ``integration_time``); the reference source, being steady, by the
    time-averaged correction of 9.1.2.

    :param levels: Single event levels of the source under test, in decibels.
    :param levels_ref: Mean room SPL per band (1D) or ``(NM, NB)`` per-position
        time-averaged levels of the reference sound source, in decibels.
    :param lw_ref: Known sound power level ``LW(RSS)`` per band, in decibels,
        under the meteorological conditions of the test.
    :param frequencies: Band mid-frequencies (Hz) for the ``K1`` criterion and
        the A-weighted total.
    :param events: The number of events ``Ne`` one measurement encompasses
        (Eq. 23); ``None`` when ``levels`` is per event or already the mean
        of one event.
    :param background_levels: Time-averaged background levels for the ``K1``
        correction of ``levels`` (per position, or a single spectrum).
    :param integration_time: The interval ``T`` of the single event levels, in
        seconds; required with ``background_levels``.
    :param background_levels_ref: Background levels matching ``levels_ref``.
    :param temperature: Air temperature ``theta`` in the room, in degrees Celsius.
    :param static_pressure: Static pressure ``ps`` in the room, in kilopascals.
    :return: :class:`ReverberationSoundEnergyResult` (``method='comparison'``).
    :raises ValueError: for a malformed or non-finite level array, a
        non-physical climate, a background without its ``integration_time``
        or without ``frequencies``, or mismatched band counts.
    """
    _validate_meteorology(temperature, static_pressure)
    event_levels, event_count = _room_event_levels(levels, events)
    if integration_time is not None:
        integration_time = require_positive(integration_time, "integration_time")
    le_st = _mean_level(event_levels)
    n_bands = le_st.shape[0]
    lp_rss, lw_rss, freqs = _comparison_inputs(n_bands, levels_ref, lw_ref, frequencies)

    _position_sampling_warnings(event_levels, stacklevel=2)

    k1_st = np.zeros(n_bands, dtype=np.float64)
    if background_levels is not None:
        exposure = _background_exposure(background_levels, integration_time)
        le_st, k1_st = _background_corrected_mean(
            event_levels, exposure, _require_frequencies(freqs, "background_levels")
        )
    lp_rss = _reference_source_level(lp_rss, levels_ref, background_levels_ref, freqs)

    c2 = _c2_correction(temperature, static_pressure)
    lj = np.asarray(lw_rss + (le_st - lp_rss) + c2, dtype=np.float64)

    nan_band = np.full(n_bands, np.nan, dtype=np.float64)
    return ReverberationSoundEnergyResult(
        frequencies=freqs,
        sound_energy_level=lj,
        mean_event_level=le_st,
        absorption_area=nan_band,
        waterhouse_correction=nan_band,
        background_correction=k1_st,
        c1=float("nan"),
        c2=c2,
        speed_of_sound=_speed_of_sound(temperature),
        sound_energy_level_a=_a_weighted_total(lj, freqs),
        method="comparison",
        events=event_count,
        integration_time=integration_time,
    )


def octave_band_levels(
    levels: np.ndarray, frequencies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Octave-band levels from one-third-octave band levels (ISO 3741 Annex F).

    The level in the :math:`i`-th octave band, :math:`1 \le i \le 8` for the
    mid-band frequencies 63 Hz to 8 kHz, is the energy sum of the three
    one-third-octave bands :math:`k = 3i-2` to :math:`3i` of Table F.1 that
    make it up, for sound power levels (Eq. F.1) and sound energy levels
    (Eq. F.4) alike:

    .. math::

       L_{Ji} = 10 \log_{10} \sum_{k=3i-2}^{3i} 10^{0.1 L_{Jk}}

    Every octave the input touches must be supplied whole: Table F.1 numbers
    the one-third-octave bands from :math:`k = 1` at 50 Hz, so the 63 Hz
    octave is the 50, 63 and 80 Hz thirds and the 8 kHz octave the 6,3, 8 and
    10 kHz thirds, and a band whose triplet is incomplete cannot be summed.

    :param levels: Band levels in decibels, with the bands on the last axis
        (``(NB,)``, or ``(..., NB)`` for several spectra at once).
    :param frequencies: The ``NB`` nominal one-third-octave mid-band
        frequencies of ``levels``, in hertz, from 50 Hz to 10 kHz.
    :return: ``(octave mid-band frequencies, octave-band levels)``, the
        frequencies ascending and the levels with the octaves on the last
        axis.
    :raises ValueError: for a frequency outside Table F.1, a repeated
        frequency, an octave whose three thirds are not all present, or
        levels that do not carry one value per frequency.
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size == 0:
        msg = (
            "'frequencies' must be a non-empty 1-D array of nominal "
            "one-third-octave mid-band frequencies."
        )
        raise ValueError(msg)
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != freqs.size:
        msg = (
            "'levels' must carry one value per band of 'frequencies' on its last axis."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "'levels' must contain only finite values."
        raise ValueError(msg)
    if not np.all(np.isfinite(freqs)):
        # Checked before the loop below rounds each one: round(inf) raises
        # OverflowError, which is not the ValueError this function documents.
        msg = "'frequencies' must contain only finite values."
        raise ValueError(msg)
    columns: dict[int, list[int]] = {}
    for column, f in enumerate(freqs):
        nominal = round(float(f))
        if nominal not in _THIRD_OCTAVE_NOMINAL:
            msg = (
                "'frequencies' must be nominal one-third-octave mid-band "
                f"frequencies from 50 Hz to 10 kHz (ISO 3741:2010 Table F.1); got {f:g}."
            )
            raise ValueError(msg)
        # Table F.1 counts k from 1 at 50 Hz; octave i holds k = 3i-2 .. 3i.
        k = _THIRD_OCTAVE_NOMINAL.index(nominal) + 1
        octave = (k + _THIRDS_PER_OCTAVE - 1) // _THIRDS_PER_OCTAVE
        members = columns.setdefault(octave, [])
        if column in members or any(round(float(freqs[j])) == nominal for j in members):
            msg = f"'frequencies' must not repeat a band; {nominal} Hz appears twice."
            raise ValueError(msg)
        members.append(column)
    for octave, members in columns.items():
        if len(members) != _THIRDS_PER_OCTAVE:
            msg = (
                "'frequencies' must supply the three one-third-octave bands of "
                "every octave it touches (ISO 3741:2010 Eq. F.1, k = 3i-2 to 3i); "
                f"the {_OCTAVE_NOMINAL[octave - 1]} Hz octave has {len(members)}."
            )
            raise ValueError(msg)
    order = sorted(columns)
    octave_freqs = np.array([_OCTAVE_NOMINAL[i - 1] for i in order], dtype=np.float64)
    summed = np.stack(
        [energy_sum(arr[..., columns[i]], axis=-1) for i in order], axis=-1
    )
    return octave_freqs, np.asarray(summed, dtype=np.float64)
