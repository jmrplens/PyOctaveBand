#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Sound power level of a noise source measured in a reverberation test room:
ISO 3741:2010 (precision method, accuracy grade 1).

The source is placed in a hard-walled reverberation room whose reverberant
field is sampled by microphones. Two methods are provided.

The **direct method** derives the sound power from the mean corrected room
sound pressure level ``Lp(ST)`` and the equivalent absorption area ``A`` of the
room (ISO 3741:2010 clause 9.1.4, Eq. 20)::

    Lp(ST) = 10*lg( (1/NM) * sum_i 10^(0,1*Lpi) )                     (Eq. 16)
    A      = (55,26/c) * (V/T60)          (Sabine)                    (clause 9.1.4)
    c      = 20,05 * sqrt(273 + theta)    speed of sound, m/s
    LW = Lp(ST) + 10*lg(A/A0) + 4,34*(A/S) + 10*lg(1 + S*c/(8*V*f))
                + C1 + C2 - 6                                         (Eq. 20)

``10*lg(1 + S*c/(8*V*f))`` is the Waterhouse boundary correction (energy stored
near the room boundaries); it vanishes as the frequency grows. ``C1`` (Eq. 20,
reference-quantity correction) and ``C2`` (radiation-impedance correction) carry
the result to the reference meteorological conditions of clause 4 (23,0 C,
101,325 kPa, 50 %)::

    C1 = -10*lg(ps/ps0) + 5*lg((273,15+theta)/314)                   (clause 9.1.4)
    C2 = -10*lg(ps/ps0) + 15*lg((273,15+theta)/296)

The **comparison method** replaces the absorption-area terms by a reference
sound source (RSS) of known sound power ``LW(RSS)`` measured at the same
positions (ISO 3741:2010 clause 9.1.5, Eq. 21)::

    LW = LW(RSS) + ( Lp(ST) - Lp(RSS) + C2 )                         (Eq. 21)

Both methods cover the one-third-octave bands from 100 Hz to 10 kHz (clause
8.1). Octave-band, A-weighted and total levels follow ISO 3741 Annex F, which
reuses the ISO 3744 Annex E A-weighting band corrections.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from .._internal.levels_math import energy_mean, energy_sum
from .sound_power import (
    SoundPowerWarning,
    _a_weighting_corrections,
)

_A0 = 1.0  #: Reference absorption area, in square metres (ISO 3741, Eq. 20).
_PS0 = 101.325  #: Reference static pressure, in kPa (ISO 3741 clause 4).
_THETA0 = 314.0  #: Reference temperature for C1, in K (ISO 3741 clause 9.1.4).
_THETA1 = 296.0  #: Reference temperature for C2, in K (ISO 3741 clause 9.1.4).


@dataclass(frozen=True)
class ReverberationSoundPowerResult:
    """Result of an ISO 3741:2010 reverberation-room sound power determination.

    ``sound_power_level`` is the per-band ``LW`` (Eq. 20 direct method, Eq. 21
    comparison method). ``mean_pressure_level`` is the mean corrected room level
    ``Lp(ST)`` (Eq. 16). For the direct method ``absorption_area`` is the Sabine
    equivalent absorption area ``A`` per band and ``waterhouse_correction`` the
    boundary term ``10*lg(1 + S*c/(8*V*f))``; both are ``NaN`` for the
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
    or ``'comparison'``."""

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
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso3741 import render_reverberation_power_report

        return render_reverberation_power_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _validate_meteorology(temperature: float, static_pressure: float) -> None:
    """Guard the meteorological inputs before the log10/sqrt of C1/C2 and c.

    A non-finite or ``<= -273 degC`` temperature makes ``sqrt(273 + theta)``
    complex/zero, and a non-finite or non-positive static pressure makes
    ``lg(ps/ps0)`` undefined; both are rejected with a clean ``ValueError``."""
    if not np.isfinite(temperature) or temperature <= -273.0:
        raise ValueError(
            "'temperature' must be finite and greater than -273 degC."
        )
    if not np.isfinite(static_pressure) or static_pressure <= 0.0:
        raise ValueError("'static_pressure' must be finite and positive.")


def _speed_of_sound(temperature: float) -> float:
    """Speed of sound ``c = 20,05*sqrt(273 + theta)`` (ISO 3741 clause 9.1.4)."""
    return float(20.05 * np.sqrt(273.0 + temperature))


def _c1_correction(temperature: float, static_pressure: float) -> float:
    """Reference-quantity correction ``C1`` (ISO 3741:2010 clause 9.1.4)."""
    return float(
        -10.0 * np.log10(static_pressure / _PS0)
        + 5.0 * np.log10((273.15 + temperature) / _THETA0)
    )


def _c2_correction(temperature: float, static_pressure: float) -> float:
    """Radiation-impedance correction ``C2`` (ISO 3741:2010 clause 9.1.4)."""
    return float(
        -10.0 * np.log10(static_pressure / _PS0)
        + 15.0 * np.log10((273.15 + temperature) / _THETA1)
    )


def _mean_level(levels: np.ndarray) -> np.ndarray:
    """Energy mean over microphone positions (rows), returning one value per
    band. A 1D input is treated as a single averaged spectrum (ISO 3741
    Eq. 16)."""
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return energy_mean(arr, axis=0)
    raise ValueError("'levels' must be a 1D spectrum or a 2D (positions, bands) array.")


def _k1_eq14(delta: np.ndarray, frequencies: np.ndarray) -> tuple[np.ndarray, bool]:
    """Background-noise correction ``K1`` from ``dLp`` (ISO 3741:2010 Eq. 14).

    ``K1 = -10*lg(1 - 10^(-0,1*dLp))``. The precision-grade qualification is
    frequency dependent (clause 9.1.2): ``dLp >= 15 dB`` -> ``K1 = 0``; below
    the lower criterion (6 dB for bands <= 200 Hz and >= 6 300 Hz, 10 dB for
    250 Hz to 5 000 Hz) ``K1`` is clamped to the criterion value (1,26 dB /
    0,46 dB) and the levels become upper bounds. ``delta`` may be per band
    ``(NB,)`` or per position and band ``(NM, NB)``; the second returned value
    flags whether any element fell below the lower criterion."""
    low = np.where((frequencies <= 200.0) | (frequencies >= 6300.0), 6.0, 10.0)
    clamped = np.maximum(delta, low)
    k1 = -10.0 * np.log10(1.0 - 10.0 ** (-0.1 * clamped))
    k1 = np.where(delta >= 15.0, 0.0, k1)
    return np.asarray(k1, dtype=np.float64), bool(np.any(delta < low))


def _background_corrected_mean(
    levels: np.ndarray,
    background_levels: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Background-corrected mean room level and effective per-band ``K1``.

    With per-position 2D ``levels`` the correction follows ISO 3741:2010
    clauses 9.1.2/9.1.3 exactly: the correction ``K1i`` is computed at each
    microphone position (Eq. 14), each position level is corrected first
    (``Lpi(ST) = L'pi(ST) - K1i``, Eq. 15) and the corrected levels are then
    energy-averaged (Eq. 16). ``background_levels`` may be per position
    (matching shape) or a single ``(NB,)`` spectrum used at every position.
    The returned per-band ``K1`` is the effective correction (uncorrected mean
    minus corrected mean).

    With 1D pre-averaged ``levels`` the per-position information is gone, so a
    single ``K1`` is computed from the averaged spectra -- an approximation of
    the per-position procedure that is exact only when every position has the
    same source-to-background margin. Prefer per-position input."""
    arr = np.asarray(levels, dtype=np.float64)
    raw_mean = _mean_level(levels)
    if arr.ndim == 2:
        bg = np.asarray(background_levels, dtype=np.float64)
        if bg.ndim == 1:
            bg = np.broadcast_to(bg, arr.shape)
        if bg.shape != arr.shape:
            raise ValueError(
                "'background_levels' must match the per-position 'levels' "
                "shape or be a single (bands,) spectrum."
            )
        k1i, clamped = _k1_eq14(arr - bg, frequencies)
        corrected = energy_mean(arr - k1i, axis=0)
    else:
        k1, clamped = _k1_eq14(raw_mean - _mean_level(background_levels), frequencies)
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
    upward the floor is 70 m^3."""
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
    """Emit advisory :class:`SoundPowerWarning`\\ s when the room or the
    microphone sampling fails an ISO 3741 qualification criterion.

    The determination still proceeds and returns a result; the warnings flag
    that the room must be qualified per Annex C/D or that more microphone
    positions are needed (ISO 3741:2010, clauses 5.2, 5.3, 8.3, 8.4.2.2)."""
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
    below_6k3 = frequencies < 6300.0
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
    spectrum carries no per-position information and is skipped."""
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim != 2:
        return
    n_positions = arr.shape[0]
    if n_positions < 6:
        warnings.warn(
            f"Only {n_positions} microphone position(s) were supplied; an "
            "unqualified reverberation room requires at least 6 "
            "(ISO 3741:2010, 8.3, 8.4.1).",
            SoundPowerWarning,
            stacklevel=stacklevel,
        )
    if n_positions >= 2:
        s_m = np.std(arr, axis=0, ddof=1)
        if np.any(s_m > 1.5):
            warnings.warn(
                "Inter-position standard deviation exceeds the ISO 3741 sM "
                "criterion (1,5 dB) in one or more bands; the source may radiate "
                "significant discrete tones, requiring more microphone/source "
                "positions or room qualification per Annex D (ISO 3741:2010, "
                "8.4.2.2, Eq. 10).",
                SoundPowerWarning,
                stacklevel=stacklevel,
            )


def _a_weighted_total(
    sound_power_level: np.ndarray, frequencies: np.ndarray | None
) -> float:
    """A-weighted total ``LWA`` (ISO 3741 Annex F Eq. F.2). ``NaN`` for several
    bands without frequencies; equal to ``LW`` for a single band."""
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
    t60: np.ndarray,
    volume: float,
    surface_area: float,
    frequencies: np.ndarray,
    *,
    background_levels: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundPowerResult:
    """Sound power level in a reverberation room, direct method (ISO 3741:2010).

    ``levels`` is either a 1D per-band spectrum of the mean room sound pressure
    level or a 2D ``(NM, NB)`` array (one row per microphone position, one
    column per band) that is energy-averaged over positions (Eq. 16). The sound
    power level in each band follows Eq. (20)::

        LW = Lp(ST) + 10*lg(A/A0) + 4,34*(A/S) + 10*lg(1 + S*c/(8*V*f))
                    + C1 + C2 - 6

    with the Sabine equivalent absorption area ``A = (55,26/c)*(V/T60)`` and the
    speed of sound ``c = 20,05*sqrt(273 + theta)``. The Waterhouse term
    ``10*lg(1 + S*c/(8*V*f))`` needs the band mid-frequencies, so ``frequencies``
    is required. ``C1`` and ``C2`` carry the result to the reference
    meteorological conditions (clause 4).

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
    if volume <= 0 or surface_area <= 0:
        raise ValueError("'volume' and 'surface_area' must be positive.")
    _validate_meteorology(temperature, static_pressure)
    mean_level = _mean_level(levels)
    n_bands = mean_level.shape[0]
    freqs = np.asarray(frequencies, dtype=np.float64)
    t60_arr = np.broadcast_to(np.asarray(t60, dtype=np.float64), (n_bands,)).copy()
    if freqs.shape != (n_bands,):
        raise ValueError("'frequencies' length must match the number of bands.")
    if np.any(t60_arr <= 0.0):
        raise ValueError("'t60' values must be positive.")
    if np.any(freqs <= 0.0):
        raise ValueError("'frequencies' must be positive.")

    _room_qualification_warnings(levels, t60_arr, volume, surface_area, freqs)

    if background_levels is not None:
        mean_level, k1 = _background_corrected_mean(levels, background_levels, freqs)
    else:
        k1 = np.zeros(n_bands, dtype=np.float64)

    c = _speed_of_sound(temperature)
    c1 = _c1_correction(temperature, static_pressure)
    c2 = _c2_correction(temperature, static_pressure)
    absorption = (55.26 / c) * (volume / t60_arr)
    waterhouse = 10.0 * np.log10(1.0 + surface_area * c / (8.0 * volume * freqs))

    lw = (
        mean_level
        + 10.0 * np.log10(absorption / _A0)
        + 4.34 * (absorption / surface_area)
        + waterhouse
        + c1
        + c2
        - 6.0
    )
    lw = np.asarray(lw, dtype=np.float64)

    return ReverberationSoundPowerResult(
        frequencies=freqs,
        sound_power_level=lw,
        mean_pressure_level=mean_level,
        absorption_area=np.asarray(absorption, dtype=np.float64),
        waterhouse_correction=np.asarray(waterhouse, dtype=np.float64),
        background_correction=k1,
        c1=c1,
        c2=c2,
        speed_of_sound=c,
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
    """Sound power level in a reverberation room, comparison method (ISO 3741).

    A reference sound source of known per-band sound power ``lw_ref`` is
    measured at the same microphone positions as the source under test. The
    sound power level in each band follows Eq. (21)::

        LW = LW(RSS) + ( Lp(ST) - Lp(RSS) + C2 )

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
    lp_rss = _mean_level(levels_ref)
    lw_rss = np.asarray(lw_ref, dtype=np.float64)
    n_bands = lp_st.shape[0]
    if lp_rss.shape != (n_bands,) or lw_rss.shape != (n_bands,):
        raise ValueError(
            "'levels', 'levels_ref' and 'lw_ref' must span the same bands."
        )

    freqs = None if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    if freqs is not None and freqs.shape != (n_bands,):
        raise ValueError("'frequencies' length must match the number of bands.")

    # Microphone-sampling advisories (<6 positions, sM > 1,5 dB) apply to the
    # per-position measurement of the source under test, exactly as in the
    # direct method; there is no room geometry here, so no V/S check. Emitted
    # only after the shape validations so malformed input raises cleanly first.
    _position_sampling_warnings(levels, stacklevel=2)

    k1_st = np.zeros(n_bands, dtype=np.float64)
    if background_levels is not None:
        if freqs is None:
            raise ValueError("'frequencies' are required to apply 'background_levels'.")
        lp_st, k1_st = _background_corrected_mean(levels, background_levels, freqs)
    if background_levels_ref is not None:
        if freqs is None:
            raise ValueError(
                "'frequencies' are required to apply 'background_levels_ref'."
            )
        lp_rss, _ = _background_corrected_mean(
            levels_ref, background_levels_ref, freqs
        )

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
