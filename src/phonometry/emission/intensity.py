#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Two-microphone (p-p) sound intensity per IEC 61043:1993 and the
ISO 9614-1:1993 field indicators.

A p-p probe holds two pressure microphones a fixed distance ``spacing``
(dr) apart. The mean of the two pressures is taken as the sound pressure
at the probe reference point, while the pressure differential is used to
derive the particle velocity component along the probe axis
(IEC 61043:1993, definition 3.2)::

    p(t) = (p1(t) + p2(t)) / 2
    u(t) = -(1 / (rho * dr)) * integral of (p2(t) - p1(t)) dt
    I    = < p(t) * u(t) >          (time average)

For stationary signals the time-averaged intensity reduces to the
imaginary part of the one-sided cross spectrum G12 of the two microphone
pressures (the frequency-domain form of the same finite-difference
estimator)::

    I(f) = -Im{G12(f)} / (2 * pi * f * rho * dr)

The finite-difference gradient underestimates the true plane-wave
intensity by the factor ``sin(k*dr) / (k*dr)`` with ``k = 2*pi*f/c``;
IEC 61043:1993 clause 7.3 specifies the probe intensity response with
exactly this argument (``Ff = dr * f * 2 * pi / c``) and Table 3 lists
the resulting nominal response (e.g. -10,5 dB at 6,3 kHz for a 25 mm
separation). Below ``f = 0,1 * c / dr`` (k*dr < 0,63) the bias stays
under about 0,3 dB (factor >= 0,935).

Field indicators F1 (temporal variability), F2 (surface
pressure-intensity), F3 (negative partial power) and F4 (field
non-uniformity) follow ISO 9614-1:1993 Annex A (normative), equations
(A.1)-(A.9). F1 is measured in the initial test (clause 8.2) at one
typical position on the initial measurement surface and qualifies the
*field*, not the surface: it is the coefficient of variation of M
short-time-averaged samples of the normal intensity at that fixed
point, and Table B.3 calls for action code (e) when it exceeds 0,6.
The dynamic capability index is
``Ld = delta_pI0 - K`` (ISO 9614-1 clause 3.12, equation (10)); the
instrument is adequate for a measurement when ``Ld > F2`` (criterion 1,
Annex B equation (B.1)). The residual index ``delta_pI0`` that feeds it
is classified against IEC 61043:1993 Table 2 by
:func:`phonometry.metrology.intensity_compliance.intensity_class_compliance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from .._internal.levels_math import energy_mean
from .._internal.utils import _typesignal
from ..metrology.frequencies import _genfreqs
from ..metrology.spectra import (
    _default_nperseg,
    _welch_autospectrum,
    _welch_cross_spectrum,
)

#: Temporal-variability limit of ISO 9614-1:1993 Table B.3: a field whose F1
#: exceeds 0,6 calls for action code (e) before the surface is measured.
TEMPORAL_VARIABILITY_LIMIT = 0.6

#: Reference sound intensity, in watts per square metre (ISO 9614-1, A.2.3).
_I0 = 1.0e-12
#: Reference sound pressure, in pascals.
_P0 = 2.0e-5
#: Upper ``k*dr`` bound (rad) for the finite-difference bias correction. The
#: exact reciprocal ``(k*dr)/sin(k*dr)`` diverges as ``k*dr -> pi`` (the first
#: spatial-aliasing null, ``f = c/(2*dr)``), so a handful of near-null bins
#: would otherwise dominate the summed totals. IEC 61043:1993 (7.3) only
#: characterises the probe over its usable range, well short of the null;
#: applying the correction up to ``pi/2`` and holding it constant beyond keeps
#: the totals bounded while leaving the low-frequency region (where the bias
#: matters and is trustworthy) exact.
_BIAS_CORRECTION_MAX_KDR = np.pi / 2.0


def _finite_difference_correction(k_dr: np.ndarray) -> np.ndarray:
    """Reciprocal ``(k*dr)/sin(k*dr)`` of the finite-difference response.

    IEC 61043:1993 (7.3). The correction is clamped at ``k*dr = pi/2``: below
    the cutoff the exact reciprocal is returned (so the trustworthy
    low-frequency region is left unchanged), and at or above it the factor is
    held constant at its ``pi/2`` value (``pi/2``). This avoids the divergence
    as ``k*dr -> pi`` that would let near-null bins blow up the totals.
    """
    kd = np.clip(np.asarray(k_dr, dtype=np.float64), 0.0, _BIAS_CORRECTION_MAX_KDR)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(kd > 0.0, kd / np.sin(kd), 1.0)


@dataclass(frozen=True)
class IntensityResult:
    """Result of a p-p sound intensity measurement.

    Per-band arrays are ``None`` unless a band ``fraction`` was requested.
    ``intensity`` is signed (positive along the probe axis, from
    microphone 1 towards microphone 2); ``intensity_level`` is computed
    from the magnitude, ``10*lg(|I|/1e-12)`` dB, with the sign reported
    separately in ``direction`` (+1/-1). ``pressure_intensity_index`` is
    ``Lp - LI`` (the single-position form of the ISO 9614-1:1993 F2
    indicator, equation (A.3)). ``bias_correction`` is the multiplicative
    factor ``(k*dr)/sin(k*dr)`` compensating the finite-difference
    underestimation at each band centre (IEC 61043:1993, 7.3); it is NaN
    at and beyond the first null ``k*dr >= pi``. ``max_valid_frequency``
    is the usable-bandwidth bound ``0,1*c/spacing`` (bias < ~0,3 dB).
    ``spacing`` retains the microphone separation the measurement was
    reduced with so :meth:`plot_geometry` can draw the probe; it is
    appended after the original fields and ``None`` for hand-built
    results.
    """

    frequency: np.ndarray | None
    intensity: np.ndarray | None
    intensity_level: np.ndarray | None
    pressure_level: np.ndarray | None
    pressure_intensity_index: np.ndarray | None
    direction: np.ndarray | None
    bias_correction: np.ndarray | None
    total_intensity: float
    total_intensity_level: float
    total_pressure_level: float
    total_pressure_intensity_index: float
    total_direction: int
    max_valid_frequency: float
    spacing: float | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot Lp vs LI per band with the pressure-intensity index.

        Requires per-band data (call ``sound_intensity(..., fraction=...)``)
        and matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_intensity

        check_language(language)
        return plot_intensity(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the p-p probe with its spacer to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_intensity_result_geometry

        check_language(language)
        return plot_intensity_result_geometry(self, ax=ax, language=language, **kwargs)


@dataclass(frozen=True)
class FieldIndicators:
    """ISO 9614-1:1993 Annex A field indicators over a measurement surface.

    ``f2`` is the surface pressure-intensity indicator (equation (A.3)),
    ``f3`` the negative partial power indicator (equation (A.6)) and
    ``f4`` the field non-uniformity indicator (equation (A.8)).
    ``f3 - f2 > 0`` reveals negative partial power flowing through parts
    of the surface. The instrument's dynamic capability index must
    satisfy ``Ld > f2`` (criterion 1, equation (B.1)); the number of
    positions N must satisfy ``N > C * f4**2`` (criterion 2, equation
    (B.2)).

    ``f1`` is the temporal variability indicator (equation (A.1)), present
    only when :func:`field_indicators` was given the ``temporal_intensity``
    samples it is computed from; it is ``None`` otherwise. Unlike the other
    three it describes the *field* rather than the surface: it is evaluated
    at one typical position from M short-time-averaged samples of the
    normal intensity (clause 8.2, the initial test), and Table B.3
    requires action when it exceeds :data:`TEMPORAL_VARIABILITY_LIMIT`.

    With per-position *and* per-band input (2D arrays passed to
    :func:`field_indicators`) the indicators are per-band arrays and
    ``frequency`` carries the band centres; with 1D per-position input they
    are scalars and ``frequency`` is ``None``.
    """

    f2: float | np.ndarray
    f3: float | np.ndarray
    f4: float | np.ndarray
    frequency: np.ndarray | None = None
    f1: float | np.ndarray | None = None

    def field_is_stationary(
        self, limit: float = TEMPORAL_VARIABILITY_LIMIT
    ) -> bool | np.ndarray:
        """Whether the temporal variability stays within the Table B.3 limit.

        ISO 9614-1:1993 Table B.3 lists ``F1 > 0,6`` as the condition that
        calls for action code (e), i.e. reducing the temporal variability of
        the extraneous intensity, measuring during quieter periods or
        lengthening the averaging time at each position. A field at exactly
        the limit is not flagged.

        :param limit: The Table B.3 threshold (default
            :data:`TEMPORAL_VARIABILITY_LIMIT`, 0,6).
        :return: ``True``/``False`` for a scalar ``f1``, or a boolean array
            per band for a per-band ``f1``.
        :raises ValueError: If the result carries no ``f1`` (call
            :func:`field_indicators` with ``temporal_intensity``, or use
            :func:`temporal_variability_indicator` directly).
        """
        if self.f1 is None:
            raise ValueError(
                "This result carries no F1; call field_indicators(..., "
                "temporal_intensity=...) or temporal_variability_indicator()."
            )
        f1 = np.asarray(self.f1, dtype=np.float64)
        ok = f1 <= float(limit)
        return bool(ok) if ok.ndim == 0 else ok

    def plot(
        self,
        ax: Axes | None = None,
        *,
        dynamic_capability: float | np.ndarray | None = None,
        language: str = "en",
        **kwargs: Any,
    ) -> Axes:
        """Plot the per-band indicators F2/F3, the Ld line and F4.

        F1 and its Table B.3 limit of 0,6 are drawn beside F4 when the
        result carries them, that is when ``temporal_intensity`` was
        supplied to :func:`field_indicators`.

        Requires per-band data (call :func:`field_indicators` with 2D
        ``(positions, bands)`` arrays and ``frequencies``) and matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param dynamic_capability: Optional instrument dynamic capability
            index ``Ld`` in dB (scalar or per band), drawn as the
            criterion-1 reference line (``Ld > F2``, equation (B.1)); see
            :func:`dynamic_capability_index`.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the F2 curve ``plot`` call.
        :raises ValueError: If the result carries no per-band data.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_field_indicators

        check_language(language)
        return plot_field_indicators(
            self, ax=ax, dynamic_capability=dynamic_capability,
            language=language, **kwargs
        )


def _level(value: float, reference: float) -> float:
    """10*lg(value/reference) dB with a tiny-floor guard for zero values."""
    return float(10.0 * np.log10(max(value, np.finfo(float).tiny) / reference))


def _level_array(values: np.ndarray, reference: float) -> np.ndarray:
    guarded = np.maximum(values, np.finfo(float).tiny)
    return np.asarray(10.0 * np.log10(guarded / reference), dtype=np.float64)


def sound_intensity(
    p1: list[float] | np.ndarray,
    p2: list[float] | np.ndarray,
    fs: int,
    spacing: float,
    rho: float = 1.204,
    c: float = 343.0,
    fraction: int | None = None,
    limits: list[float] | None = None,
    bias_correct: bool = False,
) -> IntensityResult:
    """
    Sound intensity from a two-microphone (p-p) probe (IEC 61043:1993).

    The one-sided cross spectrum ``G12`` of the two microphone pressures
    is estimated with Welch-averaged, Hann-windowed segments
    (:func:`scipy.signal.csd`) and converted to the intensity spectral
    density along the probe axis (definition 3.2 of IEC 61043:1993 gives
    the underlying p-p formulation)::

        I(f) = -Im{G12(f)} / (2 * pi * f * rho * spacing)

    Positive intensity flows from microphone 1 towards microphone 2.
    The mean-square pressure is taken from the mean signal
    ``(p1 + p2)/2`` at the probe reference point. When ``fraction`` is
    given, both quantities are integrated into octave (1) or one-third
    octave (3) bands using the ANSI S1.11/IEC 61260-1 band edges of
    :func:`phonometry.nominal_frequencies`; bands without any spectral
    bin are dropped. Broadband totals are always computed (over
    ``limits`` when provided, otherwise over all positive frequencies).

    The pressure-intensity index ``Lp - LI`` is reported per band and
    broadband; in a free plane progressive wave it equals
    ``10*lg(rho*c/400)`` = 0,14 dB (IEC 61043:1993 clause 5 note), while
    large values flag reactive or noisy fields (compare with the
    instrument dynamic capability, ISO 9614-1:1993 criterion 1).

    Usable bandwidth: the finite-difference gradient biases the result
    by the factor ``sin(k*spacing)/(k*spacing)`` (IEC 61043:1993, 7.3);
    results are essentially unbiased (< ~0,3 dB) below
    ``max_valid_frequency = 0,1 * c / spacing``, and ``bias_correction``
    provides the per-band compensation factor.

    :param p1: Pressure signal of microphone 1, in pascals (1D).
    :param p2: Pressure signal of microphone 2, in pascals (1D).
    :param fs: Sample rate in Hz.
    :param spacing: Microphone separation dr, in metres.
    :param rho: Air density, in kg/m^3. Default 1,204 (20 degC).
    :param c: Speed of sound, in m/s. Default 343,0.
    :param fraction: ``None`` (broadband only), 1 (octave bands) or
        3 (one-third octave bands).
    :param limits: [f_min, f_max] band limits in Hz (default
        [12, 20000], as in :func:`phonometry.nominal_frequencies`).
    :param bias_correct: If True, apply the per-bin finite-difference
        correction ``(k*spacing)/sin(k*spacing)`` (IEC 61043:1993, 7.3) to the
        intensity spectral density before summing the band and broadband
        totals, so the totals no longer under-read as the frequency approaches
        ``max_valid_frequency``. The reciprocal diverges as ``k*spacing -> pi``
        (the first spatial-aliasing null at ``c/(2*spacing)``, inside the
        default band range for close spacings), so it is applied only over the
        probe's usable range (up to ``k*spacing = pi/2``) and held constant
        beyond, keeping the totals bounded instead of letting a few near-null
        bins dominate them. Default False keeps the exact legacy totals; the
        per-band ``bias_correction`` factor (same clamped definition) is
        reported either way.
    :return: :class:`IntensityResult`.
    """
    x1 = _typesignal(p1)
    x2 = _typesignal(p2)
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("sound_intensity expects 1D pressure signals.")
    if x1.size != x2.size:
        raise ValueError(
            f"Microphone signals must have the same length, got {x1.size} and {x2.size}."
        )
    if fs <= 0:
        raise ValueError("Sample rate 'fs' must be positive.")
    if spacing <= 0:
        raise ValueError("Microphone 'spacing' must be positive.")
    if rho <= 0:
        raise ValueError("Air density 'rho' must be positive.")
    if c <= 0:
        raise ValueError("Speed of sound 'c' must be positive.")
    if fraction is not None and fraction not in (1, 3):
        raise ValueError("'fraction' must be None, 1 (octave) or 3 (one-third octave).")
    if limits is not None and (
        len(limits) != 2 or limits[0] <= 0 or limits[1] <= limits[0]
    ):
        raise ValueError("'limits' must be [f_min, f_max] with 0 < f_min < f_max.")
    if x1.size < 32:
        raise ValueError(f"Signals too short for a spectral estimate: {x1.size} samples.")

    # Hann-windowed Welch/CSD averaging through the shared spectral core
    # (50% overlap, detrend off); the shared default segment length targets
    # a resolution of at most 4 Hz so low-frequency bands are resolved.
    nperseg = _default_nperseg(x1.size, float(fs))
    f, g12 = _welch_cross_spectrum(x1, x2, float(fs), nperseg, nperseg // 2)
    _, spp = _welch_autospectrum(
        (x1 + x2) / 2.0, float(fs), nperseg, nperseg // 2
    )
    df = float(f[1] - f[0])
    pos = f > 0
    fpos = f[pos]
    # I(f) = -Im{G12(f)} / (2*pi*f*rho*dr) per frequency bin (W/m^2/Hz).
    i_density = -np.imag(g12[pos]) / (2.0 * np.pi * fpos * rho * spacing)
    p_density = spp[pos]

    if bias_correct:
        # IEC 61043:1993, 7.3: undo the finite-difference response
        # sin(k*dr)/(k*dr) per bin. The reciprocal is clamped at k*dr = pi/2
        # (see _finite_difference_correction) so bins near the first null do
        # not diverge and dominate the totals.
        k_dr_full = 2.0 * np.pi * fpos * spacing / c
        i_density = i_density * _finite_difference_correction(k_dr_full)

    if limits is not None:
        broad = (fpos >= limits[0]) & (fpos <= limits[1])
    else:
        broad = np.ones(fpos.size, dtype=bool)
    total_i = float(np.sum(i_density[broad]) * df)
    total_msp = float(np.sum(p_density[broad]) * df)
    total_li = _level(abs(total_i), _I0)
    total_lp = _level(total_msp, _P0**2)

    frequency: np.ndarray | None = None
    intensity: np.ndarray | None = None
    intensity_level: np.ndarray | None = None
    pressure_level: np.ndarray | None = None
    pressure_intensity_index: np.ndarray | None = None
    direction: np.ndarray | None = None
    bias_correction: np.ndarray | None = None

    if fraction is not None:
        band_limits = [float(limits[0]), float(limits[1])] if limits else [12.0, 20000.0]
        freq, freq_d, freq_u, _ = _genfreqs(band_limits, fraction, fs)
        centers: list[float] = []
        band_i: list[float] = []
        band_msp: list[float] = []
        for fc, lo, hi in zip(freq, freq_d, freq_u):
            mask = (fpos > lo) & (fpos <= hi)
            if not np.any(mask):
                continue  # band narrower than the spectral resolution
            centers.append(float(fc))
            band_i.append(float(np.sum(i_density[mask]) * df))
            band_msp.append(float(np.sum(p_density[mask]) * df))
        frequency = np.asarray(centers, dtype=np.float64)
        intensity = np.asarray(band_i, dtype=np.float64)
        intensity_level = _level_array(np.abs(intensity), _I0)
        pressure_level = _level_array(np.asarray(band_msp, dtype=np.float64), _P0**2)
        pressure_intensity_index = pressure_level - intensity_level
        direction = np.where(intensity >= 0.0, 1, -1)
        # IEC 61043:1993, 7.3: finite-difference response sin(k*dr)/(k*dr)
        # with k*dr = 2*pi*f*dr/c; the reported correction is its reciprocal,
        # clamped at k*dr = pi/2 to match the applied bias_correct path and to
        # avoid diverging near the first null (_finite_difference_correction).
        k_dr = 2.0 * np.pi * frequency * spacing / c
        bias_correction = _finite_difference_correction(k_dr)

    return IntensityResult(
        frequency=frequency,
        intensity=intensity,
        intensity_level=intensity_level,
        pressure_level=pressure_level,
        pressure_intensity_index=pressure_intensity_index,
        direction=direction,
        bias_correction=bias_correction,
        total_intensity=total_i,
        total_intensity_level=total_li,
        total_pressure_level=total_lp,
        total_pressure_intensity_index=total_lp - total_li,
        total_direction=1 if total_i >= 0.0 else -1,
        max_valid_frequency=0.1 * c / spacing,
        spacing=float(spacing),
    )


# A.2.3 makes the sign of the surface sum a condition of the test itself, so
# F4 (and F3 with it) cites the standard.
_F4_NON_POSITIVE = (
    "The algebraic mean normal intensity over the measurement surface is not "
    "positive: the test conditions do not satisfy ISO 9614-1 (clause A.2.3)."
)
# A.2.1 places no positivity condition on the M short-time samples at one
# position, so F1 states the reason without attributing it to the standard:
# F1 as defined would come out negative and the Table B.3 criterion vacuous.
_F1_NON_POSITIVE = (
    "The algebraic mean of the short-time normal intensity samples is not "
    "positive, so the temporal variability indicator F1 of ISO 9614-1 "
    "(equation (A.1)) is not meaningful at this position: the field is not "
    "flowing outwards there on average."
)


def _coefficient_of_variation(
    samples: np.ndarray, *, non_positive_message: str
) -> float:
    """``(1/mean) * sqrt(sum((x - mean)^2) / (n - 1))`` of one sample set.

    The common closed form of the ISO 9614-1:1993 indicators F1 (equation
    (A.1), M short-time samples at one position) and F4 (equation (A.8), N
    positions on the surface): the sample standard deviation of the normal
    intensity normalized by its algebraic mean.

    The two indicators reject a non-positive mean for different reasons, so
    the caller supplies the message: A.2.3 makes a negative surface sum a
    failure of the test conditions, whereas A.2.1 states no such condition on
    the M temporal samples and the rejection there is this library's own.
    """
    # A NaN or infinite sample slips past the positivity test below and turns
    # the whole indicator into a silent NaN, so it is rejected up front.
    if not np.all(np.isfinite(samples)):
        raise ValueError("The normal intensity samples must all be finite.")
    mean = float(np.mean(samples))
    if mean <= 0.0:
        raise ValueError(non_positive_message)
    deviation = float(
        np.sqrt(np.sum((samples - mean) ** 2) / (samples.size - 1))
    )
    return deviation / mean


def temporal_variability_indicator(
    short_time_intensity: list[float] | np.ndarray,
) -> float | np.ndarray:
    """
    ISO 9614-1:1993 temporal variability indicator F1 (equation (A.1)).

    In the initial test a "typical" measurement position is chosen on an
    initial measurement surface and the normal sound intensity is sampled
    there M times with a short averaging time (clause 8.2). The indicator
    is the coefficient of variation of those samples::

        F1 = (1 / In) * sqrt( sum_k (Ink - In)**2 / (M - 1) )

    with ``In`` the mean of the M samples (equation (A.2)). It is
    dimensionless, and it is zero for a perfectly steady field, so it
    qualifies the *stationarity of the field*, not the uniformity of the
    surface (that is F4). Note 9 of Annex A recommends M = 10 samples, with
    a short averaging time of 8 s to 12 s (or a whole number of cycles) for
    periodic signals. Table B.3 requires corrective action when
    ``F1 > 0,6`` (:data:`TEMPORAL_VARIABILITY_LIMIT`).

    :param short_time_intensity: The M short-time-averaged signed normal
        intensity samples ``Ink`` at the fixed position, in W/m^2; 1D
        ``(samples,)`` for one frequency band, or 2D ``(samples, bands)`` to
        evaluate every band at once (Annex A.1 requires F1 in each band used
        for the sound-power determination).
    :return: F1 as a float for 1D input, or a per-band
        :class:`numpy.ndarray` for 2D input.
    :raises ValueError: If fewer than two samples are supplied, the input is
        not 1D or 2D, or the mean intensity of a band is not positive. A.2.1
        states no positivity condition on the M samples, but F1 normalizes by
        that mean, so a non-positive one leaves the indicator meaningless and
        the Table B.3 criterion vacuous; this library rejects it rather than
        return a negative F1.
    """
    samples = np.atleast_1d(np.asarray(short_time_intensity, dtype=np.float64))
    if samples.ndim not in (1, 2):
        raise ValueError(
            "temporal_variability_indicator expects 1D (samples,) or 2D "
            "(samples, bands) arrays."
        )
    if samples.shape[0] < 2:
        raise ValueError(
            "At least two short-time samples are required for F1 "
            "(ISO 9614-1 Annex A Note 9 recommends M = 10)."
        )
    if samples.ndim == 1:
        return _coefficient_of_variation(
            samples, non_positive_message=_F1_NON_POSITIVE
        )
    return np.asarray(
        [
            _coefficient_of_variation(
                samples[:, b], non_positive_message=_F1_NON_POSITIVE
            )
            for b in range(samples.shape[1])
        ],
        dtype=np.float64,
    )


def _field_indicators_1d(
    lp: np.ndarray, i_n: np.ndarray
) -> tuple[float, float, float]:
    """F2/F3/F4 of one band from the per-position ``Lpi`` and ``Ini``."""
    # Equation (A.4): surface pressure level.
    lp_surface = energy_mean(lp)
    # Equation (A.5): level of the mean magnitude of the normal intensity.
    li_abs = _level(float(np.mean(np.abs(i_n))), _I0)
    # Equation (A.8) with (A.9): normalized non-uniformity of In (the same
    # coefficient of variation F1 uses, over the N positions instead of the M
    # short-time samples). It rejects a non-positive algebraic mean first.
    f4 = _coefficient_of_variation(i_n, non_positive_message=_F4_NON_POSITIVE)
    # Equation (A.7): algebraic surface intensity level.
    li_alg = _level(float(np.mean(i_n)), _I0)
    return lp_surface - li_abs, lp_surface - li_alg, f4


def _temporal_indicator_for(
    temporal_intensity: list[float] | np.ndarray | None,
    n_bands: int,
    *,
    scalar_surface: bool,
) -> float | np.ndarray | None:
    """F1 from the optional short-time samples, checked against the band count.

    ``None`` in, ``None`` out: F1 is optional on a :class:`FieldIndicators`,
    because it comes from the initial test and not from the surface scan.

    F1 follows the shape of its siblings. A 1D per-position surface carries
    scalar F2/F3/F4, so a one-column ``(M, 1)`` sample array is unwrapped to a
    scalar F1 too; otherwise
    :meth:`FieldIndicators.field_is_stationary` would answer such a result
    with ``array([True])`` where ``f2``/``f3``/``f4`` are plain floats. A 2D
    ``(positions, 1)`` surface keeps per-band arrays throughout, so F1 stays a
    length-one array there.

    :raises ValueError: If the samples do not carry one column per band of the
        surface input.
    """
    if temporal_intensity is None:
        return None
    f1 = temporal_variability_indicator(temporal_intensity)
    f1_size = 1 if np.ndim(f1) == 0 else np.size(f1)
    if f1_size != n_bands:
        raise ValueError(
            f"'temporal_intensity' must carry one column per band, got "
            f"{f1_size} for {n_bands} band(s)."
        )
    if scalar_surface and np.ndim(f1) != 0:
        return float(np.asarray(f1).reshape(()))
    if not scalar_surface and np.ndim(f1) == 0:
        return np.asarray([f1], dtype=np.float64)
    return f1


def field_indicators(
    pressure_levels: list[float] | np.ndarray,
    normal_intensity: list[float] | np.ndarray,
    frequencies: list[float] | np.ndarray | None = None,
    *,
    temporal_intensity: list[float] | np.ndarray | None = None,
) -> FieldIndicators:
    """
    ISO 9614-1:1993 Annex A field indicators F1 to F4.

    Given the sound pressure level ``Lpi`` (dB) and the signed normal
    sound intensity ``Ini`` (W/m^2) measured at each of the N discrete
    positions on the measurement surface:

    - F2 = Lp - L|In| (equation (A.3)), with the surface pressure level
      from equation (A.4) and the level of the mean magnitude of the
      normal intensity from equation (A.5);
    - F3 = Lp - LIn (equation (A.6)), with the algebraic surface
      intensity level from equation (A.7);
    - F4 = (1/|mean In|) * sqrt(sum((Ini - mean In)^2) / (N - 1))
      (equations (A.8)-(A.9)).

    F1, the temporal variability indicator (equation (A.1)), does not come
    from the surface scan: it is evaluated in the initial test at one
    typical position from M short-time-averaged intensity samples
    (clause 8.2), and again immediately before and after the measurement
    on any one measurement surface (Annex B, B.1.4). Passing those samples
    as ``temporal_intensity`` fills ``f1`` on the result; see
    :func:`temporal_variability_indicator`.

    The inputs are either 1D per-position arrays (one frequency band,
    scalar indicators) or 2D ``(positions, bands)`` arrays (the
    indicators are evaluated band by band and returned as per-band
    arrays; the plottable form). If the algebraic mean intensity of any
    band is not positive the test conditions do not satisfy ISO 9614-1
    in that band (clause A.2.3) and a ``ValueError`` is raised.

    :param pressure_levels: Lpi at each position, in decibels; 1D
        ``(positions,)`` or 2D ``(positions, bands)``.
    :param normal_intensity: Signed normal intensity Ini at each
        position, in W/m^2; same shape as ``pressure_levels``.
    :param frequencies: Band centre frequencies in Hz, one per column of
        the 2D input (optional for 1D input, where it is ignored beyond
        a length check).
    :param temporal_intensity: Optional M short-time-averaged normal
        intensity samples ``Ink`` at one fixed position, in W/m^2; 1D
        ``(samples,)`` for a single band or 2D ``(samples, bands)`` with
        one column per band of the surface input. Supplying it fills
        ``f1`` on the returned result.
    :return: :class:`FieldIndicators`.
    """
    lp = np.atleast_1d(np.asarray(pressure_levels, dtype=np.float64))
    i_n = np.atleast_1d(np.asarray(normal_intensity, dtype=np.float64))
    if lp.ndim not in (1, 2) or i_n.ndim not in (1, 2):
        raise ValueError(
            "field_indicators expects 1D (positions,) or 2D (positions, bands) "
            "arrays."
        )
    if lp.shape != i_n.shape:
        raise ValueError(
            f"'pressure_levels' and 'normal_intensity' must have the same "
            f"shape, got {lp.shape} and {i_n.shape}."
        )
    if lp.shape[0] < 2:
        raise ValueError("At least two measurement positions are required.")

    n_bands = lp.shape[1] if lp.ndim == 2 else 1
    freqs: np.ndarray | None = None
    if frequencies is not None:
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        if freqs.size != n_bands:
            raise ValueError(
                f"'frequencies' must have one entry per band, got {freqs.size} "
                f"for {n_bands} band(s)."
            )

    f1 = _temporal_indicator_for(
        temporal_intensity, n_bands, scalar_surface=lp.ndim == 1
    )

    if lp.ndim == 1:
        f2, f3, f4 = _field_indicators_1d(lp, i_n)
        return FieldIndicators(f2=f2, f3=f3, f4=f4, frequency=freqs, f1=f1)

    per_band = [
        _field_indicators_1d(lp[:, b], i_n[:, b]) for b in range(lp.shape[1])
    ]
    values = np.asarray(per_band, dtype=np.float64)
    return FieldIndicators(
        f2=values[:, 0], f3=values[:, 1], f4=values[:, 2], frequency=freqs, f1=f1
    )


def dynamic_capability_index(
    pressure_residual_intensity_index: float, bias_error_factor: float = 10.0
) -> float:
    """
    Dynamic capability index Ld (ISO 9614-1:1993, clause 3.12).

    ``Ld = delta_pI0 - K`` (equation (10)), where ``delta_pI0`` is the
    instrument pressure-residual intensity index (clause 3.11, equation
    (9); determined per IEC 61043:1993, which requires the Table 2
    minima per class) and ``K`` the bias error factor of Table 1: 10 dB
    for precision (grade 1) and engineering (grade 2) measurements, 7 dB
    for survey (grade 3). The measurement arrangement is adequate when
    ``Ld > F2`` (criterion 1, Annex B equation (B.1)).

    :param pressure_residual_intensity_index: delta_pI0 in decibels.
    :param bias_error_factor: K in decibels (default 10,0).
    :return: Ld in decibels.
    """
    if bias_error_factor <= 0:
        raise ValueError("'bias_error_factor' must be positive.")
    return float(pressure_residual_intensity_index - bias_error_factor)
