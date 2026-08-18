#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Core processing logic and FilterBank class for phonometry.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, cast, overload

import numpy as np
from scipy import signal

from .._internal.utils import _downsamplingfactor, _resample_to_length
from .._internal.warnings import PhonometryWarning
from ..io._resolve import refuse_foreign_rate, resolve_fs, resolve_samples
from ..io._signal import Signal
from .design import _cheby2_headroom, _design_sos_filter
from .frequencies import _genfreqs


class FilterBankWarning(PhonometryWarning):
    """Warns about fractional-octave filter-bank processing pitfalls."""


#: Filter families the bank knows how to design (see ``OctaveFilterBank``).
_VALID_FILTERS = ["butter", "cheby1", "cheby2", "ellip", "bessel"]


@dataclass(frozen=True)
class FilterDesign:
    """How the band-pass filters of a bank are designed.

    The defaults are the design used everywhere in the library: Butterworth
    sections, evaluated band by band on a decimated sample rate.

    :ivar filter_type: Type of filter ('butter', 'cheby1', 'cheby2', 'ellip',
        'bessel'). Only ``butter`` meets IEC 61260-1:2014 class 1 with the
        default parameters (``cheby2`` also does once ``attenuation`` >= 70 dB,
        see below); ``cheby1``/``ellip``/``bessel`` fail on passband ripple or
        roll-off regardless of parameters (default 'butter').
    :ivar ripple: Passband ripple in dB, for cheby1 and ellip (default 0.1).
    :ivar attenuation: Stopband attenuation in dB (default 72.0). For the
        ``cheby2`` filter scipy pins the equiripple deep-stopband floor at
        exactly this value, so it must be >= 70 dB for the bank to meet the
        IEC 61260-1:2014 class 1 deep-stopband limit (Omega >= G^4). The
        72 dB default clears class 1 with the same +0.400 dB passband margin
        as ``butter``.
    :ivar resample: If True, resampling is performed: each band is filtered on
        a decimated sample rate (default True).
    """

    filter_type: str = "butter"
    ripple: float = 0.1
    attenuation: float = 72.0
    resample: bool = True


@dataclass(frozen=True)
class LevelCalibration:
    """How the energy in a band becomes a level reading.

    :ivar factor: Calibration factor for SPL calculation: multiplies the
        digital amplitude to obtain pascals (default 1.0).
    :ivar dbfs: If True, calculate SPL in dBFS, where 0 dB is a full-scale
        RMS of 1.0, instead of dB SPL re 20 uPa; ``factor`` is then not
        applied (default False).
    """

    factor: float = 1.0
    dbfs: bool = False


@dataclass(frozen=True)
class BlockProcessing:
    """How the bank carries its filter state from one block to the next.

    :ivar stateful: If True, carry filter state between calls. Useful for
        block processing (default False).
    :ivar steady_ic: If True, calculate steady state initial conditions for
        filter (default False).
    """

    stateful: bool = False
    steady_ic: bool = False


@dataclass(frozen=True)
class ResponsePlot:
    """The filter-response plot drawn while the bank is being designed.

    :ivar show: If True, show the filter response plot (default False).
    :ivar file: Path to save the filter response plot (default None).
    """

    show: bool = False
    file: str | None = None


#: The defaults of the bank and of :func:`octave_filter`: the IEC 61260-1
#: class 1 design, no calibration, no carried state and no response plot.
#: One shared instance each: the bundles are frozen, so a call cannot
#: mutate them.
_DEFAULT_DESIGN = FilterDesign()
_DEFAULT_CALIBRATION = LevelCalibration()
_DEFAULT_BLOCK_PROCESSING = BlockProcessing()
_DEFAULT_RESPONSE_PLOT = ResponsePlot()


def _validate_bank_design(
    fs: int,
    fraction: float,
    order: int,
    limits: list[float] | None,
    design: FilterDesign,
    block_processing: BlockProcessing,
) -> list[float]:
    """Check the bank design arguments and resolve the frequency limits.

    Raises :class:`ValueError` for any unusable combination and returns the
    limits actually used, filling in the default ``[12, 20000]`` Hz span when
    none was given.
    """
    if fs <= 0:
        raise ValueError("Sample rate 'fs' must be positive.")
    if fraction <= 0:
        raise ValueError("Bandwidth 'fraction' must be positive.")
    if order <= 0:
        raise ValueError("Filter 'order' must be positive.")
    if limits is None:
        limits = [12, 20000]
    if len(limits) != 2:
        raise ValueError("Limits must be a list of two frequencies [f_min, f_max].")
    if limits[0] <= 0 or limits[1] <= 0:
        raise ValueError("Limit frequencies must be positive.")
    if limits[0] >= limits[1]:
        raise ValueError("The lower limit must be less than the upper limit.")
    if design.filter_type not in _VALID_FILTERS:
        raise ValueError(f"Invalid filter_type. Must be one of {_VALID_FILTERS}")
    if design.resample and block_processing.stateful:
        # a stateful resampling algorithm would be required...
        raise ValueError("Resampling and stateful behaviour (block processing) are not supported.")
    return limits


class OctaveFilterBank:
    """
    A class-based representation of an Octave Filter Bank.
    Allows for pre-calculating and reusing filter coefficients.
    """
    
    def __init__(
        self,
        fs: int,
        fraction: float = 1,
        order: int = 6,
        limits: list[float] | None = None,
        *,
        design: FilterDesign = _DEFAULT_DESIGN,
        calibration: LevelCalibration = _DEFAULT_CALIBRATION,
        block_processing: BlockProcessing = _DEFAULT_BLOCK_PROCESSING,
        response_plot: ResponsePlot = _DEFAULT_RESPONSE_PLOT,
    ) -> None:
        """
        Initialize the Octave Filter Bank.

        :param fs: Sample rate in Hz.
        :param fraction: Bandwidth fraction (e.g., 1 for octave, 3 for 1/3 octave).
        :param order: Filter order.
        :param limits: Frequency limits [f_min, f_max].
        :param design: How the band filters are designed: family, ripple,
            stopband attenuation and multirate decimation
            (:class:`FilterDesign`). Only ``butter`` meets IEC 61260-1:2014
            class 1 with the default parameters (``cheby2`` also does once
            ``attenuation`` >= 70 dB); ``cheby1``/``ellip``/``bessel`` fail on
            passband ripple or roll-off regardless of parameters.
        :param calibration: How band energy becomes a level: calibration
            factor and dBFS switch (:class:`LevelCalibration`).
        :param block_processing: Whether the bank carries its filter state
            between calls, and how that state starts (:class:`BlockProcessing`).
        :param response_plot: Whether to show or save the filter response plot
            drawn while designing the bank (:class:`ResponsePlot`).
        """
        limits = _validate_bank_design(fs, fraction, order, limits, design, block_processing)

        self.fs = fs
        self.fraction = fraction
        self.order = order
        self.limits = limits
        self.filter_type = design.filter_type
        self.ripple = design.ripple
        self.attenuation = design.attenuation
        self.calibration_factor = calibration.factor
        self.dbfs = calibration.dbfs
        self.stateful = block_processing.stateful

        # Generate frequencies
        self.freq, self.freq_d, self.freq_u, self.nominal_freq = _genfreqs(limits, fraction, fs)
        self.num_bands = len(self.freq)


        # Calculate factors and design SOS
        if design.resample:
            headroom = 1.25
            if design.filter_type == "cheby2":
                # The cheby2 stopband extends above the band's upper edge;
                # 5% safety margin so it clears the decimated Nyquist.
                headroom = max(
                    headroom, 1.05 * _cheby2_headroom(fraction, order, design.attenuation)
                )
            self.factor = _downsamplingfactor(self.freq_u, fs, headroom)
        else:
            self.factor = np.ones(self.num_bands, dtype=int)

        self.sos = _design_sos_filter(
            self.freq, self.freq_d, self.freq_u, fs, order, self.factor,
            design.filter_type, design.ripple, design.attenuation,
            response_plot.show, response_plot.file
        )

        # Calculate initial conditions for filter state
        if self.stateful:
            self._init_filter_state(block_processing.steady_ic)


    def _init_filter_state(self, steady_ic: bool) -> None:
        """Initialize filter state (zi) for stateful block-wise processing.

        Uses lazy initialization: zi arrays are allocated on first use in
        _filter_and_resample() so the channel count matches the actual input.
        """
        self.zi: list[np.ndarray] = [np.array([]) for _ in range(self.num_bands)]
        self._steady_ic = steady_ic


    def __repr__(self) -> str:
        return (
            f"OctaveFilterBank(fs={self.fs}, fraction={self.fraction}, order={self.order}, "
            f"limits={self.limits}, filter_type='{self.filter_type}', "
            f"num_bands={self.num_bands})"
        )

    @overload
    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: Literal[False] = False,
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: Literal[True] = True,
        nominal: Literal[False] = False,
        zero_phase: bool = False,
    ) -> tuple[np.ndarray, list[float]]: ...

    @overload
    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: Literal[True],
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: Literal[True] = True,
        nominal: Literal[False] = False,
        zero_phase: bool = False,
    ) -> tuple[np.ndarray, list[float], list[np.ndarray]]: ...

    @overload
    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: Literal[False] = False,
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: Literal[False] = False,
        nominal: Literal[False] = False,
        zero_phase: bool = False,
    ) -> tuple[None, list[float]]: ...

    @overload
    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: Literal[True],
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: Literal[False] = False,
        nominal: Literal[False] = False,
        zero_phase: bool = False,
    ) -> tuple[None, list[float], list[np.ndarray]]: ...

    @overload
    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: Literal[False] = False,
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: Literal[True] = True,
        nominal: Literal[True] = ...,
        zero_phase: bool = False,
    ) -> tuple[np.ndarray, list[str]]: ...

    @overload
    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: Literal[True],
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: Literal[True] = True,
        nominal: Literal[True] = ...,
        zero_phase: bool = False,
    ) -> tuple[np.ndarray, list[str], list[np.ndarray]]: ...

    @overload
    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: Literal[False] = False,
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: Literal[False] = False,
        nominal: Literal[True] = ...,
        zero_phase: bool = False,
    ) -> tuple[None, list[str]]: ...

    @overload
    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: Literal[True],
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: Literal[False] = False,
        nominal: Literal[True] = ...,
        zero_phase: bool = False,
    ) -> tuple[None, list[str], list[np.ndarray]]: ...

    def filter(
        self,
        x: Signal | list[float] | np.ndarray,
        sigbands: bool = False,
        mode: str = "rms",
        detrend: bool = True,
        calculate_level: bool = True,
        nominal: bool = False,
        zero_phase: bool = False,
    ) -> tuple[np.ndarray | None, list[float] | list[str]] | tuple[np.ndarray | None, list[float] | list[str], list[np.ndarray]]:
        """
        Apply the pre-designed filter bank to a signal.

        :param x: Input signal (1D array or 2D array [channels, samples]), or
            a :class:`phonometry.io.Signal`. A Signal recorded at another rate
            than the one this bank was designed for is refused rather than
            filtered; a calibrated one is filtered in pascals unless the bank
            carries a calibration factor of its own, or reads in dBFS.
        :param sigbands: If True, also return the signal in the time domain divided into bands.
        :param mode: 'rms' for energy-based level, 'peak' for peak-holding level.
            Note: 'peak' includes the filter's onset transient; a tone that
            starts abruptly can overshoot by ~1 dB. For steady signals,
            discard the first ~5/f_low seconds or use longer signals.
        :param detrend: If True, remove DC offset from signal before filtering (Default: True).
        :param calculate_level: If True, calculate SPL.
        :param nominal: If True, return IEC 61260-1 nominal frequency labels (List[str]) instead of exact floats.
        :param zero_phase: If True, filter with ``sosfiltfilt`` (forward-backward):
            no group delay, but the effective stopband attenuation doubles and
            the effective passband narrows. The narrowing lowers the measured
            broadband band level by about 0.2 to 0.3 dB per band relative to
            forward filtering (a pure in-band tone is unaffected, since it sits
            where both passes are ~0 dB). Prefer forward filtering when the
            absolute band SPL must match single-pass conventions; use zero-phase
            when preserving the temporal envelope matters (e.g. reverberation
            decay, ISO 3382-2 Clause 7.3). Offline analysis only; incompatible
            with stateful mode.
        :return: A tuple containing (SPL_array, Frequencies_list) or (SPL_array, Frequencies_list, signals).
        """
        if zero_phase and self.stateful:
            raise ValueError("zero_phase is not compatible with stateful processing.")

        x_proc, is_multichannel = self._prepare_signal(x, detrend)
        num_channels = x_proc.shape[0]

        # Process signal across all bands and channels
        spl, xb = self._process_bands(
            x_proc, num_channels, sigbands, mode=mode,
            calculate_level=calculate_level, zero_phase=zero_phase,
        )

        # Format output based on input dimensionality
        if not is_multichannel:
            if spl is not None:
                spl = spl[0]
            if sigbands and xb is not None:
                xb = [band[0] for band in xb]

        # Return a copy: the bank (possibly shared via the octave_filter()
        # design cache) must not be corrupted by callers mutating the list.
        freq_out: list[float] | list[str] = list(self.nominal_freq) if nominal else list(self.freq)

        if sigbands and xb is not None:
            return spl, freq_out, xb
        return spl, freq_out

    def _prepare_signal(
        self, x: Signal | list[float] | np.ndarray, detrend: bool
    ) -> tuple[np.ndarray, bool]:
        """Coerce the input to a 2D ``[channels, samples]`` array for filtering.

        Optionally removes the DC offset (warning when that is done on a
        stateful bank, where it can introduce discontinuities between blocks)
        and reports whether the caller supplied multichannel input, so the
        caller can squeeze the results back to 1D.

        A :class:`~phonometry.io.Signal` recorded at another rate is refused
        here rather than filtered by a bank designed for a different one.
        Its calibration is honoured only when the bank carries none of its
        own: the bank's ``LevelCalibration`` is the explicit knob, and
        applying both would square the factor. A dBFS bank ignores it
        outright -- dBFS is referenced to digital full scale, so converting
        the samples to pascals first would offset every band by
        ``20*log10(factor)`` while still calling the result dBFS.
        """
        refuse_foreign_rate(x, self.fs, "filter bank")
        calibrate = self.calibration_factor == 1.0 and not self.dbfs
        x_proc = resolve_samples(x, calibrate=calibrate)

        if detrend:
            if self.stateful:
                warnings.warn(
                    "Detrending is not recommended during block processing "
                    "as it can introduce discontinuities between blocks.",
                    FilterBankWarning,
                    stacklevel=3,
                )
            # Axis -1 handles both 1D and 2D arrays correctly
            x_proc = signal.detrend(x_proc, axis=-1, type='constant')

        is_multichannel = x_proc.ndim > 1
        if not is_multichannel:
            x_proc = x_proc[np.newaxis, :]  # Standardize to 2D
        return x_proc, is_multichannel

    def spectrogram(
        self,
        x: Signal | list[float] | np.ndarray,
        window_time: float = 0.125,
        overlap: float = 0.5,
        mode: str = "rms",
        detrend: bool = True,
        zero_phase: bool = False,
    ) -> tuple[np.ndarray, list[float], np.ndarray]:
        """
        Short-time fractional-octave analysis: level per band over time.

        :param x: Input signal (1D array or 2D array [channels, samples]), or
            a :class:`phonometry.io.Signal`. It takes the same door as
            :meth:`filter`: a Signal at another rate is refused, and a
            calibrated one is analysed in pascals unless this bank carries a
            calibration factor of its own or reads in dBFS. The two methods
            must not report different levels for the same recording.
        :param window_time: Analysis window length in seconds.
        :param overlap: Window overlap fraction in [0, 1).
        :param mode: 'rms' or 'peak' (per window).
        :param detrend: If True, remove DC offset before filtering.
        :param zero_phase: If True, filter bands forward-backward so their
            group delays don't skew the frames (offline analysis only).
        :return: Tuple (levels, freq, times). ``levels`` has shape
            (num_bands, num_frames) for 1D input and
            (channels, num_bands, num_frames) for 2D input; ``times`` holds
            each window's center in seconds.
        """
        if self.stateful:
            raise ValueError("spectrogram() is not supported on stateful banks.")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1).")

        refuse_foreign_rate(x, self.fs, "filter bank")
        x_proc = resolve_samples(
            x, calibrate=self.calibration_factor == 1.0 and not self.dbfs
        )
        is_multichannel = x_proc.ndim > 1
        n_samples = x_proc.shape[-1]
        win = round(window_time * self.fs)
        hop = max(1, round(win * (1 - overlap)))
        if win <= 0 or win > n_samples:
            raise ValueError("window_time must be positive and shorter than the signal.")

        # Filter once per band at full rate so windows stay time-aligned
        # across bands regardless of per-band decimation.
        if detrend:
            x_proc = signal.detrend(x_proc, axis=-1, type='constant')
        x_2d = x_proc if is_multichannel else x_proc[np.newaxis, :]
        _, bands_opt = self._process_bands(
            x_2d, x_2d.shape[0], sigbands=True, mode=mode,
            calculate_level=False, zero_phase=zero_phase,
        )
        # sigbands=True always fills the band list
        bands = cast(list[np.ndarray], bands_opt)

        starts = np.arange(0, n_samples - win + 1, hop)
        times = (starts + win / 2) / self.fs

        levels = np.zeros((x_2d.shape[0], self.num_bands, len(starts)))
        # Frames are processed in chunks: the strided view itself is free,
        # but reducing it materializes temporaries of chunk*win samples, so
        # chunking keeps memory bounded for long signals / high overlap.
        frame_chunk = 256
        for b, yb in enumerate(bands):
            windows = np.lib.stride_tricks.sliding_window_view(yb, win, axis=-1)[:, ::hop, :]
            for j0 in range(0, windows.shape[1], frame_chunk):
                seg = windows[:, j0:j0 + frame_chunk, :]
                levels[:, b, j0:j0 + frame_chunk] = self._calculate_level(seg, mode)

        if not is_multichannel:
            return levels[0], list(self.freq), times
        return levels, list(self.freq), times

    def _process_bands(
        self,
        x_proc: np.ndarray,
        num_channels: int,
        sigbands: bool,
        mode: str = "rms",
        calculate_level: bool = True,
        zero_phase: bool = False,
    ) -> tuple[np.ndarray | None, list[np.ndarray] | None]:
        """
        Process signal through each frequency band.

        :param x_proc: Standardized 2D input signal [channels, samples].
        :param num_channels: Number of channels.
        :param sigbands: If True, return filtered bands.
        :param mode: 'rms' or 'peak'.
        :param calculate_level: If True, calculate SPL
        :param zero_phase: If True, use forward-backward filtering.
        :return: A tuple containing (SPL_array, Optional_List_of_filtered_signals).
        """
        if calculate_level:
            spl = np.zeros([num_channels, self.num_bands])
        else:
            spl = None
        xb: list[np.ndarray] | None = [np.array([]) for _ in range(self.num_bands)] if sigbands else None

        for idx in range(self.num_bands):
            # Vectorized processing for all channels
            filtered_signal = self._filter_and_resample(x_proc, idx, zero_phase)

            if calculate_level and spl is not None:
                # Sound Level Calculation (returns array of shape [num_channels])
                spl[:, idx] = self._calculate_level(filtered_signal, mode)

            if sigbands and xb is not None:
                # Restore original length
                # filtered_signal is [channels, downsampled_samples]
                y_resampled = _resample_to_length(filtered_signal, int(self.factor[idx]), x_proc.shape[1])
                xb[idx] = y_resampled

        return spl, xb


    def _filter_and_resample(self, x: np.ndarray, idx: int, zero_phase: bool = False) -> np.ndarray:
        """Resample and filter for a specific band (vectorized)."""
        if self.factor[idx] > 1:
            # axis=-1 is default for resample_poly, but being explicit is good
            sd = signal.resample_poly(x, 1, self.factor[idx], axis=-1)
        else:
            sd = x

        if zero_phase:
            # sosfiltfilt requires padlen < n - 1; heavily decimated bands can
            # be shorter than the default padding, so clamp it.
            n_sections = self.sos[idx].shape[0]
            padlen = min(3 * (2 * n_sections + 1), max(sd.shape[-1] - 2, 0))
            y = signal.sosfiltfilt(self.sos[idx], sd, axis=-1, padlen=padlen)
        elif self.stateful:
            n_channels = sd.shape[0]
            # Lazy init: allocate zi with correct channel count on first use
            if self.zi[idx].ndim < 3 or self.zi[idx].shape[1] != n_channels:
                n_sections = self.sos[idx].shape[0]
                if not self._steady_ic:
                    self.zi[idx] = np.zeros((n_sections, n_channels, 2))
                else:
                    zi_base = signal.sosfilt_zi(self.sos[idx])
                    self.zi[idx] = np.tile(zi_base[:, np.newaxis, :], (1, n_channels, 1))
            y, self.zi[idx] = signal.sosfilt(self.sos[idx], sd, axis=-1, zi=self.zi[idx])
        else:
            y = signal.sosfilt(self.sos[idx], sd, axis=-1)

        # sosfilt supports axis=-1 by default
        return cast(np.ndarray, y)

    def _calculate_level(self, y: np.ndarray, mode: str) -> float | np.ndarray:
        """Calculate the level (RMS or Peak) in dB."""
        if mode.lower() == "rms":
            # Use norm for better performance and reduced memory overhead
            # RMS = ||y|| / sqrt(N)
            val_linear = np.linalg.norm(y, axis=-1) / np.sqrt(y.shape[-1])
        elif mode.lower() == "peak":
            val_linear = np.max(np.abs(y), axis=-1)
        else:
            raise ValueError("Invalid mode. Use 'rms' or 'peak'.")

        eps = np.finfo(float).eps
        
        # Ensure val_linear is at least eps to avoid log(0)
        val_linear = np.maximum(val_linear, eps)

        if self.dbfs:
            # dBFS: 0 dB is RMS = 1.0 or Peak = 1.0
            return cast(np.ndarray, 20 * np.log10(val_linear))
        
        # Physical SPL: apply sensitivity and use 20uPa reference
        pressure_pa = val_linear * self.calibration_factor
        return cast(np.ndarray, 20 * np.log10(np.maximum(pressure_pa, eps) / 2e-5))


@lru_cache(maxsize=32)
def _cached_filter_bank(
    fs: int,
    fraction: float,
    order: int,
    limits: tuple[float, ...] | None,
    design: FilterDesign,
    calibration: LevelCalibration,
) -> OctaveFilterBank:
    """Design (or reuse) a stateless filter bank for octave_filter()."""
    return OctaveFilterBank(
        fs=fs,
        fraction=fraction,
        order=order,
        limits=list(limits) if limits is not None else None,
        design=design,
        calibration=calibration,
    )


@overload
def octave_filter(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    fraction: float = 1,
    order: int = 6,
    limits: list[float] | None = None,
    *,
    sigbands: Literal[False] = False,
    detrend: bool = True,
    mode: str = "rms",
    nominal: Literal[False] = False,
    design: FilterDesign = _DEFAULT_DESIGN,
    calibration: LevelCalibration = _DEFAULT_CALIBRATION,
    response_plot: ResponsePlot = _DEFAULT_RESPONSE_PLOT,
) -> tuple[np.ndarray, list[float]]: ...


@overload
def octave_filter(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    fraction: float = 1,
    order: int = 6,
    limits: list[float] | None = None,
    *,
    sigbands: Literal[True] = True,
    detrend: bool = True,
    mode: str = "rms",
    nominal: Literal[False] = False,
    design: FilterDesign = _DEFAULT_DESIGN,
    calibration: LevelCalibration = _DEFAULT_CALIBRATION,
    response_plot: ResponsePlot = _DEFAULT_RESPONSE_PLOT,
) -> tuple[np.ndarray, list[float], list[np.ndarray]]: ...


@overload
def octave_filter(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    fraction: float = 1,
    order: int = 6,
    limits: list[float] | None = None,
    *,
    sigbands: Literal[False] = False,
    detrend: bool = True,
    mode: str = "rms",
    nominal: Literal[True] = ...,
    design: FilterDesign = _DEFAULT_DESIGN,
    calibration: LevelCalibration = _DEFAULT_CALIBRATION,
    response_plot: ResponsePlot = _DEFAULT_RESPONSE_PLOT,
) -> tuple[np.ndarray, list[str]]: ...


@overload
def octave_filter(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    fraction: float = 1,
    order: int = 6,
    limits: list[float] | None = None,
    *,
    sigbands: Literal[True] = True,
    detrend: bool = True,
    mode: str = "rms",
    nominal: Literal[True] = ...,
    design: FilterDesign = _DEFAULT_DESIGN,
    calibration: LevelCalibration = _DEFAULT_CALIBRATION,
    response_plot: ResponsePlot = _DEFAULT_RESPONSE_PLOT,
) -> tuple[np.ndarray, list[str], list[np.ndarray]]: ...


def octave_filter(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    fraction: float = 1,
    order: int = 6,
    limits: list[float] | None = None,
    *,
    sigbands: bool = False,
    detrend: bool = True,
    mode: str = "rms",
    nominal: bool = False,
    design: FilterDesign = _DEFAULT_DESIGN,
    calibration: LevelCalibration = _DEFAULT_CALIBRATION,
    response_plot: ResponsePlot = _DEFAULT_RESPONSE_PLOT,
) -> tuple[np.ndarray, list[float]] | tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[float], list[np.ndarray]] | tuple[np.ndarray, list[str], list[np.ndarray]]:
    """
    Filter a signal with octave or fractional octave filter bank.

    This method uses a filter bank with Second-Order Sections (SOS) coefficients.
    To obtain the correct coefficients, automatic subsampling is applied to the
    signal in each filtered band.

    Multichannel support: If x is 2D (channels, samples), each channel is filtered.

    :param x: Input signal (1D array or 2D array [channels, samples]), or a
        :class:`phonometry.io.Signal` read from a measurement file.
    :type x: Union[List[float], np.ndarray, phonometry.io.Signal]
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :type fs: Optional[int]
    :param fraction: Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1, 2/3-octave b=1.5. Default: 1.
    :type fraction: float
    :param order: Order of the filter. Default: 6.
    :type order: int
    :param limits: Minimum and maximum limit frequencies [f_min, f_max]. Default [12, 20000].
    :type limits: Optional[List[float]]
    :param sigbands: If True, also return the signal in the time domain divided into bands.
    :type sigbands: bool
    :param detrend: If True, remove DC offset before filtering. Default: True.
    :type detrend: bool
    :param mode: 'rms' or 'peak'. Default: 'rms'.
    :param nominal: If True, return IEC 61260-1 nominal frequency labels (List[str]) instead of exact floats.
    :param design: How the band filters are designed: family, ripple, stopband
        attenuation and multirate decimation (:class:`FilterDesign`). The
        default 'butter' family is the only one that meets IEC 61260-1 class 1
        with the default parameters; for ``cheby2`` scipy pins the
        deep-stopband floor at exactly ``attenuation``, so it must be >= 70 dB
        to clear the class 1 limit (matches :class:`OctaveFilterBank`).
    :param calibration: How band energy becomes a level: calibration factor
        and dBFS switch (:class:`LevelCalibration`). This is the explicit
        knob: when its ``factor`` is left at 1.0, a calibrated
        :class:`~phonometry.io.Signal` supplies its own and the band levels
        come out in dB SPL; when it carries a factor, the object's is not
        applied on top (that would square it). ``dbfs=True`` ignores both
        the object and the factor, being referenced to digital full scale.
    :param response_plot: Whether to show or save the filter response plot
        (:class:`ResponsePlot`). Plotting bypasses the design cache.
    :return: A tuple containing (SPL_array, Frequencies_list) or (SPL_array, Frequencies_list, signals).
        When *nominal=True*, the frequency list contains ``List[str]`` labels instead of floats.
    :rtype: Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[str]],
        Tuple[np.ndarray, List[float], List[np.ndarray]],
        Tuple[np.ndarray, List[str], List[np.ndarray]]]
    """
    fs = resolve_fs(x, fs)
    if response_plot.show or response_plot.file:
        # Plotting has side effects: bypass the cache.
        filter_bank = OctaveFilterBank(
            fs=fs,
            fraction=fraction,
            order=order,
            limits=limits,
            design=design,
            calibration=calibration,
            response_plot=response_plot,
        )
    else:
        # The bank is immutable in non-stateful mode: reuse the design.
        # Pass limits through as-is (tuple for hashability); the bank
        # constructor is the single place that validates them and owns
        # the default when None. The option bundles are frozen dataclasses,
        # so they are hashable and compare by value: equal options hit the
        # same cache entry.
        limits_key = tuple(map(float, limits)) if limits is not None else None
        filter_bank = _cached_filter_bank(fs, fraction, order, limits_key, design, calibration)

    return filter_bank.filter(x, sigbands=sigbands, mode=mode, detrend=detrend, nominal=nominal)  # type: ignore[call-overload,no-any-return]
