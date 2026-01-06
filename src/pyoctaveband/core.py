#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Core processing logic and FilterBank class for pyoctaveband.
"""

from __future__ import annotations

from typing import List, Tuple, cast

import numpy as np
from scipy import signal

from .filter_design import _design_sos_filter
from .frequencies import _genfreqs
from .utils import _downsamplingfactor, _resample_to_length, _typesignal


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
        limits: List[float] | None = None,
        filter_type: str = "butter",
        ripple: float = 0.1,
        attenuation: float = 60.0,
        show: bool = False,
        plot_file: str | None = None,
        calibration_factor: float = 1.0,
        dbfs: bool = False,
    ) -> None:
        """
        Initialize the Octave Filter Bank.

        :param fs: Sample rate in Hz.
        :param fraction: Bandwidth fraction (e.g., 1 for octave, 3 for 1/3 octave).
        :param order: Filter order.
        :param limits: Frequency limits [f_min, f_max].
        :param filter_type: Type of filter ('butter', 'cheby1', 'cheby2', 'ellip', 'bessel').
        :param ripple: Passband ripple in dB.
        :param attenuation: Stopband attenuation in dB.
        :param show: If True, show the filter response plot.
        :param plot_file: Path to save the filter response plot.
        :param calibration_factor: Calibration factor for SPL calculation.
        :param dbfs: If True, calculate SPL in dBFS.
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
            
        valid_filters = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
        if filter_type not in valid_filters:
            raise ValueError(f"Invalid filter_type. Must be one of {valid_filters}")

        self.fs = fs
        self.fraction = fraction
        self.order = order
        self.limits = limits
        self.filter_type = filter_type
        self.ripple = ripple
        self.attenuation = attenuation
        self.calibration_factor = calibration_factor
        self.dbfs = dbfs

        # Generate frequencies
        self.freq, self.freq_d, self.freq_u = _genfreqs(limits, fraction, fs)
        self.num_bands = len(self.freq)

        # Calculate factors and design SOS
        self.factor = _downsamplingfactor(self.freq_u, fs)
        self.sos = _design_sos_filter(
            self.freq, self.freq_d, self.freq_u, fs, order, self.factor, 
            filter_type, ripple, attenuation, show, plot_file
        )

    def filter(
        self, 
        x: List[float] | np.ndarray, 
        sigbands: bool = False,
        mode: str = "rms",
        detrend: bool = True
    ) -> Tuple[np.ndarray, List[float]] | Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Apply the pre-designed filter bank to a signal.

        :param x: Input signal (1D array or 2D array [channels, samples]).
        :param sigbands: If True, also return the signal in the time domain divided into bands.
        :param mode: 'rms' for energy-based level, 'peak' for peak-holding level.
        :param detrend: If True, remove DC offset from signal before filtering (Default: True).
        :return: A tuple containing (SPL_array, Frequencies_list) or (SPL_array, Frequencies_list, signals).
        """
        
        # Convert input to numpy array
        x_proc = _typesignal(x)

        # Handle DC offset removal
        if detrend:
            # Axis -1 handles both 1D and 2D arrays correctly
            x_proc = signal.detrend(x_proc, axis=-1, type='constant')

        # Handle multichannel detection
        is_multichannel = x_proc.ndim > 1
        if not is_multichannel:
            x_proc = x_proc[np.newaxis, :]  # Standardize to 2D

        num_channels = x_proc.shape[0]

        # Process signal across all bands and channels
        spl, xb = self._process_bands(x_proc, num_channels, sigbands, mode=mode)

        # Format output based on input dimensionality
        if not is_multichannel:
            spl = spl[0]
            if sigbands and xb is not None:
                xb = [band[0] for band in xb]

        if sigbands and xb is not None:
            return spl, self.freq, xb
        else:
            return spl, self.freq

    def _process_bands(
        self,
        x_proc: np.ndarray,
        num_channels: int,
        sigbands: bool,
        mode: str = "rms"
    ) -> Tuple[np.ndarray, List[np.ndarray] | None]:
        """
        Process signal through each frequency band.

        :param x_proc: Standardized 2D input signal [channels, samples].
        :param num_channels: Number of channels.
        :param sigbands: If True, return filtered bands.
        :param mode: 'rms' or 'peak'.
        :return: A tuple containing (SPL_array, Optional_List_of_filtered_signals).
        """
        spl = np.zeros([num_channels, self.num_bands])
        xb: List[np.ndarray] | None = [np.array([]) for _ in range(self.num_bands)] if sigbands else None

        for idx in range(self.num_bands):
            for ch in range(num_channels):
                # Core DSP logic extracted to reduce complexity
                filtered_signal = self._filter_and_resample(x_proc[ch], idx)

                # Sound Level Calculation
                spl[ch, idx] = self._calculate_level(filtered_signal, mode)

                if sigbands and xb is not None:
                    # Restore original length
                    y_resampled = _resample_to_length(filtered_signal, int(self.factor[idx]), x_proc.shape[1])
                    if ch == 0:
                        xb[idx] = np.zeros([num_channels, x_proc.shape[1]])
                    xb[idx][ch] = y_resampled
        return spl, xb

    def _filter_and_resample(self, x_ch: np.ndarray, idx: int) -> np.ndarray:
        """Resample and filter a single channel for a specific band."""
        if self.factor[idx] > 1:
            sd = signal.resample_poly(x_ch, 1, self.factor[idx])
        else:
            sd = x_ch
        
        return cast(np.ndarray, signal.sosfilt(self.sos[idx], sd))

    def _calculate_level(self, y: np.ndarray, mode: str) -> float:
        """Calculate the level (RMS or Peak) in dB."""
        if mode.lower() == "rms":
            val_linear = np.std(y)
        elif mode.lower() == "peak":
            val_linear = np.max(np.abs(y))
        else:
            raise ValueError("Invalid mode. Use 'rms' or 'peak'.")

        if self.dbfs:
            # dBFS: 0 dB is RMS = 1.0 or Peak = 1.0
            return float(20 * np.log10(np.max([val_linear, np.finfo(float).eps])))
        
        # Physical SPL: apply sensitivity and use 20uPa reference
        pressure_pa = val_linear * self.calibration_factor
        return float(20 * np.log10(np.max([pressure_pa, np.finfo(float).eps]) / 2e-5))