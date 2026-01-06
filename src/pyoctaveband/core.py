#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Core processing logic and FilterBank class for pyoctaveband.
"""

from typing import List, Optional, Tuple, Union

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
        limits: Optional[List[float]] = None,
        filter_type: str = "butter",
        ripple: float = 0.1,
        attenuation: float = 60.0,
        show: bool = False,
        plot_file: Optional[str] = None,
        calibration_factor: float = 1.0,
        dbfs: bool = False,
    ):
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
        x: Union[List[float], np.ndarray], 
        sigbands: bool = False
    ) -> Union[Tuple[np.ndarray, List[float]], Tuple[np.ndarray, List[float], List[np.ndarray]]]:
        """Apply the pre-designed filter bank to a signal."""
        
        # Convert input to numpy array
        x_proc = _typesignal(x)

        # Handle multichannel detection
        is_multichannel = x_proc.ndim > 1
        if not is_multichannel:
            x_proc = x_proc[np.newaxis, :]  # Standardize to 2D

        num_channels = x_proc.shape[0]

        # Process signal across all bands and channels
        spl, xb = self._process_bands(x_proc, num_channels, sigbands)

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
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """Process signal through each frequency band."""
        spl = np.zeros([num_channels, self.num_bands])
        xb: Optional[List[np.ndarray]] = [np.array([]) for _ in range(self.num_bands)] if sigbands else None

        for idx in range(self.num_bands):
            for ch in range(num_channels):
                # Resample signal for this specific band to improve accuracy
                # Use resample_poly for better stability than FFT-based resample
                if self.factor[idx] > 1:
                    sd = signal.resample_poly(x_proc[ch], 1, self.factor[idx])
                else:
                    sd = x_proc[ch]
                
                y = signal.sosfilt(self.sos[idx], sd)

                # Sound Level Calculation
                rms = np.std(y)
                if self.dbfs:
                    # dBFS: 0 dB is RMS = 1.0 (Standard digital scale)
                    # Note: A full-scale sine (peak 1.0) will result in -3.01 dBFS RMS
                    val = 20 * np.log10(np.max([rms, np.finfo(float).eps]))
                else:
                    # Physical SPL: apply sensitivity and use 20uPa reference
                    pressure_pa = rms * self.calibration_factor
                    val = 20 * np.log10(np.max([pressure_pa, np.finfo(float).eps]) / 2e-5)
                
                spl[ch, idx] = val

                if sigbands and xb is not None:
                    # Restore original length
                    y_resampled = _resample_to_length(y, int(self.factor[idx]), x_proc.shape[1])
                    if ch == 0:
                        xb[idx] = np.zeros([num_channels, x_proc.shape[1]])
                    xb[idx][ch] = y_resampled
        return spl, xb
