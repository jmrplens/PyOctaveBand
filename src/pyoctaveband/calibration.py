#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Calibration utilities for mapping digital signals to physical SPL levels.
"""

from __future__ import annotations

from typing import List

import numpy as np


def calculate_sensitivity(
    ref_signal: List[float] | np.ndarray, 
    target_spl: float = 94.0, 
    ref_pressure: float = 2e-5
) -> float:
    """
    Calculate the calibration factor (multiplier) to convert digital units 
    to Pascals based on a reference recording (e.g., 1kHz @ 94dB).
    
    :param ref_signal: Recording of the calibration tone.
    :param target_spl: The known SPL level of the calibrator (default 94 dB).
    :param ref_pressure: Reference pressure (default 20 microPascals).
    :return: Calibration factor (sensitivity multiplier).
    """
    rms_ref = np.sqrt(np.mean(np.array(ref_signal)**2))
    if rms_ref == 0:
        raise ValueError("Reference signal is silent, cannot calibrate.")
        
    factor = (ref_pressure * 10**(target_spl / 20)) / rms_ref
    return float(factor)
