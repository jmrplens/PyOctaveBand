#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Calibration utilities for mapping digital signals to physical SPL levels.
"""

from typing import List, Union

import numpy as np


def calculate_sensitivity(
    ref_signal: Union[List[float], np.ndarray], 
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
    rms_ref = np.std(ref_signal)
    if rms_ref == 0:
        raise ValueError("Reference signal is silent, cannot calibrate.")
        
    # target_spl = 20 * log10( (rms_ref * factor) / ref_pressure )
    factor = (ref_pressure * 10**(target_spl / 20)) / rms_ref
    return float(factor)
