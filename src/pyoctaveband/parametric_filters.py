#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Weighting filters (A, C, Z) and time weighting utilities for audio analysis.
Implementation according to IEC 61672-1:2013.
"""

from __future__ import annotations

from typing import List, Tuple, cast

import numpy as np
from scipy import signal

from .utils import _typesignal


def weighting_filter(x: List[float] | np.ndarray, fs: int, curve: str = "A") -> np.ndarray:
    """
    Apply frequency weighting (A or C) to a signal.
    
    :param x: Input signal.
    :param fs: Sample rate.
    :param curve: 'A', 'C' or 'Z' (Z is zero weighting/bypass).
    :return: Weighted signal.
    """
    x_proc = _typesignal(x)
    curve = curve.upper()
    
    if curve == "Z":
        return x_proc
        
    if curve not in ["A", "C"]:
        raise ValueError("Weighting curve must be 'A', 'C' or 'Z'")

    # Analog ZPK for A and C weighting
    # f1, f2, f3, f4 constants as per IEC 61672-1
    f1 = 20.598997
    f4 = 12194.217
    
    if curve == "A":
        f2 = 107.65265
        f3 = 737.86223
        # Zeros at 0 Hz
        z = np.array([0, 0, 0, 0])
        # Poles
        p = np.array([-2*np.pi*f1, -2*np.pi*f1, -2*np.pi*f4, -2*np.pi*f4, 
                      -2*np.pi*f2, -2*np.pi*f3])
        # k chosen to give 0 dB at 1000 Hz
        # Reference gain at 1000Hz for A weighting: 10^(A1000/20) = 1.0 (0 dB)
        k = 3.5174303309e13
        
        # Recalculate k to ensure 0dB at 1kHz
        w = 2 * np.pi * 1000
        h = k * np.prod(1j * w - z) / np.prod(1j * w - p)
        k = k / np.abs(h)
        
    else: # C weighting
        z = np.array([0, 0])
        p = np.array([-2*np.pi*f1, -2*np.pi*f1, -2*np.pi*f4, -2*np.pi*f4])
        k = 5.91797e8
        
        # Recalculate k to ensure 0dB at 1kHz
        w = 2 * np.pi * 1000
        h = k * np.prod(1j * w - z) / np.prod(1j * w - p)
        k = k / np.abs(h)

    zd, pd, kd = signal.bilinear_zpk(z, p, k, fs)
    sos = signal.zpk2sos(zd, pd, kd)
    
    return cast(np.ndarray, signal.sosfilt(sos, x_proc))


def time_weighting(x: List[float] | np.ndarray, fs: int, mode: str = "fast") -> np.ndarray:
    """
    Apply time weighting to a signal (Exponential averaging).
    
    :param x: Input signal (usually the squared signal x^2).
    :param fs: Sample rate.
    :param mode: 'fast' (125ms), 'slow' (1000ms), 'impulse' (35ms rise).
    :return: Time-weighted squared signal (sound pressure level envelope).
    """
    x_proc = _typesignal(x)
    
    modes = {
        "fast": 0.125,
        "slow": 1.0,
        "impulse": 0.035
    }
    
    if mode.lower() not in modes:
        raise ValueError(f"Invalid time weighting mode. Use {list(modes.keys())}")
        
    tau = modes[mode.lower()]
    
    # RC filter implementation: y[n] = y[n-1] + (x[n] - y[n-1]) * dt / tau
    # This is a first order IIR filter
    alpha = 1 - np.exp(-1 / (fs * tau))
    b = [alpha]
    a = [1, -(1 - alpha)]
    
    # We apply the weighting to the squared signal to get the Mean Square value
    return cast(np.ndarray, signal.lfilter(b, a, x_proc**2))


def linkwitz_riley(
    x: List[float] | np.ndarray, 
    fs: int, 
    freq: float, 
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linkwitz-Riley crossover filter (Butterworth squared).
    Splits signal into low and high bands with flat sum response.
    
    :param x: Input signal.
    :param fs: Sample rate.
    :param freq: Crossover frequency.
    :param order: Total order (must be even, typically 2 or 4).
    :return: (low_pass_signal, high_pass_signal)
    """
    x_proc = _typesignal(x)
    if order % 2 != 0:
        raise ValueError("Linkwitz-Riley order must be even (typically 2 or 4).")
    
    # A Linkwitz-Riley filter of order N is two Butterworth filters of order N/2 in series
    half_order = order // 2
    wn = freq / (fs / 2)
    
    sos_lp = signal.butter(half_order, wn, btype='low', output='sos')
    sos_hp = signal.butter(half_order, wn, btype='high', output='sos')
    
    # Pass twice
    lp = signal.sosfilt(sos_lp, x_proc)
    lp = signal.sosfilt(sos_lp, lp)
    
    hp = signal.sosfilt(sos_hp, x_proc)
    hp = signal.sosfilt(sos_hp, hp)
    
    return lp, hp
