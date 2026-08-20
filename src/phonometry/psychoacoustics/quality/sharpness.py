#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Sharpness per DIN 45692:2009-08.

Sharpness rates the high-frequency emphasis of a sound in acum, computed
as the g(z)-weighted first moment of the specific loudness pattern
(DIN 45692 Equation 1) over the 24 Bark critical-band scale, normalized so
the reference sound - critical-band-wide narrowband noise at 1 kHz
(920 Hz to 1080 Hz) at 60 dB overall level - yields exactly 1.00 acum
(clause 6). The informative Annex B variants of Aures and von Bismarck are
provided as alternative methods with the literal factor 0.11 printed in
Formulas (B.1)/(B.2); with it the reference sound lands near, not exactly
at, 1.00 acum (~0.96 Aures / ~1.02 von Bismarck through this front-end).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..loudness.zwicker import ZwickerLoudness, loudness_zwicker

if TYPE_CHECKING:
    from ...io._signal import Signal

_DZ = 0.1  # Bark step of the ISO 532-1 specific-loudness pattern
_Z = np.arange(1, 241) * _DZ  # Bark bin centers (0.1 .. 24.0), reference convention


def _g_din(z: np.ndarray) -> np.ndarray:
    """DIN 45692 Eq. (1) weighting: 1 up to 15.8 Bark, rising beyond."""
    g = np.ones_like(z)
    high = z > 15.8
    g[high] = 0.15 * np.exp(0.42 * (z[high] - 15.8)) + 0.85
    return g


def _g_bismarck(z: np.ndarray) -> np.ndarray:
    """DIN 45692 Annex B (informative), von Bismarck weighting."""
    g = np.ones_like(z)
    high = z > 15.0
    g[high] = 0.2 * np.exp(0.308 * (z[high] - 15.0)) + 0.8
    return g


def _g_aures(z: np.ndarray, total_n: float) -> np.ndarray:
    """DIN 45692 Annex B (informative), Aures loudness-dependent weighting."""
    g: np.ndarray = (
        0.078 * np.exp(0.171 * z) / z * (total_n / np.log(total_n * 0.05 + 1.0))
    )
    return g


def _moment(specific: np.ndarray, method: str) -> float:
    """Raw g(z)-weighted first moment of the specific-loudness pattern.

    Returns the un-normalized moment ``sum(N' g(z) z) dz / sum(N') dz`` for
    the requested weighting; the per-variant normalization constant that
    anchors the DIN 45692 reference sound to 1.00 acum is applied by the
    caller (``_k_din`` / ``_k_bismarck`` / ``_k_aures``).
    """
    n_total = float(np.sum(specific) * _DZ)
    if n_total <= 0:
        return 0.0
    if method == "din":
        num = float(np.sum(specific * _g_din(_Z) * _Z) * _DZ)
        return num / n_total
    if method == "bismarck":
        num = float(np.sum(specific * _g_bismarck(_Z) * _Z) * _DZ)
        return num / n_total
    if method == "aures":
        num = float(np.sum(specific * _g_aures(_Z, n_total) * _Z) * _DZ)
        return num / n_total
    raise ValueError("method must be 'din', 'aures' or 'bismarck'")


def reference_sound(
    fs: int = 48000, seconds: float = 2.0, seed: int = 45692
) -> np.ndarray:
    """The DIN 45692 clause 6 standard test signal.

    Critical-band-wide narrowband noise: 920 Hz to 1080 Hz, 60 dB overall
    level (deterministic realization; the clause fixes only the band and
    the level).
    """
    from scipy import signal as sp_signal

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(fs * seconds))
    sos = sp_signal.butter(8, [920.0, 1080.0], btype="band", fs=fs, output="sos")
    nb = sp_signal.sosfilt(sos, white)
    return np.asarray(nb / np.sqrt(np.mean(nb**2)) * 2e-5 * 10 ** (60.0 / 20))


def _reference_specific() -> np.ndarray:
    """Specific loudness of the clause 6 reference sound (signal path)."""
    return loudness_zwicker(reference_sound(), 48000, stationary=True).specific


_K_DIN: float | None = None

#: Literal factor of the informative Annex B variants, Formulas (B.1)/(B.2):
#: S_A = 0,11 * moment_A and S_B = 0,11 * moment_B. Unlike the normative
#: clause 5.2 k (derived so the reference sound is exactly 1.00 acum), the
#: Annex prints a fixed 0.11 -- with it the clause 6 reference sound lands
#: NEAR, not exactly at, 1 acum through this front-end (measured ~0.96 acum
#: for Aures and ~1.02 for von Bismarck). Deriving per-variant constants
#: instead would shift every Aures result ~4 % against other implementations
#: of the published formulas, so the literal is used.
_K_ANNEX_B = 0.11


def _k_din() -> float:
    """DIN 45692 normalization constant (0.105 <= k < 0.115, clause 5.2)."""
    global _K_DIN
    if _K_DIN is None:
        _K_DIN = 1.0 / _moment(_reference_specific(), "din")
        if not 0.105 <= _K_DIN < 0.115:  # pragma: no cover - sanity guard
            raise RuntimeError(f"k={_K_DIN} outside the DIN 45692 range")
    return _K_DIN


def sharpness_din_from_specific(
    specific: np.ndarray, method: Literal["din", "aures", "bismarck"] = "din"
) -> float:
    """
    Sharpness in acum from a specific-loudness pattern (DIN 45692 Eq. 1).

    :param specific: Specific loudness N'(z) at 0.1 Bark steps (240 values),
        e.g. ``ZwickerLoudness.specific``.
    :param method: ``'din'`` (default, Eq. 1), or the informative Annex B
        variants ``'aures'`` / ``'bismarck'``.
    :return: Sharpness in acum (0.0 for silence).
    """
    specific = np.asarray(specific, dtype=np.float64)
    if specific.shape != (240,):
        raise ValueError("specific must be the 240-bin ISO 532-1 pattern.")
    if method == "din":
        return _k_din() * _moment(specific, "din")
    if method in ("bismarck", "aures"):
        # Annex B (B.1)/(B.2): the published literal 0.11, NOT a derived
        # per-variant anchor -- see the _K_ANNEX_B note.
        return _K_ANNEX_B * _moment(specific, method)
    raise ValueError("method must be 'din', 'aures' or 'bismarck'")


def sharpness_din(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    field: Literal["free", "diffuse"] = "free",
    method: Literal["din", "aures", "bismarck"] = "din",
    calibration_factor: float | None = None,
) -> float:
    """
    Sharpness of a signal per DIN 45692:2009 (stationary analysis).

    The specific loudness comes from the ISO 532-1 Zwicker stationary
    method (the DIN 45631 basis named by DIN 45692); the sharpness is its
    g(z)-weighted first moment, normalized so the reference sound of
    clause 6 (critical-band-wide noise, 1 kHz, 60 dB) gives 1.00 acum.
    Verification tolerance per clause 6: 5 % or 0.05 acum against the
    Table A.2/A.3 target values. A small, systematic negative bias grows
    with centre frequency for narrowband stimuli (about -0.03 acum at
    2.5 kHz to -0.15 acum at 8.5 kHz, from the specific-loudness upper-slope
    handling near the 24 Bark edge); it stays within the 5 % / 0.05 acum
    tolerance and is a known DIN 45692 implementation property.

    :param x: Input signal (1D), in Pa after ``calibration_factor``.
        Accepts a :class:`phonometry.io.Signal`; the rate and the factor are
        resolved by :func:`~phonometry.psychoacoustics.loudness_zwicker`, which
        this is a weighted moment of, so both arrive here on exactly the terms
        they have there.
    :param fs: Sample rate in Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit
        value that disagrees with it raises.
    :param field: ``'free'`` (default) or ``'diffuse'``.
    :param method: ``'din'`` (default), ``'aures'`` or ``'bismarck'``.
    :param calibration_factor: Digital-units-to-Pa factor. An explicit
        value wins over the one a Signal carries.
    :return: Sharpness in acum.
    """
    zw: ZwickerLoudness = loudness_zwicker(
        x, fs, field=field, stationary=True, calibration_factor=calibration_factor
    )
    return sharpness_din_from_specific(zw.specific, method=method)
