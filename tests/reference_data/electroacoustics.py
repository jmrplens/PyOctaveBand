#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Electroacoustic measurements: harmonic distortion and coherence.

Closed-form anchors for the two quantities that say how linear a chain is.
The distortion values follow from a harmonic amplitude set by definition
(THD referred to the fundamental and to the total, and the second-order
ratio); the clipped-sine values follow from the Fourier series of the
waveform itself; the coherence value follows from a flat signal-to-noise
ratio. None of them needs a published table because each is exact.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Electroacoustic distortion (IEC 60268-3:2013) and frequency response
# (Bendat & Piersol, Random Data 4e). All quantities are exact analytic
# oracles evaluated on synthetic signals with known harmonic / intermodulation
# amplitudes, or on a known LTI path.
# ---------------------------------------------------------------------------
# A 1 kHz fundamental (a1 = 1) with harmonics a2 = 0.1, a3 = 0.05, a4 = 0.02.
#   THD_F = sqrt(a2^2 + a3^2 + a4^2) / a1            = 0.1135782
#   THD_R = sqrt(a2^2 + a3^2 + a4^2) / sqrt(sum a^2) = 0.1128526
#   d2    = a2 / sqrt(sum a^2)                       = 0.0993612
DISTORTION_HARMONICS = (1.0, 0.1, 0.05, 0.02)  # a1..a4
DISTORTION_THD_F = 0.11357816691600547
DISTORTION_THD_R = 0.11285260010027609
DISTORTION_D2 = 0.09936117403949127

# Clipped-sine THD oracle: a unit sine symmetrically clipped at 0.7, sampled
# at 48 samples per period, has these odd-harmonic Fourier amplitudes and
# THD_F over n <= 10 (independent single-period Fourier series of the sampled
# waveform). The continuous-time fundamental is b1 = (2/pi)(arcsin 0.7 +
# 0.7 sqrt(0.51)) = 0.8118795956258127; the sampled value differs by the
# 6.5e-4 aliasing of the clipped wave's high harmonics, so the sampled value
# is the one pinned here.
CLIPPED_SINE_THD_F = 0.13794482640558078
CLIPPED_SINE_B1 = 0.8124127489373637
CLIPPED_SINE_B3 = 0.1087038092372312
CLIPPED_SINE_B5 = 0.0205013791213361
CLIPPED_SINE_B7 = 0.0165310026995253
CLIPPED_SINE_B9 = 0.0070120099075438

# Ordinary coherence of a signal-plus-independent-noise output with a flat
# (frequency-independent) SNR: gamma^2 = SNR / (1 + SNR). At SNR = 10 -> 0.90909.
COHERENCE_SNR = 10.0
COHERENCE_EXPECTED = COHERENCE_SNR / (1.0 + COHERENCE_SNR)
