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

# ---------------------------------------------------------------------------
# ITU-R BS.468-4 Table 1 (printed p. 2), the psophometric weighting for
# audio-frequency noise in sound broadcasting: frequency in Hz, nominal
# response in dB re 1 kHz, and the permissible difference between the nominal
# curve and the response of the measuring equipment. Clause 1 makes the
# passive network of Fig. 1a the primitive and this table "the values of this
# response at various frequencies", so these 21 rows are an oracle for the
# library's network, not its definition - which is why they live here and not
# in ``src/``. IEC 60268-1:1985 Appendix A Table AI prints the same 21
# response values (its note: "in accordance with CCIR Recommendation 468-2").
#
# The tolerance column is reproduced as printed, including the 0 at 6 300 Hz.
# That row is unsatisfiable as written - no physical network and no digital
# filter has exactly the nominal gain at a frequency, and the +/-1 % component
# tolerance the same figure blesses already moves it by up to 0.11 dB - so a
# test must substitute its own budget there and say so. AES17-2015 Table 1
# prints +/-0,01 dB for the same row, which is the only sign in these
# documents that anyone noticed.
ITU_R_468_TABLE1: tuple[tuple[float, float, float], ...] = (
    (31.5, -29.9, 2.0),
    (63.0, -23.9, 1.4),
    (100.0, -19.8, 1.0),
    (200.0, -13.8, 0.85),
    (400.0, -7.8, 0.7),
    (800.0, -1.9, 0.55),
    (1000.0, 0.0, 0.5),
    (2000.0, 5.6, 0.5),
    (3150.0, 9.0, 0.5),
    (4000.0, 10.5, 0.5),
    (5000.0, 11.7, 0.5),
    (6300.0, 12.2, 0.0),
    (7100.0, 12.0, 0.2),
    (8000.0, 11.4, 0.4),
    (9000.0, 10.1, 0.6),
    (10000.0, 8.1, 0.8),
    (12500.0, 0.0, 1.2),
    (14000.0, -5.3, 1.4),
    (16000.0, -11.7, 1.6),
    (20000.0, -22.2, 2.0),
    (31500.0, -42.7, 2.8),
)

#: Half the 0,1 dB quantum Table 1 is printed to, and therefore the largest
#: difference rounding the nominal curve to that table can produce.
ITU_R_468_TABLE1_ROUNDING_DB = 0.05

#: Bound on |Fig. 1a network - printed Table 1 cell|, over all 21 rows. It is
#: the quantum above plus a hair, and the hair is one row: at 100 Hz the
#: network reads -19.850221 dB, 0.000221 dB past the rounding boundary, so
#: the printed -19,8 sits on the far side of a tie that component values given
#: to four significant figures cannot resolve. Every other row is inside
#: 0.0384 dB and the rms over all 21 is 0.0264 dB, which is the signature of a
#: rounding residual rather than a modelling error. A test must therefore not
#: demand that the network reproduce the table by exact rounding.
ITU_R_468_NETWORK_VS_TABLE1_DB = 0.0503

#: Gain shift AES17-2015 5.2.7 adds to this curve to make the "CCIR-RMS"
#: filter, which puts unity gain at 2 kHz: "The standard weighting filter
#: shall conform to ITU-R BS.468-4, with an additional gain of -5,63 dB."
ITU_R_468_AES17_OFFSET_DB = -5.63

#: Four rows of AES17-2015 Table 1 (printed p. 10), in dB, as a cross-check on
#: the curve from a second published document.
ITU_R_468_AES17_ROWS: tuple[tuple[float, float], ...] = (
    (63.0, -29.5),
    (3150.0, 3.4),
    (8000.0, 5.8),
    (12500.0, -5.6),
)

#: Tolerance for comparing the Fig. 1a network against those AES17 cells. The
#: comparison crosses two independent roundings to 0,1 dB: AES17 derived its
#: table from BS.468-4's *rounded* Table 1 rather than from the curve (5 of
#: the 21 rows differ from ``round(network - 5.63)``, at 100, 200, 400, 3 150
#: and 8 000 Hz), so the residual is up to 0.0802 dB, at 100 Hz. The AES17
#: offset itself was taken from the curve, not the table: the network reads
#: +5.6292 dB at 2 kHz, which rounds to the printed 5,63 and not to Table 1's
#: +5,6.
ITU_R_468_AES17_TOL_DB = 0.09
