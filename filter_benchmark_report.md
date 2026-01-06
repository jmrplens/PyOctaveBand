# Filter Architecture Benchmark Report

This report compares the performance and characteristics of the available filter types.

## 1. Spectral Isolation (at 1kHz)
| Filter Type | Peak SPL (dB) | Atten. -1 Oct (dB) | Atten. +1 Oct (dB) | Atten. -2 Oct (dB) | Atten. +2 Oct (dB) |
|---|---|---|---|---|---|
| butter | 90.96 | 270.0 | 26.7 | 282.3 | 54.8 |
| cheby1 | 90.30 | 269.4 | 41.0 | 282.1 | 54.8 |
| cheby2 | 90.82 | 270.1 | 48.5 | 283.3 | 58.9 |
| ellip | 90.69 | 270.0 | 43.9 | 282.6 | 55.2 |
| bessel | 89.53 | 269.2 | 27.4 | 282.0 | 56.1 |

## 2. Stability and Performance
| Filter Type | Max IR Tail Energy | Stability Status | Avg. Execution Time (s) |
|---|---|---|---|
| butter | 3.62e-09 | ✅ Stable | 0.0279 |
| cheby1 | 5.69e-07 | ✅ Stable | 0.0284 |
| cheby2 | 2.11e-07 | ✅ Stable | 0.0289 |
| ellip | 1.12e-06 | ⚠️ Ringing | 0.0300 |
| bessel | 3.95e-14 | ✅ Stable | 0.0383 |

## 3. Analysis Summary
- **Butterworth:** Best compromise, maximally flat passband.
- **Chebyshev I:** Steeper roll-off than Butterworth but with passband ripple.
- **Chebyshev II:** Flat passband, ripple in the stopband.
- **Elliptic:** Steepest transition but ripples in both passband and stopband.
- **Bessel:** Best phase response and minimal ringing (group delay), but slowest roll-off.