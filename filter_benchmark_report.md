# Filter Architecture Benchmark Report

This report compares the performance and characteristics of the available filter types.

## 1. Spectral Isolation (at 1kHz)
| Filter Type | Peak SPL (dB) | Atten. -1 Oct (dB) | Atten. +1 Oct (dB) | Atten. -2 Oct (dB) | Atten. +2 Oct (dB) |
|---|---|---|---|---|---|
| butter | 90.96 | 40.0 | 32.3 | 46.8 | 57.7 |
| cheby1 | 90.96 | 39.8 | 40.2 | 46.5 | 57.2 |
| cheby2 | 90.96 | 42.5 | 50.4 | 49.2 | 61.6 |
| ellip | 90.95 | 39.9 | 45.1 | 46.6 | 57.1 |
| bessel | 90.54 | 41.6 | 33.6 | 48.4 | 60.1 |

## 2. Stability and Performance
| Filter Type | Max IR Tail Energy | Stability Status | Avg. Execution Time (s) |
|---|---|---|---|
| butter | 1.29e-09 | ✅ Stable | 0.0353 |
| cheby1 | 2.04e-07 | ✅ Stable | 0.0348 |
| cheby2 | 2.12e-07 | ✅ Stable | 0.0355 |
| ellip | 4.95e-07 | ✅ Stable | 0.0359 |
| bessel | 4.21e-15 | ✅ Stable | 0.0451 |

## 3. Analysis Summary
- **Butterworth:** Best compromise, maximally flat passband.
- **Chebyshev I:** Steeper roll-off than Butterworth but with passband ripple.
- **Chebyshev II:** Flat passband, ripple in the stopband.
- **Elliptic:** Steepest transition but ripples in both passband and stopband.
- **Bessel:** Best phase response and minimal ringing (group delay), but slowest roll-off.