# PyOctaveBand: Technical Benchmark Report

Generated on: 2026-01-07 08:31:49

**Environment:** fs=48000Hz, Python optimized with Numba.

## 1. Spectral Isolation & Numerical Precision
Evaluates how well the filter isolates a 1kHz tone and its numerical accuracy relative to the theoretical RMS (-3.01 dBFS).
| Filter Type | Peak (dBFS) | Precision Error (dB) | Atten. -1 Oct | Atten. +1 Oct | Atten. -2 Oct |
|:---|:---:|:---:|:---:|:---:|:---:|
| butter | -3.0121 | 1.85e-03 | 43.0 dB | 32.4 dB | 49.8 dB |
| cheby1 | -3.0182 | 7.87e-03 | 42.8 dB | 41.1 dB | 49.5 dB |
| cheby2 | -3.0151 | 4.80e-03 | 45.5 dB | 53.0 dB | 52.2 dB |
| ellip | -3.0269 | 1.66e-02 | 42.9 dB | 48.0 dB | 49.6 dB |
| bessel | -3.4357 | 4.25e-01 | 44.6 dB | 33.7 dB | 51.4 dB |

## 2. Signal Theory & Quality Metrics
Validation of passband ripple (central 80%), phase linearity (GD Std), and total reconstruction flatness.
| Filter Type | Passband Ripple | GD Std Dev (ms) | Flatness Error | Recommended For |
|:---|:---:|:---:|:---:|:---:|
| butter | 0.2462 dB | 2698.713 ms | 4.55 dB | Standard Audio |
| cheby1 | 0.1000 dB | 3394.606 ms | 4.79 dB | High Selectivity |
| cheby2 | 28.7270 dB | 4854.467 ms | 60.22 dB | Out-of-band Rejection |
| ellip | 0.1000 dB | 4600.809 ms | 4.78 dB | Out-of-band Rejection |
| bessel | 5.8771 dB | 1122.052 ms | 10.54 dB | Transient Analysis |

## 3. Stability, Latency & Speed
IR Tail Energy < 1e-6 indicates high numerical stability. Latency is measured at the 1kHz band peak.
| Filter Type | IR Tail Energy | Latency (ms) | Status | Exec Time (ms) |
|:---|:---:|:---:|:---:|:---:|
| butter | 1.29e-09 | 1.85 | ✅ Stable | 34.64 |
| cheby1 | 2.04e-07 | 2.31 | ✅ Stable | 34.52 |
| cheby2 | 2.12e-07 | 3.23 | ✅ Stable | 35.48 |
| ellip | 4.95e-07 | 1.85 | ✅ Stable | 35.88 |
| bessel | 2.34e-13 | 1.85 | ✅ Stable | 45.61 |

## 4. Multichannel Performance (Vectorization)
Measures the average execution time for a 1-second signal through a 1/3 Octave Filter Bank.
| Channels | Total Time (ms) | Time per Channel (ms) | Speedup Factor |
|:---|:---:|:---:|:---:|
| 1 | 42.66 | 42.66 | 1.00x |
| 2 | 66.34 | 33.17 | 1.29x |
| 4 | 125.25 | 31.31 | 1.36x |
| 8 | 234.03 | 29.25 | 1.46x |
| 16 | 447.93 | 28.00 | 1.52x |

## 5. Architecture Summary
- **Butterworth:** Maximally flat passband. Standard for acoustic measurement.
- **Chebyshev I:** High selectivity, but introduces ripples in the passband.
- **Chebyshev II:** Flat passband with ripples in the stopband (excellent for isolation).
- **Elliptic:** Minimum transition width at the cost of ripple in both regions.
- **Bessel:** Linear phase response. Lowest latency and best transient preservation.