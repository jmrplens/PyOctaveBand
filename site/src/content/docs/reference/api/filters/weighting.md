---
title: "filters.weighting"
description: "Weighting filters (A, B, C, D, G, AU, Z), time weighting utilities and the Linkwitz-Riley crossover."
sidebar:
  label: "weighting"
---

Weighting filters (A, B, C, D, G, AU, Z), time weighting utilities and
the Linkwitz-Riley crossover.

A/C/Z per IEC 61672-1:2013; G (infrasound) per ISO 7196:1995.

B is the historical weighting of ANSI S1.4-1983 (Appendix C): the C curve
with one extra zero at the origin and one extra real pole at
$f_5 = 158.48932$ Hz. It was dropped from the sound-level-meter
standards
when IEC 61672-1 replaced IEC 60651 (first edition 2002) and is provided for
historical data and older national codes only.

AU per IEC 61012:1990: the A weighting cascaded with the U low-pass
(six poles, Table 2: a double real pole at -12 200 Hz and complex pairs at
-7 850 +/- j8 800 Hz and -2 900 +/- j12 150 Hz) for measuring audible sound
in the presence of ultrasound. It is flat relative to A up to 10 kHz and
cuts steeply above (U alone, Table 1: -2.8 dB at 12.5 kHz; -61.8 dB at
40 kHz). The Table 2 poles reproduce every Table 1 nominal value within
0.05 dB.

D per the withdrawn IEC 537:1976 (aircraft-noise weighting): implemented
from the widely published rational transfer function

$$
\frac{k s \left( s^2 + 6532 s + 4.0975 \times 10^7 \right)} {(s + 1776.3)(s + 7288.5) \left( s^2 + 21514 s + 3.8836 \times 10^8 \right)}
$$

with `k` renormalized to exactly 0 dB at
1 kHz. The standard itself is withdrawn and unavailable, so the constants
are corroborated against two independent implementations: SQAT
(`sound_level_meter/Gen_weighting_filters.m`: identical zeros and poles;
note its display-only `freqResp` line prints 1773.6 where its pole list,
and every other source, has 1776.3) and librosa (`librosa.D_weighting`,
an independent frequency-domain closed form; agreement within 0.002 dB
from 10 Hz to 20 kHz). The response also reproduces the tabulated IEC 537
curve republished in the NASA Handbook of Aircraft Noise Metrics
(NASA CR-3406, 1981, Table SLD-I) within 0.1 dB at every one-third-octave
frequency from 50 Hz to 10 kHz except 1600 Hz (0.15 dB) and 2500 Hz
(0.28 dB), where that table appears to round a different source curve.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## linkwitz_riley

```python
linkwitz_riley(
    x: list[float] | np.ndarray,
    fs: int,
    freq: float,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]
```

Linkwitz-Riley crossover filter (Butterworth squared).
Splits signal into low and high bands with flat sum response.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal. |
| `fs` | Sample rate. |
| `freq` | Crossover frequency. |
| `order` | Total order (must be even, typically 2 or 4). |

**Returns:** (low_pass_signal, high_pass_signal)

## time_weighting

```python
time_weighting(
    x: list[float] | np.ndarray,
    fs: int,
    mode: str = 'fast',
    initial_state: str | float | np.ndarray | None = None,
) -> np.ndarray
```

Apply time weighting to a signal (Exponential averaging).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (raw pressure/voltage). The function squares it internally. |
| `fs` | Sample rate. |
| `mode` | 'fast' (125ms), 'slow' (1000ms), 'impulse' (35ms rise, 1500ms fall). |
| `initial_state` | Previous mean-square output state `y[-1]`. Use None/'zero' for zero initialization (default), 'first' to initialize from the first input energy, or a scalar/array broadcastable to the input shape without the time axis. |

**Returns:** Time-weighted squared signal (sound pressure level envelope).

## TimeWeighting

```python
TimeWeighting(fs: int, mode: str = 'fast')
```

Stateful time weighting for block processing.

Wraps [`time_weighting`](/phonometry/reference/api/filters/weighting/#time_weighting) carrying the exponential integrator state
across blocks, so concatenated block outputs equal a single continuous call.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate in Hz. |
| `mode` | 'fast' (125 ms), 'slow' (1000 ms) or 'impulse' (35 ms / 1.5 s). |

### TimeWeighting.process()

```python
TimeWeighting.process(x: list[float] | np.ndarray) -> np.ndarray
```

Apply time weighting to a block, continuing from the previous block.

### TimeWeighting.reset()

```python
TimeWeighting.reset() -> None
```

Forget the carried state (the next block starts from rest).

## weighting_filter

```python
weighting_filter(
    x: list[float] | np.ndarray,
    fs: int,
    curve: str = 'A',
    high_accuracy: bool = True,
) -> np.ndarray
```

Apply a frequency weighting to a signal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal. |
| `fs` | Sample rate. |
| `curve` | 'A', 'C' (IEC 61672-1), 'B' (ANSI S1.4-1983, historical), 'D' (withdrawn IEC 537 aircraft-noise weighting), 'G' (ISO 7196 infrasound), 'AU' (IEC 61012) or 'Z' (bypass). |
| `high_accuracy` | Use internal oversampling for IEC 61672-1 class 1 accuracy at high frequencies (default True). |

**Returns:** Weighted signal.

## WeightingFilter

```python
WeightingFilter(
    fs: int,
    curve: str = 'A',
    stateful: bool = False,
    steady_ic: bool = False,
    high_accuracy: bool | None = None,
)
```

Class-based frequency weighting filter (A, B, C, D, G, AU, Z).
Allows pre-calculating and reusing filter coefficients.

Initialize the weighting filter.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate in Hz. |
| `curve` | 'A', 'C' (IEC 61672-1), 'B' (ANSI S1.4-1983, historical: removed from the IEC sound-level-meter standards), 'D' (withdrawn IEC 537 aircraft-noise weighting), 'G' (ISO 7196 infrasound), 'AU' (IEC 61012, audible sound in the presence of ultrasound) or 'Z'. |
| `stateful` | If True, the weighting filter is stateful. Useful for block processing. |
| `steady_ic` | If True, calculate steady state initial conditions for filter. |
| `high_accuracy` | If True, design and run the filter at an internal oversampled rate (target >= 144 kHz) so the response stays within IEC 61672-1 class 1 tolerances up to 16 kHz, provided 16 kHz is well clear of the input Nyquist frequency (fs >= 40 kHz). At 48 kHz this oversamples x3, keeping the deviation from the design goal to -0.44 dB at the 16 kHz nominal frequency and -0.86 dB at the 20 kHz one. Oversampling cannot rescue the top of the band at low sample rates, because the resampling stages it adds around the sections carry an anti-alias transition band centred on the input Nyquist frequency: above roughly 0.9 x fs/2 the response rolls off steeply whatever the design rate. What the roll-off costs is per curve, since each is graded against its own design goal: at fs = 32 kHz the 15 848.9 Hz nominal point falls 16.2 dB below the A goal but 15.3 dB below the C one (class 1 allows -16.0 dB there, class 2 has no lower limit), so the verified class at 32 kHz is 2 for A and still 1 for C. At fs = 16 kHz the 7 943.3 Hz point falls 12.0 dB below the A goal and 13.7 dB below the C one (class 1 allows -2.5 dB, class 2 -5.0 dB), so neither curve verifies to any class there. The plain bilinear design holds class 1 for fs >= 40 kHz (-2.8 dB at the 12.5 kHz nominal frequency at 48 kHz, -3.5 dB at 44.1 kHz, inside the +2.0/-5.0 class 1 limits), degrades to class 2 between 22.05 and 32 kHz and meets no class at fs \<= 20 kHz. Defaults to True except in stateful mode (the internal FIR resampling is incompatible with block processing). |

### WeightingFilter.filter()

```python
WeightingFilter.filter(x: list[float] | np.ndarray) -> np.ndarray
```

Apply the weighting filter to a signal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D [channels, samples]). |

**Returns:** Weighted signal.
