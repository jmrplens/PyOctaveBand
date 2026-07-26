---
title: "psychoacoustics.fluctuation_strength"
description: "Fluctuation strength after Fastl & Zwicker / Osses et al."
sidebar:
  label: "fluctuation_strength"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Fluctuation strength after Fastl & Zwicker / Osses et al.

Fluctuation strength `F` (unit: vacil) rates the slow loudness fluctuations of
a sound. It has a band-pass dependence on modulation frequency peaking near
4 Hz, and its fixed point is defined (Fastl & Zwicker, *Psychoacoustics: Facts
and Models*, Chapter 10) as **1 vacil** for a 1 kHz tone at 60 dB, 100 %
amplitude-modulated at 4 Hz. There is no ISO standard for fluctuation strength.

Two routes are provided:

* [`fluctuation_strength_am_noise`](/phonometry/reference/api/psychoacoustics/fluctuation-strength/#fluctuation_strength_am_noise) -- the closed form (Fastl & Zwicker
  Eq. 10.2) for a sinusoidally amplitude-modulated broadband noise, exact in the
  modulation factor, level and modulation frequency.
* [`fluctuation_strength`](/phonometry/reference/api/psychoacoustics/fluctuation-strength/#fluctuation_strength) -- the signal model of Osses, García &
  Kohlrausch (2016), which estimates `F` from an arbitrary calibrated pressure
  signal. It sums specific contributions over 47 auditory filters,
  `F = C_FS·Σ (m*_i)^p_m·|k_{i-2}·k_i|^p_k·g(z_i)^p_g` (Osses 2016 Eq. 1),
  where `m*` is a generalised modulation depth, `k` a cross-covariance
  between neighbouring bands and `g(z)` a frequency weighting.

The closed form and the 1-vacil calibration are exact. The signal model has no
normative oracle; it is implemented clean-room from the Osses 2016 paper and
cross-checked against the literature fluctuation-strength values of Fastl &
Zwicker (Osses 2016 Table 1) and the open SQAT reference implementation. Like
the ECMA-418-2 roughness, the model reproduces the reference trends and the
1-vacil anchor within a documented tolerance rather than to the last digit.

**Front-end deviations from Osses 2016** (besides the re-fitted `H(fmod)`
corners documented at `_HP_LO`): the paper analyses 2 s frames with 90 %
overlap and 50 ms raised-cosine gating and screens components against the
absolute hearing threshold; this implementation uses 50 % overlap, a Hann
analysis window and relative floors (component magnitude > max/10^4,
excitation level > max - 60 dB), with `k = 1` at the filter-bank edges.
Measured consequences: a steady 1 kHz tone returns ~0.09 vacil instead of ~0
(analysis-window envelope leakage through the 2 Hz high-pass) and steady
broadband noise returns ~0.4 vacil (partly physical level fluctuation of the
noise itself).

ECMA-418-2:2025 (4th ed.) introduced a normative hearing-model fluctuation
strength (its Clause 9, block size 65536); this module implements the
Osses/Fastl & Zwicker model, not that method.

## fluctuation_strength

```python
fluctuation_strength(
    signal_in: NDArray[np.float64],
    fs: float,
) -> FluctuationStrengthResult
```

Fluctuation strength of a calibrated signal (Osses 2016 model).

Estimates `F` in vacil from an arbitrary calibrated sound-pressure signal
by the model of Osses, García & Kohlrausch (2016): an outer/middle-ear
transmission, a 47-band excitation-pattern filter bank, the generalised
modulation depth `m*` from the band-pass-filtered envelope, the
cross-covariance `k` between neighbouring bands and the weighted sum of
Eq. (1). The signal is analysed in 2 s frames; the overall value is the
median across frames. The Osses 2016 model is defined for the **free
field** only (its `a0` transmission is the free-field characteristic);
there is no diffuse-field variant, so this function takes no `field`
argument.

:::note
No ISO standard defines fluctuation strength and no numeric oracle
anchors this signal model. It is implemented clean-room from the Osses
2016 paper and reproduces the literature reference values (Osses 2016
Table 1) and the 1-vacil calibration point within a documented
tolerance -- the reference method itself only agrees qualitatively for
FM tones. For an exact figure on AM broadband noise use
[`fluctuation_strength_am_noise`](/phonometry/reference/api/psychoacoustics/fluctuation-strength/#fluctuation_strength_am_noise).

**Calibration.** The constant `C_FS` (Eq. 1) is not the paper's literal
0.2490 -- that was fitted to the paper's own 4096-tap FIR front-end. Here
it is derived once from the 1-vacil reference stimulus (1 kHz, 60 dB,
`m = 1`, 4 Hz) run through this implementation's front-end, so the
reference returns **exactly 1.00 vacil by construction** (`C_FS ≈ 0.28`;
see `_c_fs`).

**Achieved tolerance** (against Osses 2016 Table 1 literature values):
the reference tone is 1.00 vacil exactly; the AM-tone 70 dB sweep
(`fmod = 1, 2, 4, 8, 16, 32` Hz) has Pearson correlation ≈ 0.98 with
the literature, peaks at 4 Hz and stays within a factor ≈ 2.1 at every
point; the AM-tone carrier sweep at `fmod = 4 Hz` reproduces the
Fastl & Zwicker Fig. 10.5 trend (low-mid plateau, roll-off at 8 kHz);
the AM broadband-noise 60 dB sweep shows the correct band-pass shape
(maximum at 4 Hz, monotone tails) but overshoots the absolute
pass-band level by up to ~3x (the excitation front-end spreads the
modulated energy across bands). FM-tone accuracy is *not* pursued --
the reference method itself overestimates it above 4 Hz.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal_in` | Calibrated sound-pressure signal (1-D), in Pa. |
| `fs` | Sample rate, in Hz (resampled to the model rate if needed). |

**Returns:** A [`FluctuationStrengthResult`](/phonometry/reference/api/psychoacoustics/fluctuation-strength/#fluctuationstrengthresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the signal is invalid or `fs` is not positive. |

## fluctuation_strength_am_noise

```python
fluctuation_strength_am_noise(
    level_db: float,
    modulation_factor: float,
    mod_frequency: float,
) -> float
```

Fluctuation strength of AM broadband noise (Fastl & Zwicker Eq. 10.2).

`F = 5.8·(1.25·m − 0.25)·[0.05·(L/dB) − 1] / [(fmod/5Hz)² + (4Hz/fmod) +
1.5]` vacil, the closed form for a sinusoidally amplitude-modulated
broadband noise of level `L`, modulation factor `m` and modulation
frequency `fmod`. Exact; the result is clamped at `0` (the formula goes
negative below ~20 dB or `m < 0.2` where the sensation vanishes).

**Parameters**

| Name | Description |
| :--- | :--- |
| `level_db` | Broadband-noise level `L`, in dB. |
| `modulation_factor` | Modulation factor `m` (0..1). |
| `mod_frequency` | Modulation frequency `fmod`, in Hz. |

**Returns:** Fluctuation strength, in vacil (>= 0).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `modulation_factor` is outside [0, 1] or `mod_frequency` is not positive/finite, or the level is not finite. |

## FluctuationStrengthResult

```python
FluctuationStrengthResult(
    fluctuation_strength: float,
    specific: NDArray[np.float64],
    bark_axis: NDArray[np.float64],
    time_dependent: NDArray[np.float64],
)
```

Fluctuation strength of a signal (Osses 2016 model).

**Attributes**

| Name | Description |
| :--- | :--- |
| `fluctuation_strength` | Overall fluctuation strength `F`, in vacil (the median over the analysis frames). |
| `specific` | Specific fluctuation strength `f(z)` per auditory filter, in vacil/Bark (averaged over frames), shape `(47,)`. |
| `bark_axis` | Centre critical-band rate `z_i` of each filter, in Bark, shape `(47,)`. |
| `time_dependent` | Fluctuation strength per analysis frame, in vacil. |

### FluctuationStrengthResult.plot()

```python
FluctuationStrengthResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the specific fluctuation strength against critical-band rate.
