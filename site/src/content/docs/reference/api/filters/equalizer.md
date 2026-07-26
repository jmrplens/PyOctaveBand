---
title: "metrology.equalizer"
description: "Parametric equalizer biquads per the RBJ Audio EQ Cookbook."
sidebar:
  label: "equalizer"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Parametric equalizer biquads per the RBJ Audio EQ Cookbook.

Second-order (biquad) IIR sections designed from the closed-form recipes of
Robert Bristow-Johnson's *Audio EQ Cookbook*, the de-facto reference for
parametric equalization (published as a W3C Working Group Note:
https://www.w3.org/TR/audio-eq-cookbook/). Every filter type of the cookbook
is available:

* `peaking` - the bell of a parametric EQ: gain `G` dB exactly at `f0`,
  0 dB exactly at DC and Nyquist.
* `lowshelf` / `highshelf` - shelving filters: gain `G` dB exactly at
  DC (low shelf) or Nyquist (high shelf), 0 dB at the opposite end, with the
  transition centred on `f0` (the midpoint-gain frequency).
* `lowpass` / `highpass` - second-order Butterworth-style sections with
  a resonance set by `Q` (`Q = 1/sqrt(2)` is the Butterworth alignment;
  the magnitude at `f0` is exactly `Q`).
* `bandpass` - constant 0 dB peak gain at `f0`.
* `bandpass_skirt` - the cookbook's constant-skirt-gain variant (peak
  gain `Q`).
* `notch` - a null exactly at `f0`, 0 dB at DC and Nyquist.
* `allpass` - unit magnitude everywhere; only the phase turns (360
  degrees across the band, steepest at `f0`).

Each section is parameterized exactly as the cookbook defines: sample rate
`fs`, centre/corner frequency `f0`, gain `gain_db` (peaking and
shelves only) and one of

* `q` - the quality factor (default `1/sqrt(2)`),
* `bw` - the bandwidth in octaves, mapped through the cookbook's
  digital-domain relation `alpha = sin(w0) * sinh(ln(2)/2 * BW *
  w0/sin(w0))` (the `w0/sin(w0)` factor compensates the bilinear
  frequency warping), or
* `slope` - the shelf-slope parameter `S` (shelves only; `S = 1` is
  the steepest slope that stays monotonic).

For the peaking filter the bandwidth is measured between the midpoint-gain
(`G/2` dB) frequencies; for band-pass and notch, between the -3 dB
frequencies - both as the cookbook states.

The design is exact, not approximate: the cookbook's formulas are the
bilinear transform of second-order analog prototypes with the frequency
prewarped so the analog `f0` lands exactly on the digital `f0`.
Closed-form consequences pinned by the test suite: the peaking gain at
`f0` is exactly `G` dB, shelves reach exactly `G` dB at their shelved
end and exactly 0 dB at the other, the all-pass magnitude is exactly 1 at
every frequency, and the half-gain bandwidth honours the cookbook's Q
definition.

Reference: Bristow-Johnson, R. *Audio EQ Cookbook*. Republished as a W3C
Working Group Note (ed. R. Toy), 8 June 2021.
https://www.w3.org/TR/audio-eq-cookbook/

## EQResponseResult

```python
EQResponseResult(
    frequencies: NDArray[np.float64],
    magnitude_db: NDArray[np.float64],
    phase_rad: NDArray[np.float64],
    section_magnitude_db: NDArray[np.float64],
    sos: NDArray[np.float64],
    fs: float,
    sections: tuple[EQSection, ...],
)
```

Frequency response of a parametric-EQ cascade (RBJ Audio EQ Cookbook).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Evaluation frequencies, in Hz (log-spaced). |
| `magnitude_db` | Cascade magnitude, in dB. |
| `phase_rad` | Cascade phase, unwrapped, in radians. |
| `section_magnitude_db` | Per-section magnitude, in dB, shape `(n_sections, n_frequencies)` - the cascade magnitude is their sum. |
| `sos` | The designed cascade, shape `(n_sections, 6)` (scipy SOS layout, normalized so every section's `a0` is 1). |
| `fs` | Sample rate, in Hz. |
| `sections` | The [`EQSection`](/phonometry/reference/api/filters/equalizer/#eqsection) specifications, in cascade order. |

### EQResponseResult.plot()

```python
EQResponseResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    show_sections: bool = True,
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the cascade magnitude and phase response.

With `ax` given, only the magnitude panel is drawn on it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes for the magnitude panel, or `None` for a fresh two-panel (magnitude + phase) figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `show_sections` | Overlay each section's individual response under the cascade curve (default `True`). |

**Returns:** The magnitude axes (`ax` given) or the array of two axes.

## EQSection

```python
EQSection(
    filter_type: EQFilterType,
    f0: float,
    gain_db: float = 0.0,
    q: float | None = None,
    bw: float | None = None,
    slope: float | None = None,
)
```

One biquad of the RBJ Audio EQ Cookbook.

**Attributes**

| Name | Description |
| :--- | :--- |
| `filter_type` | `'peaking'`, `'lowshelf'`, `'highshelf'`, `'lowpass'`, `'highpass'`, `'bandpass'` (constant 0 dB peak gain), `'bandpass_skirt'` (constant skirt gain, peak gain `Q`), `'notch'` or `'allpass'`. |
| `f0` | Centre/corner frequency, in Hz (must sit below Nyquist). |
| `gain_db` | Gain `G` in dB - peaking and shelving types only. |
| `q` | Quality factor. Exactly one of `q`, `bw` and `slope` may be given; with none, `q = 1/sqrt(2)` (Butterworth alignment). |
| `bw` | Bandwidth in octaves (peaking: between the midpoint-gain frequencies; band-pass/notch: between the -3 dB frequencies). |
| `slope` | Shelf-slope parameter `S` (shelves only; `S = 1` is the steepest monotonic slope, and `S` must stay below the gain-dependent bound `(A + 1/A)/(A + 1/A - 2)` with `A = 10^(gain_db/40)`, beyond which the cookbook's alpha turns complex; at 0 dB gain `A = 1`, the bound is unbounded and every positive slope is admissible). |

## parametric_eq

```python
parametric_eq(
    x: list[float] | np.ndarray,
    fs: float,
    sections: EQSection | Sequence[EQSection],
) -> np.ndarray
```

Apply a parametric-EQ cascade to a signal (RBJ Audio EQ Cookbook).

Convenience wrapper around [`ParametricEQ`](/phonometry/reference/api/filters/equalizer/#parametriceq) for one-shot use.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D `[channels, samples]`). |
| `fs` | Sample rate in Hz. |
| `sections` | One [`EQSection`](/phonometry/reference/api/filters/equalizer/#eqsection) or a sequence of them. |

**Returns:** Equalized signal.

## ParametricEQ

```python
ParametricEQ(
    fs: float,
    sections: EQSection | Sequence[EQSection],
    stateful: bool = False,
    steady_ic: bool = False,
)
```

Cascade of RBJ Audio EQ Cookbook biquads.

Designs one second-order section per [`EQSection`](/phonometry/reference/api/filters/equalizer/#eqsection) and runs them in
series as a numerically robust SOS cascade, following the house style of
[`WeightingFilter`](/phonometry/reference/api/filters/parametric-filters/#weightingfilter)
(reusable coefficients, optional stateful block processing).

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate in Hz. |
| `sections` | One [`EQSection`](/phonometry/reference/api/filters/equalizer/#eqsection) or a sequence of them; the cascade applies them in order. |
| `stateful` | If True, `filter` carries the filter state across calls (block processing). |
| `steady_ic` | If True, initialize the state at steady state. |

### ParametricEQ.filter()

```python
ParametricEQ.filter(x: list[float] | np.ndarray) -> np.ndarray
```

Apply the EQ cascade to a signal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D or 2D `[channels, samples]`). |

**Returns:** Equalized signal.

### ParametricEQ.response()

```python
ParametricEQ.response(
    n_points: int = 2048,
    *,
    f_min: float = 10.0,
    f_max: float | None = None,
) -> EQResponseResult
```

Evaluate the cascade frequency response on a log-spaced grid.

**Parameters**

| Name | Description |
| :--- | :--- |
| `n_points` | Number of evaluation frequencies. |
| `f_min` | Lowest frequency, in Hz. |
| `f_max` | Highest frequency, in Hz (default: Nyquist). |

**Returns:** An [`EQResponseResult`](/phonometry/reference/api/filters/equalizer/#eqresponseresult) (carries the SOS cascade and plots the magnitude/phase response).
