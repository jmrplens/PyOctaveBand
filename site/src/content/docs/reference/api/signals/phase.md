---
title: "signals.phase"
description: "Phase utilities: minimum phase, group delay and excess phase."
sidebar:
  label: "phase"
---

Phase utilities: minimum phase, group delay and excess phase.

For a causal, stable, minimum-phase system the log-magnitude and the phase
of the frequency response form a Hilbert-transform pair (Bendat & Piersol,
*Random Data*, 4th ed., 2010, Sec. 13.1.4; Oppenheim & Schafer,
*Discrete-Time Signal Processing*, Ch. 12): the phase is fully determined
by $\lvert H(f) \rvert$. This module computes that **minimum
phase** with the real cepstrum -- fold the inverse transform of
$\ln \lvert H \rvert$ onto positive
quefrencies and transform back -- and derives the standard decomposition
of a measured response,

$$
H(f) = H_{\mathrm{min}}(f) \, H_{\mathrm{ap}}(f) \quad \text{with} \quad \phi_{\mathrm{excess}}(f) = \phi(f) - \phi_{\mathrm{min}}(f)
$$

whose all-pass **excess phase** collects pure latency and any genuine
non-minimum-phase behaviour (reflections, non-invertible zeros): the part
of the phase no stable causal equalizer can remove. The **group delay**
$\tau_\mathrm{g} = -(1/2\pi) \, d\phi/df$ is estimated from the unwrapped
phase.

Sampling precautions (documented, and the reason for the `oversample`
padding): the estimate operates on a *uniformly sampled* one-sided response
(DC to Nyquist inclusive, the layout of `numpy.fft.rfft` and of
[`phonometry.impulse_response`](/phonometry/reference/api/rooms/impulse-response/#impulse_response) spectra). The real cepstrum of the
sampled log-magnitude is time-aliased when the grid is coarse relative to
how sharp the response is, so the magnitude is resampled onto an
`oversample` times denser grid by exact trigonometric (zero-padded
even-sequence) interpolation before the logarithm and the phase is read
back on the original bins. Magnitude zeros (a response that vanishes at
DC or Nyquist, e.g. a band-pass) are not minimum-phase-representable:
bins below a relative floor are clipped, and the reconstruction is only
accurate away from them. On a strictly minimum-phase response (all poles
and zeros inside the unit circle) sampled on an adequate grid the
reconstruction matches the true phase to better than `1e-12` rad -- the
tolerance the biquad oracle pins in the tests; near-circle zeros on
coarse grids degrade it and are what `oversample` mitigates.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## excess_phase

```python
excess_phase(
    response: NDArray[np.complex128] | list[float],
    *,
    oversample: int = 8,
) -> NDArray[np.float64]
```

Excess phase: measured phase minus the minimum phase of
$\lvert H \rvert$.

$\phi_{\mathrm{excess}} = \operatorname{unwrap}(\arg H) - \phi_{\mathrm{min}}$ is the phase of the all-pass
factor in $H = H_{\mathrm{min}} H_{\mathrm{ap}}$: zero for a
minimum-phase system,
$-2 \pi f t_0$ for a pure latency $t_0$, and it
additionally bends
wherever the response has non-minimum-phase zeros. Equalizing a
response down to its excess phase is the realizability limit of any
stable causal inverse filter.

**Parameters**

| Name | Description |
| :--- | :--- |
| `response` | One-sided complex response, uniformly sampled from DC to Nyquist inclusive (`rfft` layout). |
| `oversample` | Cepstral anti-aliasing factor for the minimum-phase reconstruction (default 8). |

**Returns:** Excess phase per bin, in radians (continuous, 0 at DC).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## group_delay

```python
group_delay(
    response: NDArray[np.complex128] | list[float],
    fs: float,
) -> NDArray[np.float64]
```

Group delay $\tau_\mathrm{g}(f) = -(1/2\pi) \, d\phi/df$ of a response.

The phase is unwrapped and differentiated with second-order central
differences (one-sided at the grid ends). The estimate is exact for a
linear phase and accurate to $O(df^2)$ otherwise; the unwrapping
requires the response to be sampled densely enough that the phase
advances less than $\pi$ per bin -- a pure delay of `D` samples
needs fewer than `response.size` of them, i.e. the underlying
impulse response must fit the record the grid implies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `response` | One-sided complex response, uniformly sampled from DC to Nyquist inclusive (`rfft` layout). |
| `fs` | Sample rate the response grid corresponds to, in Hz. |

**Returns:** Group delay per bin, in seconds.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## minimum_phase

```python
minimum_phase(
    response: NDArray[np.complex128] | NDArray[np.float64] | list[float],
    *,
    oversample: int = 8,
) -> NDArray[np.complex128]
```

Minimum-phase response with the magnitude of `response`.

Computes the phase that the Hilbert relation between log-magnitude and
phase assigns to $\lvert H(f) \rvert$ (Bendat & Piersol
Sec. 13.1.4) via the real cepstrum: the inverse transform of
$\ln \lvert H \rvert$ is folded onto positive
quefrencies (doubling them, keeping the ends; the folding core is
shared with [`phonometry.signals.cepstrum`](/phonometry/reference/api/signals/cepstrum/)) and transformed
back, so `exp` of the result is the unique stable, causal, causally
invertible response with that magnitude. The input phase, if any, is
ignored: passing a plain magnitude array works.

**Parameters**

| Name | Description |
| :--- | :--- |
| `response` | One-sided response (or magnitude), uniformly sampled from DC to Nyquist inclusive (`rfft` layout). |
| `oversample` | Cepstral anti-aliasing factor: the magnitude is resampled onto a grid this many times denser (exact trigonometric interpolation) before the log and the cepstrum (default 8). Raise it if the magnitude has very sharp features -- near-circle zeros, deep notches -- relative to the grid. |

**Returns:** Complex minimum-phase response on the same bins (same magnitude, reconstructed phase).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

:::note
Magnitude zeros cannot be represented by a minimum-phase system;
they are floored at 1e-15 of the peak and the phase near them is
not reliable (see the module docstring).
:::

## phase_decomposition

```python
phase_decomposition(
    response: NDArray[np.complex128] | list[float],
    fs: float,
    *,
    oversample: int = 8,
) -> PhaseDecompositionResult
```

Decompose a response into its minimum-phase and all-pass parts.

Bundles [`minimum_phase`](/phonometry/reference/api/signals/phase/#minimum_phase), [`excess_phase`](/phonometry/reference/api/signals/phase/#excess_phase) and
[`group_delay`](/phonometry/reference/api/signals/phase/#group_delay) on one frequency axis: the minimum phase carries
everything an equalizer can invert, the excess phase is the residual
all-pass (latency plus non-minimum-phase zeros), and the two group
delays quantify both in seconds.

**Parameters**

| Name | Description |
| :--- | :--- |
| `response` | One-sided complex response, uniformly sampled from DC to Nyquist inclusive (`rfft` layout) -- e.g. `numpy.fft.rfft(ir)` of a measured impulse response. |
| `fs` | Sample rate of the underlying record, in Hz. |
| `oversample` | Cepstral anti-aliasing factor (default 8). |

**Returns:** A [`PhaseDecompositionResult`](/phonometry/reference/api/signals/phase/#phasedecompositionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## PhaseDecompositionResult

```python
PhaseDecompositionResult(
    frequencies: NDArray[np.float64],
    magnitude: NDArray[np.float64],
    phase: NDArray[np.float64],
    minimum_phase: NDArray[np.float64],
    excess_phase: NDArray[np.float64],
    group_delay: NDArray[np.float64],
    excess_group_delay: NDArray[np.float64],
    minimum_phase_response: NDArray[np.complex128],
    fs: float,
)
```

Minimum-phase / all-pass decomposition of a frequency response.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency axis, in Hz. |
| `magnitude` | $\lvert H(f) \rvert$ (shared by the measured and minimum-phase responses). |
| `phase` | Measured phase, unwrapped and referenced to DC, rad. |
| `minimum_phase` | Phase reconstructed from $\lvert H \rvert$ alone, rad. |
| `excess_phase` | `phase - minimum_phase`: the all-pass part, rad. |
| `group_delay` | Group delay of the measured response, s. |
| `excess_group_delay` | Group delay of the all-pass part alone, s (constant $t_0$ for a pure latency). |
| `minimum_phase_response` | Complex minimum-phase response $\lvert H \rvert \exp(j \, \text{minimum\_phase})$. |
| `fs` | Sample rate of the underlying record, in Hz. |

### PhaseDecompositionResult.plot()

```python
PhaseDecompositionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the magnitude, the phase decomposition and the group delay.

Three stacked panels: $\lvert H \rvert$ in dB, the
measured / minimum /
excess phases, and the total and excess group delays. With `ax`
given, only the phase panel is drawn on it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
