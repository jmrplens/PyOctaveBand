---
title: "electroacoustics.frequency_response"
description: "Frequency-response and coherence estimators (Bendat & Piersol)."
sidebar:
  label: "frequency_response"
---

Frequency-response and coherence estimators (Bendat & Piersol).

Two-channel (input/output) system identification from measured signals, using
the Welch-averaged cross- and auto-spectral densities. Following Bendat &
Piersol, *Random Data: Analysis and Measurement Procedures* (4th ed., 2010):

* the **H1** estimator $H_1 = G_{xy} / G_{xx}$ (unbiased when the
  noise is on the output),
* the **H2** estimator $H_2 = G_{yy} / G_{yx}$ (unbiased when the
  noise is on the input),
* the **ordinary coherence**
  $\gamma^2 = \lvert G_{xy} \rvert^2 / (G_{xx} \cdot G_{yy}) \in [0, 1]$, the fraction of the output power linearly explained by the
  input.

For a noiseless linear time-invariant path both estimators recover the true
transfer function and the coherence is unity; additive output noise biases H2
but not H1 and pulls the coherence down to
$\mathrm{SNR} / (1 + \mathrm{SNR})$, which is the analytic oracle used
to verify the implementation.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## coherence

```python
coherence(
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    fs: float,
    *,
    nperseg: int | None = None,
    overlap: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Ordinary coherence $\gamma^2(f)$ between `x` and `y`
(Bendat & Piersol).

$\gamma^2 = \lvert G_{xy} \rvert^2 / (G_{xx} G_{yy}) \in [0, 1]$: unity for a noiseless linear path and
$\mathrm{SNR}/(1+\mathrm{SNR})$ with additive output noise.
Averaging over several segments is required for a meaningful estimate
(a single segment gives $\gamma^2 \equiv 1$).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | First signal, 1-D. |
| `y` | Second signal, 1-D, same length as `x`. |
| `fs` | Sample rate, in Hz. |
| `nperseg` | Welch segment length; `None` picks a default. |
| `overlap` | Segment overlap fraction (default 0.5). |

**Returns:** `(frequencies, gamma_squared)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## FrequencyResponseResult

```python
FrequencyResponseResult(
    frequencies: NDArray[np.float64],
    response: NDArray[np.complex128],
    magnitude_db: NDArray[np.float64],
    phase: NDArray[np.float64],
    coherence: NDArray[np.float64],
    estimator: str,
)
```

Estimated frequency response of an input/output path (Bendat &
Piersol).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency axis, in Hz. |
| `response` | Complex frequency-response estimate `H(f)`. |
| `magnitude_db` | Magnitude $20 \lg \lvert H \rvert$, in dB. |
| `phase` | Phase of `H`, in radians (unwrapped). |
| `coherence` | Ordinary coherence $\gamma^2(f) \in [0, 1]$. |
| `estimator` | Estimator used (`'H1'` or `'H2'`). |

### FrequencyResponseResult.plot()

```python
FrequencyResponseResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the Bode magnitude/phase and the coherence.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## transfer_function

```python
transfer_function(
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    fs: float,
    *,
    estimator: Literal['H1', 'H2'] = 'H1',
    nperseg: int | None = None,
    overlap: float = 0.5,
) -> FrequencyResponseResult
```

Estimate the frequency response from input `x` to output `y`.

$H_1 = G_{xy} / G_{xx}$ (the default; unbiased for output
noise) or $H_2 = G_{yy} / G_{yx}$ (unbiased for input noise),
from Welch-averaged Hann segments. The ordinary coherence
$\gamma^2 = \lvert G_{xy} \rvert^2 / (G_{xx} G_{yy})$ is
returned alongside as a data-quality indicator.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input (reference) signal, 1-D. |
| `y` | Output (response) signal, 1-D, same length as `x`. |
| `fs` | Sample rate, in Hz. |
| `estimator` | `'H1'` (default) or `'H2'`. |
| `nperseg` | Welch segment length; `None` picks a length targeting about 4 Hz resolution (as in [`intensity`](/phonometry/reference/api/power/intensity/)). |
| `overlap` | Segment overlap fraction (default 0.5). |

**Returns:** A [`FrequencyResponseResult`](/phonometry/reference/api/electroacoustics/frequency-response/#frequencyresponseresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |
