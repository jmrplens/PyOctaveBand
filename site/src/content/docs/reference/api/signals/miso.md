---
title: "signals.miso"
description: "Multiple and partial coherence of a multiple-input/single-output system."
sidebar:
  label: "miso"
---

Multiple and partial coherence of a multiple-input/single-output system.

When several partially correlated sources drive one response, the ordinary
coherence of each source with the output is misleading: a source that only
*correlates* with the true cause inherits a spurious coherence through it.
Bendat & Piersol, *Random Data: Analysis and Measurement Procedures*
(4th ed., 2010, Chapter 7), resolve this with the multiple-input/output
(MISO) coherence functions, computed here from the Welch cross-spectral
machinery of [`phonometry.signals.spectra`](/phonometry/reference/api/signals/spectra/):

* the **ordinary coherence**
  $\gamma^2_{iy} = \lvert G_{iy} \rvert^2 / (G_{ii} G_{yy})$
  (Eq. 7.109) of each input with the output, taken on its own;
* the **multiple coherence**
  $\gamma^2_{y:x} = G_{vv}/G_{yy} = 1 - G_{nn}/G_{yy}$ (Eq. 7.35):
  the fraction of the output autospectrum linearly explained by *all* inputs
  jointly, obtained from the input cross-spectral matrix $G_{xx}$ and
  the input-output vector $G_{iy}$ (matrix form
  $\gamma^2_{y:x} = G_{iy}^{\mathsf{H}} G_{xx}^{-1} G_{iy} / G_{yy}$,
  Eqs. 7.170-7.192). For additive uncorrelated output noise of per-band
  signal-to-noise ratio $\mathrm{SNR} = G_{vv}/G_{nn}$ this is exactly
  $\mathrm{SNR}/(1+\mathrm{SNR})$;
* the **partial coherence** $\gamma^2_{iy \cdot (i-1)!} = \lvert G_{iy \cdot (i-1)!} \rvert^2 / (G_{ii \cdot (i-1)!} G_{yy})$
  (Eq. 7.87): the coherence of input `i` with the output once the linear
  effect of the inputs *before it in the conditioning order* has been
  removed. The 4th-edition definition uses the *total* output
  $G_{yy}$ in the denominator (not the conditioned output), which
  makes the partial coherences of the ordered inputs add up to the multiple
  coherence, $\gamma^2_{y:x} = \sum \gamma^2_{iy \cdot (i-1)!}$
  (Eq. 7.116), and reduce exactly to the ordinary coherences when the
  inputs are mutually uncorrelated (Eq. 7.117);
* the **conditioned (residual) spectra** $G_{ij \cdot r!}$ computed by
  the Gaussian-elimination recursion $G_{ij \cdot r!} = G_{ij \cdot (r-1)!} - G_{rj \cdot (r-1)!} G_{ir \cdot (r-1)!} / G_{rr \cdot (r-1)!}$ (Eq. 7.94, base case Eq. 7.95), the Schur complement
  that removes the linear effect of the pivot input `r` from every
  remaining record;
* the **partial (cumulative) coherent output spectra**
  $G_{v_i} = \lvert L_{iy} \rvert^2 G_{ii \cdot (i-1)!} = \gamma^2_{iy \cdot (i-1)!} G_{yy}$ (Eq. 7.86): the share of output power
  the `i`-th ordered input contributes, so that
  $G_{yy} = \sum G_{v_i} + G_{nn}$ (Eqs. 7.88-7.89, 7.121). Comparing
  them band by band answers "which source dominates here?".

The random errors follow Bendat & Piersol Section 9.3: conditioning on the
$i-1$ preceding inputs costs $i-1$ degrees of freedom, so the
`i`-th ordered input carries $n_\mathrm{d} - (i-1)$ effective averages
(Eqs. 9.100/9.101) and the `q`-input multiple coherence carries
$n_\mathrm{d} - (q-1)$ (Eqs. 9.98/9.99).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## miso_coherence

```python
miso_coherence(
    inputs: Signal | Sequence[Signal | NDArray[np.float64] | list[float]] | NDArray[np.float64],
    output: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    order: Sequence[int] | None = None,
    window: str = 'hann',
    nperseg: int | None = None,
    overlap: float = 0.5,
    scaling: Literal['density', 'spectrum'] = 'density',
) -> MISOCoherenceResult
```

Multiple and partial coherence of a MISO system (Bendat & Piersol 7).

Estimates every auto- and cross-spectrum of the `q` inputs and the
output by the shared Welch core of
[`cross_spectral_density`](/phonometry/reference/api/signals/spectra/#cross_spectral_density) (Hann taper
and 50 % overlap by default, no detrending), then:

* reports the **ordinary coherence** of each input with the output
  (Eq. 7.109);
* forms the **multiple coherence** $\gamma^2_{y:x}$ (Eq. 7.35)
  from the residual output spectrum left by the Gaussian-elimination
  conditioning of Section 7.3;
* conditions the inputs in `order` to get the **partial coherences**
  $\gamma^2_{iy \cdot (i-1)!}$ (Eq. 7.87) and the **partial
  coherent output spectra** $G_{v_i}$ (Eq. 7.86), which decompose
  the output power source by source
  ($\sum_i G_{v_i} + G_{nn} = G_{yy}$).

The partial coherences and the coherent-output decomposition depend on
the conditioning order; the ordinary and multiple coherences do not.
Absent a physical basis, Bendat & Piersol (Section 7.2.4) recommend
ordering the inputs by descending ordinary coherence with the output.

**Parameters**

| Name | Description |
| :--- | :--- |
| `inputs` | The `q` input records (`q >= 2`), a sequence of equal-length 1-D arrays or a 2-D `(q, n)` array. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) whose channels are the records, or a sequence with Signals among its elements; their calibration is applied to the samples, and Signals recorded at different rates are refused rather than arbitrated. |
| `output` | The output record, 1-D, same length as the inputs. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples, so the output autospectrum and the coherent output spectra come out in Pa²/Hz, or Pa² for `scaling='spectrum'`. Every coherence is a ratio and does not move. |
| `fs` | Sample rate, in Hz. Required when both records are bare arrays; either may be a [`Signal`](/phonometry/reference/api/io/io/#signal) and supply it, and two Signals recorded at different rates are refused rather than arbitrated. |
| `order` | Conditioning order as input indices (default `0..q-1`). |
| `window` | Segment taper (default Hann). |
| `nperseg` | Welch segment length; `None` picks a default. |
| `overlap` | Segment overlap fraction in [0, 1) (default 0.5). |
| `scaling` | `'density'` (units²/Hz) or `'spectrum'` (units²). |

**Returns:** A [`MISOCoherenceResult`](/phonometry/reference/api/signals/miso/#misocoherenceresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## MISOCoherenceResult

```python
MISOCoherenceResult(
    frequencies: NDArray[np.float64],
    n_inputs: int,
    order: tuple[int, ...],
    ordinary_coherence: NDArray[np.float64],
    multiple_coherence: NDArray[np.float64],
    partial_coherence: NDArray[np.float64],
    coherent_output_spectra: NDArray[np.float64],
    output_psd: NDArray[np.float64],
    noise_psd: NDArray[np.float64],
    multiple_coherence_random_error: NDArray[np.float64],
    coherent_output_random_error: NDArray[np.float64],
    n_segments: int,
    n_averages: float,
    resolution_bandwidth: float,
    window: str,
    nperseg: int,
    overlap: float,
    scaling: str,
)
```

Multiple and partial coherence of a MISO system (B&P Chapter 7).

Every per-input array is indexed by the *original* input index (the order
in which the records were passed), so `ordinary_coherence[i]` and
`coherent_output_spectra[i]` refer to the same physical source; the
conditioning that produced the partial coherences is recorded in
`order`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-sided frequency axis, in Hz. |
| `n_inputs` | Number of inputs `q` (`q >= 2`). |
| `order` | Conditioning order actually applied, as original input indices; `partial_coherence[order[k]]` is conditioned on the inputs `order[:k]`. |
| `ordinary_coherence` | $\gamma^2_{iy}(f) \in [0, 1]$ per input (Eq. 7.109), shape `(q, F)`: each input against the output on its own. |
| `multiple_coherence` | $\gamma^2_{y:x}(f) \in [0, 1]$ (Eq. 7.35): the fraction of output power explained by all inputs jointly. Equals the sum of the partial coherences (Eq. 7.116) and `1 - noise_psd/output_psd`. |
| `partial_coherence` | $\gamma^2_{iy \cdot (i-1)!}(f) \in [0, 1]$ per input (Eq. 7.87, total-output denominator), shape `(q, F)`: the coherence of the input with the output once the linear effect of the inputs preceding it in `order` is removed. |
| `coherent_output_spectra` | Partial coherent output spectrum $G_{v_i}$ per input (Eq. 7.86), shape `(q, F)`: the output power the input contributes, with `sum(coherent_output_spectra) + noise_psd = output_psd`. |
| `output_psd` | Measured output autospectrum $\hat{G}_{yy}(f)$. |
| `noise_psd` | Residual (uncorrelated) output spectrum $\hat{G}_{nn} = \hat{G}_{yy \cdot q!}$ after removing every input (Eq. 7.121). |
| `multiple_coherence_random_error` | Normalized random error of $\gamma^2_{y:x}$ (Eq. 9.98), using $n_\mathrm{d} - (q-1)$ effective averages. |
| `coherent_output_random_error` | Normalized random error of each $G_{v_i}$ (Eq. 9.100), shape `(q, F)`, using $n_\mathrm{d} - (i-1)$ effective averages for the `i`-th ordered input. |
| `n_segments` | Raw number of (possibly overlapped) segments averaged. |
| `n_averages` | Effective number of independent averages $n_\mathrm{d}$. |
| `resolution_bandwidth` | Effective noise bandwidth $B_\mathrm{e}$, in Hz. |
| `window` | Taper name. |
| `nperseg` | Segment length, in samples. |
| `overlap` | Segment overlap fraction. |
| `scaling` | `'density'` or `'spectrum'`. |

### MISOCoherenceResult.dominant_input()

```python
MISOCoherenceResult.dominant_input() -> NDArray[np.intp]
```

Index of the input contributing the most output power per bin.

Returns, for every frequency, the original input index whose partial
coherent output spectrum `coherent_output_spectra` is largest -
the source that dominates that band. Bins where every contribution is
zero report the first input (index 0).

**Returns:** Integer array of length `len(frequencies)`.

### MISOCoherenceResult.plot()

```python
MISOCoherenceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the per-input coherent output spectra and multiple coherence.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
