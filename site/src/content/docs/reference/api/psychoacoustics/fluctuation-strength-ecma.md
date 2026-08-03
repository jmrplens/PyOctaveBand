---
title: "psychoacoustics.quality.fluctuation_strength_ecma"
description: "Psychoacoustic fluctuation strength per ECMA-418-2:2025 (4th ed., Clause 9)."
sidebar:
  label: "fluctuation_strength_ecma"
---

Psychoacoustic fluctuation strength per ECMA-418-2:2025 (4th ed., Clause 9).

Clean-room implementation of the fluctuation-strength signal chain of
ECMA-418-2:2025 (Clause 9, Sottek Hearing Model). The shared auditory
front-end (Clause 5: outer/middle-ear filter, 53-band auditory filter bank,
compressive nonlinearity of Formulae 23-25) is reused from
[`phonometry.psychoacoustics.loudness.ecma`](/phonometry/reference/api/psychoacoustics/ecma/); this module adds the
fluctuation-strength chain, which mirrors the roughness chain of Clause 7 but
replaces the DFT-based envelope analysis with High-resolution Spectral
Analysis (HSA):

* fluctuation-strength zero-padding (Clause 5.1.2.2) and segmentation
  (Clause 5.1.5.2) with the fixed block/hop $s_b = 65536$ /
  $s_h = 16384$
  (Clause 9.1.1);
* the Hilbert envelope of each critical-band block and a factor-32
  downsampling to 1500 Hz (Clause 9.1.2, Formula 119);
* the envelope-dependent analysis windows with quieter-period detection
  (Clause 9.1.3, Formula 120);
* the HSA of the windowed envelopes -- a least-squares fit of window-kernel
  spectral line pairs to the k = 0..48 DFT bins (Clause 9.1.4,
  Formulae 121-142);
* the identification of prominent spectral line pairs from the local maxima
  of the power spectrum and the local minima of the HSA error over a
  logarithmic modulation-rate grid (Clause 9.1.5, Formulae 143-146);
* the band-pass modulation-rate weighting (Clause 9.1.6, Formulae 147-148),
  the damped-Newton fine tuning of the dominant modulation rate
  (Clause 9.1.7, Formulae 149-152), the harmonic analysis (Clause 9.1.8,
  Formulae 153-156) and the centre-of-gravity weighting (Clause 9.1.9,
  Formulae 157-158);
* the scaling with the HSA-based specific loudness (Clause 9.1.10,
  Formulae 159-161); and
* the interpolation to 50 Hz, the distribution-dependent nonlinear transform
  with the calibration constant `c_F`, the first-order smoothing
  (Clause 9.1.11, Formulae 162-168) and the aggregation into the average
  specific fluctuation strength F'(z) (Clause 9.1.12), the time-dependent
  fluctuation strength F(l50) (Clause 9.1.13, Formula 169) and the
  representative 90th-percentile single value F (Clause 9.1.14).

The API is monaural: the quadratic-mean binaural combination of
Formula (170) (Clause 9.1.15) is not implemented -- analyse each channel
separately.

The calibration constant `c_F` of Formula (163) is the standard's tabulated
value (not reverse-fit), and the chain reproduces the Clause 9 reference
point: a 1 kHz carrier 100 %-amplitude-modulated at 4 Hz with an overall
sound pressure level of 60 dB SPL converges to 0.9958 vacil_HMS against the
defined 1 vacil_HMS as the 90th percentile settles (0.9931 for a 5 s signal,
0.9957 at 8 s, 0.9958 by 12 s). Footnote 47 allows a +/-0.25 % adjustment
of c_F, which is not used. The level convention follows Clause 7/9 as
established for the roughness metric: the stated 60 dB is the overall RMS
level of the modulated signal, not the carrier level.

Clause 9 interpretation notes (all resolved by internal consistency and
pinned by the calibration signal; the confirmed defects are recorded in
`docs/ERRATA.md`):

* Formula (127) prints the phase factor
  $\exp(-j 2\pi f_n (\tilde{s}_b - n_{ze} + n_{zb} - 1))$; the DFT of
  the rectangular analysis window requires $\pi$
  in place of $2\pi$ (with $2\pi$ the HSA cannot reproduce the
  very
  spectra it fits, breaking the exact-recovery property claimed for it).
* Formula (144) subtracts 1 from the three-bin centroid before scaling by
  `delta_f`; with the 0-based bin convention stated below Formula (122)
  (bin k maps to $k \tilde{r}_s / \tilde{s}_b$) that offset shifts
  every modulation
  rate one bin low, so the centroid is used without the offset.
* Clause 9.1.7 states the Newton constants (differential step 1e-5, damped
  step cap 2e-4, stop tolerance 1e-7) without units; read in Hz they cap the
  total displacement at 2e-3 Hz, which makes the fine tuning inert and the
  failure check against 1.25 * delta_f (~0.92 Hz) unreachable. They are
  self-consistent as normalized frequencies (f / r~_s), which is how this
  module applies them.
* The amplitude of a spectral line pair (Formulae 146-147, 155, 157-160) is
  taken as the squared magnitude of the half-line solution components of
  Formula (123), $x_{2m}^2 + x_{2m+1}^2 = |\hat{p}_m|^2 / 4$: with
  that reading
  Formula (160) is exactly the RMS of the modelled band signal (the loudness
  chain of Formulae 22-23 applied to the harmonic complex) and the tabulated
  c_F reproduces the calibration signal.
* The \< 0.125 Hz discard of Clause 9.1.7 names `f_c,1,opt`; it is applied
  here to the rate that survives the failure check (the original
  `f~_c,imax` when the fine tuning was cancelled), since discarding a
  block on a diverged optimizer output would drop healthy modulation.
* For signals shorter than the ~0.74 s transient (fewer than 37 frames at
  50 Hz), the Clause 9.1.12/9.1.14 discard of l50 = 0..35 would leave no
  frames; the aggregation then falls back to all frames instead of
  reporting an empty (zero) result.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## EcmaFluctuationStrength

```python
EcmaFluctuationStrength(
    fluctuation_strength: float,
    specific_fluctuation_strength: np.ndarray,
    bark: np.ndarray,
    centre_frequencies: np.ndarray,
    time: np.ndarray,
    fluctuation_strength_vs_time: np.ndarray,
    specific_fluctuation_strength_vs_time: np.ndarray,
    field: str,
)
```

Result of an ECMA-418-2:2025 (Sottek) fluctuation-strength calculation.

`fluctuation_strength` is the single representative fluctuation
strength F in vacil_HMS (the 90th percentile of F(l50), Clause 9.1.14).
`specific_fluctuation_strength` is the average specific fluctuation
strength F'(z) in vacil_HMS/Bark_HMS over the 53 auditory bands
(Clause 9.1.12), with `bark` the critical-band-rate scale z
(0.5..26.5 Bark_HMS) and `centre_frequencies` the band centre
frequencies F(z). `time` and `fluctuation_strength_vs_time` hold the
time-dependent fluctuation strength F(l50) at 50 Hz (Formula 169);
`specific_fluctuation_strength_vs_time` is the time-dependent specific
fluctuation strength F'(l50, z) (Formula 168) of shape
`(n_times, 53)`. `field` records the assumed sound field.

### EcmaFluctuationStrength.plot()

```python
EcmaFluctuationStrength.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the fluctuation-strength result (see `._plotting`).

Draws the time-dependent fluctuation strength F(l50) and a
specific-fluctuation-strength heatmap. Requires matplotlib
(`pip install phonometry[plot]`).

## fluctuation_strength_ecma

```python
fluctuation_strength_ecma(
    signal_in: np.ndarray,
    fs: float,
    field: Literal['free', 'diffuse'] = 'free',
) -> EcmaFluctuationStrength
```

Psychoacoustic fluctuation strength per ECMA-418-2:2025 (Clause 9).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal_in` | Calibrated sound pressure signal in pascals. |
| `fs` | Sampling rate in Hz. Signals not at 48 kHz are resampled (Clause 5.1.1). |
| `field` | `"free"` (default) or `"diffuse"` sound field, selecting the outer/middle-ear filter of Clause 5.1.3. |

**Returns:** An [`EcmaFluctuationStrength`](/phonometry/reference/api/psychoacoustics/fluctuation-strength-ecma/#ecmafluctuationstrength) with the single value F (Clause 9.1.14), the average specific fluctuation strength F'(z) (Clause 9.1.12) and the time-dependent fluctuation strength F(l50) (Formula 169).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an empty, multichannel or non-finite signal, a non-finite or non-positive `fs`, or an unknown `field`. |

A 1 kHz carrier 100 %-amplitude-modulated at 4 Hz with an overall level
of 60 dB SPL yields 1 vacil_HMS (Clause 9 calibration; reproduced to
0.9958 vacil_HMS with the tabulated c_F of Formula (163) once the 90th
percentile settles, by a 12 s signal).
