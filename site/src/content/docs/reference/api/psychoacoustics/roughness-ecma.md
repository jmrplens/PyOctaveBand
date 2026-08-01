---
title: "psychoacoustics.roughness_ecma"
description: "Psychoacoustic roughness per ECMA-418-2:2025 (4th ed., Sottek Hearing Model)."
sidebar:
  label: "roughness_ecma"
---

Psychoacoustic roughness per ECMA-418-2:2025 (4th ed., Sottek Hearing Model).

Clean-room implementation of the roughness signal chain of ECMA-418-2:2025
(Clause 7). The shared auditory front-end (Clause 5: outer/middle-ear filter,
53-band gammatone-like filter bank, half-wave rectification, compressive
nonlinearity to the specific basis loudness `N'_basis(l, z)` of Formula 25)
is reused from [`.loudness_ecma`](/phonometry/reference/api/psychoacoustics/loudness-ecma/#loudness_ecma); this module adds the roughness-specific
chain:

* roughness-specific zero-padding (Clause 5.1.2.2) and segmentation
  (Clause 5.1.5.2) with the fixed block/hop $s_b = 16384$ /
  $s_h = 4096$
  (Clause 7.1.1);
* the Hilbert envelope of each critical-band block and a factor-32 downsampling
  to 1500 Hz (Clause 7.1.2, Formula 65);
* the scaled envelope power spectrum `Phi_E,l,z(k)` (Clause 7.1.3,
  Formulae 66-67);
* the two-step noise reduction of the envelope spectra (Clause 7.1.4,
  Formulae 68-71);
* the four-stage spectral weighting (Clause 7.1.5, Formulae 72-96): peak
  picking with a quadratic-fit modulation-rate refinement and a bias
  correction, the high-modulation-rate weighting, the fundamental
  modulation-rate estimation, and the low-modulation-rate weighting;
* the interpolation to 50 Hz, the distribution-dependent nonlinear transform
  with the calibration constant `c_R`, and the asymmetric time smoothing
  (Clause 7.1.7, Formulae 103-110); and
* the average specific roughness `R'(z)` (Clause 7.1.8), the time-dependent
  roughness `R(l50)` (Clause 7.1.9, Formula 111) and the representative
  90th-percentile single value `R` (Clause 7.1.10).

The optional entropy weighting of Clause 7.1.6 requires an external rotational
speed signal and is not implemented (see `notes-ecma418-2-roughness.md`).
The API is monaural: the quadratic-mean binaural combination of
Formula (112) (Clause 7.1.11) is not implemented -- analyse each channel
separately.

The calibration constant `c_R` of Formula (104) is the standard's tabulated
value (not reverse-fit), and the chain reproduces the Clause 7 reference
point exactly: a 1 kHz carrier 100 %-amplitude-modulated at 70 Hz with an
overall sound pressure level of 60 dB SPL computes to 0.9999 asper against
the defined 1 asper. Note the level convention: Clause 7 states the *sound
pressure level of the signal* (its overall RMS level), not the level of the
unmodulated carrier -- a fully modulated signal whose carrier alone sits at
60 dB is +1.76 dB hot overall and reads ~4 % high.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## EcmaRoughness

```python
EcmaRoughness(
    roughness: float,
    specific_roughness: np.ndarray,
    bark: np.ndarray,
    centre_frequencies: np.ndarray,
    time: np.ndarray,
    roughness_vs_time: np.ndarray,
    specific_roughness_vs_time: np.ndarray,
    field: str,
)
```

Result of an ECMA-418-2:2025 (Sottek) roughness calculation.

`roughness` is the single representative roughness R in asper (the
90th percentile of R(l50), Clause 7.1.10). `specific_roughness` is the
average specific roughness R'(z) in asper/Bark_HMS over the 53 auditory
bands (Clause 7.1.8), with `bark` the critical-band-rate scale z
(0.5..26.5 Bark_HMS) and `centre_frequencies` the band centre
frequencies F(z). `time` and `roughness_vs_time` hold the
time-dependent roughness R(l50) at 50 Hz (Formula 111);
`specific_roughness_vs_time` is the time-dependent specific roughness
R'(l50, z) (Formula 109) of shape `(n_times, 53)`. `field` records the
assumed sound field.

### EcmaRoughness.plot()

```python
EcmaRoughness.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the roughness result (see `._plotting`).

Draws the time-dependent roughness R(l50) and a specific-roughness
heatmap. Requires matplotlib (`pip install phonometry[plot]`).

## roughness_ecma

```python
roughness_ecma(
    signal_in: np.ndarray,
    fs: float,
    field: Literal['free', 'diffuse'] = 'free',
) -> EcmaRoughness
```

Psychoacoustic roughness per ECMA-418-2:2025 (Sottek Hearing Model).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal_in` | Calibrated sound pressure signal in pascals. |
| `fs` | Sampling rate in Hz. Signals not at 48 kHz are resampled (Clause 5.1.1). |
| `field` | `"free"` (default) or `"diffuse"` sound field, selecting the outer/middle-ear filter of Clause 5.1.3. |

**Returns:** An [`EcmaRoughness`](/phonometry/reference/api/psychoacoustics/roughness-ecma/#ecmaroughness) with the single value R (Clause 7.1.10), the average specific roughness R'(z) (Clause 7.1.8) and the time-dependent roughness R(l50) (Formula 111).

A 1 kHz carrier 100 %-amplitude-modulated at 70 Hz with an overall level
of 60 dB SPL yields 1 asper (Clause 7 calibration; reproduced to 0.9999
asper with the tabulated c_R of Formula (104)).
