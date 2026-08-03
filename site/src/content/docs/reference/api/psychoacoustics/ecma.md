---
title: "psychoacoustics.loudness.ecma"
description: "Psychoacoustic loudness per ECMA-418-2:2025 (4th ed., Sottek Hearing Model)."
sidebar:
  label: "ecma"
---

Psychoacoustic loudness per ECMA-418-2:2025 (4th ed., Sottek Hearing Model).

Clean-room implementation of the loudness signal chain of ECMA-418-2:2025:

* the shared auditory *front-end* of Clause 5 - trigonometric fade-in and
  zero-padding (5.1.2), the cascaded outer & middle/inner ear filter
  (5.1.3, Table 1), the 53-band gammatone-like auditory filter bank
  (5.1.4, Formulae 6-17), band-dependent segmentation (5.1.5, Table 4),
  half-wave rectification (5.1.6), block RMS (5.1.7), the compressive
  nonlinearity that maps sound pressure to specific loudness (5.1.8,
  Formula 23, Table 2) and the threshold in quiet (5.1.9, Table 3),
  yielding the specific basis loudness `N'_basis(l, z)` (Formula 25);
* the autocorrelation-based split into tonal and noise specific loudness
  with the full Clause 6.2.3 band averaging, including the
  cross-block-size-group ACF recomputation (Clause 6.2.2-6.2.7,
  Formulae 27-48; `_tonal_noise_split`, shared with the tonality
  metric so both report the same underlying N'_tonal); and
* the loudness assembly of Clause 8 - the tonal/noise power average
  (8.1.1, Formulae 113-114), the average specific loudness (8.1.2),
  the time-dependent loudness (8.1.3, Formula 116) and the single
  representative value (8.1.4, Formula 117).

The front-end helpers (`_ear_filter_sos`, `_auditory_bandpass`
and the band-parameter tables) are written to be reused by the later
tonality and roughness metrics of the same standard without refactoring.

The calibration constant `c_N` of Formula (23) is fixed by the standard so
that a 1 kHz sinusoid at 40 dB SPL yields 1 sone_HMS. With the full
Clause 6.2.3 averaging this chain computes 0.9845 sone_HMS (-1.55 %,
outside the +/-0.25 % adjustment the standard allows for c_N; c_N is kept
at the verbatim tabulated value rather than retuned). The residual is
driven by the mandated band averaging around the block-size-boundary bands
excited by the tone's lower flank (~800-900 Hz): without any band
averaging the chain reads 0.955, with averaging restricted to
same-block-size neighbours 0.996, and with the full cross-group
recomputation 0.9845. The block-time smoothing stage and the fade-in/LP
transient contribute \< 0.01 % (the value is signal-length invariant).

The API is monaural: the quadratic-mean binaural combination of
Formula (118) (Clause 8.1.5) is not implemented -- analyse each channel
separately.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## EcmaLoudness

```python
EcmaLoudness(
    loudness: float,
    specific_loudness: np.ndarray,
    bark: np.ndarray,
    centre_frequencies: np.ndarray,
    time: np.ndarray,
    loudness_vs_time: np.ndarray,
    field: str,
)
```

Result of an ECMA-418-2:2025 (Sottek) loudness calculation.

`loudness` is the single representative loudness N in sone_HMS
(Formula 117).  `specific_loudness` is the average specific loudness
N'(z) in sone_HMS/Bark_HMS over the 53 auditory bands (Formula 115),
with `bark` the critical-band-rate scale z (0,5..26,5 Bark_HMS) and
`centre_frequencies` the band centre frequencies F(z).  `time` and
`loudness_vs_time` hold the time-dependent loudness N(l) at 187,5 Hz
(Formula 116).  `field` records the assumed sound field.

### EcmaLoudness.plot()

```python
EcmaLoudness.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the average specific loudness N'(z) (see `..._plot.psychoacoustics`).

Adds a loudness-vs-time panel.  Requires matplotlib
(`pip install phonometry[plot]`).

## loudness_ecma

```python
loudness_ecma(
    signal_in: np.ndarray,
    fs: float,
    field: Literal['free', 'diffuse'] = 'free',
) -> EcmaLoudness
```

Psychoacoustic loudness per ECMA-418-2:2025 (Sottek Hearing Model).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal_in` | Calibrated sound pressure signal in pascals. |
| `fs` | Sampling rate in Hz. Signals not at 48 kHz are resampled (Clause 5.1.1). |
| `field` | `"free"` (default) or `"diffuse"` sound field, selecting the outer/middle-ear filter of Clause 5.1.3. |

**Returns:** An [`EcmaLoudness`](/phonometry/reference/api/psychoacoustics/ecma/#ecmaloudness) with the single loudness value N (Formula 117), the average specific loudness N'(z) (Formula 115) and the time-dependent loudness N(l) (Formula 116).

The 1 kHz / 40 dB SPL sinusoid defines 1 sone_HMS via the calibration
constant of Formula (23); with the full Clause 6.2.3 averaging this
chain computes 0.9845 sone_HMS for it (see the module docstring for the
residual's origin). Monaural only (no Clause 8.1.5 binaural combination).
