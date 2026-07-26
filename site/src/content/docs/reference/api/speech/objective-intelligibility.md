---
title: "hearing.objective_intelligibility"
description: "Short-Time Objective Intelligibility (STOI and ESTOI)."
sidebar:
  label: "objective_intelligibility"
---

Short-Time Objective Intelligibility (STOI and ESTOI).

Implements the two correlation-based objective intelligibility measures that
predict the intelligibility of time-frequency weighted noisy speech, where the
speech-transmission index ([`phonometry.hearing.sti`](/phonometry/reference/api/speech/sti/)) and the speech
intelligibility index ([`phonometry.hearing.sii`](/phonometry/reference/api/speech/sii/)) are less appropriate:

* **STOI** (Taal, Hendriks, Heusdens and Jensen 2011, *An Algorithm for
  Intelligibility Prediction of Time-Frequency Weighted Noisy Speech*, IEEE
  TASLP 19(7); short version Taal et al. 2010, ICASSP): the average, over
  short-time segments and one-third-octave bands, of the correlation between
  the clean and degraded short-time temporal envelopes, after a per-segment
  normalisation and a signal-to-distortion clipping.
* **ESTOI** (Jensen and Taal 2016, *An Algorithm for Predicting the
  Intelligibility of Speech Masked by Modulated Noise Maskers*, IEEE/ACM TASLP
  24(11)): the extended measure that mean- and variance-normalises the
  short-time band-by-frame spectrogram over both rows (band envelopes) and
  columns (spectra), so the intermediate index is a spectral correlation. It
  tracks intelligibility better under modulated maskers and competing talkers,
  where STOI's band-independent correlation overestimates.

Both measures share the same front end: resampling to 10 kHz, a 256-sample
(25.6 ms) Hann-windowed, 50 %-overlapping short-time Fourier transform
zero-padded to 512 points, removal of the clean-speech frames more than 40 dB
below the loudest clean frame, a 15-band one-third-octave grouping of the DFT
magnitudes from a lowest centre of 150 Hz, and 30-frame (384 ms) analysis
segments. The output is a scalar in roughly `[0, 1]` with a monotonic
relation to the fraction of correctly understood words: 1 for a degraded
signal identical to the clean one, and near 0 for uncorrelated noise.

The band grouping snaps each one-third-octave edge to the nearest DFT bin and
the analysis window is MATLAB's `hanning` (`numpy.hanning` of length
`N+2` with its zero end-points dropped), matching the reference MATLAB
implementation of the authors; `pystoi` reproduces the same conventions and
is used as an external cross-check in the test suite (it is not a runtime
dependency).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## stoi

```python
stoi(
    clean: ArrayLike,
    degraded: ArrayLike,
    fs: int,
    *,
    extended: bool = False,
) -> STOIResult
```

Short-time objective intelligibility of degraded speech.

Computes STOI (Taal et al. 2011) or, with `extended=True`, ESTOI (Jensen
& Taal 2016): a scalar with a monotonic relation to the intelligibility of
`degraded` relative to the clean reference `clean`. Both signals are
resampled internally to 10 kHz, split into one-third-octave short-time
envelopes, and compared over 384 ms segments.

**Parameters**

| Name | Description |
| :--- | :--- |
| `clean` | The clean reference speech (1-D). |
| `degraded` | The degraded/processed speech (1-D, same length as `clean`, same sample rate). |
| `fs` | Sample rate of both signals, in hertz. |
| `extended` | Use ESTOI (spectral correlation, robust to modulated maskers) instead of STOI. |

**Returns:** A [`STOIResult`](/phonometry/reference/api/speech/objective-intelligibility/#stoiresult) with the index `.value` and its `.plot()`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the signals are not equal-length finite 1-D arrays, `fs` is not positive, or fewer than 30 short-time frames survive the silent-frame removal (too little speech to score). |

## STOIResult

```python
STOIResult(
    value: float,
    extended: bool,
    segment_scores: NDArray[np.float64],
    band_scores: NDArray[np.float64] | None,
    band_frequencies: NDArray[np.float64],
    sample_rate: int,
)
```

Result of a STOI or ESTOI intelligibility computation.

**Attributes**

| Name | Description |
| :--- | :--- |
| `value` | The overall intelligibility index (a scalar with a monotonic relation to the fraction of correctly understood words; 1 when the degraded signal equals the clean one). |
| `extended` | `True` for ESTOI (Jensen & Taal 2016), `False` for STOI (Taal et al. 2011). |
| `segment_scores` | Per-segment intermediate intelligibility (averaged over bands for STOI, the spectral correlation `d_m` for ESTOI). |
| `band_scores` | Per-band mean intermediate correlation over the segments (STOI only; `None` for ESTOI, whose index mixes the bands). |
| `band_frequencies` | The 15 one-third-octave band centre frequencies, in hertz. |
| `sample_rate` | The internal sample rate the measure runs at (10 kHz). |

### STOIResult.plot()

```python
STOIResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the intermediate intelligibility that averages to the index.

For STOI this is the mean correlation per one-third-octave band; for
ESTOI, whose index mixes the bands, it is the spectral correlation per
analysis segment. Requires matplotlib (`pip install phonometry[plot]`);
returns the `Axes`.
