---
title: "psychoacoustics.loudness.moore_glasberg_time"
description: "Time-varying loudness per ISO 532-3:2023 (Moore-Glasberg-Schlittenlacher)."
sidebar:
  label: "moore_glasberg_time"
---

Time-varying loudness per ISO 532-3:2023 (Moore-Glasberg-Schlittenlacher).

Clean-room implementation of the time-varying loudness model of ISO 532-3:2023,
the extension of the stationary Moore-Glasberg method of ISO 532-2:2017 to
sounds whose loudness changes over time.  The procedure of Clause 7 turns a
calibrated pressure waveform into two time histories:

* the **short-term loudness** `S'(t)` in sone (clause 7.8), the momentary
  loudness of a short segment such as a spoken word or a musical note; and
* the **long-term loudness** `S''(t)` in sone (clause 7.9), the loudness of a
  longer segment such as a whole sentence or musical phrase.

The signal chain is:

* a running short-term spectrum computed from six parallel Hann-windowed FFTs
  whose segment durations (2, 4, 8, 16, 32 and 64 ms) trade temporal against
  spectral resolution, each contributing only its own frequency range, updated
  every 1 ms (clause 7.3);
* the level-dependent rounded-exponential excitation pattern of ISO 532-2
  sampled on the ERB-number grid from 1.75 Cam to 39 Cam in 0.25 Cam steps
  (clause 7.4, Formulae 1-6);
* the compressive transformation of excitation into instantaneous specific
  loudness `N'(i)` in sone/Cam (clause 7.5, Formulae 7-9, Tables 2-4, with
  $C = 0.063$ sone/Cam);
* an attack/release temporal smoothing of the specific loudness at every centre
  frequency to the short-term specific loudness (clause 7.6, Formulae 10-13);
* the across-frequency smoothing and binaural inhibition of ISO 532-2
  (clause 7.7, Formulae 14-17), and integration over the ERB-number scale to
  the short-term loudness of each ear and their binaural sum (clause 7.8);
* a slower attack/release temporal smoothing of the short-term loudness to the
  long-term loudness (clause 7.9, Formulae 18-21).

The loudness of a sound lasting up to about 5 s is well predicted by the
maximum of the long-term loudness (clause 7.9); this `n_max` and the
corresponding loudness level in phon (Table 5) are the headline results the
standard asks to report (clause 9).  A steady 1 kHz tone at 40 dB SPL presented
binaurally in a free field yields a long-term loudness of 1.000 sone (40 phon)
by definition of the sone; the additive spectral calibration of clause 7.3
(nominally +3.32 dB per component) is set so this anchor holds exactly.

**Conformance mode for the sampling rate.**  Clause 5 prescribes converting
the input to 32 kHz and clause 7.3 uses fixed 2048-point FFTs; this
implementation deliberately processes at the **native sampling rate** and
grows the FFT to `max(2048, next_pow2(segment))` so the 64 ms Hann window
is never truncated at 44.1/48 kHz (see
`test_low_freq_band_not_truncated_across_sample_rates`).  The spectral
calibration `_SPECTRAL_CAL_DB` was fixed against the 32 kHz anchor; at
other rates the Annex C.1 anchor reproduces with a 0.3 % cross-rate spread
(a 0.5 s tone reads `n_max` = 0.9918 / 0.9926 / 0.9900 at 32 / 44.1 /
48 kHz; a settled 1.3 s tone 1.0000 / 1.0000 / 0.9982), far inside the
standard's 2.8 phon expanded uncertainty.  These values are pinned by the
test suite; resample to 32 kHz first if letter-of-standard processing is
required.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## loudness_moore_glasberg_time

```python
loudness_moore_glasberg_time(
    signal: Signal | Sequence[float] | np.ndarray,
    fs: float | None = None,
    *,
    field: Literal['free', 'diffuse', 'eardrum'] = 'free',
    presentation: Literal['binaural', 'diotic', 'monaural'] = 'binaural',
    percentiles: Sequence[float] = (1.0, 5.0, 10.0, 50.0, 90.0, 95.0),
) -> MooreGlasbergTimeVaryingLoudness
```

Time-varying loudness of a calibrated pressure signal (ISO 532-3:2023).

Implements the Moore-Glasberg-Schlittenlacher method: a running short-term
spectrum (six parallel FFTs, clause 7.3) feeds the ISO 532-2 excitation and
specific-loudness model (clauses 7.4, 7.5), which is integrated over time to
the short-term loudness `S'(t)` (clauses 7.6-7.8) and long-term loudness
`S''(t)` (clause 7.9).  The signal must be calibrated so its samples are
the instantaneous sound pressure in pascals.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | Calibrated pressure signal in pascals.  A 1-D array is treated as diotic (the same sound at both ears) for a binaural/diotic presentation, or as the single active ear for a monaural presentation; a two-channel `(n, 2)` array gives the left and right ear signals. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), which is where "calibrated" comes from without arithmetic: this model reads absolute levels, so an uncalibrated record is taken as if one digital unit were one pascal and the answer is wrong by however far that is from true. |
| `fs` | Sampling rate in Hz (positive). Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `field` | Listening condition setting the outer-ear transfer: `"free"` (frontal free field, default), `"diffuse"` (diffuse field) or `"eardrum"` (levels already at the tympanic membrane). |
| `presentation` | `"binaural"`/`"diotic"` (default) or `"monaural"`. |
| `percentiles` | Fractions (percent) for which the exceeded long-term loudness is reported. |

**Returns:** A [`MooreGlasbergTimeVaryingLoudness`](/phonometry/reference/api/psychoacoustics/moore-glasberg-time/#mooreglasbergtimevaryingloudness).

A steady 1 kHz tone at 40 dB SPL, binaural, free field, yields a peak
long-term loudness of 1.000 sone (40 phon) by definition of the sone.

## MooreGlasbergTimeVaryingLoudness

```python
MooreGlasbergTimeVaryingLoudness(
    time: np.ndarray,
    short_term_loudness: np.ndarray,
    long_term_loudness: np.ndarray,
    short_term_loudness_level: np.ndarray,
    long_term_loudness_level: np.ndarray,
    n_max: float,
    loudness_level_max: float,
    percentiles: dict[float, float],
    field: str,
    presentation: str,
)
```

Result of an ISO 532-3:2023 time-varying loudness calculation.

`time` is the frame time axis in seconds (1 ms spacing, clause 7.3).
`short_term_loudness` and `long_term_loudness` are the binaural
short-term `S'(t)` (clause 7.8) and long-term `S''(t)` (clause 7.9)
loudness traces in sone; `short_term_loudness_level` and
`long_term_loudness_level` are the corresponding loudness levels in phon
(Table 5).  `n_max` is the maximum of the long-term loudness (sone) - the
predictor of the loudness of sounds up to about 5 s (clause 7.9) - and
`loudness_level_max` the phon value it maps to.  `percentiles` gives the
long-term-loudness values (sone) exceeded for the stated fraction of the
active trace (e.g. `percentiles[5]` is the level exceeded 5 % of the
time); the standard itself reports only the peak long-term loudness
(clause 9), the percentiles are provided as a convenience.  `field` and
`presentation` echo the listening conditions.

### MooreGlasbergTimeVaryingLoudness.plot()

```python
MooreGlasbergTimeVaryingLoudness.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the short-term and long-term loudness against time.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.  See `phonometry._plot.psychoacoustics`.
