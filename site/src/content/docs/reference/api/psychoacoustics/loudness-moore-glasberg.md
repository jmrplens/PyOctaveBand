---
title: "psychoacoustics.loudness_moore_glasberg"
description: "Stationary loudness per ISO 532-2:2017 (Moore-Glasberg method)."
sidebar:
  label: "loudness_moore_glasberg"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Stationary loudness per ISO 532-2:2017 (Moore-Glasberg method).

Clean-room implementation of the spectral loudness model of ISO 532-2:2017,
the third calculation method of the ISO 532 series (distinct from the Zwicker
method of ISO 532-1 and the Sottek model of ECMA-418-2).  The procedure of
Clause 7 (Figure 1) transforms a stationary sound spectrum into a loudness
value through five stages:

* the fixed outer-ear and middle-ear transfer functions that map the recorded
  spectrum to the spectrum at the cochlea (clauses 7.2, 7.3, Table 1);
* the level-dependent rounded-exponential (roex) auditory filter bank that
  turns that spectrum into an excitation pattern along the ERB-number scale
  (clause 7.4, Formulae 1-6), sampled at ERB-number `i` from 1.8 Cam to
  38.9 Cam in 0.1 Cam steps;
* the compressive transformation of excitation into specific loudness
  `N'(i)` in sone/Cam (clause 7.5, Formulae 7-9, Tables 2-4);
* the binaural inhibition model that combines the two ears (clause 8.1,
  Formulae 10-13); and
* the integration of specific loudness over the ERB-number scale to the total
  loudness `N` in sone, with the loudness level `L_N` in phon obtained by
  inverting the loudness/loudness-level relationship of Table 5 (clause 8.2).

The specific-loudness calibration constant `C` of Formula (7) is the value
tabulated by the standard (0.0617 sone/Cam); a 1 kHz tone at 40 dB SPL
presented binaurally in a free field yields 1.000 sone by definition of the
sone (clause 3.17), which this implementation reproduces without tuning.

The stationary method is spectrum based.  [`loudness_moore_glasberg_from_spectrum`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/#loudness_moore_glasberg_from_spectrum)
takes the exact sinusoidal-component representation of clauses 5.2/5.4,
[`loudness_moore_glasberg_from_third_octave`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/#loudness_moore_glasberg_from_third_octave) takes the 29 one-third-octave
band levels of clause 5.5, and [`loudness_moore_glasberg`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/#loudness_moore_glasberg) forms the
narrowband (FFT) line spectrum of a calibrated pressure signal and feeds it to
the exact sinusoidal-component method.

## loudness_moore_glasberg

```python
loudness_moore_glasberg(
    x: list[float] | np.ndarray,
    fs: float,
    *,
    field: Literal['free', 'diffuse', 'eardrum'] = 'free',
    presentation: Literal['binaural', 'diotic', 'monaural'] = 'binaural',
) -> MooreGlasbergLoudness
```

Moore-Glasberg loudness of a calibrated stationary pressure signal.

Convenience wrapper around
[`loudness_moore_glasberg_from_spectrum`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/#loudness_moore_glasberg_from_spectrum): the signal's narrowband
(FFT) line spectrum is formed and each bin is passed to the exact
sinusoidal-component method of ISO 532-2:2017 (clauses 5.2/5.4).  Because
the model computes the excitation pattern per spectral component
(Formula 5), a pure tone enters as a single line - so a calibrated 1 kHz
tone at 40 dB SPL yields 1.000 sone / 40 phon (the definitional anchor),
matching [`loudness_moore_glasberg_from_spectrum`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/#loudness_moore_glasberg_from_spectrum).  The signal must be
calibrated so that `x` is the instantaneous sound pressure in pascals.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Single-channel calibrated pressure signal in pascals. |
| `fs` | Sampling rate in Hz (positive). |
| `field` | `"free"` (default), `"diffuse"` or `"eardrum"`. |
| `presentation` | `"binaural"` (default; alias `"diotic"`) or `"monaural"`. |

**Returns:** A [`MooreGlasbergLoudness`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/#mooreglasbergloudness).

## loudness_moore_glasberg_from_spectrum

```python
loudness_moore_glasberg_from_spectrum(
    components: Sequence[tuple[float, float]] | np.ndarray,
    *,
    field: Literal['free', 'diffuse', 'eardrum'] = 'free',
    presentation: Literal['binaural', 'diotic', 'monaural'] = 'binaural',
) -> MooreGlasbergLoudness
```

Moore-Glasberg loudness from a sinusoidal-component spectrum.

Implements the exact spectral input of ISO 532-2:2017 clauses 5.2/5.4:
the sound is specified as a set of discrete sinusoidal components, each a
`(frequency_Hz, level_dB_SPL)` pair (levels re 20 uPa in the stated
sound field).  Bands of noise can be represented by the equivalent set of
closely spaced components of clause 5.3.

**Parameters**

| Name | Description |
| :--- | :--- |
| `components` | Sequence of `(frequency_Hz, level_dB)` pairs (or an `(n, 2)` array).  An empty spectrum yields zero loudness. |
| `field` | Listening condition setting the outer-ear transfer: `"free"` (frontal free field, default), `"diffuse"` (diffuse field) or `"eardrum"` (levels already specified at the tympanic membrane, e.g. a flat earphone). |
| `presentation` | `"binaural"` (default; `"diotic"` is an equivalent alias: the same sound at both ears) or `"monaural"` (one ear only). |

**Returns:** A [`MooreGlasbergLoudness`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/#mooreglasbergloudness).

A 1 kHz tone at 40 dB SPL in a free field presented binaurally yields
1.000 sone / 40 phon by definition of the sone.

## loudness_moore_glasberg_from_third_octave

```python
loudness_moore_glasberg_from_third_octave(
    band_levels: Sequence[float] | np.ndarray,
    *,
    field: Literal['free', 'diffuse', 'eardrum'] = 'free',
    presentation: Literal['binaural', 'diotic', 'monaural'] = 'binaural',
) -> MooreGlasbergLoudness
```

Moore-Glasberg loudness from 29 one-third-octave band levels.

Implements the practical spectral input of ISO 532-2:2017 clause 5.5: the
sound is specified by the sound pressure levels in the 29 adjacent
one-third-octave bands with nominal centre frequencies 25 Hz to 16 kHz
(IEC 61260-1).  Each band is expanded into the equivalent set of
sinusoidal components and the exact method is applied.

**Parameters**

| Name | Description |
| :--- | :--- |
| `band_levels` | 29 band levels in dB SPL (re 20 uPa), 25 Hz..16 kHz. |
| `field` | `"free"` (default), `"diffuse"` or `"eardrum"`. |
| `presentation` | `"binaural"` (default; alias `"diotic"`) or `"monaural"`. |

**Returns:** A [`MooreGlasbergLoudness`](/phonometry/reference/api/psychoacoustics/loudness-moore-glasberg/#mooreglasbergloudness).

## MooreGlasbergLoudness

```python
MooreGlasbergLoudness(
    loudness: float,
    loudness_level: float,
    specific: np.ndarray,
    erb_number: np.ndarray,
    centre_frequencies: np.ndarray,
    field: str,
    presentation: str,
)
```

Result of an ISO 532-2:2017 Moore-Glasberg loudness calculation.

`loudness` is the total loudness N in sone; `loudness_level` is the
loudness level L_N in phon (obtained by inverting the loudness/loudness-
level relationship of Table 5).  `specific` holds the specific loudness
N'(i) in sone/Cam sampled at the ERB-number grid `erb_number` (Cam),
i.e. i from 1.8 Cam to 38.9 Cam in 0.1 Cam steps (372 values), and
`centre_frequencies` the corresponding auditory-filter centre
frequencies in Hz.  For a binaural (diotic) result the specific-loudness
pattern is that of a single ear before binaural inhibition; the total
`loudness` already includes the binaural summation.  `field` and
`presentation` echo the listening conditions.

### MooreGlasbergLoudness.plot()

```python
MooreGlasbergLoudness.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the specific loudness N'(i) over the ERB-number scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.  See `._plotting`.
