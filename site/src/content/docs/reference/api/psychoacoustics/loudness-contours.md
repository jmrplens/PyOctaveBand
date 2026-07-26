---
title: "psychoacoustics.loudness_contours"
description: "Normal equal-loudness-level contours per ISO 226:2023."
sidebar:
  label: "loudness_contours"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Normal equal-loudness-level contours per ISO 226:2023.

Implements Formula (1) (clause 4.1: SPL of a pure tone from its loudness
level) and Formula (2) (clause 4.2: the inverse) with the Table 1 (p. 4)
parameters at the 29 preferred third-octave frequencies of ISO 266.

## equal_loudness_contour

```python
equal_loudness_contour(phon: float) -> tuple[np.ndarray, np.ndarray]
```

Normal equal-loudness-level contour (ISO 226:2023 Formula 1).

Returns the sound pressure levels of pure tones judged equally loud as
a 1 kHz tone at `phon` dB SPL, at the 29 preferred third-octave
frequencies of Table 1.

Validity (clause 4.1): 20 phon to 90 phon between 20 Hz and 4 kHz, and
up to 80 phon between 5 kHz and 12.5 kHz; above 80 phon the returned
contour therefore stops at 4 kHz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `phon` | Loudness level in phons (20 to 90). |

**Returns:** Tuple `(frequencies, spl)` in Hz and dB re 20 uPa.

## equal_loudness_contours

```python
equal_loudness_contours(
    phons: Sequence[float] = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0),
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> EqualLoudnessContours
```

Build the ISO 226:2023 equal-loudness-level contour family.

Evaluates [`equal_loudness_contour`](/phonometry/reference/api/psychoacoustics/loudness-contours/#equal_loudness_contour) at each level in `phons` and
[`hearing_threshold`](/phonometry/reference/api/psychoacoustics/loudness-contours/#hearing_threshold), and bundles them into an
[`EqualLoudnessContours`](/phonometry/reference/api/psychoacoustics/loudness-contours/#equalloudnesscontours) result that exposes `.plot()`. The maths
is unchanged; this is a thin, plottable wrapper around the existing
functions.

**Parameters**

| Name | Description |
| :--- | :--- |
| `phons` | Loudness levels of the contours, in phon; each must be in the 20 phon to 90 phon validity range of Formula (1). Defaults to the classic family from 20 phon to 90 phon in 10 phon steps. |
| `frequencies` | Frequency grid in Hz; `None` (default) uses the 29 preferred third-octave frequencies of Table 1. Any value given must be one of those preferred frequencies (the standard specifies no interpolation between them). |

**Returns:** An [`EqualLoudnessContours`](/phonometry/reference/api/psychoacoustics/loudness-contours/#equalloudnesscontours).

## EqualLoudnessContours

```python
EqualLoudnessContours(
    frequencies: np.ndarray,
    phons: tuple[float, ...],
    contours: np.ndarray,
    threshold: np.ndarray,
)
```

The ISO 226:2023 normal equal-loudness-level contour family.

Bundles a set of equal-loudness contours (ISO 226:2023 Formula 1) with
the threshold of hearing (Table 1) over a shared frequency grid, so the
iconic chart can be drawn with `plot`.

`frequencies` is the frequency grid in Hz (by default the 29 preferred
third-octave frequencies of Table 1). `phons` lists the loudness levels
of the contours, in phon. `contours` holds the sound pressure levels in
dB re 20 uPa as a `(len(phons), len(frequencies))` array, row `i` being
the contour for `phons[i]`; entries the standard does not define are
`nan` (Formula 1 is valid only up to 4 kHz above 80 phon, so those rows
stop at 4 kHz). `threshold` is the threshold of hearing T_f in dB re
20 uPa on the same grid.

### EqualLoudnessContours.plot()

```python
EqualLoudnessContours.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the equal-loudness contour family with the hearing threshold.

Draws sound pressure level in dB against a logarithmic frequency axis:
one line per loudness level in `phons`, plus the threshold of
hearing. Requires matplotlib (`pip install phonometry[plot]`);
returns the `Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes to draw on, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the contour `plot` calls. |

**Returns:** The axes.

## hearing_threshold

```python
hearing_threshold() -> tuple[np.ndarray, np.ndarray]
```

Threshold of hearing T_f (ISO 226:2023 Table 1).

**Returns:** Tuple `(frequencies, threshold)` in Hz and dB re 20 uPa.

## loudness_level

```python
loudness_level(spl: float, frequency: float) -> float
```

Loudness level of a pure tone (ISO 226:2023 Formula 2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `spl` | Sound pressure level of the tone in dB re 20 uPa. |
| `frequency` | Tone frequency in Hz; must be one of the 29 preferred third-octave frequencies of Table 1 (the standard specifies no interpolation between them). |

**Returns:** Loudness level in phons. Values outside 20-90 phon (80 above 4 kHz) are extrapolations the standard labels as informative only.
