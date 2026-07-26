---
title: "psychoacoustics.sharpness"
description: "Sharpness per DIN 45692:2009-08."
sidebar:
  label: "sharpness"
---

Sharpness per DIN 45692:2009-08.

Sharpness rates the high-frequency emphasis of a sound in acum, computed
as the g(z)-weighted first moment of the specific loudness pattern
(DIN 45692 Equation 1) over the 24 Bark critical-band scale, normalized so
the reference sound - critical-band-wide narrowband noise at 1 kHz
(920 Hz to 1080 Hz) at 60 dB overall level - yields exactly 1.00 acum
(clause 6). The informative Annex B variants of Aures and von Bismarck are
provided as alternative methods with the literal factor 0.11 printed in
Formulas (B.1)/(B.2); with it the reference sound lands near, not exactly
at, 1.00 acum (~0.96 Aures / ~1.02 von Bismarck through this front-end).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## sharpness_din

```python
sharpness_din(
    x: list[float] | np.ndarray,
    fs: int,
    field: Literal['free', 'diffuse'] = 'free',
    method: Literal['din', 'aures', 'bismarck'] = 'din',
    calibration_factor: float = 1.0,
) -> float
```

Sharpness of a signal per DIN 45692:2009 (stationary analysis).

The specific loudness comes from the ISO 532-1 Zwicker stationary
method (the DIN 45631 basis named by DIN 45692); the sharpness is its
g(z)-weighted first moment, normalized so the reference sound of
clause 6 (critical-band-wide noise, 1 kHz, 60 dB) gives 1.00 acum.
Verification tolerance per clause 6: 5 % or 0.05 acum against the
Table A.2/A.3 target values. A small, systematic negative bias grows
with centre frequency for narrowband stimuli (about -0.03 acum at
2.5 kHz to -0.15 acum at 8.5 kHz, from the specific-loudness upper-slope
handling near the 24 Bark edge); it stays within the 5 % / 0.05 acum
tolerance and is a known DIN 45692 implementation property.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Input signal (1D), in Pa after `calibration_factor`. |
| `fs` | Sample rate in Hz. |
| `field` | `'free'` (default) or `'diffuse'`. |
| `method` | `'din'` (default), `'aures'` or `'bismarck'`. |
| `calibration_factor` | Digital-units-to-Pa factor. |

**Returns:** Sharpness in acum.

## sharpness_din_from_specific

```python
sharpness_din_from_specific(
    specific: np.ndarray,
    method: Literal['din', 'aures', 'bismarck'] = 'din',
) -> float
```

Sharpness in acum from a specific-loudness pattern (DIN 45692 Eq. 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `specific` | Specific loudness N'(z) at 0.1 Bark steps (240 values), e.g. `ZwickerLoudness.specific`. |
| `method` | `'din'` (default, Eq. 1), or the informative Annex B variants `'aures'` / `'bismarck'`. |

**Returns:** Sharpness in acum (0.0 for silence).
