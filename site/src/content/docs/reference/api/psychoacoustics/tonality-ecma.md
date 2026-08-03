---
title: "psychoacoustics.quality.tonality_ecma"
description: "Psychoacoustic tonality per ECMA-418-2:2025 (4th ed., Sottek Hearing Model)."
sidebar:
  label: "tonality_ecma"
---

Psychoacoustic tonality per ECMA-418-2:2025 (4th ed., Sottek Hearing Model).

Clean-room implementation of the tonality signal chain of ECMA-418-2:2025
(Clause 6.2). The shared auditory front-end (Clause 5) and the ACF-based
tonal/noise decomposition with the full Clause 6.2.3 band averaging
(Clause 6.2.2-6.2.7,
[`phonometry.psychoacoustics.loudness.ecma._tonal_noise_split`](/phonometry/reference/api/psychoacoustics/ecma/)) are
reused from [`phonometry.psychoacoustics.loudness.ecma`](/phonometry/reference/api/psychoacoustics/ecma/) -- loudness and
tonality therefore report the same underlying `N'_tonal(l, z)` for the same
signal; this module adds

* the tonality output stages (Clause 6.2.8-6.2.11): the overall-SNR gate
  `q(l)` (Formulae 49-50), the time-dependent specific tonality
  $T'(l, z) = c_T q(l) N'_{tonal}(l, z)$ (Formula 51), the average
  specific tonality `T'(z)` and its frequency `f_ton,z(z)` (Formulae
  53-55), the time-dependent tonality `T(l)` with its frequency
  `f_ton(l)` (Formulae 61-62) and the representative single value `T`
  (Formulae 63-64).

The calibration factor `c_T` of Formula (51) is fixed by the standard so
that a 1 kHz sinusoid at 40 dB SPL yields 1 tu_HMS.

The API is monaural; analyse each channel separately. (Unlike its
roughness and loudness, ECMA-418-2 defines no binaural combination for
tonality.)

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## EcmaTonality

```python
EcmaTonality(
    tonality: float,
    specific_tonality: np.ndarray,
    bark: np.ndarray,
    centre_frequencies: np.ndarray,
    tonal_frequencies: np.ndarray,
    time: np.ndarray,
    tonality_vs_time: np.ndarray,
    tonal_frequency_vs_time: np.ndarray,
    field: str,
)
```

Result of an ECMA-418-2:2025 (Sottek) tonality calculation.

`tonality` is the single representative tonality T in tu_HMS
(Formula 63). `specific_tonality` is the average specific tonality
T'(z) in tu_HMS over the 53 auditory bands (Formula 53), with `bark`
the critical-band-rate scale z (0.5..26.5 Bark_HMS), `centre_frequencies`
the band centre frequencies F(z) and `tonal_frequencies` the per-band
tonal frequency f_ton,z(z) (Formula 55). `time` and `tonality_vs_time`
hold the time-dependent tonality T(l) at 187.5 Hz (Formula 61) and
`tonal_frequency_vs_time` its frequency f_ton(l) (Formula 62). `field`
records the assumed sound field.

### EcmaTonality.plot()

```python
EcmaTonality.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the average specific tonality T'(z) (see `._plotting`).

Adds a tonality-vs-time panel. Requires matplotlib
(`pip install phonometry[plot]`).

## tonality_ecma

```python
tonality_ecma(
    signal_in: np.ndarray,
    fs: float,
    field: Literal['free', 'diffuse'] = 'free',
    f_low: float | None = None,
    f_high: float | None = None,
) -> EcmaTonality
```

Psychoacoustic tonality per ECMA-418-2:2025 (Sottek Hearing Model).

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal_in` | Calibrated sound pressure signal in pascals. |
| `fs` | Sampling rate in Hz. Signals not at 48 kHz are resampled (Clause 5.1.1). |
| `field` | `"free"` (default) or `"diffuse"` sound field, selecting the outer/middle-ear filter of Clause 5.1.3. |
| `f_low` | Optional lower edge (Hz) of a user frequency band for the time-dependent tonality maximum search (Formulae 56-60). `None` uses the full range. |
| `f_high` | Optional upper edge (Hz) of the user frequency band. |

**Returns:** An [`EcmaTonality`](/phonometry/reference/api/psychoacoustics/tonality-ecma/#ecmatonality) with the single value T (Formula 63), the average specific tonality T'(z) (Formula 53), the tonal frequencies f_ton,z(z) (Formula 55) and the time-dependent tonality T(l) (Formula 61) with its frequency (Formula 62).

The 1 kHz / 40 dB SPL sinusoid yields 1 tu_HMS by construction of the
calibration factor of Formula (51).
