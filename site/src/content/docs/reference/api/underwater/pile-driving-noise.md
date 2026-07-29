---
title: "underwater.pile_driving_noise"
description: "Radiated underwater sound from percussive pile driving (ISO 18406:2017)."
sidebar:
  label: "pile_driving_noise"
---

Radiated underwater sound from percussive pile driving (ISO 18406:2017).

Percussive pile driving radiates a train of impulsive acoustic pulses, one per
hammer strike. ISO 18406 characterises them with:

* [`single_strike_sel`](/phonometry/reference/api/underwater/pile-driving-noise/#single_strike_sel) -- the single-strike sound exposure level
  `SEL_ss` of one pulse (Formulae 3-4), reusing the 1 µPa²·s reference.
* [`cumulative_sel`](/phonometry/reference/api/underwater/pile-driving-noise/#cumulative_sel) / [`cumulative_sel_identical`](/phonometry/reference/api/underwater/pile-driving-noise/#cumulative_sel_identical) -- the cumulative
  sound exposure level over N strikes (Formulae 8-9); for N identical strikes
  `SEL_cum = SEL_ss + 10·lg(N)`.
* [`pile_strike_metrics`](/phonometry/reference/api/underwater/pile-driving-noise/#pile_strike_metrics) -- a [`PileStrikeResult`](/phonometry/reference/api/underwater/pile-driving-noise/#pilestrikeresult) bundling the
  single-strike SEL, the peak sound pressure level, the SPL/Leq and the
  90 %-energy pulse duration for one recorded strike, with a `.plot()`.
* [`strike_sel_spectrum`](/phonometry/reference/api/underwater/pile-driving-noise/#strike_sel_spectrum) -- the same single-strike SEL resolved into
  fractional-octave bands (ISO 18406 6.4.2.2), the input a marine-mammal
  assessment needs: feed it to
  [`weighted_exposure`](/phonometry/reference/api/underwater/marine-mammal-weighting/#weighted_exposure) to
  obtain the weighted cumulative SEL of a piling campaign and its margin
  against the regulatory injury and TTS criteria.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## cumulative_sel

```python
cumulative_sel(single_sels: NDArray[np.float64] | list[float]) -> float
```

Cumulative sound exposure level over N strikes (ISO 18406 Formulae 8-9).

`SEL_cum = 10·lg(Σₙ 10^(SELₙ/10))` -- the energy sum of the per-strike
single-strike SELs.

**Parameters**

| Name | Description |
| :--- | :--- |
| `single_sels` | Per-strike single-strike SELs, in dB re 1 µPa²·s. |

**Returns:** Cumulative SEL, in dB re 1 µPa²·s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the sequence is empty or non-finite. |

## cumulative_sel_identical

```python
cumulative_sel_identical(sel_ss: float, n_strikes: int) -> float
```

Cumulative SEL of `n_strikes` identical strikes: `SEL_ss + 10·lg(N)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `sel_ss` | Single-strike SEL, in dB re 1 µPa²·s. |
| `n_strikes` | Number of (identical) strikes, `N ≥ 1`. |

**Returns:** Cumulative SEL, in dB re 1 µPa²·s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `n_strikes` is not a whole number `≥ 1`. |

## pile_strike_metrics

```python
pile_strike_metrics(
    pressure: NDArray[np.float64] | list[float],
    fs: float,
) -> PileStrikeResult
```

Full per-strike pile-driving metrics (ISO 18406).

Bundles the single-strike SEL, the peak sound pressure level, the SPL/Leq
and the 90 %-energy pulse duration of one recorded hammer strike.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure` | Sound-pressure time series of one strike (1-D), in Pa. |
| `fs` | Sample rate, in Hz. |

**Returns:** A [`PileStrikeResult`](/phonometry/reference/api/underwater/pile-driving-noise/#pilestrikeresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## PileStrikeResult

```python
PileStrikeResult(
    single_strike_sel: float,
    peak_spl: float,
    spl: float,
    pulse_duration: float,
    pressure: NDArray[np.float64],
    fs: float,
)
```

Per-strike pile-driving metrics (ISO 18406).

**Attributes**

| Name | Description |
| :--- | :--- |
| `single_strike_sel` | Single-strike SEL, in dB re 1 µPa²·s. |
| `peak_spl` | Zero-to-peak sound pressure level, in dB re 1 µPa. |
| `spl` | Sound pressure level (Leq over the record), in dB re 1 µPa. |
| `pulse_duration` | 90 %-energy pulse duration, in s. |
| `pressure` | The strike pressure waveform, in Pa. |
| `fs` | Sample rate, in Hz. |

### PileStrikeResult.plot()

```python
PileStrikeResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the strike waveform and its cumulative energy.

## single_strike_sel

```python
single_strike_sel(
    pressure: NDArray[np.float64] | list[float],
    fs: float,
) -> float
```

Single-strike sound exposure level `SEL_ss` (ISO 18406 Formulae 3-4).

The sound exposure level of one hammer-strike pulse, integrated over the
pulse, in dB re 1 µPa²·s.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure` | Sound-pressure time series of one strike (1-D), in Pa. |
| `fs` | Sample rate, in Hz. |

**Returns:** Single-strike SEL, in dB re 1 µPa²·s.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## strike_sel_spectrum

```python
strike_sel_spectrum(
    pressure: NDArray[np.float64] | list[float],
    fs: float,
    *,
    fraction: int = 3,
    limits: tuple[float, float] = (10.0, 20000.0),
) -> StrikeSelSpectrum
```

Band-resolved single-strike sound exposure level.

The sound exposure `E = ∫p² dt` is split between fractional-octave bands
by integrating the discrete power spectrum over each band (Parseval), so
the energy sum of the returned band levels reproduces the broadband
[`single_strike_sel`](/phonometry/reference/api/underwater/pile-driving-noise/#single_strike_sel) of the same record to within the energy that
falls outside `limits`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure` | Sound-pressure time series of one strike (1-D), in Pa. |
| `fs` | Sample rate, in Hz. |
| `fraction` | Bandwidth fraction: 1 (octave) or 3 (one-third octave). |
| `limits` | Lower and upper band-centre limits, in Hz. |

**Returns:** A [`StrikeSelSpectrum`](/phonometry/reference/api/underwater/pile-driving-noise/#strikeselspectrum).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## StrikeSelSpectrum

```python
StrikeSelSpectrum(
    frequencies: NDArray[np.float64],
    band_sel: NDArray[np.float64],
    total_sel: float,
    broadband_sel: float,
    fraction: int,
    fs: float,
)
```

Single-strike sound exposure level resolved into fractional-octave bands.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Nominal band centre frequencies, in Hz. |
| `band_sel` | Per-band single-strike SEL, in dB re 1 µPa²·s. |
| `total_sel` | Energy sum of `band_sel` over the covered bands, in dB re 1 µPa²·s. |
| `broadband_sel` | The broadband single-strike SEL of the whole record, in dB re 1 µPa²·s (equal to `total_sel` when the bands span the signal's whole occupied spectrum). |
| `fraction` | Bandwidth fraction (1 for octaves, 3 for one-third octaves). |
| `fs` | Sample rate, in Hz. |

### StrikeSelSpectrum.plot()

```python
StrikeSelSpectrum.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band single-strike SEL.
