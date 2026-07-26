---
title: "metrology.envelope"
description: "Envelope and instantaneous phase via the Hilbert transform."
sidebar:
  label: "envelope"
---

Envelope and instantaneous phase via the Hilbert transform.

Signal-envelope analysis following Bendat & Piersol, *Random Data:
Analysis and Measurement Procedures* (4th ed., 2010), Chapter 13. The
analytic signal `z(t) = x(t) + j·x̃(t)` (Eq. 13.15, with `x̃` the
Hilbert transform of `x`) yields

* the **envelope** `A(t) = [x²(t) + x̃²(t)]^½` (Eq. 13.17),
* the **instantaneous phase** `θ(t) = arctan[x̃(t)/x(t)]`, unwrapped
  (Eq. 13.18), and
* the **instantaneous frequency** `f(t) = (1/2π)·dθ/dt` (Eq. 13.19).

The analytic signal is computed the way the book recommends
(Section 13.1.1): the one-sided spectrum construction
`Z(f) = 2·X(f)` for `f > 0`, `X(0)` at DC and `0` for `f < 0`
(Eq. 13.25) - which is exactly what `scipy.signal.hilbert`
implements, and the same construction the ECMA-418-2 psychoacoustic chain
of `phonometry.psychoacoustics` applies per auditory band (its
Formulae 65/119 take `|hilbert|` and subsample by 32; the standard can
subsample directly because each band is narrow). Closed-form pairs from
Table 13.1 (`cos → sin`, an AM envelope recovered exactly) anchor the
tests.

The envelope of a band-limited signal is itself low-frequency, so the
result offers optional **decimation**: an anti-aliased zero-phase FIR
decimator for general records, or plain subsampling (`antialias=False`)
matching the ECMA-internal convention when the input is already
narrowband.

The **envelope spectrum** ([`envelope_spectrum`](/phonometry/reference/api/correlation/envelope/#envelope_spectrum)) transforms the
detected envelope itself: Section 13.3 of the book runs a band-pass
filter and a square-law envelope detector into a DC remover before
correlating (Figure 13.11), because the spectral content of the envelope
- not of the signal - is where amplitude modulations show as discrete
lines. The optional `band` argument reproduces the figure's band-pass
front end (the classical bearing-envelope chain: isolate the resonance
band, then envelope it). An AM tone with modulation frequency `f_m`
(on an analysis bin) and depth `m` puts a line of closed-form
amplitude at exactly `f_m`, the anchor the tests pin; off-bin
modulation lines read low by the taper's scalloping loss.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## envelope

```python
envelope(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    decimation_factor: int = 1,
    antialias: bool = True,
) -> EnvelopeResult
```

Envelope, instantaneous phase and frequency via Hilbert transform.

Builds the analytic signal by the one-sided spectrum construction of
Bendat & Piersol Eq. 13.25 (`scipy.signal.hilbert`) and returns the
envelope `|z(t)|`, the unwrapped instantaneous phase and the
instantaneous frequency (Eqs. 13.17-13.19). For an amplitude-modulated
carrier `u(t)·cos(2πf0t)` with `u` low-frequency and non-negative
the envelope recovers `u(t)` exactly in the ideal continuous case
(Eq. 13.27); a discrete record shows small edge effects at the record
boundaries.

The optional decimation reduces the output rate by an integer factor:
the envelope is anti-alias filtered with a zero-phase FIR decimator
by default, or plainly subsampled with `antialias=False` - the
convention the ECMA-418-2 loudness/roughness chain applies internally
after its auditory bandpass, appropriate when the input is already
narrowband. The phase and instantaneous frequency, smooth after
unwrapping and differentiated at full rate, are subsampled onto the
same time axis.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `decimation_factor` | Integer output decimation (default 1: off). |
| `antialias` | Anti-alias filter the decimated envelope (default `True`). |

**Returns:** An [`EnvelopeResult`](/phonometry/reference/api/correlation/envelope/#enveloperesult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## envelope_spectrum

```python
envelope_spectrum(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    kind: str = 'magnitude',
    window: str = 'hann',
    nfft: int | None = None,
    remove_dc: bool = True,
    band: tuple[float, float] | None = None,
) -> EnvelopeSpectrumResult
```

Amplitude spectrum of the envelope: where modulations become lines.

Follows the structure of Bendat & Piersol Section 13.3 (Figure 13.11):
a band-pass filter (optional here), an envelope detector, a DC
remover, and a spectral view of what is left. The detector is the
Hilbert envelope `A(t) = |z(t)|` (`kind="magnitude"`, the
practical default) or the book's square-law detector
`A^2(t) = x^2 + x_hat^2` (`kind="squared"`); its mean is removed
(kept in [`EnvelopeSpectrumResult.mean_level`](/phonometry/reference/api/correlation/envelope/#envelopespectrumresult)) and the remainder
is tapered and transformed once, scaled by the taper's coherent gain
so a sinusoidal modulation whose frequency falls on an analysis bin
reads out as a line at its exact amplitude. An off-bin modulation
frequency reads low by the taper's scalloping loss -- up to about
1.4 dB (~15 %) for the default Hann midway between bins -- like any
single-record amplitude spectrum.

Closed forms for an AM tone `A0 (1 + m cos(2 pi f_m t)) cos(2 pi f_c t)`
with `0 <= m < 1` and `f_m` on an analysis bin:

* `kind="magnitude"`: a line of amplitude `A0 m` at `f_m`;
  mean level `A0`.
* `kind="squared"`: lines `2 A0^2 m` at `f_m` and
  `A0^2 m^2 / 2` at `2 f_m`; mean level `A0^2 (1 + m^2/2)`.

Amplitude modulation of rotating machinery (bearing and gear defect
frequencies), mains hum and wind-turbine amplitude modulation appear
the same way: lines at the modulation frequency and its harmonics,
separated from the carrier's own spectrum. For the classical bearing
chain -- isolate a structural-resonance band excited by the defect
impacts, then envelope it -- pass the resonance band as `band`: the
record is band-pass filtered (zero-phase, so the modulation phase is
untouched) before the detector, the Figure 13.11 front end.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `kind` | `"magnitude"` (default) or `"squared"`. |
| `window` | Taper (any scipy window name; default Hann). The amplitude is corrected for the taper's coherent gain. |
| `nfft` | FFT length, at least `x.size` (default: the record length). |
| `remove_dc` | Remove the envelope mean before the transform (default `True`, the Figure 13.11 DC remover); the mean is reported either way. |
| `band` | Optional `(low, high)` band-pass edges, in Hz (`0 < low < high < fs/2`), applied to the record before envelope detection as a zero-phase 4th-order Butterworth (`scipy.signal.sosfiltfilt`, giving an 8th-order magnitude roll-off). Default `None`: detect on the record as given. |

**Returns:** An [`EnvelopeSpectrumResult`](/phonometry/reference/api/correlation/envelope/#envelopespectrumresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## EnvelopeResult

```python
EnvelopeResult(
    times: NDArray[np.float64],
    envelope: NDArray[np.float64],
    phase: NDArray[np.float64],
    instantaneous_frequency: NDArray[np.float64],
    fs: float,
    signal: NDArray[np.float64],
    signal_fs: float,
    decimation_factor: int,
    antialias: bool,
)
```

Envelope and instantaneous phase of a signal (B&P Chapter 13).

All output arrays share the (possibly decimated) time axis
`times`; the original record is kept at full rate for plotting.

**Attributes**

| Name | Description |
| :--- | :--- |
| `times` | Time axis of the outputs, in seconds. |
| `envelope` | Envelope `A(t) = \|z(t)\|` (Eq. 13.17). |
| `phase` | Unwrapped instantaneous phase `θ(t)`, in radians (Eq. 13.18). |
| `instantaneous_frequency` | `f(t) = (1/2π)·dθ/dt`, in Hz (Eq. 13.19), differentiated at full rate before any decimation. |
| `fs` | Sample rate of the outputs, in Hz (`signal_fs` divided by `decimation_factor`). |
| `signal` | The analysed record, at full rate. |
| `signal_fs` | Sample rate of `signal`, in Hz. |
| `decimation_factor` | Integer decimation applied to the outputs (1: none). |
| `antialias` | Whether the decimation was anti-alias filtered. |

### EnvelopeResult.plot()

```python
EnvelopeResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the signal with its envelope and the instantaneous frequency.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## EnvelopeSpectrumResult

```python
EnvelopeSpectrumResult(
    frequencies: NDArray[np.float64],
    amplitude: NDArray[np.float64],
    mean_level: float,
    kind: str,
    times: NDArray[np.float64],
    envelope: NDArray[np.float64],
    window: str,
    remove_dc: bool,
    fs: float,
    nfft: int,
    band: tuple[float, float] | None = None,
)
```

Amplitude spectrum of a signal's envelope (B&P Section 13.3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency axis of the spectrum, in Hz. |
| `amplitude` | One-sided amplitude spectrum of the (mean-removed) envelope: the height of a discrete modulation line in the units of the envelope itself, exact when the modulation frequency falls on an analysis bin (off-bin lines read low by the taper's scalloping loss; see [`envelope_spectrum`](/phonometry/reference/api/correlation/envelope/#envelope_spectrum)). The zero-frequency bin is not doubled. |
| `mean_level` | Mean of the detected envelope (the DC the remover of Figure 13.11 takes out): the carrier amplitude for `kind="magnitude"`, its mean square for `kind="squared"`. |
| `kind` | `"magnitude"` (Hilbert envelope `A(t)`) or `"squared"` (the book's square-law detector, `A^2(t)`). |
| `times` | Time axis of [`envelope`](/phonometry/reference/api/correlation/envelope/#envelope), in seconds. |
| `envelope` | The detector output that was transformed, at full rate (before mean removal and tapering). |
| `window` | Taper name applied before the transform. |
| `remove_dc` | Whether the envelope mean was removed first. |
| `fs` | Sample rate of the analysed record, in Hz. |
| `nfft` | FFT length used. |
| `band` | `(low, high)` edges of the zero-phase band-pass pre-filter applied before envelope detection, in Hz, or `None` (no pre-filter). |

### EnvelopeSpectrumResult.plot()

```python
EnvelopeSpectrumResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the detected envelope and its amplitude spectrum.

With `ax` given, only the spectrum panel is drawn on it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
