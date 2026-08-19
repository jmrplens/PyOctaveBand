---
title: "room.impulse_response"
description: "Impulse-response acquisition per BS EN ISO 18233:2006."
sidebar:
  label: "impulse_response"
---

Impulse-response acquisition per BS EN ISO 18233:2006.

This module provides the deterministic-excitation front end for the "new
measurement methods" of ISO 18233: generate an excitation signal, play it
through the system under test, and recover the broadband impulse response
(IR) by deconvolution. Later processing (fractional-octave filtering,
Schroeder backward integration, reverberation time) consumes this IR.

Two excitation families are implemented:

* **Swept-sine (Annex B)** -- an exponential sine sweep (ESS). The frequency
  rises exponentially with time, which mimics a pink-noise source
  (ISO 18233:2006, B.3.2) and is the recommended broadband excitation. The
  IR is recovered by linear (zero-padded, non-circular) spectral
  deconvolution (B.5, Figure B.3):
  $H = Y \overline{X} / (|X|^2 + \text{reg})$. A
  Farina inverse-filter variant (`method="farina"`) convolves the
  recording with the time-reversed, amplitude-compensated sweep (B.5,
  Figure B.2). Because a low-to-high sweep places distortion products at
  negative arrival times, harmonic distortion is separated from the linear
  IR and discarded by keeping only the causal part (B.5). To *analyse*
  those discarded products instead (per-order harmonic responses and
  THD(f) from the same recording), see
  [`phonometry.swept_sine_distortion`](/phonometry/reference/api/electroacoustics/swept-sine/#swept_sine_distortion).

* **Maximum-length sequence (Annex A)** -- an order-`N` binary sequence of
  length $2^N - 1$ generated with a linear-feedback shift register
  (LFSR). Its circular autocorrelation is a near-perfect delta (A.1), so the
  IR of a periodically excited linear system is recovered by circular
  cross-correlation of the recorded period with the sequence
  (equivalent to the Hadamard-transform recovery of A.1).

Two further excitations from the transfer-function measurement literature
complete the family:

* **Complementary Golay pair** -- two binary sequences of length
  $L = 2^n$ whose periodic autocorrelations sum to an *exact* delta of
  height $2L$ (Golay 1961; Havelock, Kuwano & Vorlaender (eds.), Handbook
  of Signal Processing in Acoustics, Springer 2008, Part I Ch. 6 by
  N. Xiang, Eq. (2)). Exciting the system with each code in turn and
  summing the two circular cross-correlations recovers the IR with zero
  correlation noise: the deterministic residue of each single-code
  correlation cancels identically, so only uncorrelated background noise
  remains (Xiang Eq. (4)). See [`golay_pair`](/phonometry/reference/api/rooms/impulse-response/#golay_pair) and
  [`golay_impulse_response`](/phonometry/reference/api/rooms/impulse-response/#golay_impulse_response).

* **Sweep with an arbitrary magnitude spectrum** -- a swept sine synthesized
  in the frequency domain by shaping its group delay so the dwell time at
  each frequency is proportional to the desired spectral power
  (Mueller & Massarani, "Transfer-Function Measurement with Sweeps", JAES
  49(6), 2001, Secs. 4.2-4.3). The sweep keeps the near-ideal crest factor
  of a swept sine while following any prescribed emphasis (pink,
  noise-floor-matched, loudspeaker-equalizing, ...). See
  [`shaped_sweep_signal`](/phonometry/reference/api/rooms/impulse-response/#shaped_sweep_signal); the recording is deconvolved with the
  ordinary spectral method of [`impulse_response`](/phonometry/reference/api/rooms/impulse-response/#impulse_response), or post-equalized
  with [`phonometry.regularized_inverse_filter`](/phonometry/reference/api/signals/inversion/#regularized_inverse_filter).

The recovered IR is broadband; ISO 18233 6.3.2 requires subsequent
fractional-octave-band weighting (IEC 61260) before computing levels or
decay curves -- that step belongs to downstream room-acoustics modules.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## golay_impulse_response

```python
golay_impulse_response(
    recorded_a: list[float] | np.ndarray,
    recorded_b: list[float] | np.ndarray,
    pair: tuple[np.ndarray, np.ndarray],
    *,
    length: int | None = None,
    fs: int | None = None,
) -> ImpulseResponseResult
```

Recover an impulse response from a complementary Golay-pair excitation.

Each code of the pair is emitted periodically (as with an MLS, record in
the steady state: at least one settling period before acquisition); the
recorded periods of each code are averaged and the IR is the sum of the
two circular cross-correlations, normalised by $2L$ (Havelock 2008,
Part I Ch. 6 (N. Xiang), Eq. (4) and the measurement procedure of
Fig. 2):

$$
h = \operatorname{IFFT}\!\left[ \overline{A}\, \operatorname{FFT}(y_a) + \overline{B}\, \operatorname{FFT}(y_b) \right] / (2L)
$$

Because the pair's autocorrelations are *exactly* complementary
(Xiang Eq. (2)), the recovery has no correlation noise: for a noiseless
linear time-invariant system the IR is exact to machine precision,
whereas an MLS leaves a small deterministic residue. Uncorrelated
background noise is only attenuated by the averaging, and the price is
a doubled excitation time and two steady states, which makes the pair
more exposed to time variance than a single sweep (Xiang, Sec. 2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `recorded_a` | Recorded response to the periodic `a` code; its length must be a positive multiple of the code length `L`. |
| `recorded_b` | Recorded response to the periodic `b` code; its length must be a positive multiple of `L` (the period counts of the two recordings may differ). |
| `pair` | The complementary pair `(a, b)` from [`golay_pair`](/phonometry/reference/api/rooms/impulse-response/#golay_pair). |
| `length` | Number of IR samples to return. Defaults to `L`; longer requests are periodic extensions. |
| `fs` | Optional sample rate in Hz, stored on the result so that [`ImpulseResponseResult.plot`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresultplot) can label a time axis in seconds (the recovery itself is sample-rate agnostic). Default `None`. |

**Returns:** An [`ImpulseResponseResult`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresult) (`method="golay"`). It behaves like the raw IR array for every downstream consumer and adds [`ImpulseResponseResult.plot`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresultplot).

:::note
As with any periodic (circular) recovery, a system IR longer than
one code period aliases back into the record; an
[`ImpulseResponseWarning`](/phonometry/reference/api/rooms/impulse-response/#impulseresponsewarning) flags undecayed energy at the end
of the period (see the note in [`mls_impulse_response`](/phonometry/reference/api/rooms/impulse-response/#mls_impulse_response) about
the heuristic's noise-floor false positives).
:::

## golay_pair

```python
golay_pair(order: int) -> tuple[np.ndarray, np.ndarray]
```

Generate a complementary Golay pair of length `2**order`.

Built with the append recursion of Golay (1961): starting from
`a1 = (+1, +1)`, `b1 = (+1, -1)`, each step appends `b` to `a`
and `-b` to `a` (Havelock, Handbook of Signal Processing in
Acoustics, Springer 2008, Part I Ch. 6 (N. Xiang), Eq. (1)). The pair is
*complementary*: the sum of the two periodic autocorrelations is exactly
$2L$ at zero lag and exactly zero everywhere else (Xiang Eq. (2)) --
an algebraic identity, not an approximation, unlike the near-delta
autocorrelation of an MLS.

**Parameters**

| Name | Description |
| :--- | :--- |
| `order` | Number of recursion steps `n` (1 to 22). Each code has $L = 2^{\text{order}}$ samples. |

**Returns:** The pair `(a, b)` as bipolar float arrays (values `+1/-1`).

## impulse_response

```python
impulse_response(
    recorded: list[float] | np.ndarray,
    reference: list[float] | np.ndarray,
    fs: int,
    *,
    method: Literal['farina'],
    f_range: tuple[float, float],
    regularization: float = ...,
    length: int | None = ...,
    return_full: bool = ...,
) -> ImpulseResponseResult

impulse_response(
    recorded: list[float] | np.ndarray,
    reference: list[float] | np.ndarray,
    fs: int,
    *,
    method: str = ...,
    f_range: tuple[float, float] | None = ...,
    regularization: float = ...,
    length: int | None = ...,
    return_full: bool = ...,
) -> ImpulseResponseResult
```

Recover the broadband impulse response by sweep deconvolution.

Implements the linear (non-circular) deconvolution of ISO 18233:2006,
B.5. Both signals are zero-padded to `len(recorded)+len(reference)-1`
to avoid circular convolution (B.5). The causal IR occupies the start of
the result; distortion products from a low-to-high sweep fall at negative
arrival times (the wrapped tail) and are discarded by returning only the
causal part (B.5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `recorded` | Recorded system response to the sweep. |
| `reference` | The emitted sweep (excitation signal). |
| `fs` | Sampling frequency in Hz (kept for API symmetry; the deconvolution itself is sample-rate agnostic). |
| `method` | `"spectral"` for spectral division $H = Y \overline{X} / (\lvert X \rvert^2 + \text{reg})$ (Figure B.3, default) or `"farina"` for convolution with the analytic inverse filter (Figure B.2). The Farina method requires `f_range` and the **exact-length, unpadded** excitation sweep as `reference` (it rebuilds the inverse filter from `reference.size/fs` as the sweep duration); a reference zero-padded to the recording length - the correct input for the spectral method - is rejected with a `ValueError` because it would silently produce a wrong inverse filter. It also assumes the reference sweep was generated with the default `amplitude`/`fade` of [`sweep_signal`](/phonometry/reference/api/rooms/impulse-response/#sweep_signal); a non-unit amplitude or custom fade yields a scaled IR, so use the spectral method in that case. |
| `f_range` | `(f1, f2)` of the sweep, required for `method="farina"` to rebuild the inverse filter; ignored for the spectral method. |
| `regularization` | Tikhonov term added to the denominator, expressed as a fraction of the peak spectral energy $\max(\lvert X \rvert^2)$ (spectral method only). Guards against amplifying noise where the sweep has little energy, e.g. outside its frequency range (B.5). Default 1e-6. |
| `length` | Number of samples of the causal IR to return. Defaults to `len(recorded)`. Ignored when `return_full` is True. |
| `return_full` | If True, return the full deconvolution sequence (causal IR at index 0, negative-time distortion products in the tail) instead of the trimmed causal IR. Default False. |

**Returns:** An [`ImpulseResponseResult`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresult) wrapping the recovered impulse response. It behaves like the raw IR array (`np.asarray(result)`, indexing, `.size`) for every downstream consumer and adds [`ImpulseResponseResult.plot`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresultplot).

## ImpulseResponseResult

```python
ImpulseResponseResult(ir: np.ndarray, fs: int | None, method: str)
```

Recovered broadband impulse response with its acquisition metadata.

Returned by [`impulse_response`](/phonometry/reference/api/rooms/impulse-response/#impulse_response) and [`mls_impulse_response`](/phonometry/reference/api/rooms/impulse-response/#mls_impulse_response).
The impulse response samples live in `ir`; `fs` is the sample rate in
Hz (or `None` when unknown, e.g. an MLS recovery called without one) and
`method` records how the IR was obtained (`"spectral"`, `"farina"`
or `"mls"`).

The object is a drop-in replacement for the raw array it used to be: it
implements `__array__`, so `np.asarray(result)` yields the IR and
the result can be passed straight to array consumers such as
[`phonometry.room_parameters`](/phonometry/reference/api/rooms/acoustics/#room_parameters), [`phonometry.decay_curve`](/phonometry/reference/api/rooms/acoustics/#decay_curve) and
[`phonometry.sti_from_impulse_response`](/phonometry/reference/api/speech/sti/#sti_from_impulse_response). Indexing, `len(result)`
and the `size`/`ndim`/`shape`/`dtype` attributes forward to `ir`.

### ImpulseResponseResult.dtype

*property*

### ImpulseResponseResult.ndim

*property*

### ImpulseResponseResult.plot()

```python
ImpulseResponseResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the impulse response: waveform and log-magnitude decay.

Draws the (normalised) time-domain waveform and, below it, the
log-magnitude envelope in dB with a Schroeder energy-decay overlay.
With `ax` given, only the decay panel is drawn on it. Requires
matplotlib (`pip install phonometry[plot]`); returns the
`Axes` (or an array of two axes).

### ImpulseResponseResult.shape

*property*

### ImpulseResponseResult.size

*property*

Number of samples in the impulse response.

## ImpulseResponseWarning

Warns about suspect recovered impulse responses (e.g. MLS aliasing).

## inverse_filter

```python
inverse_filter(
    fs: int,
    f1: float,
    f2: float,
    seconds: float,
    *,
    amplitude: float = 1.0,
    fade: float = 0.01,
) -> np.ndarray
```

Build the Farina inverse filter for an exponential sine sweep.

The inverse filter is the time-reversed sweep multiplied by an amplitude
envelope that rises by 6 dB/octave (`prop. to the instantaneous
frequency`), which whitens the ESS's pink (-3 dB/octave) spectrum so
that convolving the sweep with its inverse yields an impulse
(ISO 18233:2006, B.5, Figure B.2; Farina 2000, Bibliography [14]). The
filter is scaled to unit in-band magnitude: it is normalised by the
median in-band magnitude of the compressed pulse `sweep * inverse` so
that the deconvolution reproduces a system's true in-band level, matching
the spectral-division convention (rather than a unit pulse peak).

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sampling frequency in Hz. |
| `f1` | Sweep start frequency in Hz (same value used for the sweep). |
| `f2` | Sweep stop frequency in Hz. |
| `seconds` | Sweep duration in seconds. |
| `amplitude` | Peak amplitude used for the source sweep. Default 1.0. |
| `fade` | Fade fraction used for the source sweep. Default 0.01. |

**Returns:** The inverse-filter samples (same length as the sweep).

## mls_impulse_response

```python
mls_impulse_response(
    recorded: list[float] | np.ndarray,
    mls: list[float] | np.ndarray,
    *,
    length: int | None = None,
    fs: int | None = None,
) -> ImpulseResponseResult
```

Recover an impulse response from a periodic MLS excitation.

The recording must span an integer number of MLS periods; the periods
are averaged (raising the effective signal-to-noise ratio by 3 dB per
doubling, ISO 18233:2006, 6.3.6) and the IR is obtained by circular
cross-correlation of the averaged period with the sequence, normalised by
`2**N` (A.1). Because the sequence is periodic, the recovery is a
circular deconvolution: a system IR longer than one period aliases back
into the record (A.1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `recorded` | Recorded response, length a multiple of `2**N - 1`. |
| `mls` | The excitation sequence returned by [`mls_signal`](/phonometry/reference/api/rooms/impulse-response/#mls_signal). |
| `length` | Number of IR samples to return. Defaults to the sequence length `2**N - 1`. |
| `fs` | Optional sample rate in Hz, stored on the result so that [`ImpulseResponseResult.plot`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresultplot) can label a time axis in seconds (the recovery itself is sample-rate agnostic). Default `None`. |

**Returns:** An [`ImpulseResponseResult`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresult) wrapping the recovered impulse response. It behaves like the raw IR array for every downstream consumer and adds [`ImpulseResponseResult.plot`](/phonometry/reference/api/rooms/impulse-response/#impulseresponseresultplot).

:::note
An [`ImpulseResponseWarning`](/phonometry/reference/api/rooms/impulse-response/#impulseresponsewarning) is emitted when the recovered IR retains significant
energy at the end of the period (a circular-aliasing symptom). The
tail-RMS heuristic is advisory: a high ambient noise floor in the
recording raises the tail RMS on its own and can trigger a
false positive even when the IR fits within one period, so treat the
warning as a prompt to check the noise floor and MLS order rather than
a definitive aliasing diagnosis.
:::

## mls_signal

```python
mls_signal(order: int) -> np.ndarray
```

Generate a maximum-length sequence (MLS) of the given order.

A Fibonacci LFSR with primitive-polynomial feedback taps produces a
binary sequence of length `2**order - 1` whose circular
autocorrelation is a near-perfect periodic delta
(ISO 18233:2006, A.1). The binary values are mapped to `+1`/`-1`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `order` | Register length `N` (2 to 20). The sequence length is `2**order - 1`. |

**Returns:** The bipolar MLS samples (values in `{-1.0, +1.0}`).

## plot_excitation

```python
plot_excitation(
    signal: np.ndarray | Any,
    fs: int,
    *,
    kind: str = 'sweep',
    ax: Axes | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot an ISO 18233 excitation signal (sweep or MLS).

A documented helper for the raw arrays returned by
[`sweep_signal`](/phonometry/reference/api/rooms/impulse-response/#sweep_signal) and [`mls_signal`](/phonometry/reference/api/rooms/impulse-response/#mls_signal), which
stay plain `numpy.ndarray` (they are meant for playback). For a
swept sine the waveform and its spectrogram are drawn; for an MLS the first
samples of the bipolar sequence and its (flat) magnitude spectrum.

**Parameters**

| Name | Description |
| :--- | :--- |
| `signal` | The excitation samples (1D array-like). |
| `fs` | Sample rate in Hz (for the time and frequency axes). |
| `kind` | `"sweep"` (default) or `"mls"`. |
| `ax` | Existing axes for the top (time-domain) panel, or `None` for a fresh two-panel figure. |
| `kwargs` | Forwarded to the time-domain `plot` call. |

**Returns:** The time-domain axes (`ax` given) or the array of two axes.

## shaped_sweep_signal

```python
shaped_sweep_signal(
    fs: int,
    f1: float,
    f2: float,
    seconds: float,
    *,
    target: str | tuple[np.ndarray, np.ndarray] = 'pink',
    amplitude: float = 1.0,
    start_delay: float | None = None,
    fade: float = 0.01,
) -> ShapedSweepResult
```

Synthesize a sweep with an arbitrary target magnitude spectrum.

Implements the frequency-domain sweep construction of
Mueller & Massarani ("Transfer-Function Measurement with Sweeps", JAES
49(6), 2001, Secs. 4.2-4.3): the magnitude of the synthesis spectrum is
set to the band-limited target, and the group delay grows in proportion
to the target's spectral power (Eqs. (11)-(12)):

$$
\tau_\mathrm{G}(f) = \tau_\mathrm{G}(f - df) + C\, |H(f)|^2
$$

$$
C = \frac{\tau_\mathrm{G}(f_{\text{end}}) - \tau_\mathrm{G}(f_{\text{start}})} {\sum |H|^2}
$$

so the sweep dwells on each frequency for a time proportional to the
energy it must radiate there and its temporal envelope stays nearly
constant -- the crest factor stays close to a swept sine's ideal
3.01 dB regardless of the spectral shape (Sec. 4.3). The phase is the
integral of the group delay, corrected to land on a real spectrum at
Nyquist (Eq. (10)), and the sweep is obtained by inverse FFT over a
block at least double the sweep length so the pre-ringing of the
band-limited spectrum cannot fold onto the sweep's tail (Sec. 4.2).

Deconvolve the recording with [`impulse_response`](/phonometry/reference/api/rooms/impulse-response/#impulse_response)
(`method="spectral"`), passing `np.asarray(result)` zero-padded as
the reference, exactly as with [`sweep_signal`](/phonometry/reference/api/rooms/impulse-response/#sweep_signal); the sweep's
coloration divides out, so the target emphasis only re-weights the
measurement's noise floor (that is its purpose: SNR shaping).

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sampling frequency in Hz. |
| `f1` | Start frequency of the sweep band in Hz. Must be > 0. The magnitude rolls off over 1/6 octave *below* `f1` (clipped at the first FFT bin), so the full target level holds across `[f1, f2]`. |
| `f2` | Stop frequency in Hz. Must satisfy $f_1 < f_2 \le f_\mathrm{s}/2$; keep some margin below Nyquist so the upper roll-off has room. |
| `seconds` | Sweep duration $\tau_\mathrm{G}(f_2) - \tau_\mathrm{G}(f_1)$ in seconds. The returned signal is slightly longer (lead-in plus tail margin, see `start_delay`). |
| `target` | The magnitude shape: `"pink"` (default; 3 dB per octave falling, the classical room-measurement emphasis), `"white"` (flat), or a `(frequencies_hz, magnitude_db)` pair of arrays interpolated in dB over log-frequency (only the shape matters; any overall offset is normalised away). |
| `amplitude` | Peak amplitude of the returned sweep. Default 1.0. |
| `start_delay` | Group delay assigned to `f1`, in seconds; the same margin is left after `tau_G(f2)`, so the signal lasts `seconds + 2*start_delay`. The sweep spreads slightly beyond its nominal start (Sec. 4.2: the group delay of the lowest bin "should not be set to zero"), so the default `0.05*seconds` gives the first half-wave room to evolve. |
| `fade` | Half-Hann fade-in/out length as a fraction of the returned signal, applied to pin the ends to zero (Sec. 4.2). Default 0.01; 0.0 disables. |

**Returns:** A [`ShapedSweepResult`](/phonometry/reference/api/rooms/impulse-response/#shapedsweepresult) wrapping the sweep samples and the synthesis metadata (grid, imposed magnitude, group delay, crest factor).

## ShapedSweepResult

```python
ShapedSweepResult(
    signal: np.ndarray,
    fs: float,
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    group_delay: np.ndarray,
    f_range: tuple[float, float],
    crest_factor_db: float,
)
```

A sweep synthesized to follow an arbitrary target magnitude spectrum.

Returned by [`shaped_sweep_signal`](/phonometry/reference/api/rooms/impulse-response/#shaped_sweep_signal). The playable samples live in
`signal`; the object implements `__array__`, so it can be passed
straight to a sound-card writer or as the `reference` of
[`impulse_response`](/phonometry/reference/api/rooms/impulse-response/#impulse_response) (spectral method). The synthesis metadata --
the frequency grid, the band-limited magnitude actually imposed on the
spectrum and the group delay that encodes the sweep's time-frequency
trajectory (Mueller & Massarani 2001, Secs. 4.2-4.3) -- travels with
the result, together with the achieved crest factor.

**Attributes**

| Name | Description |
| :--- | :--- |
| `signal` | The sweep samples (peak `amplitude`). |
| `fs` | Sample rate, in Hz. |
| `frequencies` | Frequency grid of the synthesis FFT, in Hz. |
| `magnitude` | Band-limited magnitude imposed on the synthesis spectrum, normalised to a peak of 1 (linear). |
| `group_delay` | Synthesized group delay `tau_G(f)` on `frequencies`, in seconds: the time at which each frequency is swept through. |
| `f_range` | `(f1, f2)` band covered by the sweep, in Hz. |
| `crest_factor_db` | Peak-to-RMS ratio over the sweep's central (constant-envelope) interval, in dB. A time-domain swept sine has the ideal 3.01 dB; the frequency-domain synthesis stays close to it (Mueller & Massarani 2001, Sec. 4.3: normally below 4 dB). Their figure assumes the slight magnitude smoothing the paper applies before synthesis, which this module does not implement: typical shaped sweeps here measure about 4.2-4.3 dB. |

### ShapedSweepResult.plot()

```python
ShapedSweepResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the sweep waveform and its spectrum against the target.

Two stacked panels: the time-domain waveform, and the sweep's Welch
magnitude spectrum overlaid on the synthesis target (both in dB re
their in-band maximum). With `ax` given, only the spectrum panel
is drawn on it. Requires matplotlib
(`pip install phonometry[plot]`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

### ShapedSweepResult.size

*property*

Number of samples in the sweep.

## sweep_signal

```python
sweep_signal(
    fs: int,
    f1: float,
    f2: float,
    seconds: float,
    *,
    amplitude: float = 1.0,
    fade: float = 0.01,
) -> np.ndarray
```

Generate an exponential sine sweep (ESS) with exact analytic phase.

The instantaneous frequency rises exponentially from `f1` to `f2`,
$f(t) = f_1 (f_2/f_1)^{t/T}$, so the time spent per octave is
constant and the sweep mimics a pink-noise excitation
(ISO 18233:2006, B.3.2). The phase is the closed-form integral of
$2 \pi f(t)$ (Farina, AES 108th Conv., 2000; ISO 18233 Bibliography
[14]), avoiding numerical phase accumulation.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sampling frequency in Hz. |
| `f1` | Start frequency in Hz (at or below the lowest band edge to be measured, ISO 18233 B.3.1). Must be > 0. |
| `f2` | Stop frequency in Hz (at or above the highest band edge). Must satisfy $f_1 < f_2 \le f_\mathrm{s}/2$. |
| `seconds` | Sweep duration in seconds. Any duration may be used; a longer sweep raises the effective signal-to-noise ratio (B.2, B.6). |
| `amplitude` | Peak amplitude of the sweep. Default 1.0. |
| `fade` | Half-Hann fade-in/out length as a fraction of the sweep duration, applied to suppress start/stop transients (B.3.3). Default 0.01. Set to 0.0 to disable. Because the sweep frequency is logarithmic in time, the fades consume roughly `fade*log2(f2/f1)` octaves at each band edge (the fade-out lands on the highest frequencies): with the default 0.01 the top ~29 dB of the highest band is unusable, so choose `f1`/`f2` with margin beyond the analysis range (ISO 18233 B.3.1) rather than relying on a smaller fade. |

**Returns:** The sweep samples, length `round(seconds*fs)`.
