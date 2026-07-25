---
title: "metrology.cepstrum"
description: "Public API of phonometry.metrology.cepstrum (auto-generated)."
sidebar:
  label: "cepstrum"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Cepstral analysis: real/power/complex cepstrum, liftering and echo detection.

The **cepstrum** is the inverse Fourier transform of the logarithm of a
spectrum. Because the log turns the convolution `x = h * u` into the sum
`ln X = ln H + ln U` (Havelock, Kahle & Cocchi (eds.), *Handbook of Signal
Processing in Acoustics*, Springer 2008: Milner, Ch. 27, Eqs. (22)-(23)),
components that overlap hopelessly in the spectrum separate cleanly in the
cepstral domain -- the smooth spectral envelope collapses onto the low
**quefrencies** (the time-like axis of the cepstrum, in seconds) while
periodic spectral ripple from harmonics, reflections or echoes concentrates
at the quefrency of its period. Three variants are standard:

* the **power cepstrum**, the inverse transform of the log *power* spectrum
  `ln|X|^2` (Milner Fig. 21) -- real, even, phase-blind. (Convention
  note: Bogert, Healy & Tukey's original 1963 power cepstrum squares once
  more, `|IDFT(ln|X|^2)|^2`; this module follows Milner's linear
  `IDFT(ln|X|^2)` throughout.);
* the **real cepstrum**, the inverse transform of `ln|X|` -- exactly half
  the power cepstrum, and the quantity whose causal folding yields the
  minimum-phase reconstruction of [`phonometry.minimum_phase`](/phonometry/reference/api/spectra/phase/#minimum_phase)
  (Bendat & Piersol, *Random Data*, 4th ed., Sec. 13.1.4; Tohyama in
  Havelock Ch. 75 manipulates minimum-phase and all-pass components the
  same way);
* the **complex cepstrum**, the inverse transform of the full complex
  logarithm `ln|X| + j arg X` with the phase unwrapped (Neelamani in
  Havelock Ch. 87, Eq. (14)) -- invertible, hence the engine of
  homomorphic deconvolution.

**Echoes.** A single reflection `x(t) = s(t) + a s(t - t0)` multiplies the
spectrum by `1 + a e^{-j 2 pi f t0}`, whose complex logarithm expands (for
`|a| < 1`) into the exactly summable series

`ln(1 + a e^{-j theta}) = sum_{n>=1} (-1)^{n+1} (a^n / n) e^{-j n theta}`,

so the cepstrum carries a spike train at the *rahmonics* `n t0` with
amplitudes `a, -a^2/2, a^3/3, ...` (their sum is `ln(1 + a)`): a peak at
exactly the echo delay whose height reads out the reflection coefficient.
[`echo_detection`](/phonometry/reference/api/spectra/cepstrum/#echo_detection) automates that reading on the power cepstrum, where
the first rahmonic's height is `a` itself.

**Liftering** -- filtering in the quefrency domain (Milner Sec. 4.3) --
splits a log spectrum into its smooth envelope (low quefrencies kept by a
lowpass lifter) and its fine structure (highpass): the classical route to a
spectral envelope free of harmonic ripple, or to the ripple alone.

All estimators here operate on a single record with one FFT of length
`nfft` (even, at least the record length; the record is zero-padded up
to it). The discrete cepstrum is the inverse *DFT* of the log of a
*sampled* spectrum, so it is time-aliased when the log spectrum has
features sharper than the grid resolves; zero-padding `nfft` is the
remedy, exactly like the `oversample` padding of
[`phonometry.minimum_phase`](/phonometry/reference/api/spectra/phase/#minimum_phase), whose cepstral folding core
(`_fold_causal`) this module shares.

## cepstrum

```python
cepstrum(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    kind: str = 'power',
    nfft: int | None = None,
) -> CepstrumResult
```

Cepstrum of a record: power, real or complex.

* `"power"`: inverse DFT of `ln|X|^2` (Milner Fig. 21 in Havelock
  Ch. 27). Even, phase-blind; an echo of reflection coefficient `a`
  at delay `t0` shows rahmonics of amplitude
  `(-1)^{n+1} a^n / n` at quefrencies `n t0` -- height `a` at the
  delay itself.
* `"real"`: inverse DFT of `ln|X|` -- exactly half the power
  cepstrum. Folding it causally is the minimum-phase reconstruction
  (see [`phonometry.minimum_phase`](/phonometry/reference/api/spectra/phase/#minimum_phase), which shares this module's
  folding core).
* `"complex"`: inverse DFT of `ln|X| + j arg X` with the phase
  unwrapped and its linear component removed (Neelamani Eq. (14) in
  Havelock Ch. 87). Real-valued for a real record, and invertible:
  [`CepstrumResult.invert`](/phonometry/reference/api/spectra/cepstrum/#cepstrumresultinvert) returns the signal.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `kind` | `"power"` (default), `"real"` or `"complex"`. |
| `nfft` | Even FFT length, at least `x.size` (default: the record length, rounded up to even). Zero-padding reduces the cepstral time-aliasing of sharp log-spectrum features. |

**Returns:** A [`CepstrumResult`](/phonometry/reference/api/spectra/cepstrum/#cepstrumresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## CepstrumResult

```python
CepstrumResult(
    quefrencies: NDArray[np.float64],
    cepstrum: NDArray[np.float64],
    kind: str,
    fs: float,
    nfft: int,
    linear_phase_samples: int,
)
```

A cepstrum over its full quefrency axis.

The quefrency axis runs `0 .. (nfft-1)/fs`; quefrencies above
`nfft/(2 fs)` are the negative (anticausal) quefrencies of the
periodic axis. The power and real cepstra are even about that midpoint,
so their first half carries everything; the complex cepstrum is not
(its causal/anticausal split is what separates minimum-phase from
all-pass content, Havelock Ch. 75).

**Attributes**

| Name | Description |
| :--- | :--- |
| `quefrencies` | Quefrency axis, in seconds. |
| `cepstrum` | Cepstrum values (real-valued for the three kinds). |
| `kind` | `"power"`, `"real"` or `"complex"`. |
| `fs` | Sample rate of the analysed record, in Hz. |
| `nfft` | FFT length used (even; the record is zero-padded to it). |
| `linear_phase_samples` | Linear-phase component removed from the unwrapped phase before the complex cepstrum: the integer number of half-turns at the Nyquist bin, roughly minus the bulk delay in samples (0 for the other kinds, and for a minimum-phase record); restored by `invert`. |

### CepstrumResult.invert()

```python
CepstrumResult.invert() -> NDArray[np.float64]
```

Reconstruct the record from a complex cepstrum.

Applies the forward chain in reverse -- DFT, restore the removed
linear phase, exponentiate, inverse DFT (the homomorphic
deconvolution round trip of Neelamani Sec. 3.3) -- and returns the
zero-padded record, length `nfft`. Only the complex cepstrum
keeps the phase, so only it is invertible.

**Returns:** The reconstructed signal, length `nfft`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `kind` is not `"complex"`. |

### CepstrumResult.plot()

```python
CepstrumResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the cepstrum against quefrency (positive quefrencies).

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## echo_detection

```python
echo_detection(
    x: NDArray[np.float64] | list[float],
    fs: float,
    *,
    min_quefrency: float | None = None,
    max_quefrency: float | None = None,
    nfft: int | None = None,
) -> EchoDetectionResult
```

Detect an echo as the largest power-cepstrum peak in a quefrency band.

A reflection `x(t) = s(t) + a s(t - t0)` leaves a spike of height
`a` -- positive or negative with the sign of the reflection -- at
quefrency `t0` in the power cepstrum (module note; the seismic
reverberation spike trains of Neelamani Sec. 3.3 are the same
signature), regardless of the spectrum of `s` itself, which
concentrates at low quefrencies. The peak is therefore picked on
`|cepstrum|`, so an inverting reflection (`a < 0`, e.g. a
pressure-release boundary) is found at its true delay, and the
*signed* cepstrum value at the peak is returned as the reflection
coefficient. The search band starts above the low-quefrency region
occupied by the source's spectral envelope: raise `min_quefrency`
if the source is very reverberant or narrowband.

The delay is refined by quadratic interpolation of `|cepstrum|`
around the peak; see [`EchoDetectionResult`](/phonometry/reference/api/spectra/cepstrum/#echodetectionresult) for the bin-splitting
caveat on the coefficient at off-sample delays.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal (an impulse response, or any record with an in-record echo), 1-D. |
| `fs` | Sample rate, in Hz. |
| `min_quefrency` | Lower edge of the searched band, in seconds (default 16 samples, clearing the immediate quefrency-zero region; raise it above the source's envelope quefrencies when needed). |
| `max_quefrency` | Upper edge of the searched band, in seconds (default and maximum: half the FFT length, the end of the unambiguous quefrency axis). |
| `nfft` | Even FFT length, at least `x.size` (default: the record length, rounded up to even). |

**Returns:** An [`EchoDetectionResult`](/phonometry/reference/api/spectra/cepstrum/#echodetectionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## EchoDetectionResult

```python
EchoDetectionResult(
    quefrencies: NDArray[np.float64],
    cepstrum: NDArray[np.float64],
    delay: float,
    delay_samples: int,
    reflection_coefficient: float,
    search_range: tuple[float, float],
    fs: float,
    nfft: int,
)
```

An echo delay and reflection coefficient read off the power cepstrum.

**Attributes**

| Name | Description |
| :--- | :--- |
| `quefrencies` | Quefrency axis, in seconds. |
| `cepstrum` | Power cepstrum searched. |
| `delay` | The echo delay, in seconds: the quefrency of the largest `\|cepstrum\|` peak in the searched band, refined by quadratic (parabolic) interpolation of `\|cepstrum\|` through the peak sample and its two neighbours, so an off-sample delay is estimated to a fraction of a sample. |
| `delay_samples` | The peak position in whole samples (no interpolation): the index at which [`reflection_coefficient`](/phonometry/reference/api/underwater/seabed-reflection/#reflection_coefficient) is read, so `delay` differs from `delay_samples/fs` by the interpolated sub-sample offset (at most half a sample). |
| `reflection_coefficient` | Signed cepstrum value at the peak sample. For a single in-record echo `x(t) = s(t) + a s(t - t0)` the power cepstrum's first rahmonic height is exactly `a` -- of either sign -- (the `n = 1` term of the `ln(1 + a e^{-j theta})` series), so the value estimates the reflection coefficient directly, including its polarity. When the true delay falls between samples the rahmonic is split across neighbouring quefrency bins and the reported coefficient underestimates `\|a\|` (down to roughly 65 % of it for a delay midway between samples); the interpolated `delay` is unaffected. |
| `search_range` | The `(min, max)` quefrency band searched, s. |
| `fs` | Sample rate of the analysed record, in Hz. |
| `nfft` | FFT length used. |

### EchoDetectionResult.plot()

```python
EchoDetectionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the power cepstrum with the detected echo marked.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

## lifter

```python
lifter(
    x: NDArray[np.float64] | list[float],
    fs: float,
    cutoff: float,
    *,
    mode: str = 'lowpass',
    nfft: int | None = None,
) -> LifterResult
```

Lifter a record's log spectrum: keep quefrencies below or above a cutoff.

Liftering is filtering in the quefrency domain (Milner Sec. 4.3 in
Havelock Ch. 27): the real cepstrum is windowed and transformed back
to a log-magnitude spectrum. A **lowpass** lifter keeps quefrencies
below `cutoff` (and the quefrency-zero mean level), recovering the
smooth spectral envelope with the harmonic or echo ripple removed; a
**highpass** lifter keeps the complement, isolating the ripple. The two
modes are exactly complementary in dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Signal, 1-D. |
| `fs` | Sample rate, in Hz. |
| `cutoff` | Cutoff quefrency, in seconds; must resolve to at least one sample and at most `nfft/2` samples. |
| `mode` | `"lowpass"` (default) or `"highpass"`. |
| `nfft` | Even FFT length, at least `x.size` (default: the record length, rounded up to even). |

**Returns:** A [`LifterResult`](/phonometry/reference/api/spectra/cepstrum/#lifterresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs or parameters are invalid. |

## LifterResult

```python
LifterResult(
    frequencies: NDArray[np.float64],
    spectrum_db: NDArray[np.float64],
    liftered_db: NDArray[np.float64],
    quefrencies: NDArray[np.float64],
    cepstrum: NDArray[np.float64],
    cutoff: float,
    mode: str,
    fs: float,
    nfft: int,
)
```

A log spectrum split by liftering (Milner Sec. 4.3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequency axis, in Hz. |
| `spectrum_db` | Log-magnitude spectrum of the record, in dB. |
| `liftered_db` | Log-magnitude spectrum rebuilt from the kept quefrencies alone, in dB. Lowpass and highpass parts add up to `spectrum_db` exactly (the split is linear in the log). |
| `quefrencies` | Quefrency axis of the underlying real cepstrum, s. |
| `cepstrum` | The real cepstrum the lifter window was applied to. |
| `cutoff` | Lifter cutoff quefrency, in seconds. |
| `mode` | `"lowpass"` (spectral envelope) or `"highpass"` (fine structure). |
| `fs` | Sample rate of the analysed record, in Hz. |
| `nfft` | FFT length used. |

### LifterResult.plot()

```python
LifterResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | NDArray[Any]
```

Plot the cepstrum with the cutoff and the two log spectra.

With `ax` given, only the spectrum panel is drawn on it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |
