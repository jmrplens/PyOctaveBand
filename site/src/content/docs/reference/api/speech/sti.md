---
title: "speech.sti"
description: "Speech Transmission Index (STI) per IEC 60268-16:2020 (Edition 5)."
sidebar:
  label: "sti"
---

Speech Transmission Index (STI) per IEC 60268-16:2020 (Edition 5).

Implements the full-STI indirect method from impulse responses (Schroeder
modulation transfer function), the direct STIPA method on recorded signals
(Annex B) and an Ed.5-conformant STIPA test-signal generator (clauses A.4
and A.6.1). Only the male speech option exists: Edition 5 removed the
female spectrum and weighting factors (foreword, item d).

The computation chain (octave-band MTF -> auditory masking and reception
threshold correction -> effective SNR clipped to +/-15 dB -> transmission
indices -> band MTI -> weighted STI) is numerically identical between
Ed.4 (2011) clauses A.5.2-A.5.6 and Ed.5; the only Ed.5 numeric change is
the male test-signal spectrum (A.6.1).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## sti_from_impulse_response

```python
sti_from_impulse_response(
    ir: Signal | list[float] | np.ndarray,
    fs: int | None,
    snr: None,
    level: Sequence[float] | np.ndarray,
    ambient: Sequence[float] | np.ndarray,
) -> STIResult

sti_from_impulse_response(
    ir: Signal | list[float] | np.ndarray,
    fs: int | None = ...,
    snr: None = ...,
    *,
    level: Sequence[float] | np.ndarray,
    ambient: Sequence[float] | np.ndarray,
) -> STIResult

sti_from_impulse_response(
    ir: Signal | list[float] | np.ndarray,
    fs: int | None = ...,
    snr: float | Sequence[float] | np.ndarray | None = ...,
    level: Sequence[float] | np.ndarray | None = ...,
) -> STIResult
```

Full STI from a room/system impulse response (indirect method).

The impulse response is filtered into the seven octave bands 125 Hz -
8 kHz (IEC 61260-1 filters) and the modulation transfer function is
obtained from the Schroeder integral (IEC 60268-16, indirect method):

$$
m_k(f_m) = \left| \int h_k^2(t)\, e^{-j 2\pi f_m t}\, dt \right| \Big/ \int h_k^2(t)\, dt
$$

at the 14 modulation frequencies 0,63-12,5 Hz
(A.2.2). The result then follows the standard chain: optional noise
degradation, optional auditory masking and absolute reception
threshold correction, effective SNR clipped to +/-15 dB, transmission
indices, band MTIs and the male-weighted STI (Ed.5 Table A.1).

When neither `level` nor `ambient` is given the level-dependent
auditory masking and the absolute reception threshold corrections are
skipped (they require absolute band levels), matching the common
"noise-free indirect measurement" use of the standard.

:::note
The indirect method has a small positive MTF bias that grows
with reverberation time for a finite noise-carrier IR. It stays within
the IEC 60268-16 A.5.1.2 systematic-error allowance (\<= 0.01 STI) up to
about T60 = 4 s, reaching ~+0.012 STI only in very reverberant rooms
(T60 ~ 8 s, STI ~ 0.19). It is a property of the finite IR, not of the
(exact) Schroeder integration, and does not depend on IR truncation.
At very short reverberation times (T60 \< 0.25 s) the zero-phase
analysis bank smears h^2 slightly, biasing individual low-band
high-modulation-frequency MTF cells by up to ~0.02-0.04 while the
STI itself stays within ~0.001.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Impulse response (1D). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: every modulation index is normalised by the total intensity of its own band, so a factor on the record moves neither the transfer values nor the STI. The absolute levels the noise corrections need arrive through `level` and `ambient`, in dB, not from the samples. |
| `fs` | Sample rate in Hz (>= 22,5 kHz so the 8 kHz band fits). Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `snr` | Optional signal-to-noise ratio in dB, scalar or one value per octave band. Degrades m by 1/(1 + 10^(-SNR/10)); combined with `level` it is interpreted as ambient levels `level - snr` so noise is not applied twice. Mutually exclusive with `ambient`. |
| `level` | Optional speech octave-band levels in dB SPL (7 values) at the listener position; enables the auditory masking (Ed.5 Table A.2) and reception threshold (Ed.5 Table A.3) corrections. |
| `ambient` | Optional ambient noise octave-band levels in dB SPL (7 values); requires `level`. |

**Returns:** [`STIResult`](/phonometry/reference/api/speech/sti/#stiresult) with `mtf` of shape (7, 14).

## stipa

```python
stipa(
    x: Signal | list[float] | np.ndarray,
    fs: int | None,
    reference: Signal | list[float] | np.ndarray | None,
    level: Sequence[float] | np.ndarray,
    ambient: Sequence[float] | np.ndarray,
) -> STIResult

stipa(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = ...,
    reference: Signal | list[float] | np.ndarray | None = ...,
    *,
    level: Sequence[float] | np.ndarray,
    ambient: Sequence[float] | np.ndarray,
) -> STIResult

stipa(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = ...,
    reference: Signal | list[float] | np.ndarray | None = ...,
    level: Sequence[float] | np.ndarray | None = ...,
) -> STIResult
```

STIPA on a recorded test signal (direct method, Annex B).

The recording is filtered into the seven octave bands, squared and
low-passed (~100 Hz) into intensity envelopes, and the modulation
depths at the two Table B.1 modulation frequencies of each band are
measured with the sine/cosine correlation over an integer number of
periods (Ed.4 A.5.2 = Ed.5). The modulation transfer values are the
measured depths normalized by the source modulation index 0,55
(Annex B) - or by the depths measured on `reference` when the
actually emitted signal is supplied - and feed the same masking /
threshold / TI / STI chain as the full method.

Physical background noise is already contained in the recording; use
`level` (and optionally `ambient`) only to enable the absolute
level-dependent corrections, which are otherwise skipped.

An [`STIWarning`](/phonometry/reference/api/speech/sti/#stiwarning) is emitted when the recording is shorter than
the recommended 15 s (IEC 60268-16 STIPA practice, 15 s to 25 s),
because the slow modulation components are then averaged over too few
periods and the recovered modulation depths - and hence the STI - are
biased low (an ideal loopback gives STI ~0.956 at 5 s vs ~0.998 at 18 s).

**Parameters**

| Name | Description |
| :--- | :--- |
| `x` | Recorded STIPA signal (1D), 15 s to 25 s recommended. Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal), whose calibration is applied to the samples and then cancels: every modulation index is normalised by the total intensity of its own band, so a factor on the record moves neither the transfer values nor the STI. The absolute levels the noise corrections need arrive through `level` and `ambient`, in dB, not from the samples. |
| `fs` | Sample rate in Hz (>= 22,5 kHz). Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `reference` | Optional reference recording of the undistorted test signal; its measured modulation depths replace the nominal 0,55 as normalization (useful for non-conformant sources). Accepts a [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal). It is a second recording of the same test signal, so it has to share the rate: two Signals that disagree are refused rather than arbitrated, and measuring the reference on a rate that is not its own would return a perfect STI for a mismatch. Its calibration is applied like any other record's, and then cancels, because what is taken from it is a modulation depth and those are normalised. |
| `level` | Optional speech octave-band levels in dB SPL (7 values) enabling auditory masking and reception threshold corrections. |
| `ambient` | Optional ambient noise octave-band levels in dB SPL (7 values); requires `level`. |

**Returns:** [`STIResult`](/phonometry/reference/api/speech/sti/#stiresult) with `mtf` of shape (7, 2).

## stipa_signal

```python
stipa_signal(
    fs: int,
    seconds: float = 18.0,
    level_db: float | None = None,
    seed: int | None = None,
) -> np.ndarray
```

Generate an IEC 60268-16:2020 conformant STIPA test signal.

Pink-noise carriers are band-limited to half-octave bands centred on
the seven octave-band frequencies (clause A.4), set to the Ed.5 male
speech spectrum of clause A.6.1 (`-2,5; 0,5; 0; -6; -12; -18; -24`
dB re the 500 Hz band) and intensity-modulated with
`0,5 (1 + 0,55 (sin 2 pi f1 t - sin 2 pi f2 t))` - the Table B.1
frequency pair of each band, 180 degrees between components, applied
in amplitude through its square root (Annex B).

**Parameters**

| Name | Description |
| :--- | :--- |
| `fs` | Sample rate in Hz (>= 22,5 kHz). |
| `seconds` | Duration in seconds (the standard recommends 15 s to 25 s; default 18 s). |
| `level_db` | Optional overall level in dB SPL: the output is scaled so its RMS, taken as pascals, sits at `level_db` re 20 uPa. Default (None) normalizes the RMS to 0,1 (digital full scale headroom for the 12-14 dB crest factor). |
| `seed` | Seed for the pink-noise generator (None: random). |

**Returns:** Test signal, 1D array of `round(seconds * fs)` samples.

## STIResult

```python
STIResult(
    sti: float,
    mti: np.ndarray,
    mtf: np.ndarray,
    band_levels: np.ndarray | None,
    rating: str,
)
```

Result of a Speech Transmission Index computation.

`mtf` holds the modulation transfer values actually used for the
transmission indices, i.e. after the optional SNR / masking /
reception-threshold corrections and after clipping to [0, 1]; its
shape is (7, 14) for full STI and (7, 2) for STIPA. `mti` is the
per-band modulation transfer index (7,), `band_levels` echoes the
speech octave-band levels used for the level-dependent corrections
(None when they were skipped) and `rating` is the Annex F
qualification letter (`A+` .. `U`).

### STIResult.plot()

```python
STIResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band MTI bars with the STI and rating letter.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### STIResult.report()

```python
STIResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an IEC 60268-16 speech-transmission-index fiche to a PDF.

Writes a one-page voice-alarm / public-address intelligibility
verification report: a standard-basis line stating the measurement
method (the full STI indirect method from an impulse response, or the
direct STIPA method on a recorded signal), an optional metadata header
block, a per-octave-band modulation transfer index table beside the
per-band MTI bars (the result's own `plot`), the boxed
`STI = X` single number with the Annex F qualification band, an
optional verdict row and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare fiche (body, result and disclaimer only). A supplied `requirement` is read as the minimum required STI (a higher STI passes). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform signature; it has no effect on the single-layout STI fiche. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is not a supported language. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## STIWarning

Warns about suspect STI/STIPA measurements or inputs.
