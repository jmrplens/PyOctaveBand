---
title: "aircraft.aircraft_noise"
description: "Aircraft noise certification: Effective Perceived Noise Level (ICAO Annex 16)."
sidebar:
  label: "aircraft_noise"
---

Aircraft noise certification: Effective Perceived Noise Level (ICAO Annex 16).

The EPNL is the noise-certification metric for transport-category aircraft. It
is built from a half-second spectral time history (24 one-third-octave bands,
50 Hz-10 kHz) in five steps, implementing **ICAO Annex 16 Vol. I, Appendix 2**
(the analytic formulation):

* [`perceived_noisiness`](/phonometry/reference/api/aeroacoustics/aircraft-noise/#perceived_noisiness) -- per-band perceived noisiness `n` (noys),
  the analytic piecewise noy law with the Table A2-3 constants.
* [`perceived_noise_level`](/phonometry/reference/api/aeroacoustics/aircraft-noise/#perceived_noise_level) -- `PNL = 40 + (10/lg2)·lg N` from the total
  noisiness `N = 0.85·n_max + 0.15·Σn`.
* [`tone_correction`](/phonometry/reference/api/aeroacoustics/aircraft-noise/#tone_correction) -- the tone-correction factor `C` (the slope /
  "encircling" method) that penalises spectral irregularities.
* [`effective_perceived_noise_level`](/phonometry/reference/api/aeroacoustics/aircraft-noise/#effective_perceived_noise_level) -- the end-to-end metric: per-record
  `PNLT = PNL + C`, the maximum `PNLTM`, the 10 dB-down integration limits
  and the duration correction, giving `EPNL = PNLTM + D`.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## effective_perceived_noise_level

```python
effective_perceived_noise_level(
    spectra: NDArray[np.float64] | list[list[float]],
    dt: float | NDArray[np.float64] | list[float] = 0.5,
    *,
    reference_time: float = 10.0,
    procedure: str = 'aeroplane',
) -> EPNLResult
```

Effective Perceived Noise Level from a spectral time history (ICAO Annex 16).

Each record (row) is a 24-band one-third-octave spectrum sampled every
`dt` seconds. The per-record `PNL` and tone correction `C` give
`PNLT = PNL + C`; the maximum `PNLTM` and the duration correction over
the 10 dB-down window give `EPNL = PNLTM + D`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `spectra` | Spectral time history, shape `(K, 24)`, in dB. |
| `dt` | Per-record duration, in s (scalar or per record, default 0.5). |
| `reference_time` | Normalising time `T0`, in s (default 10). |
| `procedure` | Tone-correction procedure of App. 2 §4.3.1 Step 1: `"aeroplane"` (default) starts the slope analysis at the 80 Hz band (band 3), `"helicopter"` (helicopters and tilt-rotors) at the 50 Hz band (band 1), so rotor tones in the 50-80 Hz bands are not missed. |

**Returns:** An [`EPNLResult`](/phonometry/reference/api/aeroacoustics/aircraft-noise/#epnlresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the input is not a `(K, 24)` finite array, or `procedure` is not `"aeroplane"` or `"helicopter"`. |

## epnl_from_pnlt

```python
epnl_from_pnlt(
    pnlt: NDArray[np.float64] | list[float],
    dt: float | NDArray[np.float64] | list[float] = 0.5,
    *,
    reference_time: float = 10.0,
    tone_corrections: NDArray[np.float64] | list[float] | None = None,
) -> tuple[float, float, int, int]
```

EPNL from a tone-corrected perceived-noise-level time history (App. 2 §4.5-4.6).

`EPNL = 10·lg( Σ_{kF..kL} 10^{PNLT(k)/10}·Δt(k) ) − 10·lg(T0)` with the
10 dB-down integration limits about the maximum `PNLTM`. The exact
`−10·lg(T0)` form is used rather than the Annex's rounded constant 13 for
uniform 0.5 s records (difference 0.0103 dB); the ETM Table 4-4 integrated
reference reproduces the exact form to five decimals.

When `tone_corrections` is given, the **bandsharing adjustment** `ΔB`
(App. 2 §4.4.2/4.4.3, ETM GM/AMC A2 4.4.2) is applied: if the tone
correction at the PNLTM record is below the average of the records within
one second of it (five records for the uniform 0.5 s cadence), `ΔB` is
that shortfall, added to PNLTM before the 10 dB-down window is found and
included in the reported EPNL.

**Parameters**

| Name | Description |
| :--- | :--- |
| `pnlt` | The tone-corrected perceived noise levels `PNLT(k)`, in PNdB. |
| `dt` | Per-record duration, in s (scalar broadcast or per record). |
| `reference_time` | Normalising time `T0`, in s (default 10). |
| `tone_corrections` | Optional tone corrections `C(k)`, in dB, one per `pnlt` record; enables the bandsharing adjustment `ΔB` described above. Default `None` (no adjustment). |

**Returns:** `(epnl, pnltm, kF, kL)` -- EPNL in EPNdB, the peak PNLTM, and the 0-based 10 dB-down record indices (inclusive).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## EPNLResult

```python
EPNLResult(
    frequencies: NDArray[np.float64],
    times: NDArray[np.float64],
    pnl: NDArray[np.float64],
    tone_correction: NDArray[np.float64],
    pnlt: NDArray[np.float64],
    pnltm: float,
    bandsharing_adjustment: float,
    duration_correction: float,
    epnl: float,
    band_limits: tuple[int, int],
)
```

Effective Perceived Noise Level of an aircraft flyover (ICAO Annex 16).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | The 24 one-third-octave band centre frequencies, in Hz. |
| `times` | Record times, in s. |
| `pnl` | Perceived noise level per record, in PNdB. |
| `tone_correction` | Tone-correction factor per record, in dB. |
| `pnlt` | Tone-corrected perceived noise level per record, in PNdB. |
| `pnltm` | Maximum tone-corrected perceived noise level, in PNdB, including the bandsharing adjustment `ΔB` (App. 2 §4.4.3). |
| `bandsharing_adjustment` | The bandsharing adjustment `ΔB`, in dB (zero unless the tone correction at PNLTM is suppressed). |
| `duration_correction` | Duration correction `D = EPNL − PNLTM`, in dB. |
| `epnl` | Effective perceived noise level, in EPNdB. |
| `band_limits` | The 0-based 10 dB-down record indices `(kF, kL)`. |

### EPNLResult.plot()

```python
EPNLResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the PNL and PNLT time histories with PNLTM and the 10 dB-down band.

### EPNLResult.report()

```python
EPNLResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ICAO Annex 16 EPNL certification fiche to a PDF.

Writes a one-page aircraft-noise-certification data sheet: the
standard-basis line (ICAO Annex 16 Vol. I Appendix 2), an optional
TCDSN-style metadata header (aircraft, manufacturer / type-certificate
holder, applicant, measurement point), a metrics table of the
informational intermediate quantities (PNLTM, the duration correction
`D`, the 10 dB-down record window and, when non-zero, the bandsharing
adjustment), the result's own landscape PNLT-versus-time `plot`,
the boxed `EPNL = X EPNdB` result, a Level | Limit | Margin verdict
row when a certification limit is supplied, a static reference-conditions
strip and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a prediction fiche (metrics, plot and result only, no verdict). A supplied `requirement` is read as the certification EPNL limit in EPNdB (the EPNL passes at or below it). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform signature; it has no effect on the single-layout EPNL fiche. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## NOY_BANDS

*Constant* (`numpy.ndarray, shape (24,)`).

## perceived_noise_level

```python
perceived_noise_level(spl: NDArray[np.float64] | list[float]) -> float
```

Perceived noise level `PNL` (ICAO Annex 16 App. 2 §4.2), in PNdB.

`N = 0.85·n_max + 0.15·Σn` and `PNL = 40 + (10/lg2)·lg N`. If the total
noisiness is not positive the PNL is defined as 0.

**Parameters**

| Name | Description |
| :--- | :--- |
| `spl` | The 24 one-third-octave-band sound pressure levels, in dB. |

**Returns:** The perceived noise level, in PNdB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectrum is not 24 finite levels. |

## perceived_noisiness

```python
perceived_noisiness(
    spl: NDArray[np.float64] | list[float],
) -> NDArray[np.float64]
```

Per-band perceived noisiness `n` in noys (ICAO Annex 16 App. 2 §4.7).

The analytic piecewise noy law, using the Table A2-3 constants:

* `SPL ≥ SPL(a)`: `n = 10^{M(c)·(SPL − SPL(c))}`
* `SPL(b) ≤ SPL < SPL(a)`: `n = 10^{M(b)·(SPL − SPL(b))}`
* `SPL(e) ≤ SPL < SPL(b)`: `n = 0.3·10^{M(e)·(SPL − SPL(e))}`
* `SPL(d) ≤ SPL < SPL(e)`: `n = 0.1·10^{M(d)·(SPL − SPL(d))}`
* `SPL < SPL(d)`: `n = 0` (below the noy floor)

**Parameters**

| Name | Description |
| :--- | :--- |
| `spl` | The 24 one-third-octave-band sound pressure levels, in dB. |

**Returns:** The per-band perceived noisiness, in noys.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectrum is not 24 finite levels. |

## tone_correction

```python
tone_correction(
    spl: NDArray[np.float64] | list[float],
    *,
    start_band: int = 2,
) -> float
```

Tone-correction factor `C` for a spectrum (ICAO Annex 16 App. 2 §4.3).

The maximum over bands of the tone-correction factor derived from the tone
excess `F` above the smoothed background, with the 1.5 dB threshold, the
500 Hz / 5000 Hz frequency split and the 6⅔ dB cap of Table A2-2.

**Parameters**

| Name | Description |
| :--- | :--- |
| `spl` | The 24 one-third-octave-band sound pressure levels, in dB. |
| `start_band` | First band index of the slope analysis (App. 2 §4.3.1 Step 1). The default 2 (80 Hz) is the aeroplane procedure; helicopters and tilt-rotors use 0 (50 Hz). |

**Returns:** The tone-correction factor `C`, in dB (0 if no tones qualify).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectrum is not 24 finite levels. |
