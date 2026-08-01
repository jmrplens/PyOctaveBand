---
title: "metrology.intensity_compliance"
description: "IEC 61043:1993 sound-intensity instrument class verification."
sidebar:
  label: "intensity_compliance"
---

IEC 61043:1993 sound-intensity instrument class verification.

A two-microphone (p-p) intensity chain is graded by its **pressure-residual
intensity index** `delta_pI0`: feed both measurement channels the same pink
noise (or expose both probe microphones to the same pressure) and the true
intensity is exactly zero, yet the residual phase mismatch between the channels
reports a small false intensity. The difference between the indicated sound
pressure level and that residual intensity level, evaluated for an air density
of 1.2048 kg/m3, is `delta_pI0` (IEC 61043:1993, definition 3.11).

**Table 2** of the standard (EN 61043:1994 standard page 14) prescribes the
*minimum* `delta_pI0` per one-third-octave band from 50 Hz to 6.3 kHz for a
probe, a processor and a complete instrument, in class 1 and class 2, at the
nominal microphone separation of 25 mm. The table is transcribed digit for
digit below. Its Note 1 gives the separation rule: for any other microphone
separation `x` in millimetres, add $10 \log_{10}(x/25)$ dB to every figure,
so a
wider spacer both earns and demands more low-frequency margin. Note 2 restricts
the requirement to the octave-band centre frequencies for processors that only
analyse in octave bands.

Two related requirements of the same standard are exposed here as well:

* **Clause 6.1** (frequency range of processors): a class 1 processor covers at
  least 45 Hz to 7.1 kHz in one-third-octave bands (the 22 tabulated bands from
  50 Hz to 6.3 kHz). A class 2 processor covers *either* that same
  one-third-octave range *or*, alternatively, 45 Hz to 5.6 kHz in octave bands
  (the 7 octave bands from 63 Hz to 4 kHz). A verdict computed over a narrower
  set of bands attests only the bands supplied, which
  [`verify_intensity_class`](/phonometry/reference/api/power/intensity-compliance/#verify_intensity_class) flags as `range_limited`. Because the octave
  range is open to class 2 alone, a class 1 verdict reached over octave bands
  only is flagged too, and a probe (tested in one-third octaves by clause 12.4)
  cannot use the alternative at all.

  The Spanish translation UNE-EN 61043:1999 states only the octave alternative
  for class 2, dropping the one-third-octave one; this module follows the
  EN/IEC text (see `docs/ERRATA.md`).
* **Clause 8** (instrument assembled from separate components): a class 1
  instrument consists of a class 1 processor and a class 1 probe; a class 2
  instrument of any other combination of class 1 and class 2 components. See
  [`instrument_class_from_components`](/phonometry/reference/api/power/intensity-compliance/#instrument_class_from_components).

The index is also the instrument's phase-error floor in disguise. In an axially
propagating plane progressive wave the true phase difference across the spacer
is $k d$, so a residual intensity produced by a channel phase mismatch
$\phi_s$ gives $\delta_{pI0} = 10 \log_{10}(k d / \phi_s)$ (Fahy,
*Sound Intensity*
2nd ed., equation (7.16)); [`phase_mismatch_from_residual_index`](/phonometry/reference/api/power/intensity-compliance/#phase_mismatch_from_residual_index) and
[`residual_index_from_phase_mismatch`](/phonometry/reference/api/power/intensity-compliance/#residual_index_from_phase_mismatch) convert between the two. Fahy's
worked check in section 6.8 is the anchor: $\delta_{pI0} = 20$ dB means
a mismatch of one hundredth of $k d$, about 0.26 degrees at 1 kHz over
a 25 mm
separation.

The measured `delta_pI0` this module classifies is a property of the whole
probe-spacer-analyser chain and must be determined with the spacer that will be
fitted in the field; the library does not measure it. Once classified, the
ISO 9614 dynamic capability $L_d = \delta_{pI0} - K$ follows from
[`phonometry.emission.intensity.dynamic_capability_index`](/phonometry/reference/api/power/intensity/#dynamic_capability_index).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## instrument_class_from_components

```python
instrument_class_from_components(
    probe_class: int,
    processor_class: int,
) -> int
```

Class of an instrument assembled from a separate probe and processor.

IEC 61043:1993 clause 8: when a probe and a processor are supplied
separately, a class 1 instrument consists of a class 1 processor and a
class 1 probe, while a class 2 instrument consists of any other combination
of class 1 and class 2 components (class 1 processor with class 2 probe,
class 2 processor with class 1 probe, or both class 2). The rule therefore
reduces to the looser of the two component classes.

**Parameters**

| Name | Description |
| :--- | :--- |
| `probe_class` | Class of the probe (1 or 2). |
| `processor_class` | Class of the processor (1 or 2). |

**Returns:** The class of the assembled instrument (1 or 2).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If either class is not 1 or 2. The non-real-time class 2X of clause 4 is not modelled here. |

## intensity_class_compliance

```python
intensity_class_compliance(
    residual_index: list[float] | np.ndarray,
    frequencies: list[float] | np.ndarray,
    *,
    device: str = 'instrument',
    spacing: float = 0.025,
) -> IntensityInstrumentComplianceResult
```

Verify a `delta_pI0` spectrum and package the verdict as a result.

Runs [`verify_intensity_class`](/phonometry/reference/api/power/intensity-compliance/#verify_intensity_class) and stores the outcome together with
the measured spectrum and the two rescaled Table 2 masks, so the returned
object exposes `.plot()` and an accredited `.report()` fiche.

**Parameters**

| Name | Description |
| :--- | :--- |
| `residual_index` | Measured `delta_pI0` per band, in decibels. |
| `frequencies` | Band centre frequencies in Hz, one per entry. |
| `device` | `"probe"`, `"processor"` or `"instrument"`. |
| `spacing` | Microphone separation in metres (default 0.025). |

**Returns:** An [`IntensityInstrumentComplianceResult`](/phonometry/reference/api/power/intensity-compliance/#intensityinstrumentcomplianceresult).

## IntensityInstrumentComplianceResult

```python
IntensityInstrumentComplianceResult(
    overall_class: int | None,
    bands: tuple[dict[str, Any], ...],
    frequency: np.ndarray,
    residual_index: np.ndarray,
    limit_class1: np.ndarray,
    limit_class2: np.ndarray,
    device: str,
    spacing: float,
    spacing_offset_db: float,
    range_limited: bool = False,
)
```

IEC 61043:1993 class verdict of a p-p sound-intensity chain.

Wraps the outcome of [`verify_intensity_class`](/phonometry/reference/api/power/intensity-compliance/#verify_intensity_class) together with the
measured spectrum and the two Table 2 masks it was judged against, so the
result can redraw itself and render an accredited fiche.

**Attributes**

| Name | Description |
| :--- | :--- |
| `overall_class` | The loosest class every band meets (1 or 2), or `None` when at least one band meets neither. |
| `bands` | The per-band verdict dictionaries of [`verify_intensity_class`](/phonometry/reference/api/power/intensity-compliance/#verify_intensity_class), as an immutable tuple. |
| `frequency` | Nominal band centre frequencies, in Hz. |
| `residual_index` | Measured `delta_pI0` per band, in dB. |
| `limit_class1` | Class 1 minimum `delta_pI0` per band, in dB, already rescaled to `spacing`. |
| `limit_class2` | Class 2 minimum per band, in dB, likewise rescaled. |
| `device` | `"probe"`, `"processor"` or `"instrument"`. |
| `spacing` | Microphone separation the verdict applies to, in metres. |
| `spacing_offset_db` | The Table 2 Note 1 term $10 \log_{10}(x/25)$ added to the printed 25 mm figures, in dB. |
| `range_limited` | `True` when the verified bands cover neither the 22 one-third-octave bands nor the 7 octave bands of clause 6.1, so the stated class attests only the bands supplied. |

### IntensityInstrumentComplianceResult.binding_margin()

```python
IntensityInstrumentComplianceResult.binding_margin(
    device_class: int | None = None,
) -> float
```

Smallest per-band margin to a class, in dB (the binding margin).

**Parameters**

| Name | Description |
| :--- | :--- |
| `device_class` | 1 or 2; `None` (default) uses `reference_class`. |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `device_class` is not 1 or 2. |

### IntensityInstrumentComplianceResult.failing_bands()

```python
IntensityInstrumentComplianceResult.failing_bands(
    device_class: int | None = None,
) -> list[float]
```

Nominal centre frequencies of the bands that miss a class, in Hz.

**Parameters**

| Name | Description |
| :--- | :--- |
| `device_class` | 1 or 2; `None` (default) uses `reference_class`. |

### IntensityInstrumentComplianceResult.phase_mismatch()

```python
IntensityInstrumentComplianceResult.phase_mismatch(
    c: float = 343.0,
) -> np.ndarray
```

Equivalent channel phase mismatch per band, in degrees.

Converts the measured `delta_pI0` spectrum with
[`phase_mismatch_from_residual_index`](/phonometry/reference/api/power/intensity-compliance/#phase_mismatch_from_residual_index) at the result's own
microphone separation, so the verdict can be read as the phase-matching
the chain achieves.

**Parameters**

| Name | Description |
| :--- | :--- |
| `c` | Speed of sound in m/s (default 343.0). |

### IntensityInstrumentComplianceResult.plot()

```python
IntensityInstrumentComplianceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the measured `delta_pI0` over the Table 2 class masks.

See `phonometry._plot.metrology.plot_intensity_class`. Requires
matplotlib (`pip install phonometry[plot]`) and returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

### IntensityInstrumentComplianceResult.reference_class()

```python
IntensityInstrumentComplianceResult.reference_class() -> int
```

The class whose mask the fiche and the plot read margins against.

The achieved class when the chain complies, else class 2 (the loosest
class of the standard, the one it comes closest to meeting).

### IntensityInstrumentComplianceResult.report()

```python
IntensityInstrumentComplianceResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an IEC 61043 residual-index verification fiche to a PDF.

Writes a one-page accredited report: the standard-basis line, an
optional metadata header block, the per-band table of measured index,
Table 2 requirement and margin beside the mask-overlay plot, the boxed
class result, an optional verdict row against a supplied
`required_class` and the footer disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare fiche (body, result and disclaimer only). A supplied `required_class` drives the verdict row. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform signature; it has no effect on the single-layout intensity-compliance fiche. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## phase_mismatch_from_residual_index

```python
phase_mismatch_from_residual_index(
    residual_index: float | list[float] | np.ndarray,
    frequency: float | list[float] | np.ndarray,
    spacing: float,
    c: float = 343.0,
) -> np.ndarray
```

Channel phase mismatch equivalent to a pressure-residual intensity index.

In an axially propagating plane progressive wave the true phase difference
between the two sensing points is $k d = 2\pi f d / c$, and a
residual
intensity produced by a channel phase mismatch $\phi_s$ satisfies
$\delta_{pI0} = 10 \log_{10}(k d / \phi_s)$ (Fahy, *Sound Intensity*
2nd ed., equations (7.4) and (7.16)), so:

$$
\phi_s = k d \cdot 10^{-\delta_{pI0} / 10}
$$

The ratio is dimensionless, so $\phi_s$ is returned in the same
angular unit $k d$ is expressed in; degrees are used here.

**Parameters**

| Name | Description |
| :--- | :--- |
| `residual_index` | `delta_pI0` in decibels (scalar or array). |
| `frequency` | Frequency in Hz (scalar or array, broadcast against `residual_index`). |
| `spacing` | Microphone separation in metres. |
| `c` | Speed of sound in m/s (default 343.0). |

**Returns:** The equivalent phase mismatch in degrees, as a `numpy.ndarray` (0-d for scalar inputs).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `spacing`, `c` or `frequency` are not positive and finite, or if `residual_index` is not finite. |

## residual_index_from_phase_mismatch

```python
residual_index_from_phase_mismatch(
    phase_mismatch: float | list[float] | np.ndarray,
    frequency: float | list[float] | np.ndarray,
    spacing: float,
    c: float = 343.0,
) -> np.ndarray
```

Pressure-residual intensity index of a given channel phase mismatch.

The inverse of [`phase_mismatch_from_residual_index`](/phonometry/reference/api/power/intensity-compliance/#phase_mismatch_from_residual_index):

$$
\delta_{pI0} = 10 \log_{10}\frac{k d}{\phi_s}, \qquad k d = \frac{2 \pi f d}{c}
$$

with $k d$ and $\phi_s$ both in degrees (the ratio is
dimensionless).
Because $k d$ grows with frequency while a mismatch that is constant
in
degrees does not, the index rises by 10 dB per decade of frequency: this is
why a fixed phase-matching quality yields the falling low-frequency
`delta_pI0` that IEC 61043 Table 2 grades band by band, and why a wider
spacer buys $10 \log_{10}(x/25)$ dB of index exactly as Note 1 of that
table requires.

**Parameters**

| Name | Description |
| :--- | :--- |
| `phase_mismatch` | $\phi_s$ in degrees (scalar or array, > 0). |
| `frequency` | Frequency in Hz (scalar or array, broadcast against `phase_mismatch`). |
| `spacing` | Microphone separation in metres. |
| `c` | Speed of sound in m/s (default 343.0). |

**Returns:** `delta_pI0` in decibels, as a `numpy.ndarray` (0-d for scalar inputs).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `phase_mismatch` is not positive and finite, or if `spacing`, `c` or `frequency` are not positive and finite. |

## residual_index_limits

```python
residual_index_limits(
    device: str = 'instrument',
    *,
    spacing: float = 0.025,
    frequencies: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

IEC 61043:1993 Table 2 minimum pressure-residual intensity index.

Returns the class 1 and class 2 minima the standard requires of a device
kind, already rescaled to the microphone separation in use with the Note 1
rule $+10 \log_{10}(x/25)$ (`x` in millimetres, i.e.
$10 \log_{10}(\text{spacing}/0.025)$
for a spacing in metres).

**Parameters**

| Name | Description |
| :--- | :--- |
| `device` | `"probe"`, `"processor"` or `"instrument"` (the three column groups of Table 2). |
| `spacing` | Microphone separation in metres (default 0.025, the nominal separation the table is printed for). |
| `frequencies` | Band centre frequencies in Hz to report the limits at, as nominal labels or the exact base-ten centres behind them. `None` (default) returns all 22 tabulated one-third-octave bands. |

**Returns:** Tuple `(frequencies, class1, class2)` of the nominal band centres in Hz and the two minimum `delta_pI0` requirements in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `device` is unknown, `spacing` is not positive or a frequency is not a tabulated band. |

## verify_intensity_class

```python
verify_intensity_class(
    residual_index: list[float] | np.ndarray,
    frequencies: list[float] | np.ndarray,
    *,
    device: str = 'instrument',
    spacing: float = 0.025,
) -> dict[str, Any]
```

Verify a measured `delta_pI0` spectrum against IEC 61043:1993 Table 2.

Each band's measured pressure-residual intensity index is compared with the
class 1 and class 2 minima of Table 2 for the device kind, rescaled to the
microphone separation in use (Note 1, $+10 \log_{10}(x/25)$). A band meets
a class when its measured index is greater than or equal to that class's
minimum, so the margin is `measured - minimum` and a band exactly on the
limit passes. The overall class is the loosest per-band class, or `None`
when any band meets neither class.

Clause 6.1 fixes the frequency range the class attests: 45 Hz to 7.1 kHz in
one-third-octave bands (the 22 tabulated bands, 50 Hz to 6.3 kHz), or, for
an octave-band processor, 45 Hz to 5.6 kHz in octave bands (63 Hz to
4 kHz). `range_limited` is `True` when the supplied bands cover neither
of those sets, in which case the returned class attests the bands actually
verified and not the standard's full frequency range. The one-third-octave
range is required for class 1 and available to class 2, while the octave
range is offered as a class 2 alternative only, so a verdict computed over
the 7 octave bands attests a class 2 result but is still `range_limited`
when it reaches class 1. The octave
alternative is not open to a `"probe"` at all: a probe has no analysis
bands of its own and clause 12.4 determines its index at one-third-octave
intervals across the whole 50 Hz to 6.3 kHz range.

**Parameters**

| Name | Description |
| :--- | :--- |
| `residual_index` | Measured `delta_pI0` per band, in decibels. |
| `frequencies` | Band centre frequencies in Hz, one per entry of `residual_index`, as nominal Table 2 labels or the exact base-ten centres behind them. |
| `device` | `"probe"`, `"processor"` or `"instrument"`. |
| `spacing` | Microphone separation in metres (default 0.025). |

**Returns:** Dict with `overall_class` (1, 2 or `None`), `range_limited`, `bands` (a list of `{"freq", "class", "residual_index_db", "limit_class1_db", "limit_class2_db", "margin_class1_db", "margin_class2_db"}`), `device`, `spacing` and `spacing_offset_db` (the Note 1 term applied to the table).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs disagree in length, a frequency is not a tabulated band, a band is repeated, or `device`/`spacing` are invalid. |
