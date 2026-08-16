---
title: "electroacoustics.microphone"
description: "Rated microphone characteristics (IEC 60268-4)."
sidebar:
  label: "microphone"
---

Rated microphone characteristics (IEC 60268-4).

A microphone measurement/rating report gathers the *rated characteristics*
IEC 60268-4:2014 defines around a measured free-field frequency response: the
**free-field sensitivity** and its level re 1 V/Pa (clauses 11.1/11.2.1), the
**frequency response** with the manufacturer's tolerance (12.1) and the
**effective frequency range** read against that tolerance (12.2), the
**directional pattern** and the **directivity index** (13.1/13.2), the
**equivalent sound pressure level due to inherent noise** (17), the **overload
sound pressure level** at a stated total-harmonic-distortion limit (14.2/15.2),
the **rated impedance** and the **rated minimum permitted load impedance**
(10.2/10.3) and the **rated power supply** (9.1). This module bundles those
into a single [`MicrophoneCharacteristics`](/phonometry/reference/api/electroacoustics/microphone/#microphonecharacteristics) result whose `report` method
renders the IEC 60268-4 rated-characteristics fiche, with the characteristic
graphs laid out to the IEC 60263:1982 scale conventions.

Four of the characteristics are *computed* from the standard's own definitions
so the report never merely repeats a manufacturer number:

* **Sensitivity level** (11.1). The sensitivity `M` is the ratio of the
  output voltage to the sound pressure, in volts per pascal; its level is

  $$
  L_M = 20 \log_{10}(M / M_\mathrm{r}), \qquad M_\mathrm{r} = 1\ \mathrm{V/Pa}
  $$

  the rated sensitivity referring to the standard reference frequency of
  1 000 Hz (11.3). This is the first clean-room oracle: 12.5 mV/Pa returns
  $20 \log_{10} 0.0125 = -38.06$ dB re 1 V/Pa exactly.

* **Effective frequency range** (12.2). The range of frequencies over which
  the response does not deviate by more than a specified amount from the ideal
  (flat) response through the reference-frequency level. The band edges are
  the interpolated frequencies where the relative response crosses the
  `+/- tolerance` limits on either side of the reference frequency, which is
  the second oracle: a response crossing a limit at chosen frequencies returns
  exactly those frequencies.

* **Directivity index** (13.2.2). $D = 20 \log_{10}(M_0 / M_\text{diff})$
  where the diffuse-field sensitivity of a rotationally symmetric pattern
  follows 11.2.2 a):

  $$
  M_\text{diff}^2 = \frac{1}{2} \int_0^{\pi} M^2(\theta) \sin(\theta) \, d\theta
  $$

  For the ideal cardioid $M(\theta) = M_0 (1 + \cos\theta) / 2$
  the integral is $M_0^2 / 3$, so $D = 10 \log_{10} 3 = 4.77$ dB,
  the third oracle.

* **Equivalent sound pressure level due to inherent noise** (17.2 d/e). The
  equivalent sound pressure is the ratio of the weighted inherent-noise
  output voltage to the rated free-field sensitivity,
  $p_\mathrm{N} = U_\mathrm{N} / M$, and its level is
  $L_\mathrm{N} = 20 \log_{10}(p_\mathrm{N} / p_0)$ with $p_0 = 20$ uPa, the fourth
  oracle. The overload sound pressure level (15.2.2) is read from a measured
  distortion-against-level curve as the interpolated sound pressure level
  where the distortion reaches the specified limit.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## microphone_characteristics

```python
microphone_characteristics(
    frequencies: ArrayLike,
    response_db: ArrayLike,
    sensitivity_mv_per_pa: float,
    *,
    reference_frequency: float = 1000.0,
    tolerance_db: float = 2.0,
    directivity: MicrophoneDirectivity = ...,
    noise: MicrophoneNoise = ...,
    overload: MicrophoneOverload = ...,
    electrical: MicrophoneElectrical = ...,
) -> MicrophoneCharacteristics
```

Assemble the rated microphone characteristics for an IEC 60268-4 report.

The sensitivity level (11.1), the effective frequency range (12.2), the
directivity index (13.2.2, when a rear-reaching pattern is supplied) and
the equivalent noise level (17.2, when the noise voltage is supplied) are
computed from the standard's definitions; the optional directional, noise
and distortion data feed the corresponding report panels.

The optional rated characteristics are grouped as the standard groups
them, one bundle per clause, so each may be omitted whole.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Free-field response frequency axis, in Hz (1-D, > 0). |
| `response_db` | Free-field frequency response, in dB, relative to the output at a stated frequency (12.1.1); it is re-normalized to 0 dB at `reference_frequency`. |
| `sensitivity_mv_per_pa` | Rated free-field sensitivity `M` at the reference frequency, in mV/Pa (11.2.1/11.3). |
| `reference_frequency` | Stated reference frequency, in Hz; the 11.3 standard reference frequency of 1 000 Hz by default. |
| `tolerance_db` | Half-width of the response tolerance, in dB (default 2), defining the effective frequency range (12.2). |
| `directivity` | Directional characteristics of Clause 13 as a [`MicrophoneDirectivity`](/phonometry/reference/api/electroacoustics/microphone/#microphonedirectivity): the directional pattern, its stated frequency and the directivity index. |
| `noise` | Inherent noise of Clause 17 as a [`MicrophoneNoise`](/phonometry/reference/api/electroacoustics/microphone/#microphonenoise): the weighted noise voltage or the stated equivalent noise level, its weighting and the noise spectrum. |
| `overload` | Limiting characteristic of 15.2 with the 14.2 distortion curve it is read from, as a [`MicrophoneOverload`](/phonometry/reference/api/electroacoustics/microphone/#microphoneoverload). |
| `electrical` | Electrical impedance (Clause 10) and rated power supply (Clause 9) as a [`MicrophoneElectrical`](/phonometry/reference/api/electroacoustics/microphone/#microphoneelectrical). |

**Returns:** A [`MicrophoneCharacteristics`](/phonometry/reference/api/electroacoustics/microphone/#microphonecharacteristics).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## MicrophoneCharacteristics

```python
MicrophoneCharacteristics(
    frequencies: NDArray[np.float64],
    response_db: NDArray[np.float64],
    reference_frequency: float,
    sensitivity_mv_per_pa: float,
    sensitivity_level_db: float,
    tolerance_db: float,
    effective_range: tuple[float, float],
    rated_impedance: float | None,
    minimum_load_impedance: float | None,
    equivalent_noise_level_db: float | None,
    noise_weighting: str,
    max_spl_db: float | None,
    max_spl_thd_percent: float,
    distortion_spl_db: NDArray[np.float64] | None,
    distortion_thd_percent: NDArray[np.float64] | None,
    noise_frequencies: NDArray[np.float64] | None,
    noise_band_levels_db: NDArray[np.float64] | None,
    polar_angles_deg: NDArray[np.float64] | None,
    polar_db: NDArray[np.float64] | None,
    polar_frequency: float | None,
    directivity_index_db: float | None,
    powering: str | None,
    supply_current_ma: float | None,
)
```

Rated microphone characteristics for an IEC 60268-4 report.

The free-field frequency response and the rated free-field sensitivity are
the required inputs; the directional pattern, the inherent-noise spectrum
and the distortion-against-level curve are optional panels rendered when
supplied. The sensitivity level, the effective frequency range, the
directivity index and the equivalent noise level are computed from the
standard's definitions (see the module docstring).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Free-field response frequency axis, in Hz. |
| `response_db` | Free-field frequency response relative to the level at `reference_frequency` (0 dB there), in dB (12.1.1). |
| `reference_frequency` | Stated reference frequency of the rated sensitivity and the response normalization, in Hz (11.3). |
| `sensitivity_mv_per_pa` | Rated free-field sensitivity `M` at the reference frequency, in mV/Pa (11.2.1/11.3). |
| `sensitivity_level_db` | Sensitivity level $20 \log_{10}(M / 1\,\mathrm{V/Pa})$, in dB re 1 V/Pa (11.1). |
| `tolerance_db` | Half-width of the response tolerance, in dB (12.1.1). |
| `effective_range` | Computed effective frequency range `(lo, hi)` against the tolerance limits, in Hz (12.2). |
| `rated_impedance` | Rated (internal) impedance, in ohm (10.2), or `None`. |
| `minimum_load_impedance` | Rated minimum permitted load impedance, in ohm (10.3), or `None`. |
| `equivalent_noise_level_db` | Equivalent sound pressure level due to inherent noise, in dB SPL with `noise_weighting` weighting (17), or `None`. |
| `noise_weighting` | Weighting of the inherent-noise measurement (IEC 60268-1), `"A"` by default. |
| `max_spl_db` | Overload sound pressure level at `max_spl_thd_percent` total harmonic distortion, in dB SPL (15.2), or `None`. |
| `max_spl_thd_percent` | Distortion limit defining `max_spl_db`, in % (15.2.1). |
| `distortion_spl_db` | Distortion-curve sound-pressure-level axis, in dB SPL (14.2), or `None`. |
| `distortion_thd_percent` | Total harmonic distortion against level, in % (14.2), or `None`. |
| `noise_frequencies` | Inherent-noise spectrum frequency axis, in Hz (17.2 b), or `None`. |
| `noise_band_levels_db` | Inherent-noise equivalent band levels, in dB SPL (17.2 b), or `None`. |
| `polar_angles_deg` | Directional-pattern angles, in degrees (13.1), or `None`. |
| `polar_db` | Directional pattern $G(\theta)$ relative to the reference-axis response, in dB (13.1.2), or `None`. |
| `polar_frequency` | Stated frequency of the directional pattern, in Hz, or `None`. |
| `directivity_index_db` | Directivity index at `polar_frequency`, in dB (13.2), or `None`. |
| `powering` | Rated power supply description, e.g. the IEC 61938 phantom-powering designation and voltage (9.1), or `None`. |
| `supply_current_ma` | Current drawn from the power supply, in mA (9.1), or `None`. |

### MicrophoneCharacteristics.diffuse_field_sensitivity_level_db

*property*

Diffuse-field sensitivity level, in dB re 1 V/Pa, or `None`.

Per 11.2.2.1 the diffuse-field sensitivity level equals the free-field
plane-wave sensitivity level minus the directivity index (13.2).

### MicrophoneCharacteristics.plot()

```python
MicrophoneCharacteristics.plot(
    quantity: str = 'response',
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot one IEC 60268-4 microphone rated characteristic on one axes.

One concept per figure, drawn by the same shared renderer the
`.report()` fiche composes: `"response"` (the free-field response
with its tolerance band, reference-frequency and effective-range
markers, the default), `"directivity"` (the polar directional pattern
on the 25 dB reference circle), `"noise"` (the inherent-noise
band-level spectrum) and `"distortion"` (total harmonic distortion
against sound pressure level).

**Parameters**

| Name | Description |
| :--- | :--- |
| `quantity` | Which characteristic to plot (see above). |
| `ax` | Existing axes to draw on, or `None` for a fresh figure (a polar axes is created for `"directivity"`). |
| `language` | Label language, `"en"` (default) or `"es"`. |

**Returns:** The axes the characteristic was drawn on.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `quantity` or `language` is unknown, or the result carries no data for `quantity`. |
| ImportError | If matplotlib is not installed (`pip install phonometry[plot]`). |

### MicrophoneCharacteristics.report()

```python
MicrophoneCharacteristics.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render the IEC 60268-4 microphone-characteristics fiche to a PDF.

Writes a one-page rated-characteristics data sheet: the standard-basis
line (IEC 60268-4:2014, graphs to IEC 60263:1982), an optional metadata
header, the rated-characteristics table beside the free-field response
with its tolerance band and effective-range markers, the directional,
inherent-noise and distortion panels for the data supplied, a boxed
sensitivity/range result, an optional equivalent-noise verdict when a
requirement is given, and the footer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity and, through `requirement`, a maximum permitted equivalent noise level (dB SPL) the verdict row compares against. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the fiche has one layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

### MicrophoneCharacteristics.sensitivity_v_per_pa

*property*

Rated free-field sensitivity `M`, in V/Pa (11.1).

### MicrophoneCharacteristics.signal_to_noise_ratio_db

*property*

Signal-to-noise ratio re 1 Pa (94 dB SPL), in dB, or `None`.

The datasheet companion of the equivalent noise level: the level
of 1 Pa
($20 \log_{10}(1\,\mathrm{Pa} / 20\,\mathrm{\mu Pa}) = 93.98$ dB
SPL) minus the equivalent sound pressure level due to inherent
noise (17), carrying the same weighting.

## MicrophoneDirectivity

```python
MicrophoneDirectivity(
    polar: tuple[ArrayLike, ArrayLike] | None = None,
    frequency: float | None = None,
    index_db: float | None = None,
)
```

Directional characteristics of the microphone (IEC 60268-4 Clause 13).

The clause's two characteristics travel together: the directional pattern
(13.1), a curve of the free-field sensitivity level against the angle of
incidence for a stated frequency, and the directivity index read from it
(13.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `polar` | Directional pattern as `(angles_deg, relative_db)`, the pattern $G(\theta)$ relative to the reference-axis response, in degrees and dB (13.1.1/13.1.2). |
| `frequency` | Stated frequency of the directional pattern, in Hz (13.1.1 requires the frequency or frequency band to be stated). |
| `index_db` | Stated directivity index, in dB (13.2.1); when omitted it is computed from `polar` through the 11.2.2 a) diffuse-field integral if the pattern reaches the rear. |

## MicrophoneElectrical

```python
MicrophoneElectrical(
    rated_impedance: float | None = None,
    minimum_load_impedance: float | None = None,
    powering: str | None = None,
    supply_current_ma: float | None = None,
)
```

Electrical impedance and rated power supply (IEC 60268-4 Clauses 10/9).

The electrical side of the rated characteristics: the rated (internal)
impedance and the rated minimum permitted load impedance of Clause 10, and
the rated power supply of Clause 9.

**Attributes**

| Name | Description |
| :--- | :--- |
| `rated_impedance` | Rated (internal) impedance, in ohm (10.2). |
| `minimum_load_impedance` | Rated minimum permitted load impedance, in ohm (10.3). |
| `powering` | Rated power supply description (9.1), e.g. the IEC 61938 phantom-powering designation and voltage. |
| `supply_current_ma` | Current drawn from the power supply, in mA (9.1). |

## MicrophoneNoise

```python
MicrophoneNoise(
    voltage: float | None = None,
    equivalent_level_db: float | None = None,
    weighting: str = 'A',
    spectrum: tuple[ArrayLike, ArrayLike] | None = None,
)
```

Inherent noise of the microphone (IEC 60268-4 Clause 17).

The clause specifies the equivalent sound pressure level due to inherent
noise, either stated directly or computed from the weighted inherent-noise
output voltage, and the band spectrum that voltage was measured over.

**Attributes**

| Name | Description |
| :--- | :--- |
| `voltage` | Weighted r.m.s. output voltage due to inherent noise, in V (17.2 b); the equivalent noise level is computed from it. |
| `equivalent_level_db` | Stated equivalent sound pressure level due to inherent noise, in dB SPL (17.1), when not computed from `voltage`. |
| `weighting` | Weighting of the inherent-noise measurement (default `"A"`, the IEC 60268-1 6.2.1 recommendation). |
| `spectrum` | Inherent-noise spectrum as `(frequencies, band_levels_db)` in (Hz, dB SPL) (17.2 b). |

## MicrophoneOverload

```python
MicrophoneOverload(
    distortion: tuple[ArrayLike, ArrayLike] | None = None,
    thd_percent: float = 1.0,
    spl_db: float | None = None,
)
```

Overload sound pressure and the distortion it is read from (14.2/15.2).

The limiting characteristic of IEC 60268-4 15.2 is the maximum sound
pressure at which the amplitude non-linearity does not exceed a specified
limit, and 15.2.2 measures it by raising the sound pressure until the
distortion at the output reaches that limit. The total-harmonic-distortion
curve of 14.2, the limit and the resulting level are therefore one group.

**Attributes**

| Name | Description |
| :--- | :--- |
| `distortion` | Total harmonic distortion against level as `(spl_db, thd_percent)` in (dB SPL, %) (14.2). |
| `thd_percent` | Distortion limit defining the overload sound pressure level, in % (default 1, a common 15.2.1 note value). |
| `spl_db` | Stated overload sound pressure level, in dB SPL (15.2); when omitted it is read from `distortion` at `thd_percent`. |
