---
title: "environmental.wind_turbine_noise"
description: "Wind-turbine acoustic noise (IEC 61400-11:2012+A1:2018)."
sidebar:
  label: "wind_turbine_noise"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Wind-turbine acoustic noise (IEC 61400-11:2012+A1:2018).

Two closed-form quantities of the standard:

* [`apparent_sound_power_level`](/phonometry/reference/api/aeroacoustics/wind-turbine-noise/#apparent_sound_power_level) -- the A-weighted apparent sound power
  level `L_WA` referred to the equivalent point source at the rotor centre,
  from the ground-board sound pressure level and the slant distance
  ([`slant_distance`](/phonometry/reference/api/aeroacoustics/wind-turbine-noise/#slant_distance)), `L_WA = L_p − 6 + 10·lg(4π R1²/S0)` (Formula 26).
* [`wind_turbine_tonality`](/phonometry/reference/api/aeroacoustics/wind-turbine-noise/#wind_turbine_tonality) -- the tonal-audibility chain (Formulae 30-34):
  the critical bandwidth ([`critical_bandwidth`](/phonometry/reference/api/environment/measurement/#critical_bandwidth)), the masking-noise level,
  the tonality and the audibility criterion, giving the tonal audibility
  `ΔL_a` that decides whether a tone is audible.

The tonal-audibility formula itself is the ISO 1996-2 Annex C one already in
[`phonometry.environmental_measurement`](/phonometry/reference/api/environment/measurement/); what is specific to IEC 61400-11 is
how the tone and masking-noise levels and the (Zwicker) critical band are
determined from the narrowband spectrum. The rating adjustment `K_T` is the
ISO 1996-2 [`tonal_adjustment`](/phonometry/reference/api/environment/measurement/#tonal_adjustment). The
full measurement pipeline (binning, regression to standardised wind speeds,
uncertainty budgets) is out of scope; these are the underlying closed forms.

## apparent_sound_power_level

```python
apparent_sound_power_level(
    band_levels: float | NDArray[np.float64] | list[float],
    r1: float,
) -> float
```

A-weighted apparent sound power level `L_WA` (IEC 61400-11 Formula 26).

`L_WA,i = L_p,i − 6 + 10·lg(4π R1²/S0)` per one-third-octave band, energy
summed over bands (Formula 27). The `−6 dB` accounts for the ground-board
pressure doubling; `S0 = 1 m²`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `band_levels` | Background-corrected A-weighted band sound pressure levels `L_p,i`, in dB (scalar or per band). The 61400-11-specific background correction (Formula 23 subtraction with the 3-6 dB asterisk marking and the \<= 3 dB not-reported rule, subclause 9.3) is out of scope here and must be applied beforehand; note its rule set differs from the ISO 1996-2 correction in [`phonometry.environmental.measurement.residual_sound_correction`](/phonometry/reference/api/environment/measurement/#residual_sound_correction). |
| `r1` | Slant distance `R1` to the rotor centre, in m. |

**Returns:** The apparent sound power level `L_WA`, in dB re 1 pW.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## slant_distance

```python
slant_distance(
    hub_height: float,
    rotor_diameter: float,
    *,
    rotor_axis: str = 'horizontal',
) -> float
```

Slant distance `R1` from the rotor centre to the ground microphone.

With the reference microphone on the ground at the horizontal distance
`R0` and the rotor centre at height `H`, the slant distance is
`R1 = sqrt(H² + R0²)`. For a horizontal-axis turbine `R0 = H + D/2`
(IEC 61400-11 Formula 1); for a vertical-axis turbine `R0 = H + D`
(Formula 2, with `H` the height of the equator of the rotor).

**Parameters**

| Name | Description |
| :--- | :--- |
| `hub_height` | Hub height `H` (ground to rotor centre; for a vertical-axis turbine, to the rotor equator), in m. |
| `rotor_diameter` | Rotor diameter `D`, in m. |
| `rotor_axis` | `"horizontal"` (Formula 1, the default) or `"vertical"` (Formula 2). |

**Returns:** The slant distance `R1`, in m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a dimension is not positive or `rotor_axis` is not one of the two orientations. |

## wind_turbine_tonality

```python
wind_turbine_tonality(
    levels: NDArray[np.float64] | list[float],
    frequencies: NDArray[np.float64] | list[float],
    *,
    tone_frequency: float | None = None,
) -> WindTurbineTonalityResult
```

Tonal audibility of a narrowband spectrum (IEC 61400-11 Formulae 30-34).

From a uniformly-spaced narrowband spectrum, screens the candidate as a
possible tone (subclause 9.5.2: a local maximum more than 6 dB above the
critical-band energy average excluding the maximum and its two adjacent
lines), classifies the lines in the critical band about the candidate into
masking noise and tone lines, forms the masking-noise level `L_pn`
(Formula 31), the tonality `ΔL_tn = L_pt − L_pn` (Formula 32), the
audibility criterion `L_a` (Formula 34) and the tonal audibility
`ΔL_a = ΔL_tn − L_a` (Formula 33).

Per subclause 9.5.3 the tone lines are those above `L_pn,avg + 6 dB`
*and* within 10 dB of the highest such line; the frequency of the tone is
that highest line (9.5.4), which also anchors `L_a`. When the candidate
fails the 9.5.2 screening or no line classifies as "tone", the result
carries `has_identified_tone = False`: its numeric fields are
non-standard fallbacks and the spectrum must be excluded from the 9.5.1
bin averaging (see [`WindTurbineTonalityResult`](/phonometry/reference/api/aeroacoustics/wind-turbine-noise/#windturbinetonalityresult)).

Spectra must extend over the whole critical band: a truncated band leaves
the masking average over the surviving lines while Formula 31 still scales
to the full bandwidth, so a [`WindTurbineNoiseWarning`](/phonometry/reference/api/aeroacoustics/wind-turbine-noise/#windturbinenoisewarning) is issued.
Candidates below 20 Hz are outside the standard's analysis range (the
critical band would extend to negative frequencies) and are rejected.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Narrowband line levels, in dB (A-weighted, 1-2 Hz resolution). |
| `frequencies` | Line frequencies, in Hz (uniform spacing). |
| `tone_frequency` | Candidate tone frequency, in Hz; if `None` the highest-level line is used. |

**Returns:** A [`WindTurbineTonalityResult`](/phonometry/reference/api/aeroacoustics/wind-turbine-noise/#windturbinetonalityresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or the candidate lies below 20 Hz. |

## WindTurbineNoiseWarning

The tonality inputs leave the standard's stated domain of validity.

## WindTurbineTonalityResult

```python
WindTurbineTonalityResult(
    tone_frequency: float,
    critical_bandwidth: float,
    tone_level: float,
    masking_level: float,
    tonality: float,
    audibility_criterion: float,
    tonal_audibility: float,
    is_audible: bool,
    has_identified_tone: bool,
    frequencies: NDArray[np.float64],
    levels: NDArray[np.float64],
)
```

Tonal audibility of a narrowband spectrum (IEC 61400-11).

**Attributes**

| Name | Description |
| :--- | :--- |
| `tone_frequency` | The frequency of the identified tone: the spectral line with the highest level among the lines classified as "tone" (subclause 9.5.4; also the `f` of Formula 34). When no tone is identified (`has_identified_tone` is `False`) this falls back to the candidate line. |
| `critical_bandwidth` | The critical bandwidth about the candidate, Hz. |
| `tone_level` | Tone level `L_pt` (energy sum of the tone lines), in dB. |
| `masking_level` | Masking-noise level `L_pn`, in dB. |
| `tonality` | Tonality `ΔL_tn = L_pt − L_pn`, in dB. |
| `audibility_criterion` | The criterion `L_a` (Formula 34), in dB. |
| `tonal_audibility` | Tonal audibility `ΔL_a = ΔL_tn − L_a`, in dB. |
| `is_audible` | Whether an identified tone is audible (`ΔL_a > 0` *and* `has_identified_tone`). |
| `has_identified_tone` | Whether the candidate passed the 9.5.2 possible-tone screening *and* at least one spectral line was classified as "tone" (subclause 9.5.4). When `False` the numeric fields are non-standard fallbacks (the standard defines no tonality for such a spectrum) and the spectrum must be **excluded** from the 9.5.1 energy averaging of `ΔL_a,j,k` over the spectra of a bin. |
| `frequencies` | The narrowband line frequencies, in Hz. |
| `levels` | The narrowband line levels, in dB. |

### WindTurbineTonalityResult.plot()

```python
WindTurbineTonalityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the narrowband spectrum with the critical band and masking level.

### WindTurbineTonalityResult.report()

```python
WindTurbineTonalityResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a wind-turbine tonal audibility assessment fiche to a PDF.

Writes a one-page tonality-assessment report following
IEC 61400-11:2012+A1:2018 (subclauses 9.5.2 to 9.5.8): the standard-basis
line, an optional metadata header (source/situation, client, measurement
position, instrumentation and date), a two-panel body with the
critical-band / masking analysis in a metrics table (tone frequency,
critical bandwidth, tone level `L_pt`, masking-noise level `L_pn`,
tonality `ΔL_tn`, audibility criterion `L_a` and tonal audibility
`ΔL_a`) beside the narrowband-spectrum plot with the critical band,
masking level and tone marked, the boxed decisive tonal audibility
`ΔL_a` and the tone frequency with the audibility decision, an optional
verdict row and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare assessment fiche (body, result and disclaimer only). A supplied `requirement` is read as the maximum acceptable tonal audibility `ΔL_a` in dB (a lower audibility passes). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for signature parity with the other fiches; the metrics table already shows the full Formula 30-34 chain, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `language` is not one of the supported languages, or if `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |
