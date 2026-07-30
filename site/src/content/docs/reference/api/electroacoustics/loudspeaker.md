---
title: "electroacoustics.loudspeaker"
description: "Rated loudspeaker characteristics (IEC 60268-5)."
sidebar:
  label: "loudspeaker"
---

Rated loudspeaker characteristics (IEC 60268-5).

A loudspeaker measurement/rating report gathers the *rated characteristics*
IEC 60268-5:2003+A1:2007 defines around a measured on-axis response: the
**characteristic sensitivity** referred to 1 W input at 1 m (clauses 20.3/20.4),
the **effective frequency range** read from the on-axis response against a
-10 dB band (clause 21.2), the **rated impedance** and its modulus curve
(clause 16), the **total harmonic distortion** against frequency (clause 24.1)
and the **directional / polar response** (clause 23). This module bundles those
into a single [`LoudspeakerCharacteristics`](/phonometry/reference/api/electroacoustics/loudspeaker/#loudspeakercharacteristics) result whose `report` method
renders the IEC 60268-5 rated-characteristics fiche, with the characteristic
graphs laid out to the IEC 60263:1982 scale conventions.

Two of the rated characteristics are *computed* from the on-axis response so
the report never merely repeats a manufacturer number:

* **Characteristic sensitivity level** (20.3/20.4). The on-axis response is
  measured at a constant voltage `U` and distance `d`; the sensitivity
  level referred to 1 W into the rated impedance `R` at 1 m is

      L_M = L_band + 20 lg(d / d0) + 20 lg(U_p / U),   d0 = 1 m,

  where `L_band` is the energetic mean of the on-axis level over a stated
  band (20.1.2.4: the r.m.s. of the band pressures) and `U_p = sqrt(R * P0)`
  with `P0 = 1 W` is the voltage that drives 1 W into `R` (20.3.2). With the
  default drive `U = sqrt(R)` at `d = 1 m` the two corrections vanish and
  the sensitivity level equals the band mean, which for `R = 8` ohm is the
  familiar "dB / 2.83 V @ 1 m" figure. This is the clean-room oracle: a flat
  `L0` response driven at `sqrt(R)` volts and 1 m returns `L0` exactly,
  and a doubled voltage returns `L0 - 6,02` dB.

* **Effective frequency range** (21.2). The range of frequencies for which the
  on-axis response is not more than 10 dB below the level averaged over a
  one-octave band in the region of maximum sensitivity. Troughs narrower than
  1/9 octave at the -10 dB level are neglected. The band edges are the
  interpolated frequencies where the response last crosses the -10 dB
  threshold on either side of the peak, which is the second clean-room oracle:
  a response crossing the threshold at chosen frequencies returns exactly
  those frequencies.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## loudspeaker_characteristics

```python
loudspeaker_characteristics(
    frequencies: ArrayLike,
    spl_db: ArrayLike,
    rated_impedance: float,
    *,
    input_voltage: float | None = None,
    distance: float = 1.0,
    sensitivity_band: tuple[float, float] | None = None,
    tolerance_db: float = 3.0,
    rated_frequency_range: tuple[float, float] | None = None,
    rated_noise_power: float | None = None,
    rated_sinusoidal_power: float | None = None,
    resonance_frequency: float | None = None,
    impedance: tuple[ArrayLike, ArrayLike] | None = None,
    distortion: SweptSineDistortionResult | tuple[ArrayLike, ArrayLike] | None = None,
    directivity: RadiatingPistonResult | None = None,
    polar: tuple[ArrayLike, ArrayLike] | None = None,
    polar_frequency: float | None = None,
    directivity_index_db: float | None = None,
) -> LoudspeakerCharacteristics
```

Assemble the rated loudspeaker characteristics for an IEC 60268-5 report.

The characteristic sensitivity level (20.3/20.4) and the effective frequency
range (21.2) are computed from the on-axis response; the optional impedance,
distortion and directivity data feed the corresponding report panels.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | On-axis response frequency axis, in Hz (1-D, > 0). Logarithmically spaced samples are strongly recommended: the band averages behind the sensitivity level and the effective-range reference weight each sample equally, so a linearly spaced grid over-weights the high-frequency end of every band. |
| `spl_db` | On-axis sound pressure level, in dB re 20 uPa. |
| `rated_impedance` | Rated impedance `R`, in ohm (16.1). |
| `input_voltage` | Constant drive voltage of the response, in V; defaults to `sqrt(R)` (1 W into `R`, the 2,83 V @ 8 ohm convention). |
| `distance` | Measuring distance of the response, in m (default 1). |
| `sensitivity_band` | Stated band `(lo, hi)` for the characteristic sensitivity, in Hz; defaults to the one-octave band in the region of maximum sensitivity. |
| `tolerance_db` | Half-width of the plotted response tolerance band, in dB (default 3). |
| `rated_frequency_range` | Manufacturer-stated rated frequency range `(lo, hi)` in Hz (19.1). When supplied it is also the range over which the 16.1 minimum impedance modulus is evaluated. |
| `rated_noise_power` | Rated noise power, in W (18.1). |
| `rated_sinusoidal_power` | Rated sinusoidal power, in W (18.4). |
| `resonance_frequency` | Resonance frequency, in Hz (19.2). |
| `impedance` | Impedance curve as `(frequencies, modulus)` in `(Hz, ohm)` (16.2). |
| `distortion` | THD against frequency, either as a [`SweptSineDistortionResult`](/phonometry/reference/api/electroacoustics/swept-sine/#sweptsinedistortionresult) (its `thd` ratio is converted to %) or a `(frequencies, thd_percent)` pair (24.1). |
| `directivity` | A [`RadiatingPistonResult`](/phonometry/reference/api/electroacoustics/piston/#radiatingpistonresult) computed with `angles` to supply the polar response and directivity index (23.1/23.3). |
| `polar` | Polar response as `(angles_deg, relative_db)` when it is not taken from `directivity`. |
| `polar_frequency` | Frequency of the polar response, in Hz. |
| `directivity_index_db` | Directivity index at `polar_frequency`, in dB (23.3), when not taken from `directivity`. |

**Returns:** A [`LoudspeakerCharacteristics`](/phonometry/reference/api/electroacoustics/loudspeaker/#loudspeakercharacteristics).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## LoudspeakerCharacteristics

```python
LoudspeakerCharacteristics(
    frequencies: NDArray[np.float64],
    spl_db: NDArray[np.float64],
    rated_impedance: float,
    input_voltage: float,
    distance: float,
    sensitivity_band: tuple[float, float],
    tolerance_db: float,
    reference_level_db: float,
    sensitivity_level_db: float,
    effective_range: tuple[float, float],
    rated_frequency_range: tuple[float, float] | None,
    rated_noise_power: float | None,
    rated_sinusoidal_power: float | None,
    resonance_frequency: float | None,
    impedance_frequencies: NDArray[np.float64] | None,
    impedance_modulus: NDArray[np.float64] | None,
    thd_frequencies: NDArray[np.float64] | None,
    thd_percent: NDArray[np.float64] | None,
    polar_angles_deg: NDArray[np.float64] | None,
    polar_db: NDArray[np.float64] | None,
    polar_frequency: float | None,
    directivity_index_db: float | None,
)
```

Rated loudspeaker characteristics for an IEC 60268-5 report.

The on-axis response and the rated impedance are the required inputs; the
impedance modulus curve, the total-harmonic-distortion curve and the
directional/polar response are optional panels rendered when supplied. The
characteristic sensitivity level and the effective frequency range are
computed from the on-axis response (see the module docstring).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | On-axis response frequency axis, in Hz. |
| `spl_db` | On-axis sound pressure level (re 20 uPa), in dB, measured at `input_voltage` and `distance`. |
| `rated_impedance` | Rated impedance `R`, in ohm (16.1). |
| `input_voltage` | Constant drive voltage of the response, in V. |
| `distance` | Measuring distance of the response, in m. |
| `sensitivity_band` | Stated band `(lo, hi)` for the characteristic sensitivity, in Hz. |
| `tolerance_db` | Half-width of the response tolerance band, in dB. |
| `reference_level_db` | Level averaged over a one-octave band in the region of maximum sensitivity (the effective-range reference), in dB. |
| `sensitivity_level_db` | Characteristic sensitivity level referred to 1 W into `R` at 1 m (20.3/20.4), in dB. |
| `effective_range` | Computed effective frequency range `(lo, hi)`, Hz. |
| `rated_frequency_range` | Manufacturer-stated rated frequency range `(lo, hi)` in Hz (19.1), or `None`. |
| `rated_noise_power` | Rated noise power, in W (18.1), or `None`. |
| `rated_sinusoidal_power` | Rated sinusoidal power, in W (18.4), or `None`. |
| `resonance_frequency` | Resonance frequency, in Hz (19.2), or `None`. |
| `impedance_frequencies` | Impedance-curve frequency axis, Hz, or `None`. |
| `impedance_modulus` | Impedance modulus `\|Z\|`, ohm, or `None`. |
| `thd_frequencies` | THD-curve frequency axis, Hz, or `None`. |
| `thd_percent` | Total harmonic distortion, in %, or `None`. |
| `polar_angles_deg` | Polar-response angles, in degrees, or `None`. |
| `polar_db` | Polar response relative to the on-axis level, in dB, or `None`. |
| `polar_frequency` | Frequency of the polar response, in Hz, or `None`. |
| `directivity_index_db` | Directivity index at `polar_frequency`, in dB (23.3), or `None`. |

### LoudspeakerCharacteristics.characteristic_sensitivity_pa

*property*

Characteristic sensitivity as a pressure, in Pa (20.3).

The sound pressure at 1 m for 1 W into the rated impedance:
`p_M = p_ref * 10 ** (L_M / 20)`.

### LoudspeakerCharacteristics.minimum_impedance

*property*

Lowest impedance modulus over the rated range, ohm (16.1), or `None`.

IEC 60268-5 16.1 requires the lowest value of the impedance modulus
*in the rated frequency range* to be not less than 80 % of the rated
impedance, so the scan uses `rated_frequency_range` when it is
supplied. When no rated range is stated, the computed
`effective_range` stands in for it; note that the two ranges may
differ (19.1 NOTE 2), particularly for tweeters or woofers, so an
impedance dip outside the effective range is only caught when the
rated range is given.

### LoudspeakerCharacteristics.plot()

```python
LoudspeakerCharacteristics.plot(
    quantity: str = 'response',
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot one IEC 60268-5 loudspeaker rated characteristic on one axes.

One concept per figure, drawn by the same shared renderer the
`.report()` fiche composes: `"response"` (the on-axis SPL response
with its tolerance band and effective-range markers, the default),
`"impedance"` (the modulus `|Z|` with the rated and 80 %-of-rated
lines), `"thd"` (total harmonic distortion against frequency) and
`"directivity"` (the polar response on the 25 dB reference circle).

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

### LoudspeakerCharacteristics.report()

```python
LoudspeakerCharacteristics.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render the IEC 60268-5 loudspeaker-characteristics fiche to a PDF.

Writes a one-page rated-characteristics data sheet: the standard-basis
line (IEC 60268-5:2003+A1:2007, graphs to IEC 60263:1982), an optional
metadata header, the rated-characteristics table beside the on-axis
response with its tolerance band and effective-range markers, the
impedance, total-harmonic-distortion and polar-directivity panels for
the data supplied, a boxed sensitivity/range result, an optional
sensitivity verdict when a requirement is given, and the footer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity and, through `requirement`, a characteristic sensitivity level (dB) the verdict row compares against. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the fiche has one layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |
