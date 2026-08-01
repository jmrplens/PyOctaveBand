---
title: "underwater.marine_mammal_weighting"
description: "Regulatory auditory weighting and exposure criteria for marine mammals."
sidebar:
  label: "marine_mammal_weighting"
---

Regulatory auditory weighting and exposure criteria for marine mammals.

Noise-exposure assessments weight a spectrum by a hearing-group filter before
comparing it with a threshold. The filter is the same band-pass form in all
current guidance (NMFS 2018 Equation 1, Southall et al. 2019 Equation 2):

$$
W(f) = C + 10\,\log_{10}\frac{(f/f_1)^{2a}} {[1+(f/f_1)^2]^{a}\,[1+(f/f_2)^2]^{b}}
$$

with `f` in kilohertz. `C` is fixed by putting the peak of `W` at 0 dB,
so the companion **exposure function**
$E(f) = K - 10 \log_{10}(\dots) = K + C - W(f)$
has its minimum at the weighted threshold $T_w = K + C$. Only the
parameter table changes between guidance versions, so the version is explicit
in the API and is carried on every result object:

* `"nmfs-2024"` -- NOAA Fisheries, *Updated Technical Guidance*, version 3.0
  (October 2024), Table 5 and Table ES3. **The default**: it supersedes the
  2018 revision, uses $b = 5$ for every group, renames the groups to
  the
  Southall scheme (LF/HF/VHF cetaceans, PW/OW in water, PA/OA in air) and
  replaces "PTS onset" with "auditory injury (AUD INJ) onset".
* `"nmfs-2018"` -- the 2018 revision, version 2.0, Table 3 and Table ES3.
  Still cited by assessments already in flight.
* `"southall-2019"` -- Southall et al., *Aquatic Mammals* 45(2), Tables 5, 6
  and 7, the peer-reviewed criteria; adds sirenians (SI) and both in-air
  carnivore groups. Numerically identical to NMFS 2018 on the five shared
  groups.

**Group names are not portable between versions.** NMFS 2018 calls the
mid-frequency cetaceans `MF` and the porpoise-type group `HF`; NMFS 2024
and Southall call the same two `HF` and `VHF`. Each guidance version only
accepts its own codes.

The module exposes the weighting itself ([`auditory_weighting`](/phonometry/reference/api/underwater/marine-mammal-weighting/#auditory_weighting)), the
published thresholds ([`exposure_criteria`](/phonometry/reference/api/underwater/marine-mammal-weighting/#exposure_criteria)) and the assessment chain
([`weighted_exposure`](/phonometry/reference/api/underwater/marine-mammal-weighting/#weighted_exposure)), which weights a band spectrum, accumulates it over
a number of events and reports the exceedance of each applicable criterion.

Implemented clean-room from the three documents; validated against the worked
example of NMFS (2018) Appendix D ($W(1~\text{kHz})$ for the five
groups), against `C` recomputed as the peak of `W` for all three
parameter sets, and against the published $T_w = K + C$ and
injury = TTS + 20 dB identities.

:::note
NMFS (2024) Table 5 prints $C = 1.37$ dB for otariid pinnipeds in
water, and its own footnote states the value should be 1.36 dB (NMFS kept
1.37 for consistency with the U.S. Navy). Recomputing `C` from the peak
of `W`
with that row's parameters gives 1.3643 dB, confirming 1.36. This module
implements **1.36**; the printed 1.37 remains available as
`WeightingParameters.c_db_as_printed`. See `docs/ERRATA.md`.
:::

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## auditory_weighting

```python
auditory_weighting(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    group: str,
    *,
    guidance: str = 'nmfs-2024',
) -> AuditoryWeightingResult
```

Auditory weighting function `W(f)` of a marine-mammal hearing group.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Frequency or frequencies, in Hz (strictly positive). |
| `group` | Hearing-group code as used by `guidance`. |
| `guidance` | `"nmfs-2024"` (default, current), `"nmfs-2018"` or `"southall-2019"`. |

**Returns:** An [`AuditoryWeightingResult`](/phonometry/reference/api/underwater/marine-mammal-weighting/#auditoryweightingresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

## AuditoryWeightingResult

```python
AuditoryWeightingResult(
    frequencies: NDArray[np.float64],
    weighting: NDArray[np.float64],
    exposure_function: NDArray[np.float64],
    parameters: WeightingParameters,
    guidance: str,
    group: str,
    weighted_tts_onset: float,
)
```

Auditory weighting and exposure functions of one hearing group.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in Hz. |
| `weighting` | Weighting-function amplitude `W(f)`, in dB ($\le 0$). |
| `exposure_function` | Exposure function $E(f) = K + C - W(f)$, in dB (the frequency-dependent TTS-onset level). |
| `parameters` | The [`WeightingParameters`](/phonometry/reference/api/underwater/marine-mammal-weighting/#weightingparameters) used. |
| `guidance` | The guidance version. |
| `group` | Hearing-group code. |
| `weighted_tts_onset` | $T_w = K + C$, the minimum of the exposure function, in dB. |

### AuditoryWeightingResult.plot()

```python
AuditoryWeightingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the weighting function versus frequency.

## exposure_criteria

```python
exposure_criteria(
    group: str,
    *,
    guidance: str = 'nmfs-2024',
    impulsive: bool = False,
) -> ExposureCriteria
```

Published TTS and injury onset criteria of a hearing group.

**Parameters**

| Name | Description |
| :--- | :--- |
| `group` | Hearing-group code as used by `guidance`. |
| `guidance` | `"nmfs-2024"` (default), `"nmfs-2018"` or `"southall-2019"`. |
| `impulsive` | Return the impulsive-noise criteria (dual metric: a weighted SEL and an unweighted peak SPL) instead of the non-impulsive ones. |

**Returns:** An [`ExposureCriteria`](/phonometry/reference/api/underwater/marine-mammal-weighting/#exposurecriteria).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the version or the group is unknown. |

## ExposureCriteria

```python
ExposureCriteria(
    group: str,
    guidance: str,
    impulsive: bool,
    tts_sel: float | None,
    injury_sel: float | None,
    tts_peak_spl: float | None,
    injury_peak_spl: float | None,
    injury_label: str,
    sel_reference: str,
    peak_reference: str,
    source: str,
)
```

Published TTS and injury (PTS / AUD INJ) onset criteria for one group.

Sound exposure levels are weighted; peak sound pressure levels are
unweighted ("flat"), as every source states. `None` means the criterion
is not published by that guidance version.

**Attributes**

| Name | Description |
| :--- | :--- |
| `group` | Hearing-group code. |
| `guidance` | The guidance version. |
| `impulsive` | Whether these are the impulsive-noise criteria. |
| `tts_sel` | Weighted TTS-onset sound exposure level, in dB. |
| `injury_sel` | Weighted injury-onset (PTS / AUD INJ) SEL, in dB. |
| `tts_peak_spl` | Unweighted TTS-onset peak SPL, in dB. |
| `injury_peak_spl` | Unweighted injury-onset peak SPL, in dB. |
| `injury_label` | `"PTS"` or `"AUD INJ"`, as the source names it. |
| `sel_reference` | Human-readable SEL reference of the group. |
| `peak_reference` | Human-readable peak-SPL reference of the group. |
| `source` | Table the numbers come from. |

## hearing_groups

```python
hearing_groups(guidance: str = 'nmfs-2024') -> tuple[str, ...]
```

Hearing-group codes defined by a guidance version.

**Parameters**

| Name | Description |
| :--- | :--- |
| `guidance` | One of [`WEIGHTING_GUIDANCE`](/phonometry/reference/api/underwater/marine-mammal-weighting/#weighting_guidance). |

**Returns:** The group codes, in the order the source tabulates them.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the version is unknown. |

## weighted_exposure

```python
weighted_exposure(
    frequency_hz: NDArray[np.float64] | list[float],
    band_sel: NDArray[np.float64] | list[float],
    group: str,
    *,
    guidance: str = 'nmfs-2024',
    impulsive: bool = True,
    n_events: int = 1,
    peak_spl: float | None = None,
) -> WeightedExposureResult
```

Weight a band spectrum, accumulate it and compare it with the
criteria.

The per-band single-event sound exposure levels are weighted with
[`auditory_weighting`](/phonometry/reference/api/underwater/marine-mammal-weighting/#auditory_weighting), summed on an energy basis and accumulated over
`n_events` identical events ($+10 \log_{10} N$, the ISO 18406 Formula 9
identity used by [`cumulative_sel_identical`](/phonometry/reference/api/underwater/pile-driving-noise/#cumulative_sel_identical)).
The result is compared with the group's TTS and injury onset criteria; the
peak sound pressure level, if supplied, is compared **unweighted**, as the
dual-metric rule requires.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Band centre frequencies, in Hz (1-D, positive). |
| `band_sel` | Per-band single-event SEL, in dB re 1 µPa²·s (or dB re (20 µPa)²·s for an in-air group); same length. `-inf` is accepted for a band that carries no energy, which is what [`strike_sel_spectrum`](/phonometry/reference/api/underwater/pile-driving-noise/#strike_sel_spectrum) returns for bands narrower than its FFT bin spacing; such a band adds nothing to the energy sum. Both input arrays are copied, so the result never aliases the caller's data. |
| `group` | Hearing-group code as used by `guidance`. |
| `guidance` | `"nmfs-2024"` (default), `"nmfs-2018"` or `"southall-2019"`. |
| `impulsive` | Compare against the impulsive criteria (the default, the case for pile driving and air guns). |
| `n_events` | Number of identical accumulated events, $\ge 1$. |
| `peak_spl` | Unweighted zero-to-peak sound pressure level of the loudest single event, in dB; enables the peak-SPL half of the dual metric. |

**Returns:** A [`WeightedExposureResult`](/phonometry/reference/api/underwater/marine-mammal-weighting/#weightedexposureresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

## WeightedExposureResult

```python
WeightedExposureResult(
    frequencies: NDArray[np.float64],
    band_sel: NDArray[np.float64],
    weighting: NDArray[np.float64],
    weighted_band_sel: NDArray[np.float64],
    unweighted_sel: float,
    weighted_sel: float,
    cumulative_sel: float,
    peak_spl: float | None,
    n_events: int,
    criteria: ExposureCriteria,
    sel_margin: float | None,
    tts_margin: float | None,
    peak_margin: float | None,
    tts_peak_margin: float | None,
    exceeds_injury: bool,
    exceeds_tts: bool,
    guidance: str,
    group: str,
)
```

Weighted exposure of a spectrum against a hearing group's criteria.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz. |
| `band_sel` | Per-band single-event sound exposure level, in dB. |
| `weighting` | Weighting-function amplitude at each band, in dB. |
| `weighted_band_sel` | `band_sel + W(f)` per band, in dB. |
| `unweighted_sel` | Energy sum of `band_sel`, in dB. |
| `weighted_sel` | Energy sum of `weighted_band_sel`, in dB. |
| `cumulative_sel` | `weighted_sel` plus $10 \log_{10}(N)$ for the `n_events` accumulated events, in dB. |
| `peak_spl` | The unweighted peak sound pressure level supplied, in dB (`None` when not given). |
| `n_events` | Number of accumulated events (e.g. hammer strikes). |
| `criteria` | The [`ExposureCriteria`](/phonometry/reference/api/underwater/marine-mammal-weighting/#exposurecriteria) compared against. |
| `sel_margin` | `cumulative_sel - injury_sel`, in dB (`None` when the criterion is not published); positive means the criterion is exceeded. |
| `tts_margin` | `cumulative_sel - tts_sel`, in dB (or `None`). |
| `peak_margin` | `peak_spl - injury_peak_spl`, in dB (or `None`). |
| `tts_peak_margin` | `peak_spl - tts_peak_spl`, in dB (or `None`) -- the peak-SPL half of the dual metric on the TTS side, which can trip `exceeds_tts` on its own. |
| `exceeds_injury` | Whether any injury-onset criterion is reached. The test is `margin >= 0`, so an exposure landing exactly **on** the criterion counts as exceeding it; the criteria are onset thresholds and the precautionary reading is the one an assessment wants. |
| `exceeds_tts` | Whether any TTS-onset criterion is reached, on the same `margin >= 0` convention as `exceeds_injury`. |
| `guidance` | The guidance version. |
| `group` | Hearing-group code. |

### WeightedExposureResult.plot()

```python
WeightedExposureResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the unweighted and weighted band spectra with the criteria.

## WEIGHTING_GUIDANCE

*Constant* (`tuple`).

```python
WEIGHTING_GUIDANCE = ('nmfs-2024', 'nmfs-2018', 'southall-2019')
```

## weighting_parameters

```python
weighting_parameters(
    group: str,
    *,
    guidance: str = 'nmfs-2024',
) -> WeightingParameters
```

Weighting/exposure parameters of one hearing group.

**Parameters**

| Name | Description |
| :--- | :--- |
| `group` | Hearing-group code as used by `guidance` (case-insensitive). |
| `guidance` | One of [`WEIGHTING_GUIDANCE`](/phonometry/reference/api/underwater/marine-mammal-weighting/#weighting_guidance). |

**Returns:** The [`WeightingParameters`](/phonometry/reference/api/underwater/marine-mammal-weighting/#weightingparameters) row.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the version or the group is unknown. |

## WeightingParameters

```python
WeightingParameters(
    group: str,
    guidance: str,
    description: str,
    a: float,
    b: float,
    f1_khz: float,
    f2_khz: float,
    c_db: float,
    c_db_as_printed: float,
    k_db: float,
    in_air: bool,
    hearing_range_hz: tuple[float, float] | None,
)
```

Auditory weighting and exposure function parameters for one group.

**Attributes**

| Name | Description |
| :--- | :--- |
| `group` | Hearing-group code as used by its own guidance version. |
| `guidance` | The guidance version the row comes from. |
| `description` | Plain-language name of the hearing group. |
| `a` | Low-frequency exponent `a`. |
| `b` | High-frequency exponent `b`. |
| `f1_khz` | Low-frequency transition `f1`, in kHz. |
| `f2_khz` | High-frequency transition `f2`, in kHz. |
| `c_db` | Gain `C` that puts the peak of `W` at 0 dB, in dB. |
| `c_db_as_printed` | `C` exactly as printed in the source table, in dB (differs from `c_db` only for the NMFS 2024 otariid row). |
| `k_db` | Exposure-function constant `K`, in dB. |
| `in_air` | Whether the group's reference is 20 µPa (in air). |
| `hearing_range_hz` | Generalised hearing range of the group, in Hz, or `None`. Only the NMFS documents tabulate one (their Table ES1); Southall et al. do not, and the field is `None` for those rows rather than borrowed from elsewhere. |
