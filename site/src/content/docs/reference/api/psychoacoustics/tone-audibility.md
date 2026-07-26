---
title: "psychoacoustics.tone_audibility"
description: "Objective audibility of tones in noise -- engineering method (ISO/PAS 20065:2016)."
sidebar:
  label: "tone_audibility"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Objective audibility of tones in noise -- engineering method (ISO/PAS 20065:2016).

ISO/PAS 20065 is the detailed engineering method that ISO 1996-2:2017 defers to
for the audibility of prominent tones; the simplified 2007/2009 Annex C method
lives in [`phonometry.environmental_measurement`](/phonometry/reference/api/environment/measurement/). The audibility of a tone
is the amount, in decibels, by which its tone level rises above the masking
threshold of the surrounding noise.

**Critical band about the tone (Clause 5.2).** The width of the critical band
around a tone of frequency `fT` is
`Δfc = 25.0 + 75.0·(1.0 + 1.4·(fT/1000)²)^0.69` Hz (Formula (2)). Assuming a
geometric placement of the corner frequencies, `fT = √(f1·f2)` (Formula (3)),
`f1 = −Δfc/2 + √(Δfc² + 4·fT²)/2` (Formula (4)) and `f2 = f1 + Δfc`
(Formula (5)).

**Audibility of a single tone (Clause 5.3).** From the mean narrow-band level
`LS` of the masking noise (Formula (6)) the critical-band level of the masking
noise is `LG = LS + 10·lg(Δfc/Δf)` (Formula (12), `Δf` the line spacing).
The masking index is `av = −2 − lg[1 + (f/502)²·⁵]` dB (Formula (13)) and the
audibility of a tone of level `LT` (Formula (8)) is
`ΔL = LT − LG − av` dB (Formula (14)). A tone is present when `ΔL > 0`.

**Decisive and mean audibility (Clauses 5.3.8/5.3.9).** The decisive audibility
of one narrow-band spectrum is the largest tone audibility in it (Step 4). Over
`J` staggered spectra the mean audibility is the energy mean
`ΔL = 10·lg[(1/J)·Σ 10^(ΔLj/10)]` dB (Formula (20)); a spectrum in which no
tone is found contributes `ΔLj = −10 dB` (Formula (21)).

**From a critical-band spectrum.** [`mean_narrowband_level`](/phonometry/reference/api/psychoacoustics/tone-audibility/#mean_narrowband_level) determines
`LS` from the lines of the critical band by the iterative procedure of
Formula (6)/Annex D (energy average, dropping any line more than 6 dB above the
running mean, with the −1.76 dB Hanning bandwidth correction), and
[`tone_level`](/phonometry/reference/api/psychoacoustics/tone-audibility/#tone_level) sums the tonal lines contiguous with the peak for `LT`
(Formula (8)). Both reproduce the ISO/PAS 20065 Annex E worked example
(`LS = 49.22 dB`, `LT = 67.96 dB` for the 137.3 Hz tone) and are confirmed
against the parent standard DIN 45681:2005-03.

**Whole-spectrum detection.** [`analyze_spectrum`](/phonometry/reference/api/psychoacoustics/tone-audibility/#analyze_spectrum) runs the full front-end
over a spectrum (mean narrow-band level per line, peak detection (Clause 5.3.8
Step 1), tone level, the distinctness test (Clause 5.3.4) and audibility) and
returns the distinct, audible tones. It then applies Step 3: tones sharing a
critical band have their tone levels energy-summed (Formula (17), via
[`combined_tone_level`](/phonometry/reference/api/psychoacoustics/tone-audibility/#combined_tone_level), shared lines counted once) into an "FG" entry
rated at the most audible member, unless the exactly-two-tones-below-1000-Hz
exception (Formulae (18)/(19)) keeps them separate. On the Annex E example
this recovers the three tones, their combined tone level `LT = 72.15 dB`
and the decisive FG audibility `ΔL = 9.18 dB`. A decisive audibility
reproduced exactly needs the *complete* narrow-band spectrum: a spectrum
truncated to one critical band mis-estimates the mean narrow-band level of
tones near its edges.

**Two tones below 1000 Hz.** When *exactly two* tones share a critical band and
both lie below 1000 Hz, the ear can still resolve them if their spacing exceeds
`fD = 21·10^(1.2·|lg(fT/212)|^1.8)` Hz (Formulae (18)/(19)); they are then
rated separately instead of combined. [`two_tone_separation_frequency`](/phonometry/reference/api/psychoacoustics/tone-audibility/#two_tone_separation_frequency) and
[`resolve_tones_separately`](/phonometry/reference/api/psychoacoustics/tone-audibility/#resolve_tones_separately) implement this branch (Clause 5.3.8), which no
ISO/PAS 20065 worked example exercises; it is verified against the DIN 45681
Annex J reference program rather than a numeric oracle.

**Uncertainty (Clauses 5.4/6).** [`audibility_uncertainty`](/phonometry/reference/api/psychoacoustics/tone-audibility/#audibility_uncertainty) propagates the
uniform 3 dB narrow-band level uncertainty through the audibility chain to the
extended uncertainty `U` (90 % bilateral coverage), and
[`mean_audibility_uncertainty`](/phonometry/reference/api/psychoacoustics/tone-audibility/#mean_audibility_uncertainty) combines the per-spectrum values through
the Formula (20) mean. Clause 6: when fewer than 12 spectra have been
averaged, the extended uncertainty **shall** be taken into consideration.
[`analyze_spectrum`](/phonometry/reference/api/psychoacoustics/tone-audibility/#analyze_spectrum) reports the per-tone `U` on its result.

**A-weighting.** Clause 5.3.2: unweighted narrow-band spectra "shall" be
A-weighted per IEC 61672-1 before the analysis. This module is
weighting-agnostic: pass A-weighted levels (the Annex E oracles are
A-weighted); it does not apply the weighting itself.

**Application frequency range.** The functions accept any positive tone
frequency, but the standards state narrower ranges: DIN 45681:2005-03 (5.3.2)
restricts the method to `fT >= 90 Hz` and the ISO/PAS 20065 scope starts at
50 Hz; the two-tone separation frequency (Formula (19)) is printed for
`fT < 1000 Hz` (with a lower bound of 88 Hz in the DIN print, 50 Hz in the
ISO one). Results outside these ranges are extrapolations.

**Distinctness edge steepness (DIN-vs-ISO print difference).** The 5.3.4
edge-steepness test follows the DIN 45681 `fT/sqrt(2)`-on-both-edges
reading, matching its executable Annex J reference program; the ISO/PAS
20065 print shows asymmetric formulas that contradict it (see
`_is_distinct` and docs/ERRATA.md).

## analyze_spectrum

```python
analyze_spectrum(
    levels: ArrayLike,
    frequencies: ArrayLike,
    line_spacing: float,
    *,
    effective_bandwidth_factor: float = 1.5,
) -> ToneAudibilityResult
```

Detect and rate the audible tones of a narrow-band spectrum (Clause 5.3.8).

Runs the full front-end: the mean narrow-band level (Formula (6)) per line,
peak detection (Step 1), the tone level (Formulae (7)/(8)), the distinctness
test (Clause 5.3.4) and the audibility (Formula (14)). Only distinct tones
with a positive audibility are returned, bundled by [`assess_tones`](/phonometry/reference/api/psychoacoustics/tone-audibility/#assess_tones).

**Same-band combination (Step 3).** When several audible tones fall in one
critical band, the clause *requires* their tone levels to be energy-summed
(Formula (17), shared lines counted once) and the audibility recomputed at
the frequency of the most audible member, unless *exactly two* tones below
1000 Hz are spaced further apart than the separation frequency `fD`
(Formulae (18)/(19)), in which case they stay rated separately. The result
therefore contains the individual audible tones *plus* one combined "FG"
entry per multi-tone critical band, mirroring the DIN 45681 Annex I tables;
[`ToneAudibilityResult.group_sizes`](/phonometry/reference/api/psychoacoustics/tone-audibility/#toneaudibilityresult) tells them apart (1 = single tone,
`N >= 2` = FG entry combining `N` tones). The decisive audibility
(Step 4) is the maximum over all entries, FG entries included.

Reproducing a decisive audibility exactly requires the *complete*
narrow-band spectrum; a spectrum truncated to a single critical band gives
the wrong mean narrow-band level for tones near its edges.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Narrow-band levels `Li` of the spectrum, in dB. |
| `frequencies` | The line frequencies, in Hz (strictly increasing). |
| `line_spacing` | Line spacing (frequency resolution) `Δf`, in Hz. |
| `effective_bandwidth_factor` | `Δfe/Δf`; 1.5 for a Hanning window (the default), 1.0 for a rectangular window. |

**Returns:** A [`ToneAudibilityResult`](/phonometry/reference/api/psychoacoustics/tone-audibility/#toneaudibilityresult) of the detected audible tones and their same-band FG combinations.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectrum is invalid or no audible tone is found. |

## assess_tones

```python
assess_tones(
    tone_frequencies: ArrayLike,
    tone_levels: ArrayLike,
    mean_narrowband_levels: ArrayLike,
    line_spacing: float,
    *,
    extended_uncertainties: ArrayLike | None = None,
) -> ToneAudibilityResult
```

Assess the audibility of the tones of a narrow-band spectrum.

Applies the critical band (Formulae (2)-(5)), critical-band level
(Formula (12)), masking index (Formula (13)) and audibility (Formula (14))
to every tone and bundles them into a plottable [`ToneAudibilityResult`](/phonometry/reference/api/psychoacoustics/tone-audibility/#toneaudibilityresult).

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_frequencies` | Tone frequencies `fT`, in Hz. |
| `tone_levels` | Tone levels `LT`, in dB (Formula (8)). |
| `mean_narrowband_levels` | Mean narrow-band levels `LS` of the masking noise, in dB (Formula (6)). |
| `line_spacing` | Line spacing (frequency resolution) `Δf`, in Hz. |
| `extended_uncertainties` | Optional per-tone extended uncertainties `U`, in dB (see [`audibility_uncertainty`](/phonometry/reference/api/psychoacoustics/tone-audibility/#audibility_uncertainty); computed automatically by [`analyze_spectrum`](/phonometry/reference/api/psychoacoustics/tone-audibility/#analyze_spectrum), which has the per-line levels this level-based entry point lacks). |

**Returns:** A [`ToneAudibilityResult`](/phonometry/reference/api/psychoacoustics/tone-audibility/#toneaudibilityresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the arrays are empty, differ in length, or contain non-finite/non-positive values. |

## audibility_from_levels

```python
audibility_from_levels(
    tone_level: float,
    critical_band_level: float,
    masking_index: float,
) -> float
```

Audibility `ΔL` from the levels and masking index (Formula (14)).

`ΔL = LT − LG − av` dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_level` | Tone level `LT`, in dB (Formula (8)). |
| `critical_band_level` | Critical-band level `LG` of the masking noise, in dB (Formula (12)). |
| `masking_index` | Masking index `av`, in dB (Formula (13)). |

**Returns:** Audibility `ΔL`, in dB (dB above the masking threshold).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any argument is not finite. |

## audibility_uncertainty

```python
audibility_uncertainty(
    tone_line_levels: ArrayLike,
    noise_line_levels: ArrayLike,
    tone_frequency: float,
    line_spacing: float,
    *,
    sigma_level_db: float = 3.0,
    coverage_factor: float = 1.645,
) -> float
```

Extended uncertainty `U` of one tone's audibility (Clause 6).

Gaussian propagation through Formula (14) with a uniform narrow-band
level uncertainty (Formulae (22)-(27) and (29)):

```text
sigma^2 = [sum(w_T^2)/sum(w_T)^2 + sum(w_S^2)/sum(w_S)^2] * sigma_L^2
          + (4.34 * df/dfc)^2
```

with `w = 10^(0.1 L)` over the `K`
tone-containing lines and the `M` noise lines of the final Formula (6)
iteration. For an FG group (several tones combined in one critical band,
Formula (17)) pass the `N` summated *tone levels* as
`tone_line_levels` -- that reading reproduces the printed Table E.2 FG
uncertainty (3.21 dB) where a union of the individual tonal lines does
not -- and the most audible tone's noise lines.
`U = k * sigma` with `k = 1.645` (90 % bilateral coverage). No
uncertainty is assumed for the masking index or the line spacing; the
critical-bandwidth term uses `sigma_dfc = df` (Formula (26)). The
DIN 45681 Annex J reference program computes the same quantity (its
`LT_Delta`/`Ls_Delta` accumulators), differing only in printing the
third term without the `df` factor -- both readings agree to well
under 0.01 dB on the Annex E example.

Clause 5.4 / Clause 6: if fewer than 12 spectra have been averaged, the
extended uncertainty of the (mean) audibility **shall** be taken into
consideration; see [`mean_audibility_uncertainty`](/phonometry/reference/api/psychoacoustics/tone-audibility/#mean_audibility_uncertainty).

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_line_levels` | Levels of the `K` tone-containing lines, dB. |
| `noise_line_levels` | Levels of the `M` masking-noise lines kept by the final Formula (6) iteration, dB. |
| `tone_frequency` | Tone frequency `fT`, in Hz. |
| `line_spacing` | Line spacing `df`, in Hz. |
| `sigma_level_db` | Standard uncertainty of each narrow-band level (Clause 6 assumes a uniform 3 dB). |
| `coverage_factor` | Coverage factor `k` (1.645 for 90 % bilateral). |

**Returns:** Extended uncertainty `U` of the audibility, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a line set is empty/non-finite or a parameter is not positive/finite. |

## combined_tone_level

```python
combined_tone_level(
    levels: ArrayLike,
    frequencies: ArrayLike,
    tone_frequencies: ArrayLike,
    mean_narrowband_levels: ArrayLike,
    *,
    effective_bandwidth_factor: float = 1.5,
) -> float
```

Combined tone level `LT` of several tones in one critical band (Formula (17)).

`LTm = 10·lg(Σ 10^(LTm,n/10))`, the energy sum of the tonal lines of all
the tones, each spectral line counted at most once. Use it when more than one
audible tone falls in a critical band (Clause 5.3.8 Step 3); the group is
then rated at the frequency of its most audible tone.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Narrow-band levels `Li` of the spectrum, in dB. |
| `frequencies` | The line frequencies, in Hz (strictly increasing). |
| `tone_frequencies` | The tone frequencies sharing the critical band, Hz. |
| `mean_narrowband_levels` | Each tone's mean narrow-band level `LS`, dB (same length as `tone_frequencies`). |
| `effective_bandwidth_factor` | `Δfe/Δf`; 1.5 for a Hanning window. |

**Returns:** Combined tone level `LT`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid or differ in length. |

## critical_band_corners

```python
critical_band_corners(tone_frequency: float) -> tuple[float, float]
```

Lower/upper corner frequencies of the critical band (Formulae (3)-(5)).

With a geometric placement of the corners about the tone,
`f1 = −Δfc/2 + √(Δfc² + 4·fT²)/2` and `f2 = f1 + Δfc`, so that
`√(f1·f2) = fT` and `f2 − f1 = Δfc`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_frequency` | Tone frequency `fT`, in Hz. |

**Returns:** `(f1, f2)` corner frequencies, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `tone_frequency` is not positive/finite. |

## critical_band_level

```python
critical_band_level(
    mean_narrowband_level: float,
    tone_frequency: float,
    line_spacing: float,
) -> float
```

Critical-band level `LG` of the masking noise (Formula (12)).

`LG = LS + 10·lg(Δfc/Δf)` dB, spreading the mean narrow-band level `LS`
over the critical bandwidth `Δfc` relative to the line spacing `Δf`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `mean_narrowband_level` | Mean narrow-band level `LS`, in dB (Formula (6)). |
| `tone_frequency` | Tone frequency `fT`, in Hz. |
| `line_spacing` | Line spacing (frequency resolution) `Δf`, in Hz. |

**Returns:** Critical-band level `LG`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the level is not finite or a frequency is not positive/finite. |

## critical_bandwidth_engineering

```python
critical_bandwidth_engineering(tone_frequency: float) -> float
```

Width `Δfc` of the critical band about a tone (Formula (2)).

`Δfc = 25.0 + 75.0·(1.0 + 1.4·(fT/1000)²)^0.69` Hz. This is the
continuous ISO/PAS 20065 engineering-method bandwidth, distinct from the
stepped ISO 1996-2 Annex C [`critical_bandwidth`](/phonometry/reference/api/environment/measurement/#critical_bandwidth) (100 Hz / 20 %).

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_frequency` | Tone frequency `fT`, in Hz. |

**Returns:** Critical bandwidth `Δfc`, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `tone_frequency` is not positive/finite. |

## energy_sum_level

```python
energy_sum_level(
    line_levels: ArrayLike,
    *,
    effective_bandwidth_factor: float = 1.5,
) -> float
```

Energy sum of the tonal spectral lines with window correction (Formulae (7)/(8)).

For a single line (`K = 1`) the tone level is that line's level with *no*
bandwidth correction, `LT = L1` (Formula (7)). For `K > 1` lines,
`LT = 10·lg(Σ 10^(Li/10)) + 10·lg(Δf/Δfe)` dB (Formula (8)); the window
correction `10·lg(Δf/Δfe)` is `−1.76 dB` for a Hanning window
(`Δfe = 1.5·Δf`, Annex A) and `0 dB` for a rectangular window
(`Δfe = Δf`). The DIN 45681:2005-03 Annex J reference program applies the
same split (`If l = 1 Then LT = 10*Log(LT)/Log(10)` with no `−1.76`).
(The mean narrow-band level `LS` of Formula (6) is the analogous *energy
average* and always carries the correction; [`mean_narrowband_level`](/phonometry/reference/api/psychoacoustics/tone-audibility/#mean_narrowband_level)
and [`tone_level`](/phonometry/reference/api/psychoacoustics/tone-audibility/#tone_level) derive `LS` and `LT` from a critical-band
spectrum.)

**Parameters**

| Name | Description |
| :--- | :--- |
| `line_levels` | Narrow-band levels `Li` of the tonal lines to sum, in dB. |
| `effective_bandwidth_factor` | `Δfe/Δf`; 1.5 for a Hanning window (the default), 1.0 for a rectangular window. Ignored for a single line (Formula (7) applies no correction at `K = 1`). |

**Returns:** Corrected energy-sum level, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `line_levels` is empty/non-finite or the factor is not positive/finite. |

## HANNING_BANDWIDTH_FACTOR

*Constant* (`float`).

```python
HANNING_BANDWIDTH_FACTOR = 1.5
```

## masking_index

```python
masking_index(frequency: float) -> float
```

Masking index `av` of the auditory system (Formula (13)).

`av = −2 − lg[1 + (f/502)²·⁵]` dB. The value is negative and grows more
negative with frequency (see Annex C).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, in Hz. |

**Returns:** Masking index `av`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `frequency` is not positive/finite. |

## mean_audibility

```python
mean_audibility(decisive_audibilities: ArrayLike) -> float
```

Mean audibility `ΔL` over a number of spectra (Formula (20)).

`ΔL = 10·lg[(1/J)·Σ 10^(ΔLj/10)]` dB, the energy mean of the decisive
audibilities `ΔLj` of the `J` staggered narrow-band spectra. A spectrum
with no tone found contributes `ΔLj = −10 dB` (Formula (21)); pass that
value explicitly for such spectra.

**Parameters**

| Name | Description |
| :--- | :--- |
| `decisive_audibilities` | Decisive audibilities `ΔLj` of the spectra, in dB. |

**Returns:** Mean audibility `ΔL`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the input is empty or non-finite. |

## mean_audibility_uncertainty

```python
mean_audibility_uncertainty(
    decisive_audibilities: ArrayLike,
    extended_uncertainties: ArrayLike,
) -> float
```

Extended uncertainty `U` of the mean audibility (Formulae (28)/(29)).

`U = sqrt(sum (10^(0.1*dLj) * Uj)^2) / sum 10^(0.1*dLj)` -- the
energy-weighted propagation of the per-spectrum extended uncertainties
`Uj` through the Formula (20) mean (the coverage factor cancels, so
`Uj` can be passed directly). Clause 6: if fewer than 12 spectra have
been averaged this uncertainty **shall** be taken into consideration;
with 12 averages the standard reports ~+/-1.5 dB as typically achieved.

**Parameters**

| Name | Description |
| :--- | :--- |
| `decisive_audibilities` | Decisive audibilities `dLj` of the spectra, in dB. |
| `extended_uncertainties` | Extended uncertainties `Uj` of the same spectra, in dB (same length). |

**Returns:** Extended uncertainty of the mean audibility, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the arrays are empty, non-finite or of different lengths. |

## mean_narrowband_level

```python
mean_narrowband_level(
    levels: ArrayLike,
    frequencies: ArrayLike,
    tone_frequency: float,
    *,
    effective_bandwidth_factor: float = 1.5,
) -> float
```

Mean narrow-band level `LS` of the masking noise (Formula (6), Annex D).

`LS = 10·lg[(1/M)·Σ 10^(Li/10)] + 10·lg(Δf/Δfe)` dB, determined iteratively
over the lines of the critical band about `tone_frequency` (Formulae (2)-(5)
give the band). The line at the tone frequency is excluded; the average then
drops any line more than `6 dB` above the current `LS` and repeats until
`LS` is stable within `±0.005 dB` or fewer than five lines remain on
either side of the tone (Annex D). The window correction `10·lg(Δf/Δfe)` is
`−1.76 dB` for the recommended Hanning window (`Δfe = 1.5·Δf`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Narrow-band levels `Li` of the spectrum, in dB. |
| `frequencies` | The line frequencies, in Hz (strictly increasing). |
| `tone_frequency` | Tone frequency `fT`, in Hz. |
| `effective_bandwidth_factor` | `Δfe/Δf`; 1.5 for a Hanning window (the default), 1.0 for a rectangular window. |

**Returns:** Mean narrow-band level `LS`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectrum is invalid, the factor is not positive/finite, or no lines fall in the critical band. |

## NO_TONE_AUDIBILITY

*Constant* (`float`).

```python
NO_TONE_AUDIBILITY = -10.0
```

## resolve_tones_separately

```python
resolve_tones_separately(
    tone1_frequency: float,
    tone2_frequency: float,
    audibility1: float,
    audibility2: float,
) -> bool
```

Whether two tones in one critical band are rated separately (Clause 5.3.8).

Returns `True` when two tones sharing a critical band are evaluated on their
own instead of being combined into a single FG tone (Formula (17)): both tone
frequencies lie below 1000 Hz *and* their frequency difference
`|fT1 − fT2|` (Formula (18)) exceeds the separation frequency `fD`
(Formula (19)) evaluated at the more prominent tone (the larger audibility
`ΔL`). Otherwise the tones are combined. This mirrors the DIN 45681 Annex J
reference program (`If l = 2 And fT1 < 1000 And fT2 < 1000` … `If |fT1 −
fT2| > fD Then auflösen`); see [`two_tone_separation_frequency`](/phonometry/reference/api/psychoacoustics/tone-audibility/#two_tone_separation_frequency) for the
verification status.

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone1_frequency` | Frequency `fT1` of the first tone, in Hz. |
| `tone2_frequency` | Frequency `fT2` of the second tone, in Hz. |
| `audibility1` | Audibility `ΔL1` of the first tone, in dB (Formula (14)). |
| `audibility2` | Audibility `ΔL2` of the second tone, in dB (Formula (14)). On a tie (`ΔL1 == ΔL2`) the first tone is taken as the more prominent. |

**Returns:** `True` if the tones are rated separately, `False` if combined.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a frequency is not positive/finite or an audibility is not finite. |

## tone_audibility

```python
tone_audibility(
    tone_level: float,
    mean_narrowband_level: float,
    tone_frequency: float,
    line_spacing: float,
) -> float
```

Audibility `ΔL` of one tone from its levels (Formulae (12)-(14)).

Chains the critical-band level `LG` (Formula (12)), the masking index
`av` (Formula (13)) and the audibility `ΔL = LT − LG − av`
(Formula (14)) for a single tone.

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_level` | Tone level `LT`, in dB (Formula (8)). |
| `mean_narrowband_level` | Mean narrow-band level `LS` of the masking noise, in dB (Formula (6)). |
| `tone_frequency` | Tone frequency `fT`, in Hz. |
| `line_spacing` | Line spacing (frequency resolution) `Δf`, in Hz. |

**Returns:** Audibility `ΔL`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the levels are not finite or a frequency/spacing is not positive/finite. |

## tone_level

```python
tone_level(
    levels: ArrayLike,
    frequencies: ArrayLike,
    tone_frequency: float,
    mean_narrowband_level: float,
    *,
    effective_bandwidth_factor: float = 1.5,
) -> float
```

Tone level `LT` from the tonal lines about a tone (Formula (8)).

The tone energy is carried by the run of lines contiguous with the peak at
`tone_frequency` whose level stays above both `LS + 6 dB` and
`L_peak − 10 dB` (Clause 5.3.3); their energy sum with the window
correction is `LT` (via [`energy_sum_level`](/phonometry/reference/api/psychoacoustics/tone-audibility/#energy_sum_level)). A single-line run
takes its level unchanged (Formula (7), no bandwidth correction).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Narrow-band levels `Li` of the spectrum, in dB. |
| `frequencies` | The line frequencies, in Hz (strictly increasing). |
| `tone_frequency` | Tone frequency `fT` (the peak), in Hz. |
| `mean_narrowband_level` | Mean narrow-band level `LS` of the masking noise, in dB (see [`mean_narrowband_level`](/phonometry/reference/api/psychoacoustics/tone-audibility/#mean_narrowband_level)). |
| `effective_bandwidth_factor` | `Δfe/Δf`; 1.5 for a Hanning window (the default), 1.0 for a rectangular window. |

**Returns:** Tone level `LT`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectrum is invalid or the levels are not finite. |

## ToneAudibilityResult

```python
ToneAudibilityResult(
    tone_frequencies: NDArray[np.float64],
    tone_levels: NDArray[np.float64],
    mean_narrowband_levels: NDArray[np.float64],
    line_spacing: float,
    critical_bandwidths: NDArray[np.float64],
    lower_corners: NDArray[np.float64],
    upper_corners: NDArray[np.float64],
    critical_band_levels: NDArray[np.float64],
    masking_indices: NDArray[np.float64],
    audibilities: NDArray[np.float64],
    extended_uncertainties: NDArray[np.float64] | None = None,
    group_sizes: NDArray[np.int_] | None = None,
)
```

Audibility of the tones of a narrow-band spectrum (ISO/PAS 20065).

**Attributes**

| Name | Description |
| :--- | :--- |
| `tone_frequencies` | Tone frequencies `fT`, in Hz. |
| `tone_levels` | Tone levels `LT`, in dB (Formula (8)). |
| `mean_narrowband_levels` | Mean narrow-band levels `LS` of the masking noise, in dB (Formula (6)). |
| `line_spacing` | Line spacing (frequency resolution) `Δf`, in Hz. |
| `critical_bandwidths` | Critical bandwidths `Δfc`, in Hz (Formula (2)). |
| `lower_corners` | Lower corner frequencies `f1`, in Hz (Formula (4)). |
| `upper_corners` | Upper corner frequencies `f2`, in Hz (Formula (5)). |
| `critical_band_levels` | Critical-band levels `LG` of the masking noise, in dB (Formula (12)). |
| `masking_indices` | Masking indices `av`, in dB (Formula (13)). |
| `audibilities` | Audibilities `ΔL`, in dB (Formula (14)). |
| `extended_uncertainties` | Extended uncertainties `U` of the audibilities, in dB (Clause 6, 90 % bilateral coverage), or `None` when the per-line levels needed to compute them were not available ([`assess_tones`](/phonometry/reference/api/psychoacoustics/tone-audibility/#assess_tones) from bare levels). Clause 6: **shall** be taken into consideration when fewer than 12 spectra have been averaged. |
| `group_sizes` | Number of tones behind each entry, or `None` when the Step 3 combination was not performed ([`assess_tones`](/phonometry/reference/api/psychoacoustics/tone-audibility/#assess_tones) from bare levels). `1` marks an individual tone; `N >= 2` marks a combined "FG" entry whose tone level energy-sums `N` tones sharing a critical band (Clause 5.3.8 Step 3, Formula (17)), rated at the most audible member's frequency. |

### ToneAudibilityResult.audible

*property*

Boolean mask of tones that are present, i.e. `ΔL > 0` (Step 2).

### ToneAudibilityResult.decisive_audibility

*property*

Decisive audibility `ΔLj` of the spectrum: the largest `ΔL` (Step 4).

### ToneAudibilityResult.decisive_frequency

*property*

Tone frequency of the decisive (most audible) tone, in Hz.

### ToneAudibilityResult.plot()

```python
ToneAudibilityResult.plot(
    ax: Axes | None = None,
    *,
    view: str = 'audibility',
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the assessment, either as audibilities or as levels.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes to draw on, or `None` to create a figure. |
| `view` | `"audibility"` (default) draws the per-tone audibility `ΔL` against tone frequency, with the decisive tone highlighted; `"levels"` draws the tone levels `Lpt` above the critical-band masking noise `Lpn` on a continuous frequency axis, the view the `.report()` fiche embeds. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the primary artist call. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `view` is not one of the two names above. |

### ToneAudibilityResult.report()

```python
ToneAudibilityResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a tonal audibility assessment fiche to a PDF.

Writes a one-page tonal-assessment report following ISO 1996-2:2017
Annex J (the engineering method of ISO/PAS 20065:2016): the
standard-basis line, an optional metadata header (source/situation,
client, measurement position, instrumentation, date, with the analysis
line spacing `Δf` read from the result), a full-width table of the key
quantities for every detected tone (frequency `f_T`, tone level
`Lpt`, critical-band masking-noise level `Lpn`, critical bandwidth
`Δf_c` and audibility `ΔL_ta`) above the level-versus-frequency
analysis plot with the tones and their critical-band masking noise
marked, the boxed decisive audibility `ΔL_ta` and the derived tonal
adjustment `K` (Table J.1), an optional verdict row and a prominence
note, and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare assessment fiche (body, result and disclaimer only). A supplied `requirement` is read as the maximum acceptable decisive audibility `ΔL_ta` in dB (a lower audibility passes). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When True, the key-quantity table adds the per-tone extended-uncertainty column (when the result carries it). |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## two_tone_separation_frequency

```python
two_tone_separation_frequency(tone_frequency: float) -> float
```

Frequency-difference threshold `fD` for resolving two tones (Formula (19)).

`fD = 21·10^(1.2·|lg(fT/212)|^1.8)` Hz. When *exactly two* tones fall in one
critical band and both lie below 1000 Hz, the human ear can still tell them
apart (they are then rated *separately* rather than combined into a
single "FG" tone, Formula (17)) if their frequency difference
`|fT1 − fT2|` (Formula (18)) exceeds this threshold. `fT` is the frequency
of the more prominent tone (the larger audibility `ΔL`). The threshold is
`21 Hz` at `fT = 212 Hz` and grows on either side; Formula (19) is
stated for `50 Hz < fT < 1000 Hz` (Clause 5.3.8, Annex D, Note 3).

:::note
No numeric worked example exercises this branch: the Annex E
combustion-engine spectrum groups *three* tones in its critical band, so
the "exactly two tones" rule never fires there. The formula and decision
rule are implemented clean-room from the ISO/PAS 20065 text and verified
against the DIN 45681:2005-03 Annex J reference program
(`fD = 21 * 10 ^ (1.2 * Abs(Log(fT / 212) / Log(10)) ^ 1.8)`), but are
*not* anchored on a numeric oracle.
:::

**Parameters**

| Name | Description |
| :--- | :--- |
| `tone_frequency` | Frequency `fT` of the more prominent tone, in Hz. |

**Returns:** Separation-frequency threshold `fD`, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `tone_frequency` is not positive/finite. |
