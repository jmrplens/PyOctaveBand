---
title: "emission.sound_power_in_situ"
description: "Sound power and sound energy levels of a noise source determined in situ by comparison with a reference sound source: ISO 3747:2010 (engineering grade 2 and survey grade 3)."
sidebar:
  label: "sound_power_in_situ"
---

Sound power and sound energy levels of a noise source determined in situ
by comparison with a reference sound source: ISO 3747:2010 (engineering
grade 2 and survey grade 3).

The source stays where it works. A calibrated reference sound source (RSS)
of known octave-band sound power `LW(RSS)` is set beside it and the same
three or four microphone positions listen to each source in turn, in the
part of the room where the field is reverberant, that is where the excess of
sound pressure level over the free field, $\Delta L_f$, is at least
7 dB (clause 4.1, Annex A). Both sources then see the same room and the room
drops out of the algebra: the sound power level of the source under test
(ST) in each octave band is the calibrated power of the RSS carried across
by the difference of the two mean corrected levels (clause 8.3.1),

$$
L_W = L_{W(\mathrm{RSS})} - \overline{L_{p(\mathrm{RSS})}} + \overline{L_{p(\mathrm{ST})}} \tag{Eq. 11}
$$

where the mean corrected levels are the energy averages over the `n`
microphone positions (Eq. 8, 9) of the levels corrected position by
position for background noise (clause 8.1),

$$
K_{1i} = -10 \log_{10}\!\left(1 - 10^{-0.1\,\Delta L_{pi}}\right), \qquad \Delta L_{pi} = L'_{pi(\mathrm{ST})} - L_{pi(\mathrm{B})} \tag{Eq. 7}
$$

with three rules around it: a margin above 15 dB needs no correction, a
margin between 6 dB and 15 dB takes Eq. (7), and a margin below 6 dB caps
the correction at 1,3 dB and turns the band into an upper bound that the
report must flag as not meeting the background requirement. When the RSS is
run at `m` locations around a large source the calibrated powers and the
per-location means are each energy-averaged over the locations before the
subtraction (clause 8.3.2, Eq. 12).

An impulsive source is described by its sound energy level instead. The
single event levels measured at each position, either `N` events one at a
time (Eq. 13, 15) or one measurement encompassing `N` events (Eq. 16, 17),
are background-corrected with the same rule (Eq. 14), reduced to the level
of one event and averaged over positions (Eq. 18); Eq. (19) and (20) are
then Eq. (11) and (12) with $\overline{L_{E(\mathrm{ST})}}$ in place of
$\overline{L_{p(\mathrm{ST})}}$ (clause 8.5). Eq. (14) subtracts the
time-averaged background level from a time-integrated event level, exactly
as ISO 3741:2010 (9.2.2) and ISO 3744:2010 (8.3.4) print it; the text only
asks that both be measured over the same integration time `T`. As printed
the difference is a true signal-to-background margin for `T` = 1 s; the
optional `integration_time` carries the background to the event's interval
($+10 \log_{10}(T/T_0)$, clause 3.4 NOTE 1) before the subtraction.

Annex C carries either level to the reference meteorological conditions of
101,325 kPa and 23,0 °C with the radiation-impedance correction
$C_2 = -10 \log_{10}(p_\mathrm{s}/p_{\mathrm{s},0}) + 15 \log_{10}((273.15 + \theta)/296)$,
the same `C2` as ISO 3741:2010 clause 9.1.4, reused from that module; the
whole ISO 3740 family prints $\theta_\mathrm{ref}$ = 296 K beside a
23,0 °C reference, so at the reference conditions `C2` is +0,003 3 dB
rather than zero. Eq. (C.2) estimates the static pressure from the altitude
of the site. Annex D forms the A-weighted totals from the Table D.1 band
corrections, which are the ISO 3744 Annex E octave values digit for digit.

Clause 9 estimates the uncertainty as $\sigma_\mathrm{tot} = \sqrt{\sigma_{R0}^2 + \sigma_\mathrm{omc}^2}$ (Eq. 22) and
$U = k\,\sigma_\mathrm{tot}$ (Eq. 23), with the typical upper bound of
the reproducibility $\sigma_{R0}$ read from Table 2 by grade: 1,5 dB
for grade 2, which needs $\Delta L_{f\mathrm{A}} \ge 7$ dB at every
microphone position and a source directivity range within ±7 dB, and 4,0 dB
for grade 3 otherwise. Table 1, the zoning of the test environment by lines
of sight, is guidance for placing the sources and is not evaluated here.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## excess_sound_pressure_level

```python
excess_sound_pressure_level(
    level: ArrayLike,
    lw_ref: ArrayLike,
    distance: ArrayLike,
) -> np.ndarray | float
```

Excess of sound pressure level over the free field at a distance from
the reference sound source (ISO 3747:2010 Annex A, Eq. A.1).

$$
\Delta L_f(r) = L_{p(\mathrm{RSS}),r} - L_{W(\mathrm{RSS})} + 11\ \mathrm{dB} + 20 \log_{10}\frac{r}{r_0}, \qquad r_0 = 1\ \mathrm{m}
$$

The 11 dB is the spherical free-field relation $L_p = L_W - 20 \log_{10}(r/r_0) - 11$ dB, so $\Delta L_f$ is zero in a free field
and grows with the reverberant contribution; the microphone positions of
the method must lie where it is at least 7 dB (4.1, 7.4.1). Measured
with A-weighted levels the quantity is $\Delta L_{f\mathrm{A}}$,
the indicator Table 2 grades the determination by. The three arguments
broadcast against each other, so one calibrated power serves a whole
traverse of levels and distances.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | Sound pressure level(s) `Lp(RSS),r` measured at distance `r` from the reference sound source, in decibels. |
| `lw_ref` | Calibrated sound power level `LW(RSS)` of the reference source, in decibels (per band, or A-weighted). |
| `distance` | Distance(s) `r` from the microphone to the reference source, in metres. |

**Returns:** The excess `dLf(r)`, in decibels, as a float for scalar input or an array of the broadcast shape.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if any input is not finite or any `distance` is not positive. |

## InSituSoundPowerResult

```python
InSituSoundPowerResult(
    frequencies: np.ndarray,
    sound_power_level: np.ndarray,
    sound_energy_level: np.ndarray,
    mean_source_level: np.ndarray,
    mean_reference_level: np.ndarray,
    reference_levels: np.ndarray,
    reference_power_level: np.ndarray,
    background_correction: np.ndarray,
    background_correction_ref: np.ndarray,
    background_requirement_met: np.ndarray,
    c2: float,
    grade: str,
    sigma_r0: float,
    sigma_omc: float,
    sigma_tot: float,
    expanded_uncertainty: float,
    coverage_factor: float,
    sound_power_level_a: float,
    sound_energy_level_a: float,
    quantity: str,
)
```

Result of an ISO 3747:2010 in situ determination by comparison.

`quantity` says which of the two determinations this is: `'power'`
carries the octave-band sound power level `LW` (Eq. 11 or 12) in
`sound_power_level` with `sound_energy_level` all `NaN`, and
`'energy'` the sound energy level `LJ` (Eq. 19 or 20) in
`sound_energy_level` with `sound_power_level` all `NaN`. Both are
at the meteorological conditions of the test; the properties
`sound_power_level_ref` and `sound_energy_level_ref` add the
Annex C correction `c2` (Eq. C.1, C.3).

`mean_source_level` is the mean corrected level of the source under
test, $\overline{L_{p(\mathrm{ST})}}$ (Eq. 8) or
$\overline{L_{E(\mathrm{ST})}}$ (Eq. 18); `reference_levels` the
mean corrected level of the reference sound source at each of its `m`
locations (Eq. 9, 10) and `mean_reference_level` their energy mean
(the second term of Eq. 12, equal to Eq. 9 for one location);
`reference_power_level` the calibrated power of the reference source,
energy-averaged over its locations (the first term of Eq. 12).

`background_correction` is `K1i` at each microphone position and band
for the source under test (Eq. 7; for `N` events measured one at a
time it is the per-position shift the per-event corrections of Eq. 13
produce in the mean of Eq. 15), `background_correction_ref` the same for
the reference source at each location (Eq. 9, 10), and
`background_requirement_met` is `False` in every band where some
margin fell below 6 dB, so that the level there is an upper bound to be
reported as such (8.1).

`grade` is the accuracy grade Table 2 grants (`'engineering'` or
`'survey'`) and `sigma_r0` its typical reproducibility; `sigma_omc`,
`sigma_tot` and `expanded_uncertainty` are the operating-and-mounting
deviation, Eq. (22) and Eq. (23) for `coverage_factor`, `NaN` when no
`sigma_omc` was supplied. `sound_power_level_a` and
`sound_energy_level_a` are the Annex D A-weighted totals of the level
that was determined (`NaN` for the other).

### InSituSoundPowerResult.plot()

```python
InSituSoundPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the determined spectrum with the A-weighted total annotated.

One bar per octave band of `LW` (or `LJ` for an energy
determination); a band whose background margin fell below 6 dB is
hatched, because its level is an upper bound (8.1). Requires
matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### InSituSoundPowerResult.sound_energy_level_ref

*property*

`LJ` under the reference meteorological conditions, `LJ + C2`
(ISO 3747:2010 Annex C, Eq. C.3); `NaN` for a power determination.

### InSituSoundPowerResult.sound_power_level_ref

*property*

`LW` under the reference meteorological conditions, `LW + C2`
(ISO 3747:2010 Annex C, Eq. C.1); `NaN` for an energy determination.

## sound_energy_in_situ

```python
sound_energy_in_situ(
    event_levels: ArrayLike,
    levels_ref: ArrayLike,
    lw_ref: ArrayLike,
    frequencies: ArrayLike,
    *,
    events: int | None = None,
    background_levels: ArrayLike | None = None,
    background_levels_ref: ArrayLike | None = None,
    integration_time: float | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
    excess_levels: ArrayLike | None = None,
    directivity_range: float | None = None,
    sigma_omc: float | None = None,
    coverage_factor: float = 2.0,
) -> InSituSoundPowerResult
```

Sound energy level of an impulsive source in situ, by comparison with
a reference sound source (ISO 3747:2010, clauses 8.4 and 8.5).

The single event levels are given in one of the two forms clause 8.4
admits. Measured one event at a time, `event_levels` is `(n, N,
bands)`: each event is corrected for background (Eq. 13, 14) and the
`N` corrected levels are energy-averaged into the mean single event
level of the position (Eq. 15). Measured once over `N` successive
events, `event_levels` is `(n, bands)` with `events=N`: the level
is corrected (Eq. 16) and reduced by $10 \log_{10} N$ to one event
(Eq. 17). Either way the per-position levels are energy-averaged (Eq. 18)
and the sound energy level in each band is

$$
L_J = L_{W(\mathrm{RSS})} - \overline{L_{p(\mathrm{RSS})}} + \overline{L_{E(\mathrm{ST})}} \tag{Eq. 19}
$$

or its `m`-location form (Eq. 20). The reference source is measured
time-averaged, over 30 s (7.6), exactly as for a steady source.

**Parameters**

| Name | Description |
| :--- | :--- |
| `event_levels` | Measured (uncorrected) octave-band single event levels `L'Ei,q(ST)` as `(n, N, bands)`, or `L'Ei,N(ST)` of one measurement encompassing `events` events as `(n, bands)`, in decibels. |
| `levels_ref` | Time-averaged levels of the reference sound source, `(n, bands)` or `(m, n, bands)`, as in [`sound_power_in_situ`](/phonometry/reference/api/power/sound-power-in-situ/#sound_power_in_situ). |
| `lw_ref` | Calibrated sound power level of the reference source, `(bands,)` or `(m, bands)`, in decibels. |
| `frequencies` | Nominal octave mid-band frequencies, one per band. |
| `events` | The number `N` of events a 2D `event_levels` contains (Eq. 17); must be `None` with the 3D form, which counts them. |
| `background_levels` | Octave-band time-averaged background levels `Lpi(B)`, `(n, bands)` or `(bands,)`, in decibels. |
| `background_levels_ref` | Background for the reference-source measurement; `None` reuses `background_levels` (7.5). |
| `integration_time` | The integration time `T` of the event measurement, in seconds. `None` applies Eq. (14) as printed, subtracting the time-averaged background from the single event level; a value carries the background to the same interval first, $L_{pi(\mathrm{B})} + 10 \log_{10}(T/T_0)$ with `T0` = 1 s (3.4, NOTE 1), so that the margin compares like with like. The two coincide at `T` = 1 s. |
| `temperature` | Air temperature at the test, in degrees Celsius. |
| `static_pressure` | Static pressure at the test, in kilopascals. |
| `excess_levels` | `dLfA` at each microphone position, `(n,)`. |
| `directivity_range` | Half-width of the directivity range, in decibels. |
| `sigma_omc` | Operating-and-mounting standard deviation, in decibels. |
| `coverage_factor` | `k` of Eq. (23), 2 by default. |

**Returns:** [`InSituSoundPowerResult`](/phonometry/reference/api/power/sound-power-in-situ/#insitusoundpowerresult) with `quantity='energy'`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `event_levels` is not a finite 2D or 3D array, `events` is given with the 3D form or missing or not a positive integer with the 2D form, `integration_time` is not positive, or any of the refusals of [`sound_power_in_situ`](/phonometry/reference/api/power/sound-power-in-situ/#sound_power_in_situ) applies. |

## sound_power_in_situ

```python
sound_power_in_situ(
    levels: ArrayLike,
    levels_ref: ArrayLike,
    lw_ref: ArrayLike,
    frequencies: ArrayLike,
    *,
    background_levels: ArrayLike | None = None,
    background_levels_ref: ArrayLike | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
    excess_levels: ArrayLike | None = None,
    directivity_range: float | None = None,
    sigma_omc: float | None = None,
    coverage_factor: float = 2.0,
) -> InSituSoundPowerResult
```

Sound power level of a steady or non-steady source in situ, by
comparison with a reference sound source (ISO 3747:2010, clause 8.3).

The time-averaged octave-band levels of the source under test at the
`n` microphone positions are corrected for background noise position
by position (Eq. 7 with the rules of 8.1) and energy-averaged (Eq. 8);
the reference source's levels at the same positions are treated the same
way (Eq. 9), or per location and then energy-averaged over the `m`
locations together with its calibrated powers (Eq. 10, 12). The sound
power level in each band is then

$$
L_W = L_{W(\mathrm{RSS})} - \overline{L_{p(\mathrm{RSS})}} + \overline{L_{p(\mathrm{ST})}} \tag{Eq. 11}
$$

at the meteorological conditions of the test; the returned `c2` and
the `sound_power_level_ref` property carry it to the reference
conditions of Annex C, and `sound_power_level_a` is the Annex D total.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Measured (uncorrected) octave-band time-averaged levels `L'pi(ST)` of the source under test, `(n, bands)`, one row per microphone position, in decibels. |
| `levels_ref` | The same for the reference sound source, `L'pi(RSS)` already corrected for speed, temperature and static pressure per its manufacturer but not for background: `(n, bands)` for one location, or `(m, n, bands)` for `m` locations (Eq. 10). |
| `lw_ref` | Calibrated octave-band sound power level `LW(RSS)` of the reference source, `(bands,)`, or `(m, bands)` when each location was calibrated in its own similar position (Eq. 12), in decibels. |
| `frequencies` | Nominal octave mid-band frequencies, one per band, from 63 Hz to 8 kHz (Table D.1), in hertz. |
| `background_levels` | Octave-band time-averaged background levels `Lpi(B)`, `(n, bands)` or one `(bands,)` spectrum for every position, in decibels; `None` applies no correction. |
| `background_levels_ref` | Background for the reference-source measurement, same shapes; `None` reuses `background_levels`, since the procedure takes one background reading (7.5). |
| `temperature` | Air temperature at the test, in degrees Celsius. |
| `static_pressure` | Static pressure at the test, in kilopascals (see [`static_pressure_from_altitude`](/phonometry/reference/api/power/sound-power-in-situ/#static_pressure_from_altitude)). |
| `excess_levels` | A-weighted excess of sound pressure level `dLfA` at each microphone position (Annex A), `(n,)`, in decibels; with `directivity_range` it decides the grade (Table 2). |
| `directivity_range` | Range of the A-weighted directivity survey of the source (7.2), as the half-width `x` of `+/-x` dB. |
| `sigma_omc` | Standard deviation of the operating and mounting conditions of the source (9.2, E.3), in decibels; `None` leaves `sigma_tot` and the expanded uncertainty `NaN`. |
| `coverage_factor` | `k` of Eq. (23): 2 for the two-sided 95 % interval (default), 1,6 for a one-sided comparison with a limit. |

**Returns:** [`InSituSoundPowerResult`](/phonometry/reference/api/power/sound-power-in-situ/#insitusoundpowerresult) with `quantity='power'`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `levels` is not a finite `(n, bands)` array, `levels_ref`, `lw_ref` or either background does not match it, `frequencies` are not the octave centres of Table D.1, `temperature` or `static_pressure` is out of range, `excess_levels` is not one finite value per position, `directivity_range` or `sigma_omc` is negative, or `coverage_factor` is not positive. |

## static_pressure_from_altitude

```python
static_pressure_from_altitude(altitude: float) -> float
```

Static pressure at the altitude of the test site (ISO 3747:2010 Annex C,
Eq. C.2).

$$
p_\mathrm{s} = p_{\mathrm{s},0}\,(1 - a H_\mathrm{a})^{b}, \qquad a = 2{,}2560 \times 10^{-5}\ \mathrm{m}^{-1}, \quad b = 5{,}255\,3
$$

with $p_{\mathrm{s},0}$ = 101,325 kPa. The result is in kilopascals
so that it feeds `static_pressure` of [`sound_power_in_situ`](/phonometry/reference/api/power/sound-power-in-situ/#sound_power_in_situ)
directly. A site below sea level is admissible (the base exceeds one);
the formula stops meaning anything where the base reaches zero, some
44 km up, and that is refused.

**Parameters**

| Name | Description |
| :--- | :--- |
| `altitude` | Altitude of the test site `Ha`, in metres. |

**Returns:** The static pressure `ps`, in kilopascals.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `altitude` is not finite or `1 - a Ha` is not positive. |
