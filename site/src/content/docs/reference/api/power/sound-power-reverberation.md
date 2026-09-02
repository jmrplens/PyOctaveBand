---
title: "emission.sound_power_reverberation"
description: "Sound power level of a noise source measured in a reverberation test room: ISO 3741:2010 (precision method, accuracy grade 1)."
sidebar:
  label: "sound_power_reverberation"
---

Sound power level of a noise source measured in a reverberation test room:
ISO 3741:2010 (precision method, accuracy grade 1).

The source is placed in a hard-walled reverberation room whose reverberant
field is sampled by microphones. Two methods are provided.

The **direct method** derives the sound power from the mean corrected room
sound pressure level `Lp(ST)` and the equivalent absorption area `A` of the
room, with the Sabine absorption area and the speed of sound `c` in m/s
(ISO 3741:2010 clause 9.1.4, Eq. 20):

$$
L_p(\text{ST}) = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}} \sum_i 10^{0.1 L_{pi}} \right] \tag{Eq. 16}
$$

$$
A = \frac{55.26}{c} \, \frac{V}{T_{60}}
$$

$$
c = 20.05 \sqrt{273 + \theta}
$$

$$
L_W = L_p(\text{ST}) + 10 \log_{10}\frac{A}{A_0} + 4.34 \frac{A}{S} + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6 \tag{Eq. 20}
$$

$10 \log_{10}(1 + Sc/(8Vf))$ is the Waterhouse boundary correction (energy
stored near the room boundaries); it vanishes as the frequency grows. `C1`
(Eq. 20, reference-quantity correction) and `C2` (radiation-impedance
correction) carry the result to the reference meteorological conditions of
clause 4 (23.0 C, 101.325 kPa, 50 %), per clause 9.1.4:

$$
C_1 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s}0}} + 5 \log_{10}\frac{273.15 + \theta}{314}
$$

$$
C_2 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s}0}} + 15 \log_{10}\frac{273.15 + \theta}{296}
$$

The **comparison method** replaces the absorption-area terms by a reference
sound source (RSS) of known sound power `LW(RSS)` measured at the same
positions (ISO 3741:2010 clause 9.1.5, Eq. 21):

$$
L_W = L_W(\text{RSS}) + \left( L_p(\text{ST}) - L_p(\text{RSS}) + C_2 \right) \tag{Eq. 21}
$$

Both methods cover the one-third-octave bands from 100 Hz to 10 kHz (clause
8.1). Octave-band, A-weighted and total levels follow ISO 3741 Annex F, which
reuses the ISO 3744 Annex E A-weighting band corrections.

A noise burst or a transient emission is described by the **sound energy
level** $L_J = 10 \log_{10}(J/J_0)$, $J_0 = 1$ pJ (clause 3.18),
and clause 9.2 determines it by the same two methods with the single event
time-integrated sound pressure level $L_E$ (clause 3.4) in place of the
time-averaged $L_p$: the $N_\mathrm{e}$ events at each position
are reduced to the level of one event (Eq. 22 or Eq. 23), each position is
corrected for its background by $K_{1i}$ (Eq. 25, 26) as in 9.1.2, the
positions are energy-averaged (Eq. 27), and the room enters through the same
bracket as Eq. (20) or the same reference source as Eq. (21):

$$
L_J = \overline{L_E(\text{ST})} + \left[ 10 \log_{10}\frac{A}{A_0} + 4.34 \frac{A}{S} + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6 \right] \tag{Eq. 30}
$$

$$
L_J = L_W(\text{RSS}) + \left( \overline{L_E(\text{ST})} - \overline{L_p(\text{RSS})} \right) + C_2 \tag{Eq. 31}
$$

For a source steady over the whole interval $T$, clause 3.4 NOTE 1
gives $L_E = L_{p,T} + 10 \log_{10}(T/T_0)$, $T_0 = 1$ s, and
so $L_J = L_W + 10 \log_{10}(T/T_0)$. Annex F sums the one-third-octave
levels into octave bands (Eq. F.1, F.4) and A-weights them (Eq. F.2, F.5)
alike for the two quantities.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## octave_band_levels

```python
octave_band_levels(
    levels: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]
```

Octave-band levels from one-third-octave band levels (ISO 3741 Annex F).

The level in the $i$-th octave band, $1 \le i \le 8$ for the
mid-band frequencies 63 Hz to 8 kHz, is the energy sum of the three
one-third-octave bands $k = 3i-2$ to $3i$ of Table F.1 that
make it up, for sound power levels (Eq. F.1) and sound energy levels
(Eq. F.4) alike:

$$
L_{Ji} = 10 \log_{10} \sum_{k=3i-2}^{3i} 10^{0.1 L_{Jk}}
$$

Every octave the input touches must be supplied whole: Table F.1 numbers
the one-third-octave bands from $k = 1$ at 50 Hz, so the 63 Hz
octave is the 50, 63 and 80 Hz thirds and the 8 kHz octave the 6,3, 8 and
10 kHz thirds, and a band whose triplet is incomplete cannot be summed.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Band levels in decibels, with the bands on the last axis (`(NB,)`, or `(..., NB)` for several spectra at once). |
| `frequencies` | The `NB` nominal one-third-octave mid-band frequencies of `levels`, in hertz, from 50 Hz to 10 kHz. |

**Returns:** `(octave mid-band frequencies, octave-band levels)`, the frequencies ascending and the levels with the octaves on the last axis.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a frequency outside Table F.1, a repeated frequency, an octave whose three thirds are not all present, or levels that do not carry one value per frequency. |

## ReverberationSoundEnergyResult

```python
ReverberationSoundEnergyResult(
    frequencies: np.ndarray | None,
    sound_energy_level: np.ndarray,
    mean_event_level: np.ndarray,
    absorption_area: np.ndarray,
    waterhouse_correction: np.ndarray,
    background_correction: np.ndarray,
    c1: float,
    c2: float,
    speed_of_sound: float,
    sound_energy_level_a: float,
    method: str,
    events: int | None,
    integration_time: float | None,
)
```

Result of an ISO 3741:2010 reverberation-room sound energy level
determination (clause 9.2).

`sound_energy_level` is the per-band `LJ` (Eq. 30 direct method, Eq. 31
comparison method), under reference meteorological conditions as both
equations state it. `mean_event_level` is the mean corrected single event
time-integrated level in the room $\overline{L_E(\text{ST})}$
(Eq. 27). `absorption_area`, `waterhouse_correction`, `c1`, `c2`,
`speed_of_sound` and `background_correction` are what they are in
[`ReverberationSoundPowerResult`](/phonometry/reference/api/power/sound-power-reverberation/#reverberationsoundpowerresult), the background correction being
the per-band shift of the mean level after each position was corrected by
its own `K1i` (Eq. 25/26). `sound_energy_level_a` is the A-weighted
total `LJA` (Annex F Eq. F.5), computed only when `frequencies` are
supplied (`NaN` for several bands without them; equal to `LJ` for a
single band). `method` is `'direct'` or `'comparison'`. `events`
is the number of single sound emission events $N_\mathrm{e}$ the
levels were reduced from, or `None` when the caller supplied the mean
single event level of one event; `integration_time` is the interval
$T$ of the single event levels, in seconds, or `None` when no
background correction needed it.

### ReverberationSoundEnergyResult.plot()

```python
ReverberationSoundEnergyResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the LJ spectrum with the A-weighted total annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## ReverberationSoundPowerResult

```python
ReverberationSoundPowerResult(
    frequencies: np.ndarray | None,
    sound_power_level: np.ndarray,
    mean_pressure_level: np.ndarray,
    absorption_area: np.ndarray,
    waterhouse_correction: np.ndarray,
    background_correction: np.ndarray,
    c1: float,
    c2: float,
    speed_of_sound: float,
    sound_power_level_a: float,
    method: str,
)
```

Result of an ISO 3741:2010 reverberation-room sound power determination.

`sound_power_level` is the per-band `LW` (Eq. 20 direct method, Eq. 21
comparison method). `mean_pressure_level` is the mean corrected room level
`Lp(ST)` (Eq. 16). For the direct method `absorption_area` is the Sabine
equivalent absorption area `A` per band and `waterhouse_correction` the
boundary term $10 \log_{10}(1 + Sc/(8Vf))$; both are `NaN` for the
comparison method. `background_correction` is the effective per-band
background correction `K1`: with per-position input each position is
corrected by its own `K1i` (Eq. 14/15) before the energy average
(Eq. 16), and the reported value is the resulting per-band shift of the
mean level (zero when no background is supplied).
`c1` and `c2` are the reference-quantity and radiation-impedance
corrections (`c1` is `NaN` for the comparison method, which uses only
`c2`). `speed_of_sound` is `c` at the test temperature.
`sound_power_level_a` is the A-weighted total `LWA` (Annex F Eq. F.2),
computed only when `frequencies` are supplied (`NaN` for several bands
without them; equal to `LW` for a single band). `method` is `'direct'`
or `'comparison'`.

### ReverberationSoundPowerResult.plot()

```python
ReverberationSoundPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the LW spectrum with the A-weighted total annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ReverberationSoundPowerResult.report()

```python
ReverberationSoundPowerResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 3741 reverberation-room sound-power determination fiche.

Writes a one-page sound-power test sheet: the standard-basis line naming
the reverberation-room method (the direct method using the room
equivalent absorption area, or the comparison method using a reference
sound source) and the precision accuracy grade (ISO 3741:2010, grade 1),
an optional metadata header (client, noise source, test environment,
instrumentation, climate, date), a per-band table (nominal octave/
one-third-octave frequency, the mean room sound-pressure level `Lp`
and the band sound-power level `LW`), the sound-power spectrum
`LW(f)` with a nominal band axis, the boxed A-weighted sound power
level `LWA` (dB re 1 pW) with the total `LW` and the determination
method, an optional verdict row against a declared limit, and a
measurement-basis strip stating the correction model (Eq. 20 direct or
Eq. 21 comparison), the applied meteorological corrections `C1`/`C2`
and the speed of sound.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the noise source, `test_room` the reverberation test room, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared A-weighted sound-power limit the fiche checks the result against (lower is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the background correction `K1` and, for the direct method, the equivalent absorption area `A` and the Waterhouse boundary correction `Cw`. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## sound_energy_comparison

```python
sound_energy_comparison(
    levels: np.ndarray,
    levels_ref: np.ndarray,
    lw_ref: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    events: int | None = None,
    background_levels: np.ndarray | None = None,
    integration_time: float | None = None,
    background_levels_ref: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundEnergyResult
```

Sound energy level in a reverberation room, comparison method
(ISO 3741:2010 clause 9.2.5).

A reference sound source of known per-band sound power `lw_ref` runs
steadily at the same microphone positions as the source under test, whose
single event levels `levels` take any of the forms
[`sound_energy_reverberation`](/phonometry/reference/api/power/sound-power-reverberation/#sound_energy_reverberation) accepts. The sound energy level in each
band follows Eq. (31):

$$
L_J = L_W(\text{RSS}) + \left( \overline{L_E(\text{ST})} - \overline{L_p(\text{RSS})} \right) + C_2
$$

where $\overline{L_E(\text{ST})}$ is the mean corrected single event
level of the source under test (Eq. 27), $\overline{L_p(\text{RSS})}$
the mean corrected time-averaged level of the reference source (Eq. 17) and
`C2` the radiation-impedance correction. The room terms and `C1` cancel
between the two sources exactly as in [`sound_power_comparison`](/phonometry/reference/api/power/sound-power-reverberation/#sound_power_comparison). The
source under test is background-corrected as in
[`sound_energy_reverberation`](/phonometry/reference/api/power/sound-power-reverberation/#sound_energy_reverberation) (its background compared as an exposure
over `integration_time`); the reference source, being steady, by the
time-averaged correction of 9.1.2.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Single event levels of the source under test, in decibels. |
| `levels_ref` | Mean room SPL per band (1D) or `(NM, NB)` per-position time-averaged levels of the reference sound source, in decibels. |
| `lw_ref` | Known sound power level `LW(RSS)` per band, in decibels, under the meteorological conditions of the test. |
| `frequencies` | Band mid-frequencies (Hz) for the `K1` criterion and the A-weighted total. |
| `events` | The number of events `Ne` one measurement encompasses (Eq. 23); `None` when `levels` is per event or already the mean of one event. |
| `background_levels` | Time-averaged background levels for the `K1` correction of `levels` (per position, or a single spectrum). |
| `integration_time` | The interval `T` of the single event levels, in seconds; required with `background_levels`. |
| `background_levels_ref` | Background levels matching `levels_ref`. |
| `temperature` | Air temperature `theta` in the room, in degrees Celsius. |
| `static_pressure` | Static pressure `ps` in the room, in kilopascals. |

**Returns:** [`ReverberationSoundEnergyResult`](/phonometry/reference/api/power/sound-power-reverberation/#reverberationsoundenergyresult) (`method='comparison'`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a malformed or non-finite level array, a non-physical climate, a background without its `integration_time` or without `frequencies`, or mismatched band counts. |

## sound_energy_reverberation

```python
sound_energy_reverberation(
    levels: np.ndarray,
    t60: float | np.ndarray,
    volume: float,
    surface_area: float,
    frequencies: np.ndarray,
    *,
    events: int | None = None,
    background_levels: np.ndarray | None = None,
    integration_time: float | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundEnergyResult
```

Sound energy level in a reverberation room, direct method (ISO 3741:2010
clause 9.2.4).

`levels` holds the single event time-integrated sound pressure levels
$L'_{Ei(\mathrm{ST})}$ measured through a period that encompasses the
whole of the event, its decay included (clause 8.5.1; a moving microphone
is not permitted for non-repetitive impulsive noise): a 1D per-band
spectrum already averaged over the room, a 2D `(NM, NB)` array of one
event's level at each position, a 3D `(Ne, NM, NB)` array of the
$N_\mathrm{e}$ events measured one at a time (reduced by Eq. 22), or
a 1D/2D level of one measurement encompassing `events` successive events
(reduced by Eq. 23). Each position is corrected for its background by
$K_{1i}$ (Eq. 25/26, the frequency-dependent criterion of 9.1.2),
the positions are energy-averaged (Eq. 27) and the sound energy level in
each band follows Eq. (30):

$$
L_J = \overline{L_E(\text{ST})} + \left[ 10 \log_{10}\frac{A}{A_0} + 4.34 \frac{A}{S} + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6 \right]
$$

with every term of the bracket exactly as in [`sound_power_reverberation`](/phonometry/reference/api/power/sound-power-reverberation/#sound_power_reverberation)
(Eq. 20), so the level is stated under the reference meteorological
conditions of clause 4. The background is the time-averaged level the
standard has measured over the same integration time $T$ as the
events (clause 9.2.2), and it is compared as its exposure over that
$T$, $L_{pi(\mathrm{B})} + 10 \log_{10}(T/T_0)$ (clause 3.4
NOTE 1), so that the energies Eq. (25) subtracts share one reference;
`integration_time` is therefore required with `background_levels`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Single event levels, in decibels, in one of the four forms above. |
| `t60` | Reverberation time `T60` per band, in seconds (scalar or one value per band). |
| `volume` | Reverberation-room volume `V`, in cubic metres. |
| `surface_area` | Total room surface area `S`, in square metres. |
| `frequencies` | One-third-octave band mid-frequencies, Hz (required: the Waterhouse term and the `K1` criterion need them). |
| `events` | The number of events `Ne` one measurement encompasses (Eq. 23); `None` when `levels` is per event or already the mean of one event. |
| `background_levels` | Time-averaged background levels for `K1`: per-position `(NM, NB)` (or a single `(NB,)` spectrum used at every position) with per-position `levels`, applied per position before the energy average; with 1D `levels` a single `K1` from the averaged spectra approximates the per-position procedure. |
| `integration_time` | The interval `T` of the single event levels, in seconds; required with `background_levels`. |
| `temperature` | Air temperature `theta` in the room, in degrees Celsius. |
| `static_pressure` | Static pressure `ps` in the room, in kilopascals. |

**Returns:** [`ReverberationSoundEnergyResult`](/phonometry/reference/api/power/sound-power-reverberation/#reverberationsoundenergyresult) (`method='direct'`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a malformed or non-finite level array, a non-physical room or climate, a background without its `integration_time`, or mismatched band counts. |

## sound_power_comparison

```python
sound_power_comparison(
    levels: np.ndarray,
    levels_ref: np.ndarray,
    lw_ref: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    background_levels: np.ndarray | None = None,
    background_levels_ref: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundPowerResult
```

Sound power level in a reverberation room, comparison method (ISO 3741).

A reference sound source of known per-band sound power `lw_ref` is
measured at the same microphone positions as the source under test. The
sound power level in each band follows Eq. (21):

$$
L_W = L_W(\text{RSS}) + \left( L_p(\text{ST}) - L_p(\text{RSS}) + C_2 \right)
$$

where `Lp(ST)` and `Lp(RSS)` are the mean room levels (Eq. 16/17) of the
test source and the reference source and `C2` is the radiation-impedance
correction. The absorption-area, Waterhouse and `C1` terms cancel between
the two sources, so the room absorption need not be known.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Mean room SPL per band (1D) or `(NM, NB)` per-position levels of the source under test, in decibels. |
| `levels_ref` | Same, for the reference sound source, in decibels. |
| `lw_ref` | Known sound power level `LW(RSS)` per band, in decibels. |
| `frequencies` | Band mid-frequencies (Hz) for the A-weighted total. |
| `background_levels` | Background levels for the `K1` correction of `levels` (per position, or a single spectrum; applied per position per Eq. 14/15 before the Eq. 16 average when `levels` is 2D). |
| `background_levels_ref` | Background levels matching `levels_ref`. |
| `temperature` | Air temperature `theta` in the room, in degrees Celsius. |
| `static_pressure` | Static pressure `ps` in the room, in kilopascals. |

**Returns:** [`ReverberationSoundPowerResult`](/phonometry/reference/api/power/sound-power-reverberation/#reverberationsoundpowerresult) (`method='comparison'`).

## sound_power_reverberation

```python
sound_power_reverberation(
    levels: np.ndarray,
    t60: float | np.ndarray,
    volume: float,
    surface_area: float,
    frequencies: np.ndarray,
    *,
    background_levels: np.ndarray | None = None,
    temperature: float = 23.0,
    static_pressure: float = 101.325,
) -> ReverberationSoundPowerResult
```

Sound power level in a reverberation room, direct method (ISO 3741:2010).

`levels` is either a 1D per-band spectrum of the mean room sound pressure
level or a 2D `(NM, NB)` array (one row per microphone position, one
column per band) that is energy-averaged over positions (Eq. 16). The sound
power level in each band follows Eq. (20):

$$
L_W = L_p(\text{ST}) + 10 \log_{10}\frac{A}{A_0} + 4.34 \frac{A}{S} + 10 \log_{10}\!\left( 1 + \frac{S c}{8 V f} \right) + C_1 + C_2 - 6
$$

with the Sabine equivalent absorption area
$A = (55.26/c)(V/T_{60})$ and the speed of sound
$c = 20.05 \sqrt{273 + \theta}$. The Waterhouse term
$10 \log_{10}(1 + Sc/(8Vf))$ needs the band mid-frequencies, so
`frequencies` is required. `C1` and `C2` carry the result to the
reference meteorological conditions (clause 4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Mean room SPL per band (1D) or `(NM, NB)` per-position levels, in decibels. |
| `t60` | Reverberation time `T60` per band, in seconds (scalar or one value per band). |
| `volume` | Reverberation-room volume `V`, in cubic metres. |
| `surface_area` | Total room surface area `S`, in square metres. |
| `frequencies` | One-third-octave (or octave) band mid-frequencies, Hz. |
| `background_levels` | Background levels for the `K1` correction: per-position `(NM, NB)` (or a single `(NB,)` spectrum used at every position) with per-position `levels`, applied per position (Eq. 14/15) before the energy average (Eq. 16). With 1D pre-averaged `levels` a single `K1` from the averaged spectra approximates the per-position procedure of clause 9.1.2. |
| `temperature` | Air temperature `theta` in the room, in degrees Celsius. |
| `static_pressure` | Static pressure `ps` in the room, in kilopascals. |

**Returns:** [`ReverberationSoundPowerResult`](/phonometry/reference/api/power/sound-power-reverberation/#reverberationsoundpowerresult).
