---
title: "emission.sound_power"
description: "Sound power level of a noise source from sound pressure measurements over an enveloping measurement surface: ISO 3744:2010 (engineering, accuracy grade 2) and ISO 3746:2010 (survey, accuracy grade 3)."
sidebar:
  label: "sound_power"
---

Sound power level of a noise source from sound pressure measurements over an
enveloping measurement surface: ISO 3744:2010 (engineering, accuracy grade 2)
and ISO 3746:2010 (survey, accuracy grade 3).

The source stands on one (or more) reflecting plane(s). Sound pressure levels
are measured at an array of microphone positions on a hypothetical surface of
area `S` enveloping the source (a hemisphere or a right parallelepiped). The
sound power level follows from the energy-averaged pressure level, the
background correction $K_1$, the environmental correction $K_2$
and the surface area (ISO 3744:2010 clause 8.2, equations (12), (16)-(18)):

$$
\overline{L_p} = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}} \sum_i 10^{0.1 L_{pi}} \right] \tag{Eq. 12}
$$

$$
K_1 = -10 \log_{10}\!\left( 1 - 10^{-0.1 \Delta L_p} \right) \tag{Eq. 16}
$$

$$
K_2 = 10 \log_{10}\!\left( 1 + \frac{4S}{A} \right) \tag{Eq. A.2}
$$

$$
L_p = \overline{L_p} - K_1 - K_2 \tag{Eq. 17}
$$

$$
L_W = L_p + 10 \log_{10}\frac{S}{S_0}, \qquad S_0 = 1~\text{m}^2 \tag{Eq. 18}
$$

The measurement surface area is a closed form of the source geometry: a full
hemisphere $S = 2\pi r^2$ (half $\pi r^2$, quarter
$\pi r^2/2$) for one, two or three reflecting planes (ISO 3744 clause
7.2.3); a parallelepiped $S = 4(ab+bc+ca)$ with $a = 0.5\,l_1+d$,
$b = 0.5\,l_2+d$, $c = l_3+d$ for one plane (clause 7.2.4,
equations (9)-(11)).

The A-weighted sound power level is combined from band levels with the
A-weighting band corrections $C_k$ of ISO 3744 Annex E (Tables
E.1/E.2):

$$
L_{W\mathrm{A}} = 10 \log_{10}\!\left[ \sum_k 10^{0.1 (L_{Wk} + C_k)} \right] \tag{Eq. E.1}
$$

ISO 3746:2010 shares the surfaces, the energy average and the LW/K1/K2 forms
but is coarser: fewer microphone positions (clause 8.2.1), a background
criterion of 3 dB instead of 6 dB (clause 8.4.1) and validity up to
$K_{2\mathrm{A}} \le 7$ dB instead of 4 dB (clause 4.3).

A noise burst or a transient emission has no steady power to report, and both
standards describe it by the **sound energy level** $L_J = 10 \log_{10} (J/J_0)$ instead, $J = \int P(t)\,\mathrm{d}t$ in joules and
$J_0 = 1$ pJ (ISO 3744:2010 clauses 3.22 and 3.23). Its determination
(clause 8.3; ISO 3746:2010 clause 8.4) is the chain above with the single
event time-integrated sound pressure level $L_E = 10 \log_{10}\!\left[ \int p^2\,\mathrm{d}t / E_0\right]$, $E_0 = (20\ \mu\mathrm{Pa})^2\, \mathrm{s}$ (clause 3.4), in place of the time-averaged $L_p$: the
$N_\mathrm{e}$ events at each position are combined into the level of one
event (Eq. 19 or Eq. 20), the positions are averaged as in 8.2.2 (Eq. 12), the
background and the environment are corrected by the same $K_1$ and
$K_2$ (Eq. 21, 22) and the surface term closes it:

$$
L_J = \overline{L_E} + 10 \log_{10}\frac{S}{S_0} \tag{Eq. 23}
$$

For a source that is steady over the whole interval $T$, clause 3.4
NOTE 1 gives $L_E = L_{p,T} + 10 \log_{10}(T/T_0)$ with $T_0 = 1$
s, so $L_J = L_W + 10 \log_{10}(T/T_0)$: the energy a steady source
radiates in $T$ seconds. Annex E carries the band levels to the
A-weighted $L_{J\mathrm{A}}$ with the same $C_k$ as $L_{W\mathrm{A}}$
(Eq. E.2), and Annex G refers either level to the reference atmosphere with
the corrections $C_1 + C_2$ (Eq. G.1, G.3), required above 500 m of
altitude or below 10 degrees C (clauses 8.2.5 and 8.3.6).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## background_noise_correction

```python
background_noise_correction(
    source_levels: np.ndarray,
    background_levels: np.ndarray,
    grade: Grade = 'engineering',
) -> np.ndarray
```

Background-noise correction `K1` per band (ISO 3744:2010 Eq. 16).

$K_1 = -10 \log_{10}\left( 1 - 10^{-0.1 \Delta L_p} \right)$ with
$\Delta L_p = L_{\text{source}} - L_{\text{background}}$. For
$\Delta L_p$ strictly above the upper criterion (15 dB engineering,
10 dB survey) the background is negligible and $K_1 = 0$; at the
criterion itself Eq. (16) still applies (ISO 3744:2010, 8.2.3:
$6 \le \Delta L_p \le 15$ dB; ISO 3746:2010, 8.3.3:
$3 \le \Delta L_p \le 10$ dB). For $\Delta L_p$ below the
lower criterion (6 dB engineering, 3 dB survey) the accuracy is reduced:
`K1` is clamped to its value at that criterion and a
[`SoundPowerWarning`](/phonometry/reference/api/power/sound-power/#soundpowerwarning) is emitted, the result then being an upper
bound (clause 8.2.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_levels` | Levels with the source operating, in decibels. |
| `background_levels` | Background-noise levels, in decibels. |
| `grade` | `'engineering'` (ISO 3744) or `'survey'` (ISO 3746). |

**Returns:** `K1` per band, in decibels.

## environmental_correction

```python
environmental_correction(
    surface_area: float,
    *,
    absorption_area: float | np.ndarray | None = None,
    reverberation_time: float | np.ndarray | None = None,
    volume: float | None = None,
    mean_absorption_coefficient: float | np.ndarray | None = None,
    room_surface: float | None = None,
) -> float | np.ndarray
```

Environmental correction `K2` (ISO 3744:2010 Eq. A.2).

$K_2 = 10 \log_{10}\left( 1 + 4 S / A \right)$ where `A` is the
equivalent sound absorption area of the room. `A` is taken directly
from `absorption_area`, or from
the Sabine reverberation time $A = 0.16 V / T$ (Eq. A.3,
`reverberation_time` + `volume`), or from the mean absorption
coefficient $A = \alpha S_v$ (Eq. A.7,
`mean_absorption_coefficient` + `room_surface`). With no room data the
field is treated as free and $K_2 = 0$;
supplying only one member of a pair raises `ValueError` rather than
silently falling back to the free-field result.

The room absorption is frequency dependent (`T`, `alpha` and hence `A`
vary with the band). Passing `absorption_area`, `reverberation_time` or
`mean_absorption_coefficient` as a per-band array returns `K2` per band
with that shape; scalar inputs return a scalar, unchanged.

**Parameters**

| Name | Description |
| :--- | :--- |
| `surface_area` | Measurement surface area `S`, in square metres. |
| `absorption_area` | Equivalent absorption area `A` (m^2), scalar or per band. |
| `reverberation_time` | Sabine `T` (s), scalar or per band, with `volume` (Eq. A.3). |
| `volume` | Room volume `V` (m^3), with `reverberation_time`. |
| `mean_absorption_coefficient` | `alpha` in (0, 1], scalar or per band, with `room_surface` (Eq. A.7). |
| `room_surface` | Room boundary area `Sv` (m^2), with `alpha`. |

**Returns:** `K2` in decibels; a scalar for scalar inputs, otherwise an array per band.

## mean_single_event_level

```python
mean_single_event_level(
    levels: np.ndarray,
    *,
    events: int | None = None,
) -> np.ndarray
```

Mean single event time-integrated level of one event (ISO 3744 Eq. 19/20).

Two ways of measuring $N_\mathrm{e}$ sound emission events reach the
same quantity, the single event time-integrated sound pressure level of
*one* event at a microphone position. Measured one at a time
(`events=None`), the first axis of `levels` holds one entry per event
and the mean is their energy average, ISO 3744:2010 Eq. (19):

$$
L'_{Ei(\mathrm{ST})} = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{e}} \sum_{q=1}^{N_\mathrm{e}} 10^{0.1 L'_{Ei,q(\mathrm{ST})}} \right]
$$

Measured as one level that encompasses $N_\mathrm{e}$ successive
events (`events` $= N_\mathrm{e}$), the level of one event is the
measurement less $10 \log_{10} N_\mathrm{e}$, Eq. (20):

$$
L'_{Ei(\mathrm{ST})} = L'_{Ei,N_\mathrm{e}(\mathrm{ST})} - 10 \log_{10} N_\mathrm{e}
$$

ISO 3746:2010 Eq. (16)/(17) and ISO 3741:2010 Eq. (22)/(23) print the
same pair. Both modes require at least five events (clause 8.3.1); fewer
are accepted with a [`SoundPowerWarning`](/phonometry/reference/api/power/sound-power/#soundpowerwarning).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | With `events=None`, the per-event levels in decibels with the events on the first axis (`(Ne,)`, `(Ne, NB)` or `(Ne, NM, NB)`); otherwise the one measurement encompassing `events` events, of any shape. |
| `events` | `None` when each entry of the first axis is one event, else the number of events the single measurement encompasses. |

**Returns:** The level of one event, in decibels: `levels` less its first axis for `events=None`, the shape of `levels` otherwise.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for non-finite levels, an empty first axis or a non-positive `events`. |

## measurement_positions

```python
measurement_positions(
    surface: Surface,
    *,
    radius: float,
    reflecting_planes: int = 1,
    tones: bool = True,
    grade: Grade = 'engineering',
) -> np.ndarray
```

Normative microphone coordinates on the measurement surface.

For a `'hemisphere'` the coordinates come from ISO 3744:2010 Annex B:
Table B.1 for sources that may emit discrete tones (`tones=True`) and
Table B.2 for broadband sources. The engineering grade uses the 10 key
positions for one reflecting plane (5 for two, 3 for three); the survey
grade uses the reduced arrays of ISO 3746:2010 clause 8.2.1 (positions
4, 5, 6, 10 for one plane). Coordinates are scaled by `radius` and
returned as an `(N, 3)` array of Cartesian `(x, y, z)` in metres.

**Parameters**

| Name | Description |
| :--- | :--- |
| `surface` | `'hemisphere'` (only shape with a coordinate table). |
| `radius` | Hemisphere radius `r`, in metres. |
| `reflecting_planes` | Number of reflecting planes (1, 2 or 3). |
| `tones` | If True use Table B.1, else Table B.2. |
| `grade` | `'engineering'` or `'survey'`. |

**Returns:** `(N, 3)` microphone coordinates, in metres.

## plot_microphone_positions

```python
plot_microphone_positions(
    positions: ArrayLike,
    ax: Axes3D | None = None,
    *,
    radius: float | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes3D
```

Draw a microphone position array on its measurement surface, in 3-D.

Numbered microphone points with a wireframe of the hemisphere (or full
sphere when positions dip below the reflecting plane) of the given
`radius`; pairs with
[`measurement_positions`](/phonometry/reference/api/power/sound-power/#measurement_positions) and
[`precision_positions`](/phonometry/reference/api/power/sound-power-anechoic/#precision_positions), whose `(N, 3)`
arrays it accepts directly.

**Parameters**

| Name | Description |
| :--- | :--- |
| `positions` | Cartesian microphone positions, shape `(N, 3)`, in metres. |
| `ax` | Existing 3-D axes (`projection="3d"`), or `None` to create a figure. |
| `radius` | Surface radius for the wireframe, in metres; `None` uses the largest position norm. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the microphone `scatter`. |

**Returns:** The 3-D axes.

## reference_atmosphere_correction

```python
reference_atmosphere_correction(
    temperature: float,
    static_pressure: float | None = None,
    *,
    altitude: float | None = None,
) -> ReferenceAtmosphereCorrection
```

Corrections to reference meteorological conditions (ISO 3744 Annex G).

A sound power or sound energy level determined by Eq. (18) or Eq. (23)
holds for the meteorological conditions at the time and place of the test.
Above 500 m of altitude or below 10 degrees C the standard requires it to
be carried to the reference static pressure 101,325 kPa and air
temperature 23,0 degrees C (clauses 8.2.5 and 8.3.6) by adding

$$
C_1 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}} + 5 \log_{10}\frac{273.15 + \theta}{\theta_0}, \qquad C_2 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}} + 15 \log_{10}\frac{273.15 + \theta}{\theta_1}
$$

with $\theta_0 = 314$ K and $\theta_1 = 296$ K, so that
$L_{W\mathrm{ref,atm}} = L_W + C_1 + C_2$ (Eq. G.1) and
$L_{J\mathrm{ref,atm}} = L_J + C_1 + C_2$ (Eq. G.3). $C_1$
accounts for the different reference quantities of the pressure and power
decibels through the characteristic impedance of the air, and is omitted
when $K_2$ came from the absolute comparison test of A.2;
$C_2$ is the radiation-impedance correction of a monopole, a mean
value for other sources. Where the static pressure was not measured it is
estimated from the altitude of the site by Eq. (G.2),
$p_\mathrm{s} = p_{\mathrm{s},0}(1 - aH_\mathrm{a})^b$, with
$a = 2.2560 \times 10^{-5}$ m^-1 and $b = 5.2553$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Air temperature `theta` at the test, in degrees C. |
| `static_pressure` | Static pressure `ps` at the test, in kilopascals; give this or `altitude`. |
| `altitude` | Altitude `Ha` of the test site, in metres, from which `ps` is estimated by Eq. (G.2) when it was not measured. |

**Returns:** [`ReferenceAtmosphereCorrection`](/phonometry/reference/api/power/sound-power/#referenceatmospherecorrection) with `c1`, `c2`, their `total` and the static pressure used.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if neither or both of `static_pressure` and `altitude` are given, or either is out of range, or `temperature` is not above absolute zero. |

## ReferenceAtmosphereCorrection

```python
ReferenceAtmosphereCorrection(
    c1: float,
    c2: float,
    static_pressure: float,
    temperature: float,
)
```

The two Annex G corrections to reference meteorological conditions.

`c1` is the reference-quantity correction and `c2` the
radiation-impedance correction of ISO 3744:2010 Annex G, both in
decibels; `total` is their sum, the whole of what Eq. (G.1) adds to
$L_W$ and Eq. (G.3) to $L_J$. `static_pressure` is the
$p_\mathrm{s}$ the corrections were evaluated at, in kilopascals,
whether it was measured or estimated from the altitude by Eq. (G.2), and
`temperature` the air temperature $\theta$, in degrees Celsius.

### ReferenceAtmosphereCorrection.total

*property*

`c1 + c2`, in decibels: the correction Eq. (G.1)/(G.3) applies.

## RoomEnvironment

```python
RoomEnvironment(
    absorption_area: float | np.ndarray | None = None,
    reverberation_time: float | np.ndarray | None = None,
    volume: float | None = None,
    mean_absorption_coefficient: float | np.ndarray | None = None,
    room_surface: float | None = None,
)
```

Room data behind the environmental correction `K2` (ISO 3744 Annex A).

The three routes the standard offers to the equivalent sound absorption area
`A` of the test room, in the order [`environmental_correction`](/phonometry/reference/api/power/sound-power/#environmental_correction) tries
them: `A` itself, the Sabine reverberation time with the room volume
($A = 0.16 V / T$, Eq. A.3) and the mean absorption coefficient with
the area of the room boundaries ($A = \alpha S_v$, Eq. A.7). Each
route is a pair that must be given whole; the empty environment carries no
room data at all, which is the free field ($K_2 = 0$).

Every field may also be a per-band array, in which case `K2` comes out
per band with that shape.

**Parameters**

| Name | Description |
| :--- | :--- |
| `absorption_area` | Equivalent absorption area `A` (m^2), scalar or per band. |
| `reverberation_time` | Sabine `T` (s), scalar or per band, with `volume` (Eq. A.3). |
| `volume` | Room volume `V` (m^3), with `reverberation_time`. |
| `mean_absorption_coefficient` | `alpha` in (0, 1], scalar or per band, with `room_surface` (Eq. A.7). |
| `room_surface` | Room boundary area `Sv` (m^2), with `alpha`. |

## sound_energy_pressure

```python
sound_energy_pressure(
    levels_positions: np.ndarray,
    surface: Surface,
    *,
    radius: float | None = None,
    dimensions: tuple[float, float, float] | None = None,
    distance: float | None = None,
    reflecting_planes: int = 1,
    events: int | None = None,
    background_levels: np.ndarray | None = None,
    integration_time: float | None = None,
    frequencies: np.ndarray | None = None,
    room: RoomEnvironment | None = None,
    grade: Grade = 'engineering',
    omc_uncertainty: float = 0.0,
) -> SoundEnergyResult
```

Sound energy level of a noise burst from surface single event levels
(ISO 3744:2010 clause 8.3, ISO 3746:2010 clause 8.4).

The single event time-integrated sound pressure levels
$L'_{Ei(\mathrm{ST})}$ are measured simultaneously at every
microphone position through a period that encompasses the full burst
(clause 8.3.1; a traversing microphone is not permitted). `levels_positions`
is either the `(NM, NB)` mean single event level of one event at each
position, a `(Ne, NM, NB)` array of the $N_\mathrm{e}$ events measured
one at a time (reduced by Eq. 19), or the `(NM, NB)` level of one
measurement encompassing `events` successive events (reduced by Eq. 20).
The positions are then energy-averaged as in 8.2.2 (clause 8.3.3), the
surface level is corrected for background noise and for the test
environment and the surface term added:

$$
L_J = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}} \sum_i 10^{0.1 L'_{Ei(\mathrm{ST})}} \right] - K_1 - K_2 + 10 \log_{10}\frac{S}{S_0} \tag{Eq. 22, 23}
$$

$K_1$ follows Eq. (21) with the same criteria as the time-averaged
path ([`background_noise_correction`](/phonometry/reference/api/power/sound-power/#background_noise_correction): 6 dB to 15 dB engineering, 3 dB
to 10 dB survey, clamped below the lower criterion with a
[`SoundPowerWarning`](/phonometry/reference/api/power/sound-power/#soundpowerwarning)), and $K_2$ the room data in `room`
(Annex A). The background is the time-averaged level the standard has
measured over the same integration time $T$ as the events, and it
is compared as its exposure over that $T$,
$L_{p(\mathrm{B})} + 10 \log_{10}(T/T_0)$ (clause 3.4 NOTE 1), so
that the energies Eq. (21) subtracts share one reference; this is why
`integration_time` is required whenever `background_levels` is given.
The surface area `S` comes from the geometry exactly as in
[`sound_power_pressure`](/phonometry/reference/api/power/sound-power/#sound_power_pressure), and so do the minimum number of positions,
the A-weighted total (Annex E, Eq. E.2) and the expanded uncertainty,
which clause 9.1 Eq. (24) makes the same for $L_J$ as for
$L_W$.

The level holds for the meteorological conditions of the test; above
500 m of altitude or below 10 degrees C add the Annex G correction of
[`reference_atmosphere_correction`](/phonometry/reference/api/power/sound-power/#reference_atmosphere_correction) (Eq. G.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels_positions` | Single event levels, in decibels: `(NM, NB)` means, `(Ne, NM, NB)` per-event levels, or `(NM, NB)` of one measurement of `events` events. |
| `surface` | `'hemisphere'` or `'box'`. |
| `radius` | Hemisphere radius `r` (metres), for `surface='hemisphere'`. |
| `dimensions` | Reference box `(l1, l2, l3)` (metres), for `'box'`. |
| `distance` | Measurement distance `d` (metres), for `'box'`. |
| `reflecting_planes` | Number of reflecting planes (1, 2 or 3). |
| `events` | The number of events `Ne` one measurement encompasses (Eq. 20); `None` when `levels_positions` is per event or already the mean of one event. |
| `background_levels` | `(NM, NB)` time-averaged background levels for `K1`, or a single spectrum `(NB,)` / `(1, NB)` broadcast to every position; measured over the same interval as the events. |
| `integration_time` | The interval `T` of the single event levels, in seconds; required with `background_levels`. |
| `frequencies` | Band mid-band frequencies (Hz) for the A-weighted total. |
| `room` | Room absorption data behind `K2` ([`RoomEnvironment`](/phonometry/reference/api/power/sound-power/#roomenvironment)); `None` is a room with no data at all, i.e. a free field ($K_2 = 0$). |
| `grade` | `'engineering'` (ISO 3744) or `'survey'` (ISO 3746). |
| `omc_uncertainty` | `sigma_omc` (dB), operating/mounting instability. |

**Returns:** [`SoundEnergyResult`](/phonometry/reference/api/power/sound-power/#soundenergyresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a malformed or non-finite level array, a geometry that does not describe the surface, too few positions, a background without its `integration_time`, or a mismatched `frequencies` length. |

## sound_power_pressure

```python
sound_power_pressure(
    levels_positions: np.ndarray,
    surface: Surface,
    *,
    radius: float | None = None,
    dimensions: tuple[float, float, float] | None = None,
    distance: float | None = None,
    reflecting_planes: int = 1,
    background_levels: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    room: RoomEnvironment | None = None,
    grade: Grade = 'engineering',
    omc_uncertainty: float = 0.0,
) -> SoundPowerResult
```

Sound power level from surface pressure levels (ISO 3744/3746:2010).

`levels_positions` is an `(NM, NB)` array of time-averaged sound
pressure levels: one row per microphone position, one column per frequency
band (or a single column for a directly measured A-weighted level). The
surface-averaged level is corrected for background noise (`K1`, from
`background_levels`) and for the test environment (`K2`, from the
`room` absorption data) and combined with the measurement surface area:

$$
L_W = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{M}} \sum_i 10^{0.1 L_{pi}} \right] - K_1 - K_2 + 10 \log_{10}\frac{S}{S_0}
$$

The surface area `S` is computed from the geometry: `radius` for a
`'hemisphere'` (clause 7.2.3) or `dimensions` + `distance` for a
`'box'` (clause 7.2.4). When `frequencies` are given the A-weighted
sound power level is combined via ISO 3744 Annex E.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels_positions` | `(NM, NB)` sound pressure levels, in decibels. |
| `surface` | `'hemisphere'` or `'box'`. |
| `radius` | Hemisphere radius `r` (metres), for `surface='hemisphere'`. |
| `dimensions` | Reference box `(l1, l2, l3)` (metres), for `'box'`. |
| `distance` | Measurement distance `d` (metres), for `'box'`. |
| `reflecting_planes` | Number of reflecting planes (1, 2 or 3). |
| `background_levels` | `(NM, NB)` background levels for `K1`, or a single spectrum `(NB,)` / `(1, NB)` broadcast to every position. |
| `frequencies` | Band mid-band frequencies (Hz) for the A-weighted total. |
| `room` | Room absorption data behind `K2` ([`RoomEnvironment`](/phonometry/reference/api/power/sound-power/#roomenvironment)); `None` is a room with no data at all, i.e. a free field ($K_2 = 0$). |
| `grade` | `'engineering'` (ISO 3744) or `'survey'` (ISO 3746). |
| `omc_uncertainty` | `sigma_omc` (dB), operating/mounting instability. |

**Returns:** [`SoundPowerResult`](/phonometry/reference/api/power/sound-power/#soundpowerresult).

## SoundEnergyResult

```python
SoundEnergyResult(
    frequencies: np.ndarray | None,
    sound_energy_level: np.ndarray,
    surface_event_level: np.ndarray,
    mean_event_level: np.ndarray,
    background_correction: np.ndarray,
    environmental_correction: np.ndarray,
    directivity_index: np.ndarray,
    surface_area: float,
    sound_energy_level_a: float,
    uncertainty: float,
    grade: str,
    events: int | None,
    integration_time: float | None,
)
```

Result of a sound energy level determination from surface single event
levels (ISO 3744:2010 clause 8.3, ISO 3746:2010 clause 8.4).

`sound_energy_level` is the per-band `LJ` (ISO 3744 Eq. 23);
`surface_event_level` the surface single event time-integrated sound
pressure level $\overline{L_E}$ after the K1/K2 corrections (Eq. 22);
`mean_event_level` the raw energy-averaged level
$\overline{L'_E}(\mathrm{ST})$ over the positions (clause 8.3.3, as
Eq. 12). `background_correction` (K1, Eq. 21) and
`environmental_correction` (K2) are per band. `sound_energy_level_a` is
the A-weighted total `LJA` (Eq. E.2), computed only when `frequencies`
are supplied; for a single band it equals `LJ`, and for several bands
without `frequencies` it is `NaN` (A-weighting needs the band centres).
`directivity_index` is the apparent directivity index per microphone
position and band, shape `(NM, NB)`, formed from the single event levels
exactly as clause 3.24 allows. `uncertainty` is the expanded uncertainty
$U = 2\sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}$ (95 %),
which clause 9.1 Eq. (24) makes the same for `LJ` as for `LW`.
`events` is the number of single sound emission events $N_\mathrm{e}$
the levels were reduced from, or `None` when the caller supplied the
per-position mean single event levels directly; `integration_time` is
the interval $T$ of the single event levels, in seconds, or `None`
when no background correction needed it.

### SoundEnergyResult.plot()

```python
SoundEnergyResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the LJ spectrum with the A-weighted total annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## SoundPowerResult

```python
SoundPowerResult(
    frequencies: np.ndarray | None,
    sound_power_level: np.ndarray,
    surface_pressure_level: np.ndarray,
    mean_pressure_level: np.ndarray,
    background_correction: np.ndarray,
    environmental_correction: np.ndarray,
    directivity_index: np.ndarray,
    surface_area: float,
    sound_power_level_a: float,
    uncertainty: float,
    grade: str,
)
```

Result of a sound power determination from surface pressure levels.

`sound_power_level` is the per-band `LW` (ISO 3744 Eq. 18);
`surface_pressure_level` the surface SPL `Lp` after the K1/K2
corrections (Eq. 17); `mean_pressure_level` the raw energy-averaged
level `Lp'(ST)` (Eq. 12). `background_correction` (K1) and
`environmental_correction` (K2) are per band. `sound_power_level_a` is
the A-weighted total `LWA` (Eq. E.1), computed only when `frequencies`
are supplied; for a single band it equals `LW`, and for several bands
without `frequencies` it is `NaN` (A-weighting needs the band centres).
`directivity_index` is the apparent directivity index `DIi*` per
microphone position and frequency band, shape `(NM, NB)` (Eq. 7,
evaluated per band per clause 8.4). `uncertainty` is the expanded
uncertainty
$U = 2\sqrt{\sigma_{\mathrm{R}0}^2 + \sigma_\mathrm{omc}^2}$ (95 %, ISO 3744
clause 9.5).

### SoundPowerResult.declare()

```python
SoundPowerResult.declare(
    *,
    uncertainty: float | None = None,
    mode: str = 'Operating mode 1',
    emission_pressure_level: float | None = None,
    emission_pressure_uncertainty: float | None = None,
    verification_level: float | None = None,
    machine: str | None = None,
    operating_conditions: str | None = None,
    noise_test_code: str | None = None,
    basic_standards: str | Sequence[str] = (),
    form: DeclarationForm = 'dual-number',
) -> NoiseEmissionDeclaration
```

Build an ISO 4871:1996 noise-emission declaration from this result.

Wraps the A-weighted sound power level `LWA` of this measurement as the
declared measured value `L_WA` of a single operating mode, with the
uncertainty `K_WA` defaulting to the result's own expanded uncertainty
`U` (ISO 3744/3746 clause 9.5). The declared single-number value is
$L_{W\mathrm{Ad}} = L_{W\mathrm{A}} + K_{W\mathrm{A}}$ (ISO 4871 clause 3.15).

**Parameters**

| Name | Description |
| :--- | :--- |
| `uncertainty` | `K_WA` in decibels; defaults to this result's expanded uncertainty `uncertainty`. |
| `mode` | Operating-mode label for the declaration column. |
| `emission_pressure_level` | Optional A-weighted emission sound pressure level `L_pA` at a work station, in decibels re 20 uPa. |
| `emission_pressure_uncertainty` | `K_pA` in decibels; required with `emission_pressure_level`. |
| `verification_level` | Optional verification measurement `L_1` of the A-weighted sound power level (ISO 4871 clause 6). |
| `machine` | Machine identification (clause 5 a). |
| `operating_conditions` | Operating/mounting conditions (clause 5 c). |
| `noise_test_code` | Noise test code the values were determined to (clause 5 b). |
| `basic_standards` | Basic emission standard(s) used (clause 5 b). |
| `form` | `"dual-number"` (default) or `"single-number"`. |

**Returns:** A single-mode [`NoiseEmissionDeclaration`](/phonometry/reference/api/power/declaration/#noiseemissiondeclaration).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the A-weighted sound power level is not finite (several bands were combined without `frequencies`). |

### SoundPowerResult.plot()

```python
SoundPowerResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the LW spectrum with the A-weighted total annotated.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### SoundPowerResult.report()

```python
SoundPowerResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 3744/3746 sound-power determination fiche to a PDF.

Writes a one-page sound-power test sheet: the standard-basis line naming
the applied method and accuracy grade (ISO 3744:2010 engineering grade 2
or ISO 3746:2010 survey grade 3), an optional metadata header (client,
noise source, test environment, instrumentation, climate, date), a
per-band table (nominal octave/one-third-octave frequency, the surface
sound-pressure level `Lp` and the band sound-power level `LW`), the
sound-power spectrum `LW(f)`, the boxed A-weighted sound power level
`LWA` (dB re 1 pW) with the total `LW`, the expanded uncertainty
`U` and the measurement surface area `S`, an optional verdict row
against a declared limit, and a measurement-basis strip stating the
applied background (`K1`) and environmental (`K2`) corrections.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the noise source, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared A-weighted sound-power limit the fiche checks the result against (lower is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the energy-averaged level `Lp'` and the background (`K1`) and environmental (`K2`) corrections. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## SoundPowerWarning

Non-fatal qualification issue in any of the sound-power methods.

Emitted for ISO 3744/3746 background margin below the criterion and for
`K2` beyond the method's validity limit (8.2.3, 4.3.2); for ISO 3741
reverberation-room qualification (room volume vs Table 1, mean absorption)
and microphone/source-position sampling; and for ISO 9614-2 negative total
partial power and unmet field-indicator criteria. Where a lower criterion
is only just met the returned levels represent upper bounds and must be
reported as such.
