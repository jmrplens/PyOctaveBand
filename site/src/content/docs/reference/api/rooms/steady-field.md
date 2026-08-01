---
title: "room.steady_field"
description: "Steady-state sound field in a room: room constant, critical distance, level."
sidebar:
  label: "steady_field"
---

Steady-state sound field in a room: room constant, critical distance, level.

When a source of constant sound power runs in a room, the sound pressure level
at a receiver settles to a steady value made of two parts: the **direct field**
that falls with distance as any free-field source does, and the **reverberant
field** built up by the many wall reflections, which is (to the diffuse-field
approximation) the same everywhere in the room. This module gives the classical
statistical-acoustics relations between the source power, the room's absorption
and the received level (Bies, Hansen & Howard, *Engineering Noise Control* 5th
ed., 6.4; Kuttruff, *Room Acoustics* 6th ed., 5.6), the bridge between the sound
power of `phonometry.emission` and the received level indoors.

**Room constant** $R = S \bar{\alpha} / (1 - \bar{\alpha})$ (Bies
Equation (6.44)), with the total boundary area `S` and its area-weighted
mean Sabine absorption `alpha_bar`
([`phonometry.room.mean_absorption`](/phonometry/reference/api/rooms/reverberation-prediction/#mean_absorption)). `R` has units of
area and measures how much reverberant field a given power builds up: a live
room (small `alpha_bar`) has a small `R` and a loud reverberant field, a
dead room a large `R`.

**Steady-state level** (Bies Equation (6.43)):

$$
L_p = L_W + 10 \log_{10}\!\left( \frac{Q}{4 \pi r^2} + \frac{4}{R} \right) \quad \left[ +\, 10 \log_{10}\frac{\rho c}{400} \right]
$$

with the source directivity factor `Q` ($Q = 1$ omnidirectional,
$Q = 2$ on a
hard floor, ...), the distance `r` and the room constant `R`. The first
term inside the bracket is the direct field, the second the reverberant field.
Dropping the direct term (`distance=None`) leaves
$L_p = L_W + 10 \log_{10}(4/R)$,
the reverberant field alone: the level far from the source, and the one a
diffuse-field calculation such as a room-to-room transmission
([`phonometry.noise_control.room_to_room`](/phonometry/reference/api/noise_control/room-to-room/)) works from.
The optional $10 \log_{10}(\rho c / 400)$ term corrects for a
characteristic impedance `rho c` differing from the reference 400 Pa s/m; it
is about `+0.14 dB` at 20 degC and is omitted by default (Bies notes the
`~0.1 dB` it contributes).

**Critical distance** $r_c = \sqrt{Q R / (16 \pi)}$ is where the direct
and reverberant terms are equal (setting
$Q / (4 \pi r^2) = 4 / R$ in Equation
(6.43)); closer than `rc` the direct field dominates, farther the reverberant
field does. Kuttruff's reverberation distance (Equation (5.44),
$r_c = \sqrt{A / (16 \pi)}$ for $Q = 1$) uses the Sabine
absorption area $A = S \bar{\alpha}$ in place of the room constant
$R = A / (1 - \bar{\alpha})$;
the two coincide for a small `alpha_bar` and differ by the factor
$1 - \bar{\alpha}$ otherwise. This module uses the room constant, so
`rc` is exactly the crossover of its own [`steady_state_spl`](/phonometry/reference/api/rooms/steady-field/#steady_state_spl).

**Sound power model.** Both relations above assume the radiated power itself is
a property of the machine. Norton & Karczub, *Fundamentals of Noise and
Vibration Analysis for Engineers* 2nd ed., 4.6 (Table 4.5), point out that this
is only one of three positions: a *constant-power* source radiates the same
$\Pi_0$ wherever it stands, a *constant-volume* source is loaded by
nearby reflecting boundaries and radiates $\Pi_0 Q$ (up to `+9 dB` in
a corner, the conservative upper bound a design estimate uses), and a
*constant-pressure* source, a theoretical lower bound, radiates
$\Pi_0 / Q$. Real machines sit
between the first two whenever the source is closer to the boundary than a
wavelength. `source_model` selects which of the three
[`SOURCE_POWER_MODELS`](/phonometry/reference/api/rooms/steady-field/#source_power_models) is applied.

**Schroeder frequency** $f_s = 2000 \sqrt{T / V}$ (Kuttruff Equation
(3.44),
`V` in cubic metres, `T` in seconds) marks the boundary between the
modal low-frequency regime -- where discrete room modes rule and the diffuse
assumption of `R` and `rc` fails -- and the high-frequency regime of
overlapping modes where the statistical field of this module applies.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## critical_distance

```python
critical_distance(
    room_constant: ArrayLike,
    *,
    directivity: float = 1.0,
) -> np.ndarray | float
```

Critical (reverberation) distance
$r_c = \sqrt{Q R / (16 \pi)}$.

The distance at which the direct and reverberant fields of
[`steady_state_spl`](/phonometry/reference/api/rooms/steady-field/#steady_state_spl) are equal (Bies Equation (6.43) crossover;
Kuttruff Equation (5.44) states the $Q = 1$ form with the Sabine
absorption area $A = S \bar{\alpha}$ instead of the room constant
`R`, the two differing by $1 - \bar{\alpha}$).

**Parameters**

| Name | Description |
| :--- | :--- |
| `room_constant` | Room constant `R`, m2 (scalar or per-band); from [`room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant). |
| `directivity` | Source directivity factor `Q` (`1` omnidirectional, `2` on one reflecting plane, `4` in an edge, `8` in a corner). |

**Returns:** The critical distance `rc`, m.

## room_constant

```python
room_constant(
    surface_area: float,
    mean_absorption: ArrayLike,
) -> np.ndarray | float
```

Room constant $R = S \bar{\alpha} / (1 - \bar{\alpha})$
(Bies Equation (6.44)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `surface_area` | Total boundary area `S` of the room, m2. |
| `mean_absorption` | Area-weighted mean Sabine absorption `alpha_bar` in `(0, 1)` (scalar or per-band); e.g. from [`phonometry.room.mean_absorption`](/phonometry/reference/api/rooms/reverberation-prediction/#mean_absorption). |

**Returns:** The room constant `R`, m2; a float for a scalar input, otherwise a per-band array.

## schroeder_frequency

```python
schroeder_frequency(
    reverberation_time: ArrayLike,
    volume: float,
) -> np.ndarray | float
```

Schroeder frequency $f_s = 2000 \sqrt{T / V}$
(Kuttruff Equation (3.44)).

The frequency above which room modes overlap (on average three
eigenfrequencies per resonance half-width) so the statistical, diffuse
field of this module applies; below it the sound field is ruled by discrete
modes and `R` / `rc` lose their meaning.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reverberation_time` | Reverberation time `T`, s (scalar or per-band). |
| `volume` | Room volume `V`, m3. |

**Returns:** The Schroeder frequency, Hz.

## SOURCE_POWER_MODELS

*Constant* (`dict`).

```python
SOURCE_POWER_MODELS = {'constant_power': 0.0, 'constant_volume': 1.0, 'constant_pressure': -1.0}
```

## steady_state_field

```python
steady_state_field(
    sound_power_level: float,
    surface_area: float,
    mean_absorption: float,
    *,
    distances: ArrayLike | None = None,
    directivity: float = 1.0,
    characteristic_impedance: float | None = None,
) -> SteadyFieldResult
```

Steady-state SPL versus distance for one source in a room (Bies 6.4).

Builds the room constant from `surface_area` and `mean_absorption`
(Bies Equation (6.44)), then evaluates the direct, reverberant and combined
fields (Equation (6.43)) over a distance grid together with the critical
distance (crossover of the two fields).

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_power_level` | Source sound power level `Lw`, dB re 1 pW. |
| `surface_area` | Total boundary area `S`, m2. |
| `mean_absorption` | Mean Sabine absorption `alpha_bar` in `(0, 1)`. |
| `distances` | Distance grid `r`, m; default 30 points log-spaced from one tenth of the critical distance to ten times it. |
| `directivity` | Source directivity factor `Q` (default 1). |
| `characteristic_impedance` | Optional `rho c` for the Bies $10 \log_{10}(\rho c / 400)$ term (`None` omits it). |

**Returns:** A [`SteadyFieldResult`](/phonometry/reference/api/rooms/steady-field/#steadyfieldresult).

## steady_state_spl

```python
steady_state_spl(
    sound_power_level: ArrayLike,
    distance: ArrayLike | None,
    room_constant: ArrayLike,
    *,
    directivity: float = 1.0,
    source_model: str = 'constant_power',
    characteristic_impedance: float | None = None,
) -> np.ndarray | float
```

Steady-state sound pressure level in a room (Bies Equation (6.43)).

$$
L_p = L_W + 10 \log_{10}\!\left( \frac{Q}{4 \pi r^2} + \frac{4}{R} \right)
$$

plus the optional $10 \log_{10}(\rho c / 400)$
characteristic-impedance term. The bracket sums
the direct field $Q / (4 \pi r^2)$ and the (position-independent)
reverberant field $4 / R$. Passing `distance=None` drops the
direct term and leaves the reverberant field alone,
$L_p = L_W + 10 \log_{10}(4 / R)$:
the limit far from the source, and the level a diffuse-field calculation
such as a room-to-room transmission uses.

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_power_level` | Source sound power level `Lw`, dB re 1 pW (scalar or per-band); e.g. from `phonometry.emission`. |
| `distance` | Source-receiver distance `r`, m (scalar or array), or `None` for the reverberant field alone. |
| `room_constant` | Room constant `R`, m2 (scalar or per-band); from [`room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant). |
| `directivity` | Source directivity factor `Q` (default 1; `2` on one reflecting plane, `4` in an edge, `8` in a corner). |
| `source_model` | Sound power model of Norton & Karczub 2e Table 4.5: `"constant_power"` (default, $\Pi = \Pi_0$, the position of the source does not change its radiated power), `"constant_volume"` ($\Pi = \Pi_0 Q$, reflecting boundaries raise the radiated power by $10 \lg Q$; the conservative upper bound) or `"constant_pressure"` ($\Pi = \Pi_0 / Q$, the theoretical lower bound). See [`SOURCE_POWER_MODELS`](/phonometry/reference/api/rooms/steady-field/#source_power_models). |
| `characteristic_impedance` | Air characteristic impedance `rho c`, Pa s/m. When given, the $10 \log_{10}(\rho c / 400)$ term is added (about `+0.14 dB` at 20 degC where $\rho c = 413$); `None` (default) omits it, matching the common textbook form. |

**Returns:** The steady-state SPL `Lp`, dB; a float for scalar inputs, otherwise an array broadcasting `sound_power_level`, `distance` and `room_constant`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `distance` or `room_constant` is not positive and finite, or `source_model` is not one of [`SOURCE_POWER_MODELS`](/phonometry/reference/api/rooms/steady-field/#source_power_models). |

## SteadyFieldResult

```python
SteadyFieldResult(
    distances: np.ndarray,
    direct: np.ndarray,
    reverberant: np.ndarray,
    total: np.ndarray,
    critical_distance: float,
    room_constant: float,
    sound_power_level: float,
    directivity: float,
)
```

Steady-state SPL versus distance in a room, split direct / reverberant.

**Attributes**

| Name | Description |
| :--- | :--- |
| `distances` | Source-receiver distances `r`, m. |
| `direct` | Direct-field level $L_W + 10 \log_{10}(Q / (4 \pi r^2))$ per distance, dB. |
| `reverberant` | Reverberant-field level $L_W + 10 \log_{10}(4 / R)$, dB (constant across distance; broadcast to the distance grid). |
| `total` | Combined steady-state level (Bies Equation (6.43)), dB. |
| `critical_distance` | Critical distance `rc`, m, where direct equals reverberant. |
| `room_constant` | Room constant `R`, m2. |
| `sound_power_level` | Source sound power level `Lw`, dB re 1 pW. |
| `directivity` | Source directivity factor `Q`. |

### SteadyFieldResult.plot()

```python
SteadyFieldResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot direct, reverberant and total SPL against distance.

Marks the critical distance `rc` where the direct and reverberant
fields cross. Requires matplotlib (`pip install phonometry[plot]`);
returns the `Axes`.
