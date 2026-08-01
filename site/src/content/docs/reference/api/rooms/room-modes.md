---
title: "room.room_modes"
description: "Normal modes of a rectangular room: frequencies, kinds, count and density."
sidebar:
  label: "room_modes"
---

Normal modes of a rectangular room: frequencies, kinds, count and density.

Below the Schroeder frequency a room is not a diffuse field but a handful of
discrete standing waves. For the idealised rigid-walled rectangular room of
dimensions $l_x \times l_y \times l_z$ the wave equation separates and
the eigen-
frequencies are known in closed form (Long, *Architectural Acoustics* 2nd ed.,
Equation (8.43); Kuttruff, *Room Acoustics* 6th ed., 3.1):

$$
f(n_x, n_y, n_z) = \frac{c_0}{2} \sqrt{ \left(\frac{n_x}{l_x}\right)^2 + \left(\frac{n_y}{l_y}\right)^2 + \left(\frac{n_z}{l_z}\right)^2 }
$$

with non-negative integer orders `nx, ny, nz` counting the nodal planes
perpendicular to each axis. Each mode is classified by how many of its orders
are non-zero:

* **axial** (one non-zero order): a wave bouncing between one pair of walls,
  the strongest and most audible family, e.g. the `1,0,0` fundamental;
* **tangential** (two): a wave grazing four walls, about 3 dB weaker;
* **oblique** (three): a wave involving all six walls, weaker still.

**How many modes.** Counting lattice points of the `k`-space grid inside the
positive octant of a sphere of radius $k = 2 \pi f / c_0$, with the
half- and
quarter-weight corrections for the points lying on the coordinate planes and
axes, gives the integrated mode count (Long Equation (8.45), after Morse 1948
and Pierce 1981):

$$
N(f) = \frac{4 \pi}{3} V \left(\frac{f}{c_0}\right)^3 + \frac{\pi}{4} S \left(\frac{f}{c_0}\right)^2 + \frac{L}{8} \frac{f}{c_0}
$$

with the room volume `V`, the total wall area `S` and the sum `L` of the
twelve edge lengths. Its derivative is the **modal density** in modes per hertz
(Long Equation (8.46)):

$$
\frac{dN}{df} = \frac{4 \pi V f^2}{c_0^3} + \frac{\pi}{2} \frac{S f}{c_0^2} + \frac{L}{8 c_0}
$$

The smooth count is an asymptotic estimate: at low frequency, where only a few
modes exist, the exact enumeration of [`room_modes`](/phonometry/reference/api/rooms/room-modes/#room_modes) is the honest answer,
while above the **Schroeder frequency**
([`phonometry.room.schroeder_frequency`](/phonometry/reference/api/rooms/steady-field/#schroeder_frequency)) the modes overlap so densely
that only the statistical description of
[`phonometry.room.steady_field`](/phonometry/reference/api/rooms/steady-field/) remains useful. Marking that frequency on
the mode ladder is the point of [`RoomModesResult.plot`](/phonometry/reference/api/rooms/room-modes/#roommodesresultplot).

This module describes the *eigenfrequencies* of an empty rectangular box. It is
not a prediction of the level at a receiver: for that, use the image-source
model ([`phonometry.room.image_source`](/phonometry/reference/api/rooms/image-source/)) or the steady-state statistical
field ([`phonometry.room.steady_field`](/phonometry/reference/api/rooms/steady-field/)).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## MODE_KINDS

*Constant* (`tuple`).

```python
MODE_KINDS = ('axial', 'tangential', 'oblique')
```

## room_modal_density

```python
room_modal_density(
    frequency: ArrayLike,
    dimensions: ArrayLike,
    *,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Modal density $dN/df$ in modes per hertz (Long Equation (8.46)).

$$
\frac{dN}{df} = \frac{4 \pi V f^2}{c_0^3} + \frac{\pi}{2} \frac{S f}{c_0^2} + \frac{L}{8 c_0}
$$

This is the
derivative of [`room_mode_count`](/phonometry/reference/api/rooms/room-modes/#room_mode_count). Its reciprocal is the mean spacing
between adjacent eigenfrequencies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequency `f`, Hz (scalar or array). |
| `dimensions` | Room dimensions `(lx, ly, lz)`, m. |
| `speed_of_sound` | Speed of sound `c0`, m/s. |

**Returns:** The modal density, modes/Hz.

## room_mode_count

```python
room_mode_count(
    frequency: ArrayLike,
    dimensions: ArrayLike,
    *,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Integrated number of modes below `frequency` (Long Equation (8.45)).

$$
N(f) = \frac{4 \pi}{3} V \left(\frac{f}{c_0}\right)^3 + \frac{\pi}{4} S \left(\frac{f}{c_0}\right)^2 + \frac{L}{8} \frac{f}{c_0}
$$

This is the
Morse/Pierce lattice count with its wall-plane and axis corrections. It is
a smooth asymptotic estimate, so it is only meaningful once several modes
fit below `f`; the exact enumeration is [`room_modes`](/phonometry/reference/api/rooms/room-modes/#room_modes).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Upper frequency `f`, Hz (scalar or array). |
| `dimensions` | Room dimensions `(lx, ly, lz)`, m. |
| `speed_of_sound` | Speed of sound `c0`, m/s. |

**Returns:** The (non-integer) mode count.

## room_mode_frequency

```python
room_mode_frequency(
    orders: ArrayLike,
    dimensions: ArrayLike,
    *,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Eigenfrequency of a rectangular-room mode (Long Equation (8.43)).

$$
f = \frac{c_0}{2} \sqrt{ \left(\frac{n_x}{l_x}\right)^2 + \left(\frac{n_y}{l_y}\right)^2 + \left(\frac{n_z}{l_z}\right)^2 }
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `orders` | Mode orders `(nx, ny, nz)`, non-negative integers; either a single triple or an `(N, 3)` array of triples. |
| `dimensions` | Room dimensions `(lx, ly, lz)`, m. |
| `speed_of_sound` | Speed of sound `c0`, m/s (default [`DEFAULT_SPEED_OF_SOUND`](/phonometry/reference/api/materials/road-absorption/#default_speed_of_sound)). |

**Returns:** The modal frequency in Hz; a float for a single triple, otherwise one frequency per row.

## room_modes

```python
room_modes(
    dimensions: ArrayLike,
    *,
    max_frequency: float = 200.0,
    speed_of_sound: float = 343.0,
    reverberation_time: float | None = None,
) -> RoomModesResult
```

Enumerate the normal modes of a rectangular room (Long Chapter 8).

Every order triple `(nx, ny, nz)` whose frequency
$(c_0/2) \sqrt{(n_x/l_x)^2 + (n_y/l_y)^2 + (n_z/l_z)^2}$ does not
exceed
`max_frequency` is listed, sorted by frequency and classified as axial,
tangential or oblique. The trivial `(0, 0, 0)` mode (the static pressure
solution) is excluded.

**Parameters**

| Name | Description |
| :--- | :--- |
| `dimensions` | Room dimensions `(lx, ly, lz)`, m. |
| `max_frequency` | Highest modal frequency to enumerate, Hz. |
| `speed_of_sound` | Speed of sound `c0`, m/s (default [`DEFAULT_SPEED_OF_SOUND`](/phonometry/reference/api/materials/road-absorption/#default_speed_of_sound)). |
| `reverberation_time` | Optional reverberation time `T`, s. When given, the Schroeder frequency $2000 \sqrt{T/V}$ is computed and carried in the result (and marked by [`RoomModesResult.plot`](/phonometry/reference/api/rooms/room-modes/#roommodesresultplot)). |

**Returns:** A [`RoomModesResult`](/phonometry/reference/api/rooms/room-modes/#roommodesresult).

## RoomModesResult

```python
RoomModesResult(
    orders: np.ndarray,
    frequencies: np.ndarray,
    kinds: np.ndarray,
    dimensions: tuple[float, float, float],
    speed_of_sound: float,
    max_frequency: float,
    schroeder_frequency: float | None = None,
)
```

The normal modes of a rectangular room up to a frequency limit.

**Attributes**

| Name | Description |
| :--- | :--- |
| `orders` | Mode orders, an `(N, 3)` integer array of `(nx, ny, nz)` sorted by frequency. |
| `frequencies` | Modal frequencies in ascending order, Hz. |
| `kinds` | Mode kind per row, one of [`MODE_KINDS`](/phonometry/reference/api/rooms/room-modes/#mode_kinds). |
| `dimensions` | Room dimensions `(lx, ly, lz)`, m. |
| `speed_of_sound` | Speed of sound `c0` used, m/s. |
| `max_frequency` | Frequency limit of the enumeration, Hz. |
| `schroeder_frequency` | Schroeder frequency, Hz, when a reverberation time was supplied; otherwise `None`. |

### RoomModesResult.count_by_kind()

```python
RoomModesResult.count_by_kind() -> dict[str, int]
```

Number of enumerated modes of each kind, keyed by [`MODE_KINDS`](/phonometry/reference/api/rooms/room-modes/#mode_kinds).

### RoomModesResult.density()

```python
RoomModesResult.density(frequency: ArrayLike) -> np.ndarray | float
```

Modal density at `frequency` for this room (Long Equation (8.46)).

### RoomModesResult.edge_length

*property*

Sum `L` of the twelve edge lengths, m.

### RoomModesResult.plot()

```python
RoomModesResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the mode ladder by kind and the modal density.

The upper panel stems every enumerated mode at its frequency, coloured
axial / tangential / oblique; the lower panel draws the smooth modal
density (Long Equation (8.46)). The Schroeder frequency, when known,
is marked on both. Requires matplotlib
(`pip install phonometry[plot]`).

### RoomModesResult.surface_area

*property*

Total area `S` of the six walls, m2.

### RoomModesResult.volume

*property*

Room volume `V`, m3.
