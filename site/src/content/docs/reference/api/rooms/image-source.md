---
title: "room.image_source"
description: "Synthetic room impulse response by the image-source method (rectangular room)."
sidebar:
  label: "image_source"
---

Synthetic room impulse response by the image-source method (rectangular room).

A rigid-walled (or absorbing) rectangular room -- a *shoebox* -- reflects a
point source in its six walls. Each reflection is equivalent to the free-field
sound of a mirror image of the source, so the room impulse response (RIR) is
the sum of the direct sound and one delayed, attenuated pulse per image
(Kuttruff, *Room Acoustics* 6th ed., 4.1, Equations (4.4)-(4.5); Vorlander,
*Auralization* 2nd ed., 11.4, the Allen-Berkley/Borish construction). This is
the deterministic complement of the statistical reverberation-time formulae of
[`phonometry.room.reverberation_prediction`](/phonometry/reference/api/rooms/reverberation-prediction/): where those give a single
decay rate, the image-source model gives the whole early reflection pattern and
the decay it implies.

**Image lattice.** For a room `[0, Lx] x [0, Ly] x [0, Lz]` with the source at
`(xs, ys, zs)`, mirroring a coordinate in a wall (Vorlander Equation (11.36),
`S_n = S - 2 d n`) turns the source into a regular lattice of images. Along
one axis the images sit at `2 n L +- x` for every integer `n` and the two
mirror parities; the parity index `p` and the lattice index `n` give the
number of reflections off the two walls of that axis as `|n - p|` (wall at 0)
and `|n|` (wall at `L`), so the total reflection order of an image is
`|2 n_x - p_x| + |2 n_y - p_y| + |2 n_z - p_z|` (Allen & Berkley, *J. Acoust.
Soc. Am.* 65 (1979) 943). The audible images up to order `i0` number
`(2/3)(2 i0^3 + 3 i0^2 + 4 i0)` in a shoebox (Kuttruff Equation (9.23)); the
temporal density of reflections grows as `dN/dt = 4 pi c^3 t^2 / V`
(Kuttruff Equation (4.6)).

**Per-image contribution.** Image `i` at distance `r_i` from the receiver
arrives at `t_i = r_i / c` (Vorlander Equation (11.38)) with amplitude

    A_i = [ product over walls of R_wall ^ (reflections there) ]
          * exp(-m r_i / 2) / (4 pi r_i),

the `1 / (4 pi r_i)` spherical spreading, the product of the wall
*pressure* reflection factors `R = sqrt(1 - alpha)` (Vorlander Equation
(11.39); `|R|^2 = 1 - alpha` in energy, Kuttruff 4.1) each raised to the
number of reflections that image made off that wall, and the air pressure
attenuation `exp(-m r_i / 2)` over the path (Kuttruff 4.1; `m` the
*intensity* attenuation constant, so intensity falls as `exp(-m r)`). The RIR
is the sum of unit impulses at `t_i` weighted by `A_i` (Kuttruff Equation
(4.5), `g(t) = sum_i A_i delta(t - t_i)`), assembled broadband from a single
absorption set or one curve per octave band from per-band coefficients.

The Schroeder backward integral of the synthetic RIR (see
[`phonometry.room.decay_curve`](/phonometry/reference/api/rooms/room-acoustics/#decay_curve)) reproduces the Eyring reverberation time
`T = -24 V ln 10 / (c S ln(1 - alpha_bar))` (Kuttruff Equation (5.23)) of the
same room to within a few percent, closing the loop between this deterministic
model and the statistical prediction. The construction is exact only for walls
whose reflection factor is real and angle-independent (Kuttruff 4.1: exact for
a specific wall impedance of +-1, a good approximation when the source stands a
few wavelengths from every wall); it captures specular reflections only, with
no diffraction or diffuse scattering.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## audible_image_count

```python
audible_image_count(max_order: int) -> int
```

Number of audible shoebox images up to reflection order `max_order`.

Kuttruff *Room Acoustics* 6th ed., Equation (9.23):
`(2/3)(2 i0^3 + 3 i0^2 + 4 i0)`. Every image of a rectangular room is
audible (no visibility test needed), so this is exactly the number of
impulses [`image_source_rir`](/phonometry/reference/api/rooms/image-source/#image_source_rir) sums at that order.

**Parameters**

| Name | Description |
| :--- | :--- |
| `max_order` | Reflection-order cut-off `i0` (non-negative). |

**Returns:** The audible image count.

## image_source_rir

```python
image_source_rir(
    dimensions: tuple[float, float, float],
    source: ArrayLike,
    receiver: ArrayLike,
    absorption: ArrayLike,
    *,
    fs: int,
    max_order: int = 20,
    speed_of_sound: float = 343.0,
    air_attenuation: ArrayLike = 0.0,
    duration: float | None = None,
    frequencies: ArrayLike | None = None,
) -> ImageSourceResult
```

Synthetic room impulse response of a shoebox by the image-source method.

Builds every image of the source up to reflection order `max_order`
(Vorlander Equation (11.36); Allen & Berkley 1979), then assembles the RIR
as the sum of the direct sound and one attenuated, delayed unit impulse per
image (Kuttruff Equations (4.4)-(4.5)). Each image at distance `r` arrives
at `r / c` (Vorlander Equation (11.38)) with amplitude
`[prod R_wall^n_wall] exp(-m r / 2) / (4 pi r)`: the `1 / (4 pi r)`
spherical spreading, the product of the wall pressure reflection factors
`R = sqrt(1 - alpha)` (Vorlander Equation (11.39)) over the reflections
the image made, and the air pressure attenuation `exp(-m r / 2)`.

With scalar or per-wall `absorption` (and no `frequencies`) the result
is a broadband RIR; a per-band `absorption` (or a given `frequencies`)
produces one RIR per band. The Schroeder decay of the synthetic RIR
reproduces the Eyring reverberation time of the room (Kuttruff Equation
(5.23)); feed `ir` straight to [`phonometry.room.decay_curve`](/phonometry/reference/api/rooms/room-acoustics/#decay_curve) or
[`phonometry.room.room_parameters`](/phonometry/reference/api/rooms/room-acoustics/#room_parameters).

**Parameters**

| Name | Description |
| :--- | :--- |
| `dimensions` | Room lengths `(Lx, Ly, Lz)`, m. |
| `source` | Source position `(x, y, z)`, m, strictly inside the room. |
| `receiver` | Receiver position `(x, y, z)`, m, strictly inside. |
| `absorption` | Wall absorption coefficient(s) in `[0, 1]`: a scalar (uniform), a length-6 per-wall vector (order `WALL_ORDER`), a per-band vector, or a `(6, n_bands)` per-wall per-band array. A length-6 vector is read as the six per-wall values *unless* `frequencies` declares six bands, in which case it is a per-band curve (uniform across walls); use the `(6, n_bands)` form for six per-wall values that also vary with frequency. |
| `fs` | Sample rate, Hz. |
| `max_order` | Reflection-order cut-off (total wall reflections). The shoebox has `(2/3)(2 i0^3 + 3 i0^2 + 4 i0)` audible images up to order `i0` (Kuttruff Equation (9.23)). Default 20. |
| `speed_of_sound` | Speed of sound `c`, m/s (default [`DEFAULT_SPEED_OF_SOUND`](/phonometry/reference/api/materials/road-absorption/#default_speed_of_sound)). |
| `air_attenuation` | Air *intensity* attenuation constant `m`, in neper per metre (scalar or per-band); the pressure amplitude of each path is scaled by `exp(-m r / 2)` (Kuttruff 4.1). Default 0 (air absorption neglected). Obtain a physical `m` from [`phonometry.air_absorption.air_attenuation_m`](/phonometry/reference/api/environment/air-absorption/#air_attenuation_m). |
| `duration` | RIR length, s; default the latest image arrival rounded up to the next sample. |
| `frequencies` | Optional band centre frequencies, Hz, labelling a per-band result. When given, its length must match the band count of `absorption` (or broadcast against it). |

**Returns:** An [`ImageSourceResult`](/phonometry/reference/api/rooms/image-source/#imagesourceresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive dimension/sample-rate, a source/receiver outside the room, an absorption outside `[0, 1]` or of an unsupported shape, a negative `air_attenuation`, or a `frequencies` length that does not match the band count. |

## ImageSourceResult

```python
ImageSourceResult(
    ir: np.ndarray,
    fs: int,
    frequencies: np.ndarray | None,
    times: np.ndarray,
    distances: np.ndarray,
    orders: np.ndarray,
    amplitudes: np.ndarray,
    image_positions: np.ndarray,
    dimensions: tuple[float, float, float],
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
    max_order: int,
    speed_of_sound: float,
)
```

Synthetic room impulse response by the image-source method.

`ir` is the sampled RIR: a 1D array for a broadband model, or a
`(n_bands, n_samples)` array with one decay per octave band for per-band
absorption. Each image contributes a unit impulse at the nearest sample to
its exact arrival time; the exact, sub-sample reflection table is kept
separately in `times` / `distances` / `orders` / `amplitudes` /
`image_positions` so the geometry stays exact regardless of `fs`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `ir` | Sampled impulse response (Kuttruff Equation (4.5)); shape `(n_samples,)` (broadband) or `(n_bands, n_samples)` (per band). |
| `fs` | Sample rate, Hz. |
| `frequencies` | Band centre frequencies, Hz, or `None` for a broadband model. |
| `times` | Exact arrival time `t_i = r_i / c` of every image, s (sorted ascending). |
| `distances` | Image-to-receiver distance `r_i`, m (aligned with `times`). |
| `orders` | Total reflection order of every image (aligned with `times`). |
| `amplitudes` | Exact per-image amplitude `A_i`; shape `(n_images,)` (broadband) or `(n_bands, n_images)` (per band). |
| `image_positions` | Image-source coordinates `(x, y, z)`, m, shape `(n_images, 3)` (aligned with `times`). |
| `dimensions` | Room lengths `(Lx, Ly, Lz)`, m. |
| `source` | Source position `(x, y, z)`, m. |
| `receiver` | Receiver position `(x, y, z)`, m. |
| `max_order` | Reflection-order cut-off used. |
| `speed_of_sound` | Speed of sound `c`, m/s. |

### ImageSourceResult.direct_time

*property*

Arrival time of the direct sound (order 0), s.

### ImageSourceResult.plot()

```python
ImageSourceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the reflectogram: reflection level in dB against arrival time.

Stems the per-image amplitudes (in dB re the direct sound), coloured by
reflection order, with the `1 / r` free-field envelope overlaid.
Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## reflection_density

```python
reflection_density(
    time: ArrayLike,
    volume: float,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Temporal density of reflections `dN/dt = 4 pi c^3 t^2 / V`.

Kuttruff *Room Acoustics* 6th ed., Equation (4.6): the number of image
sources per unit time whose spheres of radius `c t` sweep the receiver.
Independent of room shape. Useful to judge the reflection-order cut-off of
[`image_source_rir`](/phonometry/reference/api/rooms/image-source/#image_source_rir) (the model is complete only while its images keep
up with this density).

**Parameters**

| Name | Description |
| :--- | :--- |
| `time` | Time after the direct sound, s (scalar or array). |
| `volume` | Room volume `V`, m3. |
| `speed_of_sound` | Speed of sound `c`, m/s. |

**Returns:** `dN/dt` in reflections per second.
