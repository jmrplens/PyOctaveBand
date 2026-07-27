← [Documentation index](README.md)

# Image sources and the steady-state room field (Kuttruff / Vorländer / Bies)

Where [reverberation-time prediction](reverberation-prediction.md) gives a
single statistical decay rate and [room-acoustics measurement](room-acoustics.md)
analyses a *measured* impulse response, this page covers the two classical
*predictions* of the sound field in a rectangular room, both in
`phonometry.room`:

- the **image-source room impulse response** (`image_source_rir`): the
  deterministic early reflection pattern built by mirroring a point source in
  the six walls of a shoebox (Kuttruff *Room Acoustics* 4.1; Vorländer
  *Auralization* 11.4); and
- the **steady-state room field** (`steady_state_field` and its parts): the
  statistical direct-plus-reverberant sound pressure level a source of known
  power produces, with the room constant, critical distance and Schroeder
  frequency (Bies *Engineering Noise Control* 6.4; Kuttruff 5.6).

Together they bridge the sound power of `phonometry.emission` and the reverberation
prediction of `phonometry.room`: one gives the full RIR, the other the level
the same room settles to.

## 1. Image-source room impulse response

A rigid or absorbing rectangular room reflects a point source in its walls;
each reflection is exactly the free-field sound of a **mirror image** of the
source. Mirroring a coordinate in a wall (Vorländer Equation (11.36),
`S_n = S − 2 d n`) turns the source into a regular lattice of images, and the
room impulse response is the sum of the direct sound and one delayed,
attenuated impulse per image (Kuttruff Equations (4.4)–(4.5),
`g(t) = Σ_n A_n δ(t − t_n)`).

Image `i` at distance `r_i` from the receiver arrives at `t_i = r_i / c`
(Vorländer Equation (11.38)) with amplitude

```text
A_i = [ Π_walls R_wall^(reflections there) ] · exp(−m r_i / 2) / (4 π r_i)
```

combining the `1 / (4 π r_i)` spherical spreading, the product of the wall
**pressure reflection factors** `R = √(1 − α)` (Vorländer Equation (11.39);
`|R|² = 1 − α` in energy, Kuttruff 4.1) each raised to the number of
reflections that image made off that wall, and the air pressure attenuation
`exp(−m r_i / 2)` over the path (Kuttruff 4.1; `m` the *intensity* attenuation
constant, so intensity falls as `exp(−m r)`).

Along one axis the reflection count off the two walls of an image at lattice
index `n` and mirror parity `p` is `|n − p|` (wall at 0) and `|n|` (wall at
`L`), so the total reflection order is
`|2 n_x − p_x| + |2 n_y − p_y| + |2 n_z − p_z|` (Allen & Berkley 1979). A
shoebox has exactly `(2/3)(2 i₀³ + 3 i₀² + 4 i₀)` audible images up to order
`i₀` (Kuttruff Equation (9.23), e.g. 1560 at order 10), and the temporal
density of reflections grows as `dN/dt = 4 π c³ t² / V` (Kuttruff Equation
(4.6)).

The construction in one plan: every wall reflection is the straight-line sound
of an image in a mirror room, so the geometry alone fixes each arrival. In the
7 × 5 m room below the direct sound lands at 10.7 ms and the four first-order
images follow between 17.3 and 21.6 ms.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_room_image_sources_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_room_image_sources.svg" alt="Plan of a 7 by 5 m room in the centre of its three by three grid of dashed mirror rooms: the source at (2, 1.6) and the receiver at (5.2, 3.4) joined by the 10.7 ms direct sound, filled first-order images in the four adjacent rooms labelled with their arrival times of 17.3, 20.5 and 21.6 ms, outlined second-order images in the corner rooms at 24.6 and 25.6 ms, and the first far-wall reflection drawn twice, as the real bent path off the wall and as the straight dashed line from the image to the receiver" width="92%"></picture>

```python
import numpy as np
from phonometry import room

# A 7 x 5 x 3 m room, source and receiver placed off-centre.
res = room.image_source_rir(
    dimensions=(7.0, 5.0, 3.0),
    source=(2.0, 1.6, 1.5),
    receiver=(5.2, 3.4, 1.7),
    absorption=0.12,            # uniform wall absorption
    fs=48000,
    max_order=12,
)

print(res.ir.shape)                          # (n_samples,) broadband RIR
print(round(res.direct_time * 1000, 2))      # direct-sound arrival, ms
print(res.times.size, room.audible_image_count(12) + 1)  # images + direct source
res.plot()   # the reflectogram of the figure below

# Feed the synthetic RIR straight into the ISO 3382 decay analysis.
params = room.room_parameters(res.ir, res.fs, limits=None)
print(bool(params.t30_valid[0]))             # True: the decay window is usable
# T30 rises toward the Eyring estimate as max_order grows (see below); at a low
# order the specular tail is truncated, so treat this as an order-limited value.
print(round(float(params.t30[0]), 2))        # reverberation time, s
```

`image_source_rir` returns an `ImageSourceResult`. Its `ir` is the sampled
RIR (a 1D array broadband, or one row per octave band for per-band
absorption); the **exact** sub-sample reflection table is kept separately in
`times`, `distances`, `orders`, `amplitudes` and `image_positions`, so the
geometry stays exact regardless of the sample rate. `.plot()` draws the
reflectogram (reflection level in dB versus arrival time, coloured by order).

Pass per-band coefficients (a `(6, n_bands)` per-wall array, a per-band vector,
or a `frequencies` list) to synthesise one decay per octave band; pass a
length-6 vector to set each wall separately (order
`x0, xL, y0, yL, z0, zL`); and pass `air_attenuation` (the intensity
coefficient `m` from `air_attenuation_m`) to add the `exp(−m r / 2)` air loss.

```python
import numpy as np
from phonometry import room

freqs = [250.0, 500.0, 1000.0, 2000.0]
alpha = np.array([[0.10, 0.15, 0.25, 0.40]] * 6)     # (6 walls, 4 bands)
res = room.image_source_rir((7.0, 5.0, 3.0), (2.0, 1.6, 1.5), (5.2, 3.4, 1.7),
                            alpha, fs=48000, max_order=12, frequencies=freqs)
print(res.ir.shape)                                  # (4 bands, n_samples)
print(np.round(np.sum(res.ir ** 2, axis=1), 4))      # more absorption -> less energy
```

![Image-source reflectogram: the synthetic room impulse response of a 7x5x3 m room as a cloud of reflections coloured by reflection order, decaying under the 1/r spreading envelope with the direct sound marked at order 0](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/image_source_reflectogram.webp)

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import room

res = room.image_source_rir((7.0, 5.0, 3.0), (2.0, 1.6, 1.5),
                            (5.2, 3.4, 1.7), 0.12, fs=48000, max_order=10)

# One line: the reflectogram (level in dB re direct vs arrival time, by order).
res.plot()
plt.show()

# By hand: scatter the reflection amplitudes coloured by order.
t_ms = np.asarray(res.times) * 1e3
amp = np.asarray(res.amplitudes)
level = 20 * np.log10(np.abs(amp) / np.max(np.abs(amp)))
order = np.asarray(res.orders)
fig, ax = plt.subplots()
sc = ax.scatter(t_ms[order > 0], level[order > 0], c=order[order > 0],
                cmap="viridis", s=18)
ax.stem([t_ms[order == 0][0]], [0.0])         # direct sound
fig.colorbar(sc, label="Reflection order")
ax.set_xlabel("Arrival time [ms]"); ax.set_ylabel("Level re direct [dB]")
ax.set_xlim(0, 120); ax.set_ylim(-60, 5)
plt.show()
```

</details>

Every dot of the reflectogram is a mirror image at a definite place, and
`.plot_geometry()` shows where: the plan view below draws the room, the
source, the receiver and every image up to third order on the mirror-room
grid, coloured by reflection order.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/image_source_plan_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/image_source_plan.svg" alt="Plan view of the image-source lattice of a 7 by 5 metre room outlined in the centre, with the source as a red star at (2, 1.6), the receiver as a blue triangle at (5.2, 3.4), and the image sources at the plane of the source coloured by reflection order, first order orange, second order green and third order purple, spreading in a regular grid of mirror rooms out to about 26 metres" width="88%"></picture>

*The lattice the reflectogram comes from, to scale: each image sits in a
mirror room at the height plane of the source, and its distance to the
receiver alone fixes the arrival time and the `1/(4 pi r)` spreading of that
reflection.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import room

res = room.image_source_rir((7.0, 5.0, 3.0), (2.0, 1.6, 1.5),
                            (5.2, 3.4, 1.7), 0.3, fs=16000, max_order=3)

# One line: the image lattice in plan, coloured by reflection order.
res.plot_geometry()
plt.show()
```

</details>

**Reproducing the statistical decay.** The fitted initial decay slope of the
reverberant energy density of the synthetic RIR recovers the **Eyring**
reverberation time `T = −24 V ln 10 / (c S ln(1 − ᾱ))` (Kuttruff Equation
(5.23)): the mean reflection rate `c S / 4 V` equals
`(c/2)(1/Lx + 1/Ly + 1/Lz)`, so the specular field's initial decay rate is the
one that defines that `T`. The match is exact only in the near-cubic
limit; an elongated room sustains energy along its long axis, so its pure
*specular* decay runs slower than Eyring's diffuse-field estimate (the regime
the [Fitzroy and Arau-Puchades models](reverberation-prediction.md) correct).
The model captures specular reflections only, with no diffraction or diffuse
scattering, and is exact only for real, angle-independent wall reflection
factors (Kuttruff 4.1).

## 2. Steady-state room field

When a source of constant sound power runs in a room, the sound pressure level
settles to the sum of a **direct field** that falls with distance and a
**reverberant field** that (to the diffuse approximation) is the same
everywhere. The **room constant**

```text
R = S ᾱ / (1 − ᾱ)                       (Bies Equation (6.44))
```

with total boundary area `S` and mean Sabine absorption `ᾱ` measures how much
reverberant field a given power builds up. The **steady-state level** is

```text
Lp = Lw + 10 log10( Q / (4 π r²) + 4 / R )   [ + 10 log10(ρc / 400) ]
                                               (Bies Equation (6.43))
```

with the source directivity factor `Q` (1 omnidirectional, 2 on a hard floor,
4 in an edge, 8 in a corner). The optional `10 log10(ρc / 400)` term
(about +0.14 dB at 20 °C) corrects for a characteristic impedance differing
from 400 Pa·s/m and is omitted by default. The **critical distance**

```text
rc = √( Q R / (16 π) )
```

is where the two fields are equal (the crossover of Equation (6.43)); closer
than `rc` the direct field dominates, farther the reverberant field does.
Kuttruff's reverberation distance (Equation (5.44), `rc = √(A / 16 π)` for
`Q = 1`) uses the Sabine absorption area `A = S ᾱ` instead of the room
constant `R = A / (1 − ᾱ)`; the two coincide for a small `ᾱ` and this module
uses `R`, so `rc` is exactly the crossover of its own `steady_state_spl`.

```python
from phonometry import room

# A 90 dB source in a 100 m^2 room with 20 % mean absorption.
field = room.steady_state_field(
    sound_power_level=90.0,
    surface_area=100.0,
    mean_absorption=0.2,
)
print(round(field.room_constant, 1))          # 25.0 m^2
print(round(field.critical_distance, 2))       # 0.71 m
field.plot()                                    # direct / reverberant / total vs distance

# The building blocks are exposed individually, too:
print(round(float(room.room_constant(100.0, 0.2)), 1))            # 25.0
print(round(float(room.critical_distance(25.0)), 3))              # 0.705
print(round(float(room.steady_state_spl(90.0, 5.0, 25.0)), 2))    # far-field level
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/steady_state_field_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/steady_state_field.svg" alt="Steady-state room field of a 90 dB source in a workshop with a room constant of 62 square metres: the total sound pressure level follows the 6 dB-per-doubling direct field close to the source, crosses the constant reverberant plateau at the 1.11 m critical distance and flattens onto it beyond" width="80%"></picture>

*A 90 dB re 1 pW source in a 12 x 8 x 4 m workshop with a mean absorption of
0.15: within $r_c = 1.11$ m moving away drops the level 6 dB per doubling;
beyond it the reverberant plateau takes over and only absorption, not
distance, lowers the level (Bies 5e, §6.4).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import room

field = room.steady_state_field(
    sound_power_level=90.0,     # Lw, dB re 1 pW
    surface_area=352.0,          # a 12 x 8 x 4 m workshop
    mean_absorption=0.15,
)

# One line: direct, reverberant and total levels with rc marked.
field.plot()
plt.show()

# By hand, from the result's fields:
fig, ax = plt.subplots()
ax.semilogx(field.distances, field.direct, "--", label="Direct field")
ax.semilogx(field.distances, field.reverberant, ":", label="Reverberant field")
ax.semilogx(field.distances, field.total, label="Total")
ax.axvline(field.critical_distance, ls="-.",
           label=f"rc = {field.critical_distance:.2f} m")
ax.set_xlabel("Distance from source [m]")
ax.set_ylabel("Sound pressure level [dB]")
ax.legend()
plt.show()
```

</details>

The **Schroeder frequency**

```text
f_s = 2000 √(T / V)                       (Kuttruff Equation (3.44))
```

(`V` in m³, `T` in s) roughly marks the modal-to-diffuse transition, a
heuristic crossover rather than a sharp cutoff: well below it discrete room
modes dominate and the diffuse assumptions of `R` and `rc` grow unreliable,
well above it the modes overlap and the statistical field of this section
holds. In borderline rooms it is worth checking band by band.

```python
from phonometry import room
print(round(float(room.schroeder_frequency(1.0, 200.0)), 0))   # 141 Hz (V=200, T=1)
```

## Validation

The implementations are checked against the closed forms and the source texts'
own numeric anchors (see [CONFORMANCE.md](CONFORMANCE.md)):

- the direct-sound amplitude `1/(4 π r)` and delay `r / c` (exact geometry),
  the audible image count (Kuttruff Equation (9.23)) and the reflection
  density (Equation (4.6));
- the Eyring reverberation time recovered from the decay of the synthetic RIR
  in the near-cubic limit (documented ≈ 10 % tolerance), and an independent
  2D FDTD (`phonometry.simulation`) reproducing the rigid-wall echo delay and
  the uniform-damping `T60`;
- the room constant, the critical distance as the exact direct/reverberant
  crossover, the Schroeder frequency (Kuttruff's classroom example, `V = 200`,
  `T = 1` → 141 Hz) and the steady-state level (Bies Equation (6.43)).

## References

- Kuttruff, H. (2016). *Room acoustics* (6th ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  Section 4.1 (image sources, Equations (4.4)–(4.6)), Section 5.5–5.6
  (Eyring reverberation, reverberation distance, Equations (5.23), (5.44)),
  Section 3.6 (Schroeder frequency, Equation (3.44)) and Section 9.8 (audible
  image count, Equation (9.23)).
- Vorländer, M. (2020). *Auralization: Fundamentals of acoustics, modelling,
  simulation, algorithms and acoustic virtual reality* (2nd ed.). Springer.
  [doi:10.1007/978-3-030-51202-6](https://doi.org/10.1007/978-3-030-51202-6).
  Chapter 11 (the image-source / mirror-source model, Equations (11.36),
  (11.38), (11.39)).
- Allen, J. B., & Berkley, D. A. (1979). Image method for efficiently
  simulating small-room acoustics. *The Journal of the Acoustical Society of
  America*, 65(4), 943–950.
  [doi:10.1121/1.382599](https://doi.org/10.1121/1.382599).
  The reflection-count decomposition of the rectangular-room image lattice.
- Bies, D. A., Hansen, C. H., & Howard, C. Q. (2017). *Engineering noise
  control* (5th ed.). CRC Press.
  [doi:10.1201/9781351228152](https://doi.org/10.1201/9781351228152).
  Section 6.4 (steady-state response and the room constant, Equations
  (6.41)–(6.44)).
