← [Documentation index](../README.md)

# Rotorcraft noise: the hemisphere method (ECAC Doc 32 / NORAH2)

Helicopter noise is strongly directive, so the ECAC Doc 32 method describes the
source with a **noise hemisphere**: one-third-octave-band sound pressure levels
on a spherical grid of azimuth $\varphi$ and polar angle $\theta$, measured at
a fixed **60 m** reference distance under ICAO reference atmospheric
conditions. Placing that source at a receiver adds the propagation adjustment
$\Delta L_p = \Delta L_s + \Delta L_a + \Delta L_g$ ($+\ \Delta L_d$).

This page covers the source and propagation primitives, the flight-condition
interpolation across a hemisphere database, the hover/idle source derivation
of Table 3, the flight-path kinematics, the single-event integration to
`SEL`/`LASmax`/`EPNL` and ground-grid contours, and the terrain
mean-ground-plane and screening machinery.

The EPNL this page computes is the same quantity certification asks for: ICAO
Annex 16 Chapter 8 flies the helicopter level at 150 m over a centre
microphone, with two more 150 m to each side.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_rotorcraft_certification_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_rotorcraft_certification.svg" alt="Helicopter overflight noise certification of ICAO Annex 16 Chapter 8: a side view of a helicopter in level flight at 150 m, 492 ft, directly above a centre microphone on the ground, with the reference speed rule of 0.9 VH, and a plan-view inset where the flight track crosses a line of three microphones, one on the track and two 150 m to each side; notes state at least six overflights split equally between headwind and tailwind, EPNL in EPNdB at the three points, microphones 1.2 m above ground and the 45 degree elevation with about 212 m slant range to the sideline pair at the overhead moment" width="92%"></picture>

## 1. The noise hemisphere

The library implements the method, not the data: **no hemisphere database
ships with phonometry, and there is no NORAH file reader**, so the arrays
below are assembled by the reader — from a flight-test campaign that
de-propagates each emission direction back to the 60 m reference sphere, or
from the NORAH2 reference database, which covers eleven rotorcraft types.

A `RotorcraftHemisphere` holds the band levels on the azimuth/polar grid (with
missing bins marked `NaN`). `hemisphere_source_level` reads the source level at an
arbitrary emission direction: the grid is first gap-filled by nearest-bin
constant-value extrapolation (Eq. 14/15, equally-near bins energy-averaged;
computed once and cached), then the lookup is bilinear in the energy domain over
the four neighbouring bins (Doc 32 Eq. 13), so partially-measured cells stay
continuous with their measured corners.

```python
import numpy as np
from phonometry import aircraft

# A hemisphere: band levels of shape (azimuth, polar, frequency) at 60 m.
h = aircraft.RotorcraftHemisphere(frequencies=freqs, azimuth=phi, polar=theta, levels=levels)
h.plot()                                   # fore-aft directivity (needs matplotlib)
lv = aircraft.hemisphere_source_level(h, 0.0, 90.0)   # level per band abeam-below
```

## 2. Propagation adjustments

The hemisphere level at 60 m is carried to the receiver by three adjustments
(Doc 32 §A.4):

- **`spherical_spreading_adjustment(r)`**: $\Delta L_s = -20\log_{10}(r/60)$
  (Eq. 24).
- **`atmospheric_adjustment(freqs, r)`**: $\Delta L_a = -\alpha(f)\,(r - 60)$
  with the
  ISO 9613-1 pure-tone coefficient at the exact band centre (Eq. 26/27), reusing
  the library's `air_attenuation`; this is what the NORAH2 reference
  implementation computes. The guidance's alternative SAE (Rickley) band mapping,
  tabulated in its Table 4, coincides below 3.15 kHz and deviates by up to
  2.2 dB/km at 8-10 kHz.
- **`ground_effect_adjustment(freqs, hs, hr, dp, flow_resistivity=…)`**: the
  interference between the direct and ground-reflected rays over an impedance
  plane (Chien-Soroka, Eq. 28-35), with the Delany-Bazley one-parameter impedance
  and the CNOSSOS flow-resistivity classes `"A"`-`"H"`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_ground_effect_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_ground_effect.svg" alt="Rotorcraft ground-effect adjustment versus one-third-octave frequency for hard (asphalt) and soft (grass) ground: low-frequency reinforcement, a deep destructive interference dip, and the incoherent +3 dB high-frequency region, the dip deeper over hard ground" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import aircraft

freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
hs, hr, dp = 150.0, 1.5, 500.0                          # overflight geometry
grass = aircraft.ground_effect_adjustment(freqs, hs, hr, dp, flow_resistivity="D")
asphalt = aircraft.ground_effect_adjustment(freqs, hs, hr, dp, flow_resistivity="G")

fig, ax = plt.subplots()
ax.axhline(0.0, color="0.5", linewidth=1.0)
ax.semilogx(freqs, asphalt, marker="o", markersize=3,
            label="Hard (asphalt/concrete, class G)")
ax.semilogx(freqs, grass, marker="s", markersize=3,
            label="Soft (grass/pasture, class D)")
ax.set(xlabel="One-third-octave-band centre frequency [Hz]",
       ylabel="Ground-effect adjustment ΔLg [dB]",
       title="Rotorcraft ground effect (ECAC Doc 32, Chien-Soroka)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
```

</details>

```python
import numpy as np
from phonometry import aircraft

freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
r = 500.0
received = (lv
            + aircraft.spherical_spreading_adjustment(r)
            + aircraft.atmospheric_adjustment(freqs, r)
            + aircraft.ground_effect_adjustment(freqs, 150.0, 1.5, 500.0, flow_resistivity="D"))
```

The standard database is recorded at 60 m, the default. If a hemisphere uses a
different polar distance (`h.distance`, e.g. 70 m hover rings), pass it to both
distance-dependent adjustments:
`spherical_spreading_adjustment(r, reference_distance=h.distance)` and
`atmospheric_adjustment(freqs, r, reference_distance=h.distance)`. The three
adjustments assume straight rays: refraction by atmospheric gradients is
outside Doc 32 itself and is not modelled.

## 3. Flight conditions: interpolating between hemispheres

A database records one hemisphere per flight condition (airspeed $V$, path
angle $\gamma$): approaches at several descent angles, level flyovers at
several speeds, take-off climbs. Real flight conditions rarely coincide with a
measured one, so the NORAH2 guidance interpolates (Eq. 3-10): both axes are
normalised by their database spans (with the empirical scaling factor
$F_\text{fc} = 2$ on the
path angle), a Delaunay triangulation covers the normalised conditions, and a
query inside the convex hull blends the three hemispheres of its enveloping
triangle with inverse-distance weights in the energy domain. Outside the hull
the nearest condition is adopted unblended, which is also the behaviour ECAC
Doc 32, 1st ed. prescribes for its whole envelope (it defines no interpolation
yet).

`flight_condition_weights` returns the `(index, weight)` pairs;
`interpolated_source_level` applies them to the hemisphere lookups:

```python
import numpy as np
from phonometry import aircraft

# One hemisphere per measured flight condition (speeds in kt, angles in deg).
speeds = [50.0, 70.0, 60.0]
angles = [0.0, 0.0, 10.0]
weights = aircraft.flight_condition_weights(speeds, angles, 60.0, 2.5)
lv = aircraft.interpolated_source_level(
    [h_50_level, h_70_level, h_60_climb], speeds, angles,
    60.0, 2.5, 0.0, 90.0)               # blended level per band, abeam-below
```

The airspeed, not the ground speed, selects the hemisphere. The weights are
unit-invariant (knots or m/s, as long as the query matches the database), and a
database lookup triangulation can be passed as `triangles` (the NORAH database
ships one per type; its shipped tables triangulate the raw $(V, \gamma)$
plane, so passing them reproduces the reference implementation bin for bin).
Types whose rotor configuration is mirrored with respect to their class
reference substitute `h.mirrored()` (Eq. 2, $\varphi \to -\varphi$), and
certification-level offsets enter as `level_offset`.

## 4. Hover, idle and taxi (guidance §A.3.5, Table 3)

A hovering or idling helicopter is measured differently from a flyover: on a
**ring of ground microphones** around the stationary aircraft (the CAEP
in-ground hover practice the guidance points at), one band spectrum per ring
bearing (0° at the nose, positive to starboard), reduced to the ring's polar
distance — commonly 70 m rather than the flyover database's 60 m. The
guidance defines four conditions — in-ground hover (HIGE), out-of-ground
hover (HOGE), reduced-rpm idle and full-rpm idle — and three approaches in
descending priority: measure all four rings (Approach 1), measure the HIGE
ring and the other three in the 0° direction only (Approach 2), or measure
the HIGE ring alone and apply fixed offsets (Approach 3).

`hover_ring_hemisphere` extends the ring to a full `RotorcraftHemisphere`
"assuming constant directivity in φ": each bin $(\varphi, \theta)$ reads the
ring at the bearing $\pm\theta$ of its own half — port bins from negative
bearings, starboard bins from positive ones, and the $\varphi = 0$ column
under the aircraft takes the energy mean of the $\pm\theta$ ring values —
with periodic interpolation in the energy domain. The NORAH2 reference
implementation reads the same ring by the horizontal bearing of the emission
direction instead (constant directivity in elevation, `mapping="bearing"`);
the two readings coincide on the hemisphere rim and part by several dB at
steep emission angles.

`hover_derived_hemisphere` derives HOGE and the idles from the HIGE
hemisphere by a uniform level offset: a measured 0°-direction difference
(Approach 2, passed as `offset_db`), or the published Approach 3 constants —
+12 dB for HOGE, −12 dB for reduced-rpm idle and −2.5 dB for full-rpm idle,
all from the HIGE disk. The guidance warns that those constants come from
measurements with inverted microphones on ground plates and may not hold for
other setups, and the reference database actually ships +8/−10/−2 dB (the
`Corr_dB` corrections of its per-type interpolation files): pass them as
`offset_db` to reproduce the prototype. The Approach 3 prose derives
full-rpm idle from HOGE where its own Table 3 derives it from HIGE, 12 dB
apart; the table, backed by the database, is what is implemented (see the
[errata registry](../ERRATA.md)).

```python
import numpy as np
from phonometry import aircraft

# A hover ring with a real campaign's shape: one 31-band third-octave
# spectrum per 30° of bearing, reduced to the 70 m polar distance.
freqs = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)    # 10 Hz-10 kHz thirds
bearings = np.arange(-180.0, 151.0, 30.0)               # the ring closes at ±180°
ring_levels = (86.0 - 10.0 * np.log10(freqs / 160.0) ** 2
               + 3.5 * np.cos(np.radians(bearings))[:, None])
hige = aircraft.hover_ring_hemisphere(freqs, bearings, ring_levels, distance=70.0)
hoge = aircraft.hover_derived_hemisphere(hige, "out_of_ground_hover")   # +12 dB
full_rpm_idle = aircraft.hover_derived_hemisphere(hige, "full_rpm_idle")  # -2.5 dB
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_hover_ring_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_hover_ring.svg" alt="Two panels of the hover source derivation. On the left, the in-ground-hover ring at the 315 Hz band: band level at 70 m against ring bearing from minus 180 to 180 degrees, peaking at the nose and falling to the tail, the starboard half a few decibels above the port half. On the right, the fore-aft sections of the four derived sources against polar angle: the HIGE hemisphere built from the ring, and the out-of-ground hover, full-rpm idle and reduced-rpm idle curves at plus 12, minus 2.5 and minus 12 decibels from it, with a note that constant directivity in phi means each polar angle reads the ring at plus or minus that bearing" width="92%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import aircraft

freqs = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)   # 10 Hz-10 kHz thirds
bearings = np.arange(-180.0, 151.0, 30.0)
spectrum = 86.0 - 10.0 * np.log10(freqs / 160.0) ** 2
directivity = (3.5 * np.cos(np.radians(bearings))
               + 3.0 * np.exp(-((bearings - 120.0) ** 2) / (2.0 * 40.0**2)))
ring = spectrum[None, :] + directivity[:, None]
hige = aircraft.hover_ring_hemisphere(freqs, bearings, ring, distance=70.0)

k = int(np.argmin(np.abs(freqs - 315.0)))
theta = np.linspace(0.0, 180.0, 361)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
ax.plot(np.append(bearings, 180.0), np.append(ring[:, k], ring[0, k]),
        marker="o")
ax.set(xlabel="Ring bearing [°]  (0° nose, +90° starboard)",
       ylabel="Band level at 70 m [dB]")
for condition, label in ((None, "HIGE (from the ring)"),
                         ("out_of_ground_hover", "HOGE (+12 dB)"),
                         ("full_rpm_idle", "Full-rpm idle (-2.5 dB)"),
                         ("reduced_rpm_idle", "Reduced-rpm idle (-12 dB)")):
    h = hige if condition is None else aircraft.hover_derived_hemisphere(hige, condition)
    ax2.plot(theta, [aircraft.hemisphere_source_level(h, 0.0, t)[k] for t in theta],
             label=label)
ax2.set(xlabel="Polar angle θ [°]  (0° forward → 180° rearward)",
        ylabel="Source level at 70 m [dB]")
ax2.legend()
plt.show()
```

</details>

**Taxi** is a selection between two of these sources, not a new one: a
helicopter without wheels taxis hovering in ground effect (the HIGE
hemisphere), and a wheeled one ground-taxis on its wheels with the rotor at
idle (the full-rpm-idle hemisphere). The guidance's sentence pairs them the
other way around ("… in-ground hover and full-rpm idle respectively"); the
pairing here follows the physics of the two operations (see the errata
registry).

```python
wheeled = False                               # skid gear: taxi is a hover
taxi = full_rpm_idle if wheeled else hige     # guidance p. 19 (see caveat)
```

A hover or idle **event** is the standard single event with a stationary
track: one hemisphere, a fixed position, the receiver of interest. The
derived hemisphere carries the ring's `distance`, which the whole chain
honours as the reference distance; hover and flyover hemispheres recorded at
different distances run as separate events, which is also how the reference
implementation switches source by operating mode.

```python
t = np.arange(0.0, 30.5, 0.5)                 # 30 s of hover at 3 m
track = np.column_stack([np.zeros_like(t), np.zeros_like(t),
                         np.full_like(t, 3.0)])
res = aircraft.rotorcraft_event_level([hige], [0.0], [0.0], t, track, (100.0, 0.0))
```

What stays outside the implementation is §A.3.5's last resort when no hover
data exists at all: correcting two side-line level-flight microphones to the
150 m hover circle by the ICAO Annex 16 integrated method, a
measurement-correction chain this module (which consumes already-reduced
hemispheres) does not model.

## 5. Flight-path kinematics

`flight_path_kinematics` derives, from a time-stamped track by central finite
differences, everything the event needs (Eq. 16-21 / Doc 32 Eq. 8-10): ground
speed $V_g$, airspeed $V_A$ (zero wind), heading
$\Theta = \operatorname{atan2}(\Delta X, \Delta Y)$, curvature
$K = \Delta\Theta/\Delta S$, bank angle $\Phi = \arctan(K V_g^2 / g)$ and
path angle $\gamma = \arctan(\Delta Z / \Delta S)$.
The guidance recommends smoothing radar tracks (e.g. spline resampling to a
0.5 s cadence) before differentiating.

```python
kin = aircraft.flight_path_kinematics(times, positions)   # positions (N, 3), m
kin.plot()                                # speed and angle profiles
kin.airspeed, kin.path_angle              # select the hemisphere per point
kin.bank_angle                            # tilts the hemisphere in turns
```

## 6. The single event: `SEL`, `LASmax` and `EPNL`

`rotorcraft_event_level` runs the whole chain for one flyover at one receiver.
Per track point, the flight condition selects (or blends) the hemispheres and
the emission angles address the source level; the hemisphere frame is oriented
by the heading and tilted by the bank angle in turns (pitch attitude is
implicit in the hemispheres). The received one-third-octave history is
expressed at recorded time $t_r = t_e + r/c$ (Eq. 22, $c = 346.1\ \text{m/s}$)
and integrated: `LASmax`, `SEL` over the full history and over the certification
10 dB-down window (Doc 32 Eq. 27), and `EPNL` per ICAO Annex 16 (Doc 32
Eq. 28), reusing the library's `epnl_from_pnlt`.

```python
res = aircraft.rotorcraft_event_level(
    hemispheres, speeds, angles,          # the database
    times, positions,                     # the track (m, z up)
    receiver=(120.0, 0.0),                # ground position of the microphone
    ground=aircraft.RotorcraftGround(flow_resistivity="D"))   # grass site
res.la_max, res.sel, res.epnl             # LASmax, SEL, EPNL
res.plot()                                # the LA(t) time history
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_flyover_event_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_flyover_event.svg" alt="A-weighted level versus recorded time for a level rotorcraft flyover: a smooth rise to LASmax over the 10 dB-down window and a slower decay, annotated with the SEL and EPNL of the event" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import aircraft

# A synthetic helicopter-like hemisphere on the standard 31-band, 10° grid.
freqs = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)   # 10 Hz-10 kHz thirds
az = np.arange(-90.0, 91.0, 10.0)
po = np.arange(0.0, 181.0, 10.0)
spectrum = 88.0 - 12.0 * np.log10(freqs / 100.0) ** 2   # broad low-mid hump
levels = (spectrum[None, None, :] - 0.045 * np.abs(po - 80.0)[None, :, None]
          - 0.02 * np.abs(az)[:, None, None])
h = aircraft.RotorcraftHemisphere(freqs, az, po, levels)

speed = 30.87                                           # 60 kt, in m/s
t = np.arange(0.0, 130.01, 0.5)
track = np.column_stack([np.zeros_like(t), speed * (t - 65.0),
                         np.full_like(t, 150.0)])
event = aircraft.rotorcraft_event_level(
    [h], [speed], [0.0], t, track, (120.0, 0.0),
    ground=aircraft.RotorcraftGround(flow_resistivity="D"))
event.plot()
plt.show()
```

</details>

Radar-track workflows can hand the smoothed per-point `airspeed`, `path_angle`,
`heading` and `bank_angle` of a `RotorcraftTrackState` directly instead of
deriving them from the positions; when they are derived, the track is in metres and seconds, so the
database airspeeds must then be in m/s.

## 7. Ground-grid contours

`rotorcraft_noise_contour` evaluates the same event over a whole grid in one
vectorised pass per emission step and reduces each receiver's history to the
`SEL` (`metric="exposure"`) or `LASmax` (`metric="maximum"`) footprint:

```python
res = aircraft.rotorcraft_noise_contour(
    hemispheres, speeds, angles, times, positions,
    x=np.linspace(-2000.0, 2000.0, 81),
    y=np.linspace(-3000.0, 3000.0, 121),
    metric="exposure",
    ground=aircraft.RotorcraftGround(flow_resistivity="D"))
res.plot()                                # filled SEL contours
```

The ground may vary across the receivers without a full elevation model: the
`flow_resistivity` and `ground_elevation` of the `RotorcraftGround` accept one
value per grid point (shape `(len(y), len(x))`), and each receiver's two-ray
model then uses its local values.

## 8. Terrain: the mean ground plane and screening

Doc 32, 1st ed., assumes flat terrain; its guidance adds the machinery for
real sites. A varying vertical section is represented by its **mean ground
plane** (Eq. 36-40), the least-squares line through the terrain polyline
computed in closed form; source and receiver enter the flat-ground equations
with their **equivalent heights**, measured orthogonally to that plane and
floored at 0.1 m. Ground that changes type along the path averages its flow
resistivity by the logarithm, weighted by segment length (Eq. 41).

When terrain blocks the line of sight, the sound follows the shortest convex
path over it (the guidance's rubber band) and every touched vertex is a
**diffraction edge**. The attenuation combines the pure diffraction of the
path difference $\delta$ (Eq. 42-44,
$10 C_h \log_{10}\bigl(3 + (40/\lambda)\,C''\,\delta\bigr)$, capped at
25 dB) with the source-side and receiver-side ground effects, each over its
own mean ground plane and weighted by its image-path diffraction (Eq. 45-47,
the CNOSSOS-EU scheme the guidance adopts). The ground effect is not
evaluated separately in that regime.

```python
from phonometry import diffraction_attenuation

bands = [63.0, 250.0, 1000.0, 4000.0]
diffraction_attenuation(bands, 0.0, edge_height=2.5)   # grazing incidence
diffraction_attenuation(bands, 1.0, edge_height=2.5)   # 1 m into the shadow
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_insertion_loss_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_insertion_loss.svg" alt="Diffraction attenuation against path difference for four bands from 63 Hz to 4 kHz, over a 2.5 m edge: every curve is zero left of its own band-dependent threshold near the line of sight, the three bands with C_h = 1 cross about 4.8 dB at grazing incidence while 63 Hz crosses at 3.0 dB, and only the 4 kHz curve reaches the 25 dB cap within the plotted range" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import diffraction_attenuation

bands = np.array([63.0, 250.0, 1000.0, 4000.0])
delta = np.linspace(-0.4, 1.5, 441)
ld = np.array([diffraction_attenuation(bands, float(d), edge_height=2.5)
              for d in delta])

fig, ax = plt.subplots(figsize=(9, 6))
for i, label in enumerate(["63 Hz", "250 Hz", "1 kHz", "4 kHz"]):
    ax.plot(delta, ld[:, i], label=label)
ax.axvline(0.0, color="0.5", linestyle="--")
ax.axhline(25.0, color="0.5", linestyle=":", label="25 dB cap")
ax.set(xlabel="Path difference δ [m]",
       ylabel="Diffraction attenuation ΔLd [dB]")
ax.legend()
plt.show()
```

</details>

Every curve is flat at zero left of its own threshold (the guidance's
$(40/\lambda)\,C''\,\delta \ge -2$, so a longer wavelength still shows
attenuation further below the line of sight). Grazing incidence is
$10\,C_h\log_{10}3$, not a fixed value: the three bands whose frequency
already saturates $C_h$ at 1 (Eq. 43, $h_0$ = 2.5 m) cross the classical
$\approx 4.8$ dB, while 63 Hz, with $C_h = 0.63$ there, crosses at 3.0 dB
instead. Only the 4 kHz band is short enough to reach the 25 dB cap within
this range, at $\delta \approx 0.68$ m.

`mean_ground_plane`, `mean_flow_resistivity` and `diffraction_attenuation`
expose the pieces; `terrain_screening_adjustment` runs the whole section:

```python
import numpy as np
from phonometry import aircraft

d = [0.0, 150.0, 260.0, 300.0, 340.0, 420.0, 600.0]     # section distances
z = [0.0, 4.0, 48.0, 62.0, 40.0, 8.0, 2.0]              # terrain heights
freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)
res = aircraft.terrain_screening_adjustment(
    freqs, source=(0.0, 90.0), receiver=(600.0, 3.2), distances=d, heights=z,
    flow_resistivity="D")
res.screened, res.path_difference      # True, the rubber-band delta
res.adjustment                         # per band, replaces the flat-ground ΔLg
res.plot()                             # the section geometry
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_terrain_screening_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/rotorcraft_terrain_screening.svg" alt="Terrain screening section with a helicopter source, a hill blocking the line of sight to the microphone and the diffracted path over its crest, above the per-band ground-and-screening adjustment compared with the flat-ground comb" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import aircraft

freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
d = np.array([0.0, 150.0, 260.0, 300.0, 340.0, 420.0, 600.0])
z = np.array([0.0, 4.0, 48.0, 62.0, 40.0, 8.0, 2.0])
res = aircraft.terrain_screening_adjustment(
    freqs, (0.0, 90.0), (600.0, 3.2), d, z, flow_resistivity="D")
flat = aircraft.ground_effect_adjustment(freqs, 90.0, 1.2, 600.0,
                                         flow_resistivity="D")

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7))
res.plot(ax=ax)
ax2.axhline(0.0, color="0.5", linewidth=1.0)
ax2.semilogx(freqs, flat, ls="--", marker="s", markersize=3,
             label="Flat ground (no hill)")
ax2.semilogx(freqs, res.adjustment, marker="o", markersize=3,
             label="Screened by the hill (Eq. 45-47)")
ax2.set(xlabel="One-third-octave-band centre frequency [Hz]",
        ylabel="Ground and screening adjustment [dB]")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()
plt.show()
```

</details>

The event and contour run over real sites by passing a digital elevation
model in the `RotorcraftGround`: `terrain=(x, y, z)` on the track frame. Every
emission-receiver pair then samples its own vertical section at
`terrain_resolution` (default: the model's cell size) and evaluates it with the machinery above; the receiver
ground comes from the model. The cost grows with track points times grid
points, so keep contour grids modest with terrain.

```python
res = aircraft.rotorcraft_event_level(
    hemispheres, speeds, angles, times, positions, receiver=(1200.0, 300.0),
    ground=aircraft.RotorcraftGround(terrain=(tx, ty, tz),
                                     flow_resistivity="D"))
```

## Validation

Validated against the NORAH2 guidance Table 4 (all 31 bands, 10 Hz-10 kHz), the
closed-form inverse-square spreading, the analytic rigid-ground +6 dB and
grazing limits of the ground effect, off-node bilinear lookups on the reference
hemispheres of all eleven rotorcraft types, hand-checked interpolation
simplices, closed-form kinematics and the Lorentzian $\mathrm{SEL} - L_{ASmax}$ flyover
integral, and end to end against the NORAH2 prototype's ARP verification cases:
emission angles reproduce to 0.01°, retarded times to 0.02 s, every step level
of the hard-ground events to 0.08 dB(A) out to 18 km, `LASmax` to 0.03 dB and
`SEL` to 0.05 dB over hard ground (0.4 dB over soft ground), the 187-microphone
contour grid to 0.7 dB worst-case, `PNLTM` to 0.1 dB and `EPNL` to about
1.3 dB (the prototype's sub-noy-floor perceived-noisiness policy differs from
the published Annex 16 law). One documented
divergence remains: at far range over soft ground the prototype damps the
coherent two-ray interference of guidance Eq. 30 towards the incoherent sum
(up to 4.9 dB on individual low-level steps beyond 7 km); neither Doc 32 nor
the guidance contains such a term, and this implementation follows the
published equations.

The Table 3 hover/idle derivation is anchored in closed form (the ring-to-rim
identity, hand-computed constant-φ bins, the exact Approach 3 offsets) and
end to end against the prototype's out-of-ground-hover verification case:
its eight microphones, spanning bearings, elevations, receiver heights from
0.01 m to 8.2 m and four flow resistivities, reproduce `LASmax`/`SEL` to
0.45 dB worst case with the prototype's own conventions passed explicitly
(`mapping="bearing"` and its +8 dB database correction as `offset_db`; the
guidance's constant-φ text and published +12 dB are the defaults). The
residual peaks on the nadir microphones, where the prototype's damped
interference differs most from the published coherent Eq. 30.

The terrain machinery is anchored in closed form: the mean ground plane is
exact on linear and symmetric profiles, a flat section reproduces the
flat-ground model to machine precision and an inclined plane its analytic
rotation, the log-mean resistivity recovers the geometric mean, the grazing
diffraction gives the classical $10\log_{10} 3$, and a hand-checked hill fixes
the rubber-band path difference. Per-receiver ground handling validates end
to end against the prototype's ARP Case 3 (187 microphones, each on its own
ground elevation: every step level to 0.08 dB(A), `SEL`/`LASmax` to 0.05 dB,
the contour grid to 0.15 dB) and the mixed-ground Case 2 grid reproduces in
a single per-receiver-resistivity call. The prototype's public release does
not include a reconstructible screening case (the frame of its terrain model
could not be pinned to the published outputs), so the diffraction chain
itself rests on the closed-form anchors and its CNOSSOS-EU lineage.

## See also

- [Aircraft noise: Effective Perceived Noise Level](aircraft-noise.md):
  the ICAO Annex 16 Appendix 2 chain this page's Doc 32 Eq. 28 term calls, and
  the measurement practice a hemisphere campaign follows.
- [Airport Noise (ECAC Doc 29)](airport-noise.md): the
  fixed-wing contour method, where a noise-power-distance table plays the role
  the noise hemisphere plays here.
- [Outdoor sound propagation](../environment/propagation/outdoor-propagation.md):
  the ISO 9613-2 ground effect and the CNOSSOS ground classes this two-ray
  adjustment shares.
- API reference: [`aircraft.rotorcraft_noise`](https://jmrplens.github.io/phonometry/reference/api/aeroacoustics/rotorcraft-noise/).

## References

- European Civil Aviation Conference. (2026). *Report on standard method of
  computing rotorcraft noise contours* (ECAC.CEAC Doc 32, 1st ed.).
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-CEAC-DOC_32-REPORT_ON_STANDARD_METHOD_OF_COMPUTING_ROTORCRAFT_NOISE_CONTOURS.pdf).
  The standard rotorcraft contour method whose hemisphere source model and
  propagation adjustments this page implements.
- Olsen, H., Tuinstra, M., & van Oosten, N. (2024). *Rotorcraft noise
  modelling guidance* (Research Project NOISE SC01, deliverable D1.5d,
  contract EASA.2020.FC.06). European Union Aviation Safety Agency.
  [EASA project page](https://www.easa.europa.eu/en/research-projects/environmental-research-rotorcraft-noise),
  [free PDF](https://www.easa.europa.eu/en/downloads/132005/en).
  The equation-level guidance (Eq. 13-35) behind the implementation, with the
  Table 4 attenuation values and the reference hemispheres used as oracles.
- Chien, C. F., & Soroka, W. W. (1975). Sound propagation along an impedance
  plane. *Journal of Sound and Vibration*, 43(1), 9-20.
  [doi:10.1016/0022-460X(75)90200-X](https://doi.org/10.1016/0022-460X(75)90200-X).
  The two-ray interference solution over an impedance plane behind the
  section 2 ground-effect adjustment.
- Delany, M. E., & Bazley, E. N. (1970). Acoustical properties of fibrous
  absorbent materials. *Applied Acoustics*, 3(2), 105-116.
  [doi:10.1016/0003-682X(70)90031-9](https://doi.org/10.1016/0003-682X(70)90031-9).
  The one-parameter flow-resistivity impedance model the ground effect
  evaluates.
- Kephalopoulos, S., Paviotti, M., & Anfosso-Lédée, F. (2012). *Common noise
  assessment methods in Europe (CNOSSOS-EU)* (EUR 25379 EN). Publications
  Office of the European Union.
  [doi:10.2788/31776](https://doi.org/10.2788/31776),
  [JRC repository](https://publications.jrc.ec.europa.eu/repository/handle/JRC72550).
  The flow-resistivity ground classes `"A"`-`"H"` accepted by
  `ground_effect_adjustment`.

## Standards

ECAC Doc 32, 1st ed., *Report on Standard Method of Computing
Rotorcraft Noise Contours*; NORAH2 rotorcraft-noise modelling guidance
(EASA.2020.FC.06 SC01.D1.5d), §A.3-A.5: the noise hemisphere (§A.3.2), spherical
spreading (Eq. 24), atmospheric attenuation (Eq. 26/27, ISO 9613-1 coefficient,
Table 4), ground effect (Chien-Soroka, Eq. 28-35, Delany-Bazley impedance,
CNOSSOS flow resistivity), flight-condition interpolation (Eq. 3-10),
flight-path kinematics (Eq. 16-21 / Doc 32 Eq. 8-10), recorded time (Eq. 22),
the single-event metrics `SEL`/`LASmax` and `EPNL` (Doc 32 Eq. 27/28, ICAO
Annex 16 App. 2), the mean ground plane and equivalent heights (Eq. 36-40),
the log-mean flow resistivity (Eq. 41), the terrain screening chain
(Eq. 42-47 with the guidance's noise-path appendices; CNOSSOS-EU lineage) and
the hover/idle source derivation of §A.3.5 Table 3 (the ring-to-hemisphere
constant-φ extension, the Approach 2/3 offsets and the taxi selection rule).
Only §A.3.5's no-hover-data fallback (two side-line microphones corrected to
the 150 m hover circle by the ICAO Annex 16 integrated method) remains
outside the implementation.
