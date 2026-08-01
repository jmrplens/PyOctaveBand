← [Documentation index](README.md)

# Airport noise (ECAC Doc 29)

Certification measures one aeroplane at one reference point on one flight;
an airport study asks a different question: how loud is it *there*, on that
street, for that departure. The ECAC Doc 29 method answers it without
re-measuring anything. The aeroplane arrives as a **noise-power-distance
(NPD)** table, a measured event level against slant distance for a handful
of engine power settings; the flight arrives as a path of segments; and the
method corrects the NPD baseline segment by segment for everything the
tables could not know: the atmosphere on the day, the finite length of each
segment, how far off to the side the receiver sits, where the engines are
mounted, and the rearward lobe of a jet still on the runway.

This page covers that chain end to end, from the NPD interpolation to the
ground-grid contour, and it is validated against the reference workbook of
Doc 29 5th ed. Vol. 3. The certification metric these tables ultimately come
from, the EPNL of ICAO Annex 16, is [Aircraft noise](aircraft-noise.md); the
physical propagation ingredients (ground effect, atmospheric absorption,
barriers) are [Outdoor sound propagation](outdoor-propagation.md).

## 1. The noise-power-distance engine

The ECAC Doc 29 airport-noise method describes an aircraft with **noise-power-
distance (NPD)** tables that give the event level ($L_{Amax}$ or `SEL`) of steady straight
flight versus engine power and slant distance. `npd_level` reads an event level
for an arbitrary power and distance, interpolating **linearly in power**
(Eq. 4-3) and **log-linearly in distance** (Eq. 4-4), extrapolating from the
terminal segments beyond the tabulated envelope.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/airport_noise_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/airport_noise.svg" alt="Noise-power-distance curves for two engine power settings: the event level falls log-linearly with slant distance between the tabulated nodes, higher power giving a higher level" width="82%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import aircraft

# A schematic NPD table: SEL vs slant distance for two thrust settings.
powers = [12000.0, 20000.0]
distances = [200.0, 400.0, 630.0, 1000.0, 2000.0, 4000.0, 6300.0, 10000.0]
levels = [[98.5, 92.0, 88.2, 83.6, 76.8, 69.4, 63.9, 56.8],
          [107.2, 100.9, 97.2, 92.7, 86.0, 78.5, 72.9, 65.6]]

fig, ax = plt.subplots()
for p in (20000.0, 12000.0):
    curve = aircraft.npd_curve(powers, distances, levels, power=p)
    line, = ax.semilogx(curve.distance, curve.level, label=f"P = {p:.0f} N")
    ax.semilogx(curve.table_distances, curve.table_levels, "o", markersize=4,
                color=line.get_color())
ax.set(xlabel="Slant distance [m]", ylabel="Event level [dB]",
       title="Noise-power-distance curves (ECAC Doc 29)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
```

</details>

```python
from phonometry import aircraft

powers = [12000.0, 20000.0]                      # e.g. net thrust, N
distances = [200.0, 400.0, 1000.0, 2000.0, 6300.0, 10000.0]
levels = [[98.5, 92.0, 83.6, 76.8, 63.9, 56.8],
          [107.2, 100.9, 92.7, 86.0, 72.9, 65.6]]
aircraft.npd_level(powers, distances, levels, power=16000.0, distance=1500.0)

curve = aircraft.npd_curve(powers, distances, levels, power=20000.0)
curve.plot()   # NPD curve with the tabulated nodes (needs matplotlib)
```

## 2. The single-event calculation

The full ECAC Doc 29 single-event calculation places a flight path's noise at a
receiver by breaking the path into segments and, for each, correcting the NPD
baseline level (§4.3-4.5):

- **`impedance_adjustment`** ($T$, $p$): corrects the NPD data from their reference
  air impedance (409.81 N·s/m³) to the aerodrome's temperature and pressure
  (Eq. 4-6/4-7; +0.074 dB under the standard atmosphere).
- **`lateral_attenuation`** ($\beta$, $\ell$): excess lateral attenuation over
  soft ground
  (Eq. 4-18/4-19, AIR-5662).
- **`engine_installation_correction`** ($\varphi$, mounting): lateral-
  directivity term for wing/fuselage/propeller installations (Eq. 4-15/4-16).
- **`duration_correction`** ($V_{\text{ref}}$, $V_{\text{seg}}$): the
  speed/duration adjustment for
  exposure levels (Eq. 4-14).
- **`noise_fraction`** ($q$, $\lambda$, $d_\lambda$): the finite-segment energy
  fraction (Eq. 4-20).
- **`start_of_roll_directivity`** ($\psi$, $d_{\text{SOR}}$, engine): the
  rearward jet/turboprop
  directivity behind takeoff ground-roll segments (Eq. 4-22/4-24/4-25). Pass a
  boolean `ground_roll` mask to `event_level`/`noise_contour` to flag the takeoff
  ground-roll segments; behind them the reduced ($q = 0$) noise fraction and
  $\Delta_{\text{SOR}}$ are applied (Eq. 4-9).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/airport_contour_dark.webp">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/airport_contour.webp" alt="Single-event SEL contour of a departure: an elongated footprint along the flight track, loudest near the ground roll and decaying as the aircraft climbs away" width="90%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import aircraft

# NPD tables (SEL and LAmax) for one aircraft, two power settings.
powers = [8000.0, 12000.0]
distances = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0, 7680.0]
sel = [[98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0, 56.0],
       [104.0, 98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0]]
lmax = [[94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0, 52.0],
        [100.0, 94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0]]

# Departure: ground roll along +x, then a steady climb.
xs = np.linspace(0.0, 18000.0, 40)
z = np.clip((xs - 1500.0) * 0.11, 0.0, 2500.0)
power = np.where(xs < 3000.0, 12000.0, 10000.0)
path = np.column_stack([xs, np.zeros_like(xs), z, power, np.full_like(xs, 82.3)])
ground_roll = xs[:-1] < 1500.0   # takeoff roll: segments still on the runway

contour = aircraft.noise_contour(path, powers, distances, sel, lmax,
                                 ground_roll=ground_roll,
                                 x=np.linspace(-2500.0, 20000.0, 56),
                                 y=np.linspace(-6000.0, 6000.0, 44))
contour.plot()   # single-event SEL footprint (needs matplotlib)
plt.show()
```

</details>

The mechanism behind these ground corrections is two-path interference: the
direct wave and its ground reflection. Below, a 400 Hz source 1.5 m above a
rigid plane forms the lobe pattern, with the image source ghosted below the
ground and a receiver sitting in an interference dip.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_ground_effect_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_ground_effect.gif" alt="Animation: a 2D FDTD simulation of a 400 Hz point source 1.5 metres above rigid ground; the direct and ground-reflected wavefronts interfere and a lobe pattern forms, the ghosted image source below the ground explains the geometry, and the level on an 8 metre arc converges to the two-path image-source model with its predicted nulls" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_fdtd_ground_effect.webm)

$\Delta_{\text{SOR}}$ is what makes the departure footprint bulge rearward
behind the runway: jet-exhaust noise radiates a lobed pattern in the rear arc,
strongest at an azimuth $\psi \approx 120°$ from the nose and falling away
both abeam ($\psi = 90°$) and directly behind ($\psi = 180°$).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/airport_sor_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/airport_sor.svg" alt="Polar diagram of the start-of-roll directivity ΔSOR over the rearward semicircle for turbofan-jet and turboprop aircraft: both show a lobe near 120° from the nose and fall off directly behind the aircraft" width="70%">
</picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import aircraft

az = np.linspace(90.0, 270.0, 361)              # rearward semicircle
psi = np.where(az <= 180.0, az, 360.0 - az)     # ΔSOR is left/right symmetric
jet = [aircraft.start_of_roll_directivity(p, 300.0, "jet") for p in psi]
prop = [aircraft.start_of_roll_directivity(p, 300.0, "turboprop") for p in psi]

ax = plt.subplot(projection="polar")
ax.set_theta_zero_location("N")                 # nose up, azimuth clockwise
ax.set_theta_direction(-1)
ax.plot(np.radians(az), jet, label="Turbofan jet")
ax.plot(np.radians(az), prop, label="Turboprop")
ax.set_rlim(-16.0, 0.0)                         # radial axis: dB re abeam
ax.legend(loc="lower center")
plt.show()
```

</details>

`event_level` assembles these (Eq. 4-8/4-9) and sums the segments into the exposure
level `SEL` (Eq. 4-11) or the maximum level $L_{Amax}$ (Eq. 4-10); `noise_contour`
evaluates `event_level` over a ground grid to produce a noise contour. Mark the
takeoff ground-roll segments with the boolean `ground_roll` mask.

```python
import numpy as np
from phonometry import aircraft

# NPD tables (SEL and LAmax) for one aircraft, two power settings.
powers = [8000.0, 12000.0]
distances = [60.0, 240.0, 960.0, 3840.0]
sel = [[98.0, 86.0, 74.0, 62.0], [104.0, 92.0, 80.0, 68.0]]
lmax = [[94.0, 82.0, 70.0, 58.0], [100.0, 88.0, 76.0, 64.0]]

# A departure flight path: columns x, y, z (m), power, speed (m/s).
xs = np.linspace(0.0, 18000.0, 40)
path = np.column_stack([xs, np.zeros_like(xs), np.clip((xs - 1500) * 0.11, 0, 2500),
                        np.where(xs < 3000, 12000.0, 10000.0), np.full_like(xs, 82.3)])
ground_roll = xs[:-1] < 1500.0   # takeoff roll: segments still on the runway

aircraft.event_level(path, [2000.0, 500.0, 0.0], powers, distances, sel, lmax,
                     ground_roll=ground_roll)  # SEL at a point
contour = aircraft.noise_contour(path, powers, distances, sel, lmax, ground_roll=ground_roll,
                           x=np.linspace(-2500, 20000, 60), y=np.linspace(-6000, 6000, 48))
contour.plot()   # SEL contour over the ground (needs matplotlib)
```

Validated against the **ECAC Doc 29 5th ed. Vol 3 Part 1 reference workbook**:
the segment geometry ($\beta$, $\varphi$), lateral attenuation, engine
installation, noise fraction and the start-of-roll directivity
$\Delta_{\text{SOR}}$ (turbofan and turboprop, all 124 ground-roll reference
rows to $< 0.01\ \text{dB}$) reproduce the reference values, and the segment
energy sum matches the reference `SEL`.

The model also covers the landing rollout (`landing_roll` mask: reduced noise
fraction Eq. 4-21b, nearest-end geometry, no directivity term), per-segment
bank angle (`bank`, §4.5.2 sign convention), the §4.5.5 nearest-end lateral
geometry behind takeoff roll, the Eq. 4-13b average runway-segment speed and
the recommended 30 m floor on NPD lookups. Seven branch-covering receptor
events of the reference workbook are reproduced end-to-end in the test suite.

## See also

- [Aircraft noise](aircraft-noise.md): the ICAO Annex 16 certification
  metric behind the aircraft data.
- [Outdoor sound propagation](outdoor-propagation.md): the ISO 9613-2
  attenuation terms and the ground effect the lateral attenuation condenses
  into one curve.
- [Environmental levels](environmental-levels.md): the $L_{den}$-style
  long-term indices that a full airport study accumulates from single events.
- API reference: [`aircraft.airport_noise`](https://jmrplens.github.io/phonometry/reference/api/aeroacoustics/airport-noise/).

## References

- European Civil Aviation Conference. (2016). *Report on standard method of
  computing noise contours around civil airports* (ECAC.CEAC Doc 29, 4th ed.),
  Volume 2: Technical guide.
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf).
  The NPD interpolation of section 1 and the single-event segment chain of
  section 2.
- European Civil Aviation Conference. (2026). *Report on standard method of
  computing noise contours around civil airports* (ECAC.CEAC Doc 29, 5th ed.),
  Volume 3: Reference cases and verification framework.
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-CEAC-DOC_29_5th_Edition-REPORT_ON_STANDARD_METHOD_OF_COMPUTING_NOISE_CONTOURS_AROUND_CIVIL_AIRPORTS-Volume_3-REFERENCE_CASES_AND_VERIFICATION_FRAMEWORK.pdf).
  The reference workbook the section 2 single-event chain is validated
  against.
- SAE International. (2006). *Method for predicting lateral attenuation of
  airplane noise* (SAE AIR 5662).
  [sae.org](https://www.sae.org/standards/content/air5662/).
  The soft-ground lateral-attenuation model that Doc 29 §4.5.4 adopts in
  section 2.

## Standards

ECAC Doc 29, 4th ed., Vol 2 (2016): the NPD event-level interpolation (§4.2) and
the single-event segment calculation (duration, §4.5.1; engine installation,
§4.5.3; lateral attenuation, §4.5.4, AIR-5662; the finite-segment noise
fraction, §4.5.6; the start-of-roll directivity, §4.5.7; and segment summation,
§4.3) through to ground-grid noise contours, with the impedance adjustment
(§4.2.1). The single-event chain is validated against the ECAC Doc 29 5th ed.
Vol 3 Part 1 reference workbook.
