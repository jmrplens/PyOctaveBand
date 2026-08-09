← [Documentation index](../README.md)

# The ANP fleet database

[Airport noise (ECAC Doc 29)](airport-noise.md) computes an event level from a
noise-power-distance table and a flight path. That page supplies both by hand,
which is what you want while learning the method and what you never want
afterwards: for a real study the numbers come from the **Aircraft Noise and
Performance (ANP)** database that EASA and EUROCONTROL publish for the aircraft
types actually flying.

That database ships with phonometry. This page is the bridge between it and the
Doc 29 functions: how to open it, what one aircraft record holds, and how to go
from an aircraft identifier to an event level or a contour without writing a
table yourself.

## Opening the database

`load_anp_database()` with no argument reads the copy shipped with the package.
Point it at a directory to read any other ANP CSV export instead.

```python
from phonometry import load_anp_database

db = load_anp_database()
print(len(db.aircraft_ids))          # 155 aircraft types
ac = db.aircraft("747100")
print(ac.description)                # Boeing 747-100 / JT9DBD
print(ac.engine_type, ac.num_engines, ac.weight_class)
print(ac.power_parameter, ac.mounting)   # CNT (lb) wing
```

An `AnpAircraft` describes itself with the engine type and count and the ICAO
wake weight class, and carries two fields you will use directly. The **power
parameter** names the quantity the NPD table is indexed by: not a force in
newtons but whatever the manufacturer tabulated against, corrected net thrust in
pounds for most jets, so a power you pass to `level` has to be in those units.
The engine **mounting** is the one field of the record the Doc 29 chain itself
reads; it is derived from the ANP lateral directivity identifier, is one of
`"wing"`, `"fuselage"` or `"propeller"`, and selects the engine-installation
correction that [Airport noise](airport-noise.md) applies by hand. The choice is
not cosmetic: at small depression angles the wing and fuselage corrections differ
by more than a decibel, a propeller takes none of that correction at all, and the
shipped fleet splits 70 / 55 / 30 across the three. An unrecognised identifier
falls back to `"wing"`.

## The noise-power-distance curves

`npd_curves` returns the tabulated NPD surface for one operation (`"D"` for
departure, `"A"` for arrival) and one metric (`"SEL"` or `"LAmax"`): a level for
each combination of engine power setting and slant distance. Between the
tabulated nodes the Doc 29 interpolation is logarithmic in distance and linear
in power.

```python
from phonometry import load_anp_database

curves = load_anp_database().aircraft("A320-232").npd_curves("D", "SEL")
print(curves.powers)                       # [10000. 14000. 19000. 23000.] lb
print(curves.level(19000.0, [304.8, 1000.0, 3000.0]))
```

The distances are metres. The database tabulates them in feet, at the ten Doc 29
nodes from 200 ft to 25000 ft, and this bridge converts on read so everything
downstream stays in SI. `curves.plot()` draws the whole family:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anp_npd_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anp_npd.svg" alt="Noise-power-distance curves of a Boeing 747-100 for three tabulated thrust settings, each level falling with slant distance on a logarithmic axis, with markers on the tabulated nodes" width="82%">
</picture>

The three tabulated thrust settings are close to parallel: power mostly shifts
the level, while distance sets the shape.

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import load_anp_database

ac = load_anp_database().aircraft("747100")
curves = ac.npd_curves("D", "SEL")

fig, ax = plt.subplots(figsize=(10, 6))
curves.plot(ax=ax)
ax.set_title(f"ANP NPD Curves - {ac.description} (SEL, departure)")
ax.text(0.02, 0.06,
        f"power parameter: {ac.power_parameter}\n"
        "markers: tabulated NPD nodes",
        transform=ax.transAxes, va="bottom", fontsize=9)
plt.show()
```

</details>

## The default trajectory

An aircraft record also carries default trajectories. `profile` returns one as a
Doc 29 flight path: an `(N, 5)` array of along-track, lateral and vertical
position plus the power setting and true airspeed, with boolean masks marking
which segments are the takeoff ground roll or the landing rollout. The **stage
length** selects the trip-distance bin: a longer stage means more fuel, more
weight and a shallower climb, so the same aircraft has one profile per bin.

Only the **fixed-point** profiles are read as ready-to-use trajectories. Most
ANP entries describe their departures as procedural steps instead (climb at this
rate to that altitude, accelerate, retract flaps), which have to be flown
through a flight-mechanics performance model before they become a path. That
model is outside this bridge, so of the 155 aircraft in the shipped database 13
have a fixed-point *departure* profile and 20 a fixed-point *arrival* profile.
Doc 29 Vol. 2 Appendix G3.5 defines the bins by trip length in nautical miles
(1 is 0-500, 2 is 500-1 000, 3 is 1 000-1 500, 4 is 1 500-2 500, then 1 000 nmi
steps), and the shipped profiles cover stage lengths 1 to 7 for departures and 1
only for arrivals. Asking for one that does not exist raises a `KeyError` naming
the stage lengths that do, which is how to discover them:

```python
from phonometry import load_anp_database

db = load_anp_database()
for identifier, operation, stage in (("A320-232", "D", 1), ("747100", "D", 9)):
    try:
        db.profile(identifier, operation, stage)
    except KeyError as exc:
        print(exc)
# ... 'A320-232', operation 'D', stage length 1 (available stage lengths: [])
# ... '747100', operation 'D', stage length 9 (available: [1, 2, 3, 4, 5, 6])
```

An empty list means the type has no fixed-point profile at all and needs a
substitute trajectory. NPD curves, on the other hand, are tabulated for every
aircraft in the database.

```python
from phonometry import load_anp_database

profile = load_anp_database().aircraft("747100").profile("D", stage_length=1)
print(profile.profile_id, profile.stage_length, profile.path.shape)
profile.plot()
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anp_profile_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anp_profile.svg" alt="Default departure profile of a Boeing 747-100: altitude against along-track distance, rising from the runway through eleven fixed points, with the ground-roll points marked at zero altitude" width="82%">
</picture>

The first points sit at zero altitude and are the ground roll; the climb gradient
after them fixes the slant distance at every receiver, so it decides the contour.

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import load_anp_database

ac = load_anp_database().aircraft("747100")
profile = ac.profile("D", stage_length=1)

fig, ax = plt.subplots(figsize=(10, 6))
profile.plot(ax=ax)
ax.set_title(f"ANP Default Departure Profile - {ac.description}")
ax.text(0.98, 0.06,
        f"stage length {profile.stage_length}, "
        f"{profile.path.shape[0]} fixed points",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9)
plt.show()
```

</details>

## Straight to an event level or a contour

With both halves in the record, the aircraft can run the Doc 29 chain itself.
`event_level` places one flyover at a receiver, and `noise_contour` sweeps it
over a ground grid, each wiring the NPD curves and the default profile into the
functions the airport-noise page builds by hand.

```python
from phonometry import load_anp_database

ac = load_anp_database().aircraft("747100")
flyover = ac.event_level([3000.0, 500.0, 0.0], "D")
print(round(float(flyover.level), 1))     # 100.4 dB
```

The observer is `(x, y, z)` in metres in the runway frame, whose origin is not
the airport boundary: `x` runs along the runway centre line with `x = 0` at start
of roll for a departure and at the landing threshold for an arrival, so arrival
profiles carry negative `x` on final approach and the two frames sit nearly 35 km
apart. `y` is the lateral offset from the extended centre line, positive to
starboard, and its sign selects the depression-angle branch of the banked-segment
rule; `z` is the receiver height above local ground, left at 0 here where Doc 29
measures at 1.2 m. The metric defaults to the sound exposure level and the stage
length to 1, and the optional `temperature` and `pressure` arguments feed the
Doc 29 atmospheric impedance adjustment, defaulting to the 15 °C and 101.325 kPa
of the standard atmosphere — a bookkeeping term worth +0.07 dB there, not a
weather correction.

```python
import numpy as np
from phonometry import load_anp_database

contour = load_anp_database().aircraft("747100").noise_contour(
    "D",
    x=np.linspace(-2000.0, 12000.0, 40),
    y=np.linspace(-3000.0, 3000.0, 30),
)
print(contour.level.shape)      # (30, 40): one SEL per grid point, indexed (y, x)
contour.plot()
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anp_contour_dark.svg">
  <img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anp_contour.svg" alt="Single-event sound exposure level contour of a Boeing 747-100 departure computed from the shipped ANP database, with the take-off ground roll and the default ground track overlaid and the event_level receiver marked at 100.4 dB" width="90%">
</picture>

Everything these two return is the same result type the airport-noise page uses,
so the plotting, the contour extraction and the per-segment breakdown all work
unchanged.

## What this guide covers

**Covered.** Opening the shipped EASA ANP database, or another ANP CSV export,
with `load_anp_database`; what one aircraft record holds, including the power
parameter its NPD table is indexed by and the engine mounting the Doc 29 chain
reads; reading and interpolating the NPD surface with `npd_curves` and `level`;
the default fixed-point trajectories and their stage-length bins; and driving
the Doc 29 single-event level and ground-grid contour from an aircraft
identifier with `event_level` and `noise_contour`.

**Not covered.** The procedural-step profiles, which is how most ANP entries
describe a departure. Turning those into a flight path needs the ICAO Doc 9911
flight-mechanics performance model, which this bridge does not implement, so
only the 13 types with a fixed-point *departure* profile and the 20 with a
fixed-point *arrival* profile come with a trajectory ready to fly. The database
is read and never written: version 2.3 ships with the package and the package
does not update it.

## See also
Pages elsewhere on the site that this section leans on:

- [Airport Noise (ECAC Doc 29)](airport-noise.md): the method
  itself, built from a hand-written NPD table and flight path.
- [Aircraft noise: Effective Perceived Noise Level](aircraft-noise.md):
  the certification metric, which is measured rather than tabulated.
- API reference:
  [`aircraft.anp_fleet`](https://jmrplens.github.io/phonometry/reference/api/aeroacoustics/anp-fleet/).

## References

- EASA and EUROCONTROL, *Aircraft Noise and Performance (ANP) database*, version
  2.3 (2020). <https://www.aircraftnoisemodel.org/>. The provenance of the copy
  shipped with the package is recorded in `aircraft/data/anp/PROVENANCE.md`.
- ECAC, *Report on standard method of computing noise contours around civil
  airports*, Volume 2: Technical guide, Doc 29 4th ed. (2016).
