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
aircraft = db.aircraft("747100")
print(aircraft.description)          # Boeing 747-100 / JT9DBD
print(aircraft.engine_type, aircraft.num_engines, aircraft.weight_class)
```

An `AnpAircraft` describes itself with the engine type and count and the ICAO
wake weight class, and carries two fields you will use directly. The **power
parameter** names the quantity the NPD table is indexed by: not a force in
newtons but whatever the manufacturer tabulated against, corrected net thrust in
pounds for most jets, so a power you pass to `level` has to be in those units.
The engine **mounting** is the one field of the record the Doc 29 chain itself
reads; it is derived from the ANP lateral directivity identifier, is one of
`"wing"`, `"fuselage"` or `"propeller"`, and selects the engine-installation
correction that [Airport noise](airport-noise.md) applies by hand. The shipped
fleet splits 70 / 55 / 30 across the three, and an unrecognised identifier falls
back to `"wing"`.

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
have a fixed-point departure profile. Asking for one that does not exist raises
a `KeyError` naming the stage lengths that do. NPD curves, on the other hand,
are tabulated for every aircraft in the database.

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


## Straight to an event level or a contour

With both halves in the record, the aircraft can run the Doc 29 chain itself.
`event_level` places one flyover at a receiver, and `noise_contour` sweeps it
over a ground grid, each wiring the NPD curves and the default profile into the
functions the airport-noise page builds by hand.

```python
from phonometry import load_anp_database

aircraft = load_anp_database().aircraft("747100")
flyover = aircraft.event_level([3000.0, 500.0, 0.0], "D")
print(round(float(flyover.level), 1))     # SEL in dB at that receiver
```

The observer is `(x, y, z)` in metres in the runway frame: along the track,
across it and above the ground. The optional `temperature` and `pressure`
arguments feed the Doc 29 atmospheric impedance adjustment and default to the
15 °C and 101.325 kPa of the standard atmosphere.

```python
import numpy as np
from phonometry import load_anp_database

contour = load_anp_database().aircraft("747100").noise_contour(
    "D",
    x=np.linspace(-2000.0, 12000.0, 40),
    y=np.linspace(-3000.0, 3000.0, 30),
)
print(contour.level.shape)      # (30, 40): one SEL per grid point
```

Everything these two return is the same result type the airport-noise page uses,
so the plotting, the contour extraction and the per-segment breakdown all work
unchanged.

## References

- EASA and EUROCONTROL, *Aircraft Noise and Performance (ANP) database*, version
  2.3 (2020). <https://www.aircraftnoisemodel.org/>. The provenance of the copy
  shipped with the package is recorded in `aircraft/data/anp/PROVENANCE.md`.
- ECAC, *Report on standard method of computing noise contours around civil
  airports*, Volume 2: Technical guide, Doc 29 4th ed. (2016).
