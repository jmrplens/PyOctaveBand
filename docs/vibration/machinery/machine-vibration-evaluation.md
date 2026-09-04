← [Documentation index](../../README.md)

# Evaluating machine vibration (ISO 20816-1)

[Machine fault frequencies](machine-diagnostics.md) answers *where a fault would
show*. This page answers the question a plant asks first: **is this machine
acceptable at all**. The two are different trades. A kinematic line tells you
what is wrong; a severity grade tells you whether to keep running, and it comes
from one broad-band number measured at a bearing.

ISO 20816-1:2016 is the basis document of the ISO 20816 series, which merged
and replaced ISO 10816-1 and ISO 7919-1. It fixes the shape of the judgement
and leaves the numbers to the machine-specific parts: Part 2 for large gas
turbines, Part 3 for industrial machines, Part 4 for gas turbines, Part 5 for
hydraulic plant, Part 8 for reciprocating compressors, Part 9 for gear units.

## 1. What is measured, and where

Two kinds of measurement, and a machine may carry both.

**On non-rotating parts**, an accelerometer or velocity transducer on the
bearing, its support housing, or another structural part that responds to the
dynamic forces coming through the bearing. Clause 4.4.1 asks for three mutually
perpendicular directions at each position for acceptance testing; operational
monitoring is usually met by one or both radial directions, and the axial one
is evaluated only on thrust bearings, where direct axial dynamic force is
transmitted.

**On the rotating shaft**, a pair of non-contacting probes reading the shaft
displacement directly. Clause 4.4.2.1 places them radially in one transverse
plane, their axes within 5° of a radial line, preferably 90° ± 5° apart on the
same bearing half, and at the same positions on every bearing of the machine.
A single probe per plane is allowed only where it is known to give adequate
information.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_machine_vibration_positions_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_machine_vibration_positions.svg" alt="Elevation of a machine train on a common baseplate: a motor and a driven machine joined by a coupling, each end carried on a pedestal bearing. On the near bearing an accelerometer stands on the housing, with three arrows leaving it, one vertical, one horizontal to the left and one drawn as a foreshortened diagonal for the axial direction that leaves the plane of the drawing. On the far bearing two non-contacting probes reach the shaft from above at plus and minus forty-five degrees, and a dashed leader carries up to a cross-section of the bearing bore in which the two probes are drawn ninety degrees apart with the angle marked ninety degrees plus or minus five. A strip along the bottom shows the chain a reading runs through, from transducer to conditioning to processing" width="92%"></picture>

*The two measurement families of Clause 4.4, on one train. The axial arrow is
drawn as a foreshortened diagonal because it leaves the plane of an elevation.*

The measured quantity is broad-band, and the **vibration severity** is the
largest magnitude found at any bearing at rated speed under steady operation
(4.3.3). It is that single number the zones grade.

## 2. Four zones, not a pass and a fail

Clause 6.3.2.3 grades rather than judges:

| Zone | What it means |
| :--- | :--- |
| **A** | Where newly commissioned machines normally fall. |
| **B** | Acceptable for unrestricted long-term operation. |
| **C** | Unsatisfactory for long-term continuous operation; the machine may run a limited period until remedial action can be arranged. |
| **D** | Severe enough to cause damage. |

Three boundaries separate them, and `evaluation_zone` is the comparison
itself. It is deliberately blind to the quantity: the parts of the series set
boundaries on shaft displacement, housing velocity or housing acceleration, and
the grading is the same in all three. A magnitude exactly on a boundary is the
top of the zone below it, which is how the tables print their limits.

```python
from phonometry import vibration

# Three boundaries from whichever part of ISO 20816 applies, in mm/s r.m.s.
zones = vibration.ZoneBoundaries(2.8, 7.1, 11.2)
print(vibration.evaluation_zone([0.9, 3.4, 8.0, 20.0], zones))   # ['A' 'B' 'C' 'D']
print(vibration.evaluation_zone(2.8, zones))                     # A, the limit of A
```

The boundaries have to rise through the zones, and `ZoneBoundaries` refuses a
set that does not: out of order they would grade a good machine as a bad one
and say nothing.

## 3. The criterion is a curve, not a number

Velocity carries a severity criterion over a wide speed range, which is why the
tables are written in it. But a single velocity limit *regardless of frequency*
allows unacceptable displacement at low frequency and unacceptable acceleration
at high frequency (6.3.2.1). So the criterion of Figure 9 is flat only between
two corner frequencies, and slopes outside them. Annex C.2 writes it as
Formula (C.1), and `allowable_velocity` is that formula:

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/machine_vibration_zones_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/machine_vibration_zones.svg" alt="Log-log plot of allowable root-mean-square velocity in millimetres per second against frequency from 2 to 3000 hertz. Three curves rise together from the left, flatten between the corner frequencies at 10 hertz and 1 kilohertz, and fall together to the right; they are the limits of zones A, B and C, at 1.12, 2.87 and 7.17 millimetres per second on the plateau. The four bands they divide the plane into are shaded and labelled: zone A newly commissioned at the bottom, then zone B unrestricted operation, zone C limited operation, and zone D damage filling the top. Two dashed vertical lines mark the corner frequencies, labelled f sub x and f sub y" width="92%"></picture>

*One curve, moved onto the three boundaries by the factor `Zbound`. Below the
lower corner the criterion holds displacement constant, above the upper one it
holds acceleration constant.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import vibration

freq = np.logspace(np.log10(2.0), np.log10(3000.0), 600)
common = dict(constant_velocity_mm_s=1.12, corner_low_hz=10.0,
              corner_high_hz=1000.0)
for zone, factor in vibration.ZONE_LIMIT_FACTORS.items():
    plt.plot(freq, vibration.allowable_velocity(freq, zone_factor=factor,
                                                **common),
             label=f"limit of zone {zone}")
plt.xscale("log"); plt.yscale("log"); plt.legend()
plt.xlabel("Frequency [Hz]"); plt.ylabel("Allowable r.m.s. velocity (mm/s)")
plt.show()
```

</details>

The factor `Zbound` moves the one curve onto the three limits, and Annex C.2
prints what it takes: 1 for the limit of zone A, **2.56** for zone B and
**6.4** for zone C. Those are 1.6 squared and 1.6 to the fourth, near enough,
and the ladder of Table C.1 steps by about 1.6 as well, so the three limits
land close to its rungs without falling on them.

```python
from phonometry import vibration

common = dict(constant_velocity_mm_s=1.12, corner_low_hz=10.0,
              corner_high_hz=1000.0)
for zone, factor in vibration.ZONE_LIMIT_FACTORS.items():
    limit = vibration.allowable_velocity(100.0, zone_factor=factor, **common)
    print(zone, round(float(limit), 2))       # A 1.12 / B 2.87 / C 7.17
```

The corner frequencies and the exponents belong to the machine, and the
specific parts state them. The defaults of 1 are the physical reading of the
shape rather than a value taken from anywhere: constant displacement below the
lower corner, constant acceleration above the upper one, both of which fall at
6 dB per octave.

## 4. A change is a vector

Criterion I judges a magnitude. **Criterion II** (6.3.3) judges a *change* from
an established baseline, and it exists because a machine can go wrong inside
zone B: a significant increase or decrease can occur, and require action, long
before zone C is reached.

The trap is that a broad-band magnitude cannot express a change. Annex D makes
the point with a machine whose vibration fell from 3 mm/s to 2.5 mm/s while its
phase swung from 40° to 180°:

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/machine_vector_change_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/machine_vector_change.svg" alt="Polar diagram of vibration magnitude against phase. A solid vector runs from the origin to 3 millimetres per second at 40 degrees, the initial state; a second solid vector runs to 2.5 millimetres per second at 180 degrees, the final state; and a dashed line joins the two tips, the change. The dashed line is by far the longest of the three. The title reads that the magnitude changed by minus 0.5 millimetres per second while the vector changed by 5.17" width="70%"></picture>

*The magnitude fell by half a millimetre per second. The vibration moved by
5.2, ten times as much.*

```python
from phonometry import vibration

change = vibration.vibration_vector_change(3.0, 40.0, 2.5, 180.0)
print(round(change.magnitude, 2))          # 5.17 mm/s: the real change
print(round(change.magnitude_change, 2))   # -0.5 mm/s: what magnitudes say
```

This is why phase belongs in a trend record. A component that swings while its
contribution falls changes the machine, and a magnitude-only criterion sees a
small improvement.

## 5. Where the numbers come from

For a machine covered by a part of the series, from that part. For one that is
not, and where no experience is available, Annex C.1 offers Table C.1: a ladder
of preferred magnitudes and the range each boundary is typically drawn from,
with small machines (electric motors up to 15 kW) at the low end and large ones
on flexible supports at the high end.

```python
from phonometry import vibration

print(vibration.TYPICAL_ZONE_BOUNDARY_RANGES_MM_S["B/C"])   # (1.8, 9.3)
print(vibration.TYPICAL_BOUNDARY_LADDER_MM_S[:5])           # the first rungs
```

The ranges overlap on purpose: a large machine's A/B boundary can sit above a
small machine's B/C one, which is what makes them ranges rather than limits.
They are a starting point for agreement between supplier and customer, and the
standard says plainly that the values assigned to zone boundaries are not
themselves intended to serve as acceptance specifications (6.3.2.5).

## Covered and not covered

Covered: the four evaluation zones and the grading, the frequency-shaped
criterion of Figure 9 with the zone factors of Annex C.2, the typical ranges
and ladder of Table C.1, and the vector reading of a change from Annex D.

Not covered: the numeric zone boundaries of the machine-specific parts, which
this page takes as inputs; the ALARM and TRIP settings of 6.4, which the
standard describes without numbers and leaves to the parts; and the shaft
criterion of Figure 10, whose values Annex C.1 declines to give for machines no
part covers, on the grounds that such machines are not normally fitted with
shaft transducers.

## See also

- [Machine fault frequencies](machine-diagnostics.md): where a fault would
  show, once the severity says something is worth looking for.
- [Mechanical mobility and the FRF family (ISO 7626-1)](../structural/mechanical-mobility.md):
  the frequency-response vocabulary behind the housing response that a
  non-rotating measurement reads.
- [Human vibration exposure](../human/human-vibration.md): the same
  motion judged against a person rather than against a machine.

## References

- International Organization for Standardization. (2016). *Mechanical
  vibration — Measurement and evaluation of machine vibration — Part 1:
  General guidelines* (ISO 20816-1:2016).
  [iso.org catalogue](https://www.iso.org/standard/63180.html).
  The evaluation zones (6.3.2.3), the frequency-shaped criterion (6.3.2.1 and
  Formula (C.1)), the typical boundary ranges of Table C.1 and the vector
  analysis of a change (Annex D). This first edition cancelled and replaced
  ISO 7919-1:1996 and ISO 10816-1:1995.

## Standards

ISO 20816-1:2016, *Mechanical vibration — Measurement and evaluation of machine
vibration — Part 1: General guidelines*: the measurement positions of 4.4, the
vibration severity of 4.3.3, the four evaluation zones of 6.3.2.3, Criterion II
of 6.3.3, the velocity criterion of Figure 9 as Formula (C.1) with the zone
factors of Annex C.2, the typical boundary ranges of Table C.1, and the vector
change of Annex D.
