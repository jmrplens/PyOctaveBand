← [Documentation index](../../README.md)

# Emission Sound Pressure at a Work Station (ISO 11200 group)

The sound power level says how much noise a machine makes. It is the right
number for comparing machines and for feeding a room prediction, and it is the
wrong number for the person standing at the machine. What that person is
exposed to is the **emission sound pressure level**: the level at the work
station, with the background noise taken out and the room's reflections taken
out, so that what is left belongs to the machine. It is the number a datasheet
prints, and the [ISO 4871 declaration](sound-power.md) carries it beside
$L_{W\mathrm{A}}$.

Five standards determine it, and they differ in exactly one thing: how they get
rid of the room. ISO 11201 qualifies the environment until there is no room to
get rid of; ISO 11202 corrects for it approximately, by two routes; ISO 11204
corrects for it accurately; ISO 11203 derives the level from the sound power
level, so nothing is measured at the work station itself; ISO 11205 uses intensity and is not
implemented here.

## 1. One law, printed three times

$$
L_p = L'_p - K_1 - K_3
$$

$L'_p$ is what the meter read, $K_1$ removes the background noise and $K_3$
removes the reflections the room sent back. It is Equation (7) of ISO 11201,
(10) of ISO 11202 and (9) of ISO 11204. ISO 11201 prints it without the $K_3$
term, not because the term is absent but because its environment is qualified
so that the term is negligible.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/workstation_emission_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/workstation_emission.svg" alt="Two panels. On the left, the local environmental correction K3 against the dimensionless ratio z on a logarithmic axis from 0.05 to 3: a curve flat at 7 decibels up to z = 0.2, then falling as minus ten times the logarithm of z until it reaches zero at z = 1 and stays there. A dashed horizontal line marks the 4 decibel boundary between accuracy grade 2 and grade 3, with the region left of it shaded pink and the region right of it shaded green, and an annotation points at the corner to say the 7 decibel cap is minus ten times the logarithm of 0.2 to a tenth of a decibel. On the right, three bars for the worked example of a work station 1.6 metres from the source: a measured level of 76.9 decibels, a room correction of minus 3.7 floating between the other two, and the emission level of 73.2 that is left." width="100%"></picture>

A peak level takes **no correction at all**: ISO 11202 clause 8 and ISO 11204
clause 7 both say so, because both corrections are derived from mean-square
pressures and neither has a meaning for a single largest excursion.

## 2. The background, and where a determination stops being one

$$
K_1 = -10 \lg \left( 1 - 10^{-0.1 \Delta L} \right), \qquad
\Delta L = L'_p - L_p(\mathrm{B})
$$

Past 15 dB of margin the background is negligible and $K_1$ is zero. Below 6 dB
(grade 2) or 3 dB (grade 3) the correction is held at its value there and the
level becomes an upper bound; the second return value says so.

```python
from phonometry import emission

k1, upper_bound = emission.background_noise_correction_at_workstation(79.0, 70.0)
print(round(k1, 1), upper_bound)          # 0.6 False
```

## 3. The room, and the one ratio it comes down to

$$
K_3 = \begin{cases}
  7\ \mathrm{dB}, & z \le 0.2 \\
  -10 \lg z\ \mathrm{dB}, & 0.2 < z \le 1 \\
  0\ \mathrm{dB}, & z > 1
\end{cases}
$$

The 7 dB cap is the curve's own value rounded, not a separate rule:
$-10 \lg 0.2$ is 6.99 dB. With no directivity the correction follows the room's
own $K_2$ exactly, but only as far as that cap: a room with $K_2 = 10$ dB still
gets $K_3 = 7$ dB. Two roads reach $z$ and they are the same road, which
ISO 11204 A.1.2 states and the algebra confirms under the ISO 3744 definition
of $K_2$:

```python
import math
from phonometry import emission

absorption, surface = 47.0, 16.0
k2 = 10.0 * math.log10(1.0 + 4.0 * surface / absorption)

by_k2 = emission.environmental_ratio_from_k2(k2)
by_area = emission.environmental_ratio_from_absorption(absorption, surface)
print(round(by_k2, 9) == round(by_area, 9))     # True
```

## 4. The worked example, end to end

ISO 11200:2014 Annex B Table B.2: a machine with a clearly identifiable
dominating source in an 11 m by 8 m by 4 m assembly workshop of 1.2 s
reverberation time, the work station 1.6 m from that source.

```python
import math

import numpy as np
from phonometry import emission

surface = 2.0 * math.pi * 1.6**2
absorption = 0.16 * (11.0 * 8.0 * 4.0) / 1.2
k3 = emission.local_environmental_correction(
    emission.environmental_ratio_from_absorption(absorption, surface)
)

readings = np.array([77.5, 76.0, 77.2, 77.7, 75.9])
measured = 10.0 * np.log10(np.mean(np.power(10.0, readings / 10.0)))
level = emission.emission_sound_pressure_level(measured, local_correction_db=k3)
sigma = emission.total_standard_deviation(1.5, 1.0)

print(round(k3, 1), round(float(measured), 1), round(float(level), 1))
print(round(sigma, 1), round(emission.emission_expanded_uncertainty(sigma), 1))
print(emission.grade_from_local_correction(k3))
# 3.7 76.9 73.2
# 1.8 2.9
# engineering
```

Every one of those numbers is printed in Table B.2. Three gates decide the
grade and the worst wins: a class 2 meter makes it grade 3 whatever else is
true, the background must clear 6 dB, and $K_3$ must not exceed 4 dB.

## 5. A cycle of operating periods

$$
L_p = 10 \lg \left[ \frac{1}{T} \sum_{i=1}^{N} T_i\, 10^{0.1 L_{p,T_i}} \right]
$$

```python
from phonometry import emission

print(round(emission.subinterval_level([80.0, 90.0], [10.0, 1.0]), 1))    # 82.6
```

Ten seconds at 80 dB carry the same energy as one second at 90, so the cycle
lands at 82.6 dB; counting the two states equally would give 87.4.

## 6. Where the standard argues with itself

The two case studies of Annex B compute the same standard deviation two
different ways. Equation (C.1) is the sample standard deviation with
$1/(N-1)$; Table B.3 agrees and Table B.1 divides by $N$. With the figure the
equation gives, Table B.1's own expanded uncertainty would be 2.5 dB rather
than the 2.4 dB it prints. The library follows the equation, and
[the errata register](../../ERRATA.md) records the rest.

## Standards

ISO 11200:2014, *Guidelines for the use of basic standards for the
determination of emission sound pressure levels at a work station and at other
specified positions*: the selection guide, and the four worked case studies of
Annex B.

ISO 11201:2010, the free-field method; ISO 11202:2010, the approximate
environmental corrections of methods A.1 and A.2; ISO 11203:1995, the level
derived from the sound power level; ISO 11204:2010, the accurate environmental
corrections.
