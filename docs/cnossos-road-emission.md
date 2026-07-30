← [Documentation index](README.md)

# CNOSSOS-EU road traffic source emission

Every strategic noise map drawn in the European Union since 2021 starts from the
same two numbers per octave band: how loud a vehicle is, and how many of them go
past per hour. **Annex II of Directive 2002/49/EC**, as replaced by the CNOSSOS-EU
methods, turns those into a **directional sound power per metre of source line**,
which the propagation stage then carries to the receiver.

This guide covers the road source, section 2.2 of Annex II with the coefficient
database of Appendix F. The railway source (2.3, Appendix G) and the CNOSSOS-EU
propagation model (2.5) are separate.

## 1. Which text is implemented

Annex II is a moving target and getting the layering wrong is the easiest way to
ship a wrong table. Three instruments are in play:

| Instrument | What it does to the road source |
| :--- | :--- |
| Commission Directive (EU) **2015/996** | Replaces the whole of Annex II. Supplies formulae 2.2.1 to 2.2.20 and Tables F-1 to F-4. |
| **Corrigendum** of OJ L 5, 10.1.2018 | Corrects 2.2.1: the sound powers are calculated "for each octave band from **63 Hz to 8 kHz**", not the 125 Hz to 4 kHz the original text printed. Appendix F always covered 63 Hz to 8 kHz, so the uncorrected text contradicted its own tables. |
| Commission Delegated Directive (EU) **2021/1226** | Replaces **Table F-1** and **Table F-4** in their entirety, merges the former 4a and 4b rows of Table F-4 into one, and prints the octave-band A-weighting to be used in 2.5.5. |

The library implements the consolidated result. Tables F-2 (studded tyres) and
F-3 (junctions) have never been amended. The 2021 amendment is not cosmetic: the
current road source is about **2,5 to 3,5 dB(A) louder** than the one published
in 2015, so any comparison with pre-2021 literature carries that offset.

## 2. The source and the source line

Each vehicle is one **point source 0,05 m above the road surface**, radiating
uniformly; the first reflection on the pavement is already inside the sound
power, which is why the method calls it a semi-free-field quantity. A traffic
flow is an incoherent **source line**, ideally one line per lane at the lane
centre.

For each vehicle category `m` and octave band `i`, formula (2.2.1) turns a
single-vehicle sound power into a power per metre of line:

$$
L'_{W,\mathrm{eq,line},i,m} = L_{W,i,m} + 10\,\lg\!\left(\frac{Q_m}{1000\,v_m}\right),
$$

with `Qm` in vehicles per hour and `vm` in km/h. The vehicle power itself is the
energy sum of a **rolling** and a **propulsion** term (2.2.2), except for the
powered two-wheelers of category 4, which have no rolling noise at all and take
the propulsion term alone (2.2.3):

$$
L_{W,i,m} = 10\,\lg\!\left(10^{L_{WR,i,m}/10} + 10^{L_{WP,i,m}/10}\right).
$$

Table [2.2.a] fixes five modelled categories: **1** light motor vehicles, **2**
medium heavy vehicles, **3** heavy vehicles, **4a** mopeds and **4b**
motorcycles. (A sixth "open" category exists in the text but carries no
coefficients.)

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cnossos_road_emission_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cnossos_road_emission.svg" alt="Octave-band source-line power of an urban arterial: bars for the total and marker lines for the light, medium heavy, heavy and motorcycle contributions, showing light vehicles governing below 500 Hz and heavy vehicles at 1 kHz" width="88%"></picture>

*Light vehicles carry the flow, but 45 heavy vehicles per hour still take over
the mid frequencies: the per-metre spectrum shows exactly where each category
governs.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
from phonometry import (
    JunctionType, RoadSurface, RoadTraffic, RoadVehicleCategory, road_source_power,
)

traffic = [
    RoadTraffic(RoadVehicleCategory.LIGHT, 1200.0, 50.0),
    RoadTraffic(RoadVehicleCategory.MEDIUM_HEAVY, 90.0, 50.0),
    RoadTraffic(RoadVehicleCategory.HEAVY, 45.0, 50.0),
    RoadTraffic(RoadVehicleCategory.MOTORCYCLES, 60.0, 50.0),
]
result = road_source_power(
    traffic, surface=RoadSurface.THIN_LAYER_A, temperature=12.0, gradient=3.0,
    junction_distance=60.0, junction_type=JunctionType.CROSSING,
)
result.plot()
plt.show()
```

</details>

## 3. Rolling noise (2.2.4 to 2.2.10)

Rolling noise is tyre-road noise. It grows logarithmically with speed from the
reference speed `vref = 70 km/h`:

$$
L_{WR,i,m} = A_{R,i,m} + B_{R,i,m}\,\lg\!\left(\frac{v_m}{v_\mathrm{ref}}\right)
             + \Delta L_{WR,i,m},
$$

where the correction collects four independent effects (2.2.5): the road
surface, the studded tyres, the junction and the air temperature.

**Air temperature** (2.2.10) is a straight line, applied equally to all eight
bands: `dLW,temp = Km (20 °C − tau)`, with `K1 = 0,08 dB/°C` for light vehicles
and `K2 = K3 = 0,04 dB/°C` for the two heavy categories. Cold air makes tyres
harder and the road louder, so the correction is *positive* below 20 °C.

**Studded tyres** (2.2.6 to 2.2.9) apply to category 1 only. The per-tyre excess
`Dstud,i(v)` saturates below 50 km/h and above 90 km/h, and the fleet-level
correction weights it by the share of the year that studded tyres are on the
road, `ps = Qstud,ratio · Ts/12`:

$$
\Delta L_{\mathrm{studded},i} = 10\,\lg\!\left[(1 - p_s) + p_s\,10^{D_{\mathrm{stud},i}/10}\right].
$$

**Junctions** (2.2.17) add `CR,m,k · max(1 − |x|/100, 0)`, with `x` the distance
in metres to the nearest junction and `k` the junction type: 1 for a crossing
with traffic lights, 2 for a roundabout. The rolling coefficients `CR` are
*negative* (traffic near a junction is slower), the propulsion coefficients `CP`
positive (it is accelerating). Both vanish from 100 m out.

## 4. Propulsion noise (2.2.11 to 2.2.16)

Propulsion noise is the power train: engine, exhaust, transmission, intake. It
is **linear** in speed, not logarithmic, because at low speed the engine is
working hardest relative to the distance covered:

$$
L_{WP,i,m} = A_{P,i,m} + B_{P,i,m}\,\frac{v_m - v_\mathrm{ref}}{v_\mathrm{ref}}
             + \Delta L_{WP,i,m}.
$$

The two speed laws crossing is what makes the road source behave the way it
does: below the crossover the source is an engine, above it a tyre.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cnossos_road_speed_law_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/cnossos_road_speed_law.svg" alt="A-weighted single-vehicle sound power against speed from 20 to 130 km/h for light and heavy vehicles, with the rolling and propulsion components dashed and dotted and the crossover speed marked on each pair" width="88%"></picture>

*A light vehicle is tyre-dominated from about 30 km/h; a heavy vehicle only
around 60 km/h, which is why a lorry is still an engine at urban speeds.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    CNOSSOS_A_WEIGHTING, road_propulsion_noise, road_rolling_noise,
    road_vehicle_sound_power,
)

weights = np.asarray(CNOSSOS_A_WEIGHTING)
a_weighted = lambda bands: 10.0 * np.log10(np.sum(10.0 ** ((bands + weights) / 10.0)))

speeds = np.linspace(20.0, 130.0, 221)
fig, ax = plt.subplots()
for category, name in [("1", "Light vehicles (1)"), ("3", "Heavy vehicles (3)")]:
    rolling = [a_weighted(road_rolling_noise(category, v)) for v in speeds]
    propulsion = [a_weighted(road_propulsion_noise(category, v)) for v in speeds]
    total = [a_weighted(road_vehicle_sound_power(category, v)) for v in speeds]
    ax.plot(speeds, total, lw=2.4, label=f"{name} - total")
    ax.plot(speeds, rolling, ls="--", lw=1.2, label=f"{name} - rolling")
    ax.plot(speeds, propulsion, ls=":", lw=1.2, label=f"{name} - propulsion")
ax.set_xlabel("Speed v [km/h]")
ax.set_ylabel("A-weighted sound power [dB(A) re 1 pW]")
ax.legend(fontsize="small")
plt.show()
```

</details>

**Road gradient** (2.2.13 to 2.2.16) is the one correction whose published form
is genuinely asymmetric, and the asymmetry is not a transcription slip. For
light vehicles the downhill branch has **no speed factor**, while the two heavy
categories do and use different speed offsets:

| Category | Downhill (`s` below the breakpoint) | Flat band | Uphill |
| :--- | :--- | :--- | :--- |
| 1 | `(min(12 %, −s) − 6 %)/1 %` | −6 % to 2 % | `(min(12 %, s) − 2 %)/1,5 % · v/100` |
| 2 | `(min(12 %, −s) − 4 %)/0,7 % · (v − 20)/100` | −4 % to 0 % | `min(12 %, s)/1 % · v/100` |
| 3 | `(min(12 %, −s) − 4 %)/0,5 % · (v − 10)/100` | −4 % to 0 % | `min(12 %, s)/0,8 % · v/100` |

Category 4 takes no gradient correction. The slope saturates at 12 % in both
directions. Downhill noise comes from engine braking, uphill noise from load, so
both branches are positive. For a bidirectional flow, split the flow in two and
correct one half uphill and the other downhill.

## 5. The road surface (2.2.19, 2.2.20 and Table F-4)

Table F-4 gives, for each of fourteen surfaces, an octave-band coefficient
`alpha` and a speed coefficient `beta`, per vehicle category, with the speed
range over which the row is declared valid. The surface enters the two terms
differently:

$$
\Delta L_{WR,\mathrm{road},i,m} = \alpha_{i,m} + \beta_m\,\lg\!\left(\frac{v_m}{v_\mathrm{ref}}\right),
\qquad
\Delta L_{WP,\mathrm{road},i,m} = \min\{\alpha_{i,m};\,0\}.
$$

An absorbing surface reduces propulsion noise; a noisy one does not increase it.
That asymmetry is deliberate: the pavement can absorb the engine sound radiated
down onto it, but a rough texture only generates tyre noise, and tyre noise is
already the rolling term.

```python
from phonometry import RoadSurface, road_surface_coefficients

row = road_surface_coefficients(RoadSurface.TWO_LAYER_ZOAB_FINE)
row.speed_range          # (80.0, 130.0) km/h, the printed validity range
row.alpha["1"]           # alpha per octave band for light vehicles
row.beta["1"]            # -0.1, the speed coefficient
```

The reference surface of the first row is all zeros: it is the virtual average
of a dense asphalt concrete 0/11 and a stone mastic asphalt 0/11, between two
and seven years old, in representative maintenance condition, dry, with no
studded tyres, at 20 °C, on the flat. Under exactly those conditions every
correction in 2.2 vanishes identically, and the sound powers **are** the
Table F-1 coefficients `AR,i,m` and `AP,i,m`.

## 6. Handing the source to a propagation model

The emission stage produces a power per metre. Splitting that line into
equivalent point sources is, in the words of section 2.5.3, "outside the scope
of the current methodology": a point source standing for a segment of length
`dL` simply carries `L'W,eq,line,i + 10 lg(dL)`, which is arithmetic and is
offered as such.

```python
from phonometry import (
    RoadTraffic, RoadVehicleCategory, line_source_segment_power,
    predicted_receiver_level, road_source_power,
)

result = road_source_power([
    RoadTraffic(RoadVehicleCategory.LIGHT, 1000.0, 90.0),
    RoadTraffic(RoadVehicleCategory.HEAVY, 120.0, 80.0),
])
segment = line_source_segment_power(result.total_line_power, 20.0)  # a 20 m segment
levels = predicted_receiver_level(
    segment, 100.0, result.source_height, 4.0, frequencies=result.frequencies,
)
```

Note what this is and is not. CNOSSOS-EU has **its own** propagation method in
section 2.5 of Annex II, and it is not ISO 9613-2: the two differ in the ground
model, in the diffraction treatment and in the way favourable and homogeneous
conditions are combined. Chaining a CNOSSOS emission onto the ISO 9613-2
propagation of this library is a legitimate engineering estimate, but it is not
the normative CNOSSOS chain and must not be reported as one.

For the A-weighted total, use the octave-band weighting the Directive itself
prints in 2.5.5 as amended, rather than recomputing it:

```python
from phonometry import CNOSSOS_A_WEIGHTING

CNOSSOS_A_WEIGHTING   # (-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1)
result.a_weighted_line_power
```

## 7. Substituting a national database

Appendix F is called a *database*, not a table of constants, because Member
States may substitute measured national values, and because the same equations
have already been evaluated with two different coefficient sets (the 2015 one
and the 2021 one). Both coefficient objects can be replaced:

```python
from phonometry import ROAD_COEFFICIENTS, RoadEmissionCoefficients

national = RoadEmissionCoefficients(
    rolling_a={"1": (83.1, 89.2, 87.7, 93.1, 100.1, 96.7, 86.8, 76.2), ...},
    rolling_b=ROAD_COEFFICIENTS.rolling_b,
    propulsion_a=ROAD_COEFFICIENTS.propulsion_a,
    propulsion_b=ROAD_COEFFICIENTS.propulsion_b,
    studded_a=ROAD_COEFFICIENTS.studded_a,
    studded_b=ROAD_COEFFICIENTS.studded_b,
    junction_c=ROAD_COEFFICIENTS.junction_c,
    temperature_k=ROAD_COEFFICIENTS.temperature_k,
)
```

That mechanism is also what pins the implementation: the European Commission
published a test set of 4 875 road emission cases computed with the 2015
coefficients, and feeding the shipped equations that superseded database
reproduces every published band level to 0,005 dB, inside the two decimals the
test set prints.

## What this guide covers

**Covered.** Section 2.2 of Annex II to Directive 2002/49/EC in the consolidated
text: the traffic-flow line power (2.2.1), the vehicle power (2.2.2, 2.2.3),
rolling noise with its studded-tyre and temperature corrections (2.2.4 to
2.2.10), propulsion noise with the road-gradient correction (2.2.11 to 2.2.16),
the junction correction (2.2.17, 2.2.18) and the road-surface effect (2.2.19,
2.2.20), together with the whole Appendix F database, through
`road_source_power`, `road_vehicle_sound_power`, `road_rolling_noise`,
`road_propulsion_noise` and `road_surface_coefficients`.

**Not covered.** The railway source (2.3 and Appendix G), the industrial source
(2.4 and Appendix H), the aircraft source (2.6 and 2.7) and the CNOSSOS-EU
propagation method of section 2.5, which is a different model from the
ISO 9613-2 one implemented in this library. The open vehicle category 5 has no
coefficients in Appendix F and is therefore not modelled. How a source line is
split into point sources is declared out of scope by the method itself.

## See also

- [Outdoor Sound Propagation](outdoor-propagation.md): the ISO 9613-2 chain that
  carries a source power to a receiver.
- [Environmental noise levels](environmental-levels.md): the Lden and Lnight
  indicators the resulting maps are drawn for.
