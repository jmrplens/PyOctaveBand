← [Documentation index](../../README.md)

# HVAC noise the German way (VDI 2081)

[Duct-borne noise](duct-path.md) answers the
same question this page does, and answers it differently. That page is the
Anglo-American route, built on Long's Table 14.9 and the ASHRAE scaling law.
This one is **VDI 2081**, the German guideline, which arrives at the level in
the room by a different set of models and anchors them on a worked example of
its own.

Two methods for one question is not duplication. Each carries its own
arguments, its own tables and its own worked sheet, and where they disagree
the disagreement is worth knowing about. The library keeps them apart with
`model=`, so a calculation cannot end up half in one and half in the other.

## 1. The installation, and where each number comes from

Part 2 is a table of twenty numbered elements. It is easier to read as a
place.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_vdi2081_sheet_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_vdi2081_sheet.svg" alt="Section through a plant room and an office. In the plant room a supply fan, element 1, delivering 16 000 cubic metres per hour against 600 pascals. A duct leaves it and runs left to right through four boxes: element 2, a splitter silencer of five baffles; element 3, a branch taking 0.30 of 1.08 square metres; element 5, four metres of 0.5 by 0.4 metre straight duct; and element 14, a 160 millimetre round bend. It ends at element 19, two diffusers, and drops through the ceiling of room 102, where a standing listener is 1.5 metres from the outlet. A strip along the bottom names the equation or table each element's number is obtained from." width="100%"></picture>

*Every box is one row of Table 1. The strip under the drawing is the part a
table of results cannot show: which equation, table or measurement each number
comes from.*

The shape of the calculation is the same as any duct-noise sheet: a source, a
run that takes level out and puts some back, and a room that turns sound power
into sound pressure. What differs is every model inside it.

## 2. The source: a fan described by its assembly

ASHRAE describes a fan by its **type** and reads a row of band constants for
it. VDI 2081 describes it by its **assembly**, and gets the spectrum from a
formula. Section 4.3 gives the overall level as

$$
L_{W4} = L_\mathrm{WSM} + 10 \lg \dot{V} + 20 \lg \Delta p_\mathrm{t}
$$

Equation (13), with the representative specific sound power level
$L_\mathrm{WSM}$ of the assembly: 34 dB for a radial fan with rearwards curved
blades, 36 dB for a cylindrical rotor with forwards curved blades, 42 dB for an
axial fan with a downstream diffuser. The shape then comes from Equation (15),
one parabola in the logarithm of the Strouhal number that each assembly moves
along by its own $c_3$.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vdi2081_fan_assemblies_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vdi2081_fan_assemblies.svg" alt="Two panels. On the left, octave-band sound power level against frequency for three fan assemblies at the same duty point: radial with rearwards curved blades falling from 90 to 60 decibels, cylindrical rotor with forwards curved blades about three decibels above it, and axial with a downstream diffuser far above both, peaking near 250 hertz at 99 decibels and reaching 106 decibels overall. On the right, the band correction of Equation 15 as a continuous parabola for each of the three spectral parameters, with the eight octave markers of the sheet sitting on the curves." width="100%"></picture>

*The same air, the same pressure rise, and twelve more decibels overall for
choosing an axial machine over a radial one.*

```python
from phonometry import noise_control

for assembly in ("rr", "t", "am"):
    fan = noise_control.fan_sound_power(
        16000 / 3600,
        model="vdi2081",
        fan_total_pressure_pa=600.0,
        assembly=assembly,
        fan_speed_rpm=1250.0,
    )
    print(assembly, [round(float(v), 1) for v in fan.values[:3]])
```

Two traps live in that call.

**The pressure is the total pressure rise, not the static pressure.** The
ASHRAE law scales the static pressure; VDI 2081 scales the total. They are
different quantities, and confusing them is worth twenty times the logarithm
of their ratio. Each model therefore takes only the argument its own standard
is written on, declared through `typing.overload`, so the two cannot be
swapped by accident.

**The Strouhal number carries no impeller diameter.** It cancels between the
tip speed and the impeller circumference, so $St = 60 f / (\pi n)$ depends on
the running speed alone. A nomogram that asks for the impeller size is
answering a different question.

## 3. What the run takes out

Four models, one per element kind, each with the guideline's own table behind
it.

- **Straight duct** (Section 6.1, Table 5): decibels per metre by duct size,
  through `unlined_rectangular_duct_attenuation` and
  `unlined_circular_duct_attenuation` with `model="vdi2081"`. A rectangular
  duct of sheet steel takes far more out at 63 Hz than a round one, because
  its walls are the thing that gives.
- **Bend** (Section 6.2, Table 7): keyed on the bend's own size, through
  `elbow_insertion_loss`. The table is printed once, for a 1250 mm side, and
  carried along the frequency axis for every other size.
- **Change of section** (Section 6.3, Figure 26): the reflection at a sudden
  step, through `section_change_loss`. It is one expression,
  $10 \lg (r+1)^2/(4r)$ in the area ratio $r = S_1/S_2$, and two rules about
  when it applies: a sudden reduction reflects in every band, a sudden
  increase only below the limit frequency of the duct it arrives through.
  VDI 3733 recommends taking no more than 5 dB from it, because the printed
  value is only reached when the duct is anechoically terminated at both ends,
  and a gradual change through a long tapered adapter reflects nothing worth
  counting.
- **Branch** (Section 6.4, Equation (35)): the share of the cross section the
  branch takes, `10 lg (S1 / sum Si)`, through `split_loss`. It is written on
  area and not on flow, and the same page says the reduction does not depend
  on frequency.

```python
from phonometry import noise_control
import numpy as np

bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
run = noise_control.unlined_rectangular_duct_attenuation(
    bands, 0.500, 0.400, 4.000, model="vdi2081"
)
print([round(float(v), 2) for v in run.values])
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vdi2081_section_change_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vdi2081_section_change.svg" alt="Two panels. On the left, the reflection loss against the section ratio: a dashed grey curve is the printed expression, symmetric about a ratio of one where it is zero and rising to 9 decibels at the ends, and a solid blue curve is the same thing held at the 5 decibel ceiling, which it reaches at a ratio of 10.55 and again at 0.095. On the right, the same 0.88 decibel loss under the two frequency rules: a solid line flat across all eight octaves for a sudden reduction, and a dashed line that holds the value at 63 and 125 hertz and drops to zero above the limit frequency of 214 hertz for a sudden increase" width="100%"></picture>

*The expression is symmetric: a duct that widens by a factor reflects what the
same duct narrowing by it does. What differs is when.*

```python
from phonometry import noise_control
import numpy as np

bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
step = noise_control.section_change_loss(bands, 0.5, 0.2)
print(round(float(step.values[0]), 2))   # 0.88 dB, in every band
```

A reduction needs nothing else. The frequency column of Figure 26 reads "no
effect" for it, so no limit frequency enters the answer and neither does the
duct's shape or its size. An increase does need them, and a rectangular duct
has to be given its largest side, because that is the one thing its area does
not tell you.

A splitter silencer is not one of these: its insertion loss comes from the
maker's measurement, to ISO 7235, and the guideline says so.
`splitter_silencer_insertion_loss` is the estimate for when there is none.

## 4. What the run puts back

Air moving through a duct makes noise, and past the silencer there is often
nothing else left to hear. VDI 2081 gives it in closed form.

- **A straight run** (Section 5.2.1, Equations (16) and (17) with Figure 16)
  and **a bend or a branch** (Section 5.2.2, Equation (18) with Figures 17 and
  18), through `flow_noise_straight_duct` and `flow_noise_bend`. The second
  pair is written on a Strouhal number built from the element's own diameter
  and the flow speed through it, and both figures state that they hold only
  above $St = 1$, so the library gives no level below it rather than
  extrapolating a curve the guideline does not draw.
- **A splitter silencer's own noise** (Equation (49)), through
  `silencer_self_noise`, which depends on the speed in the gaps between the
  baffles and on the pressure drop across them. It is the reason a silencer
  has a best size: make the gaps narrower and the insertion loss rises, but so
  does the speed through them, and past a point the silencer is louder than
  what it removed.
- **The outlet itself**, through `diffuser_sound_power`.

## 5. The room step, and the two areas that are not the same number

Equation (36) turns the sound power arriving at the outlet into the level a
listener hears:

$$
L_W - L_p = -10 \lg \left( \frac{Q}{4 \pi r^{2}} + \frac{4}{A} \right)
$$

Two things about it are easy to get wrong.

**$A$ is the equivalent absorption area, not the room constant.** The room
constant is $R = A/(1 - \bar\alpha)$, and both are areas in square metres, both
positive, so substituting one for the other is a silent mistake: the number
comes out plausible and wrong. `room_effect` and `room.steady_state_spl`
therefore take `absorption_area=` and `room_constant=` as separate arguments,
and exactly one of them may be given.

**$Q$ moves with frequency.** A ceiling diffuser is more directional the
shorter the wavelength, and the guideline reads $Q$ off a chart against
frequency rather than assuming a half space. Both functions take a directivity
that varies across the bands.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vdi2081_room_step_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vdi2081_room_step.svg" alt="Two panels. On the left, the difference between sound power level and sound pressure level against frequency for the worked example's ceiling diffuser: it falls from 5.6 decibels at 63 hertz to 3.4 at 8 kilohertz as the outlet's directivity factor rises, while a half space with a directivity factor of 2 would give a flat 5.7 decibels in every band. On the right, the same difference against distance from the outlet for the equivalent absorption area of 20 square metres and for the two room constants that a mean absorption coefficient of 0.15 and of 0.4 turn it into, with the listener's 1.5 metres marked." width="100%"></picture>

*Left: the single 5,7 dB the sheet prints beside the row is not any of the
eight band values, because it is the same room with the directivity of a half
space. Right: reading the room constant into the absorption-area argument
costs about a decibel in this room, and more in a livelier one.*

```python
from phonometry import noise_control
import numpy as np

directivity = np.array([2.1, 2.4, 3.0, 4.0, 5.5, 6.7, 7.0, 7.2])
shaped = noise_control.room_effect(1.5, absorption_area=20.0, directivity=directivity)
print([round(float(v), 1) for v in np.asarray(shaped)])
print(round(float(noise_control.room_effect(1.5, absorption_area=20.0)), 1))
```

## 6. The sheet, end to end

Part 2 exists to anchor Part 1: one supply air network, worked element by
element, with every intermediate quantity printed. That makes it an oracle of
a kind the ASHRAE side of the module does not have, and for a genuinely
different model rather than a restatement of the same one.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vdi2081_chain_cascade_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vdi2081_chain_cascade.svg" alt="Octave-band levels at five stages of the worked example: at the fan, near 90 decibels and falling smoothly; past the splitter silencer, which takes forty decibels out of the middle bands and leaves a shelf near 45 decibels above 1 kilohertz; past the branch, a few decibels lower again; the sound power the two diffusers put into the room; and the sound pressure level at the listener, from 48 decibels at 63 hertz to 29 at 8 kilohertz. The A-weighted total of each stage is given in the legend, falling from 86 to 40 decibels A." width="100%"></picture>

*The silencer does almost all of the work, and what it leaves is a shelf: past
1 kHz the level no longer follows the fan at all, because what is heard there
is the noise the air makes on its way past.*

The A-weighted level in room 102 comes out at 40,0 dB(A) against the 40,0 the
sheet prints, and the unweighted total at 51,4 against 51,4. Forty-six
conformance rows hold each element of the chain to the tenth of a decibel the
table is printed to, band by band rather than by the sum: a sum is blind to
the shape, and any pair of compensating errors passes it.

Whether the room is quiet enough is then Part 2 Section 1.1, which turns an
A-weighted requirement into a limit for each octave:

```python
from phonometry import noise_control

limits = noise_control.octave_band_limits(35.0)
print([round(float(v)) for v in limits.values])
```

## 7. What the guideline itself gets wrong

Five defects are recorded in [the errata
register](../../ERRATA.md), all verified against the printed
page.

- The symbol list under Equation (36) sends the reader looking for $A$ in
  Equation (36) itself, which is where they already are.
- The English column of Section 6.7.3 calls a hemispherical radiation
  spherical, and the German column beside it does not.
- The English column of Section 6.4 says a junction's level reduction depends
  on the frequency; the German column beside it says it does not, and Figure 27
  on the same page has no frequency axis.
- Table 1 of Part 2 prints a hydraulic diameter for element 2 that is not the
  one it computes with.
- Table 1 of Part 2 credits the splitter-silencer coefficients of element 2 to
  Section 7.3.2 of Part 1, which is the airborne sound insulation of a building
  component. They are printed in Section 7.2.3.2.

The editions implemented here are Part 1:2001-07 and Part 2:2005-05. Both are
superseded, by the 2022 editions, and neither successor is held; the pair in
hand is self-consistent, because Part 2:2005 was written against Part 1:2001
and every cross-reference in its tables resolves there.

## Covered and not covered

Covered: The fan of Section 4.3 through `fan_sound_power(model="vdi2081")`, the duct,
bend, section-change and branch attenuation of Section 6, the flow noise of
Equations (16) and (17), the splitter silencer and its self-noise, the end
reflection of Section 6.6, the room step of Equation (36) and the assessment
curve of Part 2 Section 1.1, all against the worked sheet of Part 2 Table 1.

Not covered: The air-handling unit and the outdoor-propagation chapters the
superseding editions added, which are not held here; the room acoustics VDI
2081 leaves to VDI 2569; and the vibration isolation of the plant, which
Section 7.6 of Blatt 1 does cover and this library does not.

## See also

- [Duct-borne noise: fan to room](duct-path.md):
  the same question by the Anglo-American route, and the plane-wave limit that
  bounds both.
- [Silencers and mufflers](silencers.md): what
  a splitter silencer is doing, and how its insertion loss is measured.
- [Room-to-room noise](room-to-room.md): the
  path that does not go down a duct.

## References

- Verein Deutscher Ingenieure. (2001). *Geräuscherzeugung und Lärmminderung in
  Raumlufttechnischen Anlagen* (VDI 2081 Blatt 1:2001-07).
  The method: the fan of Section 4.3 with Equations (13) and (15), the duct
  attenuation of Section 6 with Tables 5 and 7, the flow noise of Equations
  (16) and (17), the splitter silencer and its self-noise of Equation (49),
  the end reflection of Section 6.6 and the room step of Equation (36).
  Superseded by Blatt 1:2022-04, which is not held here.
- Verein Deutscher Ingenieure. (2005). *Geräuscherzeugung und Lärmminderung in
  Raumlufttechnischen Anlagen — Beispiele* (VDI 2081 Blatt 2:2005-05).
  The oracle: one supply air network worked element by element in Table 1,
  with every intermediate spectrum printed, and the assessment curve of
  Section 1.1. Superseded by Blatt 2:2022-10, which is not held here.

## Standards

VDI 2081 Blatt 1:2001-07, *Geräuscherzeugung und Lärmminderung in
Raumlufttechnischen Anlagen*: the fan of Section 4.3, the duct, bend,
section-change and branch attenuation of Section 6, the flow noise of
Equations (16) and (17), the splitter silencer and its self-noise of Equation
(49), the end reflection of Section 6.6 and the room step of Equation (36).

VDI 2081 Blatt 2:2005-05, *Geräuscherzeugung und Lärmminderung in
Raumlufttechnischen Anlagen — Beispiele*: the worked supply air network of
Table 1 and the assessment curve of Section 1.1.
