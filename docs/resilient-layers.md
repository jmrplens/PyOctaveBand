← [Documentation index](README.md)

# Predicting Resilient-Layer Performance (Coverings, Floating Floors, Linings)

A resilient layer is the cheapest way to buy impact sound insulation and the
easiest to get wrong. Whether it is a carpet on a slab, a screed floating on
mineral wool or a plasterboard lining on a masonry wall, the layer is a
spring, and the only question that matters is where its resonance with the
mass it carries falls. Below that frequency it does nothing, at it the
construction is *worse* than bare, and above it the improvement climbs at a
rate the construction type decides. This guide is the prediction side of that
story, from the material data to the improvement spectrum; the measurement
side lives in
[Floor-Covering Impact Improvement (ISO 16251-1)](impact-improvement.md)
and
[Dynamic stiffness of resilient materials](dynamic-stiffness.md).

## The excitation: the tapping machine as a mass-spring-dashpot

Everything downstream depends on the force the standard tapping machine
actually injects, and that force is not a property of the machine alone. Each
hammer is a 0.5 kg mass dropped 40 mm, so it lands at
$v_0 = \sqrt{2gh} = 0.886\ \text{m/s}$, ten times a second. If the impact were
short compared with the 0.1 s between impacts, the momentum alone would fix
the force, giving the line spectrum $|F_n| = 2 m v_0 / T_i$ and the band
mean-square force $F_\text{rms}^2 = 3.9 B$ that ISO 10140 measurements on
concrete are usually reduced with.

Real floors deform. The hammer, the contact stiffness $K$ it presses into and
the floor's driving-point impedance $Z_\text{dp}$ form a
mass-spring-dashpot whose pulse splits into two cases:

- **over-critical**, $K m \ge 4 Z_\text{dp}^2$: a single positive pulse, no
  rebound. This is a lightweight walking surface (chipboard, OSB), where the
  spectrum tends to $|F_n|_\text{lower} = m v_0/T_i$;
- **under-critical**, $K m < 4 Z_\text{dp}^2$: the hammer rebounds and only the
  first positive lobe counts. This is a concrete slab or a screed, where the
  spectrum sits within 1 dB of $|F_n|_\text{upper} = 2 m v_0/T_i$ below 4 kHz
  and the short-pulse estimate is adequate.

The two limits are 6 dB apart in mean square, which is the whole reason a
tapping-machine level on a timber floor cannot be compared with one on
concrete without thinking. Above the cut-off frequency

$$
f_\text{co} = \frac{1}{2\pi}\sqrt{\frac{K}{m}}\quad\text{(under-critical)},
\qquad
f_\text{co} = \frac{1}{2\pi}\left[\frac{K}{2Z_\text{dp}} - \sqrt{\left(\frac{K}{2Z_\text{dp}}\right)^2 - \frac{K}{m}}\right]\quad\text{(over-critical)},
$$

the force falls away, and above the limiting frequency
$f_\text{limit} = Z_\text{dp}/(2\pi m)$ the hammer's own mass impedance, not
the floor, caps the injected power.

```python
import numpy as np
from phonometry import (
    hammer_impact_velocity, infinite_plate_impedance, plate_bending_stiffness,
    plate_contact_stiffness, tapping_force_spectrum,
)

print(round(hammer_impact_velocity(), 3))     # 0.886 m/s (0.5 kg dropped 40 mm)

# A 140 mm cast in-situ concrete slab: 2200 kg/m3, cL = 3800 m/s, nu = 0.2.
rho, c_l, nu, h = 2200.0, 3800.0, 0.2, 0.14
E = rho * c_l**2 * (1 - nu**2)
z = infinite_plate_impedance(plate_bending_stiffness(E, h, nu), rho * h)
k = plate_contact_stiffness(E, poisson_ratio=nu)

freqs = np.array([100, 200, 400, 800, 1600, 3150], dtype=float)
res = tapping_force_spectrum(freqs, k, z)
res.over_critical            # False: the hammer rebounds off concrete
round(res.cut_off_frequency)  # 6948 Hz, above the building acoustics range
res.peak_force               # |Fn| per band, within 1 dB of res.upper_limit
res.power_input_level        # 10 lg(Win/1 pW), rising 3 dB per doubling
res.plot()                   # the force spectrum with both asymptotes
```

## Soft floor coverings: the cut-off frequency is the design

A soft covering on a heavyweight floor changes nothing but the force input:
its mass is negligible and it barely touches the slab's loss factor or bending
stiffness. So its improvement is simply the ratio of the two force spectra,

$$
\Delta L = 20 \lg \frac{|F_n|_\text{without}}{|F_n|_\text{with}} \ \text{dB},
$$

and the whole design collapses to one number, the covering's own cut-off
frequency, set by its contact stiffness $K = E \pi r^2 / d$ with the hammer
radius $r = 15$ mm. Below $f_\text{co}$ the covering does nothing; above it
the improvement rises at 12 dB per octave. Two straight lines are therefore a
usable estimate, and thickening a homogeneous covering lowers $f_\text{co}$
and buys improvement everywhere above it.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/soft_covering_prediction_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/soft_covering_prediction.svg" alt="Predicted improvement of impact sound insulation of two soft floor coverings on a 140 mm concrete slab: a stiff PVC covering whose cut-off frequency of 2318 Hz leaves it useless across the building acoustics range, and a resiliently backed covering whose 100 Hz cut-off gives 70 dB by 5 kHz, each against its two-line 12 dB per octave estimate" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    covering_contact_stiffness, covering_improvement, infinite_plate_impedance,
    plate_bending_stiffness, plate_contact_stiffness,
)

rho, c_l, nu, h = 2200.0, 3800.0, 0.2, 0.14          # 140 mm concrete slab
E = rho * c_l**2 * (1 - nu**2)
z = infinite_plate_impedance(plate_bending_stiffness(E, h, nu), rho * h)
plate = plate_contact_stiffness(E, poisson_ratio=nu)

# The tapping machine's force is a line spectrum at multiples of 10 Hz, so the
# band improvement is the ratio of the mean-square forces over the lines in
# each band; that averaging removes the ripple of the undamped model.
bands = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                  800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000.0])
lines = np.arange(1, 700) * 10.0
edge = np.log10(2.0) / 6.0

d = 0.005
for label, modulus_over_thickness in (("PVC", 1.5e11), ("backed vinyl", 2.8e8)):
    res = covering_improvement(
        lines, covering_contact_stiffness(modulus_over_thickness * d, d),
        plate, z)
    banded = [
        10 * np.log10(np.mean(res.bare.peak_force[m] ** 2)
                      / np.mean(res.covered.peak_force[m] ** 2))
        for m in (np.abs(np.log10(lines / b)) <= edge for b in bands)
    ]
    plt.semilogx(bands, banded, marker="o", label=f"{label}: fco = "
                 f"{res.cut_off_frequency:.0f} Hz")
plt.legend()
plt.show()
```

</details>

```python
from phonometry import covering_contact_stiffness, covering_improvement

# Covering No. 2 of Hopkins Fig. 4.64: E/d = 2.8e8 N/m3, a vinyl or carpet
# with a resilient backing, 5 mm thick on the same 140 mm slab.
d = 0.005
covering = covering_contact_stiffness(2.8e8 * d, d)
res = covering_improvement([125.0, 250.0, 500.0, 1000.0], covering, k, z)
round(res.cut_off_frequency)        # 100 Hz
round(res.bare_cut_off_frequency)   # 6948 Hz, the bare slab's own cut-off
res.improvement                     # delta-L from the force ratio
res.two_line                        # 0 dB below fco, 40 lg(f/fco) above
res.plot()
```

The model treats the covering as a *linear* spring. Under the tapping
machine's high force many real coverings harden, showing two or three slopes
between 5 and 22 dB per octave rather than one, which is why a laboratory
$\Delta L$ remains the reference once a specimen exists.

## Floating floors: three laws above one resonance

A floating floor is a rigid walking surface on a resilient layer, and its
resonance follows from the two numbers known at the drawing stage, the mass
per unit area $m'$ of the slab and the EN 29052-1 dynamic stiffness per unit
area $s'$ of the layer:

$$
f_0 = 160 \sqrt{\frac{s'}{m'}}\ \text{Hz}
\qquad (s'\ \text{in MN/m}^3,\ m'\ \text{in kg/m}^2),
$$

where 160 is the standard's rounding of $1000/2\pi$. Above it, which law
applies is a question about damping, not about the layer:

| Model | Law | When |
| --- | --- | --- |
| `"cremer"` | $\Delta L = 40 \lg (f/f_0)$ | Cremer's infinite-plate result; asphalt screeds and dry floating floors, whose internal losses are high enough to behave as infinite plates (ISO 12354-2 Formula C.3) |
| `"en12354"` | $\Delta L = 30 \lg (f/f_0)$ | sand-cement and calcium-sulfate screeds, whose low loss factor makes them finite plates with a reverberant bending field (Formula C.1) |
| `"cremer_hammer"` | $40 \lg (f/f_0) + 10 \lg[1 + (f/f_\text{limit})^2]$ | a lightweight walking surface, where the hammer impedance is no longer negligible; tends to 18 dB per octave |

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/floating_floor_prediction_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/floating_floor_prediction.svg" alt="Improvement of impact sound insulation of a 35 kg per square metre floating screed on an 8 meganewton per cubic metre resilient layer: nothing below the 52.8 Hz mass-spring resonance, then the 30 lg law of EN 12354-2 reaching 59 dB at 5 kHz against the steeper 40 lg law of Cremer and the 18 dB per octave branch that includes the tapping-machine hammer impedance" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import (
    floating_floor_improvement_spectrum, floating_floor_resonance_frequency,
)

# The worked floating floor of ISO 12354-2:2017 Annex G.
f0 = floating_floor_resonance_frequency(8.0e6, 73.5)     # 52.8 Hz
freqs = np.logspace(np.log10(40.0), np.log10(5000.0), 400)
for model, kwargs in (("en12354", {}), ("cremer", {}),
                      ("cremer_hammer", {"limiting_frequency": 521.0})):
    res = floating_floor_improvement_spectrum(
        freqs, resonance_frequency=f0, model=model, **kwargs)
    plt.semilogx(freqs, res.improvement, label=model)
plt.axvline(f0, ls=":")
plt.legend()
plt.show()
```

</details>

```python
from phonometry import (
    combined_dynamic_stiffness, double_floating_floor_resonances,
    floating_floor_improvement_spectrum, floating_floor_resonance_frequency,
    weighted_floating_floor_improvement,
)

# 35 mm screed, m' = 73.5 kg/m2, on a resilient layer of s' = 8 MN/m3.
f0 = floating_floor_resonance_frequency(8.0e6, 73.5)
round(f0, 1)                                                    # 52.8 Hz

bands = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
         800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000.0]
res = floating_floor_improvement_spectrum(
    bands, resonance_frequency=f0, mass_per_area=73.5, dynamic_stiffness=8.0e6)
res.improvement          # 0.0, 2.3, 5.4, 8.3, 11.2 ... 59.3 dB
round(res.delta_lw, 1)   # 32.2 dB, the Formula (C.4) weighted improvement
res.plot()

# Two resilient layers act as springs in series (Formula C.6), which drops the
# resonance by sqrt(2) and buys about 4.5 dB everywhere above it.
round(floating_floor_resonance_frequency(
    combined_dynamic_stiffness([8.0e6, 8.0e6]), 73.5), 1)       # 37.3 Hz

# One floating floor on top of another has two resonances instead of one; the
# adverse dip disappears, but the steep rise only starts above the higher.
double_floating_floor_resonances(7.25e6, 12.78, 7.25e6, 12.78)  # (74, 194) Hz
```

The weighted single numbers come straight from the same two inputs, so a
floating floor can be sized without ever drawing a spectrum:
$\Delta L_w = 13\lg m' - 14.2\lg s' + 20.8$ dB for a sand-cement screed
(Formula C.4) and the steeper Formula (C.5) fit for asphalt and dry floors.
Heavier slabs and softer layers rate better, and both estimates are
deliberately on the safe side.

A floating floor supported on **discrete mounts** rather than a continuous
layer is a different problem: the walking surface carries a reverberant
bending-wave field and all the transmission goes through the mounts, which
Vér's two-subsystem SEA model turns into a 30 dB per decade rise rather than
40. Fewer mounts, a thicker walking surface or more internal damping all raise
it.

```python
from phonometry import resilient_mount_improvement

# 50 mm concrete walking surface on 4 mounts per m2 of 2 MN/m stiffness.
resilient_mount_improvement(
    [125.0, 250.0, 500.0, 1000.0], impedance=3.8e5, mass_per_area=115.0,
    loss_factor=0.02, mount_stiffness=2.0e6, mount_density=4.0)
```

## Wall linings: Annex D reads the rating off the resonance

A lining on a wall is the same spring problem with the sign reversed: it can
*reduce* the sound insulation, and ISO 12354-1 Annex D predicts by how much
from the resonance frequency alone. Which formula gives $f_0$ depends on how
the layer is fixed:

$$
f_0 = \frac{1}{2\pi}\sqrt{s'\left(\frac{1}{m_1'} + \frac{1}{m_2'}\right)}
\ \text{(D.1, bonded directly)},
\qquad
f_0 = \frac{1}{2\pi}\sqrt{\frac{0.111}{d}\left(\frac{1}{m_1'} + \frac{1}{m_2'}\right)}
\ \text{(D.2, studs over a filled cavity)}.
$$

Table D.1 then reads the weighted improvement off $f_0$, rounded to the
one-third-octave band it falls in. Below 200 Hz the lining helps, by
$74.4 - 20\lg f_0 - R_w/2$ dB, and the better the bare wall the less there is
to gain. From 200 Hz upwards it costs: 1 dB at 200 Hz falling to 10 dB from
630 Hz to 1600 Hz, recovering to 5 dB above. Getting the resonance well below
the range of interest is therefore the entire design brief.

```python
from phonometry import (
    lining_improvement, lining_improvement_in_situ, lining_resonance_frequency,
    weighted_lining_improvement,
)

# A 9.5 mm plasterboard laminated with 32 mm EPS (s' = 65 MN/m3), bonded with
# adhesive dabs to a 100 mm aircrete block wall of 51 kg/m2.
f0 = lining_resonance_frequency(51.0, 6.3, dynamic_stiffness=65e6)
round(f0)                                       # 542 Hz: squarely in the way
round(weighted_lining_improvement(f0, 45.0))    # -9 dB, it makes things worse

# The same lining on studs over a 100 mm mineral-wool-filled cavity.
f0 = lining_resonance_frequency(51.0, 6.3, cavity_depth=0.100)
round(f0)                                        # 71 Hz
round(weighted_lining_improvement(f0, 45.0), 1)  # +15.9 dB

# External thermal insulation systems have their own Annex D fits.
res = lining_improvement(100.0, system="mineral_wool")   # Formula (D.3)
res.delta_rw, res.delta_ra, res.delta_ratr               # (10.5, 8.0, 9.7) dB
lining_improvement(100.0, system="mineral_wool", anchors=True).delta_rw  # (D.5)
res.plot()   # the Annex D ratings against fo, with this system marked

# A laboratory rating transfers to the field through Formula (D.8).
round(lining_improvement_in_situ(10.0, 100.0, 60.0), 1)  # 4.4 dB on a Rw 60 wall
```

## What this guide covers

**Covered.** Hopkins 3.6.3's tapping-machine force pulse (Eqs. 3.85, 3.90 to
3.92, 3.95 to 3.106) through `tapping_force_spectrum`, `force_pulse`,
`plate_contact_stiffness`, `covering_contact_stiffness`,
`tapping_cut_off_frequency`, `hammer_limiting_frequency` and
`short_pulse_mean_square_force`; the soft-covering improvement of Hopkins
4.4.3.1 (Eq. 4.114 and the two-line estimate) through `covering_improvement`;
ISO 12354-2:2017 Annex C (Formulae C.1 to C.6) with Hopkins 4.4.4 and Vigran
8.4 through `floating_floor_resonance_frequency`,
`floating_floor_improvement_spectrum`, `weighted_floating_floor_improvement`,
`combined_dynamic_stiffness`, `double_floating_floor_resonances` and
`resilient_mount_improvement`; and ISO 12354-1:2017 Annex D (Formulae D.1 to
D.8 and Table D.1) through `lining_resonance_frequency`,
`weighted_lining_improvement`, `lining_improvement` and
`lining_improvement_in_situ`.

**Not covered.** The force model assumes a frequency-independent
driving-point impedance, so a joisted or battened lightweight floor, whose
impedance changes with where the hammer lands, is outside it; Hopkins points
to numerical methods there. Soft coverings are treated as linear springs, and
their measured non-linearity under the tapping machine is not modelled. There
is no per-band prediction of a lining's $\Delta R(f)$: Annex D is a
single-number method, and where ISO 12354 needs a lining spectrum it either
measures it (ISO 10140-1 Annex G) or, for a floating floor, takes
$\Delta R(f) = \Delta L(f)$, which its own Annex L calls rough. Heavy impact
sources (the rubber ball) are not covered by any of these models.

## See also

- [Floor-Covering Impact Improvement (ISO 16251-1)](impact-improvement.md):
  the measurement these predictions are checked against.
- [Dynamic stiffness of resilient materials (EN 29052-1)](dynamic-stiffness.md):
  where $s'$ comes from.
- [Predicting Sound Insulation (EN 12354)](insulation-prediction.md):
  the simplified model that consumes $\Delta L_w$ and $\Delta R_w$.
- [Detailed Per-Band Prediction (ISO 12354)](detailed-prediction.md):
  the per-band model whose floating-floor term is `floating_floor_improvement`.
- [Predicting Panel Sound Insulation](panel-sound-insulation.md):
  the mass-spring-mass resonance of a double leaf, the airborne counterpart of
  the lining resonance.
- API reference: [`building.resilient_layers`](https://jmrplens.github.io/phonometry/reference/api/building/resilient-layers/).
