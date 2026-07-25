← [Documentation index](README.md)

# Dynamic stiffness of resilient materials (EN 29052-1)

A **floating floor** is a heavy floating slab resting on a resilient layer; the
two form a mass-spring system whose **natural frequency** governs how much the
floor improves impact and airborne insulation. **EN 29052-1:1992** (identical to
ISO 9052-1:1989) measures the **dynamic stiffness per unit area** `s'` of the
resilient layer from the resonance of a standard load plate on a
200 mm × 200 mm specimen. `s'` is the input to the floating-floor term of the
EN 12354-2 impact model covered in
[Predicting Sound Insulation (EN 12354)](insulation-prediction.md). (ISO 16251-1 does not apply here:
its scope is limited to soft, locally-reacting floor coverings and explicitly
excludes floating floors.)

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/dynamic_stiffness_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/dynamic_stiffness.svg" alt="Floating-floor natural frequency as a function of the resilient layer's dynamic stiffness per unit area, for a light 40 kg/m² and a heavy 120 kg/m² floating floor, on a logarithmic stiffness axis, with a worked design point at 10 MN/m³ marked" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

# One line — from a measured resonance, the result draws its own f0(s')
# design curve with the determination marked:
res = materials.floating_floor_resonance(
    resonant_frequency=25.0, total_mass_per_area=200.0,
    floor_mass_per_area=120.0,
    airflow_resistivity=50.0, thickness=0.020, porosity=0.9,
)
res.plot()
plt.show()

# By hand, the same design curve for a light and a heavy floating floor:
s = np.logspace(np.log10(2.0), np.log10(100.0), 300)   # MN/m3
for m in (40.0, 120.0):
    plt.semilogx(s, materials.natural_frequency(s * 1e6, m), label=f"m' = {m:g} kg/m²")
plt.xlabel("Dynamic stiffness s' [MN/m³]"); plt.ylabel("Natural frequency f₀ [Hz]")
plt.legend(); plt.show()
```

</details>

## 1. Dynamic stiffness and resonance

The dynamic stiffness per unit area is a dynamic force per area divided by the
resulting change in thickness (Formula 1): `s' = (F/S)/Δd`. The resiliently
supported floor is a resonator whose natural frequency (Formula 2) and,
in the laboratory arrangement, measured resonant frequency (Formula 3) are

$$
f_0 = \frac{1}{2\pi}\sqrt{\frac{s'}{m'}}, \qquad
f_r = \frac{1}{2\pi}\sqrt{\frac{s'_t}{m'_t}},
$$

so the **apparent** dynamic stiffness follows from the resonance (Formula 4):

$$
s'_t = 4\pi^2\,m'_t\,f_r^2 .
$$

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_dynamic_stiffness_rig_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_dynamic_stiffness_rig.svg" alt="ISO 9052-1 resonance rig: a vertical exciter and an accelerometer on the load plate over the 200 mm by 200 mm resilient specimen, read as a mass-spring system whose response peak gives the resonant frequency and the apparent dynamic stiffness" width="92%"></picture>

In the test arrangement the specimen lies between the rigid foundation and a
load plate whose total mass per unit area, plate plus added load, is
200 kg/m² (8 kg on the 0.04 m² specimen). That load reproduces the static
preload of a typical floating floor, about 2 kPa. A vertical exciter drives
the plate, an accelerometer picks up its response, and the fundamental
vertical resonance $f_r$ of the plate-on-specimen system is read from the
response peak; Formula 4 turns it into $s'_t$.

```python
from phonometry import materials

# Standard 8 kg load plate on the 0.04 m2 specimen -> m't = 200 kg/m2;
# the fundamental resonance is measured at 25 Hz.
s_t = materials.apparent_dynamic_stiffness(resonant_frequency=25.0, total_mass_per_area=200.0)
print(round(s_t / 1e6, 3))                              # 4.935  MN/m3

# Installed on a 120 kg/m2 floating screed with s' = 10 MN/m3:
print(round(materials.natural_frequency(10e6, 120.0), 1))      # 45.9  Hz
```

## 2. The enclosed-gas term and airflow resistivity

For an air-permeable material the enclosed pore air adds a parallel stiffness
from its isothermal compression (Formula 7): `s'a = p₀/(d·ε)`, with `p₀` the
atmospheric pressure, `d` the loaded thickness and `ε` the porosity. The
standard's worked NOTE (`p₀ = 0.1 MPa`, `ε = 0.9`) is `s'a = 111/d` MN/m³ for
`d` in millimetres:

```python
from phonometry import materials

print(round(materials.enclosed_gas_stiffness(thickness=0.020, porosity=0.9) / 1e6, 2))
# 5.56  MN/m3   (the NOTE's 111/20 = 5.55 MN/m3)
```

The dynamic stiffness of the *installed* material is then set by the lateral
airflow resistivity `r` (clause 8.2): `s' = s't` for `r ≥ 100 kPa·s/m²`,
`s' = s't + s'a` for `10 ≤ r < 100 kPa·s/m²`, and for `r < 10 kPa·s/m²` the
method only resolves `s' = s't` when the gas term is negligible. `floating_floor_resonance`
chains the whole determination:

```python
from phonometry import materials

res = materials.floating_floor_resonance(
    resonant_frequency=25.0, total_mass_per_area=200.0,
    floor_mass_per_area=120.0,
    airflow_resistivity=50.0, thickness=0.020, porosity=0.9,
)
print(round(res.dynamic_stiffness / 1e6, 2), round(res.natural_frequency, 1))
# 10.49 47.1

res.plot()   # the f0(s') design curve with this determination marked (needs matplotlib)
```

The `DynamicStiffnessResult` carries the apparent, enclosed-gas and installed
stiffnesses, the test resonance and the installed-floor natural frequency, and
its `.plot()` draws the `f₀(s')` design curve.

**Test-report fiche.** `DynamicStiffnessResult.report(path)` renders a one-page
accredited dynamic-stiffness test report (EN 29052-1:1992 = ISO 9052-1:1989): a
metadata header (specimen, the total mass per unit area $m'_t$, the loaded
thickness $d$ in metres shown in mm, test facility, climate), a metrics table (the resonant frequency
$f_r$, the apparent stiffness $s'_t$ of Formula 4, the enclosed-gas term $s'_a$
when it applies, the installed $s'$ of clause 8.2 and the natural frequency
$f_0$ of Formula 2) beside the $f_0(s')$ design curve, and a boxed apparent
dynamic stiffness $s'_t$ (no pass/fail). Clause 9 rounds every stiffness to the
nearest MN/m³; `language="es"` renders the Spanish fiche. The fiche always
embeds the $f_0(s')$ design curve, so it needs both the report and plot extras
(`pip install "phonometry[report,plot]"`).

```python
from phonometry import ReportMetadata, materials

res = materials.floating_floor_resonance(
    resonant_frequency=45.0, total_mass_per_area=200.0,
    floor_mass_per_area=110.0,
    airflow_resistivity=50.0, thickness=0.020, porosity=0.9,
)
res.report(
    "dynamic_stiffness.pdf",
    metadata=ReportMetadata(
        specimen="20 mm mineral-wool resilient layer",
        mass_per_area=200.0, thickness=0.020,   # thickness d in metres (20 mm)
        measurement_standard="EN 29052-1",
    ),
)   # one-page fiche (needs phonometry[report,plot])
```

[![EN 29052-1 dynamic-stiffness example report: a metadata header with the total mass per unit area and the loaded thickness, a metrics table of the resonant frequency, the apparent, enclosed-gas and installed dynamic stiffnesses and the natural frequency beside the f0(s') design curve, and the boxed apparent dynamic stiffness s't](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/en29052_dynamic_stiffness_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/en29052_dynamic_stiffness_example.pdf)

## 3. What the resonance method assumes, and where it bites

The evaluation treats the rig as a single-degree-of-freedom system: the load
plate moves as a rigid piston on a massless spring. That holds while the
specimen is light against the plate and its first internal resonance sits
well above $f_r$; a heavy or very thick layer starts to act as a distributed
system and the simple Formula 4 reading degrades. Three practical pitfalls
follow from the preload:

* **`s'` is a stiffness *at the standard preload*.** Resilient layers are
  visibly non-linear in static load: mineral wool stiffens as it compresses,
  some foams soften. The 200 kg/m² load plate fixes the operating point, so
  the tabulated `s'` strictly describes floors near that surface mass.
  Designing a much heavier screed with the same `s'` extrapolates beyond the
  measurement.
* **Drive small.** The tangent stiffness is defined for small dynamic
  strains; driving the plate hard pushes the layer into its non-linear range
  and shifts the apparent resonance downward. Keep the excitation at the
  lowest level that gives a clean peak.
* **Respect the contact.** The standard seats the load plate on a thin
  bonding layer (a plaster paste) so the full specimen area carries the
  load. A dry, uneven contact concentrates the force, stiffens the response
  locally and biases $f_r$ upward.

The natural frequency that matters in the end is not the rig's $f_r$ but the
installed floor's $f_0$ from Formula 2: the floating floor only improves
insulation well above $f_0$, which is why a low `s'` (a soft layer under a
heavy slab) is the design goal.

## References

- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  Floating-floor design and the role of the resilient layer's dynamic
  stiffness in the impact-sound improvement.
- International Organization for Standardization. (1989). *Acoustics —
  Determination of dynamic stiffness — Part 1: Materials used under floating
  floors in dwellings* (ISO 9052-1:1989).
  [iso.org catalogue](https://www.iso.org/standard/16620.html).
  The international original of EN 29052-1, the method this page implements.

## Standards

EN 29052-1:1992 (= ISO 9052-1:1989), *Acoustics — Determination
of dynamic stiffness — Part 1: Materials used under floating floors in
dwellings*: the dynamic stiffness per unit area (Formula 1), the resonance
relations (Formulae 2-4), the enclosed-gas term (Formula 7, clause 8.2 NOTE
`s'a = 111/d` MN/m³) and the airflow-resistivity regimes (clause 8.2, Formulae
5-6). Conformance is anchored on the standard's own numeric NOTE plus
hand-computed closed-form values of the resonance relations.
