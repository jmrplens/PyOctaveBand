← [Documentation index](../../README.md)

# Airflow Resistance

Push air slowly through a porous absorber and it pushes back. That viscous
drag, exerted by the pore walls on the air threading through them, is the
single most informative number a porous material has: it sets how much sound
the material dissipates at low frequency, it is the first input of every
equivalent-fluid model, and it is measured with nothing more exotic than a
pump and a manometer. ISO 9053 standardises the measurement twice over: the
**static method** of Part 1 drives a steady laminar flow through the specimen
and reads the pressure drop, and the **alternating method** of Part 2 replaces
the steady flow with a 2 Hz piston so that specimens too leaky or too delicate
for a stable static reading can be measured acoustically. This guide covers
the three quantities and their units, both methods, the accredited-style test
fiche, and what the measured resistivity feeds afterwards.

## 1. The three quantities (ISO 9053)

The airflow resistance quantifies how strongly a porous material opposes a steady
or slowly-oscillating flow. Both parts share the same three quantities and units
(ISO 9053-1:2018, Clause 3):

$$
R = \frac{\Delta p}{q_v}\ \left[\text{Pa·s/m}^3\right], \qquad
R_\mathrm{s} = R\,A\ \left[\text{Pa·s/m}\right], \qquad
\sigma = \frac{R_\mathrm{s}}{d}\ \left[\text{Pa·s/m}^2\right],
$$

with $\Delta p$ the pressure difference across the specimen, $q_v$ the volumetric
flow, $A$ the cross-section and $d$ the thickness. Note the specific airflow
resistance $R_\mathrm{s}$ is in **Pa·s/m** (not Pa·s/m²); the airflow resistivity $\sigma$
is $R_\mathrm{s}$ per metre of thickness.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_airflow_resistance_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_airflow_resistance.svg" alt="Airflow resistance measurement rigs: the ISO 9053-1 static method with a specimen in a holder, a steady laminar flow q_v and a differential manometer reading the pressure drop; and the ISO 9053-2 alternating method with an oscillating piston driving a cavity terminated by the specimen or an airtight plug, and a microphone reading the cavity level" width="92%"></picture>

## 2. Static method (ISO 9053-1)

In the static method (ISO 9053-1:2018) a steady laminar flow is stepped up and the
pressure difference plotted against the linear velocity $u = q_v/A$. A regression
of at least second order **constrained through the origin**, $\Delta p = a\,u +
b\,u^2$, is fitted, and $\Delta p$ and $R_\mathrm{s}$ are read at the reference velocity
$u = 0.5\ \text{mm/s}$ (Clause 7.5); the highest velocity must not exceed
15 mm/s. Because $R_\mathrm{s} = \Delta p/u = a + b\,u$, the linear term $a$ is the
zero-velocity specific airflow resistance.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/airflow_resistance_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/airflow_resistance.svg" alt="ISO 9053-1 static-method airflow resistance: the measured pressure drop against linear airflow velocity, fitted with a through-origin quadratic, with the specific airflow resistance evaluated at the 0.5 mm/s reference velocity" width="80%"></picture>

*The slightly super-linear pressure drop is fitted through the origin; the
specific airflow resistance is the fit read at 0.5 mm/s.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import materials

area = np.pi * 0.05**2                      # 100 mm diameter cell [m^2]
u = np.array([0.5, 1, 2, 4, 8, 12]) * 1e-3  # linear velocity [m/s]
dp = 1.6e4 * u + 4.0e5 * u**2               # measured pressure drop [Pa]
r = materials.static_airflow_resistance(u, dp, area=area, thickness=0.05)

u_fit = np.linspace(0.0, 13e-3, 200)
dp_fit = r.linear_coefficient * u_fit + r.quadratic_coefficient * u_fit**2
fig, ax = plt.subplots()
ax.plot(u_fit * 1e3, dp_fit, label="Through-origin fit  dp = a u + b u^2")
ax.plot(u * 1e3, dp, "o", label="Measured pressure drop")
ax.plot(r.evaluation_velocity * 1e3, r.pressure_drop, "D",
        label="Evaluation at 0.5 mm/s")
ax.set_xlabel("Linear airflow velocity u [mm/s]")
ax.set_ylabel("Pressure drop dp [Pa]")
ax.set_title(f"R_s = {r.specific_resistance:.0f} Pa·s/m")
ax.legend()
plt.show()
```

</details>

```python
import numpy as np
from phonometry import materials

area = np.pi * 0.05**2            # 100 mm diameter cell [m^2]
u = np.array([0.5, 1, 2, 4, 8, 12]) * 1e-3      # linear velocity [m/s]
dp = 1.6e4 * u + 4.0e5 * u**2                    # measured pressure drop [Pa]

r = materials.static_airflow_resistance(u, dp, area=area, thickness=0.05)
print(round(r.specific_resistance))   # 16200   R_s [Pa*s/m]
print(round(r.resistivity))           # 324000  sigma [Pa*s/m^2]
print(round(r.linear_coefficient))    # 16000   a = R_s at u -> 0
r.plot()   # the figure above: the fitted dp(u) with the evaluation point
```


### ISO 9053-1 report (`.report()`)

`StaticAirflowResult.report(path)` renders a one-page PDF fiche laid out like an
accredited airflow-resistance test report (ISO 9053-1:2018, static method): a
standard-basis line, an optional metadata header block, a two-panel body with a
metrics table (the evaluation velocity, the fitted pressure difference, the
airflow resistance $R$, the specific airflow resistance $R_\mathrm{s}$, the airflow
resistivity $\sigma$ when a thickness is available, and the through-origin fit
coefficients $a$ and $b$) beside the fitted $\Delta p(u)$ curve (the result's
own `.plot()`), the boxed specific airflow resistance $R_\mathrm{s}$ with $R$ and
$\sigma$ alongside, and a footer with the fixed disclaimer. ISO 9053-1 is a
material characterisation, so the fiche carries no pass/fail verdict.

It uses the same `ReportMetadata` container and rendering engine as the other
fiches. The descriptive fields that apply here are `client`, `manufacturer`,
`specimen`, `thickness` (the specimen thickness $d$, in metres, shown in
millimetres), `test_room`, `test_date`, `temperature`, `relative_humidity`,
`measurement_standard`, `laboratory`, `operator`, `report_id` and `notes`. The
`requirement` field is ignored (ISO 9053-1 has no verdict). The fiche embeds
the fitted curve, so rendering needs both reportlab and matplotlib
(`pip install "phonometry[report,plot]"`); only `engine="reportlab"` is
supported. The fiche renders in English by default; pass `language="es"` for a
Spanish fiche (translated fixed strings and a comma decimal separator).

```python
from phonometry import materials, ReportMetadata

r = materials.static_airflow_resistance(u, dp, area=area, thickness=0.05)
r.report(
    "airflow_fiche.pdf",
    metadata=ReportMetadata(
        specimen="50 mm porous absorber (open-cell)",
        thickness=0.050,
        measurement_standard="ISO 9053-1",
        test_room="Static airflow rig, 100 mm cell",
        laboratory="Phonometry Reference Laboratory",
    ),
)                                  # R_s, R and sigma at u = 0.5 mm/s
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![ISO 9053-1 static airflow-resistance example report: metadata header with the specimen thickness, the metrics table with the evaluation velocity, the fitted pressure difference, the airflow resistance R, the specific airflow resistance R_s, the airflow resistivity sigma and the through-origin fit coefficients beside the fitted pressure-drop curve, and the boxed specific airflow resistance R_s with R and sigma alongside](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso9053_airflow_resistance_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso9053_airflow_resistance_example.pdf)

*Static airflow-resistance fiche (`StaticAirflowResult.report`), $R_\mathrm{s}$, $R$ and $\sigma$ at 0.5 mm/s.*


## 3. Alternating method (ISO 9053-2)

In the alternating method (ISO 9053-2:2020) a piston oscillating at 1–4 Hz drives
an alternating flow into a cavity terminated either by the specimen or by an
airtight plug; the resistance follows from the sound-pressure-level difference
between the two terminations (Formula (2)):

$$
R = \frac{\kappa'\,P_\mathrm{S}}{2\pi f\,V}\cdot\frac{h_\mathrm{t}}{h_\mathrm{s}}\cdot
    10^{(L_{p,\mathrm{s}} - L_{p,\mathrm{t}})/20},
$$

where $\kappa'$ is the **effective** ratio of specific heats. Heat conduction
between the oscillating air and the cavity walls makes the compression not fully
adiabatic; the normative Annex A corrects $\kappa$ down to

$$
\kappa' = \frac{\kappa}{\sqrt{1 + (\kappa-1)\tfrac{S}{V}b
          + \tfrac{1}{2}\big((\kappa-1)\tfrac{S}{V}b\big)^2}},
\qquad b = \sqrt{\frac{2 c_0 l_\mathrm{h}}{\omega}},\quad
l_\mathrm{h} = \frac{k_\mathrm{a}}{\rho_0 c_0 C_\mathrm{P}},
$$

with $S$ and $V$ the cavity surface and volume and $b$ the thermal
boundary-layer thickness. For the Annex A.3 example cavity this gives
$\kappa' = 1.370$, about 2 % below the adiabatic 1.4008.

```python
from phonometry import materials

# Annex A.3 cavity: closed cylinder 100 mm x 100 mm, piston at 2 Hz
kp = materials.effective_kappa(cavity_surface=0.0471, cavity_volume=7.854e-4, frequency=2.0)
print(round(kp, 3))               # 1.37   effective ratio of specific heats

R = materials.alternating_airflow_resistance(
    level_specimen=74.0, level_termination=90.0,
    piston_stroke_specimen=14e-3, piston_stroke_termination=1.4e-3,
    frequency=2.0, cavity_volume=7.854e-4, kappa_prime=kp,
)
print(round(R))                   # 222956  airflow resistance R [Pa*s/m^3]
```

Pass the `effective_kappa` result to `alternating_airflow_resistance` for an
Annex-A-conforming figure; its `kappa_prime` argument otherwise defaults to the
uncorrected adiabatic 1.4. The call warns (via `AirflowResistanceWarning`) when
the piston frequency leaves 1–4 Hz or the Formula (3)/(4) validity criteria fail.


## 4. What the resistivity feeds

The flow resistivity is not an end in itself: almost every consumer of the
number is a **porous-material model**. The empirical Delany-Bazley and Miki
regressions predict a porous material's complex characteristic impedance and
wavenumber from $\sigma$ *alone*, through the dimensionless ratio
$X = \rho_0 f/\sigma$; the physics-based Johnson-Champoux-Allard model keeps
$\sigma$ as the first of its five parameters. Fitted to a tube or
reverberation-room measurement once, those models then predict the layer at
any thickness, backing or incidence angle. The whole model family, and the
multilayer solver that stacks the layers, lives in
[Porous and Multilayer Absorbers](porous-absorbers.md); the regression
validity window of Delany-Bazley ($0.01 < X < 1$) is stated there in the same
$X$ that $\sigma$ defines.

Because $\sigma$ enters the models as a ratio, its useful range is bounded on
both sides. A layer with too little resistivity barely couples to the sound
field: air moves through it freely and little energy is dissipated. Too much
resistivity and the layer behaves like a wall: the wave reflects off the
front face before the pores can absorb it. Between the two lies the classic
design window for the total flow resistance of a layer of thickness $d$,

$$
\rho_0 c_0 \;\lesssim\; R_\mathrm{s} = \sigma\,d \;\lesssim\; 4\,\rho_0 c_0,
$$

that is, a specific flow resistance of one to four times the characteristic
impedance of air ($\rho_0 c_0 \approx 410$ Pa·s/m). A 50 mm blanket therefore
wants a resistivity of roughly 8 kPa·s/m² to 33 kPa·s/m², which is exactly
where commercial absorber products cluster. The window is absorber-design
practice from the literature, not a requirement of either part of ISO 9053.

Typical orders of magnitude, for orientation rather than design: glass wools
run from a few kPa·s/m² in light thermal grades to some tens of kPa·s/m² in
dense acoustic boards; rock wools sit somewhat higher at equal density;
open-cell foams (melamine, polyurethane) span roughly 5 kPa·s/m² to
30 kPa·s/m²; and fibrous felts and compressed boards can exceed
100 kPa·s/m², at which point they act more as resistive facings than as bulk
absorbers. Within one product family the resistivity rises steeply with bulk
density and falls with fibre diameter, which is why nominally identical
products from two production runs can differ by tens of percent, and why a
measured $\sigma$, not a catalogue value, should anchor any model fit that
will be trusted quantitatively.

The number also travels beyond the porous models. The perforated-panel and
microperforated-panel impedances contain the same viscous physics evaluated
in a single hole; the [slit absorber](metamaterial-absorbers.md) inherits it per
slit; and when an [impedance tube](impedance-tube.md) measurement disagrees
with a model prediction, the first parameter to re-measure, before touching
the model, is $\sigma$.

## See also

- [Porous and Multilayer Absorbers](porous-absorbers.md): the Delany-Bazley,
  Miki and Johnson-Champoux-Allard models that consume the resistivity, and
  the multilayer transfer-matrix solver.
- [Impedance Tube](impedance-tube.md): the normal-incidence measurement that
  a $\sigma$-anchored model is usually fitted against.
- [Sound Absorption Measurement and Rating](absorption-measurement.md): the
  reverberation-room measurement and the ISO 11654 rating of the finished
  absorber.
- API reference: [`materials.absorbers.airflow_resistance`](https://jmrplens.github.io/phonometry/reference/api/materials/airflow-resistance/).

## References

- Allard, J. F., & Atalla, N. (2009). *Propagation of sound in porous media:
  Modelling sound absorbing materials* (2nd ed.). Wiley.
  ISBN 978-0-470-74661-5.
  [doi:10.1002/9780470747339](https://doi.org/10.1002/9780470747339).
  The porous-material theory that consumes the measured resistivity: viscous
  and thermal dissipation in the pores, and the models of section 4.
- International Organization for Standardization. (2018). *Acoustics —
  Determination of airflow resistance — Part 1: Static airflow method*
  (ISO 9053-1:2018).
  [iso.org catalogue](https://www.iso.org/standard/69869.html).
  The static method of section 2, with the through-origin regression and the
  0.5 mm/s reference velocity.
- International Organization for Standardization. (2020). *Acoustics —
  Determination of airflow resistance — Part 2: Alternating airflow method*
  (ISO 9053-2:2020).
  [iso.org catalogue](https://www.iso.org/standard/76744.html).
  The alternating method of section 3, with the Annex A effective ratio of
  specific heats, validated against the standard's own Annex A.3 worked
  example.

## Standards

ISO 9053-1:2018 (static airflow resistance); ISO 9053-2:2020 (alternating
airflow resistance). Every equation is derived from the standard text; the
[conformance report](../../CONFORMANCE.md) validates the library against the
ISO 9053-2 Annex A.3 worked example and closed-form identities.
