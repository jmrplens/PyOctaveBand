← [Documentation index](../../README.md)

# Structure-borne sound power of building equipment (EN 15657)

Building service equipment (pumps, fans, boilers, sanitary appliances)
injects **structure-borne sound power** into the building structure it is fixed
to, which then re-radiates as airborne noise in adjoining rooms. **EN 15657:2018**
measures it with the **reception-plate method**: the source is mounted on a
plate of known mass per unit area $m$ and area $S$ whose structural loss
factor $\eta$ is known, and the plate's spatial-average vibratory velocity is
measured. Formula (14) gives the power *injected into that particular plate*;
the plate-independent source quantities (the equivalent blocked force,
Formula 15; the characteristic reception-plate power level $L_{W\mathrm{sn}}$,
Formula 17; and the equivalent free velocity and source mobility,
Formulae 18/19) are derived from it and are what the EN 12354-5
installed-equipment prediction consumes.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/structure_borne_power_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/structure_borne_power.svg" alt="Reception-plate structure-borne sound power level per one-third-octave band of a source determined on a low-mobility and a high-mobility reception plate, which agree within the method" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# The same pump-like source measured on a heavy and on a light reception plate.
bands = np.array([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3150.0])
lv_low = np.array([88.0, 90.0, 87.0, 84.0, 80.0, 76.0, 71.0])

low = building.reception_plate_power(lv_low, bands, mass_per_area=600.0,
                                     area=2.0, reverberation_time=0.8)
high = building.reception_plate_power(lv_low + 6.0, bands, mass_per_area=150.0,
                                      area=2.0, reverberation_time=0.5)

# One line — the L_Ws(f) bars of one determination with its band-summed total:
low.plot()
plt.show()

# By hand, comparing the two plates from the results' fields:
x = np.arange(bands.size)
fig, ax = plt.subplots()
ax.bar(x - 0.2, low.power_level, width=0.4, label="low-mobility plate")
ax.bar(x + 0.2, high.power_level, width=0.4, label="high-mobility plate")
ax.set_xticks(x, [f"{b:g}" for b in bands])
ax.set(xlabel="Frequency [Hz]",
       ylabel="Structure-borne power level $L_{Ws}$ [dB re 1 pW]")
ax.legend()
plt.show()
```

</details>

## 1. The reception-plate relations

Why a plate at all? Characterising the source by its contact forces directly
would mean instrumenting every fixing point in up to six components each
(three forces, three moments), on a machine that must keep running normally.
The reception plate sidesteps the whole contact problem: let the source run
on a resonant plate whose dissipation is known, wait for the steady state,
and then the power the plate dissipates equals the power the source injects,
over all contacts and components at once. One spatial average of the plate
velocity replaces the entire force-measurement problem.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_reception_plate_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_reception_plate.svg" alt="EN 15657 reception plate: the source machine standing on a resiliently supported plate, accelerometers averaging the plate velocity, and the plate power balance converting the velocity level into the injected structure-borne power level" width="92%"></picture>

The power a resonant plate dissipates is
$P = \omega\,\eta\,(m S)\,\langle v^2 \rangle$, so the injected
power level in one-third-octave bands is (Formula 14)

$$
L_{W\mathrm{s}} = 10\log_{10}\!\left(\frac{2\pi f\,\eta\,m\,S}{f_0\,m_0\,S_0}\right)
         + L_v - 60 \;\;[\mathrm{dB\ re\ 1\ pW}],
$$

with references $f_0 = 1\ \text{Hz}$, $m_0 = 1\ \text{kg}$,
$S_0 = 1\ \text{m}^2$; the $-60\ \text{dB}$ term is $10\log_{10}(v_0^2/P_0)$ for
the EN 15657 velocity reference $v_0 = 10^{-9}\ \text{m/s}$. The plate
velocity is the energetic spatial average over the $N$ positions (Formula 12)
and the loss factor comes from the structural reverberation time $T_\mathrm{s}$
(Formula 13, identical to the ISO 10848 total loss factor):

$$
L_v = 10\log_{10}\!\Big(\tfrac{1}{N}\textstyle\sum 10^{L_{v,i}/10}\Big), \qquad
\eta = \frac{2.2}{f\,T_\mathrm{s}}.
$$

```python
import numpy as np
from phonometry import building

bands = np.array([100.0, 200.0, 400.0, 800.0])
lv_i = np.array([88.0, 90.0, 87.0, 89.0, 86.0, 90.0])   # six plate positions @ 200 Hz
print(round(building.spatial_mean_velocity_level(lv_i), 2))    # 88.6 dB re 1 nm/s

# Power level injected into the reception plate (loss factor from Ts):
res = building.reception_plate_power(
    velocity_level=np.array([90.0, 87.0, 82.0, 77.0]),
    frequency=bands, mass_per_area=600.0, area=2.0, reverberation_time=0.8,
)
print(np.round(res.power_level, 1))     # per-band L_Ws
print(round(res.total_level, 1))        # band-summed level [dB re 1 pW]

res.plot()   # the L_Ws(f) bars with the band-summed total, as in the figure above (needs matplotlib)
```

## 2. Low- and high-mobility plates

Two reception plates bracket the installation conditions. On the *low-mobility*
(heavy) plate the source's own dynamics barely change the plate's point mobility
or loss factor; the *high-mobility* (light) plate is dynamically loaded by the
source, so its reverberation time and mobility are measured with the source
attached. The plate-injected power plus the plate's point mobility (see
[mechanical mobility](../../vibration/structural/mechanical-mobility.md)) yield the source description
for the EN 12354-5 model through the conversion chain below.

## 3. From plate power to source quantities (Formulae 15–19)

The plate-injected $L_{W\mathrm{s}}$ is **not** a source descriptor: the same source
injects a different power into a different receiver. EN 15657 derives the
plate-independent quantities: the **equivalent blocked force level**
(Formula 15, re $F_0 = 10^{-6}\ \text{N}$) from the low-mobility plate,

$$
L_{F\mathrm{b,eq}} = L_{W\mathrm{s,low}} - 10\log_{10}\frac{\mathrm{Re}\{Y_\mathrm{R,low,eq}\}}{Y_0},
$$

the **characteristic reception-plate power level** that EN 12354-5 consumes
(Formula 17), referred to the standard 10 cm concrete plate of characteristic
mobility $Y_{\mathrm{R},\infty,\mathrm{low}} = 5\times10^{-6}\ \text{m/(N·s)}$
(clause 7.2.4),

$$
L_{W\mathrm{sn}} = L_{F\mathrm{b,eq}} + 10\log_{10}\frac{Y_{\mathrm{R},\infty,\mathrm{low}}}{Y_0},
$$

and, from the high-mobility plate, the **equivalent free velocity level**
(Formula 18, re $10^{-9}\ \text{m/s}$) and the **source mobility**
$|Y_\mathrm{S,eq}|$
(Formula 19). The EN 12354-5 Annex I mobility correction
(`installed_power_from_reception_plate`, see
[installed structure-borne sound](installed-structure-borne.md)) then refers
$L_{W\mathrm{sn}}$ to the actual receiving element.

```python
from phonometry import building

# EN 12354-5 Annex I.3 (flushing cistern, wall contact, 63 Hz): measured on a
# plate of Y = 5.34e-6 m/(N·s); the wall's characteristic mobility is 24.1e-6.
lfb = building.equivalent_blocked_force_level(61.7, 5.34e-6)      # Formula (15)
lwsn = building.characteristic_reception_plate_power(lfb)         # Formula (17)
inst = building.installed_power_from_reception_plate(lwsn, 24.1e-6)  # Annex I
print(round(float(lwsn), 1), round(float(inst), 1))         # 61.4 68.2  (Table I.8)

# Free velocity (Formula 18) + blocked force close the source mobility (19):
lvf = building.equivalent_free_velocity_level(70.0, 1.0e-2)
print(float(building.source_mobility_from_levels(lvf, lfb)))      # |Y_S,eq| in m/(N·s)
```

Two pieces of the standard stay on the operator's side of the line. Formula 16
— the equivalent point mobility of the plate as the arithmetic mean of
$\mathrm{Re}\{Y\}$ over its contact points — is not implemented: the functions
above take an already-known plate mobility rather than deriving it from
per-point measurements. The Annex C power-substitution method is not
implemented either; the library takes an already-averaged $L_v$, so a
substitution determination must be reduced to a plate power level by hand
(Formulae C.1/C.2) first. And nothing here checks the facility: the plate's
dimensions, density, aspect ratio and loss factor, the position count and
clearances, the background correction and the operating conditions are the
operator's responsibility, and a value computed from a non-conforming plate is
returned without complaint.

The direct source-side counterpart is the ISO 9611 free velocity level (re
$v_0 = 5\times10^{-8}\ \text{m/s}$) measured at the contact points of
resiliently mounted
machinery; its equation (9) position average is `mean_free_velocity_level()`.

## 4. The characterization report (`.report()`)

A characterization ends as a *document*. `StructureBornePowerResult.report(path)`
writes a one-page PDF fiche laid out like a sound-power test sheet: the
standard-basis line naming the EN 15657:2018 reception-plate method (Formula
14), an optional metadata header (client, source equipment, test environment,
instrumentation, climate, date), a per-band table (nominal
octave/one-third-octave frequency, the spatial mean plate velocity level $L_v$
and the injected structure-borne sound power level $L_{W\mathrm{s}}$), the $L_{W\mathrm{s}}(f)$
spectrum with a nominal band axis, and a boxed band-summed total $L_{W\mathrm{s}}$ (dB re
1 pW) with the plate mass per area $m$ and area $S$.

The relevant `ReportMetadata` fields are `client`, `specimen` (the source
equipment), `test_room` (the test environment), `instrumentation`,
`temperature`, `relative_humidity`, `pressure`, `test_date` and the footer
identity `laboratory`, `operator`, `report_id` and `notes`; the plate mass and
area come from the result itself. Supplying `requirement` adds a PASS/FAIL
verdict against a declared upper limit on the total $L_{W\mathrm{s}}$ (lower is better).
`verbose=True` adds the plate loss factor $\eta$ column, and `language="es"`
renders the Spanish fiche. The basis strip states Formula 14 and the conversion
to the plate-independent source quantities (Formulae 15/17) required before
EN 12354-5. Rendering needs the optional `phonometry[report]` extra (reportlab),
plus matplotlib for the spectrum.

```python
import numpy as np
from phonometry import ReportMetadata, reception_plate_power

freqs = np.array([125, 250, 500, 1000, 2000, 4000], float)
lv = np.array([88.0, 90, 86, 82, 78, 73])   # spatial mean plate velocity level [dB]
res = reception_plate_power(
    lv, freqs, mass_per_area=25.0, area=1.2, reverberation_time=0.3,
)
res.report(
    "structure_borne_power.pdf",
    metadata=ReportMetadata(
        client="Example building services contractor",
        specimen="Circulation pump (wall-mounted)",
        test_room="Reception-plate test rig (heavy concrete plate)",
        laboratory="Phonometry reference example",
        report_id="EXAMPLE-15657",
    ),
)   # total L_Ws ~ 65 dB re 1 pW
```

The rendered example fiche, regenerated with `make reports`, is kept in the
repository. Click the preview to open the PDF:

[![EN 15657 structure-borne sound power example report: a header with the client, the source equipment, the reception-plate test rig and the accelerometer and climate, the octave-band table (125 Hz to 4 kHz) of spatial mean plate velocity levels Lv and injected structure-borne sound power levels L_Ws, the L_Ws(f) spectrum with a nominal band axis, and the boxed band-summed total L_Ws (dB re 1 pW) with the plate mass per area m = 25 kg/m2 and area S = 1.20 m2, closed by a basis strip stating the Formula 14 relation and the conversion to the plate-independent source quantities required before EN 12354-5](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/en15657_structure_borne_power_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/en15657_structure_borne_power_example.pdf)

*Structure-borne sound power fiche (`StructureBornePowerResult.report`), an
EN 15657 reception-plate characterization with the boxed total L_Ws.*

## See also

- API reference: [`building.measurement.structure_borne_power`](https://jmrplens.github.io/phonometry/reference/api/building/structure-borne-power/).
- Theory: [Point mobilities and radiation efficiency](../../reference/theory/vibration.md#point-mobilities-and-radiation-efficiency-cremer-5-hopkins-29): the point mobility, the source-receiver mismatch and the injected power they define.

## References

- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-borne
  sound: Structural vibrations and sound radiation at audio frequencies*
  (3rd ed.). Springer. ISBN 978-3-540-22696-3.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728).
  The plate power balance and the source-receiver mobility framework behind
  the reception-plate method and its Formula 15-19 source quantities.
- International Organization for Standardization. (1996). *Acoustics —
  Characterization of sources of structure-borne sound with respect to sound
  radiation from connected structures — Measurement of velocity at the
  contact points of machinery when resiliently mounted* (ISO 9611:1996).
  [iso.org catalogue](https://www.iso.org/standard/17424.html).
  The free-velocity source characterization that complements the
  reception-plate quantities.

## Standards

EN 15657:2018, *Acoustic properties of building elements and
buildings — Laboratory measurement of structure-borne sound from building
service equipment for all installation conditions*: the reception-plate method
(clause 7), the spatial mean velocity level (Formula 12), the plate loss factor
$\eta = 2.2/(f\,T_\mathrm{s})$ (Formula 13), the plate-injected power level $L_{W\mathrm{s}}$
(Formula 14) and the source-quantity chain: equivalent blocked force
(Formula 15), characteristic reception-plate power level (Formula 17,
$Y_{\mathrm{R},\infty,\mathrm{low}} = 5\times10^{-6}\ \text{m/(N·s)}$), equivalent free
velocity (Formula 18) and source mobility (Formula 19). ISO 9611:1996: the
free-velocity source characterization (equation (9),
$v_0 = 5\times10^{-8}\ \text{m/s}$). The plate velocity levels are referred to
$v_0 = 10^{-9}\ \text{m/s}$. Conformance is anchored on the resonant-plate
power balance $P = \omega\,\eta\,(m S)\,\langle v^2 \rangle$ (of which
Formula 14 is the level), the loss-factor identity, and the EN 12354-5
Annex I.3 Table I.8 conversion of the flushing-cistern source.
