← [Documentation index](../../README.md)

# Installed structure-borne sound from equipment (EN 12354-5)

**EN 12354-5:2009** predicts the sound pressure level in a receiving room caused
by building service equipment (pumps, fans, lifts, water installations) that
injects **structure-borne sound** into the building. It closes the
structural-vibroacoustics chain: the source is described by its characteristic
structure-borne sound power level $L_{Ws,c}$, derived from the EN 15657
reception-plate measurement through the Formula (15)/(17) conversion and a
mobility correction (**not** the raw plate-injected level;
see [EN 15657](structure-borne-power.md)); the source and receiver point
mobilities set how much power is actually coupled into the structure, and the
building transmission carries it to the receiving room. The Annex I mobility
correction `installed_power_from_reception_plate` refers the characteristic
reception-plate level $L_{Ws,n}$ to the actual receiver,
$L_{Ws,\mathrm{inst}} = L_{Ws,n} + 10\log_{10}(Y_{\infty,i} / Y_{\infty,\mathrm{rec}})$
with $Y_{\infty,\mathrm{rec}} = 5\cdot 10^{-6}\ \text{m/(N·s)}$;
with the source mobility instead it yields $L_{Ws,c}$ (Annex I.3, Table I.8).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/installed_structure_borne_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/installed_structure_borne.svg" alt="The EN 12354-5 cascade per octave band: the characteristic structure-borne power level, the installed power level after subtracting the coupling term, the per-path normalised sound pressure levels, and their energetic total" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# EN 15657 characteristic source power and illustrative point mobilities.
bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
lws_c = np.array([78.0, 82.0, 84.0, 81.0, 77.0, 72.0, 66.0])
ys = (2e-4 + 1e-4j) * (bands / 250.0)        # source mobility
yi = (3e-5 + 1e-5j) * np.ones_like(bands)    # receiver mobility
dc = np.array([float(building.coupling_term(a, b)) for a, b in zip(ys, yi)])

# Two transmission paths: the excited floor and a flanking wall.
paths = [
    {"adjustment_term": 6.0, "element_area": 12.0,
     "flanking_reduction_index": np.linspace(44.0, 62.0, 7)},
    {"adjustment_term": 7.0, "element_area": 9.0,
     "flanking_reduction_index": np.linspace(46.0, 64.0, 7)},
]
res = building.installed_source_prediction(lws_c, dc, paths, frequencies=bands)
res.plot()   # characteristic and installed power, per-path levels and the total
plt.show()
```

</details>

## 1. Coupling and installed power

Only part of the characteristic power is injected into the supporting element;
the loss is the **coupling term** $D_C$, positive whenever the two mobilities
are well mismatched, set for a point excitation by the source mobility $Y_s$
and the receiver mobility $Y_i$ (Formula 19b):

$$
D_{C,i} = 10\log_{10}\frac{|Y_s + Y_i|^2}{|Y_s|\,\mathrm{Re}\{Y_i\}},
$$

which reduces to $10\log_{10}(|Y_s|/\mathrm{Re}\{Y_i\})$ for a **force source** (high source
mobility, Formula 19c) and to $-10\log_{10}(|Y_s|\,\mathrm{Re}\{Z_i\})$ for a **velocity source**
(low source mobility, Formula 19d); an elastic support adds its transfer
mobility $Y_k$ inside the modulus (Formula 19e). The **installed** power level is
then (Formula 18b) $L_{Ws,\mathrm{inst}} = L_{Ws,c} - D_C$.

The physics behind $D_C$ is the classical power input of a point-excited
plate: only the real part of the receiver's driving-point mobility absorbs
power, and the mismatch between $Y_s$ and $Y_i$ decides how much of the
source's capability ever enters the structure (Hopkins 2007, Section 2.8).
The mobilities themselves come from the
[mechanical-mobility chain](../../vibration/structural/mechanical-mobility.md): measured per ISO 7626,
or from the infinite-plate closed forms of
[panel theory](panel-sound-insulation.md) when no measurement exists. A pump
on a concrete slab is the textbook force source: its casing mobility is
orders of magnitude above the slab's, so $D_C$ collapses to Formula 19c and
the injected power no longer depends on the receiver at all.

```python
from phonometry import building

# A near-force source (Y_s >> Y_i) on a concrete floor:
dc = building.coupling_term(2e-4 + 1e-4j, 3e-5 + 1e-5j)
print(round(float(dc), 2))                                          # 9.86 dB
print(round(float(building.installed_structure_borne_power_level(82.0, dc)), 1))  # installed L_Ws
```

Nothing measures the source mobility of a boiler, so clause D.1.3 builds it out
of the machine's own parts and Table D.1 gives the six closed forms it uses.
`typical_element_mobility` is that table: the rows are `"mass"`
$[2\pi f M]^{-1}$, `"bar_end"` $[\rho c_L S]^{-1}$, `"beam"`
$[7{,}6\,\rho t w\sqrt{c_L t f}]^{-1}$, `"plate"` $[2{,}3\,c_L\rho t^2]^{-1}$,
`"pipe"` $[63\,\rho t r\sqrt{c_L r f}]^{-1}$ and `"mass_spring"`
$\left[\left(\frac{2\pi f\eta}{s(1+\eta^2)}\right)^2 + \left(\frac{2\pi f}{s(1+\eta^2)} - \frac{1}{2\pi f M}\right)^2\right]^{1/2}$,
each in m/N.s. `TABLE_D1_QUANTITIES` is the table's "describing quantities"
column, and only those may be passed; the four rows whose expression contains
$f$ take `frequency` and the other two refuse it.

```python
from phonometry import building

# A 120 kg pump on four rubber mounts, 1,0e6 N/m each, loss factor 0,1.
y_feet = building.typical_element_mobility(
    "mass_spring", frequency=125.0, mass=120.0, stiffness=4.0e6, loss_factor=0.1
)
print(f"{float(y_feet):.3g}")                       # 0.000185 m/(N.s)
```

## 2. Transmission to the receiving room

EN 12354 parts 1 and 2 handle rooms excited through the air and by the
standard tapping machine; part 5 covers the sources that shake the building
*directly*: pumps, fans, lifts, whirlpool baths, cisterns and the pipework
that ties them into walls and floors. Once the installed power is in the
structure, several elements radiate into the receiving room: the excited
element itself and every element the vibration reaches across the junctions.
Each excited-element/radiating-element pair $i \to j$ is a transmission path with
its own adjustment term and flanking index:

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_installed_paths_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_installed_paths.svg" alt="EN 12354-5 installation paths: a pump on resilient mounts injects structure-borne power into the floor slab, the excited floor radiates into the receiving room below, a second path travels along the slab into the flanking wall, and the prediction cascade runs from the characteristic power to the normalised sound pressure level" width="92%"></picture>

Each transmission path $i \to j$ gives a normalised sound pressure level from the
installed power, the structure-to-airborne adjustment term $D_{sa}$, the flanking
sound reduction index $R_{ij,\mathrm{ref}}$ (EN 12354-1) and the element area
(Formula 18a):

$$
L_{n,s,ij} = L_{Ws,\mathrm{inst},i} - D_{sa,i} - R_{ij,\mathrm{ref}}
             - 10\log_{10}\frac{S_i}{S_0} - 10\log_{10}\frac{A_0}{4},
$$

with $S_0 = A_0 = 10\ \text{m}^2$, and the paths combine energetically (Formula 17).

$D_{sa}$ is normally **negative** — the standard's own Annex I columns run from
about $-14$ dB at 63 Hz to $-45$ dB at 2 kHz — and Formula (18a) subtracts it,
so a negative value *raises* the predicted level. Annex F.2 gives the working
form $D_{sa,i} = 10\log_{10}(400 f_{c,i}\sigma_i / m_i f^2)$, which is
`structure_to_airborne_adjustment`; passing a positive number in its place
leaves the prediction 20 to 40 dB low with no error raised.

```python
import numpy as np
from phonometry import building

bands = np.array([250.0, 500.0, 1000.0])
res = building.installed_source_prediction(
    characteristic_power_level=np.array([80.0, 82.0, 78.0]),
    coupling_term=np.array([9.0, 10.0, 11.0]),
    paths=[
        {"adjustment_term": np.array([-19.0, -25.0, -31.0]),
         "flanking_reduction_index": np.array([50.0, 52.0, 55.0]), "element_area": 12.0},
        {"adjustment_term": np.array([-22.0, -28.0, -34.0]),
         "flanking_reduction_index": np.array([52.0, 54.0, 57.0]), "element_area": 8.0},
    ],
    frequencies=bands,
)
print(np.round(res.total_level, 1))      # total L_n,s per band
print(round(res.overall_level, 1))       # band-summed level [dB]

res.plot()   # the per-path and total L_n,s cascade, as in the figure above (needs matplotlib)
```

The `InstalledSourceResult` carries the per-path levels, the total per band, the
installed power level and `.overall_level`, and its `.plot()` draws the whole
cascade.

Where the receiving room is more than one junction away, clause F.1 subtracts an
adjustment $\Delta K$ from the summed junction $K_{ij}$ to cover the transmission
by wave types other than bending waves: `multi_junction_adjustment` returns the
4 dB it gives for two junctions and the 6 dB for three or more, with the
resulting $K_{ij}$ floored at `MINIMUM_MULTI_JUNCTION_KIJ` ($-5$ dB).

When the equipment itself cannot be characterised, clause D.1.2.3 substitutes a
source of known force level and Table F.1 gives that level for the ISO tapping
machine, in octave bands from 31,5 Hz to 4 kHz: 139, 142, 145, 148, 151, 154,
156 and 156 dB. `tapping_machine_force_level` returns them,
`tapping_machine_force_level_estimate` the closed form printed beside the table
(valid only up to about 1000 Hz, and the only route to one-third-octave values),
and `tapping_machine_characteristic_power_level` /
`tapping_machine_coupling_term` turn them into the $L_{Ws,c}$ and $D_C$ of
Formulae (D.9a) and (D.9b). The table's levels are re $10^{-6}$ N despite the
"re 1 pN" its caption prints; see [the errata register](../../ERRATA.md).

```python
import numpy as np
from phonometry import building

bands = np.array(building.TABLE_F1_OCTAVE_BANDS)
lw_c = building.tapping_machine_characteristic_power_level(
    bands, building.tapping_machine_force_level()
)
dc = building.tapping_machine_coupling_term(bands, 1.07e-6)   # 220 mm concrete
print(np.round(lw_c - dc, 1))   # [79.3 82.3 85.3 88.3 91.3 94.3 96.3 96.3]
```

## 3. The prediction report (`.report()`)

A prediction ends as a *document*. `InstalledSourceResult.report(path)` writes a
one-page PDF fiche, clearly labelled a prediction and not a measurement: a
prediction-basis line naming EN 12354-5:2009, an optional metadata header
(client, source equipment, receiving room, instrumentation, climate, date), a
per-band table (nominal octave/one-third-octave frequency, the installed
structure-borne power level $L_{Ws,\mathrm{inst}}$, each transmission path's
normalised SPL
$L_{n,s,ij}$ and the combined total $L_{n,s}$), the per-path and total
$L_{n,s}(f)$
spectra, and a boxed band-summed total $L_{n,s}$ (dB) with the installed power
total and the path count.

The relevant `ReportMetadata` fields are `client`, `specimen` (the source
equipment), `test_room` (the receiving room), `instrumentation` and the footer
identity `laboratory`, `operator`, `report_id` and `notes`. Supplying
`requirement` adds a PASS/FAIL verdict against a declared upper limit on the
overall $L_{n,s}$ (lower is better). `verbose=True` adds one column per
transmission path (up to five); otherwise only the installed power and the
combined total are shown. `language="es"` renders the Spanish fiche. The basis
strip states Formulae 18a/17 and the prediction disclaimer. Rendering needs the
optional `phonometry[report]` extra (reportlab), plus matplotlib for the plot.

```python
import numpy as np
from phonometry import ReportMetadata, installed_source_prediction

bands = np.array([63, 125, 250, 500, 1000, 2000], float)
lwc = np.array([84.4, 82.5, 69.9, 67.6, 61.6, 49.9])   # characteristic power [dB]
dsa = np.array([-13.6, -17.3, -17.4, -20.0, -26.9, -32.9])
paths = [
    {"adjustment_term": dsa,
     "flanking_reduction_index": np.array([43.0, 46, 50.2, 54.7, 64.6, 73]),
     "element_area": 12.8},
    {"adjustment_term": dsa,
     "flanking_reduction_index": np.array([37.0, 41.2, 35.9, 37.7, 49, 57.8]),
     "element_area": 12.8},
]
res = installed_source_prediction(lwc, 16.2, paths, frequencies=bands)
res.report(
    "installed_structure_borne.pdf",
    metadata=ReportMetadata(
        client="Example dwelling refurbishment",
        specimen="WC flushing cistern (wall-fixed)",
        test_room="Receiving room: adjacent bedroom",
        report_id="EXAMPLE-12354-5",
        requirement=45.0,
    ),
)   # overall L_n,s ~ 43 dB -> declared limit 45 dB: PASS
```

The rendered example fiche, regenerated with `make reports`, is kept in the
repository. Click the preview to open the PDF:

[![EN 12354-5 installed structure-borne prediction example report, clearly labelled a prediction and not a measurement: a header with the client, the source equipment, the receiving room and the identity, the octave-band table (63 Hz to 2 kHz) of the installed structure-borne power level L_Ws,inst, the two flanking paths normalised SPL L_n,s,ij and the combined total L_n,s, the per-path and total L_n,s(f) spectra, and the boxed band-summed total L_n,s with the installed power total and the path count, closed by a basis strip stating Formulae 18a/17 and the prediction disclaimer, with a PASS verdict against the declared 45 dB limit](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/en12354_5_installed_structure_borne_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/en12354_5_installed_structure_borne_example.pdf)

*Installed structure-borne prediction fiche (`InstalledSourceResult.report`), an
EN 12354-5 estimate of the normalised structure-borne SPL in the receiving
room.*

## See also

- [Structure-borne sound power of equipment (EN 15657)](structure-borne-power.md):
  the reception-plate characterisation that supplies $L_{Ws,c}$ and the source
  mobility this prediction consumes.
- [Mechanical mobility and the FRF family (ISO 7626-1)](../../vibration/structural/mechanical-mobility.md):
  the measured $Y_s$ and $Y_i$ behind the coupling term.
- [Bending-wave transmission at plate junctions](../../vibration/structural/junction-transmission.md):
  the junction physics that carries the installed power to the flanking
  radiators.
- [Predicting Sound Insulation (EN 12354)](insulation-prediction.md): the
  airborne and impact members of the same prediction family.

## References

- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-borne
  sound: Structural vibrations and sound radiation at audio frequencies*
  (3rd ed.). Springer. ISBN 978-3-540-22696-3.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728).
  The source-receiver mobility coupling and the structure-borne transmission
  across junctions behind the coupling term and the path model.
- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  Section 2.8 (driving-point impedance and mobility): the power a mechanical
  source injects into a plate through the source and receiver mobilities.

## Standards

EN 12354-5:2009, *Building acoustics — Estimation of acoustic
performance of buildings from the performance of elements — Part 5: Sound levels
due to service equipment*: the coupling term (clause 4.4.3, Formulae 19a-19e),
the installed structure-borne power level (Formula 18b), the structure-to-
airborne adjustment term (clause 4.4.4, Formulae 20a/20b), and the normalised
sound pressure level per path and its energetic combination (Formulae 18a, 17).
Conformance is anchored on the coupling-term force-source limit and the
standard's own Annex I worked examples: the whirlpool bath of I.2
(Table I.6a: mobility correction and path 11) and the flushing cistern of I.3
(Tables I.8/I.9: source conversion, all four transmission paths, the
Formula 17 total and its 29 dB(A) closure), within the ±0.15 dB rounding of
the printed one-decimal intermediates. The informative tables of the annexes
are implemented as named lookups: Table D.1 (mobility of typical construction
elements), Table F.1 (force level of the ISO tapping machine) with Formulae
(D.9a) and (D.9b), the adjustment term of Formula (F.3) and the multi-junction
$\Delta K$ of clause F.1. $R_{ij,\mathrm{ref}}$ remains an input (from
measurement or EN 12354-1). Only the 2009 edition is implemented, not the
EN 12354-5:2023 revision.
