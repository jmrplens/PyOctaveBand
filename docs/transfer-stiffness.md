← [Documentation index](README.md)

# Dynamic transfer stiffness of resilient elements (ISO 10846)

The vibro-acoustic transfer property of a resilient element (a vibration
isolator, mount, bellows or hose) is its **dynamic transfer stiffness**
$k_{2,1}$, the frequency-dependent ratio of the *blocking force* on the output
(receiver) side to the displacement on the input (source) side
(ISO 10846-1, 3.7):

$$
k_{2,1} = \frac{F_{2,b}}{u_1} \quad [\mathrm{N/m}].
$$

Because a vibration isolator is only effective between structures of large
driving-point stiffness, the force it delivers to the receiver approximates this
blocking force (ISO 10846-1, Eq. 7), so $k_{2,1}$ is the quantity that
characterises the isolator's transmission. $k_{2,1}$ feeds the structure-borne
source and building prediction standards: ISO 9611, EN 15657 and EN 12354-5.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/transfer_stiffness_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/transfer_stiffness.svg" alt="Dynamic transfer stiffness level of a Kelvin-Voigt isolator: the true level (flat at 120 dB, rising with the damping term at high frequency) and the indirect-method estimate, which diverges at the mass/spring resonance where the transmissibility is not small and converges to the true level above three times the resonance" width="82%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import vibration

# Kelvin-Voigt isolator k + jwc loaded by an 8 kg blocking mass.
k, c, m2 = 1.0e6, 120.0, 8.0
f0 = np.sqrt(k / m2) / (2.0 * np.pi)
f = np.logspace(np.log10(f0 / 5.0), np.log10(f0 * 40.0), 600)

k_true = k + 1j * 2.0 * np.pi * f * c
t = vibration.base_transmissibility(f, m2, k, c)
k_indirect = vibration.transfer_stiffness_indirect(f, t, m2)  # warns where T is not small

# One line — the indirect determination bundled as a result draws its own
# Lk(f) level spectrum:
res = vibration.indirect_transfer_stiffness_result(f, t, blocking_mass=m2)
res.plot()
plt.show()

# By hand, the true element stiffness against the indirect-method estimate:
fig, ax = plt.subplots()
ax.semilogx(f, vibration.transfer_stiffness_level(k_true),
            label="true $L_k$ of $k+j\\omega c$")
ax.semilogx(f, vibration.transfer_stiffness_level(k_indirect), "--",
            label="indirect method $-(2\\pi f)^2 m_2 T$")
ax.axvline(f0, color="0.6", linestyle=":", label="resonance $f_0$")
ax.set(xlabel="Frequency [Hz]", ylabel="Transfer stiffness level $L_k$ [dB re 1 N/m]")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
```

</details>

## 1. The transfer-stiffness level and loss factor

Results are reported as a **level** re the reference stiffness
$k_0 = 1\ \text{N/m}$ (ISO 10846-2 and -3, 3.17), and in the low-frequency range
where inertial forces in the element are negligible the **loss factor** is the
tangent of the phase angle of $k_{2,1}$ (ISO 10846-1, 3.8):

$$
L_k = 10\lg\frac{|k_{2,1}|^2}{k_0^2} = 20\lg\frac{|k_{2,1}|}{k_0}, \qquad
\eta = \frac{\mathrm{Im}(k_{2,1})}{\mathrm{Re}(k_{2,1})}.
$$

```python
from phonometry import vibration

# A resilient mount with |k2,1| = 1 MN/m and a 5 % loss factor:
k = 1e6 * (1.0 + 0.05j)
print(round(float(vibration.transfer_stiffness_level(k)), 2))   # 120.00  dB re 1 N/m
print(round(float(vibration.loss_factor(k)), 3))                # 0.05
```

## 2. Direct and indirect determination

Why a *blocked* force rather than, say, the isolator's transmissibility?
Because a transmissibility is a property of a whole assembly: it changes with
whatever masses and stiffnesses the isolator happens to connect, so data
measured on one rig would not transfer to another installation. The blocked
force per unit input displacement is a property of the element alone, and it
predicts the force the element delivers to any receiver that is much stiffer
than the element itself, which is exactly the situation a vibration isolator
is designed for. ISO 10846 therefore sandwiches the isolator between a driven
input mass and an output that is either rigidly blocked or loaded with a
known mass:

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_transfer_stiffness_rig_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_transfer_stiffness_rig.svg" alt="ISO 10846 transfer-stiffness rigs: the isolator under test between a driven excitation mass and either a blocked output with a force transducer, the direct method, or a resiliently supported blocking mass, the indirect method" width="92%"></picture>

The **direct method** (ISO 10846-2) measures the blocked output force and the
input displacement, $k_{2,1} = F_{2,b}/u_1$. The **indirect method**
(ISO 10846-3) loads the output with a compact blocking mass $m_2$ and measures
the vibration transmissibility $T = u_2/u_1$; the blocking force is then the
inertia force of the mass (ISO 10846-3, Eq. 1):

$$
k_{2,1} = -(2\pi f)^2\,(m_2 + m_f)\,T \qquad (T \ll 1),
$$

with $m_f$ the mass of the output flange. The approximation is valid well above
the mass/spring resonance, where $T$ is small.

```python
import numpy as np
from phonometry import vibration

# Indirect method: a 10 kg blocking mass, transmissibility 0.01 at 500 Hz.
k = vibration.transfer_stiffness_indirect(500.0, 0.01, blocking_mass=10.0)
print(f"{abs(complex(k)):.3e}")            # 9.870e+05  N/m

# Bundle a swept measurement into a result carrying its level and loss factor:
f = np.logspace(1.5, 3.3, 200)
t = vibration.base_transmissibility(f, mass=8.0, stiffness=1e6, damping=120.0)
res = vibration.indirect_transfer_stiffness_result(f, t, blocking_mass=8.0)
print(round(float(res.level[-1]), 1))      # ~126  dB re 1 N/m (high-f)

res.plot()   # the Lk(f) level spectrum, as in the figure above (needs matplotlib)
```

The `TransferStiffnessResult` carries the complex $k_{2,1}$ and exposes `.level`,
`.loss_factor`, `.magnitude`, `.to("impedance"/"apparent_mass")` and `.plot()`.

**Test-report fiche.** `TransferStiffnessResult.report(path)` renders a one-page
dynamic-transfer-stiffness characterisation report for a resilient element
(ISO 10846-1:2008 definition; determined by the direct method, ISO 10846-2:2008,
or the indirect blocking-mass method, ISO 10846-3:2002). The transfer stiffness
is a continuous frequency-response function, not an octave-band quantity, so the
sheet presents it honestly as the $L_k(f)$ level spectrum plus a compact table
of characteristic points (the determination method, the blocking mass for the
indirect method, the frequency range, and the low-frequency stiffness plateau
$|k_{2,1}|$, its level $L_k$ and the loss factor there), and a boxed
low-frequency $L_k$ (the plateau that characterises the element below its
internal resonances). It is a characterisation, so there is no pass/fail
verdict; `language="es"` renders the Spanish fiche. The fiche always embeds the
$L_k(f)$ spectrum, so it needs both the report and plot extras
(`pip install "phonometry[report,plot]"`).

[![ISO 10846 dynamic-transfer-stiffness example report: a metadata header, a table of the FRF characteristic points (method, frequency range and low-frequency stiffness, level and loss factor) beside the transfer-stiffness level spectrum, and the boxed low-frequency level](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10846_transfer_stiffness_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10846_transfer_stiffness_example.pdf)

The two methods split the frequency axis between them. The direct method
works from 1 Hz, the lower bound of the ISO 10846-2 scope (in practice the
floor is set by the rig and its instrumentation), up to where the test rig's
own resonances intrude (typically a few hundred hertz for large elements); the
indirect method only becomes valid well above the blocking-mass/spring
resonance, where the transmissibility is small, and extends the
characterisation into the kilohertz range. A full isolator dataset is
usually the two spliced together.

## 3. Validity of the indirect method

ISO 10846-3 (clause 6) requires the $T \ll 1$ approximation to be accurate
within **1 dB** (12 % of the stiffness magnitude), which bounds the usable
frequency range on both sides:

* **Impedance mismatch (Inequality 2).** Valid only where
  $\Delta L_{1,2} = L_{a1} - L_{a2} \ge 20\ \text{dB}$, i.e. $|T| \le 0.1$, the
  constant `TRANSMISSIBILITY_LIMIT`. `transfer_stiffness_indirect` computes the
  per-band $|T|$ and emits a `PhonometryWarning` when any band exceeds it
  (routine near or below the mass/spring resonance, as in the figure above).
* **Rigid blocking mass (Inequality 3).** Above an upper frequency $f_3$ the
  blocking mass no longer moves as a rigid body; results are valid only while
  its measured effective mass $m_{2,\text{eff}} = 2F_2/(a'_1 + a''_1)$ (Eq. 4)
  stays within 1 dB of the rigid mass:
  $10\lg(m_{2,\text{eff}}^2/m_2^2) \le 1\ \text{dB}$.
* **Linearity (clause 7.6).** Two input spectra 10 dB apart must give
  transfer-stiffness levels within 1.5 dB.

The blocking-force idealisation itself is quantified by ISO 10846-1, Eq. (6):
for an isolator of output driving-point stiffness $k_{2,2}$ on a termination of
stiffness $k_t$, the delivered force is $F_2/F_{2,b} = 1/(1 + k_{2,2}/k_t)$,
within 10 % of the blocking force for $|k_{2,2}| < 0.1\,|k_t|$ (Eq. 7):

```python
import warnings
from phonometry import vibration

# |T| = 0.5 violates Inequality (2): the indirect result is flagged.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    vibration.transfer_stiffness_indirect(50.0, 0.5, blocking_mass=10.0)
print(caught[0].category.__name__)                     # PhonometryWarning

# Blocking-force approximation at the 10 % limit (ISO 10846-1, Eq. 6):
print(round(abs(complex(vibration.blocking_force_ratio(1e5, 1e6))), 4))   # 0.9091
```

## 4. Relation to the FRF family

The dynamic stiffness is a member of the frequency-response-function family
(ISO 10846-1, Annex A / Table A.2): it is the reciprocal of the receptance and
relates to the mechanical impedance $Z$ and effective mass $m_\text{eff}$ by
$k = j\omega Z = -\omega^2 m_\text{eff}$. These conversions are the same as the
[mechanical-mobility](mechanical-mobility.md) `convert_frf` pivot:

```python
from phonometry import vibration

k = 1e6 + 5e4j                                  # N/m, at 250 Hz
Z = vibration.convert_frf(k, 250.0, "dynamic_stiffness", "impedance")
print(abs(complex(vibration.convert_frf(Z, 250.0, "impedance", "dynamic_stiffness"))))  # 1.0012e6
```

## 5. Test-report fiche

`TransferStiffnessResult.report(path)` renders a one-page
dynamic-transfer-stiffness characterisation report for a resilient element
(ISO 10846-1:2008 definition; determined by the direct method, ISO 10846-2:2008,
or the indirect blocking-mass method, ISO 10846-3:2002). The transfer stiffness
is a continuous frequency-response function, not an octave-band quantity, so the
sheet presents it honestly as the $L_k(f)$ level spectrum plus a compact table
of characteristic points (the determination method, the blocking mass for the
indirect method, the frequency range, and the low-frequency stiffness plateau
$|k_{2,1}|$, its level $L_k$ and the loss factor there), and a boxed
low-frequency $L_k$ (the plateau that characterises the element below its
internal resonances). It is a characterisation, so there is no pass/fail
verdict; `language="es"` renders the Spanish fiche. The fiche always embeds the
$L_k(f)$ spectrum, so it needs both the report and plot extras
(`pip install "phonometry[report,plot]"`).

```python
import numpy as np
from phonometry import ReportMetadata, TransferStiffnessResult, transfer_stiffness_direct

freqs = np.array([20, 31.5, 50, 80, 125, 200, 315, 500, 800, 1250, 2000], dtype=float)
k21 = 1e6 + 1j * (2 * np.pi * freqs) * 80.0     # Kelvin-Voigt element k + jwc
u1 = 1e-6 + 0j
k = transfer_stiffness_direct(k21 * u1, u1)     # direct method: k2,1 = F2,b/u1
res = TransferStiffnessResult(frequencies=freqs, transfer_stiffness=k)
res.report(
    "transfer_stiffness.pdf",
    metadata=ReportMetadata(
        specimen="Rubber vibration isolator",
        measurement_standard="ISO 10846-2",
    ),
)   # one-page fiche (needs phonometry[report,plot])
```

The example fiche is regenerated with `make reports` and kept in the
repository. Click the preview to open the PDF:

[![ISO 10846 dynamic-transfer-stiffness example report: a metadata header, a table of the FRF characteristic points (the determination method, the frequency range and the low-frequency stiffness, level and loss factor) beside the transfer-stiffness level spectrum, and the boxed low-frequency level](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10846_transfer_stiffness_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10846_transfer_stiffness_example.pdf)

*Dynamic-transfer-stiffness fiche (`TransferStiffnessResult.report`): the FRF
characteristic points and the transfer-stiffness level spectrum.*

## References

- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-borne
  sound: Structural vibrations and sound radiation at audio frequencies*
  (3rd ed.). Springer. ISBN 978-3-540-22696-3.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728).
  Vibration isolation theory: why an isolator's performance depends on the
  source and receiver mobilities, the physics behind the blocked-force
  characterisation.
- International Organization for Standardization. (2008). *Acoustics and
  vibration — Laboratory measurement of vibro-acoustic transfer properties of
  resilient elements — Part 1: Principles and guidelines* (ISO 10846-1:2008).
  [iso.org catalogue](https://www.iso.org/standard/38936.html).
  The principles part of the series: the blocking-force idealisation and the
  FRF relations this page implements.

## Standards

ISO 10846 (parts 1-5), *Acoustics and vibration — Laboratory
measurement of vibro-acoustic transfer properties of resilient elements*: the
dynamic transfer stiffness $k_{2,1} = F_{2,b}/u_1$ and its FRF relations (Part 1,
clause 5 and Annex A / Table A.2), the level $L_k$ re 1 N/m and the loss factor
(Parts 2 and 3, clauses 3.8/3.17), the direct method (Part 2) and the indirect
method $k_{2,1} = -(2\pi f)^2 (m_2 + m_f) T$ (Part 3, Formula 1) with its
validity
conditions (Part 3, clause 6: Inequalities 2 and 3; clause 7.6 linearity) and
the blocking-force approximation (Part 1, Eqs. 6/7). Parts 4 and 5 extend the
same quantities to elements other than supports and to the driving-point
low-frequency method. Conformance is anchored on the standard's closed-form
definitions: the level of a decade of stiffness, the indirect inertia relation,
the Table-A.2 identity $k = j\omega Z$, the
$|T| = 0.1 \leftrightarrow \Delta L_{1,2} = 20\ \text{dB}$ validity
limit and its 1 dB (12 %) accuracy bound, the Eq. (6) force ratio $1/1.1$, and
the 7.6 linearity criterion.
