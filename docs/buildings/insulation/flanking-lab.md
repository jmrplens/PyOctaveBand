← [Documentation index](../../README.md)

# Laboratory Flanking Transmission (ISO 10848)

EN 12354 predicts a building from its junctions, and ISO 10848 is how a
junction gets its number. Two structurally decoupled elements meet in a
laboratory junction rig; shakers and tapping machines drive one, and the
velocity level differences across the junction condense into the
**vibration reduction index** $K_{ij}$, together with the overall flanking
descriptors $D_\mathrm{n,f}$ (airborne) and $L_\mathrm{n,f}$ (impact). This guide covers
that measurement chain: the direction-averaged velocity level difference,
the equivalent absorption lengths, the SEA validity checks of Part 4 and
the three accredited fiches. The empirical junction values a prediction
falls back on when no measurement exists live in
[Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md); the
wave-theory $K_{ij}$ of ideal plate junctions is derived in
[Bending-wave transmission at plate junctions](../../vibration/structural/junction-transmission.md).

## The vibration reduction index and the flanking descriptors

ISO 10848:2006/2010 is the laboratory method that **measures** the junction
**vibration reduction index** $K_{ij}$ that the [EN 12354 prediction](../design/insulation-prediction.md) takes
as an input, together with the overall flanking descriptors $D_\mathrm{n,f}$
(airborne) and $L_\mathrm{n,f}$ (impact). It is the measurement counterpart of the
empirical `junction_vibration_reduction()` of that prediction.

**Vibration reduction index (Formula (13)).**
$K_{ij} = \overline{D}_{v,ij} + 10\log_{10}\!\big(l_{ij} / \sqrt{a_i a_j}\big)$ dB, from
the direction-averaged velocity level difference
$\overline{D}_{v,ij} = \tfrac{1}{2}(D_{v,ij} + D_{v,ji})$ (Formula (11), which
makes $K_{ij}$ symmetric), the common-edge junction length $l_{ij}$ and the
**equivalent absorption lengths** $a_j = 2.2\pi^2 S_j /(T_{\mathrm{s},j} c_0)\sqrt{f_\mathrm{ref}/f}$
(Formula (12), $f_\mathrm{ref} = 1000$ Hz). For lightweight well-damped elements
$a_j = S_j / l_0$ ($l_0 = 1$ m) and Formula (13) reduces to the simplified
Formula (14). The related **total loss factor** is $\eta = 2.2/(f T_\mathrm{s})$.

**Overall descriptors.** $D_\mathrm{n,f} = L_1 - L_2 - 10\log_{10}(A/A_0)$ (Formula (4),
airborne) and $L_\mathrm{n,f} = L_2 + 10\log_{10}(A/A_0)$ (Formula (5), tapping machine),
$A_0 = 10\ \text{m}^2$; their $D_\mathrm{n,f,w}$ / $L_\mathrm{n,f,w}$ single numbers reuse the
ISO 717 rating engines. The single-number $\overline{K}_{ij}$ is the arithmetic
mean over 200–1250 Hz for one-third-octave bands, or over 125–1000 Hz for
octave bands (Annex A).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/flanking_transmission_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/flanking_transmission.svg" alt="ISO 10848 junction vibration reduction index Kij rising across one-third-octave bands from 100 Hz to 5000 Hz for a rigid T-junction of two heavy walls, with the single-number mean Kij over 200-1250 Hz drawn as a dashed line" width="80%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630,
         800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
# Direction-averaged velocity level difference of a rigid T-junction (dB):
dv = np.array([4.5, 4.8, 5.2, 5.6, 6.0, 6.5, 7.0, 7.6, 8.1, 8.7,
               9.2, 9.8, 10.3, 10.9, 11.4, 11.9, 12.3, 12.7])
res = building.vibration_reduction_index(
    dv, junction_length=4.0, area_i=12.0, area_j=10.0, frequency=freqs,
    structural_reverberation_time_i=0.35, structural_reverberation_time_j=0.40,
)
print(res.single_number)   # mean Kij over 200-1250 Hz (Annex A)
res.plot()
plt.show()
```

</details>

**Validity.** $K_{ij}$ rests on a statistical-energy-analysis simplification:
`strong_coupling_satisfied()` checks the Formula (15) inequality and, for the
heavy junctions of Part 4, `modal_density()`, `band_mode_count()` and
`modal_overlap_factor()` (Formulae (5)/(4)/(6)) quantify where the mode count
is too low for $K_{ij}$ to be reliable. Pass the per-band modal overlap factor
to `vibration_reduction_index(..., modal_overlap=M)`: bands with $M < 0.25$
are flagged in `result.bracketed` and excluded from the single-number
$\overline{K}_{ij}$, as Part 4 Clause 9 requires. Because ISO 10848 contains no
worked numeric example, conformance is anchored on closed-form identities
(simplified $K_{ij}$, $a_j$ at $f_\mathrm{ref}$, $\eta$).

Those checks are the only enforcement there is. The acquisition itself —
shielding the other elements, a shield meeting $\Delta R_\text{min}$, the
position counts and separations, the accelerometer mass-loading inequality,
$T_\mathrm{s}$ measured on the elements in their installed state rather than assumed —
is the operator's responsibility, and nothing here checks any of it.

```python
import numpy as np
from phonometry import building

freqs = [200, 250, 315, 400, 500, 630, 800, 1000, 1250]
lij, s_i, s_j = 4.0, 12.0, 10.0     # junction length (m), element areas (m^2)
ts = np.linspace(0.30, 0.10, 9)     # structural reverberation time Ts (s)
dv_ij = [5.6, 6.0, 6.5, 7.0, 7.6, 8.1, 8.7, 9.2, 9.8]    # element i excited (dB)
dv_ji = [6.4, 6.8, 7.3, 7.8, 8.4, 8.9, 9.5, 10.0, 10.6]  # element j excited (dB)

# Kij from both excitation directions (symmetric via the direction average):
dbar = building.direction_averaged_level_difference(dv_ij, dv_ji)
res = building.vibration_reduction_index(dbar, lij, s_i, s_j, frequency=freqs,
                                structural_reverberation_time_i=ts,
                                structural_reverberation_time_j=ts)
res.k_ij           # Kij per band (Formula (13))
res.single_number  # mean Kij over 200-1250 Hz, or None without the band set
res.octave_bands() # Kij in octave bands (its single number averages 125-1000 Hz)

# Overall airborne flanking descriptor and a Part-4 modal-overlap validity check:
dnf = building.normalized_flanking_level_difference(np.full(9, 75.0), np.full(9, 42.0),
                                           absorption_area=np.full(9, 12.0))
m = building.modal_overlap_factor(s_i, critical_frequency=85.0,
                         structural_reverberation_time=ts)
res_m = building.vibration_reduction_index(dbar, lij, s_i, s_j, frequency=freqs,
                                  modal_overlap=m)   # M < 0.25 bands bracketed
res_m.bracketed    # per-band flags; bracketed bands leave the single number

# With 16 one-third-octave (or 5 octave) bands, dnf.plot() draws Dn,f vs the
# shifted ISO 717-1 reference with Dn,f,w annotated (needs matplotlib):
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/flanking_level_difference_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/flanking_level_difference.svg" alt="Normalized flanking level difference Dn,f per one-third-octave band against the shifted ISO 717-1 reference curve, with the unfavourable deviations shaded and the Dn,f,w rating annotated" width="80%"></picture>

*The overall flanking descriptor $D_\mathrm{n,f}$ is an airborne quantity, so its
single number $D_\mathrm{n,f,w}$ comes from the unchanged ISO 717-1 engine; it
drops straight into the EN 12354-1 model as the flanking-path datum of the
tested junction (the impact counterpart $L_\mathrm{n,f}$ rates per ISO 717-2 the
same way).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import building

# A lightweight junction in the laboratory: source-room level, receiving-room
# level over the flanking path, and the receiving-room absorption area.
l1 = np.full(16, 80.0)
dnf_target = np.array([48, 49, 50, 51, 52, 54, 55, 57,
                       58, 59, 60, 61, 62, 63, 64, 65], dtype=float)
dnf = building.normalized_flanking_level_difference(
    l1, l1 - dnf_target, absorption_area=np.full(16, 10.0)
)

# One line — Dn,f vs the shifted ISO 717-1 reference:
dnf.plot()
plt.show()

# By hand, from the rating the result carries:
w = dnf.rating
fig, ax = plt.subplots()
ax.semilogx(w.band_centers, dnf.d_n_f, "o-", label="Dn,f (flanking)")
ax.semilogx(w.band_centers, w.shifted_reference, "s--",
            label="shifted reference")
ax.fill_between(w.band_centers, w.measured, w.shifted_reference,
                where=w.measured < w.shifted_reference, interpolate=True,
                alpha=0.3, label="unfavourable deviations")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Normalized flanking level difference [dB]")
ax.set_title(f"Dn,f,w = {w.rating} dB  (C={w.c:+d}; Ctr={w.ctr:+d})")
ax.legend()
plt.show()
```

</details>

## Suspended ceilings: the plenum flanking path (ISO 140-9, Vigran 9.2.3)

Two offices separated by a partition that stops at the suspended ceiling share
one continuous plenum above it. Sound leaves the source room through the
ceiling tiles, travels sideways over the partition and comes back down through
the tiles of the receiving room. That path is often the weakest link in an
open-plan fit-out, and it is not what a partition's $R_\mathrm{w}$ describes.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/ceiling_plenum_flanking_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/ceiling_plenum_flanking.svg" alt="Two panels: the ceiling-plenum flanking reduction index Rcl per octave band for two plenum depths against the sum of the two ceiling reduction indices, and the normalized ceiling attenuation of an accredited ASTM E1414 test report with the fitted ASTM E413 contour and its shaded deficiencies giving CAC 34" width="92%"></picture>

*Left: the plenum charges a fixed penalty against the sum of the two ceilings,
and a deeper plenum pays some of it back. Right: the same measured quantity
rated as a ceiling attenuation class.*

**The one-dimensional model.** Mechel's one-dimensional variant, as presented
by Vigran in Section 9.2.3, treats the plenum as a duct lined on one side. The
ceiling on each side has a transmission factor $\tau_\mathrm{S} = \tau_\mathrm{S,pl}\tau_\mathrm{S,a}$
(plates times any plenum absorber, Eq. (9.14)); the injected power splits,
a fraction $s_\mathrm{S}$ heading for the partition, and decays as $\exp(-mx)$ with the
power attenuation coefficient $m = 2\,\mathrm{Re}\{\Gamma\} = -2\,\mathrm{Im}\{k'\}$
(Eqs. (9.15) and (9.16)). Integrating over the ceiling length on both sides
gives Eq. (9.18), whose receiving side carries the leakage back into the room,
$m'_\mathrm{R} = m_\mathrm{R} + s_\mathrm{R}\tau_\mathrm{R}/h$ (Eq. (9.17)). Vigran prints the *unprimed* $m_\mathrm{R}$ in
that expression's denominator; that is a misprint, recorded in
[the errata register](../../ERRATA.md), and the derived $m'_\mathrm{R}$ is used here. Read
literally, the printed form is non-monotonic in the plenum damping (adding
absorber would predict a worse path than none) and unbounded as $m_\mathrm{R} \to 0$. For a plenum with little
attenuation and $s_\mathrm{S} = s_\mathrm{R} = 0{,}5$ that collapses to the compact form that
makes the geometry visible (Eqs. (9.19) and (9.20)):

$$
R_\mathrm{cl} = R_\mathrm{S} + R_\mathrm{R} - 10\log_{10}\!\left[\frac{\varepsilon^2 L_\mathrm{R}}{4h}\right],
$$

with $\varepsilon = 1$ for totally absorbing plenum sidewalls and
$\varepsilon = 2$ for totally reflecting ones. A deep plenum helps, a long room
hurts, and doubling the tile insulation helps twice over because $R_\mathrm{S}$ and
$R_\mathrm{R}$ both appear. Referred to the partition area instead of the ceiling,
$R_\mathrm{cl,p} = R_\mathrm{cl} + 10\log_{10}(H_\mathrm{S}/L_\mathrm{S})$ (Eq. (9.13)), which is what lets the
ceiling path be added to the direct path as transmission factors.

```python
from phonometry import (
    partition_referenced_reduction_index,
    plenum_flanking_reduction_index,
)

freqs = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
ceiling = [17.0, 21.0, 25.0, 29.0, 32.0, 30.0, 38.0]   # 9.5 mm plasterboard

# Vigran's example geometry: LS = LR = 4.75 m, plenum 0.43 m, reflecting walls.
res = plenum_flanking_reduction_index(
    ceiling, ceiling, ceiling_length=4.75, plenum_height=0.43, frequency=freqs
)
print(round(res.geometry_term, 1))    # 10.4 dB charged against RS + RR
res.plot()                            # Rcl against the two ceilings

# A lined plenum attenuates the sideways path (Eq. (9.18) instead of (9.20)):
damped = plenum_flanking_reduction_index(
    ceiling, ceiling, ceiling_length=4.75, plenum_height=0.43,
    attenuation_source=[0.5] * 7, attenuation_receiving=[0.5] * 7,
)

# Referred to the partition instead of the ceiling (Eq. (9.13)):
print(partition_referenced_reduction_index(res.reduction_index, 2.7, 4.75))
```

**The measured quantity.** A ceiling is not rated by $R_\mathrm{cl}$ but by the
**normalized ceiling attenuation** $D_\mathrm{n,c} = D - 10\log_{10}(A/A_0)$ (ISO 140-9:1985
clause 3.3), with $A$ the receiving-room equivalent absorption area and the
reference $A_0 = 10\ \text{m}^2$. The facility has two rooms of at least
50 m³ whose volumes differ by at least 10 %, a dividing wall tapered to at most
100 mm at the top, and a plenum 650 mm to 760 mm deep with one sidewall and
both end walls lined; the standard prints the required lining absorption
$\alpha_\mathrm{s} \ge 0{,}65$ at 125 Hz and $\ge 0{,}80$ from 250 Hz to 4000 Hz, and
requires $\alpha < 0{,}10$ on the other sidewall and on the plenum ceiling.
The North American counterpart, ASTM E1414, uses $A_0 = 12\ \text{m}^2$, so an
ASTM value runs about $10\log_{10}(12/10) = 0{,}79$ dB higher for the same rooms.

ISO rates $D_\mathrm{n,c}$ with the ISO 717-1 curve, giving $D_\mathrm{n,c,w}$; ASTM E1414
rates it through ASTM E413 as the **ceiling attenuation class**. E413 rounds the
data to the nearest integer (clause 5.2), raises its reference contour in 1 dB
steps while the sum of the deficiencies stays at or below 32 dB and no single
deficiency exceeds 8 dB (clauses 5.3 and 5.4), and reads the rating off the
shifted contour at 500 Hz (clause 5.5).

```python
from phonometry import (
    ceiling_attenuation_class,
    normalized_ceiling_attenuation,
    weighted_rating,
)

# ASTM E1414 normalizes to A0 = 12 m2, ISO 140-9 to A0 = 10 m2.
dnc = normalized_ceiling_attenuation(l1, l2, absorption, reference_area=12.0)

# A 28 mm perforated plaster acoustic tile, measured to ASTM E1414 (CAC 34).
dnc = [14.4, 18.6, 21.7, 24.1, 23.4, 30.3, 33.7, 35.2,
       41.6, 44.2, 42.1, 36.8, 35.7, 36.0, 36.9, 37.9]
res = ceiling_attenuation_class(dnc)
print(res.rating, res.deficiency_sum, res.max_deficiency)   # 34, 27.0, 7.0
res.plot()                                                  # Dn,c vs the contour

# The ISO single number of the same spectrum, shifted to A0 = 10 m2:
iso = [v - 0.79 for v in dnc]
print(weighted_rating(iso).rating)                          # Dn,c,w
```

## ISO 10848 flanking-transmission reports (`.report()`)

Each of the three results renders a one-page PDF fiche. `VibrationReductionResult.report()`
writes a **junction characterization** report of $K_{ij}$
(ISO 10848-1:2006): the standard-basis line, an optional metadata header, the
per-band $K_{ij}$ table beside the $K_{ij}(f)$ curve and a
boxed single-number mean $K_{ij}$ over the Annex A band range, with the
count of averaged and bracketed bands. Bands bracketed for poor modal overlap
($M < 0.25$, ISO 10848-4:2010 Clause 9) print their value in brackets and are
excluded from the mean; `verbose=True` adds a column stating whether each band
enters the mean.

`FlankingLevelDifferenceResult.report()` and `FlankingImpactLevelResult.report()`
write **measurement** reports of the overall descriptors $D_\mathrm{n,f}$
(airborne) and $L_\mathrm{n,f}$ (impact, tapping machine), reusing the same
two-panel insulation layout: the per-band quantity beside the
measured-versus-shifted-ISO 717-reference curve and the boxed single number
$D_\mathrm{n,f,w}$ (C; Ctr) (ISO 717-1) or $L_\mathrm{n,f,w}$ (CI) (ISO 717-2).
`verbose=True`
annexes the ISO 717 evaluation per band (the value, the shifted reference and
the unfavourable deviation). A `requirement` supplied on the `ReportMetadata`
adds a verdict ($D_\mathrm{n,f,w}$ passes at or above it, $L_\mathrm{n,f,w}$ at or below
it), and
`language="es"` renders every fiche in Spanish. reportlab is required, and
matplotlib too for the figure the fiche embeds (`pip install
"phonometry[report,plot]"`).

```python
import numpy as np
from phonometry import building, ReportMetadata

freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630,
         800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
dv = np.array([4.5, 4.8, 5.2, 5.6, 6.0, 6.5, 7.0, 7.6, 8.1, 8.7,
               9.2, 9.8, 10.3, 10.9, 11.4, 11.9, 12.3, 12.7])
m = np.full(18, 1.0); m[:3] = 0.1                        # bracket the low bands
kij = building.vibration_reduction_index(
    dv, junction_length=4.0, area_i=12.0, area_j=10.0, frequency=freqs,
    structural_reverberation_time_i=0.35, structural_reverberation_time_j=0.40,
    modal_overlap=m,
)
kij.report("Kij.pdf", metadata=ReportMetadata(specimen="Rigid T-junction"))

l1 = np.full(16, 80.0)
dnf = np.array([48, 49, 50, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65],
               dtype=float)
dres = building.normalized_flanking_level_difference(
    l1, l1 - dnf, absorption_area=np.full(16, 10.0)
)
dres.report("Dnf.pdf", metadata=ReportMetadata(requirement=55.0))   # Dn,f,w (C; Ctr)

recv = np.array([58, 57, 56, 55, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32],
                dtype=float)
lres = building.normalized_flanking_impact_level(recv, absorption_area=np.full(16, 10.0))
lres.report("Lnf.pdf", metadata=ReportMetadata(requirement=55.0))   # Ln,f,w (CI)
```

The example fiches are regenerated with `make reports` and kept in the
repository. Click a preview to open the PDF:

[![ISO 10848-1 junction report: metadata header, per-band Kij table beside the Kij curve, boxed single-number mean Kij with the count of averaged and bracketed bands](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10848_kij_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10848_kij_example.pdf)

*Vibration reduction index fiche (`VibrationReductionResult.report`), mean $K_{ij}$.*

[![ISO 10848-2 airborne flanking report: per-band Dn,f table beside the measured-versus-shifted-reference curve, boxed Dn,f,w (C; Ctr) and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10848_dnf_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10848_dnf_example.pdf)

*Flanking level difference fiche (`FlankingLevelDifferenceResult.report`), $D_\mathrm{n,f,w}$ (C; Ctr).*

[![ISO 10848-2 impact flanking report: per-band Ln,f table beside the measured-versus-shifted-reference curve, boxed Ln,f,w (CI) and a PASS verdict](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10848_lnf_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iso10848_lnf_example.pdf)

*Flanking impact level fiche (`FlankingImpactLevelResult.report`), $L_\mathrm{n,f,w}$ (CI).*

## References

- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  The reference monograph for flanking transmission and the vibration
  reduction index, including the statistical-energy-analysis footing of the
  validity checks.
- International Organization for Standardization. (2006). *Acoustics —
  Laboratory measurement of the flanking transmission of airborne and
  impact sound between adjoining rooms — Part 1: Frame document*
  (ISO 10848-1:2006).
  [iso.org catalogue](https://www.iso.org/standard/38284.html).
  The frame document: definitions, the vibration reduction index, the
  equivalent absorption length and the normalized flanking descriptors.
- International Organization for Standardization. (2010). *Acoustics —
  Laboratory measurement of the flanking transmission of airborne and
  impact sound between adjoining rooms — Part 4: Application to junctions
  with at least one heavy element* (ISO 10848-4:2010).
  [iso.org catalogue](https://www.iso.org/standard/45360.html).
  The heavy-junction application with the modal-density and modal-overlap
  validity checks behind the bracketed bands.

- International Organization for Standardization. (1985). *Acoustics —
  Measurement of sound insulation in buildings and of building elements —
  Part 9: Laboratory measurement of room-to-room airborne sound insulation of
  a suspended ceiling with a plenum above it* (ISO 140-9:1985).
  [iso.org catalogue](https://www.iso.org/standard/3944.html).
  The normalized ceiling attenuation $D_\mathrm{n,c}$ (clause 3.3) and the plenum
  test facility.
- ASTM International. (2022). *Classification for Rating Sound Insulation*
  (ASTM E413-22).
  [astm.org catalogue](https://www.astm.org/e0413-22.html).
  The reference contour and fitting rules that ASTM E1414 invokes for the
  ceiling attenuation class.
- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  Section 9.2.3 presents Mechel's one-dimensional plenum model.

## Standards

ISO 10848-1:2006, ISO 10848-2:2006, ISO 10848-3:2006 and ISO 10848-4:2010,
which cover the laboratory measurement of flanking transmission: the
vibration reduction index $K_{ij}$, the equivalent absorption length, the
normalized flanking descriptors $D_\mathrm{n,f}$ / $L_\mathrm{n,f}$ and the modal-overlap
validity checks that feed the EN 12354 prediction. Parts 2, 3 and 4 differ in
which junction and specimen types they apply to; phonometry implements only
the Part 1 formulae generically, plus the Part 4 modal-overlap validity
check, not the facility-specific test setups the other parts describe.
Because ISO 10848 contains no worked numeric example, conformance is
anchored on closed-form identities (simplified $K_{ij}$, $a_j$ at
$f_\mathrm{ref}$, $\eta$).


The suspended-ceiling branch adds ISO 140-9:1985, which defines the normalized
ceiling attenuation $D_\mathrm{n,c}$ and its laboratory, and ASTM E413-22, the rating
classification that ASTM E1414 invokes for the ceiling attenuation class. The
one-dimensional plenum model itself is not standardized: it comes from Mechel
through Vigran Section 9.2.3, and every published output of it is a figure, so
it is anchored on its closed forms and on structural properties a wrong
reading breaks: monotonicity in the plenum damping, the bound
$\tau_\mathrm{cl} \le 1$, and the convergence of Eq. (9.18) to Eq. (9.20). The
$D_\mathrm{n,c}$ measurement chain and the class are anchored on accredited ASTM E1414
laboratory reports. The plenum propagation constant $k'$ of a lined duct is an
input here, not a prediction.

## See also

- [Predicting Sound Insulation (EN 12354)](../design/insulation-prediction.md): the
  flanking model that consumes the measured $K_{ij}$, and its empirical
  Annex E junction values.
- [Bending-wave transmission at plate junctions](../../vibration/structural/junction-transmission.md):
  the wave-theory transmission coefficients behind an ideal junction's
  $K_{ij}$.
- [Laboratory Insulation Measurement](insulation-lab.md): the ISO 10140
  suite whose direct transmission these flanking paths bypass.
- [Insulation Ratings (ISO 717)](insulation-ratings.md): the
  reference-curve engines behind $D_\mathrm{n,f,w}$ and $L_\mathrm{n,f,w}$.
- [Room-to-room and open-plan acoustics](../rooms/open-plan-acoustics.md): the
  open-plan context the ceiling flanking path usually appears in.
- API reference: [`building.measurement.flanking_transmission`](https://jmrplens.github.io/phonometry/reference/api/building/flanking-transmission/)
  and [`building.prediction.ceiling_plenum`](https://jmrplens.github.io/phonometry/reference/api/building/ceiling-plenum/).
