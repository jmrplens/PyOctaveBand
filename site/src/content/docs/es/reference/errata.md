---
title: "Erratas de las fuentes publicadas"
description: "Defectos encontrados en las normas, los documentos de guía y los libros de los que parte la librería: erratas de imprenta, ejemplos resueltos que contradicen su propio articulado y qué hace la librería con cada uno."
---

Implementar una norma en sala limpia significa volver a deducir cada fórmula,
constante y ejemplo resuelto a partir del documento fuente y no del código de
otra persona. Hecho sobre cientos de documentos, ese proceso encuentra
defectos en las propias fuentes: un ejemplo resuelto que contradice su
articulado, una constante a la que la composición tipográfica le comió un
dígito, una referencia cruzada que apunta a la ecuación equivocada.

Esta página es el registro de esos hallazgos. Cada entrada nombra la edición
impresa y el punto exacto, cita lo que dice el documento, muestra por qué no
puede ser correcto, aporta la evidencia independiente y declara qué lectura
implementa la librería y qué test de regresión la fija. Un defecto listado
aquí nunca es un defecto del *método*: en todos los casos la lectura
pretendida se ha podido establecer a partir del propio documento o de la
física.

Léela junto al
[informe de conformidad](/phonometry/es/reference/conformance/), que muestra
los números que calcula la librería; esta página explica el puñado de sitios
donde lo que está mal es el valor esperado impreso.

:::note
El registro que aparece más abajo se mantiene en inglés y se reproduce aquí
tal cual, para que el texto de cada entrada coincida palabra por palabra con
el que se ha comunicado o se comunicará a los organismos emisores. Esta
introducción sí está traducida.
:::

El registro vive en
[`docs/ERRATA.md`](https://github.com/jmrplens/phonometry/blob/main/docs/ERRATA.md)
y se trasplanta aquí en tiempo de compilación con `make site-reports`, así que
ambos no pueden discrepar.

<!-- BEGIN GENERATED BODY - transplanted from docs/ERRATA.md by scripts/generate_site_reports.py (`make site-reports`). Edit the source document, never the text below. -->

During the clean-room implementation of this library, every formula, constant
and worked example is re-derived and recomputed independently from the source
documents. That process occasionally surfaces defects in the sources
themselves: misprints, worked examples that contradict their own normative
text, and ambiguous wording. This file records each confirmed case with the
evidence, what the library does about it, and whether it has been reported.

The registry covers every kind of published source the library implements
from: standards (ISO, IEC, EN), guidance documents and technical reports
(EASA, ECAC, NRL), textbooks and journal papers. Non-normative sources are
marked as such in their entry.

Entries describe the specific printed editions cited. A defect listed here is
not a defect of the method; in every case the intended reading could be
established from the document itself or from physics, and the library
implements that reading. Where the reading changes a number the library
reports, the entry names the check or test that pins it; where the defect is a
label, a cross-reference or a table the library never reads, the entry records
that no change was required.

Status legend: **unreported** (recorded here only) / **reported** (submitted
to the issuing body, with date and reference).

---

## ISO 717-2:2020, Annex C, example C.1 (CI of the bare floor)

- **Location:** Annex C, Table C.1 and the accompanying CI computation.
- **The print:** the 2020 reprint states CI = −10 for the bare-floor example.
- **The problem:** its own normative clause A.2.1 defines CI from the energy
  sum over 100 Hz to 2500 Hz (the first fifteen one-third-octave bands). The
  2020 reprint's value only reproduces if the 3150 Hz band is included in the
  sum (83,5238 dB, rounded 84), contradicting A.2.1. The correct sum over
  100 to 2500 Hz is 83,2613 dB, rounded 83, giving CI = −11.
- **Evidence:** independent recomputation of both sums from the printed
  per-band levels; the 2013 edition of the same example prints CI = −11.
- **Library behaviour:** implements A.2.1 as written and pins CI = −11 with
  the 2013 print as the oracle ([`tests/reference_data.py`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data.py),
  conformance check
  "ISO 717-2 Annex C, Table C.1").
- **Status:** unreported.

## ISO 717-2:2020, Annex C, example C.2 (covered floor: 800 Hz value and CI chain)

- **Location:** Annex C, Table C.2 (ΔLw / ΔLlin worked example).
- **The print:** (a) the 800 Hz reference-floor value is printed as 71,0 dB;
  (b) the CI line prints "Ln,sum = 75,252 7… = 75 dB" and "CI = 75 − 15 −
  63 = −3 dB", feeding "ΔLlin = 78 − 11 − (63 − 3) = 7 dB".
- **The problem:** two independent defects. (a) The normative Table 4
  reference floor is 71,5 dB at 800 Hz; the 71,0 dB row is a misprint,
  though after rounding it does not change the example's CI (the correct
  energy sum rounds to 76 dB with either value). (b) The printed
  75,2527 dB is exactly the energy sum of the *wrong column over the wrong
  range*: the measured floor "with covering" over all sixteen bands
  100 Hz to 3150 Hz. A.2.1 defines CI from the reference floor with
  covering (the "Ln,r,0 − ΔL" column) over 100 Hz to 2500 Hz (15 bands),
  which gives 75,674 dB (printed chain) or 75,710 dB (Table 4 values) —
  both round to 76 dB, so CI,r = 76 − 15 − 63 = −2 either way, giving
  CI,Δ = −11 − (−2) = −9 and ΔLlin = 6 dB, not the printed −3 / −8 / 7 dB
  chain.
- **Evidence:** independent recomputation of every candidate sum from the
  printed per-band values; the printed 75,2527 reproduces to all printed
  digits only as the 16-band sum of the with-covering column.
- **Library behaviour:** derives the covered reference floor from the
  normative Table 4 values and sums per A.2.1, pinning ΔLw = 15 dB and
  CI,Δ = −9; the conformance check notes the provenance explicitly.
- **Status:** unreported.

## ISO 2631-5:2018, Annex C, NOTE 5 (female worked example)

- **Location:** Annex C, NOTE 5 (64 kg female, mz = 0,025 MPa/(m/s²)).
- **The print:** R = 0,97.
- **The problem:** exact recomputation of Formula (C.3) with the note's own
  inputs (mz = 0,025, age coefficient 0,039, b = 20, n = 20, N = 120) gives
  R = 0,9621, which rounds to 0,96. The same code reproduces the male example
  exactly (R = 1,2200 = printed 1,22), and the note's Sd = 1,40 MPa matches
  the exact 1,3992, so the discrepancy is confined to the last digit of the
  printed female R.
- **Evidence:** hand recomputation of the C.3 sum, term by term.
- **Library behaviour:** computes the exact value; the test anchor keeps the
  printed 0,97 with a tolerance that documents the recomputed 0,9621.
- **Status:** unreported.

## EN 12354-1:2000 Annex E.5 / ISO 12354-1:2017 E.3.4 (K24 clamp misprint)

- **Location:** EN 12354-1:2000, Annex E, clause E.5, and ISO 12354-1:2017,
  E.3.4 NOTE 4 (wall junction with flexible interlayers).
- **The print:** the bound on the K24 junction term is printed as
  "0 ≤ K24 ≤ −4 dB", an empty interval; the 2017 edition repeats the 2000
  misprint verbatim.
- **The problem:** the interval is impossible as printed; the accompanying
  figure and the physics (the term is a reduction bounded below) indicate
  −4 dB ≤ K24 ≤ 0 dB.
- **Evidence:** page render of the printed clause; corroboration against the
  figure's curve family.
- **Library behaviour:** implements the clamp as −4 ≤ K24 ≤ 0 with a misprint
  note in the docstring.
- **Status:** unreported.

## EN 12354-1:2000, Figure E.9 (E.7) (K24 stated in the figure-axis mass ratio)

- **Location:** Annex E, Figure E.9 / Formula (E.7) (junction of lightweight
  double leaf wall and homogeneous elements), the K24 line.
- **The print:** K24 = 3,0 − 14,1 M + 5,7 M² dB (for m2/m1 > 3), under a
  figure whose x-axis is m2/m1.
- **The problem:** Annex E defines M per transmission path as
  M = lg(m'⊥,i/m'i) (perpendicular element over the element carrying the
  path). The K24 path 2→4 is carried by the homogeneous element (m2 = m4)
  with the leaf (m1) perpendicular, so the per-path M is lg(m1/m2) — but the
  printed K24 line only matches its own figure's curve when M is read as the
  x-axis variable lg(m2/m1) (e.g. −2,4 dB at m2/m1 = 3, −5,4 dB at 10). Read
  with the annex's declared M, the line contradicts the figure by
  28,2·|lg(m2/m1)| dB. The same edition's other K24 line (Figure E.5,
  Formula (E.5)) *does* follow the declared per-path M, so the two K24
  prints of the 2000 edition silently use different conventions.
  ISO 12354-1:2017 E.3.5 prints the relation consistently in the per-path
  convention of its Formula (E.3), K24 = 3,0 + 14,1 M + 5,7 M²; the two
  editions agree numerically (an earlier revision of this entry read the
  2017 print as a sign misprint — re-derivation against both editions'
  figures shows it is a convention recast, not a defect of the 2017 text).
- **Evidence:** page renders of both editions (EN 12354-1:2000 printed
  pp. 43, 46, 48; ISO 12354-1:2017 printed pp. 43, 46-47); numerical
  evaluation of both forms against the Figure E.9 curve.
- **Library behaviour:** implements the per-path convention uniformly
  (`junction_vibration_reduction`, mass_ratio = m'⊥,i/m'i for every branch),
  so the E.7 double-leaf branch takes leaf-over-homogeneous ratios below 1/3
  and evaluates 3,0 + 14,1 M + 5,7 M².
- **Status:** unreported.

## EN 12354-2:2000, Formula (3) vs Annex E.3 (standardized impact level)

- **Location:** Formula (3) and worked example E.3.
- **The print:** Formula (3) defines L'nT = L'n − 10 lg(0,16·V/(A0·T0)), which
  reduces exactly to L'n − 10 lg(0,032·V), i.e. a reference volume of
  31,25 m³. Annex E.3 states "from equation (3): L'nT,w = L'n,w − 10 lg(V/30)".
- **The problem:** the annex's V/30 is a rounding of the formula's own
  constant; the two differ by a constant 0,177 dB.
- **Evidence:** direct algebra; both variants recomputed for the E.3 case
  (42,959 vs 42,782 dB, both rounding to 43 in that example).
- **Library behaviour:** implements the exact 0,032·V form and documents the
  annex's rounding.
- **Status:** unreported.

## EN 12354-3:2000, Annex F (worked example internal inconsistencies)

- **Location:** Annex F worked example.
- **The print:** (a) the printed D2m,nT row equals R' + 1,5 dB; (b) the
  printed high-frequency-band row is inconsistent with the example's own
  partial indices.
- **The problem:** Formula (13) with the example's own inputs (V = 50 m³,
  S = 11,3 m², T0 = 0,5 s) gives D2m,nT = R' + 1,69 dB, not +1,5 dB; and the
  high-band row cannot be reproduced from the example's stated partial
  results (the self-consistent values are 35,8/38,0 dB).
- **Evidence:** recomputation of Formula (13) and of the partial-index chain.
  The example's single-number result D2m,nT,w = 33 dB is insensitive to both
  and still reproduces.
- **Library behaviour:** implements Formula (13); the test data notes both
  inconsistencies next to the affected anchors.
- **Status:** unreported.

## ISO 12354-1:2017 Table L.3 / ISO 12354-2:2017 Table G.3 (perimeter sums)

- **Location:** the input-data block below Table L.3 (printed p. 81) and the
  identical block below Table G.3 (printed p. 38), which lists the perimeter
  absorption sum `Σ lk αk` of Formula (C.1) for the worked example.
- **The print:** one value per element *type*: separating floor 2,364 m
  (S = 20 m²), external wall 2,375 m (S = 11 m²), internal wall 1,840 m
  (S = 13,75 m²).
- **The problem:** Formula (C.1) needs one sum per *element*, and the example
  has five elements with three different areas. Only two of the three printed
  values reproduce the columns they are supposed to drive: 2,375 m with
  S = 11 m² gives external wall 1 exactly, and 1,840 m with S = 13,75 m²
  gives internal wall **2** exactly. The separating floor's printed 2,364 m
  does not reproduce its own column at any band (0,074 9 against the printed
  0,083 1 at 50 Hz, 0,026 4 against 0,029 0 at 500 Hz); 2,659 m does, at every
  band. The two elements with no printed value need 2,548 m (external wall 2,
  S = 13,75 m²) and 1,636 m (internal wall 1, S = 11 m²).
- **Evidence:** all five sums re-derived from Formula (C.4),
  `αk = Σj √(fc,j/fref) 10^(−Kij/10)`, over the example's own junction
  geometry with the unrounded Annex E indices: 2,659 / 2,375 / 2,548 / 1,636 /
  1,839 m. The derivation returns the two printed values that are
  self-consistent with their own columns (2,375 m, and 1,839 m against the
  printed 1,840 m) and supplies the three that are missing or wrong, and every
  `ηtot,situ` column of Table L.3 / G.3 then reproduces to 5·10⁻⁵. The printed
  values applied to the wrong element of the same type miss by far more than
  that rounding: 2,375 m on external wall 2 gives 0,108 5 against the printed
  0,114 9 at 50 Hz, and 1,840 m on internal wall 1 gives 0,085 0 against
  0,077 0.
- **Library behaviour:** `in_situ_total_loss_factor` takes `Σ lk αk` as an
  input and `perimeter_absorption_coefficient` implements Formula (C.4); the
  Annex L fixture derives all five sums that way rather than using the printed
  block, and says so ([`tests/building/test_detailed_prediction.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/test_detailed_prediction.py)).
- **Status:** unreported.

## ISO 12354-1:2017 Table L.3 / ISO 12354-2:2017 Table G.3 (external wall ηint)

- **Location:** the same input-data block, external-wall line.
- **The print:** `ηint = 0,013` for the 365 mm autoclaved aerated concrete
  external walls.
- **The problem:** the example's own element specification, and Annex B
  Table B.3 for autoclaved aerated concrete, give 0,012 5. Only 0,012 5
  reproduces the tabulated `ηtot,situ`: at 500 Hz Formula (C.1) gives
  0,012 5 + 0,001 41 + 0,034 57 = 0,048 5, the printed value, where 0,013
  would give 0,049 0.
- **Evidence:** term-by-term recomputation of Formula (C.1) for both external
  walls at every band with each candidate `ηint`.
- **Library behaviour:** the Annex L fixture uses 0,012 5.
- **Status:** unreported.

## ISO 12354-1:2017, Table L.4 (second path block labelled 2d)

- **Location:** Annex L, Table L.4 (printed p. 82), the right-hand block
  headed "Transmission path 2d".
- **The print:** the block gives `αi,situ` = 6,3 to 14,1, `Dv,ij,situ` = 11,0
  to 13,6 and `Rij` = 43,9 to 84,6 dB.
- **The problem:** those are the numbers of path **4d** (internal wall 2 to
  the separating floor), not of path 2d (external wall 2). Table L.1 of the
  same annex prints the whole R4d column, 43,9 to 84,6 dB, and the block's
  `Rij` column is that column cell for cell. What settles it band by band is
  the other two columns, which cannot be confused: external wall 2 has
  `αi,situ` = 10,3 m at 50 Hz (S = 13,75 m², ηtot = 0,114 9) while internal
  wall 2 has 6,3 m (ηtot = 0,070 3), the printed value; and `Dv,ij,situ`
  follows the floor-to-internal-wall `Kij` of 8,8 dB, which gives 11,0 to
  13,6 dB, not the floor-to-external-wall 6,4 dB, which gives 9,6 to 11,9 dB.
- **Evidence:** independent recomputation of Formulae (10), (11) and (15) for
  both candidate paths at every band. Path 4d reproduces all three columns of
  the block, `αi,situ` to 0,05 m and `Dv,ij,situ` and `Rij` to 0,05 dB, which
  is the printed resolution. Path 2d departs from the block's `Rij` column by
  0,1 dB to 7,0 dB depending on the band, and comes closest between 100 Hz and
  160 Hz (0,5 / 0,5 / 0,1 dB), so `Rij` alone does not identify the path over
  those bands; `αi,situ` (10,3 against 6,3 m at 50 Hz) and `Dv,ij,situ`
  (1,4 dB to 1,7 dB apart in every band) do.
- **Library behaviour:** the test that asserts the block builds it as path 4d
  and names the mislabelling.
- **Status:** unreported.

## ISO 12354-1:2017, Table L.1 (non-integer weighted ratings)

- **Location:** Annex L, Table L.1 (printed p. 79), the `Rw` row and the
  sentence below it, and the corresponding `Ln,w` row of ISO 12354-2:2017
  Table G.1.
- **The print:** the `Rw` row gives one decimal for every path (75,1 / 84,5 /
  70,6 / … and 57,8 in the total column) while the sentence immediately below
  states `R'w (C ; Ctr) = 57,9 (−2 ; −8) dB`.
- **The problem:** ISO 717-1 rates by shifting the reference curve **in 1 dB
  steps**, so a weighted rating is an integer; the printed one-decimal values
  are the reference curve shifted *continuously* until the sum of unfavourable
  deviations equals exactly 32,0 dB. The row truncates that continuous value
  to one decimal and the sentence rounds it, which is why the same quantity
  appears twice as 57,8 and 57,9. The spectrum adaptation terms inherit the
  offset: with the ISO 717-1 rating of 57 dB they are C = −1 and Ctr = −7, and
  the printed (−2 ; −8) is exactly the pair shifted by the same 0,86 dB.
- **Evidence:** a continuous-shift solve of the ISO 717-1 reference curve
  against the printed per-band spectra reproduces every printed value in both
  rows (RDd 75,12 against 75,1; RD1 84,54 against 84,5; R11 70,66 against
  70,6; the total 57,86 against 57,8 / 57,9; on the impact side Ln,Df1 29,58
  against 29,6 and the total 40,98 against 41,0), whereas the ISO 717-1
  1 dB-step ratings of the same spectra are 75, 84, 70 and 57 dB.
- **Library behaviour:** `weighted_rating` / `weighted_impact_rating`
  implement ISO 717-1/-2 as written, so the detailed model returns
  `R'w = 57 dB` and `L'n,w = 41 dB (CI = 2)` for the example; the test pins
  those and documents the printed values.
- **Status:** unreported.

## ISO 12354-2:2017, Table G.1 (50 Hz to 80 Hz flanking columns)

- **Location:** Annex G, Table G.1 (printed p. 36), the four `Ln,Df` columns,
  50 Hz, 63 Hz and 80 Hz rows.
- **The print:** `Ln,Df1` = 47,3 / 44,9 / 46,2 dB.
- **The problem:** Table G.4 of the same annex prints the same path Df for
  external wall 1, from the same inputs, as 47,8 / 45,9 / 47,0 dB. The two
  tables cannot both be right, and from 100 Hz upwards they agree exactly.
- **Evidence:** Formula (12) evaluated from the annex's own Table G.3 columns
  (`Ln,situ`, `Rsitu`) and the Table G.4 `Dv,ij,situ` and `ΔLsitu` columns
  gives 47,80 / 45,85 / 46,95 dB, reproducing the printed 47,8 / 45,9 / 47,0
  of Table G.4 to 0,05 dB and Table G.1 only from 100 Hz upwards. Carrying the
  same recomputation through the whole chain puts external wall 2 low by
  0,5 dB to 1,0 dB over the same three bands and the two internal walls low by
  up to 0,5 dB at 50 Hz and 63 Hz (their 80 Hz cells agree). From 100 Hz
  upwards no flanking column deviates by more than 0,15 dB. Correcting the
  affected cells raises the printed total `L'n` only slightly: 58,6 to
  58,7 dB at 50 Hz, 57,0 to 57,2 dB at 63 Hz, 55,9 to 56,1 dB at 80 Hz.
- **Library behaviour:** the test asserts Table G.4 in full, the Table G.1
  direct column over the whole range, and the Table G.1 flanking columns from
  100 Hz upwards, naming the disagreement.
- **Status:** unreported.

## ISO 12354-2:2017, Table G.8 (junction Kij and m'i)

- **Location:** Annex G, Table G.8 (printed p. 40), the internal wall to
  external wall rigid T junction.
- **The print:** row "Int. wall 1/2 - Ext. wall 1/2" gives Kij = 6,6 dB; the
  row below it, "Ext. wall 1/2 - Ext. wall 1/2", gives m'i = 2,19 kg/m².
- **The problem:** two independent misprints. The rigid-T corner branch
  K12 = 5,7 + 5,7 M² with M = lg(360/219) = 0,215 6 gives 5,97, i.e. **6,0**,
  and ISO 12354-1:2017 Table L.8 prints 6,0 for the identical junction of the
  identical example. And the external wall's mass per unit area is 219,0
  kg/m² throughout the example, not 2,19 (a factor 100).
- **Evidence:** Annex E evaluation of the corner branch; the same table's own
  other rows and the whole of ISO 12354-1 Annex L use 219,0 kg/m².
- **Library behaviour:** uses 6,0 dB and 219,0 kg/m².
- **Status:** unreported.

## ISO 12354-2:2017, Table G.6 (mislabelled row)

- **Location:** Annex G, Table G.6 (printed p. 40), internal wall to
  separating floor rigid cross junction.
- **The print:** a row labelled "Ext. wall 1/2 - Int. wall 1/2" with
  m'i = 360,0, m'⊥ = 484,0 and Kij = 11,0 dB.
- **The problem:** Table G.6 describes the *internal wall to separating floor*
  cross junction; no external wall meets it. The masses and the value are
  those of the in-line internal-wall path, and ISO 12354-1:2017 Table L.6
  prints the same row correctly as "Int. wall 1/2 - Int. wall 1/2".
- **Evidence:** the rigid-cross through branch 8,7 + 17,1 M + 5,7 M² with
  M = lg(484/360) gives 10,99, the printed 11,0, for the internal wall.
- **Library behaviour:** treats the row as the internal-wall in-line path.
- **Status:** unreported.

## ISO 12354-1:2017 Table L.10 / ISO 12354-2:2017 Table G.10 (element label)

- **Location:** the simplified-model input table of both parts, fourth row.
- **The print:** "Internal wall 4 (F = f = 4)".
- **The problem:** the example has two internal walls; the element indexed
  F = f = 4 is internal wall **2** (5,00 m x 2,75 m, S = 13,75 m²), as the
  detailed-model tables of the same annexes label it.
- **Evidence:** the row's own S = 13,75 m² and lij = 5,0 m match internal
  wall 2 of Table L.1 / G.1.
- **Library behaviour:** none needed; the numbers are unaffected.
- **Status:** unreported.

## ISO 15186-1:2000, Clause 3.9, Formula (8) (sign of the 10 lg N term)

- **Location:** Clause 3.9, Formula (8), the intensity element normalized
  level difference for N small building elements measured together.
- **The print:** DI,n,e = Lp1 − 6 − (LIn + 10 lg(Sm/A0) + 10 lg(N)), i.e. the
  10 lg N term is subtracted.
- **The problem:** the subtracted sign cannot be derived. Measuring N
  identical units within one measurement surface raises the transmitted
  power (and hence LIn + 10 lg Sm) by 10 lg N, so recovering the per-unit
  DI,n,e requires *adding* 10 lg N. The pressure-based equivalent,
  ISO 10140-2:2010 Formula (6), prints exactly that correction
  (Dn,e = L1 − L2 + 10 lg(nA0/A)), and ISO 15186-2:2010 Formula (12) prints
  Formula (8) without any N term (the N = 1 case, with which both signs
  agree). As printed, installing more units would *lower* the per-unit
  rating by 20 lg N relative to the derivable value.
- **Evidence:** page render of the printed formula; derivation from the
  diffuse-field receiving-room relation L2 = LW + 10 lg(4/A) against
  ISO 10140-2:2010 Formula (6); cross-check against ISO 15186-2:2010
  Formula (12) and Hopkins, *Sound Insulation* (2007), Eq. 3.45.
- **Library behaviour:** implements the derivable per-unit form
  (`intensity_element_normalized_difference`, +10 lg N) and emits a warning
  whenever n > 1, where the result deviates from the print.
- **Status:** unreported.

## ISO 10848-1:2006, Clause 8.1.1, Formula (20) (spurious π in the critical frequency)

- **Location:** Clause 8.1.1, Formula (20), the thin-plate critical frequency
  used by the test-facility flanking criterion of Formula (19).
- **The print:** fc = c0² / (1,8 cL · h · π).
- **The problem:** the constant 1,8 is itself the rounded 2π/√12 ≈ 1,814 of
  the thin-plate dispersion relation, so the extra π double-counts it and
  would misplace fc by a factor π (e.g. a 100 mm concrete element with
  cL = 3500 m/s: 187 Hz without the π, 59 Hz with it, far from any measured
  coincidence dip).
- **Evidence:** derivation from the thin-plate dispersion relation (Hopkins,
  *Sound Insulation* (2007), Eq. 2.201, fc = c0²/(1,8 cL h)); ISO
  12354-1:2017 prints the same π-free form in its symbol definitions
  (fc = c0²/(1,8 cL t)).
- **Library behaviour:** implements the π-free form
  (`phonometry.building.flanking_transmission.critical_frequency`), with a
  misprint note in the docstring.
- **Status:** corrected upstream — ISO 10848-1:2017 (second edition) prints
  the π-free form in its Formula (5), fc = c0²/(1,8 h cL), confirming the
  2006 print as a misprint. No report is needed. The entry is retained
  because the library cites the 2006 edition, whose print carries the
  defect; the 2017 edition stands as the confirmation.

## ISO 12999-1:2020, Table 4 (missing 500 Hz row)

- **Location:** Table 4 (in-situ uncertainties per band).
- **The print:** the 2020 edition's table omits the 500 Hz row that the 2014
  edition prints (situation B 1,2 dB / situation C 0,8 dB).
- **The problem:** likely an editorial omission; the surrounding rows are
  unchanged between editions and the text does not mention removing the band.
- **Evidence:** side-by-side comparison of the 2014 and 2020 prints.
- **Library behaviour:** follows the 2020 print as published, with the
  omission documented in the module.
- **Status:** unreported.

## ISO 12999-2:2020, Clause 8 wording vs Tables 4 and 5

- **Location:** Clause 8 (expression of results) and Tables 4/5.
- **The print:** the clause wording instructs rounding the standard
  uncertainty u before forming the expanded uncertainty U = k·u.
- **The problem:** the document's own Tables 4 and 5 only reproduce when U is
  computed from the unrounded u and rounded last; the literal clause wording
  fails half of the printed table entries.
- **Evidence:** recomputation of all 25 table entries under both conventions
  (round-last: 25 of 25 match; round-first: 10 of 20 mismatch).
- **Library behaviour:** rounds last, matching the tables; the convention is
  documented and tested.
- **Status:** unreported.

## ISO 10052:2021, Table 4 volume-range header

- **Location:** Table 4 (reverberation-index estimator), volume-range header.
- **The print:** the header reads "60 ≤ V < 150" while the body text says the
  method applies to rooms "up to 150 m³".
- **The problem:** the boundary V = 150 m³ is included by the text and
  excluded by the header.
- **Evidence:** direct comparison of header and clause text.
- **Library behaviour:** accepts V = 150 (follows the text), with the
  ambiguity noted.
- **Status:** unreported.

## ISO 17208-2:2019, Clause 5 uncertainty band coverage

- **Location:** Clause 5 (representative expanded uncertainties).
- **The print:** 5 dB for the low-frequency bands (10 Hz to 100 Hz), 3 dB for
  the mid-frequency bands (125 Hz to 16 000 Hz), 4 dB for the high-frequency
  bands (above 20 000 Hz).
- **The problem:** the band list leaves the 20 kHz one-third-octave band
  unassigned (nothing covers 16 kHz to 20 kHz inclusive).
- **Evidence:** the clause's own enumeration.
- **Library behaviour:** applies the conservative 4 dB high-band value from
  just above 16 kHz, with the gap documented.
- **Status:** unreported.

## ECMA-418-1:2024 (3rd edition), clause 4.1.2 (upper limit of the range of interest)

- **Location:** clause 4.1.2, the stated frequency range of interest for
  discrete tones.
- **The print:** "between 89,1 Hz and 11 220 Hz inclusive".
- **The problem:** every formula and table of the standard uses 11 200 Hz:
  the Table 2/3 band-edge fits end at 11 200 Hz, and Formulae (13)/(26)
  treat the upper end of the criterion range consistently with 11 200 Hz.
  No other clause mentions 11 220 Hz.
- **Evidence:** cross-check of the clause 4.1.2 prose against Tables 2 and 3
  and the criterion formulas of clauses 11.5/12.6.
- **Library behaviour:** uses the internally consistent 89,1 Hz to
  11 200 Hz range (upper end exclusive per the formulas), with a code note
  in [`tonality.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/tonality.py).
- **Status:** unreported.

## ECMA-418-2:2025 (4th edition), clause 5.1.5.2 (last block index)

- **Location:** clause 5.1.5.2, the segmentation of the zero-padded signal
  for the roughness/fluctuation-strength block sizes.
- **The print:** the index of the last block is given as
  l_last = ceil((n + s_b)/s_h).
- **The problem:** the formula is internally inconsistent: blocks placed at
  that index overrun the zero-padded signal defined by clause 5.1.2.2, and
  the resulting Formula (103) time grid becomes non-monotonic. The only
  self-consistent reading is to stop at the last block that fits inside the
  padded signal and align it flush with its end.
- **Evidence:** direct evaluation of the block start indices against the
  padded length for the clause 7.1.1 block/hop sizes; the flush-to-end
  reading reproduces the Clause 7 roughness calibration (1 asper) to
  0,9999.
- **Library behaviour:** implements the flush-to-end reading with a code
  note in [`roughness_ecma.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/roughness_ecma.py).
- **Status:** unreported.

## ECMA-418-2:2025 (4th edition), clause 9.1.4, Formula (127) (HSA kernel phase)

- **Location:** clause 9.1.4, Formula (127), the spectral kernel of the
  envelope analysis window used by the High-resolution Spectral Analysis.
- **The print:** the kernel's phase factor is
  exp(−j·2π·f_n(k)·(s̃_b − n_ze + n_zb − 1)).
- **The problem:** the kernel is, by construction, the DFT of the
  rectangular analysis window of Formula (120) modulated to the candidate
  rate; that is the model Formula (124) fits to the measured DFT spectrum.
  That DFT has the phase exp(−j·π·f_n·(s̃_b − n_ze + n_zb − 1)); the
  printed factor doubles it (and is also inconsistent with the π arguments
  of the printed sine terms of the same formula). With the printed phase
  the fitted model cannot reproduce the spectrum of a noiseless windowed
  sinusoid, contradicting the clause's own statement that the HSA achieves
  "theoretically infinite resolution for signals without noise".
- **Evidence:** independent derivation of the window DFT plus numerical
  recomputation: with π the least-squares fit recovers the constant part,
  amplitudes and phases of synthetic noiseless envelopes to machine
  precision and the Formula (135) residual vanishes; with the printed 2π
  the kernel deviates from the window DFT by amounts of the order of the
  kernel itself and the residual stays of the order of the signal energy.
- **Library behaviour:** implements the π reading, pinned by a regression
  test on the exact recovery of synthetic line pairs.
- **Status:** unreported.

## ECMA-418-2:2025 (4th edition), clause 9.1.5, Formula (144) (bin offset)

- **Location:** clause 9.1.5, Formula (144), the modulation rate of a local
  maximum of the envelope power spectrum.
- **The print:** the rate is the three-bin amplitude-weighted centroid of
  the peak position **minus one**, scaled by Δf.
- **The problem:** clause 9.1.4 (below Formula (122)) defines the spectral
  index k as mapping to the modulation rate k·r̃_s/s̃_b with k starting at
  0. A symmetric local maximum at bin k has centroid k, and the printed
  formula then assigns it the rate (k − 1)·Δf, one full bin (0,73 Hz) low,
  which at fluctuation-strength rates is fatal (a true 1,46 Hz modulation
  would be reported as 0,73 Hz). The offset is only consistent with
  1-based spectral-line positions, contradicting the standard's own
  definition of k.
- **Evidence:** cross-check of Formula (144) against the k-to-rate mapping
  stated below Formula (122).
- **Library behaviour:** uses the centroid directly (no offset) with the
  0-based k of Formula (122).
- **Status:** unreported.

## ECMA-418-2:2025 (4th edition), clause 9.1.7 (units of the fine-tuning constants)

- **Location:** clause 9.1.7, Formulae (149)-(152), the damped Newton fine
  tuning of the dominant modulation rate.
- **The print:** differential step Δx = 10⁻⁵, damped-step cap 2·10⁻⁴, stop
  tolerance 10⁻⁷ and an iteration limit of 40, with the starting point
  x₀ = f̃_c,imax (a rate in Hz) and the failure check
  |f_c,1,opt − f̃_c,imax| > 1,25·Δf.
- **The problem:** the constants carry no units. Read in Hz, the damped
  step is capped at 5·10⁻⁵ Hz per iteration (2·10⁻³ Hz over all 40
  iterations), so the tuning cannot move appreciably and the 1,25·Δf
  (≈ 0,92 Hz) failure check is unreachable; the whole clause would be
  inert. Read as normalized modulation rates f/r̃_s (the variable in which
  the Formula (127) kernel frequencies are expressed), the same constants
  give a 0,075 Hz damped per-iteration cap (≈ 2,9 Hz over the 39
  iterations), a 1,5·10⁻⁴ Hz stop tolerance and a reachable failure check,
  all consistent with the clause's purpose.
- **Evidence:** dimensional analysis of the printed constants against the
  0,7324 Hz spectral resolution and the failure threshold.
- **Library behaviour:** applies the constants as normalized modulation
  rates.
- **Status:** unreported.

## ECMA-418-2:2025 (4th edition), clause 9 introduction (broken cross-reference)

- **Location:** clause 9, third paragraph of the introduction, on the
  HSA-based loudness prediction.
- **The print:** "loudness scaling is improved by using HSA-based loudness
  prediction (see Clause 0)".
- **The problem:** "Clause 0" does not exist; the HSA-based loudness
  scaling is described in clause 9.1.10 (an unresolved field reference).
- **Evidence:** the clause listing of the standard itself.
- **Library behaviour:** none required (the intended target is
  unambiguous).
- **Status:** unreported.

## ISO/PAS 20065:2016, clause 5.3.4 (edge steepness of a distinct tone)

- **Location:** clause 5.3.4, Formulae (10)/(11), the minimum edge steepness
  of a distinct tone.
- **The print:** asymmetric formulas: the lower-edge steepness is scaled by
  f_T/2 and the upper-edge steepness by f_T (no divisor).
- **The problem:** the parent standard DIN 45681:2005-03 prints f_T/sqrt(2)
  on **both** edges, and its executable Annex J reference program does the
  same (`Frequenz(i)/Sqr(2)`). The two prints cannot both be satisfied; the
  ISO version is plausibly a typesetting corruption of the sqrt(2) factor
  (the radical dropped on one edge and halved on the other). Versus the DIN
  program the ISO print is sqrt(2) more lenient on the lower edge and
  sqrt(2) stricter on the upper; borderline tones with one-sided edge
  steepness around 17 to 34 dB/octave flip classification between the two
  readings.
- **Evidence:** side-by-side comparison of the ISO print, the DIN 45681
  print and the DIN Annex J program.
- **Library behaviour:** follows the DIN/sqrt(2) reading (it matches the
  only executable reference), with the choice recorded in
  [`tone_audibility.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/tone_audibility.py).
- **Status:** unreported.

## DIN 45681:2005-03, Anhang I, Tabelle I.6, row "6 FG"

- **Location:** Anhang I, Beispiel I.2 (combustion engine, spectrum j = 1),
  Tabelle I.6, the combined row "6 FG" for the three tones k = 6/7/8
  (592,2 / 629,8 / 643,3 Hz, tone levels 78,31 / 75,00 / 79,75 dB).
- **The print:** L_T = 81,11 dB together with delta L = 9,12 dB (with
  L_S = 59,53, L_G = 76,16, a_v = -2,40 at 592,2 Hz).
- **The problem:** the two cells contradict each other. The printed
  delta L = 9,12 dB only reproduces from the *plain* Formula (17) energy sum
  of the three tone levels (82,87 dB): 82,87 - 76,16 + 2,40 = 9,11. The
  printed L_T = 81,11 dB is consistent instead with the Anmerkung 2
  shared-line dedupe (the 629,8/643,3 Hz tonal runs overlap), which would
  give delta L = 7,35 dB. Every other FG row of the annex is internally
  consistent (e.g. "2 FG": L_T = 72,15, delta L = 9,18, both from the same
  sum; no lines shared there).
- **Evidence:** recomputation of both readings from the printed per-tone
  levels of Tabelle I.6; the same-page "2 FG" row as the consistent control.
- **Library behaviour:** `combined_tone_level` follows Anmerkung 2 (shared
  lines counted once), which reproduces the printed "2 FG" oracle; for the
  "6 FG" row only the delta L chain is pinned, with the contradiction
  recorded in `tests/reference_data.py`.
- **Status:** unreported.

## IEC 60268-3:2013, clause 14.12.9.2 f) (DIM denominator)

- **Location:** clause 14.12.9.2, item f), the formula for the dynamic
  intermodulation distortion d_DIM.
- **The print:** the denominator of the printed formula is "U2".
- **The problem:** the defining clause 14.12.9.1 states the ratio of the
  r.m.s. sum of the Table 2 intermodulation product voltages "to the
  amplitude of the output voltage at the frequency f_s", i.e. the 15 kHz
  sine component U_s, the Otala convention. The symbol U2 is used throughout
  14.12 for the total output voltage, which contradicts 14.12.9.1 (the test
  signal is dominated by the 3,15 kHz square wave, so the two denominators
  differ by several dB). Item d) of the same clause measures "the amplitudes
  of the sinusoidal signal U_s", which the f) formula then never uses.
- **Evidence:** side-by-side reading of 14.12.9.1, 14.12.9.2 d) and
  14.12.9.2 f); the historical DIM literature (Otala) defines the ratio to
  the sine amplitude.
- **Library behaviour:** follows the 14.12.9.1 definition (reference = the
  output amplitude at f_s), with a code comment at the reference measurement
  in [`distortion.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/electroacoustics/distortion.py).
- **Status:** unreported.

## UNE-EN 61043:1999, clause 6.1 (class 2 frequency range dropped in translation)

- **Location:** clause 6.1 "Rango de frecuencias", the class 2 sentence, of
  UNE-EN 61043 (April 1999), which declares itself "la versión oficial, en
  español, de la Norma Europea EN 61043 de enero 1994, que a su vez adopta la
  Norma Internacional CEI 61043:1993".
- **The print:** a single sentence, "Los procesadores de clase 2 deberán
  cubrir, al menos, el rango desde 45 Hz a 5,6 kHz en bandas de octava."
- **The problem:** the EN/IEC text gives class 2 processors two alternative
  ranges, not one: "Class 2 processors shall, at least, cover the range from
  45 Hz to 7,1 kHz in one-third octave bands, **or** the range from 45 Hz to
  5,6 kHz in one octave bands" (BS EN 61043:1994, clause 6.1). The
  translation drops the first alternative. The omission is normative rather
  than editorial: it removes one of the two ways clause 6.1 can be satisfied,
  and a reader of the Spanish text alone would conclude that class 2 is
  *defined* over octave bands, so that a one-third-octave chain verified over
  the 22 tabulated bands from 50 Hz to 6,3 kHz could not attest class 2 over
  its full range.
- **Evidence:** side-by-side reading of clause 6.1 in both prints. The class 1
  sentence is word-for-word equivalent in the two documents, so the divergence
  is confined to the class 2 sentence. The Spanish print also contradicts
  itself: its Table 2 tabulates the pressure-residual intensity index for
  class 2 processors at all 22 one-third-octave centres, and its faithfully
  translated Note 2 ("Para procesadores con análisis en bandas de octavas
  únicamente, los requisitos se aplican únicamente a las frecuencias centrales
  de las bandas de octava") carves out octave-only processors as a special
  case. Both are redundant if every class 2 processor is an octave-band one.
- **Library behaviour:** implements the EN/IEC reading.
  [`verify_intensity_class`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/metrology/intensity_compliance.py)
  treats the full 22-band one-third-octave set as attesting either class, and
  the 7-band octave set (63 Hz to 4 kHz) as a class 2 alternative that never
  attests class 1, with both branches pinned by regression tests
  ([`tests/metrology/test_intensity_compliance.py`](https://github.com/jmrplens/phonometry/blob/main/tests/metrology/test_intensity_compliance.py)).
- **Status:** unreported (national translation, not the issuing body's text).

## ISO/PAS 1996-3:2022, Clause 5 (cross-references of r and d)

- **Location:** Clause 5, Formula (2), the definitions of the symbols of the
  prominence P = 3 lg[r/(dB/s)] + 2 lg(d/dB).
- **The print:** "r is the onset rate (OR) as defined in 3.4" and "d is the
  level difference (LD) as defined in 3.5".
- **The problem:** the two cross-references are swapped. The document's own
  terms and definitions set 3.4 as the *level difference* LD ("difference in
  decibels of L_pAF between the level of the end point L_e and the level of
  the starting point L_s of the onset") and 3.5 as the *onset rate* OR
  ("slope in decibels per second of the straight line that gives the best
  approximation to the onset"). Read literally, Formula (2) would take three
  times the logarithm of a level difference plus twice the logarithm of a
  slope, inverting the weights the method assigns to the two quantities. The
  spelled-out names in the same list ("the onset rate (OR)", "the level
  difference (LD)") and the units given for each ("dB/s" for r, "dB" for d)
  make the intended reading unambiguous.
- **Evidence:** side-by-side reading of 3.4, 3.5 and the Clause 5 symbol
  list; the units printed with each symbol contradict the clause numbers
  printed with them.
- **Library behaviour:** implements the spelled-out reading, weighting the
  onset rate by 3 and the level difference by 2 (`predicted_prominence` in
  [`impulse_prominence.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environmental/impulse_prominence.py)),
  which is also the NT ACOU 112:2002 form the PAS carries over.
- **Status:** unreported.

## ISO 9613-2:1996, Table 2 (15 °C / 80 % / 1 kHz cell)

- **Location:** Table 2, "Atmospheric attenuation coefficient α for octave
  bands of noise", row 15 °C / 80 % relative humidity, column 1 kHz.
- **The print:** α = 4,1 dB/km.
- **The problem:** Table 2 is a rounded extract of ISO 9613-1, to which the
  clause itself defers ("For values of α at atmospheric conditions not
  covered in table 2, see ISO 9613-1"). Evaluating the ISO 9613-1 pure-tone
  formula at 1 kHz, 15 °C, 80 % RH and 101,325 kPa gives 4,1511 dB/km, which
  rounds to 4,2, not the printed 4,1. The neighbouring cells of the same row
  round correctly (2 kHz: 8,338 -> printed 8,3; 4 kHz: 23,86 -> 23,7 at the
  exact band centre), as do the 1 kHz cells of the other rows (15 °C / 50 %:
  4,164 -> printed 4,2), so the defect is confined to this cell.
- **Evidence:** independent evaluation of the ISO 9613-1 coefficient at both
  the nominal and the exact band-centre frequency (4,1511 dB/km either way,
  1 kHz being both).
- **Library behaviour:** unaffected. The library never reads Table 2: it
  computes A_atm from the ISO 9613-1 formula directly
  ([`air_absorption.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environmental/air_absorption.py)),
  so it yields 4,15 dB/km for this condition.
- **Status:** unreported.

## ANSI S3.5-1997, Annex C worked examples (official WG S3-79 errata)

- **Location:** Annex C, Table C.1 (octave-band worked example, p. 21) and
  Table C.2 (one-third-octave worked example, p. 22) of the 1997 printing.
- **The print:** (a) Table C.1, row i = 5, the level-distortion factor Li
  under Step 6 is printed as 0.10; (b) Table C.2, first row, the
  self-speech-masking slope Ci is printed as −45.59.
- **The problem:** both cells contradict the standard's own normative
  formulas. (a) Clause 5.7 with the example's inputs (E′5 = 20 dB,
  U5 = 9.33 dB) gives L5 = 1 − (20 − 9.33 − 10)/160 = 0.9958, which prints
  to two decimals as 1.00, not 0.10. (b) Clause 5.4 with the example's
  inputs (B1 = 40 dB, f1 = 160 Hz) gives
  C1 = −80 + 0.6 (40 + 10 lg 160 − 6.353) = −46.587, which prints as −46.59,
  not −45.59; the example's Zi column is only consistent with the corrected
  slope (Z2 recomputes to 34.658 = printed 34.66 dB, whereas the misprinted
  slope would give 34.76 dB). The Table C.1 example is the octave-band
  procedure and the Table C.2 example the one-third-octave procedure, so one
  cell of each is affected.
- **Evidence:** the official errata list published by ASA Working Group
  S3-79, the committee that maintains ANSI S3.5, on its support site
  (sii.to): "Page 21, Table C1, row i=5, column Li under Step 6: the value
  printed as 0.10 should be changed to 1.00" and "Page 22, Table C2, the
  first row of numbers, value −45.59 should be −46.59"; plus independent
  recomputation of both cells from the normative clauses (above). The same
  list carries five further corrections (a reference spelling, the
  Tables 1-4 caption wording recorded in the next entry, the insertion gain Gi
  missing from Eq. 23, and two Annex B wording fixes about the audio-visual
  approximation); none of those touches a formula this library implements.
- **Library behaviour:** unaffected; the library computes the corrected
  values from the normative clauses and always did. Its Annex C.2 anchors
  ([`tests/reference_data.py`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data.py),
  `ANSIS3_5_ANNEX_C1*` and `ANSIS3_5_ANNEX_C2*`) pin the errata-consistent
  chain of both examples, cross-checked to double precision against the working
  group's own reference implementation `SII.C` and its published test-case
  results. The Table C.1 cell is pinned directly: the level-distortion factor
  of clause 5.7 for row i = 5 of the Annex C.1 octave-band example computes to
  0.99581, which prints as the corrected 1.00.
- **Status:** published corrections by the issuing working group; nothing
  to report upstream.

## ANSI S3.5-1997, captions of Tables 1 to 4 (official WG S3-79 erratum)

- **Location:** the captions of Tables 1, 2, 3 and 4 (pp. 3-5 of the 1997
  printing), the constant tables of the four band procedures: critical band
  (21 bands), equally-contributing critical band (17 bands), one-third octave
  (18 bands) and octave (6 bands).
- **The print:** each caption lists the quantities the table tabulates and
  includes the phrase "hearing threshold levels,".
- **The problem:** none of the four tables tabulates a hearing threshold
  level. Each carries the band centre frequency (and, for Tables 1, 2 and 4,
  the band limits), the band-importance function Ii, the standard speech
  spectrum level Ui by vocal effort and the reference internal noise spectrum
  level Xi. The hearing threshold level Ti′ is a *user input* to the procedure
  (clause 5.5, where the equivalent internal noise spectrum level is
  Xi′ = Xi + Ti′), which is exactly the quantity the caption invites the
  reader to look for in the table and to confuse with Xi.
- **Evidence:** the official errata list published by ASA Working Group S3-79,
  the committee that maintains ANSI S3.5, on its support site (sii.to):
  "Pages 3-5, Tables 1-4: in each caption the phrase 'hearing threshold
  levels,' should be deleted"; plus the tables themselves, which have no such
  column.
- **Library behaviour:** unaffected. The four tables are implemented with the
  columns they actually carry, exposed per procedure by `sii_procedure()` as
  `band_importance`, `speech_spectrum` (Ui) and `internal_noise` (Xi), and the
  hearing threshold stays the `threshold=` argument of
  `speech_intelligibility_index` ([`src/phonometry/hearing/sii.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/hearing/sii.py)).
- **Status:** published correction by the issuing working group; nothing to
  report upstream.

## NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (27)

- **Location:** section A.4.2, Eq. (27) (atmospheric absorption coefficient).
- **The print:** the coefficient 6,6928·10⁻⁶ is paired with the relaxation
  frequency frO = 630,7 Hz.
- **The problem:** evaluated as printed, the equation yields nonsense
  (14,3 dB/km at 500 Hz against the guidance's own Table 4 value of 3,1).
  The physically correct pairing (6,6928·10⁻⁶ with the oxygen relaxation
  frequency, about 75 692 Hz at the reference conditions, and 1,3415·10⁻⁶
  with 630,7 Hz) reproduces Table 4 and the ISO 9613-1 pure-tone coefficient
  to 0,02 dB/km.
- **Evidence:** numeric evaluation of both pairings against Table 4.
- **Library behaviour:** implements the correct pairing; the module docstring
  carries a defensive note so the misprint is not transcribed as a "fix".
- **Status:** unreported.

## NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (21)

- **Location:** section A.3.3, Eq. (21) (flight path angle).
- **The print:** γ = acos(ΔZ/ΔS).
- **The problem:** the arccosine of the climb-to-path ratio returns the
  complement of the path angle (90° in level flight, where γ must be 0°) and
  contradicts the guidance's own use of γ as the climb/descent angle
  throughout section A.3. ECAC Doc 32, 1st ed., Eq. (10) prints the correct
  form, γ = atan(ΔZ/ΔS) with the horizontal ΔS of its Eq. (8).
- **Evidence:** evaluation in level flight; cross-check against Doc 32
  Eq. (10) and against the NORAH2 prototype input files, whose ``Vang``
  columns are climb/descent angles (0° in level segments).
- **Library behaviour:** ``flight_path_kinematics`` implements the Doc 32
  ``atan`` form; the result docstring carries the defensive note.
- **Status:** unreported.

## NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), §A.3.1 triangulation

- **Location:** section A.3.1, steps 2 to 4 (flight-condition interpolation),
  against the triangulation lookup tables shipped with the NORAH2 database
  (``*_triangulation.int``).
- **The print:** steps 2 and 3 normalise the database conditions (spans, with
  F_fc = 2 on the path angle) and step 4 computes "the Delaunay triangulation
  for the database flight conditions γ̄_j and V̄_j", i.e. of the normalised
  points, offering a lookup table as an equivalent.
- **The problem:** the lookup tables shipped with the database (which the
  guidance says are part of the hemisphere data and should not be edited) are
  the Delaunay triangulation of the raw (V, γ) conditions, not of the
  normalised ones: for the R22 set, 14 of the 27 shipped triangles differ
  from the Delaunay triangulation of the normalised conditions. A Delaunay
  triangulation is not invariant under the anisotropic normalisation, so the
  two prescriptions select different enveloping triangles for part of the
  envelope. The distance weights of Eq. (7)/(8) do use the normalised
  coordinates in the prototype (verified against its blended outputs).
- **Evidence:** recomputation of both triangulations for the R22 database;
  bin-for-bin reproduction of the prototype's per-step hemisphere selection
  with the shipped tables, and of its blended levels with normalised-space
  weights, to 0,05 dB.
- **Library behaviour:** ``flight_condition_weights`` follows the printed
  method (Delaunay of the normalised conditions) by default and accepts the
  database lookup table via ``triangles``, which reproduces the reference
  implementation exactly.
- **Status:** unreported.

## NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (46)

- **Location:** section A.4.5, Eq. (46) (source-side ground effect weighted
  by diffraction).
- **The print:** the weighting exponent reads (ΔL_g,s′ − ΔL_d,s)/20.
- **The problem:** no term ΔL_g,s′ exists; the prose directly below the
  equation defines ΔL_d,s′ as "the attenuation due to the diffraction between
  the image source S′ and R", the receiver-side companion Eq. (47) prints the
  parallel term correctly as ΔL_d,r′, and the CNOSSOS-EU method the section
  is based on writes Δ_ground(S,O) with Δ_dif(S′,R) in that position. The
  subscript g is a misprint for d.
- **Evidence:** internal consistency of the section (its own prose and
  Eq. (47)) and the CNOSSOS-EU source of the equations.
- **Library behaviour:** implements the image-source diffraction term
  ΔL_d,s′ as defined by the prose.
- **Status:** unreported.

## NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), §A.4.5 cross-references

- **Location:** section A.4.5, the definitions under Eq. (46) and Eq. (47).
- **The print:** ΔL_d,s′, ΔL_d,s and ΔL_d,r′ are said to be "calculated as
  per eq. 44" (four occurrences: two under Eq. (46) and two under Eq. (47)).
- **The problem:** Eq. (44) is the multiple-diffraction coefficient C″; the
  attenuation due to diffraction is Eq. (42). The three cross-references point
  at the auxiliary coefficient instead of the formula they describe.
- **Evidence:** the terms are attenuations in dB, which only Eq. (42)
  produces; Eq. (44) is a dimensionless coefficient consumed by Eq. (42).
- **Library behaviour:** evaluates the image-path and direct diffraction
  terms with Eq. (42), using Eq. (44) for C″ inside it.
- **Status:** unreported.

## RANDI 3.1 Physics Description (NRL, Breeding et al.), Table 2

- **Location:** Table 2 (representative ship source levels).
- **The print:** two cells deviate from the report's own Eqs. (2) to (5)
  evaluated with the Table 1 average lengths and speeds: the Merchant value
  at 25 Hz (about 3 dB high) and the Tanker value at 300 Hz (about 1 dB low).
  The Fishing Vessel row is not reproducible from the Table 1 averages at
  all (a constant offset of about 3,8 dB suggests different assumed inputs).
- **The problem:** the report does not state the exact inputs used for
  Table 2, and two cells contradict its own equations while every Large
  Tanker and Super Tanker cell agrees to 0,06 dB.
- **Evidence:** recomputation of all 25 cells from Eqs. (2) to (5).
- **Library behaviour:** the regression test pins the reproducible rows and
  excludes the contradicting cells with the rationale in the test.
- **Status:** unreported (technical report rather than a standard).

## Osses, García & Kohlrausch (2016), fluctuation-strength model, Eq. (3)

- **Location:** Eq. (3), the critical-band-rate (Bark) transformation of the
  excitation-pattern front-end.
- **The print:** z(f) = 13·arctan(0,76·10⁻⁴·f) + 3,5·arctan((f/7500)²).
- **The problem:** the first coefficient is the Zwicker-Terhardt 0,76·10⁻³
  with the exponent misprinted. The paper's own anchors disprove the print:
  it states 0,5 Bark = 50 Hz and 23,5 Bark = 13,2 kHz (section 2.1.2) and
  15 Bark = 2,7 kHz (section 3.1), all of which require 10⁻³. With 10⁻⁴,
  z(1 kHz) = 1,05 instead of 8,51 Bark and the model's 47 filter centres
  would span 491 Hz to 20 kHz instead of 50 Hz to 13,2 kHz.
- **Evidence:** evaluation of Eq. (3) under both exponents against the
  paper's printed Bark/frequency anchors.
- **Library behaviour:** implements 0,76·10⁻³ with a note at the formula;
  the carrier-frequency sweep test would catch a regression to the printed
  value ([`fluctuation_strength.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/fluctuation_strength.py)).
- **Status:** unreported (conference paper rather than a standard).

## Medwin & Clay, Fundamentals of Acoustical Oceanography (1998), Eq. 3.4.29

- **Location:** the Francois-Garrison boric-acid term as transcribed by the
  textbook (Eq. 3.4.29).
- **The print:** the boric-acid factor is printed as A1 = (8,68/c)·10^(0,78 pH − 5).
- **The problem:** the original paper (Francois & Garrison 1982, JASA 72,
  Part II, Eq. (10) and Fig. 7) prints 8,86; the digits are transposed. Only
  8,86 reproduces the paper's own Table IV: with 8,68 the boric-dominated
  cells at 0,6 to 30 kHz sit up to 1,7 % below the printed totals (worst
  relative case 2 kHz, 10 °C, S = 35: 0,1209 vs the printed 0,123 dB/km).
- **Evidence:** recomputation of all sampled Table IV cells under both
  coefficients against the paper's printed values.
- **Library behaviour:** implements the paper's 8,86 with a defensive note;
  the pinned Table IV set includes the boric-dominated rows.
- **Status:** unreported (textbook rather than a standard).

---

## Maa (1998), "Potential of microperforated panel absorber", JASA 104(5), Eq. (5b)

- **Location:** Eq. (5b), the mass-reactance coefficient of the microperforated
  panel, printed as k_m = 1 + [1 + k²/2]^(−1/2) + 0,85 d/t.
- **The print:** the first bracket term reads (1 + k²/2)^(−1/2).
- **The problem:** the same paper's Eq. (4), from which (5b) is factored,
  prints the term as (3² + k²/2)^(−1/2), and only that form reproduces the
  Crandall low-k limit Z1 → (4/3)·jωρ0t of the paper's own Eq. (3a): at
  k → 0 the printed (5b) gives an internal mass factor of 2 instead of 4/3.
  The paper's own Fig. 1 confirms it: with 0,85·d/t = 0,85 the plotted k_m
  starts near 2,2 (= 4/3 + 0,85) at k = 0,1, not at 2,85.
- **Evidence:** recomputation of both bracket variants against Eq. (4),
  Eq. (3a) and the Fig. 1 curve; the exact Bessel solution of Eq. (2) agrees
  with Eq. (4) within Maa's stated ~6 % only with the 3² form (the 1 form
  errs by >30 % at low k).
- **Library behaviour:** implements the exact Eq. (2) (no approximation), so
  the misprint does not enter the code; the regression test
  ``test_maa_exact_vs_wide_range_approximation`` pins the exact solution to
  the corrected Eq. (4) form.
- **Status:** unreported (journal paper; the correct form appears in Maa's
  earlier 1975/1987 papers and in secondary literature).

## Jiménez, Groby, Pagneux & Romero-García (2017), Appl. Sci. 7(6), 618, Eqs. (7)-(8)

- **Location:** Eqs. (7) and (8), the rectangular-duct visco-thermal
  effective density and bulk modulus (Stinson's series, used for the square
  necks and cavities of the slit + Helmholtz-resonator absorber).
- **The print:** the leading normalising constant of both series is 4:
  ρ_eff = −ρ0·a²b²/(4·G_ρ²·Σ) and the matching 4·(γ−1)·G_κ²/(a²b²) factor
  inside κ_eff.
- **The problem:** the correct constant is 64 (a factor-16 error). Only 64
  reproduces the exact limits of the model: as the boundary layers vanish
  ρ_eff → ρ0 and κ_eff → κ0 (the printed 4 gives 16·ρ0), and at DC the
  square duct's jω·ρ_eff tends to the exact Shah-London Poiseuille flow
  resistivity: the series value a⁶/(64·S0) = 28,4542 matches fRe/2 = 28,455
  (in units of η/a²), where S0 is the double transverse-mode sum at G = 0;
  the printed 4 gives sixteen times that.
- **Evidence:** evaluation of both constants against the boundary-layer-free
  limits and the Shah-London exact square-duct value; the wide-duct limit of
  the series also only matches the papers' own slit model (Eq. (6)) with 64.
- **Library behaviour:** implements 64 with a docstring note; the limits are
  pinned in
  [`tests/materials/test_slow_sound_absorber.py`](https://github.com/jmrplens/phonometry/blob/main/tests/materials/test_slow_sound_absorber.py)
  and the conformance check "Poiseuille limit (Stinson 1991)".
- **Status:** unreported (journal paper rather than a standard).

## Jiménez et al. (2017), Appl. Sci. 7(6), 618 / Sci. Rep. 7, 5389, slit-radiation term

- **Location:** Appl. Sci. Eq. (3), the characteristic radiation impedance of
  the slits, and the identical Methods reprint in the metadiffusers paper
  (Sci. Rep. 7, 5389, Eq. (5)).
- **The print:** Z_Δl_slit = −iω·Δl_slit·ρ0/(φt·S0).
- **The problem:** the term models the added radiation mass of the slit
  mouth, but the printed −iω prefactor is an opposite-time-convention
  (e^{−iωt}) expression inconsistent with the papers' otherwise e^{+iωt}
  transfer-matrix chain (the +i off-diagonal slit matrices of Appl. Sci.
  Eq. (2) and the −i cotangent-type resonator impedance). Transcribed
  literally into that chain, the correction raises the slit-panel resonance
  where an added mass must lower it: for a 1 mm slit with a 30 mm lattice
  step and 50 mm period the absorption peak moves from 378,6 Hz to 386,8 Hz
  as printed, against 370,8 Hz with the mass sign. The neck end corrections
  of the same model behave correctly (they lower the resonator resonance).
- **Evidence:** numerical evaluation of both signs of the correction against
  the uncorrected panel; the direction of the neck end corrections of the
  same papers as the consistent control.
- **Library behaviour:** uses the added-mass sign (+jω in the e^{+jωt}
  convention of the library), conjugating the printed term exactly as it
  conjugates the papers' Stinson duct series; direction and peak are pinned
  by ``test_slit_radiation_correction_lowers_resonance`` in
  [`tests/materials/test_slow_sound_absorber.py`](https://github.com/jmrplens/phonometry/blob/main/tests/materials/test_slow_sound_absorber.py).
- **Status:** unreported (journal papers rather than standards).

## Attenborough & Van Renterghem, Predicting Outdoor Sound 2e (2021), Table 5.1

- **Location:** Table 5.1, "Coefficient and exponent values in the Delany and
  Bazley, Miki and modified Miki models", row "Miki [6,7]", coefficient r.
- **The print:** r = 0,0109.
- **The problem:** the original source (Miki 1990, J. Acoust. Soc. Jpn (E)
  11(1), Eq. (34)) prints beta(f) = (ω/c0)[1 + 0,109·(f/σ)^(−0,618)]; the
  table drops a digit. With 0,0109 the real part of the Miki wavenumber at
  f/σ = 0,01 is 1,19 instead of 2,89, inconsistent with the same table's
  Delany-Bazley row (2,79 via ρ0 = 1,2) and with the "modified Miki" row
  the book itself derives from it.
- **Evidence:** digit check against the original Miki (1990) paper (Eqs.
  (30)–(34)) and cross-computation of both variants at the fit-range edge.
- **Library behaviour:** implements Miki's original 0,109; the digitization
  point f/σ = 0,1 is pinned in ``tests/reference_data.py`` and in the
  conformance check "Miki 1990 Eqs. (30)-(34)".
- **Status:** unreported (textbook rather than a standard).

## Attenborough & Van Renterghem, Predicting Outdoor Sound 2e (2021), Eq. (5.13)

- **Location:** Eq. (5.13), the Johnson-Champoux-Allard bulk complex density,
  with G(Λ) = sqrt(1 − 4iTηρ0ω/(R_S²Λ²Ω²)).
- **The print:** the tortuosity T appears to the first power inside G(Λ).
- **The problem:** Johnson et al. (1987) and the standard JCA formulation
  (Cox & D'Antonio 3e Eq. (6.19); Allard & Atalla) carry T² = α∞² there. The
  first-power print breaks the high-frequency asymptote that defines the
  viscous characteristic length: with T² the density tends to
  (Tρ0/Ω)(1 + (1 − j)δ_v/Λ) with δ_v = sqrt(2η/ρ0ω), while the printed form
  tends to a δ_v/(Λ·sqrt(T)) correction, which for T = 2 means an error of 29 % in the
  boundary-layer term for the same Λ.
- **Evidence:** asymptotic expansion of both variants against the Johnson
  et al. definition of Λ and against Cox & D'Antonio Eq. (6.19); the
  library's high-frequency JCA test pins the T² behaviour.
- **Library behaviour:** implements the standard T² form (Cox & D'Antonio
  Eq. (6.19)); the asymptote is pinned in
  ``test_high_frequency_density_asymptote``.
- **Status:** unreported (textbook rather than a standard).

## Bies, Hansen & Howard, Engineering Noise Control 5e (2017), Eq. (8.141)

- **Location:** Section 8.9.1, Eq. (8.141) (printed p. 461), the transmission
  loss of a muffler from the elements of its total four-pole matrix.
- **The print:** TL = 10 lg[ ((1+Mₙ)/(1+M₁))² · ¼ · |(Z_A1/Z_An)·T11 +
  T12/Z_An + Z_A1·T21 + (Z_An/Z_A1)·T22|² ], i.e. with the impedance ratio
  Z_A1/Z_An weighting T11 and its inverse weighting T22.
- **The problem:** the source the equation itself cites (Munjal, *Acoustics
  of Ducts and Mufflers* 2e, Eq. (3.27), p. 105) carries the overall
  prefactor Z_An/Z_A1 (equivalently √(S₁/Sₙ) inside a 20 lg form) with T11
  unweighted and Z_A1/Z_An on T22. As printed, Eq. (8.141) fails the
  sudden-expansion limit: a zero-length element (T = I) between S₁ = 0,01 m²
  and Sₙ = 0,02 m² is a sudden area expansion with the classic
  TL = 10 lg[(1+m)²/(4m)] = 0,512 dB (m = Sₙ/S₁ = 2), but the printed
  equation gives ¼·(Z_A1/Z_An + Z_An/Z_A1)² = 1,938 dB. Reading the ratios
  as an overall Z_A1/Z_An prefactor instead is also wrong: it gives 6,532 dB
  on the same oracle and violates reciprocity (11,34 vs −0,70 dB for an
  expansion chamber between unequal pipes; a negative TL for a passive
  element). The misprint is invisible whenever the inlet and outlet areas
  are equal, where every variant reduces to Eq. (8.148).
- **Evidence:** numeric evaluation of the zero-length identity element and
  of an unequal-port expansion chamber under the printed form, the inverted
  prefactor and Munjal Eq. (3.27); only Munjal's form reproduces the
  sudden-expansion classic (0,512 dB, both directions) and is reciprocal.
- **Library behaviour:** `transmission_loss` in
  [`silencers.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/silencers.py) implements
  Munjal Eq. (3.27), with the sudden-expansion limit and TL reciprocity
  pinned by regression tests
  ([`tests/noise_control/test_silencers.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_silencers.py))
  and a defensive note at the formula.
- **Status:** unreported (textbook rather than a standard).

## Long, Architectural Acoustics 2e (2014), Eq. (18.24) (sign of the microphone directivity)

- **Location:** Chapter 18, "Multiple Open Microphones", Eq. (18.24) (printed
  p. 699), the gain-before-feedback stability criterion generalised to several
  open microphones.
- **The print:** Z_S + L_H-M + ΔL_nom ≤ L_H-L **+** D_M(θ) − 10, with the
  microphone directivity index entering the right-hand side with a plus sign.
- **The problem:** Eq. (18.24) is the number-of-open-microphones
  generalisation of Eq. (18.20) (printed p. 698), which reads
  Z_S + L_H-M ≤ L_H-L **−** D_M(θ) − 10 and which follows in turn from the
  oscillation condition Eq. (18.19), Z_S + L_H-M = L_H-L − D_M(θ), obtained
  by substituting the feedback-loop gain G_S = L_H-M − L_H-L + D_M(θ)
  (Eq. (18.18)) into Z_S + G_S = 0 (Eq. (18.16)). Setting N_m = 1 makes
  ΔL_nom = 0, so Eq. (18.24) must reduce to Eq. (18.20) and does not. The
  sign matters physically: D_M(θ) is "usually negative" in Long's own
  definition (about −2 to −3 dB for a cardioid pointed at the talker), so as
  printed a directional microphone would *cost* gain before feedback instead
  of buying it, inverting the chapter's own conclusion that "it is prudent to
  incorporate a cardioid or hypercardioid microphone into a system".
- **Evidence:** the printed text extracts as `Z S þ L HM þ DL nom  L HL þ D M
  ðqÞ  10`, where `þ` is the ligature this PDF uses for "+" throughout (it
  also renders the unambiguous pluses of Eqs. (18.16) and (18.19)), against
  `L HL  D M ðqÞ  10` in Eq. (18.20) where the same position holds a minus.
  The minus sign is the one that reproduces Long's own worked special cases
  at N_m = 1: with Z_S = −6 dB, Eq. (18.21) gives
  L_H-M ≤ L_H-L − D_M(θ) − 4 (an omnidirectional microphone 4 dB below the
  average audience level), and Eq. (18.22) gives L_H-M ≤ L_H-L − 2 for a
  cardioid at D_M = −2 dB. Neither special case is recoverable from the
  printed Eq. (18.24).
- **Library behaviour:** `feedback_stability` in
  [`sound_reinforcement.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/electroacoustics/sound_reinforcement.py)
  implements the sign of Eq. (18.20), with a note at the criterion. Both of
  Long's special cases are pinned by regression tests
  ([`tests/electroacoustics/test_sound_reinforcement.py`](https://github.com/jmrplens/phonometry/blob/main/tests/electroacoustics/test_sound_reinforcement.py))
  and by the conformance checks "Long, Architectural Acoustics 2e,
  Eq. (18.21)" and "Eq. (18.22)".
- **Status:** unreported (textbook rather than a standard, so non-normative).

## Long, Architectural Acoustics 2e (2014), Eq. (17.53) (constant of the communication bound)

- **Location:** Chapter 17, "Restaurant Design", Eq. (17.53) (printed p. 666),
  the minimum absorption per occupied table for adequate cross-table
  communication.
- **The print:** A_tab > 6,33 r_s².
- **The problem:** the bound is Eq. (17.52),
  L_SN = 10 lg[Q/(4πr²)] + 10 lg[A_tab/4], solved for A_tab at the stated
  threshold L_SN > −6 dB, which gives A_tab > 16π·10^(−0,6)·r_s²/Q. With the
  Q = 2 the chapter uses for a talker, that constant is 6,313, not 6,33.
- **Evidence:** the immediately following Eq. (17.54) is the same closed form
  at the privacy threshold L_SN < −9 dB, and its printed constant 3,16 is
  exactly what 16π·10^(−0,9)/2 = 3,1640 gives, confirming both the formula and
  Q = 2. Only the −6 dB constant is off. Recovering 6,33 from the same closed
  form would require Q = 1,995, i.e. no consistent alternative assumption
  produces it. Long's own prose one paragraph later ("at least 6.3 or more
  square meters (68 sq ft) of absorption per table") converts to 6,317 m²,
  agreeing with 6,313 to the precision of his rounding.
- **Library behaviour:** `absorption_per_table` in
  [`crowd_noise.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/crowd_noise.py) computes the bound
  from Eq. (17.52) rather than hardcoding either constant, so both bounds stay
  mutually consistent; the 6,313 value and the printed 3,16 are pinned by
  regression tests
  ([`tests/room/test_crowd_noise.py`](https://github.com/jmrplens/phonometry/blob/main/tests/room/test_crowd_noise.py)) and
  the 3,16 constant by the conformance check "Long, Architectural
  Acoustics 2e, Eq. (17.54)".
- **Status:** unreported (textbook rather than a standard, so non-normative).
## Long, Architectural Acoustics 2e (2014), Table 14.7 (round elbow rows)

- **Location:** Chapter 14, Table 14.7, "Insertion Loss of Round Elbows"
  (printed p. 541), indexed by the frequency-width product f w (kHz times
  inches).
- **The print:** four rows only: f w < 1,9 → 0 dB; 1,9 < f w < 3,8 → 1 dB;
  3,8 < f w < 7,5 → 2 dB; f w > 15 → 3 dB.
- **The problem:** the band 7,5 < f w < 15 has no row, while the neighbouring
  Tables 14.5 and 14.6 (square elbows, same source and same index) both carry
  six rows covering it. A duct-borne calculation lands in that band routinely:
  a 24 in elbow at 500 Hz has f w = 12.
- **Evidence:** the same data adapted from the same ASHRAE source appear in
  Bies, Hansen & Howard, *Engineering Noise Control* 5e, Table 8.11, indexed
  by W/λ (= 0,074 f w). Its round-elbow column has six rows and gives 3 dB for
  0,55 ≤ W/λ < 1,11, which is exactly the 7,5 < f w < 15 band Long omits; the
  other five rows of the two tables agree entry for entry.
- **Library behaviour:** `elbow_insertion_loss` in
  [`hvac.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) carries the six-row
  round column with 3 dB in the missing band, pinned by
  `test_elbow_tables_by_frequency_width_product`
  ([`tests/noise_control/test_hvac_long.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_hvac_long.py)).
- **Status:** unreported (textbook rather than a standard).

## Long, Architectural Acoustics 2e (2014), Eq. 13.28 (units of U_G)

- **Location:** Chapter 13, Eq. 13.28 (printed p. 521), the normalised
  pressure-drop coefficient ξ = 334,9·ΔP/(ρ0·U_G²) of the diffuser sound-power
  model.
- **The print:** the nomenclature under the equation gives "U_G = flow
  velocity prior to the diffuser (ft/min)" and, on the next line,
  "= Q/(60·S_G) (for Q in cfm)".
- **The problem:** the two statements contradict each other. Q in ft³/min
  divided by 60·S_G is a velocity in **ft/s**, not ft/min, and only the ft/s
  reading makes the constant right: 334,9/ρ0 with ρ0 = 0,075 lb/ft³ is
  4465·ΔP/U², which is the standard velocity-pressure relation
  ΔP/(U/4005)² only when U is converted from ft/s. Read as ft/min the
  coefficient comes out 3600 times too small, and Eq. 13.27 then gives
  −67 dB for the worked diffuser of Table 14.9 instead of its printed 33 dB.
  Eq. 13.27 itself declares U_G in ft/s, so the "(ft/min)" label under
  Eq. 13.28 is the odd one out.
- **Evidence:** dimensional check of Q/(60·S_G); reconstruction of the
  334,9/ρ0 constant from the velocity-pressure relation; and the worked
  diffuser row of Table 14.9, which the ft/s reading reproduces to better
  than 1 dB in the five bands that carry a level while the ft/min reading
  misses it by 100 dB.
- **Library behaviour:** `diffuser_sound_power` in
  [`hvac.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) reads U_G in ft/s
  internally (SI at the interface), with the Table 14.9 row pinned by
  `test_diffuser_sound_power_reproduces_the_table_14_9_row`
  ([`tests/noise_control/test_hvac_long.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_hvac_long.py))
  and the conformance check "Long 2e Eqs. 13.27-13.33".
- **Status:** unreported (textbook rather than a standard).

## Norton & Karczub, Fundamentals of Noise and Vibration Analysis for Engineers 2e (2003), Eq. (6.56)

- **Location:** Section 6.6.1, Eq. (6.56), the coupling loss factor of two
  homogeneous plates joined by ``N`` point connections (printed p. 418).
- **The print:** the denominator bracket
  ``(rho_s1^2 h1^2 cL1^2 + rho_s2^2 h2^2 cL2^2)`` appears to the first power.
- **The problem:** as printed the expression is not dimensionless. The
  prefactor ``4 N h1 cL1 / (sqrt(3) omega S1)`` already has the dimensions of
  ``m^2 s^-1`` over ``m^2 s^-1``, i.e. unity, so the remaining ratio of the two
  bracketed products must be dimensionless too. That requires the sum to be
  squared, ``A1 A2 / (A1 + A2)^2``.
- **Evidence:** the book's own answer to problem 6.13 (printed p. 617). With
  the squared denominator the twelve-bolt aluminium pair gives
  eta_12 = 1,43·10⁻² at 125 Hz against the printed 1,44·10⁻², and matches the
  whole 125 Hz to 2 kHz column to better than 0,7 %; with the printed
  (unsquared) denominator the result is not a loss factor at all.
- **Library behaviour:** `point_connection_coupling_loss_factor` in
  [`junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/vibration/junction_transmission.py)
  implements the squared form, with the printed column pinned by a regression
  test ([`tests/vibration/test_junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/test_junction_transmission.py))
  and a note at the formula.
- **Status:** unreported (textbook rather than a standard).

## Norton & Karczub 2e (2003), problem 6.13 answer (eta_21 column)

- **Location:** Answers to problems, problem 6.13 (printed p. 617), the two
  ``eta_21`` columns of the welded and bolted tables.
- **The print:** for the two aluminium plates (plate 1: 3 mm, 2,5 m × 1,2 m;
  plate 2: 5,5 mm, 2,0 m × 1,2 m) the answer gives, at 125 Hz,
  eta_21 = 5,77·10⁻³ (welded) and 2,64·10⁻² (bolted).
- **The problem:** both columns are exactly the corresponding ``eta_12``
  column multiplied by ``h2/h1 = 1,833``. The SEA consistency relationship is
  ``n_1 eta_12 = n_2 eta_21`` (Eq. 6.8) with the flat-plate modal density
  ``n = S sqrt(12) / (2 cL h)`` of Eq. (6.25), so the correct factor is
  ``n_1/n_2 = (S_1 h_2)/(S_2 h_1) = 2,292``. The printed column drops the plate
  area ratio ``S_1/S_2 = 1,25``.
- **Evidence:** the ratio of the printed columns is 1,8333 to five digits in
  every band of both tables, which is ``h2/h1`` exactly; the ``eta_12`` columns
  themselves reproduce from Eqs. (6.52) to (6.56) to better than 0,7 %.
- **Library behaviour:** the ``eta_12`` columns are used as the regression
  oracle; ``eta_21`` is obtained from Eq. (6.8) with the full modal densities,
  and a test pins the 2,292 ratio explicitly
  ([`tests/vibration/test_junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/test_junction_transmission.py)).
- **Status:** unreported (textbook rather than a standard).

## Norton & Karczub 2e (2003), problem 6.10 (platform area)

- **Location:** Problems, problem 6.10 (printed pp. 593-594) and its answer
  (printed p. 617): a satellite platform coupled to an aluminium cylinder,
  500 Hz octave, printed answers eta_12 = 4,26·10⁻⁴, eta_21 = 3,92·10⁻⁴ and
  Pi_in = 1,31 W.
- **The print:** the statement gives the aluminium platform as
  "5 mm thick and 3,5 m × 3 m", i.e. 10,5 m².
- **The problem:** that area is inconsistent with the three printed answers.
  Eq. (6.12) fixes ``E_1/E_2 = (eta_2 + eta_21)/eta_12 = 6,554`` from the
  printed loss factors alone, whereas the stated geometry with the printed
  velocities (27,2 and 13,2 mm/s) gives 7,88. The energy ratio is independent
  of the modal densities and of the wave speed, so no choice of those can
  reconcile it; only the platform area can. The area the answers imply is
  8,73 m², which is 3,5 × 3 minus the π(0,75 m)² footprint of the cylinder
  that Fig. P6.10 shows standing on the platform.
- **Evidence:** with 8,73 m² the inversion of Eqs. (6.15), (6.8) and (6.10)
  returns eta_12 = 4,256·10⁻⁴, eta_21 = 3,910·10⁻⁴ and Pi_in = 1,306 W, i.e.
  all three printed answers within 0,4 %; the cylinder's own energy and modal
  density come out unchanged either way.
- **Library behaviour:** `power_injection_clf` in
  [`experimental_sea.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/vibration/experimental_sea.py)
  implements the inversion as published; the regression test uses the free
  platform area and documents the discrepancy
  ([`tests/vibration/test_experimental_sea.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/test_experimental_sea.py)).
- **Status:** unreported (textbook rather than a standard).

## Norton & Karczub 2e (2003), problem 3.14 (structural loss factor)

- **Location:** Problems, problem 3.14 (printed p. 580) and its answer
  (printed p. 611): the octave-band transmission loss of a 20 mm particle
  board panel.
- **The print:** the statement gives the panel a structural loss factor of
  "~1,5 × 10⁻²"; the answer gives 27 dB at 8 kHz and 38,6 dB at 16 kHz.
- **The problem:** those two values are above the panel's critical frequency
  (4885 Hz for Appendix 4 particle board, fc·t = 97,7 m/s) and therefore
  follow Cremer's Eq. (3.110), which contains ``10 lg(eta)``. With
  eta = 1,5·10⁻² the equation gives 37,0 dB and 48,5 dB, ten decibels above
  the printed answers; with eta = 1,5·10⁻³ it gives 27,0 dB and 38,5 dB.
- **Evidence:** the 10 dB offset is exactly one decade of ``10 lg(eta)``, and
  the frequency dependence of the printed pair independently fixes
  fc = 4939 Hz against the Appendix 4 value of 4885 Hz. The eight values below
  coincidence reproduce exactly from Eq. (3.104) and do not involve eta.
- **Library behaviour:** the regression test uses eta = 1,5·10⁻³, the value
  the printed answers require
  ([`tests/building/test_panel_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/test_panel_transmission.py)).
- **Status:** unreported (textbook rather than a standard).

---

## Vigran, Building Acoustics (2008), Eq. (9.18) (receiving-side coefficient)

- **Location:** Section 9.2.3.2, Eq. (9.18) (printed p. 339), the transmission
  factor of the one-dimensional suspended-ceiling plenum model after
  Mechel (1980).
- **The print:** the denominator reads ``mS LS · mR LR h`` with the **unprimed**
  ``mR``, while the exponent of the same expression carries the primed
  ``m'R = mR + sR tauR / h`` of Eq. (9.17).
- **The problem:** the two sides of the plenum are integrated the same way. The
  receiving-side integral is
  ``int_0^LR exp(-eps m'R x) dx = (1 - exp(-eps m'R LR)) / (eps m'R)``, so the
  factor that normalises it must be ``m'R LR``, exactly as the source-side one
  is ``mS LS``. Read literally, the printed expression is not a transmission
  factor at all: it carries a spurious ``m'R / mR = 1 + sR tauR / (h mR)``, so
  it grows without bound as the plenum damping falls. Two consequences are
  visible with ordinary inputs (``LS = LR = 5 m``, ``h = 0,6 m``,
  ``RS = RR = 25 dB``, ``eps = 2``, ``sS = sR = 0,5``): the model becomes
  **non-monotonic in the damping**, giving ``Rcl = 40,26 dB`` at
  ``mR = 0,01 1/m`` but only ``26,48 dB`` at ``10^-4``, i.e. adding plenum
  absorber is predicted to make the flanking path worse than leaving it bare;
  and it **breaks energy conservation**, returning ``tau_cl = 4,45`` at
  ``RS = RR = 6 dB``, ``mR = 0,01`` and ``tau_cl = 829`` at ``RS = RR = 0 dB``,
  ``mR = 10^-3``.
- **Evidence:** with ``m'R`` in the denominator every one of those pathologies
  disappears: ``tau_cl`` is monotonically decreasing in the damping, is bounded
  above by 1 because ``(1 - exp(-eps m'R LR)) / (eps m'R LR) <= 1``, and reduces
  to Vigran's own small-attenuation result, Eq. (9.19)
  ``tau_cl = eps^2 tauS tauR LR / (4h)``, whenever ``mS LS`` and ``m'R LR`` are
  both small. With the printed ``mR`` the same limit picks up the factor
  ``m'R / mR``, which diverges, so Eq. (9.18) as printed does not reduce to
  Eq. (9.19) at all: the two equations the book presents as a pair are
  inconsistent with each other.
- **Library behaviour:** `plenum_flanking_reduction_index` in
  [`ceiling_plenum.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/building/ceiling_plenum.py)
  implements the derived ``m'R`` in both the exponent and the denominator, with
  the reading documented at the formula, and rejects a transmission factor
  above unity rather than reporting a negative sound reduction index. Tests pin
  the monotonicity, the ``tau_cl <= 1`` bound, the convergence to Eq. (9.20)
  and the size of the Eq. (9.17) leakage term at a realistic ceiling
  ([`tests/building/test_ceiling_plenum.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/test_ceiling_plenum.py)).
- **Status:** unreported (textbook rather than a standard). Mechel's original
  1980 paper, which Vigran reproduces, was not available to check whether the
  misprint originates there.

## Real Decreto 1367/2007, Annex IV A.3.3 (Kf and Ki threshold tables)

- **Location:** Annex IV, section A.3.3, the ``Kf`` (low-frequency) and
  ``Ki`` (impulsive) correction tables, middle row of each.
- **The print:** both tables print the 3 dB row as "Si 10 > Lf <= 15" and
  "Si 10 > Li <= 15" respectively (BOE-A-2007-18397, consolidated text).
- **The problem:** the condition as printed is unsatisfiable. It reads
  "10 greater than Lf" and "Lf at most 15" simultaneously, which would
  select levels below 10 dB, but the row above it already assigns those to
  0 dB ("Si Lf <= 10") and the row below covers "Si Lf > 15". The three
  rows only partition the range under the reading `10 < Lf <= 15`, so the
  ">" is a typeset inversion of "<".
- **Evidence:** the bracketing rows leave no other consistent reading; the
  identical construction appears in both tables, and the equivalent tables
  in the autonomous-community noise regulations that transpose this Annex
  print `10 < Lf <= 15`.
- **Library behaviour:**
  [`low_frequency_correction`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environmental/spanish_regulation.py)
  and `impulsive_correction` implement `10 < L <= 15`, with a regression
  test pinning the three branches at the 10 dB and 15 dB boundaries.
- **Status:** unreported (national regulation, not a standards body).

---

## Commission Directive (EU) 2015/996, Annex II 2.2.1 (octave-band range of the road source)

- **Location:** the Annex, point 2.2.1, second paragraph under the heading
  "Traffic flow" (OJ L 168, 1.7.2015, p. 8).
- **The print:** "these sound power levels are calculated for each octave band
  i from 125 Hz to 4 kHz".
- **The problem:** the road source model contradicts its own coefficient
  database. Every table of Appendix F, both in the 2015 text and in the
  version replaced by (EU) 2021/1226, is printed over the eight octave bands
  **63 Hz to 8 kHz**, and point 2.1.1 of the same Annex defines the frequency
  range of the method as 63 Hz to 8 kHz. A calculation restricted to
  125 Hz - 4 kHz would silently discard four of the eight tabulated bands.
- **Evidence:** corrected by the corrigendum published in OJ L 5, 10.1.2018,
  p. 35, which reads in full: 'On page 8, in the Annex, in point 2.2.1, in the
  second paragraph under the heading "Traffic flow": for: "each octave band i
  from 125 Hz to 4 kHz", read: "each octave band i from 63 Hz to 8 kHz"'. The
  same corrigendum also adds "octave bands" to the frequency range of 2.1.1.
- **Library behaviour:**
  [`cnossos_road`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environmental/cnossos_road.py) works over
  the corrected 63 Hz to 8 kHz grid (`ROAD_OCTAVE_BANDS`), pinned by
  `test_octave_bands_are_the_corrected_range` and by the workbook cases, whose
  published levels cover all eight bands.
- **Status:** corrected by the issuing body (corrigendum of 10 January 2018);
  recorded because the uncorrected 2015 text is still the one most often
  downloaded and quoted.

---

## Ainslie, Principles of Sonar Performance Modelling (2010), Eq. (9.57)

*Textbook, not a standard.*

- **Location:** Section 9.1.1.2.4 (printed p. 457), the transition range
  between the mode-stripping and single-mode regimes of the Weston flux model.
- **The print:** r_MS ≈ k²·He³/(9·η), where H is the water depth, He the
  Weston effective depth of Eq. (9.55), k = ω/c_w and η the reflection loss
  gradient.
- **The problem:** the sentence immediately above it prescribes the
  derivation, "estimated by equating θ_n and θ_eff with n = 3/2", and the two
  angles are printed on the same page:
  - Eq. (9.47), θ_eff = (π·H/(4·η·r))^½, with the **true water depth H** (it
    comes from the multipath integral Eq. (9.46), whose 1/(r·H) prefactor is
    the cylinder area A_CS = 2π·r·H of Eq. (9.44), so H is the depth that
    counts bottom bounces);
  - Eq. (9.56), θ_n ≈ n·π/(k·He), with the **effective depth He** (mode angles
    are set by the apparent pressure-release boundary).

  Equating them at n = 3/2 gives π·H/(4·η·r) = 9·π²/(4·k²·He²), that is
  **r_MS = k²·He²·H/(9·π·η)**. The printed form is larger by π·He/H. The
  factor π is unconditional: it survives even if He is substituted for H in
  Eq. (9.47), which is presumably how the printed He³ arose, and that reading
  would give k²·He³/(9·π·η), still π below the print. The residual He/H is the
  depth substitution itself, and it tends to 1 at high frequency. The other
  transition of the same section, Eq. (9.50) r_CS = π·H/(4·η·ψc²), follows its
  own derivation exactly (it is where Eq. (9.42) and Eq. (9.49) cross), so the
  defect is confined to Eq. (9.57).
- **Evidence:** the symbolic re-derivation above, checked numerically for
  H = 50 m, f = 250 Hz, c_w = 1500 m/s over the Table 9.1 sand seabed
  (η = 0,28 Np/rad, ψc = 33,56°, He = 53,63 m, k = 1,047 m⁻¹):

  | | r_MS | θ_eff there (Eq. 9.47) |
  |---|---|---|
  | derivation, k²·He²·H/(9·π·η) | 19,9 km | 4,808° |
  | printed Eq. (9.57), k²·He³/(9·η) | 67,1 km | 2,619° |

  The ratio 67,1/19,9 is π·He/H = 3,3695 to every digit carried. The angle
  column is an independent check that does not depend on how the derivation is
  read: the first two mode angles of Eq. (9.56) are θ₁ = 3,205° and
  θ₂ = 6,410°, so θ_3/2 = 4,808°. At the derived range the effective angle is
  exactly θ_3/2, halfway between the first two modes, which is what the text
  asks for. At the printed range it has fallen to 2,619°, **below θ₁ itself**:
  the second mode would have been stripped long before, so that range cannot
  be where the single-mode regime begins.
- **Library behaviour:** `weston_regime_boundaries` in
  [`weston_regimes.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/underwater/weston_regimes.py)
  implements the derivation-consistent k²·He²·H/(9·π·η), which is also what
  keeps θ_eff defined with H everywhere the module evaluates Eq. (9.47). The
  equating rule is pinned by
  `test_mode_stripping_boundary_equates_theta_eff_with_mode_3_over_2`, which
  rebuilds both angles from the printed equations rather than from the
  implementation, and the shared definition of θ_eff by
  `test_composite_loss_and_the_boundary_use_the_same_effective_angle` (both in
  [`tests/underwater/test_weston_regimes.py`](https://github.com/jmrplens/phonometry/blob/main/tests/underwater/test_weston_regimes.py)).
- **Status:** unreported (textbook rather than a standard).

---

## NMFS (2024) Updated Technical Guidance v3.0, Table 5 / Table ES2 (otariid C)

*Regulatory guidance document, not a standard.*

- **Location:** Table 5 (printed p. 25), repeated as Table ES2 (printed p. 3):
  the auditory weighting parameter C of the otariid pinniped in-water group
  (OW / OCW).
- **The print:** C = 1,37 dB.
- **The problem:** the correct value is 1,36 dB. NMFS states so itself in the
  table's own footnote: "During the public comment period, an error was
  identified with the Navy's rounding, where this value should be 1.36,
  instead of 1.37. Because this is such a minor error and to remain consistent
  with the Navy, NMFS decided rely upon the value the Navy originally
  provided." The document therefore knowingly publishes the wrong digit.
- **Evidence:** independent recomputation of C from its own definition, the
  negated peak of W(f), with the same row's parameters a = 1,58, b = 5,
  f1 = 2,53 kHz, f2 = 43,8 kHz: C = 1,3643 dB, which rounds to 1,36. The
  published weighted TTS onset of the same row (179 dB = K + C with K = 178)
  is unaffected by the third digit.
- **Library behaviour:**
  [`marine_mammal_weighting.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/underwater/marine_mammal_weighting.py)
  implements 1,36 and keeps the printed 1,37 available as
  `WeightingParameters.c_db_as_printed`, so an assessment that must reproduce
  the published table verbatim still can. Pinned by
  `test_nmfs_2024_otariid_c_uses_the_corrected_1_36`.
- **Status:** unreported (the issuing body has already documented it).

---

## Southall et al. (2019), Aquatic Mammals 45(2), Table 7 (impulsive peak SPL)

*Peer-reviewed journal paper, not a standard.*

- **Location:** Table 7 (printed p. 156), the impulsive-noise TTS and PTS onset
  criteria; the two in-air carnivore rows PCA and OCA.
- **The print:** PCA TTS peak SPL 138 and PTS peak SPL 144; OCA TTS peak SPL
  161 and PTS peak SPL 167 dB re 20 µPa.
- **The problem:** all four are typographical errors. The authors' own errata
  (*Aquatic Mammals* 45(5), 569-572, DOI 10.1578/AM.45.5.2019.569, printed
  pp. 569-570) reprints the corrected table: PCA 155 and 161, OCA 170 and 176.
  The same errata also clarifies that the column headed "B" in Table 5 is the
  parameter b of Eq. (2).
- **Evidence:** two independent lines, one on the mechanism of the error and
  one on the value that replaces it. Note first what does *not* discriminate:
  the PTS peak = TTS peak + 6 dB rule of printed p. 156 is satisfied by the
  printed pair as well (144 − 138 = 6, just as 161 − 155 = 6), so it says
  nothing about which pair is right.
  - **Mechanism.** In each printed row the peak-SPL TTS entry is numerically
    identical to that same row's PTS-onset **SEL** entry: PCA 123 / 138 / 138 /
    144 and OCA 146 / 161 / 161 / 167, reading TTS SEL, TTS peak, PTS SEL, PTS
    peak. A value duplicated from the neighbouring column, with the +6 dB rule
    then applied to the duplicate, is the signature of a column slip, and it is
    exactly the two entries per row the errata replaces.
  - **Value.** The corrected numbers are what the article's own extrapolation
    rule produces. Printed p. 155 sets the impulsive peak-SPL TTS onset of a
    group without direct data at the hearing threshold at the frequency of best
    sensitivity f₀ plus 159 dB, and works it through for PCW: "Peak SPL TTS
    onset was estimated as 212 dB re 1 µPa (53 dB at f₀ + 159 dB)". Evaluating
    the Table 2 group audiogram at the Table 4 f₀ gives PCA −4,6 dB re 20 µPa
    at 2,3 kHz and OCA 11,4 dB re 20 µPa at 10 kHz, hence 154,4 and 170,4,
    which round to the corrected **155** and **170** and are nowhere near the
    printed 138 and 161. The same computation reproduces the three undisputed
    rows the errata does not touch (SI 219,6 against a published 220; PCW 212,5
    against 212; OCW 226,1 against 226), so the rule is validated before it is
    applied to the two rows in question.
  - **A second, unrepaired inconsistency.** Printed p. 155 also states that for
    the in-air carnivores specifically "a nominal 15 dB offset is used ...
    between the SEL-based TTS threshold and the peak SPL-based threshold",
    which reproduces the *printed* 138 and 161 from the SEL column. The errata
    resolves the conflict in favour of the +159 dB rule, so it supersedes that
    sentence as well as the table; the sentence is left standing in the article.
- **Library behaviour:** the errata-corrected values are the ones implemented
  in [`marine_mammal_weighting.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/underwater/marine_mammal_weighting.py),
  pinned by `test_southall_table_7_errata_values_are_implemented`, with the
  +159 dB rule itself checked against the audiogram for PCW, PCA and OCA in
  `test_southall_impulsive_peak_spl_is_threshold_at_f0_plus_159_db`.
- **Status:** reported by the authors themselves (errata published 2019).

## Directive (EU) 2015/996, Annex II 2.3.2 (roughness conversion in km/h)

- **Location:** the "Definition" paragraph of *Wheel and rail roughness*
  (OJ L 168, 1.7.2015, p. 19) and the first paragraph after formula (2.3.11)
  (p. 21).
- **The print:** "it shall be converted to a frequency spectrum f = v/λ, where
  f is the centre band frequency of a given 1/3 octave band in Hz, λ is the
  wavelength in m, and **v is the train speed in km/h**", and, for impact
  noise, "using the relation λ = v/f, where f is the 1/3 octave band centre
  frequency in Hz and **v is the s-th vehicle speed of the t-th vehicle type in
  km/h**".
- **The problem:** dimensionally impossible. A frequency in hertz is a speed in
  metres per second divided by a wavelength in metres; reading the speed in
  km/h places the whole roughness spectrum a factor 3,6 too low in frequency,
  which is more than an octave and a half.
- **Evidence:** both paragraphs read off rendered page images of the Official
  Journal PDF, not extracted text. The corrigendum of OJ L 5, 10.1.2018, p. 35
  replaces "km/h" by "m/s" in both places. The flow equation (2.3.2) genuinely
  does take its speed in km/h, which is what makes the misprint plausible.
- **Library behaviour:** `roughness_to_frequency` converts the speed to m/s
  before dividing, as corrected, and its docstring says so. The reference
  implementation the Commission published with the source module does the same,
  and the 123 committed workbook cases would not reproduce otherwise.
- **Status:** unreported (corrected by the issuing body in 2018).

## Directive (EU) 2015/996, Appendix G, Table G-1, second table (wrong symbol)

- **Location:** Table G-1, "Coefficients Lr,TR,i and Lr,VEH,i for rail and wheel
  roughness", second table (OJ L 168, 1.7.2015, pp. 130-131).
- **The print:** the second table is headed **Lr,VEH,i**, the same symbol as the
  first.
- **The problem:** its two columns are "EN ISO 3095:2013 (Well maintained and
  very smooth)" and "Average network (Normally maintained smooth)", which are
  the rail-roughness classes E and M of digit 2 of the track descriptor in
  Table [2.3.b]. The table is the **rail** roughness Lr,TR,i, the quantity the
  table's own title announces and which is otherwise missing from Appendix G.
- **Evidence:** read off a rendered page image of the Official Journal PDF. The
  corrigendum of OJ L 5, 10.1.2018 re-titles it Lr,TR,i, and Commission
  Delegated Directive (EU) 2021/1226 Annex point (20)(a) reprints it under that
  symbol when it replaces it.
- **Library behaviour:** `rail_roughness` returns Table G-1b as the rail
  roughness of (2.3.7) and `wheel_roughness` returns Table G-1a as the wheel
  roughness, which is the only assignment under which the classes of
  Table [2.3.b] can be reached at all.
- **Status:** unreported (corrected by the issuing body in 2018).

## Directive (EU) 2015/996, Appendix G, Table G-5, 6 350 Hz row (50 dB notch)

- **Location:** Table G-5, "Coefficients LW,0,idling for traction noise", the
  6 350 Hz row of the "Diesel locomotive (c. 2 200 kW)" pair (OJ L 168,
  1.7.2015, p. 138).
- **The print:** Source A **31,4** dB and Source B **30,7** dB.
- **The problem:** both are about 50 dB below their own neighbours in the same
  column: 90,5 / 89,5 dB at 5 000 Hz and 81,2 / 80,6 dB at 8 000 Hz. No physical
  traction source has a 50 dB notch one third of an octave wide, and no other
  column of the table has anything comparable. The leading digit 8 was lost.
- **Evidence:** read off a rendered page image of the Official Journal PDF.
  Commission Delegated Directive (EU) 2021/1226 Annex point (20)(f) replaces the
  4th column, 25th row by "81,4" and the 5th column, 25th row by "80,7",
  restoring the monotone roll-off. The same two values appear as 31,41 and
  30,71 in the IMAGINE catalogue file the Commission distributes with its
  reference source module, so the error predates the Directive.
- **Library behaviour:** ships the corrected 81,4 / 80,7 and pins them, together
  with the assertion that neither value is more than 10 dB from either
  neighbour, in `test_table_g5_carries_the_2021_correction_at_6300_hz`.
- **Status:** unreported (corrected by the issuing body in 2021).

## Directive (EU) 2015/996, Appendix G, band and wavelength labels

- **Location:** the frequency column of Tables G-3, G-5 and G-6 and the
  wavelength column of Table G-1 (OJ L 168, 1.7.2015, pp. 129-140).
- **The print:** the 1/3-octave band centres are labelled **316 Hz**,
  **3 160 Hz** and **6 350 Hz**, and the wavelengths **120 mm**, **12 mm**,
  **3,2 mm** and **1,2 mm**.
- **The problem:** neither series is the preferred one. The nominal 1/3-octave
  centres of IEC 61260-1 are 315, 3 150 and 6 300 Hz, and the R10 preferred
  numbers around those wavelengths are 125, 12,5, 3,15 and 1,25 mm. The
  Commission's own catalogue files, distributed with the reference source
  module, use the preferred wavelength series throughout.
- **Evidence:** read off rendered page images of the Official Journal PDF.
  Commission Delegated Directive (EU) 2021/1226 Annex points (20)(d), (f) and
  (g) replace the three frequency labels, and the tables it replaces outright
  carry the preferred wavelengths; but point (20) leaves **Table G-1a**
  untouched, so the wavelength labels 120, 12, 3,2 and 1,2 mm still stand in
  the consolidated text, on the one table that keeps them.
- **Library behaviour:** the frequency grid is the IEC 61260-1 one throughout.
  The wavelength grids are kept as printed, one per table, and each roughness
  spectrum is resampled on its own grid rather than forced onto a common one,
  which is what `_WAVELENGTHS_WHEEL` and `_WAVELENGTHS_STANDARD` are for; the
  difference between the two is pinned by
  `test_wheel_roughness_keeps_the_non_standard_wavelength_grid`.
- **Status:** unreported (frequency labels corrected by the issuing body in
  2021; the Table G-1a wavelength labels stand).

## Directive (EU) 2015/996, Annex II 2.3.2, curve squeal (unassigned endpoints)

- **Location:** the *Squeal* paragraph (OJ L 168, 1.7.2015, p. 21).
- **The print:** "The emission level to be used is determined for curves with
  radius below **or equal to** 500 m and for sharper curves and branch-outs of
  points with radii below 300 m", and then "squeal noise shall be considered by
  adding 8 dB for **R < 300 m** and 5 dB for **300 m < R < 500 m**".
- **The problem:** the two open intervals leave R = 300 m and R = 500 m with no
  excess at all, and R = 500 m is explicitly inside the scope the same
  paragraph has just set. A 500 m curve therefore falls out of a rule written
  to include it.
- **Evidence:** read off a rendered page image of the Official Journal PDF.
  Commission Delegated Directive (EU) 2021/1226 Annex point (4)(b) replaces the
  paragraph with a table whose intervals are closed, "R <= 300 m" and
  "300 m < R <= 500 m".
- **Library behaviour:** `curve_squeal_excess` implements the 2021 table, so
  R = 300 m returns 8 dB and R = 500 m returns 5 dB; the boundaries are pinned
  in `test_curve_squeal_rule_of_2021`.
- **Status:** unreported (corrected by the issuing body in 2021).

---

## Allard & Atalla, Propagation of Sound in Porous Media 2e (2009), Eq. (6.85)

*Textbook, not a standard.*

- **Location:** Sect. 6.5.2 (printed p. 123), the second form of the shear-wave
  velocity ratio `mu3`.
- **The print:** `mu3 = (N delta3^2 - w^2 rho11) / (w^2 rho22)`, offered as an
  alternative to Eq. (6.84), `mu3 = -rho12 / rho22`.
- **The problem:** the two printed forms are not equal. Substituting the shear
  wavenumber of Eq. (6.83), `delta3^2 = (w^2/N)(rho11 rho22 - rho12^2)/rho22`,
  into the printed Eq. (6.85) gives `-rho12^2 / rho22^2`, which is Eq. (6.84)
  multiplied by the spurious factor `rho12/rho22`. The denominator should read
  `w^2 rho12`.
- **Evidence:** the book's own derivation. Eq. (6.80), printed p. 122, is
  `-w^2 rho11 psi_s - w^2 rho12 psi_f = N grad^2 psi_s = -N delta3^2 psi_s`, so
  `(N delta3^2 - w^2 rho11) psi_s = w^2 rho12 psi_f` and therefore
  `mu3 = psi_f/psi_s = (N delta3^2 - w^2 rho11)/(w^2 rho12)`. With that reading
  the two forms agree identically wherever `rho12` is non-zero; at `rho12 = 0`
  the corrected quotient is `0/0` while Eq. (6.84) stays defined and gives
  `mu3 = -rho12/rho22 = 0`, which is the value to use there. The printed form instead
  differs from Eq. (6.84) by the factor `rho12/rho22`, so it coincides with it
  only where that ratio is exactly 0 or exactly 1. With
  `rho12/rho22 = rho0/(phi rho_eq) - 1` those two cases ask for
  `rho_eq = rho0/phi` and `rho_eq = rho0/(2 phi)`, both real; the effective
  density of a lossy porous medium is complex, so neither is ever met.
- **Library behaviour:** `biot_waves` implements Eq. (6.84) as printed, and
  `test_shear_velocity_ratio_matches_the_corrected_second_printed_form` checks
  it against the corrected Eq. (6.85) over four decades of frequency, and also
  asserts that the form exactly as printed disagrees.
- **Status:** unreported.

---

## Allard & Atalla 2e (2009), Eq. (11.48) and Table 11.1 (poroelastic layer)

*Textbook, not a standard.*

- **Location:** Sect. 11.3.3 (printed pp. 251-252), the fluid normal stress
  `sigma33^f` of a poroelastic layer and the matrix `[Gamma]` it feeds.
- **The print:** Eq. (11.48) reads

  ```text
  sigma33^f = sum (Q + R mu_i)(kt^2 + ki3^2)
              { -(A_i - A'_i) cos(ki3 x3) + j (A_i - A'_i) sin(k33 x3) }
  ```

  and Table 11.1 writes `k_{i3}` in the two columns that carry `mu1`, `D1`
  and `E1`.
- **The problem:** two independent misprints in the same equation, plus a
  subscript slip in the table.
  - The coefficient of the *symmetric* amplitude `(A_i + A'_i)` is missing:
    Eq. (11.48) attaches both terms to `(A_i - A'_i)`, which would leave the
    first and third columns of `[Gamma]` with no `sigma33^f` entry at all,
    contradicting Table 11.1, whose row 6 prints `-E1 cos(k13 x3)` and
    `-E2 cos(k23 x3)` in exactly those columns. The first term is
    `-(A_i + A'_i) cos(ki3 x3)`.
  - The sine carries `k33`, the *shear* wave-number component, inside a sum
    over the two compressional waves `i = 1, 2`. It must be `ki3`. Table 11.1
    again gives the intended reading: its row 6 has `j E1 sin(k13 x3)` and
    `j E2 sin(k23 x3)`, and zero in both shear columns, because a shear wave
    produces no dilatation and therefore no `sigma33^f`.
  - Table 11.1 prints the running subscript `k_{i3}` in its first two columns,
    which belong to the first compressional wave alone: the `mu1`, `D1` and
    `E1` in the same columns make `k_{13}` the only consistent reading.
- **Evidence:** the two readings above are forced by Table 11.1, which the same
  page declares to be the tabulation of Eqs. (11.37), (11.38) and
  (11.46)-(11.48). They are also what the stress-strain relation Eq. (11.41),
  `sigma33^f = R div u_f + Q div u_s`, gives when the displacement potentials
  of Eqs. (11.22)-(11.25) are differentiated directly.
- **Library behaviour:** the `[Gamma]` of Table 11.1 is implemented with the
  corrected readings, and
  `test_gamma_matches_the_field_rebuilt_from_the_potentials` checks all
  thirty-six of its entries at three frequencies, three depths and three angles
  of incidence against the field rebuilt from Eqs. (11.22)-(11.28) without
  going through the table.
- **Status:** unreported.

---

## Allard & Atalla 2e (2009), Sect. 6.6.3 (thickness of the second sample)

*Textbook, not a standard.*

- **Location:** Sect. 6.6.3, printed p. 129, the two glass-wool samples whose
  measured and predicted surface impedances are Figures 6.10 and 6.11.
- **The print:** the first sentence says the impedances are shown "for
  l = 10 cm and l = 5,4 cm"; two sentences later the peak of the second sample
  is placed at "860 Hz for l = 5,6 cm", and the caption of Figure 6.11 says
  "l = 5,6 cm".
- **The problem:** the two thicknesses cannot both be right.
- **Evidence:** textual, and only textual. Two printed statements carry
  5,6 cm, the sentence about the 860 Hz peak and the independent caption of
  Figure 6.11, against one carrying 5,4 cm; a single slip in the opening
  sentence is the shorter explanation than the same slip made twice.
  The numbers do **not** settle it, and this entry does not claim they do.
  The book gives no peak-finding rule, and the answer follows the rule chosen:
  - Taking the peak as the maximum of `Im(Zs)`, Eq. (6.107) on the fully
    specified Table 6.1 glass wool gives 863,5 Hz for 5,6 cm (+0,4 % against
    the printed 860) and 896,2 Hz for 5,4 cm (+4,2 %), which favours 5,6 cm.
    But the same rule puts the undisputed 10 cm sample at 480,0 Hz against
    its printed 470, a +2,1 % bias of the same size as the effect being
    resolved.
  - Taking the peak as the maximum of `|Zs - Zs,rigid|`, which is the
    departure the same paragraph describes ("close to each other, except
    around the peaks which are not predicted by the one-wave model"), the
    10 cm sample lands at 469,2 Hz (-0,2 %) and **both** printed frequencies
    then come out of the pair (10 cm, 5,4 cm): 861,2 Hz for 5,4 cm (+0,1 %)
    against 831,0 Hz for 5,6 cm (-3,4 %). That rule favours 5,4 cm.
  - Scaling the 10 cm peak is no help either, and leans the other way from
    the conclusion: 470 x (10/5,4) = 870 Hz is 10 Hz from the published 860,
    470 x (10/5,6) = 839 Hz is 21 Hz from it.
  - The agreement of "860 Hz" with "5,6 cm" is in any case partly circular,
    since both sit in the same clause: it tests that sentence against itself,
    not which of the two sentences is the misprint.
- **Library behaviour:** recorded, with no effect on the implementation.
  `test_impedance_peak_of_the_thin_layer_resolves_the_printed_thickness` pins
  the 5,6 cm peak against the published 860 Hz under the `Im(Zs)` rule and
  checks that the 5,4 cm reading is the worse of the two under that rule.
- **Status:** unreported, and the weakest of the four entries here: the
  conclusion rests on the two-against-one reading of the printed page, not on
  a computation.

---

## Allard & Atalla 2e (2009), Sect. 6.5.4 (the frame-borne velocity ratio)

*Textbook, not a standard.*

- **Location:** Sect. 6.5.4, printed p. 125, the one sentence of the book that
  quotes computed values of `mu_b` for the Table 6.1 glass wool.
- **The print:** "The ratio modulus `|mu_b|` of the velocities of the frame
  and the air for the frame-borne wave decreases from 1,0 at 50 Hz to 0,82 at
  1500 Hz."
- **The problem:** the two quoted values are the *real part* of `mu_b`, not
  its modulus. `mu_b` is complex, and the sentence names the modulus
  explicitly.
- **Evidence:** on the fully specified Table 6.1 material the model gives
  `mu_b(1500 Hz) = 0,811 + 0,473 j`. Its real part is **0,811**, 1,1 % from
  the printed 0,82; its modulus is **0,939**, 14,5 % away. Read as the real
  part, the sentence is right at both ends and describes a monotone decrease:
  `Re(mu_b)` is 1,002 at 50 Hz and passes through 0,82 at 1467 Hz, 2,2 % from
  the printed 1500 Hz. Read as the modulus it is right at neither: `|mu_b|`
  is 1,002 at 50 Hz but *rises* to 1,008 by 400 Hz before turning over, and
  only reaches 0,82 at 2634 Hz, 76 % above the printed frequency. No
  admissible reading of the printed inputs closes that gap. With the loss
  factor at 0 or at 0,2, the viscous length halved or doubled,
  `Lambda' = 2 Lambda` in place of the printed 1,1e-4 m, the resistivity
  halved or doubled, the tortuosity at 1 or the Poisson coefficient at 0,3,
  `|mu_b(1500)|` moves only between 0,874 and 1,073. The closest of the eight,
  0,874 at zero loss factor, is still 6,6 % from the printed 0,82, and it
  loses the 495 Hz branch crossing of the same section altogether; the only
  variant that keeps that crossing (`Lambda' = 2 Lambda`, 495,2 Hz) leaves
  `|mu_b|` at 0,937. Reading the sentence as `Re(mu_b)` needs no variant at
  all.
- **Library behaviour:** `biot_waves` computes `mu_b` from Eq. (6.71) as
  printed. The conformance row and
  `test_frame_borne_velocity_ratio_matches_the_two_published_values` are
  written against `Re(mu_b)`, and say so.
- **Status:** unreported.

---

## Related source properties that are not errata

Recorded here to prevent future "fixes" that would break agreement with the
published sources:

- **Francois-Garrison pure-water term:** the two published A3 cubics do not
  meet exactly at the 20 °C switch (a step of 1·10⁻⁷·f² dB/km, 0,1 dB/km at
  1 MHz). Inherent in the published coefficients.
- **Ainslie-McColm simplification:** the paper's "within 10 % of
  Francois-Garrison" claim is marginally exceeded at the extreme corners of
  its stated domain (10,4 % at −6 °C / 1 MHz; 12,3 % at 7 km depth). A
  property of the published fit; both transcriptions verified digit-for-digit.
- **CNOSSOS-EU Annex II 2.3, missing equation number:** the railway section
  numbers its formulae (2.3.1), (2.3.2), (2.3.4), (2.3.5)..., with no (2.3.3)
  anywhere in Annex II. Verified on a rendered page image of OJ L 168, p. 17,
  where (2.3.2) and (2.3.4) sit one above the other. Nothing is missing from
  the method; only the numbering skips.
- **CNOSSOS-EU corrigendum of 2018, Table G-3 column codes:** the corrigendum
  is reported to head the seven Lr,TR columns "B/S B/M B/H B/S B/M B/H B/H",
  where the first three should read "M/S M/M M/H" and the last "W", and
  Commission Delegated Directive (EU) 2021/1226 Annex point (20)(c) does
  replace that header with the corrected codes plus a new column D. It is left
  unregistered because the corrigendum itself is published only as HTML on
  EUR-Lex and no rendering of it could be obtained here, and this registry does
  not record a claim about a printed symbol that has not been read off the
  page. The 2015 print of the same table, which was read, carries descriptive
  headers ("Mono-block sleeper on soft rail pad" and so on) and no defect.
- **Long, Architectural Acoustics 2e, Chapter 17, adjacent-table level:** the
  restaurant example states that "at an adjacent table 3 m (10 ft) away, the
  direct field level from our conversation is about 54 dB", where his own
  Eq. (17.50) with the Q = 2 and L_W = 70 dB that yield his 60 dB at 1,2 m
  gives 52,5 dB. It is left unregistered because the intended reading cannot
  be established from the book: 54 dB is also what the same equation gives at
  2,5 m (54,1 dB, and 2,5 m is the table spacing the next paragraph derives),
  and what a single 6 dB distance doubling from the rounded 60 dB would give,
  while the printed "3 m (10 ft)" is self-consistent in both units and is
  repeated in the preceding paragraph. `speech_direct_level` evaluates
  Eq. (17.50) as printed, so it returns 52,5 dB there; do not "correct" it
  toward 54 dB.
- **ICAO Annex 16 EPNL constant:** the Annex's rounded constant 13 for
  uniform 0,5 s records differs from the exact −10·lg(T0) form by 0,0103 dB;
  the library uses the exact form, which the ETM's integrated reference
  reproduces to five decimals.
- **Long Table 14.9 element rows:** the worked duct-borne sheet of Chapter 14
  was produced by a commercial program, as the text introducing it states, and
  several of its element rows do not follow from the tables printed beside
  them: the fan row (90/86/82/79/77/75/71/61 dB) is not what Eq. 13.1 gives
  with the Table 13.5 forward-curved constants at that duty
  (99/99/89/84/82/77/72/67 dB, and not a level shift of it), and the
  flexible-duct row (14/14/16/15/17/22/16/13 dB) is not the Table 14.4 entry
  for 12 in by 6 ft (3/5/10/15/17/16/9 dB). The library implements the printed
  equations and tables, and uses the sheet only for what it genuinely pins,
  the cascade arithmetic; its element rows are fed in as published in
  [`tests/noise_control/test_duct_path.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_duct_path.py).
  The sheet's own rounding is likewise not always self-consistent (supply row
  3 prints a *Sum* of 49 dB at 500 Hz where 76 − 28 = 48, then a *Combined*
  consistent with 48), which is why the comparison runs at the 1 dB the
  printed sheet carries.

<!-- END GENERATED BODY -->
