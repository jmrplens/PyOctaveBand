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

A claim that turns on the exact characters of a formula, constant, coefficient,
symbol, inequality or table cell is verified against a **rendered image** of the
cited page, and its Evidence bullet records that render: source file, PDF page
index, printed folio and dpi. Extracted text may locate a page; it is never
quoted as "the print", because PDF text layers delete glyphs silently (most of
the sources cited here emit no `√` at all, so `f_T/√2` extracts as `f_T/2`).
The page offset of each document is established empirically, because it differs
per document and drifts between chapters of the same book. Entries that rest on
something other than a render say so, either in a leading notice or on the
allowlist of
[`scripts/check_errata_evidence.py`](https://github.com/jmrplens/phonometry/blob/main/scripts/check_errata_evidence.py),
which is the check that enforces the rule; see
[CONTRIBUTING.md](https://github.com/jmrplens/phonometry/blob/main/CONTRIBUTING.md#6-filing-an-errata-entry).


---

## ISO 717-2:2020, Annex C, example C.1 (CI of the bare floor)

- **Location:** Annex C, Table C.1 (printed p. 17) and the accompanying $C_I$
  computation printed in the same cell.
- **The print:** `Ln,sum = 83,523 8… = 84 dB` and `CI = 84 − 15 − 79 = −10 dB`
  for the bare-floor example.
- **The problem:** two independent defects in the same cell. (a) Clause A.2.1
  defines $C_I$ from the energy sum over 100 Hz to 2500 Hz (the first fifteen
  one-third-octave bands); the printed value only reproduces if the 3150 Hz
  band is included, contradicting A.2.1. The correct sum over 100 Hz to
  2500 Hz is 83,2613 dB, rounded 83, giving $C_I = -11$. (b) Even read as
  the sixteen-band sum the printed digits are wrong in the last place: the
  bare-floor $L_n$ column sums to 83,523 4 dB, not the printed 83,523 **8** dB.
  The defect is confined to that cell, since the with-covering column of the
  same table prints `Ln,sum = 76,059 3…` and recomputes to 76,059 29 dB,
  reproducing every printed digit. Neither (a) nor (b) changes the rounded
  84 dB, so only (a) moves $C_I$.
- **Evidence:** independent recomputation of both sums from the printed
  per-band levels (16 bands 83,523 37 dB, 15 bands 83,261 27 dB, with-covering
  16 bands 76,059 29 dB); the 2013 edition of the same example prints
  $C_I = -11$. Render: `plan/[ISO 717-2_2020] Acoustics Rating of sound
  insulation in buildings and of building elem.pdf`, PDF page 23, printed
  p. 17, 200 dpi.
- **Library behaviour:** implements A.2.1 as written and pins $C_I = -11$ with
  the 2013 print as the oracle ([`tests/reference_data.py`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data.py),
  conformance check
  "ISO 717-2 Annex C, Table C.1").
- **Status:** unreported.

## ISO 717-2:2020, Annex C, example C.2 (covered floor: 800 Hz value and CI chain)

- **Location:** Annex C, Table C.2 (printed p. 18), the $\Delta L_w$ /
  $\Delta L_\text{lin}$ worked example.
- **The print:** (a) the 800 Hz reference-floor value is printed as 71,0 dB;
  (b) the CI line prints "Ln,sum = 75,252 7… = 75 dB" and "CI = 75 − 15 −
  63 = −3 dB", feeding "ΔLlin = 78 − 11 − (63 − 3) = 7 dB".
- **The problem:** two independent defects. (a) The normative Table 4
  reference floor is 71,5 dB at 800 Hz, and the column itself is a clean
  +0,5 dB per one-third octave ramp from 67,0 dB at 100 Hz to 72,0 dB at
  1000 Hz, which the printed 71,0 dB breaks by repeating the 630 Hz cell.
  The misprint propagates along its own row and into the table's total,
  three further cells the table prints and this entry previously did not
  name: the $L_{n,r,0} - \Delta L$ cell at 800 Hz is printed 64,0 dB
  ($= 71{,}0 - 7{,}0$) where 71,5 gives 64,5; the unfavourable deviation is
  printed 3,0 dB ($= 64{,}0 - 61$) where the corrected cell gives 3,5; and the
  printed `Sum 27,9` is the sum of the thirteen unfavourable deviations
  including that 3,0, where the corrected chain gives 28,4. None of it moves
  the rating: 28,4 dB is still below the 32,0 dB shift criterion, so
  $L_{n,w,r} = 63\ \text{dB}$ and $\Delta L_w = 15\ \text{dB}$ either way.
  (b) The printed 75,2527 dB is exactly the energy sum of the *wrong column
  over the wrong range*: the measured floor "with covering" over all sixteen
  bands 100 Hz to 3150 Hz. A.2.1 defines $C_I$ from the reference floor with
  covering (the "Ln,r,0 − ΔL" column) over 100 Hz to 2500 Hz (15 bands),
  which gives 75,674 dB (printed chain) or 75,710 dB (corrected 800 Hz cell),
  both round to 76 dB, so $C_{I,r} = 76 - 15 - 63 = -2$ either way, giving
  $C_{I,\Delta} = -11 - (-2) = -9$ and $\Delta L_\text{lin} = 6\ \text{dB}$,
  not the printed −3 / −8 / 7 dB chain.
- **Evidence:** independent recomputation of every candidate sum and of every
  cell of the 800 Hz row from the printed per-band values; the printed
  75,2527 reproduces to all printed digits only as the 16-band sum of the
  with-covering column, and every other cell of the $L_{n,r,0} - \Delta L$ and
  deviation columns reproduces exactly from the printed reference floor, so
  the 800 Hz row is the only one that does not. Render: `plan/[ISO
  717-2_2020] Acoustics Rating of sound insulation in buildings and of
  building elem.pdf`, PDF page 24, printed p. 18, 200 dpi.
- **Library behaviour:** derives the covered reference floor from the
  normative Table 4 values and sums per A.2.1, pinning
  $\Delta L_w = 15\ \text{dB}$ and $C_{I,\Delta} = -9$; the conformance check
  notes the provenance explicitly.
- **Status:** unreported.

## ISO 2631-5:2018, Annex C worked examples (male displayed formula, female R)

- **Location:** Annex C: the displayed male worked example (82 kg male,
  $m_z = 0{,}029\ \text{MPa}/(\text{m/s}^2)$, printed p. 19) and NOTE 5
  (64 kg female, $m_z = 0{,}025\ \text{MPa}/(\text{m/s}^2)$, printed p. 20).
- **The print:** (a) the male example is displayed as
  `R = { Σ(i=0..20−1) [ 1,62 MPa (120)^(1/6) / (6,75 MPa − 0,052 MPa (20+i)) ]^6 }^(1/6) ≈ 1,22`;
  (b) NOTE 5 states R = 0,97 for the female case.
- **The problem:** two independent defects. (a) The displayed male formula
  omits the $-S_{\text{stat},i}$ term that normative Formula (C.3) puts in
  the denominator, and that the same annex fixes at
  $S_\text{stat} = 0{,}029 \cdot 9{,}81 = 0{,}281\ \text{MPa}$ in the
  sentence that follows the where-list of Formula (C.3). Evaluated exactly
  as displayed the sum gives $R = 1{,}1497$, which prints as 1,15, not the
  printed 1,22; restoring the missing term gives 1,2168 with the printed
  $S_\text{stat} = 0{,}281\ \text{MPa}$ and 1,2177 with the exact
  $m_z \cdot 9{,}81 = 0{,}2845\ \text{MPa}$, i.e. the printed 1,22 either
  way. The printed *result* is therefore right and the printed *formula* is
  not. (b) Exact recomputation of Formula (C.3) with NOTE 5's own inputs
  ($m_z = 0{,}025$, age coefficient 0,039, $b = 20$, $n = 20$, $N = 120$)
  gives $R = 0{,}9621$, which rounds to 0,96; the same code reproduces the
  male example exactly, and the note's $S_d = 1{,}40\ \text{MPa}$ matches
  the exact 1,3992, so the discrepancy is confined to the last digit of the
  printed female $R$.
- **Evidence:** term-by-term recomputation of the C.3 sum under both readings
  of the denominator, with the male example as the discriminator: the
  printed 1,22 is reachable only with $-S_\text{stat}$, and 1,15 only
  without it. Renders:
  `plan/[ISO 2631-5_2018] Human exposure to whole-body vibration - Part 5
  Multiple shocks.pdf`, PDF page 25, printed p. 19, 300 dpi (male display),
  and PDF page 26, printed p. 20, 300 dpi (NOTE 5).
- **Library behaviour:** implements Formula (C.3) as written, with
  $-S_\text{stat}$; the male anchor pins 1,22 and the female test anchor keeps
  the printed 0,97 with a tolerance that documents the recomputed 0,9621.
- **Status:** unreported.

## EN 12354-1:2000 Formula (E.5) / ISO 12354-1:2017 E.3.4 (K24 clamp misprint)

- **Location:** EN 12354-1:2000, Annex E, the wall-junction-with-flexible-
  interlayers block printed under Figure E.5 and numbered Formula (E.5)
  (printed p. 46), and ISO 12354-1:2017, E.3.4 NOTE 4. Annex E of the 2000
  edition has only two numbered clauses, E.1 "Determination methods" and E.2
  "Empirical data", so "E.5" is a formula number, not a clause; an earlier
  revision of this entry cited it as a clause.
- **The print:** `K24 = 3,7 + 14,1 M + 5,7 M² dB; 0 ≤ K24 ≤ −4 dB ;
  0 dB / octave`, i.e. the bound on the K24 junction term is an empty
  interval; the 2017 edition repeats the 2000 misprint verbatim.
- **The problem:** the interval is impossible as printed; the accompanying
  figure and the physics (the term is a reduction bounded below) indicate
  $-4\ \text{dB} \le K_{24} \le 0\ \text{dB}$.
- **Evidence:** the Figure E.5 curve family on the same page runs the $K_{24}$
  branch from 0 dB down to about −4 dB over the plotted mass ratios, which is
  the interval read in the other order. Render: `plan/[BS EN 12354-1_2000] --
  Building acoustics. Estimation of acoustic performance in buildings from the
  performance of elements. Airborne sound insulation between rooms.pdf`, PDF
  page 48, printed p. 46, 250 dpi.
- **Library behaviour:** implements the clamp as $-4 \le K_{24} \le 0$ with a
  misprint note in the docstring.
- **Status:** unreported.

## EN 12354-1:2000, Figure E.9 (E.7) (K24 stated in the figure-axis mass ratio)

- **Location:** Annex E, Figure E.9 / Formula (E.7) (junction of lightweight
  double leaf wall and homogeneous elements), the $K_{24}$ line.
- **The print:** K24 = 3,0 − 14,1 M + 5,7 M² dB (for m2/m1 > 3), under a
  figure whose x-axis is m2/m1.
- **The problem:** Annex E defines $M$ per transmission path as
  `M = lg(m'⊥,i/m'i)` (perpendicular element over the element
  carrying the path). The $K_{24}$ path 2→4 is carried by the homogeneous
  element ($m_2 = m_4$) with the leaf ($m_1$) perpendicular, so the per-path
  $M$ is $\log_{10}(m_1/m_2)$ — but the printed $K_{24}$ line only matches its own
  figure's curve when $M$ is read as the x-axis variable $\log_{10}(m_2/m_1)$
  (e.g. −2,4 dB at $m_2/m_1 = 3$, −5,4 dB at 10). Read with the annex's
  declared $M$, the line contradicts the figure by
  $28{,}2 \cdot |\log_{10}(m_2/m_1)|\ \text{dB}$. The same edition's other $K_{24}$
  line (Figure E.5, Formula (E.5)) *does* follow the declared per-path $M$,
  so the two $K_{24}$ prints of the 2000 edition silently use different
  conventions. ISO 12354-1:2017 E.3.5 prints the relation consistently in the
  per-path convention of its Formula (E.3),
  `K24 = 3,0 + 14,1 M + 5,7 M²`; the two editions agree numerically
  (an earlier revision of this entry read the 2017 print as a sign misprint —
  re-derivation against both editions' figures shows it is a convention
  recast, not a defect of the 2017 text).
- **Evidence:** numerical evaluation of both forms against the Figure E.9
  curve. Renders: `plan/[BS EN 12354-1_2000] -- Building acoustics. Estimation
  of acoustic performance in buildings from the performance of elements.
  Airborne sound insulation between rooms.pdf`, PDF page 50, printed p. 48,
  200 dpi (Figure E.9 with its m2/m1 axis and the Formula (E.7) block), and
  PDF page 48, printed p. 46, 250 dpi (the Formula (E.5) block, whose K24 line
  does follow the declared per-path M); `plan/[ISO 12354-1_2017] airborne
  insulation between rooms prediction.pdf`, PDF page 53, printed p. 47,
  200 dpi (E.3.5, where the 2017 edition prints
  `K24 = 3,0 + 14,1 M + 5,7 M²` beside a Figure E.7 that carries no mass-ratio
  axis at all).
- **Library behaviour:** implements the per-path convention uniformly
  (`junction_vibration_reduction`, mass_ratio = $m'_{\perp,i}/m'_i$ for every
  branch), so the E.7 double-leaf branch takes leaf-over-homogeneous ratios
  below 1/3 and evaluates $3{,}0 + 14{,}1 M + 5{,}7 M^2$.
- **Status:** unreported.

## EN 12354-2:2000, Formula (3) vs Annex E.3 (standardized impact level)

- **Location:** Formula (3) and worked example E.3.
- **The print:** Formula (3) defines L'nT = L'n − 10 lg(0,16·V/(A0·T0)), which
  reduces exactly to L'n − 10 lg(0,032·V), i.e. a reference volume of
  31,25 m³. Annex E.3 states "from equation (3): L'nT,w = L'n,w − 10 lg(V/30)".
- **The problem:** the annex's $V/30$ is a rounding of the formula's own
  constant; the two differ by a constant 0,177 dB.
- **Evidence:** direct algebra; both variants recomputed for the E.3 case
  (42,959 vs 42,782 dB, both rounding to 43 in that example). Render:
  `plan/[BS EN 12354-2_2000] -- Building acoustics. Estimation of acoustic
  performance in buildings from the performance of elements. Impact sound
  insulation between rooms.pdf`, PDF page 7, printed p. 5, 200 dpi.
- **Library behaviour:** implements the exact $0{,}032 \cdot V$ form and
  documents the annex's rounding.
- **Status:** unreported.

## EN 12354-3:2000, Formula (5) (reduced form of the normalized level difference)

- **Location:** clause 3.1.5 "Relations between quantities", Formula (5)
  (printed p. 6).
- **The print:** `D2m,n = D2m,nT − 10 lg[0,16 V/(T0 A0)] = D2m,nT −
  10 lg 0,32 V dB`.
- **The problem:** the reduced form is off by a factor of ten. Six lines above
  it, the where-list of clause 3.1.4 defines `A0` as "the reference
  equivalent sound absorption area, in square metres, for dwellings given as
  10 m²", and the where-list of clause 3.1.3 on the preceding page defines
  `T0` as "the reference reverberation time, in seconds, for dwellings given
  as 0,5 s". So $0{,}16/(T_0 A_0) = 0{,}16/5 = 0{,}032$, not 0,32.
  Applied as printed, the reduced form shifts every normalized façade level
  difference by exactly $10\log_{10} 10 = 10\ \text{dB}$. The exact analogue in the
  companion part, EN 12354-2:2000 Formula (3), prints the same algebra
  correctly: `L'nT = L'n − 10 lg[0,16 V/(A0 T0)] = L'n − 10 lg 0,032 V dB`.
  ISO 12354-3:2017 dropped the reduced form altogether: its Formula (5) prints
  only `D2m,n = D2m,nT − 10 lg[Csab V/(A0 T0)]` with `Csab = 0,16 s/m`.
- **Evidence:** direct algebra with the standard's own $A_0$ and $T_0$, and the
  side-by-side comparison with the correctly reduced Formula (3) of Part 2.
  Renders: `plan/[BS EN 12354-3_2000] Building acoustics - Airborne sound
  insulation against outdoor sound.pdf`, PDF page 8, printed p. 6, 600 dpi
  (Formula (5) and the `A0 = 10 m²` definition), and PDF page 7, printed p. 5,
  200 dpi (`T0 = 0,5 s`); `plan/[BS EN 12354-2_2000] -- Building acoustics.
  Estimation of acoustic performance in buildings from the performance of
  elements. Impact sound insulation between rooms.pdf`, PDF page 7, printed
  p. 5, 200 dpi (Formula (3)); `plan/[ISO 12354-3_2017] airborne insulation
  against outdoor sound prediction.pdf`, PDF page 12, printed p. 6, 300 dpi
  (the 2017 Formulae (4) and (5)).
- **Library behaviour:** unaffected. No code path implements the reduced form:
  the façade model computes $D_{2m,nT}$ from Formula (13)
  ([`facade.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/building/prediction/facade.py)),
  and the survey method converts with the unreduced
  $D_{2m,n} = D_{2m} + k + 10\log_{10}[A_0 T_0/(0{,}16 V)]$ of ISO 10052 Clause 3.15
  ([`survey_insulation.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/building/measurement/survey_insulation.py)).
  The two standardization constants that *are* pre-folded elsewhere in the
  library are both correct: $0{,}032$ for the Part 2 impact form and $0{,}32$
  for the Part 1 airborne form $D_{nT} = R' + 10\log_{10}(0{,}16 V/(T_0 S_s))$,
  where the denominator is an area rather than $A_0$.
- **Status:** unreported.

## EN 12354-3:2000, Formula (13) vs its own Annex F example (the "6" constant)

- **Location:** clause 4.1, Formula (13) (printed p. 9), against the worked
  example of Annex F (printed pp. 27-28).
- **The print:** Formula (13) gives
  `D2m,nT = R' + ΔLfs + 10 lg[V/(6 T0 S)] dB`, while the Annex F.1.3 result
  table prints a `D2m,nT` row that is exactly `R' + 1,5 dB` in all five octave
  bands and in the single-number column (25,9/23,0/26,4/36,9/39,0 against
  24,4/21,5/24,9/35,4/37,5, and 29,3 against 27,8).
- **The problem:** the *example* is self-consistent and the *formula* is the
  outlier. With the example's own inputs ($V = 50\ \text{m}^3$,
  $S = 11{,}3\ \text{m}^2$, $T_0 = 0{,}5\ \text{s}$, $\Delta L_\text{fs} = 0$),
  the Sabine form gives
  $10\log_{10}[0{,}16 \cdot 50/(0{,}5 \cdot 11{,}3)] = 1{,}5116\ \text{dB}$, which
  is the printed +1,5 dB row; Formula (13) as printed gives
  $10\log_{10}[50/(6 \cdot 0{,}5 \cdot 11{,}3)] = 1{,}6893\ \text{dB}$.
  The gap is the constant: Formula (13)'s "6" is a rounded $1/0{,}16 = 6{,}25$,
  and $10\log_{10}(6{,}25/6) = 0{,}177\ \text{dB}$ is exactly the discrepancy.
  ISO 12354-3:2017
  replaced it with an explicit Sabine constant, printing Formula (4) as
  `D2m,nT = R' + ΔLfs + [10 lg(Csab V/(T0 S))]` with `Csab = 0,16 s/m`, which
  is the constant the 2000 example already used. A previous revision of this
  entry attributed the 1,5 dB row to the example; the attribution is the other
  way round.
- **Evidence:** evaluation of both constants against the printed Annex F
  rows, which agree with 0,16 to the 0,05 dB the table carries and disagree
  with the rounded 6 by a uniform 0,18 dB; and the 2017 recast, which adopts
  the example's constant. The example's single-number result
  $D_{2m,nT,w} = 33\ \text{dB}$ is insensitive to the difference and
  reproduces either way.
  Renders: `plan/[BS EN 12354-3_2000] Building acoustics - Airborne sound
  insulation against outdoor sound.pdf`, PDF page 11, printed p. 9, 170 dpi
  (Formula (13)), PDF page 29, printed p. 27, 170 dpi (F.1.1/F.1.2 inputs) and
  PDF page 30, printed p. 28, 170 dpi (the F.1.3 result table); `plan/[ISO
  12354-3_2017] airborne insulation against outdoor sound prediction.pdf`,
  PDF page 12, printed p. 6, 300 dpi (Formula (4) with `Csab`).
- **Library behaviour:** implements Formula (13) as printed, with the rounded
  6; the test data records that the Annex F rows follow the exact 0,16
  constant and sit 0,18 dB below the model.
- **Status:** unreported.

## EN 12354-3:2000, Annex F.1.3 (the 1 kHz and 2 kHz R' cells)

- **Location:** Annex F, table F.1.3 "Results for façade" (printed p. 28),
  the `R' (equation 10)` row.
- **The print:** R' = 24,4 / 21,5 / 24,9 / 35,4 / 37,5 dB at
  125 / 250 / 500 / 1000 / 2000 Hz.
- **The problem:** the last two cells do not follow from the table's own
  element rows. Formula (10), $R' = -10\log_{10} \sum \tau_{e,i}$, applied to the
  four $-10\log_{10} \tau_e$ columns printed immediately above gives
  24,41 / 21,50 / 24,86 / **35,78** / **37,99** dB. The first three cells
  reproduce to the 0,05 dB the table carries; the 1 kHz and 2 kHz cells are
  printed 0,4 dB and 0,5 dB low.
- **Evidence:** energy summation of the printed element rows band by band
  (1 kHz: 60,7 / 40,0 / 46,6 / 38,5 dB; 2 kHz: 66,7 / 41,0 / 43,6 / 44,5 dB).
  The $D_{2m,nT}$ row below is a uniform $R' + 1{,}5\ \text{dB}$ in every
  band including those two, so it inherits the same offset, and the
  single-number result $D_{2m,nT,w} = 33\ \text{dB}$ is insensitive to it and
  still reproduces. Render:
  `plan/[BS EN 12354-3_2000] Building acoustics - Airborne sound insulation
  against outdoor sound.pdf`, PDF page 30, printed p. 28, 170 dpi.
- **Library behaviour:** the test data notes the inconsistency next to the
  affected anchor.
- **Status:** unreported.

## ISO 12354-1:2017 Table L.3 / ISO 12354-2:2017 Table G.3 (perimeter sums)

- **Location:** the input-data block below Table L.3 (printed p. 81) and the
  identical block below Table G.3 (printed p. 38), which lists the perimeter
  absorption sum $\sum l_k \alpha_k$ of Formula (C.1) for the worked example.
- **The print:** one value per element *type*: separating floor 2,364 m
  (S = 20 m²), external wall 2,375 m (S = 11 m²), internal wall 1,840 m
  (S = 13,75 m²).
- **The problem:** Formula (C.1) needs one sum per *element*, and the example
  has five elements with three different areas. Only two of the three printed
  values reproduce the columns they are supposed to drive: 2,375 m with
  $S = 11\ \text{m}^2$ gives external wall 1 exactly, and 1,840 m with
  $S = 13{,}75\ \text{m}^2$ gives internal wall **2** exactly. The separating
  floor's printed 2,364 m does not reproduce its own column at any band
  (0,074 9 against the printed 0,083 1 at 50 Hz, 0,026 4 against 0,029 0 at
  500 Hz); 2,659 m does, at every band. The two elements with no printed value
  need 2,548 m (external wall 2, $S = 13{,}75\ \text{m}^2$) and 1,636 m
  (internal wall 1, $S = 11\ \text{m}^2$).
- **Evidence:** all five sums re-derived from Formula (C.4),
  $\alpha_k = \sum_j \sqrt{f_{c,j}/f_\text{ref}}\ 10^{-K_{ij}/10}$, over the
  example's own junction geometry with the unrounded Annex E indices:
  2,659 / 2,375 / 2,548 / 1,636 / 1,839 m. The derivation returns the two
  printed values that are self-consistent with their own columns (2,375 m, and
  1,839 m against the printed 1,840 m) and supplies the three that are missing
  or wrong, and every $\eta_\text{tot,situ}$ column of Table L.3 / G.3 then
  reproduces to $5 \cdot 10^{-5}$. The printed values applied to the wrong
  element of the same type miss by far more than that rounding: 2,375 m on
  external wall 2 gives 0,108 5 against the printed 0,114 9 at 50 Hz, and
  1,840 m on internal wall 1 gives 0,085 0 against 0,077 0.
- **Library behaviour:** `in_situ_total_loss_factor` takes $\sum l_k \alpha_k$
  as an input and `perimeter_absorption_coefficient` implements Formula (C.4);
  the Annex L fixture derives all five sums that way rather than using the
  printed block, and says so ([`tests/building/prediction/test_detailed_model.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/prediction/test_detailed_model.py)).
- **Status:** unreported.

## ISO 12354-1:2017 Table L.3 / ISO 12354-2:2017 Table G.3 (external wall ηint)

- **Location:** the same input-data block, external-wall line.
- **The print:** `ηint = 0,013` for the 365 mm autoclaved aerated concrete
  external walls.
- **The problem:** the example's own element specification, and Annex B
  Table B.3 for autoclaved aerated concrete, give 0,012 5. Only 0,012 5
  reproduces the tabulated $\eta_\text{tot,situ}$: at 500 Hz Formula (C.1)
  gives $0{,}012\,5 + 0{,}001\,41 + 0{,}034\,57 = 0{,}048\,5$, the printed
  value, where 0,013 would give 0,049 0.
- **Evidence:** term-by-term recomputation of Formula (C.1) for both external
  walls at every band with each candidate $\eta_\text{int}$.
- **Library behaviour:** the Annex L fixture uses 0,012 5.
- **Status:** unreported.

## ISO 12354-1:2017, Table L.4 (second path block labelled 2d)

- **Location:** Annex L, Table L.4 (printed p. 82), the right-hand block
  headed "Transmission path 2d".
- **The print:** the block gives `αi,situ` = 6,3 to 14,1, `Dv,ij,situ` = 11,0
  to 13,6 and `Rij` = 43,9 to 84,6 dB.
- **The problem:** those are the numbers of path **4d** (internal wall 2 to
  the separating floor), not of path 2d (external wall 2). Table L.1 of the
  same annex prints the whole $R_{4d}$ column, 43,9 to 84,6 dB, and the
  block's $R_{ij}$ column is that column cell for cell. What settles it band
  by band is the other two columns, which cannot be confused: external wall 2
  has $\alpha_{i,\text{situ}} = 10{,}3\ \text{m}$ at 50 Hz
  ($S = 13{,}75\ \text{m}^2$, $\eta_\text{tot} = 0{,}114\,9$) while internal
  wall 2 has 6,3 m ($\eta_\text{tot} = 0{,}070\,3$), the printed value; and
  $D_{v,ij,\text{situ}}$ follows the floor-to-internal-wall $K_{ij}$ of
  8,8 dB, which gives 11,0 to 13,6 dB, not the floor-to-external-wall 6,4 dB,
  which gives 9,6 to 11,9 dB.
- **Evidence:** independent recomputation of Formulae (10), (11) and (15) for
  both candidate paths at every band. Path 4d reproduces all three columns of
  the block, $\alpha_{i,\text{situ}}$ to 0,05 m and $D_{v,ij,\text{situ}}$ and
  $R_{ij}$ to 0,05 dB, which is the printed resolution. Path 2d departs from
  the block's $R_{ij}$ column by 0,1 dB to 7,0 dB depending on the band, and
  comes closest between 100 Hz and 160 Hz (0,5 / 0,5 / 0,1 dB), so $R_{ij}$
  alone does not identify the path over those bands;
  $\alpha_{i,\text{situ}}$ (10,3 against 6,3 m at 50 Hz) and
  $D_{v,ij,\text{situ}}$ (1,4 dB to 1,7 dB apart in every band) do.
- **Library behaviour:** the test that asserts the block builds it as path 4d
  and names the mislabelling.
- **Status:** unreported.

## ISO 12354-1:2017, Table L.1 (non-integer weighted ratings)

- **Location:** Annex L, Table L.1 (printed p. 79), the $R_w$ row and the
  sentence below it, and the corresponding $L_{n,w}$ row of ISO 12354-2:2017
  Table G.1.
- **The print:** the `Rw` row gives one decimal for every path (75,1 / 84,5 /
  70,6 / … and 57,8 in the total column) while the sentence immediately below
  states `R'w (C ; Ctr) = 57,9 (−2 ; −8) dB`.
- **The problem:** ISO 717-1 rates by shifting the reference curve **in 1 dB
  steps**, so a weighted rating is an integer; the printed one-decimal values
  are the reference curve shifted *continuously* until the sum of unfavourable
  deviations equals exactly 32,0 dB. The airborne $R_w$ row of Table L.1
  *truncates* that continuous value to one decimal while the sentence below it
  rounds, which is why the same quantity appears twice as 57,8 and 57,9; the
  impact $L_{n,w}$ row of Table G.1 rounds instead (29,58 prints as 29,6 and
  40,98 as 41,0), so the truncation is a property of the airborne row only.
  The spectrum adaptation terms inherit the offset: with the ISO 717-1 rating
  of 57 dB they are $C = -1$ and $C_\text{tr} = -7$, and
  the printed (−2 ; −8) is exactly the pair shifted by the same 0,86 dB.
- **Evidence:** a continuous-shift solve of the ISO 717-1 reference curve
  against the printed per-band spectra reproduces every printed value in both
  rows ($R_{Dd}$ 75,12 against 75,1; $R_{D1}$ 84,54 against 84,5; $R_{11}$
  70,66 against 70,6; the total 57,86 against 57,8 / 57,9; on the impact side
  $L_{n,Df1}$ 29,58 against 29,6 and the total 40,98 against 41,0), whereas
  the ISO 717-1 1 dB-step ratings of the same spectra are 75, 84, 70 and
  57 dB. Render:
  `plan/[ISO 12354-1_2017] airborne insulation between rooms prediction.pdf`,
  PDF page 85, printed p. 79, 200 dpi (the `Rw` row and the sentence below
  it).
- **Library behaviour:** `weighted_rating` / `weighted_impact_rating`
  implement ISO 717-1/-2 as written, so the detailed model returns
  $R'_w = 57\ \text{dB}$ and $L'_{n,w} = 41\ \text{dB}$ ($C_I = 2$) for the
  example; the test pins those and documents the printed values.
- **Status:** unreported.

## ISO 12354-2:2017, Table G.1 (50 Hz to 80 Hz flanking columns)

- **Location:** Annex G, Table G.1 (printed p. 36), the four $L_{n,Df}$ columns,
  50 Hz, 63 Hz and 80 Hz rows.
- **The print:** `Ln,Df1` = 47,3 / 44,9 / 46,2 dB.
- **The problem:** Table G.4 of the same annex prints the same path Df for
  external wall 1, from the same inputs, as 47,8 / 45,9 / 47,0 dB. The two
  tables cannot both be right, and from 100 Hz upwards they agree exactly.
- **Evidence:** Formula (12) evaluated from the annex's own Table G.3 columns
  ($L_{n,\text{situ}}$, $R_\text{situ}$) and the Table G.4
  $D_{v,ij,\text{situ}}$ and $\Delta L_\text{situ}$ columns gives
  47,80 / 45,85 / 46,95 dB, reproducing the printed 47,8 / 45,9 / 47,0
  of Table G.4 to 0,05 dB and Table G.1 only from 100 Hz upwards. Carrying the
  same recomputation through the whole chain puts external wall 2 low by
  0,5 dB to 1,0 dB over the same three bands and the two internal walls low by
  up to 0,5 dB at 50 Hz and 63 Hz (their 80 Hz cells agree). From 100 Hz
  upwards no flanking column deviates by more than 0,15 dB. Correcting the
  affected cells raises the printed total $L'_n$ only slightly: 58,6 to
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
  $K_{12} = 5{,}7 + 5{,}7 M^2$ with $M = \log_{10}(360/219) = 0{,}215\,6$ gives
  5,97, i.e. **6,0**, and ISO 12354-1:2017 Table L.8 prints 6,0 for the
  identical junction of the identical example. And the external wall's mass per
  unit area is 219,0 kg/m² throughout the example, not 2,19 (a factor 100).
- **Evidence:** Annex E evaluation of the corner branch; the same table's own
  other rows and the whole of ISO 12354-1 Annex L use 219,0 kg/m². Renders:
  `plan/[ISO 12354-2_2017] impact sound insulation between rooms
  prediction.pdf`, PDF page 46, printed p. 40, 200 dpi (Table G.8, whose mass
  columns are headed `m'i` and `m'orthogonal`), and `plan/[ISO 12354-1_2017]
  airborne insulation between rooms prediction.pdf`, PDF page 89, printed
  p. 83, 200 dpi (Table L.8, printing 6,0 dB and 219,0 kg/m² for the same
  junction).
- **Library behaviour:** uses 6,0 dB and 219,0 kg/m².
- **Status:** unreported.

## ISO 12354-2:2017, Table G.6 (mislabelled row)

- **Location:** Annex G, Table G.6 (printed p. 40), internal wall to
  separating floor rigid cross junction.
- **The print:** a row labelled "Ext. wall 1/2 – Int. wall 1/2" with
  m'i = 360,0, m'orthogonal = 484,0 and Kij = 11,0 dB.
- **The problem:** Table G.6 describes the *internal wall to separating floor*
  cross junction; no external wall meets it. The masses and the value are
  those of the in-line internal-wall path, and ISO 12354-1:2017 Table L.6
  prints the same row correctly as "Int. wall 1/2 - Int. wall 1/2".
- **Evidence:** the rigid-cross through branch $8{,}7 + 17{,}1 M + 5{,}7 M^2$
  with $M = \log_{10}(484/360)$ gives 10,99, the printed 11,0, for the internal wall.
  Render: `plan/[ISO 12354-2_2017] impact sound insulation between rooms
  prediction.pdf`, PDF page 46, printed p. 40, 200 dpi.
- **Library behaviour:** treats the row as the internal-wall in-line path.
- **Status:** unreported.

## ISO 12354-1:2017 Table L.10 / ISO 12354-2:2017 Table G.10 (element label)

- **Location:** the simplified-model input table of both parts, fourth row:
  Table L.10 (printed p. 84) and Table G.10 (printed p. 41).
- **The print:** ISO 12354-1 prints "Internal wall 4 (F = f = 4)";
  ISO 12354-2 prints "Internal wall 4 (f4)": the two parts label the row
  differently, and an earlier revision of this entry quoted the Part 1 form
  for both.
- **The problem:** the example has two internal walls; the element indexed
  $F = f = 4$ is internal wall **2** (5,00 m x 2,75 m,
  $S = 13{,}75\ \text{m}^2$), as the detailed-model tables of the same
  annexes label it.
- **Evidence:** the row's own $S = 13{,}75\ \text{m}^2$ and
  $l_{ij} = 5{,}0\ \text{m}$ match internal wall 2 of Table L.1 / G.1.
  Renders: `plan/[ISO 12354-1_2017] airborne
  insulation between rooms prediction.pdf`, PDF page 90, printed p. 84,
  200 dpi, and `plan/[ISO 12354-2_2017] impact sound insulation between rooms
  prediction.pdf`, PDF page 47, printed p. 41, 200 dpi.
- **Library behaviour:** none needed; the numbers are unaffected.
- **Status:** unreported.

## ISO 12354-1:2017, Table D.1 (1 600 Hz covered by two rows)

- **Location:** Annex D, Table D.1 (printed p. 39), which reads the weighted
  sound reduction index improvement of an interior lining off its resonance
  frequency.
- **The print:** the last two rows are "630 to 1 600 -> -10" and
  "1 600 <= f0 <= 5 000 -> -5".
- **The problem:** 1 600 Hz belongs to both rows, with different values, and
  Clause D.2.2 requires $f_0$ to be "rounded to the centre frequency of the
  one-third-octave band in which fo falls", so 1 600 Hz is a value the table
  is actually read at rather than an unreachable edge. Because the rounding is
  mandatory, the ambiguity is not a single point: every raw resonance
  frequency in the 1 600 Hz band, that is from 1 412,5 Hz to 1 778,3 Hz
  (ISO 266 band edges), lands on it. Every other boundary in the table is a
  distinct band centre (200, 250, 315, 400, 500 Hz), and no other pair of rows
  overlaps.
- **Evidence:** the printed table itself, on a 600 dpi render of pdf p. 45:
  the two rows are separately ruled and share the endpoint verbatim, "630 to
  1 600" and "1 600 <= f0 <= 5 000". Neither row can be discarded, because
  630 Hz to 1 250 Hz has no other entry and 2 000 Hz to 5 000 Hz has none
  either. The predecessor edition settles which reading was intended:
  EN 12354-1:2000 Table D.3 prints the same pair of rows as "630 - 1 600 ->
  -10" and "> 1 600 -> -5", strictly greater, so in 2000 exactly 1 600 Hz
  unambiguously took -10 dB. The 2017 rewrite replaced "> 1 600" with
  "1 600 <= f0 <= 5 000" while leaving "630 to 1 600" untouched, which is what
  creates the overlap.
- **Library behaviour:** `weighted_lining_improvement` returns the more
  conservative -10 dB at exactly 1 600 Hz and -5 dB above it, the 2000
  reading, with the ambiguity named in the docstring and pinned in
  [`tests/building/prediction/test_resilient_layers.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/prediction/test_resilient_layers.py).
- **Status:** unreported.

- **Related, not an erratum:** NOTE 1 of the same table sets a floor of 0 dB
  on the 30 Hz to 160 Hz branch $74{,}4 - 20\log_{10}(f_0) - R_w/2$. Inside the
  validity box Clause D.2.2 states for the table
  ($30\ \text{Hz} \le f_0 \le 160\ \text{Hz}$,
  $20\ \text{dB} \le R_w \le 60\ \text{dB}$) the branch never reaches it: its
  minimum is $74{,}4 - 20\log_{10}(160) - 60/2 = 0{,}32\ \text{dB}$. The floor is
  therefore inactive for every input the table is stated for, but it was not
  always: the 2000 edition tabulated the low branch as four discrete rows
  ending in "160 -> 28 - Rw/2", whose minimum is $28 - 60/2 = -2\ \text{dB}$,
  so NOTE 1 was operative there. The 2017 continuous fit sits 2,3 dB above it
  at that corner and left the note vestigial. The library keeps the floor
  because the note is still printed.

## ISO 15186-1, Clause 3.9, Formula (8) (sign of the 10 lg N term)

- **Location:** Clause 3.9, Formula (8) (printed p. 3), the intensity element
  normalized level difference for N small building elements measured together.
  The print read here is **BS EN ISO 15186-1:2003**, the identical-text
  British adoption; the entry previously carried the heading ":2000", the year
  of the ISO edition the library's docstrings cite, which is not the copy that
  was read.
- **The print:** DI,n,e = Lp1 − 6 − (LIn + 10 lg(Sm/A0) + 10 lg(N)), i.e. the
  10 lg N term is subtracted.
- **The problem:** the subtracted sign cannot be derived. Measuring $N$
  identical units within one measurement surface raises the transmitted
  power (and hence $L_{In} + 10\log_{10} S_m$) by $10\log_{10} N$, so recovering the
  per-unit $D_{I,n,e}$ requires *adding* $10\log_{10} N$. The pressure-based
  equivalent, ISO 10140-2:2010 Formula (6), prints exactly that correction
  ($D_{n,e} = L_1 - L_2 + 10\log_{10}(nA_0/A)$), and ISO 15186-2:2010 Formula (12)
  prints Formula (8) without any $N$ term (the $N = 1$ case, with which both
  signs agree). As printed, installing more units would *lower* the per-unit
  rating by $20\log_{10} N$ relative to the derivable value.
- **Evidence:** derivation from the diffuse-field receiving-room relation
  $L_2 = L_W + 10\log_{10}(4/A)$ against ISO 10140-2:2010 Formula (6); cross-check
  against ISO 15186-2:2010 Formula (12) and Hopkins, *Sound Insulation* (2007),
  Eq. 3.45. Render: `plan/[BS EN ISO 15186-1_2003] sound insulation via sound
  intensity lab.pdf`, PDF page 11, printed p. 3, 400 dpi, which shows the
  `+ 10 lg(N)` inside the subtracted bracket.
- **Library behaviour:** implements the derivable per-unit form
  (`intensity_element_normalized_difference`, $+10\log_{10} N$) and emits a warning
  whenever $n > 1$, where the result deviates from the print.
- **Status:** unreported.

## ISO 10848-1:2006, Clause 8.1.1, Formula (20) (spurious π in the critical frequency)

- **Location:** Clause 8.1.1, Formula (20), the thin-plate critical frequency
  used by the test-facility flanking criterion of Formula (19).
- **The print:** fc = c0² / (1,8 cL · h · π).
- **The problem:** the constant 1,8 is itself the rounded
  $2\pi/\sqrt{12} \approx 1{,}814$ of the thin-plate dispersion relation, so
  the extra $\pi$ double-counts it and would misplace $f_c$ by a factor $\pi$
  (e.g. a 100 mm concrete element with $c_L = 3500\ \text{m/s}$: 187 Hz
  without the $\pi$, 59 Hz with it, far from any measured coincidence dip).
- **Evidence:** derivation from the thin-plate dispersion relation (Hopkins,
  *Sound Insulation* (2007), Eq. 2.201, $f_c = c_0^2/(1{,}8 c_L h)$); ISO
  12354-1:2017 prints the same $\pi$-free form in its symbol definitions
  ($f_c = c_0^2/(1{,}8 c_L t)$).
- **Library behaviour:** implements the $\pi$-free form
  (`phonometry.building.measurement.flanking_transmission.critical_frequency`), with a
  misprint note in the docstring.
- **Status:** corrected upstream — ISO 10848-1:2017 (second edition) prints
  the $\pi$-free form in its Formula (5), $f_c = c_0^2/(1{,}8 h c_L)$,
  confirming the 2006 print as a misprint. No report is needed. The entry is
  retained
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

- **Location:** Clause 8 **"Reporting uncertainties"** (printed pp. 5-6), the
  where-list under Formula (10), against the worked Tables 4 and 5 (printed
  p. 7). An earlier revision of this entry called the clause "expression of
  results", which is not its printed title.
- **The print:** the where-list defines *u* as "the standard uncertainty
  determined in accordance with Clause 5, Clause 6 or Clause 7 **rounded to
  two decimal digits for absorption coefficients** or one decimal digit for
  all other quantities", and Formula (10) then forms U = k·u.
- **The problem:** the document's own Tables 4 and 5 only reproduce when $U$
  is computed from the unrounded $u$ and rounded last. Neither table prints a
  $u$ column at all (each has only the coefficient $\alpha_s$ or $\alpha_p$
  and $\pm U$ with $k = 2$), so the printed $U$ values are the whole of the
  evidence, and 11 of the 25 are unreachable under the literal clause wording.
- **Evidence:** recomputation of all 25 entries (Table 4: 20 rows, Table 5:
  5 rows) from Formula (1) with the Table 1 constants and from Formula (4)
  with the Table 2 constants, under both conventions. Round-last reproduces
  25 of 25; round-first misses 11 of 25 (63, 125, 160, 200, 250, 1250, 1600,
  2000, 3150 and 4000 Hz of Table 4, and 250 Hz of Table 5). An earlier
  revision of this entry quoted the count as "10 of 20", which is neither the
  right numerator nor the right number of entries. Render: `plan/[ISO
  12999-2_2020] building acoustics uncertainty part 2.pdf`, PDF page 13,
  printed p. 7, 200 dpi (Tables 4 and 5 with their two data columns).
- **Library behaviour:** rounds last, matching the tables; the convention is
  documented and tested.
- **Status:** unreported.

## ISO 12999-2:2020, Table 5 (octave-band data under a one-third-octave header)

- **Location:** clause 8, Table 5 "Example for the practical sound absorption
  coefficient, αp, and its expanded uncertainty under reproducibility
  conditions" (printed p. 7).
- **The print:** the frequency column of Table 5 is headed **"One-third octave
  midband frequency / Hz"** and its rows are 250, 500, 1 000, 2 000 and
  4 000 Hz.
- **The problem:** those five frequencies are the **octave**-band series of
  ISO 11654, which is what the practical sound absorption coefficient
  $\alpha_p$ is defined over; they are not a one-third-octave series, and no
  one-third octave band is missing between them. The document contradicts
  itself on the same quantity two pages earlier: Table 2, which supplies the
  $m$ and $n$ constants of Formula (4) for exactly these five frequencies, is
  headed
  "Octave midband frequency". The same header text stands over Table 4 on the
  same page, where it is correct: that table carries a genuine one-third-octave
  series, 63 Hz to 5 000 Hz in 20 rows.
- **Evidence:** the five tabulated frequencies themselves, and the
  "Octave midband frequency" header of Table 2 for the same $\alpha_p$
  constants.
  Renders: `plan/[ISO 12999-2_2020] building acoustics uncertainty part
  2.pdf`, PDF page 13, printed p. 7, 200 dpi (Tables 4 and 5, one above the other),
  and PDF page 11, printed p. 5, 200 dpi (Table 2).
- **Library behaviour:** `_TABLE2` in
  [`absorption_uncertainty.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/materials/absorption_uncertainty.py)
  is keyed by *octave* midband frequency, following Table 2 and the ISO 11654
  definition of $\alpha_p$ rather than the Table 5 header.
- **Status:** unreported.

## ISO 10052:2021, Table 4 volume-range header

- **Location:** Table 4 (reverberation-index estimator), volume-range header.
- **The print:** the header reads "60 ≤ V < 150" while the body text says the
  method applies to rooms "up to 150 m³".
- **The problem:** the boundary $V = 150\ \text{m}^3$ is included by the text
  and excluded by the header.
- **Evidence:** direct comparison of header and clause text.
- **Library behaviour:** accepts $V = 150$ (follows the text), with the
  ambiguity noted.
- **Status:** unreported.

## ISO 17208-2:2019, Clause 5 uncertainty band coverage

- **Location:** Clause 5 (representative expanded uncertainties), printed
  p. 4.
- **The print:** "5 dB for the low frequency (10 Hz to 100 Hz) bands, 3 dB for
  the mid frequency (125 Hz to 16 000 Hz) bands, and 4 dB for the high
  frequency (**>20 000 Hz**) bands".
- **The problem:** the 20 kHz one-third-octave band itself is left unassigned:
  the mid range ends at 16 kHz *inclusive* and the high range starts strictly
  above 20 kHz. ISO 17208-1:2016, from which clause 5 says the values are
  taken, prints the same three ranges with "**≥20 000 Hz**", which closes the
  gap; Part 2 degraded the ≥ to a >. The 20 kHz band is not a corner case for
  this document: ISO 17208-1 Table 1 requires the measurement to cover
  "20 000 Hz (minimum)" as its upper one-third-octave band. An earlier
  revision of this entry said "nothing covers 16 kHz to 20 kHz inclusive",
  which is wrong at the lower end: 16 kHz is covered.
- **Evidence:** the two clauses side by side. Renders: `plan/[ISO 17208-2_2019]
  Underwater acoustics - Radiated noise from ships Part 2 (source-level
  computation).pdf`, PDF page 10, printed p. 4, 250 dpi, and `plan/[ISO
  17208-1_2016] Underwater acoustics - Radiated noise from ships Part 1 (deep
  water).pdf`, PDF page 22, printed p. 16, 250 dpi.
- **Library behaviour:** applies the conservative 4 dB high-band value from
  the 20 kHz band upwards, following Part 1, with the gap documented.
- **Status:** unreported.

## ECMA-418-1:2024 (3rd edition), clause 4.1.1 NOTE 2 (upper limit of the discrete-tone range)

- **Location:** clause **4.1.1** "frequency range of interest", NOTE 2
  (printed p. 2). An earlier revision of this entry cited clause 4.1.2, which
  is the definition of "ITT equipment" and says nothing about frequency.
- **The print:** "From viewpoint of test implementation by using FFT analyser,
  the frequency range of discrete tones are between 89,1 Hz and 11 220 Hz
  inclusive, referred to *the discrete tone frequency range of interest*."
- **The problem:** every formula and table of the standard works to
  11 200 Hz: the Table 2 and Table 3 band-edge fits are stated for
  $11\,200 \ge f_t > 1\,600$, and clauses 10, 12.3 and 12.4 permit FFT data
  with $f_1 < 89{,}1\ \text{Hz}$ and $f_2 > 11\,200\ \text{Hz}$. The two
  numbers are the same quantity to different precision rather than a
  typographical error:
  $10\,000 \cdot 2^{1/6} = 11\,224{,}6\ \text{Hz}$ is the upper edge of the
  10 kHz
  one-third-octave band that closes the range of interest, which rounds to
  11 220 Hz at four significant figures and to 11 200 Hz at three. An earlier
  revision of this entry called it a typo and added that "no other clause
  mentions 11 220 Hz"; the last x-axis tick of Figure 6 (printed p. 20) is
  labelled 11220. What clause 4.1 does carry is a structural defect: 4.1.2
  "ITT equipment" repeats 4.1.1's NOTE 1 verbatim ("This range was selected to
  be identical to that of ECMA-74:2022, 3.1.3"), although 4.1.2 defines no
  range at all, and clause 10 then cross-references "NOTE 1 of 4.1.2" for the
  discrete-tone range, which is the duplicated note rather than the NOTE 2
  that states it.
- **Evidence:** the arithmetic above, and the Table 2/3 ranges and Figure 6
  axis read side by side with NOTE 2. Renders: `plan/ECMA-418-1_3rd_edition_december_2024.pdf`, PDF page 10, printed p. 2, 300 dpi (clause 4.1.1
  NOTE 2), PDF page 25, printed p. 17, 300 dpi (Table 2) and PDF page 28,
  printed p. 20, 220 dpi (Figure 6 and its 11220 tick).
- **Library behaviour:** uses the internally consistent 89,1 Hz to
  11 200 Hz range (upper end exclusive per the formulas), with a code note
  in [`tonality.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/tonality.py).
- **Status:** unreported.

## ECMA-418-1:2024 (3rd edition), Formula (21) (repeated constant term)

- **Location:** clause 12.3, Formula (21) (printed p. 17), the curve fit for
  the lower band-edge frequency $f_{1,L}$ of the lower critical band.
- **The print:** `f_1,L = C_L,0 + C_L,0 f_t + C_L,2 f_t²`.
- **The problem:** the linear coefficient repeats the constant term. The
  where-list immediately below the formula declares "`C_L,0`, `C_L,1`,
  `C_L,2` are constants given in Table 2", Table 2 tabulates a $C_{L,1}$
  column, and the parallel Formula (22) for the upper band edge prints
  `f_2,U = C_U,0 + C_U,1 f_t + C_U,2 f_t²` correctly. The misprint is
  numerically fatal, not cosmetic: over the middle fit range
  ($171{,}4 \le f_t \le 1\,600$) Table 2 gives $C_{L,0} = -149{,}5$ and
  $C_{L,1} = 1{,}001$, so the printed form returns
  $-149{,}5 - 149{,}5 f_t - 6{,}90 \cdot 10^{-5} f_t^2$, negative everywhere,
  instead of a band edge a little below $f_t$.
- **Evidence:** the formula, its own where-list and Table 2 on one page, with
  Formula (22) as the consistent control. Render: `plan/ECMA-418-1_3rd_edition_december_2024.pdf`, PDF page 25, printed p. 17, 300 dpi.
- **Library behaviour:** implements the $C_{L,1}$ reading, which is the only one
  that returns a usable band edge, with a code note in
  [`tonality.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/tonality.py).
- **Status:** unreported.

## ECMA-418-1:2024 (3rd edition), clause 11.3 (unresolved field references)

- **Location:** clause 11.3 "Determination of masking noise level" (printed
  p. 12), the sentence introducing the critical bandwidth.
- **The print:** "The critical bandwidth Δf_c is determined from Formula
  **Error! Reference source not found.Error! Reference source not found.**
  with f_0 set equal to the frequency of the discrete tone under
  investigation, f_t".
- **The problem:** two unresolved word-processor field references were
  typeset, in bold, in place of the formula numbers, and shipped in the
  published third edition. The intended targets are unambiguous from the rest
  of the sentence, which goes on to name Formulae (4) and (5) or (7) and (8)
  for the band edges: the critical bandwidth itself is Formula (2), and
  Formula (3) is the relation $f_2 - f_1 = \Delta f_c$ that turns it into band
  edges.
- **Evidence:** the rendered clause. Render: `plan/ECMA-418-1_3rd_edition_december_2024.pdf`, PDF page 20, printed p. 12, 300 dpi.
- **Library behaviour:** none required; the library implements the critical
  bandwidth from Formulae (3)/(6) directly.
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
  That DFT has the phase
  $\exp(-j \cdot \pi \cdot f_n \cdot (\tilde{s}_b - n_{ze} + n_{zb} - 1))$;
  the printed factor doubles it (and is also inconsistent with the $\pi$
  arguments of the printed sine terms of the same formula). With the printed
  phase the fitted model cannot reproduce the spectrum of a noiseless
  windowed sinusoid, contradicting the clause's own statement that the HSA
  achieves "theoretically infinite resolution for signals without noise".
- **Evidence:** independent derivation of the window DFT plus numerical
  recomputation: with $\pi$ the least-squares fit recovers the constant part,
  amplitudes and phases of synthetic noiseless envelopes to machine
  precision and the Formula (135) residual vanishes; with the printed $2\pi$
  the kernel deviates from the window DFT by amounts of the order of the
  kernel itself and the residual stays of the order of the signal energy.
- **Library behaviour:** implements the $\pi$ reading, pinned by a regression
  test on the exact recovery of synthetic line pairs.
- **Status:** unreported.

## ECMA-418-2:2025 (4th edition), clause 9.1.5, Formula (144) (bin offset)

- **Location:** clause 9.1.5, Formula (144), the modulation rate of a local
  maximum of the envelope power spectrum.
- **The print:** the rate is the three-bin amplitude-weighted centroid of
  the peak position **minus one**, scaled by Δf.
- **The problem:** clause 9.1.4 (below Formula (122)) defines the spectral
  index $k$ as mapping to the modulation rate
  $k \cdot \tilde{r}_s/\tilde{s}_b$ with $k$ starting at 0. A symmetric local
  maximum at bin $k$ has centroid $k$, and the printed formula then assigns it
  the rate $(k - 1) \cdot \Delta f$, one full bin (0,73 Hz) low, which at
  fluctuation-strength rates is fatal (a true 1,46 Hz modulation would be
  reported as 0,73 Hz). The offset is only consistent with 1-based
  spectral-line positions, contradicting the standard's own definition of $k$.
- **Evidence:** cross-check of Formula (144) against the $k$-to-rate mapping
  stated below Formula (122).
- **Library behaviour:** uses the centroid directly (no offset) with the
  0-based $k$ of Formula (122).
- **Status:** unreported.

## ECMA-418-2:2025 (4th edition), clause 9.1.7 (units of the fine-tuning constants)

- **Location:** clause 9.1.7, Formulae (149)-(152), the damped Newton fine
  tuning of the dominant modulation rate.
- **The print:** differential step Δx = 10⁻⁵, damped-step cap 2·10⁻⁴, stop
  tolerance 10⁻⁷ and an iteration limit of 40, with the starting point
  x₀ = f̃_c,imax (a rate in Hz) and the failure check
  |f_c,1,opt − f̃_c,imax| > 1,25·Δf.
- **The problem:** the constants carry no units. Read in Hz, the damped
  step is capped at $5 \cdot 10^{-5}\ \text{Hz}$ per iteration
  ($2 \cdot 10^{-3}\ \text{Hz}$ over all 40 iterations), so the tuning cannot
  move appreciably and the $1{,}25 \cdot \Delta f$
  ($\approx 0{,}92\ \text{Hz}$) failure check is unreachable; the whole clause
  would be inert. Read as normalized modulation rates $f/\tilde{r}_s$ (the
  variable in which the Formula (127) kernel frequencies are expressed), the
  same constants give a 0,075 Hz damped per-iteration cap
  ($\approx 2{,}9\ \text{Hz}$ over the 39 iterations), a
  $1{,}5 \cdot 10^{-4}\ \text{Hz}$ stop tolerance and a reachable failure
  check, all consistent with the clause's purpose.
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

- **Location:** clause 5.3.4, Formulae (10)/(11) (printed p. 9), the minimum
  edge steepness of a distinct tone.
- **The print:** the two edges are scaled differently:
  `ΔL_u = (f_T/2)·(L_Tmax − L_u)/(f_T − f_u) ≥ 24 dB` and
  `ΔL_o = f_T·(L_Tmax − L_o)/(f_o − f_T) ≥ 24 dB`.
- **The problem:** the parent standard DIN 45681:2005-03 prints
  `f_T/sqrt(2)` on **both** edges (Gleichungen (10)/(11), printed p. 14), and
  its executable Anhang J reference program does the same
  (`Frequenz(i)/Sqr(2)`). The two prints cannot both be satisfied. Neither
  ISO factor is the DIN one: on the lower edge $1/2 < 1/\sqrt{2}$, so the ISO
  print returns a level difference sqrt(2) **smaller** and is therefore
  **stricter**; on the upper edge the divisor is absent altogether, so the ISO
  print returns sqrt(2) **larger** and is **more lenient**. An earlier
  revision of this entry had the two directions the other way round and
  described the upper edge as "halved", where in fact the divisor is missing
  rather than halved. Borderline tones with one-sided edge steepness between
  $24/\sqrt{2} = 17$ and $24 \cdot \sqrt{2} = 34\ \text{dB/octave}$ flip
  classification between the two readings.
- **Evidence:** side-by-side comparison of the ISO print, the DIN 45681 print
  and the DIN Anhang J program. The DIN radicals are exactly the case the
  render rule exists for: `pdftotext` drops the `√` glyph from both DIN
  formulae, so the extracted text reads `f_T/2` and matches the ISO print,
  while the rendered page reads `f_T/√2`. Renders: `plan/[ISO PAS 20065_2016]
  Acoustics -- Objective method for assessing the audibility of tones in noise
  -- Engineering method.pdf`, PDF page 13, printed p. 9, 250 dpi, and
  `plan/DIN 45681 - Tonhaltigkeit und Tonzuschlag (63p).pdf`, PDF page 14,
  printed p. 14, 400 dpi.
- **Library behaviour:** follows the DIN/sqrt(2) reading (it matches the
  only executable reference), with the choice recorded in
  [`tone_audibility.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/tone_audibility.py).
- **Status:** unreported.

## DIN 45681:2005-03, Anhang I, Tabelle I.6, row "6 FG"

- **Location:** Anhang I, Beispiel I.2 (combustion engine, spectrum $j = 1$),
  Tabelle I.6, the combined row "6 FG" for the three tones $k = 6/7/8$
  (592,2 / 629,8 / 643,3 Hz, tone levels 78,31 / 75,00 / 79,75 dB).
- **The print:** L_T = 81,11 dB together with delta L = 9,12 dB (with
  L_S = 59,53, L_G = 76,16, a_v = -2,40 at 592,2 Hz).
- **The problem:** the two cells contradict each other. The printed
  $\Delta L = 9{,}12\ \text{dB}$ only reproduces from the *plain* Formula (17)
  energy sum of the three tone levels (82,873 4 dB):
  $82{,}87 - 76{,}16 + 2{,}40 = 9{,}11$. The printed
  $L_T = 81{,}11\ \text{dB}$ is that same sum less exactly 1,763 dB, and taken
  at face value it would give $\Delta L = 7{,}35\ \text{dB}$.
- **Evidence:** recomputation from the printed per-tone levels of
  Tabelle I.6. The offset is the discriminator and it is a constant, not a
  deduplication: $82{,}873\,4 - 81{,}11 = 1{,}763\ \text{dB}$, and 1,76 dB is
  $10\log_{10} 1{,}5$, the standard's own Hanning effective-bandwidth correction
  (clause 5.3.2). The same offset appears in the "5 FG" row of Tabelle I.10
  (printed p. 46), where the two member tones at 705,2 and 732,1 Hz have
  $L_T = 55{,}12$ and 54,23 dB, sum to 57,708 dB, and are printed as 55,95 dB,
  1,758 dB lower, and there the printed $\Delta L = 3{,}22\ \text{dB}$ follows
  the printed $L_T$ exactly ($55{,}95 - 55{,}28 + 2{,}55 = 3{,}22$), so the
  Tabelle I.10 row is internally consistent and the Tabelle I.6 row is not.
  The third combined row, "2 FG" of the same Tabelle I.6, carries no offset at
  all: its three member levels 64,56 / 67,96 / 68,63 dB sum to 72,149 dB
  against a printed 72,15 dB, and its $\Delta L$ follows. A previous revision
  of this entry
  attributed the 81,11 dB cell to the Anmerkung 2 shared-line deduplication;
  that diagnosis is unsupported, because a deduplication removes an arbitrary
  amount of energy while all the offsets observed here are the same 1,76 dB.
  Renders: `plan/DIN 45681 - Tonhaltigkeit und Tonzuschlag (63p).pdf`, PDF
  page 41, printed p. 41, 220 dpi (Tabelle I.6) and PDF page 46, printed
  p. 46, 250 dpi (Tabelle I.10).
- **Library behaviour:** `combined_tone_level` follows Anmerkung 2 (shared
  lines counted once), which reproduces the printed "2 FG" oracle; for the
  "6 FG" row only the $\Delta L$ chain is pinned, with the contradiction
  recorded in `tests/reference_data.py`.
- **Status:** unreported.

## DIN 45681:2005-03, Anhang I, Tabellen I.2 and I.10 (wrong spectrum index in a column header)

- **Location:** Anhang I, the column headers of Tabelle I.2 (printed p. 37,
  spectrum $j = 2$) and Tabelle I.10 (printed p. 46, spectrum $j = 24$).
- **The print:** every column of Tabelle I.2 is subscripted with the spectrum
  index 2 (`f_T 2,k`, `f_1 2,k`, `f_2 2,k`, `L_S 2,k`, `L_T 2,k`,
  `L_G 2,k`, `a_v 2,k`, `u_2,k`) except the audibility column, which is
  headed **`ΔL_1,k`**. Every column of Tabelle I.10 is subscripted 24
  (`f_T 24,k`, `ΔL 24,k`, `f_1 24,k`, `f_2 24,k`, `L_S 24,k`, `L_T 24,k`,
  `L_G 24,k`, `u 24,k`) except the masking column, which is headed
  **`a_v 1,k`**.
- **The problem:** both tables carry the spectrum index of the *first*
  spectrum in one column. Tabelle I.2's own caption reads "des zweiten
  Spektrums (j = 2)" and Tabelle I.10's "des 24. Spektrums (j = 24)", and the
  body values belong to those spectra: the $\Delta L$ column of Tabelle I.2 is
  the audibility of the $j = 2$ tones (8,53 dB at 627,2 Hz, which the
  Anmerkung below the table calls "die maßgebliche Differenz ΔL_2"), and the
  $a_v$ column of Tabelle I.10 is the masking index of the $j = 24$ tones.
  The index 1 is right in exactly one table of the annex, Tabelle I.6, which
  is the $j = 1$
  table of Beispiel I.2 and carries both `ΔL_1,k` and `a_v 1,k` legitimately.
- **Evidence:** the tables' own captions, their neighbouring column
  subscripts, and the Anmerkung under each. Renders: `plan/DIN 45681 -
  Tonhaltigkeit und Tonzuschlag (63p).pdf`, PDF page 37, printed p. 37,
  250 dpi (Tabelle I.2), PDF page 46, printed p. 46, 250 dpi (Tabelle I.10)
  and PDF page 41, printed p. 41, 220 dpi (Tabelle I.6, the consistent j = 1
  control).
- **Library behaviour:** none needed; the numbers are unaffected. The
  regression fixtures index both tables by their caption's spectrum.
- **Status:** unreported.

## IEC 60268-3:2013, clause 14.12.9.2 f) (DIM denominator)

- **Location:** clause 14.12.9.2, item f) (printed p. 39), the formula for the
  dynamic intermodulation distortion $d_\text{DIM}$.
- **The print:** `d_DIM = (Σ_{i=1}^{9} U'_i²)^{1/2} / U_2 × 100 %`.
- **The problem:** the denominator is one of the nine terms of its own
  numerator. Table 2 of the same clause (printed p. 38) defines `U_2` as the
  intermodulation component at `f_s − 2f_q = 8,70 kHz`, and item d)
  defines `U_1, U_2, … U_i` as exactly those components, so the sum
  $i = 1\ldots9$ runs over $U_1 \ldots U_9$ and includes $U_2$. Meanwhile the
  defining clause
  14.12.9.1 states the ratio of the r.m.s. sum of the Table 2 intermodulation
  product voltages "to the amplitude of the output voltage at the frequency
  f_s", i.e. the 15 kHz sine component $U_s$, the Otala convention, and item
  d) measures "the amplitudes of the sinusoidal signal `U_s`" precisely so
  that it can be used, which the f) formula then never does. The denominator
  should be $U_s$. An earlier revision of this entry said that "U2 is used
  throughout 14.12 for the total output voltage"; that is false, in both the
  English and the French print.
- **Evidence:** Table 2, item d) and item f) read together in both language
  columns of the bilingual edition; the historical DIM literature (Otala)
  defines the ratio to the sine amplitude. Renders: `plan/[IEC 60268-3_2013]
  Sound system equipment - Part 3_ Amplifiers.pdf`, PDF page 40, printed
  p. 38, 250 dpi (Table 2 and item a)), PDF page 41, printed p. 39, 400 dpi
  (the item f) formula), and PDF page 101, printed p. 99, 250 dpi (the French
  Tableau 2 and clause 14.12.9.1, which give `U_2` the same 8,70 kHz
  definition and the same `f_s` reference).
- **Library behaviour:** follows the 14.12.9.1 definition (reference = the
  output amplitude at $f_s$), with a code comment at the reference measurement
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
  prominence $P = 3\log_{10}[r/(\text{dB/s})] + 2\log_{10}(d/\text{dB})$.
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
  difference (LD)") and the units given for each ("dB/s" for $r$, "dB" for $d$)
  make the intended reading unambiguous.
- **Evidence:** side-by-side reading of 3.4, 3.5 and the Clause 5 symbol
  list; the units printed with each symbol contradict the clause numbers
  printed with them.
- **Library behaviour:** implements the spelled-out reading, weighting the
  onset rate by 3 and the level difference by 2 (`predicted_prominence` in
  [`impulse_prominence.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/assessment/impulse_prominence.py)),
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
  computes $A_\text{atm}$ from the ISO 9613-1 formula directly
  ([`air_absorption.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/propagation/air_absorption.py)),
  so it yields 4,15 dB/km for this condition.
- **Status:** unreported.

## ANSI S3.5-1997, Annex C worked examples (official WG S3-79 errata)

> **Not verified on a render.** ANSI S3.5-1997 is not held locally
> (`plan/SII.pdf` is the vignette of the R package `SII`, not the standard),
> so what this entry calls "the print" is the working group's own description
> of it, not a page this project has read. The recomputations below are
> independent and do reproduce, but the *printed characters* rest on the
> errata list alone. The standard is on the maintainer's pending-acquisition
> list; when a copy arrives the entry is to be re-verified against a render of
> printed pp. 21-22 and this notice removed.

- **Location:** Annex C, Table C.1 (octave-band worked example, p. 21) and
  Table C.2 (one-third-octave worked example, p. 22) of the 1997 printing.
- **The print (per the working group's errata):** (a) Table C.1, row i = 5,
  the level-distortion factor Li under Step 6 is printed as 0.10; (b)
  Table C.2, first row, the self-speech-masking slope Ci is printed as
  −45.59.
- **The problem:** both cells contradict the standard's own normative
  formulas. (a) Clause 5.7 with the example's inputs ($E'_5 = 20\ \text{dB}$,
  $U_5 = 9.33\ \text{dB}$) gives $L_5 = 1 - (20 - 9.33 - 10)/160 = 0.9958$,
  which prints to two decimals as 1.00, not 0.10. (b) Clause 5.4 with the
  example's inputs ($B_1 = 40\ \text{dB}$, $f_1 = 160\ \text{Hz}$) gives
  $C_1 = -80 + 0.6 (40 + 10\log_{10} 160 - 6.353) = -46.587$, which prints as
  −46.59, not −45.59; the example's $Z_i$ column is only consistent with the
  corrected slope ($Z_2$ recomputes to 34.658 = printed 34.66 dB, whereas the
  misprinted
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
  missing from Eq. 23, and two Annex B fixes, a cross-reference "B16" that
  should read "B15" and a wording change about the audio-visual
  approximation); none of those touches a formula this library implements.
  The list is archived locally as
  `plan/sii-asa-wg-s3-79/sii.to-errata-2026-07-30.txt`. It is not a page
  render and cannot substitute for one, which is why this entry carries the
  notice above.
- **Library behaviour:** unaffected; the library computes the corrected
  values from the normative clauses and always did. Its Annex C.2 anchors
  ([`tests/reference_data.py`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data.py),
  `ANSIS3_5_ANNEX_C1*` and `ANSIS3_5_ANNEX_C2*`) pin the errata-consistent
  chain of both examples, cross-checked to double precision against the working
  group's own reference implementation `SII.C` and its published test-case
  results. The Table C.1 cell is pinned directly: the level-distortion factor
  of clause 5.7 for row $i = 5$ of the Annex C.1 octave-band example computes to
  0.99581, which prints as the corrected 1.00.
- **Status:** published corrections by the issuing working group; nothing
  to report upstream.

## ANSI S3.5-1997, captions of Tables 1 to 4 (official WG S3-79 erratum)

> **Not verified on a render.** As with the entry above, ANSI S3.5-1997 is not
> held locally, so the wording of the four captions is taken from the working
> group's errata list rather than from a page this project has read. The
> argument that the tables carry no threshold column is independent and does
> hold against the transcribed constants. Re-verify against a render of
> printed pp. 3-5 when the standard is acquired.

- **Location:** the captions of Tables 1, 2, 3 and 4 (pp. 3-5 of the 1997
  printing), the constant tables of the four band procedures: critical band
  (21 bands), equally-contributing critical band (17 bands), one-third octave
  (18 bands) and octave (6 bands).
- **The print (per the working group's errata):** each caption lists the
  quantities the table tabulates and includes the phrase "hearing threshold
  levels,".
- **The problem:** none of the four tables tabulates a hearing threshold
  level. Each carries the band centre frequency (and, for Tables 1, 2 and 4,
  the band limits), the band-importance function $I_i$, the standard speech
  spectrum level $U_i$ by vocal effort and the reference internal noise
  spectrum level $X_i$. The hearing threshold level $T'_i$ is a *user input*
  to the procedure (clause 5.5, where the equivalent internal noise spectrum
  level is $X'_i = X_i + T'_i$), which is exactly the quantity the caption
  invites the reader to look for in the table and to confuse with $X_i$.
- **Evidence:** the official errata list published by ASA Working Group S3-79,
  the committee that maintains ANSI S3.5, on its support site (sii.to):
  "Pages 3-5, Tables 1-4: In each of the **figure** captions the phrase
  'hearing threshold levels,' should be deleted" (archived as
  `plan/sii-asa-wg-s3-79/sii.to-errata-2026-07-30.txt`; an earlier revision of
  this entry dropped the word "figure" from the quotation); plus the tables
  themselves, which have no such column.
- **Library behaviour:** unaffected. The four tables are implemented with the
  columns they actually carry, exposed per procedure by `sii_procedure()` as
  `band_importance`, `speech_spectrum` ($U_i$) and `internal_noise` ($X_i$),
  and the
  hearing threshold stays the `threshold=` argument of
  `speech_intelligibility_index` ([`src/phonometry/speech/sii.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/speech/sii.py)).
- **Status:** published correction by the issuing working group; nothing to
  report upstream.

## NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (27)

- **Location:** section A.4.2, Eq. (27) (atmospheric absorption coefficient)
  and the sentence defining its symbols, printed p. 21.
- **The print:** Eq. (27) pairs the coefficient 6,6928·10⁻⁶ with `f_rO` and
  1,3415·10⁻⁶ with `f_rN`, and the sentence below reads "the variables
  f_rN = 75692 Hz and f_rO = 630.7 Hz represent the vibrational relaxation
  frequencies of oxygen and nitrogen respectively".
- **The problem:** the two subscripts are swapped in the definition sentence.
  The *values* match the *names* it gives them (75 692 Hz is the oxygen
  relaxation frequency and 630,7 Hz the nitrogen one at the reference
  conditions), but they are assigned to the opposite symbols, so the equation
  as printed multiplies the oxygen coefficient by the nitrogen relaxation
  frequency and vice versa. Evaluated that way it gives 14,2 dB/km at 500 Hz
  against the guidance's own Table 4 value of 3,1 dB/km; with $f_{rO}$ and
  $f_{rN}$ exchanged it gives 3,07 dB/km, reproducing Table 4 and the ISO 9613-1
  pure-tone coefficient to 0,02 dB/km. An earlier revision of this entry
  quoted the printed value as 14,3 dB/km and framed the defect as a wrong
  pairing of the coefficients rather than as swapped subscripts in the
  definition.
- **Evidence:** numeric evaluation of Eq. (27) with the printed assignment and
  with the assignment exchanged, against the Table 4 500 Hz cell on the same
  page. Render: `plan/aircraft-sources/norah2/NORAH2_D1.5d_rotorcraft_noise_modelling_guidance.pdf`, PDF page 20, printed p. 21, 250 dpi.
- **Library behaviour:** implements the correct pairing; the module docstring
  carries a defensive note so the misprint is not transcribed as a "fix".
- **Status:** unreported.

## NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), Eq. (21)

- **Location:** section A.3.3, Eq. (21) (flight path angle).
- **The print:** γ = acos(ΔZ/ΔS).
- **The problem:** the arccosine of the climb-to-path ratio returns the
  complement of the path angle (90° in level flight, where $\gamma$ must be
  0°) and contradicts the guidance's own use of $\gamma$ as the climb/descent
  angle throughout section A.3. ECAC Doc 32, 1st ed., Eq. (10) prints the
  correct form, $\gamma = \text{atan}(\Delta Z/\Delta S)$ with the horizontal
  $\Delta S$ of its Eq. (8).
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
  the Delaunay triangulation of the raw $(V, \gamma)$ conditions, not of the
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
- **The problem:** no term $\Delta L_{g,s'}$ exists; the prose directly below
  the equation defines `ΔL_d,s′` as "the attenuation due to the
  diffraction between the image source S′ and R", the receiver-side companion
  Eq. (47) prints the parallel term correctly as `ΔL_d,r′`, and the
  CNOSSOS-EU method the section is based on writes `Δ_ground(S,O)`
  with `Δ_dif(S′,R)` in that position. The
  subscript g is a misprint for d.
- **Evidence:** internal consistency of the section (its own prose and
  Eq. (47)) and the CNOSSOS-EU source of the equations.
- **Library behaviour:** implements the image-source diffraction term
  $\Delta L_{d,s'}$ as defined by the prose.
- **Status:** unreported.

## NORAH2 rotorcraft guidance SC01.D1.5d (EASA.2020.FC.06), §A.4.5 cross-references

- **Location:** section A.4.5, the definitions under Eq. (46) (printed p. 32)
  and Eq. (47) (printed p. 33).
- **The print:** four cross-references to eq. 44, in **three** different
  wordings: "calculated as per eq. 44" for ΔL_d,s′ and again for ΔL_d,s under
  Eq. (46); "calculated as in eq. 44" for ΔL_d,r′ under Eq. (47); and
  "calculated as in Subsection eq. 44" for ΔL_d,s under Eq. (47). An earlier
  revision of this entry quoted all four with the first wording.
- **The problem:** Eq. (44) is the multiple-diffraction coefficient $C''$; the
  attenuation due to diffraction is Eq. (42). All four cross-references point
  at the auxiliary coefficient instead of the formula they describe, and the
  fourth also carries a dangling "Subsection" with no subsection number after
  it.
- **Evidence:** the terms are attenuations in dB, which only Eq. (42)
  produces; Eq. (44) is a dimensionless coefficient consumed by Eq. (42).
  Renders: `plan/aircraft-sources/norah2/NORAH2_D1.5d_rotorcraft_noise_modelling_guidance.pdf`, PDF page 31, printed p. 32, 250 dpi, and PDF
  page 32, printed p. 33, 250 dpi.
- **Library behaviour:** evaluates the image-path and direct diffraction
  terms with Eq. (42), using Eq. (44) for $C''$ inside it.
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
- **The problem:** the first coefficient is the Zwicker-Terhardt
  $0{,}76 \cdot 10^{-3}$ with the exponent misprinted. The paper's own anchors
  disprove the print: it states 0,5 Bark = 50 Hz and
  23,5 Bark = 13,2 kHz (section 2.1.2) and
  15 Bark = 2,7 kHz (section 3.1), all of which require
  $10^{-3}$. With $10^{-4}$, $z(1\ \text{kHz}) = 1{,}05$ instead of
  8,51 Bark and the model's 47 filter centres
  would span 491 Hz to 20 kHz instead of 50 Hz to 13,2 kHz.
- **Evidence:** evaluation of Eq. (3) under both exponents against the
  paper's printed Bark/frequency anchors.
- **Library behaviour:** implements $0{,}76 \cdot 10^{-3}$ with a note at the
  formula;
  the carrier-frequency sweep test would catch a regression to the printed
  value ([`fluctuation_strength.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/fluctuation_strength.py)).
- **Status:** unreported (conference paper rather than a standard).

## Medwin & Clay, Fundamentals of Acoustical Oceanography (1998), Eq. (3.4.30) (boric-acid coefficient)

- **Location:** the Francois-Garrison boric-acid term as transcribed by the
  textbook, **Eq. (3.4.30), printed p. 110**. An earlier revision of this
  entry cited Eq. 3.4.29, which is the total-absorption sum of the three
  terms on printed p. 109; the boric-acid block is the equation after it.
- **The print:** `A1 = (8.68/c)·10^(0.78 pH − 5) dB km⁻¹ kHz⁻¹`.
- **The problem:** the original paper (Francois & Garrison 1982, JASA 72,
  Part II, Eq. (10) and Fig. 7) prints 8,86; the digits are transposed. Only
  8,86 reproduces the paper's own Table IV: with 8,68 the boric-dominated
  cells at 0,6 to 30 kHz sit up to 1,7 % below the printed totals (worst
  relative case 2 kHz, 10 °C, $S = 35$: 0,1209 vs the printed 0,123 dB/km).
- **Evidence:** recomputation of all sampled Table IV cells under both
  coefficients against the paper's printed values. Renders:
  `plan/underwater-sources/Fundamentals-of-acoustical-oceanography.pdf`, PDF
  page 131, printed p. 110, 300 dpi (the Eq. (3.4.30) block), and
  `plan/underwater-sources/Francois-Garrison-1982-P2-boric-acid-total-absorption-equation-JASA72.pdf`,
  PDF page 8, printed p. 1886, 300 dpi (the paper's own
  `A1 = (8.86/c)·10^(0.78 pH − 5)`).
- **Library behaviour:** implements the paper's 8,86 with a defensive note;
  the pinned Table IV set includes the boric-dominated rows.
- **Status:** unreported (textbook rather than a standard).

## Medwin & Clay (1998), Eq. (3.4.30) (sound speed printed as q)

- **Location:** the same Eq. (3.4.30) block, printed p. 110, its last line.
- **The print:** `q = 1412 + 3.21T + 1.19 S + 0.0167 z m/s.`
- **The problem:** the quantity the block needs is the sound speed `c`, which
  is what the two lines above it divide by (`A1 = 8.68/c`, and `A2 = 21.44 S/c`
  in the magnesium-sulfate block on the same page). No symbol `q` is defined
  anywhere in the section, so the transcribed system is not closed: a reader
  following the printed symbols has no value for `c`. Francois & Garrison
  1982 Part II prints the same polynomial as
  `c = 1412 + 3.21 T + 1.19 S + 0.0167 D`, introduced by "where c is the sound
  speed (m/s), given approximately by".
- **Evidence:** the block's own use of `c` two lines above, and the source
  paper. Renders: `plan/underwater-sources/Fundamentals-of-acoustical-oceanography.pdf`, PDF page 131, printed p. 110, 300 dpi, and
  `plan/underwater-sources/Francois-Garrison-1982-P2-boric-acid-total-absorption-equation-JASA72.pdf`,
  PDF page 8, printed p. 1886, 300 dpi.
- **Library behaviour:** unaffected; the absorption model takes the sound
  speed from the same polynomial under the name `c`.
- **Status:** unreported (textbook rather than a standard).

---

## Maa (1998), "Potential of microperforated panel absorber", JASA 104(5), Eq. (5b)

- **Location:** Eq. (5b), the mass-reactance coefficient of the microperforated
  panel, printed as k_m = 1 + [1 + k²/2]^(−1/2) + 0,85 d/t.
- **The print:** the first bracket term reads (1 + k²/2)^(−1/2).
- **The problem:** the same paper's Eq. (4), from which (5b) is factored,
  prints the term as (3² + k²/2)^(−1/2), and only that form reproduces the
  Crandall low-k limit $Z_1 \to (4/3) j\omega\rho_0 t$ of the paper's own
  Eq. (3a): at
  $k \to 0$ the printed (5b) gives an internal mass factor of 2 instead of 4/3.
  The paper's own Fig. 1 confirms it: with $0{,}85 \cdot d/t = 0{,}85$ the
  plotted $k_m$ starts near 2,2 ($= 4/3 + 0{,}85$) at $k = 0{,}1$, not at 2,85.
- **Evidence:** recomputation of both bracket variants against Eq. (4),
  Eq. (3a) and the Fig. 1 curve; the exact Bessel solution of Eq. (2) agrees
  with Eq. (4) within Maa's stated ~6 % only with the 3² form (the 1 form
  errs by >30 % at low k). Render: `plan/[Maa_1998_JASA] potential of
  microperforated panel absorber.pdf`, PDF page 2, printed p. 2862, 500 dpi,
  which carries Eq. (4) with `(3² + k²/2)^(−1/2)` and Eq. (5b) with
  `[1 + k²/2]^(−1/2)` fifteen lines apart on the same column.
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
  $\rho_\text{eff} \to \rho_0$ and $\kappa_\text{eff} \to \kappa_0$ (the
  printed 4 gives $16 \cdot \rho_0$), and at DC the square duct's
  $j\omega \cdot \rho_\text{eff}$ tends to the exact Shah-London Poiseuille
  flow resistivity: the series value $a^6/(64 \cdot S_0) = 28{,}4542$ matches
  $fRe/2 = 28{,}455$ (in units of $\eta/a^2$), where $S_0$ is the double
  transverse-mode sum at $G = 0$;
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
  mouth, but the printed $-i\omega$ prefactor is an opposite-time-convention
  ($e^{-i\omega t}$) expression inconsistent with the papers' otherwise
  $e^{+i\omega t}$ transfer-matrix chain (the $+i$ off-diagonal slit matrices
  of Appl. Sci. Eq. (2) and the $-i$ cotangent-type resonator impedance).
  Transcribed
  literally into that chain, the correction raises the slit-panel resonance
  where an added mass must lower it: for a 1 mm slit with a 30 mm lattice
  step and 50 mm period the absorption peak moves from 378,6 Hz to 386,8 Hz
  as printed, against 370,8 Hz with the mass sign. The neck end corrections
  of the same model behave correctly (they lower the resonator resonance).
- **Evidence:** numerical evaluation of both signs of the correction against
  the uncorrected panel; the direction of the neck end corrections of the
  same papers as the consistent control.
- **Library behaviour:** uses the added-mass sign ($+j\omega$ in the
  $e^{+j\omega t}$ convention of the library), conjugating the printed term
  exactly as it
  conjugates the papers' Stinson duct series; direction and peak are pinned
  by ``test_slit_radiation_correction_lowers_resonance`` in
  [`tests/materials/test_slow_sound_absorber.py`](https://github.com/jmrplens/phonometry/blob/main/tests/materials/test_slow_sound_absorber.py).
- **Status:** unreported (journal papers rather than standards).

## Attenborough & Van Renterghem, Predicting Outdoor Sound 2e (2021), Table 5.1

- **Location:** Table 5.1, "Coefficient and exponent values in the Delany and
  Bazley, Miki and modified Miki models", row "Miki [6,7]", coefficient $r$.
- **The print:** r = 0,0109.
- **The problem:** the original source (Miki 1990, J. Acoust. Soc. Jpn (E)
  11(1), Eq. (34)) prints
  $\beta(f) = (\omega/c_0)[1 + 0{,}109 \cdot (f/\sigma)^{-0{,}618}]$; the
  table drops a digit. With 0,0109 the real part of the Miki wavenumber at
  $f/\sigma = 0{,}01$ is 1,19 instead of 2,89, inconsistent with the same
  table's Delany-Bazley row (2,79 via $\rho_0 = 1{,}2$) and with the
  "modified Miki" row
  the book itself derives from it.
- **Evidence:** digit check against the original Miki (1990) paper (Eqs.
  (30)–(34)) and cross-computation of both variants at the fit-range edge.
- **Library behaviour:** implements Miki's original 0,109; the digitization
  point $f/\sigma = 0{,}1$ is pinned in ``tests/reference_data.py`` and in the
  conformance check "Miki 1990 Eqs. (30)-(34)".
- **Status:** unreported (textbook rather than a standard).

## Attenborough & Van Renterghem, Predicting Outdoor Sound 2e (2021), Eq. (5.13)

- **Location:** Eq. (5.13), the Johnson-Champoux-Allard bulk complex density,
  with $G(\Lambda) = \sqrt{1 - 4iT\eta\rho_0\omega/(R_S^2\Lambda^2\Omega^2)}$.
- **The print:** the tortuosity T appears to the first power inside G(Λ).
- **The problem:** Johnson et al. (1987) and the standard JCA formulation
  (Cox & D'Antonio 3e Eq. (6.19); Allard & Atalla) carry
  $T^2 = \alpha_\infty^2$ there. The
  first-power print breaks the high-frequency asymptote that defines the
  viscous characteristic length: with $T^2$ the density tends to
  $(T\rho_0/\Omega)(1 + (1 - j)\delta_v/\Lambda)$ with
  $\delta_v = \sqrt{2\eta/\rho_0\omega}$, while the printed form
  tends to a $\delta_v/(\Lambda\sqrt{T})$ correction, which for $T = 2$ means
  an error of 29 % in the boundary-layer term for the same $\Lambda$.
- **Evidence:** asymptotic expansion of both variants against the Johnson
  et al. definition of $\Lambda$ and against Cox & D'Antonio Eq. (6.19); the
  library's high-frequency JCA test pins the $T^2$ behaviour.
- **Library behaviour:** implements the standard $T^2$ form (Cox & D'Antonio
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
  prefactor $Z_{An}/Z_{A1}$ (equivalently $\sqrt{S_1/S_n}$ inside a $20\log_{10}$
  form) with $T_{11}$ unweighted and $Z_{A1}/Z_{An}$ on $T_{22}$. As printed,
  Eq. (8.141) fails the sudden-expansion limit: a zero-length element
  ($T = I$) between $S_1 = 0{,}01\ \text{m}^2$
  and $S_n = 0{,}02\ \text{m}^2$ is a sudden area expansion with the classic
  $TL = 10\log_{10}[(1+m)^2/(4m)] = 0{,}512\ \text{dB}$ ($m = S_n/S_1 = 2$), but the
  printed equation gives
  $\tfrac{1}{4} \cdot (Z_{A1}/Z_{An} + Z_{An}/Z_{A1})^2 = 1{,}938\ \text{dB}$.
  Reading the ratios
  as an overall $Z_{A1}/Z_{An}$ prefactor instead is also wrong: it gives
  6,532 dB
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
  oscillation condition Eq. (18.19), $Z_S + L_{H-M} = L_{H-L} - D_M(\theta)$,
  obtained by substituting the feedback-loop gain
  $G_S = L_{H-M} - L_{H-L} + D_M(\theta)$
  (Eq. (18.18)) into $Z_S + G_S = 0$ (Eq. (18.16)). Setting $N_m = 1$ makes
  $\Delta L_\text{nom} = 0$, so Eq. (18.24) must reduce to Eq. (18.20) and
  does not. The
  sign matters physically: $D_M(\theta)$ is "usually negative" in Long's own
  definition (about −2 to −3 dB for a cardioid pointed at the talker), so as
  printed a directional microphone would *cost* gain before feedback instead
  of buying it, inverting the chapter's own conclusion that "it is prudent to
  incorporate a cardioid or hypercardioid microphone into a system".
- **Evidence:** a 1200 dpi render of the printed equation reads
  `Z_S + L_{H−M} + ΔL_nom ≤ L_{H−L} + D_M(θ) − 10`, against
  `Z_S + L_{H−M} ≤ L_{H−L} − D_M(θ) − 10` two pages earlier, where the same
  position holds a minus. (An earlier revision of this entry quoted the
  `pdftotext` extraction, `Z S þ L HM þ DL nom  L HL þ D M ðqÞ  10`, in which
  `þ` is the ligature this PDF uses for "+" and every minus sign has been
  dropped entirely; that extraction cannot distinguish a plus from a minus and
  should never have been the evidence.) Renders: `plan/books/rooms/Long (2014)
  Architectural Acoustics 2e (Academic Press).pdf`, PDF page 697, printed
  p. 699, 1200 dpi (Eq. (18.24)), and PDF page 696, printed p. 698, 200 dpi
  (Eqs. (18.19) to (18.22)). The minus sign is the one that reproduces Long's
  own worked special cases
  at $N_m = 1$: with $Z_S = -6\ \text{dB}$, Eq. (18.21) gives
  $L_{H-M} \le L_{H-L} - D_M(\theta) - 4$ (an omnidirectional microphone 4 dB
  below the average audience level), and Eq. (18.22) gives
  $L_{H-M} \le L_{H-L} - 2$ for a
  cardioid at $D_M = -2\ \text{dB}$. Neither special case is recoverable from
  the
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
- **The print:** `A_tab > 6.33 r_s²`.
- **The problem:** the bound is Eq. (17.52),
  $L_\text{SN} = 10\log_{10}[Q/(4\pi r^2)] + 10\log_{10}[A_\text{tab}/4]$, solved for
  $A_\text{tab}$ at the stated threshold $L_\text{SN} > -6\ \text{dB}$, which
  gives $A_\text{tab} > 16\pi \cdot 10^{-0{,}6} r_s^2/Q$. With the
  $Q = 2$ the chapter uses for a talker, that constant is 6,3130, not 6,33.
  The gap is 0,27 %, i.e. the last printed digit: 6,33 is what
  $16\pi \cdot 10^{-0{,}6}/2$ returns if $10^{-0{,}6}$ is carried coarsely as
  0,252 instead
  of 0,251 19. This is graded as a rounding-level discrepancy rather than a
  structural error of the formula, since the formula itself is confirmed by
  its companion (below) and no consistent alternative assumption reproduces
  6,33 (it would require $Q = 1{,}995$).
- **Evidence:** the immediately following Eq. (17.54) is the same closed form
  at the privacy threshold $L_\text{SN} < -9\ \text{dB}$, and its printed
  constant 3,16 is exactly what $16\pi \cdot 10^{-0{,}9}/2 = 3{,}1640$ gives,
  confirming both the formula and
  $Q = 2$. Only the −6 dB constant is off. What does *not* discriminate is
  Long's prose one paragraph later, "at least 6.3 or more square meters
  (68 sq ft) of absorption per table": 6,313 m² is 67,95 ft² and 6,33 m² is
  68,14 ft², so both print as 68 sq ft, and both round to 6,3 m². An earlier
  revision of this entry offered that conversion as corroboration. Render:
  `plan/books/rooms/Long (2014) Architectural Acoustics 2e (Academic
  Press).pdf`, PDF page 665, printed p. 666, 250 dpi (Eqs. (17.52) to (17.54)
  and the paragraph that follows).
- **Library behaviour:** `absorption_per_table` in
  [`crowd_noise.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/crowd_noise.py) computes the bound
  from Eq. (17.52) rather than hardcoding either constant, so both bounds stay
  mutually consistent; the 6,313 value and the printed 3,16 are pinned by
  regression tests
  ([`tests/room/test_crowd_noise.py`](https://github.com/jmrplens/phonometry/blob/main/tests/room/test_crowd_noise.py)) and
  the 3,16 constant by the conformance check "Long, Architectural
  Acoustics 2e, Eq. (17.54)".
- **Status:** unreported (textbook rather than a standard, so non-normative);
  graded as a rounding discrepancy rather than a structural defect.

## Long, Architectural Acoustics 2e (2014), Table 14.7 (round elbow rows)

- **Location:** Chapter 14, Table 14.7, "Insertion Loss of Round Elbows"
  (printed p. 541), indexed by the frequency-width product $f w$ (kHz times
  inches).
- **The print:** four rows only: f w < 1,9 → 0 dB; 1,9 < f w < 3,8 → 1 dB;
  3,8 < f w < 7,5 → 2 dB; f w > 15 → 3 dB.
- **The problem:** the band $7{,}5 < f w < 15$ has no row at all, so the table
  jumps from $3{,}8 < f w < 7{,}5$ straight to $f w > 15$. A duct-borne
  calculation lands in that band routinely: a 24 in elbow at 500 Hz has
  $f w = 12$.
- **Evidence:** the same data adapted from the same ASHRAE source appear in
  Bies, Hansen & Howard, *Engineering Noise Control* 5e, Table 8.11, indexed
  by $W/\lambda$ ($= 0{,}074\,f w$). Its round-elbow column has six rows,
  0/1/2/3/3/3, and gives 3 dB for $0{,}55 \le W/\lambda < 1{,}11$, which is
  exactly the $7{,}5 < f w < 15$ band
  Long omits. Long's four rows map onto Bies' six as follows: the first three
  agree entry for entry, the fourth ($f w > 15$, 3 dB) legitimately merges
  Bies' two identical top rows, and the band with no row is Bies' fourth. An
  earlier revision of this entry said that "Tables 14.5 and 14.6 both carry
  six rows" and that "the other five rows of the two tables agree entry for
  entry"; on the page, Table 14.5 carries six rows and Table 14.6 five (it
  merges the same two identical top bands, legitimately), and Table 14.7
  prints four, so neither count is right. Render: `plan/books/rooms/Long
  (2014) Architectural Acoustics 2e (Academic Press).pdf`, PDF page 542,
  printed p. 541, 200 dpi (Table 14.7), and PDF page 541, printed p. 540,
  200 dpi (Tables 14.5 and 14.6).
- **Library behaviour:** `elbow_insertion_loss` in
  [`hvac.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) carries the six-row
  round column with 3 dB in the missing band, pinned by
  `test_elbow_tables_by_frequency_width_product`
  ([`tests/noise_control/test_hvac_long.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_hvac_long.py)).
- **Status:** unreported (textbook rather than a standard).

## Long, Architectural Acoustics 2e (2014), Eq. 13.28 (units of U_G)

- **Location:** Chapter 13, Eq. 13.28 (printed p. 521), the normalised
  pressure-drop coefficient $\xi = 334{,}9 \cdot \Delta P/(\rho_0 U_G^2)$ of
  the diffuser sound-power model.
- **The print:** the nomenclature under the equation gives "U_G = flow
  velocity prior to the diffuser (ft/min)" and, on the next line,
  "= Q/(60·S_G) (for Q in cfm)".
- **The problem:** the two statements contradict each other. $Q$ in ft³/min
  divided by $60 S_G$ is a velocity in **ft/s**, not ft/min, and only the ft/s
  reading makes the constant right: $334{,}9/\rho_0$ with
  $\rho_0 = 0{,}075\ \text{lb/ft}^3$ is
  $4465 \cdot \Delta P/U^2$, which is the standard velocity-pressure relation
  $\Delta P/(U/4005)^2$ only when $U$ is converted from ft/s. Read as ft/min
  the coefficient comes out 3600 times too small. Eq. 13.27 itself declares
  $U_G$ in ft/s, so the "(ft/min)" label under Eq. 13.28 is the odd one out.
- **Evidence:** dimensional check of $Q/(60 S_G)$; reconstruction of the
  $334{,}9/\rho_0$ constant from the velocity-pressure relation; and the peak
  frequency. What does **not** discriminate is the overall level: Eq. 13.27
  carries $30\log_{10}\xi + 60\log_{10} U_G$, and substituting Eq. 13.28 makes the
  velocity cancel identically,
  $30\log_{10}\xi + 60\log_{10} U_G = 30\log_{10}(334{,}9 \Delta P/\rho_0)$. For the
  Table 14.9 supply diffuser ($S_G = 4\ \text{ft}^2$, $Q = 312$ cfm,
  $\Delta P = 0{,}05$ in w.g.)
  both readings therefore return the same $L_W = 45{,}18\ \text{dB}$. An
  earlier revision
  of this entry claimed that the ft/min reading "misses it by 100 dB", which
  is arithmetically impossible for a quantity that does not depend on the
  velocity at all. What does discriminate is Eq. 13.32, $f_P = 48{,}8 U_G$,
  which is the only other place $U_G$ enters: read in ft/s the approach
  velocity
  is 1,3 ft/s and the peak falls at 63,4 Hz, i.e. in the 63 Hz octave, so the
  Eq. 13.31 shape puts 33,4 dB in that band against the printed 33; read in
  ft/min it is 78 ft/min, the peak moves to 3 806 Hz, and the same shape puts
  −8,2 dB in the 63 Hz band. Render: `plan/books/rooms/Long (2014)
  Architectural Acoustics 2e (Academic Press).pdf`, PDF page 522, printed
  p. 521, 250 dpi (Eq. 13.28 with its "(ft/min)" nomenclature, Eq. 13.27's
  "(ft/s)" nomenclature at the top of the same page, and Eqs. 13.29 to 13.33).
- **Library behaviour:** `diffuser_sound_power` in
  [`hvac.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) reads $U_G$ in ft/s
  internally (SI at the interface), with the Table 14.9 row pinned by
  `test_diffuser_sound_power_reproduces_the_table_14_9_row`
  ([`tests/noise_control/test_hvac_long.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_hvac_long.py))
  and the conformance check "Long 2e Eqs. 13.27-13.33".
- **Status:** unreported (textbook rather than a standard).

## Vigran, Building Acoustics (2008), Figure 8.37 caption (carpet stiffness exponent)

- **Non-normative source** (textbook).
- **Location:** section 8.4.2, the caption of Figure 8.37 on printed p. 320 /
  pdf p. 341, which labels the predicted improvement curves of two floor
  coverings laid on a heavyweight floor.
- **The print:** "Predicted improvement with a linear model: stiffness of
  carpet squares 3.2·10^6 N/m, vinyl covering 5.2·10^6 N/m." (Vigran writes
  the decimal separator as a period.)
- **The problem:** the carpet exponent is one order too high. The body text
  introducing the figure, on printed p. 321, says of the carpet squares that
  "we have assumed that the covering has the same stiffness as used in
  Figure 8.36", and Figure 8.36 is labelled `s = 3.2·10^5 N/m` inside the
  plot, the same value the body text on printed p. 320 gives for it. The
  vinyl value in the same caption is correct.
- **Evidence:** printed p. 320 states `3.2·10^5 N/m` "giving a resonance
  frequency f0 of approximately 130 Hz with a hammer mass of 0.5 kg", and
  `sqrt(3.2e5/0.5)/(2 pi) = 127,3 Hz` reproduces that while
  `sqrt(3.2e6/0.5)/(2 pi) = 402,6 Hz` is a frequency that appears nowhere in
  the section. The same arithmetic applied to the caption's vinyl value gives
  `sqrt(5.2e6/0.5)/(2 pi) = 513,3 Hz` against the "approximately 510 Hz"
  printed on p. 321, which fixes the formula and the hammer mass the author
  used. Graphically, the two dashed prediction curves of Fig. 8.37 are about
  two octaves apart, matching the stiffness ratio 5.2e6/3.2e5 = 16,25 (a
  factor 4,03 in frequency) and not 5.2e6/3.2e6 = 1,63 (a factor 1,27).
  Verified on 600 dpi renders of pdf pp. 340, 341 and 342.
- **Library behaviour:** none needed; the library takes the covering stiffness
  from the user through `covering_contact_stiffness`, and the printed cut-off
  frequencies it is anchored on come from Hopkins rather than from this
  caption.
- **Status:** unreported.

## Norton & Karczub, Fundamentals of Noise and Vibration Analysis for Engineers 2e (2003), Eq. (6.56)

- **Location:** Section 6.6.1, Eq. (6.56), the coupling loss factor of two
  homogeneous plates joined by $N$ point connections (printed p. 418).
- **The print:** the denominator bracket
  ``(rho_s1^2 h1^2 cL1^2 + rho_s2^2 h2^2 cL2^2)`` appears to the first power.
- **The problem:** as printed the expression is not dimensionless. The
  prefactor $4 N h_1 c_{L1}/(\sqrt{3}\,\omega S_1)$ already has the dimensions
  of $\text{m}^2\,\text{s}^{-1}$ over $\text{m}^2\,\text{s}^{-1}$, i.e. unity,
  so the remaining ratio of the two
  bracketed products must be dimensionless too. That requires the sum to be
  squared, $A_1 A_2/(A_1 + A_2)^2$.
- **Evidence:** the book's own answer to problem 6.13 (printed p. 617). With
  the squared denominator the twelve-bolt aluminium pair gives
  $\eta_{12} = 1{,}43 \cdot 10^{-2}$ at 125 Hz against the printed
  1,44·10⁻², and matches the
  whole 125 Hz to 2 kHz column to better than 0,7 %; with the printed
  (unsquared) denominator the result is not a loss factor at all.
- **Library behaviour:** `point_connection_coupling_loss_factor` in
  [`junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/vibration/structural/junction_transmission.py)
  implements the squared form, with the printed column pinned by a regression
  test ([`tests/vibration/structural/test_junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/structural/test_junction_transmission.py))
  and a note at the formula.
- **Status:** unreported (textbook rather than a standard).

## Norton & Karczub 2e (2003), problem 6.13 answer (eta_21 column)

- **Location:** Answers to problems, problem 6.13 (printed p. 617), the two
  $\eta_{21}$ columns of the welded and bolted tables.
- **The print:** for the two aluminium plates (plate 1: 3 mm, 2,5 m × 1,2 m;
  plate 2: 5,5 mm, 2,0 m × 1,2 m) the answer gives, at 125 Hz,
  eta_21 = 5,77·10⁻³ (welded) and 2,64·10⁻² (bolted).
- **The problem:** both columns are exactly the corresponding $\eta_{12}$
  column multiplied by $h_2/h_1 = 1{,}833$. The SEA consistency relationship
  is $n_1 \eta_{12} = n_2 \eta_{21}$ (Eq. 6.8) with the flat-plate modal
  density $n = S\sqrt{12}/(2 c_L h)$ of Eq. (6.25), so the correct factor is
  $n_1/n_2 = (S_1 h_2)/(S_2 h_1) = 2{,}292$. The printed column drops the
  plate area ratio $S_1/S_2 = 1{,}25$.
- **Evidence:** the ratio of the printed columns is 1,8333 to five digits in
  every band of both tables, which is $h_2/h_1$ exactly; the $\eta_{12}$
  columns themselves reproduce from Eqs. (6.52) to (6.56) to better than
  0,7 %.
- **Library behaviour:** the $\eta_{12}$ columns are used as the regression
  oracle; $\eta_{21}$ is obtained from Eq. (6.8) with the full modal
  densities,
  and a test pins the 2,292 ratio explicitly
  ([`tests/vibration/structural/test_junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/structural/test_junction_transmission.py)).
- **Status:** unreported (textbook rather than a standard).

## Norton & Karczub 2e (2003), problem 6.10 (platform area)

- **Location:** Problems, problem 6.10 (printed pp. 593-594) and its answer
  (printed p. 617): a satellite platform coupled to an aluminium cylinder,
  500 Hz octave, printed answers $\eta_{12} = 4{,}26 \cdot 10^{-4}$,
  $\eta_{21} = 3{,}92 \cdot 10^{-4}$ and $\Pi_\text{in} = 1{,}31\ \text{W}$.
- **The print:** the statement gives the aluminium platform as
  "5 mm thick and 3,5 m × 3 m", i.e. 10,5 m².
- **The problem:** that area is inconsistent with the three printed answers.
  Eq. (6.12) fixes $E_1/E_2 = (\eta_2 + \eta_{21})/\eta_{12} = 6{,}554$ from the
  printed loss factors alone, whereas the stated geometry with the printed
  velocities (27,2 and 13,2 mm/s) gives 7,88. The energy ratio is independent
  of the modal densities and of the wave speed, so no choice of those can
  reconcile it; only the platform area can. The area the answers imply is
  8,73 m², which is $3{,}5 \times 3$ minus the $\pi(0{,}75\ \text{m})^2$
  footprint of the cylinder that Fig. P6.10 shows standing on the platform.
- **Evidence:** with 8,73 m² the inversion of Eqs. (6.15), (6.8) and (6.10)
  returns $\eta_{12} = 4{,}256 \cdot 10^{-4}$,
  $\eta_{21} = 3{,}910 \cdot 10^{-4}$ and $\Pi_\text{in} = 1{,}306\ \text{W}$,
  i.e.
  all three printed answers within 0,4 %; the cylinder's own energy and modal
  density come out unchanged either way.
- **Library behaviour:** `power_injection_clf` in
  [`experimental_sea.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/vibration/structural/experimental_sea.py)
  implements the inversion as published; the regression test uses the free
  platform area and documents the discrepancy
  ([`tests/vibration/structural/test_experimental_sea.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/structural/test_experimental_sea.py)).
- **Status:** unreported (textbook rather than a standard).

## Norton & Karczub 2e (2003), problem 3.14 (structural loss factor)

- **Location:** Problems, problem 3.14 (printed p. 580) and its answer
  (printed p. 611): the octave-band transmission loss of a 20 mm particle
  board panel.
- **The print:** the statement gives the panel a structural loss factor of
  "~1,5 × 10⁻²"; the answer gives 27 dB at 8 kHz and 38,6 dB at 16 kHz.
- **The problem:** those two values are above the panel's critical frequency
  (4885 Hz for Appendix 4 particle board, $f_c t = 97{,}7\ \text{m/s}$) and
  therefore follow Cremer's Eq. (3.110), which contains $10\log_{10}(\eta)$. With
  $\eta = 1{,}5 \cdot 10^{-2}$ the equation gives 37,0 dB and 48,5 dB, ten
  decibels above the printed answers; with $\eta = 1{,}5 \cdot 10^{-3}$ it
  gives 27,0 dB and 38,5 dB.
- **Evidence:** the 10 dB offset is exactly one decade of $10\log_{10}(\eta)$, and
  the frequency dependence of the printed pair independently fixes
  $f_c = 4939\ \text{Hz}$ against the Appendix 4 value of 4885 Hz. The eight
  values below
  coincidence reproduce exactly from Eq. (3.104) and do not involve eta. The
  discrepancy is a decade in a printed exponent, so the two figures were read
  as images rather than through the text layer. Renders:
  `plan/books/vibration/Norton & Karczub (2003) Fundamentals of Noise and
  Vibration Analysis for Engineers 2e (CUP).pdf`, PDF page 600, printed
  p. 580, 300 dpi (the problem statement, "a structural loss factor of
  ~1.5 × 10⁻²"), and PDF page 631, printed p. 611, 250 dpi (the answer table,
  27 dB at 8 kHz and 38,6 dB at 16 kHz in both columns).
- **Library behaviour:** the regression test uses $\eta = 1{,}5 \cdot 10^{-3}$,
  the value
  the printed answers require
  ([`tests/building/prediction/test_panel_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/prediction/test_panel_transmission.py)).
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
  $\int_0^{L_R} \exp(-\varepsilon m'_R x)\,dx = (1 - \exp(-\varepsilon m'_R L_R))/(\varepsilon m'_R)$,
  so the
  factor that normalises it must be $m'_R L_R$, exactly as the source-side one
  is $m_S L_S$. Read literally, the printed expression is not a transmission
  factor at all: it carries a spurious
  $m'_R/m_R = 1 + s_R \tau_R/(h m_R)$, so
  it grows without bound as the plenum damping falls. Two consequences are
  visible with ordinary inputs ($L_S = L_R = 5\ \text{m}$,
  $h = 0{,}6\ \text{m}$, $R_S = R_R = 25\ \text{dB}$, $\varepsilon = 2$,
  $s_S = s_R = 0{,}5$): the model becomes
  **non-monotonic in the damping**, giving $R_\text{cl} = 40{,}26\ \text{dB}$
  at $m_R = 0{,}01\ \text{1/m}$ but only $26{,}48\ \text{dB}$ at $10^{-4}$,
  i.e. adding plenum
  absorber is predicted to make the flanking path worse than leaving it bare;
  and it **breaks energy conservation**, returning $\tau_\text{cl} = 4{,}45$ at
  $R_S = R_R = 6\ \text{dB}$, $m_R = 0{,}01$ and $\tau_\text{cl} = 829$ at
  $R_S = R_R = 0\ \text{dB}$, $m_R = 10^{-3}$.
- **Evidence:** with $m'_R$ in the denominator every one of those pathologies
  disappears: $\tau_\text{cl}$ is monotonically decreasing in the damping, is
  bounded above by 1 because
  $(1 - \exp(-\varepsilon m'_R L_R))/(\varepsilon m'_R L_R) \le 1$, and reduces
  to Vigran's own small-attenuation result, Eq. (9.19)
  $\tau_\text{cl} = \varepsilon^2 \tau_S \tau_R L_R/(4h)$, whenever
  $m_S L_S$ and $m'_R L_R$ are
  both small. With the printed $m_R$ the same limit picks up the factor
  $m'_R/m_R$, which diverges, so Eq. (9.18) as printed does not reduce to
  Eq. (9.19) at all: the two equations the book presents as a pair are
  inconsistent with each other.
- **Library behaviour:** `plenum_flanking_reduction_index` in
  [`ceiling_plenum.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/building/prediction/ceiling_plenum.py)
  implements the derived $m'_R$ in both the exponent and the denominator, with
  the reading documented at the formula, and rejects a transmission factor
  above unity rather than reporting a negative sound reduction index. Tests pin
  the monotonicity, the $\tau_\text{cl} \le 1$ bound, the convergence to
  Eq. (9.20)
  and the size of the Eq. (9.17) leakage term at a realistic ceiling
  ([`tests/building/prediction/test_ceiling_plenum.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/prediction/test_ceiling_plenum.py)).
- **Status:** unreported (textbook rather than a standard). Mechel's original
  1980 paper, which Vigran reproduces, was not available to check whether the
  misprint originates there.

## Real Decreto 1367/2007, Annex IV A.3.3 (Kf and Ki threshold tables)

- **Location:** Annex IV, section A.3.3, the $K_f$ (low-frequency) and
  $K_i$ (impulsive) correction tables, middle row of each.
- **The print:** both tables print the 3 dB row as "Si 10 > Lf <= 15" and
  "Si 10 > Li <= 15" respectively (BOE-A-2007-18397, consolidated text).
- **The problem:** the condition as printed is unsatisfiable. It reads
  "10 greater than Lf" and "Lf at most 15" simultaneously, which would
  select levels below 10 dB, but the row above it already assigns those to
  0 dB ("Si Lf <= 10") and the row below covers "Si Lf > 15". The three
  rows only partition the range under the reading $10 < L_f \le 15$, so the
  ">" is a typeset inversion of "<".
- **Evidence:** the bracketing rows leave no other consistent reading; the
  identical construction appears in both tables, and the equivalent tables
  in the autonomous-community noise regulations that transpose this Annex
  print `10 < Lf <= 15`.
- **Library behaviour:**
  [`low_frequency_correction`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/assessment/spain.py)
  and `impulsive_correction` implement $10 < L \le 15$, with a regression
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
  125 Hz - 4 kHz would silently discard the 63 Hz and 8 kHz bands, which
  Appendix F tabulates like every other.
- **Evidence:** corrected by the corrigendum published in OJ L 5, 10.1.2018,
  p. 35, which reads in full: 'On page 8, in the Annex, in point 2.2.1, in the
  second paragraph under the heading "Traffic flow": for: "each octave band i
  from 125 Hz to 4 kHz", read: "each octave band i from 63 Hz to 8 kHz"'. The
  same corrigendum also adds "octave bands" to the frequency range of 2.1.1.
- **Library behaviour:**
  [`cnossos_road`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/sources/cnossos_road.py) works over
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
  derivation, "estimated by equating θ_n and θ_eff with n = 3/2". The two
  angles are four printed pages apart, not on the same page as an earlier
  revision of this entry stated:
  - Eq. (9.47), $\theta_\text{eff} = (\pi H/(4\eta r))^{1/2}$, **printed
    p. 453**, with the **true water depth $H$** (it comes from the multipath
    integral Eq. (9.46), whose $1/(rH)$ prefactor is the cylinder area
    $A_\text{CS} = 2\pi r H$ of Eq. (9.44), so $H$
    is the depth that counts bottom bounces);
  - Eq. (9.56), $\theta_n \approx n\pi/(k H_e)$, **printed p. 457**, with the
    **effective depth $H_e$** (mode angles are set by the apparent
    pressure-release boundary).

  Equating them at $n = 3/2$ gives $\pi H/(4\eta r) = 9\pi^2/(4k^2H_e^2)$,
  that is **$r_\text{MS} = k^2H_e^2H/(9\pi\eta)$**. The printed form is larger
  by $\pi H_e/H$. The
  factor $\pi$ is unconditional: it survives even if $H_e$ is substituted for
  $H$ in Eq. (9.47), which is presumably how the printed $H_e^3$ arose, and
  that reading would give $k^2H_e^3/(9\pi\eta)$, still $\pi$ below the print.
  The residual $H_e/H$ is the
  depth substitution itself, and it tends to 1 at high frequency. The other
  transition of the same section, Eq. (9.50)
  $r_\text{CS} = \pi H/(4\eta\psi_c^2)$, follows its
  own derivation exactly (it is where Eq. (9.42) and Eq. (9.49) cross), so the
  defect is confined to Eq. (9.57).
- **Evidence:** the symbolic re-derivation above, checked numerically for
  $H = 50\ \text{m}$, $f = 250\ \text{Hz}$, $c_w = 1500\ \text{m/s}$ over the
  Table 9.1 sand seabed ($\eta = 0{,}28\ \text{Np/rad}$,
  $\psi_c = 33{,}56^\circ$, $H_e = 53{,}63\ \text{m}$,
  $k = 1{,}047\ \text{m}^{-1}$):

  | | $r_\text{MS}$ | $\theta_\text{eff}$ there (Eq. 9.47) |
  |---|---|---|
  | derivation, $k^2H_e^2H/(9\pi\eta)$ | 19,9 km | 4,808° |
  | printed Eq. (9.57), $k^2H_e^3/(9\eta)$ | 67,1 km | 2,619° |

  The ratio $67{,}1/19{,}9$ is $\pi H_e/H = 3{,}3695$ to every digit carried.
  The angle
  column is an independent check that does not depend on how the derivation is
  read: the first two mode angles of Eq. (9.56) are $\theta_1 = 3{,}205^\circ$
  and $\theta_2 = 6{,}410^\circ$, so $\theta_{3/2} = 4{,}808^\circ$. At the
  derived range the effective angle is
  exactly $\theta_{3/2}$, halfway between the first two modes, which is what
  the text
  asks for. At the printed range it has fallen to 2,619°, **below $\theta_1$
  itself**:
  the second mode would have been stripped long before, so that range cannot
  be where the single-mode regime begins. Both printed formulae are confirmed
  on renders of `plan/books/underwater/Ainslie (2010) Principles of Sonar
  Performance Modelling (Springer).pdf`: PDF page 483, printed p. 453, 250 dpi
  (Eqs. (9.42) to (9.47)), and PDF page 487, printed p. 457, 250 dpi
  (Eqs. (9.54) to (9.57)).
- **Library behaviour:** `weston_regime_boundaries` in
  [`weston_regimes.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/underwater/weston_regimes.py)
  implements the derivation-consistent $k^2H_e^2H/(9\pi\eta)$, which is also
  what keeps $\theta_\text{eff}$ defined with $H$ everywhere the module
  evaluates Eq. (9.47). The
  equating rule is pinned by
  `test_mode_stripping_boundary_equates_theta_eff_with_mode_3_over_2`, which
  rebuilds both angles from the printed equations rather than from the
  implementation, and the shared definition of $\theta_\text{eff}$ by
  `test_composite_loss_and_the_boundary_use_the_same_effective_angle` (both in
  [`tests/underwater/test_weston_regimes.py`](https://github.com/jmrplens/phonometry/blob/main/tests/underwater/test_weston_regimes.py)).
- **Status:** unreported (textbook rather than a standard).

---

## NMFS (2024) Updated Technical Guidance v3.0, Table 5 / Table ES2 (otariid C)

*Regulatory guidance document, not a standard.*

- **Location:** Table 5 (printed p. 25), repeated as Table ES2 (printed p. 3):
  the auditory weighting parameter $C$ of the otariid pinniped in-water group
  (OW / OCW).
- **The print:** C = 1,37 dB.
- **The problem:** the correct value is 1,36 dB. NMFS states so itself in the
  table's own footnote: "During the public comment period, an error was
  identified with the Navy's rounding, where this value should be 1.36,
  instead of 1.37. Because this is such a minor error and to remain consistent
  with the Navy, NMFS decided rely upon the value the Navy originally
  provided." The document therefore knowingly publishes the wrong digit.
- **Evidence:** independent recomputation of $C$ from its own definition, the
  negated peak of $W(f)$, with the same row's parameters $a = 1{,}58$,
  $b = 5$, $f_1 = 2{,}53\ \text{kHz}$, $f_2 = 43{,}8\ \text{kHz}$:
  $C = 1{,}3643\ \text{dB}$, which rounds to 1,36. The
  published weighted TTS onset of the same row
  ($179\ \text{dB} = K + C$ with $K = 178$) is unaffected by the third digit.
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
  printed pair as well ($144 - 138 = 6$, just as $161 - 155 = 6$), so it says
  nothing about which pair is right.
  - **Mechanism.** In each printed row the peak-SPL TTS entry is numerically
    identical to that same row's PTS-onset **SEL** entry: PCA 123 / 138 / 138 /
    144 and OCA 146 / 161 / 161 / 167, reading TTS SEL, TTS peak, PTS SEL, PTS
    peak. A value duplicated from the neighbouring column, with the +6 dB rule
    then applied to the duplicate, is the signature of a column slip, and it is
    exactly the two entries per row the errata replaces.
  - **Value.** The corrected numbers are close to what the article's
    extrapolation rule produces, with the caveat that the rule is not stated
    for these rows. Printed p. 155 sets the impulsive peak-SPL TTS onset of a
    group without direct data at the hearing threshold at the frequency of best
    sensitivity $f_0$ plus 159 dB, and restricts that rule explicitly to the
    in-water groups: "For other species groups **in water** (LF, SI, PCW, and
    OCW), 159 dB was added to the value of the hearing threshold at f₀".
    It works the rule through for PCW: "Peak SPL TTS onset was estimated as
    212 dB re 1 µPa (53 dB at f₀ + 159 dB)". Evaluating the Table 2 group
    audiogram at the Table 4 $f_0$ reproduces the three in-water rows the errata
    does not touch (SI 219,6 against a published 220; PCW 212,5 against 212;
    OCW 226,1 against 226), which validates the rule where the article applies
    it. Extending it to the two in-air carnivore rows, which the article does
    not do, gives PCA −4,6 dB re 20 µPa at 2,3 kHz and OCA 11,4 dB re 20 µPa
    at 10 kHz, hence **154,4** and **170,4**. Those reproduce the corrected
    155 and 170 to within 0,6 dB and are 16 dB and 9 dB away from the printed
    138 and 161, which is what makes them corroborating rather than
    confirming; note that 154,4 rounds to 154, not to 155, and an earlier
    revision of this entry claimed that it rounded to the corrected value.
  - **A second, unrepaired inconsistency.** Printed p. 155 states that for the
    in-air carnivores specifically "a nominal 15 dB offset is used ... between
    the SEL-based TTS threshold and the peak SPL-based threshold", which
    reproduces the *printed* 138 and 161 from the SEL column. That sentence,
    not the +159 dB rule, is the one the article's own method applies to PCA
    and OCA. The errata resolves the conflict in favour of values consistent
    with the +159 dB rule, so it supersedes the sentence as well as the table;
    the sentence is left standing in the article.
  Renders, all of `plan/underwater-sources/Southall-2019-with-errata.pdf`,
  which bundles the article and the errata: PDF page 31, printed p. 155,
  300 dpi (the "in water (LF, SI, PCW, and OCW)" restriction and the 15 dB
  in-air offset in the same paragraph); PDF page 32, printed p. 156, 200 dpi
  (the article's Table 7, with the PCA row 123 / 138 / 138 / 144 and the OCA
  row 146 / 161 / 161 / 167 that show the duplication); and PDF page 110,
  printed p. 570, 250 dpi (the errata's reprinted Table 7, with PCA
  123 / 155 / 138 / 161 and OCA 146 / 170 / 161 / 176).
- **Library behaviour:** the errata-corrected values are the ones implemented
  in [`marine_mammal_weighting.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/underwater/marine_mammal_weighting.py),
  pinned by `test_southall_table_7_errata_values_are_implemented`, with the
  +159 dB rule itself checked against the audiogram in
  `test_southall_impulsive_peak_spl_is_threshold_at_f0_plus_159_db` for the
  in-water groups the article restricts it to and, separately and with the
  extrapolation labelled as such, for PCA and OCA.
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
  Table [2.3.b]. The table is the **rail** roughness $L_{r,TR,i}$, the
  quantity the table's own title announces and which is otherwise missing from
  Appendix G.
- **Evidence:** read off a rendered page image of the Official Journal PDF. The
  corrigendum of OJ L 5, 10.1.2018 re-titles it $L_{r,TR,i}$, and Commission
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
- **The problem:** the two open intervals leave $R = 300\ \text{m}$ and
  $R = 500\ \text{m}$ with no
  excess at all, and $R = 500\ \text{m}$ is explicitly inside the scope the same
  paragraph has just set. A 500 m curve therefore falls out of a rule written
  to include it.
- **Evidence:** read off a rendered page image of the Official Journal PDF.
  Commission Delegated Directive (EU) 2021/1226 Annex point (4)(b) replaces the
  paragraph with a table whose intervals are closed, "R <= 300 m" and
  "300 m < R <= 500 m".
- **Library behaviour:** `curve_squeal_excess` implements the 2021 table, so
  $R = 300\ \text{m}$ returns 8 dB and $R = 500\ \text{m}$ returns 5 dB; the
  boundaries are pinned
  in `test_curve_squeal_rule_of_2021`.
- **Status:** unreported (corrected by the issuing body in 2021).

---

## Allard & Atalla, Propagation of Sound in Porous Media 2e (2009), Eq. (6.85)

*Textbook, not a standard.*

- **Location:** Sect. 6.5.2 (printed p. 123), the second form of the shear-wave
  velocity ratio $\mu_3$.
- **The print:** `mu3 = (N delta3^2 - w^2 rho11) / (w^2 rho22)`, offered as an
  alternative to Eq. (6.84), `mu3 = -rho12 / rho22`.
- **The problem:** the two printed forms are not equal. Substituting the shear
  wavenumber of Eq. (6.83),
  $\delta_3^2 = (\omega^2/N)(\rho_{11}\rho_{22} - \rho_{12}^2)/\rho_{22}$,
  into the printed Eq. (6.85) gives $-\rho_{12}^2/\rho_{22}^2$, which is
  Eq. (6.84)
  multiplied by the spurious factor $\rho_{12}/\rho_{22}$. The denominator
  should read $\omega^2\rho_{12}$.
- **Evidence:** the book's own derivation. Eq. (6.80), printed p. 122, is
  $-\omega^2\rho_{11}\psi_s - \omega^2\rho_{12}\psi_f = N\nabla^2\psi_s = -N\delta_3^2\psi_s$,
  so $(N\delta_3^2 - \omega^2\rho_{11})\psi_s = \omega^2\rho_{12}\psi_f$ and
  therefore
  $\mu_3 = \psi_f/\psi_s = (N\delta_3^2 - \omega^2\rho_{11})/(\omega^2\rho_{12})$.
  With that reading
  the two forms agree identically wherever $\rho_{12}$ is non-zero; at
  $\rho_{12} = 0$
  the corrected quotient is $0/0$ while Eq. (6.84) stays defined and gives
  $\mu_3 = -\rho_{12}/\rho_{22} = 0$, which is the value to use there. The
  printed form instead
  differs from Eq. (6.84) by the factor $\rho_{12}/\rho_{22}$, so it coincides
  with it
  only where that ratio is exactly 0 or exactly 1. With
  $\rho_{12}/\rho_{22} = \rho_0/(\phi\rho_\text{eq}) - 1$ those two cases ask
  for $\rho_\text{eq} = \rho_0/\phi$ and $\rho_\text{eq} = \rho_0/(2\phi)$,
  both real; the effective
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
  $\sigma_{33}^f$ of a poroelastic layer and the matrix $[\Gamma]$ it feeds.
- **The print:** Eq. (11.48) reads

  ```text
  sigma33^f = sum (Q + R mu_i)(kt^2 + ki3^2)
              { -(A_i - A'_i) cos(ki3 x3) + j (A_i - A'_i) sin(k33 x3) }
  ```

  and Table 11.1 writes `k_{i3}` in the two columns that carry `mu1`, `D1`
  and `E1`.
- **The problem:** two independent misprints in the same equation, plus a
  subscript slip in the table.
  - The coefficient of the *symmetric* amplitude $(A_i + A'_i)$ is missing:
    Eq. (11.48) attaches both terms to $(A_i - A'_i)$, which would leave the
    first and third columns of $[\Gamma]$ with no $\sigma_{33}^f$ entry at
    all, contradicting Table 11.1, whose row 6 prints `-E1 cos(k13 x3)` and
    `-E2 cos(k23 x3)` in exactly those columns. The first term is
    $-(A_i + A'_i)\cos(k_{i3} x_3)$.
  - The sine carries `k33`, the *shear* wave-number component, inside a sum
    over the two compressional waves $i = 1, 2$. It must be `ki3`. Table 11.1
    again gives the intended reading: its row 6 has `j E1 sin(k13 x3)` and
    `j E2 sin(k23 x3)`, and zero in both shear columns, because a shear wave
    produces no dilatation and therefore no $\sigma_{33}^f$.
  - Table 11.1 prints the running subscript `k_{i3}` in its first two columns,
    which belong to the first compressional wave alone: the `mu1`, `D1` and
    `E1` in the same columns make `k_{13}` the only consistent reading.
- **Evidence:** the two readings above are forced by Table 11.1, which the same
  page declares to be the tabulation of Eqs. (11.37), (11.38) and
  (11.46)-(11.48). They are also what the stress-strain relation Eq. (11.41),
  $\sigma_{33}^f = R\,\mathrm{div}\,u_f + Q\,\mathrm{div}\,u_s$, gives when
  the displacement potentials
  of Eqs. (11.22)-(11.25) are differentiated directly.
- **Library behaviour:** the $[\Gamma]$ of Table 11.1 is implemented with the
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
  - Taking the peak as the maximum of $\text{Im}(Z_s)$, Eq. (6.107) on the fully
    specified Table 6.1 glass wool gives 863,5 Hz for 5,6 cm (+0,4 % against
    the printed 860) and 896,2 Hz for 5,4 cm (+4,2 %), which favours 5,6 cm.
    But the same rule puts the undisputed 10 cm sample at 480,0 Hz against
    its printed 470, a +2,1 % bias of the same size as the effect being
    resolved.
  - Taking the peak as the maximum of $|Z_s - Z_{s,\text{rigid}}|$, which is the
    departure the same paragraph describes ("close to each other, except
    around the peaks which are not predicted by the one-wave model"), the
    10 cm sample lands at 469,2 Hz (-0,2 %) and **both** printed frequencies
    then come out of the pair (10 cm, 5,4 cm): 861,2 Hz for 5,4 cm (+0,1 %)
    against 831,0 Hz for 5,6 cm (-3,4 %). That rule favours 5,4 cm.
  - Scaling the 10 cm peak is no help either, and leans the other way from
    the conclusion: $470 \times (10/5{,}4) = 870\ \text{Hz}$ is 10 Hz from the
    published 860, $470 \times (10/5{,}6) = 839\ \text{Hz}$ is 21 Hz from it.
  - The agreement of "860 Hz" with "5,6 cm" is in any case partly circular,
    since both sit in the same clause: it tests that sentence against itself,
    not which of the two sentences is the misprint.
- **Library behaviour:** recorded, with no effect on the implementation.
  `test_impedance_peak_of_the_thin_layer_resolves_the_printed_thickness` pins
  the 5,6 cm peak against the published 860 Hz under the $\text{Im}(Z_s)$ rule and
  checks that the 5,4 cm reading is the worse of the two under that rule.
- **Status:** unreported, and the weakest of the four entries here: the
  conclusion rests on the two-against-one reading of the printed page, not on
  a computation.

---

## Allard & Atalla 2e (2009), Sect. 6.5.4 (the frame-borne velocity ratio)

*Textbook, not a standard.*

- **Location:** Sect. 6.5.4, printed p. 125, the one sentence of the book that
  quotes computed values of $\mu_b$ for the Table 6.1 glass wool.
- **The print:** "The ratio modulus `|mu_b|` of the velocities of the frame
  and the air for the frame-borne wave decreases from 1,0 at 50 Hz to 0,82 at
  1500 Hz."
- **The problem:** the two quoted values are the *real part* of $\mu_b$, not
  its modulus. $\mu_b$ is complex, and the sentence names the modulus
  explicitly.
- **Evidence:** on the fully specified Table 6.1 material the model gives
  $\mu_b(1500\ \text{Hz}) = 0{,}811 + 0{,}473j$. Its real part is **0,811**,
  1,1 % from
  the printed 0,82; its modulus is **0,939**, 14,5 % away. Read as the real
  part, the sentence is right at both ends and describes a monotone decrease:
  $\text{Re}(\mu_b)$ is 1,002 at 50 Hz and passes through 0,82 at 1467 Hz,
  2,2 % from
  the printed 1500 Hz. Read as the modulus it is right at neither: $|\mu_b|$
  is 1,002 at 50 Hz but *rises* to 1,008 by 400 Hz before turning over, and
  only reaches 0,82 at 2634 Hz, 76 % above the printed frequency. No
  admissible reading of the printed inputs closes that gap. With the loss
  factor at 0 or at 0,2, the viscous length halved or doubled,
  $\Lambda' = 2\Lambda$ in place of the printed 1,1e-4 m, the resistivity
  halved or doubled, the tortuosity at 1 or the Poisson coefficient at 0,3,
  $|\mu_b(1500)|$ moves only between 0,874 and 1,073. The closest of the
  eight,
  0,874 at zero loss factor, is still 6,6 % from the printed 0,82, and it
  loses the 495 Hz branch crossing of the same section altogether; the only
  variant that keeps that crossing ($\Lambda' = 2\Lambda$, 495,2 Hz) leaves
  $|\mu_b|$ at 0,937. Reading the sentence as $\text{Re}(\mu_b)$ needs no
  variant at all.
- **Library behaviour:** `biot_waves` computes $\mu_b$ from Eq. (6.71) as
  printed. The conformance row and
  `test_frame_borne_velocity_ratio_matches_the_two_published_values` are
  written against $\text{Re}(\mu_b)$, and say so.
- **Status:** unreported.

---

## Related source properties that are not errata

Recorded here to prevent future "fixes" that would break agreement with the
published sources:

- **ISO 12354-1:2017 Table L.8 / ISO 12354-2:2017 Table G.8, first row:** the
  row labelled "Int. wall 1/2 – Ext. wall 1/2" prints
  $m'_i = 219{,}0\ \text{kg/m}^2$ and
  $m'_{\perp i}$ (Part 2: m'orthogonal) = 360,0 kg/m², which is the assignment
  for a path *leaving the external wall*, the opposite of the direction the
  row's own label gives. Read in the row's direction the element carrying the
  path is the internal wall, so $m'_i$ should be 360,0 and the perpendicular
  mass 219,0. It is a labelling slip and nothing else: the branch is the
  rigid-T **corner** branch $K_{12} = 5{,}7 + 5{,}7 M^2$, where only $M^2$
  enters, so both
  assignments return the same 5,965 → 6,0 dB. The second row of each table,
  "Ext. wall 1/2 – Ext. wall 1/2", is the through branch
  $5{,}7 + 14{,}1 M + 5{,}7 M^2$, where the sign of $M$ does matter, and it is
  labelled and populated consistently ($M = \log_{10}(360/219)$ gives 9,006 → the
  printed 9,0).
  Renders: `plan/[ISO 12354-1_2017] airborne insulation between rooms
  prediction.pdf`, PDF page 89, printed p. 83, 200 dpi, and `plan/[ISO
  12354-2_2017] impact sound insulation between rooms prediction.pdf`, PDF
  page 46, printed p. 40, 200 dpi. Not registered as an erratum because no
  number depends on it; registered here so that a future reader does not
  "correct" the library's per-path convention to match the printed row.
- **Francois-Garrison pure-water term:** the two published $A_3$ cubics do not
  meet exactly at the 20 °C switch (a step of
  $1 \cdot 10^{-7} f^2\ \text{dB/km}$, 0,1 dB/km at
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
  is reported to head the seven $L_{r,TR}$ columns "B/S B/M B/H B/S B/M B/H B/H",
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
  Eq. (17.50) with the $Q = 2$ and $L_W = 70\ \text{dB}$ that yield his 60 dB
  at 1,2 m
  gives 52,5 dB. It is left unregistered because the intended reading cannot
  be established from the book: 54 dB is also what the same equation gives at
  2,5 m (54,1 dB, and 2,5 m is the table spacing the next paragraph derives),
  and what a single 6 dB distance doubling from the rounded 60 dB would give,
  while the printed "3 m (10 ft)" is self-consistent in both units and is
  repeated in the preceding paragraph. `speech_direct_level` evaluates
  Eq. (17.50) as printed, so it returns 52,5 dB there; do not "correct" it
  toward 54 dB.
- **ICAO Annex 16 EPNL constant:** the Annex's rounded constant 13 for
  uniform 0,5 s records differs from the exact $-10\log_{10}(T_0)$ form by 0,0103 dB;
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
  3 prints a *Sum* of 49 dB at 500 Hz where $76 - 28 = 48$, then a *Combined*
  consistent with 48), which is why the comparison runs at the 1 dB the
  printed sheet carries.

<!-- END GENERATED BODY -->
