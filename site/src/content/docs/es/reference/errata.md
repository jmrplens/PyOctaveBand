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
implements that reading with a regression test pinning it.

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

---

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
- **ICAO Annex 16 EPNL constant:** the Annex's rounded constant 13 for
  uniform 0,5 s records differs from the exact −10·lg(T0) form by 0,0103 dB;
  the library uses the exact form, which the ETM's integrated reference
  reproduces to five decimals.

<!-- END GENERATED BODY -->
