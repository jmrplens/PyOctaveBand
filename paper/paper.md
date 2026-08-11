---
title: "phonometry: standard-conformant acoustic measurement in Python"
tags:
  - Python
  - acoustics
  - signal processing
  - psychoacoustics
  - building acoustics
  - noise
  - metrology
authors:
  - name:
      given-names: José Manuel
      surname: Requena-Plens
    orcid: 0000-0003-1250-6212
    affiliation: 1
affiliations:
  - name: Independent researcher, València, Spain
    index: 1
date: 27 July 2026
bibliography: paper.bib
---

# Summary

`phonometry` is a pure-Python library, built on NumPy [@harris2020array] and
SciPy [@virtanen2020scipy], for acoustic measurement, analysis and prediction.
It covers sound level metrology, fractional-octave filtering, psychoacoustics,
room and building acoustics, acoustic materials, human vibration, environmental
and transport noise, underwater acoustics, electroacoustics and wave
simulation. Every result is a typed, frozen dataclass that carries the inputs
it was computed from, draws its own figure, and, for the metrics that certify
or rate something, renders an accredited-style report.

What distinguishes the library is not the breadth of that list but how each
entry on it is built and verified. Every metric is implemented from the text of
its governing standard, clause by clause, rather than from a secondary
description or another implementation. Where a standard publishes worked
examples, tolerance tables or nominal responses, those numbers are transcribed
into the test suite as oracles: the IEC 61672-1 weighting tolerances
[@iec61672], the IEC 61260-1 filter class limits [@iec61260], the ISO 532-1
loudness test signals [@iso532], the ISO 717 rating examples [@iso717], and
hundreds more. At the time of writing, continuous integration enforces 533
numerical conformance checks across 57 domains against 362 referenced
standards on every change, and the per-check report (standard and clause,
normative expected value, computed value, delta, verdict) is published with
the documentation
(<https://jmrplens.github.io/phonometry/reference/conformance/>), alongside a
bibliography of every source the guides cite, standards included with their
catalogue records
(<https://jmrplens.github.io/phonometry/reference/bibliography/>). A
regression that moves a computed value outside a standard's acceptance limit
fails the build, so the conformance claim cannot silently stop being true
between releases. \autoref{fig:mask} shows one such check as the user sees it:
the realized A and C frequency weightings against the class limits of
IEC 61672-1:2013 Table 3.

![Deviation of the library's A and C frequency weightings (48 kHz) from the
design goal, against the class 1 and class 2 acceptance limits of
IEC 61672-1:2013 Table 3. The same tolerance table is transcribed in the test
suite, so this figure and the CI gate share their
oracle.\label{fig:mask}](weighting-class-mask.png){ width=90% }

# Statement of need

Acousticians, noise consultants and researchers routinely have to defend a
number against a tolerance table: a certified sound level meter quantity, a
weighted insulation rating, a declared sound power level. Implementations of
the same standardized metric disagree more often than practitioners expect,
and when they do, the user is left to guess which one the standard would side
with, because the verification evidence behind each implementation is
typically partial, informal or unpublished.

`phonometry` exists to close that gap. It grew out of the author's work as an
acoustics researcher: the clause-by-clause readings, the measurement practice
and the standards library behind the implementations predate the package,
which translates that accumulated domain knowledge into verified code and
didactic documentation. Its contribution is a method as much as a library:
implement from the primary normative source, transcribe the standard's own
reference values as test oracles, and publish the full conformance evidence. Where a standard publishes no numerical example, the
implementation is anchored to the closed-form expressions of the normative
text and pinned with a synthesized case; the conformance report labels these
weaker anchorings as such instead of rounding them up. The library targets
practitioners who need defensible numbers (measurement against IEC 61672-1
and IEC 61260-1, ratings under ISO 717 and related building-acoustics
standards, environmental indicators under ISO 1996), educators who want each
quantity traceable to its defining clause, and researchers who need a
verified baseline against which to compare new methods.

# State of the field

The scientific-Python acoustics ecosystem is strong on foundations and thin on
normative verification. `python-acoustics` [@pythonacoustics] long served as
the general-purpose toolbox for acousticians, covering weighting, octave bands
and several ISO calculations, but its repository is archived and unmaintained.
`pyfar` [@pyfar] is an excellent, actively developed family of packages for
acoustics research (signal classes, DSP, I/O of measurement formats), designed
as a foundation rather than as an implementation of the standardized metric
catalogue. Specialised packages validate individual metrics against their
reference data, as `MOSQITO` [@mosqito] does for its loudness implementations
against the ISO 532 test signals.

Contributing the conformance method to one of these projects was considered
and rejected for a scope reason: per-check normative verification only pays
off when it is systematic, across every metric a library ships and enforced in
its own continuous integration, and none of the existing projects is
structured around that contract. What no library in this space publishes is
conformance evidence that is complete (every shipped metric), granular (per
check, with the normative expected value and the delta) and live (regenerated
and enforced on every change), so that the verification cannot drift from the
code it describes. That is the niche `phonometry` occupies; for
signal-processing foundations it happily coexists with `pyfar`.

# Software design

Three design decisions carry the method. First, every computation returns a
typed, frozen dataclass rather than a bare array: the result object carries
the inputs and parameters it was computed from, exposes a `plot()` method that
draws the conventional figure for that quantity, and, for metrics whose
standards define a reporting format (insulation ratings, absorption classes,
declared noise emission), a `report()` method that renders an
accredited-laboratory-style fiche citing the governing clauses
(\autoref{fig:fiche}). The trade-off is verbosity against provenance: a
frozen result is a self-documenting evidence trail, which is what a
defensible number needs, at the cost of one attribute access compared with
returning arrays.

![The `report()` method of an airborne insulation result: a single-page fiche
in the layout of an accredited laboratory report, stating the measured curve,
the shifted reference curve, the weighted rating and spectrum adaptation
terms, and the ISO 717-1 clauses each value comes
from.\label{fig:fiche}](iso717-report-fiche.png){ width=55% }

Second, the implementation is pure Python on NumPy and SciPy, with optional
Numba acceleration for the heaviest kernels and block-processing entry points
for signals that do not fit in memory. A compiled core would be faster still,
but clause-by-clause auditability of the numerical code, the property the
whole project trades on, favours readable Python whose lines can be mapped to
the formulas they implement.

Third, the oracles live inside the test suite, not in a separate validation
document. Standards' worked examples and tolerance tables are transcribed as
test expectations, and the published conformance report is generated from the
same run that gates merges. The alternative, a hand-maintained validation
report, was rejected because it can silently drift from the code; here the
report cannot exist in a state the tests did not produce. The library is
organised in eighteen domain namespaces over a flat API that remains importable
directly, and the documentation, in English and Spanish, pairs every metric
with a guide that teaches the method and a reference page that cites the
exact clauses and formula numbers implemented; machine-readable copies of
every page are published alongside the rendered site.

# Research impact statement

The library has seen research use under its previous name, `PyOctaveBand`,
released in 2020 and downloaded and integrated by third parties since; a
published example is the DNN-based hearing-aid processing line of work of
@drakopoulos2023, whose released models use it for octave-band analysis. A
second, distinctive form of impact runs in the other direction: re-deriving
formulas and recomputing worked examples from the source documents has
surfaced defects in the published standards themselves (misprints, worked
examples that contradict their own normative clauses, broken
cross-references). Each confirmed case is documented in a public errata
registry with the evidence, the reading the library implements, and whether
it has been reported to the issuing body, so the verification work feeds back
into the standards it verifies. The per-check conformance report also gives
researchers a citable, reproducible baseline: a paper comparing a new loudness
or insulation model against `phonometry` can point to the exact normative
checks its baseline passes.

# AI usage disclosure

Generative AI tools (large language models) were used as an implementation
assistant throughout the development of the library, its documentation and
this paper: drafting code and tests from the author's clause-by-clause
readings of the normative texts and drafting documentation prose, always
under the author's direction and review. The domain knowledge the library
encodes, the normative readings, the measurement practice and the resolution
of ambiguities in the texts, comes from the author's prior research work in
acoustics. The verification framework this
paper describes is also the mechanism by which AI-assisted output is
controlled: every metric, however drafted, must reproduce the standard's own
worked examples and tolerance tables in the conformance suite before it
ships, and the per-check report of that suite is public. The core design
decisions are the author's: the normative-source-only implementation policy,
the oracle-transcription method, the result-object architecture and the
errata registry. All content, including AI-assisted content, was reviewed and
approved by the author.

# Acknowledgements

`phonometry` was first released in 2020 as `PyOctaveBand`, an octave-band
filtering package; the project was renamed when its scope outgrew the name.
The author thanks the users who reported numerical discrepancies over the
years (several of the conformance checks exist because someone asked why two
implementations disagreed), and the colleagues of his research years at the
Universitat Politècnica de València, at the Gandia campus (EPSG) and at the
Institute for Instrumentation in Molecular Imaging (i3M, UPV--CSIC), where
the measurement practice behind this library was formed. This work received
no external funding.

# References
