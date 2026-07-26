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
  - name: José Manuel Requena Plens
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
hundreds more. At the time of writing, continuous integration enforces 427
numerical conformance checks against 278 referenced standards on every change,
and the per-check report (standard and clause, normative expected value,
computed value, delta, verdict) is published with the documentation
(<https://jmrplens.github.io/phonometry/reference/conformance/>), alongside a
bibliography listing all 278 referenced standards with their editions
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
weighted insulation rating, a declared sound power level. General-purpose
scientific audio toolkits such as `python-acoustics` [@pythonacoustics] and
`pyfar` [@pyfar] provide excellent signal-processing foundations, and
specialised packages validate individual metrics against their reference
data, as `MOSQITO` [@mosqito] does for its loudness implementations against
the ISO 532 test signals. What no library in this space publishes is
conformance evidence that is systematic across every metric it ships, per
check, and enforced in continuous integration, so that the verification
cannot drift from the code it describes. When implementations disagree, and
they do, the user is otherwise left to guess which one the standard would
side with.

`phonometry` exists to close that gap. Its contribution is a method as much as
a library: implement from the primary normative source, transcribe the
standard's own reference values as test oracles, and publish the full
conformance evidence. Where a standard publishes no numerical example, the
implementation is anchored to the closed-form expressions of the normative
text and pinned with a synthesized case; the conformance report labels these
weaker anchorings as such instead of rounding them up. The process of
re-deriving formulas and recomputing worked examples has also surfaced defects
in the published standards themselves: misprints, worked examples that
contradict their own normative clauses, and broken cross-references. These are
documented in a public errata registry, with evidence, the reading the library
implements, and whether the defect has been reported to the issuing body,
rather than being silently worked around.

The library targets practitioners who need defensible numbers (measurement
against IEC 61672-1 and IEC 61260-1, ratings under ISO 717 and related
building-acoustics standards, environmental indicators under ISO 1996),
educators who want each quantity traceable to its defining clause, and
researchers who need a verified baseline against which to compare new methods.
The documentation, in English and Spanish, pairs every metric with a guide
that teaches the underlying method and cites the governing clauses.

# Functionality and design

The library is organised in twelve domain namespaces (metrology,
psychoacoustics, building, environment, underwater, aircraft and others) over
a flat API that remains importable directly. Every computation returns a
typed, frozen dataclass rather than a bare array: the result object carries
the inputs and parameters it was computed from, exposes a `plot()` method
that draws the conventional figure for that quantity, and, for metrics whose
standards define a reporting format (insulation ratings, absorption classes,
declared noise emission), a `report()` method that renders an
accredited-laboratory-style fiche citing the governing clauses. This makes
results self-documenting: the object a script passes around is also the
evidence trail a report needs.

The implementation is pure Python on NumPy and SciPy, with optional Numba
acceleration for the heaviest numerical kernels and block-processing entry
points for signals that do not fit in memory. The documentation, in English
and Spanish, pairs every metric with a guide that teaches the method and a
reference page that cites the exact clauses and formula numbers implemented;
machine-readable copies of every page are published alongside the rendered
site. The library has seen downstream research use under its previous name,
for example in hearing-aid signal-processing research [@dnnha].

# Acknowledgements

`phonometry` was first released in 2020 as `PyOctaveBand`, an octave-band
filtering package; the project was renamed when its scope outgrew the name.
The author thanks the users who reported numerical discrepancies over the
years: several of the conformance checks exist because someone asked why two
implementations disagreed.

# References
