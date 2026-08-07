---
title: "Reference"
description: "The part of the site that does not teach: the theory behind every module, the numerical conformance report that pins each metric to its standard clause, and the bibliography of every source the guides cite."
---

The guides show how to measure something. This section answers the three
questions that come after: **why is the formula this one**, **is the
implementation actually right**, and **where does it come from**. Nothing here
is a tutorial, so nothing here has to be read in order; these are pages to
arrive at from a guide, from a report you are writing, or from a reviewer's
question.

Use it when you are defending a result rather than producing one. If a
colleague asks which clause of ISO 3382-1 the reverberation time follows, the
theory pages say so; if a client asks for evidence that the library computes
it correctly, the conformance report shows the standard's own expected value
next to the computed one; if a journal asks for the source of a model, the
bibliography has the DOI.

If you are only looking for the signature of a function, you want the
[API reference](/phonometry/reference/api/) instead: it is generated from the
source docstrings and lives in its own section of the sidebar.

## [Theory](/phonometry/reference/theory/)

The standards, the mathematics and the design decisions behind every module,
in six domain pages: signal analysis, perception and hearing, rooms and
buildings, materials and surfaces, environment and transport, and vibration.
Start here when a guide states a result and you want the derivation, the
clause it implements, or the reason one formulation was chosen over another.
The theory for the underwater modules is the exception: it lives inline with
[its guides](/phonometry/underwater/).

## [Conformance report](/phonometry/reference/conformance/)

The numerical evidence. Every check names a standard, a clause or table, the
normative expected value, the value the library computes, the delta and a
pass/fail verdict. It is regenerated and enforced by CI on every pull request,
so it describes the code as it is now rather than as it was documented once.
Read it when you need to show that a number is defensible, or to see exactly
which parts of a standard are implemented and which are not.

## [Errata in published sources](/phonometry/reference/errata/)

The other side of the same coin. Re-deriving every formula and worked example
from the source document occasionally shows that the *document* is wrong: an
example that contradicts its own normative clause, a misprinted constant, a
cross-reference pointing at the wrong equation. Each confirmed case is
recorded with the printed edition, the evidence, the reading the library
implements and the test that pins it. Read it when a printed expected value
and the library disagree, before assuming the library is at fault.

## [Glossary of quantities](/phonometry/reference/glossary/)

Every acoustic quantity the guides compute, with its symbol, a one-sentence
definition, its unit, the standard and clause that defines it, and the guide
that implements it. Read it when a symbol in a formula needs a name, or a name
needs the clause it came from.

## [Bibliography](/phonometry/reference/bibliography/)

A curated selection of the books and papers the guides lean on, in one list
grouped by domain, each entry with a verified DOI or official publisher link,
half a sentence on what it supports, and the guide pages that cite it. It is a
reading list for the field rather than the complete citation index: the
authoritative list for any one page is that page's own References section,
generated from its frontmatter.
