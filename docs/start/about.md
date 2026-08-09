← [Documentation index](../README.md)

# About

A library that asks you to trust its numbers owes you three things: the name of
the person behind them, the method they come from, and a way to tell that
person when one of them is wrong. This page is those three.

## Who maintains phonometry

I am José Manuel Requena Plens. I wrote phonometry — originally released as
PyOctaveBand — and I maintain it: the implementations, the test suite, the
documentation and the releases. There is no company behind it and no team, so
anything wrong here is mine to fix.

The identity is verifiable rather than asserted:

- **ORCID:** [0000-0003-1250-6212](https://orcid.org/0000-0003-1250-6212)
- **Google Scholar:** [scholar.google.com/citations?user=9b0kPaUAAAAJ](https://scholar.google.com/citations?user=9b0kPaUAAAAJ)
- **ResearchGate:** [Jose Requena Plens](https://www.researchgate.net/profile/Jose-Requena-Plens-2)
- **GitHub:** [@jmrplens](https://github.com/jmrplens)
- **Personal site:** [jmrp.io](https://jmrp.io)

The same identity is published on
[MathWorks File Exchange](https://www.mathworks.com/matlabcentral/profile/authors/5890853),
[LinkedIn](https://www.linkedin.com/in/jmrplens),
[Mastodon](https://mstdn.jmrp.io/@jmrplens) and
[Keyoxide](https://keyoxide.org/0A993B268654DBBA52B7E8D3FCF653391E2C91FC),
which lists the OpenPGP key that signs my work.

The background behind it is acoustics and signal processing, and it is on the
record: Telecommunications Engineer specialised in Sound and Image at the
Universidad de Alicante (2011-2018), MSc in Acoustics Engineering at the
Universitat Politècnica de València (2018-2019), and acoustics researcher at
the UPV from 2020 to 2023, publishing on acoustic metasurfaces, sound diffusers
and acoustic field prediction — the papers are under the ORCID and Scholar
profiles above. Today I work in industrial R&D as a firmware and software
engineer.

The library grew out of the measurement side of that work. I kept needing
octave-band levels and sound level meter quantities I could defend against a
tolerance table, so I built them once, properly, instead of re-deriving them
per project.

## How the library is built

The method is the whole point of the project, and it is deliberately narrow.

1. **Each metric is implemented from the text of its governing standard**,
   clause by clause, rather than from a secondary description of it or from
   somebody else's implementation. Where a formula is ambiguous, the ambiguity
   is resolved from the document itself or from the physics, and the reading
   chosen is written down.
2. **The standard's own reference values are transcribed into the test
   suite.** Where a standard publishes a worked example, a tolerance table or a
   set of nominal responses — the IEC 61672-1 Table 3 weighting values and its
   Table 4 tone-burst responses, the IEC 61260-1 Table 1 filter limits, the
   ISO 226 Annex B contours, and so on — those numbers become the expected
   values of real tests.
3. **CI enforces them on every change.** A regression that moves a computed
   value outside the standard's acceptance limit fails the build, so a claim
   cannot quietly stop being true between releases.

Not every standard publishes a numerical example. Where one does not, the
implementation is anchored to the closed-form expressions of the normative text
and pinned with a case synthesized to a known result. That is a weaker
guarantee than a transcribed worked example, and the
[conformance report](../CONFORMANCE.md) says so in the expected-value cell
itself rather than as a flag you could filter away: look for the words "closed
form" or "analytic". The distinction is worth stating plainly. A closed-form
check proves the implementation agrees with the equations as written; it cannot
catch a misreading of the clause those equations came from. That is why the
[errata registry](../ERRATA.md) exists beside it.

The evidence is public. The conformance report lists, per check, the standard
and clause, the normative expected value, the value the library computes, the
delta and a pass or fail verdict, and CI regenerates it on every pull request
so it cannot drift from the code. [Why phonometry](why-phonometry.md) walks the
method end to end on a single case — time weighting under IEC 61672-1:2013 —
if you want to see it applied before trusting it in general.

## Reporting an error

Please report it. A wrong number nobody mentions stays wrong.

Open an issue at
[github.com/jmrplens/phonometry/issues](https://github.com/jmrplens/phonometry/issues).
The most useful report names the standard and clause you are checking against,
the expected value, the value phonometry produced, and a short snippet that
reproduces it. If the signal cannot be shared, the parameters alone are usually
enough to rebuild a case.

If an issue tracker is not an option, write to [mail@jmrp.io](mailto:mail@jmrp.io);
the issue is still the better channel, because the report and the fix stay
public next to the code they concern.

**That includes errors in the standards themselves.** Re-deriving formulae and
recomputing worked examples from the source documents occasionally turns up
defects in the sources: misprints, worked examples that contradict their own
normative clauses, ambiguous wording. Those are not silently worked around.
Each confirmed case is recorded in [`ERRATA.md`](../ERRATA.md) with the
location, the evidence, the reading the library implements and whether it has
been reported to the issuing body; the site transplants that same file at build
time, so the two editions cannot disagree. The registry covers standards
(ISO, IEC, EN), guidance documents and technical reports, textbooks and journal
papers alike. Disagreeing with one of those readings is exactly the kind of
issue worth opening.

## Cite this software

If phonometry contributed to published work, please cite it. The archived
record and its DOI are on Zenodo:

**[doi.org/10.5281/zenodo.21215280](https://doi.org/10.5281/zenodo.21215280)**

That is the *concept* DOI: it always resolves to the most recent archived
release, and each release also carries its own version DOI on the same record.
Cite the version you actually ran.

APA:

> Requena-Plens, J. M. (2026). *phonometry: acoustic measurement toolkit for
> Python (formerly PyOctaveBand)* (Version 3.3.0) [Computer software].
> https://doi.org/10.5281/zenodo.21215280

BibTeX:

```bibtex
@software{requenaplens_phonometry,
  author  = {Requena-Plens, Jos{\'e} M.},
  title   = {phonometry: acoustic measurement toolkit for Python
             (formerly PyOctaveBand)},
  year    = {2026},
  version = {3.3.0},
  doi     = {10.5281/zenodo.21215280},
  url     = {https://jmrplens.github.io/phonometry/},
  license = {MIT}
}
```

Both entries derive from
[`CITATION.cff`](https://github.com/jmrplens/phonometry/blob/main/CITATION.cff)
in the repository, which is the authoritative metadata and the file GitHub and
Zenodo read. With a reference manager, import that file rather than copying the
block above, and adjust the version and year to the release you used.

## Licence

phonometry is released under the
[MIT licence](https://github.com/jmrplens/phonometry/blob/main/LICENSE), so it
can be used in commercial and academic work, modified and redistributed, as
long as the copyright notice and the licence text travel with it. The software
comes without warranty: it is verified against the standards it implements, as
described above, but **it is not a calibrated instrument and it carries no
accreditation**. Deciding whether a result is fit for a purpose stays with the
person using it.

Concretely, for the laboratories and consultancies this is mostly written for:
the library computes and documents, it does not measure, so **traceability
stays with the hardware chain** — a calibrator and a microphone with valid
certificates, and type-approved instrumentation wherever the measurement has
legal effect. Accreditation bodies expect **calculation software to be
validated for its intended use, with the record retained**, and the artefact
for that is the conformance report of the exact version that ran: pin the
version, archive its report with the measurement data, and keep the errata
registry alongside, since that is where every difference between the
implemented reading and a printed source is documented. The uncertainty
machinery propagates what it is given: the calibrator tolerance and the
pre/post drift are still yours to enter.

The standards themselves are copyrighted by their issuing bodies and are not
redistributed here. This documentation cites clauses and reference values as
far as it takes to explain and verify the implementations; it is not a
substitute for buying the documents.

## Where to go from here

[Getting Started](getting-started.md) installs the library and runs a first
calibrated analysis, and [All guides](../README.md) is the map of the whole
library grouped by topic. To check a number rather than install anything, the
[conformance report](../CONFORMANCE.md) and the [errata registry](../ERRATA.md)
are the two pages that answer that.
