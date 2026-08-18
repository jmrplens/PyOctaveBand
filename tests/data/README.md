# Committed oracle data — the convention

Some suites are validated against reference material published by a standards
body, a research project or a manufacturer. That material is often large,
usually copyrighted and sometimes explicitly non-redistributable, so it cannot
simply be dropped into the repository. This folder holds what **is** committed,
and every dataset here follows the same rules.

## Where the data lives

| Location | Versioned | Content |
| :--- | :--- | :--- |
| `tests/data/<dataset>/` | yes | The small committed oracle: a derived measurement series, or a lossless extract of a representative subset. A few megabytes at most, each with the README described below. |
| `tests/data-local/<dataset>/` | **no** (gitignored) | A full local copy of the original set, for whoever has one. Hundreds of megabytes are fine here; nothing in the repository depends on it. |

Every suite resolves its data through `tests/oracle_data.py`, always in the
same order:

1. the dataset's historical environment override, when it is set and points at
   an existing copy (`STIPA_VERIFICATION_DATA`, `EBU_LOUDNESS_TEST_SET`,
   `NORAH2_DATA`);
2. `tests/data-local/<dataset>/`;
3. the committed data under `tests/data/<dataset>/`.

(`iso532_1/` predates the split and has no separate full set: its committed
folder *is* the oracle, and `ISO532_1_TESTDATA` relocates that same folder.)

**The assertions on the committed data must never skip.** That is the whole
point: CI has neither the environment variable nor `tests/data-local/`, so it
takes route 3, and route 3 has to assert real published values. A suite that
steps over itself in CI is a suite nobody but the maintainer runs.

What may still skip on route 3 is a test that exists *only* to exercise the
full set and has no committed counterpart — the six EBU cases that read the
programme audio, and the inventory guard on the full STIPA download. Those
report as skipped in CI by design. The rule is about coverage, not about the
skip count: every published value the full set pins must also be pinned by
something that runs without it.

`pytest` prints the resolution of every dataset in its run header, so a green
run is never ambiguous about which oracle produced it:

```text
oracle data:
  stipa-verification: full set from tests/data-local/stipa-verification
  ebu-loudness-test-set: committed oracle (block-loudness series, tests/data/broadcast/)
  norah2: committed oracle (44-file ARP subset, tests/data/norah2/)
```

## What to commit, in order of preference

1. **Derived** — a measurement series, a transcribed table, expected values
   computed from the original material. Preferred, because a derived product is
   our own computation and sidesteps the licensing question entirely: it cannot
   reconstruct the source material. `tests/data/broadcast/` is the example, 36 kB
   of block-loudness envelopes standing in for 289 MB of audio that may not be
   redistributed at all.
2. **Extract** — a byte-identical (or bit-exact) subset of the original files.
   Only when the test genuinely needs the real material through the real signal
   path, only a small representative subset, and only when the terms allow
   redistribution. `tests/data/norah2/` (44 of ~360 files, 676 kB) and
   `tests/data/stipa/` (27 of 49 signals, 5,8 MB) are the examples.
3. **The full set** — essentially never. Establish the terms first: *free to
   download* is not *free to redistribute*.

Keep the committed footprint in the low megabytes. The repository history is
immutable and audio does not delta-compress, so a careless import is permanent.

## The README every dataset needs

Four sections, following `tests/data/iso532_1/README.md`:

1. **What this folder contains** — a table giving the provenance file by file,
   distinguishing byte-identical copies from transcribed or derived values.
2. **Source and authorship** — the copyright holder, the official public URL,
   the retrieval date, and a link to (or quotation of) the licensing terms.
3. **Purpose and scope of use** — why this particular redistribution is
   legitimate, an explicit statement that **this repository's MIT licence does
   not cover these files** and whose terms do, and what is deliberately
   excluded.
4. **Removal policy** — who to contact, and the fact that the suite degrades
   gracefully (or, where it does not, what would have to be removed with it).

## Current datasets

| Folder | Kind | Size | Original |
| :--- | :--- | ---: | :--- |
| `audio/` | extract + derived | 6,9 MB | Real measurement audio with published values, five sources (see its README) |
| `broadcast/` | derived | 36 kB | EBU loudness test set (289 MB) |
| `cnossos/` | derived + transcribed | 13 kB | CIRCABC CNOSSOS-EU emission test workbooks (10 MB) |
| `iso532_1/` | extract + derived | 4,6 MB | ISO 532-1:2017 electronic attachment |
| `norah2/` | extract | 676 kB | NORAH2 V2.0.74 public release |
| `stipa/` | extract | 5,8 MB | stipa.info IEC 60268-16 verification bench (133 MB) |
