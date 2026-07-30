# EBU programme-loudness — derived validation data

## What this folder contains

`ebu_programme_block_loudness.npz` holds **block-loudness series** measured from
the authentic-programme cases of the *EBU loudness test set* (v04). They are used
by the test suite to validate the gating and loudness-range stages of
`phonometry.broadcast.program_loudness` against the EBU Tech 3341/3342 targets
without any network access.

| Array | Blocks | Source case | Content |
| :--- | :--- | :--- | :--- |
| `nlr_momentary` | 400 ms / 100 ms hop | Tech 3341 case 7 (NLR narration) | Momentary loudness series, LUFS (the BS.1770-5 integrated-loudness gate input) |
| `nlr_short_term` | 3 s / 100 ms hop | Tech 3342 case 5 (NLR) | Short-term loudness series, LUFS (the loudness-range input) |
| `wlr_momentary` | 400 ms / 100 ms hop | Tech 3341 case 8 (WLR movie/drama) | Momentary loudness series, LUFS |
| `wlr_short_term` | 3 s / 100 ms hop | Tech 3342 case 6 (WLR) | Short-term loudness series, LUFS |

The series were measured with `phonometry.broadcast.program_loudness` from
`seq-3341-7_seq-3342-5-24bit.wav` (NLR) and
`seq-3341-2011-8_seq-3342-6-24bit-v02.wav` (WLR), then stored rounded to
0.0001 LU (three orders of magnitude below the official tolerances). They pin
the integrated loudness to `-23.0 LUFS` and the loudness range to `5 LU` (NLR)
and `15 LU` (WLR), the EBU-published targets for those cases.

## Why only the derived series, not the audio

Because the audio may not be redistributed, by two independent constraints.

The EBU's own *Use of EBU audio test sequences* (v1.0, July 2019) states:
*"You may not use any EBU Test Sequences for business, commercial or for-profit
activities. You may not copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the EBU Test Sequences. All rights, titles and interests
relating to any and all specific EBU Test Sequences belong to the EBU."* The
set's own `readme.txt` adds: *"The material may only be used for technical
testing purposes."* That rules out committing **any** file of the set, not only
the programme cases.

On top of that, cases 7/8 and 5/6 are licensed feature-film excerpts: the
`readme.txt` credits *The Misfortunates* (dir. Felix van Groeningen, prod.
Menuet, Belgium) and *Bloody Mondays and Strawberry Pies* (dir. Coco Schreiber,
prod. Bonanza Films, the Netherlands), whose rights holders gave permission to
the EBU, not to downstream users.

What is committed is therefore **measurement data, not audio**: a per-block
loudness envelope at a 100 ms hop is our own computation and cannot reconstruct
the programme signal.

## Source and authorship

- Derived from the **EBU loudness test set** (v04), © EBU (European Broadcasting
  Union), <https://tech.ebu.ch/publications/ebu_loudness_test_set>
  (retrieved 2026-07-30; the page now offers v05).
- Terms of use:
  <https://tech.ebu.ch/files/live/sites/tech/files/shared/testmaterial/use%20of%20EBU%20AUDIO%20test%20sequences.pdf>
  (*Use of EBU audio test sequences*, v1.0, July 2019), quoted above.
- The target values reproduced by these series are defined in **EBU Tech 3341**
  (loudness metering) and **EBU Tech 3342** (loudness range), which in turn build
  on ITU-R BS.1770.
- The `.npz` itself is an original transcription of numeric measurements, not an
  EBU file. Following the EBU's quotation clause: © EBU for the test sequences
  these measurements were taken from.

## Purpose and scope of use

The series are consumed by `tests/broadcast/test_ebu_material_oracle.py` to
demonstrate that this library's independent gating and loudness-range
implementation reproduces the EBU authentic-programme targets, everywhere
including CI cells without network access. They are **not** part of the
`phonometry` package and are not installed with it.

What they cannot cover is the chain *upstream* of the block envelope on
authentic programme material: 24-bit WAV decoding, the BS.1770 K-weighting
front end, the 400 ms / 100 ms segmentation and the channel weighting. The same
module asserts those end to end from the original audio, at the same
tolerances, wherever a local copy of the set exists (`EBU_LOUDNESS_TEST_SET`,
or `tests/data-local/ebu-loudness-test-set/`); where the audio is there it also
re-measures the envelopes and checks them against this file, so the derived
oracle cannot drift away from the material it represents. Those stages are
covered in CI by the synthesizable EBU cases in
`tests/broadcast/test_program_loudness.py`, built from the specifications.

## Removal policy

If you represent the EBU and consider that publishing these derived series
exceeds the intended use of the test set, please open an issue or contact the
maintainer (see `CITATION.cff`) and they will be removed promptly, together
with the `tests/broadcast/test_ebu_material_oracle.py` tests that read them.
