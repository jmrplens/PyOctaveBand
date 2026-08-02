# ISO 532-1:2017 electronic attachment — validation data

## What this folder contains

Reference test signals and validation data from the **electronic attachment of
ISO 532-1:2017** (*Acoustics — Methods for calculating loudness — Part 1:
Zwicker method*), used by this library's test suite and conformance report to
validate the `loudness_zwicker` implementation against the standard's own
Annex B programme, exactly as Annex B specifies.

| File(s) | Origin in the attachment | Content |
| :--- | :--- | :--- |
| `Annex B.5/Test signal 14 … 25 *.wav` | `Annex B.5/` (unchanged names) | The twelve recorded technical signals for time-varying loudness validation |
| `iso532_1_test_signal_10.wav` … `13.wav` | `Annex B.4/Test signal 10 (tone pulse 1 kHz 10 ms 70 dB).wav` … `13 (combined tone pulses 1 kHz).wav` | Tone-pulse synthetic signals (renamed; content byte-identical) |
| `iso532_1_test_signal_1_levels.txt` | `Annex B.2/Test signal 1.txt` | Third-octave levels of test signal 1 (renamed; content byte-identical) |
| `iso532_1_annexB_expected.json` | transcribed from the `Results and tests …` workbooks (Annexes B.2–B.5) | Expected loudness values and tolerances used as test oracles |
| `iso532_1_annexB4_traces.npz` | transcribed from `Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx` | Reference loudness-vs-time traces for signals 10–13 |

All `.wav` and `.txt` files are **byte-identical** to the files in the official
attachment archive (verified by checksum at the time of import). The `.json`
and `.npz` files are transcriptions of numeric values from the attachment's
result workbooks, not original ISO files.

## Source and authorship

- © ISO 2017. These files are part of the ISO 532-1:2017 electronic
  attachment, authored and published by the International Organization for
  Standardization (ISO/TC 43, Acoustics).
- Official source (publicly downloadable):
  <https://standards.iso.org/iso/532/-1/ed-1/en/> —
  `ISO 532-1 - Program etc.zip` (retrieved 2026-07-08).
- ISO licensing information:
  <https://www.iso.org/terms-conditions-licence-agreement.html#Customer-Licence>
  and `copyright@iso.org`.

## Purpose and scope of use

ISO publishes this attachment so that implementations of the ISO 532-1 method
can be validated against the standard's reference signals and tolerances
(Annex B). The copies in this folder serve **exactly that purpose**: they are
consumed by `tests/psychoacoustics/test_loudness_zwicker.py` and `scripts/conformance_report.py` to
demonstrate conformance of this library's independent implementation. They are
**not** part of the `phonometry` package, are not installed with it, and are
not covered by this repository's MIT licence — ISO's terms apply to them.

The remainder of the attachment (the reference C programme, executables and
result workbooks) is intentionally **not** included: this library's
implementation is written independently from the text of the standard.

## Removal policy

If you represent ISO and consider that this redistribution exceeds the
intended use of the electronic attachment, please open an issue or contact the
maintainer (see `CITATION.cff`) and the files will be removed promptly. The
test suite degrades gracefully: without this folder the affected tests skip
(set `ISO532_1_TESTDATA` to a local copy of the attachment to restore them).
