# NORAH2 ARP verification subset — rotorcraft oracle data

## What this folder contains

`norah2_arp_extract.zip` holds the **verification subset** of the NORAH2 public
release (V2.0.74) consumed by `tests/aircraft/test_rotorcraft_norah2.py`: the
R22 hemisphere set with its triangulation lookup (`Hemispheres/R22_*`) and the
ARP single-event cases used as oracles (input flight paths `.inp`, per-step
microphone histories `.his` and per-microphone metrics `.onl` for the Case 2,
Case 3 and Case 4 operations listed in that module). The internal layout of the
original archive is preserved, so the same parser code reads either source.

It is a byte-identical extract of 44 of the ~360 files of the public release;
nothing is modified, resampled or re-encoded.

## Source and authorship

- Extracted from the **NORAH2 V2.0.74 public release**, the prototype and
  database published by the EASA research project **EASA.2020.FC.06** (NORAH,
  noise of rotorcraft assessed by a hemisphere method). The rotorcraft module
  of this library is a clean-room implementation of ECAC Doc 32 and the NORAH2
  modelling guidance (SC01.D1.5d) and shares no code with the prototype.
- The extract carries only measurement-derived model data (source-noise
  hemispheres and verification-case inputs/outputs), no software: the
  prototype executables and the full database are **not** committed.

## Purpose and scope of use

The subset lets the end-to-end oracle suite run everywhere, including CI cells
without access to the full archive, validating the flight-condition
interpolation, kinematics/retarded time, `rotorcraft_event_level` and
`rotorcraft_noise_contour` against the published prototype outputs. When the
full public release is available locally the suite prefers it: an extraction
pointed at by `NORAH2_DATA`, or
`tests/data-local/norah2/NORAH2_V2.0.74_public.zip` (see
`tests/data/README.md`). Every case reads the same files from either source, so
the assertions do not differ. The data are **not** part of the `phonometry`
package and are not installed with it.

## Removal policy

If you represent EASA or the NORAH project consortium and consider that
committing this verification subset exceeds the intended use of the public
release, please open an issue or contact the maintainer (see `CITATION.cff`)
and it will be removed promptly, together with the CI use of the tests that
read it.
