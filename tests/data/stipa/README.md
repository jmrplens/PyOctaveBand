# IEC 60268-16 STIPA certified verification bench — extract

## What this folder contains

`stipa_certified_extract.zip` holds **27 of the 49 signals** of the certified
STIPA verification bench published on [stipa.info](https://www.stipa.info), the
test signals IEC 60268-16:2020 (rev 5) Annexes A and C prescribe for verifying a
STIPA implementation. They are consumed by
`tests/hearing/test_stipa_certified.py`.

| Verification suite | Committed | Of | Why these |
| :--- | ---: | ---: | :--- |
| `Annex C.3.2` direct-method modulation depth | 3 | 11 | `m = 0` (null), `m = 0,9` (worst measured STI deviation of the staircase, 0,0031) and `m = 1` (top of the dynamic range, worst measured MTF deviation, 0,0065) |
| `Annex C.3.3` indirect-method modulation depth | 7 | 7 | the complete suite: two pure decaying carriers per file encode to a few tens of kilobytes |
| `Annex C.4.2` filter-bank slope | 14 | 14 | the complete suite, for the same reason; this is the criterion that forces an effective skirt attenuation above 41 dB |
| `Annex A.2.2` weighting/redundancy factors | 2 | 6 | the lowest (`125+250`) and the highest-STI (`1000+2000`) band pair |
| `Annex A.3.1.2` filter-bank phase distortion | 1 | 11 | `TI = 0,9`, the worst measured phase bias (-0,0029) inside the normative `TI = 0,1 .. 0,9` range |

The 49 WAV files are 133 MB of 48 kHz 16-bit PCM which does not compress as
audio, so committing the bench as WAV is out of the question. Each signal here
is stored **losslessly**: the *n*-th order difference of its `int16` samples as
little-endian `int32`, LZMA-compressed inside the zip, with `manifest.json`
recording the order, the sample rate, the length and the **SHA-256 of the
original samples**. Decoding is *n* cumulative sums, and the test suite checks
the digest on every load, so what the tests analyse is bit-exact with what
stipa.info publishes. Only the 44-byte RIFF header of each WAV is not
reproduced; the audio samples are.

`make_extract.py` rebuilds the archive from a local copy of the full bench:

```
python tests/data/stipa/make_extract.py tests/data-local/stipa-verification
```

## Source and authorship

- The verification test signals were developed and are published by
  **Embedded Acoustics BV** (Jan Verhave), Delft, the Netherlands, at
  <https://www.stipa.info/index.php/download-test-signals> (retrieved
  2026-07-30), together with the signal description *IEC-60268-16 revision 5
  test signal description*, v1.0, 3 June 2020.
- Terms, quoted from that page: *"All of the above verification test signals
  were developed by Embedded Acoustics and may be used for commercial and
  non-commercial use free of charge."* The same page is explicit that the
  signals are **not** STIPA measurement signals and that the STIPA measurement
  signals sold by instrument manufacturers are separately copyrighted and
  require a paid licence to redistribute; none of those is used here.
- **What that grant does and does not say.** It grants unrestricted use, free
  of charge, for any purpose, and it draws the redistribution line explicitly
  around the *measurement* signals — not around these. It does not, however,
  contain the words "redistribute" or "copy" about the verification signals
  themselves, so this extract is committed on the reading that a free-of-charge
  grant with no restriction attached, for material published precisely so that
  implementations can be verified against it, covers carrying a subset in a
  verification suite. That reading has not been confirmed in writing by
  Embedded Acoustics. If you are Embedded Acoustics and it is wrong, see the
  removal policy below; the extract comes out and the suites fall back to the
  local-only bench.
- The encoded blobs in the zip are a lossless re-encoding of that material, not
  original Embedded Acoustics files; `manifest.json` is ours.

## Purpose and scope of use

Embedded Acoustics publishes the bench so that STIPA implementations can be
verified against the criteria of IEC 60268-16 Annexes A and C. The extract in
this folder serves **exactly that purpose**: it lets
`tests/hearing/test_stipa_certified.py` demonstrate that this library's
independent implementation meets those criteria on every runner, rather than
only on a machine that has the full download. The files are **not** part of the
`phonometry` package, are not installed with it, and are **not covered by this
repository's MIT licence** — the free-of-charge terms above apply to them.

Deliberately excluded: the bundled reference `.m` Matlab sources (the library is
written from the standard, not from them), the original zip archives, and the
signal-description PDF. The 22 bench signals that are not committed are the
intermediate points of the C.3.2 staircase, four of the six A.2.2 band pairs and
ten of the eleven A.3.1.2 phase points; they are the most costly to encode
(nearly full-entropy multi-tone signals) and each is a further point on a curve
whose ends and worst case are already committed. They run automatically when a
full local copy is present.

## Removal policy

If you represent Embedded Acoustics BV and consider that committing this
extract exceeds the intended use of the verification bench, please open an
issue or contact the maintainer (see `CITATION.cff`) and it will be removed
promptly, together with the CI use of the tests that read it.
