---
title: "Same bits on every machine"
description: "Why floating point drifts between hosts, what phonometry pins so a designed filter is the same filter byte for byte on any CPU, BLAS or thread count, what still separates the platforms, and the measured evidence behind the claim."
---

A weighting filter designed by phonometry is the same filter, byte for byte,
on every machine that shares a C library: any CPU, with or without AVX-512;
any BLAS kernel set; any thread count; any hash seed. Across C libraries it
is not, yet: Linux, macOS and Windows each design their own bytes for the
same filter, because a handful of transcendentals still come from the
platform's `math` once per design, and this page names them and measures
what they do. On one platform the coefficients a laptop designs are the
coefficients a cluster designs, so a conformance verdict, a published figure
or a filter shipped inside a measurement chain is a reproducible artifact
rather than one machine's account of itself.

That sentence is not how scientific Python normally behaves, and this page
records what it cost, with the numbers.

## Why floating point drifts between hosts

IEEE 754 pins every *single* operation: given two doubles, their sum,
difference, product, quotient and square root are the same bit pattern on
every conforming machine. Everything beyond a single operation is where
reproducibility leaks, through three separate doors.

**Reductions are free to reassociate.** A sum of many terms has no defined
order, and a BLAS picks the order that suits the vector registers it finds at
run time. phonometry measured what that costs on its own weighting fit: two
OpenBLAS kernel sets agreed on every accept decision of the
Levenberg&ndash;Marquardt search for thirty-nine consecutive steps while the
costs they were comparing drifted from 3&nbsp;&times;&nbsp;10⁻¹³ to
1&nbsp;&times;&nbsp;10⁻⁴ apart, and the two runs then landed on **different
filters**: 5&nbsp;&times;&nbsp;10⁻⁵ apart in the leading coefficient,
0.002&nbsp;dB apart in response, a visible 0.0225&nbsp;pt apart in a plotted
curve. An optimiser in a shallow valley amplifies whatever the last bit does.

**Vectorised transcendentals are dispatched.** numpy chooses `log`, `exp`,
`tan`, `power`, `arctan` and `sin` kernels by what the CPU offers, and the
AVX-512 kernels do not agree to the last bit with the ones a machine without
AVX-512 runs. Measured under Intel SDE emulating Skylake-X against the same
host natively: every one of those six returns a different digest over the
same inputs. The leak reaches further than the obvious call sites:
`numpy.geomspace` is `10 ** linspace(...)`, so the frequency grid every
residual of the fit is evaluated on came out different under AVX-512, and
every step downstream inherited it.

**The platform's own libm is a platform choice.** `math.log` is whatever
`log` the C library ships. glibc's and Apple's are both accurate to about
half an ulp, which means that on a handful of inputs they legitimately round
to *opposite* neighbours, so a design loop built on `math.log` returned
different filters on macOS than on Linux with nothing anywhere to say so.
The same holds for every other transcendental the C library ships, which is
why the evidence below still finds three sets of bytes.

## What phonometry pins

The deterministic design path answers each door in kind, in
`filters._weighting_design` and `filters._pinned_log`:

- **Its own reductions.** Every sum the fit compares or solves with is folded
  pairwise in a shape fixed by the code, and the normal equations are solved
  by Gaussian elimination written out, the same algorithm LAPACK's `dgesv`
  runs with the order no longer a library's choice. Pairwise folding is also
  the *more accurate* order: O(&epsilon;&nbsp;log&nbsp;n) rounding against
  the O(&epsilon;&nbsp;n) of a running total, so nothing was traded away.
- **Real arithmetic where complex arithmetic dispatches.** Complex multiply
  and complex magnitude fuse operations differently per CPU, so the two
  places the path handed numpy a complex expression are written out in real
  arithmetic.
- **Pinned transcendentals.** Outside the iteration, each transcendental is
  taken from the C library once per design. Those few calls, an `exp`, a
  `tan`, an `atan`, a `pow`, a `log10` or two and a `sin` or two, are the
  one place a platform's libm still enters a design, and they are enough to
  separate the platforms: the evidence below has the three digests.
  Spelling them in pinned numpy the way the logarithm is spelled is the
  remaining step to one digest everywhere. Inside the iteration, the
  logarithm runs
  three quarters of a million times per design, and the loop that once pinned
  it cost a factor of four; it is now glibc's own table-driven `log`
  algorithm spelled in numpy operations IEEE&nbsp;754 fixes exactly, plain
  arithmetic, integer bit work and table lookups, with the fused
  multiply-adds of the original reproduced through exact error-free
  transformations. Its bits are its own on every machine, which is stronger
  than calling anyone's libm.
- **The grid spelled out.** The geometric frequency grid is generated term
  for term with the endpoints pinned, so the fit evaluates the same
  frequencies everywhere.

## The evidence

Every claim above is measured, and most of them are re-measured by tests that
run in CI:

- **91,658,333 inputs, zero exceptions.** Every distinct value the whole
  design corpus feeds its logarithm was captured and compared bit for bit
  against glibc's `log`: identical on all of them, plus roughly a hundred
  million adversarial draws over the full exponent range, the near-one band,
  the subnormals and the branch edges. An adversarial search also found the
  honest limit: about one input in thirty million, in one narrow band, whose
  true logarithm sits within half an ulp of both neighbours and where the
  two implementations part by the last bit. None of those can reach a
  design, because on the design path the pinned routine *is* the definition.
- **The corpus is byte-identical.** The designs of seven curves at nineteen
  sampling rates, 133 in all, hash to the same single SHA-256 before and
  after the logarithm was vectorised: no shipped coefficient, figure or
  conformance value moved.
- **Digest for digest across environments.** The designs are bit-identical
  between this host and the same host under AVX-512 emulation, under every
  selectable OpenBLAS kernel set, every thread count and every hash seed;
  the environment-versus-environment comparisons are pinned as tests.
- **Each platform its own bytes, measured.** With the logarithm pinned, a
  design no longer depends on which `log` the C library ships, but the
  once-per-design calls still do, and they are enough: the A weighting at
  48&nbsp;kHz hashes to `991833ff389afe91` on Linux (glibc, x86-64),
  `4efa1818bd80e23a` on macOS (arm64) and `79852a57e60961c8` on Windows
  (x86-64), the same value under Python 3.13 and 3.14 on each. The three
  digests are pinned as a test in the CI matrix, so a platform that drifts
  from its own value turns CI red. The response tests that run on the same
  three platforms hold the A design within 0.013&nbsp;dB of its analog
  prototype everywhere in its fit band, which bounds what the three byte
  strings can differ by in response.

## See it yourself

Two designs are the same design, and the bytes have a name:

```python
import hashlib
from phonometry import filters

once = filters.WeightingFilter(48000, "A").sos
again = filters.WeightingFilter(48000, "A").sos
print(once.tobytes() == again.tobytes())                 # True
digest = hashlib.sha256(once.tobytes()).hexdigest()
print(digest[:16])                                       # 991833ff389afe91 on Linux
```

The second line is the point: those sixteen hex digits, the head of the
SHA-256, are not "what my machine got", they are the digest of this release's
A weighting at 48&nbsp;kHz on any glibc Linux machine on x86-64, the leg the
CI matrix pins, and the evidence above gives the macOS and Windows values to
expect. If a future release moves a
coefficient deliberately, the digest moves with it and the change is a
documented event, never a property of your hardware.

## Scope

The guarantee covers the deterministic weighting design path end to end on
one platform, and block processing composes with it: a stateful cascade fed
block by block is bit-identical to one pass over the whole record, so
streaming does not spend the guarantee. General analysis functions built on numpy and SciPy in their
ordinary form remain reproducible on one machine but may differ in the last
bits across CPUs, which is the normal condition of scientific Python; the
deterministic treatment is applied where a filter's identity matters.

The full engineering account lives with the code, in the module docstrings of
[`filters/_weighting_design.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/filters/_weighting_design.py)
and
[`filters/_pinned_log.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/filters/_pinned_log.py),
and the tests that hold it are in
[`tests/filters/test_weighting_design.py`](https://github.com/jmrplens/phonometry/blob/main/tests/filters/test_weighting_design.py)
and
[`tests/filters/test_pinned_log.py`](https://github.com/jmrplens/phonometry/blob/main/tests/filters/test_pinned_log.py).

## See also

- [Conformance report](/phonometry/reference/conformance/) — what the numbers are checked against; this page is why the checks read the same everywhere.
- [Errata in published sources](/phonometry/reference/errata/) — the other half of the evidence story: where the printed expected value is the thing that is wrong.
- [Frequency weightings](/phonometry/signals/levels/weighting/) — the guide to the curves the deterministic design realises.
- [Block processing](/phonometry/signals/filters/block-processing/) — the streaming identity the scope section leans on.
