---
title: "filters.weighting_compliance"
description: "IEC 61672-1:2013 frequency-weighting class verification."
sidebar:
  label: "weighting_compliance"
---

IEC 61672-1:2013 frequency-weighting class verification.

A/C/Z frequency-weighting acceptance limits transcribed from
BS EN 61672-1:2013, **Table 3** (standard page 22): the design-goal responses
and the class 1 and class 2 upper/lower limits at the 34 nominal frequencies
from 10 Hz to 20 kHz. A lower limit of `-inf` means only the upper limit
applies (subclause 5.5.6 checks measured deviations at the nominal frequencies).

The historical **B weighting** is verified against ANSI S1.4-1983: design
goals from the B column of **Table IV** (whose A and C columns equal IEC
61672-1:2013 Table 3 digit for digit) and tolerance limits from **Table V**,
whose instrument Types 1 and 2 fill the class 1 / class 2 verdict slots (the
stricter laboratory Type 0 mask is exercised by the CI conformance report).
The **AU weighting** is verified against IEC 61012:1990: design goals are the
sum of the nominal A response and the **Table 1** nominal U response (with the
subclause 2.2 explicit AU values at 25/31.5/40 kHz), checked against the
Table 1 tolerances for the filter as a separate unit, the tighter of the two
tolerance readings the standard offers. IEC 61012 publishes a single
tolerance set, so both verdict slots carry the same margin for AU.

One subject: the weighting network a sound level meter applies to the whole
signal, whose acceptance limits qualify the deviation of its measured relative
response from a design goal at the nominal frequencies. The band-filter class
limits of IEC 61260-1, which qualify a relative attenuation against a mask
around each mid-band frequency, live in [`phonometry.filters.compliance`](/phonometry/reference/api/filters/compliance/).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## verify_weighting_class

```python
verify_weighting_class(
    wf: WeightingFilter,
    *,
    sweep_points: int = 4096,
) -> dict[str, Any]
```

Verify a frequency-weighting filter against its standard's tolerances.

`A`/`C`/`Z` are checked against IEC 61672-1:2013 Table 3 (classes 1
and 2). The historical `B` weighting is checked against ANSI S1.4-1983:
Table IV design goals with the Table V tolerance limits, whose instrument
Types 1 and 2 fill the class 1 / class 2 verdict slots (an
`overall_class` of 1 then reads "ANSI S1.4-1983 Type 1"). `AU` is
checked against IEC 61012:1990: design goals are nominal A + nominal U
(Table 1, plus the subclause 2.2 explicit AU values at 25/31.5/40 kHz)
with the single Table 1 tolerance set for the filter as a separate unit,
so both class slots carry the same margin and `overall_class` is 1
(complies) or `None`. `G` is not supported here (ISO 7196 defines one
+/-1 dB instrumentation tolerance, no class structure; the CI conformance
report pins it), nor is `D` (the tolerance tables of the withdrawn
IEC 537 did not survive it; the conformance report pins the D response
against its published transfer function and tabulated curve).

The filter's relative response (normalized to its 1 kHz gain) is evaluated
at the *exact* base-10 frequency behind each nominal label below
the Nyquist frequency (IEC 61672-1 Table 3 NOTE: the design goals are
computed at $f = 1000 \cdot 10^{0.1 (n - 30)}$, e.g.
15 848.9 Hz for
"16 kHz"; IEC 61672-3:2013 subclause 13.3 tests the deviation at the same
exact frequencies, and IEC 61012 Table 1 lists the same exact
frequencies). The deviation from the design-goal weighting is checked
against the two acceptance masks.

A dense logarithmic sweep between the checked frequencies additionally
enforces IEC 61672-1 subclause 5.5.7: at any frequency between two
adjacent nominal frequencies, the deviation of the response from the
analytic design goal (Annex E for A/C/Z, the ANSI S1.4-1983 Appendix C
formulas for B, the A response cascaded with the IEC 61012 Table 2 poles
for AU) must stay within the *larger* of the two adjacent limits. Without
it a resonance or notch between the nominal frequencies would go
unnoticed (for B and AU the sweep is applied as the analogous engineering
check). Both the per-frequency verdicts and the sweep must pass for
`overall_class`. The sweep samples `sweep_points` grid frequencies; a
violation narrower than the grid spacing could in principle fall between
samples, so raise `sweep_points` for higher-Q suspects (the verdict
attests the sampled grid, not a continuous proof).

The response is taken over the whole path a signal travels through
[`filter`](/phonometry/reference/api/filters/weighting/#weightingfilterfilter), not over the designed
second-order sections alone: with `high_accuracy` the sections are
reached through an interpolation and a decimation stage, and the
anti-alias filter of those stages has its transition band on the input
Nyquist frequency, so it dominates the response above roughly
`0.9 * fs / 2`. Whenever the highest checked row falls there (the
8 kHz row of a 16 kHz system, the 16 kHz row of a 32 kHz one) a verdict
read from the sections alone would attest a filter the user never runs.
The response is still computed in closed form (the L folded spectral
images of the cascade), so it stays exact and deterministic. The `Z`
weighting is a flat bypass and always complies.

When rows that carry a *finite lower* acceptance limit fall at or
above the Nyquist frequency (e.g. the 8-16 kHz class 1 rows of a 16 kHz
sampled system, or the 25-40 kHz AU rows of a 48 kHz one), they cannot be
checked and `range_limited` is `True`: the returned class then
attests conformance over the checked frequencies only, not conformance
over the standard's full frequency range.

**Parameters**

| Name | Description |
| :--- | :--- |
| `wf` | The weighting filter to verify (`A`, `B`, `C`, `AU` or `Z`). |
| `sweep_points` | Number of points of the 5.5.7 between-nominals sweep (>= 64). |

**Returns:** Dict with `overall_class` (1, 2 or None), `range_limited` (see above), `bands`: a list of `{"freq", "class", "deviation_db", "margin_class1_db", "margin_class2_db"}` where `freq` is the nominal label, a positive margin means the limits are met with that much room, and `between_nominals`: `{"worst_freq", "margin_class1_db", "margin_class2_db"}` for the sweep.

## weighting_class_limits

```python
weighting_class_limits(
    weighting_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

IEC 61672-1:2013 Table 3 acceptance limits for a performance class.

The limits apply to every IEC 61672-1 weighting (A, C, Z); they qualify
the deviation of the measured relative response from the design goal at
each nominal frequency, not the response itself. (The B and AU masks that
`verify_weighting_class` uses come from ANSI S1.4-1983 Table V and
IEC 61012:1990 Table 1 instead and are not returned here.)

**Parameters**

| Name | Description |
| :--- | :--- |
| `weighting_class` | 1 or 2 (IEC 61672-1:2013 performance class). |

**Returns:** Tuple `(frequencies, lower, upper)` of the 34 nominal frequencies (Hz) and the lower/upper deviation limits in dB. A lower limit of `-inf` means only the upper limit applies.
