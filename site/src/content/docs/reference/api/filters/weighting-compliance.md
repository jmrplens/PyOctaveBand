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

IEC 61672-1:2013 defines only classes 1 and 2. **Type 0**, the tightest of the
four instrument types, lives in the superseded **IEC 651:1979 Table V**, held
here as the identical British adoption BS 5969:1981. Its four masks differ
numerically from the 2013 edition (e.g. Type 0 is +2/-3 dB at both 16 and
20 kHz where class 1 is +2.5/-16 and +3/-inf), so the two editions are kept as
separate mask tables selected by the `edition` argument (`"2013"` default
-> classes 1/2; `"1979"` -> Types 0/1/2/3, offered as classes 0-3).

The historical **B weighting** is verified against ANSI S1.4-1983: design
goals from the B column of **Table IV** (whose A and C columns equal IEC
61672-1:2013 Table 3 digit for digit) and tolerance limits from **Table V**,
whose instrument Types 1 and 2 fill the class 1 / class 2 verdict slots. The
ANSI Type 0 column is a *different* mask from the IEC 651 one - two-sided and
stricter at 10/12.5/16 Hz where IEC 651 is upper-only - so the two are carried
as the two editions they are and never merged.
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
    edition: str = '2013',
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

`edition="1979"` swaps in the tolerance table of the superseded
IEC 651:1979 instead, whose **Table V** publishes the laboratory-grade
**Type 0** mask that IEC 61672-1 has no equivalent for, and three further
types. Class N is then the standard's instrument Type N, so an
`overall_class` of 0 reads "IEC 651:1979 Type 0". That edition covers
`A`, `B` and `C` (the weightings of subclause 3.2, whose Table IV
design goals equal the ones used above digit for digit): its Table V
footnote makes one mask govern every weighting characteristic, so `B`
does not borrow the ANSI limits there. It is a genuinely different mask,
not a rename - Type 0 is +2/-3 dB at 16 kHz and at 20 kHz, where class 1
is +2.5/-16 and +3/-inf, so an error class 1 cannot see is visible under
Type 0.

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
unnoticed (for B, for AU and under the 1979 edition, whose tables are
tabulated at the nominal frequencies and nowhere else, the sweep is
applied as the analogous engineering check). Both the per-frequency
verdicts and the sweep must pass for
`overall_class`. The sweep samples `sweep_points` grid frequencies; a
violation narrower than the grid spacing could in principle fall between
samples, so raise `sweep_points` for higher-Q suspects (the verdict
attests the sampled grid, not a continuous proof).

The response is taken over the whole path a signal travels through
[`filter`](/phonometry/reference/api/filters/weighting/#weightingfilterfilter), which is one cascade of
second-order sections at the input rate for every curve and in both
stateful and single-shot use. It used to be more than that: the sections
were reached through an interpolation and a decimation stage whose
anti-alias filter had its transition band on the input Nyquist frequency
and dominated the response above roughly `0.9 * fs / 2`, so a verdict
read from the sections alone attested a filter the user never ran. That is
why the verdict is measured through `_runtime_frequency_response`
rather than through `sosfreqz` here, and it stays that way so the next
stage added to the path cannot go unmodelled. The `Z` weighting is a
flat bypass and always complies.

When rows that carry a *finite lower* acceptance limit fall at or
above the Nyquist frequency (e.g. the 8-16 kHz class 1 rows of a 16 kHz
sampled system, or the 25-40 kHz AU rows of a 48 kHz one), they cannot be
checked and `range_limited` is `True`: the returned class then
attests conformance over the checked frequencies only, not conformance
over the standard's full frequency range.

**Parameters**

| Name | Description |
| :--- | :--- |
| `wf` | The weighting filter to verify (`A`, `B`, `C`, `AU` or `Z`; `A`, `B` or `C` for `edition="1979"`). |
| `sweep_points` | Number of points of the 5.5.7 between-nominals sweep (>= 64). |
| `edition` | `"2013"` (IEC 61672-1:2013, classes 1/2) or `"1979"` (IEC 651:1979, Types 0/1/2/3 offered as classes 0-3). |

**Returns:** Dict with `overall_class` (the strictest class of the edition that every checked frequency and the sweep meet, or `None`), `range_limited` (see above), `bands`: a list of `{"freq", "class", "deviation_db", "margin_class<c>_db"}` for each class `c` of the edition, where `freq` is the nominal label and a positive margin means the limits are met with that much room, and `between_nominals`: `{"worst_freq", "margin_class<c>_db"}` for the sweep.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the edition is unknown, the edition does not define the filter's curve, or `sweep_points` is below 64. |

## weighting_class_limits

```python
weighting_class_limits(
    weighting_class: int,
    *,
    edition: str = '2013',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Acceptance limits of one performance class of a weighting standard.

The limits apply to every weighting the edition defines; they qualify the
deviation of the measured relative response from the design goal at each
nominal frequency, not the response itself. Under `edition="2013"` they
come from IEC 61672-1:2013 Table 3 and govern A, C and Z (the B and AU
masks that `verify_weighting_class` uses come from ANSI S1.4-1983
Table V and IEC 61012:1990 Table 1 instead and are not returned here).
Under `edition="1979"` they come from IEC 651:1979 Table V, whose
footnote makes one mask govern every weighting characteristic, B included.

**Parameters**

| Name | Description |
| :--- | :--- |
| `weighting_class` | Performance class: 1 or 2 for `edition="2013"`; 0, 1, 2 or 3 for `edition="1979"`, where class N is the standard's instrument Type N. |
| `edition` | `"2013"` (IEC 61672-1:2013, classes 1/2) or `"1979"` (IEC 651:1979, which adds the stricter Type 0 and a Type 3). |

**Returns:** Tuple `(frequencies, lower, upper)` of the 34 nominal frequencies (Hz) and the lower/upper deviation limits in dB. A lower limit of `-inf` means only the upper limit applies.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if the edition is unknown or does not define the requested class. |
