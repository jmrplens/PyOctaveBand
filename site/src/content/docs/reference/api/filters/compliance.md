---
title: "metrology.compliance"
description: "IEC 61260-1:2014 filter and IEC 61672-1:2013 weighting class verification."
sidebar:
  label: "compliance"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

IEC 61260-1:2014 filter and IEC 61672-1:2013 weighting class verification.

**Filters.** Acceptance limits on relative attenuation transcribed from the
official text (BS EN 61260-1:2014, **Table 1**, standard pages 15-16):
octave-band breakpoint frequencies with class 1 and class 2 minimum/maximum
limits. Fractional-octave-band breakpoints are derived with Formulas (9) and
(10) (subclauses 5.10.3-5.10.4) and limits between breakpoints are interpolated
linearly in lg(Omega) per Formula (11) (subclause 5.10.6). Relative attenuation
is `deltaA(Omega) = A(Omega) - Aref` (Formula 8) with `A = Lin - Lout`
(Formula 7); here `Aref` is the attenuation at the exact mid-band frequency
(subclause 5.9: the pass-band reference attenuation).

IEC 61260-1:2014 defines only classes 1 and 2. **Class 0** (the tightest,
laboratory-grade class) lives only in the withdrawn **IEC 61260:1995 /
EN 61260:1995 Table 1** and its US twin **ANSI S1.11-2004 Table 1**, whose
class 1/2 masks differ numerically from the 2014 edition (e.g. the 2014
pass-band reference tolerance is ±0.4 dB for class 1 vs ±0.3 dB in 1995, and
the 2014 stop-band edge minimum is +1.2 dB vs +2.0 dB in 1995). The two editions
are therefore kept as separate mask tables selected by the `edition` argument
(`"2014"` default -> classes 1/2; `"1995"` -> classes 0/1/2). The 1995 /
ANSI-2004 octave-band table was transcribed digit-for-digit and cross-checked
between the two standards (they agree exactly).

**Weightings.** A/C/Z frequency-weighting acceptance limits transcribed from
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

## class_limits

```python
class_limits(
    fraction: float,
    filter_class: int,
    omega: np.ndarray,
    *,
    edition: str = '2014',
) -> tuple[np.ndarray, np.ndarray]
```

Acceptance limits on relative attenuation at normalized frequencies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `fraction` | Bandwidth designator denominator b (1 for octave, 3 for one-third octave, ...). |
| `filter_class` | Performance class: 1 or 2 for `edition="2014"`; 0, 1 or 2 for `edition="1995"`. |
| `omega` | Normalized frequencies f/fm (> 0). |
| `edition` | `"2014"` (IEC 61260-1:2014, classes 1/2) or `"1995"` (IEC 61260:1995 / ANSI S1.11-2004, classes 0/1/2). |

**Returns:** Tuple (minimum, maximum) relative attenuation in dB per point; the maximum is `+inf` outside the pass-band.

:::note
The exact band-edge point `Omega = G^(1/2)` is treated as pass-band.
The 1995 edition's Table 1 prints a dedicated minimum (+2.3/+2.0/
+1.6 dB) *at* that single frequency, which this convention relaxes to
the pass-band minimum; the discrepancy has measure zero -- any
continuous response violating the edge row is caught at `edge + eps`
by the interpolated stop-band mask. The 2014 edition defines only the
`G^(1/2) - eps` and `G^(1/2) + eps` rows, which the masks match
exactly.
:::

## filter_class_compliance

```python
filter_class_compliance(
    bank: OctaveFilterBank,
    *,
    num_points: int = 32768,
    edition: str = '2014',
) -> FilterComplianceResult
```

Verify a filter bank and package the verdict as a reportable result.

Runs [`verify_filter_class`](/phonometry/reference/api/filters/compliance/#verify_filter_class) and stores the outcome together with the
bank's second-order sections, mid-band frequencies, per-band decimation
factors and sampling rate, so the returned object can redraw the measured
relative attenuation and render an accredited `.report()` fiche without
keeping a reference to the bank.

**Parameters**

| Name | Description |
| :--- | :--- |
| `bank` | The filter bank to verify. |
| `num_points` | Frequency grid points per band (>= 16). |
| `edition` | `"2014"` (IEC 61260-1:2014, classes 1/2) or `"1995"` (IEC 61260:1995 / ANSI S1.11-2004, adds the stricter class 0). |

**Returns:** A [`FilterComplianceResult`](/phonometry/reference/api/filters/compliance/#filtercomplianceresult).

## FilterComplianceResult

```python
FilterComplianceResult(
    overall_class: int | None,
    bands: tuple[dict[str, Any], ...],
    fraction: int,
    edition: str,
    sos: tuple[np.ndarray, ...],
    band_frequencies: np.ndarray,
    factors: tuple[int, ...],
    fs: float,
    num_points: int,
    range_limited: bool = False,
)
```

IEC 61260-1 class-compliance verdict of an [`OctaveFilterBank`](/phonometry/reference/api/filters/core/#octavefilterbank).

Wraps the dictionary of [`verify_filter_class`](/phonometry/reference/api/filters/compliance/#verify_filter_class) together with the
minimal filter-bank data needed to redraw the measured relative-attenuation
curve, so the result exposes the standard `plot` / `report` pair without
holding a reference to the (possibly stateful) bank.

**Attributes**

| Name | Description |
| :--- | :--- |
| `overall_class` | The strictest class every band meets (0/1/2), or `None` when at least one band meets no class of the edition. |
| `bands` | The per-band verdict dictionaries of [`verify_filter_class`](/phonometry/reference/api/filters/compliance/#verify_filter_class) (one `{"freq", "class", "margin_class<c>_db", ...}` per band), as an immutable tuple. |
| `fraction` | Bandwidth designator `b` (1 for octave, 3 for one-third-octave). |
| `edition` | `"2014"` (IEC 61260-1:2014, classes 1/2) or `"1995"` (IEC 61260:1995 / ANSI S1.11-2004, classes 0/1/2). |
| `sos` | Per-band second-order sections of the analysed bank (one array per band), kept so the relative attenuation can be recomputed with `scipy.signal.sosfreqz` exactly as the verifier does. |
| `band_frequencies` | The exact mid-band frequencies `f_m` in Hz. |
| `factors` | Per-band decimation factor; the band's processing sample rate is `fs / factor` (the multirate rate the SOS were designed at). Stored because the response must be evaluated at that decimated rate, which the verifier's public return does not expose. |
| `fs` | The bank's full sampling rate in Hz. |
| `num_points` | Frequency grid points per band used by the verification, retained so the redrawn curve matches the analysed grid. |
| `range_limited` | `True` when at least one band's stop-band mask extends beyond its processing Nyquist frequency, so the verification could not exercise the full Table 1 mask there (the multirate anti-aliasing removes signal energy beyond it, but the limits are not demonstrated); the stated class then attests the verified frequency range and the `.report()` fiche prints a qualifying note. |

### FilterComplianceResult.available_classes()

```python
FilterComplianceResult.available_classes() -> list[int]
```

The performance classes carried by the per-band verdict dictionaries.

Reads the `margin_class<n>_db` keys of a band verdict, so it reflects
the edition (the 1995 edition adds class 0; the 2014 edition keeps only
classes 1 and 2). An empty result (a bank with no bands in range)
carries no verdicts, so this returns an empty list.

### FilterComplianceResult.plot()

```python
FilterComplianceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the worst-margin band against its class-limit corridor.

Draws the measured relative attenuation of the binding band over the
acceptance corridor of the achieved (or, when non-compliant, the
loosest) class; see `phonometry._plot.metrology.plot_filter_class`.
Requires matplotlib (`pip install phonometry[plot]`) and returns the
`Axes`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `language` | Label language, `"en"` (default) or `"es"`. |

### FilterComplianceResult.reference_class()

```python
FilterComplianceResult.reference_class() -> int
```

The class whose corridor the fiche/plot overlays.

The achieved overall class when the bank complies, else the loosest
class of the edition (the one it comes closest to meeting).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result carries no bands, so there is no reference class to report. |

### FilterComplianceResult.report()

```python
FilterComplianceResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an IEC 61260-1 filter-class-compliance fiche to a PDF.

Writes a one-page accredited report: the standard-basis line, an
optional metadata header block, a per-band classification table beside
the mask-overlay plot (the result's own `plot`), the boxed
class-compliance result, an optional verdict row against a supplied
`required_class` and a footer with the fixed disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a prediction fiche (body, result and disclaimer only). A supplied `required_class` drives the verdict row. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform signature; it has no effect on the single-layout filter-compliance fiche. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## verify_aircraft_noise_system

```python
verify_aircraft_noise_system(
    *,
    directional: dict[float, dict[float, float]] | None = None,
    frequency_response: dict[float, float] | None = None,
    linearity: dict[str, float] | None = None,
    resolution: float | None = None,
) -> dict[str, Any]
```

Verify measured performance against IEC 61265:1995 tolerances.

Each supplied measurement is checked against the standard's limit; the
one-third-octave filtering itself is covered by the IEC 61260 class-2
verification (subclause 4.6) and is not repeated here.

**Parameters**

| Name | Description |
| :--- | :--- |
| `directional` | Microphone directional response as `{frequency_hz: {angle_deg: \|Δsensitivity\| dB}}` (Table 1, §4.4.2). |
| `frequency_response` | System response deviations `{frequency_hz: deviation_db}` against the ±1.5 dB limit (§4.5.1). |
| `linearity` | Level non-linearity `{"reference": dB, "other": dB}` against the ±0.4/±0.5 dB limits (§4.5.2). |
| `resolution` | Readout resolution, in dB, against the 0.1 dB limit (§4.7). |

**Returns:** `{"passed": bool, "checks": [{"quantity", "limit", "value", "ok", ...}]}`; `passed` is the conjunction of every check.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a frequency or angle is out of the tabulated range. |

## verify_filter_class

```python
verify_filter_class(
    bank: OctaveFilterBank,
    num_points: int = 32768,
    *,
    edition: str = '2014',
) -> dict[str, Any]
```

Verify a filter bank against the IEC 61260 class limits.

Each band's relative attenuation (referenced to the attenuation at its
exact mid-band frequency) is checked against every acceptance-limit class of
the selected edition's Table 1, evaluated on a dense frequency grid up to
the band's processing Nyquist. The Table 1 breakpoint frequencies inside
that range are always included in the evaluation, so the pass-band
constraints are checked even if the grid were coarse. Frequencies beyond
the processing Nyquist cannot carry signal energy at the band's decimated
rate (the multirate anti-aliasing filter removes them), so they are
treated as compliant; because the Table 1 limits there are nevertheless
not demonstrated, the returned `range_limited` flag is set whenever a
band's stop-band mask extends beyond its processing Nyquist, and the
per-band `checked_to_omega` records how far the check reached.

**Parameters**

| Name | Description |
| :--- | :--- |
| `bank` | The filter bank to verify (its designed SOS are analyzed; works for stateful and stateless banks alike). |
| `num_points` | Number of frequency grid points per band (>= 16). |
| `edition` | `"2014"` (IEC 61260-1:2014, classes 1/2) or `"1995"` (IEC 61260:1995 / ANSI S1.11-2004, adds the stricter class 0). |

**Returns:** Dict with `overall_class` (the strictest class every band meets, or `None`), `range_limited` (`True` when at least one band's stop-band mask extends beyond its processing Nyquist, so the returned class attests the verified frequency range rather than the full Table 1 mask; see above) and `bands`: a list of `{"freq", "class", "checked_to_omega", "margin_class<c>_db"}` for each class `c` of the edition, where a positive margin means the limits are met with that much room and `checked_to_omega` is the highest normalized frequency the band's verification could reach (its processing Nyquist over `f_m`).

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
computed at `f = 1000 * 10^(0.1 (n - 30))`, e.g. 15 848.9 Hz for
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

The response is taken from the designed second-order sections (evaluated
with `sosfreqz` at their design rate), so it is exact and deterministic;
it does not model the runtime resampling stages that `high_accuracy`
adds around them, whose anti-alias response is flat across the audio band
checked here. The `Z` weighting is a flat bypass and always complies.

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
