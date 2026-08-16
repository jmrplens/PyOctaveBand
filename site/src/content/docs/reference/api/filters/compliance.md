---
title: "filters.compliance"
description: "IEC 61260-1:2014 band-filter class verification."
sidebar:
  label: "compliance"
---

IEC 61260-1:2014 band-filter class verification.

Acceptance limits on relative attenuation transcribed from the
official text (BS EN 61260-1:2014, **Table 1**, standard pages 15-16):
octave-band breakpoint frequencies with class 1 and class 2 minimum/maximum
limits. Fractional-octave-band breakpoints are derived with Formulas (9) and
(10) (subclauses 5.10.3-5.10.4) and limits between breakpoints are interpolated
linearly in $\log_{10} \Omega$ per Formula (11) (subclause 5.10.6).
Relative attenuation is
$\Delta A(\Omega) = A(\Omega) - A_{\mathrm{ref}}$ (Formula 8) with
$A = L_{\mathrm{in}} - L_{\mathrm{out}}$
(Formula 7); here $A_{\mathrm{ref}}$ is the attenuation at the exact
mid-band frequency
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

One subject: the class limits of a band filter, a mask around each mid-band
frequency the filter's own relative attenuation is measured against. The
acceptance limits of the A/B/C/AU/Z frequency weightings, which qualify a
network applied to the whole signal against a design-goal response, live in
[`phonometry.filters.weighting_compliance`](/phonometry/reference/api/filters/weighting-compliance/).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

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
| `omega` | Normalized frequencies $f/f_\mathrm{m}$ (> 0). |
| `edition` | `"2014"` (IEC 61260-1:2014, classes 1/2) or `"1995"` (IEC 61260:1995 / ANSI S1.11-2004, classes 0/1/2). |

**Returns:** Tuple (minimum, maximum) relative attenuation in dB per point; the maximum is `+inf` outside the pass-band.

:::note
The exact band-edge point $\Omega = G^{1/2}$ is treated as
pass-band.
The 1995 edition's Table 1 prints a dedicated minimum (+2.3/+2.0/
+1.6 dB) *at* that single frequency, which this convention relaxes to
the pass-band minimum; the discrepancy has measure zero -- any
continuous response violating the edge row is caught at
$\text{edge} + \epsilon$
by the interpolated stop-band mask. The 2014 edition defines only
the $G^{1/2} - \epsilon$ and $G^{1/2} + \epsilon$
rows, which the masks match
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
loosest) class; see `phonometry._plot.filters.plot_filter_class`.
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
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

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
