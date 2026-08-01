---
title: "room.open_plan"
description: "Open-plan-office spatial metrics per ISO 3382-3:2012."
sidebar:
  label: "open_plan"
---

Open-plan-office spatial metrics per ISO 3382-3:2012.

From a line of measurement positions across workstations (ISO 3382-3:2012,
5.2.2) this module derives the single-number quantities of Clause 4:

- the spatial decay rate of the A-weighted sound pressure level of speech
  `D2,S` and the nominal A-weighted speech level at 4 m `Lp,A,S,4m`
  (Clause 6.2), obtained by a least-squares fit of the A-weighted speech
  level against $\log_{10}(r/r_0)$ (logarithmic distance axis,
  $r_0 = 1$ m)
  using only positions in the 2 m to 16 m range (Equation (5)); and
- the distraction distance `rD` (STI = 0.50) and privacy distance
  `rP` (STI = 0.20) (Clause 3.6, 3.7, 6.3), obtained from a linear
  regression of the speech transmission index against distance on a
  linear axis.

The A-weighted speech levels `Lp,A,S,n` (Clause 6.2, Equation (4)) and
the per-position STI (full IEC 60268-16 method, spatially averaged
background noise per Clause 6.3) are taken as inputs; this module performs
only the regressions and threshold read-offs of Clause 6.

The distraction and privacy distances are read from the fitted STI line,
extrapolating beyond the measured range when necessary (the regression-line
method of Clause 6.3, Figure 3 b). A distance is reported as `nan` when
the STI does not decrease with distance (non-negative fitted slope) or when
the crossing would fall at or before the source, realising the standard's
note that it "can prove impossible to determine the privacy distance if
STI > 0.20 in all positions" (Clause 6.3).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## open_plan_metrics

```python
open_plan_metrics(
    positions_m: list[float] | np.ndarray,
    spl_a_speech: list[float] | np.ndarray,
    sti_values: list[float] | np.ndarray,
) -> OpenPlanResult
```

Open-plan-office single-number quantities per ISO 3382-3:2012.

Computes the spatial decay rate `D2,S` and the nominal A-weighted
speech level at 4 m `Lp,A,S,4m` (Clause 6.2, Equation (5)) from a
least-squares fit of the A-weighted speech level against
$\log_{10}(r/r_0)$ ($r_0 = 1$ m) restricted to positions in the
2 m to 16 m range, and
the distraction distance `rD` (STI = 0.50) and privacy distance
`rP` (STI = 0.20) from a linear regression of STI against distance
(Clause 3.6, 3.7, 6.3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `positions_m` | Distances `r` from the source to each measurement position, in metres (ISO 3382-3:2012, 5.2.3 d). |
| `spl_a_speech` | A-weighted speech level `Lp,A,S,n` at each position, in dB (Clause 6.2, Equation (4)). |
| `sti_values` | Speech transmission index at each position (full IEC 60268-16 method with spatially averaged background noise, Clause 6.3). |

**Returns:** [`OpenPlanResult`](/phonometry/reference/api/rooms/open-plan/#openplanresult) with `d2s`, `lp_as_4m`, `rd` and `rp`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If fewer than four positions are given (Clause 5.2.2) or the three arrays differ in length. |

## OpenPlanResult

```python
OpenPlanResult(
    d2s: float,
    lp_as_4m: float,
    rd: float,
    rp: float,
    positions_m: np.ndarray | None = None,
)
```

Single-number open-plan-office quantities (ISO 3382-3:2012, Cl. 4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `d2s` | Spatial decay rate of the A-weighted SPL of speech in dB (per distance doubling), from the least-squares fit of Clause 6.2, Equation (5). `nan` when fewer than two positions lie in the 2 m to 16 m range. |
| `lp_as_4m` | Nominal A-weighted speech level at 4 m in dB, read off the same regression line (Clause 3.3, 6.2). `nan` under the same condition as `d2s`. |
| `rd` | Distraction distance in m, where the linear STI-vs-distance regression crosses 0.50 (Clause 3.6, 6.3). `nan` when the fitted STI does not decrease with distance or the crossing is non-positive. |
| `rp` | Privacy distance in m, where the same regression crosses 0.20 (Clause 3.7, 6.3), possibly extrapolated beyond the measured range. `nan` under the same condition as `rd`. |
| `positions_m` | The microphone distances the metrics were fitted on, in metres, retained so `plot_geometry` can draw the line; `None` for hand-built results. |

### OpenPlanResult.plot()

```python
OpenPlanResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the spatial decay of speech with `rD`/`rP` marked.

Redraws the Clause 6.2 regression line from `d2s` and
`lp_as_4m` and marks the distraction and privacy distances.
Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### OpenPlanResult.plot_geometry()

```python
OpenPlanResult.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the measurement line to scale, rD and rP marked.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

### OpenPlanResult.report()

```python
OpenPlanResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an open-plan office acoustics fiche to a PDF (ISO 3382-3).

Writes a one-page report laid out like an open-plan-office
speech-privacy measurement report: the standard-basis line, an optional
metadata header block (client, office/zone, description, floor area,
source and measurement positions, climate ...), a compact metrics table
of the four single-number quantities of Clause 4 (the spatial decay
rate `D2,S`, the nominal 4 m A-weighted speech level `Lp,A,S,4m`,
the distraction distance `rD` and the privacy distance `rP`), the
result's own spatial-decay plot (`plot`, the Clause 6.2
regression on the logarithmic distance axis with the 4 m read-off and
the `rD` / `rP` crossings), the boxed `D2,S`, an optional verdict
row and a footer with the fixed disclaimer.

ISO 3382-3 characterises a space rather than defining an intrinsic
pass/fail, so the verdict row appears only when a target spatial decay
rate is supplied through `metadata.requirement` (read as the minimum
acceptable `D2,S` in dB, reflecting the informative quality ranges of
Annex A where a larger spatial decay is better; the room passes at or
above it).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare characterisation fiche (metrics, plot and result only). The open-plan-specific fields `area` (floor area), `source_positions` and `receiver_positions` (the number of measurement positions) populate the header, alongside `client`, `test_room` (the office or zone), `specimen` (the description and furnishing state), `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `measurement_standard`, `test_date`, `laboratory`, `operator`, `report_id` and `notes`; `requirement` is read as the minimum acceptable `D2,S`. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for parity with the other fiches; the fiche has a single stacked body layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab or, for the embedded spatial-decay chart, matplotlib is not installed. The fiche embeds the result's own plot whenever the regression is defined, so both are required (`pip install "phonometry[report,plot]"`). |

## plot_open_plan_geometry

```python
plot_open_plan_geometry(
    positions: ArrayLike,
    ax: Axes | None = None,
    *,
    rd: float | None = None,
    rp: float | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the open-plan measurement line to scale.

Source at the origin, the microphone line across the workstations, and
the distraction and privacy distances marked on the axis when given.

**Parameters**

| Name | Description |
| :--- | :--- |
| `positions` | Microphone distances from the source, in metres (1-D). |
| `ax` | Existing axes, or `None` to create a figure. |
| `rd` | Distraction distance, in metres, or `None`. |
| `rp` | Privacy distance, in metres, or `None`. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the microphone scatter. |

**Returns:** The axes.
