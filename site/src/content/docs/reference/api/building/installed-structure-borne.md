---
title: "building.installed_structure_borne"
description: "Public API of phonometry.building.installed_structure_borne (auto-generated)."
sidebar:
  label: "installed_structure_borne"
---

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

Installed structure-borne sound from service equipment (EN 12354-5:2009).

EN 12354-5 predicts the sound pressure level in a receiving room caused by
building service equipment that injects **structure-borne sound** into the
building. The chain closes the structural-vibroacoustics series:

1. The source strength is its *characteristic structure-borne sound power level*
   `L_Ws,c`. It is **not** the raw reception-plate power of EN 15657
   Formula (14): that plate-injected level must first be converted to the
   plate-independent `L_Ws,n` (EN 15657 Formulae (15)/(17); see
   [`phonometry.building.structure_borne_power`](/phonometry/reference/api/building/structure-borne-power/)) and then referred to the
   actual receiver with the Annex I mobility correction
   ([`installed_power_from_reception_plate`](/phonometry/reference/api/building/installed-structure-borne/#installed_power_from_reception_plate)),
   `L_Ws,inst,i = L_Ws,n + 10 lg(Y_inf,i / Y_inf,rec)` with the reference
   plate mobility `Y_inf,rec = 5e-6 m/(N.s)`, or equivalently to the
   characteristic level `L_Ws,c = L_Ws,n + 10 lg(Y_s / Y_inf,rec)` with the
   source mobility (Annex I.3, Table I.8), from which `D_C` is subtracted.
2. Only part of that power is actually injected into the supporting element; the
   loss is the **coupling term** `D_C` (clause 4.4.3), positive in the usual
   mobility-mismatched cases (see [`coupling_term`](/phonometry/reference/api/building/installed-structure-borne/#coupling_term) for the exception),
   set by the source mobility `Y_s` and the receiver mobility `Y_i`
   (Formula 19b):
   `D_C,i = 10 lg( |Y_s + Y_i|**2 / (|Y_s| Re{Y_i}) )`, which reduces to
   `10 lg(|Y_s|/Re{Y_i})` for a force source (high source mobility,
   Formula 19c) and to `-10 lg(|Y_s| Re{Z_i})` for a velocity source (low
   source mobility, Formula 19d). An elastic support adds its transfer
   mobility `Y_k` inside the modulus (Formula 19e).
3. The **installed** power level is then `L_Ws,inst,i = L_Ws,c - D_C,i`
   (Formula 18b).
4. The normalised sound pressure level in the receiving room for one path (i->j)
   follows from the installed power, the structure-to-airborne adjustment term
   `D_sa` (clause 4.4.4), the flanking sound reduction index `R_ij,ref` and
   the element area (Formula 18a):
   `L_n,s,ij = L_Ws,inst,i - D_sa,i - R_ij,ref - 10 lg(S_i/S0) - 10 lg(A0/4)`
   with `S0 = A0 = 10 m2`; the paths combine energetically (Formula 17).

The source and receiver mobilities/impedances are those of
[`phonometry.mechanical_mobility`](/phonometry/reference/api/vibration/mechanical-mobility/) and [`phonometry.transfer_stiffness`](/phonometry/reference/api/vibration/transfer-stiffness/).

## coupling_term

```python
coupling_term(
    source_mobility: ArrayLike,
    receiver_mobility: ArrayLike,
    *,
    transfer_mobility: ArrayLike = 0.0,
) -> np.ndarray
```

Coupling term `D_C` for a point excitation (EN 12354-5, Formula 19b/19e).

`D_C = 10 lg( |Y_s + Y_i + Y_k|**2 / (|Y_s| Re{Y_i}) )` -- the loss between
the characteristic and the injected structure-borne power. `Y_k` is the
transfer mobility of an elastic support (Formula 19e; 0 for a rigid
connection, Formula 19b).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_mobility` | Source point mobility `Y_s` (complex, non-zero), in m/(N.s). |
| `receiver_mobility` | Receiver point mobility `Y_i` (complex, positive real part). |
| `transfer_mobility` | Elastic-support transfer mobility `Y_k` (Default: 0.0). |

**Returns:** The coupling term `D_C`, in dB. Positive whenever the source and receiver mobilities are well mismatched (the usual installed case), but **not** guaranteed non-negative: near a mounting resonance where `Y_s` and `Y_i` are of comparable magnitude and opposite phase the numerator `|Y_s + Y_i|²` collapses and `D_C` goes negative (the installed power then exceeds the characteristic level; e.g. `Y_s = j·1e-4`, `Y_i = 1e-5 − j·1e-4` m/(N·s) gives `D_C ≈ −10 dB`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `Y_s` is zero/non-finite or `Re{Y_i}` is not positive and finite. |

## coupling_term_force_source

```python
coupling_term_force_source(
    source_mobility: ArrayLike,
    receiver_mobility: ArrayLike,
) -> np.ndarray
```

Coupling term for a force source, high source mobility (Formula 19c).

`D_C = 10 lg(|Y_s| / Re{Y_i})`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_mobility` | Source point mobility `Y_s` (complex, non-zero). |
| `receiver_mobility` | Receiver point mobility `Y_i` (complex, positive real part). |

**Returns:** The coupling term `D_C`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `Y_s` is zero/non-finite or `Re{Y_i}` is not positive and finite. |

## coupling_term_velocity_source

```python
coupling_term_velocity_source(
    source_mobility: ArrayLike,
    receiver_impedance: ArrayLike,
) -> np.ndarray
```

Coupling term for a velocity source, low source mobility (Formula 19d).

`D_C = -10 lg(|Y_s| Re{Z_i})`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_mobility` | Source point mobility `Y_s` (complex, non-zero). |
| `receiver_impedance` | Receiver point impedance `Z_i` (complex, positive real part). |

**Returns:** The coupling term `D_C`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `Y_s` is zero/non-finite or `Re{Z_i}` is not positive and finite. |

## installed_power_from_reception_plate

```python
installed_power_from_reception_plate(
    reception_plate_level: ArrayLike,
    receiver_mobility: ArrayLike,
    *,
    plate_mobility: float = 5e-06,
) -> np.ndarray
```

Mobility correction of the reception-plate power (EN 12354-5, Annex I).

`L_Ws,inst,i = L_Ws,n,i + 10 lg(Y_inf,i / Y_inf,rec)`, which refers the
characteristic reception-plate power level `L_Ws,n` (EN 15657
Formula (17), re the 10 cm concrete plate `Y_inf,rec = 5e-6 m/(N.s)`)
to the characteristic mobility `Y_inf,i` of the actual receiving
element (floor, wall), yielding the installed power of that element as in
the Annex I.2 whirlpool example. The same correction with the *source*
mobility instead of `Y_inf,i` yields the characteristic level
`L_Ws,c` (Annex I.3, Table I.8), from which
[`installed_structure_borne_power_level`](/phonometry/reference/api/building/installed-structure-borne/#installed_structure_borne_power_level) subtracts `D_C`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reception_plate_level` | Power level to re-refer (per band), in dB re 1 pW: either the characteristic level `L_Ws,n` (EN 15657 Formula 17, referred to the default 5e-6 m/(N.s) plate) or a raw Formula (14) plate power together with the mobility of the plate it was measured on, passed as `plate_mobility`. |
| `receiver_mobility` | Characteristic mobility `Y_inf,i` of the receiving element (per band; complex values use their magnitude), in m/(N.s). |
| `plate_mobility` | Mobility the input level is referred to (Default: the EN 15657 reference plate, `Y_inf,rec = 5e-6 m/(N.s)`; pass the measured plate mobility when the input is a raw Formula (14) level). |

**Returns:** The mobility-corrected power level, in dB re 1 pW.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive receiver or plate mobility. |

## installed_source_prediction

```python
installed_source_prediction(
    characteristic_power_level: ArrayLike,
    coupling_term: ArrayLike,
    paths: list[dict[str, Any]],
    *,
    frequencies: ArrayLike | None = None,
) -> InstalledSourceResult
```

Predict the installed structure-borne SPL over several paths (EN 12354-5).

The band count is set by the widest per-band input (the source levels or
any path's `adjustment_term` / `flanking_reduction_index`); every
per-band input must carry one value or that count, and single values
broadcast across the bands (a single-number source level with per-band
path data is valid, and the result's `installed_power_level` is
broadcast to the band count).

**Parameters**

| Name | Description |
| :--- | :--- |
| `characteristic_power_level` | Characteristic level `L_Ws,c` (per band or a single value), in dB. |
| `coupling_term` | Coupling term `D_C` (per band or a single value), in dB. |
| `paths` | One dict per transmission path with keys `adjustment_term` (`D_sa`), `flanking_reduction_index` (`R_ij,ref`) and `element_area` (`S_i`), each per band where applicable. |
| `frequencies` | Band centre frequencies, in hertz, or `None`. |

**Returns:** The [`InstalledSourceResult`](/phonometry/reference/api/building/installed-structure-borne/#installedsourceresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | if `paths` is empty, a path is missing a required key, or a per-band input matches neither one value nor the band count. |

## installed_structure_borne_power_level

```python
installed_structure_borne_power_level(
    characteristic_power_level: ArrayLike,
    coupling_term: ArrayLike,
) -> np.ndarray
```

Installed structure-borne power level (EN 12354-5, Formula 18b).

`L_Ws,inst,i = L_Ws,c - D_C,i`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `characteristic_power_level` | Characteristic level `L_Ws,c` (per band), in dB: the EN 15657 reception-plate level converted with Formulae (15)/(17) and the source-mobility correction (see the module docstring), **not** the raw plate-injected Formula (14) level. |
| `coupling_term` | Coupling term `D_C,i` (per band), in dB. |

**Returns:** The installed structure-borne power level `L_Ws,inst`, in dB.

## InstalledSourceResult

```python
InstalledSourceResult(
    path_levels: np.ndarray,
    total_level: np.ndarray,
    installed_power_level: np.ndarray,
    frequencies: np.ndarray | None = None,
)
```

Installed structure-borne sound prediction (EN 12354-5).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in hertz, or `None`. |
| `path_levels` | Per-path normalised SPL `L_n,s,ij` (paths x bands), dB. |
| `total_level` | Combined normalised SPL `L_n,s` per band, in dB. |
| `installed_power_level` | Installed power level `L_Ws,inst` per band, dB. |

### InstalledSourceResult.overall_level

*property*

Band-summed total level `10 lg(sum 10^(0.1 L_n,s))`, in dB.

### InstalledSourceResult.plot()

```python
InstalledSourceResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-path and total normalised sound pressure levels.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### InstalledSourceResult.report()

```python
InstalledSourceResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an EN 12354-5 installed structure-borne prediction fiche.

Writes a one-page **prediction** sheet (an estimate, not a
measurement): a prediction-basis line naming EN 12354-5:2009, an
optional metadata header (client, source equipment, receiving room,
instrumentation, climate, date), a per-band table (nominal
octave/one-third-octave frequency, the installed structure-borne power
level `L_Ws,inst`, each transmission path's normalised SPL
`L_n,s,ij` and the combined total `L_n,s`), the per-path and total
`L_n,s(f)` spectra, the boxed band-summed total `L_n,s` (dB) with
the installed power total and the path count, an optional verdict row
against a declared limit, and a basis strip stating Formulae 18a/17 and
the prediction disclaimer.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the source equipment, `test_room` the receiving room, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared upper limit on the overall `L_n,s` (lower is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds one column per transmission path (up to five); otherwise only the installed power and the combined total are shown. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## REFERENCE_AREA

*Constant* (`float`).

```python
REFERENCE_AREA = 10.0
```

## structure_borne_pressure_level_path

```python
structure_borne_pressure_level_path(
    installed_power_level: ArrayLike,
    adjustment_term: ArrayLike,
    flanking_reduction_index: ArrayLike,
    element_area: float,
    *,
    reference_area: float = 10.0,
) -> np.ndarray
```

Normalised structure-borne SPL for one path i->j (Formula 18a).

`L_n,s,ij = L_Ws,inst,i - D_sa,i - R_ij,ref - 10 lg(S_i/S0) - 10 lg(A0/4)`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `installed_power_level` | Installed power level `L_Ws,inst,i`, in dB. |
| `adjustment_term` | Structure-to-airborne adjustment `D_sa,i` (clause 4.4.4 / Annex F), in dB. |
| `flanking_reduction_index` | Flanking sound reduction index `R_ij,ref` re `S0` (EN 12354-1), in dB. |
| `element_area` | Supporting-element area `S_i`, in m^2 (> 0). |
| `reference_area` | Reference area `S0 = A0` (Default: 10 m^2). |

**Returns:** The normalised path sound pressure level `L_n,s,ij`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive area. |

## total_structure_borne_pressure_level

```python
total_structure_borne_pressure_level(path_levels: ArrayLike) -> np.ndarray
```

Combine path sound pressure levels energetically (Formula 17).

`L_n,s = 10 lg( sum_j 10^(L_n,s,ij/10) )`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path_levels` | Path levels `L_n,s,ij`; sum is over the first axis (paths), broadcasting any trailing band axis. |

**Returns:** The total normalised sound pressure level `L_n,s`, in dB.
