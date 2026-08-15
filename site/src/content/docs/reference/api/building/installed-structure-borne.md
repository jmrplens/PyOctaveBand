---
title: "building.prediction.installed_structure_borne"
description: "Installed structure-borne sound from service equipment (EN 12354-5:2009)."
sidebar:
  label: "installed_structure_borne"
---

Installed structure-borne sound from service equipment (EN 12354-5:2009).

EN 12354-5 predicts the sound pressure level in a receiving room caused by
building service equipment that injects **structure-borne sound** into the
building. The chain closes the structural-vibroacoustics series:

1. The source strength is its *characteristic structure-borne sound power level*
   `L_Ws,c`. It is **not** the raw reception-plate power of EN 15657
   Formula (14): that plate-injected level must first be converted to the
   plate-independent `L_Ws,n` (EN 15657 Formulae (15)/(17); see
   [`phonometry.building.measurement.structure_borne_power`](/phonometry/reference/api/building/structure-borne-power/)) and then referred to the
   actual receiver with the Annex I mobility correction
   ([`installed_power_from_reception_plate`](/phonometry/reference/api/building/installed-structure-borne/#installed_power_from_reception_plate)),
   $L_{W\mathrm{s,inst},i} = L_{W\mathrm{s,n}} + 10 \log_{10}( Y_{\infty,i} / Y_{\infty,\mathrm{rec}} )$
   with the reference plate mobility
   $Y_{\infty,\mathrm{rec}} = 5 \cdot 10^{-6}$ m/(N.s), or equivalently to the
   characteristic level
   $L_{W\mathrm{s,c}} = L_{W\mathrm{s,n}} + 10 \log_{10}( Y_\mathrm{s} / Y_{\infty,\mathrm{rec}} )$ with the
   source mobility (Annex I.3, Table I.8), from which `D_C` is subtracted.
2. Only part of that power is actually injected into the supporting element; the
   loss is the **coupling term** `D_C` (clause 4.4.3), positive in the usual
   mobility-mismatched cases (see [`coupling_term`](/phonometry/reference/api/building/installed-structure-borne/#coupling_term) for the exception),
   set by the source mobility `Y_s` and the receiver mobility `Y_i`
   (Formula 19b):
   $D_{\mathrm{C},i} = 10 \log_{10}\left( |Y_\mathrm{s} + Y_i|^2 / (|Y_\mathrm{s}| \operatorname{Re}\{Y_i\}) \right)$, which reduces to
   $10 \log_{10}( |Y_\mathrm{s}| / \operatorname{Re}\{Y_i\} )$ for a force source
   (high source mobility,
   Formula 19c) and to $-10 \log_{10}( |Y_\mathrm{s}| \operatorname{Re}\{Z_i\} )$
   for a velocity source (low
   source mobility, Formula 19d). An elastic support adds its transfer
   mobility `Y_k` inside the modulus (Formula 19e).
3. The **installed** power level is then
   $L_{W\mathrm{s,inst},i} = L_{W\mathrm{s,c}} - D_{\mathrm{C},i}$
   (Formula 18b).
4. The normalised sound pressure level in the receiving room for one path (i->j)
   follows from the installed power, the structure-to-airborne adjustment term
   `D_sa` (clause 4.4.4), the flanking sound reduction index `R_ij,ref` and
   the element area (Formula 18a):
   $L_{\mathrm{n,s},ij} = L_{W\mathrm{s,inst},i} - D_{\mathrm{sa},i} - R_{ij,\mathrm{ref}} - 10 \log_{10}(S_i/S_0) - 10 \log_{10}(A_0/4)$
   with $S_0 = A_0 = 10$ m²; the paths combine energetically
   (Formula 17).

The source and receiver mobilities/impedances are those of
[`phonometry.vibration.structural.mechanical_mobility`](/phonometry/reference/api/vibration/mechanical-mobility/) and [`phonometry.vibration.structural.transfer_stiffness`](/phonometry/reference/api/vibration/transfer-stiffness/).

**The informative tables.** Every term of that chain is a number the user
would otherwise copy out of the standard, so the two tables of the informative
annexes are here as named lookups:

- **Table D.1** (Annex D) estimates the mobility of typical construction
  elements from their own dimensions, which is how clause D.1.3 builds up a
  *source* mobility `Y_s` for step 2 out of the machine's mass, feet,
  chassis panels and pipework: [`typical_element_mobility`](/phonometry/reference/api/building/installed-structure-borne/#typical_element_mobility), whose
  `structure` argument is the table's first column and whose keywords are
  its second ([`TABLE_D1_QUANTITIES`](/phonometry/reference/api/building/installed-structure-borne/#table_d1_quantities)).
- **Table F.1** (Annex F) gives the octave-band force level `L_F` of the ISO
  tapping machine, the substitution source of clause D.1.2.3:
  [`tapping_machine_force_level`](/phonometry/reference/api/building/installed-structure-borne/#tapping_machine_force_level), with
  [`tapping_machine_characteristic_power_level`](/phonometry/reference/api/building/installed-structure-borne/#tapping_machine_characteristic_power_level) and
  [`tapping_machine_coupling_term`](/phonometry/reference/api/building/installed-structure-borne/#tapping_machine_coupling_term) turning it into the `L_Ws,c` and
  `D_C` of step 3.

Annex F also supplies the two terms step 4 takes: the adjustment term `D_sa`
of Formula (F.3) ([`structure_to_airborne_adjustment`](/phonometry/reference/api/building/installed-structure-borne/#structure_to_airborne_adjustment)) and the
multi-junction adjustment `dK` of clause F.1
([`multi_junction_adjustment`](/phonometry/reference/api/building/installed-structure-borne/#multi_junction_adjustment)), which the flanking reduction index
`R_ij,ref` of a path more than one junction away is built with.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

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

$D_\mathrm{C} = 10 \log_{10}\left( |Y_\mathrm{s} + Y_i + Y_k|^2 / (|Y_\mathrm{s}| \operatorname{Re}\{Y_i\}) \right)$ -- the loss between
the characteristic and the injected structure-borne power. `Y_k` is the
transfer mobility of an elastic support (Formula 19e; 0 for a rigid
connection, Formula 19b).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_mobility` | Source point mobility `Y_s` (complex, non-zero), in m/(N.s). |
| `receiver_mobility` | Receiver point mobility `Y_i` (complex, positive real part). |
| `transfer_mobility` | Elastic-support transfer mobility `Y_k` (Default: 0.0). |

**Returns:** The coupling term `D_C`, in dB. Positive whenever the source and receiver mobilities are well mismatched (the usual installed case), but **not** guaranteed non-negative: near a mounting resonance where `Y_s` and `Y_i` are of comparable magnitude and opposite phase the numerator $\lvert Y_\mathrm{s} + Y_i \rvert^2$ collapses and `D_C` goes negative (the installed power then exceeds the characteristic level; e.g. $Y_\mathrm{s} = j \cdot 10^{-4}$, $Y_i = 10^{-5} - j \cdot 10^{-4}$ m/(N·s) gives $D_\mathrm{C} \approx -10$ dB).

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

$$
D_\mathrm{C} = 10 \log_{10}\frac{|Y_\mathrm{s}|}{\operatorname{Re}\{Y_i\}}
$$

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

$$
D_\mathrm{C} = -10 \log_{10}\left( |Y_\mathrm{s}| \operatorname{Re}\{Z_i\} \right)
$$

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

$L_{W\mathrm{s,inst},i} = L_{W\mathrm{s,n},i} + 10 \log_{10}( Y_{\infty,i} / Y_{\infty,\mathrm{rec}} )$, which refers the
characteristic reception-plate power level `L_Ws,n` (EN 15657
Formula (17), re the 10 cm concrete plate
$Y_{\infty,\mathrm{rec}} = 5 \cdot 10^{-6}$ m/(N.s))
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
| `plate_mobility` | Mobility the input level is referred to (Default: the EN 15657 reference plate, $Y_{\infty,\mathrm{rec}} = 5 \cdot 10^{-6}$ m/(N.s); pass the measured plate mobility when the input is a raw Formula (14) level). |

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

The band count is set by the widest per-band input (the characteristic
power level, the `coupling_term` or any path's `adjustment_term` /
`flanking_reduction_index`); every
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

$$
L_{W\mathrm{s,inst},i} = L_{W\mathrm{s,c}} - D_{\mathrm{C},i}
$$

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

Band-summed total level $10 \log_{10}(\sum 10^{0.1 L_\mathrm{n,s}})$,
in dB.

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

## MINIMUM_MULTI_JUNCTION_KIJ

*Constant* (`float`).

```python
MINIMUM_MULTI_JUNCTION_KIJ = -5.0
```

## multi_junction_adjustment

```python
multi_junction_adjustment(junctions: int) -> float
```

Vibration reduction index adjustment `dK` (EN 12354-5, clause F.1).

When the receiving room is more than one junction away from the equipment,
Formula (F.1) sums the junction `Kij` along the path and subtracts an
adjustment `dK` that covers the transmission by wave types other than
bending waves. Clause F.1 estimates it from published data as 4 dB for two
junctions and 6 dB for three or more, with the resulting `Kij` floored at
[`MINIMUM_MULTI_JUNCTION_KIJ`](/phonometry/reference/api/building/installed-structure-borne/#minimum_multi_junction_kij).

**Parameters**

| Name | Description |
| :--- | :--- |
| `junctions` | Number of junctions the transmission path crosses (>= 1). |

**Returns:** The adjustment `dK`, in dB (0,0 for a single junction).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for fewer than one junction. |

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

$$
L_{\mathrm{n,s},ij} = L_{W\mathrm{s,inst},i} - D_{\mathrm{sa},i} - R_{ij,\mathrm{ref}} - 10 \log_{10}\frac{S_i}{S_0} - 10 \log_{10}\frac{A_0}{4}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `installed_power_level` | Installed power level `L_Ws,inst,i`, in dB. |
| `adjustment_term` | Structure-to-airborne adjustment `D_sa,i` (clause 4.4.4 / Annex F), in dB. |
| `flanking_reduction_index` | Flanking sound reduction index `R_ij,ref` re `S0` (EN 12354-1), in dB. |
| `element_area` | Supporting-element area `S_i`, in m^2 (> 0). |
| `reference_area` | Reference area $S_0 = A_0$ (Default: 10 m^2). |

**Returns:** The normalised path sound pressure level `L_n,s,ij`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive area. |

## structure_to_airborne_adjustment

```python
structure_to_airborne_adjustment(
    frequency: ArrayLike,
    critical_frequency: float,
    mass_per_area: float,
    *,
    radiation_factor: ArrayLike = 1.0,
) -> np.ndarray
```

Adjustment term `D_sa` of a supporting element (Formula F.3).

$$
D_{\mathrm{sa},i} = 10 \log_{10} \frac{400 f_{\mathrm{c},i} \sigma_i}{m_i f^2}
$$

the ratio of injected structure-borne power to incident airborne power that
leaves the same free-vibration energy in the element. Clause F.2 gives it
for a force excitation perpendicular to a homogeneous supporting element,
exact above the critical frequency (where the radiation factor saturates at
1) and a good approximation over the whole range.

`D_sa` is normally **negative**, and Formula (18a) subtracts it, so it
raises the predicted level: this is the value
[`structure_borne_pressure_level_path`](/phonometry/reference/api/building/installed-structure-borne/#structure_borne_pressure_level_path) takes as `adjustment_term`,
sign included.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequency `f`, in hertz (> 0). |
| `critical_frequency` | Critical frequency `fc,i` of the element, in hertz (> 0). |
| `mass_per_area` | Mass per unit area `mi` of the element, in kg/m^2 (> 0). |
| `radiation_factor` | Radiation factor `sigma_i` of the element (Default: 1.0, its value above `fc`; EN 12354-1:2000 Annex B estimates it below). |

**Returns:** The adjustment term `D_sa,i`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive frequency, critical frequency, mass or radiation factor. |

## TABLE_D1_QUANTITIES

*Constant* (`dict`).

```python
TABLE_D1_QUANTITIES = {'mass': ('mass',), 'bar_end': ('density', 'longitudinal_velocity', 'area'), 'beam': ('density', 'longitudinal_velocity', 'thickness', 'width'), 'plate': ('density', 'longitudinal_velocity', 'thickness'), 'pipe': ('density', 'longitudinal_velocity', 'thickness', 'radius'), 'mass_spring': ('mass', 'stiffness', 'loss_factor')}
```

## TABLE_F1_FORCE_LEVEL

*Constant* (`tuple`).

```python
TABLE_F1_FORCE_LEVEL = (139.0, 142.0, 145.0, 148.0, 151.0, 154.0, 156.0, 156.0)
```

## TABLE_F1_OCTAVE_BANDS

*Constant* (`tuple`).

```python
TABLE_F1_OCTAVE_BANDS = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
```

## tapping_machine_characteristic_power_level

```python
tapping_machine_characteristic_power_level(
    frequency: ArrayLike,
    force_level: ArrayLike,
) -> np.ndarray
```

Characteristic power level of the tapping machine (Formula D.9a).

$$
L_{W\mathrm{s,c}} = L_F - 5 - 10 \log_{10} f
$$

The standard notes the result is about 115 dB re 1 pW per one-third octave
for the ISO tapping machine, treated in clause D.1.3 as a force source with
the mass-like source mobility of its 0,5 kg hammers. Pair it with
[`tapping_machine_coupling_term`](/phonometry/reference/api/building/installed-structure-borne/#tapping_machine_coupling_term) and hand both to
[`installed_source_prediction`](/phonometry/reference/api/building/installed-structure-borne/#installed_source_prediction).

The formula only balances with the reference force `F_0 = 1e-6` N, since
it carries no term for $F_0^2 / W_0$; that is what pins the reading
of Table F.1 against its own printed "re 1 pN" caption.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequency `f`, in hertz (> 0). |
| `force_level` | Force level `L_F`, in dB re 1e-6 N (Table F.1 or [`tapping_machine_force_level_estimate`](/phonometry/reference/api/building/installed-structure-borne/#tapping_machine_force_level_estimate)). |

**Returns:** The characteristic power level `L_Ws,c`, in dB re 1 pW.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive frequency. |

## tapping_machine_coupling_term

```python
tapping_machine_coupling_term(
    frequency: ArrayLike,
    receiver_mobility: ArrayLike,
    *,
    hammer_mass: float = 0.5,
) -> np.ndarray
```

Coupling term of the tapping machine (EN 12354-5, Formula D.9b).

$$
D_{\mathrm{C},i} = -10 \log_{10}(\omega M Y_i) + 10 \log_{10}\left[ 1 + (\omega M Y_i)^2 \right]
$$

with $\omega = 2 \pi f$ and `M` the hammer mass of clause D.1.3,
which the standard takes as 0,5 kg. It is the mass-like-source form of
Formula (19b), for a machine standing on a plate-like element of real
mobility `Y_i`; Annex F Formulae (F.4) to (F.6b) estimate that `Y_i`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequency `f`, in hertz (> 0). |
| `receiver_mobility` | Real mobility `Y_i` of the supporting element, in m/(N.s) (> 0). |
| `hammer_mass` | Source mass `M`, in kilograms (Default: [`TAPPING_HAMMER_MASS`](/phonometry/reference/api/building/resilient-layers/#tapping_hammer_mass), the 0,5 kg of clause D.1.3 and ISO 10140-5). |

**Returns:** The coupling term `D_C,i`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive frequency, mobility or mass. |

## tapping_machine_force_level

```python
tapping_machine_force_level() -> np.ndarray
```

Tabulated ISO tapping machine force level (EN 12354-5, Table F.1).

The eight octave-band values of Table F.1, in the order of
[`TABLE_F1_OCTAVE_BANDS`](/phonometry/reference/api/building/installed-structure-borne/#table_f1_octave_bands) (31,5 Hz to 4 kHz). Clause F.4.2
offers them for the tapping machine used in place of an electrodynamic
shaker when measuring `D_Fp,n` (Formula F.9), and clause D.1.2.3
restricts the source to low-mobility receiving structures.

Feed them to [`tapping_machine_characteristic_power_level`](/phonometry/reference/api/building/installed-structure-borne/#tapping_machine_characteristic_power_level) for the
`L_Ws,c` that [`installed_source_prediction`](/phonometry/reference/api/building/installed-structure-borne/#installed_source_prediction) takes.

**Returns:** The force level `L_F` per octave band, in dB re 1e-6 N (the table caption prints "re 1 pN"; see `docs/ERRATA.md`).

## tapping_machine_force_level_estimate

```python
tapping_machine_force_level_estimate(
    frequency: ArrayLike,
    *,
    bandwidth: str = 'octave',
) -> np.ndarray
```

Closed form printed under Table F.1 for the tapping machine force level.

$$
L_F = 10 \log_{10} \frac{2{,}5 f}{10^{-12}} \quad\text{(octave)}, \qquad L_F = 10 \log_{10} \frac{0{,}8 f}{10^{-12}} \quad\text{(1/3 octave)}
$$

The standard qualifies this with "up till about 1000 Hz": it reproduces the
first six tabulated values to the printed decibel, and above that it
departs from the table, which
flattens at 156 dB (the closed form gives 157 dB at 2 kHz and 160 dB at
4 kHz). Use [`tapping_machine_force_level`](/phonometry/reference/api/building/installed-structure-borne/#tapping_machine_force_level) for the octave bands the
table covers and this only where it does not, chiefly the one-third-octave
bands the standard does not tabulate.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Band centre frequency `f`, in hertz (> 0). |
| `bandwidth` | `"octave"` (coefficient 2,5) or `"third"` (coefficient 0,8). |

**Returns:** The force level `L_F`, in dB re 1e-6 N (the standard prints "re 1 pN"; see `docs/ERRATA.md`).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive frequency or an unknown bandwidth. |

## total_structure_borne_pressure_level

```python
total_structure_borne_pressure_level(path_levels: ArrayLike) -> np.ndarray
```

Combine path sound pressure levels energetically (Formula 17).

$$
L_\mathrm{n,s} = 10 \log_{10}\!\left( \sum_j 10^{L_{\mathrm{n,s},ij}/10} \right)
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `path_levels` | Path levels `L_n,s,ij`; sum is over the first axis (paths), broadcasting any trailing band axis. |

**Returns:** The total normalised sound pressure level `L_n,s`, in dB.

## typical_element_mobility

```python
typical_element_mobility(
    structure: str,
    *,
    frequency: ArrayLike | None = None,
    mass: float | None = None,
    density: float | None = None,
    longitudinal_velocity: float | None = None,
    thickness: float | None = None,
    width: float | None = None,
    area: float | None = None,
    radius: float | None = None,
    stiffness: float | None = None,
    loss_factor: float | None = None,
) -> np.ndarray
```

Named lookup of EN 12354-5, Table D.1 (mobility of typical elements).

Returns the mobility magnitude `|Y|` the third column of Table D.1
prints, in m/(N.s), for the row named by *structure*:

- `"mass"`, described by `mass` $M$ [kg]:
  $\lvert Y \rvert = \left[ 2 \pi f M \right]^{-1}$.
- `"bar_end"`, described by `density` $\rho$ [kg/m3],
  `longitudinal_velocity` $c_\mathrm{L}$ [m/s] and `area` $S$
  [m2]: $\lvert Y \rvert = \left[ \rho c_\mathrm{L} S \right]^{-1}$.
- `"beam"`, described by $\rho$, $c_\mathrm{L}$, `thickness`
  $t$ [m] and `width` $w$ [m]:
  $\lvert Y \rvert = \left[ 7{,}6\, \rho t w \sqrt{c_\mathrm{L} t f} \right]^{-1}$.
- `"plate"`, described by $\rho$, $c_\mathrm{L}$ and $t$:
  $\lvert Y \rvert = \left[ 2{,}3\, c_\mathrm{L} \rho t^2 \right]^{-1}$.
- `"pipe"`, described by $\rho$, $c_\mathrm{L}$, $t$ and
  `radius` $r$ [m]:
  $\lvert Y \rvert = \left[ 63\, \rho t r \sqrt{c_\mathrm{L} r f} \right]^{-1}$.
- `"mass_spring"`, described by $M$, `stiffness` $s$
  [N/m] and `loss_factor` $\eta$ [-]:
  $\lvert Y \rvert = \left[ \left( \frac{2 \pi f \eta}{s (1 + \eta^2)} \right)^2 + \left( \frac{2 \pi f}{s (1 + \eta^2)} - \frac{1}{2 \pi f M} \right)^2 \right]^{1/2}$.

[`TABLE_D1_QUANTITIES`](/phonometry/reference/api/building/installed-structure-borne/#table_d1_quantities) carries the "Describing quantities" column as
the keyword names of this function, and only the quantities a row describes
may be supplied.
Frequency is not among them, because it is the band frequency of the
prediction and not a property of the element, but it appears in four of the
six expressions: `"mass"`, `"beam"`, `"pipe"` and `"mass_spring"`
require `frequency` and the other two reject it.

Clause D.1.3 offers the table for building up a **source** mobility `Y_s`
from the machine's own parts, which is what [`coupling_term`](/phonometry/reference/api/building/installed-structure-borne/#coupling_term) needs and
what a measurement of the equipment does not give. Annex F Formulae (F.4)
to (F.6b) cover the *receiver* mobility `Y_i` of the supporting element;
the `"plate"` row is the same quantity as Formula (F.4)
$Y_{i,\infty} = 1 / (8\sqrt{m B'})$ and as
[`phonometry.vibration.structural.point_mobility.infinite_plate_mobility`](/phonometry/reference/api/vibration/point-mobility/#infinite_plate_mobility),
written in $\rho$, $c_\mathrm{L}$ and $t$ instead of mass and
bending stiffness.

The `"mass_spring"` row is the machine on non-rigid feet: its second
bracket holds the two reactances, which cancel at the mass-spring
resonance $f_0 = (2\pi)^{-1}\sqrt{s(1 + \eta^2)/M}$ and leave the
mobility at its damping-limited minimum, the frequency at which the mount
injects the most power. `loss_factor=0` returns exactly zero there,
which [`coupling_term`](/phonometry/reference/api/building/installed-structure-borne/#coupling_term) then rejects.

**Parameters**

| Name | Description |
| :--- | :--- |
| `structure` | Table D.1 row name. |
| `frequency` | Frequency `f`, in hertz, for the frequency-dependent rows only. |
| `mass` | Mass `M`, in kilograms (rows `"mass"`, `"mass_spring"`). |
| `density` | Density `rho`, in kg/m^3. |
| `longitudinal_velocity` | Quasi-longitudinal wave speed `cL`, in m/s. |
| `thickness` | Thickness `t`, in metres. |
| `width` | Beam width `w`, in metres (row `"beam"`). |
| `area` | Cross-sectional area `S`, in m^2 (row `"bar_end"`). |
| `radius` | Pipe radius `r`, in metres (row `"pipe"`). |
| `stiffness` | Support stiffness `s`, in N/m (row `"mass_spring"`). |
| `loss_factor` | Support loss factor `eta` (row `"mass_spring"`). |

**Returns:** The mobility magnitude `|Y|`, in m/(N.s).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown row, a missing or surplus describing quantity, a `frequency` that the row does not take (or lacks), or a non-positive value. |
