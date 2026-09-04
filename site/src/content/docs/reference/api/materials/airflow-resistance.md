---
title: "materials.absorbers.airflow_resistance"
description: "Airflow resistance of porous materials: ISO 9053-1 and ISO 9053-2."
sidebar:
  label: "airflow_resistance"
---

Airflow resistance of porous materials: ISO 9053-1 and ISO 9053-2.

Two standardised measurement methods share the same three quantities and units
(ISO 9053-1:2018, Clause 3; ISO 9053-2:2020, Clause 3):

- **Airflow resistance** $R = \Delta p / q_v$ in Pa\*s/m3, with
  $\Delta p$ the air pressure difference across the specimen (Pa) and
  `q_v` the volumetric airflow rate through it (m3/s)
  (ISO 9053-1:2018, 3.1).
- **Specific airflow resistance** $R_\mathrm{s} = R\,A$ in **Pa\*s/m** (not
  Pa\*s/m2), with `A` the cross-sectional area of the specimen perpendicular
  to the flow (m2) (ISO 9053-1:2018, 3.2). Equivalently
  $R_\mathrm{s} = \Delta p / u$ with `u` the linear airflow velocity, since
  $u = q_v / A$.
- **Airflow resistivity** $\sigma = R_\mathrm{s} / d$ in Pa\*s/m2, with `d` the
  specimen thickness in the flow direction (m), for homogeneous materials
  (ISO 9053-1:2018, 3.3). Equivalently $\sigma = R\,A / d$.

The linear airflow velocity is $u = q_v / A$ (ISO 9053-1:2018, 3.4).

**Static (DC) method, ISO 9053-1:2018.** A steady unidirectional flow in the
laminar regime is used. The recommended reference linear airflow velocity is
$u = 0.5\times 10^{-3}$ m/s (0.5 mm/s, clause 7.5); if measured stepwise
the highest velocity shall not exceed `15e-3 m/s` (15 mm/s), beyond which
the flow may be non-linear. When measured stepwise the pressure difference is
plotted against `u` and fitted with a regression of at least second order
constrained through the origin, $\Delta p = a u + b u^2$;
$\Delta p$ and `R_s` are then evaluated at
$u = 0.5\times 10^{-3}$ m/s (clause 7.5). Because
$R_\mathrm{s} = \Delta p / u = a + b u$, the linear coefficient `a` is the
zero-velocity specific airflow resistance.

**Alternating (AC) method, ISO 9053-2:2020.** A sinusoidally moving piston
(frequency 1 Hz to 4 Hz, typically 2 Hz; clause 6.2) drives an alternating volume
flow into an air cavity terminated either by the specimen or by an airtight
termination. The airflow resistance follows from the sound-pressure-level
difference between the two terminations (ISO 9053-2:2020, Formula (2), 8.7):

$$
R = \frac{\kappa' P_\mathrm{S}}{2\pi f V} \, \frac{h_\mathrm{t}}{h_\mathrm{s}} \, 10^{(L_{p\mathrm{s}} - L_{p\mathrm{t}})/20}
$$

with `kappa'` the effective ratio of specific heats for air (Annex A),
`P_S` the static (atmospheric) pressure (Pa), `f` the piston frequency (Hz),
`V` the cavity volume with the airtight termination (m3), `h_t`/`h_s` the
piston stroke amplitudes with the airtight termination / specimen cell, and
`L_ps`/`L_pt` the cavity sound pressure levels with the specimen /
airtight termination (dB). Only the level *difference* enters, so the sound level
device needs no absolute calibration (clause 8.7). The RMS piston volume flow is
$q_v = 2\pi f h A_\mathrm{P}$ (ISO 9053-2:2020, 6.2), with `h` the stroke
amplitude and `A_P` the piston cross-sectional area.

The **effective** ratio of specific heats `kappa'` accounts for heat conduction
between the oscillating air and the cavity walls, which makes the compression not
fully adiabatic. ISO 9053-2:2020 Annex A (normative) gives its evaluation from the
cavity geometry and air properties ([`effective_kappa`](/phonometry/reference/api/materials/airflow-resistance/#effective_kappa), Formula (A.7)); the
Annex A.3 worked example yields $\kappa' = 1.370$ (about 2 % below the
adiabatic $\kappa = 1.4008$). When no cavity/air data are supplied,
[`alternating_airflow_resistance`](/phonometry/reference/api/materials/airflow-resistance/#alternating_airflow_resistance) falls back to the **uncorrected adiabatic**
value $\kappa = 1.4$ (Formula (A.1)); for a conforming result compute
`kappa'` per Annex A and pass it explicitly.

Neither part defines a temperature/atmospheric normalisation of the result.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## airflow_resistance

```python
airflow_resistance(pressure_drop: float, volume_flow_rate: float) -> float
```

Airflow resistance $R = \Delta p / q_v$ (ISO 9053-1:2018, 3.1).

`pressure_drop` is the pressure difference $\Delta p$ across the
specimen (Pa) and `volume_flow_rate` is the volumetric airflow rate
`q_v` (m3/s). Returns `R` in Pa\*s/m3.

## airflow_resistivity

```python
airflow_resistivity(specific_resistance: float, thickness: float) -> float
```

Airflow resistivity $\sigma = R_\mathrm{s} / d$ (ISO 9053-1:2018, 3.3).

`specific_resistance` is `R_s` (Pa\*s/m) and `thickness` is `d` (m),
the specimen thickness in the flow direction. Returns `sigma` in Pa\*s/m2.

## AirflowResistanceWarning

Advisory for out-of-range or non-conforming ISO 9053 airflow inputs.

## alternating_airflow_resistance

```python
alternating_airflow_resistance(
    level_specimen: float,
    level_termination: float,
    *,
    piston_stroke_specimen: float,
    piston_stroke_termination: float,
    frequency: float,
    cavity_volume: float,
    static_pressure: float = 101325.0,
    kappa_prime: float = 1.4,
    background_level: float | None = None,
) -> float
```

Alternating-method airflow resistance (ISO 9053-2:2020, Formula (2), 8.7).

Implements:

$$
R = \frac{\kappa' P_\mathrm{S}}{2\pi f V} \, \frac{h_\mathrm{t}}{h_\mathrm{s}} \, 10^{(L_{p\mathrm{s}} - L_{p\mathrm{t}})/20}
$$

`level_specimen` (`L_ps`) and `level_termination` (`L_pt`) are the
cavity sound pressure levels (dB) with the specimen cell and the airtight
termination; `piston_stroke_specimen` (`h_s`) and
`piston_stroke_termination` (`h_t`) the corresponding stroke amplitudes
(m); `frequency` the piston frequency `f` (Hz, 1-4 Hz); `cavity_volume`
the airtight-termination cavity volume `V` (m3); `static_pressure` the
atmospheric pressure `P_S` (Pa, default 101325); `kappa_prime` the
effective ratio of specific heats `kappa'`; `background_level` the optional
cavity background level `L_pb` (dB) for the Formula (4) check. Returns `R` in
Pa\*s/m3.

`kappa_prime` defaults to the **uncorrected adiabatic**
$\kappa = 1.4$ (Formula (A.1)). For a result conforming to the
normative Annex A, compute the heat-conduction-corrected `kappa'` with
[`effective_kappa`](/phonometry/reference/api/materials/airflow-resistance/#effective_kappa) from the cavity geometry and pass it here (the
Annex A.3 example gives $\kappa' = 1.370$).

Emits [`AirflowResistanceWarning`](/phonometry/reference/api/materials/airflow-resistance/#airflowresistancewarning) when the piston frequency is outside
1-4 Hz or when the Formula (3)/(4) validity criteria are not met.

## ANNEX_A_AIR

*Constant* (`phonometry.fluids._state.Fluid`).

## effective_kappa

```python
effective_kappa(
    cavity_surface: float,
    cavity_volume: float,
    frequency: float,
    *,
    fluid: Fluid = ...,
) -> float
```

Effective ratio of specific heats `kappa'` (ISO 9053-2:2020, Annex A, Formula (A.7)).

Heat conduction between the oscillating air and the cavity walls makes the
compression not fully adiabatic, lowering `kappa` to:

$$
\kappa' = \frac{\kappa}{\sqrt{1 + (\kappa - 1)\frac{S}{V} b + 0.5 \left( (\kappa - 1)\frac{S}{V} b \right)^{2}}} \tag{A.7}
$$

with `b` the thermal boundary-layer thickness (Formulae (A.4)/(A.5),
[`thermal_boundary_layer_thickness`](/phonometry/reference/api/materials/airflow-resistance/#thermal_boundary_layer_thickness)), `S` the total internal surface area
of the air cavity (m2) and `V` its volume (m3).

`cavity_surface` is `S` (m2), `cavity_volume` `V` (m3) and `frequency`
the piston frequency `f` (Hz); `specific_heat_ratio` `kappa` (adiabatic) and
the remaining air properties default to air at the Annex A.3 reference state,
computed from IEC 61094-2:2009 Annex F (see
[`thermal_boundary_layer_thickness`](/phonometry/reference/api/materials/airflow-resistance/#thermal_boundary_layer_thickness) on why those differ from the pair
Annex A.3 prints). Returns the dimensionless `kappa'` for use in
[`alternating_airflow_resistance`](/phonometry/reference/api/materials/airflow-resistance/#alternating_airflow_resistance); the Annex A.3 worked example
($S = 0.0471$ m2, $V = 7.854\times 10^{-4}$ m3,
$f = 2$ Hz) yields $\kappa' = 1.370$.

## linear_airflow_velocity

```python
linear_airflow_velocity(volume_flow_rate: float, area: float) -> float
```

Linear airflow velocity $u = q_v / A$ (ISO 9053-1:2018, 3.4).

`volume_flow_rate` is `q_v` (m3/s) and `area` is `A` (m2); returns
`u` in m/s.

## piston_volume_flow_rate

```python
piston_volume_flow_rate(
    frequency: float,
    stroke_amplitude: float,
    piston_area: float,
) -> float
```

RMS piston volume flow $q_v = 2\pi f h A_\mathrm{P}$ (ISO 9053-2, 6.2).

`frequency` is the piston frequency `f` (Hz), `stroke_amplitude` the
stroke amplitude `h` (m) and `piston_area` the piston cross-section
`A_P` (m2). Returns `q_v` in m3/s.

## specific_airflow_resistance

```python
specific_airflow_resistance(
    resistance: float | None = None,
    area: float | None = None,
    *,
    pressure_drop: float | None = None,
    velocity: float | None = None,
) -> float
```

Specific airflow resistance `R_s` in Pa\*s/m (ISO 9053-1:2018, 3.2).

Two equivalent routes are accepted; supply exactly one:

- `resistance` (`R`, Pa\*s/m3) and `area` (`A`, m2):
  $R_\mathrm{s} = R\,A$.
- `pressure_drop` ($\Delta p$, Pa) and `velocity` (`u`, m/s):
  $R_\mathrm{s} = \Delta p / u$ (from $R_\mathrm{s} = R\,A$ with
  $u = q_v / A$).

The unit is pascal second per metre (Pa\*s/m), not Pa\*s/m2.

## static_airflow_resistance

```python
static_airflow_resistance(
    velocities: ArrayLike,
    pressure_drops: ArrayLike,
    area: float,
    thickness: float | None = None,
    *,
    evaluation_velocity: float = 0.0005,
) -> StaticAirflowResult
```

Stepwise static-method airflow resistance (ISO 9053-1:2018, clause 7.5).

Fits the measured pressure difference against the linear airflow velocity
with a second-order regression constrained through the origin,
$\Delta p = a u + b u^2$, and evaluates the resistances at
`evaluation_velocity` (the clause 7.5 reference `0.5e-3 m/s` by default).

`velocities` are the linear airflow velocities `u` (m/s) and
`pressure_drops` the matching pressure differences $\Delta p$ (Pa)
of at least two measurement steps; `area` is the cross-section `A` (m2)
and `thickness` the specimen thickness `d` (m, optional, enabling
`sigma`).

Because $R_\mathrm{s} = \Delta p / u = a + b u$, the returned
`linear_coefficient` `a` is the zero-velocity specific airflow
resistance. A velocity above the clause 7.5 upper limit (15 mm/s) raises
[`AirflowResistanceWarning`](/phonometry/reference/api/materials/airflow-resistance/#airflowresistancewarning).

## StaticAirflowResult

```python
StaticAirflowResult(
    resistance: float,
    specific_resistance: float,
    resistivity: float | None,
    evaluation_velocity: float,
    pressure_drop: float,
    linear_coefficient: float,
    quadratic_coefficient: float,
)
```

Result of an ISO 9053-1:2018 stepwise (static-method) determination.

`resistance` (`R`, Pa\*s/m3), `specific_resistance` (`R_s`, Pa\*s/m) and
`resistivity` (`sigma`, Pa\*s/m2; `None` when no thickness is supplied)
are evaluated at `evaluation_velocity` (m/s, the ISO 9053-1 clause 7.5
reference 0.5 mm/s by default). `linear_coefficient` (`a`) and
`quadratic_coefficient` (`b`) are the through-origin fit
$\Delta p = a u + b u^2$ (clause 7.5); `a` is the zero-velocity
specific airflow resistance (Pa\*s/m). `pressure_drop` is the fitted
$\Delta p$ at `evaluation_velocity` (Pa).

### StaticAirflowResult.plot()

```python
StaticAirflowResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the fitted `dp(u)` curve with the evaluation point.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### StaticAirflowResult.report()

```python
StaticAirflowResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 9053-1 static airflow-resistance test-report fiche to a PDF.

Writes a one-page accredited airflow-resistance report
(ISO 9053-1:2018, static/direct airflow method): the standard-basis
line, an optional metadata header block (client, manufacturer,
specimen, the specimen thickness `d`, test facility, date, climate
...), a two-panel body with a compact metrics table (the evaluation
velocity, the fitted pressure difference `dp`, the airflow resistance
`R`, the specific airflow resistance `R_s`, the airflow resistivity
`sigma` when a thickness is available, and the through-origin fit
coefficients `a` and `b`) beside the fitted `dp(u)` curve, a boxed
specific airflow resistance `R_s` with the airflow resistance `R`
and the resistivity `sigma` alongside, and a footer with the fixed
disclaimer. ISO 9053-1 is a material characterisation, so there is no
pass/fail verdict.

The clause 7.5 stepwise procedure fits $\Delta p = a u + b u^2$
through the origin and evaluates the resistances at the reference
velocity $u = 0.5$ mm/s; the linear coefficient `a` is the
zero-velocity
specific airflow resistance. Resistance quantities are printed to the
nearest whole Pa\*s unit and the evaluation velocity to one decimal
place (mm/s).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a body-and-disclaimer fiche. The applicable descriptive fields are `client`, `manufacturer`, `specimen`, `thickness` (the specimen thickness `d`, in metres, shown in millimetres), `test_room`, `test_date`, `temperature`, `relative_humidity`, `measurement_standard`, `laboratory`, `operator`, `report_id` and `notes`. The `requirement` field is ignored (ISO 9053-1 has no verdict). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the airflow-resistance fiche has a single body layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English, decimal point) or `"es"` (Spanish, decimal comma). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab or matplotlib is not installed. The fiche always embeds the fitted `dp(u)` curve, so both are required (`pip install "phonometry[report,plot]"`). |

## thermal_boundary_layer_thickness

```python
thermal_boundary_layer_thickness(
    frequency: float,
    *,
    fluid: Fluid = ...,
) -> float
```

Thermal boundary-layer thickness `b` (ISO 9053-2:2020, Formulae (A.4)/(A.5)).

$$
l_\mathrm{h} = \frac{k_\mathrm{a}}{\rho_0 c_0 C_\mathrm{P}} \tag{A.5}
$$

$$
b = \sqrt{\frac{2 c_0 l_\mathrm{h}}{\omega}}, \qquad \omega = 2\pi f \tag{A.4}
$$

`frequency` is the piston frequency `f` (Hz); `speed_of_sound` `c0` (m/s),
`air_density` `rho0` (kg/m3), `specific_heat_cp` `C_P` (J/(kg\*K)) and
`thermal_conductivity` `k_a` (J/(s\*m\*K)) are air properties, defaulting to air
at 23 degC, 101,325 kPa and 50 % relative humidity computed from IEC 61094-2:2009
Annex F. Note that `c0` cancels: `b` is
$\sqrt{2 k_\mathrm{a} / (\rho_0 C_\mathrm{P} \omega)}$, so only the pair
`k_a`/`C_P` and the density move it.

ISO 9053-2:2020 Annex A.3 prints a `k_a`/`C_P` pair 1,0800 times smaller than
Annex F gives at that state, which it credits to IEC 61094-2:2009 but which cannot
be found there (see `docs/ERRATA.md`). The defaults here are the Annex F values;
with the Annex A.3 example ($f = 2$ Hz) both pairs give `1.83e-3 m`, because
the printed pair preserves the tabulated diffusivity.
