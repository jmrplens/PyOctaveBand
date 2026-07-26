---
title: "noise_control.hvac"
description: "HVAC duct acoustics: end reflection, bends, plenums and flow-generated noise."
sidebar:
  label: "hvac"
---

HVAC duct acoustics: end reflection, bends, plenums and flow-generated noise.

A ventilation duct network attenuates fan noise through several mechanisms
that add up along the path, and it *regenerates* noise wherever the airflow is
disturbed. This module gathers the engineering methods of Bies, Hansen &
Howard, *Engineering Noise Control* 5th ed., Chapter 8, for the passive
attenuations -- **duct end reflection** (§8.13, Table 8.14), **bends/elbows**
(§8.11, Table 8.11) and **plenum chambers** (§8.17, Wells' method) -- and for
the **flow-generated (self) noise** of straight ducts and bends (§8.15).

The end-reflection and elbow methods are empirical look-up tables (ASHRAE);
they are interpolated over the duct size and, for the elbows, over the
frequency-to-width ratio `W / lambda`. The plenum and flow-noise methods are
closed forms evaluated directly.

:::note
Bies 5th ed. gives the duct end reflection only as the ASHRAE Table 8.14
look-up (there is no closed form in this edition); this module reproduces
that table and interpolates it. Rectangular ducts use the equivalent
diameter `D = sqrt(4 S / pi)`.
:::

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## elbow_insertion_loss

```python
elbow_insertion_loss(
    frequencies: ArrayLike,
    width: float,
    *,
    bend_type: str = 'square',
    vanes: bool = False,
    lined: bool = False,
    speed_of_sound: float = 343.0,
) -> HvacSpectrumResult
```

Duct bend/elbow insertion loss per bend (Bies Table 8.11, ASHRAE).

Indexed by the frequency-to-width ratio `W / lambda` (`lambda = c / f`).
Lined bends assume the lining extends at least three duct diameters up- and
downstream. Round bends are treated as unlined with no vanes.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `width` | Duct width `W` in the plane of the bend, m. |
| `bend_type` | `"square"` or `"round"`. |
| `vanes` | Turning vanes fitted (square bends only). |
| `lined` | Acoustically lined bend (square bends only). |
| `speed_of_sound` | Speed of sound `c`, m/s. |

**Returns:** A [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the insertion loss, dB per bend.

## end_reflection_loss

```python
end_reflection_loss(
    frequencies: ArrayLike,
    diameter: float,
    *,
    termination: str = 'flush',
    speed_of_sound: float = 343.0,
) -> HvacSpectrumResult
```

Duct end reflection loss (Bies Table 8.14, ASHRAE).

The low-frequency reflection of sound back up a duct at its open
termination into a room. Interpolated over `log` diameter and `log`
frequency from Table 8.14; it passes exactly through the tabulated
`(diameter, octave band)` nodes.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz (1-D array). |
| `diameter` | Duct internal diameter `D`, m (use `D = sqrt(4 S / pi)` for a rectangular duct of area `S`). |
| `termination` | `"flush"` (duct flush with a wall/ceiling) or `"free"` (free space / suspended in the room). |
| `speed_of_sound` | Speed of sound `c`, m/s (kept for signature symmetry; the table is indexed by frequency directly). |

**Returns:** A [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the reflection loss, dB.

## flow_noise_bend

```python
flow_noise_bend(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
    height: float,
    *,
    density: float = 1.206,
) -> HvacSpectrumResult
```

Flow-generated octave-band sound power of a mitred bend (Bies Eqs. (8.252), (8.254)).

`L_WB = L_Ws - 10 log10(1 + 0.165 N_s^2) + 30 log10(U) - 103` with the
stream power level `L_Ws = 30 log10(U) + 10 log10(S) + 10 log10(rho) + 117`
(Bies Eq. (8.252)) and the Strouhal number `N_s = f H / U` (`H` the duct
height in the plane of the bend). The radiated sound power grows as the
sixth power of the stream speed at low `N_s` (the inner-corner drag
dipole) and the eighth power at high `N_s` (the outer-corner shear
quadrupole); equivalently, the *efficiency* referenced to the stream power
grows as `U^3` and `U^5` respectively.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centre frequencies `f`, Hz (1-D array). |
| `flow_velocity` | Mean flow speed `U`, m/s. |
| `area` | Duct cross-sectional area `S`, m2. |
| `height` | Duct height `H` in the plane of the bend, m. |
| `density` | Air density `rho`, kg/m3. |

**Returns:** A [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the band sound power level, dB re 1e-12 W.

## flow_noise_straight_duct

```python
flow_noise_straight_duct(
    frequencies: ArrayLike,
    flow_velocity: float,
    area: float,
) -> HvacSpectrumResult
```

Flow-generated octave-band sound power of a straight duct (Bies Eq. (8.251)).

`L_WB = 7 + 50 log10(U) + 10 log10(S) - 2 - 26 log10(1.14 + 0.02 f / U)`
in dB re 1e-12 W (VDI 2081-1), for airflow speed `U` in a duct of area
`S`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centre frequencies `f`, Hz (1-D array). |
| `flow_velocity` | Mean flow speed `U`, m/s. |
| `area` | Duct cross-sectional area `S`, m2. |

**Returns:** A [`HvacSpectrumResult`](/phonometry/reference/api/noise_control/hvac/#hvacspectrumresult) of the band sound power level, dB re 1e-12 W.

## HvacSpectrumResult

```python
HvacSpectrumResult(
    frequencies: np.ndarray,
    values: np.ndarray,
    quantity: str,
    label: str,
)
```

A per-frequency HVAC quantity (attenuation or regenerated power level).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz. |
| `values` | The quantity per frequency (dB, or dB re 1e-12 W for a sound power level). |
| `quantity` | What `values` holds (`"attenuation"` or `"sound_power_level"`). |
| `label` | A short human label of the element. |

### HvacSpectrumResult.plot()

```python
HvacSpectrumResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the quantity against a continuous log-frequency axis.

Requires matplotlib (`pip install phonometry[plot]`).

### HvacSpectrumResult.report()

```python
HvacSpectrumResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an HVAC duct-noise-spectrum fiche to `path`.

Writes a one-page HVAC-noise sheet: the method-basis line naming the
reported quantity and the Bies, Hansen & Howard chapter (Engineering
Noise Control 5th ed., Chapter 8), an optional metadata header (client,
duct element, test environment, instrumentation, climate, date), a
per-band table (nominal frequency and the reported quantity) beside the
spectrum, the boxed single-number result (for a regenerated-noise
spectrum the A-weighted sound power level `L_WA` re 1 pW with the
overall unweighted total; for an attenuation spectrum the mean
attenuation with its band range), an optional verdict row against a
declared limit, and a method-basis strip stating the reported quantity's
relation.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the duct element, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared maximum A-weighted sound power level for a regenerated-noise spectrum (lower is better) or a declared minimum mean attenuation for an attenuation spectrum (more is better). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` a regenerated-noise table adds the A-weighting correction and the A-weighted band level columns. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |

## plenum_attenuation

```python
plenum_attenuation(
    exit_area: float,
    line_of_sight: float,
    wall_area: float,
    mean_absorption: ArrayLike,
    *,
    angle: float = 0.0,
) -> np.ndarray | float
```

Plenum-chamber transmission loss by Wells' method (Bies Eq. (8.275)).

`TL = -10 log10[ S_out ( cos(theta) / (pi r^2) + (1 - alpha) / (S_w alpha) ) ]`,
where the reverberant term uses the plenum room constant
`R = S_w alpha / (1 - alpha)` ([`phonometry.room.room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant)). The
method holds above the inlet cut-on and when the plenum is large compared
with the wavelength; it underpredicts the low-frequency loss by 5-10 dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `exit_area` | Outlet-opening area `S_out`, m2. |
| `line_of_sight` | Straight-line inlet-to-outlet distance `r`, m. |
| `wall_area` | Total internal wall area `S_w`, m2. |
| `mean_absorption` | Mean Sabine wall absorption `alpha` in `(0, 1)` (scalar or per-band). |
| `angle` | Angle `theta` between the inlet axis and the line to the outlet, rad (default 0). |

**Returns:** The transmission loss, dB (float for scalar absorption, else a per-band array).
