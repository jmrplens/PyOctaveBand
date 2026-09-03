---
title: "environment.sources.cnossos_rail"
description: "CNOSSOS-EU railway source emission (Directive 2002/49/EC Annex II, 2.3)."
sidebar:
  label: "cnossos_rail"
---

CNOSSOS-EU railway source emission (Directive 2002/49/EC Annex II, 2.3).

The common noise assessment methods of the European Union describe a railway
track as **two** incoherent source lines at the centre of the track, at
$h = 0.5$ m (source A) and $h = 4.0$ m (source B) above the plane
tangent to
the two upper rail surfaces. Every physical source of a vehicle is allocated to
one of the two heights and contributes a directional sound power per metre of
line

$$
L'_{W,\mathrm{eq,line},i}(\psi,\phi) = L_{W,0,\mathrm{dir},i}(\psi,\phi) + 10 \log_{10}\left( \frac{Q}{1000 v} \right) \tag{2.3.2}
$$

for a running train, or $+\, 10 \log_{10}(T_\mathrm{idle} / (T_\mathrm{ref} L))$ (2.3.4)
for an idling
one. This module implements the whole of 2.3 together with the coefficient
database of Appendix G, in the twenty-four 1/3-octave bands from 50 Hz to
10 kHz, and energy-sums them into the eight octave bands the propagation stage
consumes.

Which text is implemented
-------------------------
Annex II was replaced by Commission Directive (EU) 2015/996, corrected by the
corrigendum of OJ L 5, 10.1.2018 and amended by Commission Delegated Directive
(EU) 2021/1226. The consolidated text (02002L0049) is what is implemented here,
and every shipped table records the instrument it comes from:

* the roughness-to-frequency conversion uses $f = v/\lambda$ with
  **v in m/s**
  as corrected in 2018 (the 2015 text says km/h, which is wrong by a factor
  3.6);
* the whole of Appendix G is the corrigendum's replacement text, with Tables
  G-1b, G-2, G-3a, G-4 and G-7 as **replaced** by (EU) 2021/1226 and Tables
  G-1a, G-3b, G-3c, G-5 and G-6 as re-issued in 2018 with the band labels
  corrected in 2021. The letter suffixes are this module's shorthand: the
  Official Journal prints two tables under the number G-1 (wheel then rail
  roughness) and three sections under G-3, and names neither set;
* curve squeal follows the 2021 rule (5 dB / 8 dB by radius, with a separate
  tram rule and a turnout rule), not the 2015 one;
* bridge noise is a **separate source** built on the transfer function
  `L_H,bridge,i` of Table G-7 (2.3.18 as replaced in 2021), not the constant
  `C_bridge` of 2015;
* the vertical directivity of source A is the 2021 form, with no absolute-value
  bars and identically zero for $\psi \le 0$. The superseded 2015 form is
  available through [`DirectivityEdition`](/phonometry/reference/api/environment/cnossos-rail/#directivityedition) for comparison with pre-2021
  studies, because the two differ over the whole lower half space.

What is verified against digits and what is not
-----------------------------------------------
Annex II prints **no worked example** for the railway source. The end-to-end
chain implemented here is pinned against the emission test workbook published
with the Commission's CNOSSOS-EU source module, which was computed with the
**2015** coefficient database; the shipped tables are therefore verified as
transcriptions, and the equations that combine them are verified end to end
against an independent implementation. Two points are interpretation, not
transcription, and are documented as such:
[`RoughnessInterpolation`](/phonometry/reference/api/environment/cnossos-rail/#roughnessinterpolation) (the Directive describes the wavelength-to-
frequency resampling in prose only) and the horizontal directivity of traction
noise (2.3.15 enumerates rolling, impact, squeal, braking, fans and aerodynamic
effects; the reference module applies the dipole to every source, which is what
is done here).

Scope
-----
This is the **emission** stage only. Splitting a source line into equivalent
point sources is explicitly outside the scope of the method (2.5.3), and the
CNOSSOS propagation model is not ISO 9613-2, so the hand-off to
[`outdoor_propagation`](/phonometry/reference/api/environment/outdoor-propagation/) mixes two methods and is a
convenience, not a normative chain.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AERODYNAMIC_REFERENCE_SPEED

*Constant* (`float`).

```python
AERODYNAMIC_REFERENCE_SPEED = 300.0
```

## aerodynamic_sound_power

```python
aerodynamic_sound_power(
    speed: float = 300.0,
    *,
    reference: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    alpha: float = 50.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Aerodynamic sound power of (2.3.13) and (2.3.14), in dB.

$L_{W,0,i} = L_{W,0,h,i}(v_0) + \alpha_{h,i} \log_{10}(v/v_0)$ with
$v_0 = 300$ km/h.
At the reference speed the result is Table G-6 verbatim.

**Parameters**

| Name | Description |
| :--- | :--- |
| `speed` | Train speed `v`, in km/h. |
| `reference` | `(source A, source B)` reference spectra at `v_0`, or `None` (the default) for Table G-6. |
| `alpha` | Speed exponent `alpha_h,i`; Table G-6 gives 50 in every band. |

**Returns:** `(source A spectrum, source B spectrum)`, 24 values each.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the speed is not positive. |

## AERODYNAMIC_THRESHOLD_SPEED

*Constant* (`float`).

```python
AERODYNAMIC_THRESHOLD_SPEED = 200.0
```

## BrakeType

```python
BrakeType(*values)
```

Digit 3 of the vehicle descriptor, Table [2.3.a].

## bridge_transfer

```python
bridge_transfer(bridge: BridgeType | str) -> NDArray[np.float64]
```

Bridge transfer function `L_H,bridge,i` of Table G-7, in dB per axle.

**Parameters**

| Name | Description |
| :--- | :--- |
| `bridge` | A [`BridgeType`](/phonometry/reference/api/environment/cnossos-rail/#bridgetype) member or its column label. |

**Returns:** The 24 1/3-octave values, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the column is not tabulated. |

## BridgeType

```python
BridgeType(*values)
```

Columns of Table G-7, labelled by the A-weighted bridge excess.

## contact_filter

```python
contact_filter(
    filter_: ContactFilter | tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Contact filter `A_3` of Table G-2.

**Parameters**

| Name | Description |
| :--- | :--- |
| `filter_` | A [`ContactFilter`](/phonometry/reference/api/environment/cnossos-rail/#contactfilter) member or the `(wheel load in kN, wheel diameter in mm)` pair that labels the column. |

**Returns:** `(wavelengths in mm, levels in dB)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the combination is not tabulated. |

## ContactFilter

```python
ContactFilter(*values)
```

Columns of Table G-2, as `(wheel load in kN, wheel diameter in mm)`.

## curve_squeal_excess

```python
curve_squeal_excess(
    radius: float,
    *,
    tram: bool = False,
    turnout: bool = False,
    track_length: float = 50.0,
) -> float
```

Curve-squeal excess added to the rolling noise, in dB.

The rule is the one (EU) 2021/1226 Annex point (4)(b) substituted for the
2015 text: for trains, 8 dB at $R \le 300$ m and 5 dB at
$300 < R \le 500$ m over at least 50 m of curve, and 8 dB on switch
turnouts with $R \le 300$ m whatever their length; for trams, 5 dB
on
curves and switch turnouts with $R \le 200$ m. The excess applies
at all
frequencies.

**Parameters**

| Name | Description |
| :--- | :--- |
| `radius` | Curve radius `R`, in m. |
| `tram` | `True` for a tram, which follows its own rule. |
| `turnout` | `True` for a switch turnout, where the minimum curve length does not apply. |
| `track_length` | Length of track along the curve `l_track`, in m. |

**Returns:** The excess, in dB (0.0 where no squeal is modelled).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the radius or the track length is not positive. |

## DirectivityEdition

```python
DirectivityEdition(*values)
```

Which text of the vertical directivity (2.3.16) to evaluate.

## horizontal_directivity

```python
horizontal_directivity(
    phi: float,
    *,
    frequencies: ArrayLike = (50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0),
) -> NDArray[np.float64]
```

Horizontal directivity `dL_W,dir,hor,i` of (2.3.15), in dB.

$10 \log_{10}(0.01 + 0.99 \sin^2 \phi)$: a dipole, identical in every
band, equal
to 0 dB broadside ($\phi = 90^\circ$) and to
$10 \log_{10} 0.01 = -20$ dB along
the track. The Directive offers it "by default" for rolling, impact,
squeal, braking, fans and aerodynamic effects; since no other horizontal
directivity is given and traction noise includes the fans, it is applied
here to every source, as the Commission's reference module does.

**Parameters**

| Name | Description |
| :--- | :--- |
| `phi` | Horizontal angle `phi`, in degrees, measured from the direction of travel (Figure [2.3.b]). |
| `frequencies` | Midband frequencies, used only for the array shape. |

**Returns:** The correction, in dB, one value per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the angle is not finite. |

## impact_roughness

```python
impact_roughness(
    single: ArrayLike,
    joint_density: float,
) -> NDArray[np.float64]
```

Impact roughness `L_R,IMPACT,i` of (2.3.12), in dB.

$L_{\mathrm{R,IMPACT},i} = L_{\mathrm{R,IMPACT-SINGLE},i} + 10 \log_{10}(n_l/0.01)$, so at the
tabulated density of one joint per 100 m the table is returned verbatim.

**Parameters**

| Name | Description |
| :--- | :--- |
| `single` | Single-impact roughness on the frequency grid, in dB. |
| `joint_density` | Joint density `n_l`, in m^-1. |

**Returns:** `L_R,IMPACT,i`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the joint density is negative or not finite. |

## impact_roughness_single

```python
impact_roughness_single() -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Impact roughness `L_R,IMPACT-SINGLE` of Table G-4.

The table is given for a joint density $n_l = 0.01$ m^-1, that is one
switch, joint or crossing per 100 m, which is also the default the
Directive prescribes for jointed track.

**Returns:** `(wavelengths in mm, levels in dB)`.

## octave_bands_from_third_octaves

```python
octave_bands_from_third_octaves(levels: ArrayLike) -> NDArray[np.float64]
```

Energy-sum a 24-band 1/3-octave spectrum into the eight octave bands.

Annex II 2.3.2 requires the directional sound power to be derived in 1/3
octave bands and then "expressed in octave bands by energetically adding
each pertaining 1/3 octave band together into the corresponding octave
band".

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | The 24 1/3-octave levels from 50 Hz to 10 kHz, in dB. |

**Returns:** The eight octave levels from 63 Hz to 8 kHz, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectrum is not 24 bands. |

## rail_roughness

```python
rail_roughness(
    roughness: RailRoughnessClass | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Rail roughness `L_r,TR` of Table G-1b.

Only the two maintained classes `E` and `M` are tabulated; the `N`
and `B` classes of Table [2.3.b] carry no spectrum in Appendix G and have
to be supplied by the Member State.

**Parameters**

| Name | Description |
| :--- | :--- |
| `roughness` | The [`RailRoughnessClass`](/phonometry/reference/api/environment/cnossos-rail/#railroughnessclass) of digit 2 of the track descriptor. |

**Returns:** `(wavelengths in mm, levels in dB)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the class carries no spectrum in Table G-1b. |

## RailJoints

```python
RailJoints(*values)
```

Digit 5 of the track descriptor, Table [2.3.b].

## RailPad

```python
RailPad(*values)
```

Digit 3 of the track descriptor: rail-pad **dynamic** stiffness.

(EU) 2021/1226 Annex point (3) replaced "acoustic" stiffness by
**dynamic** stiffness and re-worded the hard class as
"Hard (800-1 000 MN/m)".

## RailRoughnessClass

```python
RailRoughnessClass(*values)
```

Digit 2 of the track descriptor, Table [2.3.b].

## RAILWAY_MINIMUM_SPEED

*Constant* (`float`).

```python
RAILWAY_MINIMUM_SPEED = 50.0
```

## RAILWAY_OCTAVE_BANDS

*Constant* (`tuple`).

```python
RAILWAY_OCTAVE_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
```

## RAILWAY_SOURCE_HEIGHTS

*Constant* (`tuple`).

```python
RAILWAY_SOURCE_HEIGHTS = (0.5, 4.0)
```

## railway_source_power

```python
railway_source_power(
    traffic: RailwayVehicle | list[RailwayVehicle] | tuple[RailwayVehicle, ...],
    track: RailwayTrack,
    *,
    psi: float = 0.0,
    phi: float = 90.0,
    reference_time: float = 12.0,
    minimum_speed: float | None = None,
    interpolation: RoughnessInterpolation = ...,
    directivity_edition: DirectivityEdition = ...,
) -> RailwayEmissionResult
```

Directional sound power per metre of a railway source line (2.3.1).

Assembles, for every vehicle of the traffic and both source heights, the
rolling noise (2.3.8)-(2.3.11), the impact noise (2.3.12), the curve squeal,
the traction noise, the aerodynamic noise (2.3.13)-(2.3.14) and the bridge
noise (2.3.18); applies the directivity of (2.3.15)-(2.3.17); adds the flow
term of (2.3.2) or (2.3.4); and energy-sums everything over the traffic.

Rolling, impact, squeal and bridge noise sit at source A. Traction and
aerodynamic noise are tabulated separately for the two heights, so their
split between A and B is read from the data rather than assumed. Rolling
noise is excluded while a vehicle idles, and impact noise is not modelled
below the minimum speed nor while idling.

**Parameters**

| Name | Description |
| :--- | :--- |
| `traffic` | One [`RailwayVehicle`](/phonometry/reference/api/environment/cnossos-rail/#railwayvehicle) or a sequence of them. |
| `track` | The [`RailwayTrack`](/phonometry/reference/api/environment/cnossos-rail/#railwaytrack) of the section. |
| `psi` | Vertical angle `psi` to the receiver, in degrees. |
| `phi` | Horizontal angle `phi` to the receiver, in degrees; the default 90 deg is broadside, where the dipole correction is 0 dB. |
| `reference_time` | Reference period `T_ref` of (2.3.4), in the same unit as `idling_time`. |
| `minimum_speed` | Speed floor used to determine the total effective roughness, in km/h; `None` (the default) selects 50 km/h, or 30 km/h for a tram. Pass `0` to switch the floor off, which also switches off the exclusion of impact noise below it. |
| `interpolation` | The [`RoughnessInterpolation`](/phonometry/reference/api/environment/cnossos-rail/#roughnessinterpolation) rule. |
| `directivity_edition` | Which text of (2.3.16) to evaluate. |

**Returns:** A [`RailwayEmissionResult`](/phonometry/reference/api/environment/cnossos-rail/#railwayemissionresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the traffic is empty or an input is invalid. |

## RAILWAY_THIRD_OCTAVE_BANDS

*Constant* (`tuple`).

```python
RAILWAY_THIRD_OCTAVE_BANDS = (50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0)
```

## RailwayEmissionResult

```python
RailwayEmissionResult(
    third_octave_frequencies: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    heights: tuple[float, float],
    third_octave_line_power: NDArray[np.float64],
    line_power: NDArray[np.float64],
    total_line_power: NDArray[np.float64],
    components: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = ...,
)
```

Directional sound power per metre of a CNOSSOS-EU railway source.

**Attributes**

| Name | Description |
| :--- | :--- |
| `third_octave_frequencies` | The 24 1/3-octave midband frequencies, Hz. |
| `frequencies` | The eight octave midband frequencies, Hz. |
| `heights` | Heights of the two equivalent source lines, in m. |
| `third_octave_line_power` | `L'_W,eq,line,i(psi,phi)` per source height and 1/3-octave band, in dB re 1 pW per metre. |
| `line_power` | The same, energy-summed into octave bands. |
| `total_line_power` | The two heights summed, per octave band. |
| `components` | The 1/3-octave sound power of each physical source before the flow term and the directivity, keyed by `"rolling"`, `"traction"`, `"aerodynamic"` and `"bridge"`, each holding the `(source A, source B)` pair. |

### RailwayEmissionResult.plot()

```python
RailwayEmissionResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-metre line power of the two equivalent source heights.

## RailwayTrack

```python
RailwayTrack(
    rail_roughness: tuple[Any, Any],
    track_transfer: Any,
    impact_roughness: tuple[Any, Any] | None = None,
    joint_density: float = 0.01,
    bridge_transfer: Any | None = None,
    squeal_excess: float = 0.0,
    length: float = 100.0,
)
```

The Appendix G data of one track section.

**Attributes**

| Name | Description |
| :--- | :--- |
| `rail_roughness` | `(wavelengths in mm, levels in dB)` of Table G-1b. |
| `track_transfer` | `L_H,TR,i` of Table G-3a, 24 values in dB. |
| `impact_roughness` | `(wavelengths in mm, levels in dB)` of Table G-4, or `None` where there is no joint, switch or crossing. |
| `joint_density` | Joint density `n_l`, in m^-1. |
| `bridge_transfer` | `L_H,bridge,i` of Table G-7 where the section is on a bridge, or `None`. |
| `squeal_excess` | Curve-squeal excess in dB, from [`curve_squeal_excess`](/phonometry/reference/api/environment/cnossos-rail/#curve_squeal_excess). |
| `length` | Length `L` of the track section, in m; used only by the idling flow term (2.3.4). |

## RailwayVehicle

```python
RailwayVehicle(
    stock: RollingStock,
    flow_rate: float = 0.0,
    speed: float = 0.0,
    condition: RunningCondition = ...,
    idling_time: float = 0.0,
)
```

One vehicle of the traffic on a track section.

**Attributes**

| Name | Description |
| :--- | :--- |
| `stock` | The [`RollingStock`](/phonometry/reference/api/environment/cnossos-rail/#rollingstock) data of the vehicle type. |
| `flow_rate` | Average number of vehicles per hour `Q`. |
| `speed` | Their speed `v` on the track section, in km/h. |
| `condition` | The [`RunningCondition`](/phonometry/reference/api/environment/cnossos-rail/#runningcondition) `c`. |
| `idling_time` | Total idling time `T_idle` within `T_ref`, in the same unit as `T_ref`; used only when `condition` is idling. |

## REFERENCE_JOINT_DENSITY

*Constant* (`float`).

```python
REFERENCE_JOINT_DENSITY = 0.01
```

## rolling_sound_power

```python
rolling_sound_power(
    roughness: ArrayLike,
    transfer: ArrayLike,
    axles: float,
) -> NDArray[np.float64]
```

One rolling-noise component of (2.3.8) to (2.3.10), in dB.

$L_{W,0,i} = L_{\mathrm{R,TOT},i} + L_{H,i} + 10 \log_{10}(N_\mathrm{a})$: the same
addition serves the
track, the wheel and the freight superstructure, each with its own transfer
function. All three sit at source A.

**Parameters**

| Name | Description |
| :--- | :--- |
| `roughness` | Total effective roughness `L_R,TOT,i`, in dB. |
| `transfer` | Transfer function `L_H,i`, in dB per axle. |
| `axles` | Number of axles per vehicle `N_a`. |

**Returns:** The component sound power, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `axles` is not a positive number. |

## RollingStock

```python
RollingStock(
    axles: int,
    wheel_roughness: tuple[Any, Any],
    contact_filter: tuple[Any, Any],
    wheel_transfer: Any,
    superstructure_transfer: Any | None = None,
    traction: tuple[Any, Any] | None = None,
    aerodynamic: tuple[Any, Any] | None = None,
    aerodynamic_alpha: float = 50.0,
    tram: bool = False,
)
```

The Appendix G data of one vehicle type, on its own wavelength grids.

Every field is the spectrum the method needs, so a Member State substitutes
its own database simply by building this object from its own tables rather
than from the [`cnossos_rail`](/phonometry/reference/api/environment/cnossos-rail/) look-ups.

**Attributes**

| Name | Description |
| :--- | :--- |
| `axles` | Number of axles per vehicle `N_a`. |
| `wheel_roughness` | `(wavelengths in mm, levels in dB)` of Table G-1a. |
| `contact_filter` | `(wavelengths in mm, levels in dB)` of Table G-2. |
| `wheel_transfer` | `L_H,VEH,i` of Table G-3b, 24 values in dB. |
| `superstructure_transfer` | `L_H,VEH,SUP,i` of Table G-3c for a freight wagon, or `None` for any other vehicle type. |
| `traction` | `(source A, source B)` spectra of Table G-5, or `None` for an unpowered vehicle. |
| `aerodynamic` | `(source A, source B)` reference spectra of Table G-6 at `v_0`, or `None` to leave aerodynamic noise out. |
| `aerodynamic_alpha` | Speed exponent of (2.3.13) and (2.3.14). |
| `tram` | `True` for a tram or light metro, which uses the lower minimum speed and the tram squeal rule. |

## roughness_to_frequency

```python
roughness_to_frequency(
    levels: ArrayLike,
    wavelengths: ArrayLike,
    speed: float,
    *,
    frequencies: ArrayLike = (50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0),
    interpolation: RoughnessInterpolation = ...,
) -> NDArray[np.float64]
```

Resample a roughness spectrum from wavelength onto frequency.

A roughness level is tabulated against wavelength and has to be read at
$\lambda = v/f$ with **v in m/s** (the corrigendum of OJ L 5,
10.1.2018;
the 2015 text says km/h, which is wrong by a factor 3.6). The value at the
wanted wavelength is obtained from the two neighbouring tabulated bands
according to `interpolation`; beyond the ends of the table the end value
is held.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Roughness levels of the table, in dB. |
| `wavelengths` | Wavelengths of the table, in mm, in any monotonic order. |
| `speed` | Train speed `v`, in km/h. |
| `frequencies` | Target midband frequencies, in Hz. |
| `interpolation` | The [`RoughnessInterpolation`](/phonometry/reference/api/environment/cnossos-rail/#roughnessinterpolation) rule. |

**Returns:** The spectrum on the target frequency grid, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is invalid. |

## RoughnessInterpolation

```python
RoughnessInterpolation(*values)
```

How a roughness spectrum is resampled from wavelength onto frequency.

The Directive describes the resampling in prose only: "the two
corresponding 1/3 octave bands defined in the wavelength domain shall be
averaged energetically and proportionally". No formula and no example is
given, so the rule has to be chosen, and the choice is the single largest
interpretation risk of the railway model.

## RunningCondition

```python
RunningCondition(*values)
```

Running condition `c` of 2.3.2.

Only two conditions are modelled: constant speed, which the Directive says
is valid as well when the train accelerates or decelerates, and idling.

## superstructure_transfer

```python
superstructure_transfer() -> NDArray[np.float64]
```

Superstructure transfer `L_H,VEH,SUP,i` of Table G-3c, in dB per axle.

Only one superstructure is tabulated, the "EU standard" of vehicle type
`a` (freight), and it is 0.0 dB in every band, so (2.3.10) reduces to
$L_{\mathrm{R,TOT},i} + 10 \log_{10}(N_\mathrm{a})$. The contribution is considered
for freight wagons only.

**Returns:** The 24 1/3-octave values, all zero.

## total_effective_roughness

```python
total_effective_roughness(
    rail: ArrayLike,
    wheel: ArrayLike,
    filter_: ArrayLike,
) -> NDArray[np.float64]
```

Total effective roughness `L_R,TOT,i` of (2.3.7), in dB.

$L_{\mathrm{R,TOT},i} = 10 \log_{10}(10^{L_{\mathrm{r,TR},i}/10} + 10^{L_{\mathrm{r,VEH},i}/10}) + A_{3,i}$. All
three spectra must already be on the frequency grid, that is resampled with
[`roughness_to_frequency`](/phonometry/reference/api/environment/cnossos-rail/#roughness_to_frequency) at the speed of interest.

**Parameters**

| Name | Description |
| :--- | :--- |
| `rail` | Rail roughness `L_r,TR,i`, in dB. |
| `wheel` | Wheel roughness `L_r,VEH,i`, in dB. |
| `filter_` | Contact filter `A_3,i`, in dB. |

**Returns:** `L_R,TOT,i`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectra are not 24 bands each. |

## track_transfer

```python
track_transfer(track: TrackTransferClass | str) -> NDArray[np.float64]
```

Track transfer function `L_H,TR,i` of Table G-3a, in dB per axle.

**Parameters**

| Name | Description |
| :--- | :--- |
| `track` | A [`TrackTransferClass`](/phonometry/reference/api/environment/cnossos-rail/#tracktransferclass) member or its column code. |

**Returns:** The 24 1/3-octave values, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the column is not tabulated. |

## TrackBase

```python
TrackBase(*values)
```

Digit 1 of the track descriptor, Table [2.3.b].

## TrackCurvature

```python
TrackCurvature(*values)
```

Digit 6 of the track descriptor, Table [2.3.b].

## TrackDescriptor

```python
TrackDescriptor(
    base: TrackBase,
    roughness: RailRoughnessClass,
    pad: RailPad,
    measure: TrackMeasure = ...,
    joints: RailJoints = ...,
    curvature: TrackCurvature = ...,
)
```

The six-digit track descriptor of Table [2.3.b].

**Attributes**

| Name | Description |
| :--- | :--- |
| `base` | Digit 1, the [`TrackBase`](/phonometry/reference/api/environment/cnossos-rail/#trackbase). |
| `roughness` | Digit 2, the [`RailRoughnessClass`](/phonometry/reference/api/environment/cnossos-rail/#railroughnessclass). |
| `pad` | Digit 3, the [`RailPad`](/phonometry/reference/api/environment/cnossos-rail/#railpad) dynamic stiffness. |
| `measure` | Digit 4, the [`TrackMeasure`](/phonometry/reference/api/environment/cnossos-rail/#trackmeasure). |
| `joints` | Digit 5, the [`RailJoints`](/phonometry/reference/api/environment/cnossos-rail/#railjoints). |
| `curvature` | Digit 6, the [`TrackCurvature`](/phonometry/reference/api/environment/cnossos-rail/#trackcurvature). |

### TrackDescriptor.code

*property*

The descriptor written back out as a string.

### TrackDescriptor.from_code()

*classmethod*

```python
TrackDescriptor.from_code(code: str) -> TrackDescriptor
```

Parse a descriptor such as `"BMSNNN"`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `code` | The six-character descriptor. |

**Returns:** The parsed [`TrackDescriptor`](/phonometry/reference/api/environment/cnossos-rail/#trackdescriptor).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the code is not a valid six-digit descriptor. |

## TrackMeasure

```python
TrackMeasure(*values)
```

Digit 4 of the track descriptor, Table [2.3.b].

## TrackTransferClass

```python
TrackTransferClass(*values)
```

Columns of Table G-3a, `track base / rail pad` of the track descriptor.

## traction_sound_power

```python
traction_sound_power(
    vehicle: TractionVehicle | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Traction sound power per vehicle of Table G-5, in dB.

Because the Directive models only constant speed and idling and takes the
source strength at maximum load,
$L_{W,0,\mathrm{const},i} = L_{W,0,\mathrm{idling},i}$, so this
one table serves both running conditions.

**Parameters**

| Name | Description |
| :--- | :--- |
| `vehicle` | A [`TractionVehicle`](/phonometry/reference/api/environment/cnossos-rail/#tractionvehicle) member or its description. |

**Returns:** `(source A spectrum, source B spectrum)`, 24 values each.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the vehicle is not tabulated. |

## TractionVehicle

```python
TractionVehicle(*values)
```

Columns of Table G-5, the traction sound power per vehicle.

## TRAM_MINIMUM_SPEED

*Constant* (`float`).

```python
TRAM_MINIMUM_SPEED = 30.0
```

## VehicleDescriptor

```python
VehicleDescriptor(
    vehicle_type: VehicleType,
    axles: int,
    brake: BrakeType,
    measure: WheelMeasure = ...,
)
```

The four-digit vehicle descriptor of Table [2.3.a].

**Attributes**

| Name | Description |
| :--- | :--- |
| `vehicle_type` | Digit 1, the [`VehicleType`](/phonometry/reference/api/environment/cnossos-rail/#vehicletype). |
| `axles` | Digit 2, the number of axles per vehicle. |
| `brake` | Digit 3, the [`BrakeType`](/phonometry/reference/api/environment/cnossos-rail/#braketype). |
| `measure` | Digit 4, the [`WheelMeasure`](/phonometry/reference/api/environment/cnossos-rail/#wheelmeasure). |

### VehicleDescriptor.code

*property*

The descriptor written back out as a string.

### VehicleDescriptor.from_code()

*classmethod*

```python
VehicleDescriptor.from_code(code: str) -> VehicleDescriptor
```

Parse a descriptor such as `"a4cn"` or `"h16nn"`.

The second digit is the actual number of axles, so it may run to more
than one character.

**Parameters**

| Name | Description |
| :--- | :--- |
| `code` | The descriptor, first digit to last. |

**Returns:** The parsed [`VehicleDescriptor`](/phonometry/reference/api/environment/cnossos-rail/#vehicledescriptor).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the code is not a valid four-digit descriptor. |

## VehicleType

```python
VehicleType(*values)
```

Digit 1 of the vehicle descriptor, Table [2.3.a].

## vertical_directivity

```python
vertical_directivity(
    psi: float,
    *,
    frequencies: ArrayLike = (50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0),
    height: int = 1,
    aerodynamic: bool = False,
    edition: DirectivityEdition = ...,
) -> NDArray[np.float64]
```

Vertical directivity `dL_W,dir,ver,i` of (2.3.16) and (2.3.17), in dB.

Source A (`height = 1`) follows (2.3.16), which (EU) 2021/1226 Annex
point (4)(d) replaced: the absolute-value bars of the 2015 text are gone
and the correction is identically zero for $\psi \le 0$. Source B
(`height = 2`) follows (2.3.17) for the aerodynamic effect only,
$10 \log_{10}(\cos^2 \psi)$ for $\psi < 0$, and is omni-directional
for every
other source.

**Parameters**

| Name | Description |
| :--- | :--- |
| `psi` | Vertical angle `psi`, in degrees (Figure [2.3.b]). |
| `frequencies` | Midband frequencies `f_c,i`, in Hz. |
| `height` | `1` for source A at 0.5 m, `2` for source B at 4.0 m. |
| `aerodynamic` | `True` to select the aerodynamic source at `height = 2`; ignored at `height = 1`. |
| `edition` | Which text of (2.3.16) to evaluate. |

**Returns:** The correction, in dB, one value per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the angle is not finite or the height is not 1 or 2. |

## wheel_roughness

```python
wheel_roughness(
    brake: BrakeType | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Wheel roughness `L_r,VEH` of Table G-1a.

**Parameters**

| Name | Description |
| :--- | :--- |
| `brake` | The [`BrakeType`](/phonometry/reference/api/environment/cnossos-rail/#braketype) of digit 3 of the vehicle descriptor. |

**Returns:** `(wavelengths in mm, levels in dB)`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the brake type is not tabulated. |

## wheel_transfer

```python
wheel_transfer(diameter: WheelDiameter | float) -> NDArray[np.float64]
```

Wheel transfer function `L_H,VEH,i` of Table G-3b, in dB per axle.

**Parameters**

| Name | Description |
| :--- | :--- |
| `diameter` | A [`WheelDiameter`](/phonometry/reference/api/environment/cnossos-rail/#wheeldiameter) member or the diameter in mm. |

**Returns:** The 24 1/3-octave values, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the diameter is not tabulated. |

## WheelDiameter

```python
WheelDiameter(*values)
```

Columns of Table G-3b, the wheel diameter in mm, all "no measure".

## WheelMeasure

```python
WheelMeasure(*values)
```

Digit 4 of the vehicle descriptor, Table [2.3.a].
