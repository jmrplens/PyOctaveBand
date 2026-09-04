---
title: "environment.propagation.outdoor_propagation"
description: "Outdoor sound propagation: ISO 9613-2:1996 general method of calculation."
sidebar:
  label: "outdoor_propagation"
---

Outdoor sound propagation: ISO 9613-2:1996 general method of calculation.

This part of ISO 9613 predicts octave-band attenuation of sound propagating
outdoors from a point source to a receiver under conditions favourable to
propagation (moderate downwind, or the equivalent moderate temperature
inversion; ISO 9613-2:1996, clause 5). The equivalent-continuous downwind
octave-band sound pressure level is (ISO 9613-2:1996):

$$
L_{fT}(DW) = L_W + D_\mathrm{c} - A \tag{Eq. 3}
$$

with $L_W$ the octave-band sound power level, $D_\mathrm{c}$ the
directivity correction
(directivity index plus a solid-angle index `DOmega`) and `A` the
octave-band attenuation, itself a sum of physical mechanisms:

$$
A = A_\mathrm{div} + A_\mathrm{atm} + A_\mathrm{gr} + A_\mathrm{bar} + A_\mathrm{misc} \tag{Eq. 4}
$$

Implemented here are the four general terms of clause 7:

* `Adiv` geometrical divergence, $20 \log_{10}(d/d_0) + 11$ (Eq. (7));
* `Aatm` atmospheric absorption, $\alpha d$ (Eq. (8)) with `alpha`
  the ISO 9613-1 coefficient supplied by [`air_absorption`](/phonometry/reference/api/environment/air-absorption/);
* `Agr` ground effect, both the general per-region method of 7.3.1 with the
  Table 3 functions `a'/b'/c'/d'` (Eq. (9)) and the alternative simplified
  method of 7.3.2 (Eq. (10));
* `Abar` screening by a barrier, $D_z - A_\mathrm{gr}$ with the `Dz`
  diffraction
  formula of Eq. (14) including the `C2`/`C3` factors, the pathlength
  difference `z` (Eq. (16)/(17)), the meteorological factor `Kmet`
  (Eq. (18)) and the 20 dB (single) / 25 dB (double) limits.

The long-term average level follows from the meteorological correction `Cmet`
(Eq. (6), (21), (22), clause 8). `Amisc` (foliage, industrial sites, housing;
annex A) and reflections from vertical obstacles (clause 7.5) are informative and
left to the caller. Accuracy of the method is stated in Table 5 (clause 9): within
+/-1 dB to +/-3 dB for broadband noise up to 1000 m.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## atmospheric_absorption

```python
atmospheric_absorption(
    distance: float,
    frequencies: ArrayLike = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    temperature: float = 20.0,
    relative_humidity: float | None = None,
    pressure: float = 101.325,
) -> NDArray[np.float64]
```

Attenuation due to atmospheric absorption (ISO 9613-2:1996, Eq. (8)).

$A_\mathrm{atm} = \alpha d$ with `alpha` the ISO 9613-1 atmospheric
attenuation
coefficient (here in dB/m, from [`air_attenuation`](/phonometry/reference/api/environment/air-absorption/#air_attenuation))
at each octave-band midband frequency. Eq. (8) writes `alpha` in dB/km
with $A_\mathrm{atm} = \alpha_{\text{dB/km}} \, d / 1000$; the two forms
are identical.

`alpha` is evaluated at the *exact* base-10 midband frequency behind
each nominal band label (e.g. 7 943.3 Hz for the "8 kHz" band), the
convention behind the ISO 9613-2 Table 2 coefficients (they come from
ISO 9613-1 Table 1 at exact midbands; at 8 kHz the nominal-frequency
evaluation would run ~1.3 % high). Each supplied frequency is snapped to
the nearest exact midband.

**Parameters**

| Name | Description |
| :--- | :--- |
| `distance` | Source-to-receiver distance `d`, in metres. |
| `frequencies` | Octave-band midband frequencies, in hertz. |
| `temperature` | Air temperature, in degrees Celsius. |
| `relative_humidity` | Relative humidity, in percent (default 70). |
| `pressure` | Atmospheric pressure, in kilopascals. |

**Returns:** `Aatm` per band, in decibels.

## AtmosphericConditions

```python
AtmosphericConditions(
    temperature: float = 20.0,
    relative_humidity: float | None = None,
    pressure: float = 101.325,
)
```

State of the air behind the atmospheric absorption term `Aatm`.

The three quantities the ISO 9613-1 attenuation coefficient `alpha` is a
function of (ISO 9613-2:1996, Eq. (8) and Table 2). The defaults are the
reference conditions of the tabulated coefficients.

**Parameters**

| Name | Description |
| :--- | :--- |
| `temperature` | Air temperature, in degrees Celsius. |
| `relative_humidity` | Relative humidity, in percent; `None` uses 70. |
| `pressure` | Atmospheric pressure, in kilopascals. |

## Barrier

```python
Barrier(
    source_to_edge: float,
    edge_to_receiver: float,
    parallel_distance: float = 0.0,
    edge_separation: float | None = None,
    ground_reflections_by_image: bool = False,
    lateral: bool = False,
    line_of_sight_clear: bool = False,
)
```

Screening obstacle for the ISO 9613-2 barrier term (clause 7.4).

The barrier is described by the diffraction geometry that feeds the
pathlength-difference equations (16)/(17) directly, which is the cleanest
match to the `Dz` formula of Eq. (14).

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_to_edge` | Distance `dss` from the source to the (first) diffraction edge, in metres (ISO 9613-2:1996, Eq. (16)). |
| `edge_to_receiver` | Distance `dsr` from the (second) diffraction edge to the receiver, in metres. |
| `parallel_distance` | Component `a` of the source-receiver separation parallel to the barrier edge, in metres (0 for a purely 2-D section). |
| `edge_separation` | Spacing `e` between the two diffraction edges for double (thick-barrier) diffraction, in metres; `None` selects single diffraction (Eq. (16), $C_3 = 1$). When given, Eq. (17) and the `C3` factor of Eq. (15) are used with the 25 dB limit. |
| `ground_reflections_by_image` | When `True` the ground reflections are assumed to be handled separately by image sources, so $C_2 = 40$; otherwise $C_2 = 20$ (Eq. (14)). |
| `lateral` | When `True` the diffraction is around a vertical edge (Eq. (13)): $A_\mathrm{bar} = D_z$ (the ground term is not cancelled) and $K_\mathrm{met} = 1$. Default `False` selects top-edge diffraction (Eq. (12)). |
| `line_of_sight_clear` | When `True` the line of sight between source and receiver passes *above* the top edge: ISO 9613-2:1996 (text after Eq. (16)) then gives the path difference `z` a negative sign, and Eq. (14) is still evaluated (with $K_\mathrm{met} = 1$, Eq. (18)), so `Dz` falls continuously from $10 \log_{10} 3 = 4.8$ dB at grazing to 0 for deeper geometries. The edge distances stay the unsigned geometric lengths; only the sign convention of `z` changes. |

### Barrier.is_double

*property*

Whether double diffraction (Eq. (17)/(15)) applies (`e` given).

## barrier_attenuation

```python
barrier_attenuation(
    barrier: Barrier,
    distance: float,
    frequencies: ArrayLike = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
) -> NDArray[np.float64]
```

Barrier diffraction attenuation `Dz` (ISO 9613-2:1996, Eq. (14)).

$$
D_z = 10 \log_{10}\!\left[ 3 + \frac{C_2}{\lambda} \, C_3 \, z \, K_\mathrm{met} \right] ~\text{dB}
$$

with $C_2 = 20$ (or 40 when ground reflections are handled by image
sources), $C_3 = 1$ for single diffraction or Eq. (15) for double,
the pathlength difference `z` (Eq. (16)/(17)),
$\lambda = 340/f$ and the
meteorological factor `Kmet` (Eq. (18), 1 for $z \le 0$). `Dz`
is limited to 20 dB (single) or 25 dB (double). When the line of sight
passes above the top edge (`Barrier(line_of_sight_clear=True)`) `z`
takes a negative sign (ISO 9613-2:1996, text after Eq. (16)) and Eq. (14)
still applies: `Dz` falls continuously from $10 \log_{10} 3 = 4.8$ dB
at grazing ($z = 0$) towards 0 as the clearance deepens, clamped at
0 (the logarithm's argument is floored at 1 -- a barrier below the sight
line never amplifies).

**Parameters**

| Name | Description |
| :--- | :--- |
| `barrier` | Barrier geometry ([`Barrier`](/phonometry/reference/api/environment/outdoor-propagation/#barrier)). |
| `distance` | Straight-line source-to-receiver distance `d`, in metres. |
| `frequencies` | Octave-band midband frequencies, in hertz. |

**Returns:** `Dz` per band, in decibels (>= 0).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `distance` or any frequency is not positive. |

## DEFAULT_FREQUENCIES

*Constant* (`tuple`).

```python
DEFAULT_FREQUENCIES = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
```

## directivity_omega

```python
directivity_omega(
    source_height: float,
    receiver_height: float,
    projected_distance: float,
) -> float
```

Solid-angle directivity index `DOmega` (ISO 9613-2:1996, Eq. (11)).

Accounts for the apparent increase in source power from ground reflection
near the source when the alternative ground method (Eq. (10)) is used:

$$
D_\Omega = 10 \log_{10}\!\left\{ 1 + \frac{d_\mathrm{p}^2 + (h_\mathrm{s} - h_\mathrm{r})^2}{d_\mathrm{p}^2 + (h_\mathrm{s} + h_\mathrm{r})^2} \right\} ~\text{dB}
$$

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_height` | Source height `hs`, in metres. |
| `receiver_height` | Receiver height `hr`, in metres. |
| `projected_distance` | Ground-plane projected distance `dp`, in metres. |

**Returns:** `DOmega`, in decibels (0 to ~3 dB).

## DirectivityCorrection

```python
DirectivityCorrection(index: float = 0.0, d_omega: float = 0.0)
```

Directivity correction $D_\mathrm{c} = D_i + D_\Omega$ (Eq. (3)).

The two terms the standard adds to the sound power level of a point source:
the directivity index of the source itself and the solid-angle index of the
space it radiates into. Both are zero for an omnidirectional source
radiating into free space.

**Parameters**

| Name | Description |
| :--- | :--- |
| `index` | Source directivity index `Di`, in decibels. |
| `d_omega` | Solid-angle index `DOmega`, in decibels (see [`directivity_omega`](/phonometry/reference/api/environment/outdoor-propagation/#directivity_omega) for the alternative ground method). |

## geometric_divergence

```python
geometric_divergence(distance: float) -> float
```

Attenuation due to geometrical divergence (ISO 9613-2:1996, Eq. (7)).

Spherical spreading in the free field from a point source:

$$
A_\mathrm{div} = 20 \log_{10}(d/d_0) + 11~\text{dB}, \qquad d_0 = 1~\text{m}
$$

The `+11` ($= 10 \log_{10} 4\pi$) sets the sound pressure level at the
reference distance $d_0 = 1$ m from an omnidirectional point source
(Note 7).

**Parameters**

| Name | Description |
| :--- | :--- |
| `distance` | Straight-line source-to-receiver distance `d`, in metres. |

**Returns:** `Adiv`, in decibels (51 dB at 100 m, 11 dB at 1 m).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `distance` is not positive. |

## ground_attenuation

```python
ground_attenuation(
    distance: float,
    source_height: float,
    receiver_height: float,
    frequencies: ArrayLike = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    ground_source: float = 0.0,
    ground_middle: float = 0.0,
    ground_receiver: float = 0.0,
    projected_distance: float | None = None,
) -> NDArray[np.float64]
```

Ground attenuation by the general per-region method (7.3.1, Eq. (9)).

$A_\mathrm{gr} = A_\mathrm{s} + A_\mathrm{r} + A_\mathrm{m}$ (source, receiver and middle regions),
each evaluated
with the Table 3 expressions and its ground factor `G` (0 = hard, 1 =
porous, in between = porous fraction). For the source region
$G = G_\mathrm{s}$ and $h = h_\mathrm{s}$; for the receiver region
$G = G_\mathrm{r}$ and $h = h_\mathrm{r}$ (Table 3,
note 1). The middle-region term uses the overlap factor `q` of note 2:

$$
q = 0 \quad \text{if } d_\mathrm{p} \le 30 (h_\mathrm{s} + h_\mathrm{r})
$$

$$
q = 1 - \frac{30 (h_\mathrm{s} + h_\mathrm{r})}{d_\mathrm{p}} \quad \text{if } d_\mathrm{p} > 30 (h_\mathrm{s} + h_\mathrm{r})
$$

with $A_\mathrm{m} = -3q$ at 63 Hz and $A_\mathrm{m} = -3q(1 - G_\mathrm{m})$ above.

**Parameters**

| Name | Description |
| :--- | :--- |
| `distance` | Straight-line source-to-receiver distance `d`, in metres. |
| `source_height` | Source height `hs` above ground, in metres. |
| `receiver_height` | Receiver height `hr` above ground, in metres. |
| `frequencies` | Octave-band midband frequencies, in hertz. Table 3 is defined for the eight nominal octave bands 63 Hz-8 kHz only; any other requested frequency is snapped to the nearest nominal octave band for the Table 3 lookup. |
| `ground_source` | Ground factor `Gs` of the source region ([0, 1]). |
| `ground_middle` | Ground factor `Gm` of the middle region ([0, 1]). |
| `ground_receiver` | Ground factor `Gr` of the receiver region ([0, 1]). |
| `projected_distance` | Ground-plane projected distance `dp`, in metres; defaults to $\sqrt{d^2 - (h_\mathrm{s} - h_\mathrm{r})^2}$. |

**Returns:** `Agr` per band, in decibels (negative denotes a net gain).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a ground factor is outside `[0, 1]`, `distance` is not positive, a height is negative, or a frequency is not positive. |

## ground_attenuation_alternative

```python
ground_attenuation_alternative(distance: float, mean_height: float) -> float
```

Ground attenuation by the alternative A-weighted method (7.3.2, Eq. (10)).

Valid only when the A-weighted receiver level alone is of interest, the sound
propagates over porous or mostly-porous ground and is not a pure tone
(ISO 9613-2:1996, 7.3.2):

$$
A_\mathrm{gr} = 4.8 - \frac{2 h_\mathrm{m}}{d} \left[ 17 + \frac{300}{d} \right] \ge 0~\text{dB}
$$

Negative results are replaced by zero. When this method is used, add the
solid-angle index [`directivity_omega`](/phonometry/reference/api/environment/outdoor-propagation/#directivity_omega) (Eq. (11)) to `Dc` in Eq. (3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `distance` | Source-to-receiver distance `d`, in metres. |
| `mean_height` | Mean height `hm` of the propagation path above the ground ($h_\mathrm{m} = F/d$, figure 3), in metres. |

**Returns:** `Agr`, in decibels (>= 0).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `distance` is not positive. |

## GroundFactors

```python
GroundFactors(
    source: float = 0.0,
    middle: float = 0.0,
    receiver: float = 0.0,
)
```

Ground factors `G` of the three regions (ISO 9613-2:1996, clause 7.3.1).

The general ground method splits the path into a source region, a middle
region and a receiver region, each with its own factor between 0 (hard
ground: paving, water, ice, compacted ground) and 1 (porous ground: grass,
trees, farmland). The default is hard ground throughout.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source` | Ground factor `Gs` of the source region ([0, 1]). |
| `middle` | Ground factor `Gm` of the middle region ([0, 1]). |
| `receiver` | Ground factor `Gr` of the receiver region ([0, 1]). |

## mean_path_height

```python
mean_path_height(
    profile_distances: ArrayLike,
    profile_heights: ArrayLike,
    source_height: float,
    receiver_height: float,
    distance: float | None = None,
) -> float
```

Mean height `hm` of the propagation path above the ground (Figure 3).

The alternative ground method of ISO 9613-2:1996, 7.3.2 is written in one
quantity the standard defines by a drawing: $h_\mathrm{m} = F/d$,
with `F` the area between the straight source-to-receiver ray and the
ground beneath it (Figure 3). This computes that area for a ground given as
a polyline, which is the shape a terrain model reduces to once it is cut
along the vertical plane through source and receiver.

Source and receiver stand on the two ends of the profile, so the ray runs
from $g(0) + h_\mathrm{s}$ to $g(d_\mathrm{p}) + h_\mathrm{r}$
and the profile fixes both the ground-projected length and the height
difference the slant distance is built from.

**Parameters**

| Name | Description |
| :--- | :--- |
| `profile_distances` | Ground-projected distance of each profile point from the source, in metres, strictly increasing. |
| `profile_heights` | Ground height at each profile point, in metres, on any one datum (only differences are used). |
| `source_height` | Source height `hs` above the ground at the first profile point, in metres. |
| `receiver_height` | Receiver height `hr` above the ground at the last profile point, in metres. |
| `distance` | Source-to-receiver distance `d` to divide the area by, in metres; `None` takes the slant distance the profile implies. |

**Returns:** `hm`, in metres.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the two sequences disagree in length, describe fewer than two points, the distances are not strictly increasing, `source_height` or `receiver_height` is negative or not finite, or `distance` is not positive and finite. The profile heights themselves are unrestricted: they are read on any datum, so a ground below it is a negative number rather than a mistake. |

:::note
Ground rising above the ray subtracts from `F`, exactly as the area of
Figure 3 is drawn. A path the ground screens outright is no longer a
7.3.2 path at all: its obstacle belongs in the screening term
$A_\mathrm{bar}$ of 7.4.
:::

## meteorological_correction

```python
meteorological_correction(
    projected_distance: float,
    source_height: float,
    receiver_height: float,
    c0: float,
) -> float
```

Meteorological correction `Cmet` (ISO 9613-2:1996, Eq. (21)/(22)).

$$
C_\mathrm{met} = 0 \quad \text{if } d_\mathrm{p} \le 10 (h_\mathrm{s} + h_\mathrm{r})
$$

$$
C_\mathrm{met} = C_0 \left[ 1 - \frac{10 (h_\mathrm{s} + h_\mathrm{r})}{d_\mathrm{p}} \right] \quad \text{if } d_\mathrm{p} > 10 (h_\mathrm{s} + h_\mathrm{r})
$$

`C0` (dB) reflects local wind and temperature-gradient statistics; practical
values lie in 0..~5 dB (note 22). Subtract `Cmet` from `LAT(DW)` for the
long-term average level (Eq. (6)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `projected_distance` | Ground-plane projected distance `dp`, in metres. |
| `source_height` | Source height `hs`, in metres. |
| `receiver_height` | Receiver height `hr`, in metres. |
| `c0` | Meteorological factor `C0`, in decibels. |

**Returns:** `Cmet`, in decibels (>= 0 for $C_0 \ge 0$).

## outdoor_propagation_attenuation

```python
outdoor_propagation_attenuation(
    distance: float,
    source_height: float,
    receiver_height: float,
    frequencies: ArrayLike = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    ground_source: float = 0.0,
    ground_middle: float = 0.0,
    ground_receiver: float = 0.0,
    barrier: Barrier | None = None,
    temperature: float = 20.0,
    relative_humidity: float | None = None,
    pressure: float = 101.325,
    projected_distance: float | None = None,
) -> OutdoorAttenuation
```

Total octave-band outdoor attenuation (ISO 9613-2:1996, Eq. (4)).

Assembles the four general terms of clause 7 into
$A = A_\mathrm{div} + A_\mathrm{atm} + A_\mathrm{gr} + A_\mathrm{bar}$ (the informative
`Amisc` is omitted). The ground effect uses the
general per-region method (7.3.1). With a barrier, the top-edge insertion
loss $A_\mathrm{bar} = D_z - A_\mathrm{gr}$ (Eq. (12)) folds the ground effect of
the screened path into `Dz` (note 13); for a lateral (vertical-edge)
barrier $A_\mathrm{bar} = D_z$
(Eq. (13)) and the ground term is retained.

**Parameters**

| Name | Description |
| :--- | :--- |
| `distance` | Straight-line source-to-receiver distance `d`, in metres. |
| `source_height` | Source height `hs` above ground, in metres. |
| `receiver_height` | Receiver height `hr` above ground, in metres. |
| `frequencies` | Octave-band midband frequencies, in hertz. The ground term snaps each frequency to the nearest nominal octave band (Table 3 is octave-band only) and the atmospheric term evaluates the exact base-10 midband behind it (see [`atmospheric_absorption`](/phonometry/reference/api/environment/outdoor-propagation/#atmospheric_absorption)). |
| `ground_source` | Ground factor `Gs` of the source region ([0, 1], 0 = hard, 1 = porous). |
| `ground_middle` | Ground factor `Gm` of the middle region ([0, 1]). |
| `ground_receiver` | Ground factor `Gr` of the receiver region ([0, 1]). |
| `barrier` | Optional screening obstacle ([`Barrier`](/phonometry/reference/api/environment/outdoor-propagation/#barrier)). |
| `temperature` | Air temperature, in degrees Celsius. |
| `relative_humidity` | Relative humidity, in percent (default 70). |
| `pressure` | Atmospheric pressure, in kilopascals. |
| `projected_distance` | Ground-plane projected distance `dp`, in metres; defaults to $\sqrt{d^2 - (h_\mathrm{s} - h_\mathrm{r})^2}$. |

**Returns:** [`OutdoorAttenuation`](/phonometry/reference/api/environment/outdoor-propagation/#outdoorattenuation) with the per-band term breakdown.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `distance` is not positive. |

## OutdoorAttenuation

```python
OutdoorAttenuation(
    frequencies: NDArray[np.float64],
    a_div: NDArray[np.float64],
    a_atm: NDArray[np.float64],
    a_gr: NDArray[np.float64],
    a_bar: NDArray[np.float64],
    a_total: NDArray[np.float64],
    d_omega: NDArray[np.float64],
)
```

Per-octave-band ISO 9613-2 attenuation breakdown (clause 7).

Every array is aligned with `frequencies`. The terms sum, band by band,
to `a_total` (ISO 9613-2:1996, Eq. (4) without the informative
`Amisc`), so users can see the divergence, atmospheric, ground and barrier
contributions separately.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Nominal octave-band midband frequencies, in hertz. |
| `a_div` | Geometrical divergence `Adiv` (Eq. (7)), in dB, per band (identical across bands). |
| `a_atm` | Atmospheric absorption `Aatm` (Eq. (8)), in dB, per band. |
| `a_gr` | Ground effect `Agr` (Eq. (9) or (10)), in dB, per band. A negative value denotes a net gain from ground reflection. |
| `a_bar` | Screening `Abar` (Eq. (12)/(13)), in dB, per band (>= 0). |
| `a_total` | Total attenuation `A` (Eq. (4)), in dB, per band. |
| `d_omega` | Solid-angle directivity index `DOmega` (Eq. (11)), in dB; non-zero only for the alternative ground method of 7.3.2. |

### OutdoorAttenuation.plot()

```python
OutdoorAttenuation.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the stacked per-band attenuation terms with the total.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### OutdoorAttenuation.report()

```python
OutdoorAttenuation.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
    source_emission: SourceEmission | None = None,
) -> str
```

Render a one-page ISO 9613-2 outdoor-propagation prediction fiche.

Writes a prediction sheet (clearly labelled a prediction, not a
measurement) laid out like an environmental-noise propagation
calculation: the standard-basis line naming ISO 9613-2:1996 (general
method, conditions favourable to propagation), an optional metadata
header (source/situation, client, receiver position, meteorological
conditions, date), a per-band table of the attenuation terms
(`Adiv`, `Aatm`, `Agr`, `Abar` and the total `A`) and the
attenuation-breakdown plot, closed by a boxed single result and a footer
identity/disclaimer block.

When a `source_emission` is supplied, the fiche also lists the source
power level `Lw` and the composed downwind level `LfT(DW)` per band
and boxes the A-weighted downwind level `LAT(DW)` at the receiver, with
an optional PASS/FAIL verdict against a declared limit level (a lower
level is better). Without it the fiche boxes the octave-band range of
the total attenuation `A`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`specimen` the source/situation, `client`, `test_room` the receiver position), the `temperature` / `relative_humidity` / `pressure` conditions and the footer identity. A supplied `requirement` is read as the maximum acceptable A-weighted downwind level in dB (used only when a `source_emission` is given). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When True and a `source_emission` is supplied, the per-band table adds the A-weighted band level (`LfT(DW)` plus the band A-weighting), whose energy sum is the boxed `LAT(DW)`. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |
| `source_emission` | Optional [`SourceEmission`](/phonometry/reference/api/environment/outdoor-propagation/#sourceemission) (the source sound power `Lw` and directivity, plus an optional meteorological correction) that turns the attenuation breakdown into the boxed A-weighted downwind level at the receiver. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`, `language` is unknown, or a supplied `source_emission` sound power does not match the number of frequency bands. |
| ImportError | If reportlab or matplotlib is not installed (`pip install "phonometry[report,plot]"`). |

## predicted_receiver_level

```python
predicted_receiver_level(
    sound_power_level: ArrayLike,
    geometry: PropagationGeometry,
    *,
    frequencies: ArrayLike = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    ground: GroundFactors | None = None,
    barrier: Barrier | None = None,
    atmosphere: AtmosphericConditions | None = None,
    directivity: DirectivityCorrection | None = None,
    c0: float | None = None,
) -> NDArray[np.float64]
```

Predicted octave-band receiver level (ISO 9613-2:1996, Eq. (3)/(6)).

Composes the downwind octave-band sound pressure level:

$$
L_{fT}(DW) = L_W + D_\mathrm{c} - A, \qquad D_\mathrm{c} = D_i + D_\Omega
$$

from the total attenuation [`outdoor_propagation_attenuation`](/phonometry/reference/api/environment/outdoor-propagation/#outdoor_propagation_attenuation). When `c0`
is given, the meteorological correction `Cmet` (Eq. (21)/(22)) is subtracted
band by band to approximate the long-term average level `LfT(LT)` (Eq. (6));
the standard applies `Cmet` to the A-weighted level, so this is a per-band
convenience.

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_power_level` | Octave-band sound power level `Lw`, in decibels (re 1 pW), one value per frequency. |
| `geometry` | Source-to-receiver geometry of the path ([`PropagationGeometry`](/phonometry/reference/api/environment/outdoor-propagation/#propagationgeometry)): the distance `d`, the two heights and the optional projected distance `dp`. |
| `frequencies` | Octave-band midband frequencies, in hertz. |
| `ground` | Ground factors of the three regions ([`GroundFactors`](/phonometry/reference/api/environment/outdoor-propagation/#groundfactors)); `None` is hard ground throughout. |
| `barrier` | Optional screening obstacle ([`Barrier`](/phonometry/reference/api/environment/outdoor-propagation/#barrier)). |
| `atmosphere` | Air temperature, humidity and pressure behind the atmospheric absorption term ([`AtmosphericConditions`](/phonometry/reference/api/environment/outdoor-propagation/#atmosphericconditions)); `None` is the reference air of the tabulated coefficients. |
| `directivity` | Directivity correction `Dc = Di + DOmega` ([`DirectivityCorrection`](/phonometry/reference/api/environment/outdoor-propagation/#directivitycorrection)); `None` is an omnidirectional source in free space ($D_\mathrm{c} = 0$). |
| `c0` | Meteorological factor `C0`, in decibels; `None` returns the downwind level `LfT(DW)` ($C_\mathrm{met} = 0$). |

**Returns:** Predicted octave-band level per frequency, in decibels.

## PropagationGeometry

```python
PropagationGeometry(
    distance: float,
    source_height: float,
    receiver_height: float,
    projected_distance: float | None = None,
)
```

Source-to-receiver geometry of the propagation path (ISO 9613-2:1996).

The three lengths every term of the method needs: the straight-line
distance `d` of the divergence and atmospheric terms (Eq. (7)/(8)), and
the two heights above the ground the ground effect of clause 7.3.1 and the
meteorological correction of clause 8 are written in.

**Parameters**

| Name | Description |
| :--- | :--- |
| `distance` | Straight-line source-to-receiver distance `d`, in metres. |
| `source_height` | Source height `hs` above ground, in metres. |
| `receiver_height` | Receiver height `hr` above ground, in metres. |
| `projected_distance` | Ground-plane projected distance `dp`, in metres; `None` defaults to $\sqrt{d^2 - (h_\mathrm{s} - h_\mathrm{r})^2}$. |

## region_ground_factors

```python
region_ground_factors(
    segment_lengths: ArrayLike,
    segment_ground_factors: ArrayLike,
    source_height: float,
    receiver_height: float,
) -> GroundFactors
```

Ground factors of the three regions from a path crossing several grounds.

ISO 9613-2:1996, 7.3.1 splits the ground projection of the path into a
source region 30 $h_\mathrm{s}$ long, a receiver region 30
$h_\mathrm{r}$ long and whatever middle region is left between them,
and asks for one ground factor per region. Where the path runs over ground
of more than one kind, the standard does not say how to reduce the several
factors to the three the method wants. ISO/TR 17534-3:2015, 6.2.5 settles
it for quality-assured software: each region takes the mean of `G`
weighted by the length of the ground projection it covers.

The ground projection of the path is the concatenation of the segments, so
$d_\mathrm{p}$ is their total length; give the segments in order from
the source.

**Parameters**

| Name | Description |
| :--- | :--- |
| `segment_lengths` | Length of each segment of the ground projection, in metres, ordered from the source; each must be positive. |
| `segment_ground_factors` | Ground factor `G` of each segment ([0, 1]), aligned with `segment_lengths`. |
| `source_height` | Source height `hs` above ground, in metres. |
| `receiver_height` | Receiver height `hr` above ground, in metres. |

**Returns:** The [`GroundFactors`](/phonometry/reference/api/environment/outdoor-propagation/#groundfactors) of the source, middle and receiver regions.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the two sequences disagree in length or are empty, a length is not positive, a factor is outside `[0, 1]`, or a height is negative or not finite. |

:::note
Where the two outer regions meet or overlap there is no middle region,
and the returned [`GroundFactors.middle`](/phonometry/reference/api/environment/outdoor-propagation/#groundfactors) is the mean over the whole
projection. It does not reach the result: the overlap factor `q` of
Table 3, note 2 is nought over exactly that range, which drops the
middle-region term altogether.
:::

## SourceEmission

```python
SourceEmission(
    sound_power_level: ArrayLike,
    directivity_index: float = 0.0,
    d_omega: float = 0.0,
    cmet: float | None = None,
)
```

Source emission terms for the ISO 9613-2 downwind receiver level (Eq. (3)).

Passed to [`OutdoorAttenuation.report`](/phonometry/reference/api/environment/outdoor-propagation/#outdoorattenuationreport) so the prediction fiche can box
the A-weighted downwind level at the receiver from an octave-band
attenuation breakdown. The level is composed as
$L_{fT}(DW) = L_W + D_\mathrm{c} - A$
with the directivity correction $D_\mathrm{c} = D_i + D_\Omega$
(ISO 9613-2:1996, Eq. (3)); an optional meteorological correction `cmet`
is subtracted for the long-term average level (Eq. (6)).

This report-time object keeps the emission out of
[`outdoor_propagation_attenuation`](/phonometry/reference/api/environment/outdoor-propagation/#outdoor_propagation_attenuation) (which stays purely an attenuation
calculation), so the receiver level is a presentation concern of the fiche.

**Parameters**

| Name | Description |
| :--- | :--- |
| `sound_power_level` | Octave-band source sound power level `Lw` (dB re 1 pW), one value per band of the attenuation result. |
| `directivity_index` | Source directivity index `Di`, in decibels. |
| `d_omega` | Solid-angle index `DOmega`, in decibels (see [`directivity_omega`](/phonometry/reference/api/environment/outdoor-propagation/#directivity_omega) for the alternative ground method). |
| `cmet` | Optional meteorological correction `Cmet` (dB), obtained from [`meteorological_correction`](/phonometry/reference/api/environment/outdoor-propagation/#meteorological_correction); `None` reports the downwind level `LfT(DW)` directly ($C_\mathrm{met} = 0$). |
