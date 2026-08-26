---
title: "environment.propagation.ground_barriers"
description: "Spherical-wave ground effect and advanced barrier diffraction."
sidebar:
  label: "ground_barriers"
---

Spherical-wave ground effect and advanced barrier diffraction.

This module extends the tabulated ground and barrier terms of ISO 9613-2 (see
[`phonometry.environment.propagation.outdoor_propagation`](/phonometry/reference/api/environment/outdoor-propagation/)) with the underlying wave
acoustics: the spherical-wave reflection coefficient of a finite-impedance
ground and the wave-theoretic diffraction of a screen, both in a homogeneous
(non-refracting, non-turbulent) atmosphere.

Ground effect (Weyl-Van der Pol)
--------------------------------
The sound field of a point source above a locally reacting ground is the sum of
a direct wave and a reflected wave weighted by the spherical-wave reflection
coefficient `Q` (Attenborough & Van Renterghem, *Predicting Outdoor Sound*
2e, 2021, Eq. (2.40a); Salomons, *Computational Atmospheric Acoustics*, 2001,
Eq. (3.2)):

$$
p = \frac{e^{ikR_1}}{4 \pi R_1} + Q \, \frac{e^{ikR_2}}{4 \pi R_2}
$$

with $R_1$ the source-receiver distance, $R_2$ the image-source
distance and (Attenborough Eq. (2.40c) / Salomons Eq. (D.58)):

$$
Q = R_\mathrm{p} + (1 - R_\mathrm{p}) F(w)
$$

$$
R_\mathrm{p} = \frac{Z \cos\theta - 1}{Z \cos\theta + 1} \tag{Salomons Eq. D.59}
$$

$$
F(w) = 1 + i \sqrt{\pi} \, w \exp(-w^2) \operatorname{erfc}(-i w) \tag{Salomons Eq. D.60}
$$

$$
w = \sqrt{i k R_2 / 2} \, \left( \cos\theta + \frac{1}{Z} \right) \tag{Salomons Eq. D.57}
$$

Here `Z` is the normalized (by `rho c`) surface impedance of the ground,
`theta` is the angle of incidence from the ground normal
($\cos\theta = (h_\mathrm{s} + h_\mathrm{r})/R_2$) and $F(w)$ is the boundary-loss
factor written through the scaled complementary error function
$\exp(-w^2) \operatorname{erfc}(-i w)$, i.e.
the Faddeeva function `scipy.special.wofz`. The relative sound level (the
"excess attenuation", dB re free field) is:

$$
\Delta L = 20 \log_{10} \left| 1 + Q \, \frac{R_1}{R_2} \, e^{i k (R_2 - R_1)} \right| \tag{Salomons Eq. 3.4}
$$

Limits reproduced by the implementation: an acoustically hard ground
($|Z| \to \infty$) gives $R_\mathrm{p} \to 1$, so
$(1 - R_\mathrm{p}) \to 0$ and $Q \to 1$
regardless of the boundary loss (the ground wave vanishes), and
$\Delta L$ reaches
`+6 dB` in phase (Salomons Sec. 3.4); at grazing incidence
($h_\mathrm{s}, h_\mathrm{r} \to 0$, $\cos\theta \to 0$) $R_\mathrm{p} \to -1$; and as
the range grows ($R_2 \to \infty$) $|w| \to \infty$ and
$F \to 0$. The ground impedance is taken in
the $e^{-i \omega t}$ time convention of Salomons, in which a passive
ground has $\operatorname{Im}(Z) > 0$; it may be supplied directly or
derived from the porous models of `phonometry.materials`
([`delany_bazley`](/phonometry/reference/api/materials/porous/#delany_bazley) / [`miki`](/phonometry/reference/api/materials/porous/#miki)),
which model a semi-infinite porous ground whose surface impedance equals the
characteristic impedance of the medium. The materials domain works in the
opposite $e^{+j \omega t}$ convention ($\operatorname{Im}(Z) < 0$
for a passive medium), so
any impedance obtained from a porous model is conjugated internally before it
enters the formulas above.

Barrier diffraction
-------------------
Three levels of screening beyond the ISO 9613-2 `Dz` term are provided:

* the Kurze-Anderson closed form in the Fresnel number `N` (Bies, Hansen &
  Howard, *Engineering Noise Control* 5e, 2017, Eq. (5.138); Kurze & Anderson,
  1971),
  $\Delta = 5 + 20 \log_{10}\!\left[ \sqrt{2 \pi N} / \tanh\sqrt{2 \pi N} \right]$,
  which tends to `5 dB` at $N \to 0$ and stays within about 1.5 dB of
  Maekawa's point-source curve for all `N` (a very good fit for
  $N > 0.5$);

* the wave-theoretic insertion loss of a rigid thin screen (half-plane), the
  flat-wedge limit of the MacDonald / Hadden & Pierce solution
  (Attenborough Eqs. (9.19)-(9.20)), obtained from the auxiliary Fresnel
  functions and correctly giving `6 dB` at the shadow boundary (the field is
  halved);

* the coherent barrier-on-ground model that combines the four source-image /
  receiver-image diffracted paths with the spherical-wave reflection
  coefficient `Q` above (Attenborough Ch. 9; Bies Sec. 5.3.5), which shows the
  ground-barrier interference structure a purely energetic sum cannot.

Thick barriers (or two parallel thin screens) are handled by the double-edge
Fresnel number $N = (2/\lambda)(A + B + e - d)$ (Bies Eq. (5.157)).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## barrier_insertion_loss

```python
barrier_insertion_loss(
    frequencies: ArrayLike,
    source_height: float,
    barrier_distance: float,
    barrier_height: float,
    receiver_distance: float,
    receiver_height: float,
    *,
    method: Literal['kurze_anderson', 'exact'] = 'exact',
    thickness: float | None = None,
    ground_impedance: ArrayLike | PorousMediumResult | None = None,
    ground_flow_resistivity: float | None = None,
    ground_model: Literal['delany_bazley', 'miki'] = 'delany_bazley',
    speed_of_sound: float = 343.0,
    air_density: float = 1.205,
) -> BarrierInsertionLoss
```

Insertion loss of a thin, thick or ground-coupled barrier.

The 2-D geometry places the source at `(0, source_height)`, the (near)
diffraction edge at `(barrier_distance, barrier_height)` and the receiver
at `(receiver_distance, receiver_height)`. Three models are available:

* `method="kurze_anderson"`: the closed form
  [`kurze_anderson_attenuation`](/phonometry/reference/api/environment/ground-barriers/#kurze_anderson_attenuation) of the Fresnel number
  [`fresnel_number`](/phonometry/reference/api/environment/ground-barriers/#fresnel_number) (Bies Eqs. (5.134)/(5.138)); with `thickness` the
  double-edge Fresnel number $N = (2/\lambda)(A + B + e - d)$ of
  Bies Eq. (5.157) is used, `e` being the top width.
* `method="exact"` without ground: the wave-theoretic insertion loss of the
  rigid thin screen (`_screen_field`, MacDonald / Hadden & Pierce),
  $\mathrm{IL} = 20 \log_{10} |p_{\text{free}} / p_{\text{diffracted}}|$.
* `method="exact"` with a ground (`ground_impedance` or
  `ground_flow_resistivity`): the coherent four-path model. The field with
  the barrier sums the four source-image / receiver-image diffracted paths,
  each ground reflection weighted by the spherical-wave coefficient `Q`
  ([`spherical_reflection_coefficient`](/phonometry/reference/api/environment/ground-barriers/#spherical_reflection_coefficient)); the field without the barrier
  is the two-ray ground field. This exposes the ground-barrier interference
  structure (Attenborough Ch. 9; Bies Sec. 5.3.5). As a first-order
  simplification a single `Q` (evaluated over the overall
  source-receiver geometry) weights every bounce rather than a separate
  coefficient per image path; the model is coherent and reciprocal but not
  a full boundary-element solution.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in hertz. |
| `source_height` | Source height, in metres. |
| `barrier_distance` | Horizontal source-to-barrier distance, in metres. |
| `barrier_height` | Barrier (edge) height, in metres. |
| `receiver_distance` | Horizontal source-to-receiver distance, in metres (`> barrier_distance`). |
| `receiver_height` | Receiver height, in metres. |
| `method` | `"kurze_anderson"` or `"exact"`. |
| `thickness` | Top width `e` of a thick barrier (double diffraction), in metres; `None` for a thin screen. |
| `ground_impedance` | Normalized ground impedance for the coherent ground model (`"exact"` only), in the $e^{-i \omega t}$ convention ($\operatorname{Im}(Z) > 0$ for a passive ground); a `PorousMediumResult` is conjugated internally from the materials' $e^{+j \omega t}$ convention. |
| `ground_flow_resistivity` | Effective flow resistivity `sigma` (Pa s/m2) for the ground model, as an alternative to `ground_impedance`. |
| `ground_model` | Porous model for `ground_flow_resistivity`. |
| `speed_of_sound` | Speed of sound `c`, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |

**Returns:** A [`BarrierInsertionLoss`](/phonometry/reference/api/environment/ground-barriers/#barrierinsertionloss).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | On a non-positive/ordered geometry, or if a ground is requested with `method="kurze_anderson"`. |

## BarrierInsertionLoss

```python
BarrierInsertionLoss(
    frequencies: Real,
    insertion_loss: Real,
    fresnel_number: Real,
    method: str,
    ground: bool,
    source_height: float | None = None,
    barrier_distance: float | None = None,
    barrier_height: float | None = None,
    receiver_distance: float | None = None,
    receiver_height: float | None = None,
    thickness: float | None = None,
)
```

Per-frequency barrier insertion loss (IL vs frequency).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in hertz. |
| `insertion_loss` | Insertion loss $\mathrm{IL} = 20 \log_{10} \lvert p_{\text{without}} / p_{\text{with}} \rvert$, in decibels, per frequency. |
| `fresnel_number` | Fresnel number `N` per frequency (single-edge geometry; the double-edge `N` for a thick barrier). |
| `method` | Diffraction model used, and the tag the fiche and the plot name it by: `"kurze_anderson"` or `"exact"`, the two the library implements. Any other tag is refused at construction. |
| `ground` | Whether the coherent four-path ground model was applied. |
| `source_height` | Source height the loss was computed for, in metres, retained (with the other five geometry fields) so `plot_geometry` can draw the section; appended after the original fields and `None` for hand-built results. |
| `barrier_distance` | Source-to-barrier horizontal distance, in metres. |
| `barrier_height` | Barrier height, in metres. |
| `receiver_distance` | Source-to-receiver horizontal distance, in metres. |
| `receiver_height` | Receiver height, in metres. |
| `thickness` | Thick-barrier top width, in metres, or `None`. |

### BarrierInsertionLoss.plot()

```python
BarrierInsertionLoss.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the insertion loss versus frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### BarrierInsertionLoss.plot_geometry()

```python
BarrierInsertionLoss.plot_geometry(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the source-barrier-receiver section to scale.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the result does not retain its geometry. |

### BarrierInsertionLoss.report()

```python
BarrierInsertionLoss.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a one-page barrier insertion-loss prediction fiche to a PDF.

Writes a prediction sheet (clearly labelled a prediction, not a
measurement) laid out like an outdoor-noise barrier calculation: the
standard-basis line naming the diffraction model used (the
wave-theoretic rigid-screen model for `method="exact"` or the
Kurze-Anderson closed form for `method="kurze_anderson"`, a
wave-acoustics complement to the ISO 9613-2 screening term), an optional
metadata header (source/situation, client, receiver position, date), a
per-band table of the insertion loss `IL` (and, in verbose mode, the
Fresnel number `N`), the insertion-loss spectrum plot, a boxed mean
insertion loss over the octave bands, an optional PASS/FAIL verdict
against a declared minimum required insertion loss (a higher insertion
loss is better) and a footer identity/disclaimer block.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header identity (`specimen` the source/situation, `client`, `test_room` the receiver position) and the footer identity. A supplied `requirement` is read as the minimum required mean insertion loss in dB. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When True, the per-band table adds the Fresnel number `N` column. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab or matplotlib is not installed (`pip install "phonometry[report,plot]"`). |

## fresnel_number

```python
fresnel_number(
    source_to_edge: float,
    edge_to_receiver: float,
    direct_distance: float,
    frequencies: ArrayLike,
    speed_of_sound: float = 343.0,
) -> Real
```

Fresnel number $N = (2/\lambda)(A + B - d)$ (Bies Eq. (5.134)).

`A` and `B` are the two segments of the shortest source-edge-receiver
path and `d` is the straight source-receiver distance. `N` is positive
when the receiver is in the shadow zone ($A + B > d$) and negative
in the bright zone.

**Parameters**

| Name | Description |
| :--- | :--- |
| `source_to_edge` | Path segment `A` from source to edge, in metres. |
| `edge_to_receiver` | Path segment `B` from edge to receiver, in metres. |
| `direct_distance` | Straight source-receiver distance `d`, in metres. |
| `frequencies` | Frequencies, in hertz. |
| `speed_of_sound` | Speed of sound `c`, in m/s. |

**Returns:** Fresnel number `N` per frequency.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a distance is not positive. |

## ground_effect

```python
ground_effect(
    frequencies: ArrayLike,
    source_height: float,
    receiver_height: float,
    distance: float,
    *,
    impedance: ArrayLike | PorousMediumResult,
    model: Literal['delany_bazley', 'miki'] = ...,
    speed_of_sound: float = ...,
    air_density: float = ...,
) -> SphericalGroundResult

ground_effect(
    frequencies: ArrayLike,
    source_height: float,
    receiver_height: float,
    distance: float,
    *,
    flow_resistivity: float,
    model: Literal['delany_bazley', 'miki'] = ...,
    speed_of_sound: float = ...,
    air_density: float = ...,
) -> SphericalGroundResult
```

Spherical-wave ground effect above a finite-impedance ground.

Assembles the two-ray field
$p = e^{ikR_1}/(4 \pi R_1) + Q \, e^{ikR_2}/(4 \pi R_2)$
with the spherical-wave reflection coefficient `Q` of
[`spherical_reflection_coefficient`](/phonometry/reference/api/environment/ground-barriers/#spherical_reflection_coefficient) and reports the relative sound
level
$\Delta L = 20 \log_{10}\left| 1 + Q (R_1/R_2) e^{i k (R_2 - R_1)} \right|$ (Salomons Eq. (3.4)),
i.e. the level re the free field.

The ground surface impedance is either supplied through `impedance` (a
normalized complex array/scalar, or a
[`PorousMediumResult`](/phonometry/reference/api/materials/porous/#porousmediumresult)) or derived from an
effective `flow_resistivity` (in Pa s/m2) via the `model` porous model
of the materials domain. Exactly one of the two must be given.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in hertz. |
| `source_height` | Source height `hs`, in metres. |
| `receiver_height` | Receiver height `hr`, in metres. |
| `distance` | Horizontal source-receiver distance, in metres. |
| `impedance` | Normalized ground impedance ($e^{-i \omega t}$ convention, $\operatorname{Im}(Z) > 0$ for a passive ground), or a `PorousMediumResult` (which is conjugated internally from the materials' $e^{+j \omega t}$ convention). |
| `flow_resistivity` | Effective flow resistivity `sigma` (Pa s/m2); grassland is about `2e5` (Salomons Sec. 3.1). The porous model raises a [`PorousAbsorberWarning`](/phonometry/reference/api/materials/porous/#porousabsorberwarning) when the lowest bands fall below its published fit range $0.01 < \rho f / \sigma < 1$ (it still extrapolates a value there). |
| `model` | Porous model for `flow_resistivity` (`"delany_bazley"` or `"miki"`). |
| `speed_of_sound` | Speed of sound `c`, in m/s. |
| `air_density` | Air density `rho`, in kg/m3. |

**Returns:** A [`SphericalGroundResult`](/phonometry/reference/api/environment/ground-barriers/#sphericalgroundresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If neither or both of `impedance`/`flow_resistivity` are given, a height is negative, or the distance is not positive. |

## kurze_anderson_attenuation

```python
kurze_anderson_attenuation(fresnel_number: ArrayLike) -> Real
```

Kurze-Anderson barrier attenuation (Bies Eq. (5.138); Kurze & Anderson, 1971).

$$
\Delta = 5 + 20 \log_{10}\!\left[ \frac{\sqrt{2 \pi N}} {\tanh\sqrt{2 \pi N}} \right] \qquad \text{dB}
$$

For $N \to 0$ the ratio tends to 1 and $\Delta \to 5$ dB; for
$N < 0$
(bright zone) the square root is imaginary and `tanh` becomes `tan`, so
the expression continues smoothly until, below $N = -0.2$ (the
illuminated-zone limit of Maekawa's curve), the diffraction is taken as
negligible (0 dB) rather than let the closed form oscillate through the
tangent poles. It stays within about 1.5 dB of Maekawa's point-source curve
for all `N` (a very good fit for $N > 0.5$). The result is clamped
at 0 dB (a barrier never amplifies).

**Parameters**

| Name | Description |
| :--- | :--- |
| `fresnel_number` | Fresnel number `N` (scalar or array). |

**Returns:** Attenuation `Delta`, in decibels (>= 0), matching the input shape.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-numeric or non-finite `fresnel_number`. |

## plot_barrier_geometry

```python
plot_barrier_geometry(
    ax: Axes | None = None,
    *,
    source_height: float,
    barrier_distance: float,
    barrier_height: float,
    receiver_distance: float,
    receiver_height: float,
    thickness: float | None = None,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the source-barrier-receiver section to scale.

Ground line, thin (or thick) screen, the direct path cut by the screen
and the diffracted path over the top edge(s), with the path-length
difference annotated. Distances follow
[`barrier_insertion_loss`](/phonometry/reference/api/environment/ground-barriers/#barrier_insertion_loss):
`receiver_distance` is horizontal from the source.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `source_height` | Source height above ground, in metres. |
| `barrier_distance` | Source-to-barrier horizontal distance, in metres. |
| `barrier_height` | Barrier height, in metres. |
| `receiver_distance` | Source-to-receiver horizontal distance, in metres (> `barrier_distance`). |
| `receiver_height` | Receiver height above ground, in metres. |
| `thickness` | Barrier top width, in metres; `None` draws a thin screen. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the barrier rectangle. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a height that is not finite and non-negative, a distance that is not finite and positive, a receiver no further than the barrier, or a non-positive thickness. |

## spherical_reflection_coefficient

```python
spherical_reflection_coefficient(
    frequencies: ArrayLike,
    normalized_impedance: ArrayLike,
    source_height: float,
    receiver_height: float,
    distance: float,
    speed_of_sound: float = 343.0,
) -> Complex
```

Spherical-wave reflection coefficient `Q` (Weyl-Van der Pol).

Implements $Q = R_\mathrm{p} + (1 - R_\mathrm{p}) F(w)$ (Attenborough Eq. (2.40c);
Salomons Eq. (D.58)) with the plane-wave coefficient `Rp` (Eq. (D.59)),
the boundary-loss factor
$F(w) = 1 + i \sqrt{\pi} \, w \exp(-w^2) \operatorname{erfc}(-i w)$
(Eq. (D.60), evaluated through `scipy.special.wofz`) and the
numerical distance
$w = \sqrt{i k R_2 / 2} \, (\cos\theta + 1/Z)$ (Eq. (D.57)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in hertz. |
| `normalized_impedance` | Ground surface impedance normalized by `rho c` (complex, per frequency or scalar), in the $e^{-i \omega t}$ time convention (a passive ground has $\operatorname{Im}(Z) > 0$). |
| `source_height` | Source height `hs` above the ground, in metres. |
| `receiver_height` | Receiver height `hr` above the ground, in metres. |
| `distance` | Horizontal source-receiver distance, in metres. |
| `speed_of_sound` | Speed of sound `c`, in m/s. |

**Returns:** Complex `Q` per frequency.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a height is negative, the distance is not positive, the impedance is zero, or its shape does not match the frequencies. |

## SphericalGroundResult

```python
SphericalGroundResult(
    frequencies: Real,
    excess_attenuation: Real,
    reflection_coefficient: Complex,
    plane_reflection_coefficient: Complex,
    boundary_loss: Complex,
    normalized_impedance: Complex,
    r_direct: float,
    r_reflected: float,
)
```

Spherical-wave ground-effect result (Weyl-Van der Pol).

Every array is aligned with `frequencies`.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in hertz. |
| `excess_attenuation` | Relative sound level $\Delta L$ (dB re free field, Salomons Eq. (3.4)); positive is enhancement (up to +6 dB over hard ground), negative is the ground-effect dip. |
| `reflection_coefficient` | Spherical-wave reflection coefficient `Q` (complex, Attenborough Eq. (2.40c)). |
| `plane_reflection_coefficient` | Plane-wave reflection coefficient `Rp` (complex, Salomons Eq. (D.59)). |
| `boundary_loss` | Boundary-loss factor `F(w)` (complex, Eq. (D.60)). |
| `normalized_impedance` | Normalized surface impedance `Z` used. |
| `r_direct` | Direct source-receiver distance `R1`, in metres. |
| `r_reflected` | Image-source distance `R2`, in metres. |

### SphericalGroundResult.plot()

```python
SphericalGroundResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the excess attenuation $\Delta L$ versus frequency.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.
