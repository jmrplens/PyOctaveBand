---
title: "building.prediction.masonry_cavity_wall"
description: "Wall ties in masonry cavity walls: the structural bridge across the cavity (Hopkins 2007, Sections 3.11.3.2 and 4.3.5.4.1)."
sidebar:
  label: "masonry_cavity_wall"
---

Wall ties in masonry cavity walls: the structural bridge across the cavity
(Hopkins 2007, Sections 3.11.3.2 and 4.3.5.4.1).

A masonry cavity wall is a double leaf, and the textbook double-leaf prediction
([`phonometry.double_wall_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#double_wall_transmission_loss)) treats the cavity as pure
air. Real cavity walls are stitched together by **wall ties** every few
courses. Those ties do two things the air-only model cannot see: they add a
mechanical spring in parallel with the air spring, which pushes the
mass-spring-mass resonance up, and they open a structure-borne path from one
leaf to the other, which caps the insulation the pair can reach.

**Dynamic stiffness of a tie (Section 3.11.3.2).** A tie is characterised by a
single number `sX mm`, its dynamic stiffness at a cavity width `X`,
measured on two nominally identical 100 mm concrete cubes from the
mass-spring-mass resonance of the pair,
$s_{X\,\mathrm{mm}} = 2 \pi^2 f_{msm}^2 m_{av}$ (Eq. 3.202).
[`WALL_TIE_STIFFNESS`](/phonometry/reference/api/building/masonry-cavity-wall/#wall_tie_stiffness) carries Hopkins' Table A4, whose 50 mm
rows come from Hopkins, Wilson & Craik (1999) and whose 100 mm row comes from
Hall & Hopkins (2001).

**The tie array as a spring in parallel with the air (Eq. 4.89).** `N` ties
over a plate of area `S` add $N k / S$ to the cavity air stiffness
`s_a`:

$$
f_{msm} = \frac{1}{2\pi} \sqrt{ \frac{s_a + N k / S} {\rho_{s1} \rho_{s2} / (\rho_{s1} + \rho_{s2})} } \tag{Eq. 4.89}
$$

Below `fmsm` the two leaves move as one plate of the combined mass. Stiff ties
are therefore doubly bad: they raise the resonance into the rating range *and*
bridge the cavity. Use [`wall_tie_stiffness_per_area`](/phonometry/reference/api/building/masonry-cavity-wall/#wall_tie_stiffness_per_area) and pass the result
as `tie_stiffness_per_area` to
[`phonometry.mass_spring_mass_resonance`](/phonometry/reference/api/building/panel-transmission/#mass_spring_mass_resonance) or
[`phonometry.double_wall_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#double_wall_transmission_loss).

**The structure-borne path (Eqs. 4.84 to 4.88).** Each tie is a point
connection between two plates. With the driving-point mobilities `Yi`,
`Yj` of the two leaves (infinite thin plates,
$Y = 1/(8 \sqrt{B' m''})$, Eq. 2.190) and the connector mobility of a
linear spring $Y_c = i \omega / k$ (Eq. 4.88), `N` identical
uncorrelated connections give the coupling loss factor

$$
\eta_{ij} = \frac{N}{\omega m_i} \frac{\operatorname{Re}\{Y_j\}}{| Y_i + Y_j + Y_c |^2} \tag{Eq. 4.87}
$$

The plate area cancels ($N/m_i = n/\rho_{s1}$ with `n` ties per m2),
so [`wall_tie_coupling_loss_factor`](/phonometry/reference/api/building/masonry-cavity-wall/#wall_tie_coupling_loss_factor) needs only the tie density. A rigid
connection (screw, nail, bolt, or a tie so stiff it never yields) is the
limit $Y_c = 0$, where the only frequency dependence left is the
$1/\omega$ and $\eta_{ij}$ falls as $1/f$. Once
$|Y_c| = \omega/k$ overtakes the plate mobilities a resilient tie adds
two more powers, so $\eta_{ij}$ falls as $1/f^3$ and the *ratio*
to the rigid ceiling as $1/f^2$. That is exactly why a butterfly tie at
1.7 MN/m and a vertical-twist tie at 94 MN/m behave so differently: the soft
one enters that regime inside the building acoustics range, the stiff one
stays on the rigid ceiling for another two octaves.

:::note
The *inputs* of this model are printed data: Table A4 here, confirmed
value for value by Hopkins, Wilson & Craik (1999) Table 1, which prints the
same 1.7 / 16.1 / 94.0 MN/m at a 50 mm cavity. Craik & Wilson (1995)
Table 1 measures the same *tie types* at an 85 mm cavity and reports
1.1 and 4.3 MN/m for the butterfly and double-triangle ties, so it
corroborates the ordering and the order of magnitude but not the values;
the dynamic stiffness is defined at a given cavity width and changes with
it. The *output* is not printed anywhere: every published sound reduction
index of a bridged masonry cavity wall is a figure, so the per-band
transmission-loss penalty from wall ties has no numeric oracle. The
resonance shift does: Hopkins Fig. 4.35 prints $f_{msm} = 26$ Hz
without ties and $f_{msm} = 50$ Hz with them for the same wall.
:::

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## wall_tie_coupling_loss_factor

```python
wall_tie_coupling_loss_factor(
    frequency: ArrayLike,
    mass1: float,
    mass2: float,
    bending_stiffness1: float,
    bending_stiffness2: float,
    *,
    ties_per_area: float,
    tie: str | float | None = None,
) -> WallTieCouplingResult
```

Coupling loss factor of a wall-tie array (Hopkins Eqs. 4.87 and 4.88).

$$
\eta_{ij} = \frac{N}{\omega m_i} \frac{\operatorname{Re}\{Y_j\}}{| Y_i + Y_j + Y_c |^2} \tag{Eq. 4.87}
$$

for `N` identical, uncorrelated point connections, with the leaves
modelled as infinite thin plates ($Y = 1/(8 \sqrt{B' m''})$,
Eq. 2.190) and each tie as a linear spring ($Y_c = i \omega / k$,
Eq. 4.88). Since $m_i = \rho_{s1} S$ and $N = n S$, the plate
area cancels and only the tie density `n` enters.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency` | Frequencies `f`, in hertz (array, > 0). |
| `mass1` | Surface density `rho_s1` of the excited leaf, in kg/m^2 (> 0). |
| `mass2` | Surface density `rho_s2` of the receiving leaf, in kg/m^2 (> 0). |
| `bending_stiffness1` | Bending stiffness per unit width `B'` of leaf 1, in N.m (> 0); see [`phonometry.vibration.structural.point_mobility.plate_bending_stiffness`](/phonometry/reference/api/vibration/point-mobility/#plate_bending_stiffness). |
| `bending_stiffness2` | Bending stiffness per unit width of leaf 2, in N.m. |
| `ties_per_area` | Number of ties per unit area `n`, in 1/m^2 (> 0). |
| `tie` | A name from [`WALL_TIE_STIFFNESS`](/phonometry/reference/api/building/masonry-cavity-wall/#wall_tie_stiffness), an explicit dynamic stiffness `k` in N/m, or `None` for a rigid connection ($Y_c = 0$, the screw/nail/bolt limit). |

**Returns:** A [`WallTieCouplingResult`](/phonometry/reference/api/building/masonry-cavity-wall/#walltiecouplingresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or an unknown tie name. |

## WALL_TIE_STIFFNESS

*Constant* (`dict`).

```python
WALL_TIE_STIFFNESS = {'butterfly': (0.05, 1700000.0), 'double_triangle': (0.05, 16100000.0), 'vertical_twist': (0.05, 94000000.0), 'vertical_twist_100mm': (0.1, 43400000.0)}
```

## wall_tie_stiffness

```python
wall_tie_stiffness(tie: str) -> tuple[float, float]
```

Look up a wall tie in Hopkins Table A4.

**Parameters**

| Name | Description |
| :--- | :--- |
| `tie` | One of `"butterfly"`, `"double_triangle"`, `"vertical_twist"` (all at a 50 mm cavity) or `"vertical_twist_100mm"`. |

**Returns:** `(cavity_width, stiffness)`: the cavity width in m at which the tie was measured, and its dynamic stiffness `sX mm` in N/m.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for an unknown tie name. |

## wall_tie_stiffness_per_area

```python
wall_tie_stiffness_per_area(ties_per_area: float, tie: str | float) -> float
```

Stiffness per unit area of a tie array, $N k / S$ (Hopkins
Eq. 4.89).

The term that acts in parallel with the cavity air stiffness `s_a` in the
mass-spring-mass resonance. Feed it to
[`phonometry.mass_spring_mass_resonance`](/phonometry/reference/api/building/panel-transmission/#mass_spring_mass_resonance) or
[`phonometry.double_wall_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#double_wall_transmission_loss) as
`tie_stiffness_per_area`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ties_per_area` | Number of ties per unit area $n = N/S$, in 1/m^2 (> 0). |
| `tie` | A name from [`WALL_TIE_STIFFNESS`](/phonometry/reference/api/building/masonry-cavity-wall/#wall_tie_stiffness), or an explicit dynamic stiffness `k` of one tie, in N/m (> 0). |

**Returns:** The stiffness per unit area $n k$, in N/m^3.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive input or an unknown tie name. |

## WallTieCouplingResult

```python
WallTieCouplingResult(
    frequencies: np.ndarray,
    coupling_loss_factor: np.ndarray,
    mobility1: float,
    mobility2: float,
    connector_mobility: np.ndarray,
    ties_per_area: float,
    tie_stiffness: float | None,
    rigid_coupling_loss_factor: np.ndarray,
)
```

Structure-borne coupling of a wall-tie array (Hopkins Eqs. 4.87/4.88).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies, in hertz. |
| `coupling_loss_factor` | Coupling loss factor `eta_ij` from leaf 1 to leaf 2 per frequency (dimensionless). |
| `mobility1` | Driving-point mobility `Yi` of leaf 1, in m/(N.s). |
| `mobility2` | Driving-point mobility `Yj` of leaf 2, in m/(N.s). |
| `connector_mobility` | Magnitude $\lvert Y_c \rvert = \omega/k$ of the tie mobility per frequency, in m/(N.s); all zeros for a rigid connection. |
| `ties_per_area` | Number of ties per unit area `n`, in 1/m^2. |
| `tie_stiffness` | Dynamic stiffness `k` of one tie, in N/m, or `None` for a rigid connection. |
| `rigid_coupling_loss_factor` | The $Y_c = 0$ coupling loss factor per frequency, the ceiling a resilient tie is measured against. |

### WallTieCouplingResult.plot()

```python
WallTieCouplingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `eta_ij(f)` against the rigid-connection ceiling.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.
