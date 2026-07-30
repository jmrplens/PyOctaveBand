---
title: "building.ceiling_plenum"
description: "Suspended-ceiling plenum flanking path (Vigran 9.2.3 after Mechel 1980; ISO 140-9 / ISO 10848-2; ASTM E1414 / ASTM E413)."
sidebar:
  label: "ceiling_plenum"
---

Suspended-ceiling plenum flanking path (Vigran 9.2.3 after Mechel 1980;
ISO 140-9 / ISO 10848-2; ASTM E1414 / ASTM E413).

Two offices separated by a partition that stops at the suspended ceiling share
one continuous plenum above it. Sound leaves the source room through the
ceiling tiles, travels sideways over the partition and comes back down through
the tiles of the receiving room. That path is often the weakest link in an
open-plan fit-out, and it is *not* what a partition's `Rw` describes.

**The one-dimensional model (Vigran Eqs. (9.14) to (9.20)).** Mechel's
one-dimensional variant treats the plenum as a duct lined on one side. The
ceiling on each side has a transmission factor `tauS = tauS,pl * tauS,a`
(plates times the prospective plenum absorber, Eq. (9.14)); the power injected
into the plenum splits, a fraction `sS` heading for the partition, and decays
as `exp(-m x)` with the power attenuation coefficient
`m = 2 Re{Gamma} = -2 Im{k'}` (Eqs. (9.15) and (9.16)). Integrating over the
ceiling length on both sides gives (Eq. (9.18)):

```text
tau_cl = sS sR tauS tauR LR / (mS LS mR LR h)
         * (1 - exp(-eps mS LS)) (1 - exp(-eps m'R LR))
```

with the receiving-side coefficient increased by the leakage back into the room,
`m'R = mR + sR tauR / h` (Eq. (9.17)). Vigran prints the exponents with a
factor 2 for totally reflecting plenum sidewalls and states that totally
absorbing ones give the same expression "without the factor 2", so the factor is
the same `eps` that the compact form carries. For a plenum with little
attenuation (`mS LS`, `mR LR` \<\< 1) and `sS = sR = 0,5` it collapses to
the result that makes the geometry visible (Eqs. (9.19) and (9.20)):

```text
tau_cl = eps**2 tauS tauR LR / (4 h)
Rcl    = RS + RR - 10 lg[eps**2 LR / (4 h)]
```

with `eps = 1` for totally absorbing plenum sidewalls and `eps = 2` for
totally reflecting ones. Referred to the partition area instead of the ceiling,
`Rcl,p = Rcl + 10 lg(HS/LS)` (Eq. (9.13)). A deep plenum helps (the
`-10 lg` term shrinks), a long room hurts, and doubling the tile insulation
helps twice over because `RS` and `RR` both appear.

**The measured quantity (ISO 140-9:1985 clause 3.3, ISO 10848-2).** A ceiling
is not rated by `Rcl` but by the **normalized ceiling attenuation**
`Dn,c = D - 10 lg(A/A0)`, with `A` the receiving-room equivalent absorption
area and the reference `A0 = 10 m2`. The laboratory has two rooms of at least
50 m3 whose volumes differ by at least 10 %, a dividing wall tapered to at most
100 mm at the top, and a plenum 650 mm to 760 mm deep with one sidewall and both
end walls lined; the standard prints the required lining absorption
`alpha_s >= 0,65` at 125 Hz and `>= 0,80` from 250 Hz to 4000 Hz, and
requires `alpha < 0,10` on the other sidewall and on the plenum ceiling. The
North American counterpart, ASTM E1414, uses `A0 = 12 m2`, so an ASTM value
runs about `10 lg(12/10) = 0,79 dB` higher than the ISO one.

**Single number.** ISO rates `Dn,c` with the ISO 717-1 curve
([`phonometry.weighted_rating`](/phonometry/reference/api/building/insulation/#weighted_rating), giving `Dn,c,w`); ASTM E1414 rates it
through ASTM E413 as the **ceiling attenuation class** (CAC). E413 rounds the
data to the nearest integer (clause 5.2), shifts its reference contour upward in
1 dB steps while the sum of the deficiencies stays at or below 32 dB and no
single deficiency exceeds 8 dB (clauses 5.3 and 5.4), and reads the rating off
the shifted contour at 500 Hz (clause 5.5). See
[`ceiling_attenuation_class`](/phonometry/reference/api/building/ceiling-plenum/#ceiling_attenuation_class).

:::note
The one-dimensional plenum model has **no published numeric output**: every
result in Vigran (Figs. 9.11 to 9.13) and in Mechel's *Formulas of Acoustics*
(Sections I.21 and I.22) is a figure. The functions here are anchored on the
closed forms and on the internal consistency between Eq. (9.18) and its
small-attenuation limit Eq. (9.20). The measurement chain
([`normalized_ceiling_attenuation`](/phonometry/reference/api/building/ceiling-plenum/#normalized_ceiling_attenuation), [`ceiling_attenuation_class`](/phonometry/reference/api/building/ceiling-plenum/#ceiling_attenuation_class))
*is* anchored on accredited ASTM E1414 laboratory reports.
:::

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## ceiling_attenuation_class

```python
ceiling_attenuation_class(
    attenuation: ArrayLike,
    frequency: ArrayLike | None = None,
) -> CeilingAttenuationResult
```

Ceiling attenuation class CAC (ASTM E413-22 clause 5, via ASTM E1414).

Rounds the data to the nearest integer (clause 5.2), then raises the
[`CEILING_ATTENUATION_CONTOUR`](/phonometry/reference/api/building/ceiling-plenum/#ceiling_attenuation_contour) in 1 dB steps to the highest position
at which the sum of the deficiencies is at most 32 dB (clause 5.4.1) and no
single deficiency exceeds 8 dB (clause 5.4.2). The rating is the shifted
contour read at 500 Hz (clause 5.5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `attenuation` | Normalized ceiling attenuation `Dn,c` in the 16 one-third-octave bands 125 Hz to 4000 Hz, in dB. |
| `frequency` | Optional band centre frequencies, in Hz; when given they must be exactly the 16 contour bands. |

**Returns:** A [`CeilingAttenuationResult`](/phonometry/reference/api/building/ceiling-plenum/#ceilingattenuationresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a wrong number of bands or mismatched frequencies. |

## CEILING_ATTENUATION_CONTOUR

*Constant* (`dict`).

```python
CEILING_ATTENUATION_CONTOUR = {125.0: -16.0, 160.0: -13.0, 200.0: -10.0, 250.0: -7.0, 315.0: -4.0, 400.0: -1.0, 500.0: 0.0, 630.0: 1.0, 800.0: 2.0, 1000.0: 3.0, 1250.0: 4.0, 1600.0: 4.0, 2000.0: 4.0, 2500.0: 4.0, 3150.0: 4.0, 4000.0: 4.0}
```

## CeilingAttenuationResult

```python
CeilingAttenuationResult(
    frequencies: np.ndarray,
    measured: np.ndarray,
    rounded: np.ndarray,
    shifted_reference: np.ndarray,
    deficiencies: np.ndarray,
    deficiency_sum: float,
    max_deficiency: float,
    rating: int,
)
```

Ceiling attenuation class (ASTM E1414 rated through ASTM E413).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centre frequencies, in Hz. |
| `measured` | Normalized ceiling attenuation `Dn,c` per band, in dB, as supplied. |
| `rounded` | The same data rounded to the nearest integer (clause 5.2), which is what the contour is fitted to. |
| `shifted_reference` | The fitted reference contour, in dB. |
| `deficiencies` | Per-band deficiency (shifted contour minus data, floored at zero), in dB. |
| `deficiency_sum` | Sum of the deficiencies, in dB (at most 32). |
| `max_deficiency` | Largest single deficiency, in dB (at most 8). |
| `rating` | The ceiling attenuation class CAC, read off the shifted contour at 500 Hz (clause 5.5), in dB. |

### CeilingAttenuationResult.plot()

```python
CeilingAttenuationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `Dn,c` against the fitted ASTM E413 contour.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## normalized_ceiling_attenuation

```python
normalized_ceiling_attenuation(
    level_source: ArrayLike,
    level_receiving: ArrayLike,
    absorption_area: ArrayLike,
    *,
    reference_area: float = 10.0,
) -> np.ndarray
```

Normalized ceiling attenuation `Dn,c` (ISO 140-9:1985, clause 3.3).

`Dn,c = (L1 - L2) - 10 lg(A/A0)`, the level difference between two rooms
sharing a common ceiling plenum, normalized to a reference equivalent
absorption area. ISO 140-9 and ISO 10848-2 use `A0 = 10 m2`;
ASTM E1414 uses `A0 = 12 m2`, which makes an ASTM value about 0,79 dB
higher for the same rooms.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level_source` | Source-room sound pressure level `L1` per band, in dB. |
| `level_receiving` | Receiving-room level `L2` per band, in dB. |
| `absorption_area` | Receiving-room equivalent absorption area `A` per band, in m2 (> 0); a scalar is broadcast over the bands. |
| `reference_area` | Reference area `A0`, in m2 (Default: 10 m2). |

**Returns:** The normalized ceiling attenuation `Dn,c` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for mismatched shapes or a non-positive area. |

## partition_referenced_reduction_index

```python
partition_referenced_reduction_index(
    reduction_index: ArrayLike,
    room_height: float,
    room_length: float,
) -> np.ndarray
```

Refer `Rcl` to the partition area instead of the ceiling (Eq. (9.13)).

`Rcl,p = Rcl + 10 lg(HS/LS)`, with `HS` the height and `LS` the length
of the sending room. Referring every path to one common area (the partition)
is what lets the ceiling path be added to the direct path as transmission
factors.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reduction_index` | Ceiling/plenum path `Rcl` per band, in dB. |
| `room_height` | Sending-room height `HS`, in m (> 0). |
| `room_length` | Sending-room length `LS`, in m (> 0). |

**Returns:** The partition-referenced `Rcl,p` per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for a non-positive dimension. |

## plenum_flanking_reduction_index

```python
plenum_flanking_reduction_index(
    reduction_index_source: ArrayLike,
    reduction_index_receiving: ArrayLike,
    *,
    ceiling_length: float,
    plenum_height: float,
    sidewalls: str = 'reflecting',
    frequency: ArrayLike | None = None,
    attenuation_source: ArrayLike | None = None,
    attenuation_receiving: ArrayLike | None = None,
    source_length: float | None = None,
    split_source: float = 0.5,
    split_receiving: float = 0.5,
) -> PlenumFlankingResult
```

Ceiling/plenum flanking reduction index `Rcl` (Vigran Eqs. (9.18)-(9.20)).

With no attenuation coefficients this is the compact undamped form
`Rcl = RS + RR - 10 lg[eps**2 LR/(4h)]` (Eq. (9.20)). Supplying the plenum
power attenuation coefficients `mS` and `mR` (Eq. (9.16),
`m = -2 Im{k'}` of the lined duct) switches to the full Eq. (9.18), whose
receiving-side exponent carries the leakage term `m'R = mR + sR tauR/h`
(Eq. (9.17)).

**Parameters**

| Name | Description |
| :--- | :--- |
| `reduction_index_source` | Source-side ceiling `RS` per band, in dB (the ceiling plates and any plenum absorber together, Eq. (9.14)). |
| `reduction_index_receiving` | Receiving-side ceiling `RR` per band, in dB. |
| `ceiling_length` | Receiving-side ceiling length `LR`, in m (> 0). |
| `plenum_height` | Plenum height `h`, in m (> 0). |
| `sidewalls` | `"reflecting"` (`eps = 2`, Default) or `"absorbing"` (`eps = 1`); `eps` scales both the geometry penalty of Eq. (9.20) and the exponents of Eq. (9.18). |
| `frequency` | Optional band centre frequencies, in Hz. |
| `attenuation_source` | Optional plenum power attenuation coefficient `mS` per band, in 1/m (> 0); switches to Eq. (9.18). |
| `attenuation_receiving` | Optional `mR` per band, in 1/m (> 0); required together with *attenuation_source*. |
| `source_length` | Source-side ceiling length `LS`, in m (Default: equal to *ceiling_length*). |
| `split_source` | Power split `sS` towards the partition, in `(0, 1]` (Default: 0,5). |
| `split_receiving` | Power split `sR` on the receiving side, in `(0, 1]` (Default: 0,5). |

**Returns:** A [`PlenumFlankingResult`](/phonometry/reference/api/building/ceiling-plenum/#plenumflankingresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | for mismatched shapes, an unknown sidewall case, or a non-positive dimension. |

## PlenumFlankingResult

```python
PlenumFlankingResult(
    frequencies: np.ndarray | None,
    reduction_index: np.ndarray,
    transmission_factor: np.ndarray,
    reduction_index_source: np.ndarray,
    reduction_index_receiving: np.ndarray,
    geometry_term: float | None,
    penalty: np.ndarray,
    model: str,
    epsilon: float,
    plenum_height: float,
    ceiling_length: float,
)
```

Ceiling/plenum flanking path of a suspended ceiling (Vigran 9.2.3).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in Hz, or `None`. |
| `reduction_index` | Sound reduction index `Rcl` of the ceiling/plenum path per band, in dB. |
| `transmission_factor` | The transmission factor `tau_cl` per band. |
| `reduction_index_source` | Source-side ceiling `RS` per band, in dB. |
| `reduction_index_receiving` | Receiving-side ceiling `RR` per band, in dB. |
| `geometry_term` | The geometry penalty `10 lg[eps**2 LR/(4h)]`, in dB, or `None` for the attenuated model, whose penalty is per band. |
| `penalty` | The per-band difference `RS + RR - Rcl`, in dB: what the plenum takes off the sum of the two ceilings. |
| `model` | `"undamped"` (Eq. (9.20)) or `"attenuated"` (Eq. (9.18)). |
| `epsilon` | The sidewall constant `eps` (1 absorbing, 2 reflecting). |
| `plenum_height` | Plenum height `h`, in m. |
| `ceiling_length` | Receiving-side ceiling length `LR`, in m. |

### PlenumFlankingResult.plot()

```python
PlenumFlankingResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot `Rcl` against the two ceiling reduction indices.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.
