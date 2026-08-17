---
title: "building.prediction.linings"
description: "Prediction of wall and ceiling linings: the weighted improvement an additional layer gives the element behind it (ISO 12354-1:2017 Annex D)."
sidebar:
  label: "linings"
---

Prediction of wall and ceiling linings: the weighted improvement an additional
layer gives the element behind it (ISO 12354-1:2017 Annex D).

A lining is not a floor covering. It is added to a wall or a ceiling, not laid
under a tapping machine, and what it changes is the airborne sound reduction
index of the basic element it is fixed to. Annex D therefore rates it as a
shift of a single number rather than as a spectrum, and it reads that shift off
one quantity: the mass-spring resonance of the lining against the element. That
single frequency is what makes the annex one subject and this module one file.

A lining improves or *degrades* the sound insulation depending on where its
resonance falls, so Annex D predicts the weighted improvement from `fo`
alone: Formula (D.1) for a layer bonded directly to the wall, Formula (D.2) for
one on studs over a filled cavity, then Table D.1 for interior linings
([`weighted_lining_improvement`](/phonometry/reference/api/building/linings/#weighted_lining_improvement)), Formulae (D.3) to (D.6) for exterior
thermal systems and (D.7) for stud systems ([`lining_improvement`](/phonometry/reference/api/building/linings/#lining_improvement)), and
Formula (D.8) to carry a laboratory rating to the field
([`lining_improvement_in_situ`](/phonometry/reference/api/building/linings/#lining_improvement_in_situ)).

Citations are to ISO 12354-1:2017. One printed defect is relevant here and is
recorded in `docs/ERRATA.md`: the overlap of the last two rows of Table D.1
at 1 600 Hz.

Several relations used here carry no published worked example, so they are
implemented as printed and checked only for self-consistency: the cavity
stiffness $0.111/d$ of Formula (D.2) and the exterior-system and stud
fits of Formulae (D.3) to (D.8). The guide "Predicting Resilient-Layer
Performance" says which pieces have an oracle and which do not.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## lining_improvement

```python
lining_improvement(
    resonance_frequency: float,
    *,
    system: LiningSystem = 'mineral_wool',
    anchors: bool = False,
    glued_area: float | None = None,
) -> LiningImprovementResult
```

Single-number ratings of an additional layer (Formulae D.3 to D.7).

For the reference situation of ISO 12354-1:2017 Annex D, a system applied
to a heavy basic wall of about 350 kg/m²:

* `system="mineral_wool"` (Formula D.3), an exterior thermal system on
  mineral wool with 40 % glued area and no anchors:
  $\Delta R_\mathrm{w} = -36 \log_{10}(f_\mathrm{o}) + 82.5$,
  $\Delta R_\mathrm{A} = -42 \log_{10}(f_\mathrm{o}) + 92.0$,
  $\Delta R_\mathrm{A,tr} = -39 \log_{10}(f_\mathrm{o}) + 87.7$, each floored at −4 dB.
* `system="foam"` (Formula D.4), the same on PS, EPS or EEPS foams:
  $-33 \log_{10}(f_\mathrm{o}) + 76.0$, $-33 \log_{10}(f_\mathrm{o}) + 74.0$,
  $-36 \log_{10}(f_\mathrm{o}) + 77.0$, floored at −3 dB.
* `system="studs"` (Formula D.7), a layer on studs not directly fixed to
  the basic wall: $-20 \log_{10}(f_\mathrm{o}) + 48$, $-22 \log_{10}(f_\mathrm{o}) + 51$,
  $-24 \log_{10}(f_\mathrm{o}) + 54$, floored at −4 dB.

`anchors=True` applies Formula (D.5) for 4 to 10 anchors or battens per
m² ($0.66 \Delta R_\mathrm{w,ref} - 1.2$ and its two companions), and
`glued_area` applies Formula (D.6),
$\Delta R - 0.05\,\%S_o + 2.0$, for a glued area other than the
40 % reference. Both corrections are applied after the floor of the
reference formula, in the order the annex states them.

The annex places the $\ge -4$ dB (or $\ge -3$ dB) floor
inside Formulae (D.3) and (D.4) and says nothing about re-applying it
after (D.5) and (D.6), so this function does not: a fully glued system on
anchors can
return about −6.8 dB, below the reference floor. That is the annex read
literally, and the reason the two corrections are exposed as flags rather
than folded into the fit.

**Parameters**

| Name | Description |
| :--- | :--- |
| `resonance_frequency` | Resonance frequency `fo`, in Hz ([`lining_resonance_frequency`](/phonometry/reference/api/building/linings/#lining_resonance_frequency)). |
| `system` | `"mineral_wool"`, `"foam"` or `"studs"`. |
| `anchors` | Apply the Formula (D.5) anchor/batten correction. |
| `glued_area` | Glued area `%So` as a percentage of the element area, greater than 0 and at most 100, or `None` to keep the 40 % reference. Formula (D.6) divides by the glued area, so a wholly unglued system is not a case of it: use `anchors` for a mechanically fixed lining. It corrects the glued exterior systems only, so it is rejected for `system="studs"`. |

**Returns:** A [`LiningImprovementResult`](/phonometry/reference/api/building/linings/#liningimprovementresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, `system` is unknown, or `glued_area` is out of range or combined with `system="studs"`. |

## lining_improvement_in_situ

```python
lining_improvement_in_situ(
    laboratory_improvement: float,
    resonance_frequency: float,
    base_rating_in_situ: float,
) -> float
```

Transfer a weighted lining improvement to the field (Formula D.8).

Even when the per-band improvement is invariant, its single-number rating
still depends on the basic element it sits on, so ISO 12354-1:2017
Formula (D.8) shifts the laboratory rating by $a X$ with

$a = 1.35 \log_{10}(f_\mathrm{o}) - 3.5$, capped at 0, and
$X = R_\mathrm{w,situ} - 53$, clamped to `[−10, +7]`.

The same formula applies to `ΔRw`, `ΔRA` and `ΔRA,tr`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `laboratory_improvement` | Laboratory rating `ΔRlab` measured to ISO 10140-1:2016 Annex G for the heavy basic element, in dB. |
| `resonance_frequency` | Resonance frequency `fo` of the system, in Hz. |
| `base_rating_in_situ` | Weighted sound reduction index `Rw,situ` of the basic element in the field situation, in dB. |

**Returns:** The field rating `ΔRsitu`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not finite, or `fo` is not positive. |

## lining_resonance_frequency

```python
lining_resonance_frequency(
    base_mass_per_area: float,
    lining_mass_per_area: float,
    *,
    dynamic_stiffness: float | None = None,
    cavity_depth: float | None = None,
) -> float
```

Resonance `fo` of a lining on a basic element (Formulae D.1/D.2).

Exactly one of the two branches applies:

* `dynamic_stiffness` (Formula D.1), for an insulation layer fixed
  **directly** to the basic construction, without studs or battens:
  $f_\mathrm{o} = \sqrt{s' (1/m'_1 + 1/m'_2)}/(2 \pi)$.
* `cavity_depth` (Formula D.2), for a layer built on metal or wooden
  studs **not** connected to the basic element, with the cavity filled by a
  porous layer of airflow resistivity $r \ge 5$ kPa·s/m²:
  $f_\mathrm{o} = \sqrt{(0.111/d)(1/m'_1 + 1/m'_2)}/(2 \pi)$, i.e. the
  near-isothermal stiffness of the filled cavity replaces `s'`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `base_mass_per_area` | Mass per unit area `m'1` of the basic structural element, in kg/m². |
| `lining_mass_per_area` | Mass per unit area `m'2` of the additional layer, in kg/m². |
| `dynamic_stiffness` | Dynamic stiffness per unit area `s'` of the insulation layer (EN 29052-1), in N/m³. |
| `cavity_depth` | Depth `d` of the stud cavity, in m. |

**Returns:** The resonance frequency `fo`, in Hz.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If an input is not positive and finite, or if the two branches are both given or both omitted. |

## LiningImprovementResult

```python
LiningImprovementResult(
    resonance_frequency: float,
    system: LiningSystem,
    delta_rw: float,
    delta_ra: float,
    delta_ratr: float,
    anchors: bool = False,
    glued_area: float | None = None,
)
```

Single-number ratings of an additional layer (ISO 12354-1 Annex D).

**Attributes**

| Name | Description |
| :--- | :--- |
| `resonance_frequency` | Resonance frequency `fo` of the system, in Hz. |
| `system` | `"mineral_wool"`, `"foam"` (exterior systems glued to the wall, Formulae D.3/D.4) or `"studs"` (Formula D.7). |
| `delta_rw` | Improvement of the weighted sound reduction index `ΔRw`, in dB. |
| `delta_ra` | Improvement of the A-weighted rating `ΔRA`, in dB. |
| `delta_ratr` | Improvement of the traffic-weighted rating `ΔRA,tr`, in dB. |
| `anchors` | `True` when the Formula (D.5) anchor correction was applied. |
| `glued_area` | Glued area as a percentage of the element area, or `None` when the 40 % reference was kept. |

### LiningImprovementResult.plot()

```python
LiningImprovementResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the Annex D ratings against the resonance frequency.

Draws the three Annex D curves over the tabulated range with this
system's own resonance marked, the analogue of Figures D.2 and D.3.
Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### LiningImprovementResult.ratings

*property*

`(ΔRw, ΔRA, ΔRA,tr)` as a tuple, in dB.

## weighted_lining_improvement

```python
weighted_lining_improvement(
    resonance_frequency: float,
    base_rating: float,
) -> float
```

Weighted improvement `ΔRw` of an interior lining (Table D.1).

ISO 12354-1:2017 Table D.1 reads `ΔRw` off the lining's resonance
frequency, rounded to the centre of the one-third-octave band in which it
falls. Below 200 Hz the improvement also depends on the bare element:
$\Delta R_\mathrm{w} = 74.4 - 20 \log_{10}(f_\mathrm{o}) - R_\mathrm{w}/2$, never below 0 dB
(NOTE 1). At and above 200 Hz the lining *degrades* the insulation, by
1 dB at 200 Hz down to 10 dB from 630 Hz to 1 600 Hz, recovering to 5 dB
from 1 600 Hz to 5 000 Hz.

Table D.1 is stated for basic elements with $20 \le R_\mathrm{w} \le 60$ dB.
Its last two rows both cover 1 600 Hz with different values; this function
takes the more conservative −10 dB there (see `docs/ERRATA.md`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `resonance_frequency` | Resonance frequency `fo` of the lining, in Hz ([`lining_resonance_frequency`](/phonometry/reference/api/building/linings/#lining_resonance_frequency)); must fall in the 30 Hz to 5 000 Hz range Table D.1 covers. |
| `base_rating` | Weighted sound reduction index `Rw` of the bare wall or floor, in dB. |

**Returns:** The weighted improvement `ΔRw`, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `fo` is outside the tabulated range or an input is not finite. |
