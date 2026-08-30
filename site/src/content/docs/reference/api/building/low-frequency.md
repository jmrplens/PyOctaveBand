---
title: "building.measurement.low_frequency"
description: "Low-frequency procedure of ISO 16283, shared by all three parts."
sidebar:
  label: "low_frequency"
---

Low-frequency procedure of ISO 16283, shared by all three parts.

Below 100 Hz a dwelling-sized room has too few modes for the central
microphone positions of the default procedure to stand for the whole volume,
so ISO 16283 adds a second measurement in the room corners and combines the
two. The procedure is **not optional**: Part 1 Clause 8.1, Part 2 Clause 8.1
and Part 3 Clause 7.3.1 all say it *shall* be used for the 50 Hz, 63 Hz and
80 Hz one-third-octave bands once the room volume, calculated to the nearest
cubic metre, is smaller than 25 m³. Most bedrooms and bathrooms are under that
line, which is why this sits under the field-measurement entry points rather
than beside them.

**The corner level.** With the source running, the highest level of the set of
measured corners is taken, band by band, and the values for the three bands may
come from three different corners (the NOTE under Formula (12)). Where a single
loudspeaker or tapping machine is moved between q positions those q maxima are
energy-averaged, Part 1 Formula (12) and Part 2 Formula (15):

$$
L_\mathrm{Corner} = 10 \lg \frac{p^2_\mathrm{Corner,1} + \cdots + p^2_\mathrm{Corner,q}}{q\,p_0^2}
$$

Part 3 numbers no such formula: Clause 7.3.4 defines $L_\mathrm{2,Corner}$
as the maximum over corners and averages the *level difference* over
loudspeaker positions later (Clause 9.6.3, Formula (8)). The maximum is the
q = 1 case of the formula above, so the same code answers all three.

**The combination.** The low-frequency energy-average level weighs the corner
level one third against two thirds of the default-procedure level. Part 1
Formula (13), Part 2 Formula (16) and Part 3 Formula (5) print it identically,
only the subscripts of the level symbols changing:

$$
L_\mathrm{LF} = 10 \lg \left[ \frac{10^{0,1 L_\mathrm{Corner}} + (2 \cdot 10^{0,1 L})}{3} \right]
$$

**The reverberation time.** Under the same 25 m³ trigger, Part 1 Clause 10.4,
Part 2 Clause 10.4 and Part 3 Clause 8.4 stop the 50 Hz, 63 Hz and 80 Hz
one-third-octave reverberation times being measured at all and put one 63 Hz
*octave* band value in their place, used for all three bands. It is a
prescription about what to measure, not a claim that the octave value equals
the three one-third-octave ones: in a small room there are too few modes for a
one-third-octave decay to be single-sloped, and in timber or steel frame
construction the decay can be shorter than the analyser's own one-third-octave
filter (NOTE 1 and NOTE 2 under each of those clauses). Below the trigger there
is no default value to fall back on either, because Clause 10.3 / 8.3 confines
the default reverberation-time procedure to 100 Hz and above once the room is
under 25 m³.

**Which room.** Part 1 applies the corner procedure to "the source and/or
receiving room when *its* volume" is under the line, so a 18 m³ source room
next to a 40 m³ receiving room gets the corner treatment on $L_1$ alone.
Parts 2 and 3 have only a receiving room to treat. The 63 Hz octave
substitution is keyed to the **receiving** room in all three parts, Part 1
included, so that asymmetry is real and this module encodes it: a source-room
procedure that carries a 63 Hz octave reverberation time is refused.

**Which methods.** Part 3 restricts the whole procedure to the element and
global *loudspeaker* methods; Clause 6 NOTE 1 records that there is no
experience of running it with traffic as the source, and the heading of
Clause 7.3 carries the restriction. Part 2 restricts it to the tapping machine
(the heading of its Clause 8), while the 63 Hz octave reverberation time of its
Clause 10.4 also feeds the rubber-ball quantity $L'_\mathrm{i,Fmax,V,T}$.

**No numeric oracle.** Neither part publishes a worked example of this
procedure: Annexes B and C of Parts 1 and 2 are blank recording forms and the
"Examples" of Annexes D and E are loudspeaker-position drawings. The
conformance of this module therefore rests on closed forms and on printed
numbers rather than on a tabulated result; see
`scripts.conformance.domains.building` and `docs/CONFORMANCE.md` for the
checks that stand in for one.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## apply_low_frequency_procedure

```python
apply_low_frequency_procedure(
    level: ArrayLike,
    frequencies: ArrayLike,
    procedure: LowFrequencyProcedure,
    *,
    reverberation_time: ArrayLike | None = None,
    room: str = 'receiving',
) -> LowFrequencyResult
```

Run the low-frequency procedure over one room's band values.

The single implementation behind
[`airborne_insulation`](/phonometry/reference/api/building/insulation/#airborne_insulation),
[`impact_insulation`](/phonometry/reference/api/building/insulation/#impact_insulation) and
[`facade_insulation`](/phonometry/reference/api/building/insulation/#facade_insulation): it
takes $L_\mathrm{Corner}$ from the corner levels, combines it with
the default-procedure level into $L_\mathrm{LF}$, writes the result
back over the 50 Hz, 63 Hz and 80 Hz bands, and puts the 63 Hz octave
reverberation time over the same three bands.

`room` decides which half of the procedure is in force. Clause 10.4
(Part 3: Clause 8.4) is a receiving-room clause in all three parts, even in
Part 1 where the corner procedure itself also admits the source room, so a
`"receiving"` call carries both halves and a `"source"` call carries
neither: it takes no reverberation times and refuses a procedure that
brings a 63 Hz octave value.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | Energy-average levels of the whole measurement, in dB, one value per band. |
| `frequencies` | Band centre frequencies, in Hz, same length. |
| `procedure` | The room's corner measurements and volume. |
| `reverberation_time` | Reverberation times of the whole measurement, in seconds. Required for `room="receiving"`; refused for `room="source"`. |
| `room` | `"receiving"` (default) or `"source"`. It selects whether Clause 10.4 applies, and it is quoted in the messages so a two-room airborne measurement says which side failed. |

**Returns:** A [`LowFrequencyResult`](/phonometry/reference/api/building/low-frequency/#lowfrequencyresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `room` is neither name, if the band counts disagree, if the 50 Hz, 63 Hz or 80 Hz band is absent from `frequencies`, if a receiving room is missing either half of Clause 10.4, or if a source room brings either half of it. |

## corner_level

```python
corner_level(corner_levels: ArrayLike) -> np.ndarray
```

Corner sound pressure level $L_\mathrm{Corner}$ per band.

Two shapes are accepted, and they are the two the standards describe.

A `(corners, bands)` array is one source position: the level is the
highest of the measured corners in each band, taken independently per band
because "the values for $L_\mathrm{Corner}$ may be associated with
different corners in the room" (NOTE under Part 1 Formula (12)). This is
also the case of loudspeakers operated simultaneously (Part 1 Clause 8.5,
first paragraph) and the whole of Part 3, whose Clause 7.3.4 defines
$L_\mathrm{2,Corner}$ as that maximum and numbers no formula.

A `(positions, corners, bands)` array is a single source moved between
`q` positions: the maximum is taken per position and the `q` results
are energy-averaged, which is Part 1 Formula (12) and Part 2 Formula (15)
written in levels rather than in mean-square pressures. The two forms agree
because $p^2/p_0^2 = 10^{L/10}$, and the `(corners, bands)` shape
is the same formula at `q = 1`.

Corner levels are assumed already corrected for background noise, which
Part 2 Formula (15) states in its own where-list and Parts 1 and 3 place in
their background-noise clause (Part 1 Clause 9.1, Part 3 Clause 7.4.1, both
requiring a background measurement in every corner used).

**Parameters**

| Name | Description |
| :--- | :--- |
| `corner_levels` | Corner sound pressure levels, in dB, as `(corners, bands)` or `(positions, corners, bands)`. |

**Returns:** $L_\mathrm{Corner}$, one value per band, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the array is not two- or three-dimensional, is empty, or holds a non-finite value. |

## LOW_FREQUENCY_BANDS

*Constant* (`tuple`).

```python
LOW_FREQUENCY_BANDS = (50.0, 63.0, 80.0)
```

## low_frequency_level

```python
low_frequency_level(level: ArrayLike, corner: ArrayLike) -> np.ndarray
```

Combine the default and corner levels into $L_\mathrm{LF}$.

Part 1 Formula (13), Part 2 Formula (16) and Part 3 Formula (5), which are
the same expression under three sets of subscripts:

$$
L_\mathrm{LF} = 10 \lg \left[ \frac{10^{0,1 L_\mathrm{Corner}} + (2 \cdot 10^{0,1 L})}{3} \right]
$$

The two levels are weighted one third to two thirds, so the result
degenerates to `L` when the corner level equals it, rises with the corner
level and can never fall further than $10 \lg(2/3) = -1,76$ dB below
`L`, however quiet the corners are.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | Energy-average level `L` from the default procedure, in dB, one value per band. |
| `corner` | Corner level $L_\mathrm{Corner}$ from [`corner_level`](/phonometry/reference/api/building/low-frequency/#corner_level), in dB, same bands. |

**Returns:** $L_\mathrm{LF}$, in dB, one value per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the two shapes differ, either is empty, or either holds a non-finite value. |

## low_frequency_procedure_applies

```python
low_frequency_procedure_applies(volume: float) -> bool
```

Whether a room of this volume triggers the low-frequency procedure.

The printed condition is the same in all three parts: the volume,
"calculated to the nearest cubic metre", is "smaller than 25 m³"
(Part 1 Clause 8.1 and 10.4, Part 2 Clause 8.1 and 10.4, Part 3
Clause 7.3.1 and 8.4). The comparison is strict, so a room that rounds to
25 m³ exactly does not trigger, and neither does anything larger.

The rounding is half away from zero, `floor(V + 0,5)`, which is the rule
the rest of this tree rounds printed quantities with; the standards give no
tie rule of their own. It matters on the boundary: `V = 24,5` m³ rounds
to 25 m³ here and does **not** trigger, where Python's built-in
`round`, which is half-to-even, would answer 24 and trigger.

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Room volume `V`, in m³. |

**Returns:** `True` when the low-frequency procedure is required.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `volume` is not a positive, finite number. |

## LOW_FREQUENCY_VOLUME_LIMIT

*Constant* (`float`).

```python
LOW_FREQUENCY_VOLUME_LIMIT = 25.0
```

## LowFrequencyProcedure

```python
LowFrequencyProcedure(
    volume: float,
    corner_levels: Sequence[float] | np.ndarray,
    reverberation_63_octave: float | None = None,
)
```

The extra measurements ISO 16283 asks for in a room under 25 m³.

One of these describes one room. Part 1 tests the source and the receiving
room independently, so an airborne measurement may carry two, one, or
neither; Parts 2 and 3 have only a receiving room.

**Attributes**

| Name | Description |
| :--- | :--- |
| `volume` | Volume `V` of the room the corners were measured in, in m³. It is this room's own volume that decides the trigger (Part 1 Clause 8.1, "in the source and/or receiving room when **its** volume"). |
| `corner_levels` | Corner sound pressure levels, in dB, already corrected for background noise, over the **three low-frequency bands only** and in 50 / 63 / 80 Hz order: `(corners, 3)` for one source position, or `(positions, corners, 3)` for a source moved between positions. Only those three bands are measured in the corners at all, so the corner sheet is three columns wide whatever range the default procedure covered. |
| `reverberation_63_octave` | Reverberation time measured in the 63 Hz **octave** band, in seconds, which replaces the 50 Hz, 63 Hz and 80 Hz one-third-octave values (Part 1 and Part 2 Clause 10.4, Part 3 Clause 8.4). Required for the receiving room, which is the only room those clauses speak about; must be `None` for a source room. |

## LowFrequencyResult

```python
LowFrequencyResult(
    frequencies: np.ndarray,
    level: np.ndarray,
    reverberation_time: np.ndarray | None,
    low_frequency_bands: np.ndarray,
    l_default: np.ndarray,
    l_corner: np.ndarray,
    l_lf: np.ndarray,
    volume: float,
    reverberation_63_octave: float | None,
)
```

What the low-frequency procedure did to one room's band values.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies of the whole measurement, in Hz, as supplied. |
| `level` | The energy-average levels of the whole measurement, in dB, with the 50 Hz, 63 Hz and 80 Hz bands replaced by $L_\mathrm{LF}$ and every other band untouched. |
| `reverberation_time` | The reverberation times of the whole measurement, in seconds, with those same three bands replaced by the 63 Hz octave value; `None` for a source room, which Clause 10.4 does not speak about. |
| `low_frequency_bands` | The three band centres the procedure was applied at, in Hz, as they were spelled in `frequencies`. |
| `l_default` | The default-procedure levels at those three bands, in dB, before the combination. |
| `l_corner` | $L_\mathrm{Corner}$ at those three bands, in dB (Part 1 Formula (12), Part 2 Formula (15), Part 3 Clause 7.3.4). |
| `l_lf` | $L_\mathrm{LF}$ at those three bands, in dB (Part 1 Formula (13), Part 2 Formula (16), Part 3 Formula (5)). |
| `volume` | Volume of the room, in m³, that put the procedure in force. |
| `reverberation_63_octave` | The 63 Hz octave reverberation time, in seconds, or `None` when none was substituted. |

### LowFrequencyResult.plot()

```python
LowFrequencyResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the three low-frequency bands, default against corner and LF.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## LowFrequencyWarning

A low-frequency measurement that falls short of a sampling requirement.

Raised for the corner count of Part 1 Clause 8.3, Part 2 Clause 8.3 and
Part 3 Clause 7.3.2, which the arithmetic of Formula (12) does not depend
on: four corners and three give the same maximum-then-average, and it is
the report that has to say the room was undersampled.
