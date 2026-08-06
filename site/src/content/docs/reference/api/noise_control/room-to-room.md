---
title: "noise_control.room_to_room"
description: "Room-to-room noise reduction: source room, partition, receiving room, criterion."
sidebar:
  label: "room_to_room"
---

Room-to-room noise reduction: source room, partition, receiving room, criterion.

The other classic bookkeeping exercise of noise control, beside the duct-borne
cascade of [`phonometry.noise_control.duct_path`](/phonometry/reference/api/noise_control/duct-path/). A machine runs in one
room, a partition separates that room from an occupied one, and the question is
the octave-band spectrum in the occupied room and whether it meets a design
criterion. The answer is a short chain, and every link of it already exists in
the library: the reverberant level the machine builds up in the source room
([`phonometry.room.steady_state_spl`](/phonometry/reference/api/rooms/steady-field/#steady_state_spl) over the room constant of
[`phonometry.room.room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant)), the transmission loss of the partition
(measured, or predicted by [`phonometry.building.prediction.panel_transmission`](/phonometry/reference/api/building/panel-transmission/)), the
absorption of the receiving room
([`phonometry.room.equivalent_absorption_area`](/phonometry/reference/api/rooms/enclosed-space-absorption/#equivalent_absorption_area)) and the rating of what
arrives ([`phonometry.room.noise_criterion`](/phonometry/reference/api/rooms/noise-criteria/#noise_criterion)). What this module adds is the
one step that lives nowhere else, and the result object that carries the whole
path.

**The step.** Norton & Karczub, *Fundamentals of Noise and Vibration Analysis
for Engineers* 2nd ed., 4.9, balance the steady-state power crossing the
partition against the power the receiving room absorbs and the power that leaks
back, and arrive at Equation (4.101):

$$
\mathrm{NR} = \mathrm{TL} - 10 \log_{10}\!\left[\frac{S_w}{S_2 \alpha_2 + \tau S_w}\right],
$$

with $\mathrm{NR} = L_{p1} - L_{p2}$ the noise reduction between the two
reverberant
fields, $\mathrm{TL} = 10 \log_{10}(1/\tau)$ the transmission loss of the
partition, `S_w`
the area of the partition and `S_2 alpha_2` the equivalent absorption area of
the receiving room. The `tau S_w` term is the power the partition itself
passes back and is normally negligible beside the room absorption; it is
switched on with `include_partition_transmission`.

The single most useful thing the equation says is that **the noise reduction is
not the transmission loss**. A large partition into a hard room delivers less
than its `TL`; a small partition into a well-absorbing room delivers more.
Norton also warns that the measured noise reduction runs a few decibels below
the prediction because of flanking transmission through mechanical connections
and air leaks, which `flanking_penalty` applies as an explicit debit.

**The source-room level.** In a plant room the receiver of interest is the
partition, not a point near the machine, so the level that drives the
transmission is the reverberant field alone,
$L_{p1} = L_W + 10 \log_{10}(4 / R_1)$. That is
[`phonometry.room.steady_state_spl`](/phonometry/reference/api/rooms/steady-field/#steady_state_spl) at `distance=None`, and this module
delegates to it rather than repeating it. Norton's Table 4.5 adds the choice of
*sound power model*: a machine standing in the intersection of a floor and a
wall radiates more power than its free-space `L_W` if it behaves as a
constant-volume source, and `source_model="constant_volume"` is the
conservative upper bound a design estimate uses.

**The verdict.** [`RoomToRoomResult.required_transmission_loss`](/phonometry/reference/api/noise_control/room-to-room/#roomtoroomresultrequired_transmission_loss) inverts the
chain: given the source-room level and the criterion curve, the partition
transmission loss that would just meet it, band by band. That is the number a
partition is specified from.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## DesignCriterion

```python
DesignCriterion(
    family: str = 'NC',
    target: float | None = None,
    flanking_penalty: float = 0.0,
)
```

The design criterion the receiving room is held to.

The verdict half of the chain: which room-criterion family the received
spectrum is rated against, the curve it has to stay under, and the
allowance a design sheet keeps for the transmission the calculation does
not model.

**Parameters**

| Name | Description |
| :--- | :--- |
| `family` | Room-criterion family, `"NC"` (default) or `"RC"`. |
| `target` | The design criterion value (e.g. `45` for NC 45), or `None` for no target, which leaves the verdicts undefined. |
| `flanking_penalty` | Decibels debited from the predicted noise reduction for flanking transmission through mechanical connections and air leaks (Norton's "a few dB"). Default `0`. |

## room_to_room_transmission

```python
room_to_room_transmission(
    frequencies: ArrayLike,
    transmission_loss: ArrayLike,
    partition_area: float,
    receiving_absorption: ArrayLike,
    *,
    source: SourceRoom | None = None,
    include_partition_transmission: bool = False,
    criterion: DesignCriterion | None = None,
    label: str = 'Room to room',
) -> RoomToRoomResult
```

Sound transmission from one room to another (Norton 2e Equation (4.101)).

Computes the noise reduction the partition and the receiving room deliver
together, and the reverberant spectrum in the receiving room. The
source-room level is either given directly as `source.level` or built
from a sound power level and the source room's room constant, in which case
the reverberant field alone is used ([`phonometry.room.steady_state_spl`](/phonometry/reference/api/rooms/steady-field/#steady_state_spl)
at `distance=None`), which is the level that drives the transmission
across the partition.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centre frequencies, Hz (1-D array). |
| `transmission_loss` | Transmission loss of the partition `TL`, dB; a scalar or one value per band. Measured, tabulated, or predicted by [`phonometry.building.prediction.panel_transmission`](/phonometry/reference/api/building/panel-transmission/). |
| `partition_area` | Area of the partition between the rooms `S_w`, m2. |
| `receiving_absorption` | Equivalent absorption area of the receiving room `S_2 alpha_2` per band, m2; e.g. from [`phonometry.room.equivalent_absorption_area`](/phonometry/reference/api/rooms/enclosed-space-absorption/#equivalent_absorption_area). |
| `source` | The source room ([`SourceRoom`](/phonometry/reference/api/noise_control/room-to-room/#sourceroom)): its level, or its sound power level with the room constant, directivity and sound power model that turn it into one. Exactly one of the two descriptions is required, so the empty default is rejected. |
| `include_partition_transmission` | When `True` the `tau S_w` term of Equation (4.101) is added to the receiving-room absorption, with $\tau = 10^{-\mathrm{TL}/10}$. Default `False`, the form hand calculations use. |
| `criterion` | The design criterion ([`DesignCriterion`](/phonometry/reference/api/noise_control/room-to-room/#designcriterion)): the family, the target curve and the flanking allowance. `None` is the default criterion, an `"NC"` family with no target. |
| `label` | A short human label of the chain. |

**Returns:** A [`RoomToRoomResult`](/phonometry/reference/api/noise_control/room-to-room/#roomtoroomresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the spectra do not share one value per band, if neither or both source descriptions are given, or if the criterion family or the sound power model is unknown. |

## RoomToRoomResult

```python
RoomToRoomResult(
    frequencies: np.ndarray,
    source_level: np.ndarray,
    transmission_loss: np.ndarray,
    partition_area: float,
    receiving_absorption: np.ndarray,
    noise_reduction: np.ndarray,
    received_level: np.ndarray,
    flanking_penalty: float,
    source_power_level: np.ndarray | None,
    criterion: str,
    target: float | None,
    label: str,
)
```

The room-to-room chain of one partition (Norton 2e, 4.9).

Built by [`room_to_room_transmission`](/phonometry/reference/api/noise_control/room-to-room/#room_to_room_transmission). The four spectra are the four
rows a hand calculation writes down: the level in the source room, the
transmission loss of the partition, the noise reduction the partition and
the receiving room deliver together, and what arrives.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centre frequencies, Hz. |
| `source_level` | Reverberant sound pressure level in the source room `L_p1`, dB. |
| `transmission_loss` | Transmission loss of the partition `TL`, dB. |
| `partition_area` | Area of the partition `S_w`, m2. |
| `receiving_absorption` | Equivalent absorption area of the receiving room `S_2 alpha_2` per band, m2. |
| `noise_reduction` | The delivered noise reduction `NR` per band, dB: Equation (4.101) less `flanking_penalty`. |
| `received_level` | Reverberant sound pressure level in the receiving room $L_{p2} = L_{p1} - \mathrm{NR}$, dB. |
| `flanking_penalty` | The debit applied to the predicted noise reduction for flanking transmission and air leaks, dB. |
| `source_power_level` | The source sound power level `L_W` the source level was built from, dB re 1 pW, or `None` when `source.level` was given directly. |
| `criterion` | `"NC"` or `"RC"`, the room-criterion family. |
| `target` | The design criterion value (e.g. `45` for NC 45), or `None`. |
| `label` | A short human label of the chain. |

### RoomToRoomResult.criterion_curve

*property*

The design curve of `target` at the analysis bands, dB.

`None` when the chain declared no target. Comparable band by band
with `received_level`.

### RoomToRoomResult.exceedance

*property*

How far the receiving room is over the criterion, band by band, dB.

Positive where the curve is exceeded, `None` without a target. It is
also the transmission-loss deficit of the partition, band for band.

### RoomToRoomResult.meets_target

*property*

`True` when the receiving room stays under the curve everywhere.

`None` without a target. The band-by-band test of a design sheet, not
the two-step designation procedure behind `rating`.

### RoomToRoomResult.plot()

```python
RoomToRoomResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the source-room level, the noise reduction and what arrives.

The source-room and receiving-room spectra are drawn against the design
criterion curve, with the band-by-band noise reduction on a twin axis.
Requires matplotlib (`pip install phonometry[plot]`).

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the received-level `Axes.plot`. |

**Returns:** The axes.

### RoomToRoomResult.rating

*property*

The rating of the spectrum arriving in the receiving room.

An [`NCResult`](/phonometry/reference/api/rooms/noise-criteria/#ncresult) for the `"NC"`
family, an [`RCResult`](/phonometry/reference/api/rooms/noise-criteria/#rcresult) for `"RC"`.

### RoomToRoomResult.required_transmission_loss

*property*

Partition `TL` that would just meet the criterion, per band, dB.

The chain of Equation (4.101) solved for the transmission loss,

$$
\mathrm{TL}_{req} = L_{p1} - L_{p2,\mathrm{target}} + 10 \log_{10}(S_w / S_2 \alpha_2) + \text{penalty},
$$

with $L_{p2,\mathrm{target}}$ the design criterion curve. The
`tau S_w` term is
left out of the inverse (it depends on the answer), which is how the
equation is used to specify a partition. `None` when no target was
declared.

### RoomToRoomResult.table()

```python
RoomToRoomResult.table() -> list[dict[str, Any]]
```

The chain as a list of printed rows, source room to verdict.

Each entry has `label`, `kind` (one of `"source_power"`,
`"source"`, `"transmission_loss"`, `"absorption"`,
`"noise_reduction"`, `"flanking"`, `"received"`, `"criterion"`,
`"required"`) and `values` (a per-band array). The rows are the ones
a hand calculation writes down, in that order.

**Returns:** The list of row dictionaries, in printing order.

## SourceRoom

```python
SourceRoom(
    level: ArrayLike | None = None,
    power_level: ArrayLike | None = None,
    room_constant: ArrayLike | None = None,
    directivity: float = 1.0,
    model: str = 'constant_power',
)
```

The source room of the chain: what drives the partition.

Two ways to say the same thing, and exactly one of them is given. Either
the reverberant level in the source room is known (`level`), or the
machine's sound power level is (`power_level`), in which case the room
constant of the source room is needed to build the reverberant field
$L_{p1} = L_W + 10 \log_{10}(4 / R_1)$ (Norton 2e, 4.7), and the
sound power model of Table 4.5 decides whether the position of the source
in the room raises or lowers the power it radiates.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | Reverberant sound pressure level in the source room `L_p1`, dB (scalar or per band). Mutually exclusive with `power_level`. |
| `power_level` | Sound power level of the source `L_W`, dB re 1 pW (scalar or per band). Requires `room_constant`. |
| `room_constant` | Room constant of the source room `R_1`, m2 (scalar or per band); from [`phonometry.room.room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant). |
| `directivity` | Directivity factor `Q` of the source in the source room (`1` in free space, `2` on one plane, `4` in an edge, `8` in a corner). Only affects the level through `model`, because the reverberant field itself is position-independent. |
| `model` | Sound power model of Norton Table 4.5: `"constant_power"` (default, the radiated power does not depend on the source position), `"constant_volume"` (the conservative upper bound, the power rises by $10 \log_{10} Q$) or `"constant_pressure"` (the lower bound, it falls by $10 \log_{10} Q$). |
