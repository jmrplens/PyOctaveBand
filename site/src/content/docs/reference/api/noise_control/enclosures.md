---
title: "noise_control.enclosures"
description: "Insertion loss of a close or free-standing machine enclosure."
sidebar:
  label: "enclosures"
---

Insertion loss of a close or free-standing machine enclosure.

Wrapping a machine in a sealed enclosure reduces the radiated noise by the
transmission loss of its panels, *minus* a penalty for the reverberant build-up
inside the small, hard cavity. Bies, Hansen & Howard, *Engineering Noise
Control* 5th ed., §7.4.2 (Eqs. (7.103), (7.111)) write the net reduction as

$$
\mathrm{IL} = R - C, \qquad C = 10 \log_{10}\!\left[0.3 + \frac{S_\mathrm{E} (1 - \alpha_\mathrm{i})}{S_\mathrm{i} \alpha_\mathrm{i}}\right],
$$

where `R` is the field-incidence transmission loss of the enclosure panels,
`S_E` the external surface area, `S_i` the internal surface area (including
the machine) and `alpha_i` the mean absorption of the enclosure interior. The
reverberant term is exactly `S_E` over the interior **room constant**
$R_\mathrm{i} = S_\mathrm{i} \alpha_\mathrm{i} / (1 - \alpha_\mathrm{i})$
([`phonometry.room.room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant)), so

$$
C = 10 \log_{10}(0.3 + S_\mathrm{E} / R_\mathrm{i}).
$$

A hard interior (`alpha_i` small) makes `C` large and wastes much of the
panel `R`; lining the enclosure drives `C` toward its floor
$10 \log_{10} 0.3 = -5.2$ dB (a fully absorbing interior, where
$\mathrm{IL} = R + 5.2$).
Bies terms this net reduction the enclosure *noise reduction*; it is the
insertion loss of the enclosure.

Norton & Karczub, *Fundamentals of Noise and Vibration Analysis for Engineers*
2nd ed., 4.10 (Equation (4.115)) derive the same design equation from the same
power balance but **without the `0.3`**:

$$
\mathrm{IL} = R - 10 \log_{10}(S_\mathrm{E} / R_\mathrm{i}),
$$

so a well-lined enclosure keeps rising instead of levelling off at
$R + 5.2$. The two agree to a few tenths of a decibel while
$S_\mathrm{E} / R_\mathrm{i}$ stays above about
unity (a hard or lightly lined interior) and diverge once the lining takes over,
where Bies' floor is the safer statement. `model` selects between them, and
the difference is the reason a published worked answer has to be reproduced with
the model its author used.

Turned around, the same equation answers the question a designer actually asks:
*given the noise reduction I need, what transmission loss must the panels have?*
That is [`enclosure_required_transmission_loss`](/phonometry/reference/api/noise_control/enclosures/#enclosure_required_transmission_loss), which returns the required
`R` per band in the same [`EnclosureResult`](/phonometry/reference/api/noise_control/enclosures/#enclosureresult).

**The panel transmission loss `R` is supplied by the caller** -- measured, or
predicted by a panel model -- as a per-band array, a callable of frequency, or
a panel prediction result (a [`phonometry.building.SoundReductionResult`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult)
or [`phonometry.building.ApertureTransmissionResult`](/phonometry/reference/api/building/aperture-transmission/#aperturetransmissionresult), matched structurally
so no dependency on `building` is introduced). This module never predicts
`R` itself; it combines a given `R` with the interior absorption. The
interior room constant reuses [`phonometry.room.room_constant`](/phonometry/reference/api/rooms/steady-field/#room_constant).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## enclosure_insertion_loss

```python
enclosure_insertion_loss(
    panel_transmission_loss: ArrayLike | Callable[[NDArray[np.float64]], ArrayLike] | PanelTransmissionResult,
    external_area: float,
    internal_area: float,
    internal_absorption: ArrayLike,
    *,
    frequencies: ArrayLike | None = None,
    model: str = 'bies',
) -> EnclosureResult
```

Net insertion loss of a machine enclosure (Bies Eqs. (7.103), (7.111)).

$\mathrm{IL} = R - C$ with
$C = 10 \log_{10}(0.3 + S_\mathrm{E} / R_\mathrm{i})$ and the interior room
constant $R_\mathrm{i} = S_\mathrm{i} \alpha_\mathrm{i} / (1 - \alpha_\mathrm{i})$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `panel_transmission_loss` | Panel transmission loss `R` per band, dB. One of: a per-band array (measured); a callable mapping a frequency array to per-band `R` (then `frequencies` is required); or a panel prediction result carrying `transmission_loss` and `frequencies`, such as the [`SoundReductionResult`](/phonometry/reference/api/building/panel-transmission/#soundreductionresult) of [`phonometry.building.single_panel_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#single_panel_transmission_loss) / [`phonometry.building.double_wall_transmission_loss`](/phonometry/reference/api/building/panel-transmission/#double_wall_transmission_loss) or the [`ApertureTransmissionResult`](/phonometry/reference/api/building/aperture-transmission/#aperturetransmissionresult) of [`phonometry.building.composite_transmission_loss`](/phonometry/reference/api/building/aperture-transmission/#composite_transmission_loss) (its `frequencies` are then used unless *frequencies* is given). This function does not predict `R` itself. |
| `external_area` | External enclosure surface area `S_E`, m2. |
| `internal_area` | Internal surface area `S_i` (including the machine), m2. |
| `internal_absorption` | Mean interior absorption `alpha_i` in `(0, 1)` (scalar or per-band). |
| `frequencies` | Band centre frequencies, Hz; required when `panel_transmission_loss` is a callable, optional otherwise (used to label the result and the plot). |
| `model` | Interior build-up model, one of [`ENCLOSURE_MODELS`](/phonometry/reference/api/noise_control/enclosures/#enclosure_models): `"bies"` (default) carries the `0.3` floor of Bies Equation (7.111), `"norton"` the bare $C = 10 \log_{10}(S_\mathrm{E} / R_\mathrm{i})$ of Norton & Karczub Equation (4.115). |

**Returns:** An [`EnclosureResult`](/phonometry/reference/api/noise_control/enclosures/#enclosureresult).

## ENCLOSURE_MODELS

*Constant* (`dict`).

```python
ENCLOSURE_MODELS = {'bies': 0.3, 'norton': 0.0}
```

## enclosure_required_transmission_loss

```python
enclosure_required_transmission_loss(
    insertion_loss: ArrayLike | Callable[[NDArray[np.float64]], ArrayLike],
    external_area: float,
    internal_area: float,
    internal_absorption: ArrayLike,
    *,
    frequencies: ArrayLike | None = None,
    model: str = 'bies',
) -> EnclosureResult
```

Panel `R` an enclosure needs to deliver a given insertion loss.

The design equation of [`enclosure_insertion_loss`](/phonometry/reference/api/noise_control/enclosures/#enclosure_insertion_loss) solved the other
way, $R = \mathrm{IL} + C$: the enclosure geometry and its interior
lining fix the
build-up correction `C`, and the panels have to make up the rest. This is
the number an enclosure is specified from, and the form Norton & Karczub use
in 4.10 when the target `IL` is the gap between the level a machine
produces in the room and a noise-criterion curve.

**Parameters**

| Name | Description |
| :--- | :--- |
| `insertion_loss` | Required insertion loss `IL` per band, dB (a per-band array, or a callable of the frequency array, in which case `frequencies` is required). |
| `external_area` | External enclosure surface area `S_E`, m2. |
| `internal_area` | Internal surface area `S_i`, m2, including the surface of the enclosed machine. |
| `internal_absorption` | Mean interior absorption `alpha_i` in `(0, 1)` (scalar or per-band); e.g. from [`phonometry.room.mean_absorption`](/phonometry/reference/api/rooms/reverberation-prediction/#mean_absorption) over the lined enclosure walls and the machine surface. |
| `frequencies` | Band centre frequencies, Hz; required for a callable `insertion_loss`, optional otherwise. |
| `model` | Interior build-up model, one of [`ENCLOSURE_MODELS`](/phonometry/reference/api/noise_control/enclosures/#enclosure_models) (see [`enclosure_insertion_loss`](/phonometry/reference/api/noise_control/enclosures/#enclosure_insertion_loss)). |

**Returns:** An [`EnclosureResult`](/phonometry/reference/api/noise_control/enclosures/#enclosureresult) whose `panel_transmission_loss` is the **required** `R` and whose `insertion_loss` is the requested target, so it plots and reports like a forward calculation.

## EnclosureResult

```python
EnclosureResult(
    frequencies: np.ndarray | None,
    panel_transmission_loss: np.ndarray,
    correction: np.ndarray,
    insertion_loss: np.ndarray,
    external_area: float,
    internal_area: float,
    room_constant: np.ndarray,
)
```

Insertion loss of a machine enclosure over frequency (Bies §7.4.2).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Frequencies `f`, Hz, or `None` if the panel `R` was given as a bare per-band array with no frequency labels. |
| `panel_transmission_loss` | The supplied panel transmission loss `R` per band, dB. |
| `correction` | The interior-build-up correction `C` per band, dB. |
| `insertion_loss` | The net enclosure insertion loss $\mathrm{IL} = R - C$, dB. |
| `external_area` | External enclosure surface area `S_E`, m2. |
| `internal_area` | Internal surface area `S_i`, m2. |
| `room_constant` | Interior room constant `R_i` per band, m2. |

### EnclosureResult.plot()

```python
EnclosureResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the panel `R`, correction `C` and net insertion loss.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### EnclosureResult.report()

```python
EnclosureResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a machine-enclosure insertion-loss fiche to `path`.

Writes a one-page enclosure-performance sheet: the method-basis line
naming the Bies, Hansen & Howard insertion-loss model
(Engineering Noise Control 5th ed., section 7.4.2), an optional metadata
header (client, enclosed machine, test environment, instrumentation,
climate, date), a per-band table (nominal frequency, the supplied panel
transmission loss `R`, the interior build-up correction `C` and the
net insertion loss $\mathrm{IL} = R - C$) beside the `R`, `C`
and `IL` curves, the boxed mean insertion loss over the analysis
bands with the external and internal surface areas, an optional verdict
row against a declared minimum, and a method-basis strip stating
$\mathrm{IL} = R - C$ with
$C = 10 \log_{10}(0.3 + S_\mathrm{E} / R_\mathrm{i})$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the header (`client`, `specimen` the enclosed machine, `test_room` the test environment, `instrumentation`, `temperature`, `relative_humidity`, `pressure`, `test_date`), the footer identity (`laboratory`, `operator`, `report_id`, `notes`) and, via `requirement`, a declared minimum mean insertion loss (more insertion loss is better). The surface areas come from the result itself. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True` the per-band table adds the interior room constant `R_i` column. |
| `language` | Fiche language: `"en"` (default) or `"es"`. |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab (or, for the figure, matplotlib) is not installed (`pip install phonometry[report]`). |
