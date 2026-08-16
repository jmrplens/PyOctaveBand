---
title: "underwater.propagation.closed_form"
description: "Underwater sound propagation: propagation loss (closed-form)."
sidebar:
  label: "closed_form"
---

Underwater sound propagation: propagation loss (closed-form).

Propagation loss `PL`, $N_\mathrm{PL}$ (dB) is the difference between
the source level in a given direction and the mean-square sound pressure level
at the receiver, $N_\mathrm{PL}(x) = L_\mathrm{S} - L_p(x)$ (ISO 18405:2017,
3.4.1.4). Here it is the sum of geometrical spreading and volume absorption:

* [`spreading_loss`](/phonometry/reference/api/underwater/closed-form/#spreading_loss) -- geometrical spreading, $20 \log_{10} R$
  (spherical), $10 \log_{10} R$ (cylindrical) or spherical-then-cylindrical
  (`"practical"`).
* [`seawater_absorption`](/phonometry/reference/api/underwater/closed-form/#seawater_absorption) -- the volume absorption coefficient
  $\alpha$ in
  dB/km, from three coexisting formulations selectable through `model`:
  Francois & Garrison (1982, the default and reference), Ainslie & McColm (1998,
  a legible simplification of it) and Thorp (1967, a frequency-only form).
* [`propagation_loss`](/phonometry/reference/api/underwater/closed-form/#propagation_loss) -- the total
  $\mathrm{PL} = \text{spreading} + \alpha R$ versus range,
  returned as a [`PropagationLossResult`](/phonometry/reference/api/underwater/closed-form/#propagationlossresult) with a `.plot()`.

Not *transmission loss*, which ISO 18405:2017, 3.4.1.3 reserves for the
reduction in a specified level between two specified points and whose Note 6 to
entry deprecates as a synonym of this quantity.

Sources (clean-room, implemented from the published equations): Francois &
Garrison, JASA 72 (1982) via Medwin & Clay; Ainslie & McColm, JASA 103 (1998);
Thorp (1967) via Etter (2003). Absorption is validated by the mutual agreement of
Francois-Garrison and Ainslie-McColm (~10 % as the latter's paper states;
marginally exceeded at the extreme corners of the stated domain, e.g. 10.4 % at
T = −6 °C / 1 MHz and 12.3 % at z = 7 km, a property of the published
simplification, both transcriptions verified digit-for-digit).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## propagation_loss

```python
propagation_loss(
    range_m: NDArray[np.float64] | list[float] | float,
    frequency_hz: float,
    *,
    law: str = 'spherical',
    temperature: float = 10.0,
    salinity: float = 35.0,
    depth: float = 0.0,
    ph: float = 8.0,
    model: str = 'francois-garrison',
    transition_range: float | None = None,
) -> PropagationLossResult
```

Total propagation loss
$\mathrm{PL} = \text{spreading} + \alpha R$ versus range.

**Parameters**

| Name | Description |
| :--- | :--- |
| `range_m` | Range(s) from the source, in metres (scalar or array). |
| `frequency_hz` | Acoustic frequency, in Hz. |
| `law` | Spreading law (see [`spreading_loss`](/phonometry/reference/api/underwater/closed-form/#spreading_loss)). |
| `temperature` | Temperature `T`, in degrees Celsius. |
| `salinity` | Salinity `S`, in parts per thousand. |
| `depth` | Depth, in metres. |
| `ph` | Acidity (default 8). |
| `model` | Absorption model (see [`seawater_absorption`](/phonometry/reference/api/underwater/closed-form/#seawater_absorption)). |
| `transition_range` | Transition range for the `"practical"` law, in m. |

**Returns:** A [`PropagationLossResult`](/phonometry/reference/api/underwater/closed-form/#propagationlossresult).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |

## PropagationLossResult

```python
PropagationLossResult(
    range_m: NDArray[np.float64],
    pl: NDArray[np.float64],
    spreading: NDArray[np.float64],
    absorption: NDArray[np.float64],
    frequency: float,
    absorption_coefficient: float,
    law: str,
    model: str,
)
```

Propagation loss versus range (closed-form).

**Attributes**

| Name | Description |
| :--- | :--- |
| `range_m` | Ranges from the source, in metres. |
| `pl` | Total propagation loss per range, in dB. |
| `spreading` | Geometrical-spreading contribution per range, in dB. |
| `absorption` | Volume-absorption contribution per range, in dB. |
| `frequency` | The acoustic frequency, in Hz. |
| `absorption_coefficient` | The absorption coefficient $\alpha$, in dB/km. |
| `law` | The spreading law used. |
| `model` | The absorption model used. |

### PropagationLossResult.plot()

```python
PropagationLossResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the propagation loss versus range with its two contributions.

## seawater_absorption

```python
seawater_absorption(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    *,
    temperature: float = 10.0,
    salinity: float = 35.0,
    depth: float = 0.0,
    ph: float = 8.0,
    model: str = 'francois-garrison',
) -> NDArray[np.float64]
```

Volume absorption coefficient $\alpha$, in dB/km.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequency_hz` | Acoustic frequency, in Hz (scalar or array). |
| `temperature` | Temperature `T`, in degrees Celsius. |
| `salinity` | Salinity `S`, in parts per thousand. |
| `depth` | Depth, in metres (`>= 0`). |
| `ph` | Acidity (used by Francois-Garrison and Ainslie-McColm; default 8). |
| `model` | `"francois-garrison"` (default), `"ainslie-mccolm"` or `"thorp"` (the Thorp 1967 frequency-only form of Etter, valid below ~50 kHz; ignores `temperature`/`salinity`/`depth`/`ph`). |

**Returns:** Absorption coefficient per frequency, in dB/km.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `model` is unknown or an input is invalid. |

## spreading_loss

```python
spreading_loss(
    range_m: NDArray[np.float64] | list[float] | float,
    *,
    law: str = 'spherical',
    transition_range: float | None = None,
) -> NDArray[np.float64]
```

Geometrical spreading loss, in dB.

`"spherical"` gives $20 \log_{10}(R)$ (free field), `"cylindrical"`
gives $10 \log_{10}(R)$ (perfect waveguide) and `"practical"` is
spherical up to `transition_range` `R0` then cylindrical:
$20 \log_{10}(R_0) + 10 \log_{10}(R/R_0)$ (mode stripping in a channel).

**Parameters**

| Name | Description |
| :--- | :--- |
| `range_m` | Range `R` from the source, in metres (scalar or array, strictly positive). |
| `law` | `"spherical"` (default), `"cylindrical"` or `"practical"`. |
| `transition_range` | Transition range `R0` in metres; required for `"practical"`. |

**Returns:** Spreading loss per range, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the inputs are invalid. |
