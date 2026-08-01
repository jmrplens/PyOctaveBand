---
title: "electroacoustics.sound_reinforcement"
description: "Gain before feedback of a sound-reinforcement system."
sidebar:
  label: "sound_reinforcement"
---

Gain before feedback of a sound-reinforcement system.

A public-address system is a closed loop: the loudspeaker feeds the audience,
but it also feeds the microphone that drives it. Long (*Architectural
Acoustics* 2nd ed., Chapter 18, Equations (18.13) to (18.24)) writes the loop
in terms of two decibel gains,

* the **open-loop system gain** $Z_S$ (Equation (18.17)), the level
  the loudspeaker produces at an average listener minus the level the
  talker produces at the microphone,

  $$
  Z_S = L_{H\text{-}L} - L_{T\text{-}M}
  $$

  so $Z_S = -6$ dB (a typical auditorium or church) means the
  amplified sound at the listener sits 6 dB below what the talker delivers
  to the microphone, i.e. a comfortable conversational level at twice the
  talker-to-microphone distance;

* the **feedback-loop gain** $G_S$ (Equation (18.18)), the part of
  that output that returns to the microphone,

  $$
  G_S = L_{H\text{-}M} - L_{H\text{-}L} + D_M(\theta)
  $$

  with $D_M(\theta)$ the directivity index of the microphone toward
  the loudspeaker *relative to* the talker (zero for an omnidirectional
  microphone, about -2 to -3 dB for a cardioid pointed at the talker).

Summing the infinite series of round trips (Equation (18.14)) makes the system
oscillate when the loop gain reaches unity, that is (Equation (18.16))

$$
Z_S + G_S = 0
$$

Long takes a **feedback stability margin** of 10 dB for an equalised system
(other authors quote 12 dB unequalised and 6 dB carefully equalised): 6 dB of
it covers a tone that adds in phase with a reflection from a hard surface, the
remaining 4 dB is safety. With several microphones open at once the returned
signals add at the mixer, which is accounted for by the *number of open
microphones* correction (Equation (18.23))
$\Delta L_\text{nom} = 10 \log_{10} N_m$. The stability criterion is
then Equation (18.24),

$$
Z_S + L_{H\text{-}M} + \Delta L_\text{nom} \le L_{H\text{-}L} - D_M(\theta) - 10
$$

:::note
Long prints Equation (18.24) with $+ D_M(\theta)$ on the
right-hand side, which contradicts Equations (18.20) to (18.22) it
generalises (and would flip the benefit of a directional microphone
into a penalty; see `docs/ERRATA.md`). This module implements the
sign of Equation (18.20), so that $N_m = 1$ reproduces Long's
own special cases: with $Z_S = -6$ dB the criterion collapses to
$L_{H\text{-}M} \le L_{H\text{-}L} - D_M(\theta) - 4$
(Equation (18.21)), which for an omnidirectional microphone puts the
loudspeaker level at the microphone 4 dB below the average level in
the audience, and for a cardioid ($D_M = -2$ dB) 2 dB below it
(Equation (18.22)).
:::

Note what the criterion does *not* contain: the loudspeaker type, its number
or its power. Only the sound field it produces at two points and the type and
orientation of the microphone matter. Long also excludes the reverberant field
deliberately, as it is uniform and therefore does not depend on where the
microphone is or where it points; the direct-to-reverberant ratio enters the
design separately, through the intelligibility table of his Table 18.3
(better than -3 dB excellent, -3 to -6 very good, -6 to -9 good, -9 to -12
fair, -12 to -15 poor, below -15 very poor).

This module is a closed-form design calculator anchored on those equations.
Long gives no fully numeric worked case for them, so the tests anchor on the
closed forms and on the special cases he states in words.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## CARDIOID_RELATIVE_DIRECTIVITY

*Constant* (`float`).

```python
CARDIOID_RELATIVE_DIRECTIVITY = -2.0
```

## DEFAULT_STABILITY_MARGIN

*Constant* (`float`).

```python
DEFAULT_STABILITY_MARGIN = 10.0
```

## feedback_loop_gain

```python
feedback_loop_gain(
    level_loudspeaker_at_microphone: float,
    level_loudspeaker_at_listener: float,
    *,
    microphone_directivity: float = 0.0,
) -> float
```

Feedback-loop gain $G_S$ (Long Equation (18.18)).

$G_S = L_{H\text{-}M} - L_{H\text{-}L} + D_M(\theta)$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level_loudspeaker_at_microphone` | Direct-field level $L_{H\text{-}M}$ the loudspeaker produces at the microphone, dB. |
| `level_loudspeaker_at_listener` | Direct-field level $L_{H\text{-}L}$ the loudspeaker produces at an average listener, dB. |
| `microphone_directivity` | Directivity index $D_M(\theta)$ of the microphone toward the loudspeaker relative to the talker, dB (0 for an omnidirectional microphone, about [`CARDIOID_RELATIVE_DIRECTIVITY`](/phonometry/reference/api/electroacoustics/sound-reinforcement/#cardioid_relative_directivity) for a cardioid). |

**Returns:** The feedback-loop gain $G_S$, dB.

## feedback_stability

```python
feedback_stability(
    open_loop_gain: float,
    level_loudspeaker_at_microphone: float,
    level_loudspeaker_at_listener: float,
    *,
    microphone_directivity: float = 0.0,
    open_microphones: int = 1,
    stability_margin: float = 10.0,
) -> FeedbackStabilityResult
```

Stability of a reinforcement loop (Long Equations (18.16) to
(18.24)).

The loop gain $Z_S + G_S + \Delta L_\text{nom}$ is compared
with the oscillation threshold of Equation (18.16) reduced by the
stability margin, that is Equation (18.24) written with the sign of
Equation (18.20) (see the module docstring).

**Parameters**

| Name | Description |
| :--- | :--- |
| `open_loop_gain` | Open-loop system gain $Z_S = L_{H\text{-}L} - L_{T\text{-}M}$, dB; about -6 dB for a typical auditorium or church. |
| `level_loudspeaker_at_microphone` | Direct-field level $L_{H\text{-}M}$ produced by the loudspeaker system at the microphone, dB. |
| `level_loudspeaker_at_listener` | Direct-field level $L_{H\text{-}L}$ produced by the loudspeaker system at an average listener, dB. |
| `microphone_directivity` | Directivity index $D_M(\theta)$ of the microphone toward the loudspeaker relative to the talker, dB. |
| `open_microphones` | Number $N_m$ of microphones open at once. |
| `stability_margin` | Required margin below oscillation, dB (default [`DEFAULT_STABILITY_MARGIN`](/phonometry/reference/api/electroacoustics/sound-reinforcement/#default_stability_margin), Long's equalised-system value). |

**Returns:** A [`FeedbackStabilityResult`](/phonometry/reference/api/electroacoustics/sound-reinforcement/#feedbackstabilityresult).

## FeedbackStabilityResult

```python
FeedbackStabilityResult(
    open_loop_gain: float,
    feedback_loop_gain: float,
    nom_correction: float,
    loop_gain: float,
    stability_margin: float,
    margin: float,
    headroom: float,
    is_stable: bool,
    maximum_open_loop_gain: float,
    maximum_level_at_microphone: float,
    level_loudspeaker_at_microphone: float,
    level_loudspeaker_at_listener: float,
    microphone_directivity: float,
    open_microphones: int,
)
```

Gain structure and stability verdict of a reinforcement loop.

**Attributes**

| Name | Description |
| :--- | :--- |
| `open_loop_gain` | Open-loop system gain $Z_S$, dB (Equation (18.17)). |
| `feedback_loop_gain` | Feedback-loop gain $G_S$, dB (Equation (18.18)). |
| `nom_correction` | Number-of-open-microphones correction $\Delta L_\text{nom}$, dB (Equation (18.23)). |
| `loop_gain` | Total loop gain $Z_S + G_S + \Delta L_\text{nom}$, dB. The system oscillates at 0 dB (Equation (18.16)). |
| `stability_margin` | Required margin below oscillation, dB. |
| `margin` | Margin actually available, `-loop_gain`, dB. |
| `headroom` | Gain that may still be added before the required margin is used up, `-stability_margin - loop_gain`, dB; negative when the criterion of Equation (18.24) is already violated. |
| `is_stable` | Whether the criterion of Equation (18.24) holds. |
| `maximum_open_loop_gain` | Largest $Z_S$ the loop tolerates, dB. |
| `maximum_level_at_microphone` | Largest $L_{H\text{-}M}$ the loop tolerates, dB, for the given $Z_S$ (Equations (18.20) to (18.22)). |
| `level_loudspeaker_at_microphone` | Input $L_{H\text{-}M}$, dB. |
| `level_loudspeaker_at_listener` | Input $L_{H\text{-}L}$, dB. |
| `microphone_directivity` | Input $D_M(\theta)$, dB. |
| `open_microphones` | Input $N_m$. |

### FeedbackStabilityResult.plot()

```python
FeedbackStabilityResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the gain structure against the oscillation and margin
lines.

Bars for $Z_S$, $G_S$ and $\Delta L_\text{nom}$
accumulate into the total loop gain, with the 0 dB oscillation
threshold of Equation (18.16) and the required stability margin
marked. Requires matplotlib (`pip install phonometry[plot]`).

## open_microphone_correction

```python
open_microphone_correction(open_microphones: int) -> float
```

Number-of-open-microphones correction (Long Equation (18.23)).

$\Delta L_\text{nom} = 10 \log_{10} N_m$: doubling the number
of simultaneously open microphones costs about 3 dB of gain before
feedback.

**Parameters**

| Name | Description |
| :--- | :--- |
| `open_microphones` | Number $N_m$ of microphones open at once (>= 1). |

**Returns:** The correction $\Delta L_\text{nom}$, dB.

## plot_sound_reinforcement_geometry

```python
plot_sound_reinforcement_geometry(
    talker_distance: float,
    microphone_distance: float,
    listener_distance: float,
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Draw the four points of a reinforcement feedback loop.

A schematic in the layout of Long's Figure 18.15, with each path annotated
with its own length: the talker `T` close in front of the microphone
`M`, the flown loudspeaker `H` above and downstage of it, and the
average listener `L` out in the audience. The signal path
`T -> M` and `H -> L` is solid, the feedback path `H -> M` dashed.
The two loudspeaker paths are the direct-field levels `L_H-M` and
`L_H-L` that drive
[`phonometry.electroacoustics.feedback_stability`](/phonometry/reference/api/electroacoustics/sound-reinforcement/#feedback_stability); the drawing is
deliberately *not* to scale, because a talker 0.3 m from the microphone
and a listener 20 m from the loudspeaker cannot share one usable scale.

**Parameters**

| Name | Description |
| :--- | :--- |
| `talker_distance` | Talker-to-microphone distance, m. |
| `microphone_distance` | Loudspeaker-to-microphone distance, m. |
| `listener_distance` | Loudspeaker-to-listener distance, m. |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | Label language, `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the feedback-path `Axes.plot`. |

**Returns:** The axes.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If any distance is not positive and finite. |
