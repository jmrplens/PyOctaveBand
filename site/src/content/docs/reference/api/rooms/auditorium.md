---
title: "room.auditorium"
description: "Auditorium measures derived from impulse responses (ISO 3382-1:2009, Annex A)."
sidebar:
  label: "auditorium"
---

Auditorium measures derived from impulse responses (ISO 3382-1:2009, Annex A).

Annex A is **informative**. Nothing about the sound strength is normative in
ISO 3382-1:2009: not Equations (A.1) to (A.9), not the 1 dB just-noticeable
difference of Table A.1, not the typical range. What the annex fixes is the
definition everyone quotes, and that is what this module implements.

Reverberation time says what the room does to energy over time and nothing
about how loud the room is. Sound strength, G, is the quantity that does:
the energy of the measured impulse response against the energy the same
source puts out at 10 m in a free field (A.2.1, Equation (A.1)). It is the
one measure in Table A.1 that needs a calibrated source, and it is the
reason the annex spends four equations on how to obtain that free-field
reference when there is no anechoic room 10 m across to measure it in.

The three printed routes to the reference are all here, and they are
routes to the same number:

- measure at a distance `d >= 3 m` and correct by the inverse square law,
  Equations (A.4) and (A.8);
- measure the source in a reverberation room of known absorption area,
  Equation (A.5);
- take the source's sound power level and subtract the free-field spread,
  Equation (A.9).

The last two carry printed integers, 37 dB and 31 dB, and both are the
correctly rounded value of a closed form: $10\lg(1600\pi)$ is
37,0127 dB and $10\lg(400\pi)$ is 30,9921 dB. Rounded to whole
decibels they land 6 dB apart, where the exact offsets are
$10\lg 4 = 6{,}0206$ dB apart, so the reverberation-room route and
the sound-power route cannot agree to better than 0,0206 dB. That is
2 % of the 1 dB just-noticeable difference Table A.1 prints for G, and
this module reproduces the standard's integers rather than the closed
forms: a library that quietly used 30,9921 dB would disagree with every
hand calculation done from the printed page.

Both closed forms hold at a characteristic impedance of exactly
400 N s m\ :sup:`-3`, which is the value that makes the three reference
quantities consistent: $p_0^2 S_0 / \rho c = (20\ \mu\mathrm{Pa})^2 / 400 = 1$ pW, the reference sound power. Neither equation prints that
caveat. Air at 20 degrees and 101,325 kPa is nearer 413 N s m\ :sup:`-3`,
worth 0,14 dB, an order of magnitude more than either rounding: the offsets
are a convention of the decibel scales, not a property of the air in the
hall, and this module does not make them follow the weather.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AuditoriumWarning

A measurement outside the conditions ISO 3382-1:2009 prints for it.

## DIFFUSE_FIELD_REFERENCE_OFFSET_DB

*Constant* (`float`).

```python
DIFFUSE_FIELD_REFERENCE_OFFSET_DB = 37.0
```

## directivity_energy_average

```python
directivity_energy_average(
    levels: ArrayLike,
    axis: int = -1,
) -> NDArray[np.float64] | float
```

Average a free-field measurement over bearings around the source.

The note under ISO 3382-1:2009, Equation (A.4) asks for the free-field
reference to be measured all the way round the source and combined as
an energy mean, so that the source's own directivity does not decide
the reference level:

$$
\overline{L} = 10 \lg \left( \frac{1}{N} \sum_{i=1}^{N} 10^{L_i/10} \right)\ \mathrm{dB}
$$

An arithmetic mean of the decibel values is a different number, and a
lower one, for any source that is not omnidirectional.

The note asks for the measurement "at every 12,5 degrees", and 12,5
degrees does not divide 360: 28 steps stop at 350 degrees and 29
overshoot to 362,5. There is no set of bearings that follows the
printed instruction literally, so what this function checks is what
the instruction can mean, a uniform sampling of the full turn no
coarser than the printed step: `N >= 29` bearings, 360/N degrees
apart. The 5 degree survey of 4.2.1, which the same standard uses for
the source directivity of Table 1, does divide 360 exactly, into 72.

The mean itself is the one
[`phonometry.building.energy_average_level`](/phonometry/reference/api/building/insulation/#energy_average_level) computes; this
function adds the bearing count the note constrains, and averages one
band per row when it is handed a two-dimensional survey.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | The levels measured at the bearings, in dB, one per bearing, evenly spaced around the source. A two-dimensional array averages one band per row (or per column, see `axis`), which is the shape the rest of this module works in. |
| `axis` | Axis the bearings run along; the last by default. |

**Returns:** Their energy mean, in dB: a `float` for a single turn, an array with `axis` removed for several.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If fewer than `ceil(360 / 12,5) = 29` bearings lie along `axis`, or the levels are empty or non-finite. |

## free_field_reference_level

```python
free_field_reference_level(
    level: ArrayLike,
    distance: float,
) -> NDArray[np.float64] | float
```

Free-field level at 10 m from one measured at another distance.

ISO 3382-1:2009, Equations (A.4) and (A.8), which are the same
inverse-square correction written once for the exposure level and once
for the stationary-source pressure level:

$$
L_{pE,10} = L_{pE,d} + 20 \lg (d/10)\ \mathrm{dB}
$$

Both are printed for a point "d (>= 3 m) from the source", far enough
for the free field to have taken over; a shorter distance raises
[`AuditoriumWarning`](/phonometry/reference/api/rooms/auditorium/#auditoriumwarning).

The note under (A.4) adds that the measurement is to be repeated around
the source and energy-averaged, "at every 12,5 degrees", to average out
the source's own directivity. Feed this function the averaged level,
not one bearing; [`directivity_energy_average`](/phonometry/reference/api/rooms/auditorium/#directivity_energy_average) is that mean, and
its docstring says what the printed step does and does not determine.

**Parameters**

| Name | Description |
| :--- | :--- |
| `level` | The level measured at `distance`, in dB. Any shape. |
| `distance` | Source-to-microphone distance of that measurement, in metres. |

**Returns:** The level referred to 10 m, in dB, in the shape of `level`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `distance` is not a positive, finite length. |

## MAXIMUM_DIRECTIVITY_STEP_DEG

*Constant* (`float`).

```python
MAXIMUM_DIRECTIVITY_STEP_DEG = 12.5
```

## MINIMUM_REFERENCE_DISTANCE_M

*Constant* (`float`).

```python
MINIMUM_REFERENCE_DISTANCE_M = 3.0
```

## REFERENCE_DISTANCE_M

*Constant* (`float`).

```python
REFERENCE_DISTANCE_M = 10.0
```

## reverberation_room_reference_level

```python
reverberation_room_reference_level(
    reverberation_room_level: ArrayLike,
    absorption_area: ArrayLike,
) -> NDArray[np.float64] | float
```

Free-field reference level at 10 m from a reverberation-room measurement.

ISO 3382-1:2009, Equation (A.5):

$$
L_{pE,10} = L_{pE} + 10 \lg (A/S_0) - 37\ \mathrm{dB}
$$

with $S_0 = 1$ m². The route exists because a room 10 m across
and anechoic is rarer than a reverberation room: the source is measured
in the diffuse field instead, and the absorption area converts that
reading into the free-field one. `A` follows from the room's
reverberation time through Sabine's formula, Equation (A.6), which the
library publishes as
[`phonometry.room.sabine_absorption_area`](/phonometry/reference/api/rooms/steady-field/#sabine_absorption_area). Watch the constant when
reproducing a hand calculation: (A.6) prints $A = 0{,}16\,V/T$,
which is $24 \ln 10 / c_0$ at $c_0 = 345{,}4$ m/s, while
that function defaults to 343 m/s and so to 0,1611. The difference moves
$10\lg(A/S_0)$ by 0,030 dB; pass `speed_of_sound=345.39` to get
the printed constant back.

The printed 37 dB is $10\lg(1600\pi) = 37{,}0127$ dB rounded;
see the module docstring for what that rounding costs.

(A.5) also carries no Waterhouse correction, unlike the reverberation-room
sound power method of ISO 3741 that it otherwise mirrors. The omitted
$10\lg(1 + S\lambda/8V)$ is worth over a decibel in the 125 Hz
band of a small room, above the 1 dB just-noticeable difference Table A.1
gives G. That is a property of the printed method, and this function
reproduces the method rather than quietly improving it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `reverberation_room_level` | Spatial-average sound pressure exposure level measured in the reverberation room, in dB. The standard calls this $L_{pE}$ too, which is the same symbol it gave the level measured in the hall under test five equations earlier; substituting (A.5) into (A.1) as printed would cancel the hall out of G altogether. The two roles get different names here, and :doc:`the errata register </reference/errata>` carries the rest. |
| `absorption_area` | Equivalent sound absorption area of that room, in m², broadcast against `reverberation_room_level`. |

**Returns:** The reference exposure level at 10 m, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the absorption area is not positive. |

## sound_pressure_exposure_level

```python
sound_pressure_exposure_level(
    ir: Signal | list[float] | NDArray[np.float64],
    fs: int | None = None,
    *,
    limits: tuple[float, float] | None = (125.0, 4000.0),
    fraction: int = 1,
) -> NDArray[np.float64] | float
```

Sound pressure exposure level of an impulse response, per band.

ISO 3382-1:2009, Equation (A.2):

$$
L_{pE} = 10 \lg \left[ \frac{1}{T_0} \int_0^{\infty} \frac{p^2(t)}{p_0^2} \mathrm{d}t \right] \ \mathrm{dB}
$$

with $T_0 = 1$ s and $p_0 = 20$ uPa. Time zero is the start
of the direct sound (A.2.1), found per band with the A.3.4 trigger, and
the integral runs to the end of the response supplied.

A.2.1 asks that end to lie at or beyond the point where the decay curve
has fallen 30 dB. Whether a room response reaches it is visible in the
`dynamic_range` of
[`phonometry.room.room_parameters`](/phonometry/reference/api/rooms/acoustics/#room_parameters), and
[`sound_strength`](/phonometry/reference/api/rooms/auditorium/#sound_strength) raises [`AuditoriumWarning`](/phonometry/reference/api/rooms/auditorium/#auditoriumwarning) when it does
not; this function evaluates whatever it is given, because the same
equation is (A.3) applied to a free-field response, which has no
reverberant decay to reach 30 dB of.

This is the quantity Equations (A.3) to (A.5) also operate on: the same
function applied to the free-field response at 10 m gives
$L_{pE,10}$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Measured impulse response (1D). A [`phonometry.io.Signal`](/phonometry/reference/api/io/io/#signal) brings its calibration, which is applied; a bare array is read as pascals, so its exposure level is only referenced to 20 uPa if the samples already are. |
| `fs` | Sample rate in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own, and an explicit value that disagrees with it raises instead of silently winning. |
| `limits` | `(f_min, f_max)` band-centre limits in Hz; default the octave bands 125 Hz to 4 kHz (ISO 3382-1:2009, 5.1). `None` integrates the broadband response as a single band. |
| `fraction` | Bandwidth fraction (1 = octave, 3 = one-third octave). |

**Returns:** The exposure level in dB, one entry per band, or a `float` for a broadband response.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the response is not one-dimensional or is silent, if `limits` is malformed, or if a band has no energy. |

## sound_strength

```python
sound_strength(
    ir: Signal | list[float] | NDArray[np.float64],
    reference_ir: Signal | list[float] | NDArray[np.float64] | None = None,
    fs: int | None = None,
    *,
    reference_level: ArrayLike | None = None,
    limits: tuple[float, float] | None = (125.0, 4000.0),
    fraction: int = 1,
) -> SoundStrengthResult
```

Sound strength G of a measured impulse response, per band.

ISO 3382-1:2009, Equation (A.1):

$$
G = 10 \lg \frac{\int_0^{\infty} p^2(t) \mathrm{d}t} {\int_0^{\infty} p_{10}^2(t) \mathrm{d}t} = L_{pE} - L_{pE,10}\ \mathrm{dB}
$$

Equation (A.7) is the same difference between stationary-source levels
rather than exposure levels, which is why this function accepts the
reference either as a second impulse response or as a level already in
hand: exactly one of `reference_ir` and `reference_level` is
required. A level obtained from
[`free_field_reference_level`](/phonometry/reference/api/rooms/auditorium/#free_field_reference_level),
[`reverberation_room_reference_level`](/phonometry/reference/api/rooms/auditorium/#reverberation_room_reference_level) or, through
[`sound_strength_from_power`](/phonometry/reference/api/rooms/auditorium/#sound_strength_from_power), from the source's power level, goes
in the second slot.

Both responses are split into the same bands and each is integrated
from its own direct sound, so a common gain on the pair cancels
exactly. A gain applied to only one of them does not: the two
recordings must share a calibration, which is the whole reason G needs
a calibrated source where every other measure in Table A.1 does not.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Impulse response measured in the room under test (1D). |
| `reference_ir` | Impulse response of the same source at 10 m in a free field (1D). Mutually exclusive with `reference_level`. |
| `fs` | Sample rate in Hz. Required for bare arrays; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own. |
| `reference_level` | The free-field reference exposure level $L_{pE,10}$ in dB, as a scalar or one value per band. Mutually exclusive with `reference_ir`. |
| `limits` | `(f_min, f_max)` band-centre limits in Hz; default the octave bands 125 Hz to 4 kHz (ISO 3382-1:2009, 5.1). `None` measures the broadband response as a single band. |
| `fraction` | Bandwidth fraction (1 = octave, 3 = one-third octave). |

**Returns:** A [`SoundStrengthResult`](/phonometry/reference/api/rooms/auditorium/#soundstrengthresult) with one entry per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If neither or both reference forms are given, if a response is not one-dimensional or is silent, or if `reference_level` does not broadcast onto the band axis. |

## sound_strength_from_power

```python
sound_strength_from_power(
    pressure_level: ArrayLike,
    power_level: ArrayLike,
) -> NDArray[np.float64] | float
```

Sound strength from the source's sound power level.

ISO 3382-1:2009, Equation (A.9):

$$
G = L_p - L_W + 31\ \mathrm{dB}
$$

The third route to G, and the only one that needs no free-field
measurement at all: the 31 dB is the spread of a point source over the
sphere of radius 10 m, $10\lg(4\pi \cdot 100) = 30{,}9921$ dB,
rounded. A.2.1 asks for the source's power level to be measured to
ISO 3741, which the library implements in
[`phonometry.emission.sound_power`](/phonometry/reference/api/power/sound-power/).

**Parameters**

| Name | Description |
| :--- | :--- |
| `pressure_level` | Sound pressure level at the measurement point in the room under test, in dB. |
| `power_level` | Sound power level of the source, in dB, broadcast against `pressure_level`. |

**Returns:** The sound strength G, in dB.

## SOUND_STRENGTH_POWER_OFFSET_DB

*Constant* (`float`).

```python
SOUND_STRENGTH_POWER_OFFSET_DB = 31.0
```

## SoundStrengthResult

```python
SoundStrengthResult(
    frequency: NDArray[np.float64] | None,
    strength: NDArray[np.float64],
    exposure_level: NDArray[np.float64],
    reference_level: NDArray[np.float64],
)
```

Per-band sound strength G and the two levels it is the difference of.

`frequency` holds the exact band centre frequencies in Hz, or is
`None` for a broadband measurement, in which case every array has
length 1. `strength` is G in dB (ISO 3382-1:2009, Equation (A.1)),
`exposure_level` the sound pressure exposure level of the response
measured in the room (Equation (A.2)) and `reference_level` that of
the free-field response at 10 m (Equation (A.3)), however it was
obtained. All three are in decibels and `strength` is exactly
`exposure_level - reference_level`.

### SoundStrengthResult.plot()

```python
SoundStrengthResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes | np.ndarray
```

Plot the per-band sound strength against the Table A.1 range.

With `ax` given, only the strength panel is drawn on it; otherwise a
second panel shows the two levels G is the difference of. Requires
matplotlib (`pip install phonometry[plot]`); returns the
`Axes` (or an array of two).
