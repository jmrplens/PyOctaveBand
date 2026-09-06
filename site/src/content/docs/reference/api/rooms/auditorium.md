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
source puts out at 10 m in a free field (A.2.1, Equation (A.1)). It is one
of the two measures in Table A.1 that need a calibrated source, the other
being the late lateral sound level of A.2.5, which is referred to the same
place; and it is the reason the annex spends four equations on how to obtain
that free-field reference when there is no anechoic room 10 m across to
measure it in.

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
$400~\text{N s/m}^3$, which is the value that makes the three reference
quantities consistent: $p_0^2 S_0 / \rho c = (20\ \mu\mathrm{Pa})^2 / 400 = 1$ pW, the reference sound power. Neither equation prints that
caveat. Air at 20 degrees and 101,325 kPa is nearer $413~\text{N s/m}^3$,
worth 0,14 dB, an order of magnitude more than either rounding: the offsets
are a convention of the decibel scales, not a property of the air in the
hall, and this module does not make them follow the weather.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AuditoriumQuantity

```python
AuditoriumQuantity(
    symbol: str,
    aspect: str,
    averaging_bands_hz: tuple[float, ...],
    just_noticeable_difference: float | None,
    relative_jnd: bool,
    typical_range: tuple[float, float],
    energy_averaged: bool,
    unit: str,
)
```

One row of ISO 3382-1:2009, Table A.1.

`symbol` is the quantity as the table prints it, `aspect` the
subjective listener aspect it is grouped under, and
`averaging_bands_hz` the octave bands its single number is the average
over, which is **not** the same set for every row: five quantities
average two bands and two average four.

`just_noticeable_difference` is `None` for the late lateral sound
level, whose JND the table prints as "Not known"; it is a fraction
rather than an absolute difference when `relative_jnd` is True, which
is only the early decay time's "Rel. 5 %". `energy_averaged` is True
only for the late lateral level, which footnote a sends to
Equation (A.17) while every other row is averaged arithmetically.

`typical_range` is the pair the table prints, and footnote b conditions
it: frequency-averaged values at single positions in unoccupied concert
and multi-purpose halls up to 25 000 m³. It is not a range for one band,
for an occupied hall, or for a spatial average.

## AuditoriumWarning

A measurement outside the conditions ISO 3382-1:2009 prints for it.

## DIFFUSE_FIELD_REFERENCE_OFFSET_DB

*Constant* (`float`).

```python
DIFFUSE_FIELD_REFERENCE_OFFSET_DB = 37.0
```

## DIRECTIVITY_ARC_DEG

*Constant* (`float`).

```python
DIRECTIVITY_ARC_DEG = 30.0
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

## DIRECTIVITY_STEP_DEG

*Constant* (`float`).

```python
DIRECTIVITY_STEP_DEG = 5.0
```

## DIRECTIVITY_SURVEY_DISTANCE_M

*Constant* (`float`).

```python
DIRECTIVITY_SURVEY_DISTANCE_M = 1.5
```

## EARLY_ENERGY_LIMIT_S

*Constant* (`float`).

```python
EARLY_ENERGY_LIMIT_S = 0.08
```

## early_lateral_energy_fraction

```python
early_lateral_energy_fraction(
    ir: Signal | list[float] | NDArray[np.float64],
    lateral_ir: Signal | list[float] | NDArray[np.float64],
    fs: int | None = None,
    *,
    weighting: str = 'squared',
    limits: tuple[float, float] | None = (125.0, 4000.0),
    fraction: int = 1,
) -> LateralEnergyResult
```

Early lateral energy fraction of a pair of responses, per band.

ISO 3382-1:2009, Equation (A.14), the share of the first 80 ms that
arrives from the side:

$$
J_\mathrm{LF} = \frac{\int_{0,005}^{0,080} p_L^2(t)\ \mathrm{d}t} {\int_{0}^{0,080} p^2(t)\ \mathrm{d}t}
$$

and Equation (A.15), the cosine-weighted variant that A.2.4 calls
subjectively more accurate:

$$
J_\mathrm{LFC} = \frac{\int_{0,005}^{0,080} \left| p_L(t) \cdot p(t) \right|\ \mathrm{d}t} {\int_{0}^{0,080} p^2(t)\ \mathrm{d}t}
$$

The modulus in (A.15) is printed and it matters: the figure-of-eight
response changes sign with the side a reflection comes from, so without
it two mirror-image reflections cancel to zero instead of adding.

The two lower limits differ. The numerator starts at 5 ms because the
null of the figure-of-eight microphone is pointed at the source and the
5 ms keeps whatever leaks through it out of the lateral share; the
denominator starts at 0 because the direct sound belongs to the total.

**Time zero comes from the omnidirectional response.** A figure-of-eight
microphone aimed as A.2.4 asks has no direct sound to trigger on, so its
own onset would land on the first strong reflection and shift both
limits by that much. Both integrals here are timed from the direct sound
of `ir`, band by band.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Impulse response measured with the omnidirectional microphone at the measurement point (1D). |
| `lateral_ir` | Impulse response measured at the same point with a figure-of-eight microphone whose null points at the source (1D). |
| `fs` | Sample rate in Hz. Required for bare arrays; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own. |
| `weighting` | `"squared"` for $J_\mathrm{LF}$, Equation (A.14), whose contributions vary as the square of the cosine of the angle of incidence; `"cosine"` for $J_\mathrm{LFC}$, Equation (A.15), whose contributions vary as the cosine itself. |
| `limits` | `(f_min, f_max)` band-centre limits in Hz; default the octave bands 125 Hz to 4 kHz. Table A.1 averages this quantity over 125 Hz to 1 kHz. `None` measures the broadband responses. |
| `fraction` | Bandwidth fraction (1 = octave, 3 = one-third octave). |

**Returns:** A [`LateralEnergyResult`](/phonometry/reference/api/rooms/auditorium/#lateralenergyresult) with one entry per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `weighting` is not one of [`LATERAL_WEIGHTINGS`](/phonometry/reference/api/rooms/auditorium/#lateral_weightings), if a response is not one-dimensional or is silent, if the two responses do not split into the same number of bands or are of different lengths, or if they are too short for the 80 ms window. |

## EARLY_LATERAL_WINDOW_S

*Constant* (`tuple`).

```python
EARLY_LATERAL_WINDOW_S = (0.005, 0.08)
```

## EARLY_SUPPORT_WINDOW_S

*Constant* (`tuple`).

```python
EARLY_SUPPORT_WINDOW_S = (0.02, 0.1)
```

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

## gliding_directivity_deviation

```python
gliding_directivity_deviation(levels: ArrayLike) -> NDArray[np.float64]
```

Deviation of each gliding arc from the whole-turn reference (4.2.1).

ISO 3382-1:2009, 4.2.1 qualifies a source by averaging its free-field
output over "gliding" 30 degree arcs and comparing each of them with the
reference, which is "a 360 degree energetic average in the measurement
plane". Where no turntable is available it asks for measurements every
5 degrees, and for gliding averages "each covering six neighbouring
points", which is 30 degrees of a 72-point survey.

Both averages here are energetic, which is what the printed word makes
of the reference and what the arcs must therefore be for the comparison
to mean anything. The standard does not say so of the arcs, and it does
not say whether the six points of a window lead, trail or straddle their
arc either; the window here starts at each measured bearing and runs
forwards, and it wraps, so a survey of `N` bearings gives `N`
deviations.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | Sound pressure levels around the source, in dB, one per bearing, evenly spaced over a full turn. |

**Returns:** One deviation per bearing, in dB, of the arc that starts there against the whole-turn reference.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the bearings do not divide the arc a whole number of times, or there are too few of them to make one. |

## IACC_EARLY_WINDOW_S

*Constant* (`tuple`).

```python
IACC_EARLY_WINDOW_S = (0.0, 0.08)
```

## IACC_JND

*Constant* (`float`).

```python
IACC_JND = 0.075
```

## IACC_LATE_START_S

*Constant* (`float`).

```python
IACC_LATE_START_S = 0.08
```

## IACC_SEARCH_S

*Constant* (`float`).

```python
IACC_SEARCH_S = 0.001
```

## interaural_cross_correlation

```python
interaural_cross_correlation(
    left: Signal | list[float] | NDArray[np.float64],
    right: Signal | list[float] | NDArray[np.float64],
    fs: int | None = None,
    *,
    window: tuple[float, float | None] = (0.0, None),
    limits: tuple[float, float] | None = (125.0, 4000.0),
    fraction: int = 1,
) -> InterauralCorrelationResult
```

Interaural cross correlation of a binaural response, per band.

ISO 3382-1:2009, Equation (B.1) defines the normalised function

$$
\mathrm{IACF}_{t_1,t_2}(\tau) = \frac{\int_{t_1}^{t_2} p_l(t)\, p_r(t + \tau)\ \mathrm{d}t} {\sqrt{\int_{t_1}^{t_2} p_l^2(t)\ \mathrm{d}t \int_{t_1}^{t_2} p_r^2(t)\ \mathrm{d}t}}
$$

and Equation (B.2) takes the coefficient from it:

$$
\mathrm{IACC}_{t_1,t_2} = \max \left| \mathrm{IACF}_{t_1,t_2} \right| \quad \text{for } -1\ \mathrm{ms} < \tau < +1\ \mathrm{ms}
$$

Both the square root and the modulus are printed, and both matter. The
root is what bounds the function by one; a text layer that drops it
leaves a quantity that is not a correlation at all. The modulus is what
makes two anti-phase ears give 1 rather than 0, which is the answer that
matches what they describe, two signals as dissimilar as they can be
only in sign.

B.4 puts three windows to it. The general form runs from the direct
sound to a time of the order of the reverberation time, which is the
default here; [`IACC_EARLY_WINDOW_S`](/phonometry/reference/api/rooms/auditorium/#iacc_early_window_s) is the early coefficient and
[`IACC_LATE_START_S`](/phonometry/reference/api/rooms/auditorium/#iacc_late_start_s) starts the reverberant one. Time zero is the
direct sound, located on the two channels together so that a listener
turned away from the source does not move it.

**Parameters**

| Name | Description |
| :--- | :--- |
| `left` | Impulse response at the entrance to the left ear canal (1D). |
| `right` | Impulse response at the entrance to the right ear canal (1D), of the same length. |
| `fs` | Sample rate in Hz. Required for bare arrays; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own. |
| `window` | `(t_1, t_2)` in seconds from the direct sound, with `t_2` `None` for the end of the response. |
| `limits` | `(f_min, f_max)` band-centre limits in Hz; default the octave bands 125 Hz to 4 kHz, which is the range B.4 names. `None` correlates the broadband responses. |
| `fraction` | Bandwidth fraction (1 = octave, 3 = one-third octave). |

**Returns:** An [`InterauralCorrelationResult`](/phonometry/reference/api/rooms/auditorium/#interauralcorrelationresult) with one entry per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a response is not one-dimensional or is silent, if the two channels are of different lengths, if the window starts before the direct sound or is empty or reversed, or if a band has no energy inside it. |

## InterauralCorrelationResult

```python
InterauralCorrelationResult(
    frequency: NDArray[np.float64] | None,
    coefficient: NDArray[np.float64],
    delay: NDArray[np.float64],
    lag: NDArray[np.float64],
    correlation: NDArray[np.float64],
)
```

Per-band interaural cross correlation (ISO 3382-1:2009, Annex B).

`frequency` holds the exact band centre frequencies in Hz, or is
`None` for a broadband measurement. `coefficient` is the IACC of
Equation (B.2), the largest magnitude the normalised correlation
function reaches inside the +/-1 ms search window, and `delay` the lag
in seconds at which it reaches it: positive when the right ear is the
later of the two, because (B.1) evaluates the right channel at
$t + \tau$.

`lag` and `correlation` carry the function itself over the search
window, one row per band, so it can be drawn rather than summarised.
B.4 assumes a just-noticeable difference of 0,075.

### InterauralCorrelationResult.plot()

```python
InterauralCorrelationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the correlation function over the search window, per band.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## late_lateral_average

```python
late_lateral_average(levels: ArrayLike) -> float
```

Frequency-average the late lateral level over its four octave bands.

ISO 3382-1:2009, Equation (A.17):

$$
L_{J,\mathrm{avg}} = 10 \lg \left[ 0{,}25 \sum_{i=1}^{4} 10^{L_{J_i}/10} \right] \ \mathrm{dB}
$$

The 0,25 is one quarter, so this is an energy **mean** and four equal
band values return that value unchanged. It is the one exception
footnote a of Table A.1 makes: every other quantity in that table is
averaged arithmetically over its bands, and only $L_J$ is averaged
over energy. The four bands are the 125 Hz, 250 Hz, 500 Hz and 1 kHz
octaves, in that order.

**Parameters**

| Name | Description |
| :--- | :--- |
| `levels` | The four octave-band values of $L_J$, in dB, in the order of [`LATE_LATERAL_AVERAGE_BANDS_HZ`](/phonometry/reference/api/rooms/auditorium/#late_lateral_average_bands_hz). |

**Returns:** $L_{J,\mathrm{avg}}$ in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If four values are not given, or they are not finite. |

## LATE_LATERAL_AVERAGE_BANDS_HZ

*Constant* (`tuple`).

```python
LATE_LATERAL_AVERAGE_BANDS_HZ = (125.0, 250.0, 500.0, 1000.0)
```

## late_lateral_sound_level

```python
late_lateral_sound_level(
    ir: Signal | list[float] | NDArray[np.float64],
    lateral_ir: Signal | list[float] | NDArray[np.float64],
    reference_ir: Signal | list[float] | NDArray[np.float64] | None = None,
    fs: int | None = None,
    *,
    reference_level: ArrayLike | None = None,
    limits: tuple[float, float] | None = (125.0, 4000.0),
    fraction: int = 1,
) -> LateLateralResult
```

Late lateral sound level of a measured response, per band.

ISO 3382-1:2009, Equation (A.16), the level of the lateral energy that
arrives after the early window, against the same free-field reference
the sound strength uses:

$$
L_J = 10 \lg \frac{\int_{0,080}^{\infty} p_L^2(t)\ \mathrm{d}t} {\int_{0}^{\infty} p_{10}^2(t)\ \mathrm{d}t} \ \mathrm{dB}
$$

Where $J_\mathrm{LF}$ is a fraction and cancels its own
calibration, $L_J$ is a level and does not: it needs the
calibrated omnidirectional source of A.2.1 and a free-field reference,
which reaches this function the same two ways it reaches
[`sound_strength`](/phonometry/reference/api/rooms/auditorium/#sound_strength). It also needs the relative sensitivity of the
two microphones to have been calibrated in a free field (A.3.2).

**Time zero comes from the omnidirectional response**, for the reason
given under [`early_lateral_energy_fraction`](/phonometry/reference/api/rooms/auditorium/#early_lateral_energy_fraction): a figure-of-eight
microphone aimed as A.2.5 asks has no direct sound to trigger on, and an
80 ms boundary placed on its first reflection is not the printed one.
`ir` supplies that time zero and nothing else.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Impulse response measured with the omnidirectional microphone at the measurement point (1D), for its direct sound. |
| `lateral_ir` | Impulse response measured at the same point with a figure-of-eight microphone whose null points at the source (1D). |
| `reference_ir` | Impulse response of the same source at 10 m in a free field (1D). Mutually exclusive with `reference_level`. |
| `fs` | Sample rate in Hz. Required for bare arrays; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own. |
| `reference_level` | The free-field reference exposure level $L_{pE,10}$ in dB, as a scalar or one value per band. Mutually exclusive with `reference_ir`. |
| `limits` | `(f_min, f_max)` band-centre limits in Hz; default the octave bands 125 Hz to 4 kHz. Equation (A.17) averages this quantity over 125 Hz to 1 kHz. `None` measures the broadband responses. |
| `fraction` | Bandwidth fraction (1 = octave, 3 = one-third octave). |

**Returns:** A [`LateLateralResult`](/phonometry/reference/api/rooms/auditorium/#latelateralresult) with one entry per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If neither or both reference forms are given, if a response is not one-dimensional or is silent, if the two hall responses are of different lengths, or if they are too short to reach the 80 ms boundary. |

## LATE_LATERAL_START_S

*Constant* (`float`).

```python
LATE_LATERAL_START_S = 0.08
```

## LATE_SUPPORT_WINDOW_S

*Constant* (`tuple`).

```python
LATE_SUPPORT_WINDOW_S = (0.1, 1.0)
```

## LateLateralResult

```python
LateLateralResult(
    frequency: NDArray[np.float64] | None,
    level: NDArray[np.float64],
    reference_level: NDArray[np.float64],
)
```

Per-band late lateral sound level (ISO 3382-1:2009, A.2.5).

`frequency` holds the exact band centre frequencies in Hz, or is
`None` for a broadband measurement. `level` is
$L_J$ in dB (Equation (A.16)) and `reference_level` the sound
pressure exposure level of the free-field response at 10 m it is
referred to, however that was obtained. Table A.1 gives
$L_J$ a typical range of -14 dB to +1 dB over the 125 Hz to 1 kHz
octave bands and no just-noticeable difference at all: "Not known".

### LateLateralResult.plot()

```python
LateLateralResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band late lateral level against the Table A.1 range.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## LATERAL_WEIGHTINGS

*Constant* (`tuple`).

```python
LATERAL_WEIGHTINGS = ('squared', 'cosine')
```

## LateralEnergyResult

```python
LateralEnergyResult(
    frequency: NDArray[np.float64] | None,
    energy_fraction: NDArray[np.float64],
    weighting: str,
)
```

Per-band early lateral energy fraction (ISO 3382-1:2009, A.2.4).

`frequency` holds the exact band centre frequencies in Hz, or is
`None` for a broadband measurement. `energy_fraction` is
$J_\mathrm{LF}$ or $J_\mathrm{LFC}$ depending on
`weighting`, which is `"squared"` for Equation (A.14) and
`"cosine"` for Equation (A.15). Both are dimensionless and Table A.1
gives them a just-noticeable difference of 0,05 and a typical range of
0,05 to 0,35 over the 125 Hz to 1 kHz octave bands.

### LateralEnergyResult.plot()

```python
LateralEnergyResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the per-band lateral fraction against the Table A.1 range.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## MAX_SOURCE_DIRECTIVITY_DEVIATION_DB

*Constant* (`dict`).

```python
MAX_SOURCE_DIRECTIVITY_DEVIATION_DB = {125.0: 1.0, 250.0: 1.0, 500.0: 1.0, 1000.0: 3.0, 2000.0: 5.0, 4000.0: 6.0}
```

## MAXIMUM_DIRECTIVITY_STEP_DEG

*Constant* (`float`).

```python
MAXIMUM_DIRECTIVITY_STEP_DEG = 12.5
```

## MID_FREQUENCY_OCTAVES_HZ

*Constant* (`tuple`).

```python
MID_FREQUENCY_OCTAVES_HZ = (500.0, 1000.0)
```

## MID_FREQUENCY_THIRD_OCTAVES_HZ

*Constant* (`tuple`).

```python
MID_FREQUENCY_THIRD_OCTAVES_HZ = (400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0)
```

## minimum_receiver_positions

```python
minimum_receiver_positions(seats: ArrayLike) -> NDArray[np.float64] | float
```

Fewest microphone positions a hall of a given size wants (Table A.2).

ISO 3382-1:2009, Table A.2 prints three pairs, 500 seats to 6 positions,
1 000 to 8 and 2 000 to 10. They lie exactly on a straight line in the
logarithm of the seat count, two positions per doubling:

$$
N_\mathrm{min} = 6 + 2 \lg_2 (S / 500)
$$

which reproduces all three printed integers to the last bit. A.4 caps
what that line may be used for: it asks for "a minimum of between 6 and
10 representative microphone positions", so the result is clamped to the
range the table covers, and this function will not extrapolate a
5 000-seat arena to thirteen positions on the strength of three rows.

A.4 also says the positions are to be evenly distributed over all
audience seating areas, and that a hall broken into separate areas such
as balconies and under-balcony areas will need more than this.

**Parameters**

| Name | Description |
| :--- | :--- |
| `seats` | The number of seats in the hall. |

**Returns:** The minimum number of positions, not rounded, clamped to the 6 to 10 the table and A.4 between them authorise.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a seat count is not positive. |

## MINIMUM_REFERENCE_DISTANCE_M

*Constant* (`float`).

```python
MINIMUM_REFERENCE_DISTANCE_M = 3.0
```

## octave_pair_averages

```python
octave_pair_averages(
    values: ArrayLike,
    frequency: ArrayLike,
) -> dict[str, float]
```

The low, mid and high pair averages of ISO 3382-1:2009, A.5.

A more concise presentation than a full band table: the 125 Hz and
250 Hz results averaged into a low-frequency one, 500 Hz and 1 kHz into
a mid-frequency one, and 2 kHz and 4 kHz into a high-frequency one, all
arithmetically.

This is a different product from the single number of
[`single_number_average`](/phonometry/reference/api/rooms/auditorium/#single_number_average), and the two must not be confused. They
coincide for the five quantities Table A.1 averages over 500 Hz and
1 kHz, and they do not for the lateral ones, whose single number spans
four bands. A.5 also warns that lateral energy fractions in the 4 kHz
octave are not usually thought to be subjectively important, so the high
pair means little for them.

**Parameters**

| Name | Description |
| :--- | :--- |
| `values` | Per-band values, one per entry of `frequency`. |
| `frequency` | The band centre frequencies in Hz. |

**Returns:** `{"low": ..., "mid": ..., "high": ...}` in the values' unit.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the two arrays disagree in length, or if the band axis does not carry all six octaves from 125 Hz to 4 kHz. |

## OCTAVE_PAIRS_HZ

*Constant* (`dict`).

```python
OCTAVE_PAIRS_HZ = {'low': (125.0, 250.0), 'mid': (500.0, 1000.0), 'high': (2000.0, 4000.0)}
```

## perceptibly_different

```python
perceptibly_different(
    symbol: str,
    first: ArrayLike,
    second: ArrayLike,
) -> bool
```

Whether two values of a quantity differ by its Table A.1 JND.

The comparison is absolute for every quantity but the early decay time,
whose JND the table prints as "Rel. 5 %", a fraction of the value rather
than a difference in seconds. The late lateral sound level has no JND at
all, printed as "Not known", and this function refuses to invent one.

**Parameters**

| Name | Description |
| :--- | :--- |
| `symbol` | The quantity, as [`TABLE_A1`](/phonometry/reference/api/rooms/auditorium/#table_a1) keys it. |
| `first` | One value, in the quantity's unit. |
| `second` | The other, of the same shape. |

**Returns:** True when the two differ by at least the just-noticeable difference, so a listener could be expected to hear it.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the symbol is not one Table A.1 prints, or if the table prints no just-noticeable difference for it. |

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
| `reverberation_room_level` | Spatial-average sound pressure exposure level measured in the reverberation room, in dB. The standard calls this $L_{pE}$ too, which is the same symbol it gave the level measured in the hall under test a page earlier; substituting (A.5) into (A.1) as printed would cancel the hall out of G altogether. The two roles get different names here, and `docs/ERRATA.md` carries the rest. |
| `absorption_area` | Equivalent sound absorption area of that room, in m², broadcast against `reverberation_room_level`. |

**Returns:** The reference exposure level at 10 m, in dB.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the absorption area is not a positive, finite area. |

## single_number_average

```python
single_number_average(
    symbol: str,
    values: ArrayLike,
    frequency: ArrayLike,
) -> float
```

The Table A.1 single number of one auditorium quantity.

Footnote a of ISO 3382-1:2009, Table A.1 makes this the arithmetical
average over the octave bands the table names for that quantity, with
one exception: the late lateral sound level, which
[`late_lateral_average`](/phonometry/reference/api/rooms/auditorium/#late_lateral_average) energy-averages through Equation (A.17).
A.5 asks for the index "m" to be applied to the symbol of the result.

The band set is per quantity and is not always the mid pair. The sound
strength, early decay time, clarity, definition and centre time average
the 500 Hz and 1 kHz octaves; the lateral quantities average 125 Hz to
1 kHz, four bands. A.5 prints both cases as examples for exactly that
reason, $G_m$ over two bands and $J_{\mathrm{LF}m}$ over
four, and an accessor that hard-codes the mid pair is wrong for half the
table.

**Parameters**

| Name | Description |
| :--- | :--- |
| `symbol` | The quantity, as [`TABLE_A1`](/phonometry/reference/api/rooms/auditorium/#table_a1) keys it. |
| `values` | Its per-band values, one per entry of `frequency`. |
| `frequency` | The band centre frequencies in Hz, as a result carries them. |

**Returns:** The single number, in the quantity's own unit.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the symbol is not one Table A.1 prints, if the two arrays disagree in length, or if the band axis does not carry every band the quantity averages. |

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

## source_directivity_limit

```python
source_directivity_limit(centre: float) -> float
```

The Table 1 limit for one octave band, in dB.

**Parameters**

| Name | Description |
| :--- | :--- |
| `centre` | Nominal octave band centre frequency in Hz. |

**Returns:** The maximum deviation from omnidirectionality the table prints.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If Table 1 has no row for that band. It prints six, 125 Hz to 4 kHz, and a survey in the 63 Hz or 8 kHz octave has no printed limit to be held to rather than the nearest one. |

## STAGE_DIRECT_WINDOW_S

*Constant* (`tuple`).

```python
STAGE_DIRECT_WINDOW_S = (0.0, 0.01)
```

## stage_support

```python
stage_support(
    ir: Signal | list[float] | NDArray[np.float64],
    fs: int | None = None,
    *,
    limits: tuple[float, float] | None = (250.0, 2000.0),
    fraction: int = 1,
) -> StageSupportResult
```

Early and late stage support of a platform response, per band.

ISO 3382-1:2009, Equations (C.1) and (C.2), the reflected energy a
musician hears from the platform against the direct sound of their own
instrument:

$$
ST_\mathrm{Early} = 10 \lg \frac{\int_{0,020}^{0,100} p^2(t)\ \mathrm{d}t} {\int_{0}^{0,010} p^2(t)\ \mathrm{d}t}\ \mathrm{dB}, \qquad ST_\mathrm{Late} = 10 \lg \frac{\int_{0,100}^{1,000} p^2(t)\ \mathrm{d}t} {\int_{0}^{0,010} p^2(t)\ \mathrm{d}t}\ \mathrm{dB}
$$

Both are measured with the source and the microphone 1,0 m apart on the
platform, at the same height, with nothing reflecting within 2 m
(C.2.1 and C.2.3). Because they share a denominator, their difference is
a property of the hall alone and does not depend on how loud the direct
sound was.

**Two limits are in the equations and not in the prose.** C.2.1
describes the early support as "the reflected energy within the first
0,1 s", but (C.1) starts at 20 ms, discarding the 10 ms to 20 ms
interval; C.2.2 describes the late support as "the reflected energy
after the first 0,1 s" with no upper bound, but (C.2) stops at one
second, which matters in any hall whose reverberation time is longer
than that. The equations govern, and `docs/ERRATA.md` records the rest.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ir` | Impulse response measured on the platform (1D), 1,0 m from the acoustic centre of an omnidirectional source. |
| `fs` | Sample rate in Hz. Required for a bare array; a [`Signal`](/phonometry/reference/api/io/io/#signal) brings its own. |
| `limits` | `(f_min, f_max)` band-centre limits in Hz; default the 250 Hz to 2 kHz octave bands C.2.4 averages over. `None` measures the broadband response. |
| `fraction` | Bandwidth fraction (1 = octave, 3 = one-third octave). C.2.4 asks for octave bands. |

**Returns:** A [`StageSupportResult`](/phonometry/reference/api/rooms/auditorium/#stagesupportresult) with one entry per band.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the response is not one-dimensional or is silent, if a band has no energy, or if the response is shorter than the one second Equation (C.2) integrates to. |

## STAGE_SUPPORT_BANDS_HZ

*Constant* (`tuple`).

```python
STAGE_SUPPORT_BANDS_HZ = (250.0, 500.0, 1000.0, 2000.0)
```

## STAGE_SUPPORT_DISTANCE_M

*Constant* (`float`).

```python
STAGE_SUPPORT_DISTANCE_M = 1.0
```

## STAGE_SUPPORT_HEIGHTS_M

*Constant* (`tuple`).

```python
STAGE_SUPPORT_HEIGHTS_M = (1.0, 1.5)
```

## STAGE_SUPPORT_POSITIONS

*Constant* (`int`).

```python
STAGE_SUPPORT_POSITIONS = 3
```

## STAGE_SUPPORT_SINGLE_NUMBER_STANDARD_DEVIATION_DB

*Constant* (`float`).

```python
STAGE_SUPPORT_SINGLE_NUMBER_STANDARD_DEVIATION_DB = 0.3
```

## STAGE_SUPPORT_STANDARD_DEVIATION_DB

*Constant* (`float`).

```python
STAGE_SUPPORT_STANDARD_DEVIATION_DB = 1.0
```

## StageSupportResult

```python
StageSupportResult(
    frequency: NDArray[np.float64] | None,
    early: NDArray[np.float64],
    late: NDArray[np.float64],
)
```

Per-band stage support (ISO 3382-1:2009, Annex C).

`frequency` holds the exact band centre frequencies in Hz, or is
`None` for a broadband measurement. `early` is
$ST_\mathrm{Early}$ in dB (Equation (C.1)) and `late`
$ST_\mathrm{Late}$ in dB (Equation (C.2)), both referred to the
same direct sound. Table C.1 gives their typical ranges as -24 dB to
-8 dB and -24 dB to -10 dB and prints "Not known" for both
just-noticeable differences, so this module has none.

### StageSupportResult.plot()

```python
StageSupportResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot both supports per band against the Table C.1 ranges.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

## TABLE_A1

*Constant* (`dict`).

## TEST_REPORT_ITEMS

*Constant* (`tuple`).

```python
TEST_REPORT_ITEMS = ('a statement that the measurements were made in conformity with ISO 3382-1', 'name and place of the room tested', 'sketch plan of the room, with an indication of the scale', 'volume of the room, with an explanation of how it is defined if the room is not completely enclosed', 'for rooms for speech and music, the number and type of seats, their upholstery and covering, and which parts of the seat are covered', 'a description of the shape and material of the walls and the ceiling', 'state or states of occupancy during measurements, and the number of occupants', 'condition of any variable equipment, such as curtains, public-address or electronic reverberation enhancement systems', 'for theatres, whether the safety curtain or decorative curtains were up or down', 'description, where appropriate, of the stage furnishing, including any concert enclosure', 'temperature and relative humidity in the room during the measurement', 'description of measuring apparatus, source and microphones, and whether tape recorders were employed', 'description of the sound signal used', 'coverage chosen, including the source and microphone positions and their heights, preferably shown on a plan', 'date of measurement and name of the measuring organization')
```
