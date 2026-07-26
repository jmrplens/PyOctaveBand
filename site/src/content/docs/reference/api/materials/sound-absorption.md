---
title: "materials.sound_absorption"
description: "Sound absorption in a reverberation room: BS EN ISO 354:2003."
sidebar:
  label: "sound_absorption"
---

Sound absorption in a reverberation room: BS EN ISO 354:2003.

The mean reverberation time of a reverberation room is measured empty and with
the test specimen installed. From those two reverberation times the equivalent
sound absorption area of the specimen is obtained via Sabine's equation, and for
a plane absorber the sound absorption coefficient follows by dividing by the
covered area (ISO 354:2003, Clauses 4 and 8.1).

Equivalent sound absorption area (ISO 354:2003, Eq. (5) empty room / Eq. (7) with
specimen; identical form):

```text
A = 55,3 * V / (c * T) - 4 * V * m
```

with `V` the room volume (m3), `c` the speed of sound (m/s), `T` the
reverberation time (s) and `m` the power attenuation coefficient of air (1/m).
The speed of sound follows Eq. (6), valid for 15 degC to 30 degC:

```text
c = (331 + 0,6 * t/degC) m/s
```

The equivalent sound absorption area of the specimen and its absorption
coefficient (ISO 354:2003, Eq. (8) and Eq. (9)):

```text
AT = A2 - A1 = 55,3 * V * (1/(c2*T2) - 1/(c1*T1)) - 4 * V * (m2 - m1)
alpha_s = AT / S
```

`alpha_s` may exceed 1,0 (e.g. from diffraction/edge effects) and is not a
percentage (ISO 354:2003, Clause 3.7 NOTE 2); it is therefore never clamped.

The air attenuation coefficient `m` is defined by ISO 354 only through its
conversion from the ISO 9613-1 attenuation coefficient `alpha` (in dB/m)
(ISO 354:2003, 8.1.2.1):

```text
m = alpha / (10 * lg e)
```

ISO 354 otherwise defers the calculation of `alpha` entirely to ISO 9613-1.
`m` is therefore a user-supplied per-band parameter here (default 0, i.e. no air
correction); a caller holding ISO 9613-1 `alpha` values can convert them with
[`attenuation_from_alpha`](/phonometry/reference/api/materials/sound-absorption/#attenuation_from_alpha).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## absorption_area

```python
absorption_area(
    t60: ArrayLike,
    volume: float,
    *,
    temperature: float = 20.0,
    speed_of_sound: float | None = None,
    m: ArrayLike = 0.0,
) -> NDArray[np.float64]
```

Equivalent sound absorption area of a room (ISO 354:2003, Eq. (5)/(7)).

`A = 55,3 * V / (c * T) - 4 * V * m`. This is Sabine's equation with the
air-absorption term; it gives the empty-room area `A1` from `T1` or the
with-specimen area `A2` from `T2` (both equations have identical form).

**Parameters**

| Name | Description |
| :--- | :--- |
| `t60` | Reverberation time(s) `T`, in seconds (scalar or per band). |
| `volume` | Room volume `V`, in cubic metres. |
| `temperature` | Air temperature, in degrees Celsius, used to compute the speed of sound via Eq. (6) when `speed_of_sound` is not given (default 20 degC, i.e. c = 343 m/s). A temperature outside 15..30 degC emits an [`AbsorptionWarning`](/phonometry/reference/api/materials/sound-absorption/#absorptionwarning). A room volume below the 150 m3 minimum of clause 6.1.1 likewise emits an advisory [`AbsorptionWarning`](/phonometry/reference/api/materials/sound-absorption/#absorptionwarning). |
| `speed_of_sound` | Explicit speed of sound `c`, in m/s; overrides `temperature` and Eq. (6) when supplied. |
| `m` | Power attenuation coefficient of air `m`, in 1/m (a scalar or an array matching the shape of `t60`; default 0, i.e. no air correction). A per-band `m` whose shape differs from `t60` raises `ValueError`. Obtain it from an ISO 9613-1 attenuation coefficient with [`attenuation_from_alpha`](/phonometry/reference/api/materials/sound-absorption/#attenuation_from_alpha). |

**Returns:** Equivalent sound absorption area `A`, in square metres, with the shape of `t60`.

## absorption_coefficient

```python
absorption_coefficient(
    t1: ArrayLike,
    t2: ArrayLike,
    volume: float,
    sample_area: float,
    *,
    temperature1: float = 20.0,
    temperature2: float | None = None,
    speed_of_sound1: float | None = None,
    speed_of_sound2: float | None = None,
    m1: ArrayLike = 0.0,
    m2: ArrayLike = 0.0,
) -> NDArray[np.float64]
```

Sound absorption coefficient of a plane absorber (ISO 354:2003, Eq. (9)).

Builds the equivalent sound absorption area of the specimen from Eq. (8),
`AT = A2 - A1 = 55,3*V*(1/(c2*T2) - 1/(c1*T1)) - 4*V*(m2 - m1)`, using the
empty-room reverberation time `T1` and the with-specimen time `T2`, then
returns `alpha_s = AT / S` (Eq. (9)).

The two measurements may be at different temperatures; `c1` and `c2` are
resolved independently. `alpha_s` is returned unclamped and may exceed 1,0
(Clause 3.7 NOTE 2). Because adding an absorber must reduce the reverberation
time, `T2 >= T1` (`alpha_s <= 0`) is non-physical and emits an
[`AbsorptionWarning`](/phonometry/reference/api/materials/sound-absorption/#absorptionwarning). A room volume below the 150 m3 minimum of
clause 6.1.1, or a sample area outside the clause 6.2.1.1 range
(`10 m2 <= S <= 12 m2`, upper limit scaled by `(V/200)^(2/3)` when
`V > 200 m3`), each emit an advisory [`AbsorptionWarning`](/phonometry/reference/api/materials/sound-absorption/#absorptionwarning).

**Parameters**

| Name | Description |
| :--- | :--- |
| `t1` | Empty-room reverberation time(s) `T1`, in seconds. |
| `t2` | With-specimen reverberation time(s) `T2`, in seconds. |
| `volume` | Room volume `V`, in cubic metres. |
| `sample_area` | Area `S` covered by the test specimen, in square metres (for both-sides-exposed absorbers, the area of the two sides; Clause 3.7 NOTE 1). |
| `temperature1` | Empty-room air temperature, in degrees Celsius (default 20). Used for `c1` via Eq. (6) unless `speed_of_sound1` is given. |
| `temperature2` | With-specimen air temperature, in degrees Celsius; defaults to `temperature1`. Used for `c2` unless `speed_of_sound2` is given. |
| `speed_of_sound1` | Explicit `c1` in m/s; overrides `temperature1`. |
| `speed_of_sound2` | Explicit `c2` in m/s; overrides `temperature2`. Defaults to `speed_of_sound1` when that is given but `c2` is not, so overriding only `c1` applies the same speed to both measurements. |
| `m1` | Empty-room air attenuation coefficient `m1`, in 1/m (default 0). |
| `m2` | With-specimen air attenuation coefficient `m2`, in 1/m (default 0). |

**Returns:** Sound absorption coefficient `alpha_s` with the broadcast shape of `t1` and `t2`.

## AbsorptionWarning

Advisory for out-of-range or non-physical ISO 354 absorption inputs.

## attenuation_from_alpha

```python
attenuation_from_alpha(alpha: ArrayLike) -> NDArray[np.float64]
```

Air power attenuation coefficient `m` from ISO 9613-1 `alpha`.

Applies the ISO 354:2003 (8.1.2.1) conversion `m = alpha / (10 * lg e)`,
where `alpha` is the attenuation coefficient in decibels per metre used by
ISO 9613-1 and `m` is the power attenuation coefficient in reciprocal
metres entering Eq. (5)/(7)/(8). ISO 354 itself provides no `alpha` table
or formula (it defers to ISO 9613-1); this helper only performs the unit
conversion for a caller who already holds `alpha` values.

**Parameters**

| Name | Description |
| :--- | :--- |
| `alpha` | Attenuation coefficient, in dB/m (scalar or per band). |

**Returns:** Power attenuation coefficient `m`, in 1/m.

## measure_sound_absorption

```python
measure_sound_absorption(
    frequencies: ArrayLike,
    t_empty: ArrayLike,
    t_specimen: ArrayLike,
    *,
    volume: float,
    area: float,
    temperature: float = 20.0,
    humidity: float | None = None,
    speed_of_sound: float | None = None,
    m: ArrayLike = 0.0,
) -> SoundAbsorptionMeasurement
```

Measure the sound absorption of a plane absorber (ISO 354:2003).

Assembles a [`SoundAbsorptionMeasurement`](/phonometry/reference/api/materials/sound-absorption/#soundabsorptionmeasurement) from the one-third-octave
reverberation times of the empty room (`T1`) and of the room with the
specimen installed (`T2`). The equivalent sound absorption areas `A1`
and `A2` follow from Sabine's equation (Eq. (5)/(7), delegated to
[`absorption_area`](/phonometry/reference/api/materials/sound-absorption/#absorption_area)) and the sound absorption coefficient
`alpha_s = (A2 - A1) / S` from Eq. (8)/(9) (delegated to
[`absorption_coefficient`](/phonometry/reference/api/materials/sound-absorption/#absorption_coefficient)); no formula is re-derived here.

Both measurements are taken at the same air temperature and, for the air
attenuation term, the same climatic conditions (ISO 354:2003, 6.3), so a
single `temperature` and `m` apply to both. Use the lower-level
[`absorption_coefficient`](/phonometry/reference/api/materials/sound-absorption/#absorption_coefficient) directly when the empty-room and
with-specimen climates differ.

**Parameters**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centre frequencies, in Hz (the ISO 354 range is 100 Hz to 5000 Hz); a 1-D array matching `t_empty` and `t_specimen`. |
| `t_empty` | Empty-room reverberation time `T1`, per band, in seconds. |
| `t_specimen` | With-specimen reverberation time `T2`, per band, in seconds. |
| `volume` | Reverberation-room volume `V`, in cubic metres. A volume below the 150 m3 minimum of clause 6.1.1 emits an advisory [`AbsorptionWarning`](/phonometry/reference/api/materials/sound-absorption/#absorptionwarning). |
| `area` | Area `S` covered by the test specimen, in square metres. An area outside the clause 6.2.1.1 range (10 m2 to 12 m2, upper limit scaled by `(V/200)^(2/3)` for `V > 200 m3`) emits an advisory [`AbsorptionWarning`](/phonometry/reference/api/materials/sound-absorption/#absorptionwarning). |
| `temperature` | Air temperature during the test, in degrees Celsius (default 20). Used for the speed of sound via Eq. (6) unless `speed_of_sound` is given; a temperature outside 15..30 degC emits an [`AbsorptionWarning`](/phonometry/reference/api/materials/sound-absorption/#absorptionwarning). |
| `humidity` | Relative humidity during the test, in % (informational; recorded on the result but not used in the computation, which sees the climate only through `m`). `None` leaves it unrecorded. |
| `speed_of_sound` | Explicit speed of sound `c`, in m/s; overrides `temperature` and Eq. (6) when supplied. |
| `m` | Power attenuation coefficient of air `m`, in 1/m (a scalar or a per-band array matching `frequencies`; default 0, i.e. no air correction). Obtain it from an ISO 9613-1 attenuation coefficient with [`attenuation_from_alpha`](/phonometry/reference/api/materials/sound-absorption/#attenuation_from_alpha). |

**Returns:** A frozen [`SoundAbsorptionMeasurement`](/phonometry/reference/api/materials/sound-absorption/#soundabsorptionmeasurement).

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If the frequency and reverberation-time arrays do not share one shape, or an input is non-physical (see [`absorption_coefficient`](/phonometry/reference/api/materials/sound-absorption/#absorption_coefficient)). |

## SoundAbsorptionMeasurement

```python
SoundAbsorptionMeasurement(
    frequencies: NDArray[np.float64],
    t_empty: NDArray[np.float64],
    t_specimen: NDArray[np.float64],
    volume: float,
    area: float,
    temperature: float,
    humidity: float | None,
    speed_of_sound: float,
    air_attenuation: NDArray[np.float64],
    absorption_area_empty: NDArray[np.float64],
    absorption_area_with_specimen: NDArray[np.float64],
    alpha_s: NDArray[np.float64],
)
```

A reverberation-room sound absorption measurement (ISO 354:2003).

The one-third-octave outcome of a plane-absorber test: the mean
reverberation time measured empty (`T1`) and with the specimen (`T2`),
the equivalent sound absorption areas `A1` (Eq. (5)) and `A2` (Eq. (7))
they give through Sabine's equation, and the sound absorption coefficient
`alpha_s` (Eq. (8)/(9)) of the specimen. Build it with
[`measure_sound_absorption`](/phonometry/reference/api/materials/sound-absorption/#measure_sound_absorption); the frozen instance then exposes
`plot` (`alpha_s` versus frequency) and `report` (an
accredited ISO 354 test-report PDF).

A single-number rating (the practical coefficient `alpha_p` and the
weighted coefficient `alpha_w`) is defined by ISO 11654, not ISO 354, and
is therefore not produced here; pass `alpha_s` to
[`weighted_absorption_from_third_octave`](/phonometry/reference/api/materials/absorption-rating/#weighted_absorption_from_third_octave) for it.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | One-third-octave band centre frequencies, in Hz (the ISO 354 range is 100 Hz to 5000 Hz). |
| `t_empty` | Mean reverberation time of the empty room `T1`, per band, in seconds. |
| `t_specimen` | Mean reverberation time of the room with the specimen `T2`, per band, in seconds. |
| `volume` | Reverberation-room volume `V`, in cubic metres. |
| `area` | Area `S` covered by the test specimen, in square metres. |
| `temperature` | Air temperature during the test, in degrees Celsius. |
| `humidity` | Relative humidity during the test, in %, or `None` when not recorded. It is informational: humidity enters ISO 354 only through the air attenuation coefficient `m` (via ISO 9613-1), never directly. |
| `speed_of_sound` | Propagation speed of sound `c` used in the Sabine inversion, in m/s (from Eq. (6) unless it was given explicitly). |
| `air_attenuation` | Power attenuation coefficient of air `m`, per band, in 1/m (`0` when no air correction was applied). |
| `absorption_area_empty` | Equivalent sound absorption area of the empty room `A1` (Eq. (5)), per band, in square metres. |
| `absorption_area_with_specimen` | Equivalent sound absorption area of the room containing the specimen `A2` (Eq. (7)), per band, in square metres. |
| `alpha_s` | Sound absorption coefficient `alpha_s` (Eq. (8)/(9)), per band. It may exceed 1,0 (Clause 3.7 NOTE 2) and is never clamped. |

### SoundAbsorptionMeasurement.equivalent_absorption_area

*property*

Equivalent sound absorption area of the specimen `AT = A2 - A1`.

The ISO 354:2003 Eq. (8) quantity, per band, in square metres; dividing
it by the specimen area `S` gives `alpha_s` (Eq. (9)).

### SoundAbsorptionMeasurement.plot()

```python
SoundAbsorptionMeasurement.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the sound absorption coefficient `alpha_s` versus frequency.

Draws `alpha_s` over the one-third-octave band axis (ISO 354). Values
above 1,0 are kept (Clause 3.7 NOTE 2), so the axis grows to show them.
Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes` and never calls `plt.show`.

**Parameters**

| Name | Description |
| :--- | :--- |
| `ax` | Existing axes, or `None` to create a figure. |
| `language` | `"en"` (default) or `"es"`. |
| `kwargs` | Forwarded to the `alpha_s` curve `plot` call. |

**Returns:** The axes.

### SoundAbsorptionMeasurement.report()

```python
SoundAbsorptionMeasurement.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 354 sound-absorption test-report fiche to a PDF.

Writes a one-page accredited reverberation-room report: the
standard-basis line, an optional metadata header block (client,
specimen, area `S`, room volume `V`, mounting, climate ...), a
two-panel body with the one-third-octave `alpha_s` table beside the
`alpha_s` curve, and a footer with the fixed disclaimer. ISO 354 is a
characterisation, so there is no pass/fail verdict and no single-number
rating (the weighted `alpha_w` is an ISO 11654 quantity, out of scope
here).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a body-and-disclaimer fiche without a metadata header. The `requirement` field is ignored (ISO 354 has no verdict). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | When `True`, the table adds the reverberation times `T1`/`T2` and the equivalent absorption areas `A1`/`A2`. |
| `language` | Fiche language: `"en"` (default, English, decimal point) or `"es"` (Spanish, decimal comma). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |
