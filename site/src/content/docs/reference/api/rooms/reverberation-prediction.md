---
title: "room.reverberation_prediction"
description: "Reverberation-time prediction from room geometry and absorption."
sidebar:
  label: "reverberation_prediction"
---

Reverberation-time prediction from room geometry and absorption.

Predicts the reverberation time `T` of an enclosed space from its volume,
boundary areas and the sound-absorption coefficients of its surfaces, through
the classical statistical-acoustics formulae. Given a diffuse field and an
exponential energy decay, `T` is the time for the level to fall by 60 dB.

Five models are provided, in order of increasing account of a non-uniform
absorption distribution:

* **Sabine** -- the original diffuse-field estimate,
  $T = k V / (A + 4 m V)$
  with the total equivalent absorption area
  $A = \sum_i S_i \alpha_i$ and the
  air term $4 m V$. Exact only for low, uniform absorption.
* **Eyring** (Norris-Eyring) -- replaces `A` by
  $-S \ln(1 - \bar{\alpha})$ with
  the mean absorption $\bar{\alpha} = A / S$ over the total surface
  `S`; correct in the strong-absorption limit where Sabine overestimates
  `T`.
* **Millington-Sette** -- $-\sum_i S_i \ln(1 - \alpha_i)$ sums the
  Eyring term
  per surface, so a single perfectly absorbing surface drives `T` to zero.
* **Fitzroy** -- an *area-weighted arithmetic* mean of three axial Eyring
  reverberation times, one per pair of opposing walls; captures rooms with the
  absorption concentrated on one axis (e.g. a carpeted, otherwise hard room).
* **Arau-Puchades** -- an *area-weighted geometric* mean of the same three
  axial Eyring times (Arau-Puchades, *Acustica* 65 (1988) 163):
  $T = \prod_i T_i^{S_i / S}$. Recommended by its author over Fitzroy
  for anisotropic rooms.

The Sabine constant is $k = 24 \ln 10 / c_0$
($= 55.26 / c_0$); with the
default $c_0 = 343$ m/s it takes the familiar textbook value
`0.161`. (The
[`enclosed_space_absorption`](/phonometry/reference/api/rooms/enclosed-space-absorption/) EN 12354-6 model instead rounds
`k` to `55.3` and uses $c_0 = 345.6$ to pin the factor at exactly
`0.16`.)

Air absorption enters every model through the `air_attenuation` power
coefficient `m` (in neper per metre) as the additive term $4 m V$;
obtain a physical `m` from temperature and humidity with
[`phonometry.air_absorption.air_attenuation_m`](/phonometry/reference/api/environment/air-absorption/#air_attenuation_m).

Each model enforces its own mathematical domain on the absorption
coefficients. Sabine's linear sum is finite for any non-negative coefficient,
so it accepts measured ISO 354 values at or above 1 (up to a unit-error guard
at 2). The logarithmic models are stricter exactly where the maths requires
it: Millington-Sette needs *every* coefficient below 1, while Eyring, Fitzroy
and Arau-Puchades need each *mean* entering $\ln(1 - \alpha)$ below 1.

The Fitzroy and Arau-Puchades models require a rectangular (shoebox) room and
take the room `dimensions` together with the mean absorption of each of the
three wall pairs. All five reduce to Eyring for a uniform absorption
distribution, and Eyring reduces to Sabine as the absorption tends to zero --
the identities the conformance suite anchors on, absent a machine-readable
worked example in the source texts.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## arau_puchades_reverberation_time

```python
arau_puchades_reverberation_time(
    dimensions: Sequence[float],
    absorptions: Sequence[ArrayLike],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Arau-Puchades reverberation time -- area-weighted geometric mean of axial times.

$T = \prod_i T_i^{S_i / S}$ with $T_i$ the Eyring time of the
wall pair perpendicular to axis `i` (Arau-Puchades, *Acustica* 65
(1988) 163, Formula 18). Preferred by its author over Fitzroy for rooms
with an anisotropic absorption distribution. Reduces to Eyring for a
uniform distribution. Each input is itself a mean entering
$\ln(1 - \alpha_i)$, so each must be below 1.

**Parameters**

| Name | Description |
| :--- | :--- |
| `dimensions` | Room lengths `(Lx, Ly, Lz)`, m. |
| `absorptions` | Mean absorption `(alpha_x, alpha_y, alpha_z)` of the three wall pairs (perpendicular to x, y, z); each a scalar or per-band array. |
| `air_attenuation` | Air power-attenuation coefficient `m` (1/m). |
| `speed_of_sound` | Speed of sound `c0`, m/s. |

**Returns:** The reverberation time `T`, s.

## eyring_reverberation_time

```python
eyring_reverberation_time(
    volume: float,
    surfaces: Sequence[Surface],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Eyring (Norris-Eyring) reverberation time.

$$
T = \frac{k V}{-S \ln(1 - \bar{\alpha}) + 4 m V}
$$

with the total surface `S`
and its area-weighted mean absorption `alpha_bar`.

The formula constrains only the *mean*: $\ln(1 - \bar{\alpha})$
requires $\bar{\alpha} < 1$, while individual coefficients at or
above 1 (a measured
ISO 354 outcome) are accepted as long as the mean stays below 1 and each
coefficient stays within the shared unit-error ceiling of 2.

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Room volume `V`, m3. |
| `surfaces` | Sequence of `(area, absorption_coefficient)` pairs. |
| `air_attenuation` | Air power-attenuation coefficient `m` (1/m). |
| `speed_of_sound` | Speed of sound `c0`, m/s. |

**Returns:** The reverberation time `T`, s.

## fitzroy_reverberation_time

```python
fitzroy_reverberation_time(
    dimensions: Sequence[float],
    absorptions: Sequence[ArrayLike],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Fitzroy reverberation time -- area-weighted arithmetic mean of axial times.

$T = \sum_i (S_i / S)\, T_i$ with $T_i$ the Eyring time of
the wall pair
perpendicular to axis `i` (Fitzroy, *J. Acoust. Soc. Am.* 31 (1959) 893).
Equivalent to
$T = \frac{k V}{S^2} \sum_i \frac{S_i}{-\ln(1 - \alpha_i)}$
without air. Reduces to Eyring for a uniform absorption distribution.
Each input is itself a mean entering $\ln(1 - \alpha_i)$, so each
must be below 1.

**Parameters**

| Name | Description |
| :--- | :--- |
| `dimensions` | Room lengths `(Lx, Ly, Lz)`, m. |
| `absorptions` | Mean absorption `(alpha_x, alpha_y, alpha_z)` of the three wall pairs (perpendicular to x, y, z); each a scalar or per-band array. |
| `air_attenuation` | Air power-attenuation coefficient `m` (1/m). |
| `speed_of_sound` | Speed of sound `c0`, m/s. |

**Returns:** The reverberation time `T`, s.

## mean_absorption

```python
mean_absorption(surfaces: Sequence[Surface]) -> np.ndarray | float
```

Area-weighted mean absorption coefficient
$\bar{\alpha} = A / S$.

**Parameters**

| Name | Description |
| :--- | :--- |
| `surfaces` | Sequence of `(area, absorption_coefficient)` pairs; each coefficient a scalar or a per-band array. |

**Returns:** The mean absorption $\sum_i S_i \alpha_i / \sum_i S_i$; a float for scalar coefficients, otherwise a per-band array.

## millington_sette_reverberation_time

```python
millington_sette_reverberation_time(
    volume: float,
    surfaces: Sequence[Surface],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Millington-Sette reverberation time.

$$
T = \frac{k V}{-\sum_i S_i \ln(1 - \alpha_i) + 4 m V}
$$

The Eyring absorption
term is summed surface by surface rather than through a single mean.
A surface
approaching total absorption ($\alpha_i \to 1$) drives `T` to
zero.
Because the logarithm applies per surface, *every* coefficient must be
strictly below 1; measured ISO 354 coefficients at or above 1 are outside
this model's domain (use Sabine, or Eyring while the mean stays below 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Room volume `V`, m3. |
| `surfaces` | Sequence of `(area, absorption_coefficient)` pairs. |
| `air_attenuation` | Air power-attenuation coefficient `m` (1/m). |
| `speed_of_sound` | Speed of sound `c0`, m/s. |

**Returns:** The reverberation time `T`, s.

## reverberation_time_models

```python
reverberation_time_models(
    dimensions: Sequence[float],
    absorptions: Sequence[ArrayLike],
    *,
    air_attenuation: ArrayLike = 0.0,
    frequencies: ArrayLike | None = None,
    speed_of_sound: float = 343.0,
) -> ReverberationModelResult
```

Predict the reverberation time of a rectangular room by all five models.

A convenience front-end that builds the six boundary surfaces of the room
from `dimensions` and the three wall-pair mean absorptions, then evaluates
[`sabine_reverberation_time`](/phonometry/reference/api/rooms/reverberation-prediction/#sabine_reverberation_time), [`eyring_reverberation_time`](/phonometry/reference/api/rooms/reverberation-prediction/#eyring_reverberation_time),
[`millington_sette_reverberation_time`](/phonometry/reference/api/rooms/reverberation-prediction/#millington_sette_reverberation_time), [`fitzroy_reverberation_time`](/phonometry/reference/api/rooms/reverberation-prediction/#fitzroy_reverberation_time)
and [`arau_puchades_reverberation_time`](/phonometry/reference/api/rooms/reverberation-prediction/#arau_puchades_reverberation_time) on a common footing. Because
the bundle evaluates the logarithmic models too, the inputs must satisfy
the strictest of the five domains: every absorption below 1.

**Parameters**

| Name | Description |
| :--- | :--- |
| `dimensions` | Room lengths `(Lx, Ly, Lz)`, m. |
| `absorptions` | Mean absorption `(alpha_x, alpha_y, alpha_z)` of the three wall pairs (perpendicular to x, y, z); each a scalar or a per-band array aligned with `frequencies`. |
| `air_attenuation` | Air power-attenuation coefficient `m` (1/m), scalar or per-band. |
| `frequencies` | Band centre frequencies, in hertz, used only to label the result and its plot; defaults to an integer index over the bands. |
| `speed_of_sound` | Speed of sound `c0`, m/s. |

**Returns:** The [`ReverberationModelResult`](/phonometry/reference/api/rooms/reverberation-prediction/#reverberationmodelresult).

## ReverberationModelResult

```python
ReverberationModelResult(
    frequencies: np.ndarray,
    sabine: np.ndarray,
    eyring: np.ndarray,
    millington_sette: np.ndarray,
    fitzroy: np.ndarray,
    arau_puchades: np.ndarray,
    volume: float,
    surface_area: float,
)
```

Predicted reverberation time of a rectangular room by five models.

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Band centre frequencies, in hertz. |
| `sabine` | Sabine reverberation time per band, s. |
| `eyring` | Eyring (Norris-Eyring) reverberation time per band, s. |
| `millington_sette` | Millington-Sette reverberation time per band, s. |
| `fitzroy` | Fitzroy reverberation time per band, s. |
| `arau_puchades` | Arau-Puchades reverberation time per band, s. |
| `volume` | Room volume `V`, m3. |
| `surface_area` | Total boundary area `S`, m2. |

### ReverberationModelResult.models

*property*

The five reverberation-time curves keyed by model name.

### ReverberationModelResult.plot()

```python
ReverberationModelResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the reverberation-time curves of the five models.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ReverberationModelResult.report()

```python
ReverberationModelResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render a reverberation-time prediction fiche to a PDF.

Writes a one-page report of the reverberation time predicted by the
five statistical-acoustics models (Sabine, Eyring, Millington-Sette,
Fitzroy and Arau-Puchades): a standard-basis line marking it a
design-stage prediction, an optional metadata header block (client,
room, description, room volume, total surface area, climate ...), a
per-band table with one reverberation-time column per model beside the
result's own model-comparison plot (`plot`), the boxed
mid-frequency reverberation time from Arau-Puchades (the recommended
model for a non-uniform absorption distribution) with the per-model
spread alongside, and a footer with the fixed disclaimer. This is a
prediction, not a measurement: the five models bracket the
reverberation time likely to occur. A supplied
`metadata.requirement` is printed as a target reverberation-time
reference line without a PASS/FAIL verdict, since a room reverberation
time is a target range rather than a strictly higher/lower-is-better
quantity.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare prediction fiche (body, result and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for parity with the other fiches; the model table already shows every computed value, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |

## sabine_reverberation_time

```python
sabine_reverberation_time(
    volume: float,
    surfaces: Sequence[Surface],
    *,
    air_attenuation: ArrayLike = 0.0,
    speed_of_sound: float = 343.0,
) -> np.ndarray | float
```

Sabine reverberation time $T = k V / (A + 4 m V)$.

$A = \sum_i S_i \alpha_i$ is finite for any non-negative
coefficient, so
unlike the logarithmic models Sabine accepts coefficients at or above 1:
measured ISO 354 reverberation-room values of 1.05 to 1.20 (the edge
effect) and the exact 1.0 that the ISO 11654 practical rating caps at are
legitimate inputs. Coefficients above 2 are rejected as a probable unit
error (a percentage passed instead of a fraction).

**Parameters**

| Name | Description |
| :--- | :--- |
| `volume` | Room volume `V`, m3. |
| `surfaces` | Sequence of `(area, absorption_coefficient)` pairs; each coefficient a scalar or a per-band array in `[0, 2]`. |
| `air_attenuation` | Air power-attenuation coefficient `m`, in neper per metre (scalar or per-band); see [`phonometry.air_absorption.air_attenuation_m`](/phonometry/reference/api/environment/air-absorption/#air_attenuation_m). Default `0` (air absorption neglected). |
| `speed_of_sound` | Speed of sound `c0`, m/s (default [`DEFAULT_SPEED_OF_SOUND`](/phonometry/reference/api/materials/road-absorption/#default_speed_of_sound), giving the factor `0.161`). |

**Returns:** The reverberation time `T`, s; a float for scalar inputs, otherwise a per-band array.
