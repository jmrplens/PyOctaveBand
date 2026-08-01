---
title: "room.enclosed_space_absorption"
description: "Sound absorption in enclosed spaces (EN 12354-6:2003)."
sidebar:
  label: "enclosed_space_absorption"
---

Sound absorption in enclosed spaces (EN 12354-6:2003).

Estimates the total equivalent sound absorption area of a room and the
resulting reverberation time from the absorption of its surfaces and objects
(the normative Clause 4 model).

The total equivalent absorption area sums the surface contributions, the
equivalent absorption areas of objects and object arrays and the air
absorption:

$$
A = \sum_i \alpha_{s,i} S_i + \sum_j A_{\text{obj},j} + \sum_k \alpha_{s,k} S_k + A_{\text{air}} \tag{Formula 1}
$$

with the air term $A_{\text{air}} = 4 m V (1 - \psi)$ (Formula 2),
the object fraction $\psi = \sum V_{\text{obj}} / V$ (Formula 3) and,
for hard irregular objects, the empirical equivalent area
$A_{\text{obj}} = V_{\text{obj}}^{2/3}$ (Formula 4). The reverberation
time follows from $T = \frac{55.3}{c_0} \frac{V (1 - \psi)}{A}$
(Formula 5); with the standard's $c_0 = 345.6$ m/s the factor
$55.3/c_0$ is the familiar `0.16`.

The informative Annex D method for irregular spaces / uneven absorption
distribution is out of scope.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## air_absorption_area

```python
air_absorption_area(
    m: ArrayLike,
    volume: float,
    object_fraction: float = 0.0,
) -> np.ndarray | float
```

Equivalent absorption area of the air (Formula 2).

**Parameters**

| Name | Description |
| :--- | :--- |
| `m` | Power attenuation coefficient of air, in Neper per metre (see `AIR_ATTENUATION`). |
| `volume` | Volume of the empty enclosed space `V`, m3. |
| `object_fraction` | Object fraction `psi` (0-1). |

**Returns:** The air absorption area $A_{\text{air}} = 4 m V (1 - \psi)$, m2.

## enclosed_space_reverberation

```python
enclosed_space_reverberation(
    surfaces: Sequence[tuple[float, ArrayLike]],
    volume: float,
    *,
    objects: ArrayLike = (),
    object_fraction: float = 0.0,
    air_condition: str | None = None,
    frequencies: ArrayLike = ...,
    speed_of_sound: float = 345.6,
) -> ReverberationResult
```

Predict the absorption area and reverberation time per octave band.

Chains the total equivalent absorption area (Formula 1, with the air term
of Formula 2 from `AIR_ATTENUATION` when `air_condition` is given)
and the reverberation time (Formula 5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `surfaces` | Sequence of `(area, absorption_coefficient)` pairs; each coefficient a per-band array aligned with `frequencies`. |
| `volume` | Volume of the empty enclosed space `V`, m3. |
| `objects` | Equivalent absorption areas of the objects `Aobj`, m2. |
| `object_fraction` | Object fraction `psi` (0-1). |
| `air_condition` | A key of `AIR_ATTENUATION` (e.g. `"20C_50-70"`) to include air absorption, or `None` to neglect it. |
| `frequencies` | Octave-band centre frequencies, in hertz. The built-in `air_condition` profiles require the standard [`OCTAVE_BANDS`](/phonometry/reference/api/materials/absorption-rating/#octave_bands). |
| `speed_of_sound` | Speed of sound `c0`, m/s. |

**Returns:** The [`ReverberationResult`](/phonometry/reference/api/rooms/enclosed-space-absorption/#reverberationresult).

## equivalent_absorption_area

```python
equivalent_absorption_area(
    surfaces: Sequence[tuple[float, ArrayLike]],
    *,
    objects: ArrayLike = (),
    air_area: ArrayLike = 0.0,
) -> np.ndarray | float
```

Total equivalent sound absorption area (Formula 1).

**Parameters**

| Name | Description |
| :--- | :--- |
| `surfaces` | Sequence of `(area, absorption_coefficient)` pairs, one per surface (or object array treated as a surface). The absorption coefficient may be a scalar or a per-band array. |
| `objects` | Equivalent absorption areas of the objects `Aobj`, m2 (see [`hard_object_absorption`](/phonometry/reference/api/rooms/enclosed-space-absorption/#hard_object_absorption)): a single value, a 1-D sequence of one value per object, or a 2-D array `(n_objects, n_bands)` for per-band values; summed over the objects. |
| `air_area` | Air absorption area `Aair`, m2 (see [`air_absorption_area`](/phonometry/reference/api/rooms/enclosed-space-absorption/#air_absorption_area)); scalar or per-band. |

**Returns:** The total equivalent absorption area `A`, m2; a float for all-scalar inputs, otherwise a per-band array.

## hard_object_absorption

```python
hard_object_absorption(object_volume: ArrayLike) -> np.ndarray
```

Equivalent absorption area of a hard object (Formula 4).

An empirical estimate for hard, irregularly shaped objects (machinery,
furniture) whose equivalent area is not otherwise available.

**Parameters**

| Name | Description |
| :--- | :--- |
| `object_volume` | Volume `Vobj` of the hard object(s), m3. |

**Returns:** The equivalent absorption area $A_{\text{obj}} = V_{\text{obj}}^{2/3}$, m2.

## object_fraction

```python
object_fraction(object_volumes: ArrayLike, volume: float) -> float
```

Object fraction `psi` of an enclosed space (Formula 3).

**Parameters**

| Name | Description |
| :--- | :--- |
| `object_volumes` | Volumes of the objects and object arrays, m3. |
| `volume` | Volume of the empty enclosed space `V`, m3. |

**Returns:** The object fraction $\psi = \sum V_{\text{obj}} / V$.

## reverberation_time

```python
reverberation_time(
    absorption_area: ArrayLike,
    volume: float,
    *,
    object_fraction: float = 0.0,
    speed_of_sound: float = 345.6,
) -> np.ndarray | float
```

Reverberation time from the equivalent absorption area (Formula 5).

**Parameters**

| Name | Description |
| :--- | :--- |
| `absorption_area` | Total equivalent absorption area `A`, m2. |
| `volume` | Volume of the empty enclosed space `V`, m3. |
| `object_fraction` | Object fraction `psi` (0-1). |
| `speed_of_sound` | Speed of sound `c0`, m/s (default `SPEED_OF_SOUND`, giving the factor `0.16`). |

**Returns:** The reverberation time $T = \frac{55.3}{c_0} \frac{V (1 - \psi)}{A}$, s.

## ReverberationResult

```python
ReverberationResult(
    frequencies: np.ndarray,
    absorption_area: np.ndarray,
    reverberation_time: np.ndarray,
    volume: float,
    object_fraction: float,
)
```

Absorption area and reverberation time of an enclosed space (Clause 4).

**Attributes**

| Name | Description |
| :--- | :--- |
| `frequencies` | Octave-band centre frequencies, in hertz. |
| `absorption_area` | Total equivalent absorption area `A` per band, m2. |
| `reverberation_time` | Reverberation time `T` per band, s. |
| `volume` | Volume of the empty enclosed space, m3. |
| `object_fraction` | Object fraction `psi` (0-1). |

### ReverberationResult.plot()

```python
ReverberationResult.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the reverberation time over the octave bands.

Requires matplotlib (`pip install phonometry[plot]`); returns the
`Axes`.

### ReverberationResult.report()

```python
ReverberationResult.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an enclosed-space absorption/reverberation fiche to a PDF.

Writes a one-page report characterising an enclosed space by the
EN 12354-6:2003 Clause 4 model: a standard-basis line, an optional
metadata header block (client, room, description, room volume, object
fraction, climate ...), a per-band table of the equivalent sound
absorption area `A` and the reverberation time `T` beside the
result's own reverberation-time plot (`plot`), the boxed
mid-frequency reverberation time with the mid-frequency absorption area
alongside, and a footer with the fixed disclaimer. EN 12354-6 gives a
diffuse-field estimate rather than a measurement; a supplied
`metadata.requirement` is printed as a target reverberation-time
reference line without a PASS/FAIL verdict, since a room reverberation
time is a target range rather than a strictly higher/lower-is-better
quantity.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata); `None` produces a bare characterisation fiche (body, result and disclaimer only). |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for parity with the other fiches; the band table already shows both A and T, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"`. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`), or matplotlib is missing for the embedded figure (`pip install phonometry[plot]`). |
