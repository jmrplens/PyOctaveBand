---
title: "aircraft.anp_fleet"
description: "EASA ANP fleet database bridge for the ECAC Doc 29 airport-noise chain."
sidebar:
  label: "anp_fleet"
---

EASA ANP fleet database bridge for the ECAC Doc 29 airport-noise chain.

The ECAC Doc 29 method in [`phonometry.aircraft.airport_noise`](/phonometry/reference/api/aeroacoustics/airport-noise/) places an
aircraft's noise at a receiver from a Noise-Power-Distance (NPD) table and a
flight profile. Both come, for real aircraft types, from the **Aircraft Noise
and Performance (ANP)** database maintained by EUROCONTROL/EASA: per aircraft it
tabulates NPD curves (`LAmax` and `SEL` versus slant distance for a set of
engine power settings, per operation mode) and default trajectories.

This module reads the ANP database tables (the semicolon-delimited CSV exports)
and exposes, for a given aircraft identifier and operation:

* [`AnpNpdCurves`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpnpdcurves) -- the NPD curves (`LAmax`/`SEL` versus distance for
  each tabulated power), with a `.plot()`;
* [`AnpProfile`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpprofile) -- the default fixed-point trajectory as a Doc 29 flight
  path `(N, 5)` with the takeoff/landing ground-roll masks, with a `.plot()`;
* [`AnpAircraft`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpaircraft) -- the aircraft metadata plus convenience wiring
  ([`AnpAircraft.event_level`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpaircraftevent_level), [`AnpAircraft.noise_contour`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpaircraftnoise_contour)) that feeds
  the NPD curves and the profile straight into the existing Doc 29 functions.

[`load_anp_database`](/phonometry/reference/api/aeroacoustics/anp-fleet/#load_anp_database) returns an [`AnpDatabase`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpdatabase). Called without a path
it loads the full EASA ANP database (archive version 2.3) shipped with the
package (see `aircraft/data/anp/PROVENANCE.md`); pointed at a directory it
reads any other ANP CSV export the user provides.

Only the fixed-point trajectories are read as ready-to-use profiles; procedural
step profiles (which require the ICAO Doc 9911 / Doc 29 Vol 2 flight-mechanics
performance model) are outside this bridge, and NPD curves are available for
every aircraft regardless.

Source (clean-room, implemented from the published table format): EASA ANP
database v2.3 (2020) and the ECAC Doc 29 4th ed. Vol 2 NPD/profile conventions.

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## AnpAircraft

```python
AnpAircraft(
    aircraft_id: str,
    description: str,
    engine_type: str,
    num_engines: int,
    weight_class: str,
    mounting: str,
    npd_id: str,
    power_parameter: str,
    _database: AnpDatabase,
)
```

One ANP aircraft type: metadata plus NPD/profile access and Doc 29 wiring.

**Attributes**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier (e.g. `"747100"`). |
| `description` | Human-readable aircraft/engine description. |
| `engine_type` | `"Jet"`, `"Turboprop"` or `"Piston"`. |
| `num_engines` | Number of engines. |
| `weight_class` | ICAO wake weight class. |
| `mounting` | Doc 29 engine mounting (`"wing"`/`"fuselage"`/`"propeller"`). |
| `npd_id` | ANP noise identifier. |
| `power_parameter` | Name/unit of the NPD power parameter. |

### AnpAircraft.event_level()

```python
AnpAircraft.event_level(
    observer: NDArray[np.float64] | list[float],
    operation: str,
    *,
    stage_length: int = 1,
    metric: str = 'exposure',
    temperature: float = 15.0,
    pressure: float = 101.325,
) -> FlyoverResult
```

Single-event level at a receiver (see [`AnpDatabase.event_level`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpdatabaseevent_level)).

### AnpAircraft.noise_contour()

```python
AnpAircraft.noise_contour(
    operation: str,
    *,
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    stage_length: int = 1,
    metric: str = 'exposure',
    temperature: float = 15.0,
    pressure: float = 101.325,
) -> NoiseContourResult
```

Single-event ground contour (see [`AnpDatabase.noise_contour`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpdatabasenoise_contour)).

### AnpAircraft.npd_curves()

```python
AnpAircraft.npd_curves(operation: str, metric: str = 'SEL') -> AnpNpdCurves
```

NPD curves for this aircraft (see [`AnpDatabase.npd_curves`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpdatabasenpd_curves)).

### AnpAircraft.profile()

```python
AnpAircraft.profile(
    operation: str,
    stage_length: int = 1,
    *,
    profile_id: str | None = None,
) -> AnpProfile
```

Fixed-point profile (see [`AnpDatabase.profile`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpdatabaseprofile)).

## AnpDatabase

```python
AnpDatabase(
    aircraft: Mapping[str, dict[str, str]],
    npd: Mapping[tuple[str, str, str], tuple[NDArray[np.float64], NDArray[np.float64]]],
    distances: NDArray[np.float64],
    profiles: Mapping[tuple[str, str, str, int], NDArray[np.float64]],
)
```

A parsed ANP database (aircraft metadata, NPD curves and default profiles).

Build one with [`load_anp_database`](/phonometry/reference/api/aeroacoustics/anp-fleet/#load_anp_database). NPD curves are available for every
aircraft; default profiles are available for aircraft that have a fixed-point
trajectory in the database.

### AnpDatabase.aircraft()

```python
AnpDatabase.aircraft(aircraft_id: str) -> AnpAircraft
```

Return the [`AnpAircraft`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpaircraft) for an identifier.

**Raises**

| Exception | When |
| :--- | :--- |
| KeyError | If the identifier is not in the database. |

### AnpDatabase.aircraft_ids

*property*

Sorted list of aircraft identifiers in the database.

### AnpDatabase.event_level()

```python
AnpDatabase.event_level(
    aircraft_id: str,
    observer: NDArray[np.float64] | list[float],
    operation: str,
    *,
    stage_length: int = 1,
    metric: str = 'exposure',
    temperature: float = 15.0,
    pressure: float = 101.325,
) -> FlyoverResult
```

Doc 29 single-event level of an ANP aircraft at a receiver.

Feeds the aircraft's default fixed-point profile and NPD curves into
[`phonometry.aircraft.airport_noise.event_level`](/phonometry/reference/api/aeroacoustics/airport-noise/#event_level).

**Parameters**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier. |
| `observer` | Receiver position `(x, y, z)`, in metres. |
| `operation` | `"departure"`/`"D"` or `"arrival"`/`"A"`. |
| `stage_length` | ANP stage length (default 1). |
| `metric` | `"exposure"` (SEL) or `"maximum"` (LAmax). |
| `temperature` | Aerodrome air temperature, in °C. |
| `pressure` | Aerodrome air pressure, in kPa. |

**Returns:** A [`FlyoverResult`](/phonometry/reference/api/aeroacoustics/airport-noise/#flyoverresult).

### AnpDatabase.noise_contour()

```python
AnpDatabase.noise_contour(
    aircraft_id: str,
    operation: str,
    *,
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    stage_length: int = 1,
    metric: str = 'exposure',
    temperature: float = 15.0,
    pressure: float = 101.325,
) -> NoiseContourResult
```

Doc 29 single-event ground contour of an ANP aircraft.

Feeds the aircraft's default fixed-point profile and NPD curves into
[`phonometry.aircraft.airport_noise.noise_contour`](/phonometry/reference/api/aeroacoustics/airport-noise/#noise_contour).

**Parameters**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier. |
| `operation` | `"departure"`/`"D"` or `"arrival"`/`"A"`. |
| `x` | Grid x coordinates (along-track), in metres. |
| `y` | Grid y coordinates (lateral), in metres. |
| `stage_length` | ANP stage length (default 1). |
| `metric` | `"exposure"` (SEL) or `"maximum"` (LAmax). |
| `temperature` | Aerodrome air temperature, in °C. |
| `pressure` | Aerodrome air pressure, in kPa. |

**Returns:** A [`NoiseContourResult`](/phonometry/reference/api/aeroacoustics/airport-noise/#noisecontourresult).

### AnpDatabase.npd_curves()

```python
AnpDatabase.npd_curves(
    aircraft_id: str,
    operation: str,
    metric: str = 'SEL',
) -> AnpNpdCurves
```

NPD curves for an aircraft, operation and noise metric.

**Parameters**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier. |
| `operation` | `"departure"`/`"D"` or `"arrival"`/`"A"`. |
| `metric` | `"SEL"` (default) or `"LAmax"`. |

**Returns:** An [`AnpNpdCurves`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpnpdcurves).

**Raises**

| Exception | When |
| :--- | :--- |
| KeyError | If the aircraft has no NPD data for the request. |
| ValueError | If the metric or operation is unknown. |

### AnpDatabase.profile()

```python
AnpDatabase.profile(
    aircraft_id: str,
    operation: str,
    stage_length: int = 1,
    *,
    profile_id: str | None = None,
) -> AnpProfile
```

Fixed-point trajectory for an aircraft, operation and stage length.

Aircraft may ship several fixed-point profiles for the same operation
and stage length (e.g. weight variants). With `profile_id=None` the
`"DEFAULT"` profile is selected when present; otherwise the single
available profile is used, and an ambiguous request (several profiles,
none named `"DEFAULT"`) raises listing the identifiers.

**Parameters**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier. |
| `operation` | `"departure"`/`"D"` or `"arrival"`/`"A"`. |
| `stage_length` | ANP stage length (default 1). |
| `profile_id` | Optional ANP profile identifier (e.g. `"DEFAULT"`, `"3000LB"`); `None` (default) selects as described above. |

**Returns:** An [`AnpProfile`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpprofile) (a Doc 29 flight path with ground-roll masks).

**Raises**

| Exception | When |
| :--- | :--- |
| KeyError | If the aircraft is unknown, has no fixed-point profile for the request, or `profile_id` is not among the available ones. |
| ValueError | If `profile_id` is `None` and several profiles exist with none of them named `"DEFAULT"`. |

## AnpNpdCurves

```python
AnpNpdCurves(
    aircraft_id: str,
    npd_id: str,
    metric: str,
    operation: str,
    power_parameter: str,
    powers: NDArray[np.float64],
    distances: NDArray[np.float64],
    levels: NDArray[np.float64],
)
```

ANP Noise-Power-Distance curves for one aircraft, metric and operation.

**Attributes**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier. |
| `npd_id` | ANP noise identifier (shared by aircraft with the same NPD set). |
| `metric` | `"SEL"` or `"LAmax"`. |
| `operation` | `"A"` (arrival) or `"D"` (departure). |
| `power_parameter` | Name/unit of the power setting (e.g. corrected net thrust). |
| `powers` | Tabulated engine power settings (1-D, strictly increasing). |
| `distances` | Tabulated slant distances, in metres (1-D, strictly increasing). |
| `levels` | Tabulated event levels, shape `(len(powers), len(distances))`, in dB. |

The `powers`, `distances` and `levels` arrays are read-only views shared
with the parent database; copy them before mutating.

### AnpNpdCurves.level()

```python
AnpNpdCurves.level(
    power: float,
    distance: NDArray[np.float64] | list[float] | float,
) -> NDArray[np.float64]
```

Interpolated NPD level `L(P, d)` (Doc 29 Eq. 4-3/4-4).

**Parameters**

| Name | Description |
| :--- | :--- |
| `power` | Query engine power setting. |
| `distance` | Query slant distance(s), in metres. |

**Returns:** The interpolated level per query distance, in dB.

### AnpNpdCurves.plot()

```python
AnpNpdCurves.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the NPD curve at each tabulated power versus slant distance.

## AnpProfile

```python
AnpProfile(
    aircraft_id: str,
    operation: str,
    profile_id: str,
    stage_length: int,
    path: NDArray[np.float64],
    ground_roll: NDArray[np.bool_],
    landing_roll: NDArray[np.bool_],
)
```

Default fixed-point trajectory of an ANP aircraft as a Doc 29 flight path.

**Attributes**

| Name | Description |
| :--- | :--- |
| `aircraft_id` | ANP aircraft identifier. |
| `operation` | `"A"` (arrival) or `"D"` (departure). |
| `profile_id` | ANP profile label (usually `"DEFAULT"`). |
| `stage_length` | ANP stage length (trip-distance/weight bin). |
| `path` | Flight-path points, shape `(N, 5)`: `x, y, z` (m, along-track, lateral, altitude), engine power setting and true airspeed (m/s). |
| `ground_roll` | Boolean mask (length `N-1`) of takeoff ground-roll segments. |
| `landing_roll` | Boolean mask (length `N-1`) of landing rollout segments. |

`path` is a read-only view shared with the parent database; copy it before
mutating.

### AnpProfile.plot()

```python
AnpProfile.plot(
    ax: Axes | None = None,
    *,
    language: str = 'en',
    **kwargs: Any,
) -> Axes
```

Plot the trajectory altitude versus along-track distance.

## load_anp_database

```python
load_anp_database(path: Path | str | None = None) -> AnpDatabase
```

Load an EASA ANP database (aircraft, NPD curves and default profiles).

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Directory of an ANP CSV export (the `*Aircraft.csv`, `*NPD_data.csv`, `*fixed_point_profiles.csv` tables). If `None` (default), loads the full EASA ANP database v2.3 shipped with the package (see `aircraft/data/anp/PROVENANCE.md`). |

**Returns:** An [`AnpDatabase`](/phonometry/reference/api/aeroacoustics/anp-fleet/#anpdatabase).

**Raises**

| Exception | When |
| :--- | :--- |
| FileNotFoundError | If a required table is missing. |
