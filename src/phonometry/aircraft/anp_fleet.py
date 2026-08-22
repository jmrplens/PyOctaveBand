#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EASA ANP fleet database bridge for the ECAC Doc 29 airport-noise chain.

The ECAC Doc 29 method in :mod:`phonometry.aircraft.airport_noise` places an
aircraft's noise at a receiver from a Noise-Power-Distance (NPD) table and a
flight profile. Both come, for real aircraft types, from the **Aircraft Noise
and Performance (ANP)** database maintained by EUROCONTROL/EASA: per aircraft it
tabulates NPD curves (``LAmax`` and ``SEL`` versus slant distance for a set of
engine power settings, per operation mode) and default trajectories.

This module reads the ANP database tables (the semicolon-delimited CSV exports)
and exposes, for a given aircraft identifier and operation:

* :class:`AnpNpdCurves` -- the NPD curves (``LAmax``/``SEL`` versus distance for
  each tabulated power), with a ``.plot()``;
* :class:`AnpProfile` -- the default fixed-point trajectory as a Doc 29 flight
  path ``(N, 5)`` with the takeoff/landing ground-roll masks, with a ``.plot()``;
* :class:`AnpAircraft` -- the aircraft metadata plus convenience wiring
  (:meth:`AnpAircraft.event_level`, :meth:`AnpAircraft.noise_contour`) that feeds
  the NPD curves and the profile straight into the existing Doc 29 functions.

:func:`load_anp_database` returns an :class:`AnpDatabase`. Called without a path
it loads the full EASA ANP database (archive version 2.3) shipped with the
package (see ``aircraft/data/anp/PROVENANCE.md``); pointed at a directory it
reads any other ANP CSV export the user provides.

Only the fixed-point trajectories are read as ready-to-use profiles; procedural
step profiles (which require the ICAO Doc 9911 / Doc 29 Vol 2 flight-mechanics
performance model) are outside this bridge, and NPD curves are available for
every aircraft regardless.

Source (clean-room, implemented from the published table format): EASA ANP
database v2.3 (2020) and the ECAC Doc 29 4th ed. Vol 2 NPD/profile conventions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_axis_count,
    require_equal_counts,
    require_ranks,
    require_same_length,
)
from .airport_noise import (
    AerodromeAtmosphere,
    FlightSegmentState,
    FlyoverResult,
    NoiseContourResult,
    event_level,
    noise_contour,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Feet-to-metres conversion (NPD slant distances and profile altitudes/distances).
_FT_M = 0.3048
#: Knots-to-metres-per-second conversion (profile true airspeed).
_KT_MS = 0.514444
#: Altitude below which a profile point counts as on the ground, in metres.
_GROUND_ALTITUDE_M = 1.0
#: Minimum number of ``L_<ft>ft`` slant-distance columns an NPD table must
#: provide: the NPD level lookup (Doc 29 Eq. 4-3/4-4) interpolates over the
#: distance grid, which needs at least two tabulated distances.
_MIN_NPD_DISTANCES = 2
#: Lateral-directivity identifier -> Doc 29 engine mounting.
_MOUNTING = {"wing": "wing", "fuselage": "fuselage", "prop": "propeller"}
#: ANP operation code for departure (on-ground segments are a ground roll).
_DEPARTURE = "D"
#: ANP operation code for arrival (on-ground segments are a landing roll).
_ARRIVAL = "A"
#: Operation aliases -> ANP operation code (``"A"`` arrival, ``"D"`` departure).
_OPERATION = {
    "a": _ARRIVAL,
    "arrival": _ARRIVAL,
    "arrivals": _ARRIVAL,
    "approach": _ARRIVAL,
    "landing": _ARRIVAL,
    "d": _DEPARTURE,
    "departure": _DEPARTURE,
    "departures": _DEPARTURE,
    "takeoff": _DEPARTURE,
    "take-off": _DEPARTURE,
}
#: Supported NPD noise metrics for the Doc 29 chain.
_METRICS = ("SEL", "LAmax")
#: The ANP database's identifier for an aircraft's default fixed-point profile,
#: selected when the caller passes ``profile_id=None``.
_DEFAULT_PROFILE_ID = "DEFAULT"


def _operation_code(operation: str) -> str:
    """Normalise an operation label to the ANP code ``"A"``/``"D"``."""
    key = str(operation).strip().lower()
    if key not in _OPERATION:
        msg = (
            f"'operation' must be 'departure'/'D' or 'arrival'/'A', got {operation!r}."
        )
        raise ValueError(msg)
    return _OPERATION[key]


def _rows(text: str) -> list[dict[str, str]]:
    """Parse a semicolon-delimited ANP CSV table into a list of row mappings."""
    reader = csv.DictReader(text.splitlines(), delimiter=";")
    return [
        {(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader
    ]


def _pick(name: str, tables: Mapping[str, str]) -> str:
    """Resolve one logical ANP table by a case-insensitive filename keyword.

    Accepts both the archive naming (``ANP2.3_NPD_data.csv``) and the curated
    subset naming (``NPD_data.csv``).
    """
    matches = [f for f in tables if name in f.lower()]
    if len(matches) > 1:
        msg = (
            f"ambiguous ANP table for {name!r}: {sorted(matches)}. Keep a single "
            f"file per table in the export directory."
        )
        raise ValueError(msg)
    if not matches:
        msg = f"no ANP table matching {name!r} found (looked in: {sorted(tables)})."
        raise FileNotFoundError(msg)
    return tables[matches[0]]


def _distances_m(header: Iterable[str]) -> NDArray[np.float64]:
    """Slant distances (metres) parsed from the ``L_<ft>ft`` NPD column headers."""
    dist_ft: list[float] = []
    for col in header:
        c = col.strip()
        if c.startswith("L_") and c.endswith("ft"):
            dist_ft.append(float(c[2:-2]))
    if len(dist_ft) < _MIN_NPD_DISTANCES:
        msg = "NPD table has fewer than two 'L_<ft>ft' distance columns."
        raise ValueError(msg)
    distances = np.asarray(dist_ft, dtype=np.float64) * _FT_M
    if np.any(np.diff(distances) <= 0.0):
        msg = "NPD 'L_<ft>ft' distance columns must be strictly increasing."
        raise ValueError(msg)
    distances.flags.writeable = False  # shared across every AnpNpdCurves
    return distances


@dataclass(frozen=True)
class AnpNpdCurves:
    """ANP Noise-Power-Distance curves for one aircraft, metric and operation.

    :ivar aircraft_id: ANP aircraft identifier.
    :ivar npd_id: ANP noise identifier (shared by aircraft with the same NPD set).
    :ivar metric: ``"SEL"`` or ``"LAmax"``.
    :ivar operation: ``"A"`` (arrival) or ``"D"`` (departure).
    :ivar power_parameter: Name/unit of the power setting (e.g. corrected net thrust).
    :ivar powers: Tabulated engine power settings (1-D, strictly increasing).
    :ivar distances: Tabulated slant distances, in metres (1-D, strictly increasing).
    :ivar levels: Tabulated event levels, shape ``(len(powers), len(distances))``, in dB.

    The ``powers``, ``distances`` and ``levels`` arrays are read-only views shared
    with the parent database; copy them before mutating.
    """

    aircraft_id: str
    npd_id: str
    metric: str
    operation: str
    power_parameter: str
    powers: NDArray[np.float64]
    distances: NDArray[np.float64]
    levels: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Reject a curve set whose level table does not match its own two axes.

        The table's axes are what say which power setting and which slant
        distance every level belongs to, and this class is where a CSV export
        becomes one: a malformed export reaches the Doc 29 chain, the ``.plot()``
        and :meth:`level` as a table whose entries are attached to the wrong
        settings, and each of those reports it, if at all, as a shape it could
        not use rather than as the aircraft whose table was short. Checking on
        construction names the export where it was read.

        The two axes are pinned separately because they are independent: the
        bundled database holds sets of anything from two power settings to
        thirteen over the same ten distances, so one count over both would
        reject every set but the two whose axes happen to be equally long.

        :raises ValueError: if the level table disagrees with the powers or the
            distances.
        """
        require_ranks(self, powers=1, distances=1, levels=2)
        require_same_length(self, "powers", "levels", axis="power setting")
        require_same_length(self, "distances", ("levels", 1), axis="slant distance")

    def level(
        self, power: float, distance: NDArray[np.float64] | list[float] | float
    ) -> NDArray[np.float64]:
        """Interpolated NPD level ``L(P, d)`` (Doc 29 Eq. 4-3/4-4).

        :param power: Query engine power setting.
        :param distance: Query slant distance(s), in metres.
        :return: The interpolated level per query distance, in dB.
        """
        from .airport_noise import npd_level

        return npd_level(self.powers, self.distances, self.levels, power, distance)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the NPD curve at each tabulated power versus slant distance."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_anp_npd

        return plot_anp_npd(self, ax=ax, language=check_language(language), **kwargs)


@dataclass(frozen=True)
class AnpProfile:
    """Default fixed-point trajectory of an ANP aircraft as a Doc 29 flight path.

    :ivar aircraft_id: ANP aircraft identifier.
    :ivar operation: ``"A"`` (arrival) or ``"D"`` (departure).
    :ivar profile_id: ANP profile label (usually ``"DEFAULT"``).
    :ivar stage_length: ANP stage length (trip-distance/weight bin).
    :ivar path: Flight-path points, shape ``(N, 5)``: ``x, y, z`` (m, along-track,
        lateral, altitude), engine power setting and true airspeed (m/s).
    :ivar ground_roll: Boolean mask (length ``N-1``) of takeoff ground-roll segments.
    :ivar landing_roll: Boolean mask (length ``N-1``) of landing rollout segments.

    ``path`` is a read-only view shared with the parent database; copy it before
    mutating.
    """

    aircraft_id: str
    operation: str
    profile_id: str
    stage_length: int
    path: NDArray[np.float64]
    ground_roll: NDArray[np.bool_]
    landing_roll: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Reject a profile whose roll masks do not cover its path's segments.

        The masks are per segment and the path is per point, so an ``N``-point
        trajectory takes masks of ``N-1``; ``path`` is the authority here and
        the masks are what have to follow it. The figure marks the runway by
        OR-ing the two masks and folding the result onto both endpoints of
        every roll span, so a mask of any other length raises from numpy about
        operands it could not broadcast, naming neither the mask nor the
        profile it was read from. The Doc 29 chain does check the masks, but
        against the path it was handed at the call, so it reports the trajectory
        rather than the export the pair actually came from.

        :raises ValueError: if a mask does not carry one entry per path segment.
        """
        require_ranks(self, path=2, ground_roll=1, landing_roll=1)
        owner = type(self).__name__
        require_equal_counts(
            owner,
            {
                "path": max(len(self.path) - 1, 0),
                "ground_roll": require_axis_count(
                    self.ground_roll, owner, "ground_roll", "segment", rank=None
                ),
                "landing_roll": require_axis_count(
                    self.landing_roll, owner, "landing_roll", "segment", rank=None
                ),
            },
            "segment",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the trajectory altitude versus along-track distance."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_anp_profile

        return plot_anp_profile(
            self, ax=ax, language=check_language(language), **kwargs
        )


@dataclass(frozen=True)
class AnpAircraft:
    """One ANP aircraft type: metadata plus NPD/profile access and Doc 29 wiring.

    :ivar aircraft_id: ANP aircraft identifier (e.g. ``"747100"``).
    :ivar description: Human-readable aircraft/engine description.
    :ivar engine_type: ``"Jet"``, ``"Turboprop"`` or ``"Piston"``.
    :ivar num_engines: Number of engines.
    :ivar weight_class: ICAO wake weight class.
    :ivar mounting: Doc 29 engine mounting (``"wing"``/``"fuselage"``/``"propeller"``).
    :ivar npd_id: ANP noise identifier.
    :ivar power_parameter: Name/unit of the NPD power parameter.
    """

    aircraft_id: str
    description: str
    engine_type: str
    num_engines: int
    weight_class: str
    mounting: str
    npd_id: str
    power_parameter: str
    _database: AnpDatabase = field(repr=False, compare=False)

    def npd_curves(self, operation: str, metric: str = "SEL") -> AnpNpdCurves:
        """NPD curves for this aircraft (see :meth:`AnpDatabase.npd_curves`)."""
        return self._database.npd_curves(self.aircraft_id, operation, metric)

    def profile(
        self, operation: str, stage_length: int = 1, *, profile_id: str | None = None
    ) -> AnpProfile:
        """Fixed-point profile (see :meth:`AnpDatabase.profile`)."""
        return self._database.profile(
            self.aircraft_id, operation, stage_length, profile_id=profile_id
        )

    def event_level(
        self,
        observer: NDArray[np.float64] | list[float],
        operation: str,
        *,
        stage_length: int = 1,
        metric: str = "exposure",
        temperature: float = 15.0,
        pressure: float = 101.325,
    ) -> FlyoverResult:
        """Single-event level at a receiver (see :meth:`AnpDatabase.event_level`)."""
        return self._database.event_level(
            self.aircraft_id,
            observer,
            operation,
            stage_length=stage_length,
            metric=metric,
            temperature=temperature,
            pressure=pressure,
        )

    def noise_contour(
        self,
        operation: str,
        *,
        x: NDArray[np.float64] | list[float],
        y: NDArray[np.float64] | list[float],
        stage_length: int = 1,
        metric: str = "exposure",
        temperature: float = 15.0,
        pressure: float = 101.325,
    ) -> NoiseContourResult:
        """Single-event ground contour (see :meth:`AnpDatabase.noise_contour`)."""
        return self._database.noise_contour(
            self.aircraft_id,
            operation,
            x=x,
            y=y,
            stage_length=stage_length,
            metric=metric,
            temperature=temperature,
            pressure=pressure,
        )


class AnpDatabase:
    """A parsed ANP database (aircraft metadata, NPD curves and default profiles).

    Build one with :func:`load_anp_database`. NPD curves are available for every
    aircraft; default profiles are available for aircraft that have a fixed-point
    trajectory in the database.
    """

    def __init__(
        self,
        aircraft: Mapping[str, dict[str, str]],
        npd: Mapping[
            tuple[str, str, str], tuple[NDArray[np.float64], NDArray[np.float64]]
        ],
        distances: NDArray[np.float64],
        profiles: Mapping[tuple[str, str, str, int], NDArray[np.float64]],
    ) -> None:
        self._aircraft = dict(aircraft)
        self._npd = dict(npd)
        self._distances = distances
        self._profiles = dict(profiles)

    @property
    def aircraft_ids(self) -> list[str]:
        """Sorted list of aircraft identifiers in the database."""
        return sorted(self._aircraft)

    def aircraft(self, aircraft_id: str) -> AnpAircraft:
        """Return the :class:`AnpAircraft` for an identifier.

        :raises KeyError: If the identifier is not in the database.
        """
        if aircraft_id not in self._aircraft:
            msg = (
                f"aircraft {aircraft_id!r} not in this ANP database "
                f"(available: {self.aircraft_ids})."
            )
            raise KeyError(msg)
        m = self._aircraft[aircraft_id]
        lat = m.get("Lateral Directivity Identifier", "").strip().lower()
        return AnpAircraft(
            aircraft_id=aircraft_id,
            description=m.get("Description", ""),
            engine_type=m.get("Engine Type", ""),
            num_engines=int(float(m.get("Number Of Engines", "0") or 0)),
            weight_class=m.get("Weight Class", ""),
            mounting=_MOUNTING.get(lat, "wing"),
            npd_id=m.get("NPD_ID", ""),
            power_parameter=m.get("Power Parameter", ""),
            _database=self,
        )

    def npd_curves(
        self, aircraft_id: str, operation: str, metric: str = "SEL"
    ) -> AnpNpdCurves:
        """NPD curves for an aircraft, operation and noise metric.

        :param aircraft_id: ANP aircraft identifier.
        :param operation: ``"departure"``/``"D"`` or ``"arrival"``/``"A"``.
        :param metric: ``"SEL"`` (default) or ``"LAmax"``.
        :return: An :class:`AnpNpdCurves`.
        :raises KeyError: If the aircraft has no NPD data for the request.
        :raises ValueError: If the metric or operation is unknown.
        """
        if metric not in _METRICS:
            msg = f"'metric' must be one of {_METRICS}, got {metric!r}."
            raise ValueError(msg)
        op = _operation_code(operation)
        m = self._aircraft.get(aircraft_id)
        if m is None:
            msg = f"aircraft {aircraft_id!r} not in this ANP database."
            raise KeyError(msg)
        npd_id = m.get("NPD_ID", "")
        key = (npd_id, metric, op)
        if key not in self._npd:
            msg = (
                f"no {metric} NPD data for aircraft {aircraft_id!r} "
                f"(NPD_ID {npd_id!r}), operation {op!r}."
            )
            raise KeyError(msg)
        powers, levels = self._npd[key]
        return AnpNpdCurves(
            aircraft_id=aircraft_id,
            npd_id=npd_id,
            metric=metric,
            operation=op,
            power_parameter=m.get("Power Parameter", ""),
            powers=powers,
            distances=self._distances,
            levels=levels,
        )

    def profile(
        self,
        aircraft_id: str,
        operation: str,
        stage_length: int = 1,
        *,
        profile_id: str | None = None,
    ) -> AnpProfile:
        """Fixed-point trajectory for an aircraft, operation and stage length.

        Aircraft may ship several fixed-point profiles for the same operation
        and stage length (e.g. weight variants). With ``profile_id=None`` the
        ``"DEFAULT"`` profile is selected when present; otherwise the single
        available profile is used, and an ambiguous request (several profiles,
        none named ``"DEFAULT"``) raises listing the identifiers.

        :param aircraft_id: ANP aircraft identifier.
        :param operation: ``"departure"``/``"D"`` or ``"arrival"``/``"A"``.
        :param stage_length: ANP stage length (default 1).
        :param profile_id: Optional ANP profile identifier (e.g. ``"DEFAULT"``,
            ``"3000LB"``); ``None`` (default) selects as described above.
        :return: An :class:`AnpProfile` (a Doc 29 flight path with ground-roll masks).
        :raises KeyError: If the aircraft is unknown, has no fixed-point profile
            for the request, or ``profile_id`` is not among the available ones.
        :raises ValueError: If ``profile_id`` is ``None`` and several profiles
            exist with none of them named ``"DEFAULT"``.
        """
        if aircraft_id not in self._aircraft:
            msg = (
                f"aircraft {aircraft_id!r} not in this ANP database "
                f"(available: {self.aircraft_ids})."
            )
            raise KeyError(msg)
        op = _operation_code(operation)
        stage = int(stage_length)
        ids = sorted(
            pid
            for (a, o, pid, sl) in self._profiles
            if a == aircraft_id and o == op and sl == stage
        )
        if not ids:
            avail = sorted(
                {
                    sl
                    for (a, o, _pid, sl) in self._profiles
                    if a == aircraft_id and o == op
                }
            )
            msg = (
                f"no fixed-point profile for aircraft {aircraft_id!r}, operation "
                f"{op!r}, stage length {stage_length} (available stage lengths: "
                f"{avail}). Aircraft with only procedural-step profiles are not "
                f"supported by this bridge."
            )
            raise KeyError(msg)
        if profile_id is not None:
            pid = str(profile_id)
            if pid not in ids:
                msg = (
                    f"no fixed-point profile {pid!r} for aircraft "
                    f"{aircraft_id!r}, operation {op!r}, stage length "
                    f"{stage_length} (available profiles: {ids})."
                )
                raise KeyError(msg)
        elif _DEFAULT_PROFILE_ID in ids:
            pid = _DEFAULT_PROFILE_ID
        elif len(ids) == 1:
            pid = ids[0]
        else:
            msg = (
                f"aircraft {aircraft_id!r}, operation {op!r}, stage length "
                f"{stage_length} has several fixed-point profiles and none is "
                f"{_DEFAULT_PROFILE_ID!r}: {ids}. Pass profile_id= to choose one."
            )
            raise ValueError(msg)
        path = self._profiles[(aircraft_id, op, pid, stage)]
        # Ground-roll segments run along the runway: both endpoints at field
        # elevation. Tabulated ground points sit at exactly 0 m and the lowest
        # airborne point is above 150 m, so a 1 m threshold separates them.
        on_ground = np.abs(path[:, 2]) <= _GROUND_ALTITUDE_M
        seg_zero = on_ground[:-1] & on_ground[1:]
        ground_roll = seg_zero & (op == _DEPARTURE)
        landing_roll = seg_zero & (op == _ARRIVAL)
        return AnpProfile(
            aircraft_id=aircraft_id,
            operation=op,
            profile_id=pid,
            stage_length=stage,
            path=path,
            ground_roll=ground_roll,
            landing_roll=landing_roll,
        )

    def _doc29_inputs(
        self,
        aircraft_id: str,
        operation: str,
        stage_length: int,
    ) -> tuple[
        AnpAircraft,
        AnpProfile,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Gather (aircraft, profile, powers, distances, SEL, LAmax) for the chain."""
        acft = self.aircraft(aircraft_id)
        prof = self.profile(aircraft_id, operation, stage_length)
        sel = self.npd_curves(aircraft_id, operation, "SEL")
        lmax = self.npd_curves(aircraft_id, operation, "LAmax")
        # The Doc 29 chain reads SEL and LAmax on a single (power, distance) grid,
        # so the two metrics must share power settings. They always do in the ANP
        # database; this guards a malformed user-supplied export.
        if not np.array_equal(sel.powers, lmax.powers):
            msg = (
                f"SEL and LAmax NPD power settings differ for aircraft "
                f"{aircraft_id!r}, operation {operation!r}: {sel.powers} vs "
                f"{lmax.powers}."
            )
            raise ValueError(msg)
        return acft, prof, sel.powers, sel.distances, sel.levels, lmax.levels

    def event_level(
        self,
        aircraft_id: str,
        observer: NDArray[np.float64] | list[float],
        operation: str,
        *,
        stage_length: int = 1,
        metric: str = "exposure",
        temperature: float = 15.0,
        pressure: float = 101.325,
    ) -> FlyoverResult:
        """Doc 29 single-event level of an ANP aircraft at a receiver.

        Feeds the aircraft's default fixed-point profile and NPD curves into
        :func:`phonometry.aircraft.airport_noise.event_level`.

        :param aircraft_id: ANP aircraft identifier.
        :param observer: Receiver position ``(x, y, z)``, in metres.
        :param operation: ``"departure"``/``"D"`` or ``"arrival"``/``"A"``.
        :param stage_length: ANP stage length (default 1).
        :param metric: ``"exposure"`` (SEL) or ``"maximum"`` (LAmax).
        :param temperature: Aerodrome air temperature, in °C.
        :param pressure: Aerodrome air pressure, in kPa.
        :return: A :class:`~phonometry.aircraft.airport_noise.FlyoverResult`.
        """
        acft, prof, p, d, sel, lmax = self._doc29_inputs(
            aircraft_id, operation, stage_length
        )
        return event_level(
            prof.path,
            observer,
            p,
            d,
            sel,
            lmax,
            mounting=acft.mounting,
            metric=metric,
            atmosphere=AerodromeAtmosphere(temperature, pressure),
            segments=FlightSegmentState(
                ground_roll=prof.ground_roll, landing_roll=prof.landing_roll
            ),
        )

    def noise_contour(
        self,
        aircraft_id: str,
        operation: str,
        *,
        x: NDArray[np.float64] | list[float],
        y: NDArray[np.float64] | list[float],
        stage_length: int = 1,
        metric: str = "exposure",
        temperature: float = 15.0,
        pressure: float = 101.325,
    ) -> NoiseContourResult:
        """Doc 29 single-event ground contour of an ANP aircraft.

        Feeds the aircraft's default fixed-point profile and NPD curves into
        :func:`phonometry.aircraft.airport_noise.noise_contour`.

        :param aircraft_id: ANP aircraft identifier.
        :param operation: ``"departure"``/``"D"`` or ``"arrival"``/``"A"``.
        :param x: Grid x coordinates (along-track), in metres.
        :param y: Grid y coordinates (lateral), in metres.
        :param stage_length: ANP stage length (default 1).
        :param metric: ``"exposure"`` (SEL) or ``"maximum"`` (LAmax).
        :param temperature: Aerodrome air temperature, in °C.
        :param pressure: Aerodrome air pressure, in kPa.
        :return: A :class:`~phonometry.aircraft.airport_noise.NoiseContourResult`.
        """
        acft, prof, p, d, sel, lmax = self._doc29_inputs(
            aircraft_id, operation, stage_length
        )
        return noise_contour(
            prof.path,
            p,
            d,
            sel,
            lmax,
            x=x,
            y=y,
            mounting=acft.mounting,
            metric=metric,
            atmosphere=AerodromeAtmosphere(temperature, pressure),
            segments=FlightSegmentState(
                ground_roll=prof.ground_roll, landing_roll=prof.landing_roll
            ),
        )


def _read_tables(path: Path | str | None) -> dict[str, str]:
    """Return ``{filename: text}`` for the bundled subset or a user directory."""
    if path is None:
        from importlib.resources import files

        root = files("phonometry.aircraft.data.anp")
        out: dict[str, str] = {}
        for entry in root.iterdir():
            if entry.name.lower().endswith(".csv"):
                # utf-8-sig tolerates a leading BOM in exported ANP CSVs.
                out[entry.name] = entry.read_text(encoding="utf-8-sig")
        return out
    import pathlib

    directory = pathlib.Path(path)
    if not directory.is_dir():
        msg = f"ANP database path {path!r} is not a directory."
        raise NotADirectoryError(msg)
    files_found = sorted(directory.glob("*.csv")) + sorted(directory.glob("*.CSV"))
    if not files_found:
        msg = f"no .csv ANP tables found in {path!r}."
        raise FileNotFoundError(msg)
    return {f.name: f.read_text(encoding="utf-8-sig") for f in files_found}


def _parse_npd(
    text: str,
) -> tuple[
    dict[tuple[str, str, str], tuple[NDArray[np.float64], NDArray[np.float64]]],
    NDArray[np.float64],
]:
    """Parse the NPD table into ``{(npd_id, metric, op): (powers, levels)}``."""
    rows = _rows(text)
    if not rows:
        msg = "empty NPD table."
        raise ValueError(msg)
    level_cols = [c for c in rows[0] if c.startswith("L_") and c.endswith("ft")]
    distances = _distances_m(rows[0].keys())
    grouped: dict[tuple[str, str, str], list[tuple[float, list[float]]]] = {}
    for row in rows:
        metric = row["Noise Metric"]
        if metric not in _METRICS:
            continue
        key = (row["NPD_ID"], metric, row["Op Mode"])
        power = float(row["Power Setting"])
        levels = [float(row[c]) for c in level_cols]
        grouped.setdefault(key, []).append((power, levels))
    npd: dict[
        tuple[str, str, str], tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = {}
    for key, entries in grouped.items():
        entries.sort(key=lambda e: e[0])
        powers = np.asarray([e[0] for e in entries], dtype=np.float64)
        levels_arr = np.asarray([e[1] for e in entries], dtype=np.float64)
        powers.flags.writeable = False  # exposed by reference on AnpNpdCurves
        levels_arr.flags.writeable = False
        npd[key] = (powers, levels_arr)
    return npd, distances


def _parse_profiles(
    text: str,
) -> dict[tuple[str, str, str, int], NDArray[np.float64]]:
    """Parse fixed-point profiles into ``{(acft, op, profile_id, stage): path}``.

    The path is the Doc 29 ``(N, 5)`` array ``x, y, z, power, speed`` in SI units,
    ordered by point number. Each profile is keyed by its full ANP identity
    (aircraft, operation, profile identifier, stage length), so aircraft with
    several fixed-point profiles for the same operation and stage length (e.g.
    weight variants) stay separate. Point numbers must be unique and consecutive
    within each profile; a violation means rows from distinct profiles collided
    (or the CSV is malformed), so it raises instead of silently interleaving.
    """
    rows = _rows(text)
    grouped: dict[tuple[str, str, str, int], list[tuple[int, list[float]]]] = {}
    for row in rows:
        key = (
            row["ACFT_ID"],
            row["Op Type"],
            row["Profile_ID"],
            int(float(row["Stage Length"])),
        )
        point = int(float(row["Point Number"]))
        x = float(row["Distance (ft)"]) * _FT_M
        z = float(row["Altitude AFE (ft)"]) * _FT_M
        speed = float(row["TAS (kt)"]) * _KT_MS
        power = float(row["Power Setting"])
        grouped.setdefault(key, []).append((point, [x, 0.0, z, power, speed]))
    profiles: dict[tuple[str, str, str, int], NDArray[np.float64]] = {}
    for key, pts in grouped.items():
        pts.sort(key=lambda e: e[0])
        numbers = [p[0] for p in pts]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            acft, op, pid, stage = key
            msg = (
                f"fixed-point profile for aircraft {acft!r}, operation {op!r}, "
                f"profile {pid!r}, stage length {stage} has duplicate or "
                f"non-consecutive point numbers {numbers}; the table is "
                f"malformed."
            )
            raise ValueError(msg)
        path = np.asarray([p[1] for p in pts], dtype=np.float64)
        path.flags.writeable = False  # exposed by reference on AnpProfile
        profiles[key] = path
    return profiles


def load_anp_database(path: Path | str | None = None) -> AnpDatabase:
    """Load an EASA ANP database (aircraft, NPD curves and default profiles).

    :param path: Directory of an ANP CSV export (the ``*Aircraft.csv``,
        ``*NPD_data.csv``, ``*fixed_point_profiles.csv`` tables). If ``None``
        (default), loads the full EASA ANP database v2.3 shipped with the
        package (see ``aircraft/data/anp/PROVENANCE.md``).
    :return: An :class:`AnpDatabase`.
    :raises FileNotFoundError: If a required table is missing.
    """
    tables = _read_tables(path)
    aircraft_rows = _rows(_pick("aircraft", tables))
    aircraft = {row["ACFT_ID"]: row for row in aircraft_rows}
    npd, distances = _parse_npd(_pick("npd", tables))
    profiles = _parse_profiles(_pick("fixed_point", tables))
    return AnpDatabase(
        aircraft=aircraft, npd=npd, distances=distances, profiles=profiles
    )
