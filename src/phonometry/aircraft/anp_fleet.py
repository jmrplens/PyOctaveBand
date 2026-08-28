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

Aircraft whose default trajectory is published as *procedural steps* rather than
as fixed points are reached through :mod:`phonometry.aircraft.flight_performance`,
the ECAC Doc 29 Vol. 2 Appendix B performance model:
:meth:`AnpDatabase.flight_profile` flies the published procedure for an
aerodrome and its weather, and :meth:`AnpDatabase.procedural_profile` returns the
result as the same Doc 29 flight path the fixed-point bridge produces, so either
kind of profile feeds the same chain. NPD curves are available for every aircraft
regardless.

Source (clean-room, implemented from the published table format): EASA ANP
database v2.3 (2020) and the ECAC Doc 29 4th ed. Vol 2 NPD/profile conventions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._internal.validation import (
    require_axis_count,
    require_equal_counts,
    require_ranks,
    require_same_length,
)
from .airport_noise import (
    AerodromeAtmosphere,
    EventMetric,
    FlightSegmentState,
    FlyoverResult,
    NoiseContourResult,
    event_level,
    noise_contour,
)
from .flight_performance import (
    AerodynamicCoefficients,
    ApproachStep,
    DepartureStep,
    FlightProfile,
    JetEngineCoefficients,
    PerformanceAircraft,
    PropellerEngineCoefficients,
    approach_profile,
    departure_profile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from .flight_performance import Aerodrome

#: Feet-to-metres conversion (NPD slant distances and profile altitudes/distances).
_FT_M = 0.3048
#: Knots-to-metres-per-second conversion (profile true airspeed).
_KT_MS = 0.514444
#: Altitude below which a profile point counts as on the ground, in metres.
_GROUND_ALTITUDE_M = 1.0
#: Standard sea-level pressure, kPa: the unit the Doc 29 atmospheric impedance
#: adjustment reads, where Appendix B works in inHg.
_STANDARD_PRESSURE_KPA = 101.325
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
#: ANP column headings read from more than one table. Named because the tables
#: spell them exactly this way, spaces, capitals and parenthesised unit and all,
#: and a typo in one copy of several is a table that quietly loads short.
_COL_STAGE_LENGTH = "Stage Length"
_COL_DISTANCE_FT = "Distance (ft)"
_COL_THRUST_RATING = "Thrust Rating"


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


def _select_profile_id(
    ids: list[str], profile_id: str | None, aircraft_id: str, what: str
) -> str:
    """Pick one profile identifier out of those an aircraft publishes.

    The same rule the fixed-point bridge follows: an explicit request wins,
    ``"DEFAULT"`` is taken when it is there, a lone profile is taken because
    there is nothing to choose between, and anything else is ambiguous and says
    so rather than picking alphabetically.

    :raises KeyError: if *profile_id* is not among *ids*.
    :raises ValueError: if the choice is ambiguous.
    """
    if profile_id is not None:
        pid = str(profile_id)
        if pid not in ids:
            msg = (
                f"no procedural-step profile {pid!r} for aircraft "
                f"{aircraft_id!r}, {what} (available profiles: {ids})."
            )
            raise KeyError(msg)
        return pid
    if _DEFAULT_PROFILE_ID in ids:
        return _DEFAULT_PROFILE_ID
    if len(ids) == 1:
        return ids[0]
    msg = (
        f"aircraft {aircraft_id!r}, {what} has several procedural-step "
        f"profiles and none is {_DEFAULT_PROFILE_ID!r}: {ids}. Pass "
        f"profile_id= to choose one."
    )
    raise ValueError(msg)


def _stage_label(stage_length: int | str) -> str:
    """Normalise an ANP stage length to the label its tables are keyed by.

    The stage-length column is not a number. Alongside the numbered
    trip-distance bins the weights and procedural-step tables carry ``"M"`` for
    a maximum-weight procedure, so an ``int`` key would drop every profile that
    uses it. A number is written without its decimal point either way, so
    ``1``, ``"1"`` and ``"1.0"`` all name the same bin.

    :raises ValueError: for a number that falls between the bins. They are
        whole, and truncating ``1.5`` to bin 1 would answer with a profile flown
        at a different trip length and say nothing about the substitution.
    """
    text = str(stage_length).strip().upper()
    try:
        number = float(text)
    except ValueError:
        return text
    if not number.is_integer():
        msg = (
            f"'stage_length' must name a whole ANP bin or 'M'; got "
            f"{stage_length!r}, which falls between two bins."
        )
        raise ValueError(msg)
    return str(int(number))


def _optional(value: str) -> float | None:
    """One ANP cell as a number, or ``None`` for the blank the table leaves.

    A blank is a quantity the row does not define -- the take-off ground-roll
    coefficient of a flap setting no take-off uses, the rate of climb of a step
    that is not an Accelerate -- and the performance model refuses a zero
    standing in for one, so the distinction survives the read.
    """
    text = value.strip()
    return float(text) if text else None


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
    :ivar stage_length: ANP stage length (trip-distance/weight bin). Usually one
        of the numbered bins, but the database's own column also carries ``"M"``
        for a maximum-weight procedure, so the label is kept as it is read.
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
    stage_length: int | str
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

        The column count of ``path`` is pinned too, not only its rank: the
        type documents shape ``(N, 5)`` and the figure reads column 2 for the
        altitude, so a path short of it passes every per-segment count above
        and dies at ``path[:, 2]`` with an ``IndexError`` naming neither the
        field nor the profile.

        :raises ValueError: if a mask does not carry one entry per path
            segment, or ``path`` does not carry the five documented columns.
        """
        require_ranks(self, path=2, ground_roll=1, landing_roll=1)
        path_columns = 5
        if np.shape(self.path)[1] != path_columns:
            msg = (
                "AnpProfile: 'path' must have shape (N, 5) "
                "(x, y, z, power, speed); got shape "
                f"{np.shape(self.path)}."
            )
            raise ValueError(msg)
        owner = type(self).__name__
        require_equal_counts(
            owner,
            {
                "path segments": max(len(self.path) - 1, 0),
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
        aerodrome: Aerodrome | None = None,
        stage_length: int | str = 1,
        metric: EventMetric = "exposure",
        temperature: float | None = None,
        pressure: float | None = None,
    ) -> FlyoverResult:
        """Single-event level at a receiver (see :meth:`AnpDatabase.event_level`)."""
        return self._database.event_level(
            self.aircraft_id,
            observer,
            operation,
            aerodrome=aerodrome,
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
        aerodrome: Aerodrome | None = None,
        stage_length: int | str = 1,
        metric: EventMetric = "exposure",
        temperature: float | None = None,
        pressure: float | None = None,
    ) -> NoiseContourResult:
        """Single-event ground contour (see :meth:`AnpDatabase.noise_contour`)."""
        return self._database.noise_contour(
            self.aircraft_id,
            operation,
            x=x,
            y=y,
            aerodrome=aerodrome,
            stage_length=stage_length,
            metric=metric,
            temperature=temperature,
            pressure=pressure,
        )


@dataclass(frozen=True)
class _PerformanceTables:
    """The five ANP tables the Doc 29 Appendix B model reads, parsed.

    Kept apart from the NPD and fixed-point tables because they are optional:
    an export that carries only NPD curves and trajectories is a complete
    database for the fixed-point bridge, and only a procedural-step profile
    needs these. Each field is keyed by aircraft identifier so one lookup
    gathers everything one aeroplane needs.
    """

    jet: dict[str, dict[str, JetEngineCoefficients]] = field(default_factory=dict)
    propeller: dict[str, dict[str, PropellerEngineCoefficients]] = field(
        default_factory=dict
    )
    aerodynamic: dict[str, dict[tuple[str, str], AerodynamicCoefficients]] = field(
        default_factory=dict
    )
    weights: dict[tuple[str, str], float] = field(default_factory=dict)
    departure_steps: dict[tuple[str, str, str], tuple[DepartureStep, ...]] = field(
        default_factory=dict
    )
    approach_steps: dict[tuple[str, str], tuple[ApproachStep, ...]] = field(
        default_factory=dict
    )


class AnpDatabase:
    """A parsed ANP database (aircraft metadata, NPD curves and default profiles).

    Build one with :func:`load_anp_database`. NPD curves are available for every
    aircraft; default profiles are available for aircraft that have a fixed-point
    trajectory in the database, and procedural-step profiles for those whose
    coefficient and step tables the export carries.
    """

    def __init__(
        self,
        aircraft: Mapping[str, dict[str, str]],
        npd: Mapping[
            tuple[str, str, str], tuple[NDArray[np.float64], NDArray[np.float64]]
        ],
        distances: NDArray[np.float64],
        profiles: Mapping[tuple[str, str, str, int], NDArray[np.float64]],
        performance: _PerformanceTables | None = None,
    ) -> None:
        self._aircraft = dict(aircraft)
        self._npd = dict(npd)
        self._distances = distances
        self._profiles = dict(profiles)
        self._performance = (
            performance if performance is not None else _PerformanceTables()
        )

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
        stage_length: int | str = 1,
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
        # Normalised first and outside the guard below, so a number between the
        # bins is refused as that, with its own message, rather than swept into
        # the one for a non-numeric label.
        label = _stage_label(stage_length)
        try:
            stage = int(label)
        except ValueError:
            # The fixed-point table is keyed by the numbered bins alone; "M" is
            # a procedural-step label. Left to int() this arrives as a bare
            # ValueError quoting a string and naming neither the argument nor
            # the way in.
            msg = (
                f"'stage_length' {stage_length!r} names no fixed-point bin for "
                f"aircraft {aircraft_id!r}: the fixed-point table is keyed by "
                "the numbered trip-distance bins, and a maximum-weight "
                "procedure is flown with flight_profile() instead, which needs "
                "an aerodrome."
            )
            raise ValueError(msg) from None
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
                f"{avail}). An aircraft with only procedural-step profiles is "
                f"flown with flight_profile() instead, which needs an aerodrome."
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

    def performance_aircraft(self, aircraft_id: str) -> PerformanceAircraft:
        """The aeroplane's ECAC Doc 29 Vol. 2 Appendix B coefficient set.

        Gathers the engine, aerodynamic and weight tables into the
        :class:`~phonometry.aircraft.flight_performance.PerformanceAircraft` the
        performance model takes. The engine count, the maximum sea-level static
        thrust and the maximum landing weight come from the aircraft table; the
        approach weight follows from the last of those as Doc 29 defines it.

        :param aircraft_id: ANP aircraft identifier.
        :return: A
            :class:`~phonometry.aircraft.flight_performance.PerformanceAircraft`.
        :raises KeyError: If the aircraft is not in the database.
        :raises ValueError: If the export carries no performance tables at all,
            which is the case for an NPD-only export.
        """
        meta = self._aircraft.get(aircraft_id)
        if meta is None:
            msg = (
                f"aircraft {aircraft_id!r} not in this ANP database "
                f"(available: {self.aircraft_ids})."
            )
            raise KeyError(msg)
        tables = self._performance
        if not tables.aerodynamic:
            msg = (
                "this ANP export carries no performance tables (the "
                "aerodynamic, engine-coefficient and procedural-step CSVs), so "
                "no procedure can be flown from it. Load the bundled database, "
                "or point load_anp_database() at a full export."
            )
            raise ValueError(msg)
        return PerformanceAircraft(
            aircraft_id=aircraft_id,
            engines=int(float(meta.get("Number Of Engines", "0") or 0)),
            max_static_thrust_lb=float(
                meta.get("Max Sea Level Static Thrust (lb)", "0") or 0.0
            ),
            max_landing_weight_lb=float(
                meta.get("Max Gross Landing Weight (lb)", "0") or 0.0
            ),
            jet_coefficients=tables.jet.get(aircraft_id, {}),
            propeller_coefficients=tables.propeller.get(aircraft_id, {}),
            aerodynamic_coefficients=tables.aerodynamic.get(aircraft_id, {}),
        )

    def procedural_steps(
        self,
        aircraft_id: str,
        operation: str,
        stage_length: int | str = 1,
        *,
        profile_id: str | None = None,
    ) -> tuple[DepartureStep, ...] | tuple[ApproachStep, ...]:
        """The published procedure for an aircraft, as procedural steps.

        The rows of the ANP departure or approach procedural-step table, in step
        order, as the types
        :mod:`~phonometry.aircraft.flight_performance` flies. Approach
        procedures are not tabulated per stage length, so ``stage_length`` is
        read only for a departure.

        :param aircraft_id: ANP aircraft identifier.
        :param operation: ``"departure"``/``"D"`` or ``"arrival"``/``"A"``.
        :param stage_length: ANP stage length (default 1), departures only.
        :param profile_id: Optional ANP profile identifier; ``None`` (default)
            selects ``"DEFAULT"`` when present, or the single available one.
        :return: The steps, in order.
        :raises KeyError: If the aircraft has no procedural-step profile for the
            request, or ``profile_id`` is not among the available ones.
        :raises ValueError: If ``profile_id`` is ``None`` and several profiles
            exist with none of them named ``"DEFAULT"``.
        """
        return self._selected_procedural_steps(
            aircraft_id, operation, stage_length, profile_id=profile_id
        )[1]

    def _selected_procedural_steps(
        self,
        aircraft_id: str,
        operation: str,
        stage_length: int | str,
        *,
        profile_id: str | None,
    ) -> tuple[str, tuple[DepartureStep, ...] | tuple[ApproachStep, ...]]:
        """:meth:`procedural_steps`, and the identifier the procedure was found under.

        The flown profile is labelled with the procedure it was actually taken
        from, which is not always the one the caller named: ``profile_id=None``
        takes ``"DEFAULT"`` where the export publishes it and the lone profile
        where it publishes exactly one under another name.
        """
        op = _operation_code(operation)
        stage = _stage_label(stage_length)
        keyed: dict[str, tuple[DepartureStep, ...] | tuple[ApproachStep, ...]] = {}
        if op == _DEPARTURE:
            keyed.update(
                (pid, steps)
                for (acft, pid, sl), steps in self._performance.departure_steps.items()
                if acft == aircraft_id and sl == stage
            )
            what = f"departure, stage length {stage_length}"
        else:
            keyed.update(
                (pid, steps)
                for (acft, pid), steps in self._performance.approach_steps.items()
                if acft == aircraft_id
            )
            what = "approach"
        if not keyed:
            msg = f"no procedural-step profile for aircraft {aircraft_id!r}, {what}."
            raise KeyError(msg)
        pid = _select_profile_id(sorted(keyed), profile_id, aircraft_id, what)
        return pid, keyed[pid]

    def flight_profile(
        self,
        aircraft_id: str,
        operation: str,
        *,
        aerodrome: Aerodrome,
        stage_length: int | str = 1,
        profile_id: str | None = None,
        weight_lb: float | None = None,
    ) -> FlightProfile:
        """Fly an aircraft's published procedure into a Doc 29 flight profile.

        The ECAC Doc 29 Vol. 2 Appendix B model applied to this aircraft's ANP
        procedural steps: the profile depends on the aerodrome and its weather,
        which is why one has to be given and why the answer is not a table entry.

        :param aircraft_id: ANP aircraft identifier.
        :param operation: ``"departure"``/``"D"`` or ``"arrival"``/``"A"``.
        :param aerodrome: The
            :class:`~phonometry.aircraft.flight_performance.Aerodrome` to fly
            from, carrying its elevation, temperature, pressure and headwind.
        :param stage_length: ANP stage length (default 1), which selects the
            take-off weight of a departure.
        :param profile_id: Optional ANP profile identifier.
        :param weight_lb: Weight to fly at, lb. ``None`` (default) takes the
            ANP default weight for the stage length on a departure, and 90 % of
            the maximum landing weight on an arrival, as Doc 29 defines it.
        :return: A
            :class:`~phonometry.aircraft.flight_performance.FlightProfile`.
        :raises KeyError: If the aircraft has no procedural-step profile or no
            default weight for the request.
        """
        op = _operation_code(operation)
        acft = self.performance_aircraft(aircraft_id)
        pid, steps = self._selected_procedural_steps(
            aircraft_id, op, stage_length, profile_id=profile_id
        )
        if op == _ARRIVAL:
            return approach_profile(
                acft,
                cast("Sequence[ApproachStep]", steps),
                aerodrome=aerodrome,
                weight_lb=weight_lb,
                procedure_id=pid,
            )
        weight = weight_lb
        if weight is None:
            key = (aircraft_id, _stage_label(stage_length))
            if key not in self._performance.weights:
                available = sorted(
                    sl
                    for (acft_id, sl) in self._performance.weights
                    if acft_id == aircraft_id
                )
                msg = (
                    f"no default departure weight for aircraft {aircraft_id!r}, "
                    f"stage length {stage_length} (available stage lengths: "
                    f"{available}). Pass weight_lb= to fly another weight."
                )
                raise KeyError(msg)
            weight = self._performance.weights[key]
        return departure_profile(
            acft,
            cast("Sequence[DepartureStep]", steps),
            weight_lb=weight,
            aerodrome=aerodrome,
            procedure_id=pid,
        )

    def procedural_profile(
        self,
        aircraft_id: str,
        operation: str,
        *,
        aerodrome: Aerodrome,
        stage_length: int | str = 1,
        profile_id: str | None = None,
        weight_lb: float | None = None,
    ) -> AnpProfile:
        """A flown procedure as the flight path the Doc 29 noise chain reads.

        :meth:`flight_profile` converted into the same :class:`AnpProfile` the
        fixed-point bridge returns, so an aircraft that publishes procedural
        steps and one that publishes fixed points feed
        :meth:`event_level` and :meth:`noise_contour` alike. The units change on
        the way (Appendix B works in feet, knots and pounds; the noise chain in
        metres and metres per second) and the power setting stays the corrected
        net thrust per engine the NPD tables are indexed on.

        Takes the same arguments as :meth:`flight_profile`.

        :return: An :class:`AnpProfile`.
        """
        flown = self.flight_profile(
            aircraft_id,
            operation,
            aerodrome=aerodrome,
            stage_length=stage_length,
            profile_id=profile_id,
            weight_lb=weight_lb,
        )
        path = np.column_stack(
            (
                flown.distance_ft * _FT_M,
                np.zeros(len(flown.points)),
                flown.altitude_ft * _FT_M,
                flown.corrected_net_thrust_lb,
                flown.true_airspeed_kt * _KT_MS,
            )
        )
        path.flags.writeable = False  # exposed by reference on AnpProfile
        # A synthesised profile puts its ground points at exactly zero height,
        # so the same threshold the fixed-point bridge uses separates them.
        on_ground = np.abs(path[:, 2]) <= _GROUND_ALTITUDE_M
        seg_zero = on_ground[:-1] & on_ground[1:]
        return AnpProfile(
            aircraft_id=aircraft_id,
            operation=flown.operation,
            profile_id=flown.procedure_id,
            stage_length=_stage_label(stage_length),
            path=path,
            ground_roll=seg_zero & (flown.operation == _DEPARTURE),
            landing_roll=seg_zero & (flown.operation == _ARRIVAL),
        )

    def _doc29_inputs(
        self,
        aircraft_id: str,
        operation: str,
        stage_length: int | str,
        aerodrome: Aerodrome | None = None,
    ) -> tuple[
        AnpAircraft,
        AnpProfile,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Gather (aircraft, profile, powers, distances, SEL, LAmax) for the chain.

        Without an aerodrome the trajectory is the tabulated one, which is what
        every caller wanted while the fixed points were all this bridge could
        fly. With one it is the published procedure flown at that field, which
        is the only trajectory most of the fleet has.
        """
        acft = self.aircraft(aircraft_id)
        prof = (
            self.profile(aircraft_id, operation, stage_length)
            if aerodrome is None
            else self.procedural_profile(
                aircraft_id, operation, aerodrome=aerodrome, stage_length=stage_length
            )
        )
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
        aerodrome: Aerodrome | None = None,
        stage_length: int | str = 1,
        metric: EventMetric = "exposure",
        temperature: float | None = None,
        pressure: float | None = None,
    ) -> FlyoverResult:
        """Doc 29 single-event level of an ANP aircraft at a receiver.

        Feeds the aircraft's profile and NPD curves into
        :func:`phonometry.aircraft.airport_noise.event_level`.

        :param aircraft_id: ANP aircraft identifier.
        :param observer: Receiver position ``(x, y, z)``, in metres.
        :param operation: ``"departure"``/``"D"`` or ``"arrival"``/``"A"``.
        :param aerodrome: Fly the published procedural steps at this field
            through the Appendix B performance model instead of reading the
            tabulated fixed-point trajectory. Most ANP types publish only
            steps, so for them this is not an alternative but the only way in.
        :param stage_length: ANP stage length (default 1).
        :param metric: ``"exposure"`` (SEL) or ``"maximum"`` (LAmax).
        :param temperature: Air temperature at the field, in °C, for the
            atmospheric impedance adjustment. Left unset it follows
            *aerodrome*, or the standard atmosphere when there is none.
        :param pressure: Air pressure at the field, in kPa, likewise.
        :return: A :class:`~phonometry.aircraft.airport_noise.FlyoverResult`.
        """
        acft, prof, p, d, sel, lmax = self._doc29_inputs(
            aircraft_id, operation, stage_length, aerodrome
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
            atmosphere=_impedance_atmosphere(aerodrome, temperature, pressure),
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
        aerodrome: Aerodrome | None = None,
        stage_length: int | str = 1,
        metric: EventMetric = "exposure",
        temperature: float | None = None,
        pressure: float | None = None,
    ) -> NoiseContourResult:
        """Doc 29 single-event ground contour of an ANP aircraft.

        Feeds the aircraft's profile and NPD curves into
        :func:`phonometry.aircraft.airport_noise.noise_contour`.

        :param aircraft_id: ANP aircraft identifier.
        :param operation: ``"departure"``/``"D"`` or ``"arrival"``/``"A"``.
        :param x: Grid x coordinates (along-track), in metres.
        :param y: Grid y coordinates (lateral), in metres.
        :param aerodrome: Fly the published procedural steps at this field
            through the Appendix B performance model instead of reading the
            tabulated fixed-point trajectory. Most ANP types publish only
            steps, so for them this is not an alternative but the only way in.
        :param stage_length: ANP stage length (default 1).
        :param metric: ``"exposure"`` (SEL) or ``"maximum"`` (LAmax).
        :param temperature: Air temperature at the field, in °C, for the
            atmospheric impedance adjustment. Left unset it follows
            *aerodrome*, or the standard atmosphere when there is none.
        :param pressure: Air pressure at the field, in kPa, likewise.
        :return: A :class:`~phonometry.aircraft.airport_noise.NoiseContourResult`.
        """
        acft, prof, p, d, sel, lmax = self._doc29_inputs(
            aircraft_id, operation, stage_length, aerodrome
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
            atmosphere=_impedance_atmosphere(aerodrome, temperature, pressure),
            segments=FlightSegmentState(
                ground_roll=prof.ground_roll, landing_roll=prof.landing_roll
            ),
        )


def _impedance_atmosphere(
    aerodrome: Aerodrome | None, temperature: float | None, pressure: float | None
) -> AerodromeAtmosphere:
    """Conditions for the Doc 29 atmospheric impedance adjustment.

    Appendix B and the impedance adjustment of section 4 read the same weather
    at the same field: there is one air, and asking the caller to describe it
    twice invites a profile flown at 40 degrees C whose levels are corrected
    for 15. So an :class:`~phonometry.aircraft.flight_performance.Aerodrome`
    answers for both unless the caller overrides a value, and with no aerodrome
    the standard atmosphere stands as it always did.

    The pressure the adjustment wants is the one at the field, where the
    aerodrome carries the sea-level QNH, so Eq. B-4's ratio at the elevation is
    what converts between them.
    """
    if aerodrome is None:
        return AerodromeAtmosphere(
            15.0 if temperature is None else temperature,
            _STANDARD_PRESSURE_KPA if pressure is None else pressure,
        )
    at_field_kpa = (
        aerodrome.pressure_ratio(aerodrome.elevation_ft) * _STANDARD_PRESSURE_KPA
    )
    return AerodromeAtmosphere(
        aerodrome.temperature_c if temperature is None else temperature,
        at_field_kpa if pressure is None else pressure,
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
            # Normalised for the same reason as the aerodynamic table: the
            # lookup compares against an already-normalised code, so a row
            # spelling its operation "d" would simply never be found and the
            # aeroplane would report no fixed-point profile at all.
            _operation_code(row["Op Type"]),
            row["Profile_ID"],
            int(float(row[_COL_STAGE_LENGTH])),
        )
        point = int(float(row["Point Number"]))
        x = float(row[_COL_DISTANCE_FT]) * _FT_M
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


def _parse_performance(tables: Mapping[str, str]) -> _PerformanceTables:
    """Parse the five performance tables an export carries, skipping any absent.

    These are optional: an export limited to NPD curves and fixed-point
    trajectories is complete for the fixed-point bridge, and the absence is
    reported later, against the procedure that needed them, rather than as a
    load failure that stops an NPD-only export from being read at all.
    """
    out = _PerformanceTables()
    if (text := _optional_table("jet_engine", tables)) is not None:
        for row in _rows(text):
            out.jet.setdefault(row["ACFT_ID"], {})[row[_COL_THRUST_RATING]] = (
                JetEngineCoefficients(
                    e=float(row["E"]),
                    f=float(row["F"]),
                    ga=float(row["Ga"]),
                    gb=float(row["Gb"]),
                    h=float(row["H"]),
                )
            )
    if (text := _optional_table("propeller_engine", tables)) is not None:
        for row in _rows(text):
            out.propeller.setdefault(row["ACFT_ID"], {})[row[_COL_THRUST_RATING]] = (
                PropellerEngineCoefficients(
                    efficiency=float(row["Propeller Efficiency"]),
                    power_hp=float(row["Installed Net Propulsive Power (hp)"]),
                )
            )
    if (text := _optional_table("aerodynamic", tables)) is not None:
        for row in _rows(text):
            # Normalised on read, not compared raw: the column below is chosen
            # by an exact match while PerformanceAircraft.flap folds case when
            # it looks the row up again, so a table spelling the operation "d"
            # would be stored with the landing coefficient and still be found
            # by a departure. Eq. B-15 would then rotate at the wrong speed
            # with nothing to show for it.
            op = _operation_code(row["Op Type"])
            # The take-off speed coefficient C and the landing one D live in
            # separate columns, and a row fills whichever its operation flies.
            speed = _optional(row["C"] if op == _DEPARTURE else row["D"])
            out.aerodynamic.setdefault(row["ACFT_ID"], {})[(op, row["Flap_ID"])] = (
                AerodynamicCoefficients(
                    drag_ratio=float(row["R"]),
                    ground_roll_coefficient=_optional(row["B"]),
                    speed_coefficient=speed,
                )
            )
    if (text := _optional_table("weights", tables)) is not None:
        for row in _rows(text):
            out.weights[(row["ACFT_ID"], _stage_label(row[_COL_STAGE_LENGTH]))] = float(
                row["Weight (lb)"]
            )
    if (text := _optional_table("departure_procedural", tables)) is not None:
        grouped: dict[tuple[str, str, str], list[tuple[int, DepartureStep]]] = {}
        for row in _rows(text):
            dkey = (
                row["ACFT_ID"],
                row["Profile_ID"],
                _stage_label(row[_COL_STAGE_LENGTH]),
            )
            step = DepartureStep(
                step_type=row["Step Type"],
                thrust_rating=row[_COL_THRUST_RATING],
                flap_id=row["Flap_ID"],
                end_altitude_ft=_optional(row["End Point Altitude (ft)"]),
                rate_of_climb_ft_per_min=_optional(row["Rate Of Climb (ft/min)"]),
                end_calibrated_airspeed_kt=_optional(row["End Point CAS (kt)"]),
                energy_share_percent=_optional(row["Accel Percentage (%)"]),
                distance_ft=_optional(row.get(_COL_DISTANCE_FT, "")),
            )
            grouped.setdefault(dkey, []).append((int(float(row["Step Number"])), step))
        for dkey, steps in grouped.items():
            steps.sort(key=lambda e: e[0])
            out.departure_steps[dkey] = tuple(step for _n, step in steps)
    if (text := _optional_table("approach_procedural", tables)) is not None:
        by_profile: dict[tuple[str, str], list[tuple[int, ApproachStep]]] = {}
        for row in _rows(text):
            akey = (row["ACFT_ID"], row["Profile_ID"])
            astep = ApproachStep(
                step_type=row["Step Type"],
                flap_id=row["Flap_ID"],
                start_altitude_ft=_optional(row["Start Altitude(ft)"]),
                start_calibrated_airspeed_kt=_optional(row["Start CAS (kt)"]),
                descent_angle_deg=_optional(row["Descent Angle (deg)"]),
                touchdown_roll_ft=_optional(row["Touchdown Roll (ft)"]),
                distance_ft=_optional(row[_COL_DISTANCE_FT]),
                start_thrust_percent=_optional(row["Start Thrust"]),
            )
            by_profile.setdefault(akey, []).append(
                (int(float(row["Step Number"])), astep)
            )
        for akey, asteps in by_profile.items():
            asteps.sort(key=lambda e: e[0])
            out.approach_steps[akey] = tuple(step for _n, step in asteps)
    return out


def _optional_table(name: str, tables: Mapping[str, str]) -> str | None:
    """One logical table by filename keyword, or ``None`` when the export omits it."""
    try:
        return _pick(name, tables)
    except FileNotFoundError:
        return None


def load_anp_database(path: Path | str | None = None) -> AnpDatabase:
    """Load an EASA ANP database (aircraft, NPD curves and default profiles).

    :param path: Directory of an ANP CSV export (the ``*Aircraft.csv``,
        ``*NPD_data.csv``, ``*fixed_point_profiles.csv`` tables, plus the
        optional performance tables the procedural-step model reads:
        ``*engine_coefficients.csv``, ``*Aerodynamic_coefficients.csv``,
        ``*weights.csv`` and the two ``*procedural_steps.csv``). If ``None``
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
        aircraft=aircraft,
        npd=npd,
        distances=distances,
        profiles=profiles,
        performance=_parse_performance(tables),
    )
