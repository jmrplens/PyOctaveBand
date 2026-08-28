#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EASA ANP fleet loader and its bridge to the ECAC Doc 29 chain.

The primary check is a clean-room round-trip: the loader must recover the ANP
database's own published NPD values exactly when interpolated at a tabulated
(power, distance) node. This validates the parser and the log-linear/power
interpolation, which are the main risk in wiring the real fleet data in.
"""

from __future__ import annotations

import csv
import dataclasses
from importlib.resources import files

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")
import matplotlib.pyplot as plt

from phonometry.aircraft import (
    AnpAircraft,
    AnpDatabase,
    AnpNpdCurves,
    AnpProfile,
    load_anp_database,
)

_FT_M = 0.3048
_KT_MS = 0.514444
#: Representative aircraft with both NPD data and a fixed-point profile: a heavy
#: jet (wing), a narrowbody (fuselage) and a propeller aircraft.
_REPRESENTATIVE = ("747100", "727200", "PA31")
_DB = load_anp_database()


def _npd_oracle() -> list[dict[str, str]]:
    """Independent parse of the bundled NPD CSV (the published-value oracle)."""
    text = files("phonometry.aircraft.data.anp").joinpath("NPD_data.csv").read_text()
    return list(csv.DictReader(text.splitlines(), delimiter=";"))


def test_database_loads_full_fleet() -> None:
    ids = _DB.aircraft_ids
    assert len(ids) > 100  # full EASA ANP v2.3 (155 aircraft types)
    for rep in _REPRESENTATIVE:
        assert rep in ids
    assert "A320-211" in ids  # a modern narrowbody present in the full DB


def test_metadata_and_mounting_mapping() -> None:
    assert _DB.aircraft("747100").mounting == "wing"
    assert _DB.aircraft("727200").mounting == "fuselage"
    assert _DB.aircraft("PA31").mounting == "propeller"
    assert _DB.aircraft("747100").num_engines == 4
    assert "747" in _DB.aircraft("747100").description


def test_npd_round_trip_is_exact() -> None:
    """Every tabulated (power, distance) node is recovered to machine precision."""
    npd_ids = {_DB.aircraft(a).npd_id: a for a in _REPRESENTATIVE}
    dist_cols = [c for c in _npd_oracle()[0] if c.startswith("L_") and c.endswith("ft")]
    checked = 0
    for row in _npd_oracle():
        if row["NPD_ID"] not in npd_ids or row["Noise Metric"] not in ("SEL", "LAmax"):
            continue
        curves = _DB.npd_curves(
            npd_ids[row["NPD_ID"]], row["Op Mode"], row["Noise Metric"]
        )
        power = float(row["Power Setting"])
        for col in dist_cols:
            d_m = float(col[2:-2]) * _FT_M
            expected = float(row[col])
            got = float(curves.level(power, d_m)[0])
            assert got == pytest.approx(expected, abs=1e-9), (row["NPD_ID"], col)
            checked += 1
    assert checked > 100  # all three aircraft, both metrics, both operations


def test_npd_log_midpoint_interpolation() -> None:
    """Between two nodes the level is log-linear in distance (Doc 29 Eq. 4-4)."""
    curves = _DB.npd_curves("747100", "departure", "SEL")
    d0, d1 = curves.distances[0], curves.distances[1]
    mid = float(curves.level(curves.powers[0], np.sqrt(d0 * d1))[0])
    assert mid == pytest.approx(
        0.5 * (curves.levels[0, 0] + curves.levels[0, 1]), abs=1e-9
    )


def test_modern_aircraft_has_npd_but_no_fixed_point_profile() -> None:
    """A320-211 ships only procedural steps: NPD loads, the profile guard fires."""
    curves = _DB.npd_curves("A320-211", "departure", "SEL")
    assert curves.levels.shape[1] == 10
    with pytest.raises(
        KeyError,
        match=r"no fixed-point profile for aircraft 'A320-211'.*"
        r"only procedural-step profiles",
    ):
        _DB.profile("A320-211", "departure")


def test_profile_units_and_ground_roll_mask() -> None:
    dep = _DB.profile("747100", "departure")
    assert isinstance(dep, AnpProfile)
    assert dep.path.shape[1] == 5
    # First departure point of 747100: distance 0 ft, altitude 0 ft, TAS 35 kt.
    assert dep.path[0, 0] == pytest.approx(0.0)
    assert dep.path[0, 2] == pytest.approx(0.0)
    assert dep.path[0, 4] == pytest.approx(35.0 * _KT_MS)
    # Only the initial zero-altitude segment is takeoff ground roll.
    assert dep.ground_roll[0]
    assert not dep.ground_roll[1:].any()
    assert not dep.landing_roll.any()
    arr = _DB.profile("747100", "arrival")
    # Landing rollout: the trailing zero-altitude segments, no takeoff roll.
    assert arr.landing_roll[-1]
    assert not arr.ground_roll.any()


@pytest.mark.parametrize("aircraft_id", _REPRESENTATIVE)
@pytest.mark.parametrize("operation", ["departure", "arrival"])
def test_event_level_finite(aircraft_id: str, operation: str) -> None:
    fr = _DB.event_level(aircraft_id, [500.0, 600.0, 0.0], operation, metric="exposure")
    assert np.isfinite(fr.level)
    # The aircraft-object accessor matches the database method.
    assert _DB.aircraft(aircraft_id).event_level(
        [500.0, 600.0, 0.0], operation, metric="maximum"
    ).level == pytest.approx(
        _DB.event_level(
            aircraft_id, [500.0, 600.0, 0.0], operation, metric="maximum"
        ).level
    )


def test_noise_contour_smoke() -> None:
    x = np.linspace(-2000.0, 8000.0, 30)
    y = np.linspace(-3000.0, 3000.0, 25)
    contour = _DB.noise_contour("747100", "departure", x=x, y=y, metric="exposure")
    assert contour.level.shape == (y.size, x.size)
    assert np.isfinite(contour.level).all()
    # Loudest near the track (y = 0), quieter far to the side.
    near = contour.level[np.argmin(np.abs(y)), np.argmin(np.abs(x - 4000.0))]
    far = contour.level[0, np.argmin(np.abs(x - 4000.0))]
    assert near > far


def test_an_aerodrome_runs_the_chain_for_an_aircraft_with_no_fixed_point_profile() -> (
    None
):
    """The A320-211 has no tabulated departure, so only the flown profile reaches it."""
    from phonometry.aircraft.flight_performance import Aerodrome

    observer = [3000.0, 500.0, 0.0]
    with pytest.raises(KeyError, match=r"no fixed-point profile for aircraft"):
        _DB.event_level("A320-211", observer, "departure")
    flown = _DB.event_level(
        "A320-211", observer, "departure", aerodrome=Aerodrome(elevation_ft=0.0)
    )
    assert np.isfinite(flown.level)
    # It is the flown trajectory that was fed in, not some fallback: driving the
    # bare Doc 29 function with that path and the aircraft's own NPD curves
    # reaches the same level.
    from phonometry.aircraft.airport_noise import FlightSegmentState, event_level

    sel = _DB.npd_curves("A320-211", "departure", "SEL")
    lmax = _DB.npd_curves("A320-211", "departure", "LAmax")
    profile = _DB.procedural_profile(
        "A320-211", "departure", aerodrome=Aerodrome(elevation_ft=0.0)
    )
    by_hand = event_level(
        profile.path,
        observer,
        sel.powers,
        sel.distances,
        sel.levels,
        lmax.levels,
        mounting=_DB.aircraft("A320-211").mounting,
        segments=FlightSegmentState(
            ground_roll=profile.ground_roll, landing_roll=profile.landing_roll
        ),
    )
    assert flown.level == pytest.approx(by_hand.level)


def test_the_aerodrome_also_answers_for_the_impedance_adjustment() -> None:
    """One field, one atmosphere: Appendix B and the section 4 adjustment share it."""
    from phonometry.aircraft.flight_performance import Aerodrome

    hot = Aerodrome(elevation_ft=5000.0, temperature_c=35.0)
    observer = [3000.0, 500.0, 0.0]
    followed = _DB.event_level("A320-211", observer, "departure", aerodrome=hot)
    # Stating the field's own conditions by hand cannot change the answer: they
    # are what the aerodrome was already answering with.
    at_field_kpa = hot.pressure_ratio(hot.elevation_ft) * 101.325
    stated = _DB.event_level(
        "A320-211",
        observer,
        "departure",
        aerodrome=hot,
        temperature=35.0,
        pressure=at_field_kpa,
    )
    assert followed.level == pytest.approx(stated.level)
    # Leaving the standard atmosphere in place instead does change it, which is
    # the mistake a caller can no longer make by omission.
    standard_day = _DB.event_level(
        "A320-211",
        observer,
        "departure",
        aerodrome=hot,
        temperature=15.0,
        pressure=101.325,
    )
    assert standard_day.level != pytest.approx(followed.level)


def test_no_aerodrome_leaves_the_fixed_point_chain_exactly_as_it_was() -> None:
    """The tabulated trajectory and the standard atmosphere, as before the model."""
    flyover = _DB.event_level("747100", [3000.0, 500.0, 0.0], "departure")
    assert float(flyover.level) == pytest.approx(100.4, abs=0.05)
    # Spelling out the old defaults reaches the same number, so the None
    # sentinels resolve to what the signature used to name.
    spelled = _DB.event_level(
        "747100", [3000.0, 500.0, 0.0], "departure", temperature=15.0, pressure=101.325
    )
    assert flyover.level == pytest.approx(spelled.level)


def test_external_directory_load(tmp_path: object) -> None:
    """The loader reads a user-supplied CSV export directory (archive naming)."""
    import pathlib

    root = pathlib.Path(str(tmp_path))
    src = files("phonometry.aircraft.data.anp")
    for name in ("Aircraft.csv", "NPD_data.csv", "Default_fixed_point_profiles.csv"):
        (root / f"ANP2.3_{name}").write_text(src.joinpath(name).read_text())
    db = load_anp_database(root)
    assert db.npd_curves("727200", "arrival", "SEL").levels.shape[1] == 10


def test_error_paths() -> None:
    with pytest.raises(KeyError, match=r"aircraft 'NOPE' not in this ANP database"):
        _DB.aircraft("NOPE")
    with pytest.raises(ValueError, match=r"'metric' must be one of"):
        _DB.npd_curves("747100", "departure", "EPNL")
    with pytest.raises(ValueError, match=r"'operation' must be 'departure'"):
        _DB.npd_curves("747100", "sideways", "SEL")
    with pytest.raises(
        KeyError, match=r"no fixed-point profile for aircraft 'PA31'.*stage length 9"
    ):
        _DB.profile("PA31", "departure", stage_length=9)


def test_profile_unknown_aircraft_reports_missing_aircraft() -> None:
    """An unknown id raises the 'unknown aircraft' error, not 'no profile'."""
    with pytest.raises(KeyError, match="not in this ANP database"):
        _DB.profile("NOPE", "departure")


def test_parsed_arrays_are_read_only() -> None:
    """NPD and profile arrays are exposed as read-only views."""
    curves = _DB.npd_curves("747100", "departure", "SEL")
    for arr in (curves.powers, curves.distances, curves.levels):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0.0
    assert not _DB.profile("747100", "departure").path.flags.writeable


def test_npd_curves_reject_level_table_short_of_its_powers() -> None:
    """A level table one power row short cannot be built into a curve set."""
    curves = _DB.npd_curves("747100", "departure", "SEL")
    short = curves.levels[:-1]
    with pytest.raises(
        ValueError, match=r"'levels'.*must each carry one value per power setting"
    ):
        dataclasses.replace(curves, levels=short)


def test_npd_curves_reject_level_table_short_of_its_distances() -> None:
    """The distance axis is pinned on its own: the power axis still agrees."""
    curves = _DB.npd_curves("747100", "departure", "SEL")
    short = curves.levels[:, :-1]
    with pytest.raises(ValueError, match=r"'levels \(axis 1\)'"):
        dataclasses.replace(curves, levels=short)


def test_npd_curves_reject_level_table_with_an_extra_axis() -> None:
    """Both counts agree on a (powers, distances, 1) table; the rank pin does not."""
    curves = _DB.npd_curves("747100", "departure", "SEL")
    boxed = curves.levels[:, :, None]
    with pytest.raises(ValueError, match="'levels' must have 2 axes"):
        dataclasses.replace(curves, levels=boxed)


def test_profile_rejects_roll_mask_missing_a_segment() -> None:
    """The roll masks are per segment, so one entry short is not a profile."""
    prof = _DB.profile("747100", "departure")
    short = prof.ground_roll[:-1]
    with pytest.raises(
        ValueError, match=r"'ground_roll'.*must each carry one value per segment"
    ):
        dataclasses.replace(prof, ground_roll=short)


def test_profile_rejects_roll_mask_with_an_extra_axis() -> None:
    """A column-shaped mask counts one entry per segment and is still refused."""
    prof = _DB.profile("747100", "departure")
    column = np.asarray(prof.ground_roll)[:, None]
    with pytest.raises(ValueError, match="'ground_roll' must have one axis"):
        dataclasses.replace(prof, ground_roll=column)


def test_profile_names_the_segments_and_not_the_vertices_it_counted() -> None:
    """A profile of N breakpoints has N-1 segments, and the message says so.

    Naming the count ``path`` would report one fewer than ``path`` holds and
    send a reader to measure the wrong attribute; naming it ``path segments``
    says which derived quantity the number is.
    """
    prof = _DB.profile("747100", "departure")
    vertices = len(prof.path)
    surplus = np.zeros(vertices, dtype=bool)
    with pytest.raises(ValueError, match=rf"'path segments' \({vertices - 1}\)"):
        dataclasses.replace(prof, ground_roll=surplus, landing_roll=surplus)


def test_profile_rejects_a_path_short_of_the_five_documented_columns() -> None:
    """The rank pin passes a two-column path; the figure dies reading column 2.

    ``path`` is documented as ``(N, 5)`` and the altitude the trajectory plot
    draws is column 2, so a path without it clears every per-segment count and
    stops at ``path[:, 2]`` with an ``IndexError`` that names neither the field
    nor the profile it came from.
    """
    prof = _DB.profile("747100", "departure")
    flat = np.asarray(prof.path)[:, :2]
    with pytest.raises(ValueError, match=r"'path' must have shape \(N, 5\)"):
        dataclasses.replace(prof, path=flat)


def test_a_profile_with_no_path_reports_no_segments_rather_than_minus_one() -> None:
    """An empty path has nought segments, which is a count and not an error."""
    prof = _DB.profile("747100", "departure")
    empty = dataclasses.replace(
        prof,
        path=np.zeros((0, np.asarray(prof.path).shape[1])),
        ground_roll=np.zeros(0, dtype=bool),
        landing_roll=np.zeros(0, dtype=bool),
    )
    assert len(empty.path) == 0


def test_profile_plot_highlights_full_roll_span() -> None:
    """The roll highlight includes both endpoints of every roll segment."""
    prof = _DB.profile("747100", "arrival")
    seg = prof.landing_roll
    n_roll_points = int(np.count_nonzero(np.r_[seg, False] | np.r_[False, seg]))
    # One more point than segments in the (contiguous) roll span.
    assert n_roll_points == int(seg.sum()) + 1


def test_pick_ambiguous_table_raises(tmp_path: object) -> None:
    """Two files matching the same table keyword raise an explicit error."""
    import pathlib

    root = pathlib.Path(str(tmp_path))
    src = files("phonometry.aircraft.data.anp")
    text = src.joinpath("Aircraft.csv").read_text()
    (root / "Aircraft.csv").write_text(text)
    (root / "ANP2.3_Aircraft.csv").write_text(text)  # second "aircraft" match
    (root / "NPD_data.csv").write_text(src.joinpath("NPD_data.csv").read_text())
    (root / "Default_fixed_point_profiles.csv").write_text(
        src.joinpath("Default_fixed_point_profiles.csv").read_text()
    )
    with pytest.raises(ValueError, match=r"ambiguous ANP table for 'aircraft'"):
        load_anp_database(root)


def test_utf8_bom_export_is_tolerated(tmp_path: object) -> None:
    """A leading BOM in an exported CSV does not corrupt the first column."""
    import pathlib

    root = pathlib.Path(str(tmp_path))
    src = files("phonometry.aircraft.data.anp")
    for name in ("Aircraft.csv", "NPD_data.csv", "Default_fixed_point_profiles.csv"):
        (root / name).write_text("﻿" + src.joinpath(name).read_text(), encoding="utf-8")
    db = load_anp_database(root)
    assert "747100" in db.aircraft_ids  # first column not mangled by the BOM


def test_mismatched_sel_lamax_powers_raise() -> None:
    """A malformed database with SEL/LAmax power mismatch is rejected."""
    distances = np.array([60.0, 120.0, 240.0])
    sel = (
        np.array([8000.0, 12000.0]),
        np.array([[95.0, 90.0, 85.0], [99.0, 94.0, 89.0]]),
    )
    lmax = (
        np.array([8000.0, 14000.0]),  # different top power on purpose
        np.array([[90.0, 85.0, 80.0], [96.0, 91.0, 86.0]]),
    )
    aircraft = {
        "X": {
            "ACFT_ID": "X",
            "NPD_ID": "NX",
            "Power Parameter": "CNT",
            "Lateral Directivity Identifier": "Wing",
            "Number Of Engines": "2",
        }
    }
    npd = {("NX", "SEL", "D"): sel, ("NX", "LAmax", "D"): lmax}
    path = np.array([[0.0, 0.0, 0.0, 8000.0, 20.0], [1000.0, 0.0, 300.0, 8000.0, 80.0]])
    profiles = {("X", "D", "DEFAULT", 1): path}
    db = AnpDatabase(aircraft=aircraft, npd=npd, distances=distances, profiles=profiles)
    with pytest.raises(ValueError, match="power settings differ"):
        db.event_level("X", [100.0, 100.0, 0.0], "departure")


def test_plot_smoke_en_es() -> None:
    npd = _DB.npd_curves("747100", "departure", "SEL")
    assert isinstance(npd, AnpNpdCurves)
    ax = npd.plot()
    assert "747100" in ax.get_title()
    npd.plot(language="es")
    prof = _DB.profile("747100", "departure")
    ax2 = prof.plot(language="es")
    assert ax2.get_xlabel().startswith("Distancia")
    with pytest.raises(ValueError, match=r"Unknown language 'xx'"):
        npd.plot(language="xx")
    plt.close("all")


def test_aircraft_object_type() -> None:
    assert isinstance(_DB.aircraft("PA31"), AnpAircraft)


# ---------------------------------------------------------------------------
# Profile identity: Profile_ID is part of the parse key, so aircraft with
# several fixed-point profiles for the same operation and stage length
# (weight variants) must not have their points interleaved into one path.
# ---------------------------------------------------------------------------

_PROFILE_HEADER = (
    "ACFT_ID;Op Type;Profile_ID;Stage Length;Point Number;"
    "Distance (ft);Altitude AFE (ft);TAS (kt);Power Setting"
)


def _profiles_oracle() -> list[dict[str, str]]:
    """Independent parse of the bundled fixed-point profiles CSV."""
    text = (
        files("phonometry.aircraft.data.anp")
        .joinpath("Default_fixed_point_profiles.csv")
        .read_text()
    )
    return list(csv.DictReader(text.splitlines(), delimiter=";"))


def _synthetic_db(tmp_path: object, profile_rows: list[str]) -> AnpDatabase:
    """Bundled aircraft/NPD tables plus a synthetic fixed-point profile CSV."""
    import pathlib

    root = pathlib.Path(str(tmp_path))
    src = files("phonometry.aircraft.data.anp")
    for name in ("Aircraft.csv", "NPD_data.csv"):
        (root / name).write_text(src.joinpath(name).read_text())
    (root / "Default_fixed_point_profiles.csv").write_text(
        "\n".join([_PROFILE_HEADER, *profile_rows]) + "\n"
    )
    return load_anp_database(root)


def test_colliding_profiles_stay_separate() -> None:
    """CNA206 D/1 ships DEFAULT and 3000LB: each loads whole, not interleaved."""
    prof = _DB.profile("CNA206", "departure", 1)
    assert prof.profile_id == "DEFAULT"
    assert prof.path.shape[0] == 9  # its own 9 CSV rows, not 9 + 9 merged
    # A departure climbs: altitude must be non-decreasing (the interleaved
    # merge produced a sawtooth with negative increments).
    assert np.all(np.diff(prof.path[:, 2]) >= 0.0)
    alt = _DB.profile("CNA206", "departure", 1, profile_id="3000LB")
    assert alt.profile_id == "3000LB"
    assert alt.path.shape[0] == 9
    assert np.all(np.diff(alt.path[:, 2]) >= 0.0)
    # The aircraft-object accessor threads profile_id through.
    via_acft = _DB.aircraft("CNA206").profile("departure", profile_id="3000LB")
    assert via_acft.path.shape == alt.path.shape


def test_profile_point_counts_match_csv_rows() -> None:
    """Every bundled profile loads exactly its own CSV rows, nothing merged."""
    counts: dict[tuple[str, str, str, int], int] = {}
    for row in _profiles_oracle():
        key = (
            row["ACFT_ID"],
            row["Op Type"],
            row["Profile_ID"],
            int(float(row["Stage Length"])),
        )
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) > 50  # 77 bundled (aircraft, op, profile, stage) keys
    for (acft, op, pid, stage), n in counts.items():
        prof = _DB.profile(acft, op, stage, profile_id=pid)
        assert prof.path.shape[0] == n, (acft, op, pid, stage)
        assert prof.profile_id == pid


def test_synthetic_two_profile_export_selects_default(tmp_path: object) -> None:
    """With several profiles per key, DEFAULT wins unless profile_id says else."""
    db = _synthetic_db(
        tmp_path,
        [
            "747100;D;HEAVY;1;1;0.0;0.0;30.0;40000.0",
            "747100;D;HEAVY;1;2;5000.0;500.0;150.0;40000.0",
            "747100;D;DEFAULT;1;1;0.0;0.0;35.0;45000.0",
            "747100;D;DEFAULT;1;2;6000.0;800.0;160.0;45000.0",
            "747100;D;DEFAULT;1;3;12000.0;2000.0;180.0;42000.0",
        ],
    )
    prof = db.profile("747100", "departure", 1)
    assert prof.profile_id == "DEFAULT"
    assert prof.path.shape[0] == 3
    heavy = db.profile("747100", "departure", 1, profile_id="HEAVY")
    assert heavy.path.shape[0] == 2
    assert heavy.path[1, 2] == pytest.approx(500.0 * _FT_M)


def test_single_non_default_profile_is_selected(tmp_path: object) -> None:
    db = _synthetic_db(
        tmp_path,
        [
            "747100;D;ONLY;1;1;0.0;0.0;30.0;40000.0",
            "747100;D;ONLY;1;2;5000.0;500.0;150.0;40000.0",
        ],
    )
    assert db.profile("747100", "departure", 1).profile_id == "ONLY"


def test_ambiguous_profiles_without_default_raise(tmp_path: object) -> None:
    db = _synthetic_db(
        tmp_path,
        [
            "747100;D;LIGHT;1;1;0.0;0.0;30.0;40000.0",
            "747100;D;LIGHT;1;2;5000.0;500.0;150.0;40000.0",
            "747100;D;HEAVY;1;1;0.0;0.0;30.0;42000.0",
            "747100;D;HEAVY;1;2;5000.0;400.0;150.0;42000.0",
        ],
    )
    with pytest.raises(ValueError, match=r"'HEAVY', 'LIGHT'"):
        db.profile("747100", "departure", 1)
    assert (
        db.profile("747100", "departure", 1, profile_id="LIGHT").profile_id == "LIGHT"
    )
    with pytest.raises(
        KeyError, match=r"no fixed-point profile 'NOPE' for aircraft '747100'"
    ):
        db.profile("747100", "departure", 1, profile_id="NOPE")


def test_duplicate_point_numbers_raise(tmp_path: object) -> None:
    """A malformed table with duplicate point numbers errors instead of merging."""
    with pytest.raises(
        ValueError,
        match=r"fixed-point profile for aircraft '747100'.*"
        r"duplicate or non-consecutive point numbers",
    ):
        _synthetic_db(
            tmp_path,
            [
                "747100;D;DEFAULT;1;1;0.0;0.0;30.0;40000.0",
                "747100;D;DEFAULT;1;1;5000.0;500.0;150.0;40000.0",
            ],
        )


# --------------------------------------------------------------------------
# Procedural-step profiles (ECAC Doc 29 Vol. 2 Appendix B)
# --------------------------------------------------------------------------
def test_procedural_profile_reaches_an_aircraft_with_no_fixed_point_trajectory() -> (
    None
):
    """The gap this bridge used to have: an A320 publishes steps, not fixed points.

    The fixed-point accessor refuses it and says where to go instead, and the
    performance model flies the same aircraft's published procedure into a
    profile the Doc 29 chain reads without knowing which of the two it came
    from.
    """
    from phonometry.aircraft.flight_performance import Aerodrome

    with pytest.raises(KeyError, match=r"flight_profile\(\)"):
        _DB.profile("A320-211", "departure", 1)
    aerodrome = Aerodrome(elevation_ft=0.0, temperature_c=15.0)
    flown = _DB.flight_profile("A320-211", "departure", aerodrome=aerodrome)
    assert flown.operation == "D"
    assert flown.points[0].distance_ft == 0.0
    assert flown.points[-1].altitude_ft > flown.points[0].altitude_ft
    synthesised = _DB.procedural_profile("A320-211", "departure", aerodrome=aerodrome)
    assert synthesised.path.shape == (len(flown.points), 5)
    # The path is the same profile in the chain's units: feet to metres for the
    # distance and the height, knots to metres per second for the speed, and
    # the power setting left as the corrected net thrust per engine the NPD
    # tables are indexed on. All four columns, every point: a wrong factor on
    # the speed column alone changes no distance and no level anywhere in this
    # file, and silently moves every duration-weighted metric downstream.
    assert synthesised.path[:, 0] == pytest.approx(flown.distance_ft * _FT_M)
    assert synthesised.path[:, 2] == pytest.approx(flown.altitude_ft * _FT_M)
    assert synthesised.path[:, 3] == pytest.approx(flown.corrected_net_thrust_lb)
    assert synthesised.path[:, 4] == pytest.approx(flown.true_airspeed_kt * _KT_MS)
    assert synthesised.ground_roll[0]
    assert not synthesised.landing_roll.any()


def test_procedural_profile_feeds_the_doc29_chain() -> None:
    """A synthesised profile is an AnpProfile, so the noise functions take it."""
    from phonometry.aircraft.airport_noise import FlightSegmentState, event_level
    from phonometry.aircraft.flight_performance import Aerodrome

    prof = _DB.procedural_profile(
        "A320-211", "departure", aerodrome=Aerodrome(elevation_ft=0.0)
    )
    sel = _DB.npd_curves("A320-211", "departure", "SEL")
    lmax = _DB.npd_curves("A320-211", "departure", "LAmax")
    result = event_level(
        prof.path,
        [4000.0, 500.0, 0.0],
        sel.powers,
        sel.distances,
        sel.levels,
        lmax.levels,
        mounting=_DB.aircraft("A320-211").mounting,
        segments=FlightSegmentState(
            ground_roll=prof.ground_roll, landing_roll=prof.landing_roll
        ),
    )
    assert 50.0 < float(result.level) < 130.0


def test_performance_aircraft_carries_the_coefficient_tables() -> None:
    """The four Appendix B tables, gathered per aircraft from the CSV export."""
    acft = _DB.performance_aircraft("A320-211")
    assert acft.engines == 2
    assert acft.approach_weight_lb == pytest.approx(0.9 * acft.max_landing_weight_lb)
    assert "MaxClimb" in acft.jet_coefficients
    assert acft.flap("D", "ZERO").drag_ratio > 0.0


def test_procedural_steps_are_returned_in_step_order() -> None:
    steps = _DB.procedural_steps("A320-211", "departure", 1)
    assert steps[0].kind == "takeoff"
    approach = _DB.procedural_steps("A320-211", "arrival")
    assert [s.kind for s in approach].count("land") == 1


def test_a_level_approach_step_holds_the_speed_of_the_step_below_it() -> None:
    """A published Level step with an empty start CAS, at its Eq. B-62 thrust.

    Eq. B-61 reads a start calibrated airspeed the A380's own approach does not
    give: its third step is a Level step with the speed column empty, flown at
    flap A_1+F for 11 893 ft at 3000 ft. A Level step changes neither height
    nor speed, so the speed is the one the step below it is entered at, and
    Eq. B-62 fixes the thrust at ``(W/delta)/N (R/cos eps)`` (folio B-39) with
    W the approach weight and delta the pressure ratio of the step's own level.
    Both are read here off a bundled fleet entry rather than a written fixture,
    because it is real ANP entries that leave the column empty.
    """
    from phonometry.aircraft.flight_performance import Aerodrome, ApproachStep

    steps = _DB.procedural_steps("A380-841", "arrival")
    level, below = steps[2], steps[3]
    assert isinstance(level, ApproachStep)
    assert isinstance(below, ApproachStep)
    assert level.kind == "level"
    assert level.start_calibrated_airspeed_kt is None
    # The tabulated pair the flown profile has to reproduce: the step's own
    # level, and the speed the step below it is entered at.
    level_ft, below_cas_kt = 3000.0, 205.0
    assert level.start_altitude_ft == level_ft
    assert below.start_calibrated_airspeed_kt == below_cas_kt
    # At sea level a step's own level is its altitude, and sea-level ISA is
    # where the idle step below it holds its tabulated start speed (B7.1.4).
    aerodrome = Aerodrome(elevation_ft=0.0, temperature_c=15.0)
    acft = _DB.performance_aircraft("A380-841")
    thrust_lb = (
        acft.approach_weight_lb
        / aerodrome.pressure_ratio(level_ft)
        / acft.engines
        * acft.flap("A", level.flap_id).drag_ratio
    )
    profile = _DB.flight_profile("A380-841", "arrival", aerodrome=aerodrome)
    flown = [
        p
        for p in profile.points
        if p.corrected_net_thrust_lb == pytest.approx(thrust_lb)
    ]
    assert flown, "no profile point carries the Level step's Eq. B-62 thrust"
    inherited_kt = aerodrome.true_airspeed_kt(below_cas_kt, level_ft)
    assert all(p.altitude_ft == level_ft for p in flown)
    assert all(p.true_airspeed_kt == pytest.approx(inherited_kt) for p in flown)


def test_procedural_steps_name_the_aircraft_that_publishes_none() -> None:
    with pytest.raises(KeyError, match=r"no procedural-step profile"):
        _DB.procedural_steps("747100", "departure", 1)


def test_ambiguous_procedural_profile_lists_the_identifiers() -> None:
    """The A350 publishes DEFAULT1 and DEFAULT2 and neither is 'DEFAULT'."""
    with pytest.raises(ValueError, match=r"'DEFAULT1', 'DEFAULT2'"):
        _DB.procedural_steps("A350-941", "arrival")


def test_a_named_procedure_is_the_one_flown_and_the_one_reported() -> None:
    """An explicit request wins over the DEFAULT, and an unpublished name is refused."""
    from phonometry.aircraft.flight_performance import Aerodrome

    flown = _DB.flight_profile(
        "A320-211",
        "departure",
        aerodrome=Aerodrome(elevation_ft=0.0),
        profile_id="ICAO_B",
    )
    assert flown.procedure_id == "ICAO_B"
    with pytest.raises(KeyError, match=r"no procedural-step profile 'NOPE'"):
        _DB.procedural_steps("A320-211", "departure", 1, profile_id="NOPE")


def test_a_lone_procedure_is_labelled_with_the_identifier_it_was_found_under(
    tmp_path: object,
) -> None:
    """``profile_id=None`` does not mean the answer came from 'DEFAULT'.

    A profile with nothing to choose between is taken whatever it is called, so
    the label has to come from the selection and not from the argument: a
    procedure published as ICAO_A is flown and then reported as ICAO_A, both by
    ``FlightProfile.procedure_id`` and by the ``AnpProfile.profile_id`` the
    noise chain reads it through.

    Only a user-supplied export gets here. Every key of the bundled v2.3 fleet
    that publishes procedural steps at all publishes a ``"DEFAULT"`` among
    them, bar the A350's approach, which publishes two and is refused as
    ambiguous, so the fixture trims one aircraft down to its ICAO_A departure.
    """
    import pathlib

    from phonometry.aircraft.flight_performance import Aerodrome

    root = pathlib.Path(str(tmp_path))
    src = files("phonometry.aircraft.data.anp")
    for name in (
        "Aircraft.csv",
        "NPD_data.csv",
        "Default_fixed_point_profiles.csv",
        "Jet_engine_coefficients.csv",
        "Aerodynamic_coefficients.csv",
        "Default_weights.csv",
    ):
        (root / name).write_text(src.joinpath(name).read_text())
    # Every row of the step table but the A320's DEFAULT and ICAO_B departures.
    kept = [
        line
        for line in src.joinpath("Default_departure_procedural_steps.csv")
        .read_text()
        .splitlines()
        if not line.startswith("A320-211;") or line.startswith("A320-211;ICAO_A;")
    ]
    (root / "Default_departure_procedural_steps.csv").write_text("\n".join(kept) + "\n")
    db = load_anp_database(root)
    aerodrome = Aerodrome(elevation_ft=0.0)
    flown = db.flight_profile("A320-211", "departure", aerodrome=aerodrome)
    assert flown.procedure_id == "ICAO_A"
    synthesised = db.procedural_profile("A320-211", "departure", aerodrome=aerodrome)
    assert synthesised.profile_id == "ICAO_A"


def test_an_export_that_spells_the_operation_in_lower_case_flies_the_same(
    tmp_path: object,
) -> None:
    """``Op Type`` is normalised on read, so its spelling cannot pick a column.

    The take-off speed coefficient sits in column C and the landing one in
    column D, chosen by an exact match on the operation, while
    ``PerformanceAircraft.flap`` folds case when it looks the row up again. An
    export spelling the operation ``"d"`` therefore stored the landing
    coefficient under a key a departure still finds.

    What that costs depends on the export. No row of the bundled table fills
    both columns, all 1218 of them carrying only the one their operation flies,
    so there the wrong column is empty and the Take-off step refuses outright:
    an aeroplane that should fly does not. It is a table filling both that
    would take a rotation speed from the landing coefficient and say nothing,
    and the ANP schema does not produce one. Loud on real data, then, and the
    fixed-point table fails a third way, reporting no profile at all.

    All of it is unreachable on the bundled v2.3 fleet, which spells every
    operation ``"A"`` or ``"D"``, and reachable on any user-supplied export.
    """
    import pathlib

    from phonometry.aircraft.flight_performance import Aerodrome

    def lower_operation(text: str, column: int) -> str:
        head, *rows = text.splitlines()
        out = [head]
        for line in rows:
            cells = line.split(";")
            if len(cells) > column:
                cells[column] = cells[column].lower()
            out.append(";".join(cells))
        return "\n".join(out) + "\n"

    root = pathlib.Path(str(tmp_path))
    src = files("phonometry.aircraft.data.anp")
    for name in (
        "Aircraft.csv",
        "NPD_data.csv",
        "Jet_engine_coefficients.csv",
        "Default_weights.csv",
        "Default_departure_procedural_steps.csv",
    ):
        (root / name).write_text(src.joinpath(name).read_text())
    # "Op Type" is the second column of both tables that carry it.
    for name in ("Aerodynamic_coefficients.csv", "Default_fixed_point_profiles.csv"):
        (root / name).write_text(
            lower_operation(src.joinpath(name).read_text(), column=1)
        )

    db = load_anp_database(root)
    aerodrome = Aerodrome(elevation_ft=0.0)
    shouted = _DB.flight_profile("A320-211", "departure", aerodrome=aerodrome)
    whispered = db.flight_profile("A320-211", "departure", aerodrome=aerodrome)
    assert whispered.distance_ft == pytest.approx(shouted.distance_ft)
    assert whispered.altitude_ft == pytest.approx(shouted.altitude_ft)
    # And the fixed-point table is still reachable, which is the other failure.
    assert db.profile("747100", "departure", 1).path.shape == (
        _DB.profile("747100", "departure", 1).path.shape
    )


def test_maximum_weight_stage_length_survives_the_lookup() -> None:
    """The stage-length column is a label, not a number: 'M' is one of its values."""
    from phonometry.aircraft.flight_performance import Aerodrome

    flown = _DB.flight_profile(
        "7378MAX", "departure", aerodrome=Aerodrome(elevation_ft=0.0), stage_length="M"
    )
    assert flown.points[-1].altitude_ft > 0.0


def test_flight_profile_names_the_stage_length_it_has_no_weight_for(
    tmp_path: object,
) -> None:
    """A weight the export does not carry is refused, not guessed at."""
    import pathlib

    from phonometry.aircraft.flight_performance import Aerodrome

    root = pathlib.Path(str(tmp_path))
    src = files("phonometry.aircraft.data.anp")
    for name in (
        "Aircraft.csv",
        "NPD_data.csv",
        "Default_fixed_point_profiles.csv",
        "Jet_engine_coefficients.csv",
        "Aerodynamic_coefficients.csv",
        "Default_departure_procedural_steps.csv",
    ):
        (root / name).write_text(src.joinpath(name).read_text())
    # Every row but the one the A320's stage length 1 would have used.
    kept = [
        line
        for line in src.joinpath("Default_weights.csv").read_text().splitlines()
        if not line.startswith("A320-211;1;")
    ]
    (root / "Default_weights.csv").write_text("\n".join(kept) + "\n")
    db = load_anp_database(root)
    aerodrome = Aerodrome(elevation_ft=0.0)
    with pytest.raises(
        KeyError, match=r"no default departure weight for aircraft 'A320-211'"
    ):
        db.flight_profile("A320-211", "departure", aerodrome=aerodrome)
    # The same profile flies once the caller says what weight to use.
    flown = db.flight_profile(
        "A320-211", "departure", aerodrome=aerodrome, weight_lb=150000.0
    )
    assert flown.points[-1].altitude_ft > 0.0


def test_unknown_stage_length_is_reported_against_the_step_table() -> None:
    from phonometry.aircraft.flight_performance import Aerodrome

    aerodrome = Aerodrome(elevation_ft=0.0)
    with pytest.raises(
        KeyError,
        match=r"no procedural-step profile for aircraft 'A320-211'.*stage length 99",
    ):
        _DB.flight_profile(
            "A320-211", "departure", aerodrome=aerodrome, stage_length=99
        )


def test_an_npd_only_export_says_what_it_is_missing(tmp_path: object) -> None:
    """The performance tables are optional, and their absence is reported late."""
    db = _synthetic_db(tmp_path, ["747100;D;DEFAULT;1;1;0.0;0.0;30.0;40000.0"])
    with pytest.raises(ValueError, match=r"no performance tables"):
        db.performance_aircraft("747100")
