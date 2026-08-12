#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
End-to-end validation of the rotorcraft event chain against the NORAH2
reference implementation (EASA.2020.FC.06 prototype, V2.0.74 public release).

The prototype's ARP verification cases carry, per single-event operation, the
input flight path (``.inp``), the per-step history at one microphone
(``.his``: emission angles, retarded times, slant distances, hemisphere
selection, ``L_A`` and ``PNLT``) and the per-microphone metrics over the whole
grid (``.onl``: ``SEL``, ``LASmax``, ``PNLTM``, ``EPNL``). Those outputs are
the oracle for the flight-condition interpolation, the kinematics/retarded
time, :func:`rotorcraft_event_level` and :func:`rotorcraft_noise_contour`.

The verification subset consumed here (the R22 hemisphere set and the ARP
single-event cases below) is committed under ``tests/data/norah2/`` with its
provenance in the README, so the suite runs everywhere including CI and never
skips. The full public release is preferred when present, resolved by
``tests/oracle_data.py``: an extraction pointed at by ``NORAH2_DATA``, then
``tests/data-local/norah2/NORAH2_V2.0.74_public.zip``. Every case below reads
the same files from either source, so the assertions do not differ; the run
header names the copy actually used.

Case 1 (terrain screening) is not exercised: the frame of the public release's
DEM is not reconstructible (a global template match finds no consistent
placement against the printed outputs), so the terrain chain is anchored to
closed-form oracles in the unit suite instead of fitted to that case.

Measured deviations on which the tolerances rest (see the module docstring of
``phonometry.aircraft.rotorcraft_noise`` for the method):

- Emission angles reproduce to 0.01 deg (the prototype clamps its printed
  azimuth to the +-90 deg hemisphere span, so the comparison clamps too) and
  recorded times to 0.013 s, bounded at 0.02 s below (the prototype evaluates
  c from the temperature, 346.19 m/s, where Doc 32 5.1 states the constant
  346.1 m/s).
- Hard-ground events (Case 4, sigma 1e6, 0.2 m microphone; radar-track event)
  reproduce every step level to 0.08 dB out to 19 km and SEL/LASmax to
  0.03 dB.
- Soft-ground events (Case 2, sigma 2e5, 1.2 m microphone) reproduce LASmax to
  0.02 dB and SEL to 0.4 dB; individual far steps diverge by up to 4.9 dB
  because the prototype damps the coherent two-ray interference of guidance
  Eq. 30 towards the incoherent sum where the in-band comb phase is large.
  Neither ECAC Doc 32 nor the guidance contains such a coherence-loss term;
  this implementation follows the published Eq. 28-35. The divergence is
  bounded here so a regression cannot hide behind it.
- ``PNLTM`` reproduces to 0.09 dB (with the helicopter 50 Hz tone-correction
  start band of Annex 16 App. 2 4.3.1); the per-step ``PNLT`` and the Case-4
  ``EPNL`` carry up to ~1.3 dB because the prototype's perceived-noisiness
  policy for bands below the Annex 16 noy floor differs from the published
  Table A2-3 law (this implementation is validated against the ICAO ETM
  worked examples elsewhere).
"""

from __future__ import annotations

import pathlib
import re
import warnings
import zipfile

import matplotlib

matplotlib.use("Agg")
import numpy as np
import oracle_data
import pytest

from phonometry.aircraft.rotorcraft_noise import (
    FlightConditionInterpolation,
    RotorcraftAtmosphere,
    RotorcraftGround,
    RotorcraftHemisphere,
    RotorcraftTrackState,
    hemisphere_source_level,
    hover_derived_hemisphere,
    hover_ring_hemisphere,
    rotorcraft_event_level,
    rotorcraft_noise_contour,
)

# Resolution order (tests/oracle_data.py): NORAH2_DATA pointing at an existing
# extraction root, then the full archive in the gitignored tests/data-local/,
# then the committed verification subset (see tests/data/norah2/README.md),
# which is what CI runs.
_RELEASE = oracle_data.resolve(oracle_data.NORAH2_RELEASE)
_EXTRACT = oracle_data.DATA / "norah2" / "norah2_arp_extract.zip"

_PREFIXES = (
    "NORAH2_V2.0.74_public/Hemispheres/R22_",
    "NORAH2_V2.0.74_public/ARP/Case2/SE/R22_H1_DEP_STD1_NE",
    "NORAH2_V2.0.74_public/ARP/Case2/SE/R22_H1_APP_STD1_NE",
    "NORAH2_V2.0.74_public/ARP/Case2/SE/A600_H1_APP_STD1_NE",
    "NORAH2_V2.0.74_public/ARP/Case2/SE/A600_APP_RT1",
    "NORAH2_V2.0.74_public/ARP/Case3/SE/CH7_H1_APP_STD1_NE",
    "NORAH2_V2.0.74_public/ARP/Case3/SE/A600_H1_APP_STD2_NE",
    "NORAH2_V2.0.74_public/ARP/Case4/SE/R22_H1_APP_STD2_NE",
    "NORAH2_V2.0.74_public/ARP/Case4/SE/R22_DEP_RT2",
)


@pytest.fixture(scope="session")
def norah_root(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """The NORAH2 extraction root, from the best copy available here."""
    if _RELEASE.origin == "env":
        # NORAH2_DATA has always pointed at an already extracted release.
        assert _RELEASE.path is not None
        return _RELEASE.path
    source = (
        _RELEASE.path / oracle_data.NORAH2_ARCHIVE
        if _RELEASE.path is not None
        else _EXTRACT
    )
    root = tmp_path_factory.mktemp("norah2")
    with zipfile.ZipFile(source) as zf:
        members = [m for m in zf.namelist() if m.startswith(_PREFIXES)]
        assert members, "unexpected zip layout"
        zf.extractall(root, members=members)
    return root / "NORAH2_V2.0.74_public"


# --------------------------------------------------------------------------- #
# NORAH/HELENA format parsers (test-side plumbing, not library API)
# --------------------------------------------------------------------------- #


def _parse_hem(path: pathlib.Path) -> tuple[np.ndarray, ...]:
    """Azimuth, polar, band and level arrays of a .hem hemisphere file."""
    lines = path.read_text(errors="replace").splitlines()
    theta = phi = freqs = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("THETAOBSAC"):
            theta = np.array([float(v) for v in lines[i + 1].split()])
            i += 2
            continue
        if ln.startswith("PHIOBSAC") and "=" not in ln:
            phi = np.array([float(v) for v in lines[i + 1].split()])
            i += 2
            continue
        if ln.startswith("NFREQ"):
            freqs = np.array([float(v) for v in lines[i + 1].split()])
            i += 2
            break
        i += 1
    assert phi is not None, path
    assert theta is not None, path
    assert freqs is not None, path
    levels = np.full((phi.size, theta.size, freqs.size), np.nan)
    row_ix = None
    for ln in lines[i:]:
        m = re.match(r"\s*PHIOBSAC\s*=\s*(-?[\d.]+)", ln)
        if m:
            row_ix = int(np.argmin(np.abs(phi - float(m.group(1)))))
            continue
        parts = ln.split()
        if row_ix is not None and len(parts) == freqs.size + 1:
            col_ix = int(np.argmin(np.abs(theta - float(parts[0]))))
            row = np.array([float(v) for v in parts[1:]])
            row[row <= -998.0] = np.nan
            levels[row_ix, col_ix, :] = row
    return phi, theta, freqs, levels


def _parse_ring(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, float]:
    """Bearings, bands, band levels and polar distance of a ring-format .hem.

    The hover-disk format carries a single ``PHIOBSEC`` bearing axis instead
    of the hemisphere's two angle axes; each data row is the bearing followed
    by the band levels (and derived summary columns, not read here).
    """
    lines = path.read_text(errors="replace").splitlines()
    dist = bearings = freqs = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("POLDIST"):
            dist = float(ln.split()[1])
        if ln.startswith("PHIOBSEC") and bearings is None and len(ln.split()) > 1:
            bearings = np.array([float(v) for v in lines[i + 1].split()])
            i += 2
            continue
        if ln.startswith("NFREQ"):
            freqs = np.array([float(v) for v in lines[i + 1].split()])
            i += 2
            break
        i += 1
    assert dist is not None, path
    assert bearings is not None, path
    assert freqs is not None, path
    levels = np.full((bearings.size, freqs.size), np.nan)
    for ln in lines[i:]:
        parts = ln.split()
        if len(parts) > freqs.size:
            try:
                bearing = float(parts[0])
            except ValueError:
                continue
            row = np.array([float(v) for v in parts[1:freqs.size + 1]])
            levels[int(np.argmin(np.abs(bearings - bearing)))] = row
    assert np.all(np.isfinite(levels)), path
    return bearings, freqs, levels, dist


def _parse_int(path: pathlib.Path) -> tuple[list[tuple[int, float, float, str]],
                                            list[tuple[int, ...]]]:
    """Hemisphere table and triangulation of a *_triangulation.int file."""
    txt = path.read_text()
    hems = [(int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4))
            for m in re.finditer(r"^(\d+)\t([\d.+-]+)\t([\d.+-]+)\t(\S+\.hem)", txt, re.MULTILINE)]
    tris: list[tuple[int, ...]] = []
    in_tri = False
    for line in txt.splitlines():
        if line.startswith("iTri"):
            in_tri = True
            continue
        if in_tri:
            parts = line.split()
            if len(parts) == 4 and parts[0].isdigit():
                tris.append(tuple(int(v) for v in parts[1:]))
            elif parts and not parts[0].isdigit():
                break
    return hems, tris


def _parse_inp(path: pathlib.Path) -> dict:
    """Constants and the time-dependent flight-path table of an .inp file."""
    txt = path.read_text()
    rows = []
    started = False
    for line in txt.splitlines():
        parts = line.split()
        if started and len(parts) >= 10:
            try:
                rows.append([float(parts[i]) for i in (0, 1, 2, 3)]
                            + [float(parts[i]) for i in (5, 6, 7, 8, 9)])
            except ValueError:
                pass
        if line.strip().startswith("sec"):
            started = True
    arr = np.asarray(rows)
    heli = re.search(r"HELIHEM\s*=\s*(\S+)", txt)
    assert heli is not None
    return {"heli": heli.group(1), "times": arr[:, 0], "positions": arr[:, 1:4],
            "speed": arr[:, 4], "vang": arr[:, 5], "heading": arr[:, 6],
            "roll": arr[:, 7], "ddb": arr[:, 8]}


def _parse_his(path: pathlib.Path) -> tuple[dict, np.ndarray]:
    """The first-microphone header metrics and history rows of a .his file."""
    lines = path.read_text().splitlines()
    header: dict | None = None
    rows = []
    for i, line in enumerate(lines):
        if line.strip().startswith("m          m          m"):
            vals = [float(v) for v in lines[i + 1].split()]
            keys = ["XMICM", "YMICM", "ZMIC", "HMIC", "SIGMA", "LAE", "LAMAX",
                    "PNLTM", "EPNL", "SELA", "T1pnlt", "T2pnlt", "I10db"]
            header = dict(zip(keys, vals))
            continue
        parts = line.split()
        if header is not None and len(parts) == 12:
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                pass
    assert header is not None, path
    return header, np.asarray(rows)


def _parse_onl(path: pathlib.Path) -> np.ndarray:
    """Per-microphone metric rows of a SingleEvt .onl file."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 12 and parts[0].isdigit():
            rows.append([float(v) for v in parts[1:]])
    return np.asarray(rows)  # XMICM YMICM ZMIC HMIC SIGMA EPNL PNLTM LAE LAMAX SELA I10db


@pytest.fixture(scope="session")
def r22_set(norah_root: pathlib.Path) -> dict:
    """The R22 hemisphere set with the shipped triangulation lookup."""
    hems, tris = _parse_int(norah_root / "Hemispheres" / "R22_triangulation.int")
    objs, speeds, angles, ids = [], [], [], []
    for ih, spd, ang, fname in hems:
        if spd <= 0.0:
            continue  # the hover disk uses a different (ring) format
        phi, theta, freqs, lv = _parse_hem(norah_root / "Hemispheres" / fname)
        objs.append(RotorcraftHemisphere(freqs, phi, theta, lv))
        speeds.append(spd)
        angles.append(ang)
        ids.append(ih)
    id2pos = {ih: i for i, ih in enumerate(ids)}
    triangles = [[id2pos[a], id2pos[b], id2pos[c]] for a, b, c in tris
                 if a in id2pos and b in id2pos and c in id2pos]
    return {"hemispheres": objs, "airspeeds": np.array(speeds),
            "path_angles": np.array(angles), "triangles": np.array(triangles)}


def _run_event(case: pathlib.Path, r22: dict) -> tuple[dict, np.ndarray, object]:
    header, rows = _parse_his(case.with_suffix(".his"))
    inp = _parse_inp(case.with_suffix(".inp"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sub-noy-floor spectra warn in PNL
        res = rotorcraft_event_level(
            r22["hemispheres"], r22["airspeeds"], r22["path_angles"],
            inp["times"], inp["positions"], (header["XMICM"], header["YMICM"]),
            level_offset=inp["ddb"],
            ground=RotorcraftGround(receiver_height=header["HMIC"],
                                    ground_elevation=header["ZMIC"],
                                    flow_resistivity=header["SIGMA"]),
            track_state=RotorcraftTrackState(
                airspeed=inp["speed"], path_angle=inp["vang"],
                heading=inp["heading"], bank_angle=inp["roll"]),
            interpolation=FlightConditionInterpolation(
                triangles=r22["triangles"]),
            atmosphere=RotorcraftAtmosphere(atmospheric_method="sae"))
    return header, rows, res


def _assert_geometry(rows: np.ndarray, res: object) -> None:
    """Emission angles, distances and retarded times against the .his rows."""
    ref_trec, ref_dist = rows[:, 4], rows[:, 5]
    ref_theta, ref_phi = rows[:, 6], rows[:, 7]
    assert np.max(np.abs(res.polar - ref_theta)) < 0.01
    # The prototype clamps the printed azimuth to the +-90 deg hemisphere span.
    assert np.max(np.abs(np.clip(res.azimuth, -90.0, 90.0) - ref_phi)) < 0.01
    assert np.max(np.abs(res.distance - ref_dist)) < 0.05
    # 346.1 m/s (Doc 32) vs the prototype's temperature-derived 346.19 m/s.
    assert np.max(np.abs(res.times - ref_trec)) < 0.02


def test_case4_hard_ground_event(norah_root: pathlib.Path, r22_set: dict) -> None:
    # R22 approach, microphone at the origin on sigma = 1e6 ground (hr 0.2 m):
    # the full chain reproduces every step out to 18 km.
    case = norah_root / "ARP" / "Case4" / "SE" / "R22_H1_APP_STD2_NE"
    header, rows, res = _run_event(case, r22_set)
    _assert_geometry(rows, res)
    d_la = res.a_levels - rows[:, 9]
    assert np.max(np.abs(d_la)) < 0.08
    assert res.la_max == pytest.approx(header["LAMAX"], abs=0.03)
    assert res.sel == pytest.approx(header["SELA"], abs=0.05)
    # PNLT wherever both sides define it; the prototype prints -999 when its
    # perceived noisiness is zero, this implementation returns NaN there.
    ref_pnlt = rows[:, 10]
    both = np.isfinite(res.pnlt) & (ref_pnlt > -998.0)
    assert both.sum() > 1000
    assert np.max(np.abs(res.pnlt[both] - ref_pnlt[both])) < 1.5
    assert np.sum(np.isnan(res.pnlt)) == np.sum(ref_pnlt <= -998.0)
    assert res.pnltm == pytest.approx(header["PNLTM"], abs=0.05)
    assert res.epnl == pytest.approx(header["EPNL"], abs=1.3)


def test_hover_ring_rim_reproduces_ring_rows(norah_root: pathlib.Path) -> None:
    # The committed in-ground-hover ring measurement (70 m polar distance,
    # 13 bearings every 30 deg, 31 bands): every ring bearing meets the
    # hemisphere rim at (phi, theta) = (+-90, |bearing|), band for band, and
    # a hover event over the derived hemisphere runs finite.
    bearings, freqs, levels, dist = _parse_ring(
        norah_root / "Hemispheres" / "R22_Ingroundhover_0kts_0deg_test.hem")
    assert dist == 70.0
    assert bearings.size == 13
    assert freqs.size == 31
    hige = hover_ring_hemisphere(freqs, bearings, levels, distance=dist)
    assert hige.distance == 70.0
    for bearing in (150.0, -120.0):
        k = int(np.nonzero(bearings == bearing)[0][0])
        side = 90.0 if bearing > 0 else -90.0
        got = hemisphere_source_level(hige, side, abs(bearing))
        assert np.allclose(got, levels[k])
    t = np.array([0.0, 1.0])
    pos = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sub-noy-floor spectra warn in PNL
        res = rotorcraft_event_level(
            [hige], [0.0], [0.0], t, pos, (100.0, 0.0),
            track_state=RotorcraftTrackState(airspeed=0.0, path_angle=0.0,
                                             heading=0.0, bank_angle=0.0))
    assert np.isfinite(res.la_max)
    assert np.isfinite(res.sel)


def test_case4_hover_ogh_event(norah_root: pathlib.Path) -> None:
    # Case 4 out-of-ground hover (iMode "OGH", the rotorcraft held at
    # (0, 0, 40 m)): the prototype derives the source from the in-ground-hover
    # ring by the horizontal bearing of the emission direction and applies its
    # own database correction, +8 dB in the shipped .int files where the
    # guidance's Table 3 publishes +12, so both are passed explicitly here.
    bearings, freqs, levels, dist = _parse_ring(
        norah_root / "Hemispheres" / "R22_Ingroundhover_0kts_0deg_test.hem")
    hige = hover_ring_hemisphere(freqs, bearings, levels, distance=dist,
                                 mapping="bearing")
    hoge = hover_derived_hemisphere(hige, "out_of_ground_hover", offset_db=8.0)
    case = norah_root / "ARP" / "Case4" / "SE" / "R22_DEP_RT2"
    inp = _parse_inp(case.with_suffix(".inp"))
    header, rows = _parse_his(case.with_suffix(".his"))
    mics = _parse_onl(case.with_suffix(".onl"))
    assert mics.shape[0] == 8
    # The single .inp step held for one second: the event needs two track
    # points, and a constant 1 s history leaves SEL equal to LA.
    t = np.array([0.0, 1.0])
    pos = np.repeat(inp["positions"], 2, axis=0)
    state = RotorcraftTrackState(
        airspeed=float(inp["speed"][0]), path_angle=float(inp["vang"][0]),
        heading=float(inp["heading"][0]), bank_angle=float(inp["roll"][0]))
    nadir = None
    for mic in mics:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sub-noy-floor spectra warn in PNL
            res = rotorcraft_event_level(
                [hoge], [0.0], [0.0], t, pos, (mic[0], mic[1]),
                ground=RotorcraftGround(receiver_height=mic[3],
                                        ground_elevation=mic[2],
                                        flow_resistivity=mic[4]),
                track_state=state,
                atmosphere=RotorcraftAtmosphere(atmospheric_method="sae"))
        # Measured deviations peak at +0.42 dB on the nadir and near-nadir
        # microphones (the prototype damps the coherent two-ray interference
        # of Eq. 30, as documented for the flyover cases); the six oblique
        # microphones reproduce to 0.3 dB across bearings, elevations,
        # heights (0.01-8.2 m) and ground classes (2e5-3e6 Pa.s/m2).
        assert res.la_max == pytest.approx(mic[8], abs=0.45), mic[:2]
        assert res.sel == pytest.approx(mic[9], abs=0.45), mic[:2]
        if mic[0] == 0.0 and mic[1] == 0.0:
            nadir = res
    # The .his row of the nadir microphone (hr = 0.2 m): geometry, retarded
    # time, the per-step LA and the tone-corrected maximum. The prototype
    # prints THETAOBS 90 / PHIOBSEC 0 under the vertical, fixing the bearing
    # convention of the hover-disk lookup. Its EPNL is not compared: on a
    # degenerate single-step history the prototype applies its fixed
    # 10*lg(0.5/10) duration correction, which measures its printing
    # convention rather than the event chain.
    assert nadir is not None
    assert nadir.distance[0] == pytest.approx(rows[0, 5], abs=0.05)   # DOBS
    assert nadir.polar[0] == pytest.approx(rows[0, 6], abs=0.01)      # THETAOBS
    assert nadir.azimuth[0] == pytest.approx(rows[0, 7], abs=0.01)    # PHIOBSAC
    assert nadir.times[0] - nadir.emission_times[0] == pytest.approx(
        rows[0, 4], abs=0.01)                                         # trec
    assert nadir.a_levels[0] == pytest.approx(rows[0, 9], abs=0.45)   # LA
    assert nadir.la_max == pytest.approx(header["LAMAX"], abs=0.45)
    assert nadir.pnltm == pytest.approx(header["PNLTM"], abs=0.5)


def test_case2_soft_ground_event(norah_root: pathlib.Path, r22_set: dict) -> None:
    # R22 departure over sigma = 2e5 ground (hr 1.2 m): metrics reproduce;
    # far steps carry the prototype's undocumented interference damping
    # (bounded here so a regression in this implementation cannot hide).
    case = norah_root / "ARP" / "Case2" / "SE" / "R22_H1_DEP_STD1_NE"
    header, rows, res = _run_event(case, r22_set)
    _assert_geometry(rows, res)
    d_la = res.a_levels - rows[:, 9]
    near = rows[:, 9] >= header["LAMAX"] - 20.0
    assert np.max(np.abs(d_la[near])) < 2.0
    assert np.max(np.abs(d_la)) < 5.5
    assert np.all(d_la < 0.5)   # this implementation never exceeds the prototype
    assert res.la_max == pytest.approx(header["LAMAX"], abs=0.03)
    assert res.sel == pytest.approx(header["SELA"], abs=0.4)
    assert res.pnltm == pytest.approx(header["PNLTM"], abs=0.15)
    assert res.epnl == pytest.approx(header["EPNL"], abs=0.6)


def test_case2_approach_event(norah_root: pathlib.Path, r22_set: dict) -> None:
    case = norah_root / "ARP" / "Case2" / "SE" / "R22_H1_APP_STD1_NE"
    header, rows, res = _run_event(case, r22_set)
    _assert_geometry(rows, res)
    near = rows[:, 9] >= header["LAMAX"] - 20.0
    assert np.max(np.abs((res.a_levels - rows[:, 9])[near])) < 2.0
    assert res.la_max == pytest.approx(header["LAMAX"], abs=0.03)
    assert res.sel == pytest.approx(header["SELA"], abs=0.4)


def test_case2_class_offset_event(norah_root: pathlib.Path, r22_set: dict) -> None:
    # The A600 flies the same profile on the R22 hemispheres with the Eq. 2
    # certification offset per flight condition (+2 dB on the level segments,
    # 0 dB on the approach), carried by the per-point level_offset.
    a600 = norah_root / "ARP" / "Case2" / "SE" / "A600_H1_APP_STD1_NE"
    r22c = norah_root / "ARP" / "Case2" / "SE" / "R22_H1_APP_STD1_NE"
    header_a, rows_a, res_a = _run_event(a600, r22_set)
    _, rows_r, _ = _run_event(r22c, r22_set)
    inp = _parse_inp(a600.with_suffix(".inp"))
    assert set(np.unique(inp["ddb"])) == {0.0, 2.0}
    # The prototype's own histories differ by exactly the ddB column.
    assert np.allclose(rows_a[:, 9] - rows_r[:, 9], inp["ddb"], atol=0.011)
    near = rows_a[:, 9] >= header_a["LAMAX"] - 20.0
    assert np.max(np.abs((res_a.a_levels - rows_a[:, 9])[near])) < 2.2
    assert res_a.la_max == pytest.approx(header_a["LAMAX"], abs=0.03)
    assert res_a.sel == pytest.approx(header_a["SELA"], abs=0.4)


def test_radar_track_event(norah_root: pathlib.Path, r22_set: dict) -> None:
    # Radar-track operation (curved approach, 120-128 kt, above the R22
    # envelope: nearest-neighbour selection throughout) on hard ground.
    case = norah_root / "ARP" / "Case2" / "SE" / "A600_APP_RT1"
    header, rows, res = _run_event(case, r22_set)
    _assert_geometry(rows, res)
    assert np.max(np.abs(res.a_levels - rows[:, 9])) < 0.15
    assert res.la_max == pytest.approx(header["LAMAX"], abs=0.03)
    assert res.sel == pytest.approx(header["SELA"], abs=0.1)
    assert res.pnltm == pytest.approx(header["PNLTM"], abs=0.2)
    assert res.epnl == pytest.approx(header["EPNL"], abs=0.2)


def test_case2_contour_grid(norah_root: pathlib.Path, r22_set: dict) -> None:
    # The 11 x 17 microphone grid of the Case 2 departure .onl, per ground
    # class (the prototype grid mixes sigma = 2e5 and 8e5 microphones; the
    # contour takes one ground per run).
    case = norah_root / "ARP" / "Case2" / "SE" / "R22_H1_DEP_STD1_NE"
    mics = _parse_onl(case.with_suffix(".onl"))
    inp = _parse_inp(case.with_suffix(".inp"))
    xs, ys = np.unique(mics[:, 0]), np.unique(mics[:, 1])
    assert xs.size == 11
    assert ys.size == 17
    kwargs = {
        "x": xs, "y": ys, "level_offset": inp["ddb"],
        "track_state": RotorcraftTrackState(
            airspeed=inp["speed"], path_angle=inp["vang"],
            heading=inp["heading"], bank_angle=inp["roll"]),
        "interpolation": FlightConditionInterpolation(
            triangles=r22_set["triangles"]),
        "atmosphere": RotorcraftAtmosphere(atmospheric_method="sae")}
    for sigma, tol_sel, tol_max in ((2.0e5, 0.7, 0.7), (8.0e5, 0.15, 0.7)):
        ground = RotorcraftGround(receiver_height=1.2,
                                  ground_elevation=float(mics[0, 2]),
                                  flow_resistivity=sigma)
        sel = rotorcraft_noise_contour(
            r22_set["hemispheres"], r22_set["airspeeds"], r22_set["path_angles"],
            inp["times"], inp["positions"], metric="exposure",
            ground=ground, **kwargs)
        lam = rotorcraft_noise_contour(
            r22_set["hemispheres"], r22_set["airspeeds"], r22_set["path_angles"],
            inp["times"], inp["positions"], metric="maximum",
            ground=ground, **kwargs)
        sub = mics[mics[:, 4] == sigma]
        assert sub.shape[0] > 30
        d_sel, d_max = [], []
        for mic in sub:
            i = int(np.nonzero(ys == mic[1])[0][0])
            j = int(np.nonzero(xs == mic[0])[0][0])
            d_sel.append(sel.level[i, j] - mic[9])
            d_max.append(lam.level[i, j] - mic[8])
        assert np.max(np.abs(d_sel)) < tol_sel, sigma
        assert np.max(np.abs(d_max)) < tol_max, sigma
        assert np.percentile(np.abs(d_sel), 50) < 0.3
        assert np.percentile(np.abs(d_max), 50) < 0.1


def test_reference_hemisphere_node_lookup(norah_root: pathlib.Path,
                                          r22_set: dict) -> None:
    # The parsed database reproduces the factual node values baked into the
    # synthetic suite (test_rotorcraft_noise._A109-style checks use inline
    # copies; here the real file feeds the same lookup).
    phi, theta, freqs, levels = _parse_hem(
        norah_root / "Hemispheres" / "R22_Flyover_55kts_0deg.hem")
    h = RotorcraftHemisphere(freqs, phi, theta, levels)
    assert phi.size == 19
    assert theta.size == 19
    assert freqs.size == 31
    from phonometry.aircraft.rotorcraft_noise import hemisphere_source_level

    k = int(np.nonzero(freqs == 500.0)[0][0])
    node = levels[9, 9, k]   # phi = 0, theta = 90
    assert np.isfinite(node)
    assert hemisphere_source_level(h, 0.0, 90.0)[k] == pytest.approx(node)


def test_case3_elevated_receiver_events(norah_root: pathlib.Path, r22_set: dict) -> None:
    # Case 3 places every microphone on its own ground elevation (an explicit
    # per-receiver ZMIC ramp) over sigma = 1e5 ground, with the track
    # pre-truncated to 2 km of the grid: an end-to-end oracle for the
    # receiver-side ground handling at every step.
    for name, epnl_tol in (("CH7_H1_APP_STD1_NE", 1.0), ("A600_H1_APP_STD2_NE", 1.8)):
        case = norah_root / "ARP" / "Case3" / "SE" / name
        header, rows, res = _run_event(case, r22_set)
        _assert_geometry(rows, res)
        assert np.max(np.abs(res.a_levels - rows[:, 9])) < 0.12
        assert res.la_max == pytest.approx(header["LAMAX"], abs=0.05)
        assert res.sel == pytest.approx(header["SELA"], abs=0.05)
        assert res.pnltm == pytest.approx(header["PNLTM"], abs=0.05)
        assert res.epnl == pytest.approx(header["EPNL"], abs=epnl_tol)


def test_case3_contour_with_elevation_grid(norah_root: pathlib.Path,
                                           r22_set: dict) -> None:
    # One contour call with the per-receiver ground-elevation array of the
    # Case 3 grid file reproduces every microphone of the .onl.
    case = norah_root / "ARP" / "Case3" / "SE" / "CH7_H1_APP_STD1_NE"
    mics = _parse_onl(case.with_suffix(".onl"))
    inp = _parse_inp(case.with_suffix(".inp"))
    xs, ys = np.unique(mics[:, 0]), np.unique(mics[:, 1])
    elevation = np.full((ys.size, xs.size), np.nan)
    ref_sel = np.full_like(elevation, np.nan)
    ref_max = np.full_like(elevation, np.nan)
    for mic in mics:
        i = int(np.nonzero(ys == mic[1])[0][0])
        j = int(np.nonzero(xs == mic[0])[0][0])
        elevation[i, j] = mic[2]
        ref_sel[i, j] = mic[9]
        ref_max[i, j] = mic[8]
    assert np.all(np.isfinite(elevation))
    assert np.unique(mics[:, 4]).tolist() == [1.0e5]
    kwargs = {
        "x": xs, "y": ys, "level_offset": inp["ddb"],
        "ground": RotorcraftGround(receiver_height=1.2,
                                   ground_elevation=elevation,
                                   flow_resistivity=1.0e5),
        "track_state": RotorcraftTrackState(
            airspeed=inp["speed"], path_angle=inp["vang"],
            heading=inp["heading"], bank_angle=inp["roll"]),
        "interpolation": FlightConditionInterpolation(
            triangles=r22_set["triangles"]),
        "atmosphere": RotorcraftAtmosphere(atmospheric_method="sae")}
    sel = rotorcraft_noise_contour(
        r22_set["hemispheres"], r22_set["airspeeds"], r22_set["path_angles"],
        inp["times"], inp["positions"], metric="exposure", **kwargs)
    lam = rotorcraft_noise_contour(
        r22_set["hemispheres"], r22_set["airspeeds"], r22_set["path_angles"],
        inp["times"], inp["positions"], metric="maximum", **kwargs)
    assert np.nanmax(np.abs(sel.level - ref_sel)) < 0.15
    assert np.nanmax(np.abs(lam.level - ref_max)) < 0.4
    assert np.nanpercentile(np.abs(sel.level - ref_sel), 50) < 0.05
    assert np.nanpercentile(np.abs(lam.level - ref_max), 50) < 0.05


def test_case2_contour_single_call_with_sigma_map(norah_root: pathlib.Path,
                                                  r22_set: dict) -> None:
    # The Case 2 grid mixes two ground classes across its microphones; one
    # contour call with the per-receiver flow-resistivity array reproduces
    # the whole .onl (the per-class scalar runs of the PR-2-era test remain
    # as the reference for each subset).
    case = norah_root / "ARP" / "Case2" / "SE" / "R22_H1_DEP_STD1_NE"
    mics = _parse_onl(case.with_suffix(".onl"))
    inp = _parse_inp(case.with_suffix(".inp"))
    xs, ys = np.unique(mics[:, 0]), np.unique(mics[:, 1])
    sigma = np.full((ys.size, xs.size), np.nan)
    ref_sel = np.full_like(sigma, np.nan)
    ref_max = np.full_like(sigma, np.nan)
    for mic in mics:
        i = int(np.nonzero(ys == mic[1])[0][0])
        j = int(np.nonzero(xs == mic[0])[0][0])
        sigma[i, j] = mic[4]
        ref_sel[i, j] = mic[9]
        ref_max[i, j] = mic[8]
    assert np.all(np.isfinite(sigma))
    assert sorted(np.unique(sigma).tolist()) == [2.0e5, 8.0e5]
    kwargs = {
        "x": xs, "y": ys, "level_offset": inp["ddb"],
        "ground": RotorcraftGround(receiver_height=1.2,
                                   ground_elevation=float(mics[0, 2]),
                                   flow_resistivity=sigma),
        "track_state": RotorcraftTrackState(
            airspeed=inp["speed"], path_angle=inp["vang"],
            heading=inp["heading"], bank_angle=inp["roll"]),
        "interpolation": FlightConditionInterpolation(
            triangles=r22_set["triangles"]),
        "atmosphere": RotorcraftAtmosphere(atmospheric_method="sae")}
    sel = rotorcraft_noise_contour(
        r22_set["hemispheres"], r22_set["airspeeds"], r22_set["path_angles"],
        inp["times"], inp["positions"], metric="exposure", **kwargs)
    lam = rotorcraft_noise_contour(
        r22_set["hemispheres"], r22_set["airspeeds"], r22_set["path_angles"],
        inp["times"], inp["positions"], metric="maximum", **kwargs)
    # Tolerances follow the per-class PR-2 measurements (the soft class
    # carries the documented far-tail interference divergence into SEL).
    assert np.nanmax(np.abs(sel.level - ref_sel)) < 0.7
    assert np.nanmax(np.abs(lam.level - ref_max)) < 0.7
    assert np.nanpercentile(np.abs(sel.level - ref_sel), 50) < 0.3
