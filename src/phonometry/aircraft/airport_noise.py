#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Noise-Power-Distance (NPD) event-level interpolation (ECAC Doc 29).

The ECAC Doc 29 airport-noise method describes an aircraft's noise emission with
**Noise-Power-Distance (NPD)** tables: the event noise level (``LAmax`` or the
sound exposure level ``SEL``) of an aircraft in steady straight flight on an
infinite path, tabulated over a grid of engine power settings and slant
distances at reference conditions. Placing an aircraft's noise at a receiver
starts by reading a level from this table for an arbitrary power and distance.

* :func:`npd_level` -- the interpolated event level ``L(P, d)``, linear in power
  and log-linear in distance (Eqs. 4-3/4-4), with extrapolation beyond the
  tabulated envelope.
* :func:`npd_curve` -- the level over a distance sweep at one power, returned as
  an :class:`NpdLevelResult` with a ``.plot()``.

The single-event stage segments a flight path and adjusts the NPD baseline per
segment (§4.3-4.5): :func:`impedance_adjustment` (Eq. 4-6/4-7),
:func:`lateral_attenuation` (β, ℓ), :func:`engine_installation_correction`
(φ, mounting), :func:`duration_correction`, the finite-segment
:func:`noise_fraction` and, behind takeoff ground-roll segments, the
:func:`start_of_roll_directivity` ``ΔSOR`` (§4.5.7). :func:`event_level`
assembles and sums them into the ``SEL``/``LAmax`` of a movement (mark takeoff
ground-roll segments with its ``ground_roll`` mask), and :func:`noise_contour`
evaluates it over a ground grid.

Source (clean-room, implemented from the standard text): ECAC Doc 29, 4th ed.,
Vol 2 (2016), §4.2-4.5. Validated per-term and end-to-end against the ECAC
Doc 29 5th ed. Vol 3 Part 1 reference workbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


#: NPD reference speed (160 kn) in ft/s, used by the noise-fraction scaled
#: distance (ECAC Doc 29 §4.5.6).
_VREF_FTS = 270.05
#: Engine-installation coefficients (a, b, c) by mounting (Eq. 4-15).
_INSTALLATION = {
    "wing": (0.0039, 0.062, 0.8786),
    "fuselage": (0.1225, 0.329, 1.0),
}


def lateral_attenuation(elevation_deg: float, lateral_m: float) -> float:
    """Excess lateral attenuation ``Λ(β, ℓ)`` over soft ground (Eq. 4-18/4-19).

    :param elevation_deg: Elevation angle ``β`` of the (equivalent level) path,
        in degrees.
    :param lateral_m: Lateral displacement ``ℓ`` from the ground track, in metres.
    :return: The lateral attenuation ``Λ``, in dB (subtracted from the level).
    :raises ValueError: If ``lateral_m`` is negative or non-finite.
    """
    ell = float(lateral_m)
    beta = float(elevation_deg)
    if not np.isfinite(ell) or ell < 0.0 or not np.isfinite(beta):
        raise ValueError("'lateral_m' must be non-negative and inputs finite.")
    gamma = 1.089 * (1.0 - np.exp(-0.00274 * ell)) if ell <= 914.0 else 1.0
    if beta < 0.0:
        lam = 10.857
    elif beta <= 50.0:
        lam = 1.137 - 0.0229 * beta + 9.72 * np.exp(-0.142 * beta)
    else:
        lam = 0.0
    return float(gamma * lam)


def engine_installation_correction(depression_deg: float, mounting: str = "wing") -> float:
    """Engine-installation lateral-directivity correction ``ΔI(φ)`` (Eq. 4-15/4-16).

    :param depression_deg: Depression angle ``φ`` (from the wing plane), in degrees.
    :param mounting: ``"wing"``, ``"fuselage"`` or ``"propeller"``.
    :return: The correction ``ΔI``, in dB (added to the level).
    :raises ValueError: If ``mounting`` is unknown or the angle is non-finite.
    """
    phi = float(depression_deg)
    if not np.isfinite(phi):
        raise ValueError("'depression_deg' must be finite.")
    key = mounting.strip().lower()
    if key in ("propeller", "prop"):
        return 0.0
    if key not in _INSTALLATION:
        raise ValueError(f"'mounting' must be 'wing', 'fuselage' or 'propeller', got {mounting!r}.")
    a, b, c = _INSTALLATION[key]
    phi_eff = np.radians(max(phi, 0.0))  # for φ<0, ΔI(φ)=ΔI(0)
    num = (a * np.cos(phi_eff) ** 2 + np.sin(phi_eff) ** 2) ** b
    den = c * np.sin(2.0 * phi_eff) ** 2 + np.cos(2.0 * phi_eff) ** 2
    return float(10.0 * np.log10(num / den))


def duration_correction(reference_speed: float, segment_speed: float) -> float:
    """Duration correction ``ΔV = 10·log10(Vref/Vseg)`` (Eq. 4-14, exposure only).

    :param reference_speed: NPD reference speed ``Vref`` (any consistent unit).
    :param segment_speed: Segment speed ``Vseg`` (same unit).
    :return: The duration correction ``ΔV``, in dB.
    :raises ValueError: If a speed is not strictly positive.
    """
    vref = float(reference_speed)
    vseg = float(segment_speed)
    if not (np.isfinite(vref) and vref > 0.0 and np.isfinite(vseg) and vseg > 0.0):
        raise ValueError("'reference_speed' and 'segment_speed' must be positive.")
    return float(10.0 * np.log10(vref / vseg))


def noise_fraction(q: float, segment_length: float, scaled_distance: float) -> float:
    """Finite-segment correction (noise fraction) ``ΔF`` (Eq. 4-20, exposure only).

    :param q: Signed distance from the segment start ``S1`` to the perpendicular
        foot ``Sp``, in metres (negative behind the segment).
    :param segment_length: Segment length ``λ``, in metres (``> 0``).
    :param scaled_distance: The scaled distance ``dλ`` (Appendix E), in metres
        (``> 0``).
    :return: The finite-segment correction ``ΔF``, in dB (``<= 0``, floored at
        −150 dB).
    :raises ValueError: If ``segment_length`` or ``scaled_distance`` is not
        positive.
    """
    lam = float(segment_length)
    dl = float(scaled_distance)
    if not (np.isfinite(lam) and lam > 0.0 and np.isfinite(dl) and dl > 0.0):
        raise ValueError("'segment_length' and 'scaled_distance' must be positive.")
    # Relative distances from the perpendicular foot Sp to the segment ends,
    # scaled by dλ: to the start S1 (−q) and to the end S2 (λ−q). The full
    # infinite path (α1→−∞, α2→+∞) then gives F=1, i.e. ΔF=0.
    a1 = -float(q) / dl
    a2 = (lam - float(q)) / dl
    frac = (a2 / (1.0 + a2**2) + np.arctan(a2) - a1 / (1.0 + a1**2) - np.arctan(a1)) / np.pi
    if frac <= 0.0:
        return -150.0
    return float(max(10.0 * np.log10(frac), -150.0))


#: Reference specific acoustic impedance of the NPD (ANP) data, in N·s/m³
#: (Eq. 4-6). #: Air impedance at the standard atmosphere (δ = θ = 1), Eq. 4-7.
_ZC_REF = 409.81
_ZC_STD = 416.86
#: Standard mean-sea-level pressure (kPa) and temperature (°C), Eq. 4-7.
_P0_KPA = 101.325
_T0_C = 15.0


def impedance_adjustment(temperature: float = _T0_C, pressure: float = _P0_KPA) -> float:
    """Acoustic-impedance adjustment of the standard NPD data (Eq. 4-6/4-7).

    The ANP NPD levels are normalised to a reference specific acoustic impedance
    of 409.81 N·s/m³. At the aerodrome's temperature and pressure the air
    impedance is ``ρc = 416.86·(δ/√θ)`` with ``δ = p/p0`` and
    ``θ = (T+273.15)/(T0+273.15)``, and the adjustment ``10·log10(ρc/409.81)`` is
    added to the NPD levels. Under the standard atmosphere it is +0.074 dB.

    :param temperature: Aerodrome air temperature ``T``, in °C (default 15 °C).
    :param pressure: Aerodrome air pressure ``p``, in kPa (default 101.325 kPa).
    :return: The impedance adjustment, in dB (added to the NPD level).
    :raises ValueError: If the pressure is not positive or inputs are non-finite.
    """
    t = float(temperature)
    p = float(pressure)
    if not (np.isfinite(t) and np.isfinite(p) and p > 0.0):
        raise ValueError("'pressure' must be positive and inputs finite.")
    if t <= -273.15:
        raise ValueError("'temperature' must be above absolute zero (−273.15 °C).")
    delta = p / _P0_KPA
    theta = (t + 273.15) / (_T0_C + 273.15)
    zc = _ZC_STD * delta / np.sqrt(theta)
    return float(10.0 * np.log10(zc / _ZC_REF))


#: Normalising distance for the start-of-roll directivity scaling (762 m), Eq. 4-25.
_DSOR0_M = 762.0


def start_of_roll_directivity(
    azimuth_deg: float, distance_m: float, engine: str = "jet") -> float:
    """Start-of-roll (ground-roll) directivity correction ``ΔSOR`` (Eq. 4-22/4-25).

    Behind a takeoff ground-roll segment, jet-exhaust noise radiates a lobed
    rearward pattern. ``ΔSOR`` adjusts the segment level relative to the level to
    the side of the start of roll, as a function of the azimuth ``ψ`` between the
    aircraft forward axis and the observer (Eq. 4-24a for turbofan jets, 4-24b for
    turboprops), scaled beyond 762 m by ``dSOR,0/dSOR`` (Eq. 4-25). It is only
    applied behind takeoff ground-roll segments (``90° ≤ ψ ≤ 180°``); ahead of the
    aircraft (``ψ < 90°``) it is zero.

    :param azimuth_deg: Azimuth ``ψ`` from the forward axis to the observer, in
        degrees (``ψ = arccos(q/dSOR)``, in ``[90, 180]`` behind the aircraft).
        Values below 90° return 0; values above 180° are clamped to 180°.
    :param distance_m: Distance ``dSOR`` from the observer to the segment start,
        in metres.
    :param engine: ``"jet"`` (turbofan, Eq. 4-24a) or ``"turboprop"`` (Eq. 4-24b).
    :return: The directivity correction ``ΔSOR``, in dB (added to the level).
    :raises ValueError: If ``engine`` is unknown or the inputs are invalid.
    """
    psi = float(azimuth_deg)
    dsor = float(distance_m)
    if not (np.isfinite(psi) and np.isfinite(dsor) and dsor > 0.0):
        raise ValueError("'distance_m' must be positive and inputs finite.")
    key = engine.strip().lower()
    if key not in ("jet", "turbofan", "turboprop", "prop", "propeller"):
        raise ValueError(f"'engine' must be 'jet' or 'turboprop', got {engine!r}.")
    if psi < 90.0:  # ahead of / abeam the aircraft: no rearward directivity
        return 0.0
    psi = min(psi, 180.0)
    if key in ("jet", "turbofan"):
        r = np.pi * psi / 180.0
        d0 = (2329.44 - 8.0573 * psi + 11.51 * np.exp(r)
              - 3.4601 * psi / np.log(r) - 17403338.3 * np.log(r) / psi**2)
    else:
        d0 = (-34643.898 + 30722161.987 / psi - 11491573930.510 / psi**2
              + 2349285669062.0 / psi**3 - 283584441904272.0 / psi**4
              + 20227150391251300.0 / psi**5 - 790084471305203000.0 / psi**6
              + 13050687178273800000.0 / psi**7)
    if dsor > _DSOR0_M:
        d0 *= _DSOR0_M / dsor
    return float(d0)


def _clean_table(
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    levels: NDArray[np.float64] | list[list[float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    p = np.asarray(powers, dtype=np.float64).ravel()
    d = np.asarray(distances, dtype=np.float64).ravel()
    lv = np.asarray(levels, dtype=np.float64)
    if p.size < 2 or d.size < 2:
        raise ValueError("'powers' and 'distances' must each have at least two values.")
    if lv.shape != (p.size, d.size):
        raise ValueError("'levels' must have shape (len(powers), len(distances)).")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(d)) and np.all(np.isfinite(lv))):
        raise ValueError("'powers', 'distances' and 'levels' must be finite.")
    if np.any(d <= 0.0):
        raise ValueError("'distances' must be strictly positive (slant range in m).")
    if np.any(np.diff(p) <= 0.0):
        raise ValueError("'powers' must be strictly increasing.")
    if np.any(np.diff(d) <= 0.0):
        raise ValueError("'distances' must be strictly increasing.")
    return p, d, lv


def _interp_distance(logd_tab: NDArray[np.float64], row: NDArray[np.float64],
                     logd: NDArray[np.float64]) -> NDArray[np.float64]:
    """Log-linear interpolation/extrapolation of one NPD row over distance (Eq. 4-4)."""
    idx = np.clip(np.searchsorted(logd_tab, logd) - 1, 0, logd_tab.size - 2)
    x0 = logd_tab[idx]
    x1 = logd_tab[idx + 1]
    y0 = row[idx]
    y1 = row[idx + 1]
    return np.asarray(y0 + (y1 - y0) / (x1 - x0) * (logd - x0), dtype=np.float64)


@dataclass(frozen=True)
class NpdLevelResult:
    """NPD event level over a distance sweep at one power (ECAC Doc 29).

    :ivar distance: Slant distances, in metres.
    :ivar level: Interpolated event level per distance, in dB.
    :ivar power: The engine power setting queried.
    :ivar table_distances: The tabulated slant distances, in metres.
    :ivar table_levels: The tabulated levels at the queried power, in dB.
    """

    distance: NDArray[np.float64]
    level: NDArray[np.float64]
    power: float
    table_distances: NDArray[np.float64]
    table_levels: NDArray[np.float64]

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the interpolated level versus slant distance (log axis)."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_npd_level

        return plot_npd_level(self, ax=ax, language=check_language(language), **kwargs)


def npd_level(
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    levels: NDArray[np.float64] | list[list[float]],
    power: float,
    distance: NDArray[np.float64] | list[float] | float,
) -> NDArray[np.float64]:
    """Interpolated NPD event level ``L(P, d)`` (ECAC Doc 29 §4.2, Eqs. 4-3/4-4).

    Interpolates log-linearly in slant distance (Eq. 4-4) at the two bracketing
    tabulated powers, then linearly in power (Eq. 4-3). Queries outside the
    tabulated envelope are extrapolated from the terminal segments.

    :param powers: Tabulated engine power settings (1-D, strictly increasing).
    :param distances: Tabulated slant distances, in metres (1-D, strictly
        increasing, positive).
    :param levels: Tabulated event levels, shape ``(len(powers), len(distances))``,
        in dB.
    :param power: Query engine power setting ``P``.
    :param distance: Query slant distance(s) ``d``, in metres.
    :return: The interpolated event level per query distance, in dB.
    :raises ValueError: If the table or inputs are invalid.
    """
    p, d, lv = _clean_table(powers, distances, levels)
    dq = np.atleast_1d(np.asarray(distance, dtype=np.float64))
    if dq.size == 0 or not np.all(np.isfinite(dq)) or np.any(dq <= 0.0):
        raise ValueError("'distance' must be finite and strictly positive.")
    pq = float(power)
    if not np.isfinite(pq):
        raise ValueError("'power' must be finite.")

    logd_tab = np.log10(d)
    logdq = np.log10(dq)
    # Log-linear interpolation over distance for every tabulated power row.
    rows = np.array([_interp_distance(logd_tab, lv[i], logdq) for i in range(p.size)])
    # Linear interpolation/extrapolation over power between the bracketing rows.
    j = int(np.clip(np.searchsorted(p, pq) - 1, 0, p.size - 2))
    frac = (pq - p[j]) / (p[j + 1] - p[j])
    return np.asarray(rows[j] + (rows[j + 1] - rows[j]) * frac, dtype=np.float64)


def npd_curve(
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    levels: NDArray[np.float64] | list[list[float]],
    power: float,
    query_distances: NDArray[np.float64] | list[float] | None = None,
) -> NpdLevelResult:
    """NPD event level over a distance sweep at one power setting.

    :param powers: Tabulated engine power settings.
    :param distances: Tabulated slant distances, in metres.
    :param levels: Tabulated event levels, shape ``(len(powers), len(distances))``.
    :param power: Query engine power setting.
    :param query_distances: Distances to evaluate, in metres; defaults to a log
        sweep across the tabulated envelope.
    :return: An :class:`NpdLevelResult`.
    :raises ValueError: If the inputs are invalid.
    """
    p, d, lv = _clean_table(powers, distances, levels)
    if query_distances is None:
        dq = np.logspace(np.log10(d[0]), np.log10(d[-1]), 200)
    else:
        dq = np.atleast_1d(np.asarray(query_distances, dtype=np.float64))
    level = npd_level(p, d, lv, power, dq)
    row = npd_level(p, d, lv, power, d)  # tabulated levels at the queried power
    return NpdLevelResult(
        distance=dq,
        level=level,
        power=float(power),
        table_distances=d,
        table_levels=row,
    )


# ===========================================================================
# Single-event flight-path calculation (ECAC Doc 29 §4.3-4.5)
# ===========================================================================

#: Reference speed 160 kn in m/s and the noise-fraction d0 = (2/π)·Vref·t0 (m).
_VREF_MS = 160.0 * 0.514444
_D0_M = (2.0 / np.pi) * _VREF_MS * 1.0
#: Recommended lower limit on NPD lookup distances, in metres (Doc 29 §4.2).
_NPD_FLOOR_M = 30.0


def _ground_track_offset(
    seg: NDArray[np.float64], s1: NDArray[np.float64], obs: NDArray[np.float64],
) -> float:
    """Horizontal perpendicular distance to the ground track (projected to z = 0)."""
    seg_g = seg.copy()
    seg_g[2] = 0.0
    gl = float(np.linalg.norm(seg_g))
    if gl <= 0.0:  # vertical ground track (degenerate): horizontal offset from s1
        return float(np.hypot(obs[0] - s1[0], obs[1] - s1[1]))
    ug = seg_g / gl
    og = obs - s1
    og[2] = 0.0
    return float(np.linalg.norm(og - np.dot(og, ug) * ug))


def _segment_angles(
    u: NDArray[np.float64], s1: NDArray[np.float64], obs: NDArray[np.float64],
    q: float, length: float, dp: float, lateral: float,
    z_foot: float, z_near: float, bank_deg: float,
) -> tuple[float, float]:
    """Elevation ``beta`` and depression ``phi`` for one segment, in degrees.

    §4.5.5 equivalent level path (Fig. 4-6): rotating the observer-segment
    triangle about the ground track makes the elevation of the perpendicular
    point ``β = arccos(ℓ/dp)``, which differs from ``atan2(z, ℓ)`` on inclined
    segments. Alongside the segment ``β`` uses that equivalent angle; behind or
    ahead it uses the nearest segment end over the same horizontal offset.
    The engine-installation depression angle ``φ = β ± ε`` uses the equivalent
    angle of the perpendicular point on the EXTENDED line (Eq. 4-17): the sign
    is positive for observers to starboard (right of the flight direction) and
    negative for observers to port (§4.5.2), with the bank angle ``ε`` measured
    counter-clockwise about the roll axis (positive in a left turn, starboard
    wing up).
    """
    # Observer side: +1 to starboard (right of the flight direction, where the
    # z-component of the horizontal cross product u × (obs − s1) is negative),
    # −1 to port, 0 on the ground track (§4.5.2: φ = β + ε to starboard,
    # φ = β − ε to port).
    side = -float(np.sign(u[0] * (obs[1] - s1[1]) - u[1] * (obs[0] - s1[0])))
    if lateral <= 0.0:  # directly overhead: elevation/depression are ±90°
        beta = 90.0 if z_near >= 0.0 else -90.0
        return beta, (90.0 if z_foot >= 0.0 else -90.0) + side * float(bank_deg)
    eq_angle = float(np.degrees(np.arccos(np.clip(lateral / dp, 0.0, 1.0)))) if dp > 0.0 else 90.0
    eq_angle = eq_angle if z_foot >= 0.0 else -eq_angle
    if 0.0 <= q <= length:
        beta = eq_angle
    else:
        beta = float(np.degrees(np.arctan2(z_near, lateral)))
    phi = eq_angle + side * float(bank_deg)
    return beta, phi


def _segment_geometry(
    s1: NDArray[np.float64], s2: NDArray[np.float64], obs: NDArray[np.float64],
    bank_deg: float = 0.0,
) -> tuple[float, float, float, float, float, float, float]:
    """Return ``(length, q, dp, ds, beta_deg, phi_deg, lateral_m)`` for a segment.

    ``s1``, ``s2`` and ``obs`` are 3-D points ``(x, y, z)`` (any consistent unit);
    the flight direction is ``s1 -> s2`` (ECAC Doc 29 Fig. 4-2). ``beta`` is the
    elevation angle for lateral attenuation (from the nearest actual segment
    point) and ``phi`` the depression angle for the engine-installation effect
    (from the perpendicular foot on the extended segment line), per §4.5.4-4.5.5.
    """
    seg = s2 - s1
    length = float(np.linalg.norm(seg))
    if length <= 0.0:  # degenerate (duplicate waypoints): caller skips it
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    u = seg / length
    q = float(np.dot(obs - s1, u))
    foot = s1 + q * u
    dp = float(np.linalg.norm(obs - foot))
    d1 = float(np.linalg.norm(obs - s1))
    d2 = float(np.linalg.norm(obs - s2))
    ds = d1 if q < 0.0 else (d2 if q > length else dp)
    lateral = _ground_track_offset(seg, s1, obs)
    z_foot = float(foot[2] - obs[2])
    z_near = float((s1[2] if q < 0.0 else (s2[2] if q > length else foot[2])) - obs[2])
    beta, phi = _segment_angles(u, s1, obs, q, length, dp, lateral,
                                z_foot, z_near, bank_deg)
    return length, q, dp, ds, beta, phi, lateral


@dataclass(frozen=True)
class FlyoverResult:
    """Single-event noise level of an aircraft movement at a receiver.

    :ivar level: The event level, in dB (SEL for ``metric="exposure"``, else the
        maximum level).
    :ivar metric: ``"exposure"`` (SEL) or ``"maximum"`` (LAmax).
    :ivar segment_levels: Per-segment contribution, in dB.
    :ivar observer: Receiver position ``(x, y, z)``, in metres.
    """

    level: float
    metric: str
    segment_levels: NDArray[np.float64]
    observer: NDArray[np.float64]

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-segment contributions to the event level."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_flyover

        return plot_flyover(self, ax=ax, language=check_language(language), **kwargs)


def _validate_path(path: NDArray[np.float64] | list[list[float]]) -> NDArray[np.float64]:
    """Coerce and validate a flight path to a finite ``(N, 5)`` array."""
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 5 or pts.shape[0] < 2:
        raise ValueError("'path' must have shape (N, 5) with N >= 2 (x,y,z,power,speed).")
    if not np.all(np.isfinite(pts)):
        raise ValueError("'path' must contain only finite values.")
    if np.any(pts[:, 3] < 0.0):
        raise ValueError("'path' power settings (column 4) must be non-negative.")
    if np.any(pts[:, 4] < 0.0):
        raise ValueError("'path' speeds (column 5) must be non-negative.")
    return pts


def _validate_ground_roll(
    ground_roll: NDArray[np.bool_] | list[bool] | None, n_points: int,
) -> NDArray[np.bool_] | None:
    """Coerce and validate the ground-roll segment mask (length ``N-1``)."""
    if ground_roll is None:
        return None
    gr = np.asarray(ground_roll, dtype=bool).ravel()
    if gr.shape != (n_points - 1,):
        raise ValueError(f"'ground_roll' must have length {n_points - 1} (one per segment).")
    return gr


def _validate_bank(
    bank: NDArray[np.float64] | list[float] | None, n_points: int,
) -> NDArray[np.float64] | None:
    """Coerce an optional per-segment bank-angle array (length ``N-1``)."""
    if bank is None:
        return None
    bk = np.asarray(bank, dtype=np.float64).ravel()
    if bk.shape != (n_points - 1,):
        raise ValueError(f"'bank' must have length {n_points - 1} (one per segment).")
    return bk


def _attenuation_geometry(
    s1: NDArray[np.float64], s2: NDArray[np.float64], obs: NDArray[np.float64],
    q: float, length: float, beta: float, phi: float, lateral: float,
    key: str, roll_behind: bool, roll_ahead: bool,
) -> tuple[float, float, float]:
    """Lateral-attenuation and installation angles for one segment (§4.5.5).

    Returns ``(beta_att, ell_att, phi_att)``: the general equivalent-path
    values, or the nearest-end geometry (``beta1 = asin(z1/d1)``,
    ``l = OC1 = sqrt(d1^2 - z1^2)`` and the d2/z2 analogues) for maximum-level
    metrics behind/ahead of the segment and for exposure metrics behind
    takeoff / ahead of landing ground roll.
    """
    behind, ahead = q < 0.0, q > length
    use_end = (key == "maximum" and (behind or ahead)) or roll_behind or roll_ahead
    if not use_end:
        return beta, lateral, phi
    end = s2 if ahead else s1
    d_end = float(np.linalg.norm(obs - end))
    z_end = float(end[2] - obs[2])
    if d_end <= 0.0:
        return 90.0, 0.0, 90.0
    beta_end = float(np.degrees(np.arcsin(np.clip(z_end / d_end, -1.0, 1.0))))
    oc = float(np.hypot(obs[0] - end[0], obs[1] - end[1]))
    return beta_end, oc, beta_end


def _segment_noise_fraction(
    q: float, length: float, d_lambda: float,
    roll_behind: bool, roll_ahead: bool,
) -> float:
    """Finite-segment fraction: general Eq. 4-20 or the reduced roll forms."""
    if roll_behind:
        return noise_fraction(0.0, length, d_lambda)      # Eq. 4-21a
    if roll_ahead:
        return noise_fraction(length, length, d_lambda)   # Eq. 4-21b
    return noise_fraction(q, length, d_lambda)            # Eq. 4-20


def _event_level_core(
    pts: NDArray[np.float64], obs: NDArray[np.float64],
    p: NDArray[np.float64], d: NDArray[np.float64],
    le: NDArray[np.float64], lm: NDArray[np.float64],
    vref: float, imp: float, mounting: str, key: str,
    ground_roll: NDArray[np.bool_] | None = None,
    landing_roll: NDArray[np.bool_] | None = None,
    bank: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Segment loop and summation shared by :func:`event_level`/:func:`noise_contour`.

    ``pts`` must be pre-validated and ``p, d, le, lm`` pre-cleaned; this hot path
    is called once per grid point by :func:`noise_contour`, so all input coercion
    is hoisted out to the callers. ``ground_roll`` marks takeoff ground-roll
    segments (start-of-roll directivity and reduced noise fraction behind the
    aircraft, Eq. 4-21a/4-22..4-25) and ``landing_roll`` marks landing rollout
    segments (reduced fraction Eq. 4-21b ahead, semi-circular directivity, no
    SOR); both use the §4.5.5 nearest-end lateral geometry and the Eq. 4-13b
    average segment speed. NPD lookups clamp to the recommended 30 m floor
    (Doc 29 §4.2).
    """
    engine = "turboprop" if mounting.strip().lower() in ("propeller", "prop") else "jet"
    seg_levels = []
    for i in range(pts.shape[0] - 1):
        s1, s2 = pts[i, :3], pts[i + 1, :3]
        p1, p2 = pts[i, 3], pts[i + 1, 3]
        v1, v2 = pts[i, 4], pts[i + 1, 4]
        eps = float(bank[i]) if bank is not None else 0.0
        length, q, dp, ds, beta, phi, lateral = _segment_geometry(s1, s2, obs, bank_deg=eps)
        if length <= 0.0:
            continue
        is_takeoff = ground_roll is not None and bool(ground_roll[i])
        is_landing = landing_roll is not None and bool(landing_roll[i])
        roll_behind = is_takeoff and q < 0.0     # behind the start of roll
        roll_ahead = is_landing and q > length   # ahead of the landing rollout
        beta_att, ell_att, phi_att = _attenuation_geometry(
            s1, s2, obs, q, length, beta, phi, lateral, key, roll_behind, roll_ahead)
        frac = np.clip(q / length, 0.0, 1.0)
        p_seg = np.sqrt(max(p1**2 + frac * (p2**2 - p1**2), 0.0))
        lam_att = lateral_attenuation(beta_att, ell_att)
        di = engine_installation_correction(phi_att, mounting)
        sor = start_of_roll_directivity(np.degrees(np.arccos(np.clip(q / ds, -1.0, 1.0))),
                                        max(ds, 1e-9), engine) if roll_behind else 0.0
        if key == "maximum":
            base = float(npd_level(p, d, lm, p_seg, max(ds, _NPD_FLOOR_M))[0])
            seg_levels.append(base + imp + di - lam_att + sor)
        else:
            dist = max(ds if (roll_behind or roll_ahead) else dp, _NPD_FLOOR_M)
            le_d = float(npd_level(p, d, le, p_seg, dist)[0])
            lm_d = float(npd_level(p, d, lm, p_seg, dist)[0])
            if is_takeoff or is_landing:
                # Eq. 4-13b: runway segments use the arithmetic mean speed,
                # regardless of the observer position (V = 0 endpoints are
                # legitimate at the very start of the takeoff roll).
                v_seg = 0.5 * (v1 + v2)
            else:
                v_seg = np.sqrt(max(v1**2 + frac * (v2**2 - v1**2), 0.0))
            if v_seg <= 0.0:
                raise ValueError(
                    "segment with zero mean speed (stationary segment in 'path').")
            dv = duration_correction(vref, v_seg)
            d_lambda = _D0_M * 10.0 ** ((le_d - lm_d) / 10.0)
            df = _segment_noise_fraction(q, length, d_lambda, roll_behind, roll_ahead)
            seg_levels.append(le_d + imp + dv + di - lam_att + df + sor)

    seg_arr = np.asarray(seg_levels, dtype=np.float64)
    if seg_arr.size == 0:
        total = float("-inf")
    elif key == "maximum":
        total = float(np.max(seg_arr))
    else:
        total = float(10.0 * np.log10(np.sum(10.0 ** (seg_arr / 10.0))))
    return total, seg_arr


def _npd_level_grid(
    p: NDArray[np.float64], lv: NDArray[np.float64],
    logd_tab: NDArray[np.float64], pq: NDArray[np.float64],
    dq: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorised NPD lookup ``L(P, d)`` for per-observer power and distance.

    Same arithmetic as :func:`npd_level` (Eq. 4-3/4-4), expression for
    expression, so the scalar and grid paths agree to machine precision;
    ``pq`` and ``dq`` are 1-D arrays of equal length.
    """
    logdq = np.log10(dq)
    k = np.clip(np.searchsorted(logd_tab, logdq) - 1, 0, logd_tab.size - 2)
    x0, x1 = logd_tab[k], logd_tab[k + 1]
    j = np.clip(np.searchsorted(p, pq) - 1, 0, p.size - 2)
    row_j = lv[j, k] + (lv[j, k + 1] - lv[j, k]) / (x1 - x0) * (logdq - x0)
    row_j1 = lv[j + 1, k] + (lv[j + 1, k + 1] - lv[j + 1, k]) / (x1 - x0) * (logdq - x0)
    frac = (pq - p[j]) / (p[j + 1] - p[j])
    return np.asarray(row_j + (row_j1 - row_j) * frac, dtype=np.float64)


def _grid_segment_frame(
    s1: NDArray[np.float64], s2: NDArray[np.float64],
    u: NDArray[np.float64], length: float, obs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-observer geometry of one segment, the array form of :func:`_segment_geometry`.

    Returns ``(q, dp, ds, behind, ahead, lateral, z_foot, z_near)`` with one
    entry per observer row of ``obs``.
    """
    q = (obs - s1) @ u
    foot = s1 + q[:, None] * u
    dp_ = np.linalg.norm(obs - foot, axis=1)
    behind, ahead = q < 0.0, q > length
    ds = np.where(behind, np.linalg.norm(obs - s1, axis=1),
                  np.where(ahead, np.linalg.norm(obs - s2, axis=1), dp_))
    # _ground_track_offset over all observers.
    seg_g = (s2 - s1).copy()
    seg_g[2] = 0.0
    gl = float(np.linalg.norm(seg_g))
    if gl > 0.0:
        ug = seg_g / gl
        og = obs - s1
        og[:, 2] = 0.0
        lateral = np.linalg.norm(og - np.outer(og @ ug, ug), axis=1)
    else:  # vertical ground track (degenerate)
        lateral = np.hypot(obs[:, 0] - s1[0], obs[:, 1] - s1[1])
    z_foot = foot[:, 2] - obs[:, 2]
    z_near = np.where(behind, s1[2], np.where(ahead, s2[2], foot[:, 2])) - obs[:, 2]
    return q, dp_, ds, behind, ahead, lateral, z_foot, z_near


def _grid_angles(
    u: NDArray[np.float64], s1: NDArray[np.float64], obs: NDArray[np.float64],
    dp_: NDArray[np.float64], lateral: NDArray[np.float64],
    z_foot: NDArray[np.float64], z_near: NDArray[np.float64],
    behind: NDArray[np.bool_], ahead: NDArray[np.bool_], eps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``beta``/``phi`` per observer (§4.5.2/4.5.5), array form of :func:`_segment_angles`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dp_ > 0.0, lateral / np.where(dp_ > 0.0, dp_, 1.0), 0.0)
        eq_angle = np.where(dp_ > 0.0,
                            np.degrees(np.arccos(np.clip(ratio, 0.0, 1.0))), 90.0)
    eq_angle = np.where(z_foot >= 0.0, eq_angle, -eq_angle)
    beta = np.where(behind | ahead,
                    np.degrees(np.arctan2(z_near, lateral)), eq_angle)
    # Observer side: +1 to starboard, −1 to port, 0 on the ground track
    # (§4.5.2: φ = β + ε to starboard, φ = β − ε to port).
    side = -np.sign(u[0] * (obs[:, 1] - s1[1]) - u[1] * (obs[:, 0] - s1[0]))
    phi = eq_angle + side * eps
    overhead = lateral <= 0.0
    beta = np.where(overhead, np.where(z_near >= 0.0, 90.0, -90.0), beta)
    phi = np.where(overhead,
                   np.where(z_foot >= 0.0, 90.0, -90.0) + side * eps, phi)
    return beta, phi


def _grid_attenuation_geometry(
    s1: NDArray[np.float64], s2: NDArray[np.float64], obs: NDArray[np.float64],
    beta: NDArray[np.float64], phi: NDArray[np.float64],
    lateral: NDArray[np.float64], ahead: NDArray[np.bool_],
    use_end: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Nearest-end ``beta``/``l``/``phi`` (§4.5.5), array form of :func:`_attenuation_geometry`."""
    end = np.where(ahead[:, None], s2, s1)
    d_end = np.linalg.norm(obs - end, axis=1)
    z_end = end[:, 2] - obs[:, 2]
    degenerate = d_end <= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_end = np.degrees(np.arcsin(np.clip(
            np.where(degenerate, 0.0, z_end / np.where(degenerate, 1.0, d_end)),
            -1.0, 1.0)))
    beta_end = np.where(degenerate, 90.0, beta_end)
    oc = np.where(degenerate, 0.0,
                  np.hypot(obs[:, 0] - end[:, 0], obs[:, 1] - end[:, 1]))
    return (np.where(use_end, beta_end, beta), np.where(use_end, oc, lateral),
            np.where(use_end, beta_end, phi))


def _grid_lateral_attenuation(
    beta_att: NDArray[np.float64], ell_att: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``Λ(β, ℓ)`` (Eq. 4-18/4-19), the array form of :func:`lateral_attenuation`."""
    gamma = np.where(ell_att <= 914.0,
                     1.089 * (1.0 - np.exp(-0.00274 * ell_att)), 1.0)
    lam = np.where(beta_att < 0.0, 10.857,
                   np.where(beta_att <= 50.0,
                            1.137 - 0.0229 * beta_att
                            + 9.72 * np.exp(-0.142 * beta_att), 0.0))
    return np.asarray(gamma * lam, dtype=np.float64)


def _grid_installation(
    phi_att: NDArray[np.float64], mount_key: str,
) -> NDArray[np.float64] | float:
    """``ΔI(φ)`` (Eq. 4-15/4-16), the array form of :func:`engine_installation_correction`."""
    if mount_key in ("propeller", "prop"):
        return 0.0
    a_i, b_i, c_i = _INSTALLATION[mount_key]
    phi_eff = np.radians(np.maximum(phi_att, 0.0))
    num = (a_i * np.cos(phi_eff) ** 2 + np.sin(phi_eff) ** 2) ** b_i
    den = c_i * np.sin(2.0 * phi_eff) ** 2 + np.cos(2.0 * phi_eff) ** 2
    return np.asarray(10.0 * np.log10(num / den), dtype=np.float64)


def _grid_sor(
    q: NDArray[np.float64], ds: NDArray[np.float64],
    roll_behind: NDArray[np.bool_], engine_jet: bool,
) -> NDArray[np.float64]:
    """``ΔSOR`` (Eq. 4-22..4-25), the array form of :func:`start_of_roll_directivity`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.degrees(np.arccos(np.clip(
            q / np.where(ds > 0.0, ds, 1e-9), -1.0, 1.0)))
    psi_c = np.clip(psi, 90.0, 180.0)
    if engine_jet:
        r = np.pi * psi_c / 180.0
        d0 = (2329.44 - 8.0573 * psi_c + 11.51 * np.exp(r)
              - 3.4601 * psi_c / np.log(r) - 17403338.3 * np.log(r) / psi_c**2)
    else:
        d0 = (-34643.898 + 30722161.987 / psi_c - 11491573930.510 / psi_c**2
              + 2349285669062.0 / psi_c**3 - 283584441904272.0 / psi_c**4
              + 20227150391251300.0 / psi_c**5 - 790084471305203000.0 / psi_c**6
              + 13050687178273800000.0 / psi_c**7)
    dsor = np.maximum(ds, 1e-9)
    d0 = np.where(dsor > _DSOR0_M, d0 * (_DSOR0_M / dsor), d0)
    return np.asarray(np.where(roll_behind & (psi >= 90.0), d0, 0.0), dtype=np.float64)


def _grid_noise_fraction(
    q: NDArray[np.float64], length: float, d_lambda: NDArray[np.float64],
    roll_behind: NDArray[np.bool_], roll_ahead: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """``ΔF`` (Eq. 4-20/4-21a/4-21b), the array form of :func:`_segment_noise_fraction`."""
    qf = np.where(roll_behind, 0.0, np.where(roll_ahead, length, q))
    a1 = -qf / d_lambda
    a2 = (length - qf) / d_lambda
    frac = (a2 / (1.0 + a2**2) + np.arctan(a2)
            - a1 / (1.0 + a1**2) - np.arctan(a1)) / np.pi
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(
            frac <= 0.0, -150.0,
            np.maximum(10.0 * np.log10(np.where(frac > 0.0, frac, 1.0)), -150.0)),
            dtype=np.float64)


class _GridContext(NamedTuple):
    """Per-call constants of the vectorised contour kernel."""

    obs: NDArray[np.float64]
    p: NDArray[np.float64]
    le: NDArray[np.float64]
    lm: NDArray[np.float64]
    logd_tab: NDArray[np.float64]
    vref: float
    imp: float
    mount_key: str
    engine_jet: bool
    maximum: bool


def _grid_segment_level(
    ctx: _GridContext, seg_pts: NDArray[np.float64], eps: float,
    is_takeoff: bool, is_landing: bool,
) -> NDArray[np.float64]:
    """Per-observer event level of one non-degenerate segment (Eq. 4-8/4-9).

    Assembles the NPD baseline and every correction through the ``_grid_*``
    helpers, mirroring one iteration of the scalar :func:`_event_level_core`
    loop expression for expression.
    """
    s1, s2 = seg_pts[0, :3], seg_pts[1, :3]
    seg = s2 - s1
    length = float(np.linalg.norm(seg))
    u = seg / length
    obs = ctx.obs
    q, dp_, ds, behind, ahead, lateral, z_foot, z_near = _grid_segment_frame(
        s1, s2, u, length, obs)
    beta, phi = _grid_angles(u, s1, obs, dp_, lateral, z_foot, z_near,
                             behind, ahead, eps)
    roll_behind = behind & is_takeoff
    roll_ahead = ahead & is_landing
    use_end = roll_behind | roll_ahead
    if ctx.maximum:
        use_end = use_end | behind | ahead
    beta_att, ell_att, phi_att = _grid_attenuation_geometry(
        s1, s2, obs, beta, phi, lateral, ahead, use_end)
    lam_att = _grid_lateral_attenuation(beta_att, ell_att)
    di = _grid_installation(phi_att, ctx.mount_key)
    frac = np.clip(q / length, 0.0, 1.0)
    p1, p2 = seg_pts[0, 3], seg_pts[1, 3]
    p_seg = np.sqrt(np.maximum(p1**2 + frac * (p2**2 - p1**2), 0.0))
    sor = _grid_sor(q, ds, roll_behind, ctx.engine_jet) if is_takeoff else 0.0
    if ctx.maximum:
        base = _npd_level_grid(ctx.p, ctx.lm, ctx.logd_tab, p_seg,
                               np.maximum(ds, _NPD_FLOOR_M))
        return np.asarray(base + ctx.imp + di - lam_att + sor, dtype=np.float64)
    dist = np.maximum(np.where(use_end, ds, dp_), _NPD_FLOOR_M)
    le_d = _npd_level_grid(ctx.p, ctx.le, ctx.logd_tab, p_seg, dist)
    lm_d = _npd_level_grid(ctx.p, ctx.lm, ctx.logd_tab, p_seg, dist)
    v1, v2 = seg_pts[0, 4], seg_pts[1, 4]
    if is_takeoff or is_landing:
        # Eq. 4-13b: runway segments use the arithmetic mean speed.
        v_seg = np.full(obs.shape[0], 0.5 * (v1 + v2))
    else:
        v_seg = np.sqrt(np.maximum(v1**2 + frac * (v2**2 - v1**2), 0.0))
    if np.any(v_seg <= 0.0):
        raise ValueError(
            "segment with zero mean speed (stationary segment in 'path').")
    dv = 10.0 * np.log10(ctx.vref / v_seg)
    d_lambda = _D0_M * 10.0 ** ((le_d - lm_d) / 10.0)
    df = _grid_noise_fraction(q, length, d_lambda, roll_behind, roll_ahead)
    return np.asarray(le_d + ctx.imp + dv + di - lam_att + df + sor,
                      dtype=np.float64)


def _grid_event_levels(
    pts: NDArray[np.float64], obs: NDArray[np.float64],
    p: NDArray[np.float64], d: NDArray[np.float64],
    le: NDArray[np.float64], lm: NDArray[np.float64],
    vref: float, imp: float, mounting: str, key: str,
    ground_roll: NDArray[np.bool_] | None = None,
    landing_roll: NDArray[np.bool_] | None = None,
    bank: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Vectorised counterpart of :func:`_event_level_core` over many observers.

    ``obs`` has shape ``(G, 3)``; returns the event level per observer, shape
    ``(G,)``. One pass per segment with every quantity broadcast over the
    observers through :func:`_grid_segment_level` and the ``_grid_*`` helpers,
    each of which mirrors its scalar counterpart expression for expression, so
    both paths agree to machine precision; the golden baseline and the
    scalar-equivalence test guard that.
    """
    mount_key = mounting.strip().lower()
    engine_installation_correction(0.0, mounting)   # validate 'mounting' once
    if not (np.isfinite(vref) and vref > 0.0):
        raise ValueError("'reference_speed' must be positive.")
    ctx = _GridContext(obs, p, le, lm, np.log10(d), vref, imp, mount_key,
                       mount_key not in ("propeller", "prop"), key == "maximum")
    n_obs = obs.shape[0]
    total = np.full(n_obs, -np.inf)
    energy = np.zeros(n_obs)
    any_segment = False
    for i in range(pts.shape[0] - 1):
        if float(np.linalg.norm(pts[i + 1, :3] - pts[i, :3])) <= 0.0:
            continue  # degenerate (duplicate waypoints): skip
        any_segment = True
        eps = float(bank[i]) if bank is not None else 0.0
        seg_level = _grid_segment_level(
            ctx, pts[i:i + 2], eps,
            ground_roll is not None and bool(ground_roll[i]),
            landing_roll is not None and bool(landing_roll[i]))
        if ctx.maximum:
            total = np.maximum(total, seg_level)
        else:
            energy += 10.0 ** (seg_level / 10.0)
    if not any_segment:
        return np.full(n_obs, -np.inf)
    if ctx.maximum:
        return total
    with np.errstate(divide="ignore"):
        return np.asarray(10.0 * np.log10(energy), dtype=np.float64)


def event_level(
    path: NDArray[np.float64] | list[list[float]],
    observer: NDArray[np.float64] | list[float],
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    exposure_levels: NDArray[np.float64] | list[list[float]],
    maximum_levels: NDArray[np.float64] | list[list[float]],
    *,
    reference_speed: float = _VREF_MS,
    mounting: str = "wing",
    metric: str = "exposure",
    temperature: float = _T0_C,
    pressure: float = _P0_KPA,
    ground_roll: NDArray[np.bool_] | list[bool] | None = None,
    landing_roll: NDArray[np.bool_] | list[bool] | None = None,
    bank: NDArray[np.float64] | list[float] | None = None,
) -> FlyoverResult:
    """Single-event noise level of a flight path at a receiver (ECAC Doc 29).

    Assembles the segment event levels (Eq. 4-8/4-9), the NPD baseline plus the
    duration, engine-installation, lateral-attenuation and finite-segment
    (noise-fraction) corrections, and the start-of-roll directivity behind takeoff
    ground-roll segments, and combines them into the exposure level ``SEL``
    (energy sum, Eq. 4-11) or the maximum level (Eq. 4-10).

    :param path: Flight-path points, shape ``(N, 5)``: columns ``x, y, z`` (m),
        engine power setting and speed (m/s). ``N-1`` segments are formed.
    :param observer: Receiver position ``(x, y, z)``, in metres.
    :param powers: NPD tabulated power settings.
    :param distances: NPD tabulated slant distances, in metres.
    :param exposure_levels: NPD exposure (SEL) levels, shape ``(P, D)``.
    :param maximum_levels: NPD maximum levels, shape ``(P, D)``.
    :param reference_speed: NPD reference speed, in m/s (default 160 kn).
    :param mounting: Engine mounting (``"wing"``/``"fuselage"``/``"propeller"``).
    :param metric: ``"exposure"`` (SEL) or ``"maximum"`` (LAmax).
    :param temperature: Aerodrome air temperature, in °C (impedance adjustment).
    :param pressure: Aerodrome air pressure, in kPa (impedance adjustment).
    :param ground_roll: Optional boolean mask of length ``N-1`` marking takeoff
        ground-roll segments; these receive the start-of-roll directivity ``ΔSOR``
        and reduced noise fraction behind the aircraft (§4.5.6-4.5.7).
    :param landing_roll: Optional boolean mask of length ``N-1`` marking landing
        rollout segments; ahead of them the reduced fraction (Eq. 4-21b), the
        nearest-end lateral geometry and no directivity term apply (§4.5.5-4.5.6).
    :param bank: Optional per-segment bank angle ``ε`` in degrees (length
        ``N-1``), measured counter-clockwise about the roll axis (positive in
        a left turn, starboard wing up); the depression angle becomes
        ``φ = β + ε`` for observers to starboard (right) of the track and
        ``φ = β − ε`` for observers to port (§4.5.2).
    :return: A :class:`FlyoverResult`. If every segment is degenerate
        (zero length) the level is ``-inf``.
    :raises ValueError: If the inputs are invalid.
    """
    pts = _validate_path(path)
    obs = np.asarray(observer, dtype=np.float64).ravel()
    if obs.shape != (3,) or not np.all(np.isfinite(obs)):
        raise ValueError("'observer' must be a finite (x, y, z) point.")
    key = metric.strip().lower()
    if key not in ("exposure", "maximum"):
        raise ValueError("'metric' must be 'exposure' or 'maximum'.")
    gr = _validate_ground_roll(ground_roll, pts.shape[0])
    lr = _validate_ground_roll(landing_roll, pts.shape[0])
    bk = _validate_bank(bank, pts.shape[0])
    p, d, le = _clean_table(powers, distances, exposure_levels)
    _, _, lm = _clean_table(powers, distances, maximum_levels)
    imp = impedance_adjustment(temperature, pressure)
    total, seg_arr = _event_level_core(
        pts, obs, p, d, le, lm, float(reference_speed), imp, mounting, key, gr, lr, bk)
    return FlyoverResult(level=total, metric=key, segment_levels=seg_arr, observer=obs)


@dataclass(frozen=True)
class NoiseContourResult:
    """Single-event noise level over a ground grid (ECAC Doc 29).

    :ivar x: Grid x coordinates, in metres.
    :ivar y: Grid y coordinates, in metres.
    :ivar level: Event level over the grid ``(len(y), len(x))``, in dB.
    :ivar metric: ``"exposure"`` (SEL) or ``"maximum"`` (LAmax).
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    level: NDArray[np.float64]
    metric: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot filled noise contours over the ground plane."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_noise_contour

        return plot_noise_contour(self, ax=ax, language=check_language(language), **kwargs)


def noise_contour(
    path: NDArray[np.float64] | list[list[float]],
    powers: NDArray[np.float64] | list[float],
    distances: NDArray[np.float64] | list[float],
    exposure_levels: NDArray[np.float64] | list[list[float]],
    maximum_levels: NDArray[np.float64] | list[list[float]],
    *,
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    reference_speed: float = _VREF_MS,
    mounting: str = "wing",
    metric: str = "exposure",
    temperature: float = _T0_C,
    pressure: float = _P0_KPA,
    ground_roll: NDArray[np.bool_] | list[bool] | None = None,
    landing_roll: NDArray[np.bool_] | list[bool] | None = None,
    bank: NDArray[np.float64] | list[float] | None = None,
) -> NoiseContourResult:
    """Single-event noise level over a ground grid (ECAC Doc 29 contour).

    Evaluates :func:`event_level` at every grid point ``(xi, yj, 0)``.

    :param path: Flight-path points, shape ``(N, 5)`` (see :func:`event_level`).
    :param powers: NPD tabulated power settings.
    :param distances: NPD tabulated slant distances, in metres.
    :param exposure_levels: NPD exposure (SEL) levels, shape ``(P, D)``.
    :param maximum_levels: NPD maximum levels, shape ``(P, D)``.
    :param x: Grid x coordinates, in metres.
    :param y: Grid y coordinates, in metres.
    :param reference_speed: NPD reference speed, in m/s.
    :param mounting: Engine mounting.
    :param metric: ``"exposure"`` (SEL) or ``"maximum"`` (LAmax).
    :param temperature: Aerodrome air temperature, in °C (impedance adjustment).
    :param pressure: Aerodrome air pressure, in kPa (impedance adjustment).
    :param ground_roll: Optional boolean mask (length ``N-1``) of takeoff
        ground-roll segments (see :func:`event_level`).
    :param landing_roll: Optional boolean mask (length ``N-1``) of landing
        rollout segments (see :func:`event_level`).
    :param bank: Optional per-segment bank angle ``ε`` in degrees, length
        ``N-1`` (see :func:`event_level`).
    :return: A :class:`NoiseContourResult`.
    :raises ValueError: If the inputs are invalid.
    """
    gx = np.asarray(x, dtype=np.float64).ravel()
    gy = np.asarray(y, dtype=np.float64).ravel()
    if gx.size < 2 or gy.size < 2:
        raise ValueError("'x' and 'y' must each have at least two grid points.")
    # Validate and clean the shared inputs once, not per grid point.
    pts = _validate_path(path)
    key = metric.strip().lower()
    if key not in ("exposure", "maximum"):
        raise ValueError("'metric' must be 'exposure' or 'maximum'.")
    gr = _validate_ground_roll(ground_roll, pts.shape[0])
    lr = _validate_ground_roll(landing_roll, pts.shape[0])
    bk = _validate_bank(bank, pts.shape[0])
    p, d, le = _clean_table(powers, distances, exposure_levels)
    _, _, lm = _clean_table(powers, distances, maximum_levels)
    vref = float(reference_speed)
    imp = impedance_adjustment(temperature, pressure)
    # One vectorised pass per flight-path segment over the whole grid (the
    # per-point scalar loop is O(grid × segments) Python calls; this is
    # numerically identical, see _grid_event_levels).
    xx, yy = np.meshgrid(gx, gy)
    obs = np.zeros((xx.size, 3))
    obs[:, 0] = xx.ravel()
    obs[:, 1] = yy.ravel()
    levels = _grid_event_levels(
        pts, obs, p, d, le, lm, vref, imp, mounting, key, gr, lr, bk)
    return NoiseContourResult(
        x=gx, y=gy, level=levels.reshape(gy.size, gx.size), metric=key)
