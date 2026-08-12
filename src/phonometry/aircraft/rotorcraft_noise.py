#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Rotorcraft noise by the hemisphere method (ECAC Doc 32 / NORAH2).

The ECAC Doc 32 rotorcraft-noise method describes a helicopter's highly directive
source with a **noise hemisphere**: one-third-octave-band sound pressure levels on
a spherical grid of azimuth ``φ`` and polar angle ``θ`` at a fixed 60 m reference
distance (at ICAO reference atmospheric conditions). Placing that source at a
receiver adds the propagation adjustment
:math:`\Delta L_p = \Delta L_s + \Delta L_a + \Delta L_g` (plus
:math:`\Delta L_d` with shielding): spherical spreading, atmospheric
absorption, ground effect and, later, shielding. Those adjustments depend on the
path and not on the rotorcraft, and live in
:mod:`~phonometry.aircraft.rotorcraft_propagation`; this module is the source
that emits and the event that receives.

This module provides the source primitives and the single-event method built on
them (clean-room, from the NORAH2 guidance SC01.D1.5d, the basis of ECAC
Doc 32):

* :func:`hemisphere_source_level` -- the interpolated source level ``L(fc, φ, θ)``
  from a :class:`RotorcraftHemisphere`, bilinear over the 10° grid (Eq. 13) with
  nearest-bin fill outside the measured coverage (Eq. 14/15).
* :func:`hover_ring_hemisphere` / :func:`hover_derived_hemisphere` -- the
  hover/idle source derivation of guidance §A.3.5 (Table 3): the ground-ring
  measurement of in-ground hover extended to a hemisphere assuming constant
  directivity in ``φ``, and the out-of-ground-hover and idle hemispheres
  derived from it by the published offsets or a measured 0°-direction
  difference.
* :func:`flight_condition_weights` / :func:`interpolated_source_level` -- the
  flight-condition interpolation across a hemisphere set: distance-scaled
  triangulation inside the convex hull of the normalised ``(V̄, γ̄)`` database
  conditions, nearest neighbour outside (Eq. 3-10).
* :func:`flight_path_kinematics` -- track kinematics by central finite
  differences: ground speed, airspeed, heading, curvature, bank and path angle
  (Eq. 16-21 / Doc 32 Eq. 8-10).
* :func:`rotorcraft_event_level` -- the received one-third-octave time history of
  a single event at recorded time (Eq. 1/22/23) and its integrated metrics:
  ``LASmax``, ``SEL`` (Doc 32 Eq. 27) and ``EPNL`` (Doc 32 Eq. 28, ICAO Annex 16).
* :func:`rotorcraft_noise_contour` -- the single-event ``SEL``/``LASmax`` ground
  grid.

Source (clean-room): ECAC Doc 32, 1st ed.; NORAH2 rotorcraft-noise modelling
guidance (EASA.2020.FC.06 SC01.D1.5d), §A.3 and §A.5. The event chain is
validated end to end against the NORAH2 reference implementation outputs for the
ARP verification cases (angles, retarded times, hemisphere selection, per-step
levels and event metrics).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .._internal.validation import (
    require_choice,
    require_positive,
    require_positive_array,
)
from .rotorcraft_propagation import (
    _C,
    _RH,
    _absorption_coefficient,
    _ground_effect,
    _resolve_flow_resistivity,
    _screening_core,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Standard acceleration of gravity, in m/s² (Eq. 20).
_G0 = 9.80665


@dataclass(frozen=True)
class RotorcraftHemisphere:
    """A rotorcraft noise hemisphere (ECAC Doc 32 §A.3.2).

    One-third-octave-band sound pressure levels on a regular azimuth/polar grid at
    the 60 m reference distance (ICAO reference atmosphere). Missing bins (outside
    the measured coverage) are ``NaN`` and filled by nearest-bin extrapolation on
    lookup.

    :ivar frequencies: Band centre frequencies, in Hz, shape ``(F,)``.
    :ivar azimuth: Azimuth angles ``φ``, in degrees, shape ``(A,)`` (``-90``
        port … ``+90`` starboard).
    :ivar polar: Polar angles ``θ``, in degrees, shape ``(P,)`` (``0`` forward …
        ``180`` rearward).
    :ivar levels: Band levels, in dB, shape ``(A, P, F)``.
    :ivar distance: Reference distance, in metres (default 60). The standard
        NORAH database uses 60 m; when the data uses another polar distance
        (e.g. 70 m hover rings), pass this value as ``reference_distance`` to
        :func:`~phonometry.aircraft.rotorcraft_propagation.spherical_spreading_adjustment`
        and
        :func:`~phonometry.aircraft.rotorcraft_propagation.atmospheric_adjustment`
        so the propagation chain honours it.
    """

    frequencies: NDArray[np.float64]
    azimuth: NDArray[np.float64]
    polar: NDArray[np.float64]
    levels: NDArray[np.float64]
    distance: float = _RH

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the hemisphere directivity for one band (polar section)."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_rotorcraft_hemisphere

        return plot_rotorcraft_hemisphere(self, ax=ax, language=check_language(language), **kwargs)

    def mirrored(self) -> RotorcraftHemisphere:
        """The hemisphere with the azimuth axis reversed (``φ → −φ``).

        Doc 32 Eq. 2 substitutes a class member whose main/tail-rotor
        configuration is mirrored with respect to the class reference (the
        bracketed types of its Table 2, e.g. ``[A600]`` in the ``R22`` class)
        by reversing the hemisphere azimuth angle.

        :return: A new :class:`RotorcraftHemisphere` with mirrored azimuth.
        """
        az = np.asarray(self.azimuth, dtype=np.float64)
        lv = np.asarray(self.levels, dtype=np.float64)
        return RotorcraftHemisphere(
            frequencies=np.asarray(self.frequencies, dtype=np.float64).copy(),
            azimuth=-az[::-1].copy(),
            polar=np.asarray(self.polar, dtype=np.float64).copy(),
            levels=lv[::-1, :, :].copy(),
            distance=self.distance,
        )

    def _filled(self) -> NDArray[np.float64]:
        """The gap-filled level grid (Eq. 14/15), computed once and cached.

        The cache relies on the frozen-dataclass contract: mutating the
        ``levels`` array in place after the first lookup leaves it stale.
        """
        cached = self.__dict__.get("_filled_cache")
        if cached is None:
            cached = _fill_grid(
                np.asarray(self.azimuth, dtype=np.float64),
                np.asarray(self.polar, dtype=np.float64),
                np.asarray(self.levels, dtype=np.float64))
            object.__setattr__(self, "_filled_cache", cached)
        return np.asarray(cached, dtype=np.float64)


def hemisphere_source_level(
    hemisphere: RotorcraftHemisphere, azimuth_deg: float, polar_deg: float,
) -> NDArray[np.float64]:
    """Interpolated source level ``L(fc, φ, θ)`` from a hemisphere (Eq. 13-15).

    The grid is first gap-filled by nearest-bin constant-value extrapolation
    (Eq. 14/15, computed once per hemisphere and cached), then the query is a
    bilinear interpolation in the energy domain over the four neighbouring
    azimuth/polar bins (Eq. 13). Filling the grid before interpolating keeps
    partially-measured cells continuous with their fully-measured neighbours
    (the valid corners still contribute) instead of snapping to a single bin.

    Queries outside the grid clamp to the boundary node and edge-interpolate;
    Eq. 14/15 taken literally would return the single nearest node, which
    coincides on the boundary nodes but is discontinuous alongside them, so the
    smoother clamp is intentional. Bands with no filled bin anywhere in the
    grid return ``NaN``.

    :param hemisphere: The :class:`RotorcraftHemisphere` source description.
    :param azimuth_deg: Emission azimuth ``φ``, in degrees.
    :param polar_deg: Emission polar angle ``θ``, in degrees.
    :return: Band levels at ``(φ, θ)``, in dB, shape ``(F,)``.
    """
    out = _source_levels(hemisphere, np.asarray([azimuth_deg], dtype=np.float64),
                         np.asarray([polar_deg], dtype=np.float64))
    return np.asarray(out[0], dtype=np.float64)


def _source_levels(
    hemisphere: RotorcraftHemisphere,
    azimuth_deg: NDArray[np.float64], polar_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorised :func:`hemisphere_source_level` over ``M`` queries, shape ``(M, F)``.

    The gap-filled grid has every bin finite except bands with no data at all
    (which stay ``NaN`` for every bin), so the four-corner energy blend needs
    no per-corner ``NaN`` handling: an all-``NaN`` band is ``NaN`` regardless
    of the corner weights, exactly as in the scalar lookup.
    """
    az = np.asarray(hemisphere.azimuth, dtype=np.float64)
    po = np.asarray(hemisphere.polar, dtype=np.float64)
    lv = hemisphere._filled()
    phi = np.clip(azimuth_deg, az[0], az[-1])
    theta = np.clip(polar_deg, po[0], po[-1])

    ia, wa = _axis_cells(az, phi)
    ip, wp = _axis_cells(po, theta)
    ia1 = np.minimum(ia + 1, az.size - 1)   # size-1 axis: weight 0, index clamped
    ip1 = np.minimum(ip + 1, po.size - 1)
    wa = wa[:, None]
    wp = wp[:, None]
    energy = ((1.0 - wa) * (1.0 - wp) * 10.0 ** (lv[ia, ip, :] / 10.0)
              + wa * (1.0 - wp) * 10.0 ** (lv[ia1, ip, :] / 10.0)
              + (1.0 - wa) * wp * 10.0 ** (lv[ia, ip1, :] / 10.0)
              + wa * wp * 10.0 ** (lv[ia1, ip1, :] / 10.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(10.0 * np.log10(energy), dtype=np.float64)


def _axis_cells(
    nodes: NDArray[np.float64], values: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Lower cell indices and fractional weights of ``values`` on a node axis.

    Size-1 axes (a single measured row or column) are handled explicitly: the
    only node is the cell with zero fractional weight. On a size-1 axis the
    upper corner index ``i + 1`` would overflow, so the index clamps to 0 and
    the weight to 0, which zeroes the upper-corner contribution.
    """
    if nodes.size == 1:
        return np.zeros(values.shape, dtype=np.intp), np.zeros(values.shape)
    i = np.clip(np.searchsorted(nodes, values) - 1, 0, nodes.size - 2)
    step = nodes[i + 1] - nodes[i]
    w = np.where(step > 0.0, (values - nodes[i]) / np.where(step > 0.0, step, 1.0), 0.0)
    return np.asarray(i, dtype=np.intp), np.asarray(w, dtype=np.float64)


def _fill_grid(
    azimuth: NDArray[np.float64], polar: NDArray[np.float64],
    levels: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Nearest-bin gap fill of a hemisphere grid (Eq. 14/15).

    Every empty ``(φ, θ)`` bin of each band takes the level of its angularly
    nearest filled bin, :math:`\rho = \arccos(x \cdot x_{m,n})`, compared
    through the dot product itself (monotone in ``ρ`` and, unlike the angle,
    well-conditioned near :math:`\rho = 0`). Equally-near bins are
    energy-averaged, as required by the
    guidance under Eq. 14/15. Bands with no filled bin at all stay ``NaN``.
    """
    n_az, n_po, n_f = levels.shape
    # Unit emission directions (Eq. 11 with rh = 1): x = cosθ, y = sinθ·sinφ,
    # z = sinθ·cosφ, one row per (φ, θ) bin in row-major grid order.
    phi = np.radians(azimuth)[:, None]
    theta = np.radians(polar)[None, :]
    vecs = np.stack([np.broadcast_to(np.cos(theta), (n_az, n_po)),
                     np.sin(theta) * np.sin(phi),
                     np.sin(theta) * np.cos(phi)], axis=-1).reshape(-1, 3)
    dots = np.clip(vecs @ vecs.T, -1.0, 1.0)
    flat = levels.reshape(n_az * n_po, n_f).copy()
    for b in range(n_f):
        band = flat[:, b]
        filled = np.isfinite(band)
        if filled.all() or not filled.any():
            continue
        d = dots[np.ix_(~filled, filled)]
        nearest = d >= d.max(axis=1, keepdims=True) - 1e-9   # ties: energy average
        e = 10.0 ** (band[filled] / 10.0)
        flat[~filled, b] = 10.0 * np.log10(
            (nearest * e).sum(axis=1) / nearest.sum(axis=1))
    return flat.reshape(n_az, n_po, n_f)


# --------------------------------------------------------------------------- #
# Hover, idle and taxi source derivation (guidance §A.3.5, Table 3)
# --------------------------------------------------------------------------- #


#: The published Table 3 (Approach 3) offsets from the in-ground-hover disk, in
#: dB (guidance §A.3.5). The guidance's own note warns they were derived from
#: inverted microphones on ground plates and may not hold for other setups.
_TABLE3_OFFSETS_DB: Final[Mapping[str, float]] = MappingProxyType({
    "out_of_ground_hover": 12.0,   # HOGE = HIGE + 12 dB
    "reduced_rpm_idle": -12.0,     # Gr. idle = HIGE - 12 dB
    "full_rpm_idle": -2.5,         # Fl. idle = HIGE - 2.5 dB
})


def _grid_axis(
    start: float, stop: float, step: float, name: str,
) -> NDArray[np.float64]:
    """A regular node axis from ``start`` to ``stop``; ``step`` must divide it."""
    s = require_positive(step, name)
    n = round((stop - start) / s)
    if n < 1 or abs((stop - start) / s - n) > 1e-9:
        raise ValueError(f"'{name}' must divide the {stop - start:g} degree span evenly.")
    return np.linspace(start, stop, n + 1)


def _ring_levels(
    bearings: NDArray[np.float64], levels: NDArray[np.float64],
    query_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Ring band levels at the query bearings, shape ``query_deg.shape + (F,)``.

    Periodic linear interpolation in the energy domain (the module's
    interpolation domain throughout): the ring closes on itself, so the cell
    between the last and the first bearing spans the wrap. A duplicated
    ±180° endpoint pair leaves that wrap cell empty, which no query reaches.
    """
    nodes = np.concatenate([bearings, bearings[:1] + 360.0])
    energy = 10.0 ** (np.concatenate([levels, levels[:1]], axis=0) / 10.0)
    q = bearings[0] + np.mod(query_deg - bearings[0], 360.0)
    i = np.clip(np.searchsorted(nodes, q) - 1, 0, nodes.size - 2)
    step = nodes[i + 1] - nodes[i]
    w = np.where(step > 0.0, (q - nodes[i]) / np.where(step > 0.0, step, 1.0), 0.0)
    blend = (1.0 - w[..., None]) * energy[i] + w[..., None] * energy[i + 1]
    return np.asarray(10.0 * np.log10(blend), dtype=np.float64)


def hover_ring_hemisphere(
    frequencies: NDArray[np.float64] | list[float],
    bearings: NDArray[np.float64] | list[float],
    levels: NDArray[np.float64] | list[list[float]],
    *,
    distance: float = 70.0,
    azimuth_step: float = 10.0,
    polar_step: float = 10.0,
    mapping: str = "constant_phi",
) -> RotorcraftHemisphere:
    r"""Noise hemisphere from a ground-ring hover measurement (guidance §A.3.5).

    In-ground hover, idle and their derived conditions are measured on a ring
    of ground microphones around the stationary rotorcraft (the CAEP in-ground
    hover practice the guidance points at): one band spectrum per ring bearing,
    ``0°`` at the nose and positive to starboard, reduced to the polar distance
    of the ring. Table 3 (Approaches 2/3) extends that ring to the full
    hemisphere "assuming constant directivity in φ"; the guidance prints no
    formula for the extension, so the two readings the data supports are
    provided and documented here:

    - ``"constant_phi"`` (default): the level depends only on the polar angle
      ``θ``, port bins reading the ring at ``−θ`` and starboard bins at
      ``+θ`` (each ring bearing meets the hemisphere rim at ``φ = ±90°``,
      ``θ = |bearing|``, and slides inward at constant ``φ`` from there). The
      ``φ = 0`` column under the aircraft takes the energy mean of the
      ``±θ`` ring values. This is the literal reading of the guidance text,
      and it preserves the port/starboard asymmetry the ring measures.
    - ``"bearing"``: the level depends only on the horizontal bearing of the
      emission direction, ``β = atan2(sin θ sin φ, cos θ)`` (constant
      directivity in elevation instead of in azimuth). This is what the
      NORAH2 reference implementation evaluates -- its out-of-ground-hover
      verification case is reproduced with this mapping (and diverges from
      ``"constant_phi"`` by several dB at steep emission angles, where the
      two readings part).

    Ring lookups interpolate periodically in the energy domain. The returned
    hemisphere carries ``distance`` (hover rings are commonly reduced to
    70 m rather than the 60 m of the flyover database), which the event chain
    and the propagation adjustments honour as the reference distance.

    :param frequencies: Band centre frequencies, in Hz, shape ``(F,)``.
    :param bearings: Ring bearings, in degrees within ``[-180, 180]``,
        strictly increasing, shape ``(B,)`` (``0`` at the nose, positive
        starboard). The ring closes periodically; a duplicated ``±180°``
        endpoint pair is accepted.
    :param levels: Ring band levels, in dB at the ring's polar distance,
        shape ``(B, F)``.
    :param distance: Polar distance of the ring, in metres (default 70).
    :param azimuth_step: Azimuth grid step, in degrees; must divide the
        180° span (default 10, the NORAH grid).
    :param polar_step: Polar grid step, in degrees; must divide the 180°
        span (default 10).
    :param mapping: ``"constant_phi"`` (guidance text) or ``"bearing"``
        (NORAH2 reference implementation), see above.
    :return: A :class:`RotorcraftHemisphere` on the requested grid.
    :raises ValueError: If the inputs are invalid.
    """
    freqs = require_positive_array(frequencies, "frequencies")
    brg = np.atleast_1d(np.asarray(bearings, dtype=np.float64))
    lv = np.asarray(levels, dtype=np.float64)
    if brg.ndim != 1 or brg.size < 2:
        raise ValueError("'bearings' must be 1-D with at least two ring directions.")
    if not np.all(np.isfinite(brg)) or np.any(np.diff(brg) <= 0.0):
        raise ValueError("'bearings' must be finite and strictly increasing.")
    if brg[0] < -180.0 or brg[-1] > 180.0:
        raise ValueError("'bearings' must lie within [-180, 180] degrees.")
    if lv.shape != (brg.size, freqs.size):
        raise ValueError("'levels' must have shape (B, F) matching 'bearings' "
                         "and 'frequencies'.")
    if not np.all(np.isfinite(lv)):
        raise ValueError("'levels' must be finite.")
    dist = require_positive(distance, "distance")
    key = require_choice(mapping, "mapping", ("constant_phi", "bearing"))
    az = _grid_axis(-90.0, 90.0, azimuth_step, "azimuth_step")
    po = _grid_axis(0.0, 180.0, polar_step, "polar_step")

    if key == "bearing":
        phi = np.radians(az)[:, None]
        theta = np.radians(po)[None, :]
        beta = np.degrees(np.arctan2(np.sin(theta) * np.sin(phi), np.cos(theta)))
        grid = _ring_levels(brg, lv, beta)
    else:
        starboard = _ring_levels(brg, lv, po)
        port = _ring_levels(brg, lv, -po)
        grid = np.empty((az.size, po.size, freqs.size), dtype=np.float64)
        # The azimuth axis is symmetric by construction, so each node's side
        # is fixed by its index: port before the middle, starboard after it
        # and, with an even cell count, the phi = 0 column exactly in the
        # middle. Selecting by index (not by comparing the node against 0.0)
        # keeps the middle column the energy mean even when a step such as
        # 180/14 leaves its floating-point node a few ulp away from zero.
        cells = az.size - 1
        grid[: (cells + 1) // 2] = port
        grid[cells // 2 + 1:] = starboard
        if cells % 2 == 0:
            energy = 0.5 * (10.0 ** (starboard / 10.0) + 10.0 ** (port / 10.0))
            grid[cells // 2] = 10.0 * np.log10(energy)
    return RotorcraftHemisphere(
        frequencies=freqs.copy(), azimuth=az, polar=po, levels=grid,
        distance=dist)


def hover_derived_hemisphere(
    hemisphere: RotorcraftHemisphere,
    condition: str,
    *,
    offset_db: float | None = None,
) -> RotorcraftHemisphere:
    """HOGE/idle hemisphere derived from in-ground hover (guidance Table 3).

    Table 3 derives the out-of-ground-hover and idle sources from the
    in-ground-hover directivity pattern by a level offset, applied here
    uniformly to every band and bin: the table is stated on ``LA`` levels,
    and a constant spectral shift moves the ``LA`` by exactly that value
    (which is also how the NORAH2 reference database applies its
    corrections). With ``offset_db`` the offset is the measured 0°-direction
    difference of Approach 2, ``LA_cond(0°) − LA_HIGE(0°)``; without it the
    Approach 3 constants apply (+12 / −12 / −2.5 dB for
    ``"out_of_ground_hover"`` / ``"reduced_rpm_idle"`` / ``"full_rpm_idle"``,
    with the guidance's caveat that they come from inverted microphones on
    ground plates). The corrections shipped with the NORAH2 public database
    differ from the published constants (+8 / −10 / −2 dB in every type's
    interpolation file); pass them as ``offset_db`` to reproduce the
    reference implementation.

    :param hemisphere: The in-ground-hover :class:`RotorcraftHemisphere`
        (typically from :func:`hover_ring_hemisphere`; a fully measured
        Approach 1 hemisphere works the same).
    :param condition: ``"out_of_ground_hover"``, ``"reduced_rpm_idle"`` or
        ``"full_rpm_idle"``.
    :param offset_db: Explicit offset from in-ground hover, in dB (Approach 2
        or a database correction). Default ``None``: the Approach 3 constant
        of ``condition``.
    :return: A new :class:`RotorcraftHemisphere` at the same grid and
        reference distance, every measured bin shifted by the offset
        (``NaN`` bins stay ``NaN``).
    :raises ValueError: If the inputs are invalid.
    """
    key = require_choice(condition, "condition", tuple(_TABLE3_OFFSETS_DB))
    if offset_db is None:
        offset = _TABLE3_OFFSETS_DB[key]
    elif math.isfinite(offset_db):
        offset = float(offset_db)
    else:
        raise ValueError("'offset_db' must be finite.")
    return RotorcraftHemisphere(
        frequencies=np.asarray(hemisphere.frequencies, dtype=np.float64).copy(),
        azimuth=np.asarray(hemisphere.azimuth, dtype=np.float64).copy(),
        polar=np.asarray(hemisphere.polar, dtype=np.float64).copy(),
        levels=np.asarray(hemisphere.levels, dtype=np.float64) + offset,
        distance=hemisphere.distance)


# --------------------------------------------------------------------------- #
# Flight-condition interpolation (guidance Eq. 3-10)
# --------------------------------------------------------------------------- #


def flight_condition_weights(
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    airspeed: float,
    path_angle: float,
    *,
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
) -> list[tuple[int, float]]:
    r"""Hemisphere blending weights for a flight condition (Eq. 3-10).

    The database flight conditions and the query are scaled by the database
    spans, :math:`\bar{V} = V/(V_{\mathrm{max}} - V_{\mathrm{min}})` and
    :math:`\bar{\gamma} = F_{fc} \cdot \gamma
    / (\gamma_{\mathrm{max}} - \gamma_{\mathrm{min}})` with
    the empirical flight-condition scaling factor :math:`F_{fc} = 2`: the
    guidance's
    normalisation (Eq. 3-6), which subtracts no minima -- a shared offset
    cancels in the distances ``δ_j`` (Eq. 7) either way. Inside the
    convex hull of the database conditions the enveloping Delaunay triangle
    contributes with inverse-distance weights
    :math:`(1/\delta_j)/\sum (1/\delta_j)`,
    :math:`\delta_j = \sqrt{(\bar{\gamma}-\bar{\gamma}_j)^2
    + (\bar{V}-\bar{V}_j)^2}` (Eq. 7/8); outside it (and whenever no
    triangulation exists, e.g. collinear conditions) the nearest database
    condition is adopted unblended (Eq. 9/10). A query on a database condition
    returns that hemisphere alone. ECAC Doc 32, 1st ed., §4.1 defines no
    interpolation ("select the most appropriate hemisphere"); this is the
    interpolation of the NORAH2 guidance §A.3.1 on which the NORAH database and
    reference implementation operate, and it degrades to the Doc 32 behaviour
    outside the measured envelope.

    The scaling is span-based, so the weights do not depend on the units of
    ``airspeeds`` or ``path_angles`` as long as the query uses the same units
    as the database conditions.

    :param airspeeds: Database hemisphere airspeeds ``V_j``, shape ``(J,)``.
    :param path_angles: Database hemisphere path angles ``γ_j``, in degrees,
        shape ``(J,)`` (negative for descent).
    :param airspeed: Query airspeed ``V_A`` (the airspeed, not the ground
        speed, selects the hemisphere; guidance §A.3.3).
    :param path_angle: Query path angle ``γ``, in degrees.
    :param scaling_factor: Flight-condition scaling factor ``F_fc`` applied to
        the normalised path angle (default 2, the guidance's empirical value).
    :param triangles: Optional precomputed triangulation, shape ``(T, 3)``
        0-based indices into the database conditions (guidance §A.3.1 step 4
        admits a lookup table; the NORAH database ships one per type). Default
        ``None`` computes the Delaunay triangulation of the normalised
        conditions. The shipped NORAH lookup tables triangulate the raw
        ``(V, γ)`` plane instead of the normalised one, so passing them
        reproduces the reference implementation bin for bin.
    :return: The ``(index, weight)`` pairs, weights summing to 1.
    :raises ValueError: If the inputs are invalid.
    """
    v = np.atleast_1d(np.asarray(airspeeds, dtype=np.float64))
    g = np.atleast_1d(np.asarray(path_angles, dtype=np.float64))
    if v.ndim != 1 or v.shape != g.shape or v.size < 1:
        raise ValueError("'airspeeds' and 'path_angles' must be 1-D of equal, non-zero size.")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(g))):
        raise ValueError("'airspeeds' and 'path_angles' must be finite.")
    ffc = require_positive(scaling_factor, "scaling_factor")
    if not np.isfinite(airspeed) or not np.isfinite(path_angle):
        raise ValueError("'airspeed' and 'path_angle' must be finite.")
    if v.size == 1:
        return [(0, 1.0)]

    vspan = float(v.max() - v.min())
    gspan = float(g.max() - g.min())
    vn = v / vspan if vspan > 0.0 else np.zeros_like(v)
    gn = ffc * g / gspan if gspan > 0.0 else np.zeros_like(g)
    qv = airspeed / vspan if vspan > 0.0 else 0.0
    qg = ffc * path_angle / gspan if gspan > 0.0 else 0.0
    pts = np.column_stack([vn, gn])
    q = np.array([qv, qg])
    delta = np.hypot(vn - qv, gn - qg)

    exact = int(np.argmin(delta))
    if delta[exact] < 1e-12:                       # on a database condition
        return [(exact, 1.0)]

    simplex = _enveloping_simplex(pts, q, v.size, triangles)
    if simplex is None:                            # outside the hull (Eq. 9/10)
        return [(exact, 1.0)]
    d = delta[simplex]
    w = (1.0 / d) / np.sum(1.0 / d)
    order = np.argsort(simplex)
    return [(int(simplex[i]), float(w[i])) for i in order]


def _enveloping_simplex(
    pts: NDArray[np.float64], q: NDArray[np.float64], n: int,
    triangles: NDArray[np.int_] | list[list[int]] | None,
) -> NDArray[np.intp] | None:
    """The triangle of ``pts`` (given or Delaunay) enveloping ``q``, or ``None``."""
    if triangles is not None:
        return _simplex_from_table(pts, q, n, triangles)
    from scipy.spatial import Delaunay, QhullError

    try:
        dt = Delaunay(pts)
    except QhullError:                             # collinear/duplicate conditions
        return None
    simplex = int(dt.find_simplex(q))
    if simplex < 0:
        return None
    return np.asarray(dt.simplices[simplex], dtype=np.intp)


def _simplex_from_table(
    pts: NDArray[np.float64], q: NDArray[np.float64], n: int,
    triangles: NDArray[np.int_] | list[list[int]],
) -> NDArray[np.intp] | None:
    """The first triangle of a lookup table enveloping ``q``, or ``None``."""
    tri = np.asarray(triangles, dtype=np.intp)
    if tri.ndim != 2 or tri.shape[1] != 3 or tri.size == 0:
        raise ValueError("'triangles' must have shape (T, 3).")
    if tri.min() < 0 or tri.max() >= n:
        raise ValueError("'triangles' indices must address the database conditions.")
    for row in tri:
        p0, p1, p2 = pts[row]
        m = np.column_stack([p1 - p0, p2 - p0])
        try:
            lam = np.linalg.solve(m, q - p0)
        except np.linalg.LinAlgError:              # degenerate triangle
            continue
        if lam[0] >= -1e-9 and lam[1] >= -1e-9 and lam.sum() <= 1.0 + 1e-9:
            return np.asarray(row, dtype=np.intp)
    return None


def interpolated_source_level(
    hemispheres: Sequence[RotorcraftHemisphere],
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    airspeed: float,
    path_angle: float,
    azimuth_deg: float,
    polar_deg: float,
    *,
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
) -> NDArray[np.float64]:
    """Source level at a flight condition between hemispheres (Eq. 8/10 over Eq. 13).

    Blends :func:`hemisphere_source_level` lookups of the hemispheres selected
    by :func:`flight_condition_weights` in the energy domain (Eq. 8).

    :param hemispheres: The database hemispheres, one per flight condition.
    :param airspeeds: Database airspeeds ``V_j``, shape ``(J,)``.
    :param path_angles: Database path angles ``γ_j``, in degrees, shape ``(J,)``.
    :param airspeed: Query airspeed ``V_A`` (same units as ``airspeeds``).
    :param path_angle: Query path angle ``γ``, in degrees.
    :param azimuth_deg: Emission azimuth ``φ``, in degrees.
    :param polar_deg: Emission polar angle ``θ``, in degrees.
    :param scaling_factor: Flight-condition scaling factor ``F_fc`` (default 2).
    :param triangles: Optional precomputed triangulation (see
        :func:`flight_condition_weights`).
    :return: Band levels at the reference distance, in dB, shape ``(F,)``.
    :raises ValueError: If the inputs are invalid.
    """
    freqs = _common_frequencies(hemispheres, airspeeds)
    weights = flight_condition_weights(
        airspeeds, path_angles, airspeed, path_angle,
        scaling_factor=scaling_factor, triangles=triangles)
    energy = np.zeros(freqs.shape, dtype=np.float64)
    for j, w in weights:
        energy += w * 10.0 ** (hemisphere_source_level(
            hemispheres[j], azimuth_deg, polar_deg) / 10.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(10.0 * np.log10(energy), dtype=np.float64)


def _common_frequencies(
    hemispheres: Sequence[RotorcraftHemisphere],
    airspeeds: NDArray[np.float64] | list[float],
) -> NDArray[np.float64]:
    """The shared band grid of a hemisphere set (validated)."""
    if len(hemispheres) == 0:
        raise ValueError("'hemispheres' must not be empty.")
    n = np.atleast_1d(np.asarray(airspeeds, dtype=np.float64)).size
    if len(hemispheres) != n:
        raise ValueError("'hemispheres' and the flight conditions must have equal length.")
    freqs = np.asarray(hemispheres[0].frequencies, dtype=np.float64)
    for h in hemispheres[1:]:
        if not np.array_equal(np.asarray(h.frequencies, dtype=np.float64), freqs):
            raise ValueError("All hemispheres must share one band grid.")
    return freqs


# --------------------------------------------------------------------------- #
# Flight-path kinematics (guidance Eq. 16-21 / Doc 32 Eq. 8-10)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FlightPathKinematics:
    r"""Kinematics of a rotorcraft track (guidance Eq. 16-21 / Doc 32 Eq. 8-10).

    All rates come from central finite differences around each track point.

    :ivar times: Track times, in s, shape ``(N,)``.
    :ivar positions: Track positions ``(x, y, z)``, in metres, shape ``(N, 3)``.
    :ivar ground_speed: Ground speed ``V_g`` (Eq. 16), in m/s, shape ``(N,)``.
    :ivar airspeed: Airspeed ``V_A`` (Eq. 17, zero-wind), in m/s, shape ``(N,)``.
    :ivar heading: Heading
        :math:`\Theta = \operatorname{atan2}(\Delta X, \Delta Y)` (Eq. 19),
        in degrees, shape ``(N,)``.
    :ivar curvature: Track curvature :math:`K = \Delta\Theta/\Delta S`
        (Eq. 18), in rad/m, shape
        ``(N,)`` (zero where the ground speed vanishes).
    :ivar bank_angle: Bank angle
        :math:`\Phi = \arctan(K \cdot V_g^2/g)` (Eq. 20), in degrees,
        positive starboard down, shape ``(N,)``.
    :ivar path_angle: Path angle
        :math:`\gamma = \arctan(\Delta Z/\Delta S)` (Doc 32 Eq. 10), in
        degrees, positive climbing, shape ``(N,)``.

    .. note::
        The guidance prints Eq. 21 as
        :math:`\gamma = \arccos(\Delta Z/\Delta S)`, which returns the
        complement of the path angle (90° in level flight) and is dimensionally
        inconsistent with its use; ECAC Doc 32 Eq. 10 states the correct
        ``atan`` form, which this implementation follows.
    """

    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    ground_speed: NDArray[np.float64]
    airspeed: NDArray[np.float64]
    heading: NDArray[np.float64]
    curvature: NDArray[np.float64]
    bank_angle: NDArray[np.float64]
    path_angle: NDArray[np.float64]

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the speed and angle profiles along the track."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_flight_path_kinematics

        return plot_flight_path_kinematics(self, ax=ax, language=check_language(language), **kwargs)


def flight_path_kinematics(
    times: NDArray[np.float64] | list[float],
    positions: NDArray[np.float64] | list[list[float]],
    *,
    gravity: float = _G0,
) -> FlightPathKinematics:
    r"""Track kinematics by central finite differences (Eq. 16-21 / Doc 32 Eq. 8-10).

    Computes, at every point of a time-stamped track, the ground speed ``V_g``
    (Eq. 16), the zero-wind airspeed ``V_A`` (Eq. 17), the heading
    :math:`\Theta = \operatorname{atan2}(\Delta X, \Delta Y)` (Eq. 19), the
    curvature :math:`K = \Delta\Theta/\Delta S` (Eq. 18), the
    bank angle :math:`\Phi = \arctan(K \cdot V_g^2/g)` (Eq. 20) and the path
    angle
    :math:`\gamma = \arctan(\Delta Z/\Delta S)` (Doc 32 Eq. 10). The
    airspeed, not the ground speed,
    selects the hemisphere (guidance §A.3.3); the guidance recommends smoothing
    radar tracks (e.g. spline resampling) before differentiating.

    :param times: Track times, in s, strictly increasing, shape ``(N,)``,
        :math:`N \ge 2`.
    :param positions: Track positions ``(x, y, z)``, in metres, shape ``(N, 3)``
        (x east, y north, z up; any consistent right-handed ground frame works,
        headings are then relative to its y axis).
    :param gravity: Acceleration of gravity ``g`` in m/s² (default 9.80665).
    :return: A :class:`FlightPathKinematics`.
    :raises ValueError: If the inputs are invalid.
    """
    t, p = _validated_track(times, positions)
    g0 = require_positive(gravity, "gravity")

    vx = np.gradient(p[:, 0], t)
    vy = np.gradient(p[:, 1], t)
    vz = np.gradient(p[:, 2], t)
    vg = np.hypot(vx, vy)
    va = np.sqrt(vx**2 + vy**2 + vz**2)
    heading = np.degrees(np.arctan2(vx, vy))
    # ΔΘ/Δt over the unwrapped heading, divided by ΔS/Δt = V_g (Eq. 18).
    dtheta_dt = np.gradient(np.unwrap(np.radians(heading)), t)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.where(vg > 0.0, dtheta_dt / np.where(vg > 0.0, vg, 1.0), 0.0)
    # K·V_g² = (ΔΘ/Δt)·V_g (Eq. 20): the product form cannot overflow through
    # the intermediate 1/V_g division when the ground speed is minute.
    bank = np.degrees(np.arctan(dtheta_dt * vg / g0))
    path_angle = np.degrees(np.arctan2(vz, vg))
    return FlightPathKinematics(
        times=t, positions=p, ground_speed=vg, airspeed=va, heading=heading,
        curvature=np.asarray(curvature, dtype=np.float64), bank_angle=bank,
        path_angle=path_angle)


# --------------------------------------------------------------------------- #
# Single event and contour (guidance §A.4-A.5 / Doc 32 §5-6)
# --------------------------------------------------------------------------- #


def _a_weighting_db(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
    """IEC 61672-1 A-weighting at the exact frequencies, in dB (Doc 32 Eq. 25)."""
    f = np.asarray(frequencies, dtype=np.float64)
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    num = f4**2 * f**4
    den = ((f**2 + f1**2) * np.sqrt((f**2 + f2**2) * (f**2 + f3**2)) * (f**2 + f4**2))
    ra = num / den
    f0 = 1000.0
    ra0 = (f4**2 * f0**4) / ((f0**2 + f1**2)
                             * np.sqrt((f0**2 + f2**2) * (f0**2 + f3**2))
                             * (f0**2 + f4**2))
    return np.asarray(20.0 * np.log10(ra / ra0), dtype=np.float64)


def _emission_angles(
    position: NDArray[np.float64],
    receivers: NDArray[np.float64],
    heading_deg: float,
    bank_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Emission azimuth ``φ``, polar angle ``θ`` and slant distance per receiver.

    The hemisphere frame follows Doc 32 Eq. 3 (x forward, y starboard, z down)
    oriented by the heading and, in turns, tilted about the forward axis by the
    bank angle (guidance §A.3.4). The frame is not pitched by the path angle:
    pitch attitude is implicit in the hemispheres (guidance §A.2.1), and the
    NORAH2 reference implementation reproduces its emission angles only with
    the level (yaw plus roll) orientation.

    :param position: Rotorcraft position ``(x, y, z)``, shape ``(3,)``.
    :param receivers: Receiver positions, shape ``(G, 3)``.
    :param heading_deg: Heading ``Θ``, in degrees.
    :param bank_deg: Bank angle ``Φ``, in degrees (positive starboard down).
    :return: ``(φ, θ, r)`` in degrees, degrees and metres, each shape ``(G,)``.
    """
    h = np.radians(heading_deg)
    fwd = np.array([np.sin(h), np.cos(h), 0.0])
    right = np.array([np.cos(h), -np.sin(h), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    # Tilt about the forward axis; at zero bank cos = 1 and sin = 0 exactly,
    # so the rotation is the identity and needs no special case.
    b = np.radians(bank_deg)
    right, down = (np.cos(b) * right + np.sin(b) * down,
                   -np.sin(b) * right + np.cos(b) * down)
    d = receivers - position[None, :]
    dist = np.sqrt(np.sum(d**2, axis=1))
    safe = np.where(dist > 0.0, dist, 1.0)
    u = d / safe[:, None]
    xb = u @ fwd
    yb = u @ right
    zb = u @ down
    theta = np.degrees(np.arccos(np.clip(xb, -1.0, 1.0)))
    phi = np.degrees(np.arctan2(yb, zb))
    return phi, theta, dist


@dataclass(frozen=True)
class RotorcraftEventResult:
    r"""A rotorcraft single-event time history at a receiver (Doc 32 §6.1).

    :ivar frequencies: Band centre frequencies, in Hz, shape ``(F,)``.
    :ivar emission_times: Emission times ``t_e``, in s, shape ``(K,)``.
    :ivar times: Recorded times :math:`t_r = t_e + r/c` (Eq. 22), in s, shape
        ``(K,)``.
    :ivar distance: Slant distance ``r`` per step, in metres, shape ``(K,)``.
    :ivar azimuth: Emission azimuth ``φ`` per step, in degrees, shape ``(K,)``.
    :ivar polar: Emission polar angle ``θ`` per step, in degrees, shape ``(K,)``.
    :ivar band_levels: Received (unweighted) band levels, in dB, shape
        ``(K, F)``.
    :ivar a_levels: A-weighted overall level ``L_A(t)`` per step, in dB(A),
        shape ``(K,)``.
    :ivar la_max: Maximum A-weighted level ``LASmax``, in dB(A).
    :ivar sel: Sound exposure level over the full history (Doc 32 Eq. 27,
        :math:`t_0 = 1` s), in dB(A). The full-history integration is the
        land-use
        planning convention of the NORAH2 reference implementation.
    :ivar sel_10db: Sound exposure level restricted to the 10 dB-down window
        about ``LASmax`` (the certification convention), in dB(A).
    :ivar pnlt: Tone-corrected perceived noise level per step, in TPNdB, shape
        ``(K,)``; ``NaN`` where undefined (zero total noisiness, or the band
        grid does not cover the 24 noy bands 50 Hz-10 kHz).
    :ivar pnltm: Maximum ``PNLT`` (with the Annex 16 bandsharing adjustment),
        in TPNdB; ``NaN`` if no step has a defined ``PNLT``.
    :ivar epnl: Effective perceived noise level (Doc 32 Eq. 28 / ICAO Annex 16),
        in EPNdB; ``NaN`` if no step has a defined ``PNLT``.
    """

    frequencies: NDArray[np.float64]
    emission_times: NDArray[np.float64]
    times: NDArray[np.float64]
    distance: NDArray[np.float64]
    azimuth: NDArray[np.float64]
    polar: NDArray[np.float64]
    band_levels: NDArray[np.float64]
    a_levels: NDArray[np.float64]
    la_max: float
    sel: float
    sel_10db: float
    pnlt: NDArray[np.float64]
    pnltm: float
    epnl: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the A-weighted level time history with its event metrics."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_rotorcraft_event

        return plot_rotorcraft_event(self, ax=ax, language=check_language(language), **kwargs)


@dataclass(frozen=True)
class RotorcraftNoiseContourResult:
    """Rotorcraft single-event noise level over a ground grid (Doc 32 §6.3).

    :ivar x: Grid x coordinates, in metres, shape ``(nx,)``.
    :ivar y: Grid y coordinates, in metres, shape ``(ny,)``.
    :ivar level: Event level over the grid, in dB(A), shape ``(ny, nx)``.
    :ivar metric: ``"exposure"`` (SEL) or ``"maximum"`` (LASmax).
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    level: NDArray[np.float64]
    metric: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot filled noise contours over the ground plane."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_rotorcraft_noise_contour

        return plot_rotorcraft_noise_contour(self, ax=ax, language=check_language(language), **kwargs)


def _per_point(
    value: float | NDArray[np.float64] | list[float] | None, n: int, name: str,
) -> NDArray[np.float64] | None:
    """A per-track-point parameter: scalar broadcast, ``(N,)`` array, or ``None``."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"'{name}' must be a scalar or match the track length.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' must be finite.")
    return arr


def _validated_track(
    times: NDArray[np.float64] | list[float],
    positions: NDArray[np.float64] | list[list[float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The validated ``(times, positions)`` track arrays."""
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(positions, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("'times' must be 1-D with at least two points.")
    if p.shape != (t.size, 3):
        raise ValueError("'positions' must have shape (N, 3) matching 'times'.")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
        raise ValueError("'times' and 'positions' must be finite.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("'times' must be strictly increasing.")
    return t, p


def _reference_distance(hemispheres: Sequence[RotorcraftHemisphere]) -> float:
    """The shared hemisphere reference distance (validated)."""
    rref = float(hemispheres[0].distance)
    for h in hemispheres[1:]:
        if float(h.distance) != rref:
            raise ValueError("All hemispheres must share one reference distance.")
    return require_positive(rref, "hemisphere distance")


@dataclass(frozen=True)
class RotorcraftAtmosphere:
    """The air a rotorcraft event propagates through (Eq. 26/27).

    The ICAO reference conditions of the hemisphere database are the defaults,
    so an event flown at those conditions needs no atmosphere at all. The Doc 29
    airport chain keeps its own
    :class:`~phonometry.aircraft.airport_noise.AerodromeAtmosphere` instead of
    sharing this one: it corrects a broadband NPD level with the impedance of
    Eq. 4-7 alone, with no band-by-band absorption to ask the humidity or the
    method about, and at a different reference temperature.

    :ivar temperature: Air temperature, in °C (default 25, ICAO reference).
    :ivar relative_humidity: Relative humidity, in % (default 70).
    :ivar pressure: Ambient pressure, in kPa (default 101.325).
    :ivar atmospheric_method: ``"iso9613"`` for the pure-tone Eq. 26/27 term
        (the guidance text), or ``"sae"`` for the SAE ARP 5534 band-integrated
        mapping used by the NORAH2 reference implementation (they agree to
        ~0.05 dB below 3.15 kHz).
    """

    temperature: float = 25.0
    relative_humidity: float = 70.0
    pressure: float = 101.325
    atmospheric_method: str = "iso9613"


@dataclass(frozen=True)
class RotorcraftGround:
    """The ground a rotorcraft event stands on (guidance §A.4.3-A.4.5).

    Flat ground at the track datum by default: the microphone height, the
    elevation of the site and the ground type feed the two-ray ground effect,
    and an optional elevation model replaces the flat plane with real terrain.

    :ivar receiver_height: Microphone height above local ground, in metres
        (default 1.2).
    :ivar ground_elevation: Ground elevation ``z`` at the receivers, in metres
        on the track datum (default 0); source and receiver heights above
        ground follow from it. A contour grid also accepts one value per grid
        point (shape ``(len(y), len(x))``) for receivers on uneven sites
        without a full elevation model.
    :ivar flow_resistivity: Ground flow resistivity ``σ`` in Pa·s/m², or a
        CNOSSOS class letter (see
        :func:`~phonometry.aircraft.rotorcraft_propagation.ground_effect_adjustment`).
        A contour grid also accepts one value per grid point (shape
        ``(len(y), len(x))``) for heterogeneous ground across the receivers
        (each receiver's two-ray model uses its local value).
    :ivar terrain: Optional digital elevation model ``(x, y, z)`` on the
        track frame (``x`` and ``y`` strictly increasing, ``z`` of shape
        ``(len(y), len(x))``, all in metres on the track datum). When given,
        every emission-receiver pair is evaluated over its sampled vertical
        section (guidance §A.4.4/A.4.5): mean-ground-plane ground effect with
        equivalent heights, and rubber-band diffraction where terrain blocks
        the line of sight; ``ground_elevation`` is then taken from the model.
        The model must cover the whole track and every receiver (fabricating
        terrain beyond its edges is refused).
    :ivar terrain_resolution: Section sampling step along the path, in
        metres (default: the elevation model's cell size; sections are capped
        at 20000 sampling intervals).
    """

    receiver_height: float = 1.2
    ground_elevation: float | NDArray[np.float64] | list[float] | list[list[float]] = 0.0
    flow_resistivity: float | str | np.floating[Any] | np.integer[Any] \
        | NDArray[np.float64] | list[float] | list[list[float]] = "G"
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | Sequence[NDArray[np.float64]] | None = None
    terrain_resolution: float | None = None


@dataclass(frozen=True)
class RotorcraftTrackState:
    """Per-point flight state of a rotorcraft track (Eq. 16-21).

    Every field left unset is derived from the track itself by
    :func:`flight_path_kinematics`; a radar-track workflow that has already
    smoothed these quantities hands them over instead. Each is a scalar
    (broadcast over the track) or an array of shape ``(N,)``.

    :ivar airspeed: Airspeed ``V_A``, in the units of the database
        ``airspeeds`` (the derived values are in m/s).
    :ivar path_angle: Path angle ``γ``, in degrees (negative descending).
    :ivar heading: Heading ``Θ``, in degrees.
    :ivar bank_angle: Bank angle ``Φ``, in degrees (positive starboard down).
    """

    airspeed: float | NDArray[np.float64] | list[float] | None = None
    path_angle: float | NDArray[np.float64] | list[float] | None = None
    heading: float | NDArray[np.float64] | list[float] | None = None
    bank_angle: float | NDArray[np.float64] | list[float] | None = None


@dataclass(frozen=True)
class FlightConditionInterpolation:
    """How a flight condition blends the database hemispheres (Eq. 3-10).

    The two settings of :func:`flight_condition_weights`, which the event and
    contour entry points hand it per track point.

    :ivar scaling_factor: Flight-condition scaling factor ``F_fc`` applied to
        the normalised path angle (default 2, the guidance's empirical value).
    :ivar triangles: Optional precomputed triangulation, shape ``(T, 3)``
        0-based indices into the database conditions (default ``None``: the
        Delaunay triangulation of the normalised conditions). See
        :func:`flight_condition_weights`.
    """

    scaling_factor: float = 2.0
    triangles: NDArray[np.int_] | list[list[int]] | None = None


#: The defaults of the event entry points, one shared instance each: the
#: bundles are frozen, so a call cannot mutate what the next one receives.
_ICAO_ATMOSPHERE = RotorcraftAtmosphere()
_FLAT_GROUND = RotorcraftGround()
_DERIVED_TRACK_STATE = RotorcraftTrackState()
_GUIDANCE_INTERPOLATION = FlightConditionInterpolation()


@dataclass(frozen=True)
class _EventSetup:
    """The validated inputs of a single-event run.

    Source database, track state, ground and atmosphere, grouped once by
    :func:`_event_setup` so the per-receiver machinery passes one object
    around instead of the Doc 32 parameter list.
    """

    hemispheres: tuple[RotorcraftHemisphere, ...]
    airspeeds: NDArray[np.float64]
    path_angles: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    speed: NDArray[np.float64]
    gamma: NDArray[np.float64]
    heading: NDArray[np.float64]
    bank: NDArray[np.float64]
    offsets: NDArray[np.float64]
    ground_elevation: float | NDArray[np.float64]
    receiver_height: float
    sigma: float | NDArray[np.float64]
    alpha: NDArray[np.float64]
    rref: float
    scaling_factor: float
    triangles: NDArray[np.int_] | list[list[int]] | None
    band_integrated: bool
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
    terrain_resolution: float


def _event_setup(
    hemispheres: Sequence[RotorcraftHemisphere],
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    times: NDArray[np.float64] | list[float],
    positions: NDArray[np.float64] | list[list[float]],
    *,
    level_offset: float | NDArray[np.float64] | list[float],
    atmosphere: RotorcraftAtmosphere,
    ground: RotorcraftGround,
    track_state: RotorcraftTrackState,
    interpolation: FlightConditionInterpolation,
) -> _EventSetup:
    """Validate the shared event/contour inputs into one :class:`_EventSetup`.

    The keyword tail mirrors the public functions one for one (the Doc 32
    single-event parameter set); both call it before adding their own
    receiver or grid arguments. Per-receiver ``ground_elevation`` and
    ``flow_resistivity`` arrays are validated here and shaped against the
    receiver grid by the caller.
    """
    freqs = _common_frequencies(hemispheres, airspeeds)
    rref = _reference_distance(hemispheres)
    t, p = _validated_track(times, positions)
    hr = require_positive(ground.receiver_height, "receiver_height")
    dem = _validated_terrain(ground.terrain)
    if dem is not None:
        _require_dem_coverage(dem, p[:, 0], p[:, 1], "track")
    spacing = _section_spacing(dem, ground.terrain_resolution)
    sigma = _setup_resistivity(ground.flow_resistivity, dem)
    elevation = _setup_ground_elevation(ground.ground_elevation, dem)
    method = require_choice(
        atmosphere.atmospheric_method, "atmospheric_method", ("iso9613", "sae"))
    spd, gam, hdg, bank = _resolved_track_state(t, p, track_state)
    off = _per_point(level_offset, t.size, "level_offset")
    offsets = off if off is not None else np.zeros(t.size)
    alpha = _absorption_coefficient(
        freqs, atmosphere.temperature, atmosphere.relative_humidity,
        atmosphere.pressure)
    return _EventSetup(
        hemispheres=tuple(hemispheres),
        airspeeds=np.atleast_1d(np.asarray(airspeeds, dtype=np.float64)),
        path_angles=np.atleast_1d(np.asarray(path_angles, dtype=np.float64)),
        frequencies=freqs, times=t, positions=p, speed=spd, gamma=gam,
        heading=hdg, bank=bank, offsets=offsets,
        ground_elevation=elevation, receiver_height=hr,
        sigma=sigma, alpha=alpha, rref=rref,
        scaling_factor=interpolation.scaling_factor,
        triangles=interpolation.triangles, band_integrated=method == "sae",
        terrain=dem, terrain_resolution=spacing)


def _section_spacing(
    dem: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None,
    terrain_resolution: float | None,
) -> float:
    """The section sampling step: the given resolution or the model's cell size."""
    if dem is None:
        return 0.0
    if terrain_resolution is not None:
        return require_positive(terrain_resolution, "terrain_resolution")
    return float(min(np.min(np.diff(dem[0])), np.min(np.diff(dem[1]))))


def _setup_resistivity(
    flow_resistivity: float | str | np.floating[Any] | np.integer[Any]
    | NDArray[np.float64] | list[float] | list[list[float]],
    dem: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None,
) -> float | NDArray[np.float64]:
    """The scalar (or per-receiver) flow resistivity of an event run."""
    if isinstance(flow_resistivity, str):
        return _resolve_flow_resistivity(flow_resistivity)
    if np.ndim(flow_resistivity) == 0:
        return _resolve_flow_resistivity(float(np.asarray(flow_resistivity)))
    if dem is not None:
        raise ValueError("With 'terrain', 'flow_resistivity' must be a single "
                         "value or class (per-path maps are not supported).")
    return require_positive_array(
        np.asarray(flow_resistivity, dtype=np.float64).ravel(), "flow_resistivity")


def _setup_ground_elevation(
    ground_elevation: float | NDArray[np.float64] | list[float] | list[list[float]],
    dem: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None,
) -> float | NDArray[np.float64]:
    """The scalar (or per-receiver) ground elevation of an event run."""
    if np.isscalar(ground_elevation):
        if not np.isfinite(ground_elevation):
            raise ValueError("'ground_elevation' must be finite.")
        return float(ground_elevation)  # type: ignore[arg-type]
    arr = np.asarray(ground_elevation, dtype=np.float64).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError("'ground_elevation' must be finite.")
    if dem is not None:
        raise ValueError("With 'terrain', 'ground_elevation' comes from the "
                         "elevation model and must be left scalar.")
    return arr


def _require_dem_coverage(
    dem: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    x: NDArray[np.float64], y: NDArray[np.float64], what: str,
) -> None:
    """Reject points outside the elevation model's horizontal extent.

    Every vertical section joins a track point to a receiver, so covering
    both keeps all sampled points inside the model; silently clamping to the
    edge would fabricate terrain instead.
    """
    tx, ty, _ = dem
    if (np.min(x) < tx[0] or np.max(x) > tx[-1]
            or np.min(y) < ty[0] or np.max(y) > ty[-1]):
        raise ValueError(f"'terrain' must cover the whole {what} (x in "
                         f"[{tx[0]:g}, {tx[-1]:g}], y in [{ty[0]:g}, {ty[-1]:g}]).")


def _validated_terrain(
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | Sequence[NDArray[np.float64]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    """The validated ``(x, y, z)`` digital elevation model, or ``None``."""
    if terrain is None:
        return None
    if len(terrain) != 3:
        raise ValueError("'terrain' must be an (x, y, z) elevation model.")
    tx = np.asarray(terrain[0], dtype=np.float64).ravel()
    ty = np.asarray(terrain[1], dtype=np.float64).ravel()
    tz = np.asarray(terrain[2], dtype=np.float64)
    if tx.size < 2 or ty.size < 2 or np.any(np.diff(tx) <= 0) or np.any(np.diff(ty) <= 0):
        raise ValueError("'terrain' x and y must be strictly increasing with >= 2 points.")
    if tz.shape != (ty.size, tx.size) or not np.all(np.isfinite(tz)):
        raise ValueError("'terrain' z must be finite with shape (len(y), len(x)).")
    if not (np.all(np.isfinite(tx)) and np.all(np.isfinite(ty))):
        raise ValueError("'terrain' coordinates must be finite.")
    return tx, ty, tz


def _dem_height(
    dem: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    x: NDArray[np.float64], y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bilinear elevation lookup.

    Coverage is validated upstream (:func:`_require_dem_coverage`); the edge
    clamp only guards floating-point round-off exactly on the boundary.
    """
    tx, ty, tz = dem
    cx = np.clip(np.searchsorted(tx, x) - 1, 0, tx.size - 2)
    cy = np.clip(np.searchsorted(ty, y) - 1, 0, ty.size - 2)
    wx = np.clip((x - tx[cx]) / (tx[cx + 1] - tx[cx]), 0.0, 1.0)
    wy = np.clip((y - ty[cy]) / (ty[cy + 1] - ty[cy]), 0.0, 1.0)
    return np.asarray(
        (1 - wy) * (1 - wx) * tz[cy, cx] + (1 - wy) * wx * tz[cy, cx + 1]
        + wy * (1 - wx) * tz[cy + 1, cx] + wy * wx * tz[cy + 1, cx + 1],
        dtype=np.float64)


def _event_histories(
    setup: _EventSetup,
    receivers: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Received time histories: ``(t_rec, L_A)`` shape ``(K, G)`` and spectra.

    One vectorised pass per emission step over all receivers (Eq. 1/22/23).
    The unweighted band levels of the first receiver come back as well, shape
    ``(K, F)`` (the single-receiver event needs them for the perceived-noise
    metrics; the cost is negligible next to the ``(K, G)`` histories).
    """
    from .atmospheric_absorption import _sae_band

    freqs = setup.frequencies
    positions = setup.positions
    aw = _a_weighting_db(freqs)
    n_k = setup.times.size
    n_g = receivers.shape[0]
    trec = np.empty((n_k, n_g), dtype=np.float64)
    la = np.empty((n_k, n_g), dtype=np.float64)
    spectra = np.empty((n_k, freqs.size), dtype=np.float64)
    ref_band = _sae_band(setup.alpha * setup.rref)   # only used when band_integrated
    weight_cache: dict[tuple[float, float], list[tuple[int, float]]] = {}

    for k in range(n_k):
        key = (float(setup.speed[k]), float(setup.gamma[k]))
        weights = weight_cache.get(key)
        if weights is None:
            weights = flight_condition_weights(
                setup.airspeeds, setup.path_angles, key[0], key[1],
                scaling_factor=setup.scaling_factor, triangles=setup.triangles)
            weight_cache[key] = weights
        phi, theta, dist = _emission_angles(
            positions[k], receivers, setup.heading[k], setup.bank[k])
        dist = np.maximum(dist, 1e-6)
        energy = np.zeros((n_g, freqs.size), dtype=np.float64)
        for j, w in weights:
            energy += w * 10.0 ** (_source_levels(setup.hemispheres[j], phi, theta) / 10.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            src = 10.0 * np.log10(energy)
        dls = -20.0 * np.log10(dist / setup.rref)
        if setup.band_integrated:
            dla = -(_sae_band(setup.alpha[None, :] * dist[:, None]) - ref_band[None, :])
        else:
            dla = -setup.alpha[None, :] * (dist[:, None] - setup.rref)
        dp = np.hypot(receivers[:, 0] - positions[k, 0], receivers[:, 1] - positions[k, 1])
        if setup.terrain is not None:
            dlg = _terrain_adjustments(setup, setup.terrain, positions[k], receivers, dp)
        else:
            hs = float(positions[k, 2]) - setup.ground_elevation
            dlg = _ground_effect(freqs, hs, setup.receiver_height, dp, setup.sigma)
        spl = src + setup.offsets[k] + dls[:, None] + dla + dlg
        with np.errstate(divide="ignore", invalid="ignore"):
            la[k] = 10.0 * np.log10(np.nansum(10.0 ** ((spl + aw[None, :]) / 10.0), axis=1))
        trec[k] = setup.times[k] + dist / _C
        spectra[k] = spl[0]
    return trec, la, spectra


#: Upper bound on samples per vertical section (a 200 km path at the 10 m
#: default resolution; guards degenerate terrain_resolution requests).
_MAX_SECTION_SAMPLES = 20001


def _terrain_adjustments(
    setup: _EventSetup,
    dem: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    position: NDArray[np.float64],
    receivers: NDArray[np.float64],
    dp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Ground-and-screening adjustments over the elevation model, ``(G, F)``.

    For every receiver the vertical section from the emission point to the
    microphone is sampled from the elevation model at the configured
    resolution and evaluated by the §A.4.4/A.4.5 machinery (mean ground
    plane, equivalent heights, rubber-band diffraction). This is a per-pair
    scalar path: with an elevation model the cost grows with track points
    times receivers.
    """
    freqs = setup.frequencies
    sigma = float(np.atleast_1d(np.asarray(setup.sigma, dtype=np.float64))[0])
    out = np.empty((receivers.shape[0], freqs.size), dtype=np.float64)
    sx, sy, sz = float(position[0]), float(position[1]), float(position[2])
    for i in range(receivers.shape[0]):
        span = float(dp[i])
        n = min(max(2, math.ceil(span / setup.terrain_resolution) + 1),
                _MAX_SECTION_SAMPLES)
        t = np.linspace(0.0, 1.0, n)
        px = sx + (receivers[i, 0] - sx) * t
        py = sy + (receivers[i, 1] - sy) * t
        pz = _dem_height(dem, px, py)
        d = span * t
        if span <= 1e-6:                      # receiver under the source
            hs = sz - float(pz[0])
            out[i] = _ground_effect(freqs, hs, setup.receiver_height,
                                    np.asarray([0.0]), sigma)[0]
            continue
        sigma_seg = np.full(n - 1, sigma)
        adj, _, _, _ = _screening_core(
            freqs, (0.0, sz), (span, float(receivers[i, 2])), d, pz, sigma_seg)
        out[i] = adj
    return out


def _exposure_level(
    la: NDArray[np.float64], trec: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""``SEL`` (Doc 32 Eq. 27, :math:`t_0 = 1` s) per receiver, ``(K, G)``.

    Trapezoidal integration of the received A-weighted energy over recorded
    time, the integration the NORAH2 reference implementation applies over the
    full history.
    """
    energy = np.trapezoid(10.0 ** (la / 10.0), trec, axis=0)
    with np.errstate(divide="ignore"):
        return np.asarray(10.0 * np.log10(energy), dtype=np.float64)


def _event_metrics(
    freqs: NDArray[np.float64],
    trec: NDArray[np.float64],
    la: NDArray[np.float64],
    spectra: NDArray[np.float64],
) -> tuple[float, float, float, NDArray[np.float64], float, float]:
    """The single-receiver metrics ``(LASmax, SEL, SEL_10dB, PNLT, PNLTM, EPNL)``."""
    from .certification import (
        _ten_db_down_limits,
        epnl_from_pnlt,
        perceived_noise_level,
        tone_correction,
    )

    la_max = float(np.max(la))
    sel = float(_exposure_level(la[:, None], trec[:, None])[0])
    kf, kl = _ten_db_down_limits(la, la_max - 10.0)
    if kl > kf:
        sel_10db = float(_exposure_level(la[kf:kl + 1, None], trec[kf:kl + 1, None])[0])
    else:  # degenerate single-record window
        sel_10db = float(la[kf] + 10.0 * np.log10(np.gradient(trec)[kf]))

    pnlt = np.full(la.shape, np.nan)
    tcs = np.zeros(la.shape)
    noy = _noy_band_indices(freqs)
    if noy is not None:
        for k in range(la.size):
            row = spectra[k, noy]
            if not np.all(np.isfinite(row)):
                continue
            pnl = perceived_noise_level(row)
            if pnl <= 0.0:      # zero total noisiness: PNLT undefined
                continue
            # start_band=0: the slope analysis starts at 50 Hz for helicopters
            # (ICAO Annex 16 App. 2 §4.3.1 Step 1), not the aeroplane 80 Hz.
            tcs[k] = tone_correction(row, start_band=0)
            pnlt[k] = pnl + tcs[k]
    valid = np.isfinite(pnlt)
    if np.any(valid):
        dt = np.gradient(trec)
        epnl, pnltm, _, _ = epnl_from_pnlt(
            pnlt[valid], dt[valid], tone_corrections=tcs[valid])
    else:
        epnl = pnltm = float("nan")
    return la_max, sel, sel_10db, pnlt, pnltm, epnl


def _noy_band_indices(frequencies: NDArray[np.float64]) -> NDArray[np.intp] | None:
    """Indices of the 24 noy bands (50 Hz-10 kHz) in a band grid, or ``None``."""
    from .certification import NOY_BANDS

    idx = []
    for band in NOY_BANDS:
        hits = np.nonzero(np.isclose(frequencies, band, rtol=0.06))[0]
        if hits.size != 1:
            return None
        idx.append(int(hits[0]))
    return np.asarray(idx, dtype=np.intp)


def _resolved_track_state(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    state: RotorcraftTrackState,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-point ``(V_A, γ, Θ, Φ)``: explicit overrides, else derived (Eq. 16-21)."""
    n = times.size
    spd = _per_point(state.airspeed, n, "airspeed")
    gam = _per_point(state.path_angle, n, "path_angle")
    hdg = _per_point(state.heading, n, "heading")
    bank = _per_point(state.bank_angle, n, "bank_angle")
    if spd is None or gam is None or hdg is None or bank is None:
        kin = flight_path_kinematics(times, positions)
        spd = kin.airspeed if spd is None else spd
        gam = kin.path_angle if gam is None else gam
        hdg = kin.heading if hdg is None else hdg
        bank = kin.bank_angle if bank is None else bank
    return spd, gam, hdg, bank


def rotorcraft_event_level(
    hemispheres: Sequence[RotorcraftHemisphere],
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    times: NDArray[np.float64] | list[float],
    positions: NDArray[np.float64] | list[list[float]],
    receiver: tuple[float, float] | NDArray[np.float64] | list[float],
    *,
    level_offset: float | NDArray[np.float64] | list[float] = 0.0,
    atmosphere: RotorcraftAtmosphere = _ICAO_ATMOSPHERE,
    ground: RotorcraftGround = _FLAT_GROUND,
    track_state: RotorcraftTrackState = _DERIVED_TRACK_STATE,
    interpolation: FlightConditionInterpolation = _GUIDANCE_INTERPOLATION,
) -> RotorcraftEventResult:
    r"""Rotorcraft single-event level at a receiver (Doc 32 §6.1 / guidance §A.5.1).

    For every track point the flight condition selects (or blends, Eq. 3-10)
    the hemispheres, the emission angles address the source level (Eq. 13-15)
    and the propagation adjustment
    :math:`\Delta L_p = \Delta L_s + \Delta L_a + \Delta L_g` (Eq. 23-35)
    places
    it at the receiver. The received one-third-octave history is expressed at
    recorded time :math:`t_r = t_e + r/c` (Eq. 22) and integrated into
    ``LASmax``,
    ``SEL`` (Doc 32 Eq. 27) and ``EPNL`` (Doc 32 Eq. 28, ICAO Annex 16 App. 2,
    reusing
    :func:`~phonometry.aircraft.certification.epnl_from_pnlt`).

    The flight condition per point comes from the ``track_state`` overrides
    when given (e.g. the smoothed values of a radar-track workflow),
    otherwise from :func:`flight_path_kinematics` on the track itself, in which
    case the database ``airspeeds`` must be in m/s. The hemisphere frame is
    oriented by the heading and tilted by the bank angle in turns (guidance
    §A.3.4); pitch attitude is implicit in the hemispheres.

    :param hemispheres: The database hemispheres, one per flight condition.
    :param airspeeds: Database airspeeds ``V_j``, shape ``(J,)`` (same units as
        the ``airspeed`` values used for selection).
    :param path_angles: Database path angles ``γ_j``, in degrees, shape ``(J,)``.
    :param times: Track times, in s, strictly increasing, shape ``(N,)``.
    :param positions: Track positions ``(x, y, z)``, in metres, shape ``(N, 3)``
        (z up, above the ground elevation datum).
    :param receiver: Receiver ground position ``(x, y)``, in metres.
    :param level_offset: Source-level offset ``ΔEPNL`` added to the hemisphere
        levels (Eq. 2 class substitution), in dB (default 0). Scalar or per
        track point, shape ``(N,)``: Chapter-8 substitutions correct climb,
        level and descent conditions with different certification levels.
    :param atmosphere: The air the event propagates through, a
        :class:`RotorcraftAtmosphere` (default: the ICAO reference conditions
        of the database).
    :param ground: The ground under the event, a :class:`RotorcraftGround`
        (default: flat ground at the track datum, CNOSSOS class ``"G"``, a
        1.2 m microphone). A single receiver takes its scalar fields only;
        the per-grid-point arrays are for the contour.
    :param track_state: Per-point airspeed, path angle, heading and bank
        angle, a :class:`RotorcraftTrackState` (default: all derived from the
        track by :func:`flight_path_kinematics`).
    :param interpolation: How the flight condition blends the database
        hemispheres, a :class:`FlightConditionInterpolation` (default:
        ``F_fc = 2`` over the Delaunay triangulation).
    :return: A :class:`RotorcraftEventResult`.
    :raises ValueError: If the inputs are invalid.
    """
    setup = _event_setup(
        hemispheres, airspeeds, path_angles, times, positions,
        level_offset=level_offset, atmosphere=atmosphere, ground=ground,
        track_state=track_state, interpolation=interpolation)
    rx = np.asarray(receiver, dtype=np.float64).ravel()
    if rx.size != 2 or not np.all(np.isfinite(rx)):
        raise ValueError("'receiver' must be a finite (x, y) ground position.")
    if not (np.isscalar(setup.sigma) and np.isscalar(setup.ground_elevation)):
        raise ValueError("A single receiver takes scalar 'flow_resistivity' and "
                         "'ground_elevation'; arrays are for the contour grid.")
    if setup.terrain is not None:
        _require_dem_coverage(setup.terrain, rx[:1], rx[1:2], "receiver")
        local = float(_dem_height(setup.terrain, rx[:1], rx[1:2])[0])
    else:
        local = float(np.atleast_1d(setup.ground_elevation)[0])
    receivers = np.array([[rx[0], rx[1], local + setup.receiver_height]])
    trec, la, spectra = _event_histories(setup, receivers)
    phi, theta, dist = _track_emission_geometry(
        setup.positions, receivers[0], setup.heading, setup.bank)
    la_max, sel, sel_10db, pnlt, pnltm, epnl = _event_metrics(
        setup.frequencies, trec[:, 0], la[:, 0], spectra)
    return RotorcraftEventResult(
        frequencies=setup.frequencies, emission_times=setup.times,
        times=trec[:, 0], distance=dist, azimuth=phi, polar=theta,
        band_levels=spectra, a_levels=la[:, 0], la_max=la_max, sel=sel,
        sel_10db=sel_10db, pnlt=pnlt, pnltm=pnltm, epnl=epnl)


def _track_emission_geometry(
    positions: NDArray[np.float64], receiver: NDArray[np.float64],
    heading: NDArray[np.float64], bank: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Emission ``(φ, θ, r)`` of every track point towards one receiver."""
    n = positions.shape[0]
    phi = np.empty(n)
    theta = np.empty(n)
    dist = np.empty(n)
    rx = receiver[None, :]
    for k in range(n):
        f, th, d = _emission_angles(positions[k], rx, heading[k], bank[k])
        phi[k], theta[k], dist[k] = f[0], th[0], d[0]
    return phi, theta, dist


def rotorcraft_noise_contour(
    hemispheres: Sequence[RotorcraftHemisphere],
    airspeeds: NDArray[np.float64] | list[float],
    path_angles: NDArray[np.float64] | list[float],
    times: NDArray[np.float64] | list[float],
    positions: NDArray[np.float64] | list[list[float]],
    *,
    x: NDArray[np.float64] | list[float],
    y: NDArray[np.float64] | list[float],
    metric: str = "exposure",
    level_offset: float | NDArray[np.float64] | list[float] = 0.0,
    atmosphere: RotorcraftAtmosphere = _ICAO_ATMOSPHERE,
    ground: RotorcraftGround = _FLAT_GROUND,
    track_state: RotorcraftTrackState = _DERIVED_TRACK_STATE,
    interpolation: FlightConditionInterpolation = _GUIDANCE_INTERPOLATION,
) -> RotorcraftNoiseContourResult:
    """Rotorcraft single-event level over a ground grid (Doc 32 §6.3).

    Evaluates the event of :func:`rotorcraft_event_level` at every grid point
    ``(xi, yj)`` in one vectorised pass per emission step, and reduces the
    received histories to the exposure (``SEL``, Doc 32 Eq. 27) or maximum
    (``LASmax``) level.

    :param hemispheres: The database hemispheres, one per flight condition.
    :param airspeeds: Database airspeeds ``V_j``, shape ``(J,)``.
    :param path_angles: Database path angles ``γ_j``, in degrees, shape ``(J,)``.
    :param times: Track times, in s, strictly increasing, shape ``(N,)``.
    :param positions: Track positions ``(x, y, z)``, in metres, shape ``(N, 3)``.
    :param x: Grid x coordinates, in metres (at least 2).
    :param y: Grid y coordinates, in metres (at least 2).
    :param metric: ``"exposure"`` (SEL) or ``"maximum"`` (LASmax).
    :param level_offset: Source-level offset ``ΔEPNL`` (Eq. 2), in dB, scalar
        or per track point.
    :param atmosphere: The air the event propagates through, a
        :class:`RotorcraftAtmosphere`.
    :param ground: The ground under the grid, a :class:`RotorcraftGround`. Its
        ``ground_elevation`` and ``flow_resistivity`` also accept one value per
        grid point (shape ``(len(y), len(x))``), and its ``terrain`` model must
        cover the whole track and grid: every emission-receiver pair then
        samples its own vertical section, so the cost grows with track points
        times grid points; keep contour grids modest with terrain.
    :param track_state: Per-point airspeed, path angle, heading and bank angle
        (see :func:`rotorcraft_event_level`).
    :param interpolation: How the flight condition blends the database
        hemispheres, a :class:`FlightConditionInterpolation`.
    :return: A :class:`RotorcraftNoiseContourResult`.
    :raises ValueError: If the inputs are invalid.
    """
    setup = _event_setup(
        hemispheres, airspeeds, path_angles, times, positions,
        level_offset=level_offset, atmosphere=atmosphere, ground=ground,
        track_state=track_state, interpolation=interpolation)
    gx = np.asarray(x, dtype=np.float64).ravel()
    gy = np.asarray(y, dtype=np.float64).ravel()
    if gx.size < 2 or gy.size < 2 or not (np.all(np.isfinite(gx)) and np.all(np.isfinite(gy))):
        raise ValueError("'x' and 'y' must each be finite with at least two grid points.")
    key = require_choice(metric, "metric", ("exposure", "maximum"))
    xx, yy = np.meshgrid(gx, gy)
    n_g = xx.size
    for name, value in (("flow_resistivity", ground.flow_resistivity),
                        ("ground_elevation", ground.ground_elevation)):
        if np.isscalar(value) or isinstance(value, str):
            continue
        shape = np.asarray(value).shape
        if shape not in ((n_g,), (gy.size, gx.size)):
            raise ValueError(f"A per-receiver '{name}' must carry one value per grid "
                             "point, shape (len(y), len(x)).")
    if setup.terrain is not None:
        _require_dem_coverage(setup.terrain, gx, gy, "receiver grid")
        local = _dem_height(setup.terrain, xx.ravel(), yy.ravel())
    elif np.isscalar(setup.ground_elevation):
        local = np.full(n_g, float(np.atleast_1d(setup.ground_elevation)[0]))
    else:
        local = np.asarray(setup.ground_elevation, dtype=np.float64)
    receivers = np.column_stack([
        xx.ravel(), yy.ravel(), local + setup.receiver_height])
    trec, la, _ = _event_histories(setup, receivers)
    if key == "exposure":
        level = _exposure_level(la, trec)
    else:
        level = np.max(la, axis=0)
    return RotorcraftNoiseContourResult(
        x=gx, y=gy, level=level.reshape(gy.size, gx.size), metric=key)
