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
absorption, ground effect and, later, shielding.

This module provides the source and propagation primitives and the single-event
method built on them (clean-room, from the NORAH2 guidance SC01.D1.5d, the basis
of ECAC Doc 32):

* :func:`hemisphere_source_level` -- the interpolated source level ``L(fc, φ, θ)``
  from a :class:`RotorcraftHemisphere`, bilinear over the 10° grid (Eq. 13) with
  nearest-bin fill outside the measured coverage (Eq. 14/15).
* :func:`spherical_spreading_adjustment` --
  :math:`\Delta L_s = -20 \cdot \log_{10}(r/60)` (Eq. 24).
* :func:`atmospheric_adjustment` --
  :math:`\Delta L_a = -\alpha(f) \cdot (r - 60)` with the ISO 9613-1
  pure-tone coefficient (Eq. 26/27), reusing
  :func:`~phonometry.environment.propagation.air_absorption.air_attenuation`.
* :func:`ground_effect_adjustment` -- ``ΔLg`` for a point source over an impedance
  plane (Chien-Soroka, Eq. 28-35) with the Delany-Bazley one-parameter impedance
  and the CNOSSOS flow-resistivity classes.
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
guidance (EASA.2020.FC.06 SC01.D1.5d), §A.3-A.5. The atmospheric term is validated
against the guidance Table 4 (one-third-octave attenuation per km at ICAO
reference conditions); the event chain is validated end to end against the NORAH2
reference implementation outputs for the ARP verification cases (angles, retarded
times, hemisphere selection, per-step levels and event metrics).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_choice,
    require_non_negative,
    require_positive,
    require_positive_array,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Hemisphere reference distance, in metres (ECAC Doc 32 §A.3.2 / Eq. 24).
_RH = 60.0
#: Speed of sound at ICAO reference conditions, in m/s (Eq. 22).
_C = 346.1
#: Standard acceleration of gravity, in m/s² (Eq. 20).
_G0 = 9.80665
#: CNOSSOS-EU flow resistivity by ground class, in Pa·s/m² (Doc 32 Table 5 / D1.5d Table 5).
_FLOW_RESISTIVITY = {
    "A": 12.5e3,   # very soft (snow, moss)
    "B": 31.5e3,   # soft forest floor
    "C": 80.0e3,   # uncompacted loose ground (turf, grass)
    "D": 200.0e3,  # normal uncompacted ground (pasture)
    "E": 500.0e3,  # compacted field and gravel
    "F": 2.0e6,    # compacted dense ground (gravel road, car park)
    "G": 20.0e6,   # hard surfaces (asphalt, concrete)
    "H": 200.0e6,  # very hard and dense (dense asphalt, water)
}


def spherical_spreading_adjustment(
    distance: float, *, reference_distance: float = _RH,
) -> float:
    r"""Spherical-spreading adjustment ``ΔLs`` of the hemisphere level (Eq. 24).

    The hemisphere levels are defined at the reference distance ``rh`` (60 m in
    the standard database), so at slant distance ``r`` the geometric spreading
    adjustment is :math:`\Delta L_s = -20 \cdot \log_{10}(r/r_h)`.

    :param distance: Slant distance ``r`` from the rotorcraft to the observer, in
        metres (``> 0``).
    :param reference_distance: Hemisphere reference distance ``rh``, in metres
        (default 60). Pass :attr:`RotorcraftHemisphere.distance` when the data
        uses a non-standard polar distance (e.g. 70 m hover rings).
    :return: The spreading adjustment ``ΔLs``, in dB (added to the level).
    :raises ValueError: If a distance is not strictly positive.
    """
    r = require_positive(distance, "distance")
    rh = require_positive(reference_distance, "reference_distance")
    return float(-20.0 * np.log10(r / rh))


def atmospheric_adjustment(
    frequencies: NDArray[np.float64] | list[float],
    distance: float,
    *,
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
    reference_distance: float = _RH,
) -> NDArray[np.float64]:
    r"""Atmospheric-absorption adjustment ``ΔLa`` of the hemisphere level (Eq. 26/27).

    The hemisphere already includes absorption out to the reference distance
    ``rh``, so only the excess path :math:`r - r_h` is corrected:
    :math:`\Delta L_a = -\alpha(f) \cdot (r - r_h)` with the ISO 9613-1
    pure-tone coefficient ``α``
    evaluated at the exact band centre (Eq. 26/27, ICAO reference atmosphere by
    default). This matches the guidance Eq. 27 to 0.02 dB/km and the NORAH2
    reference implementation. The guidance's alternative per-band mapping (SAE
    method by Rickley et al., its Table 4) coincides below 3.15 kHz and deviates
    by up to 2.2 dB/km at 8-10 kHz; for a path-dependent band mapping use
    :func:`~phonometry.aircraft.atmospheric_absorption.sae_band_attenuation`.

    .. note::
        The printed guidance Eq. 27 pairs the coefficient ``6.6928e-6`` with
        :math:`f_{r,O} = 630.7` Hz, which evaluates to nonsense (14.3 dB/km
        at 500 Hz against Table 4's 3.1). The physically correct pairing
        (``6.6928e-6``
        with the oxygen relaxation frequency, ``1.3415e-6`` with 630.7 Hz)
        reproduces Table 4 and this implementation to 0.02 dB/km; do not
        "fix" the code by transcribing the typo.

    Bands below the 50 Hz floor of the ISO 9613-1 tabulation (the NORAH grid
    starts at 10 Hz) use the same analytic formulas; the advisory out-of-range
    warning is suppressed because ``α`` is negligible there (Table 4 lists
    0.0 dB/km for every band up to 50 Hz). The suppression only applies while
    every band stays within the 10 kHz top of the NORAH grid; above that the
    advisory warning propagates, since ``α`` is large and extrapolated.

    :param frequencies: One-third-octave-band centre frequencies, in Hz.
    :param distance: Slant distance ``r``, in metres (``> 0``; below ``rh``
        the adjustment is a small positive value, i.e. less absorption than the
        reference path).
    :param temperature: Air temperature, in °C (default 25 °C, ICAO reference).
    :param relative_humidity: Relative humidity, in % (default 70 %).
    :param pressure: Ambient pressure, in kPa (default 101.325).
    :param reference_distance: Hemisphere reference distance ``rh``, in metres
        (default 60). Pass :attr:`RotorcraftHemisphere.distance` when the data
        uses a non-standard polar distance.
    :return: The adjustment ``ΔLa`` per band, in dB (added to the level,
        :math:`\le 0` for :math:`r \ge r_h`).
    :raises ValueError: If a distance is not strictly positive.
    """
    f = require_positive_array(frequencies, "frequencies")
    r = require_positive(distance, "distance")
    rh = require_positive(reference_distance, "reference_distance")
    alpha = _absorption_coefficient(f, temperature, relative_humidity, pressure)
    return np.asarray(-alpha * (r - rh), dtype=np.float64)


def _delany_bazley_impedance(
    frequencies: NDArray[np.float64], flow_resistivity: float) -> NDArray[np.complex128]:
    """Delany-Bazley one-parameter normalised surface impedance ``Zs`` (Eq. 35)."""
    ratio = frequencies / flow_resistivity
    real = 1.0 + 0.0511 * ratio ** (-0.754)
    imag = 0.0768 * ratio ** (-0.732)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def ground_effect_adjustment(
    frequencies: NDArray[np.float64] | list[float],
    source_height: float,
    receiver_height: float,
    horizontal_distance: float,
    *,
    flow_resistivity: float | str = "G",
) -> NDArray[np.float64]:
    r"""Ground-effect adjustment ``ΔLg`` over an impedance plane (Eq. 28-35).

    A point source over a locally-reacting impedance ground produces interference
    between the direct and reflected rays. With the spherical reflection
    coefficient ``Q`` (Chien-Soroka) and the Delany-Bazley impedance,
    :math:`\Delta L_g = 10 \cdot \log_{10}\{1 + (r_1/r_2)^2 \lvert Q \rvert^2
    + 2 (r_1/r_2) \lvert Q \rvert \cdot I\}` (Eq. 29), where ``I``
    (Eq. 30) is the in-band interference factor.

    :param frequencies: One-third-octave-band centre frequencies, in Hz.
    :param source_height: Source height above the ground ``hs``, in metres
        (clamped to ``>= 0.1``).
    :param receiver_height: Receiver height above the ground ``hr``, in metres
        (clamped to ``>= 0.1``).
    :param horizontal_distance: Horizontal source-receiver distance ``dp``, in
        metres (``> 0``).
    :param flow_resistivity: Ground flow resistivity ``σ`` in Pa·s/m², or a
        CNOSSOS class letter ``"A"``-``"H"``. The default ``"G"`` (20e6, hard
        surfaces) is the CNOSSOS class covering the paved surroundings typical
        of heliports; the guidance's own suggestions, concrete
        :math:`\sigma = 65 \times 10^6` for city areas and grass
        :math:`\sigma = 200 \times 10^3` for rural areas (§A.4.3), can be
        passed as numeric values.
    :return: The adjustment ``ΔLg`` per band, in dB (added to the level).
    :raises ValueError: If the inputs are invalid.
    """
    f = require_positive_array(frequencies, "frequencies")
    hs = require_non_negative(source_height, "source_height")
    hr = require_non_negative(receiver_height, "receiver_height")
    dp = require_positive(horizontal_distance, "horizontal_distance")
    sigma = _resolve_flow_resistivity(flow_resistivity)
    grid = _ground_effect(f, hs, hr, np.asarray([dp], dtype=np.float64), sigma)
    return np.asarray(grid[0], dtype=np.float64)


def _resolve_flow_resistivity(flow_resistivity: float | str) -> float:
    """The flow resistivity ``σ`` in Pa·s/m² from a value or CNOSSOS class letter."""
    if isinstance(flow_resistivity, str):
        key = flow_resistivity.strip().upper()
        if key not in _FLOW_RESISTIVITY:
            valid = ", ".join(sorted(_FLOW_RESISTIVITY))
            raise ValueError(
                f"'flow_resistivity' class must be one of {valid}, got {flow_resistivity!r}.")
        return _FLOW_RESISTIVITY[key]
    return require_positive(flow_resistivity, "flow_resistivity")


def _ground_effect(
    frequencies: NDArray[np.float64],
    source_height: float | NDArray[np.float64],
    receiver_height: float,
    horizontal_distances: NDArray[np.float64],
    sigma: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """``ΔLg`` (Eq. 28-35) for one emission over many receivers, shape ``(G, F)``.

    The validated core of :func:`ground_effect_adjustment`, broadcast over a
    vector of horizontal distances (grid receivers); the source height above
    the local ground and the flow resistivity may vary per receiver as well
    (scalars broadcast). Heights below 0.1 m clamp to 0.1 m (guidance §A.4.4);
    a zero horizontal distance (receiver directly below the source) is
    admitted, the two-ray geometry stays finite there.
    """
    from scipy.special import wofz

    f = frequencies[None, :]
    hs = np.maximum(np.atleast_1d(np.asarray(source_height, dtype=np.float64)), 0.1)[:, None]
    hr = max(receiver_height, 0.1)
    dp = horizontal_distances[:, None]

    r1 = np.hypot(dp, hs - hr)                    # direct path
    r2 = np.hypot(dp, hs + hr)                    # reflected path (image source)
    dr = r2 - r1                                  # path-length difference ΔR
    cos_xi = (hs + hr) / r2                       # incidence angle from the normal
    k = 2.0 * np.pi * f / _C

    sig = np.atleast_1d(np.asarray(sigma, dtype=np.float64))[:, None]
    zs = _delany_bazley_impedance_grid(np.asarray(frequencies), sig)
    rp = (zs * cos_xi - 1.0) / (zs * cos_xi + 1.0)
    d_num = (1.0 + 1j) / 2.0 * np.sqrt(k * r2) * (1.0 / zs + cos_xi)
    f_loss = 1.0 + 1j * d_num * np.sqrt(np.pi) * wofz(d_num)   # F(d) = 1 + i·d·√π·w(d)
    q = rp + (1.0 - rp) * f_loss
    q_mag = np.abs(q)
    psi = np.angle(q)

    a = 0.727 * f * dr / _C
    sinc = np.sinc(a / np.pi)                                 # sin(a)/a, = 1 at a = 0
    inter = sinc * np.cos(6.325 * f * dr / _C + psi)          # I (Eq. 30)
    ratio = r1 / r2
    arg = 1.0 + ratio**2 * q_mag**2 + 2.0 * ratio * q_mag * inter
    return np.asarray(10.0 * np.log10(np.maximum(arg, 1e-15)), dtype=np.float64)


def _delany_bazley_impedance_grid(
    frequencies: NDArray[np.float64], sigma_column: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """``Zs`` (Eq. 35) broadcast over a ``(G, 1)`` flow-resistivity column."""
    ratio = frequencies[None, :] / sigma_column
    real = 1.0 + 0.0511 * ratio ** (-0.754)
    imag = 0.0768 * ratio ** (-0.732)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


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
        :func:`spherical_spreading_adjustment` and :func:`atmospheric_adjustment`
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
# Topography and screening (guidance §A.4.4-A.4.5, Eq. 36-47)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MeanGroundPlaneResult:
    r"""A mean ground plane fitted to a terrain section (guidance Eq. 36-40).

    ECAC Doc 32, 1st ed., assumes flat terrain; its guidance (§A.4.4)
    represents a varying vertical section by the least-squares line
    :math:`z = a \cdot d + b` through the terrain polyline, evaluated in
    closed form from the per-segment integrals (Eq. 37-40). Equivalent source
    and
    receiver heights are then measured orthogonally to this plane and
    substituted into the flat-ground equations.

    :ivar slope: The fitted slope ``a`` (Eq. 37).
    :ivar intercept: The fitted intercept ``b``, in metres (Eq. 38).
    :ivar distances: The section distances ``d``, in metres, shape ``(M,)``.
    :ivar heights: The terrain heights ``z(d)``, in metres, shape ``(M,)``.
    """

    slope: float
    intercept: float
    distances: NDArray[np.float64]
    heights: NDArray[np.float64]

    def height(self, distance: float | NDArray[np.float64]) -> NDArray[np.float64]:
        r"""The plane height :math:`a\,d + b` at ``distance``, in metres."""
        return np.asarray(self.slope * np.asarray(distance, dtype=np.float64)
                          + self.intercept, dtype=np.float64)

    def equivalent_height(self, distance: float, height: float) -> float:
        """The orthogonal (equivalent) height of a point above the plane.

        Positive above the plane; the guidance substitutes these equivalent
        heights, floored at 0.1 m for source and receiver, into the
        flat-ground equations (§A.4.4).
        """
        return float((height - self.slope * distance - self.intercept)
                     / math.hypot(1.0, self.slope))

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the terrain section and the fitted mean ground plane."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_mean_ground_plane

        return plot_mean_ground_plane(self, ax=ax, language=check_language(language), **kwargs)


def mean_ground_plane(
    distances: NDArray[np.float64] | list[float],
    heights: NDArray[np.float64] | list[float],
) -> MeanGroundPlaneResult:
    r"""The mean ground plane of a terrain section (guidance Eq. 36-40).

    Fits :math:`z = a \cdot d + b` to the polyline of straight segments that
    form the
    terrain profile by continuous least squares (the residual is integrated
    along ``d``, not summed over the vertices), using the closed forms of
    Eq. 37/38 with the segment integrals ``A`` and ``B`` of Eq. 39/40.

    :param distances: Section distances ``d``, in metres, strictly
        increasing, shape ``(M,)`` with :math:`M \ge 2` (arbitrary spacing).
    :param heights: Terrain heights ``z(d)``, in metres, shape ``(M,)``.
    :return: A :class:`MeanGroundPlaneResult`.
    :raises ValueError: If the inputs are invalid.
    """
    d, z = _validated_section(distances, heights)
    a, b = _mean_plane_coefficients(d, z)
    return MeanGroundPlaneResult(slope=a, intercept=b, distances=d, heights=z)


def _validated_section(
    distances: NDArray[np.float64] | list[float],
    heights: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The validated ``(d, z)`` arrays of a terrain section."""
    d = np.atleast_1d(np.asarray(distances, dtype=np.float64))
    z = np.atleast_1d(np.asarray(heights, dtype=np.float64))
    if d.ndim != 1 or d.shape != z.shape or d.size < 2:
        raise ValueError("'distances' and 'heights' must be 1-D of equal size >= 2.")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(z))):
        raise ValueError("'distances' and 'heights' must be finite.")
    if np.any(np.diff(d) <= 0.0):
        raise ValueError("'distances' must be strictly increasing.")
    return d, z


def _mean_plane_coefficients(
    d: NDArray[np.float64], z: NDArray[np.float64],
) -> tuple[float, float]:
    """The Eq. 37/38 closed-form least-squares line through a polyline."""
    ak = np.diff(z) / np.diff(d)
    bk = z[:-1] - ak * d[:-1]
    big_a = (2.0 / 3.0) * np.sum(ak * np.diff(d**3)) + np.sum(bk * np.diff(d**2))
    big_b = np.sum(ak * np.diff(d**2)) + 2.0 * np.sum(bk * np.diff(d))
    span = float(d[-1] - d[0])
    a = 3.0 * (2.0 * big_a - big_b * (d[-1] + d[0])) / span**3
    b = (2.0 * float(d[-1]**3 - d[0]**3) / span**4 * big_b
         - 3.0 * (d[-1] + d[0]) / span**3 * big_a)
    return float(a), float(b)


def mean_flow_resistivity(
    lengths: NDArray[np.float64] | list[float],
    resistivities: NDArray[np.float64] | list[float],
) -> float:
    r"""Logarithmic mean flow resistivity along a path (guidance Eq. 41).

    When the ground type changes along a terrain profile, the guidance
    averages the flow resistivity by the logarithm, weighted by the length of
    each ground segment:
    :math:`\bar{\sigma} = 10^{\sum d_i \cdot \log_{10}(\sigma_i)
    / \sum d_i}`.

    :param lengths: Segment lengths ``dᵢ``, in metres (``> 0``), shape ``(n,)``.
    :param resistivities: Segment flow resistivities ``σᵢ``, in Pa·s/m²
        (``> 0``), shape ``(n,)``.
    :return: The mean flow resistivity ``σ̄``, in Pa·s/m².
    :raises ValueError: If the inputs are invalid.
    """
    d = require_positive_array(lengths, "lengths")
    sig = require_positive_array(resistivities, "resistivities")
    if d.shape != sig.shape:
        raise ValueError("'lengths' and 'resistivities' must have equal shape.")
    return float(10.0 ** (np.sum(d * np.log10(sig)) / np.sum(d)))


def diffraction_attenuation(
    frequencies: NDArray[np.float64] | list[float],
    path_difference: float,
    *,
    edge_height: float,
    edge_span: float = 0.0,
    capped: bool = True,
) -> NDArray[np.float64]:
    r"""Pure diffraction attenuation ``ΔLd`` per band (guidance Eq. 42-44).

    :math:`\Delta L_d = 10 \cdot C_h \cdot \log_{10}(3 + (40/\lambda)
    \cdot C'' \cdot \delta)` where the argument is at least 1
    (below it the attenuation is 0),
    :math:`C_h = \min(f_m \cdot h_0/250, 1)` (Eq. 43) and
    :math:`C''` accounts for multiple diffraction (Eq. 44: 1 for a single
    edge or an edge span :math:`e \le 0.3` m,
    :math:`(1 + (5\lambda/e)^2)/(1/3 + (5\lambda/e)^2)` otherwise).
    A negative path difference (edge below the line of sight) still yields a
    small attenuation down to
    :math:`(40/\lambda) \cdot C'' \cdot \delta = -2`; for bands with
    :math:`\delta < -\lambda/20` the screening chain evaluates the
    clear-path ground effect
    instead of the diffraction (§A.4.5). At grazing incidence
    (:math:`\delta = 0`) the attenuation is the classical
    :math:`10 \cdot \log_{10}(3) \approx 4.8` dB.

    The attenuation is returned positive (a loss); in the Doc 32 Eq. 23
    chain, whose adjustments are added to the level, it enters with a minus
    sign. The wavelength uses the Doc 32 reference speed of sound
    :math:`c = 346.1` m/s.

    :param frequencies: One-third-octave-band centre frequencies, in Hz.
    :param path_difference: Path difference ``δ`` between the diffracted and
        the direct path, in metres (negative when the edge lies below the
        line of sight).
    :param edge_height: Edge height ``h0`` above the mean ground plane(s), in
        metres (the greatest of the two side values for a terrain edge;
        ``≥ 0``).
    :param edge_span: Distance ``e`` between the first and last diffraction
        edges, in metres (default 0: single diffraction).
    :param capped: Apply the 25 dB upper bound of §A.4.5 (default). The
        image-path terms inside the ground-diffraction weighting (Eq. 46/47)
        are evaluated unbounded.
    :return: The attenuation ``ΔLd`` per band, in dB (``≥ 0``).
    :raises ValueError: If the inputs are invalid.
    """
    f = require_positive_array(frequencies, "frequencies")
    if not np.isfinite(path_difference):
        raise ValueError("'path_difference' must be finite.")
    h0 = require_non_negative(edge_height, "edge_height")
    e = require_non_negative(edge_span, "edge_span")
    lam = _C / f
    c2 = np.ones_like(f)
    if e > 0.3:
        c2 = (1.0 + (5.0 * lam / e) ** 2) / (1.0 / 3.0 + (5.0 * lam / e) ** 2)
    arg = 3.0 + 40.0 / lam * c2 * path_difference
    ch = np.minimum(f * h0 / 250.0, 1.0)
    ld = np.where(arg >= 1.0, 10.0 * ch * np.log10(np.maximum(arg, 1.0)), 0.0)
    if capped:
        ld = np.minimum(ld, 25.0)
    return np.asarray(ld, dtype=np.float64)


@dataclass(frozen=True)
class TerrainScreeningResult:
    r"""Ground and screening over a terrain section (guidance §A.4.4-A.4.5).

    :ivar frequencies: Band centre frequencies, in Hz, shape ``(F,)``.
    :ivar adjustment: The combined ground-and-screening adjustment per band,
        in dB, added to the received level in the Doc 32 Eq. 23 chain (it
        replaces the flat-ground ``ΔLg``): the mean-ground-plane ground
        effect when the line of sight is clear,
        :math:`-(\Delta L_d + \Delta L_g)` of Eq. 45
        when terrain blocks it.
    :ivar screened: Whether terrain blocks the line of sight (any profile
        point strictly above it).
    :ivar path_difference: The rubber-band path difference ``δ``, in metres
        (``NaN`` when unscreened).
    :ivar diffraction_points: The diffracting edges ``(d, z)`` on the convex
        propagation path, shape ``(n, 2)`` (empty when unscreened).
    :ivar source: The source ``(d, z)``, in metres.
    :ivar receiver: The receiver ``(d, z)``, in metres.
    :ivar distances: The section distances, in metres, shape ``(M,)``.
    :ivar heights: The section terrain heights, in metres, shape ``(M,)``.
    """

    frequencies: NDArray[np.float64]
    adjustment: NDArray[np.float64]
    screened: bool
    path_difference: float
    diffraction_points: NDArray[np.float64]
    source: tuple[float, float]
    receiver: tuple[float, float]
    distances: NDArray[np.float64]
    heights: NDArray[np.float64]

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the section geometry: terrain, line of sight and sound path."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_terrain_screening

        return plot_terrain_screening(self, ax=ax, language=check_language(language), **kwargs)


def terrain_screening_adjustment(
    frequencies: NDArray[np.float64] | list[float],
    source: tuple[float, float],
    receiver: tuple[float, float],
    distances: NDArray[np.float64] | list[float],
    heights: NDArray[np.float64] | list[float],
    *,
    flow_resistivity: float | str | NDArray[np.float64] | list[float] = "G",
) -> TerrainScreeningResult:
    r"""Ground effect and terrain screening over a vertical section (§A.4.4-A.4.5).

    The terrain profile between the source and the receiver decides the
    propagation regime:

    * **Line of sight clear** (no profile point strictly above it): the
      section's mean ground plane (Eq. 36-40) supplies equivalent orthogonal
      heights (floored at 0.1 m) and the flat-ground two-ray model of
      §A.4.3 evaluates on the plane, with the log-mean flow resistivity
      (Eq. 41) when it varies along the path. Terrain points below the line
      of sight are never treated as diffracting obstacles (the guidance's
      topography rule, which avoids accidental screening in flat terrain).
    * **Blocked**: the sound follows the shortest convex path over the
      terrain (the guidance's rubber band); its vertices are the diffraction
      edges. The attenuation combines the pure diffraction of the path
      difference ``δ`` (Eq. 42-44, capped at 25 dB) with the source-side and
      receiver-side ground effects weighted by their image-path diffractions
      (Eq. 45-47), each side using its own mean ground plane, equivalent
      heights and log-mean flow resistivity. The ground effect is not
      evaluated separately in this regime; bands with
      :math:`\delta < -\lambda/20` fall
      back to the clear-path evaluation (with terrain-only obstacles
      :math:`\delta > 0`, so the rule engages for constructed screens below
      the line of sight rather than for terrain).

    ECAC Doc 32, 1st ed., defines no screening or topography (its Eq. 12
    propagation chain ends at the flat-ground ``ΔLg``); this implements the
    NORAH2 guidance sections A.4.4/A.4.5 and its noise-path appendices,
    whose diffraction equations follow CNOSSOS-EU.

    :param frequencies: One-third-octave-band centre frequencies, in Hz.
    :param source: Source ``(d, z)`` in the section, in metres.
    :param receiver: Receiver ``(d, z)`` in the section, in metres (the
        microphone point, i.e. ground plus microphone height).
    :param distances: Terrain section distances ``d``, in metres, strictly
        increasing, covering ``[source d, receiver d]``.
    :param heights: Terrain heights ``z(d)``, in metres.
    :param flow_resistivity: Ground flow resistivity: a value in Pa·s/m², a
        CNOSSOS class letter, or one value per profile segment (shape
        ``(M−1,)``) averaged per sub-path by Eq. 41.
    :return: A :class:`TerrainScreeningResult`.
    :raises ValueError: If the inputs are invalid.
    """
    f = require_positive_array(frequencies, "frequencies")
    d, z = _validated_section(distances, heights)
    src = (float(source[0]), float(source[1]))
    rcv = (float(receiver[0]), float(receiver[1]))
    if not (np.isfinite(src[0]) and np.isfinite(src[1])
            and np.isfinite(rcv[0]) and np.isfinite(rcv[1])):
        raise ValueError("'source' and 'receiver' must be finite (d, z) points.")
    if src[0] >= rcv[0]:
        raise ValueError("'source' must lie at a smaller section distance than 'receiver'.")
    if d[0] > src[0] + 1e-9 or d[-1] < rcv[0] - 1e-9:
        raise ValueError("The terrain section must cover [source d, receiver d].")
    sigma_seg = _segment_resistivities(flow_resistivity, d.size - 1)
    d, z, sigma_seg = _cropped_section(d, z, sigma_seg, src[0], rcv[0])
    adjustment, screened, delta, points = _screening_core(f, src, rcv, d, z, sigma_seg)
    return TerrainScreeningResult(
        frequencies=f, adjustment=adjustment, screened=screened,
        path_difference=delta, diffraction_points=points, source=src,
        receiver=rcv, distances=d, heights=z)


def _segment_resistivities(
    flow_resistivity: float | str | NDArray[np.float64] | list[float],
    n_segments: int,
) -> NDArray[np.float64]:
    """Per-segment ``σ`` from a scalar, class letter or per-segment array."""
    if isinstance(flow_resistivity, str):
        return np.full(n_segments, _resolve_flow_resistivity(flow_resistivity))
    if np.isscalar(flow_resistivity):
        return np.full(n_segments, _resolve_flow_resistivity(float(flow_resistivity)))  # type: ignore[arg-type]
    arr = require_positive_array(flow_resistivity, "flow_resistivity")
    if arr.shape != (n_segments,):
        raise ValueError("Per-segment 'flow_resistivity' must have one value per "
                         "profile segment.")
    return arr


def _cropped_section(
    d: NDArray[np.float64], z: NDArray[np.float64],
    sigma_seg: NDArray[np.float64], d_lo: float, d_hi: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The section restricted to ``[d_lo, d_hi]`` with interpolated ends."""
    keep = (d > d_lo) & (d < d_hi)
    dd = np.concatenate([[d_lo], d[keep], [d_hi]])
    zz = np.interp(dd, d, z)
    seg_mid = 0.5 * (dd[:-1] + dd[1:])
    idx = np.clip(np.searchsorted(d, seg_mid) - 1, 0, sigma_seg.size - 1)
    return dd, zz, sigma_seg[idx]


def _upper_hull(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """The upper convex hull of ``(d, z)`` points sorted by ``d``."""
    hull: list[np.ndarray] = []
    for pt in points:
        while len(hull) >= 2:
            u, v = hull[-2], hull[-1]
            if (v[0] - u[0]) * (pt[1] - u[1]) - (v[1] - u[1]) * (pt[0] - u[0]) >= 0.0:
                hull.pop()
            else:
                break
        hull.append(pt)
    return np.asarray(hull)


def _side_ground(
    f: NDArray[np.float64],
    d: NDArray[np.float64], z: NDArray[np.float64],
    sigma_seg: NDArray[np.float64],
    p_lo: tuple[float, float], p_hi: tuple[float, float],
    clamp_lo: bool, clamp_hi: bool,
) -> tuple[NDArray[np.float64], float, float, float]:
    """Mean-plane ground effect between two points of the section.

    Returns the flat-ground adjustment ``ΔLg`` (Eq. 28-35) evaluated with
    the equivalent orthogonal heights over the sub-profile's mean ground
    plane and its Eq. 41 mean resistivity, together with the two equivalent
    heights and the plane's slope (the image-point construction needs it).
    ``clamp_lo``/``clamp_hi`` apply the 0.1 m floor of §A.4.4 to the heights
    handed onward (real sources and receivers; diffraction edges pass
    unfloored so ``h0`` and the images use their true elevation); inside the
    two-ray core itself every height is floored at 0.1 m regardless.
    """
    dd, zz, seg = _cropped_section(d, z, sigma_seg, p_lo[0], p_hi[0])
    a, b = _mean_plane_coefficients(dd, zz)
    scale = math.hypot(1.0, a)
    h_lo = (p_lo[1] - a * p_lo[0] - b) / scale
    h_hi = (p_hi[1] - a * p_hi[0] - b) / scale
    if clamp_lo:
        h_lo = max(h_lo, 0.1)
    if clamp_hi:
        h_hi = max(h_hi, 0.1)
    # Separation of the feet of the orthogonal projections, along the plane.
    s_lo = (p_lo[0] + a * (p_lo[1] - b)) / scale
    s_hi = (p_hi[0] + a * (p_hi[1] - b)) / scale
    lengths = np.hypot(np.diff(dd), np.diff(zz))
    sigma = mean_flow_resistivity(lengths, seg)
    dlg = _ground_effect(f, h_lo, h_hi, np.asarray([abs(s_hi - s_lo)]), sigma)[0]
    return np.asarray(dlg, dtype=np.float64), h_lo, h_hi, a


def _screening_core(
    f: NDArray[np.float64],
    src: tuple[float, float], rcv: tuple[float, float],
    d: NDArray[np.float64], z: NDArray[np.float64],
    sigma_seg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool, float, NDArray[np.float64]]:
    """The combined ground-and-screening adjustment of a section (Eq. 45-47)."""
    interior = slice(1, -1) if d.size > 2 else slice(0, 0)
    los = src[1] + (rcv[1] - src[1]) * (d - src[0]) / (rcv[0] - src[0])
    above = np.zeros(d.size, dtype=bool)
    above[interior] = z[interior] > los[interior]

    # Clear line of sight: mean-ground-plane ground effect over the full path.
    clear, _, _, _ = _side_ground(f, d, z, sigma_seg, src, rcv, True, True)
    if not np.any(above):
        return clear, False, float("nan"), np.empty((0, 2))

    # Blocked: rubber band over the terrain (the shortest convex path).
    pts = np.vstack([[src[0], src[1]], np.column_stack([d[above], z[above]]),
                     [rcv[0], rcv[1]]])
    hull = _upper_hull(pts)
    edges = hull[1:-1]
    if edges.shape[0] == 0:   # numerically grazing: treat as clear
        return clear, False, float("nan"), np.empty((0, 2))
    seg_len = np.hypot(np.diff(hull[:, 0]), np.diff(hull[:, 1]))
    direct = math.hypot(rcv[0] - src[0], rcv[1] - src[1])
    delta = float(np.sum(seg_len) - direct)
    # e: the distance between the first and last diffraction edges, measured
    # along the intermediate edges (equal to the chord for one or two edges).
    span = float(np.sum(np.hypot(np.diff(edges[:, 0]), np.diff(edges[:, 1]))))

    o_first = (float(edges[0, 0]), float(edges[0, 1]))
    o_last = (float(edges[-1, 0]), float(edges[-1, 1]))
    ag_s, zs, zo_s, a_s = _side_ground(f, d, z, sigma_seg, src, o_first, True, False)
    ag_r, zo_r, zr, a_r = _side_ground(f, d, z, sigma_seg, o_last, rcv, False, True)
    h0 = max(max(zo_s, 0.0), max(zo_r, 0.0))

    delta_img_s = _image_path_difference(hull, _mirrored_point(src, zs, a_s),
                                         side="source")
    delta_img_r = _image_path_difference(hull, _mirrored_point(rcv, zr, a_r),
                                         side="receiver")
    ld = diffraction_attenuation(f, delta, edge_height=h0, edge_span=span)
    ld_free = diffraction_attenuation(f, delta, edge_height=h0, edge_span=span,
                                      capped=False)
    ld_s = diffraction_attenuation(f, delta_img_s, edge_height=h0,
                                   edge_span=span, capped=False)
    ld_r = diffraction_attenuation(f, delta_img_r, edge_height=h0,
                                   edge_span=span, capped=False)
    # Eq. 46/47: the side ground attenuations, weighted by their image-path
    # diffractions (A = −ΔLg maps the Eq. 23 adjustments onto attenuations).
    lg_s = -20.0 * np.log10(
        1.0 + (10.0 ** (ag_s / 20.0) - 1.0) * 10.0 ** (-(ld_s - ld_free) / 20.0))
    lg_r = -20.0 * np.log10(
        1.0 + (10.0 ** (ag_r / 20.0) - 1.0) * 10.0 ** (-(ld_r - ld_free) / 20.0))
    total = ld + lg_s + lg_r                       # Eq. 45, an attenuation
    # Per-band trigger (§A.4.5): bands with δ < −λ/20 keep the clear path.
    screened_band = delta >= -(_C / f) / 20.0
    adjustment = np.where(screened_band, -total, clear)
    return np.asarray(adjustment, dtype=np.float64), True, delta, edges


def _mirrored_point(
    point: tuple[float, float], equivalent_height: float, slope: float,
) -> tuple[float, float]:
    r"""The orthogonal mirror image of a point across a side mean plane.

    The image sits at twice the (orthogonal) equivalent height along the
    plane's downward normal, :math:`p - 2 \cdot h \cdot \hat{n}` with
    :math:`\hat{n} = (-a, 1)/\sqrt{1+a^2}`
    (guidance Figure 8: S′/R′ are the images in relation to the side mean
    ground planes).
    """
    h = max(equivalent_height, 0.0)
    scale = math.hypot(1.0, slope)
    return (point[0] + 2.0 * h * slope / scale, point[1] - 2.0 * h / scale)


def _image_path_difference(
    hull: NDArray[np.float64], image: tuple[float, float], side: str,
) -> float:
    """The rubber-band path difference from an image point (Eq. 46/47).

    The diffracted path follows the same edges with the source (or receiver)
    replaced by its mirror image across the side mean plane.
    """
    pts = hull.copy()
    if side == "source":
        pts[0] = image
    else:
        pts[-1] = image
    seg_len = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    direct = math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
    return float(np.sum(seg_len) - direct)


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


def _absorption_coefficient(
    frequencies: NDArray[np.float64], temperature: float, relative_humidity: float,
    pressure: float,
) -> NDArray[np.float64]:
    """ISO 9613-1 pure-tone ``α`` in dB/m at the exact band centres (Eq. 27).

    The single suppression site for the advisory out-of-range warning of the
    sub-50 Hz bands of the standard NORAH grid (``α`` is negligible there);
    bands above 10 kHz keep the advisory, since ``α`` is large and
    extrapolated. :func:`atmospheric_adjustment` and the event chain share it.
    """
    import warnings

    from ..environment.propagation.air_absorption import (
        AtmosphericAbsorptionWarning,
        air_attenuation,
    )

    with warnings.catch_warnings():
        if frequencies.max() <= 10000.0:
            warnings.filterwarnings(
                "ignore", message="One or more frequencies are outside",
                category=AtmosphericAbsorptionWarning)
        alpha = air_attenuation(
            frequencies, temperature, relative_humidity, pressure, exact_midband=True)
    return np.asarray(alpha, dtype=np.float64)


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
    receiver_height: float,
    ground_elevation: float | NDArray[np.float64] | list[float] | list[list[float]],
    airspeed: float | NDArray[np.float64] | list[float] | None,
    path_angle: float | NDArray[np.float64] | list[float] | None,
    heading: float | NDArray[np.float64] | list[float] | None,
    bank_angle: float | NDArray[np.float64] | list[float] | None,
    flow_resistivity: float | str | np.floating[Any] | np.integer[Any]
    | NDArray[np.float64] | list[float] | list[list[float]],
    temperature: float,
    relative_humidity: float,
    pressure: float,
    level_offset: float | NDArray[np.float64] | list[float],
    scaling_factor: float,
    triangles: NDArray[np.int_] | list[list[int]] | None,
    atmospheric_method: str,
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | Sequence[NDArray[np.float64]] | None,
    terrain_resolution: float | None,
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
    hr = require_positive(receiver_height, "receiver_height")
    dem = _validated_terrain(terrain)
    if dem is not None:
        _require_dem_coverage(dem, p[:, 0], p[:, 1], "track")
    spacing = _section_spacing(dem, terrain_resolution)
    sigma = _setup_resistivity(flow_resistivity, dem)
    ground = _setup_ground_elevation(ground_elevation, dem)
    method = require_choice(atmospheric_method, "atmospheric_method", ("iso9613", "sae"))
    spd, gam, hdg, bank = _resolved_track_state(
        t, p, airspeed, path_angle, heading, bank_angle)
    off = _per_point(level_offset, t.size, "level_offset")
    offsets = off if off is not None else np.zeros(t.size)
    alpha = _absorption_coefficient(freqs, temperature, relative_humidity, pressure)
    return _EventSetup(
        hemispheres=tuple(hemispheres),
        airspeeds=np.atleast_1d(np.asarray(airspeeds, dtype=np.float64)),
        path_angles=np.atleast_1d(np.asarray(path_angles, dtype=np.float64)),
        frequencies=freqs, times=t, positions=p, speed=spd, gamma=gam,
        heading=hdg, bank=bank, offsets=offsets,
        ground_elevation=ground, receiver_height=hr,
        sigma=sigma, alpha=alpha, rref=rref, scaling_factor=scaling_factor,
        triangles=triangles, band_integrated=method == "sae",
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
    airspeed: float | NDArray[np.float64] | list[float] | None,
    path_angle: float | NDArray[np.float64] | list[float] | None,
    heading: float | NDArray[np.float64] | list[float] | None,
    bank_angle: float | NDArray[np.float64] | list[float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-point ``(V_A, γ, Θ, Φ)``: explicit overrides, else derived (Eq. 16-21)."""
    n = times.size
    spd = _per_point(airspeed, n, "airspeed")
    gam = _per_point(path_angle, n, "path_angle")
    hdg = _per_point(heading, n, "heading")
    bank = _per_point(bank_angle, n, "bank_angle")
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
    receiver_height: float = 1.2,
    ground_elevation: float = 0.0,
    airspeed: float | NDArray[np.float64] | list[float] | None = None,
    path_angle: float | NDArray[np.float64] | list[float] | None = None,
    heading: float | NDArray[np.float64] | list[float] | None = None,
    bank_angle: float | NDArray[np.float64] | list[float] | None = None,
    flow_resistivity: float | str | np.floating[Any] | np.integer[Any] = "G",
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
    level_offset: float | NDArray[np.float64] | list[float] = 0.0,
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
    atmospheric_method: str = "iso9613",
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | Sequence[NDArray[np.float64]] | None = None,
    terrain_resolution: float | None = None,
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

    The flight condition per point comes from the ``airspeed``/``path_angle``
    overrides when given (e.g. the smoothed values of a radar-track workflow),
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
    :param receiver_height: Microphone height above local ground, in metres
        (default 1.2).
    :param ground_elevation: Ground elevation ``z`` at the site, in metres on
        the track datum (default 0); source and receiver heights above ground
        follow from it.
    :param airspeed: Per-point airspeed override, scalar or shape ``(N,)``.
    :param path_angle: Per-point path-angle override, in degrees.
    :param heading: Per-point heading override, in degrees.
    :param bank_angle: Per-point bank-angle override, in degrees (positive
        starboard down).
    :param flow_resistivity: Ground flow resistivity ``σ`` in Pa·s/m², or a
        CNOSSOS class letter (see :func:`ground_effect_adjustment`).
    :param temperature: Air temperature, in °C (default 25, ICAO reference).
    :param relative_humidity: Relative humidity, in % (default 70).
    :param pressure: Ambient pressure, in kPa (default 101.325).
    :param level_offset: Source-level offset ``ΔEPNL`` added to the hemisphere
        levels (Eq. 2 class substitution), in dB (default 0). Scalar or per
        track point, shape ``(N,)``: Chapter-8 substitutions correct climb,
        level and descent conditions with different certification levels.
    :param scaling_factor: Flight-condition scaling factor ``F_fc`` (default 2).
    :param triangles: Optional precomputed flight-condition triangulation (see
        :func:`flight_condition_weights`).
    :param atmospheric_method: ``"iso9613"`` for the pure-tone Eq. 26/27 term
        (the guidance text), or ``"sae"`` for the SAE ARP 5534 band-integrated
        mapping used by the NORAH2 reference implementation (they agree to
        ~0.05 dB below 3.15 kHz).
    :param terrain: Optional digital elevation model ``(x, y, z)`` on the
        track frame (``x`` and ``y`` strictly increasing, ``z`` of shape
        ``(len(y), len(x))``, all in metres on the track datum). When given,
        every emission-receiver pair is evaluated over its sampled vertical
        section (guidance §A.4.4/A.4.5): mean-ground-plane ground effect with
        equivalent heights, and rubber-band diffraction where terrain blocks
        the line of sight; ``ground_elevation`` is then taken from the model.
        The model must cover the whole track and the receiver (fabricating
        terrain beyond its edges is refused).
    :param terrain_resolution: Section sampling step along the path, in
        metres (default: the elevation model's cell size; sections are capped
        at 20000 sampling intervals).
    :return: A :class:`RotorcraftEventResult`.
    :raises ValueError: If the inputs are invalid.
    """
    setup = _event_setup(
        hemispheres, airspeeds, path_angles, times, positions,
        receiver_height=receiver_height, ground_elevation=ground_elevation,
        airspeed=airspeed, path_angle=path_angle, heading=heading,
        bank_angle=bank_angle, flow_resistivity=flow_resistivity,
        temperature=temperature, relative_humidity=relative_humidity,
        pressure=pressure, level_offset=level_offset,
        scaling_factor=scaling_factor, triangles=triangles,
        atmospheric_method=atmospheric_method, terrain=terrain,
        terrain_resolution=terrain_resolution)
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
    receiver_height: float = 1.2,
    ground_elevation: float | NDArray[np.float64] | list[list[float]] = 0.0,
    airspeed: float | NDArray[np.float64] | list[float] | None = None,
    path_angle: float | NDArray[np.float64] | list[float] | None = None,
    heading: float | NDArray[np.float64] | list[float] | None = None,
    bank_angle: float | NDArray[np.float64] | list[float] | None = None,
    flow_resistivity: float | str | NDArray[np.float64] | list[list[float]] = "G",
    temperature: float = 25.0,
    relative_humidity: float = 70.0,
    pressure: float = 101.325,
    level_offset: float | NDArray[np.float64] | list[float] = 0.0,
    scaling_factor: float = 2.0,
    triangles: NDArray[np.int_] | list[list[int]] | None = None,
    atmospheric_method: str = "iso9613",
    terrain: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | Sequence[NDArray[np.float64]] | None = None,
    terrain_resolution: float | None = None,
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
    :param receiver_height: Microphone height above local ground, in metres.
    :param ground_elevation: Ground elevation, in metres on the track datum:
        a scalar, or one value per grid point (shape ``(len(y), len(x))``)
        for receivers on uneven sites without a full elevation model.
    :param airspeed: Per-point airspeed override (see
        :func:`rotorcraft_event_level`).
    :param path_angle: Per-point path-angle override, in degrees.
    :param heading: Per-point heading override, in degrees.
    :param bank_angle: Per-point bank-angle override, in degrees.
    :param flow_resistivity: Ground flow resistivity ``σ`` in Pa·s/m², a
        CNOSSOS class letter, or one value per grid point (shape
        ``(len(y), len(x))``) for heterogeneous ground across the receivers
        (each receiver's two-ray model uses its local value).
    :param temperature: Air temperature, in °C.
    :param relative_humidity: Relative humidity, in %.
    :param pressure: Ambient pressure, in kPa.
    :param level_offset: Source-level offset ``ΔEPNL`` (Eq. 2), in dB, scalar
        or per track point.
    :param scaling_factor: Flight-condition scaling factor ``F_fc`` (default 2).
    :param triangles: Optional precomputed flight-condition triangulation.
    :param atmospheric_method: ``"iso9613"`` or ``"sae"`` (see
        :func:`rotorcraft_event_level`).
    :param terrain: Optional digital elevation model ``(x, y, z)`` (see
        :func:`rotorcraft_event_level`); it must cover the whole track and
        grid. Every emission-receiver pair then samples its own vertical
        section, so the cost grows with track points times grid points; keep
        contour grids modest with terrain.
    :param terrain_resolution: Section sampling step, in metres (default: the
        elevation model's cell size; sections are capped at 20000 sampling
        intervals).
    :return: A :class:`RotorcraftNoiseContourResult`.
    :raises ValueError: If the inputs are invalid.
    """
    setup = _event_setup(
        hemispheres, airspeeds, path_angles, times, positions,
        receiver_height=receiver_height, ground_elevation=ground_elevation,
        airspeed=airspeed, path_angle=path_angle, heading=heading,
        bank_angle=bank_angle, flow_resistivity=flow_resistivity,
        temperature=temperature, relative_humidity=relative_humidity,
        pressure=pressure, level_offset=level_offset,
        scaling_factor=scaling_factor, triangles=triangles,
        atmospheric_method=atmospheric_method, terrain=terrain,
        terrain_resolution=terrain_resolution)
    gx = np.asarray(x, dtype=np.float64).ravel()
    gy = np.asarray(y, dtype=np.float64).ravel()
    if gx.size < 2 or gy.size < 2 or not (np.all(np.isfinite(gx)) and np.all(np.isfinite(gy))):
        raise ValueError("'x' and 'y' must each be finite with at least two grid points.")
    key = require_choice(metric, "metric", ("exposure", "maximum"))
    xx, yy = np.meshgrid(gx, gy)
    n_g = xx.size
    for name, value in (("flow_resistivity", flow_resistivity),
                        ("ground_elevation", ground_elevation)):
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
