#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Rotorcraft propagation, ground effect and screening (ECAC Doc 32 / NORAH2).

Between the noise hemisphere that describes a helicopter and the receiver on the
ground lies the path, and the path knows nothing about rotorcraft. The ECAC
Doc 32 propagation chain
:math:`\Delta L_p = \Delta L_\mathrm{s} + \Delta L_\mathrm{a} + \Delta L_\mathrm{g}` adds spherical
spreading, atmospheric absorption and the ground effect of a point source over
an impedance plane; the NORAH2 guidance extends that last term to real terrain,
where a fitted mean ground plane replaces the flat ground and a blocked line of
sight becomes diffraction. Every function here takes a geometry and a spectrum;
:mod:`~phonometry.aircraft.rotorcraft_noise` holds the hemisphere source and the
event chain that call them.

This module provides the propagation primitives of the method (clean-room, from
the NORAH2 guidance SC01.D1.5d, the basis of ECAC Doc 32):

* :func:`spherical_spreading_adjustment` --
  :math:`\Delta L_\mathrm{s} = -20 \cdot \log_{10}(r/60)` (Eq. 24).
* :func:`atmospheric_adjustment` --
  :math:`\Delta L_\mathrm{a} = -\alpha(f) \cdot (r - 60)` with the ISO 9613-1
  pure-tone coefficient (Eq. 26/27), reusing
  :func:`~phonometry.environment.propagation.air_absorption.air_attenuation`.
* :func:`ground_effect_adjustment` -- ``ΔLg`` for a point source over an impedance
  plane (Chien-Soroka, Eq. 28-35) with the Delany-Bazley one-parameter impedance
  and the CNOSSOS flow-resistivity classes.
* :func:`mean_ground_plane` -- the least-squares plane through a terrain section
  (Eq. 36-40), whose equivalent orthogonal heights carry a varying profile into
  the flat-ground equations.
* :func:`mean_flow_resistivity` -- the log-mean flow resistivity of a path that
  crosses several ground types (Eq. 41).
* :func:`diffraction_attenuation` -- the pure diffraction attenuation ``ΔLd`` of
  a path difference (Eq. 42-44).
* :func:`terrain_screening_adjustment` -- the combined ground-and-screening
  adjustment over a vertical section (§A.4.4-A.4.5, Eq. 45-47): the
  mean-ground-plane ground effect while the line of sight is clear, the
  rubber-band diffraction over the terrain once it is blocked.

ECAC Doc 32, 1st ed., defines no topography or screening at all: its Eq. 12
chain ends at the flat-ground ``ΔLg``. The mean ground plane, the log-mean flow
resistivity and the diffraction of §A.4.4-A.4.5 come from the guidance, whose
diffraction equations follow CNOSSOS-EU.

Source (clean-room): ECAC Doc 32, 1st ed.; NORAH2 rotorcraft-noise modelling
guidance (EASA.2020.FC.06 SC01.D1.5d), §A.4. The atmospheric term is validated
against the guidance Table 4 (one-third-octave attenuation per km at ICAO
reference conditions); the ground and screening chain is validated end to end,
inside the event chain, against the NORAH2 reference implementation outputs for
the ARP verification cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
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
#: CNOSSOS-EU flow resistivity by ground class, in Pa·s/m² (Doc 32 Table 5 / D1.5d Table 5).
_FLOW_RESISTIVITY = {
    "A": 12.5e3,  # very soft (snow, moss)
    "B": 31.5e3,  # soft forest floor
    "C": 80.0e3,  # uncompacted loose ground (turf, grass)
    "D": 200.0e3,  # normal uncompacted ground (pasture)
    "E": 500.0e3,  # compacted field and gravel
    "F": 2.0e6,  # compacted dense ground (gravel road, car park)
    "G": 20.0e6,  # hard surfaces (asphalt, concrete)
    "H": 200.0e6,  # very hard and dense (dense asphalt, water)
}


# --------------------------------------------------------------------------- #
# Spreading, absorption and flat ground (guidance §A.4.3, Eq. 24-35)
# --------------------------------------------------------------------------- #


def spherical_spreading_adjustment(
    distance: float,
    *,
    reference_distance: float = _RH,
) -> float:
    r"""Spherical-spreading adjustment ``ΔLs`` of the hemisphere level (Eq. 24).

    The hemisphere levels are defined at the reference distance ``rh`` (60 m in
    the standard database), so at slant distance ``r`` the geometric spreading
    adjustment is :math:`\Delta L_\mathrm{s} = -20 \cdot \log_{10}(r/r_\mathrm{h})`.

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
    ``rh``, so only the excess path :math:`r - r_\mathrm{h}` is corrected:
    :math:`\Delta L_\mathrm{a} = -\alpha(f) \cdot (r - r_\mathrm{h})` with the ISO 9613-1
    pure-tone coefficient ``α``
    evaluated at the exact band centre (Eq. 26/27, ICAO reference atmosphere by
    default). This matches the guidance Eq. 27 to 0.02 dB/km and the NORAH2
    reference implementation. The guidance's alternative per-band mapping (SAE
    method by Rickley et al., its Table 4) coincides below 3.15 kHz and deviates
    by up to 2.2 dB/km at 8-10 kHz; for a path-dependent band mapping use
    :func:`~phonometry.aircraft.atmospheric_absorption.sae_band_attenuation`.

    .. note::
        The printed guidance Eq. 27 pairs the coefficient ``6.6928e-6`` with
        :math:`f_\mathrm{r,O} = 630.7` Hz, which evaluates to nonsense (14.3 dB/km
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
        :math:`\le 0` for :math:`r \ge r_\mathrm{h}`).
    :raises ValueError: If a distance is not strictly positive.
    """
    f = require_positive_array(frequencies, "frequencies")
    r = require_positive(distance, "distance")
    rh = require_positive(reference_distance, "reference_distance")
    alpha = _absorption_coefficient(f, temperature, relative_humidity, pressure)
    return np.asarray(-alpha * (r - rh), dtype=np.float64)


def _absorption_coefficient(
    frequencies: NDArray[np.float64],
    temperature: float,
    relative_humidity: float,
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
                "ignore",
                message="One or more frequencies are outside",
                category=AtmosphericAbsorptionWarning,
            )
        alpha = air_attenuation(
            frequencies, temperature, relative_humidity, pressure, exact_midband=True
        )
    return np.asarray(alpha, dtype=np.float64)


def _delany_bazley_impedance(
    frequencies: NDArray[np.float64], flow_resistivity: float
) -> NDArray[np.complex128]:
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
    :math:`\Delta L_\mathrm{g} = 10 \cdot \log_{10}\{1 + (r_1/r_2)^2 \lvert Q \rvert^2
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
                f"'flow_resistivity' class must be one of {valid}, got {flow_resistivity!r}."
            )
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
    hs = np.maximum(np.atleast_1d(np.asarray(source_height, dtype=np.float64)), 0.1)[
        :, None
    ]
    hr = max(receiver_height, 0.1)
    dp = horizontal_distances[:, None]

    r1 = np.hypot(dp, hs - hr)  # direct path
    r2 = np.hypot(dp, hs + hr)  # reflected path (image source)
    dr = r2 - r1  # path-length difference ΔR
    cos_xi = (hs + hr) / r2  # incidence angle from the normal
    k = 2.0 * np.pi * f / _C

    sig = np.atleast_1d(np.asarray(sigma, dtype=np.float64))[:, None]
    zs = _delany_bazley_impedance_grid(np.asarray(frequencies), sig)
    rp = (zs * cos_xi - 1.0) / (zs * cos_xi + 1.0)
    d_num = (1.0 + 1j) / 2.0 * np.sqrt(k * r2) * (1.0 / zs + cos_xi)
    # The boundary-loss factor F(d), which is 1 + i·d·√π·w(d) with w the
    # Faddeeva function that scipy exposes as wofz.
    f_loss = 1.0 + 1j * d_num * np.sqrt(np.pi) * wofz(d_num)
    q = rp + (1.0 - rp) * f_loss
    q_mag = np.abs(q)
    psi = np.angle(q)

    a = 0.727 * f * dr / _C
    sinc = np.sinc(a / np.pi)  # sin(a)/a, = 1 at a = 0
    inter = sinc * np.cos(6.325 * f * dr / _C + psi)  # I (Eq. 30)
    ratio = r1 / r2
    arg = 1.0 + ratio**2 * q_mag**2 + 2.0 * ratio * q_mag * inter
    return np.asarray(10.0 * np.log10(np.maximum(arg, 1e-15)), dtype=np.float64)


def _delany_bazley_impedance_grid(
    frequencies: NDArray[np.float64],
    sigma_column: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """``Zs`` (Eq. 35) broadcast over a ``(G, 1)`` flow-resistivity column."""
    ratio = frequencies[None, :] / sigma_column
    real = 1.0 + 0.0511 * ratio ** (-0.754)
    imag = 0.0768 * ratio ** (-0.732)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


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
        return np.asarray(
            self.slope * np.asarray(distance, dtype=np.float64) + self.intercept,
            dtype=np.float64,
        )

    def equivalent_height(self, distance: float, height: float) -> float:
        """The orthogonal (equivalent) height of a point above the plane.

        Positive above the plane; the guidance substitutes these equivalent
        heights, floored at 0.1 m for source and receiver, into the
        flat-ground equations (§A.4.4).
        """
        return float(
            (height - self.slope * distance - self.intercept)
            / math.hypot(1.0, self.slope)
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the terrain section and the fitted mean ground plane."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_mean_ground_plane

        return plot_mean_ground_plane(
            self, ax=ax, language=check_language(language), **kwargs
        )


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
    d: NDArray[np.float64],
    z: NDArray[np.float64],
) -> tuple[float, float]:
    """The Eq. 37/38 closed-form least-squares line through a polyline."""
    ak = np.diff(z) / np.diff(d)
    bk = z[:-1] - ak * d[:-1]
    big_a = (2.0 / 3.0) * np.sum(ak * np.diff(d**3)) + np.sum(bk * np.diff(d**2))
    big_b = np.sum(ak * np.diff(d**2)) + 2.0 * np.sum(bk * np.diff(d))
    span = float(d[-1] - d[0])
    a = 3.0 * (2.0 * big_a - big_b * (d[-1] + d[0])) / span**3
    b = (
        2.0 * float(d[-1] ** 3 - d[0] ** 3) / span**4 * big_b
        - 3.0 * (d[-1] + d[0]) / span**3 * big_a
    )
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

    :math:`\Delta L_\mathrm{d} = 10 \cdot C_h \cdot \log_{10}(3 + (40/\lambda)
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
        :math:`-(\Delta L_\mathrm{d} + \Delta L_\mathrm{g})` of Eq. 45
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

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the section geometry: terrain, line of sight and sound path."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_terrain_screening

        return plot_terrain_screening(
            self, ax=ax, language=check_language(language), **kwargs
        )


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
    if not (
        np.isfinite(src[0])
        and np.isfinite(src[1])
        and np.isfinite(rcv[0])
        and np.isfinite(rcv[1])
    ):
        raise ValueError("'source' and 'receiver' must be finite (d, z) points.")
    if src[0] >= rcv[0]:
        raise ValueError(
            "'source' must lie at a smaller section distance than 'receiver'."
        )
    if d[0] > src[0] + 1e-9 or d[-1] < rcv[0] - 1e-9:
        raise ValueError("The terrain section must cover [source d, receiver d].")
    sigma_seg = _segment_resistivities(flow_resistivity, d.size - 1)
    d, z, sigma_seg = _cropped_section(d, z, sigma_seg, src[0], rcv[0])
    adjustment, screened, delta, points = _screening_core(f, src, rcv, d, z, sigma_seg)
    return TerrainScreeningResult(
        frequencies=f,
        adjustment=adjustment,
        screened=screened,
        path_difference=delta,
        diffraction_points=points,
        source=src,
        receiver=rcv,
        distances=d,
        heights=z,
    )


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
        raise ValueError(
            "Per-segment 'flow_resistivity' must have one value per profile segment."
        )
    return arr


def _cropped_section(
    d: NDArray[np.float64],
    z: NDArray[np.float64],
    sigma_seg: NDArray[np.float64],
    d_lo: float,
    d_hi: float,
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
    d: NDArray[np.float64],
    z: NDArray[np.float64],
    sigma_seg: NDArray[np.float64],
    p_lo: tuple[float, float],
    p_hi: tuple[float, float],
    clamp_lo: bool,
    clamp_hi: bool,
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
    src: tuple[float, float],
    rcv: tuple[float, float],
    d: NDArray[np.float64],
    z: NDArray[np.float64],
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
    pts = np.vstack(
        [[src[0], src[1]], np.column_stack([d[above], z[above]]), [rcv[0], rcv[1]]]
    )
    hull = _upper_hull(pts)
    edges = hull[1:-1]
    if edges.shape[0] == 0:  # numerically grazing: treat as clear
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

    delta_img_s = _image_path_difference(
        hull, _mirrored_point(src, zs, a_s), side="source"
    )
    delta_img_r = _image_path_difference(
        hull, _mirrored_point(rcv, zr, a_r), side="receiver"
    )
    ld = diffraction_attenuation(f, delta, edge_height=h0, edge_span=span)
    ld_free = diffraction_attenuation(
        f, delta, edge_height=h0, edge_span=span, capped=False
    )
    ld_s = diffraction_attenuation(
        f, delta_img_s, edge_height=h0, edge_span=span, capped=False
    )
    ld_r = diffraction_attenuation(
        f, delta_img_r, edge_height=h0, edge_span=span, capped=False
    )
    # Eq. 46/47: the side ground attenuations, weighted by their image-path
    # diffractions (A = −ΔLg maps the Eq. 23 adjustments onto attenuations).
    lg_s = -20.0 * np.log10(
        1.0 + (10.0 ** (ag_s / 20.0) - 1.0) * 10.0 ** (-(ld_s - ld_free) / 20.0)
    )
    lg_r = -20.0 * np.log10(
        1.0 + (10.0 ** (ag_r / 20.0) - 1.0) * 10.0 ** (-(ld_r - ld_free) / 20.0)
    )
    total = ld + lg_s + lg_r  # Eq. 45, an attenuation
    # Per-band trigger (§A.4.5): bands with δ < −λ/20 keep the clear path.
    screened_band = delta >= -(_C / f) / 20.0
    adjustment = np.where(screened_band, -total, clear)
    return np.asarray(adjustment, dtype=np.float64), True, delta, edges


def _mirrored_point(
    point: tuple[float, float],
    equivalent_height: float,
    slope: float,
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
    hull: NDArray[np.float64],
    image: tuple[float, float],
    side: str,
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
