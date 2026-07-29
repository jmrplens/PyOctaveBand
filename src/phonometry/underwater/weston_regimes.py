#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Weston's shallow-water propagation regimes (flux theory).

A source in a shallow-water waveguide loses energy in four successive range
regimes, each with its own power law. The boundaries between them follow from
the seabed reflectivity alone, which makes the set an inexpensive analytic
reference for any numerical propagation model:

* **spherical spreading** -- ``F = 1/r²`` (``20·lg r``), while the sound has not
  yet felt the boundaries;
* **cylindrical spreading** -- ``F = 2·ψc/(r·H)`` (``10·lg r``), once the energy
  is confined to a cylinder of height ``H`` and only rays within the critical
  angle ``ψc`` survive;
* **mode stripping** -- ``F = (π/(η·H))^½ · r^−3/2`` (``15·lg r``), once the
  accumulated reflection loss has eroded the steep paths;
* **single mode** -- an exponential decay dominated by the lowest-order mode.

Everything here is implemented clean-room from Ainslie, *Principles of Sonar
Performance Modelling* (Springer 2010), §9.1.1.2 (printed pp. 452-458):
Equations (9.42) to (9.61) and the seabed properties of Table 9.1
(:data:`WESTON_SEABEDS`). The quantity computed is Ainslie's **propagation
factor** ``F`` (units m⁻²), reported as the propagation loss
``PL = −10·lg F`` dB re 1 m², which equals the usual transmission loss for a
point source in free water.

The regime formulae are energy-flux (incoherent) results: they describe the
range-averaged field, not its modal interference. That is exactly what makes
them a usable cross-check for :mod:`phonometry.underwater.numerical_propagation`
-- the range average of a normal-mode or parabolic-equation field over many
interference cycles converges on the cylindrical-spreading law, with
``ψc = π/2`` for a totally reflecting (pressure-release) bottom.

.. note::
    Ainslie's Equation (9.57) for the mode-stripping/single-mode transition is
    printed as ``r_MS ≈ k²·He³/(9·η)``. Equating the effective angle of
    Equation (9.47) with the mode angle of Equation (9.56) at ``n = 3/2``, as
    the accompanying text prescribes, gives ``k²·He³/(9·π·η)`` instead: the
    printed form is a factor ``π`` too large. This module implements the
    derivation-consistent value and records the discrepancy in
    ``docs/ERRATA.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import erf

from .._internal.validation import require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Regime labels, in order of increasing range.
WESTON_REGIMES = ("spherical", "cylindrical", "mode-stripping", "single-mode")


@dataclass(frozen=True)
class WestonSeabed:
    """Characteristic seabed properties (Ainslie Table 9.1, printed p. 454).

    :ivar name: Sediment name.
    :ivar grain_size: Grain size ``Mz`` (phi units).
    :ivar sound_speed_ratio: ``c_sed/c_w``.
    :ivar density_ratio: ``ρ_sed/ρ_w``.
    :ivar attenuation_db_per_wavelength: ``β_sed``, in dB per wavelength.
    :ivar loss_parameter: ``ε = β_sed/(40·π·lg e)`` (Equation 9.23).
    :ivar sound_speed_gradient: ``c'``, the sediment sound-speed gradient, in
        s⁻¹ (0 for sand, 1 for mud).
    """

    name: str
    grain_size: float
    sound_speed_ratio: float
    density_ratio: float
    attenuation_db_per_wavelength: float
    loss_parameter: float
    sound_speed_gradient: float


#: The two characteristic seabeds tabulated in Ainslie Table 9.1: medium sand
#: (``Mz = 1.5``, a reflecting bottom with a critical angle) and mud
#: (``Mz = 8``, refracting, no critical angle).
WESTON_SEABEDS: dict[str, WestonSeabed] = {
    "sand": WestonSeabed(
        name="sand",
        grain_size=1.5,
        sound_speed_ratio=1.20,
        density_ratio=2.1,
        attenuation_db_per_wavelength=0.88,
        loss_parameter=0.0161,
        sound_speed_gradient=0.0,
    ),
    "mud": WestonSeabed(
        name="mud",
        grain_size=8.0,
        sound_speed_ratio=1.00,
        density_ratio=1.4,
        attenuation_db_per_wavelength=0.09,
        loss_parameter=0.00165,
        sound_speed_gradient=1.0,
    ),
}


def _seabed(seabed: str | WestonSeabed) -> WestonSeabed:
    if isinstance(seabed, WestonSeabed):
        return seabed
    key = str(seabed).strip().lower()
    if key not in WESTON_SEABEDS:
        raise ValueError(
            f"'seabed' must be one of {tuple(WESTON_SEABEDS)} or a WestonSeabed, got {seabed!r}."
        )
    return WESTON_SEABEDS[key]


def critical_grazing_angle(sound_speed_ratio: float) -> float:
    """Critical grazing angle ``ψc = arccos(c_w/c_sed)``, in radians.

    A seabed slower than the water (``c_sed ≤ c_w``, e.g. mud) has **no**
    critical angle; the function then returns ``0``, which correctly switches
    the reflection-loss gradient to the refracting-sediment branch of
    :func:`reflection_loss_gradient`.

    :param sound_speed_ratio: ``c_sed/c_w``, dimensionless and positive.
    :return: The critical grazing angle, in radians (``0`` if none exists).
    :raises ValueError: If the ratio is not positive and finite.
    """
    ratio = require_positive(sound_speed_ratio, "sound_speed_ratio")
    if ratio <= 1.0:
        return 0.0
    return float(np.arccos(1.0 / ratio))


def loss_parameter(attenuation_db_per_wavelength: float) -> float:
    """Sediment loss parameter ``ε = β_sed/(40·π·lg e)`` (Ainslie Eq. 9.23).

    :param attenuation_db_per_wavelength: ``β_sed``, in dB per wavelength.
    :return: The dimensionless loss parameter ``ε``.
    :raises ValueError: If the attenuation is negative or non-finite.
    """
    beta = float(attenuation_db_per_wavelength)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("'attenuation_db_per_wavelength' must be non-negative and finite.")
    return float(beta / (40.0 * np.pi * np.log10(np.e)))


def reflection_loss_gradient(
    seabed: str | WestonSeabed = "sand", *, frequency_hz: float | None = None
) -> float:
    """Reflection loss gradient ``η``, in nepers per radian.

    The rate at which the seabed reflection loss grows with grazing angle,
    ``|R(θ)| ≈ exp(−η·θ)`` (Ainslie Eq. 9.45). Two branches:

    * a **reflecting** seabed with a critical angle (sand, coarse silt),
      ``η = 2·ε·(ρ_sed/ρ_w)·cos²ψc/sin³ψc`` (Eq. 9.51), frequency-independent;
    * a **refracting** seabed with none (mud, clay, fine silt),
      ``η = 2·ω·ε/c'`` (Eq. 9.53), proportional to frequency.

    :param seabed: ``"sand"``, ``"mud"`` or an explicit :class:`WestonSeabed`.
    :param frequency_hz: Acoustic frequency, in Hz; required only for the
        refracting branch (``c' > 0``).
    :return: The reflection loss gradient ``η``, in Np/rad.
    :raises ValueError: If the frequency is missing or invalid for a refracting
        seabed.
    """
    bed = _seabed(seabed)
    psi_c = critical_grazing_angle(bed.sound_speed_ratio)
    if psi_c > 0.0:
        return float(
            2.0
            * bed.loss_parameter
            * bed.density_ratio
            * np.cos(psi_c) ** 2
            / np.sin(psi_c) ** 3
        )
    if bed.sound_speed_gradient <= 0.0:
        raise ValueError(
            "a seabed without a critical angle needs a positive 'sound_speed_gradient'"
            " to use the refracting branch of Equation (9.53)."
        )
    if frequency_hz is None:
        raise ValueError("'frequency_hz' is required for a refracting seabed (Equation 9.53).")
    f = require_positive(frequency_hz, "frequency_hz")
    return float(2.0 * (2.0 * np.pi * f) * bed.loss_parameter / bed.sound_speed_gradient)


def effective_depth(
    water_depth: float,
    frequency_hz: float,
    *,
    seabed: str | WestonSeabed = "sand",
    sound_speed: float = 1500.0,
) -> float:
    """Weston effective water depth ``He`` (Ainslie Eq. 9.55), in metres.

    ``He = H + (ρ_sed/ρ_w)/((ω/c_w)·sin ψc)``: the depth at which a
    pressure-release boundary appears to lie, a short distance below the true
    seabed. Only meaningful for a seabed with a critical angle.

    :param water_depth: Water-column depth ``H``, in metres.
    :param frequency_hz: Acoustic frequency, in Hz.
    :param seabed: ``"sand"``, ``"mud"`` or a :class:`WestonSeabed`.
    :param sound_speed: Water sound speed ``c_w``, in m/s.
    :return: The effective depth ``He``, in metres.
    :raises ValueError: If the seabed has no critical angle or an input is
        invalid.
    """
    h = require_positive(water_depth, "water_depth")
    f = require_positive(frequency_hz, "frequency_hz")
    c = require_positive(sound_speed, "sound_speed")
    bed = _seabed(seabed)
    psi_c = critical_grazing_angle(bed.sound_speed_ratio)
    if psi_c <= 0.0:
        raise ValueError(
            "'effective_depth' needs a seabed with a critical angle (c_sed > c_w);"
            f" {bed.name!r} has none."
        )
    k = 2.0 * np.pi * f / c
    return float(h + bed.density_ratio / (k * np.sin(psi_c)))


def waveguide_cutoff_frequency(
    water_depth: float,
    *,
    seabed: str | WestonSeabed = "sand",
    sound_speed: float = 1500.0,
) -> float:
    """Shallow-water waveguide cut-off frequency ``fc`` (Ainslie Eq. 9.60), in Hz.

    ``fc = (π − ρ_sed/ρ_w)/(2·π·sin ψc) · c_w/H`` -- below it no mode is cut on
    and ducted propagation does not occur.

    :param water_depth: Water-column depth ``H``, in metres.
    :param seabed: ``"sand"``, ``"mud"`` or a :class:`WestonSeabed`.
    :param sound_speed: Water sound speed ``c_w``, in m/s.
    :return: The cut-off frequency, in Hz.
    :raises ValueError: If the seabed has no critical angle or an input is
        invalid.
    """
    h = require_positive(water_depth, "water_depth")
    c = require_positive(sound_speed, "sound_speed")
    bed = _seabed(seabed)
    psi_c = critical_grazing_angle(bed.sound_speed_ratio)
    if psi_c <= 0.0:
        raise ValueError(
            "'waveguide_cutoff_frequency' needs a seabed with a critical angle"
            f" (c_sed > c_w); {bed.name!r} has none."
        )
    return float(
        (np.pi - bed.density_ratio) / (2.0 * np.pi * np.sin(psi_c)) * c / h
    )


@dataclass(frozen=True)
class WestonRegimeBoundaries:
    """Range boundaries between Weston's four propagation regimes.

    :ivar spherical_to_cylindrical: Range at which ``1/r²`` and ``2ψc/(rH)``
        are equal, ``H/(2·ψc)``, in metres.
    :ivar cylindrical_to_mode_stripping: Ainslie Eq. (9.50)
        ``r_CS = π·H/(4·η·ψc²)``, in metres (``inf`` for a lossless bottom).
    :ivar mode_stripping_to_single_mode: ``r_MS = k²·He³/(9·π·η)``, in metres
        (``inf`` for a lossless bottom). See the module note on Eq. (9.57).
    :ivar critical_angle: Critical grazing angle ``ψc``, in radians.
    :ivar reflection_loss_gradient: ``η``, in Np/rad.
    :ivar effective_depth: Weston effective depth ``He``, in metres.
    :ivar cutoff_frequency: Waveguide cut-off frequency, in Hz (``nan`` when the
        seabed has no critical angle).
    :ivar mode_count: Number of cut-on modes, ``(ω/c_w)·He·sin ψc/π``
        (Eq. 9.58), as a real number.
    """

    spherical_to_cylindrical: float
    cylindrical_to_mode_stripping: float
    mode_stripping_to_single_mode: float
    critical_angle: float
    reflection_loss_gradient: float
    effective_depth: float
    cutoff_frequency: float
    mode_count: float


def weston_regime_boundaries(
    frequency_hz: float,
    water_depth: float,
    *,
    seabed: str | WestonSeabed = "sand",
    sound_speed: float = 1500.0,
    critical_angle: float | None = None,
    reflection_loss_gradient_value: float | None = None,
) -> WestonRegimeBoundaries:
    """Regime boundaries of a shallow-water waveguide (Ainslie §9.1.1.2).

    :param frequency_hz: Acoustic frequency, in Hz.
    :param water_depth: Water-column depth ``H``, in metres.
    :param seabed: ``"sand"``, ``"mud"`` or a :class:`WestonSeabed`.
    :param sound_speed: Water sound speed ``c_w``, in m/s.
    :param critical_angle: Override the seabed critical angle ``ψc``, in
        degrees. Use ``90`` for the ideal totally reflecting waveguide.
    :param reflection_loss_gradient_value: Override ``η``, in Np/rad. Use ``0``
        for a lossless bottom (no mode stripping, no single-mode regime).
    :return: A :class:`WestonRegimeBoundaries`.
    :raises ValueError: If an input is invalid.

    .. note::
        The two overrides are independent: overriding ``critical_angle``
        alone leaves ``η`` computed from the seabed's *own* critical angle
        through Equation (9.51), which mixes two different bottoms. Pass both
        together (as the ideal-waveguide case ``critical_angle=90`` with
        ``reflection_loss_gradient_value=0`` does) whenever the intent is a
        hypothetical seabed rather than a tweak of the tabulated one.
    """
    f = require_positive(frequency_hz, "frequency_hz")
    h = require_positive(water_depth, "water_depth")
    c = require_positive(sound_speed, "sound_speed")
    bed = _seabed(seabed)
    psi_c, eta = _angle_and_gradient(
        bed, f, critical_angle, reflection_loss_gradient_value
    )
    k = 2.0 * np.pi * f / c
    if psi_c < np.pi / 2.0 and bed.sound_speed_ratio > 1.0:
        h_eff = h + bed.density_ratio / (k * np.sin(psi_c))
        f_cut = (np.pi - bed.density_ratio) / (2.0 * np.pi * np.sin(psi_c)) * c / h
    else:
        # An ideal (totally reflecting) bottom puts the pressure-release
        # boundary at the seabed itself and cuts on the first mode at c/(2H).
        h_eff = h
        f_cut = c / (2.0 * h) if psi_c >= np.pi / 2.0 else float("nan")
    r_ss_cs = h / (2.0 * psi_c)
    if eta > 0.0:
        r_cs = np.pi * h / (4.0 * eta * psi_c**2)
        # Equation (9.57) with the factor pi restored (see the module note).
        r_ms = k**2 * h_eff**3 / (9.0 * np.pi * eta)
    else:
        r_cs = float("inf")
        r_ms = float("inf")
    return WestonRegimeBoundaries(
        spherical_to_cylindrical=float(r_ss_cs),
        cylindrical_to_mode_stripping=float(r_cs),
        mode_stripping_to_single_mode=float(r_ms),
        critical_angle=float(psi_c),
        reflection_loss_gradient=float(eta),
        effective_depth=float(h_eff),
        cutoff_frequency=float(f_cut),
        mode_count=float(k * h_eff * np.sin(psi_c) / np.pi),
    )


def _angle_and_gradient(
    bed: WestonSeabed,
    frequency_hz: float,
    critical_angle: float | None,
    gradient: float | None,
) -> tuple[float, float]:
    """Resolve ``(ψc, η)`` from the seabed and the optional overrides."""
    if critical_angle is None:
        psi_c = critical_grazing_angle(bed.sound_speed_ratio)
        if psi_c <= 0.0:
            raise ValueError(
                f"seabed {bed.name!r} has no critical angle; pass 'critical_angle'"
                " explicitly (in degrees) to fix the trapped-ray cone."
            )
    else:
        deg = float(critical_angle)
        if not np.isfinite(deg) or not (0.0 < deg <= 90.0):
            raise ValueError("'critical_angle' must lie in (0, 90] degrees.")
        psi_c = np.radians(deg)
    if gradient is None:
        eta = reflection_loss_gradient(bed, frequency_hz=frequency_hz)
    else:
        eta = float(gradient)
        if not np.isfinite(eta) or eta < 0.0:
            raise ValueError("'reflection_loss_gradient_value' must be non-negative and finite.")
    return float(psi_c), float(eta)


@dataclass(frozen=True)
class WestonPropagationResult:
    """Weston regime propagation loss versus range.

    :ivar range_m: Ranges from the source, in metres.
    :ivar propagation_loss: Composite propagation loss ``PL = −10·lg F`` per
        range, in dB re 1 m².
    :ivar propagation_factor: The composite propagation factor ``F``, in m⁻².
    :ivar regime: The active regime label at each range (one of
        :data:`WESTON_REGIMES`).
    :ivar spherical: Spherical-spreading loss ``20·lg r`` at every range, in dB.
    :ivar cylindrical: Cylindrical-spreading loss (Eq. 9.42) at every range, dB.
    :ivar mode_stripping: Mode-stripping loss (Eq. 9.49) at every range, dB
        (``nan`` when the bottom is lossless: without reflection loss there is
        nothing to strip).
    :ivar single_mode: Single-mode loss (Eq. 9.54) at every range, in dB.
    :ivar multipath: Loss from the continuous multipath integral (Eq. 9.46),
        which joins the cylindrical and mode-stripping regimes smoothly, in dB.
    :ivar boundaries: The :class:`WestonRegimeBoundaries` in force.
    :ivar frequency: Acoustic frequency, in Hz.
    :ivar water_depth: Water-column depth ``H``, in metres.
    :ivar source_depth: Source depth ``z0``, in metres.
    :ivar receiver_depth: Receiver depth ``z``, in metres.
    :ivar seabed: Name of the seabed used.
    """

    range_m: NDArray[np.float64]
    propagation_loss: NDArray[np.float64]
    propagation_factor: NDArray[np.float64]
    regime: NDArray[np.str_]
    spherical: NDArray[np.float64]
    cylindrical: NDArray[np.float64]
    mode_stripping: NDArray[np.float64]
    single_mode: NDArray[np.float64]
    multipath: NDArray[np.float64]
    boundaries: WestonRegimeBoundaries
    frequency: float
    water_depth: float
    source_depth: float
    receiver_depth: float
    seabed: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the composite loss with each regime's law and the boundaries."""
        from .._i18n import check_language
        from .._plot.underwater import plot_weston_regimes

        return plot_weston_regimes(self, ax=ax, language=check_language(language), **kwargs)


def _to_db(factor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Propagation loss ``−10·lg F``, with a non-positive factor mapped to ``nan``."""
    with np.errstate(divide="ignore"):
        return np.asarray(-10.0 * np.log10(np.where(factor > 0.0, factor, np.nan)),
                          dtype=np.float64)


def weston_propagation_loss(
    range_m: NDArray[np.float64] | list[float] | float,
    frequency_hz: float,
    water_depth: float,
    *,
    seabed: str | WestonSeabed = "sand",
    sound_speed: float = 1500.0,
    source_depth: float | None = None,
    receiver_depth: float | None = None,
    critical_angle: float | None = None,
    reflection_loss_gradient_value: float | None = None,
) -> WestonPropagationResult:
    """Propagation loss across Weston's four shallow-water regimes.

    Assembles the piecewise loss from Ainslie's Equations (9.42), (9.49) and
    (9.54), switching regime at the boundaries of
    :func:`weston_regime_boundaries`, and returns each regime's own law over the
    whole range grid so the transitions can be drawn.

    :param range_m: Range(s) from the source, in metres (scalar or array,
        strictly positive).
    :param frequency_hz: Acoustic frequency, in Hz.
    :param water_depth: Water-column depth ``H``, in metres.
    :param seabed: ``"sand"``, ``"mud"`` or a :class:`WestonSeabed`.
    :param sound_speed: Water sound speed ``c_w``, in m/s.
    :param source_depth: Source depth ``z0``, in metres; defaults to ``H/2``
        (used only by the single-mode formula).
    :param receiver_depth: Receiver depth ``z``, in metres; defaults to ``H/2``.
    :param critical_angle: Override ``ψc``, in degrees (``90`` for an ideal
        totally reflecting waveguide).
    :param reflection_loss_gradient_value: Override ``η``, in Np/rad (``0`` for
        a lossless bottom: no mode stripping, no single-mode regime).
    :return: A :class:`WestonPropagationResult`.
    :raises ValueError: If an input is invalid.
    """
    f = require_positive(frequency_hz, "frequency_hz")
    h = require_positive(water_depth, "water_depth")
    c = require_positive(sound_speed, "sound_speed")
    r = np.atleast_1d(np.asarray(range_m, dtype=np.float64))
    if r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("'range_m' must be finite and non-empty.")
    if np.any(r <= 0.0):
        raise ValueError("'range_m' must be strictly positive.")
    z0 = h / 2.0 if source_depth is None else float(source_depth)
    zr = h / 2.0 if receiver_depth is None else float(receiver_depth)
    for name, value in (("source_depth", z0), ("receiver_depth", zr)):
        if not np.isfinite(value) or not (0.0 <= value <= h):
            raise ValueError(f"'{name}' must lie within the water column [0, H].")

    bed = _seabed(seabed)
    bounds = weston_regime_boundaries(
        f, h, seabed=bed, sound_speed=c, critical_angle=critical_angle,
        reflection_loss_gradient_value=reflection_loss_gradient_value,
    )
    psi_c = bounds.critical_angle
    eta = bounds.reflection_loss_gradient
    h_eff = bounds.effective_depth
    lam = c / f

    f_ss = 1.0 / r**2                                            # Eq. (9.43)/(9.44)
    f_cs = 2.0 * psi_c / (r * h)                                 # Eq. (9.42)
    if eta > 0.0:
        theta_eff = np.sqrt(np.pi * h / (4.0 * eta * r))         # Eq. (9.47)
        f_mp = (2.0 * theta_eff / (r * h)) * erf(
            np.sqrt(np.pi) * psi_c / (2.0 * theta_eff))          # Eq. (9.46)
        f_ms = np.sqrt(np.pi / (eta * h)) * r ** (-1.5)          # Eq. (9.49)
    else:
        f_mp = f_cs
        f_ms = np.zeros_like(r)
    f_sm = (                                                     # Eq. (9.54)
        4.0 * lam / (h_eff**2 * r)
        * np.sin(np.pi * z0 / h_eff) ** 2
        * np.sin(np.pi * zr / h_eff) ** 2
        * np.exp(-eta * lam**2 * r / (4.0 * h_eff**3))
    )

    labels = np.full(r.shape, WESTON_REGIMES[0], dtype="<U14")
    factor = f_ss.copy()
    in_cs = r >= bounds.spherical_to_cylindrical
    labels[in_cs] = WESTON_REGIMES[1]
    factor[in_cs] = f_cs[in_cs]
    in_ms = r >= bounds.cylindrical_to_mode_stripping
    labels[in_ms] = WESTON_REGIMES[2]
    factor[in_ms] = f_ms[in_ms]
    in_sm = r >= bounds.mode_stripping_to_single_mode
    labels[in_sm] = WESTON_REGIMES[3]
    factor[in_sm] = f_sm[in_sm]

    return WestonPropagationResult(
        range_m=r,
        propagation_loss=_to_db(factor),
        propagation_factor=factor,
        regime=labels,
        spherical=_to_db(f_ss),
        cylindrical=_to_db(f_cs),
        mode_stripping=_to_db(f_ms),
        single_mode=_to_db(f_sm),
        multipath=_to_db(f_mp),
        boundaries=bounds,
        frequency=f,
        water_depth=h,
        source_depth=z0,
        receiver_depth=zr,
        seabed=bed.name,
    )
