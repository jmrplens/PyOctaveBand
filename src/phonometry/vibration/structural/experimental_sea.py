#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Experimental statistical energy analysis: coupling loss factors from measured
energies (Norton & Karczub Ch. 6).

Statistical energy analysis (SEA) has two routes to the coupling loss factor
``eta_ij``. The **predictive** route derives it from a wave transmission
coefficient at the junction, which is what
:func:`~phonometry.vibration.structural.junction_transmission.coupling_loss_factor` does
with the closed-form coefficients of
:func:`~phonometry.vibration.structural.junction_transmission.junction_transmission`. The
**experimental** route, implemented here, inverts the steady-state power
balance from *measured* subsystem energies: it needs no model of the junction
at all, and it is the only route open for real joints (welds, bolt rows,
spot welds, adhesives) whose wave behaviour is not tractable.

For two subsystems the steady-state power balance is (Norton Eqs. 6.10 and
6.11, generalised to a drive on either subsystem):

.. math::

   \Pi_1 = \omega \left[ (\eta_1 + \eta_{12}) E_1
   - \eta_{21} E_2 \right] \tag{6.10}

   \Pi_2 = \omega \left[ (\eta_2 + \eta_{21}) E_2
   - \eta_{12} E_1 \right] \tag{6.11}

with :math:`E_i = M_i \langle v_i^2 \rangle` the band energy of subsystem
``i`` (mass times the space- and time-averaged mean-square velocity),
``eta_i`` its internal loss factor and ``omega`` the band centre frequency
in rad/s. Two inversions follow.

**Single drive plus reciprocity** (:func:`power_injection_clf`). Drive
subsystem 1 only. The second equation with :math:`\Pi_2 = 0` gives one
relation between ``eta_12`` and ``eta_21``; the SEA consistency
(reciprocity) relationship :math:`\eta_{12} n_1 = \eta_{21} n_2` (Eq. 6.8)
supplies the second, so with the modal densities ``n_1``, ``n_2`` known
(Eq. 6.15):

.. math::

   \eta_{12} = \frac{\eta_2 E_2}{E_1 - E_2\, n_1/n_2}

   \eta_{21} = \eta_{12} \frac{n_1}{n_2}

   \Pi_{\text{in}} = \omega (\eta_1 E_1 + \eta_2 E_2)

The input power reduces to the total dissipated power, as it must in the
steady state: substituting the balance of subsystem 2 into Eq. (6.10)
cancels the two coupling terms exactly. The bracket
:math:`E_1 - E_2\, n_1/n_2` is positive exactly when the *modal* energy of
the driven subsystem exceeds that of the receiver,
:math:`E_1/n_1 > E_2/n_2`; a measurement that violates it is not a
two-subsystem SEA system and is rejected.

**Two drives, no reciprocity assumed** (:func:`power_injection_matrix`). The
classical power-injection method drives each subsystem in turn and measures
both energies each time, giving four equations for the four unknowns
``eta_1``, ``eta_2``, ``eta_12``, ``eta_21`` with no prior assumption at all.
Reciprocity then becomes a *check* on the measurement rather than an input:
:attr:`PowerInjectionResult.modal_density_ratio` compares the measured
:math:`\eta_{12}/\eta_{21}` with the expected :math:`n_2/n_1`.

Modal densities for the usual subsystems are provided as well
(:func:`flat_plate_modal_density`, :func:`cylindrical_shell_modal_density`,
:func:`bar_modal_density`, :func:`beam_modal_density`), following Norton
Eqs. 6.23-6.29. The flat-plate expression
:math:`n(f) = S \sqrt{12} / (2 c_\mathrm{L} t)` is the same quantity as EN 12354-4's
:math:`n = \pi S f_\mathrm{c} / c_0^2`
(:func:`~phonometry.building.measurement.flanking_transmission.modal_density`), only
parametrised by the plate itself rather than by its critical frequency; the
two agree identically and a regression test pins that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import (
    require_choice,
    require_positive,
    require_positive_array,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PowerInjectionResult",
    "bar_modal_density",
    "beam_modal_density",
    "cylindrical_shell_modal_density",
    "flat_plate_modal_density",
    "power_injection_clf",
    "power_injection_matrix",
    "ring_frequency",
]

#: Bandwidth factor ``F = sqrt(f_upper/f_lower)`` of the usual band widths,
#: used by the above-ring-frequency cylinder modal density (Norton Eq. 6.29).
_BANDWIDTH_FACTOR: dict[str, float] = {"third": 1.122, "octave": 1.414}


# ---------------------------------------------------------------------------
# Modal densities (Norton 6.4.1)
# ---------------------------------------------------------------------------
def bar_modal_density(length: float, longitudinal_wave_speed: float) -> float:
    r"""Modal density of a uniform bar in longitudinal vibration (Eq. 6.23).

    :math:`n(f) = 2 L / c_\mathrm{L}`, independent of frequency.

    :param length: Bar length ``L``, in m (> 0).
    :param longitudinal_wave_speed: Bar wave speed
        :math:`c_\mathrm{L} = \sqrt{E/\rho}`, in m/s (> 0).
    :return: The modal density ``n(f)``, in modes per hertz.
    :raises ValueError: for a non-positive input.
    """
    ell = require_positive(length, "length")
    c_l = require_positive(longitudinal_wave_speed, "longitudinal_wave_speed")
    return 2.0 * ell / c_l


def beam_modal_density(
    frequency: ArrayLike,
    length: float,
    mass_per_length: float,
    bending_stiffness: float,
) -> NDArray[np.float64]:
    r"""Modal density of a uniform beam in flexure (Norton Eq. 6.24).

    :math:`n(f) = L (\rho A / E I)^{1/4} / \sqrt{2\pi f}`: unlike every
    other subsystem here, it *decreases* with frequency.

    :param frequency: Band centre frequency ``f``, in hertz (scalar or array,
        > 0).
    :param length: Beam length ``L``, in m (> 0).
    :param mass_per_length: Mass per unit length ``rho A``, in kg/m (> 0).
    :param bending_stiffness: Flexural stiffness ``E I``, in N.m^2 (> 0).
    :return: The modal density ``n(f)``, in modes per hertz.
    :raises ValueError: for a non-positive input.
    """
    f = require_positive_array(frequency, "frequency")
    ell = require_positive(length, "length")
    mu = require_positive(mass_per_length, "mass_per_length")
    ei = require_positive(bending_stiffness, "bending_stiffness")
    return np.asarray(
        ell * (mu / ei) ** 0.25 / np.sqrt(2.0 * np.pi * f), dtype=np.float64
    )


def flat_plate_modal_density(
    area: float, thickness: float, longitudinal_wave_speed: float
) -> float:
    r"""Modal density of a flat plate in flexure (Norton Eq. 6.25).

    :math:`n(f) = S \sqrt{12} / (2 c_\mathrm{L} t)`, independent of frequency, with
    the plate (quasi-longitudinal) wave speed
    :math:`c_\mathrm{L} = \sqrt{E / (\rho (1 - \nu^2))}`.

    :param area: Plate surface area ``S``, in m^2 (> 0).
    :param thickness: Plate thickness ``t``, in m (> 0).
    :param longitudinal_wave_speed: Plate wave speed ``cL``, in m/s (> 0).
    :return: The modal density ``n(f)``, in modes per hertz.
    :raises ValueError: for a non-positive input.
    """
    s = require_positive(area, "area")
    t = require_positive(thickness, "thickness")
    c_l = require_positive(longitudinal_wave_speed, "longitudinal_wave_speed")
    return s * math.sqrt(12.0) / (2.0 * c_l * t)


def ring_frequency(mean_radius: float, longitudinal_wave_speed: float) -> float:
    r"""Ring frequency of a cylindrical shell (Norton Eq. 6.26).

    :math:`f_\mathrm{r} = c_\mathrm{L} / (2\pi a_\mathrm{m})`: the frequency at which the shell vibrates
    uniformly in the breathing mode. Above it a cylinder behaves like a flat
    plate; below it the modes group by circumferential order and the modal
    density is no longer a simple function of frequency.

    :param mean_radius: Mean shell radius ``a_m``, in m (> 0).
    :param longitudinal_wave_speed: Plate wave speed ``cL``, in m/s (> 0).
    :return: The ring frequency ``fr``, in hertz.
    :raises ValueError: for a non-positive input.
    """
    a_m = require_positive(mean_radius, "mean_radius")
    c_l = require_positive(longitudinal_wave_speed, "longitudinal_wave_speed")
    return c_l / (2.0 * math.pi * a_m)


def cylindrical_shell_modal_density(
    frequency: ArrayLike,
    area: float,
    thickness: float,
    mean_radius: float,
    longitudinal_wave_speed: float,
    *,
    band: str = "octave",
) -> NDArray[np.float64]:
    r"""Average modal density of a thin-walled cylinder (Norton 6.27-6.29).

    The semi-empirical approximations of Szechenyi, as collected by Clarkson &
    Pope, in three regimes of :math:`x = f / f_\mathrm{r}` around the ring frequency
    :func:`ring_frequency`:

    .. math::

       n = \frac{5 S}{\pi c_\mathrm{L} t} \sqrt{x},
       \qquad x \le 0.48 \tag{6.27}

       n = \frac{7.2 S}{\pi c_\mathrm{L} t}\, x,
       \qquad 0.48 < x \le 0.83 \tag{6.28}

       n = \frac{2 S}{\pi c_\mathrm{L} t} \left[ 2 + \frac{0.596}{F - 1/F}
       \left( F \arccos\frac{1.745}{F^2 x^2}
       - \frac{1}{F} \arccos\frac{1.745 F^2}{x^2} \right) \right],
       \qquad x > 0.83 \tag{6.29}

    with the bandwidth factor
    :math:`F = \sqrt{f_{\text{upper}}/f_{\text{lower}}}`. These are
    *average* values: they do not resolve the large fluctuations that the
    cut-on of successive circumferential orders produces below the ring
    frequency, which for long thin shells can be substantial.

    :param frequency: Band centre frequency ``f``, in hertz (scalar or array,
        > 0).
    :param area: Shell surface area ``S``, in m^2 (> 0).
    :param thickness: Wall thickness ``t``, in m (> 0).
    :param mean_radius: Mean shell radius ``a_m``, in m (> 0).
    :param longitudinal_wave_speed: Plate wave speed ``cL``, in m/s (> 0).
    :param band: Analysis bandwidth, ``"octave"`` (Default,
        :math:`F = 1.414`) or ``"third"`` (:math:`F = 1.122`).
    :return: The modal density ``n(f)``, in modes per hertz.
    :raises ValueError: for a non-positive input or an unknown band.
    """
    f = require_positive_array(frequency, "frequency")
    s = require_positive(area, "area")
    t = require_positive(thickness, "thickness")
    c_l = require_positive(longitudinal_wave_speed, "longitudinal_wave_speed")
    f_r = ring_frequency(mean_radius, longitudinal_wave_speed)
    factor = _BANDWIDTH_FACTOR[require_choice(band, "band", ("octave", "third"))]

    x = f / f_r
    base = s / (math.pi * c_l * t)
    low = 5.0 * base * np.sqrt(x)
    mid = 7.2 * base * x
    # Eq. (6.29): the arccos arguments leave [-1, 1] just above the 0,83
    # threshold for narrow bands, so clip them; the branch is only selected
    # where the expression is meaningful anyway.
    arg_1 = np.clip(1.745 / (factor**2 * np.maximum(x, 1e-12) ** 2), -1.0, 1.0)
    arg_2 = np.clip(1.745 * factor**2 / np.maximum(x, 1e-12) ** 2, -1.0, 1.0)
    shape = 0.596 / (factor - 1.0 / factor)
    high = (
        2.0
        * base
        * (2.0 + shape * (factor * np.arccos(arg_1) - np.arccos(arg_2) / factor))
    )
    n = np.where(x <= 0.48, low, np.where(x <= 0.83, mid, high))
    return np.asarray(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# Power-injection inversion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerInjectionResult:
    """Loss-factor budget of a two-subsystem SEA model, per band.

    All arrays share the band axis :attr:`frequencies`.

    :ivar frequencies: Band centre frequencies ``f``, in hertz.
    :ivar coupling_loss_factor12: ``eta_12``, subsystem 1 to 2.
    :ivar coupling_loss_factor21: ``eta_21``, subsystem 2 to 1.
    :ivar internal_loss_factor1: ``eta_1`` of subsystem 1.
    :ivar internal_loss_factor2: ``eta_2`` of subsystem 2.
    :ivar energy1: Band energy ``E_1``, in joules.
    :ivar energy2: Band energy ``E_2``, in joules.
    :ivar input_power1: Power injected into subsystem 1, in watts.
    :ivar input_power2: Power injected into subsystem 2, in watts.
    :ivar modal_density1: Modal density ``n_1``, in modes per hertz, or
        ``None`` when the inversion did not need it.
    :ivar modal_density2: Modal density ``n_2``, or ``None``.
    :ivar method: ``"single-drive"`` (reciprocity assumed) or ``"two-drive"``
        (full power-injection matrix).
    """

    frequencies: NDArray[np.float64]
    coupling_loss_factor12: NDArray[np.float64]
    coupling_loss_factor21: NDArray[np.float64]
    internal_loss_factor1: NDArray[np.float64]
    internal_loss_factor2: NDArray[np.float64]
    energy1: NDArray[np.float64]
    energy2: NDArray[np.float64]
    input_power1: NDArray[np.float64]
    input_power2: NDArray[np.float64]
    modal_density1: NDArray[np.float64] | None
    modal_density2: NDArray[np.float64] | None
    method: str

    @property
    def input_power(self) -> NDArray[np.float64]:
        """Total power injected into the system, in watts."""
        return np.asarray(self.input_power1 + self.input_power2, dtype=np.float64)

    @property
    def transmitted_power(self) -> NDArray[np.float64]:
        r"""Net power flowing from subsystem 1 to 2, in watts.

        :math:`\Pi_{12} = \omega (\eta_{12} E_1 - \eta_{21} E_2)`; negative
        when the net flow runs the other way.
        """
        omega = 2.0 * np.pi * self.frequencies
        net = omega * (
            self.coupling_loss_factor12 * self.energy1
            - self.coupling_loss_factor21 * self.energy2
        )
        return np.asarray(net, dtype=np.float64)

    @property
    def dissipated_power(self) -> NDArray[np.float64]:
        r"""Total power dissipated internally, in watts.

        :math:`\omega (\eta_1 E_1 + \eta_2 E_2)`, which equals
        :attr:`input_power` in the steady state and is the round-trip check
        of the inversion.
        """
        omega = 2.0 * np.pi * self.frequencies
        loss = omega * (
            self.internal_loss_factor1 * self.energy1
            + self.internal_loss_factor2 * self.energy2
        )
        return np.asarray(loss, dtype=np.float64)

    @property
    def coupling_strength(self) -> NDArray[np.float64]:
        r"""Coupling ratio :math:`\eta_{12} / \eta_1`.

        SEA subsystems should be *weakly* coupled: values well below 1 mean
        the junction leaks far less power than the subsystem dissipates, the
        regime where the two-subsystem model is trustworthy.
        """
        return np.asarray(
            self.coupling_loss_factor12 / self.internal_loss_factor1,
            dtype=np.float64,
        )

    @property
    def modal_density_ratio(self) -> NDArray[np.float64]:
        r"""Ratio :math:`n_1 / n_2` implied by the measured coupling loss
        factors.

        From the reciprocity relationship
        :math:`\eta_{12} n_1 = \eta_{21} n_2`, so it equals
        :math:`\eta_{21} / \eta_{12}`. For a ``"two-drive"`` inversion, where
        reciprocity was never assumed, comparing this with the modal densities
        computed from the geometry is the consistency check on the
        measurement.
        """
        return np.asarray(
            self.coupling_loss_factor21 / self.coupling_loss_factor12,
            dtype=np.float64,
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the loss-factor budget against frequency.

        The two coupling loss factors and the two internal loss factors on one
        log axis: their ordering is the whole diagnosis, because SEA is only
        valid where the coupling stays below the internal damping.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the ``eta_12`` curve.
        """
        from ..._i18n import check_language
        from ..._plot.vibration import plot_power_injection

        check_language(language)
        return plot_power_injection(self, ax=ax, language=language, **kwargs)


def _broadcast_bands(
    frequency: ArrayLike, values: dict[str, ArrayLike]
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Validate a band axis and broadcast every named positive array onto it."""
    f = require_positive_array(frequency, "frequency")
    f = np.atleast_1d(f)
    out: dict[str, NDArray[np.float64]] = {}
    for name, value in values.items():
        arr = np.atleast_1d(require_positive_array(value, name))
        if arr.size not in (1, f.size):
            raise ValueError(
                f"'{name}' must be a scalar or match the {f.size} frequency "
                f"bands, got {arr.size} values."
            )
        out[name] = np.broadcast_to(arr, f.shape).astype(np.float64)
    return f, out


def power_injection_clf(
    frequency: ArrayLike,
    energy1: ArrayLike,
    energy2: ArrayLike,
    internal_loss_factor1: ArrayLike,
    internal_loss_factor2: ArrayLike,
    modal_density1: ArrayLike,
    modal_density2: ArrayLike,
) -> PowerInjectionResult:
    r"""Coupling loss factors from a single-drive energy measurement.

    Subsystem 1 is driven and subsystem 2 receives power only through the
    junction. Inverting the steady-state balance of subsystem 2 together with
    the reciprocity relationship :math:`\eta_{12} n_1 = \eta_{21} n_2` gives

    .. math::

       \eta_{12} = \frac{\eta_2 E_2}{E_1 - E_2\, n_1/n_2}

       \eta_{21} = \eta_{12} \frac{n_1}{n_2}

       \Pi_{\text{in}} = \omega (\eta_1 E_1 + \eta_2 E_2)

    Energies are :math:`E_i = M_i \langle v_i^2 \rangle` with ``M_i`` the
    subsystem mass and :math:`\langle v_i^2 \rangle` the space- and
    time-averaged mean-square velocity in the band.

    :param frequency: Band centre frequencies ``f``, in hertz (> 0).
    :param energy1: Band energy ``E_1`` of the driven subsystem, in J (> 0).
    :param energy2: Band energy ``E_2`` of the receiving subsystem, in J
        (> 0).
    :param internal_loss_factor1: ``eta_1`` (> 0), scalar or per band.
    :param internal_loss_factor2: ``eta_2`` (> 0), scalar or per band.
    :param modal_density1: ``n_1``, in modes per hertz (> 0).
    :param modal_density2: ``n_2``, in modes per hertz (> 0).
    :return: A :class:`PowerInjectionResult` (method ``"single-drive"``).
    :raises ValueError: for a non-positive input, mismatched band lengths, or
        a measurement with :math:`E_1/n_1 \le E_2/n_2` (the receiving
        subsystem
        holds at least as much modal energy as the driven one, so no
        two-subsystem SEA model fits it).
    """
    f, v = _broadcast_bands(
        frequency,
        {
            "energy1": energy1,
            "energy2": energy2,
            "internal_loss_factor1": internal_loss_factor1,
            "internal_loss_factor2": internal_loss_factor2,
            "modal_density1": modal_density1,
            "modal_density2": modal_density2,
        },
    )
    e_1, e_2 = v["energy1"], v["energy2"]
    eta_1, eta_2 = v["internal_loss_factor1"], v["internal_loss_factor2"]
    n_1, n_2 = v["modal_density1"], v["modal_density2"]

    ratio = n_1 / n_2
    denominator = e_1 - e_2 * ratio
    if np.any(denominator <= 0.0):
        bad = f[denominator <= 0.0]
        raise ValueError(
            "the driven subsystem must hold more modal energy than the "
            "receiver (E1/n1 > E2/n2) for a two-subsystem SEA inversion; "
            f"violated at {np.array2string(bad, precision=4)} Hz."
        )
    eta_12 = eta_2 * e_2 / denominator
    eta_21 = eta_12 * ratio
    power_in = 2.0 * np.pi * f * (eta_1 * e_1 + eta_2 * e_2)
    return PowerInjectionResult(
        frequencies=f,
        coupling_loss_factor12=np.asarray(eta_12, dtype=np.float64),
        coupling_loss_factor21=np.asarray(eta_21, dtype=np.float64),
        internal_loss_factor1=eta_1,
        internal_loss_factor2=eta_2,
        energy1=e_1,
        energy2=e_2,
        input_power1=np.asarray(power_in, dtype=np.float64),
        input_power2=np.zeros_like(f),
        modal_density1=n_1,
        modal_density2=n_2,
        method="single-drive",
    )


def power_injection_matrix(
    frequency: ArrayLike,
    energies: ArrayLike,
    input_powers: ArrayLike,
) -> PowerInjectionResult:
    """Full two-drive power-injection inversion (no reciprocity assumed).

    Each subsystem is driven in turn with a known injected power while both
    band energies are measured, giving four equations for the four unknowns
    ``eta_1``, ``eta_2``, ``eta_12``, ``eta_21``. Because reciprocity is not
    used, :attr:`PowerInjectionResult.modal_density_ratio` becomes an
    independent check on the measurement.

    :param frequency: Band centre frequencies ``f``, in hertz (> 0), shape
        ``(nb,)``.
    :param energies: Measured band energies, in joules, shape ``(2, 2, nb)``:
        ``energies[i][j]`` is the energy of subsystem ``i`` while subsystem
        ``j`` is driven (all > 0).
    :param input_powers: Injected powers, in watts, shape ``(2, nb)``:
        ``input_powers[j]`` is the power injected into subsystem ``j`` during
        test ``j`` (all > 0).
    :return: A :class:`PowerInjectionResult` (method ``"two-drive"``); its
        ``energy1``/``energy2`` and ``input_power1``/``input_power2`` report
        the first test (subsystem 1 driven).
    :raises ValueError: for a bad shape, a non-positive value, or a singular
        energy matrix in some band.
    """
    f = np.atleast_1d(require_positive_array(frequency, "frequency"))
    e = np.asarray(energies, dtype=np.float64)
    p = np.asarray(input_powers, dtype=np.float64)
    if e.shape != (2, 2, f.size):
        raise ValueError(f"'energies' must have shape (2, 2, {f.size}), got {e.shape}.")
    if p.shape != (2, f.size):
        raise ValueError(
            f"'input_powers' must have shape (2, {f.size}), got {p.shape}."
        )
    if not np.all(np.isfinite(e)) or np.any(e <= 0.0):
        raise ValueError("'energies' must be finite and positive.")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("'input_powers' must be finite and positive.")

    omega = 2.0 * np.pi * f
    # Test j gives  (eta1 + eta12) E1j - eta21 E2j = P1j/omega  and
    #               (eta2 + eta21) E2j - eta12 E1j = P2j/omega,
    # with P1 = (Pin1, 0) and P2 = (0, Pin2) across the two tests.
    p_1 = np.stack([p[0], np.zeros_like(f)], axis=0)
    p_2 = np.stack([np.zeros_like(f), p[1]], axis=0)
    # a = eta1 + eta12, b = eta21 from the subsystem-1 rows.
    m_a = np.stack(
        [
            np.stack([e[0, 0], -e[1, 0]], axis=-1),
            np.stack([e[0, 1], -e[1, 1]], axis=-1),
        ],
        axis=-2,
    )
    # c = eta2 + eta21, d = eta12 from the subsystem-2 rows.
    m_c = np.stack(
        [
            np.stack([e[1, 0], -e[0, 0]], axis=-1),
            np.stack([e[1, 1], -e[0, 1]], axis=-1),
        ],
        axis=-2,
    )
    if np.any(np.abs(np.linalg.det(m_a)) < np.finfo(np.float64).tiny):
        raise ValueError(
            "the measured energy matrix is singular in at least one band; the "
            "two drive tests must produce independent energy distributions."
        )
    # numpy>=2 only treats the right-hand side as a stack of vectors when it is
    # 1-D, so give it the explicit trailing axis and drop it again.
    ab = np.linalg.solve(m_a, (p_1.T / omega[:, None])[..., None])[..., 0]
    cd = np.linalg.solve(m_c, (p_2.T / omega[:, None])[..., None])[..., 0]
    eta_21, eta_12 = ab[:, 1], cd[:, 1]
    eta_1 = ab[:, 0] - eta_12
    eta_2 = cd[:, 0] - eta_21
    return PowerInjectionResult(
        frequencies=f,
        coupling_loss_factor12=np.asarray(eta_12, dtype=np.float64),
        coupling_loss_factor21=np.asarray(eta_21, dtype=np.float64),
        internal_loss_factor1=np.asarray(eta_1, dtype=np.float64),
        internal_loss_factor2=np.asarray(eta_2, dtype=np.float64),
        energy1=np.asarray(e[0, 0], dtype=np.float64),
        energy2=np.asarray(e[1, 0], dtype=np.float64),
        input_power1=np.asarray(p[0], dtype=np.float64),
        input_power2=np.zeros_like(f),
        modal_density1=None,
        modal_density2=None,
        method="two-drive",
    )
