#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Radiation efficiency of a plate in bending (Hopkins 2007, Sound Insulation,
Section 2.9; Leppington et al. 1982; Maidanik 1962).

The **radiation efficiency** ``sigma`` of a vibrating plate relates the
airborne sound power it radiates, in watts, to its mean-square surface
velocity:

.. math::

   P = \rho_0 c_0 S \langle v^2 \rangle \sigma

so ``sigma`` is exactly the radiation factor ``epsilon`` that
:func:`phonometry.sound_power_from_vibration` (ISO/TS 7849) otherwise takes as a
measured input: this module predicts it from the plate geometry and its
coincidence (critical) frequency, closing the ISO 7849 chain without a power
measurement, and it supplies the *resonant* transmission path of the single- and
double-leaf sound-reduction-index predictions in
:mod:`phonometry.building.panel_transmission`.

**Coincidence (critical) frequency (Hopkins Eq. 2.201).** Below it the free
bending wavelength is shorter than the acoustic wavelength, so the plate
radiates weakly; above it the bending wave is supersonic and radiates
efficiently:

.. math::

   f_c = \frac{c_0^2}{2\pi} \sqrt{\frac{m''}{B'}} \tag{Eq. 2.201}

with ``m''`` the mass per unit area (kg/m^2) and ``B'`` the bending
stiffness per unit width (N.m). This is the closed form
:math:`f_c = 0.55\,c_0^2 / (c_L h)` in terms of the plate longitudinal wave
speed ``cL`` and thickness ``h``.

**Frequency-averaged efficiency (Hopkins 2.9.4, "method no. 1").** With
:math:`\mu = \sqrt{f_c / f}` (Eq. 2.228), the perimeter ``U``, area ``S``,
the boundary constant :math:`C_{BC}` (1 simply supported, 2 clamped) and the
baffle-orientation constant :math:`C_{OB}` (1 plate flush in an infinite
baffle, 2 baffles perpendicular to the edges), the efficiency below ``fc``
is

.. math::

   \sigma = \frac{U}{2\pi \mu k S \sqrt{\mu^2 - 1}}
   \left[ \ln\frac{\mu + 1}{\mu - 1} + \frac{2\mu}{\mu^2 - 1} \right]
   \left[ C_{BC} C_{OB} - \mu^{-8} \left( C_{BC} C_{OB} - 1 \right) \right]
   \tag{Eq. 2.227}

with :math:`k = 2\pi f / c_0` the acoustic wavenumber. Above ``fc``,
:math:`\sigma = 1 / \sqrt{1 - \mu^2} = (1 - f_c/f)^{-0.5}` (Eq. 2.229), so
:math:`\sigma \to 1` well above coincidence. In the band that contains
``fc``,

.. math::

   \sigma \approx \left( 0.5 - 0.15 \frac{L_1}{L_2} \right)
   \sqrt{k_{fc} L_1}, \qquad k_{fc} = \frac{2\pi f_c}{c_0}
   \tag{Eq. 2.230}

with ``L1`` the smaller and ``L2`` the larger plate dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.validation import require_choice, require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Default speed of sound in air ``c0``, m/s (20 degC).
_SPEED_OF_SOUND: float = 343.0
#: Boundary-condition constant ``C_BC`` (Hopkins Eq. 2.227).
_C_BC: dict[str, float] = {"simply_supported": 1.0, "clamped": 2.0}
#: Baffle-orientation constant ``C_OB`` (Hopkins Eq. 2.227).
_C_OB: dict[str, float] = {"infinite": 1.0, "perpendicular": 2.0}
#: Half a one-third octave (ratio) around fc for the at-coincidence band.
_COINCIDENCE_HALF_WIDTH: float = 2.0 ** (1.0 / 6.0)

__all__ = [
    "RadiationEfficiencyResult",
    "coincidence_frequency",
    "radiation_efficiency",
]


def coincidence_frequency(
    mass_per_area: float,
    bending_stiffness: float,
    *,
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> float:
    r"""Coincidence (critical) frequency ``fc`` of a thin plate (Hopkins 2.201).

    :math:`f_c = (c_0^2 / 2\pi) \sqrt{m'' / B'}` (identical to Bies Eq. 7.3).

    :param mass_per_area: Mass per unit area ``m''``, in kg/m^2.
    :param bending_stiffness: Bending stiffness per unit width ``B'``, in N.m
        (see :func:`phonometry.vibration.point_mobility.plate_bending_stiffness`).
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :return: The coincidence frequency ``fc``, in hertz.
    :raises ValueError: for a non-positive input.
    """
    m2 = require_positive(mass_per_area, "mass_per_area")
    b = require_positive(bending_stiffness, "bending_stiffness")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    return float(c0**2 / (2.0 * np.pi) * np.sqrt(m2 / b))


@dataclass(frozen=True)
class RadiationEfficiencyResult:
    """Frequency-averaged plate radiation efficiency (Hopkins 2.9.4).

    :ivar frequencies: Band centre frequencies, in hertz.
    :ivar radiation_efficiency: Radiation efficiency ``sigma`` per band.
    :ivar critical_frequency: Coincidence frequency ``fc``, in hertz.
    :ivar length_x: Plate dimension ``Lx``, in m.
    :ivar length_y: Plate dimension ``Ly``, in m.
    :ivar boundary: Boundary condition (``"simply_supported"`` / ``"clamped"``).
    :ivar baffle: Baffle orientation (``"infinite"`` / ``"perpendicular"``).
    """

    frequencies: np.ndarray
    radiation_efficiency: np.ndarray
    critical_frequency: float
    length_x: float
    length_y: float
    boundary: str
    baffle: str

    @property
    def radiation_index(self) -> np.ndarray:
        r"""Radiation index :math:`10 \log_{10}(\sigma)` per band, in dB."""
        sigma = np.asarray(self.radiation_efficiency, dtype=np.float64)
        return np.asarray(10.0 * np.log10(sigma), dtype=np.float64)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the radiation efficiency ``sigma(f)`` on log-log axes.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.vibration import plot_radiation_efficiency

        return plot_radiation_efficiency(self, ax=ax, language=check_language(language), **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the baffled plate to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_radiation_result_geometry

        check_language(language)
        return plot_radiation_result_geometry(self, ax=ax, language=language, **kwargs)


def _sigma_below(
    freq: NDArray[np.float64], fc: float, c0: float, u: float, s: float, cc: float
) -> NDArray[np.float64]:
    """Radiation efficiency below the critical frequency (Hopkins Eq. 2.227)."""
    mu = np.sqrt(fc / freq)
    k = 2.0 * np.pi * freq / c0
    mu2m1 = mu**2 - 1.0
    prefactor = u / (2.0 * np.pi * mu * k * s * np.sqrt(mu2m1))
    bracket = np.log((mu + 1.0) / (mu - 1.0)) + 2.0 * mu / mu2m1
    bc_term = cc - mu ** (-8.0) * (cc - 1.0)
    return np.asarray(prefactor * bracket * bc_term, dtype=np.float64)


def radiation_efficiency(
    frequency: ArrayLike,
    length_x: float,
    length_y: float,
    critical_frequency: float,
    *,
    boundary: str = "simply_supported",
    baffle: str = "infinite",
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> RadiationEfficiencyResult:
    r"""Frequency-averaged radiation efficiency of a plate (Hopkins 2.9.4).

    Implements Hopkins "method no. 1" (Eqs 2.227, 2.229, 2.230): the below-,
    above- and at-coincidence expressions of Leppington/Maidanik. The band whose
    centre lies closest (on a log scale) to *critical_frequency* uses the
    at-coincidence expression (Eq. 2.230); all others use the below/above
    expressions.

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param length_x: Plate dimension ``Lx``, in m (> 0).
    :param length_y: Plate dimension ``Ly``, in m (> 0).
    :param critical_frequency: Coincidence frequency ``fc``, in hertz (> 0);
        see :func:`coincidence_frequency`.
    :param boundary: ``"simply_supported"`` (:math:`C_{BC} = 1`) or
        ``"clamped"`` (:math:`C_{BC} = 2`).
    :param baffle: ``"infinite"`` (:math:`C_{OB} = 1`, plate flush in a rigid
        baffle) or ``"perpendicular"`` (:math:`C_{OB} = 2`, baffles
        perpendicular to the edges).
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :return: A :class:`RadiationEfficiencyResult`.
    :raises ValueError: for a non-positive input or unknown boundary/baffle.
    """
    freq = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    if freq.ndim != 1 or freq.size == 0:
        raise ValueError("'frequency' must be a non-empty 1-D array.")
    if np.any(freq <= 0.0):
        raise ValueError("'frequency' must be positive.")
    lx = require_positive(length_x, "length_x")
    ly = require_positive(length_y, "length_y")
    fc = require_positive(critical_frequency, "critical_frequency")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    boundary = require_choice(boundary, "boundary", tuple(_C_BC))
    baffle = require_choice(baffle, "baffle", tuple(_C_OB))

    u = 2.0 * (lx + ly)
    s = lx * ly
    l1, l2 = (lx, ly) if lx <= ly else (ly, lx)
    cc = _C_BC[boundary] * _C_OB[baffle]

    sigma = np.empty_like(freq)
    # The at-coincidence expression (Eq. 2.230) applies to the band that
    # *contains* fc; a band centre within half a one-third octave of fc is
    # taken as that band (Hopkins evaluates the below/above forms at the band
    # centre, so this avoids their mu = 1 singularity only where a band
    # genuinely sits on coincidence). Every other band uses the below/above
    # form, so a range entirely to one side of fc never invokes Eq. 2.230.
    at_fc = (freq >= fc / _COINCIDENCE_HALF_WIDTH) & (
        freq <= fc * _COINCIDENCE_HALF_WIDTH
    )
    below = (freq < fc) & ~at_fc
    above = (freq >= fc) & ~at_fc

    if np.any(below):
        sigma[below] = _sigma_below(freq[below], fc, c0, u, s, cc)
    if np.any(above):
        mu = np.sqrt(fc / freq[above])
        sigma[above] = 1.0 / np.sqrt(1.0 - mu**2)
    if np.any(at_fc):
        k_fc = 2.0 * np.pi * fc / c0
        sigma[at_fc] = (0.5 - 0.15 * l1 / l2) * np.sqrt(k_fc * l1)

    return RadiationEfficiencyResult(
        frequencies=freq,
        radiation_efficiency=np.asarray(sigma, dtype=np.float64),
        critical_frequency=fc,
        length_x=lx,
        length_y=ly,
        boundary=boundary,
        baffle=baffle,
    )
