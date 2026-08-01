#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Slow-sound slit panels loaded with Helmholtz resonators (perfect absorbers).

A rigid panel perforated by a periodic array of thin closed slits, whose upper
wall is loaded by an array of Helmholtz resonators (HRs), behaves as a
deep-subwavelength, locally reacting sound absorber. The resonators slow the
sound inside the slit, pulling the slit resonance down to the deep
subwavelength regime, and the intrinsic visco-thermal losses can be tuned to
exactly balance the leakage of the structure (critical coupling), giving
perfect absorption at a chosen frequency and angle. The model follows the
transfer-matrix treatment of Jimenez, Groby, Pagneux and Romero-Garcia
(*Iridescent Perfect Absorption in Critically-Coupled Acoustic Metamaterials
Using the Transfer Matrix Method*, Appl. Sci. 2017, 7, 618) together with the
resonator model and end corrections detailed in the supplementary material of
Jimenez, Huang, Romero-Garcia, Pagneux and Groby (*Ultra-thin metamaterial for
perfect and quasi-omnidirectional sound absorption*, Appl. Phys. Lett. 2016,
109, 121902).

The building blocks, all in the ``e^{+j w t}`` convention used throughout
phonometry (a passive medium has :math:`\operatorname{Im}(k) < 0`):

* **Visco-thermal effective parameters.** The slit of height ``h`` uses the
  narrow-channel effective density and bulk modulus (Appl. Sci. Eq. (6);
  Appl. Phys. Lett. Eqs. (A1)-(A2)):

  :math:`\rho_s = \rho_0 [1 - \tanh((h/2) G_\rho) / ((h/2) G_\rho)]^{-1}`
  and :math:`\kappa_s = \kappa_0 [1 + (\gamma - 1) \tanh((h/2) G_\kappa) /
  ((h/2) G_\kappa)]^{-1}`

  with :math:`G_\rho = \sqrt{j \omega \rho_0 / \eta}` and
  :math:`G_\kappa = \sqrt{j \omega \mathrm{Pr} \rho_0 / \eta}`. The square
  necks and cavities use the rectangular-duct series of Stinson (1991),
  reproduced as Appl. Sci. Eqs. (7)-(8) with the transverse wavenumbers
  :math:`\alpha_k = (2k+1) \pi / a` and :math:`\beta_m = (2m+1) \pi / b`.
  The duct series is printed in the opposite time convention of the source;
  it is returned conjugated here so the neck and cavity share the
  :math:`e^{+j\omega t}` passivity of the slit. Both models are pinned in the
  tests to their exact limits: the effective density tends to ``rho0`` and the
  bulk modulus to ``kappa0`` as the boundary layers vanish, and
  :math:`j\omega\rho` tends to the Poiseuille flow resistivity of the channel
  as :math:`\omega \to 0` (:math:`12\eta/h^2` for the slit,
  :math:`28.454\,\eta/w^2` for a square duct).

* **Helmholtz-resonator impedance.** Each resonator is a neck (length ``l_n``,
  side ``w_n``) over a closed cavity (length ``l_c``, side ``w_c``); its
  impedance follows Appl. Phys. Lett. Eq. (A23) with the neck-to-cavity
  radiation end correction of Eqs. (A24)-(A26).

* **Transfer matrix.** The panel is the chain
  ``M_dl (M_s M_HR M_s)...`` of half-lattice slit steps (Appl. Sci. Eq. (2)),
  resonators as point shunt scatterers (Eq. (3)) and the slit-radiation end
  correction (Eq. (3)/(A27)). The slit-radiation series impedance is printed
  in the sources as ``-i w dl_slit rho0 / (phi_t S0)``; like the duct series
  it is used conjugated here (``+j w``), so it acts as the added radiation
  mass it models and lowers the slit-panel resonance. The rigidly-backed
  reflection factor is
  :math:`R = (T_{11} \cos(\theta) - Z_0 T_{21}) /
  (T_{11} \cos(\theta) + Z_0 T_{21})` with
  :math:`Z_0 = \rho_0 c_0 / S_0` (Eq. (4)), and
  :math:`\alpha = 1 - \lvert R \rvert^2`. Perfect absorption (critical
  coupling) is reached when the reflection zero sits on the real-frequency
  axis, i.e. :math:`\operatorname{Re}(Z) \cos(\theta) = Z_0` and
  :math:`\operatorname{Im}(Z) = 0` with :math:`Z = T_{11} / T_{21}` the
  acoustic surface impedance (Eq. (9)).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import root

from .._internal.types import Real
from .._internal.validation import require_positive, require_positive_array
from .._internal.warnings import PhonometryWarning
from .porous_absorber import (
    _AIR_DENSITY,
    _AIR_VISCOSITY,
    _ATMOSPHERIC_PRESSURE,
    _HEAT_CAPACITY_RATIO,
    _PRANDTL_NUMBER,
    _SPEED_OF_SOUND,
    Complex,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Default number of transverse modes kept per axis in the duct series.
_DUCT_TERMS = 40

__all__ = [
    "CriticalCouplingResult",
    "HelmholtzResonator",
    "SlitResonatorAbsorberResult",
    "SlowSoundAbsorberWarning",
    "critical_coupling_design",
    "helmholtz_resonator_impedance",
    "rectangular_duct_properties",
    "slit_effective_properties",
    "slit_helmholtz_absorber",
]


class SlowSoundAbsorberWarning(PhonometryWarning):
    """Advisory for slow-sound absorber use outside the modelled regime."""


@dataclass(frozen=True)
class HelmholtzResonator:
    """A square-cross-section Helmholtz resonator loading a slit.

    ``neck_length`` ``l_n`` and ``neck_side`` ``w_n`` describe the neck,
    ``cavity_length`` ``l_c`` and ``cavity_side`` ``w_c`` the closed cavity;
    all lengths are in metres.
    """

    neck_length: float
    neck_side: float
    cavity_length: float
    cavity_side: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the resonator cross-section to scale (dimensioned).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_helmholtz_resonator_geometry

        check_language(language)
        return plot_helmholtz_resonator_geometry(
            self, ax=ax, language=language, **kwargs
        )


# ---------------------------------------------------------------------------
# Visco-thermal effective parameters
# ---------------------------------------------------------------------------
def slit_effective_properties(
    frequency: ArrayLike,
    *,
    slit_height: float,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
    prandtl_number: float = _PRANDTL_NUMBER,
    heat_capacity_ratio: float = _HEAT_CAPACITY_RATIO,
    atmospheric_pressure: float = _ATMOSPHERIC_PRESSURE,
) -> tuple[Complex, Complex]:
    r"""Effective density and bulk modulus of a narrow slit of height ``h``.

    .. math::

       \rho_s = \rho_0 \left[1 - \frac{\tanh(x_\rho)}{x_\rho}\right]^{-1}

       \kappa_s = \kappa_0 \left[1
       + (\gamma - 1) \frac{\tanh(x_\kappa)}{x_\kappa}\right]^{-1}

    with :math:`x_\rho = (h/2) \sqrt{j \omega \rho_0 / \eta}` and
    :math:`x_\kappa = (h/2) \sqrt{j \omega \mathrm{Pr} \rho_0 / \eta}`
    (Appl. Sci. 2017 Eq. (6); Appl. Phys. Lett. 2016 Eqs. (A1)-(A2)).
    :math:`\kappa_0 = \gamma P_0`.

    :param frequency: Frequency vector ``f``, in hertz.
    :param slit_height: Slit height ``h``, in metres.
    :param air_density: Air density ``rho0``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of air, in Pa s.
    :param prandtl_number: Prandtl number ``Pr`` of air.
    :param heat_capacity_ratio: Ratio of specific heats ``gamma``.
    :param atmospheric_pressure: Static pressure ``P0``, in Pa.
    :return: ``(rho_s, kappa_s)`` complex arrays shaped like ``frequency``.
    """
    f = require_positive_array(frequency, "frequency")
    h = require_positive(slit_height, "slit_height")
    rho0 = require_positive(air_density, "air_density")
    eta = require_positive(viscosity, "viscosity")
    pr = require_positive(prandtl_number, "prandtl_number")
    gamma = require_positive(heat_capacity_ratio, "heat_capacity_ratio")
    p0 = require_positive(atmospheric_pressure, "atmospheric_pressure")
    omega = 2.0 * np.pi * f
    x_rho = (h / 2.0) * np.sqrt(1j * omega * rho0 / eta)
    x_kap = (h / 2.0) * np.sqrt(1j * omega * pr * rho0 / eta)
    rho_s = rho0 / (1.0 - np.tanh(x_rho) / x_rho)
    kap_s = (gamma * p0) / (1.0 + (gamma - 1.0) * np.tanh(x_kap) / x_kap)
    return (
        np.asarray(rho_s, dtype=np.complex128),
        np.asarray(kap_s, dtype=np.complex128),
    )


def rectangular_duct_properties(
    frequency: ArrayLike,
    *,
    side: float,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
    prandtl_number: float = _PRANDTL_NUMBER,
    heat_capacity_ratio: float = _HEAT_CAPACITY_RATIO,
    atmospheric_pressure: float = _ATMOSPHERIC_PRESSURE,
    sum_terms: int = _DUCT_TERMS,
) -> tuple[Complex, Complex]:
    r"""Effective density and bulk modulus of a square duct of the given side.

    The Stinson (1991) rectangular-duct series (Appl. Sci. 2017 Eqs. (7)-(8)),

    .. math::

       \rho = \frac{-\rho_0 a^2 b^2}{64 G_\rho^2 S_\rho}

       \kappa = \frac{\kappa_0}
       {\gamma + 64 (\gamma - 1) G_\kappa^2 / (a^2 b^2) S_\kappa}

    with
    :math:`S = \sum_k \sum_m [\alpha_k^2 \beta_m^2
    (\alpha_k^2 + \beta_m^2 - G^2)]^{-1}`,
    :math:`\alpha_k = (2k+1) \pi / a`, :math:`\beta_m = (2m+1) \pi / b`,
    :math:`G_\rho^2 = j \omega \rho_0 / \eta` and
    :math:`G_\kappa^2 = j \omega \mathrm{Pr} \rho_0 / \eta`. Here the duct
    is square (:math:`a = b`, both equal to ``side``). The series is
    transcribed in the source's time convention and returned conjugated so
    the result is passive in the :math:`e^{+j\omega t}` convention
    (:math:`\operatorname{Im}(k) < 0`). The normalising constant 64 is fixed
    by the exact limits :math:`\rho \to \rho_0`, :math:`\kappa \to \kappa_0`
    as the boundary layers vanish and by the Poiseuille resistivity
    :math:`28.454\,\eta/\text{side}^2` as :math:`\omega \to 0`.

    :param frequency: Frequency vector ``f``, in hertz.
    :param side: Square-duct side length, in metres.
    :param air_density: Air density ``rho0``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of air, in Pa s.
    :param prandtl_number: Prandtl number ``Pr`` of air.
    :param heat_capacity_ratio: Ratio of specific heats ``gamma``.
    :param atmospheric_pressure: Static pressure ``P0``, in Pa.
    :param sum_terms: Transverse modes kept per axis (default 40).
    :return: ``(rho, kappa)`` complex arrays shaped like ``frequency``.
    """
    f = require_positive_array(frequency, "frequency")
    a = require_positive(side, "side")
    rho0 = require_positive(air_density, "air_density")
    eta = require_positive(viscosity, "viscosity")
    pr = require_positive(prandtl_number, "prandtl_number")
    gamma = require_positive(heat_capacity_ratio, "heat_capacity_ratio")
    p0 = require_positive(atmospheric_pressure, "atmospheric_pressure")
    n = int(sum_terms)
    if n < 1:
        raise ValueError("'sum_terms' must be at least 1.")

    kappa0 = gamma * p0
    omega = 2.0 * np.pi * np.atleast_1d(f).astype(np.float64)
    idx = np.arange(n)
    trans = ((2.0 * idx + 1.0) * np.pi / a) ** 2
    ak2 = trans[:, None]
    bm2 = trans[None, :]
    base = ak2 * bm2
    sum_ab = ak2 + bm2
    g_rho2 = (1j * omega * rho0 / eta)[:, None, None]
    g_kap2 = (1j * omega * pr * rho0 / eta)[:, None, None]
    s_rho = np.sum(1.0 / (base[None] * (sum_ab[None] - g_rho2)), axis=(1, 2))
    s_kap = np.sum(1.0 / (base[None] * (sum_ab[None] - g_kap2)), axis=(1, 2))
    rho = -rho0 * a**4 / (64.0 * g_rho2[:, 0, 0] * s_rho)
    kap = kappa0 / (gamma + (64.0 * (gamma - 1.0) * g_kap2[:, 0, 0] / a**4) * s_kap)
    rho = np.conj(rho).reshape(np.shape(f))
    kap = np.conj(kap).reshape(np.shape(f))
    return np.asarray(rho, dtype=np.complex128), np.asarray(kap, dtype=np.complex128)


def _neck_slit_correction(neck_side: float, slit_height: float, lattice_step: float) -> float:
    """Neck-to-slit radiation end correction ``Delta l_2`` (APL Eqs. (A25)-(A26)).

    Circular-duct fit evaluated with area-equivalent radii; the slit radius is
    taken from its ``h * a`` cross-section.
    """
    rn = neck_side / np.sqrt(np.pi)
    rt = np.sqrt(slit_height * lattice_step / np.pi)
    x = rn / rt
    return float(
        0.82
        * (1.0 - 0.235 * x - 1.32 * x**2 + 1.54 * x**3 - 0.86 * x**4)
        * rn
    )


def _neck_cavity_correction(neck_side: float, cavity_side: float) -> float:
    """Neck-to-cavity radiation end correction ``Delta l_1`` (APL Eq. (A24))."""
    rn = neck_side / np.sqrt(np.pi)
    rc = cavity_side / np.sqrt(np.pi)
    x = rn / rc
    return float(0.82 * (1.0 - 1.35 * x + 0.31 * x**3) * rn)


def _neck_slit_correction_2d(neck_width: float, slit_height: float) -> float:
    r"""2-D neck-to-slit end correction ``Delta l_2`` (Sci. Rep. Eq. (12)).

    The 2-D ducts use their half-width as the radiation radius, so the
    Dubos fit is evaluated with the width ratio directly and
    :math:`0.82 (w_n / 2) = 0.41 w_n` as printed in the source, exactly as
    the paper applies it. The fit was derived for a neck narrower than
    the main waveguide; optimised metadiffuser geometries can present
    :math:`w_n > h`, where the polynomial turns negative and the resonator
    effectively decouples from the slit (the caller warns about it).
    """
    x = neck_width / slit_height
    return float(
        0.41
        * (1.0 - 0.235 * x - 1.32 * x**2 + 1.54 * x**3 - 0.86 * x**4)
        * neck_width
    )


def _neck_cavity_correction_2d(neck_width: float, cavity_width: float) -> float:
    """2-D neck-to-cavity end correction ``Delta l_1`` (Sci. Rep. Eq. (11))."""
    x = neck_width / cavity_width
    return float(0.41 * (1.0 - 1.35 * x + 0.31 * x**3) * neck_width)


def _slit_resonator_ducts(
    f: Real, wn: float, wc: float, slit_height: float | None,
    lattice_step: float | None, end_correction: bool, air: dict[str, Any],
) -> tuple[Complex, Complex, Complex, Complex, Complex, Complex, float]:
    """Duct parameters of the 2-D resonator (Sci. Rep. Eqs. (8)-(12))."""
    if slit_height is None or lattice_step is None:
        raise ValueError(
            "The 'slit' resonator geometry requires 'slit_height' and "
            "'lattice_step'."
        )
    h = require_positive(slit_height, "slit_height")
    a = require_positive(lattice_step, "lattice_step")
    rho_n, kap_n = slit_effective_properties(f, slit_height=wn, **air)
    rho_c, kap_c = slit_effective_properties(f, slit_height=wc, **air)
    z_n = np.asarray(np.sqrt(kap_n * rho_n) / (wn * a),
                     dtype=np.complex128)
    z_c = np.asarray(np.sqrt(kap_c * rho_c) / (wc * a),
                     dtype=np.complex128)
    dl = 0.0
    if end_correction:
        if wn > h:
            warnings.warn(
                "The resonator neck is wider than the slit "
                f"({wn:g} m > {h:g} m); the Dubos neck-to-slit end "
                "correction is outside its fitted domain and turns "
                "negative, which effectively decouples the resonator.",
                SlowSoundAbsorberWarning,
                stacklevel=3,
            )
        dl = _neck_cavity_correction_2d(wn, wc)
        dl += _neck_slit_correction_2d(wn, h)
    return rho_n, kap_n, rho_c, kap_c, z_n, z_c, dl


def _square_resonator_ducts(
    f: Real, wn: float, wc: float, slit_height: float | None,
    lattice_step: float | None, end_correction: bool,
    props: dict[str, Any],
) -> tuple[Complex, Complex, Complex, Complex, Complex, Complex, float]:
    """Duct parameters of the square resonator (APL Eqs. (A23)-(A26))."""
    rho_n, kap_n = rectangular_duct_properties(f, side=wn, **props)
    rho_c, kap_c = rectangular_duct_properties(f, side=wc, **props)
    z_n = np.asarray(np.sqrt(kap_n * rho_n) / wn**2, dtype=np.complex128)
    z_c = np.asarray(np.sqrt(kap_c * rho_c) / wc**2, dtype=np.complex128)
    dl = 0.0
    if end_correction:
        dl = _neck_cavity_correction(wn, wc)
        if slit_height is not None and lattice_step is not None:
            dl += _neck_slit_correction(
                wn, require_positive(slit_height, "slit_height"),
                require_positive(lattice_step, "lattice_step"),
            )
    return rho_n, kap_n, rho_c, kap_c, z_n, z_c, dl


def helmholtz_resonator_impedance(
    frequency: ArrayLike,
    resonator: HelmholtzResonator,
    *,
    slit_height: float | None = None,
    lattice_step: float | None = None,
    end_correction: bool = True,
    geometry: str = "square",
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
    prandtl_number: float = _PRANDTL_NUMBER,
    heat_capacity_ratio: float = _HEAT_CAPACITY_RATIO,
    atmospheric_pressure: float = _ATMOSPHERIC_PRESSURE,
    sum_terms: int = _DUCT_TERMS,
) -> Complex:
    r"""Acoustic impedance of a Helmholtz resonator with visco-thermal losses.

    With the default ``geometry="square"`` the neck and cavity are square
    ducts using the effective parameters of
    :func:`rectangular_duct_properties`; the impedance is Appl. Phys. Lett. 2016
    Eq. (A23) with the neck-to-cavity radiation correction of Eq. (A24) and,
    when ``slit_height`` and ``lattice_step`` are supplied, the neck-to-slit
    correction of Eqs. (A25)-(A26) added to the total neck length correction:

    .. math::

       Z_{\mathrm{HR}} = -j \frac{\cos(k_n l_n) \cos(k_c l_c)
       - Z_n k_n \mathrm{dl} \cos(k_n l_n) \sin(k_c l_c) / Z_c
       - Z_n \sin(k_n l_n) \sin(k_c l_c) / Z_c}
       {\sin(k_n l_n) \cos(k_c l_c) / Z_n
       - k_n \mathrm{dl} \sin(k_n l_n) \sin(k_c l_c) / Z_c
       + \cos(k_n l_n) \sin(k_c l_c) / Z_c}

    with :math:`Z_n = \sqrt{\kappa_n \rho_n} / w_n^2`,
    :math:`k_n = \omega \sqrt{\rho_n / \kappa_n}` (and likewise for the
    cavity), reducing to Eq. (A22) when ``dl = 0``.

    With ``geometry="slit"`` the resonator is two-dimensional (the neck and
    cavity are slit-like ducts spanning the lattice step): the effective
    parameters come from :func:`slit_effective_properties` with the neck and
    cavity widths, the duct sections are ``w_n a`` and ``w_c a``, and the end
    corrections are the 2-D fits of Sci. Rep. 7:5389 Eqs. (11)-(12); both
    ``slit_height`` and ``lattice_step`` are then required.

    :param frequency: Frequency vector ``f``, in hertz.
    :param resonator: The :class:`HelmholtzResonator` geometry.
    :param slit_height: Slit height ``h`` for the neck-to-slit correction; if
        ``None`` that correction is omitted (``"square"`` only).
    :param lattice_step: Lattice step ``a`` for the neck-to-slit correction.
    :param end_correction: Include the radiation end corrections (default True).
    :param geometry: ``"square"`` (default) for square-duct necks and
        cavities, ``"slit"`` for the two-dimensional resonator model.
    :param air_density: Air density ``rho0``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of air, in Pa s.
    :param prandtl_number: Prandtl number ``Pr`` of air.
    :param heat_capacity_ratio: Ratio of specific heats ``gamma``.
    :param atmospheric_pressure: Static pressure ``P0``, in Pa.
    :param sum_terms: Transverse modes kept per axis in the duct series.
    :return: Complex acoustic impedance ``Z_HR``, in Pa s/m3, shaped like
        ``frequency``.
    """
    f = require_positive_array(frequency, "frequency")
    ln = require_positive(resonator.neck_length, "neck_length")
    wn = require_positive(resonator.neck_side, "neck_side")
    lc = require_positive(resonator.cavity_length, "cavity_length")
    wc = require_positive(resonator.cavity_side, "cavity_side")
    if geometry not in ("square", "slit"):
        raise ValueError("'geometry' must be 'square' or 'slit'.")
    air: dict[str, Any] = {
        "air_density": air_density,
        "viscosity": viscosity,
        "prandtl_number": prandtl_number,
        "heat_capacity_ratio": heat_capacity_ratio,
        "atmospheric_pressure": atmospheric_pressure,
    }
    omega = 2.0 * np.pi * f
    if geometry == "slit":
        rho_n, kap_n, rho_c, kap_c, z_n, z_c, dl = _slit_resonator_ducts(
            f, wn, wc, slit_height, lattice_step, end_correction, air,
        )
    else:
        rho_n, kap_n, rho_c, kap_c, z_n, z_c, dl = _square_resonator_ducts(
            f, wn, wc, slit_height, lattice_step, end_correction,
            dict(air, sum_terms=sum_terms),
        )
    k_n = omega * np.sqrt(rho_n / kap_n)
    k_c = omega * np.sqrt(rho_c / kap_c)
    cos_n, sin_n = np.cos(k_n * ln), np.sin(k_n * ln)
    cos_c, sin_c = np.cos(k_c * lc), np.sin(k_c * lc)
    num = (
        cos_n * cos_c
        - z_n * k_n * dl * cos_n * sin_c / z_c
        - z_n * sin_n * sin_c / z_c
    )
    den = (
        sin_n * cos_c / z_n
        - k_n * dl * sin_n * sin_c / z_c
        + cos_n * sin_c / z_c
    )
    return np.asarray(-1j * num / den, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Panel transfer matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlitResonatorAbsorberResult:
    r"""Prediction of a slit panel loaded with Helmholtz resonators.

    All spectra share the shape of ``frequency``. ``surface_impedance`` is
    the acoustic surface impedance :math:`Z = T_{11} / T_{21}` in Pa s/m3 of
    the rigidly backed panel, ``normalized_impedance`` its ratio to
    :math:`Z_0 = \rho_0 c_0 / S_0`, ``reflection`` the plane-wave reflection
    factor ``R(theta)``, ``absorption`` the coefficient
    :math:`\alpha = 1 - \lvert R \rvert^2`, ``effective_wavenumber`` and
    ``effective_impedance`` the retrieved ``k_eff`` and ``Z_eff``
    (Appl. Sci. 2017 Eq. (5)), and ``transfer_matrix`` the total 2x2 chain
    matrix with shape ``(2, 2, len(frequency))``.

    The trailing fields retain the panel geometry the prediction was run
    with (``resonators``, ``slit_height``, ``lattice_step``, ``period``) so
    :meth:`plot_geometry` can draw the cross-section; they are appended after
    the original fields and default to ``None`` for hand-built results.
    """

    frequency: Real
    angle: float
    surface_impedance: Complex
    normalized_impedance: Complex
    reflection: Complex
    absorption: Real
    effective_wavenumber: Complex
    effective_impedance: Complex
    transfer_matrix: Complex
    resonators: tuple[HelmholtzResonator, ...] | None = None
    slit_height: float | None = None
    lattice_step: float | None = None
    period: float | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the absorption spectrum ``alpha(f)`` with ``|R|`` overlaid.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.materials import plot_slit_resonator_absorber

        check_language(language)
        return plot_slit_resonator_absorber(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw one period of the panel cross-section to scale (dimensioned).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_slit_absorber_result_geometry

        check_language(language)
        return plot_slit_absorber_result_geometry(
            self, ax=ax, language=language, **kwargs
        )


def _slit_radiation_length(slit_height: float, period: float, terms: int = 400) -> float:
    """Slit-to-free-air radiation end correction ``Delta l_slit`` (APL Eq. (A27))."""
    phit = slit_height / period
    n = np.arange(1, terms + 1)
    return float(slit_height * phit * np.sum(np.sin(n * np.pi * phit) ** 2 / (n * np.pi * phit) ** 3))


def _panel_transfer_matrix(
    omega: Real,
    resonators: tuple[HelmholtzResonator, ...],
    *,
    slit_height: float,
    lattice_step: float,
    period: float,
    slit_radiation: bool,
    end_correction: bool,
    resonator_geometry: str,
    rho0: float,
    props: dict[str, Any],
) -> Complex:
    """Total chain matrix ``M_dl (M_s M_HR M_s)...`` shaped ``(2, 2, nf)``."""
    f = omega / (2.0 * np.pi)
    area_slit = slit_height * lattice_step
    area_cell = period * lattice_step
    phit = slit_height / period
    rho_s, kap_s = slit_effective_properties(
        f, slit_height=slit_height,
        air_density=props["air_density"], viscosity=props["viscosity"],
        prandtl_number=props["prandtl_number"],
        heat_capacity_ratio=props["heat_capacity_ratio"],
        atmospheric_pressure=props["atmospheric_pressure"],
    )
    k_s = omega * np.sqrt(rho_s / kap_s)
    z_s = np.sqrt(kap_s * rho_s) / area_slit
    nf = omega.size
    ones = np.ones(nf, dtype=np.complex128)
    zeros = np.zeros(nf, dtype=np.complex128)
    # Half-lattice slit step (Appl. Sci. Eq. (2)).
    arg = k_s * lattice_step / 2.0
    cos_h, sin_h = np.cos(arg), np.sin(arg)
    ms = np.array([[cos_h, 1j * z_s * sin_h], [1j * sin_h / z_s, cos_h]])
    # Front slit-radiation correction (series impedance). The papers print
    # this term as -i w dl rho0 / (phi S0); like the duct series it is
    # conjugated here to the e^{+j w t} convention, where a radiation mass is
    # +j w rho0 dl / (phi S0) and must lower the slit-panel resonance.
    dl_slit = _slit_radiation_length(slit_height, period) if slit_radiation else 0.0
    z_dl = 1j * omega * dl_slit * rho0 / (phit * area_cell)
    total = np.array([[ones, z_dl], [zeros, ones]])
    for res in resonators:
        z_hr = helmholtz_resonator_impedance(
            f, res, slit_height=slit_height, lattice_step=lattice_step,
            end_correction=end_correction, geometry=resonator_geometry,
            **props,
        )
        m_hr = np.array([[ones, zeros], [ones / z_hr, ones]])
        cell = _matmul(_matmul(ms, m_hr), ms)
        total = _matmul(total, cell)
    return total


def _matmul(a: Complex, b: Complex) -> Complex:
    """Frequency-wise product of two ``(2, 2, nf)`` chain matrices."""
    return np.asarray(
        [
            [a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0], a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1]],
            [a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0], a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]],
        ],
        dtype=np.complex128,
    )


def slit_helmholtz_absorber(
    frequency: ArrayLike,
    resonators: HelmholtzResonator | list[HelmholtzResonator] | tuple[HelmholtzResonator, ...],
    *,
    slit_height: float,
    lattice_step: float,
    period: float,
    angle: float = 0.0,
    end_correction: bool = True,
    slit_radiation: bool = True,
    resonator_geometry: str = "square",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
    prandtl_number: float = _PRANDTL_NUMBER,
    heat_capacity_ratio: float = _HEAT_CAPACITY_RATIO,
    atmospheric_pressure: float = _ATMOSPHERIC_PRESSURE,
) -> SlitResonatorAbsorberResult:
    r"""Transfer-matrix prediction of a slit panel loaded with resonators.

    The panel is a periodic array (period ``d`` along the panel face) of thin
    closed slits of height ``h``, each loaded from its upper wall by the given
    ``resonators`` spaced by the lattice step ``a`` (Appl. Sci. 2017,
    Section 2). The total chain matrix is
    :math:`T = M_{\mathrm{dl}} (M_s M_{\mathrm{HR}} M_s) \cdots` over the
    ``N`` resonators, where each resonator sits between two half-lattice
    slit steps; the rigidly-backed reflection factor is
    :math:`R = (T_{11} \cos(\theta) - Z_0 T_{21}) /
    (T_{11} \cos(\theta) + Z_0 T_{21})` with
    :math:`Z_0 = \rho_0 c_0 / S_0`, :math:`S_0 = d a`, and
    :math:`\alpha = 1 - \lvert R \rvert^2` (Eq. (4)). The structure is
    locally reacting, so the internal chain does not depend on ``theta``;
    only the front air impedance carries ``cos(theta)``.

    :param frequency: Frequency vector ``f``, in hertz.
    :param resonators: One :class:`HelmholtzResonator` or a sequence of them,
        ordered from the panel face towards the rigid backing.
    :param slit_height: Slit height ``h``, in metres.
    :param lattice_step: Resonator lattice step ``a`` along the slit, in metres;
        the slit depth is :math:`L = N a`.
    :param period: Slit array period ``d`` along the face, in metres
        (:math:`d \ge h`).
    :param angle: Polar angle of incidence ``theta``, in radians
        (:math:`0 \le \theta < \pi/2 - 10^{-6}`).
    :param end_correction: Include the resonator radiation end corrections.
    :param slit_radiation: Include the slit-to-free-air radiation correction.
    :param speed_of_sound: Speed of sound ``c0`` in air, in m/s.
    :param air_density: Air density ``rho0``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of air, in Pa s.
    :param prandtl_number: Prandtl number ``Pr`` of air.
    :param heat_capacity_ratio: Ratio of specific heats ``gamma``.
    :param atmospheric_pressure: Static pressure ``P0``, in Pa.
    :return: A :class:`SlitResonatorAbsorberResult`.
    """
    f = require_positive_array(frequency, "frequency")
    if isinstance(resonators, HelmholtzResonator):
        resonators = (resonators,)
    res = tuple(resonators)
    if not res:
        raise ValueError("'resonators' must contain at least one resonator.")
    h = require_positive(slit_height, "slit_height")
    a = require_positive(lattice_step, "lattice_step")
    d = require_positive(period, "period")
    if h > d:
        raise ValueError("'slit_height' must not exceed 'period'.")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    theta = float(angle)
    if not 0.0 <= theta < np.pi / 2.0 - 1e-6:
        raise ValueError("'angle' must satisfy 0 <= angle < pi/2 - 1e-6.")

    props: dict[str, Any] = {
        "air_density": rho0,
        "viscosity": require_positive(viscosity, "viscosity"),
        "prandtl_number": require_positive(prandtl_number, "prandtl_number"),
        "heat_capacity_ratio": require_positive(heat_capacity_ratio, "heat_capacity_ratio"),
        "atmospheric_pressure": require_positive(atmospheric_pressure, "atmospheric_pressure"),
    }
    omega = 2.0 * np.pi * f
    tm = _panel_transfer_matrix(
        omega, res, slit_height=h, lattice_step=a, period=d,
        slit_radiation=slit_radiation, end_correction=end_correction,
        resonator_geometry=resonator_geometry, rho0=rho0, props=props,
    )
    t11, t12, t21, t22 = tm[0, 0], tm[0, 1], tm[1, 0], tm[1, 1]
    area_cell = d * a
    z0 = rho0 * c0 / area_cell
    cos_t = float(np.cos(theta))
    z_in = t11 / t21
    r = (t11 * cos_t - z0 * t21) / (t11 * cos_t + z0 * t21)
    alpha = 1.0 - np.abs(r) ** 2
    length = len(res) * a
    k_eff = np.arccos((t11 + t22) / 2.0) / length
    z_eff = np.sqrt(t12 / t21)
    return SlitResonatorAbsorberResult(
        frequency=f,
        angle=theta,
        surface_impedance=np.asarray(z_in, dtype=np.complex128),
        normalized_impedance=np.asarray(z_in / z0, dtype=np.complex128),
        reflection=np.asarray(r, dtype=np.complex128),
        absorption=np.asarray(alpha, dtype=np.float64),
        effective_wavenumber=np.asarray(k_eff, dtype=np.complex128),
        effective_impedance=np.asarray(z_eff, dtype=np.complex128),
        transfer_matrix=np.asarray(tm, dtype=np.complex128),
        resonators=res,
        slit_height=slit_height,
        lattice_step=lattice_step,
        period=period,
    )


# ---------------------------------------------------------------------------
# Critical-coupling (perfect-absorption) design helper
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CriticalCouplingResult:
    """Outcome of a critical-coupling (perfect-absorption) design.

    ``resonator`` and ``slit_height`` are the solved geometry that places the
    reflection zero on the real-frequency axis at ``target_frequency`` and
    ``angle``; ``absorption`` is the modelled coefficient there (``~1``) and
    ``normalized_impedance`` the achieved ``Z cos(theta) / Z0`` (``~1``).
    ``converged`` flags whether the root find met its tolerance.
    """

    target_frequency: float
    angle: float
    resonator: HelmholtzResonator
    slit_height: float
    absorption: float
    normalized_impedance: complex
    converged: bool


def _acoustic_surface_impedance(
    f0: float,
    resonator: HelmholtzResonator,
    *,
    slit_height: float,
    lattice_step: float,
    period: float,
    end_correction: bool,
    slit_radiation: bool,
    rho0: float,
    props: dict[str, Any],
) -> complex:
    r"""Acoustic surface impedance :math:`Z = T_{11} / T_{21}` at ``f0``."""
    omega = np.array([2.0 * np.pi * f0], dtype=np.float64)
    tm = _panel_transfer_matrix(
        omega, (resonator,), slit_height=slit_height, lattice_step=lattice_step,
        period=period, slit_radiation=slit_radiation, end_correction=end_correction,
        resonator_geometry="square", rho0=rho0, props=props,
    )
    return complex(tm[0, 0][0] / tm[1, 0][0])


def critical_coupling_design(
    target_frequency: float,
    resonator: HelmholtzResonator,
    *,
    lattice_step: float,
    period: float,
    angle: float = 0.0,
    slit_height_bounds: tuple[float, float] = (0.2e-3, 5.0e-3),
    cavity_length_bounds: tuple[float, float] = (2.0e-3, 200.0e-3),
    end_correction: bool = True,
    slit_radiation: bool = True,
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
    prandtl_number: float = _PRANDTL_NUMBER,
    heat_capacity_ratio: float = _HEAT_CAPACITY_RATIO,
    atmospheric_pressure: float = _ATMOSPHERIC_PRESSURE,
) -> CriticalCouplingResult:
    r"""Solve resonator/slit geometry for perfect absorption at a frequency.

    Critical coupling (perfect absorption) requires the acoustic surface
    impedance :math:`Z = T_{11} / T_{21}` of the rigidly-backed panel to
    satisfy :math:`\operatorname{Re}(Z) \cos(\theta) = Z_0` and
    :math:`\operatorname{Im}(Z) = 0` at ``target_frequency``
    (Appl. Sci. 2017 Eq. (9)), i.e. the reflection zero lies on the
    real-frequency axis. Holding the neck geometry and cavity side of
    ``resonator`` fixed, this tunes the cavity length (which sets the
    resonance frequency) and the slit height (which sets the visco-thermal
    leakage balance) to meet both conditions, so ``alpha ~ 1`` at the design
    point.

    :param target_frequency: Design frequency ``f0``, in hertz.
    :param resonator: Base geometry; its ``cavity_length`` is used as the
        initial guess and its neck and cavity side are held fixed.
    :param lattice_step: Resonator lattice step ``a``, in metres.
    :param period: Slit array period ``d``, in metres.
    :param angle: Design angle of incidence ``theta``, in radians.
    :param slit_height_bounds: Search bounds for the slit height, in metres.
    :param cavity_length_bounds: Search bounds for the cavity length, in metres.
    :param end_correction: Include the resonator radiation end corrections.
    :param slit_radiation: Include the slit-to-free-air radiation correction.
    :param speed_of_sound: Speed of sound ``c0`` in air, in m/s.
    :param air_density: Air density ``rho0``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of air, in Pa s.
    :param prandtl_number: Prandtl number ``Pr`` of air.
    :param heat_capacity_ratio: Ratio of specific heats ``gamma``.
    :param atmospheric_pressure: Static pressure ``P0``, in Pa.
    :return: A :class:`CriticalCouplingResult`. A
        :class:`SlowSoundAbsorberWarning` is emitted (via :func:`warnings.warn`)
        if the solver does not reach perfect absorption within tolerance.
    """
    f0 = require_positive(target_frequency, "target_frequency")
    a = require_positive(lattice_step, "lattice_step")
    d = require_positive(period, "period")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    theta = float(angle)
    if not 0.0 <= theta < np.pi / 2.0 - 1e-6:
        raise ValueError("'angle' must satisfy 0 <= angle < pi/2 - 1e-6.")
    h_lo, h_hi = slit_height_bounds
    lc_lo, lc_hi = cavity_length_bounds
    props: dict[str, Any] = {
        "air_density": rho0,
        "viscosity": require_positive(viscosity, "viscosity"),
        "prandtl_number": require_positive(prandtl_number, "prandtl_number"),
        "heat_capacity_ratio": require_positive(heat_capacity_ratio, "heat_capacity_ratio"),
        "atmospheric_pressure": require_positive(atmospheric_pressure, "atmospheric_pressure"),
    }
    cos_t = float(np.cos(theta))
    z0 = rho0 * c0 / (d * a)

    def residual(x: np.ndarray) -> list[float]:
        lc, h = float(x[0]), float(x[1])
        if not (lc_lo <= lc <= lc_hi and h_lo <= h <= h_hi and h <= d):
            return [1e6, 1e6]
        cand = HelmholtzResonator(resonator.neck_length, resonator.neck_side, lc, resonator.cavity_side)
        z = _acoustic_surface_impedance(
            f0, cand, slit_height=h, lattice_step=a, period=d,
            end_correction=end_correction, slit_radiation=slit_radiation,
            rho0=rho0, props=props,
        )
        diff = z * cos_t - z0
        return [diff.real / z0, diff.imag / z0]

    guess = np.array([require_positive(resonator.cavity_length, "cavity_length"), 1.0e-3])
    sol = root(residual, guess, method="hybr", tol=1e-10)
    lc_opt, h_opt = float(sol.x[0]), float(sol.x[1])
    designed = HelmholtzResonator(
        resonator.neck_length, resonator.neck_side, lc_opt, resonator.cavity_side
    )
    z_final = _acoustic_surface_impedance(
        f0, designed, slit_height=h_opt, lattice_step=a, period=d,
        end_correction=end_correction, slit_radiation=slit_radiation,
        rho0=rho0, props=props,
    )
    r = (z_final * cos_t - z0) / (z_final * cos_t + z0)
    alpha = float(1.0 - abs(r) ** 2)
    converged = bool(
        sol.success
        and lc_lo <= lc_opt <= lc_hi
        and h_lo <= h_opt <= h_hi
        and alpha > 0.999
    )
    if not converged:
        warnings.warn(
            "critical_coupling_design did not reach perfect absorption within "
            f"tolerance (alpha = {alpha:.4f}); try different bounds or base "
            "geometry.",
            SlowSoundAbsorberWarning,
            stacklevel=2,
        )
    return CriticalCouplingResult(
        target_frequency=f0,
        angle=theta,
        resonator=designed,
        slit_height=h_opt,
        absorption=alpha,
        normalized_impedance=complex(z_final * cos_t / z0),
        converged=converged,
    )
