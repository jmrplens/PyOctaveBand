#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Four-microphone transfer-matrix method for the transmission of a specimen.

**ASTM E2611-19**, the two-tube method: the specimen sits between an upstream
and a downstream tube section with two microphones on each side, and the wave
field is decomposed into forward/backward amplitudes on each side
(Eqs. (17)-(20)). The face pressures and particle velocities are formed
(Eq. (21)) and the transfer matrix ``[[T11, T12], [T21, T22]]`` is solved from
a two-load (Eq. (22)) or a symmetric one-load (Eq. (24)) measurement.
Transmission loss (Eq. (26)), hard-backed reflection/absorption
(Eqs. (27)/(28)) and the material wavenumber/characteristic impedance
(Eqs. (29)/(30)) all read out of those four poles, which is what makes the
standard one subject: everything here exists to fill the matrix or to
interpret it.

Time convention :math:`e^{+j\omega t}` with the forward wave carried by
:math:`e^{-jkx}` (Eq. (21)); air properties from Clause 8.2/8.3, Eqs. (4)/(5),
use temperature in **degrees Celsius**. Both differ from the ISO 10534-2
ansatz of :mod:`~phonometry.materials.absorbers.impedance_tube`, whose
wavenumber is :math:`k_0 = k_0' - jk_0''` and whose air properties take
kelvin; the two are **not** interchangeable, so the air-property and
working-range helpers are named per standard and each stays with the method
that prescribes it. What the two transfer methods genuinely share - the tube
cross-section and the plane-wave working-range arithmetic - is imported from
that module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .impedance_tube import (
    _DIAMETER_POSITIVE,
    ImpedanceTubeWarning,
    _canonical_shape,
    _frequency_range,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._internal.types import Real

Complex = NDArray[np.complex128]

#: Leading constant of ASTM E2611-19 Eq. (4), speed of sound (20,047).
_ASTM_C_CONST = 20.047
#: Celsius-to-kelvin offset used by ASTM E2611-19 Eqs. (4)/(5) (273,15).
_ASTM_T0 = 273.15
#: Reference air density of ASTM E2611-19 Eq. (5), in kg/m3 (1,290).
_ASTM_RHO_REF = 1.290
#: Reference atmospheric pressure of ASTM E2611-19 Eq. (5), in kPa (101,325).
_ASTM_P_REF = 101.325

#: Upper-frequency plane-wave factor ``K``, circular tube (ASTM E2611-19,
#: 6.2.4.1, Eq. (2)).
_ASTM_KU_CIRCULAR = 0.586
#: Upper-frequency plane-wave factor ``K``, rectangular tube with ``d`` the
#: largest section dimension (ASTM E2611-19, 6.2.5).
_ASTM_KU_RECTANGULAR = 0.500
#: Microphone-spacing factor for the upper limit: ``s`` no larger than 80 % of
#: ``c / (2 f_u)``, i.e. ``f_u s < 0,40 c`` (ASTM E2611-19, 6.5.4).
_ASTM_KU_SPACING = 0.40
#: Lower-limit factor: spacing shall be greater than 1 % of the wavelength
#: (ASTM E2611-19, 6.2.3), i.e. ``f_l = c / (100 s)``.
_ASTM_LOWER_WAVELENGTH_FRACTION = 100.0

#: Shared message of the ``characteristic_impedance`` validation, repeated by
#: every entry point that takes the air characteristic impedance ``rho c``.
_IMPEDANCE_POSITIVE = "'characteristic_impedance' must be positive."

__all__ = [
    "TransferMatrix",
    "air_density_astm",
    "air_layer_transfer_matrix",
    "face_quantities",
    "plane_wave_frequency_range_astm",
    "speed_of_sound_astm",
    "transfer_matrix_one_load",
    "transfer_matrix_two_load",
    "wave_decomposition",
]


# ---------------------------------------------------------------------------
# Air properties (ASTM E2611-19 works in degrees Celsius; the ISO 10534-2
# pair, in kelvin, lives with the two-microphone method).
# ---------------------------------------------------------------------------
def speed_of_sound_astm(temperature: ArrayLike) -> Real:
    r"""Speed of sound in air (ASTM E2611-19, Eq. (4)).

    :math:`c = 20.047 \sqrt{273.15 + T}`.

    :param temperature: Room temperature ``T``, in **degrees Celsius**.
    :return: Speed of sound ``c``, in metres per second.
    """
    t = np.asarray(temperature, dtype=np.float64)
    if np.any(t <= -_ASTM_T0):
        raise ValueError("'temperature' must exceed -273,15 degC.")
    return np.asarray(_ASTM_C_CONST * np.sqrt(_ASTM_T0 + t), dtype=np.float64)


def air_density_astm(
    temperature: ArrayLike, atmospheric_pressure: ArrayLike = _ASTM_P_REF
) -> Real:
    r"""Air density (ASTM E2611-19, Eq. (5)).

    :math:`\rho = 1.290 \, \frac{P}{101.325} \, \frac{273.15}{273.15 + T}`.

    :param temperature: Room temperature ``T``, in **degrees Celsius**.
    :param atmospheric_pressure: Atmospheric pressure ``P``, in kilopascals
        (default 101,325 kPa).
    :return: Air density ``rho``, in kilograms per cubic metre.
    """
    t = np.asarray(temperature, dtype=np.float64)
    p = np.asarray(atmospheric_pressure, dtype=np.float64)
    if np.any(t <= -_ASTM_T0):
        raise ValueError("'temperature' must exceed -273,15 degC.")
    if np.any(p <= 0.0):
        raise ValueError("'atmospheric_pressure' must be positive (kPa).")
    return np.asarray(
        _ASTM_RHO_REF * (p / _ASTM_P_REF) * (_ASTM_T0 / (_ASTM_T0 + t)),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Plane-wave working range (ASTM E2611-19, 6.2.3-6.2.5, 6.5.4).
# ---------------------------------------------------------------------------
def plane_wave_frequency_range_astm(
    spacing: float,
    speed_of_sound: float,
    *,
    diameter: float | None = None,
    shape: str = "circular",
) -> tuple[float, float]:
    r"""Working plane-wave frequency range ``(f_l, f_u)`` (ASTM E2611-19).

    The upper limit is the smaller of the microphone-spacing bound
    :math:`s \le 0.8 c / (2 f_\mathrm{u})`, i.e. :math:`f_\mathrm{u} s < 0.40 c` (6.5.4), and,
    when the tube ``diameter`` is given, the cut-on bound
    :math:`f_\mathrm{u} < K c / d` with :math:`K = 0.586` for a circular tube
    (6.2.4.1, Eq. (2)) or :math:`K = 0.500` for a rectangular tube with ``d``
    the largest section dimension (6.2.5). The lower limit follows 6.2.3: the
    spacing shall be greater than 1 % of the wavelength, i.e.
    :math:`f_\mathrm{l} = c / (100 s)`.

    With two different spacings ``s1``/``s2``, call with the larger one for
    the upper bound and the smaller one for the lower bound (each bound is
    binding for every microphone pair).

    :param spacing: Microphone spacing ``s``, in metres.
    :param speed_of_sound: Speed of sound ``c``, in metres per second.
    :param diameter: Tube diameter (circular) or largest section dimension
        (rectangular/square) ``d``, in metres; ``None`` applies only the
        spacing bound.
    :param shape: ``"circular"``, ``"rectangular"`` or ``"square"``.
    :return: Tuple ``(f_l, f_u)`` of the lower and upper frequency limits, in Hz.
    """
    return _frequency_range(
        spacing,
        speed_of_sound,
        diameter=diameter,
        shape=shape,
        ku_circular=_ASTM_KU_CIRCULAR,
        ku_rectangular=_ASTM_KU_RECTANGULAR,
        ku_spacing=_ASTM_KU_SPACING,
        lower_fraction=_ASTM_LOWER_WAVELENGTH_FRACTION,
    )


def _warn_astm_plane_wave(
    wavenumber: ArrayLike,
    *,
    s1: float,
    s2: float,
    diameter: float,
    shape: str,
    stacklevel: int,
) -> None:
    r"""Advise when wavenumbers leave the ASTM E2611-19 plane-wave range.

    The check runs on the real part of ``k`` so no speed of sound is needed:
    :math:`f s < 0.40 c` maps to :math:`k s < 0.80 \pi` (6.5.4),
    :math:`f d < K c` to :math:`k d < 2 \pi K` (6.2.4.1/6.2.5) and the
    greater-than-1 %-of-wavelength spacing bound to
    :math:`k s > 0.02 \pi` (6.2.3). The upper spacing bound binds
    every microphone pair (largest spacing), the lower one the smallest.
    """
    if s1 <= 0.0 or s2 <= 0.0:
        raise ValueError("'s1' and 's2' must be positive.")
    if diameter <= 0.0:
        raise ValueError(_DIAMETER_POSITIVE)
    k = np.real(np.asarray(wavenumber, dtype=np.complex128))
    ku = _ASTM_KU_CIRCULAR if shape == "circular" else _ASTM_KU_RECTANGULAR
    two_pi = 2.0 * np.pi
    k_upper = min(
        two_pi * _ASTM_KU_SPACING / max(s1, s2),
        two_pi * ku / diameter,
    )
    k_lower = two_pi / (_ASTM_LOWER_WAVELENGTH_FRACTION * min(s1, s2))
    if np.any(k < k_lower) or np.any(k > k_upper):
        warnings.warn(
            "Wavenumbers outside the ASTM E2611-19 plane-wave working range "
            "(6.2.3-6.2.5, 6.5.4) for the given microphone spacings and tube "
            "cross-section; results there are advisory. See "
            "plane_wave_frequency_range_astm() for the limits in hertz.",
            ImpedanceTubeWarning,
            stacklevel=stacklevel + 1,
        )


def wave_decomposition(
    h1: ArrayLike,
    h2: ArrayLike,
    h3: ArrayLike,
    h4: ArrayLike,
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    wavenumber: ArrayLike,
    diameter: float | None = None,
    shape: str = "circular",
) -> tuple[Complex, Complex, Complex, Complex]:
    r"""Decompose the wave field into ``(A, B, C, D)`` (ASTM E2611-19, Eqs. (17)-(20)).

    The exponents are implemented exactly as printed:

    .. math::

       A = \frac{j \left( H_1 e^{-jkl_1} - H_2 e^{-jk(l_1+s_1)} \right)}
       {2 \sin(k s_1)}

       B = \frac{j \left( H_2 e^{+jk(l_1+s_1)} - H_1 e^{+jkl_1} \right)}
       {2 \sin(k s_1)}

       C = \frac{j \left( H_3 e^{+jk(l_2+s_2)} - H_4 e^{+jkl_2} \right)}
       {2 \sin(k s_2)}

       D = \frac{j \left( H_4 e^{-jkl_2} - H_3 e^{-jk(l_2+s_2)} \right)}
       {2 \sin(k s_2)}

    ``A``/``B`` are the forward/backward complex amplitudes on the upstream
    (source) side and ``C``/``D`` those on the downstream side, all referenced
    to the front face :math:`x = 0`. With the :math:`e^{+j\omega t}` /
    forward-:math:`e^{-jkx}`
    convention these exponents correspond to the microphone whose transfer
    function is ``H2`` sitting nearest the front face at distance ``l1`` (and
    ``H1`` at :math:`l_1 + s_1`), and to ``H3`` nearest the downstream side at
    ``l2`` (and ``H4`` at :math:`l_2 + s_2`), with ``l1``, ``l2`` measured
    from the front reference plane. The convention was locked down against the analytic
    air-layer transfer matrix (see :func:`air_layer_transfer_matrix`).

    :param h1: Transfer function ``H1,ref`` (upstream, farther microphone).
    :param h2: Transfer function ``H2,ref`` (upstream, nearer microphone).
    :param h3: Transfer function ``H3,ref`` (downstream, nearer microphone).
    :param h4: Transfer function ``H4,ref`` (downstream, farther microphone).
    :param l1: Distance ``l1`` from the front reference plane, in metres.
    :param s1: Upstream microphone spacing ``s1``, in metres.
    :param l2: Distance ``l2`` from the front reference plane, in metres.
    :param s2: Downstream microphone spacing ``s2``, in metres.
    :param wavenumber: Air wavenumber ``k`` (real or complex), scalar or per band.
    :param diameter: Optional tube diameter (circular) or largest section
        dimension (rectangular/square), in metres, that activates the
        plane-wave working-range check (6.2.3-6.2.5, 6.5.4).
    :param shape: Tube cross-section, ``"circular"``, ``"rectangular"`` or
        ``"square"``.
    :return: Tuple ``(A, B, C, D)`` of complex amplitudes.
    """
    if s1 <= 0.0 or s2 <= 0.0:
        raise ValueError("'s1' and 's2' must be positive.")
    canonical = _canonical_shape(shape)
    if diameter is not None:
        _warn_astm_plane_wave(
            wavenumber,
            s1=s1,
            s2=s2,
            diameter=diameter,
            shape=canonical,
            stacklevel=2,
        )
    ha = np.asarray(h1, dtype=np.complex128)
    hb = np.asarray(h2, dtype=np.complex128)
    hc = np.asarray(h3, dtype=np.complex128)
    hd = np.asarray(h4, dtype=np.complex128)
    k = np.asarray(wavenumber, dtype=np.complex128)
    two_sin1 = 2.0 * np.sin(k * s1)
    two_sin2 = 2.0 * np.sin(k * s2)
    a = 1j * (ha * np.exp(-1j * k * l1) - hb * np.exp(-1j * k * (l1 + s1))) / two_sin1
    b = 1j * (hb * np.exp(1j * k * (l1 + s1)) - ha * np.exp(1j * k * l1)) / two_sin1
    c = 1j * (hc * np.exp(1j * k * (l2 + s2)) - hd * np.exp(1j * k * l2)) / two_sin2
    d = 1j * (hd * np.exp(-1j * k * l2) - hc * np.exp(-1j * k * (l2 + s2))) / two_sin2
    return (
        np.asarray(a, dtype=np.complex128),
        np.asarray(b, dtype=np.complex128),
        np.asarray(c, dtype=np.complex128),
        np.asarray(d, dtype=np.complex128),
    )


def face_quantities(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    *,
    wavenumber: ArrayLike,
    thickness: float,
    characteristic_impedance: float,
) -> tuple[Complex, Complex, Complex, Complex]:
    r"""Face pressures and particle velocities (ASTM E2611-19, Eq. (21)).

    .. math::

       p_0 = A + B, \qquad
       p_d = C e^{-jkd} + D e^{+jkd}

       u_0 = \frac{A - B}{\rho c}, \qquad
       u_d = \frac{C e^{-jkd} - D e^{+jkd}}{\rho c}

    :param a: Upstream forward amplitude ``A``.
    :param b: Upstream backward amplitude ``B``.
    :param c: Downstream forward amplitude ``C``.
    :param d: Downstream backward amplitude ``D``.
    :param wavenumber: Air wavenumber ``k``.
    :param thickness: Specimen thickness ``d``, in metres.
    :param characteristic_impedance: Characteristic impedance ``rho c``, in rayls.
    :return: Tuple ``(p0, pd, u0, ud)`` of face pressures and velocities.
    """
    if characteristic_impedance <= 0.0:
        raise ValueError(_IMPEDANCE_POSITIVE)
    av = np.asarray(a, dtype=np.complex128)
    bv = np.asarray(b, dtype=np.complex128)
    cv = np.asarray(c, dtype=np.complex128)
    dv = np.asarray(d, dtype=np.complex128)
    k = np.asarray(wavenumber, dtype=np.complex128)
    ep = np.exp(-1j * k * thickness)
    em = np.exp(1j * k * thickness)
    p0 = av + bv
    pd = cv * ep + dv * em
    u0 = (av - bv) / characteristic_impedance
    ud = (cv * ep - dv * em) / characteristic_impedance
    return (
        np.asarray(p0, dtype=np.complex128),
        np.asarray(pd, dtype=np.complex128),
        np.asarray(u0, dtype=np.complex128),
        np.asarray(ud, dtype=np.complex128),
    )


@dataclass(frozen=True)
class TransferMatrix:
    r"""Acoustic transfer matrix ``[[T11, T12], [T21, T22]]`` (ASTM E2611-19).

    Relates the pressure and normal particle velocity across a specimen,
    :math:`[p; u]_{x=0} = T \, [p; u]_{x=d}` (Eq. (16)). Each entry is complex and
    may be scalar or a per-frequency array of matching shape.

    The trailing fields retain the measurement context when the matrix comes
    out of :func:`transfer_matrix_two_load` / :func:`transfer_matrix_one_load`
    (tube geometry ``l1``/``s1``/``l2``/``s2``, specimen ``thickness``, tube
    ``diameter`` and canonical cross-section ``shape``, the ``frequency``
    vector when supplied to the solver, and the air
    ``air_characteristic_impedance`` ``rho c``); all default to ``None`` so a
    hand-built matrix (for example :func:`air_layer_transfer_matrix`) is
    unchanged.
    """

    t11: Complex
    t12: Complex
    t21: Complex
    t22: Complex
    l1: float | None = None
    s1: float | None = None
    l2: float | None = None
    s2: float | None = None
    thickness: float | None = None
    diameter: float | None = None
    shape: str | None = None
    frequency: Real | None = None
    air_characteristic_impedance: float | None = None

    def determinant(self) -> Complex:
        r"""Determinant :math:`T_{11} T_{22} - T_{12} T_{21}` (unity for a reciprocal specimen)."""
        return np.asarray(
            self.t11 * self.t22 - self.t12 * self.t21, dtype=np.complex128
        )

    def transmission_loss(self, characteristic_impedance: float) -> Real:
        r"""Normal-incidence transmission loss in dB (ASTM E2611-19, Eq. (26)).

        With

        .. math::

           t = \frac{2 e^{jkd}}
           {T_{11} + T_{12}/(\rho c) + \rho c \, T_{21} + T_{22}}
           \tag{Eq. 25}

           TL = 20 \log_{10} \left| \frac{1}{t} \right|
           = 20 \log_{10} \frac{\lvert T_{11} + T_{12}/(\rho c)
           + \rho c \, T_{21} + T_{22} \rvert}{2}
           \tag{Eq. 26}

        (the :math:`e^{jkd}` factor has unit magnitude for
        a real wavenumber).

        :param characteristic_impedance: Characteristic impedance ``rho c``.
        :return: Transmission loss ``TLn``, in decibels.
        """
        if characteristic_impedance <= 0.0:
            raise ValueError(_IMPEDANCE_POSITIVE)
        rc = characteristic_impedance
        combo = self.t11 + self.t12 / rc + rc * self.t21 + self.t22
        return np.asarray(20.0 * np.log10(np.abs(combo) / 2.0), dtype=np.float64)

    def reflection_hard_backed(self, characteristic_impedance: float) -> Complex:
        r"""Hard-backed reflection coefficient (ASTM E2611-19, Eq. (27)).

        :math:`R = (T_{11} - \rho c T_{21}) / (T_{11} + \rho c T_{21})`.

        :param characteristic_impedance: Characteristic impedance ``rho c``.
        :return: Complex reflection coefficient ``R``.
        """
        if characteristic_impedance <= 0.0:
            raise ValueError(_IMPEDANCE_POSITIVE)
        rc = characteristic_impedance
        return np.asarray(
            (self.t11 - rc * self.t21) / (self.t11 + rc * self.t21),
            dtype=np.complex128,
        )

    def absorption_hard_backed(self, characteristic_impedance: float) -> Real:
        r"""Hard-backed absorption coefficient (ASTM E2611-19, Eq. (28)).

        :math:`\alpha = 1 - \lvert R \rvert^2`.

        :param characteristic_impedance: Characteristic impedance ``rho c``.
        :return: Absorption coefficient ``alpha``.
        """
        r = self.reflection_hard_backed(characteristic_impedance)
        return np.asarray(1.0 - np.abs(r) ** 2, dtype=np.float64)

    def material_wavenumber(self, thickness: float) -> Complex:
        r"""Propagation wavenumber inside the material (ASTM E2611-19, Eq. (29)).

        :math:`k' = \arccos(T_{11}) / d` (complex ``arccos``).

        :param thickness: Specimen thickness ``d``, in metres.
        :return: Complex material wavenumber ``k'``, in reciprocal metres.
        """
        if thickness <= 0.0:
            raise ValueError("'thickness' must be positive.")
        t11 = np.asarray(self.t11, dtype=np.complex128)
        return np.asarray(np.arccos(t11) / thickness, dtype=np.complex128)

    def characteristic_impedance_material(self) -> Complex:
        r"""Characteristic impedance of the material (ASTM E2611-19, Eq. (30)).

        :math:`Z = \sqrt{T_{12} / T_{21}}`.

        :return: Complex characteristic impedance ``Z``, in rayls.
        """
        t12 = np.asarray(self.t12, dtype=np.complex128)
        t21 = np.asarray(self.t21, dtype=np.complex128)
        return np.asarray(np.sqrt(t12 / t21), dtype=np.complex128)

    def plot(
        self,
        frequency: ArrayLike | None = None,
        characteristic_impedance: float | None = None,
        ax: Axes | None = None,
        *,
        language: str = "en",
        **kwargs: Any,
    ) -> Axes:
        """Plot the transmission loss with the hard-backed absorption overlaid.

        Reads the four-pole entries out as the two ASTM E2611-19 spectra a
        laboratory quotes: the normal-incidence transmission loss ``TLn(f)``
        (Eq. (26), the primary curve, left axis) and the hard-backed
        absorption coefficient ``alpha(f)`` (Eq. (28), a muted companion on a
        0..1 right axis). The four-pole entries carry no frequency axis of
        their own, so the plot needs the measurement's ``frequency`` vector
        (matching the shape of the entries) and the air characteristic
        impedance ``rho c``. A matrix built by the solvers retains both
        (``self.frequency`` / ``self.air_characteristic_impedance``), so
        ``plot()`` takes no arguments there; only a hand-built matrix (for
        example :func:`air_layer_transfer_matrix`) must supply them.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` of the transmission-loss curve.

        :param frequency: Frequency vector ``f``, in hertz, matching the shape
            of the matrix entries; ``None`` uses the stored ``frequency``.
        :param characteristic_impedance: Characteristic impedance ``rho c`` of
            the air in the tube, in rayls; ``None`` uses the stored
            ``air_characteristic_impedance``.
        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Plot language: ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the transmission-loss ``plot`` call.
        :return: The axes.
        :raises ValueError: If ``frequency`` or ``characteristic_impedance``
            is neither supplied nor stored on the matrix.
        """
        if frequency is None:
            frequency = self.frequency
        if characteristic_impedance is None:
            characteristic_impedance = self.air_characteristic_impedance
        if frequency is None or characteristic_impedance is None:
            raise ValueError(
                "'frequency' and 'characteristic_impedance' must be supplied "
                "when the matrix does not retain them (hand-built matrices)."
            )
        from ..._i18n import check_language
        from ..._plot.materials import plot_transfer_matrix

        check_language(language)
        return plot_transfer_matrix(
            self,
            frequency,
            characteristic_impedance,
            ax=ax,
            language=language,
            **kwargs,
        )

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the four-microphone tube to scale (dimensioned side view).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the matrix does not retain its tube geometry
            (``l1``/``s1``/``l2``/``s2``/``thickness``).
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_transfer_matrix_geometry

        check_language(language)
        return plot_transfer_matrix_geometry(self, ax=ax, language=language, **kwargs)


def air_layer_transfer_matrix(
    wavenumber: ArrayLike, thickness: float, characteristic_impedance: float
) -> TransferMatrix:
    r"""Analytic transfer matrix of a pure air layer of thickness ``d``.

    .. math::

       T = [[\cos(k d),\; j \rho c \sin(k d)],\;
       [j \sin(k d) / (\rho c),\; \cos(k d)]]

    the classical loss-free layer used to validate the ASTM E2611-19 reduction
    (it is reciprocal, :math:`\operatorname{det}(T) = 1`, and symmetric,
    :math:`T_{11} = T_{22}`).

    :param wavenumber: Air wavenumber ``k``.
    :param thickness: Layer thickness ``d``, in metres.
    :param characteristic_impedance: Characteristic impedance ``rho c``, in rayls.
    :return: The air-layer :class:`TransferMatrix`.
    """
    if characteristic_impedance <= 0.0:
        raise ValueError(_IMPEDANCE_POSITIVE)
    if thickness <= 0.0:
        raise ValueError("'thickness' must be positive.")
    rc = characteristic_impedance
    k = np.asarray(wavenumber, dtype=np.complex128)
    kd = k * thickness
    cos = np.asarray(np.cos(kd), dtype=np.complex128)
    sin = np.asarray(np.sin(kd), dtype=np.complex128)
    return TransferMatrix(
        t11=cos,
        t12=np.asarray(1j * rc * sin, dtype=np.complex128),
        t21=np.asarray(1j * sin / rc, dtype=np.complex128),
        t22=cos,
    )


def _face_from_loads(
    load: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    wavenumber: ArrayLike,
    characteristic_impedance: float,
) -> tuple[Complex, Complex, Complex, Complex]:
    """Face pressures/velocities for one termination (Eqs. (17)-(21))."""
    a, b, c, d = wave_decomposition(
        load[0],
        load[1],
        load[2],
        load[3],
        l1=l1,
        s1=s1,
        l2=l2,
        s2=s2,
        wavenumber=wavenumber,
    )
    return face_quantities(
        a,
        b,
        c,
        d,
        wavenumber=wavenumber,
        thickness=thickness,
        characteristic_impedance=characteristic_impedance,
    )


#: Relative floor below which a two-/one-load solve denominator is treated as
#: near-singular (catastrophic cancellation) and flagged (ASTM E2611-19).
_MATRIX_COND_EPS = 1e-9


def _warn_ill_conditioned(
    den: NDArray[np.complex128],
    scale: NDArray[np.float64],
    context: str,
    *,
    stacklevel: int,
) -> None:
    """Flag a near-singular transfer-matrix solve (poor conditioning)."""
    den_mag = np.abs(np.asarray(den, dtype=np.complex128))
    bad = den_mag < _MATRIX_COND_EPS * np.asarray(scale, dtype=np.float64)
    n = int(np.count_nonzero(bad))
    if n:
        warnings.warn(
            f"{context}: the solve denominator is near-singular at {n} "
            "frequency point(s) (the loads are insufficiently different or the "
            "geometry is near a resonance); results there are unreliable.",
            ImpedanceTubeWarning,
            stacklevel=stacklevel,
        )


def _measurement_context(
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    diameter: float | None,
    shape: str,
    frequency: ArrayLike | None,
    characteristic_impedance: float,
) -> dict[str, Any]:
    """Context fields a solver retains on the :class:`TransferMatrix`."""
    return {
        "l1": l1,
        "s1": s1,
        "l2": l2,
        "s2": s2,
        "thickness": thickness,
        "diameter": diameter,
        "shape": shape if diameter is not None else None,
        "frequency": (
            np.asarray(frequency, dtype=np.float64) if frequency is not None else None
        ),
        "air_characteristic_impedance": characteristic_impedance,
    }


def transfer_matrix_two_load(
    load_a: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    load_b: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    wavenumber: ArrayLike,
    characteristic_impedance: float,
    frequency: ArrayLike | None = None,
    diameter: float | None = None,
    shape: str = "circular",
) -> TransferMatrix:
    r"""Two-load transfer matrix (ASTM E2611-19, Eqs. (17)-(22)).

    Each load is the tuple ``(H1, H2, H3, H4)`` of the four microphone transfer
    functions measured with a different downstream termination. The two loads
    give four equations for the four unknowns (Eq. (22)):

    .. math::

       \begin{aligned}
       \mathrm{DEN} &= p_{da} u_{db} - p_{db} u_{da} \\
       T_{11} &= (p_{0a} u_{db} - p_{0b} u_{da}) / \mathrm{DEN} \\
       T_{12} &= (p_{0b} p_{da} - p_{0a} p_{db}) / \mathrm{DEN} \\
       T_{21} &= (u_{0a} u_{db} - u_{0b} u_{da}) / \mathrm{DEN} \\
       T_{22} &= (p_{da} u_{0b} - p_{db} u_{0a}) / \mathrm{DEN}
       \end{aligned}

    :param load_a: Microphone transfer functions ``(H1, H2, H3, H4)`` for load a.
    :param load_b: Microphone transfer functions ``(H1, H2, H3, H4)`` for load b.
    :param l1: Upstream reference distance ``l1``, in metres.
    :param s1: Upstream microphone spacing ``s1``, in metres.
    :param l2: Downstream reference distance ``l2``, in metres.
    :param s2: Downstream microphone spacing ``s2``, in metres.
    :param thickness: Specimen thickness ``d``, in metres.
    :param wavenumber: Air wavenumber ``k``.
    :param characteristic_impedance: Characteristic impedance ``rho c``.
    :param frequency: Optional frequency vector ``f``, in hertz, retained on
        the result so :meth:`TransferMatrix.plot` needs no arguments.
    :param diameter: Optional tube diameter (circular) or largest section
        dimension (rectangular/square), in metres, that activates the
        plane-wave working-range check (6.2.3-6.2.5, 6.5.4).
    :param shape: Tube cross-section, ``"circular"``, ``"rectangular"`` or
        ``"square"``.
    :return: The specimen :class:`TransferMatrix` (measurement context
        retained on the result).
    """
    canonical = _canonical_shape(shape)
    if diameter is not None:
        _warn_astm_plane_wave(
            wavenumber,
            s1=s1,
            s2=s2,
            diameter=diameter,
            shape=canonical,
            stacklevel=2,
        )
    p0a, pda, u0a, uda = _face_from_loads(
        load_a,
        l1=l1,
        s1=s1,
        l2=l2,
        s2=s2,
        thickness=thickness,
        wavenumber=wavenumber,
        characteristic_impedance=characteristic_impedance,
    )
    p0b, pdb, u0b, udb = _face_from_loads(
        load_b,
        l1=l1,
        s1=s1,
        l2=l2,
        s2=s2,
        thickness=thickness,
        wavenumber=wavenumber,
        characteristic_impedance=characteristic_impedance,
    )
    den = pda * udb - pdb * uda
    _warn_ill_conditioned(
        np.asarray(den, dtype=np.complex128),
        np.abs(pda * udb) + np.abs(pdb * uda),
        "transfer_matrix_two_load",
        stacklevel=2,
    )
    return TransferMatrix(
        t11=np.asarray((p0a * udb - p0b * uda) / den, dtype=np.complex128),
        t12=np.asarray((p0b * pda - p0a * pdb) / den, dtype=np.complex128),
        t21=np.asarray((u0a * udb - u0b * uda) / den, dtype=np.complex128),
        t22=np.asarray((pda * u0b - pdb * u0a) / den, dtype=np.complex128),
        **_measurement_context(
            l1=l1,
            s1=s1,
            l2=l2,
            s2=s2,
            thickness=thickness,
            diameter=diameter,
            shape=canonical,
            frequency=frequency,
            characteristic_impedance=characteristic_impedance,
        ),
    )


def transfer_matrix_one_load(
    load: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    wavenumber: ArrayLike,
    characteristic_impedance: float,
    frequency: ArrayLike | None = None,
    diameter: float | None = None,
    shape: str = "circular",
) -> TransferMatrix:
    r"""One-load transfer matrix, symmetric specimen (ASTM E2611-19, Eqs. (23)-(24)).

    Valid only for a reciprocal **and** symmetric specimen
    (:math:`T_{11} = T_{22}` and :math:`T_{11} T_{22} - T_{12} T_{21} = 1`,
    Eq. (23)). A single termination suffices:

    .. math::

       \begin{aligned}
       \mathrm{DEN} &= p_0 u_d + p_d u_0 \\
       T_{11} = T_{22} &= (p_d u_d + p_0 u_0) / \mathrm{DEN} \\
       T_{12} &= (p_0^{2} - p_d^{2}) / \mathrm{DEN} \\
       T_{21} &= (u_0^{2} - u_d^{2}) / \mathrm{DEN}
       \end{aligned}

    :param load: Microphone transfer functions ``(H1, H2, H3, H4)``.
    :param l1: Upstream reference distance ``l1``, in metres.
    :param s1: Upstream microphone spacing ``s1``, in metres.
    :param l2: Downstream reference distance ``l2``, in metres.
    :param s2: Downstream microphone spacing ``s2``, in metres.
    :param thickness: Specimen thickness ``d``, in metres.
    :param wavenumber: Air wavenumber ``k``.
    :param characteristic_impedance: Characteristic impedance ``rho c``.
    :param frequency: Optional frequency vector ``f``, in hertz, retained on
        the result so :meth:`TransferMatrix.plot` needs no arguments.
    :param diameter: Optional tube diameter (circular) or largest section
        dimension (rectangular/square), in metres, that activates the
        plane-wave working-range check (6.2.3-6.2.5, 6.5.4).
    :param shape: Tube cross-section, ``"circular"``, ``"rectangular"`` or
        ``"square"``.
    :return: The specimen :class:`TransferMatrix` (measurement context
        retained on the result).
    """
    canonical = _canonical_shape(shape)
    if diameter is not None:
        _warn_astm_plane_wave(
            wavenumber,
            s1=s1,
            s2=s2,
            diameter=diameter,
            shape=canonical,
            stacklevel=2,
        )
    p0, pd, u0, ud = _face_from_loads(
        load,
        l1=l1,
        s1=s1,
        l2=l2,
        s2=s2,
        thickness=thickness,
        wavenumber=wavenumber,
        characteristic_impedance=characteristic_impedance,
    )
    den = p0 * ud + pd * u0
    _warn_ill_conditioned(
        np.asarray(den, dtype=np.complex128),
        np.abs(p0 * ud) + np.abs(pd * u0),
        "transfer_matrix_one_load",
        stacklevel=2,
    )
    t_diag = np.asarray((pd * ud + p0 * u0) / den, dtype=np.complex128)
    return TransferMatrix(
        t11=t_diag,
        t12=np.asarray((p0**2 - pd**2) / den, dtype=np.complex128),
        t21=np.asarray((u0**2 - ud**2) / den, dtype=np.complex128),
        t22=t_diag,
        **_measurement_context(
            l1=l1,
            s1=s1,
            l2=l2,
            s2=s2,
            thickness=thickness,
            diameter=diameter,
            shape=canonical,
            frequency=frequency,
            characteristic_impedance=characteristic_impedance,
        ),
    )
