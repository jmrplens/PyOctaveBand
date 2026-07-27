#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Impedance-tube material characterisation.

Three complementary standardised methods are implemented, each kept in its
own sign convention (they are **not** interchangeable):

* **BS EN ISO 10534-2:2001** - two-microphone transfer-function method. The
  complex reflection factor ``r`` at the sample surface is obtained from the
  measured transfer function ``H12`` between two microphones, and from it the
  surface impedance and the normal-incidence absorption coefficient
  (Clause 7, Eqs. (17)-(20)). Time convention ``e^{+j w t}``; the incident
  wave carries ``e^{+j k0 x}`` and the reflected wave ``e^{-j k0 x}`` (Annex D,
  Eqs. (D.1)-(D.8)). The complex wavenumber is ``k0 = k0' - j k0''`` with the
  attenuation constant ``k0''`` (Clause 2.6, Annex A). Air properties from
  Clause 7.2, Eqs. (5)/(7), use temperature in **kelvin**.

* **BS EN ISO 10534-1:2001** - standing-wave-ratio method. The reflection
  magnitude, phase, absorption coefficient and normalised impedance follow
  from the measured standing-wave ratio and the position of the first pressure
  minimum (Clause 5, Eqs. (12)-(26)).

* **ASTM E2611-19** - four-microphone transfer-matrix method. The wave field
  is decomposed into forward/backward amplitudes on each side of the specimen
  (Eqs. (17)-(20)), the face pressures and particle velocities are formed
  (Eq. (21)) and the transfer matrix ``[[T11, T12], [T21, T22]]`` is solved
  from a two-load (Eq. (22)) or a symmetric one-load (Eq. (24)) measurement.
  Transmission loss (Eq. (26)), hard-backed reflection/absorption
  (Eqs. (27)/(28)) and the material wavenumber/characteristic impedance
  (Eqs. (29)/(30)) follow. Time convention ``e^{+j w t}`` with the forward
  wave carried by ``e^{-j k x}`` (Eq. (21)); air properties from Clause 8.2/8.3,
  Eqs. (4)/(5), use temperature in **degrees Celsius**.

The two standards adopt different sign ansaetze and different temperature
units on purpose; the helpers are named per standard so the two are never
mixed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.types import Real
from .._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

Complex = NDArray[np.complex128]

#: Reference speed of sound of ISO 10534-2 Eq. (5), in m/s (343,2 exactly).
_ISO_C_REF = 343.2
#: Reference temperature of ISO 10534-2 Eqs. (5)/(7), in kelvin (293 exactly).
_ISO_T_REF = 293.0
#: Reference air density of ISO 10534-2 Eq. (7), in kg/m3 (1,186 exactly).
_ISO_RHO_REF = 1.186
#: Reference atmospheric pressure of ISO 10534-2 Eq. (7), in kPa (101,325).
_ISO_P_REF = 101.325
#: Leading constant of the ISO 10534-2 Eq. (A.18) attenuation estimate.
_ISO_ATTEN_CONST = 1.94e-2

#: Leading constant of ASTM E2611-19 Eq. (4), speed of sound (20,047).
_ASTM_C_CONST = 20.047
#: Celsius-to-kelvin offset used by ASTM E2611-19 Eqs. (4)/(5) (273,15).
_ASTM_T0 = 273.15
#: Reference air density of ASTM E2611-19 Eq. (5), in kg/m3 (1,290).
_ASTM_RHO_REF = 1.290
#: Reference atmospheric pressure of ASTM E2611-19 Eq. (5), in kPa (101,325).
_ASTM_P_REF = 101.325

#: Upper-frequency plane-wave factor, circular tube (ISO 10534-2 Eq. (2)).
_ISO_KU_CIRCULAR = 0.58
#: Upper-frequency plane-wave factor, rectangular tube (ISO 10534-2 Eq. (3)).
_ISO_KU_RECTANGULAR = 0.50
#: Microphone-spacing factor for the upper limit (ISO 10534-2 Eq. (4)).
_ISO_KU_SPACING = 0.45
#: Lower-limit factor: spacing recommended > 5 % of the wavelength (Clause 4.2),
#: i.e. ``f_l = c0 / (20 s)``.
_ISO_LOWER_WAVELENGTH_FRACTION = 20.0

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

#: Aliases accepted for the tube cross-section ``shape`` arguments. A square
#: tube is the rectangular case with equal sides (ISO 10534-2, 4.1;
#: ASTM E2611-19, 6.2.5).
_SHAPE_ALIASES = {
    "circular": "circular",
    "rectangular": "rectangular",
    "square": "rectangular",
}

__all__ = [
    "ImpedanceTubeResult",
    "ImpedanceTubeWarning",
    "TransferMatrix",
    "absorption_from_reflection",
    "air_density_astm",
    "air_density_iso",
    "air_layer_transfer_matrix",
    "apply_mic_calibration",
    "characteristic_impedance",
    "face_quantities",
    "hydraulic_diameter",
    "mic_calibration_factor",
    "normalized_surface_admittance",
    "normalized_surface_impedance",
    "plane_wave_frequency_range",
    "plane_wave_frequency_range_astm",
    "reflection_factor",
    "speed_of_sound_astm",
    "speed_of_sound_iso",
    "standing_wave_absorption",
    "standing_wave_normalized_impedance",
    "standing_wave_ratio_from_level",
    "standing_wave_reflection",
    "standing_wave_reflection_magnitude",
    "surface_impedance",
    "transfer_matrix_one_load",
    "transfer_matrix_two_load",
    "tube_attenuation_constant",
    "tube_wavenumber",
    "two_microphone_impedance",
    "wave_decomposition",
]


class ImpedanceTubeWarning(PhonometryWarning):
    """Advisory for out-of-plane-wave-range impedance-tube frequencies."""


# ---------------------------------------------------------------------------
# Air properties (kept separate: ISO uses kelvin, ASTM uses Celsius).
# ---------------------------------------------------------------------------
def speed_of_sound_iso(temperature: ArrayLike) -> Real:
    """Speed of sound in air (ISO 10534-2:2001, Eq. (5)).

    ``c0 = 343,2 * sqrt(T / 293)``.

    :param temperature: Air temperature ``T``, in **kelvin**.
    :return: Speed of sound ``c0``, in metres per second.
    """
    t = np.asarray(temperature, dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("'temperature' must be positive (kelvin).")
    return np.asarray(_ISO_C_REF * np.sqrt(t / _ISO_T_REF), dtype=np.float64)


def air_density_iso(
    temperature: ArrayLike, atmospheric_pressure: ArrayLike = _ISO_P_REF
) -> Real:
    """Air density (ISO 10534-2:2001, Eq. (7)).

    ``rho = rho0 * (pa * T0) / (p0 * T)`` with ``rho0 = 1,186 kg/m3``,
    ``T0 = 293 K`` and ``p0 = 101,325 kPa``.

    :param temperature: Air temperature ``T``, in **kelvin**.
    :param atmospheric_pressure: Atmospheric pressure ``pa``, in kilopascals
        (default 101,325 kPa).
    :return: Air density ``rho``, in kilograms per cubic metre.
    """
    t = np.asarray(temperature, dtype=np.float64)
    pa = np.asarray(atmospheric_pressure, dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("'temperature' must be positive (kelvin).")
    if np.any(pa <= 0.0):
        raise ValueError("'atmospheric_pressure' must be positive (kPa).")
    return np.asarray(
        _ISO_RHO_REF * (pa * _ISO_T_REF) / (_ISO_P_REF * t), dtype=np.float64
    )


def speed_of_sound_astm(temperature: ArrayLike) -> Real:
    """Speed of sound in air (ASTM E2611-19, Eq. (4)).

    ``c = 20,047 * sqrt(273,15 + T)``.

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
    """Air density (ASTM E2611-19, Eq. (5)).

    ``rho = 1,290 * (P / 101,325) * (273,15 / (273,15 + T))``.

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
# Cross-section helpers, wavenumber and tube attenuation (ISO 10534-2).
# ---------------------------------------------------------------------------
def _canonical_shape(shape: str) -> str:
    """Normalise a cross-section name to ``"circular"``/``"rectangular"``."""
    try:
        return _SHAPE_ALIASES[shape]
    except KeyError:
        raise ValueError(
            "'shape' must be 'circular', 'rectangular' or 'square'."
        ) from None


def hydraulic_diameter(width: float, height: float) -> float:
    """Hydraulic diameter of a rectangular tube, ``4 A / P`` (ISO 10534-2, A.2.1.5).

    For a rectangular cross-section of side lengths ``w`` and ``h`` the ratio
    of four times the area to the perimeter reduces to
    ``d_h = 2 w h / (w + h)``; a square tube gives ``d_h`` equal to the side
    length. This is the ``d`` the Eq. (A.18) attenuation estimate expects for
    rectangular tubes (see :func:`tube_attenuation_constant`).

    :param width: Inner side length ``w``, in metres.
    :param height: Inner side length ``h``, in metres.
    :return: Hydraulic diameter ``d_h = 4 A / P``, in metres.
    """
    if width <= 0.0 or height <= 0.0:
        raise ValueError("'width' and 'height' must be positive.")
    return float(2.0 * width * height / (width + height))


def tube_attenuation_constant(
    frequency: ArrayLike, speed_of_sound: float, diameter: float
) -> Real:
    """Lower-bound tube attenuation constant ``k0''`` (ISO 10534-2, Eq. (A.18)).

    ``k0'' = 1,94e-2 * sqrt(f) / (c0 * d)`` (nepers per metre). This ignores
    porous-wall and object losses and is therefore a lower limit (Clause A.2.1.5).

    :param frequency: Frequency ``f``, in hertz (scalar or per band).
    :param speed_of_sound: Speed of sound ``c0``, in metres per second.
    :param diameter: Circular-tube diameter ``d``, in metres, or the hydraulic
        diameter ``4 * area / perimeter`` for a rectangular tube (see
        :func:`hydraulic_diameter`).
    :return: Attenuation constant ``k0''``, in nepers per metre.
    """
    if speed_of_sound <= 0.0:
        raise ValueError("'speed_of_sound' must be positive.")
    if diameter <= 0.0:
        raise ValueError("'diameter' must be positive.")
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f < 0.0):
        raise ValueError("'frequency' must be non-negative.")
    return np.asarray(
        _ISO_ATTEN_CONST * np.sqrt(f) / (speed_of_sound * diameter),
        dtype=np.float64,
    )


def tube_wavenumber(
    frequency: ArrayLike,
    speed_of_sound: float,
    *,
    attenuation: ArrayLike | None = None,
) -> Complex:
    """Complex wavenumber ``k0 = k0' - j k0''`` (ISO 10534-2, Clause 2.6).

    The real part is ``k0' = 2 pi f / c0`` (Eq. (2)); the optional attenuation
    constant ``k0''`` enters with a **minus** sign on the imaginary part
    (Clause 2.6 NOTE, Eq. (A.1)).

    :param frequency: Frequency ``f``, in hertz (scalar or per band).
    :param speed_of_sound: Speed of sound ``c0``, in metres per second.
    :param attenuation: Attenuation constant ``k0''``, in nepers per metre
        (scalar or matching ``frequency``); ``None`` gives the lossless real
        wavenumber. Obtain a lower-bound estimate from
        :func:`tube_attenuation_constant`.
    :return: Complex wavenumber ``k0``, in reciprocal metres.
    """
    if speed_of_sound <= 0.0:
        raise ValueError("'speed_of_sound' must be positive.")
    f = np.asarray(frequency, dtype=np.float64)
    k_real = 2.0 * np.pi * f / speed_of_sound
    if attenuation is None:
        k_imag: NDArray[np.float64] = np.zeros_like(k_real)
    else:
        k_imag = np.asarray(attenuation, dtype=np.float64)
    return np.asarray(k_real - 1j * k_imag, dtype=np.complex128)


# ---------------------------------------------------------------------------
# ISO 10534-2: reflection factor, impedance, absorption.
# ---------------------------------------------------------------------------
def reflection_factor(
    h12: ArrayLike,
    *,
    spacing: float,
    x1: float,
    wavenumber: ArrayLike,
) -> Complex:
    """Complex reflection factor at the sample surface (ISO 10534-2, Eq. (17)).

    ``r = ((H12 - HI) / (HR - H12)) * exp(+2 j k0 x1)`` with the incident- and
    reflected-wave transfer functions ``HI = exp(-j k0 s)`` (Eq. (D.5)) and
    ``HR = exp(+j k0 s)`` (Eq. (D.6)), ``s`` the microphone spacing and ``x1``
    the distance from the sample to the **farther** microphone (Clause 7.7).

    :param h12: Measured transfer function ``H12`` between microphone
        positions 1 and 2 (Clause 7.6, Eq. (14)); complex, scalar or per band.
        It must already be corrected for microphone mismatch (see
        :func:`apply_mic_calibration`).
    :param spacing: Microphone spacing ``s = x1 - x2``, in metres.
    :param x1: Distance from the sample surface to the farther microphone
        (position 1), in metres.
    :param wavenumber: Complex wavenumber ``k0`` (from :func:`tube_wavenumber`),
        scalar or per band.
    :return: Complex reflection factor ``r`` at the reference plane.
    """
    if spacing <= 0.0:
        raise ValueError("'spacing' must be positive.")
    if x1 <= 0.0:
        raise ValueError("'x1' must be positive.")
    h = np.asarray(h12, dtype=np.complex128)
    k0 = np.asarray(wavenumber, dtype=np.complex128)
    h_i = np.exp(-1j * k0 * spacing)
    h_r = np.exp(1j * k0 * spacing)
    r = (h - h_i) / (h_r - h) * np.exp(2j * k0 * x1)
    return np.asarray(r, dtype=np.complex128)


def normalized_surface_impedance(reflection: ArrayLike) -> Complex:
    """Normalised surface impedance ``Z / (rho c0)`` (ISO 10534-2, Eq. (19)).

    ``Z / (rho c0) = (1 + r) / (1 - r)``.

    :param reflection: Complex reflection factor ``r``.
    :return: Normalised surface impedance ``Z / (rho c0)`` (complex).
    """
    r = np.asarray(reflection, dtype=np.complex128)
    return np.asarray((1.0 + r) / (1.0 - r), dtype=np.complex128)


def surface_impedance(
    reflection: ArrayLike, characteristic_impedance: float
) -> Complex:
    """Absolute surface impedance ``Z`` (ISO 10534-2, Eq. (19)).

    ``Z = rho c0 * (1 + r) / (1 - r)``.

    :param reflection: Complex reflection factor ``r``.
    :param characteristic_impedance: Characteristic impedance of air
        ``rho c0``, in rayls (``rho`` and ``c0`` from the Clause 7.2 helpers).
    :return: Surface impedance ``Z``, in rayls (complex).
    """
    if characteristic_impedance <= 0.0:
        raise ValueError("'characteristic_impedance' must be positive.")
    return np.asarray(
        characteristic_impedance * normalized_surface_impedance(reflection),
        dtype=np.complex128,
    )


def normalized_surface_admittance(reflection: ArrayLike) -> Complex:
    """Normalised surface admittance ``G rho c0`` (ISO 10534-2, Eq. (20)).

    ``G rho c0 = (rho c0) / Z = (1 - r) / (1 + r)``.

    :param reflection: Complex reflection factor ``r``.
    :return: Normalised surface admittance (complex).
    """
    r = np.asarray(reflection, dtype=np.complex128)
    return np.asarray((1.0 - r) / (1.0 + r), dtype=np.complex128)


def absorption_from_reflection(reflection: ArrayLike) -> Real:
    """Normal-incidence absorption coefficient (ISO 10534-2, Eq. (18)).

    ``alpha = 1 - |r|^2``. This form is shared with ISO 10534-1 Eq. (9) and
    ASTM E2611-19 Eq. (28).

    :param reflection: Complex reflection factor ``r``.
    :return: Absorption coefficient ``alpha`` (real).
    """
    r = np.asarray(reflection, dtype=np.complex128)
    return np.asarray(1.0 - np.abs(r) ** 2, dtype=np.float64)


def mic_calibration_factor(
    h12_config1: ArrayLike, h12_config2: ArrayLike
) -> Complex:
    """Microphone-mismatch calibration factor ``Hc`` (ISO 10534-2, Eq. (10)).

    ``Hc = sqrt(H12^I / H12^II)`` from a transfer function measured on an
    absorptive specimen in the standard configuration (I) and with the two
    microphones physically interchanged (II) - the cabling to the analyser is
    **not** swapped (Clause 7.5.2).

    :param h12_config1: Transfer function ``H12^I`` in the standard configuration.
    :param h12_config2: Transfer function ``H12^II`` with microphones swapped.
    :return: Complex calibration factor ``Hc``.
    """
    h1 = np.asarray(h12_config1, dtype=np.complex128)
    h2 = np.asarray(h12_config2, dtype=np.complex128)
    return np.asarray(np.sqrt(h1 / h2), dtype=np.complex128)


def apply_mic_calibration(
    h12_uncorrected: ArrayLike, calibration_factor: ArrayLike
) -> Complex:
    """Apply the microphone calibration factor (ISO 10534-2, Eq. (13)).

    ``H12 = H12_uncorrected / Hc``.

    :param h12_uncorrected: Uncorrected measured transfer function.
    :param calibration_factor: Calibration factor ``Hc`` from
        :func:`mic_calibration_factor`.
    :return: Corrected transfer function ``H12``.
    """
    h = np.asarray(h12_uncorrected, dtype=np.complex128)
    hc = np.asarray(calibration_factor, dtype=np.complex128)
    return np.asarray(h / hc, dtype=np.complex128)


def plane_wave_frequency_range(
    spacing: float,
    speed_of_sound: float,
    *,
    diameter: float | None = None,
    shape: str = "circular",
) -> tuple[float, float]:
    """Working plane-wave frequency range ``(f_l, f_u)`` (ISO 10534-2, 4.2-4.5).

    The upper limit is the smaller of the microphone-spacing bound
    ``f_u s < 0,45 c0`` (Eq. (4)) and, when the tube ``diameter`` is given, the
    cut-on bound ``f_u d < 0,58 c0`` for a circular tube (Eq. (2)) or
    ``< 0,50 c0`` for a rectangular tube (Eq. (3)). The lower limit uses the
    Clause 4.2 guideline that the spacing exceed 5 % of the wavelength, i.e.
    ``f_l = c0 / (20 s)``.

    :param spacing: Microphone spacing ``s``, in metres.
    :param speed_of_sound: Speed of sound ``c0``, in metres per second.
    :param diameter: Tube diameter (circular) or maximum lateral dimension
        (rectangular/square) ``d``, in metres; ``None`` applies only the
        spacing bound.
    :param shape: ``"circular"``, ``"rectangular"`` or ``"square"`` (a square
        tube is the rectangular bound with ``d`` the side length).
    :return: Tuple ``(f_l, f_u)`` of the lower and upper frequency limits, in Hz.
    """
    return _frequency_range(
        spacing,
        speed_of_sound,
        diameter=diameter,
        shape=shape,
        ku_circular=_ISO_KU_CIRCULAR,
        ku_rectangular=_ISO_KU_RECTANGULAR,
        ku_spacing=_ISO_KU_SPACING,
        lower_fraction=_ISO_LOWER_WAVELENGTH_FRACTION,
    )


def plane_wave_frequency_range_astm(
    spacing: float,
    speed_of_sound: float,
    *,
    diameter: float | None = None,
    shape: str = "circular",
) -> tuple[float, float]:
    """Working plane-wave frequency range ``(f_l, f_u)`` (ASTM E2611-19).

    The upper limit is the smaller of the microphone-spacing bound
    ``s <= 0,8 c / (2 f_u)``, i.e. ``f_u s < 0,40 c`` (6.5.4), and, when the
    tube ``diameter`` is given, the cut-on bound ``f_u < K c / d`` with
    ``K = 0,586`` for a circular tube (6.2.4.1, Eq. (2)) or ``K = 0,500`` for
    a rectangular tube with ``d`` the largest section dimension (6.2.5). The
    lower limit follows 6.2.3: the spacing shall be greater than 1 % of the
    wavelength, i.e. ``f_l = c / (100 s)``.

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


def _frequency_range(
    spacing: float,
    speed_of_sound: float,
    *,
    diameter: float | None,
    shape: str,
    ku_circular: float,
    ku_rectangular: float,
    ku_spacing: float,
    lower_fraction: float,
) -> tuple[float, float]:
    """Shared ISO/ASTM plane-wave frequency-range arithmetic."""
    if spacing <= 0.0:
        raise ValueError("'spacing' must be positive.")
    if speed_of_sound <= 0.0:
        raise ValueError("'speed_of_sound' must be positive.")
    canonical = _canonical_shape(shape)
    f_upper = ku_spacing * speed_of_sound / spacing
    if diameter is not None:
        if diameter <= 0.0:
            raise ValueError("'diameter' must be positive.")
        factor = ku_circular if canonical == "circular" else ku_rectangular
        f_upper = min(f_upper, factor * speed_of_sound / diameter)
    f_lower = speed_of_sound / (lower_fraction * spacing)
    return f_lower, f_upper


def _warn_frequency_range(
    frequency: NDArray[np.float64],
    f_lower: float,
    f_upper: float,
    *,
    stacklevel: int,
) -> None:
    """Advise when any frequency falls outside the plane-wave range."""
    if np.any(frequency < f_lower) or np.any(frequency > f_upper):
        warnings.warn(
            f"Frequencies outside the plane-wave range "
            f"[{f_lower:.1f}, {f_upper:.1f}] Hz (ISO 10534-2:2001, Eqs. (1)-(4)); "
            "results there are advisory.",
            ImpedanceTubeWarning,
            stacklevel=stacklevel,
        )


@dataclass(frozen=True)
class ImpedanceTubeResult:
    """Two-microphone impedance-tube result (ISO 10534-2:2001).

    All arrays share the shape of ``frequency``. ``reflection`` is the complex
    reflection factor ``r`` at the sample surface (Eq. (17)),
    ``surface_impedance`` the absolute surface impedance ``Z`` in rayls
    (Eq. (19)), ``normalized_impedance`` the ratio ``Z / (rho c0)`` (Eq. (19))
    and ``absorption`` the normal-incidence coefficient ``alpha = 1 - |r|^2``
    (Eq. (18)).

    The trailing fields retain the tube geometry the reduction was run with
    (microphone ``spacing`` ``s``, distance ``x1`` from the sample to the
    farther microphone, tube ``diameter`` and cross-section ``shape``, stored
    canonically as ``"circular"``/``"rectangular"`` - a ``"square"`` input is
    kept as ``"rectangular"``); they default to ``None`` when not supplied to
    :func:`two_microphone_impedance`.
    """

    frequency: Real
    reflection: Complex
    surface_impedance: Complex
    normalized_impedance: Complex
    absorption: Real
    spacing: float | None = None
    x1: float | None = None
    diameter: float | None = None
    shape: str | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the absorption spectrum ``alpha(f)`` with ``|r|`` overlaid.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.materials import plot_impedance_tube

        check_language(language)
        return plot_impedance_tube(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 10534-2 impedance-tube test-report fiche to a PDF.

        Writes a one-page accredited normal-incidence report (BS EN ISO
        10534-2:2001, two-microphone transfer-function method): the
        standard-basis line, an optional metadata header block (client,
        specimen, tube diameter ``d``, microphone spacing ``s``, the measured
        frequency range, mounting, climate ...), a two-panel body with the
        per-frequency table (frequency, absorption ``alpha`` and the
        real/imaginary parts of the normalised surface impedance
        ``z = Z / (rho c0)``) beside the ``alpha(f)`` curve, and a footer with
        the fixed disclaimer. ISO 10534-2 is a characterisation, so there is no
        pass/fail verdict and no single-number rating (the random-incidence
        weighted ``alpha_w`` is an ISO 11654 / ISO 354 quantity, not comparable
        to the normal-incidence coefficient reported here).

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche whose header shows only the
            measured frequency range. The applicable descriptive/geometric
            fields are ``client``, ``manufacturer``, ``specimen``,
            ``tube_diameter``, ``tube_shape``, ``mic_spacing``, ``mounting``,
            ``test_room``,
            ``test_date``, ``temperature``, ``pressure``,
            ``measurement_standard``, ``laboratory``, ``operator``,
            ``report_id`` and ``notes``. The ``requirement`` field is ignored
            (ISO 10534-2 has no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the value table inserts the
            reflection-factor magnitude ``|r|`` column.
        :param language: Fiche language: ``"en"`` (default, English, decimal
            point) or ``"es"`` (Spanish, decimal comma).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso10534 import render_iso10534_report

        return render_iso10534_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def two_microphone_impedance(
    h12: ArrayLike,
    *,
    frequency: ArrayLike,
    spacing: float,
    x1: float,
    speed_of_sound: float,
    characteristic_impedance: float,
    attenuation: ArrayLike | None = None,
    diameter: float | None = None,
    shape: str = "circular",
) -> ImpedanceTubeResult:
    """Full two-microphone reduction (ISO 10534-2:2001, Clause 7).

    Builds the complex wavenumber (Clause 2.6), the reflection factor
    (Eq. (17)), the surface impedance (Eq. (19)) and the absorption coefficient
    (Eq. (18)) from the measured transfer function ``H12``. When ``diameter`` is
    supplied, frequencies outside the plane-wave range (Eqs. (1)-(4)) raise an
    :class:`ImpedanceTubeWarning`; the results are still returned.

    :param h12: Measured (mismatch-corrected) transfer function ``H12``.
    :param frequency: Frequency vector ``f``, in hertz.
    :param spacing: Microphone spacing ``s``, in metres.
    :param x1: Distance from the sample to the farther microphone, in metres.
    :param speed_of_sound: Speed of sound ``c0``, in m/s (see
        :func:`speed_of_sound_iso`).
    :param characteristic_impedance: Characteristic impedance ``rho c0``, in
        rayls.
    :param attenuation: Optional tube attenuation constant ``k0''``, in
        nepers/m (see :func:`tube_attenuation_constant`).
    :param diameter: Optional tube diameter/lateral dimension, in metres, that
        activates the plane-wave range check.
    :param shape: Tube cross-section, ``"circular"``, ``"rectangular"`` or
        ``"square"``.
    :return: An :class:`ImpedanceTubeResult` (the tube geometry is retained on
        the result).
    """
    f = np.asarray(frequency, dtype=np.float64)
    k0 = tube_wavenumber(f, speed_of_sound, attenuation=attenuation)
    r = reflection_factor(h12, spacing=spacing, x1=x1, wavenumber=k0)
    canonical = _canonical_shape(shape)
    if diameter is not None:
        f_lower, f_upper = plane_wave_frequency_range(
            spacing, speed_of_sound, diameter=diameter, shape=canonical
        )
        _warn_frequency_range(f, f_lower, f_upper, stacklevel=2)
    return ImpedanceTubeResult(
        frequency=f,
        reflection=r,
        surface_impedance=surface_impedance(r, characteristic_impedance),
        normalized_impedance=normalized_surface_impedance(r),
        absorption=absorption_from_reflection(r),
        spacing=spacing,
        x1=x1,
        diameter=diameter,
        shape=canonical if diameter is not None else None,
    )


# ---------------------------------------------------------------------------
# ISO 10534-1: standing-wave-ratio method.
# ---------------------------------------------------------------------------
def standing_wave_ratio_from_level(level_difference: ArrayLike) -> Real:
    """Standing-wave ratio from a level difference (ISO 10534-1, Eq. (15)).

    ``s = 10^(dL / 20)`` with ``dL = L_max - L_min`` in decibels.

    :param level_difference: Level difference ``dL = L_max - L_min``, in dB.
    :return: Standing-wave ratio ``s`` (>= 1).
    """
    dl = np.asarray(level_difference, dtype=np.float64)
    if np.any(dl < 0.0):
        raise ValueError("'level_difference' must be non-negative.")
    return np.asarray(10.0 ** (dl / 20.0), dtype=np.float64)


def _check_swr(swr: NDArray[np.float64]) -> None:
    if np.any(swr < 1.0):
        raise ValueError("Standing-wave ratio 's' must be >= 1.")


def standing_wave_reflection_magnitude(swr: ArrayLike) -> Real:
    """Reflection magnitude from the standing-wave ratio (ISO 10534-1, Eq. (14)).

    ``|r| = (s - 1) / (s + 1)``.

    :param swr: Standing-wave ratio ``s`` (>= 1).
    :return: Reflection magnitude ``|r|`` in ``[0, 1]``.
    """
    s = np.asarray(swr, dtype=np.float64)
    _check_swr(s)
    return np.asarray((s - 1.0) / (s + 1.0), dtype=np.float64)


def standing_wave_absorption(swr: ArrayLike) -> Real:
    """Absorption coefficient from the standing-wave ratio (ISO 10534-1).

    Combining ``alpha = 1 - |r|^2`` (Eq. (9)) with ``|r| = (s - 1)/(s + 1)``
    (Eq. (14)) gives ``alpha = 4 s / (s + 1)^2``.

    :param swr: Standing-wave ratio ``s`` (>= 1).
    :return: Absorption coefficient ``alpha`` in ``[0, 1]``.
    """
    s = np.asarray(swr, dtype=np.float64)
    _check_swr(s)
    return np.asarray(4.0 * s / (s + 1.0) ** 2, dtype=np.float64)


def _standing_wave_phase(
    first_min_distance: NDArray[np.float64], wavelength: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Reflection phase from the first minimum (ISO 10534-1, Eq. (20))."""
    if np.any(wavelength <= 0.0):
        raise ValueError("'wavelength' must be positive.")
    if np.any(first_min_distance < 0.0):
        raise ValueError("'first_min_distance' must be non-negative.")
    return np.asarray(
        np.pi * (4.0 * first_min_distance / wavelength - 1.0), dtype=np.float64
    )


def standing_wave_reflection(
    swr: ArrayLike, first_min_distance: ArrayLike, wavelength: ArrayLike
) -> Complex:
    """Complex reflection factor from the standing wave (ISO 10534-1, Eqs. (17)-(23)).

    ``r = |r| e^{j phi}`` with ``|r| = (s - 1)/(s + 1)`` (Eq. (14)) and the
    phase at the first pressure minimum ``phi = pi (4 x_min1 / lambda0 - 1)``
    (Eq. (20)).

    :param swr: Standing-wave ratio ``s`` (>= 1).
    :param first_min_distance: Distance ``x_min1`` from the reference plane to
        the first pressure minimum (toward the source), in metres.
    :param wavelength: Wavelength ``lambda0``, in metres (Eq. (27)).
    :return: Complex reflection factor ``r``.
    """
    magnitude = standing_wave_reflection_magnitude(swr)
    phase = _standing_wave_phase(
        np.asarray(first_min_distance, dtype=np.float64),
        np.asarray(wavelength, dtype=np.float64),
    )
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex128)


def standing_wave_normalized_impedance(
    swr: ArrayLike, first_min_distance: ArrayLike, wavelength: ArrayLike
) -> Complex:
    """Normalised impedance from the standing wave (ISO 10534-1, Eqs. (24)-(26)).

    ``z = Z / Z0 = (1 + r) / (1 - r)``; the real/imaginary split is Eqs. (25)/(26).

    :param swr: Standing-wave ratio ``s`` (>= 1).
    :param first_min_distance: Distance ``x_min1`` to the first minimum, in metres.
    :param wavelength: Wavelength ``lambda0``, in metres.
    :return: Normalised surface impedance ``z`` (complex).
    """
    r = standing_wave_reflection(swr, first_min_distance, wavelength)
    return np.asarray((1.0 + r) / (1.0 - r), dtype=np.complex128)


# ---------------------------------------------------------------------------
# ASTM E2611-19: four-microphone transfer-matrix method.
# ---------------------------------------------------------------------------
def _warn_astm_plane_wave(
    wavenumber: ArrayLike,
    *,
    s1: float,
    s2: float,
    diameter: float,
    shape: str,
    stacklevel: int,
) -> None:
    """Advise when wavenumbers leave the ASTM E2611-19 plane-wave range.

    The check runs on the real part of ``k`` so no speed of sound is needed:
    ``f s < 0,40 c`` maps to ``k s < 0,80 pi`` (6.5.4), ``f d < K c`` to
    ``k d < 2 pi K`` (6.2.4.1/6.2.5) and the greater-than-1 %-of-wavelength
    spacing bound to ``k s > 0,02 pi`` (6.2.3). The upper spacing bound binds
    every microphone pair (largest spacing), the lower one the smallest.
    """
    if diameter <= 0.0:
        raise ValueError("'diameter' must be positive.")
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
    """Decompose the wave field into ``(A, B, C, D)`` (ASTM E2611-19, Eqs. (17)-(20)).

    The exponents are implemented exactly as printed::

        A = j (H1 e^{-j k l1}       - H2 e^{-j k (l1+s1)}) / (2 sin(k s1))
        B = j (H2 e^{+j k (l1+s1)}  - H1 e^{+j k l1})      / (2 sin(k s1))
        C = j (H3 e^{+j k (l2+s2)}  - H4 e^{+j k l2})      / (2 sin(k s2))
        D = j (H4 e^{-j k l2}       - H3 e^{-j k (l2+s2)}) / (2 sin(k s2))

    ``A``/``B`` are the forward/backward complex amplitudes on the upstream
    (source) side and ``C``/``D`` those on the downstream side, all referenced
    to the front face ``x = 0``. With the ``e^{+j w t}`` / forward-``e^{-j k x}``
    convention these exponents correspond to the microphone whose transfer
    function is ``H2`` sitting nearest the front face at distance ``l1`` (and
    ``H1`` at ``l1 + s1``), and to ``H3`` nearest the downstream side at ``l2``
    (and ``H4`` at ``l2 + s2``), with ``l1``, ``l2`` measured from the front
    reference plane. The convention was locked down against the analytic
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
            wavenumber, s1=s1, s2=s2, diameter=diameter, shape=canonical,
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
    """Face pressures and particle velocities (ASTM E2611-19, Eq. (21)).

    ``p0 = A + B``, ``pd = C e^{-j k d} + D e^{+j k d}``,
    ``u0 = (A - B) / (rho c)``, ``ud = (C e^{-j k d} - D e^{+j k d}) / (rho c)``.

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
        raise ValueError("'characteristic_impedance' must be positive.")
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
    """Acoustic transfer matrix ``[[T11, T12], [T21, T22]]`` (ASTM E2611-19).

    Relates the pressure and normal particle velocity across a specimen,
    ``[p; u]_{x=0} = T [p; u]_{x=d}`` (Eq. (16)). Each entry is complex and
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
        """Determinant ``T11 T22 - T12 T21`` (unity for a reciprocal specimen)."""
        return np.asarray(
            self.t11 * self.t22 - self.t12 * self.t21, dtype=np.complex128
        )

    def transmission_loss(self, characteristic_impedance: float) -> Real:
        """Normal-incidence transmission loss in dB (ASTM E2611-19, Eq. (26)).

        With ``t = 2 e^{j k d} / (T11 + T12/(rho c) + rho c T21 + T22)``
        (Eq. (25)), ``TL = 20 log10 |1/t| = 20 log10 |T11 + T12/(rho c) +
        rho c T21 + T22| / 2`` (the ``e^{j k d}`` factor has unit magnitude for
        a real wavenumber).

        :param characteristic_impedance: Characteristic impedance ``rho c``.
        :return: Transmission loss ``TLn``, in decibels.
        """
        if characteristic_impedance <= 0.0:
            raise ValueError("'characteristic_impedance' must be positive.")
        rc = characteristic_impedance
        combo = self.t11 + self.t12 / rc + rc * self.t21 + self.t22
        return np.asarray(20.0 * np.log10(np.abs(combo) / 2.0), dtype=np.float64)

    def reflection_hard_backed(self, characteristic_impedance: float) -> Complex:
        """Hard-backed reflection coefficient (ASTM E2611-19, Eq. (27)).

        ``R = (T11 - rho c T21) / (T11 + rho c T21)``.

        :param characteristic_impedance: Characteristic impedance ``rho c``.
        :return: Complex reflection coefficient ``R``.
        """
        if characteristic_impedance <= 0.0:
            raise ValueError("'characteristic_impedance' must be positive.")
        rc = characteristic_impedance
        return np.asarray(
            (self.t11 - rc * self.t21) / (self.t11 + rc * self.t21),
            dtype=np.complex128,
        )

    def absorption_hard_backed(self, characteristic_impedance: float) -> Real:
        """Hard-backed absorption coefficient (ASTM E2611-19, Eq. (28)).

        ``alpha = 1 - |R|^2``.

        :param characteristic_impedance: Characteristic impedance ``rho c``.
        :return: Absorption coefficient ``alpha``.
        """
        r = self.reflection_hard_backed(characteristic_impedance)
        return np.asarray(1.0 - np.abs(r) ** 2, dtype=np.float64)

    def material_wavenumber(self, thickness: float) -> Complex:
        """Propagation wavenumber inside the material (ASTM E2611-19, Eq. (29)).

        ``k' = arccos(T11) / d`` (complex ``arccos``).

        :param thickness: Specimen thickness ``d``, in metres.
        :return: Complex material wavenumber ``k'``, in reciprocal metres.
        """
        if thickness <= 0.0:
            raise ValueError("'thickness' must be positive.")
        t11 = np.asarray(self.t11, dtype=np.complex128)
        return np.asarray(np.arccos(t11) / thickness, dtype=np.complex128)

    def characteristic_impedance_material(self) -> Complex:
        """Characteristic impedance of the material (ASTM E2611-19, Eq. (30)).

        ``Z = sqrt(T12 / T21)``.

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
        0..1 right axis). The matrix itself carries no frequency axis, so the
        ``frequency`` vector of the measurement (matching the shape of the
        entries) and the air characteristic impedance ``rho c`` are needed:
        both default to the values retained from the solver call
        (``self.frequency`` / ``self.air_characteristic_impedance``) and only
        have to be supplied for a hand-built matrix.

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
        from .._i18n import check_language
        from .._plot.materials import plot_transfer_matrix

        check_language(language)
        return plot_transfer_matrix(
            self, frequency, characteristic_impedance, ax=ax,
            language=language, **kwargs,
        )


def air_layer_transfer_matrix(
    wavenumber: ArrayLike, thickness: float, characteristic_impedance: float
) -> TransferMatrix:
    """Analytic transfer matrix of a pure air layer of thickness ``d``.

    ``T = [[cos(k d), j rho c sin(k d)], [j sin(k d) / (rho c), cos(k d)]]`` -
    the classical loss-free layer used to validate the ASTM E2611-19 reduction
    (it is reciprocal, ``det(T) = 1``, and symmetric, ``T11 = T22``).

    :param wavenumber: Air wavenumber ``k``.
    :param thickness: Layer thickness ``d``, in metres.
    :param characteristic_impedance: Characteristic impedance ``rho c``, in rayls.
    :return: The air-layer :class:`TransferMatrix`.
    """
    if characteristic_impedance <= 0.0:
        raise ValueError("'characteristic_impedance' must be positive.")
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
        load[0], load[1], load[2], load[3],
        l1=l1, s1=s1, l2=l2, s2=s2, wavenumber=wavenumber,
    )
    return face_quantities(
        a, b, c, d,
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
            np.asarray(frequency, dtype=np.float64)
            if frequency is not None else None
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
    """Two-load transfer matrix (ASTM E2611-19, Eqs. (17)-(22)).

    Each load is the tuple ``(H1, H2, H3, H4)`` of the four microphone transfer
    functions measured with a different downstream termination. The two loads
    give four equations for the four unknowns (Eq. (22))::

        DEN = p_da u_db - p_db u_da
        T11 = (p0a u_db - p0b u_da) / DEN
        T12 = (p0b p_da - p0a p_db) / DEN
        T21 = (u0a u_db - u0b u_da) / DEN
        T22 = (p_da u0b - p_db u0a) / DEN

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
            wavenumber, s1=s1, s2=s2, diameter=diameter, shape=canonical,
            stacklevel=2,
        )
    p0a, pda, u0a, uda = _face_from_loads(
        load_a, l1=l1, s1=s1, l2=l2, s2=s2, thickness=thickness,
        wavenumber=wavenumber, characteristic_impedance=characteristic_impedance,
    )
    p0b, pdb, u0b, udb = _face_from_loads(
        load_b, l1=l1, s1=s1, l2=l2, s2=s2, thickness=thickness,
        wavenumber=wavenumber, characteristic_impedance=characteristic_impedance,
    )
    den = pda * udb - pdb * uda
    _warn_ill_conditioned(
        np.asarray(den, dtype=np.complex128),
        np.abs(pda * udb) + np.abs(pdb * uda),
        "transfer_matrix_two_load", stacklevel=2,
    )
    return TransferMatrix(
        t11=np.asarray((p0a * udb - p0b * uda) / den, dtype=np.complex128),
        t12=np.asarray((p0b * pda - p0a * pdb) / den, dtype=np.complex128),
        t21=np.asarray((u0a * udb - u0b * uda) / den, dtype=np.complex128),
        t22=np.asarray((pda * u0b - pdb * u0a) / den, dtype=np.complex128),
        **_measurement_context(
            l1=l1, s1=s1, l2=l2, s2=s2, thickness=thickness,
            diameter=diameter, shape=canonical, frequency=frequency,
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
    """One-load transfer matrix, symmetric specimen (ASTM E2611-19, Eqs. (23)-(24)).

    Valid only for a reciprocal **and** symmetric specimen (``T11 = T22`` and
    ``T11 T22 - T12 T21 = 1``, Eq. (23)). A single termination suffices::

        DEN = p0 ud + pd u0
        T11 = T22 = (pd ud + p0 u0) / DEN
        T12 = (p0^2 - pd^2) / DEN
        T21 = (u0^2 - ud^2) / DEN

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
            wavenumber, s1=s1, s2=s2, diameter=diameter, shape=canonical,
            stacklevel=2,
        )
    p0, pd, u0, ud = _face_from_loads(
        load, l1=l1, s1=s1, l2=l2, s2=s2, thickness=thickness,
        wavenumber=wavenumber, characteristic_impedance=characteristic_impedance,
    )
    den = p0 * ud + pd * u0
    _warn_ill_conditioned(
        np.asarray(den, dtype=np.complex128),
        np.abs(p0 * ud) + np.abs(pd * u0),
        "transfer_matrix_one_load", stacklevel=2,
    )
    t_diag = np.asarray((pd * ud + p0 * u0) / den, dtype=np.complex128)
    return TransferMatrix(
        t11=t_diag,
        t12=np.asarray((p0**2 - pd**2) / den, dtype=np.complex128),
        t21=np.asarray((u0**2 - ud**2) / den, dtype=np.complex128),
        t22=t_diag,
        **_measurement_context(
            l1=l1, s1=s1, l2=l2, s2=s2, thickness=thickness,
            diameter=diameter, shape=canonical, frequency=frequency,
            characteristic_impedance=characteristic_impedance,
        ),
    )


def characteristic_impedance(density: float, speed_of_sound: float) -> float:
    """Characteristic impedance of air ``rho c`` (rayls).

    A convenience for both standards (ISO 10534-2 Clause 7.2; ASTM E2611-19
    Clause 8.2/8.3): the real product of air density and speed of sound.

    :param density: Air density ``rho``, in kg/m3.
    :param speed_of_sound: Speed of sound ``c``, in m/s.
    :return: Characteristic impedance ``rho c``, in rayls.
    """
    if density <= 0.0 or speed_of_sound <= 0.0:
        raise ValueError("'density' and 'speed_of_sound' must be positive.")
    return float(density * speed_of_sound)
