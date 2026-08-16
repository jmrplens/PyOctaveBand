#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Two-microphone transfer-function method in the impedance tube.

**BS EN ISO 10534-2:2001**: the complex reflection factor ``r`` at the sample
surface is obtained from the measured transfer function ``H12`` between two
microphones flush-mounted in the wall of a tube terminated by the specimen,
and from it the surface impedance and the normal-incidence absorption
coefficient (Clause 7, Eqs. (17)-(20)). Time convention
:math:`e^{+j\omega t}`; the incident wave carries :math:`e^{+jk_0x}` and the
reflected wave :math:`e^{-jk_0x}` (Annex D, Eqs. (D.1)-(D.8)). The complex
wavenumber is :math:`k_0 = k_0' - jk_0''` with the attenuation constant
:math:`k_0''` (Clause 2.6, Annex A). Air properties from Clause 7.2,
Eqs. (5)/(7), use temperature in **kelvin**.

The tube itself is described here as well - its cross-section, the hydraulic
diameter a rectangular tube reports, the complex wavenumber, the lower-bound
wall attenuation and the plane-wave working range - because a specimen is
characterised only where the field in the tube is a plane wave, and because
the four-microphone method measures in the same tube and reuses that
arithmetic.

The other two standardised impedance-tube methods are their own modules, each
kept in its own sign convention (they are **not** interchangeable):
:mod:`~phonometry.materials.absorbers.standing_wave` for the probe-traverse
standing-wave-ratio method of BS EN ISO 10534-1:2001, and
:mod:`~phonometry.materials.absorbers.four_microphone` for the transmission
transfer-matrix method of ASTM E2611-19, whose air properties are given in
degrees Celsius and whose forward wave carries the opposite exponent sign.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._internal.types import Real
from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

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

#: Upper-frequency plane-wave factor, circular tube (ISO 10534-2 Eq. (2)).
_ISO_KU_CIRCULAR = 0.58
#: Upper-frequency plane-wave factor, rectangular tube (ISO 10534-2 Eq. (3)).
_ISO_KU_RECTANGULAR = 0.50
#: Microphone-spacing factor for the upper limit (ISO 10534-2 Eq. (4)).
_ISO_KU_SPACING = 0.45
#: Lower-limit factor: spacing recommended > 5 % of the wavelength (Clause 4.2),
#: i.e. ``f_l = c0 / (20 s)``.
_ISO_LOWER_WAVELENGTH_FRACTION = 20.0

#: Shared validation message for the tube diameter arguments.
_DIAMETER_POSITIVE = "'diameter' must be positive."

#: Shared validation message for the speed-of-sound arguments.
_SPEED_OF_SOUND_POSITIVE = "'speed_of_sound' must be positive."

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
    "absorption_from_reflection",
    "air_density_iso",
    "apply_mic_calibration",
    "characteristic_impedance",
    "hydraulic_diameter",
    "mic_calibration_factor",
    "normalized_surface_admittance",
    "normalized_surface_impedance",
    "plane_wave_frequency_range",
    "reflection_factor",
    "speed_of_sound_iso",
    "surface_impedance",
    "tube_attenuation_constant",
    "tube_wavenumber",
    "two_microphone_impedance",
]


class ImpedanceTubeWarning(PhonometryWarning):
    """Advisory for out-of-plane-wave-range impedance-tube frequencies."""


# ---------------------------------------------------------------------------
# Air properties (ISO 10534-2 works in kelvin; the ASTM E2611-19 pair, in
# degrees Celsius, lives with the four-microphone method).
# ---------------------------------------------------------------------------
def speed_of_sound_iso(temperature: ArrayLike) -> Real:
    r"""Speed of sound in air (ISO 10534-2:2001, Eq. (5)).

    :math:`c_0 = 343.2 \sqrt{T / 293}`.

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
    r"""Air density (ISO 10534-2:2001, Eq. (7)).

    :math:`\rho = \rho_0 (p_\mathrm{a} T_0) / (p_0 T)` with :math:`\rho_0 = 1.186`
    kg/m3, :math:`T_0 = 293` K and :math:`p_0 = 101.325` kPa.

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
    r"""Hydraulic diameter of a rectangular tube, :math:`4A/P` (ISO 10534-2, A.2.1.5).

    For a rectangular cross-section of side lengths ``w`` and ``h`` the ratio
    of four times the area to the perimeter reduces to
    :math:`d_\mathrm{h} = 2wh/(w + h)`; a square tube gives ``d_h`` equal to the side
    length. This is the ``d`` the Eq. (A.18) attenuation estimate expects for
    rectangular tubes (see :func:`tube_attenuation_constant`).

    :param width: Inner side length ``w``, in metres.
    :param height: Inner side length ``h``, in metres.
    :return: Hydraulic diameter :math:`d_\mathrm{h} = 4A/P`, in metres.
    """
    if width <= 0.0 or height <= 0.0:
        raise ValueError("'width' and 'height' must be positive.")
    return float(2.0 * width * height / (width + height))


def tube_attenuation_constant(
    frequency: ArrayLike, speed_of_sound: float, diameter: float
) -> Real:
    r"""Lower-bound tube attenuation constant ``k0''`` (ISO 10534-2, Eq. (A.18)).

    :math:`k_0'' = 1.94\times 10^{-2} \sqrt{f} / (c_0 d)`
    (nepers per metre). This ignores
    porous-wall and object losses and is therefore a lower limit (Clause A.2.1.5).

    :param frequency: Frequency ``f``, in hertz (scalar or per band).
    :param speed_of_sound: Speed of sound ``c0``, in metres per second.
    :param diameter: Circular-tube diameter ``d``, in metres, or the hydraulic
        diameter ``4 * area / perimeter`` for a rectangular tube (see
        :func:`hydraulic_diameter`).
    :return: Attenuation constant ``k0''``, in nepers per metre.
    """
    if speed_of_sound <= 0.0:
        raise ValueError(_SPEED_OF_SOUND_POSITIVE)
    if diameter <= 0.0:
        raise ValueError(_DIAMETER_POSITIVE)
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
    r"""Complex wavenumber :math:`k_0 = k_0' - jk_0''` (ISO 10534-2, Clause 2.6).

    The real part is :math:`k_0' = 2\pi f/c_0` (Eq. (2)); the optional attenuation
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
        raise ValueError(_SPEED_OF_SOUND_POSITIVE)
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
    r"""Complex reflection factor at the sample surface (ISO 10534-2, Eq. (17)).

    .. math::

       r = \frac{H_{12} - H_\mathrm{I}}{H_\mathrm{R} - H_{12}} \, e^{+2jk_0x_1}

    with the incident- and reflected-wave transfer functions
    :math:`H_\mathrm{I} = e^{-jk_0s}` (Eq. (D.5)) and :math:`H_\mathrm{R} = e^{+jk_0s}`
    (Eq. (D.6)), ``s`` the microphone spacing and ``x1`` the distance from
    the sample to the **farther** microphone (Clause 7.7).

    :param h12: Measured transfer function ``H12`` between microphone
        positions 1 and 2 (Clause 7.6, Eq. (14)); complex, scalar or per band.
        It must already be corrected for microphone mismatch (see
        :func:`apply_mic_calibration`).
    :param spacing: Microphone spacing :math:`s = x_1 - x_2`, in metres.
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
    r"""Normalised surface impedance :math:`Z/(\rho c_0)` (ISO 10534-2, Eq. (19)).

    :math:`Z / (\rho c_0) = (1 + r) / (1 - r)`.

    :param reflection: Complex reflection factor ``r``.
    :return: Normalised surface impedance :math:`Z/(\rho c_0)` (complex).
    """
    r = np.asarray(reflection, dtype=np.complex128)
    return np.asarray((1.0 + r) / (1.0 - r), dtype=np.complex128)


def surface_impedance(
    reflection: ArrayLike, characteristic_impedance: float
) -> Complex:
    r"""Absolute surface impedance ``Z`` (ISO 10534-2, Eq. (19)).

    :math:`Z = \rho c_0 (1 + r) / (1 - r)`.

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
    r"""Normalised surface admittance :math:`G \rho c_0` (ISO 10534-2, Eq. (20)).

    :math:`G \rho c_0 = (\rho c_0) / Z = (1 - r) / (1 + r)`.

    :param reflection: Complex reflection factor ``r``.
    :return: Normalised surface admittance (complex).
    """
    r = np.asarray(reflection, dtype=np.complex128)
    return np.asarray((1.0 - r) / (1.0 + r), dtype=np.complex128)


def absorption_from_reflection(reflection: ArrayLike) -> Real:
    r"""Normal-incidence absorption coefficient (ISO 10534-2, Eq. (18)).

    :math:`\alpha = 1 - |r|^2`. This form is shared with ISO 10534-1 Eq. (9) and
    ASTM E2611-19 Eq. (28).

    :param reflection: Complex reflection factor ``r``.
    :return: Absorption coefficient ``alpha`` (real).
    """
    r = np.asarray(reflection, dtype=np.complex128)
    return np.asarray(1.0 - np.abs(r) ** 2, dtype=np.float64)


def mic_calibration_factor(
    h12_config1: ArrayLike, h12_config2: ArrayLike
) -> Complex:
    r"""Microphone-mismatch calibration factor ``Hc`` (ISO 10534-2, Eq. (10)).

    :math:`H_\mathrm{c} = \sqrt{H_{12}^{I} / H_{12}^{II}}` from a transfer function measured on an
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
    r"""Apply the microphone calibration factor (ISO 10534-2, Eq. (13)).

    :math:`H_{12} = H_{12,\text{uncorrected}} / H_\mathrm{c}`.

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
    r"""Working plane-wave frequency range ``(f_l, f_u)`` (ISO 10534-2, 4.2-4.5).

    The upper limit is the smaller of the microphone-spacing bound
    :math:`f_\mathrm{u} s < 0.45 c_0` (Eq. (4)) and, when the tube ``diameter`` is
    given, the cut-on bound :math:`f_\mathrm{u} d < 0.58 c_0` for a circular tube
    (Eq. (2)) or :math:`f_\mathrm{u} d < 0.50 c_0` for a rectangular tube (Eq. (3)).
    The lower limit uses the Clause 4.2 guideline that the spacing exceed
    5 % of the wavelength, i.e. :math:`f_\mathrm{l} = c_0 / (20 s)`.

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
        raise ValueError(_SPEED_OF_SOUND_POSITIVE)
    canonical = _canonical_shape(shape)
    f_upper = ku_spacing * speed_of_sound / spacing
    if diameter is not None:
        if diameter <= 0.0:
            raise ValueError(_DIAMETER_POSITIVE)
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
    r"""Two-microphone impedance-tube result (ISO 10534-2:2001).

    All arrays share the shape of ``frequency``. ``reflection`` is the complex
    reflection factor ``r`` at the sample surface (Eq. (17)),
    ``surface_impedance`` the absolute surface impedance ``Z`` in rayls
    (Eq. (19)), ``normalized_impedance`` the ratio :math:`Z/(\rho c_0)`
    (Eq. (19)) and ``absorption`` the normal-incidence coefficient
    :math:`\alpha = 1 - \lvert r\rvert^2` (Eq. (18)).

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
        from ..._i18n import check_language
        from ..._plot.materials import plot_impedance_tube

        check_language(language)
        return plot_impedance_tube(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the two-microphone tube to scale (dimensioned side view).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its tube geometry
            (``spacing``/``x1``).
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_impedance_tube_result_geometry

        check_language(language)
        return plot_impedance_tube_result_geometry(
            self, ax=ax, language=language, **kwargs
        )

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        r"""Render an ISO 10534-2 impedance-tube test-report fiche to a PDF.

        Writes a one-page accredited normal-incidence report (BS EN ISO
        10534-2:2001, two-microphone transfer-function method): the
        standard-basis line, an optional metadata header block (client,
        specimen, tube diameter ``d``, microphone spacing ``s``, the measured
        frequency range, mounting, climate ...), a two-panel body with the
        per-frequency table (frequency, absorption ``alpha`` and the
        real/imaginary parts of the normalised surface impedance
        :math:`z = Z/(\rho c_0)`) beside the ``alpha(f)`` curve, and a footer with
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
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso10534 import render_iso10534_report

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
