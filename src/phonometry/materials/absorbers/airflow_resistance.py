#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Airflow resistance of porous materials: ISO 9053-1 and ISO 9053-2.

Two standardised measurement methods share the same three quantities and units
(ISO 9053-1:2018, Clause 3; ISO 9053-2:2020, Clause 3):

- **Airflow resistance** :math:`R = \Delta p / q_v` in Pa*s/m3, with
  :math:`\Delta p` the air pressure difference across the specimen (Pa) and
  ``q_v`` the volumetric airflow rate through it (m3/s)
  (ISO 9053-1:2018, 3.1).
- **Specific airflow resistance** :math:`R_\mathrm{s} = R\,A` in **Pa*s/m** (not
  Pa*s/m2), with ``A`` the cross-sectional area of the specimen perpendicular
  to the flow (m2) (ISO 9053-1:2018, 3.2). Equivalently
  :math:`R_\mathrm{s} = \Delta p / u` with ``u`` the linear airflow velocity, since
  :math:`u = q_v / A`.
- **Airflow resistivity** :math:`\sigma = R_\mathrm{s} / d` in Pa*s/m2, with ``d`` the
  specimen thickness in the flow direction (m), for homogeneous materials
  (ISO 9053-1:2018, 3.3). Equivalently :math:`\sigma = R\,A / d`.

The linear airflow velocity is :math:`u = q_v / A` (ISO 9053-1:2018, 3.4).

**Static (DC) method, ISO 9053-1:2018.** A steady unidirectional flow in the
laminar regime is used. The recommended reference linear airflow velocity is
:math:`u = 0.5\times 10^{-3}` m/s (0.5 mm/s, clause 7.5); if measured stepwise
the highest velocity shall not exceed ``15e-3 m/s`` (15 mm/s), beyond which
the flow may be non-linear. When measured stepwise the pressure difference is
plotted against ``u`` and fitted with a regression of at least second order
constrained through the origin, :math:`\Delta p = a u + b u^2`;
:math:`\Delta p` and ``R_s`` are then evaluated at
:math:`u = 0.5\times 10^{-3}` m/s (clause 7.5). Because
:math:`R_\mathrm{s} = \Delta p / u = a + b u`, the linear coefficient ``a`` is the
zero-velocity specific airflow resistance.

**Alternating (AC) method, ISO 9053-2:2020.** A sinusoidally moving piston
(frequency 1 Hz to 4 Hz, typically 2 Hz; clause 6.2) drives an alternating volume
flow into an air cavity terminated either by the specimen or by an airtight
termination. The airflow resistance follows from the sound-pressure-level
difference between the two terminations (ISO 9053-2:2020, Formula (2), 8.7):

.. math::

   R = \frac{\kappa' P_\mathrm{S}}{2\pi f V} \, \frac{h_\mathrm{t}}{h_\mathrm{s}} \,
   10^{(L_{p\mathrm{s}} - L_{p\mathrm{t}})/20}

with ``kappa'`` the effective ratio of specific heats for air (Annex A),
``P_S`` the static (atmospheric) pressure (Pa), ``f`` the piston frequency (Hz),
``V`` the cavity volume with the airtight termination (m3), ``h_t``/``h_s`` the
piston stroke amplitudes with the airtight termination / specimen cell, and
``L_ps``/``L_pt`` the cavity sound pressure levels with the specimen /
airtight termination (dB). Only the level *difference* enters, so the sound level
device needs no absolute calibration (clause 8.7). The RMS piston volume flow is
:math:`q_v = 2\pi f h A_\mathrm{P}` (ISO 9053-2:2020, 6.2), with ``h`` the stroke
amplitude and ``A_P`` the piston cross-sectional area.

The **effective** ratio of specific heats ``kappa'`` accounts for heat conduction
between the oscillating air and the cavity walls, which makes the compression not
fully adiabatic. ISO 9053-2:2020 Annex A (normative) gives its evaluation from the
cavity geometry and air properties (:func:`effective_kappa`, Formula (A.7)); the
Annex A.3 worked example yields :math:`\kappa' = 1.370` (about 2 % below the
adiabatic :math:`\kappa = 1.4008`). When no cavity/air data are supplied,
:func:`alternating_airflow_resistance` falls back to the **uncorrected adiabatic**
value :math:`\kappa = 1.4` (Formula (A.1)); for a conforming result compute
``kappa'`` per Annex A and pass it explicitly.

Neither part defines a temperature/atmospheric normalisation of the result.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from ..._report.metadata import ReportMetadata

__all__ = [
    "AirflowResistanceWarning",
    "StaticAirflowResult",
    "airflow_resistance",
    "airflow_resistivity",
    "alternating_airflow_resistance",
    "effective_kappa",
    "linear_airflow_velocity",
    "piston_volume_flow_rate",
    "specific_airflow_resistance",
    "static_airflow_resistance",
    "thermal_boundary_layer_thickness",
]

#: Standard atmospheric (static) pressure ``P_S`` (Pa), ISO 9053-2:2020 Formula (2).
_STANDARD_STATIC_PRESSURE = 101325.0
#: Uncorrected adiabatic ratio of specific heats for air; the fallback used for
#: ``kappa'`` when no cavity/air data are supplied. NOT the Annex A output: Annex A
#: applies a heat-conduction correction that lowers it (worked example ``kappa' = 1.370``).
_ADIABATIC_KAPPA = 1.4
#: Reference air properties from IEC 61094-2:2009 (23 C, 101.325 kPa, 50 % RH) as used
#: in the ISO 9053-2:2020 Annex A.3 worked example; defaults of :func:`effective_kappa`.
_ANNEX_A_SPEED_OF_SOUND = 345.9  # c0 (m/s)
_ANNEX_A_AIR_DENSITY = 1.186  # rho0 (kg/m3)
_ANNEX_A_HEAT_RATIO = 1.4008  # kappa, adiabatic (ISO 9053-2:2020 Annex A.3)
_ANNEX_A_THERMAL_CONDUCTIVITY = 0.02355  # k_a (J/(s*m*K))
_ANNEX_A_SPECIFIC_HEAT_CP = 938.7  # C_P (J/(kg*K))
#: Reference linear airflow velocity, ISO 9053-1:2018 clause 7.5 (m/s).
_STATIC_REFERENCE_VELOCITY = 0.5e-3
#: Upper linear-velocity limit of the static method, ISO 9053-1:2018 clause 7.5 (m/s).
_STATIC_MAX_VELOCITY = 15.0e-3
#: Recommended specimen flow-velocity range, ISO 9053-2:2020 clause 6.2 (m/s).
_ALT_VELOCITY_RANGE = (0.5e-3, 4.0e-3)
#: Piston frequency range, ISO 9053-2:2020 clause 6.2 (Hz).
_ALT_FREQUENCY_RANGE = (1.0, 4.0)
#: Upper bound of the validity criterion, ISO 9053-2:2020 Formula (3).
_ALT_VALIDITY_LIMIT = 0.3
#: Required specimen-to-background level margin, ISO 9053-2:2020 Formula (4) (dB).
_ALT_BACKGROUND_MARGIN = 10.0

#: Validation messages shared by the entry points that take the same quantity.
_AREA_POSITIVE_MSG = "'area' must be positive."
_FREQUENCY_POSITIVE_MSG = "'frequency' must be positive."


class AirflowResistanceWarning(PhonometryWarning):
    """Advisory for out-of-range or non-conforming ISO 9053 airflow inputs."""


@dataclass(frozen=True)
class StaticAirflowResult:
    r"""Result of an ISO 9053-1:2018 stepwise (static-method) determination.

    ``resistance`` (``R``, Pa*s/m3), ``specific_resistance`` (``R_s``, Pa*s/m) and
    ``resistivity`` (``sigma``, Pa*s/m2; ``None`` when no thickness is supplied)
    are evaluated at ``evaluation_velocity`` (m/s, the ISO 9053-1 clause 7.5
    reference 0.5 mm/s by default). ``linear_coefficient`` (``a``) and
    ``quadratic_coefficient`` (``b``) are the through-origin fit
    :math:`\Delta p = a u + b u^2` (clause 7.5); ``a`` is the zero-velocity
    specific airflow resistance (Pa*s/m). ``pressure_drop`` is the fitted
    :math:`\Delta p` at ``evaluation_velocity`` (Pa).
    """

    resistance: float
    specific_resistance: float
    resistivity: float | None
    evaluation_velocity: float
    pressure_drop: float
    linear_coefficient: float
    quadratic_coefficient: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the fitted ``dp(u)`` curve with the evaluation point.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_static_airflow

        check_language(language)
        return plot_static_airflow(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        r"""Render an ISO 9053-1 static airflow-resistance test-report fiche to a PDF.

        Writes a one-page accredited airflow-resistance report
        (ISO 9053-1:2018, static/direct airflow method): the standard-basis
        line, an optional metadata header block (client, manufacturer,
        specimen, the specimen thickness ``d``, test facility, date, climate
        ...), a two-panel body with a compact metrics table (the evaluation
        velocity, the fitted pressure difference ``dp``, the airflow resistance
        ``R``, the specific airflow resistance ``R_s``, the airflow resistivity
        ``sigma`` when a thickness is available, and the through-origin fit
        coefficients ``a`` and ``b``) beside the fitted ``dp(u)`` curve, a boxed
        specific airflow resistance ``R_s`` with the airflow resistance ``R``
        and the resistivity ``sigma`` alongside, and a footer with the fixed
        disclaimer. ISO 9053-1 is a material characterisation, so there is no
        pass/fail verdict.

        The clause 7.5 stepwise procedure fits :math:`\Delta p = a u + b u^2`
        through the origin and evaluates the resistances at the reference
        velocity :math:`u = 0.5` mm/s; the linear coefficient ``a`` is the
        zero-velocity
        specific airflow resistance. Resistance quantities are printed to the
        nearest whole Pa*s unit and the evaluation velocity to one decimal
        place (mm/s).

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche. The applicable descriptive
            fields are ``client``, ``manufacturer``, ``specimen``, ``thickness``
            (the specimen thickness ``d``, in metres, shown in millimetres),
            ``test_room``, ``test_date``, ``temperature``, ``relative_humidity``,
            ``measurement_standard``, ``laboratory``, ``operator``, ``report_id``
            and ``notes``. The ``requirement`` field is ignored (ISO 9053-1 has
            no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            airflow-resistance fiche has a single body layout, so it has no
            effect.
        :param language: Fiche language: ``"en"`` (default, English, decimal
            point) or ``"es"`` (Spanish, decimal comma).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab or matplotlib is not installed. The
            fiche always embeds the fitted ``dp(u)`` curve, so both are required
            (``pip install "phonometry[report,plot]"``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso9053 import render_static_airflow_report

        return render_static_airflow_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def linear_airflow_velocity(volume_flow_rate: float, area: float) -> float:
    r"""Linear airflow velocity :math:`u = q_v / A` (ISO 9053-1:2018, 3.4).

    ``volume_flow_rate`` is ``q_v`` (m3/s) and ``area`` is ``A`` (m2); returns
    ``u`` in m/s.
    """
    if volume_flow_rate < 0.0:
        raise ValueError("'volume_flow_rate' must be non-negative.")
    if area <= 0.0:
        raise ValueError(_AREA_POSITIVE_MSG)
    return volume_flow_rate / area


def airflow_resistance(pressure_drop: float, volume_flow_rate: float) -> float:
    r"""Airflow resistance :math:`R = \Delta p / q_v` (ISO 9053-1:2018, 3.1).

    ``pressure_drop`` is the pressure difference :math:`\Delta p` across the
    specimen (Pa) and ``volume_flow_rate`` is the volumetric airflow rate
    ``q_v`` (m3/s). Returns ``R`` in Pa*s/m3.
    """
    if pressure_drop < 0.0:
        raise ValueError("'pressure_drop' must be non-negative.")
    if volume_flow_rate <= 0.0:
        raise ValueError("'volume_flow_rate' must be positive.")
    return pressure_drop / volume_flow_rate


def specific_airflow_resistance(
    resistance: float | None = None,
    area: float | None = None,
    *,
    pressure_drop: float | None = None,
    velocity: float | None = None,
) -> float:
    r"""Specific airflow resistance ``R_s`` in Pa*s/m (ISO 9053-1:2018, 3.2).

    Two equivalent routes are accepted; supply exactly one:

    - ``resistance`` (``R``, Pa*s/m3) and ``area`` (``A``, m2):
      :math:`R_\mathrm{s} = R\,A`.
    - ``pressure_drop`` (:math:`\Delta p`, Pa) and ``velocity`` (``u``, m/s):
      :math:`R_\mathrm{s} = \Delta p / u` (from :math:`R_\mathrm{s} = R\,A` with
      :math:`u = q_v / A`).

    The unit is pascal second per metre (Pa*s/m), not Pa*s/m2.
    """
    from_resistance = resistance is not None and area is not None
    from_pressure = pressure_drop is not None and velocity is not None
    if from_resistance == from_pressure:
        raise ValueError(
            "Provide exactly one route: ('resistance' and 'area') or "
            "('pressure_drop' and 'velocity')."
        )
    if resistance is not None and area is not None:
        if resistance < 0.0:
            raise ValueError("'resistance' must be non-negative.")
        if area <= 0.0:
            raise ValueError(_AREA_POSITIVE_MSG)
        return resistance * area
    if pressure_drop is not None and velocity is not None:
        if pressure_drop < 0.0:
            raise ValueError("'pressure_drop' must be non-negative.")
        if velocity <= 0.0:
            raise ValueError("'velocity' must be positive.")
        return pressure_drop / velocity
    raise ValueError(  # pragma: no cover - unreachable, guarded above
        "Provide exactly one route: ('resistance' and 'area') or "
        "('pressure_drop' and 'velocity')."
    )


def airflow_resistivity(specific_resistance: float, thickness: float) -> float:
    r"""Airflow resistivity :math:`\sigma = R_\mathrm{s} / d` (ISO 9053-1:2018, 3.3).

    ``specific_resistance`` is ``R_s`` (Pa*s/m) and ``thickness`` is ``d`` (m),
    the specimen thickness in the flow direction. Returns ``sigma`` in Pa*s/m2.
    """
    if specific_resistance < 0.0:
        raise ValueError("'specific_resistance' must be non-negative.")
    if thickness <= 0.0:
        raise ValueError("'thickness' must be positive.")
    return specific_resistance / thickness


def _warn_static_velocity_range(
    velocities: NDArray[np.float64], stacklevel: int
) -> None:
    """Advise when a stepwise velocity exceeds the ISO 9053-1 clause 7.5 limit."""
    top = float(np.max(velocities))
    if top > _STATIC_MAX_VELOCITY:
        warnings.warn(
            f"Highest linear airflow velocity {top:g} m/s exceeds the "
            f"{_STATIC_MAX_VELOCITY:g} m/s limit of ISO 9053-1:2018 clause 7.5; "
            "the flow may be non-linear and the result is advisory.",
            AirflowResistanceWarning,
            stacklevel=stacklevel,
        )


def static_airflow_resistance(
    velocities: ArrayLike,
    pressure_drops: ArrayLike,
    area: float,
    thickness: float | None = None,
    *,
    evaluation_velocity: float = _STATIC_REFERENCE_VELOCITY,
) -> StaticAirflowResult:
    r"""Stepwise static-method airflow resistance (ISO 9053-1:2018, clause 7.5).

    Fits the measured pressure difference against the linear airflow velocity
    with a second-order regression constrained through the origin,
    :math:`\Delta p = a u + b u^2`, and evaluates the resistances at
    ``evaluation_velocity`` (the clause 7.5 reference ``0.5e-3 m/s`` by default).

    ``velocities`` are the linear airflow velocities ``u`` (m/s) and
    ``pressure_drops`` the matching pressure differences :math:`\Delta p` (Pa)
    of at least two measurement steps; ``area`` is the cross-section ``A`` (m2)
    and ``thickness`` the specimen thickness ``d`` (m, optional, enabling
    ``sigma``).

    Because :math:`R_\mathrm{s} = \Delta p / u = a + b u`, the returned
    ``linear_coefficient`` ``a`` is the zero-velocity specific airflow
    resistance. A velocity above the clause 7.5 upper limit (15 mm/s) raises
    :class:`AirflowResistanceWarning`.
    """
    u = np.asarray(velocities, dtype=np.float64)
    dp = np.asarray(pressure_drops, dtype=np.float64)
    if u.ndim != 1 or dp.ndim != 1:
        raise ValueError("'velocities' and 'pressure_drops' must be 1-D.")
    if u.shape != dp.shape:
        raise ValueError("'velocities' and 'pressure_drops' must have equal length.")
    if u.size < 2:
        raise ValueError("At least two measurement steps are required.")
    if bool(np.any(u <= 0.0)):
        raise ValueError("All velocities must be positive.")
    if bool(np.any(dp < 0.0)):
        raise ValueError("All pressure drops must be non-negative.")
    if area <= 0.0:
        raise ValueError(_AREA_POSITIVE_MSG)
    if thickness is not None and thickness <= 0.0:
        raise ValueError("'thickness' must be positive.")
    if evaluation_velocity <= 0.0:
        raise ValueError("'evaluation_velocity' must be positive.")

    _warn_static_velocity_range(u, stacklevel=2)

    # Through-origin second-order fit: dp = a*u + b*u**2 (no constant term).
    design = np.stack([u, u**2], axis=1)
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(design, dp, rcond=None)
    a = float(coeffs[0])
    b = float(coeffs[1])

    dp_eval = a * evaluation_velocity + b * evaluation_velocity**2
    # The specific resistance R_s is dp/u, which for a through-origin fit is
    # a + b*u; and since R_s = R*A, dividing it by the area gives R.
    specific = dp_eval / evaluation_velocity
    resistance = specific / area
    resistivity = None if thickness is None else specific / thickness

    return StaticAirflowResult(
        resistance=resistance,
        specific_resistance=specific,
        resistivity=resistivity,
        evaluation_velocity=evaluation_velocity,
        pressure_drop=dp_eval,
        linear_coefficient=a,
        quadratic_coefficient=b,
    )


def piston_volume_flow_rate(
    frequency: float, stroke_amplitude: float, piston_area: float
) -> float:
    r"""RMS piston volume flow :math:`q_v = 2\pi f h A_\mathrm{P}` (ISO 9053-2, 6.2).

    ``frequency`` is the piston frequency ``f`` (Hz), ``stroke_amplitude`` the
    stroke amplitude ``h`` (m) and ``piston_area`` the piston cross-section
    ``A_P`` (m2). Returns ``q_v`` in m3/s.
    """
    if frequency <= 0.0:
        raise ValueError(_FREQUENCY_POSITIVE_MSG)
    if stroke_amplitude < 0.0:
        raise ValueError("'stroke_amplitude' must be non-negative.")
    if piston_area <= 0.0:
        raise ValueError("'piston_area' must be positive.")
    # Normative clause 6.2 form q_v = 2*pi*f*h*A_P verbatim; Annex A.2 prints
    # the rms variant j*omega*A_P*h/sqrt(2) (internal tension in the standard,
    # the normative text wins).
    return 2.0 * math.pi * frequency * stroke_amplitude * piston_area


def thermal_boundary_layer_thickness(
    frequency: float,
    *,
    speed_of_sound: float = _ANNEX_A_SPEED_OF_SOUND,
    air_density: float = _ANNEX_A_AIR_DENSITY,
    specific_heat_cp: float = _ANNEX_A_SPECIFIC_HEAT_CP,
    thermal_conductivity: float = _ANNEX_A_THERMAL_CONDUCTIVITY,
) -> float:
    r"""Thermal boundary-layer thickness ``b`` (ISO 9053-2:2020, Formulae (A.4)/(A.5)).

    .. math::

       l_\mathrm{h} = \frac{k_\mathrm{a}}{\rho_0 c_0 C_\mathrm{P}} \tag{A.5}

       b = \sqrt{\frac{2 c_0 l_\mathrm{h}}{\omega}}, \qquad \omega = 2\pi f \tag{A.4}

    ``frequency`` is the piston frequency ``f`` (Hz); ``speed_of_sound`` ``c0`` (m/s),
    ``air_density`` ``rho0`` (kg/m3), ``specific_heat_cp`` ``C_P`` (J/(kg*K)) and
    ``thermal_conductivity`` ``k_a`` (J/(s*m*K)) are air properties, defaulting to the
    IEC 61094-2:2009 values used in ISO 9053-2:2020 Annex A.3. Returns ``b`` in metres;
    with the Annex A.3 example (:math:`f = 2` Hz) this is ``1.83e-3 m``.
    """
    if frequency <= 0.0:
        raise ValueError(_FREQUENCY_POSITIVE_MSG)
    if speed_of_sound <= 0.0:
        raise ValueError("'speed_of_sound' must be positive.")
    if air_density <= 0.0:
        raise ValueError("'air_density' must be positive.")
    if specific_heat_cp <= 0.0:
        raise ValueError("'specific_heat_cp' must be positive.")
    if thermal_conductivity <= 0.0:
        raise ValueError("'thermal_conductivity' must be positive.")
    omega = 2.0 * math.pi * frequency
    diffusion_length = thermal_conductivity / (
        air_density * speed_of_sound * specific_heat_cp
    )
    return math.sqrt(2.0 * speed_of_sound * diffusion_length / omega)


def effective_kappa(
    cavity_surface: float,
    cavity_volume: float,
    frequency: float,
    *,
    speed_of_sound: float = _ANNEX_A_SPEED_OF_SOUND,
    air_density: float = _ANNEX_A_AIR_DENSITY,
    specific_heat_ratio: float = _ANNEX_A_HEAT_RATIO,
    specific_heat_cp: float = _ANNEX_A_SPECIFIC_HEAT_CP,
    thermal_conductivity: float = _ANNEX_A_THERMAL_CONDUCTIVITY,
) -> float:
    r"""Effective ratio of specific heats ``kappa'`` (ISO 9053-2:2020, Annex A, Formula (A.7)).

    Heat conduction between the oscillating air and the cavity walls makes the
    compression not fully adiabatic, lowering ``kappa`` to:

    .. math::

       \kappa' = \frac{\kappa}{\sqrt{1 + (\kappa - 1)\frac{S}{V} b
       + 0.5 \left( (\kappa - 1)\frac{S}{V} b \right)^{2}}} \tag{A.7}

    with ``b`` the thermal boundary-layer thickness (Formulae (A.4)/(A.5),
    :func:`thermal_boundary_layer_thickness`), ``S`` the total internal surface area
    of the air cavity (m2) and ``V`` its volume (m3).

    ``cavity_surface`` is ``S`` (m2), ``cavity_volume`` ``V`` (m3) and ``frequency``
    the piston frequency ``f`` (Hz); ``specific_heat_ratio`` ``kappa`` (adiabatic) and
    the remaining air properties default to the ISO 9053-2:2020 Annex A.3 values.
    Returns the dimensionless ``kappa'`` for use in
    :func:`alternating_airflow_resistance`; the Annex A.3 worked example
    (:math:`S = 0.0471` m2, :math:`V = 7.854\times 10^{-4}` m3,
    :math:`f = 2` Hz) yields :math:`\kappa' = 1.370`.
    """
    if cavity_surface <= 0.0:
        raise ValueError("'cavity_surface' must be positive.")
    if cavity_volume <= 0.0:
        raise ValueError("'cavity_volume' must be positive.")
    if specific_heat_ratio <= 0.0:
        raise ValueError("'specific_heat_ratio' must be positive.")
    boundary_thickness = thermal_boundary_layer_thickness(
        frequency,
        speed_of_sound=speed_of_sound,
        air_density=air_density,
        specific_heat_cp=specific_heat_cp,
        thermal_conductivity=thermal_conductivity,
    )
    surface_to_volume = cavity_surface / cavity_volume
    term = (specific_heat_ratio - 1.0) * surface_to_volume * boundary_thickness
    return specific_heat_ratio / math.sqrt(1.0 + term + 0.5 * term**2)


def _warn_alternating_validity(
    ratio_term: float,
    level_specimen: float,
    background_level: float | None,
    stacklevel: int,
) -> None:
    """Check the ISO 9053-2:2020 Formula (3)/(4) validity criteria."""
    if ratio_term >= _ALT_VALIDITY_LIMIT:
        warnings.warn(
            f"Validity term (h_t/h_s)*10**((L_ps-L_pt)/20) = {ratio_term:g} is not "
            f"below {_ALT_VALIDITY_LIMIT:g} (ISO 9053-2:2020 Formula (3)); adjust "
            "specimen size, cavity volume, piston frequency or stroke length.",
            AirflowResistanceWarning,
            stacklevel=stacklevel,
        )
    if (
        background_level is not None
        and level_specimen - background_level <= _ALT_BACKGROUND_MARGIN
    ):
        warnings.warn(
            f"Specimen-to-background margin {level_specimen - background_level:g} dB "
            f"is not above {_ALT_BACKGROUND_MARGIN:g} dB (ISO 9053-2:2020 "
            "Formula (4)); background noise may bias the result.",
            AirflowResistanceWarning,
            stacklevel=stacklevel,
        )


def alternating_airflow_resistance(
    level_specimen: float,
    level_termination: float,
    *,
    piston_stroke_specimen: float,
    piston_stroke_termination: float,
    frequency: float,
    cavity_volume: float,
    static_pressure: float = _STANDARD_STATIC_PRESSURE,
    kappa_prime: float = _ADIABATIC_KAPPA,
    background_level: float | None = None,
) -> float:
    r"""Alternating-method airflow resistance (ISO 9053-2:2020, Formula (2), 8.7).

    Implements:

    .. math::

       R = \frac{\kappa' P_\mathrm{S}}{2\pi f V} \, \frac{h_\mathrm{t}}{h_\mathrm{s}} \,
       10^{(L_{p\mathrm{s}} - L_{p\mathrm{t}})/20}

    ``level_specimen`` (``L_ps``) and ``level_termination`` (``L_pt``) are the
    cavity sound pressure levels (dB) with the specimen cell and the airtight
    termination; ``piston_stroke_specimen`` (``h_s``) and
    ``piston_stroke_termination`` (``h_t``) the corresponding stroke amplitudes
    (m); ``frequency`` the piston frequency ``f`` (Hz, 1-4 Hz); ``cavity_volume``
    the airtight-termination cavity volume ``V`` (m3); ``static_pressure`` the
    atmospheric pressure ``P_S`` (Pa, default 101325); ``kappa_prime`` the
    effective ratio of specific heats ``kappa'``; ``background_level`` the optional
    cavity background level ``L_pb`` (dB) for the Formula (4) check. Returns ``R`` in
    Pa*s/m3.

    ``kappa_prime`` defaults to the **uncorrected adiabatic**
    :math:`\kappa = 1.4` (Formula (A.1)). For a result conforming to the
    normative Annex A, compute the heat-conduction-corrected ``kappa'`` with
    :func:`effective_kappa` from the cavity geometry and pass it here (the
    Annex A.3 example gives :math:`\kappa' = 1.370`).

    Emits :class:`AirflowResistanceWarning` when the piston frequency is outside
    1-4 Hz or when the Formula (3)/(4) validity criteria are not met.
    """
    if frequency <= 0.0:
        raise ValueError(_FREQUENCY_POSITIVE_MSG)
    if cavity_volume <= 0.0:
        raise ValueError("'cavity_volume' must be positive.")
    if piston_stroke_specimen <= 0.0:
        raise ValueError("'piston_stroke_specimen' must be positive.")
    if piston_stroke_termination <= 0.0:
        raise ValueError("'piston_stroke_termination' must be positive.")
    if static_pressure <= 0.0:
        raise ValueError("'static_pressure' must be positive.")
    if kappa_prime <= 0.0:
        raise ValueError("'kappa_prime' must be positive.")

    low, high = _ALT_FREQUENCY_RANGE
    if not low <= frequency <= high:
        warnings.warn(
            f"Piston frequency {frequency:g} Hz is outside the ISO 9053-2:2020 "
            f"clause 6.2 range [{low:g}, {high:g}] Hz; the result is advisory.",
            AirflowResistanceWarning,
            stacklevel=2,
        )

    stroke_ratio = piston_stroke_termination / piston_stroke_specimen
    level_factor = float(10.0 ** ((level_specimen - level_termination) / 20.0))
    ratio_term = stroke_ratio * level_factor
    _warn_alternating_validity(
        ratio_term, level_specimen, background_level, stacklevel=2
    )

    prefactor = (
        kappa_prime * static_pressure / (2.0 * math.pi * frequency * cavity_volume)
    )
    return prefactor * ratio_term
