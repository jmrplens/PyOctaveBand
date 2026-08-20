#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Dynamic stiffness of resilient materials under floating floors (EN 29052-1:1992).

A floating floor is a heavy floating slab resting on a resilient layer; the
combination is a mass-spring system whose natural frequency governs the impact
and airborne improvement of the floor. EN 29052-1 (identical to ISO 9052-1:1989)
measures the **dynamic stiffness per unit area** ``s'`` of the resilient layer
from the resonance of a standard load plate on a 200 mm x 200 mm specimen.

The dynamic stiffness per unit area is the ratio of a dynamic force per area to
the resulting change in thickness (Formula 1):

.. math::

   s' = \frac{F/S}{\Delta d} \qquad [\text{N/m}^3]

The resiliently supported floor is a mass-spring resonator; its natural
frequency (Formula 2) and, in the laboratory arrangement, the measured resonant
frequency (Formula 3) are:

.. math::

   f_0 = \frac{1}{2\pi} \sqrt{\frac{s'}{m'}}
   \qquad \text{(installed floor)}

   f_\mathrm{r} = \frac{1}{2\pi} \sqrt{\frac{s'_\mathrm{t}}{m'_\mathrm{t}}}
   \qquad \text{(test arrangement)}

so the *apparent* dynamic stiffness follows from the resonance (Formula 4):

.. math::

   s'_\mathrm{t} = 4 \pi^2 m'_\mathrm{t} f_\mathrm{r}^2

With an air-permeable resilient material the enclosed gas adds a parallel
stiffness (Formula 7), from the isothermal compression of the pore air:

.. math::

   s'_\mathrm{a} = \frac{p_0}{d\,\epsilon}

(:math:`s'_\mathrm{a} = 111/d` MN/m3 for :math:`p_0 = 0.1` MPa,
:math:`\epsilon = 0.9` and ``d`` in mm, the standard's worked NOTE). The
dynamic stiffness of the installed material is then obtained by airflow
resistivity ``r`` (clause 8.2):

.. math::

   s' = s'_\mathrm{t}, \qquad
   r \ge 100~\text{kPa}\cdot\text{s/m}^2 \tag{Formula 5}

   s' = s'_\mathrm{t} + s'_\mathrm{a}, \qquad
   10 \le r < 100~\text{kPa}\cdot\text{s/m}^2 \tag{Formula 6}

For :math:`r < 10` kPa.s/m2, ``s'a`` follows Formula 7; the method only
applies when :math:`s'_\mathrm{t} \gg s'_\mathrm{a}`, otherwise ``s'`` cannot be resolved.

This module is the resilient-layer characterisation feeding the floating-floor
term of the EN 12354-2 impact model
(:mod:`phonometry.building.prediction.simplified_model`). It does **not** feed
ISO 16251-1 (:mod:`phonometry.building.measurement.floor_covering_improvement`), whose
scope is limited to soft, locally-reacting floor coverings; floating floors
are explicitly excluded there.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ..._report.metadata import ReportMetadata


from ..._internal.types import as_float_or_array
from ..._internal.validation import require_positive
from ..._internal.warnings import PhonometryWarning

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

#: Atmospheric pressure ``p0`` used by EN 29052-1 for the enclosed-gas term
#: (clause 8.2 NOTE: ``p0 = 0,1 MPa``), in pascals. The standard rounds one
#: atmosphere to 0,1 MPa; pass the true 101 325 Pa explicitly if preferred.
STANDARD_ATMOSPHERIC_PRESSURE = 1.0e5

#: Airflow-resistivity thresholds of clause 8.2, in kPa.s/m2.
_HIGH_RESISTIVITY = 100.0
_LOW_RESISTIVITY = 10.0

#: ``2*pi`` and ``4*pi**2`` for the resonance relations.
_TWO_PI = 2.0 * np.pi
_FOUR_PI_SQ = 4.0 * np.pi**2


class DynamicStiffnessWarning(PhonometryWarning):
    """Advisory when the enclosed-gas term makes ``s'`` unresolvable (clause 8.2)."""


# ---------------------------------------------------------------------------
# Resonance relations (clauses 3, 8.1).
# ---------------------------------------------------------------------------


def apparent_dynamic_stiffness(
    resonant_frequency: ArrayLike, total_mass_per_area: float
) -> np.ndarray | float:
    r"""Apparent dynamic stiffness per unit area ``s't`` (Formula 4).

    Inverts the test resonance :math:`f_\mathrm{r} = (1/2\pi)\sqrt{s'_\mathrm{t}/m'_\mathrm{t}}` to
    :math:`s'_\mathrm{t} = 4 \pi^2 m'_\mathrm{t} f_\mathrm{r}^2`.

    :param resonant_frequency: Extrapolated resonant frequency ``fr``, in hertz
        (scalar or array).
    :param total_mass_per_area: Total mass per unit area used during the test
        ``m't``, in kg/m2 (the load plate plus fittings over the 0,04 m2
        specimen; the standard's plate gives
        :math:`m'_\mathrm{t} = 8~\text{kg} / 0.04~\text{m}^2 = 200` kg/m2).
    :return: The apparent dynamic stiffness per unit area ``s't``, in N/m3
        (numerically MN/m3 when divided by 1e6).
    """
    total_mass_per_area = require_positive(total_mass_per_area, "total_mass_per_area")
    fr = np.asarray(resonant_frequency, dtype=np.float64)
    if np.any(fr <= 0.0):
        raise ValueError("'resonant_frequency' must be positive.")
    return as_float_or_array(_FOUR_PI_SQ * total_mass_per_area * fr**2)


def enclosed_gas_stiffness(
    thickness: ArrayLike,
    porosity: float,
    *,
    atmospheric_pressure: float = STANDARD_ATMOSPHERIC_PRESSURE,
) -> np.ndarray | float:
    r"""Enclosed-gas dynamic stiffness per unit area ``s'a`` (Formula 7).

    The isothermal compression of the pore air adds a stiffness in parallel
    with the material's structure: :math:`s'_\mathrm{a} = p_0 / (d\,\epsilon)`.

    :param thickness: Thickness ``d`` of the specimen under the static load, in
        **metres** (scalar or array).
    :param porosity: Porosity ``epsilon`` of the specimen (0-1).
    :param atmospheric_pressure: Atmospheric pressure ``p0``, in pascals
        (default :data:`STANDARD_ATMOSPHERIC_PRESSURE`, the standard's 0,1 MPa).
    :return: The enclosed-gas dynamic stiffness per unit area ``s'a``, in N/m3.

    .. note::
        With the standard's :math:`p_0 = 0.1` MPa and :math:`\epsilon = 0.9`
        this reduces to :math:`s'_\mathrm{a} = 111/d` MN/m3 for ``d`` in millimetres
        (clause 8.2 NOTE).
    """
    atmospheric_pressure = require_positive(
        atmospheric_pressure, "atmospheric_pressure"
    )
    if not 0.0 < porosity <= 1.0:
        raise ValueError("'porosity' must be in the range (0, 1].")
    d = np.asarray(thickness, dtype=np.float64)
    if np.any(d <= 0.0):
        raise ValueError("'thickness' must be positive.")
    return as_float_or_array(atmospheric_pressure / (d * porosity))


def installed_dynamic_stiffness(
    apparent_stiffness: float,
    airflow_resistivity: float,
    *,
    gas_stiffness: float = 0.0,
) -> float:
    r"""Dynamic stiffness per unit area ``s'`` of the installed material (clause 8.2).

    Combines the apparent stiffness with the enclosed-gas term according to the
    lateral airflow resistivity ``r``:

    * :math:`r \ge 100` kPa.s/m2 -> :math:`s' = s'_\mathrm{t}` (Formula 5);
    * :math:`10 \le r < 100` kPa.s/m2 -> :math:`s' = s'_\mathrm{t} + s'_\mathrm{a}`
      (Formula 6);
    * :math:`r < 10` kPa.s/m2 -> the standard only requires the qualitative
      criterion :math:`s'_\mathrm{t} \gg s'_\mathrm{a}` (clause 8.2). This implementation
      applies its own engineering threshold: ``s'a`` below 10 % of ``s't`` is treated as
      negligible and :math:`s' = s'_\mathrm{t}` (a :class:`DynamicStiffnessWarning` is
      emitted; clause 8.2 requires the error caused by disregarding ``s'a`` to
      be stated in the test report); above it the result is ``nan``, as the
      method cannot resolve ``s'``.

    :param apparent_stiffness: Apparent dynamic stiffness ``s't``, in N/m3.
    :param airflow_resistivity: Lateral airflow resistivity ``r``, in kPa.s/m2
        (ISO 9053).
    :param gas_stiffness: Enclosed-gas dynamic stiffness ``s'a``, in N/m3 (see
        :func:`enclosed_gas_stiffness`); needed for :math:`r < 100` kPa.s/m2.
    :return: The installed dynamic stiffness per unit area ``s'``, in N/m3
        (``nan`` when the method cannot resolve it).
    """
    apparent_stiffness = require_positive(apparent_stiffness, "apparent_stiffness")
    if airflow_resistivity <= 0.0:
        raise ValueError("'airflow_resistivity' must be positive.")
    if gas_stiffness < 0.0:
        raise ValueError("'gas_stiffness' must be non-negative.")

    if airflow_resistivity >= _HIGH_RESISTIVITY:
        return apparent_stiffness
    if airflow_resistivity >= _LOW_RESISTIVITY:
        return apparent_stiffness + gas_stiffness
    # r < 10 kPa.s/m2: the enclosed gas is only negligible for a firm structure.
    if gas_stiffness > 0.1 * apparent_stiffness:
        warnings.warn(
            "for airflow resistivity below 10 kPa.s/m2 with a non-negligible "
            "enclosed-gas stiffness, EN 29052-1 cannot resolve s' (clause 8.2); "
            "returning nan.",
            DynamicStiffnessWarning,
            stacklevel=2,
        )
        return float("nan")
    warnings.warn(
        "airflow resistivity below 10 kPa.s/m2: s' is taken as s't with the "
        "enclosed-gas term disregarded; EN 29052-1 requires the reason and "
        "the estimated error caused by disregarding s'a to be stated in the "
        "test report (clause 8.2).",
        DynamicStiffnessWarning,
        stacklevel=2,
    )
    return apparent_stiffness


def natural_frequency(
    dynamic_stiffness: ArrayLike, mass_per_area: float
) -> np.ndarray | float:
    r"""Natural frequency ``f0`` of the resiliently supported floor (Formula 2).

    :math:`f_0 = (1/2\pi)\sqrt{s'/m'}`.

    :param dynamic_stiffness: Dynamic stiffness per unit area ``s'``, in N/m3
        (scalar or array).
    :param mass_per_area: Mass per unit area of the supported floor ``m'``, in
        kg/m2.
    :return: The natural frequency ``f0``, in hertz.
    """
    mass_per_area = require_positive(mass_per_area, "mass_per_area")
    s = np.asarray(dynamic_stiffness, dtype=np.float64)
    if np.any(s <= 0.0):
        raise ValueError("'dynamic_stiffness' must be positive.")
    return as_float_or_array(np.sqrt(s / mass_per_area) / _TWO_PI)


# ---------------------------------------------------------------------------
# Bundled measurement result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicStiffnessResult:
    """Dynamic stiffness of a resilient layer and the floating-floor resonance.

    :ivar apparent_stiffness: Apparent dynamic stiffness ``s't``, in N/m3.
    :ivar gas_stiffness: Enclosed-gas dynamic stiffness ``s'a``, in N/m3.
    :ivar dynamic_stiffness: Installed dynamic stiffness ``s'``, in N/m3.
    :ivar resonant_frequency: Measured test resonant frequency ``fr``, in hertz.
    :ivar floor_mass_per_area: Supported-floor mass per unit area ``m'``, kg/m2.
    :ivar natural_frequency: Installed-floor natural frequency ``f0``, in hertz.
    """

    apparent_stiffness: float
    gas_stiffness: float
    dynamic_stiffness: float
    resonant_frequency: float
    floor_mass_per_area: float
    natural_frequency: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot ``f0(s')`` with this design point marked.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_dynamic_stiffness

        check_language(language)
        return plot_dynamic_stiffness(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an EN 29052-1 dynamic-stiffness test-report fiche to a PDF.

        Writes a one-page accredited dynamic-stiffness report (EN 29052-1:1992,
        identical to ISO 9052-1:1989): the standard-basis line, an optional
        metadata header block (client, specimen, the total mass per unit area
        ``m't`` used during the test, the loaded specimen thickness ``d``, test
        facility, date, climate ...), a two-panel body with a compact metrics
        table (the resonant frequency ``fr``, the apparent dynamic stiffness
        ``s't`` of Formula 4, the enclosed-gas term ``s'a`` of Formula 7 when it
        applies, the installed dynamic stiffness ``s'`` of Clause 8.2 and the
        supported-floor natural frequency ``f0`` of Formula 2) beside the
        ``f0(s')`` design curve, a boxed apparent dynamic stiffness ``s't`` with
        the installed ``s'`` and the resonance ``fr`` alongside, and a footer
        with the fixed disclaimer. EN 29052-1 is a characterisation, so there is
        no pass/fail verdict.

        Clause 9 requires every dynamic stiffness per unit area to be stated in
        meganewtons per cubic metre to the nearest meganewton per cubic metre,
        so the stiffness values are rounded to the nearest MN/m3; the
        frequencies are shown to 0,1 Hz.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche. The applicable descriptive
            fields are ``client``, ``manufacturer``, ``specimen``,
            ``mass_per_area`` (the total mass per unit area ``m't``),
            ``thickness`` (the loaded specimen thickness ``d``, in metres, shown
            in millimetres), ``test_room``,
            ``test_date``, ``temperature``, ``relative_humidity``,
            ``measurement_standard``, ``laboratory``, ``operator``,
            ``report_id`` and ``notes``. The ``requirement`` field is ignored
            (EN 29052-1 has no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            dynamic-stiffness fiche has a single body layout, so it has no
            effect.
        :param language: Fiche language: ``"en"`` (default, English, decimal
            point) or ``"es"`` (Spanish, decimal comma).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab or matplotlib is not installed. The
            fiche always embeds the ``f0(s')`` design curve, so both are
            required (``pip install "phonometry[report,plot]"``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso9052 import render_dynamic_stiffness_report

        return render_dynamic_stiffness_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def floating_floor_resonance(
    resonant_frequency: float,
    total_mass_per_area: float,
    floor_mass_per_area: float,
    *,
    airflow_resistivity: float = float("inf"),
    thickness: float | None = None,
    porosity: float | None = None,
    atmospheric_pressure: float = STANDARD_ATMOSPHERIC_PRESSURE,
) -> DynamicStiffnessResult:
    r"""Full EN 29052-1 chain: measured resonance -> installed ``s'`` and ``f0``.

    Chains the apparent dynamic stiffness (Formula 4), the enclosed-gas term
    (Formula 7, when ``thickness`` and ``porosity`` are given), the airflow
    resistivity combination (clause 8.2) and the installed-floor natural
    frequency (Formula 2).

    :param resonant_frequency: Measured resonant frequency ``fr``, in hertz.
    :param total_mass_per_area: Test total mass per unit area ``m't``, kg/m2.
    :param floor_mass_per_area: Supported-floor mass per unit area ``m'``, kg/m2.
    :param airflow_resistivity: Lateral airflow resistivity ``r``, in kPa.s/m2
        (default ``inf`` -> the high-resistivity case :math:`s' = s'_\mathrm{t}`).
    :param thickness: Specimen thickness ``d`` under load, in metres. Required
        together with ``porosity`` for the enclosed-gas term, which applies
        when :math:`r < 100` kPa.s/m2. That condition is on the *value* of
        ``airflow_resistivity`` rather than on a literal, so a signature
        cannot state it: it is checked here and raises.
    :param porosity: Specimen porosity ``epsilon``, required with
        ``thickness`` (see above).
    :param atmospheric_pressure: Atmospheric pressure ``p0``, in pascals.
    :return: The :class:`DynamicStiffnessResult`.
    """
    s_apparent = float(
        apparent_dynamic_stiffness(resonant_frequency, total_mass_per_area)
    )
    s_gas = 0.0
    if airflow_resistivity < _HIGH_RESISTIVITY:
        if thickness is None or porosity is None:
            raise ValueError(
                "'thickness' and 'porosity' are required for the enclosed-gas "
                "term when airflow_resistivity < 100 kPa.s/m2."
            )
        s_gas = float(
            enclosed_gas_stiffness(
                thickness, porosity, atmospheric_pressure=atmospheric_pressure
            )
        )
    s_installed = installed_dynamic_stiffness(
        s_apparent, airflow_resistivity, gas_stiffness=s_gas
    )
    f0 = (
        float(natural_frequency(s_installed, floor_mass_per_area))
        if np.isfinite(s_installed)
        else float("nan")
    )
    return DynamicStiffnessResult(
        apparent_stiffness=s_apparent,
        gas_stiffness=s_gas,
        dynamic_stiffness=s_installed,
        resonant_frequency=float(resonant_frequency),
        floor_mass_per_area=float(floor_mass_per_area),
        natural_frequency=f0,
    )
