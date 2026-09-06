#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Control valve hydrodynamic noise (IEC 60534-8-4:2005).

A liquid does not compress, so a control valve on a water line cannot make
the shock cells that IEC 60534-8-3 spends five regimes on. It makes two
things instead, and the whole of this part is the sum of them:

* **turbulence** in the jet leaving the vena contracta, whose acoustical
  efficiency is a straight line in the jet velocity, Equation (8);
* **cavitation**, once the differential pressure passes the point where the
  liquid flashes to vapour in the vena contracta and the bubbles collapse
  again downstream. Equation (9) gives that its own efficiency, and it is a
  steep function of how far past the threshold the valve is: a fifth power of
  :math:`x_F/x_{Fzp1}` multiplied by an exponential.

The threshold is the **characteristic pressure ratio** :math:`x_{Fz}`, a
measured property of the valve (IEC 60534-8-2) that Equations (3a) and (3b)
estimate when no measurement exists, corrected to the working inlet pressure
by Equation (3c). Everything in the method turns on where the operating ratio
:math:`x_F` of Equation (1) sits with respect to it, which is why Annex A's
third example perturbs :math:`x_{Fz}` by 0,1 and watches the answer move
14 dB.

After the source, the chain is the same shape as the aerodynamic one: an
internal level at the pipe wall, Equation (10); a transmission loss through
the wall, negative by construction and anchored at the ring frequency,
Equations (14) and (15); and a level 1 m outside, Equations (18a) and (18b).
The band-by-band route of 5.4 spreads the internal level around the peak
frequency with Equations (20a) and (20b) and gives the wall a
frequency-dependent loss with Equations (22a) and (22b).

**Six defects of the printed document**, all confirmed on the page and all
recorded in ``docs/ERRATA.md``:

* Equation (12) is printed **twice, differently**: Clause 5.1 gives the
  Strouhal number a leading 0,02 and no valve style modifier, Annex A's
  Table A.1 gives it 0,036 and a factor :math:`F_d^{0,75}`. Only the annex
  form reproduces the annex's own printed :math:`N_{Str}`, which is why
  :data:`STROUHAL_CONSTANTS` carries both and the default is ``"annex"``.
* Table A.1 prints the band transmission loss as ``TL(8 000 Hz) = 51,76 dB``,
  positive, where its own two inputs sum to :math:`-51{,}763` dB.
* The seat diameter formula of 6.3.2 b), :math:`d_o = 5{,}2\sqrt{N_{34}C_n}`,
  returns millimetres for a symbol Clause 3 declares in metres, which is why
  :func:`last_stage_seat_diameter_mm` carries the unit in its name.
* Equation (23b) computes each stage's inlet pressure from the **next**
  stage's, which contradicts Equation (23a) and runs the pressure the wrong
  way along the trim.
* Equations (18a) and (18b) are printed with conditions on two different
  thresholds, :math:`x_F \le x_{Fz}` and :math:`x_{Fzp1} < x_F \le 1`, which
  divide the domain between them only at the one inlet pressure where those
  two are equal. Every other regime statement in the document tests the
  corrected ratio, and so does this module.
* Three intermediates of Table A.1 do not follow from the intermediates
  printed beside them, by up to 0,08 dB.

Clause 6, the multistage trim, is the same method with per-stage inputs:
:func:`stage_conditions` splits the differential, and either the stages are
summed in energy by Equation (27) or, for a fixed device with increasing flow
areas, only the last stage is calculated at all.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._internal.validation import (
    require_choice,
    require_finite_array,
    require_positive,
    require_positive_array,
)
from .valves import (
    AIR_SOUND_SPEED_M_S,
    PIPE_SOUND_SPEED_M_S,
    ValveNoiseWarning,
    jet_diameter,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ACOUSTIC_POWER_RATIOS",
    "AIR_DENSITY_KG_M3",
    "band_internal_levels",
    "CAPACITY_SCALE_CONSTANTS",
    "cavitation_differential",
    "cavitation_distribution",
    "cavitation_efficiency",
    "CAVITATION_FLOOR_WIDTH",
    "cavitation_peak_frequency",
    "cavitation_transmission_loss",
    "combine_stage_levels",
    "corrected_incipient_ratio",
    "differential_pressure_ratio",
    "HydrodynamicValveNoise",
    "incipient_cavitation_ratio",
    "internal_sound_pressure_level",
    "jet_strouhal_number",
    "last_stage_differential",
    "last_stage_seat_diameter_mm",
    "mechanical_stream_power",
    "multihole_incipient_cavitation_ratio",
    "pipe_ring_frequency",
    "REFERENCE_INLET_PRESSURE_PA",
    "reference_transmission_loss",
    "stage_conditions",
    "StageConditions",
    "STROUHAL_CONSTANTS",
    "transmission_loss_correction",
    "turbulent_distribution",
    "turbulent_efficiency",
    "turbulent_peak_frequency",
    "uniform_passage_style_modifier",
    "valve_hydrodynamic_noise",
    "vena_contracta_velocity",
]

# --------------------------------------------------------------------------
# Printed constants
# --------------------------------------------------------------------------

#: Table 1's numerical constant :math:`N_{34}`, keyed by the flow coefficient
#: the valve is rated in. It is the scale that lets one formula written for
#: :math:`K_v` be read with :math:`C_v`, and it appears wherever a flow
#: coefficient meets a length: Equations (3a) and (12), and the seat diameter
#: estimate of 6.3.2. Equation (3b) has no flow coefficient in it, so it has
#: no :math:`N_{34}` either; the constant printed at full size on its
#: baseline is :math:`N_o`, a count of holes.
CAPACITY_SCALE_CONSTANTS = {"Cv": 1.17, "Kv": 1.0}

#: Table 2's acoustic power ratio :math:`r_W`, the fraction of the sound
#: power that is radiated into the pipe rather than lost in the valve body.
#: The table gives three values only: a quarter for every globe and rotary
#: trim, a half for the butterflies, and one for an expander, which has no
#: body to lose anything in. The two swing-through and fluted-vane rows are
#: printed "to 70 deg", a limit on the travel rather than on the valve, and
#: the table says nothing about either of them past it.
ACOUSTIC_POWER_RATIOS = {
    "globe parabolic plug": 0.25,
    "globe 3 V-port plug": 0.25,
    "globe 4 V-port plug": 0.25,
    "globe 6 V-port plug": 0.25,
    "globe 60 hole drilled cage": 0.25,
    "globe 120 hole drilled cage": 0.25,
    "butterfly swing-through": 0.5,
    "butterfly fluted vane": 0.5,
    "butterfly 60 deg flat disk": 0.5,
    "eccentric rotary plug": 0.25,
    "segmented ball 90 deg": 0.25,
    "expander": 1.0,
}

#: Density of the air outside the pipe, from the Clause 3 symbol list, in
#: kg/m³. Equation (15) uses it against the pipe wall's own impedance, so it
#: is the value the transmission loss is printed for and not the air on the
#: day.
AIR_DENSITY_KG_M3 = 1.293

#: The inlet pressure Equation (3a) and Figures 4 to 9 are drawn for, in Pa.
#: Equation (3c) moves :math:`x_{Fz}` from here to the working pressure.
REFERENCE_INLET_PRESSURE_PA = 6.0e5

#: The two leading constants Equation (12) is printed with. ``"annex"`` is
#: Table A.1's :math:`0{,}036\,F_L^2\,C\,F_d^{0,75}` and ``"clause"`` is
#: Clause 5.1's :math:`0{,}02\,F_L^2\,C`, without the style modifier. They
#: are not the same function of the valve: for Annex A's :math:`F_d = 0{,}42`
#: the annex form is 6 % below the clause form, and for a plain single-port
#: valve at :math:`F_d = 1` it is 80 % above it.
STROUHAL_CONSTANTS = {"annex": 0.036, "clause": 0.02}

#: The width in :math:`x_F` of the band just above incipient cavitation where
#: the NOTE to Equation (17) floors the efficiency ratio. Inside it the
#: bracket of (17) cannot fall below 1, so the cavitating transmission loss
#: cannot be worse than the turbulent one.
CAVITATION_FLOOR_WIDTH = 0.1

#: The multiplier on the efficiency ratio inside Equation (17).
_CAVITATION_TRANSMISSION_FACTOR = 250.0

#: Equation (10)'s leading constant, which carries the reference pressure and
#: the unit conversion of the whole internal level.
_INTERNAL_LEVEL_COEFFICIENT = 3.2e9

#: Equation (8): the acoustical efficiency of the turbulent jet at
#: :math:`U_{vc} = c_L`.
_TURBULENT_EFFICIENCY = 1.0e-4

#: Equation (9)'s leading constant.
_CAVITATION_EFFICIENCY = 0.32

#: The distance the external levels of (18a), (18b) and (21) are given at,
#: in m. It enters the spreading term twice, once on each side of the pipe.
_MEASUREMENT_DISTANCE_M = 1.0

#: Equation (3b)'s two printed constants, for multihole trims.
_MULTIHOLE_OFFSET = 4.5
_MULTIHOLE_FACTOR = 1650.0

#: Equation (3a)'s numerator and the multiplier on the style modifier.
_INCIPIENT_NUMERATOR = 0.90
_INCIPIENT_FACTOR = 3.0

#: The exponent of Equation (3c).
_INLET_CORRECTION_EXPONENT = 0.125

#: The unnumbered seat diameter formula of 6.3.2 b).
_SEAT_DIAMETER_CONSTANT = 5.2

#: Equation (13)'s leading factor and its two exponents.
_CAVITATION_PEAK_FACTOR = 6.0
_CAVITATION_PEAK_EXPONENTS = (2.0, 2.5)

#: Equations (20a) and (20b): the prefactor of the rising branch, then the
#: pair of exponents and the trailing constant of each.
_DISTRIBUTION_PREFACTOR = 0.25
_TURBULENT_DISTRIBUTION = (3.0, -1.0, 3.1)
_CAVITATION_DISTRIBUTION = (1.5, -1.5, 3.5)

#: The coefficient and the exponent of Equations (16b) and (22b).
_LOSS_CORRECTION_COEFFICIENT = -20.0
_LOSS_CORRECTION_EXPONENT = 1.5

#: Equation (15)'s stand-alone term, in dB.
_RING_LOSS_OFFSET = -10.0

#: The band edges 5.4.1 prints for the one-third-octave route, in Hz.
_BAND_RANGE_HZ = (50.0, 20000.0)

#: The differential pressure ratio at which the outlet reaches the vapour
#: pressure and the liquid flashes. Clause 5.1 prints its cavitating region
#: closed at this value, but Equation (9) divides by :math:`1 - x_F` and so
#: has no value on the boundary itself, which is where this module stops.
_FLASHING_RATIO = 1.0

#: The smallest number of stages Clause 6 is about.
_MINIMUM_STAGES = 2

#: How far the stage coefficients may miss the series law before
#: :func:`stage_conditions` says so, as a fraction of the valve's own
#: resistance.
_SERIES_LAW_TOLERANCE = 0.05


# --------------------------------------------------------------------------
# 4 Preliminary calculations
# --------------------------------------------------------------------------


def differential_pressure_ratio(
    *, inlet_pressure: float, outlet_pressure: float, vapour_pressure: float
) -> float:
    r"""Equation (1): the differential pressure ratio.

    .. math::

       x_F = \frac{p_1 - p_2}{p_1 - p_v}

    The denominator is the differential the valve would need to take the
    liquid all the way down to its vapour pressure, so :math:`x_F` says how
    far towards flashing this operating point is, and 1 is the whole way.

    :param inlet_pressure: :math:`p_1`, absolute, in Pa.
    :param outlet_pressure: :math:`p_2`, absolute, in Pa.
    :param vapour_pressure: :math:`p_v` of the liquid at the inlet
        temperature, absolute, in Pa.
    :return: :math:`x_F`, dimensionless.
    :raises ValueError: If a pressure is not positive and finite, if the
        valve does not drop pressure, or if the inlet is already at the
        vapour pressure.
    """
    p1 = require_positive(inlet_pressure, "inlet_pressure")
    p2 = require_positive(outlet_pressure, "outlet_pressure")
    pv = require_positive(vapour_pressure, "vapour_pressure")
    if p2 > p1:
        msg = (
            "A control valve drops pressure, so 'outlet_pressure' cannot be "
            f"above 'inlet_pressure'; got {outlet_pressure!r} and "
            f"{inlet_pressure!r} Pa."
        )
        raise ValueError(msg)
    if pv >= p1:
        msg = (
            "Equation (1) divides by p_1 - p_v, so the liquid must arrive "
            f"above its vapour pressure; got p_1 = {inlet_pressure!r} Pa and "
            f"p_v = {vapour_pressure!r} Pa."
        )
        raise ValueError(msg)
    return float((p1 - p2) / (p1 - pv))


def cavitation_differential(
    *,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
    pressure_recovery: float,
) -> float:
    r"""Equation (2): the differential the jet velocity is computed from.

    .. math::

       \Delta p_c = \min\left[(p_1 - p_2),\; F_L^2 (p_1 - p_v)\right]

    The second candidate is where the flow chokes. Past it the valve cannot
    turn any more differential into velocity, so :math:`\Delta p_c` stops
    following :math:`p_1 - p_2` and Equation (5) stops accelerating the jet,
    even though the noise keeps rising because cavitation takes over.

    The printed equation says "lower than … or …", with no ``min`` operator
    and no inequality; the minimum is what it means.

    :param inlet_pressure: :math:`p_1`, absolute, in Pa.
    :param outlet_pressure: :math:`p_2`, absolute, in Pa.
    :param vapour_pressure: :math:`p_v`, absolute, in Pa.
    :param pressure_recovery: :math:`F_L` of the valve, dimensionless.
    :return: :math:`\Delta p_c`, in Pa.
    :raises ValueError: If a pressure is not positive and finite, if the
        valve does not drop pressure, or if the recovery factor is outside
        the range a recovery factor lives in.
    """
    p1 = require_positive(inlet_pressure, "inlet_pressure")
    p2 = require_positive(outlet_pressure, "outlet_pressure")
    pv = require_positive(vapour_pressure, "vapour_pressure")
    recovery = _require_recovery(pressure_recovery)
    if p2 > p1:
        msg = (
            "A control valve drops pressure, so 'outlet_pressure' cannot be "
            f"above 'inlet_pressure'; got {outlet_pressure!r} and "
            f"{inlet_pressure!r} Pa."
        )
        raise ValueError(msg)
    if pv >= p1:
        msg = (
            "Equation (2) needs the inlet above the vapour pressure; got "
            f"p_1 = {inlet_pressure!r} Pa and p_v = {vapour_pressure!r} Pa."
        )
        raise ValueError(msg)
    return float(min(p1 - p2, recovery**2 * (p1 - pv)))


def incipient_cavitation_ratio(
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    *,
    coefficient: str = "Cv",
) -> float:
    r"""Equation (3a): where cavitation becomes audible, estimated.

    .. math::

       x_{Fz} = \frac{0{,}90}
                     {\sqrt{1 + 3 F_d \sqrt{\dfrac{C}{N_{34} F_L}}}}

    4.2 asks for a measured :math:`x_{Fz}` (IEC 60534-8-2) and offers this
    only as an estimate, warning that a prediction built on it "can create
    uncertainties as illustrated in Annex A". Annex A's third example is
    exactly that illustration: 0,1 on this number moves the answer 14 dB.

    The nesting is worth reading twice. The outer radical covers the whole of
    :math:`1 + 3F_d\sqrt{\cdots}`; the inner one covers only the capacity
    group. A valve with a small style modifier, a cage full of small holes,
    keeps :math:`x_{Fz}` high and stays quiet longer.

    :param flow_coefficient: :math:`C` at the travel being examined.
    :param style_modifier: :math:`F_d`, the valve style modifier, taken from
        IEC 60534-8-3 (4.3 prints no table of its own).
    :param pressure_recovery: :math:`F_L`, dimensionless.
    :param coefficient: Which flow coefficient ``flow_coefficient`` is,
        ``"Cv"`` or ``"Kv"``, which selects :math:`N_{34}` from Table 1.
    :return: :math:`x_{Fz}` at an inlet pressure of 6 × 10⁵ Pa,
        dimensionless.
    :raises ValueError: If a value is not positive and finite, or the
        coefficient is not one Table 1 prints a constant for.
    """
    kind = require_choice(coefficient, "coefficient", tuple(CAPACITY_SCALE_CONSTANTS))
    capacity = require_positive(flow_coefficient, "flow_coefficient")
    modifier = require_positive(style_modifier, "style_modifier")
    recovery = _require_recovery(pressure_recovery)
    inner = math.sqrt(capacity / (CAPACITY_SCALE_CONSTANTS[kind] * recovery))
    return float(
        _INCIPIENT_NUMERATOR / math.sqrt(1.0 + _INCIPIENT_FACTOR * modifier * inner)
    )


def multihole_incipient_cavitation_ratio(
    passages: int, hole_diameter: float, pressure_recovery: float
) -> float:
    r"""Equation (3b): the same threshold for a multihole trim.

    .. math::

       x_{Fz} = \frac{1}
                     {\sqrt{4{,}5 + 1\,650\,\dfrac{N_o d_H^2}{F_L}}}

    A multihole trim is not described by its capacity and style modifier but
    by how many holes it has and how big they are, which is what this form
    takes. The group :math:`N_o d_H^2` is the total hole area to within
    :math:`\pi/4`, so two trims with the same open area and different hole
    counts get the same threshold here.

    :param passages: :math:`N_o`, the number of independent, identical flow
        passages.
    :param hole_diameter: :math:`d_H`, the hole diameter, in m.
    :param pressure_recovery: :math:`F_L`, dimensionless.
    :return: :math:`x_{Fz}` at an inlet pressure of 6 × 10⁵ Pa,
        dimensionless.
    :raises ValueError: If the passage count is not a whole number of one or
        more, or another value is not positive and finite.
    """
    count = _require_count(passages, "passages")
    diameter = require_positive(hole_diameter, "hole_diameter")
    recovery = _require_recovery(pressure_recovery)
    inner = _MULTIHOLE_OFFSET + _MULTIHOLE_FACTOR * count * diameter**2 / recovery
    return float(1.0 / math.sqrt(inner))


def corrected_incipient_ratio(incipient_ratio: float, inlet_pressure: float) -> float:
    r"""Equation (3c): the threshold moved to the working inlet pressure.

    .. math::

       x_{Fzp1} = x_{Fz}
                  \left(\frac{6 \times 10^5}{p_1}\right)^{0,125}

    Equation (3a) and the charts of Figures 4 to 9 are drawn at 6 × 10⁵ Pa.
    Raising the inlet pressure lowers the threshold, because the same
    differential ratio now means a larger absolute pressure drop and a
    livelier vena contracta, but the eighth-power root makes it a slow
    correction: ten times the inlet pressure moves the threshold by a
    quarter.

    :param incipient_ratio: :math:`x_{Fz}` at 6 × 10⁵ Pa, measured or from
        :func:`incipient_cavitation_ratio`.
    :param inlet_pressure: :math:`p_1`, absolute, in Pa.
    :return: :math:`x_{Fzp1}`, dimensionless.
    :raises ValueError: If a value is not positive and finite, or the
        threshold is at or above 1, where the method has already stopped.
    """
    ratio = _require_threshold(incipient_ratio, "incipient_ratio")
    p1 = require_positive(inlet_pressure, "inlet_pressure")
    return float(
        ratio * (REFERENCE_INLET_PRESSURE_PA / p1) ** _INLET_CORRECTION_EXPONENT
    )


def vena_contracta_velocity(
    differential: float, density: float, pressure_recovery: float
) -> float:
    r"""Equation (5): the jet velocity.

    .. math::

       U_{vc} = \frac{1}{F_L}\sqrt{\frac{2 \Delta p_c}{\rho_L}}

    Bernoulli's velocity for the differential of Equation (2), divided by the
    recovery factor because :math:`F_L` is defined as the fraction of the
    ideal velocity head the valve actually reaches at the vena contracta.

    :param differential: :math:`\Delta p_c` from
        :func:`cavitation_differential`, in Pa.
    :param density: :math:`\rho_L` of the liquid, in kg/m³.
    :param pressure_recovery: :math:`F_L`, dimensionless.
    :return: :math:`U_{vc}`, in m/s.
    :raises ValueError: If a value is not positive and finite, or the
        recovery factor is outside its range.
    """
    drop = require_positive(differential, "differential")
    rho = require_positive(density, "density")
    recovery = _require_recovery(pressure_recovery)
    return float(math.sqrt(2.0 * drop / rho) / recovery)


def mechanical_stream_power(
    mass_flow: float, velocity: float, pressure_recovery: float
) -> float:
    r"""Equation (6): the stream power the valve dissipates.

    .. math::

       W_m = \frac{\dot m\, U_{vc}^2\, F_L^2}{2}

    The kinetic power of the jet, :math:`\dot m U_{vc}^2/2`, scaled back by
    :math:`F_L^2` to the part of it that is actually thrown away rather than
    recovered as pressure downstream. Equation (7a) then takes a part in
    :math:`10^{-6}` of this and calls it sound.

    :param mass_flow: :math:`\dot m`, in kg/s.
    :param velocity: :math:`U_{vc}` from :func:`vena_contracta_velocity`, in
        m/s.
    :param pressure_recovery: :math:`F_L`, dimensionless.
    :return: :math:`W_m`, in W.
    :raises ValueError: If a value is not positive and finite, or the
        recovery factor is outside its range.
    """
    flow = require_positive(mass_flow, "mass_flow")
    speed = require_positive(velocity, "velocity")
    recovery = _require_recovery(pressure_recovery)
    return float(flow * speed**2 * recovery**2 / 2.0)


# --------------------------------------------------------------------------
# 5 Noise predictions
# --------------------------------------------------------------------------


def turbulent_efficiency(velocity: float, sound_speed: float) -> float:
    r"""Equation (8): the acoustical efficiency of the turbulent jet.

    .. math::

       \eta_{turb} = 10^{-4}\left(\frac{U_{vc}}{c_L}\right)

    5.1 argues the case: at these velocities the jet is slow enough to be a
    monopole, and a monopole's efficiency rises with the first power of the
    Mach number, reaching :math:`10^{-4}` when the jet reaches the speed of
    sound in the liquid. Water carries sound at about 1 400 m/s and a control
    valve jet runs at tens of metres per second, so the efficiency comes out
    in the :math:`10^{-6}` range: one part in a million of the stream power.

    :param velocity: :math:`U_{vc}`, in m/s.
    :param sound_speed: :math:`c_L` in the liquid, in m/s.
    :return: :math:`\eta_{turb}`, dimensionless.
    :raises ValueError: If a value is not positive and finite.
    """
    speed = require_positive(velocity, "velocity")
    sonic = require_positive(sound_speed, "sound_speed")
    if speed > sonic:
        msg = (
            "Equation (8) reaches its constant when the jet reaches the "
            "speed of sound in the liquid, which a control valve does not "
            f"do; got {velocity!r} m/s against {sound_speed!r} m/s. Check "
            "the two are not the other way round."
        )
        raise ValueError(msg)
    return float(_TURBULENT_EFFICIENCY * speed / sonic)


def cavitation_efficiency(
    *,
    turbulent: float,
    differential: float,
    choked_differential: float,
    pressure_ratio: float,
    corrected_ratio: float,
) -> float:
    r"""Equation (9): what the collapsing bubbles add.

    .. math::

       \eta_{cav} = 0{,}32\, \eta_{turb}
         \sqrt{\frac{p_1 - p_2}{\Delta p_c}\cdot\frac{1}{x_{Fzp1}}}\;
         e^{5 x_{Fzp1}}
         \left(\frac{1 - x_{Fzp1}}{1 - x_F}\right)^{0,5}
         \left(\frac{x_F}{x_{Fzp1}}\right)^{5}
         \left(x_F - x_{Fzp1}\right)^{1,5}

    Three of those factors are what makes cavitation noise behave the way it
    does. :math:`(x_F - x_{Fzp1})^{1,5}` starts the term at exactly zero on
    the threshold, so the two regimes meet without a step;
    :math:`(x_F/x_{Fzp1})^5` then makes it climb almost vertically once the
    threshold is passed; and :math:`(1-x_F)^{-0,5}` sends it towards
    infinity as the valve approaches flashing, which is where the method
    stops.

    :param turbulent: :math:`\eta_{turb}` from :func:`turbulent_efficiency`.
    :param differential: :math:`p_1 - p_2`, in Pa.
    :param choked_differential: :math:`\Delta p_c` from
        :func:`cavitation_differential`, in Pa.
    :param pressure_ratio: :math:`x_F` of Equation (1).
    :param corrected_ratio: :math:`x_{Fzp1}` of Equation (3c).
    :return: :math:`\eta_{cav}`, dimensionless, and exactly zero on the
        threshold.
    :raises ValueError: If a value is not positive and finite, if the
        operating point is below the threshold, or if it is at or past
        flashing, where the equation has no value.
    """
    efficiency = require_positive(turbulent, "turbulent")
    drop = require_positive(differential, "differential")
    choked = require_positive(choked_differential, "choked_differential")
    ratio = require_positive(pressure_ratio, "pressure_ratio")
    threshold = _require_threshold(corrected_ratio, "corrected_ratio")
    if ratio < threshold:
        msg = (
            "Equation (9) is the cavitating branch, so 'pressure_ratio' must "
            f"be at or above 'corrected_ratio'; got {pressure_ratio!r} and "
            f"{corrected_ratio!r}."
        )
        raise ValueError(msg)
    if ratio >= _FLASHING_RATIO:
        msg = (
            "5.1 prints the cavitating region closed at x_F = 1, but "
            "Equation (9) divides by 1 - x_F and has no value on that "
            f"boundary, where the liquid flashes; got {pressure_ratio!r}."
        )
        raise ValueError(msg)
    return float(
        _CAVITATION_EFFICIENCY
        * efficiency
        * math.sqrt(drop / choked / threshold)
        * math.exp(5.0 * threshold)
        * ((1.0 - threshold) / (1.0 - ratio)) ** 0.5
        * (ratio / threshold) ** 5
        * (ratio - threshold) ** 1.5
    )


def internal_sound_pressure_level(
    *,
    sound_power: float,
    density: float,
    sound_speed: float,
    internal_diameter: float,
) -> float:
    r"""Equation (10): the level inside, at the pipe wall.

    .. math::

       L_{pi} = 10 \lg\left(
         \frac{3{,}2 \times 10^9\, W_a\, \rho_L\, c_L}{D_i^2}\right)

    The sound power is spread over the pipe cross-section and turned into a
    pressure through the impedance of the liquid, which is why the density
    and the speed of sound multiply rather than divide: water's impedance is
    3 400 times air's, so the same acoustic power makes a level some 35 dB
    higher inside a water line than inside an air line. Levels of 150 dB in
    the pipe are ordinary here, and it is the transmission loss, not the
    source, that makes the outside habitable.

    In the printed equation the density has lost its Greek base glyph and
    reads as a bare subscript; Table A.1 prints the same equation with
    :math:`\rho_L` intact, which settles it.

    :param sound_power: :math:`W_a` of Equation (7a) or (7b), in W.
    :param density: :math:`\rho_L` of the liquid, in kg/m³.
    :param sound_speed: :math:`c_L` in the liquid, in m/s.
    :param internal_diameter: :math:`D_i` of the downstream pipe, in m.
    :return: :math:`L_{pi}`, in dB re 2 × 10⁻⁵ Pa.
    :raises ValueError: If a value is not positive and finite.
    """
    power = require_positive(sound_power, "sound_power")
    rho = require_positive(density, "density")
    sonic = require_positive(sound_speed, "sound_speed")
    bore = require_positive(internal_diameter, "internal_diameter")
    return float(
        10.0 * math.log10(_INTERNAL_LEVEL_COEFFICIENT * power * rho * sonic / bore**2)
    )


def jet_strouhal_number(  # noqa: PLR0913
    *,
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    corrected_ratio: float,
    valve_diameter: float,
    seat_diameter: float,
    inlet_pressure: float,
    vapour_pressure: float,
    coefficient: str = "Cv",
    form: str = "annex",
) -> float:
    r"""Equation (12): the Strouhal number of the jet.

    .. math::

       N_{STR} = \frac{0{,}036\, F_L^2\, C\, F_d^{0,75}}
                      {N_{34}\, x_{Fzp1}^{1,5}\, d\, d_o}
                 \left(\frac{1}{p_1 - p_v}\right)^{0,57}

    This is the one place where the two printings of the standard disagree
    with each other. The form above is Table A.1's; Clause 5.1 prints the
    same equation with a leading 0,02 and **no** :math:`F_d^{0,75}`. Only the
    annex form reproduces the annex's own :math:`N_{Str} = 0{,}399`, so it is
    the default here; pass ``form="clause"`` for the normative text's version
    and see ``docs/ERRATA.md``.

    Unlike the Strouhal number of a free jet, which is a constant near 0,2,
    this one is a fitted group that carries the whole geometry of the valve
    and comes out anywhere between about 0,2 and 0,5.

    :param flow_coefficient: :math:`C` at the travel being examined.
    :param style_modifier: :math:`F_d`, used only by the ``"annex"`` form.
    :param pressure_recovery: :math:`F_L`, dimensionless.
    :param corrected_ratio: :math:`x_{Fzp1}` of Equation (3c).
    :param valve_diameter: :math:`d`, the valve inlet internal diameter, in
        m.
    :param seat_diameter: :math:`d_o`, the seat or orifice diameter, in m.
    :param inlet_pressure: :math:`p_1`, absolute, in Pa.
    :param vapour_pressure: :math:`p_v`, absolute, in Pa.
    :param coefficient: ``"Cv"`` or ``"Kv"``, selecting :math:`N_{34}`.
    :param form: Which printing of Equation (12) to use, ``"annex"`` or
        ``"clause"``.
    :return: :math:`N_{STR}`, dimensionless.
    :raises ValueError: If a value is not positive and finite, if the inlet
        is at or below the vapour pressure, or if a choice is not one the
        standard prints.
    """
    kind = require_choice(coefficient, "coefficient", tuple(CAPACITY_SCALE_CONSTANTS))
    printing = require_choice(form, "form", tuple(STROUHAL_CONSTANTS))
    capacity = require_positive(flow_coefficient, "flow_coefficient")
    modifier = require_positive(style_modifier, "style_modifier")
    recovery = _require_recovery(pressure_recovery)
    threshold = _require_threshold(corrected_ratio, "corrected_ratio")
    inlet = require_positive(valve_diameter, "valve_diameter")
    seat = require_positive(seat_diameter, "seat_diameter")
    p1 = require_positive(inlet_pressure, "inlet_pressure")
    pv = require_positive(vapour_pressure, "vapour_pressure")
    if pv >= p1:
        msg = (
            "Equation (12) divides by p_1 - p_v; got p_1 = "
            f"{inlet_pressure!r} Pa and p_v = {vapour_pressure!r} Pa."
        )
        raise ValueError(msg)
    style = modifier**0.75 if printing == "annex" else 1.0
    numerator = STROUHAL_CONSTANTS[printing] * recovery**2 * capacity * style
    denominator = CAPACITY_SCALE_CONSTANTS[kind] * threshold**1.5 * inlet * seat
    return float(numerator / denominator * (1.0 / (p1 - pv)) ** 0.57)


def turbulent_peak_frequency(
    strouhal_number: float, velocity: float, jet: float
) -> float:
    r"""Equation (11): the peak frequency of the turbulent noise.

    .. math::

       f_{p,turb} = N_{STR}\, \frac{U_{vc}}{D_j}

    A jet radiates around the frequency at which its own eddies pass a fixed
    point, which is the velocity divided by the size of the eddies. The jet
    diameter of Equation (4) stands for that size.

    :param strouhal_number: :math:`N_{STR}` from
        :func:`jet_strouhal_number`.
    :param velocity: :math:`U_{vc}`, in m/s.
    :param jet: :math:`D_j` of Equation (4), in m.
    :return: :math:`f_{p,turb}`, in Hz.
    :raises ValueError: If a value is not positive and finite.
    """
    strouhal = require_positive(strouhal_number, "strouhal_number")
    speed = require_positive(velocity, "velocity")
    diameter = require_positive(jet, "jet")
    return float(strouhal * speed / diameter)


def cavitation_peak_frequency(
    turbulent_peak: float, pressure_ratio: float, corrected_ratio: float
) -> float:
    r"""Equation (13): the peak frequency of the cavitation noise.

    .. math::

       f_{p,cav} = 6 f_{p,turb}
         \left(\frac{1 - x_F}{1 - x_{Fzp1}}\right)^{2}
         \left(\frac{x_{Fzp1}}{x_F}\right)^{2,5}

    Both brackets are the reciprocals of the ones in Equation (9), and that
    is deliberate rather than a misprint: the same factors that make the
    cavitation *level* rise as the valve is opened further into cavitation
    make its peak frequency fall, because the bubbles grow larger and take
    longer to collapse. Just past the threshold the collapse is fast and the
    noise is hissy, six times the turbulent peak; deep into cavitation it
    drops back down into a rumble.

    :param turbulent_peak: :math:`f_{p,turb}` from :func:`turbulent_peak_frequency`,
        in Hz.
    :param pressure_ratio: :math:`x_F` of Equation (1).
    :param corrected_ratio: :math:`x_{Fzp1}` of Equation (3c).
    :return: :math:`f_{p,cav}`, in Hz.
    :raises ValueError: If a value is not positive and finite, or the
        operating point is at or past flashing.
    """
    peak = require_positive(turbulent_peak, "turbulent_peak")
    ratio = require_positive(pressure_ratio, "pressure_ratio")
    threshold = _require_threshold(corrected_ratio, "corrected_ratio")
    if ratio >= _FLASHING_RATIO:
        msg = (
            "Equation (13) is finite at x_F = 1, but the sound power that "
            "would go with it is not: Equation (9) divides by 1 - x_F, so "
            "the cavitating branch stops below the flashing point; got "
            f"{pressure_ratio!r}."
        )
        raise ValueError(msg)
    high, low = _CAVITATION_PEAK_EXPONENTS
    return float(
        _CAVITATION_PEAK_FACTOR
        * peak
        * ((1.0 - ratio) / (1.0 - threshold)) ** high
        * (threshold / ratio) ** low
    )


def pipe_ring_frequency(
    internal_diameter: float, *, pipe_sound_speed: float = PIPE_SOUND_SPEED_M_S
) -> float:
    r"""Equation (14): the ring frequency of the pipe.

    .. math::

       f_r = \frac{c_p}{\pi D_i}

    The frequency at which one wavelength of a compressional wave in the wall
    material wraps exactly once around the circumference. The wall is at its
    most transparent there, so the transmission loss of Equation (15) is
    anchored at this frequency and Equations (16b) and (22b) only ever make
    it worse.

    :param internal_diameter: :math:`D_i`, in m.
    :param pipe_sound_speed: :math:`c_p`, 5 000 m/s for steel, in m/s.
    :return: :math:`f_r`, in Hz.
    :raises ValueError: If a value is not positive and finite.
    """
    bore = require_positive(internal_diameter, "internal_diameter")
    wall = require_positive(pipe_sound_speed, "pipe_sound_speed")
    return float(wall / (math.pi * bore))


def reference_transmission_loss(  # noqa: PLR0913
    internal_diameter: float,
    wall_thickness: float,
    *,
    pipe_density: float,
    pipe_sound_speed: float = PIPE_SOUND_SPEED_M_S,
    air_density: float = AIR_DENSITY_KG_M3,
    air_sound_speed: float = AIR_SOUND_SPEED_M_S,
) -> float:
    r"""Equation (15): the transmission loss at the ring frequency.

    .. math::

       TL_{fr} = -10 - 10 \lg\left(
         \frac{c_p \rho_p t_p}{c_o \rho_o D_i}\right)

    A mass law written as a ratio of two impedances: the wall's, per unit
    area, against the air's, scaled by how much wall there is per unit bore.
    Both terms are negative, and the standard keeps them that way, so this
    quantity is a **negative number that is added** to the internal level all
    the way to Equation (18). A DN 100 steel pipe with a 3,6 mm wall comes
    out at −44,7 dB.

    :param internal_diameter: :math:`D_i`, in m.
    :param wall_thickness: :math:`t_p`, in m.
    :param pipe_density: :math:`\rho_p`, 7 800 kg/m³ for steel.
    :param pipe_sound_speed: :math:`c_p`, 5 000 m/s for steel, in m/s.
    :param air_density: :math:`\rho_o` outside the pipe, in kg/m³.
    :param air_sound_speed: :math:`c_o` outside the pipe, in m/s.
    :return: :math:`TL_{fr}`, in dB, negative.
    :raises ValueError: If a value is not positive and finite.
    """
    bore = require_positive(internal_diameter, "internal_diameter")
    thickness = require_positive(wall_thickness, "wall_thickness")
    rho_pipe = require_positive(pipe_density, "pipe_density")
    wall = require_positive(pipe_sound_speed, "pipe_sound_speed")
    rho_air = require_positive(air_density, "air_density")
    air = require_positive(air_sound_speed, "air_sound_speed")
    return float(
        _RING_LOSS_OFFSET
        - 10.0 * math.log10(wall * rho_pipe * thickness / (air * rho_air * bore))
    )


def transmission_loss_correction(
    frequency: ArrayLike, ring: float
) -> NDArray[np.float64]:
    r"""Equations (16b) and (22b): how far the wall is from its ring.

    .. math::

       \Delta TL(f) = -20 \log\left[
         \left(\frac{f_r}{f}\right)
         + \left(\frac{f}{f_r}\right)^{1,5}\right]

    One expression covers both printed equations: (16b) evaluates it at the
    turbulent peak frequency and (22b) at each band. The bracket is a sum of
    two branches, one falling as :math:`1/f` and one rising as
    :math:`f^{1,5}`, so the correction is worst far from the ring frequency
    on either side. It is never zero: where the two branches together are
    smallest, at :math:`(2/3)^{0,4} f_r = 0{,}85 f_r`, the bracket is still
    1,96 and the correction still costs 5,85 dB, and at :math:`f_r` itself it
    costs 6,02.

    :param frequency: :math:`f`, in Hz. A scalar or a 1-D array.
    :param ring: :math:`f_r` from :func:`pipe_ring_frequency`, in Hz.
    :return: :math:`\Delta TL`, in dB, one value per frequency, and always
        negative: the correction is worth at least 5,85 dB even where it is
        smallest.
    :raises ValueError: If a value is not positive and finite.
    """
    bands = require_positive_array(frequency, "frequency")
    anchor = require_positive(ring, "ring")
    bracket = (anchor / bands) + (bands / anchor) ** _LOSS_CORRECTION_EXPONENT
    return np.asarray(
        _LOSS_CORRECTION_COEFFICIENT * np.log10(bracket), dtype=np.float64
    )


def cavitation_transmission_loss(
    turbulent_loss: float,
    *,
    turbulent_peak: float,
    cavitation_peak: float,
    efficiency_ratio: float,
    pressure_ratio: float | None = None,
    corrected_ratio: float | None = None,
) -> float:
    r"""Equation (17): the transmission loss once the valve cavitates.

    .. math::

       TL_{cav} = TL_{turb} + 10 \lg\left(
         250\, \frac{f_{p,cav}^{1,5}}{f_{p,turb}^{2}}\,
         \frac{\eta_{cav}}{\eta_{turb} + \eta_{cav}}\right)

    Cavitation noise peaks higher in frequency than turbulent noise, and the
    pipe wall passes high frequencies better, so the correction is normally
    positive: the wall becomes *less* effective when the valve cavitates,
    which is one reason cavitating valves are heard from far away.

    The NOTE to the equation floors the efficiency ratio at
    :math:`f_{p,turb}^2/(250 f_{p,cav}^{1,5})` while :math:`x_F` is within
    0,1 of the threshold, which is exactly the value that makes the bracket
    equal 1. Just above incipient cavitation, where the cavitating efficiency
    is still a small fraction of the total, the floor therefore keeps the
    cavitating transmission loss from falling below the turbulent one. Pass
    both ratios to apply it; leave them out to evaluate the equation as
    printed.

    :param turbulent_loss: :math:`TL_{turb}` of Equation (16a), in dB.
    :param turbulent_peak: :math:`f_{p,turb}`, in Hz.
    :param cavitation_peak: :math:`f_{p,cav}`, in Hz.
    :param efficiency_ratio: :math:`\eta_{cav}/(\eta_{turb}+\eta_{cav})`.
    :param pressure_ratio: :math:`x_F`, for the NOTE's floor.
    :param corrected_ratio: :math:`x_{Fzp1}`, for the NOTE's floor.
    :return: :math:`TL_{cav}`, in dB, negative.
    :raises ValueError: If a value is not positive and finite, or only one of
        the two ratios the floor needs was given.
    """
    if not math.isfinite(turbulent_loss):
        msg = "'turbulent_loss' must be a finite level in dB."
        raise ValueError(msg)
    turbulent = require_positive(turbulent_peak, "turbulent_peak")
    cavitating = require_positive(cavitation_peak, "cavitation_peak")
    ratio = _require_share(efficiency_ratio, "efficiency_ratio")
    if (pressure_ratio is None) != (corrected_ratio is None):
        msg = (
            "The NOTE's floor needs both 'pressure_ratio' and "
            "'corrected_ratio', or neither."
        )
        raise ValueError(msg)
    group = _CAVITATION_TRANSMISSION_FACTOR * cavitating**1.5 / turbulent**2
    if pressure_ratio is not None and corrected_ratio is not None:
        operating = require_positive(pressure_ratio, "pressure_ratio")
        threshold = _require_threshold(corrected_ratio, "corrected_ratio")
        if threshold < operating < threshold + CAVITATION_FLOOR_WIDTH:
            ratio = max(ratio, 1.0 / group)
    return float(turbulent_loss + 10.0 * math.log10(group * ratio))


def _distribution(
    frequency: ArrayLike,
    peak: float,
    shape: tuple[float, float, float],
) -> NDArray[np.float64]:
    """The common shape of Equations (20a) and (20b)."""
    bands = require_positive_array(frequency, "frequency")
    centre = require_positive(peak, "peak")
    rising, falling, offset = shape
    ratio = bands / centre
    bracket = _DISTRIBUTION_PREFACTOR * ratio**rising + ratio**falling
    return np.asarray(-10.0 * np.log10(bracket) - offset, dtype=np.float64)


def turbulent_distribution(frequency: ArrayLike, peak: float) -> NDArray[np.float64]:
    r"""Equation (20a): how turbulent noise spreads over the bands.

    .. math::

       F_{turb}(f_i) = -10 \lg\left[
         \frac{1}{4}\left(\frac{f_i}{f_{p,turb}}\right)^{3}
         + \left(\frac{f_i}{f_{p,turb}}\right)^{-1}\right] - 3{,}1

    A band correction, in dB, that adds to the overall internal level. The
    two terms in the bracket are the two sides of the peak: below it the
    :math:`f^{-1}` term dominates and the level rises at 3 dB per octave;
    above it the :math:`f^{3}` term takes over and the level falls at 9 dB
    per octave. The trailing 3,1 dB is a printed offset and not a
    normalisation: over the band set of 5.4.1 these corrections do not sum
    back to :math:`L_{pi}`, they sum about 5 dB above it, so the band route
    and the overall route of Equation (18a) are two answers and not one
    answer twice. The maximum is not exactly at :math:`f_{p,turb}` either:
    the quarter in front of the rising branch puts it at
    :math:`(4/3)^{1/4} f_{p,turb}`, a few per cent above.

    The negative exponent is easy to lose. Text extracted from the printed
    page renders it as a bare 1, which flattens the low-frequency side.

    :param frequency: :math:`f_i`, the band centres, in Hz.
    :param peak: :math:`f_{p,turb}` from :func:`turbulent_peak_frequency`, in Hz.
    :return: :math:`F_{turb}(f_i)`, in dB.
    :raises ValueError: If a value is not positive and finite.
    """
    return _distribution(frequency, peak, _TURBULENT_DISTRIBUTION)


def cavitation_distribution(frequency: ArrayLike, peak: float) -> NDArray[np.float64]:
    r"""Equation (20b): how cavitation noise spreads over the bands.

    .. math::

       F_{cav}(f_i) = -10 \lg\left[
         \frac{1}{4}\left(\frac{f_i}{f_{p,cav}}\right)^{1,5}
         + \left(\frac{f_i}{f_{p,cav}}\right)^{-1,5}\right] - 3{,}5

    The same shape as Equation (20a) with every numeral changed. Both
    exponents are :math:`\pm 1,5` instead of 3 and −1, so both flanks fall at
    the same 4,5 dB per octave and the hump is symmetric, but about
    :math:`\sqrt[3]{4}\, f_{p,cav}`, two thirds of an octave above the
    frequency Equation (13) names, because the quarter in front of the
    rising branch shifts the maximum up. Against Equation (20a)'s 3 dB up and 9 dB down
    that is a far broader spectrum: cavitation is heard as a wide band of
    gravel where turbulence is heard as a hiss around one frequency.

    :param frequency: :math:`f_i`, the band centres, in Hz.
    :param peak: :math:`f_{p,cav}` from :func:`cavitation_peak_frequency`,
        in Hz.
    :return: :math:`F_{cav}(f_i)`, in dB.
    :raises ValueError: If a value is not positive and finite.
    """
    return _distribution(frequency, peak, _CAVITATION_DISTRIBUTION)


def band_internal_levels(
    frequency: ArrayLike,
    internal_level: float,
    *,
    turbulent_peak: float,
    cavitation_peak: float | None = None,
    cavitation_fraction: float = 0.0,
) -> NDArray[np.float64]:
    r"""Equations (19a) and (19b): the internal level, band by band.

    .. math::

       L_{pi}(f_i) = L_{pi} + F_{turb}(f_i)

    .. math::

       L_{pi}(f_i) = L_{pi} + 10 \lg\left(
         \frac{\eta_{turb}}{\eta_{turb}+\eta_{cav}} 10^{0,1 F_{turb}(f_i)}
         + \frac{\eta_{cav}}{\eta_{turb}+\eta_{cav}} 10^{0,1 F_{cav}(f_i)}
         \right)

    The cavitating form is the turbulent and the cavitating spectra added in
    energy, each weighted by the share of the sound power its own efficiency
    accounts for. Since the two peak frequencies differ by a factor of a few,
    the sum is a two-humped spectrum, and which hump is taller is decided by
    :math:`\eta_{cav}/(\eta_{turb}+\eta_{cav})` alone.

    :param frequency: :math:`f_i`, the band centres, in Hz.
    :param internal_level: :math:`L_{pi}` of Equation (10), in dB.
    :param turbulent_peak: :math:`f_{p,turb}`, in Hz.
    :param cavitation_peak: :math:`f_{p,cav}`, in Hz, or ``None`` for the
        turbulent branch.
    :param cavitation_fraction:
        :math:`\eta_{cav}/(\eta_{turb}+\eta_{cav})`, between 0 and 1. Zero
        gives Equation (19a) whatever else is passed.
    :return: :math:`L_{pi}(f_i)`, in dB, one value per band.
    :raises ValueError: If a value is out of range, or the cavitating branch
        was asked for without its peak frequency.
    """
    if not 0.0 <= cavitation_fraction <= 1.0 or not math.isfinite(cavitation_fraction):
        msg = (
            "'cavitation_fraction' is a share of the sound power, so it must "
            f"be between 0 and 1; got {cavitation_fraction!r}."
        )
        raise ValueError(msg)
    if not math.isfinite(internal_level):
        msg = "'internal_level' must be a finite level in dB."
        raise ValueError(msg)
    turbulent = turbulent_distribution(frequency, turbulent_peak)
    if cavitation_fraction <= 0.0:
        return np.asarray(internal_level + turbulent, dtype=np.float64)
    if cavitation_peak is None:
        msg = (
            "Equation (19b) needs 'cavitation_peak' as well as a non-zero "
            "'cavitation_fraction'."
        )
        raise ValueError(msg)
    cavitating = cavitation_distribution(frequency, cavitation_peak)
    mixed = (1.0 - cavitation_fraction) * 10.0 ** (
        0.1 * turbulent
    ) + cavitation_fraction * 10.0 ** (0.1 * cavitating)
    return np.asarray(internal_level + 10.0 * np.log10(mixed), dtype=np.float64)


def _spreading(internal_diameter: float, wall_thickness: float) -> float:
    r"""The geometric term of Equations (18a), (18b) and (21).

    .. math::

       10 \lg\left(\frac{D_i + 2 t_p + 2}{D_i + 2 t_p}\right)

    The bare 2 in the numerator is twice the 1 m measuring distance, in
    metres, so both diameters have to be in metres for the bracket to mean
    anything. For a DN 100 pipe it is 12,7 dB.
    """
    outside = internal_diameter + 2.0 * wall_thickness
    return float(10.0 * math.log10((outside + 2.0 * _MEASUREMENT_DISTANCE_M) / outside))


@dataclass(frozen=True)
class HydrodynamicValveNoise:
    r"""What IEC 60534-8-4 says about one operating point on a liquid line.

    :ivar regime: ``"turbulent"`` or ``"cavitating"``, from the test of 5.1:
        the valve cavitates when :math:`p_1 - p_2` exceeds
        :math:`x_{Fzp1}(p_1 - p_v)`.
    :ivar pressure_ratio: :math:`x_F` of Equation (1).
    :ivar differential: :math:`p_1 - p_2`, in Pa.
    :ivar cavitation_differential: :math:`\Delta p_c` of Equation (2), in Pa.
        It stops following the differential once the flow chokes.
    :ivar incipient_ratio: :math:`x_{Fz}`, the threshold as given, at
        6 × 10⁵ Pa.
    :ivar corrected_ratio: :math:`x_{Fzp1}` of Equation (3c), the threshold
        at the working inlet pressure. This is the number the regime test is
        made against.
    :ivar jet_diameter: :math:`D_j` of Equation (4), in m.
    :ivar velocity: :math:`U_{vc}` of Equation (5), in m/s.
    :ivar stream_power: :math:`W_m` of Equation (6), in W.
    :ivar turbulent_efficiency: :math:`\eta_{turb}` of Equation (8).
    :ivar cavitation_efficiency: :math:`\eta_{cav}` of Equation (9), or
        ``None`` in the turbulent regime.
    :ivar sound_power: :math:`W_a` of Equation (7a) or (7b), in W.
    :ivar internal_level: :math:`L_{pi}` of Equation (10), in dB.
    :ivar strouhal_number: :math:`N_{STR}` of Equation (12).
    :ivar turbulent_peak: :math:`f_{p,turb}` of Equation (11), in Hz.
    :ivar cavitation_peak: :math:`f_{p,cav}` of Equation (13), in Hz, or
        ``None`` in the turbulent regime.
    :ivar pipe_ring_frequency: :math:`f_r` of Equation (14), in Hz.
    :ivar reference_transmission_loss: :math:`TL_{fr}` of Equation (15), in
        dB, negative.
    :ivar turbulent_transmission_loss: :math:`TL_{turb}` of Equation (16a),
        in dB.
    :ivar cavitation_transmission_loss: :math:`TL_{cav}` of Equation (17),
        in dB, or ``None`` in the turbulent regime.
    :ivar transmission_loss: whichever of the two the regime calls for,
        which is what Equation (18a) or (18b) uses.
    :ivar external_level: :math:`L_{pAe,1m}` of Equation (18a) or (18b), in
        dB at 1 m from the pipe wall. The standard calls it A-weighted, but
        neither equation applies a weighting: the label describes what the
        fit was made against, not an operation on this number.
    :ivar frequency: The band centres of 5.4.1, in Hz.
    :ivar band_internal_level: :math:`L_{pi}(f_i)` of Equation (19a) or
        (19b), in dB.
    :ivar band_transmission_loss: :math:`TL(f_i)` of Equation (22a), in dB.
    :ivar band_external_level: :math:`L_{pe,1m}(f_i)` of Equation (21), in
        dB, unweighted.
    """

    regime: str
    pressure_ratio: float
    differential: float
    cavitation_differential: float
    incipient_ratio: float
    corrected_ratio: float
    jet_diameter: float
    velocity: float
    stream_power: float
    turbulent_efficiency: float
    cavitation_efficiency: float | None
    sound_power: float
    internal_level: float
    strouhal_number: float
    turbulent_peak: float
    cavitation_peak: float | None
    pipe_ring_frequency: float
    reference_transmission_loss: float
    turbulent_transmission_loss: float
    cavitation_transmission_loss: float | None
    transmission_loss: float
    external_level: float
    frequency: NDArray[np.float64]
    band_internal_level: NDArray[np.float64]
    band_transmission_loss: NDArray[np.float64]
    band_external_level: NDArray[np.float64]


def _default_bands() -> NDArray[np.float64]:
    """The one-third-octave band centres 5.4.1 prints, 50 Hz to 20 kHz."""
    from ..filters.frequencies import normalized_frequencies

    bands = np.asarray(normalized_frequencies(3), dtype=np.float64)
    low, high = _BAND_RANGE_HZ
    return np.asarray(bands[(bands >= low) & (bands <= high)], dtype=np.float64)


def valve_hydrodynamic_noise(  # noqa: PLR0913
    *,
    mass_flow: float,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
    liquid_density: float,
    liquid_sound_speed: float,
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    incipient_ratio: float,
    power_ratio: float,
    valve_diameter: float,
    seat_diameter: float,
    internal_diameter: float,
    wall_thickness: float,
    pipe_density: float,
    coefficient: str = "Cv",
    strouhal_form: str = "annex",
    frequency: ArrayLike | None = None,
    pipe_sound_speed: float = PIPE_SOUND_SPEED_M_S,
    air_density: float = AIR_DENSITY_KG_M3,
    air_sound_speed: float = AIR_SOUND_SPEED_M_S,
) -> HydrodynamicValveNoise:
    r"""The whole of Clauses 4 and 5, from the operating point to 1 m.

    The chain is the standard's own: the pressure ratios of 4.1 and 4.2, the
    geometry and the stream power of 4.4 to 4.6, the regime test and the two
    efficiencies of 5.1, the pipe transmission loss of 5.2, the external
    level of 5.3, and the band route of 5.4 alongside it.

    Which regime the valve is in is decided once, on
    :math:`p_1 - p_2` against :math:`x_{Fzp1}(p_1 - p_v)`, and it selects the
    sound power of Equation (7a) or (7b), the transmission loss of (16a) or
    (17), the external level of (18a) or (18b), and the band spectrum of
    (19a) or (19b) together. On the threshold itself Equation (9) returns
    exactly zero, so the two branches meet without a step.

    :param mass_flow: :math:`\dot m`, in kg/s.
    :param inlet_pressure: :math:`p_1`, absolute, in Pa.
    :param outlet_pressure: :math:`p_2`, absolute, in Pa.
    :param vapour_pressure: :math:`p_v` of the liquid, absolute, in Pa.
    :param liquid_density: :math:`\rho_L`, in kg/m³.
    :param liquid_sound_speed: :math:`c_L`, in m/s.
    :param flow_coefficient: :math:`C` at the travel being examined.
    :param style_modifier: :math:`F_d`, from IEC 60534-8-3.
    :param pressure_recovery: :math:`F_L`, dimensionless.
    :param incipient_ratio: :math:`x_{Fz}` at 6 × 10⁵ Pa, measured to
        IEC 60534-8-2 or estimated with
        :func:`incipient_cavitation_ratio`. Equation (3c) corrects it here.
    :param power_ratio: :math:`r_W` from Table 2, the share of the sound
        power radiated into the pipe. See :data:`ACOUSTIC_POWER_RATIOS`.
    :param valve_diameter: :math:`d`, the valve inlet internal diameter, in
        m.
    :param seat_diameter: :math:`d_o`, in m.
    :param internal_diameter: :math:`D_i` of the downstream pipe, in m.
    :param wall_thickness: :math:`t_p`, in m.
    :param pipe_density: :math:`\rho_p`, in kg/m³.
    :param coefficient: ``"Cv"`` or ``"Kv"``.
    :param strouhal_form: Which printing of Equation (12) to follow,
        ``"annex"`` or ``"clause"``; see :data:`STROUHAL_CONSTANTS`.
    :param frequency: The band centres to report, in Hz. The default is the
        one-third-octave set 5.4.1 prints, 50 Hz to 20 kHz.
    :param pipe_sound_speed: :math:`c_p`, in m/s.
    :param air_density: :math:`\rho_o`, in kg/m³.
    :param air_sound_speed: :math:`c_o`, in m/s.
    :return: A :class:`HydrodynamicValveNoise` carrying every printed
        intermediate as well as the level at 1 m.
    :raises ValueError: If a value is outside the range its equation is
        written for, or if the operating point is at or past flashing, where
        Equations (9) and (13) divide by zero.
    """
    p1 = require_positive(inlet_pressure, "inlet_pressure")
    p2 = require_positive(outlet_pressure, "outlet_pressure")
    pv = require_positive(vapour_pressure, "vapour_pressure")
    rho = require_positive(liquid_density, "liquid_density")
    sonic = require_positive(liquid_sound_speed, "liquid_sound_speed")
    recovery = _require_recovery(pressure_recovery)
    share = _require_share(power_ratio, "power_ratio")
    bore = require_positive(internal_diameter, "internal_diameter")
    thickness = require_positive(wall_thickness, "wall_thickness")

    ratio = differential_pressure_ratio(
        inlet_pressure=p1, outlet_pressure=p2, vapour_pressure=pv
    )
    if ratio >= _FLASHING_RATIO:
        msg = (
            "At x_F = 1 the outlet is at the vapour pressure and the liquid "
            "flashes; Equation (9) divides by 1 - x_F, so the method stops "
            f"below it. Got x_F = {ratio:.3f}."
        )
        raise ValueError(msg)
    differential = p1 - p2
    choked = cavitation_differential(
        inlet_pressure=p1,
        outlet_pressure=p2,
        vapour_pressure=pv,
        pressure_recovery=recovery,
    )
    threshold = corrected_incipient_ratio(incipient_ratio, p1)
    cavitating = differential > threshold * (p1 - pv)

    jet = jet_diameter(
        flow_coefficient, style_modifier, recovery, coefficient=coefficient
    )
    velocity = vena_contracta_velocity(choked, rho, recovery)
    stream_power = mechanical_stream_power(mass_flow, velocity, recovery)
    turbulent = turbulent_efficiency(velocity, sonic)
    cavitation = (
        cavitation_efficiency(
            turbulent=turbulent,
            differential=differential,
            choked_differential=choked,
            pressure_ratio=ratio,
            corrected_ratio=threshold,
        )
        if cavitating
        else None
    )
    total_efficiency = turbulent + (cavitation or 0.0)
    sound_power = total_efficiency * stream_power * share
    internal = internal_sound_pressure_level(
        sound_power=sound_power,
        density=rho,
        sound_speed=sonic,
        internal_diameter=bore,
    )

    strouhal = jet_strouhal_number(
        flow_coefficient=flow_coefficient,
        style_modifier=style_modifier,
        pressure_recovery=recovery,
        corrected_ratio=threshold,
        valve_diameter=valve_diameter,
        seat_diameter=seat_diameter,
        inlet_pressure=p1,
        vapour_pressure=pv,
        coefficient=coefficient,
        form=strouhal_form,
    )
    turbulent_peak = turbulent_peak_frequency(strouhal, velocity, jet)
    cavitation_peak = (
        cavitation_peak_frequency(turbulent_peak, ratio, threshold)
        if cavitating
        else None
    )

    ring = pipe_ring_frequency(bore, pipe_sound_speed=pipe_sound_speed)
    reference = reference_transmission_loss(
        bore,
        thickness,
        pipe_density=pipe_density,
        pipe_sound_speed=pipe_sound_speed,
        air_density=air_density,
        air_sound_speed=air_sound_speed,
    )
    turbulent_loss = reference + float(
        transmission_loss_correction(turbulent_peak, ring)[0]
    )
    fraction = 0.0 if cavitation is None else cavitation / total_efficiency
    cavitation_loss = (
        cavitation_transmission_loss(
            turbulent_loss,
            turbulent_peak=turbulent_peak,
            cavitation_peak=cavitation_peak,
            efficiency_ratio=fraction,
            pressure_ratio=ratio,
            corrected_ratio=threshold,
        )
        if cavitating and cavitation_peak is not None and fraction > 0.0
        else None
    )
    loss = turbulent_loss if cavitation_loss is None else cavitation_loss

    spreading = _spreading(bore, thickness)
    bands = (
        _default_bands()
        if frequency is None
        else require_positive_array(frequency, "frequency")
    )
    band_internal = band_internal_levels(
        bands,
        internal,
        turbulent_peak=turbulent_peak,
        cavitation_peak=cavitation_peak,
        cavitation_fraction=fraction,
    )
    band_loss = reference + transmission_loss_correction(bands, ring)
    return HydrodynamicValveNoise(
        regime="cavitating" if cavitating else "turbulent",
        pressure_ratio=float(ratio),
        differential=float(differential),
        cavitation_differential=float(choked),
        incipient_ratio=float(incipient_ratio),
        corrected_ratio=float(threshold),
        jet_diameter=float(jet),
        velocity=float(velocity),
        stream_power=float(stream_power),
        turbulent_efficiency=float(turbulent),
        cavitation_efficiency=None if cavitation is None else float(cavitation),
        sound_power=float(sound_power),
        internal_level=float(internal),
        strouhal_number=float(strouhal),
        turbulent_peak=float(turbulent_peak),
        cavitation_peak=None if cavitation_peak is None else float(cavitation_peak),
        pipe_ring_frequency=float(ring),
        reference_transmission_loss=float(reference),
        turbulent_transmission_loss=float(turbulent_loss),
        cavitation_transmission_loss=(
            None if cavitation_loss is None else float(cavitation_loss)
        ),
        transmission_loss=float(loss),
        external_level=float(internal + loss - spreading),
        frequency=bands,
        band_internal_level=band_internal,
        band_transmission_loss=np.asarray(band_loss, dtype=np.float64),
        band_external_level=np.asarray(
            band_internal + band_loss - spreading, dtype=np.float64
        ),
    )


# --------------------------------------------------------------------------
# 6 Multistage trim
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageConditions:
    r"""What one throttling stage of a multistage trim sees.

    :ivar inlet_pressure: :math:`p_{1,i}` of Equations (23a) and (23b), in
        Pa.
    :ivar outlet_pressure: :math:`p_{2,i}` of Equations (24a) and (24b), in
        Pa.
    :ivar pressure_ratio: :math:`x_{F,i}` of Equation (26), the stage's own
        differential pressure ratio, which 6.3 tests against that stage's
        :math:`x_{Fzp1,i}`.
    """

    inlet_pressure: float
    outlet_pressure: float
    pressure_ratio: float


def stage_conditions(
    *,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
    stage_coefficients: Sequence[float],
    flow_coefficient: float,
) -> tuple[StageConditions, ...]:
    r"""Equations (23a) to (24b) and (26): the differential, stage by stage.

    .. math::

       p_{1,1} = p_1, \qquad
       p_{1,i} = p_{1,i-1} - \frac{p_1 - p_2}{\left(C_{i-1}/C\right)^2},
       \qquad
       p_{2,i} = p_{1,i+1}, \qquad p_{2,n} = p_2

    Each stage takes a share of the total differential in inverse proportion
    to the square of its own capacity, which is the series law for flow
    resistances: :math:`1/C^2 = \sum_i 1/C_i^2`. A trim whose stages all have
    the same :math:`C_i` splits the drop evenly; one with an increasing flow
    area, the device of 6.3.2 and Figure 2, puts most of the drop in the
    first stages and leaves the last one working at a differential small
    enough not to cavitate.

    Equation (23b) is printed with :math:`p_{1,i+1}` on the right, which
    would compute each stage's inlet from the **next** stage's and run the
    pressure backwards along the trim, contradicting (23a). The recursion
    implemented here is the forward one the index :math:`C_{i-1}` calls for;
    see ``docs/ERRATA.md``.

    :param inlet_pressure: :math:`p_1` at the valve, absolute, in Pa.
    :param outlet_pressure: :math:`p_2` at the valve, absolute, in Pa.
    :param vapour_pressure: :math:`p_v`, absolute, in Pa.
    :param stage_coefficients: :math:`C_i`, the rated flow coefficient of
        each stage in flow order, two or more of them.
    :param flow_coefficient: :math:`C` of the whole valve, in the same units.
    :return: One :class:`StageConditions` per stage, in flow order.
    :raises ValueError: If a value is not positive and finite, if fewer than
        two stages were given, or if the stages between them would take more
        than the differential the valve has.
    :raises ValveNoiseWarning: If the stage coefficients miss the series law
        by more than 5 %, which leaves the last stage carrying a differential
        nobody chose.
    """
    p1 = require_positive(inlet_pressure, "inlet_pressure")
    p2 = require_positive(outlet_pressure, "outlet_pressure")
    pv = require_positive(vapour_pressure, "vapour_pressure")
    capacity = require_positive(flow_coefficient, "flow_coefficient")
    stages = require_positive_array(stage_coefficients, "stage_coefficients")
    if stages.size < _MINIMUM_STAGES:
        msg = (
            "Clause 6 is about trims with more than one stage, so "
            f"'stage_coefficients' needs at least {_MINIMUM_STAGES} values; "
            f"got {stages.size}."
        )
        raise ValueError(msg)
    if p2 >= p1:
        msg = (
            "A control valve drops pressure, so 'outlet_pressure' must be "
            f"below 'inlet_pressure'; got {outlet_pressure!r} and "
            f"{inlet_pressure!r} Pa."
        )
        raise ValueError(msg)
    if pv >= p2:
        msg = (
            "Equation (26) needs every stage above the vapour pressure; got "
            f"p_2 = {outlet_pressure!r} Pa and p_v = {vapour_pressure!r} Pa."
        )
        raise ValueError(msg)
    shares = float(np.sum((capacity / stages) ** 2))
    if abs(shares - 1.0) > _SERIES_LAW_TOLERANCE:
        warnings.warn(
            "The stage coefficients account for "
            f"{shares:.0%} of the valve's own resistance, where the series "
            "law 1/C^2 = sum(1/C_i^2) asks for 100 %. Equation (24b) fixes "
            "the last stage's outlet at the valve outlet, so that stage "
            "absorbs the whole of the difference and its pressure ratio is "
            "what moves.",
            ValveNoiseWarning,
            stacklevel=2,
        )
    total = p1 - p2
    inlets = [p1]
    for previous in stages[:-1]:
        inlets.append(inlets[-1] - total / (previous / capacity) ** 2)
    if inlets[-1] <= p2:
        msg = (
            "The stage coefficients take more than the valve's differential: "
            f"the last stage would start at {inlets[-1]:.0f} Pa, at or below "
            f"the outlet pressure {outlet_pressure!r} Pa. The series law "
            "1/C^2 = sum(1/C_i^2) says each C_i is larger than C."
        )
        raise ValueError(msg)
    outlets = [*inlets[1:], p2]
    return tuple(
        StageConditions(
            inlet_pressure=float(stage_inlet),
            outlet_pressure=float(stage_outlet),
            pressure_ratio=float((stage_inlet - stage_outlet) / (stage_inlet - pv)),
        )
        for stage_inlet, stage_outlet in zip(inlets, outlets, strict=True)
    )


def combine_stage_levels(*levels: float) -> float:
    r"""Equation (27): the stages of a multistage trim, added.

    .. math::

       L_{pAe,1m} = 10 \lg \sum_{i=1}^{n} 10^{0,1 L_{pAe,1m,i}}

    6.3.1 calculates each stage as if it were a valve of its own and sums
    them in energy here. That is the branch for a trim whose stages all
    radiate into the pipe, Figures 1 and 3; the fixed device of 6.3.2 with
    increasing flow areas does not use it, because everything but the last
    stage is absorbed inside the trim.

    :param levels: :math:`L_{pAe,1m,i}`, one per stage, in dB.
    :return: Their energy sum, in dB.
    :raises ValueError: If fewer than two levels are given, or one is not
        finite.
    """
    if len(levels) < _MINIMUM_STAGES:
        msg = (
            "Equation (27) sums the stages of a multistage trim, so it needs "
            f"at least {_MINIMUM_STAGES} levels; got {len(levels)}."
        )
        raise ValueError(msg)
    values = require_finite_array(list(levels), "levels")
    return float(10.0 * np.log10(np.sum(10.0 ** (values / 10.0))))


def last_stage_differential(
    *,
    inlet_pressure: float,
    outlet_pressure: float,
    vapour_pressure: float,
    corrected_ratio: float,
) -> float:
    r"""Equation (28): the differential of the last stage of a fixed device.

    .. math::

       \Delta p_c = \min\left[
         (p_{1,n} - p_2),\; x_{Fzp1,n}(p_{1,n} - p_v)\right]

    This is **not** Equation (2) with different symbols. Equation (2) caps
    the differential at the choking point, :math:`F_L^2(p_1 - p_v)`; this one
    caps it at the *cavitation threshold* of the last stage,
    :math:`x_{Fzp1,n}(p_{1,n} - p_v)`, which is a smaller number. A fixed
    multistage device is designed so that the last stage never cavitates, and
    the cap says so.

    :param inlet_pressure: :math:`p_{1,n}` of the last stage, in Pa.
    :param outlet_pressure: :math:`p_2` at the valve outlet, in Pa.
    :param vapour_pressure: :math:`p_v`, in Pa.
    :param corrected_ratio: :math:`x_{Fzp1,n}` of the last stage.
    :return: :math:`\Delta p_c`, in Pa.
    :raises ValueError: If a value is not positive and finite, or the last
        stage does not drop pressure.
    """
    p1n = require_positive(inlet_pressure, "inlet_pressure")
    p2 = require_positive(outlet_pressure, "outlet_pressure")
    pv = require_positive(vapour_pressure, "vapour_pressure")
    threshold = _require_threshold(corrected_ratio, "corrected_ratio")
    if p2 >= p1n:
        msg = (
            "The last stage drops pressure, so 'outlet_pressure' must be "
            f"below 'inlet_pressure'; got {outlet_pressure!r} and "
            f"{inlet_pressure!r} Pa."
        )
        raise ValueError(msg)
    if pv >= p1n:
        msg = (
            "Equation (28) needs the last stage above the vapour pressure; "
            f"got p_1n = {inlet_pressure!r} Pa and p_v = "
            f"{vapour_pressure!r} Pa."
        )
        raise ValueError(msg)
    return float(min(p1n - p2, threshold * (p1n - pv)))


def uniform_passage_style_modifier(passages: int) -> float:
    r"""Equation (29): the style modifier of a last stage full of openings.

    .. math::

       F_d = \sqrt{\frac{1}{N_o}}

    IEC 60534-8-3 defines :math:`F_d` as the hydraulic diameter of one
    passage over the diameter of the single orifice of the same total area.
    For :math:`N_o` identical round openings that ratio collapses to
    :math:`1/\sqrt{N_o}`, which is what this equation prints. Sixteen
    openings therefore give a quarter of the jet diameter, a sixteenth of the
    jet area, and a peak frequency four times higher.

    :param passages: :math:`N_o`, the number of uniform openings within the
        last stage.
    :return: :math:`F_d`, dimensionless.
    :raises ValueError: If the count is not a whole number of one or more.
    """
    count = _require_count(passages, "passages")
    return float(math.sqrt(1.0 / count))


def last_stage_seat_diameter_mm(
    flow_coefficient: float, *, coefficient: str = "Cv"
) -> float:
    r"""6.3.2 b): the seat diameter of the last stage, estimated.

    .. math::

       d_o = 5{,}2 \sqrt{N_{34}\, C_n}

    The one display formula in the standard that carries no equation number,
    and the one whose unit does not survive its own arithmetic: Clause 3
    declares :math:`d_o` in metres, and for any real last stage this returns
    tens. It is millimetres, which is why the unit is in the name of this
    function; see ``docs/ERRATA.md``. Equation (12) then wants the result in
    metres, so divide by 1 000 before passing it on.

    :param flow_coefficient: :math:`C_n` of the exit stage.
    :param coefficient: ``"Cv"`` or ``"Kv"``, selecting :math:`N_{34}`.
    :return: :math:`d_o`, in **millimetres**.
    :raises ValueError: If the coefficient is not positive and finite, or is
        not one Table 1 prints a constant for.
    """
    kind = require_choice(coefficient, "coefficient", tuple(CAPACITY_SCALE_CONSTANTS))
    capacity = require_positive(flow_coefficient, "flow_coefficient")
    return float(
        _SEAT_DIAMETER_CONSTANT * math.sqrt(CAPACITY_SCALE_CONSTANTS[kind] * capacity)
    )


# --------------------------------------------------------------------------
# Shared guards
# --------------------------------------------------------------------------


def _require_recovery(value: float) -> float:
    """A liquid pressure recovery factor, which lives in ``(0, 1]``."""
    recovery = require_positive(value, "pressure_recovery")
    if recovery > 1.0:
        msg = (
            "'pressure_recovery' is the fraction of the ideal velocity head "
            f"the valve reaches, so it cannot exceed 1; got {value!r}."
        )
        raise ValueError(msg)
    return recovery


def _require_threshold(value: float, name: str) -> float:
    """A cavitation threshold, which is a differential pressure ratio below 1.

    Equation (3a) caps :math:`x_{Fz}` at 0,90 and Equation (3b) at 0,47, and
    Equation (3c) only lowers it further at any inlet pressure above
    6 × 10⁵ Pa. A threshold at or above 1 would sit past flashing, where the
    method has stopped, and it would take the bracket
    :math:`(1-x_{Fzp1})/(1-x_F)` of Equation (9) negative.
    """
    ratio = require_positive(value, name)
    if ratio >= _FLASHING_RATIO:
        msg = (
            f"'{name}' is a differential pressure ratio at which cavitation "
            "starts, which Equations (3a) and (3b) cap well below 1; got "
            f"{value!r}. Check it is a fraction and not a percentage."
        )
        raise ValueError(msg)
    return ratio


def _require_share(value: float, name: str) -> float:
    """A share of something, which lives in ``(0, 1]``.

    Table 2's acoustic power ratio and the efficiency ratio of Equation (17)
    are both fractions of a whole, and both are printed as decimals. Reading
    either as a percentage is a silent 20 dB.
    """
    share = require_positive(value, name)
    if share > 1.0:
        msg = (
            f"'{name}' is a fraction of a whole, so it cannot exceed 1; got "
            f"{value!r}. Table 2 prints quarters and halves, not percentages."
        )
        raise ValueError(msg)
    return share


def _require_count(value: int, name: str) -> int:
    """A count of passages or openings: a whole number of one or more."""
    count = int(value)
    if count != value or count < 1:
        msg = (
            f"'{name}' counts flow passages, so it must be a whole number of "
            f"one or more; got {value!r}."
        )
        raise ValueError(msg)
    return count
