#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Control valve aerodynamic noise (IEC 60534-8-3:2010).

A control valve throttles a compressible fluid by turning pressure into
velocity and then throwing that velocity away in a free jet inside the pipe.
A small, well-characterised fraction of the jet's stream power comes back as
sound, most of it radiated not by the valve but by the **pipe wall
downstream**, which is why the method ends in a transmission loss and not in a
sound power level.

The standard is a chain with a branch in the middle. The branch is the
**regime**: how far the throttling has gone, from subsonic flow in the vena
contracta (regime I) through the onset of choking to the fully developed
shock cells of regime V. Five printed pressure ratios,
Equations (3) to (7), cut the differential pressure ratio :math:`x` into
those five intervals, and Table 3 gives each one its own Mach number, its own
acoustical efficiency and its own peak frequency. Everything before the
branch (the pressure ratios, the jet diameter) and everything after it (the
internal level at the pipe wall, the pipe transmission loss, the level
outside) is common to all five.

**What is new in the 2010 edition, and what this module therefore does.** The
1997 method produced one number. This one produces a **third-octave
spectrum**: Equation (19) spreads the internal level around the peak
frequency, Equation (20a) gives the pipe a transmission loss that changes with
frequency through the ring and coincidence frequencies of Equations (21) to
(23), and only Equation (25) collapses the result back to a single A-weighted
level at 1 m. The band set is the 33 one-third-octave bands from 12,5 Hz to
20 kHz, printed as Table 5.

**Three things in Annex A do not reproduce themselves**, and all three are
recorded in ``docs/ERRATA.md``:

* The piping geometry factor is printed as :math:`F_p = 0{,}98`, but every one
  of the six printed vena contracta pressures needs :math:`0{,}984` to come
  out. The five examples that print a value of :math:`p_{vc}` all give
  :math:`(F_{LP}/F_P)^2 = 0{,}647\,83`, which is :math:`F_p = 0{,}984` to six
  digits and not :math:`0{,}98`.
* The equivalent orifice diameter is printed as :math:`d_o = 0{,}010` m in all
  six columns, where Equation (8c) with the annex's own :math:`N_O = 6` and
  :math:`A = 0{,}00137` m² gives :math:`0{,}102` m. The valve style modifier
  printed on the next row, :math:`F_d = 0{,}30`, is the ratio of the printed
  :math:`d_H = 0{,}030` m to :math:`0{,}102` m, so the annex computed with the
  larger value and printed the smaller one.
* Two frequency factors of Table A.2 are printed one power of ten low,
  :math:`G_{x,5}` and :math:`G_{x,10}`, in a column Table 6 makes
  proportional to :math:`f_i^4` and which therefore has to rise. The
  transmission losses printed two rows below them are what the corrected
  factors give.

This module implements Clause 5, the standard trim case. The noise-reducing
trims of Clause 6, the expander of Clause 7 and the hydrodynamic case of
IEC 60534-8-4 are separate.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._internal.validation import (
    require_choice,
    require_positive,
)
from .._internal.warnings import PhonometryWarning

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = [
    "AERODYNAMIC_A_WEIGHTING_DB",
    "AIR_SOUND_SPEED_M_S",
    "DEFAULT_EXPANDER",
    "EXPANDER_PIPE_MACH_LIMIT",
    "FLOW_COEFFICIENT_CONSTANTS",
    "GLOBE_CONTRACTION_COEFFICIENT",
    "MACH_LIMIT_STANDARD_TRIM",
    "PIPE_SOUND_SPEED_M_S",
    "PIPE_WALL_MACH_LIMIT",
    "REGIME_CHOKED",
    "REGIME_CONSTANT_EFFICIENCY",
    "REGIME_COUNT",
    "REGIME_SHOCK",
    "REGIME_SUBSONIC",
    "REGIME_SUPERSONIC",
    "STRUCTURAL_LOSS_REFERENCE_HZ",
    "UNIVERSAL_GAS_CONSTANT",
    "VALVE_ACOUSTIC_STYLES",
    "AerodynamicValveNoise",
    "Expander",
    "ExpanderNoise",
    "PipeFrequencies",
    "RegimeBoundaries",
    "ValveNoiseWarning",
    "coincidence_frequencies",
    "combine_internal_levels",
    "expander_noise",
    "flow_regime",
    "internal_spectrum",
    "jet_diameter",
    "pressure_ratio_boundaries",
    "pipe_transmission_loss",
    "valve_aerodynamic_noise",
    "valve_style_modifier",
]

# --------------------------------------------------------------------------
# Printed constants
# --------------------------------------------------------------------------

#: Table 1's numerical constant :math:`N_{14}` of Equation (9), keyed by the
#: flow coefficient the valve is rated in. The two figures are the same
#: constant expressed for the two coefficients, but they are rounded to two
#: digits each, so a valve rated both ways comes out with jet diameters about
#: 1 % apart rather than identical: 4,9/4,6 is 1,065 where the exact
#: :math:`K_v` to :math:`C_v` conversion asks for :math:`\sqrt{1{,}156}`,
#: which is 1,075.
FLOW_COEFFICIENT_CONSTANTS = {"Cv": 4.6e-3, "Kv": 4.9e-3}

#: The universal gas constant as the standard prints it in Clause 4, in
#: J/(kmol K). It is paired with a molecular mass in kg/kmol, which is the
#: form Equations (10) and (14) are written for.
UNIVERSAL_GAS_CONSTANT = 8314.0

#: Table 4: the valve correction factor for acoustical efficiency
#: :math:`A_\eta` and the Strouhal number at the peak frequency
#: :math:`St_p`, keyed by valve style and flow direction as the table prints
#: them. NOTE 1 of the table calls these typical only; a manufacturer states
#: the actual pair, and this module takes them as arguments rather than
#: looking them up, so the table is here to be read and not to be relied on.
VALVE_ACOUSTIC_STYLES: dict[str, tuple[float, float]] = {
    "globe parabolic plug": (-4.2, 0.19),
    "globe V-port plug": (-4.2, 0.19),
    "globe ported cage": (-3.8, 0.2),
    "globe multihole to open": (-4.8, 0.2),
    "globe multihole to close": (-4.4, 0.2),
    "butterfly eccentric": (-4.2, 0.3),
    "butterfly swing-through": (-4.2, 0.3),
    "butterfly fluted vane": (-4.2, 0.3),
    "butterfly 60 deg flat disk": (-4.2, 0.3),
    "eccentric rotary plug": (-3.6, 0.3),
    "segmented ball 90 deg": (-3.6, 0.3),
    "drilled hole plate": (-4.8, 0.2),
    "expander": (-3.0, 0.2),
}

#: Speed of sound in air of NOTE 3 to Equations (22) and (23), in m/s. It is
#: the value the coincidence frequencies are printed for, not a property of
#: the air around the pipe on the day.
AIR_SOUND_SPEED_M_S = 343.0

#: Speed of sound in the pipe wall of NOTE 4 to Equations (21) and (23), in
#: m/s, for steel. Clause 1 restricts the whole method to steel and steel
#: alloy pipes for exactly this reason.
PIPE_SOUND_SPEED_M_S = 5000.0

#: The reference frequency :math:`f_s` of Equation (20c), in Hz. The symbol
#: list gives it as 1 Hz and the equation never says so again, which makes
#: the structural loss factor look dimensionless when it is not.
STRUCTURAL_LOSS_REFERENCE_HZ = 1.0

#: NOTE 1 to Equation (15): above this Mach number at the valve outlet the
#: accuracy of Clause 5 cannot be maintained and Clause 7 is used instead.
MACH_LIMIT_STANDARD_TRIM = 0.3

#: NOTE 2 to Equation (16): the pipe Mach number is limited to this value
#: when the velocity correction is computed, however fast the pipe runs.
PIPE_WALL_MACH_LIMIT = 0.3

#: NOTE 1 to Equation (34): the downstream pipe velocity of Clause 7 is
#: capped at this Mach number however fast the pipe would otherwise run.
EXPANDER_PIPE_MACH_LIMIT = 0.8

#: The contraction coefficient of NOTE 1 to Equation (35), for straight
#: pattern globe valves. The note puts some rotary valves as low as 0,7 and
#: says there are no data for the rest, so this is a default and not a fact
#: about a valve on a bench.
GLOBE_CONTRACTION_COEFFICIENT = 0.93

#: Equation (36)'s additive constant, which keeps the stream power finite
#: when the expander opens into a pipe of its own diameter.
_EXPANDER_RESIDUAL = 0.2

#: The five regimes of Clause 5.2, by the number the clause prints. Regime I
#: is subsonic in the vena contracta, II and III are choked with a growing
#: jet, and IV and V are the shock-cell regimes where the peak frequency is
#: set by the cell spacing rather than by the jet.
REGIME_SUBSONIC = 1
REGIME_CHOKED = 2
REGIME_SUPERSONIC = 3
REGIME_SHOCK = 4
REGIME_CONSTANT_EFFICIENCY = 5
REGIME_COUNT = REGIME_CONSTANT_EFFICIENCY

#: Table 7, the A weighting at each of the 33 one-third-octave bands from
#: 12,5 Hz to 20 kHz, in dB. The standard prints its own rounded copy rather
#: than pointing at IEC 61672-1, and Equation (25) sums with these.
AERODYNAMIC_A_WEIGHTING_DB = (
    -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1,
    -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0.0,
    0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5,
    -4.3, -6.6, -9.3,
)  # fmt: skip

#: The reference sound power of NOTE 5 to Clause 4, in W.
_REFERENCE_SOUND_POWER_W = 1e-12

#: Equation (18)'s leading coefficient, which carries the reference pressure
#: and the pipe geometry together.
_INTERNAL_LEVEL_COEFFICIENT = 3.2e9

#: Equation (20a)'s leading coefficient.
_TRANSMISSION_COEFFICIENT = 8.25e-7

#: The characteristic impedance of air that Equation (20a) prints as a bare
#: 415 with no symbol and no unit, in Pa s/m.
_AIR_IMPEDANCE = 415.0

#: Equation (24) measures the level one metre from the pipe wall, so the
#: cylindrical spreading term carries one metre on each side, in m.
_MEASUREMENT_DISTANCE_M = 1.0

#: Equation (7)'s printed denominator, and the argument of the regime V Mach
#: number in Table 3. It is the same 22 in both places.
_CONSTANT_EFFICIENCY_FACTOR = 22.0

#: Equations (19)'s two exponents, printed with a decimal point in a document
#: that otherwise uses a comma.
_SPECTRUM_HIGH_EXPONENT = 2.5
_SPECTRUM_LOW_EXPONENT = 1.7

#: Equation (20b)'s two branch points on the valve outlet diameter, in m. A
#: pipe wider than the first one is not damped at all; one narrower than the
#: second takes the full 9 dB.
_UNDAMPED_OUTLET_M = 0.15
_FULLY_DAMPED_OUTLET_M = 0.05

#: Equation (43) combines at least the valve trim with the expander.
_MINIMUM_SOURCES = 2

#: The peak-frequency coefficient of regimes IV and V in Table 3.
_SHOCK_PEAK_COEFFICIENT = 1.4

#: Table 3's exponent on the jet Mach number in regimes II and III, and on
#: the root of two in regimes IV and V.
_EFFICIENCY_MACH_EXPONENT = 6.6


class ValveNoiseWarning(PhonometryWarning):
    """A valve read outside the conditions IEC 60534-8-3 prints for it."""


@dataclass(frozen=True)
class RegimeBoundaries:
    r"""The four pressure ratios that cut Clause 5.2 into five regimes.

    :ivar vena_contracta: :math:`x_{vcc}`, where the flow in the vena
        contracta first reaches the speed of sound, Equation (3).
    :ivar critical: :math:`x_C`, the same point seen from the valve inlet,
        Equation (4).
    :ivar break_point: :math:`x_B`, where the jet stops growing and shock
        cells take over, Equation (6).
    :ivar constant_efficiency: :math:`x_{CE}`, where the acoustical
        efficiency stops rising with pressure ratio, Equation (7).
    :ivar recovery: :math:`\alpha`, the recovery correction factor of
        Equation (5), which the other two are written in terms of.
    """

    vena_contracta: float
    critical: float
    break_point: float
    constant_efficiency: float
    recovery: float


def pressure_ratio_boundaries(
    specific_heat_ratio: float, pressure_recovery: float
) -> RegimeBoundaries:
    r"""The regime boundaries of Equations (3) to (7).

    .. math::

       x_{vcc} = 1 - \left(\frac{2}{\gamma + 1}\right)^{\gamma/(\gamma-1)},
       \qquad
       x_C = F_L^2\, x_{vcc},
       \qquad
       \alpha = \frac{1 - x_{vcc}}{1 - x_C}

    .. math::

       x_B = 1 - \frac{1}{\alpha}
             \left(\frac{1}{\gamma}\right)^{\gamma/(\gamma-1)},
       \qquad
       x_{CE} = 1 - \frac{1}{22\,\alpha}

    :param specific_heat_ratio: :math:`\gamma` of the flowing fluid.
    :param pressure_recovery: :math:`F_L`, or :math:`F_{LP}/F_p` when the
        valve has attached fittings, which is what the NOTE to Table 3 asks
        for and what every example in Annex A uses.
    :return: The four boundaries and the recovery factor behind two of them.
    :raises ValueError: If either argument is not positive and finite, or if
        the specific heat ratio is not above one.
    """
    gamma = require_positive(specific_heat_ratio, "specific_heat_ratio")
    if not math.isfinite(gamma) or gamma <= 1.0:
        msg = (
            "'specific_heat_ratio' is the ratio of specific heats of a gas, "
            f"so it must be a finite number above 1; got {gamma!r}."
        )
        raise ValueError(msg)
    recovery_factor = require_positive(pressure_recovery, "pressure_recovery")
    if not math.isfinite(recovery_factor):
        msg = "'pressure_recovery' must be a positive, finite factor."
        raise ValueError(msg)

    exponent = gamma / (gamma - 1.0)
    vena_contracta = 1.0 - (2.0 / (gamma + 1.0)) ** exponent
    critical = recovery_factor**2 * vena_contracta
    alpha = (1.0 - vena_contracta) / (1.0 - critical)
    break_point = 1.0 - (1.0 / alpha) * (1.0 / gamma) ** exponent
    constant_efficiency = 1.0 - 1.0 / (_CONSTANT_EFFICIENCY_FACTOR * alpha)
    return RegimeBoundaries(
        vena_contracta=float(vena_contracta),
        critical=float(critical),
        break_point=float(break_point),
        constant_efficiency=float(constant_efficiency),
        recovery=float(alpha),
    )


def flow_regime(pressure_ratio: float, boundaries: RegimeBoundaries) -> int:
    r"""Which of the five regimes of Clause 5.2 a pressure ratio falls in.

    The clause prints the five intervals half open, each one closed at the
    top: :math:`x \le x_C`, then :math:`x_C < x \le x_{vcc}`, then
    :math:`x_{vcc} < x \le x_B`, then :math:`x_B < x \le x_{CE}`, and finally
    :math:`x_{CE} < x`.

    Table 3 prints the last one as :math:`x_{CE} \le x`, which would put the
    single point :math:`x = x_{CE}` in two regimes at once. Clause 5.2 is the
    normative text and its list is consistent, so this follows the clause;
    ``docs/ERRATA.md`` records the disagreement.

    :param pressure_ratio: :math:`x` of Equation (1).
    :param boundaries: The output of :func:`pressure_ratio_boundaries`.
    :return: The regime number, 1 to 5.
    :raises ValueError: If the pressure ratio is not a finite number in
        (0, 1).
    """
    x = float(pressure_ratio)
    if not math.isfinite(x) or not 0.0 < x < 1.0:
        msg = (
            "'pressure_ratio' is (p_1 - p_2)/p_1, so it must be a finite "
            f"number strictly between 0 and 1; got {pressure_ratio!r}."
        )
        raise ValueError(msg)
    if x <= boundaries.critical:
        return REGIME_SUBSONIC
    if x <= boundaries.vena_contracta:
        return REGIME_CHOKED
    if x <= boundaries.break_point:
        return REGIME_SUPERSONIC
    if x <= boundaries.constant_efficiency:
        return REGIME_SHOCK
    return REGIME_CONSTANT_EFFICIENCY


def valve_style_modifier(
    passage_area: float, wetted_perimeter: float, passages: int
) -> float:
    r"""The valve style modifier of Equations (8a) to (8c).

    .. math::

       d_H = \frac{4A}{l_w}, \qquad
       d_o = \sqrt{\frac{4 N_o A}{\pi}}, \qquad
       F_d = \frac{d_H}{d_o}

    :math:`F_d` compares the hydraulic diameter of one flow passage with the
    diameter of the single circular orifice that would pass the same total
    area. A cage full of small holes has a small :math:`F_d` and a small jet;
    a single large port has :math:`F_d` near one.

    :param passage_area: :math:`A`, the area of a single flow passage, in m².
    :param wetted_perimeter: :math:`l_w` of that passage, in m.
    :param passages: :math:`N_o`, the number of independent flow passages.
    :return: :math:`F_d`, dimensionless.
    :raises ValueError: If an argument is not positive and finite, or if the
        passage count is not a whole number.
    """
    area = require_positive(passage_area, "passage_area")
    perimeter = require_positive(wetted_perimeter, "wetted_perimeter")
    count = int(passages)
    if count != passages or count < 1:
        msg = (
            "'passages' counts the independent flow passages, so it must be "
            f"a whole number of one or more; got {passages!r}."
        )
        raise ValueError(msg)
    if not math.isfinite(area) or not math.isfinite(perimeter):
        msg = "'passage_area' and 'wetted_perimeter' must be finite."
        raise ValueError(msg)
    hydraulic = 4.0 * area / perimeter
    orifice = math.sqrt(4.0 * count * area / math.pi)
    return float(hydraulic / orifice)


def jet_diameter(
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    *,
    coefficient: str = "Cv",
) -> float:
    r"""The jet diameter of Equation (9).

    .. math::

       D_j = N_{14}\, F_d \sqrt{C\, F_{LP}/F_P}

    :param flow_coefficient: :math:`C`, the required flow coefficient of the
        valve at the travel being examined.
    :param style_modifier: :math:`F_d`, from :func:`valve_style_modifier`.
    :param pressure_recovery: :math:`F_{LP}/F_p`, or :math:`F_L` for a valve
        with no attached fittings.
    :param coefficient: Which flow coefficient ``flow_coefficient`` is,
        ``"Cv"`` or ``"Kv"``, which selects :math:`N_{14}` from Table 1.
    :return: :math:`D_j`, in m.
    :raises ValueError: If a value is not positive and finite, or the
        coefficient is not one Table 1 prints a constant for.
    """
    kind = require_choice(coefficient, "coefficient", tuple(FLOW_COEFFICIENT_CONSTANTS))
    capacity = require_positive(flow_coefficient, "flow_coefficient")
    modifier = require_positive(style_modifier, "style_modifier")
    recovery_factor = require_positive(pressure_recovery, "pressure_recovery")
    if not all(math.isfinite(value) for value in (capacity, modifier, recovery_factor)):
        msg = "The jet diameter needs finite arguments."
        raise ValueError(msg)
    return float(
        FLOW_COEFFICIENT_CONSTANTS[kind]
        * modifier
        * math.sqrt(capacity * recovery_factor)
    )


def _regime_state(
    regime: int,
    *,
    pressure_ratio: float,
    boundaries: RegimeBoundaries,
    gamma: float,
    recovery: float,
    inlet_pressure: float,
    inlet_density: float,
    inlet_temperature: float,
) -> tuple[float, float, float, float]:
    """Table 3's Mach number, temperature, sonic velocity and velocity head.

    Returns ``(mach, temperature, sonic_velocity, jet_velocity)`` where the
    last is the speed the stream power of Table 3's last column is formed
    from: :math:`M_{vc} c_{vc}` in regime I and :math:`c_{vcc}` above it,
    which is the same quantity once the vena contracta is choked.
    """
    subsonic = 1.0 - pressure_ratio / recovery**2
    if regime == REGIME_SUBSONIC:
        mach = math.sqrt(
            (2.0 / (gamma - 1.0)) * (subsonic ** ((1.0 - gamma) / gamma) - 1.0)
        )
        temperature = inlet_temperature * subsonic ** ((gamma - 1.0) / gamma)
        sonic = math.sqrt(
            gamma
            * (inlet_pressure / inlet_density)
            * subsonic ** ((gamma - 1.0) / gamma)
        )
        return mach, temperature, sonic, mach * sonic
    temperature = 2.0 * inlet_temperature / (gamma + 1.0)
    sonic = math.sqrt((2.0 * gamma / (gamma + 1.0)) * (inlet_pressure / inlet_density))
    if regime == REGIME_CONSTANT_EFFICIENCY:
        mach = math.sqrt(
            (2.0 / (gamma - 1.0))
            * (_CONSTANT_EFFICIENCY_FACTOR ** ((gamma - 1.0) / gamma) - 1.0)
        )
    else:
        mach = math.sqrt(
            (2.0 / (gamma - 1.0))
            * (
                (1.0 / (boundaries.recovery * (1.0 - pressure_ratio)))
                ** ((gamma - 1.0) / gamma)
                - 1.0
            )
        )
    return mach, temperature, sonic, sonic


def _acoustical_efficiency(
    regime: int,
    *,
    correction: float,
    recovery: float,
    mach: float,
    pressure_ratio: float,
    vena_contracta_ratio: float,
) -> float:
    """Table 3's acoustical efficiency factor, regime by regime."""
    scale = 10.0**correction
    exponent = _EFFICIENCY_MACH_EXPONENT * recovery**2
    if regime == REGIME_SUBSONIC:
        return float(scale * recovery**2 * mach**3)
    if regime == REGIME_CHOKED:
        return float(scale * (pressure_ratio / vena_contracta_ratio) * mach**exponent)
    if regime == REGIME_SUPERSONIC:
        return float(scale * mach**exponent)
    return float(scale * (mach**2 / 2.0) * math.sqrt(2.0) ** exponent)


def _peak_frequency(
    regime: int, *, strouhal: float, mach: float, sonic: float, jet: float
) -> float:
    """Table 3's peak frequency, regime by regime."""
    if regime <= REGIME_SUPERSONIC:
        return float(strouhal * mach * sonic / jet)
    return float(
        _SHOCK_PEAK_COEFFICIENT * strouhal * sonic / (jet * math.sqrt(mach**2 - 1.0))
    )


def internal_spectrum(
    internal_level: float, peak_frequency: float, frequency: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Equation (19): the internal level spread over the third-octave bands.

    .. math::

       L_{pi}(f_i) = L_{pi} - 8 - 10 \lg\left\{
         \left[1 + \left(\frac{f_i}{2 f_p}\right)^{2,5}\right]
         \left[1 + \left(\frac{f_p}{2 f_i}\right)^{1,7}\right]\right\}

    The two brackets are not symmetric: the spectrum falls as
    :math:`f^{-2,5}` above the peak and as :math:`f^{1,7}` below it, so a
    valve is heard further above its peak than below it. The 8 dB is what
    turns an overall level into a one-third-octave one; the NOTE to Table 7
    puts 3 dB there for octave bands instead.

    :param internal_level: :math:`L_{pi}` of Equation (18), in dB.
    :param peak_frequency: :math:`f_p` from Table 3, in Hz.
    :param frequency: The band centre frequencies, in Hz.
    :return: The internal level in each band, in dB.
    :raises ValueError: If the peak frequency is not positive and finite, or
        a band centre is not.
    """
    peak = require_positive(peak_frequency, "peak_frequency")
    bands = np.asarray(frequency, dtype=np.float64)
    if not math.isfinite(peak):
        msg = "'peak_frequency' must be a positive, finite frequency in Hz."
        raise ValueError(msg)
    if bands.ndim != 1 or bands.size == 0:
        msg = "'frequency' must be a non-empty one-dimensional band axis in Hz."
        raise ValueError(msg)
    if not np.all(np.isfinite(bands)) or np.any(bands <= 0.0):
        msg = "'frequency' must carry positive, finite band centres in Hz."
        raise ValueError(msg)
    high = (bands / (2.0 * peak)) ** _SPECTRUM_HIGH_EXPONENT
    low = (peak / (2.0 * bands)) ** _SPECTRUM_LOW_EXPONENT
    shape = 10.0 * np.log10((1.0 + high) * (1.0 + low))
    return np.asarray(internal_level - 8.0 - shape, dtype=np.float64)


@dataclass(frozen=True)
class PipeFrequencies:
    """The three frequencies Clause 5.5 shapes the transmission loss with.

    :ivar ring: :math:`f_r` of Equation (21), where the pipe rings as a
        circumference of one wavelength.
    :ivar internal_coincidence: :math:`f_o` of Equation (22).
    :ivar external_coincidence: :math:`f_g` of Equation (23).
    """

    ring: float
    internal_coincidence: float
    external_coincidence: float


def coincidence_frequencies(
    internal_diameter: float,
    wall_thickness: float,
    downstream_sound_speed: float,
    *,
    pipe_sound_speed: float = PIPE_SOUND_SPEED_M_S,
    air_sound_speed: float = AIR_SOUND_SPEED_M_S,
) -> PipeFrequencies:
    r"""Equations (21), (22) and (23).

    .. math::

       f_r = \frac{c_s}{\pi D_i}, \qquad
       f_o = \frac{f_r}{4}\left(\frac{c_2}{c_a}\right), \qquad
       f_g = \frac{\sqrt{3}}{\pi t_S}\frac{c_a^2}{c_s}

    :param internal_diameter: :math:`D_i` of the downstream pipe, in m.
    :param wall_thickness: :math:`t_S` of the pipe wall, in m.
    :param downstream_sound_speed: :math:`c_2` in the fluid downstream of the
        valve, in m/s.
    :param pipe_sound_speed: :math:`c_s`, 5 000 m/s for steel by NOTE 4.
    :param air_sound_speed: :math:`c_a`, 343 m/s by NOTE 3.
    :return: The three frequencies, in Hz.
    :raises ValueError: If any argument is not positive and finite.
    """
    diameter = require_positive(internal_diameter, "internal_diameter")
    thickness = require_positive(wall_thickness, "wall_thickness")
    downstream = require_positive(downstream_sound_speed, "downstream_sound_speed")
    wall = require_positive(pipe_sound_speed, "pipe_sound_speed")
    air = require_positive(air_sound_speed, "air_sound_speed")
    values = (diameter, thickness, downstream, wall, air)
    if not all(math.isfinite(value) for value in values):
        msg = "The coincidence frequencies need finite arguments."
        raise ValueError(msg)
    ring = wall / (math.pi * diameter)
    return PipeFrequencies(
        ring=float(ring),
        internal_coincidence=float((ring / 4.0) * (downstream / air)),
        external_coincidence=float(
            (math.sqrt(3.0) / (math.pi * thickness)) * air**2 / wall
        ),
    )


def _frequency_factors(
    frequency: NDArray[np.float64], pipe: PipeFrequencies
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Table 6's two frequency factors, band by band.

    ``G_y``'s branch in the low column tests the *internal coincidence*
    frequency against the external one rather than the band, which is how
    the table prints it and is not a slip of this implementation.
    """
    ring = pipe.ring
    internal = pipe.internal_coincidence
    external = pipe.external_coincidence
    below = frequency < internal
    g_x = np.where(
        below,
        (internal / ring) ** (2.0 / 3.0) * (frequency / internal) ** 4,
        np.where(frequency < ring, np.sqrt(frequency / ring), 1.0),
    )
    low_g_y = internal / external if internal < external else 1.0
    g_y = np.where(
        below,
        low_g_y,
        np.where(frequency < external, frequency / external, 1.0),
    )
    return (
        np.asarray(g_x, dtype=np.float64),
        np.asarray(g_y, dtype=np.float64),
    )


def _damping_factor(valve_outlet_diameter: float) -> float:
    """Equation (20b): the damping factor, a cubic in the outlet diameter."""
    diameter = valve_outlet_diameter
    if diameter > _UNDAMPED_OUTLET_M:
        return 0.0
    if diameter < _FULLY_DAMPED_OUTLET_M:
        return 9.0
    return float(
        -16660.0 * diameter**3 + 6370.0 * diameter**2 - 813.0 * diameter + 35.8
    )


def pipe_transmission_loss(
    frequency: NDArray[np.float64],
    *,
    internal_diameter: float,
    wall_thickness: float,
    valve_outlet_diameter: float,
    downstream_density: float,
    downstream_sound_speed: float,
    pipe_density: float,
    pipe_sound_speed: float = PIPE_SOUND_SPEED_M_S,
    air_sound_speed: float = AIR_SOUND_SPEED_M_S,
    atmospheric_pressure: float = 1.01325e5,
    standard_pressure: float = 1.01325e5,
) -> NDArray[np.float64]:
    r"""Equation (20a): what the pipe wall keeps in, band by band.

    .. math::

       TL(f_i) = 10 \lg\left[
         8{,}25\times10^{-7}
         \left(\frac{c_2}{t_S f_i}\right)^2
         \frac{G_x(f_i)}
              {\dfrac{\rho_2 c_2 + 2\pi t_S f_i \rho_s \eta_s(f_i)}
                     {415\, G_y(f_i)} + 1}
         \frac{p_a}{p_s}\right] - \Delta TL

    The result is a large negative number, and Equation (24) *adds* it to the
    internal level, so the sign is not a convention this module chose.

    :param frequency: The band centre frequencies, in Hz.
    :param internal_diameter: :math:`D_i`, in m.
    :param wall_thickness: :math:`t_S`, in m.
    :param valve_outlet_diameter: :math:`D`, in m, which selects the damping
        factor of Equation (20b) and is the valve outlet and not the pipe.
    :param downstream_density: :math:`\rho_2`, in kg/m³.
    :param downstream_sound_speed: :math:`c_2`, in m/s.
    :param pipe_density: :math:`\rho_s` of the pipe material, in kg/m³.
    :param pipe_sound_speed: :math:`c_s`, in m/s.
    :param air_sound_speed: :math:`c_a`, in m/s.
    :param atmospheric_pressure: :math:`p_a`, in Pa.
    :param standard_pressure: :math:`p_s`, in Pa.
    :return: The transmission loss in each band, in dB, negative.
    :raises ValueError: If an argument is not positive and finite.
    """
    bands = np.asarray(frequency, dtype=np.float64)
    if bands.ndim != 1 or bands.size == 0:
        msg = "'frequency' must be a non-empty one-dimensional band axis in Hz."
        raise ValueError(msg)
    if not np.all(np.isfinite(bands)) or np.any(bands <= 0.0):
        msg = "'frequency' must carry positive, finite band centres in Hz."
        raise ValueError(msg)
    thickness = require_positive(wall_thickness, "wall_thickness")
    outlet = require_positive(valve_outlet_diameter, "valve_outlet_diameter")
    density = require_positive(downstream_density, "downstream_density")
    sound_speed = require_positive(downstream_sound_speed, "downstream_sound_speed")
    wall_density = require_positive(pipe_density, "pipe_density")
    ambient = require_positive(atmospheric_pressure, "atmospheric_pressure")
    reference = require_positive(standard_pressure, "standard_pressure")

    pipe = coincidence_frequencies(
        internal_diameter,
        thickness,
        sound_speed,
        pipe_sound_speed=pipe_sound_speed,
        air_sound_speed=air_sound_speed,
    )
    g_x, g_y = _frequency_factors(bands, pipe)
    structural = np.sqrt(STRUCTURAL_LOSS_REFERENCE_HZ / (100.0 * bands))
    stiffness = (
        density * sound_speed
        + 2.0 * math.pi * thickness * bands * wall_density * structural
    ) / (_AIR_IMPEDANCE * g_y)
    ratio = (sound_speed / (thickness * bands)) ** 2
    inside = _TRANSMISSION_COEFFICIENT * ratio * g_x / (stiffness + 1.0)
    loss = 10.0 * np.log10(inside * (ambient / reference))
    return np.asarray(loss - _damping_factor(outlet), dtype=np.float64)


@dataclass(frozen=True)
class Expander:
    r"""The transition piece downstream of the valve (Clause 7).

    A valve whose outlet is narrower than the pipe it discharges into makes a
    second jet, at the step. Clause 7 is the method for it, and 7.1 limits
    the method to a transition of 30 degrees total included angle: a steeper
    cone makes the flow unstable in ways the standard does not model.

    :ivar contraction: :math:`\beta` of Equation (35). NOTE 1 puts it at 0,93
        for straight pattern globe valves and as low as 0,7 for some rotary
        ones, and says there are no data for the rest.
    :ivar efficiency_correction: :math:`A_\eta` for the expander, which is its
        own row of Table 4 and not the valve's: the table prints -3,0.
    :ivar strouhal_number: :math:`St_p` for the expander, 0,2 in Table 4.
    """

    contraction: float = GLOBE_CONTRACTION_COEFFICIENT
    efficiency_correction: float = -3.0
    strouhal_number: float = 0.2


#: The straight pattern globe valve of NOTE 1 with the expander row of
#: Table 4, which is what Annex A's sixth example is and what a caller who
#: does not say otherwise gets.
DEFAULT_EXPANDER = Expander()


@dataclass(frozen=True)
class ExpanderNoise:
    r"""What Clause 7 says the flow leaving the valve outlet makes.

    :ivar pipe_velocity: :math:`U_p` of Equation (34), in m/s, after the
        Mach 0,8 cap.
    :ivar inlet_velocity: :math:`U_R` of Equation (35), in m/s, after the
        sonic cap.
    :ivar mach: :math:`M_R` of Equation (39).
    :ivar stream_power: :math:`W_{mR}` of Equation (36), in W.
    :ivar acoustical_efficiency: :math:`\eta_R` of Equation (38).
    :ivar sound_power: :math:`W_{aR}` of Equation (40), in W.
    :ivar peak_frequency: :math:`f_{pR}` of Equation (37), in Hz.
    :ivar internal_level: :math:`L_{piR}` of Equation (41), in dB.
    :ivar band_internal_level: :math:`L_{piR}(f_i)` of Equation (42), in dB.
    """

    pipe_velocity: float
    inlet_velocity: float
    mach: float
    stream_power: float
    acoustical_efficiency: float
    sound_power: float
    peak_frequency: float
    internal_level: float
    band_internal_level: NDArray[np.float64]


def expander_noise(  # noqa: PLR0913
    frequency: NDArray[np.float64],
    *,
    mass_flow: float,
    downstream_density: float,
    downstream_sound_speed: float,
    internal_diameter: float,
    throat_diameter: float,
    velocity_correction: float,
    expander: Expander = DEFAULT_EXPANDER,
) -> ExpanderNoise:
    r"""Clause 7: the noise the flow makes leaving the valve outlet.

    .. math::

       U_p = \frac{4 \dot m}{\pi \rho_2 D_i^2}, \qquad
       U_R = \frac{U_p D_i^2}{\beta d_i^2}, \qquad
       M_R = \frac{U_R}{c_2}

    .. math::

       W_{mR} = \frac{\dot m U_R^2}{2}
                \left[\left(1 - \frac{d_i^2}{D_i^2}\right)^2
                + 0{,}2\right],
       \qquad
       \eta_R = 10^{A_\eta} M_R^3, \qquad
       f_{pR} = \frac{St_p U_R}{d_i}

    The two caps are the clause's: :math:`U_p` is limited to Mach 0,8 and
    :math:`U_R` to the sonic velocity, so a step that would otherwise be
    computed as supersonic is computed at Mach one instead.

    :param frequency: The band centre frequencies, in Hz.
    :param mass_flow: :math:`\dot m`, in kg/s.
    :param downstream_density: :math:`\rho_2`, in kg/m³.
    :param downstream_sound_speed: :math:`c_2`, in m/s.
    :param internal_diameter: :math:`D_i` of the downstream pipe, in m.
    :param throat_diameter: :math:`d_i`, the smaller of the valve outlet and
        the expander inlet, in m.
    :param velocity_correction: :math:`L_g` of Equation (16), in dB, which
        Equation (41) adds exactly as Equation (18) does.
    :param expander: The transition piece.
    :return: An :class:`ExpanderNoise`.
    :raises ValueError: If a value is not positive and finite, or the throat
            is wider than the pipe.
    """
    bands = np.asarray(frequency, dtype=np.float64)
    flow = require_positive(mass_flow, "mass_flow")
    rho2 = require_positive(downstream_density, "downstream_density")
    c2 = require_positive(downstream_sound_speed, "downstream_sound_speed")
    bore = require_positive(internal_diameter, "internal_diameter")
    throat = require_positive(throat_diameter, "throat_diameter")
    beta = require_positive(expander.contraction, "expander.contraction")
    if throat > bore:
        msg = (
            "'throat_diameter' is the smaller of the valve outlet and the "
            f"expander inlet, so it cannot exceed the pipe bore; got "
            f"{throat_diameter!r} m against {internal_diameter!r} m."
        )
        raise ValueError(msg)

    pipe_velocity = min(
        4.0 * flow / (math.pi * rho2 * bore**2),
        EXPANDER_PIPE_MACH_LIMIT * c2,
    )
    inlet_velocity = min(pipe_velocity * bore**2 / (beta * throat**2), c2)
    mach = inlet_velocity / c2
    area_ratio = throat**2 / bore**2
    stream_power = (
        flow * inlet_velocity**2 / 2.0 * ((1.0 - area_ratio) ** 2 + _EXPANDER_RESIDUAL)
    )
    efficiency = 10.0**expander.efficiency_correction * mach**3
    sound_power = efficiency * stream_power
    peak = expander.strouhal_number * inlet_velocity / throat
    internal_level = (
        10.0
        * math.log10(_INTERNAL_LEVEL_COEFFICIENT * sound_power * rho2 * c2 / bore**2)
        + velocity_correction
    )
    return ExpanderNoise(
        pipe_velocity=float(pipe_velocity),
        inlet_velocity=float(inlet_velocity),
        mach=float(mach),
        stream_power=float(stream_power),
        acoustical_efficiency=float(efficiency),
        sound_power=float(sound_power),
        peak_frequency=float(peak),
        internal_level=float(internal_level),
        band_internal_level=internal_spectrum(internal_level, peak, bands),
    )


def combine_internal_levels(
    *levels: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Equation (43): two internal spectra at the same pipe wall, added.

    .. math::

       L_{piS}(f_i) = 10 \lg\left(
         10^{L_{pi}(f_i)/10} + 10^{L_{piR}(f_i)/10}\right)

    The valve trim and the expander are two sources inside one pipe, so they
    add in energy and not in level, and the sum is what Equation (24) then
    takes through the wall.

    :param levels: Two or more band level arrays of the same shape, in dB.
    :return: Their energy sum, in dB.
    :raises ValueError: If fewer than two are given, or they disagree in
        shape.
    """
    if len(levels) < _MINIMUM_SOURCES:
        msg = (
            "Equation (43) combines the valve trim with the expander, so "
            f"'levels' needs at least two spectra; got {len(levels)}."
        )
        raise ValueError(msg)
    arrays = [np.asarray(level, dtype=np.float64) for level in levels]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        msg = f"Every spectrum must have the same shape; got {sorted(shapes)}."
        raise ValueError(msg)
    total = sum(10.0 ** (array / 10.0) for array in arrays)
    return np.asarray(10.0 * np.log10(total), dtype=np.float64)


@dataclass(frozen=True)
class AerodynamicValveNoise:
    r"""What IEC 60534-8-3 Clause 5 says about one operating point.

    :ivar regime: Which of the five regimes of Clause 5.2 the valve is in.
    :ivar boundaries: The four pressure ratios that placed it there.
    :ivar pressure_ratio: :math:`x` of Equation (1).
    :ivar vena_contracta_pressure: :math:`p_{vc}` of Equation (2), in Pa. It
        goes negative past the choking point, where the equation is being
        read outside the range it means anything in.
    :ivar jet_diameter: :math:`D_j` of Equation (9), in m.
    :ivar mach: The Mach number Table 3 uses in this regime.
    :ivar acoustical_efficiency: :math:`\eta`, the fraction of the stream
        power that leaves as sound.
    :ivar stream_power: :math:`W_m`, in W.
    :ivar sound_power: :math:`W_a` of Equation (11), in W.
    :ivar sound_power_level: :math:`L_{wi}` of Equation (12), in dB.
    :ivar peak_frequency: :math:`f_p` from Table 3, in Hz.
    :ivar outlet_mach: :math:`M_o` of Equation (15), which Clause 5 is only
        valid below 0,3.
    :ivar pipe_mach: :math:`M_2` of Equation (17), before the 0,3 limit.
    :ivar velocity_correction: :math:`L_g` of Equation (16), in dB.
    :ivar internal_level: :math:`L_{pi}` of Equation (18), in dB.
    :ivar frequency: The 33 one-third-octave band centres of Table 5, in Hz.
    :ivar band_internal_level: :math:`L_{pi}(f_i)` of Equation (19), in dB.
    :ivar band_transmission_loss: :math:`TL(f_i)` of Equation (20a), in dB.
    :ivar band_external_level: :math:`L_{pe,1m}(f_i)` of Equation (24), in dB.
    :ivar external_level: :math:`L_{pAe,1m}` of Equation (25), in dB.
    :ivar pipe_frequencies: The ring and coincidence frequencies the
        transmission loss is shaped by.
    :ivar expander: What Clause 7 says the flow leaving the valve outlet
        makes, or ``None`` when no expander was given. When it is present its
        spectrum is already in ``band_external_level`` and in
        ``external_level``, combined with the trim by Equation (43).
    """

    regime: int
    boundaries: RegimeBoundaries
    pressure_ratio: float
    vena_contracta_pressure: float
    jet_diameter: float
    mach: float
    acoustical_efficiency: float
    stream_power: float
    sound_power: float
    sound_power_level: float
    peak_frequency: float
    outlet_mach: float
    pipe_mach: float
    velocity_correction: float
    internal_level: float
    frequency: NDArray[np.float64]
    band_internal_level: NDArray[np.float64]
    band_transmission_loss: NDArray[np.float64]
    band_external_level: NDArray[np.float64]
    external_level: float
    pipe_frequencies: PipeFrequencies
    expander: ExpanderNoise | None


def _third_octave_bands() -> NDArray[np.float64]:
    """Table 5's 33 one-third-octave band centres, 12,5 Hz to 20 kHz."""
    from ..filters.frequencies import normalized_frequencies

    return np.asarray(normalized_frequencies(3), dtype=np.float64)


def valve_aerodynamic_noise(  # noqa: PLR0913
    *,
    mass_flow: float,
    inlet_pressure: float,
    outlet_pressure: float,
    inlet_density: float,
    inlet_temperature: float,
    specific_heat_ratio: float,
    molecular_mass: float,
    flow_coefficient: float,
    style_modifier: float,
    pressure_recovery: float,
    valve_outlet_diameter: float,
    internal_diameter: float,
    wall_thickness: float,
    pipe_density: float,
    efficiency_correction: float,
    strouhal_number: float,
    coefficient: str = "Cv",
    expander: Expander | None = None,
    pipe_sound_speed: float = PIPE_SOUND_SPEED_M_S,
    air_sound_speed: float = AIR_SOUND_SPEED_M_S,
    atmospheric_pressure: float = 1.01325e5,
    standard_pressure: float = 1.01325e5,
) -> AerodynamicValveNoise:
    r"""The whole of Clause 5, from the operating point to the level at 1 m.

    The chain is Clause 5.7's own flow chart: the pressure ratios of 5.1 and
    5.2, the geometry of 5.3, the regime-dependent stream power and
    acoustical efficiency of 5.4, then the pipe transmission loss of 5.5 and
    the external level of 5.6, which are common to every regime.

    :param mass_flow: :math:`\dot m`, in kg/s.
    :param inlet_pressure: :math:`p_1`, absolute, in Pa.
    :param outlet_pressure: :math:`p_2`, absolute, in Pa.
    :param inlet_density: :math:`\rho_1`, in kg/m³.
    :param inlet_temperature: :math:`T_1`, absolute, in K.
    :param specific_heat_ratio: :math:`\gamma`.
    :param molecular_mass: :math:`M`, in kg/kmol.
    :param flow_coefficient: :math:`C` at the travel being examined.
    :param style_modifier: :math:`F_d`, from :func:`valve_style_modifier`.
    :param pressure_recovery: :math:`F_L`, or :math:`F_{LP}/F_p` with
        attached fittings.
    :param valve_outlet_diameter: :math:`D`, in m.
    :param internal_diameter: :math:`D_i` of the downstream pipe, in m.
    :param wall_thickness: :math:`t_S`, in m.
    :param pipe_density: :math:`\rho_s`, in kg/m³.
    :param efficiency_correction: :math:`A_\eta` from Table 4.
    :param strouhal_number: :math:`St_p` from Table 4.
    :param coefficient: ``"Cv"`` or ``"Kv"``, selecting :math:`N_{14}`.
    :param expander: The transition piece downstream of the valve. Give one
        when the valve outlet is narrower than the pipe and the outlet Mach
        number has passed 0,3, which is when NOTE 1 to Equation (15) sends
        the calculation to Clause 7; the flow leaving the outlet is then a
        second source and Equation (43) adds it to the trim.
    :param pipe_sound_speed: :math:`c_s`, in m/s.
    :param air_sound_speed: :math:`c_a`, in m/s.
    :param atmospheric_pressure: :math:`p_a`, in Pa.
    :param standard_pressure: :math:`p_s`, in Pa.
    :return: An :class:`AerodynamicValveNoise` carrying every printed
        intermediate as well as the level at 1 m.
    :raises ValueError: If a value is outside the range its equation is
        written for.
    """
    p1 = require_positive(inlet_pressure, "inlet_pressure")
    p2 = require_positive(outlet_pressure, "outlet_pressure")
    if p2 >= p1:
        msg = (
            "A control valve drops pressure, so 'outlet_pressure' must be "
            f"below 'inlet_pressure'; got {outlet_pressure!r} and "
            f"{inlet_pressure!r} Pa."
        )
        raise ValueError(msg)
    flow = require_positive(mass_flow, "mass_flow")
    rho1 = require_positive(inlet_density, "inlet_density")
    t1 = require_positive(inlet_temperature, "inlet_temperature")
    mass = require_positive(molecular_mass, "molecular_mass")
    outlet = require_positive(valve_outlet_diameter, "valve_outlet_diameter")
    bore = require_positive(internal_diameter, "internal_diameter")

    boundaries = pressure_ratio_boundaries(specific_heat_ratio, pressure_recovery)
    gamma = float(specific_heat_ratio)
    recovery = float(pressure_recovery)
    x = (p1 - p2) / p1
    regime = flow_regime(x, boundaries)
    vena_contracta_pressure = p1 * (1.0 - x / recovery**2)
    jet = jet_diameter(
        flow_coefficient, style_modifier, recovery, coefficient=coefficient
    )

    mach, _temperature, sonic, velocity = _regime_state(
        regime,
        pressure_ratio=x,
        boundaries=boundaries,
        gamma=gamma,
        recovery=recovery,
        inlet_pressure=p1,
        inlet_density=rho1,
        inlet_temperature=t1,
    )
    efficiency = _acoustical_efficiency(
        regime,
        correction=float(efficiency_correction),
        recovery=recovery,
        mach=mach,
        pressure_ratio=x,
        vena_contracta_ratio=boundaries.vena_contracta,
    )
    stream_power = flow * velocity**2 / 2.0
    sound_power = efficiency * stream_power
    peak = _peak_frequency(
        regime,
        strouhal=float(strouhal_number),
        mach=mach,
        sonic=sonic,
        jet=jet,
    )

    rho2 = rho1 * (p2 / p1)
    c2 = math.sqrt(gamma * UNIVERSAL_GAS_CONSTANT * t1 / mass)
    outlet_mach = 4.0 * flow / (math.pi * outlet**2 * rho2 * c2)
    pipe_mach = 4.0 * flow / (math.pi * bore**2 * rho2 * c2)
    limited = min(pipe_mach, PIPE_WALL_MACH_LIMIT)
    velocity_correction = 16.0 * math.log10(1.0 / (1.0 - limited))
    internal_level = (
        10.0
        * math.log10(_INTERNAL_LEVEL_COEFFICIENT * sound_power * rho2 * c2 / bore**2)
        + velocity_correction
    )

    bands = _third_octave_bands()
    band_internal = internal_spectrum(internal_level, peak, bands)
    outlet_noise: ExpanderNoise | None = None
    if expander is not None:
        outlet_noise = expander_noise(
            bands,
            mass_flow=flow,
            downstream_density=rho2,
            downstream_sound_speed=c2,
            internal_diameter=bore,
            throat_diameter=min(outlet, bore),
            velocity_correction=velocity_correction,
            expander=expander,
        )
        band_internal = combine_internal_levels(
            band_internal, outlet_noise.band_internal_level
        )
    elif outlet_mach > MACH_LIMIT_STANDARD_TRIM:
        warnings.warn(
            f"The valve outlet is at Mach {outlet_mach:.2g} and NOTE 1 to "
            f"Equation (15) holds Clause 5 to {MACH_LIMIT_STANDARD_TRIM:g}, "
            "so the flow leaving the outlet is a second source this result "
            "does not carry. Pass an 'expander' to add the Clause 7 term.",
            ValveNoiseWarning,
            stacklevel=2,
        )
    band_loss = pipe_transmission_loss(
        bands,
        internal_diameter=bore,
        wall_thickness=wall_thickness,
        valve_outlet_diameter=outlet,
        downstream_density=rho2,
        downstream_sound_speed=c2,
        pipe_density=pipe_density,
        pipe_sound_speed=pipe_sound_speed,
        air_sound_speed=air_sound_speed,
        atmospheric_pressure=atmospheric_pressure,
        standard_pressure=standard_pressure,
    )
    spreading = 10.0 * np.log10(
        (bore + 2.0 * wall_thickness + 2.0 * _MEASUREMENT_DISTANCE_M)
        / (bore + 2.0 * wall_thickness)
    )
    band_external = band_internal + band_loss - spreading
    weighted = band_external + np.asarray(AERODYNAMIC_A_WEIGHTING_DB, dtype=np.float64)
    external_level = 10.0 * math.log10(float(np.sum(10.0 ** (weighted / 10.0))))

    pipe = coincidence_frequencies(
        bore,
        wall_thickness,
        c2,
        pipe_sound_speed=pipe_sound_speed,
        air_sound_speed=air_sound_speed,
    )
    return AerodynamicValveNoise(
        regime=regime,
        boundaries=boundaries,
        pressure_ratio=float(x),
        vena_contracta_pressure=float(vena_contracta_pressure),
        jet_diameter=float(jet),
        mach=float(mach),
        acoustical_efficiency=float(efficiency),
        stream_power=float(stream_power),
        sound_power=float(sound_power),
        sound_power_level=float(
            10.0 * math.log10(sound_power / _REFERENCE_SOUND_POWER_W)
        ),
        peak_frequency=float(peak),
        outlet_mach=float(outlet_mach),
        pipe_mach=float(pipe_mach),
        velocity_correction=float(velocity_correction),
        internal_level=float(internal_level),
        frequency=bands,
        band_internal_level=band_internal,
        band_transmission_loss=band_loss,
        band_external_level=np.asarray(band_external, dtype=np.float64),
        external_level=float(external_level),
        pipe_frequencies=pipe,
        expander=outlet_noise,
    )
