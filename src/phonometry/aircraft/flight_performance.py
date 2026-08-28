#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ECAC Doc 29 flight performance: procedural steps into a flight profile.

A published departure or arrival procedure is not a trajectory. It is a list of
*procedural steps* -- "climb at take-off thrust to 1500 ft", "accelerate to
210.6 kt at 984.3 ft/min", "descend on a 3 degree slope from 3000 ft at 180 kt"
-- and the aeroplane's own aerodynamic and engine coefficients. **ECAC Doc 29
5th ed., Volume 2, Appendix B** is the flight-mechanics model that turns the one
into the other: a *flight profile*, an ordered list of profile points carrying
distance along the ground track, height above the aerodrome, true airspeed and
corrected net thrust per engine. Corrected net thrust is what the NPD tables of
:mod:`phonometry.aircraft.airport_noise` are indexed on, so this model is what
stands between a published procedure and a noise contour.

* :class:`Aerodrome` -- the aerodrome and its weather, and the five atmosphere
  ratios of B3 that every equation below reads.
* :class:`PerformanceAircraft` with :class:`JetEngineCoefficients`,
  :class:`PropellerEngineCoefficients` and :class:`AerodynamicCoefficients` --
  the ANP coefficient tables the equations take their constants from.
* :class:`DepartureStep` and :class:`ApproachStep` -- one row each of a
  published procedure.
* :func:`departure_profile` and :func:`approach_profile` -- the model, returning
  a :class:`FlightProfile` of :class:`ProfilePoint`.

Units are the standard's and they are English throughout (B2): feet, knots,
pounds, pounds of thrust per engine, degrees Celsius in the thrust equations and
inches of mercury for pressure. Doc 29 keeps them "due to the history of the
overarching method [...] and the strong association that aviation has with
English units", and pins two conversion constants at deliberately imprecise
legacy values that must not be improved (footnotes 30 and 31, folios B-7/B-8).

Departures run forward from brake release and arrivals run **backwards** from
touchdown, which is why an arrival profile carries negative distances until the
aeroplane is on the runway (folio B-5).

Source (clean-room, implemented from the published standard): ECAC.CEAC Doc 29,
5th edition, Volume 2 "Technical Guide", Appendix B, folios B-1 to B-49.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_finite_fields

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

# --------------------------------------------------------------------------
# Constants fixed by the standard (B2.1, B2.2, B3)
# --------------------------------------------------------------------------
#: Standard gravitational acceleration, ft/s^2 (the one row of table B2.1).
_G_FT_S2 = 32.174
#: Knots to feet per second (table B2.2). Squared it is 2.8487, which Doc 29
#: rounds to 2.85 in Eq. B-14 and carries as k^2 in Eq. B-24 and Eq. B-41.
_KT_FT_S = 1.68781
#: Standard sea-level pressure, inHg (B3, the p_O of Eq. B-4). The precise
#: value is 29.9212553;
#: footnote 30 (folio B-7) keeps the shorter one "for legacy purposes and must
#: be used to match calculations from this method", so it is not improved here.
_STANDARD_PRESSURE_INHG = 29.92
#: Standard sea-level temperature in degrees Rankine, 459.67 + 59 degF, the
#: denominator of the temperature ratio of Eq. B-3 (B3).
_STANDARD_TEMPERATURE_R = 518.67
#: Rankine offset: degrees Rankine = degrees Fahrenheit + 459.67.
_RANKINE_OFFSET = 459.67
#: Tropospheric temperature lapse rate, degF/ft (Eq. B-2).
_LAPSE_F_PER_FT = 0.003566
#: Pressure-ratio exponent of the standard atmosphere (Eq. B-4, Eq. B-6).
_PRESSURE_EXPONENT = 5.256

#: Doc 29's own modelling default headwind, kt (B4.4). Eq. B-16's equivalent
#: take-off distance is defined *into* this wind, and Eq. B-17, B-22, B-28 and
#: B-33 correct away from it, which is why 8 appears as a bare number in each.
DEFAULT_HEADWIND_KT = 8.0

#: Approach weight as a fraction of the maximum landing weight (folio B-31,
#: repeated at B-39, B-46 and B-47): "W is the aircraft approach weight (lb),
#: which corresponds to 90% of the MLW available in the ANP 'Aircraft' table".
#: The ANP ``Default_weights`` arrival row is *not* this weight and is not used.
APPROACH_WEIGHT_FRACTION = 0.9

#: Climb-angle factor K of Eq. B-21, for a calibrated airspeed at or below
#: :data:`_CLIMB_K_SPEED_KT`. It "accounts for the effects on climb gradient of
#: climbing into an 8-knot headwind and the acceleration inherent in climbing at
#: constant Calibrated Airspeed [...] It avoids the otherwise need for an
#: iterative solution" (folio B-19), which is why a Climb step never iterates.
_CLIMB_K_LOW = 1.01
#: Climb-angle factor K above :data:`_CLIMB_K_SPEED_KT` (Eq. B-21, Eq. B-13).
_CLIMB_K_HIGH = 0.95
#: Calibrated airspeed, kt, at which K switches (Eq. B-21, Eq. B-13).
_CLIMB_K_SPEED_KT = 200.0
#: The 0.95 of Eq. B-24 and Eq. B-32, the descending counterpart of K: the
#: accelerating segment's own allowance for an 8 kt headwind. Eq. B-26 divides
#: by it again, so the height gained over the segment is the uncorrected one.
_ACCELERATE_WIND_FACTOR = 0.95
#: The 1.03 of Eq. B-40, B-48, B-76 and B-77: what K is to a climb, for a
#: descent at constant calibrated airspeed into an 8 kt headwind.
_DESCENT_WIND_CONSTANT = 1.03
#: Engine breakpoint temperature T_B of Eq. B-11, degC (folio B-12).
_BREAK_TEMPERATURE_C = 30.0
#: The 0.006 per degC of Eq. B-11's high-temperature thrust lapse.
_HIGH_TEMPERATURE_LAPSE = 0.006
#: Horsepower to pounds of thrust at one knot, Eq. B-12
#: (550 ft.lbf/s per hp / 1.68781 ft/s per kt = 325.9, printed as 326).
_HP_TO_LB_KT = 326.0
#: Length of the transition segment inserted on a thrust-rating change (B6.1.6,
#: Eq. B-34) or at an arrival step boundary (B7.1.7, Eq. B-43), ft. Halved when
#: the step itself is shorter than twice this, so the inserted point can never
#: overtake the point it was inserted before.
_TRANSITION_LENGTH_FT = 1000.0
#: Engine-out climb gradient G', per cent, by number of engines (B4.3, folio
#: B-13). Aeroplanes with automatic thrust restoration take 0 instead; footnote
#: 35 records that only five types ever had it and no ANP field identifies them,
#: so they are reached only by passing the gradient explicitly.
_ENGINE_OUT_GRADIENT_PERCENT = {2: 1.2, 3: 1.5, 4: 1.7}

#: Iteration seed of the Accelerate step (B6.1.3): "The initial estimate is
#: Point2_Height = Point1_Height + 250 feet", ft.
_ACCELERATE_SEED_FT = 250.0
#: Convergence tolerance of the Accelerate step (B6.1.3): iterate "until the
#: difference between successive estimates of height at Point2 is less than (or
#: equal to) one foot", ft. The bound is closed, as the standard writes it.
_ACCELERATE_TOLERANCE_FT = 1.0
#: Iteration cap for the Accelerate step's fixed point.
#:
#: **Decision.** B6.1.3 gives a seed, a tolerance and a stop condition and no
#: bound on the number of passes, and says nothing about an iterate that
#: oscillates rather than converging. The
#: fixed point is a shallow one -- the gradient is nearly constant over the few
#: hundred feet a single Accelerate step climbs -- and every reference case of
#: Doc 29 Volume 3 settles in fewer than ten passes, so fifty is two decimal
#: orders of headroom over what convergence costs and still terminates. Reaching
#: it raises, naming the step and the last two iterates, because a profile that
#: silently returns a half-converged altitude is worse than no profile.
_ACCELERATE_ITERATION_LIMIT = 50
#: Gradient margin of B6.1.3: when ``a_max - G g`` falls below ``0.02 g`` "the
#: climb gradient can be limited to G = amax/g - 0.02, in effect reducing the
#: desired climb rate in order to maintain acceptable acceleration".
#:
#: **Decision.** The standard phrases the clamp permissively ("can be limited"),
#: so applying it is a choice. It is applied here, because declining it walks
#: straight into the abort below on the very cases the clamp exists for.
_ACCELERATE_GRADIENT_MARGIN = 0.02
#: Gradient floor of B6.1.3: "If G < 0.01 it should be concluded there is not
#: enough thrust to achieve the acceleration and climb specified; the
#: calculation should be terminated, and the procedure steps revised."
#:
#: **Decision.** B6.1.3 does not define "terminated"; the 4th edition's footnote
#: 30 asks only that "the computer model should be programmed to inform the user
#: of the inconsistency". It raises here, naming the step and the
#: gradient, rather than truncating the profile: a profile that stops mid-climb
#: reaches the noise calculation as a shorter flight, not as an error.
_ACCELERATE_GRADIENT_FLOOR = 0.01

#: High-temperature thrust rating for each rated one (Eq. B-10).
#:
#: **Decision.** Folio B-11 describes these rows as carrying "the 'HighTemp'
#: suffix", and no ANP release does: v2.3 spells them ``MaxTkoffHiTemp``,
#: ``MaxClimbHiTemp``, ``IdleApproachHiTemp``, ``ReduTkoffHiTemp``,
#: ``MaxContHiTemp`` and ``ReduceClimbHiTemp``, abbreviating the stem
#: differently in each case (``MaxTakeoff`` becomes ``MaxTkoff`` here and
#: ``ReduTkoff`` there). A rule written as a string operation therefore finds no
#: high-temperature coefficients for any aircraft in the fleet and falls back to
#: Eq. B-11 for all of them without saying so, which is why the pairing is a
#: table. Keys and values are compared case-folded.
_HIGH_TEMPERATURE_RATING = {
    "maxtakeoff": "MaxTkoffHiTemp",
    "maxclimb": "MaxClimbHiTemp",
    "idleapproach": "IdleApproachHiTemp",
    "reducetakeoff": "ReduTkoffHiTemp",
    "reduceclimb": "ReduceClimbHiTemp",
    "maxcontinuous": "MaxContHiTemp",
}
#: Thrust rating of a step whose thrust is solved from the force balance rather
#: than looked up: the level-flight thrust of Eq. B-30 (B6.1.4).
_ADAPTED_THRUST = "adaptedthrust"
#: Thrust rating of a step flown at the engine-out floor of Eq. B-13 (B4.3).
_MINIMUM_THRUST = "minimumthrust"
#: Thrust rating an idle descent reads its coefficients from (B7.1.2, B7.1.4).
_IDLE_APPROACH = "idleapproach"

#: ANP operation code for a departure, and for the ``D`` half of the merged
#: take-off/landing speed-coefficient column of ``Aerodynamic_coefficients``.
_DEPARTURE = "D"
#: ANP operation code for an arrival.
_ARRIVAL = "A"

_DEPARTURE_STEP_TYPES = ("takeoff", "climb", "accelerate", "level", "level-accelerate")
_APPROACH_STEP_TYPES = (
    "descend",
    "descend-decel",
    "descend-idle",
    "level",
    "level-decel",
    "level-idle",
    "land",
    "decelerate",
)
#: Approach step types flown at idle thrust. Figure B-3 exempts an idle step
#: followed by another idle step from the transition point every other change of
#: step type, flap or descent angle requires.
_IDLE_STEP_TYPES = ("descend-idle", "level-idle")
#: Approach step types that hold altitude, so their transition point sits at the
#: step's own start altitude and their Point2 thrust is their Point1 thrust.
_LEVEL_STEP_TYPES = ("level", "level-decel", "level-idle")
#: Spellings of a step type that mean the canonical one. Doc 29's prose writes
#: "Take-off" where every table writes "Takeoff", and underscores stand in for
#: hyphens in some exports; nothing else varies.
_STEP_TYPE_ALIASES = {"take-off": "takeoff", "land-decel": "decelerate"}


def _canonical_step_type(step_type: str) -> str:
    """Fold one step-type spelling onto the canonical lowercase, hyphenated one."""
    key = str(step_type).strip().lower().replace("_", "-")
    return _STEP_TYPE_ALIASES.get(key, key)


def _fold(name: str) -> str:
    """Fold an identifier for a case-insensitive, padding-insensitive lookup.

    The tables disagree with themselves about case and padding: Doc 29 Volume 3
    spells one thrust rating ``MaxTakeoff`` in its coefficient sheet and
    ``MaxTakeOff`` in the procedure that reads it, and flap identifiers arrive
    as ``'ZERO  '`` from one export and ``'ZERO'`` from another. A case-sensitive
    lookup of either finds nothing and reports a missing coefficient for a row
    that is there. Appendix B gives no normalisation rule for either identifier:
    B4.1 and B4.2 index their coefficient tables by the step's thrust rating and
    Eq. B-21 takes ``R`` from the ANP ``Aerodynamic_Coefficients`` table by its
    ``Flap_ID``, and neither says how the two spellings should be matched.
    """
    return str(name).strip().lower()


def _transition_offset(length_ft: float) -> float:
    """Distance back from the following point at which a transition sits, ft.

    Eq. B-34 and Eq. B-43 both place it 1000 ft into the step, halving that for
    a step shorter than 2000 ft so the inserted point stays inside its own
    segment instead of crossing the point it was inserted next to.
    """
    return min(0.5 * length_ft, _TRANSITION_LENGTH_FT)


# --------------------------------------------------------------------------
# Aerodrome and atmosphere (B3)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Aerodrome:
    """The aerodrome, its weather and the runway a procedure is flown from.

    Every atmosphere ratio of B3 is a function of altitude above mean sea level
    given these five numbers, and every equation of Appendix B reads at least
    one of them.

    :ivar elevation_ft: Aerodrome elevation above mean sea level ``Eapt``, ft.
    :ivar temperature_c: Air temperature at the aerodrome ``Tapt``, in degC.
        Doc 29 writes it in degF; it is taken in degC here because that is what
        the reference cases, the thrust equations and the rest of this package
        use, and converted on the way in.
    :ivar sea_level_pressure_inhg: Aerodrome pressure reduced to sea level
        ``Papt`` -- the QNH, not the pressure at the field -- in inHg.
    :ivar headwind_kt: Headwind component ``w``, kt; negative for a tailwind.
        Defaults to Doc 29's own modelling default of 8 kt (B4.4).
    :ivar runway_gradient: Runway gradient ``GR``, positive uphill,
        dimensionless: the rise over the run between the two runway ends.

    The validity envelope Doc 29 claims for the coefficients is "air
    temperatures up to 43 degrees C, aerodrome altitudes up to 6,000 ft and
    across the range of weights specified in the ANP database" (B1). Nothing
    here enforces it: outside it the equations still evaluate, and the
    coefficients, not the arithmetic, are what stop being adequate.
    """

    elevation_ft: float
    temperature_c: float = 15.0
    sea_level_pressure_inhg: float = _STANDARD_PRESSURE_INHG
    headwind_kt: float = DEFAULT_HEADWIND_KT
    runway_gradient: float = 0.0

    def __post_init__(self) -> None:
        """Reject an aerodrome the atmosphere equations cannot be evaluated at.

        Eq. B-4 raises the pressure ratio to the power ``1/5.256``, so a
        pressure of zero or less leaves real arithmetic altogether and surfaces
        as a bare ``ValueError`` from ``math.pow`` naming neither the field nor
        the aerodrome. A runway gradient of one or more is a runway climbing at
        45 degrees or steeper, which is not a runway; Eq. B-18 divides by
        ``a - g GR`` and a gradient that large drives the divisor negative, so
        the take-off would come back *shorter* the steeper the hill.

        A non-finite field is refused for all five: each is a measured or
        forecast condition that a caller either has or does not, and a ``NaN``
        elevation propagates silently through every altitude of the profile.

        :raises ValueError: if a field is not finite, the sea-level pressure is
            not positive, or the runway gradient is not below 1 in magnitude.
        """
        require_finite_fields(
            self,
            "elevation_ft",
            "temperature_c",
            "sea_level_pressure_inhg",
            "headwind_kt",
            "runway_gradient",
        )
        if self.sea_level_pressure_inhg <= 0.0:
            msg = (
                "Aerodrome: 'sea_level_pressure_inhg' must be positive; got "
                f"{self.sea_level_pressure_inhg!r} inHg."
            )
            raise ValueError(msg)
        if abs(self.runway_gradient) >= 1.0:
            msg = (
                "Aerodrome: 'runway_gradient' must be below 1 in magnitude (a "
                f"rise over a run, not an angle); got {self.runway_gradient!r}."
            )
            raise ValueError(msg)

    @property
    def _temperature_f(self) -> float:
        """Aerodrome temperature, degF (Eq. B-1 read the other way)."""
        return self.temperature_c * 9.0 / 5.0 + 32.0

    def temperature_c_at(self, altitude_ft: float) -> float:
        """Air temperature at *altitude_ft* above mean sea level, degC.

        Eq. B-2 lapses from the temperature *at the aerodrome*, not from a
        sea-level value: at field elevation the temperature is ``Tapt`` however
        high the field is. Eq. B-9 reads this as its ``T``.
        """
        lapsed_f = self._temperature_f - _LAPSE_F_PER_FT * (
            altitude_ft - self.elevation_ft
        )
        return (lapsed_f - 32.0) * 5.0 / 9.0

    def temperature_ratio(self, altitude_ft: float) -> float:
        """Temperature ratio ``theta`` at *altitude_ft* above MSL (Eq. B-3).

        Air temperature at the aeroplane over standard sea-level temperature,
        both absolute. Eq. B-16 reads it directly and Eq. B-5 divides by it.
        """
        return (
            _RANKINE_OFFSET
            + self._temperature_f
            - _LAPSE_F_PER_FT * (altitude_ft - self.elevation_ft)
        ) / _STANDARD_TEMPERATURE_R

    def pressure_ratio(self, altitude_ft: float) -> float:
        """Pressure ratio ``delta`` at *altitude_ft* above MSL (Eq. B-4).

        Ambient pressure over 29.92 inHg. Every force balance in Appendix B
        divides the weight by it, because ``W/delta`` is the weight the thrust
        equations' *corrected* thrust has to lift.

        :raises ValueError: above the tropopause the lapsed temperature Eq. B-4
            divides by reaches absolute zero and the bracket turns negative,
            where raising it to a fractional power leaves the reals; Python
            answers that with a complex number rather than an error, and a
            complex pressure ratio propagates silently into every thrust of the
            profile. Doc 29 claims the coefficients only up to 6,000 ft anyway.
        """
        base = (self.sea_level_pressure_inhg / _STANDARD_PRESSURE_INHG) ** (
            1.0 / _PRESSURE_EXPONENT
        )
        bracket = base - _LAPSE_F_PER_FT * altitude_ft / _STANDARD_TEMPERATURE_R
        if bracket <= 0.0:
            msg = (
                "Eq. B-4 has no real pressure ratio at "
                f"{altitude_ft!r} ft above mean sea level, which is at or above "
                "the altitude its standard atmosphere lapses to absolute zero."
            )
            raise ValueError(msg)
        return float(bracket**_PRESSURE_EXPONENT)

    def density_ratio(self, altitude_ft: float) -> float:
        """Density ratio ``sigma`` at *altitude_ft* above MSL (Eq. B-5).

        The ratio calibrated and true airspeed differ by: Eq. B-7 divides a
        calibrated airspeed by its square root to get the true one.
        """
        return self.pressure_ratio(altitude_ft) / self.temperature_ratio(altitude_ft)

    def pressure_altitude_ft(self, altitude_ft: float) -> float:
        """Pressure altitude ``h`` for *altitude_ft* above MSL, ft (Eq. B-6).

        The altitude the standard atmosphere would put this pressure at, which
        is what Eq. B-9's ``Ga h`` and ``Gb h^2`` terms read -- not the
        geometric altitude. The two coincide only at a QNH of exactly 29.92
        inHg; at 30.71 inHg over a sea-level aerodrome the aeroplane sits at
        0 ft and flies at a pressure altitude of -723 ft.
        """
        ratio = float(self.pressure_ratio(altitude_ft) ** (1.0 / _PRESSURE_EXPONENT))
        return (_STANDARD_TEMPERATURE_R / _LAPSE_F_PER_FT) * (1.0 - ratio)

    def true_airspeed_kt(
        self, calibrated_airspeed_kt: float, altitude_ft: float
    ) -> float:
        """True airspeed from a calibrated one at *altitude_ft* (Eq. B-7), kt."""
        return calibrated_airspeed_kt / math.sqrt(self.density_ratio(altitude_ft))

    def calibrated_airspeed_kt(
        self, true_airspeed_kt: float, altitude_ft: float
    ) -> float:
        """Calibrated airspeed from a true one at *altitude_ft* (Eq. B-8), kt."""
        return true_airspeed_kt * math.sqrt(self.density_ratio(altitude_ft))


#: The reference atmosphere the tabulated Descend-Idle and Level-Idle step
#: parameters were derived in: "provided for ISA reference conditions at a
#: sea-level airport" (folio B-36).
#:
#: **Decision.** Neither B7.1.2 nor B7.1.4 restates what that means, and only
#: one reading closes the algebra: a sea-level field, 15 degC,
#: 29.92 inHg, with the step's own height above the field as the altitude. The
#: headwind is not part of it -- Eq. B-57 and Eq. B-73 carry the *actual* wind
#: into the reference deceleration, which is what keeps the adjustment an
#: identity when the actual conditions are these.
_ISA_SEA_LEVEL = Aerodrome(elevation_ft=0.0, temperature_c=15.0)


# --------------------------------------------------------------------------
# Coefficient tables
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class JetEngineCoefficients:
    """One ANP ``Jet_Engine_Coefficients`` row: the Eq. B-9 thrust polynomial.

    ``CNT = E + F Vc + Ga h + Gb h^2 + H T`` gives the corrected net thrust per
    engine in lb, for one aeroplane and one thrust rating.

    :ivar e: Constant term ``E``, lb.
    :ivar f: Calibrated-airspeed coefficient ``F``, lb/kt.
    :ivar ga: Pressure-altitude coefficient ``Ga``, lb/ft.
    :ivar gb: Squared pressure-altitude coefficient ``Gb``, lb/ft2.
    :ivar h: Temperature coefficient ``H``, lb/degC.

    The units are the 5th edition's symbol list (folio B-2) and the Volume 3
    column headers; the 4th edition printed four units for these five symbols.
    """

    e: float
    f: float
    ga: float
    gb: float
    h: float

    def __post_init__(self) -> None:
        """Reject a coefficient row that would put a ``NaN`` in every thrust.

        A coefficient is a published table entry, never a measurement that can
        come back undetermined, and each of the five multiplies a term that is
        present at every profile point: a non-finite one does not surface where
        it was read but as a thrust of ``nan`` at the far end, which the NPD
        lookup then interpolates on and reports as a level.

        :raises ValueError: if any coefficient is not finite.
        """
        require_finite_fields(self, "e", "f", "ga", "gb", "h")

    def corrected_net_thrust_lb(
        self,
        *,
        calibrated_airspeed_kt: float,
        pressure_altitude_ft: float,
        temperature_c: float,
    ) -> float:
        """Corrected net thrust per engine, lb (Eq. B-9, and Eq. B-10 in kind).

        :param calibrated_airspeed_kt: Calibrated airspeed ``Vc``, kt.
        :param pressure_altitude_ft: Pressure altitude ``h`` of Eq. B-6, ft.
        :param temperature_c: Air temperature at the aeroplane ``T``, degC.
        """
        return (
            self.e
            + self.f * calibrated_airspeed_kt
            + self.ga * pressure_altitude_ft
            + self.gb * pressure_altitude_ft**2
            + self.h * temperature_c
        )


@dataclass(frozen=True)
class PropellerEngineCoefficients:
    """One ANP ``Propeller_Engine_Coefficients`` row: the Eq. B-12 thrust.

    ``CNT = (326 eta Pp / Vt) / delta`` for a piston or turboprop aeroplane.

    :ivar efficiency: Propeller efficiency ``eta``, dimensionless.
    :ivar power_hp: Installed net propulsive power ``Pp`` per engine, hp.
    """

    efficiency: float
    power_hp: float

    def __post_init__(self) -> None:
        """Reject an efficiency or power that cannot produce a thrust.

        Eq. B-12 is a ratio: the whole thrust is proportional to both fields, so
        a zero or negative one is not a weak engine but an aeroplane the model
        cannot fly, and a non-finite one reaches the profile as a ``NaN``
        thrust rather than as the row it was read from.

        :raises ValueError: if either field is not finite and positive.
        """
        require_finite_fields(self, "efficiency", "power_hp")
        for name in ("efficiency", "power_hp"):
            if getattr(self, name) <= 0.0:
                msg = (
                    f"PropellerEngineCoefficients: '{name}' must be positive; "
                    f"got {getattr(self, name)!r}."
                )
                raise ValueError(msg)

    def corrected_net_thrust_lb(
        self, *, true_airspeed_kt: float, pressure_ratio: float
    ) -> float:
        """Corrected net thrust per engine, lb (Eq. B-12).

        :param true_airspeed_kt: True airspeed ``Vt``, kt. Eq. B-12 is singular
            at rest, so the caller supplies the floor B4.2 pins for the ground
            roll: "the minimum value of V_T is assumed to be the initial climb
            speed", which at the take-off Point1 is the Point2 true airspeed.
        :param pressure_ratio: ``delta`` of Eq. B-4 at the point's altitude.
        :raises ValueError: if the true airspeed is not positive.
        """
        if not true_airspeed_kt > 0.0:
            msg = (
                "PropellerEngineCoefficients: 'true_airspeed_kt' must be "
                "positive, since Eq. B-12 divides by it; got "
                f"{true_airspeed_kt!r} kt."
            )
            raise ValueError(msg)
        return (
            _HP_TO_LB_KT * self.efficiency * self.power_hp / true_airspeed_kt
        ) / pressure_ratio


@dataclass(frozen=True)
class AerodynamicCoefficients:
    """One ANP ``Aerodynamic_Coefficients`` row: a flap configuration.

    :ivar drag_ratio: ``R``, the drag-over-lift ratio of the configuration,
        dimensionless. Every force balance in Appendix B carries it.
    :ivar ground_roll_coefficient: ``B``, ft/lb, of Eq. B-16, or ``None`` for a
        configuration no take-off is flown in.
    :ivar speed_coefficient: The take-off speed coefficient ``C`` of Eq. B-15 on
        a departure and the landing speed coefficient ``D`` of Eq. B-75 on an
        arrival, kt/sqrt(lb), or ``None`` for a configuration that is neither
        taken off nor landed in. One field for the two because no flap
        configuration is ever both: the ANP table keys them by operation and
        fills the matching column, and Doc 29 Volume 3 merges the pair into a
        single ``C/D`` column for the same reason.

    A missing coefficient is ``None``, the dash the printed table prints, not a
    zero: a zero ``B`` is a take-off with no ground roll at all.
    """

    drag_ratio: float
    ground_roll_coefficient: float | None = None
    speed_coefficient: float | None = None

    def __post_init__(self) -> None:
        """Reject a configuration whose drag ratio is missing or not finite.

        ``R`` is the one coefficient every row of the ANP table carries, because
        every force balance in Appendix B reads it -- the climb angle of
        Eq. B-21, the level thrust of Eq. B-30, the approach thrust of Eq. B-40.
        The other two are legitimately absent, so they are optional here and
        refused at the step that needs one, naming that step.

        :raises ValueError: if any coefficient present is not finite.
        """
        require_finite_fields(
            self, "drag_ratio", "ground_roll_coefficient", "speed_coefficient"
        )


@dataclass(frozen=True)
class PerformanceAircraft:
    """One aeroplane's Appendix B coefficient set.

    :ivar aircraft_id: ANP aircraft identifier.
    :ivar engines: Number of engines supplying thrust ``N``.
    :ivar max_static_thrust_lb: Maximum sea-level static thrust per engine, lb.
        Read only by Eq. B-79 and Eq. B-81, where a Decelerate step's start
        thrust is a percentage of it.
    :ivar max_landing_weight_lb: Maximum gross landing weight, lb. The approach
        weight is 90 % of it (Eq. B-75, Eq. B-76).
    :ivar jet_coefficients: Eq. B-9 coefficients per thrust rating.
    :ivar propeller_coefficients: Eq. B-12 coefficients per thrust rating.
    :ivar aerodynamic_coefficients: Flap configurations per
        ``(operation, flap identifier)``, with the operation ``"A"`` or ``"D"``.

    Which of the two thrust forms applies is decided by which table carries a
    row, not by the engine-type label: B4.1 is headed "jet and (certain)
    turboprop" for Eq. B-9 and B4.2 "piston and (some) turboprop" for Eq. B-12,
    and neither says which turboprop is which, so a turboprop appears under
    either heading and only its coefficient rows say under which.
    """

    aircraft_id: str
    engines: int
    max_static_thrust_lb: float
    max_landing_weight_lb: float
    jet_coefficients: Mapping[str, JetEngineCoefficients] = field(default_factory=dict)
    propeller_coefficients: Mapping[str, PropellerEngineCoefficients] = field(
        default_factory=dict
    )
    aerodynamic_coefficients: Mapping[tuple[str, str], AerodynamicCoefficients] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Reject an aeroplane no procedure could be flown with.

        The engine count divides in Eq. B-21, Eq. B-30 and every approach thrust
        equation, and Eq. B-13 divides by ``N - 1``, so a count of zero is a
        division by zero several equations deep and a count of one is an
        aeroplane the engine-out floor does not apply to at all. Both weights
        are read as denominators or as scale factors, never as differences, so
        neither may be zero, negative or undetermined.

        Whether the aeroplane carries jet or propeller coefficients is not
        checked here: an aeroplane with neither is a legitimate row of the
        aircraft table, and the step that needs a rating is where the absence
        can be reported against the rating it was looking for.

        :raises ValueError: if the engine count is below one, or a weight or
            thrust is not finite and positive.
        """
        require_finite_fields(self, "max_static_thrust_lb", "max_landing_weight_lb")
        if self.engines < 1:
            msg = (
                "PerformanceAircraft: 'engines' must be at least 1; got "
                f"{self.engines!r}."
            )
            raise ValueError(msg)
        for name in ("max_static_thrust_lb", "max_landing_weight_lb"):
            if getattr(self, name) <= 0.0:
                msg = (
                    f"PerformanceAircraft: '{name}' must be positive; got "
                    f"{getattr(self, name)!r}."
                )
                raise ValueError(msg)

    @property
    def approach_weight_lb(self) -> float:
        """Approach weight, lb: 90 % of the maximum landing weight (folio B-31).

        Not the ANP ``Default_weights`` arrival entry, which is a different
        number for most of the fleet; Doc 29 names the aircraft table's landing
        weight and the fraction explicitly, three times.
        """
        return APPROACH_WEIGHT_FRACTION * self.max_landing_weight_lb

    def flap(self, operation: str, flap_id: str) -> AerodynamicCoefficients:
        """Aerodynamic coefficients for one flap configuration.

        :param operation: ``"A"`` (arrival) or ``"D"`` (departure).
        :param flap_id: Flap identifier as the procedure spells it.
        :raises KeyError: if the aeroplane has no such configuration. Eq. B-21
            takes ``R`` from the ANP ``Aerodynamic_Coefficients`` table for the
            step's own ``Flap_ID`` and names no fallback for an identifier that
            is not in it, so this raises rather than substituting a default: a
            silently substituted drag ratio changes every climb angle of the
            profile and nothing downstream can tell.
        """
        wanted = (_fold(operation), _fold(flap_id))
        for (op, fid), value in self.aerodynamic_coefficients.items():
            if (_fold(op), _fold(fid)) == wanted:
                return value
        msg = (
            f"aircraft {self.aircraft_id!r}: no aerodynamic coefficients for "
            f"operation {operation!r}, flap {flap_id!r} (available: "
            f"{sorted((str(o), str(f)) for o, f in self.aerodynamic_coefficients)})."
        )
        raise KeyError(msg)


# --------------------------------------------------------------------------
# Procedural steps
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DepartureStep:
    """One row of an ANP departure procedural-step table (B6.1).

    :ivar step_type: ``"Takeoff"``, ``"Climb"``, ``"Accelerate"``, ``"Level"``
        or ``"Level-Accelerate"``, in whatever case and hyphenation the table
        spells it; :attr:`kind` is the folded form the model works in.
    :ivar thrust_rating: Thrust rating the step is flown at, as the table
        spells it. ``"AdaptedThrust"`` marks a Level step, whose thrust is
        solved rather than looked up, and ``"MinimumThrust"`` the engine-out
        floor of Eq. B-13.
    :ivar flap_id: Flap identifier, as the table spells it.
    :ivar end_altitude_ft: End-point height above the aerodrome of a Climb step,
        ft; an em dash for every other step type, which is why it is optional.
    :ivar rate_of_climb_ft_per_min: Rate of climb of an Accelerate step, ft/min.
    :ivar end_calibrated_airspeed_kt: End-point calibrated airspeed of an
        Accelerate or Level-Accelerate step, kt.
    :ivar energy_share_percent: Acceleration percentage (energy share factor) of
        an Accelerate or Level-Accelerate step, per cent.
    :ivar distance_ft: Track distance of a Level step, ft.
    :ivar bank_angle_deg: Bank angle ``eps`` over the step, degrees.

    Four of these are quantities the step type simply does not define, and the
    ANP table leaves each blank; they are ``None`` here and rendered as an em
    dash, never as a zero, since a zero rate of climb is a level acceleration
    and a zero distance is a step that goes nowhere.

    A step given both a rate of climb and an energy share factor keeps both, and
    the model prefers the energy share factor: "The ROC-values are altitude and
    atmosphere conditions dependent whereas ESF values adapt to changing airport
    elevations and atmosphere conditions" (B6.1.3, folio B-21), so of the two
    only the energy share factor still means what the manufacturer intended at
    another aerodrome. B6.1.3 leaves the choice to the model, putting it as
    advice: "it is preferable to use ESF values".

    The bank angle is an input rather than something derived, because Eq. B-14
    needs a turn radius and a turn radius needs the ground track, which this
    model does not build and Appendix B assumes it is given. Zero, the default,
    is straight flight,
    where every ``R/cos(eps)`` in Appendix B reduces to ``R``.
    """

    step_type: str
    thrust_rating: str
    flap_id: str
    end_altitude_ft: float | None = None
    rate_of_climb_ft_per_min: float | None = None
    end_calibrated_airspeed_kt: float | None = None
    energy_share_percent: float | None = None
    distance_ft: float | None = None
    bank_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        """Reject a step whose own type cannot be flown from what it carries.

        Each departure step type solves for a different unknown and reads a
        different parameter to do it (B6.1.1 to B6.1.5), so the parameter a type
        needs is not optional *for that type* even though it is absent from
        every other row of the same table. Left out, a Climb step reaches the
        solver with no end altitude and fails inside Eq. B-23 as an arithmetic
        error on ``None``, several frames from the row that was short.

        An Accelerate step needs a gradient from somewhere, and Doc 29 supplies
        two alternatives; the check is that at least one of them is present, not
        that exactly one is, because ANP datasets routinely publish both.

        :raises ValueError: if the step type is not one of the five of B6.1, or
            the parameters that type is solved from are missing.
        """
        kind = self.kind
        if kind not in _DEPARTURE_STEP_TYPES:
            msg = (
                f"DepartureStep: 'step_type' must be one of "
                f"{_DEPARTURE_STEP_TYPES}; got {self.step_type!r}."
            )
            raise ValueError(msg)
        require_finite_fields(
            self,
            "end_altitude_ft",
            "rate_of_climb_ft_per_min",
            "end_calibrated_airspeed_kt",
            "energy_share_percent",
            "distance_ft",
            "bank_angle_deg",
        )
        if kind == "climb" and self.end_altitude_ft is None:
            msg = (
                "DepartureStep: 'end_altitude_ft' must be given for a Climb "
                "step, which is solved for the distance that reaches it "
                "(Eq. B-23)."
            )
            raise ValueError(msg)
        if kind == "level" and self.distance_ft is None:
            msg = (
                "DepartureStep: 'distance_ft' must be given for a Level step, "
                "whose length is its only geometric input (Eq. B-29)."
            )
            raise ValueError(msg)
        if kind in ("accelerate", "level-accelerate"):
            if self.end_calibrated_airspeed_kt is None:
                msg = (
                    "DepartureStep: 'end_calibrated_airspeed_kt' must be given "
                    f"for a {self.step_type!r} step, which accelerates to it "
                    "(Eq. B-24, Eq. B-32)."
                )
                raise ValueError(msg)
            no_gradient = (
                self.rate_of_climb_ft_per_min is None
                and self.energy_share_percent is None
            )
            if kind == "accelerate" and no_gradient:
                msg = (
                    "DepartureStep: an Accelerate step must carry "
                    "'rate_of_climb_ft_per_min' or 'energy_share_percent', "
                    "which is what splits the excess thrust between "
                    "accelerating and climbing (Eq. B-25)."
                )
                raise ValueError(msg)

    @property
    def kind(self) -> str:
        """The step type folded onto Doc 29's own vocabulary, lowercase.

        The raw :attr:`step_type` is kept as the table spells it so a
        transcription stays diffable against the sheet it came from; this is
        what the model branches on.
        """
        return _canonical_step_type(self.step_type)


@dataclass(frozen=True)
class ApproachStep:
    """One row of an ANP approach procedural-step table (B7.1).

    :ivar step_type: ``"Descend"``, ``"Descend-Decel"``, ``"Descend-Idle"``,
        ``"Level"``, ``"Level-Decel"``, ``"Level-Idle"``, ``"Land"`` or
        ``"Decelerate"``, in whatever case the table spells it.
    :ivar flap_id: Flap identifier, as the table spells it.
    :ivar start_altitude_ft: Height above the aerodrome at the *start* of the
        step, ft. An approach step is anchored at its top, not its bottom.
    :ivar start_calibrated_airspeed_kt: Calibrated airspeed at the start of the
        step, kt.
    :ivar descent_angle_deg: Descent angle, degrees, **positive by convention**
        as the 5th edition declares it and as the ANP tables store it. The 4th
        edition took it negative and wrote its equations to suit.
    :ivar touchdown_roll_ft: Distance from touchdown to the Land step's Point2,
        ft; defined only for a Land step.
    :ivar distance_ft: Track length of a Level, Level-Decel, Level-Idle or
        Decelerate step, ft.
    :ivar start_thrust_percent: Start thrust of a Decelerate step, as a
        percentage of maximum sea-level static thrust (Eq. B-79, Eq. B-81).
    :ivar bank_angle_deg: Bank angle ``eps`` over the step, degrees.

    Doc 29 Volume 3's own workbook keeps a Level-Idle step's length in the
    *descent angle* column, with the distance column empty. That is a defect of
    that workbook, not of the format: the length belongs in
    :attr:`distance_ft` here, where the ANP release also puts it.
    """

    step_type: str
    flap_id: str
    start_altitude_ft: float | None = None
    start_calibrated_airspeed_kt: float | None = None
    descent_angle_deg: float | None = None
    touchdown_roll_ft: float | None = None
    distance_ft: float | None = None
    start_thrust_percent: float | None = None
    bank_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        """Reject a step whose own type cannot be flown from what it carries.

        The approach sweep runs backwards, each step computing its own Point1
        from the next step's, so a missing parameter does not stop at the row
        that lacks it: a Descend step with no descent angle divides by
        ``tan(None)`` while placing the step *above* it, and the first thing the
        caller sees is a profile whose top is somewhere else entirely.

        A Land step is checked for its touchdown roll and a Decelerate step for
        its length and start thrust, both of which are read forward from
        touchdown rather than backwards (Eq. B-78 to B-81). The terminal
        Decelerate step of every published procedure carries a length of zero
        and emits no point, so a length of zero is accepted and a missing one
        is not.

        :raises ValueError: if the step type is not one of the eight of B7.1, or
            the parameters that type is solved from are missing.
        """
        kind = self.kind
        if kind not in _APPROACH_STEP_TYPES:
            msg = (
                f"ApproachStep: 'step_type' must be one of "
                f"{_APPROACH_STEP_TYPES}; got {self.step_type!r}."
            )
            raise ValueError(msg)
        require_finite_fields(
            self,
            "start_altitude_ft",
            "start_calibrated_airspeed_kt",
            "descent_angle_deg",
            "touchdown_roll_ft",
            "distance_ft",
            "start_thrust_percent",
            "bank_angle_deg",
        )
        if kind == "land":
            if self.touchdown_roll_ft is None:
                msg = (
                    "ApproachStep: 'touchdown_roll_ft' must be given for a Land "
                    "step, which is where its Point2 sits (B7.1.5)."
                )
                raise ValueError(msg)
        elif kind == "decelerate":
            if self.distance_ft is None or self.start_thrust_percent is None:
                msg = (
                    "ApproachStep: 'distance_ft' and 'start_thrust_percent' "
                    "must both be given for a Decelerate step, which reads its "
                    "end values from the step that follows it (Eq. B-80, "
                    f"Eq. B-81); got {self.distance_ft!r} ft and "
                    f"{self.start_thrust_percent!r} %."
                )
                raise ValueError(msg)
        else:
            if self.start_altitude_ft is None:
                msg = (
                    "ApproachStep: 'start_altitude_ft' must be given for an "
                    "airborne step, which is anchored at its top rather than "
                    f"its bottom; {self.step_type!r} got "
                    f"{self.start_altitude_ft!r} ft."
                )
                raise ValueError(msg)
            # A plain Level step is the one airborne step whose speed the table
            # may leave out, and several ANP entries do: it holds its speed by
            # definition, so the speed at its top is the speed at its bottom,
            # which the step below it already fixes. Every other airborne step
            # decelerates or changes height, and reads the start speed to say by
            # how much.
            if kind != "level" and self.start_calibrated_airspeed_kt is None:
                msg = (
                    "ApproachStep: 'start_calibrated_airspeed_kt' must be given "
                    f"for a {self.step_type!r} step, whose true airspeed at the "
                    "top of the step comes from it (Eq. B-39, Eq. B-49, "
                    "Eq. B-61)."
                )
                raise ValueError(msg)
            if kind.startswith("descend") and not self.descent_angle_deg:
                msg = (
                    "ApproachStep: 'descent_angle_deg' must be given and "
                    f"non-zero for a {self.step_type!r} step, whose length is "
                    "the height it loses over its tangent (Eq. B-42); got "
                    f"{self.descent_angle_deg!r} deg."
                )
                raise ValueError(msg)
            if kind in _LEVEL_STEP_TYPES and self.distance_ft is None:
                msg = (
                    "ApproachStep: 'distance_ft' must be given for a "
                    f"{self.step_type!r} step, whose length is its only "
                    "geometric input (Eq. B-64, Eq. B-68)."
                )
                raise ValueError(msg)

    @property
    def kind(self) -> str:
        """The step type folded onto Doc 29's own vocabulary, lowercase."""
        return _canonical_step_type(self.step_type)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProfilePoint:
    """One point of a Doc 29 flight profile.

    :ivar distance_ft: Distance along the ground track, ft. Measured from brake
        release on a departure, and from touchdown on an arrival, where it is
        negative while the aeroplane is still airborne (folio B-5).
    :ivar altitude_ft: Height above the aerodrome elevation, ft.
    :ivar true_airspeed_kt: True airspeed, kt.
    :ivar corrected_net_thrust_lb: Corrected net thrust ``Fn/delta`` per engine,
        lb. This is the power setting the NPD tables are indexed on.
    """

    distance_ft: float
    altitude_ft: float
    true_airspeed_kt: float
    corrected_net_thrust_lb: float

    def __post_init__(self) -> None:
        """Reject a point that is not somewhere a profile can pass through.

        All four are required and finite: nothing about a profile point is
        legitimately undetermined, since each is solved from the step's own
        parameters rather than measured, and a ``NaN`` altitude or thrust
        reaches the NPD interpolation as a level rather than as an error.

        Height and true airspeed are refused below zero -- Doc 29 heights are
        above the aerodrome and its profiles neither dig nor fly backwards --
        while distance is signed on purpose, because an arrival is solved
        backwards from touchdown.

        **Corrected net thrust is not required positive.** An idle descent
        legitimately returns a negative one: with the reference turbofan's own
        published idle coefficients, 3000 ft at 250 kt gives
        ``1100 - 6.5(250) + 0.17(3000) - 1e-5(3000)^2 = -105 lb``, which is
        drag, not a defect, and the reference results tabulate it.

        :raises ValueError: if a field is not finite, or the height or true
            airspeed is negative.
        """
        require_finite_fields(
            self,
            "distance_ft",
            "altitude_ft",
            "true_airspeed_kt",
            "corrected_net_thrust_lb",
        )
        for name in ("altitude_ft", "true_airspeed_kt"):
            if getattr(self, name) < 0.0:
                msg = (
                    f"ProfilePoint: '{name}' must be non-negative; got "
                    f"{getattr(self, name)!r}."
                )
                raise ValueError(msg)


@dataclass(frozen=True)
class FlightProfile:
    """A flight profile: the fixed-point trajectory a procedure flies (B1).

    :ivar aircraft_id: ANP aircraft identifier.
    :ivar operation: ``"D"`` (departure) or ``"A"`` (arrival).
    :ivar procedure_id: Identifier of the procedure the steps came from.
    :ivar points: The profile points, ordered along the ground track.

    This is the vertical-plane half of a Doc 29 flight path. Section 3.6 is what
    turns it into three dimensions -- splitting segments at ground-track nodes,
    sub-segmenting the rolls and the climb, merging in the ground track -- and
    none of that happens here.
    """

    aircraft_id: str
    operation: str
    procedure_id: str
    points: tuple[ProfilePoint, ...]

    def __post_init__(self) -> None:
        """Reject a profile that does not run one way along its own track.

        A profile is an ordered path, and everything downstream reads it as one:
        the segmentation of Doc 29 3.6 interpolates *within* each segment, the
        noise calculation integrates along it, and the figure joins the points
        in the order it is handed. A pair out of order therefore does not fail,
        it draws and integrates a segment that doubles back, and on an arrival
        -- solved backwards from touchdown, where a sign error puts the top of
        the descent on the wrong side of the runway -- that is exactly the bug
        this guard is here to catch.

        Distances may repeat: a step of zero length is a legitimate degenerate
        case, and a transition point placed at the boundary of one coincides
        with the point after it. They may not decrease.

        Two points is the shortest profile that has a direction; a Take-off step
        alone emits exactly that.

        :raises ValueError: if the operation is not ``"A"`` or ``"D"``, fewer
            than two points are given, or the distances decrease.
        """
        if self.operation not in (_ARRIVAL, _DEPARTURE):
            msg = (
                f"FlightProfile: 'operation' must be {_ARRIVAL!r} (arrival) or "
                f"{_DEPARTURE!r} (departure); got {self.operation!r}."
            )
            raise ValueError(msg)
        least_points = 2
        if len(self.points) < least_points:
            msg = (
                "FlightProfile: 'points' must carry at least two points, the "
                f"shortest path that has a direction; got {len(self.points)}."
            )
            raise ValueError(msg)
        for index, (before, after) in enumerate(
            zip(self.points, self.points[1:], strict=False)
        ):
            if after.distance_ft < before.distance_ft:
                msg = (
                    "FlightProfile: 'points' must not decrease in distance "
                    f"along the track; point {index + 2} is at "
                    f"{after.distance_ft!r} ft, behind point {index + 1} at "
                    f"{before.distance_ft!r} ft."
                )
                raise ValueError(msg)

    def _column(self, name: str) -> NDArray[np.float64]:
        return np.asarray([getattr(p, name) for p in self.points], dtype=np.float64)

    @property
    def distance_ft(self) -> NDArray[np.float64]:
        """Distance along the ground track per point, ft."""
        return self._column("distance_ft")

    @property
    def altitude_ft(self) -> NDArray[np.float64]:
        """Height above the aerodrome per point, ft."""
        return self._column("altitude_ft")

    @property
    def true_airspeed_kt(self) -> NDArray[np.float64]:
        """True airspeed per point, kt."""
        return self._column("true_airspeed_kt")

    @property
    def corrected_net_thrust_lb(self) -> NDArray[np.float64]:
        """Corrected net thrust per engine per point, lb."""
        return self._column("corrected_net_thrust_lb")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the profile's height and thrust against distance along the track."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_flight_profile

        return plot_flight_profile(
            self, ax=ax, language=check_language(language), **kwargs
        )


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Flight:
    """One aeroplane at one weight in one atmosphere: the shared solver state.

    Every equation of Appendix B reads some of these four and nothing else, so
    they are gathered rather than threaded through a dozen parameter lists.
    """

    aircraft: PerformanceAircraft
    aerodrome: Aerodrome
    weight_lb: float
    operation: str

    # -- atmosphere and thrust ---------------------------------------------
    def altitude_ft(self, height_ft: float) -> float:
        """Altitude above mean sea level for a height above the aerodrome, ft."""
        return self.aerodrome.elevation_ft + height_ft

    def corrected_weight_lb(self, altitude_ft: float) -> float:
        """``W/delta``, the corrected weight the thrust equations balance, lb."""
        return self.weight_lb / self.aerodrome.pressure_ratio(altitude_ft)

    def rated_thrust_lb(
        self,
        rating: str,
        *,
        altitude_ft: float,
        calibrated_airspeed_kt: float,
        true_airspeed_kt: float,
    ) -> float:
        """Corrected net thrust per engine at a rated thrust setting, lb (B4).

        Jet or propeller is decided by which coefficient table carries a row for
        this aeroplane, B4.1 and B4.2 splitting the turboprops between them
        without a rule. For the jet form, the high-temperature
        value is computed too and **the smaller of the two is retained** (folio
        B-12), unconditionally: the 5th edition dropped the 4th's test against a
        break temperature, which no ANP table publishes per aircraft anyway.

        The high-temperature value comes from the paired ``HiTemp`` rating when
        the aeroplane has one, and otherwise from Eq. B-11's fallback, which is
        applied on departures only -- "For approach procedures, high-temperature
        thrust estimation is less relevant" (folio B-12).

        :raises KeyError: if neither coefficient table carries this rating.
        :raises ValueError: if both do, since Eq. B-9 and Eq. B-12 then disagree
            about the same aeroplane's thrust and neither B4.1 nor B4.2 breaks
            the tie.
        """
        jet = self.aircraft.jet_coefficients
        propeller = self.aircraft.propeller_coefficients
        folded = _fold(rating)
        prop_row = next((v for k, v in propeller.items() if _fold(k) == folded), None)
        row = next((v for k, v in jet.items() if _fold(k) == folded), None)
        if prop_row is not None and row is not None:
            msg = (
                f"aircraft {self.aircraft.aircraft_id!r} carries both jet and "
                f"propeller coefficients for thrust rating {rating!r}, and "
                "Eq. B-9 and Eq. B-12 disagree about what its thrust is. The "
                "ANP database keeps the two tables disjoint; this export does "
                "not."
            )
            raise ValueError(msg)
        if prop_row is not None:
            return prop_row.corrected_net_thrust_lb(
                true_airspeed_kt=true_airspeed_kt,
                pressure_ratio=self.aerodrome.pressure_ratio(altitude_ft),
            )
        if row is None:
            msg = (
                f"aircraft {self.aircraft.aircraft_id!r}: no engine "
                f"coefficients for thrust rating {rating!r} (jet ratings: "
                f"{sorted(str(k) for k in jet)}; propeller ratings: "
                f"{sorted(str(k) for k in propeller)})."
            )
            raise KeyError(msg)
        pressure_altitude_ft = self.aerodrome.pressure_altitude_ft(altitude_ft)
        temperature_c = self.aerodrome.temperature_c_at(altitude_ft)
        thrust = row.corrected_net_thrust_lb(
            calibrated_airspeed_kt=calibrated_airspeed_kt,
            pressure_altitude_ft=pressure_altitude_ft,
            temperature_c=temperature_c,
        )
        high_name = _HIGH_TEMPERATURE_RATING.get(folded)
        high_row = (
            next((v for k, v in jet.items() if _fold(k) == _fold(high_name)), None)
            if high_name is not None
            else None
        )
        if high_row is not None:
            high = high_row.corrected_net_thrust_lb(
                calibrated_airspeed_kt=calibrated_airspeed_kt,
                pressure_altitude_ft=pressure_altitude_ft,
                temperature_c=temperature_c,
            )
        elif self.operation == _DEPARTURE:
            # Eq. B-11. It carries no altitude terms, so above the break
            # temperature the corrected net thrust stops depending on pressure
            # altitude at all: the reference cases report the same 23170.7 lb at
            # brake release for a sea-level and a 5000 ft aerodrome at 40 degC.
            lapse = (1.0 - _HIGH_TEMPERATURE_LAPSE * temperature_c) / (
                1.0 - _HIGH_TEMPERATURE_LAPSE * _BREAK_TEMPERATURE_C
            )
            high = (
                row.f * calibrated_airspeed_kt
                + (row.e + row.h * _BREAK_TEMPERATURE_C) * lapse
            )
        else:
            high = thrust
        return min(thrust, high)

    def engine_out_thrust_lb(
        self,
        *,
        altitude_ft: float,
        calibrated_airspeed_kt: float,
        drag_ratio: float,
        bank_rad: float,
        engine_out_gradient_percent: float | None = None,
    ) -> float:
        """The minimum reduced ("deep cutback") thrust, lb (B4.3, Eq. B-13).

        Not a rated thrust but a floor: the thrust that would still meet the
        engine-out climb gradient the certification case requires, then flown
        with every engine running. Footnote 36 records that the model knowingly
        understates it, since "the R coefficient is not adjusted".

        :raises ValueError: for a single-engine aeroplane, where the ``N - 1``
            denominator has nothing to divide by and the safety case the
            gradient comes from does not exist.
        """
        engines = self.aircraft.engines
        if engines < 2:  # noqa: PLR2004 -- the N-1 denominator of Eq. B-13
            msg = (
                "Eq. B-13 minimum reduced thrust does not apply to a "
                f"single-engine aeroplane; aircraft "
                f"{self.aircraft.aircraft_id!r} has {engines} engine."
            )
            raise ValueError(msg)
        gradient = (
            _ENGINE_OUT_GRADIENT_PERCENT.get(engines, _ENGINE_OUT_GRADIENT_PERCENT[4])
            if engine_out_gradient_percent is None
            else engine_out_gradient_percent
        )
        k_factor = (
            _CLIMB_K_LOW
            if calibrated_airspeed_kt <= _CLIMB_K_SPEED_KT
            else _CLIMB_K_HIGH
        )
        return (self.corrected_weight_lb(altitude_ft) / (engines - 1)) * (
            math.sin(math.atan(0.01 * gradient)) / k_factor
            + drag_ratio / math.cos(bank_rad)
        )

    def level_thrust_lb(
        self, *, altitude_ft: float, drag_ratio: float, bank_rad: float
    ) -> float:
        """Thrust that holds height and speed, lb (Eq. B-30, Eq. B-62).

        Eq. B-21 solved for the thrust at a climb angle of zero. This is what a
        Level step's ``AdaptedThrust`` rating means: an output, not a lookup.
        """
        return (
            drag_ratio
            / (self.aircraft.engines * math.cos(bank_rad))
            * self.corrected_weight_lb(altitude_ft)
        )

    def step_thrust_lb(
        self,
        rating: str,
        *,
        altitude_ft: float,
        calibrated_airspeed_kt: float,
        true_airspeed_kt: float,
        drag_ratio: float,
        bank_rad: float,
    ) -> float:
        """Corrected net thrust for one departure step's rating, lb.

        Dispatches the three ways a departure step's thrust can be arrived at: a
        rated lookup (B4.1, B4.2), the engine-out floor (B4.3) and the
        level-flight balance (B6.1.4). The two computed ones are named as
        ratings by the ANP tables even though neither is a row of a coefficient
        table: B4.3 and B6.1.4 define them as things to solve for, and only the
        procedure tables spell them in the thrust rating column.
        """
        folded = _fold(rating)
        if folded == _ADAPTED_THRUST:
            return self.level_thrust_lb(
                altitude_ft=altitude_ft, drag_ratio=drag_ratio, bank_rad=bank_rad
            )
        if folded == _MINIMUM_THRUST:
            return self.engine_out_thrust_lb(
                altitude_ft=altitude_ft,
                calibrated_airspeed_kt=calibrated_airspeed_kt,
                drag_ratio=drag_ratio,
                bank_rad=bank_rad,
            )
        return self.rated_thrust_lb(
            rating,
            altitude_ft=altitude_ft,
            calibrated_airspeed_kt=calibrated_airspeed_kt,
            true_airspeed_kt=true_airspeed_kt,
        )

    # -- wind ---------------------------------------------------------------
    def _ground_speed_ratio(self, speed_kt: float) -> float:
        """Ground speed actually flown over the one Doc 29's 8 kt reference assumed.

        :raises ValueError: if the headwind reaches the speed being corrected,
            where the aeroplane makes no progress over the ground and the ratio
            changes sign rather than growing. B4.4 fixes the 8 kt reference wind
            and sets no bound on the actual one, while Eq. B-17 and Eq. B-22
            both carry it into a denominator; Doc 29's own reference cases stay
            between -5 and 8 kt.
        """
        reference = speed_kt - DEFAULT_HEADWIND_KT
        actual = speed_kt - self.aerodrome.headwind_kt
        if reference <= 0.0 or actual <= 0.0:
            msg = (
                "the headwind must stay below the airspeed it is corrected "
                f"against (Eq. B-17, Eq. B-22); {self.aerodrome.headwind_kt!r} "
                f"kt of headwind against {speed_kt!r} kt leaves no ground speed."
            )
            raise ValueError(msg)
        return actual / reference

    def wind_scaled_distance(
        self, distance_ft: float, speed_kt: float, *, squared: bool = False
    ) -> float:
        """Rescale a ground distance from the 8 kt reference wind to the actual one.

        Eq. B-17 (squared, because a ground roll goes as the square of the
        ground speed reached) and Eq. B-28 and B-33 (not, because an
        acceleration segment's length goes with the mean ground speed). Less
        headwind than the reference means more ground covered for the same air
        distance, so the correction lengthens the segment.
        """
        ratio = self._ground_speed_ratio(speed_kt)
        return distance_ft * (ratio**2 if squared else ratio)

    def wind_scaled_angle(
        self, gamma_rad: float, calibrated_airspeed_kt: float
    ) -> float:
        """Rescale a climb angle from the 8 kt reference wind to the actual one.

        Eq. B-22, and the *reciprocal* of the distance correction above: the air
        distance the climb takes is fixed, so covering more ground over it makes
        the angle seen from the ground shallower, not steeper.
        """
        return gamma_rad / self._ground_speed_ratio(calibrated_airspeed_kt)


def _drag_ratio(flight: _Flight, step: DepartureStep | ApproachStep) -> float:
    """``R`` of the step's own flap configuration."""
    return flight.aircraft.flap(flight.operation, step.flap_id).drag_ratio


# --------------------------------------------------------------------------
# Departure (B6)
# --------------------------------------------------------------------------
def _takeoff_points(
    flight: _Flight, step: DepartureStep
) -> tuple[ProfilePoint, ProfilePoint]:
    """Brake release and rotation, the two points of a Take-off step (B6.1.1).

    The ground roll is not the real one. Eq. B-16 gives an *equivalent* take-off
    distance, "the distance along the runway from brake release to the point
    where a straight-line extension of the initial landing-gear-retracted climb
    flight path intersects the runway" (folio B-15), which is why the rotation
    point sits at zero height and a distance no wheels leave the ground at.
    """
    aero = flight.aircraft.flap(flight.operation, step.flap_id)
    if aero.speed_coefficient is None or aero.ground_roll_coefficient is None:
        msg = (
            f"aircraft {flight.aircraft.aircraft_id!r}: flap {step.flap_id!r} "
            "carries no take-off speed coefficient C or ground-roll "
            "coefficient B, so Eq. B-15 and Eq. B-16 cannot be evaluated for a "
            "Take-off step flown in it."
        )
        raise ValueError(msg)
    altitude_ft = flight.altitude_ft(0.0)
    # Eq. B-15: the take-off calibrated airspeed grows as the square root of the
    # weight, so a heavier aeroplane rotates faster as well as later.
    takeoff_cas_kt = aero.speed_coefficient * math.sqrt(flight.weight_lb)
    rotation_tas_kt = flight.aerodrome.true_airspeed_kt(takeoff_cas_kt, altitude_ft)
    rotation_thrust_lb = flight.step_thrust_lb(
        step.thrust_rating,
        altitude_ft=altitude_ft,
        calibrated_airspeed_kt=takeoff_cas_kt,
        true_airspeed_kt=rotation_tas_kt,
        drag_ratio=aero.drag_ratio,
        bank_rad=math.radians(step.bank_angle_deg),
    )
    # Eq. B-16. The weight enters squared and through the pressure ratio, so a
    # heavier, hotter or higher take-off runs quadratically longer.
    roll_8kt_ft = (
        aero.ground_roll_coefficient
        * flight.aerodrome.temperature_ratio(altitude_ft)
        * flight.corrected_weight_lb(altitude_ft) ** 2
        / (flight.aircraft.engines * rotation_thrust_lb)
    )
    # Eq. B-17 before Eq. B-18: the gradient correction reads the wind-corrected
    # distance, so the order is fixed by the algebra rather than by preference.
    roll_wind_ft = flight.wind_scaled_distance(
        roll_8kt_ft, takeoff_cas_kt, squared=True
    )
    roll_ft = _gradient_corrected_roll_ft(flight, roll_wind_ft, rotation_tas_kt)
    return (
        ProfilePoint(
            distance_ft=0.0,
            altitude_ft=0.0,
            true_airspeed_kt=0.0,
            # Brake release is at rest, and Eq. B-12 divides by the true
            # airspeed, so B4.2 pins its floor at the rotation speed: "It is
            # actually assumed to be the True Airspeed at the take-off rotation
            # point (Point2)". The jet form takes the calibrated airspeed of
            # zero it really has.
            corrected_net_thrust_lb=flight.step_thrust_lb(
                step.thrust_rating,
                altitude_ft=altitude_ft,
                calibrated_airspeed_kt=0.0,
                true_airspeed_kt=rotation_tas_kt,
                drag_ratio=aero.drag_ratio,
                bank_rad=math.radians(step.bank_angle_deg),
            ),
        ),
        ProfilePoint(
            distance_ft=roll_ft,
            altitude_ft=0.0,
            true_airspeed_kt=rotation_tas_kt,
            corrected_net_thrust_lb=rotation_thrust_lb,
        ),
    )


def _gradient_corrected_roll_ft(
    flight: _Flight, roll_wind_ft: float, rotation_tas_kt: float
) -> float:
    """Eq. B-18: the ground roll on a sloping runway, ft.

    The mean acceleration along the runway is the one that reaches the rotation
    *true* airspeed over the wind-corrected distance, and gravity takes
    ``g GR`` of it back uphill.

    Two departures from the printed text, both deliberate and neither
    detectable against Doc 29's own reference cases, which carry no runway
    gradient at all:

    * ``k^2`` is restored. Folio B-17 declares ``a`` to be "the average
      acceleration (ft/s^2)" and writes it as a squared speed in *knots* over a
      distance in feet, which is not an acceleration in any unit; the factor
      1.68781^2 that Eq. B-24 and Eq. B-41 carry explicitly is missing. Left
      out, ``a`` is understated 2.85-fold and the gradient correction
      correspondingly overstated: at a 1 % upslope the take-off distance comes
      out 8.8 % long.
    * The true airspeed is ``Vc/sqrt(sigma)``, the 5th edition's placement. The
      4th edition prints ``Vc sqrt(sigma)``, which is not a speed the aeroplane
      ever has and would make the acceleration *rise* with altitude. The two
      agree at sea level, which is presumably how the disagreement survived.
    """
    speed_ft_s = _KT_FT_S * rotation_tas_kt
    acceleration = speed_ft_s**2 / (2.0 * roll_wind_ft)
    resisted = acceleration - _G_FT_S2 * flight.aerodrome.runway_gradient
    if resisted <= 0.0:
        msg = (
            "the runway gradient must leave a positive mean acceleration "
            f"(Eq. B-18); {flight.aerodrome.runway_gradient!r} takes "
            f"{_G_FT_S2 * flight.aerodrome.runway_gradient:.3f} ft/s^2 from a "
            f"mean acceleration of {acceleration:.3f} ft/s^2."
        )
        raise ValueError(msg)
    return roll_wind_ft * acceleration / resisted


def _climb_point(
    flight: _Flight, step: DepartureStep, start: ProfilePoint
) -> ProfilePoint:
    """The end point of a Climb step (B6.1.2).

    Calibrated airspeed and thrust rating are held; the true airspeed rises on
    its own as the air thins (Eq. B-19), and the mean climb angle of Eq. B-21 is
    what the excess of thrust over drag buys.
    """
    end_height_ft = float(step.end_altitude_ft or 0.0)
    start_altitude_ft = flight.altitude_ft(start.altitude_ft)
    end_altitude_ft = flight.altitude_ft(end_height_ft)
    bank_rad = math.radians(step.bank_angle_deg)
    drag_ratio = _drag_ratio(flight, step)
    # Eq. B-19 and Eq. B-20: constant calibrated airspeed, rising true airspeed.
    end_tas_kt = start.true_airspeed_kt * math.sqrt(
        flight.aerodrome.density_ratio(start_altitude_ft)
        / flight.aerodrome.density_ratio(end_altitude_ft)
    )
    cas_kt = flight.aerodrome.calibrated_airspeed_kt(end_tas_kt, end_altitude_ft)
    end_thrust_lb = flight.step_thrust_lb(
        step.thrust_rating,
        altitude_ft=end_altitude_ft,
        calibrated_airspeed_kt=cas_kt,
        true_airspeed_kt=end_tas_kt,
        drag_ratio=drag_ratio,
        bank_rad=bank_rad,
    )
    # Eq. B-21 is a mid-step balance: both the thrust and the corrected weight
    # are evaluated halfway up, including when the thrust is Eq. B-13's floor,
    # whose own corrected weight then cancels against the one below it.
    #
    # The calibrated airspeed is held over the whole step, so the jet form reads
    # it unchanged and only the pressure altitude moves. For the propeller form,
    # which reads a *true* airspeed, Eq. B-21 prints the root mean square of the
    # two endpoint values, V_T = sqrt(0.5 ((Point2_TAS)^2 + (Point1_TAS)^2)).
    # This model does not use that expression, because it contradicts the
    # mid-step altitude the same page names for the same quantity: at a held
    # calibrated airspeed the true airspeed halfway up is Eq. B-7 evaluated
    # there, not a mean of the ends. The printed mean is the larger, by 0.0225
    # kt on a 1500 ft climb and 0.1066 kt on a 2500 ft one, and Eq. B-12 puts
    # thrust in inverse proportion to it, so the printed reading buys a
    # shallower climb: it lays the last point of case 56 down 544.9 ft long,
    # against the 0.15 ft the departure distances are otherwise matched to.
    # Recorded under Eq. (B-21) in docs/ERRATA.md.
    mid_altitude_ft = 0.5 * (start_altitude_ft + end_altitude_ft)
    mid_tas_kt = flight.aerodrome.true_airspeed_kt(cas_kt, mid_altitude_ft)
    mid_thrust_lb = flight.step_thrust_lb(
        step.thrust_rating,
        altitude_ft=mid_altitude_ft,
        calibrated_airspeed_kt=cas_kt,
        true_airspeed_kt=mid_tas_kt,
        drag_ratio=drag_ratio,
        bank_rad=bank_rad,
    )
    k_factor = _CLIMB_K_LOW if cas_kt <= _CLIMB_K_SPEED_KT else _CLIMB_K_HIGH
    sin_gamma = k_factor * (
        flight.aircraft.engines
        * mid_thrust_lb
        / flight.corrected_weight_lb(mid_altitude_ft)
        - drag_ratio / math.cos(bank_rad)
    )
    if not 0.0 < sin_gamma < 1.0:
        msg = (
            "the thrust available over a Climb step must exceed its drag "
            f"(Eq. B-21); step {step.step_type!r} at {step.thrust_rating!r} to "
            f"{end_height_ft} ft gives sin(gamma) = {sin_gamma!r}, which is "
            "not a climb."
        )
        raise ValueError(msg)
    gamma_rad = math.asin(sin_gamma)
    gamma_wind_rad = flight.wind_scaled_angle(gamma_rad, cas_kt)
    # Eq. B-23.
    distance_ft = start.distance_ft + (end_height_ft - start.altitude_ft) / math.tan(
        gamma_wind_rad
    )
    return ProfilePoint(
        distance_ft=distance_ft,
        altitude_ft=end_height_ft,
        true_airspeed_kt=end_tas_kt,
        corrected_net_thrust_lb=end_thrust_lb,
    )


def _accelerate_gradient(
    flight: _Flight,
    step: DepartureStep,
    *,
    max_acceleration_ft_s2: float,
    mean_tas_kt: float,
) -> float:
    """The climb gradient ``G`` of an Accelerate step, dimensionless (B6.1.3).

    The energy share factor is preferred over the rate of climb wherever both
    are published, which B6.1.3 puts as advice rather than as a rule ("it is
    preferable to use ESF values"): only it "adapt[s] to changing airport
    elevations and atmosphere conditions", the rate of climb having been
    tabulated for the atmosphere the manufacturer assumed.
    """
    if step.energy_share_percent is not None:
        # Eq. B-25. Substituted into Eq. B-24 it collapses that denominator to
        # 2 a_max (ESF/100), which is why the energy-share branch is well posed
        # wherever the rate-of-climb branch is not.
        return (
            max_acceleration_ft_s2
            / _G_FT_S2
            * (1.0 - step.energy_share_percent / 100.0)
        )
    rate = float(step.rate_of_climb_ft_per_min or 0.0)
    return rate / (60.0 * _KT_FT_S * mean_tas_kt)


def _accelerate_point(
    flight: _Flight, step: DepartureStep, start: ProfilePoint
) -> ProfilePoint:
    """The end point of an Accelerate step (B6.1.3), by iteration.

    The one step of Appendix B that does not close in one pass. The end height
    sets the density, which sets the true airspeed reached, which with the
    mid-step thrust sets the segment length, which sets the end height again:
    "Since they are interdependent, the output height above airfield elevation,
    True Airspeed, CNT and ground Distance at Point2 must be calculated by
    iteration."
    """
    end_cas_kt = float(step.end_calibrated_airspeed_kt or 0.0)
    start_altitude_ft = flight.altitude_ft(start.altitude_ft)
    bank_rad = math.radians(step.bank_angle_deg)
    drag_ratio = _drag_ratio(flight, step)
    end_altitude_ft = start_altitude_ft + _ACCELERATE_SEED_FT
    segment_ft = 0.0
    for _iteration in range(_ACCELERATE_ITERATION_LIMIT):
        end_tas_kt = flight.aerodrome.true_airspeed_kt(end_cas_kt, end_altitude_ft)
        mean_tas_kt = math.sqrt(0.5 * (end_tas_kt**2 + start.true_airspeed_kt**2))
        mid_altitude_ft = 0.5 * (start_altitude_ft + end_altitude_ft)
        mid_thrust_lb = flight.step_thrust_lb(
            step.thrust_rating,
            altitude_ft=mid_altitude_ft,
            calibrated_airspeed_kt=flight.aerodrome.calibrated_airspeed_kt(
                mean_tas_kt, mid_altitude_ft
            ),
            true_airspeed_kt=mean_tas_kt,
            drag_ratio=drag_ratio,
            bank_rad=bank_rad,
        )
        max_acceleration_ft_s2 = _G_FT_S2 * (
            flight.aircraft.engines
            * mid_thrust_lb
            / flight.corrected_weight_lb(mid_altitude_ft)
            - drag_ratio / math.cos(bank_rad)
        )
        gradient = _accelerate_gradient(
            flight,
            step,
            max_acceleration_ft_s2=max_acceleration_ft_s2,
            mean_tas_kt=mean_tas_kt,
        )
        spare = max_acceleration_ft_s2 - gradient * _G_FT_S2
        if spare < _ACCELERATE_GRADIENT_MARGIN * _G_FT_S2:
            gradient = max_acceleration_ft_s2 / _G_FT_S2 - _ACCELERATE_GRADIENT_MARGIN
            spare = _ACCELERATE_GRADIENT_MARGIN * _G_FT_S2
        if gradient < _ACCELERATE_GRADIENT_FLOOR:
            msg = (
                "there is not enough thrust to achieve the acceleration and "
                f"climb specified (B6.1.3); step {step.step_type!r} to "
                f"{end_cas_kt} kt needs a climb gradient of {gradient!r}, below "
                f"the {_ACCELERATE_GRADIENT_FLOOR} floor. The procedure steps "
                "have to be revised."
            )
            raise ValueError(msg)
        # Eq. B-24 and Eq. B-26. The 0.95 here and the division by it there
        # cancel, so the height gained is the one before the wind correction.
        segment_ft = (
            _ACCELERATE_WIND_FACTOR
            * _KT_FT_S**2
            * (end_tas_kt**2 - start.true_airspeed_kt**2)
            / (2.0 * spare)
        )
        updated_ft = start_altitude_ft + segment_ft * gradient / _ACCELERATE_WIND_FACTOR
        converged = abs(updated_ft - end_altitude_ft) <= _ACCELERATE_TOLERANCE_FT
        end_altitude_ft = updated_ft
        if converged:
            break
    else:
        msg = (
            "the Accelerate step's height did not converge within "
            f"{_ACCELERATE_ITERATION_LIMIT} iterations (B6.1.3); step "
            f"{step.step_type!r} to {end_cas_kt} kt last moved to "
            f"{end_altitude_ft!r} ft over a segment of {segment_ft!r} ft."
        )
        raise ValueError(msg)
    end_tas_kt = flight.aerodrome.true_airspeed_kt(end_cas_kt, end_altitude_ft)
    # Eq. B-28: the wind correction reads the mean true airspeed of the finished
    # segment, not the mean the iteration was driven with.
    mean_tas_kt = math.sqrt(0.5 * (start.true_airspeed_kt**2 + end_tas_kt**2))
    segment_wind_ft = flight.wind_scaled_distance(segment_ft, mean_tas_kt)
    end_height_ft = end_altitude_ft - flight.aerodrome.elevation_ft
    return ProfilePoint(
        distance_ft=start.distance_ft + segment_wind_ft,
        altitude_ft=end_height_ft,
        true_airspeed_kt=end_tas_kt,
        corrected_net_thrust_lb=flight.step_thrust_lb(
            step.thrust_rating,
            altitude_ft=end_altitude_ft,
            calibrated_airspeed_kt=end_cas_kt,
            true_airspeed_kt=end_tas_kt,
            drag_ratio=drag_ratio,
            bank_rad=bank_rad,
        ),
    )


def _level_point(
    flight: _Flight, step: DepartureStep, start: ProfilePoint
) -> ProfilePoint:
    """The end point of a Level step (B6.1.4): thrust is the only unknown."""
    altitude_ft = flight.altitude_ft(start.altitude_ft)
    return ProfilePoint(
        distance_ft=start.distance_ft + float(step.distance_ft or 0.0),
        altitude_ft=start.altitude_ft,
        true_airspeed_kt=start.true_airspeed_kt,
        corrected_net_thrust_lb=flight.level_thrust_lb(
            altitude_ft=altitude_ft,
            drag_ratio=_drag_ratio(flight, step),
            bank_rad=math.radians(step.bank_angle_deg),
        ),
    )


def _level_accelerate_point(
    flight: _Flight, step: DepartureStep, start: ProfilePoint
) -> ProfilePoint:
    """The end point of a Level-Accelerate step (B6.1.5).

    An Accelerate step with the whole excess thrust spent on speed. Because the
    height is fixed the density is too, so the end true airspeed is known before
    anything else and "iteration is not required".
    """
    end_cas_kt = float(step.end_calibrated_airspeed_kt or 0.0)
    altitude_ft = flight.altitude_ft(start.altitude_ft)
    bank_rad = math.radians(step.bank_angle_deg)
    drag_ratio = _drag_ratio(flight, step)
    end_tas_kt = flight.aerodrome.true_airspeed_kt(end_cas_kt, altitude_ft)
    mean_tas_kt = math.sqrt(0.5 * (start.true_airspeed_kt**2 + end_tas_kt**2))
    mid_thrust_lb = flight.step_thrust_lb(
        step.thrust_rating,
        altitude_ft=altitude_ft,
        calibrated_airspeed_kt=flight.aerodrome.calibrated_airspeed_kt(
            mean_tas_kt, altitude_ft
        ),
        true_airspeed_kt=mean_tas_kt,
        drag_ratio=drag_ratio,
        bank_rad=bank_rad,
    )
    # Eq. B-31.
    max_acceleration_ft_s2 = _G_FT_S2 * (
        flight.aircraft.engines
        * mid_thrust_lb
        / flight.corrected_weight_lb(altitude_ft)
        - drag_ratio / math.cos(bank_rad)
    )
    if max_acceleration_ft_s2 <= 0.0:
        msg = (
            "a Level-Accelerate step must have thrust to spare over its drag "
            f"(Eq. B-31); step {step.step_type!r} to {end_cas_kt} kt gives a "
            f"mean acceleration of {max_acceleration_ft_s2!r} ft/s^2."
        )
        raise ValueError(msg)
    # Eq. B-32 and Eq. B-33.
    segment_ft = (
        _ACCELERATE_WIND_FACTOR
        * _KT_FT_S**2
        * (end_tas_kt**2 - start.true_airspeed_kt**2)
        / (2.0 * max_acceleration_ft_s2)
    )
    segment_wind_ft = flight.wind_scaled_distance(segment_ft, mean_tas_kt)
    return ProfilePoint(
        distance_ft=start.distance_ft + segment_wind_ft,
        altitude_ft=start.altitude_ft,
        true_airspeed_kt=end_tas_kt,
        corrected_net_thrust_lb=flight.step_thrust_lb(
            step.thrust_rating,
            altitude_ft=altitude_ft,
            calibrated_airspeed_kt=end_cas_kt,
            true_airspeed_kt=end_tas_kt,
            drag_ratio=drag_ratio,
            bank_rad=bank_rad,
        ),
    )


def _transition_point(
    flight: _Flight, step: DepartureStep, start: ProfilePoint, end: ProfilePoint
) -> ProfilePoint:
    """The intermediate point a change of thrust rating inserts (B6.1.6).

    Not a step: a 1000 ft segment spliced into the *start* of the step that
    follows the change, so the thrust discontinuity has somewhere to happen.
    Point1's thrust still belongs to the previous rating, so this point is the
    first one at the new one, and it is placed "based on the same flight path
    gradient of that of the overall segment" -- which is why the whole step has
    to be solved before the point inserted at its start can be.
    """
    length_ft = end.distance_ft - start.distance_ft
    offset_ft = _transition_offset(length_ft)
    fraction = offset_ft / length_ft if length_ft > 0.0 else 0.0
    height_ft = start.altitude_ft + fraction * (end.altitude_ft - start.altitude_ft)
    altitude_ft = flight.altitude_ft(height_ft)
    if step.kind == "climb":
        # Eq. B-36: a climb holds its calibrated airspeed, so the true airspeed
        # at the inserted point follows the density and not the distance.
        tas_kt = start.true_airspeed_kt * math.sqrt(
            flight.aerodrome.density_ratio(flight.altitude_ft(start.altitude_ft))
            / flight.aerodrome.density_ratio(altitude_ft)
        )
    else:
        # An acceleration is linear in kinetic energy along the segment, so the
        # interpolation is on the square of the speed. A Level step, whose
        # endpoints share a speed, gives that speed back either way.
        tas_kt = math.sqrt(
            start.true_airspeed_kt**2
            + fraction * (end.true_airspeed_kt**2 - start.true_airspeed_kt**2)
        )
    if _fold(step.thrust_rating) == _ADAPTED_THRUST:
        # A Level step has no rating to look the inserted point's thrust up
        # from: its thrust is adapted. B6.1.6 says only to use "the engine
        # coefficients corresponding to the specific Thrust Rating on the
        # current step"; the reference cases settle it, giving
        # the inserted point the same Eq. B-30 thrust as the step it starts.
        thrust_lb = end.corrected_net_thrust_lb
    else:
        thrust_lb = flight.step_thrust_lb(
            step.thrust_rating,
            altitude_ft=altitude_ft,
            calibrated_airspeed_kt=flight.aerodrome.calibrated_airspeed_kt(
                tas_kt, altitude_ft
            ),
            true_airspeed_kt=tas_kt,
            drag_ratio=_drag_ratio(flight, step),
            bank_rad=math.radians(step.bank_angle_deg),
        )
    return ProfilePoint(
        distance_ft=start.distance_ft + offset_ft,
        altitude_ft=height_ft,
        true_airspeed_kt=tas_kt,
        corrected_net_thrust_lb=thrust_lb,
    )


def departure_profile(
    aircraft: PerformanceAircraft,
    steps: Sequence[DepartureStep],
    *,
    weight_lb: float,
    aerodrome: Aerodrome,
    procedure_id: str = "",
) -> FlightProfile:
    """Fly a departure procedure's steps into a flight profile (Doc 29 B6).

    The profile is built forward from brake release, "the starting parameters
    for each segment being equal to those at the end of the preceding segment"
    (B1). Every step contributes one point, except the Take-off step, which
    contributes two, and any step that changes thrust rating, which is preceded
    by an inserted transition point (B6.1.6).

    :param aircraft: The aeroplane's coefficient set.
    :param steps: The procedure's steps, in order, starting with a Take-off.
    :param weight_lb: Take-off weight, lb -- the ANP ``Default_weights`` entry
        for the stage length being flown.
    :param aerodrome: Aerodrome, weather and runway gradient.
    :param procedure_id: Identifier of the procedure, carried into the result.
    :return: A :class:`FlightProfile` with ``operation="D"``.
    :raises ValueError: if the procedure does not start with a Take-off step,
        or a step cannot be flown as specified.
    """
    if not steps or steps[0].kind != "takeoff":
        first = steps[0].step_type if steps else None
        msg = (
            "a departure procedure must start with a Take-off step, which is "
            "what anchors the profile at brake release (B6.1.1); the first "
            f"step is {first!r}."
        )
        raise ValueError(msg)
    if weight_lb <= 0.0 or not math.isfinite(weight_lb):
        msg = f"'weight_lb' must be positive; got {weight_lb!r}."
        raise ValueError(msg)
    flight = _Flight(
        aircraft=aircraft,
        aerodrome=aerodrome,
        weight_lb=weight_lb,
        operation=_DEPARTURE,
    )
    solvers = {
        "climb": _climb_point,
        "accelerate": _accelerate_point,
        "level": _level_point,
        "level-accelerate": _level_accelerate_point,
    }
    brake_release, rotation = _takeoff_points(flight, steps[0])
    points = [brake_release, rotation]
    current = rotation
    rating = steps[0].thrust_rating
    for step in steps[1:]:
        # B6.1.3: "any subsequent Climb step with an input altitude lower than
        # the end point altitude of the current Accelerate step shall be
        # skipped", because an Accelerate step chooses its own end height and
        # may already have flown through the one a later Climb step aims at.
        if step.kind == "climb" and float(step.end_altitude_ft or 0.0) <= (
            current.altitude_ft
        ):
            continue
        end = solvers[step.kind](flight, step, current)
        if _fold(step.thrust_rating) != _fold(rating):
            points.append(_transition_point(flight, step, current, end))
        points.append(end)
        current = end
        rating = step.thrust_rating
    return FlightProfile(
        aircraft_id=aircraft.aircraft_id,
        operation=_DEPARTURE,
        procedure_id=procedure_id,
        points=tuple(points),
    )


# --------------------------------------------------------------------------
# Approach (B7)
# --------------------------------------------------------------------------
def _needs_transition(step: ApproachStep, following: ApproachStep) -> bool:
    """Whether a transition point belongs at the end of *step* (B7.1.7).

    Figure B-3 in one line: a change of step type, flap angle or descent angle
    calls for one, "to reflect a rapid change in the thrust between the two
    consecutive steps", unless both steps are flown at idle, where there is no
    thrust change to reflect however much else changes. This subsumes the Case 1
    and Case 2 splits each step type states separately, and is the authority
    where they seem to disagree.
    """
    if step.kind in _IDLE_STEP_TYPES and following.kind in _IDLE_STEP_TYPES:
        return False
    return (
        step.kind != following.kind
        or _fold(step.flap_id) != _fold(following.flap_id)
        or step.descent_angle_deg != following.descent_angle_deg
    )


def _descent_deceleration_ft_s2(
    *,
    start_tas_kt: float,
    end_tas_kt: float,
    headwind_kt: float,
    gamma_rad: float,
    height_drop_ft: float,
) -> float:
    """Mean deceleration along a decelerating descent, ft/s^2 (inside Eq. B-41).

    Negative when decelerating, which is what makes Eq. B-41 return less thrust
    than level flight at the same speed. It is the kinetic energy the segment
    sheds over the length actually flown, so the denominator is twice the slant
    length -- which the 5th edition reaches through the height lost rather than
    through the ground distance, and which is the same quantity either way.

    The ground speeds are the plain ``V - w``. Doc 29 prints them divided by
    ``cos(gamma)``, as the 4th edition did through its ground-distance
    denominator, and the reference cases refuse it: on the reference approach
    flown entirely at that step type, the plain speeds land within 0.05 lb of
    the tabulated thrust at all twelve points, and the divided ones from 0.3 to
    6.4 lb out at each of the nine the deceleration reaches, always low. See
    docs/ERRATA.md under Eq. (B-41). Dividing both the speeds
    *and* the path length by ``cos(gamma)`` counts the slope twice; the drag
    term beside it keeps the ``cos(gamma)`` Doc 29 prints there, which the same
    five points confirm to the same 0.04 lb.
    """
    slant_ft = height_drop_ft / math.sin(gamma_rad)
    return (
        _KT_FT_S**2
        * ((end_tas_kt - headwind_kt) ** 2 - (start_tas_kt - headwind_kt) ** 2)
        / (2.0 * slant_ft)
    )


def _level_deceleration_ft_s2(
    *, start_tas_kt: float, end_tas_kt: float, headwind_kt: float, length_ft: float
) -> float:
    """Mean deceleration along a decelerating level segment, ft/s^2 (Eq. B-63)."""
    return (
        _KT_FT_S**2
        * ((end_tas_kt - headwind_kt) ** 2 - (start_tas_kt - headwind_kt) ** 2)
        / (2.0 * length_ft)
    )


def _headwind_thrust_correction_lb(
    flight: _Flight,
    *,
    altitude_ft: float,
    gamma_rad: float,
    calibrated_airspeed_kt: float,
) -> float:
    """Eq. B-48 and Eq. B-77: the thrust a non-standard headwind costs, lb.

    Doc 29's descent thrust is written for its 8 kt reference wind, and this
    puts back the difference. The sign follows the 5th edition's positive
    descent angle: with less headwind than the reference the correction is
    negative, and the reference cases confirm it -- a zero-wind approach lands
    at 4724.1 lb where the 8 kt reference gives 4957.3 lb. The 4th edition,
    which took the descent angle negative, added the same expression and so
    corrected the other way.
    """
    return (
        _DESCENT_WIND_CONSTANT
        * flight.corrected_weight_lb(altitude_ft)
        * math.sin(gamma_rad)
        * (flight.aerodrome.headwind_kt - DEFAULT_HEADWIND_KT)
        / (flight.aircraft.engines * calibrated_airspeed_kt)
    )


def _land_points(
    flight: _Flight,
    step: ApproachStep,
    *,
    approach_gamma_rad: float,
    following: ApproachStep,
) -> tuple[ProfilePoint, ProfilePoint]:
    """Touchdown and the end of the touchdown roll (B7.1.5).

    Point1 is the anchor of the whole approach: distance zero, height zero, and
    every airborne step above it is placed by counting backwards from here.
    Point2 then reaches *forward* into the Decelerate step that follows, which
    is where the two sweeps of an approach meet.
    """
    aero = flight.aircraft.flap(flight.operation, step.flap_id)
    if aero.speed_coefficient is None:
        msg = (
            f"aircraft {flight.aircraft.aircraft_id!r}: flap {step.flap_id!r} "
            "carries no landing speed coefficient C, so Eq. B-75 cannot be "
            "evaluated for a Land step flown in it."
        )
        raise ValueError(msg)
    altitude_ft = flight.altitude_ft(0.0)
    # Eq. B-75 returns a true airspeed directly, unlike its take-off counterpart.
    touchdown_tas_kt = (
        aero.speed_coefficient
        * math.sqrt(flight.weight_lb)
        / math.sqrt(flight.aerodrome.density_ratio(altitude_ft))
    )
    # Eq. B-76. Alone of the descent force balances it carries no cos(eps),
    # which is harmless on a runway, where the bank angle is zero.
    thrust_lb = (
        flight.corrected_weight_lb(altitude_ft)
        / flight.aircraft.engines
        * (aero.drag_ratio - math.sin(approach_gamma_rad) / _DESCENT_WIND_CONSTANT)
    ) + _headwind_thrust_correction_lb(
        flight,
        altitude_ft=altitude_ft,
        gamma_rad=approach_gamma_rad,
        calibrated_airspeed_kt=flight.aerodrome.calibrated_airspeed_kt(
            touchdown_tas_kt, altitude_ft
        ),
    )
    touchdown = ProfilePoint(
        distance_ft=0.0,
        altitude_ft=0.0,
        true_airspeed_kt=touchdown_tas_kt,
        corrected_net_thrust_lb=thrust_lb,
    )
    return touchdown, ProfilePoint(
        distance_ft=float(step.touchdown_roll_ft or 0.0),
        altitude_ft=0.0,
        # Eq. B-78 and Eq. B-79: both end values are the *following* Decelerate
        # step's start values, the thrust as a percentage of the maximum
        # sea-level static thrust rather than as a coefficient lookup.
        true_airspeed_kt=flight.aerodrome.true_airspeed_kt(
            float(following.start_calibrated_airspeed_kt or 0.0), altitude_ft
        ),
        corrected_net_thrust_lb=float(following.start_thrust_percent or 0.0)
        * flight.aircraft.max_static_thrust_lb
        / 100.0,
    )


def _idle_thrust_lb(
    flight: _Flight, *, altitude_ft: float, calibrated_airspeed_kt: float
) -> float:
    """Idle thrust at a point of an idle descent, lb (B7.1.2, B7.1.4).

    A lookup rather than a solve: the ``IdleApproach`` rating of Eq. B-9. "The
    presence of a Descend-Idle step in the approach procedural step profile of a
    given aircraft type requires the availability, for that aircraft, of jet
    coefficients [...] for the 'IdleApproach' thrust rating."
    """
    return flight.rated_thrust_lb(
        _IDLE_APPROACH,
        altitude_ft=altitude_ft,
        calibrated_airspeed_kt=calibrated_airspeed_kt,
        true_airspeed_kt=flight.aerodrome.true_airspeed_kt(
            calibrated_airspeed_kt, altitude_ft
        ),
    )


def _isa_deceleration_ft_s2(
    *,
    start_cas_kt: float,
    end_cas_kt: float,
    start_height_ft: float,
    end_height_ft: float,
    headwind_kt: float,
    gamma_rad: float | None,
    length_ft: float | None,
) -> float:
    """The deceleration the tabulated idle-step parameters were derived at, ft/s^2.

    Eq. B-57 for a Descend-Idle step and Eq. B-73 for a Level-Idle one. The
    tabulated start altitude and start calibrated airspeed "are provided for ISA
    reference conditions at a sea-level airport", and what is held fixed away
    from those conditions is *this deceleration*, not the tabulated pair
    (folio B-36).

    Speeds are in ft/s throughout, and so is the headwind: Eq. B-58 and Eq. B-59
    declare it in knots while combining it with speeds Eq. B-57 puts in ft/s,
    and only ft/s closes the algebra in all three.
    """
    start_ft_s = _KT_FT_S * _ISA_SEA_LEVEL.true_airspeed_kt(
        start_cas_kt, start_height_ft
    )
    end_ft_s = _KT_FT_S * _ISA_SEA_LEVEL.true_airspeed_kt(end_cas_kt, end_height_ft)
    wind_ft_s = _KT_FT_S * headwind_kt
    if gamma_rad is None:
        distance_ft = float(length_ft or 0.0)
        return ((end_ft_s - wind_ft_s) ** 2 - (start_ft_s - wind_ft_s) ** 2) / (
            2.0 * distance_ft
        )
    slant_ft = (start_height_ft - end_height_ft) / math.sin(gamma_rad)
    return (
        _along_slope_ft_s(end_ft_s, wind_ft_s, gamma_rad) ** 2
        - _along_slope_ft_s(start_ft_s, wind_ft_s, gamma_rad) ** 2
    ) / (2.0 * slant_ft)


def _along_slope_ft_s(speed_ft_s: float, wind_ft_s: float, gamma_rad: float) -> float:
    """The along-slope ground speed of Eq. B-57 to B-59, ft/s.

    ``sqrt(V^2 - w^2 sin^2 gamma) - w cos gamma``: the wind is horizontal and
    the flight path is not, so the component that opposes the aeroplane along
    its own descending path is what is subtracted.
    """
    return math.sqrt(
        max(speed_ft_s**2 - (wind_ft_s * math.sin(gamma_rad)) ** 2, 0.0)
    ) - wind_ft_s * math.cos(gamma_rad)


@dataclass(frozen=True)
class _Anchor:
    """The next step's Point1, which the step above it is solved backwards from.

    ``Pt1(NextSeg)`` in Doc 29's own notation: the recursion of Eq. B-42 and
    Eq. B-64 reaches down to the Land step's Point1 at distance zero, and every
    airborne step's own Point1 is placed relative to the one below it.
    """

    point: ProfilePoint
    step: ApproachStep


def _approach_point1(
    flight: _Flight, step: ApproachStep, anchor: _Anchor
) -> ProfilePoint:
    """The Point1 of one airborne approach step, solved from the step below it."""
    kind = step.kind
    bank_rad = math.radians(step.bank_angle_deg)
    # An idle step's thrust is a coefficient lookup, not a force balance, so it
    # reads no drag ratio and the ANP tables leave its flap identifier empty
    # accordingly. Asking for one anyway would refuse a published procedure over
    # a configuration nothing in the step needs.
    drag_ratio = 0.0 if kind in _IDLE_STEP_TYPES else _drag_ratio(flight, step)
    engines = flight.aircraft.engines
    height_ft = float(step.start_altitude_ft or 0.0)
    cas_kt = float(step.start_calibrated_airspeed_kt or 0.0)
    length_ft = float(step.distance_ft or 0.0)
    if kind in ("descend-idle", "level-idle"):
        height_ft, cas_kt, length_ft = _non_isa_adjusted(
            flight,
            step,
            anchor,
            height_ft=height_ft,
            cas_kt=cas_kt,
            length_ft=length_ft,
        )
    altitude_ft = flight.altitude_ft(height_ft)
    if step.start_calibrated_airspeed_kt is None:
        # A Level step the table gives no speed: it holds the one the step
        # below it enters at, and the altitude does not change, so the
        # calibrated airspeed follows from that true airspeed exactly.
        tas_kt = anchor.point.true_airspeed_kt
        cas_kt = flight.aerodrome.calibrated_airspeed_kt(tas_kt, altitude_ft)
    else:
        tas_kt = flight.aerodrome.true_airspeed_kt(cas_kt, altitude_ft)
    corrected_weight_lb = flight.corrected_weight_lb(altitude_ft)
    if kind in _LEVEL_STEP_TYPES:
        distance_ft = anchor.point.distance_ft - length_ft
    else:
        gamma_rad = math.radians(float(step.descent_angle_deg or 0.0))
        drop_ft = height_ft - anchor.point.altitude_ft
        if drop_ft <= 0.0:
            msg = (
                f"a {step.step_type!r} step must start above the step below it "
                f"(Eq. B-42); it starts at {height_ft!r} ft with the next "
                f"point at {anchor.point.altitude_ft!r} ft."
            )
            raise ValueError(msg)
        distance_ft = anchor.point.distance_ft - drop_ft / math.tan(gamma_rad)
    if kind == "level":
        # Eq. B-62, which is Eq. B-30 at the approach weight: departures and
        # arrivals share one level-flight thrust law.
        thrust_lb = corrected_weight_lb / engines * (drag_ratio / math.cos(bank_rad))
    elif kind == "level-decel":
        # Eq. B-63.
        thrust_lb = (
            corrected_weight_lb
            / engines
            * (
                drag_ratio / math.cos(bank_rad)
                + _level_deceleration_ft_s2(
                    start_tas_kt=tas_kt,
                    end_tas_kt=anchor.point.true_airspeed_kt,
                    headwind_kt=flight.aerodrome.headwind_kt,
                    length_ft=length_ft,
                )
                / _G_FT_S2
            )
        )
    elif kind in ("descend-idle", "level-idle"):
        thrust_lb = _idle_thrust_lb(
            flight, altitude_ft=altitude_ft, calibrated_airspeed_kt=cas_kt
        )
    elif kind == "descend":
        # Eq. B-40. The 1.03 plays the part K plays for a climb: it "accounts
        # for the effects thrust of descending into an 8-knot headwind and the
        # deceleration inherent in descending at constant Calibrated Airspeed",
        # so a Descend step needs no iteration either. The deceleration between
        # consecutive Descend steps is deliberately left out of the thrust,
        # which the standard notes makes the result conservative.
        gamma_rad = math.radians(float(step.descent_angle_deg or 0.0))
        thrust_lb = corrected_weight_lb / engines * (
            drag_ratio / math.cos(bank_rad)
            - math.sin(gamma_rad) / _DESCENT_WIND_CONSTANT
        ) + _headwind_thrust_correction_lb(
            flight,
            altitude_ft=altitude_ft,
            gamma_rad=gamma_rad,
            calibrated_airspeed_kt=cas_kt,
        )
    else:
        # Eq. B-41, the Descend-Decel step: the deceleration is carried, and the
        # drag term is the one Doc 29 prints with cos(gamma) on this page and
        # without it on the page before.
        gamma_rad = math.radians(float(step.descent_angle_deg or 0.0))
        thrust_lb = (
            corrected_weight_lb
            / engines
            * (
                drag_ratio * math.cos(gamma_rad) / math.cos(bank_rad)
                - math.sin(gamma_rad)
                + _descent_deceleration_ft_s2(
                    start_tas_kt=tas_kt,
                    end_tas_kt=anchor.point.true_airspeed_kt,
                    headwind_kt=flight.aerodrome.headwind_kt,
                    gamma_rad=gamma_rad,
                    height_drop_ft=height_ft - anchor.point.altitude_ft,
                )
                / _G_FT_S2
            )
        )
    return ProfilePoint(
        distance_ft=distance_ft,
        altitude_ft=height_ft,
        true_airspeed_kt=tas_kt,
        corrected_net_thrust_lb=thrust_lb,
    )


def _non_isa_adjusted(
    flight: _Flight,
    step: ApproachStep,
    anchor: _Anchor,
    *,
    height_ft: float,
    cas_kt: float,
    length_ft: float,
) -> tuple[float, float, float]:
    """Hold an idle step's ISA deceleration away from sea-level ISA (B7.1.2, B7.1.4).

    The manufacturer tabulated an idle step's start altitude, start speed and
    length together, under sea-level ISA; away from those conditions the three
    stop being consistent, and Doc 29 keeps the *deceleration* rather than the
    tabulated numbers. What gives depends on what follows: a step that ends
    level or on the runway keeps its height and moves its speed (Eq. B-58), and
    one that ends still descending keeps its speed and moves its height
    (Eq. B-59, Eq. B-60). A Level-Idle step moves its length instead (Eq. B-74),
    since "the input step length and start and end speeds are not physically
    linked together" for that step type at all.

    Under sea-level ISA every branch returns its input unchanged, which is what
    makes the adjustment invisible in Doc 29's own reference cases.

    :return: The step's height (ft), start calibrated airspeed (kt) and length
        (ft), adjusted.
    """
    wind_kt = flight.aerodrome.headwind_kt
    next_cas_kt = float(anchor.step.start_calibrated_airspeed_kt or 0.0)
    next_height_ft = anchor.point.altitude_ft
    if step.kind == "level-idle":
        _warn_level_idle_inputs(step, anchor)
        reference = _isa_deceleration_ft_s2(
            start_cas_kt=cas_kt,
            end_cas_kt=next_cas_kt,
            start_height_ft=height_ft,
            end_height_ft=height_ft,
            headwind_kt=wind_kt,
            gamma_rad=None,
            length_ft=length_ft,
        )
        if reference == 0.0:
            # Equal start and end speeds: there is no deceleration to hold, and
            # Eq. B-74 would divide by zero to say so. The tabulated length is
            # the only length there is.
            return height_ft, cas_kt, length_ft
        altitude_ft = flight.altitude_ft(height_ft)
        start_ft_s = _KT_FT_S * flight.aerodrome.true_airspeed_kt(cas_kt, altitude_ft)
        end_ft_s = _KT_FT_S * flight.aerodrome.true_airspeed_kt(
            next_cas_kt, altitude_ft
        )
        wind_ft_s = _KT_FT_S * wind_kt
        adjusted_ft = ((end_ft_s - wind_ft_s) ** 2 - (start_ft_s - wind_ft_s) ** 2) / (
            2.0 * reference
        )
        return height_ft, cas_kt, adjusted_ft
    gamma_rad = math.radians(float(step.descent_angle_deg or 0.0))
    reference = _isa_deceleration_ft_s2(
        start_cas_kt=cas_kt,
        end_cas_kt=next_cas_kt,
        start_height_ft=height_ft,
        end_height_ft=next_height_ft,
        headwind_kt=wind_kt,
        gamma_rad=gamma_rad,
        length_ft=None,
    )
    wind_ft_s = _KT_FT_S * wind_kt
    next_ft_s = _KT_FT_S * anchor.point.true_airspeed_kt
    next_along = _along_slope_ft_s(next_ft_s, wind_ft_s, gamma_rad)
    slant_ft = (height_ft - next_height_ft) / math.sin(gamma_rad)
    if anchor.step.kind in ("descend", "descend-idle", "descend-decel"):
        if reference == 0.0:
            return height_ft, cas_kt, length_ft
        # Eq. B-59 and Eq. B-60: hold the speed, move the step's own top.
        start_ft_s = _KT_FT_S * flight.aerodrome.true_airspeed_kt(
            cas_kt, flight.altitude_ft(height_ft)
        )
        start_along = _along_slope_ft_s(start_ft_s, wind_ft_s, gamma_rad)
        adjusted_slant_ft = (next_along**2 - start_along**2) / (2.0 * reference)
        return (
            next_height_ft + adjusted_slant_ft * math.sin(gamma_rad),
            cas_kt,
            length_ft,
        )
    # Eq. B-58: hold the height, move the speed the step is entered at.
    squared = next_along**2 - 2.0 * slant_ft * reference
    along = math.sqrt(max(squared, 0.0)) + wind_ft_s * math.cos(gamma_rad)
    adjusted_ft_s = math.sqrt(along**2 + (wind_ft_s * math.sin(gamma_rad)) ** 2)
    adjusted_tas_kt = adjusted_ft_s / _KT_FT_S
    return (
        height_ft,
        flight.aerodrome.calibrated_airspeed_kt(
            adjusted_tas_kt, flight.altitude_ft(height_ft)
        ),
        length_ft,
    )


def _warn_level_idle_inputs(step: ApproachStep, anchor: _Anchor) -> None:
    """Raise the two consistency warnings B7.1.4 asks for, without fixing them.

    Doc 29 asks that "the system should warn the user" when consecutive
    Level-Idle steps disagree about the altitude they are flown at, and when the
    calibrated airspeed rises rather than falls along a decelerating segment.
    Neither is repaired here: the tabulated numbers are the manufacturer's, and
    a silently corrected input is a profile nobody can trace back to its table.
    """
    if anchor.step.kind != "level-idle":
        return
    if step.start_altitude_ft != anchor.step.start_altitude_ft:
        msg = (
            "consecutive Level-Idle steps should be flown at one altitude "
            f"(B7.1.4); {step.start_altitude_ft!r} ft is followed by "
            f"{anchor.step.start_altitude_ft!r} ft."
        )
        warnings.warn(msg, stacklevel=2)
    start = step.start_calibrated_airspeed_kt
    following = anchor.step.start_calibrated_airspeed_kt
    if start is not None and following is not None and start < following:
        msg = (
            "a Level-Idle step's start CAS should be at or above the following "
            f"step's (B7.1.4); {start!r} kt is followed by {following!r} kt, "
            "which is an acceleration on an idle segment."
        )
        warnings.warn(msg, stacklevel=2)


def _approach_point2(
    flight: _Flight, step: ApproachStep, point1: ProfilePoint, anchor: _Anchor
) -> ProfilePoint:
    """The transition point at the end of an airborne approach step (B7.1.7).

    Placed at most 1000 ft before the following step's Point1, on the step it
    ends, so the thrust step between two consecutive steps has a segment of its
    own to happen over rather than a discontinuity at a single point.
    """
    length_ft = anchor.point.distance_ft - point1.distance_ft
    offset_ft = _transition_offset(length_ft)
    if step.kind in _LEVEL_STEP_TYPES:
        height_ft = point1.altitude_ft
    else:
        gamma_rad = math.radians(float(step.descent_angle_deg or 0.0))
        height_ft = anchor.point.altitude_ft + offset_ft * math.tan(gamma_rad)
    altitude_ft = flight.altitude_ft(height_ft)
    fraction = offset_ft / length_ft if length_ft > 0.0 else 0.0
    # Eq. B-46, B-56, B-66 and B-72 are one interpolation: linear in the square
    # of the speed, which is linear in the kinetic energy the segment sheds.
    tas_kt = math.sqrt(
        anchor.point.true_airspeed_kt**2
        + fraction * (point1.true_airspeed_kt**2 - anchor.point.true_airspeed_kt**2)
    )
    if step.kind in _IDLE_STEP_TYPES:
        # An idle step's thrust is a lookup at every point, so this one is taken
        # at the transition point's own altitude and speed rather than scaled
        # from Point1's. B7.1.7 inserts the point without saying what thrust it
        # carries, and B7.1.2 and B7.1.4 pin V_C to the step's start CAS only
        # where they describe Point1_CNT.
        thrust_lb = _idle_thrust_lb(
            flight,
            altitude_ft=altitude_ft,
            calibrated_airspeed_kt=flight.aerodrome.calibrated_airspeed_kt(
                tas_kt, altitude_ft
            ),
        )
    elif step.kind in _LEVEL_STEP_TYPES:
        # The altitude does not change over a level step, so neither does the
        # corrected weight the thrust balances.
        thrust_lb = point1.corrected_net_thrust_lb
    else:
        # Eq. B-47: the same force balance one pressure ratio further down.
        # Doc 29 is explicit that it is derived from Point1's thrust *after*
        # that thrust has been corrected for a non-standard headwind.
        thrust_lb = (
            flight.aerodrome.pressure_ratio(flight.altitude_ft(point1.altitude_ft))
            / flight.aerodrome.pressure_ratio(altitude_ft)
            * point1.corrected_net_thrust_lb
        )
    return ProfilePoint(
        distance_ft=anchor.point.distance_ft - offset_ft,
        altitude_ft=height_ft,
        true_airspeed_kt=tas_kt,
        corrected_net_thrust_lb=thrust_lb,
    )


def _approach_descent_angle_rad(
    steps: Sequence[ApproachStep], land_index: int
) -> float:
    """The descent angle the Land step's thrust is computed at (Eq. B-76).

    Doc 29 names "the last Descend step preceding the current Land step", which
    is the last airborne step that carries a descent angle at all -- the final
    approach slope, whatever flavour of descending step flies it.
    """
    for step in reversed(steps[:land_index]):
        if step.descent_angle_deg:
            return math.radians(step.descent_angle_deg)
    msg = (
        "the Land step's thrust is computed at the descent angle of the last "
        "Descend step before it (Eq. B-76), and this procedure has no "
        "descending step before its Land step."
    )
    raise ValueError(msg)


def approach_profile(
    aircraft: PerformanceAircraft,
    steps: Sequence[ApproachStep],
    *,
    aerodrome: Aerodrome,
    weight_lb: float | None = None,
    procedure_id: str = "",
) -> FlightProfile:
    """Fly an approach procedure's steps into a flight profile (Doc 29 B7).

    Approaches are solved **backwards**. Every airborne step computes its own
    Point1 from the following step's Point1 (Eq. B-42, Eq. B-64), and the
    recursion is anchored by the Land step, whose Point1 sits at distance zero:
    hence the negative distances before touchdown. The rollout is then solved
    forwards from the same anchor, so touchdown is where the two sweeps meet.

    :param aircraft: The aeroplane's coefficient set.
    :param steps: The procedure's steps, in order, containing one Land step
        followed by its Decelerate steps.
    :param aerodrome: Aerodrome and weather.
    :param weight_lb: Approach weight, lb. ``None`` (default) takes Doc 29's
        own rule, 90 % of the aeroplane's maximum landing weight (folio B-31).
    :param procedure_id: Identifier of the procedure, carried into the result.
    :return: A :class:`FlightProfile` with ``operation="A"``.
    :raises ValueError: if the procedure carries no Land step to anchor it, or
        a step cannot be flown as specified.
    """
    kinds = [step.kind for step in steps]
    if kinds.count("land") != 1:
        msg = (
            "an approach procedure must carry exactly one Land step, which is "
            "the distance-zero anchor every airborne step is placed relative "
            f"to (B7.1.5); procedure {procedure_id!r} carries "
            f"{kinds.count('land')}."
        )
        raise ValueError(msg)
    land_index = kinds.index("land")
    rollout = [step for step in steps[land_index + 1 :] if step.kind == "decelerate"]
    if not rollout:
        msg = (
            "a Land step must be followed by at least one Decelerate step, "
            "which is where its Point2 speed and thrust are read from "
            f"(Eq. B-78, Eq. B-79); procedure {procedure_id!r} has none."
        )
        raise ValueError(msg)
    flight = _Flight(
        aircraft=aircraft,
        aerodrome=aerodrome,
        weight_lb=(
            aircraft.approach_weight_lb if weight_lb is None else float(weight_lb)
        ),
        operation=_ARRIVAL,
    )
    touchdown, roll_end = _land_points(
        flight,
        steps[land_index],
        approach_gamma_rad=_approach_descent_angle_rad(steps, land_index),
        following=rollout[0],
    )
    # Backwards over the airborne steps, each one anchored on the one below it.
    anchor = _Anchor(point=touchdown, step=steps[land_index])
    airborne: list[list[ProfilePoint]] = []
    for index in range(land_index - 1, -1, -1):
        step = steps[index]
        point1 = _approach_point1(flight, step, anchor)
        emitted = [point1]
        if _needs_transition(step, anchor.step):
            emitted.append(_approach_point2(flight, step, point1, anchor))
        airborne.append(emitted)
        anchor = _Anchor(point=point1, step=step)
    points = [point for emitted in reversed(airborne) for point in emitted]
    points.extend((touchdown, roll_end))
    # Forwards along the runway. The terminal zero-length Decelerate step emits
    # no point of its own, but it is where the one before it reads its end
    # speed and thrust from, so it is not a filler row.
    current = roll_end
    for step, following in zip(rollout, rollout[1:], strict=False):
        length_ft = float(step.distance_ft or 0.0)
        if length_ft == 0.0:
            continue
        current = ProfilePoint(
            distance_ft=current.distance_ft + length_ft,
            altitude_ft=0.0,
            true_airspeed_kt=flight.aerodrome.true_airspeed_kt(
                float(following.start_calibrated_airspeed_kt or 0.0),
                flight.altitude_ft(0.0),
            ),
            corrected_net_thrust_lb=float(following.start_thrust_percent or 0.0)
            * aircraft.max_static_thrust_lb
            / 100.0,
        )
        points.append(current)
    return FlightProfile(
        aircraft_id=aircraft.aircraft_id,
        operation=_ARRIVAL,
        procedure_id=procedure_id,
        points=tuple(points),
    )
