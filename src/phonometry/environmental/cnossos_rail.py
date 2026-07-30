#  Copyright (c) 2026. Jose M. Requena-Plens
"""
CNOSSOS-EU railway source emission (Directive 2002/49/EC Annex II, 2.3).

The common noise assessment methods of the European Union describe a railway
track as **two** incoherent source lines at the centre of the track, at
``h = 0,5 m`` (source A) and ``h = 4,0 m`` (source B) above the plane tangent to
the two upper rail surfaces. Every physical source of a vehicle is allocated to
one of the two heights and contributes a directional sound power per metre of
line

``L'_W,eq,line,i(psi,phi) = L_W,0,dir,i(psi,phi) + 10 lg( Q / (1000 v) )`` (2.3.2)

for a running train, or ``+ 10 lg( T_idle / (T_ref L) )`` (2.3.4) for an idling
one. This module implements the whole of 2.3 together with the coefficient
database of Appendix G, in the twenty-four 1/3-octave bands from 50 Hz to
10 kHz, and energy-sums them into the eight octave bands the propagation stage
consumes.

Which text is implemented
-------------------------
Annex II was replaced by Commission Directive (EU) 2015/996, corrected by the
corrigendum of OJ L 5, 10.1.2018 and amended by Commission Delegated Directive
(EU) 2021/1226. The consolidated text (02002L0049) is what is implemented here,
and every shipped table records the instrument it comes from:

* the roughness-to-frequency conversion uses ``f = v / lambda`` with **v in m/s**
  as corrected in 2018 (the 2015 text says km/h, which is wrong by a factor 3,6);
* the whole of Appendix G is the corrigendum's replacement text, with Tables
  G-1b, G-2, G-3a, G-4 and G-7 as **replaced** by (EU) 2021/1226 and Tables
  G-1a, G-3b, G-3c, G-5 and G-6 as re-issued in 2018 with the band labels
  corrected in 2021;
* curve squeal follows the 2021 rule (5 dB / 8 dB by radius, with a separate
  tram rule and a turnout rule), not the 2015 one;
* bridge noise is a **separate source** built on the transfer function
  ``L_H,bridge,i`` of Table G-7 (2.3.18 as replaced in 2021), not the constant
  ``C_bridge`` of 2015;
* the vertical directivity of source A is the 2021 form, with no absolute-value
  bars and identically zero for ``psi <= 0``. The superseded 2015 form is
  available through :class:`DirectivityEdition` for comparison with pre-2021
  studies, because the two differ over the whole lower half space.

What is verified against digits and what is not
-----------------------------------------------
Annex II prints **no worked example** for the railway source. The end-to-end
chain implemented here is pinned against the emission test workbook published
with the Commission's CNOSSOS-EU source module, which was computed with the
**2015** coefficient database; the shipped tables are therefore verified as
transcriptions, and the equations that combine them are verified end to end
against an independent implementation. Two points are interpretation, not
transcription, and are documented as such:
:class:`RoughnessInterpolation` (the Directive describes the wavelength-to-
frequency resampling in prose only) and the horizontal directivity of traction
noise (2.3.15 enumerates rolling, impact, squeal, braking, fans and aerodynamic
effects; the reference module applies the dipole to every source, which is what
is done here).

Scope
-----
This is the **emission** stage only. Splitting a source line into equivalent
point sources is explicitly outside the scope of the method (2.5.3), and the
CNOSSOS propagation model is not ISO 9613-2, so the hand-off to
:mod:`~phonometry.environmental.outdoor_propagation` mixes two methods and is a
convenience, not a normative chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "AERODYNAMIC_REFERENCE_SPEED",
    "AERODYNAMIC_THRESHOLD_SPEED",
    "RAILWAY_MINIMUM_SPEED",
    "RAILWAY_OCTAVE_BANDS",
    "RAILWAY_SOURCE_HEIGHTS",
    "RAILWAY_THIRD_OCTAVE_BANDS",
    "REFERENCE_JOINT_DENSITY",
    "TRAM_MINIMUM_SPEED",
    "BrakeType",
    "BridgeType",
    "ContactFilter",
    "DirectivityEdition",
    "RailPad",
    "RailRoughnessClass",
    "RailwayEmissionResult",
    "RailwayTrack",
    "RailwayVehicle",
    "RollingStock",
    "RoughnessInterpolation",
    "RunningCondition",
    "TrackBase",
    "TrackCurvature",
    "TrackDescriptor",
    "TrackTransferClass",
    "TractionVehicle",
    "VehicleDescriptor",
    "VehicleType",
    "WheelDiameter",
    "WheelMeasure",
    "aerodynamic_sound_power",
    "bridge_transfer",
    "contact_filter",
    "curve_squeal_excess",
    "horizontal_directivity",
    "impact_roughness",
    "impact_roughness_single",
    "octave_bands_from_third_octaves",
    "rail_roughness",
    "railway_source_power",
    "rolling_sound_power",
    "roughness_to_frequency",
    "superstructure_transfer",
    "total_effective_roughness",
    "track_transfer",
    "traction_sound_power",
    "vertical_directivity",
    "wheel_roughness",
    "wheel_transfer",
]

# ---------------------------------------------------------------------------
# Frequency and wavelength grids
# ---------------------------------------------------------------------------

#: 1/3-octave midband frequencies of the railway source, in Hz. (EU) 2021/1226
#: Annex point (1) states that the railway sound power is derived in 1/3 octave
#: bands; Appendix G tabulates 50 Hz to 10 kHz.
RAILWAY_THIRD_OCTAVE_BANDS: tuple[float, ...] = (
    50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
    5000.0, 6300.0, 8000.0, 10000.0,
)
#: Octave-band midband frequencies handed to the propagation stage, in Hz.
RAILWAY_OCTAVE_BANDS: tuple[float, ...] = (
    63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0,
)
#: Heights of the two equivalent source lines above the rail head, in m
#: (2.3.1): source A at 0,5 m and source B at 4,0 m.
RAILWAY_SOURCE_HEIGHTS: tuple[float, float] = (0.5, 4.0)
#: Speed floor used to determine the total effective roughness of a train, km/h.
RAILWAY_MINIMUM_SPEED = 50.0
#: Speed floor used instead for trams and light metro, km/h.
TRAM_MINIMUM_SPEED = 30.0
#: Speed above which aerodynamic noise is relevant (2.3.13), km/h.
AERODYNAMIC_THRESHOLD_SPEED = 200.0
#: Reference speed ``v_0`` of (2.3.13) and (2.3.14), km/h.
AERODYNAMIC_REFERENCE_SPEED = 300.0
#: Joint density the impact-noise table of Appendix G is given for, in m^-1.
REFERENCE_JOINT_DENSITY = 0.01

#: Wavelength grid of Tables G-1b, G-2 and G-4 as replaced by (EU) 2021/1226,
#: in mm: the standard 1/3-octave series from 2 000 mm to 0,8 mm.
_WAVELENGTHS_STANDARD: tuple[float, ...] = (
    2000.0, 1600.0, 1250.0, 1000.0, 800.0, 630.0, 500.0, 400.0, 315.0, 250.0,
    200.0, 160.0, 125.0, 100.0, 80.0, 63.0, 50.0, 40.0, 31.5, 25.0, 20.0, 16.0,
    12.5, 10.0, 8.0, 6.3, 5.0, 4.0, 3.15, 2.5, 2.0, 1.6, 1.25, 1.0, 0.8,
)
#: Wavelength grid of Table G-1a, which (EU) 2021/1226 did not replace, in mm.
#: It stops at 1 000 mm and keeps the non-standard steps 120, 12 and 3,2 mm
#: where the amended tables read 125, 12,5 and 3,15 mm. Each table is therefore
#: resampled on its own grid rather than forced onto a common one, which is how
#: this implementation resolves an ambiguity the Directive leaves open.
_WAVELENGTHS_WHEEL: tuple[float, ...] = (
    1000.0, 800.0, 630.0, 500.0, 400.0, 315.0, 250.0, 200.0, 160.0, 120.0,
    100.0, 80.0, 63.0, 50.0, 40.0, 31.5, 25.0, 20.0, 16.0, 12.0, 10.0, 8.0,
    6.3, 5.0, 4.0, 3.2, 2.5, 2.0, 1.6, 1.2, 1.0, 0.8,
)

_OCTAVE_SLICES: tuple[tuple[int, int], ...] = (
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24),
)


# ---------------------------------------------------------------------------
# Vehicle and track descriptors, Tables [2.3.a] and [2.3.b]
# ---------------------------------------------------------------------------


class VehicleType(Enum):
    """Digit 1 of the vehicle descriptor, Table [2.3.a]."""

    HIGH_SPEED = "h"          #: High-speed vehicle, above 200 km/h.
    SELF_PROPELLED = "m"      #: Self-propelled passenger coaches.
    HAULED = "p"              #: Hauled passenger coaches.
    CITY_TRAM = "c"           #: City tram or light-metro self-propelled coach.
    DIESEL_LOCO = "d"         #: Diesel locomotive.
    ELECTRIC_LOCO = "e"       #: Electric locomotive.
    FREIGHT = "a"             #: Any generic freight vehicle.
    OTHER = "o"               #: Other, such as maintenance vehicles.


class BrakeType(Enum):
    """Digit 3 of the vehicle descriptor, Table [2.3.a]."""

    CAST_IRON = "c"           #: Cast-iron tread brake.
    COMPOSITE = "k"           #: Composite or sinter-metal tread brake.
    NON_TREAD = "n"           #: Non-tread braked: disc, drum or magnetic.


class WheelMeasure(Enum):
    """Digit 4 of the vehicle descriptor, Table [2.3.a]."""

    NONE = "n"                #: No wheel measure.
    DAMPERS = "d"             #: Wheel dampers.
    SCREENS = "s"             #: Screens.
    OTHER = "o"               #: Other measure.


class TrackBase(Enum):
    """Digit 1 of the track descriptor, Table [2.3.b]."""

    BALLAST = "B"             #: Ballast.
    SLAB = "S"                #: Slab track.
    BALLASTED_BRIDGE = "L"    #: Ballasted bridge.
    NON_BALLASTED_BRIDGE = "N"  #: Non-ballasted bridge.
    EMBEDDED = "T"            #: Embedded track.
    OTHER = "O"               #: Other.


class RailRoughnessClass(Enum):
    """Digit 2 of the track descriptor, Table [2.3.b]."""

    WELL_MAINTAINED = "E"     #: Well maintained and very smooth.
    NORMAL = "M"              #: Normally maintained, smooth.
    NOT_WELL_MAINTAINED = "N"  #: Not well maintained.
    BAD = "B"                 #: Not maintained and in bad condition.


class RailPad(Enum):
    """Digit 3 of the track descriptor: rail-pad **dynamic** stiffness.

    (EU) 2021/1226 Annex point (3) replaced "acoustic" stiffness by
    **dynamic** stiffness and re-worded the hard class as
    "Hard (800-1 000 MN/m)".
    """

    SOFT = "S"                #: Soft, 150-250 MN/m.
    MEDIUM = "M"              #: Medium, 250 to 800 MN/m.
    HARD = "H"                #: Hard, 800-1 000 MN/m.


class TrackMeasure(Enum):
    """Digit 4 of the track descriptor, Table [2.3.b]."""

    NONE = "N"                #: No additional measure.
    RAIL_DAMPER = "D"         #: Rail damper.
    LOW_BARRIER = "B"         #: Low barrier.
    ABSORBER_PLATE = "A"      #: Absorber plate on slab track.
    EMBEDDED_RAIL = "E"       #: Embedded rail.
    OTHER = "O"               #: Other measure.


class RailJoints(Enum):
    """Digit 5 of the track descriptor, Table [2.3.b]."""

    NONE = "N"                #: No joint or switch.
    SINGLE = "S"              #: Single joint or switch.
    TWO = "D"                 #: Two joints or switches per 100 m.
    MORE = "M"                #: More than two joints or switches per 100 m.


class TrackCurvature(Enum):
    """Digit 6 of the track descriptor, Table [2.3.b]."""

    STRAIGHT = "N"            #: Straight track.
    LOW = "L"                 #: Low curvature, 1 000-500 m.
    MEDIUM = "M"              #: Medium curvature, below 500 m and above 300 m.
    HIGH = "H"                #: High curvature, below 300 m.


@dataclass(frozen=True)
class VehicleDescriptor:
    """The four-digit vehicle descriptor of Table [2.3.a].

    :ivar vehicle_type: Digit 1, the :class:`VehicleType`.
    :ivar axles: Digit 2, the number of axles per vehicle.
    :ivar brake: Digit 3, the :class:`BrakeType`.
    :ivar measure: Digit 4, the :class:`WheelMeasure`.
    """

    vehicle_type: VehicleType
    axles: int
    brake: BrakeType
    measure: WheelMeasure = WheelMeasure.NONE

    @classmethod
    def from_code(cls, code: str) -> VehicleDescriptor:
        """Parse a descriptor such as ``"a4cn"`` or ``"h16nn"``.

        The second digit is the actual number of axles, so it may run to more
        than one character.

        :param code: The descriptor, first digit to last.
        :return: The parsed :class:`VehicleDescriptor`.
        :raises ValueError: If the code is not a valid four-digit descriptor.
        """
        text = str(code).strip()
        if len(text) < 4 or not text[1:-2].isdigit():
            raise ValueError(
                f"{code!r} is not a Table [2.3.a] vehicle descriptor: it must "
                "read <type><axles><brake><measure>, for example 'a4cn'."
            )
        try:
            return cls(
                vehicle_type=VehicleType(text[0]),
                axles=int(text[1:-2]),
                brake=BrakeType(text[-2]),
                measure=WheelMeasure(text[-1]),
            )
        except ValueError as exc:  # pragma: no cover - message pass-through
            raise ValueError(
                f"{code!r} is not a Table [2.3.a] vehicle descriptor: {exc}"
            ) from exc

    @property
    def code(self) -> str:
        """The descriptor written back out as a string."""
        return (
            f"{self.vehicle_type.value}{self.axles}"
            f"{self.brake.value}{self.measure.value}"
        )


@dataclass(frozen=True)
class TrackDescriptor:
    """The six-digit track descriptor of Table [2.3.b].

    :ivar base: Digit 1, the :class:`TrackBase`.
    :ivar roughness: Digit 2, the :class:`RailRoughnessClass`.
    :ivar pad: Digit 3, the :class:`RailPad` dynamic stiffness.
    :ivar measure: Digit 4, the :class:`TrackMeasure`.
    :ivar joints: Digit 5, the :class:`RailJoints`.
    :ivar curvature: Digit 6, the :class:`TrackCurvature`.
    """

    base: TrackBase
    roughness: RailRoughnessClass
    pad: RailPad
    measure: TrackMeasure = TrackMeasure.NONE
    joints: RailJoints = RailJoints.NONE
    curvature: TrackCurvature = TrackCurvature.STRAIGHT

    @classmethod
    def from_code(cls, code: str) -> TrackDescriptor:
        """Parse a descriptor such as ``"BMSNNN"``.

        :param code: The six-character descriptor.
        :return: The parsed :class:`TrackDescriptor`.
        :raises ValueError: If the code is not a valid six-digit descriptor.
        """
        text = str(code).strip()
        if len(text) != 6:
            raise ValueError(
                f"{code!r} is not a Table [2.3.b] track descriptor: it must "
                "have exactly six digits, for example 'BMSNNN'."
            )
        try:
            return cls(
                base=TrackBase(text[0]),
                roughness=RailRoughnessClass(text[1]),
                pad=RailPad(text[2]),
                measure=TrackMeasure(text[3]),
                joints=RailJoints(text[4]),
                curvature=TrackCurvature(text[5]),
            )
        except ValueError as exc:  # pragma: no cover - message pass-through
            raise ValueError(
                f"{code!r} is not a Table [2.3.b] track descriptor: {exc}"
            ) from exc

    @property
    def code(self) -> str:
        """The descriptor written back out as a string."""
        return "".join(
            part.value for part in (
                self.base, self.roughness, self.pad, self.measure, self.joints,
                self.curvature,
            )
        )


class RunningCondition(Enum):
    """Running condition ``c`` of 2.3.2.

    Only two conditions are modelled: constant speed, which the Directive says
    is valid as well when the train accelerates or decelerates, and idling.
    """

    CONSTANT = 1              #: ``c = 1``, constant speed.
    IDLING = 2                #: ``c = 2``, idling.


class DirectivityEdition(Enum):
    """Which text of the vertical directivity (2.3.16) to evaluate."""

    #: The consolidated text: (EU) 2021/1226 Annex point (4)(d) removed the
    #: absolute-value bars and made the correction identically zero for
    #: ``psi <= 0``.
    CURRENT = "2021/1226"
    #: Commission Directive (EU) 2015/996 as published: the whole expression
    #: inside absolute-value bars, over the whole range ``-pi/2 < psi < pi/2``.
    #: Kept because the two forms differ over the entire lower half space, so a
    #: comparison with a pre-2021 study needs it.
    ORIGINAL_2015 = "2015/996"


class RoughnessInterpolation(Enum):
    """How a roughness spectrum is resampled from wavelength onto frequency.

    The Directive describes the resampling in prose only: "the two
    corresponding 1/3 octave bands defined in the wavelength domain shall be
    averaged energetically and proportionally". No formula and no example is
    given, so the rule has to be chosen, and the choice is the single largest
    interpretation risk of the railway model.
    """

    #: Linear interpolation of the **levels** between the two neighbouring
    #: wavelength bands, weighted proportionally to the wavelength. This is what
    #: the Commission's reference source module does, and it is what the
    #: emission test workbook shipped with it is reproduced with.
    PROPORTIONAL = "proportional"
    #: Linear interpolation of the **energies**, weighted the same way: a
    #: literal reading of "averaged energetically". It differs from
    #: :attr:`PROPORTIONAL` by up to about 1 dB on a steep spectrum.
    ENERGY = "energy"


# ---------------------------------------------------------------------------
# Appendix G, Table G-1a - wheel roughness L_r,VEH,i, as re-issued by the
# corrigendum of OJ L 5, 10.1.2018 with the band labels corrected by
# (EU) 2021/1226 Annex point (20)(d). Columns are the brake type of digit 3 of
# the vehicle descriptor. Values in dB, on the wavelength grid
# _WAVELENGTHS_WHEEL.
# ---------------------------------------------------------------------------
_TABLE_G1A: dict[str, tuple[float, ...]] = {
    "c": (
        2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.4, 0.6, 2.6, 5.8, 8.8, 11.1,
        11.0, 9.8, 7.5, 5.1, 3.0, 1.3, 0.2, -0.7, -1.2, -1.0, 0.3, 0.2, 1.3,
        3.1, 3.1, 3.1, 3.1, 3.1,
    ),
    "k": (
        -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.3,
        -4.6, -4.9, -5.2, -6.3, -6.8, -7.2, -7.3, -7.3, -7.1, -6.9, -6.7, -6.0,
        -3.7, -2.4, -2.6, -2.5, -2.5, -2.5, -2.5, -2.5,
    ),
    "n": (
        -5.9, -5.9, -5.9, -5.9, -5.9, -5.9, 2.3, 2.8, 2.6, 1.2, 2.1, 0.9, -0.3,
        -1.6, -2.9, -4.9, -7.0, -8.6, -9.3, -9.5, -10.1, -10.3, -10.3, -10.8,
        -10.9, -9.5, -9.5, -9.5, -9.5, -9.5, -9.5, -9.5,
    ),
}

# ---------------------------------------------------------------------------
# Table G-1b - rail roughness L_r,TR,i, as replaced by (EU) 2021/1226 Annex
# point (20)(a): a new wavelength grid running to 2 000 mm and a new M column.
# ---------------------------------------------------------------------------
_TABLE_G1B: dict[str, tuple[float, ...]] = {
    "E": (
        17.1, 17.1, 17.1, 17.1, 17.1, 17.1, 17.1, 17.1, 15.0, 13.0, 11.0, 9.0,
        7.0, 4.9, 2.9, 0.9, -1.1, -3.2, -5.0, -5.6, -6.2, -6.8, -7.4, -8.0,
        -8.6, -9.2, -9.8, -10.4, -11.0, -11.6, -12.2, -12.8, -13.4, -14.0,
        -14.0,
    ),
    "M": (
        35.0, 31.0, 28.0, 25.0, 23.0, 20.0, 17.0, 13.5, 10.5, 9.0, 6.5, 5.5,
        5.0, 3.5, 2.0, 0.1, -0.2, -0.3, -0.8, -3.0, -5.0, -7.0, -8.0, -9.0,
        -10.0, -12.0, -13.0, -14.0, -15.0, -16.0, -17.0, -18.0, -19.0, -19.0,
        -19.0,
    ),
}

# ---------------------------------------------------------------------------
# Table G-2 - contact filter A_3,i, as replaced by (EU) 2021/1226 Annex point
# (20)(b), which reads "wheel load" where 2015/996 read "axle load" and changed
# the column order and every value. Keys are (wheel load in kN, wheel diameter
# in mm).
# ---------------------------------------------------------------------------
_ELEVEN_ZEROS: tuple[float, ...] = (0.0,) * 11
_TABLE_G2: dict[tuple[float, float], tuple[float, ...]] = {
    (50.0, 360.0): _ELEVEN_ZEROS + (
        0.0, 0.0, 0.0, -0.1, -0.2, -0.3, -0.6, -1.0, -1.8, -3.2, -5.4, -8.7,
        -12.2, -16.7, -17.7, -17.8, -20.7, -22.1, -22.8, -24.0, -24.5, -24.7,
        -27.0, -27.8,
    ),
    (50.0, 680.0): _ELEVEN_ZEROS + (
        0.0, 0.0, -0.1, -0.2, -0.3, -0.7, -1.2, -2.0, -4.1, -6.0, -9.2, -13.8,
        -17.2, -17.7, -18.6, -21.5, -22.3, -23.1, -24.4, -24.5, -25.0, -28.0,
        -28.8, -29.6,
    ),
    (50.0, 920.0): _ELEVEN_ZEROS + (
        0.0, -0.1, -0.1, -0.3, -0.6, -1.1, -1.3, -3.5, -5.3, -8.0, -12.0,
        -16.8, -17.7, -18.0, -21.5, -21.8, -22.8, -24.0, -24.5, -25.0, -27.3,
        -28.1, -28.9, -29.7,
    ),
    (25.0, 920.0): _ELEVEN_ZEROS + (
        0.0, 0.0, 0.0, -0.1, -0.3, -0.5, -1.1, -1.8, -3.3, -5.3, -7.9, -12.8,
        -16.8, -17.7, -18.2, -20.5, -22.0, -22.8, -24.2, -24.5, -25.0, -27.4,
        -28.2, -29.0,
    ),
    (100.0, 920.0): _ELEVEN_ZEROS + (
        -0.1, -0.2, -0.3, -0.6, -1.0, -1.8, -3.2, -5.4, -8.7, -12.2, -16.7,
        -17.7, -17.8, -20.7, -22.1, -22.8, -24.0, -24.5, -24.7, -27.0, -27.8,
        -28.6, -29.4, -30.2,
    ),
}

# ---------------------------------------------------------------------------
# Table G-3a - track transfer function L_H,TR,i, as replaced by (EU) 2021/1226
# Annex point (20)(c), which fixed the column codes the 2018 corrigendum got
# wrong and added the new column D. Sound power level per axle, dB.
# ---------------------------------------------------------------------------
_TABLE_G3A: dict[str, tuple[float, ...]] = {
    "M/S": (
        53.3, 59.3, 67.2, 75.9, 79.2, 81.8, 84.2, 88.6, 91.0, 94.5, 97.0, 99.2,
        104.0, 107.1, 108.3, 108.5, 109.7, 110.0, 110.0, 110.0, 110.3, 110.0,
        110.1, 110.6,
    ),
    "M/M": (
        50.9, 57.8, 66.5, 76.8, 80.9, 83.3, 85.8, 90.0, 91.6, 93.9, 95.6, 97.4,
        101.7, 104.4, 106.0, 106.8, 108.3, 108.9, 109.1, 109.4, 109.9, 109.9,
        110.3, 111.0,
    ),
    "M/H": (
        50.1, 57.2, 66.3, 77.2, 81.6, 84.0, 86.5, 90.7, 92.1, 94.3, 95.8, 97.0,
        100.3, 102.5, 104.2, 105.4, 107.1, 107.9, 108.2, 108.7, 109.4, 109.7,
        110.4, 111.4,
    ),
    "B/S": (
        50.9, 56.6, 64.3, 72.3, 75.4, 78.5, 81.8, 86.6, 89.1, 91.9, 94.5, 97.5,
        104.0, 107.9, 108.9, 108.8, 109.8, 110.2, 110.1, 110.1, 110.3, 109.9,
        110.0, 110.4,
    ),
    "B/M": (
        50.0, 56.1, 64.1, 72.5, 75.8, 79.1, 83.6, 88.7, 89.6, 89.7, 90.6, 93.8,
        100.6, 104.7, 106.3, 107.1, 108.8, 109.3, 109.4, 109.7, 110.0, 109.8,
        110.0, 110.5,
    ),
    "B/H": (
        49.8, 55.9, 64.0, 72.5, 75.9, 79.4, 84.4, 89.7, 90.2, 90.2, 90.8, 93.1,
        97.9, 101.1, 103.4, 105.4, 107.7, 108.5, 108.7, 109.1, 109.6, 109.6,
        109.9, 110.6,
    ),
    "W": (
        44.0, 51.0, 59.9, 70.8, 75.1, 76.9, 77.2, 80.9, 85.3, 92.5, 97.0, 98.7,
        102.8, 105.4, 106.5, 106.4, 107.5, 108.1, 108.4, 108.7, 109.1, 109.1,
        109.5, 110.2,
    ),
    "D": (
        75.4, 77.4, 81.4, 87.1, 88.0, 89.7, 83.4, 87.7, 89.8, 97.5, 99.0,
        100.8, 104.9, 111.8, 113.9, 115.5, 114.9, 118.2, 118.3, 118.4, 118.9,
        117.5, 117.9, 118.6,
    ),
}

# ---------------------------------------------------------------------------
# Table G-3b - wheel transfer function L_H,VEH,i. Values unchanged since
# 2015/996; the band labels 316, 3 160 and 6 350 Hz were corrected to 315,
# 3 150 and 6 300 Hz by (EU) 2021/1226 Annex point (20)(d). Columns are the
# wheel diameter, all "no measure". Sound power level per axle, dB.
# ---------------------------------------------------------------------------
_TABLE_G3B: dict[float, tuple[float, ...]] = {
    920.0: (
        75.4, 77.3, 81.1, 84.1, 83.3, 84.3, 86.0, 90.1, 89.8, 89.0, 88.8, 90.4,
        92.4, 94.9, 100.4, 104.6, 109.6, 114.9, 115.0, 115.0, 115.5, 115.6,
        116.0, 116.7,
    ),
    840.0: (
        75.4, 77.3, 81.1, 84.1, 82.8, 83.3, 84.1, 86.9, 87.9, 89.9, 90.9, 91.5,
        91.5, 93.0, 98.7, 101.6, 107.6, 111.9, 114.5, 114.5, 115.0, 115.1,
        115.5, 116.2,
    ),
    680.0: (
        75.4, 77.3, 81.1, 84.1, 82.8, 83.3, 83.9, 86.3, 88.0, 92.2, 93.9, 92.5,
        90.9, 90.4, 93.2, 93.5, 99.6, 104.9, 108.0, 111.0, 111.5, 111.6, 112.0,
        112.7,
    ),
    1200.0: (
        75.4, 77.3, 81.1, 84.1, 82.8, 83.3, 84.5, 90.4, 90.4, 89.9, 90.1, 91.3,
        91.5, 93.6, 100.5, 104.6, 115.6, 115.9, 116.0, 116.0, 116.5, 116.6,
        117.0, 117.7,
    ),
}

#: Table G-3c - superstructure transfer function ``L_H,VEH,SUP,i`` of the only
#: tabulated superstructure ("EU standard", vehicle type ``a``): 0,0 dB in every
#: 1/3-octave band from 50 Hz to 10 kHz.
_TABLE_G3C: tuple[float, ...] = (0.0,) * 24

# ---------------------------------------------------------------------------
# Table G-4 - impact-noise roughness L_R,IMPACT,i for a single switch, joint or
# crossing per 100 m, as replaced by (EU) 2021/1226 Annex point (20)(e).
# ---------------------------------------------------------------------------
_TABLE_G4: tuple[float, ...] = (
    22.0, 22.0, 22.0, 22.0, 22.0, 20.0, 16.0, 15.0, 14.0, 15.0, 14.0, 12.0,
    11.0, 10.0, 9.0, 8.0, 6.0, 3.0, 2.0, -3.0, -8.0, -13.0, -17.0, -19.0,
    -22.0, -25.0, -26.0, -32.0, -35.0, -40.0, -43.0, -45.0, -47.0, -49.0,
    -50.0,
)


class TractionVehicle(Enum):
    """Columns of Table G-5, the traction sound power per vehicle."""

    DIESEL_LOCO_800 = "diesel locomotive, c. 800 kW"
    DIESEL_LOCO_2200 = "diesel locomotive, c. 2 200 kW"
    DIESEL_MULTIPLE_UNIT = "diesel multiple unit"
    ELECTRIC_LOCO = "electric locomotive"
    ELECTRIC_MULTIPLE_UNIT = "electric multiple unit"


# ---------------------------------------------------------------------------
# Table G-5 - traction sound power per vehicle, source A then source B. Values
# unchanged since 2015/996 except the 6 300 Hz pair of the 2 200 kW diesel
# locomotive, which (EU) 2021/1226 Annex point (20)(f) corrected from 31,4/30,7
# to 81,4/80,7; the same point corrected the band labels. Because
# L_W,0,const,i = L_W,0,idling,i, this one table serves both running conditions.
# ---------------------------------------------------------------------------
_TABLE_G5: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    TractionVehicle.DIESEL_LOCO_800.value: (
        (
            98.9, 94.8, 92.6, 94.6, 92.8, 92.8, 93.0, 94.8, 94.6, 95.7, 95.6,
            98.6, 95.2, 95.1, 95.1, 94.1, 94.1, 99.4, 92.5, 89.5, 87.0, 84.1,
            81.5, 79.2,
        ),
        (
            103.2, 100.0, 95.5, 94.0, 93.3, 93.6, 92.9, 92.7, 92.4, 92.8, 92.8,
            96.8, 92.7, 93.0, 92.9, 93.1, 93.2, 98.3, 91.5, 88.7, 86.0, 83.4,
            80.9, 78.7,
        ),
    ),
    TractionVehicle.DIESEL_LOCO_2200.value: (
        (
            99.4, 107.3, 103.1, 102.1, 99.3, 99.3, 99.5, 101.3, 101.1, 102.2,
            102.1, 101.1, 101.7, 101.6, 99.3, 96.0, 93.7, 101.9, 89.5, 87.1,
            90.5, 81.4, 81.2, 79.6,
        ),
        (
            103.7, 112.5, 106.0, 101.5, 99.8, 100.1, 99.4, 99.2, 98.9, 99.3,
            99.3, 99.3, 99.2, 99.5, 97.1, 95.0, 92.8, 100.8, 88.5, 86.3, 89.5,
            80.7, 80.6, 79.1,
        ),
    ),
    TractionVehicle.DIESEL_MULTIPLE_UNIT.value: (
        (
            82.6, 82.5, 89.3, 90.3, 93.5, 99.5, 98.7, 95.5, 90.3, 91.4, 91.3,
            90.3, 90.9, 91.8, 92.8, 92.8, 90.8, 88.1, 85.2, 83.2, 81.7, 78.8,
            76.2, 73.9,
        ),
        (
            86.9, 87.7, 92.2, 89.7, 94.0, 100.3, 98.6, 93.4, 88.1, 88.5, 88.5,
            88.5, 88.4, 89.7, 90.6, 91.8, 89.9, 87.0, 84.2, 82.4, 80.7, 78.1,
            75.6, 73.4,
        ),
    ),
    TractionVehicle.ELECTRIC_LOCO.value: (
        (
            87.9, 90.8, 91.6, 94.6, 94.8, 96.8, 104.0, 100.8, 99.6, 101.7,
            98.6, 95.6, 95.2, 96.1, 92.1, 89.1, 87.1, 85.4, 83.5, 81.5, 80.0,
            78.1, 76.5, 75.2,
        ),
        (
            92.2, 96.0, 94.5, 94.0, 95.3, 97.6, 103.9, 98.7, 97.4, 98.8, 95.8,
            93.8, 92.7, 94.0, 89.9, 88.1, 86.2, 84.3, 82.5, 80.7, 79.0, 77.4,
            75.9, 74.7,
        ),
    ),
    TractionVehicle.ELECTRIC_MULTIPLE_UNIT.value: (
        (
            80.5, 81.4, 80.5, 82.2, 80.0, 79.7, 79.6, 96.4, 80.5, 81.3, 97.2,
            79.5, 79.8, 86.7, 81.7, 82.7, 80.7, 78.0, 75.1, 72.1, 69.6, 66.7,
            64.1, 61.8,
        ),
        (
            84.8, 86.6, 83.4, 81.6, 80.5, 80.5, 79.5, 94.3, 78.3, 78.4, 94.4,
            77.7, 77.3, 84.6, 79.5, 81.7, 79.8, 76.9, 74.1, 71.3, 68.6, 66.0,
            63.5, 61.3,
        ),
    ),
}

# ---------------------------------------------------------------------------
# Table G-6 - aerodynamic sound power per vehicle at v_0 = 300 km/h for a 20 m
# vehicle, source A then source B. Values unchanged since 2015/996; band labels
# corrected by (EU) 2021/1226 Annex point (20)(g). The speed exponents are
# alpha_1 = alpha_2 = 50 in every band.
# ---------------------------------------------------------------------------
_TABLE_G6_A: tuple[float, ...] = (
    112.6, 113.2, 115.7, 117.4, 115.3, 115.0, 114.9, 116.4, 115.9, 116.3,
    116.2, 115.2, 115.8, 115.7, 115.7, 114.7, 114.7, 115.0, 114.5, 113.1,
    112.1, 110.6, 109.6, 108.8,
)
_TABLE_G6_B: tuple[float, ...] = (
    36.7, 38.5, 39.0, 37.5, 36.8, 37.1, 36.4, 36.2, 35.9, 36.3, 36.3, 36.3,
    36.2, 36.5, 36.4, 105.2, 110.3, 110.4, 105.6, 37.2, 37.5, 37.9, 38.4, 39.2,
)
_AERODYNAMIC_ALPHA = 50.0


class BridgeType(Enum):
    """Columns of Table G-7, labelled by the A-weighted bridge excess."""

    PLUS_10_DBA = "+10 dB(A)"
    PLUS_15_DBA = "+15 dB(A)"


# ---------------------------------------------------------------------------
# Table G-7 - bridge transfer function L_H,bridge,i, as replaced by
# (EU) 2021/1226 Annex point (20)(h). It completely replaces the C_bridge
# constants of 2015/996 (1 dB for concrete or masonry, 4 dB for steel with
# ballasted track). Sound power level per axle, dB.
# ---------------------------------------------------------------------------
_TABLE_G7: dict[str, tuple[float, ...]] = {
    BridgeType.PLUS_10_DBA.value: (
        85.2, 87.1, 91.0, 94.0, 94.4, 96.0, 92.5, 96.7, 97.4, 99.4, 100.7,
        102.5, 107.1, 109.8, 112.0, 107.2, 106.8, 107.3, 99.3, 91.4, 86.9,
        79.7, 75.1, 70.8,
    ),
    BridgeType.PLUS_15_DBA.value: (
        90.1, 92.1, 96.0, 99.5, 99.9, 101.5, 99.6, 103.8, 104.5, 106.5, 107.8,
        109.6, 116.1, 118.8, 120.9, 109.5, 109.1, 109.6, 102.0, 94.1, 89.6,
        83.6, 79.0, 74.7,
    ),
}

#: Curve-squeal excess added to the rolling-noise sound power at all
#: frequencies, in dB, as replaced by (EU) 2021/1226 Annex point (4)(b).
_SQUEAL_TRAIN_TIGHT = 8.0
_SQUEAL_TRAIN_MODERATE = 5.0
_SQUEAL_TRAM = 5.0
_SQUEAL_TRAIN_TIGHT_RADIUS = 300.0
_SQUEAL_TRAIN_MODERATE_RADIUS = 500.0
_SQUEAL_TRAM_RADIUS = 200.0
_SQUEAL_MINIMUM_TRACK_LENGTH = 50.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finite(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"'{name}' must be a finite number.")
    return scalar


def _speed(value: float, name: str = "speed") -> float:
    v = _finite(value, name)
    if v <= 0.0:
        raise ValueError(f"'{name}' must be a positive number of km/h.")
    return v


def _spectrum(values: Any, name: str, size: int = 24) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(
            f"'{name}' must hold {size} values, one per band; got "
            f"{array.shape[0] if array.ndim == 1 else array.shape}."
        )
    return array


def _energy_sum(*spectra: NDArray[np.float64]) -> NDArray[np.float64]:
    total = sum(10.0 ** (np.asarray(s, dtype=np.float64) / 10.0) for s in spectra)
    with np.errstate(divide="ignore"):
        return np.asarray(10.0 * np.log10(total), dtype=np.float64)


# ---------------------------------------------------------------------------
# Appendix G look-ups
# ---------------------------------------------------------------------------


def wheel_roughness(brake: BrakeType | str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wheel roughness ``L_r,VEH`` of Table G-1a.

    :param brake: The :class:`BrakeType` of digit 3 of the vehicle descriptor.
    :return: ``(wavelengths in mm, levels in dB)``.
    :raises ValueError: If the brake type is not tabulated.
    """
    key = brake.value if isinstance(brake, BrakeType) else str(brake)
    if key not in _TABLE_G1A:
        raise ValueError(
            f"Unknown brake type {key!r}; Table G-1a tabulates: "
            + ", ".join(_TABLE_G1A)
        )
    return (
        np.asarray(_WAVELENGTHS_WHEEL, dtype=np.float64),
        np.asarray(_TABLE_G1A[key], dtype=np.float64),
    )


def rail_roughness(
    roughness: RailRoughnessClass | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rail roughness ``L_r,TR`` of Table G-1b.

    Only the two maintained classes ``E`` and ``M`` are tabulated; the ``N``
    and ``B`` classes of Table [2.3.b] carry no spectrum in Appendix G and have
    to be supplied by the Member State.

    :param roughness: The :class:`RailRoughnessClass` of digit 2 of the track
        descriptor.
    :return: ``(wavelengths in mm, levels in dB)``.
    :raises ValueError: If the class carries no spectrum in Table G-1b.
    """
    key = (
        roughness.value if isinstance(roughness, RailRoughnessClass)
        else str(roughness)
    )
    if key not in _TABLE_G1B:
        raise ValueError(
            f"Rail roughness class {key!r} has no spectrum in Table G-1b, "
            "which only tabulates " + " and ".join(_TABLE_G1B)
            + "; supply a national spectrum instead."
        )
    return (
        np.asarray(_WAVELENGTHS_STANDARD, dtype=np.float64),
        np.asarray(_TABLE_G1B[key], dtype=np.float64),
    )


class ContactFilter(Enum):
    """Columns of Table G-2, as ``(wheel load in kN, wheel diameter in mm)``."""

    LOAD_50_DIAMETER_360 = (50.0, 360.0)
    LOAD_50_DIAMETER_680 = (50.0, 680.0)
    LOAD_50_DIAMETER_920 = (50.0, 920.0)
    LOAD_25_DIAMETER_920 = (25.0, 920.0)
    LOAD_100_DIAMETER_920 = (100.0, 920.0)


def contact_filter(
    filter_: ContactFilter | tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Contact filter ``A_3`` of Table G-2.

    :param filter_: A :class:`ContactFilter` member or the ``(wheel load in kN,
        wheel diameter in mm)`` pair that labels the column.
    :return: ``(wavelengths in mm, levels in dB)``.
    :raises ValueError: If the combination is not tabulated.
    """
    key = filter_.value if isinstance(filter_, ContactFilter) else (
        float(filter_[0]), float(filter_[1])
    )
    if key not in _TABLE_G2:
        raise ValueError(
            f"Unknown contact filter {key!r}; Table G-2 tabulates the wheel "
            "load and wheel diameter pairs "
            + ", ".join(str(k) for k in _TABLE_G2)
        )
    return (
        np.asarray(_WAVELENGTHS_STANDARD, dtype=np.float64),
        np.asarray(_TABLE_G2[key], dtype=np.float64),
    )


class TrackTransferClass(Enum):
    """Columns of Table G-3a, ``track base / rail pad`` of the track descriptor."""

    MONOBLOCK_SOFT = "M/S"          #: Concrete mono-block sleeper, soft pad.
    MONOBLOCK_MEDIUM = "M/M"        #: Concrete mono-block sleeper, medium pad.
    MONOBLOCK_HARD = "M/H"          #: Concrete mono-block sleeper, hard pad.
    BIBLOCK_SOFT = "B/S"            #: Concrete bi-block sleeper, soft pad.
    BIBLOCK_MEDIUM = "B/M"          #: Concrete bi-block sleeper, medium pad.
    BIBLOCK_HARD = "B/H"            #: Concrete bi-block sleeper, hard pad.
    WOODEN = "W"                    #: Wooden sleepers.
    DIRECT_FASTENING = "D"          #: Direct fastening on bridges.


def track_transfer(track: TrackTransferClass | str) -> NDArray[np.float64]:
    """Track transfer function ``L_H,TR,i`` of Table G-3a, in dB per axle.

    :param track: A :class:`TrackTransferClass` member or its column code.
    :return: The 24 1/3-octave values, in dB.
    :raises ValueError: If the column is not tabulated.
    """
    key = track.value if isinstance(track, TrackTransferClass) else str(track)
    if key not in _TABLE_G3A:
        raise ValueError(
            f"Unknown track transfer column {key!r}; Table G-3a tabulates: "
            + ", ".join(_TABLE_G3A)
        )
    return np.asarray(_TABLE_G3A[key], dtype=np.float64)


class WheelDiameter(Enum):
    """Columns of Table G-3b, the wheel diameter in mm, all "no measure"."""

    MM_920 = 920.0
    MM_840 = 840.0
    MM_680 = 680.0
    MM_1200 = 1200.0


def wheel_transfer(diameter: WheelDiameter | float) -> NDArray[np.float64]:
    """Wheel transfer function ``L_H,VEH,i`` of Table G-3b, in dB per axle.

    :param diameter: A :class:`WheelDiameter` member or the diameter in mm.
    :return: The 24 1/3-octave values, in dB.
    :raises ValueError: If the diameter is not tabulated.
    """
    key = diameter.value if isinstance(diameter, WheelDiameter) else float(diameter)
    if key not in _TABLE_G3B:
        raise ValueError(
            f"Unknown wheel diameter {key!r} mm; Table G-3b tabulates: "
            + ", ".join(str(k) for k in _TABLE_G3B)
        )
    return np.asarray(_TABLE_G3B[key], dtype=np.float64)


def superstructure_transfer() -> NDArray[np.float64]:
    """Superstructure transfer ``L_H,VEH,SUP,i`` of Table G-3c, in dB per axle.

    Only one superstructure is tabulated, the "EU standard" of vehicle type
    ``a`` (freight), and it is 0,0 dB in every band, so (2.3.10) reduces to
    ``L_R,TOT,i + 10 lg(N_a)``. The contribution is considered for freight
    wagons only.

    :return: The 24 1/3-octave values, all zero.
    """
    return np.asarray(_TABLE_G3C, dtype=np.float64)


def impact_roughness_single() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Impact roughness ``L_R,IMPACT-SINGLE`` of Table G-4.

    The table is given for a joint density ``n_l = 0,01 m^-1``, that is one
    switch, joint or crossing per 100 m, which is also the default the
    Directive prescribes for jointed track.

    :return: ``(wavelengths in mm, levels in dB)``.
    """
    return (
        np.asarray(_WAVELENGTHS_STANDARD, dtype=np.float64),
        np.asarray(_TABLE_G4, dtype=np.float64),
    )


def traction_sound_power(
    vehicle: TractionVehicle | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Traction sound power per vehicle of Table G-5, in dB.

    Because the Directive models only constant speed and idling and takes the
    source strength at maximum load, ``L_W,0,const,i = L_W,0,idling,i``, so this
    one table serves both running conditions.

    :param vehicle: A :class:`TractionVehicle` member or its description.
    :return: ``(source A spectrum, source B spectrum)``, 24 values each.
    :raises ValueError: If the vehicle is not tabulated.
    """
    key = vehicle.value if isinstance(vehicle, TractionVehicle) else str(vehicle)
    if key not in _TABLE_G5:
        raise ValueError(
            f"Unknown traction vehicle {key!r}; Table G-5 tabulates: "
            + ", ".join(_TABLE_G5)
        )
    low, high = _TABLE_G5[key]
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def aerodynamic_sound_power(
    speed: float = AERODYNAMIC_REFERENCE_SPEED,
    *,
    reference: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    alpha: float = _AERODYNAMIC_ALPHA,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Aerodynamic sound power of (2.3.13) and (2.3.14), in dB.

    ``L_W,0,i = L_W,0,h,i(v_0) + alpha_h,i lg(v/v_0)`` with ``v_0 = 300 km/h``.
    At the reference speed the result is Table G-6 verbatim.

    :param speed: Train speed ``v``, in km/h.
    :param reference: ``(source A, source B)`` reference spectra at ``v_0``, or
        ``None`` (the default) for Table G-6.
    :param alpha: Speed exponent ``alpha_h,i``; Table G-6 gives 50 in every band.
    :return: ``(source A spectrum, source B spectrum)``, 24 values each.
    :raises ValueError: If the speed is not positive.
    """
    v = _speed(speed)
    low, high = (
        (np.asarray(_TABLE_G6_A, dtype=np.float64),
         np.asarray(_TABLE_G6_B, dtype=np.float64))
        if reference is None
        else (_spectrum(reference[0], "reference[0]"),
              _spectrum(reference[1], "reference[1]"))
    )
    shift = _finite(alpha, "alpha") * np.log10(v / AERODYNAMIC_REFERENCE_SPEED)
    return low + shift, high + shift


def bridge_transfer(bridge: BridgeType | str) -> NDArray[np.float64]:
    """Bridge transfer function ``L_H,bridge,i`` of Table G-7, in dB per axle.

    :param bridge: A :class:`BridgeType` member or its column label.
    :return: The 24 1/3-octave values, in dB.
    :raises ValueError: If the column is not tabulated.
    """
    key = bridge.value if isinstance(bridge, BridgeType) else str(bridge)
    if key not in _TABLE_G7:
        raise ValueError(
            f"Unknown bridge type {key!r}; Table G-7 tabulates: "
            + ", ".join(_TABLE_G7)
        )
    return np.asarray(_TABLE_G7[key], dtype=np.float64)


# ---------------------------------------------------------------------------
# Roughness, transfer functions and the physical sources
# ---------------------------------------------------------------------------


def roughness_to_frequency(
    levels: Any,
    wavelengths: Any,
    speed: float,
    *,
    frequencies: Any = RAILWAY_THIRD_OCTAVE_BANDS,
    interpolation: RoughnessInterpolation = RoughnessInterpolation.PROPORTIONAL,
) -> NDArray[np.float64]:
    """Resample a roughness spectrum from wavelength onto frequency.

    A roughness level is tabulated against wavelength and has to be read at
    ``lambda = v/f`` with **v in m/s** (the corrigendum of OJ L 5, 10.1.2018;
    the 2015 text says km/h, which is wrong by a factor 3,6). The value at the
    wanted wavelength is obtained from the two neighbouring tabulated bands
    according to ``interpolation``; beyond the ends of the table the end value
    is held.

    :param levels: Roughness levels of the table, in dB.
    :param wavelengths: Wavelengths of the table, in mm, in any monotonic order.
    :param speed: Train speed ``v``, in km/h.
    :param frequencies: Target midband frequencies, in Hz.
    :param interpolation: The :class:`RoughnessInterpolation` rule.
    :return: The spectrum on the target frequency grid, in dB.
    :raises ValueError: If an input is invalid.
    """
    y = np.asarray(levels, dtype=np.float64)
    lam = np.asarray(wavelengths, dtype=np.float64) / 1000.0
    if y.ndim != 1 or y.shape != lam.shape:
        raise ValueError(
            "'levels' and 'wavelengths' must be one-dimensional and of the "
            f"same length; got {y.shape} and {lam.shape}."
        )
    if np.any(lam <= 0.0):
        raise ValueError("'wavelengths' must all be positive.")
    freqs = np.asarray(frequencies, dtype=np.float64)
    if np.any(freqs <= 0.0):
        raise ValueError("'frequencies' must all be positive.")
    order = np.argsort(lam)
    lam, y = lam[order], y[order]
    wanted = _speed(speed) / 3.6 / freqs
    if interpolation is RoughnessInterpolation.ENERGY:
        return np.asarray(
            10.0 * np.log10(np.interp(wanted, lam, 10.0 ** (y / 10.0))),
            dtype=np.float64,
        )
    return np.asarray(np.interp(wanted, lam, y), dtype=np.float64)


def total_effective_roughness(
    rail: Any, wheel: Any, filter_: Any
) -> NDArray[np.float64]:
    """Total effective roughness ``L_R,TOT,i`` of (2.3.7), in dB.

    ``L_R,TOT,i = 10 lg(10^(L_r,TR,i/10) + 10^(L_r,VEH,i/10)) + A_3,i``. All
    three spectra must already be on the frequency grid, that is resampled with
    :func:`roughness_to_frequency` at the speed of interest.

    :param rail: Rail roughness ``L_r,TR,i``, in dB.
    :param wheel: Wheel roughness ``L_r,VEH,i``, in dB.
    :param filter_: Contact filter ``A_3,i``, in dB.
    :return: ``L_R,TOT,i``, in dB.
    :raises ValueError: If the spectra are not 24 bands each.
    """
    return _energy_sum(
        _spectrum(rail, "rail"), _spectrum(wheel, "wheel")
    ) + _spectrum(filter_, "filter_")


def impact_roughness(single: Any, joint_density: float) -> NDArray[np.float64]:
    """Impact roughness ``L_R,IMPACT,i`` of (2.3.12), in dB.

    ``L_R,IMPACT,i = L_R,IMPACT-SINGLE,i + 10 lg(n_l/0,01)``, so at the
    tabulated density of one joint per 100 m the table is returned verbatim.

    :param single: Single-impact roughness on the frequency grid, in dB.
    :param joint_density: Joint density ``n_l``, in m^-1.
    :return: ``L_R,IMPACT,i``, in dB.
    :raises ValueError: If the joint density is negative or not finite.
    """
    density = _finite(joint_density, "joint_density")
    if density < 0.0:
        raise ValueError("'joint_density' must be a non-negative number of m^-1.")
    spectrum = _spectrum(single, "single")
    if density == 0.0:
        return np.full_like(spectrum, -np.inf)
    return np.asarray(
        spectrum + 10.0 * np.log10(density / REFERENCE_JOINT_DENSITY),
        dtype=np.float64,
    )


def rolling_sound_power(
    roughness: Any, transfer: Any, axles: float
) -> NDArray[np.float64]:
    """One rolling-noise component of (2.3.8) to (2.3.10), in dB.

    ``L_W,0,i = L_R,TOT,i + L_H,i + 10 lg(N_a)``: the same addition serves the
    track, the wheel and the freight superstructure, each with its own transfer
    function. All three sit at source A.

    :param roughness: Total effective roughness ``L_R,TOT,i``, in dB.
    :param transfer: Transfer function ``L_H,i``, in dB per axle.
    :param axles: Number of axles per vehicle ``N_a``.
    :return: The component sound power, in dB.
    :raises ValueError: If ``axles`` is not a positive number.
    """
    n_a = _finite(axles, "axles")
    if n_a <= 0.0:
        raise ValueError("'axles' must be a positive number of axles per vehicle.")
    return np.asarray(
        _spectrum(roughness, "roughness")
        + _spectrum(transfer, "transfer")
        + 10.0 * np.log10(n_a),
        dtype=np.float64,
    )


def curve_squeal_excess(
    radius: float,
    *,
    tram: bool = False,
    turnout: bool = False,
    track_length: float = _SQUEAL_MINIMUM_TRACK_LENGTH,
) -> float:
    """Curve-squeal excess added to the rolling noise, in dB.

    The rule is the one (EU) 2021/1226 Annex point (4)(b) substituted for the
    2015 text: for trains, 8 dB at ``R <= 300 m`` and 5 dB at
    ``300 m < R <= 500 m`` over at least 50 m of curve, and 8 dB on switch
    turnouts with ``R <= 300 m`` whatever their length; for trams, 5 dB on
    curves and switch turnouts with ``R <= 200 m``. The excess applies at all
    frequencies.

    :param radius: Curve radius ``R``, in m.
    :param tram: ``True`` for a tram, which follows its own rule.
    :param turnout: ``True`` for a switch turnout, where the minimum curve
        length does not apply.
    :param track_length: Length of track along the curve ``l_track``, in m.
    :return: The excess, in dB (0,0 where no squeal is modelled).
    :raises ValueError: If the radius or the track length is not positive.
    """
    r = _finite(radius, "radius")
    if r <= 0.0:
        raise ValueError("'radius' must be a positive number of metres.")
    length = _finite(track_length, "track_length")
    if length < 0.0:
        raise ValueError("'track_length' must be a non-negative number of metres.")
    if tram:
        return _SQUEAL_TRAM if r <= _SQUEAL_TRAM_RADIUS else 0.0
    if turnout:
        return _SQUEAL_TRAIN_TIGHT if r <= _SQUEAL_TRAIN_TIGHT_RADIUS else 0.0
    if length < _SQUEAL_MINIMUM_TRACK_LENGTH:
        return 0.0
    if r <= _SQUEAL_TRAIN_TIGHT_RADIUS:
        return _SQUEAL_TRAIN_TIGHT
    if r <= _SQUEAL_TRAIN_MODERATE_RADIUS:
        return _SQUEAL_TRAIN_MODERATE
    return 0.0


def horizontal_directivity(
    phi: float, *, frequencies: Any = RAILWAY_THIRD_OCTAVE_BANDS
) -> NDArray[np.float64]:
    """Horizontal directivity ``dL_W,dir,hor,i`` of (2.3.15), in dB.

    ``10 lg(0,01 + 0,99 sin^2 phi)``: a dipole, identical in every band, equal
    to 0 dB broadside (``phi = 90 deg``) and to ``10 lg 0,01 = -20 dB`` along
    the track. The Directive offers it "by default" for rolling, impact,
    squeal, braking, fans and aerodynamic effects; since no other horizontal
    directivity is given and traction noise includes the fans, it is applied
    here to every source, as the Commission's reference module does.

    :param phi: Horizontal angle ``phi``, in degrees, measured from the
        direction of travel (Figure [2.3.b]).
    :param frequencies: Midband frequencies, used only for the array shape.
    :return: The correction, in dB, one value per band.
    :raises ValueError: If the angle is not finite.
    """
    angle = np.radians(_finite(phi, "phi"))
    value = 10.0 * np.log10(0.01 + 0.99 * np.sin(angle) ** 2)
    return np.full(len(np.asarray(frequencies, dtype=np.float64)), value)


def vertical_directivity(
    psi: float,
    *,
    frequencies: Any = RAILWAY_THIRD_OCTAVE_BANDS,
    height: int = 1,
    aerodynamic: bool = False,
    edition: DirectivityEdition = DirectivityEdition.CURRENT,
) -> NDArray[np.float64]:
    """Vertical directivity ``dL_W,dir,ver,i`` of (2.3.16) and (2.3.17), in dB.

    Source A (``height = 1``) follows (2.3.16), which (EU) 2021/1226 Annex
    point (4)(d) replaced: the absolute-value bars of the 2015 text are gone
    and the correction is identically zero for ``psi <= 0``. Source B
    (``height = 2``) follows (2.3.17) for the aerodynamic effect only,
    ``10 lg(cos^2 psi)`` for ``psi < 0``, and is omni-directional for every
    other source.

    :param psi: Vertical angle ``psi``, in degrees (Figure [2.3.b]).
    :param frequencies: Midband frequencies ``f_c,i``, in Hz.
    :param height: ``1`` for source A at 0,5 m, ``2`` for source B at 4,0 m.
    :param aerodynamic: ``True`` to select the aerodynamic source at
        ``height = 2``; ignored at ``height = 1``.
    :param edition: Which text of (2.3.16) to evaluate.
    :return: The correction, in dB, one value per band.
    :raises ValueError: If the angle is not finite or the height is not 1 or 2.
    """
    angle = np.radians(_finite(psi, "psi"))
    freqs = np.asarray(frequencies, dtype=np.float64)
    if height not in (1, 2):
        raise ValueError("'height' must be 1 (source A) or 2 (source B).")
    if height == 2:
        if aerodynamic and angle < 0.0:
            return np.full(len(freqs), 10.0 * np.log10(np.cos(angle) ** 2))
        return np.zeros(len(freqs))
    shape = (40.0 / 3.0) * (
        (2.0 / 3.0) * np.sin(2.0 * angle) - np.sin(angle)
    ) * np.log10((freqs + 600.0) / 200.0)
    if edition is DirectivityEdition.ORIGINAL_2015:
        return np.asarray(np.abs(shape), dtype=np.float64)
    return np.asarray(np.where(angle > 0.0, shape, 0.0), dtype=np.float64)


def octave_bands_from_third_octaves(levels: Any) -> NDArray[np.float64]:
    """Energy-sum a 24-band 1/3-octave spectrum into the eight octave bands.

    Annex II 2.3.2 requires the directional sound power to be derived in 1/3
    octave bands and then "expressed in octave bands by energetically adding
    each pertaining 1/3 octave band together into the corresponding octave
    band".

    :param levels: The 24 1/3-octave levels from 50 Hz to 10 kHz, in dB.
    :return: The eight octave levels from 63 Hz to 8 kHz, in dB.
    :raises ValueError: If the spectrum is not 24 bands.
    """
    spectrum = _spectrum(levels, "levels")
    energy = 10.0 ** (spectrum / 10.0)
    with np.errstate(divide="ignore"):
        return np.asarray(
            [10.0 * np.log10(np.sum(energy[a:b])) for a, b in _OCTAVE_SLICES],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Vehicle and track data, and the assembled source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollingStock:
    """The Appendix G data of one vehicle type, on its own wavelength grids.

    Every field is the spectrum the method needs, so a Member State substitutes
    its own database simply by building this object from its own tables rather
    than from the :mod:`~phonometry.environmental.cnossos_rail` look-ups.

    :ivar axles: Number of axles per vehicle ``N_a``.
    :ivar wheel_roughness: ``(wavelengths in mm, levels in dB)`` of Table G-1a.
    :ivar contact_filter: ``(wavelengths in mm, levels in dB)`` of Table G-2.
    :ivar wheel_transfer: ``L_H,VEH,i`` of Table G-3b, 24 values in dB.
    :ivar superstructure_transfer: ``L_H,VEH,SUP,i`` of Table G-3c for a freight
        wagon, or ``None`` for any other vehicle type.
    :ivar traction: ``(source A, source B)`` spectra of Table G-5, or ``None``
        for an unpowered vehicle.
    :ivar aerodynamic: ``(source A, source B)`` reference spectra of Table G-6
        at ``v_0``, or ``None`` to leave aerodynamic noise out.
    :ivar aerodynamic_alpha: Speed exponent of (2.3.13) and (2.3.14).
    :ivar tram: ``True`` for a tram or light metro, which uses the lower
        minimum speed and the tram squeal rule.
    """

    axles: int
    wheel_roughness: tuple[Any, Any]
    contact_filter: tuple[Any, Any]
    wheel_transfer: Any
    superstructure_transfer: Any | None = None
    traction: tuple[Any, Any] | None = None
    aerodynamic: tuple[Any, Any] | None = None
    aerodynamic_alpha: float = _AERODYNAMIC_ALPHA
    tram: bool = False


@dataclass(frozen=True)
class RailwayVehicle:
    """One vehicle of the traffic on a track section.

    :ivar stock: The :class:`RollingStock` data of the vehicle type.
    :ivar flow_rate: Average number of vehicles per hour ``Q``.
    :ivar speed: Their speed ``v`` on the track section, in km/h.
    :ivar condition: The :class:`RunningCondition` ``c``.
    :ivar idling_time: Total idling time ``T_idle`` within ``T_ref``, in the
        same unit as ``T_ref``; used only when ``condition`` is idling.
    """

    stock: RollingStock
    flow_rate: float = 0.0
    speed: float = 0.0
    condition: RunningCondition = RunningCondition.CONSTANT
    idling_time: float = 0.0


@dataclass(frozen=True)
class RailwayTrack:
    """The Appendix G data of one track section.

    :ivar rail_roughness: ``(wavelengths in mm, levels in dB)`` of Table G-1b.
    :ivar track_transfer: ``L_H,TR,i`` of Table G-3a, 24 values in dB.
    :ivar impact_roughness: ``(wavelengths in mm, levels in dB)`` of Table G-4,
        or ``None`` where there is no joint, switch or crossing.
    :ivar joint_density: Joint density ``n_l``, in m^-1.
    :ivar bridge_transfer: ``L_H,bridge,i`` of Table G-7 where the section is on
        a bridge, or ``None``.
    :ivar squeal_excess: Curve-squeal excess in dB, from
        :func:`curve_squeal_excess`.
    :ivar length: Length ``L`` of the track section, in m; used only by the
        idling flow term (2.3.4).
    """

    rail_roughness: tuple[Any, Any]
    track_transfer: Any
    impact_roughness: tuple[Any, Any] | None = None
    joint_density: float = REFERENCE_JOINT_DENSITY
    bridge_transfer: Any | None = None
    squeal_excess: float = 0.0
    length: float = 100.0


@dataclass(frozen=True)
class RailwayEmissionResult:
    """Directional sound power per metre of a CNOSSOS-EU railway source.

    :ivar third_octave_frequencies: The 24 1/3-octave midband frequencies, Hz.
    :ivar frequencies: The eight octave midband frequencies, Hz.
    :ivar heights: Heights of the two equivalent source lines, in m.
    :ivar third_octave_line_power: ``L'_W,eq,line,i(psi,phi)`` per source height
        and 1/3-octave band, in dB re 1 pW per metre.
    :ivar line_power: The same, energy-summed into octave bands.
    :ivar total_line_power: The two heights summed, per octave band.
    :ivar components: The 1/3-octave sound power of each physical source before
        the flow term and the directivity, keyed by ``"rolling"``,
        ``"traction"``, ``"aerodynamic"`` and ``"bridge"``, each holding the
        ``(source A, source B)`` pair.
    """

    third_octave_frequencies: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    heights: tuple[float, float]
    third_octave_line_power: NDArray[np.float64]
    line_power: NDArray[np.float64]
    total_line_power: NDArray[np.float64]
    components: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = field(
        default_factory=dict
    )

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-metre line power of the two equivalent source heights."""
        from .._i18n import check_language
        from .._plot.environmental import plot_cnossos_rail_emission

        return plot_cnossos_rail_emission(
            self, ax=ax, language=check_language(language), **kwargs
        )


def _resample(
    table: tuple[Any, Any],
    speed: float,
    interpolation: RoughnessInterpolation,
) -> NDArray[np.float64]:
    wavelengths, levels = table
    return roughness_to_frequency(
        levels, wavelengths, speed, interpolation=interpolation
    )


def _rolling_and_bridge(
    vehicle: RailwayVehicle,
    track: RailwayTrack,
    roughness_speed: float,
    interpolation: RoughnessInterpolation,
    include_impact: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Rolling noise at source A and, where present, the bridge source."""
    stock = vehicle.stock
    total = total_effective_roughness(
        _resample(track.rail_roughness, roughness_speed, interpolation),
        _resample(stock.wheel_roughness, roughness_speed, interpolation),
        _resample(stock.contact_filter, roughness_speed, interpolation),
    )
    if include_impact and track.impact_roughness is not None:
        total = _energy_sum(
            total,
            impact_roughness(
                _resample(track.impact_roughness, roughness_speed, interpolation),
                track.joint_density,
            ),
        )  # (2.3.11)
    components = [
        rolling_sound_power(total, track.track_transfer, stock.axles),
        rolling_sound_power(total, stock.wheel_transfer, stock.axles),
    ]
    if stock.superstructure_transfer is not None:
        components.append(
            rolling_sound_power(total, stock.superstructure_transfer, stock.axles)
        )
    rolling = _energy_sum(*components) + _finite(track.squeal_excess, "squeal_excess")
    bridge = (
        rolling_sound_power(total, track.bridge_transfer, stock.axles)
        if track.bridge_transfer is not None
        else None
    )  # (2.3.18)
    return rolling, bridge


def railway_source_power(
    traffic: RailwayVehicle | list[RailwayVehicle] | tuple[RailwayVehicle, ...],
    track: RailwayTrack,
    *,
    psi: float = 0.0,
    phi: float = 90.0,
    reference_time: float = 12.0,
    minimum_speed: float | None = None,
    interpolation: RoughnessInterpolation = RoughnessInterpolation.PROPORTIONAL,
    directivity_edition: DirectivityEdition = DirectivityEdition.CURRENT,
) -> RailwayEmissionResult:
    """Directional sound power per metre of a railway source line (2.3.1).

    Assembles, for every vehicle of the traffic and both source heights, the
    rolling noise (2.3.8)-(2.3.11), the impact noise (2.3.12), the curve squeal,
    the traction noise, the aerodynamic noise (2.3.13)-(2.3.14) and the bridge
    noise (2.3.18); applies the directivity of (2.3.15)-(2.3.17); adds the flow
    term of (2.3.2) or (2.3.4); and energy-sums everything over the traffic.

    Rolling, impact, squeal and bridge noise sit at source A. Traction and
    aerodynamic noise are tabulated separately for the two heights, so their
    split between A and B is read from the data rather than assumed. Rolling
    noise is excluded while a vehicle idles, and impact noise is not modelled
    below the minimum speed nor while idling.

    :param traffic: One :class:`RailwayVehicle` or a sequence of them.
    :param track: The :class:`RailwayTrack` of the section.
    :param psi: Vertical angle ``psi`` to the receiver, in degrees.
    :param phi: Horizontal angle ``phi`` to the receiver, in degrees; the
        default 90 deg is broadside, where the dipole correction is 0 dB.
    :param reference_time: Reference period ``T_ref`` of (2.3.4), in the same
        unit as ``idling_time``.
    :param minimum_speed: Speed floor used to determine the total effective
        roughness, in km/h; ``None`` (the default) selects 50 km/h, or 30 km/h
        for a tram. Pass ``0`` to switch the floor off, which also switches off
        the exclusion of impact noise below it.
    :param interpolation: The :class:`RoughnessInterpolation` rule.
    :param directivity_edition: Which text of (2.3.16) to evaluate.
    :return: A :class:`RailwayEmissionResult`.
    :raises ValueError: If the traffic is empty or an input is invalid.
    """
    vehicles = [traffic] if isinstance(traffic, RailwayVehicle) else list(traffic)
    if not vehicles:
        raise ValueError("'traffic' must carry at least one vehicle.")
    if _finite(reference_time, "reference_time") <= 0.0:
        raise ValueError("'reference_time' must be a positive period.")
    if _finite(track.length, "length") <= 0.0:
        raise ValueError("'length' must be a positive number of metres.")

    n_bands = len(RAILWAY_THIRD_OCTAVE_BANDS)
    per_height: list[list[NDArray[np.float64]]] = [[], []]
    components: dict[str, list[list[NDArray[np.float64]]]] = {
        name: [[], []] for name in ("rolling", "traction", "aerodynamic", "bridge")
    }
    hor = horizontal_directivity(phi)
    ver = [
        vertical_directivity(psi, height=1, edition=directivity_edition),
        vertical_directivity(psi, height=2),
    ]
    ver_aero_b = vertical_directivity(psi, height=2, aerodynamic=True)

    for vehicle in vehicles:
        stock = vehicle.stock
        idling = vehicle.condition is RunningCondition.IDLING
        floor = (
            (TRAM_MINIMUM_SPEED if stock.tram else RAILWAY_MINIMUM_SPEED)
            if minimum_speed is None
            else _finite(minimum_speed, "minimum_speed")
        )
        sources: list[list[NDArray[np.float64]]] = [[], []]
        # (EU) 2021/1226 point (4)(c): bridge noise is at source A and
        # omni-directional, so it bypasses the directivity corrections.
        omni: list[NDArray[np.float64]] = []
        if not idling:
            speed = _speed(vehicle.speed)
            rolling, bridge = _rolling_and_bridge(
                vehicle, track, max(speed, floor), interpolation,
                include_impact=speed >= floor,
            )
            sources[0].append(rolling)
            components["rolling"][0].append(rolling)
            if bridge is not None:
                omni.append(bridge)
                components["bridge"][0].append(bridge)
            if stock.aerodynamic is not None and speed > AERODYNAMIC_THRESHOLD_SPEED:
                low, high = aerodynamic_sound_power(
                    speed, reference=stock.aerodynamic, alpha=stock.aerodynamic_alpha
                )
                sources[0].append(low)
                sources[1].append(high + ver_aero_b)
                components["aerodynamic"][0].append(low)
                components["aerodynamic"][1].append(high)
        if stock.traction is not None:
            for h, spectrum in enumerate(stock.traction):
                sources[h].append(np.asarray(spectrum, dtype=np.float64))
                components["traction"][h].append(
                    np.asarray(spectrum, dtype=np.float64)
                )
        if idling:
            idle = _finite(vehicle.idling_time, "idling_time")
            if idle < 0.0:
                raise ValueError("'idling_time' must be a non-negative period.")
            flow = (
                -np.inf if idle == 0.0
                else 10.0 * np.log10(idle / (reference_time * track.length))
            )  # (2.3.4)
        else:
            q = _finite(vehicle.flow_rate, "flow_rate")
            if q < 0.0:
                raise ValueError("'flow_rate' must be a non-negative number per hour.")
            flow = (
                -np.inf if q == 0.0
                else 10.0 * np.log10(q / (1000.0 * _speed(vehicle.speed)))
            )  # (2.3.2)
        for h in (0, 1):
            emitted = list(sources[h])
            directed = (
                [_energy_sum(*emitted) + hor + ver[h]] if emitted else []
            )  # (2.3.5)
            if h == 0:
                directed.extend(omni)
            if not directed:
                continue
            per_height[h].append(_energy_sum(*directed) + flow)

    third = np.asarray(
        [
            _energy_sum(*rows) if rows else np.full(n_bands, -np.inf)
            for rows in per_height
        ],
        dtype=np.float64,
    )  # (2.3.1)
    octaves = np.asarray(
        [octave_bands_from_third_octaves(row) for row in third], dtype=np.float64
    )
    return RailwayEmissionResult(
        third_octave_frequencies=np.asarray(
            RAILWAY_THIRD_OCTAVE_BANDS, dtype=np.float64
        ),
        frequencies=np.asarray(RAILWAY_OCTAVE_BANDS, dtype=np.float64),
        heights=RAILWAY_SOURCE_HEIGHTS,
        third_octave_line_power=third,
        line_power=octaves,
        total_line_power=_energy_sum(*octaves),
        components={
            name: (
                _energy_sum(*rows[0]) if rows[0] else np.full(n_bands, -np.inf),
                _energy_sum(*rows[1]) if rows[1] else np.full(n_bands, -np.inf),
            )
            for name, rows in components.items()
        },
    )
