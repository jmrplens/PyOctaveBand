#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
CNOSSOS-EU road traffic source emission (Directive 2002/49/EC Annex II, 2.2).

The common noise assessment methods of the European Union describe a road as an
incoherent **source line** of point sources 0.05 m above the pavement. Each
vehicle category ``m`` radiates a directional sound power per metre of line

.. math::

   L'_{W,eq,line,i,m} = L_{W,i,m}
   + 10 \log_{10}\left( \frac{Q_m}{1000 v_m} \right) \tag{2.2.1}

built from a **rolling** term (2.2.4) and a **propulsion** term (2.2.11), each
with its own corrections for road surface, air temperature, studded tyres, road
gradient and the acceleration or deceleration near a junction. This module
implements the whole of 2.2 together with the coefficient database of
Appendix F, in the eight octave bands from 63 Hz to 8 kHz.

Which text is implemented
-------------------------
Annex II was replaced by Commission Directive (EU) 2015/996, corrected by the
corrigendum of OJ L 5, 10.1.2018 and amended by Commission Delegated Directive
(EU) 2021/1226. The consolidated text (02002L0049) is what is implemented here:

* the equations of 2.2.2 to 2.2.6 come unchanged from 2015/996;
* the octave-band range of 2.2.1 is **63 Hz to 8 kHz** as corrected in 2018
  (the 2015 text said "125 Hz to 4 kHz", which never matched Appendix F);
* Tables F-1 and F-4 are the versions **replaced by 2021/1226**; Tables F-2 and
  F-3 are unchanged since 2015/996.

Each shipped table records in its comment which instrument it comes from. The
superseded 2015/996 coefficients are not shipped, but any coefficient set can be
supplied through :class:`RoadEmissionCoefficients` and
:class:`RoadSurfaceCoefficients`, which is also how a Member State substitutes
its own national database (the reason Appendix F is called a *database* and not
a table of constants).

Scope
-----
This is the **emission** stage only. It produces the source power that the
propagation stage consumes; splitting the source line into point sources is
explicitly outside the scope of the method (2.5.3), so
:func:`line_source_segment_power` is offered as plain arithmetic, clearly
labelled as such. The CNOSSOS propagation model itself is *not* ISO 9613-2; the
hand-off to :func:`~phonometry.environmental.outdoor_propagation.predicted_receiver_level`
mixes two methods and is a convenience, not a normative chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "CNOSSOS_A_WEIGHTING",
    "ROAD_COEFFICIENTS",
    "ROAD_OCTAVE_BANDS",
    "ROAD_REFERENCE_SPEED",
    "ROAD_REFERENCE_TEMPERATURE",
    "ROAD_SOURCE_HEIGHT",
    "JunctionType",
    "RoadEmissionCoefficients",
    "RoadEmissionResult",
    "RoadSurface",
    "RoadSurfaceCoefficients",
    "RoadTraffic",
    "RoadVehicleCategory",
    "line_source_segment_power",
    "road_propulsion_noise",
    "road_rolling_noise",
    "road_source_power",
    "road_surface_coefficients",
    "road_vehicle_sound_power",
]

#: Octave-band midband frequencies of the road source, in Hz (2.2.1 as
#: corrected by the corrigendum of OJ L 5, 10.1.2018: 63 Hz to 8 kHz).
ROAD_OCTAVE_BANDS: tuple[float, ...] = (
    63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0,
)
#: Reference speed ``v_ref`` of (2.2.4), (2.2.6), (2.2.11) and (2.2.19), km/h.
ROAD_REFERENCE_SPEED = 70.0
#: Reference air temperature ``tau_ref`` of (2.2.10), degrees Celsius.
ROAD_REFERENCE_TEMPERATURE = 20.0
#: Height of the equivalent point source above the road surface (2.2.1), m.
ROAD_SOURCE_HEIGHT = 0.05
#: Speed below which the sound power of a vehicle is frozen (2.2.1), km/h.
_ROAD_MINIMUM_SPEED = 20.0
#: Distance beyond which the junction correction vanishes (2.2.17), m.
_JUNCTION_RANGE = 100.0
#: Largest gradient magnitude the gradient correction accounts for (2.2.13), %.
_MAX_GRADIENT = 12.0

#: Octave-band A-weighting ``AWC_f,i`` prescribed by 2.5.5 as amended by
#: Commission Delegated Directive (EU) 2021/1226 Annex point (8)(b), in dB.
#: Printed explicitly by the Directive, so it is used verbatim rather than
#: recomputed from IEC 61672-1.
CNOSSOS_A_WEIGHTING: tuple[float, ...] = (
    -26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1,
)


class RoadVehicleCategory(Enum):
    """Vehicle categories of Table [2.2.a].

    Categories 1 to 4 are mandatory; the "open" category 5 has no coefficients
    in Appendix F and is therefore not modelled.
    """

    #: Light motor vehicles: passenger cars, delivery vans <= 3,5 t, SUVs, MPVs.
    LIGHT = "1"
    #: Medium heavy vehicles: delivery vans > 3,5 t, buses, motorhomes, two axles.
    MEDIUM_HEAVY = "2"
    #: Heavy vehicles: heavy duty vehicles, touring cars, buses, three axles or more.
    HEAVY = "3"
    #: Powered two-wheelers: two-, three- and four-wheel mopeds.
    MOPEDS = "4a"
    #: Powered two-wheelers: motorcycles, tricycles and quadricycles.
    MOTORCYCLES = "4b"


class JunctionType(Enum):
    """Junction types ``k`` of Table F-3 for the speed-variation correction."""

    #: No junction within range: the correction is identically zero.
    NONE = 0
    #: ``k = 1``, a crossing with traffic lights.
    CROSSING = 1
    #: ``k = 2``, a roundabout.
    ROUNDABOUT = 2


class RoadSurface(Enum):
    """Road surfaces tabulated in Table F-4 (as replaced by 2021/1226)."""

    REFERENCE = "reference road surface"
    ONE_LAYER_ZOAB = "1-layer ZOAB"
    TWO_LAYER_ZOAB = "2-layer ZOAB"
    TWO_LAYER_ZOAB_FINE = "2-layer ZOAB (fine)"
    SMA_NL5 = "SMA-NL5"
    SMA_NL8 = "SMA-NL8"
    BRUSHED_DOWN_CONCRETE = "brushed down concrete"
    OPTIMISED_BRUSHED_DOWN_CONCRETE = "optimised brushed down concrete"
    FINE_BROOMED_CONCRETE = "fine broomed concrete"
    WORKED_SURFACE = "worked surface"
    HARD_ELEMENTS_HERRINGBONE = "hard elements in herringbone"
    HARD_ELEMENTS_NOT_HERRINGBONE = "hard elements not in herringbone"
    QUIET_HARD_ELEMENTS = "quiet hard elements"
    THIN_LAYER_A = "thin layer A"
    THIN_LAYER_B = "thin layer B"


# ---------------------------------------------------------------------------
# Table F-1 - rolling and propulsion coefficients, as replaced by Commission
# Delegated Directive (EU) 2021/1226 Annex point (19)(a). Rows are
# (A_R, B_R, A_P, B_P) over the eight octave bands 63 Hz .. 8 kHz, in dB.
# ---------------------------------------------------------------------------
_TABLE_F1: dict[str, tuple[tuple[float, ...], ...]] = {
    "1": (
        (83.1, 89.2, 87.7, 93.1, 100.1, 96.7, 86.8, 76.2),
        (30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0),
        (97.9, 92.5, 90.7, 87.2, 84.7, 88.0, 84.4, 77.1),
        (-1.3, 7.2, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0),
    ),
    "2": (
        (88.7, 93.2, 95.7, 100.9, 101.7, 95.1, 87.8, 83.6),
        (30.0, 35.8, 32.6, 23.8, 30.1, 36.2, 38.3, 40.1),
        (105.5, 100.2, 100.5, 98.7, 101.0, 97.8, 91.2, 85.0),
        (-1.9, 4.7, 6.4, 6.5, 6.5, 6.5, 6.5, 6.5),
    ),
    "3": (
        (91.7, 96.2, 98.2, 104.9, 105.1, 98.5, 91.1, 85.6),
        (30.0, 33.5, 31.3, 25.4, 31.8, 37.1, 38.6, 40.6),
        (108.8, 104.2, 103.5, 102.9, 102.6, 98.5, 93.8, 87.5),
        (0.0, 3.0, 4.6, 5.0, 5.0, 5.0, 5.0, 5.0),
    ),
    "4a": (
        (0.0,) * 8,
        (0.0,) * 8,
        (93.0, 93.0, 93.5, 95.3, 97.2, 100.4, 95.8, 90.9),
        (4.2, 7.4, 9.8, 11.6, 15.7, 18.9, 20.3, 20.6),
    ),
    "4b": (
        (0.0,) * 8,
        (0.0,) * 8,
        (99.9, 101.9, 96.7, 94.4, 95.2, 94.7, 92.1, 88.6),
        (3.2, 5.9, 11.9, 11.6, 11.5, 12.6, 11.1, 12.0),
    ),
}

# ---------------------------------------------------------------------------
# Table F-2 - studded-tyre coefficients a_i and b_i (category 1 only),
# unchanged since Commission Directive (EU) 2015/996.
# ---------------------------------------------------------------------------
_TABLE_F2_A: tuple[float, ...] = (0.0, 0.0, 0.0, 2.6, 2.9, 1.5, 2.3, 9.2)
_TABLE_F2_B: tuple[float, ...] = (0.0, 0.0, 0.0, -3.1, -6.4, -14.0, -22.4, -11.4)

# ---------------------------------------------------------------------------
# Table F-3 - acceleration and deceleration coefficients C_R,m,k and C_P,m,k,
# unchanged since Commission Directive (EU) 2015/996.
# Values are ((C_R, C_P) for k = 1 crossing, (C_R, C_P) for k = 2 roundabout).
# ---------------------------------------------------------------------------
_TABLE_F3: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "1": ((-4.5, 5.5), (-4.4, 3.1)),
    "2": ((-4.0, 9.0), (-2.3, 6.7)),
    "3": ((-4.0, 9.0), (-2.3, 6.7)),
    "4a": ((0.0, 0.0), (0.0, 0.0)),
    "4b": ((0.0, 0.0), (0.0, 0.0)),
}

#: Temperature coefficients ``K_m`` of (2.2.10), in dB per degree Celsius:
#: 0,08 for light vehicles, 0,04 for the two heavy categories, and no
#: temperature dependence for the powered two-wheelers, which have no rolling
#: noise at all.
_TEMPERATURE_K: dict[str, float] = {
    "1": 0.08, "2": 0.04, "3": 0.04, "4a": 0.0, "4b": 0.0,
}

# ---------------------------------------------------------------------------
# Table F-4 - road-surface coefficients alpha_i,m and beta_m, as replaced by
# Commission Delegated Directive (EU) 2021/1226 Annex point (19)(b), which
# merged the former 4a and 4b rows into a single 4a/4b row. Each entry is
# (v_min, v_max, {category: (alpha over the eight bands, beta)}), speeds in
# km/h and coefficients in dB (beta is dimensionless).
# ---------------------------------------------------------------------------
_ZERO_8: tuple[float, ...] = (0.0,) * 8


def _surface_row(
    v_min: float | None,
    v_max: float | None,
    light: tuple[tuple[float, ...], float],
    medium: tuple[tuple[float, ...], float],
    heavy: tuple[tuple[float, ...], float],
    two_wheelers: tuple[tuple[float, ...], float] = (_ZERO_8, 0.0),
) -> tuple[float | None, float | None, dict[str, tuple[tuple[float, ...], float]]]:
    """Assemble one Table F-4 row (the 4a/4b row is shared by both categories)."""
    return v_min, v_max, {
        "1": light, "2": medium, "3": heavy, "4a": two_wheelers, "4b": two_wheelers,
    }


_TABLE_F4: dict[
    str, tuple[float | None, float | None, dict[str, tuple[tuple[float, ...], float]]]
] = {
    RoadSurface.REFERENCE.value: _surface_row(
        None, None, (_ZERO_8, 0.0), (_ZERO_8, 0.0), (_ZERO_8, 0.0),
    ),
    RoadSurface.ONE_LAYER_ZOAB.value: _surface_row(
        50.0, 130.0,
        ((0.0, 5.4, 4.3, 4.2, -1.0, -3.2, -2.6, 0.8), -6.5),
        ((7.9, 4.3, 5.3, -0.4, -5.2, -4.6, -3.0, -1.4), 0.2),
        ((9.3, 5.0, 5.5, -0.4, -5.2, -4.6, -3.0, -1.4), 0.2),
    ),
    RoadSurface.TWO_LAYER_ZOAB.value: _surface_row(
        50.0, 130.0,
        ((1.6, 4.0, 0.3, -3.0, -4.0, -6.2, -4.8, -2.0), -3.0),
        ((7.3, 2.0, -0.3, -5.2, -6.1, -6.0, -4.4, -3.5), 4.7),
        ((8.3, 2.2, -0.4, -5.2, -6.2, -6.1, -4.5, -3.5), 4.7),
    ),
    RoadSurface.TWO_LAYER_ZOAB_FINE.value: _surface_row(
        80.0, 130.0,
        ((-1.0, 3.0, -1.5, -5.3, -6.3, -8.5, -5.3, -2.4), -0.1),
        ((7.9, 0.1, -1.9, -5.9, -6.1, -6.8, -4.9, -3.8), -0.8),
        ((9.4, 0.2, -1.9, -5.9, -6.1, -6.7, -4.8, -3.8), -0.9),
    ),
    RoadSurface.SMA_NL5.value: _surface_row(
        40.0, 80.0,
        ((10.3, -0.9, 0.9, 1.8, -1.8, -2.7, -2.0, -1.3), -1.6),
        (_ZERO_8, 0.0), (_ZERO_8, 0.0),
    ),
    RoadSurface.SMA_NL8.value: _surface_row(
        40.0, 80.0,
        ((6.0, 0.3, 0.3, 0.0, -0.6, -1.2, -0.7, -0.7), -1.4),
        (_ZERO_8, 0.0), (_ZERO_8, 0.0),
    ),
    RoadSurface.BRUSHED_DOWN_CONCRETE.value: _surface_row(
        70.0, 120.0,
        ((8.2, -0.4, 2.8, 2.7, 2.5, 0.8, -0.3, -0.1), 1.4),
        ((0.3, 4.5, 2.5, -0.2, -0.1, -0.5, -0.9, -0.8), 5.0),
        ((0.2, 5.3, 2.5, -0.2, -0.1, -0.6, -1.0, -0.9), 5.5),
    ),
    RoadSurface.OPTIMISED_BRUSHED_DOWN_CONCRETE.value: _surface_row(
        70.0, 80.0,
        ((-0.2, -0.7, 1.4, 1.2, 1.1, -1.6, -2.0, -1.8), 1.0),
        ((-0.7, 3.0, -2.0, -1.4, -1.8, -2.7, -2.0, -1.9), -6.6),
        ((-0.5, 4.2, -1.9, -1.3, -1.7, -2.5, -1.8, -1.8), -6.6),
    ),
    RoadSurface.FINE_BROOMED_CONCRETE.value: _surface_row(
        70.0, 120.0,
        ((8.0, -0.7, 4.8, 2.2, 1.2, 2.6, 1.5, -0.6), 7.6),
        ((0.2, 8.6, 7.1, 3.2, 3.6, 3.1, 0.7, 0.1), 3.2),
        ((0.1, 9.8, 7.4, 3.2, 3.1, 2.4, 0.4, 0.0), 2.0),
    ),
    RoadSurface.WORKED_SURFACE.value: _surface_row(
        50.0, 130.0,
        ((8.3, 2.3, 5.1, 4.8, 4.1, 0.1, -1.0, -0.8), -0.3),
        ((0.1, 6.3, 5.8, 1.8, -0.6, -2.0, -1.8, -1.6), 1.7),
        ((0.0, 7.4, 6.2, 1.8, -0.7, -2.1, -1.9, -1.7), 1.4),
    ),
    RoadSurface.HARD_ELEMENTS_HERRINGBONE.value: _surface_row(
        30.0, 60.0,
        ((27.0, 16.2, 14.7, 6.1, 3.0, -1.0, 1.2, 4.5), 2.5),
        ((29.5, 20.0, 17.6, 8.0, 6.2, -1.0, 3.1, 5.2), 2.5),
        ((29.4, 21.2, 18.2, 8.4, 5.6, -1.0, 3.0, 5.8), 2.5),
    ),
    RoadSurface.HARD_ELEMENTS_NOT_HERRINGBONE.value: _surface_row(
        30.0, 60.0,
        ((31.4, 19.7, 16.8, 8.4, 7.2, 3.3, 7.8, 9.1), 2.9),
        ((34.0, 23.6, 19.8, 10.5, 11.7, 8.2, 12.2, 10.0), 2.9),
        ((33.8, 24.7, 20.4, 10.9, 10.9, 6.8, 12.0, 10.8), 2.9),
    ),
    RoadSurface.QUIET_HARD_ELEMENTS.value: _surface_row(
        30.0, 60.0,
        ((26.8, 13.7, 11.9, 3.9, -1.8, -5.8, -2.7, 0.2), -1.7),
        ((9.2, 5.7, 4.8, 2.3, 4.4, 5.1, 5.4, 0.9), 0.0),
        ((9.1, 6.6, 5.2, 2.6, 3.9, 3.9, 5.2, 1.1), 0.0),
    ),
    RoadSurface.THIN_LAYER_A.value: _surface_row(
        40.0, 130.0,
        ((10.4, 0.7, -0.6, -1.2, -3.0, -4.8, -3.4, -1.4), -2.9),
        ((13.8, 5.4, 3.9, -0.4, -1.8, -2.1, -0.7, -0.2), 0.5),
        ((14.1, 6.1, 4.1, -0.4, -1.8, -2.1, -0.7, -0.2), 0.3),
    ),
    RoadSurface.THIN_LAYER_B.value: _surface_row(
        40.0, 130.0,
        ((6.8, -1.2, -1.2, -0.3, -4.9, -7.0, -4.8, -3.2), -1.8),
        ((13.8, 5.4, 3.9, -0.4, -1.8, -2.1, -0.7, -0.2), 0.5),
        ((14.1, 6.1, 4.1, -0.4, -1.8, -2.1, -0.7, -0.2), 0.3),
    ),
}


@dataclass(frozen=True)
class RoadEmissionCoefficients:
    """Appendix F coefficient database for the road source.

    Holds Tables F-1, F-2 and F-3 together with the temperature coefficients
    ``K_m`` of (2.2.10). The default instance,
    :data:`ROAD_COEFFICIENTS`, carries the tables of the consolidated
    Directive; supplying another instance is how a Member State substitutes a
    national database, and how the superseded 2015/996 coefficients can be
    reproduced for comparison with pre-2021 studies.

    :ivar rolling_a: ``A_R,i,m`` per category, eight octave bands, in dB.
    :ivar rolling_b: ``B_R,i,m`` per category, eight octave bands, in dB.
    :ivar propulsion_a: ``A_P,i,m`` per category, eight octave bands, in dB.
    :ivar propulsion_b: ``B_P,i,m`` per category, eight octave bands, in dB.
    :ivar studded_a: ``a_i`` of Table F-2 (category 1 only), in dB.
    :ivar studded_b: ``b_i`` of Table F-2 (category 1 only), in dB.
    :ivar junction_c: ``(C_R, C_P)`` of Table F-3 per category and junction
        type, in dB.
    :ivar temperature_k: ``K_m`` of (2.2.10), in dB per degree Celsius.
    """

    rolling_a: dict[str, tuple[float, ...]]
    rolling_b: dict[str, tuple[float, ...]]
    propulsion_a: dict[str, tuple[float, ...]]
    propulsion_b: dict[str, tuple[float, ...]]
    studded_a: tuple[float, ...]
    studded_b: tuple[float, ...]
    junction_c: dict[str, tuple[tuple[float, float], tuple[float, float]]]
    temperature_k: dict[str, float]


#: The Appendix F database of the consolidated Directive: Tables F-1 and F-4 as
#: replaced by (EU) 2021/1226, Tables F-2 and F-3 as published in (EU) 2015/996.
ROAD_COEFFICIENTS = RoadEmissionCoefficients(
    rolling_a={k: v[0] for k, v in _TABLE_F1.items()},
    rolling_b={k: v[1] for k, v in _TABLE_F1.items()},
    propulsion_a={k: v[2] for k, v in _TABLE_F1.items()},
    propulsion_b={k: v[3] for k, v in _TABLE_F1.items()},
    studded_a=_TABLE_F2_A,
    studded_b=_TABLE_F2_B,
    junction_c=_TABLE_F3,
    temperature_k=_TEMPERATURE_K,
)


@dataclass(frozen=True)
class RoadSurfaceCoefficients:
    """One row of Table F-4: the acoustic signature of a road surface.

    :ivar name: The surface description as printed in Table F-4.
    :ivar alpha: ``alpha_i,m`` per category, eight octave bands, in dB.
    :ivar beta: ``beta_m`` per category (dimensionless speed coefficient).
    :ivar speed_range: ``(v_min, v_max)`` over which the row is declared valid,
        in km/h, or ``None`` for the reference surface, which carries no range.
        Carried for the reader, not enforced: Table F-4 prints these columns
        without attaching a consequence to them, so refusing a speed outside
        the range would be stricter than the method itself.
    """

    name: str
    alpha: dict[str, tuple[float, ...]]
    beta: dict[str, float]
    speed_range: tuple[float, float] | None


def road_surface_coefficients(surface: RoadSurface | str) -> RoadSurfaceCoefficients:
    """Look up a Table F-4 row.

    :param surface: A :class:`RoadSurface` member or its description string.
    :return: The :class:`RoadSurfaceCoefficients` of that surface.
    :raises ValueError: If the surface is not tabulated in Table F-4.
    """
    key = surface.value if isinstance(surface, RoadSurface) else str(surface)
    row = _TABLE_F4.get(key)
    if row is None:
        raise ValueError(
            f"Unknown road surface {key!r}; Table F-4 tabulates: "
            + ", ".join(sorted(_TABLE_F4))
        )
    v_min, v_max, per_category = row
    return RoadSurfaceCoefficients(
        name=key,
        alpha={k: v[0] for k, v in per_category.items()},
        beta={k: v[1] for k, v in per_category.items()},
        speed_range=None if v_min is None or v_max is None else (v_min, v_max),
    )


@dataclass(frozen=True)
class RoadTraffic:
    """The traffic of one vehicle category on a source line.

    :ivar category: The :class:`RoadVehicleCategory` of the flow.
    :ivar flow_rate: Hourly flow ``Q_m``, in vehicles per hour.
    :ivar speed: Average speed ``v_m``, in km/h. Sound powers are frozen at
        20 km/h below that speed (2.2.1); the flow term still uses the true
        speed.
    :ivar studded_fraction: ``Q_stud,ratio`` of (2.2.7), the fraction of the
        light-vehicle flow fitted with studded tyres over the studded period.
        Ignored for every category other than category 1 (2.2.9).
    """

    category: RoadVehicleCategory
    flow_rate: float
    speed: float
    studded_fraction: float = 0.0


def _finite(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"'{name}' must be a finite number.")
    return scalar


def _bands(values: tuple[float, ...]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def _combine_rolling_and_propulsion(
    key: str, rolling: np.ndarray, propulsion: np.ndarray
) -> np.ndarray:
    """(2.2.3): the energy sum of the two source terms, in dB.

    Categories 4a and 4b carry no rolling term, so their total is the
    propulsion power alone. The rule lives here rather than at each call site
    because a copy of it that no test reaches is a copy that can drift.
    """
    if key in ("4a", "4b"):
        return np.asarray(propulsion, dtype=np.float64)
    return np.asarray(
        10.0 * np.log10(10.0 ** (rolling / 10.0) + 10.0 ** (propulsion / 10.0)),
        dtype=np.float64,
    )


def _category_key(category: RoadVehicleCategory | str) -> str:
    key = category.value if isinstance(category, RoadVehicleCategory) else str(category)
    if key not in _TABLE_F1:
        raise ValueError(
            f"Unknown vehicle category {key!r}; Table [2.2.a] categories with "
            "coefficients in Appendix F are: " + ", ".join(_TABLE_F1)
        )
    return key


def _require_category(key: str, coefficients: RoadEmissionCoefficients) -> None:
    """Check that a substituted Appendix F database covers the category.

    :func:`_category_key` only checks that the category exists in Table
    [2.2.a]; a national database supplied through ``coefficients`` may cover
    fewer categories, and reaching for a missing row must fail as an invalid
    input rather than as a bare lookup error.
    """
    missing = [
        name
        for name, table in (
            ("rolling_a", coefficients.rolling_a),
            ("rolling_b", coefficients.rolling_b),
            ("propulsion_a", coefficients.propulsion_a),
            ("propulsion_b", coefficients.propulsion_b),
            ("junction_c", coefficients.junction_c),
            ("temperature_k", coefficients.temperature_k),
        )
        if key not in table
    ]
    if missing:
        raise ValueError(
            f"The supplied coefficient database has no category {key!r} in "
            + ", ".join(missing)
            + "; a substituted Appendix F database must cover every category "
            "of the modelled traffic."
        )


def _surface_of(
    surface: RoadSurface | str | RoadSurfaceCoefficients, key: str
) -> RoadSurfaceCoefficients:
    """Resolve a surface and check that it covers the vehicle category.

    A substituted Table F-4 row may carry fewer categories than the published
    one, and reaching for a missing one has to fail as an invalid input.
    """
    row = surface if isinstance(surface, RoadSurfaceCoefficients) else (
        road_surface_coefficients(surface)
    )
    if key not in row.alpha or key not in row.beta:
        raise ValueError(
            f"The road surface {row.name!r} has no coefficients for category "
            f"{key!r}; a substituted Table F-4 row must cover every category "
            "of the modelled traffic."
        )
    return row


def _junction_factor(distance: float | None, junction: JunctionType) -> float:
    """``Max(1 - |x|/100 ; 0)`` of (2.2.17)/(2.2.18); zero without a junction."""
    if junction is JunctionType.NONE or distance is None:
        return 0.0
    return max(1.0 - abs(_finite(distance, "junction_distance")) / _JUNCTION_RANGE, 0.0)


def _studded_correction(
    key: str,
    speed: float,
    studded_fraction: float,
    studded_months: float,
    coefficients: RoadEmissionCoefficients,
) -> NDArray[np.float64]:
    """``dL_studdedtyres,i,m`` of (2.2.6) to (2.2.9), in dB per octave band."""
    if not 0.0 <= studded_fraction <= 1.0:
        raise ValueError("'studded_fraction' must be a fraction in [0, 1].")
    if not 0.0 <= studded_months <= 12.0:
        raise ValueError("'studded_months' must be a number of months in [0, 12].")
    if key != "1" or studded_months <= 0.0 or studded_fraction <= 0.0:
        return np.zeros(len(ROAD_OCTAVE_BANDS))
    # (2.2.6): the speed dependence saturates below 50 km/h and above 90 km/h.
    clipped = min(max(speed, 50.0), 90.0)
    d_stud = _bands(coefficients.studded_a) + _bands(coefficients.studded_b) * np.log10(
        clipped / ROAD_REFERENCE_SPEED
    )
    p_s = studded_fraction * studded_months / 12.0  # (2.2.7)
    return np.asarray(
        10.0 * np.log10((1.0 - p_s) + p_s * 10.0 ** (d_stud / 10.0)), dtype=np.float64
    )  # (2.2.8)


def _gradient_correction(key: str, slope: float, speed: float) -> float:
    """``dL_WP,grad,i,m`` of (2.2.13) to (2.2.16), in dB (equal in every band).

    Note the deliberate asymmetry of the published formulae: the downhill
    branch of category 1 carries no speed factor, while categories 2 and 3 do,
    and their uphill branches subtract no constant from the slope.
    """
    if key in ("4a", "4b"):
        return 0.0
    if key == "1":
        if slope < -6.0:
            return (min(_MAX_GRADIENT, -slope) - 6.0) / 1.0
        if slope > 2.0:
            return (min(_MAX_GRADIENT, slope) - 2.0) / 1.5 * speed / 100.0
        return 0.0
    if key == "2":
        if slope < -4.0:
            return (min(_MAX_GRADIENT, -slope) - 4.0) / 0.7 * (speed - 20.0) / 100.0
        if slope > 0.0:
            return min(_MAX_GRADIENT, slope) / 1.0 * speed / 100.0
        return 0.0
    if slope < -4.0:
        return (min(_MAX_GRADIENT, -slope) - 4.0) / 0.5 * (speed - 10.0) / 100.0
    if slope > 0.0:
        return min(_MAX_GRADIENT, slope) / 0.8 * speed / 100.0
    return 0.0


def road_rolling_noise(
    category: RoadVehicleCategory | str,
    speed: float,
    *,
    surface: RoadSurface | str | RoadSurfaceCoefficients = RoadSurface.REFERENCE,
    temperature: float = ROAD_REFERENCE_TEMPERATURE,
    studded_fraction: float = 0.0,
    studded_months: float = 0.0,
    junction_distance: float | None = None,
    junction_type: JunctionType = JunctionType.NONE,
    coefficients: RoadEmissionCoefficients = ROAD_COEFFICIENTS,
) -> NDArray[np.float64]:
    r"""Rolling-noise sound power ``L_WR,i,m`` of one vehicle (2.2.4)/(2.2.5).

    :math:`L_{WR,i,m} = A_{R,i,m} + B_{R,i,m} \log_{10}(v_m/v_{ref}) +
    dL_{WR,i,m}` with the
    correction term ``dL_WR`` collecting the road surface (2.2.19), the studded
    tyres (2.2.8), the junction (2.2.17) and the air temperature (2.2.10).
    Categories 4a and 4b have no rolling noise: their Table F-1 rows are zero,
    and their total power is the propulsion term alone (2.2.3).

    :param category: Vehicle category (Table [2.2.a]).
    :param speed: Average speed ``v_m``, in km/h; values below 20 km/h are
        raised to 20 km/h (2.2.1).
    :param surface: Road surface, as a :class:`RoadSurface`, its description or
        an explicit :class:`RoadSurfaceCoefficients`.
    :param temperature: Air temperature ``tau``, in degrees Celsius.
    :param studded_fraction: ``Q_stud,ratio`` of (2.2.7).
    :param studded_months: ``T_s`` of (2.2.7), the months per year over which
        studded tyres are in use.
    :param junction_distance: Distance ``x`` to the junction, in m.
    :param junction_type: Junction type ``k`` (Table F-3).
    :param coefficients: The Appendix F database to use.
    :return: ``L_WR,i,m`` over the eight octave bands, in dB re 1 pW.
    :raises ValueError: If an input is invalid.
    """
    key = _category_key(category)
    _require_category(key, coefficients)
    v = max(_positive_speed(speed), _ROAD_MINIMUM_SPEED)
    row = _surface_of(surface, key)
    log_speed = np.log10(v / ROAD_REFERENCE_SPEED)

    base = _bands(coefficients.rolling_a[key]) + _bands(coefficients.rolling_b[key]) * log_speed
    # (2.2.19) road-surface effect on rolling noise.
    d_road = _bands(row.alpha[key]) + row.beta[key] * log_speed
    d_stud = _studded_correction(
        key, v, _finite(studded_fraction, "studded_fraction"),
        _finite(studded_months, "studded_months"), coefficients,
    )
    factor = _junction_factor(junction_distance, junction_type)
    d_acc = (
        coefficients.junction_c[key][junction_type.value - 1][0] * factor
        if junction_type is not JunctionType.NONE
        else 0.0
    )
    # (2.2.10) air-temperature correction, equal in every octave band.
    d_temp = coefficients.temperature_k[key] * (
        ROAD_REFERENCE_TEMPERATURE - _finite(temperature, "temperature")
    )
    return np.asarray(base + d_road + d_stud + d_acc + d_temp, dtype=np.float64)


def road_propulsion_noise(
    category: RoadVehicleCategory | str,
    speed: float,
    *,
    surface: RoadSurface | str | RoadSurfaceCoefficients = RoadSurface.REFERENCE,
    gradient: float = 0.0,
    junction_distance: float | None = None,
    junction_type: JunctionType = JunctionType.NONE,
    coefficients: RoadEmissionCoefficients = ROAD_COEFFICIENTS,
) -> NDArray[np.float64]:
    r"""Propulsion-noise sound power ``L_WP,i,m`` of one vehicle (2.2.11)/(2.2.12).

    :math:`L_{WP,i,m} = A_{P,i,m} + B_{P,i,m} (v_m - v_{ref})/v_{ref} +
    dL_{WP,i,m}` with
    ``dL_WP`` collecting the road surface (2.2.20), the road gradient
    (2.2.13)-(2.2.16) and the junction (2.2.18). Unlike rolling noise, the
    surface term is :math:`\min\{\alpha_{i,m}; 0\}`: an absorbing surface
    reduces
    propulsion noise, a noisy one does not increase it.

    :param category: Vehicle category (Table [2.2.a]).
    :param speed: Average speed ``v_m``, in km/h; values below 20 km/h are
        raised to 20 km/h (2.2.1).
    :param surface: Road surface, as a :class:`RoadSurface`, its description or
        an explicit :class:`RoadSurfaceCoefficients`.
    :param gradient: Road slope ``s``, in per cent, positive uphill in the
        direction of travel. For a bidirectional flow, split the flow in two
        and correct one half uphill and the other downhill.
    :param junction_distance: Distance ``x`` to the junction, in m.
    :param junction_type: Junction type ``k`` (Table F-3).
    :param coefficients: The Appendix F database to use.
    :return: ``L_WP,i,m`` over the eight octave bands, in dB re 1 pW.
    :raises ValueError: If an input is invalid.
    """
    key = _category_key(category)
    _require_category(key, coefficients)
    v = max(_positive_speed(speed), _ROAD_MINIMUM_SPEED)
    row = _surface_of(surface, key)

    base = _bands(coefficients.propulsion_a[key]) + _bands(
        coefficients.propulsion_b[key]
    ) * ((v - ROAD_REFERENCE_SPEED) / ROAD_REFERENCE_SPEED)
    # (2.2.20): only an absorbing surface (alpha < 0) affects propulsion noise.
    d_road = np.minimum(_bands(row.alpha[key]), 0.0)
    factor = _junction_factor(junction_distance, junction_type)
    d_acc = (
        coefficients.junction_c[key][junction_type.value - 1][1] * factor
        if junction_type is not JunctionType.NONE
        else 0.0
    )
    d_grad = _gradient_correction(key, _finite(gradient, "gradient"), v)
    return np.asarray(base + d_road + d_acc + d_grad, dtype=np.float64)


def _positive_speed(speed: float) -> float:
    v = _finite(speed, "speed")
    if v <= 0.0:
        raise ValueError("'speed' must be a positive number of km/h.")
    return v


def road_vehicle_sound_power(
    category: RoadVehicleCategory | str,
    speed: float,
    *,
    surface: RoadSurface | str | RoadSurfaceCoefficients = RoadSurface.REFERENCE,
    temperature: float = ROAD_REFERENCE_TEMPERATURE,
    gradient: float = 0.0,
    studded_fraction: float = 0.0,
    studded_months: float = 0.0,
    junction_distance: float | None = None,
    junction_type: JunctionType = JunctionType.NONE,
    coefficients: RoadEmissionCoefficients = ROAD_COEFFICIENTS,
) -> NDArray[np.float64]:
    """Total sound power ``L_W,i,m`` of a single vehicle (2.2.2)/(2.2.3).

    For categories 1, 2 and 3 the rolling and propulsion terms are summed
    energetically (2.2.2); for the powered two-wheelers of category 4 the
    propulsion term is the whole of it (2.2.3).

    :param category: Vehicle category (Table [2.2.a]).
    :param speed: Average speed ``v_m``, in km/h.
    :param surface: Road surface (Table F-4).
    :param temperature: Air temperature ``tau``, in degrees Celsius.
    :param gradient: Road slope ``s``, in per cent.
    :param studded_fraction: ``Q_stud,ratio`` of (2.2.7).
    :param studded_months: ``T_s`` of (2.2.7), in months.
    :param junction_distance: Distance ``x`` to the junction, in m.
    :param junction_type: Junction type ``k`` (Table F-3).
    :param coefficients: The Appendix F database to use.
    :return: ``L_W,i,m`` over the eight octave bands, in dB re 1 pW.
    :raises ValueError: If an input is invalid.
    """
    key = _category_key(category)
    propulsion = road_propulsion_noise(
        key, speed, surface=surface, gradient=gradient,
        junction_distance=junction_distance, junction_type=junction_type,
        coefficients=coefficients,
    )
    if key in ("4a", "4b"):
        return propulsion
    rolling = road_rolling_noise(
        key, speed, surface=surface, temperature=temperature,
        studded_fraction=studded_fraction, studded_months=studded_months,
        junction_distance=junction_distance, junction_type=junction_type,
        coefficients=coefficients,
    )
    return _combine_rolling_and_propulsion(key, rolling, propulsion)


@dataclass(frozen=True)
class RoadEmissionResult:
    """Directional sound power per metre of a CNOSSOS-EU road source line.

    :ivar frequencies: Octave-band midband frequencies, in Hz (63 Hz to 8 kHz).
    :ivar categories: The vehicle categories of the modelled flows, in the
        order of the rows of the per-category arrays.
    :ivar rolling: ``L_WR,i,m`` per category and band, in dB re 1 pW. The row
        of a powered two-wheeler is the zero row Table F-1 prints for it, and
        does **not** enter its sound power, which is the propulsion term alone
        (2.2.3).
    :ivar propulsion: ``L_WP,i,m`` per category and band, in dB re 1 pW.
    :ivar vehicle_power: ``L_W,i,m`` per category and band (2.2.2)/(2.2.3), in
        dB re 1 pW.
    :ivar line_power: ``L'_W,eq,line,i,m`` per category and band (2.2.1), in dB
        re 1 pW per metre.
    :ivar total_line_power: The energy sum of ``line_power`` over the
        categories, in dB re 1 pW per metre: the source strength of the line.
    :ivar source_height: Height of the equivalent point source above the road
        surface, in m (0,05 m, fixed by 2.2.1).
    """

    frequencies: NDArray[np.float64]
    categories: tuple[RoadVehicleCategory, ...]
    rolling: NDArray[np.float64]
    propulsion: NDArray[np.float64]
    vehicle_power: NDArray[np.float64]
    line_power: NDArray[np.float64]
    total_line_power: NDArray[np.float64]
    source_height: float = ROAD_SOURCE_HEIGHT

    @property
    def a_weighted_line_power(self) -> float:
        """A-weighted total line power, in dB(A) re 1 pW per metre.

        Uses the octave-band weighting ``AWC_f,i`` printed in 2.5.5 as amended
        by (EU) 2021/1226, i.e. :data:`CNOSSOS_A_WEIGHTING`.
        """
        weighted = self.total_line_power + np.asarray(CNOSSOS_A_WEIGHTING)
        return float(10.0 * np.log10(np.sum(10.0 ** (weighted / 10.0))))

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-metre line power spectrum and its category breakdown."""
        from .._i18n import check_language
        from .._plot.environmental import plot_cnossos_road_emission

        return plot_cnossos_road_emission(
            self, ax=ax, language=check_language(language), **kwargs
        )


def road_source_power(
    traffic: RoadTraffic | list[RoadTraffic] | tuple[RoadTraffic, ...],
    *,
    surface: RoadSurface | str | RoadSurfaceCoefficients = RoadSurface.REFERENCE,
    temperature: float = ROAD_REFERENCE_TEMPERATURE,
    gradient: float = 0.0,
    studded_months: float = 0.0,
    junction_distance: float | None = None,
    junction_type: JunctionType = JunctionType.NONE,
    coefficients: RoadEmissionCoefficients = ROAD_COEFFICIENTS,
) -> RoadEmissionResult:
    r"""Directional sound power per metre of a road source line (2.2.1).

    Evaluates
    :math:`L'_{W,eq,line,i,m} = L_{W,i,m} + 10 \log_{10}(Q_m/(1000 v_m))` for every
    category of the traffic mix and sums the categories energetically. The
    flow term uses the true average speed even where the sound power itself is
    frozen at 20 km/h (2.2.1).

    All corrections other than the per-category traffic are properties of the
    road segment, so they are passed once here: the surface, the air
    temperature, the gradient, the studded-tyre season and the junction.

    :param traffic: One :class:`RoadTraffic` or a sequence of them, at most one
        per vehicle category.
    :param surface: Road surface (Table F-4).
    :param temperature: Yearly average air temperature ``tau``, in degrees
        Celsius (the reference condition is 20 degC).
    :param gradient: Road slope ``s``, in per cent, positive uphill.
    :param studded_months: ``T_s`` of (2.2.7), the months per year over which
        studded tyres are in use.
    :param junction_distance: Distance ``x`` from the source to the nearest
        junction, in m; ``None`` (the default) means no junction in range.
    :param junction_type: Junction type ``k`` (Table F-3).
    :param coefficients: The Appendix F database to use.
    :return: A :class:`RoadEmissionResult`.
    :raises ValueError: If the traffic is empty, repeats a category, or carries
        an invalid flow rate or speed.
    """
    flows = [traffic] if isinstance(traffic, RoadTraffic) else list(traffic)
    if not flows:
        raise ValueError("'traffic' must carry at least one vehicle category.")
    keys = [_category_key(flow.category) for flow in flows]
    if len(set(keys)) != len(keys):
        raise ValueError("'traffic' must carry at most one flow per vehicle category.")

    rolling, propulsion, vehicle, line = [], [], [], []
    for flow, key in zip(flows, keys, strict=True):
        q = _finite(flow.flow_rate, "flow_rate")
        if q < 0.0:
            raise ValueError("'flow_rate' must be a non-negative number of vehicles/h.")
        v = _positive_speed(flow.speed)
        lwr = road_rolling_noise(
            key, v, surface=surface, temperature=temperature,
            studded_fraction=flow.studded_fraction, studded_months=studded_months,
            junction_distance=junction_distance, junction_type=junction_type,
            coefficients=coefficients,
        )
        lwp = road_propulsion_noise(
            key, v, surface=surface, gradient=gradient,
            junction_distance=junction_distance, junction_type=junction_type,
            coefficients=coefficients,
        )
        lw = _combine_rolling_and_propulsion(key, lwr, lwp)
        # (2.2.1): the flow term uses the true speed, never the 20 km/h floor.
        flow_term = -np.inf if q <= 0.0 else 10.0 * np.log10(q / (1000.0 * v))
        rolling.append(lwr)
        propulsion.append(lwp)
        vehicle.append(lw)
        line.append(lw + flow_term)

    line_power = np.asarray(line, dtype=np.float64)
    # An empty road is a legitimate input and its line power is -inf, so the
    # logarithm of the zero energy sum is silenced rather than warned about.
    with np.errstate(divide="ignore"):
        total = 10.0 * np.log10(np.sum(10.0 ** (line_power / 10.0), axis=0))
    return RoadEmissionResult(
        frequencies=np.asarray(ROAD_OCTAVE_BANDS, dtype=np.float64),
        categories=tuple(RoadVehicleCategory(key) for key in keys),
        rolling=np.asarray(rolling, dtype=np.float64),
        propulsion=np.asarray(propulsion, dtype=np.float64),
        vehicle_power=np.asarray(vehicle, dtype=np.float64),
        line_power=line_power,
        total_line_power=np.asarray(total, dtype=np.float64),
    )


def line_source_segment_power(
    line_power: NDArray[np.float64] | list[float] | float, length: float
) -> NDArray[np.float64]:
    r"""Sound power of the point source representing a segment of source line.

    :math:`L_{W,segment,i} = L'_{W,eq,line,i} + 10 \log_{10}(dL)`. This is
    arithmetic, not a
    normative rule: section 2.5.3 of Annex II states that how a line source is
    split into equivalent point sources "is outside the scope of the current
    methodology". Only the per-metre line power is defined by the method.

    :param line_power: Per-metre line power ``L'_W,eq,line,i``, in dB re 1 pW
        per metre.
    :param length: Length ``dL`` of the segment, in m.
    :return: The segment sound power, in dB re 1 pW.
    :raises ValueError: If ``length`` is not positive.
    """
    dl = _finite(length, "length")
    if dl <= 0.0:
        raise ValueError("'length' must be a positive number of metres.")
    return np.asarray(
        np.atleast_1d(np.asarray(line_power, dtype=np.float64)) + 10.0 * np.log10(dl),
        dtype=np.float64,
    )
