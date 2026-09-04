#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Quality assurance for ISO 9613-2 software, by its own test cases.

ISO 17534-1 asks software that implements an outdoor propagation method to
declare conformity against a published set of test cases; ISO/TR 17534-3
carries that set for ISO 9613-2. Nineteen scenarios are printed, each with its
input geometry and a step-by-step table of intermediate quantities, and the
document states the envelope a result has to stay inside to count as correct:
+/-0,05 dB per band and on the total.

That makes it an oracle of an unusual kind. Every other row of this report
checks one formula against one printed number; these check the whole chain of
Clause 7, composed the way the guideline composes it, against a table that was
produced to catch exactly the disagreements between implementations that a
single formula never exposes.

The seven cases here are the ones the document itself sets apart: "Test cases
T01 up to T07 can be solved by applying ISO 9613-2 exclusively" (6.1). T08 to
T19 add barriers, buildings and reflections, whose ray paths are built by the
additional recommendations of Clause 5 rather than by ISO 9613-2, and they are
not attempted.

Oracle: ISO/TR 17534-3:2015, 6.2.1 to 6.2.8, printed folios 6 to 15.
Method: ISO 9613-2:1996, Clause 7.
"""

from __future__ import annotations

import functools
import math

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, register

_QA = "Outdoor propagation quality assurance (ISO/TR 17534-3)"

#: Table 1: source and receiver, in metres. The heights are above the local
#: ground, which is what makes them serve T06 unchanged.
_SOURCE = (10.0, 10.0, 1.0)
_RECEIVER = (200.0, 50.0, 4.0)
#: Table 2: the source radiates 93 dB in every octave, so the shape of the
#: printed spectrum is the propagation and nothing else.
_LW = 93.0
#: Header row of every spectral table, Hz.
_BANDS = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
#: The A-weighting each table prints in its own row, dB. Taken from the page
#: rather than from the library, so these rows check the propagation.
_A_WEIGHTING = (-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1)
#: 6.2.1: the air the cases are calculated in.
_TEMPERATURE = 20.0
_HUMIDITY = 70.0
#: The envelope the document itself declares, in decibels: "The result values
#: in frequency bands and for the total level are considered to be correct if
#: the deviation does not exceed +/-0,05 dB."
_ENVELOPE = 0.05
#: Tables 3, 8 and 14 print two decimals of a length.
_LENGTH_TOLERANCE = 0.005

#: Ground projection of the path, m: Table 3, and the same in every case.
_DP = math.hypot(_RECEIVER[0] - _SOURCE[0], _RECEIVER[1] - _SOURCE[1])

#: Table 7 (T04) and Table 11 (T06): the path crosses three areas, and Tables 8
#: and 14 print the length of ground projection it spends in each.
_SEGMENTS = (40.88, 102.19, 51.10)
#: T04 runs from the least porous area to the most, T06 the other way about.
_G_T04 = (0.2, 0.5, 0.9)
_G_T06 = (0.9, 0.5, 0.2)

#: Table 13: with the ground rising 10 m under the receiver, the ray gains
#: 13 m from end to end, which is what lifts d3 clear of dp.
_T06_RISE = 13.0
#: Table 12 read along the path: flat as far as the contour at x = 120 m, then
#: climbing to the 10 m contour at x = 185 m and level to the receiver.
_T06_PROFILE_DISTANCES = (
    0.0,
    (120.0 - _SOURCE[0]) * _DP / (_RECEIVER[0] - _SOURCE[0]),
    (185.0 - _SOURCE[0]) * _DP / (_RECEIVER[0] - _SOURCE[0]),
    _DP,
)
_T06_PROFILE_HEIGHTS = (0.0, 0.0, 10.0, 10.0)


def _worst_band(printed: tuple[float, ...], computed: np.ndarray) -> Outcome:
    """Compare a computed spectrum with a printed one, band by band.

    The document grants its envelope to every band separately, so the one
    number that decides a case is the largest absolute deviation over the eight
    octaves. Comparing the sums instead would pass a spectrum with the right
    energy and the wrong shape, which is the disagreement between two
    implementations these tables exist to catch.
    """
    deviation = float(np.max(np.abs(np.asarray(printed, dtype=float) - computed)))
    return numeric(0.0, deviation, _ENVELOPE, unit="dB", places=4)


def _total(levels: np.ndarray, *, a_weighted: bool = False) -> float:
    """Energy sum of the octave bands, weighted or not, dB."""
    if a_weighted:
        levels = levels + np.array(_A_WEIGHTING)
    return 10.0 * math.log10(float(np.sum(10.0 ** (levels / 10.0))))


def _general_method(
    ground: ph.environment.GroundFactors, distance: float
) -> np.ndarray:
    """Receiver band levels by the general per-region ground method (7.3.1)."""
    return ph.environment.predicted_receiver_level(
        np.full(len(_BANDS), _LW),
        ph.environment.PropagationGeometry(
            distance, _SOURCE[2], _RECEIVER[2], projected_distance=_DP
        ),
        frequencies=np.array(_BANDS),
        ground=ground,
        atmosphere=ph.environment.AtmosphericConditions(
            temperature=_TEMPERATURE, relative_humidity=_HUMIDITY
        ),
    )


def _alternative_method(mean_height: float, distance: float) -> np.ndarray:
    """Receiver band levels by the alternative ground method (7.3.2).

    The guideline pairs Equation (10) with the solid-angle index of Equation
    (11), which is why ``DOmega`` appears as a row of its own in Tables 10 and
    18. The library leaves that pairing to the caller rather than wiring it
    into the general entry point, so it is made here.
    """
    bands = np.array(_BANDS)
    a_gr = ph.environment.ground_attenuation_alternative(distance, mean_height)
    d_omega = ph.environment.directivity_omega(_SOURCE[2], _RECEIVER[2], _DP)
    a_div = ph.environment.geometric_divergence(distance)
    a_atm = ph.environment.atmospheric_absorption(
        distance, bands, temperature=_TEMPERATURE, relative_humidity=_HUMIDITY
    )
    return _LW + d_omega - a_div - a_atm - a_gr


# --------------------------------------------------------------------------- #
# T01-T03: flat ground of one kind, from reflecting to porous
# --------------------------------------------------------------------------- #
#: Tables 4, 5 and 6, row "L": the band levels at the receiver, dB.
_T01_LEVELS = (39.90, 39.86, 39.70, 39.37, 38.95, 38.17, 35.47, 25.04)
_T02_LEVELS = (39.90, 36.17, 33.02, 33.20, 36.11, 36.33, 33.63, 23.20)
_T03_LEVELS = (39.90, 32.48, 26.33, 27.03, 33.27, 34.49, 31.79, 21.36)
#: The two summed columns of those rows: L and LA, dB.
_T01_TOTALS = (47.46, 44.29)
_T02_TOTALS = (44.61, 41.53)
_T03_TOTALS = (42.80, 39.14)

#: Table 3: the straight-line distance the divergence and absorption terms use.
_D3_FLAT = math.hypot(_DP, _RECEIVER[2] - _SOURCE[2])

_HOMOGENEOUS = (
    ("T01", 0.0, _T01_LEVELS, _T01_TOTALS),
    ("T02", 0.5, _T02_LEVELS, _T02_TOTALS),
    ("T03", 1.0, _T03_LEVELS, _T03_TOTALS),
)


def _homogeneous_levels(g: float) -> np.ndarray:
    """One of T01 to T03: the same ground factor in all three regions."""
    return _general_method(ph.environment.GroundFactors(g, g, g), _D3_FLAT)


def _chk_homogeneous_spectrum(g: float, printed: tuple[float, ...]) -> Outcome:
    return _worst_band(printed, _homogeneous_levels(g))


def _chk_homogeneous_total(g: float, printed: float, *, a_weighted: bool) -> Outcome:
    return numeric(
        printed,
        _total(_homogeneous_levels(g), a_weighted=a_weighted),
        _ENVELOPE,
        unit="dB",
        places=3,
    )


def _register_homogeneous() -> None:
    """Register the three flat-ground cases, three rows each."""
    for case, g, levels, (total, total_a) in _HOMOGENEOUS:
        standard = f"ISO/TR 17534-3:2015 {case}"
        register(_QA, standard, f"Receiver band levels over ground G = {g:g}, dB")(
            functools.partial(_chk_homogeneous_spectrum, g, levels)
        )
        register(_QA, standard, f"Receiver total level over ground G = {g:g}, dB")(
            functools.partial(_chk_homogeneous_total, g, total, a_weighted=False)
        )
        register(_QA, standard, f"Receiver A-weighted level over ground G = {g:g}, dB")(
            functools.partial(_chk_homogeneous_total, g, total_a, a_weighted=True)
        )


_register_homogeneous()


@register(_QA, "ISO/TR 17534-3:2015 Table 3", "Ground-projected path length dp, m")
def _chk_dp() -> Outcome:
    """The horizontal leg of every case, printed as 194,16 m."""
    return numeric(194.16, _DP, _LENGTH_TOLERANCE, unit="m", places=3)


@register(_QA, "ISO/TR 17534-3:2015 Table 3", "Straight-line path length d3, m")
def _chk_d3() -> Outcome:
    """Over flat ground the ray rises only the 3 m between source and receiver."""
    return numeric(194.19, _D3_FLAT, _LENGTH_TOLERANCE, unit="m", places=3)


@register(_QA, "ISO/TR 17534-3:2015 Table 3", "Geometrical divergence Adiv, dB")
def _chk_adiv() -> Outcome:
    """``Adiv = 20 lg(d/d0) + 11``, printed as 56,76 dB."""
    return numeric(
        56.76,
        ph.environment.geometric_divergence(_D3_FLAT),
        _ENVELOPE,
        unit="dB",
        places=3,
    )


@register(
    _QA,
    "ISO/TR 17534-3:2015 Table 3",
    "Middle-region overlap factor q (ISO 9613-2 Table 3, note 2)",
)
def _chk_q() -> Outcome:
    """``q = 1 - 30 (hs + hr) / dp`` once the two outer regions leave room."""
    q = 1.0 - 30.0 * (_SOURCE[2] + _RECEIVER[2]) / _DP
    return numeric(0.23, q, 0.005, places=4)


# --------------------------------------------------------------------------- #
# T04-T05: flat ground of three kinds, both ground methods
# --------------------------------------------------------------------------- #
#: Table 9, row "L", dB, and its two sums.
_T04_LEVELS = (39.90, 36.24, 35.23, 36.04, 36.95, 36.57, 33.87, 23.45)
_T04_TOTALS = (45.25, 42.23)
#: Table 10, row "L", dB, and its two sums: the same scenario by 7.3.2.
_T05_LEVELS = (34.90, 34.86, 34.71, 34.38, 33.95, 33.17, 30.48, 20.05)
_T05_TOTALS = (42.46, 39.30)
#: Table 8: the three region factors the printed segment lengths average to.
_T04_REGIONS = (0.20, 0.43, 0.67)
#: Table 17: the mean height of the ray over flat ground.
_T05_MEAN_HEIGHT = 2.50


def _t04_ground() -> ph.environment.GroundFactors:
    """Table 8's Gs, Gm and Gr, from the areas of Table 7."""
    return ph.environment.region_ground_factors(
        _SEGMENTS, _G_T04, _SOURCE[2], _RECEIVER[2]
    )


def _t05_mean_height() -> float:
    """Table 17's hm: flat ground, so the profile is a single level segment."""
    return ph.environment.mean_path_height(
        (0.0, _DP), (0.0, 0.0), _SOURCE[2], _RECEIVER[2]
    )


# --------------------------------------------------------------------------- #
# T06-T07: ground that varies in both height and kind, both ground methods
# --------------------------------------------------------------------------- #
#: Table 15, row "L", dB, and its two sums.
_T06_LEVELS = (39.88, 35.65, 29.70, 29.24, 34.82, 35.83, 33.13, 22.68)
_T06_TOTALS = (43.85, 40.59)
#: Table 18, row "L", dB, and its two sums.
_T07_LEVELS = (35.36, 35.32, 35.16, 34.83, 34.40, 33.62, 30.92, 20.47)
_T07_TOTALS = (42.91, 39.75)
#: Table 14: the same segment lengths as T04, over the reversed areas.
_T06_REGIONS = (0.90, 0.60, 0.37)
#: Table 17: the ray now clears a rising slope, so hm is twice the flat value.
_T07_MEAN_HEIGHT = 4.99
#: Table 14: the slant distance the risen receiver end puts the ray at.
_D3_SLOPED = math.hypot(_DP, _T06_RISE)


def _t06_ground() -> ph.environment.GroundFactors:
    """Table 14's Gs, Gm and Gr, from the areas of Table 11."""
    return ph.environment.region_ground_factors(
        _SEGMENTS, _G_T06, _SOURCE[2], _RECEIVER[2]
    )


def _t07_mean_height() -> float:
    """Table 17's hm, from the ground profile Table 12 cuts along the path."""
    return ph.environment.mean_path_height(
        _T06_PROFILE_DISTANCES,
        _T06_PROFILE_HEIGHTS,
        _SOURCE[2],
        _RECEIVER[2],
        distance=_D3_SLOPED,
    )


_VARYING = (
    ("T04", "flat ground of three kinds", _T04_LEVELS, _T04_TOTALS),
    ("T06", "ground rising under the receiver", _T06_LEVELS, _T06_TOTALS),
)
_ALTERNATIVE = (
    ("T05", "flat ground of three kinds", _T05_LEVELS, _T05_TOTALS),
    ("T07", "ground rising under the receiver", _T07_LEVELS, _T07_TOTALS),
)


def _varying_levels(case: str) -> np.ndarray:
    """T04 or T06 by the general method, with the factors the areas imply."""
    if case == "T04":
        return _general_method(_t04_ground(), _D3_FLAT)
    return _general_method(_t06_ground(), _D3_SLOPED)


def _alternative_levels(case: str) -> np.ndarray:
    """T05 or T07 by the alternative method, with the hm the profile implies."""
    if case == "T05":
        return _alternative_method(_t05_mean_height(), _D3_FLAT)
    return _alternative_method(_t07_mean_height(), _D3_SLOPED)


def _chk_case_spectrum(
    case: str, printed: tuple[float, ...], *, general: bool
) -> Outcome:
    computed = _varying_levels(case) if general else _alternative_levels(case)
    return _worst_band(printed, computed)


def _chk_case_total(
    case: str, printed: float, *, general: bool, a_weighted: bool
) -> Outcome:
    computed = _varying_levels(case) if general else _alternative_levels(case)
    return numeric(
        printed,
        _total(computed, a_weighted=a_weighted),
        _ENVELOPE,
        unit="dB",
        places=3,
    )


def _register_case(
    case: str,
    note: str,
    levels: tuple[float, ...],
    totals: tuple[float, float],
    *,
    general: bool,
) -> None:
    """Register the three rows of one case."""
    standard = f"ISO/TR 17534-3:2015 {case}"
    method = "general" if general else "alternative"
    register(_QA, standard, f"Receiver band levels, {note}, {method} method, dB")(
        functools.partial(_chk_case_spectrum, case, levels, general=general)
    )
    register(_QA, standard, f"Receiver total level, {note}, {method} method, dB")(
        functools.partial(
            _chk_case_total, case, totals[0], general=general, a_weighted=False
        )
    )
    register(_QA, standard, f"Receiver A-weighted level, {note}, {method} method, dB")(
        functools.partial(
            _chk_case_total, case, totals[1], general=general, a_weighted=True
        )
    )


for _case, _note, _levels, _totals in _VARYING:
    _register_case(_case, _note, _levels, _totals, general=True)
for _case, _note, _levels, _totals in _ALTERNATIVE:
    _register_case(_case, _note, _levels, _totals, general=False)


def _chk_region_factor(case: str, index: int, printed: float) -> Outcome:
    """One of the three region factors a mixed-ground path averages to."""
    factors = _t04_ground() if case == "T04" else _t06_ground()
    computed = (factors.source, factors.middle, factors.receiver)[index]
    return numeric(printed, computed, 0.005, places=4)


def _register_region_factors() -> None:
    """Register Gs, Gm and Gr for the two mixed-ground cases."""
    names = ("Gs (source region)", "Gm (middle region)", "Gr (receiver region)")
    for case, table, printed in (
        ("T04", "Table 8", _T04_REGIONS),
        ("T06", "Table 14", _T06_REGIONS),
    ):
        for index, name in enumerate(names):
            register(
                _QA,
                f"ISO/TR 17534-3:2015 {table} ({case})",
                f"Region ground factor {name} over three areas",
            )(functools.partial(_chk_region_factor, case, index, printed[index]))


_register_region_factors()


@register(_QA, "ISO/TR 17534-3:2015 Table 14 (T06)", "Straight-line path length d3, m")
def _chk_d3_sloped() -> Outcome:
    """The 10 m the ground gains under the receiver lengthens the ray to 194,60 m."""
    return numeric(194.60, _D3_SLOPED, _LENGTH_TOLERANCE, unit="m", places=3)


@register(
    _QA, "ISO/TR 17534-3:2015 Table 17 (T05)", "Mean path height hm over flat ground, m"
)
def _chk_hm_flat() -> Outcome:
    """Half the sum of the two heights, near enough, at this length."""
    return numeric(_T05_MEAN_HEIGHT, _t05_mean_height(), 0.005, unit="m", places=4)


@register(
    _QA, "ISO/TR 17534-3:2015 Table 17 (T07)", "Mean path height hm over a slope, m"
)
def _chk_hm_sloped() -> Outcome:
    """The area between the ray and the profile Table 12 cuts, over d3."""
    return numeric(_T07_MEAN_HEIGHT, _t07_mean_height(), 0.005, unit="m", places=4)
