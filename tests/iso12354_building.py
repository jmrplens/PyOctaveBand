#  Copyright (c) 2026. Jose M. Requena-Plens
"""Shared ISO 12354 Annex L / Annex G building fixture (tests + conformance).

Builds the heavy homogeneous building that ISO 12354-1:2017 Annex L (airborne)
and ISO 12354-2:2017 Annex G (impact) share: two dwellings one above the other
with a 220 mm concrete separating floor carrying a floating floor, two 365 mm
autoclaved aerated concrete external walls and two 200 mm calcium-silicate
internal walls, and the eight Annex E junctions between them.

Two inputs are taken **corrected**, not as printed (both recorded in
``docs/ERRATA.md``):

* the perimeter sums ``Σ lk αk`` of Formula (C.1) are derived here from
  Formula (C.4), ``αk = Σj √(fc,j/fref) 10^(−Kij/10)``, with the *unrounded*
  Annex E junction indices. The derivation returns the two printed values that
  are self-consistent with their own columns (external wall 1: 2,375 m;
  internal wall 2: 1,840 m) and supplies the three that are not;
* the external walls' internal loss factor is 0,012 5 (the element
  specification and Annex B Table B.3), not the 0,013 of the input block.

Both ``tests/building/test_detailed_prediction.py`` and
``scripts/conformance_report.py`` import these helpers, so the per-band tables
the tests assert and the conformance rows the report prints can never be built
two different ways.
"""

from __future__ import annotations

import reference_data as ref

from phonometry import (
    HomogeneousElement,
    junction_vibration_reduction,
    perimeter_absorption_coefficient,
)

#: Critical frequencies of the three constructions, in hertz.
FC_FLOOR, FC_EXT, FC_INT = 76.8, 92.6, 128.4

#: Mass per unit area of the three constructions, in kg/m².
M_FLOOR, M_EXT, M_INT = 484.0, 219.0, 360.0

#: Separating-element area ``Ss`` of the example, in m².
SEPARATING_AREA = 20.0

#: Junction coupling length of each flanking element, in metres.
COUPLING_LENGTH = {"ext1": 4.0, "ext2": 5.0, "int1": 4.0, "int2": 5.0}


def junction_indices() -> dict[str, float]:
    """The example's Annex E junction indices, unrounded.

    Tables L.5 to L.9 / G.5 to G.9 print them rounded to 0,1 dB; the perimeter
    sums of Formula (C.4) only reproduce the printed ``Σ lk αk`` when the
    unrounded values are used, so they are re-derived from Annex E here.
    """
    return {
        "floor-ext": junction_vibration_reduction(
            "rigid_t", "corner", M_FLOOR / M_EXT
        ),
        "ext-ext": junction_vibration_reduction(
            "rigid_t", "through", M_FLOOR / M_EXT
        ),
        "floor-floor": junction_vibration_reduction(
            "rigid_cross", "through", M_INT / M_FLOOR
        ),
        "int-int": junction_vibration_reduction(
            "rigid_cross", "through", M_FLOOR / M_INT
        ),
        "floor-int": junction_vibration_reduction(
            "rigid_cross", "corner", M_INT / M_FLOOR
        ),
        "ext1-ext2": junction_vibration_reduction("corner", "corner", 1.0),
        "int-ext": junction_vibration_reduction("rigid_t", "corner", M_INT / M_EXT),
        "extT-ext": junction_vibration_reduction("rigid_t", "through", M_INT / M_EXT),
        "int1-int2": junction_vibration_reduction("rigid_cross", "through", 1.0),
    }


def perimeter_sums(kij: dict[str, float] | None = None) -> dict[str, float]:
    """``Σ lk αk`` of each element from Formula (C.4).

    Each border contributes its length times the absorption coefficient of the
    elements connected there: the separating floor butts into the external
    walls (the wall above and the wall below) and crosses the internal walls;
    each external wall runs continuous past the separating floor and past the
    internal wall it meets, and corners the other external wall; each internal
    wall crosses the floor and the other internal wall and butts into the
    external wall (which it sees on both sides of the junction).
    """
    k = junction_indices() if kij is None else kij
    alpha = perimeter_absorption_coefficient
    floor_at_ext = alpha([FC_EXT, FC_EXT], [k["floor-ext"]] * 2)
    floor_at_int = alpha(
        [FC_FLOOR, FC_INT, FC_INT],
        [k["floor-floor"], k["floor-int"], k["floor-int"]],
    )
    ext_at_floor = alpha([FC_FLOOR, FC_EXT], [k["floor-ext"], k["ext-ext"]])
    ext_at_corner = alpha([FC_EXT], [k["ext1-ext2"]])
    ext_at_int = alpha([FC_INT, FC_EXT], [k["int-ext"], k["extT-ext"]])
    int_at_floor = alpha(
        [FC_FLOOR, FC_FLOOR, FC_INT],
        [k["floor-int"], k["floor-int"], k["int-int"]],
    )
    int_at_ext = alpha([FC_EXT, FC_EXT], [k["int-ext"]] * 2)
    int_at_int = alpha([FC_INT] * 3, [k["int1-int2"]] * 3)
    return {
        "floor": 9.0 * (floor_at_ext + floor_at_int),
        "ext1": 8.0 * ext_at_floor + 2.75 * (ext_at_corner + ext_at_int),
        "ext2": 10.0 * ext_at_floor + 2.75 * (ext_at_corner + ext_at_int),
        "int1": 8.0 * int_at_floor + 2.75 * (int_at_ext + int_at_int),
        "int2": 10.0 * int_at_floor + 2.75 * (int_at_ext + int_at_int),
    }


def elements() -> dict[str, HomogeneousElement]:
    """The five elements of the Annex L / Annex G building."""
    sums = perimeter_sums()
    built: dict[str, HomogeneousElement] = {}
    for label, values in ref.ISO12354_ANNEX_L_ELEMENTS.items():
        area, length1, length2, mass, fc, eta_int, rho, c_l, _lij = values
        if label == "floor":
            eta_int = ref.ISO12354_ANNEX_L_FLOOR_ETA_INT
        built[label] = HomogeneousElement(
            label=label,
            area=area,
            length1=length1,
            length2=length2,
            mass_per_area=mass,
            critical_frequency=fc,
            internal_loss_factor=eta_int,
            perimeter_absorption=sums[label],
            density=rho,
            longitudinal_velocity=c_l,
        )
    return built


def floating_floor_resonance() -> float:
    """The floating floor's resonance ``fo = 160 √(s'/m')``, in hertz."""
    return 160.0 * (
        ref.ISO12354_ANNEX_L_FLOATING_STIFFNESS
        / ref.ISO12354_ANNEX_L_FLOATING_MASS
    ) ** 0.5
