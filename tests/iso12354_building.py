#  Copyright (c) 2026. Jose Manuel Requena Plens
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
  internal wall 2: 1,839 m against the printed 1,840 m) and supplies the three
  that are not;
* the external walls' internal loss factor is 0,012 5 (the element
  specification and Annex B Table B.3), not the 0,013 of the input block.

Both ``tests/building/prediction/test_detailed_model.py`` and
``scripts/conformance_report.py`` import these helpers, so the per-band tables
the tests assert and the conformance rows the report prints can never be built
two different ways.
"""

from __future__ import annotations

import reference_data as ref

from phonometry import building

#: Critical frequencies of the three constructions, in hertz.
FC_FLOOR, FC_EXT, FC_INT = 76.8, 92.6, 128.4

#: Mass per unit area of the three constructions, in kg/m².
M_FLOOR, M_EXT, M_INT = 484.0, 219.0, 360.0

#: Separating-element area ``Ss`` of the example, in m².
SEPARATING_AREA = 20.0

#: Junction coupling length of each flanking element, in metres (the ninth
#: field of each Annex L element row; the separating floor carries 0).
COUPLING_LENGTH = {
    label: values[8]
    for label, values in ref.ISO12354_ANNEX_L_ELEMENTS.items()
    if values[8] > 0.0
}

#: The junction each flanking element makes with the separating floor: the
#: external walls run past it in a rigid T, the internal walls cross it.
JUNCTION_KIND = {
    label: "ext" if label.startswith("ext") else "int" for label in COUPLING_LENGTH
}


def junction_indices() -> dict[str, float]:
    """The example's Annex E junction indices, unrounded.

    Tables L.5 to L.9 / G.5 to G.9 print them rounded to 0,1 dB; the perimeter
    sums of Formula (C.4) only reproduce the printed ``Σ lk αk`` when the
    unrounded values are used, so they are re-derived from Annex E here.
    """
    return {
        "floor-ext": building.junction_vibration_reduction(
            "rigid_t", "corner", M_FLOOR / M_EXT
        ),
        "ext-ext": building.junction_vibration_reduction(
            "rigid_t", "through", M_FLOOR / M_EXT
        ),
        "floor-floor": building.junction_vibration_reduction(
            "rigid_cross", "through", M_INT / M_FLOOR
        ),
        "int-int": building.junction_vibration_reduction(
            "rigid_cross", "through", M_FLOOR / M_INT
        ),
        "floor-int": building.junction_vibration_reduction(
            "rigid_cross", "corner", M_INT / M_FLOOR
        ),
        "ext1-ext2": building.junction_vibration_reduction("corner", "corner", 1.0),
        "int-ext": building.junction_vibration_reduction(
            "rigid_t", "corner", M_INT / M_EXT
        ),
        "extT-ext": building.junction_vibration_reduction(
            "rigid_t", "through", M_INT / M_EXT
        ),
        "int1-int2": building.junction_vibration_reduction(
            "rigid_cross", "through", 1.0
        ),
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
    alpha = building.perimeter_absorption_coefficient
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


def elements() -> dict[str, building.HomogeneousElement]:
    """The five elements of the Annex L / Annex G building."""
    sums = perimeter_sums()
    built: dict[str, building.HomogeneousElement] = {}
    for label, values in ref.ISO12354_ANNEX_L_ELEMENTS.items():
        area, length1, length2, mass, fc, eta_int, rho, c_l, _lij = values
        built[label] = building.HomogeneousElement(
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


def airborne_paths(situ: dict, delta_r: object) -> list:
    """The twelve flanking paths of the Annex L airborne example.

    Each of the four flanking elements contributes ``Df``, ``Fd`` and ``Ff``.
    The floating floor sits on the separating element in the source room, so
    its improvement enters the ``Df`` paths (where the separating element is
    element ``i``) and the direct path, but not ``Fd`` or ``Ff``.

    :param situ: The elements of :func:`elements` run through
        ``in_situ_element``, keyed by label.
    :param delta_r: The floating floor's improvement per band, in dB.
    :return: The twelve :class:`~phonometry.BandPath` objects, labelled as
        Table L.1 labels its columns (``D1``, ``1d``, ``11``, ...).
    """

    kij = junction_indices()
    paths = []
    for tag, name in enumerate_flanking():
        wall = situ[name]
        lij = COUPLING_LENGTH[name]
        cross, through = junction_paths(name, kij)
        paths.append(
            building.airborne_flanking_path(
                label=f"D{tag}",
                kind="Df",
                element_i=situ["floor"],
                element_j=wall,
                vibration_reduction_index=cross,
                coupling_length=lij,
                separating_area=SEPARATING_AREA,
                delta_r_i=delta_r,
            )
        )
        paths.append(
            building.airborne_flanking_path(
                label=f"{tag}d",
                kind="Fd",
                element_i=wall,
                element_j=situ["floor"],
                vibration_reduction_index=cross,
                coupling_length=lij,
                separating_area=SEPARATING_AREA,
            )
        )
        paths.append(
            building.airborne_flanking_path(
                label=f"{tag}{tag}",
                kind="Ff",
                element_i=wall,
                element_j=wall,
                vibration_reduction_index=through,
                coupling_length=lij,
                separating_area=SEPARATING_AREA,
            )
        )
    return paths


def impact_paths(situ: dict, delta_l: object) -> list:
    """The four flanking impact paths of the Annex G example.

    :param situ: The elements of :func:`elements` run through
        ``in_situ_element``, keyed by label.
    :param delta_l: The floating floor's improvement per band, in dB.
    :return: The four :class:`~phonometry.BandPath` objects, labelled as
        Table G.1 labels its columns (``Df1`` to ``Df4``).
    """

    kij = junction_indices()
    return [
        building.impact_flanking_path(
            label=f"Df{tag}",
            floor=situ["floor"],
            element_j=situ[name],
            vibration_reduction_index=junction_paths(name, kij)[0],
            coupling_length=COUPLING_LENGTH[name],
            delta_l=delta_l,
        )
        for tag, name in enumerate_flanking()
    ]


def enumerate_flanking() -> tuple[tuple[str, str], ...]:
    """The four flanking elements as ``(Table L.1 index, label)`` pairs."""
    return tuple(
        (str(index), label)
        for index, label in enumerate(("ext1", "ext2", "int1", "int2"), start=1)
    )


def junction_paths(label: str, kij: dict[str, float]) -> tuple[float, float]:
    """The ``(floor-to-element, element-through)`` indices of one junction."""
    if JUNCTION_KIND[label] == "ext":
        return kij["floor-ext"], kij["ext-ext"]
    return kij["floor-int"], kij["int-int"]


def floating_floor_resonance() -> float:
    """The floating floor's resonance ``fo = 160 √(s'/m')``, in hertz."""
    return (
        160.0
        * (ref.ISO12354_ANNEX_L_FLOATING_STIFFNESS / ref.ISO12354_ANNEX_L_FLOATING_MASS)
        ** 0.5
    )
