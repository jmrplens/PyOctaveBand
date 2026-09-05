#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The closed unit vocabulary the artefact is allowed to speak.

The checks were written one domain at a time and spelled the same unit more
than one way: ``dB(A)`` beside ``dBA``, ``Pa s/m2`` beside ``Pa·s/m²``,
``m/s^2`` beside ``m/s2``. A report is prose, so it tolerated that; a document
consumers filter and group by cannot, because two spellings of one unit are
two units to everything downstream.

:data:`ALIASES` maps every spelling the checks use onto one canonical form and
:data:`UNITS` is the resulting vocabulary, listed at the head of the artefact
so a reader sees the whole of it. The canonical form is the one ISO 80000-3
writes: a middle dot for a product, a superscript for a power. A spelling that
is in neither table is rejected rather than passed through, because a silent
new spelling is exactly how the 58 arose.

``"mismatches"`` was in the ``unit=`` position and is not a unit: it labelled a
count of differing fields. Those checks carry ``kind="count"`` and no unit at
all, which is why it is absent here.
"""

from __future__ import annotations

#: Spellings the domain modules use, mapped onto the canonical form. Only the
#: spellings that need correcting are listed; a unit already canonical is not
#: repeated here, it is in :data:`UNITS` alone.
ALIASES: dict[str, str] = {
    "dB(A)": "dBA",
    "Pa s/m2": "Pa·s/m²",
    "N.s/m": "N·s/m",
    "m.Hz": "m·Hz",
    "m/s^2": "m/s²",
    "m/s2": "m/s²",
    "m^2": "m²",
    "m2": "m²",
    "m2/s": "m²/s",
    "W/m^2": "W/m²",
    "kg/m3": "kg/m³",
    "m/(N·s)": "m/(N·s)",
}

#: Every unit the artefact may carry, canonical spelling only. Sorted at the
#: document head so the vocabulary is visible rather than implied.
UNITS: frozenset[str] = frozenset(
    {
        "%",
        "1/kg",
        "1/s",
        "Cam",
        "EPNdB",
        "Hz",
        "LKFS",
        "LU",
        "LUFS",
        "MN/m³",
        "MPa",
        "N/m",
        "N·s/m",
        "Np/rad",
        "Pa·s/m²",
        "W/m²",
        "acum",
        "asper",
        "dB",
        "dB SPL",
        "dB/km",
        "dB/m",
        "dB/oct",
        "dB/s",
        "dBA",
        "dBTP",
        "dBqps",
        "deg",
        "ft",
        "kPa",
        "kg",
        "kg/m³",
        "lb",
        "m",
        "m/(N·s)",
        "m/N",
        "m/s",
        "mm/s",
        "m/s per °C",
        "m/s²",
        "m²",
        "m²/s",
        "Pa",
        "m³",
        "m·Hz",
        "modes/Hz",
        "ms",
        "nm",
        "ohm",
        "rad",
        "rad/m",
        "s",
        "sone",
        "sone_HMS",
        "tu_HMS",
        "vacil",
        "vacil_HMS",
        "yr",
    }
)


def canonical_unit(unit: str | None) -> str | None:
    """Return the canonical spelling of ``unit``.

    :param unit: The spelling a check wrote, ``""`` or ``None`` for a
        dimensionless quantity.
    :return: The canonical spelling, or ``None`` when the quantity carries no
        unit.
    :raises ValueError: If the spelling is neither canonical nor a known
        alias. Rejecting is the point: a unit admitted because it looked
        plausible is a second spelling of something already in the vocabulary,
        which is how the report reached 58 spellings of 54 units.
    """
    if not unit:
        return None
    canonical = ALIASES.get(unit, unit)
    if canonical not in UNITS:
        msg = (
            f"unit {unit!r} is not in the conformance unit vocabulary. Add it "
            "to UNITS in scripts/conformance/units.py if it is a new unit, or "
            "to ALIASES if it is another spelling of one already there."
        )
        raise ValueError(msg)
    return canonical
