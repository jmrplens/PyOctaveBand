#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Taxonomy for the generated API reference (scripts/generate_api_docs.py).

Maps every public phonometry module to a documentation section. Sections
become directories under ``site/src/content/docs/reference/api/<section>/``
and collapsed subgroups in the site sidebar; their order here is the display
order on the index page and in the sidebar.

Modules are keyed by their full dotted name (``phonometry.<subpackage>.<mod>``)
so the mapping is unambiguous after the subpackage modularization. The bare
``"phonometry"`` entry is the package top level: the handful of names that
live in ``phonometry/__init__.py`` itself (``__version__`` and
:class:`PhonometryWarning`).

Consistency is enforced at import time (fails loudly):

- every module appears in exactly one section;
- each section only contains modules from the subpackages declared for it in
  ``_SECTION_SUBPACKAGES``. Two sections deliberately span more than one
  parent: ``filters`` adds the package top level next to ``metrology``, and
  ``aeroacoustics`` includes ``environmental.wind_turbine_noise`` because the
  section groups by audience (aircraft and wind energy) while the module
  lives with the other environmental-rating code.

The generator additionally checks the taxonomy against reality: every module
that owns a public name must be mapped here, and every mapped module must
still exist (see ``scripts/generate_api_docs.py``).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Section:
    """One API-reference section (a sidebar subgroup and URL directory)."""

    key: str
    label_en: str
    label_es: str
    modules: tuple[str, ...]


_SECTION_LIST: tuple[Section, ...] = (
    Section(
        key="filters",
        label_en="Filters and frequencies",
        label_es="Filtros y frecuencias",
        modules=(
            "phonometry",
            "phonometry.metrology.core",
            "phonometry.metrology.parametric_filters",
            "phonometry.metrology.equalizer",
            "phonometry.metrology.frequencies",
            "phonometry.metrology.compliance",
        ),
    ),
    Section(
        key="levels",
        label_en="Levels and calibration",
        label_es="Niveles y calibración",
        modules=(
            "phonometry.metrology.levels",
            "phonometry.metrology.calibration",
        ),
    ),
    Section(
        key="psychoacoustics",
        label_en="Psychoacoustics",
        label_es="Psicoacústica",
        modules=(
            "phonometry.psychoacoustics.loudness_zwicker",
            "phonometry.psychoacoustics.loudness_moore_glasberg",
            "phonometry.psychoacoustics.loudness_moore_glasberg_time",
            "phonometry.psychoacoustics.loudness_ecma",
            "phonometry.psychoacoustics.loudness_contours",
            "phonometry.psychoacoustics.sharpness",
            "phonometry.psychoacoustics.roughness_ecma",
            "phonometry.psychoacoustics.tonality",
            "phonometry.psychoacoustics.tonality_ecma",
            "phonometry.psychoacoustics.tone_audibility",
            "phonometry.psychoacoustics.fluctuation_strength",
            "phonometry.psychoacoustics.fluctuation_strength_ecma",
            "phonometry.psychoacoustics.psychoacoustic_annoyance",
        ),
    ),
    Section(
        key="speech",
        label_en="Speech",
        label_es="Habla",
        modules=(
            "phonometry.hearing.sti",
            "phonometry.hearing.sii",
            "phonometry.hearing.objective_intelligibility",
        ),
    ),
    Section(
        key="hearing",
        label_en="Hearing and exposure",
        label_es="Audición y exposición",
        modules=(
            "phonometry.hearing.threshold",
            "phonometry.hearing.noise_induced_hearing_loss",
            "phonometry.hearing.occupational_exposure",
        ),
    ),
    Section(
        key="rooms",
        label_en="Room acoustics",
        label_es="Acústica de salas",
        modules=(
            "phonometry.room.room_acoustics",
            "phonometry.room.room_ir",
            "phonometry.room.room_noise",
            "phonometry.room.open_plan",
            "phonometry.room.reverberation_prediction",
            "phonometry.room.enclosed_space_absorption",
            "phonometry.room.image_source",
            "phonometry.room.steady_field",
        ),
    ),
    Section(
        key="building",
        label_en="Building acoustics",
        label_es="Acústica de la edificación",
        modules=(
            "phonometry.building.insulation",
            "phonometry.building.panel_transmission",
            "phonometry.building.aperture_transmission",
            "phonometry.building.lab_insulation",
            "phonometry.building.survey_insulation",
            "phonometry.building.intensity_insulation",
            "phonometry.building.flanking_transmission",
            "phonometry.building.facade_prediction",
            "phonometry.building.building_prediction",
            "phonometry.building.building_uncertainty",
            "phonometry.building.floor_covering_improvement",
            "phonometry.building.structure_borne_power",
            "phonometry.building.installed_structure_borne",
        ),
    ),
    Section(
        key="materials",
        label_en="Materials and surfaces",
        label_es="Materiales y superficies",
        modules=(
            "phonometry.materials.sound_absorption",
            "phonometry.materials.absorption_rating",
            "phonometry.materials.absorption_uncertainty",
            "phonometry.materials.airflow_resistance",
            "phonometry.materials.dynamic_stiffness",
            "phonometry.materials.impedance_tube",
            "phonometry.materials.porous_absorber",
            "phonometry.materials.slow_sound_absorber",
            "phonometry.materials.scattering_diffusion",
            "phonometry.materials.diffuser_design",
            "phonometry.materials.road_absorption",
        ),
    ),
    Section(
        key="vibration",
        label_en="Vibration and structure-borne",
        label_es="Vibración y ruido estructural",
        modules=(
            "phonometry.vibration.mechanical_mobility",
            "phonometry.vibration.point_mobility",
            "phonometry.vibration.radiation_efficiency",
            "phonometry.vibration.junction_transmission",
            "phonometry.vibration.transfer_stiffness",
            "phonometry.vibration.human_vibration",
            "phonometry.vibration.multiple_shock_vibration",
        ),
    ),
    Section(
        key="environment",
        label_en="Environmental acoustics",
        label_es="Acústica ambiental",
        modules=(
            "phonometry.environmental.outdoor_propagation",
            "phonometry.environmental.ground_barriers",
            "phonometry.environmental.atmospheric_refraction",
            "phonometry.environmental.air_absorption",
            "phonometry.environmental.impulse_prominence",
            "phonometry.environmental.impulsive_sound",
            "phonometry.environmental.rating",
            "phonometry.environmental.measurement",
        ),
    ),
    Section(
        key="aeroacoustics",
        label_en="Aircraft and wind energy",
        label_es="Aeronaves y energía eólica",
        modules=(
            "phonometry.aircraft.aircraft_noise",
            "phonometry.aircraft.atmospheric_absorption",
            "phonometry.aircraft.airport_noise",
            "phonometry.aircraft.anp_fleet",
            "phonometry.aircraft.rotorcraft_noise",
            "phonometry.environmental.wind_turbine_noise",
        ),
    ),
    Section(
        key="underwater",
        label_en="Underwater acoustics",
        label_es="Acústica submarina",
        modules=(
            "phonometry.underwater.acoustics",
            "phonometry.underwater.propagation",
            "phonometry.underwater.sound_speed",
            "phonometry.underwater.sonar_equation",
            "phonometry.underwater.ocean_ambient_noise",
            "phonometry.underwater.seabed_reflection",
            "phonometry.underwater.ship_radiated_noise",
            "phonometry.underwater.ship_traffic_noise",
            "phonometry.underwater.pile_driving_noise",
            "phonometry.underwater.numerical_propagation",
        ),
    ),
    Section(
        key="power",
        label_en="Sound power and intensity",
        label_es="Potencia acústica e intensidad",
        modules=(
            "phonometry.emission.sound_power",
            "phonometry.emission.sound_power_intensity",
            "phonometry.emission.sound_power_reverberation",
            "phonometry.emission.intensity",
            "phonometry.emission.vibration_sound_power",
            "phonometry.emission.declaration",
        ),
    ),
    Section(
        key="electroacoustics",
        label_en="Electroacoustics",
        label_es="Electroacústica",
        modules=(
            "phonometry.electroacoustics.distortion",
            "phonometry.electroacoustics.frequency_response",
            "phonometry.electroacoustics.swept_sine",
            "phonometry.electroacoustics.piston",
            "phonometry.electroacoustics.loudspeaker",
            "phonometry.electroacoustics.microphone",
        ),
    ),
    Section(
        key="noise_control",
        label_en="Industrial noise control",
        label_es="Control de ruido industrial",
        modules=(
            "phonometry.noise_control.silencers",
            "phonometry.noise_control.hvac",
            "phonometry.noise_control.enclosures",
        ),
    ),
    Section(
        key="broadcast",
        label_en="Program loudness",
        label_es="Sonoridad de programa",
        modules=("phonometry.broadcast.program_loudness",),
    ),
    Section(
        key="metrology",
        label_en="Uncertainty and data quality",
        label_es="Incertidumbre y calidad de datos",
        modules=(
            "phonometry.metrology.uncertainty",
            "phonometry.metrology.random_data",
        ),
    ),
    Section(
        key="spectra",
        label_en="Spectral analysis",
        label_es="Análisis espectral",
        modules=(
            "phonometry.metrology.spectra",
            "phonometry.metrology.miso",
            "phonometry.metrology.time_frequency",
            "phonometry.metrology.signals",
            "phonometry.metrology.phase",
            "phonometry.metrology.cepstrum",
            "phonometry.metrology.synchronous_average",
            "phonometry.metrology.inversion",
        ),
    ),
    Section(
        key="simulation",
        label_en="Wave simulation",
        label_es="Simulación de ondas",
        modules=("phonometry.simulation.fdtd",),
    ),
    Section(
        key="correlation",
        label_en="Correlation & envelope",
        label_es="Correlación y envolvente",
        modules=(
            "phonometry.metrology.correlation",
            "phonometry.metrology.envelope",
        ),
    ),
)

#: Sections in display order, keyed by section key.
SECTIONS: dict[str, Section] = {s.key: s for s in _SECTION_LIST}

#: Parent subpackages allowed per section (the hard consistency contract).
#: ``""`` is the package top level (``phonometry/__init__.py``). Sections
#: spanning more than one parent are deliberate and documented in the module
#: docstring above.
_SECTION_SUBPACKAGES: dict[str, tuple[str, ...]] = {
    "filters": ("", "metrology"),
    "levels": ("metrology",),
    "psychoacoustics": ("psychoacoustics",),
    "speech": ("hearing",),
    "hearing": ("hearing",),
    "rooms": ("room",),
    "building": ("building",),
    "materials": ("materials",),
    "vibration": ("vibration",),
    "environment": ("environmental",),
    "aeroacoustics": ("aircraft", "environmental"),
    "underwater": ("underwater",),
    "power": ("emission",),
    "electroacoustics": ("electroacoustics",),
    "noise_control": ("noise_control",),
    "broadcast": ("broadcast",),
    "metrology": ("metrology",),
    "spectra": ("metrology",),
    "simulation": ("simulation",),
    "correlation": ("metrology",),
}

#: Public names whose home module cannot be derived from ``__module__``:
#: objects defined in private modules but exported publicly, module-level
#: constants without ``__module__``, and constants re-exported by more than
#: one public module. Maps public name -> full module key in the taxonomy.
OBJECT_MODULE_OVERRIDES: dict[str, str] = {
    # Defined in phonometry/__init__.py (reported as module "phonometry").
    "__version__": "phonometry",
    # Defined in phonometry._internal.warnings, exported at the top level.
    "PhonometryWarning": "phonometry",
    # Defined in phonometry._plot.room; documented helper for the ISO 18233
    # excitation signals that live in room_ir.
    "plot_excitation": "phonometry.room.room_ir",
    # Defined in underwater.acoustics, also re-exported by
    # underwater.ship_radiated_noise (identity scan is ambiguous).
    "UNDERWATER_REFERENCE_PRESSURE": "phonometry.underwater.acoustics",
    # Defined in the private phonometry._report package; documented with the
    # building insulation ratings whose report() method it drives.
    "ReportMetadata": "phonometry.building.insulation",
}


def module_section(module: str) -> Section:
    """Return the section that documents ``module`` (full dotted name).

    :param module: Full module name, e.g. ``"phonometry.metrology.levels"``.
    :raises KeyError: If the module is not mapped; new public modules must be
        added to a section in ``scripts/api_taxonomy.py``.
    """
    section = _MODULE_TO_SECTION.get(module)
    if section is None:
        raise KeyError(
            f"module {module!r} is not mapped to any API-reference section; "
            "add it to the matching Section in scripts/api_taxonomy.py"
        )
    return section


def _parent_subpackage(module: str) -> str:
    """``phonometry.metrology.levels`` -> ``metrology``; top level -> ``""``."""
    parts = module.split(".")
    return parts[1] if len(parts) > 2 else ""


def _build_module_index() -> dict[str, Section]:
    index: dict[str, Section] = {}
    for section in _SECTION_LIST:
        allowed = _SECTION_SUBPACKAGES[section.key]
        for module in section.modules:
            if module in index:
                raise ValueError(
                    f"module {module!r} is assigned to both "
                    f"{index[module].key!r} and {section.key!r}"
                )
            parent = _parent_subpackage(module)
            if parent not in allowed:
                raise ValueError(
                    f"module {module!r} (subpackage {parent!r}) does not "
                    f"belong to section {section.key!r}, which only accepts "
                    f"subpackages {allowed!r}"
                )
            index[module] = section
    for name, module in OBJECT_MODULE_OVERRIDES.items():
        if module not in index:
            raise ValueError(
                f"OBJECT_MODULE_OVERRIDES[{name!r}] points to unmapped "
                f"module {module!r}"
            )
    return index


_MODULE_TO_SECTION: dict[str, Section] = _build_module_index()
