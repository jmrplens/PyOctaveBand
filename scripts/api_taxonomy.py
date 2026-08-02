#  Copyright (c) 2026. Jose Manuel Requena Plens
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
  parent: ``filters`` adds the package top level (``phonometry`` itself),
  and ``power`` includes
  ``metrology.intensity_compliance`` (the IEC 61043 class checker sits with
  the other instrument-conformance code but documents the intensity chain
  the rest of the section measures with).

Section keys are subpackage names wherever the taxonomy allows it, so a
reader who knows where a function lives in the code can predict where its
page lives. The three sections listed above are the exceptions, and they are
deliberate.

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
            "phonometry.filters.core",
            "phonometry.filters.weighting",
            "phonometry.filters.equalizer",
            "phonometry.filters.frequencies",
            "phonometry.filters.compliance",
        ),
    ),
    Section(
        key="signals",
        label_en="Signal analysis",
        label_es="Análisis de señal",
        modules=(
            "phonometry.signals.levels",
            "phonometry.signals.spectra",
            "phonometry.signals.miso",
            "phonometry.signals.time_frequency",
            "phonometry.signals.test_signals",
            "phonometry.signals.phase",
            "phonometry.signals.cepstrum",
            "phonometry.signals.synchronous_average",
            "phonometry.signals.inversion",
            "phonometry.signals.correlation",
            "phonometry.signals.envelope",
        ),
    ),
    Section(
        key="metrology",
        label_en="Calibration and uncertainty",
        label_es="Calibración e incertidumbre",
        modules=(
            "phonometry.metrology.calibration",
            "phonometry.metrology.uncertainty",
            "phonometry.metrology.data_qualification",
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
            "phonometry.psychoacoustics.erb_scale",
        ),
    ),
    Section(
        key="speech",
        label_en="Speech",
        label_es="Habla",
        modules=(
            "phonometry.speech.sti",
            "phonometry.speech.sii",
            "phonometry.speech.objective_intelligibility",
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
            "phonometry.room.room_modes",
            "phonometry.room.crowd_noise",
        ),
    ),
    Section(
        key="building",
        label_en="Building acoustics",
        label_es="Acústica de la edificación",
        modules=(
            "phonometry.building.measurement.insulation",
            "phonometry.building.prediction.panel_transmission",
            "phonometry.building.prediction.masonry_cavity_wall",
            "phonometry.building.measurement.heavy_impact",
            "phonometry.building.prediction.ceiling_plenum",
            "phonometry.building.prediction.aperture_transmission",
            "phonometry.building.measurement.lab_insulation",
            "phonometry.building.measurement.survey_insulation",
            "phonometry.building.measurement.intensity_insulation",
            "phonometry.building.measurement.flanking_transmission",
            "phonometry.building.prediction.facade",
            "phonometry.building.prediction.global_model",
            "phonometry.building.prediction.detailed_model",
            "phonometry.building.measurement.uncertainty",
            "phonometry.building.measurement.floor_covering_improvement",
            "phonometry.building.prediction.resilient_layers",
            "phonometry.building.measurement.structure_borne_power",
            "phonometry.building.prediction.installed_structure_borne",
            "phonometry.building.regulation.spain",
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
            "phonometry.materials.biot",
            "phonometry.materials.slow_sound_absorber",
            "phonometry.materials.scattering_diffusion",
            "phonometry.materials.diffuser_design",
            "phonometry.materials.metadiffuser",
            "phonometry.materials.road_absorption",
        ),
    ),
    Section(
        key="vibration",
        label_en="Vibration and structure-borne",
        label_es="Vibración y ruido estructural",
        modules=(
            "phonometry.vibration.structural.mechanical_mobility",
            "phonometry.vibration.structural.point_mobility",
            "phonometry.vibration.structural.radiation_efficiency",
            "phonometry.vibration.structural.junction_transmission",
            "phonometry.vibration.structural.experimental_sea",
            "phonometry.vibration.machinery.diagnostics",
            "phonometry.vibration.structural.transfer_stiffness",
            "phonometry.vibration.human.exposure",
            "phonometry.vibration.human.multiple_shock",
        ),
    ),
    Section(
        key="environment",
        label_en="Environmental acoustics",
        label_es="Acústica ambiental",
        modules=(
            "phonometry.environment.propagation.outdoor_propagation",
            "phonometry.environment.sources.cnossos_road",
            "phonometry.environment.propagation.ground_barriers",
            "phonometry.environment.propagation.refraction",
            "phonometry.environment.propagation.air_absorption",
            "phonometry.environment.sources.cnossos_rail",
            "phonometry.environment.assessment.impulse_prominence",
            "phonometry.environment.assessment.impulsive_sound",
            "phonometry.environment.assessment.rating",
            "phonometry.environment.sources.wind_turbine",
            "phonometry.environment.assessment.measurement",
            "phonometry.environment.assessment.spain",
        ),
    ),
    Section(
        key="aeroacoustics",
        label_en="Aircraft noise",
        label_es="Ruido de aeronaves",
        modules=(
            "phonometry.aircraft.aircraft_noise",
            "phonometry.aircraft.atmospheric_absorption",
            "phonometry.aircraft.airport_noise",
            "phonometry.aircraft.anp_fleet",
            "phonometry.aircraft.rotorcraft_noise",
        ),
    ),
    Section(
        key="underwater",
        label_en="Underwater acoustics",
        label_es="Acústica submarina",
        modules=(
            "phonometry.underwater.acoustics",
            "phonometry.underwater.propagation",
            "phonometry.underwater.weston_regimes",
            "phonometry.underwater.sound_speed",
            "phonometry.underwater.sonar_equation",
            "phonometry.underwater.ocean_ambient_noise",
            "phonometry.underwater.seabed_reflection",
            "phonometry.underwater.ship_radiated_noise",
            "phonometry.underwater.ship_traffic_noise",
            "phonometry.underwater.pile_driving_noise",
            "phonometry.underwater.marine_mammal_audiograms",
            "phonometry.underwater.marine_mammal_weighting",
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
            "phonometry.metrology.intensity_compliance",
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
            "phonometry.electroacoustics.sound_reinforcement",
        ),
    ),
    Section(
        key="noise_control",
        label_en="Industrial noise control",
        label_es="Control de ruido industrial",
        modules=(
            "phonometry.noise_control.silencers",
            "phonometry.noise_control.hvac",
            "phonometry.noise_control.duct_path",
            "phonometry.noise_control.duct_modes",
            "phonometry.noise_control.enclosures",
            "phonometry.noise_control.room_to_room",
        ),
    ),
    Section(
        key="broadcast",
        label_en="Program loudness",
        label_es="Sonoridad de programa",
        modules=("phonometry.broadcast.program_loudness",),
    ),
    Section(
        key="simulation",
        label_en="Wave simulation",
        label_es="Simulación de ondas",
        modules=(
            "phonometry.simulation.fdtd",
            "phonometry.simulation.ntff",
            "phonometry.simulation.elastic_fdtd",
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
    "filters": ("", "filters"),
    "signals": ("signals",),
    "psychoacoustics": ("psychoacoustics",),
    "speech": ("speech",),
    "hearing": ("hearing",),
    "rooms": ("room",),
    "building": ("building",),
    "materials": ("materials",),
    "vibration": ("vibration",),
    "environment": ("environment",),
    "aeroacoustics": ("aircraft",),
    "underwater": ("underwater",),
    "power": ("emission", "metrology"),
    "electroacoustics": ("electroacoustics",),
    "noise_control": ("noise_control",),
    "broadcast": ("broadcast",),
    "metrology": ("metrology",),
    "simulation": ("simulation",),
}

#: Public names whose home module cannot be derived from ``__module__``:
#: objects defined in private modules but exported publicly, module-level
#: constants without ``__module__``, and constants re-exported by more than
#: one public module. Maps public name -> full module key in the taxonomy.
OBJECT_MODULE_OVERRIDES: dict[str, str] = {
    # Defined in phonometry/__init__.py (reported as module "phonometry").
    "__version__": "phonometry",
    # The ERB_N / Cam constants are owned by erb_scale and imported by the
    # ISO 532-2 loudness model, so a plain scan sees them in both modules.
    "ERB_C1": "phonometry.psychoacoustics.erb_scale",
    "ERB_C2": "phonometry.psychoacoustics.erb_scale",
    "CAM_C": "phonometry.psychoacoustics.erb_scale",
    # Defined in phonometry._internal.warnings, exported at the top level.
    "PhonometryWarning": "phonometry",
    # Defined in phonometry._plot.room; documented helper for the ISO 18233
    # excitation signals that live in room_ir.
    "plot_excitation": "phonometry.room.room_ir",
    # Defined in phonometry._plot.geometry; documented with the materials
    # module whose devices each drawing depicts.
    "plot_absorber_stack": "phonometry.materials.porous_absorber",
    "plot_helmholtz_resonator_geometry":
        "phonometry.materials.slow_sound_absorber",
    "plot_slit_absorber_geometry": "phonometry.materials.slow_sound_absorber",
    "plot_qrd_geometry": "phonometry.materials.diffuser_design",
    "plot_metadiffuser_panel_geometry": "phonometry.materials.metadiffuser",
    "DEFAULT_POLAR_ANGLES": "phonometry.materials.diffuser_design",
    "plot_impedance_tube_geometry": "phonometry.materials.impedance_tube",
    "plot_transmission_tube_geometry": "phonometry.materials.impedance_tube",
    "plot_silencer_geometry": "phonometry.noise_control.silencers",
    "plot_plenum_geometry": "phonometry.noise_control.hvac",
    "plot_barrier_geometry": "phonometry.environment.propagation.ground_barriers",
    "plot_microphone_positions": "phonometry.emission.sound_power",
    "plot_aperture_geometry": "phonometry.building.prediction.aperture_transmission",
    "plot_piston_geometry": "phonometry.electroacoustics.piston",
    "plot_sound_reinforcement_geometry":
        "phonometry.electroacoustics.sound_reinforcement",
    "plot_facade_elements": "phonometry.building.prediction.facade",
    "plot_double_wall_geometry": "phonometry.building.prediction.panel_transmission",
    "plot_junction_geometry": "phonometry.vibration.structural.junction_transmission",
    "plot_insitu_geometry": "phonometry.materials.road_absorption",
    "plot_dynamic_stiffness_rig": "phonometry.materials.dynamic_stiffness",
    "plot_goniometer_geometry": "phonometry.materials.scattering_diffusion",
    "plot_plate_geometry": "phonometry.vibration.structural.radiation_efficiency",
    "plot_open_plan_geometry": "phonometry.room.open_plan",
    "plot_pp_probe_geometry": "phonometry.emission.intensity",
    # Defined in underwater.acoustics, also re-exported by
    # underwater.ship_radiated_noise (identity scan is ambiguous).
    "UNDERWATER_REFERENCE_PRESSURE": "phonometry.underwater.acoustics",
    # Defined in the private phonometry._report package; documented with the
    # building insulation ratings whose report() method it drives.
    "ReportMetadata": "phonometry.building.measurement.insulation",
}


def module_section(module: str) -> Section:
    """Return the section that documents ``module`` (full dotted name).

    :param module: Full module name, e.g. ``"phonometry.signals.levels"``.
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
    """``phonometry.signals.levels`` -> ``signal``; top level -> ``""``."""
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
