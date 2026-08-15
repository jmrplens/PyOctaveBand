#  Copyright (c) 2026. Jose Manuel Requena Plens
"""To-scale geometry drawings of measurement set-ups and treatment designs.

Every renderer draws the *physical* device the way a lab manual would: a
dimensioned cross-section in metres with ``ax.set_aspect("equal")``, so a
100 mm tube really is twice as tall as a 50 mm one. They complement the
spectral ``plot()`` renderers: the geometry is what you build, the spectrum is
what you measure.

Lazy-imported from the ``plot()``/``plot_geometry()`` methods of the domain
objects and from the public ``plot_*_geometry`` functions; domain classes are
referenced only under ``TYPE_CHECKING`` so this rendering leaf never imports
domain code at module level (see ``tests/test_package_architecture.py``).
Layer dataclasses are dispatched by class name at runtime for the same reason.

The drawings are grouped by the domain whose set-up they depict, one module per
domain, over the shared drafting primitives of :mod:`._draft`. This package is
the import surface those modules are reached through: every renderer is
re-exported here, so ``from .._plot.geometry import plot_barrier_geometry``
resolves exactly as it did when the drawings were one module.
"""

from __future__ import annotations

from .building import (
    plot_aperture_geometry,
    plot_aperture_result_geometry,
    plot_double_wall_geometry,
    plot_double_wall_result_geometry,
    plot_facade_elements,
    plot_facade_result_geometry,
)
from .electroacoustics import (
    plot_piston_geometry,
    plot_piston_result_geometry,
    plot_sound_reinforcement_geometry,
)
from .emission import (
    plot_intensity_result_geometry,
    plot_microphone_positions,
    plot_pp_probe_geometry,
)
from .environment import (
    plot_barrier_geometry,
    plot_barrier_result_geometry,
)
from .materials import (
    plot_absorber_stack,
    plot_diffuser_geometry,
    plot_dynamic_stiffness_rig,
    plot_goniometer_geometry,
    plot_helmholtz_resonator_geometry,
    plot_impedance_tube_geometry,
    plot_impedance_tube_result_geometry,
    plot_insitu_geometry,
    plot_insitu_result_geometry,
    plot_layered_absorber_geometry,
    plot_metadiffuser_geometry,
    plot_metadiffuser_panel_geometry,
    plot_qrd_geometry,
    plot_slit_absorber_geometry,
    plot_slit_absorber_result_geometry,
    plot_transfer_matrix_geometry,
    plot_transmission_tube_geometry,
)
from .noise_control import (
    plot_plenum_geometry,
    plot_silencer_chain_geometry,
    plot_silencer_geometry,
    plot_silencer_result_geometry,
)
from .room import (
    plot_image_source_geometry,
    plot_open_plan_geometry,
    plot_open_plan_result_geometry,
)
from .simulation import (
    plot_fdtd_domain,
)
from .vibration import (
    plot_junction_geometry,
    plot_junction_result_geometry,
    plot_plate_geometry,
    plot_radiation_result_geometry,
)

__all__ = [
    "plot_absorber_stack",
    "plot_aperture_geometry",
    "plot_aperture_result_geometry",
    "plot_barrier_geometry",
    "plot_barrier_result_geometry",
    "plot_diffuser_geometry",
    "plot_double_wall_geometry",
    "plot_double_wall_result_geometry",
    "plot_dynamic_stiffness_rig",
    "plot_facade_elements",
    "plot_facade_result_geometry",
    "plot_fdtd_domain",
    "plot_goniometer_geometry",
    "plot_helmholtz_resonator_geometry",
    "plot_image_source_geometry",
    "plot_impedance_tube_geometry",
    "plot_impedance_tube_result_geometry",
    "plot_insitu_geometry",
    "plot_insitu_result_geometry",
    "plot_intensity_result_geometry",
    "plot_junction_geometry",
    "plot_junction_result_geometry",
    "plot_layered_absorber_geometry",
    "plot_metadiffuser_geometry",
    "plot_metadiffuser_panel_geometry",
    "plot_microphone_positions",
    "plot_open_plan_geometry",
    "plot_open_plan_result_geometry",
    "plot_piston_geometry",
    "plot_piston_result_geometry",
    "plot_plate_geometry",
    "plot_plenum_geometry",
    "plot_pp_probe_geometry",
    "plot_qrd_geometry",
    "plot_radiation_result_geometry",
    "plot_silencer_chain_geometry",
    "plot_silencer_geometry",
    "plot_silencer_result_geometry",
    "plot_slit_absorber_geometry",
    "plot_slit_absorber_result_geometry",
    "plot_sound_reinforcement_geometry",
    "plot_transfer_matrix_geometry",
    "plot_transmission_tube_geometry",
]
