#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The simulated wave fields and the clips that show them moving.

Every clip here is a 2D FDTD (or elastic FDTD) run and gets a module of its
own subject: a barrier, a ground effect, a SOFAR duct, an impedance or
transmission tube, a diffuser, a slit absorber, a plate junction. The
simulation is the expensive part and is memoised per process behind an
``lru_cache``, so the field builder and the clip that renders it belong
together: the four language x theme variants of a clip are rendered off one
computation of its field. Every clip keeps >= 48 captured frames per acoustic
period of its highest carrier, raising its playback rate with the capture
stride so the wavefronts read as continuous motion at the pacing the clip was
composed for, and every number printed on screen comes from the library
models, never from the simulation itself. What more than one of them needs --
the continuous-wave capture, the dB conversion, the weak-field display gain,
the drawn loudspeaker -- lives in :mod:`figures.fields._core`.

Importing the subject modules here keeps ``figures.fields`` naming the same
set of callables it named as a single module, which is what
:mod:`figures.registry`, :mod:`figures.simulation` and
``scripts/generate_graphs.py`` import from it, and it is also what puts every
subject module in ``sys.modules`` before the first figure is drawn, so the
theme and language switches of :func:`figures._publish` reach the palette and
language globals each of them holds.
"""

from .aperture import _aperture_fields, animate_fdtd_aperture_slit
from .barrier import _barrier_fields, animate_fdtd_barrier
from .diffusion import _diffusion_fields, animate_fdtd_diffusion
from .ducting import _ducting_fields, animate_fdtd_ducting
from .elastic import (
    _coincidence_fields,
    _plate_junction_fields,
    animate_elastic_coincidence,
    animate_elastic_plate_junction,
)
from .expansion_chamber import (
    _expansion_chamber_fields,
    animate_fdtd_expansion_chamber,
)
from .ground_effect import _ground_effect_fields, animate_fdtd_ground_effect
from .metadiffuser import (
    _meshed_metadiffuser_ntff_levels,
    _metadiffuser_fields,
    animate_fdtd_metadiffuser,
)
from .pillar_hall import (
    _pillar_fields,
    _poster_ss_for,
    animate_fdtd_pillar_hall,
)
from .refraction import _refraction_fields, animate_fdtd_refraction
from .room_modes import _room_mode_fields, animate_fdtd_room_modes
from .slit_absorber import _slit_absorber_fields, animate_fdtd_slit_absorber
from .tubes import (
    _impedance_tube_fields,
    _transmission_tube_fields,
    animate_fdtd_impedance_tube,
    animate_fdtd_transmission_tube,
    animate_standing_wave_tube,
)

__all__ = [
    "_aperture_fields",
    "_barrier_fields",
    "_coincidence_fields",
    "_diffusion_fields",
    "_ducting_fields",
    "_expansion_chamber_fields",
    "_ground_effect_fields",
    "_impedance_tube_fields",
    "_meshed_metadiffuser_ntff_levels",
    "_metadiffuser_fields",
    "_pillar_fields",
    "_plate_junction_fields",
    "_poster_ss_for",
    "_refraction_fields",
    "_room_mode_fields",
    "_slit_absorber_fields",
    "_transmission_tube_fields",
    "animate_elastic_coincidence",
    "animate_elastic_plate_junction",
    "animate_fdtd_aperture_slit",
    "animate_fdtd_barrier",
    "animate_fdtd_diffusion",
    "animate_fdtd_ducting",
    "animate_fdtd_expansion_chamber",
    "animate_fdtd_ground_effect",
    "animate_fdtd_impedance_tube",
    "animate_fdtd_metadiffuser",
    "animate_fdtd_pillar_hall",
    "animate_fdtd_refraction",
    "animate_fdtd_room_modes",
    "animate_fdtd_slit_absorber",
    "animate_fdtd_transmission_tube",
    "animate_standing_wave_tube",
]
