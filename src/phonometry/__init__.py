#  Copyright (c) 2020. Jose Manuel Requena Plens
"""Acoustic measurement toolkit, organized as one package per domain.

The library publishes nineteen domain packages and the handful of types that
cross between them. A function lives in the package that owns it and is
reached through it::

    from phonometry import building, underwater

    r = building.airborne_insulation(...)
    pl = underwater.propagation_loss(...)

Reading the domain at the call site is the point: two packages can hold a
``transmission_loss`` without either having to be renamed, and the reader of a
line sees which one it means. The generated reference documents one page per
module, and each package's own docstring says what it covers.

Four names sit at the top level because they belong to no single domain:
:class:`~phonometry.io.Signal`, the calibrated recording every package that
takes measured audio accepts; :class:`ReportMetadata`, which stamps the
accredited-report fiches of every domain; :class:`PhonometryWarning`, the base
of every diagnostic the library raises, so one
:func:`warnings.filterwarnings` rule reaches them all; and ``__version__``.
"""

from __future__ import annotations

from . import aircraft as aircraft
from . import broadcast as broadcast
from . import building as building
from . import electroacoustics as electroacoustics
from . import emission as emission
from . import environment as environment
from . import filters as filters
from . import hearing as hearing
from . import io as io
from . import materials as materials
from . import metrology as metrology
from . import noise_control as noise_control
from . import psychoacoustics as psychoacoustics
from . import room as room
from . import signals as signals
from . import simulation as simulation
from . import speech as speech
from . import underwater as underwater
from . import vibration as vibration
from ._internal.warnings import PhonometryWarning
from ._report import ReportMetadata
from ._version import __version__
from .io import Signal

__all__ = [
    "PhonometryWarning",
    "ReportMetadata",
    "Signal",
    "__version__",
    "aircraft",
    "broadcast",
    "building",
    "electroacoustics",
    "emission",
    "environment",
    "filters",
    "hearing",
    "io",
    "materials",
    "metrology",
    "noise_control",
    "psychoacoustics",
    "room",
    "signals",
    "simulation",
    "speech",
    "underwater",
    "vibration",
]
