#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Numerical conformance report for phonometry.

A maintainable registry of numerical conformance checks. Each entry pins one
(standard, quantity) pair to:

* the standard designation + clause/table citation,
* the normative expected value or range (from the standard's own worked
  examples, or a closed form synthesized to a known result),
* a callable that computes the library's result, and
* the tolerance.

Running the harness emits Markdown for a GitHub PR comment: a headline
summary followed only by collapsible sections, so the comment stays compact
by default. First a "Numerical validation - filters & weightings" showcase
(per filter architecture IEC 61260-1 class margins and A/C/G weighting
worst-case deviation vs the analytic/normative curve), then one conformance
table per domain (Standard | Quantity | Expected | Computed | Delta |
Status). Every section stays collapsed while all of its rows pass and opens
automatically when any row fails.

Expected values are pulled from a single source of truth wherever the tests
already encode them: the shared ``tests/reference_data`` tables and the
ISO 532-1 data fixtures. Where the reference is a closed form, the harness
synthesizes an input with a known output, so the check is self-verifying.

Design goals: deterministic, fast (< 1 min), no network, pure library calls.

The checks themselves live in the :mod:`conformance` package next to this
file, one module per subject; what remains here is the entry point and the
facade over it, carrying the names a caller reaches for -- ``CHECKS``,
``Check``, ``render_markdown``, ``main`` -- while each check keeps its own
module. The 500-odd private check functions are not re-exported: they are
registered, not called by name. The domain import list below is the only thing this file decides, and
it is ordered on purpose: importing a domain module is what registers its
checks, ``CHECKS`` keeps registration order, and the report renders one
section per domain in that order. **That list is the report's section order.**
Moving a line moves a section; adding a module adds its section where the line
sits.
"""

from __future__ import annotations

# The report's section order. Each name here is a module of
# ``conformance.domains``; importing it registers its checks, they are rendered
# in the order they were registered, and so this list is the order of the
# report. It is deliberately not alphabetical, which is why the import sorter
# is switched off across it.
# isort: off
from conformance.domains import (
    filters,
    fluids,
    levels,
    psychoacoustics,
    speech,
    intensity,
    building,
    building_prediction,
    outdoor,
    iso17534,
    human_vibration,
    machine_vibration,
    intelligibility,
    assessment,
    sound_quality,
    distortion,
    signal_analysis,
    underwater,
    aircraft,
    environmental_sources,
    absorbers,
    program_loudness,
    quasi_peak,
    broadcast_wave,
    fdtd,
    swept_sine,
    ground_barriers,
    panels,
    plate_junctions,
    atmospheric_refraction,
    electroacoustics,
    noise_control,
    vdi2081,
    cnossos_rail,
)

# isort: on
from conformance.artifact import (
    _BY_DESIGN,
    ARTIFACT_PATH,
    build_document,
    check_id,
    dumps,
    load,
    write,
)
from conformance.compare import document_problems
from conformance.references import Reference, ReferenceKind
from conformance.registry import (
    _DATA,
    _ROOT,
    _TESTS,
    CHECKS,
    Binding,
    Check,
    Deviation,
    Kind,
    Outcome,
    Side,
    Tolerance,
    ToleranceMode,
    Verdict,
    _fmt,
    count,
    mask,
    numeric,
    record,
    register,
)
from conformance.render import (
    _DOC_HEADER,
    _cell,
    _domains,
    _filter_verdict,
    _numerical_validation_section,
    _snap,
    _status,
    main,
    render_markdown,
)
from conformance.shared import (
    _FILTER_ARCHS,
    FilterClass,
    WeightingDeviation,
    _filter_class,
    _pass_edge,
    _weighting_deviation,
)

# Re-exported so every name the single-file harness exposed still resolves
# through ``conformance_report``. The individual check callables are reached
# through ``CHECKS`` (or through their own domain module), as they always were.
__all__ = [
    "ARTIFACT_PATH",
    "CHECKS",
    "_BY_DESIGN",
    "_DATA",
    "_DOC_HEADER",
    "_FILTER_ARCHS",
    "_ROOT",
    "_TESTS",
    "Binding",
    "Check",
    "Deviation",
    "FilterClass",
    "Kind",
    "Outcome",
    "Reference",
    "ReferenceKind",
    "Side",
    "Tolerance",
    "ToleranceMode",
    "Verdict",
    "WeightingDeviation",
    "build_document",
    "check_id",
    "document_problems",
    "dumps",
    "load",
    "vdi2081",
    "write",
    "_cell",
    "_domains",
    "_filter_class",
    "_filter_verdict",
    "_fmt",
    "_numerical_validation_section",
    "_pass_edge",
    "_snap",
    "_status",
    "_weighting_deviation",
    "absorbers",
    "aircraft",
    "assessment",
    "atmospheric_refraction",
    "broadcast_wave",
    "building",
    "building_prediction",
    "cnossos_rail",
    "count",
    "distortion",
    "electroacoustics",
    "environmental_sources",
    "fdtd",
    "filters",
    "fluids",
    "ground_barriers",
    "human_vibration",
    "intelligibility",
    "intensity",
    "iso17534",
    "levels",
    "machine_vibration",
    "main",
    "mask",
    "noise_control",
    "numeric",
    "outdoor",
    "panels",
    "plate_junctions",
    "program_loudness",
    "psychoacoustics",
    "quasi_peak",
    "record",
    "register",
    "render_markdown",
    "signal_analysis",
    "sound_quality",
    "speech",
    "swept_sine",
    "underwater",
]


if __name__ == "__main__":
    raise SystemExit(main())
