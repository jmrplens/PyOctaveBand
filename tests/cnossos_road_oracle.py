#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The (EU) 2015/996 Appendix F database, wrapped once for both readers.

``tests/reference_data/`` deliberately imports nothing from the library: it
is the transcribed oracle, and it must not depend on the code it is used to
check. But both the test suite and the conformance report need those rows as
``RoadEmissionCoefficients`` and ``RoadSurfaceCoefficients``, and each was
building them itself.

Two copies of the same wrapping is not a tidiness problem, it is a drift
hazard: if one gains a field the other does not, the conformance report starts
validating against a different database from the tests while both still pass.
This module is the single place that knows both the rows and the types, and it
is the only thing here that imports the library.

The workbook was computed with the superseded 2015 database, not the one the
library ships, which is why it lives here and not in the package.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import reference_data as ref

if TYPE_CHECKING:
    from phonometry import RoadEmissionCoefficients, RoadSurfaceCoefficients


@functools.cache
def coefficients_2015() -> RoadEmissionCoefficients:
    """Table F-1 as published in (EU) 2015/996, the workbook's own database.

    The studded-tyre, junction and temperature terms are unchanged by the 2021
    amendment, so they come from the shipped database rather than from a second
    transcription of identical numbers.
    """
    from phonometry import ROAD_COEFFICIENTS, RoadEmissionCoefficients

    table = ref.cnossos_road_2015_coefficients()
    return RoadEmissionCoefficients(
        rolling_a={k: v["AR"] for k, v in table.items()},
        rolling_b={k: v["BR"] for k, v in table.items()},
        propulsion_a={k: v["AP"] for k, v in table.items()},
        propulsion_b={k: v["BP"] for k, v in table.items()},
        studded_a=ROAD_COEFFICIENTS.studded_a,
        studded_b=ROAD_COEFFICIENTS.studded_b,
        junction_c=ROAD_COEFFICIENTS.junction_c,
        temperature_k=ROAD_COEFFICIENTS.temperature_k,
    )


@functools.cache
def surfaces_2015() -> dict[str, RoadSurfaceCoefficients]:
    """Table F-4 as published in (EU) 2015/996, keyed by the workbook's ID.

    ``speed_range`` is ``None`` because the 2015 table printed no validity
    range for these rows; the ranges arrived with the 2021 amendment.
    """
    from phonometry import RoadSurfaceCoefficients

    return {
        surface: RoadSurfaceCoefficients(
            name=name, alpha=alpha, beta=beta, speed_range=None
        )
        for surface, (name, alpha, beta) in ref.cnossos_road_2015_surfaces().items()
    }
