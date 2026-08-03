#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Aircraft-noise measurement-system tolerances (IEC 61265:1995).

The certification levels of :mod:`phonometry.aircraft.certification` are only
worth what the chain that measured them is, and IEC 61265 is the standard that
says how good that chain has to be: microphone directional response, overall
frequency response, level linearity and the resolution of the reported level.
The one-third-octave filtering itself is covered by the IEC 61260 class 2
verification of :func:`phonometry.filters.verify_filter_class` (subclause 4.6)
and is not repeated here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "verify_aircraft_noise_system",
]


#: Tabulated depression/incidence angles (degrees) of IEC 61265 Table 1.
_IEC61265_ANGLES: tuple[float, ...] = (30.0, 60.0, 90.0, 120.0, 150.0)

#: IEC 61265:1995 Table 1: maximum permitted |sensitivity(0°) − sensitivity(θ)|
#: (dB) per one-third-octave band. Rows: (f_low, f_high, limits at the angles).
_IEC61265_DIRECTIONAL: tuple[tuple[float, float, tuple[float, ...]], ...] = (
    (50.0, 1600.0, (0.5, 0.5, 1.0, 1.0, 1.0)),
    (2000.0, 2000.0, (0.5, 0.5, 1.0, 1.0, 1.0)),
    (2500.0, 2500.0, (0.5, 0.5, 1.0, 1.5, 1.5)),
    (3150.0, 3150.0, (0.5, 1.0, 1.5, 2.0, 2.0)),
    (4000.0, 4000.0, (0.5, 1.0, 2.0, 2.5, 2.5)),
    (5000.0, 5000.0, (0.5, 1.5, 2.5, 3.0, 3.0)),
    (6300.0, 6300.0, (1.0, 2.0, 3.0, 4.0, 4.0)),
    (8000.0, 8000.0, (1.5, 2.5, 4.0, 5.5, 5.5)),
    (10000.0, 10000.0, (2.0, 3.5, 5.5, 6.5, 7.5)),
)


def _iec61265_directional_limit(frequency: float, angle: float) -> float:
    """The IEC 61265 Table 1 directional tolerance for a frequency and angle.

    Per subclause 4.4.2, an incidence angle between two tabulated angles takes
    the limit of the greater tabulated angle.
    """
    row = next(
        (lims for lo, hi, lims in _IEC61265_DIRECTIONAL if lo <= frequency <= hi),
        None,
    )
    if row is None:
        raise ValueError(
            "'frequency' is not an IEC 61265 tabulated one-third-octave band "
            "(50 Hz-1.6 kHz, then 2, 2.5, 3.15, 4, 5, 6.3, 8, 10 kHz)."
        )
    if angle <= 0.0 or angle > _IEC61265_ANGLES[-1]:
        raise ValueError("'angle' must lie in (0, 150] degrees.")
    col = next(i for i, a in enumerate(_IEC61265_ANGLES) if a >= angle)
    return row[col]


def verify_aircraft_noise_system(
    *,
    directional: dict[float, dict[float, float]] | None = None,
    frequency_response: dict[float, float] | None = None,
    linearity: dict[str, float] | None = None,
    resolution: float | None = None,
) -> dict[str, Any]:
    """Verify measured performance against IEC 61265:1995 tolerances.

    Each supplied measurement is checked against the standard's limit; the
    one-third-octave filtering itself is covered by the IEC 61260 class-2
    verification (subclause 4.6) and is not repeated here.

    :param directional: Microphone directional response as
        ``{frequency_hz: {angle_deg: |Δsensitivity| dB}}`` (Table 1, §4.4.2).
    :param frequency_response: System response deviations
        ``{frequency_hz: deviation_db}`` against the ±1.5 dB limit (§4.5.1).
    :param linearity: Level non-linearity ``{"reference": dB, "other": dB}``
        against the ±0.4/±0.5 dB limits (§4.5.2).
    :param resolution: Readout resolution, in dB, against the 0.1 dB limit (§4.7).
    :return: ``{"passed": bool, "checks": [{"quantity", "limit", "value", "ok",
        ...}]}``; ``passed`` is the conjunction of every check.
    :raises ValueError: If a frequency or angle is out of the tabulated range.
    """
    checks: list[dict[str, Any]] = []

    if directional is not None:
        for freq, per_angle in directional.items():
            for angle, value in per_angle.items():
                limit = _iec61265_directional_limit(float(freq), float(angle))
                checks.append(
                    {
                        "quantity": "directional_response",
                        "frequency": float(freq),
                        "angle": float(angle),
                        "limit": limit,
                        "value": float(value),
                        "ok": abs(float(value)) <= limit,
                    }
                )

    if frequency_response is not None:
        for freq, dev in frequency_response.items():
            checks.append(
                {
                    "quantity": "frequency_response",
                    "frequency": float(freq),
                    "limit": 1.5,
                    "value": float(dev),
                    "ok": abs(float(dev)) <= 1.5,
                }
            )

    if linearity is not None:
        for kind, dev in linearity.items():
            if kind not in ("reference", "other"):
                raise ValueError("linearity keys must be 'reference' or 'other'.")
            limit = 0.4 if kind == "reference" else 0.5
            checks.append(
                {
                    "quantity": f"linearity_{kind}",
                    "limit": limit,
                    "value": float(dev),
                    "ok": abs(float(dev)) <= limit,
                }
            )

    if resolution is not None:
        res = float(resolution)
        checks.append(
            {
                "quantity": "resolution",
                "limit": 0.1,
                "value": res,
                "ok": bool(np.isfinite(res)) and 0.0 <= res <= 0.1,
            }
        )

    passed = bool(checks) and all(c["ok"] for c in checks)
    return {"passed": passed, "checks": checks}
