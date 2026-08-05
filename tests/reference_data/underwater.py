#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Underwater acoustics: the reference quantities that levels are stated against.

Underwater levels are referred to 1 uPa and 1 uPa^2 s (ISO 18405), not to
the 20 uPa of air, so every level quoted in one medium differs from the
same physical quantity quoted in the other by a fixed offset. That offset
is the whole of this module: get it wrong and every radiated noise level
of ISO 17208 and every exposure of ISO 18406 is wrong by 26 dB.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Underwater acoustics (ISO 18405 / 17208 / 18406). Reference pressure 1 µPa,
# reference exposure 1 µPa²·s. Level offset between the in-air (20 µPa) and
# underwater (1 µPa) references: 20·lg(20) = 26.0206 dB.
# ---------------------------------------------------------------------------
UW_REFERENCE_OFFSET_DB = 26.020599913279624
