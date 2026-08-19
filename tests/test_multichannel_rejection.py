#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Multichannel (2-D) time signals are rejected, never silently flattened.

A raveled stereo signal concatenates the channels into one wrong series;
every single-channel analysis entry point must refuse it instead (library
audit finding).
"""

from __future__ import annotations

import numpy as np
import pytest

import phonometry as ph

_STEREO = np.zeros((2, 4800))


@pytest.mark.parametrize(
    "analyze",
    [
        ph.psychoacoustics.loudness_ecma,
        ph.psychoacoustics.tonality_ecma,
        ph.psychoacoustics.roughness_ecma,
        ph.psychoacoustics.loudness_moore_glasberg,
    ],
    ids=["loudness_ecma", "tonality_ecma", "roughness_ecma", "moore_glasberg"],
)
def test_stereo_input_is_rejected(analyze) -> None:
    with pytest.raises(ValueError, match="1-D time series"):
        analyze(_STEREO, 48000.0)
