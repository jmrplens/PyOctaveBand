#  Copyright (c) 2026. Jose Manuel Requena Plens
"""A temperature below absolute zero is refused, not computed with.

Left unguarded these did not raise, they returned: a negative speed of sound,
and an absorption that looks like a plausible measurement. The second is the
dangerous one, because nothing about 0,495 dB/km says it came from air at
-300 degC.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import fluids, underwater
from phonometry._internal.validation import ABSOLUTE_ZERO_C


def test_sea_water_sound_speed_refuses_absolute_zero() -> None:
    """It used to return -31 457 m/s, a negative speed of sound."""
    with pytest.raises(ValueError, match="'temperature' must be finite and above"):
        fluids.sea_water_sound_speed(ABSOLUTE_ZERO_C, 35.0, 0.0)


def test_sound_speed_profile_refuses_absolute_zero() -> None:
    """The array path checked only finiteness, so -300 degC went straight through."""
    with pytest.raises(ValueError, match="'temperatures' must be finite and above"):
        underwater.sound_speed_profile([0.0, 100.0], -300.0, 35.0)


def test_seawater_absorption_refuses_absolute_zero() -> None:
    """It returned 0.495 dB/km at -300 degC, beside 0.0185 for real cold water."""
    with pytest.raises(ValueError, match="'temperature' must be finite and above"):
        underwater.seawater_absorption(1e3, temperature=-300.0)


@pytest.mark.parametrize("temperature_c", [-273.1, -273.05, -273.0])
def test_francois_garrison_refuses_its_own_pole(temperature_c: float) -> None:
    """Its pole sits 0,15 degC above absolute zero, so the physical bound misses it.

    The relaxation frequencies are printed 1245/(273 + t) and 1990/(273 + t)
    with the 273 the paper prints. The whole band (-273,15, -273,0] therefore
    clears the physical check and then overflows or divides by zero. Both
    escaped as OverflowError and ZeroDivisionError, neither of which names the
    argument that caused it.
    """
    with pytest.raises(ValueError, match="francois-garrison"):
        underwater.seawater_absorption(
            1e3, temperature=temperature_c, model="francois-garrison"
        )


def test_just_above_the_pole_still_answers() -> None:
    """The guard is drawn at the pole, not above it: -272,9 degC is computable."""
    value = underwater.seawater_absorption(
        1e3, temperature=-272.9, model="francois-garrison"
    )
    assert np.all(np.isfinite(value))


@pytest.mark.parametrize("model", ["thorp", "ainslie-mccolm"])
def test_the_other_models_have_no_pole_of_their_own(model: str) -> None:
    """Only the Francois-Garrison branch needs the second guard."""
    value = underwater.seawater_absorption(1e3, temperature=-272.9, model=model)
    assert np.all(np.isfinite(value))


def test_ordinary_water_is_untouched() -> None:
    """The guards bound what cannot exist and nothing else."""
    assert fluids.sea_water_sound_speed(10.0, 35.0, 0.0) == pytest.approx(
        1489.832, abs=1e-3
    )
