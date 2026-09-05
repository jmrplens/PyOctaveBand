#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The sudden change of duct section (VDI 2081 Part 1, Section 6.3, Figure 26).

The figure is a table of closed forms rather than a chart, so the oracle is
the expression it prints and the two frequency rules beside it, read from
printed folio 39.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import noise_control

BANDS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])


def printed(ratio: float) -> float:
    """The expression Figure 26 prints, uncapped."""
    return 10.0 * math.log10((ratio + 1.0) ** 2 / (4.0 * ratio))


def test_a_sudden_reduction_reflects_in_every_band() -> None:
    """The figure's own column says the frequency has no effect on it."""
    result = noise_control.section_change_loss(
        BANDS, 0.5, 0.2, shape="rectangular", upstream_size=0.8
    )
    assert result.values == pytest.approx(printed(2.5))
    assert result.quantity == "attenuation"


def test_a_sudden_increase_reflects_only_below_the_limit_frequency() -> None:
    """Above it the figure prints approximately nought."""
    # A 0,8 m side puts Equation (33) at 343 / 1,6 = 214,4 Hz, so the 63 and
    # 125 Hz bands are below it and the rest are above.
    result = noise_control.section_change_loss(
        BANDS, 0.2, 0.5, shape="rectangular", upstream_size=0.8
    )
    assert result.values[:2] == pytest.approx(printed(0.4))
    assert result.values[2:] == pytest.approx(0.0)


def test_the_expression_is_symmetric_in_the_ratio() -> None:
    """A ratio and its reciprocal reflect the same amount.

    ``(r + 1)^2 / 4r`` is unchanged by ``r -> 1/r``, so a duct that widens by
    a factor reflects what the same duct narrowing by it does; what differs
    between the two is only the frequency rule.
    """
    for ratio in (1.5, 2.5, 4.0, 9.0):
        assert printed(ratio) == pytest.approx(printed(1.0 / ratio))
        narrow = noise_control.section_change_loss(
            BANDS, ratio, 1.0, shape="round", upstream_size=0.4
        )
        wide = noise_control.section_change_loss(
            BANDS, 1.0, ratio, shape="round", upstream_size=0.4
        )
        assert narrow.values[0] == pytest.approx(wide.values[0])


def test_no_change_of_section_reflects_nothing() -> None:
    result = noise_control.section_change_loss(BANDS, 0.3, 0.3, shape="round")
    assert result.values == pytest.approx(0.0)


def test_the_vdi_3733_cap_binds_where_the_expression_passes_it() -> None:
    """5 dB is reached at a ratio of about 10,55, and nothing above it."""
    below = noise_control.section_change_loss(BANDS, 10.0, 1.0, shape="round")
    above = noise_control.section_change_loss(BANDS, 40.0, 1.0, shape="round")
    assert float(below.values[0]) == pytest.approx(printed(10.0))
    assert float(below.values[0]) < 5.0
    assert above.values == pytest.approx(5.0)


def test_the_cap_can_be_lifted_or_lowered_by_the_caller() -> None:
    uncapped = noise_control.section_change_loss(
        BANDS, 40.0, 1.0, shape="round", cap=100.0
    )
    assert uncapped.values == pytest.approx(printed(40.0))
    tighter = noise_control.section_change_loss(
        BANDS, 40.0, 1.0, shape="round", cap=2.0
    )
    assert tighter.values == pytest.approx(2.0)


def test_a_round_duct_takes_its_size_from_its_area() -> None:
    area = 0.05
    derived = noise_control.section_change_loss(BANDS, area, 0.02, shape="round")
    explicit = noise_control.section_change_loss(
        BANDS,
        area,
        0.02,
        shape="round",
        upstream_size=noise_control.equivalent_diameter(area),
    )
    assert derived.values == pytest.approx(explicit.values)


def test_a_rectangular_duct_without_a_size_is_refused() -> None:
    """Its largest side does not follow from its area, and the model needs it."""
    with pytest.raises(ValueError, match="largest side of a rectangular duct"):
        noise_control.section_change_loss(BANDS, 0.2, 0.5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"upstream_area": 0.0}, "upstream_area"),
        ({"downstream_area": -1.0}, "downstream_area"),
        ({"speed_of_sound": 0.0}, "speed_of_sound"),
        ({"cap": 0.0}, "cap"),
        ({"shape": "oval"}, "'shape' must be one of"),
        ({"upstream_size": 0.0}, "upstream_size"),
    ],
)
def test_what_the_model_refuses(kwargs: dict[str, object], match: str) -> None:
    call: dict[str, object] = {
        "upstream_area": 0.2,
        "downstream_area": 0.5,
        "shape": "round",
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        noise_control.section_change_loss(BANDS, **call)  # type: ignore[arg-type]


def test_the_label_says_which_way_the_section_goes() -> None:
    narrow = noise_control.section_change_loss(BANDS, 0.5, 0.2, shape="round")
    wide = noise_control.section_change_loss(BANDS, 0.2, 0.5, shape="round")
    assert "sudden reduction" in narrow.label
    assert "sudden increase" in wide.label
