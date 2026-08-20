#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The measure-then-slide pass that keeps a wave-field label on its panel.

``scripts/figures/fields/_core.py`` places two kinds of annotation against
the canvas they are drawn on rather than against the string they were written
with, because the Spanish edition of a clip is routinely the longer one:
``_fit_text_x`` slides a label along x until its rendered box is inside the
panel, and ``_fit_text_below`` slides a label set along a slope back down its
own baseline until it clears whatever it grew into.

The clips in the tree exercise the easy half of each: every label they draw
fits its panel once shifted, and the only sloped one leans the same way. The
branches pinned here are the other half -- a label leaning the other way, and
one too wide to be put inside its room by sliding at all -- which is where
the sign of a sine and the choice of edge stop being invisible.
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import pytest

pytest.importorskip("matplotlib")
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from figures.fields._core import _fit_text_below, _fit_text_x, _layout_box

GAP = 8.0


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


def _panel() -> tuple[Any, Any]:
    """A plain 10 x 10 data panel, drawn at a fixed size and resolution."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=100)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    return fig, ax


def _x_span(ax: Any, artist: Any) -> tuple[float, float]:
    """The artist's rendered box, in x data units of *ax*."""
    box = _layout_box(artist)
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([(box.x0, 0.0), (box.x1, 0.0)])
    return min(x0, x1), max(x0, x1)


def _slide_below(
    rotation: float,
) -> tuple[Any, Any, Any, Any, float, tuple[float, float]]:
    """Run ``_fit_text_below`` on a label tilted *rotation* degrees.

    The label is anchored on top of the one it has to clear, so the pass
    always has work to do. Returns the applied shift and the displacement it
    moved the anchor by, in display pixels.
    """
    fig, ax = _panel()
    other = ax.text(5.0, 5.0, "receiver arc", ha="center", va="center")
    label = ax.text(
        5.0, 5.0, "insertion loss 8 dB", rotation=rotation, ha="center", va="center"
    )
    before = ax.transData.transform(label.get_position())
    shift = _fit_text_below(fig, ax, label, other, gap=GAP)
    after = ax.transData.transform(label.get_position())
    return (
        fig,
        ax,
        label,
        other,
        shift,
        (float(after[0] - before[0]), float(after[1] - before[1])),
    )


@pytest.mark.parametrize("rotation", [-30.0, -12.0, -60.0])
def test_a_label_tilted_the_other_way_slides_the_other_way(
    rotation: float,
) -> None:
    """A negative slope travels in +x for the same drop, not in -x.

    The step comes out of the *signed* sine of the rotation. Reading the
    sine's magnitude instead would drop the label by the same amount and
    walk it the wrong way along the line -- back under the text it is
    clearing, which is the one direction the pass exists to avoid.
    """
    _fig, _ax, _label, _other, shift, (dx, dy) = _slide_below(rotation)

    assert shift > 0.0
    assert dy < 0.0  # it went down
    assert dx > 0.0  # and forward, along its own line
    # Along its own baseline: the displacement lies on the rotated line.
    assert dx == pytest.approx(dy / math.tan(math.radians(rotation)), rel=1e-6)
    assert shift == pytest.approx(math.hypot(dx, dy), rel=1e-9)


def test_the_two_slopes_are_mirror_images() -> None:
    """Same drop either way; only the direction along x changes sign."""
    *_, shift_down, (dx_down, dy_down) = _slide_below(-30.0)
    *_, shift_up, (dx_up, dy_up) = _slide_below(30.0)

    assert dy_down == pytest.approx(dy_up, rel=1e-3)
    assert dx_down == pytest.approx(-dx_up, rel=1e-3)
    assert shift_down == pytest.approx(shift_up, rel=1e-3)
    assert dx_up < 0.0 < dx_down


def test_a_level_label_can_only_go_down() -> None:
    """With no slope to slide along, the drop is straight down."""
    _fig, _ax, _label, _other, shift, (dx, dy) = _slide_below(0.0)

    assert dx == pytest.approx(0.0, abs=1e-9)
    assert dy < 0.0
    assert shift == pytest.approx(abs(dy), rel=1e-9)


@pytest.mark.parametrize("rotation", [-30.0, 0.0, 30.0])
def test_the_label_ends_up_clear_of_what_it_grew_into(
    rotation: float,
) -> None:
    fig, _ax, label, other, _shift, _delta = _slide_below(rotation)
    fig.canvas.draw()

    assert _layout_box(label).y1 <= _layout_box(other).y0 - GAP + 1e-6


def test_a_label_that_already_clears_is_left_alone() -> None:
    fig, ax = _panel()
    other = ax.text(5.0, 5.0, "receiver arc", ha="center", va="center")
    label = ax.text(
        5.0, 2.0, "insertion loss 8 dB", rotation=-30.0, ha="center", va="center"
    )
    before = label.get_position()

    assert _fit_text_below(fig, ax, label, other, gap=GAP) == 0.0
    assert label.get_position() == before


def test_a_label_too_wide_to_fit_is_anchored_in_reading_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Wider than its room, it overflows one edge instead of both.

    Netting the two overflows against each other would leave the label
    hanging over the left and the right spine at once. It is anchored flush
    against the left edge of the room instead -- the way the eye is already
    going -- and the pass says so on stderr.
    """
    fig, ax = _panel()
    label = ax.text(
        5.0,
        5.0,
        "a caption far wider than the room it is given",
        ha="left",
        va="center",
    )
    fig.canvas.draw()
    lo_before, hi_before = _x_span(ax, label)
    assert hi_before - lo_before > 2.0 - 2 * 0.2  # wider than the room

    shift = _fit_text_x(fig, ax, label, 1.0, 3.0, margin=0.2)

    assert "does not fit" in capsys.readouterr().err
    assert shift == pytest.approx(1.2 - lo_before, rel=1e-6)
    fig.canvas.draw()
    lo_after, hi_after = _x_span(ax, label)
    assert lo_after == pytest.approx(1.2, abs=1e-6)  # flush against x_lo
    assert hi_after > 2.8  # and over the far edge


def test_a_label_that_fits_is_slid_in_silently(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The everyday branch: past one spine, back by exactly the overflow."""
    fig, ax = _panel()
    label = ax.text(8.9, 5.0, "8 dB", ha="left", va="center")
    fig.canvas.draw()
    _lo_before, hi_before = _x_span(ax, label)
    assert hi_before > 9.0

    shift = _fit_text_x(fig, ax, label, 1.0, 9.0)

    assert "does not fit" not in capsys.readouterr().err
    assert shift == pytest.approx(9.0 - hi_before, rel=1e-6)
    fig.canvas.draw()
    _lo_after, hi_after = _x_span(ax, label)
    assert hi_after == pytest.approx(9.0, abs=1e-6)
