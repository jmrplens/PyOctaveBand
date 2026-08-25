#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What more than one wave-field clip needs: capture, dB and gain.

The continuous-wave capture loop and the running-RMS to dB conversion every
FDTD clip post-processes its frames with, the weak-field display gain the
clips that put a loud and a quiet region in the same colour ramp share, the
loudspeaker drawn on the clips that drive a bore, and the measure-then-slide
pass that keeps an annotation inside its panel in either language.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from ..theme import COLOR_FG

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    from numpy.typing import NDArray

    from phonometry.simulation.fdtd import FDTD2D


def _fdtd_cw_capture(
    sim: FDTD2D, frequency: float, every: int, n_frames: int, decimate: int = 2
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]
]:
    """Drive a CW simulation and capture instantaneous + running-RMS frames.

    The running mean square uses a two-period time constant, so the RMS map
    (the lobe/shadow pattern) builds up as the field settles. Returns the
    float32 frame stacks (decimated), the frame times and the final
    full-resolution RMS map for physics probes.
    """
    beta = float(np.exp(-sim.dt * frequency / 2.0))
    ms = np.zeros_like(sim.p)
    ps: list[NDArray[np.float32]] = []
    rs: list[NDArray[np.float32]] = []
    ts: list[float] = []
    for _ in range(every * n_frames):
        sim.step()
        ms = beta * ms + (1.0 - beta) * sim.p**2
        if sim.n % every == 0 and len(ps) < n_frames:
            ps.append(sim.p[::decimate, ::decimate].astype(np.float32))
            rs.append(np.sqrt(ms[::decimate, ::decimate]).astype(np.float32))
            ts.append(sim.time)
    return np.stack(ps), np.stack(rs), np.asarray(ts), np.sqrt(ms)


def _rms_to_db(
    rms_frames: NDArray[np.float32], *, floor: float = -40.0
) -> NDArray[np.float32]:
    """RMS frame stack -> dB re the final frame's maximum, clipped at floor."""
    ref = float(rms_frames[-1].max())
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms_frames / ref)
    return np.clip(db, floor, 0.0).astype(np.float32)


# --- weak-field display gain ----------------------------------------------
# A transmitted or shadowed field can sit tens of dB below the incident one
# that sets the colour scale, and a single linear diverging ramp cannot show
# both: give the quiet side half the ramp and the loud side saturates into a
# flat slab, hiding the wavefronts the clip exists to show. Compressing the
# ramp instead (signed log, asinh, gamma) has the same ceiling -- a factor of
# 250 between the two sides is a factor of 250 whatever the transfer curve --
# and it costs the loud side its shape as well.
# The treatment used across the clips is therefore a per-region display gain:
# the quiet region is drawn amplified by a fixed factor picked from the field
# itself, so each region uses the full ramp, the *shape* of both fields
# survives, and the panel states the factor (and the dB it stands for) in
# writing. The measured level annotations stay the physical ones, so nothing
# on screen is silently rescaled.
# The ladder is coarse on purpose (1-1.5-2-3-5-7 per decade): a round factor is
# readable in an annotation and stays put when the field is re-simulated. It
# starts at 5 (14 dB) so a region that already uses a fifth of the ramp is
# left alone -- the caveat such a panel would have to print costs more than
# the contrast it would buy.
_GAIN_STEPS = (
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    70.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0,
)


def _weak_field_gain(
    weak: NDArray[np.float32], vmax: float, *, quantile: float = 0.999
) -> float:
    """Rounded display gain that lifts a weak field onto a readable colour.

    ``weak`` is the quiet region of the field (any shape), ``vmax`` the
    colour-scale half-range the loud region set. The gain is the largest rung
    of :data:`_GAIN_STEPS` at or under the factor that makes the ``quantile``
    amplitude of ``weak`` fill the ramp, so the quiet region gets the whole
    colour scale and at most a thousandth of its samples saturate. A region
    already within 14 dB of the ramp is returned at 1.0 (no gain, and so no
    annotation).
    """
    peak = float(np.quantile(np.abs(weak), quantile))
    if not np.isfinite(peak) or peak <= 0.0:
        return 1.0
    raw = float(vmax) / peak
    return max((g for g in _GAIN_STEPS if g <= raw), default=1.0)


def _gain_note(region: str, gain: float) -> str:
    """English annotation for a region drawn with a display gain.

    Empty for a unit gain (nothing to declare). The dB equivalent rides along
    so the reader can put the compression next to the level annotations.
    """
    if gain <= 1.0:
        return ""
    return f"{region} drawn ×{gain:g} (+{20.0 * np.log10(gain):.0f} dB)"


def _layout_box(artist: Text) -> Bbox:
    """The rendered box of a label: its background patch, or its glyphs.

    For a label with a background patch -- a pill -- this is the patch's own
    extent, which is literal ink: a rounded box has no hidden margin. For
    plain text it is Matplotlib's *layout* box, which reserves vertical room
    for a descender the string may not have: measured against the pixels it
    actually paints, a string with no g/y/p/q/j reads around a third taller
    than its glyphs, nearly all of it below the baseline. A gap measured
    between two such boxes therefore overshoots the gap that is visible --
    close to double it for a small ``gap=`` with a descender-less string on
    either side, measured on a pair at the sizes these clips use. Horizontal
    ink is much closer to this box, ascent and descent being a vertical
    reservation, which is why :func:`_fit_text_x` reads it as ink and
    :func:`_fit_text_below` is the one that has to be read as layout.
    """
    patch = artist.get_bbox_patch()
    if patch is not None:
        return patch.get_window_extent()
    return artist.get_window_extent()


def _settle(fig: Figure) -> None:
    """Draw *fig* until its constrained layout stops moving under it.

    The first draw of a constrained-layout figure is the one that decides
    how wide the axes end up, and extents read from it still describe the
    canvas the solver started from -- a couple of percent out, which is
    exactly the size of the overhangs these helpers exist to measure. The
    second draw sees the settled layout; from there the numbers repeat.
    """
    fig.canvas.draw()
    fig.canvas.draw()


def _text_width_x(fig: Figure, ax: Axes, artist: Text) -> float:
    """The rendered width of *artist*, in x data units of *ax*."""
    _settle(fig)
    box = _layout_box(artist)
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([(box.x0, 0.0), (box.x1, 0.0)])
    return float(abs(x1 - x0))


def _fit_text_below(
    fig: Figure, ax: Axes, artist: Text, other: Text, *, gap: float = 8.0
) -> float:
    """Slide *artist* back along its own baseline until it clears *other*.

    The companion of :func:`_fit_text_x` for a label set along a slope -- a
    rotated one lying on an arc, say -- where a longer string grows into
    whatever sits above it instead of past a spine. Straight down is the
    wrong way out for such a label: the curve it names is tangent to the
    line it is drawn on and falls away from that line on both sides, so
    dropping the label across the tangent walks it into the curve, while
    sliding it back along the same line buys the same height and keeps the
    offset. Returns the applied shift in display pixels (0.0 when the label
    already cleared).

    The boxes are read as layout boxes (see :func:`_layout_box`), so the gap
    that ends up visible is wider than the ``gap`` asked for -- for two
    plain strings, close to twice it. The clips were composed against that
    behaviour, so ``gap`` is a knob for the look, not a measurement of the
    white space.
    """
    _settle(fig)
    top = _layout_box(artist).y1
    floor = _layout_box(other).y0 - gap
    if top <= floor:
        return 0.0
    # The label slides along its own baseline, so the drop it needs
    # (``floor - top``, negative) buys the step down the line through the
    # *signed* sine of the rotation: a label tilted the other way (sin < 0)
    # travels the other way in x for the same drop, which taking the sine's
    # magnitude here would have sent into the text it is clearing.
    slope = np.radians(artist.get_rotation())
    sin_t = float(np.sin(slope))
    if abs(sin_t) < 1e-6:  # a level label can only go down
        dx_px, dy_px = 0.0, floor - top
    else:
        step = (floor - top) / sin_t
        dx_px = step * float(np.cos(slope))
        dy_px = step * sin_t
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0.0, 0.0), (dx_px, dy_px)])
    x, y = artist.get_position()
    artist.set_position((float(x) + (x1 - x0), float(y) + (y1 - y0)))
    return float(np.hypot(dx_px, dy_px))


def _fit_text_x(
    fig: Figure,
    ax: Axes,
    artist: Text,
    x_lo: float,
    x_hi: float,
    *,
    margin: float = 0.0,
) -> float:
    """Slide *artist* along x until its rendered box sits inside the panel.

    An annotation anchored at a data coordinate is placed for the string it
    was written with: the Spanish translation is routinely longer and walks
    out past the spine, taking its pill with it. Measuring the box that is
    actually about to be drawn -- the background patch when the label has
    one, the glyphs otherwise -- and shifting the anchor by the overflow
    keeps every language inside ``[x_lo + margin, x_hi - margin]`` without
    moving the labels that already fit. Returns the applied shift, in data
    units, so a caller can move an arrow or a companion label with it.

    A string wider than the room it is given here cannot be put inside it by
    sliding, and the two overflows would otherwise net out into a shift that
    leaves the label hanging over both edges at once. Such a label is
    anchored at ``x_lo + margin`` instead -- it overflows in reading order,
    where the eye is already going -- and says so on stderr, next to the
    encoder notes, rather than passing silently.
    """
    _settle(fig)
    box = _layout_box(artist)
    inv = ax.transData.inverted()
    (dx0, _), (dx1, _) = inv.transform([(box.x0, 0.0), (box.x1, 0.0)])
    lo, hi = min(dx0, dx1), max(dx0, dx1)
    room_lo, room_hi = x_lo + margin, x_hi - margin
    if hi - lo > room_hi - room_lo:
        print(
            f"note: {artist.get_text()!r} does not fit "
            f"[{room_lo:.3g}, {room_hi:.3g}] even flush against one edge "
            f"({hi - lo:.3g} > {room_hi - room_lo:.3g} data units)",
            file=sys.stderr,
        )
        shift = room_lo - lo
    else:
        shift = max(0.0, room_lo - lo) - max(0.0, hi - room_hi)
    if shift:
        artist.set_x(float(artist.get_position()[0]) + shift)
    return float(shift)


def _anim_speaker(
    ax: Axes,
    x0: float,
    y_mid: float,
    bore: float,
    *,
    tip_inset: float | None = None,
    label_y: float | None = None,
) -> None:
    """Drive loudspeaker of the FDTD tube/duct clips: magnet block plus
    cone, the cone tip on the ``x0`` plane, centred on ``y_mid`` for the
    given ``bore``. The tip stops ``tip_inset`` short of each bore edge
    (3 % of the bore when omitted); a "loudspeaker" caption is drawn at
    ``label_y`` when given (``None`` skips it).
    """
    from matplotlib.patches import Polygon, Rectangle

    if tip_inset is None:
        tip_inset = 0.03 * bore
    magnet_w, cone_w = 0.05, 0.045
    ax.add_patch(
        Rectangle(
            (x0 - magnet_w - cone_w, y_mid - 0.32 * bore),
            magnet_w,
            0.64 * bore,
            facecolor="#9a9a9a",
            edgecolor=COLOR_FG,
            linewidth=0.8,
            zorder=4,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (x0 - cone_w, y_mid - 0.20 * bore),
                (x0 - cone_w, y_mid + 0.20 * bore),
                (x0, y_mid + 0.5 * bore - tip_inset),
                (x0, y_mid - 0.5 * bore + tip_inset),
            ],
            closed=True,
            facecolor="#e8b98a",
            edgecolor=COLOR_FG,
            linewidth=0.8,
            zorder=4,
        )
    )
    if label_y is not None:
        ax.text(
            x0 - magnet_w - cone_w,
            label_y,
            "loudspeaker",
            ha="left",
            va="top",
            fontsize=7.5,
        )
