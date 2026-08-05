#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What more than one wave-field clip needs: capture, dB and gain.

The continuous-wave capture loop and the running-RMS to dB conversion every
FDTD clip post-processes its frames with, the weak-field display gain the
clips that put a loud and a quiet region in the same colour ramp share, and
the loudspeaker drawn on the clips that drive a bore.
"""

from typing import Any

import numpy as np

from ..theme import COLOR_FG


def _fdtd_cw_capture(
        sim: Any, frequency: float, every: int, n_frames: int,
        decimate: int = 2) -> tuple[Any, Any, Any, Any]:
    """Drive a CW simulation and capture instantaneous + running-RMS frames.

    The running mean square uses a two-period time constant, so the RMS map
    (the lobe/shadow pattern) builds up as the field settles. Returns the
    float32 frame stacks (decimated), the frame times and the final
    full-resolution RMS map for physics probes.
    """
    beta = float(np.exp(-sim.dt * frequency / 2.0))
    ms = np.zeros_like(sim.p)
    ps: list[Any] = []
    rs: list[Any] = []
    ts: list[float] = []
    for _ in range(every * n_frames):
        sim.step()
        ms = beta * ms + (1.0 - beta) * sim.p**2
        if sim.n % every == 0 and len(ps) < n_frames:
            ps.append(sim.p[::decimate, ::decimate].astype(np.float32))
            rs.append(np.sqrt(ms[::decimate, ::decimate]).astype(np.float32))
            ts.append(sim.time)
    return np.stack(ps), np.stack(rs), np.asarray(ts), np.sqrt(ms)


def _rms_to_db(rms_frames: Any, *, floor: float = -40.0) -> Any:
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
_GAIN_STEPS = (5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0,
               200.0, 300.0, 500.0, 700.0, 1000.0)


def _weak_field_gain(weak: Any, vmax: float, *,
                     quantile: float = 0.999) -> float:
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


def _anim_speaker(ax: Any, x0: float, y_mid: float, bore: float, *,
                  tip_inset: float | None = None,
                  label_y: float | None = None) -> None:
    """Drive loudspeaker of the FDTD tube/duct clips: magnet block plus
    cone, the cone tip on the ``x0`` plane, centred on ``y_mid`` for the
    given ``bore``. The tip stops ``tip_inset`` short of each bore edge
    (3 % of the bore when omitted); a "loudspeaker" caption is drawn at
    ``label_y`` when given (``None`` skips it)."""
    from matplotlib.patches import Polygon, Rectangle

    if tip_inset is None:
        tip_inset = 0.03 * bore
    magnet_w, cone_w = 0.05, 0.045
    ax.add_patch(Rectangle((x0 - magnet_w - cone_w, y_mid - 0.32 * bore),
                           magnet_w, 0.64 * bore, facecolor="#9a9a9a",
                           edgecolor=COLOR_FG, linewidth=0.8, zorder=4))
    ax.add_patch(Polygon(
        [(x0 - cone_w, y_mid - 0.20 * bore),
         (x0 - cone_w, y_mid + 0.20 * bore),
         (x0, y_mid + 0.5 * bore - tip_inset),
         (x0, y_mid - 0.5 * bore + tip_inset)],
        closed=True, facecolor="#e8b98a", edgecolor=COLOR_FG,
        linewidth=0.8, zorder=4))
    if label_y is not None:
        ax.text(x0 - magnet_w - cone_w, label_y, "loudspeaker",
                ha="left", va="top", fontsize=7.5)
