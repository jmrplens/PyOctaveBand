#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Drawing parts more than one diagram is built from.

A part earns its place here when diagrams of different guides draw the
same object: the oblique-projected box that stands for a machine under a
measurement surface, the curved rotation arrow of a turntable, and the
pieces every structure-borne rig is assembled from (spring, accelerometer,
exciter, motion arrows, plates). They take the same ``(s, th)`` the
builders do, so a part reads inside a builder exactly like a canvas
primitive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .canvas import SVG, Theme


def _box_solid(
    s: SVG,
    th: Theme,
    bx: float,
    gy: float,
    hw: float,
    dp: float,
    ht: float,
    stroke: str = "",
    fill: str = "",
) -> None:
    """Small oblique-projected box standing on the plane at ``(bx, gy)``.

    ``hw`` is the front half-width, ``ht`` the height, ``dp`` the depth
    (oblique offset). Draws the top, front and right visible faces.
    """
    stroke = stroke or th.primary
    fill = fill or th.panel
    dxo, dyo = dp * 0.72, dp * 0.55
    ftl, ftr = (bx - hw, gy - ht), (bx + hw, gy - ht)
    fbr = (bx + hw, gy)
    btl = (bx - hw + dxo, gy - ht - dyo)
    btr = (bx + hw + dxo, gy - ht - dyo)
    bbr = (bx + hw + dxo, gy - dyo)
    # top face (lighter) then right face (shaded) then front face
    s.path(
        f"M {ftl[0]} {ftl[1]} L {ftr[0]} {ftr[1]} L {btr[0]} {btr[1]} "
        f"L {btl[0]} {btl[1]} Z",
        fill=fill,
        stroke=stroke,
        sw=1.8,
    )
    s.path(
        f"M {ftr[0]} {ftr[1]} L {fbr[0]} {fbr[1]} L {bbr[0]} {bbr[1]} "
        f"L {btr[0]} {btr[1]} Z",
        fill=th.panel,
        stroke=stroke,
        sw=1.8,
    )
    s.rect(bx - hw, gy - ht, 2 * hw, ht, fill, stroke, sw=1.8)


def _box_wire(
    s: SVG,
    th: Theme,
    bx: float,
    gy: float,
    hw: float,
    dp: float,
    ht: float,
    color: str,
    dash: str = "7,5",
) -> None:
    """Dashed oblique wireframe box (measurement surface) on the plane."""
    dxo, dyo = dp * 0.72, dp * 0.55
    fbl, fbr = (bx - hw, gy), (bx + hw, gy)
    ftl, ftr = (bx - hw, gy - ht), (bx + hw, gy - ht)
    bbl = (bx - hw + dxo, gy - dyo)
    bbr = (bx + hw + dxo, gy - dyo)
    btl = (bx - hw + dxo, gy - ht - dyo)
    btr = (bx + hw + dxo, gy - ht - dyo)
    for a, b in (
        (fbl, fbr),
        (fbr, ftr),
        (ftr, ftl),
        (ftl, fbl),
        (bbl, bbr),
        (bbr, btr),
        (btr, btl),
        (btl, bbl),
        (fbl, bbl),
        (fbr, bbr),
        (ftl, btl),
        (ftr, btr),
    ):
        s.line(a[0], a[1], b[0], b[1], color, 1.5, dash=dash)


def _rot_arrow(
    s: SVG,
    cx: float,
    cy: float,
    r: float,
    a0_deg: float,
    a1_deg: float,
    color: str,
    sw: float = 2.0,
    ry: float | None = None,
) -> None:
    """Curved rotation indicator: an elliptical arc with a head at ``a1``."""
    import math

    ryy = r if ry is None else ry
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    x0, y0 = cx + r * math.cos(a0), cy + ryy * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + ryy * math.sin(a1)
    large = 1 if abs(a1_deg - a0_deg) > 180 else 0
    sweep = 1 if a1_deg > a0_deg else 0
    s.path(
        f"M {x0:.1f} {y0:.1f} A {r:.1f} {ryy:.1f} 0 {large} {sweep} {x1:.1f} {y1:.1f}",
        stroke=color,
        sw=sw,
    )
    tang = a1 + (math.pi / 2 if sweep else -math.pi / 2)
    L, W = 10.0, 4.4
    bx, by = x1 - L * math.cos(tang), y1 - L * math.sin(tang)
    px, py = -math.sin(tang), math.cos(tang)
    s.path(
        f"M {x1:.1f} {y1:.1f} L {bx + W * px:.1f} {by + W * py:.1f} "
        f"L {bx - W * px:.1f} {by - W * py:.1f} Z",
        fill=color,
    )


# ---------------------------------------------------------------------------
# Shared structure-borne rig parts (ISO 9052-1 / ISO 7626 / ISO 10846 /
# EN 15657 / EN 12354-5 diagrams)
# ---------------------------------------------------------------------------


def _spring_v(
    s: SVG,
    x: float,
    y1: float,
    y2: float,
    color: str,
    coils: int = 4,
    width: float = 13.0,
    sw: float = 2.2,
) -> None:
    """Vertical zig-zag spring between (x, y1) and (x, y2), straight leads."""
    lead = min(10.0, (y2 - y1) * 0.12)
    n = 2 * coils
    step = (y2 - y1 - 2 * lead) / n
    d = [f"M {x:.1f} {y1:.1f}", f"L {x:.1f} {y1 + lead:.1f}"]
    for k in range(1, n):
        dx = width if k % 2 else -width
        d.append(f"L {x + dx:.1f} {y1 + lead + k * step:.1f}")
    d.append(f"L {x:.1f} {y2 - lead:.1f}")
    d.append(f"L {x:.1f} {y2:.1f}")
    s.path(" ".join(d), stroke=color, sw=sw)


def _accel(s: SVG, x: float, y: float, size: float = 14.0) -> None:
    """Accelerometer block standing on the surface point (x, y)."""
    th = s.th
    s.rect(x - size / 2, y - size, size, size, th.secondary, th.fg, rx=2.5, sw=1.3)
    s.line(x, y - size, x, y - size - 8, th.fg, 1.3)


def _exciter(
    s: SVG,
    x: float,
    y: float,
    stinger: float = 22.0,
    w: float = 74.0,
    h: float = 48.0,
    up: bool = False,
) -> None:
    """Electrodynamic exciter body with a stinger driving the point (x, y).

    ``up=False`` hangs the body above the drive point (stinger down);
    ``up=True`` stands it below the drive point (stinger up).
    """
    th = s.th
    sgn = 1.0 if up else -1.0
    s.line(x, y, x, y + sgn * stinger, th.fg, 2.2)
    top = y + stinger if up else y - stinger - h
    s.rect(x - w / 2, top, w, h, th.panel, th.primary, rx=9, sw=2)


def _motion_arrows(
    s: SVG, x: float, cy: float, half: float, color: str, sw: float = 2.0
) -> None:
    """Up-down double arrow marking a vibratory motion at x, centred on cy."""
    s.arrow(x, cy, x, cy - half, color, sw)
    s.arrow(x, cy, x, cy + half, color, sw)


def _plate_top(
    s: SVG, th: Theme, x0: float, y0: float, w: float, dp: float, t: float
) -> None:
    """Horizontal slab in oblique projection: top face, front face and the
    visible right end face. ``(x0, y0)`` = front-left corner of the top face,
    ``w`` its width, ``dp`` its oblique depth and ``t`` its thickness.
    """
    dxo, dyo = dp * 0.72, dp * 0.55
    s.path(
        f"M {x0} {y0} L {x0 + w} {y0} L {x0 + w + dxo} {y0 - dyo} "
        f"L {x0 + dxo} {y0 - dyo} Z",
        fill=th.panel,
        stroke=th.fg,
        sw=1.8,
    )
    s.rect(x0, y0, w, t, th.panel, th.fg, sw=1.8)
    s.path(
        f"M {x0 + w} {y0} L {x0 + w} {y0 + t} L {x0 + w + dxo} {y0 + t - dyo} "
        f"L {x0 + w + dxo} {y0 - dyo} Z",
        fill=th.panel,
        stroke=th.fg,
        sw=1.8,
    )


def _plate_up(
    s: SVG, th: Theme, x0: float, y_base: float, t: float, h: float, dp: float
) -> None:
    """Vertical slab in oblique projection standing on ``y_base``: front
    edge face, top face and the visible right-hand surface.
    """
    dxo, dyo = dp * 0.72, dp * 0.55
    y_top = y_base - h
    s.rect(x0, y_top, t, h, th.panel, th.fg, sw=1.8)
    s.path(
        f"M {x0} {y_top} L {x0 + t} {y_top} L {x0 + t + dxo} {y_top - dyo} "
        f"L {x0 + dxo} {y_top - dyo} Z",
        fill=th.panel,
        stroke=th.fg,
        sw=1.8,
    )
    s.path(
        f"M {x0 + t} {y_top} L {x0 + t} {y_base} "
        f"L {x0 + t + dxo} {y_base - dyo} L {x0 + t + dxo} {y_top - dyo} Z",
        fill=th.panel,
        stroke=th.fg,
        sw=1.8,
    )


def _accel_wall(s: SVG, x: float, y: float, size: float = 13.0) -> None:
    """Accelerometer block mounted on a wall surface seen in oblique view."""
    th = s.th
    s.rect(x - size / 2, y - size / 2, size, size, th.secondary, th.fg, rx=2.5, sw=1.3)
    s.line(x + size / 2, y, x + size / 2 + 8, y, th.fg, 1.3)
