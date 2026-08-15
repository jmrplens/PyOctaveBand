#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Water over steel: one incident beam, two transmitted waves, two angles."""

from functools import lru_cache
from typing import Any

import numpy as np

from ..media import _ANIM_PILL_BOX, _anim_figure, _render_clip, _translate_str
from ..theme import CMAP_FIELD, COLOR_FG, FIELD_INK, FIELD_STROKE
from ._core import _gain_note, _weak_field_gain

# The water-steel contact of the elastic-waves guide, at the bench scale of
# an immersion ultrasonic inspection: a 200 kHz beam in water gives
# lambda_w = 7.4 mm, lambda_S = 16.0 mm and lambda_P = 29.5 mm in the steel,
# so a 118 x 96 mm domain carries sixteen water wavelengths across and three
# shear wavelengths of depth. Nothing about the picture depends on the
# frequency: the critical angles and the reflection coefficient of a
# half-space are scale-free.
_MC_F0 = 200_000.0
_MC_ANGLES = (10.0, 20.0, 35.0)
# Mesh rule dx = min(smallest geometric dimension / 4, shortest wavelength
# / 8). Water is the slow medium here, so the shortest wavelength is
# lambda_w = 7.4 mm and the bound is 0.93 mm; the grid runs at 0.6 mm,
# 12.3 cells per water wavelength, 26.7 per shear wavelength in the steel.
_MC_DX = 0.0006
_MC_SPONGE = 50                    # 30 mm skirt on all four sides
_MC_SRC_ROW = 56                   # source line, 6 cells clear of the top
#                                    sponge, which eats its upgoing half
# Panel shape. Three panels across an 8 x 4.5 in canvas want a portrait
# field of about 4:5, so the domain is deliberately narrow and deep: a
# 90 x 113 mm view. Depth is where this clip's payload lives -- the 35 deg
# panel has nothing to show *below* the contact and everything to show
# above it, where the incident beam and its total reflection have to
# resolve into a standing pattern (vertical fringe spacing
# lambda_w / (2 cos 35) = 4.5 mm, five of them in the 23 mm of water shown),
# and the 10 and 20 deg panels need enough steel for the departing beams to
# separate (90 mm = 5.6 shear wavelengths).
_MC_WATER = 40                     # cells of water between source and
#                                    contact (3.2 water wavelengths); short
#                                    on purpose, because the beam axis walks
#                                    sideways by _MC_WATER tan(theta) and a
#                                    deeper source would push the 35 deg
#                                    aperture into the left sponge
_MC_STEEL = 150                    # cells of steel below the contact
_MC_NX = 250
_MC_APERTURE = 70                  # source aperture [cells], tapered
_MC_RAMP = 1.5                     # source onset [periods]
_MC_EVERY = 4                      # capture stride [solver steps]
_MC_FPS = 45
#: Colour half-range on the normalisation that puts the incident particle
#: velocity at unit amplitude; 2.0 leaves the standing pattern of the
#: incident plus a nearly total reflection room to breathe.
_MC_VLIM = 2.0


def _mc_geometry() -> tuple[int, int, slice, slice]:
    """(ny, interface row, view rows, view cols)."""
    iface = _MC_SRC_ROW + _MC_WATER
    ny = iface + _MC_STEEL + _MC_SPONGE
    rows = slice(_MC_SRC_ROW + 2, iface + _MC_STEEL)
    cols = slice(_MC_SPONGE, _MC_NX - _MC_SPONGE)
    return ny, iface, rows, cols


def _mc_snell(theta_deg: float) -> tuple[float | None, float | None]:
    """Transmitted P and SV angles [deg], ``None`` where evanescent."""
    from phonometry.simulation import STEEL, WATER

    out: list[float | None] = []
    for c_2 in (STEEL.c_p, STEEL.c_s):
        s = c_2 / WATER.c_p * float(np.sin(np.radians(theta_deg)))
        out.append(float(np.degrees(np.arcsin(s))) if s <= 1.0 else None)
    return out[0], out[1]


def _mc_reflection(theta_deg: float) -> complex:
    """Plane-wave reflection of water over an elastic half-space.

    Brekhovskikh & Godin, *Acoustics of Layered Media I* (Springer 1990),
    Eqs. 4.2.22-4.2.26: Snell angles for the transmitted P and SV waves,
    per-wave impedances ``Z = rho c / cos(theta)`` and

    ``V = (Z_l cos^2(2 theta_t) + Z_t sin^2(2 theta_t) - Z) /
    (Z_l cos^2(2 theta_t) + Z_t sin^2(2 theta_t) + Z)``.

    Past a critical angle the corresponding cosine continues as
    ``i sqrt(sin^2 - 1)``, the decaying evanescent branch, which gives the
    complex ``V`` of Eq. 4.2.29 between the critical angles and ``|V| = 1``
    with a phase beyond the shear critical angle (Eq. 4.2.31). This is the
    same oracle the solver's fluid-solid tests are judged against; it is a
    closed form, not a simulated quantity.
    """
    from phonometry.simulation import STEEL, WATER

    theta = np.radians(theta_deg)
    sin_l = np.complex128(STEEL.c_p / WATER.c_p * np.sin(theta))
    sin_t = np.complex128(STEEL.c_s / WATER.c_p * np.sin(theta))
    cos_l = np.sqrt(1.0 - sin_l**2)
    cos_t = np.sqrt(1.0 - sin_t**2)
    z = WATER.rho * WATER.c_p / np.cos(theta)
    z_l = STEEL.rho * STEEL.c_p / cos_l
    z_t = STEEL.rho * STEEL.c_s / cos_t
    cos_2t = 1.0 - 2.0 * sin_t**2
    sin_2t = 2.0 * sin_t * cos_t
    z_total = z_l * cos_2t**2 + z_t * sin_2t**2
    return complex((z_total - z) / (z_total + z))


def _mc_run(theta_deg: float) -> tuple[Any, Any]:
    """One oblique-beam run; returns (vy frames, frame times).

    A sustained, phase-graded, tapered line source in the water launches a
    beam at ``theta_deg`` onto the contact. Nothing is subtracted here and
    nothing is measured off it: see :func:`_mode_conversion_fields` for why
    this scene shows the regimes but does not price them.
    """
    from phonometry import ElasticFDTD2D
    from phonometry.simulation import STEEL, WATER

    dx = _MC_DX
    ny, iface, rows, cols = _mc_geometry()
    theta = np.radians(theta_deg)
    c_p = np.full((ny, _MC_NX), WATER.c_p)
    c_s = np.zeros((ny, _MC_NX))
    rho = np.full((ny, _MC_NX), WATER.rho)
    c_p[iface:] = STEEL.c_p
    c_s[iface:] = STEEL.c_s
    rho[iface:] = STEEL.rho
    sim = ElasticFDTD2D(c_p, c_s, dx, rho=rho, sponge_width=_MC_SPONGE)
    # Tapered aperture, centred so the beam axis meets the contact at the
    # middle of the frame whatever the incidence: a soft-edged beam instead
    # of a hard-clipped one keeps the edge diffraction out of the picture.
    centre = _MC_NX // 2 - _MC_WATER * float(np.tan(theta))
    src_cols = np.arange(_MC_SPONGE, _MC_NX - _MC_SPONGE)
    u = 2.0 * (src_cols - centre) / _MC_APERTURE
    taper = np.where(np.abs(u) < 1.0, np.cos(0.5 * np.pi * u) ** 2, 0.0)
    x_src = (src_cols + 0.5) * dx
    omega = 2.0 * np.pi * _MC_F0
    phase_x = omega * float(np.sin(theta)) * x_src / WATER.c_p
    ramp_t = _MC_RAMP / _MC_F0
    every = _MC_EVERY
    # Flight: source line to the contact along the beam (the water depth /
    # cos theta), then the slowest transmitted wave to the bottom of the
    # frame (SV at 3200 m/s), then the reflected beam back up to the top of
    # it. The 35 degree panel is the binding one.
    t_in = _MC_WATER * dx / (WATER.c_p * float(np.cos(np.radians(35.0))))
    t_solid = _MC_STEEL * dx / (STEEL.c_s * float(np.cos(np.radians(47.7))))
    t_out = (_MC_WATER - 2) * dx / (WATER.c_p
                                    * float(np.cos(np.radians(35.0))))
    n_active = int(np.ceil((ramp_t + 1.2 * (t_in + t_solid + t_out))
                           / (every * sim.dt)))
    frames: list[Any] = []
    ts: list[float] = []
    for i in range(n_active * every):
        sim.step()
        t = sim.time
        r = min(t / ramp_t, 1.0)
        drive = ((r * r * (3.0 - 2.0 * r)) * taper
                 * np.sin(omega * t - phase_x))
        sim.txx[_MC_SRC_ROW, src_cols] -= drive
        sim.tyy[_MC_SRC_ROW, src_cols] -= drive
        if (i + 1) % every == 0:
            vy = sim.collocated("vy")
            frames.append(vy[rows, cols].astype(np.float32))
            ts.append(t)
    return np.stack(frames), np.asarray(ts)


@lru_cache(maxsize=1)
def _mode_conversion_fields() -> tuple[Any, Any, Any, Any]:
    """The three oblique-incidence runs, cached.

    One sustained 200 kHz beam in water on a steel half-space, at 10, 20
    and 35 degrees: below the P critical angle (14.5 deg), between the two
    (14.5 and 27.5 deg) and beyond the S critical angle.

    **What this scene can and cannot say.** It shows the regimes, and it is
    the only picture of them in the corpus: two departing beams, then one
    with a skin where the other was, then none. It does *not* price them. A
    first cut of this clip ran each incidence twice -- once with the steel
    painted, once with the water continued through it -- and read ``|V|``
    off the difference of two probe traces, the way the solver's validation
    suite does. Watched, it printed 0.878, 0.854 and 0.985 against exact
    values of 0.937, 0.918 and 1.000: six to seven per cent low, and
    consistently so. The cause is the scene, not the solver. A beam narrow
    enough to *see* inside a 90 mm panel is 5.7 water wavelengths across, so
    the probe sits in the source's near field rather than in a formed beam,
    and the incident reading it divides by is too large. Widening the
    aperture until the reading is trustworthy (about 28 wavelengths, from
    the beam's own spreading) needs a panel four times as wide, in which the
    fringes are too fine to read. The two requirements are in direct
    conflict, so the clip keeps the picture and quotes the closed form of
    Brekhovskikh & Godin beside it; the measured column of the guide's table
    comes from the validation suite, which uses a windowed packet and a
    0.12 m standoff and has the room to do it properly.

    Timeline per the animation norms: the capture stride is 4 solver steps
    = 173 ns, i.e. 29 frames per period of the 200 kHz carrier (more than
    twice the 12 the norms require), which at 45 fps puts one carrier cycle
    at 0.64 s of screen time. The window is the 1.5-period onset ramp plus
    1.2 x the flight along the beam -- source line to the contact at 35
    degrees, the slowest transmitted wave from there to the bottom of the
    frame (SV at 3200 m/s) and the reflected beam back up to the top of it.

    Returns the three ``vy`` frame stacks, each normalised to the amplitude
    of its own settled water half, and the frame times.
    """
    stacks: list[Any] = []
    times = np.zeros(0)
    for theta in _MC_ANGLES:
        stack, times = _mc_run(theta)
        n_active = stack.shape[0]
        i_steel = _mc_geometry()[1] - _mc_geometry()[2].start
        settled = slice(3 * n_active // 4, n_active)
        # Normalise on the water half, where the incident beam plus its
        # reflection set the scale: the three panels then share one colour
        # ramp above the contact and can be read against each other.
        water = float(np.quantile(np.abs(stack[settled, :i_steel, :]), 0.999))
        stacks.append(stack / water)
    return stacks[0], stacks[1], stacks[2], times


def animate_elastic_mode_conversion(output_dir: str) -> None:
    """Mode conversion at a water-steel contact (elastic 2D FDTD).

    The one thing that separates the elastic solver from the acoustic one,
    on screen: away from normal incidence a single incident beam in the
    water leaves behind a reflected beam and *two* transmitted waves in the
    steel, compressional and shear, each refracting by Snell's law at its
    own speed and so at its own angle. At 10 degrees both propagate: the
    slower SV wave plunges at 22.1 degrees from the normal and the faster P
    wave, which Snell's law bends further out, at 43.8. At 20
    degrees, past the P critical angle of 14.5, the P wave is gone --
    nothing leaves at its angle, only a skin hugging the contact that dies
    within a wavelength -- and the shear beam alone carries the transmitted
    power, which is why the reflection is 0.918 and not the 1.000 an
    equivalent fluid bottom would return. At 35 degrees, past the shear
    critical angle of 27.5, both are evanescent and the reflection is total
    but phase-shifted. Each panel measures its own |V| by running the scene
    twice, once with the steel and once with the water continued through
    it, and subtracting the two probe traces.
    """
    from matplotlib import patheffects

    T = _translate_str
    f_a, f_b, f_c, times = _mode_conversion_fields()
    frames3 = (f_a, f_b, f_c)
    dx = _MC_DX
    _ny, iface, rows, cols = _mc_geometry()
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    # Frame coordinates in millimetres, with the contact at y = 0 and
    # depth increasing downwards.
    x0 = 0.0
    x1 = (cols.stop - cols.start) * dx * 1e3
    y0 = -(iface - rows.start) * dx * 1e3
    y1 = (rows.stop - iface) * dx * 1e3
    n_active = int(frames3[0].shape[0])
    # The steel side is faint on the water colour scale: water drives steel
    # through a thirtyfold impedance step, so the transmitted beams carry a
    # small particle velocity even where they carry real power. Each panel
    # gets its own steel gain, printed on it, because the three differ by
    # more than a shared gain can span: at 10 deg the transmitted energy
    # leaves in two beams and the local amplitude is small, while at 35 deg
    # nothing leaves and the evanescent field piles up against the contact.
    # Brightness in the steel is therefore not power, and the panel says so.
    i_steel = iface - rows.start
    settled = slice(3 * n_active // 4, n_active)
    gains = [_weak_field_gain(f[settled, i_steel:, :], _MC_VLIM)
             for f in frames3]
    shown = []
    for stack, gain in zip(frames3, gains, strict=True):
        copy = stack.copy()
        copy[:, i_steel:, :] *= gain
        shown.append(copy)

    fig = _anim_figure()
    # Reserve the bottom strip for the figure-level caption and clock:
    # fig.text does not claim space from the constrained layout on its own.
    fig.get_layout_engine().set(rect=(0.0, 0.055, 1.0, 0.945))
    fig.suptitle(T("Mode conversion: water on steel, three incidences "
                   "(elastic 2D FDTD)"))
    axes = fig.subplots(1, 3, sharey=True)
    ims: list[Any] = []
    v_txts: list[Any] = []
    for ax, theta, data, gain in zip(axes, _MC_ANGLES, shown, gains,
                                     strict=True):
        ax.grid(False)
        im = ax.imshow(data[0], origin="upper", extent=(x0, x1, y1, y0),
                       cmap=CMAP_FIELD, vmin=-_MC_VLIM, vmax=_MC_VLIM,
                       aspect="equal", interpolation="bilinear")
        ax.axhline(0.0, color=FIELD_INK, lw=1.0, zorder=4)
        ax.set_title(T(f"$\\theta$ = {theta:.0f}°"), fontsize=10,
                     )
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x [mm]", fontsize=8)
        # Incident beam arrow, drawn from the top of the frame down onto
        # the contact at the beam axis.
        depth = -y0
        ax.annotate("", xy=(0.5 * x1, 0.0),
                    xytext=(0.5 * x1 - depth * float(np.tan(
                        np.radians(theta))), y0),
                    arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                "lw": 1.2}, zorder=5)
        theta_p, theta_s = _mc_snell(theta)
        for name, angle in ((T("P"), theta_p), (T("SV"), theta_s)):
            if angle is None:
                continue
            slope = max(float(np.tan(np.radians(angle))), 1e-6)
            reach = min(0.8 * y1, 0.45 * x1 / slope)
            ax.annotate(
                "", xy=(0.5 * x1 + reach * slope, reach),
                xytext=(0.5 * x1, 0.0),
                arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                            "lw": 1.2, "ls": "--"}, zorder=5)
            # Beside the arrow at 60 % of its length, never at its tip: at
            # 47.7 deg the tip is against the right edge and a label anchored
            # there is cut off.
            ax.text(0.5 * x1 + 0.6 * reach * slope + 2.0, 0.6 * reach,
                    T(f"{name} {angle:.1f}°"), fontsize=7,
                    ha="left", va="center", color=FIELD_INK,
                    path_effects=outline, zorder=6)
        # Inside the steel, which is the region the gain applies to, and in
        # its top-right corner, clear of the incident arrowhead above and of
        # every Snell arrow below.
        ax.text(x1 - 1.5, 6.0, T(_gain_note("steel", gain)), fontsize=6.5,
                ha="right", va="top", color=FIELD_INK,
                path_effects=outline, zorder=6)
        v_txt = ax.text(0.5 * x1, y1 - 1.5, "", ha="center", va="bottom",
                        color="white", fontsize=7.5, zorder=7,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.55,
                              "edgecolor": "none"})
        ims.append(im)
        v_txts.append(v_txt)
    axes[0].set_ylabel(T("depth from the contact [mm]"), fontsize=8)
    axes[0].text(x0 + 1.5, y0 + 1.5, T("water"), fontsize=7.5, ha="left",
                 va="top", color=FIELD_INK, path_effects=outline, zorder=6)
    axes[0].text(x0 + 1.5, 1.5, T("steel"), fontsize=7.5, ha="left",
                 va="top", color=FIELD_INK, path_effects=outline, zorder=6)
    # Two lines, not one: the Spanish of these runs 10 % longer than the
    # English and a single line fills a third of the canvas edge to edge.
    verdicts = [
        T(f"P and SV both propagate\n$|V|$ = "
          f"{abs(_mc_reflection(_MC_ANGLES[0])):.3f}"),
        T(f"P evanescent, SV alone crosses\n$|V|$ = "
          f"{abs(_mc_reflection(_MC_ANGLES[1])):.3f}"),
        T(f"both evanescent\n$|V|$ = "
          f"{abs(_mc_reflection(_MC_ANGLES[2])):.3f}, with a phase"),
    ]
    crit = fig.text(0.5, 0.030,
                    T("critical angles: 14.5° (P), 27.5° (SV). Dashed: the "
                      "Snell direction of each transmitted wave"),
                    ha="center", va="bottom", fontsize=8, color=COLOR_FG)
    t_txt = fig.text(0.988, 0.030, "", ha="right", va="bottom",
                     family="monospace", fontsize=9, color=COLOR_FG)
    reveal = int(0.80 * n_active)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        for i, stack in enumerate(shown):
            ims[i].set_data(stack[kf])
            v_txts[i].set_text(verdicts[i] if k >= reveal else "")
        t_txt.set_text(T(f"$t$ = {times[kf] * 1e6:5.1f} µs"))
        return (*ims, *v_txts, crit, t_txt)

    _render_clip(fig, update, output_dir, "anim_elastic_mode_conversion",
                 frames=n_active + 2 * _MC_FPS, fps=_MC_FPS, gif_fps=5)
