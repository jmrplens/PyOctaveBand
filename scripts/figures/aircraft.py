#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the aircraft guides: certification metrics and airport noise.

The noise of an aircraft as the certification and planning documents define
it: the EPNL chain and its atmospheric absorption, the airport exposure
contours, and the rotorcraft ground effect, flyover and terrain screening of
ICAO Doc 32. Everything here is embedded by a page under ``aircraft/``.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from phonometry._plot.common import (
    format_frequency_axis,
    theme_fill,
    theme_fill_alpha,
    theme_line,
)

from .i18n import _fmt_minus
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_QUATERNARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    save_figure,
    series_colors,
)


def generate_epnl(output_dir: str) -> None:
    """ICAO aircraft-flyover EPNL: PNL/PNLT time history with the 10 dB-down window."""
    print("Generating epnl...")
    from phonometry import aircraft

    k = 41
    dt = 0.5
    idx = np.arange(k)
    # Broadband flyover spectrum with a mid-frequency emphasis, modulated by a
    # Gaussian overall-level envelope; a fan tone in the 2500 Hz band adds a
    # tone correction near the closest-point-of-approach.
    shape = 15.0 * np.exp(
        -((np.log10(aircraft.NOY_BANDS) - np.log10(400.0)) ** 2) / 0.5
    )
    gain = 30.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 5.0**2)) - 5.0
    spectra = (55.0 + shape)[None, :] + gain[:, None]
    spectra[:, 17] += 12.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 6.0**2))
    res = aircraft.effective_perceived_noise_level(spectra, dt)
    kf, kl = res.band_limits

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(res.times[kf], res.times[kl], color=COLOR_TERTIARY, alpha=0.15,
               label="10 dB-down window")
    ax.plot(res.times, res.pnl, color="#8c8c8c", linestyle="--", linewidth=1.4,
            label="PNL")
    ax.plot(res.times, res.pnlt, color=COLOR_PRIMARY, linewidth=2.2, label="PNLT")
    km = int(np.argmax(res.pnlt))
    ax.plot([res.times[km]], [res.pnltm], "o", color=COLOR_SECONDARY, markersize=9,
            label=f"PNLTM = {res.pnltm:.1f} PNdB")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level [PNdB]")
    ax.set_title(
        "ICAO Aircraft Flyover — Effective Perceived Noise Level (Annex 16)",
        pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            f"EPNL = {res.epnl:.1f} EPNdB\n"
            f"$D$ = {_fmt_minus(res.duration_correction, '+.1f')} dB",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "epnl.svg")
    plt.close()


def generate_aircraft_atmospheric_absorption(output_dir: str) -> None:
    """SAE ARP 5534 band vs pure-tone mid-band atmospheric attenuation."""
    print("Generating aircraft_atmospheric_absorption...")
    from phonometry import aircraft

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)  # 50 Hz - 10 kHz thirds
    _fig, ax = plt.subplots(figsize=(10, 6))
    for s, color in ((1000.0, COLOR_SECONDARY), (7620.0, COLOR_PRIMARY)):
        res = aircraft.sae_band_attenuation(
            freqs, s, temperature=25.0, relative_humidity=70.0
        )
        ax.plot(res.frequency, res.band_attenuation, color=color, linewidth=2.0,
                marker="o", markersize=3, label=f"SAE band ({s:.0f} m)")
        # The pure-tone mid-band curve belongs to the band curve above it, so
        # it keeps the colour and gives up weight -- as a shade, not as an
        # opacity, which on the dark page gives up the line as well.
        ax.plot(res.frequency, res.midband_attenuation,
                color=theme_line(color, ax, quiet=0.6), linewidth=1.0,
                linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Attenuation [dB]")
    ax.set_title("Aircraft Atmospheric Absorption (SAE ARP 5534)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.5, 0.95, "25 °C, 70% RH\nsolid: SAE band, dashed: pure-tone mid-band",
            transform=ax.transAxes, va="top", ha="center", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "aircraft_atmospheric_absorption.svg")
    plt.close()


def generate_airport_noise(output_dir: str) -> None:
    """ECAC Doc 29 noise-power-distance interpolation for two power settings."""
    print("Generating airport_noise...")
    from phonometry import aircraft

    # A schematic NPD table (SEL vs slant distance) for two thrust settings.
    powers = [12000.0, 20000.0]
    distances = [200.0, 400.0, 630.0, 1000.0, 2000.0, 4000.0, 6300.0, 10000.0]
    levels = [
        [98.5, 92.0, 88.2, 83.6, 76.8, 69.4, 63.9, 56.8],
        [107.2, 100.9, 97.2, 92.7, 86.0, 78.5, 72.9, 65.6],
    ]
    _fig, ax = plt.subplots(figsize=(10, 6))
    for p, color in ((20000.0, COLOR_PRIMARY), (12000.0, COLOR_SECONDARY)):
        res = aircraft.npd_curve(powers, distances, levels, p)
        ax.plot(res.distance, res.level, color=color, linewidth=2.0,
                label=f"$P$ = {p:.0f} N")
        ax.plot(res.table_distances, res.table_levels, "o", color=color, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Slant distance [m]")
    ax.set_ylabel("Event level [dB]")
    ax.set_title("Noise-Power-Distance Curves (ECAC Doc 29)", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.05, "markers: tabulated NPD nodes\nlines: log-linear interpolation",
            transform=ax.transAxes, va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "airport_noise.svg")
    plt.close()


def generate_airport_contour(output_dir: str) -> None:
    """ECAC Doc 29 single-event SEL contour for a departure flight path."""
    print("Generating airport_contour...")
    from phonometry import aircraft

    powers = [8000.0, 12000.0]
    distances = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0, 7680.0]
    sel = [[98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0, 56.0],
           [104.0, 98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0]]
    lmax = [[94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0, 52.0],
            [100.0, 94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0]]
    vref = 160.0 * 0.514444
    # Departure: ground roll then climb along +x.
    xs = np.linspace(0.0, 18000.0, 40)
    z = np.clip((xs - 1500.0) * 0.11, 0.0, 2500.0)
    power = np.where(xs < 3000.0, 12000.0, 10000.0)
    path = np.column_stack([xs, np.zeros_like(xs), z, power, np.full_like(xs, vref)])
    ground_roll = xs[:-1] < 1500.0  # takeoff roll: segments still on the runway
    res = aircraft.noise_contour(
        path,
        powers,
        distances,
        sel,
        lmax,
        segments=aircraft.FlightSegmentState(ground_roll=ground_roll),
        x=np.linspace(-2500.0, 20000.0, 56),
        y=np.linspace(-6000.0, 6000.0, 44),
    )
    ax = res.plot()
    plt.gcf().set_size_inches(10, 5.5)
    ax.set_title("Aircraft Departure SEL Contour (ECAC Doc 29)", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "airport_contour.png")
    plt.close()


def generate_airport_sor(output_dir: str) -> None:
    """ECAC Doc 29 start-of-roll directivity: the rearward jet/turboprop lobe."""
    print("Generating airport_sor...")
    from phonometry import aircraft

    dsor = 300.0  # < 762 m normalising distance: no distance de-emphasis
    # ΔSOR is defined only in the rearward arc (ψ from 90° abeam to 180° directly
    # behind, symmetric left/right), so only that half-disc is drawn. Azimuth is
    # measured clockwise from the nose; 90°/270° = abeam, 180° = directly behind.
    az = np.linspace(90.0, 270.0, 361)
    psi = np.where(az <= 180.0, az, 360.0 - az)
    jet = np.array(
        [aircraft.start_of_roll_directivity(p, dsor, "jet") for p in psi]
    )
    prop = np.array(
        [aircraft.start_of_roll_directivity(p, dsor, "turboprop") for p in psi]
    )

    # A polar Axes always reserves the full circle's square bounding box, which
    # wastes the upper half for a rearward half-disc. So the half-rose is drawn
    # by hand on a plain, equal-aspect Axes: the radius encodes ΔSOR offset from
    # the −16 dB origin, and the y-limits crop tightly to a wide rectangle.
    r0 = -16.0                                   # radial origin (dB)
    def _xy(a_deg: "np.ndarray | float", rr: "np.ndarray | float") -> tuple[Any, Any]:
        t = np.radians(a_deg)                    # azimuth clockwise from nose (up)
        return rr * np.sin(t), rr * np.cos(t)

    _fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for a in (90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0):  # radial spokes
        sx, sy = _xy(a, np.array([2.0, 16.0]))
        ax.plot(sx, sy, color=COLOR_FG, linestyle="--", linewidth=0.9, alpha=0.28, zorder=0)
    for g in (-4.0, -8.0, -12.0):                # inner radial grid arcs (dB)
        gx, gy = _xy(az, g - r0)
        ax.plot(gx, gy, color=COLOR_FG, linestyle="--", linewidth=0.9, alpha=0.28, zorder=0)
    # The 0 dB (abeam-reference) arc is the outer boundary: draw it solid and bold.
    bx, by = _xy(az, 0.0 - r0)
    ax.plot(bx, by, color=COLOR_FG, linestyle="-", linewidth=1.4, alpha=0.55, zorder=2)
    for a, lbl in ((90.0, "90°\nabeam"), (120.0, "120°"), (150.0, "150°"),
                   (180.0, "180° behind"), (210.0, "150°"), (240.0, "120°"),
                   (270.0, "90°\nabeam")):
        tx, ty = _xy(a, 19.4)
        ax.text(tx, ty, lbl, fontsize=9, color=COLOR_FG, ha="center", va="center")
    for data, color, label in ((jet, COLOR_PRIMARY, "Turbofan jet (Eq. 4-24a)"),
                               (prop, COLOR_SECONDARY, "Turboprop (Eq. 4-24b)")):
        dx, dy = _xy(az, data - r0)
        # The two lobes overlap and the guide arcs run under them, so this one
        # keeps a translucent fill; only its opacity follows the page.
        ax.fill(dx, dy, color=color, alpha=theme_fill_alpha(color, ax), zorder=1)
        ax.plot(dx, dy, color=color, linewidth=2.2, zorder=3, label=label)
    for g in (0.0, -4.0, -8.0, -12.0):           # radial dB labels down the centre
        ax.text(0.6, -(g - r0), _fmt_minus(g, ".0f"), fontsize=8, color=COLOR_FG,
                ha="left", va="center", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-21.0, 21.0)
    ax.set_ylim(-20.2, 1.5)
    ax.set_title(r"Start-of-Roll Directivity $\Delta_{SOR}$ (ECAC Doc 29 §4.5.7)",
                 pad=6)
    ax.text(0.0, 1.0,
            r"radial axis: $\Delta_{SOR}$ [dB] relative to abeam"
            r"  ·  $d_{SOR}$ = 300 m",
            fontsize=9, color=COLOR_FG, ha="center", va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9,
              frameon=False)
    plt.tight_layout()
    save_figure(output_dir, "airport_sor.svg")
    plt.close()


def generate_rotorcraft_ground_effect(output_dir: str) -> None:
    """ECAC Doc 32 ground-effect ΔLg vs frequency for soft vs hard ground."""
    print("Generating rotorcraft_ground_effect...")
    from phonometry import aircraft

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
    hs, hr, dp = 150.0, 1.5, 500.0  # overflight geometry
    grass = aircraft.ground_effect_adjustment(
        freqs, hs, hr, dp, flow_resistivity="D"
    )
    asphalt = aircraft.ground_effect_adjustment(
        freqs, hs, hr, dp, flow_resistivity="G"
    )

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.5)
    ax.plot(freqs, asphalt, color=COLOR_PRIMARY, lw=2.0, marker="o", ms=3,
            label="Hard (asphalt/concrete, class G)")
    ax.plot(freqs, grass, color=COLOR_SECONDARY, lw=2.0, marker="s", ms=3,
            label="Soft (grass/pasture, class D)")
    ax.set_xscale("log")
    ax.set_xlabel("One-third-octave-band centre frequency [Hz]")
    ax.set_ylabel(r"Ground-effect adjustment $\Delta L_\mathrm{g}$ [dB]")
    ax.set_title("Rotorcraft Ground Effect (ECAC Doc 32, Chien-Soroka)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6, which="both")
    ax.set_axisbelow(True)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.98, 0.05,
            rf"$h_\mathrm{{s}}$ = {hs:.0f} m, $h_\mathrm{{r}}$ = {hr:.1f} m, "
            rf"$d_\mathrm{{p}}$ = {dp:.0f} m",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_ground_effect.svg")
    plt.close()



def generate_rotorcraft_flyover_event(output_dir: str) -> None:
    """ECAC Doc 32 single-event LA(t) time history of a level flyover."""
    print("Generating rotorcraft_flyover_event...")
    from phonometry import aircraft

    # Synthetic helicopter-like hemisphere on the standard 31-band 10 deg grid:
    # low-frequency dominated spectrum with a mild forward-lobed directivity.
    freqs = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)     # 10 Hz-10 kHz
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    spectrum = (88.0 - 12.0 * np.log10(freqs / 100.0) ** 2)  # broad LF hump
    direct = -0.045 * np.abs(po - 80.0)                      # forward lobe
    levels = spectrum[None, None, :] + direct[None, :, None] \
        - 0.02 * np.abs(az)[:, None, None]
    hemisphere = aircraft.RotorcraftHemisphere(freqs, az, po, levels)

    speed = 30.87                                            # 60 kt, in m/s
    t = np.arange(0.0, 130.01, 0.5)
    track = np.column_stack([np.zeros_like(t), speed * (t - 65.0),
                             np.full_like(t, 150.0)])
    event = aircraft.rotorcraft_event_level(
        [hemisphere], [speed], [0.0], t, track, (120.0, 0.0),
        ground=aircraft.RotorcraftGround(flow_resistivity="D"))

    _fig, ax = plt.subplots(figsize=(10, 6))
    # The subscripts are descriptive (A weighting, A-weighted Slow maximum), so
    # they are upright -- and they have to match the library renderer of the
    # same quantity, `phonometry._plot.aircraft`, which this figure sits beside
    # in the documentation.
    ax.plot(event.times, event.a_levels, color=COLOR_PRIMARY, lw=2.0,
            label=r"Received level $L_\mathrm{A}(t)$")
    k = int(np.argmax(event.a_levels))
    ax.plot(event.times[k], event.la_max, "o", color=COLOR_SECONDARY, ms=7,
            label=rf"$L_\mathrm{{ASmax}}$ = {event.la_max:.1f} dB(A)")
    window = event.a_levels >= event.la_max - 10.0
    idx = np.nonzero(window)[0]
    ax.axvspan(event.times[idx[0]], event.times[idx[-1]], zorder=0,
               color=theme_fill(COLOR_PRIMARY, ax),
               label="10 dB-down window")
    ax.axhline(event.la_max - 10.0, color=COLOR_FG, lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel("Recorded time [s]")
    ax.set_ylabel("A-weighted sound pressure level [dB(A)]")
    ax.set_title("Rotorcraft Flyover Time History (ECAC Doc 32)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.05,
            f"SEL {event.sel:.1f} dB(A)  ·  EPNL {event.epnl:.1f} EPNdB\n"
            "level flyover, 60 kt, 150 m, 120 m sideline, grass",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_flyover_event.svg")
    plt.close()



def generate_rotorcraft_terrain_screening(output_dir: str) -> None:
    """ECAC Doc 32 / NORAH2 terrain screening: section geometry and adjustment."""
    print("Generating rotorcraft_terrain_screening...")
    from phonometry import aircraft

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
    d = np.array([0.0, 150.0, 260.0, 300.0, 340.0, 420.0, 600.0])
    z = np.array([0.0, 4.0, 48.0, 62.0, 40.0, 8.0, 2.0])
    src = (0.0, 90.0)                                       # helicopter
    rcv = (600.0, 2.0 + 1.2)  # microphone at 1.2 m
    res = aircraft.terrain_screening_adjustment(
        freqs, src, rcv, d, z, flow_resistivity="D"
    )
    flat = aircraft.ground_effect_adjustment(
        freqs, src[1], 1.2, rcv[0], flow_resistivity="D"
    )

    _fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                  gridspec_kw={"height_ratios": [1.1, 1.0]})
    res.plot(ax=ax)   # the user-facing section geometry
    ax.set_title("Rotorcraft Terrain Screening (ECAC Doc 32 / NORAH2)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    ax2.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.5)
    ax2.plot(freqs, flat, color=COLOR_TERTIARY, lw=1.6, ls="--", marker="s", ms=3,
             label="Flat ground (no hill)")
    ax2.plot(freqs, res.adjustment, color=COLOR_PRIMARY, lw=2.0, marker="o",
             ms=3, label="Screened by the hill (Eq. 45-47)")
    ax2.set_xscale("log")
    ax2.set_xlabel("One-third-octave-band centre frequency [Hz]")
    ax2.set_ylabel("Ground and screening adjustment [dB]")
    ax2.grid(color=COLOR_GRID, linestyle="--", alpha=0.6, which="both")
    ax2.set_axisbelow(True)
    format_frequency_axis(ax2, float(freqs.min()), float(freqs.max()))
    ax2.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_terrain_screening.svg")
    plt.close()


def generate_rotorcraft_insertion_loss(output_dir: str) -> None:
    """Pure diffraction attenuation ΔLd against path difference δ (Eq. 42-44)."""
    print("Generating rotorcraft_insertion_loss...")
    from phonometry import aircraft

    # A single low ridge edge, shallow enough that Ch < 1 still bites on the
    # lowest band (Eq. 43: Ch = min(f h0/250, 1)) while the three higher
    # bands already saturate at Ch = 1. The range stays inside what a 2.5 m
    # edge can plausibly deliver (a knife-edge geometry with a source and
    # receiver both a few edge-heights away tops out around 1 m of path
    # difference): only the 4 kHz band actually reaches the 25 dB cap here.
    h0 = 2.5
    bands = np.array([63.0, 250.0, 1000.0, 4000.0])
    band_labels = ["63 Hz", "250 Hz", "1 kHz", "4 kHz"]
    delta = np.linspace(-0.4, 1.5, 441)
    ld = np.array(
        [
            aircraft.diffraction_attenuation(bands, float(d), edge_height=h0)
            for d in delta
        ]
    )   # (len(delta), len(bands))

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(float(delta.min()), 0.0, color=theme_fill(COLOR_MUTED, ax),
               zorder=0, label="Below line of sight")
    colours = series_colors(len(bands))
    for colour, label, values in zip(colours, band_labels, ld.T, strict=True):
        ax.plot(delta, values, color=colour, linewidth=2.0, label=label,
                zorder=3)
    ax.axvline(0.0, color=COLOR_FG, linewidth=1.0, linestyle="--", alpha=0.6)
    # Above the band curves (zorder=3): otherwise the flat 25 dB plateau of
    # the 4 kHz curve, drawn at the same height, paints over the cap line.
    ax.axhline(25.0, color=COLOR_FG, linewidth=1.2, linestyle=":", alpha=0.9,
               label="25 dB cap (§A.4.5)", zorder=4)
    # Grazing incidence is 10*Ch*lg(3): the classical 4.8 dB only where
    # Ch = 1 (250 Hz, 1 kHz, 4 kHz here), not a frequency-independent
    # constant -- the 63 Hz curve, whose Ch = 0.63, crosses at 3.0 dB.
    grazing = 10.0 * np.log10(3.0)
    grazing_63 = 10.0 * min(63.0 * h0 / 250.0, 1.0) * np.log10(3.0)
    # Set low, in the empty wedge under the 63 Hz curve: at the range this
    # figure now plots, higher up (where it used to sit) the 250 Hz curve
    # cuts straight through the text.
    ax.annotate(
        r"$10\,C_\mathrm{h}\,\mathrm{lg}\,3$ at grazing incidence ($\delta = 0$):" "\n"
        rf"{grazing:.1f} dB where $C_\mathrm{{h}}$ = 1, {grazing_63:.1f} dB at 63 Hz",
        xy=(0.0, grazing), xytext=(0.45, 1.6),
        fontsize=9, color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.0})
    ax.set_xlabel(r"Path difference $\delta$ [m]")
    ax.set_ylabel(r"Diffraction attenuation $\Delta L_\mathrm{d}$ [dB]")
    ax.set_title("Rotorcraft Diffraction Attenuation vs Path Difference "
                 "(ECAC Doc 32 / NORAH2)", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.98, 0.05, f"edge height $h_0$ = {h0:.1f} m, single edge",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_insertion_loss.svg")
    plt.close()


def generate_anp_npd(output_dir: str) -> None:
    """NPD curves of a real ANP aircraft, one per tabulated thrust setting."""
    print("Generating anp_npd...")
    from phonometry import aircraft

    # A real fleet entry rather than a schematic table: the 747-100 is one of
    # the aircraft whose ANP record carries a fixed-point profile, so the same
    # aircraft can illustrate the profile figure below.
    airframe = aircraft.load_anp_database().aircraft("747100")
    curves = airframe.npd_curves("D", "SEL")

    _fig, ax = plt.subplots(figsize=(10, 6))
    curves.plot(ax=ax)
    ax.set_title(f"ANP NPD Curves - {airframe.description} (SEL, departure)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.text(0.02, 0.06,
            f"power parameter: {airframe.power_parameter}\n"
            "markers: tabulated NPD nodes",
            transform=ax.transAxes, va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "anp_npd.svg")
    plt.close()


def generate_anp_profile(output_dir: str) -> None:
    """Default fixed-point departure trajectory of a real ANP aircraft."""
    print("Generating anp_profile...")
    from phonometry import aircraft

    airframe = aircraft.load_anp_database().aircraft("747100")
    profile = airframe.profile("D", stage_length=1)

    _fig, ax = plt.subplots(figsize=(10, 6))
    profile.plot(ax=ax)
    ax.set_title(f"ANP Default Departure Profile - {airframe.description}",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.06,
            f"stage length {profile.stage_length}, "
            f"{profile.path.shape[0]} fixed points",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "anp_profile.svg")
    plt.close()


# --------------------------------------------------------------------------- #
# The ECAC Doc 29 single event, opened up
# --------------------------------------------------------------------------- #

#: The schematic NPD table and departure path the airport-noise guide works
#: with, so the contour, the segment breakdown and the corrections figure all
#: describe the same aeroplane on the same runway.
_DOC29_POWERS = [8000.0, 12000.0]
_DOC29_DISTANCES = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0, 7680.0]
_DOC29_SEL = [[98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0, 56.0],
              [104.0, 98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0]]
_DOC29_LMAX = [[94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0, 52.0],
               [100.0, 94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0]]
_VREF_MS = 160.0 * 0.514444


def _doc29_departure() -> "tuple[Any, Any, Any]":
    """The guide's departure: the ``(N, 5)`` path, the ground-roll mask, x."""
    xs = np.linspace(0.0, 18000.0, 40)
    z = np.clip((xs - 1500.0) * 0.11, 0.0, 2500.0)
    power = np.where(xs < 3000.0, 12000.0, 10000.0)
    path = np.column_stack([xs, np.zeros_like(xs), z, power,
                            np.full_like(xs, _VREF_MS)])
    return path, xs[:-1] < 1500.0, xs


def generate_airport_segment_breakdown(output_dir: str) -> None:
    """ECAC Doc 29: which segments of a departure carry the SEL at one receiver."""
    print("Generating airport_segment_breakdown...")
    from phonometry import aircraft

    path, ground_roll, _xs = _doc29_departure()
    obs = [3000.0, 500.0, 1.2]
    res = aircraft.event_level(
        path,
        obs,
        _DOC29_POWERS,
        _DOC29_DISTANCES,
        _DOC29_SEL,
        _DOC29_LMAX,
        segments=aircraft.FlightSegmentState(ground_roll=ground_roll),
    )
    seg = np.asarray(res.segment_levels, dtype=np.float64)
    top = int(np.argmax(seg))

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax)
    # The ground-roll segments are hatched in place: the bars are already the
    # result's own, so only their fill changes.
    for i, patch in enumerate(ax.patches[:seg.size]):
        if ground_roll[i]:
            patch.set_hatch("///")
            patch.set_edgecolor(COLOR_FG)
        if i == top:
            patch.set_facecolor(COLOR_SECONDARY)
    ax.set_ylim(0.0, float(seg.max()) + 12.0)
    ax.set_title("Doc 29 Segment Contributions at One Receiver",
                 pad=12)
    ax.annotate(f"closest segment: {seg[top]:.1f} dB",
                xy=(top, seg[top]), xytext=(top + 5.0, seg[top] + 7.0),
                color=COLOR_FG, fontsize=9,
                arrowprops={"arrowstyle": "->", "color": COLOR_FG, "lw": 1.2})
    ax.text(0.98, 0.60,
            f"receiver 3 000 m along track, 500 m to the side, 1.2 m up.\n"
            f"total SEL {float(res.level):.1f} dB; the closest segment alone\n"
            f"carries {100.0 * 10.0 ** (seg[top] / 10.0) / np.sum(10.0 ** (seg / 10.0)):.0f} % of the energy. "
            "Hatched: the take-off ground roll.",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y")
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "airport_segment_breakdown.svg")
    plt.close()


def generate_airport_segment_corrections(output_dir: str) -> None:
    """ECAC Doc 29: how large each per-segment correction is, and where it bites."""
    print("Generating airport_segment_corrections...")
    from phonometry import aircraft

    _fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (ax_i, ax_l), (ax_f, ax_v) = axes

    # (a) engine installation: the depression-angle directivity, per mounting.
    phi = np.linspace(0.0, 180.0, 361)
    for mounting, color, label in (("wing", COLOR_PRIMARY, "Wing-mounted"),
                                   ("fuselage", COLOR_SECONDARY, "Fuselage-mounted"),
                                   ("propeller", COLOR_TERTIARY, "Propeller")):
        di = np.array(
            [aircraft.engine_installation_correction(p, mounting) for p in phi]
        )
        ax_i.plot(phi, di, color=color, lw=2.0, label=label)
    ax_i.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.4)
    ax_i.set(xlabel=r"Depression angle $\varphi$ [°]",
             ylabel=r"$\Delta I(\varphi)$ [dB]", xlim=(0.0, 180.0))
    ax_i.set_title("(a) Engine installation (Eq. 4-15/4-16)", fontsize=11,
                   )
    ax_i.legend(loc="lower center", fontsize=8)

    # (b) lateral attenuation: elevation angle, for three lateral offsets.
    beta = np.linspace(0.0, 90.0, 361)
    for ell, color in ((100.0, COLOR_TERTIARY), (300.0, COLOR_SECONDARY),
                       (914.0, COLOR_PRIMARY)):
        lam = np.array([aircraft.lateral_attenuation(b, ell) for b in beta])
        ax_l.plot(beta, lam, color=color, lw=2.0, label=rf"$\ell$ = {ell:.0f} m")
    ax_l.axvline(50.0, color=COLOR_FG, lw=1.0, ls="--", alpha=0.5)
    ax_l.text(51.0, 8.0, r"$\Lambda = 0$ above 50°", fontsize=8, color=COLOR_FG)
    ax_l.set(xlabel=r"Elevation angle $\beta$ [°]",
             ylabel=r"$\Lambda(\beta, \ell)$ subtracted [dB]", xlim=(0.0, 90.0))
    ax_l.set_title("(b) Lateral attenuation (Eq. 4-18/4-19)", fontsize=11,
                   )
    ax_l.legend(loc="upper right", fontsize=8)

    # (c) noise fraction: the share of the infinite path a finite segment gives.
    d_lambda = (
        (2.0 / np.pi)
        * _VREF_MS
        * 10.0
        ** (
            (
                float(
                    aircraft.npd_level(
                        _DOC29_POWERS,
                        _DOC29_DISTANCES,
                        _DOC29_SEL,
                        12000.0,
                        526.0,
                    )[0]
                )
                - float(
                    aircraft.npd_level(
                        _DOC29_POWERS,
                        _DOC29_DISTANCES,
                        _DOC29_LMAX,
                        12000.0,
                        526.0,
                    )[0]
                )
            )
            / 10.0
        )
    )
    for length, color, label in ((464.0, COLOR_PRIMARY, r"$\lambda$ = 464 m"),
                                 (2000.0, COLOR_SECONDARY, r"$\lambda$ = 2 000 m")):
        frac = np.linspace(-1.0, 2.0, 601)
        df = np.array(
            [
                aircraft.noise_fraction(f * length, length, d_lambda)
                for f in frac
            ]
        )
        ax_f.plot(frac, df, color=color, lw=2.0, label=label)
    ax_f.axvspan(0.0, 1.0, color=theme_fill(COLOR_PRIMARY, ax_f), zorder=0)
    ax_f.text(0.5, -13.0, "observer alongside", ha="center", fontsize=8,
              color=COLOR_FG)
    ax_f.set(xlabel=r"$q/\lambda$", ylabel=r"$\Delta F$ [dB]", ylim=(-15.0, 1.0))
    ax_f.set_title(rf"(c) Noise fraction, $d_\lambda$ = {d_lambda:.0f} m (Eq. 4-20)",
                   fontsize=11)
    ax_f.legend(loc="lower right", fontsize=8)

    # (d) duration correction, with the impedance adjustment for scale.
    v = np.linspace(50.0, 130.0, 401)
    dv = np.array([aircraft.duration_correction(_VREF_MS, s) for s in v])
    ax_v.plot(v, dv, color=COLOR_PRIMARY, lw=2.0)
    ax_v.axvline(_VREF_MS, color=COLOR_FG, lw=1.0, ls="--", alpha=0.5)
    ax_v.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.4)
    # Below the impedance box: at the old 1.4 the box's backdrop painted
    # over the tail of this reading.
    ax_v.text(_VREF_MS + 1.5, 1.05, r"$V_{\mathrm{ref}}$ = 82.3 m/s (160 kn)",
              fontsize=8, color=COLOR_FG)
    ax_v.set(xlabel=r"Segment speed $V_{\mathrm{seg}}$ [m/s]",
             ylabel=r"$\Delta V$ [dB]")
    ax_v.set_title("(d) Duration correction (Eq. 4-14)", fontsize=11,
                   )
    ax_v.text(
        0.98,
        0.96,
        "impedance adjustment (Eq. 4-6/4-7), for scale:\n"
        f"15 °C, 101.3 kPa: {_fmt_minus(aircraft.impedance_adjustment(), '+.2f')} dB\n"
        f"30 °C, 101.3 kPa: {_fmt_minus(aircraft.impedance_adjustment(30.0), '+.2f')} dB\n"
        f"15 °C, 95.0 kPa: "
        f"{_fmt_minus(aircraft.impedance_adjustment(15.0, 95.0), '+.2f')} dB",
        transform=ax_v.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6},
    )

    for ax in axes.ravel():
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "airport_segment_corrections.svg")
    plt.close()


def generate_anp_contour(output_dir: str) -> None:
    """The same Doc 29 contour, driven by a real ANP record instead of a table."""
    print("Generating anp_contour...")
    from phonometry import aircraft

    record = aircraft.load_anp_database().aircraft("747100")
    profile = record.profile("D", stage_length=1)
    x = np.linspace(-2000.0, 12000.0, 40)
    y = np.linspace(-3000.0, 3000.0, 30)
    contour = record.noise_contour("D", x=x, y=y)
    flyover = record.event_level([3000.0, 500.0, 0.0], "D")

    _fig, ax = plt.subplots(figsize=(10, 5.5))
    contour.plot(ax=ax)
    # The plot works in kilometres, and the default profile runs 39 km
    # downrange: the overlays are clipped to the grid the contour covers.
    roll_end = float(profile.path[int(np.sum(profile.ground_roll)), 0]) / 1000.0
    inside = profile.path[:, 0] <= x.max()
    ax.plot([0.0, roll_end], [0.0, 0.0], color=COLOR_FG, lw=6.0,
            solid_capstyle="butt", zorder=3, label="take-off ground roll")
    ax.plot(profile.path[inside, 0] / 1000.0, profile.path[inside, 1] / 1000.0,
            color=COLOR_FG, lw=1.4, ls="--", zorder=3, label="default ground track")
    ax.plot([3.0], [0.5], "o", color=COLOR_SECONDARY, ms=8, zorder=4,
            label=f"event_level receiver: SEL {float(flyover.level):.1f} dB")
    ax.set_xlim(x.min() / 1000.0, x.max() / 1000.0)
    ax.set_ylim(y.min() / 1000.0, y.max() / 1000.0)
    ax.set_title(f"ANP Departure SEL Contour - {record.description}",
                 pad=12)
    ax.legend(loc="lower left", fontsize=8)
    ax.text(0.98, 0.94,
            f"stage length {profile.stage_length}, "
            f"{profile.path.shape[0]} fixed points, "
            f"{profile.path[-1, 2]:.0f} m at the last one",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "anp_contour.svg")
    plt.close()


# --------------------------------------------------------------------------- #
# The ECAC Doc 32 hemisphere method, opened up
# --------------------------------------------------------------------------- #

def _synthetic_hemisphere(masked: bool = False) -> "Any":
    """A helicopter-like hemisphere on the Doc 32 grid, for the figures.

    The library implements the method and ships no hemisphere database, so
    the rotorcraft figures build one: 31 bands from 10 Hz to 10 kHz on the
    19 x 19 grid of Doc 32 Appendix A, with a rearward-biased directivity and
    a mid-frequency forward-down lobe standing in for blade-vortex
    interaction. With ``masked`` the bins a ground array cannot reach are
    ``NaN``, as a measured hemisphere's are.
    """
    from phonometry import aircraft

    freqs = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)   # 10 Hz - 10 kHz
    az = np.arange(-90.0, 91.0, 10.0)                      # 19 azimuths
    po = np.arange(0.0, 181.0, 10.0)                       # 19 polar angles
    spectrum = 88.0 - 12.0 * np.log10(freqs / 100.0) ** 2
    rear = -0.030 * np.abs(po - 110.0)                     # rearward bias
    lateral = -0.020 * np.abs(az)                          # loudest beneath
    band = np.exp(-((np.log10(freqs) - np.log10(630.0)) ** 2) / 0.08)
    bvi = (7.0 * np.exp(-((po - 140.0) ** 2) / (2 * 16.0**2))[None, :, None]
           * np.exp(-(az**2) / (2 * 34.0**2))[:, None, None]
           * band[None, None, :])
    levels = (spectrum[None, None, :] + rear[None, :, None]
              + lateral[:, None, None] + bvi)
    if masked:
        unseen = ((np.abs(az)[:, None] > 60.0)
                  | (po[None, :] < 40.0) | (po[None, :] > 140.0))
        levels = np.where(unseen[:, :, None], np.nan, levels)
    return aircraft.RotorcraftHemisphere(freqs, az, po, levels)


def _level_flyover_track() -> "tuple[Any, Any, float]":
    """A 60 kt level flyover at 150 m: times, positions and the airspeed."""
    speed = 30.87                                          # 60 kt, in m/s
    t = np.arange(0.0, 130.01, 0.5)
    track = np.column_stack([np.zeros_like(t), speed * (t - 65.0),
                             np.full_like(t, 150.0)])
    return t, track, speed


def generate_rotorcraft_hemisphere(output_dir: str) -> None:
    """ECAC Doc 32: the source hemisphere, in section and over the whole grid."""
    print("Generating rotorcraft_hemisphere...")
    from phonometry import aircraft

    h = _synthetic_hemisphere(masked=True)
    freqs = np.asarray(h.frequencies)
    az = np.asarray(h.azimuth)
    po = np.asarray(h.polar)

    _fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))
    for band, color in ((100.0, COLOR_TERTIARY), (630.0, COLOR_PRIMARY),
                        (4000.0, COLOR_SECONDARY)):
        h.plot(ax=ax, band=band, color=color, lw=2.0)
    # The result's own xlabel and legend, restated with the composed angles
    # so both panels of this figure write them the same way. The legend keeps
    # the band frequencies exactly as the library snapped them (631, 3981).
    ax.set_xlabel(r"Polar angle $\theta$ [°]  (0° forward → 180° rearward)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,
              [lbl.replace("(φ = 0°)", r"($\varphi$ = 0°)") for lbl in labels],
              loc="upper right", fontsize="small")
    ax.axvspan(40.0, 140.0, color=theme_fill(COLOR_PRIMARY, ax), zorder=0)
    ax.text(90.0, ax.get_ylim()[0] + 2.0, "measured polar band", ha="center",
            fontsize=9, color=COLOR_FG)
    ax.set_title(r"Fore-aft section ($\varphi$ = 0°)", pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # The whole grid for the most directive band -- the one whose measured
    # cells span the widest range, which is what the hemisphere exists for.
    lv = np.asarray(h.levels)
    spread_per_band = np.nanmax(lv, axis=(0, 1)) - np.nanmin(lv, axis=(0, 1))
    idx = int(np.nanargmax(spread_per_band))
    grid = lv[:, :, idx]
    filled = np.array(
        [
            [
                aircraft.hemisphere_source_level(h, float(a), float(p))[idx]
                for p in po
            ]
            for a in az
        ]
    )
    cs = ax2.contourf(po, az, filled, levels=12, cmap="viridis")
    # The measured patch as an outline rather than a hatch: everything
    # outside it is the gap-filling of Eq. 6/7, not data.
    ax2.plot([40.0, 140.0, 140.0, 40.0, 40.0], [-60.0, -60.0, 60.0, 60.0, -60.0],
             color=COLOR_FG, lw=1.6, ls="--")
    ax2.text(90.0, 66.0, "measured coverage", ha="center", fontsize=9,
             color=COLOR_FG)
    ax2.figure.colorbar(cs, ax=ax2, label="Source level at 60 m [dB]")
    spread = float(np.nanmax(grid) - np.nanmin(grid))
    ax2.set_xlabel(r"Polar angle $\theta$ [°]")
    ax2.set_ylabel(r"Azimuth $\varphi$ [°]")
    ax2.set_title(f"{freqs[idx]:.0f} Hz band, the most directive one",
                  pad=10)
    ax2.text(0.02, 0.04,
             f"{spread:.1f} dB between the loudest and quietest measured cell;\n"
             "outside the dashed patch the field is gap-filled, not measured",
             transform=ax2.transAxes, va="bottom", fontsize=9,
             bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.75})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_hemisphere.svg")
    plt.close()


def generate_rotorcraft_hover_ring(output_dir: str) -> None:
    """NORAH2 guidance §A.3.5: the hover ring and the Table 3 derived sources."""
    print("Generating rotorcraft_hover_ring...")
    from phonometry import aircraft

    # A ground-ring measurement with the shape of a real campaign: one
    # 31-band spectrum per 30 deg bearing, reduced to the 70 m polar
    # distance, peaking at the nose and falling to the tail with the
    # starboard half a few decibels above the port half.
    freqs = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)   # 10 Hz - 10 kHz
    bearings = np.arange(-180.0, 151.0, 30.0)              # ring closes at 180
    spectrum = 86.0 - 10.0 * np.log10(freqs / 160.0) ** 2
    directivity = (3.5 * np.cos(np.radians(bearings))
                   + 3.0 * np.exp(-((bearings - 120.0) ** 2) / (2.0 * 40.0**2)))
    ring = spectrum[None, :] + directivity[:, None]
    hige = aircraft.hover_ring_hemisphere(freqs, bearings, ring, distance=70.0)

    k = int(np.argmin(np.abs(np.asarray(freqs) - 315.0)))
    _fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    closed_bearings = np.append(bearings, 180.0)           # periodic closure
    closed_ring = np.append(ring[:, k], ring[0, k])
    ax.plot(closed_bearings, closed_ring, marker="o", markersize=4.5,
            lw=1.8, color=COLOR_PRIMARY)
    ax.axvline(0.0, color=COLOR_MUTED, lw=1.0, ls=":")
    ax.text(4.0, float(closed_ring.min()) + 0.3, "nose", fontsize=9,
            color=COLOR_FG, ha="left")
    ax.set_xticks(np.arange(-180.0, 181.0, 60.0))
    ax.set_xlabel("Ring bearing [°]  (0° nose, +90° starboard)")
    ax.set_ylabel("Band level at 70 m [dB]")
    ax.set_title("In-ground-hover ring, 315 Hz band", pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    theta = np.linspace(0.0, 180.0, 361)
    conditions = (
        ("HIGE (from the ring)", hige, COLOR_PRIMARY, "-"),
        ("HOGE (+12 dB)",
         aircraft.hover_derived_hemisphere(hige, "out_of_ground_hover"),
         COLOR_SECONDARY, "--"),
        ("Full-rpm idle (−2.5 dB)",
         aircraft.hover_derived_hemisphere(hige, "full_rpm_idle"),
         COLOR_TERTIARY, "--"),
        ("Reduced-rpm idle (−12 dB)",
         aircraft.hover_derived_hemisphere(hige, "reduced_rpm_idle"),
         COLOR_QUATERNARY, "--"),
    )
    top = 0.0
    for label, hemisphere, color, style in conditions:
        section = np.array(
            [
                aircraft.hemisphere_source_level(hemisphere, 0.0, t)[k]
                for t in theta
            ]
        )
        top = max(top, float(section.max()))
        ax2.plot(theta, section, color=color, ls=style, lw=1.8, label=label)
    ax2.set_ylim(top=top + 8.0)          # room for the legend above the curves
    ax2.set_xticks(np.arange(0.0, 181.0, 30.0))
    ax2.set_xlabel(r"Polar angle $\theta$ [°]  (0° forward → 180° rearward)")
    ax2.set_ylabel("Source level at 70 m [dB]")
    ax2.set_title(r"Derived hover and idle sources (Table 3, $\varphi$ = 0°)",
                  pad=10)
    ax2.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.text(0.02, 0.03,
             r"constant directivity in $\varphi$: each $\theta$" "\n"
             r"reads the ring at $\pm\theta$",
             transform=ax2.transAxes, va="bottom", fontsize=9,
             bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.75})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_hover_ring.svg")
    plt.close()


def generate_rotorcraft_contour(output_dir: str) -> None:
    """ECAC Doc 32 SEL footprint, over uniform grass and across a hard strip."""
    print("Generating rotorcraft_contour...")
    from phonometry import aircraft

    h = _synthetic_hemisphere()
    t, track, speed = _level_flyover_track()
    x = np.linspace(-1500.0, 1500.0, 61)
    y = np.linspace(-2000.0, 2000.0, 81)
    # A 600 m hard strip (an apron, or a runway) crossing the track, as a
    # per-grid-point flow resistivity map of shape (len(y), len(x)).
    sigma = np.where(np.abs(y)[:, None] < 300.0, 20.0e6,
                     np.full((y.size, x.size), 200.0e3))

    _fig, axes = plt.subplots(1, 2, figsize=(11, 6.4))
    for ax, ground, title in (
            (axes[0], aircraft.RotorcraftGround(flow_resistivity="D"),
             "Uniform pasture (class D)"),
            (axes[1], aircraft.RotorcraftGround(flow_resistivity=sigma),
             "A 600 m hard strip across it")):
        res = aircraft.rotorcraft_noise_contour(
            [h],
            [speed],
            [0.0],
            t,
            track,
            x=x,
            y=y,
            metric="exposure",
            ground=ground,
        )
        res.plot(ax=ax)
        # The contour plot works in kilometres; so must its overlays.
        ax.plot(track[:, 0] / 1000.0, track[:, 1] / 1000.0, color=COLOR_FG,
                lw=1.4, ls="--", zorder=3, label="track")
        ax.plot([0.120], [0.0], "o", color=COLOR_SECONDARY, ms=7, zorder=4,
                label="the event receiver")
        ax.set_xlim(x.min() / 1000.0, x.max() / 1000.0)
        ax.set_ylim(y.min() / 1000.0, y.max() / 1000.0)
        ax.set_title(title, pad=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[1].axhline(-0.3, color=COLOR_FG, lw=1.0, ls=":")
    axes[1].axhline(0.3, color=COLOR_FG, lw=1.0, ls=":")
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_contour.svg")
    plt.close()


def generate_rotorcraft_mean_ground_plane(output_dir: str) -> None:
    """The mean ground plane, the equivalent heights, and what they change."""
    print("Generating rotorcraft_mean_ground_plane...")
    from phonometry import aircraft

    # An undulating section that slopes gently up towards the receiver, with
    # the line of sight to a helicopter 150 m up never broken.
    d = np.linspace(0.0, 800.0, 33)
    z = 0.035 * d + 6.0 * np.sin(d / 90.0) + 2.5 * np.sin(d / 31.0)
    plane = aircraft.mean_ground_plane(d, z)
    src = (0.0, 60.0)                        # helicopter on final approach
    rcv = (800.0, float(z[-1]) + 1.2)        # microphone 1.2 m over the terrain

    def _equivalent(point: "tuple[float, float]") -> float:
        """Height measured orthogonally to the fitted plane, floored at 0.1 m."""
        m, c = float(plane.slope), float(plane.intercept)
        drop = abs(point[1] - m * point[0] - c) / float(np.hypot(1.0, m))
        return max(drop, 0.1)

    hs_eq, hr_eq = _equivalent(src), _equivalent(rcv)
    hs_true, hr_true = src[1] - float(z[0]), 1.2

    _fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={"height_ratios": [1.0, 1.0]})
    plane.plot(ax=ax)
    # The result's own legend entry, restated with the composed slope symbol
    # so it matches the equivalent-height annotations beside it.
    for line in ax.get_lines():
        if line.get_label().startswith("Mean ground plane (a = "):
            line.set_label(line.get_label().replace("(a = ", r"($a$ = "))
    # The result fills the ground under the terrain with a very pale wash;
    # repaint it as an opaque theme fill so it stays legible on both pages.
    for coll in ax.collections:
        coll.set_alpha(None)
        coll.set_facecolor(theme_fill(COLOR_PRIMARY, ax))
    ax.plot([src[0]], [src[1]], "o", color=COLOR_SECONDARY, ms=9, label="source")
    ax.plot([rcv[0]], [rcv[1]], "o", color=COLOR_PRIMARY, ms=7, label="receiver")
    m, c = float(plane.slope), float(plane.intercept)
    for (px, py), color, label in ((src, COLOR_SECONDARY,
                                    rf"$h_\mathrm{{s}}$ = {hs_eq:.0f} m"),
                                   (rcv, COLOR_PRIMARY,
                                    rf"$h_\mathrm{{r}}$ = {hr_eq:.1f} m")):
        # Foot of the perpendicular onto the fitted plane.
        fx = (px + m * (py - c)) / (1.0 + m**2)
        ax.plot([px, fx], [py, m * fx + c], color=color, lw=1.6, ls="-")
        ax.plot([px, px], [py, m * px + c], color=color, lw=1.0, ls=":")
        side = -1 if px > 0.5 * float(d[-1]) else 1
        ax.annotate(label, xy=((px + fx) / 2, (py + m * fx + c) / 2),
                    xytext=(10 * side, 8), textcoords="offset points",
                    fontsize=9, color=color,
                    ha="left" if side > 0 else "right")
    ax.set_title("Mean Ground Plane and Equivalent Heights (ECAC Doc 32 / NORAH2)",
                 pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)
    true = aircraft.ground_effect_adjustment(freqs, hs_true, hr_true, rcv[0],
                                             flow_resistivity="D")
    equiv = aircraft.ground_effect_adjustment(freqs, hs_eq, hr_eq, rcv[0],
                                              flow_resistivity="D")
    ax2.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.5)
    ax2.plot(freqs, true, color=COLOR_TERTIARY, lw=1.6, ls="--", marker="s",
             ms=3, label=f"true heights ({hs_true:.0f} m, {hr_true:.1f} m)")
    ax2.plot(freqs, equiv, color=COLOR_PRIMARY, lw=2.0, marker="o", ms=3,
             label=f"equivalent heights ({hs_eq:.0f} m, {hr_eq:.1f} m)")
    ax2.set_xscale("log")
    ax2.set_xlabel("One-third-octave-band centre frequency [Hz]")
    ax2.set_ylabel(r"Ground-effect adjustment $\Delta L_\mathrm{g}$ [dB]")
    ax2.grid(color=COLOR_GRID, linestyle="--", alpha=0.6, which="both")
    ax2.set_axisbelow(True)
    format_frequency_axis(ax2, float(freqs.min()), float(freqs.max()))
    ax2.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_mean_ground_plane.svg")
    plt.close()


def generate_rotorcraft_flight_conditions(output_dir: str) -> None:
    """The plane the hemispheres are interpolated in, raw and normalised."""
    print("Generating rotorcraft_flight_conditions...")
    from scipy.spatial import Delaunay

    from phonometry import aircraft

    # A Doc 32 §4.1 condition matrix for a light twin: descent at 3° steps and
    # four airspeeds, climb at Vy, and level flight at 0.9 VH with the three
    # recommended increments.
    descent_v = np.array([30.9, 36.0, 41.2, 46.3])         # 60, 70, 80, 90 kt
    descent_g = np.array([-3.0, -6.0, -9.0, -12.0])
    speeds = list(np.repeat(descent_v, descent_g.size))
    angles = list(np.tile(descent_g, descent_v.size))
    speeds += [33.4, 33.4, 64.8, 70.0, 57.1, 49.4]
    angles += [6.0, 9.0, 0.0, 0.0, 0.0, 0.0]
    v = np.asarray(speeds)
    g = np.asarray(angles)
    inside, outside = (38.0, -7.0), (60.0, -11.0)

    scale = 2.0
    dv, dg = v.max() - v.min(), g.max() - g.min()
    raw = np.column_stack([v, g])
    norm = np.column_stack([v / dv, scale * g / dg])
    raw_tri = Delaunay(raw).simplices
    w_raw = aircraft.flight_condition_weights(v, g, *inside, triangles=raw_tri)
    w_norm = aircraft.flight_condition_weights(v, g, *inside)
    w_out = aircraft.flight_condition_weights(v, g, *outside)

    _fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.6))
    for axis, pts, tri, weights, qi, qo, title, xlabel, ylabel in (
            (ax, raw, raw_tri, w_raw, inside, outside,
             r"Raw ($V$, $\gamma$) plane — pass it as triangles=",
             "Airspeed $V$ [m/s]", r"Path angle $\gamma$ [°]"),
            (ax2, norm, Delaunay(norm).simplices, w_norm,
             (inside[0] / dv, scale * inside[1] / dg),
             (outside[0] / dv, scale * outside[1] / dg),
             "Normalised plane — the library default",
             r"$V/\Delta V$", r"$F_\mathrm{fc}\,\gamma/\Delta\gamma$")):
        axis.triplot(pts[:, 0], pts[:, 1], tri, color=COLOR_MUTED, lw=0.9,
                     zorder=1)
        # The simplex that encloses the query, and the three weights it blends.
        corner = np.array([pts[i] for i, _ in weights])
        axis.fill(corner[:, 0], corner[:, 1], color=COLOR_TERTIARY, alpha=0.18,
                  zorder=0)
        axis.plot(pts[:, 0], pts[:, 1], "o", color=COLOR_PRIMARY, ms=5,
                  zorder=2, label="database conditions")
        for (i, weight), (px, py) in zip(weights, corner, strict=True):
            axis.annotate(f"{weight:.2f}", xy=(px, py), xytext=(7, 6),
                          textcoords="offset points", fontsize=9,
                          color=COLOR_TERTIARY, fontweight="bold")
        axis.plot([qi[0]], [qi[1]], "*", color=COLOR_TERTIARY, ms=16, zorder=3,
                  label="query inside the hull")
        axis.plot([qo[0]], [qo[1]], "X", color=COLOR_SECONDARY, ms=11, zorder=3,
                  label="query outside: nearest, unblended")
        axis.plot([qo[0], pts[w_out[0][0], 0]], [qo[1], pts[w_out[0][0], 1]],
                  color=COLOR_SECONDARY, lw=1.2, ls=":", zorder=2)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.set_title(title, pad=10)
        axis.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", fontsize=8)
    # Three short lines: at the old two-line width the box's first line ran
    # under the legend in the Spanish variant.
    ax2.text(0.98, 0.96,
             rf"$V$ = {inside[0]:.0f} m/s, "
             rf"$\gamma$ = {_fmt_minus(inside[1], '.0f')}°: the two" "\n"
             "triangulations blend\n"
             f"{', '.join(str(i) for i, _ in w_raw)} and "
             f"{', '.join(str(i) for i, _ in w_norm)}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.75})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_flight_conditions.svg")
    plt.close()


def generate_rotorcraft_kinematics(output_dir: str) -> None:
    """Why a radar track is smoothed before it is differentiated."""
    print("Generating rotorcraft_kinematics...")
    from scipy.interpolate import UnivariateSpline

    from phonometry import aircraft

    # A 60 kt approach with a 600 m radius turn, sampled the way radar
    # delivers it (1 s, with position noise) and then spline-resampled to the
    # 0.5 s cadence the guidance recommends.
    rng = np.random.default_rng(20260808)
    t_true = np.arange(0.0, 120.001, 0.1)
    speed, radius, sigma = 30.87, 600.0, 4.0
    turn = np.clip((t_true - 40.0) / (radius / speed), 0.0, np.pi / 2)
    x = np.cumsum(np.cos(turn)) * speed * 0.1
    y = np.cumsum(np.sin(turn)) * speed * 0.1
    z = 300.0 - 1.2 * t_true
    t_radar = np.arange(0.0, 120.001, 1.0)
    raw = np.column_stack([np.interp(t_radar, t_true, x),
                           np.interp(t_radar, t_true, y),
                           np.interp(t_radar, t_true, z)])
    raw = raw + rng.normal(0.0, sigma, raw.shape)          # radar position noise
    t_fine = np.arange(0.0, 120.001, 0.5)
    smooth = np.column_stack([
        UnivariateSpline(t_radar, raw[:, k], s=t_radar.size * sigma**2)(t_fine)
        for k in range(3)])
    designed = float(np.degrees(np.arctan((speed**2 / radius) / 9.80665)))

    _fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))
    for axis, times, pos, title in (
            (ax, t_radar, raw, "Raw radar track, 1 s cadence"),
            (ax2, t_fine, smooth, "Smoothing spline resampled to 0.5 s")):
        kin = aircraft.flight_path_kinematics(times, pos)
        kin.plot(ax=axis)
        axis.set_title(title, pad=10)
        axis.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
        axis.set_axisbelow(True)
        # Both scales get a band clear of the traces at each end: at the
        # natural limits the dotted bank angle - drawn on the twin axis, hence
        # after the legend - climbed through the legend interior and touched
        # its entry text, and the note at the foot lay over the speed minima.
        # The Spanish legend is half the panel wide, so the note cannot share
        # the top band with it and stays where it is.
        for scale in axis.get_shared_x_axes().get_siblings(axis):
            low, high = scale.get_ylim()
            span = high - low
            scale.set_ylim(low - 0.16 * span, high + 0.34 * span)
        peak = float(np.nanmax(np.abs(kin.bank_angle)))
        axis.text(0.02, 0.04,
                  rf"peak $|\Phi|$ = {peak:.0f}°"
                  f"  ·  the turn asks for {designed:.1f}°",
                  transform=axis.transAxes, va="bottom", fontsize=9,
                  bbox={"boxstyle": "round", "facecolor": COLOR_GRID,
                        "alpha": 0.75})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_kinematics.svg")
    plt.close()
