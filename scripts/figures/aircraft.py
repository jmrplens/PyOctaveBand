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
)

from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    save_figure,
)


def generate_epnl(output_dir: str) -> None:
    """ICAO aircraft-flyover EPNL: PNL/PNLT time history with the 10 dB-down window."""
    print("Generating epnl...")
    from phonometry import NOY_BANDS, effective_perceived_noise_level

    k = 41
    dt = 0.5
    idx = np.arange(k)
    # Broadband flyover spectrum with a mid-frequency emphasis, modulated by a
    # Gaussian overall-level envelope; a fan tone in the 2500 Hz band adds a
    # tone correction near the closest-point-of-approach.
    shape = 15.0 * np.exp(-((np.log10(NOY_BANDS) - np.log10(400.0)) ** 2) / 0.5)
    gain = 30.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 5.0**2)) - 5.0
    spectra = (55.0 + shape)[None, :] + gain[:, None]
    spectra[:, 17] += 12.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 6.0**2))
    res = effective_perceived_noise_level(spectra, dt)
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
        fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            f"EPNL = {res.epnl:.1f} EPNdB\nD = {res.duration_correction:+.1f} dB",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "epnl.svg")
    plt.close()


def generate_aircraft_atmospheric_absorption(output_dir: str) -> None:
    """SAE ARP 5534 band vs pure-tone mid-band atmospheric attenuation."""
    print("Generating aircraft_atmospheric_absorption...")
    from phonometry import sae_band_attenuation

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)  # 50 Hz - 10 kHz thirds
    _fig, ax = plt.subplots(figsize=(10, 6))
    for s, color in ((1000.0, COLOR_SECONDARY), (7620.0, COLOR_PRIMARY)):
        res = sae_band_attenuation(freqs, s, temperature=25.0, relative_humidity=70.0)
        ax.plot(res.frequency, res.band_attenuation, color=color, linewidth=2.0,
                marker="o", markersize=3, label=f"SAE band ({s:.0f} m)")
        ax.plot(res.frequency, res.midband_attenuation, color=color, linewidth=1.0,
                linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Attenuation [dB]")
    ax.set_title("Aircraft Atmospheric Absorption (SAE ARP 5534)",
                 fontweight="bold", pad=12)
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
    from phonometry import npd_curve

    # A schematic NPD table (SEL vs slant distance) for two thrust settings.
    powers = [12000.0, 20000.0]
    distances = [200.0, 400.0, 630.0, 1000.0, 2000.0, 4000.0, 6300.0, 10000.0]
    levels = [
        [98.5, 92.0, 88.2, 83.6, 76.8, 69.4, 63.9, 56.8],
        [107.2, 100.9, 97.2, 92.7, 86.0, 78.5, 72.9, 65.6],
    ]
    _fig, ax = plt.subplots(figsize=(10, 6))
    for p, color in ((20000.0, COLOR_PRIMARY), (12000.0, COLOR_SECONDARY)):
        res = npd_curve(powers, distances, levels, p)
        ax.plot(res.distance, res.level, color=color, linewidth=2.0,
                label=f"P = {p:.0f} N")
        ax.plot(res.table_distances, res.table_levels, "o", color=color, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Slant distance [m]")
    ax.set_ylabel("Event level [dB]")
    ax.set_title("Noise-Power-Distance Curves (ECAC Doc 29)", fontweight="bold", pad=12)
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
    from phonometry import FlightSegmentState, noise_contour

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
    res = noise_contour(path, powers, distances, sel, lmax,
                        segments=FlightSegmentState(ground_roll=ground_roll),
                        x=np.linspace(-2500.0, 20000.0, 56), y=np.linspace(-6000.0, 6000.0, 44))
    ax = res.plot()
    plt.gcf().set_size_inches(10, 5.5)
    ax.set_title("Aircraft Departure SEL Contour (ECAC Doc 29)", fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "airport_contour.png")
    plt.close()


def generate_airport_sor(output_dir: str) -> None:
    """ECAC Doc 29 start-of-roll directivity: the rearward jet/turboprop lobe."""
    print("Generating airport_sor...")
    from phonometry import start_of_roll_directivity

    dsor = 300.0  # < 762 m normalising distance: no distance de-emphasis
    # ΔSOR is defined only in the rearward arc (ψ from 90° abeam to 180° directly
    # behind, symmetric left/right), so only that half-disc is drawn. Azimuth is
    # measured clockwise from the nose; 90°/270° = abeam, 180° = directly behind.
    az = np.linspace(90.0, 270.0, 361)
    psi = np.where(az <= 180.0, az, 360.0 - az)
    jet = np.array([start_of_roll_directivity(p, dsor, "jet") for p in psi])
    prop = np.array([start_of_roll_directivity(p, dsor, "turboprop") for p in psi])

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
        ax.text(0.6, -(g - r0), f"{g:.0f}", fontsize=8, color=COLOR_FG, ha="left",
                va="center", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-21.0, 21.0)
    ax.set_ylim(-20.2, 1.5)
    ax.set_title("Start-of-Roll Directivity ΔSOR (ECAC Doc 29 §4.5.7)",
                 fontweight="bold", pad=6)
    ax.text(0.0, 1.0, "radial axis: ΔSOR [dB] relative to abeam  ·  dSOR = 300 m",
            fontsize=9, color=COLOR_FG, ha="center", va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9,
              frameon=False)
    plt.tight_layout()
    save_figure(output_dir, "airport_sor.svg")
    plt.close()


def generate_rotorcraft_ground_effect(output_dir: str) -> None:
    """ECAC Doc 32 ground-effect ΔLg vs frequency for soft vs hard ground."""
    print("Generating rotorcraft_ground_effect...")
    from phonometry import ground_effect_adjustment

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
    hs, hr, dp = 150.0, 1.5, 500.0                         # overflight geometry
    grass = ground_effect_adjustment(freqs, hs, hr, dp, flow_resistivity="D")
    asphalt = ground_effect_adjustment(freqs, hs, hr, dp, flow_resistivity="G")

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.5)
    ax.plot(freqs, asphalt, color=COLOR_PRIMARY, lw=2.0, marker="o", ms=3,
            label="Hard (asphalt/concrete, class G)")
    ax.plot(freqs, grass, color=COLOR_SECONDARY, lw=2.0, marker="s", ms=3,
            label="Soft (grass/pasture, class D)")
    ax.set_xscale("log")
    ax.set_xlabel("One-third-octave-band centre frequency [Hz]")
    ax.set_ylabel("Ground-effect adjustment ΔLg [dB]")
    ax.set_title("Rotorcraft Ground Effect (ECAC Doc 32, Chien-Soroka)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6, which="both")
    ax.set_axisbelow(True)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.98, 0.05, f"source {hs:.0f} m, receiver {hr:.1f} m, offset {dp:.0f} m",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_ground_effect.svg")
    plt.close()



def generate_rotorcraft_flyover_event(output_dir: str) -> None:
    """ECAC Doc 32 single-event LA(t) time history of a level flyover."""
    print("Generating rotorcraft_flyover_event...")
    from phonometry import (
        RotorcraftGround,
        RotorcraftHemisphere,
        rotorcraft_event_level,
    )

    # Synthetic helicopter-like hemisphere on the standard 31-band 10 deg grid:
    # low-frequency dominated spectrum with a mild forward-lobed directivity.
    freqs = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)     # 10 Hz-10 kHz
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    spectrum = (88.0 - 12.0 * np.log10(freqs / 100.0) ** 2)  # broad LF hump
    direct = -0.045 * np.abs(po - 80.0)                      # forward lobe
    levels = spectrum[None, None, :] + direct[None, :, None] \
        - 0.02 * np.abs(az)[:, None, None]
    hemisphere = RotorcraftHemisphere(freqs, az, po, levels)

    speed = 30.87                                            # 60 kt, in m/s
    t = np.arange(0.0, 130.01, 0.5)
    track = np.column_stack([np.zeros_like(t), speed * (t - 65.0),
                             np.full_like(t, 150.0)])
    event = rotorcraft_event_level(
        [hemisphere], [speed], [0.0], t, track, (120.0, 0.0),
        ground=RotorcraftGround(flow_resistivity="D"))

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(event.times, event.a_levels, color=COLOR_PRIMARY, lw=2.0,
            label="Received level $L_A(t)$")
    k = int(np.argmax(event.a_levels))
    ax.plot(event.times[k], event.la_max, "o", color=COLOR_SECONDARY, ms=7,
            label=f"$L_{{ASmax}}$ = {event.la_max:.1f} dB(A)")
    window = event.a_levels >= event.la_max - 10.0
    idx = np.nonzero(window)[0]
    ax.axvspan(event.times[idx[0]], event.times[idx[-1]], zorder=0,
               color=theme_fill(COLOR_PRIMARY, ax),
               label="10 dB-down window")
    ax.axhline(event.la_max - 10.0, color=COLOR_FG, lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel("Recorded time [s]")
    ax.set_ylabel("A-weighted sound pressure level [dB(A)]")
    ax.set_title("Rotorcraft Flyover Time History (ECAC Doc 32)",
                 fontweight="bold", pad=12)
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
    from phonometry import ground_effect_adjustment, terrain_screening_adjustment

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
    d = np.array([0.0, 150.0, 260.0, 300.0, 340.0, 420.0, 600.0])
    z = np.array([0.0, 4.0, 48.0, 62.0, 40.0, 8.0, 2.0])
    src = (0.0, 90.0)                                       # helicopter
    rcv = (600.0, 2.0 + 1.2)                                # microphone at 1.2 m
    res = terrain_screening_adjustment(freqs, src, rcv, d, z, flow_resistivity="D")
    flat = ground_effect_adjustment(freqs, src[1], 1.2, rcv[0], flow_resistivity="D")

    _fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                  gridspec_kw={"height_ratios": [1.1, 1.0]})
    res.plot(ax=ax)   # the user-facing section geometry
    ax.set_title("Rotorcraft Terrain Screening (ECAC Doc 32 / NORAH2)",
                 fontweight="bold", pad=12)
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


def generate_anp_npd(output_dir: str) -> None:
    """NPD curves of a real ANP aircraft, one per tabulated thrust setting."""
    print("Generating anp_npd...")
    from phonometry import load_anp_database

    # A real fleet entry rather than a schematic table: the 747-100 is one of
    # the aircraft whose ANP record carries a fixed-point profile, so the same
    # aircraft can illustrate the profile figure below.
    aircraft = load_anp_database().aircraft("747100")
    curves = aircraft.npd_curves("D", "SEL")

    _fig, ax = plt.subplots(figsize=(10, 6))
    curves.plot(ax=ax)
    ax.set_title(f"ANP NPD Curves - {aircraft.description} (SEL, departure)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.text(0.02, 0.06,
            f"power parameter: {aircraft.power_parameter}\n"
            "markers: tabulated NPD nodes",
            transform=ax.transAxes, va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "anp_npd.svg")
    plt.close()


def generate_anp_profile(output_dir: str) -> None:
    """Default fixed-point departure trajectory of a real ANP aircraft."""
    print("Generating anp_profile...")
    from phonometry import load_anp_database

    aircraft = load_anp_database().aircraft("747100")
    profile = aircraft.profile("D", stage_length=1)

    _fig, ax = plt.subplots(figsize=(10, 6))
    profile.plot(ax=ax)
    ax.set_title(f"ANP Default Departure Profile - {aircraft.description}",
                 fontweight="bold", pad=12)
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
