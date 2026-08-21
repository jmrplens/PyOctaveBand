#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the insulation-design guides: the building before it is built.

What an element will achieve, assembled from element data instead of measured
in a finished room: the EN 12354 flanking model between two rooms, airborne
and impact, the theoretical insulation of a panel from its mass, stiffness and
damping, and the inputs a floor build-up consumes -- the improvement of its
resilient layer or covering, and the structure-borne power of the machine
standing on it. Everything here is embedded by a page under
``buildings/design/``.
"""

import matplotlib.pyplot as plt
import numpy as np

from phonometry._plot.common import format_frequency_axis, theme_fill

from .i18n import _LANG, _fmt_minus
from .theme import (
    _THIRD_OCTAVE_16,
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    _band_index_axis,
    save_figure,
)


def generate_prediction_flanking_demo(output_dir: str) -> None:
    """EN 12354-1 simplified flanking prediction (Annex H.3 worked example)."""
    print("Generating prediction_flanking_demo.png...")
    from phonometry.building.prediction.simplified_model import (
        FlankingPath,
        flanking_element,
        predicted_airborne_insulation,
    )

    # Annex H.3 inputs: separating wall Rs,w = 57 dB, Ss = 11.5 m², four
    # flanking elements. Columns: (label, Rw, KFf, KFd=KDf, coupling length lf).
    elements = [
        ("floor", 49, 12.4, 8.9, 4.50),
        ("ceiling", 46, 14.4, 9.2, 4.50),
        ("facade", 42, 12.6, 6.7, 2.55),
        ("wall", 33, 33.5, 15.7, 2.55),
    ]
    paths: list[FlankingPath] = []
    for name, rw, k_ff, k_side, lf in elements:
        ff, df, fd = flanking_element(
            label=name,
            r_flanking=float(rw),
            r_separating=57.0,
            k_ff=k_ff,
            k_fd=k_side,
            k_df=k_side,
            separating_area=11.5,
            coupling_length=lf,
        )
        paths.extend((ff, df, fd))
    result = predicted_airborne_insulation(r_direct=57.0, flanking_paths=paths)

    # Sort every path (direct + 12 flanking) by its share of the transmitted
    # energy, largest first.
    contribs = sorted(result.paths, key=lambda c: c.fraction, reverse=True)
    labels = [c.label for c in contribs]
    fracs = [c.fraction * 100.0 for c in contribs]
    df_orange = "#ff7f0e"
    kind_color = {
        "Dd": COLOR_TERTIARY,
        "Ff": COLOR_PRIMARY,
        "Fd": COLOR_SECONDARY,
        "Df": df_orange,
    }
    colors = [kind_color[c.kind] for c in contribs]

    direct_share = next(c.fraction for c in result.paths if c.kind == "Dd") * 100.0
    flank_share = 100.0 - direct_share

    _fig, ax = plt.subplots(figsize=(11, 6.4))
    bars = ax.bar(
        range(len(fracs)),
        fracs,
        color=colors,
        edgecolor=COLOR_FG,
        linewidth=0.7,
        zorder=3,
    )
    bars[0].set_linewidth(2.2)  # highlight the dominant path
    ax.annotate(
        "dominant path",
        xy=(0, fracs[0]),
        xytext=(1.5, fracs[0] + 3.5),
        fontsize=10,
        fontweight="bold",
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.1},
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Share of transmitted energy [%]")
    ax.set_xlabel("Transmission path")
    ax.set_ylim(0, max(fracs) + 9.0)
    ax.set_title("EN 12354-1 Flanking Transmission (Annex H.3 example)", pad=12)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=COLOR_TERTIARY, edgecolor=COLOR_FG, label="Dd — direct"),
        Patch(
            facecolor=COLOR_PRIMARY, edgecolor=COLOR_FG, label="Ff — flanking–flanking"
        ),
        Patch(
            facecolor=COLOR_SECONDARY,
            edgecolor=COLOR_FG,
            label="Fd — flanking–separating",
        ),
        Patch(
            facecolor=df_orange, edgecolor=COLOR_FG, label="Df — separating–flanking"
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    rw_dd = result.r_direct_w
    rpw = result.r_prime_w
    lines = [
        rf"$R_\mathrm{{w}}$ (Dd) = {rw_dd:.1f} dB",
        rf"$R^{{\prime}}_\mathrm{{w}}$ = {rpw:.1f} dB",
        (
            rf"$R^{{\prime}}_\mathrm{{w}} - R_\mathrm{{w}}$"
            rf" = {_fmt_minus(rpw - rw_dd, '.1f')} dB"
        ),
        f"Dd {direct_share:.1f} %   ΣFf,Fd,Df {flank_share:.1f} %",
    ]
    ax.text(
        0.985,
        0.62,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=11,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "prediction_flanking_demo.png")
    plt.close()


def generate_floor_covering_improvement(output_dir: str) -> None:
    """ISO 16251-1 floor-covering impact-sound improvement spectrum ΔL with ΔLw."""
    print("Generating floor_covering_improvement...")
    from phonometry import building

    freqs = [
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1000.0,
        1250.0,
        1600.0,
        2000.0,
        2500.0,
        3150.0,
    ]
    # A real textile carpet measured on the CSTB mock-up: the improvement
    # spectrum digitized from Figure 4 of Foret, Chene & Guigou-Carter, Forum
    # Acusticum 2011 (ISO 16251-1 series). The published weighted improvement
    # is delta-Lw = 29 dB.
    bare = np.full(16, 78.0)
    covering = bare - np.array(
        [5, 8, 10, 14, 18, 23, 30, 31, 39, 49, 53, 57, 60, 67, 68, 71], dtype=float
    )
    res = building.impact_improvement(bare, covering, freqs)
    assert res.delta_lw is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(x, 0.0, res.improvement, color=COLOR_TERTIARY, alpha=0.18, zorder=0)
    ax.plot(
        x,
        res.improvement,
        "-",
        color=COLOR_PRIMARY,
        linewidth=2.4,
        marker="o",
        markersize=6,
        zorder=5,
        label=r"$\Delta L$ (improvement)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Improvement of impact sound insulation [dB]")
    ax.set_ylim(bottom=0.0)
    ax.set_title("ISO 16251-1 Floor-Covering Impact Sound Improvement", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"$\\Delta L_\\mathrm{{w}}$ = {res.delta_lw} dB  (ISO 717-2)",
        "one-third octave, mock-up ($a_0 = 10^{-6}$ m/s²)",
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=11,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "floor_covering_improvement.png")
    plt.close()


def generate_masonry_wall_ties(output_dir: str) -> None:
    """Wall-tie coupling loss factor and the resonance shift it causes."""
    print("Generating masonry_wall_ties...")
    from phonometry import building

    _fig, (ax_clf, ax_tl) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # Two 100 mm masonry leaves, 150 and 170 kg/m2 (Hopkins Fig. 5.30
    # flanking-laboratory cavity wall), 2,5 ties per m2.
    freq = np.logspace(np.log10(50.0), np.log10(5000.0), 200)
    stiffness1 = 150.0 * 2000.0**2 * 0.1**2 / 12.0
    stiffness2 = 170.0 * 2000.0**2 * 0.1**2 / 12.0
    ties = ("butterfly", "double_triangle", "vertical_twist")
    colours = (COLOR_PRIMARY, COLOR_TERTIARY, COLOR_SECONDARY)
    for tie, colour in zip(ties, colours, strict=True):
        clf = building.wall_tie_coupling_loss_factor(
            freq,
            150.0,
            170.0,
            stiffness1,
            stiffness2,
            ties_per_area=2.5,
            tie=tie,
        )
        label = f"{tie.replace('_', ' ')} ({building.wall_tie_stiffness(tie)[1] / 1e6:g} MN/m)"
        ax_clf.loglog(
            freq,
            clf.coupling_loss_factor,
            color=colour,
            linewidth=2.4,
            zorder=4,
            label=label,
        )
    rigid = building.wall_tie_coupling_loss_factor(
        freq, 150.0, 170.0, stiffness1, stiffness2, ties_per_area=2.5
    )
    ax_clf.loglog(
        freq,
        rigid.rigid_coupling_loss_factor,
        "--",
        color=COLOR_MUTED,
        linewidth=2.0,
        zorder=3,
        label=r"rigid connection ($Y_\mathrm{c} = 0$)",
    )
    ax_clf.set_xticks([50, 125, 250, 500, 1000, 2000, 4000])
    ax_clf.set_xticklabels(["50", "125", "250", "500", "1k", "2k", "4k"])
    ax_clf.set_xlim(50.0, 5000.0)
    ax_clf.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    ax_clf.set_yticklabels(["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "1"])
    ax_clf.set_ylim(1e-8, 2.0)
    ax_clf.set_xlabel(LABEL_FREQ_HZ)
    ax_clf.set_ylabel(r"Coupling loss factor $\eta_{ij}$")
    ax_clf.set_title(
        "Wall-tie structure-borne coupling\n(point-connection model, 2.5 ties/m²)",
        pad=10,
    )
    ax_clf.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_clf.set_axisbelow(True)
    ax_clf.legend(loc="lower left", fontsize=9)

    # Hopkins Fig. 4.35: two 140 kg/m2 leaves, empty 75 mm cavity, 2,5 ties/m2
    # of s_75mm = 2e6 N/m, which lifts fmsm from 26 Hz to 50 Hz.
    bands = np.array(
        [
            20.0,
            25.0,
            31.5,
            40.0,
            50.0,
            63.0,
            80.0,
            100.0,
            125.0,
            160.0,
            200.0,
            250.0,
            315.0,
            400.0,
            500.0,
        ]
    )
    per_area = building.wall_tie_stiffness_per_area(2.5, 2.0e6)
    plain = building.double_wall_transmission_loss(bands, 140.0, 140.0, 0.075)
    tied = building.double_wall_transmission_loss(
        bands, 140.0, 140.0, 0.075, tie_stiffness_per_area=per_area
    )
    xb = np.arange(bands.size)
    ax_tl.plot(
        xb,
        plain.transmission_loss,
        "-o",
        color=COLOR_PRIMARY,
        linewidth=2.4,
        markersize=6,
        zorder=4,
        label="cavity wall, no ties",
    )
    ax_tl.plot(
        xb,
        tied.transmission_loss,
        "--s",
        color=COLOR_SECONDARY,
        linewidth=2.4,
        markersize=6,
        zorder=4,
        label="2.5 ties/m², $k$ = 2 MN/m",
    )
    positions = []
    for curve, colour in ((plain, COLOR_PRIMARY), (tied, COLOR_SECONDARY)):
        f0 = curve.resonance_frequency
        assert f0 is not None
        pos = float(np.interp(np.log10(f0), np.log10(bands), xb))
        positions.append(pos)
        ax_tl.axvline(pos, color=colour, linestyle=":", linewidth=1.8, zorder=2)
        ax_tl.annotate(
            f"$f_{{\\mathrm{{msm}}}}$ = {f0:.0f} Hz",
            xy=(pos, 0.46),
            xycoords=("data", "axes fraction"),
            ha="right" if positions[:1] else "left",
            va="bottom",
            fontsize=9,
            color=colour,
            rotation=90,
        )
    ax_tl.axvspan(
        positions[0],
        positions[1],
        color=COLOR_SECONDARY,
        alpha=0.30,
        zorder=1,
        label="combined-mass range added by the ties",
    )
    ax_tl.set_xticks(xb[::2])
    ax_tl.set_xticklabels(["20", "31.5", "50", "80", "125", "200", "315", "500"])
    ax_tl.set_xlabel(LABEL_FREQ_HZ)
    ax_tl.set_ylabel("Sound reduction index $R$ [dB]")
    ax_tl.set_title(
        "Ties stiffen the cavity\n(140 kg/m² leaves, 75 mm cavity, 2.5 ties/m²)", pad=10
    )
    ax_tl.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_tl.set_axisbelow(True)
    ax_tl.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "masonry_wall_ties.png")
    plt.close()


def generate_floating_floor_prediction(output_dir: str) -> None:
    """Floating-floor improvement DeltaL(f) under the three prediction laws."""
    print("Generating floating_floor_prediction...")
    from phonometry import building

    # The worked floating floor of ISO 12354-2:2017 Annex G: a 35 mm screed
    # (m' = 73,5 kg/m2) on a resilient layer of s' = 8 MN/m3.
    mass_per_area, stiffness = 73.5, 8.0e6
    f0 = building.floating_floor_resonance_frequency(stiffness, mass_per_area)
    delta_lw = building.weighted_floating_floor_improvement(mass_per_area, stiffness)

    bands = np.array(
        [
            50.0,
            63.0,
            80.0,
            100.0,
            125.0,
            160.0,
            200.0,
            250.0,
            315.0,
            400.0,
            500.0,
            630.0,
            800.0,
            1000.0,
            1250.0,
            1600.0,
            2000.0,
            2500.0,
            3150.0,
            4000.0,
            5000.0,
        ]
    )
    freqs = np.logspace(np.log10(40.0), np.log10(5000.0), 400)
    screed = building.floating_floor_improvement_spectrum(freqs, resonance_frequency=f0)
    asphalt = building.floating_floor_improvement_spectrum(
        freqs, resonance_frequency=f0, model="cremer"
    )
    lightweight = building.floating_floor_improvement_spectrum(
        freqs, resonance_frequency=f0, model="cremer_hammer", limiting_frequency=521.0
    )
    printed = building.floating_floor_improvement_spectrum(
        bands, resonance_frequency=f0
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_betweenx([-5.0, 100.0], 40.0, f0, color=theme_fill(COLOR_FG, ax), zorder=0)
    ax.plot(
        freqs,
        lightweight.improvement,
        "-.",
        color=COLOR_TERTIARY,
        linewidth=2.0,
        label=r"$40\,\log_{10}(f/f_0)$ + hammer term (chipboard)",
    )
    ax.plot(
        freqs,
        asphalt.improvement,
        "--",
        color=COLOR_SECONDARY,
        linewidth=2.0,
        label=r"$40\,\log_{10}(f/f_0)$ (asphalt, dry)",
    )
    ax.plot(
        freqs,
        screed.improvement,
        "-",
        color=COLOR_PRIMARY,
        linewidth=2.4,
        label=r"$30\,\log_{10}(f/f_0)$ (sand-cement screed)",
    )
    ax.plot(
        bands,
        printed.improvement,
        "o",
        color=COLOR_PRIMARY,
        markersize=5.5,
        zorder=6,
        label="ISO 12354-2 Annex G bands",
    )
    ax.axvline(f0, color=COLOR_FG, linestyle=":", linewidth=1.3)
    ax.annotate(
        "$f_0$ = 52.8 Hz",
        xy=(f0, 46.0),
        xytext=(f0 * 1.15, 46.0),
        fontsize=10,
        color=COLOR_FG,
        va="center",
    )

    ax.set_xscale("log")
    format_frequency_axis(ax, 40.0, 5000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Improvement of impact sound insulation [dB]")
    ax.set_ylim(-5.0, 100.0)
    ax.set_title(
        "Floating-Floor Impact Improvement Above the Mass-Spring Resonance", pad=12
    )
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        r"35 mm screed $m^{\prime}$ = 73.5 kg/m² on $s^{\prime}$ = 8 MN/m³",
        f"$\\Delta L_\\mathrm{{w}}$ = {delta_lw:.1f} dB  (ISO 12354-2 Formula C.4)",
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=11,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "floating_floor_prediction.png")
    plt.close()


def generate_soft_covering_prediction(output_dir: str) -> None:
    """Soft-covering DeltaL(f) from the tapping-machine contact stiffness."""
    print("Generating soft_covering_prediction...")
    from phonometry import building, vibration

    # A 140 mm cast in-situ concrete slab (Hopkins Table A2: 2 200 kg/m3,
    # cL = 3 800 m/s, nu = 0,2) carrying the two soft coverings of Hopkins
    # Fig. 4.64: E/d = 1,5e11 N/m3 (a few mm of solid PVC) and 2,8e8 N/m3
    # (a vinyl or carpet with a resilient backing).
    density, longitudinal, poisson, thickness = 2200.0, 3800.0, 0.2, 0.14
    modulus = density * longitudinal**2 * (1.0 - poisson**2)
    plate_stiffness = building.plate_contact_stiffness(modulus, poisson_ratio=poisson)
    impedance = vibration.infinite_plate_impedance(
        vibration.plate_bending_stiffness(modulus, thickness, poisson),
        density * thickness,
    )

    # covering_improvement returns the band value directly: the tapping
    # machine's force spectrum is a line spectrum at multiples of the 10 Hz
    # impact rate, and the band improvement is the ratio of the mean-square
    # forces summed over the lines that fall in each band (Hopkins Eq. 3.91).
    bands = np.array(
        [
            50.0,
            63.0,
            80.0,
            100.0,
            125.0,
            160.0,
            200.0,
            250.0,
            315.0,
            400.0,
            500.0,
            630.0,
            800.0,
            1000.0,
            1250.0,
            1600.0,
            2000.0,
            2500.0,
            3150.0,
            4000.0,
            5000.0,
        ]
    )
    layer = 0.005
    series = (
        (r"No. 1: $E/d = 1.5\times10^{11}$ N/m³", 1.5e11, COLOR_SECONDARY, "--"),
        (r"No. 2: $E/d = 2.8\times10^{8}$ N/m³", 2.8e8, COLOR_PRIMARY, "-"),
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    labels = []
    for label, stiffness_per_volume, colour, style in series:
        result = building.covering_improvement(
            bands,
            building.covering_contact_stiffness(stiffness_per_volume * layer, layer),
            plate_stiffness,
            impedance,
        )
        ax.plot(
            bands,
            result.improvement,
            style,
            color=colour,
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=label,
        )
        ax.plot(bands, result.two_line, ":", color=colour, linewidth=1.5)
        ax.axvline(
            result.cut_off_frequency,
            color=colour,
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )
        labels.append(
            f"{label.split(':')[0]}: $f_{{\\mathrm{{co}}}}$ = "
            f"{result.cut_off_frequency:.0f} Hz"
        )

    ax.plot(
        [],
        [],
        ":",
        color=COLOR_MUTED,
        linewidth=1.5,
        label="two-line estimate (0 dB, 12 dB/oct)",
    )
    ax.set_xscale("log")
    format_frequency_axis(ax, 50.0, 5000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Improvement of impact sound insulation [dB]")
    ax.set_ylim(-5.0, 80.0)
    ax.set_title(
        "Soft Floor Covering Improvement From the Hammer Contact Stiffness", pad=12
    )
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = ["140 mm concrete slab, hammer 0.5 kg, $r$ = 15 mm", *labels]
    ax.text(
        0.02,
        0.70,
        "\n".join(info),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "soft_covering_prediction.png")
    plt.close()


def generate_structure_borne_power(output_dir: str) -> None:
    """EN 15657 structure-borne sound power injected into the reception plate."""
    print("Generating structure_borne_power...")
    from phonometry import building

    bands = np.array([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3150.0])
    # A pump-like source on the two plates EN 15657 specifies. Low-mobility
    # (clause 7.2.2): 100 mm concrete of 2 300 kg/m3, so m = 230 kg/m2, over
    # 3,15 m x 2,23 m = 7,0 m2 with Ts = 0,25 s, which keeps eta >= 0,08
    # through the 50-100 Hz bands. High-mobility (clause 7.3.2): 1 mm
    # perforated steel, m = 7,9 kg/m2, over 2,0 m2 in its support frame.
    lv_low = np.array([72.0, 74.0, 71.0, 68.0, 64.0, 60.0, 55.0])
    # Their point mobilities are 4,9e-6 and 1,1e-2 m/(N.s); driven by a source
    # of |Y_S| ~ 1e-3 m/(N.s) the light plate's spatial-mean velocity level runs
    # 35,1 dB higher, of which 12,0 dB is extra injected power and the rest is
    # its far smaller eta*m*S dissipating it.
    lv_high = lv_low + 35.1
    res_low = building.reception_plate_power(
        lv_low, bands, mass_per_area=230.0, area=7.0, reverberation_time=0.25
    )
    res_high = building.reception_plate_power(
        lv_high, bands, mass_per_area=7.9, area=2.0, reverberation_time=0.5
    )

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(
        x - 0.2,
        res_low.power_level,
        width=0.4,
        color=COLOR_PRIMARY,
        edgecolor=COLOR_FG,
        linewidth=0.6,
        label="low-mobility plate",
    )
    ax.bar(
        x + 0.2,
        res_high.power_level,
        width=0.4,
        color=COLOR_SECONDARY,
        edgecolor=COLOR_FG,
        linewidth=0.6,
        label="high-mobility plate",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Structure-borne power level $L_{W\mathrm{s}}$ [dB re 1 pW]")
    ax.set_title("EN 15657 Reception-Plate Structure-Borne Sound Power", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        r"$L_{W\mathrm{s}} = 10\,\log_{10}(2\pi f\,\eta\,m\,S) + L_v - 60$ dB",
        r"$\eta = 2.2/(f\,T_\mathrm{s})$,  $v_0$ = 1 nm/s",
        "reception-plate method (clause 7)",
    ]
    ax.text(
        0.015,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "structure_borne_power.svg")
    plt.close()


def generate_installed_structure_borne(output_dir: str) -> None:
    """EN 12354-5 installed structure-borne sound: characteristic power to SPL."""
    print("Generating installed_structure_borne...")
    from phonometry import building

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    lws_c = np.array([78.0, 82.0, 84.0, 81.0, 77.0, 72.0, 66.0])  # EN 15657 source
    # Frequency-dependent source / receiver point mobilities (illustrative).
    ys = (2.0e-4 + 1.0e-4j) * (bands / 250.0)
    yi = (3.0e-5 + 1.0e-5j) * np.ones_like(bands)
    dc = np.array(
        [float(building.coupling_term(a, b)) for a, b in zip(ys, yi, strict=True)]
    )
    lws_inst = building.installed_structure_borne_power_level(lws_c, dc)
    # Dsa is negative and falls with frequency (Annex F.2, Formula F.3); the
    # standard's own Annex I columns run from about -14 dB at 63 Hz to -45 dB
    # at 2 kHz. It enters Formula (18a) with a minus sign.
    paths = [
        {
            "adjustment_term": np.array(
                [-14.0, -17.0, -20.0, -25.0, -30.0, -35.0, -40.0]
            ),
            "flanking_reduction_index": np.array(
                [44.0, 47.0, 50.0, 53.0, 56.0, 59.0, 62.0]
            ),
            "element_area": 12.0,
        },
        {
            "adjustment_term": np.array(
                [-16.0, -19.0, -23.0, -28.0, -33.0, -38.0, -43.0]
            ),
            "flanking_reduction_index": np.array(
                [46.0, 49.0, 52.0, 55.0, 58.0, 61.0, 64.0]
            ),
            "element_area": 9.0,
        },
    ]
    res = building.installed_source_prediction(lws_c, dc, paths, frequencies=bands)

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(
        x,
        lws_c,
        color=COLOR_SECONDARY,
        marker="o",
        lw=2.0,
        label=r"characteristic $L_{W\mathrm{s,c}}$ (EN 15657)",
    )
    ax.plot(
        x,
        lws_inst,
        color=COLOR_TERTIARY,
        marker="s",
        lw=2.0,
        label=r"installed $L_{W\mathrm{s,inst}} = L_{W\mathrm{s,c}}"
        r" - D_\mathrm{C}$",
    )
    for k, p in enumerate(res.path_levels):
        ax.plot(
            x,
            p,
            color=COLOR_GRID,
            lw=1.0,
            ls=":",
            marker=".",
            label=r"paths $L_{\mathrm{n,s},ij}$" if k == 0 else None,
        )
    ax.plot(
        x,
        res.total_level,
        color=COLOR_PRIMARY,
        marker="D",
        lw=2.4,
        label=r"total $L_\mathrm{n,s}$",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level [dB]")
    ax.set_title("EN 12354-5 Installed Structure-Borne Sound", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        (
            r"$D_\mathrm{C} = 10\,\log_{10}(|Y_\mathrm{s}+Y_i|^2"
            r" / (|Y_\mathrm{s}|\,\mathrm{Re}\,Y_i))$"
        ),
        (
            r"$L_{\mathrm{n,s},ij} = L_{W\mathrm{s,inst}} - D_\mathrm{sa} - R_{ij}"
            r" - 10\,\log_{10}(S_i/S_0) - 10\,\log_{10}(A_0/4)$"
        ),
        (
            r"$L_\mathrm{n,s} = 10\,\log_{10}"
            r"(\Sigma\,10^{L_{\mathrm{n,s},ij}/10})$,  "
            "$S_0 = A_0$ = 10 m²"
        ),
    ]
    ax.text(
        0.015,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "installed_structure_borne.svg")
    plt.close()


def generate_aperture_slit_geometry(output_dir: str) -> None:
    """To-scale section of a 2 mm slit through a 100 mm wall.

    The slit of the aperture transmission-loss example: a deep, narrow gap
    whose depth resonances puncture the wall's insulation. One concept: the
    tiny geometry behind a large leak.
    """
    print("Generating aperture_slit_geometry...")
    from phonometry import building

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    building.plot_aperture_geometry(0.1, ax=ax, width=0.002, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "aperture_slit_geometry.svg")
    plt.close()


def generate_double_wall_geometry(output_dir: str) -> None:
    """Mass-spring-mass double wall to scale.

    Two 12,5 mm plasterboard leaves (8,8 kg/m2 each) on a 100 mm stud
    cavity, the classic lightweight double wall, with its mass-air-mass
    resonance annotated. One concept: the geometry behind the double-wall
    resonance dip.
    """
    print("Generating double_wall_geometry...")
    from phonometry import building

    f0 = building.mass_spring_mass_resonance(8.8, 8.8, 0.1)
    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    building.plot_double_wall_geometry(
        8.8,
        8.8,
        0.1,
        ax=ax,
        resonance_frequency=f0,
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "double_wall_geometry.svg")
    plt.close()


def generate_panel_insulation_concept(output_dir: str) -> None:
    """Theoretical panel sound insulation: the four PR-I predictions."""
    print("Generating panel_insulation_concept.png...")
    from phonometry import building, vibration

    bands = np.array(
        [
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
        ],
        dtype=float,
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # Three of the four panels carry the same y label, and on one line it is
    # longer than a panel of a two-by-two grid is tall: in Spanish its head
    # ran into the title and its tail into the label of the panel below. Set
    # over two lines it clears both, in either language.
    r_label = "Sound reduction index $R$\n(transmission loss $\\mathrm{TL}$) [dB]"

    # (a) Single panel: field-incidence mass law and the coincidence dip.
    bp = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = vibration.coincidence_frequency(15.0, bp)
    ml = building.mass_law_transmission_loss(bands, 15.0, incidence="field")
    sharp = building.single_panel_transmission_loss(
        bands, 15.0, critical_frequency=fc, loss_factor=0.024
    )
    ax = axes[0, 0]
    ax.semilogx(
        bands,
        ml,
        color=COLOR_TERTIARY,
        ls="--",
        lw=1.6,
        label="field-incidence mass law",
    )
    ax.semilogx(
        bands,
        sharp.transmission_loss,
        color=COLOR_PRIMARY,
        lw=2.0,
        marker="o",
        markersize=3,
        label="single panel $R$ (Sharp)",
    )
    ax.axvline(fc, color=COLOR_SECONDARY, ls=":", lw=1.2, label=r"$f_\mathrm{c}$")
    ax.set_title("Single panel: mass law and coincidence", pad=10)
    ax.set_ylabel(r_label)
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (b) Double wall: mass-spring-mass resonance and cavity gain.
    dw = building.double_wall_transmission_loss(bands, 12.0, 12.0, 0.075)
    single = building.mass_law_transmission_loss(bands, 24.0, incidence="field")
    f0 = building.mass_spring_mass_resonance(12.0, 12.0, 0.075)
    ax = axes[0, 1]
    ax.semilogx(
        bands,
        single,
        color=COLOR_TERTIARY,
        ls="--",
        lw=1.6,
        label="single leaf (total mass)",
    )
    ax.semilogx(
        bands,
        dw.transmission_loss,
        color=COLOR_PRIMARY,
        lw=2.0,
        marker="o",
        markersize=3,
        label="double wall $R$",
    )
    ax.axvline(f0, color=COLOR_SECONDARY, ls=":", lw=1.2, label="$f_0$")
    ax.set_title("Double wall: mass-spring-mass resonance", pad=10)
    ax.set_ylabel(r_label)
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (c) Radiation efficiency of a bending plate.
    sigma = vibration.radiation_efficiency(bands, 1.5, 1.25, fc)
    ax = axes[1, 0]
    ax.loglog(
        bands,
        sigma.radiation_efficiency,
        color=COLOR_PRIMARY,
        lw=2.0,
        marker="o",
        markersize=3,
        label=r"$\sigma(f)$",
    )
    ax.axhline(1.0, color=COLOR_FG, ls=":", lw=0.9, alpha=0.5, label="$\\sigma = 1$")
    ax.axvline(fc, color=COLOR_SECONDARY, ls=":", lw=1.2, label=r"$f_\mathrm{c}$")
    ax.set_title("Radiation efficiency of a bending plate", pad=10)
    ax.set_ylabel(r"Radiation efficiency $\sigma$")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (d) Composite wall with a small aperture (open-area cap).
    wall = sharp.transmission_loss
    open_area = 0.01
    n = bands.size
    composite = np.array(
        [
            float(
                building.composite_transmission_loss(
                    [1.0 - open_area, open_area], [wall[i], 0.0]
                )
            )
            for i in range(n)
        ]
    )
    ax = axes[1, 1]
    ax.semilogx(
        bands,
        wall,
        color=COLOR_PRIMARY,
        lw=2.0,
        marker="o",
        markersize=3,
        label="solid wall alone",
    )
    ax.semilogx(
        bands,
        composite,
        color=COLOR_SECONDARY,
        lw=2.0,
        marker="s",
        markersize=3,
        label="wall + 1 % open slit",
    )
    ax.axhline(
        10.0 * np.log10(1.0 / open_area),
        color=COLOR_FG,
        ls=":",
        lw=0.9,
        alpha=0.5,
        label="open-area limit",
    )
    ax.set_title("Composite wall with a small aperture", pad=10)
    ax.set_ylabel(r_label)
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    fig.suptitle(
        "Theoretical panel sound insulation (Bies / Hopkins / Cremer)",
        fontsize=14,
    )
    plt.tight_layout()
    save_figure(output_dir, "panel_insulation_concept.png")
    plt.close()


def generate_impact_prediction_terms(output_dir: str) -> None:
    """EN 12354-2 Annex E.3 impact prediction as its Formula (21) terms."""
    print("Generating impact_prediction_terms...")
    from phonometry import building

    # EN 12354-2 Annex E.3: a 0.14 m concrete floor (m' = 322 kg/m2) with a
    # floating floor (delta-Lw = 33 dB), mean flanking mass 145 kg/m2.
    ln_eq = float(building.equivalent_impact_level(322.0))
    k = float(building.impact_flanking_correction(322.0, 145.0))
    imp = building.predicted_impact_insulation(
        ln_w_eq=ln_eq, delta_l_w=33.0, k_correction=k
    )

    labels = [
        r"$L_\mathrm{n,w,eq}$",
        r"$-\Delta L_\mathrm{w}$",
        "$+K$",
        r"$L^{\prime}_\mathrm{n,w}$",
    ]
    values = [imp.ln_w_eq, -imp.delta_l_w, imp.k_correction, imp.l_prime_n_w]
    # COLOR_MUTED rather than the gridline grey for the starting term: a bar
    # is a read value, so it has to hold up against the page on both themes.
    colors = [COLOR_MUTED, COLOR_TERTIARY, COLOR_SECONDARY, COLOR_PRIMARY]

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(np.arange(4), values, color=colors, zorder=5)
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=12)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            _fmt_minus(value, "+.1f"),
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 5 if value >= 0 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=COLOR_FG,
        )
    ax.set_ylim(min(values) - 9.0, max(values) + 9.0)

    ax.set_ylabel("Level / correction [dB]")
    ax.set_title("EN 12354-2 Impact Sound Prediction (Annex E.3)", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", zorder=0)
    ax.set_axisbelow(True)

    info = [
        r"$L^{\prime}_\mathrm{n,w} = L_\mathrm{n,w,eq} - \Delta L_\mathrm{w} + K$",
        (
            f"$L_\\mathrm{{n,w,eq}} = 164 - 35\\,\\log_{{10}}"
            f"(m^{{\\prime}}/m^{{\\prime}}_0)$ = {ln_eq:.1f} dB"
        ),
        rf"$L^{{\prime}}_\mathrm{{n,w}}$ = {imp.l_prime_n_w:.1f} dB → 45 dB",
    ]
    ax.text(
        0.985,
        0.97,
        "\n".join(info),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "impact_prediction_terms.svg")
    plt.close()


def generate_detailed_prediction_paths(output_dir: str) -> None:
    """ISO 12354-1 Annex L: which transmission path dominates each band."""
    print("Generating detailed_prediction_paths...")
    import sys
    from pathlib import Path

    # parents[2] is the repository root from scripts/figures/<module>.py. The
    # count is what broke when this builder moved down into the package, so it
    # is written the way scripts/conformance/registry.py writes it rather than
    # as .parent.parent, which reads the same and is one level short.
    tests = str(Path(__file__).resolve().parents[2] / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import iso12354_building as bld

    from phonometry import building

    # The heavy homogeneous building of ISO 12354-1:2017 Annex L: a 220 mm
    # concrete separating floor with a floating floor, two 365 mm AAC external
    # walls and two 200 mm calcium-silicate internal walls. The building and
    # its twelve flanking paths come from the shared test fixture, which the
    # conformance report reads too, so the figure cannot drift from the rows.
    bands = np.array(
        [
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
        ],
        dtype=float,
    )
    situ = {k: building.in_situ_element(e, bands) for k, e in bld.elements().items()}
    delta = building.floating_floor_improvement(
        bands, resonance_frequency=bld.floating_floor_resonance()
    )
    res = building.detailed_airborne_prediction(
        bands,
        direct_index=building.direct_reduction_index(
            situ["floor"].sound_reduction_index, delta_r_source=delta
        ),
        flanking_paths=bld.airborne_paths(situ, delta),
    )
    assert res.rating is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, bands)
    order = list(np.argsort(-res.fractions.max(axis=1)))
    palette = [
        COLOR_PRIMARY,
        COLOR_SECONDARY,
        COLOR_TERTIARY,
        "#9467bd",
        "#aec7e8",
        "#ff9896",
    ]
    bottom = np.zeros(x.size)
    for colour, k in zip(palette, order[:6], strict=False):  # six colours, rest pooled
        share = 100.0 * res.fractions[k]
        ax.bar(
            x,
            share,
            bottom=bottom,
            width=0.85,
            color=colour,
            edgecolor="none",
            zorder=2,
            label=res.paths[k].label,
        )
        bottom = bottom + share
    rest = order[6:]
    if rest:
        share = 100.0 * res.fractions[rest].sum(axis=0)
        # The pooled segment is de-emphasised *data*, so it takes the mid grey
        # that reads on both pages, not the gridline grey that hides in one.
        ax.bar(
            x,
            share,
            bottom=bottom,
            width=0.85,
            color=COLOR_MUTED,
            edgecolor="none",
            zorder=2,
            label="other paths",
        )
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Share of transmitted energy [%]")
    ax.set_title("ISO 12354-1 Detailed Model: Dominant Path per Band (Annex L)", pad=12)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.plot(
        x,
        res.r_prime,
        "-o",
        color=COLOR_FG,
        linewidth=2.0,
        markersize=4,
        zorder=5,
        label=r"$R^{\prime}$ (apparent)",
    )
    twin.set_ylabel(r"Apparent sound reduction index $R^{\prime}$ [dB]")
    handles, labels = ax.get_legend_handles_labels()
    extra, extra_labels = twin.get_legend_handles_labels()
    ax.legend(
        handles + extra, labels + extra_labels, loc="upper left", fontsize=9, ncol=3
    )

    info = [
        r"$R^{\prime} = -10\,\log_{10}(\Sigma\,10^{-R_{ij}/10})$",
        (
            rf"$R^{{\prime}}_\mathrm{{w}}$ ($C$; $C_\mathrm{{tr}}$)"
            rf" = {res.rating.rating} "
            f"({_fmt_minus(res.rating.c)}; {_fmt_minus(res.rating.ctr)}) dB"
        ),
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=10,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "detailed_prediction_paths.svg")
    plt.close()


def generate_single_panel_rating(output_dir: str) -> None:
    """Sharp single-panel R(f) prediction rated per ISO 717-1 (6 mm glass)."""
    print("Generating single_panel_rating...")
    from phonometry import building, vibration

    # 6 mm float glass: E = 62 GPa, rho = 2500 kg/m3, nu = 0.24, eta = 0.024.
    bands = np.asarray(_THIRD_OCTAVE_16, dtype=float)
    mass = 2500.0 * 0.006
    bp = vibration.plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = float(vibration.coincidence_frequency(mass, bp))
    res = building.single_panel_transmission_loss(
        bands, mass, critical_frequency=fc, loss_factor=0.024
    )
    w = res.rating()
    assert w.shifted_reference is not None and w.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(
        x,
        res.transmission_loss,
        "-o",
        color=COLOR_PRIMARY,
        linewidth=2.2,
        markersize=5,
        zorder=5,
        label="predicted $R$ (Sharp)",
    )
    ax.plot(
        x,
        w.shifted_reference,
        "--s",
        color=COLOR_FG,
        linewidth=1.8,
        markersize=5,
        zorder=5,
        label="shifted reference",
    )
    unfav = w.measured < w.shifted_reference
    ax.fill_between(
        x,
        w.measured,
        w.shifted_reference,
        where=unfav.tolist(),
        color=COLOR_SECONDARY,
        alpha=0.25,
        interpolate=True,
        zorder=1,
        label="unfavourable deviations",
    )
    idx_fc = float(np.interp(np.log10(fc), np.log10(bands), x))
    ax.axvline(
        idx_fc,
        color=COLOR_TERTIARY,
        linestyle=":",
        linewidth=1.6,
        zorder=4,
        label=rf"coincidence $f_\mathrm{{c}}$ = {fc:.0f} Hz",
    )

    ax.set_ylabel("Sound reduction index $R$ (transmission loss $\\mathrm{TL}$) [dB]")
    ax.set_title("Predicted Single-Panel Insulation Rated per ISO 717-1", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        (
            rf"$R_\mathrm{{w}}$($C$;$C_\mathrm{{tr}}$) = {w.rating}"
            f"({_fmt_minus(w.c)};{_fmt_minus(w.ctr)}) dB"
        ),
        r"6 mm float glass, $m^{\prime\prime}$ = 15 kg/m², $\eta$ = 0.024",
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=10,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "single_panel_rating.svg")
    plt.close()


def generate_plateau_transmission_loss(output_dir: str) -> None:
    """Plateau-method TL estimate against the full physical model."""
    print("Generating plateau_transmission_loss...")
    from phonometry import building

    # 6 mm float glass, 2.47 kg/m2 per mm (Norton Table 3.1); its critical
    # frequency from the same book's problem 3.13 is 2033 Hz.
    thickness_mm, f_c, eta = 6.0, 2033.0, 0.02
    mass = 2.47 * thickness_mm
    nominal = [*_THIRD_OCTAVE_16, 4000, 5000, 6300, 8000, 10000]
    bands = np.asarray(nominal, dtype=float)
    quick = building.plateau_transmission_loss(
        bands, material="glass", thickness_mm=thickness_mm, field_correction=5.5
    )
    physical = building.single_panel_transmission_loss(
        bands, mass, critical_frequency=f_c, loss_factor=eta
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, nominal)
    assert quick.plateau_start is not None and quick.plateau_end is not None
    idx_a = float(np.interp(np.log10(quick.plateau_start), np.log10(bands), x))
    idx_b = float(np.interp(np.log10(quick.plateau_end), np.log10(bands), x))
    ax.axvspan(
        idx_a,
        idx_b,
        color=theme_fill(COLOR_SECONDARY, ax),
        lw=0,
        zorder=0,
        label="coincidence plateau ($A$ to $B$)",
    )
    ax.plot(
        x,
        physical.transmission_loss,
        "-o",
        color=COLOR_PRIMARY,
        linewidth=2.2,
        markersize=5,
        zorder=5,
        label="physical model (mass law + coincidence + damping)",
    )
    ax.plot(
        x,
        quick.transmission_loss,
        "--s",
        color=COLOR_SECONDARY,
        linewidth=2.0,
        markersize=5,
        zorder=5,
        label="plateau estimate (Norton Table 3.1)",
    )
    idx_fc = float(np.interp(np.log10(f_c), np.log10(bands), x))
    ax.axvline(
        idx_fc,
        color=COLOR_TERTIARY,
        linestyle=":",
        linewidth=1.6,
        zorder=4,
        label=r"critical frequency $f_\mathrm{c}$",
    )

    ax.set_ylabel("Sound reduction index $R$ (transmission loss $\\mathrm{TL}$) [dB]")
    ax.set_title("Plateau Estimate Against the Physical Panel Model", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    info = [
        (
            f"6 mm float glass, $m^{{\\prime\\prime}}$ = {mass:.1f} kg/m², "
            f"$\\eta$ = {eta:g}"
        ),
        (
            f"plateau height 27 dB, $B/A = 10$ → $A$ = {quick.plateau_start:.0f} Hz, "
            f"$B$ = {quick.plateau_end:.0f} Hz"
        ),
        "identical below $A$; the plateau replaces the whole coincidence region",
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        color=COLOR_FG,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": panel, "edgecolor": COLOR_GRID},
    )
    plt.tight_layout()
    save_figure(output_dir, "plateau_transmission_loss.svg")
    plt.close()


def generate_orthotropic_transmission_loss(output_dir: str) -> None:
    """Corrugating a sheet flattens its transmission loss (Vigran 6.5.3).

    One concept: the same steel sheet, flat and corrugated. The flat plate has
    a single coincidence frequency far above the audio range and follows the
    mass law; corrugating it drags the lower coincidence frequency down by more
    than a decade and opens a range over which R collapses, despite the 9 %
    extra mass the developed length adds.
    """
    print("Generating orthotropic_transmission_loss...")
    from phonometry import building, vibration

    # Vigran's worked example (printed p. 96): 1 mm steel, corrugation of total
    # height 20 mm (H = 10 mm) at a 100 mm pitch, E = 2,1e11 Pa, nu = 0,3.
    modulus, thickness, poisson, eta = 2.1e11, 1.0e-3, 0.3, 0.011
    amplitude, pitch, mass_flat = 0.010, 0.100, 7.8
    flat_b = vibration.plate_bending_stiffness(modulus, thickness, poisson)
    flat_fc = vibration.coincidence_frequency(mass_flat, flat_b)
    b_x, b_z, _b_xz = building.corrugated_plate_stiffness(
        thickness,
        amplitude,
        pitch,
        youngs_modulus=modulus,
        poisson_ratio=poisson,
    )
    mass_corr = mass_flat * building.corrugated_plate_mass_factor(amplitude, pitch)
    fc1, fc2 = building.orthotropic_critical_frequencies(mass_corr, b_x, b_z)

    nominal = [*_THIRD_OCTAVE_16, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
    bands = np.asarray(nominal, dtype=float)
    flat = building.single_panel_transmission_loss(
        bands,
        mass_flat,
        critical_frequency=flat_fc,
        loss_factor=eta,
    )
    corrugated = building.orthotropic_transmission_loss(
        bands,
        mass_corr,
        critical_frequency_lower=fc1,
        critical_frequency_upper=fc2,
        loss_factor=eta,
    )
    heckl = building.orthotropic_transmission_loss(
        bands,
        mass_corr,
        critical_frequency_lower=fc1,
        critical_frequency_upper=fc2,
        method="heckl",
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, nominal)
    idx_1 = float(np.interp(np.log10(fc1), np.log10(bands), x))
    idx_2 = float(np.interp(np.log10(fc2), np.log10(bands), x))
    ax.axvspan(
        idx_1,
        idx_2,
        color=theme_fill(COLOR_TERTIARY, ax),
        lw=0,
        zorder=0,
        label=r"coincidence range $f_{\mathrm{c}1}$ to $f_{\mathrm{c}2}$",
    )
    ax.plot(
        x,
        flat.transmission_loss,
        "-o",
        color=COLOR_PRIMARY,
        linewidth=2.2,
        markersize=5,
        zorder=5,
        label=r"flat 1 mm sheet (isotropic, single $f_\mathrm{c}$)",
    )
    ax.plot(
        x,
        corrugated.transmission_loss,
        "-s",
        color=COLOR_SECONDARY,
        linewidth=2.2,
        markersize=5,
        zorder=5,
        label="corrugated sheet (orthotropic, diffuse-field integral)",
    )
    ax.plot(
        x,
        heckl.transmission_loss,
        "--",
        color=COLOR_TERTIARY,
        linewidth=1.8,
        zorder=4,
        label="Heckl's approximation",
    )

    ax.set_ylabel("Sound reduction index $R$ (transmission loss $\\mathrm{TL}$) [dB]")
    ax.set_title("Corrugating a Sheet Flattens Its Sound Reduction Index", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    worst = int(np.argmax(flat.transmission_loss - corrugated.transmission_loss))
    penalty = float(flat.transmission_loss[worst] - corrugated.transmission_loss[worst])
    info = [
        (
            rf"1 mm steel sheet, $m^{{\prime\prime}}$ = {mass_flat:.1f} kg/m², flat "
            rf"$f_\mathrm{{c}}$ = {flat_fc / 1000.0:.1f} kHz"
        ),
        (
            f"corrugated $H$ = 10 mm, $L$ = 100 mm, "
            rf"$m^{{\prime\prime}}$ = {mass_corr:.1f} kg/m², "
            rf"$f_{{\mathrm{{c}}1}}$ = {fc1:.0f} Hz, "
            rf"$f_{{\mathrm{{c}}2}}$ = {fc2 / 1000.0:.1f} kHz"
        ),
        (
            f"worst penalty {penalty:.0f} dB at {nominal[worst]:g} Hz, "
            "for a stiffer and only 9 % heavier panel"
        ),
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        color=COLOR_FG,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": panel, "edgecolor": COLOR_GRID},
    )
    plt.tight_layout()
    save_figure(output_dir, "orthotropic_transmission_loss.svg")
    plt.close()


def generate_coupling_term_regimes(output_dir: str) -> None:
    """EN 12354-5 coupling term against the source-to-receiver mobility ratio."""
    print("Generating coupling_term_regimes...")
    from phonometry import building

    # The receiver mobility of the guide's pump example, held fixed while the
    # source mobility sweeps six decades around it along the same phase.
    y_i = 3e-5 + 1e-5j
    ratio = np.logspace(-3.0, 3.0, 400)
    y_s = abs(y_i) * ratio * (2 + 1j) / abs(2 + 1j)

    exact = np.array([float(building.coupling_term(a, y_i)) for a in y_s])
    force = np.array([float(building.coupling_term_force_source(a, y_i)) for a in y_s])
    velocity = np.array(
        [float(building.coupling_term_velocity_source(a, 1.0 / y_i)) for a in y_s]
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.semilogx(
        ratio,
        exact,
        "-",
        color=COLOR_PRIMARY,
        linewidth=2.4,
        zorder=5,
        label="Formula 19b (exact, complex mobilities)",
    )
    ax.semilogx(
        ratio,
        force,
        "--",
        color=COLOR_SECONDARY,
        linewidth=1.8,
        zorder=4,
        label="force-source limit (19c)",
    )
    ax.semilogx(
        ratio,
        velocity,
        ":",
        color=COLOR_TERTIARY,
        linewidth=1.8,
        zorder=4,
        label="velocity-source limit (19d)",
    )
    for style, y_k in (("-.", 1e-4), ((0, (5, 2, 1, 2)), 1e-3)):
        with_mount = np.array(
            [float(building.coupling_term(a, y_i, transfer_mobility=y_k)) for a in y_s]
        )
        ax.semilogx(
            ratio,
            with_mount,
            linestyle=style,
            color=COLOR_MUTED,
            linewidth=1.8,
            zorder=3,
            label=f"elastic support $Y_k$ = {y_k:g} m/(N s)  (19e)",
        )

    # The exact curve's minimum: matched mobilities, best power transfer.
    low = int(np.argmin(exact))
    ax.plot(ratio[low], exact[low], "v", color=COLOR_PRIMARY, markersize=9, zorder=6)
    ax.annotate(
        f"matched mobilities: {exact[low]:.1f} dB",
        xy=(ratio[low], exact[low]),
        xytext=(ratio[low], exact[low] - 7.0),
        ha="center",
        fontsize=9,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    pump = abs(y_i) * 7.0710678 * (2 + 1j) / abs(2 + 1j)
    pump_dc = float(building.coupling_term(pump, y_i))
    ax.plot(7.0710678, pump_dc, "o", color=COLOR_FG, markersize=8, zorder=7)
    ax.annotate(
        f"pump on a concrete slab: {pump_dc:.1f} dB",
        xy=(7.0710678, pump_dc),
        xytext=(30.0, pump_dc - 12.0),
        fontsize=9,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    ax.set_xlabel(r"Mobility ratio $|Y_\mathrm{s}|/|Y_i|$")
    ax.set_ylabel(r"Coupling term $D_\mathrm{C}$ [dB]")
    ax.set_title("EN 12354-5 Coupling Term and Its Two Limits", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        (
            r"$D_\mathrm{C} = 10\,\log_{10}(|Y_\mathrm{s} + Y_i + Y_k|^2"
            r" / (|Y_\mathrm{s}|\,\mathrm{Re}\{Y_i\}))$"
        ),
        "left: velocity source (stiff receiver takes more)",
        "right: force source (stiff receiver takes less)",
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "coupling_term_regimes.svg")
    plt.close()


def generate_tapping_force_spectrum(output_dir: str) -> None:
    """The tapping machine's force spectrum on concrete and on chipboard."""
    print("Generating tapping_force_spectrum...")
    from phonometry import building, vibration

    freqs = np.logspace(np.log10(50.0), np.log10(20000.0), 500)
    surfaces = (
        ("140 mm concrete slab", 2200.0, 3800.0, 0.2, 0.14, COLOR_PRIMARY),
        ("22 mm chipboard walking surface", 650.0, 2400.0, 0.3, 0.022, COLOR_SECONDARY),
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    upper = lower = 0.0
    for label, rho, c_l, nu, h, colour in surfaces:
        modulus = rho * c_l**2 * (1 - nu**2)
        impedance = vibration.infinite_plate_impedance(
            vibration.plate_bending_stiffness(modulus, h, nu), rho * h
        )
        stiffness = building.plate_contact_stiffness(modulus, poisson_ratio=nu)
        res = building.tapping_force_spectrum(freqs, stiffness, impedance)
        upper, lower = res.upper_limit, res.lower_limit
        regime = "over-critical" if res.over_critical else "under-critical"
        ax.loglog(
            freqs,
            res.peak_force,
            "-",
            color=colour,
            linewidth=2.4,
            zorder=5,
            label=(
                f"{label} ({regime}, $f_{{\\mathrm{{co}}}}$ = "
                f"{res.cut_off_frequency:.0f} Hz)"
            ),
        )
        ax.axvline(
            res.cut_off_frequency, color=colour, linestyle=":", linewidth=1.4, zorder=3
        )
        f_limit = float(building.hammer_limiting_frequency(impedance))
        if freqs[0] < f_limit < freqs[-1]:
            ax.axvline(f_limit, color=colour, linestyle="-.", linewidth=1.2, zorder=3)
            ax.annotate(
                f"$f_{{\\mathrm{{limit}}}}$ = {f_limit:.0f} Hz",
                xy=(f_limit, lower * 0.55),
                xytext=(f_limit * 1.25, lower * 0.35),
                fontsize=9,
                color=colour,
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": colour},
            )

    ax.axhline(
        upper,
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=1.6,
        zorder=4,
        label=r"$|F_n|_{\mathrm{upper}} = 2\,m\,v_\mathrm{h}/T_\mathrm{i}$"
        r"  (rebound)",
    )
    ax.axhline(
        lower,
        color=COLOR_MUTED,
        linestyle="-.",
        linewidth=1.6,
        zorder=4,
        label=r"$|F_n|_{\mathrm{lower}} = m\,v_\mathrm{h}/T_\mathrm{i}$"
        r"  (no rebound)",
    )
    ax.annotate(
        "",
        xy=(90.0, upper),
        xytext=(90.0, lower),
        arrowprops={"arrowstyle": "<->", "lw": 1.4},
    )
    ax.text(
        97.0,
        (upper * lower) ** 0.5,
        f"6.0 dB in mean square\n({lower:.2f} N to {upper:.2f} N)",
        fontsize=9,
        color=COLOR_FG,
        va="center",
    )

    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Line force $|F_n|$ [N]")
    ax.set_title("Tapping-Machine Force: the Floor Decides the Excitation", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)

    info = [
        "0.5 kg hammer dropped 40 mm, 10 impacts per second",
        "building acoustics range shaded",
    ]
    ax.axvspan(100.0, 3150.0, color=theme_fill(COLOR_TERTIARY, ax), zorder=1)
    ax.text(
        0.985,
        0.97,
        "\n".join(info),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "tapping_force_spectrum.svg")
    plt.close()


def generate_detailed_impact_paths(output_dir: str) -> None:
    """ISO 12354-2 Annex G: the five impact paths of the worked building."""
    print("Generating detailed_impact_paths...")
    import sys
    from pathlib import Path

    tests = str(Path(__file__).resolve().parents[2] / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import iso12354_building as bld

    from phonometry import building

    bands = np.array(
        [
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
        ],
        dtype=float,
    )
    situ = {k: building.in_situ_element(e, bands) for k, e in bld.elements().items()}
    f_0 = bld.floating_floor_resonance()
    delta = building.floating_floor_improvement(bands, resonance_frequency=f_0)
    res = building.detailed_impact_prediction(
        bands,
        direct_level=building.direct_impact_level(
            situ["floor"].impact_level, delta_l=delta
        ),
        flanking_paths=bld.impact_paths(situ, delta),
    )
    assert res.rating is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, bands)
    palette = [COLOR_TERTIARY, COLOR_PRIMARY, COLOR_SECONDARY, "#9467bd", "#ff9896"]
    bottom = np.zeros(x.size)
    for colour, k in enumerate_paths(res):
        share = 100.0 * res.fractions[k]
        ax.bar(
            x,
            share,
            bottom=bottom,
            width=0.85,
            color=palette[colour % len(palette)],
            edgecolor="none",
            zorder=2,
            label=res.paths[k].label,
        )
        bottom = bottom + share
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Share of transmitted energy [%]")
    ax.set_title(
        "ISO 12354-2 Detailed Model: the Direct Path Governs (Annex G)", pad=12
    )
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.plot(
        x,
        res.l_prime_n,
        "-o",
        color=COLOR_FG,
        linewidth=2.0,
        markersize=4,
        zorder=5,
        label=r"$L^{\prime}_\mathrm{n}$ (apparent)",
    )
    twin.set_ylabel(r"Apparent normalized impact level $L^{\prime}_\mathrm{n}$ [dB]")
    idx_f0 = float(np.interp(np.log10(f_0), np.log10(bands), x))
    twin.axvline(
        idx_f0,
        color=COLOR_MUTED,
        linestyle=":",
        linewidth=1.6,
        zorder=4,
        label=f"floating floor $f_0$ = {f_0:.1f} Hz",
    )
    handles, labels = ax.get_legend_handles_labels()
    extra, extra_labels = twin.get_legend_handles_labels()
    ax.legend(
        handles + extra, labels + extra_labels, loc="upper right", fontsize=9, ncol=3
    )

    info = [
        "five paths, not thirteen: only the floor is excited",
        (
            rf"$L^{{\prime}}_\mathrm{{n,w}}$ ($C_\mathrm{{I}}$)"
            rf" = {res.rating.rating} "
            f"({_fmt_minus(res.rating.ci)}) dB"
        ),
    ]
    ax.text(
        0.015,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "detailed_impact_paths.svg")
    plt.close()


def enumerate_paths(result: object) -> list[tuple[int, int]]:
    """``(colour index, path index)`` with the largest contributor first."""
    fractions = result.fractions  # type: ignore[attr-defined]
    order = list(np.argsort(-fractions.max(axis=1)))
    return [(i, int(k)) for i, k in enumerate(order)]


def generate_radiation_efficiency_panels(output_dir: str) -> None:
    """Plate radiation efficiency for two pane sizes, and the baffled plate."""
    print("Generating radiation_efficiency_panels...")
    from phonometry import vibration

    bands = np.array(
        [
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
        ],
        dtype=float,
    )
    f_c = 2107.0  # the page's 6 mm float glass pane
    big = vibration.radiation_efficiency(bands, 1.5, 1.25, f_c)
    small = vibration.radiation_efficiency(bands, 0.5, 0.4, f_c)

    _fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    ax_l.loglog(
        bands,
        big.radiation_efficiency,
        "-o",
        color=COLOR_PRIMARY,
        linewidth=2.2,
        markersize=5,
        zorder=5,
        label="1.5 m × 1.25 m pane",
    )
    ax_l.loglog(
        bands,
        small.radiation_efficiency,
        "-s",
        color=COLOR_SECONDARY,
        linewidth=2.2,
        markersize=5,
        zorder=5,
        label="0.5 m × 0.4 m pane, same glass",
    )
    ax_l.axhline(
        1.0,
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=1.4,
        zorder=3,
        label=r"$\sigma = 1$ (as efficient as a piston)",
    )
    ax_l.axvline(
        f_c,
        color=COLOR_TERTIARY,
        linestyle=":",
        linewidth=1.6,
        zorder=3,
        label=rf"critical frequency $f_\mathrm{{c}}$ = {f_c:.0f} Hz",
    )
    peak = float(big.radiation_efficiency.max())
    # The label goes in the open wedge under the legend and left of the rise,
    # with the leader carrying the eye to the peak: anywhere along the top the
    # upper-left legend would cover its opening words, and the Spanish line is
    # the wider of the two.
    ax_l.annotate(
        f"coincidence peak: $\\sigma$ = {peak:.2f}",
        xy=(2000.0, peak),
        xytext=(0.04, 0.50),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=9,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax_l.set_xlabel(LABEL_FREQ_HZ)
    ax_l.set_ylabel(r"Radiation efficiency $\sigma$")
    ax_l.set_title("Edge Radiation, Coincidence and the Slow Return to Unity", pad=12)
    ax_l.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="upper left", fontsize=9)

    ratio = float(
        small.radiation_efficiency[bands == 100][0]
        / big.radiation_efficiency[bands == 100][0]
    )
    ax_l.text(
        0.985,
        0.03,
        (
            f"at 100 Hz the small pane radiates {ratio:.1f} times better:\n"
            "the uncancelled edge strip is a larger fraction of it"
        ),
        transform=ax_l.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )

    big.plot_geometry(ax=ax_r)
    plt.tight_layout()
    save_figure(output_dir, "radiation_efficiency_panels.svg")
    plt.close()


def generate_structure_borne_conversion(output_dir: str) -> None:
    """EN 15657 to EN 12354-5: the two quantities one correction produces."""
    print("Generating structure_borne_conversion...")
    from phonometry import building

    # The flushing cistern of EN 12354-5 Annex I.3 at its wall contact, whose
    # printed Table I.8 columns the library reproduces to a tenth of a decibel.
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0])
    l_ws = np.array([61.7, 59.8, 47.2, 44.9, 38.8, 27.2])  # on the test plate
    l_fb = building.equivalent_blocked_force_level(l_ws, 5.34e-6)  # Formula (15)
    l_wsn = building.characteristic_reception_plate_power(l_fb)  # Formula (17)
    installed = building.installed_power_from_reception_plate(l_wsn, 24.1e-6)
    characteristic = building.installed_power_from_reception_plate(l_wsn, 1.0e-3)
    d_c = 16.2  # Annex I.9

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(
        x,
        l_ws,
        "-o",
        color=COLOR_MUTED,
        linewidth=2.0,
        markersize=5,
        zorder=4,
        label=(
            r"$L_{W\mathrm{s}}$ measured on the test plate"
            r" ($Y = 5.34\times10^{-6}$)"
        ),
    )
    ax.plot(
        x,
        l_wsn,
        "-s",
        color=COLOR_PRIMARY,
        linewidth=2.4,
        markersize=6,
        zorder=5,
        label=(
            r"$L_{W\mathrm{sn}}$ on the standard plate "
            r"($Y = 5\times10^{-6}$): "
            "what is declared"
        ),
    )
    ax.plot(
        x,
        installed,
        "-^",
        color=COLOR_TERTIARY,
        linewidth=2.4,
        markersize=6,
        zorder=5,
        label=(
            r"$L_{W\mathrm{s,inst}}$ on the receiving wall "
            r"($Y = 24.1\times10^{-6}$)"
        ),
    )
    ax.plot(
        x,
        characteristic,
        "-D",
        color=COLOR_SECONDARY,
        linewidth=2.4,
        markersize=6,
        zorder=5,
        label=(
            r"$L_{W\mathrm{s,c}}$ with the source mobility "
            r"($Y = 10^{-3}$): "
            "the input to EN 12354-5"
        ),
    )
    ax.plot(
        x,
        characteristic - d_c,
        "x",
        color=COLOR_FG,
        markersize=9,
        markeredgewidth=2.0,
        zorder=6,
        label=(
            rf"$L_{{W\mathrm{{s,c}}}} - D_\mathrm{{C}}$, "
            rf"$D_\mathrm{{C}}$ = {d_c:g} dB: "
            rf"back to $L_{{W\mathrm{{s,inst}}}}$"
        ),
    )

    # The step the section exists to teach, marked in the lowest band.
    ax.annotate(
        "",
        xy=(0, installed[0]),
        xytext=(0, l_wsn[0]),
        arrowprops={"arrowstyle": "<->", "lw": 1.6},
    )
    ax.text(
        0.12,
        0.5 * (l_wsn[0] + installed[0]),
        f"+{installed[0] - l_wsn[0]:.1f} dB = "
        "$10\\,\\mathrm{lg}(24.1/5.0)$:\n"
        "a lighter receiver accepts more power",
        fontsize=9,
        color=COLOR_FG,
        va="center",
    )

    _band_index_axis(ax, bands)
    ax.set_ylabel("Structure-borne power level [dB re 1 pW]")
    ax.set_title("One Source, Four Levels: the EN 15657 Conversion Chain", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.text(
        0.985,
        0.97,
        "EN 12354-5 Annex I.3 flushing cistern, wall contact\n"
        "markers reproduce the printed Table I.8 columns",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "structure_borne_conversion.svg")
    plt.close()
