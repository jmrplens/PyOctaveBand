#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figure for the audio-files guide: what ``read()`` returns from a file.

One concept, drawn once: a measurement WAV comes back as a calibrated
waveform *and* the provenance that makes it a measurement -- the ``bext``
broadcast metadata (EBU Tech 3285), the container facts, and the calibration
that arrived through the sidecar. The file is written and read back through
the public API inside the generator, so the panel shows what a user's own
``Signal`` carries, not a mock-up. Embedded by ``docs/io/audio-files.md``
and its site twins.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .i18n import _LANG
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    save_figure,
)


def _night_measurement(fs: int) -> np.ndarray:
    """Ten seconds of facade monitoring: low background, three pass-bys.

    Digital full-scale units, sized so the calibration of the figure (20 Pa
    per full-scale unit, i.e. 0 dBFS = 120 dB SPL) puts the background near
    60 dB and the loudest pass-by near the high eighties -- the levels a
    night traffic measurement actually shows.
    """
    rng = np.random.default_rng(2026)
    n = 10 * fs
    x = 0.0012 * rng.standard_normal(n)
    for start_s, length_s, amplitude in (
        (1.6, 2.2, 0.011),
        (4.9, 1.4, 0.020),
        (7.8, 1.8, 0.008),
    ):
        start = int(start_s * fs)
        length = int(length_s * fs)
        envelope = np.hanning(length)
        x[start:start + length] += (
            amplitude * envelope * rng.standard_normal(length)
        )
    return x


def generate_signal_provenance(output_dir: str) -> None:
    """The calibrated waveform next to the provenance card it arrived with."""
    print("Generating signal_provenance.png...")
    from phonometry import io

    fs = 48000
    # Samples since midnight for the first sample: 02:37:05, exactly.
    midnight_samples = (2 * 3600 + 37 * 60 + 5) * fs
    bext = io.BroadcastMetadata(
        description="Night monitoring, facade position P3",
        originator="Hand-held class 1 analyzer",
        originator_reference="ESPHN20260621023705000012",
        origination_date="2026-06-21",
        origination_time="02:37:05",
        time_reference=midnight_samples,
        version=2,
        umid=None,
        loudness_value=None,
        loudness_range=None,
        max_true_peak_level=None,
        max_momentary_loudness=None,
        max_short_term_loudness=None,
        coding_history="A=PCM,F=48000,W=24,M=mono,T=SLM",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "night_p3.wav")
        io.write(
            path,
            io.Signal(data=_night_measurement(fs), fs=fs,
                      calibration_factor=20.0),
            subtype="PCM_24", bext=bext, sidecar=True,
        )
        sig = io.read(path)

    fig = plt.figure(figsize=(10.0, 4.6))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.16)
    ax = fig.add_subplot(grid[0, 0])
    sig.plot(ax=ax, language=_LANG)
    ax.set_title("The samples, back in pascals", pad=12)
    ax.set_xlim(0.0, sig.duration)

    card = fig.add_subplot(grid[0, 1])
    card.set_axis_off()
    card.set_title("What travelled with them", pad=12)
    card.add_patch(Rectangle(
        (0.0, 0.0), 1.0, 1.0, transform=card.transAxes,
        facecolor=COLOR_PANEL, edgecolor=COLOR_GRID, linewidth=1.0,
        zorder=0,
    ))

    origin = sig.source
    provenance = sig.provenance
    assert origin is not None and provenance is not None
    history_lines = provenance.coding_history.splitlines()
    appended = len(history_lines) - 1
    container = f"{origin.container}, {origin.bit_depth}-bit {origin.format_name}"
    recorded_on = (
        f"{provenance.origination_date} at {provenance.origination_time}"
    )
    # Thousands grouped with spaces, so the Spanish edition needs no comma
    # surgery on a number that is not a decimal.
    first_sample = (
        f"{provenance.time_reference:,} samples"
    ).replace(",", " ")
    calibration = (
        f"{sig.calibration_factor:.1f} Pa at full scale, from the sidecar"
    )
    rows: tuple[tuple[str, str], ...] = (
        ("File", os.path.basename(origin.path)),
        ("Container", container),
        ("Sample rate", f"{sig.fs / 1000:g} kHz"),
        ("Recorded by", str(provenance.originator)),
        ("Recorded on", recorded_on),
        ("First sample (since midnight)", first_sample),
        ("Coding history", history_lines[0]),
        ("", f"+ {appended} line appended by write()"),
        ("Calibration", calibration),
    )
    y = 0.915
    for label, value in rows:
        if label:
            card.text(0.06, y, label, transform=card.transAxes,
                      fontsize=8.2, color=COLOR_MUTED, va="top")
            y -= 0.048
        card.text(0.06, y, value, transform=card.transAxes,
                  fontsize=9.2, color=COLOR_FG, va="top")
        y -= 0.056
    save_figure(output_dir, "signal_provenance.png")
    plt.close()
