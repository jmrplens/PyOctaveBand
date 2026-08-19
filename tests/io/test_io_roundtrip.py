#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Round trips at metrology precision: what write+read may and may not lose.

Three graduated guarantees, each stated as an executable bound rather than
a shrug: float64 loses nothing at all; 24-bit PCM loses at most half a
quantisation step per sample; and the absolute calibrated level survives
the full write+sidecar+read cycle -- including the calibration-cancellation
theorem, checked by running the calibrator tone and the measurement through
the same disk format and confirming the derived SPL is independent of the
depth the files were stored at.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phonometry import metrology, signals
from phonometry.io import Signal, read, write

FS = 48000


def _tone(amplitude: float, frequency: float, seconds: float) -> np.ndarray:
    t = np.arange(round(seconds * FS)) / FS
    return amplitude * np.sin(2 * np.pi * frequency * t)


def test_float64_write_read_is_the_identity(tmp_path: Path) -> None:
    rng = np.random.default_rng(414243)
    x = np.tan(rng.standard_normal(4096) * 0.7) * 0.01  # awkward mantissas
    path = tmp_path / "identity.wav"
    write(path, x, FS, subtype="DOUBLE")
    assert np.asarray(read(path)).tolist() == x.tolist()


def test_pcm24_error_is_bounded_by_half_a_quantisation_step(
    tmp_path: Path,
) -> None:
    """Round-to-nearest at 24 bits: |error| <= 2^-24, and typically at it."""
    rng = np.random.default_rng(444546)
    x = rng.uniform(-0.999, 0.999, 8192)
    path = tmp_path / "q24.wav"
    write(path, x, FS, subtype="PCM_24")
    error = np.abs(np.asarray(read(path)) - x)
    assert float(error.max()) <= 2.0**-24
    # The bound is tight, not slack: some sample sits near half a step.
    assert float(error.max()) > 2.0**-26


def test_calibrated_signal_survives_write_sidecar_read_exactly(
    tmp_path: Path,
) -> None:
    """The absolute level is intact after the full cycle, bit for bit."""
    x = _tone(0.05, 1000.0, 0.5)
    sig = Signal(data=x, fs=FS, calibration_factor=17.3)
    level_before = signals.leq(np.asarray(sig), calibration_factor=17.3)
    path = tmp_path / "calibrated.wav"
    write(path, sig, subtype="DOUBLE", sidecar=True)
    reread = read(path)  # no argument: the sidecar carries the calibration
    assert reread.calibration_factor == 17.3
    assert reread.calibration_factor is not None
    level_after = signals.leq(np.asarray(reread),
                              calibration_factor=reread.calibration_factor)
    assert level_after == level_before


def test_calibration_cancellation_holds_on_disk(tmp_path: Path) -> None:
    """The scaling-divisor choice cancels out of the calibrated level.

    The theorem in the reader's docstring, executed through the disk: the
    calibrator tone and the measurement pass through the same writer and
    reader, so whatever fixed scale the format applies divides out of
    ``sensitivity() x reading``. The derived SPL must therefore agree
    between an int16 chain and a float64 chain to within the 16-bit
    quantisation residue -- observed near 1.5e-4 dB here, where the
    phase-locked tone makes the quantisation error deterministic rather
    than noise-like -- even though the integer files hold codes 32768
    times larger than the float files. The 1e-3 dB bound is a hundredth
    of the 0.1 dB indication resolution of an IEC 61672 meter.
    """
    calibrator = _tone(0.5, 1000.0, 2.0)     # the 94 dB reference take
    measurement = _tone(0.05, 250.0, 2.0)    # the quiet measurement take

    levels: list[float] = []
    for subtype in ("PCM_16", "DOUBLE"):
        cal_path = tmp_path / f"cal_{subtype}.wav"
        meas_path = tmp_path / f"meas_{subtype}.wav"
        write(cal_path, calibrator, FS, subtype=subtype)
        write(meas_path, measurement, FS, subtype=subtype)
        factor = metrology.sensitivity(np.asarray(read(cal_path)), 94.0, fs=FS)
        levels.append(float(signals.leq(np.asarray(read(meas_path)),
                                        calibration_factor=factor)))

    assert abs(levels[0] - levels[1]) < 1e-3  # dB


def test_a_measurement_archived_to_flac_keeps_its_level(
    tmp_path: Path,
) -> None:
    """WAV 24-bit to FLAC: same codes, same sidecar, same absolute level."""
    import pytest

    pytest.importorskip("soundfile")
    x = _tone(0.05, 1000.0, 0.25)
    sig = Signal(data=x, fs=FS, calibration_factor=8.25)
    wav = tmp_path / "meas.wav"
    flac = tmp_path / "meas.flac"
    write(wav, sig, subtype="PCM_24", sidecar=True)
    write(flac, sig, subtype="PCM_24", sidecar=True)
    from_wav = read(wav)
    from_flac = read(flac)
    assert from_flac.calibration_factor == 8.25
    assert np.asarray(from_flac).tolist() == np.asarray(from_wav).tolist()
