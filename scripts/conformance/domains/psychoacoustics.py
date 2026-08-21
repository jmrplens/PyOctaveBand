#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Domain 3 - Psychoacoustics.

Loudness, sharpness and the equal-loudness contours: ISO 532-1 (Zwicker),
ISO 532-2 (Moore-Glasberg), ISO 532-3 (ECMA-418-2 Sottek Hearing Model),
DIN 45692 sharpness and the ISO 226 contours.

The expected values are the fixtures the standards themselves ship. ISO 532-1
Annex B publishes both the expected loudness of its test signals and the
signals, so these checks read the committed Annex B tables and WAV files
rather than a number typed out of the text.
"""

from __future__ import annotations

import json
import math
from typing import Literal, cast

import numpy as np
import reference_data as ref

import phonometry as ph
from phonometry.psychoacoustics.quality.sharpness import reference_sound

from ..registry import _DATA, Outcome, numeric, register
from .levels import _FS


def _iso532_stationary_expected() -> tuple[float, float, float]:
    data = json.loads(
        (_DATA / "iso532_1" / "iso532_1_annexB_expected.json").read_text()
    )
    entry = data["Test signal 1.txt"]
    return entry["N"], entry["Nmin"], entry["Nmax"]


def _iso532_levels() -> np.ndarray:
    path = _DATA / "iso532_1" / "iso532_1_test_signal_1_levels.txt"
    levels = [
        float(line.split(":")[1])
        for line in path.read_text().splitlines()
        if ":" in line and not line.strip().startswith("#")
    ]
    return np.array(levels)


@register(
    "Psychoacoustics",
    "Moore, Psychology of Hearing 6e, p. 77 (Glasberg & Moore 1990)",
    "ERB_N number of 1000 Hz = 15.59 Cam",
)
def _chk_cam_of_1_khz() -> Outcome:
    cam = float(np.asarray(ph.psychoacoustics.cam_from_frequency(1000.0))[()])
    return numeric(15.59, cam, 0.005, unit="Cam", places=4)


@register(
    "Psychoacoustics",
    "Moore, Psychology of Hearing 6e, p. 76 (Glasberg & Moore 1990)",
    "ERB_N at 1 kHz vs the printed 24.7(4.37F + 1), Hz",
)
def _chk_erb_bandwidth_1_khz() -> Outcome:
    published = 24.7 * (4.37 * 1.0 + 1.0)
    erb = float(np.asarray(ph.psychoacoustics.erb_bandwidth(1000.0))[()])
    return numeric(published, erb, 3e-3, unit="Hz", rel=True, places=3)


@register(
    "Psychoacoustics",
    "ISO 532-1:2017 Annex B.2",
    "Zwicker loudness N, stationary test signal 1",
)
def _chk_iso532_stationary() -> Outcome:
    expected_n, nmin, nmax = _iso532_stationary_expected()
    res = ph.psychoacoustics.loudness_zwicker_from_spectrum(
        _iso532_levels(), field="free"
    )
    computed = float(res.loudness)
    out = numeric(expected_n, computed, 0.001, unit="sone", rel=True, places=4)
    within_band = nmin <= computed <= nmax
    return Outcome(out.expected, out.computed, out.delta, out.passed and within_band)


def _iso532_b5_signal(
    num: int,
) -> tuple[np.ndarray, int, float, Literal["free", "diffuse"]]:
    """Load a recorded ISO 532-1 Annex B.5 technical signal and its expected values.

    Returns the pressure signal, its sample rate, the expected maximum loudness
    Nmax (sone) and the sound field the ISO results workbook was computed in.
    The 16-bit WAV is scaled per Annex B.1 (full scale = 100 dB SPL, so one
    full-scale unit is 2*sqrt(2) Pa peak).
    """
    import wave

    data = json.loads(
        (_DATA / "iso532_1" / "iso532_1_annexB_expected.json").read_text()
    )
    entry = data[f"Test signal {num}"]
    matches = sorted(
        (_DATA / "iso532_1" / "Annex B.5").glob(f"Test signal {num} *.wav")
    )
    if not matches:
        msg = (
            f"No ISO 532-1 Annex B.5 WAV found for Test signal {num} "
            "(see tests/data/iso532_1/README.md)."
        )
        raise FileNotFoundError(msg)
    # WAV/RIFF is little-endian, so the dtype is fixed to '<i2'; a multi-channel
    # file keeps only channel 0.
    with wave.open(str(matches[0])) as handle:
        fs = handle.getframerate()
        n_channels = handle.getnchannels()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels)[:, 0]
    signal = raw.astype(np.float64) / 32768.0 * (2.0 * math.sqrt(2.0))
    field = str(entry["field"])
    if field not in ("free", "diffuse"):
        msg = f"unexpected sound field {field!r} in the workbook"
        raise ValueError(msg)
    return (
        signal,
        int(fs),
        float(entry["Nmax"]),
        cast("Literal['free', 'diffuse']", field),
    )


@register(
    "Psychoacoustics",
    "ISO 532-1:2017 Annex B.5",
    "Time-varying loudness Nmax, technical signal 14 (aircraft, free field)",
)
def _chk_iso532_b5_free() -> Outcome:
    signal, fs, nmax, field = _iso532_b5_signal(14)
    res = ph.psychoacoustics.loudness_zwicker(signal, fs, field=field)
    return numeric(nmax, float(res.loudness), 0.001, unit="sone", rel=True, places=4)


@register(
    "Psychoacoustics",
    "ISO 532-1:2017 Annex B.5",
    "Time-varying loudness Nmax, technical signal 15 (vehicle interior, diffuse field)",
)
def _chk_iso532_b5_diffuse() -> Outcome:
    signal, fs, nmax, field = _iso532_b5_signal(15)
    res = ph.psychoacoustics.loudness_zwicker(signal, fs, field=field)
    return numeric(nmax, float(res.loudness), 0.001, unit="sone", rel=True, places=4)


@register(
    "Psychoacoustics",
    "DIN 45692:2009 Clause 6",
    "Sharpness of the standard 1 kHz reference signal",
)
def _chk_sharpness_reference() -> Outcome:
    # Definitional: the clause 5.2 constant k is derived so this very signal
    # reads 1.00 acum. The independent Table A.2 oracle is checked below.
    computed = float(ph.psychoacoustics.sharpness_din(reference_sound(), _FS))
    return numeric(1.0, computed, 1e-9, unit="acum", places=6)


@register(
    "Psychoacoustics",
    "DIN 45692:2009 Table A.2",
    "Sharpness of critical-band noise at 2.5 kHz (2320-2700 Hz, 4 sone)",
)
def _chk_sharpness_table_a2() -> Outcome:
    # Independent (non-definitional) oracle: the hearing-test Sollwert for
    # the 2.5 kHz critical-band noise is 1.78 acum at the clause 6 loudness
    # of 4 sone; permitted deviation 5 % or 0.05 acum.
    from scipy import signal as sp_signal

    def narrowband(level_db: float) -> np.ndarray:
        rng = np.random.default_rng(7)
        white = rng.standard_normal(_FS * 2)
        sos = sp_signal.butter(8, [2320.0, 2700.0], btype="band", fs=_FS, output="sos")
        nb = sp_signal.sosfilt(sos, white)
        return np.asarray(nb / np.sqrt(np.mean(nb**2)) * 2e-5 * 10 ** (level_db / 20))

    lo, hi = 30.0, 90.0
    for _ in range(13):  # set the clause 6 loudness of 4 sone
        mid = (lo + hi) / 2
        n = ph.psychoacoustics.loudness_zwicker(
            narrowband(mid), _FS, stationary=True
        ).loudness
        lo, hi = (mid, hi) if n < 4.0 else (lo, mid)
    computed = float(ph.psychoacoustics.sharpness_din(narrowband((lo + hi) / 2), _FS))
    return numeric(1.78, computed, 0.05 * 1.78, unit="acum", places=3)


@register(
    "Psychoacoustics",
    "ISO 226:2023 Table B.1",
    "Equal-loudness contour, 60 phon @ 100 Hz",
)
def _chk_iso226_contour() -> Outcome:
    # Anchored off 1 kHz (where SPL == phon is a trivial identity) at a
    # tabulated point that exercises the Table 1 contour formula.
    phon, freq, spl_ref = ref.ISO226_2023_TABLE_B1_ANCHOR
    freqs, spl = ph.psychoacoustics.equal_loudness_contour(phon)
    computed = float(spl[np.asarray(freqs) == freq][0])
    return numeric(spl_ref, computed, 0.05, unit="dB SPL", places=3)


# --- Block-A psychoacoustics: calibrated tones (pressure in pascals) --------
def _spl_tone(freq: float, level_db: float, seconds: float) -> np.ndarray:
    """Pure tone whose RMS is ``level_db`` dB re 20 uPa (pressure in Pa)."""
    t = np.arange(int(_FS * seconds)) / _FS
    amp = math.sqrt(2.0) * 2e-5 * 10.0 ** (level_db / 20.0)
    return np.asarray(amp * np.sin(2.0 * np.pi * freq * t))


def _am_tone(
    fc: float, fmod: float, depth: float, level_db: float, seconds: float
) -> np.ndarray:
    """Amplitude-modulated tone at an OVERALL RMS level (pressure in Pa).

    ECMA-418-2 Clause 7 states the calibration level as the sound pressure
    level of the signal, i.e. the overall RMS level of the modulated waveform.
    """
    t = np.arange(int(_FS * seconds)) / _FS
    x = (1.0 + depth * np.cos(2.0 * np.pi * fmod * t)) * np.sin(2.0 * np.pi * fc * t)
    return np.asarray(x * (2e-5 * 10.0 ** (level_db / 20.0)) / np.sqrt(np.mean(x**2)))


@register(
    "Psychoacoustics",
    "ECMA-418-2:2025 Clause 5.1.8",
    "HMS loudness of a 1 kHz / 40 dB tone (c_N=0.0211964)",
)
def _chk_ecma_loudness() -> Outcome:
    # With the full Clause 6.2.3 band averaging the chain computes 0.9845
    # sone_HMS for the calibration tone; the residual's origin is documented
    # in the loudness_ecma module docstring (c_N stays the verbatim value).
    computed = float(
        ph.psychoacoustics.loudness_ecma(_spl_tone(1000.0, 40.0, 0.6), _FS).loudness
    )
    return numeric(
        ref.ECMA418_2_LOUDNESS_1KHZ_40DB_SONE,
        computed,
        0.03,
        unit="sone_HMS",
        places=4,
    )


@register(
    "Psychoacoustics",
    "ECMA-418-2:2025 Clause 6.2.8",
    "HMS tonality of a 1 kHz / 40 dB tone (c_T=2.8758615)",
)
def _chk_ecma_tonality() -> Outcome:
    computed = float(
        ph.psychoacoustics.tonality_ecma(_spl_tone(1000.0, 40.0, 0.7), _FS).tonality
    )
    return numeric(
        ref.ECMA418_2_TONALITY_1KHZ_40DB_TU,
        computed,
        0.03,
        unit="tu_HMS",
        places=4,
    )


@register(
    "Psychoacoustics",
    "ECMA-418-2:2025 Clause 7",
    "HMS roughness of a 1 kHz / 70 Hz / m=1 / overall 60 dB tone (c_R=0.0180685)",
)
def _chk_ecma_roughness() -> Outcome:
    # Clause 7 calibration: the reference signal at an overall sound pressure
    # level of 60 dB SPL is 1 asper (computed: 0.9999 with the tabulated c_R).
    sig = _am_tone(1000.0, 70.0, 1.0, 60.0, 2.0)
    computed = float(ph.psychoacoustics.roughness_ecma(sig, _FS).roughness)
    return numeric(
        ref.ECMA418_2_ROUGHNESS_1KHZ_70HZ_60DB_ASPER,
        computed,
        0.01,
        unit="asper",
        places=4,
    )


@register(
    "Psychoacoustics",
    "ISO 532-2:2017 Clause 3.17 / Annex B.1",
    "Moore-Glasberg loudness of a 1 kHz / 40 dB tone (C=0.0617)",
)
def _chk_mg_loudness() -> Outcome:
    computed = float(
        ph.psychoacoustics.loudness_moore_glasberg_from_spectrum(
            [(1000.0, 40.0)]
        ).loudness
    )
    return numeric(
        ref.ISO532_2_ANCHOR_1KHZ_40DB_SONE, computed, 0.01, unit="sone", places=4
    )


@register(
    "Psychoacoustics",
    "ISO 532-3:2023 Annex C.1",
    "Moore-Glasberg-Schlittenlacher peak LTL, steady 1 kHz / 40 dB",
)
def _chk_mg_time_loudness() -> Outcome:
    fs = 32000.0
    t = np.arange(round(0.8 * fs)) / fs
    x = math.sqrt(2.0) * 2e-5 * 10.0 ** (40.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    computed = float(ph.psychoacoustics.loudness_moore_glasberg_time(x, fs).n_max)
    return numeric(
        ref.ISO532_3_ANCHOR_1KHZ_40DB_SONE, computed, 0.02, unit="sone", places=4
    )


@register(
    "Psychoacoustics",
    "ECMA-418-2:2025 Clause 9",
    "HMS fluctuation strength of a 1 kHz / 4 Hz / m=1 / overall 60 dB tone (c_F=0.003840572)",
)
def _chk_ecma_fluctuation_strength() -> Outcome:
    # Clause 9 calibration: the reference signal at an overall sound pressure
    # level of 60 dB SPL is 1 vacil_HMS (computed: 0.9931 for this 5 s signal
    # with the tabulated c_F; 0.9958 converged for longer signals).
    sig = _am_tone(1000.0, 4.0, 1.0, 60.0, 5.0)
    computed = float(
        ph.psychoacoustics.fluctuation_strength_ecma(sig, _FS).fluctuation_strength
    )
    return numeric(
        ref.ECMA418_2_FLUCTUATION_1KHZ_4HZ_60DB_VACIL,
        computed,
        0.01,
        unit="vacil_HMS",
        places=4,
    )
