#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for the listener: loudness, intelligibility and hearing damage.

What the sound is to the person in front of it: the stationary loudness of
ISO 532-1 and the programme loudness of EBU R 128, the speech intelligibility
of IEC 60268-16 (STI) and ANSI S3.5 (SII), and the occupational exposure of
ISO 9612 with the hearing loss ISO 1999 predicts from a working life of it.
"""

from __future__ import annotations

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata


def _loudness_example() -> tuple[object, ReportMetadata, str]:
    """Loudness fiche: an ISO 532-1 Zwicker stationary loudness rating."""
    # A shaped 28-band one-third-octave spectrum (25 Hz..12.5 kHz) of a steady
    # appliance noise, descending with frequency.
    levels = np.array(
        [
            55,
            55,
            54,
            53,
            52,
            51,
            50,
            49,
            48,
            47,
            46,
            45,
            44,
            43,
            42,
            41,
            40,
            39,
            38,
            37,
            36,
            35,
            34,
            33,
            32,
            31,
            30,
            29,
        ],
        dtype=float,
    )
    result = ph.psychoacoustics.loudness_zwicker_from_spectrum(levels, field="free")
    metadata = ReportMetadata(
        specimen="Household appliance, steady operating noise",
        client="Example client",
        manufacturer="Example appliances",
        test_room="Hemi-anechoic room (example)",
        measurement_standard="ISO 532-1 method 1",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-532",
        requirement=12.0,
    )
    return result, metadata, "iso532_loudness_example.pdf"


def _program_loudness_example() -> tuple[object, ReportMetadata, str]:
    """Programme-loudness fiche: an EBU R 128 compliance measurement.

    The signal is the EBU Tech 3342 Case 1 loudness-range shape (two 20 s
    stereo 1 kHz tone segments 10 dB apart) trimmed 0.4 dB to -20.4 and
    -30.4 dBFS, giving a loudness range near 10 LU with the integrated
    loudness on the -23.0 LUFS target, inside the default +-0.2 LU QC
    tolerance of EBU R 128 item i).
    """
    fs = 48000
    t = np.arange(round(20.0 * fs)) / fs
    tone = np.sin(2.0 * np.pi * 1000.0 * t)
    seg_hi = 10.0 ** (-20.4 / 20.0) * tone
    seg_lo = 10.0 ** (-30.4 / 20.0) * tone
    mono = np.concatenate([seg_hi, seg_lo])
    signal = np.vstack([mono, mono])
    result = ph.broadcast.program_loudness(signal, fs)
    metadata = ReportMetadata(
        specimen="Reference tone sequence (-20.4 / -30.4 dBFS steps)",
        client="Example broadcaster",
        manufacturer="Example post-production",
        test_room="Reference monitoring room (example)",
        measurement_standard="EBU R 128",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-R128",
        requirement=-23.0,
    )
    return result, metadata, "ebu_r128_loudness_example.pdf"


def _occupational_exposure_example() -> tuple[object, ReportMetadata, str]:
    """ISO 9612 fiche: the Annex D task-based welders' day.

    Reproduces the ISO 9612:2009 Annex D worked example: a welder's nominal day
    split into planning/breaks (a single conservative 70 dB value, 1.5 h),
    welding (three samples, 5 h with a 4 h to 6 h duration range) and
    cutting/grinding (six samples after the 3 dB spread rule asked for more,
    1.5 h with a 1 h to 2 h range), measured with a personal sound exposure
    meter. The daily level is LEX,8h = 84.3 dB and, with the duration
    uncertainty included, U = 3.2 dB; the fiche's Directive 2003/10/EC
    assessment shows the lower action value (80 dB(A)) exceeded and the upper
    action value (85 dB(A)) and limit value (87 dB(A)) not exceeded.
    """
    tasks = [
        ph.hearing.Task(
            samples=(70.0,), duration_hours=1.5, label="Planning and breaks"
        ),
        ph.hearing.Task(
            samples=(80.1, 82.2, 79.6),
            duration_hours=5.0,
            duration_range=(4.0, 6.0),
            label="Welding",
        ),
        ph.hearing.Task(
            samples=(86.5, 92.4, 89.3, 93.2, 87.8, 86.2),
            duration_hours=1.5,
            duration_range=(1.0, 2.0),
            label="Cutting and grinding",
        ),
    ]
    result = ph.hearing.task_based_exposure(tasks, warn=False)
    metadata = ReportMetadata(
        client="Example fabrication works",
        specimen="Welders (homogeneous exposure group, 4 workers)",
        test_room="Steel assembly hall, line 2",
        instrumentation="Personal sound exposure meter (IEC 61252), s/n 0042",
        calibration="Calibrator IEC 60942 class 1, s/n 0117; field checks "
        "before/after each series within 0.3 dB",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9612",
        notes="Reproduces the ISO 9612:2009 Annex D task-based worked example.",
    )
    return result, metadata, "iso9612_exposure_example.pdf"


def _nipts_example() -> tuple[object, ReportMetadata, str]:
    """NIPTS fiche: the ISO 1999 Annex D 90 dB / 20 year prediction.

    Reproduces the ISO 1999:2013 Annex D (Table D.2) worked example of the
    noise-induced permanent threshold shift for L_EX,8h = 90 dB and 20 years of
    exposure, at the most-susceptible tenth (population fractile Q = 0.90). The
    median 4 kHz shift is N50 = 12.9 dB and the fractile value is 17.8 dB; the
    fiche boxes the shift averaged over the 2/3/4 kHz hearing-handicap set. It
    is a statistical prediction for the exposed population, not a clinical
    diagnosis.
    """
    result = ph.hearing.nipts(90.0, 20.0, 0.9)
    metadata = ReportMetadata(
        client="Example fabrication works",
        specimen="Welders (homogeneous exposure group, 4 workers)",
        test_room="Steel assembly hall, line 2",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-1999-NIPTS",
        notes="Reproduces the ISO 1999:2013 Annex D (Table D.2) worked example.",
    )
    return result, metadata, "iso1999_nipts_example.pdf"


def _htlan_example() -> tuple[object, ReportMetadata, str]:
    """HTLAN fiche: a 60-year-old worker's age-plus-noise threshold prediction.

    Predicts the hearing threshold level associated with age and noise
    (ISO 1999:2013 clause 6.1) for a 60-year-old man exposed at
    L_EX,8h = 95 dB for 30 years, at the median (population fractile Q = 0.50).
    The age component H (HTLA, database A = ISO 7029) and the noise component N
    (NIPTS) combine by H' = H + N - H*N/120; the fiche boxes the combined
    threshold averaged over the 2/3/4 kHz hearing-handicap set. It is a
    statistical prediction, not a clinical audiogram.
    """
    result = ph.hearing.htlan(60, "male", 95.0, 30.0, 0.5)
    metadata = ReportMetadata(
        client="Example fabrication works",
        specimen="Machine operator (60 years, 30 years in role)",
        test_room="Steel assembly hall, line 2",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-1999-HTLAN",
        notes="Predicted age-plus-noise threshold per ISO 1999:2013 clause 6.1.",
    )
    return result, metadata, "iso1999_htlan_example.pdf"


def _sti_example() -> tuple[object, ReportMetadata, str]:
    """STI fiche: a voice-alarm intelligibility verification (IEC 60268-16).

    A deterministic full-STI indirect measurement: the octave-band modulation
    transfer function is taken from a noise-carrier impulse response with an
    exponential energy decay p(t) ~ exp(-13.8 t / T60) at T60 = 0.8 s (the
    fixed-seed carrier makes the run reproducible). The closed-form Schroeder
    MTF of that decay, m(F) = 1 / sqrt(1 + (2 pi F T / 13.8)^2), is uniform
    across the seven bands and rates to STI = 0.639 (Annex F band D); the
    finite noise carrier adds the documented small positive bias, so the
    fiche prints STI = 0.64. The requirement is the STI >= 0.5 that
    IEC 60268-16 associates with a usable public-address / voice-alarm system,
    which the example clears.
    """
    fs = 48000
    t60 = 0.8
    rng = np.random.default_rng(0)
    n = int(2.0 * t60 * fs)
    t = np.arange(n) / fs
    ir = rng.standard_normal(n) * np.exp(-3.0 * np.log(10.0) * t / t60)
    result = ph.speech.sti_from_impulse_response(ir, fs)
    metadata = ReportMetadata(
        specimen="Concourse voice-alarm loudspeaker line",
        client="Example client",
        manufacturer="Example audio systems",
        test_room="Transport terminal concourse (example)",
        measurement_standard="IEC 60268-16",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-STI",
        requirement=0.5,  # STI = 0.64 >= 0.50 -> PASS (usable intelligibility)
    )
    return result, metadata, "iec60268_16_sti_example.pdf"


def _sii_example() -> tuple[object, ReportMetadata, str]:
    """SII fiche: a speech-audibility assessment (ANSI S3.5-1997).

    The R CRAN package "SII" worked Example C.2 (an independent implementation
    of the one-third-octave-band method): an equivalent speech spectrum of
    54 dB SPL in every band, a low-frequency ambient noise of 40, 30 and 20 dB
    in the first three bands and normal hearing. The procedure rates to
    SII = 0.851, so the fiche prints 0.851. The requirement is a minimum SII of
    0.75 (an audibility target for good intelligibility), which the example
    clears.
    """
    result = ph.speech.speech_intelligibility_index(
        np.full(18, 54.0),
        np.array([40.0, 30.0, 20.0] + [0.0] * 15),
        threshold=np.zeros(18),
    )
    metadata = ReportMetadata(
        specimen="Conversational speech in low-frequency ambient noise",
        client="Example client",
        test_room="Open-plan office listening position (example)",
        measurement_standard="ANSI S3.5-1997",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-SII",
        requirement=0.75,  # SII = 0.851 >= 0.75 -> PASS (good audibility)
    )
    return result, metadata, "ansi_s3_5_sii_example.pdf"
