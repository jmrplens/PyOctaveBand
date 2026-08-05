#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for room acoustics: the sound field inside one room.

What the finished room does to the sound in it: the ISO 3382-1/-2 parameters
read from its impulse response, the reverberation time predicted from its
absorption, the absorption area of an enclosed space (EN 12354-6), the speech
decay along an open-plan office (ISO 3382-3), and the NC and RC criteria that
rate the noise sitting in the room while nobody speaks.
"""

from __future__ import annotations

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata


def _room_acoustics_example() -> tuple[object, ReportMetadata, str]:
    """Room-acoustics fiche: ISO 3382-1/-2 parameters of a small auditorium.

    The impulse response is a deterministic single-slope synthetic decay: one
    sine carrier per octave band (125 Hz to 4 kHz), each modulated by its own
    exponential energy envelope exp(-A60*t/T) with A60 = 6*ln(10), so every
    octave-band Schroeder curve is an exact straight line. The closed-form
    decay time of a pure exponential energy decay is therefore T20 = T30 = EDT
    = T per band (ISO 3382-1:2009, 5.3.3 gives L(t) = -60*t/T dB), which fixes
    the reverberation-time column exactly at the chosen per-band values
    (1.40, 1.30, 1.20, 1.10, 1.00, 0.85 s), a plausible mid-size hall profile
    falling with frequency. The energy parameters C50/C80/D50/Ts sit slightly
    below the single-slope closed form because the octave band-pass group delay
    smears a little early energy past the 50/80 ms limits (documented in the
    room_acoustics module and its tests); the mid-frequency descriptor is
    T_mid = (T30@500 + T30@1000)/2 = 1.15 s.
    """
    fs = 48000
    a60 = 6.0 * np.log(10.0)
    bands = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
    t60 = (1.40, 1.30, 1.20, 1.10, 1.00, 0.85)
    time = np.arange(round(5.0 * fs)) / fs
    ir = np.zeros_like(time)
    for freq, decay in zip(bands, t60):
        ir += np.sin(2.0 * np.pi * freq * time) * np.exp(-0.5 * a60 * time / decay)
    result = ph.room_parameters(ir, fs)
    metadata = ReportMetadata(
        specimen="Small auditorium, unoccupied, fully furnished",
        client="Example client",
        test_room="Auditorium A (example)",
        room_volume=2830.0,
        area=340.0,
        source_positions=2,
        receiver_positions=8,
        instrumentation="Omnidirectional source + 1/2 in. microphone (example)",
        measurement_standard="ISO 3382-1",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-3382",
        requirement=1.3,
    )
    return result, metadata, "iso3382_room_acoustics_example.pdf"


def _reverberation_prediction_example() -> tuple[object, ReportMetadata, str]:
    """Reverberation-time prediction fiche: five models over the octave bands.

    A shoebox classroom 8 x 5 x 3 m (V = 120 m3, S = 158 m2, the geometry of
    the reverberation_prediction module tests) with an anisotropic absorption
    distribution: one wall pair treated with a broadband absorber (alpha rising
    with frequency), the other two pairs lightly absorptive. The anisotropy is
    what separates the five models, so the fiche compares them meaningfully.
    The values printed are computed by the classical closed-form models
    themselves (each anchored by the module tests), so the fiche needs no
    separate numeric oracle. It is a design-stage prediction, not a
    measurement: the five models bracket the reverberation time likely to
    occur.
    """
    freqs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    treated = [0.10, 0.15, 0.30, 0.45, 0.55, 0.60]  # broadband absorber wall
    side = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]      # lightly absorptive walls
    floor_ceiling = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18]
    result = ph.reverberation_time_models(
        (8.0, 5.0, 3.0), (treated, side, floor_ceiling), frequencies=freqs
    )
    metadata = ReportMetadata(
        specimen="Classroom, one wall lined with a broadband absorber",
        client="Example client",
        test_room="Classroom C1 (example)",
        temperature=20.0,
        relative_humidity=50.0,
        pressure=101.3,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-REVERB",
        requirement=0.8,
    )
    return result, metadata, "reverberation_prediction_example.pdf"


def _enclosed_space_absorption_example() -> tuple[object, ReportMetadata, str]:
    """Enclosed-space fiche: absorption area A and reverberation time T (EN 12354-6).

    A small 5 x 4 x 2.5 m meeting room (V = 50 m3) characterised over the
    standard octave bands by the EN 12354-6:2003 Clause 4 model: a carpeted
    floor, an acoustic-tile ceiling and painted-plaster walls, a few hard
    objects (their equivalent area from Formula 4) giving a small object
    fraction, and the recommended 20 degC / 50-70 % air-attenuation profile.
    A and T are computed by the tested Formula 1 / Formula 5 primitives, so the
    fiche needs no separate numeric oracle. EN 12354-6 gives a diffuse-field
    estimate, not a measurement.
    """
    surfaces = [
        (20.0, [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55]),   # carpeted floor
        (20.0, [0.20, 0.40, 0.65, 0.75, 0.80, 0.80, 0.75]),   # acoustic ceiling
        (45.0, [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05]),   # painted-plaster walls
    ]
    object_volumes = [0.5, 0.8, 0.3]  # furniture and fittings, m3
    objects = ph.hard_object_absorption(object_volumes)
    psi = ph.object_fraction(object_volumes, 50.0)
    result = ph.enclosed_space_reverberation(
        surfaces, 50.0, objects=objects, object_fraction=psi,
        air_condition="20C_50-70",
    )
    metadata = ReportMetadata(
        specimen="Meeting room, furnished",
        client="Example client",
        test_room="Meeting room M2 (example)",
        measurement_standard="EN 12354-6",
        temperature=20.0,
        relative_humidity=55.0,
        pressure=101.3,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-EN12354-6",
        requirement=0.6,
    )
    return result, metadata, "enclosed_space_absorption_example.pdf"


def _noise_criteria_example() -> tuple[object, ReportMetadata, str]:
    """Noise Criteria fiche: an office spectrum rated NC-40 (ANSI/ASA S12.2).

    The spectrum is built from the ANSI/ASA S12.2-2019 Table 1 NC-40 contour so
    the tangency rating is exact and independently verifiable. Every band is
    depressed 5 dB below its NC-40 contour except the 250 Hz octave, which is
    left on the NC-40 curve (50 dB, the Table 1 value). The SIL is 35.5 dB and
    the 250 Hz band exceeds the NC-36 curve, so per clause 5.2.2 the rating
    falls to the tangency method, which returns NC-40 with the 250 Hz band
    governing.
    """
    contour = ph.nc_curve(40.0)
    levels = contour - 5.0
    levels[4] = contour[4]  # 250 Hz sits on the NC-40 curve and governs.
    result = ph.noise_criterion(levels)
    metadata = ReportMetadata(
        specimen="Open-plan office, air handling at nominal flow",
        client="Example client",
        test_room="Office A (example)",
        room_volume=180.0,
        area=60.0,
        instrumentation="Class 1 sound level meter + octave filter set (example)",
        measurement_standard="ANSI/ASA S12.2",
        temperature=22.0,
        relative_humidity=42.0,
        pressure=101.2,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-S122-NC",
        requirement=40.0,
    )
    return result, metadata, "ansi_s12_2_noise_criteria_example.pdf"


def _room_criteria_example() -> tuple[object, ReportMetadata, str]:
    """Room Criteria fiche: an RC-35(R) rumble spectrum (ANSI/ASA S12.2 Annex D).

    The spectrum is the ANSI/ASA S12.2-2019 Annex D RC-35 Mark II curve with the
    250 Hz octave raised 8 dB. The 500/1000/2000 Hz mid bands are unchanged, so
    the mid-frequency average LMF is exactly 35 dB (clause D.4) and the rating
    is RC-35; the raised low band exceeds the reference by more than 5 dB, so
    the spectral-quality tag is rumble (clause D.3), giving RC-35(R).
    """
    levels = ph.rc_curve(35.0)
    levels[4] += 8.0  # 250 Hz rumble.
    result = ph.room_criterion(levels)
    metadata = ReportMetadata(
        specimen="Conference room, variable-air-volume terminal",
        client="Example client",
        test_room="Room B (example)",
        room_volume=140.0,
        area=48.0,
        instrumentation="Class 1 sound level meter + octave filter set (example)",
        measurement_standard="ANSI/ASA S12.2",
        temperature=21.5,
        relative_humidity=44.0,
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-S122-RC",
        requirement=40.0,
    )
    return result, metadata, "ansi_s12_2_room_criteria_example.pdf"


def _open_plan_example() -> tuple[object, ReportMetadata, str]:
    """Open-plan-office fiche: ISO 3382-3 spatial decay of a good open office.

    The measurement line is built from a closed-form model so the four
    single-number quantities are exact and independently verifiable. The
    A-weighted speech level is collinear in the logarithmic distance axis,
    Lp,A,S(r) = 62.0 - 7.0*log2(r) dB, so the ISO 3382-3:2012 Clause 6.2
    least-squares fit recovers D2,S = 7.0 dB per distance doubling and
    Lp,A,S,4m = 62 - 7*log2(4) = 48.0 dB exactly. The STI is linear in
    distance, STI(r) = 0.65 - 0.03*r, so the Clause 6.3 STI-vs-distance
    regression crosses 0.50 at rD = (0.50 - 0.65)/(-0.03) = 5.0 m and 0.20 at
    rP = (0.20 - 0.65)/(-0.03) = 15.0 m. Seven positions span the 2 m to 16 m
    range (in the 6 to 10 preferred by Clause 5.2.2); this is a "good" office
    (Annex A: D2,S >= 7 dB, Lp,A,S,4m <= 48 dB, rD <= 5 m).
    """
    positions = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0])
    spl_a_speech = 62.0 - 7.0 * np.log2(positions)
    sti = 0.65 - 0.03 * positions
    result = ph.room.open_plan_metrics(positions, spl_a_speech, sti)
    metadata = ReportMetadata(
        specimen="Furnished, unoccupied, background noise present",
        client="Example client",
        test_room="Open-plan office B (example)",
        area=420.0,
        source_positions=2,
        receiver_positions=7,
        instrumentation="Omnidirectional source + class 1 SLM (example)",
        measurement_standard="ISO 3382-3",
        temperature=22.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-3382-3",
        requirement=7.0,
    )
    return result, metadata, "iso3382_3_open_plan_example.pdf"
