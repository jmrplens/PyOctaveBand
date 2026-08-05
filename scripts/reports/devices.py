#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for the measuring and reproducing hardware itself.

What an instrument is rated at and whether it meets its class: the octave-band
filter class of IEC 61260-1 in both its editions, the class of a p-p intensity
chain (IEC 61043), and the rated characteristics a loudspeaker (IEC 60268-5)
and a microphone (IEC 60268-4) are declared with.
"""

from __future__ import annotations

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata


def _filter_class_example() -> tuple[object, ReportMetadata, str]:
    """Filter-compliance fiche: an IEC 61260-1 octave-band class verification.

    The library default (Butterworth order 6) octave bank from 125 Hz to
    4 kHz clears class 1 across every band, so the fiche boxes a Class 1
    COMPLIES result and passes the required-class-1 verdict.
    """
    bank = ph.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[125, 4000])
    result = ph.filter_class_compliance(bank)
    metadata = ReportMetadata(
        specimen="1/1-octave filter bank",
        client="Example client",
        manufacturer="Example instruments",
        test_room="Electroacoustics laboratory (example)",
        measurement_standard="IEC 61260-1:2014",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-61260",
        required_class=1,
    )
    return result, metadata, "iec61260_filter_example.pdf"


def _filter_class_1995_example() -> tuple[object, ReportMetadata, str]:
    """Filter-compliance fiche under the 1995 edition, which keeps class 0.

    IEC 61260-1:2014 dropped class 0; the older IEC 61260:1995 /
    ANSI S1.11-2004 retains it. Selecting ``edition="1995"`` verifies against
    that mask, and the default (order 6) octave bank clears the stricter
    class 0, so the fiche boxes a Class 0 COMPLIES result.
    """
    bank = ph.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[250, 4000])
    result = ph.filter_class_compliance(bank, edition="1995")
    metadata = ReportMetadata(
        specimen="1/1-octave filter bank",
        client="Example client",
        manufacturer="Example instruments",
        test_room="Electroacoustics laboratory (example)",
        measurement_standard="IEC 61260:1995",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-61260-1995",
        required_class=0,
    )
    return result, metadata, "iec61260_filter_1995_example.pdf"


def _intensity_class_example() -> tuple[object, ReportMetadata, str]:
    """IEC 61043 fiche: class verification of a p-p intensity chain.

    A complete instrument fitted with the common 12 mm spacer. The measured
    index follows the physics behind Table 2: a residual channel phase
    mismatch ``phi_s`` reads as ``delta_pI0 = 10 lg(kd/phi_s)``, so a mismatch
    that is constant in degrees already climbs 10 dB per decade (the slope of
    the requirement below 250 Hz) and levels off above 1 kHz where the
    mismatch of a real chain grows with frequency. A vent resonance of the
    capsules costs 4 dB around 100 Hz, the one band that drops out of class 1,
    so the fiche boxes a Class 2 COMPLIES result and fails the
    required-class-1 verdict, showing both halves of the layout at once.
    """
    spacing = 0.012
    freqs, _, _ = ph.residual_index_limits("instrument", spacing=spacing)
    phase_mismatch = 0.05 * np.maximum(1.0, freqs / 1000.0)  # degrees
    measured = ph.residual_index_from_phase_mismatch(phase_mismatch, freqs, spacing)
    measured = measured - 4.0 * np.exp(-((np.log(freqs / 100.0) / 0.25) ** 2))
    result = ph.intensity_class_compliance(measured, freqs, spacing=spacing)
    metadata = ReportMetadata(
        specimen="p-p sound intensity probe and analyser, 12 mm spacer",
        client="Example client",
        manufacturer="Example instruments",
        test_room="Electroacoustics laboratory (example)",
        measurement_standard="IEC 61043:1993",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-61043",
        required_class=1,
    )
    return result, metadata, "iec61043_intensity_example.pdf"


def _loudspeaker_example() -> tuple[object, ReportMetadata, str]:
    """IEC 60268-5 fiche: the rated characteristics of a two-way loudspeaker.

    A synthetic on-axis response of an 8 ohm bookshelf loudspeaker: a flat
    passband near 87 dB with a gentle ripple, a low-frequency roll-off below
    50 Hz and a high-frequency roll-off above 16 kHz, so the effective frequency
    range (IEC 60268-5 21.2) sits inside the measured band. The characteristic
    sensitivity is referred to 1 W into 8 ohm at 1 m (the default 2,83 V drive),
    and the impedance modulus, the total-harmonic-distortion curve and a baffled
    circular-piston directivity feed the impedance, THD and polar panels. The
    requirement is a characteristic sensitivity the example clears.
    """
    freqs = np.geomspace(30.0, 24000.0, 320)
    reference = 87.0
    spl = reference + 1.2 * np.sin(2.0 * np.log2(freqs / 900.0))
    spl -= 10.0 * np.log10(1.0 + (50.0 / freqs) ** 6)  # LF roll-off
    spl -= 10.0 * np.log10(1.0 + (freqs / 16000.0) ** 7)  # HF roll-off

    imp_freqs = np.geomspace(20.0, 20000.0, 260)
    impedance = (
        6.6
        + 24.0 * np.exp(-((np.log2(imp_freqs / 52.0)) ** 2) / 0.12)  # resonance peak
        + 5.0 * (imp_freqs / 20000.0) ** 1.5  # voice-coil rise
    )

    thd_freqs = np.geomspace(50.0, 5000.0, 140)
    thd_percent = 0.3 + 2.6 * np.exp(-((np.log2(thd_freqs / 70.0)) ** 2) / 0.45)

    angles = np.radians(np.linspace(0.0, 90.0, 46))
    directivity = ph.radiating_piston(
        0.075, np.array([1000.0, 2000.0, 4000.0]), angles=angles
    )

    result = ph.loudspeaker_characteristics(
        freqs,
        spl,
        8.0,
        sensitivity_band=(200.0, 4000.0),
        tolerance_db=3.0,
        rated_frequency_range=(45.0, 22000.0),
        rated_noise_power=80.0,
        rated_sinusoidal_power=120.0,
        resonance_frequency=52.0,
        impedance=(imp_freqs, impedance),
        distortion=(thd_freqs, thd_percent),
        directivity=directivity,
        polar_frequency=2000.0,
    )
    metadata = ReportMetadata(
        specimen="Two-way bookshelf loudspeaker, 165 mm woofer",
        client="Example client",
        manufacturer="Example audio",
        test_room="Anechoic chamber (example)",
        mounting="Free field, on the tweeter axis at 1 m",
        measurement_standard="IEC 60268-5",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.3,
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-60268-5",
        requirement=84.0,
    )
    return result, metadata, "iec60268_5_loudspeaker_example.pdf"


def _microphone_example() -> tuple[object, ReportMetadata, str]:
    """IEC 60268-4 fiche: the rated characteristics of a cardioid condenser mic.

    A synthetic free-field response of a phantom-powered studio condenser
    microphone: flat around the 1 kHz reference with a gentle +2 dB presence
    region near 9 kHz, a low-frequency roll-off crossing -3 dB near 30 Hz and a
    high-frequency roll-off crossing -3 dB near 19 kHz, so the effective
    frequency range (IEC 60268-4 12.2) sits inside the measured band. The rated
    free-field sensitivity of 12,5 mV/Pa gives a sensitivity level of
    20 lg 0,0125 = -38,1 dB re 1 V/Pa (11.1); the A-weighted inherent-noise
    voltage yields the equivalent noise level (17.2); the ideal-cardioid
    directional pattern at 1 kHz yields a directivity index of 10 lg 3 = 4,8 dB
    (13.2.2); and the distortion-against-level curve places the overload sound
    pressure level at the 0,5 % THD limit (15.2). The requirement is a maximum
    equivalent noise level the example clears.
    """
    freqs = np.geomspace(20.0, 20000.0, 400)
    response = (
        -10.0 * np.log10(1.0 + (30.0 / freqs) ** 4)  # LF roll-off
        - 10.0 * np.log10(1.0 + (freqs / 19000.0) ** 8)  # HF roll-off
        + 2.0 * np.exp(-((np.log2(freqs / 9000.0)) ** 2) / 0.3)  # presence
    )

    angles = np.linspace(0.0, 179.0, 359)
    cardioid_db = 20.0 * np.log10((1.0 + np.cos(np.radians(angles))) / 2.0)

    thd_spl = np.linspace(100.0, 140.0, 81)
    thd_percent = 0.5 * 10.0 ** ((thd_spl - 130.0) * 0.08)

    noise_freqs = np.geomspace(20.0, 20000.0, 31)
    noise_levels = (
        18.0 - 5.4 * np.log2(noise_freqs / 20.0) + 1.5 * np.sin(np.log2(noise_freqs))
    )

    result = ph.microphone_characteristics(
        freqs,
        response,
        12.5,
        tolerance_db=3.0,
        rated_impedance=150.0,
        minimum_load_impedance=1000.0,
        noise_voltage=1.25e-6,
        max_spl_thd_percent=0.5,
        distortion=(thd_spl, thd_percent),
        noise_spectrum=(noise_freqs, noise_levels),
        polar=(angles, cardioid_db),
        polar_frequency=1000.0,
        powering="Phantom P48 (IEC 61938)",
        supply_current_ma=3.1,
    )
    metadata = ReportMetadata(
        specimen="Cardioid condenser microphone, 25 mm capsule",
        client="Example client",
        manufacturer="Example audio",
        test_room="Anechoic chamber (example)",
        mounting="Free field, reference axis towards the source at 1 m",
        measurement_standard="IEC 60268-4",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.3,
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-60268-4",
        requirement=16.0,
    )
    return result, metadata, "iec60268_4_microphone_example.pdf"
