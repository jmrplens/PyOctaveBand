#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for vibration: human exposure and mechanical transmission.

Vibration as a dose and as a transmission path: the daily hand-arm exposure of
ISO 5349-2 and the multiple-shock response of a seated person (ISO 2631-5),
the driving-point mobility of a structure (ISO 7626) and the dynamic transfer
stiffness of a resilient mount (ISO 10846).
"""

from __future__ import annotations

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata


def _human_vibration_example() -> tuple[object, ReportMetadata, str]:
    """Human-vibration fiche: the ISO 5349-2 Annex E.3 forestry worker's day.

    Reproduces the ISO 5349-2:2001 Annex E.3 worked example (a forestry
    worker): brush-saw clearance for 2 h at a_hv = 4.6 m/s2, chain-saw felling
    for 1 h at 6.0 m/s2 and chain-saw branch stripping for 2 h at 3.6 m/s2. The
    standard gives the partial exposures A_i(8) = 2.3, 2.1 and 1.8 m/s2 and the
    combined daily exposure A(8) = 3.6 m/s2 (Eqs. (E.6)-(E.9)); the fiche shows
    the value to two decimals (3.61 m/s2), so the Directive 2002/44/EC hand-arm
    assessment places it in the action zone (>= 2.5 m/s2 EAV, < 5 m/s2 ELV).
    """
    result = ph.vibration.daily_vibration_exposure(
        [4.6, 6.0, 3.6],
        [2 * 3600.0, 1 * 3600.0, 2 * 3600.0],
        kind="hav",
        labels=[
            "Brush-saw clearance",
            "Chain-saw felling",
            "Chain-saw branch stripping",
        ],
    )
    metadata = ReportMetadata(
        client="Example forestry contractor",
        specimen="Forestry worker (right hand)",
        test_room="Managed woodland, plot 12",
        instrumentation="Hand-arm vibration meter (ISO 8041-1), s/n 0042",
        calibration="Field calibrator (ISO 8041-1) verified before/after the "
        "series within tolerance",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-5349",
        notes="Reproduces the ISO 5349-2:2001 Annex E.3 worked example.",
    )
    return result, metadata, "human_vibration_example.pdf"


def _multiple_shock_example() -> tuple[object, ReportMetadata, str]:
    """Multiple-shock fiche: the ISO 2631-5:2018 Annex C worked example.

    Reproduces the Annex C worked example, whose spinal response is five
    40 m/s2 shocks in the measured day for an 82 kg male exposed from age 20 for
    20 years at 120 days/year. From those response peaks the standard gives the
    daily acceleration dose Dzd = 1.07*(5*40**6)**(1/6) = 55.97 m/s2 (Formula 3;
    the measurement and daily periods coincide, so Dz = Dzd), the daily
    compressive stress Sd = mz*Dzd = 0.029*55.97 = 1.623 MPa (Formula C.1), the
    cumulative stress variable R = 1.22 (Formula C.3) and the probability of
    lumbar injury Pi = 0.37 (Formula C.5). Against the Table C.2 stress
    variables for men (R = 0.72 / 1.42 / 2.17 at 10 / 50 / 90 % risk of injury),
    R = 1.22 falls in the moderate band, matching the standard's own conclusion
    ("a moderate adverse health effect, 10 % < risk of injury < 50 %"). The
    result is built directly from the worked-example response peaks (which the
    standard states as the spinal response), so the fiche's numbers are the
    published Annex C values.
    """
    from phonometry.vibration.human.multiple_shock import (
        MZ_MALE,
        RISK_THRESHOLDS_MALE,
        MultipleShockResult,
        compression_dose,
        dose_from_peaks,
        injury_probability,
        injury_risk,
    )

    peaks = np.array([40.0] * 5)
    dz = dose_from_peaks(peaks)
    sd = compression_dose(dz, mz=MZ_MALE)
    r = injury_risk(sd, start_age=20, years=20, days_per_year=120, sex="male")
    result = MultipleShockResult(
        sex="male",
        acceleration_dose=dz,
        daily_dose=dz,
        compression_dose=sd,
        risk=r,
        probability=float(injury_probability(r, sex="male")),
        start_age=20.0,
        years=20,
        days_per_year=120.0,
        peaks=peaks,
        risk_thresholds=RISK_THRESHOLDS_MALE,
    )
    metadata = ReportMetadata(
        client="Example transport operator",
        specimen="82 kg male operator (seated)",
        test_room="Off-road vehicle, driver's seat",
        instrumentation="Seat-pad accelerometer (ISO 8041-1), s/n 0117",
        calibration="Reference calibrator (ISO 8041-1) verified before/after the "
        "series within tolerance",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-2631-5",
        notes="Reproduces the ISO 2631-5:2018 Annex C worked example.",
    )
    return result, metadata, "iso2631_5_multiple_shock_example.pdf"


def _mechanical_mobility_example() -> tuple[object, ReportMetadata, str]:
    """ISO 7626 fiche: the driving-point mechanical mobility of a resonator.

    The closed-form single-degree-of-freedom driving-point mobility of
    ISO 7626-1:2011 (Table 1 / 3.1.2), for a mass m = 2 kg on a stiffness
    k = 8000 N/m with viscous damping c = 5 N.s/m. The undamped natural
    frequency is f0 = (1/2pi) sqrt(k/m) = 10.07 Hz, and at that resonance the
    driving-point mobility is purely real and equal to 1/c = 0.2 m/(N.s) (the
    mobility peak measures the damping); including f0 in the log-spaced axis
    lands the peak exactly on it. These are the module test's oracle values.
    """
    import math

    mass, stiffness, damping = 2.0, 8000.0, 5.0
    f0 = math.sqrt(stiffness / mass) / (2.0 * math.pi)
    freqs = np.unique(np.append(np.logspace(0.0, np.log10(200.0), 300), f0))
    result = ph.vibration.sdof_mobility_result(freqs, mass, stiffness, damping)
    metadata = ReportMetadata(
        specimen="Machine support bracket (driving point)",
        client="Example client",
        manufacturer="Example structures",
        test_room="Modal-analysis rig (example)",
        instrumentation="Impact hammer + accelerometer, H1 estimator (ISO 7626-2)",
        measurement_standard="ISO 7626-2",
        temperature=21.0,
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-7626",
    )
    return result, metadata, "iso7626_mobility_example.pdf"


def _transfer_stiffness_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10846 fiche: the dynamic transfer stiffness of a resilient mount.

    A viscously damped resilient element (a Kelvin-Voigt mount, the module's
    documented element model) with stiffness k = 1 MN/m and damping
    c = 80 N.s/m has a transfer stiffness k2,1(f) = k + j*omega*c that is a
    plateau at low frequency (|k2,1| -> k, the static stiffness) rising with
    frequency as the damping term grows. The direct method (ISO 10846-2:2008)
    measures it as k2,1 = F2,b/u1; synthesising the blocked output force
    F2,b = k2,1 * u1 from a 1 um input displacement u1 and feeding it back
    through ``transfer_stiffness_direct`` recovers the closed form exactly, so
    the printed values match the module's tested oracle. At the 20 Hz plateau
    |k2,1| = 1.00 MN/m, L_k = 20 lg(|k2,1|/k0) = 120.0 dB re 1 N/m and the loss
    factor eta = Im/Re = 0.010 (ISO 10846-1:2008, 3.8).
    """
    freqs = np.array(
        [
            20,
            25,
            31.5,
            40,
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
        ],
        dtype=float,
    )
    stiffness, damping = 1.0e6, 80.0
    omega = 2.0 * np.pi * freqs
    k21 = stiffness + 1j * omega * damping
    u1 = 1.0e-6 + 0.0j
    measured = ph.vibration.transfer_stiffness_direct(k21 * u1, u1)
    result = ph.vibration.TransferStiffnessResult(
        frequencies=freqs, transfer_stiffness=measured, blocking_mass=None
    )
    metadata = ReportMetadata(
        specimen="Rubber vibration isolator (resilient mount)",
        client="Example client",
        manufacturer="Example elastomers",
        test_room="Transfer-stiffness rig (example)",
        instrumentation="Force transducer + accelerometers (ISO 10846-2)",
        measurement_standard="ISO 10846-2",
        temperature=21.0,
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10846",
    )
    return result, metadata, "iso10846_transfer_stiffness_example.pdf"
