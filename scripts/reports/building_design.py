#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for predicted sound insulation: the building before it is built.

Insulation computed from element and junction data instead of measured in a
finished room: the simplified single-number models of EN 12354-1/-2/-3, the
detailed per-band model of ISO 12354-1/-2 over the Annex L/G building, and the
structure-borne source data of EN 15657 with the installed power EN 12354-5
predicts from it.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata

# The ISO 12354 Annex L / Annex G building is assembled once, in tests/, and
# read from there by the tests and by scripts/conformance_report.py. The two
# detailed-prediction fiches below show that same building, so they read it
# from the same place rather than becoming a third transcription of a worked
# example whose inputs the registry already records two corrections to.
_TESTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import reference_data as ref


def _airborne_prediction_example() -> tuple[object, ReportMetadata, str]:
    """Airborne prediction fiche: EN 12354-1 Annex H.3 worked example.

    A separating wall Rs,w = 57 dB, area Ss = 11.5 m2, flanked by four elements
    (floor, ceiling, facade, internal wall) whose Annex H tabulated junction
    Kij feed twelve flanking paths; the direct Dd path plus those twelve make
    thirteen paths whose energy summation (Formula 26) gives R'w = 52.2 -> 52
    dB. The element set and its R'w are the standard's own worked example, run
    through the tested prediction code (see tests/reference_data/).
    """
    ss = 11.5
    paths = []
    # (label, R_flanking,w, KFf, KFd = KDf, coupling length lf) from Annex H.
    elements = [
        ("floor", 49.0, 12.4, 8.9, 4.5),
        ("ceiling", 46.0, 14.4, 9.2, 4.5),
        ("facade", 42.0, 12.6, 6.7, 2.55),
        ("intwall", 33.0, 33.5, 15.7, 2.55),
    ]
    for label, rw, kff, kfd, lf in elements:
        ff, df, fd = ph.flanking_element(
            label=label,
            r_flanking=rw,
            r_separating=57.0,
            k_ff=kff,
            k_fd=kfd,
            k_df=kfd,
            separating_area=ss,
            coupling_length=lf,
        )
        paths += [ff, df, fd]
    result = ph.predicted_airborne_insulation(r_direct=57.0, flanking_paths=paths)
    metadata = ReportMetadata(
        specimen="Separating wall, Rs,w = 57 dB (EN 12354-1 Annex H.3)",
        client="Example client",
        area=11.5,
        source_volume=53.0,
        receiving_volume=50.0,
        test_room="Dwelling A to dwelling B (example)",
        measurement_standard="EN/ISO 12354-1",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-12354-1",
        notes=(
            "Flanking elements: floor Rw = 49, ceiling 46, facade 42, "
            "internal wall 33 dB; junctions per Annex E/H. Simplified "
            "single-number model (Clause 4.4)."
        ),
        requirement=50.0,
    )
    return result, metadata, "iso12354_airborne_prediction_example.pdf"


def _impact_prediction_example() -> tuple[object, ReportMetadata, str]:
    """Impact prediction fiche: EN 12354-2 Annex E.3 worked example.

    A 0.14 m concrete floor (m' = 322 kg/m2) has a bare-floor equivalent level
    Ln,w,eq = 164 - 35 lg(m') = 76.2 dB (Annex B); a floating floor adds
    DLw = 33 dB and the flanking correction from Table 1 (separating 322 -> row
    300, mean flanking mass 145 -> col 150) is K = 2 dB, so L'n,w = 76 - 33 + 2
    = 45 dB (Formula 21). The element set and its L'n,w are the standard's own
    worked example, run through the tested prediction code.
    """
    ln_w_eq = ph.equivalent_impact_level(322.0)
    k = ph.impact_flanking_correction(322.0, 145.0)
    result = ph.predicted_impact_insulation(
        ln_w_eq=ln_w_eq, delta_l_w=33.0, k_correction=k
    )
    metadata = ReportMetadata(
        specimen="0.14 m concrete floor with floating floor (Annex E.3)",
        client="Example client",
        area=20.0,
        mass_per_area=322.0,
        receiving_volume=50.0,
        test_room="Dwelling above to dwelling below (example)",
        measurement_standard="EN/ISO 12354-2",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-12354-2",
        notes=(
            "Floating-floor covering improvement DLw = 33 dB; mean flanking "
            "mass 145 kg/m2 (Table 1 -> K = 2 dB). Simplified single-number "
            "model (Clause 4.3)."
        ),
        requirement=53.0,
    )
    return result, metadata, "iso12354_impact_prediction_example.pdf"


def _facade_prediction_example() -> tuple[object, ReportMetadata, str]:
    """Facade prediction fiche: EN 12354-3 Annex F worked example.

    An 11.3 m2 facade (receiving-room volume V = 50 m3, flat so ΔLfs = 0) of a
    masonry wall, a window, a small roof light and an acoustically treated air
    inlet (a Dn,e small element); energetically combining the elements'
    transmission factors (Formula 10) and the room geometry (Formula 13) gives
    D2m,nT,w = 33 dB (with R'tr,s,w = 31, Ctr = -3). The element set and its
    single-number ratings are the standard's own worked example, run through the
    tested prediction code (see tests/reference_data/).
    """
    elements = [
        ph.building.FacadeElement("Masonry wall", area=6.0, r=[41, 46, 52, 58, 64]),
        ph.building.FacadeElement("Glazing", area=4.5, r=[23, 22, 30, 36, 37]),
        ph.building.FacadeElement("Roof light", area=0.5, r=[24, 27, 30, 33, 30]),
        ph.building.FacadeElement("Air inlet", dn_e=[28, 23, 25, 38, 44]),
    ]
    result = ph.building.facade_sound_reduction(
        elements,
        area=11.3,
        volume=50.0,
        frequencies=[125, 250, 500, 1000, 2000],
        bands="octave",
    )
    metadata = ReportMetadata(
        specimen="Masonry wall + window + roof light + air inlet (Annex F)",
        client="Example client",
        area=11.3,
        receiving_volume=50.0,
        test_room="Road-traffic facade, flat reflecting (ΔLfs = 0 dB)",
        measurement_standard="EN/ISO 12354-3",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-12354-3",
        notes=(
            "Envelope: masonry wall Rw = 52, window 30, roof light 30, air "
            "inlet Dn,e (small element). Flat facade, ΔLfs = 0 dB (Annex C). "
            "Energy summation of transmission factors (Formula 10 / 13)."
        ),
        requirement=30.0,
    )
    return result, metadata, "iso12354_facade_prediction_example.pdf"


def _structure_borne_power_example() -> tuple[object, ReportMetadata, str]:
    """Structure-borne source power fiche: an EN 15657 reception-plate test.

    A pump fixed to the low-mobility reception plate EN 15657 clause 7.2.2
    specifies: 100 mm concrete of 2 300 kg/m3, so a mass per area
    m = 230 kg/m2, over 3,15 m x 2,23 m = 7,0 m2 (above the 5 m2 minimum, sides
    near sqrt(2):1) with a structural reverberation time Ts = 0,25 s, which
    gives the plate loss factor eta = 2,2/(f*Ts) (Formula 13) and keeps it at
    or above the required 0,08 through the 50-100 Hz bands. The spatial-average
    plate velocity level (Formula 12) per octave band (125 Hz to 4 kHz) is
    Lv = [70, 72, 68, 64, 60, 55] dB re 1e-9 m/s, and the structure-borne sound
    power injected into the plate is
    L_Ws = 10*lg(2*pi*f*eta*m*S) + Lv - 60 dB re 1 pW (Formula 14). The band
    levels are dominated by the 250 Hz band and sum to a total L_Ws near 65 dB
    re 1 pW. The level is specific to this reception plate; the conversion to
    the plate-independent source quantities (Formulae 15/17) precedes any
    EN 12354-5 use, as the basis strip states.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    lv = np.array([70.0, 72.0, 68.0, 64.0, 60.0, 55.0])
    result = ph.reception_plate_power(
        lv, freqs, mass_per_area=230.0, area=7.0, reverberation_time=0.25
    )
    metadata = ReportMetadata(
        client="Example building services contractor",
        specimen="Circulation pump (wall-mounted)",
        test_room="Reception-plate test rig (heavy concrete plate)",
        instrumentation="Piezoelectric accelerometer array (ISO 16063-21), s/n 0042",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-15657",
        notes="Structure-borne sound power injected into the reception plate "
        "(EN 15657:2018 reception-plate method, Formula 14).",
    )
    return result, metadata, "en15657_structure_borne_power_example.pdf"


def _installed_structure_borne_example() -> tuple[object, ReportMetadata, str]:
    """Installed structure-borne prediction fiche: an EN 12354-5 estimate.

    A WC flushing cistern fixed to a separating wall, predicting the normalised
    structure-borne sound pressure level L_n,s in the adjacent bedroom over the
    63 Hz to 2 kHz octaves (the standard's Annex I.3 worked example, wall source
    element). The characteristic source power L_Ws,c = [84,4, 82,5, 69,9, 67,6,
    61,6, 49,9] dB and the force-source coupling term D_C = 16,2 dB give the
    installed power L_Ws,inst = L_Ws,c - D_C (Formula 18b). Two flanking paths
    leave the wall (wall to floor, wall to wall), each with the wall
    structure-to-airborne adjustment D_sa and its flanking reduction index
    R_ij,ref over the element area S = 12,8 m2 (Formula 18a); the paths combine
    energetically (Formula 17) to a total L_n,s near 41 dB and an overall
    band-summed level near 43 dB. The sheet is a prediction from the source
    characterization and the element data, not a measurement; the declared
    limit of 45 dB is met.
    """
    bands = np.array([63, 125, 250, 500, 1000, 2000], dtype=float)
    characteristic_power = np.array([84.4, 82.5, 69.9, 67.6, 61.6, 49.9])
    coupling = 16.2  # dB, force-source limit for the wall (Formula 19c)
    dsa_wall = np.array([-13.6, -17.3, -17.4, -20.0, -26.9, -32.9])
    paths = [
        {
            "adjustment_term": dsa_wall,
            "flanking_reduction_index": np.array([43.0, 46.0, 50.2, 54.7, 64.6, 73.0]),
            "element_area": 12.8,
        },
        {
            "adjustment_term": dsa_wall,
            "flanking_reduction_index": np.array([37.0, 41.2, 35.9, 37.7, 49.0, 57.8]),
            "element_area": 12.8,
        },
    ]
    result = ph.installed_source_prediction(
        characteristic_power, coupling, paths, frequencies=bands
    )
    metadata = ReportMetadata(
        client="Example dwelling refurbishment",
        specimen="WC flushing cistern (wall-fixed)",
        test_room="Receiving room: adjacent bedroom",
        instrumentation="Predicted from EN 15657 source data and element mobilities",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-12354-5",
        requirement=45.0,
        notes="Predicted installed structure-borne sound "
        "(EN 12354-5:2009, Formulae 17/18); prediction, not a measurement.",
    )
    return result, metadata, "en12354_5_installed_structure_borne_example.pdf"


def _detailed_building() -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """The Annex L building in situ, its bands and the floating-floor gain.

    Assembled from the same fixture the tests and the conformance report read,
    so the two detailed fiches show the building those two already pin rather
    than a transcription of it.
    """
    import iso12354_building as bld

    bands = np.asarray(ref.ISO12354_ANNEX_L_BANDS, dtype=np.float64)
    situ = {key: ph.in_situ_element(el, bands) for key, el in bld.elements().items()}
    delta = ph.floating_floor_improvement(
        bands, resonance_frequency=bld.floating_floor_resonance()
    )
    return situ, bands, delta


def _detailed_airborne_example() -> tuple[object, ReportMetadata, str]:
    """Detailed airborne fiche: the ISO 12354-1:2017 Annex L worked example.

    The building the two parts of the standard share: two dwellings one above
    the other, a 220 mm concrete separating floor of 20 m2 carrying a floating
    floor, two 365 mm autoclaved aerated concrete external walls and two 200 mm
    calcium-silicate internal walls, joined by the eight Annex E junctions.
    The per-band model of Clause 4.2 runs the direct path and the twelve
    flanking paths band by band, and their energy summation rates to
    R'w = 57 dB, the Annex L result. Two of the annex's printed inputs are
    taken corrected rather than as printed, both registered in docs/ERRATA.md:
    the Formula (C.1) perimeter sums, derived from Formula (C.4) with the
    unrounded Annex E junction indices, and the external walls' internal loss
    factor, 0,012 5 from the element specification rather than the 0,013 of the
    input block.
    """
    import iso12354_building as bld

    situ, bands, delta = _detailed_building()
    result = ph.detailed_airborne_prediction(
        bands,
        direct_index=ph.direct_reduction_index(
            situ["floor"].sound_reduction_index, delta_r_source=delta
        ),
        flanking_paths=bld.airborne_paths(situ, delta),
    )
    metadata = ReportMetadata(
        specimen="220 mm concrete separating floor with floating floor (Annex L)",
        client="Example client",
        area=20.0,
        source_volume=50.0,
        receiving_volume=50.0,
        test_room="Dwelling above to dwelling below (example)",
        measurement_standard="EN/ISO 12354-1",
        test_date="2026-08-05",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-12354-1-L",
        notes=(
            "Detailed per-band model (Clause 4.2) over the Annex L building: "
            "365 mm AAC external walls, 200 mm calcium-silicate internal "
            "walls, eight Annex E junctions, thirteen paths."
        ),
        requirement=52.0,
    )
    return result, metadata, "iso12354_detailed_airborne_example.pdf"


def _detailed_impact_example() -> tuple[object, ReportMetadata, str]:
    """Detailed impact fiche: the ISO 12354-2:2017 Annex G worked example.

    The airborne example's building, in the impact direction: the same 220 mm
    concrete floor and floating floor excited by the tapping machine, the
    per-band normalized impact level of the direct path reduced by the
    floating floor's improvement, and the four flanking paths of Clause 4.2.
    The energy summation rates to L'n,w = 41 dB, the Annex G result.
    """
    import iso12354_building as bld

    situ, bands, delta = _detailed_building()
    result = ph.detailed_impact_prediction(
        bands,
        direct_level=ph.direct_impact_level(
            situ["floor"].impact_level, delta_l=delta
        ),
        flanking_paths=bld.impact_paths(situ, delta),
    )
    metadata = ReportMetadata(
        specimen="220 mm concrete floor with floating floor (Annex G)",
        client="Example client",
        area=20.0,
        mass_per_area=484.0,
        receiving_volume=50.0,
        test_room="Dwelling above to dwelling below (example)",
        measurement_standard="EN/ISO 12354-2",
        test_date="2026-08-05",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-12354-2-G",
        notes=(
            "Detailed per-band model (Clause 4.2) over the Annex L/G "
            "building: direct path plus four flanking paths, floating floor "
            "improvement from the resonance frequency of the annex."
        ),
        requirement=50.0,
    )
    return result, metadata, "iso12354_detailed_impact_example.pdf"
