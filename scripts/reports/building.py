#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for measured sound insulation: what a finished element achieves.

Airborne and impact insulation as it is verified and rated, never predicted:
the laboratory quantities of ISO 10140 and ISO 15186, the field quantities of
ISO 16283, the survey method of ISO 10052, the laboratory flanking descriptors
of ISO 10848, the floor-covering improvement of ISO 16251-1, and the
single-number ratings of ISO 717 that collapse every one of them to the figure
a regulation quotes.
"""

from __future__ import annotations

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata

_RATING_FREQS = np.array(
    [
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


def _airborne_example() -> tuple[object, ReportMetadata, str]:
    """Airborne fiche: a predicted single-panel sound reduction index."""
    result = ph.building.single_panel_transmission_loss(
        _RATING_FREQS, 15.0, critical_frequency=2000.0, loss_factor=0.02
    )
    metadata = ReportMetadata(
        specimen="6 mm float glass pane",
        client="Example client",
        manufacturer="Example glassworks",
        area=1.23,
        mass_per_area=15.0,
        source_volume=53.0,
        receiving_volume=51.0,
        source_temperature=21.6,
        source_relative_humidity=35.3,
        receiving_temperature=20.9,
        receiving_relative_humidity=37.4,
        pressure=101.9,
        test_room="Transmission suite (example)",
        mounting="Elastic perimeter, single glazing",
        measurement_standard="ISO 10140-2",
        test_date="2026-07-18",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-717-1",
        requirement=30.0,
    )
    return result, metadata, "iso717_airborne_example.pdf"


def _impact_example() -> tuple[object, ReportMetadata, str]:
    """Impact fiche: a normalized impact sound pressure level rating."""
    ln = np.array(
        [45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 55, 52, 49, 46],
        dtype=float,
    )
    result = ph.building.weighted_impact_rating(ln)
    metadata = ReportMetadata(
        specimen="150 mm concrete slab with a floating floor",
        client="Example client",
        manufacturer="Example floors",
        area=16.0,
        mass_per_area=360.0,
        source_volume=53.0,
        receiving_volume=51.0,
        source_temperature=20.8,
        source_relative_humidity=47.0,
        receiving_temperature=20.5,
        receiving_relative_humidity=48.0,
        pressure=100.9,
        test_room="Transmission suite (example)",
        mounting="Floating floor on a resilient layer",
        measurement_standard="ISO 16283-2",
        test_date="2026-07-18",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-717-2",
        requirement=53.0,
    )
    return result, metadata, "iso717_impact_example.pdf"


def _field_airborne_example() -> tuple[object, ReportMetadata, str]:
    """Field airborne fiche: a DnT measurement between dwellings (ISO 16283-1)."""
    l1 = np.array(
        [
            92.3,
            93.1,
            94.0,
            94.4,
            94.8,
            95.0,
            95.2,
            95.4,
            95.3,
            95.1,
            94.8,
            94.4,
            93.9,
            93.3,
            92.5,
            91.6,
        ]
    )
    d = np.array(
        [
            38.2,
            40.1,
            42.6,
            45.2,
            47.8,
            50.1,
            52.3,
            54.0,
            55.6,
            57.1,
            58.2,
            59.0,
            59.6,
            60.1,
            60.3,
            59.8,
        ]
    )
    t2 = np.array(
        [
            0.62,
            0.58,
            0.55,
            0.53,
            0.52,
            0.50,
            0.49,
            0.48,
            0.47,
            0.46,
            0.45,
            0.45,
            0.44,
            0.43,
            0.43,
            0.42,
        ]
    )
    result = ph.building.airborne_insulation(
        l1, l1 - d, t2, area=12.5, volume=30.4
    )
    metadata = ReportMetadata(
        specimen="Separating wall, 240 mm brick with independent lining",
        client="Example client",
        area=12.5,
        source_volume=32.1,
        receiving_volume=30.4,
        temperature=20.4,
        relative_humidity=52.0,
        test_room="Dwelling A living room to dwelling B living room",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-16283-1",
        requirement=50.0,
    )
    return result, metadata, "iso16283_airborne_example.pdf"


def _field_impact_example() -> tuple[object, ReportMetadata, str]:
    """Field impact fiche: a tapping-machine L'nT measurement (ISO 16283-2)."""
    li = np.array(
        [
            58.0,
            60.5,
            62.0,
            63.5,
            65.0,
            66.0,
            66.5,
            66.0,
            65.5,
            65.0,
            64.0,
            62.0,
            59.0,
            56.0,
            53.0,
            50.0,
        ]
    )
    t2 = np.array(
        [
            0.60,
            0.57,
            0.55,
            0.53,
            0.52,
            0.50,
            0.49,
            0.48,
            0.47,
            0.46,
            0.45,
            0.45,
            0.44,
            0.43,
            0.43,
            0.42,
        ]
    )
    result = ph.building.impact_insulation(li, t2, volume=30.4)
    metadata = ReportMetadata(
        specimen="Timber-joist floor with a floating chipboard deck",
        client="Example client",
        receiving_volume=30.4,
        temperature=20.1,
        relative_humidity=54.0,
        test_room="Dwelling A bedroom below dwelling B bedroom",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-16283-2",
        requirement=58.0,
    )
    return result, metadata, "iso16283_impact_example.pdf"


def _lab_airborne_example() -> tuple[object, ReportMetadata, str]:
    """Laboratory airborne fiche: a sound reduction index R (ISO 10140-2).

    The reported spectrum is the ISO 717-1:2020 Annex C worked-example sound
    reduction index (Rw = 30 (-2; -3) dB): with the free test opening area
    S = 10 m2 equal to the receiving-room absorption area A = 0,16 V / T (here
    V = 50 m3, T = 0,8 s give A = 10 m2), the term 10 lg(S/A) vanishes and
    R = L1 - L2 reproduces that published curve exactly.
    """
    r = np.array(
        [
            20.4,
            16.3,
            17.7,
            22.6,
            22.4,
            22.7,
            24.8,
            26.6,
            28.0,
            30.5,
            31.8,
            32.5,
            33.4,
            33.0,
            31.0,
            25.5,
        ]
    )
    l1 = np.full(16, 90.0)
    result = ph.building.lab_airborne_insulation(
        l1, l1 - r, np.full(16, 0.8), area=10.0, volume=50.0
    )
    metadata = ReportMetadata(
        specimen="100 mm autoclaved aerated concrete block wall",
        client="Example client",
        manufacturer="Example blockworks",
        area=10.0,
        mass_per_area=75.0,
        receiving_volume=50.0,
        source_volume=53.0,
        receiving_temperature=20.8,
        receiving_relative_humidity=46.0,
        pressure=101.3,
        test_room="Transmission suite (example)",
        mounting="Type A mounting, mortar-bedded perimeter (ISO 10140-1)",
        measurement_standard="ISO 10140-2",
        test_date="2026-07-18",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10140-2",
        requirement=30.0,
    )
    return result, metadata, "iso10140_airborne_example.pdf"


def _lab_impact_example() -> tuple[object, ReportMetadata, str]:
    """Laboratory impact fiche: a normalized impact level Ln (ISO 10140-3).

    The reported spectrum is the ISO 717-2:2020 Annex C worked-example
    normalized impact sound pressure level (Ln,w = 79 (-11) dB): with the
    receiving-room absorption area A = 0,16 V / T equal to the reference
    A0 = 10 m2 (here V = 50 m3, T = 0,8 s give A = 10 m2), the term
    10 lg(A/A0) vanishes and Ln = Li reproduces that published curve exactly.
    """
    li = np.array(
        [
            62.1,
            63.2,
            63.5,
            66.2,
            68.5,
            70.0,
            71.7,
            73.1,
            73.8,
            73.5,
            73.8,
            73.3,
            73.1,
            73.0,
            72.4,
            71.2,
        ]
    )
    result = ph.building.lab_impact_insulation(
        li, np.full(16, 0.8), volume=50.0
    )
    # The impact fiche's plot legend carries an extra "500 Hz read" entry that
    # wraps to a second row, making the embedded figure taller than the
    # airborne one; the header is kept to the essential accredited fields so
    # the sheet stays comfortably within one page across renderers.
    metadata = ReportMetadata(
        specimen="140 mm concrete slab, bare (reference floor)",
        client="Example client",
        area=10.0,
        mass_per_area=336.0,
        receiving_volume=50.0,
        receiving_temperature=20.6,
        receiving_relative_humidity=45.0,
        test_room="Transmission suite (example)",
        mounting="Bare slab, no floor covering (ISO 10140-1)",
        measurement_standard="ISO 10140-3",
        test_date="2026-07-18",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10140-3",
        requirement=80.0,
    )
    return result, metadata, "iso10140_impact_example.pdf"


def _intensity_example() -> tuple[object, ReportMetadata, str]:
    """Intensity fiche: an intensity sound reduction index RI (ISO 15186-1).

    The reported spectrum is the ISO 717-1:2020 Annex C worked-example sound
    reduction index (Rw = 30 (-2; -3) dB), reused as a documented intensity
    sound reduction index RI(f): RI is an ordinary sound reduction index rated
    by the same ISO 717-1 machinery, so feeding the receiving-side intensity
    levels LIn that make Formula (7) return that curve (with Lp1 = 85 dB, a
    measurement surface Sm = 12 m2 and a specimen S = 10 m2) pins the fiche to
    the published RI,w = 30 (-2; -3) dB. The Annex B adaptation term
    Kc = 10 lg(1 + 61,4/f) (Formula (B.2)) is annexed so the verbose table
    shows the Kc-modified index RI,M beside RI.
    """
    ri = np.array(
        [
            20.4,
            16.3,
            17.7,
            22.6,
            22.4,
            22.7,
            24.8,
            26.6,
            28.0,
            30.5,
            31.8,
            32.5,
            33.4,
            33.0,
            31.0,
            25.5,
        ]
    )
    lp1, sm, s = 85.0, 12.0, 10.0
    l_in = lp1 - 6.0 - 10.0 * np.log10(sm / s) - ri
    kc = ph.building.adaptation_term_kc(_RATING_FREQS)
    result = ph.building.intensity_sound_reduction(
        np.full(16, lp1), l_in, measurement_area=sm, area=s, kc=kc
    )
    metadata = ReportMetadata(
        specimen="100 mm autoclaved aerated concrete block wall",
        client="Example client",
        manufacturer="Example blockworks",
        area=10.0,
        mass_per_area=75.0,
        receiving_volume=50.0,
        source_volume=53.0,
        receiving_temperature=20.8,
        receiving_relative_humidity=46.0,
        pressure=101.3,
        test_room="Transmission suite (example)",
        mounting="Type A mounting, mortar-bedded perimeter (ISO 10140-1)",
        measurement_standard="ISO 15186-1",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-15186-1",
        requirement=30.0,
    )
    return result, metadata, "iso15186_intensity_example.pdf"


def _intensity_element_example() -> tuple[object, ReportMetadata, str]:
    """Element fiche: an intensity element-normalized level difference DI,n,e.

    The reported spectrum reuses the ISO 717-1:2020 Annex C worked-example
    curve (Rw = 30 (-2; -3) dB), read here as a documented element-normalized
    level difference DI,n,e(f): DI,n,e is a level difference rated by the same
    ISO 717-1 machinery, so feeding the receiving-side intensity levels LIn
    that make Formula (8) return that curve (with Lp1 = 85 dB, a measurement
    surface Sm = 12 m2 and a single element unit N = 1, referred to the
    reference absorption area A0 = 10 m2) pins the fiche to the published
    DI,n,e,w = 30 (-2; -3) dB.
    """
    d_i_n_e = np.array(
        [
            20.4,
            16.3,
            17.7,
            22.6,
            22.4,
            22.7,
            24.8,
            26.6,
            28.0,
            30.5,
            31.8,
            32.5,
            33.4,
            33.0,
            31.0,
            25.5,
        ]
    )
    lp1, sm, n = 85.0, 12.0, 1
    l_in = lp1 - 6.0 - 10.0 * np.log10(sm / 10.0) - 10.0 * np.log10(n) - d_i_n_e
    result = ph.building.intensity_element_normalized_difference(
        np.full(16, lp1), l_in, measurement_area=sm, n=n
    )
    metadata = ReportMetadata(
        specimen="Trickle ventilator in a 100 mm masonry wall",
        client="Example client",
        manufacturer="Example ventilators",
        area=0.02,
        test_room="Transmission suite (example)",
        mounting="Small-element mounting per ISO 10140-1 Annex F",
        measurement_standard="ISO 15186-1",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-15186-1-DINE",
        notes="Measurement surface Sm = 12 m2, N = 1 element unit.",
        requirement=30.0,
    )
    return result, metadata, "iso15186_element_example.pdf"


def _floor_covering_example() -> tuple[object, ReportMetadata, str]:
    """Floor-covering fiche: an ISO 16251-1 impact-improvement measurement.

    ISO 16251-1:2014 carries no filled numeric worked example (its Annex B is a
    blank report form), so the committed spectrum is a real measurement: the
    improvement of a textile carpet (laid loose on the mock-up plate) digitized
    from Figure 4 of R. Foret, J.-B. Chene and C. Guigou-Carter, "A comparison
    of the reduction of transmitted impact noise by floor coverings measured
    using ISO 140-8 and ISO/CD 16251-1", Forum Acusticum 2011, Aalborg (CSTB).
    The reduction of impact sound pressure level rises with frequency,
     delta-L = [5, 8, 10, 14, 18, 23, 30, 31, 39, 49, 53, 57, 60, 67, 68, 71] dB
    over the 16 one-third-octave bands 100 Hz to 3150 Hz (values read to
    +/- 0,5 dB from the figure's vector chart).

    The weighted improvement follows ISO 717-2:2020 Clause 5: applied to the
    heavyweight reference floor L_n,r,0 (Table 4, rated L_n,r,0,w = 78 dB and
    CI,r,0 = -11 dB), L_n,r = L_n,r,0 - delta-L rates to L_n,r,w = 49 dB, so
    delta-Lw = 78 - 49 = 29 dB (Formula (2)), reproducing the paper's published
    ISO 16251-1 value exactly; the spectrum adaptation term is CI,delta = -13 dB
    (Formula (A.4)). Both are reproduced by ``weighted_impact_improvement`` and
    ``impact_improvement_adaptation_term``. The requirement is a plausible
    minimum weighted improvement the example clears (a higher value is better).
    """
    bare = np.full(16, 78.0)  # bare-plate acceleration level (arbitrary datum)
    delta_l = np.array(
        [5, 8, 10, 14, 18, 23, 30, 31, 39, 49, 53, 57, 60, 67, 68, 71],
        dtype=float,
    )
    result = ph.building.impact_improvement(bare, bare - delta_l, _RATING_FREQS)
    metadata = ReportMetadata(
        specimen="Textile floor covering (carpet), laid loose",
        client="Example client",
        manufacturer="Example floors",
        mass_per_area=2.4,
        mounting="Laid loose on the mock-up plate (ISO 10140-1 category I)",
        test_room="Small-mock-up impact rig (example)",
        measurement_standard="ISO 16251-1",
        temperature=21.0,
        pressure=101.2,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-16251",
        requirement=20.0,
        notes=(
            "Illustrative example. The improvement spectrum is digitized from "
            "Foret, Chene and Guigou-Carter (Forum Acusticum 2011, ISO/CD "
            "16251-1 draft), not an accredited measurement."
        ),
    )
    return result, metadata, "iso16251_floor_covering_example.pdf"


#: One-third-octave centre frequencies of the ISO 10848 mandatory range,
#: 100 Hz to 5000 Hz (18 bands, Part 1 Clause 7.5), in Hz.
_FLANKING_FREQS = np.array(
    [
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


def _vibration_reduction_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10848 fiche: the vibration reduction index Kij of a rigid junction.

    ISO 10848 carries no worked numeric example, so this is an illustrative
    clean-room case built from Formula (13). A rigid T-junction of two heavy
    walls (element areas Si = 12 m2, Sj = 10 m2, common edge lij = 4 m) has a
    direction-averaged velocity level difference Dv,ij rising from 4.5 dB at
    100 Hz to 12.7 dB at 5 kHz; with the structural reverberation times
    Ts,i = 0.35 s and Ts,j = 0.40 s the equivalent absorption lengths follow
    Formula (12), giving Kij from about 3 dB at low frequency to 20 dB at
    5 kHz and a single-number mean Kij = 9.5 dB over the Annex A 200-1250 Hz
    range. The modal overlap factor brackets the three lowest bands
    (M < 0.25, ISO 10848-4:2010 Clause 9), which are excluded from the mean.
    """
    dv = np.array(
        [
            4.5,
            4.8,
            5.2,
            5.6,
            6.0,
            6.5,
            7.0,
            7.6,
            8.1,
            8.7,
            9.2,
            9.8,
            10.3,
            10.9,
            11.4,
            11.9,
            12.3,
            12.7,
        ]
    )
    modal_overlap = np.full(_FLANKING_FREQS.size, 1.0)
    modal_overlap[:3] = 0.1  # bracket the three lowest bands (poor overlap)
    result = ph.building.vibration_reduction_index(
        dv,
        junction_length=4.0,
        area_i=12.0,
        area_j=10.0,
        frequency=_FLANKING_FREQS,
        structural_reverberation_time_i=0.35,
        structural_reverberation_time_j=0.40,
        modal_overlap=modal_overlap,
    )
    metadata = ReportMetadata(
        specimen="Rigid T-junction of two 200 mm concrete walls",
        client="Example client",
        test_room="Flanking-transmission suite (example)",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10848-KIJ",
        notes="Junction vibration reduction index Kij (ISO 10848-1:2006).",
    )
    return result, metadata, "iso10848_kij_example.pdf"


def _flanking_level_difference_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10848 fiche: the normalized flanking level difference Dn,f (airborne).

    An illustrative clean-room case (ISO 10848 has no worked numeric example):
    with the source-room level L1 = 80 dB and the receiving-room equivalent
    absorption area equal to the reference A0 = 10 m2 in every band (so the
    10 lg(A/A0) term vanishes, Formula (4)), a receiving-room level rising from
    32 dB at 100 Hz gives a Dn,f rising from 48 dB to 65 dB and, per ISO 717-1,
    the single number Dn,f,w = 60 (-1; -3) dB.
    """
    dn_f = np.array(
        [48, 49, 50, 51, 52, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65],
        dtype=float,
    )
    source_level = np.full(16, 80.0)
    result = ph.building.normalized_flanking_level_difference(
        source_level, source_level - dn_f, absorption_area=np.full(16, 10.0)
    )
    metadata = ReportMetadata(
        specimen="Flanking wall over a rigid T-junction",
        client="Example client",
        test_room="Flanking-transmission suite (example)",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10848-DNF",
        requirement=55.0,  # Dn,f,w = 60 dB >= 55 dB -> PASS
        notes="Normalized flanking level difference Dn,f (ISO 10848-2:2006).",
    )
    return result, metadata, "iso10848_dnf_example.pdf"


def _flanking_impact_level_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10848 fiche: the normalized flanking impact level Ln,f (tapping machine).

    An illustrative clean-room case (ISO 10848 has no worked numeric example):
    with the receiving-room equivalent absorption area equal to the reference
    A0 = 10 m2 in every band (so the 10 lg(A/A0) term vanishes, Formula (5)),
    a receiving-room impact level falling from 58 dB at 100 Hz to 32 dB at
    3150 Hz gives an Ln,f equal to it and, per ISO 717-2, the single number
    Ln,f,w = 49 (0) dB.
    """
    receive_level = np.array(
        [58, 57, 56, 55, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32],
        dtype=float,
    )
    result = ph.building.normalized_flanking_impact_level(
        receive_level, absorption_area=np.full(16, 10.0)
    )
    metadata = ReportMetadata(
        specimen="Flanking floor over a rigid T-junction",
        client="Example client",
        test_room="Flanking-transmission suite (example)",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10848-LNF",
        requirement=55.0,  # Ln,f,w = 49 dB <= 55 dB -> PASS
        notes="Normalized flanking impact level Ln,f (ISO 10848-2:2006).",
    )
    return result, metadata, "iso10848_lnf_example.pdf"


def _survey_airborne_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10052 survey fiche: airborne DnT between dwellings (octave bands).

    The survey (control) method works in the five octave bands 125 Hz to
    2000 Hz. With the source-room level at 80 dB, a level difference D rising
    from 33 dB to 48 dB and the reverberation index estimated for a furnished
    receiving room of about 50 m3 (ISO 10052:2021 Table 4), DnT = D + k gives
    the standardized level difference and, per ISO 717-1, DnT,w = 44 (-1; -4) dB.
    """
    l1 = np.full(5, 80.0)
    d = np.array([33.0, 36.0, 40.0, 44.0, 48.0])
    k = ph.building.estimate_reverberation_index(50.0, "furnished")
    result = ph.building.survey_airborne_insulation(
        l1, l1 - d, k, volume=50.0, area=12.0
    )
    metadata = ReportMetadata(
        specimen="Separating wall between dwellings (survey method)",
        client="Example client",
        area=12.0,
        receiving_volume=50.0,
        test_room="Dwelling A living room to dwelling B living room",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10052-AIRBORNE",
        requirement=40.0,  # DnT,w >= 40 dB -> PASS
        notes="Survey-method airborne sound insulation DnT (ISO 10052:2021).",
    )
    return result, metadata, "iso10052_airborne_example.pdf"


def _survey_impact_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10052 survey fiche: impact L'nT of a floor (octave bands).

    With the energy-average tapping-machine level Li falling across the five
    octave bands and the reverberation index estimated for a furnished
    receiving room of about 50 m3, L'nT = Li - k gives the standardized impact
    level and, per ISO 717-2, its single number L'nT,w (CI). A lower impact
    level is better, so the verdict passes at or below the requirement.
    """
    li = np.array([62.0, 64.0, 63.0, 60.0, 55.0])
    k = ph.building.estimate_reverberation_index(50.0, "furnished")
    result = ph.building.survey_impact_insulation(li, k, volume=50.0)
    metadata = ReportMetadata(
        specimen="Separating floor between dwellings (survey method)",
        client="Example client",
        receiving_volume=50.0,
        test_room="Dwelling A bedroom below dwelling B bedroom",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10052-IMPACT",
        requirement=62.0,  # L'nT,w <= 62 dB -> PASS (lower is better)
        notes="Survey-method impact sound insulation L'nT (ISO 10052:2021).",
    )
    return result, metadata, "iso10052_impact_example.pdf"


def _survey_facade_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10052 survey fiche: facade D2m,nT (octave bands).

    From the outdoor level 2 m in front of the facade and the receiving-room
    level, the facade level difference D2m rises across the five octave bands;
    with the reverberation index estimated for a furnished receiving room of
    about 40 m3, D2m,nT = D2m + k gives the standardized facade level
    difference and, per ISO 717-1, its single number D2m,nT,w (C; Ctr).
    """
    l1_2m = np.full(5, 75.0)
    d2m = np.array([31.0, 34.0, 37.0, 40.0, 43.0])
    k = ph.building.estimate_reverberation_index(40.0, "furnished")
    result = ph.building.survey_facade_insulation(
        l1_2m, l1_2m - d2m, k, volume=40.0
    )
    metadata = ReportMetadata(
        specimen="Dwelling facade with a double-glazed window (survey method)",
        client="Example client",
        receiving_volume=40.0,
        test_room="Dwelling bedroom facing a residential street",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10052-FACADE",
        requirement=33.0,  # D2m,nT,w >= 33 dB -> PASS
        notes="Survey-method facade sound insulation D2m,nT (ISO 10052:2021).",
    )
    return result, metadata, "iso10052_facade_example.pdf"


def _field_facade_example() -> tuple[object, ReportMetadata, str]:
    """ISO 16283-3 fiche: field facade D2m,nT (one-third-octave bands).

    The reported spectrum is the ISO 717-1:2020 Annex C worked-example curve
    (rated 30 (-2; -3) dB): with the outdoor level 2 m in front of the facade
    set to that curve plus 40 dB, a receiving-room level of 40 dB and the
    reverberation time equal to T0 = 0,5 s in every band (so the
    standardization term vanishes), D2m,nT reproduces that published curve and
    the fiche boxes D2m,nT,w = 30 (-2; -3) dB.
    """
    annex_c = np.array(
        [20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
         28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5]
    )
    core_freqs = np.array(
        [100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
         630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0]
    )
    result = ph.building.facade_insulation(
        annex_c + 40.0, np.full(16, 40.0), np.full(16, 0.5),
        volume=62.5, frequencies=core_freqs,
    )
    metadata = ReportMetadata(
        specimen="Dwelling facade, loudspeaker method",
        client="Example client",
        receiving_volume=62.5,
        temperature=19.8,
        relative_humidity=55.0,
        test_room="Dwelling living room facing a main road",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-16283-3",
        requirement=30.0,  # D2m,nT,w >= 30 dB -> PASS
        notes="Field facade sound insulation D2m,nT (ISO 16283-3:2016).",
    )
    return result, metadata, "iso16283_facade_example.pdf"
