#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for noise control: the treatment placed between source and receiver.

What an added element takes off the level: the insertion loss of a machine
enclosure, the transmission loss of a reactive silencer, the flow-generated
noise of a duct (VDI 2081-1) and a whole fan-to-room supply path rated against
an NC criterion.
"""

from __future__ import annotations

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata


def _enclosure_example() -> tuple[object, ReportMetadata, str]:
    """Enclosure fiche: the insertion loss of a machine enclosure (Bies 7.4.2).

    A documented clean-room case combining a supplied octave-band panel
    transmission loss R = [18, 22, 28, 33, 38, 42, 45] dB (a sheet-steel
    enclosure) with an interior of mean absorption alpha_i = 0.30, external
    surface area S_E = 24 m2 and internal surface area S_i = 30 m2. The interior
    room constant R_i = S_i alpha_i / (1 - alpha_i) = 30 x 0.3 / 0.7 = 12.86 m2,
    the build-up correction C = 10 lg(0.3 + S_E / R_i) = 10 lg(0.3 + 24/12.86)
    = 3.4 dB and the net insertion loss IL = R - C (Bies, Hansen & Howard,
    Engineering Noise Control 5th ed., Eqs. (7.103), (7.111)), giving a mean
    insertion loss of 28.9 dB over the seven octave bands. The requirement is a
    plausible minimum mean insertion loss the example clears (more is better).

    The fiche is a design prediction, so the metadata names the design case and
    its model rather than a test bench: no instrumentation or climate fields
    apply.
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000], dtype=float)
    panel_r = np.array([18, 22, 28, 33, 38, 42, 45], dtype=float)
    result = ph.enclosure_insertion_loss(
        panel_r, 24.0, 30.0, 0.30, frequencies=freqs
    )
    metadata = ReportMetadata(
        specimen="Sheet-steel close-fitting machine enclosure (design case)",
        client="Example client",
        manufacturer="Example enclosures",
        test_room="Machine hall, line 3 (design case)",
        measurement_standard="Bies & Hansen 7.4.2 prediction model",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-ENCLOSURE",
        requirement=20.0,
    )
    return result, metadata, "enclosure_insertion_loss_example.pdf"


def _silencer_example() -> tuple[object, ReportMetadata, str]:
    """Silencer fiche: the transmission loss of an expansion chamber (four-pole).

    A documented clean-room case: a simple expansion chamber of length L = 0.5 m
    and area S_exp = 0.08 m2 between pipes of area S_duct = 0.01 m2 (area ratio
    m = 8), sampled at the octave-band centres 63 Hz to 4 kHz by the plane-wave
    four-pole method (Munjal, Acoustics of Ducts and Mufflers 2nd ed., Eq.
    (3.27); Bies, Hansen & Howard, Engineering Noise Control 5th ed., Eq.
    (8.111)). The transmission loss matches the closed form
    TL = 10 lg[1 + (1/4)(m - 1/m)^2 sin^2(kL)], peaking near
    10 lg[1 + (1/4)(8 - 1/8)^2] = 12.2 dB, with a mean of 8.9 dB over the seven
    bands. The requirement is a plausible minimum mean transmission loss the
    example clears (more is better).

    The fiche is a design prediction, so the metadata names the design case and
    its model rather than a test bench: no instrumentation or climate fields
    apply.
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000], dtype=float)
    result = ph.noise_control.silencers.expansion_chamber(
        freqs, 0.5, 0.08, 0.01
    )
    metadata = ReportMetadata(
        specimen="Simple expansion-chamber muffler (m = 8, design case)",
        client="Example client",
        manufacturer="Example silencers",
        test_room="Duct system design study",
        measurement_standard="Munjal Eq. (3.27) four-pole model",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-SILENCER",
        requirement=6.0,
    )
    return result, metadata, "reactive_silencer_example.pdf"


def _hvac_example() -> tuple[object, ReportMetadata, str]:
    """HVAC fiche: the flow-generated noise of a straight duct (VDI 2081-1).

    A documented clean-room case: the flow-generated octave-band sound power
    level of a straight duct carrying air at U = 12 m/s in a cross-section of
    S = 0.04 m2, L_WB = 7 + 50 lg U + 10 lg S - 2 - 26 lg(1.14 + 0.02 f / U)
    dB re 1 pW (VDI 2081-1; Bies, Hansen & Howard, Engineering Noise Control
    5th ed., Eq. (8.251)). Combining the seven octave bands with the ISO 3744
    Annex E A-weighting corrections gives the A-weighted sound power level
    L_WA = 38.8 dB(A) re 1 pW (overall unweighted L_W = 47.0 dB). The
    requirement is a plausible maximum A-weighted level the example clears
    (lower is better).

    The fiche is a design prediction, so the metadata names the design case and
    its model rather than a test bench: no instrumentation or climate fields
    apply.
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000], dtype=float)
    result = ph.noise_control.hvac.flow_noise_straight_duct(freqs, 12.0, 0.04)
    metadata = ReportMetadata(
        specimen="Straight supply duct, 0.04 m2 cross-section (design case)",
        client="Example client",
        test_room="Air-handling plant room (design case)",
        measurement_standard="VDI 2081-1 prediction model",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-HVAC",
        requirement=45.0,
    )
    return result, metadata, "hvac_duct_noise_example.pdf"


def _duct_path_example() -> tuple[object, ReportMetadata, str]:
    """Duct-path fiche: a fan-to-room supply path checked against NC 30.

    The supply half of the classic duct-borne worked sheet (Long,
    Architectural Acoustics 2nd ed., Table 14.9): a forward-curved centrifugal
    fan at 5000 cfm and 2 in w.g. feeding a 20 x 20 x 8 ft carpeted office
    through an unlined elbow, a 3 ft standard-pressure-drop silencer, two lined
    rectangular runs either side of a 25 per cent branch split, a flexible
    final run and a rectangular diffuser, with the room effect converting the
    sound power reaching the diffuser into the level at the listener.

    The element attenuations and the silencer and diffuser self-noise spectra
    are the ones printed in that sheet (silencers and air terminal devices are
    always manufacturer data in a real calculation), so the fiche exercises the
    cascade, the criterion comparison and the sheet layout on published
    numbers. The design criterion is NC 30, which the received spectrum meets.
    """
    from phonometry.noise_control.duct_path import DuctElement, duct_path
    from phonometry.noise_control.hvac import OCTAVE_BANDS

    result = duct_path(
        OCTAVE_BANDS,
        [90.0, 86.0, 82.0, 79.0, 77.0, 75.0, 71.0, 61.0],
        [
            DuctElement(
                "Elbow, 36 x 24 in, unlined",
                [0, 1, 2, 3, 3, 3, 3, 3], [41, 39, 36, 29, 20, 6, 0, 0],
                code="2",
            ),
            DuctElement(
                "Silencer, standard pressure drop, 3 ft",
                [7, 12, 16, 28, 35, 35, 28, 17], [49, 43, 44, 42, 42, 45, 35, 24],
                code="3",
            ),
            DuctElement(
                "Duct, 36 x 24 in, 5 ft, 1 in lining",
                [2, 2, 3, 7, 15, 12, 11, 9], code="4",
            ),
            DuctElement("Branch split, 25 per cent", 6.0, code="5"),
            DuctElement(
                "Duct, 18 x 12 in, 6 ft, 1 in lining",
                [3, 3, 5, 11, 25, 22, 16, 13], code="6",
            ),
            DuctElement(
                "Flexible duct, 12 in diameter, 6 ft",
                [14, 14, 16, 15, 17, 22, 16, 13], code="7",
            ),
            DuctElement(
                "Rectangular diffuser, 312 cfm",
                0.0, [33, 32, 29, 23, 15, 4, 0, 0], code="8",
            ),
        ],
        room_effect=[6, 6, 5, 5, 6, 7, 6, 6],
        source_label="Fan, centrifugal, forward-curved, 5000 cfm, 2 in w.g.",
        criterion="NC",
        target=30.0,
        label="Supply path",
    )
    metadata = ReportMetadata(
        specimen="Supply air path, roof-mounted built-up air handler (design case)",
        client="Example client",
        test_room="Open-plan office, 6.1 x 6.1 x 2.4 m, carpeted (design case)",
        measurement_standard="AHRI Standard 885 procedure; ANSI/ASA S12.2-2019 criterion",
        test_date="2026-07-29",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-DUCT-PATH",
    )
    return result, metadata, "duct_path_example.pdf"
