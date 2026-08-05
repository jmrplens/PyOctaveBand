#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Generate the committed example ``.report()`` fiches.

Mirrors :mod:`scripts.generate_graphs` for the normative report PDFs: it builds
a small set of representative results with example metadata and writes their
``.report()`` fiche to ``.github/reports/`` so the repository always carries an
up-to-date rendered example of every report kind, which the documentation links
to. Alongside every ``<name>.pdf`` it rasterizes the first page to a
``<name>.webp`` preview (via :mod:`pypdfium2`, no system Poppler needed) so the
website and the GitHub docs can show the fiche inline without a PDF viewer or a
build-time rasterizer. The preview is lossless WebP: pixel-identical to the
raster and roughly half the size of the optimized PNG it replaces (flat-color
document pages compress better lossless than lossy). Run it with
``make reports`` or ``python scripts/generate_reports.py``.

Neither the PDFs nor the WebP previews are byte-compared in CI: the embedded
plot is vector geometry whose floating-point coordinates differ by ~1 ULP
across CPUs, and the raster inherits that. The ``Example report fiches up to
date`` job regenerates the set and compares it within a tolerance instead
(:mod:`scripts.check_reports`); the fiches went unchecked for months before
that job existed, and two of them fell a plot-styling release behind
unnoticed.
"""

from __future__ import annotations

import os

# Deterministic fiche output: pin every numerical thread pool to a single
# thread BEFORE numpy/scipy import their backends, so multi-threaded reductions
# cannot reorder floating-point sums and perturb the rendered plot across
# machines. The figure generator does the same, and for the same reason: the
# CI runner has a different core count than a dev box.
for _threads_var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata

# The ISO 12354 Annex L / Annex G building is assembled once, in tests/, and
# read from there by the tests and by scripts/conformance_report.py. The two
# detailed-prediction fiches below show that same building, so they read it
# from the same place rather than becoming a third transcription of a worked
# example whose inputs the registry already records two corrections to.
_TESTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tests"))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import reference_data as ref  # noqa: E402  (needs the path above)

#: Committed output directory for the example fiches.
_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", ".github", "reports")
)

#: Pixel width of the rendered WebP preview. A4 portrait at this width is ~1415
#: px tall: crisp on the docs pages (shown at ~80 % column width) yet small
#: enough to commit. The height follows from the page aspect ratio.
_PREVIEW_WIDTH_PX = 1000

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
    result = ph.single_panel_transmission_loss(
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
    result = ph.weighted_impact_rating(ln)
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
    result = ph.airborne_insulation(l1, l1 - d, t2, area=12.5, volume=30.4)
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
    result = ph.impact_insulation(li, t2, volume=30.4)
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
    result = ph.lab_airborne_insulation(
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
    result = ph.lab_impact_insulation(li, np.full(16, 0.8), volume=50.0)
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


def _airborne_prediction_example() -> tuple[object, ReportMetadata, str]:
    """Airborne prediction fiche: EN 12354-1 Annex H.3 worked example.

    A separating wall Rs,w = 57 dB, area Ss = 11.5 m2, flanked by four elements
    (floor, ceiling, facade, internal wall) whose Annex H tabulated junction
    Kij feed twelve flanking paths; the direct Dd path plus those twelve make
    thirteen paths whose energy summation (Formula 26) gives R'w = 52.2 -> 52
    dB. The element set and its R'w are the standard's own worked example, run
    through the tested prediction code (see tests/reference_data.py).
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
    tested prediction code (see tests/reference_data.py).
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


def _absorption_example() -> tuple[object, ReportMetadata, str]:
    """Absorption fiche: an ISO 11654 weighted sound absorption rating."""
    # The fifteen one-third-octave alpha_s (200 Hz to 5000 Hz) an accredited
    # ISO 354 certificate reports, whose octave means are the practical
    # coefficients (0.35, 1.00, 0.65, 0.60, 0.55) giving alpha_w = 0.60 with an
    # M shape indicator (ISO 11654 Annex A.2 shape); a broadband porous absorber.
    alpha_s = (
        0.30,
        0.35,
        0.40,  # 250 Hz octave -> alpha_p 0.35
        1.00,
        1.00,
        1.00,  # 500 Hz octave -> alpha_p 1.00
        0.62,
        0.66,
        0.67,  # 1000 Hz octave -> alpha_p 0.65
        0.58,
        0.60,
        0.62,  # 2000 Hz octave -> alpha_p 0.60
        0.53,
        0.55,
        0.57,  # 4000 Hz octave -> alpha_p 0.55
    )
    result = ph.materials.weighted_absorption_from_third_octave(alpha_s)
    metadata = ReportMetadata(
        specimen="50 mm porous absorber over a 100 mm air gap",
        client="Example client",
        manufacturer="Example acoustics",
        area=10.8,
        mounting="Type A (mounted directly against a rigid wall)",
        test_room="Reverberation room (example)",
        measurement_standard="ISO 354",
        test_date="2026-07-20",
        temperature=21.4,
        relative_humidity=54.0,
        pressure=101.0,
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-11654",
        requirement=0.55,
    )
    return result, metadata, "iso11654_absorption_example.pdf"


def _sound_absorption_example() -> tuple[object, ReportMetadata, str]:
    """ISO 354 fiche: a reverberation-room sound-absorption measurement.

    A documented clean-room example, derived in closed form from ISO 354:2003
    Eq. (5)/(7)/(8)/(9) with no air-attenuation correction (m = 0, the
    zero-attenuation reference condition). Room volume V = 200 m3 (the ISO 354
    reference volume) and specimen area S = 10.8 m2 (clause 6.2.1.1 range); at
    20 degC the speed of sound is c = 331 + 0.6*20 = 343 m/s (Eq. (6)), so the
    Sabine constant of the inversion is 55.3*V/c = 55.3*200/343 = 32.24490 m2 s.
    With the empty-room T1 and with-specimen T2 tables below, the equivalent
    sound absorption areas A = 55.3*V/(c*T) and the coefficient
    alpha_s = (A2 - A1)/S follow. Two worked bands:

    * 500 Hz: A1 = 32.24490/7.80 = 4.13396 m2, A2 = 32.24490/4.20 = 7.67736 m2,
      alpha_s = (7.67736 - 4.13396)/10.8 = 0.328 -> 0.33.
    * 1000 Hz: A1 = 32.24490/6.90 = 4.67317 m2, A2 = 32.24490/2.85 = 11.31400 m2,
      alpha_s = (11.31400 - 4.67317)/10.8 = 0.615 -> 0.61.

    The resulting alpha_s rises from 0.02 at 100 Hz to a 0.69 plateau near
    1600 Hz and falls back to 0.34 at 5000 Hz, a broadband porous absorber.
    """
    freqs = np.array(
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
    t_empty = np.array(
        [
            9.0,
            9.0,
            8.8,
            8.6,
            8.4,
            8.2,
            8.0,
            7.8,
            7.5,
            7.2,
            6.9,
            6.6,
            6.2,
            5.8,
            5.4,
            5.0,
            4.6,
            4.2,
        ]
    )
    t_specimen = np.array(
        [
            8.4,
            8.2,
            7.7,
            7.2,
            6.5,
            5.7,
            4.9,
            4.2,
            3.6,
            3.15,
            2.85,
            2.65,
            2.55,
            2.5,
            2.55,
            2.6,
            2.7,
            2.85,
        ]
    )
    result = ph.measure_sound_absorption(
        freqs,
        t_empty,
        t_specimen,
        volume=200.0,
        area=10.8,
        temperature=20.0,
        humidity=54.0,
    )
    metadata = ReportMetadata(
        specimen="50 mm porous absorber over a 100 mm air gap",
        client="Example client",
        manufacturer="Example acoustics",
        mounting="Type A (mounted directly against a rigid wall)",
        test_room="Reverberation room (example)",
        measurement_standard="ISO 354",
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-354",
    )
    return result, metadata, "iso354_absorption_example.pdf"


def _impedance_tube_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10534-2 fiche: a two-microphone impedance-tube measurement.

    A documented clean-room example with a known closed-form absorption curve.
    The specimen is modelled as a locally-reacting resistive screen of
    normalised flow resistance theta = 1 backed by a rigidly-terminated air
    cavity of depth L, whose normalised surface impedance is the textbook
    z(f) = theta - j*cot(k0*L) (a resistive layer in series with the cavity
    reactance; Cox & D'Antonio, Acoustic Absorbers and Diffusers). From it the
    reflection factor r = (z - 1)/(z + 1) (ISO 10534-2 Eq. (19) inverted) and
    the absorption alpha = 1 - |r|^2 (Eq. (18)) follow exactly. The transfer
    function H12 that a tube would measure is synthesised from r via the
    Annex D field model (Eq. (D.7)) and fed back through
    ``two_microphone_impedance`` (Eq. (17)), so the fiche exercises the real
    reduction and its printed alpha matches the closed form.

    A 100 mm circular tube with s = 50 mm microphone spacing and the far mic
    at x1 = 100 mm works from f_l = c0/(20 s) ~ 343 Hz to the cut-on
    f_u = 0.58 c0/d ~ 1991 Hz at 20 degC (c0 = 343.29 m/s). The cavity depth
    L = c0/(4*1000 Hz) = 85.8 mm places the quarter-wave resonance at 1000 Hz,
    where the matched screen (theta = 1) gives z = 1, r = 0 and alpha = 1.00.
    Two further worked bands: at 500 Hz k0*L = pi/4, so cot = 1, z = 1 - j and
    alpha = 1 - |(-j)/(2 - j)|^2 = 1 - 1/5 = 0.80; at 1600 Hz the reactance is
    mass-like and alpha falls back to 0.68.
    """
    speed_of_sound_iso = ph.materials.speed_of_sound_iso
    air_density_iso = ph.materials.air_density_iso

    temperature_k = 293.15  # 20 degC
    pressure_kpa = 101.0
    c0 = float(speed_of_sound_iso(temperature_k))
    rho = float(air_density_iso(temperature_k, pressure_kpa))
    rc = ph.materials.characteristic_impedance(rho, c0)

    diameter, spacing, x1 = 0.100, 0.050, 0.100
    theta, cavity = 1.0, c0 / (4.0 * 1000.0)
    freqs = np.array([400, 500, 630, 800, 1000, 1250, 1600], dtype=float)

    k0 = 2.0 * np.pi * freqs / c0
    z = theta - 1j / np.tan(k0 * cavity)
    r = (z - 1.0) / (z + 1.0)
    # Synthesise H12 from the known r (ISO 10534-2 Annex D, Eq. (D.7)).
    kk = np.asarray(ph.materials.tube_wavenumber(freqs, c0))
    x2 = x1 - spacing
    h12 = (np.exp(1j * kk * x2) + r * np.exp(-1j * kk * x2)) / (
        np.exp(1j * kk * x1) + r * np.exp(-1j * kk * x1)
    )
    result = ph.materials.two_microphone_impedance(
        h12,
        frequency=freqs,
        spacing=spacing,
        x1=x1,
        speed_of_sound=c0,
        characteristic_impedance=rc,
        diameter=diameter,
        shape="circular",
    )
    metadata = ReportMetadata(
        specimen="Resistive facing over an 86 mm rigidly-backed air cavity",
        client="Example client",
        manufacturer="Example acoustics",
        tube_diameter=diameter,
        mic_spacing=spacing,
        mounting="Deliberate 86 mm backing air cavity, rigid termination",
        test_room="Impedance tube B&K 4206 (example)",
        measurement_standard="ISO 10534-2",
        temperature=20.0,
        pressure=pressure_kpa,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10534-2",
    )
    return result, metadata, "iso10534_impedance_tube_example.pdf"


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
    result = ph.loudness_zwicker_from_spectrum(levels, field="free")
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


def _tone_audibility_example() -> tuple[object, ReportMetadata, str]:
    """Tonal audibility fiche: an ISO 1996-2:2017 Annex J tone assessment.

    The narrow-band spectrum is the ISO/PAS 20065:2016 Annex E combustion-engine
    worked example (Table E.1, the 38 lines of Delta f = 2.7 Hz about the
    137.3 Hz tone). analyze_spectrum detects the three tones at 118.4 / 137.3 /
    158.8 Hz and their combined FG group; the decisive tonal audibility is the
    FG entry (Delta L_ta ~ 9.1 dB), which maps to the tonal adjustment K = 5 dB
    by ISO 1996-2:2017 Table J.1 (9 < Delta L <= 12). The complete-spectrum
    printed value is 9.18 dB (K = 5 either way); the small difference is the
    documented edge-tone masking-level underestimate on the truncated band.
    """
    frequencies = np.array(
        [
            96.9,
            99.6,
            102.3,
            105.0,
            107.7,
            110.4,
            113.0,
            115.7,
            118.4,
            121.1,
            123.8,
            126.5,
            129.2,
            131.9,
            134.6,
            137.3,
            140.0,
            142.7,
            145.3,
            148.0,
            150.7,
            153.4,
            156.1,
            158.8,
            161.5,
            164.2,
            166.9,
            169.6,
            172.3,
            175.0,
            177.6,
            180.3,
            183.0,
            185.7,
            188.4,
            191.1,
            193.8,
            196.5,
        ],
        dtype=float,
    )
    levels = np.array(
        [
            49.40,
            50.68,
            50.09,
            53.37,
            44.47,
            50.91,
            51.41,
            59.40,
            64.54,
            57.57,
            51.02,
            50.76,
            59.93,
            62.94,
            58.49,
            65.87,
            62.66,
            50.25,
            51.32,
            52.30,
            52.58,
            53.15,
            67.04,
            67.27,
            57.40,
            57.17,
            52.56,
            51.39,
            52.49,
            47.68,
            51.26,
            49.03,
            61.42,
            59.52,
            48.43,
            50.84,
            48.20,
            55.95,
        ],
        dtype=float,
    )
    result = ph.analyze_spectrum(levels, frequencies, 2.7)
    metadata = ReportMetadata(
        specimen="Combustion engine, steady operation",
        client="Example client",
        test_room="Free field, 3 m from source (example)",
        instrumentation="Class 1 analyser, FFT with 2.7 Hz lines (Hann window)",
        measurement_standard="ISO 1996-2",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-1996-TONE",
        requirement=6.0,
    )
    return result, metadata, "iso1996_tone_audibility_example.pdf"


def _impulse_prominence_example() -> tuple[object, ReportMetadata, str]:
    """Impulse-prominence fiche: an NT ACOU 112:2002 impulsive-sound assessment.

    Three candidate impulses of a pile-driving site, each ``(onset rate dB/s,
    level difference dB)``: (1200, 32), (300, 18) and (60, 11). All three qualify
    (onset rate above 10 dB/s), and the predicted prominence follows Formula 1
    P = 3 lg(OR) + 2 lg(LD): P1 = 3 lg 1200 + 2 lg 32 = 9.2375 + 3.0103 = 12.25,
    P2 = 3 lg 300 + 2 lg 18 = 9.94 and P3 = 3 lg 60 + 2 lg 11 = 7.42. The highest,
    the first impulse, governs (P = 12.25), and its LAeq adjustment is
    KI = 1.8 (12.2478 - 5) = 13.05 dB (Formula 2), rounding to 13.0 dB. The
    requirement is a plausible maximum governing prominence the example exceeds,
    so the optional verdict FAILs.
    """
    result = ph.environment.assessment.impulse_prominence(
        [1200.0, 300.0, 60.0], [32.0, 18.0, 11.0]
    )
    metadata = ReportMetadata(
        specimen="Pile-driving site, intermittent hammering",
        client="Example client",
        test_room="Free field, 25 m from source (example)",
        instrumentation="Class 1 SLM (IEC 61672-1), L_pAF logged at 100 ms",
        measurement_standard="ISO 1996-2",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-NTACOU112",
        requirement=10.0,
    )
    return result, metadata, "ntacou112_impulse_prominence_example.pdf"


def _wind_turbine_tonality_example() -> tuple[object, ReportMetadata, str]:
    """Wind-turbine tonality fiche: an IEC 61400-11:2012+A1:2018 assessment.

    A clean 500 Hz tone (a gearbox line) 30 dB above a flat 30 dB narrow-band
    floor, sampled at Delta f = 2 Hz over 440-560 Hz. The critical band about
    500 Hz is CBW = 117.256 Hz (Formula 30) and the ENBW is 1.5 * 2 = 3 Hz, so
    the masking-noise level is L_pn = 30 + 10 lg(117.256 / 3) = 45.92 dB
    (Formula 31); the single tone line gives L_pt = 60 dB, hence the tonality
    is Delta L_tn = 14.08 dB (Formula 32). The audibility criterion at 500 Hz
    is L_a = -2 - lg(1 + (500/502)^2.5) = -2.30 dB (Formula 34), so the tonal
    audibility is Delta L_a = 14.08 - (-2.30) = 16.38 dB (Formula 33): the tone
    is audible. The requirement is a plausible maximum acceptable tonal
    audibility the example exceeds, so the optional verdict FAILs.
    """
    df = 2.0
    frequencies = np.arange(440.0, 560.0 + df, df)
    levels = np.full(frequencies.size, 30.0)
    levels[int(np.argmin(np.abs(frequencies - 500.0)))] = 60.0
    result = ph.wind_turbine_tonality(levels, frequencies)
    metadata = ReportMetadata(
        specimen="Horizontal-axis wind turbine, gearbox tone",
        client="Example client",
        test_room="Ground board, downwind reference position (example)",
        instrumentation="Class 1 analyser, FFT with 2 Hz lines (Hann window)",
        measurement_standard="IEC 61400-11",
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-61400-11",
        requirement=6.0,
    )
    return result, metadata, "iec61400_wind_turbine_tonality_example.pdf"


def _epnl_example() -> tuple[object, ReportMetadata, str]:
    """EPNL fiche: an ICAO Annex 16 aircraft-noise-certification result.

    A deterministic synthetic half-second flyover: a broadband spectral shape
    peaking near 400 Hz swept by a Gaussian temporal gain (the aircraft
    approaching and receding) with a 2500 Hz fan tone that adds the tone
    correction, giving a valid ``EPNLResult`` with a peak PNLTM near mid-record
    and a clear 10 dB-down window. The requirement is a plausible certification
    EPNL limit the example passes.
    """
    k, dt = 24, 0.5
    idx = np.arange(k)
    shape = 15.0 * np.exp(
        -((np.log10(ph.aircraft.NOY_BANDS) - np.log10(400.0)) ** 2) / 0.5
    )
    gain = 24.0 * np.exp(-((idx - 12.0) ** 2) / (2 * 3.5**2)) - 3.0
    spectra = (46.0 + shape)[None, :] + gain[:, None]
    spectra[:, 17] += 10.0 * np.exp(-((idx - 12.0) ** 2) / (2 * 4.0**2))  # 2500 Hz tone
    result = ph.effective_perceived_noise_level(spectra, dt)
    metadata = ReportMetadata(
        specimen="Example twin-turbofan transport (synthetic flyover)",
        manufacturer="Example Aircraft Company",
        client="Example Aircraft Company",
        test_room="Flyover reference point",
        measurement_standard="ICAO Annex 16 Vol I Amendment 14 Chapter 4",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-EPNL",
        requirement=101.0,
    )
    return result, metadata, "icao_epnl_example.pdf"


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


def _iso4871_declaration_example() -> tuple[object, ReportMetadata, str]:
    """ISO 4871 fiche: a dual-number machinery noise-emission declaration.

    Reproduces the ISO 4871:1996 Annex B.2 example (Type 990, Model 11-TC): two
    operating modes with a measured A-weighted sound power level and an
    uncertainty of 2 dB, stated separately per the dual-number layout, plus
    emission sound pressure levels at the work station. A verification
    measurement is added per mode against the dual-number limit L_WA + K_WA
    (clause 6.2): mode 1 passes (89 <= 90) and mode 2 fails (98 > 97),
    exercising the verdict both ways.
    """
    mode1 = ph.OperatingModeDeclaration(
        mode="Operating mode 1",
        sound_power_level=88.0,
        sound_power_uncertainty=2.0,
        emission_pressure_level=78.0,
        emission_pressure_uncertainty=2.0,
        verification_level=89.0,
    )
    mode2 = ph.OperatingModeDeclaration(
        mode="Operating mode 2",
        sound_power_level=95.0,
        sound_power_uncertainty=2.0,
        emission_pressure_level=86.0,
        emission_pressure_uncertainty=2.0,
        verification_level=98.0,
    )
    result = ph.NoiseEmissionDeclaration(
        modes=(mode1, mode2),
        machine="Type 990, Model 11-TC",
        operating_conditions="50 Hz, 230 V, rated load",
        noise_test_code="ISO 3746 test code (example)",
        basic_standards=("ISO 3744", "ISO 11202"),
        form="dual-number",
    )
    metadata = ReportMetadata(
        measurement_standard="ISO 3744",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-4871",
        notes="Declaration reproduces the ISO 4871:1996 Annex B example machine.",
    )
    return result, metadata, "iso4871_declaration_example.pdf"


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


def _sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Sound-power fiche: an ISO 3744 engineering-grade determination.

    A floor-standing machine on one reflecting plane, measured on a hemisphere
    of radius r = 4 m (surface area S = 2*pi*r^2 = 100.53 m^2, ISO 3744:2010
    clause 7.2.3) at the ten key microphone positions (clause 8.1.1). The
    energy-averaged surface pressure levels (Eq. 12) per octave band (63 Hz to
    8 kHz) are 72, 76, 80, 82, 81, 78, 73 and 66 dB, each with the background
    noise a uniform 10 dB below, so the background correction is a uniform
    K1 = -10*lg(1 - 10^(-1,0)) = 0.46 dB (Eq. 16), and an equivalent absorption
    area A = 1500 m^2 gives K2 = 10*lg(1 + 4*S/A) = 1.03 dB (Eq. A.2), inside
    the 4 dB engineering validity limit. The surface level (Eq. 17) is then
    LW = Lp + 10*lg(S/S0) with 10*lg(S/S0) = 20.02 dB (Eq. 18), giving the band
    levels 90.5, 94.5, 98.5, 100.5, 99.5, 96.5, 91.5 and 84.5 dB and, with the
    Annex E octave A-weighting corrections (Table E.2, Eq. E.1), an A-weighted
    sound power level LWA = 103.7 dB(A) re 1 pW. The declared limit of
    105 dB(A) is met, so the verdict passes.
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    surface_pressure = np.array([72.0, 76, 80, 82, 81, 78, 73, 66])
    # A uniform, well-behaved field: ten identical position spectra, so the
    # energy average (Eq. 12) equals the documented surface-pressure spectrum.
    positions = np.tile(surface_pressure, (10, 1))
    result = ph.emission.sound_power_pressure(
        positions,
        "hemisphere",
        radius=4.0,
        reflecting_planes=1,
        background_levels=np.tile(surface_pressure - 10.0, (10, 1)),
        frequencies=freqs,
        absorption_area=1500.0,
        grade="engineering",
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Hemi-anechoic room over a reflecting floor",
        instrumentation="Class 1 sound level meter (IEC 61672-1), s/n 0042",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-3744",
        requirement=105.0,
        notes="Enveloping-surface pressure method over a hemisphere "
        "(ISO 3744:2010, engineering grade 2).",
    )
    return result, metadata, "iso3744_sound_power_example.pdf"


def _intensity_sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Sound-power-by-intensity fiche: an ISO 9614-2 engineering-grade scan.

    A machine enclosed by a hypothetical box divided into six equal segments of
    area Si = 0.5 m2 (measurement surface S = 3.0 m2). The probe is swept twice
    over each segment; the field is uniform, so every segment reports the same
    signed normal intensity per octave band (125 Hz to 4 kHz),
    In = [0.6, 1.0, 1.5, 1.4, 0.9, 0.5] x 1e-4 W/m2. The partial powers
    Pi = <In,i>*Si sum to the band power P = In*S (Eq. 12/6), so the band
    sound-power level is LW = 10*lg(In*S/P0), P0 = 1 pW (Eq. 13): 82.5, 84.8,
    86.5, 86.2, 84.3 and 81.8 dB. With the octave A-weighting corrections
    (-16.1, -8.6, -3.2, 0.0, 1.2, 1.0 dB) this gives LWA = 90.9 dB(A) re 1 pW.
    The two sweeps are identical (perfect repeatability), the surface SPL is a
    uniform 80 dB and the instrument pressure-residual index is delta_pI0 =
    15 dB, so the dynamic capability Ld = 15 - 10 = 5 dB clears FpI in every
    band and all six bands qualify at engineering grade. The declared limit of
    93 dB(A) is met, so the verdict passes.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    intensity = np.array([0.6e-4, 1.0e-4, 1.5e-4, 1.4e-4, 0.9e-4, 0.5e-4])
    areas = np.full(6, 0.5)
    # A uniform field: the six segments share one intensity spectrum, and the
    # two sweeps coincide, so P = In*S and the repeatability is exact.
    scan = np.tile(intensity, (6, 1))
    result = ph.emission.sound_power_intensity(
        scan,
        areas,
        normal_intensity_2=scan.copy(),
        pressure_levels=np.full((6, 6), 80.0),
        pressure_residual_index=15.0,
        frequencies=freqs,
        band_type="octave",
        grade="engineering",
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Machine hall with steady background noise",
        instrumentation="Class 1 p-p intensity probe (IEC 61043), s/n 0042",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9614",
        requirement=93.0,
        notes="Intensity-scanning method over a box surface "
        "(ISO 9614-2:1996, engineering grade 2).",
    )
    return result, metadata, "iso9614_sound_power_intensity_example.pdf"


def _reverberation_sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Reverberation-room fiche: an ISO 3741 precision-grade determination.

    A steady, broadband source measured by the direct method in a qualified
    hard-walled reverberation room of volume V = 200 m3 and total surface
    S = 240 m2 (ISO 3741:2010 clause 9.1.4). The octave-band mean room
    sound-pressure levels Lp (125 Hz to 8 kHz octave bands) peak near 500 Hz. At
    the test temperature theta = 20 degC the speed of sound is
    c = 20.05*sqrt(293) = 343.2 m/s and, with a uniform reverberation time
    T60 = 2.0 s, the Sabine equivalent absorption area
    A = (55.26/c)*(V/T60) = 16.10 m2 is constant across bands, so
    10*lg(A/A0) = 12.07 dB and 4.34*(A/S) = 0.29 dB are fixed; the Waterhouse
    boundary correction 10*lg(1 + S*c/(8*V*f)) falls from 1.50 dB at 125 Hz to
    0.03 dB at 8 kHz. At 101.325 kPa the meteorological corrections are
    C1 = 5*lg(293.15/314) = -0.15 dB and C2 = 15*lg(293.15/296) = -0.06 dB.
    Eq. (20) then gives the band levels, e.g. LW = 90.0 dB at 250 Hz, 91.6 dB
    at 500 Hz and 90.4 dB at 1 kHz, a total LW = 96.7 dB and, with the Annex F
    (Annex E) A-weighting corrections, an A-weighted sound power level
    LWA = 94.3 dB(A) re 1 pW. The declared limit of 96 dB(A) is met, so the
    verdict passes.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    # Documented mean corrected room level Lp(ST) per band (Eq. 16), in dB.
    lp = np.array([80.0, 83.0, 85.0, 84.0, 80.0, 75.0, 68.0])
    result = ph.emission.sound_power_reverberation(
        lp,
        2.0,
        volume=200.0,
        surface_area=240.0,
        frequencies=freqs,
        temperature=20.0,
        static_pressure=101.325,
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Qualified reverberation room, V = 200 m3, T60 = 2.0 s",
        instrumentation="Class 1 sound level meter (IEC 61672-1), s/n 0042",
        temperature=20.0,
        relative_humidity=50.0,
        pressure=101.325,
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-3741",
        requirement=96.0,
        notes="Reverberation-room direct method (ISO 3741:2010, precision grade 1).",
    )
    return result, metadata, "iso3741_reverberation_power_example.pdf"


def _vibration_sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Sound-power-from-vibration fiche: an ISO/TS 7849-2 engineering example.

    A documented clean-room example (ISO/TS 7849 gives no numeric worked case
    beyond the calibration example, so the band spectrum is built from the
    closed-form Eq. 12/15, as the standard directs). A machine casing of
    radiating area S = 1.6 m2 (so 10*lg(S/S0) = 2.04 dB) is surveyed over six
    octave bands (125 Hz to 4 kHz); the surface-averaged vibratory velocity
    level (Eq. 3) is Lv = [78, 82, 85, 83, 79, 74] dB re 5e-8 m/s and the
    band-wise radiation factor determined from an independent power measurement
    (Eq. 8, ISO 9614) is epsilon = [0.20, 0.45, 0.75, 0.95, 1.00, 1.00]. The
    radiated band sound-power level is
    LW = Lv + 10*lg(S/S0) + 10*lg(epsilon) + 10*lg(411/400) with the fixed
    impedance term 10*lg(411/400) = 0.12 dB, giving LW = [73.2, 80.7, 85.9,
    84.9, 81.2, 76.2] dB re 1 pW, a total LW = 90.0 dB and, with the octave
    A-weighting corrections (-16.1, -8.6, -3.2, 0.0, 1.2, 1.0 dB), an A-weighted
    sound power level LWA = 88.7 dB(A) re 1 pW. The declared limit of 90 dB(A)
    is met, so the verdict passes.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    lv = np.array([78.0, 82.0, 85.0, 83.0, 79.0, 74.0])
    eps = np.array([0.20, 0.45, 0.75, 0.95, 1.00, 1.00])
    result = ph.emission.sound_power_from_vibration(
        lv, area=1.6, radiation_factor=eps, frequencies=freqs
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Gearbox casing (steel panel)",
        test_room="Machine hall (source vibration survey)",
        instrumentation="Piezoelectric accelerometer (ISO 16063-21 calibration), s/n 0042",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-7849",
        requirement=90.0,
        notes="Sound power from surface vibration, engineering method "
        "(ISO/TS 7849-2:2009).",
    )
    return result, metadata, "iso7849_vibration_power_example.pdf"


def _structure_borne_power_example() -> tuple[object, ReportMetadata, str]:
    """Structure-borne source power fiche: an EN 15657 reception-plate test.

    A pump fixed to a reception plate of mass per area m = 25 kg/m2 and area
    S = 1,2 m2 whose structural reverberation time Ts = 0,3 s gives the plate
    loss factor eta = 2,2/(f*Ts) (Formula 13). The spatial-average plate
    velocity level (Formula 12) per octave band (125 Hz to 4 kHz) is
    Lv = [88, 90, 86, 82, 78, 73] dB re 1e-9 m/s, and the structure-borne sound
    power injected into the plate is
    L_Ws = 10*lg(2*pi*f*eta*m*S) + Lv - 60 dB re 1 pW (Formula 14). The band
    levels are dominated by the 250 Hz band and sum to a total L_Ws near 65 dB
    re 1 pW. The level is specific to this reception plate; the conversion to
    the plate-independent source quantities (Formulae 15/17) precedes any
    EN 12354-5 use, as the basis strip states.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    lv = np.array([88.0, 90.0, 86.0, 82.0, 78.0, 73.0])
    result = ph.reception_plate_power(
        lv, freqs, mass_per_area=25.0, area=1.2, reverberation_time=0.3
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


#: One-third-octave centre frequencies of ISO 17497 Table 1 / Clause 5, in Hz
#: (100 Hz to 5000 Hz, full scale).
_SCATTER_FREQS = np.array(
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


def _detailed_building() -> tuple[dict, "np.ndarray", "np.ndarray"]:
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


def _precision_sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Precision sound-power fiche: ISO 3745:2012 in an anechoic room.

    The guide's own worked example: a mid-frequency-peaked machine measured
    over the forty standardized hemisphere positions of Annex E at a radius of
    1 m (surface S = 2*pi*r^2 = 6,283 m2), with a base spectrum peaked near
    1 kHz and a 1 dB per-position spread from a seeded generator, so the fiche
    and the guide print the same numbers. The determination gives an
    A-weighted sound power level LWA = 89,3 dB(A) re 1 pW. The expanded
    uncertainty is the Clause 10.5 example: the method's own sigma_omc =
    2,0 dB at k = 2, over the ISO 3745 Table 1 reproducibility standard
    deviation of the band.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    base = 70.0 + 8.0 * np.exp(-(np.log2(freqs / 1000.0) ** 2) / 2.0)
    rng = np.random.default_rng(7)
    levels = base[None, :] + rng.normal(0.0, 1.0, (40, freqs.size))
    result = ph.emission.sound_power_anechoic(
        levels,
        "hemisphere",
        radius=1.0,
        frequencies=freqs,
        sigma_omc=2.0,
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Mid-frequency-peaked machine (guide example)",
        test_room="Qualified anechoic room, 40-position hemisphere array",
        measurement_standard="ISO 3745",
        test_date="2026-08-05",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-3745",
        notes=(
            "Precision grade: Annex E hemisphere array at r = 1 m, "
            "meteorological corrections at the 23 C / 101,325 kPa reference, "
            "expanded uncertainty from the Clause 10.5 example."
        ),
    )
    return result, metadata, "iso3745_precision_power_example.pdf"


def _scattering_example() -> tuple[object, ReportMetadata, str]:
    """ISO 17497-1 fiche: a random-incidence scattering-coefficient measurement.

    A documented clean-room example (ISO 17497-1 has no numeric worked example,
    so an end-to-end synthetic case is built from Eqs. (1)/(4)/(5)/(6), as the
    standard itself directs). A reverberation room of volume V = 200 m3 holds a
    circular test sample of area S = 10 m2 on a turntable; at 20 degC the speed
    of sound is c = 343.2 m/s (Eq. (2)) and the air attenuation is neglected
    (m = 0, the zero-attenuation reference). The four reverberation times of
    Table 2 are chosen with a perfectly symmetrical base plate (T1 = T3, so the
    base-plate scattering of Eq. (6) is exactly 0 and passes the Table 1 limits)
    and a rotating turntable whose apparent (specular) absorption grows with
    frequency as the surface relief scatters more energy out of the specular
    direction. The random-incidence absorption alpha_s (Eq. (1)) stays below the
    0.50 ceiling of Clause 6.3.4. Two worked bands, from
    s = (alpha_spec - alpha_s) / (1 - alpha_s) (Eq. (5)):

    * 500 Hz: alpha_s = 0.053, alpha_spec = 0.131, s = 0.082 -> 0.08.
    * 4000 Hz: alpha_s = 0.112, alpha_spec = 0.515, s = 0.454 -> 0.45.

    The scattering coefficient rises from 0.01 at 100 Hz to 0.55 at 5000 Hz, a
    broadband diffusing surface.
    """
    volume, area, c = 200.0, 10.0, 343.2
    t1 = np.array(
        [
            8.0,
            7.9,
            7.8,
            7.6,
            7.4,
            7.2,
            7.0,
            6.7,
            6.4,
            6.0,
            5.6,
            5.2,
            4.8,
            4.4,
            4.0,
            3.6,
            3.2,
            2.9,
        ]
    )
    t3 = t1.copy()  # symmetrical base plate: T1 = T3
    t2 = t1 * 0.90  # sample, static turntable
    t4 = t2 * (1.0 - np.linspace(0.02, 0.28, _SCATTER_FREQS.size))
    alpha_s = ph.materials.random_incidence_absorption(
        volume, area, c1=c, T1=t1, c2=c, T2=t2
    )
    alpha_spec = ph.materials.specular_absorption_coefficient(
        volume, area, c3=c, T3=t3, c4=c, T4=t4
    )
    result = ph.materials.scattering_coefficient_spectrum(
        _SCATTER_FREQS, alpha_spec, alpha_s
    )
    metadata = ReportMetadata(
        specimen="1:1 quadratic-residue diffuser (N = 7)",
        client="Example client",
        manufacturer="Example acoustics",
        area=area,
        room_volume=volume,
        mounting="Circular sample on the rotating turntable, centre displaced d/8",
        test_room="Reverberation room (example)",
        measurement_standard="ISO 17497-1",
        temperature=20.0,
        relative_humidity=54.0,
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-17497-1",
    )
    return result, metadata, "iso17497_scattering_example.pdf"


#: The 2-D single-plane source positions of ISO 17497-2 Clause 6.2.2 (0 deg and
#: +/-30 deg, +/-60 deg about the reference normal); paired with the Clause 8.4
#: source weights (0 deg -> 1, the four others -> 3).
_DIFFUSION_SOURCES = np.array([0.0, 30.0, -30.0, 60.0, -60.0])


def _diffuser_polar_energy(
    angles: np.ndarray, width: float, peak: float, specular: float = 0.0
) -> np.ndarray:
    """A synthetic reflected-level polar response (a specular lobe over a floor).

    The band energy is a diffuse floor of unity plus a specular lobe of linear
    amplitude ``peak`` and Gaussian half-width ``width`` (degrees) centred on the
    ``specular`` reflection angle; the level is ``10 lg(energy) + 60`` dB.
    """
    energy = 1.0 + peak * np.exp(-(((angles - specular) / width) ** 2))
    return 10.0 * np.log10(energy) + 60.0


def _diffusion_example() -> tuple[object, ReportMetadata, str]:
    """ISO 17497-2 fiche: a random-incidence diffusion-coefficient spectrum d(f).

    A documented clean-room example (ISO 17497-2 has no numeric worked example
    or reference polar dataset, so the polar responses are synthesised and the
    coefficient computed from Formula (5), as the standard directs). A
    single-plane goniometer sweeps 19 equal-area receivers from -90 to 90 deg
    (10 deg spacing) about the reference normal for each of the five 2-D source
    positions of Clause 6.2.2 (0 deg and +/-30 deg, +/-60 deg), whose specular
    reflection falls at the mirror angle. As frequency rises the diffuser spreads
    the reflected energy ever more evenly (the specular lobe broadens and
    flattens), so the directional coefficient d_theta (Formula (5)) of each
    source climbs with frequency. The per-band **random-incidence** coefficient
    d (Clause 8.4) is the weighted average of the five directional coefficients
    over the source positions (0 deg -> 1, the four others -> 3), computed band
    by band, and the normalised d_n (Formula (7), against a rigid flat reference
    of the same footprint) is likewise averaged over the sources. Both climb
    with frequency: d from 0.23 at 100 Hz to 0.86 at 5000 Hz. Two worked bands:
    at 500 Hz d = 0.51 (d_n = 0.35); at 4000 Hz d = 0.81 (d_n = 0.68).
    """
    angles = np.arange(-90.0, 90.5, 10.0)
    n = _SCATTER_FREQS.size
    widths = np.linspace(15.0, 70.0, n)
    peaks = np.linspace(30.0, 3.0, n)
    weights = np.array(ph.materials.TWO_DIMENSIONAL_SOURCE_WEIGHTS, dtype=float)
    d = np.empty(n)
    d_n = np.empty(n)
    for k in range(n):
        d_theta = []
        d_theta_n = []
        for source in _DIFFUSION_SOURCES:
            specular = -source  # specular reflection about the reference normal
            d_s = ph.materials.directional_diffusion_coefficient(
                _diffuser_polar_energy(angles, widths[k], peaks[k], specular)
            )
            d_ref = ph.materials.directional_diffusion_coefficient(
                _diffuser_polar_energy(angles, 0.5 * widths[k], 60.0, specular)
            )
            d_theta.append(d_s)
            d_theta_n.append(
                float(ph.materials.normalized_diffusion_coefficient(d_s, d_ref))
            )
        # Clause 8.4: average the directional coefficients over the source
        # positions, band by band, to get the random-incidence coefficient.
        d[k] = ph.materials.random_incidence_diffusion(d_theta, weights=weights)
        d_n[k] = ph.materials.random_incidence_diffusion(d_theta_n, weights=weights)
    result = ph.materials.diffusion_spectrum(_SCATTER_FREQS, d, normalized=d_n)
    metadata = ReportMetadata(
        specimen="1:1 single-plane Schroeder diffuser (N = 7)",
        client="Example client",
        manufacturer="Example acoustics",
        mounting="Single-plane diffuser, plane of maximum diffusion",
        test_room="Anechoic goniometer (example), source at 10 m, arc at 5 m",
        measurement_standard="ISO 17497-2",
        temperature=20.0,
        relative_humidity=50.0,
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-17497-2",
    )
    return result, metadata, "iso17497_diffusion_example.pdf"


def _diffusion_polar_example() -> tuple[object, ReportMetadata, str]:
    """ISO 17497-2 fiche: the single-source polar response of one band.

    The corrected 1000 Hz polar response behind the ``_diffusion_example``
    spectrum (Clause 8.5): 19 equal-area receivers from -90 to 90 deg, whose
    autocorrelation diffusion coefficient d = 0.67 (Formula (5)) for the
    normal-incidence source position.
    """
    angles = np.arange(-90.0, 90.5, 10.0)
    widths = np.linspace(15.0, 70.0, _SCATTER_FREQS.size)
    peaks = np.linspace(30.0, 3.0, _SCATTER_FREQS.size)
    band = int(np.argmin(np.abs(_SCATTER_FREQS - 1000.0)))
    levels = _diffuser_polar_energy(angles, widths[band], peaks[band])
    result = ph.materials.directional_diffusion(angles, levels)
    metadata = ReportMetadata(
        specimen="1:1 single-plane Schroeder diffuser (N = 7)",
        client="Example client",
        manufacturer="Example acoustics",
        mounting="Single-plane diffuser, normal-incidence source (0 deg)",
        test_room="Anechoic goniometer (example), source at 10 m, arc at 5 m",
        measurement_standard="ISO 17497-2",
        temperature=20.0,
        relative_humidity=50.0,
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-17497-2P",
    )
    return result, metadata, "iso17497_diffusion_polar_example.pdf"


def _dynamic_stiffness_example() -> tuple[object, ReportMetadata, str]:
    """EN 29052-1 fiche: the dynamic stiffness of a resilient floating-floor layer.

    A 20 mm mineral-wool resilient layer under the standard 8 kg load plate on
    the 0.04 m2 specimen (total mass per unit area m't = 8 kg / 0.04 m2 =
    200 kg/m2, EN 29052-1:1992 Clauses 5 and 6), whose fundamental resonance is
    measured at fr = 45.0 Hz. Formula 4 gives the apparent dynamic stiffness
    s't = 4*pi^2 * m't * fr^2 = 15.99 MN/m3 -> 16 MN/m3 (Clause 9 rounds to the
    nearest MN/m3). At the intermediate lateral airflow resistivity of
    r = 50 kPa.s/m2 the enclosed-gas term applies (Clause 8.2 b): s'a = 111/d =
    5.56 MN/m3 -> 6 MN/m3 (Clause 8.2 NOTE, d = 20 mm), so the installed
    stiffness is s' = s't + s'a = 21.54 MN/m3 -> 22 MN/m3 (Formula 6). Installed
    under a 110 kg/m2 floating screed the natural frequency is
    f0 = (1/2pi) sqrt(s'/m') = 70.4 Hz (Formula 2).
    """
    result = ph.materials.floating_floor_resonance(
        resonant_frequency=45.0,
        total_mass_per_area=200.0,
        floor_mass_per_area=110.0,
        airflow_resistivity=50.0,
        thickness=0.020,
        porosity=0.9,
    )
    metadata = ReportMetadata(
        specimen="20 mm mineral-wool resilient layer",
        client="Example client",
        manufacturer="Example insulation works",
        mass_per_area=200.0,
        thickness=0.020,
        test_room="Dynamic-stiffness rig (example), 8 kg load plate",
        measurement_standard="EN 29052-1",
        temperature=21.0,
        relative_humidity=50.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-29052-1",
    )
    return result, metadata, "en29052_dynamic_stiffness_example.pdf"


def _airflow_resistance_example() -> tuple[object, ReportMetadata, str]:
    """ISO 9053-1 fiche: the static airflow resistance of a porous specimen.

    A 50 mm porous absorber measured in a 100 mm diameter cell (cross-section
    A = pi*0.05^2 = 7.854e-3 m2, ISO 9053-1:2018 clause 7). The linear airflow
    velocity is stepped up to 12 mm/s (below the 15 mm/s clause-7.5 limit) and
    the measured pressure difference fitted through the origin with a
    second-order regression dp = a*u + b*u^2 (clause 7.5), here a = 16000 Pa*s/m
    and b = 400000 Pa*s^2/m^2. Read at the reference velocity u = 0.5 mm/s this
    gives R_s = a + b*u = 16200 Pa*s/m, an airflow resistance
    R = R_s/A = 2.06e6 Pa*s/m^3 and, for the 50 mm thickness, an airflow
    resistivity sigma = R_s/d = 324000 Pa*s/m^2.
    """
    area = np.pi * 0.05**2
    u = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0]) * 1e-3
    dp = 1.6e4 * u + 4.0e5 * u**2
    result = ph.materials.static_airflow_resistance(
        u, dp, area=area, thickness=0.05
    )
    metadata = ReportMetadata(
        specimen="50 mm porous absorber (open-cell)",
        client="Example client",
        manufacturer="Example insulation works",
        thickness=0.050,
        test_room="Static airflow rig (example), 100 mm cell",
        measurement_standard="ISO 9053-1",
        temperature=23.0,
        relative_humidity=50.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9053-1",
    )
    return result, metadata, "iso9053_airflow_resistance_example.pdf"


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
    measured = ph.transfer_stiffness_direct(k21 * u1, u1)
    result = ph.TransferStiffnessResult(
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


class _WithSourceEmission:
    """Adapter binding a source emission to an ``OutdoorAttenuation.report`` call.

    The uniform generator drives every fiche with ``result.report(path,
    metadata=...)``; the ISO 9613-2 attenuation fiche needs the source emission
    too (a report-time, display-only object), so this adapter carries it while
    keeping the generator loop unchanged.
    """

    def __init__(self, result: Any, emission: ph.SourceEmission) -> None:
        self._result = result
        self._emission = emission

    def report(self, path: str, *, metadata: ReportMetadata | None = None) -> str:
        return str(
            self._result.report(
                path, metadata=metadata, source_emission=self._emission
            )
        )


def _outdoor_attenuation_example() -> tuple[object, ReportMetadata, str]:
    """ISO 9613-2 fiche: predicted outdoor propagation attenuation with a barrier.

    An industrial point source (octave-band Lw from 95 dB at 63 Hz to 88 dB at
    8 kHz, supplied at report time via SourceEmission) 200 m upwind of a dwelling
    over porous ground (Gs = Gm = Gr = 1), screened by a noise barrier
    (source-edge = edge-receiver = 105 m, so the diffracted path exceeds the
    direct one). The divergence, atmospheric, ground and barrier terms come from
    the tested clause-7 functions (see
    tests/environment/propagation/test_outdoor_propagation.py).
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    lw = np.array([95, 100, 103, 105, 104, 101, 95, 88], dtype=float)
    barrier = ph.Barrier(source_to_edge=105.0, edge_to_receiver=105.0)
    result = ph.outdoor_propagation_attenuation(
        200.0, 4.0, 2.0, freqs, 1.0, 1.0, 1.0, barrier=barrier,
        temperature=10.0, relative_humidity=70.0,
    )
    emission = ph.SourceEmission(sound_power_level=lw)
    metadata = ReportMetadata(
        specimen="Industrial fan plant (point source)",
        client="Example client",
        test_room="Nearest dwelling facade",
        temperature=10.0,
        relative_humidity=70.0,
        pressure=101.3,
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9613-ATTEN",
        requirement=50.0,  # maximum acceptable A-weighted downwind level
        notes="Point source over porous ground with a noise barrier (ISO 9613-2).",
    )
    return (
        _WithSourceEmission(result, emission),
        metadata,
        "iso9613_outdoor_attenuation_example.pdf",
    )


def _barrier_insertion_loss_example() -> tuple[object, ReportMetadata, str]:
    """ISO 9613-2 family fiche: predicted barrier insertion loss (wave-theoretic).

    A 4 m thin noise barrier 50 m from a source (1 m high), the receiver 1.5 m
    high at 100 m, in the free field. The per-band insertion loss comes from the
    tested wave-theoretic rigid-screen model (see
    tests/environment/propagation/test_ground_barriers.py).
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    result = ph.barrier_insertion_loss(freqs, 1.0, 50.0, 4.0, 100.0, 1.5)
    metadata = ReportMetadata(
        specimen="Roadside noise barrier, 4 m high",
        client="Example client",
        test_room="Dwelling at 100 m",
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9613-BARRIER",
        requirement=8.0,  # minimum required mean insertion loss
        notes="Thin-screen diffraction, wave-theoretic model.",
    )
    return result, metadata, "iso9613_barrier_insertion_loss_example.pdf"


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


def _rd1367_example() -> tuple[object, ReportMetadata, str]:
    """RD 1367/2007 fiche: the noise assessment of an activity.

    The published worked case of Aviles Lopez & Perera Martin, Manual de
    acustica ambiental y arquitectonica, Ejemplos 3.1 to 3.3: an activity on
    residential land (acoustic area type a) open from 9 h to 21 h, with a noisy
    machine running from 9 h to 15 h. The two measured noise phases are
    LAeq,5s = 50 dB with Kt = 6 and Kf = 3 dB (so LKeq,5s = 59 dB) and
    LAeq,5s = 48 dB with Kt = 3 and Kf = 3 dB (LKeq,5s = 54 dB). Integrating
    them over the evaluation periods gives LKeq,d = 57 dB and LKeq,e = 51 dB,
    and averaging over the 303 operating days of the year LK,d = 56 dB and
    LK,e = 50 dB. Against the 55 dB of Annex III Table B1 the phase and daily
    criteria are met but the annual LK,d is not, so a new activity does not
    comply.

    The fiche renders in Spanish, the language of the regulation it applies.
    """
    day = [
        ph.NoisePhase(2.0, 0.0, label="Actividad cerrada"),
        ph.NoisePhase(6.0, 50.0, kt=6.0, kf=3.0, label="Maquina ruidosa activa"),
        ph.NoisePhase(4.0, 48.0, kt=3.0, kf=3.0, label="Resto de fuentes"),
    ]
    evening = [
        ph.NoisePhase(2.0, 48.0, kt=3.0, kf=3.0, label="Resto de fuentes"),
        ph.NoisePhase(2.0, 0.0, label="Actividad cerrada"),
    ]
    result = ph.assess_activity(
        {"day": day, "evening": evening},
        ph.activity_limits("a"),
        operating_days=303,
    )
    metadata = ReportMetadata(
        specimen="Actividad con maquinaria, horario 9 h a 21 h",
        client="Example client",
        test_room="Ambiente exterior, punto de evaluacion mas desfavorable",
        instrumentation="Sonometro integrador-promediador clase 1",
        calibration="Verificacion antes y despues, desviacion 0,1 dB",
        measurement_standard="RD 1367/2007 Anexo IV",
        test_date="2026-07-29",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-RD1367",
    )
    return result, metadata, "rd1367_activity_example.pdf"


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


#: Every example fiche the repository keeps rendered, from the file it writes
#: to the factory that builds it. New report kinds add an entry here so
#: ``make reports`` regenerates the full set. Keyed by filename so that the set
#: of rendered outputs can be read off the registry: naming it is what a caller
#: asking "is every registered fiche committed?" needs, and calling the factory
#: for it would run a full computation and build a plot for an answer already
#: written here.
_FICHES: dict[str, Callable[[], tuple[object, ReportMetadata, str]]] = {
    "iso717_airborne_example.pdf": _airborne_example,
    "iso717_impact_example.pdf": _impact_example,
    "iso16283_airborne_example.pdf": _field_airborne_example,
    "iso16283_impact_example.pdf": _field_impact_example,
    "iso10140_airborne_example.pdf": _lab_airborne_example,
    "iso10140_impact_example.pdf": _lab_impact_example,
    "iso15186_intensity_example.pdf": _intensity_example,
    "iso15186_element_example.pdf": _intensity_element_example,
    "iso12354_airborne_prediction_example.pdf": _airborne_prediction_example,
    "iso12354_impact_prediction_example.pdf": _impact_prediction_example,
    "iso12354_facade_prediction_example.pdf": _facade_prediction_example,
    "iso16251_floor_covering_example.pdf": _floor_covering_example,
    "iso11654_absorption_example.pdf": _absorption_example,
    "iso354_absorption_example.pdf": _sound_absorption_example,
    "iso10534_impedance_tube_example.pdf": _impedance_tube_example,
    "iso532_loudness_example.pdf": _loudness_example,
    "ebu_r128_loudness_example.pdf": _program_loudness_example,
    "iso1996_tone_audibility_example.pdf": _tone_audibility_example,
    "ntacou112_impulse_prominence_example.pdf": _impulse_prominence_example,
    "iec61400_wind_turbine_tonality_example.pdf": _wind_turbine_tonality_example,
    "icao_epnl_example.pdf": _epnl_example,
    "iec61260_filter_example.pdf": _filter_class_example,
    "iec61260_filter_1995_example.pdf": _filter_class_1995_example,
    "iec61043_intensity_example.pdf": _intensity_class_example,
    "iso4871_declaration_example.pdf": _iso4871_declaration_example,
    "iec60268_5_loudspeaker_example.pdf": _loudspeaker_example,
    "iec60268_4_microphone_example.pdf": _microphone_example,
    "iso9612_exposure_example.pdf": _occupational_exposure_example,
    "human_vibration_example.pdf": _human_vibration_example,
    "iso1999_nipts_example.pdf": _nipts_example,
    "iso1999_htlan_example.pdf": _htlan_example,
    "iso3382_room_acoustics_example.pdf": _room_acoustics_example,
    "reverberation_prediction_example.pdf": _reverberation_prediction_example,
    "enclosed_space_absorption_example.pdf": _enclosed_space_absorption_example,
    "ansi_s12_2_noise_criteria_example.pdf": _noise_criteria_example,
    "ansi_s12_2_room_criteria_example.pdf": _room_criteria_example,
    "iso3382_3_open_plan_example.pdf": _open_plan_example,
    "iso2631_5_multiple_shock_example.pdf": _multiple_shock_example,
    "iso7626_mobility_example.pdf": _mechanical_mobility_example,
    "iso10846_transfer_stiffness_example.pdf": _transfer_stiffness_example,
    "iso3744_sound_power_example.pdf": _sound_power_example,
    "iso9614_sound_power_intensity_example.pdf": _intensity_sound_power_example,
    "iso3741_reverberation_power_example.pdf": _reverberation_sound_power_example,
    "iso7849_vibration_power_example.pdf": _vibration_sound_power_example,
    "en15657_structure_borne_power_example.pdf": _structure_borne_power_example,
    "en12354_5_installed_structure_borne_example.pdf": _installed_structure_borne_example,
    "iso17497_scattering_example.pdf": _scattering_example,
    "iso17497_diffusion_example.pdf": _diffusion_example,
    "iso17497_diffusion_polar_example.pdf": _diffusion_polar_example,
    "en29052_dynamic_stiffness_example.pdf": _dynamic_stiffness_example,
    "iso9053_airflow_resistance_example.pdf": _airflow_resistance_example,
    "iso10848_kij_example.pdf": _vibration_reduction_example,
    "iso10848_dnf_example.pdf": _flanking_level_difference_example,
    "iso10848_lnf_example.pdf": _flanking_impact_level_example,
    "iso10052_airborne_example.pdf": _survey_airborne_example,
    "iso10052_impact_example.pdf": _survey_impact_example,
    "iso10052_facade_example.pdf": _survey_facade_example,
    "iso16283_facade_example.pdf": _field_facade_example,
    "iso9613_outdoor_attenuation_example.pdf": _outdoor_attenuation_example,
    "iso9613_barrier_insertion_loss_example.pdf": _barrier_insertion_loss_example,
    "iec60268_16_sti_example.pdf": _sti_example,
    "ansi_s3_5_sii_example.pdf": _sii_example,
    "enclosure_insertion_loss_example.pdf": _enclosure_example,
    "reactive_silencer_example.pdf": _silencer_example,
    "hvac_duct_noise_example.pdf": _hvac_example,
    "duct_path_example.pdf": _duct_path_example,
    "rd1367_activity_example.pdf": _rd1367_example,
    "iso12354_detailed_airborne_example.pdf": _detailed_airborne_example,
    "iso12354_detailed_impact_example.pdf": _detailed_impact_example,
    "iso3745_precision_power_example.pdf": _precision_sound_power_example,
}

#: The registered factories alone, in generation order.
_EXAMPLES: list[Callable[[], tuple[object, ReportMetadata, str]]] = list(
    _FICHES.values()
)


def preview_path_for(pdf_path: str) -> str:
    """Return the WebP-preview path that pairs with ``pdf_path`` (``.pdf`` -> ``.webp``)."""
    return os.path.splitext(pdf_path)[0] + ".webp"


def _write_preview(pdf_path: str) -> str:
    """Rasterize the first page of ``pdf_path`` to a lossless WebP beside it.

    The preview is what the documentation embeds inline; it is not byte-compared
    in CI (the raster inherits the vector plot's ~1 ULP cross-CPU drift), but
    :mod:`scripts.check_reports` compares it within a tolerance, so a stale
    preview fails the build like a stale figure. Lossless WebP keeps the
    preview pixel-identical to the raster at roughly half the optimized-PNG
    size; ``method=6`` is the slowest, most exhaustive encoder search, whose
    output is fixed by the input pixels (byte-stable across runs).
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[0]
        width_pt, _ = page.get_size()
        scale = _PREVIEW_WIDTH_PX / width_pt
        image = page.render(scale=scale).to_pil()
        preview = preview_path_for(pdf_path)
        # RGB (no alpha): the fiche is opaque white; a smaller, simpler image.
        image.convert("RGB").save(preview, "WEBP", lossless=True, quality=100, method=6)
    finally:
        pdf.close()
    return preview


def generate_reports(
    output_dir: str,
    examples: Sequence[Callable[[], tuple[object, ReportMetadata, str]]] | None = None,
) -> list[str]:
    """Write every example fiche (PDF + WebP preview) into ``output_dir``.

    ``examples`` restricts the run to a subset of the registered factories
    (the test suite renders one fiche per test); the default renders the
    full registry. Returns the PDF paths written; each has a paired
    ``.webp`` preview next to it (see :func:`preview_path_for`).
    """
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    for factory in _EXAMPLES if examples is None else examples:
        result, metadata, name = factory()
        path = os.path.join(output_dir, name)
        result.report(path, metadata=metadata)  # type: ignore[attr-defined]
        _write_preview(path)
        written.append(path)
    return written


def main() -> None:
    """Generate the example fiches into the committed directory (or ``--output-dir``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=_DEFAULT_DIR)
    args = parser.parse_args()
    for path in generate_reports(os.path.abspath(args.output_dir)):
        print(f"wrote {path}")
        print(f"wrote {preview_path_for(path)}")


if __name__ == "__main__":
    main()
