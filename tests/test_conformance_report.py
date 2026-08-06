#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Smoke test for the CI numerical conformance harness.

Guarantees the registry runs end to end and that every registered
(standard, quantity) check passes on the current tree, so a regression in
any conformance-covered quantity fails the suite as well as the PR comment.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import conformance_report as cr

# One xdist worker runs the whole module: the registry memoizes each check's
# outcome per process, so the full-report render and the per-check tests
# together compute every check exactly once instead of once per worker.
pytestmark = pytest.mark.xdist_group("conformance-report")


def test_registry_is_populated() -> None:
    # Realistic floors for the current registry size (95 checks / 22 domains
    # as of the tests-audit batch): loose enough to allow pruning a couple of
    # checks, tight enough that losing a whole domain fails.
    assert len(cr.CHECKS) >= 90
    # Every domain has at least one check.
    assert len(cr._domains()) >= 20


def test_block_a_psychoacoustics_checks_registered() -> None:
    """The ECMA-418-2 and ISO 532-2/3 calibration checks are wired."""
    standards = {c.standard for c in cr.CHECKS}
    assert "ECMA-418-2:2025 Clause 5.1.8" in standards  # HMS loudness
    assert "ECMA-418-2:2025 Clause 6.2.8" in standards  # HMS tonality
    assert "ECMA-418-2:2025 Clause 7" in standards  # HMS roughness
    assert "ISO 532-2:2017 Clause 3.17 / Annex B.1" in standards
    assert "ISO 532-3:2023 Annex C.1" in standards


def test_building_acoustics_checks_registered() -> None:
    """The PR-B facade / lab / prediction / uncertainty checks are wired."""
    standards = {c.standard for c in cr.CHECKS}
    assert "ISO 16283-3:2016 Clause 3.12" in standards  # facade R'45
    assert "ISO 10140-2:2010 Formula (2)" in standards  # lab airborne Rw=54
    assert "EN 12354-1:2000 Annex H.3" in standards  # R'w=52 prediction
    assert "EN 12354-2:2000 Annex E.3" in standards  # L'n,w=45 impact
    assert "ISO 12999-1:2020 Table 2" in standards  # band uncertainty
    assert "ISO 12999-1:2020 Clause 8 / Table 8" in standards  # U = k u
    # Prediction + uncertainty form their own readable domain.
    assert "Building prediction & uncertainty" in cr._domains()


def test_building_reference_data_matches_published_oracles() -> None:
    """Guard the shared reference_data oracles against their standard values.

    The report reuses these constants (single source of truth), while the
    building-standard test modules keep independent inline copies; this pins the
    shared table to the published worked-example results so neither can drift.
    """
    import reference_data as ref

    assert ref.ISO16283_3_R45_EXPECTED_DB == 38.5  # 60 - 20 - 1,5
    assert ref.ISO10140_2_REF_AIRBORNE_RW == 54  # +2-shift anchor
    assert len(ref.ISO10140_2_REF_AIRBORNE_R) == 16
    assert ref.EN12354_1_ANNEX_H3_RPRIME_W == 52  # Annex H.3
    assert ref.EN12354_1_ANNEX_H3_NUM_PATHS == 13  # 1 direct + 12 flanking
    assert ref.EN12354_2_ANNEX_E3_LPRIME_N_W == 45  # 76 - 33 + 2
    assert ref.EN12354_2_ANNEX_E3_K == 2  # Table 1
    assert ref.ISO12999_1_TABLE2_AIRBORNE_A_1000HZ == 1.8  # Table 2
    assert ref.ISO12999_1_COVERAGE_K_95 == 1.96  # Table 8 (95 %, two-sided)


def test_outdoor_and_exposure_checks_registered() -> None:
    """PR-C outdoor-propagation and occupational-exposure checks are present."""
    standards = {c.standard for c in cr.CHECKS}
    assert "ISO 9613-2:1996 Eq. (7)" in standards  # Adiv = 51 dB at 100 m
    assert "ISO 9613-2:1996 Table 3" in standards  # b'(0) ground limit
    assert "ISO 9613-2:1996 clause 7.4" in standards  # 20/25 dB barrier caps
    assert "ISO 9612:2009 Annex D" in standards  # task-based LEX,8h + U
    assert "ISO 9612:2009 Annex E" in standards  # job-based LEX,8h + U
    assert "ISO 9612:2009 Annex F" in standards  # full-day LEX,8h + U

    assert "Outdoor propagation & occupational exposure" in cr._domains()


def test_outdoor_exposure_reference_data_matches_published_oracles() -> None:
    """Pin the shared ISO 9613-2 / ISO 9612 constants to their published values."""
    import reference_data as ref

    assert ref.ISO9613_2_ADIV_100M == 51.0  # 20 lg(100) + 11
    assert ref.ISO9613_2_GROUND_BPRIME_ZERO == 10.1  # Table 3 b'(0)
    assert ref.ISO9613_2_GROUND_AGR_250_POROUS == 17.2  # 2*(-1,5 + 10,1)
    assert ref.ISO9613_2_BARRIER_CAP_SINGLE == 20.0  # clause 7.4
    assert ref.ISO9613_2_BARRIER_CAP_DOUBLE == 25.0  # clause 7.4
    assert ref.ISO9612_ANNEX_D_LEX_8H == 84.3  # welder day
    assert ref.ISO9612_ANNEX_D_U == 2.7  # case (a), duration uncertainty omitted
    assert ref.ISO9612_ANNEX_E_LEX_8H == 88.1  # production line
    assert ref.ISO9612_ANNEX_E_U == 3.8
    assert ref.ISO9612_ANNEX_F_LEX_8H == 90.1  # forklift drivers
    assert ref.ISO9612_ANNEX_F_U == 3.4
    assert len(ref.ISO9612_ANNEX_D_TASKS) == 3


def test_scattering_insitu_precision_checks_registered() -> None:
    """The PR-E scattering / in-situ / precision-power checks are wired."""
    standards = {c.standard for c in cr.CHECKS}
    assert "ISO 17497-1:2004 Eqs (1)/(4)/(5)" in standards  # scattering chain
    assert "ISO 17497-2:2012 Formula (8)" in standards  # radians area factor
    assert "ISO 13472-1:2002 Annex A" in standards  # MSA radius ~1.34 m
    assert "ISO 13472-2:2010 Clause 5.4.1" in standards  # spot f_u
    assert "ISO 3745:2012 Clause 10.5 EXAMPLE" in standards  # U = 4.1 dB
    assert "ISO 9614-3:2002 Eqs (5)/(8)/(9)" in standards  # uniform-In LW
    # The three domains form their own readable report sections.
    for domain in (
        "Scattering & diffusion (ISO 17497)",
        "In-situ road absorption (ISO 13472)",
        "Precision sound power (ISO 3745 / 9614-3)",
    ):
        assert domain in cr._domains()


def test_scattering_insitu_precision_reference_data_matches_oracles() -> None:
    """Pin the shared PR-E constants to their standard worked-example values."""
    import reference_data as ref

    assert ref.ISO17497_1_SPEED_OF_SOUND_20C == 343.2  # Eq. (2) at 20 C
    assert ref.ISO17497_1_CHAIN_SCATTERING == pytest.approx(0.09306, abs=1e-5)
    assert ref.ISO17497_2_AREA_FACTOR_ZENITH == pytest.approx(1.5710, abs=1e-4)
    assert ref.ISO13472_1_KR == pytest.approx(2.0 / 3.0)  # Clause 4.2
    assert ref.ISO13472_1_MSA_RADIUS == pytest.approx(1.34, abs=5e-3)  # Annex A
    assert ref.ISO13472_2_SPOT_FU == pytest.approx(1989.4, abs=0.1)  # 0.58c/d
    assert ref.ISO3745_U_EXPANDED == pytest.approx(4.123, abs=1e-3)  # Cl. 10.5
    assert ref.ISO3745_C1_REFERENCE == pytest.approx(-0.1282, abs=1e-4)  # Eq. 16
    assert ref.ISO9614_3_UNIFORM_LW == 80.0  # 10 lg(1e-4/1e-12)


def test_filter_binding_detail_matches_library_margin() -> None:
    """The report re-derives the binding measured value and limit with the
    public ``class_limits``; guard that its class-1 margin never diverges from
    the authoritative ``verify_filter_class`` (single source of truth)."""
    from phonometry import FilterDesign, OctaveFilterBank
    from phonometry.filters.compliance import verify_filter_class

    for arch in cr._FILTER_ARCHS:
        fc = cr._filter_class(arch, 3)
        bank = OctaveFilterBank(
            48000, fraction=3, order=6, limits=[100, 10000],
            design=FilterDesign(filter_type=arch),
        )
        bands = verify_filter_class(bank)["bands"]
        lib_min = min(b["margin_class1_db"] for b in bands)
        assert fc.min_margin1 == pytest.approx(lib_min, abs=1e-9)
        # The binding measured value must sit on the reported side of the limit.
        if fc.bind_side in ("floor", "stop"):
            assert fc.bind_measured_db - fc.bind_limit_db == pytest.approx(
                lib_min, abs=1e-3
            )


def test_human_vibration_checks_registered() -> None:
    """The PR-F human-vibration checks are wired into their own domain."""
    standards = {c.standard for c in cr.CHECKS}
    assert "ISO 8041-1:2017 Table B.8" in standards  # Wk design-goal factor
    assert "ISO 8041-1:2017 Table 1" in standards  # Wh reference-freq factor
    assert "ISO 5349-2:2001 Example E.2.1" in standards  # A(8) single tool
    assert "ISO 5349-2:2001 Example E.3" in standards  # A(8) forestry
    assert "ISO 5349-1:2001 Eq. (C.1)" in standards  # VWF lifetime
    assert "Directive 2002/44/EC Art. 3" in standards  # EAV/ELV
    assert "Human vibration (ISO 8041 / 2631 / 5349)" in cr._domains()


def test_human_vibration_reference_data_matches_oracles() -> None:
    """Pin the shared PR-F constants to their published values."""
    import reference_data as ref

    assert ref.ISO8041_1_WK_FACTOR_6P31HZ == 1.054  # Table B.8, n=8
    assert ref.ISO8041_1_WM_FACTOR_1P585HZ == 0.9342  # Table B.9, n=2
    assert ref.ISO8041_1_WH_REF_FACTOR == 0.2020  # Table 1, Wh @ 500 rad/s
    assert ref.ISO5349_2_E21_A8 == 4.1  # 7,4*sqrt(2,5/8)
    assert ref.ISO5349_2_E3_A8 == 3.6  # forestry three-task
    assert ref.ISO5349_1_VWF_A8 == 7.0
    assert ref.ISO5349_1_VWF_DY_YEARS == 4.0  # Table C.1
    assert ref.DIRECTIVE_2002_44_HAV_EAV == 2.5
    assert ref.DIRECTIVE_2002_44_HAV_ELV == 5.0
    assert ref.DIRECTIVE_2002_44_WBV_EAV == 0.5
    assert ref.DIRECTIVE_2002_44_WBV_ELV == 1.15


@pytest.mark.parametrize("check", cr.CHECKS, ids=lambda c: f"{c.standard} :: {c.quantity}")
def test_every_check_passes(check: cr.Check) -> None:
    outcome = check.run()
    assert outcome.passed, (
        f"{check.standard} / {check.quantity}: expected {outcome.expected}, "
        f"computed {outcome.computed} (delta {outcome.delta})"
    )


def test_render_markdown_is_well_formed() -> None:
    markdown, passed, total = cr.render_markdown()
    assert passed == total == len(cr.CHECKS)
    assert "## Numerical conformance report" in markdown
    assert "Numerical validation - filters &amp; weightings" in markdown
    assert f"{passed}/{total} conformance checks pass" in markdown
    # Compact by default: after the headline everything lives in collapsible
    # sections (the showcase plus one per domain), each collapsed when green.
    assert markdown.count("<details>") == len(cr._domains()) + 1
    assert markdown.count("<details open>") == 0
    assert markdown.count("</details>") == len(cr._domains()) + 1
    # One table header per domain plus the two numerical-validation tables.
    assert markdown.count("|:---") >= len(cr._domains())
    # The redesigned filters table labels non-compliant architectures as
    # "By design" (not a scary "none") and shows the measured value vs limit.
    assert "By design" in markdown
    assert "none" not in markdown
    assert "Measured rel. atten." in markdown
    assert "Class-1 limit" in markdown
    # The weightings table separates the informational max deviation from the
    # binding-frequency compliance margin.
    assert "Max dev. from nominal (info)" in markdown
    assert "Binding freq" in markdown
