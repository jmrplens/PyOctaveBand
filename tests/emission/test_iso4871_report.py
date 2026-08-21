#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the ISO 4871:1996 noise-emission declaration and its ``.report()``
fiche (declaration model + PDF rendering).

The declaration model is checked against ISO 4871's own definitions and Annex B
example: the declared single-number value is the sum ``L_WAd = L_WA + K_WA``
rounded once to the nearest decibel (clause 3.15, not the sum of the separately
rounded dual-number values of 3.16), the dual/single forms are the same
declaration, and verification passes or fails at the clause 6.2 boundary
``L_1 <= L_WAd``. The rendering itself is a
feature, so those tests assert only structural facts: a valid one-page PDF,
translated Spanish output, and rejected engines/languages.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
from report_assertions import assert_one_page

from phonometry import ReportMetadata, emission


def _annex_b_modes() -> tuple[
    emission.OperatingModeDeclaration, emission.OperatingModeDeclaration
]:
    """The ISO 4871:1996 Annex B.2 dual-number example (Type 990, Model 11-TC)."""
    mode1 = emission.OperatingModeDeclaration(
        "Operating mode 1",
        88.0,
        2.0,
        emission_pressure_level=78.0,
        emission_pressure_uncertainty=2.0,
    )
    mode2 = emission.OperatingModeDeclaration(
        "Operating mode 2",
        95.0,
        2.0,
        emission_pressure_level=86.0,
        emission_pressure_uncertainty=2.0,
    )
    return mode1, mode2


def _annex_b_declaration(**kwargs) -> emission.NoiseEmissionDeclaration:
    return emission.NoiseEmissionDeclaration(
        _annex_b_modes(),
        machine="Type 990, Model 11-TC",
        operating_conditions="50 Hz, 230 V, rated load",
        basic_standards="ISO 3744",
        **kwargs,
    )


def _extract_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)


# --- ISO 4871 definitions (clause 3.15/3.16) against Annex B -----------------


def test_declared_value_is_measured_plus_uncertainty() -> None:
    """L_WAd = L_WA + K_WA rounded to the nearest decibel (clause 3.15/3.16).

    Annex B.2 (L_WA = 88, K_WA = 2) gives the Annex B.1 single-number L_WAd = 90;
    the second mode gives 97. The emission pressure levels give 80 and 88.
    """
    mode1, mode2 = _annex_b_modes()
    assert mode1.declared_sound_power_level == 90
    assert mode2.declared_sound_power_level == 97
    assert mode1.declared_emission_pressure_level == 80
    assert mode2.declared_emission_pressure_level == 88


def test_declared_value_rounds_to_nearest_decibel() -> None:
    """A non-integer measurement + uncertainty is rounded to the nearest dB."""
    mode = emission.OperatingModeDeclaration("m", 87.6, 2.4)
    # round(87.6 + 2.4) = round(90.0) = 90.
    assert mode.declared_sound_power_level == 90


def test_declared_value_rounds_the_sum_not_the_addends() -> None:
    """Clause 3.15 rounds the sum L + K once, not L and K separately.

    L_WA = 91.4, K_WA = 2.4: the sum is 93.8, so L_WAd = 94; rounding the
    addends first would give 91 + 2 = 93, one decibel low. The same rule
    applies to the declared emission sound pressure level.
    """
    mode = emission.OperatingModeDeclaration(
        "m",
        91.4,
        2.4,
        emission_pressure_level=81.4,
        emission_pressure_uncertainty=2.4,
    )
    assert mode.declared_sound_power_level == 94
    assert mode.declared_emission_pressure_level == 84


def test_declared_value_ties_round_half_up() -> None:
    """A sum landing exactly on a half decibel rounds up (halves-up rule)."""
    mode = emission.OperatingModeDeclaration("m", 92.5, 2.0)
    # round(94.5) = 95 with the halves-up rule.
    assert mode.declared_sound_power_level == 95


def test_verification_passes_and_fails_at_the_clause_6_2_boundary() -> None:
    """Clause 6.2: verified iff L_1 <= L_WAd (boundary L_1 == L_WAd passes)."""
    at_boundary = emission.OperatingModeDeclaration(
        "m", 88.0, 2.0, verification_level=90.0
    )
    just_over = emission.OperatingModeDeclaration(
        "m", 88.0, 2.0, verification_level=91.0
    )
    under = emission.OperatingModeDeclaration("m", 88.0, 2.0, verification_level=87.0)
    assert at_boundary.verified is True
    assert just_over.verified is False
    assert under.verified is True


def test_no_verification_measurement_yields_none() -> None:
    """Without a verification measurement the verdict is undefined (None)."""
    mode1, _ = _annex_b_modes()
    assert mode1.verified is None
    assert mode1.verified_dual is None


def test_dual_number_verification_uses_separately_rounded_values() -> None:
    """Clause 6.2 verifies the dual-number form against (L + K), the sum of
    the separately rounded declared values (clause 3.16), not round(L + K).

    L_WA = 93,4, K_WA = 2,4: the dual declaration states 93 dB and 2 dB, so
    the verification limit is 95 dB, while the single-number L_WAd is
    round(95,8) = 96 dB.
    """
    verified = emission.OperatingModeDeclaration(
        "m", 93.4, 2.4, verification_level=95.0
    )
    rejected = emission.OperatingModeDeclaration(
        "m", 93.4, 2.4, verification_level=95.5
    )
    assert verified.dual_number_verification_limit == 95
    assert verified.declared_sound_power_level == 96
    assert verified.verified_dual is True
    assert rejected.verified_dual is False
    # The combined (single-number) form still verifies against L_WAd = 96.
    assert rejected.verified is True
    at_single_boundary = emission.OperatingModeDeclaration(
        "m", 93.4, 2.4, verification_level=96.0
    )
    assert at_single_boundary.verified is True
    assert at_single_boundary.verified_dual is False


# --- model validation --------------------------------------------------------


def test_emission_pressure_pair_must_be_given_together() -> None:
    """A lone emission-pressure level (no uncertainty) is rejected."""
    with pytest.raises(ValueError, match="given together"):
        emission.OperatingModeDeclaration("m", 88.0, 2.0, emission_pressure_level=78.0)


def test_negative_uncertainty_is_rejected() -> None:
    """The uncertainty K must be finite and non-negative."""
    with pytest.raises(ValueError, match="non-negative"):
        emission.OperatingModeDeclaration("m", 88.0, -1.0)


def test_non_finite_level_is_rejected() -> None:
    """A non-finite sound power level is rejected."""
    with pytest.raises(ValueError, match="finite"):
        emission.OperatingModeDeclaration("m", math.nan, 2.0)


def test_declaration_requires_at_least_one_mode() -> None:
    """A declaration with no operating mode is rejected."""
    with pytest.raises(ValueError, match="at least one operating mode"):
        emission.NoiseEmissionDeclaration(())


def test_unknown_form_is_rejected() -> None:
    """An unknown declaration form is rejected."""
    modes = _annex_b_modes()
    with pytest.raises(ValueError, match="dual-number"):
        emission.NoiseEmissionDeclaration(modes, form="triple")  # type: ignore[arg-type]


def test_basic_standards_string_is_wrapped() -> None:
    """A single basic-standard string is stored as a one-tuple."""
    decl = emission.NoiseEmissionDeclaration(
        _annex_b_modes(), basic_standards="ISO 3744"
    )
    assert decl.basic_standards == ("ISO 3744",)


def test_declaration_is_picklable() -> None:
    """The frozen declaration round-trips through pickle."""
    decl = _annex_b_declaration()
    assert pickle.loads(pickle.dumps(decl)).modes[0].declared_sound_power_level == 90


# --- SoundPowerResult.declare bridge ----------------------------------------


def test_declare_from_sound_power_result() -> None:
    """SoundPowerResult.declare wraps LWA as L_WA with the result's uncertainty."""
    # Monopole hemisphere at r = 1 m: Lp = LW - 10 lg(2 pi r^2); recover LW.
    r = 1.0
    lw = 90.0
    lp = lw - 10.0 * math.log10(2.0 * math.pi * r**2)
    result = emission.sound_power_pressure(np.full((10, 1), lp), "hemisphere", radius=r)
    decl = result.declare(uncertainty=2.0, machine="Pump X", basic_standards="ISO 3744")
    mode = decl.modes[0]
    assert mode.sound_power_level == pytest.approx(lw, abs=1e-6)
    assert mode.sound_power_uncertainty == 2.0
    assert mode.declared_sound_power_level == 92  # 90 + 2
    # The default K is the result's own expanded uncertainty.
    assert result.declare().modes[0].sound_power_uncertainty == pytest.approx(
        result.uncertainty
    )


def test_declare_requires_finite_lwa() -> None:
    """declare() needs a finite A-weighted sound power level."""
    # Several bands without frequencies leave LWA undefined (NaN).
    result = emission.sound_power_pressure(
        np.full((10, 3), 70.0), "hemisphere", radius=1.0
    )
    assert not math.isfinite(result.sound_power_level_a)
    with pytest.raises(ValueError, match="finite A-weighted"):
        result.declare()


# --- rendering ---------------------------------------------------------------


def test_dual_number_report_renders_one_page(tmp_path) -> None:
    """A dual-number declaration renders a valid one-page fiche."""
    pytest.importorskip("reportlab")
    decl = _annex_b_declaration(noise_test_code="ISO 3746 test code")
    out = tmp_path / "iso4871.pdf"
    returned = decl.report(str(out))
    assert returned == str(out)
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "DECLARED DUAL-NUMBER" in text
    assert "Operating mode 1" in text
    assert "ISO 3744" in text


def test_single_number_report_renders_one_page(tmp_path) -> None:
    """A single-number declaration renders its L_WAd table."""
    pytest.importorskip("reportlab")
    decl = _annex_b_declaration(form="single-number")
    out = tmp_path / "iso4871_single.pdf"
    decl.report(str(out))
    assert_one_page(str(out))
    assert "DECLARED SINGLE-NUMBER" in _extract_text(str(out))


def test_dual_number_report_follows_annex_b2_layout(tmp_path) -> None:
    """The dual-number table states only L and K (Annex B.2): no derived
    L_WAd row, whose mix of separately rounded addends and a once-rounded sum
    is not part of the dual layout.
    """
    pytest.importorskip("reportlab")
    decl = _annex_b_declaration()
    out = tmp_path / "iso4871_dual_layout.pdf"
    decl.report(str(out))
    text = _extract_text(str(out))
    assert "Measured A-weighted sound power level" in text
    assert "Uncertainty" in text
    assert "Declared A-weighted sound power level" not in text


def test_dual_number_verification_row_uses_rounded_sum(tmp_path) -> None:
    """A dual-number fiche verifies against L_WA + K_WA of the separately
    rounded declared values (93 + 2 = 95 for 93,4/2,4), not round(95,8) = 96.
    """
    pytest.importorskip("reportlab")
    mode = emission.OperatingModeDeclaration(
        "Operating mode 1", 93.4, 2.4, verification_level=95.0
    )
    decl = emission.NoiseEmissionDeclaration((mode,), basic_standards="ISO 3744")
    out = tmp_path / "iso4871_dual_verify.pdf"
    decl.report(str(out))
    text = _extract_text(str(out))
    assert "Verification" in text
    assert "95 dB" in text
    assert "96 dB" not in text
    assert "PASS" in text
    # The same declaration in single-number form verifies against L_WAd = 96.
    single = emission.NoiseEmissionDeclaration(
        (mode,), basic_standards="ISO 3744", form="single-number"
    )
    out2 = tmp_path / "iso4871_single_verify.pdf"
    single.report(str(out2))
    text2 = _extract_text(str(out2))
    assert "96 dB" in text2
    assert "PASS" in text2


def test_verification_verdict_renders_both_ways(tmp_path) -> None:
    """A passing and a failing verification both render in the verdict table."""
    pytest.importorskip("reportlab")
    mode1 = emission.OperatingModeDeclaration(
        "Operating mode 1", 88.0, 2.0, verification_level=89.0
    )
    mode2 = emission.OperatingModeDeclaration(
        "Operating mode 2", 95.0, 2.0, verification_level=98.0
    )
    decl = emission.NoiseEmissionDeclaration((mode1, mode2), basic_standards="ISO 3744")
    out = tmp_path / "iso4871_verify.pdf"
    decl.report(str(out), metadata=ReportMetadata(report_id="PHN-4871"))
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Verification" in text
    assert "PASS" in text
    assert "FAIL" in text


def test_spanish_report_renders_translated_fiche(tmp_path) -> None:
    """language="es" renders a one-page Spanish fiche."""
    pytest.importorskip("reportlab")
    decl = _annex_b_declaration()
    out = tmp_path / "iso4871_es.pdf"
    decl.report(str(out), language="es")
    assert_one_page(str(out))
    text = _extract_text(str(out))
    assert "Declaración de emisión sonora" in text
    assert "DOBLE NÚMERO" in text


def test_unknown_engine_rejected(tmp_path) -> None:
    """An unknown rendering engine raises ValueError."""
    pytest.importorskip("reportlab")
    decl = _annex_b_declaration()
    out = str(tmp_path / "x.pdf")
    with pytest.raises(ValueError, match="engine"):
        decl.report(out, engine="weasyprint")


def test_unknown_language_rejected(tmp_path) -> None:
    """An unknown fiche language raises ValueError."""
    decl = _annex_b_declaration()
    with pytest.raises(ValueError, match="language"):
        decl.report(str(tmp_path / "bad.pdf"), language="xx")


# --------------------------------------------------------------------------
# More modes than the sheet has width for
# --------------------------------------------------------------------------
def test_more_modes_than_the_sheet_can_print_are_refused(tmp_path) -> None:
    """The mode columns divide the page width, with nothing bounding them.

    A declaration is free to carry as many operating modes as the machine
    has; the sheet is not. Seven of them leave each column narrower than the
    three-digit level it holds, and the table starts stacking digits rather
    than saying so.
    """
    modes = tuple(
        emission.OperatingModeDeclaration(f"Mode {i}", 88.0, 2.0) for i in range(7)
    )
    declaration = emission.NoiseEmissionDeclaration(
        modes, machine="Many-mode machine", basic_standards="ISO 3744"
    )
    out = tmp_path / "too_many_modes.pdf"
    with pytest.raises(ValueError, match="does not fit the sheet"):
        declaration.report(str(out))


def test_six_modes_still_print(tmp_path) -> None:
    """Six is the widest that still holds a three-digit level per column."""
    modes = tuple(
        emission.OperatingModeDeclaration(f"Mode {i}", 88.0, 2.0) for i in range(6)
    )
    declaration = emission.NoiseEmissionDeclaration(
        modes, machine="Six-mode machine", basic_standards="ISO 3744"
    )
    assert_one_page(declaration.report(str(tmp_path / "six_modes.pdf")))
