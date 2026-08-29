#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Domain 5 - Intensity & sound power.

Sound intensity probes and the sound power derived from them: the IEC 61043
p-p probe response and its finite-difference and phase-mismatch errors, the
ISO 9614-1/-2 field indicators and grades of accuracy, and the ISO 3741/3744
sound-power determinations with their environmental and Waterhouse
corrections.

The checks synthesize a plane wave with a known intensity and a known
propagation direction, so the expected value is a closed form rather than a
tabulated one.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, record, register
from .levels import _FS

if TYPE_CHECKING:
    from phonometry.emission.sound_power_intensity_points import (
        BandType,
        DeterminationGrade,
    )


def _plane_wave_pair(
    delay_s: float, seconds: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n = int(_FS * seconds)
    freqs = np.fft.rfftfreq(n, 1.0 / _FS)
    spec = np.zeros(freqs.size, dtype=complex)
    band = (freqs >= 50.0) & (freqs <= 2000.0)
    spec[band] = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, int(band.sum())))
    p1 = np.fft.irfft(spec, n)
    p2 = np.fft.irfft(spec * np.exp(-2j * np.pi * freqs * delay_s), n)
    scale = 1.0 / np.sqrt(np.mean(p1**2))
    return p1 * scale, p2 * scale


@register(
    "Intensity & sound power",
    "IEC 61043:1993 Clause 5",
    "Plane-wave intensity I = p^2 / (rho c)",
)
def _chk_plane_wave_intensity() -> Outcome:
    rho, c, spacing = 1.204, 343.0, 0.012
    p1, p2 = _plane_wave_pair(spacing / c)
    res = ph.emission.sound_intensity(p1, p2, _FS, spacing=spacing, rho=rho, c=c)
    expected = float(np.mean(((p1 + p2) / 2.0) ** 2)) / (rho * c)
    return numeric(
        expected, float(res.total_intensity), 0.015, unit="W/m^2", rel=True, places=5
    )


@register(
    "Intensity & sound power",
    "ISO 3744:2010 Eq. 18",
    "Monopole hemisphere recovers LW (r=4 m)",
)
def _chk_monopole_lw() -> Outcome:
    lw_true, r = 95.0, 4.0
    lp = lw_true - 10.0 * math.log10(2.0 * math.pi * r**2)
    res = ph.emission.sound_power_pressure(np.full((10, 1), lp), "hemisphere", radius=r)
    return numeric(lw_true, float(res.sound_power_level[0]), 1e-9, unit="dB", places=6)


@register(
    "Intensity & sound power",
    "ISO 9614-2:1996 Eq. 12",
    "Intensity scan recovers LW of an enclosed source",
)
def _chk_intensity_scan_lw() -> Outcome:
    w = 1.0e-3  # 90 dB re 1 pW
    areas = np.array([0.5, 0.5, 0.5, 0.5])
    intensity = np.full((4, 1), w / areas.sum())
    res = ph.emission.sound_power_intensity(intensity, areas)
    return numeric(90.0, float(res.sound_power_level[0]), 1e-6, unit="dB", places=6)


@register(
    "Intensity & sound power",
    "IEC 61043:1993 Table 2",
    "Minimum delta_pI0 per band, probe/processor/instrument, class 1/2",
)
def _chk_iec61043_table2() -> Outcome:
    """All 22 bands x 3 device kinds x 2 classes against the printed table."""
    columns = {"probe": (1, 2), "processor": (3, 4), "instrument": (5, 6)}
    worst = 0.0
    for device, (col1, col2) in columns.items():
        _, class1, class2 = ph.emission.residual_index_limits(device)
        for i, row in enumerate(ref.IEC61043_TABLE2):
            worst = max(
                worst,
                abs(float(class1[i]) - row[col1]),
                abs(float(class2[i]) - row[col2]),
            )
    n = len(ref.IEC61043_TABLE2) * len(columns) * 2
    # Table 2 is a table of exact minima, so the tolerance is zero: the check
    # asks whether every tabulated figure is reproduced, not how closely.
    return numeric(
        0.0,
        worst,
        0.0,
        unit="dB",
        places=3,
        expected_label=f"{n} tabulated minima reproduced",
        computed_label=f"max absolute deviation {worst:.3f} dB",
    )


@register(
    "Intensity & sound power",
    "IEC 61043:1993 Table 2 Note 1",
    "Separation rule +10 lg(x/25) on all six columns of 25 mm minima (x = 50 mm)",
)
def _chk_iec61043_spacing_rule() -> Outcome:
    # Note 1 applies to every figure in the table, so the check sweeps all
    # six columns (probe/processor/instrument x class 1/2) over all 22
    # bands rather than one array, so a column that failed to shift cannot
    # hide behind another that did.
    expected = 10.0 * math.log10(2.0)
    offsets: list[float] = []
    for device in ("probe", "processor", "instrument"):
        base = ph.emission.residual_index_limits(device)
        wide = ph.emission.residual_index_limits(device, spacing=0.050)
        for cls in (1, 2):
            offsets.extend((np.asarray(wide[cls]) - np.asarray(base[cls])).tolist())
    # Report the single offset furthest from the rule, so one column that
    # failed to shift cannot average out against five that did.
    worst = max(offsets, key=lambda v: abs(v - expected))
    return numeric(expected, float(worst), 1e-12, unit="dB", places=6)


@register(
    "Intensity & sound power",
    "Fahy, Sound Intensity 2e, 6.8",
    "delta_pI0 = 20 dB is a phase mismatch of 0.26 deg (1 kHz, 25 mm)",
)
def _chk_iec61043_phase_mismatch() -> Outcome:
    phi = ph.emission.phase_mismatch_from_residual_index(
        ref.IEC61043_PHASE_INDEX_DB,
        ref.IEC61043_PHASE_FREQUENCY_HZ,
        ref.IEC61043_PHASE_SPACING_M,
    )
    return numeric(
        ref.IEC61043_PHASE_MISMATCH_DEG, float(phi), 0.005, unit="deg", places=4
    )


@register(
    "Intensity & sound power",
    "ISO 9614-1:1993 Eqs (A.1)/(A.2)",
    "Temporal variability F1 is the coefficient of variation of M samples",
)
def _chk_iso9614_1_f1() -> Outcome:
    samples = np.array([1.2e-5, 0.9e-5, 1.5e-5, 1.1e-5, 1.3e-5, 1.0e-5])
    # Equations (A.1)/(A.2) written out: sample standard deviation (M - 1
    # denominator) over the algebraic mean of the M short-time samples.
    mean = float(np.sum(samples) / samples.size)
    expected = (
        math.sqrt(float(np.sum((samples - mean) ** 2)) / (samples.size - 1)) / mean
    )
    return numeric(
        expected,
        float(ph.emission.temporal_variability_indicator(samples)),
        1e-12,
        places=6,
    )


@register(
    "Intensity & sound power",
    "ISO 4871:1996 clause 3.15 / Annex B",
    "Declared L_WAd = L_WA + K_WA (Annex B, L_WA=88, K_WA=2)",
)
def _chk_iso4871_declared_value() -> Outcome:
    mode = ph.emission.OperatingModeDeclaration("Operating mode 1", 88.0, 2.0)
    return numeric(
        90.0, float(mode.declared_sound_power_level), 0.0, unit="dB", places=1
    )


@register(
    "Intensity & sound power",
    "ISO 4871:1996 clause 6.2",
    "Single-machine verification boundary L_1 <= L_WAd",
)
def _chk_iso4871_verification() -> Outcome:
    at_boundary = ph.emission.OperatingModeDeclaration(
        "m", 88.0, 2.0, verification_level=90.0
    )
    just_over = ph.emission.OperatingModeDeclaration(
        "m", 88.0, 2.0, verification_level=91.0
    )
    ok = at_boundary.verified is True and just_over.verified is False
    return Outcome(
        expected="L_1=90 verified, L_1=91 rejected (L_WAd=90)",
        computed=f"90->{at_boundary.verified}, 91->{just_over.verified}",
        delta="boundary L_1 = L_WAd",
        passed=ok,
    )


def _reverb_bracket(
    t60: np.ndarray,
    volume: float,
    surface: float,
    freq: np.ndarray,
    theta: float,
    ps: float,
) -> np.ndarray:
    """Independent re-implementation of the ISO 3741 Eq. (20) bracket.

    The two constants below are deliberately different and mirror the library
    (:func:`phonometry.emission.sound_power_reverberation._speed_of_sound` and its C1/C2
    terms): the speed of sound uses the rounded 273 of ISO 3741 clause 9.1.4
    (``c = 20,05*sqrt(273 + theta)``), while the C1/C2 barometric corrections
    use the exact absolute-zero offset 273.15 in their temperature ratios.
    Matching the library keeps the "expected" bracket and the computed value on
    the same convention.
    """
    ps0, theta0, theta1 = 101.325, 314.0, 296.0
    c = 20.05 * math.sqrt(273.0 + theta)  # ISO 3741 clause 9.1.4 (rounded 273)
    a = (55.26 / c) * (volume / t60)
    waterhouse = 10.0 * np.log10(1.0 + surface * c / (8.0 * volume * freq))
    # C1/C2 use 273.15 (absolute temperature ratios), as in the library.
    c1 = -10.0 * math.log10(ps / ps0) + 5.0 * math.log10((273.15 + theta) / theta0)
    c2 = -10.0 * math.log10(ps / ps0) + 15.0 * math.log10((273.15 + theta) / theta1)
    return 10.0 * np.log10(a) + 4.34 * (a / surface) + waterhouse + c1 + c2 - 6.0


@register(
    "Intensity & sound power",
    "ISO 3741:2010 Eq. 20",
    "Reverberation-room method inverts to a known LW",
)
def _chk_reverberation_lw() -> Outcome:
    volume, surface = 200.0, 210.0
    freqs = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
    t60 = np.array([2.0, 1.8, 1.5, 1.0, 0.6])
    theta, ps = 23.0, 101.325
    lw_target = np.array([80.0, 85.0, 90.0, 82.0, 75.0])
    lp = lw_target - _reverb_bracket(t60, volume, surface, freqs, theta, ps)
    res = ph.emission.sound_power_reverberation(
        lp, t60, volume, surface, freqs, temperature=theta, static_pressure=ps
    )
    worst = float(np.max(np.abs(np.asarray(res.sound_power_level) - lw_target)))
    return numeric(0.0, worst, 1e-9, unit="dB", places=9, expected_label="0 dB error")


def _iso9614_1_band_cells(
    table: list[tuple[float | None, float | None, float | None]],
) -> list[tuple[BandType, DeterminationGrade, float, float]]:
    """Every printed per-band cell of an ISO 9614-1 graded table.

    Yields ``(band_type, grade, frequency, value)`` for the cells the print
    fills. Grade 3 fills none of them in any of the three graded tables, so
    the blanks drop out here and are checked as refusals by the test suite
    instead.
    """
    third = (
        50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
    )  # fmt: skip
    octave = (63, 125, 250, 500, 1000, 2000, 4000)
    grades: tuple[DeterminationGrade, ...] = (
        "precision",
        "engineering",
        "survey",
    )
    columns: tuple[tuple[BandType, tuple[float, float] | None, tuple[int, ...]], ...]
    cells: list[tuple[BandType, DeterminationGrade, float, float]] = []
    for row, values in zip(ref.ISO9614_1_BAND_ROWS, table, strict=True):
        columns = (("octave", row[0], octave), ("third", row[1], third))
        for band_type, span, series in columns:
            if span is None:
                continue
            for frequency in (f for f in series if span[0] <= f <= span[1]):
                for grade, value in zip(grades, values, strict=True):
                    if value is not None:
                        cells.append((band_type, grade, float(frequency), value))
    return cells


@register(
    "Intensity & sound power",
    "ISO 9614-1:1993 Table B.2",
    "Criterion-2 factor C per band and grade, and the A-weighted grade-3 value",
)
def _chk_iso9614_1_table_b2() -> Outcome:
    """Every printed cell of Table B.2, in both frequency columns."""
    cells = _iso9614_1_band_cells(ref.ISO9614_1_TABLE_B2_C)
    worst = 0.0
    for band_type, grade, frequency, value in cells:
        computed = ph.emission.position_count_factor(
            grade, frequency, band_type=band_type
        )
        worst = max(worst, abs(computed - value))
    a_weighted = ref.ISO9614_1_TABLE_B2_C_A_WEIGHTED[2]
    worst = max(worst, abs(ph.emission.position_count_factor("survey") - a_weighted))
    # An exact printed table, so the tolerance is zero: the question is whether
    # every cell is reproduced, not how closely.
    return numeric(
        0.0,
        worst,
        0.0,
        places=3,
        expected_label=f"{len(cells) + 1} tabulated values of C reproduced",
        computed_label=f"max absolute deviation {worst:.3f}",
    )


@register(
    "Intensity & sound power",
    "ISO 9614-1:1993 Table 2",
    "Standard deviation s of the determination per band and grade",
)
def _chk_iso9614_1_table_2() -> Outcome:
    """Every printed cell of Table 2, plus its A-weighted grade-3 value."""
    cells = _iso9614_1_band_cells(ref.ISO9614_1_TABLE_2_S)
    worst = 0.0
    for band_type, grade, frequency, value in cells:
        computed = ph.emission.determination_standard_deviation(
            grade, frequency, band_type=band_type
        )
        worst = max(worst, abs(computed - value))
    a_weighted = ref.ISO9614_1_TABLE_2_S_A_WEIGHTED[2]
    worst = max(
        worst,
        abs(ph.emission.determination_standard_deviation("survey") - a_weighted),
    )
    return numeric(
        0.0,
        worst,
        0.0,
        unit="dB",
        places=3,
        expected_label=f"{len(cells) + 1} tabulated values of s reproduced",
        computed_label=f"max absolute deviation {worst:.3f} dB",
    )


@register(
    "Intensity & sound power",
    "ISO 9614-1:1993 Table B.1",
    "Error factor Delta: 0,20 and 0,29 for all bands, 0,60 A-weighted",
)
def _chk_iso9614_1_table_b1() -> Outcome:
    """The three cells Table B.1 fills, and only those three."""
    expected = {
        "precision (all bands)": ref.ISO9614_1_TABLE_B1_ALL_BANDS[0],
        "engineering (all bands)": ref.ISO9614_1_TABLE_B1_ALL_BANDS[1],
        "survey (A-weighted)": ref.ISO9614_1_TABLE_B1_A_WEIGHTED[2],
    }
    computed = {
        "precision (all bands)": ph.emission.error_factor("precision"),
        "engineering (all bands)": ph.emission.error_factor("engineering"),
        "survey (A-weighted)": ph.emission.error_factor("survey", a_weighted=True),
    }
    return record({k: float(v) for k, v in expected.items()}, computed)


@register(
    "Intensity & sound power",
    "ISO 9614-1:1993 Eq. (12)",
    "Discrete positions tiling a scanned surface give the same LW",
)
def _chk_iso9614_1_power_sum() -> Outcome:
    """The Part 1 sum over positions and the Part 2 sum over segments agree.

    Both are ``sum(In,i * Si)``, so a set of discrete positions standing for
    the segments of an ISO 9614-2 scan, with the same intensities, must give
    the level the scan gives. The two are separate implementations, and this
    is the equality that keeps them one determination.
    """
    areas = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 0.5, 0.75, 1.0, 1.25, 1.5])
    rng = np.random.default_rng(96141)
    intensity = 1.0e-5 * (1.0 + rng.normal(0.0, 0.2, (areas.size, 4)))
    frequencies = np.array([250.0, 500.0, 1000.0, 2000.0])
    with warnings.catch_warnings():
        # Neither determination is qualified here, and both say so: the check
        # is about the power sum alone, so the field indicators and the
        # residual index that would drive their A-weighting screening are not
        # supplied.
        warnings.simplefilter("ignore", ph.emission.SoundPowerWarning)
        points = ph.emission.sound_power_intensity_points(
            intensity, areas, frequencies=frequencies, band_type="octave"
        )
        scan = ph.emission.sound_power_intensity(
            intensity, areas, frequencies=frequencies, band_type="octave"
        )
    worst = float(
        np.max(np.abs(np.asarray(points.sound_power_level) - scan.sound_power_level))
    )
    return numeric(0.0, worst, 1e-12, unit="dB", places=12, expected_label="0 dB error")


@register(
    "Intensity & sound power",
    "ISO 9614-1:1993 Table B.3",
    "Five action codes, each reached by the case Figure B.1 routes to it",
)
def _chk_iso9614_1_table_b3() -> Outcome:
    """The action codes a determination hands back, case by case.

    Table B.3 is the part of Annex B that says what to *change*, so the check
    drives one determination per row of the table and reads the codes back.
    """
    areas = np.full(10, 1.0)
    outward = np.full(10, 1.0e-5)
    inward = np.full(10, 1.0e-5)
    inward[0] = -3.2e-5  # (F3 - F2) just past the 3 dB gate of Figure B.1
    moderate = np.full(10, 1.0e-5)
    moderate[0] = -2.04e-5  # (F3 - F2) inside the 1 dB to 3 dB band of row 3
    concentrated = np.array(
        [4e-5, 3e-5, 2e-5, 1e-5, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6]
    )
    wandering = np.array([1.0e-5, 5.0e-5, 2.0e-6, 9.0e-5, 1.0e-6])

    def codes(
        intensity: np.ndarray,
        residual_index: float = 30.0,
        temporal: np.ndarray | None = None,
    ) -> tuple[str, ...]:
        result = ph.emission.sound_power_intensity_points(
            intensity,
            areas,
            pressure_levels=np.full(10, 80.0),
            pressure_residual_index=residual_index,
            temporal_intensity=temporal,
            frequencies=[1000.0],
            grade="precision",
        )
        return tuple(action.value for action in result.required_actions()[0])

    reached = {
        "F1 > 0,6": codes(outward, temporal=wandering),
        "F2 > Ld": codes(outward, residual_index=10.0),
        "(F3 - F2) > 3 dB": codes(inward),
        "criterion 2, 1 dB <= (F3 - F2) <= 3 dB": codes(moderate),
        "criterion 2, (F3 - F2) <= 1 dB": codes(concentrated),
    }
    expected = {
        "F1 > 0,6": ("e",),
        "F2 > Ld": ("a", "b"),
        "(F3 - F2) > 3 dB": ("a", "b"),
        "criterion 2, 1 dB <= (F3 - F2) <= 3 dB": ("c",),
        "criterion 2, (F3 - F2) <= 1 dB": ("d",),
    }
    wrong = [row for row, got in reached.items() if got != expected[row]]
    return Outcome(
        expected="; ".join(f"{row} -> {''.join(c)}" for row, c in expected.items()),
        computed="; ".join(f"{row} -> {''.join(c)}" for row, c in reached.items()),
        delta=f"{len(wrong)} of {len(expected)} rows disagree",
        passed=not wrong,
    )


@register(
    "Intensity & sound power",
    "ISO 9614-1:1993 Eq. (B.4)",
    "New positions N* on the concentrated subset of the measurement surface",
)
def _chk_iso9614_1_additional_positions() -> Outcome:
    """B.1.3 written out beside the library's own reading of it.

    The remainder falls away steadily rather than sitting flat, so that
    F4(1 - alpha) is not zero and the whole of Delta_alpha is exercised: a
    uniform remainder cancels the term the standard spends its error budget
    on, and N* would come out of Delta/alpha alone.
    """
    areas = np.full(12, 1.0)
    intensity = np.array(
        [3.0e-5, 2.4e-5, 1.8e-5, 1.4e-5]
        + [1.2e-5, 1.0e-5, 9.0e-6, 8.0e-6, 7.0e-6, 6.0e-6, 5.0e-6, 4.0e-6]
    )
    outcome = ph.emission.partial_power_concentration(
        intensity, areas, grade="engineering"
    )
    subset, remainder = intensity[:4], intensity[4:]
    alpha = float(np.sum(subset)) / float(np.sum(intensity))
    f4_subset = float(np.std(subset, ddof=1) / np.mean(subset))
    f4_remainder = float(np.std(remainder, ddof=1) / np.mean(remainder))
    delta = ref.ISO9614_1_TABLE_B1_ALL_BANDS[1]
    delta_alpha = (
        delta - (1.0 - alpha) * (2.0 / math.sqrt(remainder.size)) * f4_remainder
    ) / alpha
    expected = math.ceil(4.0 * (f4_subset / delta_alpha) ** 2)
    return numeric(
        float(expected),
        float(outcome.additional_positions),
        0.0,
        places=0,
        expected_label=f"N* = {expected} positions",
        computed_label=f"N* = {outcome.additional_positions} positions",
    )
