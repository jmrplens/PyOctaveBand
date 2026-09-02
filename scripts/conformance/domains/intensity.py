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


@register(
    "Intensity & sound power",
    "ISO 3744:2010 Eq. 23 / clause 3.4 NOTE 1",
    "Sound energy level of a source steady over T = 10 s is LW + 10 lg(T/T0)",
)
def _chk_sound_energy_identity_iso3744() -> Outcome:
    """The 8.3 chain is the 8.2 chain with L_E for L_p, field by field.

    For p^2 constant over T the integral of clause 3.4 is T p^2, so every
    position level is L_p + 10 lg(T/T0) (NOTE 1), and with the background
    compared as its own exposure over the same T the corrections K1 and K2
    coincide and L_J = L_W + 10 lg(T/T0) exactly. No worked example with L_J
    is printed in the standard; this identity is the closed-form anchor.
    """
    rng = np.random.default_rng(3744)
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    lp = np.array([70.0, 74.0, 78.0, 75.0]) + rng.normal(0.0, 0.5, (10, 4))
    bg = np.array([58.0, 60.0, 62.0, 61.0])
    room = ph.emission.RoomEnvironment(absorption_area=150.0)
    lw = ph.emission.sound_power_pressure(
        lp, "hemisphere", radius=2.0, background_levels=bg, frequencies=freqs, room=room
    )
    lj = ph.emission.sound_energy_pressure(
        lp + 10.0,
        "hemisphere",
        radius=2.0,
        background_levels=bg,
        integration_time=10.0,
        frequencies=freqs,
        room=room,
    )
    worst = float(
        np.max(np.abs(np.asarray(lj.sound_energy_level) - lw.sound_power_level - 10.0))
    )
    return numeric(
        0.0,
        worst,
        1e-9,
        unit="dB",
        places=9,
        expected_label="LJ - LW = 10 dB, 0 dB error",
    )


@register(
    "Intensity & sound power",
    "ISO 3744:2010 Eq. 20",
    "One measurement encompassing Ne = 5 events is 10 lg 5 above one event",
)
def _chk_sound_energy_events_eq20() -> Outcome:
    level = ph.emission.mean_single_event_level(np.array([90.0]), events=5)
    return numeric(
        10.0 * math.log10(5.0), 90.0 - float(level[0]), 1e-12, unit="dB", places=6
    )


@register(
    "Intensity & sound power",
    "ISO 3741:2010 Eq. 30",
    "Reverberation-room sound energy level inverts to a known LJ",
)
def _chk_reverberation_lj() -> Outcome:
    volume, surface = 200.0, 210.0
    freqs = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
    t60 = np.array([2.0, 1.8, 1.5, 1.0, 0.6])
    theta, ps = 23.0, 101.325
    lj_target = np.array([90.0, 95.0, 100.0, 92.0, 85.0])
    le = lj_target - _reverb_bracket(t60, volume, surface, freqs, theta, ps)
    res = ph.emission.sound_energy_reverberation(
        le, t60, volume, surface, freqs, temperature=theta, static_pressure=ps
    )
    worst = float(np.max(np.abs(np.asarray(res.sound_energy_level) - lj_target)))
    return numeric(0.0, worst, 1e-9, unit="dB", places=9, expected_label="0 dB error")


@register(
    "Intensity & sound power",
    "ISO 3741:2010 Eq. F.4",
    "Three equal one-third-octave bands sum to an octave level 10 lg 3 higher",
)
def _chk_octave_band_levels_annex_f() -> Outcome:
    _, octave = ph.emission.octave_band_levels(
        np.full(3, 90.0), np.array([800.0, 1000.0, 1250.0])
    )
    return numeric(
        10.0 * math.log10(3.0), float(octave[0]) - 90.0, 1e-12, unit="dB", places=6
    )


@register(
    "Intensity & sound power",
    "ISO 3744:2010 Annex G / H.4.2.7",
    "C1 + C2 of Eq. (G.1)/(G.3) vanish at 120 m altitude and 23 C",
)
def _chk_annex_g_zero_at_120m() -> Outcome:
    """H.4.2.7: 'At 120 m altitude and 23 C the correction is zero.'

    Eq. (G.2) gives p_s(120 m) = 99,89 kPa, -10 lg(p_s/p_s0) = 0,062 dB in
    each term, and the temperature terms 5 lg(296,15/314) + 15 lg(296,15/296)
    = -0,124 dB cancel the pair to under 1e-4 dB.
    """
    corr = ph.emission.reference_atmosphere_correction(23.0, altitude=120.0)
    return numeric(0.0, float(corr.total), 1e-3, unit="dB", places=5)


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
    # B.1.3's own ranking, transcribed rather than sliced: partial powers in
    # decreasing order, kept until more than half the total is accounted for.
    # On these equal areas the partial powers order as the intensities do, and
    # deriving the split here keeps the check honest if the list above is ever
    # reordered: a hard [:4] would then verify a subset B.1.3 never names.
    order = np.argsort(intensity * areas)[::-1]
    cumulative = np.cumsum((intensity * areas)[order])
    count = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="right")) + 1
    subset = intensity[order[:count]]
    remainder = intensity[order[count:]]
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


# --- ISO 3747:2010: in situ comparison with a reference sound source ---------
# The standard prints one worked number (the 9.5 EXAMPLE) and three tables the
# library reads or reproduces (Table 2, Table D.1, Table E.1); everything else
# is closed form, anchored here to the printed equations.

_ISO3747_FREQS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
_ISO3747_LW_RSS = np.array([87.0, 90.5, 92.5, 93.8, 94.0, 93.0, 90.0])
_ISO3747_ST = np.array(
    [
        [80.1, 83.4, 85.0, 84.2, 81.0, 76.5, 70.2],
        [79.0, 82.8, 84.6, 83.9, 80.4, 75.8, 69.5],
        [81.2, 84.0, 85.9, 85.0, 81.9, 77.1, 70.9],
        [80.5, 83.1, 85.3, 84.5, 81.3, 76.2, 70.0],
    ]
)
_ISO3747_RSS = np.array(
    [
        [78.5, 81.9, 83.7, 84.9, 84.8, 83.5, 79.8],
        [77.9, 81.2, 83.1, 84.3, 84.1, 82.9, 79.2],
        [79.3, 82.6, 84.4, 85.5, 85.4, 84.1, 80.3],
        [78.8, 82.1, 83.9, 85.0, 85.0, 83.7, 79.9],
    ]
)
#: The four excesses and the directivity range that earn grade 2 (Table 2).
_ISO3747_EXCESS_GRADE2 = [8.0, 9.5, 7.2, 8.8]
_ISO3747_DIRECTIVITY_GRADE2 = 3.0


@register(
    "Intensity & sound power",
    "ISO 3747:2010 9.5 EXAMPLE",
    "Expanded uncertainty U = 2 sqrt(1,5^2 + 2^2) dB, grade 2 with sigma_omc = 2,0 dB",
)
def _chk_iso3747_example_uncertainty() -> Outcome:
    res = ph.emission.sound_power_in_situ(
        _ISO3747_ST, _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS,
        sigma_omc=2.0,
        conditions=ph.emission.GradeConditions(excess_levels=_ISO3747_EXCESS_GRADE2, directivity_range=_ISO3747_DIRECTIVITY_GRADE2),
    )  # fmt: skip
    return numeric(5.0, float(res.expanded_uncertainty), 1e-12, unit="dB", places=6)


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Table 2 / Eq. 22",
    "sigma_R0 by grade: 1,5 dB (grade 2) and 4,0 dB (grade 3), sigma_tot of Table E.1 row 2",
)
def _chk_iso3747_table2_and_e1() -> Outcome:
    grade2 = ph.emission.sound_power_in_situ(
        _ISO3747_ST, _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS,
        sigma_omc=4.0,
        conditions=ph.emission.GradeConditions(excess_levels=_ISO3747_EXCESS_GRADE2, directivity_range=_ISO3747_DIRECTIVITY_GRADE2),
    )  # fmt: skip
    grade3 = ph.emission.sound_power_in_situ(
        _ISO3747_ST, _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS,
        conditions=ph.emission.GradeConditions(excess_levels=[8.0, 6.9, 7.2, 8.8], directivity_range=3.0)
    )  # fmt: skip
    # Table E.1, sigma_R0 = 1,5 dB row at sigma_omc = 4 dB: sqrt(18,25) = 4,27 -> 4,3.
    return record(
        {"sigma_R0 grade 2": 1.5, "sigma_R0 grade 3": 4.0, "sigma_tot (1,5; 4)": 4.3},
        {
            "sigma_R0 grade 2": float(grade2.sigma_r0),
            "sigma_R0 grade 3": float(grade3.sigma_r0),
            "sigma_tot (1,5; 4)": round(float(grade2.sigma_tot), 1),
        },
        unit="dB",
    )


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Eq. 7 / 8.1",
    "K1 at the 6 dB validity margin, -10 lg(1 - 10^-0,6) = 1,2563 dB, and the 1,3 dB cap below it",
)
def _chk_iso3747_k1_margin_and_cap() -> Outcome:
    background = np.full_like(_ISO3747_ST, 40.0)
    background[1, 3] = _ISO3747_ST[1, 3] - 6.0  # exactly at the validity margin
    background[2, 5] = _ISO3747_ST[2, 5] - 2.0  # far below it: the cap
    res = ph.emission.sound_power_in_situ(
        _ISO3747_ST, _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS,
        background_levels=background,
    )  # fmt: skip
    expected_edge = -10.0 * math.log10(1.0 - 10.0**-0.6)
    worst = max(
        abs(float(res.background_correction[1, 3]) - expected_edge),
        abs(float(res.background_correction[2, 5]) - 1.3),
    )
    flagged = not bool(res.background_requirement_met[5]) and bool(
        res.background_requirement_met[3]
    )
    return numeric(
        0.0,
        worst if flagged else float("inf"),
        1e-9,
        unit="dB",
        places=9,
        expected_label="K1(6 dB) = 1,2563 dB, K1(2 dB) = 1,3 dB, 4 kHz flagged",
    )


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Eq. 11 vs ISO 3741:2010 Eq. 21",
    "In situ comparison plus C2 equals the reverberation-room comparison (closed form)",
)
def _chk_iso3747_eq11_vs_iso3741() -> Outcome:
    with warnings.catch_warnings():  # four positions trip the ISO 3741 advisory
        warnings.simplefilter("ignore", ph.emission.SoundPowerWarning)
        room = ph.emission.sound_power_comparison(
            _ISO3747_ST, _ISO3747_RSS, _ISO3747_LW_RSS,
            frequencies=_ISO3747_FREQS, temperature=20.0, static_pressure=100.0,
        )  # fmt: skip
    res = ph.emission.sound_power_in_situ(
        _ISO3747_ST, _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS,
        temperature=20.0, static_pressure=100.0,
    )  # fmt: skip
    worst = float(np.max(np.abs(res.sound_power_level_ref - room.sound_power_level)))
    return numeric(
        0.0, worst, 1e-9, unit="dB", places=9, expected_label="0 dB difference"
    )


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Eq. 12 / Eq. 20",
    "m identical reference-source locations collapse to Eq. 11 / Eq. 19 (closed form)",
)
def _chk_iso3747_locations_collapse() -> Outcome:
    one = ph.emission.sound_power_in_situ(
        _ISO3747_ST, _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS
    )
    three = ph.emission.sound_power_in_situ(
        _ISO3747_ST,
        np.stack([_ISO3747_RSS] * 3),
        np.stack([_ISO3747_LW_RSS] * 3),
        _ISO3747_FREQS,
    )
    events = np.repeat(_ISO3747_ST[:, None, :], 5, axis=1)
    one_j = ph.emission.sound_energy_in_situ(
        events, _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS
    )
    two_j = ph.emission.sound_energy_in_situ(
        events, np.stack([_ISO3747_RSS] * 2), _ISO3747_LW_RSS, _ISO3747_FREQS
    )
    worst = max(
        float(np.max(np.abs(three.sound_power_level - one.sound_power_level))),
        float(np.max(np.abs(two_j.sound_energy_level - one_j.sound_energy_level))),
    )
    return numeric(
        0.0, worst, 1e-9, unit="dB", places=9, expected_label="0 dB difference"
    )


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Eq. 15 / Eq. 17",
    "N events one at a time and one measurement over N events agree (closed form)",
)
def _chk_iso3747_event_forms() -> Outcome:
    n_events = 8
    one_at_a_time = ph.emission.sound_energy_in_situ(
        np.repeat(_ISO3747_ST[:, None, :], n_events, axis=1),
        _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS,
    )  # fmt: skip
    encompassing = ph.emission.sound_energy_in_situ(
        _ISO3747_ST + 10.0 * math.log10(n_events),
        _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS, events=n_events,
    )  # fmt: skip
    worst = float(
        np.max(
            np.abs(encompassing.sound_energy_level - one_at_a_time.sound_energy_level)
        )
    )
    return numeric(
        0.0, worst, 1e-9, unit="dB", places=9, expected_label="0 dB difference"
    )


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Annex C",
    "C2 at 101,325 kPa and 23,0 degC is 15 lg(296,15/296) = 0,003 300 dB (theta_ref = 296 K)",
)
def _chk_iso3747_c2_reference_conditions() -> Outcome:
    res = ph.emission.sound_power_in_situ(
        _ISO3747_ST, _ISO3747_RSS, _ISO3747_LW_RSS, _ISO3747_FREQS
    )
    expected = 15.0 * math.log10(296.15 / 296.0)
    return numeric(expected, float(res.c2), 1e-9, unit="dB", places=6)


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Eq. C.2",
    "Static pressure at 500 m, 101,325 (1 - 2,2560e-5 x 500)^5,2553 kPa",
)
def _chk_iso3747_altitude_pressure() -> Outcome:
    expected = 101.325 * (1.0 - 2.2560e-5 * 500.0) ** 5.2553
    computed = ph.emission.static_pressure_from_altitude(500.0)
    return numeric(expected, computed, 1e-9, unit="kPa", places=4)


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Table D.1 / Eq. D.1",
    "LWA of a flat 90 dB octave spectrum, 63 Hz to 8 kHz, with the printed Ck",
)
def _chk_iso3747_a_weighted_total() -> Outcome:
    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    ck = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])  # Table D.1
    flat = np.full((3, freqs.size), 80.0)
    lw_rss = np.full(freqs.size, 90.0)
    # Equal levels for both sources: LW = LW(RSS) = 90 dB in every band.
    res = ph.emission.sound_power_in_situ(flat, flat, lw_rss, freqs)
    expected = 90.0 + 10.0 * math.log10(float(np.sum(10.0 ** (0.1 * ck))))
    return numeric(expected, float(res.sound_power_level_a), 1e-9, unit="dB", places=4)


@register(
    "Intensity & sound power",
    "ISO 3747:2010 Eq. A.1",
    "Excess over the spherical free field Lp = LW - 11 - 20 lg(r/r0): a level 7 dB above it reads dLf = 7 dB",
)
def _chk_iso3747_excess_level() -> Outcome:
    lw, r = 92.0, 4.0
    free = lw - 11.0 - 20.0 * math.log10(r)
    computed = ph.emission.excess_sound_pressure_level(free + 7.0, lw, r)
    return numeric(7.0, float(computed), 1e-12, unit="dB", places=6)


# ---------------------------------------------------------------------------
# ISO 5136:2003, sound power radiated into a duct by fans (in-duct method)
# ---------------------------------------------------------------------------
@register(
    "Intensity & sound power",
    "ISO 5136:2003 Table D.1",
    "C3,4 of the sampling tube for d = 0,5 m at U = +/-5, +/-15, +/-30 m/s, 27 bands",
)
def _chk_iso5136_table_d1() -> Outcome:
    """All 162 printed cells against Eq. (7) with the Table A.4 coefficients.

    The table prints to 0,1 dB, so the budget is half of that; four cells of
    the low bands sit exactly on a decimal half and are printed rounded away
    from zero, which is what puts the worst cell at 0,045 dB.
    """
    worst = 0.0
    for band, row in zip(ref.ISO5136_BANDS, ref.ISO5136_TABLE_D1, strict=True):
        for velocity, printed in zip(ref.ISO5136_TABLE_D1_VELOCITIES, row, strict=True):
            computed = float(
                ph.emission.flow_modal_correction(
                    [float(band)], velocity, ref.ISO5136_ANNEX_D_DIAMETER
                )[0]
            )
            worst = max(worst, abs(computed - printed))
    n = len(ref.ISO5136_BANDS) * len(ref.ISO5136_TABLE_D1_VELOCITIES)
    return numeric(
        0.0,
        worst,
        ref.ISO5136_TABLE_D1_TOLERANCE_DB,
        unit="dB",
        places=3,
        expected_label=f"{n} tabulated values reproduced to the printed 0,1 dB",
        computed_label=f"max absolute deviation {worst:.3f} dB",
    )


@register(
    "Intensity & sound power",
    "ISO 5136:2003 Eqs (D.2)/(D.3)",
    "Worked example: C3,4 = (1,85 + 0,038 U) dB at 1 kHz, U = +15 and -15 m/s",
)
def _chk_iso5136_annex_d_example() -> Outcome:
    """The two framed cells of the example, to the exact product.

    1,85 + 0,038 x 15 = 2,42 dB (printed "approx. 2,4") and
    1,85 + 0,038 x (-15) = 1,28 dB (printed "approx. 1,3").
    """
    worst = 0.0
    for velocity, _printed in (ref.ISO5136_ANNEX_D_OUTLET, ref.ISO5136_ANNEX_D_INLET):
        exact = ref.ISO5136_ANNEX_D_A0 + ref.ISO5136_ANNEX_D_A1 * velocity
        computed = float(
            ph.emission.flow_modal_correction(
                [ref.ISO5136_ANNEX_D_FREQUENCY], velocity, ref.ISO5136_ANNEX_D_DIAMETER
            )[0]
        )
        worst = max(worst, abs(computed - exact))
    return numeric(
        0.0,
        worst,
        1e-9,
        unit="dB",
        places=9,
        expected_label="2,42 dB at +15 m/s and 1,28 dB at -15 m/s reproduced",
        computed_label=f"max absolute deviation {worst:.1e} dB",
    )


@register(
    "Intensity & sound power",
    "ISO 5136:2003 Eq. (8)",
    "Nose-cone / foam-ball correction 10 lg[1/(1 - U/c)^2] at U = 20 m/s, c = 340 m/s",
)
def _chk_iso5136_eq8() -> Outcome:
    """The convective term written out: -20 lg(1 - 20/340) = 0,52658 dB."""
    expected = -20.0 * math.log10(1.0 - 20.0 / ref.ISO5136_C_NORMAL)
    computed = float(
        ph.emission.flow_modal_correction(
            [1000.0], 20.0, ref.ISO5136_ANNEX_D_DIAMETER, shield="nose-cone"
        )[0]
    )
    return numeric(expected, computed, 1e-9, unit="dB", places=5)


@register(
    "Intensity & sound power",
    "ISO 5136:2003 Eq. (12)",
    "Plane-wave relation LW - Lp = 10 lg(S/S0) - 10 lg(rho c/400), d = 0,5 m",
)
def _chk_iso5136_eq12() -> Outcome:
    """The two terms of Eq. (12) written out against the result's own rho c.

    S = pi x 0,5^2 / 4 = 0,196350 m^2, 10 lg S = -7,0691 dB; the duct air at
    20 degC and 101,325 kPa has rho c = 413,25 N s/m^3 (1,2041 kg/m^3 times
    343,20 m/s), so the impedance term is -0,1415 dB and the whole bracket
    -7,2106 dB.
    """
    res = ph.emission.sound_power_in_duct(
        np.full((3, 1), 80.0), [1000.0], ref.ISO5136_ANNEX_D_DIAMETER, 0.0
    )
    area = math.pi * ref.ISO5136_ANNEX_D_DIAMETER**2 / 4.0
    rho = 101325.0 / (287.05 * 293.15)
    c = 20.05 * math.sqrt(273.0 + 20.0)
    expected = 10.0 * math.log10(area / ref.ISO5136_S0) - 10.0 * math.log10(
        rho * c / ref.ISO5136_RHO_C_0
    )
    computed = float(res.sound_power_level[0] - res.corrected_pressure_level[0])
    return numeric(expected, computed, 1e-6, unit="dB", places=4)


@register(
    "Intensity & sound power",
    "ISO 5136:2003 Table 2 / Table 3",
    "Reproducibility sigma_R per band, 50 Hz to 10 kHz, and the extrapolated 12,5 to 20 kHz",
)
def _chk_iso5136_table_2() -> Outcome:
    """Every band of the two tables, the ranges of Table 2 unrolled."""
    expected: dict[int, float] = {}
    for low, high, sigma in ref.ISO5136_TABLE_2_SIGMA_R:
        expected.update(
            (band, sigma) for band in ref.ISO5136_BANDS if low <= band <= high
        )
    expected.update(dict(ref.ISO5136_TABLE_3_SIGMA_R))
    bands = [float(band) for band in ref.ISO5136_BANDS]
    computed = ph.emission.in_duct_reproducibility(bands)
    worst = max(
        abs(float(value) - expected[band])
        for band, value in zip(ref.ISO5136_BANDS, computed, strict=True)
    )
    return numeric(
        0.0,
        worst,
        0.0,
        unit="dB",
        places=3,
        expected_label=f"{len(expected)} tabulated values of sigma_R reproduced",
        computed_label=f"max absolute deviation {worst:.3f} dB",
    )


@register(
    "Intensity & sound power",
    "ISO 5136:2003 Annex C Table C.1",
    "A-weighting C_j of the 27 bands, read back as LWA - LW of one band at a time",
)
def _chk_iso5136_table_c1() -> Outcome:
    """Eq. (C.1) on a single band is LW + C_j, so the difference is the table."""
    worst = 0.0
    for band, cj in zip(ref.ISO5136_BANDS, ref.ISO5136_TABLE_C1, strict=True):
        res = ph.emission.sound_power_in_duct(
            [80.0], [float(band)], ref.ISO5136_ANNEX_D_DIAMETER, 0.0
        )
        worst = max(
            worst, abs(res.sound_power_level_a - float(res.sound_power_level[0]) - cj)
        )
    return numeric(
        0.0,
        worst,
        1e-9,
        unit="dB",
        places=9,
        expected_label=f"{len(ref.ISO5136_TABLE_C1)} tabulated values of C_j reproduced",
        computed_label=f"max absolute deviation {worst:.1e} dB",
    )
