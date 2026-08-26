#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.building.regulation.spain` (CTE DB-HR).

Two independent oracle families anchor this module.

**The code itself** (CTE Documento Basico HR "Proteccion frente al ruido"):
the four normalised A-weighted source spectra of Annex A Tables A.2 to A.5,
the global-index formulae (A.5) to (A.7), the facade requirement Table 2.1,
the airborne and impact requirements of clauses 2.1.1 and 2.1.2, the
reverberation limits of clause 2.2 and the rounding rule of 3.1.3.1 point 4.

**Aviles Lopez & Perera Martin, Manual de acustica ambiental y arquitectonica
(Paraninfo), chapter 7**: Ejemplo 7.2 (printed pages 394-395) publishes a full
eighteen-band ``R'`` spectrum and its global ``R'A`` = 51,4 dBA; Ejercicio 7.1
(printed page 395) a full ``D2m,nT`` spectrum and its ``D2m,nT,Atr``
= 32,8 dBA; Ejemplo 7.4 with Tabla 7.3 (printed page 408) the window-size
correction of the Catalogo de Elementos Constructivos, a 4 m2 window dropping
from ``RA`` 26 dBA to 24 dBA.

Ejemplo 7.1 (printed pages 391-392) reports the same wall as ``R'w`` = 52 dB
with ``C`` = -1 and ``Ctr`` = -5, giving 51 dBA for pink noise and 47 dBA for
traffic. That is the cross-check that settles the rounding convention: the
direct Annex A route and the ISO 717-1 route agree to the integer on both
published spectra.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from phonometry.building.measurement.insulation import weighted_rating_extended
from phonometry.building.regulation import spain as hr

#: Manual Ejemplo 7.2, printed page 394: the apparent sound reduction index
#: R' of a real field test, 100 Hz to 5 kHz.
_R_PRIME = (
    36.2,
    41.5,
    36.9,
    40.4,
    44.7,
    42.4,
    45.7,
    46.1,
    47.1,
    52.3,
    54.3,
    57.5,
    57.8,
    57.3,
    59.0,
    62.8,
    64.7,
    65.3,
)
#: Manual Ejercicio 7.1, printed page 395: the standardized facade level
#: difference D2m,nT, 100 Hz to 5 kHz.
_D2M_NT = (
    28.5,
    28.5,
    18.9,
    23.7,
    30.7,
    31.3,
    37.8,
    35.2,
    34.7,
    38.5,
    37.7,
    43.1,
    42.3,
    44.2,
    41.9,
    37.5,
    39.4,
    41.5,
)


# --------------------------------------------------------------------------- #
# Annex A: the normalised source spectra
# --------------------------------------------------------------------------- #
def test_band_set_is_the_eighteen_bands_of_annex_a() -> None:
    """Formulae (A.5)-(A.7) run over the 1/3-octave bands 100 Hz to 5 kHz."""
    assert len(hr.DB_HR_FREQUENCIES) == 18
    assert hr.DB_HR_FREQUENCIES[0] == 100.0
    assert hr.DB_HR_FREQUENCIES[-1] == 5000.0


def test_normalised_spectra_match_annex_a_tables() -> None:
    """DB-HR Annex A Tables A.2 to A.5, transcribed value by value."""
    assert hr.DB_HR_NORMALISED_SPECTRA["pink"] == (
        -30.1,
        -27.1,
        -24.4,
        -21.9,
        -19.6,
        -17.6,
        -15.8,
        -14.2,
        -12.9,
        -11.8,
        -11.0,
        -10.4,
        -10.0,
        -9.8,
        -9.7,
        -9.8,
        -10.0,
        -10.5,
    )
    assert hr.DB_HR_NORMALISED_SPECTRA["traffic"] == (
        -20.0,
        -20.0,
        -18.0,
        -16.0,
        -15.0,
        -14.0,
        -13.0,
        -12.0,
        -11.0,
        -9.0,
        -8.0,
        -9.0,
        -10.0,
        -11.0,
        -13.0,
        -15.0,
        -16.0,
        -18.0,
    )
    assert hr.DB_HR_NORMALISED_SPECTRA["aircraft"] == (
        -23.8,
        -20.2,
        -15.4,
        -13.1,
        -12.6,
        -10.4,
        -9.8,
        -9.5,
        -8.7,
        -9.5,
        -10.5,
        -11.0,
        -12.5,
        -14.9,
        -15.9,
        -18.6,
        -23.3,
        -29.9,
    )
    for spectrum in hr.DB_HR_NORMALISED_SPECTRA.values():
        assert len(spectrum) == 18


def test_railway_spectrum_equals_the_road_traffic_spectrum() -> None:
    """Table A.4 (LAef,i) is numerically identical to Table A.3 (LAtr,i)."""
    assert (
        hr.DB_HR_NORMALISED_SPECTRA["railway"] == hr.DB_HR_NORMALISED_SPECTRA["traffic"]
    )


# --------------------------------------------------------------------------- #
# Global indices - Manual Ejemplo 7.2 and Ejercicio 7.1
# --------------------------------------------------------------------------- #
def test_ra_matches_ejemplo_7_2() -> None:
    """Manual Ejemplo 7.2: R'A = 51,4 dBA from the published R' spectrum."""
    index = hr.ra(_R_PRIME)
    assert index.intermediate == pytest.approx(51.4, abs=1e-9)
    assert index.value == pytest.approx(51.44, abs=5e-3)
    assert index.reported == 51
    assert index.name == "RA"
    assert index.spectrum == "pink"
    assert "Table A.5" in index.reference


def test_d2m_nt_atr_matches_ejercicio_7_1() -> None:
    """Manual Ejercicio 7.1: D2m,nT,Atr = 32,8 dBA from the published spectrum."""
    index = hr.d2m_nt_atr(_D2M_NT)
    assert index.intermediate == pytest.approx(32.8, abs=1e-9)
    assert index.reported == 33
    assert index.name == "D2m,nT,Atr"
    assert index.spectrum == "traffic"


def test_global_index_is_minus_the_energy_sum_of_the_band_contributions() -> None:
    """Formula (A.5): I_x = -10 lg sum_i 10^((L_x,i - X_i)/10), recomputed by hand."""
    index = hr.ra(_R_PRIME)
    pink = np.asarray(hr.DB_HR_NORMALISED_SPECTRA["pink"])
    expected = -10.0 * np.log10(np.sum(10.0 ** ((pink - np.asarray(_R_PRIME)) / 10.0)))
    assert index.value == pytest.approx(float(expected), abs=1e-12)
    assert index.band_contributions == pytest.approx(pink - np.asarray(_R_PRIME))


def test_ra_tr_matches_the_traffic_figure_of_ejemplo_7_1() -> None:
    """Manual Ejemplo 7.1: the same wall is 47 dBA against traffic noise.

    The report of Ejemplo 7.1 prints R'w = 52 dB with Ctr = -5, giving
    47 dBA, which the direct Annex A route reproduces from the per-band R'
    of Ejemplo 7.2. This is a second published figure on the same specimen,
    independent of the pink-noise one.
    """
    index = hr.ra_tr(_R_PRIME)
    assert index.reported == 47
    assert index.value == pytest.approx(47.35, abs=5e-3)


def test_direct_route_and_iso717_route_agree_on_both_published_spectra() -> None:
    """The rounding convention, settled against Manual Ejemplo 7.1 and 7.2.

    The module implements the direct Annex A formula, which is the normative
    DB-HR route. On both published spectra it agrees to the integer with the
    ISO 717-1 route via the enlarged-range adaptation terms: the wall gives
    51,4 -> 51 dBA against R'w + C100-5000 = 52 - 1 = 51, and the facade gives
    32,8 -> 33 dBA against D2m,nT,w + Ctr,100-5000 = 38 - 5 = 33.

    The two halves do not carry the same weight. For the wall, Manual
    Ejemplo 7.1 prints R'w = 52 with C = -1 and Ctr = -5, so reproducing
    them is a genuine external check. For the facade the book prints neither
    D2m,nT,w = 38 nor Ctr,100-5000 = -5; both come from the library's own
    ISO 717-1 engine, so that half is a self-consistency check between the
    two routes rather than a published oracle.
    """
    frequencies = list(hr.DB_HR_FREQUENCIES)

    wall = weighted_rating_extended(list(_R_PRIME), frequencies)
    assert wall.rating == 52
    assert wall.c == -1
    assert wall.ctr == -5
    assert wall.c_100_5000 == -1
    assert hr.ra(_R_PRIME).reported == wall.rating + wall.c_100_5000 == 51

    facade = weighted_rating_extended(list(_D2M_NT), frequencies)
    assert facade.rating == 38
    assert facade.ctr_100_5000 == -5
    assert hr.d2m_nt_atr(_D2M_NT).reported == facade.rating + facade.ctr_100_5000 == 33


def test_plain_c_differs_from_the_enlarged_range_term_on_the_facade() -> None:
    """DB-HR names the terms C and Ctr, but they are the enlarged-range ones.

    On the Ejercicio 7.1 facade the core-range C is -2 while C100-5000 is -1,
    so reading "C" as the ISO 717-1 core term would give 36 instead of 37.
    """
    facade = weighted_rating_extended(list(_D2M_NT), list(hr.DB_HR_FREQUENCIES))
    assert facade.c == -2
    assert facade.c_100_5000 == -1
    assert facade.rating + facade.c != facade.rating + facade.c_100_5000


def test_dnt_a_and_ra_tr_use_the_expected_spectra() -> None:
    """DnT,A is the pink-noise index (A.7) and RA,tr the traffic one."""
    assert hr.dnt_a(_D2M_NT).spectrum == "pink"
    assert hr.dnt_a(_D2M_NT).name == "DnT,A"
    assert hr.ra_tr(_R_PRIME).spectrum == "traffic"
    assert hr.ra_tr(_R_PRIME).name == "RA,tr"


def test_railway_dominant_facade_is_assessed_as_d2m_nt_a() -> None:
    """Clause 3.1.3.4 point 1: railway noise gives D2m,nT,A, not D2m,nT,Atr.

    "Cuando el ruido exterior dominante es el ferroviario o el de estaciones
    ferroviarias, se debe usar la magnitud de aislamiento global D2m,nT,A.
    Cuando el ruido exterior dominante es el de automoviles o el de
    aeronaves, la magnitud del aislamiento global es D2m,nT,Atr." Table H.1
    prints the same split, routing railway through Formula (A.5) and road
    traffic and aircraft through (A.6).

    Table A.4 is numerically identical to Table A.3, so the two give the same
    number; only the name of the reported quantity distinguishes them, which
    is why this is asserted on the name and not just the value.
    """
    railway = hr.d2m_nt_a(_D2M_NT, spectrum="railway")
    traffic = hr.d2m_nt_atr(_D2M_NT)
    assert railway.name == "D2m,nT,A"
    assert traffic.name == "D2m,nT,Atr"
    assert railway.value == pytest.approx(traffic.value)
    assert railway.spectrum == "railway"


def test_aircraft_dominant_facade_stays_d2m_nt_atr() -> None:
    """Formula (A.6) also covers the aircraft case, keeping the D2m,nT,Atr name."""
    aircraft = hr.d2m_nt_atr(_D2M_NT, spectrum="aircraft")
    assert aircraft.name == "D2m,nT,Atr"
    assert aircraft.value != pytest.approx(hr.d2m_nt_atr(_D2M_NT).value)


def test_d2m_nt_a_defaults_to_pink_noise() -> None:
    """Formula (A.5) is the pink-noise facade quantity."""
    assert hr.d2m_nt_a(_D2M_NT).spectrum == "pink"
    assert hr.d2m_nt_a(_D2M_NT).name == "D2m,nT,A"


def test_facade_helpers_reject_each_other_s_spectra() -> None:
    """Neither helper accepts a spectrum its formula does not cover."""
    with pytest.raises(ValueError, match="d2m_nt_a"):
        hr.d2m_nt_atr(_D2M_NT, spectrum="railway")
    with pytest.raises(ValueError, match="d2m_nt_atr"):
        hr.d2m_nt_a(_D2M_NT, spectrum="traffic")


def test_railway_facade_requirement_is_stated_in_d2m_nt_a() -> None:
    """The Table 2.1 requirement names the quantity of clause 3.1.3.4 point 1.

    The numeric limit is the same column either way; the quantity the
    requirement is expressed in is not.
    """
    road = hr.db_hr_facade_requirement(62.0, "residential", "bedrooms")
    rail = hr.db_hr_facade_requirement(
        62.0, "residential", "bedrooms", dominant_noise="railway"
    )
    air = hr.db_hr_facade_requirement(
        62.0, "residential", "bedrooms", dominant_noise="aircraft"
    )
    assert road.quantity == "D2m,nT,Atr"
    assert rail.quantity == "D2m,nT,A"
    assert air.quantity == "D2m,nT,Atr"
    assert rail.limit == road.limit == 32.0


def test_global_index_selects_the_eighteen_bands_from_a_wider_spectrum() -> None:
    """A spectrum wider than Annex A is narrowed to the eighteen bands."""
    frequencies = [50.0, 63.0, 80.0, *hr.DB_HR_FREQUENCIES]
    values = [10.0, 10.0, 10.0, *_R_PRIME]
    assert hr.ra(values, frequencies=frequencies).value == pytest.approx(
        hr.ra(_R_PRIME).value, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Rounding (3.1.3.1 point 4)
# --------------------------------------------------------------------------- #
def test_rounding_exposes_exact_one_decimal_and_integer_values() -> None:
    """3.1.3.1 point 4: the requirement-defining values are rounded to an integer.

    Intermediate quantities carry one decimal, so the result object exposes
    the exact value, the one-decimal intermediate and the reported integer.
    """
    index = hr.ra(_R_PRIME)
    assert index.value > index.intermediate  # 51,443... vs 51,4
    assert index.intermediate == 51.4
    assert index.reported == 51
    assert isinstance(index.reported, int)


def test_rounding_is_half_up_not_bankers() -> None:
    """A value ending in exactly 0,5 rounds up, unlike Python's round()."""
    requirement = hr.db_hr_facade_requirement(62.0, "residential", "bedrooms")
    assert hr.check_db_hr_requirement(31.5, requirement).reported == 32.0
    assert hr.check_db_hr_requirement(32.5, requirement).reported == 33.0


# --------------------------------------------------------------------------- #
# Window size correction - Manual Ejemplo 7.4 / Tabla 7.3
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("area", "expected"),
    [
        (1.8, 0),
        (2.0, 0),
        (2.7, 0),
        (2.8, -1),
        (3.0, -1),
        (3.6, -1),
        (3.7, -2),
        (4.0, -2),
        (4.6, -2),
        (4.7, -3),
        (8.0, -3),
    ],
)
def test_window_size_correction_table(area: float, expected: int) -> None:
    """Manual Tabla 7.3 (from the CTE Catalogo de Elementos Constructivos).

    Every breakpoint of the table is checked on both sides: 0 dB up to
    2,7 m2, -1 dB over (2,7 - 3,6], -2 dB over (3,6 - 4,6] and -3 dB above.
    """
    assert hr.window_size_correction(area) == expected


def test_window_size_correction_reproduces_ejemplo_7_4() -> None:
    """Manual Ejemplo 7.4: a 4 m2 window catalogued at RA 26 dBA becomes 24 dBA."""
    assert 26 + hr.window_size_correction(4.0) == 24


# --------------------------------------------------------------------------- #
# Requirements: Table 2.1 facades
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("ld", "expected"),
    [
        (55.0, 30),
        (60.0, 30),
        (60.1, 32),
        (62.0, 32),
        (65.0, 32),
        (65.1, 37),
        (68.0, 37),
        (70.0, 37),
        (70.1, 42),
        (72.0, 42),
        (75.0, 42),
        (75.1, 47),
        (76.0, 47),
        (80.0, 47),
    ],
)
def test_facade_requirement_bedrooms_series(ld: float, expected: int) -> None:
    """DB-HR Table 2.1, residential/hospital bedrooms column, every Ld band.

    The bands are Ld <= 60, 60 < Ld <= 65, 65 < Ld <= 70, 70 < Ld <= 75 and
    Ld > 75, and this column reads 30/32/37/42/47 dBA.
    """
    req = hr.db_hr_facade_requirement(ld, "residential", "bedrooms")
    assert req.limit == float(expected)
    assert req.quantity == "D2m,nT,Atr"
    assert req.direction == "min"


@pytest.mark.parametrize(
    ("ld", "expected"),
    [(60.0, 30), (62.0, 30), (68.0, 32), (72.0, 37), (80.0, 42)],
)
def test_facade_requirement_living_series(ld: float, expected: int) -> None:
    """DB-HR Table 2.1, residential/hospital estancias column: 30/30/32/37/42 dBA."""
    assert hr.db_hr_facade_requirement(ld, "residential", "living").limit == expected


@pytest.mark.parametrize(
    ("use", "room", "expected"),
    [
        ("residential", "bedrooms", 42),
        ("residential", "living", 37),
        ("hospital", "bedrooms", 42),
        ("hospital", "living", 37),
        ("cultural", "living", 42),
        ("cultural", "classrooms", 37),
        ("sanitary", "living", 42),
        ("sanitary", "classrooms", 37),
        ("educational", "living", 42),
        ("educational", "classrooms", 37),
        ("administrative", "living", 42),
        ("administrative", "classrooms", 37),
    ],
)
def test_facade_requirement_covers_every_column_of_table_2_1(
    use: str, room: str, expected: int
) -> None:
    """Every (building use, room type) pairing of Table 2.1, at 70 < Ld <= 75.

    The residential/hospital block pairs dormitorios with the 30/32/37/42/47
    series and estancias with 30/30/32/37/42; the cultural, sanitario, docente
    and administrativo block pairs estancias with the first series and aulas
    with the second.
    """
    assert hr.db_hr_facade_requirement(72.0, use, room).limit == float(expected)


def test_facade_requirement_aircraft_increment() -> None:
    """Clause 2.1.1 a) iv: where aircraft noise dominates, Table 2.1 plus 4 dBA."""
    base = hr.db_hr_facade_requirement(62.0, "residential", "bedrooms")
    aircraft = hr.db_hr_facade_requirement(
        62.0, "residential", "bedrooms", dominant_noise="aircraft"
    )
    assert aircraft.limit == base.limit + 4.0
    assert "aircraft" in aircraft.description


def test_facade_requirement_quiet_facade_reduction() -> None:
    """Clause 2.1.1 a) iv: a sheltered facade is assessed with Ld minus 10 dBA.

    An enclosed courtyard in a zone of Ld = 72 dBA drops to Ld = 62 dBA, so a
    bedroom requirement falls from 42 dBA to 32 dBA.
    """
    sheltered = hr.db_hr_facade_requirement(
        72.0, "residential", "bedrooms", quiet_facade=True
    )
    assert sheltered.limit == 32.0
    assert (
        sheltered.limit
        == hr.db_hr_facade_requirement(62.0, "residential", "bedrooms").limit
    )


def test_facade_requirement_use_and_room_aliases() -> None:
    """The Spanish and alternative English spellings resolve to the same row."""
    reference = hr.db_hr_facade_requirement(62.0, "residential", "bedrooms").limit
    for use, room in (
        ("dwelling", "dormitorios"),
        ("housing", "bedroom"),
        ("RESIDENTIAL", "Bedrooms"),
    ):
        assert hr.db_hr_facade_requirement(62.0, use, room).limit == reference
    assert (
        hr.db_hr_facade_requirement(62.0, "docente", "aulas").limit
        == hr.db_hr_facade_requirement(62.0, "educational", "classrooms").limit
    )


# --------------------------------------------------------------------------- #
# Requirements: airborne, impact, party wall, reverberation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("receiving", "source", "quantity", "limit"),
    [
        ("protected", "same_unit", "RA", 33.0),
        ("protected", "other_unit", "DnT,A", 50.0),
        ("protected", "installations", "DnT,A", 55.0),
        ("protected", "activity", "DnT,A", 55.0),
        ("habitable", "same_unit", "RA", 33.0),
        ("habitable", "other_unit", "DnT,A", 45.0),
        ("habitable", "installations", "DnT,A", 45.0),
        ("habitable", "activity", "DnT,A", 45.0),
    ],
)
def test_airborne_requirements_clause_2_1_1(
    receiving: str, source: str, quantity: str, limit: float
) -> None:
    """DB-HR 2.1.1 a) and b), every case: the tabiqueria RA and the DnT,A rows."""
    (req,) = hr.db_hr_airborne_requirement(receiving, source)
    assert (req.quantity, req.limit, req.direction) == (quantity, limit, "min")


@pytest.mark.parametrize(
    ("receiving", "source", "opening", "enclosure"),
    [
        ("protected", "other_unit", 30.0, 50.0),
        ("habitable", "other_unit", 20.0, 50.0),
        ("habitable", "installations", 30.0, 50.0),
        ("habitable", "activity", 30.0, 50.0),
    ],
)
def test_shared_opening_variants_clause_2_1_1(
    receiving: str, source: str, opening: float, enclosure: float
) -> None:
    """When the rooms share a door or window, two RA requirements replace DnT,A."""
    reqs = hr.db_hr_airborne_requirement(receiving, source, shared_opening=True)
    assert len(reqs) == 2
    assert [r.quantity for r in reqs] == ["RA", "RA"]
    assert [r.limit for r in reqs] == [opening, enclosure]


def test_shared_opening_variant_records_the_residential_condition() -> None:
    """2.1.1 b) ii states the 20 dBA variant only for residential and hospital buildings."""
    opening, _ = hr.db_hr_airborne_requirement(
        "habitable", "other_unit", shared_opening=True
    )
    assert "residential" in opening.description
    assert "hospital" in opening.description


def test_shared_opening_is_rejected_where_the_code_states_none() -> None:
    """DB-HR gives no shared-opening variant for a protected room against plant rooms."""
    with pytest.raises(ValueError, match="no shared-opening variant"):
        hr.db_hr_airborne_requirement("protected", "installations", shared_opening=True)


@pytest.mark.parametrize(
    ("receiving", "source", "limit"),
    [
        ("protected", "other_unit", 65.0),
        ("protected", "installations", 60.0),
        ("protected", "activity", 60.0),
        ("habitable", "installations", 60.0),
        ("habitable", "activity", 60.0),
    ],
)
def test_impact_requirements_clause_2_1_2(
    receiving: str, source: str, limit: float
) -> None:
    """DB-HR 2.1.2: L'nT,w at most 65 dB against another use unit, 60 dB otherwise."""
    req = hr.db_hr_impact_requirement(receiving, source)
    assert (req.quantity, req.limit, req.direction) == ("L'nT,w", limit, "max")


def test_impact_requirement_rejects_a_case_the_code_does_not_state() -> None:
    """DB-HR 2.1.2 b) states no impact limit for a habitable room against another unit."""
    with pytest.raises(ValueError, match="no impact requirement"):
        hr.db_hr_impact_requirement("habitable", "other_unit")


def test_party_wall_requirement_clause_2_1_1_c() -> None:
    """DB-HR 2.1.1 c): 40 dBA per leaf, or 50 dBA for both leaves together.

    The two are alternatives, not cumulative requirements.
    """
    per_leaf = hr.db_hr_party_wall_requirement()
    assert (per_leaf.quantity, per_leaf.limit) == ("D2m,nT,Atr", 40.0)
    combined = hr.db_hr_party_wall_requirement("DnT,A")
    assert (combined.quantity, combined.limit) == ("DnT,A", 50.0)
    with pytest.raises(ValueError, match="quantity"):
        hr.db_hr_party_wall_requirement("RA")


@pytest.mark.parametrize(
    ("room_use", "furnished", "quantity", "limit", "unit"),
    [
        ("classroom", False, "T", 0.7, "s"),
        ("classroom", True, "T", 0.5, "s"),
        ("conference_hall", False, "T", 0.7, "s"),
        ("restaurant", False, "T", 0.9, "s"),
        ("dining_room", False, "T", 0.9, "s"),
        ("common_area", False, "A/V", 0.2, "m2/m3"),
    ],
)
def test_reverberation_requirements_clause_2_2(
    room_use: str, furnished: bool, quantity: str, limit: float, unit: str
) -> None:
    """DB-HR 2.2: 0,7 s / 0,5 s classrooms, 0,9 s restaurants, A/V >= 0,2 m2/m3."""
    req = hr.db_hr_reverberation_requirement(room_use, furnished=furnished)
    assert (req.quantity, req.limit, req.unit) == (quantity, limit, unit)
    assert req.direction == ("max" if quantity == "T" else "min")


# --------------------------------------------------------------------------- #
# Compliance checking
# --------------------------------------------------------------------------- #
def test_check_rounds_before_comparing_and_reports_the_margin() -> None:
    """3.1.3.1 point 4: the achieved value is rounded before the comparison.

    The Ejercicio 7.1 facade achieves 32,8 dBA, which reports as 33 dBA and
    therefore meets a 33 dBA requirement it would miss unrounded.
    """
    req = hr.db_hr_facade_requirement(68.0, "residential", "living")  # 32 dBA
    check = hr.check_db_hr_requirement(hr.d2m_nt_atr(_D2M_NT).value, req)
    assert check.reported == 33.0
    assert check.margin == 1.0
    assert check.complies is True

    strict = hr.db_hr_facade_requirement(72.0, "residential", "bedrooms")  # 42 dBA
    failing = hr.check_db_hr_requirement(hr.d2m_nt_atr(_D2M_NT).value, strict)
    assert failing.margin == pytest.approx(-9.0)
    assert failing.complies is False


def test_check_direction_max_inverts_the_margin() -> None:
    """For a "max" requirement the margin is limit - achieved."""
    req = hr.db_hr_impact_requirement("protected", "other_unit")  # <= 65 dB
    assert hr.check_db_hr_requirement(58.4, req).complies is True
    assert hr.check_db_hr_requirement(58.4, req).margin == pytest.approx(7.0)
    assert hr.check_db_hr_requirement(67.2, req).complies is False


def test_reverberation_check_rounds_to_one_decimal() -> None:
    """DB-HR rounds reverberation times to the first decimal (1,25 -> 1,3)."""
    req = hr.db_hr_reverberation_requirement("classroom")  # <= 0,7 s
    assert hr.check_db_hr_requirement(0.74, req).reported == 0.7
    assert hr.check_db_hr_requirement(0.74, req).complies is True
    assert hr.check_db_hr_requirement(0.75, req).reported == 0.8
    assert hr.check_db_hr_requirement(0.75, req).complies is False


def test_assessment_aggregates_every_check() -> None:
    """An assessment complies only when every one of its checks does."""
    facade = hr.db_hr_facade_requirement(62.0, "residential", "bedrooms")  # 32 dBA
    impact = hr.db_hr_impact_requirement("protected", "other_unit")  # <= 65 dB
    good = hr.assess_db_hr([(33.0, facade), (58.0, impact)])
    assert [c.complies for c in good.checks] == [True, True]
    assert good.complies is True
    mixed = hr.assess_db_hr([(33.0, facade), (70.0, impact)])
    assert mixed.complies is False


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_global_index_validation() -> None:
    """Wrong band counts, unknown spectra and non-finite values are rejected."""
    with pytest.raises(ValueError, match="spectrum"):
        hr.db_hr_global_index(_R_PRIME, "white")
    with pytest.raises(ValueError, match="eighteen"):
        hr.db_hr_global_index(_R_PRIME[:10])
    with pytest.raises(ValueError, match="finite"):
        hr.db_hr_global_index([math.nan, *_R_PRIME[1:]])
    two_dimensional = np.zeros((2, 18))
    with pytest.raises(ValueError, match="one-dimensional"):
        hr.db_hr_global_index(two_dimensional)
    with pytest.raises(ValueError, match=r"'frequencies'.*same shape"):
        hr.db_hr_global_index(_R_PRIME, frequencies=[100.0, 125.0])
    with pytest.raises(ValueError, match="positive"):
        hr.db_hr_global_index(_R_PRIME, frequencies=[0.0, *hr.DB_HR_FREQUENCIES[1:]])


def test_global_index_rejects_a_spectrum_missing_a_db_hr_band() -> None:
    """Every one of the eighteen bands must be present exactly once."""
    frequencies = [*hr.DB_HR_FREQUENCIES[:-1], 6300.0]
    with pytest.raises(ValueError, match="5000 Hz matched 0 bands"):
        hr.db_hr_global_index(_R_PRIME, frequencies=frequencies)


def test_global_index_rejects_a_spectrum_of_another_band_set() -> None:
    """The normalised spectrum is the column nothing downstream would catch.

    The index and its two rounded forms closed over the bands they were handed
    at construction, and the plot draws only the frequencies, the band values
    and their contributions, so a spectrum one band too long changes nothing
    anyone would see: the result would go on quoting an eighteen-band index
    beside a nineteen-band Annex A table.
    """
    index = hr.ra(_R_PRIME)
    one_band_too_many = np.append(index.spectrum_levels, 99.9)
    with pytest.raises(ValueError, match="'spectrum_levels'"):
        dataclasses.replace(index, spectrum_levels=one_band_too_many)


def test_global_index_rejects_a_spectrum_annex_a_does_not_tabulate() -> None:
    """The figure titles itself from a label table keyed by ``spectrum``.

    The entry point checks the key at the call, but a result rewritten by hand
    reaches that lookup unchecked, after the whole page is drawn, and dies with
    a bare ``KeyError`` naming the key alone: neither the field nor the four
    Annex A tables it could have named.
    """
    index = hr.ra(_R_PRIME)
    with pytest.raises(ValueError, match="'spectrum' must be one of"):
        dataclasses.replace(index, spectrum="brown")


def test_a_check_whose_verdict_contradicts_its_margin_is_refused() -> None:
    """The lollipop colours each check by ``complies`` and nothing else.

    A check reporting 45 dBA against a 50 dBA minimum, with the flag inverted,
    would be painted in the compliant colour beside a stem whose very geometry
    says otherwise.
    """
    requirement = hr.DbHrRequirement(
        "DnT,A", 50.0, "min", "dBA", 0, "DB-HR 2.1", "between dwellings"
    )
    with pytest.raises(ValueError, match="'complies' must agree"):
        hr.DbHrCheck(requirement, 45.0, 45.0, -5.0, complies=True)


def test_a_check_whose_margin_does_not_restate_its_comparison_is_refused() -> None:
    """The margin is the reported value against the limit, and nothing else.

    The assessment figure draws the stem to ``margin`` and the fiche prints it,
    so a margin computed against another limit reads as a comfortable pass over
    a value that never earned it.
    """
    requirement = hr.DbHrRequirement(
        "DnT,A", 50.0, "min", "dBA", 0, "DB-HR 2.1", "between dwellings"
    )
    with pytest.raises(ValueError, match="'margin' must be 'reported' minus"):
        hr.DbHrCheck(requirement, 45.0, 45.0, 5.0, complies=True)


def test_requirement_lookup_validation() -> None:
    """Unknown uses, rooms and room-pair combinations are rejected."""
    with pytest.raises(ValueError, match="Unknown"):
        hr.db_hr_facade_requirement(62.0, "industrial", "bedrooms")
    with pytest.raises(ValueError, match="dominant_noise"):
        hr.db_hr_facade_requirement(
            62.0, "residential", "bedrooms", dominant_noise="sea"
        )
    with pytest.raises(ValueError, match="finite"):
        hr.db_hr_facade_requirement(math.nan, "residential", "bedrooms")
    with pytest.raises(ValueError, match="Unknown"):
        hr.db_hr_airborne_requirement("protected", "outdoors")
    with pytest.raises(ValueError, match="room_use"):
        hr.db_hr_reverberation_requirement("swimming_pool")


def test_requirement_direction_is_validated() -> None:
    """An unchecked direction typo would silently invert the verdict.

    ``check_db_hr_requirement`` branches on "min" and treats anything else as
    "max", so the dataclass rejects any other spelling at construction.
    """
    with pytest.raises(ValueError, match="direction"):
        hr.DbHrRequirement("D2m,nT,Atr", 32.0, "minimum", "dBA", 0, "ref", "desc")
    assert hr.DbHrRequirement("T", 0.7, "max", "s", 1, "ref", "desc").direction == "max"


def test_published_spectra_are_read_only() -> None:
    """The Annex A tables are a published constant, not a mutable global."""
    with pytest.raises(TypeError):
        hr.DB_HR_NORMALISED_SPECTRA["pink"] = (0.0,) * 18  # type: ignore[index]


def test_result_does_not_alias_the_caller_s_array() -> None:
    """Mutating the input afterwards must not rewrite the stored result."""
    values = np.asarray(_R_PRIME, dtype=np.float64)
    index = hr.ra(values)
    values[0] = 0.0
    assert index.band_values[0] == pytest.approx(36.2)


def test_window_size_and_assessment_validation() -> None:
    """A window needs a positive area and an assessment at least one check."""
    with pytest.raises(ValueError, match="positive"):
        hr.window_size_correction(0.0)
    with pytest.raises(ValueError, match="positive"):
        hr.window_size_correction(-2.0)
    with pytest.raises(ValueError, match="At least one"):
        hr.assess_db_hr([])
    impact_requirement = hr.db_hr_impact_requirement("protected", "other_unit")
    with pytest.raises(ValueError, match="finite"):
        hr.check_db_hr_requirement(math.nan, impact_requirement)
