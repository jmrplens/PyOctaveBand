#  Copyright (c) 2026. Jose Manuel Requena Plens
"""building.regulation subdomain of phonometry: national building codes."""

from __future__ import annotations

from .spain import (
    DB_HR_FREQUENCIES,
    DB_HR_NORMALISED_SPECTRA,
    DbHrAssessment,
    DbHrCheck,
    DbHrGlobalIndexResult,
    DbHrRequirement,
    assess_db_hr,
    check_db_hr_requirement,
    d2m_nt_a,
    d2m_nt_atr,
    db_hr_airborne_requirement,
    db_hr_facade_requirement,
    db_hr_global_index,
    db_hr_impact_requirement,
    db_hr_party_wall_requirement,
    db_hr_reverberation_requirement,
    dnt_a,
    ra,
    ra_tr,
    window_size_correction,
)

__all__ = [
    "DB_HR_FREQUENCIES",
    "DB_HR_NORMALISED_SPECTRA",
    "DbHrAssessment",
    "DbHrCheck",
    "DbHrGlobalIndexResult",
    "DbHrRequirement",
    "assess_db_hr",
    "check_db_hr_requirement",
    "d2m_nt_a",
    "d2m_nt_atr",
    "db_hr_airborne_requirement",
    "db_hr_facade_requirement",
    "db_hr_global_index",
    "db_hr_impact_requirement",
    "db_hr_party_wall_requirement",
    "db_hr_reverberation_requirement",
    "dnt_a",
    "ra",
    "ra_tr",
    "window_size_correction",
]
