#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared normative reference data (single source of truth).

Tables transcribed verbatim from the published standards. Both the test
suite (``tests/test_*.py``) and the CI conformance report
(``scripts/conformance_report.py``) import these constants, so the report's
expected values can never drift from what the tests assert. Test modules
either import a constant directly where they assert it, or - where the
oracle lives inside a larger inline table (e.g. the ISO 1999 Annex D or
ANSI S12.2 Table D.1 transcriptions) - pin the inline copy to the shared
constant with an explicit consistency assertion. The PR-B building-acoustics
and PR-F human-vibration oracles are additionally pinned to their published
values by dedicated tests in ``tests/test_conformance_report.py``
(``test_building_reference_data_matches_published_oracles`` and
``test_human_vibration_reference_data_matches_oracles``).

The tables live in one submodule per subject, laid out like the ``tests/``
tree and the library itself, and every constant keeps the banner comment
naming the table and the edition it was transcribed from: that citation is
the reason the number can be trusted. This file re-exports every one of them
by name - never through a star import - so ``import reference_data as ref``
and ``from reference_data import X`` resolve exactly as they did while this
was a single module.

These modules are deliberately dependency-free (stdlib only) so they can be
imported in the ``pr-comment`` CI job, which installs the runtime
requirements but not ``pytest``.
"""

from .broadcast import BS468_BURST_HZ as BS468_BURST_HZ
from .broadcast import BS468_CALIBRATION_V as BS468_CALIBRATION_V
from .broadcast import BS468_OVERLOAD_BURST_MS as BS468_OVERLOAD_BURST_MS
from .broadcast import BS468_OVERLOAD_RANGE_DB as BS468_OVERLOAD_RANGE_DB
from .broadcast import BS468_OVERLOAD_TOL_DB as BS468_OVERLOAD_TOL_DB
from .broadcast import BS468_OVERSWING_TOL_DB as BS468_OVERSWING_TOL_DB
from .broadcast import BS468_REVERSIBILITY_TOL_DB as BS468_REVERSIBILITY_TOL_DB
from .broadcast import BS468_TABLE2_SINGLE_BURSTS as BS468_TABLE2_SINGLE_BURSTS
from .broadcast import BS468_TABLE3_BURST_TRAINS as BS468_TABLE3_BURST_TRAINS
from .broadcast import BS1770_ANCHOR_997_LKFS as BS1770_ANCHOR_997_LKFS
from .broadcast import EBU_TECH3341_CASE6_EXPECTED as EBU_TECH3341_CASE6_EXPECTED
from .broadcast import EBU_TECH3341_CASE6_LEVELS as EBU_TECH3341_CASE6_LEVELS
from .broadcast import EBU_TECH3341_INTEGRATED_CASES as EBU_TECH3341_INTEGRATED_CASES
from .broadcast import EBU_TECH3341_TOL_LU as EBU_TECH3341_TOL_LU
from .broadcast import (
    EBU_TECH3341_TP_OFFSET_EXPECTED as EBU_TECH3341_TP_OFFSET_EXPECTED,
)
from .broadcast import EBU_TECH3341_TP_TOL_DOWN as EBU_TECH3341_TP_TOL_DOWN
from .broadcast import EBU_TECH3341_TP_TOL_UP as EBU_TECH3341_TP_TOL_UP
from .broadcast import EBU_TECH3341_TRUE_PEAK_CASES as EBU_TECH3341_TRUE_PEAK_CASES
from .broadcast import EBU_TECH3342_LRA_CASES as EBU_TECH3342_LRA_CASES
from .broadcast import EBU_TECH3342_TOL_LU as EBU_TECH3342_TOL_LU
from .broadcast_wave import BS2088_BW64_FOURCC as BS2088_BW64_FOURCC
from .broadcast_wave import BS2088_DS64_FIELDS as BS2088_DS64_FIELDS
from .broadcast_wave import BS2088_DS64_MIN_SIZE as BS2088_DS64_MIN_SIZE
from .broadcast_wave import BS2088_SIZE_SENTINEL as BS2088_SIZE_SENTINEL
from .broadcast_wave import (
    EBU_R98_EXAMPLE1_PCM_PREFIX as EBU_R98_EXAMPLE1_PCM_PREFIX,
)
from .broadcast_wave import (
    EBU_R98_EXAMPLE2_PRIOR_LINE as EBU_R98_EXAMPLE2_PRIOR_LINE,
)
from .broadcast_wave import EBU_R98_MODES as EBU_R98_MODES
from .broadcast_wave import EBU_R98_PCM_PARAMETERS as EBU_R98_PCM_PARAMETERS
from .broadcast_wave import TECH3285_BEXT_FIELDS as TECH3285_BEXT_FIELDS
from .broadcast_wave import (
    TECH3285_BEXT_FIXED_SIZE as TECH3285_BEXT_FIXED_SIZE,
)
from .broadcast_wave import (
    TECH3285_LOUDNESS_BYTES as TECH3285_LOUDNESS_BYTES,
)
from .broadcast_wave import (
    TECH3285_LOUDNESS_EXAMPLES as TECH3285_LOUDNESS_EXAMPLES,
)
from .broadcast_wave import (
    TECH3285_LOUDNESS_UNSET as TECH3285_LOUDNESS_UNSET,
)
from .broadcast_wave import TECH3285_UMID_SIZE as TECH3285_UMID_SIZE
from .broadcast_wave import TECH3285_V0_RESERVED as TECH3285_V0_RESERVED
from .broadcast_wave import TECH3285_V1_RESERVED as TECH3285_V1_RESERVED
from .broadcast_wave import TECH3285_V2_RESERVED as TECH3285_V2_RESERVED
from .broadcast_wave import TECH3285_VERSION_1 as TECH3285_VERSION_1
from .broadcast_wave import TECH3285_VERSION_2 as TECH3285_VERSION_2
from .building import EN12354_1_ANNEX_H3_DNT_W as EN12354_1_ANNEX_H3_DNT_W
from .building import (
    EN12354_1_ANNEX_H3_DNT_W_PRINTED as EN12354_1_ANNEX_H3_DNT_W_PRINTED,
)
from .building import EN12354_1_ANNEX_H3_DNT_W_SECOND as EN12354_1_ANNEX_H3_DNT_W_SECOND
from .building import EN12354_1_ANNEX_H3_ELEMENTS as EN12354_1_ANNEX_H3_ELEMENTS
from .building import EN12354_1_ANNEX_H3_NUM_PATHS as EN12354_1_ANNEX_H3_NUM_PATHS
from .building import EN12354_1_ANNEX_H3_PATH_RW as EN12354_1_ANNEX_H3_PATH_RW
from .building import EN12354_1_ANNEX_H3_R_DIRECT as EN12354_1_ANNEX_H3_R_DIRECT
from .building import EN12354_1_ANNEX_H3_RPRIME_W as EN12354_1_ANNEX_H3_RPRIME_W
from .building import (
    EN12354_1_ANNEX_H3_SEPARATING_AREA as EN12354_1_ANNEX_H3_SEPARATING_AREA,
)
from .building import EN12354_1_ANNEX_H3_VOLUME as EN12354_1_ANNEX_H3_VOLUME
from .building import EN12354_2_ANNEX_E3_DELTA_LW as EN12354_2_ANNEX_E3_DELTA_LW
from .building import (
    EN12354_2_ANNEX_E3_FLANKING_MEAN_MASS as EN12354_2_ANNEX_E3_FLANKING_MEAN_MASS,
)
from .building import EN12354_2_ANNEX_E3_K as EN12354_2_ANNEX_E3_K
from .building import EN12354_2_ANNEX_E3_LPRIME_N_W as EN12354_2_ANNEX_E3_LPRIME_N_W
from .building import EN12354_2_ANNEX_E3_MASS as EN12354_2_ANNEX_E3_MASS
from .building import EN12354_3_ANNEX_C_DLFS as EN12354_3_ANNEX_C_DLFS
from .building import EN12354_3_ANNEX_F_AREA as EN12354_3_ANNEX_F_AREA
from .building import EN12354_3_ANNEX_F_BANDS as EN12354_3_ANNEX_F_BANDS
from .building import EN12354_3_ANNEX_F_CTR as EN12354_3_ANNEX_F_CTR
from .building import EN12354_3_ANNEX_F_D2MNT_W as EN12354_3_ANNEX_F_D2MNT_W
from .building import EN12354_3_ANNEX_F_ELEMENTS as EN12354_3_ANNEX_F_ELEMENTS
from .building import EN12354_3_ANNEX_F_INLET_DNE as EN12354_3_ANNEX_F_INLET_DNE
from .building import EN12354_3_ANNEX_F_RPRIME_LOW as EN12354_3_ANNEX_F_RPRIME_LOW
from .building import EN12354_3_ANNEX_F_RTRS_W as EN12354_3_ANNEX_F_RTRS_W
from .building import EN12354_3_ANNEX_F_VOLUME as EN12354_3_ANNEX_F_VOLUME
from .building import EN12354_4_ANNEX_G_ATTENUATION as EN12354_4_ANNEX_G_ATTENUATION
from .building import EN12354_4_ANNEX_G_BANDS as EN12354_4_ANNEX_G_BANDS
from .building import EN12354_4_ANNEX_G_CD as EN12354_4_ANNEX_G_CD
from .building import EN12354_4_ANNEX_G_CONCRETE_R as EN12354_4_ANNEX_G_CONCRETE_R
from .building import EN12354_4_ANNEX_G_DOOR_AREA as EN12354_4_ANNEX_G_DOOR_AREA
from .building import EN12354_4_ANNEX_G_DOOR_R as EN12354_4_ANNEX_G_DOOR_R
from .building import EN12354_4_ANNEX_G_LP_IN as EN12354_4_ANNEX_G_LP_IN
from .building import EN12354_4_ANNEX_G_LP_SIDE1_D5 as EN12354_4_ANNEX_G_LP_SIDE1_D5
from .building import EN12354_4_ANNEX_G_LP_SIDE1_D25 as EN12354_4_ANNEX_G_LP_SIDE1_D25
from .building import EN12354_4_ANNEX_G_LP_SIDE4_D5 as EN12354_4_ANNEX_G_LP_SIDE4_D5
from .building import EN12354_4_ANNEX_G_LP_SIDE4_D25 as EN12354_4_ANNEX_G_LP_SIDE4_D25
from .building import EN12354_4_ANNEX_G_RPRIME_CAP as EN12354_4_ANNEX_G_RPRIME_CAP
from .building import EN12354_4_ANNEX_G_SEGMENT_AREA as EN12354_4_ANNEX_G_SEGMENT_AREA
from .building import EN12354_4_ANNEX_G_SIDE1_LW_LOW as EN12354_4_ANNEX_G_SIDE1_LW_LOW
from .building import EN12354_4_ANNEX_G_SIDE1_LWA as EN12354_4_ANNEX_G_SIDE1_LWA
from .building import (
    EN12354_4_ANNEX_G_SIDE1_RPRIME_LOW as EN12354_4_ANNEX_G_SIDE1_RPRIME_LOW,
)
from .building import EN12354_4_ANNEX_G_SIDE4_LWA as EN12354_4_ANNEX_G_SIDE4_LWA
from .building import EN12354_5_ANNEX_I_BANDS as EN12354_5_ANNEX_I_BANDS
from .building import EN12354_5_ANNEX_I_TOL as EN12354_5_ANNEX_I_TOL
from .building import EN12354_5_I6A_DSA_FLOOR as EN12354_5_I6A_DSA_FLOOR
from .building import EN12354_5_I6A_LNS_11 as EN12354_5_I6A_LNS_11
from .building import EN12354_5_I6A_LWSN_FLOOR as EN12354_5_I6A_LWSN_FLOOR
from .building import EN12354_5_I6A_LWSN_INST_FLOOR as EN12354_5_I6A_LWSN_INST_FLOOR
from .building import EN12354_5_I6A_R11 as EN12354_5_I6A_R11
from .building import EN12354_5_I6A_Y_FLOOR as EN12354_5_I6A_Y_FLOOR
from .building import EN12354_5_I8_FLOOR_INSTALLED as EN12354_5_I8_FLOOR_INSTALLED
from .building import EN12354_5_I8_FLOOR_LWS as EN12354_5_I8_FLOOR_LWS
from .building import EN12354_5_I8_FLOOR_LWSC as EN12354_5_I8_FLOOR_LWSC
from .building import EN12354_5_I8_PLATE_MOBILITY as EN12354_5_I8_PLATE_MOBILITY
from .building import EN12354_5_I8_WALL_INSTALLED as EN12354_5_I8_WALL_INSTALLED
from .building import EN12354_5_I8_WALL_LWS as EN12354_5_I8_WALL_LWS
from .building import EN12354_5_I8_WALL_LWSC as EN12354_5_I8_WALL_LWSC
from .building import EN12354_5_I8_Y_FLOOR as EN12354_5_I8_Y_FLOOR
from .building import EN12354_5_I8_Y_SOURCE as EN12354_5_I8_Y_SOURCE
from .building import EN12354_5_I8_Y_WALL as EN12354_5_I8_Y_WALL
from .building import EN12354_5_I9_DC_FLOOR as EN12354_5_I9_DC_FLOOR
from .building import EN12354_5_I9_DC_WALL as EN12354_5_I9_DC_WALL
from .building import EN12354_5_I9_DSA_FLOOR as EN12354_5_I9_DSA_FLOOR
from .building import EN12354_5_I9_DSA_WALL as EN12354_5_I9_DSA_WALL
from .building import EN12354_5_I9_LNS_FLOOR_FLOOR as EN12354_5_I9_LNS_FLOOR_FLOOR
from .building import EN12354_5_I9_LNS_FLOOR_WALL as EN12354_5_I9_LNS_FLOOR_WALL
from .building import EN12354_5_I9_LNS_TOTAL as EN12354_5_I9_LNS_TOTAL
from .building import EN12354_5_I9_LNS_TOTAL_A as EN12354_5_I9_LNS_TOTAL_A
from .building import EN12354_5_I9_LNS_WALL_FLOOR as EN12354_5_I9_LNS_WALL_FLOOR
from .building import EN12354_5_I9_LNS_WALL_WALL as EN12354_5_I9_LNS_WALL_WALL
from .building import EN12354_5_I9_R_FLOOR_FLOOR as EN12354_5_I9_R_FLOOR_FLOOR
from .building import EN12354_5_I9_R_FLOOR_WALL as EN12354_5_I9_R_FLOOR_WALL
from .building import EN12354_5_I9_R_WALL_FLOOR as EN12354_5_I9_R_WALL_FLOOR
from .building import EN12354_5_I9_R_WALL_WALL as EN12354_5_I9_R_WALL_WALL
from .building import EN12354_5_I9_S_FLOOR as EN12354_5_I9_S_FLOOR
from .building import EN12354_5_I9_S_WALL as EN12354_5_I9_S_WALL
from .building import FORET2011_CARPET_FREQ as FORET2011_CARPET_FREQ
from .building import (
    FORET2011_CARPET_ISO16251_DELTA_L as FORET2011_CARPET_ISO16251_DELTA_L,
)
from .building import (
    FORET2011_CARPET_ISO16251_DELTA_LW as FORET2011_CARPET_ISO16251_DELTA_LW,
)
from .building import HOPKINS_TABLE_A2_H_FC as HOPKINS_TABLE_A2_H_FC
from .building import ISO717_1_ANNEX_C2_EXPECTED as ISO717_1_ANNEX_C2_EXPECTED
from .building import ISO717_1_ANNEX_C2_R_50_5000 as ISO717_1_ANNEX_C2_R_50_5000
from .building import ISO717_1_ANNEX_C_EXPECTED as ISO717_1_ANNEX_C_EXPECTED
from .building import ISO717_1_ANNEX_C_R as ISO717_1_ANNEX_C_R
from .building import (
    ISO717_2_ANNEX_C1_COVERED_EXPECTED as ISO717_2_ANNEX_C1_COVERED_EXPECTED,
)
from .building import ISO717_2_ANNEX_C1_COVERED_LN as ISO717_2_ANNEX_C1_COVERED_LN
from .building import ISO717_2_ANNEX_C1_EXPECTED as ISO717_2_ANNEX_C1_EXPECTED
from .building import ISO717_2_ANNEX_C1_LN as ISO717_2_ANNEX_C1_LN
from .building import ISO717_2_ANNEX_C2_CI_DELTA as ISO717_2_ANNEX_C2_CI_DELTA
from .building import ISO717_2_ANNEX_C2_DELTA_L as ISO717_2_ANNEX_C2_DELTA_L
from .building import ISO717_2_ANNEX_C2_DELTA_LW as ISO717_2_ANNEX_C2_DELTA_LW
from .building import ISO717_2_REFERENCE_FLOOR_CI as ISO717_2_REFERENCE_FLOOR_CI
from .building import ISO717_2_REFERENCE_FLOOR_FREQ as ISO717_2_REFERENCE_FLOOR_FREQ
from .building import ISO717_2_REFERENCE_FLOOR_LN_R0 as ISO717_2_REFERENCE_FLOOR_LN_R0
from .building import (
    ISO717_2_REFERENCE_FLOOR_LN_R0_W as ISO717_2_REFERENCE_FLOOR_LN_R0_W,
)
from .building import ISO9611_FREE_VELOCITY_REFERENCE as ISO9611_FREE_VELOCITY_REFERENCE
from .building import ISO9611_MEAN_EXPECTED as ISO9611_MEAN_EXPECTED
from .building import ISO9611_MEAN_LEVELS as ISO9611_MEAN_LEVELS
from .building import ISO10140_2_REF_AIRBORNE_R as ISO10140_2_REF_AIRBORNE_R
from .building import ISO10140_2_REF_AIRBORNE_RW as ISO10140_2_REF_AIRBORNE_RW
from .building import ISO10140_5_B1_HEAVY_FLOOR_R as ISO10140_5_B1_HEAVY_FLOOR_R
from .building import (
    ISO10140_5_B1_HEAVY_FLOOR_RATING as ISO10140_5_B1_HEAVY_FLOOR_RATING,
)
from .building import ISO10140_5_B1_HEAVY_WALL_R as ISO10140_5_B1_HEAVY_WALL_R
from .building import ISO10140_5_B1_HEAVY_WALL_RATING as ISO10140_5_B1_HEAVY_WALL_RATING
from .building import ISO10140_5_B1_LIGHT_WALL_R as ISO10140_5_B1_LIGHT_WALL_R
from .building import ISO10140_5_B1_LIGHT_WALL_RATING as ISO10140_5_B1_LIGHT_WALL_RATING
from .building import ISO10140_5_C1_FLOOR_C1C2_LN as ISO10140_5_C1_FLOOR_C1C2_LN
from .building import ISO10140_5_C1_FLOOR_C1C2_RATING as ISO10140_5_C1_FLOOR_C1C2_RATING
from .building import ISO10140_5_C1_FLOOR_C3_LN as ISO10140_5_C1_FLOOR_C3_LN
from .building import ISO10140_5_C1_FLOOR_C3_RATING as ISO10140_5_C1_FLOOR_C3_RATING
from .building import ISO10848_ABS_AREA as ISO10848_ABS_AREA
from .building import ISO10848_ABS_C0 as ISO10848_ABS_C0
from .building import ISO10848_ABS_LENGTH_AT_FREF as ISO10848_ABS_LENGTH_AT_FREF
from .building import ISO10848_ABS_TS as ISO10848_ABS_TS
from .building import ISO10848_KIJ_AREA as ISO10848_KIJ_AREA
from .building import ISO10848_KIJ_DBAR as ISO10848_KIJ_DBAR
from .building import ISO10848_KIJ_LIJ as ISO10848_KIJ_LIJ
from .building import ISO10848_KIJ_SIMPLIFIED as ISO10848_KIJ_SIMPLIFIED
from .building import ISO10848_LOSS_FACTOR as ISO10848_LOSS_FACTOR
from .building import ISO12354_2_TABLE_B1 as ISO12354_2_TABLE_B1
from .building import ISO12354_2_TABLE_B2 as ISO12354_2_TABLE_B2
from .building import ISO12354_2_TABLE_B2_BANDS as ISO12354_2_TABLE_B2_BANDS
from .building import ISO12354_ANNEX_G1_CI as ISO12354_ANNEX_G1_CI
from .building import ISO12354_ANNEX_G1_L_PRIME_N as ISO12354_ANNEX_G1_L_PRIME_N
from .building import ISO12354_ANNEX_G1_L_PRIME_N_W as ISO12354_ANNEX_G1_L_PRIME_N_W
from .building import ISO12354_ANNEX_G1_PATHS as ISO12354_ANNEX_G1_PATHS
from .building import ISO12354_ANNEX_G3_LN_SITU as ISO12354_ANNEX_G3_LN_SITU
from .building import ISO12354_ANNEX_G4_DELTA_L as ISO12354_ANNEX_G4_DELTA_L
from .building import ISO12354_ANNEX_G4_LN_DD as ISO12354_ANNEX_G4_LN_DD
from .building import ISO12354_ANNEX_G4_LN_DF as ISO12354_ANNEX_G4_LN_DF
from .building import ISO12354_ANNEX_G10_DELTA_LW as ISO12354_ANNEX_G10_DELTA_LW
from .building import ISO12354_ANNEX_G10_L_PRIME_N_W as ISO12354_ANNEX_G10_L_PRIME_N_W
from .building import ISO12354_ANNEX_G10_LN_EQ_0_W as ISO12354_ANNEX_G10_LN_EQ_0_W
from .building import ISO12354_ANNEX_G10_PATH_LN_W as ISO12354_ANNEX_G10_PATH_LN_W
from .building import ISO12354_ANNEX_L1_PATHS as ISO12354_ANNEX_L1_PATHS
from .building import ISO12354_ANNEX_L1_R_PRIME as ISO12354_ANNEX_L1_R_PRIME
from .building import ISO12354_ANNEX_L1_R_PRIME_W as ISO12354_ANNEX_L1_R_PRIME_W
from .building import ISO12354_ANNEX_L2_SIGMA as ISO12354_ANNEX_L2_SIGMA
from .building import ISO12354_ANNEX_L2_SIGMA_F as ISO12354_ANNEX_L2_SIGMA_F
from .building import ISO12354_ANNEX_L3_ETA as ISO12354_ANNEX_L3_ETA
from .building import (
    ISO12354_ANNEX_L3_PRINTED_EXT_ETA_INT as ISO12354_ANNEX_L3_PRINTED_EXT_ETA_INT,
)
from .building import (
    ISO12354_ANNEX_L3_PRINTED_PERIMETER as ISO12354_ANNEX_L3_PRINTED_PERIMETER,
)
from .building import ISO12354_ANNEX_L3_R_SITU as ISO12354_ANNEX_L3_R_SITU
from .building import ISO12354_ANNEX_L4_ABSORPTION as ISO12354_ANNEX_L4_ABSORPTION
from .building import ISO12354_ANNEX_L4_DELTA as ISO12354_ANNEX_L4_DELTA
from .building import ISO12354_ANNEX_L4_DV as ISO12354_ANNEX_L4_DV
from .building import ISO12354_ANNEX_L10_DELTA_RW as ISO12354_ANNEX_L10_DELTA_RW
from .building import ISO12354_ANNEX_L10_PATH_RW as ISO12354_ANNEX_L10_PATH_RW
from .building import ISO12354_ANNEX_L10_R_PRIME_W as ISO12354_ANNEX_L10_R_PRIME_W
from .building import ISO12354_ANNEX_L10_RW as ISO12354_ANNEX_L10_RW
from .building import ISO12354_ANNEX_L_BANDS as ISO12354_ANNEX_L_BANDS
from .building import ISO12354_ANNEX_L_ELEMENTS as ISO12354_ANNEX_L_ELEMENTS
from .building import ISO12354_ANNEX_L_FLOATING_F0 as ISO12354_ANNEX_L_FLOATING_F0
from .building import ISO12354_ANNEX_L_FLOATING_MASS as ISO12354_ANNEX_L_FLOATING_MASS
from .building import (
    ISO12354_ANNEX_L_FLOATING_STIFFNESS as ISO12354_ANNEX_L_FLOATING_STIFFNESS,
)
from .building import ISO12354_ANNEX_L_FLOOR_ETA_INT as ISO12354_ANNEX_L_FLOOR_ETA_INT
from .building import ISO12354_ANNEX_L_KIJ as ISO12354_ANNEX_L_KIJ
from .building import ISO12354_LIGHTWEIGHT_AREA as ISO12354_LIGHTWEIGHT_AREA
from .building import ISO12354_LIGHTWEIGHT_COUPLING as ISO12354_LIGHTWEIGHT_COUPLING
from .building import ISO12354_TABLE_G11_LN_DD as ISO12354_TABLE_G11_LN_DD
from .building import ISO12354_TABLE_G11_LN_DF as ISO12354_TABLE_G11_LN_DF
from .building import ISO12354_TABLE_G11_LN_TOTAL as ISO12354_TABLE_G11_LN_TOTAL
from .building import ISO12354_TABLE_G11_RATINGS as ISO12354_TABLE_G11_RATINGS
from .building import ISO12354_TABLE_G12_DELTA_LDI as ISO12354_TABLE_G12_DELTA_LDI
from .building import ISO12354_TABLE_G12_DELTA_LI as ISO12354_TABLE_G12_DELTA_LI
from .building import ISO12354_TABLE_G12_LN_BARE as ISO12354_TABLE_G12_LN_BARE
from .building import ISO12354_TABLE_G13_DV as ISO12354_TABLE_G13_DV
from .building import ISO12354_TABLE_G13_R_BARE as ISO12354_TABLE_G13_R_BARE
from .building import ISO12354_TABLE_G13_R_WALL as ISO12354_TABLE_G13_R_WALL
from .building import ISO12354_TABLE_L11_R_DF as ISO12354_TABLE_L11_R_DF
from .building import ISO12354_TABLE_L11_R_FF as ISO12354_TABLE_L11_R_FF
from .building import ISO12354_TABLE_L11_R_PRIME as ISO12354_TABLE_L11_R_PRIME
from .building import ISO12354_TABLE_L11_RATINGS as ISO12354_TABLE_L11_RATINGS
from .building import ISO12354_TABLE_L12_DV_FF as ISO12354_TABLE_L12_DV_FF
from .building import ISO12354_TABLE_L12_R_STAR_WALL as ISO12354_TABLE_L12_R_STAR_WALL
from .building import ISO12354_TABLE_L12_R_WALL as ISO12354_TABLE_L12_R_WALL
from .building import ISO12354_TABLE_L12_RD_FLOOR as ISO12354_TABLE_L12_RD_FLOOR
from .building import ISO12354_TABLE_L13_DELTA_R as ISO12354_TABLE_L13_DELTA_R
from .building import ISO12354_TABLE_L13_DV_DF as ISO12354_TABLE_L13_DV_DF
from .building import ISO12354_TABLE_L13_R_BARE as ISO12354_TABLE_L13_R_BARE
from .building import ISO12354_TABLE_L13_R_STAR_BARE as ISO12354_TABLE_L13_R_STAR_BARE
from .building import ISO12354_TABLE_L14_COUPLING as ISO12354_TABLE_L14_COUPLING
from .building import ISO12354_TABLE_L14_LAB_COUPLING as ISO12354_TABLE_L14_LAB_COUPLING
from .building import (
    ISO12354_TABLE_L14_SEPARATING_AREA as ISO12354_TABLE_L14_SEPARATING_AREA,
)
from .building import ISO12354_TABLE_L15_DNF as ISO12354_TABLE_L15_DNF
from .building import ISO12354_TABLE_L15_R13 as ISO12354_TABLE_L15_R13
from .building import ISO12354_TABLE_L16_DV as ISO12354_TABLE_L16_DV
from .building import ISO12354_TABLE_L16_R13_PRED as ISO12354_TABLE_L16_R13_PRED
from .building import ISO12354_TABLE_L16_R_SITU as ISO12354_TABLE_L16_R_SITU
from .building import ISO12999_1_ANNEX_B_FREQ as ISO12999_1_ANNEX_B_FREQ
from .building import ISO12999_1_ANNEX_B_RI as ISO12999_1_ANNEX_B_RI
from .building import ISO12999_1_ANNEX_B_RW as ISO12999_1_ANNEX_B_RW
from .building import ISO12999_1_ANNEX_B_RW_C50_5000 as ISO12999_1_ANNEX_B_RW_C50_5000
from .building import (
    ISO12999_1_ANNEX_B_RW_CTR50_5000 as ISO12999_1_ANNEX_B_RW_CTR50_5000,
)
from .building import ISO12999_1_ANNEX_B_U_CORR_C as ISO12999_1_ANNEX_B_U_CORR_C
from .building import ISO12999_1_ANNEX_B_U_CORR_CTR as ISO12999_1_ANNEX_B_U_CORR_CTR
from .building import ISO12999_1_ANNEX_B_U_CORR_RW as ISO12999_1_ANNEX_B_U_CORR_RW
from .building import ISO12999_1_ANNEX_B_U_UNCORR_C as ISO12999_1_ANNEX_B_U_UNCORR_C
from .building import ISO12999_1_ANNEX_B_U_UNCORR_CTR as ISO12999_1_ANNEX_B_U_UNCORR_CTR
from .building import ISO12999_1_ANNEX_B_UI as ISO12999_1_ANNEX_B_UI
from .building import ISO12999_1_COVERAGE_K_95 as ISO12999_1_COVERAGE_K_95
from .building import (
    ISO12999_1_RW_A_STANDARD_UNCERTAINTY as ISO12999_1_RW_A_STANDARD_UNCERTAINTY,
)
from .building import (
    ISO12999_1_TABLE2_AIRBORNE_A_1000HZ as ISO12999_1_TABLE2_AIRBORNE_A_1000HZ,
)
from .building import ISO15186_1_KC_B1_PRINTED as ISO15186_1_KC_B1_PRINTED
from .building import ISO15186_1_KC_BANDS as ISO15186_1_KC_BANDS
from .building import ISO15186_1_KC_TABLE_B1 as ISO15186_1_KC_TABLE_B1
from .building import ISO15186_1_REF_LP1 as ISO15186_1_REF_LP1
from .building import ISO15186_1_REF_RI as ISO15186_1_REF_RI
from .building import ISO15186_1_REF_RIW as ISO15186_1_REF_RIW
from .building import ISO15186_1_REF_S as ISO15186_1_REF_S
from .building import ISO15186_1_REF_SM as ISO15186_1_REF_SM
from .building import ISO16283_3_R45_AREA_M2 as ISO16283_3_R45_AREA_M2
from .building import ISO16283_3_R45_EXPECTED_DB as ISO16283_3_R45_EXPECTED_DB
from .building import (
    ISO16283_3_R45_LOUDSPEAKER_CORRECTION_DB as ISO16283_3_R45_LOUDSPEAKER_CORRECTION_DB,
)
from .building import ISO16283_3_R45_RECEIVE_LEVEL_DB as ISO16283_3_R45_RECEIVE_LEVEL_DB
from .building import ISO16283_3_R45_REVERB_TIME_S as ISO16283_3_R45_REVERB_TIME_S
from .building import ISO16283_3_R45_SURFACE_LEVEL_DB as ISO16283_3_R45_SURFACE_LEVEL_DB
from .building import ISO16283_3_R45_VOLUME_M3 as ISO16283_3_R45_VOLUME_M3
from .electroacoustics import CLIPPED_SINE_B1 as CLIPPED_SINE_B1
from .electroacoustics import CLIPPED_SINE_B3 as CLIPPED_SINE_B3
from .electroacoustics import CLIPPED_SINE_B5 as CLIPPED_SINE_B5
from .electroacoustics import CLIPPED_SINE_B7 as CLIPPED_SINE_B7
from .electroacoustics import CLIPPED_SINE_B9 as CLIPPED_SINE_B9
from .electroacoustics import CLIPPED_SINE_THD_F as CLIPPED_SINE_THD_F
from .electroacoustics import COHERENCE_EXPECTED as COHERENCE_EXPECTED
from .electroacoustics import COHERENCE_SNR as COHERENCE_SNR
from .electroacoustics import DISTORTION_D2 as DISTORTION_D2
from .electroacoustics import DISTORTION_HARMONICS as DISTORTION_HARMONICS
from .electroacoustics import DISTORTION_THD_F as DISTORTION_THD_F
from .electroacoustics import DISTORTION_THD_R as DISTORTION_THD_R
from .electroacoustics import ITU_R_468_AES17_OFFSET_DB as ITU_R_468_AES17_OFFSET_DB
from .electroacoustics import ITU_R_468_AES17_ROWS as ITU_R_468_AES17_ROWS
from .electroacoustics import ITU_R_468_AES17_TOL_DB as ITU_R_468_AES17_TOL_DB
from .electroacoustics import (
    ITU_R_468_NETWORK_VS_TABLE1_DB as ITU_R_468_NETWORK_VS_TABLE1_DB,
)
from .electroacoustics import ITU_R_468_TABLE1 as ITU_R_468_TABLE1
from .electroacoustics import (
    ITU_R_468_TABLE1_ROUNDING_DB as ITU_R_468_TABLE1_ROUNDING_DB,
)
from .emission import IEC61043_PHASE_FREQUENCY_HZ as IEC61043_PHASE_FREQUENCY_HZ
from .emission import IEC61043_PHASE_INDEX_DB as IEC61043_PHASE_INDEX_DB
from .emission import IEC61043_PHASE_MISMATCH_DEG as IEC61043_PHASE_MISMATCH_DEG
from .emission import IEC61043_PHASE_SPACING_M as IEC61043_PHASE_SPACING_M
from .emission import IEC61043_TABLE2 as IEC61043_TABLE2
from .emission import ISO3745_C1_REFERENCE as ISO3745_C1_REFERENCE
from .emission import ISO3745_K1_EDGE_BACKGROUND as ISO3745_K1_EDGE_BACKGROUND
from .emission import ISO3745_K1_EDGE_FLOOR as ISO3745_K1_EDGE_FLOOR
from .emission import ISO3745_K1_EDGE_FREQUENCY as ISO3745_K1_EDGE_FREQUENCY
from .emission import ISO3745_K1_EDGE_LEVEL as ISO3745_K1_EDGE_LEVEL
from .emission import ISO3745_U_COVERAGE as ISO3745_U_COVERAGE
from .emission import ISO3745_U_EXPANDED as ISO3745_U_EXPANDED
from .emission import ISO3745_U_SIGMA_OMC as ISO3745_U_SIGMA_OMC
from .emission import ISO3745_U_SIGMA_R0 as ISO3745_U_SIGMA_R0
from .emission import ISO9614_3_UNIFORM_AREAS as ISO9614_3_UNIFORM_AREAS
from .emission import ISO9614_3_UNIFORM_LW as ISO9614_3_UNIFORM_LW
from .emission import ISO9614_3_UNIFORM_POWER as ISO9614_3_UNIFORM_POWER
from .environment import CNOSSOS_A_WEIGHTING_TABLE as CNOSSOS_A_WEIGHTING_TABLE
from .environment import CNOSSOS_RAIL_2015_WAVELENGTHS as CNOSSOS_RAIL_2015_WAVELENGTHS
from .environment import CNOSSOS_RAIL_BANDS as CNOSSOS_RAIL_BANDS
from .environment import CNOSSOS_RAIL_G1A_CAST_IRON as CNOSSOS_RAIL_G1A_CAST_IRON
from .environment import CNOSSOS_RAIL_G1A_COMPOSITE as CNOSSOS_RAIL_G1A_COMPOSITE
from .environment import CNOSSOS_RAIL_G1A_NON_TREAD as CNOSSOS_RAIL_G1A_NON_TREAD
from .environment import CNOSSOS_RAIL_G1B_E as CNOSSOS_RAIL_G1B_E
from .environment import CNOSSOS_RAIL_G1B_M as CNOSSOS_RAIL_G1B_M
from .environment import CNOSSOS_RAIL_G2 as CNOSSOS_RAIL_G2
from .environment import CNOSSOS_RAIL_G3A as CNOSSOS_RAIL_G3A
from .environment import CNOSSOS_RAIL_G3B as CNOSSOS_RAIL_G3B
from .environment import CNOSSOS_RAIL_G3C as CNOSSOS_RAIL_G3C
from .environment import CNOSSOS_RAIL_G4 as CNOSSOS_RAIL_G4
from .environment import CNOSSOS_RAIL_G5 as CNOSSOS_RAIL_G5
from .environment import CNOSSOS_RAIL_G6_A as CNOSSOS_RAIL_G6_A
from .environment import CNOSSOS_RAIL_G6_ALPHA as CNOSSOS_RAIL_G6_ALPHA
from .environment import CNOSSOS_RAIL_G6_B as CNOSSOS_RAIL_G6_B
from .environment import CNOSSOS_RAIL_G7 as CNOSSOS_RAIL_G7
from .environment import CNOSSOS_RAIL_WAVELENGTHS as CNOSSOS_RAIL_WAVELENGTHS
from .environment import (
    CNOSSOS_RAIL_WHEEL_WAVELENGTHS as CNOSSOS_RAIL_WHEEL_WAVELENGTHS,
)
from .environment import CNOSSOS_ROAD_BANDS as CNOSSOS_ROAD_BANDS
from .environment import CNOSSOS_ROAD_TABLE_F1 as CNOSSOS_ROAD_TABLE_F1
from .environment import CNOSSOS_ROAD_TABLE_F2 as CNOSSOS_ROAD_TABLE_F2
from .environment import CNOSSOS_ROAD_TABLE_F3 as CNOSSOS_ROAD_TABLE_F3
from .environment import CNOSSOS_ROAD_TABLE_F4 as CNOSSOS_ROAD_TABLE_F4
from .environment import (
    CNOSSOS_ROAD_TABLE_F4_SPEED_RANGE as CNOSSOS_ROAD_TABLE_F4_SPEED_RANGE,
)
from .environment import CNOSSOS_ROAD_TEMPERATURE_K as CNOSSOS_ROAD_TEMPERATURE_K
from .environment import DIN45681_A1_BANDWIDTHS as DIN45681_A1_BANDWIDTHS
from .environment import DIN45681_I3_KT as DIN45681_I3_KT
from .environment import DIN45681_I3_MEAN_AUDIBILITY as DIN45681_I3_MEAN_AUDIBILITY
from .environment import DIN45681_I6_6FG as DIN45681_I6_6FG
from .environment import DIN45681_I6_6FG_TONE_LEVELS as DIN45681_I6_6FG_TONE_LEVELS
from .environment import DIN45681_I9_FREQUENCIES as DIN45681_I9_FREQUENCIES
from .environment import DIN45681_I9_LEVELS as DIN45681_I9_LEVELS
from .environment import DIN45681_I10_5FG as DIN45681_I10_5FG
from .environment import DIN45681_I10_DECISIVE as DIN45681_I10_DECISIVE
from .environment import DIN45681_I10_K4 as DIN45681_I10_K4
from .environment import DIN45681_I10_K5 as DIN45681_I10_K5
from .environment import DIN45681_I11_J45 as DIN45681_I11_J45
from .environment import DIN45681_I11_J48 as DIN45681_I11_J48
from .environment import DIN45681_LINE_SPACING as DIN45681_LINE_SPACING
from .environment import ISO1996_2_G2_COMBINED as ISO1996_2_G2_COMBINED
from .environment import ISO1996_2_G2_CONTRIBUTIONS as ISO1996_2_G2_CONTRIBUTIONS
from .environment import ISO1996_2_G2_EXPANDED as ISO1996_2_G2_EXPANDED
from .environment import ISO1996_2_TONAL_EXAMPLE3 as ISO1996_2_TONAL_EXAMPLE3
from .environment import ISO1996_2_TONAL_EXAMPLES as ISO1996_2_TONAL_EXAMPLES
from .environment import ISO1996_3_RAMP_ADJUSTMENT as ISO1996_3_RAMP_ADJUSTMENT
from .environment import (
    ISO1996_3_RAMP_LEVEL_DIFFERENCE as ISO1996_3_RAMP_LEVEL_DIFFERENCE,
)
from .environment import ISO1996_3_RAMP_ONSET_RATE as ISO1996_3_RAMP_ONSET_RATE
from .environment import ISO1996_3_RAMP_PROMINENCE as ISO1996_3_RAMP_PROMINENCE
from .environment import ISO9613_1_TABLE1 as ISO9613_1_TABLE1
from .environment import ISO9613_1_TABLE1_CORNER as ISO9613_1_TABLE1_CORNER
from .environment import ISO9613_1_TABLE1_MID as ISO9613_1_TABLE1_MID
from .environment import ISO9613_2_ADIV_100M as ISO9613_2_ADIV_100M
from .environment import ISO9613_2_BARRIER_CAP_DOUBLE as ISO9613_2_BARRIER_CAP_DOUBLE
from .environment import ISO9613_2_BARRIER_CAP_SINGLE as ISO9613_2_BARRIER_CAP_SINGLE
from .environment import (
    ISO9613_2_GROUND_AGR_250_POROUS as ISO9613_2_GROUND_AGR_250_POROUS,
)
from .environment import ISO9613_2_GROUND_BPRIME_ZERO as ISO9613_2_GROUND_BPRIME_ZERO
from .environment import ISO9613_2_TABLE2 as ISO9613_2_TABLE2
from .environment import ISO9613_2_TABLE2_BANDS as ISO9613_2_TABLE2_BANDS
from .environment import ISO20065_ANNEX_E_TONES as ISO20065_ANNEX_E_TONES
from .environment import ISO20065_AV_137 as ISO20065_AV_137
from .environment import ISO20065_AV_592 as ISO20065_AV_592
from .environment import (
    ISO20065_DECISIVE_AUDIBILITIES as ISO20065_DECISIVE_AUDIBILITIES,
)
from .environment import ISO20065_E1_FREQUENCIES as ISO20065_E1_FREQUENCIES
from .environment import ISO20065_E1_LEVELS as ISO20065_E1_LEVELS
from .environment import ISO20065_E1_LS as ISO20065_E1_LS
from .environment import ISO20065_E1_LT as ISO20065_E1_LT
from .environment import ISO20065_E1_LT_FG as ISO20065_E1_LT_FG
from .environment import ISO20065_E1_NOTE1_LS as ISO20065_E1_NOTE1_LS
from .environment import ISO20065_E1_TONE_FREQUENCIES as ISO20065_E1_TONE_FREQUENCIES
from .environment import ISO20065_E1_TONE_LS as ISO20065_E1_TONE_LS
from .environment import ISO20065_E2_AV as ISO20065_E2_AV
from .environment import ISO20065_E2_BAND_LIMITS as ISO20065_E2_BAND_LIMITS
from .environment import ISO20065_E2_FG_TONE_LEVELS as ISO20065_E2_FG_TONE_LEVELS
from .environment import ISO20065_E2_FG_U as ISO20065_E2_FG_U
from .environment import ISO20065_E2_LG as ISO20065_E2_LG
from .environment import ISO20065_E2_U as ISO20065_E2_U
from .environment import ISO20065_E3_FG as ISO20065_E3_FG
from .environment import ISO20065_E3_TONES as ISO20065_E3_TONES
from .environment import ISO20065_E4_DECISIVE_ROWS as ISO20065_E4_DECISIVE_ROWS
from .environment import ISO20065_E4_MEAN_UNCERTAINTY as ISO20065_E4_MEAN_UNCERTAINTY
from .environment import ISO20065_FD_137 as ISO20065_FD_137
from .environment import ISO20065_FD_212 as ISO20065_FD_212
from .environment import ISO20065_LG_137 as ISO20065_LG_137
from .environment import ISO20065_LINE_SPACING as ISO20065_LINE_SPACING
from .environment import ISO20065_MEAN_AUDIBILITY as ISO20065_MEAN_AUDIBILITY
from .environment import NTACOU112_ADJUSTMENT_P10 as NTACOU112_ADJUSTMENT_P10
from .environment import NTACOU112_PROMINENCE as NTACOU112_PROMINENCE
from .environment import (
    cnossos_rail_2015_frequency_tables as cnossos_rail_2015_frequency_tables,
)
from .environment import cnossos_rail_2015_vehicles as cnossos_rail_2015_vehicles
from .environment import (
    cnossos_rail_2015_wavelength_tables as cnossos_rail_2015_wavelength_tables,
)
from .environment import cnossos_rail_workbook_cases as cnossos_rail_workbook_cases
from .environment import (
    cnossos_road_2015_coefficients as cnossos_road_2015_coefficients,
)
from .environment import cnossos_road_2015_surfaces as cnossos_road_2015_surfaces
from .environment import cnossos_road_workbook_cases as cnossos_road_workbook_cases
from .filters import ANSIS14_F5 as ANSIS14_F5
from .filters import ANSIS14_K2 as ANSIS14_K2
from .filters import ANSIS14_TABLE4_B as ANSIS14_TABLE4_B
from .filters import ANSIS14_TABLE5 as ANSIS14_TABLE5
from .filters import IEC537_NASA_TABLE_SLD1 as IEC537_NASA_TABLE_SLD1
from .filters import IEC651_TABLE5 as IEC651_TABLE5
from .filters import IEC61012_AU_HF as IEC61012_AU_HF
from .filters import IEC61012_TABLE1 as IEC61012_TABLE1
from .filters import IEC61012_TABLE2_POLES_HZ as IEC61012_TABLE2_POLES_HZ
from .filters import IEC61260_1995_PASSBAND_MAX as IEC61260_1995_PASSBAND_MAX
from .filters import IEC61260_1995_PASSBAND_MIN as IEC61260_1995_PASSBAND_MIN
from .filters import IEC61260_1995_STOPBAND_MIN as IEC61260_1995_STOPBAND_MIN
from .filters import IEC61260_E34_EXAMPLES as IEC61260_E34_EXAMPLES
from .filters import IEC61260_TABLE_F1 as IEC61260_TABLE_F1
from .filters import IEC61672_TABLE3 as IEC61672_TABLE3
from .filters import INF as INF
from .filters import ISO7196_G_TOLERANCE_DB as ISO7196_G_TOLERANCE_DB
from .filters import ISO7196_TABLE2 as ISO7196_TABLE2
from .filters import LIBROSA_D_WEIGHTING_CONSTS as LIBROSA_D_WEIGHTING_CONSTS
from .hearing import ISO389_7_REF_FREE_1KHZ as ISO389_7_REF_FREE_1KHZ
from .hearing import (
    ISO1999_ANNEX_C_COMPRESSION_FENCE as ISO1999_ANNEX_C_COMPRESSION_FENCE,
)
from .hearing import ISO1999_ANNEX_C_H as ISO1999_ANNEX_C_H
from .hearing import ISO1999_ANNEX_C_H_MEAN as ISO1999_ANNEX_C_H_MEAN
from .hearing import ISO1999_ANNEX_C_HTLAN as ISO1999_ANNEX_C_HTLAN
from .hearing import ISO1999_ANNEX_C_N as ISO1999_ANNEX_C_N
from .hearing import ISO1999_ANNEX_C_N_4K_COMPRESSED as ISO1999_ANNEX_C_N_4K_COMPRESSED
from .hearing import ISO1999_ANNEX_C_N_MEAN as ISO1999_ANNEX_C_N_MEAN
from .hearing import ISO1999_N10_3K_100_40 as ISO1999_N10_3K_100_40
from .hearing import ISO1999_N10_4K_90_20 as ISO1999_N10_4K_90_20
from .hearing import ISO1999_N50_4K_90_20 as ISO1999_N50_4K_90_20
from .hearing import ISO7029_MEDIAN_MALE_60_4KHZ as ISO7029_MEDIAN_MALE_60_4KHZ
from .hearing import ISO7029_SU_MALE_60_1KHZ as ISO7029_SU_MALE_60_1KHZ
from .hearing import ISO9612_ANNEX_D_LEX_8H as ISO9612_ANNEX_D_LEX_8H
from .hearing import ISO9612_ANNEX_D_TASKS as ISO9612_ANNEX_D_TASKS
from .hearing import ISO9612_ANNEX_D_U as ISO9612_ANNEX_D_U
from .hearing import ISO9612_ANNEX_E_LEX_8H as ISO9612_ANNEX_E_LEX_8H
from .hearing import ISO9612_ANNEX_E_SAMPLES as ISO9612_ANNEX_E_SAMPLES
from .hearing import ISO9612_ANNEX_E_TE_HOURS as ISO9612_ANNEX_E_TE_HOURS
from .hearing import ISO9612_ANNEX_E_U as ISO9612_ANNEX_E_U
from .hearing import ISO9612_ANNEX_F_LEX_8H as ISO9612_ANNEX_F_LEX_8H
from .hearing import ISO9612_ANNEX_F_SAMPLES as ISO9612_ANNEX_F_SAMPLES
from .hearing import ISO9612_ANNEX_F_TE_HOURS as ISO9612_ANNEX_F_TE_HOURS
from .hearing import ISO9612_ANNEX_F_U as ISO9612_ANNEX_F_U
from .materials import COX3E_APPENDIX_B_QRD_BANDS as COX3E_APPENDIX_B_QRD_BANDS
from .materials import COX3E_APPENDIX_B_QRD_DN as COX3E_APPENDIX_B_QRD_DN
from .materials import COX3E_APPENDIX_B_TOLERANCE as COX3E_APPENDIX_B_TOLERANCE
from .materials import (
    DIFFUSER_FLAT_NORMALIZED_DIFFUSION as DIFFUSER_FLAT_NORMALIZED_DIFFUSION,
)
from .materials import DIFFUSER_QRD7_MAX_DEPTH as DIFFUSER_QRD7_MAX_DEPTH
from .materials import (
    DIFFUSER_QRD7_NORMALIZED_DIFFUSION_2K as DIFFUSER_QRD7_NORMALIZED_DIFFUSION_2K,
)
from .materials import (
    ISO9053_2_ANNEX_A_BOUNDARY_LAYER as ISO9053_2_ANNEX_A_BOUNDARY_LAYER,
)
from .materials import ISO9053_2_ANNEX_A_FREQUENCY as ISO9053_2_ANNEX_A_FREQUENCY
from .materials import ISO9053_2_ANNEX_A_KAPPA_PRIME as ISO9053_2_ANNEX_A_KAPPA_PRIME
from .materials import ISO9053_2_ANNEX_A_SURFACE as ISO9053_2_ANNEX_A_SURFACE
from .materials import ISO9053_2_ANNEX_A_VOLUME as ISO9053_2_ANNEX_A_VOLUME
from .materials import ISO10534_1_ABSORPTION as ISO10534_1_ABSORPTION
from .materials import (
    ISO10534_1_REFLECTION_MAGNITUDE as ISO10534_1_REFLECTION_MAGNITUDE,
)
from .materials import ISO10534_1_SWR as ISO10534_1_SWR
from .materials import ISO11654_ANNEX_A1_ALPHA_P as ISO11654_ANNEX_A1_ALPHA_P
from .materials import ISO11654_ANNEX_A1_ALPHA_W as ISO11654_ANNEX_A1_ALPHA_W
from .materials import ISO11654_ANNEX_A1_CLASS as ISO11654_ANNEX_A1_CLASS
from .materials import ISO11654_ANNEX_A1_INDICATOR as ISO11654_ANNEX_A1_INDICATOR
from .materials import ISO11654_ANNEX_A2_ALPHA_P as ISO11654_ANNEX_A2_ALPHA_P
from .materials import ISO11654_ANNEX_A2_ALPHA_W as ISO11654_ANNEX_A2_ALPHA_W
from .materials import ISO11654_ANNEX_A2_INDICATOR as ISO11654_ANNEX_A2_INDICATOR
from .materials import ISO12999_2_ALPHA_W_EXAMPLE as ISO12999_2_ALPHA_W_EXAMPLE
from .materials import ISO12999_2_ALPHA_W_U_K2 as ISO12999_2_ALPHA_W_U_K2
from .materials import ISO12999_2_DLALPHA_EXAMPLE as ISO12999_2_DLALPHA_EXAMPLE
from .materials import ISO12999_2_DLALPHA_U_K2 as ISO12999_2_DLALPHA_U_K2
from .materials import ISO12999_2_TABLE4_ALPHA_S as ISO12999_2_TABLE4_ALPHA_S
from .materials import ISO12999_2_TABLE4_FREQ as ISO12999_2_TABLE4_FREQ
from .materials import ISO12999_2_TABLE4_U_K2 as ISO12999_2_TABLE4_U_K2
from .materials import ISO12999_2_TABLE5_ALPHA_P as ISO12999_2_TABLE5_ALPHA_P
from .materials import ISO12999_2_TABLE5_FREQ as ISO12999_2_TABLE5_FREQ
from .materials import ISO12999_2_TABLE5_U_K2 as ISO12999_2_TABLE5_U_K2
from .materials import ISO13472_1_KR as ISO13472_1_KR
from .materials import ISO13472_1_MSA_RADIUS as ISO13472_1_MSA_RADIUS
from .materials import ISO13472_1_MSA_WINDOW as ISO13472_1_MSA_WINDOW
from .materials import ISO13472_2_SPOT_DIAMETER as ISO13472_2_SPOT_DIAMETER
from .materials import ISO13472_2_SPOT_FU as ISO13472_2_SPOT_FU
from .materials import ISO13472_2_SPOT_SPEED as ISO13472_2_SPOT_SPEED
from .materials import ISO17497_1_A5_ALPHA_S as ISO17497_1_A5_ALPHA_S
from .materials import ISO17497_1_A5_ALPHA_SPEC as ISO17497_1_A5_ALPHA_SPEC
from .materials import ISO17497_1_A5_U_ALPHA_S as ISO17497_1_A5_U_ALPHA_S
from .materials import ISO17497_1_A5_U_ALPHA_SPEC as ISO17497_1_A5_U_ALPHA_SPEC
from .materials import ISO17497_1_A5_U_SCATTERING as ISO17497_1_A5_U_SCATTERING
from .materials import ISO17497_1_CHAIN_ALPHA_S as ISO17497_1_CHAIN_ALPHA_S
from .materials import ISO17497_1_CHAIN_ALPHA_SPEC as ISO17497_1_CHAIN_ALPHA_SPEC
from .materials import ISO17497_1_CHAIN_C as ISO17497_1_CHAIN_C
from .materials import ISO17497_1_CHAIN_S as ISO17497_1_CHAIN_S
from .materials import ISO17497_1_CHAIN_SCATTERING as ISO17497_1_CHAIN_SCATTERING
from .materials import ISO17497_1_CHAIN_T as ISO17497_1_CHAIN_T
from .materials import ISO17497_1_CHAIN_V as ISO17497_1_CHAIN_V
from .materials import ISO17497_1_SPEED_OF_SOUND_20C as ISO17497_1_SPEED_OF_SOUND_20C
from .materials import ISO17497_2_AREA_FACTOR_ZENITH as ISO17497_2_AREA_FACTOR_ZENITH
from .materials import ISO17497_2_FLAT_DIFFUSION as ISO17497_2_FLAT_DIFFUSION
from .materials import ISO17497_2_FLAT_LEVELS as ISO17497_2_FLAT_LEVELS
from .materials import (
    ISO17497_2_NORMALIZED_DIFFUSION as ISO17497_2_NORMALIZED_DIFFUSION,
)
from .materials import (
    ISO17497_2_PREDICTION_FREQUENCY as ISO17497_2_PREDICTION_FREQUENCY,
)
from .materials import (
    ISO17497_2_QRD_DESIGN_FREQUENCY as ISO17497_2_QRD_DESIGN_FREQUENCY,
)
from .materials import ISO17497_2_QRD_DIFFUSION as ISO17497_2_QRD_DIFFUSION
from .materials import ISO17497_2_QRD_LEVELS as ISO17497_2_QRD_LEVELS
from .materials import ISO17497_2_QRD_MAX_DEPTH as ISO17497_2_QRD_MAX_DEPTH
from .materials import ISO17497_2_QRD_N as ISO17497_2_QRD_N
from .materials import ISO17497_2_QRD_PERIODS as ISO17497_2_QRD_PERIODS
from .materials import ISO17497_2_QRD_TOTAL_WIDTH as ISO17497_2_QRD_TOTAL_WIDTH
from .materials import ISO17497_2_QRD_WELL_WIDTH as ISO17497_2_QRD_WELL_WIDTH
from .materials import ISO17497_2_SPEED_OF_SOUND as ISO17497_2_SPEED_OF_SOUND
from .materials import MAA_FIG5_CAVITY as MAA_FIG5_CAVITY
from .materials import MAA_FIG5_DIAMETER as MAA_FIG5_DIAMETER
from .materials import MAA_FIG5_SEPARATION as MAA_FIG5_SEPARATION
from .materials import MAA_FIG5_THICKNESS as MAA_FIG5_THICKNESS
from .materials import MAA_TABLE_I_ALPHA0 as MAA_TABLE_I_ALPHA0
from .materials import MAA_TABLE_I_BANDWIDTH as MAA_TABLE_I_BANDWIDTH
from .materials import MAA_TABLE_I_R as MAA_TABLE_I_R
from .materials import POROUS_DB_K_EXPECTED as POROUS_DB_K_EXPECTED
from .materials import POROUS_DB_X_POINT as POROUS_DB_X_POINT
from .materials import POROUS_DB_ZC_EXPECTED as POROUS_DB_ZC_EXPECTED
from .materials import POROUS_MIKI_K_EXPECTED as POROUS_MIKI_K_EXPECTED
from .materials import POROUS_MIKI_Y_POINT as POROUS_MIKI_Y_POINT
from .materials import POROUS_MIKI_ZC_EXPECTED as POROUS_MIKI_ZC_EXPECTED
from .materials import POROUS_STATISTICAL_ALPHA_MAX as POROUS_STATISTICAL_ALPHA_MAX
from .metrology import GUM_ADDITIVE_UC as GUM_ADDITIVE_UC
from .metrology import GUM_COVERAGE_K99_16 as GUM_COVERAGE_K99_16
from .metrology import GUM_H1_CONTRIBUTIONS as GUM_H1_CONTRIBUTIONS
from .metrology import GUM_H1_INPUTS as GUM_H1_INPUTS
from .metrology import GUM_H1_U99 as GUM_H1_U99
from .metrology import GUM_H1_UC as GUM_H1_UC
from .metrology import GUM_H1_VALUE as GUM_H1_VALUE
from .metrology import GUM_H1_VEFF as GUM_H1_VEFF
from .metrology import GUM_H2_OBSERVATIONS as GUM_H2_OBSERVATIONS
from .metrology import GUM_H2_RESULTS as GUM_H2_RESULTS
from .metrology import GUM_WELCH_VEFF as GUM_WELCH_VEFF
from .metrology import GUMS1_TABLE2_INTERVAL_95 as GUMS1_TABLE2_INTERVAL_95
from .metrology import GUMS1_TABLE3_INTERVAL_95 as GUMS1_TABLE3_INTERVAL_95
from .metrology import GUMS1_TABLE3_U as GUMS1_TABLE3_U
from .psychoacoustics import ECMA418_1_DFC_1KHZ as ECMA418_1_DFC_1KHZ
from .psychoacoustics import ECMA418_1_DFC_500HZ as ECMA418_1_DFC_500HZ
from .psychoacoustics import ECMA418_1_F1_1KHZ as ECMA418_1_F1_1KHZ
from .psychoacoustics import ECMA418_1_F2_1KHZ as ECMA418_1_F2_1KHZ
from .psychoacoustics import ECMA418_1_PROX_150HZ as ECMA418_1_PROX_150HZ
from .psychoacoustics import ECMA418_1_PROX_850HZ as ECMA418_1_PROX_850HZ
from .psychoacoustics import (
    ECMA418_2_AUDIBILITY_THRESHOLD_SONE as ECMA418_2_AUDIBILITY_THRESHOLD_SONE,
)
from .psychoacoustics import (
    ECMA418_2_FLUCTUATION_1KHZ_4HZ_60DB_VACIL as ECMA418_2_FLUCTUATION_1KHZ_4HZ_60DB_VACIL,
)
from .psychoacoustics import ECMA418_2_FLUCTUATION_C_F as ECMA418_2_FLUCTUATION_C_F
from .psychoacoustics import (
    ECMA418_2_LOUDNESS_1KHZ_40DB_SONE as ECMA418_2_LOUDNESS_1KHZ_40DB_SONE,
)
from .psychoacoustics import ECMA418_2_LOUDNESS_C_N as ECMA418_2_LOUDNESS_C_N
from .psychoacoustics import (
    ECMA418_2_PROMINENT_FLUCTUATION_VACIL as ECMA418_2_PROMINENT_FLUCTUATION_VACIL,
)
from .psychoacoustics import (
    ECMA418_2_PROMINENT_ROUGHNESS_ASPER as ECMA418_2_PROMINENT_ROUGHNESS_ASPER,
)
from .psychoacoustics import (
    ECMA418_2_PROMINENT_TONALITY_TU as ECMA418_2_PROMINENT_TONALITY_TU,
)
from .psychoacoustics import (
    ECMA418_2_ROUGHNESS_1KHZ_70HZ_60DB_ASPER as ECMA418_2_ROUGHNESS_1KHZ_70HZ_60DB_ASPER,
)
from .psychoacoustics import ECMA418_2_ROUGHNESS_C_R as ECMA418_2_ROUGHNESS_C_R
from .psychoacoustics import (
    ECMA418_2_TONALITY_1KHZ_40DB_TU as ECMA418_2_TONALITY_1KHZ_40DB_TU,
)
from .psychoacoustics import ECMA418_2_TONALITY_C_T as ECMA418_2_TONALITY_C_T
from .psychoacoustics import FS_AM_BBN_60DB_LITERATURE as FS_AM_BBN_60DB_LITERATURE
from .psychoacoustics import FS_AM_TONE_70DB_LITERATURE as FS_AM_TONE_70DB_LITERATURE
from .psychoacoustics import FS_AM_TONE_FMOD_HZ as FS_AM_TONE_FMOD_HZ
from .psychoacoustics import FS_BBN_60_1_4 as FS_BBN_60_1_4
from .psychoacoustics import FS_CALIBRATION_VACIL as FS_CALIBRATION_VACIL
from .psychoacoustics import FS_CARRIER_SWEEP_HZ as FS_CARRIER_SWEEP_HZ
from .psychoacoustics import FS_CARRIER_SWEEP_VACIL as FS_CARRIER_SWEEP_VACIL
from .psychoacoustics import ISO226_2023_TABLE_B1_ANCHOR as ISO226_2023_TABLE_B1_ANCHOR
from .psychoacoustics import (
    ISO532_2_ANCHOR_1KHZ_40DB_SONE as ISO532_2_ANCHOR_1KHZ_40DB_SONE,
)
from .psychoacoustics import ISO532_2_C as ISO532_2_C
from .psychoacoustics import (
    ISO532_3_ANCHOR_1KHZ_40DB_SONE as ISO532_3_ANCHOR_1KHZ_40DB_SONE,
)
from .psychoacoustics import PA_WORKED_INPUT as PA_WORKED_INPUT
from .psychoacoustics import PA_WORKED_VALUE as PA_WORKED_VALUE
from .psychoacoustics import PA_WORKED_WFR as PA_WORKED_WFR
from .psychoacoustics import PA_WORKED_WS as PA_WORKED_WS
from .room import ANSIS12_2_NC40_SELF as ANSIS12_2_NC40_SELF
from .room import ANSIS12_2_RC31_63HZ as ANSIS12_2_RC31_63HZ
from .room import ANSIS12_2_RC35_LMF as ANSIS12_2_RC35_LMF
from .room import EN12354_6_A_BARE as EN12354_6_A_BARE
from .room import EN12354_6_A_OBJECTS as EN12354_6_A_OBJECTS
from .room import EN12354_6_ANNEX_E_BARE_SURFACES as EN12354_6_ANNEX_E_BARE_SURFACES
from .room import EN12354_6_ANNEX_E_VOLUME as EN12354_6_ANNEX_E_VOLUME
from .room import EN12354_6_T_BARE as EN12354_6_T_BARE
from .room import EVEREST_EX1_BANDS as EVEREST_EX1_BANDS
from .room import EVEREST_EX1_FLOOR_ALPHA as EVEREST_EX1_FLOOR_ALPHA
from .room import EVEREST_EX1_FLOOR_AREA as EVEREST_EX1_FLOOR_AREA
from .room import EVEREST_EX1_RT as EVEREST_EX1_RT
from .room import EVEREST_EX1_SHELL_ALPHA as EVEREST_EX1_SHELL_ALPHA
from .room import EVEREST_EX1_SHELL_AREA as EVEREST_EX1_SHELL_AREA
from .room import EVEREST_EX1_VOLUME as EVEREST_EX1_VOLUME
from .speech import ANSIS3_5_ANNEX_C1 as ANSIS3_5_ANNEX_C1
from .speech import (
    ANSIS3_5_ANNEX_C1_LEVEL_DISTORTION_I5 as ANSIS3_5_ANNEX_C1_LEVEL_DISTORTION_I5,
)
from .speech import ANSIS3_5_ANNEX_C1_NOISE as ANSIS3_5_ANNEX_C1_NOISE
from .speech import ANSIS3_5_ANNEX_C1_SPEECH as ANSIS3_5_ANNEX_C1_SPEECH
from .speech import ANSIS3_5_ANNEX_C2 as ANSIS3_5_ANNEX_C2
from .speech import ANSIS3_5_ANNEX_C2_MASKING as ANSIS3_5_ANNEX_C2_MASKING
from .speech import ANSIS3_5_BAND_IMPORTANCE_SUM as ANSIS3_5_BAND_IMPORTANCE_SUM
from .speech import ANSIS3_5_CRITICAL_IMPORTANCE_SUM as ANSIS3_5_CRITICAL_IMPORTANCE_SUM
from .speech import ANSIS3_5_CRITICAL_TABLE1 as ANSIS3_5_CRITICAL_TABLE1
from .speech import ANSIS3_5_DISTURBANCE_5000HZ as ANSIS3_5_DISTURBANCE_5000HZ
from .speech import ANSIS3_5_EQUAL_IMPORTANCE_SUM as ANSIS3_5_EQUAL_IMPORTANCE_SUM
from .speech import ANSIS3_5_LOUD_1KHZ as ANSIS3_5_LOUD_1KHZ
from .speech import ANSIS3_5_MASKING_Z_200HZ as ANSIS3_5_MASKING_Z_200HZ
from .speech import ANSIS3_5_NOISE_PLUS_LOSS as ANSIS3_5_NOISE_PLUS_LOSS
from .speech import ANSIS3_5_OCTAVE_IMPORTANCE_SUM as ANSIS3_5_OCTAVE_IMPORTANCE_SUM
from .speech import ANSIS3_5_OCTAVE_TABLE4 as ANSIS3_5_OCTAVE_TABLE4
from .speech import ANSIS3_5_OCTAVE_TABLE4_SHARED as ANSIS3_5_OCTAVE_TABLE4_SHARED
from .speech import ANSIS3_5_STANDARD_QUIET as ANSIS3_5_STANDARD_QUIET
from .speech import ANSIS3_5_THIRD_OCTAVE_TABLE3 as ANSIS3_5_THIRD_OCTAVE_TABLE3
from .speech import ANSIS3_5_WG_CB1_IMPORTANCE as ANSIS3_5_WG_CB1_IMPORTANCE
from .speech import ANSIS3_5_WG_CB1_SII as ANSIS3_5_WG_CB1_SII
from .speech import ANSIS3_5_WG_CB1_SII_EXACT as ANSIS3_5_WG_CB1_SII_EXACT
from .speech import ANSIS3_5_WG_CB_NOISE as ANSIS3_5_WG_CB_NOISE
from .speech import ANSIS3_5_WG_CB_SII as ANSIS3_5_WG_CB_SII
from .speech import ANSIS3_5_WG_CB_SII_EXACT as ANSIS3_5_WG_CB_SII_EXACT
from .speech import ANSIS3_5_WG_CB_SPEECH as ANSIS3_5_WG_CB_SPEECH
from .speech import ANSIS3_5_WG_CB_THRESHOLD as ANSIS3_5_WG_CB_THRESHOLD
from .speech import ANSIS3_5_WG_ECB1_IMPORTANCE as ANSIS3_5_WG_ECB1_IMPORTANCE
from .speech import ANSIS3_5_WG_ECB1_SII as ANSIS3_5_WG_ECB1_SII
from .speech import ANSIS3_5_WG_ECB1_SII_EXACT as ANSIS3_5_WG_ECB1_SII_EXACT
from .speech import ANSIS3_5_WG_ECB_NOISE as ANSIS3_5_WG_ECB_NOISE
from .speech import ANSIS3_5_WG_ECB_SII as ANSIS3_5_WG_ECB_SII
from .speech import ANSIS3_5_WG_ECB_SII_EXACT as ANSIS3_5_WG_ECB_SII_EXACT
from .speech import ANSIS3_5_WG_ECB_SPEECH as ANSIS3_5_WG_ECB_SPEECH
from .speech import ANSIS3_5_WG_ECB_THRESHOLD as ANSIS3_5_WG_ECB_THRESHOLD
from .speech import ANSIS3_5_WG_FLAT_CASES as ANSIS3_5_WG_FLAT_CASES
from .speech import ANSIS3_5_WG_OCTAVE1_IMPORTANCE as ANSIS3_5_WG_OCTAVE1_IMPORTANCE
from .speech import ANSIS3_5_WG_OCTAVE1_SII as ANSIS3_5_WG_OCTAVE1_SII
from .speech import ANSIS3_5_WG_OCTAVE1_SII_EXACT as ANSIS3_5_WG_OCTAVE1_SII_EXACT
from .speech import ANSIS3_5_WG_OCTAVE_NOISE as ANSIS3_5_WG_OCTAVE_NOISE
from .speech import ANSIS3_5_WG_OCTAVE_SII as ANSIS3_5_WG_OCTAVE_SII
from .speech import ANSIS3_5_WG_OCTAVE_SII_EXACT as ANSIS3_5_WG_OCTAVE_SII_EXACT
from .speech import ANSIS3_5_WG_OCTAVE_SPEECH as ANSIS3_5_WG_OCTAVE_SPEECH
from .speech import ANSIS3_5_WG_OCTAVE_THRESHOLD as ANSIS3_5_WG_OCTAVE_THRESHOLD
from .speech import ANSIS3_5_WG_TO1_IMPORTANCE as ANSIS3_5_WG_TO1_IMPORTANCE
from .speech import ANSIS3_5_WG_TO1_SII as ANSIS3_5_WG_TO1_SII
from .speech import ANSIS3_5_WG_TO1_SII_EXACT as ANSIS3_5_WG_TO1_SII_EXACT
from .speech import ANSIS3_5_WG_TO_NOISE as ANSIS3_5_WG_TO_NOISE
from .speech import ANSIS3_5_WG_TO_SII as ANSIS3_5_WG_TO_SII
from .speech import ANSIS3_5_WG_TO_SII_EXACT as ANSIS3_5_WG_TO_SII_EXACT
from .speech import ANSIS3_5_WG_TO_SPEECH as ANSIS3_5_WG_TO_SPEECH
from .speech import ANSIS3_5_WG_TO_THRESHOLD as ANSIS3_5_WG_TO_THRESHOLD
from .speech import IEC60268_16_ANNEX_M_AMBIENT as IEC60268_16_ANNEX_M_AMBIENT
from .speech import IEC60268_16_ANNEX_M_LEVEL as IEC60268_16_ANNEX_M_LEVEL
from .speech import IEC60268_16_ANNEX_M_MTF as IEC60268_16_ANNEX_M_MTF
from .speech import IEC60268_16_ANNEX_M_MTI as IEC60268_16_ANNEX_M_MTI
from .speech import IEC60268_16_ANNEX_M_STI as IEC60268_16_ANNEX_M_STI
from .underwater import UW_REFERENCE_OFFSET_DB as UW_REFERENCE_OFFSET_DB
from .vibration import DIRECTIVE_2002_44_HAV_EAV as DIRECTIVE_2002_44_HAV_EAV
from .vibration import DIRECTIVE_2002_44_HAV_ELV as DIRECTIVE_2002_44_HAV_ELV
from .vibration import DIRECTIVE_2002_44_WBV_EAV as DIRECTIVE_2002_44_WBV_EAV
from .vibration import DIRECTIVE_2002_44_WBV_ELV as DIRECTIVE_2002_44_WBV_ELV
from .vibration import ISO2631_5_ANNEX_D_A as ISO2631_5_ANNEX_D_A
from .vibration import ISO2631_5_ANNEX_D_B as ISO2631_5_ANNEX_D_B
from .vibration import ISO2631_5_ANNEX_D_FS as ISO2631_5_ANNEX_D_FS
from .vibration import ISO2631_5_DZD_MALE as ISO2631_5_DZD_MALE
from .vibration import ISO2631_5_PI_MALE as ISO2631_5_PI_MALE
from .vibration import ISO2631_5_R_FEMALE as ISO2631_5_R_FEMALE
from .vibration import ISO2631_5_R_MALE as ISO2631_5_R_MALE
from .vibration import ISO2631_5_SD_FEMALE as ISO2631_5_SD_FEMALE
from .vibration import ISO5349_1_VWF_A8 as ISO5349_1_VWF_A8
from .vibration import ISO5349_1_VWF_DY_YEARS as ISO5349_1_VWF_DY_YEARS
from .vibration import ISO5349_2_E3_A8 as ISO5349_2_E3_A8
from .vibration import ISO5349_2_E21_A8 as ISO5349_2_E21_A8
from .vibration import ISO7626_1_DECADE_COMPLIANCE as ISO7626_1_DECADE_COMPLIANCE
from .vibration import ISO7626_1_DECADE_FREQ_HZ as ISO7626_1_DECADE_FREQ_HZ
from .vibration import ISO7626_1_DECADE_MOBILITY as ISO7626_1_DECADE_MOBILITY
from .vibration import ISO7626_2_CAL_ACCELERANCE as ISO7626_2_CAL_ACCELERANCE
from .vibration import ISO7626_2_CAL_MASS_KG as ISO7626_2_CAL_MASS_KG
from .vibration import ISO7626_2_CAL_MOBILITY_100HZ as ISO7626_2_CAL_MOBILITY_100HZ
from .vibration import ISO7626_2_RANDOM_ERROR_PCT as ISO7626_2_RANDOM_ERROR_PCT
from .vibration import ISO8041_1_ANNEX_B_FACTORS as ISO8041_1_ANNEX_B_FACTORS
from .vibration import ISO8041_1_TABLE4_TRANSITIONS as ISO8041_1_TABLE4_TRANSITIONS
from .vibration import ISO8041_1_TABLE5_TOLERANCES as ISO8041_1_TABLE5_TOLERANCES
from .vibration import ISO8041_1_WB_FACTOR_1HZ as ISO8041_1_WB_FACTOR_1HZ
from .vibration import ISO8041_1_WB_FACTOR_6P31HZ as ISO8041_1_WB_FACTOR_6P31HZ
from .vibration import ISO8041_1_WB_FACTOR_100HZ as ISO8041_1_WB_FACTOR_100HZ
from .vibration import ISO8041_1_WBV_REF_FREQ_HZ as ISO8041_1_WBV_REF_FREQ_HZ
from .vibration import ISO8041_1_WC_REF_FACTOR as ISO8041_1_WC_REF_FACTOR
from .vibration import ISO8041_1_WD_FACTOR_1HZ as ISO8041_1_WD_FACTOR_1HZ
from .vibration import ISO8041_1_WD_REF_FACTOR as ISO8041_1_WD_REF_FACTOR
from .vibration import ISO8041_1_WE_FACTOR_8HZ as ISO8041_1_WE_FACTOR_8HZ
from .vibration import ISO8041_1_WF_FACTOR_0P1HZ as ISO8041_1_WF_FACTOR_0P1HZ
from .vibration import ISO8041_1_WF_FACTOR_0P1585HZ as ISO8041_1_WF_FACTOR_0P1585HZ
from .vibration import ISO8041_1_WH_REF_FACTOR as ISO8041_1_WH_REF_FACTOR
from .vibration import ISO8041_1_WH_REF_FREQ_HZ as ISO8041_1_WH_REF_FREQ_HZ
from .vibration import ISO8041_1_WJ_FACTOR_6P31HZ as ISO8041_1_WJ_FACTOR_6P31HZ
from .vibration import ISO8041_1_WJ_FACTOR_8HZ as ISO8041_1_WJ_FACTOR_8HZ
from .vibration import ISO8041_1_WK_FACTOR_6P31HZ as ISO8041_1_WK_FACTOR_6P31HZ
from .vibration import ISO8041_1_WM_FACTOR_1P585HZ as ISO8041_1_WM_FACTOR_1P585HZ
from .vibration import ISO10846_1_EQ6_FORCE_RATIO as ISO10846_1_EQ6_FORCE_RATIO
from .vibration import ISO10846_3_ACCURACY_DB as ISO10846_3_ACCURACY_DB
from .vibration import ISO10846_3_ACCURACY_FRACTION as ISO10846_3_ACCURACY_FRACTION
from .vibration import ISO10846_3_LIMIT_BIAS_RATIO as ISO10846_3_LIMIT_BIAS_RATIO
from .vibration import ISO10846_3_LIMIT_DELTA_L_DB as ISO10846_3_LIMIT_DELTA_L_DB
from .vibration import ISO10846_LINEARITY_STEP_DB as ISO10846_LINEARITY_STEP_DB
from .vibration import ISO10846_LINEARITY_TOL_DB as ISO10846_LINEARITY_TOL_DB
