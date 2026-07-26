---
title: "emission.declaration"
description: "ISO 4871:1996 declaration of noise emission values of machinery and equipment."
sidebar:
  label: "declaration"
---

ISO 4871:1996 declaration of noise emission values of machinery and equipment.

ISO 4871 is the standardised *noise-emission declaration*: the information a
manufacturer or supplier states in technical documents about the airborne noise
emitted by a machine (clause 3.13). It does not measure anything itself; it
prescribes which quantities are declared, in which of two alternative forms, and
how a declared value is verified (clauses 4 to 6).

The quantities are the A-weighted sound power level `L_WA` (the preferred and
basic quantity, ISO 4871 Note 17) and, optionally, the A-weighted emission sound
pressure level `L_pA` at a work station (clause 3.11). Each is declared in one
of two forms selected by the relevant noise test code (clause 4):

* the **dual-number** form (clause 3.16): a measured noise emission value `L`
  and its associated uncertainty `K`, stated together but separately, both
  rounded to the nearest decibel; and
* the **single-number** form (clause 3.15): the derived declared value
  `L_d`, the *sum* `L + K` of the unrounded quantities rounded once to the
  nearest decibel, an upper limit which values from repeated measurements are
  unlikely to exceed at the stated confidence level.

`K` combines the measurement uncertainty (reproducibility) and, for a batch,
the production spread (clauses 3.20 to 3.24; `K = 1,645 sigma_R` for a single
machine, Annex A.2.2). Verification (clause 6) compares a verification
measurement `L_1` against the declared form: for a single machine the
combined form is verified when `L_1 <= L_d` and the dual-number form when
`L_1 <= (L + K)`, the sum of the separately rounded declared values
(clause 6.2).

This module models a declaration as [`NoiseEmissionDeclaration`](/phonometry/reference/api/power/declaration/#noiseemissiondeclaration), a set of
[`OperatingModeDeclaration`](/phonometry/reference/api/power/declaration/#operatingmodedeclaration) values (one per operating mode, clause 4), and
renders it as an ISO 4871 declaration fiche through `.report`. A
declaration is most often built from a measured sound power via
[`declare`](/phonometry/reference/api/power/sound-power/).

> Auto-generated from the source docstrings by `scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand.

## NoiseEmissionDeclaration

```python
NoiseEmissionDeclaration(
    modes: Sequence[OperatingModeDeclaration],
    machine: str | None = None,
    operating_conditions: str | None = None,
    noise_test_code: str | None = None,
    basic_standards: Sequence[str] = ...,
    form: DeclarationForm = 'dual-number',
)
```

An ISO 4871:1996 declaration of noise emission values (clauses 4 to 6).

A declaration is one or more [`OperatingModeDeclaration`](/phonometry/reference/api/power/declaration/#operatingmodedeclaration) values (one
per operating mode, clause 4) plus the accompanying information required by
clause 5: identification of the machinery (clause 5 a), the noise test code
and basic standards used (clause 5 b) and the operating conditions
(clause 5 c). `report` renders it as a one-page ISO 4871 declaration
fiche.

**Attributes**

| Name | Description |
| :--- | :--- |
| `modes` | One or more per-operating-mode declarations. A sequence is accepted and stored as a tuple. |
| `machine` | Machine identification (model number and other identifying information, clause 5 a). |
| `operating_conditions` | Operating and mounting conditions the values refer to (clause 5 c), e.g. `"50 Hz, 230 V, rated load"`. |
| `noise_test_code` | The noise test code the values were determined to (clause 5 b), e.g. an ISO family-specific test code; `None` when none exists. |
| `basic_standards` | The basic noise-emission standard(s) used to obtain the values (clause 5 b), e.g. `("ISO 3744",)`. A single string is accepted and wrapped in a one-tuple. |
| `form` | Which declaration form the fiche presents, `"dual-number"` (default, clause 3.16: `L` and `K` separately) or `"single-number"` (clause 3.15: the derived `L_d = L + K`). |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If no operating mode is given or `form` is unknown. |

### NoiseEmissionDeclaration.report()

```python
NoiseEmissionDeclaration.report(
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    engine: str = 'reportlab',
    verbose: bool = False,
    language: str = 'en',
) -> str
```

Render an ISO 4871 noise-emission declaration fiche to a PDF.

Writes a one-page declaration data sheet: the standard-basis line
(ISO 4871:1996 and the cited basic emission standard), the machine
identification and operating conditions, the declared dual- or
single-number table across the operating-mode columns following the
ISO 4871 Annex B layouts (`L_WA` and `K_WA` for the dual-number
form, the derived `L_WAd = L_WA + K_WA` for the single-number form,
plus the emission sound pressure level when declared), the
noise-test-code and basic-standards footnote, a verification verdict
table when a verification measurement is supplied (clause 6.2, against
`L_WAd` for the single-number form or the separately rounded
`L_WA + K_WA` for the dual-number form), and the footer
identity/disclaimer block.

**Parameters**

| Name | Description |
| :--- | :--- |
| `path` | Destination path of the PDF file. |
| `metadata` | Optional [`ReportMetadata`](/phonometry/reference/api/building/insulation/#reportmetadata) supplying the laboratory identity (footer) and, through `measurement_standard`, the basic emission standard shown in the standard-basis line when `basic_standards` is empty. |
| `engine` | Rendering back end; only `"reportlab"` is supported. |
| `verbose` | Accepted for a uniform `.report()` signature; the declaration fiche has a single table layout, so it has no effect. |
| `language` | Fiche language: `"en"` (default, English) or `"es"` (Spanish, with a comma decimal separator). |

**Returns:** The written `path` as a `str`.

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If `engine` is not `"reportlab"` or `language` is unknown. |
| ImportError | If reportlab is not installed (`pip install phonometry[report]`). |

## OperatingModeDeclaration

```python
OperatingModeDeclaration(
    mode: str,
    sound_power_level: float,
    sound_power_uncertainty: float,
    emission_pressure_level: float | None = None,
    emission_pressure_uncertainty: float | None = None,
    verification_level: float | None = None,
)
```

Declared dual-number noise-emission values for one operating mode.

Holds the measured A-weighted sound power level `L_WA` and its uncertainty
`K_WA` (ISO 4871 clause 3.16), and optionally the A-weighted emission sound
pressure level `L_pA` at a work station with its uncertainty `K_pA`
(clause 3.11). The derived declared single-number values follow from
`L_d = L + K` (clause 3.15), the sum rounded to the nearest decibel. When a
verification measurement `L_1` (an A-weighted sound power level determined
for verification, clause 6) is supplied, `verified` applies the
single-machine criterion of clause 6.2 for the combined form (verified when
`L_1 <= L_WAd`) and `verified_dual` the dual-number one (verified
when `L_1 <= round(L_WA) + round(K_WA)`).

**Attributes**

| Name | Description |
| :--- | :--- |
| `mode` | Operating-mode label printed as the table column header (e.g. `"Operating mode 1"`); it identifies the operating mode of clause 5 c. |
| `sound_power_level` | Measured mean A-weighted sound power level `L_WA`, in decibels re 1 pW. |
| `sound_power_uncertainty` | Uncertainty `K_WA` of the sound power level, in decibels (finite and non-negative). |
| `emission_pressure_level` | A-weighted emission sound pressure level `L_pA` at the work station, in decibels re 20 uPa; `None` when only the sound power level is declared. |
| `emission_pressure_uncertainty` | Uncertainty `K_pA` of the emission sound pressure level, in decibels; required when `emission_pressure_level` is given and forbidden otherwise. |
| `verification_level` | A verification measurement `L_1` of the A-weighted sound power level (clause 6), in decibels; `None` when the mode is not verified. |

**Raises**

| Exception | When |
| :--- | :--- |
| ValueError | If a level is not finite, an uncertainty is not finite and non-negative, or only one member of the emission-pressure pair is given. |

### OperatingModeDeclaration.declared_emission_pressure_level

*property*

Declared single-number emission pressure level `L_pAd = L_pA + K_pA`.

The sum `L_pA + K_pA` rounded once to the nearest decibel (clause
3.15); `None` when no emission sound pressure level is declared for
the mode.

### OperatingModeDeclaration.declared_sound_power_level

*property*

Declared single-number sound power level `L_WAd = L_WA + K_WA` (3.15).

Clause 3.15 defines the declared value as the *sum* of the measured
value and its uncertainty, rounded to the nearest decibel: the sum is
formed from the unrounded quantities (clause 3.12) and rounded once,
not assembled from the separately rounded dual-number values (3.16).

### OperatingModeDeclaration.dual_number_verification_limit

*property*

Dual-number verification reference `(L + K)` (clauses 3.16, 6.2).

The dual-number form declares `L` and `K` each rounded to the
nearest decibel (clause 3.16), and clause 6.2 verifies that form
against their sum `(L + K)`; the limit is therefore
`round(L_WA) + round(K_WA)` -- not the single-number `L_WAd`
(the unrounded sum rounded once), which can differ by one decibel.

### OperatingModeDeclaration.verified

*property*

Single-machine verification verdict, combined form (clause 6.2).

`True` when the verification measurement `L_1` does not exceed the
combined (single-number) declared value `L_WAd` (`L_1 <= L_WAd`),
`False` otherwise, and `None` when no verification measurement is
supplied. For a dual-number declaration use `verified_dual`,
which compares against the separately rounded `L + K`.

### OperatingModeDeclaration.verified_dual

*property*

Single-machine verification verdict, dual-number form (clause 6.2).

`True` when the verification measurement `L_1` does not exceed the
sum of the separately rounded declared values
(`L_1 <= round(L_WA) + round(K_WA)`, clauses 3.16 and 6.2),
`False` otherwise, and `None` when no verification measurement is
supplied.
