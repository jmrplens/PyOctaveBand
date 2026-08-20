#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""ISO 4871:1996 declaration of noise emission values of machinery and equipment.

ISO 4871 is the standardised *noise-emission declaration*: the information a
manufacturer or supplier states in technical documents about the airborne noise
emitted by a machine (clause 3.13). It does not measure anything itself; it
prescribes which quantities are declared, in which of two alternative forms, and
how a declared value is verified (clauses 4 to 6).

The quantities are the A-weighted sound power level ``L_WA`` (the preferred and
basic quantity, ISO 4871 Note 17) and, optionally, the A-weighted emission sound
pressure level ``L_pA`` at a work station (clause 3.11). Each is declared in one
of two forms selected by the relevant noise test code (clause 4):

* the **dual-number** form (clause 3.16): a measured noise emission value ``L``
  and its associated uncertainty ``K``, stated together but separately, both
  rounded to the nearest decibel; and
* the **single-number** form (clause 3.15): the derived declared value
  ``L_d``, the *sum* ``L + K`` of the unrounded quantities rounded once to the
  nearest decibel, an upper limit which values from repeated measurements are
  unlikely to exceed at the stated confidence level.

``K`` combines the measurement uncertainty (reproducibility) and, for a batch,
the production spread (clauses 3.20 to 3.24; :math:`K = 1.645 \sigma_\mathrm{R}` for a
single machine, Annex A.2.2). Verification (clause 6) compares a verification
measurement ``L_1`` against the declared form: for a single machine the
combined form is verified when :math:`L_1 \le L_\mathrm{d}` and the dual-number form
when :math:`L_1 \le (L + K)`, the sum of the separately rounded declared
values (clause 6.2).

This module models a declaration as :class:`NoiseEmissionDeclaration`, a set of
:class:`OperatingModeDeclaration` values (one per operating mode, clause 4), and
renders it as an ISO 4871 declaration fiche through :meth:`.report`. A
declaration is most often built from a measured sound power via
:meth:`~phonometry.emission.sound_power.SoundPowerResult.declare`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._report.metadata import ReportMetadata

#: The two alternative forms of a noise emission declaration (ISO 4871 clause 4).
DeclarationForm = Literal["dual-number", "single-number"]


def _round_db(value: float) -> int:
    """Round a level to the nearest decibel, halves up (ISO 4871:1996, 3.15/3.16).

    The declared values are stated to the nearest decibel; a manufacturer that
    already declares rounded integers is unaffected, and a value derived from an
    unrounded measurement is rounded here.
    """
    return math.floor(float(value) + 0.5)


@dataclass(frozen=True)
class OperatingModeDeclaration:
    r"""Declared dual-number noise-emission values for one operating mode.

    Holds the measured A-weighted sound power level ``L_WA`` and its uncertainty
    ``K_WA`` (ISO 4871 clause 3.16), and optionally the A-weighted emission sound
    pressure level ``L_pA`` at a work station with its uncertainty ``K_pA``
    (clause 3.11). The derived declared single-number values follow from
    :math:`L_\mathrm{d} = L + K` (clause 3.15), the sum rounded to the nearest decibel.
    When a verification measurement ``L_1`` (an A-weighted sound power level
    determined for verification, clause 6) is supplied, :attr:`verified`
    applies the single-machine criterion of clause 6.2 for the combined form
    (verified when :math:`L_1 \le L_{W\mathrm{Ad}}`) and :attr:`verified_dual` the
    dual-number one (verified when
    :math:`L_1 \le \mathrm{round}(L_{W\mathrm{A}}) + \mathrm{round}(K_{W\mathrm{A}})`).

    :ivar mode: Operating-mode label printed as the table column header (e.g.
        ``"Operating mode 1"``); it identifies the operating mode of clause 5 c.
    :ivar sound_power_level: Measured mean A-weighted sound power level ``L_WA``,
        in decibels re 1 pW.
    :ivar sound_power_uncertainty: Uncertainty ``K_WA`` of the sound power level,
        in decibels (finite and non-negative).
    :ivar emission_pressure_level: A-weighted emission sound pressure level
        ``L_pA`` at the work station, in decibels re 20 uPa; ``None`` when only
        the sound power level is declared.
    :ivar emission_pressure_uncertainty: Uncertainty ``K_pA`` of the emission
        sound pressure level, in decibels; required when
        :attr:`emission_pressure_level` is given and forbidden otherwise.
    :ivar verification_level: A verification measurement ``L_1`` of the
        A-weighted sound power level (clause 6), in decibels; ``None`` when the
        mode is not verified.
    :raises ValueError: If a level is not finite, an uncertainty is not finite
        and non-negative, or only one member of the emission-pressure pair is
        given.
    """

    mode: str
    sound_power_level: float
    sound_power_uncertainty: float
    emission_pressure_level: float | None = None
    emission_pressure_uncertainty: float | None = None
    verification_level: float | None = None

    def __post_init__(self) -> None:
        """Validate the levels, the uncertainties and the pressure pairing."""
        if not str(self.mode).strip():
            raise ValueError("OperatingModeDeclaration.mode must be a non-empty label.")
        for name in ("sound_power_level", "verification_level"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(
                    f"OperatingModeDeclaration.{name} must be finite when given; "
                    f"got {value!r}."
                )
        for name in ("sound_power_uncertainty", "emission_pressure_uncertainty"):
            value = getattr(self, name)
            if value is not None and not (
                math.isfinite(float(value)) and float(value) >= 0.0
            ):
                raise ValueError(
                    f"OperatingModeDeclaration.{name} must be finite and "
                    f"non-negative when given; got {value!r}."
                )
        if (self.emission_pressure_level is None) != (
            self.emission_pressure_uncertainty is None
        ):
            missing = (
                "emission_pressure_uncertainty"
                if self.emission_pressure_uncertainty is None
                else "emission_pressure_level"
            )
            raise ValueError(
                "emission_pressure_level and emission_pressure_uncertainty must "
                f"be given together (ISO 4871 dual-number form); '{missing}' is "
                "missing."
            )
        if self.emission_pressure_level is not None and not math.isfinite(
            float(self.emission_pressure_level)
        ):
            raise ValueError(
                "OperatingModeDeclaration.emission_pressure_level must be finite "
                f"when given; got {self.emission_pressure_level!r}."
            )

    @property
    def declared_sound_power_level(self) -> int:
        r"""Declared single-number sound power level (ISO 4871 clause 3.15).

        Clause 3.15 defines the declared value
        :math:`L_{W\mathrm{Ad}} = L_{W\mathrm{A}} + K_{W\mathrm{A}}` as the *sum* of the measured
        value and its uncertainty, rounded to the nearest decibel: the sum is
        formed from the unrounded quantities (clause 3.12) and rounded once,
        not assembled from the separately rounded dual-number values (3.16).
        """
        return _round_db(self.sound_power_level + self.sound_power_uncertainty)

    @property
    def declared_emission_pressure_level(self) -> int | None:
        r"""Declared single-number emission pressure level (clause 3.15).

        The sum :math:`L_{p\mathrm{Ad}} = L_{p\mathrm{A}} + K_{p\mathrm{A}}` rounded once to the nearest
        decibel (clause 3.15); ``None`` when no emission sound pressure level
        is declared for the mode.
        """
        if (
            self.emission_pressure_level is None
            or self.emission_pressure_uncertainty is None
        ):
            return None
        return _round_db(
            self.emission_pressure_level + self.emission_pressure_uncertainty
        )

    @property
    def dual_number_verification_limit(self) -> int:
        """Dual-number verification reference ``(L + K)`` (clauses 3.16, 6.2).

        The dual-number form declares ``L`` and ``K`` each rounded to the
        nearest decibel (clause 3.16), and clause 6.2 verifies that form
        against their sum ``(L + K)``; the limit is therefore
        ``round(L_WA) + round(K_WA)`` -- not the single-number ``L_WAd``
        (the unrounded sum rounded once), which can differ by one decibel.
        """
        return _round_db(self.sound_power_level) + _round_db(
            self.sound_power_uncertainty
        )

    @property
    def verified(self) -> bool | None:
        r"""Single-machine verification verdict, combined form (clause 6.2).

        ``True`` when the verification measurement ``L_1`` does not exceed the
        combined (single-number) declared value ``L_WAd``
        (:math:`L_1 \le L_{W\mathrm{Ad}}`),
        ``False`` otherwise, and ``None`` when no verification measurement is
        supplied. For a dual-number declaration use :attr:`verified_dual`,
        which compares against the separately rounded ``L + K``.
        """
        if self.verification_level is None:
            return None
        return float(self.verification_level) <= self.declared_sound_power_level

    @property
    def verified_dual(self) -> bool | None:
        r"""Single-machine verification verdict, dual-number form (clause 6.2).

        ``True`` when the verification measurement ``L_1`` does not exceed the
        sum of the separately rounded declared values
        (:math:`L_1 \le \mathrm{round}(L_{W\mathrm{A}}) + \mathrm{round}(K_{W\mathrm{A}})`,
        clauses 3.16 and 6.2),
        ``False`` otherwise, and ``None`` when no verification measurement is
        supplied.
        """
        if self.verification_level is None:
            return None
        return float(self.verification_level) <= self.dual_number_verification_limit


@dataclass(frozen=True)
class NoiseEmissionDeclaration:
    r"""An ISO 4871:1996 declaration of noise emission values (clauses 4 to 6).

    A declaration is one or more :class:`OperatingModeDeclaration` values (one
    per operating mode, clause 4) plus the accompanying information required by
    clause 5: identification of the machinery (clause 5 a), the noise test code
    and basic standards used (clause 5 b) and the operating conditions
    (clause 5 c). :meth:`report` renders it as a one-page ISO 4871 declaration
    fiche.

    :ivar modes: One or more per-operating-mode declarations. A sequence is
        accepted and stored as a tuple.
    :ivar machine: Machine identification (model number and other identifying
        information, clause 5 a).
    :ivar operating_conditions: Operating and mounting conditions the values
        refer to (clause 5 c), e.g. ``"50 Hz, 230 V, rated load"``.
    :ivar noise_test_code: The noise test code the values were determined to
        (clause 5 b), e.g. an ISO family-specific test code; ``None`` when none
        exists.
    :ivar basic_standards: The basic noise-emission standard(s) used to obtain
        the values (clause 5 b), e.g. ``("ISO 3744",)``. A single string is
        accepted and wrapped in a one-tuple.
    :ivar form: Which declaration form the fiche presents, ``"dual-number"``
        (default, clause 3.16: ``L`` and ``K`` separately) or
        ``"single-number"`` (clause 3.15: the derived :math:`L_\mathrm{d} = L + K`).
    :raises ValueError: If no operating mode is given or ``form`` is unknown.
    """

    modes: Sequence[OperatingModeDeclaration]
    machine: str | None = None
    operating_conditions: str | None = None
    noise_test_code: str | None = None
    basic_standards: Sequence[str] = field(default_factory=tuple)
    form: DeclarationForm = "dual-number"

    def __post_init__(self) -> None:
        """Coerce the sequences to tuples and validate the form and mode count."""
        modes = tuple(self.modes)
        if not modes:
            raise ValueError(
                "NoiseEmissionDeclaration requires at least one operating mode."
            )
        standards = (
            (self.basic_standards,)
            if isinstance(self.basic_standards, str)
            else tuple(self.basic_standards)
        )
        if self.form not in ("dual-number", "single-number"):
            raise ValueError(
                "NoiseEmissionDeclaration.form must be 'dual-number' or "
                f"'single-number'; got {self.form!r}."
            )
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "basic_standards", standards)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        r"""Render an ISO 4871 noise-emission declaration fiche to a PDF.

        Writes a one-page declaration data sheet: the standard-basis line
        (ISO 4871:1996 and the cited basic emission standard), the machine
        identification and operating conditions, the declared dual- or
        single-number table across the operating-mode columns following the
        ISO 4871 Annex B layouts (``L_WA`` and ``K_WA`` for the dual-number
        form, the derived :math:`L_{W\mathrm{Ad}} = L_{W\mathrm{A}} + K_{W\mathrm{A}}` for the
        single-number form,
        plus the emission sound pressure level when declared), the
        noise-test-code and basic-standards footnote, a verification verdict
        table when a verification measurement is supplied (clause 6.2, against
        ``L_WAd`` for the single-number form or the separately rounded
        ``L_WA + K_WA`` for the dual-number form), and the footer
        identity/disclaimer block.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the laboratory identity (footer) and, through
            ``measurement_standard``, the basic emission standard shown in the
            standard-basis line when :attr:`basic_standards` is empty.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            declaration fiche has a single table layout, so it has no effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso4871 import render_iso4871_report

        return render_iso4871_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )
