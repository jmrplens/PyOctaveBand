#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Single-number weighted ratings of sound insulation and their spectrum
adaptation terms (ISO 717-1 airborne, ISO 717-2 impact).

ISO 717 rates a curve, not a room. Whatever produced the band values (a field
measurement to ISO 16283, a laboratory measurement to ISO 10140, a prediction
to ISO 12354), both parts of ISO 717 reduce them to a single number by the same
reference-curve method: one shift search, one pair of deviation bounds, one
rounding rule, run against the airborne reference curve of ISO 717-1 or the
impact reference curve of ISO 717-2, whose only structural difference is the
sign an unfavourable deviation has. That shared machinery, with the Table 3
reference curves and the Table 4 / Table B.1 spectra it reads, is the subject
of this module.

**Weighted rating (ISO 717-1).** The reference-curve method of Clause 4.4
shifts the reference curve of Table 3 in 1 dB steps towards the measured
curve until the sum of unfavourable deviations (measured below the
shifted reference) is as large as possible but not more than 32,0 dB for
the 16 one-third-octave bands (100 Hz to 3150 Hz) or 10,0 dB for the 5
octave bands (125 Hz to 2000 Hz). The weighted rating (``Rw``, ``R'w``,
``Dn,w``, ``DnT,w`` ...) is the shifted reference read at 500 Hz. The
spectrum adaptation terms are :math:`C = X_{\mathrm{A}1} - X_\mathrm{w}` and
:math:`C_\mathrm{tr} = X_{\mathrm{A}2} - X_\mathrm{w}`
with :math:`X_{\mathrm{A}j} = -10 \log_{10} \sum 10^{(L_{ij} - X_i)/10}` rounded to an
integer, using
the A-weighted spectra No. 1 (pink noise, ``C``) and No. 2 (urban traffic,
``Ctr``) of Table 4 (Clause 4.5, Formula (1) and (2)). Input levels are
reduced to one decimal place before use (Clause 4.4, footnote 1). The
reference values, spectra and shifting rule are identical in the 2013 and
2020 editions of ISO 717-1.

**Enlarged frequency ranges (ISO 717-1 Annex B; ISO 717-2 A.2.1 NOTE).**
When measurements cover an enlarged range, additional adaptation terms are
stated with the range as a subscript: ``C50-3150``, ``C50-5000``,
``C100-5000`` (and the ``Ctr`` counterparts) with the Table B.1 spectra, and
``CI,50-2500`` for impact. :func:`weighted_rating_extended` and
:func:`weighted_impact_rating_extended` compute them alongside the core
rating. Both accept ``one_decimal=True`` for the "1/10 dB for the expression
of uncertainty" variant of Clauses 4.4/4.5 (reference-curve shift in 0,1 dB
steps and one-decimal reductions), which ISO 12999-1:2020 Annex B requires
when stating the uncertainty of single-number values.

**Weighted impact rating (ISO 717-2).** The reference-curve method of
Clause 4.3 shifts the Table 3 impact reference curve towards the measured
curve until the sum of unfavourable deviations (here where the
**measurement exceeds** the reference, the sign opposite to airborne) is
as large as possible but not more than 32,0 dB (16 one-third-octave bands)
or 10,0 dB (5 octave bands). The rating (``Ln,w``, ``L'n,w``, ``L'nT,w``)
is the shifted reference read at 500 Hz, reduced by a further 5 dB for
octave bands (Clause 4.3.2). The spectrum adaptation term
:math:`C_\mathrm{I} = L_\mathrm{n,sum} - 15 - L_\mathrm{n,w}` uses the energetic sum ``Ln,sum``
over
100 Hz to 2500 Hz (one-third octave) or 125 Hz to 2000 Hz (octave),
rounded to an integer (Clause A.2.1, Formulae (A.1) to (A.3)). The Table 3
reference values, the shifting rule and CI are identical in the 2013 and
2020 editions of ISO 717-2 (the 2020 edition only adds Annex D for the
rubber-ball heavy/soft impactor, out of scope here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..._internal.levels_math import energy_sum
from ..._internal.validation import (
    check_engine,
    require_equal_shapes,
    require_ranks,
    require_same_length,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

# --- ISO 717-1 Table 3 reference values ----------------------------------

#: One-third-octave reference values, 100 Hz to 3150 Hz (Table 3).
_REF_THIRD_OCTAVE: tuple[int, ...] = (
    33,
    36,
    39,
    42,
    45,
    48,
    51,
    52,
    53,
    54,
    55,
    56,
    56,
    56,
    56,
    56,
)
#: Octave reference values, 125 Hz to 2000 Hz (Table 3).
_REF_OCTAVE: tuple[int, ...] = (36, 45, 52, 55, 56)

#: One-third-octave band centre frequencies, 100 Hz to 3150 Hz (16 bands).
_FREQ_THIRD_OCTAVE: tuple[float, ...] = (
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
)
#: Octave band centre frequencies, 125 Hz to 2000 Hz (5 bands).
_FREQ_OCTAVE: tuple[float, ...] = (125.0, 250.0, 500.0, 1000.0, 2000.0)

#: Number of bands in each ISO 717 band set, used to infer the band set
#: from the input length: 16 one-third-octave bands (100 Hz to 3150 Hz)
#: and 5 octave bands (125 Hz to 2000 Hz).
_N_THIRD_OCTAVE_BANDS = 16
_N_OCTAVE_BANDS = 5

#: Index of the 500 Hz band in each band set (the rating is read there).
_INDEX_500_THIRD = 7
_INDEX_500_OCTAVE = 2

#: Maximum sum of unfavourable deviations. These bounds are shared by both
#: rating paths: ISO 717-1 Clause 4.4 (airborne) and ISO 717-2 Clause 4.3
#: (impact) specify the identical 32,0 dB (16 one-third-octave bands) and
#: 10,0 dB (5 octave bands) limits.
_MAX_UNFAVOURABLE_THIRD = 32.0
_MAX_UNFAVOURABLE_OCTAVE = 10.0

#: Tolerance absorbing floating-point noise when comparing the
#: unfavourable-deviation sum (a true multiple of 0,1 dB) to the bound.
_SHIFT_TOLERANCE = 1e-6

#: Tolerance for checking that ``scale * step`` reconstructs exactly 1,
#: i.e. that the reference-curve shift step divides 1 dB exactly (1.0 or
#: 0.1 dB per ISO 717 / ISO 12999-1:2020 Annex B.2), absorbing the
#: representation error of step values like 0.1. Distinct from
#: _SHIFT_TOLERANCE, which guards the deviation-sum comparison.
_STEP_DIVISOR_TOLERANCE = 1e-12

# --- ISO 717-2 Table 3 impact reference values ---------------------------

#: One-third-octave impact reference values, 100 Hz to 3150 Hz (Table 3).
_REF_IMPACT_THIRD_OCTAVE: tuple[int, ...] = (
    62,
    62,
    62,
    62,
    62,
    62,
    61,
    60,
    59,
    58,
    57,
    54,
    51,
    48,
    45,
    42,
)
#: Octave impact reference values, 125 Hz to 2000 Hz (Table 3).
_REF_IMPACT_OCTAVE: tuple[int, ...] = (67, 67, 65, 62, 49)

#: Octave-band single-number reduction applied to L'n,w / L'nT,w
#: (ISO 717-2 Clause 4.3.2): the shifted reference at 500 Hz minus 5 dB.
_IMPACT_OCTAVE_OFFSET = -5

#: One-third-octave band count for CI (100 Hz to 2500 Hz, excludes 3150 Hz).
_CI_THIRD_OCTAVE_BANDS = 15

# --- ISO 717-1 Table 4 spectra (A-weighted, normalized to 0 dB) ----------

#: Spectrum No. 1 (pink noise, for C), one-third octave 100-3150 Hz.
_SPECTRUM1_THIRD: tuple[int, ...] = (
    -29,
    -26,
    -23,
    -21,
    -19,
    -17,
    -15,
    -13,
    -12,
    -11,
    -10,
    -9,
    -9,
    -9,
    -9,
    -9,
)
#: Spectrum No. 2 (urban traffic, for Ctr), one-third octave 100-3150 Hz.
_SPECTRUM2_THIRD: tuple[int, ...] = (
    -20,
    -20,
    -18,
    -16,
    -15,
    -14,
    -13,
    -12,
    -11,
    -9,
    -8,
    -9,
    -10,
    -11,
    -13,
    -15,
)
#: Spectrum No. 1 (for C), octave 125-2000 Hz.
_SPECTRUM1_OCTAVE: tuple[int, ...] = (-21, -14, -8, -5, -4)
#: Spectrum No. 2 (for Ctr), octave 125-2000 Hz.
_SPECTRUM2_OCTAVE: tuple[int, ...] = (-14, -10, -7, -4, -6)

# --- ISO 717-1:2020 Table B.1 spectra for the enlarged frequency ranges ----
# One-third-octave sound levels, A-weighted and normalized to 0 dB over each
# range. Spectrum No. 1 has one column for C50-3150 and one shared column for
# C50-5000 and C100-5000; spectrum No. 2 has a single column valid for Ctr in
# any enlarged range.

#: One-third-octave band centre frequencies, 50 Hz to 5000 Hz (21 bands).
_FREQ_50_5000: tuple[float, ...] = (
    50.0,
    63.0,
    80.0,
    *_FREQ_THIRD_OCTAVE,
    4000.0,
    5000.0,
)
#: Spectrum No. 1 column for C50-3150 (19 bands, 50-3150 Hz).
_SPECTRUM1_50_3150: tuple[int, ...] = (
    -40,
    -36,
    -33,
    -29,
    -26,
    -23,
    -21,
    -19,
    -17,
    -15,
    -13,
    -12,
    -11,
    -10,
    -9,
    -9,
    -9,
    -9,
    -9,
)
#: Spectrum No. 1 column for C50-5000 and C100-5000 (21 bands, 50-5000 Hz;
#: the 100-5000 Hz range uses the same column restricted to its bands).
_SPECTRUM1_50_5000: tuple[int, ...] = (
    -41,
    -37,
    -34,
    -30,
    -27,
    -24,
    -22,
    -20,
    -18,
    -16,
    -14,
    -13,
    -12,
    -11,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
)
#: Spectrum No. 2 column for Ctr in any enlarged range (21 bands, 50-5000 Hz).
_SPECTRUM2_50_5000: tuple[int, ...] = (
    -25,
    -23,
    -21,
    -20,
    -20,
    -18,
    -16,
    -15,
    -14,
    -13,
    -12,
    -11,
    -9,
    -8,
    -9,
    -10,
    -11,
    -13,
    -15,
    -16,
    -18,
)

#: The enlarged one-third-octave adaptation ranges of ISO 717-1:2020 Annex B
#: (airborne): descriptor suffix -> (band frequencies, spectrum No. 1 levels,
#: spectrum No. 2 levels).
_EXTENDED_RANGES: dict[
    str, tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...]]
] = {
    "50_3150": (_FREQ_50_5000[:19], _SPECTRUM1_50_3150, _SPECTRUM2_50_5000[:19]),
    "50_5000": (_FREQ_50_5000, _SPECTRUM1_50_5000, _SPECTRUM2_50_5000),
    "100_5000": (_FREQ_50_5000[3:], _SPECTRUM1_50_5000[3:], _SPECTRUM2_50_5000[3:]),
}

#: The enlarged CI summation range of ISO 717-2:2020 A.2.1 NOTE: 50-2500 Hz
#: (18 one-third-octave bands).
_CI_50_2500_FREQS: tuple[float, ...] = _FREQ_50_5000[:18]


_VALUES_1D_MSG = "'values_by_band' must be one-dimensional."
_VALUES_FINITE_MSG = "'values_by_band' must contain only finite values."


@dataclass(frozen=True)
class WeightedRatingResult:
    """Single-number weighted rating and adaptation terms (ISO 717-1).

    :ivar rating: Weighted rating (``Rw``, ``R'w``, ``DnT,w`` ...), the
        shifted reference read at 500 Hz, in dB (Clause 4.4). Integer.
    :ivar c: Spectrum adaptation term ``C`` (spectrum No. 1), in dB
        (Clause 4.5). Integer.
    :ivar ctr: Spectrum adaptation term ``Ctr`` (spectrum No. 2), in dB
        (Clause 4.5). Integer.
    :ivar unfavourable_sum: Sum of unfavourable deviations at the final
        shift, in dB (Clause 4.4); at most 32,0 (16 bands) or 10,0 (5
        bands).
    :ivar band_centers: Band centre frequencies of the measured curve, in
        Hz. Defaults to ``None`` for backward-compatible construction.
    :ivar measured: The measured band quantities used for the rating (after
        the one-decimal reduction of Clause 4.4), in dB. Defaults to
        ``None``.
    :ivar shifted_reference: Table 3 reference curve after the final shift,
        in dB. Defaults to ``None``.
    :ivar quantity: Always ``"airborne"``: this class carries the ISO 717-1
        airborne rating, and the renderers dispatch on this tag when handed
        the union with :class:`ImpactRatingResult`, which carries
        ``"impact"``. The field used to admit both values and promise that
        ``"impact"`` would select the impact labels; it never could, since
        the impact labels read ``ci`` off the result and this class does not
        have one, so the promise ended in the renderer's ``AttributeError``.
    """

    rating: int
    c: int
    ctr: int
    unfavourable_sum: float
    band_centers: np.ndarray | None = None
    measured: np.ndarray | None = None
    shifted_reference: np.ndarray | None = None
    quantity: Literal["airborne"] = "airborne"

    def __post_init__(self) -> None:
        """Reject a rating whose three band curves do not line up.

        The three arrays are the record of which bands the single number was
        rated over, and a fiche reads them as exactly that: ISO 717-1
        Clause 5.3 makes it declare whether the rating came from
        one-third-octave or octave bands, and it settles that from the length
        of ``band_centers`` alone. The renderers do compare the three, but
        only once a report is asked for, which can be long after the rating
        was built and passed on; :meth:`plot` compares nothing and hands the
        curves to matplotlib, whose complaint about first dimensions names
        neither the field nor the type it came from.

        The three arrays default to ``None`` on a rating built from the
        single numbers alone, and an absent curve is skipped rather than read
        as a disagreement.

        :raises ValueError: if the band curves supplied disagree.
        """
        require_ranks(self, band_centers=1, measured=1, shifted_reference=1)
        require_same_length(self, "band_centers", "measured", "shifted_reference")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the measured curve vs the shifted reference (ISO 717-1).

        Unfavourable deviations (reference above measurement) are shaded and
        ``Rw (C; Ctr)`` annotated. Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_weighted_rating

        check_language(language)
        return plot_weighted_rating(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
        symbol: str | None = None,
    ) -> str:
        """Render an ISO 717-1 airborne sound-insulation fiche to a PDF.

        Writes a one-page accredited-laboratory report: the standard-basis
        line, an optional metadata header block, the band table beside the
        measured-versus-shifted-reference plot (the result's own
        :meth:`plot`), the boxed ``Rw (C; Ctr)`` result, an optional verdict
        row and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a
            prediction fiche (body, result and disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table uses the ISO 717 Annex C
            columns (frequency, measured value, shifted reference,
            unfavourable deviation) instead of the two-column ``f | value``
            table.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :param symbol: The reported single-number quantity, as plain text:
            ``"Rw"`` (the default when ``None``), ``"R'w"``, ``"Dn,w"``,
            ``"DnT,w"`` ... per ISO 717-1 Tables 1-2, so a field measurement
            (e.g. a standardized level difference rated to ``DnT,w``) is not
            mislabelled with the laboratory descriptor.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``, ``symbol``
            is not a valid quantity-symbol shape, or the result was built
            without the per-band data (``band_centers``, ``measured``,
            ``shifted_reference``).
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        return _render_iso717(
            self,
            path,
            metadata=metadata,
            engine=engine,
            verbose=verbose,
            language=language,
            symbol=symbol,
        )


@dataclass(frozen=True)
class ImpactRatingResult:
    """Single-number weighted impact rating and CI (ISO 717-2).

    :ivar rating: Weighted impact rating (``Ln,w``, ``L'n,w``,
        ``L'nT,w``), the shifted reference read at 500 Hz, in dB
        (Clause 4.3; octave-band ratings include the -5 dB reduction of
        Clause 4.3.2). Integer.
    :ivar ci: Spectrum adaptation term ``CI`` (Clause A.2.1), in dB.
        Integer.
    :ivar unfavourable_sum: Sum of unfavourable deviations at the final
        shift, in dB (Clause 4.3); at most 32,0 (16 bands) or 10,0 (5
        bands).
    :ivar band_centers: Band centre frequencies of the measured curve, in
        Hz. Defaults to ``None`` for backward-compatible construction.
    :ivar measured: The measured impact levels used for the rating (after
        the one-decimal reduction of Clause 4.3.1), in dB. Defaults to
        ``None``.
    :ivar shifted_reference: Table 3 impact reference curve after the final
        shift, in dB. Defaults to ``None``.
    :ivar quantity: Always ``"impact"`` (ISO 717-2), selecting the impact
        labels of the ISO 717 Annex C report.
    """

    rating: int
    ci: int
    unfavourable_sum: float
    band_centers: np.ndarray | None = None
    measured: np.ndarray | None = None
    shifted_reference: np.ndarray | None = None
    quantity: Literal["impact"] = "impact"

    def __post_init__(self) -> None:
        """Reject a rating whose three band curves do not line up.

        Shading the bands where the measurement rises *above* the shifted
        reference -- the sign opposite to airborne -- is a comparison of one
        curve against the other, band for band, and :meth:`plot` makes it
        without first asking whether the two run over the same bands. The
        band centres carry as much: ISO 717-2 Clause 4.4 makes the fiche
        declare whether the rating came from one-third-octave or octave
        bands, and their length alone settles it. The renderers refuse a
        mismatch, but not until a report is asked for, and a rating is
        usually built long before that.

        The three arrays default to ``None`` on a rating built from the
        single numbers alone, and an absent curve is skipped rather than read
        as a disagreement.

        :raises ValueError: if the band curves supplied disagree.
        """
        require_ranks(self, band_centers=1, measured=1, shifted_reference=1)
        require_same_length(self, "band_centers", "measured", "shifted_reference")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the measured curve vs the shifted reference (ISO 717-2).

        Unfavourable deviations (measurement above the reference, the sign
        opposite to airborne) are shaded and ``Ln,w (CI)`` annotated.
        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_impact_rating

        check_language(language)
        return plot_impact_rating(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
        symbol: str | None = None,
    ) -> str:
        """Render an ISO 717-2 impact-insulation fiche to a PDF.

        Writes a one-page accredited-laboratory report for impact sound: the
        standard-basis line, an optional metadata header block, the band
        table beside the measured-versus-shifted-reference plot (the
        result's own :meth:`plot`), the boxed ``Ln,w (CI)`` result, an
        optional verdict row and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a
            prediction fiche (body, result and disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table uses the ISO 717 Annex C
            columns (frequency, measured value, shifted reference,
            unfavourable deviation) instead of the two-column ``f | value``
            table.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :param symbol: The reported single-number quantity, as plain text:
            ``"Ln,w"`` (the default when ``None``), ``"L'n,w"`` or
            ``"L'nT,w"`` per ISO 717-2 Table 1, so a field measurement is not
            mislabelled with the laboratory descriptor.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``, ``symbol``
            is not a valid quantity-symbol shape, or the result was built
            without the per-band data (``band_centers``, ``measured``,
            ``shifted_reference``).
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        return _render_iso717(
            self,
            path,
            metadata=metadata,
            engine=engine,
            verbose=verbose,
            language=language,
            symbol=symbol,
        )


def _render_iso717(
    result: WeightedRatingResult | ImpactRatingResult,
    path: str,
    *,
    metadata: ReportMetadata | None,
    engine: str,
    verbose: bool,
    language: str,
    symbol: str | None = None,
) -> str:
    """Validate the report request and delegate to the reportlab renderer.

    Shared by :meth:`WeightedRatingResult.report` and
    :meth:`ImpactRatingResult.report`: it rejects unknown engines and results
    built without the per-band data, then calls the reportlab renderer (which
    raises a clear :class:`ImportError` when reportlab is absent).
    """
    from ..._i18n import check_language

    check_language(language)
    check_engine(engine)
    if (
        result.band_centers is None
        or result.measured is None
        or result.shifted_reference is None
    ):
        msg = (
            "report() needs the per-band data ('band_centers', 'measured', "
            "'shifted_reference'); build the rating with weighted_rating() or "
            "weighted_impact_rating() so they are populated."
        )
        raise ValueError(msg)
    from ..._report.iso717 import render_iso717_report

    return render_iso717_report(
        result,
        path,
        metadata=metadata,
        verbose=verbose,
        language=language,
        symbol=symbol,
    )


def _round_half_up_tenths(values: np.ndarray) -> np.ndarray:
    r"""Reduce levels to one decimal place (ISO 717-1 Clause 4.4, note 1).

    Rounds each value to the nearest tenth of a decibel, half away from
    zero (:math:`\lfloor x \cdot 10 + 0.5 \rfloor / 10` for non-negative
    values, mirrored for
    negative ones).

    .. note::
        This rounds negative halves *away from zero* (−0.05 → −0.1), whereas
        the adaptation-term reductions (:func:`_adaptation_term`,
        :func:`_impact_ci`) use the ISO 80000-1 footnote form
        :math:`\lfloor x + 0.5 \rfloor` which rounds them *towards* +∞
        (−0.5 → 0). The two
        conventions differ only for exactly-half negative values, which do
        not occur with realistic (positive-level) insulation data; the
        difference is documented here rather than unified so each function
        keeps the literal form of its clause.
    """
    rounded: np.ndarray = np.sign(values) * np.floor(np.abs(values) * 10.0 + 0.5) / 10.0
    return rounded


def _resolve_band_set(
    n: int, bands: str | None
) -> tuple[tuple[int, ...], float, int, tuple[int, ...], tuple[int, ...]]:
    """Select the reference curve, bound and spectra for the band set.

    :return: ``(reference, max_unfavourable, index_500, spectrum1,
        spectrum2)``.
    """
    if bands == "third-octave" or (bands is None and n == _N_THIRD_OCTAVE_BANDS):
        if n != _N_THIRD_OCTAVE_BANDS:
            msg = f"One-third-octave rating needs 16 bands (100-3150 Hz), got {n}."
            raise ValueError(msg)
        return (
            _REF_THIRD_OCTAVE,
            _MAX_UNFAVOURABLE_THIRD,
            _INDEX_500_THIRD,
            _SPECTRUM1_THIRD,
            _SPECTRUM2_THIRD,
        )
    if bands == "octave" or (bands is None and n == _N_OCTAVE_BANDS):
        if n != _N_OCTAVE_BANDS:
            msg = f"Octave rating needs 5 bands (125-2000 Hz), got {n}."
            raise ValueError(msg)
        return (
            _REF_OCTAVE,
            _MAX_UNFAVOURABLE_OCTAVE,
            _INDEX_500_OCTAVE,
            _SPECTRUM1_OCTAVE,
            _SPECTRUM2_OCTAVE,
        )
    if bands is not None:
        msg = "'bands' must be 'third-octave', 'octave' or None."
        raise ValueError(msg)
    msg = (
        "Expected 16 one-third-octave (100-3150 Hz) or 5 octave "
        f"(125-2000 Hz) values, got {n}."
    )
    raise ValueError(msg)


def _best_shift(
    measured: np.ndarray,
    reference: np.ndarray,
    limit: float,
    step: float = 1.0,
) -> tuple[float, float]:
    r"""Largest ``step``-sized shift with unfavourable-deviation sum bounded.

    Shifts the reference by multiples of ``step`` and returns the largest
    shift for which
    :math:`\sum \max(0, \mathrm{reference} + \mathrm{shift}
    - \mathrm{measured}) \le \mathrm{limit}`
    (the sum is monotone non-decreasing in the shift), together with that
    sum. ``step`` is 1 dB for the standard rating (ISO 717-1 Clause 4.4 /
    ISO 717-2 Clause 4.3) or 0,1 dB for the one-decimal rating used in
    uncertainty statements (ISO 717 "1/10 dB for the expression of
    uncertainty"; ISO 12999-1:2020 Annex B.2 requires the 0,1 dB steps).

    Measured levels are multiples of 0,1 dB (Clause 4.4 footnote 1) and the
    reference is integer, so with both step sizes every deviation sum is a
    true multiple of 0,1 dB; a small tolerance absorbs floating-point noise
    so that a sum of exactly 32,0 (or 10,0) dB is not spuriously rejected.
    The shift is searched on an integer grid of ``step`` ticks to keep the
    0,1 dB steps exact.
    """
    scale = round(1.0 / step)
    if scale < 1 or abs(scale * step - 1.0) > _STEP_DIVISOR_TOLERANCE:
        msg = "'step' must divide 1 dB exactly (e.g. 1.0 or 0.1)."
        raise ValueError(msg)
    # Start below any feasible shift, then climb while the bound holds.
    n = (int(np.floor(np.min(measured - reference))) - 1) * scale
    while True:
        candidate = (n + 1) / scale
        next_sum = float(np.sum(np.maximum(0.0, reference + candidate - measured)))
        if next_sum > limit + _SHIFT_TOLERANCE:
            break
        n += 1
    shift = n / scale
    unfavourable = float(np.sum(np.maximum(0.0, reference + shift - measured)))
    return shift, unfavourable


def _adaptation_term(
    measured: np.ndarray, spectrum: tuple[int, ...], rating: int
) -> int:
    """Spectrum adaptation term ``Xaj - rating`` (Clause 4.5, Formula (2))."""
    x_aj = -energy_sum(np.asarray(spectrum, dtype=np.float64) - measured)
    return math.floor(x_aj + 0.5) - rating


def weighted_rating(
    values_by_band: Sequence[float] | np.ndarray,
    bands: str | None = None,
) -> WeightedRatingResult:
    """Single-number weighted rating and C / Ctr per ISO 717-1.

    Applies the reference-curve method of Clause 4.4: the Table 3
    reference curve is shifted in 1 dB steps towards the measured curve
    until the sum of unfavourable deviations is as large as possible but
    not more than 32,0 dB (16 one-third-octave bands, 100 Hz to 3150 Hz)
    or 10,0 dB (5 octave bands, 125 Hz to 2000 Hz). The rating is the
    shifted reference read at 500 Hz. The spectrum adaptation terms
    ``C`` and ``Ctr`` follow Clause 4.5 with the Table 4 spectra No. 1 and
    No. 2. Input values are first reduced to one decimal place
    (Clause 4.4, footnote 1).

    :param values_by_band: Measured band quantities (``R``, ``R'``,
        ``Dn``, ``DnT`` ...) in dB. 16 values are read as one-third-octave
        bands, 5 values as octave bands.
    :param bands: ``"third-octave"``, ``"octave"`` or ``None`` to infer
        the band set from the number of values.
    :return: :class:`WeightedRatingResult` with ``rating``, ``c``,
        ``ctr`` and ``unfavourable_sum``.
    :raises ValueError: If the number of values does not match the band
        set, or if any value is non-finite.
    """
    data = np.asarray(values_by_band, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(_VALUES_1D_MSG)
    if not np.all(np.isfinite(data)):
        raise ValueError(_VALUES_FINITE_MSG)

    reference, limit, index_500, spectrum1, spectrum2 = _resolve_band_set(
        int(data.size), bands
    )
    measured = _round_half_up_tenths(data)
    ref = np.asarray(reference, dtype=np.float64)

    shift, unfavourable = _best_shift(measured, ref, limit)
    rating = int(reference[index_500]) + round(shift)
    c = _adaptation_term(measured, spectrum1, rating)
    ctr = _adaptation_term(measured, spectrum2, rating)
    centers = _FREQ_THIRD_OCTAVE if data.size == _N_THIRD_OCTAVE_BANDS else _FREQ_OCTAVE
    return WeightedRatingResult(
        rating=rating,
        c=c,
        ctr=ctr,
        unfavourable_sum=unfavourable,
        band_centers=np.asarray(centers, dtype=np.float64),
        measured=measured,
        shifted_reference=ref + shift,
    )


def _resolve_impact_band_set(
    n: int, bands: str | None
) -> tuple[tuple[int, ...], float, int, int, int]:
    """Select the impact reference curve, bound, indices for the band set.

    :return: ``(reference, max_unfavourable, index_500, octave_offset,
        ci_band_count)``.
    """
    if bands == "third-octave" or (bands is None and n == _N_THIRD_OCTAVE_BANDS):
        if n != _N_THIRD_OCTAVE_BANDS:
            msg = (
                f"One-third-octave impact rating needs 16 bands (100-3150 Hz), got {n}."
            )
            raise ValueError(msg)
        return (
            _REF_IMPACT_THIRD_OCTAVE,
            _MAX_UNFAVOURABLE_THIRD,
            _INDEX_500_THIRD,
            0,
            _CI_THIRD_OCTAVE_BANDS,
        )
    if bands == "octave" or (bands is None and n == _N_OCTAVE_BANDS):
        if n != _N_OCTAVE_BANDS:
            msg = f"Octave impact rating needs 5 bands (125-2000 Hz), got {n}."
            raise ValueError(msg)
        return (
            _REF_IMPACT_OCTAVE,
            _MAX_UNFAVOURABLE_OCTAVE,
            _INDEX_500_OCTAVE,
            _IMPACT_OCTAVE_OFFSET,
            5,
        )
    if bands is not None:
        msg = "'bands' must be 'third-octave', 'octave' or None."
        raise ValueError(msg)
    msg = (
        "Expected 16 one-third-octave (100-3150 Hz) or 5 octave "
        f"(125-2000 Hz) values, got {n}."
    )
    raise ValueError(msg)


def _impact_ci(measured: np.ndarray, rating: int, n_bands: int) -> int:
    r"""Spectrum adaptation term ``CI`` (ISO 717-2 Clause A.2.1).

    :math:`C_\mathrm{I} = L_\mathrm{n,sum} - 15 - L_\mathrm{n,w}` with the energetic sum
    :math:`L_\mathrm{n,sum} = 10 \log_{10} \sum
    10^{L_i/10}` over the CI range (one-third octave 100-2500 Hz, i.e.
    the first 15 bands; octave 125-2000 Hz), rounded to an integer
    (round half up), Formulae (A.1) to (A.3).
    """
    l_sum = energy_sum(measured[:n_bands])
    return math.floor(l_sum + 0.5) - 15 - rating


def weighted_impact_rating(
    values_by_band: Sequence[float] | np.ndarray,
    bands: str | None = None,
) -> ImpactRatingResult:
    r"""Single-number weighted impact rating and CI per ISO 717-2.

    Applies the reference-curve method of Clause 4.3: the Table 3 impact
    reference curve is shifted in 1 dB steps towards the measured curve
    until the sum of unfavourable deviations is as large as possible but
    not more than 32,0 dB (16 one-third-octave bands, 100 Hz to 3150 Hz)
    or 10,0 dB (5 octave bands, 125 Hz to 2000 Hz). For impact sound an
    unfavourable deviation occurs where the **measurement exceeds** the
    reference (the sign opposite to ISO 717-1 airborne). The rating is the
    shifted reference read at 500 Hz; for octave bands it is then reduced
    by 5 dB (Clause 4.3.2). The spectrum adaptation term ``CI`` follows
    Clause A.2.1. Input values are first reduced to one decimal place
    (Clause 4.3.1, footnote 1).

    The shift search reuses the verified engine of :func:`weighted_rating`
    on the negated curves: minimising
    :math:`\sum \max(0, \text{measured} - (\text{ref} + k))` over ``k``
    equals maximising
    :math:`\sum \max(0, (-\text{ref}) + (-k) - (-\text{measured}))`, the
    airborne problem, so no separate search is duplicated.

    :param values_by_band: Measured impact levels (``Ln``, ``L'n``,
        ``L'nT``) in dB. 16 values are read as one-third-octave bands, 5
        values as octave bands.
    :param bands: ``"third-octave"``, ``"octave"`` or ``None`` to infer
        the band set from the number of values.
    :return: :class:`ImpactRatingResult` with ``rating``, ``ci`` and
        ``unfavourable_sum``.
    :raises ValueError: If the number of values does not match the band
        set, or if any value is non-finite.
    """
    data = np.asarray(values_by_band, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(_VALUES_1D_MSG)
    if not np.all(np.isfinite(data)):
        raise ValueError(_VALUES_FINITE_MSG)

    reference, limit, index_500, octave_offset, ci_bands = _resolve_impact_band_set(
        int(data.size), bands
    )
    measured = _round_half_up_tenths(data)
    ref = np.asarray(reference, dtype=np.float64)

    # Impact shift is the airborne search on the negated curves: the
    # returned shift m maximises Σ max(0, (-ref)+m-(-meas)); the impact
    # shift is k = -m, so the rating is ref_500 - m. The unfavourable sum
    # is identical under negation.
    shift, unfavourable = _best_shift(-measured, -ref, limit)
    rating = int(reference[index_500]) - round(shift) + octave_offset
    ci = _impact_ci(measured, rating, ci_bands)
    centers = _FREQ_THIRD_OCTAVE if data.size == _N_THIRD_OCTAVE_BANDS else _FREQ_OCTAVE
    return ImpactRatingResult(
        rating=rating,
        ci=ci,
        unfavourable_sum=unfavourable,
        band_centers=np.asarray(centers, dtype=np.float64),
        measured=measured,
        shifted_reference=ref - shift,
    )


#: ISO 717-2:2020 Table 4: normalized impact sound pressure level ``Ln,r,0`` of
#: the heavyweight reference floor, 16 one-third-octave bands 100 Hz to 3150 Hz,
#: in dB. Its weighted rating is ``Ln,r,0,w = 78 dB`` (Clause 5.2).
_IMPACT_REFERENCE_FLOOR = (
    67.0,
    67.5,
    68.0,
    68.5,
    69.0,
    69.5,
    70.0,
    70.5,
    71.0,
    71.5,
    72.0,
    72.0,
    72.0,
    72.0,
    72.0,
    72.0,
)
_IMPACT_REFERENCE_FLOOR_RATING = 78  # Ln,r,0,w (Table 4 / Clause 5.2)
#: Spectrum adaptation term of the bare reference floor (ISO 717-2:2020
#: Clause A.2.2): ``CI,r,0 = −11 dB``.
_IMPACT_REFERENCE_FLOOR_CI = -11


def weighted_impact_improvement(
    delta_l: Sequence[float] | np.ndarray,
) -> int:
    r"""Weighted reduction of impact level ``ΔLw`` (ISO 717-2:2020 §5).

    Relates a measured improvement spectrum ``ΔL`` to the heavyweight
    reference
    floor of Table 4: the reference level with the covering is
    :math:`L_\mathrm{n,r} = L_\mathrm{n,r,0} - \Delta L` (Formula (1)) and the weighted
    improvement is
    :math:`\Delta L_\mathrm{w} = L_\mathrm{n,r,0,w} - L_\mathrm{n,r,w} = 78 - L_\mathrm{n,r,w}`
    (Formula (2)), where ``Ln,r,w`` is
    the ISO 717-2 weighted rating of ``Ln,r`` from
    :func:`weighted_impact_rating`.

    :param delta_l: The reduction of impact sound pressure level ``ΔL`` per band,
        in dB; 16 one-third-octave values from 100 Hz to 3150 Hz (e.g. from a
        floor-covering measurement to ISO 10140-3 or ISO 16251-1).
    :return: The weighted reduction ``ΔLw``, in dB (rounded, per ISO 717-2).
    :raises ValueError: If ``delta_l`` is not 16 one-third-octave values, or is
        non-finite.
    """
    dl = np.asarray(delta_l, dtype=np.float64)
    if dl.shape != (16,):
        msg = "'delta_l' must give the 16 one-third-octave values 100-3150 Hz."
        raise ValueError(msg)
    if not np.all(np.isfinite(dl)):
        msg = "'delta_l' must contain only finite values."
        raise ValueError(msg)
    ln_r = np.asarray(_IMPACT_REFERENCE_FLOOR, dtype=np.float64) - dl
    ln_r_w = weighted_impact_rating(ln_r).rating
    return _IMPACT_REFERENCE_FLOOR_RATING - ln_r_w


def impact_improvement_adaptation_term(
    delta_l: Sequence[float] | np.ndarray,
) -> int:
    r"""Adaptation term ``CI,Δ`` of a floor covering (ISO 717-2:2020 A.2.2).

    :math:`C_{\mathrm{I},\Delta} = C_\mathrm{I,r,0} - C_\mathrm{I,r}` (Formula (A.4)) with
    :math:`C_\mathrm{I,r,0} = -11` dB (the
    bare Table 4 reference floor) and ``CI,r`` the ISO 717-2 spectrum
    adaptation term of the reference floor with the covering under test,
    :math:`L_\mathrm{n,r} = L_\mathrm{n,r,0} - \Delta L` (Formula (1)). Together with
    :func:`weighted_impact_improvement` it yields the single-number reduction
    for a flat spectrum, :math:`\Delta L_\mathrm{lin} = \Delta L_\mathrm{w} + C_{\mathrm{I},\Delta}`
    (Formula (A.5)). ISO 16251-1
    Clause 8 e) requires this term in the statement of results.

    :param delta_l: The reduction of impact sound pressure level ``ΔL`` per
        band, in dB; 16 one-third-octave values from 100 Hz to 3150 Hz.
    :return: The spectrum adaptation term ``CI,Δ``, in dB (integer).
    :raises ValueError: If ``delta_l`` is not 16 one-third-octave values, or
        is non-finite.
    """
    dl = np.asarray(delta_l, dtype=np.float64)
    if dl.shape != (16,):
        msg = "'delta_l' must give the 16 one-third-octave values 100-3150 Hz."
        raise ValueError(msg)
    if not np.all(np.isfinite(dl)):
        msg = "'delta_l' must contain only finite values."
        raise ValueError(msg)
    ln_r = np.asarray(_IMPACT_REFERENCE_FLOOR, dtype=np.float64) - dl
    ci_r = weighted_impact_rating(ln_r).ci
    return _IMPACT_REFERENCE_FLOOR_CI - ci_r


# --- ISO 717 enlarged frequency ranges and one-decimal ratings ------------


@dataclass(frozen=True)
class ExtendedWeightedRatingResult:
    """Weighted rating with the enlarged-range adaptation terms (ISO 717-1 Annex B).

    All values are integers unless the result was computed with
    ``one_decimal=True`` (the "1/10 dB for the expression of uncertainty"
    variant of Clauses 4.4/4.5), in which case they carry one decimal place.
    An extended term is ``None`` when the supplied bands do not cover its
    frequency range.

    :ivar rating: Weighted rating (``Rw``, ``R'w``, ...) from the core
        100-3150 Hz bands, in dB.
    :ivar c: Core spectrum adaptation term ``C`` (100-3150 Hz), in dB.
    :ivar ctr: Core spectrum adaptation term ``Ctr`` (100-3150 Hz), in dB.
    :ivar c_50_3150: ``C50-3150``, in dB, or ``None``.
    :ivar c_50_5000: ``C50-5000``, in dB, or ``None``.
    :ivar c_100_5000: ``C100-5000``, in dB, or ``None``.
    :ivar ctr_50_3150: ``Ctr,50-3150``, in dB, or ``None``.
    :ivar ctr_50_5000: ``Ctr,50-5000``, in dB, or ``None``.
    :ivar ctr_100_5000: ``Ctr,100-5000``, in dB, or ``None``.
    :ivar core: The integer-mode :class:`WeightedRatingResult` of the core
        bands (independent of ``one_decimal``), for plotting and the
        unfavourable-deviation sum.
    :ivar band_centers: Band centre frequencies of the full (enlarged-range)
        measured curve, in Hz. Defaults to ``None`` for
        backward-compatible construction.
    :ivar measured: The measured band quantities over the full enlarged
        range (after the one-decimal reduction of Clause 4.4), in dB.
        Defaults to ``None``.
    """

    rating: float
    c: float
    ctr: float
    c_50_3150: float | None
    c_50_5000: float | None
    c_100_5000: float | None
    ctr_50_3150: float | None
    ctr_50_5000: float | None
    ctr_100_5000: float | None
    core: WeightedRatingResult
    band_centers: np.ndarray | None = None
    measured: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Reject an enlarged-range rating whose curve and bands disagree.

        The Annex B plot shades from the ends of ``band_centers`` in to the
        ends of the core 100-3150 Hz axis, to mark which part of the curve is
        the enlarged range, and only afterwards draws ``measured`` against
        those same centres. The shading is therefore laid down before
        anything has compared the two lengths: centres that are not this
        curve's own axis put the enlarged-range mark where the curve does not
        reach, and what the caller is finally handed is matplotlib
        complaining about first dimensions.

        ``core`` is not compared here: it carries the 16 core bands whatever
        the enlarged range is, and checks its own three curves.

        :raises ValueError: if ``band_centers`` and ``measured`` disagree.
        """
        require_ranks(self, band_centers=1, measured=1)
        require_same_length(self, "band_centers", "measured")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the enlarged-range curve vs the shifted reference (Annex B).

        The measured curve is drawn over the full enlarged range, the
        ISO 717-1 reference curve (after the final shift) over the 16 core
        bands 100-3150 Hz, with the unfavourable deviations shaded on the
        core bands and the bands outside the core range marked as the
        enlarged range; the title carries ``Rw (C; Ctr)`` and every Annex B
        adaptation term the input covered. Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_extended_weighted_rating

        check_language(language)
        return plot_extended_weighted_rating(self, ax=ax, language=language, **kwargs)


@dataclass(frozen=True)
class ExtendedImpactRatingResult:
    """Weighted impact rating with ``CI,50-2500`` (ISO 717-2:2020 A.2.1 NOTE).

    Values are integers unless computed with ``one_decimal=True``.

    :ivar rating: Weighted impact rating (``Ln,w``, ...) from the core
        100-3150 Hz bands, in dB.
    :ivar ci: Core spectrum adaptation term ``CI`` (100-2500 Hz), in dB.
    :ivar ci_50_2500: Enlarged-range term ``CI,50-2500``, in dB, or ``None``
        when the supplied bands do not cover 50-2500 Hz.
    :ivar core: The integer-mode :class:`ImpactRatingResult` of the core
        bands (independent of ``one_decimal``).
    :ivar band_centers: Band centre frequencies of the full (enlarged-range)
        measured curve, in Hz. Defaults to ``None`` for
        backward-compatible construction.
    :ivar measured: The measured impact levels over the full enlarged range
        (after the one-decimal reduction of Clause 4.3), in dB. Defaults to
        ``None``.
    """

    rating: float
    ci: float
    ci_50_2500: float | None
    core: ImpactRatingResult
    band_centers: np.ndarray | None = None
    measured: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Reject an enlarged-range rating whose curve and bands disagree.

        The band centres are the record of which bands ``CI,50-2500`` was
        summed over, and the plot reads them a second time to mark which part
        of ``measured`` lies outside the core 100-3150 Hz set -- shading that
        span from the ends of the centres before it has looked at the curve
        at all. Centres that are not this curve's own axis therefore mis-mark
        the enlarged range first and fail second, in matplotlib's words
        rather than the library's.

        ``core`` is not compared here: it carries the 16 core bands whatever
        the enlarged range is, and checks its own three curves.

        :raises ValueError: if ``band_centers`` and ``measured`` disagree.
        """
        require_ranks(self, band_centers=1, measured=1)
        require_same_length(self, "band_centers", "measured")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the enlarged-range curve vs the shifted reference (ISO 717-2).

        The measured curve is drawn over the full enlarged range, the
        ISO 717-2 reference curve (after the final shift) over the 16 core
        bands 100-3150 Hz, with the unfavourable deviations (measurement
        above the reference) shaded on the core bands and the bands outside
        the core range marked as the enlarged range; the title carries the
        impact rating with ``CI`` and, when covered, ``CI,50-2500``.
        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_extended_impact_rating

        check_language(language)
        return plot_extended_impact_rating(self, ax=ax, language=language, **kwargs)


def _reduce(value: float, one_decimal: bool) -> float:
    """Round half-up to an integer, or to one decimal (ISO 80000-1 footnote)."""
    if one_decimal:
        return math.floor(value * 10.0 + 0.5) / 10.0
    return float(math.floor(value + 0.5))


def _match_bands(
    frequencies: np.ndarray, targets: tuple[float, ...]
) -> np.ndarray | None:
    """Indices of ``targets`` within ``frequencies`` (6 % tolerance), or None."""
    indices: list[int] = []
    for target in targets:
        hits = np.nonzero(np.abs(frequencies - target) <= 0.06 * target)[0]
        if hits.size != 1:
            return None
        indices.append(int(hits[0]))
    return np.asarray(indices, dtype=np.intp)


def _validated_extended_input(
    owner: str,
    values_by_band: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the extended-range input; return (measured, freqs, core_idx).

    *owner* is the entry point the caller typed, so that the message about
    the two spectra names it rather than this validator, which two entry
    points share.
    """
    data = np.asarray(values_by_band, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(_VALUES_1D_MSG)
    if not np.all(np.isfinite(data)):
        raise ValueError(_VALUES_FINITE_MSG)
    if frequencies is None:
        if data.size != len(_FREQ_THIRD_OCTAVE):
            msg = (
                "Without 'frequencies' the input must be the 16 core "
                "one-third-octave bands 100-3150 Hz; pass the band centre "
                "frequencies for an enlarged range."
            )
            raise ValueError(msg)
        freqs = np.asarray(_FREQ_THIRD_OCTAVE, dtype=np.float64)
    else:
        freqs = np.asarray(frequencies, dtype=np.float64)
        require_equal_shapes(
            owner,
            {"values_by_band": data.shape, "frequencies": freqs.shape},
            "band",
        )
        if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0.0):
            msg = "'frequencies' must contain positive values."
            raise ValueError(msg)
    core_idx = _match_bands(freqs, _FREQ_THIRD_OCTAVE)
    if core_idx is None:
        msg = (
            "The input must contain the 16 core one-third-octave bands "
            "100-3150 Hz (ISO 717 rates the single number on that range)."
        )
        raise ValueError(msg)
    return _round_half_up_tenths(data), freqs, core_idx


def weighted_rating_extended(
    values_by_band: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray | None = None,
    *,
    one_decimal: bool = False,
) -> ExtendedWeightedRatingResult:
    r"""Weighted rating with enlarged-range adaptation terms (ISO 717-1 An. B).

    Computes the weighted rating from the core one-third-octave bands
    100-3150 Hz (Clause 4.4) and, for every enlarged frequency range covered
    by the input, the additional spectrum adaptation terms of Annex B
    (``C50-3150``, ``C50-5000``, ``C100-5000`` and the ``Ctr`` counterparts)
    with the Table B.1 spectra: :math:`C_j = X_{\mathrm{A}j} - X_\mathrm{w}` where ``XAj`` sums
    over the
    bands of the enlarged range (Clause 4.5 with Annex B).

    With ``one_decimal=True`` the reference-curve shift runs in 0,1 dB steps
    and every reduction keeps one decimal place; the variant Clauses 4.4/4.5
    prescribe "for the expression of uncertainty" and ISO 12999-1:2020
    Annex B requires for the uncertainty of single-number values.

    :param values_by_band: Measured band quantities (``R``, ``R'``, ``Dn``,
        ``DnT`` ...) in dB, one-third-octave bands.
    :param frequencies: Band centre frequencies, in Hz (one per value).
        ``None`` assumes exactly the 16 core bands 100-3150 Hz. The 16 core
        bands must always be present; extended terms are formed for each
        Annex B range whose bands are all present.
    :param one_decimal: Use the 0,1 dB shift and one-decimal reductions.
    :return: An :class:`ExtendedWeightedRatingResult`.
    :raises ValueError: If the input is not one-dimensional and finite, the
        band counts differ, or the core bands are missing.
    """
    measured, freqs, core_idx = _validated_extended_input(
        "weighted_rating_extended", values_by_band, frequencies
    )
    core_measured = measured[core_idx]
    ref = np.asarray(_REF_THIRD_OCTAVE, dtype=np.float64)
    step = 0.1 if one_decimal else 1.0
    shift, _ = _best_shift(core_measured, ref, _MAX_UNFAVOURABLE_THIRD, step)
    rating = float(_REF_THIRD_OCTAVE[_INDEX_500_THIRD]) + shift

    def _term(bands: np.ndarray, spectrum: Sequence[int]) -> float:
        x_aj = -energy_sum(np.asarray(spectrum, dtype=np.float64) - bands)
        return _reduce(float(x_aj), one_decimal) - rating

    c = _term(core_measured, _SPECTRUM1_THIRD)
    ctr = _term(core_measured, _SPECTRUM2_THIRD)
    extended: dict[str, float | None] = {}
    for suffix, (band_freqs, spectrum1, spectrum2) in _EXTENDED_RANGES.items():
        idx = _match_bands(freqs, band_freqs)
        if idx is None:
            extended[f"c_{suffix}"] = None
            extended[f"ctr_{suffix}"] = None
            continue
        for name, spectrum in (
            (f"c_{suffix}", spectrum1),
            (f"ctr_{suffix}", spectrum2),
        ):
            term = _term(measured[idx], spectrum)
            extended[name] = term if one_decimal else int(term)

    return ExtendedWeightedRatingResult(
        rating=rating if one_decimal else int(rating),
        c=c if one_decimal else int(c),
        ctr=ctr if one_decimal else int(ctr),
        c_50_3150=extended["c_50_3150"],
        c_50_5000=extended["c_50_5000"],
        c_100_5000=extended["c_100_5000"],
        ctr_50_3150=extended["ctr_50_3150"],
        ctr_50_5000=extended["ctr_50_5000"],
        ctr_100_5000=extended["ctr_100_5000"],
        core=weighted_rating(np.asarray(values_by_band, dtype=np.float64)[core_idx]),
        band_centers=freqs,
        measured=measured,
    )


def weighted_impact_rating_extended(
    values_by_band: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray | None = None,
    *,
    one_decimal: bool = False,
) -> ExtendedImpactRatingResult:
    r"""Weighted impact rating with ``CI,50-2500`` (ISO 717-2:2020 A.2.1).

    Computes the weighted impact rating from the core one-third-octave bands
    100-3150 Hz (Clause 4.3) and, when the input covers 50-2500 Hz, the
    enlarged-range spectrum adaptation term ``CI,50-2500`` of the A.2.1 NOTE:
    the energetic sum runs over 50-2500 Hz instead of 100-2500 Hz in
    Formula (A.1), :math:`C_\mathrm{I} = L_\mathrm{n,sum} - 15 - L_\mathrm{n,w}`.

    With ``one_decimal=True`` the reference-curve shift runs in 0,1 dB steps
    and the sums keep one decimal place (Clauses 4.3.1/4.4; e.g. the
    reference floor yields :math:`L_\mathrm{n,r,0,w} = 77.6` dB and
    :math:`C_\mathrm{I,r,0} = -10.3` dB
    as printed in A.2.2).

    :param values_by_band: Measured impact levels (``Ln``, ``L'n``, ``L'nT``)
        in dB, one-third-octave bands.
    :param frequencies: Band centre frequencies, in Hz (one per value).
        ``None`` assumes exactly the 16 core bands 100-3150 Hz.
    :param one_decimal: Use the 0,1 dB shift and one-decimal reductions.
    :return: An :class:`ExtendedImpactRatingResult`.
    :raises ValueError: If the input is not one-dimensional and finite, the
        band counts differ, or the core bands are missing.
    """
    measured, freqs, core_idx = _validated_extended_input(
        "weighted_impact_rating_extended", values_by_band, frequencies
    )
    core_measured = measured[core_idx]
    ref = np.asarray(_REF_IMPACT_THIRD_OCTAVE, dtype=np.float64)
    step = 0.1 if one_decimal else 1.0
    # Impact shift = airborne search on the negated curves (see
    # weighted_impact_rating).
    shift, _ = _best_shift(-core_measured, -ref, _MAX_UNFAVOURABLE_THIRD, step)
    rating = float(_REF_IMPACT_THIRD_OCTAVE[_INDEX_500_THIRD]) - shift

    def _ci_over(bands: np.ndarray) -> float:
        l_sum = _reduce(float(energy_sum(bands)), one_decimal)
        return l_sum - 15.0 - rating

    ci = _ci_over(core_measured[:_CI_THIRD_OCTAVE_BANDS])
    idx = _match_bands(freqs, _CI_50_2500_FREQS)
    ci_50_2500: float | None = None
    if idx is not None:
        value = _ci_over(measured[idx])
        ci_50_2500 = value if one_decimal else int(value)

    return ExtendedImpactRatingResult(
        rating=rating if one_decimal else int(rating),
        ci=ci if one_decimal else int(ci),
        ci_50_2500=ci_50_2500,
        core=weighted_impact_rating(
            np.asarray(values_by_band, dtype=np.float64)[core_idx]
        ),
        band_centers=freqs,
        measured=measured,
    )
