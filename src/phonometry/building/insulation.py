#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Field airborne sound insulation (ISO 16283-1:2014) and impact sound
insulation (ISO 16283-2), with single-number weighted ratings and
spectrum adaptation terms (ISO 717-1 airborne, ISO 717-2 impact).

**Field quantities (ISO 16283-1:2014).** From the energy-average sound
pressure levels in the source and receiving rooms this module forms the
level difference ``D = L1 - L2`` (Clause 3.12, Formula (1)), the
standardized level difference ``DnT = D + 10 lg(T/T0)`` with the
reference reverberation time ``T0 = 0,5 s`` (Clause 3.13, Formula (2)),
and the apparent sound reduction index
``R' = D + 10 lg(S/A)`` with the Sabine equivalent absorption area
``A = 0,16 V / T`` (Clause 3.14/3.15, Formula (4) and (5)). Source and
receiving levels may be supplied already averaged (one value per band) or
as several microphone positions, which are then energy-averaged with
``10 lg( (1/n) sum 10^(Li/10) )`` (Clause 7.8, Formula (9)). All
quantities are evaluated per one-third-octave band over the core range
100 Hz to 3150 Hz (Clause 5), the caller having already applied any
background-noise correction (Clause 9.2).

**Weighted rating (ISO 717-1).** The reference-curve method of Clause 4.4
shifts the reference curve of Table 3 in 1 dB steps towards the measured
curve until the sum of unfavourable deviations (measured below the
shifted reference) is as large as possible but not more than 32,0 dB for
the 16 one-third-octave bands (100 Hz to 3150 Hz) or 10,0 dB for the 5
octave bands (125 Hz to 2000 Hz). The weighted rating (``Rw``, ``R'w``,
``Dn,w``, ``DnT,w`` ...) is the shifted reference read at 500 Hz. The
spectrum adaptation terms are ``C = XA1 - Xw`` and ``Ctr = XA2 - Xw``
with ``XAj = -10 lg sum 10^((Lij - Xi)/10)`` rounded to an integer, using
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

**Field impact quantities (ISO 16283-2).** With the tapping machine as the
impact source this module forms, from the energy-average impact sound
pressure level ``Li`` in the receiving room, the standardized impact sound
pressure level ``L'nT = Li - 10 lg(T/T0)`` with ``T0 = 0,5 s`` (Clause
3.13, Formula (1)) and the normalized impact sound pressure level
``L'n = Li + 10 lg(A/A0)`` with the Sabine absorption area ``A = 0,16 V/T``
and the reference area ``A0 = 10 m²`` (Clause 3.14, Formula (2)). Levels
may be supplied already averaged or as several microphone positions, then
energy-averaged (Clause 7.8, Formula (10)), over the core one-third-octave
range 100 Hz to 3150 Hz (Clause 5.1).

**Field façade quantities (ISO 16283-3:2016).** With an outdoor sound
source this module forms, from the level 2 m in front of the façade
``L1,2m`` and the receiving-room level ``L2``, the level difference
``D2m = L1,2m - L2`` (Clause 3.14), its standardized form
``D2m,nT = D2m + 10 lg(T/T0)`` with ``T0 = 0,5 s`` (Clause 3.15) and
normalized form ``D2m,n = D2m - 10 lg(A/A0)`` with the Sabine absorption
area ``A = 0,16 V/T`` (Clause 3.17) and reference ``A0 = 10 m²``
(Clause 3.16): the global loudspeaker / traffic quantities
``Dls,2m,*`` / ``Dtr,2m,*``. When a surface level ``L1,s`` (microphone on
the test element) with the element area ``S`` and volume are given it
forms the apparent sound reduction index
``R'45° = L1,s - L2 + 10 lg(S/A) - 1,5`` for the loudspeaker element method
(Clause 3.12) or ``R'tr,s = L1,s - L2 + 10 lg(S/A) - 3`` for the
road-traffic element method (Clause 3.13). These quantities are defined by
unnumbered formulas inline in the Clause 3 terms; positions are
energy-averaged with the surface-level formula (Clause 9.5.1, Formula (7)).
Quantities are evaluated over the core one-third-octave range 100 Hz to
3150 Hz (Clause 5), optionally extended to 50-5000 Hz. The façade quantity
is airborne, so its single-number rating uses the **ISO 717-1 airborne**
reference curve and method (Clause 2, Annex F) via :func:`weighted_rating`
unchanged.

**Weighted impact rating (ISO 717-2).** The reference-curve method of
Clause 4.3 shifts the Table 3 impact reference curve towards the measured
curve until the sum of unfavourable deviations (here where the
**measurement exceeds** the reference, the sign opposite to airborne) is
as large as possible but not more than 32,0 dB (16 one-third-octave bands)
or 10,0 dB (5 octave bands). The rating (``Ln,w``, ``L'n,w``, ``L'nT,w``)
is the shifted reference read at 500 Hz, reduced by a further 5 dB for
octave bands (Clause 4.3.2). The spectrum adaptation term
``CI = Ln,sum - 15 - Ln,w`` uses the energetic sum ``Ln,sum`` over
100 Hz to 2500 Hz (one-third octave) or 125 Hz to 2000 Hz (octave),
rounded to an integer (Clause A.2.1, Formulae (A.1) to (A.3)). The Table 3
reference values, the shifting rule and CI are identical in the 2013 and
2020 editions of ISO 717-2 (the 2020 edition only adds Annex D for the
rubber-ball heavy/soft impactor, out of scope here).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.levels_math import energy_mean, energy_sum
from .._internal.types import as_float_or_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

# --- ISO 717-1 Table 3 reference values ----------------------------------

#: One-third-octave reference values, 100 Hz to 3150 Hz (Table 3).
_REF_THIRD_OCTAVE: tuple[int, ...] = (
    33, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 56, 56, 56, 56,
)
#: Octave reference values, 125 Hz to 2000 Hz (Table 3).
_REF_OCTAVE: tuple[int, ...] = (36, 45, 52, 55, 56)

#: One-third-octave band centre frequencies, 100 Hz to 3150 Hz (16 bands).
_FREQ_THIRD_OCTAVE: tuple[float, ...] = (
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0,
)
#: Octave band centre frequencies, 125 Hz to 2000 Hz (5 bands).
_FREQ_OCTAVE: tuple[float, ...] = (125.0, 250.0, 500.0, 1000.0, 2000.0)

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

# --- ISO 717-2 Table 3 impact reference values ---------------------------

#: One-third-octave impact reference values, 100 Hz to 3150 Hz (Table 3).
_REF_IMPACT_THIRD_OCTAVE: tuple[int, ...] = (
    62, 62, 62, 62, 62, 62, 61, 60, 59, 58, 57, 54, 51, 48, 45, 42,
)
#: Octave impact reference values, 125 Hz to 2000 Hz (Table 3).
_REF_IMPACT_OCTAVE: tuple[int, ...] = (67, 67, 65, 62, 49)

#: Octave-band single-number reduction applied to L'n,w / L'nT,w
#: (ISO 717-2 Clause 4.3.2): the shifted reference at 500 Hz minus 5 dB.
_IMPACT_OCTAVE_OFFSET = -5

#: One-third-octave band count for CI (100 Hz to 2500 Hz, excludes 3150 Hz).
_CI_THIRD_OCTAVE_BANDS = 15

#: Reference absorption area A0 for the normalized level (Clause 3.14).
_A0_IMPACT = 10.0

# --- ISO 16283-3 façade sound insulation ---------------------------------

#: Reference absorption area A0 for D2m,n (Clause 3.16, dwellings).
_A0_FACADE = 10.0

#: Angle-of-incidence corrections in the apparent sound reduction index:
#: -1,5 dB for the loudspeaker method at 45° (Clause 3.12) and -3 dB for
#: the road-traffic method with all-angle incidence (Clause 3.13).
_FACADE_CORRECTION = {"loudspeaker": 1.5, "road_traffic": 3.0}

# --- ISO 717-1 Table 4 spectra (A-weighted, normalized to 0 dB) ----------

#: Spectrum No. 1 (pink noise, for C), one-third octave 100-3150 Hz.
_SPECTRUM1_THIRD: tuple[int, ...] = (
    -29, -26, -23, -21, -19, -17, -15, -13, -12, -11, -10, -9, -9, -9, -9, -9,
)
#: Spectrum No. 2 (urban traffic, for Ctr), one-third octave 100-3150 Hz.
_SPECTRUM2_THIRD: tuple[int, ...] = (
    -20, -20, -18, -16, -15, -14, -13, -12, -11, -9, -8, -9, -10, -11, -13, -15,
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
    50.0, 63.0, 80.0, *_FREQ_THIRD_OCTAVE, 4000.0, 5000.0,
)
#: Spectrum No. 1 column for C50-3150 (19 bands, 50-3150 Hz).
_SPECTRUM1_50_3150: tuple[int, ...] = (
    -40, -36, -33, -29, -26, -23, -21, -19, -17, -15,
    -13, -12, -11, -10, -9, -9, -9, -9, -9,
)
#: Spectrum No. 1 column for C50-5000 and C100-5000 (21 bands, 50-5000 Hz;
#: the 100-5000 Hz range uses the same column restricted to its bands).
_SPECTRUM1_50_5000: tuple[int, ...] = (
    -41, -37, -34, -30, -27, -24, -22, -20, -18, -16, -14,
    -13, -12, -11, -10, -10, -10, -10, -10, -10, -10,
)
#: Spectrum No. 2 column for Ctr in any enlarged range (21 bands, 50-5000 Hz).
_SPECTRUM2_50_5000: tuple[int, ...] = (
    -25, -23, -21, -20, -20, -18, -16, -15, -14, -13, -12,
    -11, -9, -8, -9, -10, -11, -13, -15, -16, -18,
)

#: The enlarged one-third-octave adaptation ranges of ISO 717-1:2020 Annex B
#: (airborne): descriptor suffix -> (band frequencies, spectrum No. 1 levels,
#: spectrum No. 2 levels).
_EXTENDED_RANGES: dict[str, tuple[
    tuple[float, ...], tuple[int, ...], tuple[int, ...]
]] = {
    "50_3150": (
        _FREQ_50_5000[:19], _SPECTRUM1_50_3150, _SPECTRUM2_50_5000[:19]
    ),
    "50_5000": (_FREQ_50_5000, _SPECTRUM1_50_5000, _SPECTRUM2_50_5000),
    "100_5000": (
        _FREQ_50_5000[3:], _SPECTRUM1_50_5000[3:], _SPECTRUM2_50_5000[3:]
    ),
}

#: The enlarged CI summation range of ISO 717-2:2020 A.2.1 NOTE: 50-2500 Hz
#: (18 one-third-octave bands).
_CI_50_2500_FREQS: tuple[float, ...] = _FREQ_50_5000[:18]


_VALUES_1D_MSG = "'values_by_band' must be one-dimensional."
_VALUES_FINITE_MSG = "'values_by_band' must contain only finite values."


@dataclass(frozen=True)
class AirborneInsulationResult:
    """Per-band field airborne sound insulation (ISO 16283-1:2014).

    :ivar d: Level difference ``D = L1 - L2`` per band, in dB
        (Clause 3.12, Formula (1)).
    :ivar dnt: Standardized level difference ``DnT`` per band, in dB
        (Clause 3.13, Formula (2)).
    :ivar r_prime: Apparent sound reduction index ``R'`` per band, in dB
        (Clause 3.14, Formula (4)), or ``None`` when the partition area
        and receiving-room volume were not supplied.
    :ivar l1: Energy-average source-room levels the quantities were formed
        from, in dB (after any position averaging, Formula (9)). Defaults
        to ``None`` for backward-compatible construction.
    :ivar l2: Energy-average receiving-room levels, in dB. Defaults to
        ``None``.
    :ivar t2: Receiving-room reverberation time per band, in seconds.
        Defaults to ``None``.
    :ivar t0: Reference reverberation time ``T0`` used for ``DnT``, in
        seconds. Defaults to ``None``.
    """

    d: np.ndarray
    dnt: np.ndarray
    r_prime: np.ndarray | None
    l1: np.ndarray | None = None
    l2: np.ndarray | None = None
    t2: np.ndarray | None = None
    t0: float | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-band insulation quantities (``DnT``, ``D``, ``R'``).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_airborne_insulation

        check_language(language)
        return plot_airborne_insulation(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        quantity: str = "dnt",
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 16283-1 field airborne sound-insulation report to a PDF.

        Writes the one-page field test report of ISO 16283-1:2014 Clause 14
        in the layout of the recommended Annex B form: the standard-basis
        line, an optional metadata header block (client, construction, room
        volumes, partition area ...), the one-third-octave table beside the
        measured-versus-shifted-reference curve, the boxed field rating
        ``DnT,w (C; Ctr)`` or ``R'w (C; Ctr)`` (evaluated per ISO 717-1 over
        the 16 core bands), the mandatory field-method statement, an optional
        verdict row and a footer with the identity block and disclaimer.

        :param path: Destination path of the PDF file.
        :param quantity: The reported field quantity: ``"dnt"`` (default, the
            standardized level difference of Annex B Figure B.1) or
            ``"r_prime"`` (the apparent sound reduction index of Figure B.2;
            requires the result to carry ``r_prime``).
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table shows the measurement chain
            per band (energy-average ``L1`` and ``L2``, reverberation time
            ``T`` and the quantity) instead of the two-column ``f | value``
            form; it requires the result to carry ``l1``, ``l2`` and ``t2``
            (populated by :func:`airborne_insulation`).
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` or ``quantity`` is unknown, the
            selected quantity is not available, the result does not hold the
            16 core one-third-octave bands (100 Hz to 3150 Hz) the ISO 717-1
            rating needs, or ``verbose=True`` without the per-band chain.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``).
        """
        return _render_iso16283(
            self, path, quantity=quantity, metadata=metadata, engine=engine,
            verbose=verbose, language=language,
        )


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
    :ivar quantity: ``"airborne"`` (ISO 717-1, sound reduction index) or
        ``"impact"`` (ISO 717-2), selecting the labels of the ISO 717
        Annex C report. Defaults to ``"airborne"``.
    """

    rating: int
    c: int
    ctr: int
    unfavourable_sum: float
    band_centers: np.ndarray | None = None
    measured: np.ndarray | None = None
    shifted_reference: np.ndarray | None = None
    quantity: str = "airborne"

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the measured curve vs the shifted reference (ISO 717-1).

        Unfavourable deviations (reference above measurement) are shaded and
        ``Rw (C; Ctr)`` annotated. Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_weighted_rating

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
            (``pip install phonometry[report]``).
        """
        return _render_iso717(
            self, path, metadata=metadata, engine=engine, verbose=verbose,
            language=language, symbol=symbol,
        )


@dataclass(frozen=True)
class ImpactInsulationResult:
    """Per-band field impact sound insulation (ISO 16283-2).

    :ivar l_n_t: Standardized impact sound pressure level
        ``L'nT = Li - 10 lg(T/T0)`` per band, in dB (Clause 3.13,
        Formula (1)).
    :ivar l_n: Normalized impact sound pressure level
        ``L'n = Li + 10 lg(A/A0)`` per band, in dB (Clause 3.14,
        Formula (2)), or ``None`` when the receiving-room volume was not
        supplied.
    :ivar li: Energy-average impact sound pressure levels the quantities
        were formed from, in dB (after any position averaging,
        Formula (10)). Defaults to ``None`` for backward-compatible
        construction.
    :ivar t2: Receiving-room reverberation time per band, in seconds.
        Defaults to ``None``.
    :ivar t0: Reference reverberation time ``T0`` used for ``L'nT``, in
        seconds. Defaults to ``None``.
    """

    l_n_t: np.ndarray
    l_n: np.ndarray | None
    li: np.ndarray | None = None
    t2: np.ndarray | None = None
    t0: float | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-band impact levels (``L'nT`` and, if present, ``L'n``).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_impact_insulation

        check_language(language)
        return plot_impact_insulation(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        quantity: str = "l_n_t",
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 16283-2 field impact sound-insulation report to a PDF.

        Writes the one-page field test report of ISO 16283-2:2020 Clause 14
        in the layout of the recommended Annex C form: the standard-basis
        line, an optional metadata header block, the one-third-octave table
        beside the measured-versus-shifted-reference curve, the boxed field
        rating ``L'nT,w (CI)`` or ``L'n,w (CI)`` (evaluated per ISO 717-2
        over the 16 core bands), the mandatory field-method statement, an
        optional verdict row and a footer with the identity block and
        disclaimer.

        :param path: Destination path of the PDF file.
        :param quantity: The reported field quantity: ``"l_n_t"`` (default,
            the standardized level of Annex C Figure C.1) or ``"l_n"`` (the
            normalized level of Figure C.2; requires the result to carry
            ``l_n``).
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table shows the measurement chain
            per band (energy-average ``Li``, reverberation time ``T`` and the
            quantity) instead of the two-column ``f | value`` form; it
            requires the result to carry ``li`` and ``t2`` (populated by
            :func:`impact_insulation`).
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` or ``quantity`` is unknown, the
            selected quantity is not available, the result does not hold the
            16 core one-third-octave bands (100 Hz to 3150 Hz) the ISO 717-2
            rating needs, or ``verbose=True`` without the per-band chain.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``).
        """
        return _render_iso16283(
            self, path, quantity=quantity, metadata=metadata, engine=engine,
            verbose=verbose, language=language,
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
    quantity: str = "impact"

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the measured curve vs the shifted reference (ISO 717-2).

        Unfavourable deviations (measurement above the reference, the sign
        opposite to airborne) are shaded and ``Ln,w (CI)`` annotated.
        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_impact_rating

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
            (``pip install phonometry[report]``).
        """
        return _render_iso717(
            self, path, metadata=metadata, engine=engine, verbose=verbose,
            language=language, symbol=symbol,
        )


@dataclass(frozen=True)
class FacadeInsulationResult:
    """Per-band field façade sound insulation (ISO 16283-3).

    :ivar d_2m: Level difference ``D2m = L1,2m - L2`` per band, in dB
        (Clause 3.14; ``Dls,2m`` loudspeaker, ``Dtr,2m`` traffic).
    :ivar d_2m_nt: Standardized level difference
        ``D2m,nT = D2m + 10 lg(T/T0)`` per band, in dB (Clause 3.15).
    :ivar d_2m_n: Normalized level difference
        ``D2m,n = D2m - 10 lg(A/A0)`` per band, in dB (Clause 3.16), or
        ``None`` when the receiving-room volume was not supplied.
    :ivar r_prime: Apparent sound reduction index ``R'45°`` (loudspeaker,
        Clause 3.12) or ``R'tr,s`` (road traffic, Clause 3.13) per band, in
        dB, or ``None`` unless a surface level
        together with the element area and receiving-room volume were
        supplied.
    :ivar frequencies: Band centre frequencies, in Hz, or ``None``.
    :ivar method: The sound source of the measurement: ``"loudspeaker"``
        (45° incidence) or ``"road_traffic"``. It selects which apparent
        index ``r_prime`` is (``R'45`` or ``R'tr,s``) and how the report
        labels it.
    """

    d_2m: np.ndarray
    d_2m_nt: np.ndarray
    d_2m_n: np.ndarray | None
    r_prime: np.ndarray | None
    frequencies: np.ndarray | None = None
    method: str = "loudspeaker"

    def __post_init__(self) -> None:
        if self.method not in _FACADE_CORRECTION:
            raise ValueError(
                "'method' must be 'loudspeaker' or 'road_traffic', got "
                f"{self.method!r}."
            )

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-band façade insulation profile (ISO 16283-3).

        Draws the standardized level difference and any other available
        quantities (``D2m``, ``D2m,n``, ``R'``) against frequency. Requires
        matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_facade_insulation

        check_language(language)
        return plot_facade_insulation(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        quantity: str = "d_2m_nt",
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 16283-3 field facade sound-insulation report to a PDF.

        Writes the one-page field facade test report of ISO 16283-3:2016: the
        standard-basis line, an optional metadata header block, the
        one-third-octave table beside the measured-versus-shifted-reference
        curve, the boxed ISO 717-1 field rating of the reported ``quantity``
        (``D2m,nT,w`` for ``d_2m_nt``, ``D2m,n,w`` for ``d_2m_n`` or, for
        ``r_prime``, ``R'45,w`` / ``R'tr,s,w`` following the result's
        ``method``, each with ``C; Ctr`` and evaluated over the 16 core
        bands), the mandatory field-method statement, an optional verdict row
        and a footer with the identity block and disclaimer.

        :param path: Destination path of the PDF file.
        :param quantity: The reported facade quantity: ``"d_2m_nt"`` (default,
            the standardized facade level difference ``D2m,nT``), ``"d_2m_n"``
            (the normalized facade level difference ``D2m,n``; requires the
            result to carry ``d_2m_n``) or ``"r_prime"`` (the apparent sound
            reduction index; requires the result to carry ``r_prime``). The
            ``r_prime`` fiche is labelled ``R'45`` (Clause 3.12, loudspeaker
            method) or ``R'tr,s`` (Clause 3.13, road traffic method)
            according to the result's ``method``.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`;
            ``None`` produces a lightweight fiche (body, rating and
            disclaimer only).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table shows the ISO 717 evaluation
            per band (the reported quantity, the shifted reference and the
            unfavourable deviation) instead of the two-column ``f | value``
            form.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` or ``quantity`` is unknown, the
            selected quantity is not available, or the result does not hold
            the 16 core one-third-octave bands (100 Hz to 3150 Hz) the
            ISO 717-1 rating needs.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``).
        """
        return _render_iso16283_facade(
            self, path, quantity=quantity, metadata=metadata, engine=engine,
            verbose=verbose, language=language,
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
    from .._i18n import check_language

    check_language(language)
    if engine != "reportlab":
        raise ValueError(
            f"Unknown report engine {engine!r}; only 'reportlab' is supported."
        )
    if (
        result.band_centers is None
        or result.measured is None
        or result.shifted_reference is None
    ):
        raise ValueError(
            "report() needs the per-band data ('band_centers', 'measured', "
            "'shifted_reference'); build the rating with weighted_rating() or "
            "weighted_impact_rating() so they are populated."
        )
    from .._report.iso717 import render_iso717_report

    return render_iso717_report(
        result, path, metadata=metadata, verbose=verbose, language=language,
        symbol=symbol,
    )


#: The field quantities each result kind can report, mapping the ``quantity``
#: argument of ``report()`` to the attribute holding the per-band curve and
#: the per-band chain columns ``verbose=True`` needs.
_FIELD_QUANTITIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "AirborneInsulationResult": (("dnt", "r_prime"), ("l1", "l2", "t2")),
    "ImpactInsulationResult": (("l_n_t", "l_n"), ("li", "t2")),
}


def _render_iso16283(
    result: AirborneInsulationResult | ImpactInsulationResult,
    path: str,
    *,
    quantity: str,
    metadata: ReportMetadata | None,
    engine: str,
    verbose: bool,
    language: str,
) -> str:
    """Validate the field-report request and delegate to the renderer.

    Shared by :meth:`AirborneInsulationResult.report` and
    :meth:`ImpactInsulationResult.report`: it rejects unknown engines and
    quantities, requires the 16 core one-third-octave bands (100 Hz to
    3150 Hz, ISO 16283-1/-2 Clause 5) so the ISO 717 rating is defined,
    evaluates that rating, and calls the reportlab renderer (which raises a
    clear :class:`ImportError` when reportlab is absent).
    """
    from .._i18n import check_language

    check_language(language)
    if engine != "reportlab":
        raise ValueError(
            f"Unknown report engine {engine!r}; only 'reportlab' is supported."
        )
    kind = type(result).__name__
    quantities, chain_attrs = _FIELD_QUANTITIES[kind]
    if quantity not in quantities:
        expected = " or ".join(repr(q) for q in quantities)
        raise ValueError(
            f"Unknown field quantity {quantity!r}; expected {expected}."
        )
    curve = getattr(result, quantity)
    if curve is None:
        raise ValueError(
            f"The result carries no {quantity!r} curve; build it with the "
            "'area'/'volume' arguments so the normalized quantity is "
            "computed."
        )
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (len(_FREQ_THIRD_OCTAVE),):
        raise ValueError(
            "The field report rates the 16 core one-third-octave bands "
            f"(100 Hz to 3150 Hz, ISO 16283 Clause 5); got {curve.shape} "
            "band values."
        )
    if verbose:
        missing = [a for a in chain_attrs if getattr(result, a) is None]
        if missing:
            raise ValueError(
                "verbose=True needs the per-band measurement chain "
                f"({', '.join(chain_attrs)}); {', '.join(missing)} "
                "missing. Build the result with airborne_insulation() / "
                "impact_insulation() so they are populated."
            )
    if kind == "ImpactInsulationResult":
        rating: WeightedRatingResult | ImpactRatingResult = (
            weighted_impact_rating(curve)
        )
    else:
        rating = weighted_rating(curve)

    from .._report.iso16283 import render_iso16283_report

    return render_iso16283_report(
        result, rating, path, quantity=quantity, metadata=metadata,
        verbose=verbose, language=language,
    )


#: The reportable ISO 16283-3 facade quantities, mapping the ``quantity``
#: argument of :meth:`FacadeInsulationResult.report` to the attribute holding
#: the per-band curve.
_FACADE_FIELD_QUANTITIES: tuple[str, ...] = ("d_2m_nt", "d_2m_n", "r_prime")


def _render_iso16283_facade(
    result: FacadeInsulationResult,
    path: str,
    *,
    quantity: str,
    metadata: ReportMetadata | None,
    engine: str,
    verbose: bool,
    language: str,
) -> str:
    """Validate the facade-report request and delegate to the renderer.

    Rejects unknown engines and quantities, requires the 16 core
    one-third-octave bands (100 Hz to 3150 Hz) so the ISO 717-1 rating is
    defined, evaluates that rating and calls the reportlab renderer (which
    raises a clear :class:`ImportError` when reportlab is absent).
    """
    from .._i18n import check_language

    check_language(language)
    if engine != "reportlab":
        raise ValueError(
            f"Unknown report engine {engine!r}; only 'reportlab' is supported."
        )
    if quantity not in _FACADE_FIELD_QUANTITIES:
        expected = " or ".join(repr(q) for q in _FACADE_FIELD_QUANTITIES)
        raise ValueError(
            f"Unknown facade quantity {quantity!r}; expected {expected}."
        )
    curve = getattr(result, quantity)
    if curve is None:
        raise ValueError(
            f"The result carries no {quantity!r} curve; build it with the "
            "'volume' (and, for R', 'surface_level'/'area') arguments so the "
            "quantity is computed."
        )
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (len(_FREQ_THIRD_OCTAVE),):
        raise ValueError(
            "The facade field report rates the 16 core one-third-octave bands "
            f"(100 Hz to 3150 Hz, ISO 16283-3 Clause 5); got {curve.shape} "
            "band values."
        )
    rating = weighted_rating(curve)

    from .._report.iso16283 import render_iso16283_facade_report

    return render_iso16283_facade_report(
        result, rating, path, quantity=quantity, metadata=metadata,
        verbose=verbose, language=language,
    )


def _round_half_up_tenths(values: np.ndarray) -> np.ndarray:
    """Reduce levels to one decimal place (ISO 717-1 Clause 4.4, note 1).

    Rounds each value to the nearest tenth of a decibel, half away from
    zero (``floor(x*10 + 0,5)/10`` for non-negative values, mirrored for
    negative ones).

    .. note::
        This rounds negative halves *away from zero* (−0,05 → −0,1), whereas
        the adaptation-term reductions (:func:`_adaptation_term`,
        :func:`_impact_ci`) use the ISO 80000-1 footnote form
        ``floor(x + 0,5)`` which rounds them *towards* +∞ (−0,5 → 0). The two
        conventions differ only for exactly-half negative values, which do
        not occur with realistic (positive-level) insulation data; the
        difference is documented here rather than unified so each function
        keeps the literal form of its clause.
    """
    rounded: np.ndarray = np.sign(values) * np.floor(np.abs(values) * 10.0 + 0.5) / 10.0
    return rounded


def energy_average_level(
    levels: Sequence[float] | np.ndarray, axis: int = -1
) -> np.ndarray | float:
    """
    Energy-average sound pressure level (ISO 16283-1:2014, Formula (9)).

    Combines sound pressure levels measured at several microphone
    positions into ``L = 10 lg( (1/n) sum_i 10^(Li/10) )``.

    :param levels: Sound pressure levels, in dB, at the ``n`` positions to
        be averaged along ``axis``.
    :param axis: Axis over which to average (default the last axis).
    :return: The energy-average level, in dB; a scalar ``float`` when the
        result is zero-dimensional, otherwise an array.
    :raises ValueError: If ``levels`` is empty or contains non-finite
        values.
    """
    data = np.asarray(levels, dtype=np.float64)
    if data.size == 0:
        raise ValueError("'levels' must not be empty.")
    if not np.all(np.isfinite(data)):
        raise ValueError("'levels' must contain only finite values.")
    return as_float_or_array(energy_mean(data, axis=axis))


def _as_band_levels(
    levels: Sequence[float] | np.ndarray, name: str
) -> np.ndarray:
    """Coerce room levels to per-band values, energy-averaging positions.

    A one-dimensional input is taken as one already-averaged level per
    band; a two-dimensional input is read as ``(positions, bands)`` and
    energy-averaged over the positions (Formula (9)).
    """
    data = np.asarray(levels, dtype=np.float64)
    if data.ndim == 1:
        out = data
    elif data.ndim == 2:
        out = np.asarray(energy_average_level(data, axis=0), dtype=np.float64)
    else:
        raise ValueError(f"'{name}' must be 1-D or 2-D (positions x bands).")
    if out.size == 0:
        raise ValueError(f"'{name}' must not be empty.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"'{name}' must contain only finite values.")
    return out


def airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    area: float | None = None,
    volume: float | None = None,
    t0: float = 0.5,
) -> AirborneInsulationResult:
    """
    Field airborne sound insulation per ISO 16283-1:2014.

    Computes, per frequency band, the level difference ``D = L1 - L2``
    (Formula (1)), the standardized level difference
    ``DnT = D + 10 lg(T/T0)`` (Formula (2)) and, when the partition area
    and receiving-room volume are given, the apparent sound reduction
    index ``R' = D + 10 lg(S/A)`` with ``A = 0,16 V / T`` (Formula (4)
    and (5)).

    ``l1`` and ``l2`` may be one value per band (already energy-averaged)
    or a two-dimensional ``(positions, bands)`` array, in which case the
    positions are energy-averaged with Formula (9). The band levels are
    assumed already corrected for background noise (Clause 9.2).

    :param l1: Source-room sound pressure levels, in dB.
    :param l2: Receiving-room sound pressure levels, in dB.
    :param t2: Receiving-room reverberation time per band, in seconds.
    :param area: Area ``S`` of the common partition, in m² (optional;
        required together with ``volume`` for ``R'``).
    :param volume: Receiving-room volume ``V``, in m³ (optional; required
        together with ``area`` for ``R'``).
    :param t0: Reference reverberation time ``T0``, in seconds (default
        0,5 s for dwellings, Clause 3.13).
    :return: :class:`AirborneInsulationResult` with ``d``, ``dnt`` and
        ``r_prime`` (the latter ``None`` unless ``area`` and ``volume``
        are both given).
    :raises ValueError: If the band counts of ``l1``, ``l2`` and ``t2``
        differ, if only one of ``area``/``volume`` is supplied, if
        ``t2``/``t0`` are not positive, or if inputs are non-finite.
    """
    l1_bands = _as_band_levels(l1, "l1")
    l2_bands = _as_band_levels(l2, "l2")
    t = np.asarray(t2, dtype=np.float64)

    if not (l1_bands.shape == l2_bands.shape == t.shape):
        raise ValueError("'l1', 'l2' and 't2' must share the same band count.")
    if t.ndim != 1:
        raise ValueError("'t2' must be one-dimensional (one value per band).")
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("'t2' must contain positive, finite values.")
    if t0 <= 0.0:
        raise ValueError("'t0' must be positive.")

    d = l1_bands - l2_bands
    dnt = d + 10.0 * np.log10(t / t0)

    if (area is None) != (volume is None):
        raise ValueError(
            "'area' and 'volume' must be given together to compute R'."
        )
    r_prime: np.ndarray | None = None
    if area is not None and volume is not None:
        if area <= 0.0 or volume <= 0.0:
            raise ValueError("'area' and 'volume' must be positive.")
        absorption = 0.16 * volume / t
        r_prime = d + 10.0 * np.log10(area / absorption)

    return AirborneInsulationResult(
        d=d, dnt=dnt, r_prime=r_prime,
        l1=l1_bands, l2=l2_bands, t2=t, t0=t0,
    )


def _resolve_band_set(
    n: int, bands: str | None
) -> tuple[tuple[int, ...], float, int, tuple[int, ...], tuple[int, ...]]:
    """Select the reference curve, bound and spectra for the band set.

    :return: ``(reference, max_unfavourable, index_500, spectrum1,
        spectrum2)``.
    """
    if bands == "third-octave" or (bands is None and n == 16):
        if n != 16:
            raise ValueError(
                "One-third-octave rating needs 16 bands (100-3150 Hz), "
                f"got {n}."
            )
        return (
            _REF_THIRD_OCTAVE,
            _MAX_UNFAVOURABLE_THIRD,
            _INDEX_500_THIRD,
            _SPECTRUM1_THIRD,
            _SPECTRUM2_THIRD,
        )
    if bands == "octave" or (bands is None and n == 5):
        if n != 5:
            raise ValueError(
                "Octave rating needs 5 bands (125-2000 Hz), " f"got {n}."
            )
        return (
            _REF_OCTAVE,
            _MAX_UNFAVOURABLE_OCTAVE,
            _INDEX_500_OCTAVE,
            _SPECTRUM1_OCTAVE,
            _SPECTRUM2_OCTAVE,
        )
    if bands is not None:
        raise ValueError("'bands' must be 'third-octave', 'octave' or None.")
    raise ValueError(
        "Expected 16 one-third-octave (100-3150 Hz) or 5 octave "
        f"(125-2000 Hz) values, got {n}."
    )


def _best_shift(
    measured: np.ndarray,
    reference: np.ndarray,
    limit: float,
    step: float = 1.0,
) -> tuple[float, float]:
    """Largest ``step``-sized shift with unfavourable-deviation sum in ``limit``.

    Shifts the reference by multiples of ``step`` and returns the largest
    shift for which ``sum max(0, reference + shift - measured) <= limit``
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
    if scale < 1 or abs(scale * step - 1.0) > 1e-12:
        raise ValueError("'step' must divide 1 dB exactly (e.g. 1.0 or 0.1).")
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
    """
    Single-number weighted rating and C / Ctr per ISO 717-1.

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
    centers = _FREQ_THIRD_OCTAVE if data.size == 16 else _FREQ_OCTAVE
    return WeightedRatingResult(
        rating=rating,
        c=c,
        ctr=ctr,
        unfavourable_sum=unfavourable,
        band_centers=np.asarray(centers, dtype=np.float64),
        measured=measured,
        shifted_reference=ref + shift,
    )


# --- ISO 16283-2 field impact sound insulation ---------------------------


def impact_insulation(
    li: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
    t0: float = 0.5,
) -> ImpactInsulationResult:
    """
    Field impact sound insulation per ISO 16283-2 (tapping machine).

    Computes, per frequency band, the standardized impact sound pressure
    level ``L'nT = Li - 10 lg(T/T0)`` (Formula (1)) and, when the
    receiving-room volume is given, the normalized impact sound pressure
    level ``L'n = Li + 10 lg(A/A0)`` with the Sabine equivalent absorption
    area ``A = 0,16 V / T`` (Formula (6)) and the reference absorption area
    ``A0 = 10 m²`` (Formula (2)).

    ``li`` may be one value per band (already energy-averaged) or a
    two-dimensional ``(positions, bands)`` array, in which case the
    positions are energy-averaged with Formula (10). The band levels are
    assumed already corrected for background noise (Clause 9).

    :param li: Energy-average impact sound pressure levels, in dB.
    :param t2: Receiving-room reverberation time per band, in seconds.
    :param volume: Receiving-room volume ``V``, in m³ (optional; required
        for ``L'n``).
    :param t0: Reference reverberation time ``T0``, in seconds (default
        0,5 s for dwellings, Clause 3.13).
    :return: :class:`ImpactInsulationResult` with ``l_n_t`` and ``l_n``
        (the latter ``None`` unless ``volume`` is given).
    :raises ValueError: If the band counts of ``li`` and ``t2`` differ, if
        ``t2``/``t0``/``volume`` are not positive, or if inputs are
        non-finite.
    """
    li_bands = _as_band_levels(li, "li")
    t = np.asarray(t2, dtype=np.float64)

    if li_bands.shape != t.shape:
        raise ValueError("'li' and 't2' must share the same band count.")
    if t.ndim != 1:
        raise ValueError("'t2' must be one-dimensional (one value per band).")
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("'t2' must contain positive, finite values.")
    if t0 <= 0.0:
        raise ValueError("'t0' must be positive.")

    l_n_t = li_bands - 10.0 * np.log10(t / t0)

    l_n: np.ndarray | None = None
    if volume is not None:
        if volume <= 0.0:
            raise ValueError("'volume' must be positive.")
        absorption = 0.16 * volume / t
        l_n = li_bands + 10.0 * np.log10(absorption / _A0_IMPACT)

    return ImpactInsulationResult(
        l_n_t=l_n_t, l_n=l_n, li=li_bands, t2=t, t0=t0,
    )


def facade_insulation(
    l1_2m: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    area: float | None = None,
    volume: float | None = None,
    surface_level: Sequence[float] | np.ndarray | None = None,
    method: str = "loudspeaker",
    t0: float = 0.5,
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> FacadeInsulationResult:
    """
    Field façade sound insulation per ISO 16283-3:2016.

    Computes, per frequency band, the global-method level difference
    ``D2m = L1,2m - L2`` (Clause 3.14), its standardized form
    ``D2m,nT = D2m + 10 lg(T/T0)`` (Clause 3.15) and, when the
    receiving-room volume is given, its normalized form
    ``D2m,n = D2m - 10 lg(A/A0)`` with the Sabine equivalent absorption
    area ``A = 0,16 V/T`` (Clause 3.17) and ``A0 = 10 m²`` (Clause 3.16).
    When a surface level ``L1,s`` (microphone on the test element),
    together with the element area ``S`` and the volume, is supplied it
    also computes the apparent sound reduction index of the element
    method: ``R'45° = L1,s - L2 + 10 lg(S/A) - 1,5`` for a loudspeaker
    source (Clause 3.12) or ``R'tr,s = L1,s - L2 + 10 lg(S/A) - 3`` for a
    road-traffic source (Clause 3.13). The defining formulas are unnumbered
    inline in the Clause 3 terms.

    ``l1_2m``, ``l2`` and ``surface_level`` may be one value per band
    (already energy-averaged) or a two-dimensional ``(positions, bands)``
    array, in which case the positions are energy-averaged with the
    surface-level formula (Clause 9.5.1, Formula (7)). Band levels are
    assumed already corrected for background
    noise. The single-number rating uses the ISO 717-1 airborne reference
    curve (Annex F); pass the desired 16-band quantity to
    :func:`weighted_rating`.

    :param l1_2m: Outdoor sound pressure levels 2 m in front of the façade,
        in dB.
    :param l2: Receiving-room sound pressure levels, in dB.
    :param t2: Receiving-room reverberation time per band, in seconds.
    :param area: Area ``S`` of the test element, in m² (optional; required
        with ``volume`` and ``surface_level`` for ``R'``).
    :param volume: Receiving-room volume ``V``, in m³ (optional; required
        for ``D2m,n`` and for ``R'``).
    :param surface_level: Outdoor surface level ``L1,s`` on the test
        element, in dB (optional; required with ``area`` and ``volume`` for
        ``R'``).
    :param method: ``"loudspeaker"`` (45° incidence, -1,5 dB) or
        ``"road_traffic"`` (all-angle incidence, -3 dB); selects the ``R'``
        correction (Clause 3.12 / 3.13).
    :param t0: Reference reverberation time ``T0``, in seconds (default
        0,5 s for dwellings, Clause 3.15).
    :param frequencies: Optional band centre frequencies, in Hz, carried
        on the result for plotting.
    :return: :class:`FacadeInsulationResult` with ``d_2m``, ``d_2m_nt``,
        ``d_2m_n`` (``None`` unless ``volume`` is given) and ``r_prime``
        (``None`` unless ``surface_level``, ``area`` and ``volume`` are all
        given).
    :raises ValueError: If band counts differ, if ``method`` is unknown, if
        ``t2``/``t0``/``area``/``volume`` are not positive, if ``area`` is
        given without ``surface_level``, if ``surface_level`` and ``area`` are
        given without ``volume``, if ``frequencies`` is given with a length
        that differs from the band count, or if inputs are non-finite.
        Supplying ``surface_level`` alone is not an error: ``r_prime`` simply
        stays ``None``.
    """
    if method not in _FACADE_CORRECTION:
        raise ValueError(
            "'method' must be 'loudspeaker' or 'road_traffic', got "
            f"{method!r}."
        )

    l1_bands = _as_band_levels(l1_2m, "l1_2m")
    l2_bands = _as_band_levels(l2, "l2")
    t = np.asarray(t2, dtype=np.float64)

    if not (l1_bands.shape == l2_bands.shape == t.shape):
        raise ValueError(
            "'l1_2m', 'l2' and 't2' must share the same band count."
        )
    if t.ndim != 1:
        raise ValueError("'t2' must be one-dimensional (one value per band).")
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("'t2' must contain positive, finite values.")
    if t0 <= 0.0:
        raise ValueError("'t0' must be positive.")

    d_2m = l1_bands - l2_bands
    d_2m_nt = d_2m + 10.0 * np.log10(t / t0)

    if volume is not None and volume <= 0.0:
        raise ValueError("'volume' must be positive.")
    if area is not None and area <= 0.0:
        raise ValueError("'area' must be positive.")
    if area is not None and surface_level is None:
        raise ValueError(
            "'area' requires 'surface_level' to compute the apparent sound "
            "reduction index R'."
        )
    if surface_level is not None and area is not None and volume is None:
        raise ValueError(
            "'volume' is required with 'surface_level' and 'area' to compute "
            "the apparent sound reduction index R'."
        )

    # Sabine equivalent absorption area A = 0,16 V / T (Clause 3.17).
    absorption = 0.16 * volume / t if volume is not None else None

    d_2m_n: np.ndarray | None = None
    if absorption is not None:
        d_2m_n = d_2m - 10.0 * np.log10(absorption / _A0_FACADE)

    r_prime: np.ndarray | None = None
    if surface_level is not None and area is not None and absorption is not None:
        surf_bands = _as_band_levels(surface_level, "surface_level")
        if surf_bands.shape != l2_bands.shape:
            raise ValueError(
                "'surface_level' must share the band count of 'l2'."
            )
        r_prime = (
            surf_bands
            - l2_bands
            + 10.0 * np.log10(area / absorption)
            - _FACADE_CORRECTION[method]
        )

    freqs = (
        np.asarray(frequencies, dtype=np.float64)
        if frequencies is not None
        else None
    )
    if freqs is not None and freqs.shape != d_2m.shape:
        raise ValueError(
            "'frequencies' must have one value per band; got "
            f"{freqs.size} for {d_2m.size} bands."
        )
    return FacadeInsulationResult(
        d_2m=d_2m,
        d_2m_nt=d_2m_nt,
        d_2m_n=d_2m_n,
        r_prime=r_prime,
        frequencies=freqs,
        method=method,
    )


def _resolve_impact_band_set(
    n: int, bands: str | None
) -> tuple[tuple[int, ...], float, int, int, int]:
    """Select the impact reference curve, bound, indices for the band set.

    :return: ``(reference, max_unfavourable, index_500, octave_offset,
        ci_band_count)``.
    """
    if bands == "third-octave" or (bands is None and n == 16):
        if n != 16:
            raise ValueError(
                "One-third-octave impact rating needs 16 bands "
                f"(100-3150 Hz), got {n}."
            )
        return (
            _REF_IMPACT_THIRD_OCTAVE,
            _MAX_UNFAVOURABLE_THIRD,
            _INDEX_500_THIRD,
            0,
            _CI_THIRD_OCTAVE_BANDS,
        )
    if bands == "octave" or (bands is None and n == 5):
        if n != 5:
            raise ValueError(
                "Octave impact rating needs 5 bands (125-2000 Hz), "
                f"got {n}."
            )
        return (
            _REF_IMPACT_OCTAVE,
            _MAX_UNFAVOURABLE_OCTAVE,
            _INDEX_500_OCTAVE,
            _IMPACT_OCTAVE_OFFSET,
            5,
        )
    if bands is not None:
        raise ValueError("'bands' must be 'third-octave', 'octave' or None.")
    raise ValueError(
        "Expected 16 one-third-octave (100-3150 Hz) or 5 octave "
        f"(125-2000 Hz) values, got {n}."
    )


def _impact_ci(measured: np.ndarray, rating: int, n_bands: int) -> int:
    """Spectrum adaptation term ``CI`` (ISO 717-2 Clause A.2.1).

    ``CI = Ln,sum - 15 - Ln,w`` with the energetic sum ``Ln,sum = 10 lg
    Σ 10^(Li/10)`` over the CI range (one-third octave 100-2500 Hz, i.e.
    the first 15 bands; octave 125-2000 Hz), rounded to an integer
    (round half up), Formulae (A.1) to (A.3).
    """
    l_sum = energy_sum(measured[:n_bands])
    return math.floor(l_sum + 0.5) - 15 - rating


def weighted_impact_rating(
    values_by_band: Sequence[float] | np.ndarray,
    bands: str | None = None,
) -> ImpactRatingResult:
    """
    Single-number weighted impact rating and CI per ISO 717-2.

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
    on the negated curves: minimising ``Σ max(0, measured - (ref + k))``
    over ``k`` equals maximising ``Σ max(0, (-ref) + (-k) - (-measured))``,
    the airborne problem, so no separate search is duplicated.

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

    reference, limit, index_500, octave_offset, ci_bands = (
        _resolve_impact_band_set(int(data.size), bands)
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
    centers = _FREQ_THIRD_OCTAVE if data.size == 16 else _FREQ_OCTAVE
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
    67.0, 67.5, 68.0, 68.5, 69.0, 69.5, 70.0, 70.5,
    71.0, 71.5, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0,
)
_IMPACT_REFERENCE_FLOOR_RATING = 78  # Ln,r,0,w (Table 4 / Clause 5.2)
#: Spectrum adaptation term of the bare reference floor (ISO 717-2:2020
#: Clause A.2.2): ``CI,r,0 = −11 dB``.
_IMPACT_REFERENCE_FLOOR_CI = -11


def weighted_impact_improvement(
    delta_l: Sequence[float] | np.ndarray,
) -> int:
    """Weighted reduction of impact sound pressure level ``ΔLw`` (ISO 717-2:2020 §5).

    Relates a measured improvement spectrum ``ΔL`` to the heavyweight reference
    floor of Table 4: the reference level with the covering is
    ``Ln,r = Ln,r,0 − ΔL`` (Formula (1)) and the weighted improvement is
    ``ΔLw = Ln,r,0,w − Ln,r,w = 78 − Ln,r,w`` (Formula (2)), where ``Ln,r,w`` is
    the ISO 717-2 weighted rating of ``Ln,r`` from :func:`weighted_impact_rating`.

    :param delta_l: The reduction of impact sound pressure level ``ΔL`` per band,
        in dB; 16 one-third-octave values from 100 Hz to 3150 Hz (e.g. from a
        floor-covering measurement to ISO 10140-3 or ISO 16251-1).
    :return: The weighted reduction ``ΔLw``, in dB (rounded, per ISO 717-2).
    :raises ValueError: If ``delta_l`` is not 16 one-third-octave values, or is
        non-finite.
    """
    dl = np.asarray(delta_l, dtype=np.float64)
    if dl.shape != (16,):
        raise ValueError(
            "'delta_l' must give the 16 one-third-octave values 100-3150 Hz."
        )
    if not np.all(np.isfinite(dl)):
        raise ValueError("'delta_l' must contain only finite values.")
    ln_r = np.asarray(_IMPACT_REFERENCE_FLOOR, dtype=np.float64) - dl
    ln_r_w = weighted_impact_rating(ln_r).rating
    return _IMPACT_REFERENCE_FLOOR_RATING - ln_r_w


def impact_improvement_adaptation_term(
    delta_l: Sequence[float] | np.ndarray,
) -> int:
    """Spectrum adaptation term ``CI,Δ`` of a floor covering (ISO 717-2:2020 A.2.2).

    ``CI,Δ = CI,r,0 − CI,r`` (Formula (A.4)) with ``CI,r,0 = −11 dB`` (the
    bare Table 4 reference floor) and ``CI,r`` the ISO 717-2 spectrum
    adaptation term of the reference floor with the covering under test,
    ``Ln,r = Ln,r,0 − ΔL`` (Formula (1)). Together with
    :func:`weighted_impact_improvement` it yields the single-number reduction
    for a flat spectrum, ``ΔLlin = ΔLw + CI,Δ`` (Formula (A.5)). ISO 16251-1
    Clause 8 e) requires this term in the statement of results.

    :param delta_l: The reduction of impact sound pressure level ``ΔL`` per
        band, in dB; 16 one-third-octave values from 100 Hz to 3150 Hz.
    :return: The spectrum adaptation term ``CI,Δ``, in dB (integer).
    :raises ValueError: If ``delta_l`` is not 16 one-third-octave values, or
        is non-finite.
    """
    dl = np.asarray(delta_l, dtype=np.float64)
    if dl.shape != (16,):
        raise ValueError(
            "'delta_l' must give the 16 one-third-octave values 100-3150 Hz."
        )
    if not np.all(np.isfinite(dl)):
        raise ValueError("'delta_l' must contain only finite values.")
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
    """

    rating: float
    ci: float
    ci_50_2500: float | None
    core: ImpactRatingResult


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
    values_by_band: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the extended-range input; return (measured, freqs, core_idx)."""
    data = np.asarray(values_by_band, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(_VALUES_1D_MSG)
    if not np.all(np.isfinite(data)):
        raise ValueError(_VALUES_FINITE_MSG)
    if frequencies is None:
        if data.size != len(_FREQ_THIRD_OCTAVE):
            raise ValueError(
                "Without 'frequencies' the input must be the 16 core "
                "one-third-octave bands 100-3150 Hz; pass the band centre "
                "frequencies for an enlarged range."
            )
        freqs = np.asarray(_FREQ_THIRD_OCTAVE, dtype=np.float64)
    else:
        freqs = np.asarray(frequencies, dtype=np.float64)
        if freqs.shape != data.shape:
            raise ValueError(
                "'frequencies' must have one value per band of "
                "'values_by_band'."
            )
        if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0.0):
            raise ValueError("'frequencies' must contain positive values.")
    core_idx = _match_bands(freqs, _FREQ_THIRD_OCTAVE)
    if core_idx is None:
        raise ValueError(
            "The input must contain the 16 core one-third-octave bands "
            "100-3150 Hz (ISO 717 rates the single number on that range)."
        )
    return _round_half_up_tenths(data), freqs, core_idx


def weighted_rating_extended(
    values_by_band: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray | None = None,
    *,
    one_decimal: bool = False,
) -> ExtendedWeightedRatingResult:
    """Weighted rating with enlarged-range adaptation terms (ISO 717-1 Annex B).

    Computes the weighted rating from the core one-third-octave bands
    100-3150 Hz (Clause 4.4) and, for every enlarged frequency range covered
    by the input, the additional spectrum adaptation terms of Annex B
    (``C50-3150``, ``C50-5000``, ``C100-5000`` and the ``Ctr`` counterparts)
    with the Table B.1 spectra: ``Cj = XAj − Xw`` where ``XAj`` sums over the
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
        values_by_band, frequencies
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
        for name, spectrum in ((f"c_{suffix}", spectrum1), (f"ctr_{suffix}", spectrum2)):
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
    )


def weighted_impact_rating_extended(
    values_by_band: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray | None = None,
    *,
    one_decimal: bool = False,
) -> ExtendedImpactRatingResult:
    """Weighted impact rating with ``CI,50-2500`` (ISO 717-2:2020 A.2.1).

    Computes the weighted impact rating from the core one-third-octave bands
    100-3150 Hz (Clause 4.3) and, when the input covers 50-2500 Hz, the
    enlarged-range spectrum adaptation term ``CI,50-2500`` of the A.2.1 NOTE:
    the energetic sum runs over 50-2500 Hz instead of 100-2500 Hz in
    Formula (A.1), ``CI = Ln,sum − 15 − Ln,w``.

    With ``one_decimal=True`` the reference-curve shift runs in 0,1 dB steps
    and the sums keep one decimal place (Clauses 4.3.1/4.4; e.g. the
    reference floor yields ``Ln,r,0,w = 77,6 dB`` and ``CI,r,0 = −10,3 dB``
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
        values_by_band, frequencies
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
    )
