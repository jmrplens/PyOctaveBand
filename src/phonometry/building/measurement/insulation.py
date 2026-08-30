#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Field measurement of sound insulation: airborne (ISO 16283-1:2014), impact
(ISO 16283-2:2020) and facade (ISO 16283-3:2016).

**Field quantities (ISO 16283-1:2014).** From the energy-average sound
pressure levels in the source and receiving rooms this module forms the
level difference :math:`D = L_1 - L_2` (Clause 3.12, Formula (1)), the
standardized level difference :math:`D_\mathrm{nT} = D + 10 \log_{10}(T/T_0)` with the
reference reverberation time :math:`T_0 = 0.5` s (Clause 3.13,
Formula (2)), and the apparent sound reduction index
:math:`R' = D + 10 \log_{10}(S/A)` with the Sabine equivalent absorption area
:math:`A = 0.16 V / T` (Clause 3.14/3.15, Formula (4) and (5)). Source and
receiving levels may be supplied already averaged (one value per band) or
as several microphone positions, which are then energy-averaged with
:math:`10 \log_{10}\left( \frac{1}{n} \sum 10^{L_j/10} \right)` (Clause 7.8.1,
Formula (9)). The quantities are evaluated per one-third-octave band, the
caller having already applied any background-noise correction (Clause 9.2).

**Field impact quantities (ISO 16283-2:2020).** With the tapping machine as
the impact source this module forms, from the energy-average impact sound
pressure level ``Li`` in the receiving room, the standardized impact sound
pressure level :math:`L'_\mathrm{nT} = L_\mathrm{i} - 10 \log_{10}(T/T_0)` with
:math:`T_0 = 0.5` s (Clause
3.13, Formula (1)) and the normalized impact sound pressure level
:math:`L'_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)` with the Sabine absorption area
:math:`A = 0.16 V/T` (Clause 3.14, Formula (2))
and the reference area :math:`A_0 = 10` m² (Clause 3.15, Formula (3)).
Levels
may be supplied already averaged or as several microphone positions, then
energy-averaged (Clause 7.8.1, Formula (11)).

**Field façade quantities (ISO 16283-3:2016).** With an outdoor sound
source this module forms, from the level 2 m in front of the façade
``L1,2m`` and the receiving-room level ``L2``, the level difference
:math:`D_{2\mathrm{m}} = L_{1,2\mathrm{m}} - L_2` (Clause 3.14), its standardized form
:math:`D_{2\mathrm{m,nT}} = D_{2\mathrm{m}} + 10 \log_{10}(T/T_0)` with :math:`T_0 = 0.5` s
(Clause 3.15) and
normalized form :math:`D_{2\mathrm{m,n}} = D_{2\mathrm{m}} - 10 \log_{10}(A/A_0)` with the Sabine
absorption
area :math:`A = 0.16 V/T` (Clause 3.17) and reference :math:`A_0 = 10` m²
(Clause 3.16): the global loudspeaker / traffic quantities
``Dls,2m,*`` / ``Dtr,2m,*``. When a surface level ``L1,s`` (microphone on
the test element) with the element area ``S`` and volume are given it
forms the apparent sound reduction index
:math:`R'_{45^\circ} = L_{1,\mathrm{s}} - L_2 + 10 \log_{10}(S/A) - 1.5` for the
loudspeaker element method
(Clause 3.12) or :math:`R'_\mathrm{tr,s} = L_{1,\mathrm{s}} - L_2 + 10 \log_{10}(S/A) - 3` for
the
road-traffic element method (Clause 3.13). These quantities are defined by
unnumbered formulas inline in the Clause 3 terms; positions are
energy-averaged with the surface-level formula (Clause 9.5.1, Formula (7)).
The façade quantity is airborne, so its single-number rating uses the
**ISO 717-1 airborne** reference curve and method (Clause 2, Annex F) via
:func:`weighted_rating` unchanged.

**Frequency range.** The three parts require the same 16 core one-third-octave
bands, 100 Hz to 3150 Hz (Part 1 and Part 3 Clause 5, Part 2 Clause 5.1), and
make the low range 50 Hz to 80 Hz and the high range 4000 Hz to 5000 Hz
optional additions. The functions below evaluate whatever bands they are given
and impose no range of their own; it is
:mod:`~phonometry.building.measurement.ratings` that fixes a range, taking
either those 16 one-third-octave bands or the 5 octave bands from 125 Hz to
2000 Hz, and the ``report()`` fiche that needs exactly the 16.

When the optional low range **is** measured, the low-frequency procedure of
:mod:`phonometry.building.measurement.low_frequency` becomes mandatory for the
50 Hz, 63 Hz and 80 Hz bands in any room whose volume, to the nearest cubic
metre, is under 25 m³. Each function below takes it through a
:class:`~.low_frequency.LowFrequencyProcedure`: ``source_low_frequency`` and
``receiver_low_frequency`` for the airborne pair of rooms, ``low_frequency``
for the single receiving room of the other two. Computing those three bands
from the default procedure alone in a room that small is not the ISO 16283
quantity, so a function handed the volume and the band centres without a
procedure raises a :class:`~.low_frequency.LowFrequencyWarning` rather than
answering silently. The one exception is the road-traffic façade, which
ISO 16283-3 Clause 6 gives the default procedure and nothing else.

The single-number rating of any of these curves is the subject of
:mod:`phonometry.building.measurement.ratings`, which implements ISO 717-1
(airborne) and ISO 717-2 (impact); the ``report()`` methods below call it to
box the rating on the field report form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ..._internal.levels_math import energy_mean
from ..._internal.types import as_float_or_array
from ..._internal.validation import (
    check_engine,
    require_equal_counts,
    require_equal_shapes,
    require_ranks,
    require_same_length,
)
from .low_frequency import (
    LowFrequencyProcedure,
    _warn_when_the_procedure_is_required,
    apply_low_frequency_procedure,
)
from .ratings import (
    _FREQ_THIRD_OCTAVE,
    ImpactRatingResult,
    WeightedRatingResult,
    weighted_impact_rating,
    weighted_rating,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata
    from .low_frequency import LowFrequencyResult

# The ISO 717 rating machinery now lives in :mod:`.ratings`. Until every caller
# in the tree reads it from there, the names this module used to define stay
# importable from this path, unchanged; the ``__all__`` below re-exports them.
from .ratings import (
    _IMPACT_REFERENCE_FLOOR,
    _INDEX_500_THIRD,
    _REF_IMPACT_THIRD_OCTAVE,
    _REF_THIRD_OCTAVE,
    _SPECTRUM1_50_5000,
    _SPECTRUM2_50_5000,
    ExtendedImpactRatingResult,
    ExtendedWeightedRatingResult,
    impact_improvement_adaptation_term,
    weighted_impact_improvement,
    weighted_impact_rating_extended,
    weighted_rating_extended,
)

# The ISO 717 names above are listed here so they keep resolving through this
# module; everything else is defined below.
__all__ = [
    "_IMPACT_REFERENCE_FLOOR",
    "_INDEX_500_THIRD",
    "_REF_IMPACT_THIRD_OCTAVE",
    "_REF_THIRD_OCTAVE",
    "_SPECTRUM1_50_5000",
    "_SPECTRUM2_50_5000",
    "AirborneInsulationResult",
    "ExtendedImpactRatingResult",
    "ExtendedWeightedRatingResult",
    "FacadeInsulationResult",
    "ImpactInsulationResult",
    "ImpactRatingResult",
    "LowFrequencyProcedure",
    "WeightedRatingResult",
    "airborne_insulation",
    "energy_average_level",
    "facade_insulation",
    "impact_improvement_adaptation_term",
    "impact_insulation",
    "weighted_impact_improvement",
    "weighted_impact_rating",
    "weighted_impact_rating_extended",
    "weighted_rating",
    "weighted_rating_extended",
]

#: Reference absorption area A0 for the normalized impact level (Clause 3.15).
_A0_IMPACT = 10.0

# --- ISO 16283-3 façade sound insulation ---------------------------------

#: Reference absorption area A0 for D2m,n (Clause 3.16, dwellings).
_A0_FACADE = 10.0

#: The façade source ISO 16283-3 admits the low-frequency procedure for
#: (Clause 6 and the heading of Clause 7.3, "element or global loudspeaker
#: methods").
_FACADE_LOUDSPEAKER = "loudspeaker"

#: Angle-of-incidence corrections in the apparent sound reduction index:
#: -1,5 dB for the loudspeaker method at 45° (Clause 3.12) and -3 dB for
#: the road-traffic method with all-angle incidence (Clause 3.13).
_FACADE_CORRECTION = {_FACADE_LOUDSPEAKER: 1.5, "road_traffic": 3.0}


@dataclass(frozen=True)
class AirborneInsulationResult:
    r"""Per-band field airborne sound insulation (ISO 16283-1:2014).

    :ivar d: Level difference :math:`D = L_1 - L_2` per band, in dB
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
    :ivar t2: Receiving-room reverberation time per band, in seconds, after any
        63 Hz octave substitution. Defaults to ``None``.
    :ivar t0: Reference reverberation time ``T0`` used for ``DnT``, in
        seconds. Defaults to ``None``.
    :ivar source_low_frequency: What the ISO 16283-1 Clause 8 low-frequency
        procedure did to ``l1``, or ``None`` when the source room was not under
        25 m³ or carried no corner measurements.
    :ivar receiver_low_frequency: The same for ``l2`` and, through Clause 10.4,
        for ``t2``.
    """

    d: np.ndarray
    dnt: np.ndarray
    r_prime: np.ndarray | None
    l1: np.ndarray | None = None
    l2: np.ndarray | None = None
    t2: np.ndarray | None = None
    t0: float | None = None
    source_low_frequency: LowFrequencyResult | None = None
    receiver_low_frequency: LowFrequencyResult | None = None

    def __post_init__(self) -> None:
        """Reject a measurement whose per-band columns cover different bands.

        The verbose ISO 16283-1 table prints the measurement chain a band at a
        time, ``L1``, ``L2``, ``T`` and the reported quantity in one row, and
        the plot lays ``D``, ``DnT`` and ``R'`` over one band-index axis: this
        result carries no band centres, so the curves are drawn against
        ``Band 1`` onwards rather than against frequency. Both pair the columns
        by position, so a chain column longer than the rest is printed only as
        far as the frequency header, setting a source-room level beside a
        receiving-room level from a different band, and the sheet shows a
        derivation that never took place. The quantities are carried already
        formed rather than recomputed from the chain printed next to them, so
        nothing downstream compares the two.

        ``None`` stands for a quantity the measurement did not produce, so it
        is passed over rather than counted.

        Every column must also be finite. No measurement here has an
        undeterminable band: :func:`airborne_insulation` refuses non-finite
        levels and reverberation times on the way in, so a NaN can only be
        smuggled into a hand-built result, and without this pin it would
        render as a literal ``nan`` cell inside the accredited chain table
        while the rating box beside it printed normally.

        :raises ValueError: if any per-band column disagrees with the rest,
            or carries a non-finite value.
        """
        require_ranks(self, d=1, dnt=1, r_prime=1, l1=1, l2=1, t2=1)
        require_same_length(self, "d", "dnt", "r_prime", "l1", "l2", "t2")
        for name in ("d", "dnt", "r_prime", "l1", "l2", "t2"):
            value = getattr(self, name)
            if value is not None and not np.all(
                np.isfinite(np.asarray(value, dtype=np.float64))
            ):
                msg = (
                    f"AirborneInsulationResult: '{name}' must contain only "
                    "finite values."
                )
                raise ValueError(msg)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band insulation quantities (``DnT``, ``D``, ``R'``).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_airborne_insulation

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
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        return _render_iso16283(
            self,
            path,
            quantity=quantity,
            metadata=metadata,
            engine=engine,
            verbose=verbose,
            language=language,
        )


@dataclass(frozen=True)
class ImpactInsulationResult:
    r"""Per-band field impact sound insulation (ISO 16283-2).

    :ivar l_n_t: Standardized impact sound pressure level
        :math:`L'_\mathrm{nT} = L_\mathrm{i} - 10 \log_{10}(T/T_0)` per band, in dB (Clause 3.13,
        Formula (1)).
    :ivar l_n: Normalized impact sound pressure level
        :math:`L'_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)` per band, in dB (Clause 3.14,
        Formula (2)), or ``None`` when the receiving-room volume was not
        supplied.
    :ivar li: Energy-average impact sound pressure levels the quantities
        were formed from, in dB (after any position averaging,
        Formula (11)). Defaults to ``None`` for backward-compatible
        construction.
    :ivar t2: Receiving-room reverberation time per band, in seconds, after any
        63 Hz octave substitution. Defaults to ``None``.
    :ivar t0: Reference reverberation time ``T0`` used for ``L'nT``, in
        seconds. Defaults to ``None``.
    :ivar low_frequency: What the ISO 16283-2 Clause 8 low-frequency procedure
        did to ``li`` and, through Clause 10.4, to ``t2``; ``None`` when the
        receiving room was not under 25 m³ or carried no corner measurements.
    """

    l_n_t: np.ndarray
    l_n: np.ndarray | None
    li: np.ndarray | None = None
    t2: np.ndarray | None = None
    t0: float | None = None
    low_frequency: LowFrequencyResult | None = None

    def __post_init__(self) -> None:
        """Reject a measurement whose per-band columns cover different bands.

        The verbose ISO 16283-2 table prints ``Li``, ``T`` and the reported
        level in one row per band, and the plot lays ``L'nT`` beside ``L'n``
        over one band-index axis: this result carries no band centres either,
        so the curves are drawn against ``Band 1`` onwards rather than against
        frequency. Both pair the columns by position, so a chain column longer
        than the rest is printed only as far as the frequency header and shows
        an impact level next to the reverberation time of a different band. The
        levels are carried already formed, not recomputed from the chain beside
        them, so nothing downstream compares the two.

        ``None`` stands for a quantity the measurement did not produce, so it
        is passed over rather than counted.

        Every column must also be finite. No measurement here has an
        undeterminable band: :func:`impact_insulation` refuses non-finite
        levels and reverberation times on the way in, so a NaN can only be
        smuggled into a hand-built result, and without this pin it would
        render as a literal ``nan`` cell inside the accredited chain table
        while the rating box beside it printed normally.

        :raises ValueError: if any per-band column disagrees with the rest,
            or carries a non-finite value.
        """
        require_ranks(self, l_n_t=1, l_n=1, li=1, t2=1)
        require_same_length(self, "l_n_t", "l_n", "li", "t2")
        for name in ("l_n_t", "l_n", "li", "t2"):
            value = getattr(self, name)
            if value is not None and not np.all(
                np.isfinite(np.asarray(value, dtype=np.float64))
            ):
                msg = (
                    f"ImpactInsulationResult: '{name}' must contain only finite values."
                )
                raise ValueError(msg)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band impact levels (``L'nT`` and, if present, ``L'n``).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_impact_insulation

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
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        return _render_iso16283(
            self,
            path,
            quantity=quantity,
            metadata=metadata,
            engine=engine,
            verbose=verbose,
            language=language,
        )


@dataclass(frozen=True)
class FacadeInsulationResult:
    r"""Per-band field façade sound insulation (ISO 16283-3).

    :ivar d_2m: Level difference :math:`D_{2\mathrm{m}} = L_{1,2\mathrm{m}} - L_2` per band,
        in dB (Clause 3.14; ``Dls,2m`` loudspeaker, ``Dtr,2m`` traffic).
    :ivar d_2m_nt: Standardized level difference
        :math:`D_{2\mathrm{m,nT}} = D_{2\mathrm{m}} + 10 \log_{10}(T/T_0)` per band, in dB
        (Clause 3.15).
    :ivar d_2m_n: Normalized level difference
        :math:`D_{2\mathrm{m,n}} = D_{2\mathrm{m}} - 10 \log_{10}(A/A_0)` per band, in dB
        (Clause 3.16), or
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
    :ivar low_frequency: What the ISO 16283-3 Clause 7.3 low-frequency
        procedure did to ``L2`` and, through Clause 8.4, to the reverberation
        time; ``None`` when the receiving room was not under 25 m³ or carried
        no corner measurements.
    """

    d_2m: np.ndarray
    d_2m_nt: np.ndarray
    d_2m_n: np.ndarray | None
    r_prime: np.ndarray | None
    frequencies: np.ndarray | None = None
    method: str = "loudspeaker"
    low_frequency: LowFrequencyResult | None = None

    def __post_init__(self) -> None:
        """Reject an unknown ``method``, or columns that cover different bands.

        An unknown source would be rendered and rated as the loudspeaker
        quantity ``R'45``, so it is refused first.

        The band check is the one the readers cannot make. The ISO 16283-3
        fiche reports one quantity at a time and counts that curve alone, the
        16 core one-third-octave bands the ISO 717-1 rating needs; the columns
        beside it are never looked at, so a result carrying seventeen values of
        ``D2m`` next to sixteen of ``D2m,nT`` writes a clean sheet that says
        nothing about the disagreement. The plot is what does pair them, every
        available quantity against ``frequencies``, and all it can answer with
        is matplotlib's shape message, which names neither the field nor the
        result.

        ``None`` stands for a quantity the measurement did not produce, so it
        is passed over rather than counted.

        :raises ValueError: if ``method`` is unknown, or a per-band column
            disagrees with the rest.
        """
        if self.method not in _FACADE_CORRECTION:
            msg = (
                "'method' must be 'loudspeaker' or 'road_traffic', got "
                f"{self.method!r}."
            )
            raise ValueError(msg)
        require_ranks(self, d_2m=1, d_2m_nt=1, d_2m_n=1, r_prime=1, frequencies=1)
        require_same_length(self, "d_2m", "d_2m_nt", "d_2m_n", "r_prime", "frequencies")

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band façade insulation profile (ISO 16283-3).

        Draws the standardized level difference and any other available
        quantities (``D2m``, ``D2m,n``, ``R'``) against frequency. Requires
        matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_facade_insulation

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
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        return _render_iso16283_facade(
            self,
            path,
            quantity=quantity,
            metadata=metadata,
            engine=engine,
            verbose=verbose,
            language=language,
        )


#: The field quantities each result kind can report, mapping the ``quantity``
#: argument of ``report()`` to the attribute holding the per-band curve and
#: the per-band chain columns ``verbose=True`` needs.
_FIELD_QUANTITIES: dict[type, tuple[tuple[str, ...], tuple[str, ...]]] = {
    AirborneInsulationResult: (("dnt", "r_prime"), ("l1", "l2", "t2")),
    ImpactInsulationResult: (("l_n_t", "l_n"), ("li", "t2")),
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
    3150 Hz, ISO 16283-1 Clause 5 and ISO 16283-2 Clause 5.1) so the
    ISO 717 rating is defined,
    evaluates that rating, and calls the reportlab renderer (which raises a
    clear :class:`ImportError` when reportlab is absent).
    """
    from ..._i18n import check_language

    check_language(language)
    check_engine(engine)
    quantities, chain_attrs = _FIELD_QUANTITIES[type(result)]
    if quantity not in quantities:
        expected = " or ".join(repr(q) for q in quantities)
        msg = f"Unknown field quantity {quantity!r}; expected {expected}."
        raise ValueError(msg)
    curve = getattr(result, quantity)
    if curve is None:
        msg = (
            f"The result carries no {quantity!r} curve; build it with the "
            "'area'/'volume' arguments so the normalized quantity is "
            "computed."
        )
        raise ValueError(msg)
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (len(_FREQ_THIRD_OCTAVE),):
        msg = (
            "The field report rates the 16 core one-third-octave bands "
            "(100 Hz to 3150 Hz, ISO 16283-1 Clause 5 and ISO 16283-2 "
            f"Clause 5.1); got {curve.shape} band values."
        )
        raise ValueError(msg)
    if verbose:
        missing = [a for a in chain_attrs if getattr(result, a) is None]
        if missing:
            msg = (
                "verbose=True needs the per-band measurement chain "
                f"({', '.join(chain_attrs)}); {', '.join(missing)} "
                "missing. Build the result with airborne_insulation() / "
                "impact_insulation() so they are populated."
            )
            raise ValueError(msg)
    if isinstance(result, ImpactInsulationResult):
        rating: WeightedRatingResult | ImpactRatingResult = weighted_impact_rating(
            curve
        )
    else:
        rating = weighted_rating(curve)

    from ..._report.iso16283 import render_iso16283_report

    return render_iso16283_report(
        result,
        rating,
        path,
        quantity=quantity,
        metadata=metadata,
        verbose=verbose,
        language=language,
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
    from ..._i18n import check_language

    check_language(language)
    check_engine(engine)
    if quantity not in _FACADE_FIELD_QUANTITIES:
        expected = " or ".join(repr(q) for q in _FACADE_FIELD_QUANTITIES)
        msg = f"Unknown facade quantity {quantity!r}; expected {expected}."
        raise ValueError(msg)
    curve = getattr(result, quantity)
    if curve is None:
        msg = (
            f"The result carries no {quantity!r} curve; build it with the "
            "'volume' (and, for R', 'surface_level'/'area') arguments so the "
            "quantity is computed."
        )
        raise ValueError(msg)
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (len(_FREQ_THIRD_OCTAVE),):
        msg = (
            "The facade field report rates the 16 core one-third-octave bands "
            f"(100 Hz to 3150 Hz, ISO 16283-3 Clause 5); got {curve.shape} "
            "band values."
        )
        raise ValueError(msg)
    rating = weighted_rating(curve)

    from ..._report.iso16283 import render_iso16283_facade_report

    return render_iso16283_facade_report(
        result,
        rating,
        path,
        quantity=quantity,
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


def energy_average_level(
    levels: Sequence[float] | np.ndarray, axis: int = -1
) -> np.ndarray | float:
    r"""Energy-average sound pressure level (ISO 16283-1:2014, Formula (9)).

    Combines sound pressure levels measured at several microphone
    positions into
    :math:`L = 10 \log_{10}\left( \frac{1}{n} \sum_j 10^{L_j/10} \right)`.

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
        msg = "'levels' must not be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(data)):
        msg = "'levels' must contain only finite values."
        raise ValueError(msg)
    return as_float_or_array(energy_mean(data, axis=axis))


def _as_band_levels(levels: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Coerce room levels to per-band values, energy-averaging positions.

    A one-dimensional input is taken as one already-averaged level per
    band; a two-dimensional input is read as ``(positions, bands)`` and
    energy-averaged over the positions (Formula (9)).
    """
    data = np.asarray(levels, dtype=np.float64)
    if data.ndim == 1:
        out = data
    elif data.ndim == 2:  # noqa: PLR2004
        out = np.asarray(energy_average_level(data, axis=0), dtype=np.float64)
    else:
        msg = f"'{name}' must be 1-D or 2-D (positions x bands)."
        raise ValueError(msg)
    if out.size == 0:
        msg = f"'{name}' must not be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(out)):
        msg = f"'{name}' must contain only finite values."
        raise ValueError(msg)
    return out


def _validate_reverberation(t: np.ndarray, t0: float) -> None:
    """Check the receiving-room reverberation time and its reference.

    ``t2`` must hold one positive, finite value per band and the reference
    time ``T0`` must be positive; the same requirement applies to every
    ISO 16283 part.
    """
    if t.ndim != 1:
        msg = "'t2' must be one-dimensional (one value per band)."
        raise ValueError(msg)
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        msg = "'t2' must contain positive, finite values."
        raise ValueError(msg)
    if t0 <= 0.0:
        msg = "'t0' must be positive."
        raise ValueError(msg)


def _require_frequencies(
    frequencies: Sequence[float] | np.ndarray | None, owner: str
) -> np.ndarray:
    """Demand the band centres the low-frequency procedure needs to place itself.

    The measurement functions are otherwise band-agnostic: they map whatever
    band vector they are handed. The low-frequency procedure is not, because it
    rewrites three named bands and leaves the rest alone, so it has to be told
    which columns those are.

    :param frequencies: Band centre frequencies, in Hz, or ``None``.
    :param owner: Function name, for the message.
    :return: The frequencies as a float array.
    :raises ValueError: If ``frequencies`` is ``None``.
    """
    if frequencies is None:
        msg = (
            f"{owner}: the ISO 16283 low-frequency procedure rewrites the "
            "50 Hz, 63 Hz and 80 Hz bands only, so 'frequencies' must be given "
            "with the low-frequency arguments to say which bands those are."
        )
        raise ValueError(msg)
    return np.asarray(frequencies, dtype=np.float64)


def _check_low_frequency_volume(
    procedure: LowFrequencyProcedure, volume: float | None, owner: str
) -> None:
    """Reject a receiving-room volume given twice with two different values.

    ``volume`` sizes the Sabine absorption area and the procedure's own volume
    decides the 25 m³ trigger; they are the same measured room, so a
    disagreement is a caller mistake that would otherwise normalise one
    quantity against a room the trigger never saw.

    :param procedure: The receiving-room low-frequency procedure.
    :param volume: The receiving-room volume passed alongside it, or ``None``.
    :param owner: Function name, for the message.
    :raises ValueError: If both are given and they differ.
    """
    if volume is None:
        return
    if not math.isclose(float(volume), float(procedure.volume), rel_tol=1e-12):
        msg = (
            f"{owner}: 'volume' is {float(volume):g} m³ and the receiving-room "
            f"low-frequency procedure says {float(procedure.volume):g} m³; "
            "they describe the same room and must agree."
        )
        raise ValueError(msg)


@overload
def airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    area: float,
    volume: float,
    t0: float = ...,
    frequencies: Sequence[float] | np.ndarray | None = ...,
    source_low_frequency: LowFrequencyProcedure | None = ...,
    receiver_low_frequency: LowFrequencyProcedure | None = ...,
) -> AirborneInsulationResult: ...


@overload
def airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    t0: float = ...,
    frequencies: Sequence[float] | np.ndarray | None = ...,
    source_low_frequency: LowFrequencyProcedure | None = ...,
    receiver_low_frequency: LowFrequencyProcedure | None = ...,
) -> AirborneInsulationResult: ...


def airborne_insulation(
    l1: Sequence[float] | np.ndarray,
    l2: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    area: float | None = None,
    volume: float | None = None,
    t0: float = 0.5,
    frequencies: Sequence[float] | np.ndarray | None = None,
    source_low_frequency: LowFrequencyProcedure | None = None,
    receiver_low_frequency: LowFrequencyProcedure | None = None,
) -> AirborneInsulationResult:
    r"""Field airborne sound insulation per ISO 16283-1:2014.

    Computes, per frequency band, the level difference
    :math:`D = L_1 - L_2`
    (Formula (1)), the standardized level difference
    :math:`D_\mathrm{nT} = D + 10 \log_{10}(T/T_0)` (Formula (2)) and, when the partition
    area
    and receiving-room volume are given, the apparent sound reduction
    index :math:`R' = D + 10 \log_{10}(S/A)` with :math:`A = 0.16\,V/T`
    (Formula (4) and (5)).

    ``l1`` and ``l2`` may be one value per band (already energy-averaged)
    or a two-dimensional ``(positions, bands)`` array, in which case the
    positions are energy-averaged with Formula (9). The band levels are
    assumed already corrected for background noise (Clause 9.2).

    **Rooms under 25 m³.** Clause 8.1 makes the low-frequency procedure
    mandatory for the 50 Hz, 63 Hz and 80 Hz bands "in the source and/or
    receiving room when *its* volume is smaller than 25 m³ (calculated to the
    nearest cubic metre)", so the two rooms are tested independently and either
    may carry a :class:`~.low_frequency.LowFrequencyProcedure`. Whichever room
    carries one has its level at those three bands replaced by
    :math:`L_\mathrm{LF}` (Formula (13)) before ``D`` is formed. Clause 10.4 is
    a receiving-room clause alone, so the 63 Hz octave reverberation time is
    taken from ``receiver_low_frequency`` and required there: a small source
    room beside a large receiving room changes ``L1`` and leaves ``t2`` as
    measured. Note 4 to entry of Clause 3.14 records that ``R'`` determined
    this way no longer maps exactly onto the sound-power ratio of Formula (3).

    Clause 8.1 is a *shall*, so a receiving room that rounds below 25 m³ and a
    ``frequencies`` vector naming the three bands are together enough to know
    the procedure was owed. When ``receiver_low_frequency`` is then absent this
    function computes ``D`` from the default procedure alone and raises a
    :class:`~.low_frequency.LowFrequencyWarning` saying that those three bands
    are not the ISO 16283 quantity. Only the receiving room, because only its
    volume is an argument here.

    :param l1: Source-room sound pressure levels, in dB.
    :param l2: Receiving-room sound pressure levels, in dB.
    :param t2: Receiving-room reverberation time per band, in seconds.
    :param area: Area ``S`` of the common partition, in m² (optional;
        required together with ``volume`` for ``R'``).
    :param volume: Receiving-room volume ``V``, in m³ (optional; required
        together with ``area`` for ``R'``).
    :param t0: Reference reverberation time ``T0``, in seconds (default
        0,5 s for dwellings, Clause 3.13).
    :param frequencies: Band centre frequencies, in Hz; required with either
        low-frequency argument, ignored otherwise (the result carries no band
        axis of its own).
    :param source_low_frequency: Source-room corner measurements and volume
        (Clause 8). Must not carry a 63 Hz octave reverberation time.
    :param receiver_low_frequency: Receiving-room corner measurements and
        volume (Clause 8), which must also carry the 63 Hz octave reverberation
        time of Clause 10.4.
    :return: :class:`AirborneInsulationResult` with ``d``, ``dnt`` and
        ``r_prime`` (the latter ``None`` unless ``area`` and ``volume``
        are both given), plus whichever low-frequency records were produced.
    :raises ValueError: If the band counts of ``l1``, ``l2`` and ``t2``
        differ, if only one of ``area``/``volume`` is supplied, if
        ``t2``/``t0`` are not positive, if inputs are non-finite, if a
        low-frequency argument is given without ``frequencies``, if ``volume``
        disagrees with the receiving-room procedure, or if the 63 Hz octave
        reverberation time is missing from the receiving room or present on the
        source room. A room that does not round below 25 m³ is refused where
        the corner measurements are described, by
        :class:`~.low_frequency.LowFrequencyProcedure` itself.
    :warns LowFrequencyWarning: when the receiving room rounds below 25 m³ and
        ``frequencies`` names the three bands but no ``receiver_low_frequency``
        answers Clause 8.1.
    """
    l1_bands = _as_band_levels(l1, "l1")
    l2_bands = _as_band_levels(l2, "l2")
    t = np.asarray(t2, dtype=np.float64)

    # Compared by count, not by shape, so a `t2` that carries an extra axis
    # falls through to `_validate_reverberation`, which says that is what is
    # wrong. Comparing shapes answered a (1, n) array against n bands with a
    # band-count complaint, which was false: the counts matched.
    require_equal_counts(
        "airborne_insulation",
        {"l1": l1_bands.size, "l2": l2_bands.size, "t2": t.size},
        "band",
    )
    _validate_reverberation(t, t0)

    source_lf: LowFrequencyResult | None = None
    receiver_lf: LowFrequencyResult | None = None
    if source_low_frequency is not None or receiver_low_frequency is not None:
        freqs = _require_frequencies(frequencies, "airborne_insulation")
        if source_low_frequency is not None:
            source_lf = apply_low_frequency_procedure(
                l1_bands, freqs, source_low_frequency, room="source"
            )
            l1_bands = source_lf.level
        if receiver_low_frequency is not None:
            _check_low_frequency_volume(
                receiver_low_frequency, volume, "airborne_insulation"
            )
            receiver_lf = apply_low_frequency_procedure(
                l2_bands,
                freqs,
                receiver_low_frequency,
                reverberation_time=t,
                room="receiving",
            )
            l2_bands = receiver_lf.level
            # `reverberation_time=` was passed, so the substituted vector is
            # never None on this branch; the test narrows the type rather than
            # guarding against a case the call above has already excluded.
            substituted = receiver_lf.reverberation_time
            if substituted is not None:
                t = substituted

    d = l1_bands - l2_bands
    dnt = d + 10.0 * np.log10(t / t0)

    if (area is None) != (volume is None):
        msg = "'area' and 'volume' must be given together to compute R'."
        raise ValueError(msg)
    r_prime: np.ndarray | None = None
    if area is not None and volume is not None:
        if area <= 0.0 or volume <= 0.0:
            msg = "'area' and 'volume' must be positive."
            raise ValueError(msg)
        absorption = 0.16 * volume / t
        r_prime = d + 10.0 * np.log10(area / absorption)

    # Only the receiving room, because only its volume is an argument here:
    # Clause 8.1 tests the two rooms on their own volumes and a source room
    # small enough to trigger is something the caller has to declare.
    _warn_when_the_procedure_is_required(
        volume,
        frequencies,
        receiver_low_frequency,
        owner="airborne_insulation",
        argument="receiver_low_frequency",
    )

    return AirborneInsulationResult(
        d=d,
        dnt=dnt,
        r_prime=r_prime,
        l1=l1_bands,
        l2=l2_bands,
        t2=t,
        t0=t0,
        source_low_frequency=source_lf,
        receiver_low_frequency=receiver_lf,
    )


# --- ISO 16283-2 field impact sound insulation ---------------------------


def impact_insulation(
    li: Sequence[float] | np.ndarray,
    t2: Sequence[float] | np.ndarray,
    *,
    volume: float | None = None,
    t0: float = 0.5,
    frequencies: Sequence[float] | np.ndarray | None = None,
    low_frequency: LowFrequencyProcedure | None = None,
) -> ImpactInsulationResult:
    r"""Field impact sound insulation per ISO 16283-2:2020 (tapping machine).

    Computes, per frequency band, the standardized impact sound pressure
    level :math:`L'_\mathrm{nT} = L_\mathrm{i} - 10 \log_{10}(T/T_0)` (Formula (1)) and, when the
    receiving-room volume is given, the normalized impact sound pressure
    level :math:`L'_\mathrm{n} = L_\mathrm{i} + 10 \log_{10}(A/A_0)` (Formula (3)) with the
    Sabine equivalent absorption
    area :math:`A = 0.16\,V/T` (Formula (2)) and the reference absorption
    area
    :math:`A_0 = 10` m².

    ``li`` may be one value per band (already energy-averaged) or a
    two-dimensional ``(positions, bands)`` array, in which case the
    positions are energy-averaged with Formula (11). The band levels are
    assumed already corrected for background noise (Clause 9).

    **Rooms under 25 m³.** Clause 8.1 makes the low-frequency procedure
    mandatory for the 50 Hz, 63 Hz and 80 Hz bands in the receiving room once
    its volume, to the nearest cubic metre, is under 25 m³, and Clause 10.4
    puts one 63 Hz octave reverberation time in place of the three
    one-third-octave ones. Pass both through ``low_frequency``. Part 2 confines
    the corner procedure to the tapping machine (the heading of its Clause 8),
    which is the only impact source this function models; the rubber ball is
    :mod:`phonometry.building.measurement.heavy_impact`.

    Clause 8.1 is a *shall*, so a ``volume`` that rounds below 25 m³ beside a
    ``frequencies`` vector naming the three bands is enough to know the
    procedure was owed. With no ``low_frequency`` to run it, this function
    answers from the default procedure alone and raises a
    :class:`~.low_frequency.LowFrequencyWarning` saying that those three bands
    are not the ISO 16283 quantity.

    :param li: Energy-average impact sound pressure levels, in dB.
    :param t2: Receiving-room reverberation time per band, in seconds.
    :param volume: Receiving-room volume ``V``, in m³ (optional; required
        for ``L'n``).
    :param t0: Reference reverberation time ``T0``, in seconds (default
        0,5 s for dwellings, Clause 3.13).
    :param frequencies: Band centre frequencies, in Hz; required with
        ``low_frequency``, ignored otherwise (the result carries no band axis
        of its own).
    :param low_frequency: Receiving-room corner measurements, volume and 63 Hz
        octave reverberation time (Clause 8 and Clause 10.4).
    :return: :class:`ImpactInsulationResult` with ``l_n_t`` and ``l_n``
        (the latter ``None`` unless ``volume`` is given), plus the
        low-frequency record when one was produced.
    :raises ValueError: If the band counts of ``li`` and ``t2`` differ, if
        ``t2``/``t0``/``volume`` are not positive, if inputs are
        non-finite, if ``low_frequency`` is given without ``frequencies``, if
        ``volume`` disagrees with the procedure, or if the 63 Hz octave
        reverberation time is missing. A room that does not round below 25 m³
        is refused where the corner measurements are described, by
        :class:`~.low_frequency.LowFrequencyProcedure` itself.
    :warns LowFrequencyWarning: when ``volume`` rounds below 25 m³ and
        ``frequencies`` names the three bands but no ``low_frequency`` answers
        Clause 8.1.
    """
    li_bands = _as_band_levels(li, "li")
    t = np.asarray(t2, dtype=np.float64)

    # Compared by count, not by shape, so a `t2` that carries an extra axis
    # falls through to `_validate_reverberation`, which says that is what is
    # wrong. Comparing shapes answered a (1, n) array against n bands with a
    # band-count complaint, which was false: the counts matched.
    require_equal_counts(
        "impact_insulation", {"li": li_bands.size, "t2": t.size}, "band"
    )
    _validate_reverberation(t, t0)

    low_frequency_result: LowFrequencyResult | None = None
    if low_frequency is not None:
        freqs = _require_frequencies(frequencies, "impact_insulation")
        _check_low_frequency_volume(low_frequency, volume, "impact_insulation")
        low_frequency_result = apply_low_frequency_procedure(
            li_bands, freqs, low_frequency, reverberation_time=t, room="receiving"
        )
        li_bands = low_frequency_result.level
        substituted = low_frequency_result.reverberation_time
        if substituted is not None:
            t = substituted

    l_n_t = li_bands - 10.0 * np.log10(t / t0)

    l_n: np.ndarray | None = None
    if volume is not None:
        if volume <= 0.0:
            msg = "'volume' must be positive."
            raise ValueError(msg)
        absorption = 0.16 * volume / t
        l_n = li_bands + 10.0 * np.log10(absorption / _A0_IMPACT)

    _warn_when_the_procedure_is_required(
        volume,
        frequencies,
        low_frequency,
        owner="impact_insulation",
        argument="low_frequency",
    )

    return ImpactInsulationResult(
        l_n_t=l_n_t,
        l_n=l_n,
        li=li_bands,
        t2=t,
        t0=t0,
        low_frequency=low_frequency_result,
    )


def _validate_facade_geometry(
    area: float | None,
    volume: float | None,
    surface_level: Sequence[float] | np.ndarray | None,
) -> None:
    """Check the optional façade geometry against the element method.

    ``area`` and ``volume`` must be positive when given, and the apparent
    sound reduction index needs the surface level, the element area and the
    receiving-room volume together.
    """
    if volume is not None and volume <= 0.0:
        msg = "'volume' must be positive."
        raise ValueError(msg)
    if area is not None and area <= 0.0:
        msg = "'area' must be positive."
        raise ValueError(msg)
    if area is not None and surface_level is None:
        msg = (
            "'area' requires 'surface_level' to compute the apparent sound "
            "reduction index R'."
        )
        raise ValueError(msg)
    if surface_level is not None and area is not None and volume is None:
        msg = (
            "'volume' is required with 'surface_level' and 'area' to compute "
            "the apparent sound reduction index R'."
        )
        raise ValueError(msg)


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
    low_frequency: LowFrequencyProcedure | None = None,
) -> FacadeInsulationResult:
    r"""Field façade sound insulation per ISO 16283-3:2016.

    Computes, per frequency band, the global-method level difference
    :math:`D_{2\mathrm{m}} = L_{1,2\mathrm{m}} - L_2` (Clause 3.14), its standardized form
    :math:`D_{2\mathrm{m,nT}} = D_{2\mathrm{m}} + 10 \log_{10}(T/T_0)` (Clause 3.15) and, when the
    receiving-room volume is given, its normalized form
    :math:`D_{2\mathrm{m,n}} = D_{2\mathrm{m}} - 10 \log_{10}(A/A_0)` with the Sabine equivalent
    absorption
    area :math:`A = 0.16\,V/T` (Clause 3.17) and :math:`A_0 = 10` m²
    (Clause 3.16).
    When a surface level ``L1,s`` (microphone on the test element),
    together with the element area ``S`` and the volume, is supplied it
    also computes the apparent sound reduction index of the element
    method: :math:`R'_{45^\circ} = L_{1,\mathrm{s}} - L_2 + 10 \log_{10}(S/A) - 1.5` for a
    loudspeaker
    source (Clause 3.12) or
    :math:`R'_\mathrm{tr,s} = L_{1,\mathrm{s}} - L_2 + 10 \log_{10}(S/A) - 3` for a
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

    **Rooms under 25 m³.** Clause 7.3.1 makes the low-frequency procedure
    mandatory for the 50 Hz, 63 Hz and 80 Hz bands in the receiving room once
    its volume, to the nearest cubic metre, is under 25 m³, and Clause 8.4 puts
    one 63 Hz octave reverberation time in place of the three one-third-octave
    ones. Pass both through ``low_frequency``. A loudspeaker-method call whose
    ``volume`` rounds below the line and whose ``frequencies`` names the three
    bands, with no ``low_frequency`` to run the procedure, answers from the
    default procedure alone and raises a
    :class:`~.low_frequency.LowFrequencyWarning` saying that those three bands
    are not the ISO 16283 quantity.

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
        on the result for plotting; required with ``low_frequency``.
    :param low_frequency: Receiving-room corner measurements, volume and 63 Hz
        octave reverberation time (Clause 7.3 and Clause 8.4). Loudspeaker
        methods only: Clause 6 says that for the element and global road
        traffic methods "only the default procedure shall be used", so a small
        room measured against traffic is conforming without corners and is
        neither refused nor warned about.
    :return: :class:`FacadeInsulationResult` with ``d_2m``, ``d_2m_nt``,
        ``d_2m_n`` (``None`` unless ``volume`` is given) and ``r_prime``
        (``None`` unless ``surface_level``, ``area`` and ``volume`` are all
        given), plus the low-frequency record when one was produced.
    :raises ValueError: If band counts differ, if ``method`` is unknown, if
        ``t2``/``t0``/``area``/``volume`` are not positive, if ``area`` is
        given without ``surface_level``, if ``surface_level`` and ``area`` are
        given without ``volume``, if ``frequencies`` is given with a shape
        that differs from the band axis, if inputs are non-finite, if
        ``low_frequency`` is given without ``frequencies`` or with
        ``method="road_traffic"``, if ``volume`` disagrees with the procedure,
        or if the 63 Hz octave reverberation time is missing. A room that does
        not round below 25 m³ is refused where the corner measurements are
        described, by :class:`~.low_frequency.LowFrequencyProcedure` itself.
        Supplying ``surface_level`` alone is not an error: ``r_prime`` simply
        stays ``None``.
    :warns LowFrequencyWarning: when a loudspeaker-method ``volume`` rounds
        below 25 m³ and ``frequencies`` names the three bands but no
        ``low_frequency`` answers Clause 7.3.1.
    """
    if method not in _FACADE_CORRECTION:
        msg = f"'method' must be 'loudspeaker' or 'road_traffic', got {method!r}."
        raise ValueError(msg)

    l1_bands = _as_band_levels(l1_2m, "l1_2m")
    l2_bands = _as_band_levels(l2, "l2")
    t = np.asarray(t2, dtype=np.float64)

    # Compared by count, not by shape, so a `t2` that carries an extra axis
    # falls through to `_validate_reverberation`, which says that is what is
    # wrong. Comparing shapes answered a (1, n) array against n bands with a
    # band-count complaint, which was false: the counts matched.
    require_equal_counts(
        "facade_insulation",
        {"l1_2m": l1_bands.size, "l2": l2_bands.size, "t2": t.size},
        "band",
    )
    _validate_reverberation(t, t0)

    low_frequency_result: LowFrequencyResult | None = None
    if low_frequency is not None:
        if method != _FACADE_LOUDSPEAKER:
            msg = (
                "ISO 16283-3 Clause 6 confines the low-frequency procedure to "
                "the element and global loudspeaker methods, its NOTE 1 "
                "recording that there is no experience of running it with "
                "traffic as the source; 'low_frequency' cannot be combined "
                f"with method={method!r}."
            )
            raise ValueError(msg)
        freqs_lf = _require_frequencies(frequencies, "facade_insulation")
        _check_low_frequency_volume(low_frequency, volume, "facade_insulation")
        low_frequency_result = apply_low_frequency_procedure(
            l2_bands, freqs_lf, low_frequency, reverberation_time=t, room="receiving"
        )
        l2_bands = low_frequency_result.level
        substituted = low_frequency_result.reverberation_time
        if substituted is not None:
            t = substituted

    d_2m = l1_bands - l2_bands
    d_2m_nt = d_2m + 10.0 * np.log10(t / t0)

    _validate_facade_geometry(area, volume, surface_level)

    # Sabine equivalent absorption area A = 0,16 V / T (Clause 3.17).
    absorption = 0.16 * volume / t if volume is not None else None

    d_2m_n: np.ndarray | None = None
    if absorption is not None:
        d_2m_n = d_2m - 10.0 * np.log10(absorption / _A0_FACADE)

    r_prime: np.ndarray | None = None
    if surface_level is not None and area is not None and absorption is not None:
        surf_bands = _as_band_levels(surface_level, "surface_level")
        require_equal_shapes(
            "facade_insulation",
            {"surface_level": surf_bands.shape, "l2": l2_bands.shape},
            "band",
        )
        r_prime = (
            surf_bands
            - l2_bands
            + 10.0 * np.log10(area / absorption)
            - _FACADE_CORRECTION[method]
        )

    freqs = (
        np.asarray(frequencies, dtype=np.float64) if frequencies is not None else None
    )
    if freqs is not None:
        require_equal_shapes(
            "facade_insulation",
            {"l1_2m": l1_bands.shape, "frequencies": freqs.shape},
            "band",
        )

    # Loudspeaker methods only. Clause 6 says "for the element and global road
    # traffic methods, only the default procedure shall be used", so a small
    # room measured against traffic is conforming without corners and there is
    # nothing to warn about.
    if method == _FACADE_LOUDSPEAKER:
        _warn_when_the_procedure_is_required(
            volume,
            frequencies,
            low_frequency,
            owner="facade_insulation",
            argument="low_frequency",
        )

    return FacadeInsulationResult(
        d_2m=d_2m,
        d_2m_nt=d_2m_nt,
        d_2m_n=d_2m_n,
        r_prime=r_prime,
        frequencies=freqs,
        method=method,
        low_frequency=low_frequency_result,
    )
