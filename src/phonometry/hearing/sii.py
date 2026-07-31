#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Speech Intelligibility Index (SII) per ANSI S3.5-1997 (R2017).

Implements all four band procedures of ANSI S3.5-1997, *American National
Standard Methods for the Calculation of the Speech Intelligibility Index*:

- ``method="critical-band"``: 21 critical bands, 100 Hz to 9500 Hz (Table 1).
- ``method="equally-contributing"``: 17 equally-contributing critical bands,
  300 Hz to 6400 Hz (Table 2).
- ``method="one-third-octave"``: 18 one-third-octave bands, 160 Hz to 8000 Hz
  (Table 3). The library default.
- ``method="octave"``: 6 octave bands, 177 Hz to 11314 Hz (Table 4).

From an equivalent speech spectrum level, an equivalent noise spectrum level
and an equivalent hearing threshold, every procedure runs the same chain: the
self-speech masking, the upward spread of masking, the equivalent internal
noise and disturbance, the level-distortion factor and the band-audibility
function, whose importance-weighted sum is the index ``SII`` in [0, 1]
(clause 6). Only the band table and the geometry of the spread of masking
change from procedure to procedure: the critical-band and
equally-contributing procedures spread the masking between tabulated band
limits, the one-third-octave procedure between band centre frequencies, and
the octave-band procedure omits the spread entirely (its bands are already
wider than the spread being modelled).

The band-importance functions, the standard speech spectrum levels by vocal
effort and the reference internal noise spectrum levels are the standard's own
tabulated constants (Tables 1 to 4). Spectrum levels are as defined in clauses
3.11 and 3.55.

The implementation reproduces the reference implementation ``SII.C`` of ASA
Working Group S3-79 (the committee that maintains ANSI S3.5) to double
precision on all eight of its official test cases (``CB.TST``, ``CB_1.TST``,
``ECB.TST``, ``ECB_1.TST``, ``TO.TST``, ``TO_1.TST``, ``OCTAVE.TST`` and
``OCTAVE_1.TST``, two per procedure, the ``_1`` variants exercising an
alternative band-importance function; ``CB_1.TST`` and ``ECB_1.TST`` are the
same confirmation twice, so the eight are seven independent ones), and
computes the Annex C worked examples with the working group's official errata
applied (see ``docs/ERRATA.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from numpy.typing import ArrayLike

from .._internal.warnings import _warn_renamed

#: The four band procedures of ANSI S3.5-1997, in the order of its Tables 1
#: to 4: the critical-band procedure (21 bands), the equally-contributing
#: critical-band procedure (17 bands), the one-third-octave-band procedure
#: (18 bands, the library default) and the octave-band procedure (6 bands).
SII_METHODS: tuple[str, ...] = (
    "critical-band",
    "equally-contributing",
    "one-third-octave",
    "octave",
)

# ---------------------------------------------------------------------------
# Normative constants - ANSI S3.5-1997, one-third-octave-band method (Table 3).
# ---------------------------------------------------------------------------

#: One-third-octave band centre frequencies, in hertz (18 bands, Table 3).
BAND_CENTERS: np.ndarray = np.array(
    [160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
     1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0],
    dtype=np.float64,
)

#: Band-importance function ``Ii`` (Table 3, average speech material); sums to 1.
BAND_IMPORTANCE: np.ndarray = np.array(
    [0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711, 0.0818,
     0.0844, 0.0882, 0.0898, 0.0868, 0.0844, 0.0771, 0.0527, 0.0364, 0.0185],
    dtype=np.float64,
)

#: Standard speech spectrum level ``Ui`` by vocal effort (Table 3), dB SPL.
_SPEECH_NORMAL: np.ndarray = np.array(
    [32.41, 34.48, 34.75, 33.98, 34.59, 34.27, 32.06, 28.30, 25.01, 23.00,
     20.15, 17.32, 13.18, 11.55, 9.33, 5.31, 2.59, 1.13],
    dtype=np.float64,
)
_SPEECH_RAISED: np.ndarray = np.array(
    [33.81, 33.92, 38.98, 38.57, 39.11, 40.15, 38.78, 36.37, 33.86, 31.89,
     28.58, 25.32, 22.35, 20.15, 16.78, 11.47, 7.67, 5.07],
    dtype=np.float64,
)
_SPEECH_LOUD: np.ndarray = np.array(
    [35.29, 37.76, 41.55, 43.78, 43.30, 44.85, 45.55, 44.05, 42.16, 40.53,
     37.70, 34.39, 30.98, 28.21, 25.41, 18.35, 13.87, 11.39],
    dtype=np.float64,
)
_SPEECH_SHOUT: np.ndarray = np.array(
    [30.77, 36.65, 42.50, 46.51, 47.40, 49.24, 51.21, 51.44, 51.31, 49.63,
     47.65, 44.32, 40.80, 38.13, 34.41, 28.24, 23.45, 20.72],
    dtype=np.float64,
)

#: Standard speech spectra by vocal effort (Table 3): normal, raised, loud and
#: shout.
_SPEECH_SPECTRA: dict[str, np.ndarray] = {
    "normal": _SPEECH_NORMAL,
    "raised": _SPEECH_RAISED,
    "loud": _SPEECH_LOUD,
    "shout": _SPEECH_SHOUT,
}

#: Reference internal noise spectrum level ``Xi`` (Table 3), dB SPL.
REFERENCE_INTERNAL_NOISE: np.ndarray = np.array(
    [0.6, -1.7, -3.9, -6.1, -8.2, -9.7, -10.8, -11.9, -12.5, -13.5, -15.4,
     -17.7, -21.2, -24.2, -25.9, -23.6, -15.8, -7.1],
    dtype=np.float64,
)

_N_BANDS = BAND_CENTERS.size
VOCAL_EFFORTS: tuple[str, ...] = ("normal", "raised", "loud", "shout")

# ---------------------------------------------------------------------------
# Normative constants - ANSI S3.5-1997, critical-band method (Table 1).
#
# The 21 critical bands are the classical Bark-scale bands: the tabulated band
# limits below run from 100 Hz to 9500 Hz and the tabulated centre frequencies
# are the nominal Bark centres. Unlike the one-third-octave table, Table 1
# prints the band limits, and the procedure's spread of masking is expressed
# in terms of them (see ``_masking_geometry``).
# ---------------------------------------------------------------------------

#: Critical-band limits, in hertz (22 edges bounding the 21 bands, Table 1).
_CRITICAL_EDGES: np.ndarray = np.array(
    [100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0,
     1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0,
     6400.0, 7700.0, 9500.0],
    dtype=np.float64,
)

#: Nominal critical-band centre frequencies, in hertz (Table 1).
_CRITICAL_CENTERS: np.ndarray = np.array(
    [150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0, 1370.0,
     1600.0, 1850.0, 2150.0, 2500.0, 2900.0, 3400.0, 4000.0, 4800.0, 5800.0,
     7000.0, 8500.0],
    dtype=np.float64,
)

#: Critical-band importance function ``Ii`` (Table 1); sums to 1.
_CRITICAL_IMPORTANCE: np.ndarray = np.array(
    [0.0103, 0.0261, 0.0419, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577,
     0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0460,
     0.0343, 0.0226, 0.0110],
    dtype=np.float64,
)

#: Critical-band standard speech spectrum level ``Ui``, normal vocal effort
#: (Table 1), dB SPL.
_CRITICAL_SPEECH_NORMAL: np.ndarray = np.array(
    [31.44, 34.75, 34.14, 34.58, 33.17, 30.64, 27.59, 25.01, 23.52, 22.28,
     20.15, 18.29, 16.37, 13.80, 12.21, 11.09, 9.33, 5.84, 3.47, 1.78, -0.14],
    dtype=np.float64,
)

#: Critical-band reference internal noise spectrum level ``Xi`` (Table 1),
#: dB SPL.
_CRITICAL_INTERNAL_NOISE: np.ndarray = np.array(
    [1.5, -3.9, -7.2, -8.9, -10.3, -11.4, -12.0, -12.5, -13.2, -14.0, -15.4,
     -16.9, -18.8, -21.2, -23.2, -24.9, -25.9, -24.2, -19.0, -11.7, -6.0],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Normative constants - ANSI S3.5-1997, equally-contributing critical-band
# method (Table 2).
#
# Table 2 is the 300 Hz - 6400 Hz span of Table 1, i.e. critical bands 3 to 19,
# with every band given the same importance so that each contributes equally.
# The band limits, speech spectrum and reference internal noise are therefore
# the Table 1 rows of that span, which is how the constants are derived here
# rather than transcribed twice.
# ---------------------------------------------------------------------------

_EQUAL_SPAN = slice(2, 19)

#: Number of equally-contributing critical bands (Table 2).
_N_EQUAL = 17

#: Equally-contributing band-importance function ``Ii`` (Table 2): the same
#: 0.0588 in all 17 bands. It sums to 0.9996 rather than to exactly one, which
#: is the printed constant rounded to four decimals (1/17 = 0.058823...).
_EQUAL_IMPORTANCE: np.ndarray = np.full(_N_EQUAL, 0.0588, dtype=np.float64)

# ---------------------------------------------------------------------------
# Normative constants - ANSI S3.5-1997, octave-band method (Table 4).
#
# Six octave bands with nominal centre frequencies 250 Hz to 8000 Hz. The
# tabulated speech spectrum levels and reference internal noise spectrum levels
# are the Table 3 rows at the same six centre frequencies: both are spectrum
# (per-hertz) levels, so they do not depend on the analysis bandwidth.
# ---------------------------------------------------------------------------

#: Octave-band limits, in hertz (7 edges bounding the 6 bands, Table 4).
_OCTAVE_EDGES: np.ndarray = np.array(
    [177.0, 354.0, 707.0, 1414.0, 2828.0, 5657.0, 11314.0], dtype=np.float64
)

#: Nominal octave-band centre frequencies, in hertz (Table 4).
_OCTAVE_CENTERS: np.ndarray = np.array(
    [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0], dtype=np.float64
)

#: Octave-band importance function ``Ii`` (Table 4); sums to 1.
_OCTAVE_IMPORTANCE: np.ndarray = np.array(
    [0.0617, 0.1671, 0.2373, 0.2648, 0.2142, 0.0549], dtype=np.float64
)

#: Octave-band standard speech spectrum level ``Ui``, normal vocal effort
#: (Table 4), dB SPL.
_OCTAVE_SPEECH_NORMAL: np.ndarray = np.array(
    [34.75, 34.27, 25.01, 17.32, 9.33, 1.13], dtype=np.float64
)

#: Octave-band reference internal noise spectrum level ``Xi`` (Table 4), dB SPL.
_OCTAVE_INTERNAL_NOISE: np.ndarray = np.array(
    [-3.9, -9.7, -12.5, -17.7, -25.9, -7.1], dtype=np.float64
)

# ---------------------------------------------------------------------------
# The band procedures, and the shared calculation chain.
# ---------------------------------------------------------------------------

#: Self-speech masking offset below the speech spectrum level, dB (clause 5.4).
_SELF_SPEECH_MASKING = 24.0

#: Speech level at which the level-distortion factor starts to bite: the
#: standard speech spectrum for normal vocal effort raised by this many dB
#: (clause 5.7).
_LEVEL_DISTORTION_OFFSET = 10.0

#: Denominator of the level-distortion factor, dB (clause 5.7).
_LEVEL_DISTORTION_RANGE = 160.0

#: Peak-to-rms allowance added to the speech spectrum level, dB (clause 5.8).
_SPEECH_PEAK = 15.0

#: Dynamic range of the band-audibility function, dB (clause 5.8).
_AUDIBILITY_RANGE = 30.0


@dataclass(frozen=True)
class _BandProcedure:
    """Internal band table and masking geometry of one ANSI S3.5 procedure.

    :ivar method: The public ``method=`` name.
    :ivar frequencies: Nominal band centre frequencies, in hertz.
    :ivar band_edges: Band limits, in hertz (one more than the band count).
    :ivar band_importance: Band-importance function ``Ii``.
    :ivar internal_noise: Reference internal noise spectrum level ``Xi``.
    :ivar speech_spectrum: Standard speech spectrum level ``Ui``, normal effort.
    :ivar bandwidth_db: The level-independent part of the masking slope ``Ci``
        (clause 5.4), or ``None`` when the procedure has no spread of masking.
        Together with :attr:`bandwidth_offset_db` it is ``10 log10(Wi)``, split
        exactly as the standard prints the slope for that procedure; see
        :func:`_equivalent_masking` for why the split is kept.
    :ivar bandwidth_offset_db: Constant subtracted from ``Bi + bandwidth_db``
        inside the slope, in decibels: the printed ``6.353`` of the
        one-third-octave form, and ``0.0`` for the procedures whose slope is
        printed directly in terms of the band width.
    :ivar spread_decades: ``log10(Fi / Fk_upper)``, the frequency separation
        that the masking slope is applied over, as a lower-triangular matrix
        indexed ``[i, k]``, or ``None`` when there is no spread of masking.
    """

    method: str
    frequencies: np.ndarray
    band_edges: np.ndarray
    band_importance: np.ndarray
    internal_noise: np.ndarray
    speech_spectrum: np.ndarray
    bandwidth_db: np.ndarray | None
    bandwidth_offset_db: float
    spread_decades: np.ndarray | None


def _masking_geometry(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masking-slope bandwidth term and band separation from band limits.

    For a procedure whose bands are given by their limits (the critical-band
    and equally-contributing critical-band procedures of Tables 1 and 2), the
    masking slope of clause 5.4 uses the band width ``Wi = fi_upper -
    fi_lower`` and the masking spreads from the upper limit of the masker band
    ``k`` up to the (geometric) centre frequency of the masked band ``i``.
    """
    lower, upper = edges[:-1], edges[1:]
    bandwidth_db = 10.0 * np.log10(upper - lower)
    centers = np.sqrt(lower * upper)
    return bandwidth_db, np.log10(centers[:, np.newaxis] / upper[np.newaxis, :])


def _third_octave_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Masking-slope bandwidth term and band separation of Table 3.

    The one-third-octave procedure states the same geometry in terms of the
    band centre frequencies instead of the band limits, with the two constants
    of clause 5.4 printed rounded: ``10 log10(Wi) = 10 log10(fi) - 6.353``
    (a one-third-octave band is ``0.2316 fi`` wide) and the separation
    ``0.89 fi / fk`` (the upper limit of band ``k`` is ``2**(1/6) fk``). The
    printed constants are used as printed, so the procedure reproduces the
    standard's own worked examples digit for digit.

    The ``6.353`` is deliberately *not* folded into the returned array. The
    standard prints this slope as ``-80 + 0.6 (Bi + 10 lg fi - 6.353)``, which
    evaluates as ``(Bi + 10 lg fi) - 6.353``; folding the constant in first
    would evaluate ``Bi + (10 lg fi - 6.353)`` instead, and floating-point
    addition is not associative. The two differ in the last bits, so the fold
    would silently shift a shipped, released quantity. It is returned as the
    procedure's ``bandwidth_offset_db`` and subtracted after the addition.
    """
    f = BAND_CENTERS
    return 10.0 * np.log10(f), np.log10(
        0.89 * f[:, np.newaxis] / f[np.newaxis, :]
    )


def _build_procedures() -> dict[str, _BandProcedure]:
    """Assemble the four band procedures of ANSI S3.5-1997 Tables 1 to 4."""
    critical_bandwidth_db, critical_spread = _masking_geometry(_CRITICAL_EDGES)
    third_bandwidth_db, third_spread = _third_octave_geometry()
    equal_bandwidth_db, equal_spread = _masking_geometry(
        _CRITICAL_EDGES[_EQUAL_SPAN.start : _EQUAL_SPAN.stop + 1]
    )
    return {
        "critical-band": _BandProcedure(
            method="critical-band",
            frequencies=_CRITICAL_CENTERS,
            band_edges=_CRITICAL_EDGES,
            band_importance=_CRITICAL_IMPORTANCE,
            internal_noise=_CRITICAL_INTERNAL_NOISE,
            speech_spectrum=_CRITICAL_SPEECH_NORMAL,
            bandwidth_db=critical_bandwidth_db,
            bandwidth_offset_db=0.0,
            spread_decades=critical_spread,
        ),
        "equally-contributing": _BandProcedure(
            method="equally-contributing",
            frequencies=_CRITICAL_CENTERS[_EQUAL_SPAN],
            band_edges=_CRITICAL_EDGES[_EQUAL_SPAN.start : _EQUAL_SPAN.stop + 1],
            band_importance=_EQUAL_IMPORTANCE,
            internal_noise=_CRITICAL_INTERNAL_NOISE[_EQUAL_SPAN],
            speech_spectrum=_CRITICAL_SPEECH_NORMAL[_EQUAL_SPAN],
            bandwidth_db=equal_bandwidth_db,
            bandwidth_offset_db=0.0,
            spread_decades=equal_spread,
        ),
        "one-third-octave": _BandProcedure(
            method="one-third-octave",
            frequencies=BAND_CENTERS,
            # Table 3 prints the centre frequencies; the exact one-third-octave
            # limits 2**(-+1/6) fi bound the same bands.
            band_edges=np.concatenate(
                (BAND_CENTERS * 2.0 ** (-1.0 / 6.0),
                 BAND_CENTERS[-1:] * 2.0 ** (1.0 / 6.0))
            ),
            band_importance=BAND_IMPORTANCE,
            internal_noise=REFERENCE_INTERNAL_NOISE,
            speech_spectrum=_SPEECH_NORMAL,
            bandwidth_db=third_bandwidth_db,
            bandwidth_offset_db=6.353,
            spread_decades=third_spread,
        ),
        "octave": _BandProcedure(
            method="octave",
            frequencies=_OCTAVE_CENTERS,
            band_edges=_OCTAVE_EDGES,
            band_importance=_OCTAVE_IMPORTANCE,
            internal_noise=_OCTAVE_INTERNAL_NOISE,
            speech_spectrum=_OCTAVE_SPEECH_NORMAL,
            # The octave-band procedure has no spread of masking: an octave
            # band is already wider than the upward spread being modelled, so
            # the equivalent masking spectrum level is the equivalent noise
            # spectrum level itself.
            bandwidth_db=None,
            bandwidth_offset_db=0.0,
            spread_decades=None,
        ),
    }


_PROCEDURES: dict[str, _BandProcedure] = _build_procedures()


def _procedure(method: str) -> _BandProcedure:
    """Return the band procedure named ``method``.

    :raises ValueError: for an unknown procedure name.
    """
    try:
        return _PROCEDURES[method]
    except KeyError:
        raise ValueError(
            f"Unknown SII method {method!r}; choose from "
            f"{', '.join(SII_METHODS)}."
        ) from None


@dataclass(frozen=True)
class SIIResult:
    """Result of a Speech Intelligibility Index computation (ANSI S3.5-1997).

    :ivar sii: The overall Speech Intelligibility Index in [0, 1] (clause 6).
    :ivar band_audibility: Per-band audibility function ``Ai`` (clause 5.8).
    :ivar band_importance: Per-band importance function ``Ii`` used (the
        procedure's own table, or the alternative function supplied).
    :ivar frequencies: Band centre frequencies of the procedure, in hertz.
    :ivar speech_spectrum: Equivalent speech spectrum level ``Ei'`` per band.
    :ivar disturbance: Equivalent disturbance spectrum level ``Di`` (clause 5.6).
    :ivar masking: Equivalent masking spectrum level ``Zi`` (clause 5.4).
    :ivar level_distortion: Per-band level-distortion factor ``Li`` in [0, 1]
        (clause 5.7), unity until the speech spectrum level rises above the
        standard normal-effort spectrum by more than 10 dB.
    :ivar method: The band procedure used, one of :data:`SII_METHODS`.
    """

    sii: float
    band_audibility: np.ndarray
    band_importance: np.ndarray
    frequencies: np.ndarray
    speech_spectrum: np.ndarray
    disturbance: np.ndarray
    masking: np.ndarray
    level_distortion: np.ndarray
    method: str = "one-third-octave"

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-band audibility weighted by importance, with the SII.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.hearing import plot_sii

        return plot_sii(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ANSI S3.5-1997 speech-intelligibility-index fiche to a PDF.

        Writes a one-page speech-audibility report: a standard-basis line
        naming the band procedure, an optional metadata header block, a per-band
        table over that procedure's bands (the equivalent speech spectrum
        ``Ei'``, the band-importance function ``Ii`` and the band-audibility
        function ``Ai``) beside the audibility and importance-weighted
        contribution bars (the result's own :meth:`plot`), the boxed
        ``SII = X`` single number, an optional verdict row and a footer with the
        fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a bare fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the minimum required SII (a higher SII
            passes).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the left table adds the equivalent
            disturbance spectrum level ``Di`` column (clause 5.6).
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or
            ``language`` is not a supported language.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.sii import render_sii_report

        return render_sii_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def standard_speech_spectrum(vocal_effort: str = "normal") -> np.ndarray:
    """Standard speech spectrum level by vocal effort (ANSI S3.5-1997 Table 3).

    :param vocal_effort: One of ``"normal"``, ``"raised"``, ``"loud"``,
        ``"shout"``.
    :return: The 18-band equivalent speech spectrum level ``Ui``, in dB SPL.
    :raises ValueError: for an unknown vocal effort.
    """
    try:
        return _SPEECH_SPECTRA[vocal_effort].copy()
    except KeyError:
        raise ValueError(
            f"Unknown vocal_effort {vocal_effort!r}; choose from "
            f"{', '.join(VOCAL_EFFORTS)}."
        ) from None


@dataclass(frozen=True)
class StandardSpeechSpectrum:
    """The ANSI S3.5-1997 standard speech spectra by vocal effort (Table 3).

    Bundles the standard speech spectrum level ``Ui`` of one or more vocal
    efforts (ANSI S3.5-1997 Table 3) over the 18 one-third-octave bands, so the
    spectra can be drawn with :meth:`plot`. Build it with
    :func:`standard_speech_spectra`; the frozen instance is a thin, plottable
    wrapper and re-runs none of the maths (the band levels are the tabulated
    constants that :func:`standard_speech_spectrum` returns).

    :ivar frequencies: The 18 one-third-octave band centre frequencies, in hertz
        (160 Hz to 8000 Hz).
    :ivar vocal_efforts: The vocal efforts carried, in order; each one of
        ``"normal"``, ``"raised"``, ``"loud"`` or ``"shout"``.
    :ivar levels: The standard speech spectrum level ``Ui``, in dB SPL, as a
        ``(len(vocal_efforts), 18)`` array; row ``i`` is the spectrum for
        ``vocal_efforts[i]``.
    """

    frequencies: np.ndarray
    vocal_efforts: tuple[str, ...]
    levels: np.ndarray

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
        """Plot the standard speech spectrum level versus frequency band.

        Draws the standard speech spectrum level (dB SPL) over the 18
        one-third-octave bands (160 Hz to 8000 Hz) on a categorical band axis;
        each vocal effort in :attr:`vocal_efforts` is one labelled line, so the
        whole spectrum lifting with vocal effort reads at a glance.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the per-effort ``plot`` calls.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.hearing import plot_standard_speech_spectrum

        return plot_standard_speech_spectrum(
            self, ax=ax, language=check_language(language), **kwargs
        )


def standard_speech_spectra(
    vocal_efforts: str | Sequence[str] = VOCAL_EFFORTS,
) -> StandardSpeechSpectrum:
    """Build the plottable ANSI S3.5-1997 standard speech spectra (Table 3).

    Collects the standard speech spectrum level ``Ui`` of the requested vocal
    efforts (via :func:`standard_speech_spectrum`) into a
    :class:`StandardSpeechSpectrum` that exposes ``.plot()``. The band levels
    are unchanged; this is a thin, plottable wrapper around the existing
    function, which still returns the bare per-band array.

    :param vocal_efforts: A single vocal-effort name or a sequence of names,
        each one of ``"normal"``, ``"raised"``, ``"loud"`` or ``"shout"``.
        Defaults to the full family in the Table 3 order.
    :return: A frozen :class:`StandardSpeechSpectrum`.
    :raises ValueError: for an unknown vocal effort, or an empty selection.
    """
    efforts = (
        (vocal_efforts,)
        if isinstance(vocal_efforts, str)
        else tuple(vocal_efforts)
    )
    if not efforts:
        raise ValueError(
            "'vocal_efforts' cannot be empty; choose at least one of "
            f"{', '.join(VOCAL_EFFORTS)}."
        )
    levels = np.array(
        [standard_speech_spectrum(effort) for effort in efforts],
        dtype=np.float64,
    )
    return StandardSpeechSpectrum(
        frequencies=BAND_CENTERS.copy(),
        vocal_efforts=efforts,
        levels=levels,
    )


def _as_band_vector(
    values: ArrayLike, name: str, procedure: _BandProcedure | None = None
) -> np.ndarray:
    """Validate and return a band vector of the procedure's length."""
    proc = _PROCEDURES["one-third-octave"] if procedure is None else procedure
    expected = proc.band_importance.size
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.ndim != 1 or arr.size != expected:
        # The nominal centre frequencies, not the band limits: for the
        # one-third-octave procedure the limits are the computed 2**(-+1/6) fi,
        # and "142.544 Hz - 8979.7 Hz" is a worse hint than "160 Hz - 8000 Hz".
        centres = proc.frequencies
        raise ValueError(
            f"{name!r} must be a 1-D vector of {expected} "
            f"{proc.method} band values "
            f"({centres[0]:g} Hz - {centres[-1]:g} Hz); got shape {arr.shape}."
        )
    return arr


def _equivalent_masking(
    noise: np.ndarray, self_masked: np.ndarray, procedure: _BandProcedure
) -> np.ndarray:
    """Equivalent masking spectrum level ``Zi`` of clause 5.4.

    The masker level ``Bi`` of each band spreads upward in frequency with the
    level-dependent slope ``Ci = -80 + 0.6 (Bi + 10 log10 Wi)`` dB per decade,
    and the spread contributions add on an energy basis to the equivalent noise
    of the masked band. The octave-band procedure carries no spread of masking,
    so its equivalent masking spectrum level is the equivalent noise spectrum
    level itself.

    The slope is summed in the order the standard prints it for the procedure
    at hand: ``(Bi + 10 lg fi) - 6.353`` for the one-third-octave form, and
    ``Bi + 10 lg Wi`` for the band-limit form, whose offset is ``0.0`` and
    whose subtraction is therefore exact. Floating-point addition is not
    associative, so summing in any other order shifts the result by a few
    units in the last place. That matters here beyond tidiness: this is the
    library's shipped SII, it feeds a report fiche, and a released quantity
    should not drift in its last bits because the code was refactored.
    """
    bandwidth_db, spread_decades = procedure.bandwidth_db, procedure.spread_decades
    if bandwidth_db is None or spread_decades is None:
        return noise.copy()
    slope = -80.0 + 0.6 * (
        self_masked + bandwidth_db - procedure.bandwidth_offset_db
    )
    masking = np.empty(noise.size, dtype=np.float64)
    masking[0] = self_masked[0]
    for i in range(1, noise.size):
        spread = 10.0 ** (
            0.1 * (self_masked[:i] + 3.32 * slope[:i] * spread_decades[i, :i])
        )
        masking[i] = 10.0 * np.log10(10.0 ** (0.1 * noise[i]) + np.sum(spread))
    return masking


def _procedure_speech_spectrum(
    vocal_effort: str, procedure: _BandProcedure
) -> np.ndarray:
    """Standard speech spectrum level ``Ui`` of a procedure, by vocal effort.

    Tables 1, 2 and 4 print all four vocal-effort columns in the standard, but
    only their normal-effort column is carried here, which is the one the
    level-distortion factor of clause 5.7 needs; the four vocal-effort spectra
    of Table 3 are available on the one-third-octave procedure through
    :func:`standard_speech_spectrum`.
    """
    if procedure.method == "one-third-octave":
        return standard_speech_spectrum(vocal_effort)
    if vocal_effort == "normal":
        return procedure.speech_spectrum.copy()
    if vocal_effort in VOCAL_EFFORTS:
        raise ValueError(
            f"The {procedure.method!r} procedure carries the standard speech "
            f"spectrum for normal vocal effort only; {vocal_effort!r} is "
            "carried for the one-third-octave procedure. Pass an explicit "
            "equivalent speech spectrum level instead."
        )
    raise ValueError(
        f"Unknown vocal_effort {vocal_effort!r}; choose from "
        f"{', '.join(VOCAL_EFFORTS)}."
    )


def speech_intelligibility_index(
    speech_spectrum: ArrayLike,
    noise_spectrum: ArrayLike | None = None,
    *,
    threshold: ArrayLike | None = None,
    method: str = "one-third-octave",
    band_importance: ArrayLike | None = None,
) -> SIIResult:
    """Speech Intelligibility Index (ANSI S3.5-1997, any of the four methods).

    All spectra are equivalent spectrum levels (clauses 3.11/3.55) sampled at
    the band centres of the chosen procedure: 18 one-third-octave bands from
    160 Hz to 8000 Hz by default, or the 21 critical bands, the 17
    equally-contributing critical bands or the 6 octave bands.

    :param speech_spectrum: Equivalent speech spectrum level ``Ei'``, in dB SPL.
        A vocal-effort name (``"normal"``, ``"raised"``, ``"loud"`` or
        ``"shout"``) selects the corresponding standard speech spectrum; only
        ``"normal"`` is tabulated outside the one-third-octave procedure.
    :param noise_spectrum: Equivalent noise spectrum level ``Ni'``, in dB SPL;
        ``None`` uses a quiet field (``-80`` dB in every band).
    :param threshold: Equivalent hearing threshold ``Ti'``, in dB HL; ``None``
        uses normal hearing (``0`` in every band).
    :param method: The band procedure, one of :data:`SII_METHODS`:
        ``"critical-band"`` (21 bands, Table 1), ``"equally-contributing"``
        (17 bands, Table 2), ``"one-third-octave"`` (18 bands, Table 3, the
        default) or ``"octave"`` (6 bands, Table 4).
    :param band_importance: Alternative band-importance function ``Ii``, one
        value per band, replacing the procedure's tabulated function (the
        standard's Annex B tabulates functions for specific speech test
        materials); ``None`` uses the tabulated average-speech function.
    :return: An :class:`SIIResult` with the overall index and its ``.plot()``.
    :raises ValueError: if a spectrum has the wrong length, or the method or
        effort name is unknown.
    """
    proc = _procedure(method)
    n_bands = proc.band_importance.size
    if isinstance(speech_spectrum, str):
        e = _procedure_speech_spectrum(speech_spectrum, proc)
    else:
        e = _as_band_vector(speech_spectrum, "speech_spectrum", proc)
    n = (
        np.full(n_bands, -80.0)
        if noise_spectrum is None
        else _as_band_vector(noise_spectrum, "noise_spectrum", proc)
    )
    t = (
        np.zeros(n_bands)
        if threshold is None
        else _as_band_vector(threshold, "threshold", proc)
    )
    importance = (
        proc.band_importance.copy()
        if band_importance is None
        else _as_band_vector(band_importance, "band_importance", proc)
    )

    # Clause 5.4 - self-speech masking and the upward spread of masking.
    b = np.maximum(n, e - _SELF_SPEECH_MASKING)
    z = _equivalent_masking(n, b, proc)

    # Clause 5.5/5.6 - equivalent internal noise and disturbance. Di is "the
    # larger of" Zi and Xi' (clause 5.6) - a maximum, not an energy sum. The
    # WG S3-79 reference implementation SII.C branches on z >= x' exactly so,
    # the official Hornsby SII worksheet computes it as =MAX() in every band,
    # and the R CRAN worked example C.1 confirms (8000 Hz row: Di = Xi' = -7.1).
    xp = proc.internal_noise + t
    d = np.maximum(z, xp)

    # Clause 5.7/5.8 - level distortion, band audibility and the index. The
    # level-distortion factor Li compares Ei' with the *normal* standard
    # speech spectrum plus 10 dB for every vocal effort (clause 5.7, Formula
    # 6.19 uses Ui for normal vocal effort only) - confirmed against SII.C
    # (whose u[] is the normal-effort spectrum for every input and every
    # procedure) and the official worksheet; not a bug.
    level_factor = np.clip(
        1.0
        - (e - proc.speech_spectrum - _LEVEL_DISTORTION_OFFSET)
        / _LEVEL_DISTORTION_RANGE,
        0.0,
        1.0,
    )
    audibility = np.clip((e - d + _SPEECH_PEAK) / _AUDIBILITY_RANGE, 0.0, 1.0)
    a = level_factor * audibility
    sii = float(np.sum(importance * a))

    return SIIResult(
        sii=sii,
        band_audibility=a,
        band_importance=importance,
        frequencies=proc.frequencies.copy(),
        speech_spectrum=e,
        disturbance=d,
        masking=z,
        level_distortion=level_factor,
        method=proc.method,
    )


@dataclass(frozen=True)
class SIIProcedure:
    """The tabulated band table of one ANSI S3.5-1997 band procedure.

    Bundles the normative constants of one of the standard's four band
    procedures (Tables 1 to 4) so they can be inspected and drawn with
    :meth:`plot`. Build it with :func:`sii_procedure`; the frozen instance is a
    plottable view of the tabulated constants and runs none of the SII maths.

    :ivar method: The procedure name, one of :data:`SII_METHODS`.
    :ivar frequencies: Nominal band centre frequencies, in hertz.
    :ivar band_edges: Band limits, in hertz; one value more than the number of
        bands, so band ``i`` runs from ``band_edges[i]`` to ``band_edges[i+1]``.
    :ivar band_importance: Band-importance function ``Ii`` for average speech
        material. It sums to one, except for the equally-contributing
        procedure, whose printed 0.0588 per band sums to 0.9996.
    :ivar internal_noise: Reference internal noise spectrum level ``Xi``, in
        dB SPL.
    :ivar speech_spectrum: Standard speech spectrum level ``Ui`` for normal
        vocal effort, in dB SPL.
    """

    method: str
    frequencies: np.ndarray
    band_edges: np.ndarray
    band_importance: np.ndarray
    internal_noise: np.ndarray
    speech_spectrum: np.ndarray

    def plot(self, ax: Axes | None = None, *, language: str = "en",
             **kwargs: Any) -> Axes:
        """Plot the band-importance function of the procedure versus frequency.

        Draws the band-importance function ``Ii`` as a step over the band
        limits, so procedures with different band counts and widths can be
        overlaid on the same logarithmic frequency axis and compared directly.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the step ``plot`` call.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.hearing import plot_sii_procedure

        return plot_sii_procedure(
            self, ax=ax, language=check_language(language), **kwargs
        )


def sii_procedure(method: str = "one-third-octave") -> SIIProcedure:
    """Build the plottable band table of an ANSI S3.5-1997 band procedure.

    :param method: The band procedure, one of :data:`SII_METHODS`.
    :return: A frozen :class:`SIIProcedure` carrying the band centre
        frequencies, band limits, band-importance function, reference internal
        noise spectrum level and normal-effort standard speech spectrum level
        of that procedure.
    :raises ValueError: for an unknown procedure name.
    """
    proc = _procedure(method)
    return SIIProcedure(
        method=proc.method,
        frequencies=proc.frequencies.copy(),
        band_edges=proc.band_edges.copy(),
        band_importance=proc.band_importance.copy(),
        internal_noise=proc.internal_noise.copy(),
        speech_spectrum=proc.speech_spectrum.copy(),
    )


# --- Deprecated alias (phonometry 3.1 rename; remove in 4.0) -------------

def __getattr__(name: str) -> Any:
    """PEP 562 shim warning for the renamed band-center constant."""
    if name == "BAND_CENTRES":
        _warn_renamed("BAND_CENTRES", "BAND_CENTERS")
        return BAND_CENTERS
    raise AttributeError(f"module 'phonometry.sii' has no attribute {name!r}")
