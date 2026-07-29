#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Speech Intelligibility Index (SII) per ANSI S3.5-1997 (R2017).

Implements the one-third-octave-band method (18 bands, 160 Hz - 8000 Hz) of
ANSI S3.5-1997, *American National Standard Methods for the Calculation of the
Speech Intelligibility Index*. From an equivalent speech spectrum level, an
equivalent noise spectrum level and an equivalent hearing threshold, the
procedure forms the self-speech masking, the upward spread of masking, the
equivalent internal noise and disturbance, the level-distortion factor and the
band-audibility function, and weights the latter by the band-importance function
of Table 3 to give the index ``SII`` in [0, 1] (clause 6).

The band-importance function (Table 3, average speech material), the standard
speech spectrum levels by vocal effort (Table 3) and the reference internal
noise spectrum level (Table 3) are the standard's own tabulated constants.
Spectrum levels are as defined in clauses 3.11 and 3.55.

The implementation reproduces the reference implementation ``SII.C`` of ASA
Working Group S3-79 (the committee that maintains ANSI S3.5) to double
precision on its official one-third-octave test cases ``TO.TST`` and
``TO_1.TST``, and computes the Annex C worked examples with the working
group's official errata applied (see ``docs/ERRATA.md``).
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


@dataclass(frozen=True)
class SIIResult:
    """Result of a Speech Intelligibility Index computation (ANSI S3.5-1997).

    :ivar sii: The overall Speech Intelligibility Index in [0, 1] (clause 6).
    :ivar band_audibility: Per-band audibility function ``Ai`` (clause 5.8).
    :ivar band_importance: Per-band importance function ``Ii`` used (Table 3).
    :ivar frequencies: One-third-octave band centre frequencies, in hertz.
    :ivar speech_spectrum: Equivalent speech spectrum level ``Ei'`` per band.
    :ivar disturbance: Equivalent disturbance spectrum level ``Di`` (clause 5.6).
    :ivar masking: Equivalent masking spectrum level ``Zi`` (clause 5.4).
    """

    sii: float
    band_audibility: np.ndarray
    band_importance: np.ndarray
    frequencies: np.ndarray
    speech_spectrum: np.ndarray
    disturbance: np.ndarray
    masking: np.ndarray

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

        Writes a one-page speech-audibility report: a standard-basis line, an
        optional metadata header block, a per-one-third-octave-band table (the
        equivalent speech spectrum ``Ei'``, the band-importance function ``Ii``
        of Table 3 and the band-audibility function ``Ai``) beside the
        audibility and importance-weighted contribution bars (the result's own
        :meth:`plot`), the boxed ``SII = X`` single number, an optional verdict
        row and a footer with the fixed disclaimer.

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
            (``pip install phonometry[report]``).
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


def _as_band_vector(values: ArrayLike, name: str) -> np.ndarray:
    """Validate and return a length-18 one-third-octave band vector."""
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.ndim != 1 or arr.size != _N_BANDS:
        raise ValueError(
            f"{name!r} must be a 1-D vector of {_N_BANDS} one-third-octave "
            f"band values (160 Hz - 8000 Hz); got shape {arr.shape}."
        )
    return arr


def speech_intelligibility_index(
    speech_spectrum: ArrayLike,
    noise_spectrum: ArrayLike | None = None,
    *,
    threshold: ArrayLike | None = None,
) -> SIIResult:
    """Speech Intelligibility Index (ANSI S3.5-1997, one-third-octave method).

    All spectra are equivalent spectrum levels (clauses 3.11/3.55) sampled at
    the 18 one-third-octave band centres from 160 Hz to 8000 Hz.

    :param speech_spectrum: Equivalent speech spectrum level ``Ei'``, in dB SPL.
        A vocal-effort name (``"normal"``, ``"raised"``, ``"loud"`` or
        ``"shout"``) selects the corresponding standard speech spectrum
        (Table 3).
    :param noise_spectrum: Equivalent noise spectrum level ``Ni'``, in dB SPL;
        ``None`` uses a quiet field (``-80`` dB in every band).
    :param threshold: Equivalent hearing threshold ``Ti'``, in dB HL; ``None``
        uses normal hearing (``0`` in every band).
    :return: An :class:`SIIResult` with the overall index and its ``.plot()``.
    :raises ValueError: if a spectrum has the wrong length or effort name.
    """
    if isinstance(speech_spectrum, str):
        e = standard_speech_spectrum(speech_spectrum)
    else:
        e = _as_band_vector(speech_spectrum, "speech_spectrum")
    n = (
        np.full(_N_BANDS, -80.0)
        if noise_spectrum is None
        else _as_band_vector(noise_spectrum, "noise_spectrum")
    )
    t = (
        np.zeros(_N_BANDS)
        if threshold is None
        else _as_band_vector(threshold, "threshold")
    )
    f = BAND_CENTERS

    # Clause 5.4 - self-speech masking and the upward spread of masking.
    v = e - 24.0
    b = np.maximum(n, v)
    c = -80.0 + 0.6 * (b + 10.0 * np.log10(f) - 6.353)
    z = np.empty(_N_BANDS)
    z[0] = b[0]
    for i in range(1, _N_BANDS):
        contrib = 10.0 ** (0.1 * (b[:i] + 3.32 * c[:i] * np.log10(0.89 * f[i] / f[:i])))
        z[i] = 10.0 * np.log10(10.0 ** (0.1 * n[i]) + np.sum(contrib))

    # Clause 5.5/5.6 - equivalent internal noise and disturbance. Di is "the
    # larger of" Zi and Xi' (clause 5.6) - a maximum, not an energy sum. The
    # WG S3-79 reference implementation SII.C branches on z >= x' exactly so,
    # the official Hornsby SII worksheet computes it as =MAX() in every band,
    # and the R CRAN worked example C.1 confirms (8000 Hz row: Di = Xi' = -7.1).
    xp = REFERENCE_INTERNAL_NOISE + t
    d = np.maximum(z, xp)

    # Clause 5.7/5.8 - level distortion, band audibility and the index. The
    # level-distortion factor Li compares Ei' with the *normal* standard
    # speech spectrum plus 10 dB for every vocal effort (clause 5.7, Formula
    # 6.19 uses Ui of Table 3 for normal vocal effort only) - confirmed
    # against SII.C (whose u[] is the normal-effort spectrum for every input)
    # and the official worksheet; not a bug.
    level_factor = np.clip(1.0 - (e - _SPEECH_NORMAL - 10.0) / 160.0, 0.0, 1.0)
    audibility = np.clip((e - d + 15.0) / 30.0, 0.0, 1.0)
    a = level_factor * audibility
    sii = float(np.sum(BAND_IMPORTANCE * a))

    return SIIResult(
        sii=sii,
        band_audibility=a,
        band_importance=BAND_IMPORTANCE.copy(),
        frequencies=f.copy(),
        speech_spectrum=e,
        disturbance=d,
        masking=z,
    )


# --- Deprecated alias (phonometry 3.1 rename; remove in 4.0) -------------

def __getattr__(name: str) -> Any:
    """PEP 562 shim warning for the renamed band-center constant."""
    if name == "BAND_CENTRES":
        _warn_renamed("BAND_CENTRES", "BAND_CENTERS")
        return BAND_CENTERS
    raise AttributeError(f"module 'phonometry.sii' has no attribute {name!r}")
