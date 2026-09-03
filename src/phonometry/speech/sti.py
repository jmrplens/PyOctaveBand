#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Speech Transmission Index (STI) per IEC 60268-16:2020 (Edition 5).

Implements the full-STI indirect method from impulse responses (Schroeder
modulation transfer function), the direct STIPA method on recorded signals
(Annex B), an Ed.5-conformant STIPA test-signal generator (clauses A.4
and A.6.1) and the Annex M adjustment of a measured result to other speech
and occupancy-noise levels. Only the male speech option exists: Edition 5
removed the female spectrum and weighting factors (foreword, item d).

The computation chain (octave-band MTF -> auditory masking and reception
threshold correction -> effective SNR clipped to +/-15 dB -> transmission
indices -> band MTI -> weighted STI) is numerically identical between
Ed.4 (2011) clauses A.5.2-A.5.6 and Ed.5; the only Ed.5 numeric change is
the male test-signal spectrum (A.6.1).

The level adjustment of :func:`sti_adjusted_for_levels` is the four-step
procedure of Ed.4 (2011) Annex M, verified against the printed
intermediates of its Table M.1. The Ed.5 foreword (item g) says greater
information is given in Annex M about these adjustments, and its table of
contents grows the annex from three printed pages to ten and adds a flow
chart of the steps; the body of the Ed.5 annex could not be obtained, so
what those pages add beside this procedure is unknown here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata
    from ..io._signal import Signal
from scipy import signal

from .._internal.utils import _typesignal
from .._internal.validation import (
    check_engine,
    require_finite_fields,
    require_ranks,
    require_same_length,
)
from .._internal.warnings import PhonometryWarning
from ..filters.core import OctaveFilterBank
from ..filters.frequencies import nominal_frequencies
from ..io._resolve import apply_calibration, resolve_fs, resolve_pair_fs


class STIWarning(PhonometryWarning):
    """Warns about suspect STI/STIPA measurements or inputs."""


# Octave-band analysis range: 125 Hz - 8 kHz (Ed.4 A.5.1 = Ed.5), 7 bands.
_BAND_LIMITS = (125.0, 8000.0)
_NUM_BANDS = 7

# The 8 kHz octave band extends to 11.2 kHz (IEC 61260-1 exact upper edge
# 7943.3 * 2^0.5 = 11220 Hz), so fs/2 must exceed it.
_MIN_FS = 22500

#: Rejection message for a non-positive sampling rate.
_FS_NOT_POSITIVE_MSG = "Sample rate 'fs' must be positive."

# The 14 full-STI modulation frequencies, 0.63-12.5 Hz in nominal
# one-third-octave steps (Ed.4 A.2.2 = Ed.5).
_MOD_FREQS = np.array(
    [0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.0, 12.5]
)

# Male octave-band weighting factors alpha and redundancy factors beta
# (Ed.5 Table A.1; identical to the male rows of Ed.4 Table A.3).
_ALPHA_MALE = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
_BETA_MALE = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])

# Absolute speech reception threshold ART_k in dB SPL per octave band
# (Ed.5 Table A.3; identical to Ed.4 Table A.2).
_ART_DB = np.array([46.0, 27.0, 12.0, 6.5, 7.5, 8.0, 12.0])

# Segment breakpoints, in dB of total level L of the next lower octave
# band, of the four-segment auditory masking function amdB (Ed.5 Table
# A.2 = Ed.4 Table A.1): the steep 1.8 dB/dB middle segment runs from
# 63 dB to 67 dB (0.5 dB/dB on either side), and amdB saturates at its
# constant -10 dB from 100 dB up. All three are levels; 100.0 here is
# unrelated to _ENVELOPE_LPF_HZ = 100.0 below, which is a frequency.
_AMDB_STEEP_SEGMENT_START_DB = 63.0
_AMDB_STEEP_SEGMENT_END_DB = 67.0
_AMDB_SATURATION_DB = 100.0

# Modulation-transfer validity ceiling: a value above 1.3 means the
# measurement is likely invalid (IEC 60268-16 A.5.3 NOTE 1, Ed.4 = Ed.5).
_MTF_INVALID_ABOVE = 1.3

# STIPA modulation frequency pairs per octave band (Table B.1, unchanged
# between Ed.4 and Ed.5) and the source modulation index per component.
_STIPA_F1 = np.array([1.60, 1.00, 0.63, 2.00, 1.25, 0.80, 2.50])
_STIPA_F2 = np.array([8.00, 5.00, 3.15, 10.00, 6.25, 4.00, 12.50])
_STIPA_MOD_INDEX = 0.55  # Annex B: m = 0.55 per component, unchanged in Ed.5

#: Recommended minimum STIPA recording length (IEC 60268-16 STIPA practice
#: recommends 15 s to 25 s). Below this the recovered modulation depths are
#: biased low - and hence the STI too - because the slow modulation
#: components are averaged over too few periods: on an ideal loopback the
#: recovered STI is ~0.956 at 5 s, ~0.994 at 15 s and ~0.998 at 18 s.
_STIPA_MIN_SECONDS = 15.0

# Ed.5 male speech test-signal spectrum, clause A.6.1, in dB relative to
# the 500 Hz band (125 Hz and 250 Hz reduced by 6.2 dB and 3.2 dB vs Ed.4
# Table A.4 - the only numeric change of Edition 5).
_SPEECH_SPECTRUM_ED5 = np.array([-2.5, 0.5, 0.0, -6.0, -12.0, -18.0, -24.0])

# Intensity-envelope low-pass cut-off, ~100 Hz (Ed.4 A.5.2 = Ed.5).
_ENVELOPE_LPF_HZ = 100.0

# Annex F (informative) qualification bands: edges 0.36-0.76 in 0.04
# steps; categories U (< 0.36), J..A, A+ (>= 0.76). Unchanged in Ed.5.
_RATING_EDGES = (0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76)
_RATING_LETTERS = ("U", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A", "A+")


@dataclass(frozen=True)
class STIResult:
    """Result of a Speech Transmission Index computation.

    ``mtf`` holds the modulation transfer values actually used for the
    transmission indices, i.e. after the optional SNR / masking /
    reception-threshold corrections and after clipping to [0, 1]; its
    shape is (7, 14) for full STI and (7, 2) for STIPA. ``mti`` is the
    per-band modulation transfer index (7,), ``band_levels`` and
    ``ambient_levels`` echo the speech and background-noise octave-band
    levels used for the level-dependent corrections (None when they were
    skipped) and ``rating`` is the Annex F qualification letter
    (``A+`` .. ``U``).

    The two level spectra are what :meth:`adjusted_for_levels` reads to
    undo the corrections they produced, so a result that carries them can
    be moved to another speech and occupancy-noise condition.
    """

    sti: float
    mti: np.ndarray
    mtf: np.ndarray
    band_levels: np.ndarray | None
    rating: str
    ambient_levels: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Reject a result whose per-band quantities disagree.

        ``mti`` is the per-band index the fiche tabulates beside the band
        levels and the modulation matrix, all three on the same band axis.

        ``sti`` is pinned finite. Every path to a result runs through
        :func:`_truncated_mtf`, which refuses a modulation transfer matrix
        carrying anything that is not a finite number, and the chain from
        there to the index is arithmetic on values clipped into [0, 1]: no
        measurement can hand back an ``STI`` that is not a number. One built
        by hand crashed the boxed statement of the fiche inside the display
        rounder with "cannot convert float NaN to integer", a message naming
        neither the field nor the result it came from.

        ``ambient_levels`` without ``band_levels`` is refused: noise levels
        only enter the chain against a speech spectrum, and a result holding
        one half of the pair would let :meth:`adjusted_for_levels` undo a
        correction that was never applied.

        :raises ValueError: if the band axes disagree, ``sti`` is not
            finite, or ambient levels arrive without speech levels.
        """
        require_ranks(self, mti=1, mtf=2, band_levels=1, ambient_levels=1)
        require_same_length(self, "mti", ("mtf", 0), "band_levels", "ambient_levels")
        require_finite_fields(self, "sti")
        if self.band_levels is None and self.ambient_levels is not None:
            msg = (
                "STIResult: 'ambient_levels' needs the speech 'band_levels' "
                "they were combined with; got ambient levels alone."
            )
            raise ValueError(msg)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-band MTI bars with the STI and rating letter.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.speech import plot_sti

        return plot_sti(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an IEC 60268-16 speech-transmission-index fiche to a PDF.

        Writes a one-page voice-alarm / public-address intelligibility
        verification report: a standard-basis line stating the measurement
        method (the full STI indirect method from an impulse response, or the
        direct STIPA method on a recorded signal), an optional metadata header
        block, a per-octave-band modulation transfer index table beside the
        per-band MTI bars (the result's own :meth:`plot`), the boxed
        ``STI = X`` single number with the Annex F qualification band, an
        optional verdict row and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a bare fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the minimum required STI (a higher STI
            passes).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform signature; it has no effect on
            the single-layout STI fiche.
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
        check_engine(engine)
        from .._report.sti import render_sti_report

        return render_sti_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )

    def adjusted_for_levels(
        self,
        *,
        operational_level: Sequence[float] | np.ndarray,
        operational_ambient: Sequence[float] | np.ndarray | None = None,
    ) -> STIResult:
        """This result moved to another speech and noise condition (Annex M).

        The measurement condition is the one this result already carries in
        ``band_levels`` and ``ambient_levels``, so only the target condition
        is passed: what the room would have scored occupied, or with the
        talker raised to the operational level. See
        :func:`sti_adjusted_for_levels`, which does the work and states the
        edition the procedure comes from.

        :param operational_level: Speech octave-band levels of the condition
            being simulated, in dB SPL (7 values).
        :param operational_ambient: Occupancy-noise octave-band levels of
            that condition, in dB SPL (7 values); ``None`` simulates a
            silent room.
        :return: A new :class:`STIResult` at the operational levels.
        :raises ValueError: if this result carries no speech band levels,
            i.e. it was computed without the level-dependent corrections and
            its modulation transfer matrix holds nothing to undo.
        """
        if self.band_levels is None:
            msg = (
                "Adjusting to other levels needs the speech band levels the "
                "measurement was corrected with, and this STIResult carries "
                "none: recompute it with 'level' (and 'ambient'), or call "
                "sti_adjusted_for_levels on the measured matrix directly."
            )
            raise ValueError(msg)
        return sti_adjusted_for_levels(
            self.mtf,
            measured_level=self.band_levels,
            measured_ambient=self.ambient_levels,
            operational_level=operational_level,
            operational_ambient=operational_ambient,
        )


def _rating(sti: float) -> str:
    """Annex F qualification letter for an STI value (band edges 0.36-0.76).

    Note: the qualification bands are 0.04 STI wide, so every band edge flips a
    letter under +/-0.005 of STI jitter. Because typical STIPA measurement
    uncertainty (+/-0.02 to 0.03) exceeds the 0.02 half-width, a value near an
    edge can legitimately be graded one letter either way; treat the letter as
    indicative near a boundary and report the numeric STI alongside it.
    """
    idx = int(np.searchsorted(np.asarray(_RATING_EDGES), sti, side="right"))
    return _RATING_LETTERS[idx]


def _masking_amdb(level: np.ndarray | float) -> np.ndarray:
    r"""Level-dependent auditory masking ``amdB`` in dB (Ed.5 Table A.2).

    Four-segment continuous function of the total (signal + noise) level
    ``L`` of the next lower octave band, identical to Ed.4 Table A.1:
    :math:`0.5 L - 65` (:math:`L < 63`),
    :math:`1.8 L - 146.9` (:math:`63 \le L < 67`),
    :math:`0.5 L - 59.8` (:math:`67 \le L < 100`) and
    :math:`-10` (:math:`L \ge 100`).
    """
    lvl = np.asarray(level, dtype=np.float64)
    return np.asarray(
        np.select(
            [
                lvl < _AMDB_STEEP_SEGMENT_START_DB,
                lvl < _AMDB_STEEP_SEGMENT_END_DB,
                lvl < _AMDB_SATURATION_DB,
            ],
            [0.5 * lvl - 65.0, 1.8 * lvl - 146.9, 0.5 * lvl - 59.8],
            default=-10.0,
        )
    )


def _octave_bank(fs: int) -> OctaveFilterBank:
    """Octave filter bank for the 7 STI bands (IEC 61260-1 class 0/1)."""
    bank = OctaveFilterBank(fs=fs, fraction=1, order=6, limits=list(_BAND_LIMITS))
    if bank.num_bands != _NUM_BANDS:
        msg = (
            f"Expected {_NUM_BANDS} octave bands between 125 Hz and 8 kHz, "
            f"got {bank.num_bands} at fs={fs}."
        )
        raise ValueError(msg)
    return bank


def _validate_band_vector(
    values: Sequence[float] | np.ndarray, name: str
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (_NUM_BANDS,):
        msg = (
            f"'{name}' must contain exactly {_NUM_BANDS} octave-band values "
            f"(125 Hz - 8 kHz), got shape {arr.shape}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must contain only finite dB values."
        raise ValueError(msg)
    return arr


def _truncated_mtf(mtf: np.ndarray, *, stacklevel: int = 4) -> np.ndarray:
    """Validate the modulation transfer matrix and truncate it to 1,0.

    Ed.4 A.5.3 NOTE 1 (= Ed.5): a value above 1,3 means the measurement is
    invalid, and every value is truncated to 1,0 before the chain runs.
    ``stacklevel`` points the warning at the caller's caller: the default 4
    fits the public functions that route through :func:`_sti_from_mtf`, and
    :func:`sti_adjusted_for_levels`, which calls here directly, passes 3.
    """
    m = np.array(mtf, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != _NUM_BANDS:  # noqa: PLR2004
        msg = (
            f"'mtf' must have shape ({_NUM_BANDS}, n_modulation_frequencies), "
            f"got {m.shape}."
        )
        raise ValueError(msg)
    if np.any(m < 0.0) or not np.all(np.isfinite(m)):
        msg = "Modulation transfer values must be finite and >= 0."
        raise ValueError(msg)
    if np.any(m > _MTF_INVALID_ABOVE):
        warnings.warn(
            "Modulation transfer values above 1.3 detected: the measurement "
            "is likely invalid (IEC 60268-16 A.5.3). Values truncated to 1.0.",
            STIWarning,
            stacklevel=stacklevel,
        )
    return np.minimum(m, 1.0)


def _snr_vector(
    snr: float | Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    """The per-band signal-to-noise ratios, broadcast from a scalar if given."""
    if snr is None:
        return None
    snr_arr = np.asarray(snr, dtype=np.float64)
    if snr_arr.ndim == 0:
        return np.full(_NUM_BANDS, float(snr_arr))
    if snr_arr.shape != (_NUM_BANDS,):
        msg = (
            f"'snr' must be a scalar or a vector of {_NUM_BANDS} "
            f"octave-band values, got shape {snr_arr.shape}."
        )
        raise ValueError(msg)
    return snr_arr


@dataclass(frozen=True)
class _LevelCorrection:
    r"""The level-dependent correction of one speech and noise condition.

    One entry per octave band, in the order Table M.1 of Annex M tabulates
    them, so the whole of A.5.3 can be read off a single condition:

    - ``snr``: speech level minus noise level, dB (None without noise);
    - ``noise_transfer``: :math:`m_k(f)` for noise only,
      :math:`I_k / (I_k + I_{\mathrm{n},k})`;
    - ``combined_level``: the combined speech and noise level, dB;
    - ``masking_db``: the auditory masking factor ``amf`` in dB, read off
      the combined level of the *next lower* band. The 125 Hz band has no
      band below it to mask it, so its entry is ``-inf`` dB, which is the
      zero masking intensity that Table M.1 prints as "not applicable";
    - ``intensity``: :math:`I_k`, the combined squared sound pressure as a
      plain ratio to :math:`p_0^2 = (20\ \mu\mathrm{Pa})^2`;
    - ``intensity_masking`` and ``intensity_threshold``:
      :math:`I_{\mathrm{am},k}` and :math:`I_{\mathrm{rt},k}` on that same
      ratio scale;
    - ``masking_threshold_transfer``:
      :math:`I_k / (I_k + I_{\mathrm{am},k} + I_{\mathrm{rt},k})`;
    - ``factor``: the whole correction
      :math:`I_{\mathrm{s},k} / (I_k + I_{\mathrm{am},k} +
      I_{\mathrm{rt},k})`, the product of the two transfers above, which
      multiplies every modulation transfer value of the band.
    """

    speech_level: np.ndarray
    ambient_level: np.ndarray | None
    snr: np.ndarray | None
    noise_transfer: np.ndarray
    combined_level: np.ndarray
    masking_db: np.ndarray
    intensity: np.ndarray
    intensity_masking: np.ndarray
    intensity_threshold: np.ndarray
    masking_threshold_transfer: np.ndarray
    factor: np.ndarray


def _level_correction(
    speech_level: np.ndarray, ambient_level: np.ndarray | None
) -> _LevelCorrection:
    """Every term of the A.5.3 correction for one condition (Ed.4 = Ed.5).

    The single place the auditory masking of Table A.2 and the absolute
    reception threshold of Table A.3 are turned into a multiplicative
    correction, so the forward chain and the Annex M adjustment cannot
    drift apart: the adjustment divides by the factor this returns for the
    measurement condition and multiplies by the one it returns for the
    operational condition.
    """
    i_signal = 10.0 ** (speech_level / 10.0)
    i_noise = (
        np.zeros(_NUM_BANDS)
        if ambient_level is None
        else 10.0 ** (ambient_level / 10.0)
    )
    i_total = i_signal + i_noise
    level_total = 10.0 * np.log10(i_total)
    # Masking only acts on the next higher band; 125 Hz is unmasked.
    masking_db = np.full(_NUM_BANDS, -np.inf)
    masking_db[1:] = _masking_amdb(level_total[:-1])
    i_masking = np.zeros(_NUM_BANDS)
    i_masking[1:] = i_total[:-1] * 10.0 ** (masking_db[1:] / 10.0)
    i_threshold = 10.0 ** (_ART_DB / 10.0)
    return _LevelCorrection(
        speech_level=speech_level,
        ambient_level=ambient_level,
        snr=None if ambient_level is None else speech_level - ambient_level,
        noise_transfer=i_signal / i_total,
        combined_level=level_total,
        masking_db=masking_db,
        intensity=i_total,
        intensity_masking=i_masking,
        intensity_threshold=i_threshold,
        masking_threshold_transfer=i_total / (i_total + i_masking + i_threshold),
        factor=i_signal / (i_total + i_masking + i_threshold),
    )


def _corrected_mtf(
    m: np.ndarray,
    snr_arr: np.ndarray | None,
    level: Sequence[float] | np.ndarray | None,
    ambient: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply the noise and level-dependent corrections of A.5.3.

    Without absolute levels only the signal-to-noise correction applies; with
    them the auditory masking and absolute reception threshold join it.
    Returns the corrected matrix and the validated speech and noise band
    levels, which the result carries.
    """
    if level is None:
        if ambient is not None:
            msg = (
                "'ambient' requires the speech octave-band levels 'level' to "
                "form the intensity-domain correction; pass 'snr' instead."
            )
            raise ValueError(msg)
        if snr_arr is not None:
            # I_k / (I_k + I_n,k) with I_n,k = I_k * 10^(-SNR/10).
            m = m / (1.0 + 10.0 ** (-snr_arr[:, np.newaxis] / 10.0))
        # No absolute level information: the auditory masking and absolute
        # reception threshold corrections are skipped.
        return m, None, None

    band_levels = _validate_band_vector(level, "level")
    if ambient is not None:
        ambient_arr = _validate_band_vector(ambient, "ambient")
    elif snr_arr is not None:
        ambient_arr = band_levels - snr_arr
    else:
        ambient_arr = None
    correction = _level_correction(band_levels, ambient_arr)
    return m * correction.factor[:, np.newaxis], band_levels, ambient_arr


def _index_from_corrected_mtf(
    m: np.ndarray,
    band_levels: np.ndarray | None,
    ambient_levels: np.ndarray | None,
) -> STIResult:
    r"""A.5.4 to A.5.6 on a matrix that already carries every correction.

    Effective SNR clipped to +/-15 dB, transmission indices
    :math:`TI = (SNR_{\mathrm{eff}} + 15)/30`, the band MTIs and the
    male-weighted STI truncated to 1,0. ``m`` must already lie in [0, 1];
    :func:`_truncated_mtf` is what puts it there.
    """
    # m = 0 and m = 1 map to the clip limits through the log divergences.
    with np.errstate(divide="ignore"):
        snr_eff = 10.0 * np.log10(m / (1.0 - m))
    snr_eff = np.clip(snr_eff, -15.0, 15.0)
    ti = (snr_eff + 15.0) / 30.0  # A.5.5
    mti = ti.mean(axis=1)

    # A.5.6 with the Ed.5 Table A.1 male factors; truncated to 1,0 (the
    # known alpha/beta artefact can push the raw sum above 1).
    sti = float(
        np.dot(_ALPHA_MALE, mti) - np.dot(_BETA_MALE, np.sqrt(mti[:-1] * mti[1:]))
    )
    sti = min(sti, 1.0)
    return STIResult(
        sti=sti,
        mti=mti,
        mtf=m,
        band_levels=band_levels,
        rating=_rating(sti),
        ambient_levels=ambient_levels,
    )


def _sti_from_mtf(
    mtf: np.ndarray,
    snr: float | Sequence[float] | np.ndarray | None = None,
    level: Sequence[float] | np.ndarray | None = None,
    ambient: Sequence[float] | np.ndarray | None = None,
) -> STIResult:
    r"""STI computation chain from a matrix of modulation transfer values.

    Applies, in order: m > 1 truncation (Ed.4 A.5.3 NOTE 1), the optional
    signal-to-noise and level-dependent corrections (Ed.4 A.5.3 = Ed.5),
    the effective SNR with +/-15 dB limits (A.5.4), the transmission
    indices :math:`TI = (SNR_{\mathrm{eff}} + 15)/30` (A.5.5), the band MTIs
    and the final
    weighted STI truncated to 1.0 (A.5.6, male factors of Ed.5 Table A.1).

    Noise handling: ``snr`` alone multiplies ``m`` by
    :math:`1/(1 + 10^{-SNR/10})`,
    which is exactly the :math:`I_k/(I_k + I_{\mathrm{n},k})` factor of the standard.
    When ``level`` is provided the full intensity-domain correction
    :math:`m' = m \cdot I_k / (I_k + I_{\mathrm{am},k} + I_{\mathrm{rt},k} + I_{\mathrm{n},k})` is used
    instead, with ``ambient`` (or ``level - snr``) defining :math:`I_{\mathrm{n},k}`,
    so the noise degradation is never applied twice.
    """
    m = _truncated_mtf(mtf)

    if snr is not None and ambient is not None:
        msg = "Provide either 'snr' or 'ambient' noise levels, not both."
        raise ValueError(msg)

    return _index_from_corrected_mtf(
        *_corrected_mtf(m, _snr_vector(snr), level, ambient)
    )


def sti_adjusted_for_levels(
    mtf: np.ndarray,
    *,
    measured_level: Sequence[float] | np.ndarray,
    measured_ambient: Sequence[float] | np.ndarray | None = None,
    operational_level: Sequence[float] | np.ndarray,
    operational_ambient: Sequence[float] | np.ndarray | None = None,
) -> STIResult:
    r"""STI of a measured MTF matrix moved to other speech and noise levels.

    A room is measured empty, at whatever level the test signal ran; the
    question is what the STI would be with the room occupied and the talker
    at the operational level. The four steps of IEC 60268-16 Annex M answer
    it without measuring again:

    1. acquire the modulation transfer matrix together with the speech and
       background-noise octave-band levels that were present during the
       measurement (``mtf``, ``measured_level``, ``measured_ambient``);
    2. divide out the correction those levels produced, which removes the
       background noise, the auditory masking and the reception threshold
       and leaves the matrix of the transmission channel alone;
    3. multiply by the correction the operational levels produce, putting
       the occupancy noise, masking and threshold of the simulated
       condition back in;
    4. process the resulting matrix into the index by the usual A.5.4 to
       A.5.6 chain.

    Steps 2 and 3 run the one level-dependent correction of clause A.5.3
    that the forward chain applies, once inverted and once as it stands, so
    the adjustment cannot drift from the masking and threshold model the
    rest of the module uses.

    What is implemented is the Ed.4 (2011) Annex M procedure, verified
    against the printed intermediates of its Table M.1 worked example. The
    Ed.5 foreword (item g) says greater information is given in Annex M
    about these adjustments, and its table of contents grows the annex from
    three printed pages to ten; the body of the Ed.5 annex could not be
    obtained, so what those pages add beside this procedure is unknown
    here.

    .. note:: ``mtf`` is the matrix *as measured*, with the noise, masking
       and threshold of the measurement still in it, which is what
       :attr:`STIResult.mtf` holds after a measurement run
       with ``level`` and ``ambient``. Feeding a matrix that never had them
       applied removes what was never added and lowers the result.
       :meth:`STIResult.adjusted_for_levels` takes the measurement levels
       from the result itself and is the safe route.

    :param mtf: Measured modulation transfer matrix, shape
        (7, n_modulation_frequencies).
    :param measured_level: Speech octave-band levels during the
        measurement, dB SPL (7 values).
    :param measured_ambient: Background-noise octave-band levels during the
        measurement, dB SPL (7 values); ``None`` for a measurement whose
        matrix carries masking and threshold but no noise.
    :param operational_level: Speech octave-band levels of the condition
        being simulated, dB SPL (7 values).
    :param operational_ambient: Occupancy-noise octave-band levels of that
        condition, dB SPL (7 values); ``None`` simulates a silent room.
    :return: :class:`STIResult` at the operational levels, carrying them in
        ``band_levels`` and ``ambient_levels``.
    :raises ValueError: if a level vector is not 7 finite values, or ``mtf`` is not
        a (7, n) matrix of finite non-negative values.
    """
    measured = _level_correction(
        _validate_band_vector(measured_level, "measured_level"),
        None
        if measured_ambient is None
        else _validate_band_vector(measured_ambient, "measured_ambient"),
    )
    operational = _level_correction(
        _validate_band_vector(operational_level, "operational_level"),
        None
        if operational_ambient is None
        else _validate_band_vector(operational_ambient, "operational_ambient"),
    )
    # Step 2 leaves the channel alone and may exceed 1,0; step 3 puts the
    # operational condition back, and only then is the matrix truncated.
    source = _truncated_mtf(mtf, stacklevel=3) / measured.factor[:, np.newaxis]
    adjusted = _truncated_mtf(source * operational.factor[:, np.newaxis], stacklevel=3)
    return _index_from_corrected_mtf(
        adjusted, operational.speech_level, operational.ambient_level
    )


@overload
def sti_from_impulse_response(
    ir: Signal | list[float] | np.ndarray,
    fs: int | None,
    snr: None,
    level: Sequence[float] | np.ndarray,
    ambient: Sequence[float] | np.ndarray,
) -> STIResult: ...


@overload
def sti_from_impulse_response(
    ir: Signal | list[float] | np.ndarray,
    fs: int | None = ...,
    snr: None = ...,
    *,
    level: Sequence[float] | np.ndarray,
    ambient: Sequence[float] | np.ndarray,
) -> STIResult: ...


@overload
def sti_from_impulse_response(
    ir: Signal | list[float] | np.ndarray,
    fs: int | None = ...,
    snr: float | Sequence[float] | np.ndarray | None = ...,
    level: Sequence[float] | np.ndarray | None = ...,
) -> STIResult: ...


def sti_from_impulse_response(
    ir: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    snr: float | Sequence[float] | np.ndarray | None = None,
    level: Sequence[float] | np.ndarray | None = None,
    ambient: Sequence[float] | np.ndarray | None = None,
) -> STIResult:
    r"""Full STI from a room/system impulse response (indirect method).

    The impulse response is filtered into the seven octave bands 125 Hz -
    8 kHz (IEC 61260-1 filters) and the modulation transfer function is
    obtained from the Schroeder integral (IEC 60268-16, indirect method):

    .. math::

       m_k(f_m) = \left| \int h_k^2(t)\, e^{-j 2\pi f_m t}\, dt \right|
       \Big/ \int h_k^2(t)\, dt

    at the 14 modulation frequencies 0,63-12,5 Hz
    (A.2.2). The result then follows the standard chain: optional noise
    degradation, optional auditory masking and absolute reception
    threshold correction, effective SNR clipped to +/-15 dB, transmission
    indices, band MTIs and the male-weighted STI (Ed.5 Table A.1).

    When neither ``level`` nor ``ambient`` is given the level-dependent
    auditory masking and the absolute reception threshold corrections are
    skipped (they require absolute band levels), matching the common
    "noise-free indirect measurement" use of the standard.

    .. note:: The indirect method has a small positive MTF bias that grows
       with reverberation time for a finite noise-carrier IR. It stays within
       the IEC 60268-16 A.5.1.2 systematic-error allowance (<= 0.01 STI) up to
       about T60 = 4 s, reaching ~+0.012 STI only in very reverberant rooms
       (T60 ~ 8 s, STI ~ 0.19). It is a property of the finite IR, not of the
       (exact) Schroeder integration, and does not depend on IR truncation.
       At very short reverberation times (T60 < 0.25 s) the zero-phase
       analysis bank smears h^2 slightly, biasing individual low-band
       high-modulation-frequency MTF cells by up to ~0.02-0.04 while the
       STI itself stays within ~0.001.

    :param ir: Impulse response (1D). Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: every modulation index is normalised by
        the total intensity of its own band, so a factor on the record
        moves neither the transfer values nor the STI. The absolute levels
        the noise corrections need arrive through ``level`` and
        ``ambient``, in dB, not from the samples.
    :param fs: Sample rate in Hz (>= 22,5 kHz so the 8 kHz band fits).
        Required for a bare array; a :class:`~phonometry.io.Signal` brings
        its own, and an explicit value that disagrees with it raises
        instead of silently winning.
    :param snr: Optional signal-to-noise ratio in dB, scalar or one value
        per octave band. Degrades m by 1/(1 + 10^(-SNR/10)); combined
        with ``level`` it is interpreted as ambient levels
        ``level - snr`` so noise is not applied twice. Mutually
        exclusive with ``ambient``.
    :param level: Optional speech octave-band levels in dB SPL (7 values)
        at the listener position; enables the auditory masking (Ed.5
        Table A.2) and reception threshold (Ed.5 Table A.3) corrections.
    :param ambient: Optional ambient noise octave-band levels in dB SPL
        (7 values); requires ``level``.
    :return: :class:`STIResult` with ``mtf`` of shape (7, 14).
    """
    fs = resolve_fs(ir, fs, name="ir")
    ir_proc = apply_calibration(ir, _typesignal(ir, name="ir"))
    if ir_proc.ndim != 1:
        msg = "sti_from_impulse_response expects a 1D impulse response."
        raise ValueError(msg)
    if fs <= 0:
        raise ValueError(_FS_NOT_POSITIVE_MSG)
    if fs < _MIN_FS:
        msg = (
            f"Sample rate 'fs' must be >= {_MIN_FS} Hz: the 8 kHz octave band "
            "extends to 11.2 kHz (IEC 61260-1)."
        )
        raise ValueError(msg)
    if not np.any(ir_proc):
        msg = "Impulse response 'ir' is silent."
        raise ValueError(msg)

    bank = _octave_bank(fs)
    _, _, bands = bank.filter(
        ir_proc, sigbands=True, detrend=False, calculate_level=False, zero_phase=True
    )
    p_bands = np.asarray(bands) ** 2  # h_k^2(t), shape (7, n)
    denom = p_bands.sum(axis=1)
    if np.any(denom <= 0.0):
        msg = "An octave band of the impulse response has no energy."
        raise ValueError(msg)

    t = np.arange(ir_proc.shape[-1]) / fs
    mtf = np.empty((_NUM_BANDS, _MOD_FREQS.size))
    for j, fm in enumerate(_MOD_FREQS):
        kernel = np.exp(-2j * np.pi * fm * t)
        mtf[:, j] = np.abs(p_bands @ kernel) / denom
    return _sti_from_mtf(mtf, snr=snr, level=level, ambient=ambient)


def _intensity_envelopes(x: np.ndarray, fs: int) -> np.ndarray:
    """Octave-band intensity envelopes I_k(t) = x_k^2(t) low-passed at
    ~100 Hz (Ed.4 A.5.2 = Ed.5), shape (7, n).

    Zero-phase band filtering: forward-backward filtering has no phase
    distortion (the A.3.1.2 edge-carrier bias criterion, < 0,01 STI over
    TI 0,1-0,9) and doubles the effective skirt attenuation to >= 41 dB
    at the adjacent octave (the C.4.2 filter-slope criterion, m >= 0,5,
    steeper than IEC 61260-1 class 1). Single-pass filtering fails both
    verification criteria of IEC 60268-16:2020.
    """
    bank = _octave_bank(fs)
    _, _, bands = bank.filter(x, sigbands=True, calculate_level=False, zero_phase=True)
    sos = signal.butter(4, _ENVELOPE_LPF_HZ, btype="lowpass", fs=fs, output="sos")
    env = np.empty((_NUM_BANDS, x.shape[-1]))
    for k, band in enumerate(bands):
        # np.asarray, not a cast: a bank on the default calibration hands
        # back Signals, and a Signal has no arithmetic of its own.
        env[k] = signal.sosfiltfilt(sos, np.asarray(band) ** 2)
    return env


def _stipa_modulation_depths(env: np.ndarray, fs: int) -> np.ndarray:
    r"""Modulation depths :math:`mdr_{k,f_m}` of the intensity envelopes at
    the two Table B.1 frequencies of each band, over an integer number of
    periods (Ed.4 A.5.2):

    .. math::

       mdr = \frac{2 \sqrt{\left( \sum I \sin \right)^2
       + \left( \sum I \cos \right)^2}}{\sum I}
    """
    n = env.shape[-1]
    duration = n / fs
    mdr = np.empty((_NUM_BANDS, 2))
    empty_bands: list[int] = []
    for k in range(_NUM_BANDS):
        for j, fm in enumerate((float(_STIPA_F1[k]), float(_STIPA_F2[k]))):
            periods = int(np.floor(duration * fm))
            if periods < 1:
                msg = (
                    f"Signal too short for STIPA: it must contain at least one "
                    f"full period of the {fm} Hz modulation (>= {1.0 / fm:.2f} s; "
                    "the standard recommends 15 s to 25 s)."
                )
                raise ValueError(msg)
            n_use = round(periods / fm * fs)
            seg = env[k, :n_use]
            total = float(np.sum(seg))
            if total <= 0.0:
                # A band without energy has an undefined modulation depth.
                # Report m = 0 (TI = 0, the worst-case reading of a real
                # STIPA meter on a dead band) and warn: the IEC 60268-16
                # C.4.2 filter-slope verification signals legitimately
                # carry energy in only two of the seven bands.
                if k not in empty_bands:
                    empty_bands.append(k)
                mdr[k, j] = 0.0
                continue
            t = np.arange(n_use) / fs
            s = float(np.dot(seg, np.sin(2.0 * np.pi * fm * t)))
            c = float(np.dot(seg, np.cos(2.0 * np.pi * fm * t)))
            mdr[k, j] = 2.0 * float(np.hypot(s, c)) / total
    if empty_bands:
        warnings.warn(
            f"No energy in octave band(s) {empty_bands} (125 Hz - 8 kHz, "
            "index 0-6): modulation depths set to 0 (TI = 0) there. A full "
            "STIPA measurement needs all seven carriers of the test signal.",
            STIWarning,
            stacklevel=3,
        )
    return mdr


@overload
def stipa(
    x: Signal | list[float] | np.ndarray,
    fs: int | None,
    reference: Signal | list[float] | np.ndarray | None,
    level: Sequence[float] | np.ndarray,
    ambient: Sequence[float] | np.ndarray,
) -> STIResult: ...


@overload
def stipa(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = ...,
    reference: Signal | list[float] | np.ndarray | None = ...,
    *,
    level: Sequence[float] | np.ndarray,
    ambient: Sequence[float] | np.ndarray,
) -> STIResult: ...


@overload
def stipa(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = ...,
    reference: Signal | list[float] | np.ndarray | None = ...,
    level: Sequence[float] | np.ndarray | None = ...,
) -> STIResult: ...


def stipa(
    x: Signal | list[float] | np.ndarray,
    fs: int | None = None,
    reference: Signal | list[float] | np.ndarray | None = None,
    level: Sequence[float] | np.ndarray | None = None,
    ambient: Sequence[float] | np.ndarray | None = None,
) -> STIResult:
    """STIPA on a recorded test signal (direct method, Annex B).

    The recording is filtered into the seven octave bands, squared and
    low-passed (~100 Hz) into intensity envelopes, and the modulation
    depths at the two Table B.1 modulation frequencies of each band are
    measured with the sine/cosine correlation over an integer number of
    periods (Ed.4 A.5.2 = Ed.5). The modulation transfer values are the
    measured depths normalized by the source modulation index 0,55
    (Annex B) - or by the depths measured on ``reference`` when the
    actually emitted signal is supplied - and feed the same masking /
    threshold / TI / STI chain as the full method.

    Physical background noise is already contained in the recording; use
    ``level`` (and optionally ``ambient``) only to enable the absolute
    level-dependent corrections, which are otherwise skipped.

    An :class:`STIWarning` is emitted when the recording is shorter than
    the recommended 15 s (IEC 60268-16 STIPA practice, 15 s to 25 s),
    because the slow modulation components are then averaged over too few
    periods and the recovered modulation depths - and hence the STI - are
    biased low (an ideal loopback gives STI ~0.956 at 5 s vs ~0.998 at 18 s).

    :param x: Recorded STIPA signal (1D), 15 s to 25 s recommended. Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples and then cancels: every modulation index is normalised by
        the total intensity of its own band, so a factor on the record
        moves neither the transfer values nor the STI. The absolute levels
        the noise corrections need arrive through ``level`` and
        ``ambient``, in dB, not from the samples.
    :param fs: Sample rate in Hz (>= 22,5 kHz). Required for a bare array;
        a :class:`~phonometry.io.Signal` brings its own, and an explicit
        value that disagrees with it raises instead of silently winning.
    :param reference: Optional reference recording of the undistorted
        test signal; its measured modulation depths replace the nominal
        0,55 as normalization (useful for non-conformant sources). Accepts
        a :class:`phonometry.io.Signal`. It is a second recording of the
        same test signal, so it has to share the rate: two Signals that
        disagree are refused rather than arbitrated, and measuring the
        reference on a rate that is not its own would return a perfect STI
        for a mismatch. Its calibration is applied like any other record's,
        and then cancels, because what is taken from it is a modulation
        depth and those are normalised.
    :param level: Optional speech octave-band levels in dB SPL (7 values)
        enabling auditory masking and reception threshold corrections.
    :param ambient: Optional ambient noise octave-band levels in dB SPL
        (7 values); requires ``level``.
    :return: :class:`STIResult` with ``mtf`` of shape (7, 2).
    """
    # The reference is a second recording of the same test signal, so it has
    # to have been made at the same rate: measuring its modulation depths on
    # a rate that is not its own returns a perfect STI for a mismatch.
    fs = (
        resolve_fs(x, fs, name="x")
        if reference is None
        else resolve_pair_fs(x, reference, fs, names=("x", "reference"))
    )
    x_proc = apply_calibration(x, _typesignal(x))
    if x_proc.ndim != 1:
        msg = "stipa expects a 1D signal."
        raise ValueError(msg)
    if fs <= 0:
        raise ValueError(_FS_NOT_POSITIVE_MSG)
    if fs < _MIN_FS:
        msg = (
            f"Sample rate 'fs' must be >= {_MIN_FS} Hz: the 8 kHz octave band "
            "extends to 11.2 kHz (IEC 61260-1)."
        )
        raise ValueError(msg)

    if x_proc.size / fs < _STIPA_MIN_SECONDS:
        warnings.warn(
            f"STIPA recording is {x_proc.size / fs:.1f} s, shorter than the "
            f"recommended {_STIPA_MIN_SECONDS:.0f} s (IEC 60268-16 STIPA "
            "practice, 15 s to 25 s): the recovered modulation depths, and "
            "hence the STI, are biased low.",
            STIWarning,
            stacklevel=2,
        )

    mdr = _stipa_modulation_depths(_intensity_envelopes(x_proc, fs), fs)
    if reference is not None:
        ref_proc = apply_calibration(
            reference, _typesignal(reference, name="reference")
        )
        if ref_proc.ndim != 1:
            msg = "'reference' must be a 1D signal."
            raise ValueError(msg)
        mdt = _stipa_modulation_depths(_intensity_envelopes(ref_proc, fs), fs)
        if np.any(mdt <= 0.0):
            msg = "'reference' has zero modulation depth in a band."
            raise ValueError(msg)
        mtf = mdr / mdt
    else:
        mtf = mdr / _STIPA_MOD_INDEX
    return _sti_from_mtf(mtf, level=level, ambient=ambient)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian pink (1/f power) noise of length ``n`` via FFT shaping."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n)
    freq[0] = freq[1]  # placeholder; DC is zeroed below
    spectrum /= np.sqrt(freq)
    spectrum[0] = 0.0
    return np.asarray(np.fft.irfft(spectrum, n))


def stipa_signal(
    fs: int,
    seconds: float = 18.0,
    level_db: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate an IEC 60268-16:2020 conformant STIPA test signal.

    Pink-noise carriers are band-limited to half-octave bands centred on
    the seven octave-band frequencies (clause A.4), set to the Ed.5 male
    speech spectrum of clause A.6.1 (``-2,5; 0,5; 0; -6; -12; -18; -24``
    dB re the 500 Hz band) and intensity-modulated with
    ``0,5 (1 + 0,55 (sin 2 pi f1 t - sin 2 pi f2 t))`` - the Table B.1
    frequency pair of each band, 180 degrees between components, applied
    in amplitude through its square root (Annex B).

    :param fs: Sample rate in Hz (>= 22,5 kHz).
    :param seconds: Duration in seconds (the standard recommends 15 s to
        25 s; default 18 s).
    :param level_db: Optional overall level in dB SPL: the output is
        scaled so its RMS, taken as pascals, sits at ``level_db`` re
        20 uPa. Default (None) normalizes the RMS to 0,1 (digital full
        scale headroom for the 12-14 dB crest factor).
    :param seed: Seed for the pink-noise generator (None: random).
    :return: Test signal, 1D array of ``round(seconds * fs)`` samples.
    """
    if fs <= 0:
        raise ValueError(_FS_NOT_POSITIVE_MSG)
    if fs < _MIN_FS:
        msg = (
            f"Sample rate 'fs' must be >= {_MIN_FS} Hz: the 8 kHz half-octave "
            "carrier extends beyond 9.4 kHz."
        )
        raise ValueError(msg)
    if seconds <= 0:
        msg = "'seconds' must be positive."
        raise ValueError(msg)

    n = round(seconds * fs)
    rng = np.random.default_rng(seed)
    pink = _pink_noise(n, rng)
    t = np.arange(n) / fs
    centers, _, _, _ = nominal_frequencies(1, list(_BAND_LIMITS))
    half_octave = 2.0**0.25
    out = np.zeros(n)
    for k in range(_NUM_BANDS):
        sos = signal.butter(
            5,
            [centers[k] / half_octave, centers[k] * half_octave],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        carrier = signal.sosfiltfilt(sos, pink)
        carrier = carrier / np.sqrt(np.mean(carrier**2))
        envelope = 0.5 * (
            1.0
            + _STIPA_MOD_INDEX
            * (
                np.sin(2.0 * np.pi * _STIPA_F1[k] * t)
                - np.sin(2.0 * np.pi * _STIPA_F2[k] * t)
            )
        )
        # Intensity modulation: amplitude carries the square root. The
        # exact 1:5 frequency ratios keep the envelope non-negative; the
        # clip only guards against float round-off.
        out += (
            10.0 ** (_SPEECH_SPECTRUM_ED5[k] / 20.0)
            * carrier
            * np.sqrt(np.maximum(envelope, 0.0))
        )

    rms = float(np.sqrt(np.mean(out**2)))
    if level_db is None:
        return out * (0.1 / rms)
    return np.asarray(out * (2e-5 * 10.0 ** (level_db / 20.0) / rms))
