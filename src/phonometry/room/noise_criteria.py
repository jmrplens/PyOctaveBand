#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Room-noise rating curves per ANSI/ASA S12.2-2019.

Implements the two spectrum-in rating methods of ANSI/ASA S12.2-2019, *Criteria
for Evaluating Room Noise*:

* **Noise Criteria (NC)** by the two-step procedure of clause 5.2.2. The
  speech interference level SIL (clause 3.2, the average of the 500, 1000,
  2000 and 4000 Hz octave-band levels) selects the NC-(SIL) curve; when no
  octave band exceeds that curve, the spectrum is designated NC-(SIL), and
  otherwise the rating is determined by the tangency method (clause 5.2.3:
  the value of the highest NC curve of Table 1 touched by the spectrum,
  reported together with the governing band). The tangency rating is always
  evaluated and kept available on the result.
* **Room Criteria Mark II (RC)** (Annex D, Table D.1). The numerical rating is
  the mid-frequency average ``LMF`` (500/1000/2000 Hz) rounded to the nearest
  decibel (clause D.4); the spectral tag ``N``/``R``/``H`` follows the
  deviation rules of clause D.3.

Both methods evaluate octave-band sound pressure levels over the 16 Hz to
8000 Hz bands tabulated by the standard. The RC Mark II curves are generated
from the -5 dB/octave rule of Annex D (16 Hz equal to 31.5 Hz, with the low
frequencies not dropping below 55 dB), which reproduces Table D.1 exactly.

The balanced noise criteria (NCB), the room noise criterion for fluctuating
low-frequency noise (RNC, Annex A), the acoustically induced vibration and
rattle classification (``RV``, clause D.3.4, which needs the Table 6 test)
and the numeric quality-assessment index (QAI, clause D.5 - deferred by the
standard to external references) are not implemented here.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_equal_shapes,
    require_ranks,
    require_same_length,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from .._report.metadata import ReportMetadata


# ---------------------------------------------------------------------------
# Normative constants - ANSI/ASA S12.2-2019.
# ---------------------------------------------------------------------------

#: Octave-band centre frequencies, in hertz (Table 1 / Table D.1).
OCTAVE_BANDS: np.ndarray = np.array(
    [16.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
    dtype=np.float64,
)

#: NC curve designations (value at 1000 Hz), Table 1.
NC_INDICES: np.ndarray = np.array(
    [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], dtype=np.float64
)

#: NC curve octave-band sound pressure levels, in dB (Table 1), one row per
#: entry of :data:`NC_INDICES`, columns aligned with :data:`OCTAVE_BANDS`.
NC_CURVES: np.ndarray = np.array(
    [
        [78, 61, 47, 36, 28, 22, 18, 14, 12, 11],  # NC-15
        [79, 63, 50, 40, 33, 26, 22, 20, 17, 16],  # NC-20
        [80, 65, 54, 44, 37, 31, 27, 24, 22, 22],  # NC-25
        [81, 68, 57, 48, 41, 35, 32, 29, 28, 27],  # NC-30
        [82, 71, 60, 52, 45, 40, 36, 34, 33, 32],  # NC-35
        [84, 74, 64, 56, 50, 44, 41, 39, 38, 37],  # NC-40
        [85, 76, 67, 60, 54, 49, 46, 44, 43, 42],  # NC-45
        [87, 79, 71, 64, 58, 54, 51, 49, 48, 47],  # NC-50
        [89, 82, 74, 67, 62, 58, 56, 54, 53, 52],  # NC-55
        [90, 85, 77, 71, 66, 63, 60, 59, 58, 57],  # NC-60
        [90, 88, 80, 75, 71, 68, 65, 64, 63, 62],  # NC-65
        [90, 90, 84, 79, 75, 72, 71, 70, 68, 68],  # NC-70
    ],
    dtype=np.float64,
)

#: Integer octave steps of each band relative to 1000 Hz (Annex D generation).
_RC_OCTAVE_STEPS: np.ndarray = np.array(
    [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3], dtype=np.float64
)
_RC_LOW_FREQUENCY_FLOOR = 55.0  # dB, the 31.5 Hz floor (Annex D).

_N_BANDS = OCTAVE_BANDS.size
_F1000_INDEX = 6  # index of the 1000 Hz band in OCTAVE_BANDS.

#: Indices of the four speech-interference bands 500/1000/2000/4000 Hz
#: (clause 3.2: SIL is their arithmetic average).
_SIL_BANDS = slice(5, 9)

#: Indices of the octave bands clause D.4 requires for an RC Mark II rating
#: (at least the bands from 31.5 Hz through 4000 Hz).
_RC_REQUIRED_BANDS = slice(1, 9)

#: Bounds of the tabulated RC Mark II family: Table D.1 tabulates RC-25
#: through RC-50 and clause D.3.5 defines the RC-NN(A) label only for
#: integer ratings in that range; outside it the reference curve extrapolates.
_RC_FAMILY_MIN = 25
_RC_FAMILY_MAX = 50

#: Clause D.3 deviation limits, in dB, over the reference RC Mark II curve:
#: rumble ("R") when a band at or below 500 Hz exceeds it by more than
#: _RC_RUMBLE_LIMIT_DB, hiss ("H") when a band at or above 1000 Hz exceeds
#: it by more than _RC_HISS_LIMIT_DB.
_RC_RUMBLE_LIMIT_DB = 5.0
_RC_HISS_LIMIT_DB = 3.0


@dataclass(frozen=True)
class NCResult:
    """Result of a Noise Criteria (NC) rating (ANSI/ASA S12.2-2019, 5.2).

    :ivar rating: The reported NC designation, following the two-step
        procedure of clause 5.2.2: when no octave band exceeds the NC-(SIL)
        curve chosen from the speech interference level, the designation is
        NC-(SIL); otherwise it is the tangency rating (clause 5.2.3). NaN
        when the spectrum lies outside the NC-15 to NC-70 family of Table 1
        (see ``out_of_range``).
    :ivar governing_frequency: Band, in hertz, where the tangency touch
        occurs (for a spectrum above the family, the band of maximum
        exceedance over the NC-70 curve); NaN for a SIL-designated spectrum,
        which has no governing band.
    :ivar frequencies: Octave-band centre frequencies evaluated, in hertz.
    :ivar levels: Measured octave-band sound pressure levels, in dB.
    :ivar sil: Speech interference level, in dB: the arithmetic average of
        the 500, 1000, 2000 and 4000 Hz octave-band levels (clause 3.2);
        NaN when any of the four bands is missing.
    :ivar tangency_rating: The tangency-method rating (clause 5.2.3), always
        evaluated; NaN when the spectrum lies outside the Table 1 family.
    :ivar method: ``"SIL"`` when the designation is the clause 5.2.2
        NC-(SIL) value, ``"tangency"`` otherwise.
    :ivar out_of_range: ``None`` inside the NC-15 to NC-70 family;
        ``"above"`` when a band exceeds the NC-70 curve (no NC rating is
        defined, ``label`` reads ``">NC-70"``); ``"below"`` when every
        measured band lies below the NC-15 curve (``label`` reads
        ``"<NC-15"``).
    """

    rating: float
    governing_frequency: float
    frequencies: np.ndarray
    levels: np.ndarray
    sil: float = float("nan")
    tangency_rating: float = float("nan")
    method: str = "tangency"
    out_of_range: str | None = None

    def __post_init__(self) -> None:
        """Reject a rating whose spectrum and band axis disagree.

        The rating names a governing band, so a spectrum of the wrong length
        names the wrong one.

        :raises ValueError: if 'frequencies' and 'levels' differ in length.
        """
        require_ranks(self, frequencies=1, levels=1)
        require_same_length(self, "frequencies", "levels")

    @property
    def label(self) -> str:
        """The textual NC designation.

        ``"NC-44"`` for a SIL-designated spectrum, ``"NC-51 (125 Hz)"`` for a
        tangency rating with its governing band (the form of clause 5.2.3),
        and ``">NC-70 (63 Hz)"`` / ``"<NC-15"`` for a spectrum outside the
        Table 1 family, for which the standard defines no NC rating.
        """
        if self.out_of_range == "above":
            return f">NC-70 ({self.governing_frequency:g} Hz)"
        if self.out_of_range == "below":
            return "<NC-15"
        if self.method == "SIL":
            return f"NC-{self.rating:g}"
        return f"NC-{self.rating:g} ({self.governing_frequency:g} Hz)"

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the measured spectrum against the NC curves.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_noise_criterion

        check_language(language)
        return plot_noise_criterion(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a Noise Criteria (NC) assessment fiche to a PDF.

        Writes a one-page room-noise assessment report (ANSI/ASA S12.2-2019):
        the standard-basis line, an optional metadata header block, the
        measured octave-band levels beside the measured spectrum plotted
        against the NC curve family (the result's own :meth:`plot`), the boxed
        NC designation (the clause 5.2.2 NC-(SIL) value, or the tangency
        rating with its governing band; ``">NC-70"`` / ``"<NC-15"`` outside
        the Table 1 family), an optional verdict row and a footer with the
        fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            assessment fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the maximum acceptable NC rating (a lower
            rating is quieter, so the room passes at or below it).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table adds the per-band NC contour
            value read by the tangency method.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            raise ValueError(msg)
        from .._report.ansi_s12_2 import render_nc_report

        return render_nc_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


@dataclass(frozen=True)
class RCResult:
    """Result of a Room Criteria Mark II rating (ANSI/ASA S12.2-2019, Annex D).

    :ivar rating: Numerical RC designation ``LMF`` rounded to the nearest dB.
    :ivar lmf: Mid-frequency average (500/1000/2000 Hz), in dB (clause D.4).
    :ivar classification: Spectral tag, ``"N"`` (neutral), ``"R"`` (rumble) or
        ``"H"`` (hiss) per clause D.3, or ``"RH"`` when both the rumble and
        the hiss deviation tests fire. Clause D.3.5 admits only the letters
        N, R, H or the combination RV, so the combined ``"RH"`` tag is a
        diagnostic extension of this library, not a clause D.3.5 designation.
        The vibration/rattle tag ``RV`` (clause D.3.4) needs the Table 6
        criterion test and is not implemented.
    :ivar reference_curve: The RC Mark II curve used for classification, in dB.
    :ivar frequencies: Octave-band centre frequencies evaluated, in hertz.
    :ivar levels: Measured octave-band sound pressure levels, in dB.
    """

    rating: int
    lmf: float
    classification: str
    reference_curve: np.ndarray
    frequencies: np.ndarray
    levels: np.ndarray

    def __post_init__(self) -> None:
        """Reject a rating whose spectrum, curve and band axis disagree.

        The fiche draws the reference curve over the measured spectrum and
        tabulates both, so a curve of the wrong length is compared band by
        band against the wrong bands.

        :raises ValueError: if the three do not share one length.
        """
        require_ranks(self, frequencies=1, levels=1, reference_curve=1)
        require_same_length(self, "frequencies", "levels", "reference_curve")

    @property
    def label(self) -> str:
        """The room-criterion label in the ``RC-NN(A)`` form of clause D.3.5.

        Clause D.3.5 admits N, R, H or RV as the tag; when both the rumble
        and hiss deviations fire, this library labels the spectrum with the
        combined ``RH`` extension (see :attr:`classification`).
        """
        return f"RC-{self.rating}({self.classification})"

    @property
    def out_of_family(self) -> bool:
        """True when the rating falls outside the tabulated RC-25 to RC-50 family.

        Table D.1 tabulates the RC Mark II curves for ratings 25 through 50
        and clause D.3.5 defines the ``RC-NN(A)`` label for integers in that
        range. Outside it the reference curve is generated by the same
        -5 dB/octave rule of Annex D, but the designation extrapolates beyond
        the standard's tabulated family.
        """
        return not _RC_FAMILY_MIN <= self.rating <= _RC_FAMILY_MAX

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the measured spectrum against the reference RC Mark II curve.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_room_criterion

        check_language(language)
        return plot_room_criterion(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a Room Criteria Mark II (RC) assessment fiche to a PDF.

        Writes a one-page room-noise assessment report (ANSI/ASA S12.2-2019,
        Annex D): the standard-basis line, an optional metadata header block,
        the measured octave-band levels beside the measured spectrum plotted
        against the reference RC Mark II curve (the result's own :meth:`plot`),
        the boxed ``RC-nn(tag)`` rating with its mid-frequency average and
        spectral-quality tag, an optional verdict row and a footer with the
        fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            assessment fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the maximum acceptable RC rating (a lower
            rating is quieter, so the room passes at or below it).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table adds the reference RC Mark II
            curve and the measured deviation from it.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            raise ValueError(msg)
        from .._report.ansi_s12_2 import render_rc_report

        return render_rc_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def nc_curve(index: float) -> np.ndarray:
    """Octave-band levels of a Noise Criteria curve (ANSI/ASA S12.2-2019 Table 1).

    :param index: The NC designation. Integer designations from 15 to 70 in
        steps of five return the tabulated curve; intermediate values are
        linearly interpolated band by band.
    :return: The 10-band curve levels, in dB, aligned with :data:`OCTAVE_BANDS`.
    :raises ValueError: if ``index`` is outside the tabulated range.
    """
    if index < NC_INDICES[0] or index > NC_INDICES[-1]:
        msg = (
            f"NC index {index} is outside the tabulated range "
            f"[{NC_INDICES[0]:.0f}, {NC_INDICES[-1]:.0f}]."
        )
        raise ValueError(msg)
    return np.array(
        [np.interp(index, NC_INDICES, NC_CURVES[:, k]) for k in range(_N_BANDS)]
    )


def rc_curve(index: float) -> np.ndarray:
    """Octave-band levels of a Room Criteria Mark II curve (Annex D, Table D.1).

    The curve has a constant slope of -5 dB/octave keyed to its value at
    1000 Hz; the 31.5 Hz level does not drop below 55 dB and the 16 Hz level
    equals the 31.5 Hz level.

    :param index: The RC designation (value at 1000 Hz).
    :return: The 10-band curve levels, in dB, aligned with :data:`OCTAVE_BANDS`.
    """
    curve = index - 5.0 * _RC_OCTAVE_STEPS
    low = max(index + 25.0, _RC_LOW_FREQUENCY_FLOOR)
    curve[1] = low  # 31.5 Hz floor
    curve[0] = low  # 16 Hz equals 31.5 Hz
    return curve


def _criterion_curve_at(
    family: str, index: float, frequencies: ArrayLike
) -> np.ndarray:
    """An NC or RC curve resampled onto an arbitrary octave-band grid.

    Design sheets compare a received spectrum band by band with the criterion
    curve, so the curve has to be read at the analysis bands rather than at the
    ten bands of Table 1. Each requested frequency takes the value of the
    nearest Table 1 band on a logarithmic frequency scale.

    :param family: ``"NC"`` or ``"RC"``.
    :param index: The curve designation (e.g. ``45`` for NC 45).
    :param frequencies: Band centre frequencies to sample at, in hertz.
    :return: The curve levels at ``frequencies``, in dB.
    """
    curve = nc_curve(index) if family == "NC" else rc_curve(index)
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    return np.array(
        [curve[int(np.argmin(np.abs(np.log2(OCTAVE_BANDS / f))))] for f in freqs]
    )


def _align_levels(
    levels: ArrayLike, frequencies: ArrayLike | None, owner: str
) -> np.ndarray:
    """Validate levels and align them to :data:`OCTAVE_BANDS`.

    :param levels: Octave-band sound pressure levels, in dB.
    :param frequencies: Optional band centre frequencies matching ``levels``.
    :param owner: Name of the entry point the caller typed, for the messages.
    :return: The levels on the ten :data:`OCTAVE_BANDS`, absent bands NaN.
    :raises ValueError: for malformed inputs or unknown band frequencies.
    """
    lv = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if lv.ndim != 1:
        msg = "levels must be a 1-D vector of octave-band levels."
        raise ValueError(msg)
    if frequencies is None:
        if lv.size != _N_BANDS:
            msg = (
                f"levels must have {_N_BANDS} octave-band values (16 Hz - "
                f"8000 Hz) when frequencies are not given; got {lv.size}."
            )
            raise ValueError(msg)
        return lv
    fr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    require_equal_shapes(owner, {"levels": lv.shape, "frequencies": fr.shape}, "band")
    aligned = np.full(_N_BANDS, np.nan)
    for f, level in zip(fr, lv, strict=True):
        matches = np.isclose(OCTAVE_BANDS, f, rtol=0.03)
        if not matches.any():
            msg = (
                f"frequency {f} Hz is not one of the ANSI S12.2 octave bands "
                f"(16 Hz - 8000 Hz)."
            )
            raise ValueError(msg)
        aligned[np.argmax(matches)] = level
    return aligned


def noise_criterion(
    levels: ArrayLike, frequencies: ArrayLike | None = None
) -> NCResult:
    """Noise Criteria (NC) rating of a spectrum (ANSI/ASA S12.2-2019, 5.2.2).

    Follows the standard's two-step rating procedure. First the speech
    interference level SIL (clause 3.2: the average of the 500, 1000, 2000
    and 4000 Hz levels) selects the NC-(SIL) curve; when no octave band
    exceeds that curve, the spectrum is designated NC-(SIL). Otherwise the
    rating is determined by the tangency method (clause 5.2.3): for each band
    the NC index whose Table 1 curve passes through the measured level is
    found by interpolation, the rating is the maximum over bands and the
    governing band is where that maximum occurs. The tangency rating is
    always evaluated and kept on ``tangency_rating``; when the four SIL bands
    are not all supplied, the tangency rating is the designation.

    A spectrum outside the Table 1 family has no NC rating: when a band
    exceeds the NC-70 curve the result is flagged ``out_of_range="above"``
    (with the governing band at the maximum exceedance over NC-70), and when
    every measured band lies below the NC-15 curve it is flagged
    ``out_of_range="below"``; in both cases ``rating`` is NaN and ``label``
    reads ``">NC-70"`` / ``"<NC-15"``.

    :param levels: Octave-band sound pressure levels, in dB. Without
        ``frequencies`` this must be the 10 bands from 16 Hz to 8000 Hz.
    :param frequencies: Optional band centre frequencies, in hertz, matching
        ``levels``; a subset of the ANSI S12.2 octave bands may be supplied.
    :return: An :class:`NCResult` with the designation and its ``.plot()``.
    :raises ValueError: for malformed inputs or unknown band frequencies.
    """
    aligned = _align_levels(levels, frequencies, "noise_criterion")
    valid = ~np.isnan(aligned)
    if not valid.any():
        msg = "no valid octave-band levels were supplied."
        raise ValueError(msg)

    sil_levels = aligned[_SIL_BANDS]
    sil = float(np.mean(sil_levels)) if not np.isnan(sil_levels).any() else float("nan")

    per_band = np.full(_N_BANDS, np.nan)
    above = np.zeros(_N_BANDS, dtype=bool)
    for k in np.flatnonzero(valid):
        if aligned[k] > NC_CURVES[-1, k]:
            above[k] = True  # louder than the NC-70 curve in this band.
        elif aligned[k] >= NC_CURVES[0, k]:
            per_band[k] = np.interp(aligned[k], NC_CURVES[:, k], NC_INDICES)
        # A band below the NC-15 curve touches no curve and stays NaN.

    def build(
        rating: float,
        governing_frequency: float,
        tangency_rating: float,
        method: str,
        out_of_range: str | None,
    ) -> NCResult:
        return NCResult(
            rating=rating,
            governing_frequency=governing_frequency,
            frequencies=OCTAVE_BANDS.copy(),
            levels=aligned,
            sil=sil,
            tangency_rating=tangency_rating,
            method=method,
            out_of_range=out_of_range,
        )

    if above.any():
        # Above the family: no NC rating exists; the governing band is the
        # one with the maximum exceedance over the NC-70 curve.
        exceedance = np.where(above, aligned - NC_CURVES[-1], -np.inf)
        governing = int(np.argmax(exceedance))
        return build(
            float("nan"),
            float(OCTAVE_BANDS[governing]),
            float("nan"),
            "tangency",
            "above",
        )
    if not np.isfinite(per_band).any():
        # Every measured band lies below the NC-15 curve.
        return build(float("nan"), float("nan"), float("nan"), "tangency", "below")

    governing = int(np.nanargmax(per_band))
    tangency = float(per_band[governing])

    if math.isfinite(sil):
        nc_sil = float(np.rint(sil))
        if NC_INDICES[0] <= nc_sil <= NC_INDICES[-1]:
            curve = nc_curve(nc_sil)
            if not np.any(aligned[valid] > curve[valid]):
                # Clause 5.2.2: no band exceeds the NC-(SIL) curve, so the
                # spectrum is designated NC-(SIL) with no governing band.
                return build(nc_sil, float("nan"), tangency, "SIL", None)
    return build(tangency, float(OCTAVE_BANDS[governing]), tangency, "tangency", None)


def room_criterion(levels: ArrayLike, frequencies: ArrayLike | None = None) -> RCResult:
    """Room Criteria Mark II rating (ANSI/ASA S12.2-2019, Annex D).

    The numerical rating is the mid-frequency average ``LMF`` of the 500,
    1000 and 2000 Hz levels rounded to the nearest decibel (clause D.4). The
    spectral tag is neutral (``"N"``) when the levels at and below 500 Hz do
    not exceed the reference RC curve by more than 5 dB and the levels at and
    above 1000 Hz do not exceed it by more than 3 dB; rumble (``"R"``) when a
    low band exceeds by more than 5 dB; hiss (``"H"``) when a high band
    exceeds by more than 3 dB (clause D.3). When both deviations fire, the
    library tags the spectrum ``"RH"``, a diagnostic extension beyond the
    clause D.3.5 letters (see :class:`RCResult`).

    Clause D.4 evaluates a spectrum that includes at least the octave bands
    from 31.5 Hz through 4000 Hz; when any of those bands is missing a
    :class:`UserWarning` is emitted, since the absent bands are silently
    skipped by the spectral-tag deviation tests. A rating outside the
    tabulated RC-25 to RC-50 family is flagged by
    :attr:`RCResult.out_of_family`.

    :param levels: Octave-band sound pressure levels, in dB. Without
        ``frequencies`` this must be the 10 bands from 16 Hz to 8000 Hz.
    :param frequencies: Optional band centre frequencies, in hertz, matching
        ``levels``.
    :return: An :class:`RCResult` with the rating, tag and its ``.plot()``.
    :raises ValueError: if the 500/1000/2000 Hz mid-frequency bands are absent.
    """
    aligned = _align_levels(levels, frequencies, "room_criterion")
    mid = aligned[5:8]  # 500, 1000, 2000 Hz.
    if np.isnan(mid).any():
        msg = (
            "the 500, 1000 and 2000 Hz octave bands are required to compute "
            "the mid-frequency average (RC rating)."
        )
        raise ValueError(msg)
    required = aligned[_RC_REQUIRED_BANDS]
    if np.isnan(required).any():
        missing = OCTAVE_BANDS[_RC_REQUIRED_BANDS][np.isnan(required)]
        warnings.warn(
            "ANSI/ASA S12.2-2019, clause D.4 rates a spectrum that includes "
            "at least the 31.5 Hz to 4000 Hz octave bands; the missing bands "
            f"({', '.join(f'{f:g}' for f in missing)} Hz) are skipped by the "
            "spectral-tag deviation tests.",
            stacklevel=2,
        )
    lmf = float(np.mean(mid))
    rating = int(np.rint(lmf))
    reference = rc_curve(float(rating))
    deviation = aligned - reference

    low = slice(0, 6)  # 16 - 500 Hz (at and below 500 Hz).
    high = slice(6, _N_BANDS)  # 1000 - 8000 Hz (at and above 1000 Hz).
    low_dev = deviation[low][~np.isnan(deviation[low])]
    high_dev = deviation[high][~np.isnan(deviation[high])]
    rumble = low_dev.size > 0 and np.max(low_dev) > _RC_RUMBLE_LIMIT_DB
    hiss = high_dev.size > 0 and np.max(high_dev) > _RC_HISS_LIMIT_DB
    tag = ("R" if rumble else "") + ("H" if hiss else "")
    classification = tag or "N"

    return RCResult(
        rating=rating,
        lmf=lmf,
        classification=classification,
        reference_curve=reference,
        frequencies=OCTAVE_BANDS.copy(),
        levels=aligned,
    )
