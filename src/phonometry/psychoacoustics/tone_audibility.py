#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Objective audibility of tones in noise -- engineering method (ISO/PAS 20065:2016).

ISO/PAS 20065 is the detailed engineering method that ISO 1996-2:2017 defers to
for the audibility of prominent tones; the simplified 2007/2009 Annex C method
lives in :mod:`phonometry.environment.assessment.measurement`. The audibility of a tone
is the amount, in decibels, by which its tone level rises above the masking
threshold of the surrounding noise.

**Critical band about the tone (Clause 5.2).** The width of the critical band
around a tone of frequency ``fT`` is
:math:`\Delta f_c = 25.0 + 75.0 (1.0 + 1.4 (f_T/1000)^2)^{0.69}` Hz
(Formula (2)). Assuming a geometric placement of the corner frequencies,
:math:`f_T = \sqrt{f_1 f_2}` (Formula (3)),
:math:`f_1 = -\Delta f_c/2 + \sqrt{\Delta f_c^2 + 4 f_T^2}/2` (Formula (4))
and :math:`f_2 = f_1 + \Delta f_c` (Formula (5)).

**Audibility of a single tone (Clause 5.3).** From the mean narrow-band level
``LS`` of the masking noise (Formula (6)) the critical-band level of the
masking noise is :math:`L_G = L_S + 10 \log_{10}(\Delta f_c/\Delta f)`
(Formula (12), ``Δf`` the line spacing). The masking index is
:math:`a_v = -2 - \log_{10}[1 + (f/502)^{2.5}]` dB (Formula (13)) and the
audibility of a tone of level ``LT`` (Formula (8)) is
:math:`\Delta L = L_T - L_G - a_v` dB (Formula (14)). A tone is present when
:math:`\Delta L > 0`.

**Decisive and mean audibility (Clauses 5.3.8/5.3.9).** The decisive audibility
of one narrow-band spectrum is the largest tone audibility in it (Step 4). Over
``J`` staggered spectra the mean audibility is the energy mean
:math:`\Delta L = 10 \log_{10}[(1/J) \sum_j 10^{\Delta L_j/10}]` dB (Formula (20));
a spectrum in which no tone is found contributes :math:`\Delta L_j = -10` dB
(Formula (21)).

**From a critical-band spectrum.** :func:`mean_narrowband_level` determines
``LS`` from the lines of the critical band by the iterative procedure of
Formula (6)/Annex D (energy average, dropping any line more than 6 dB above the
running mean, with the −1.76 dB Hanning bandwidth correction), and
:func:`tone_level` sums the tonal lines contiguous with the peak for ``LT``
(Formula (8)). Both reproduce the ISO/PAS 20065 Annex E worked example
(:math:`L_S = 49.22` dB, :math:`L_T = 67.96` dB for the 137.3 Hz tone) and
are confirmed against the parent standard DIN 45681:2005-03.

**Whole-spectrum detection.** :func:`analyze_spectrum` runs the full front-end
over a spectrum (mean narrow-band level per line, peak detection (Clause 5.3.8
Step 1), tone level, the distinctness test (Clause 5.3.4) and audibility) and
returns the distinct, audible tones. It then applies Step 3: tones sharing a
critical band have their tone levels energy-summed (Formula (17), via
:func:`combined_tone_level`, shared lines counted once) into an "FG" entry
rated at the most audible member, unless the exactly-two-tones-below-1000-Hz
exception (Formulae (18)/(19)) keeps them separate. On the Annex E example
this recovers the three tones, their combined tone level
:math:`L_T = 72.15` dB and the decisive FG audibility
:math:`\Delta L = 9.18` dB. A decisive audibility
reproduced exactly needs the *complete* narrow-band spectrum: a spectrum
truncated to one critical band mis-estimates the mean narrow-band level of
tones near its edges.

**Two tones below 1000 Hz.** When *exactly two* tones share a critical band and
both lie below 1000 Hz, the ear can still resolve them if their spacing exceeds
:math:`f_D = 21 \cdot 10^{1.2 |\log_{10}(f_T/212)|^{1.8}}` Hz
(Formulae (18)/(19)); they are then
rated separately instead of combined. :func:`two_tone_separation_frequency` and
:func:`resolve_tones_separately` implement this branch (Clause 5.3.8), which no
ISO/PAS 20065 worked example exercises; it is verified against the DIN 45681
Annex J reference program rather than a numeric oracle.

**Uncertainty (Clauses 5.4/6).** :func:`audibility_uncertainty` propagates the
uniform 3 dB narrow-band level uncertainty through the audibility chain to the
extended uncertainty ``U`` (90 % bilateral coverage), and
:func:`mean_audibility_uncertainty` combines the per-spectrum values through
the Formula (20) mean. Clause 6: when fewer than 12 spectra have been
averaged, the extended uncertainty **shall** be taken into consideration.
:func:`analyze_spectrum` reports the per-tone ``U`` on its result.

**A-weighting.** Clause 5.3.2: unweighted narrow-band spectra "shall" be
A-weighted per IEC 61672-1 before the analysis. This module is
weighting-agnostic: pass A-weighted levels (the Annex E oracles are
A-weighted); it does not apply the weighting itself.

**Application frequency range.** The functions accept any positive tone
frequency, but the standards state narrower ranges: DIN 45681:2005-03 (5.3.2)
restricts the method to :math:`f_T \ge 90` Hz and the ISO/PAS 20065 scope
starts at 50 Hz; the two-tone separation frequency (Formula (19)) is printed
for :math:`f_T < 1000` Hz (with a lower bound of 88 Hz in the DIN print,
50 Hz in the ISO one). Results outside these ranges are extrapolations.

**Distinctness edge steepness (DIN-vs-ISO print difference).** The 5.3.4
edge-steepness test follows the DIN 45681 :math:`f_T/\sqrt{2}`-on-both-edges
reading, matching its executable Annex J reference program; the ISO/PAS
20065 print shows asymmetric formulas that contradict it (see
``_is_distinct`` and docs/ERRATA.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from .._report.metadata import ReportMetadata

#: Constant term of the critical bandwidth Δfc (Formula (2)), in Hz.
_DFC_CONSTANT = 25.0
#: Frequency-dependent scale of the critical bandwidth (Formula (2)), in Hz.
_DFC_SCALE = 75.0
#: Inner coefficient of the critical bandwidth (Formula (2)).
_DFC_INNER = 1.4
#: Exponent of the critical bandwidth (Formula (2)).
_DFC_EXPONENT = 0.69
#: Reference frequency of the critical bandwidth (Formula (2)), in Hz.
_DFC_REFERENCE = 1000.0

#: Reference frequency of the masking index av (Formula (13)), in Hz.
_MASKING_REFERENCE_FREQUENCY = 502.0
#: Exponent of the masking index av (Formula (13)).
_MASKING_EXPONENT = 2.5
#: Constant offset (dB) of the masking index av (Formula (13)).
_MASKING_OFFSET = 2.0

#: Effective-bandwidth factor Δfe/Δf for a Hanning window (Annex A): 1.5.
HANNING_BANDWIDTH_FACTOR = 1.5

#: Audibility (dB) assigned to a spectrum with no tone found (Formula (21)).
NO_TONE_AUDIBILITY = -10.0


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return scalar


def _finite(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"'{name}' must be finite.")
    return scalar


# --------------------------------------------------------------------------- #
# Critical band about the tone (Clause 5.2)
# --------------------------------------------------------------------------- #
def critical_bandwidth_engineering(tone_frequency: float) -> float:
    r"""Width ``Δfc`` of the critical band about a tone (Formula (2)).

    :math:`\Delta f_c = 25.0 + 75.0 (1.0 + 1.4 (f_T/1000)^2)^{0.69}` Hz.
    This is the
    continuous ISO/PAS 20065 engineering-method bandwidth, distinct from the
    stepped ISO 1996-2 Annex C :func:`~phonometry.environment.assessment.measurement.
    critical_bandwidth` (100 Hz / 20 %).

    :param tone_frequency: Tone frequency ``fT``, in Hz.
    :return: Critical bandwidth ``Δfc``, in Hz.
    :raises ValueError: If ``tone_frequency`` is not positive/finite.
    """
    ft = _positive(tone_frequency, "tone_frequency")
    inner = 1.0 + _DFC_INNER * (ft / _DFC_REFERENCE) ** 2
    return float(_DFC_CONSTANT + _DFC_SCALE * inner**_DFC_EXPONENT)


def critical_band_corners(tone_frequency: float) -> tuple[float, float]:
    r"""Lower/upper corner frequencies of the critical band (Formulae (3)-(5)).

    With a geometric placement of the corners about the tone,
    :math:`f_1 = -\Delta f_c/2 + \sqrt{\Delta f_c^2 + 4 f_T^2}/2` and
    :math:`f_2 = f_1 + \Delta f_c`, so that :math:`\sqrt{f_1 f_2} = f_T` and
    :math:`f_2 - f_1 = \Delta f_c`.

    :param tone_frequency: Tone frequency ``fT``, in Hz.
    :return: ``(f1, f2)`` corner frequencies, in Hz.
    :raises ValueError: If ``tone_frequency`` is not positive/finite.
    """
    ft = _positive(tone_frequency, "tone_frequency")
    dfc = critical_bandwidth_engineering(ft)
    # Rationalised form of f1 = −Δfc/2 + √(Δfc²+4·fT²)/2: numerically stable at
    # low frequencies where Δfc ≫ fT and the direct form loses precision to the
    # subtraction of two nearly equal terms.
    f1 = 2.0 * ft**2 / (np.sqrt(dfc**2 + 4.0 * ft**2) + dfc)
    return float(f1), float(f1 + dfc)


# --------------------------------------------------------------------------- #
# Audibility of a single tone (Clause 5.3)
# --------------------------------------------------------------------------- #
def masking_index(frequency: float) -> float:
    r"""Masking index ``av`` of the auditory system (Formula (13)).

    :math:`a_v = -2 - \log_{10}[1 + (f/502)^{2.5}]` dB. The value is negative and
    grows more
    negative with frequency (see Annex C).

    :param frequency: Frequency ``f``, in Hz.
    :return: Masking index ``av``, in dB.
    :raises ValueError: If ``frequency`` is not positive/finite.
    """
    f = _positive(frequency, "frequency")
    return float(
        -_MASKING_OFFSET
        - np.log10(1.0 + (f / _MASKING_REFERENCE_FREQUENCY) ** _MASKING_EXPONENT)
    )


def critical_band_level(
    mean_narrowband_level: float, tone_frequency: float, line_spacing: float
) -> float:
    r"""Critical-band level ``LG`` of the masking noise (Formula (12)).

    :math:`L_G = L_S + 10 \log_{10}(\Delta f_c/\Delta f)` dB, spreading the mean
    narrow-band level ``LS``
    over the critical bandwidth ``Δfc`` relative to the line spacing ``Δf``.

    :param mean_narrowband_level: Mean narrow-band level ``LS``, in dB
        (Formula (6)).
    :param tone_frequency: Tone frequency ``fT``, in Hz.
    :param line_spacing: Line spacing (frequency resolution) ``Δf``, in Hz.
    :return: Critical-band level ``LG``, in dB.
    :raises ValueError: If the level is not finite or a frequency is not
        positive/finite.
    """
    ls = _finite(mean_narrowband_level, "mean_narrowband_level")
    df = _positive(line_spacing, "line_spacing")
    dfc = critical_bandwidth_engineering(tone_frequency)
    return float(ls + 10.0 * np.log10(dfc / df))


def audibility_from_levels(
    tone_level: float, critical_band_level: float, masking_index: float
) -> float:
    r"""Audibility ``ΔL`` from the levels and masking index (Formula (14)).

    :math:`\Delta L = L_T - L_G - a_v` dB.

    :param tone_level: Tone level ``LT``, in dB (Formula (8)).
    :param critical_band_level: Critical-band level ``LG`` of the masking
        noise, in dB (Formula (12)).
    :param masking_index: Masking index ``av``, in dB (Formula (13)).
    :return: Audibility ``ΔL``, in dB (dB above the masking threshold).
    :raises ValueError: If any argument is not finite.
    """
    lt = _finite(tone_level, "tone_level")
    lg = _finite(critical_band_level, "critical_band_level")
    av = _finite(masking_index, "masking_index")
    return float(lt - lg - av)


def energy_sum_level(
    line_levels: ArrayLike,
    *,
    effective_bandwidth_factor: float = HANNING_BANDWIDTH_FACTOR,
) -> float:
    r"""Energy sum of the tonal spectral lines with window correction (Formulae (7)/(8)).

    For a single line (:math:`K = 1`) the tone level is that line's level with
    *no* bandwidth correction, :math:`L_T = L_1` (Formula (7)). For
    :math:`K > 1` lines,
    :math:`L_T = 10 \log_{10}[\sum_i 10^{L_i/10}] + 10 \log_{10}(\Delta f/\Delta f_e)` dB
    (Formula (8)); the window correction
    :math:`10 \log_{10}(\Delta f/\Delta f_e)` is ``−1.76 dB`` for a Hanning window
    (:math:`\Delta f_e = 1.5 \Delta f`, Annex A) and ``0 dB`` for a
    rectangular window
    (:math:`\Delta f_e = \Delta f`). The DIN 45681:2005-03 Annex J reference
    program applies the same split
    (``If l = 1 Then LT = 10*Log(LT)/Log(10)`` with no ``−1.76``).
    (The mean narrow-band level ``LS`` of Formula (6) is the analogous *energy
    average* and always carries the correction; :func:`mean_narrowband_level`
    and :func:`tone_level` derive ``LS`` and ``LT`` from a critical-band
    spectrum.)

    :param line_levels: Narrow-band levels ``Li`` of the tonal lines to sum,
        in dB.
    :param effective_bandwidth_factor: :math:`\Delta f_e/\Delta f`; 1.5
        for a Hanning window
        (the default), 1.0 for a rectangular window. Ignored for a single
        line (Formula (7) applies no correction at :math:`K = 1`).
    :return: Corrected energy-sum level, in dB.
    :raises ValueError: If ``line_levels`` is empty/non-finite or the factor is
        not positive/finite.
    """
    levels = np.asarray(line_levels, dtype=np.float64)
    if levels.size == 0:
        raise ValueError("'line_levels' must contain at least one line.")
    if not np.all(np.isfinite(levels)):
        raise ValueError("'line_levels' must be finite.")
    factor = _positive(effective_bandwidth_factor, "effective_bandwidth_factor")
    if levels.size == 1:
        # Formula (7): a single-line tone takes its level unchanged; the
        # bandwidth correction applies only to sums of K > 1 lines.
        return float(levels[0])
    total = np.sum(10.0 ** (levels / 10.0))
    return float(10.0 * np.log10(total) - 10.0 * np.log10(factor))


# --------------------------------------------------------------------------- #
# Narrow-band front-end: LS and LT from a critical-band spectrum
# (Clauses 5.3.2/5.3.3, Annex D)
# --------------------------------------------------------------------------- #
def _validate_spectrum(
    levels: ArrayLike, frequencies: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce and validate a narrow-band ``(levels, frequencies)`` spectrum."""
    lev = np.asarray(levels, dtype=np.float64)
    freq = np.asarray(frequencies, dtype=np.float64)
    if lev.ndim != 1 or freq.ndim != 1:
        raise ValueError("'levels' and 'frequencies' must be one-dimensional.")
    if lev.size != freq.size:
        raise ValueError("'levels' and 'frequencies' must share their length.")
    if lev.size == 0:
        raise ValueError("'levels' and 'frequencies' must not be empty.")
    if not (np.all(np.isfinite(lev)) and np.all(np.isfinite(freq))):
        raise ValueError("'levels' and 'frequencies' must be finite.")
    if not np.all(np.diff(freq) > 0.0):
        raise ValueError("'frequencies' must be strictly increasing.")
    if freq[0] <= 0.0:
        raise ValueError("'frequencies' must be positive.")
    return lev, freq


#: Line-exclusion / tone-energy margin above the mean narrow-band level (dB).
_TONE_MARGIN = 6.0
#: Convergence tolerance of the iterative mean narrow-band level (dB).
_LS_TOLERANCE = 0.005
#: Minimum lines each side of the tone required to keep iterating (Annex D).
_LS_MIN_SIDE = 5
#: Level drop (dB) below the tone peak that bounds its tonal lines (Clause 5.3.3).
_TONE_PEAK_DROP = 10.0


def mean_narrowband_level(
    levels: ArrayLike,
    frequencies: ArrayLike,
    tone_frequency: float,
    *,
    effective_bandwidth_factor: float = HANNING_BANDWIDTH_FACTOR,
) -> float:
    r"""Mean narrow-band level ``LS`` of the masking noise (Formula (6), Annex D).

    .. math::

       L_S = 10 \log_{10}\!\left[ \frac{1}{M} \sum_i 10^{L_i/10} \right]
       + 10 \log_{10}\frac{\Delta f}{\Delta f_e}

    in dB, determined iteratively over the lines of the critical band about
    ``tone_frequency`` (Formulae (2)-(5) give the band). The line at the tone
    frequency is excluded; the average then
    drops any line more than ``6 dB`` above the current ``LS`` and repeats until
    ``LS`` is stable within ``±0.005 dB`` or fewer than five lines remain on
    either side of the tone (Annex D). The window correction
    :math:`10 \log_{10}(\Delta f/\Delta f_e)` is
    ``−1.76 dB`` for the recommended Hanning window
    (:math:`\Delta f_e = 1.5 \Delta f`).

    :param levels: Narrow-band levels ``Li`` of the spectrum, in dB.
    :param frequencies: The line frequencies, in Hz (strictly increasing).
    :param tone_frequency: Tone frequency ``fT``, in Hz.
    :param effective_bandwidth_factor: :math:`\Delta f_e/\Delta f`; 1.5
        for a Hanning window
        (the default), 1.0 for a rectangular window.
    :return: Mean narrow-band level ``LS``, in dB.
    :raises ValueError: If the spectrum is invalid, the factor is not
        positive/finite, or no lines fall in the critical band.
    """
    ls, _ = _mean_narrowband_level_lines(
        levels, frequencies, tone_frequency,
        effective_bandwidth_factor=effective_bandwidth_factor,
    )
    return ls


def _mean_narrowband_level_lines(
    levels: ArrayLike,
    frequencies: ArrayLike,
    tone_frequency: float,
    *,
    effective_bandwidth_factor: float = HANNING_BANDWIDTH_FACTOR,
) -> tuple[float, list[int]]:
    """``LS`` plus the indices of the lines kept by the final iteration.

    The kept-line set is the ``M`` noise lines that Clause 6 uses in the
    uncertainty of the audibility (see :func:`audibility_uncertainty`).
    """
    lev, freq = _validate_spectrum(levels, frequencies)
    ft = _positive(tone_frequency, "tone_frequency")
    factor = _positive(effective_bandwidth_factor, "effective_bandwidth_factor")
    correction = 10.0 * np.log10(factor)

    f1, f2 = critical_band_corners(ft)
    under = int(np.argmin(np.abs(freq - ft)))
    band = [
        i for i in range(lev.size) if f1 <= freq[i] <= f2 and i != under
    ]
    if not band:
        raise ValueError("No spectral lines fall in the critical band.")

    def _ls(indices: list[int]) -> float:
        return float(
            10.0 * np.log10(np.mean(10.0 ** (lev[indices] / 10.0))) - correction
        )

    kept = band
    ls = _ls(kept)
    for _ in range(200):
        pruned = [i for i in kept if lev[i] <= ls + _TONE_MARGIN]
        left = sum(1 for i in pruned if i < under)
        right = sum(1 for i in pruned if i > under)
        if left < _LS_MIN_SIDE or right < _LS_MIN_SIDE or pruned == kept:
            break
        kept = pruned
        new_ls = _ls(kept)
        if abs(new_ls - ls) <= _LS_TOLERANCE:
            ls = new_ls
            break
        ls = new_ls
    return ls, kept


def tone_level(
    levels: ArrayLike,
    frequencies: ArrayLike,
    tone_frequency: float,
    mean_narrowband_level: float,
    *,
    effective_bandwidth_factor: float = HANNING_BANDWIDTH_FACTOR,
) -> float:
    r"""Tone level ``LT`` from the tonal lines about a tone (Formula (8)).

    The tone energy is carried by the run of lines contiguous with the peak at
    ``tone_frequency`` whose level stays above both :math:`L_S + 6` dB and
    :math:`L_{peak} - 10` dB (Clause 5.3.3); their energy sum with the window
    correction is ``LT`` (via :func:`energy_sum_level`). A single-line run
    takes its level unchanged (Formula (7), no bandwidth correction).

    :param levels: Narrow-band levels ``Li`` of the spectrum, in dB.
    :param frequencies: The line frequencies, in Hz (strictly increasing).
    :param tone_frequency: Tone frequency ``fT`` (the peak), in Hz.
    :param mean_narrowband_level: Mean narrow-band level ``LS`` of the masking
        noise, in dB (see :func:`mean_narrowband_level`).
    :param effective_bandwidth_factor: :math:`\Delta f_e/\Delta f`; 1.5
        for a Hanning window
        (the default), 1.0 for a rectangular window.
    :return: Tone level ``LT``, in dB.
    :raises ValueError: If the spectrum is invalid or the levels are not finite.
    """
    lev, freq = _validate_spectrum(levels, frequencies)
    ft = _positive(tone_frequency, "tone_frequency")
    ls = _finite(mean_narrowband_level, "mean_narrowband_level")
    peak = int(np.argmin(np.abs(freq - ft)))
    threshold = max(ls + _TONE_MARGIN, lev[peak] - _TONE_PEAK_DROP)

    low = peak
    while low - 1 >= 0 and lev[low - 1] > threshold:
        low -= 1
    high = peak
    while high + 1 < lev.size and lev[high + 1] > threshold:
        high += 1
    return energy_sum_level(
        lev[low : high + 1], effective_bandwidth_factor=effective_bandwidth_factor
    )


# --------------------------------------------------------------------------- #
# Full-spectrum tone detection (Clause 5.3.8 Steps 1-3, distinctness 5.3.4)
# --------------------------------------------------------------------------- #
#: Max tone bandwidth ΔfR = 26·(1 + 0.001·fT) Hz for distinctness (Formula (9)).
_DISTINCT_BW_CONSTANT = 26.0
_DISTINCT_BW_SLOPE = 0.001
#: Minimum edge steepness of a distinct tone, in dB/octave (Formulae (10)/(11)).
_DISTINCT_EDGE_STEEPNESS = 24.0


def _tone_line_span(lev: NDArray[np.float64], peak: int, ls: float) -> tuple[int, int]:
    """Contiguous run of tonal lines about ``peak`` (levels above LS+6, peak−10)."""
    threshold = max(ls + _TONE_MARGIN, lev[peak] - _TONE_PEAK_DROP)
    low = peak
    while low - 1 >= 0 and lev[low - 1] > threshold:
        low -= 1
    high = peak
    while high + 1 < lev.size and lev[high + 1] > threshold:
        high += 1
    return low, high


def _is_distinct(
    lev: NDArray[np.float64],
    freq: NDArray[np.float64],
    peak: int,
    low: int,
    high: int,
    line_spacing: float,
) -> bool:
    r"""Distinctness test: bandwidth (Formula (9)) and edge steepness (10)/(11)).

    NOTE (DIN-vs-ISO print difference): ISO/PAS 20065:2016 5.3.4 prints
    *asymmetric* edge-steepness formulas, :math:`f_T/2` on the lower edge and
    :math:`f_T` (no divisor) on the upper. DIN 45681:2005-03 prints
    :math:`f_T/\sqrt{2}` on BOTH edges and its executable Annex J reference
    program does the same (``Frequenz(i)/Sqr(2)``). The two cannot both be
    satisfied; this implementation follows the DIN :math:`\sqrt{2}` reading
    (the ISO print is plausibly a typesetting corruption of it, see
    docs/ERRATA.md). Relative to DIN, the ISO print is :math:`\sqrt{2}`
    STRICTER on the lower edge (:math:`1/2 < 1/\sqrt{2}`) and :math:`\sqrt{2}`
    MORE LENIENT on the upper (no divisor at all). Borderline tones with a
    one-sided edge steepness in roughly
    [17, 34] dB/oct flip classification between the two readings.
    """
    n_lines = high - low + 1
    max_bandwidth = _DISTINCT_BW_CONSTANT * (1.0 + _DISTINCT_BW_SLOPE * freq[peak])
    if n_lines * line_spacing > max_bandwidth:
        return False
    ft = freq[peak]
    if low - 1 >= 0:
        lower = ft * (lev[peak] - lev[low - 1]) / (
            np.sqrt(2.0) * (ft - freq[low - 1])
        )
        if lower < _DISTINCT_EDGE_STEEPNESS:
            return False
    if high + 1 < lev.size:
        upper = ft * (lev[peak] - lev[high + 1]) / (
            np.sqrt(2.0) * (freq[high + 1] - ft)
        )
        if upper < _DISTINCT_EDGE_STEEPNESS:
            return False
    return True


def _detect_tones(
    lev: NDArray[np.float64],
    freq: NDArray[np.float64],
    line_spacing: float,
    factor: float,
) -> list[tuple[int, int, int, float]]:
    """Detect distinct tone peaks (Clause 5.3.8 Steps 1-2, 5.3.4).

    Returns ``(peak, low, high, LS)`` records: a local maximum that rises more
    than 6 dB above its mean narrow-band level, whose tonal-line run passes the
    distinctness criteria. The search resumes past each tone's line run so that
    secondary maxima inside a tone are absorbed, not double-counted.
    """
    n = lev.size
    tones: list[tuple[int, int, int, float]] = []

    def _ls_at(index: int) -> float | None:
        """LS at a line, or None if its critical band holds no other line."""
        try:
            return mean_narrowband_level(
                lev, freq, float(freq[index]), effective_bandwidth_factor=factor
            )
        except ValueError:
            return None

    i = 0
    while i < n - 1:
        # Advance off any flank to a local maximum (a tone cannot sit on a slope).
        # ``i > 0`` guards the left neighbour so line 0 is not compared against
        # the wrapped-around last line of the spectrum.
        while i < n - 1 and (
            lev[i + 1] > lev[i] or (i > 0 and lev[i - 1] > lev[i])
        ):
            i += 1
        peak = i
        ls = _ls_at(peak)
        if ls is None:
            i += 1
            continue
        # Extend over the run above LS+6, tracking the strongest line as the peak.
        j = i
        while j < n and lev[j] > ls + _TONE_MARGIN:
            if lev[j] > lev[peak]:
                peak = j
            j += 1
        resume = j
        if lev[peak] > ls + _TONE_MARGIN:
            if peak != i:
                recomputed = _ls_at(peak)
                if recomputed is None:
                    i = max(resume, i + 1)
                    continue
                ls = recomputed
            low, high = _tone_line_span(lev, peak, ls)
            if _is_distinct(lev, freq, peak, low, high, line_spacing):
                tones.append((peak, low, high, ls))
        i = max(resume, i + 1)
    return tones


def _step3_groups(
    tones: list[tuple[float, float, float, float, float, list[int]]],
) -> list[tuple[int, ...]]:
    """Clause 5.3.8 Step 3 clusters of tones sharing a critical band.

    Each tone's own critical band yields a candidate set; the bandwidth grows
    with frequency, so the candidate sets of a chain of tones need not be
    identical even when they overlap. Candidates sharing a member therefore
    merge into one connected component, giving exactly one FG entry per
    cluster. Clusters of exactly two tones below 1000 Hz further apart than
    fD are rated separately (Formulae (18)/(19)) and yield no FG entry.
    """
    clusters: list[set[int]] = []
    for band_tone in tones:
        f1, f2 = critical_band_corners(band_tone[0])
        members = {i for i, t in enumerate(tones) if f1 <= t[0] <= f2}
        if len(members) < 2:
            continue
        for c in [c for c in clusters if c & members]:
            members |= c
        clusters = [c for c in clusters if not (c & members)]
        clusters.append(members)
    groups: list[tuple[int, ...]] = []
    for cluster in clusters:
        members_t = tuple(sorted(cluster))
        if len(members_t) == 2:
            (ta, tb) = (tones[members_t[0]], tones[members_t[1]])
            if resolve_tones_separately(ta[0], tb[0], ta[4], tb[4]):
                continue  # rated separately, no FG entry
        groups.append(members_t)
    return groups


def analyze_spectrum(
    levels: ArrayLike,
    frequencies: ArrayLike,
    line_spacing: float,
    *,
    effective_bandwidth_factor: float = HANNING_BANDWIDTH_FACTOR,
) -> ToneAudibilityResult:
    r"""Detect and rate the audible tones of a narrow-band spectrum (Clause 5.3.8).

    Runs the full front-end: the mean narrow-band level (Formula (6)) per line,
    peak detection (Step 1), the tone level (Formulae (7)/(8)), the distinctness
    test (Clause 5.3.4) and the audibility (Formula (14)). Only distinct tones
    with a positive audibility are returned, bundled by :func:`assess_tones`.

    **Same-band combination (Step 3).** When several audible tones fall in one
    critical band, the clause *requires* their tone levels to be energy-summed
    (Formula (17), shared lines counted once) and the audibility recomputed at
    the frequency of the most audible member, unless *exactly two* tones below
    1000 Hz are spaced further apart than the separation frequency ``fD``
    (Formulae (18)/(19)), in which case they stay rated separately. The result
    therefore contains the individual audible tones *plus* one combined "FG"
    entry per multi-tone critical band, mirroring the DIN 45681 Annex I tables;
    :attr:`ToneAudibilityResult.group_sizes` tells them apart (1 = single tone,
    :math:`N \ge 2` = FG entry combining ``N`` tones). The decisive audibility
    (Step 4) is the maximum over all entries, FG entries included.

    Reproducing a decisive audibility exactly requires the *complete*
    narrow-band spectrum; a spectrum truncated to a single critical band gives
    the wrong mean narrow-band level for tones near its edges.

    :param levels: Narrow-band levels ``Li`` of the spectrum, in dB.
    :param frequencies: The line frequencies, in Hz (strictly increasing).
    :param line_spacing: Line spacing (frequency resolution) ``Δf``, in Hz.
    :param effective_bandwidth_factor: :math:`\Delta f_e/\Delta f`; 1.5
        for a Hanning window
        (the default), 1.0 for a rectangular window.
    :return: A :class:`ToneAudibilityResult` of the detected audible tones and
        their same-band FG combinations.
    :raises ValueError: If the spectrum is invalid or no audible tone is found.
    """
    lev, freq = _validate_spectrum(levels, frequencies)
    df = _positive(line_spacing, "line_spacing")
    factor = _positive(effective_bandwidth_factor, "effective_bandwidth_factor")
    detected = _detect_tones(lev, freq, df, factor)

    tones = []  # (freq, LT, LS, U, dL, low, high) of each audible tone
    for peak, low, high, ls in detected:
        lt = energy_sum_level(
            lev[low : high + 1], effective_bandwidth_factor=factor
        )
        delta = tone_audibility(lt, ls, float(freq[peak]), df)
        if delta > 0.0:
            # Extended uncertainty U (Clause 6) from the K tone lines and the
            # M noise lines of the final Formula (6) iteration.
            _, kept = _mean_narrowband_level_lines(
                lev, freq, float(freq[peak]), effective_bandwidth_factor=factor
            )
            u = audibility_uncertainty(
                lev[low : high + 1], lev[kept], float(freq[peak]), df
            )
            tones.append((float(freq[peak]), lt, ls, u, delta, kept))
    if not tones:
        raise ValueError("No audible tone was detected in the spectrum.")

    # Clause 5.3.8 Step 3: combine the tone levels of tones sharing a critical
    # band (Formula (17)), rated at the most audible member, except exactly
    # two tones below 1000 Hz further apart than fD (Formulae (18)/(19)).
    groups = _step3_groups(tones)

    entries = [(t[0], t[1], t[2], t[3], 1) for t in tones]
    for group in groups:
        anchor = max((tones[i] for i in group), key=lambda t: t[4])
        lt_fg = combined_tone_level(
            lev,
            freq,
            [tones[i][0] for i in group],
            [tones[i][2] for i in group],
            effective_bandwidth_factor=factor,
        )
        # Clause 6 note for summated tones: the N summated tone levels stand
        # in for the K tone-containing lines of the uncertainty.
        u_fg = audibility_uncertainty(
            [tones[i][1] for i in group], lev[anchor[5]], anchor[0], df
        )
        entries.append((anchor[0], lt_fg, anchor[2], u_fg, len(group)))

    result = assess_tones(
        [e[0] for e in entries],
        [e[1] for e in entries],
        [e[2] for e in entries],
        df,
        extended_uncertainties=[e[3] for e in entries],
    )
    return ToneAudibilityResult(
        tone_frequencies=result.tone_frequencies,
        tone_levels=result.tone_levels,
        mean_narrowband_levels=result.mean_narrowband_levels,
        line_spacing=result.line_spacing,
        critical_bandwidths=result.critical_bandwidths,
        lower_corners=result.lower_corners,
        upper_corners=result.upper_corners,
        critical_band_levels=result.critical_band_levels,
        masking_indices=result.masking_indices,
        audibilities=result.audibilities,
        extended_uncertainties=result.extended_uncertainties,
        group_sizes=np.array([e[4] for e in entries], dtype=np.int_),
    )


def combined_tone_level(
    levels: ArrayLike,
    frequencies: ArrayLike,
    tone_frequencies: ArrayLike,
    mean_narrowband_levels: ArrayLike,
    *,
    effective_bandwidth_factor: float = HANNING_BANDWIDTH_FACTOR,
) -> float:
    r"""Combined tone level ``LT`` of several tones in one critical band (Formula (17)).

    :math:`L_{Tm} = 10 \log_{10}[\sum_n 10^{L_{Tm,n}/10}]`, the energy sum of the
    tonal lines of all
    the tones, each spectral line counted at most once. Use it when more than one
    audible tone falls in a critical band (Clause 5.3.8 Step 3); the group is
    then rated at the frequency of its most audible tone.

    :param levels: Narrow-band levels ``Li`` of the spectrum, in dB.
    :param frequencies: The line frequencies, in Hz (strictly increasing).
    :param tone_frequencies: The tone frequencies sharing the critical band, Hz.
    :param mean_narrowband_levels: Each tone's mean narrow-band level ``LS``, dB
        (same length as ``tone_frequencies``).
    :param effective_bandwidth_factor: :math:`\Delta f_e/\Delta f`; 1.5
        for a Hanning window.
    :return: Combined tone level ``LT``, in dB.
    :raises ValueError: If the inputs are invalid or differ in length.
    """
    lev, freq = _validate_spectrum(levels, frequencies)
    fts = np.asarray(tone_frequencies, dtype=np.float64)
    lss = np.asarray(mean_narrowband_levels, dtype=np.float64)
    if fts.ndim != 1 or lss.ndim != 1 or fts.size != lss.size:
        raise ValueError(
            "'tone_frequencies' and 'mean_narrowband_levels' must be "
            "one-dimensional and share their length."
        )
    if fts.size == 0:
        raise ValueError("At least one tone is required.")
    marked: set[int] = set()
    for ft, ls in zip(fts, lss):
        peak = int(np.argmin(np.abs(freq - ft)))
        low, high = _tone_line_span(lev, peak, float(ls))
        marked.update(range(low, high + 1))
    return energy_sum_level(
        lev[sorted(marked)], effective_bandwidth_factor=effective_bandwidth_factor
    )


# --------------------------------------------------------------------------- #
# Separate evaluation of two tones below 1000 Hz (Clause 5.3.8, Formulae (18)/(19))
# --------------------------------------------------------------------------- #
#: Scale factor of the two-tone separation frequency fD (Formula (19)), in Hz.
_SEPARATION_SCALE = 21.0
#: Inner coefficient of the two-tone separation frequency fD (Formula (19)).
_SEPARATION_COEFFICIENT = 1.2
#: Reference frequency of the two-tone separation frequency fD (Formula (19)), Hz.
_SEPARATION_REFERENCE = 212.0
#: Exponent of the two-tone separation frequency fD (Formula (19)).
_SEPARATION_EXPONENT = 1.8
#: Upper frequency of the separate two-tone evaluation (Clause 5.3.8), in Hz.
_TWO_TONE_MAX_FREQUENCY = 1000.0


def two_tone_separation_frequency(tone_frequency: float) -> float:
    r"""Frequency-difference threshold ``fD`` for resolving two tones (Formula (19)).

    :math:`f_D = 21 \cdot 10^{1.2 |\log_{10}(f_T/212)|^{1.8}}` Hz. When *exactly
    two* tones fall in one
    critical band and both lie below 1000 Hz, the human ear can still tell them
    apart (they are then rated *separately* rather than combined into a
    single "FG" tone, Formula (17)) if their frequency difference
    :math:`|f_{T1} - f_{T2}|` (Formula (18)) exceeds this threshold. ``fT`` is
    the frequency
    of the more prominent tone (the larger audibility ``ΔL``). The threshold is
    ``21 Hz`` at :math:`f_T = 212` Hz and grows on either side; Formula (19)
    is stated for :math:`50~\text{Hz} < f_T < 1000~\text{Hz}` (Clause 5.3.8,
    Annex D, Note 3).

    .. note::
        No numeric worked example exercises this branch: the Annex E
        combustion-engine spectrum groups *three* tones in its critical band, so
        the "exactly two tones" rule never fires there. The formula and decision
        rule are implemented clean-room from the ISO/PAS 20065 text and verified
        against the DIN 45681:2005-03 Annex J reference program
        (``fD = 21 * 10 ^ (1.2 * Abs(Log(fT / 212) / Log(10)) ^ 1.8)``), but are
        *not* anchored on a numeric oracle.

    :param tone_frequency: Frequency ``fT`` of the more prominent tone, in Hz.
    :return: Separation-frequency threshold ``fD``, in Hz.
    :raises ValueError: If ``tone_frequency`` is not positive/finite.
    """
    ft = _positive(tone_frequency, "tone_frequency")
    exponent = (
        _SEPARATION_COEFFICIENT
        * abs(np.log10(ft / _SEPARATION_REFERENCE)) ** _SEPARATION_EXPONENT
    )
    return float(_SEPARATION_SCALE * 10.0**exponent)


def resolve_tones_separately(
    tone1_frequency: float,
    tone2_frequency: float,
    audibility1: float,
    audibility2: float,
) -> bool:
    r"""Whether two tones in one critical band are rated separately (Clause 5.3.8).

    Returns ``True`` when two tones sharing a critical band are evaluated on their
    own instead of being combined into a single FG tone (Formula (17)): both tone
    frequencies lie below 1000 Hz *and* their frequency difference
    :math:`|f_{T1} - f_{T2}|` (Formula (18)) exceeds the separation
    frequency ``fD``
    (Formula (19)) evaluated at the more prominent tone (the larger audibility
    ``ΔL``). Otherwise the tones are combined. This mirrors the DIN 45681 Annex J
    reference program (``If l = 2 And fT1 < 1000 And fT2 < 1000`` … ``If |fT1 −
    fT2| > fD Then auflösen``); see :func:`two_tone_separation_frequency` for the
    verification status.

    :param tone1_frequency: Frequency ``fT1`` of the first tone, in Hz.
    :param tone2_frequency: Frequency ``fT2`` of the second tone, in Hz.
    :param audibility1: Audibility ``ΔL1`` of the first tone, in dB (Formula (14)).
    :param audibility2: Audibility ``ΔL2`` of the second tone, in dB (Formula (14)).
        On a tie (:math:`\Delta L_1 = \Delta L_2`) the first tone is taken as
        the more prominent.
    :return: ``True`` if the tones are rated separately, ``False`` if combined.
    :raises ValueError: If a frequency is not positive/finite or an audibility is
        not finite.
    """
    f1 = _positive(tone1_frequency, "tone1_frequency")
    f2 = _positive(tone2_frequency, "tone2_frequency")
    dl1 = _finite(audibility1, "audibility1")
    dl2 = _finite(audibility2, "audibility2")
    if f1 >= _TWO_TONE_MAX_FREQUENCY or f2 >= _TWO_TONE_MAX_FREQUENCY:
        return False
    dominant = f1 if dl1 >= dl2 else f2
    return abs(f1 - f2) > two_tone_separation_frequency(dominant)


# --------------------------------------------------------------------------- #
# Decisive and mean audibility (Clauses 5.3.8/5.3.9)
# --------------------------------------------------------------------------- #
def mean_audibility(decisive_audibilities: ArrayLike) -> float:
    r"""Mean audibility ``ΔL`` over a number of spectra (Formula (20)).

    :math:`\Delta L = 10 \log_{10}[(1/J) \sum_j 10^{\Delta L_j/10}]` dB, the energy
    mean of the decisive
    audibilities ``ΔLj`` of the ``J`` staggered narrow-band spectra. A
    spectrum with no tone found contributes :math:`\Delta L_j = -10` dB
    (Formula (21)); pass that
    value explicitly for such spectra.

    :param decisive_audibilities: Decisive audibilities ``ΔLj`` of the spectra,
        in dB.
    :return: Mean audibility ``ΔL``, in dB.
    :raises ValueError: If the input is empty or non-finite.
    """
    deltas = np.asarray(decisive_audibilities, dtype=np.float64)
    if deltas.size == 0:
        raise ValueError("'decisive_audibilities' must not be empty.")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("'decisive_audibilities' must be finite.")
    return float(10.0 * np.log10(np.mean(10.0 ** (deltas / 10.0))))


# --------------------------------------------------------------------------- #
# Uncertainty of the audibility (Clauses 5.4 and 6)
# --------------------------------------------------------------------------- #
#: Uniform standard uncertainty of every narrow-band level (Clause 6), in dB.
SIGMA_NARROWBAND_LEVEL = 3.0
#: Coverage factor k for a 90 % bilateral confidence interval (Clause 6).
COVERAGE_FACTOR_90 = 1.645


def audibility_uncertainty(
    tone_line_levels: ArrayLike,
    noise_line_levels: ArrayLike,
    tone_frequency: float,
    line_spacing: float,
    *,
    sigma_level_db: float = SIGMA_NARROWBAND_LEVEL,
    coverage_factor: float = COVERAGE_FACTOR_90,
) -> float:
    r"""Extended uncertainty ``U`` of one tone's audibility (Clause 6).

    Gaussian propagation through Formula (14) with a uniform narrow-band
    level uncertainty (Formulae (22)-(27) and (29)):

    .. math::

       \sigma^2 = \left[ \frac{\sum w_T^2}{\left(\sum w_T\right)^2}
       + \frac{\sum w_S^2}{\left(\sum w_S\right)^2} \right] \sigma_L^2
       + \left( 4.34 \, \frac{\Delta f}{\Delta f_c} \right)^2

    with :math:`w = 10^{0.1 L}` over the ``K``
    tone-containing lines and the ``M`` noise lines of the final Formula (6)
    iteration. For an FG group (several tones combined in one critical band,
    Formula (17)) pass the ``N`` summated *tone levels* as
    ``tone_line_levels`` -- that reading reproduces the printed Table E.2 FG
    uncertainty (3.21 dB) where a union of the individual tonal lines does
    not -- and the most audible tone's noise lines.
    :math:`U = k \sigma` with :math:`k = 1.645` (90 % bilateral coverage). No
    uncertainty is assumed for the masking index or the line spacing; the
    critical-bandwidth term uses :math:`\sigma_{\Delta f_c} = \Delta f`
    (Formula (26)). The
    DIN 45681 Annex J reference program computes the same quantity (its
    ``LT_Delta``/``Ls_Delta`` accumulators), differing only in printing the
    third term without the ``df`` factor -- both readings agree to well
    under 0.01 dB on the Annex E example.

    Clause 5.4 / Clause 6: if fewer than 12 spectra have been averaged, the
    extended uncertainty of the (mean) audibility **shall** be taken into
    consideration; see :func:`mean_audibility_uncertainty`.

    :param tone_line_levels: Levels of the ``K`` tone-containing lines, dB.
    :param noise_line_levels: Levels of the ``M`` masking-noise lines kept by
        the final Formula (6) iteration, dB.
    :param tone_frequency: Tone frequency ``fT``, in Hz.
    :param line_spacing: Line spacing ``df``, in Hz.
    :param sigma_level_db: Standard uncertainty of each narrow-band level
        (Clause 6 assumes a uniform 3 dB).
    :param coverage_factor: Coverage factor ``k`` (1.645 for 90 % bilateral).
    :return: Extended uncertainty ``U`` of the audibility, in dB.
    :raises ValueError: If a line set is empty/non-finite or a parameter is
        not positive/finite.
    """
    lt = np.asarray(tone_line_levels, dtype=np.float64)
    ls = np.asarray(noise_line_levels, dtype=np.float64)
    if lt.size == 0 or ls.size == 0:
        raise ValueError("Both line sets must contain at least one line.")
    if not (np.all(np.isfinite(lt)) and np.all(np.isfinite(ls))):
        raise ValueError("Line levels must be finite.")
    ft = _positive(tone_frequency, "tone_frequency")
    df = _positive(line_spacing, "line_spacing")
    sigma = _positive(sigma_level_db, "sigma_level_db")
    k = _positive(coverage_factor, "coverage_factor")

    w_t = 10.0 ** (0.1 * lt)
    w_s = 10.0 ** (0.1 * ls)
    level_term = float(
        np.sum(w_t**2) / np.sum(w_t) ** 2 + np.sum(w_s**2) / np.sum(w_s) ** 2
    )
    dfc = critical_bandwidth_engineering(ft)
    variance = level_term * sigma**2 + (4.34 * df / dfc) ** 2  # Formula (27)
    return float(k * np.sqrt(variance))  # Formula (29)


def mean_audibility_uncertainty(
    decisive_audibilities: ArrayLike,
    extended_uncertainties: ArrayLike,
) -> float:
    r"""Extended uncertainty ``U`` of the mean audibility (Formulae (28)/(29)).

    .. math::

       U = \frac{\sqrt{\sum_j \left( 10^{0.1 \Delta L_j} U_j \right)^2}}
       {\sum_j 10^{0.1 \Delta L_j}}

    the energy-weighted propagation of the per-spectrum extended uncertainties
    ``Uj`` through the Formula (20) mean (the coverage factor cancels, so
    ``Uj`` can be passed directly). Clause 6: if fewer than 12 spectra have
    been averaged this uncertainty **shall** be taken into consideration;
    with 12 averages the standard reports ~+/-1.5 dB as typically achieved.

    :param decisive_audibilities: Decisive audibilities ``dLj`` of the
        spectra, in dB.
    :param extended_uncertainties: Extended uncertainties ``Uj`` of the same
        spectra, in dB (same length).
    :return: Extended uncertainty of the mean audibility, in dB.
    :raises ValueError: If the arrays are empty, non-finite or of different
        lengths.
    """
    deltas = np.asarray(decisive_audibilities, dtype=np.float64)
    uncertainties = np.asarray(extended_uncertainties, dtype=np.float64)
    if deltas.size == 0:
        raise ValueError("'decisive_audibilities' must not be empty.")
    if deltas.shape != uncertainties.shape:
        raise ValueError(
            "'decisive_audibilities' and 'extended_uncertainties' must share "
            "their length."
        )
    if not (np.all(np.isfinite(deltas)) and np.all(np.isfinite(uncertainties))):
        raise ValueError("Inputs must be finite.")
    weights = 10.0 ** (0.1 * deltas)
    return float(
        np.sqrt(np.sum((weights * uncertainties) ** 2)) / np.sum(weights)
    )


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToneAudibilityResult:
    r"""Audibility of the tones of a narrow-band spectrum (ISO/PAS 20065).

    :ivar tone_frequencies: Tone frequencies ``fT``, in Hz.
    :ivar tone_levels: Tone levels ``LT``, in dB (Formula (8)).
    :ivar mean_narrowband_levels: Mean narrow-band levels ``LS`` of the masking
        noise, in dB (Formula (6)).
    :ivar line_spacing: Line spacing (frequency resolution) ``Δf``, in Hz.
    :ivar critical_bandwidths: Critical bandwidths ``Δfc``, in Hz (Formula (2)).
    :ivar lower_corners: Lower corner frequencies ``f1``, in Hz (Formula (4)).
    :ivar upper_corners: Upper corner frequencies ``f2``, in Hz (Formula (5)).
    :ivar critical_band_levels: Critical-band levels ``LG`` of the masking
        noise, in dB (Formula (12)).
    :ivar masking_indices: Masking indices ``av``, in dB (Formula (13)).
    :ivar audibilities: Audibilities ``ΔL``, in dB (Formula (14)).
    :ivar extended_uncertainties: Extended uncertainties ``U`` of the
        audibilities, in dB (Clause 6, 90 % bilateral coverage), or ``None``
        when the per-line levels needed to compute them were not available
        (:func:`assess_tones` from bare levels). Clause 6: **shall** be taken
        into consideration when fewer than 12 spectra have been averaged.
    :ivar group_sizes: Number of tones behind each entry, or ``None`` when the
        Step 3 combination was not performed (:func:`assess_tones` from bare
        levels). ``1`` marks an individual tone; :math:`N \ge 2` marks a
        combined
        "FG" entry whose tone level energy-sums ``N`` tones sharing a critical
        band (Clause 5.3.8 Step 3, Formula (17)), rated at the most audible
        member's frequency.
    """

    tone_frequencies: NDArray[np.float64]
    tone_levels: NDArray[np.float64]
    mean_narrowband_levels: NDArray[np.float64]
    line_spacing: float
    critical_bandwidths: NDArray[np.float64]
    lower_corners: NDArray[np.float64]
    upper_corners: NDArray[np.float64]
    critical_band_levels: NDArray[np.float64]
    masking_indices: NDArray[np.float64]
    audibilities: NDArray[np.float64]
    extended_uncertainties: NDArray[np.float64] | None = None
    group_sizes: NDArray[np.int_] | None = None

    @property
    def audible(self) -> NDArray[np.bool_]:
        r"""Boolean mask of tones that are present, i.e. :math:`\Delta L > 0`
        (Step 2)."""
        return self.audibilities > 0.0

    @property
    def decisive_audibility(self) -> float:
        """Decisive audibility ``ΔLj`` of the spectrum: the largest ``ΔL`` (Step 4)."""
        return float(np.max(self.audibilities))

    @property
    def decisive_frequency(self) -> float:
        """Tone frequency of the decisive (most audible) tone, in Hz."""
        return float(self.tone_frequencies[int(np.argmax(self.audibilities))])

    def plot(
        self,
        ax: Axes | None = None,
        *,
        view: str = "audibility",
        language: str = "en",
        **kwargs: Any,
    ) -> Axes:
        """Plot the assessment, either as audibilities or as levels.

        :param ax: Existing axes to draw on, or ``None`` to create a figure.
        :param view: ``"audibility"`` (default) draws the per-tone audibility
            ``ΔL`` against tone frequency, with the decisive tone highlighted;
            ``"levels"`` draws the tone levels ``Lpt`` above the critical-band
            masking noise ``Lpn`` on a continuous frequency axis, the view the
            ``.report()`` fiche embeds.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the primary artist call.
        :return: The axes.
        :raises ValueError: If ``view`` is not one of the two names above.
        """
        from .._i18n import check_language
        from .._plot.psychoacoustics import (
            plot_tone_audibility,
            plot_tone_audibility_levels,
        )

        if view not in ("audibility", "levels"):
            raise ValueError(
                f"Unknown view {view!r}; use 'audibility' or 'levels'."
            )
        render = plot_tone_audibility if view == "audibility" else plot_tone_audibility_levels
        return render(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a tonal audibility assessment fiche to a PDF.

        Writes a one-page tonal-assessment report following ISO 1996-2:2017
        Annex J (the engineering method of ISO/PAS 20065:2016): the
        standard-basis line, an optional metadata header (source/situation,
        client, measurement position, instrumentation, date, with the analysis
        line spacing ``Δf`` read from the result), a full-width table of the key
        quantities for every detected tone (frequency ``f_T``, tone level
        ``Lpt``, critical-band masking-noise level ``Lpn``, critical bandwidth
        ``Δf_c`` and audibility ``ΔL_ta``) above the level-versus-frequency
        analysis plot with the tones and their critical-band masking noise
        marked, the boxed decisive audibility ``ΔL_ta`` and the derived tonal
        adjustment ``K`` (Table J.1), an optional verdict row and a prominence
        note, and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            assessment fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the maximum acceptable decisive
            audibility ``ΔL_ta`` in dB (a lower audibility passes).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When True, the key-quantity table adds the per-tone
            extended-uncertainty column (when the result carries it).
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
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso1996_tone import render_tone_audibility_report

        return render_tone_audibility_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


# Module named ``tone_audibility`` (distinct from the ISO 1996-2 Annex C
# ``tonal_audibility`` in :mod:`phonometry.environment.assessment.measurement`).


def tone_audibility(
    tone_level: float,
    mean_narrowband_level: float,
    tone_frequency: float,
    line_spacing: float,
) -> float:
    r"""Audibility ``ΔL`` of one tone from its levels (Formulae (12)-(14)).

    Chains the critical-band level ``LG`` (Formula (12)), the masking index
    ``av`` (Formula (13)) and the audibility
    :math:`\Delta L = L_T - L_G - a_v`
    (Formula (14)) for a single tone.

    :param tone_level: Tone level ``LT``, in dB (Formula (8)).
    :param mean_narrowband_level: Mean narrow-band level ``LS`` of the masking
        noise, in dB (Formula (6)).
    :param tone_frequency: Tone frequency ``fT``, in Hz.
    :param line_spacing: Line spacing (frequency resolution) ``Δf``, in Hz.
    :return: Audibility ``ΔL``, in dB.
    :raises ValueError: If the levels are not finite or a frequency/spacing is
        not positive/finite.
    """
    lg = critical_band_level(mean_narrowband_level, tone_frequency, line_spacing)
    av = masking_index(tone_frequency)
    return audibility_from_levels(tone_level, lg, av)


def assess_tones(
    tone_frequencies: ArrayLike,
    tone_levels: ArrayLike,
    mean_narrowband_levels: ArrayLike,
    line_spacing: float,
    *,
    extended_uncertainties: ArrayLike | None = None,
) -> ToneAudibilityResult:
    """Assess the audibility of the tones of a narrow-band spectrum.

    Applies the critical band (Formulae (2)-(5)), critical-band level
    (Formula (12)), masking index (Formula (13)) and audibility (Formula (14))
    to every tone and bundles them into a plottable :class:`ToneAudibilityResult`.

    :param tone_frequencies: Tone frequencies ``fT``, in Hz.
    :param tone_levels: Tone levels ``LT``, in dB (Formula (8)).
    :param mean_narrowband_levels: Mean narrow-band levels ``LS`` of the masking
        noise, in dB (Formula (6)).
    :param line_spacing: Line spacing (frequency resolution) ``Δf``, in Hz.
    :param extended_uncertainties: Optional per-tone extended uncertainties
        ``U``, in dB (see :func:`audibility_uncertainty`; computed
        automatically by :func:`analyze_spectrum`, which has the per-line
        levels this level-based entry point lacks).
    :return: A :class:`ToneAudibilityResult`.
    :raises ValueError: If the arrays are empty, differ in length, or contain
        non-finite/non-positive values.
    """
    freqs = np.asarray(tone_frequencies, dtype=np.float64)
    lt = np.asarray(tone_levels, dtype=np.float64)
    ls = np.asarray(mean_narrowband_levels, dtype=np.float64)
    df = _positive(line_spacing, "line_spacing")
    if freqs.ndim != 1 or lt.ndim != 1 or ls.ndim != 1:
        raise ValueError(
            "'tone_frequencies', 'tone_levels' and 'mean_narrowband_levels' "
            "must be one-dimensional."
        )
    if not (freqs.size == lt.size == ls.size):
        raise ValueError(
            "'tone_frequencies', 'tone_levels' and 'mean_narrowband_levels' "
            "must share their length."
        )
    if freqs.size == 0:
        raise ValueError("At least one tone is required.")
    if not (np.all(np.isfinite(freqs)) and np.all(freqs > 0.0)):
        raise ValueError("'tone_frequencies' must be positive and finite.")
    if not (np.all(np.isfinite(lt)) and np.all(np.isfinite(ls))):
        raise ValueError(
            "'tone_levels' and 'mean_narrowband_levels' must be finite."
        )

    uncertainties: NDArray[np.float64] | None = None
    if extended_uncertainties is not None:
        uncertainties = np.asarray(extended_uncertainties, dtype=np.float64)
        if uncertainties.shape != freqs.shape:
            raise ValueError(
                "'extended_uncertainties' must match 'tone_frequencies' in length."
            )
        if not np.all(np.isfinite(uncertainties)):
            raise ValueError("'extended_uncertainties' must be finite.")

    dfc = np.array([critical_bandwidth_engineering(f) for f in freqs])
    # Corner frequencies (Formulae 4/5) from the already-computed Δfc, in the
    # numerically stable form (see critical_band_corners).
    f1 = 2.0 * freqs**2 / (np.sqrt(dfc**2 + 4.0 * freqs**2) + dfc)
    f2 = f1 + dfc
    lg = ls + 10.0 * np.log10(dfc / df)
    av = np.array([masking_index(f) for f in freqs])
    delta = lt - lg - av
    return ToneAudibilityResult(
        tone_frequencies=freqs,
        tone_levels=lt,
        mean_narrowband_levels=ls,
        line_spacing=df,
        critical_bandwidths=dfc,
        lower_corners=f1,
        upper_corners=f2,
        critical_band_levels=lg,
        masking_indices=av,
        audibilities=delta,
        extended_uncertainties=uncertainties,
    )
