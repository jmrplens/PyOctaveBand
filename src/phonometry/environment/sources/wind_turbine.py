#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Wind-turbine acoustic noise (IEC 61400-11:2012+A1:2018).

Two closed-form quantities of the standard:

* :func:`apparent_sound_power_level` -- the A-weighted apparent sound power
  level ``L_WA`` referred to the equivalent point source at the rotor centre,
  from the ground-board sound pressure level and the slant distance
  (:func:`slant_distance`),
  :math:`L_{WA} = L_p - 6 + 10 \cdot \log_{10}(4\pi R_1^2/S_0)` (Formula 26).
* :func:`wind_turbine_tonality` -- the tonal-audibility chain (Formulae 30-34):
  the critical bandwidth (:func:`critical_bandwidth`), the masking-noise level,
  the tonality and the audibility criterion, giving the tonal audibility
  ``ΔL_a`` that decides whether a tone is audible.

The tonal-audibility formula itself is the ISO 1996-2 Annex C one already in
:mod:`~phonometry.environment.assessment.measurement`; what is specific to IEC 61400-11 is
how the tone and masking-noise levels and the (Zwicker) critical band are
determined from the narrowband spectrum. The rating adjustment ``K_T`` is the
ISO 1996-2 :func:`~phonometry.environment.assessment.measurement.tonal_adjustment`. The
full measurement pipeline (binning, regression to standardised wind speeds,
uncertainty budgets) is out of scope; these are the underlying closed forms.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..._report.metadata import ReportMetadata


class WindTurbineNoiseWarning(PhonometryWarning):
    """The tonality inputs leave the standard's stated domain of validity."""

#: Reference area ``S0`` for the apparent sound power level (m²).
_REFERENCE_AREA = 1.0
#: The apparent-sound-power ground-board pressure-doubling term (dB).
_BOARD_TERM = 6.0
#: Effective-noise-bandwidth Hanning factor (× frequency resolution).
_HANNING_ENBW_FACTOR = 1.5
#: Low-frequency band where the critical band is fixed to 20-120 Hz.
_LOW_FREQ_MIN, _LOW_FREQ_MAX = 20.0, 70.0
_LOW_FREQ_BANDWIDTH = 100.0


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return scalar


def slant_distance(
    hub_height: float, rotor_diameter: float, *, rotor_axis: str = "horizontal"
) -> float:
    r"""Slant distance ``R1`` from the rotor centre to the ground microphone.

    With the reference microphone on the ground at the horizontal distance
    ``R0`` and the rotor centre at height ``H``, the slant distance is
    :math:`R_1 = \sqrt{H^2 + R_0^2}`. For a horizontal-axis turbine
    :math:`R_0 = H + D/2`
    (IEC 61400-11 Formula 1); for a vertical-axis turbine :math:`R_0 = H + D`
    (Formula 2, with ``H`` the height of the equator of the rotor).

    :param hub_height: Hub height ``H`` (ground to rotor centre; for a
        vertical-axis turbine, to the rotor equator), in m.
    :param rotor_diameter: Rotor diameter ``D``, in m.
    :param rotor_axis: ``"horizontal"`` (Formula 1, the default) or
        ``"vertical"`` (Formula 2).
    :return: The slant distance ``R1``, in m.
    :raises ValueError: If a dimension is not positive or ``rotor_axis`` is
        not one of the two orientations.
    """
    h = _positive(hub_height, "hub_height")
    d = _positive(rotor_diameter, "rotor_diameter")
    if rotor_axis == "horizontal":
        r0 = h + d / 2.0
    elif rotor_axis == "vertical":
        r0 = h + d
    else:
        raise ValueError("'rotor_axis' must be 'horizontal' or 'vertical'.")
    return float(np.hypot(h, r0))


def apparent_sound_power_level(
    band_levels: float | NDArray[np.float64] | list[float], r1: float
) -> float:
    r"""A-weighted apparent sound power level ``L_WA`` (IEC 61400-11 Formula 26).

    :math:`L_{WA,i} = L_{p,i} - 6 + 10 \cdot \log_{10}(4\pi R_1^2/S_0)` per
    one-third-octave band, energy
    summed over bands (Formula 27). The ``−6 dB`` accounts for the ground-board
    pressure doubling; :math:`S_0 = 1` m².

    :param band_levels: Background-corrected A-weighted band sound pressure
        levels ``L_p,i``, in dB (scalar or per band). The 61400-11-specific
        background correction (Formula 23 subtraction with the 3-6 dB
        asterisk marking and the <= 3 dB not-reported rule, subclause 9.3) is
        out of scope here and must be applied beforehand; note its rule set
        differs from the ISO 1996-2 correction in
        :func:`~phonometry.environment.assessment.measurement.residual_sound_correction`.
    :param r1: Slant distance ``R1`` to the rotor centre, in m.
    :return: The apparent sound power level ``L_WA``, in dB re 1 pW.
    :raises ValueError: If the inputs are invalid.
    """
    levels = np.atleast_1d(np.asarray(band_levels, dtype=np.float64))
    if levels.size == 0 or not np.all(np.isfinite(levels)):
        raise ValueError("'band_levels' must be finite and non-empty.")
    r = _positive(r1, "r1")
    geometry = 10.0 * np.log10(4.0 * np.pi * r**2 / _REFERENCE_AREA)
    lwa_bands = levels - _BOARD_TERM + geometry
    return float(10.0 * np.log10(np.sum(10.0 ** (lwa_bands / 10.0))))


def critical_bandwidth(fc: float) -> float:
    r"""Critical bandwidth about a tone (IEC 61400-11 Formula 30), in Hz.

    :math:`\mathrm{CBW} = 25 + 75 \cdot [1 + 1.4 \cdot
    (f_c/1000)^2]^{0.69}`. For a candidate tone in the
    20-70 Hz range the critical band is fixed to 20-120 Hz (100 Hz wide),
    per subclause 9.5.3.

    :param fc: Tone (local-maximum) frequency, in Hz.
    :return: The critical bandwidth, in Hz.
    :raises ValueError: If the frequency is not positive.
    """
    f = _positive(fc, "fc")
    if _LOW_FREQ_MIN <= f <= _LOW_FREQ_MAX:
        return _LOW_FREQ_BANDWIDTH
    return float(25.0 + 75.0 * (1.0 + 1.4 * (f / 1000.0) ** 2) ** 0.69)


def _critical_band_edges(fc: float) -> tuple[float, float]:
    """Lower and upper edges of the critical band about a tone (Hz).

    For a candidate tone in the 20-70 Hz range the band is the fixed absolute
    20-120 Hz band (subclause 9.5.3), not one centred on ``fc``; otherwise it is
    the Zwicker critical band ``fc ± CBW/2``.
    """
    f = _positive(fc, "fc")
    if _LOW_FREQ_MIN <= f <= _LOW_FREQ_MAX:
        return _LOW_FREQ_MIN, _LOW_FREQ_MIN + _LOW_FREQ_BANDWIDTH
    cbw = critical_bandwidth(f)
    return f - cbw / 2.0, f + cbw / 2.0


def _energy_mean(levels_db: NDArray[np.float64]) -> float:
    return float(10.0 * np.log10(np.mean(10.0 ** (levels_db / 10.0))))


#: Tone-classification margins (dB): masking-line criterion and tone criterion.
_MASKING_MARGIN = 6.0
_TONE_MARGIN = 6.0
#: 9.5.2 possible-tone margin above the screening energy average (dB).
_POSSIBLE_TONE_MARGIN = 6.0
#: Tone lines stay classified while within this level of the highest tone line
#: (dB, subclause 9.5.3: "within 10 dB of the highest level").
_TONE_GROUP_DROP = 10.0
#: Audibility criterion constants of Formula 34.
_LA_OFFSET, _LA_PIVOT, _LA_EXP = -2.0, 502.0, 2.5


@dataclass(frozen=True)
class WindTurbineTonalityResult:
    r"""Tonal audibility of a narrowband spectrum (IEC 61400-11).

    :ivar tone_frequency: The frequency of the identified tone: the spectral
        line with the highest level among the lines classified as "tone"
        (subclause 9.5.4; also the ``f`` of Formula 34). When no tone is
        identified (``has_identified_tone`` is ``False``) this falls back to
        the candidate line.
    :ivar critical_bandwidth: The critical bandwidth about the candidate, Hz.
    :ivar tone_level: Tone level ``L_pt`` (energy sum of the tone lines), in dB.
    :ivar masking_level: Masking-noise level ``L_pn``, in dB.
    :ivar tonality: Tonality :math:`\Delta L_{tn} = L_{pt} - L_{pn}`, in dB.
    :ivar audibility_criterion: The criterion ``L_a`` (Formula 34), in dB.
    :ivar tonal_audibility: Tonal audibility
        :math:`\Delta L_a = \Delta L_{tn} - L_a`, in dB.
    :ivar is_audible: Whether an identified tone is audible
        (:math:`\Delta L_a > 0` *and* ``has_identified_tone``).
    :ivar has_identified_tone: Whether the candidate passed the 9.5.2
        possible-tone screening *and* at least one spectral line was
        classified as "tone" (subclause 9.5.4). When ``False`` the numeric
        fields are non-standard fallbacks (the standard defines no tonality
        for such a spectrum) and the spectrum must be **excluded** from the
        9.5.1 energy averaging of ``ΔL_a,j,k`` over the spectra of a bin.
    :ivar frequencies: The narrowband line frequencies, in Hz.
    :ivar levels: The narrowband line levels, in dB.
    """

    tone_frequency: float
    critical_bandwidth: float
    tone_level: float
    masking_level: float
    tonality: float
    audibility_criterion: float
    tonal_audibility: float
    is_audible: bool
    has_identified_tone: bool
    frequencies: NDArray[np.float64]
    levels: NDArray[np.float64]

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the narrowband spectrum with the critical band and masking level."""
        from ..._i18n import check_language
        from ..._plot.environment import plot_wind_turbine_tonality

        return plot_wind_turbine_tonality(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a wind-turbine tonal audibility assessment fiche to a PDF.

        Writes a one-page tonality-assessment report following
        IEC 61400-11:2012+A1:2018 (subclauses 9.5.2 to 9.5.8): the standard-basis
        line, an optional metadata header (source/situation, client, measurement
        position, instrumentation and date), a two-panel body with the
        critical-band / masking analysis in a metrics table (tone frequency,
        critical bandwidth, tone level ``L_pt``, masking-noise level ``L_pn``,
        tonality ``ΔL_tn``, audibility criterion ``L_a`` and tonal audibility
        ``ΔL_a``) beside the narrowband-spectrum plot with the critical band,
        masking level and tone marked, the boxed decisive tonal audibility
        ``ΔL_a`` and the tone frequency with the audibility decision, an optional
        verdict row and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a bare
            assessment fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the maximum acceptable tonal audibility
            ``ΔL_a`` in dB (a lower audibility passes).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for signature parity with the other fiches; the
            metrics table already shows the full Formula 30-34 chain, so it has
            no effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``language`` is not one of the supported
            languages, or if ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iec61400 import render_wind_turbine_tonality_report

        return render_wind_turbine_tonality_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _validate_narrowband(
    levels: NDArray[np.float64] | list[float],
    frequencies: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Coerce and validate a uniformly-spaced narrowband spectrum."""
    lv = np.asarray(levels, dtype=np.float64)
    fr = np.asarray(frequencies, dtype=np.float64)
    if lv.ndim != 1 or lv.shape != fr.shape or lv.size < 3:
        raise ValueError("'levels' and 'frequencies' must be 1-D and the same length (>= 3).")
    if not (np.all(np.isfinite(lv)) and np.all(np.isfinite(fr))):
        raise ValueError("'levels' and 'frequencies' must be finite.")
    diffs = np.diff(fr)
    if np.any(diffs <= 0.0):
        raise ValueError("'frequencies' must be strictly increasing.")
    df = float(np.median(diffs))
    if np.any(np.abs(diffs - df) > 1e-3 * df):
        raise ValueError("'frequencies' must be uniformly spaced (a narrowband spectrum).")
    return lv, fr, df


def _candidate_peak(
    lv: NDArray[np.float64], fr: NDArray[np.float64],
    tone_frequency: float | None,
) -> int:
    """Index of the candidate line: the spectrum maximum, or the validated
    nearest bin to the requested frequency."""
    if tone_frequency is None:
        return int(np.argmax(lv))
    tf = float(tone_frequency)
    if not np.isfinite(tf) or tf < float(fr[0]) or tf > float(fr[-1]):
        raise ValueError(
            "'tone_frequency' must be finite and inside the spectrum's "
            f"frequency range [{fr[0]:.1f}, {fr[-1]:.1f}] Hz."
        )
    return int(np.argmin(np.abs(fr - tf)))


def _screen_possible_tone(
    lv: NDArray[np.float64], in_band: NDArray[np.bool_], peak: int,
) -> bool:
    """9.5.2 possible-tone screen: local maximum more than 6 dB above the
    critical-band energy average that excludes the maximum line and its two
    adjacent lines."""
    screen_indices = np.nonzero(in_band)[0]
    screen_indices = screen_indices[np.abs(screen_indices - peak) > 1]
    local_max = (peak == 0 or lv[peak] >= lv[peak - 1]) and (
        peak == lv.size - 1 or lv[peak] >= lv[peak + 1]
    )
    return bool(
        screen_indices.size
        and local_max
        and lv[peak] > _energy_mean(lv[screen_indices]) + _POSSIBLE_TONE_MARGIN
    )


def wind_turbine_tonality(
    levels: NDArray[np.float64] | list[float],
    frequencies: NDArray[np.float64] | list[float],
    *,
    tone_frequency: float | None = None,
) -> WindTurbineTonalityResult:
    r"""Tonal audibility of a narrowband spectrum (IEC 61400-11 Formulae 30-34).

    From a uniformly-spaced narrowband spectrum, screens the candidate as a
    possible tone (subclause 9.5.2: a local maximum more than 6 dB above the
    critical-band energy average excluding the maximum and its two adjacent
    lines), classifies the lines in the critical band about the candidate into
    masking noise and tone lines, forms the masking-noise level ``L_pn``
    (Formula 31), the tonality :math:`\Delta L_{tn} = L_{pt} - L_{pn}`
    (Formula 32), the
    audibility criterion ``L_a`` (Formula 34) and the tonal audibility
    :math:`\Delta L_a = \Delta L_{tn} - L_a` (Formula 33).

    Per subclause 9.5.3 the tone lines are those above ``L_pn,avg + 6 dB``
    *and* within 10 dB of the highest such line; the frequency of the tone is
    that highest line (9.5.4), which also anchors ``L_a``. When the candidate
    fails the 9.5.2 screening or no line classifies as "tone", the result
    carries ``has_identified_tone = False``: its numeric fields are
    non-standard fallbacks and the spectrum must be excluded from the 9.5.1
    bin averaging (see :class:`WindTurbineTonalityResult`).

    Spectra must extend over the whole critical band: a truncated band leaves
    the masking average over the surviving lines while Formula 31 still scales
    to the full bandwidth, so a :class:`WindTurbineNoiseWarning` is issued.
    Candidates below 20 Hz are outside the standard's analysis range (the
    critical band would extend to negative frequencies) and are rejected.

    :param levels: Narrowband line levels, in dB (A-weighted, 1-2 Hz resolution).
    :param frequencies: Line frequencies, in Hz (uniform spacing).
    :param tone_frequency: Candidate tone frequency, in Hz; if ``None`` the
        highest-level line is used.
    :return: A :class:`WindTurbineTonalityResult`.
    :raises ValueError: If the inputs are invalid or the candidate lies below
        20 Hz.
    """
    lv, fr, df = _validate_narrowband(levels, frequencies)

    peak = _candidate_peak(lv, fr, tone_frequency)
    fc = float(fr[peak])
    if fc < _LOW_FREQ_MIN:
        raise ValueError(
            "The candidate tone lies below 20 Hz: outside the standard's "
            "analysis range (the critical band would extend below 0 Hz)."
        )
    cbw = critical_bandwidth(fc)
    lo, hi = _critical_band_edges(fc)
    if fr[0] > lo + 1e-9 or fr[-1] < hi - 1e-9:
        warnings.warn(
            "The spectrum does not cover the whole critical band "
            f"[{lo:.1f}, {hi:.1f}] Hz: the masking level is averaged over the "
            "surviving lines while Formula 31 scales to the full bandwidth, "
            "biasing the tonal audibility.",
            WindTurbineNoiseWarning,
            stacklevel=2,
        )
    in_band = (fr >= lo) & (fr <= hi)
    band = lv[in_band]

    # L_70%: energy mean of the 70 % lowest-level lines in the critical band.
    n_low = max(1, round(0.7 * band.size))
    l70 = _energy_mean(np.sort(band)[:n_low])
    masking = band[band < l70 + _MASKING_MARGIN]
    l_pn_avg = _energy_mean(masking) if masking.size else l70
    tone_threshold = l_pn_avg + _TONE_MARGIN

    possible_tone = _screen_possible_tone(lv, in_band, peak)

    # Tone lines (9.5.3, "adjacent" struck out by A1:2018): every line in the
    # critical band above the tone threshold and within 10 dB of the highest
    # such line ("the line having the greatest level is identified; lines are
    # then only classified as tone if within 10 dB of the highest level"),
    # whether or not contiguous with it.
    above = in_band & (lv > tone_threshold)
    if np.any(above):
        highest_level = float(np.max(lv[above]))
        is_tone = above & (highest_level - lv <= _TONE_GROUP_DROP)
    else:
        is_tone = above
    tone_positions = np.nonzero(is_tone)[0]
    has_identified_tone = possible_tone and tone_positions.size > 0
    if not has_identified_tone:
        # Non-standard fallback so the numeric fields stay defined; flagged
        # by has_identified_tone = False.
        tone_positions = np.array([peak])
    # The frequency of the tone is the classified line with the highest level
    # (9.5.4); it anchors the reported frequency and Formula 34.
    f_tone = float(fr[tone_positions[int(np.argmax(lv[tone_positions]))]])
    # Energy-sum the tone lines per contiguous run (9.5.5), applying the Hanning
    # ÷1.5 correction to any run that spans two or more adjacent lines.
    runs = np.split(tone_positions, np.nonzero(np.diff(tone_positions) != 1)[0] + 1)
    tone_energy = 0.0
    for run in runs:
        run_energy = float(np.sum(10.0 ** (lv[run] / 10.0)))
        if run.size >= 2:
            run_energy /= _HANNING_ENBW_FACTOR
        tone_energy += run_energy
    l_pt = 10.0 * np.log10(tone_energy)

    enbw = _HANNING_ENBW_FACTOR * df
    l_pn = l_pn_avg + 10.0 * np.log10(cbw / enbw)
    tonality = l_pt - l_pn
    l_a = _LA_OFFSET - np.log10(1.0 + (f_tone / _LA_PIVOT) ** _LA_EXP)
    delta_la = float(tonality - l_a)

    return WindTurbineTonalityResult(
        tone_frequency=f_tone,
        critical_bandwidth=cbw,
        tone_level=float(l_pt),
        masking_level=float(l_pn),
        tonality=float(tonality),
        audibility_criterion=float(l_a),
        tonal_audibility=delta_la,
        is_audible=bool(delta_la > 0.0 and has_identified_tone),
        has_identified_tone=has_identified_tone,
        frequencies=fr,
        levels=lv,
    )
