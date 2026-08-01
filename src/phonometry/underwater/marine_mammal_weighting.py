#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Regulatory auditory weighting and exposure criteria for marine mammals.

Noise-exposure assessments weight a spectrum by a hearing-group filter before
comparing it with a threshold. The filter is the same band-pass form in all
current guidance (NMFS 2018 Equation 1, Southall et al. 2019 Equation 2):

.. math::
    W(f) = C + 10\,\log_{10}\frac{(f/f_1)^{2a}}
                              {[1+(f/f_1)^2]^{a}\,[1+(f/f_2)^2]^{b}}

with ``f`` in kilohertz. ``C`` is fixed by putting the peak of ``W`` at 0 dB,
so the companion **exposure function**
:math:`E(f) = K - 10 \log_{10}(\dots) = K + C - W(f)`
has its minimum at the weighted threshold :math:`T_w = K + C`. Only the
parameter table changes between guidance versions, so the version is explicit
in the API and is carried on every result object:

* ``"nmfs-2024"`` -- NOAA Fisheries, *Updated Technical Guidance*, version 3.0
  (October 2024), Table 5 and Table ES3. **The default**: it supersedes the
  2018 revision, uses :math:`b = 5` for every group, renames the groups to
  the Southall scheme (LF/HF/VHF cetaceans, PW/OW in water, PA/OA in air) and
  replaces "PTS onset" with "auditory injury (AUD INJ) onset".
* ``"nmfs-2018"`` -- the 2018 revision, version 2.0, Table 3 and Table ES3.
  Still cited by assessments already in flight.
* ``"southall-2019"`` -- Southall et al., *Aquatic Mammals* 45(2), Tables 5, 6
  and 7, the peer-reviewed criteria; adds sirenians (SI) and both in-air
  carnivore groups. Numerically identical to NMFS 2018 on the five shared
  groups.

**Group names are not portable between versions.** NMFS 2018 calls the
mid-frequency cetaceans ``MF`` and the porpoise-type group ``HF``; NMFS 2024
and Southall call the same two ``HF`` and ``VHF``. Each guidance version only
accepts its own codes.

The module exposes the weighting itself (:func:`auditory_weighting`), the
published thresholds (:func:`exposure_criteria`) and the assessment chain
(:func:`weighted_exposure`), which weights a band spectrum, accumulates it over
a number of events and reports the exceedance of each applicable criterion.

Implemented clean-room from the three documents; validated against the worked
example of NMFS (2018) Appendix D (:math:`W(1~\text{kHz})` for the five
groups), against ``C`` recomputed as the peak of ``W`` for all three
parameter sets, and against the published :math:`T_w = K + C` and
injury = TTS + 20 dB identities.

.. note::
    NMFS (2024) Table 5 prints :math:`C = 1.37` dB for otariid pinnipeds in
    water, and its own footnote states the value should be 1.36 dB (NMFS kept
    1.37 for consistency with the U.S. Navy). Recomputing ``C`` from the peak
    of ``W``
    with that row's parameters gives 1.3643 dB, confirming 1.36. This module
    implements **1.36**; the printed 1.37 remains available as
    ``WeightingParameters.c_db_as_printed``. See ``docs/ERRATA.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Selectable guidance versions, most recent first.
WEIGHTING_GUIDANCE = ("nmfs-2024", "nmfs-2018", "southall-2019")


@dataclass(frozen=True)
class WeightingParameters:
    """Auditory weighting and exposure function parameters for one group.

    :ivar group: Hearing-group code as used by its own guidance version.
    :ivar guidance: The guidance version the row comes from.
    :ivar description: Plain-language name of the hearing group.
    :ivar a: Low-frequency exponent ``a``.
    :ivar b: High-frequency exponent ``b``.
    :ivar f1_khz: Low-frequency transition ``f1``, in kHz.
    :ivar f2_khz: High-frequency transition ``f2``, in kHz.
    :ivar c_db: Gain ``C`` that puts the peak of ``W`` at 0 dB, in dB.
    :ivar c_db_as_printed: ``C`` exactly as printed in the source table, in dB
        (differs from ``c_db`` only for the NMFS 2024 otariid row).
    :ivar k_db: Exposure-function constant ``K``, in dB.
    :ivar in_air: Whether the group's reference is 20 µPa (in air).
    :ivar hearing_range_hz: Generalised hearing range of the group, in Hz, or
        ``None``. Only the NMFS documents tabulate one (their Table ES1);
        Southall et al. do not, and the field is ``None`` for those rows rather
        than borrowed from elsewhere.
    """

    group: str
    guidance: str
    description: str
    a: float
    b: float
    f1_khz: float
    f2_khz: float
    c_db: float
    c_db_as_printed: float
    k_db: float
    in_air: bool
    hearing_range_hz: tuple[float, float] | None


def _row(
    group: str, guidance: str, description: str, a: float, b: float, f1: float,
    f2: float, c: float, k: float, *,
    hearing_range: tuple[float, float] | None = None,
    in_air: bool = False, c_printed: float | None = None,
) -> WeightingParameters:
    return WeightingParameters(
        group=group, guidance=guidance, description=description, a=a, b=b,
        f1_khz=f1, f2_khz=f2, c_db=c,
        c_db_as_printed=c if c_printed is None else c_printed,
        k_db=k, in_air=in_air, hearing_range_hz=hearing_range,
    )


#: Group descriptions shared by the three parameter tables. The NMFS 2018 "MF"
#: row and the NMFS 2024 / Southall "HF" row are the same animals under
#: different codes, which is why the same description text recurs.
_LF_CETACEANS = "Low-frequency cetaceans"
_HF_CETACEANS = "High-frequency cetaceans"

#: NMFS (2018) revision v2.0, Table 3 = Table ES2 (printed pp. 18 and 3) with
#: the generalised hearing ranges of Table ES1 (printed p. 2).
_NMFS_2018: dict[str, WeightingParameters] = {
    "LF": _row("LF", "nmfs-2018", _LF_CETACEANS,
               1.0, 2.0, 0.2, 19.0, 0.13, 179.0, hearing_range=(7.0, 35e3)),
    "MF": _row("MF", "nmfs-2018", "Mid-frequency cetaceans",
               1.6, 2.0, 8.8, 110.0, 1.20, 177.0, hearing_range=(150.0, 160e3)),
    "HF": _row("HF", "nmfs-2018", _HF_CETACEANS,
               1.8, 2.0, 12.0, 140.0, 1.36, 152.0, hearing_range=(275.0, 160e3)),
    "PW": _row("PW", "nmfs-2018", "Phocid pinnipeds (underwater)",
               1.0, 2.0, 1.9, 30.0, 0.75, 180.0, hearing_range=(50.0, 86e3)),
    "OW": _row("OW", "nmfs-2018", "Otariid pinnipeds (underwater)",
               2.0, 2.0, 0.94, 25.0, 0.64, 198.0, hearing_range=(60.0, 39e3)),
}

#: NMFS (2024) update v3.0, Table 5 = Table ES2 (printed pp. 25 and 3) with the
#: generalised hearing ranges of Table ES1 (printed p. 2). The otariid ``C`` is
#: implemented as 1.36 dB (see the module note).
_NMFS_2024: dict[str, WeightingParameters] = {
    "LF": _row("LF", "nmfs-2024", _LF_CETACEANS,
               0.99, 5.0, 0.168, 26.6, 0.12, 177.0, hearing_range=(7.0, 36e3)),
    "HF": _row("HF", "nmfs-2024", _HF_CETACEANS,
               1.55, 5.0, 1.73, 129.0, 0.32, 181.0, hearing_range=(150.0, 160e3)),
    "VHF": _row("VHF", "nmfs-2024", "Very high-frequency cetaceans",
                2.23, 5.0, 5.93, 186.0, 0.91, 160.0, hearing_range=(200.0, 165e3)),
    "PW": _row("PW", "nmfs-2024", "Phocid pinnipeds (underwater)",
               1.63, 5.0, 0.81, 68.3, 0.29, 175.0, hearing_range=(40.0, 90e3)),
    "OW": _row("OW", "nmfs-2024", "Otariid pinnipeds (underwater)",
               1.58, 5.0, 2.53, 43.8, 1.36, 178.0, hearing_range=(60.0, 68e3),
               c_printed=1.37),
    "PA": _row("PA", "nmfs-2024", "Phocid pinnipeds (in air)",
               2.05, 5.0, 0.74, 24.4, 0.83, 133.0, hearing_range=(42.0, 52e3),
               in_air=True),
    "OA": _row("OA", "nmfs-2024", "Otariid pinnipeds (in air)",
               1.35, 5.0, 1.75, 32.5, 1.18, 156.0, hearing_range=(90.0, 40e3),
               in_air=True),
}

#: Southall et al. (2019) Table 5 (printed p. 149). The article tabulates no
#: generalised hearing range, so those fields stay ``None``.
_SOUTHALL_2019: dict[str, WeightingParameters] = {
    "LF": _row("LF", "southall-2019", _LF_CETACEANS,
               1.0, 2.0, 0.20, 19.0, 0.13, 179.0),
    "HF": _row("HF", "southall-2019", _HF_CETACEANS,
               1.6, 2.0, 8.8, 110.0, 1.20, 177.0),
    "VHF": _row("VHF", "southall-2019", "Very high-frequency cetaceans",
                1.8, 2.0, 12.0, 140.0, 1.36, 152.0),
    "SI": _row("SI", "southall-2019", "Sirenians",
               1.8, 2.0, 4.3, 25.0, 2.62, 183.0),
    "PCW": _row("PCW", "southall-2019", "Phocid carnivores (underwater)",
                1.0, 2.0, 1.9, 30.0, 0.75, 180.0),
    "OCW": _row("OCW", "southall-2019", "Other marine carnivores (underwater)",
                2.0, 2.0, 0.94, 25.0, 0.64, 198.0),
    "PCA": _row("PCA", "southall-2019", "Phocid carnivores (in air)",
                2.0, 2.0, 0.75, 8.3, 1.50, 132.0, in_air=True),
    "OCA": _row("OCA", "southall-2019", "Other marine carnivores (in air)",
                1.4, 2.0, 2.0, 20.0, 1.39, 156.0, in_air=True),
}

_PARAMETERS: dict[str, dict[str, WeightingParameters]] = {
    "nmfs-2024": _NMFS_2024,
    "nmfs-2018": _NMFS_2018,
    "southall-2019": _SOUTHALL_2019,
}


@dataclass(frozen=True)
class ExposureCriteria:
    """Published TTS and injury (PTS / AUD INJ) onset criteria for one group.

    Sound exposure levels are weighted; peak sound pressure levels are
    unweighted ("flat"), as every source states. ``None`` means the criterion
    is not published by that guidance version.

    :ivar group: Hearing-group code.
    :ivar guidance: The guidance version.
    :ivar impulsive: Whether these are the impulsive-noise criteria.
    :ivar tts_sel: Weighted TTS-onset sound exposure level, in dB.
    :ivar injury_sel: Weighted injury-onset (PTS / AUD INJ) SEL, in dB.
    :ivar tts_peak_spl: Unweighted TTS-onset peak SPL, in dB.
    :ivar injury_peak_spl: Unweighted injury-onset peak SPL, in dB.
    :ivar injury_label: ``"PTS"`` or ``"AUD INJ"``, as the source names it.
    :ivar sel_reference: Human-readable SEL reference of the group.
    :ivar peak_reference: Human-readable peak-SPL reference of the group.
    :ivar source: Table the numbers come from.
    """

    group: str
    guidance: str
    impulsive: bool
    tts_sel: float | None
    injury_sel: float | None
    tts_peak_spl: float | None
    injury_peak_spl: float | None
    injury_label: str
    sel_reference: str
    peak_reference: str
    source: str


#: One criteria row: ``(tts_sel, injury_sel, tts_peak_spl, injury_peak_spl)``.
_CriteriaRow = tuple[float | None, float | None, float | None, float | None]

#: Non-impulsive and impulsive criteria per guidance version and group.
#: NMFS 2018: Table 3 (weighted TTS onset) and Table ES3 (PTS onset).
_CRITERIA_2018_CONTINUOUS: dict[str, _CriteriaRow] = {
    "LF": (179.0, 199.0, None, None), "MF": (178.0, 198.0, None, None),
    "HF": (153.0, 173.0, None, None), "PW": (181.0, 201.0, None, None),
    "OW": (199.0, 219.0, None, None),
}
_CRITERIA_2018_IMPULSIVE: dict[str, _CriteriaRow] = {
    "LF": (None, 183.0, None, 219.0), "MF": (None, 185.0, None, 230.0),
    "HF": (None, 155.0, None, 202.0), "PW": (None, 185.0, None, 218.0),
    "OW": (None, 203.0, None, 232.0),
}
#: NMFS 2024: Table 5 / Table ES3, completed with the Navy Phase 4 impulsive
#: TTS values of Table A.E-2 (printed p. 43) that ES3 does not repeat.
_CRITERIA_2024_CONTINUOUS: dict[str, _CriteriaRow] = {
    "LF": (177.0, 197.0, None, None), "HF": (181.0, 201.0, None, None),
    "VHF": (161.0, 181.0, None, None), "PW": (175.0, 195.0, None, None),
    "OW": (179.0, 199.0, None, None), "PA": (134.0, 154.0, None, None),
    "OA": (157.0, 177.0, None, None),
}
_CRITERIA_2024_IMPULSIVE: dict[str, _CriteriaRow] = {
    "LF": (168.0, 183.0, 216.0, 222.0), "HF": (178.0, 193.0, 224.0, 230.0),
    "VHF": (144.0, 159.0, 196.0, 202.0), "PW": (168.0, 183.0, 217.0, 223.0),
    "OW": (170.0, 185.0, 224.0, 230.0), "PA": (125.0, 140.0, 156.0, 162.0),
    "OA": (148.0, 163.0, 171.0, 177.0),
}
#: Southall et al. (2019) Table 6 (non-impulsive) and Table 7 as corrected by
#: the errata of Aquatic Mammals 45(5), printed p. 570.
_CRITERIA_SOUTHALL_CONTINUOUS: dict[str, _CriteriaRow] = {
    "LF": (179.0, 199.0, None, None), "HF": (178.0, 198.0, None, None),
    "VHF": (153.0, 173.0, None, None), "SI": (186.0, 206.0, None, None),
    "PCW": (181.0, 201.0, None, None), "OCW": (199.0, 219.0, None, None),
    "PCA": (134.0, 154.0, None, None), "OCA": (157.0, 177.0, None, None),
}
_CRITERIA_SOUTHALL_IMPULSIVE: dict[str, _CriteriaRow] = {
    "LF": (168.0, 183.0, 213.0, 219.0), "HF": (170.0, 185.0, 224.0, 230.0),
    "VHF": (140.0, 155.0, 196.0, 202.0), "SI": (175.0, 190.0, 220.0, 226.0),
    "PCW": (170.0, 185.0, 212.0, 218.0), "OCW": (188.0, 203.0, 226.0, 232.0),
    # Errata-corrected peak SPL values (printed 138/144 and 161/167).
    "PCA": (123.0, 138.0, 155.0, 161.0), "OCA": (146.0, 161.0, 170.0, 176.0),
}

_CRITERIA: dict[str, tuple[dict[str, _CriteriaRow], dict[str, _CriteriaRow], str, str, str]] = {
    "nmfs-2018": (
        _CRITERIA_2018_CONTINUOUS, _CRITERIA_2018_IMPULSIVE, "PTS",
        "NMFS (2018) v2.0 Table 3", "NMFS (2018) v2.0 Table ES3",
    ),
    "nmfs-2024": (
        _CRITERIA_2024_CONTINUOUS, _CRITERIA_2024_IMPULSIVE, "AUD INJ",
        "NMFS (2024) v3.0 Table 5 / Table ES3",
        "NMFS (2024) v3.0 Table ES3 / Table A.E-2",
    ),
    "southall-2019": (
        _CRITERIA_SOUTHALL_CONTINUOUS, _CRITERIA_SOUTHALL_IMPULSIVE, "PTS",
        "Southall et al. (2019) Table 6",
        "Southall et al. (2019) Table 7 (errata-corrected)",
    ),
}


def check_guidance(guidance: str) -> str:
    """Normalise and validate a guidance-version name.

    :param guidance: One of :data:`WEIGHTING_GUIDANCE` (case-insensitive,
        underscores accepted).
    :return: The canonical name.
    :raises ValueError: If the version is unknown.
    """
    key = str(guidance).strip().lower().replace("_", "-")
    if key not in _PARAMETERS:
        raise ValueError(f"'guidance' must be one of {WEIGHTING_GUIDANCE}, got {guidance!r}.")
    return key


def hearing_groups(guidance: str = "nmfs-2024") -> tuple[str, ...]:
    """Hearing-group codes defined by a guidance version.

    :param guidance: One of :data:`WEIGHTING_GUIDANCE`.
    :return: The group codes, in the order the source tabulates them.
    :raises ValueError: If the version is unknown.
    """
    return tuple(_PARAMETERS[check_guidance(guidance)])


def weighting_parameters(group: str, *, guidance: str = "nmfs-2024") -> WeightingParameters:
    """Weighting/exposure parameters of one hearing group.

    :param group: Hearing-group code as used by ``guidance`` (case-insensitive).
    :param guidance: One of :data:`WEIGHTING_GUIDANCE`.
    :return: The :class:`WeightingParameters` row.
    :raises ValueError: If the version or the group is unknown.
    """
    key = check_guidance(guidance)
    table = _PARAMETERS[key]
    name = str(group).strip().upper()
    if name not in table:
        raise ValueError(
            f"'group' must be one of {tuple(table)} for guidance {key!r}, got {group!r}."
            " Group codes are not portable between guidance versions."
        )
    return table[name]


def _band_pass_db(f_khz: NDArray[np.float64], p: WeightingParameters) -> NDArray[np.float64]:
    r""":math:`10 \log_{10}` of the band-pass ratio shared by ``W(f)`` and
    ``E(f)``."""
    ratio = (f_khz / p.f1_khz) ** (2.0 * p.a) / (
        (1.0 + (f_khz / p.f1_khz) ** 2) ** p.a
        * (1.0 + (f_khz / p.f2_khz) ** 2) ** p.b
    )
    return np.asarray(10.0 * np.log10(ratio), dtype=np.float64)


@dataclass(frozen=True)
class AuditoryWeightingResult:
    r"""Auditory weighting and exposure functions of one hearing group.

    :ivar frequencies: Frequencies, in Hz.
    :ivar weighting: Weighting-function amplitude ``W(f)``, in dB
        (:math:`\le 0`).
    :ivar exposure_function: Exposure function
        :math:`E(f) = K + C - W(f)`, in dB (the frequency-dependent
        TTS-onset level).
    :ivar parameters: The :class:`WeightingParameters` used.
    :ivar guidance: The guidance version.
    :ivar group: Hearing-group code.
    :ivar weighted_tts_onset: :math:`T_w = K + C`, the minimum of the
        exposure function, in dB.
    """

    frequencies: NDArray[np.float64]
    weighting: NDArray[np.float64]
    exposure_function: NDArray[np.float64]
    parameters: WeightingParameters
    guidance: str
    group: str
    weighted_tts_onset: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the weighting function versus frequency."""
        from .._i18n import check_language
        from .._plot.underwater import plot_auditory_weighting

        return plot_auditory_weighting(self, ax=ax, language=check_language(language), **kwargs)


def _positive_frequencies(
    frequency_hz: NDArray[np.float64] | list[float] | float,
) -> NDArray[np.float64]:
    f = np.array(frequency_hz, dtype=np.float64, copy=True, ndmin=1)
    if f.size == 0 or not np.all(np.isfinite(f)):
        raise ValueError("'frequency_hz' must be finite and non-empty.")
    if np.any(f <= 0.0):
        raise ValueError("'frequency_hz' must be strictly positive.")
    return f


def auditory_weighting(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    group: str,
    *,
    guidance: str = "nmfs-2024",
) -> AuditoryWeightingResult:
    """Auditory weighting function ``W(f)`` of a marine-mammal hearing group.

    :param frequency_hz: Frequency or frequencies, in Hz (strictly positive).
    :param group: Hearing-group code as used by ``guidance``.
    :param guidance: ``"nmfs-2024"`` (default, current), ``"nmfs-2018"`` or
        ``"southall-2019"``.
    :return: An :class:`AuditoryWeightingResult`.
    :raises ValueError: If an input is invalid.
    """
    params = weighting_parameters(group, guidance=guidance)
    f = _positive_frequencies(frequency_hz)
    shape = _band_pass_db(f / 1000.0, params)
    weighting = params.c_db + shape
    return AuditoryWeightingResult(
        frequencies=f,
        weighting=weighting,
        exposure_function=params.k_db - shape,
        parameters=params,
        guidance=params.guidance,
        group=params.group,
        weighted_tts_onset=float(params.k_db + params.c_db),
    )


def exposure_criteria(
    group: str, *, guidance: str = "nmfs-2024", impulsive: bool = False
) -> ExposureCriteria:
    """Published TTS and injury onset criteria of a hearing group.

    :param group: Hearing-group code as used by ``guidance``.
    :param guidance: ``"nmfs-2024"`` (default), ``"nmfs-2018"`` or
        ``"southall-2019"``.
    :param impulsive: Return the impulsive-noise criteria (dual metric: a
        weighted SEL and an unweighted peak SPL) instead of the non-impulsive
        ones.
    :return: An :class:`ExposureCriteria`.
    :raises ValueError: If the version or the group is unknown.
    """
    params = weighting_parameters(group, guidance=guidance)
    cont, imp, label, cont_source, imp_source = _CRITERIA[params.guidance]
    tts_sel, injury_sel, tts_peak, injury_peak = (imp if impulsive else cont)[params.group]
    sel_ref = "dB re (20 µPa)²·s" if params.in_air else "dB re 1 µPa²·s"
    peak_ref = "dB re 20 µPa" if params.in_air else "dB re 1 µPa"
    return ExposureCriteria(
        group=params.group,
        guidance=params.guidance,
        impulsive=impulsive,
        tts_sel=tts_sel,
        injury_sel=injury_sel,
        tts_peak_spl=tts_peak,
        injury_peak_spl=injury_peak,
        injury_label=label,
        sel_reference=sel_ref,
        peak_reference=peak_ref,
        source=imp_source if impulsive else cont_source,
    )


@dataclass(frozen=True)
class WeightedExposureResult:
    r"""Weighted exposure of a spectrum against a hearing group's criteria.

    :ivar frequencies: Band centre frequencies, in Hz.
    :ivar band_sel: Per-band single-event sound exposure level, in dB.
    :ivar weighting: Weighting-function amplitude at each band, in dB.
    :ivar weighted_band_sel: ``band_sel + W(f)`` per band, in dB.
    :ivar unweighted_sel: Energy sum of ``band_sel``, in dB.
    :ivar weighted_sel: Energy sum of ``weighted_band_sel``, in dB.
    :ivar cumulative_sel: ``weighted_sel`` plus :math:`10 \log_{10}(N)` for the
        ``n_events`` accumulated events, in dB.
    :ivar peak_spl: The unweighted peak sound pressure level supplied, in dB
        (``None`` when not given).
    :ivar n_events: Number of accumulated events (e.g. hammer strikes).
    :ivar criteria: The :class:`ExposureCriteria` compared against.
    :ivar sel_margin: ``cumulative_sel - injury_sel``, in dB (``None`` when the
        criterion is not published); positive means the criterion is exceeded.
    :ivar tts_margin: ``cumulative_sel - tts_sel``, in dB (or ``None``).
    :ivar peak_margin: ``peak_spl - injury_peak_spl``, in dB (or ``None``).
    :ivar tts_peak_margin: ``peak_spl - tts_peak_spl``, in dB (or ``None``) --
        the peak-SPL half of the dual metric on the TTS side, which can trip
        ``exceeds_tts`` on its own.
    :ivar exceeds_injury: Whether any injury-onset criterion is reached. The
        test is ``margin >= 0``, so an exposure landing exactly **on** the
        criterion counts as exceeding it; the criteria are onset thresholds and
        the precautionary reading is the one an assessment wants.
    :ivar exceeds_tts: Whether any TTS-onset criterion is reached, on the same
        ``margin >= 0`` convention as ``exceeds_injury``.
    :ivar guidance: The guidance version.
    :ivar group: Hearing-group code.
    """

    frequencies: NDArray[np.float64]
    band_sel: NDArray[np.float64]
    weighting: NDArray[np.float64]
    weighted_band_sel: NDArray[np.float64]
    unweighted_sel: float
    weighted_sel: float
    cumulative_sel: float
    peak_spl: float | None
    n_events: int
    criteria: ExposureCriteria
    sel_margin: float | None
    tts_margin: float | None
    peak_margin: float | None
    tts_peak_margin: float | None
    exceeds_injury: bool
    exceeds_tts: bool
    guidance: str
    group: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the unweighted and weighted band spectra with the criteria."""
        from .._i18n import check_language
        from .._plot.underwater import plot_weighted_exposure

        return plot_weighted_exposure(self, ax=ax, language=check_language(language), **kwargs)


def _energy_sum(levels: NDArray[np.float64]) -> float:
    return float(10.0 * np.log10(np.sum(10.0 ** (levels / 10.0))))


def _margin(value: float, criterion: float | None) -> float | None:
    return None if criterion is None else float(value - criterion)


def weighted_exposure(
    frequency_hz: NDArray[np.float64] | list[float],
    band_sel: NDArray[np.float64] | list[float],
    group: str,
    *,
    guidance: str = "nmfs-2024",
    impulsive: bool = True,
    n_events: int = 1,
    peak_spl: float | None = None,
) -> WeightedExposureResult:
    r"""Weight a band spectrum, accumulate it and compare it with the
    criteria.

    The per-band single-event sound exposure levels are weighted with
    :func:`auditory_weighting`, summed on an energy basis and accumulated over
    ``n_events`` identical events (:math:`+10 \log_{10} N`, the ISO 18406 Formula 9
    identity used by :func:`~phonometry.underwater.cumulative_sel_identical`).
    The result is compared with the group's TTS and injury onset criteria; the
    peak sound pressure level, if supplied, is compared **unweighted**, as the
    dual-metric rule requires.

    :param frequency_hz: Band centre frequencies, in Hz (1-D, positive).
    :param band_sel: Per-band single-event SEL, in dB re 1 µPa²·s (or
        dB re (20 µPa)²·s for an in-air group); same length. ``-inf`` is
        accepted for a band that carries no energy, which is what
        :func:`~phonometry.underwater.pile_driving_noise.strike_sel_spectrum`
        returns for bands narrower than its FFT bin spacing; such a band adds
        nothing to the energy sum. Both input arrays are copied, so the result
        never aliases the caller's data.
    :param group: Hearing-group code as used by ``guidance``.
    :param guidance: ``"nmfs-2024"`` (default), ``"nmfs-2018"`` or
        ``"southall-2019"``.
    :param impulsive: Compare against the impulsive criteria (the default, the
        case for pile driving and air guns).
    :param n_events: Number of identical accumulated events, :math:`\ge 1`.
    :param peak_spl: Unweighted zero-to-peak sound pressure level of the
        loudest single event, in dB; enables the peak-SPL half of the dual
        metric.
    :return: A :class:`WeightedExposureResult`.
    :raises ValueError: If an input is invalid.
    """
    f = _positive_frequencies(frequency_hz)
    sel = np.array(band_sel, dtype=np.float64, copy=True, ndmin=1)
    if sel.shape != f.shape:
        raise ValueError("'band_sel' must have the same length as 'frequency_hz'.")
    # -inf is admitted as the level of a band that carries no energy (see
    # StrikeSelSpectrum.band_sel); it is the neutral element of the energy sum.
    if np.any(np.isnan(sel)) or np.any(sel == np.inf):
        raise ValueError("'band_sel' must be finite, or -inf for a band with no energy.")
    n_float = float(n_events)
    if not n_float.is_integer() or int(n_float) < 1:
        raise ValueError("'n_events' must be a whole number of events, at least 1.")
    n = int(n_float)

    weights = auditory_weighting(f, group, guidance=guidance)
    weighted_bands = sel + weights.weighting
    weighted = _energy_sum(weighted_bands)
    cumulative = weighted + 10.0 * np.log10(n)
    criteria = exposure_criteria(group, guidance=guidance, impulsive=impulsive)

    peak: float | None = None
    if peak_spl is not None:
        peak = float(peak_spl)
        if not np.isfinite(peak):
            raise ValueError("'peak_spl' must be finite.")

    sel_margin = _margin(cumulative, criteria.injury_sel)
    tts_margin = _margin(cumulative, criteria.tts_sel)
    peak_margin = None if peak is None else _margin(peak, criteria.injury_peak_spl)
    peak_tts_margin = None if peak is None else _margin(peak, criteria.tts_peak_spl)
    return WeightedExposureResult(
        frequencies=f,
        band_sel=sel,
        weighting=weights.weighting,
        weighted_band_sel=np.asarray(weighted_bands, dtype=np.float64),
        unweighted_sel=_energy_sum(sel),
        weighted_sel=weighted,
        cumulative_sel=float(cumulative),
        peak_spl=peak,
        n_events=n,
        criteria=criteria,
        sel_margin=sel_margin,
        tts_margin=tts_margin,
        peak_margin=peak_margin,
        tts_peak_margin=peak_tts_margin,
        exceeds_injury=_any_positive(sel_margin, peak_margin),
        exceeds_tts=_any_positive(tts_margin, peak_tts_margin),
        guidance=weights.guidance,
        group=weights.group,
    )


def _any_positive(*margins: float | None) -> bool:
    """Whether any published margin has been reached (``>= 0``, criterion included)."""
    return any(m is not None and m >= 0.0 for m in margins)
