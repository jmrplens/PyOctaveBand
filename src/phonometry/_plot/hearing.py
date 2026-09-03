#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the hearing domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _LEGEND_UPPER_RIGHT,
    _fractile_band,
    _freq_axis,
    _new_axes,
    theme_fill,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..hearing.hearing_protectors import (
        AssumedProtectionResult,
        HMLRatingResult,
        ProtectedLevelResult,
        SNRRatingResult,
    )
    from ..hearing.noise_induced_hearing_loss import HtlanResult, NiptsResult
    from ..hearing.occupational_exposure import ExposureResult
    from ..hearing.threshold import AgeThresholdResult

# Tolerance under which the requested population fractile counts as the
# median (0.5), so the separate fractile curve, which would duplicate the
# median line, is skipped.
_MEDIAN_FRACTILE_EPS = 1e-9

#: Spanish translations of the fixed strings rendered by the hearing
#: ``.plot()`` renderers, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
#: Labels the renderers repeat; the Spanish table is keyed by the same
#: constants, so a label is written once.
_FREQ_LABEL = "Frequency [Hz]"
_ATTENUATION_LABEL = "Sound attenuation [dB]"
_BAND_LEVEL_LABEL = "A-weighted band level [dB]"
_REDUCTION_LABEL = "Predicted noise level reduction [dB]"
_C_MINUS_A_LABEL = "$L_{p,C} - L_{p,A}$ [dB]"
_SUBJECT_LABEL = "Test subject"
_SPREAD_LABEL = r"$\pm s_f$"
_HML_TITLE = "ISO 4869-2 HML method — $H$ = {h}, $M$ = {m}, $L$ = {l} dB"
#: The curve of Formulae (16) and (17) carries its own label rather than a
#: slice of the title: deriving one from the other by splitting on the dash
#: breaks the moment a translation punctuates differently.
_HML_CURVE_LABEL = "$PNR$ from $H$ = {h}, $M$ = {m}, $L$ = {l} dB"
_NIPTS_LABEL = "NIPTS [dB]"
_FRACTILE_LABEL = "Fractile {v}"

_STRINGS: dict[str, str] = {
    _FREQ_LABEL: "Frecuencia [Hz]",
    _ATTENUATION_LABEL: "Atenuación acústica [dB]",
    _BAND_LEVEL_LABEL: "Nivel de banda ponderado A [dB]",
    _REDUCTION_LABEL: "Reducción prevista del nivel de ruido [dB]",
    _C_MINUS_A_LABEL: _C_MINUS_A_LABEL,
    _SUBJECT_LABEL: "Sujeto de ensayo",
    "mean attenuation $m_f$": r"atenuación media $m_f$",
    _SPREAD_LABEL: _SPREAD_LABEL,
    "assumed protection $APV_{{f{x}}}$": "protección supuesta $APV_{{f{x}}}$",
    "ISO 4869-2 assumed protection values — {x} % performance": "ISO 4869-2 valores de protección supuesta — rendimiento del {x} %",
    _HML_TITLE: "ISO 4869-2 método HML — $H$ = {h}, $M$ = {m}, $L$ = {l} dB",
    _HML_CURVE_LABEL: "$PNR$ a partir de $H$ = {h}, $M$ = {m}, $L$ = {l} dB",
    "ISO 4869-2 single number rating — $SNR$ = {snr} dB": "ISO 4869-2 índice de número único — $SNR$ = {snr} dB",
    "ISO 4869-2 octave-band method — $L'_{{p,A{x}}}$ = {level} dB": "ISO 4869-2 método por bandas de octava — $L'_{{p,A{x}}}$ = {level} dB",
    "per subject": "por sujeto",
    "reference noises (Table 2)": "ruidos de referencia (Tabla 2)",
    "protected band level": "nivel de banda protegido",
    r"$H$, $M$, $L$ anchors": r"anclas $H$, $M$, $L$",
    "Median": "Mediana",
    "Median $N_{50}$": "Mediana $N_{50}$",
    "Threshold deviation from age 18 [dB]": "Desviación del umbral respecto a 18 años [dB]",
    _NIPTS_LABEL: _NIPTS_LABEL,
    "Age (HTLA, ISO 7029)": "Edad (HTLA, ISO 7029)",
    "Noise (NIPTS)": "Ruido (NIPTS)",
    "Age + noise (HTLAN)": "Edad + ruido (HTLAN)",
    "Hearing threshold level [dB]": "Nivel del umbral de audición [dB]",
    "A-weighted level [dB]": "Nivel ponderado A [dB]",
    _FRACTILE_LABEL: "Fractil {v}",
    "male": "hombre",
    "female": "mujer",
    "ISO 7029 hearing threshold — {sex}, age {age}": "ISO 7029 umbral de audición — {sex}, edad {age}",
    r"ISO 1999 NIPTS — $L_\mathrm{{EX,8h}}$ = {lex} dB, {years} yr": r"ISO 1999 NIPTS — $L_\mathrm{{EX,8h}}$ = {lex} dB, {years} años",
    "ISO 1999 HTLAN — {sex}, age {age}, {lex} dB / {years} yr": "ISO 1999 HTLAN — {sex}, edad {age}, {lex} dB / {years} años",
    r"ISO 9612 daily noise exposure — $L_\mathrm{{EX,8h}}$ = {lex} dB ($U$ = {u} dB)": r"ISO 9612 exposición diaria al ruido — $L_\mathrm{{EX,8h}}$ = {lex} dB "
    r"($U$ = {u} dB)",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_age_threshold(
    result: AgeThresholdResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Median age-related hearing threshold with the 10-90 % fractile band.

    :param result: An :class:`~phonometry.hearing.AgeThresholdResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the median line ``plot``.
    :return: The axes.
    """
    from .._i18n import decimal_comma, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    median = np.asarray(result.median, dtype=np.float64)
    su = np.asarray(result.spread_upper, dtype=np.float64)
    sl = np.asarray(result.spread_lower, dtype=np.float64)

    _fractile_band(ax, freqs, median, sl, su, color=_C_PRIMARY, language=language)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Median", language))
    ax.plot(freqs, median, "o-", **kwargs)
    if abs(result.fractile - 0.5) > _MEDIAN_FRACTILE_EPS:
        ax.plot(
            freqs,
            np.asarray(result.threshold, dtype=np.float64),
            "s--",
            color=_C_REFERENCE,
            label=_t(_FRACTILE_LABEL, language).format(
                v=decimal_comma(f"{result.fractile:g}", language)
            ),
        )
    _freq_axis(ax, freqs, language=language)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Threshold deviation from age 18 [dB]", language))
    ax.invert_yaxis()  # audiogram convention: worse hearing downward
    ax.set_title(
        _t("ISO 7029 hearing threshold — {sex}, age {age}", language).format(
            sex=_t(result.sex, language), age=decimal_comma(f"{result.age:g}", language)
        )
    )
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_nipts(
    result: NiptsResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Median NIPTS spectrum with the 10-90 % fractile band (ISO 1999).

    :param result: A :class:`~phonometry.hearing.noise_induced_hearing_loss.NiptsResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the median line ``plot``.
    :return: The axes.
    """
    from .._i18n import decimal_comma, fmt_minus, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    median = np.asarray(result.median, dtype=np.float64)
    du = np.asarray(result.spread_upper, dtype=np.float64)
    dl = np.asarray(result.spread_lower, dtype=np.float64)

    _fractile_band(
        ax,
        freqs,
        median,
        dl,
        du,
        color=_C_SECONDARY,
        floor=0.0,
        language=language,
    )
    kwargs.setdefault("color", _C_SECONDARY)
    kwargs.setdefault("label", _t("Median $N_{50}$", language))
    ax.plot(freqs, median, "o-", **kwargs)
    if abs(result.fractile - 0.5) > _MEDIAN_FRACTILE_EPS:
        ax.plot(
            freqs,
            np.asarray(result.value, dtype=np.float64),
            "s--",
            color=_C_REFERENCE,
            label=_t(_FRACTILE_LABEL, language).format(
                v=decimal_comma(f"{result.fractile:g}", language)
            ),
        )
    _freq_axis(ax, freqs, language=language)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_NIPTS_LABEL, language))
    ax.invert_yaxis()  # audiogram convention: worse hearing downward
    # l_ex carries no lower bound (only a domain warning), so sign it with
    # fmt_minus: an ASCII hyphen here would be shorter than the U+2212 the
    # axis ticks beside it already draw. The duration is validated positive.
    ax.set_title(
        _t(
            r"ISO 1999 NIPTS — $L_\mathrm{{EX,8h}}$ = {lex} dB, {years} yr", language
        ).format(
            lex=decimal_comma(fmt_minus(result.l_ex, "g"), language),
            years=decimal_comma(f"{result.years:g}", language),
        )
    )
    ax.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_htlan(
    result: HtlanResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Age, noise and combined hearing threshold components (ISO 1999, 6.1).

    :param result: A :class:`~phonometry.hearing.noise_induced_hearing_loss.HtlanResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the combined-threshold line ``plot``.
    :return: The axes.
    """
    from .._i18n import decimal_comma, fmt_minus, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    ax.plot(
        freqs,
        np.asarray(result.htla, dtype=np.float64),
        "o-",
        color=_C_PRIMARY,
        label=_t("Age (HTLA, ISO 7029)", language),
    )
    ax.plot(
        freqs,
        np.asarray(result.nipts, dtype=np.float64),
        "^-",
        color=_C_SECONDARY,
        label=_t("Noise (NIPTS)", language),
    )
    kwargs.setdefault("color", _C_REFERENCE)
    kwargs.setdefault("label", _t("Age + noise (HTLAN)", language))
    ax.plot(freqs, np.asarray(result.threshold, dtype=np.float64), "s--", **kwargs)
    _freq_axis(ax, freqs, language=language)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Hearing threshold level [dB]", language))
    ax.invert_yaxis()  # audiogram convention: worse hearing downward
    ax.set_title(
        _t("ISO 1999 HTLAN — {sex}, age {age}, {lex} dB / {years} yr", language).format(
            sex=_t(result.sex, language),
            age=decimal_comma(f"{result.age:g}", language),
            lex=decimal_comma(fmt_minus(result.l_ex, "g"), language),
            years=decimal_comma(f"{result.years:g}", language),
        )
    )
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_occupational_exposure(
    result: ExposureResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Per-task contributions to the daily exposure level (ISO 9612).

    One bar per task (its contribution to ``LEX,8h``), with the combined
    ``LEX,8h`` and the one-sided upper limit ``LEX,8h + U`` as horizontal
    lines.

    :param result: An
        :class:`~phonometry.hearing.occupational_exposure.ExposureResult` from the
        task-based strategy (the one that carries per-task contributions).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the task :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    :raises ValueError: If the result carries no per-task contributions.
    """
    from .._i18n import format_number, localize_axes

    if not result.tasks:
        msg = (
            "plot() needs per-task contributions; only task_based_exposure() "
            "results carry them (the job/full-day strategies do not)."
        )
        raise ValueError(msg)
    ax = ax if ax is not None else _new_axes()
    contributions = [t.lex_8h_contribution for t in result.tasks]
    labels = [t.label for t in result.tasks]
    positions = np.arange(len(contributions), dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.bar(positions, contributions, **kwargs)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.axhline(
        result.lex_8h,
        color=_C_REFERENCE,
        ls="--",
        label=r"$L_\mathrm{EX,8h}$ = "
        + format_number(result.lex_8h, language, decimals=1)
        + " dB",
    )
    ax.axhline(
        result.upper_limit,
        color=_C_MUTED,
        ls=":",
        label=r"$L_\mathrm{EX,8h} + U$ = "
        + format_number(result.upper_limit, language, decimals=1)
        + " dB",
    )
    top = max(result.upper_limit, max(contributions))
    bottom = min(0.0, min(contributions))
    ax.set_ylim(bottom * 1.12 if bottom < 0.0 else 0.0, top * 1.12)
    ax.set_ylabel(_t("A-weighted level [dB]", language))
    ax.set_title(
        _t(
            r"ISO 9612 daily noise exposure — $L_\mathrm{{EX,8h}}$ = {lex} dB ($U$ = {u} dB)",
            language,
        ).format(
            lex=format_number(result.lex_8h, language, decimals=1),
            u=format_number(result.expanded_uncertainty, language, decimals=1),
        )
    )
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    # localize_axes leaves the categorical task-label axis (a FuncFormatter) alone.
    localize_axes(ax, language)
    return ax


def plot_assumed_protection(
    result: AssumedProtectionResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Assumed protection values against the distribution they came from.

    Draws the mean attenuation with its standard deviation shaded either side,
    and the assumed protection value on top, so the gap Formula (1) opens
    between the two is the picture. Works for
    :class:`~phonometry.hearing.hearing_protectors.AssumedProtectionResult`.

    :param result: An assumed-protection result.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``APV`` curve.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    mean = np.asarray(result.mean_attenuation, dtype=np.float64)
    spread = np.asarray(result.standard_deviation, dtype=np.float64)
    _freq_axis(ax, freqs, language=language)
    ax.fill_between(
        freqs,
        mean - spread,
        mean + spread,
        color=theme_fill(_C_PRIMARY, ax),
        zorder=0,
        label=_t(_SPREAD_LABEL, language),
    )
    ax.plot(
        freqs,
        mean,
        "-o",
        color=_C_PRIMARY,
        lw=2.0,
        ms=4,
        zorder=3,
        label=_t("mean attenuation $m_f$", language),
    )
    apv_kwargs = dict(kwargs)
    apv_kwargs.setdefault(
        "label",
        _t("assumed protection $APV_{{f{x}}}$", language).format(x=result.performance),
    )
    apv_kwargs.setdefault("color", _C_SECONDARY)
    apv_kwargs.setdefault("linewidth", 2.4)
    ax.plot(
        freqs,
        np.asarray(result.apv, dtype=np.float64),
        "--s",
        ms=4,
        zorder=4,
        **apv_kwargs,
    )
    ax.set_ylabel(_t(_ATTENUATION_LABEL, language))
    ax.set_title(
        _t("ISO 4869-2 assumed protection values — {x} % performance", language).format(
            x=result.performance
        )
    )
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_hml_rating(
    result: HMLRatingResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """The ``HML`` two-segment line, over the reference noises behind it.

    Draws the predicted noise level reduction Formulas (16) and (17) give as a
    function of ``LpC - LpA``, with the three anchors marked and the eight
    reference noises of Table 2 scattered at their own differences. Works for
    :class:`~phonometry.hearing.hearing_protectors.HMLRatingResult`.

    :param result: An ``HML`` rating result.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``PNR`` curve.
    :return: The axes.
    """
    from .._i18n import localize_axes
    from ..hearing.hearing_protectors import HML_REFERENCE_C_MINUS_A

    ax = ax if ax is not None else _new_axes()
    high, medium, low = result.reported
    # The published triple is what the two segments are built from, so the
    # drawn line is the one a user of the rating would apply.
    left = np.linspace(-4.0, 2.0, 2)
    right = np.linspace(2.0, 12.0, 2)
    curve_kwargs = dict(kwargs)
    curve_kwargs.setdefault("color", _C_PRIMARY)
    curve_kwargs.setdefault("linewidth", 2.4)
    curve_kwargs.setdefault(
        "label", _t(_HML_CURVE_LABEL, language).format(h=high, m=medium, l=low)
    )
    ax.plot(left, medium - (high - medium) / 4.0 * (left - 2.0), **curve_kwargs)
    # The two segments are one line with a corner, so the second takes every
    # option the first did. Only the label is dropped, to keep one legend
    # entry for what the reader sees as a single curve.
    right_kwargs = {k: v for k, v in curve_kwargs.items() if k != "label"}
    ax.plot(right, medium - (medium - low) / 8.0 * (right - 2.0), **right_kwargs)
    ax.plot(
        [-2.0, 2.0, 10.0],
        [high, medium, low],
        "o",
        color=_C_SECONDARY,
        ms=7,
        zorder=4,
        label=_t(r"$H$, $M$, $L$ anchors", language),
    )
    differences = np.asarray(HML_REFERENCE_C_MINUS_A, dtype=np.float64)
    ax.plot(
        np.repeat(differences, result.predicted_reduction.shape[0]),
        result.predicted_reduction.T.reshape(-1),
        ".",
        color=_C_MUTED,
        ms=3,
        alpha=0.55,
        zorder=1,
        label=_t("reference noises (Table 2)", language),
    )
    ax.set_xlabel(_t(_C_MINUS_A_LABEL, language))
    ax.set_ylabel(_t(_REDUCTION_LABEL, language))
    ax.set_title(_t(_HML_TITLE, language).format(h=high, m=medium, l=low))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_snr_rating(
    result: SNRRatingResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """The per-subject ratings the single number was reduced from.

    Draws ``SNRj`` for each test subject as a bar with the mean and the
    reported single number across them, so the spread Formula (19) subtracts
    is visible. Works for
    :class:`~phonometry.hearing.hearing_protectors.SNRRatingResult`.

    :param result: An ``SNR`` rating result.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the per-subject bars.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    per_subject = np.asarray(result.subject_snr, dtype=np.float64)
    positions = np.arange(1, per_subject.size + 1)
    bar_kwargs = dict(kwargs)
    bar_kwargs.setdefault("color", _C_PRIMARY)
    bar_kwargs.setdefault("label", _t("per subject", language))
    ax.bar(positions, per_subject, width=0.7, zorder=2, **bar_kwargs)
    ax.axhline(
        result.mean,
        color=_C_SECONDARY,
        ls="--",
        lw=1.6,
        zorder=3,
        label=f"$SNR_m$ = {result.mean:.1f} dB",
    )
    ax.axhline(
        result.reported,
        color=_C_REFERENCE,
        ls="-",
        lw=1.8,
        zorder=3,
        label=f"$SNR_{{{result.performance}}}$ = {result.reported} dB",
    )
    ax.set_xticks(positions)
    ax.set_xlabel(_t(_SUBJECT_LABEL, language))
    ax.set_ylabel(_t(_REDUCTION_LABEL, language))
    ax.set_title(
        _t("ISO 4869-2 single number rating — $SNR$ = {snr} dB", language).format(
            snr=result.reported
        )
    )
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_protected_level(
    result: ProtectedLevelResult,
    ax: Axes | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """The A-weighted band levels the protector leaves at the ear.

    Only the octave-band method sees a spectrum, so this draws its per-band
    result with the total marked. Works for
    :class:`~phonometry.hearing.hearing_protectors.ProtectedLevelResult`.

    :param result: An octave-band protected-level result.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the band bars.
    :return: The axes.
    :raises ValueError: for an ``HML`` or ``SNR`` result, which carries no
        spectrum.
    """
    from .._i18n import localize_axes

    if result.band_levels is None or result.frequencies is None:
        msg = (
            f"The {result.method} method has no spectrum to draw: it answers "
            "from the C- and A-weighted levels alone. Only the octave-band "
            "method carries per-band results."
        )
        raise ValueError(msg)
    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    positions = np.arange(freqs.size)
    bar_kwargs = dict(kwargs)
    bar_kwargs.setdefault("color", _C_PRIMARY)
    bar_kwargs.setdefault("label", _t("protected band level", language))
    ax.bar(
        positions,
        np.asarray(result.band_levels, dtype=np.float64),
        width=0.7,
        **bar_kwargs,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_BAND_LEVEL_LABEL, language))
    performance = result.performance if result.performance is not None else ""
    ax.set_title(
        _t(
            "ISO 4869-2 octave-band method — $L'_{{p,A{x}}}$ = {level} dB", language
        ).format(x=performance, level=result.reported_level)
    )
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax
