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
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..hearing.noise_induced_hearing_loss import HtlanResult, NiptsResult
    from ..hearing.occupational_exposure import ExposureResult
    from ..hearing.threshold import AgeThresholdResult

#: Spanish translations of the fixed strings rendered by the hearing
#: ``.plot()`` renderers, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
#: Labels the renderers repeat; the Spanish table is keyed by the same
#: constants, so a label is written once.
_FREQ_LABEL = "Frequency [Hz]"
_NIPTS_LABEL = "NIPTS [dB]"
_FRACTILE_LABEL = "Fractile {v}"

_STRINGS: dict[str, str] = {
    _FREQ_LABEL: "Frecuencia [Hz]",
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
    if abs(result.fractile - 0.5) > 1e-9:
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
    if abs(result.fractile - 0.5) > 1e-9:
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
