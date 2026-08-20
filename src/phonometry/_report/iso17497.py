#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 17497-1/-2 surface-scattering and diffusion fiches (reportlab renderer).

Renders the two sound-scattering surface descriptors of ISO 17497 to one-page
PDFs laid out like accredited test reports:

* :func:`render_scattering_report` renders a
  :class:`~phonometry.materials.diffusers.reverberation_room_scattering.ScatteringResult` as a
  random-incidence *scattering* coefficient report (ISO 17497-1:2004+A1:2014):
  the standard-basis line, an optional metadata header block, a two-panel body
  with the per-band table (frequency, the random-incidence absorption
  ``alpha_s`` and the scattering coefficient ``s``) beside the ``s(f)`` curve on
  a categorical one-third-octave band axis, a boxed characterisation headline
  and the footer identity/disclaimer block. ``verbose`` adds the specular
  absorption ``alpha_spec`` column.

* :func:`render_diffusion_spectrum_report` renders a
  :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionSpectrum` as a
  directional *diffusion* coefficient report (ISO 17497-2:2012, Clause 8.5):
  the per-band table (frequency, the directional diffusion coefficient ``d``
  and, when present, the normalised ``d_n``) beside the ``d(f)`` band-axis
  curve. ``verbose`` adds the normalised ``d_n`` column when it was measured.

* :func:`render_diffusion_polar_report` renders a
  :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionResult` (the polar
  response of a single source position) as the Clause 8.5 polar-response report:
  the corrected receiver-angle / reflected-level table beside the semicircular
  polar plot, with the directional diffusion coefficient ``d_theta`` boxed.

Both parts of ISO 17497 are characterisations, so these fiches carry no
pass/fail verdict and no single-number rating (the scattering and diffusion
coefficients answer different questions and are not interchangeable).

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the ISO 17497 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import functools
import html
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _REPORTLAB_HINT,
    band_table,
    band_table_header_style,
    build_document,
    document_styles,
    fmt_meta,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
)

if TYPE_CHECKING:
    from ..materials.diffusers.reverberation_room_scattering import ScatteringResult
    from ..materials.diffusers.scattering_diffusion import (
        DiffusionResult,
        DiffusionSpectrum,
    )
    from .metadata import ReportMetadata

#: Header of the band centre-frequency column of every ISO 17497 table.
_FREQUENCY_COLUMN = "f [Hz]"


def _c2(value: float, language: str = "en") -> str:
    """Two decimals (coefficients ``s``, ``d``; ISO 17497 rounds to 0,01)."""
    return format_number(value, language, decimals=2)


def _d1(value: float, language: str = "en") -> str:
    """One decimal (polar reflected levels; ISO 17497-2 rounds to 0,1 dB)."""
    return format_number(value, language, decimals=1)


def _common_metadata_pairs(
    metadata: ReportMetadata | None,
    freq_range: str | None,
    language: str = "en",
    *,
    include_room_fields: bool = True,
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs shared by the ISO 17497 fiches.

    The descriptive fields (client, specimen, mounting, room, climate ...) come
    from the :class:`ReportMetadata` when one is supplied; the measured frequency
    range is taken from the result. Only fields that are set are returned, so
    empty rows never appear. The frequency range is a label-safe formatted
    string; the remaining values are user free text and are XML-escaped so a
    stray ``&`` or ``<`` cannot break reportlab's ``Paragraph`` parser.

    ``include_room_fields`` controls the reverberation-room quantities (sample
    area ``S`` and room volume ``V``): they are shown for the ISO 17497-1
    scattering fiche but suppressed for the ISO 17497-2 diffusion fiches, which
    are free-field methods where those fields do not apply (so a metadata object
    reused across a Part 1 and a Part 2 campaign never leaks them into the
    diffusion sheet).
    """

    def _md(name: str) -> Any:
        return getattr(metadata, name) if metadata is not None else None

    area = _md("area") if include_room_fields else None
    room_volume = _md("room_volume") if include_room_fields else None
    temperature = _md("temperature")
    humidity = _md("relative_humidity")
    pressure = _md("pressure")

    freq_label = t("Frequency range [Hz]", language)
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), _md("client")),
        (t("Manufacturer", language), _md("manufacturer")),
        (t("Description", language), _md("specimen")),
        (
            t("Sample area S [m<super>2</super>]", language),
            fmt_meta(area, language) if area is not None else None,
        ),
        (
            t("Room volume V [m<super>3</super>]", language),
            fmt_meta(room_volume, language) if room_volume is not None else None,
        ),
        (freq_label, freq_range),
        (t("Mounting", language), _md("mounting")),
        (t("Test room", language), _md("test_room")),
        (t("Date of test", language), _md("test_date")),
        (
            t("Temperature [&#176;C]", language),
            fmt_meta(temperature, language) if temperature is not None else None,
        ),
        (
            t("Relative humidity [%]", language),
            fmt_meta(humidity, language) if humidity is not None else None,
        ),
        (
            t("Ambient pressure [kPa]", language),
            fmt_meta(pressure, language) if pressure is not None else None,
        ),
    ]
    return [
        (label, value if label == freq_label else html.escape(str(value)))
        for label, value in specs
        if value is not None
    ]


def _freq_range(freqs: np.ndarray, language: str) -> str | None:
    """The ``{lo} to {hi}`` frequency-range string, or ``None`` when empty."""
    if not freqs.size:
        return None
    return t("{lo} to {hi}", language).format(
        lo=round(float(freqs.min())), hi=round(float(freqs.max()))
    )


def _basis_line(
    metadata: ReportMetadata | None,
    with_standard: str,
    without_standard: str,
    language: str,
) -> str:
    """The standard-basis line, naming the measurement standard when supplied."""
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        return t(with_standard, language).format(
            standard=html.escape(measurement_standard)
        )
    return t(without_standard, language)


def _header_flow(
    title: str,
    basis: str,
    header_pairs: list[tuple[str, str]],
    title_style: Any,
    basis_style: Any,
) -> list[Any]:
    """The shared title/basis/metadata-grid opening of an ISO 17497 fiche."""
    from reportlab.platypus import Spacer

    from ._layout import fiche_paragraph

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(basis, basis_style),
    ]
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))
    return flow


# ---------------------------------------------------------------------------
# ISO 17497-1: random-incidence scattering coefficient.
# ---------------------------------------------------------------------------
def _scattering_table(
    result: ScatteringResult, verbose: bool, language: str = "en"
) -> Any:
    """Build the per-band ``f | alpha_s | s`` table (``alpha_spec`` if verbose)."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    head_style = band_table_header_style()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    a_s = np.asarray(result.random_incidence, dtype=np.float64)
    s = np.asarray(result.scattering, dtype=np.float64)
    if verbose:
        spec = np.asarray(result.specular, dtype=np.float64)
        header = [
            fiche_paragraph(t(_FREQUENCY_COLUMN, language), head_style),
            fiche_paragraph("&#945;<sub>s</sub>", head_style),
            fiche_paragraph("&#945;<sub>spec</sub>", head_style),
            fiche_paragraph("s", head_style),
        ]
        rows: list[list[Any]] = [header]
        for fk, ak, spk, sk in zip(freqs, a_s, spec, s):
            rows.append(
                [
                    f"{round(fk)}",
                    _c2(ak, language),
                    _c2(spk, language),
                    _c2(sk, language),
                ]
            )
        col_widths = [22 * mm, 22 * mm, 24 * mm, 20 * mm]
    else:
        header = [
            fiche_paragraph(t(_FREQUENCY_COLUMN, language), head_style),
            fiche_paragraph("&#945;<sub>s</sub>", head_style),
            fiche_paragraph("s", head_style),
        ]
        rows = [header]
        for fk, ak, sk in zip(freqs, a_s, s):
            rows.append(
                [
                    f"{round(fk)}",
                    _c2(ak, language),
                    _c2(sk, language),
                ]
            )
        col_widths = [20 * mm, 18 * mm, 18 * mm]
    return band_table(rows, col_widths, len(freqs))


def render_scattering_report(
    result: ScatteringResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 17497-1 scattering-coefficient fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.materials.diffusers.reverberation_room_scattering.ScatteringResult`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` renders the body
        with only the measured frequency range. The ``requirement`` field is
        ignored (ISO 17497-1 has no verdict).
    :param verbose: When ``True``, the value table adds the specular absorption
        ``alpha_spec`` column.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    s = np.asarray(result.scattering, dtype=np.float64)
    if freqs.shape != s.shape:
        msg = (
            "render_scattering_report() needs 'frequencies' and 'scattering' "
            "of equal length."
        )
        raise ValueError(msg)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Surface scattering measurement", language)
    basis = _basis_line(
        metadata,
        "{standard} reverberation-room measurement of the random-incidence "
        "scattering coefficient per ISO 17497-1:2004+A1:2014.",
        "Reverberation-room measurement of the random-incidence scattering "
        "coefficient per ISO 17497-1:2004+A1:2014.",
        language,
    )
    header_pairs = _common_metadata_pairs(
        metadata, _freq_range(freqs, language), language
    )
    flow = _header_flow(title, basis, header_pairs, title_style, basis_style)

    from .._plot.materials import plot_scattering_report

    caption = (
        t(
            "One-third-octave &#945;<sub>s</sub>, &#945;<sub>spec</sub> and "
            "scattering coefficient s",
            language,
        )
        if verbose
        else t(
            "One-third-octave &#945;<sub>s</sub> and scattering coefficient s", language
        )
    )
    left_cell = [
        fiche_paragraph(caption, caption_style),
        _scattering_table(result, verbose, language),
    ]
    plot_fn = functools.partial(plot_scattering_report, result)
    if verbose:
        plot_drawing = render_figure_drawing(
            plot_fn, 84 * mm, y_top=None, language=language
        )
        flow.append(
            two_panel_body(
                left_cell, plot_drawing, left_width_mm=90.0, plot_width_mm=84.0
            )
        )
    else:
        plot_drawing = render_figure_drawing(
            plot_fn, 116 * mm, y_top=None, language=language
        )
        flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    lo = round(float(freqs.min())) if freqs.size else 0
    hi = round(float(freqs.max())) if freqs.size else 0
    statement = t(
        "Random-incidence scattering coefficient <b>s</b>, {lo} Hz to {hi} Hz",
        language,
    ).format(lo=lo, hi=hi)
    flow.append(result_box(statement, styles, accent))
    flow.extend(footer_flow(metadata, language))
    return build_document(path, flow, title)


# ---------------------------------------------------------------------------
# ISO 17497-2: directional diffusion coefficient spectrum d(f).
# ---------------------------------------------------------------------------
def _diffusion_table(
    result: DiffusionSpectrum, show_normalized: bool, language: str = "en"
) -> Any:
    """Build the per-band ``f | d`` table (adds ``d_n`` when requested)."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    head_style = band_table_header_style()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    d = np.asarray(result.diffusion, dtype=np.float64)
    if show_normalized and result.normalized is not None:
        d_n = np.asarray(result.normalized, dtype=np.float64)
        header = [
            fiche_paragraph(t(_FREQUENCY_COLUMN, language), head_style),
            fiche_paragraph("d", head_style),
            fiche_paragraph("d<sub>n</sub>", head_style),
        ]
        rows: list[list[Any]] = [header]
        for fk, dk, dnk in zip(freqs, d, d_n):
            rows.append(
                [
                    f"{round(fk)}",
                    _c2(dk, language),
                    _c2(dnk, language),
                ]
            )
        col_widths = [20 * mm, 18 * mm, 18 * mm]
    else:
        header = [
            fiche_paragraph(t(_FREQUENCY_COLUMN, language), head_style),
            fiche_paragraph("d", head_style),
        ]
        rows = [header]
        for fk, dk in zip(freqs, d):
            rows.append([f"{round(fk)}", _c2(dk, language)])
        col_widths = [28 * mm, 28 * mm]
    return band_table(rows, col_widths, len(freqs))


def render_diffusion_spectrum_report(
    result: DiffusionSpectrum,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 17497-2 diffusion-coefficient fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionSpectrum`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` renders the body
        with only the measured frequency range. The ``requirement`` field is
        ignored (ISO 17497-2 has no verdict).
    :param verbose: When ``True`` and a normalised spectrum is present, the
        value table adds the normalised ``d_n`` column.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    d = np.asarray(result.diffusion, dtype=np.float64)
    if freqs.shape != d.shape:
        msg = (
            "render_diffusion_spectrum_report() needs 'frequencies' and "
            "'diffusion' of equal length."
        )
        raise ValueError(msg)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Surface diffusion measurement", language)
    basis = _basis_line(
        metadata,
        "{standard} free-field measurement of the directional diffusion "
        "coefficient per ISO 17497-2:2012.",
        "Free-field measurement of the directional diffusion coefficient per "
        "ISO 17497-2:2012.",
        language,
    )
    header_pairs = _common_metadata_pairs(
        metadata,
        _freq_range(freqs, language),
        language,
        include_room_fields=False,
    )
    flow = _header_flow(title, basis, header_pairs, title_style, basis_style)

    from .._plot.materials import plot_diffusion_report

    show_normalized = verbose and result.normalized is not None
    caption = (
        t(
            "One-third-octave diffusion coefficient d and normalised d<sub>n</sub>",
            language,
        )
        if show_normalized
        else t("One-third-octave diffusion coefficient d", language)
    )
    left_cell = [
        fiche_paragraph(caption, caption_style),
        _diffusion_table(result, show_normalized, language),
    ]
    plot_fn = functools.partial(plot_diffusion_report, result)
    plot_drawing = render_figure_drawing(
        plot_fn, 116 * mm, y_top=None, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    lo = round(float(freqs.min())) if freqs.size else 0
    hi = round(float(freqs.max())) if freqs.size else 0
    statement = t(
        "Diffusion coefficient <b>d</b>, {lo} Hz to {hi} Hz", language
    ).format(lo=lo, hi=hi)
    flow.append(result_box(statement, styles, accent))
    flow.extend(footer_flow(metadata, language))
    return build_document(path, flow, title)


# ---------------------------------------------------------------------------
# ISO 17497-2: single-source polar response.
# ---------------------------------------------------------------------------
def _polar_table(result: DiffusionResult, language: str = "en") -> Any:
    """Build the corrected ``angle | reflected level`` polar-response table."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    head_style = band_table_header_style()
    angles = np.asarray(result.angles, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.float64)
    header = [
        fiche_paragraph(t("Angle &#952; [&#176;]", language), head_style),
        fiche_paragraph("L [dB]", head_style),
    ]
    rows: list[list[Any]] = [header]
    for ang, lev in zip(angles, levels):
        rows.append([_d1(ang, language), _d1(lev, language)])
    return band_table(rows, [28 * mm, 28 * mm], len(angles))


def render_diffusion_polar_report(
    result: DiffusionResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 17497-2 polar-response fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionResult`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` renders the body
        and disclaimer only. The ``requirement`` field is ignored (ISO 17497-2
        has no verdict).
    :param verbose: Accepted for signature parity; the polar-response fiche has
        no extended table.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # signature parity: the polar fiche has no extended table
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    angles = np.asarray(result.angles, dtype=np.float64)
    levels = np.asarray(result.levels, dtype=np.float64)
    if angles.shape != levels.shape:
        msg = (
            "render_diffusion_polar_report() needs 'angles' and 'levels' of "
            "equal length."
        )
        raise ValueError(msg)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Surface diffusion measurement", language)
    basis = _basis_line(
        metadata,
        "{standard} free-field polar-response measurement of the directional "
        "diffusion coefficient per ISO 17497-2:2012.",
        "Free-field polar-response measurement of the directional diffusion "
        "coefficient per ISO 17497-2:2012.",
        language,
    )
    header_pairs = _common_metadata_pairs(
        metadata, None, language, include_room_fields=False
    )
    flow = _header_flow(title, basis, header_pairs, title_style, basis_style)

    from .._plot.materials import plot_diffusion_polar_report

    caption = t("Corrected polar response L per receiver angle", language)
    left_cell = [
        fiche_paragraph(caption, caption_style),
        _polar_table(result, language),
    ]
    plot_fn = functools.partial(plot_diffusion_polar_report, result)
    plot_drawing = render_figure_drawing(
        plot_fn,
        108 * mm,
        y_top=None,
        subplot_kw={"projection": "polar"},
        language=language,
    )
    flow.append(
        two_panel_body(left_cell, plot_drawing, left_width_mm=64.0, plot_width_mm=110.0)
    )
    flow.append(Spacer(1, 8))

    statement = t(
        "Directional diffusion coefficient <b>d</b> = {value}", language
    ).format(value=_c2(float(result.coefficient), language))
    flow.append(result_box(statement, styles, accent))
    flow.extend(footer_flow(metadata, language))
    return build_document(path, flow, title)
