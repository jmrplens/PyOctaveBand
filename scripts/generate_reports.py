#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Generate the committed example ``.report()`` fiches.

Mirrors :mod:`scripts.generate_graphs` for the normative report PDFs: it builds
a small set of representative results with example metadata and writes their
``.report()`` fiche to ``.github/reports/`` so the repository always carries an
up-to-date rendered example of every report kind, which the documentation links
to. Alongside every ``<name>.pdf`` it rasterizes the first page to a
``<name>.webp`` preview (via :mod:`pypdfium2`, no system Poppler needed) so the
website and the GitHub docs can show the fiche inline without a PDF viewer or a
build-time rasterizer. The preview is lossless WebP: pixel-identical to the
raster and roughly half the size of the optimized PNG it replaces (flat-color
document pages compress better lossless than lossy). Run it with
``make reports`` or ``python scripts/generate_reports.py``.

This file is the command line of the :mod:`reports` package and the registry
that orders it: the builders live in ``scripts/reports/``, grouped by the
subject of the result each fiche prints, and :data:`_FICHES` below names the
file every one of them writes. The first import is :mod:`reports` itself, whose
``__init__`` pins the numerical thread pools before numpy or scipy can import a
multi-threaded backend; that is what keeps the rendered plot identical across
machines, so nothing numerical may be imported above it.

Neither the PDFs nor the WebP previews are byte-compared in CI: the embedded
plot is vector geometry whose floating-point coordinates differ by ~1 ULP
across CPUs, and the raster inherits that. The ``Example report fiches up to
date`` job regenerates the set and compares it within a tolerance instead
(:mod:`scripts.check_reports`); the fiches went unchecked for months before
that job existed, and two of them fell a plot-styling release behind
unnoticed.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence

# The fiche builders live in the ``reports`` package beside this file. Running
# this script puts ``scripts/`` on the import path for free, but loading the
# file by its path (as the test suite does) does not, so the directory is added
# here and both entry points reach the same package.
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import reports  # noqa: F401  (its __init__ pins the thread pools first)
from reports.building import (
    _airborne_example,
    _field_airborne_example,
    _field_facade_example,
    _field_impact_example,
    _flanking_impact_level_example,
    _flanking_level_difference_example,
    _floor_covering_example,
    _impact_example,
    _intensity_element_example,
    _intensity_example,
    _lab_airborne_example,
    _lab_impact_example,
    _survey_airborne_example,
    _survey_facade_example,
    _survey_impact_example,
    _vibration_reduction_example,
)
from reports.building_design import (
    _airborne_prediction_example,
    _detailed_airborne_example,
    _detailed_impact_example,
    _facade_prediction_example,
    _impact_prediction_example,
    _installed_structure_borne_example,
    _structure_borne_power_example,
)
from reports.devices import (
    _filter_class_1995_example,
    _filter_class_example,
    _intensity_class_example,
    _loudspeaker_example,
    _microphone_example,
)
from reports.emission import (
    _epnl_example,
    _intensity_sound_power_example,
    _iso4871_declaration_example,
    _precision_sound_power_example,
    _reverberation_sound_power_example,
    _sound_power_example,
    _vibration_sound_power_example,
)
from reports.environment import (
    _barrier_insertion_loss_example,
    _impulse_prominence_example,
    _outdoor_attenuation_example,
    _rd1367_example,
    _tone_audibility_example,
    _wind_turbine_tonality_example,
)
from reports.materials import (
    _absorption_example,
    _airflow_resistance_example,
    _diffusion_example,
    _diffusion_polar_example,
    _dynamic_stiffness_example,
    _impedance_tube_example,
    _scattering_example,
    _sound_absorption_example,
)
from reports.noise_control import (
    _duct_path_example,
    _enclosure_example,
    _hvac_example,
    _silencer_example,
)
from reports.perception import (
    _htlan_example,
    _loudness_example,
    _nipts_example,
    _occupational_exposure_example,
    _program_loudness_example,
    _sii_example,
    _sti_example,
)
from reports.room import (
    _enclosed_space_absorption_example,
    _noise_criteria_example,
    _open_plan_example,
    _reverberation_prediction_example,
    _room_acoustics_example,
    _room_criteria_example,
)
from reports.vibration import (
    _human_vibration_example,
    _mechanical_mobility_example,
    _multiple_shock_example,
    _transfer_stiffness_example,
)

from phonometry import ReportMetadata

#: Committed output directory for the example fiches.
_DEFAULT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", ".github", "reports")
)

#: Pixel width of the rendered WebP preview. A4 portrait at this width is ~1415
#: px tall: crisp on the docs pages (shown at ~80 % column width) yet small
#: enough to commit. The height follows from the page aspect ratio.
_PREVIEW_WIDTH_PX = 1000


#: Every example fiche the repository keeps rendered, from the file it writes
#: to the factory that builds it. New report kinds add an entry here so
#: ``make reports`` regenerates the full set. Keyed by filename so that the set
#: of rendered outputs can be read off the registry: naming it is what a caller
#: asking "is every registered fiche committed?" needs, and calling the factory
#: for it would run a full computation and build a plot for an answer already
#: written here.
_FICHES: dict[str, Callable[[], tuple[object, ReportMetadata, str]]] = {
    "iso717_airborne_example.pdf": _airborne_example,
    "iso717_impact_example.pdf": _impact_example,
    "iso16283_airborne_example.pdf": _field_airborne_example,
    "iso16283_impact_example.pdf": _field_impact_example,
    "iso10140_airborne_example.pdf": _lab_airborne_example,
    "iso10140_impact_example.pdf": _lab_impact_example,
    "iso15186_intensity_example.pdf": _intensity_example,
    "iso15186_element_example.pdf": _intensity_element_example,
    "iso12354_airborne_prediction_example.pdf": _airborne_prediction_example,
    "iso12354_impact_prediction_example.pdf": _impact_prediction_example,
    "iso12354_facade_prediction_example.pdf": _facade_prediction_example,
    "iso16251_floor_covering_example.pdf": _floor_covering_example,
    "iso11654_absorption_example.pdf": _absorption_example,
    "iso354_absorption_example.pdf": _sound_absorption_example,
    "iso10534_impedance_tube_example.pdf": _impedance_tube_example,
    "iso532_loudness_example.pdf": _loudness_example,
    "ebu_r128_loudness_example.pdf": _program_loudness_example,
    "iso1996_tone_audibility_example.pdf": _tone_audibility_example,
    "ntacou112_impulse_prominence_example.pdf": _impulse_prominence_example,
    "iec61400_wind_turbine_tonality_example.pdf": _wind_turbine_tonality_example,
    "icao_epnl_example.pdf": _epnl_example,
    "iec61260_filter_example.pdf": _filter_class_example,
    "iec61260_filter_1995_example.pdf": _filter_class_1995_example,
    "iec61043_intensity_example.pdf": _intensity_class_example,
    "iso4871_declaration_example.pdf": _iso4871_declaration_example,
    "iec60268_5_loudspeaker_example.pdf": _loudspeaker_example,
    "iec60268_4_microphone_example.pdf": _microphone_example,
    "iso9612_exposure_example.pdf": _occupational_exposure_example,
    "human_vibration_example.pdf": _human_vibration_example,
    "iso1999_nipts_example.pdf": _nipts_example,
    "iso1999_htlan_example.pdf": _htlan_example,
    "iso3382_room_acoustics_example.pdf": _room_acoustics_example,
    "reverberation_prediction_example.pdf": _reverberation_prediction_example,
    "enclosed_space_absorption_example.pdf": _enclosed_space_absorption_example,
    "ansi_s12_2_noise_criteria_example.pdf": _noise_criteria_example,
    "ansi_s12_2_room_criteria_example.pdf": _room_criteria_example,
    "iso3382_3_open_plan_example.pdf": _open_plan_example,
    "iso2631_5_multiple_shock_example.pdf": _multiple_shock_example,
    "iso7626_mobility_example.pdf": _mechanical_mobility_example,
    "iso10846_transfer_stiffness_example.pdf": _transfer_stiffness_example,
    "iso3744_sound_power_example.pdf": _sound_power_example,
    "iso9614_sound_power_intensity_example.pdf": _intensity_sound_power_example,
    "iso3741_reverberation_power_example.pdf": _reverberation_sound_power_example,
    "iso7849_vibration_power_example.pdf": _vibration_sound_power_example,
    "en15657_structure_borne_power_example.pdf": _structure_borne_power_example,
    "en12354_5_installed_structure_borne_example.pdf": _installed_structure_borne_example,
    "iso17497_scattering_example.pdf": _scattering_example,
    "iso17497_diffusion_example.pdf": _diffusion_example,
    "iso17497_diffusion_polar_example.pdf": _diffusion_polar_example,
    "en29052_dynamic_stiffness_example.pdf": _dynamic_stiffness_example,
    "iso9053_airflow_resistance_example.pdf": _airflow_resistance_example,
    "iso10848_kij_example.pdf": _vibration_reduction_example,
    "iso10848_dnf_example.pdf": _flanking_level_difference_example,
    "iso10848_lnf_example.pdf": _flanking_impact_level_example,
    "iso10052_airborne_example.pdf": _survey_airborne_example,
    "iso10052_impact_example.pdf": _survey_impact_example,
    "iso10052_facade_example.pdf": _survey_facade_example,
    "iso16283_facade_example.pdf": _field_facade_example,
    "iso9613_outdoor_attenuation_example.pdf": _outdoor_attenuation_example,
    "iso9613_barrier_insertion_loss_example.pdf": _barrier_insertion_loss_example,
    "iec60268_16_sti_example.pdf": _sti_example,
    "ansi_s3_5_sii_example.pdf": _sii_example,
    "enclosure_insertion_loss_example.pdf": _enclosure_example,
    "reactive_silencer_example.pdf": _silencer_example,
    "hvac_duct_noise_example.pdf": _hvac_example,
    "duct_path_example.pdf": _duct_path_example,
    "rd1367_activity_example.pdf": _rd1367_example,
    "iso12354_detailed_airborne_example.pdf": _detailed_airborne_example,
    "iso12354_detailed_impact_example.pdf": _detailed_impact_example,
    "iso3745_precision_power_example.pdf": _precision_sound_power_example,
}

#: The registered factories alone, in generation order.
_EXAMPLES: list[Callable[[], tuple[object, ReportMetadata, str]]] = list(
    _FICHES.values()
)


def preview_path_for(pdf_path: str) -> str:
    """Return the WebP-preview path that pairs with ``pdf_path`` (``.pdf`` -> ``.webp``)."""
    return os.path.splitext(pdf_path)[0] + ".webp"


def _write_preview(pdf_path: str) -> str:
    """Rasterize the first page of ``pdf_path`` to a lossless WebP beside it.

    The preview is what the documentation embeds inline; it is not byte-compared
    in CI (the raster inherits the vector plot's ~1 ULP cross-CPU drift), but
    :mod:`scripts.check_reports` compares it within a tolerance, so a stale
    preview fails the build like a stale figure. Lossless WebP keeps the
    preview pixel-identical to the raster at roughly half the optimized-PNG
    size; ``method=6`` is the slowest, most exhaustive encoder search, whose
    output is fixed by the input pixels (byte-stable across runs).
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[0]
        width_pt, _ = page.get_size()
        scale = _PREVIEW_WIDTH_PX / width_pt
        image = page.render(scale=scale).to_pil()
        preview = preview_path_for(pdf_path)
        # RGB (no alpha): the fiche is opaque white; a smaller, simpler image.
        image.convert("RGB").save(preview, "WEBP", lossless=True, quality=100, method=6)
    finally:
        pdf.close()
    return preview


def generate_reports(
    output_dir: str,
    examples: Sequence[Callable[[], tuple[object, ReportMetadata, str]]] | None = None,
) -> list[str]:
    """Write every example fiche (PDF + WebP preview) into ``output_dir``.

    ``examples`` restricts the run to a subset of the registered factories
    (the test suite renders one fiche per test); the default renders the
    full registry. Returns the PDF paths written; each has a paired
    ``.webp`` preview next to it (see :func:`preview_path_for`).
    """
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    for factory in _EXAMPLES if examples is None else examples:
        result, metadata, name = factory()
        path = os.path.join(output_dir, name)
        result.report(path, metadata=metadata)  # type: ignore[attr-defined]
        _write_preview(path)
        written.append(path)
    return written


def main() -> None:
    """Generate the example fiches into the committed directory (or ``--output-dir``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=_DEFAULT_DIR)
    args = parser.parse_args()
    for path in generate_reports(os.path.abspath(args.output_dir)):
        print(f"wrote {path}")
        print(f"wrote {preview_path_for(path)}")


if __name__ == "__main__":
    main()
