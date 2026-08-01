#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The example-report generator keeps producing valid one-page fiches.

The rendered PDFs (and their WebP previews) under ``.github/reports/`` are not
byte-checked (their vector plot differs by ~1 ULP across CPUs, and the raster
inherits it); this test only guards that ``scripts/generate_reports.py`` still
runs and writes a valid single-page PDF plus a valid WebP preview for every
registered example. One test per registered example: each fiche is rendered
exactly once (PDF and preview asserted together on the same run) and the
examples spread across the xdist workers instead of forming one monolithic
test.
"""

from __future__ import annotations

import importlib.util
import pathlib
from collections.abc import Callable

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("svglib")
pytest.importorskip("pypdf")
pytest.importorskip("pypdfium2")
pytest.importorskip("PIL")

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "generate_reports.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("_generate_reports", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_generator()


def test_generator_registers_examples() -> None:
    assert _MODULE._EXAMPLES, "the generator registered no examples"
    # No factory is registered twice: a duplicate entry would render the
    # same fiche twice and hide a forgotten registration elsewhere. (The
    # filenames cannot collide, being the registry's own keys.)
    names = [factory.__name__ for factory in _MODULE._EXAMPLES]
    assert len(set(names)) == len(names)
    assert all(name.endswith(".pdf") for name in _MODULE._FICHES)


def test_every_registered_fiche_is_committed() -> None:
    """Registering a fiche and never committing it is a silent hole.

    One example was registered in the generator, documented, and left without
    its rendered PDF and preview in the tree for months: the documentation
    linked to files that were not there. The staleness job catches it too, but
    this needs no regeneration, so it fails in the same second as the commit
    that forgets it.

    Reads the filenames off the registry keys rather than calling each factory
    for the name it returns: the answer is a path lookup, and running all 67
    factories to get it would compute and plot every worked example first.
    ``test_each_example_writes_a_one_page_pdf_and_preview`` is what confirms a
    factory still writes the file it is registered under.
    """
    committed = pathlib.Path(_MODULE._DEFAULT_DIR)
    missing = []
    for name in _MODULE._FICHES:
        for path in (committed / name, committed / _MODULE.preview_path_for(name)):
            if not path.is_file():
                missing.append(path.name)
    assert not missing, (
        f"registered fiches with no committed render: {sorted(missing)} - "
        "run 'make reports' and commit the result"
    )


@pytest.mark.parametrize(
    "factory", _MODULE._EXAMPLES, ids=lambda factory: factory.__name__.strip("_")
)
def test_each_example_writes_a_one_page_pdf_and_preview(
    factory: Callable[..., object], tmp_path: pathlib.Path
) -> None:
    from PIL import Image
    from pypdf import PdfReader

    written = _MODULE.generate_reports(str(tmp_path), examples=[factory])
    assert len(written) == 1
    p = pathlib.Path(written[0])
    # The factory writes the file it is registered under. This is what lets
    # test_every_registered_fiche_is_committed trust the registry keys.
    assert _MODULE._FICHES.get(p.name) is factory
    assert p.is_file() and p.stat().st_size > 0
    with open(p, "rb") as handle:
        assert handle.read(4) == b"%PDF"
    assert len(PdfReader(str(p)).pages) == 1
    preview = pathlib.Path(_MODULE.preview_path_for(str(p)))
    assert preview.suffix == ".webp"
    assert preview.is_file() and preview.stat().st_size > 0
    with Image.open(preview) as image:
        assert image.format == "WEBP"
        assert image.width == _MODULE._PREVIEW_WIDTH_PX
