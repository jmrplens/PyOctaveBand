#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The glyph-outlining engine of the diagram plates, one test per risk.

The plates are committed byte for byte, so the engine's measurement and
emission are part of the repository's contract: pen advances that keep
boundary spaces, clusters that never split a combining mark from its
base, a deterministic fallback chain with a hard error past its end, and
an atlas whose ids are content-derived. The canary pins the private
matplotlib layout helper the shaping rides on, so a wheel bump fails
here instead of in the artwork.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from diagrams.canvas import _label_runs
from diagrams.outline import (
    _FACES,
    FaceRun,
    GlyphStore,
    Run,
    advance,
    assert_capabilities,
    emit_runs,
    segment,
    shape,
)

_SANS = (False, False, False)
_SANS_FACE = _FACES[_SANS][0]

#: The mono faces are the one chain that still carries a fallback: they
#: lack characters this corpus sets, and defer to the proportional face
#: of the same weight. The proportional chains stand alone now that every
#: face comes from the one DejaVu family.
_MONO = (True, False, False)
_MONO_FACE = _FACES[_MONO][0]
_FALLBACK_FACE = _FACES[_MONO][1]


def test_capabilities_hold_here() -> None:
    # The suite runs in the same environment that generates the plates;
    # if this raises, no other test in the file is meaningful.
    assert_capabilities()


def test_raqm_canary_layout_helper_shapes_marks() -> None:
    # Private API pin: matplotlib._text_helpers.layout must exist and
    # must shape a combining mark with zero pen advance, attached inside
    # the base's extent. A wheel that breaks either fails here, not in
    # the committed artwork.
    from matplotlib import _text_helpers

    assert callable(_text_helpers.layout)
    assert advance("T̂", _SANS_FACE, 100) == advance("T", _SANS_FACE, 100)
    glyphs = shape("T̂", _SANS_FACE)
    assert len(glyphs) == 2
    _, mark_x, mark_y = glyphs[1]
    assert 0 < mark_x < advance("T", _SANS_FACE, 100)
    assert mark_y > 0  # raised onto the capital, not drawn on the baseline


def test_advance_is_the_pen_never_the_ink() -> None:
    # A boundary space must keep its width: the ink box would eat it.
    assert advance("a ", _SANS_FACE, 20) > advance("a", _SANS_FACE, 20)
    assert advance(" ", _SANS_FACE, 20) > 0.0
    assert advance("", _SANS_FACE, 20) == 0.0


def test_advance_scales_linearly_with_size() -> None:
    w = advance("Measured", _SANS_FACE, 10)
    assert advance("Measured", _SANS_FACE, 20) == pytest.approx(2 * w)


def test_combining_marks_shape_in_all_four_sans_faces() -> None:
    for bold in (False, True):
        for italic in (False, True):
            face = _FACES[(False, bold, italic)][0]
            assert advance("T̂", face, 100) == advance("T", face, 100)
            glyphs = shape("T̂", face)
            assert len(glyphs) == 2
            assert 0 <= glyphs[1][1] <= advance("T", face, 100)


def test_six_face_matrix_measures_and_shapes() -> None:
    for face_key, chain in _FACES.items():
        assert chain
        for path in chain:
            assert path.is_file()
        runs = segment("Level 3", face_key)
        assert runs == [FaceRun("Level 3", chain[0])]
        assert advance("Level 3", chain[0], 20) > 0.0
        assert shape("Level 3", chain[0])


def test_segmentation_prefers_the_primary_face() -> None:
    # ℓ is one of the corpus glyphs the mono faces lack: it moves to the
    # proportional face while its neighbours stay primary, and the pieces
    # are maximal substrings.
    runs = segment("aℓb", _MONO)
    assert runs == [
        FaceRun("a", _MONO_FACE),
        FaceRun("ℓ", _FALLBACK_FACE),
        FaceRun("b", _MONO_FACE),
    ]


def test_fallback_never_splits_a_combining_cluster() -> None:
    # The circumflex is covered by both faces, its base ℓ only by the
    # fallback: the whole cluster must move together so the mark is
    # shaped against the base it modifies.
    runs = segment("aℓ̂b", _MONO)
    assert runs == [
        FaceRun("a", _MONO_FACE),
        FaceRun("ℓ̂", _FALLBACK_FACE),
        FaceRun("b", _MONO_FACE),
    ]


def test_uncovered_character_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="no face in the chain draws"):
        segment("a͸b", _SANS)  # unassigned codepoint: no face maps it


def test_whitespace_only_run_advances_the_pen_and_draws_nothing() -> None:
    store = GlyphStore()
    markup = emit_runs(
        store,
        [Run("a", _SANS, 0.0, 1.0), Run("  ", _SANS, 0.0, 1.0),
         Run("b", _SANS, 0.0, 1.0)],
        10.0, 20.0, 20, "#1a1a1a",
    )
    assert store.defs()
    assert '<path d=""' not in store.defs()
    groups = re.findall(r'translate\(([-\d.]+) ', markup)
    assert len(groups) == 2  # the whitespace run emitted no group at all
    gap = float(groups[1]) - float(groups[0])
    expected = advance("a", _SANS_FACE, 20) + advance("  ", _SANS_FACE, 20)
    assert gap == pytest.approx(expected, abs=0.02)


def test_atlas_ids_are_content_derived() -> None:
    store = GlyphStore()
    emit_runs(store, [Run("A", _SANS, 0.0, 1.0)], 0.0, 0.0, 20, "#000")
    defs = store.defs()
    match = re.search(r'id="([^"]+)"', defs)
    assert match is not None
    ref = match.group(1)
    # Stem plus hexadecimal glyph index, and the stem is the exact face:
    # "DejaVuSans" is a prefix of the bold and oblique stems, so only the
    # full match rules them out.
    assert re.fullmatch(r"DejaVuSans-[0-9a-f]+", ref)
    assert not ref.startswith(("axes_", "figure_"))
    # The same glyph in a fresh document gets the same id: content-derived,
    # never positional.
    other = GlyphStore()
    emit_runs(other, [Run("A", _SANS, 0.0, 1.0)], 5.0, 5.0, 14, "#000")
    assert re.search(r'id="([^"]+)"', other.defs()).group(1) == ref  # type: ignore[union-attr]


def test_plain_labels_collapse_and_math_preserves() -> None:
    # The explicit ASCII collapse reproduces what the viewer did to the
    # live-text plates; the math branch keeps its run whitespace verbatim.
    assert _label_runs("two   spaces  collapse") == [
        Run("two spaces collapse", _SANS, 0.0, 1.0)
    ]
    assert _label_runs(" boundary trimmed ") == [
        Run("boundary trimmed", _SANS, 0.0, 1.0)
    ]
    nbsp = _label_runs("10 dB")
    assert nbsp == [Run("10 dB", _SANS, 0.0, 1.0)]  # NBSP survives
    math = _label_runs("$CN = $  end")
    # Composer whitespace is kept verbatim, and the same-style prose tail
    # merges into the run without collapsing the boundary spaces.
    assert math == [Run("CN =   end", _SANS, 0.0, 1.0)]


def test_empty_label_composes_to_no_runs() -> None:
    assert _label_runs("") == []
    assert _label_runs("   ") == []


def test_bold_call_styles_math_variables_bolditalic() -> None:
    runs = _label_runs("$f_c$ corner", bold=True)
    assert runs[0] == Run("f", (False, True, True), 0.0, 1.0)
    assert runs[-1].face == (False, True, False)


def test_mono_or_whole_string_italic_with_math_is_refused() -> None:
    with pytest.raises(ValueError, match="mono"):
        _label_runs("$L_p$", mono=True)
    with pytest.raises(ValueError, match="italic"):
        _label_runs("$L_p$", italic=True)


def test_script_runs_scale_and_shift_as_element_fractions() -> None:
    runs = _label_runs("$L_p$", bold=False)
    assert runs == [
        Run("L", (False, False, True), 0.0, 1.0),
        Run("p", (False, False, True), 0.22, 0.70),
    ]
    markup = emit_runs(GlyphStore(), runs, 0.0, 100.0, 20, "#000")
    # The subscript drops by 0.22 of the *element* size (4.4 px at 20 px)
    # and scales to 0.70 of it: the confusable pair, pinned.
    assert 'translate(0 100) scale(0.2 -0.2)' in markup
    assert 'translate(' in markup
    assert '104.4) scale(0.14 -0.14)' in markup


def test_emission_carries_fill_and_no_style_attribute() -> None:
    markup = emit_runs(GlyphStore(), _label_runs("ab"), 0.0, 0.0, 20, "#123456")
    assert 'fill="#123456"' in markup
    assert "style=" not in markup


def test_collision_census_names_each_way_a_label_can_land_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One plate composed to fail all four ways, and every way reported."""
    from diagrams import outline, registry
    from diagrams.canvas import SVG, Theme

    def build(s: SVG, th: Theme) -> None:
        # Two labels meeting on one line.
        s.text(60, 100, "left run", 16, th.fg, anchor="start")
        s.text(110, 100, "right run", 16, th.fg, anchor="start")
        # A label wider than the stroked box it sits in.
        s.rect(60, 160, 80, 40, "none", th.fg)
        s.text(100, 185, "wider than its box", 16, th.fg)
        # A rule drawn through a label.
        s.text(400, 300, "struck through", 16, th.fg)
        s.line(400, 260, 400, 340, th.fg)
        # A filled panel drawn after the label it covers.
        s.text(400, 400, "covered", 16, th.fg)
        s.rect(340, 380, 120, 40, th.panel, "none")

    monkeypatch.setattr(registry, "DIAGRAMS",
                        {"plate": (build, "Title", 560)})
    kinds = {hit.kind for hit in outline._collisions()}
    assert kinds == {"overprint", "escapes", "struck", "painted over"}
