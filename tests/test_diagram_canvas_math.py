#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The ``$...$`` composer of the hand-drawn diagram canvas.

``diagrams.canvas.SVG.text`` composes a string carrying ``$...$`` markup
into styled ``<tspan>`` runs: italic variables, dropped/raised scripts at
reduced size, upright prose, operators and acronyms. These tests pin the
emitted SVG exactly -- the diagrams are committed byte for byte, so the
composer's output is part of the repository's contract -- and they pin the
path a plain string takes, which must stay byte-identical to what the
canvas emitted before the composer existed.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from diagrams import canvas
from diagrams.canvas import _FONT, LIGHT, SVG


def _element(s: str, **kwargs: object) -> str:
    svg = SVG(900, 560, LIGHT)
    svg.text(450, 100, s, **kwargs)  # type: ignore[arg-type]
    assert len(svg.parts) == 1
    return svg.parts[0]


def test_plain_string_element_unchanged() -> None:
    # The exact element the canvas emitted before the composer existed.
    element = _element("probe A, 30 mm above", size=14, anchor="start")
    assert element == (
        f'<text x="450" y="100" font-family="{_FONT}" font-size="14" '
        f'fill="#1a1a1a" text-anchor="start">probe A, 30 mm above</text>'
    )


def test_plain_string_still_escapes_metacharacters() -> None:
    assert ">A &amp; B &lt;C&gt;<" in _element("A & B <C>")


def test_subscript_composition() -> None:
    element = _element("$L_p$", size=20)
    assert element == (
        f'<text x="450" y="100" font-family="{_FONT}" font-size="20" '
        f'fill="#1a1a1a" text-anchor="middle" xml:space="preserve">'
        f'<tspan font-style="italic">L</tspan>'
        f'<tspan dy="4.4" font-size="14.0" font-style="italic">p</tspan>'
        f"</text>"
    )


def test_superscript_composition() -> None:
    element = _element("$c^2$", size=20)
    assert '<tspan font-style="italic">c</tspan>' in element
    assert '<tspan dy="-7.6" font-size="14.0">2</tspan>' in element


def test_baseline_restored_after_script() -> None:
    # After Z's subscript the minus sign returns to the baseline: the dy
    # bookkeeping must emit the exact opposite shift.
    element = _element("$Z_2−Z_1$", size=20)
    assert '<tspan dy="4.4" font-size="14.0">2</tspan>' in element
    assert '<tspan dy="-4.4">−</tspan>' in element


def test_braced_subscript_retokenized_at_script_size() -> None:
    # Letters inside a braced script are variables at script size; the
    # comma stays upright at script size beside them.
    element = _element("$L_{p,s}$", size=20)
    assert (
        '<tspan dy="4.4" font-size="14.0" font-style="italic">p</tspan>'
        '<tspan font-size="14.0">,</tspan>'
        '<tspan font-size="14.0" font-style="italic">s</tspan>'
    ) in element


def test_operator_names_and_acronyms_stay_upright() -> None:
    element = _element("$CN = c·dt·√2/dx ≤ 1$", size=17)
    assert "<tspan>CN = </tspan>" in element
    assert '<tspan font-style="italic">c</tspan>' in element
    assert "<tspan>·dt·√2/dx ≤ 1</tspan>" in element


def test_greek_letters_are_italic_variables() -> None:
    # Adjacent single-letter variables share one italic tspan; the prime
    # is an upright glyph after its greek base.
    assert '<tspan font-style="italic">ρc</tspan>' in _element("$ρc$")
    element = _element("$κ′$")
    assert '<tspan font-style="italic">κ</tspan><tspan>′</tspan>' in element


def test_combining_mark_travels_with_its_letter() -> None:
    assert '<tspan font-style="italic">T̂</tspan>' in _element("$T̂$")


def test_prose_upright_and_single_text_chunk() -> None:
    element = _element("measurement cell → $L_{p,s}$ ($h_s$)", size=15)
    assert "<tspan>measurement cell → </tspan>" in element
    assert 'xml:space="preserve"' in element
    # No tspan carries coordinates of its own: one text chunk, so the
    # text-anchor of the element keeps positioning the composed whole.
    assert element.count('x="') == 1
    assert element.count(' y="') == 1


def test_anchor_and_bold_carry_over() -> None:
    element = _element("$f_n$ = 295 kHz", size=15, anchor="end", bold=True)
    assert 'text-anchor="end"' in element
    assert 'font-weight="600"' in element


def test_mono_and_italic_ignored_for_math() -> None:
    element = _element("$L_p$", mono=True, italic=True)
    assert f'font-family="{_FONT}"' in element
    assert "Consolas" not in element
    # The only italics are the composed variables, not the whole string.
    assert 'text-anchor="middle" xml:space=' in element


def test_math_and_prose_escape_metacharacters() -> None:
    element = _element("A & B $a<b$")
    assert "<tspan>A &amp; B </tspan>" in element
    assert '<tspan font-style="italic">a</tspan>' in element
    assert "<tspan>&lt;</tspan>" in element


def test_unbalanced_markup_raises() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        _element("broken $L_p")


def test_multicharacter_script_without_braces_raises() -> None:
    # $f_max$ would silently set f with an m subscript and push "ax" back
    # to the baseline; the guard names the string and the ambiguous piece.
    with pytest.raises(ValueError) as excinfo:
        _element("cut-off $f_max$ of the array")
    assert "'_max'" in str(excinfo.value)
    assert "'cut-off $f_max$ of the array'" in str(excinfo.value)
    with pytest.raises(ValueError, match="ambiguous script"):
        _element("$p_ref$")
    # An opening bracket right after the script character is the same trap.
    with pytest.raises(ValueError, match="ambiguous script"):
        _element("$f_n(2h)$")


def test_backslash_in_math_raises() -> None:
    # There are no LaTeX commands in the diagram composer; a backslash
    # would be published as a literal glyph.
    with pytest.raises(ValueError) as excinfo:
        _element("$\\theta_0$ incidence")
    assert "backslash" in str(excinfo.value)
    # repr doubles the backslash in the quoted pieces of the message.
    assert r"'\\theta_0'" in str(excinfo.value)
    assert r"'$\\theta_0$ incidence'" in str(excinfo.value)


def test_unclosed_script_brace_raises_with_context() -> None:
    with pytest.raises(ValueError) as excinfo:
        _element("$L_{p$ level")
    assert "unclosed script brace" in str(excinfo.value)
    assert "'_{p'" in str(excinfo.value)
    assert "'$L_{p$ level'" in str(excinfo.value)
    with pytest.raises(ValueError, match="unclosed script brace"):
        _element("$c^{2$")


def test_nested_script_raises() -> None:
    with pytest.raises(ValueError) as excinfo:
        _element("$L_{p_1}$")
    assert "nested script" in str(excinfo.value)
    assert "'$L_{p_1}$'" in str(excinfo.value)
    with pytest.raises(ValueError, match="nested script"):
        _element("$a_{b^2}$")


def test_empty_script_payload_raises() -> None:
    for broken in ("$L_$", "$L^$", "$L_{}$"):
        with pytest.raises(ValueError, match="empty script"):
            _element(broken)


def test_index_subscripts_are_italic() -> None:
    # Letters inside a script are indices: italic, unlike the same run at
    # the baseline, where two or more Latin letters read as an acronym.
    element = _element("$K_{ij}$", size=20)
    assert ('<tspan dy="4.4" font-size="14.0" font-style="italic">ij</tspan>'
            in element)
    element = _element("$η_{ij}$", size=20)
    assert '<tspan font-style="italic">η</tspan>' in element
    assert ('<tspan dy="4.4" font-size="14.0" font-style="italic">ij</tspan>'
            in element)


def test_descriptive_subscripts_stay_upright() -> None:
    # The curated descriptive subscripts are word abbreviations and keep
    # the roman the standards print them in.
    element = _element("$L_{Aeq}$", size=20)
    assert '<tspan dy="4.4" font-size="14.0">Aeq</tspan>' in element
    element = _element("$f_{max}$", size=20)
    assert '<tspan dy="4.4" font-size="14.0">max</tspan>' in element


def test_tilde_travels_with_its_letter() -> None:
    # x̃ (x + U+0303) styles as one glyph; splitting the pair into tspans
    # of different style makes Chromium misplace the mark.
    assert '<tspan font-style="italic">x̃</tspan>' in _element("$x̃$")
    element = _element("$x̃_{ref}$", size=20)
    assert '<tspan font-style="italic">x̃</tspan>' in element
    assert '<tspan dy="4.4" font-size="14.0">ref</tspan>' in element


def test_translation_happens_before_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_lookup(s: str, *, translate: bool) -> str:
        assert translate
        return {"speed $c_0$": "celeridad $c_0$"}[s]

    monkeypatch.setattr(canvas, "lookup", fake_lookup)
    svg = SVG(900, 560, LIGHT, lang="es")
    svg.text(450, 100, "speed $c_0$")
    assert "<tspan>celeridad </tspan>" in svg.parts[0]
    assert '<tspan font-style="italic">c</tspan>' in svg.parts[0]


def test_render_composes_math_in_the_title() -> None:
    plain = SVG(900, 560, LIGHT).render("Plain title")
    assert ">Plain title</text>" in plain
    composed = SVG(900, 560, LIGHT).render("Rating $R_w$ measured")
    assert 'xml:space="preserve"' in composed
    assert (
        '<tspan font-style="italic">R</tspan>'
        '<tspan dy="5.7" font-size="18.2" font-style="italic">w</tspan>'
        '<tspan dy="-5.7"> measured</tspan>'
    ) in composed


def test_composition_is_deterministic() -> None:
    def build() -> str:
        svg = SVG(900, 560, LIGHT)
        svg.text(450, 100, "$v ← v − (dt/ρ·dx)·grad p$,  then  $p$", size=13)
        svg.text(450, 130, "$TL(f) = 20 log_{10} |I(f) / T(f)|$", size=20)
        return svg.render("A title with $f_c$ inside")

    assert build() == build()
