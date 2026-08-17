#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The ``$...$`` composer of the hand-drawn diagram canvas.

``diagrams.canvas.SVG.text`` composes a string carrying ``$...$`` markup
into styled runs -- italic variables, dropped/raised scripts at reduced
size, upright prose, operators and acronyms -- and bakes them to glyph
outlines. These tests pin the composition (which runs exist and how they
are styled) and the emitted markup's load-bearing parts (the source-string
comment, the group transforms that carry position, scale and baseline
shifts, and the face each run draws from) -- the diagrams are committed
byte for byte, so the composer's output is part of the repository's
contract. The guard tests pin the error messages that turn a markup typo
into a failed generation instead of a silently mis-set published diagram.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from diagrams import canvas
from diagrams.canvas import LIGHT, SVG, _label_runs, _math_runs
from diagrams.i18n import _ES
from diagrams.outline import measure

#: The glyph-id stem each face writes, one per DejaVu file the plates set.
_REG = "DejaVuSans"
_BOLD = "DejaVuSans-Bold"
_ITAL = "DejaVuSans-Oblique"
_BOLDITAL = "DejaVuSans-BoldOblique"


def _uses(markup: str, stem: str) -> int:
    """How many glyph ids of exactly this face the markup carries.

    Both the ``<defs>`` atlas and the ``<use>`` references are counted. A
    bare substring test would not do here: ``DejaVuSans`` is a prefix of
    ``DejaVuSans-Bold``, and that of ``DejaVuSans-BoldOblique``, so a
    regular-face assertion would silently count the bold and italic
    glyphs too. The hexadecimal glyph index that closes every id is what
    pins the stem exactly.
    """
    return len(re.findall(rf'="#?{re.escape(stem)}-[0-9a-f]+"', markup))


def _element(s: str, **kwargs: object) -> str:
    svg = SVG(900, 560, LIGHT)
    svg.text(450, 100, s, **kwargs)  # type: ignore[arg-type]
    assert len(svg.parts) == 1
    return svg.parts[0]


def _group_ys(element: str) -> list[str]:
    """The y of every run group's translate, in emission order."""
    return re.findall(r"translate\([\d.\-]+ ([\d.\-]+)\)", element)


def test_plain_string_bakes_to_one_commented_group() -> None:
    element = _element("probe A, 30 mm above", size=14, anchor="start")
    assert element.startswith("<!-- probe A, 30 mm above -->")
    assert element.count("<g ") == 1
    # Start-anchored: the pen starts exactly at the call's x.
    assert 'transform="translate(450 100) scale(0.14 -0.14)"' in element
    assert 'fill="#1a1a1a"' in element
    # One <use> per inked glyph; the three spaces advance the pen only.
    assert element.count("<use ") == len("probe A, 30 mm above".replace(" ", ""))
    assert _uses(element, _REG) == element.count("<use ")
    assert "<text" not in element
    assert "tspan" not in element


def test_comment_escapes_metacharacters() -> None:
    element = _element("A & B <C>")
    assert "<!-- A &amp; B &lt;C&gt; -->" in element


def test_empty_label_emits_nothing() -> None:
    svg = SVG(900, 560, LIGHT)
    svg.text(450, 100, "")
    svg.text(450, 120, "   ")
    assert svg.parts == []


def test_subscript_composition() -> None:
    # The chunk list is the contract: italic base, italic dropped script.
    assert _math_runs("$L_p$") == [
        ("L", True, 0.0, 1.0), ("p", True, 0.22, 0.70),
    ]
    element = _element("$L_p$", size=20)
    # Script placement is the confusable pair, pinned: the glyph is drawn
    # at 0.70 of the element size but dropped by 0.22 of the *element*
    # size (4.4 px at 20 px), never of the script size.
    assert "scale(0.2 -0.2)" in element
    assert "104.4) scale(0.14 -0.14)" in element
    assert _uses(element, _ITAL) == 2


def test_superscript_composition() -> None:
    element = _element("$c^2$", size=20)
    assert _math_runs("$c^2$") == [
        ("c", True, 0.0, 1.0), ("2", False, -0.38, 0.70),
    ]
    # The superscript rises by 0.38 of the element size: 100 - 7.6.
    assert "92.4) scale(0.14 -0.14)" in element
    assert _uses(element, _ITAL) == 1
    assert _uses(element, _REG) == 1


def test_baseline_restored_after_script() -> None:
    # After Z's subscript the minus sign returns to the baseline: with
    # absolute per-run placement, its group sits back at the call's y.
    element = _element("$Z_2−Z_1$", size=20)
    assert _group_ys(element) == ["100", "104.4", "100", "100", "104.4"]


def test_braced_subscript_retokenized_at_script_size() -> None:
    # Letters inside a braced script are variables at script size; the
    # comma stays upright at script size beside them. `i` and `j` are
    # indices, so both halves take the italic default.
    assert _math_runs("$L_{i,j}$") == [
        ("L", True, 0.0, 1.0),
        ("i", True, 0.22, 0.70),
        (",", False, 0.22, 0.70),
        ("j", True, 0.22, 0.70),
    ]


def test_operator_names_and_acronyms_stay_upright() -> None:
    assert _math_runs("$CN = c·dt·√2/dx ≤ 1$") == [
        ("CN = ", False, 0.0, 1.0),
        ("c", True, 0.0, 1.0),
        ("·dt·√2/dx ≤ 1", False, 0.0, 1.0),
    ]


def test_greek_letters_are_italic_variables() -> None:
    # Adjacent single-letter variables merge into one italic run; the
    # prime is an upright glyph after its greek base.
    assert _math_runs("$ρc$") == [("ρc", True, 0.0, 1.0)]
    assert _math_runs("$κ′$") == [
        ("κ", True, 0.0, 1.0), ("′", False, 0.0, 1.0),
    ]


def test_combining_mark_travels_with_its_letter() -> None:
    assert _math_runs("$T̂$") == [("T̂", True, 0.0, 1.0)]
    element = _element("$T̂$")
    # One italic group, two glyphs: the base and its attached mark.
    assert element.count("<g ") == 1
    assert element.count("<use ") == 2
    assert _uses(element, _ITAL) == 2


def test_prose_and_math_share_one_pen_with_whitespace_verbatim() -> None:
    element = _element("measurement cell → $L_{p,s}$ ($h_s$)", size=15)
    assert element.startswith("<!-- measurement cell → $L_{p,s}$ ($h_s$) -->")
    # Run whitespace is kept verbatim: the upright prose before the math
    # merges with nothing and its boundary space keeps its advance, so
    # the italic L starts strictly right of the prose run.
    xs = [float(m) for m in
          re.findall(r"translate\(([\d.\-]+) ", element)]
    assert xs == sorted(xs)
    assert "<text" not in element


def test_anchor_resolves_from_the_measured_width() -> None:
    s = "$f_n$ = 295 kHz"
    runs = _label_runs(s, bold=True)
    width = measure(runs, 15)
    element = _element(s, size=15, anchor="end", bold=True)
    first_x = float(re.search(r"translate\(([\d.\-]+) ", element).group(1))  # type: ignore[union-attr]
    assert first_x == pytest.approx(450 - width, abs=0.02)
    # Bold styles the math's italic runs into BoldItalic.
    assert _uses(element, _BOLDITAL)
    assert _uses(element, _BOLD)


def test_math_and_prose_escape_metacharacters_in_the_comment() -> None:
    element = _element("A & B $a<b$")
    assert "<!-- A &amp; B $a&lt;b$ -->" in element
    # Four runs: upright prose, italic a, the upright < glyph, italic b.
    assert element.count("<g ") == 4


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
    # An opening bracket right after the script character is the same trap,
    # but braces around the whole tail would swallow the argument into the
    # subscript, so the advice must point at bracing the script alone.
    with pytest.raises(ValueError) as excinfo:
        _element("$f_n(2h)$")
    assert "ambiguous script '_n' before '('" in str(excinfo.value)
    assert "_{n}(...)" in str(excinfo.value)


def test_comma_glued_to_unbraced_script_raises() -> None:
    # $L_p,i$ would set p as the subscript and push ",i" back to the
    # baseline; the guard demands braces. A spaced-off comma is a list
    # separator and stays legal.
    with pytest.raises(ValueError) as excinfo:
        _element("$L_p,i$")
    assert "ambiguous comma '_p,i'" in str(excinfo.value)
    assert "'$L_p,i$'" in str(excinfo.value)
    assert _math_runs("$a_x , a_y , a_z$") == [
        ("a", True, 0.0, 1.0), ("x", True, 0.22, 0.70),
        (" , ", False, 0.0, 1.0),
        ("a", True, 0.0, 1.0), ("y", True, 0.22, 0.70),
        (" , ", False, 0.0, 1.0),
        ("a", True, 0.0, 1.0), ("z", True, 0.22, 0.70),
    ]
    # Braced, the comma composes inside the script, mixing the italic
    # indices and the roman descriptive runs. `w` and `eq` are descriptive
    # and merge into one upright run; `n` cannot join them, because this
    # set cannot tell the normalized `n` of $L_n$ from the harmonic order
    # of $H_n$ and so leaves the run italic.
    assert _math_runs("$L_{n,w,eq}$") == [
        ("L", True, 0.0, 1.0),
        ("n", True, 0.22, 0.70),
        (",w,eq", False, 0.22, 0.70),
    ]


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


def test_consecutive_scripts_stay_legal() -> None:
    # $x_1^2$ is not nesting: the subscript and the superscript both hang
    # off the baseline symbol, drop first and rise after.
    assert _math_runs("$x_1^2$") == [
        ("x", True, 0.0, 1.0),
        ("1", False, 0.22, 0.70),
        ("2", False, -0.38, 0.70),
    ]
    element = _element("$x_1^2$", size=20)
    assert _group_ys(element) == ["100", "104.4", "92.4"]


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
    assert _math_runs("$K_{ij}$") == [
        ("K", True, 0.0, 1.0), ("ij", True, 0.22, 0.70),
    ]
    assert _math_runs("$η_{ij}$") == [
        ("η", True, 0.0, 1.0), ("ij", True, 0.22, 0.70),
    ]


def test_capital_greek_is_upright_and_lowercase_stays_italic() -> None:
    # Capital Greek letters are operators and descriptors in this corpus
    # and are set upright at every level, matching the roman Δ of
    # difference of ISO 80000-2, the Δ_SOR print of ECAC Doc 29 §4.5.7 and
    # the upright \Delta mathtext sets in the matplotlib figures.
    assert _math_runs("$ΔL_i$") == [
        ("Δ", False, 0.0, 1.0),
        ("L", True, 0.0, 1.0),
        ("i", True, 0.22, 0.70),
    ]
    # Doc 29 prints the SOR subscript in italic (eq. 4-23): it is not on
    # the curated roman list, so it takes the italic index default.
    assert _math_runs("$Δ_{SOR}$") == [
        ("Δ", False, 0.0, 1.0), ("SOR", True, 0.22, 0.70),
    ]
    assert all(not italic for _, italic, _, _ in
               _math_runs("banked by $Φ$ in turns"))
    # Lowercase Greek keeps the italic-variable rule, base and script.
    assert _math_runs("$θ$") == [("θ", True, 0.0, 1.0)]


def test_descriptive_subscripts_stay_upright() -> None:
    # The curated descriptive subscripts are word abbreviations and keep
    # the roman the standards print them in.
    assert _math_runs("$L_{Aeq}$") == [
        ("L", True, 0.0, 1.0), ("Aeq", False, 0.22, 0.70),
    ]
    assert _math_runs("$f_{max}$") == [
        ("f", True, 0.0, 1.0), ("max", False, 0.22, 0.70),
    ]


def test_tilde_travels_with_its_letter() -> None:
    # x̃ (x + U+0303) styles as one glyph run; splitting the pair into
    # runs of different style would detach the mark from its base.
    assert _math_runs("$x̃$") == [("x̃", True, 0.0, 1.0)]
    assert _math_runs("$x̃_{ref}$") == [
        ("x̃", True, 0.0, 1.0), ("ref", False, 0.22, 0.70),
    ]


def test_translation_happens_before_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_lookup(s: str, *, translate: bool) -> str:
        assert translate
        return {"speed $c_0$": "celeridad $c_0$"}[s]

    monkeypatch.setattr(canvas, "lookup", fake_lookup)
    svg = SVG(900, 560, LIGHT, lang="es")
    svg.text(450, 100, "speed $c_0$")
    # The pipeline is strictly tr -> compose -> measure -> outline:
    # the comment and the composed runs both carry the Spanish string.
    assert svg.parts[0].startswith("<!-- celeridad $c_0$ -->")
    assert _uses(svg.parts[0], _ITAL)


def test_fit_gate_raises_on_overflow_and_passes_the_boundary() -> None:
    svg = SVG(900, 560, LIGHT)
    width = measure(_label_runs("overhang"), 17)
    # Just inside the half-pixel grace: no error.
    svg.text(900 - width + 0.4, 100, "overhang", anchor="start")
    assert svg.parts
    # Just past it: the gate names the string and the measured span.
    with pytest.raises(ValueError, match=r"spans .* on a 900 px sheet"):
        svg.text(900 - width + 0.6, 100, "overhang", anchor="start")
    with pytest.raises(ValueError, match="900 px sheet"):
        svg.text(-1, 130, "left overhang", anchor="start")


def test_fit_gate_report_mode_records_and_continues(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PHONO_DIAGRAM_FIT", "report")
    svg = SVG(900, 560, LIGHT)
    svg.text(899, 100, "collected, not fatal", anchor="start")
    assert svg.parts  # emitted anyway: one run collects the worklist
    assert "collected, not fatal" in capsys.readouterr().err


def test_render_routes_the_title_through_the_same_emission() -> None:
    plain = SVG(900, 560, LIGHT).render("Plain title")
    assert "<title>Plain title</title>" in plain
    assert "<!-- Plain title -->" in plain
    assert "<defs>" in plain
    assert _uses(plain, _BOLD) >= 2  # bold outlined glyphs, defs + uses
    composed = SVG(900, 560, LIGHT).render("Rating $R_w$ measured")
    assert "<title>Rating $R_w$ measured</title>" in composed
    # The w subscript drops 0.22 of the 22 px title: y = 30 + 4.84, at
    # 0.7 of the title size.
    assert "34.84) scale(0.154 -0.154)" in composed
    assert _uses(composed, _BOLDITAL)
    assert "<text" not in composed
    assert "xml:space" not in composed


def test_title_steps_down_only_as_far_as_the_sheet_demands() -> None:
    svg = SVG(900, 560, LIGHT)
    assert svg.title_size("Plain title") == canvas._TITLE_SIZES[0]
    # Each step is taken at the width the step above stops keeping its
    # margin at, and the floor is returned rather than a further step.
    room = 900 - 2 * canvas._TITLE_MARGIN
    for larger, smaller in zip(canvas._TITLE_SIZES, canvas._TITLE_SIZES[1:]):
        wide = "W" * (int(room / measure(_label_runs("W", bold=True), larger)) + 1)
        assert svg.title_size(wide) <= smaller
    assert svg.title_size("W" * 200) == canvas._TITLE_SIZES[-1]


def test_title_steps_down_for_the_margin_not_only_for_the_sheet() -> None:
    # A title that clears the sheet at the top size but eats into the
    # white the plate keeps at its edges still steps down: the defect
    # this catches is a title reading as if it had run off the page.
    svg = SVG(900, 560, LIGHT)
    unit = measure(_label_runs("W", bold=True), canvas._TITLE_SIZES[0])
    wide = "W" * int((900 - canvas._TITLE_MARGIN) / unit)
    assert measure(_label_runs(wide, bold=True), canvas._TITLE_SIZES[0]) < 900
    assert svg.title_size(wide) < canvas._TITLE_SIZES[0]


def test_a_stepped_down_title_carries_its_scripts_down_with_it() -> None:
    # Long enough to need the floor, short enough that the floor holds it.
    title = "Rating $R_w$ against a reference" + " curve" * 8
    assert SVG(900, 560, LIGHT).title_size(title) == 20
    out = SVG(900, 560, LIGHT).render(title)
    # The w subscript drops 0.22 of the size the element got, not of the
    # nominal one: y = 30 + 4.40, at 0.7 of 20 px.
    assert "34.4) scale(0.14 -0.14)" in out
    assert "34.84) scale(0.154 -0.154)" not in out


def test_emission_carries_no_style_attribute_and_no_checker_ids() -> None:
    svg = SVG(900, 560, LIGHT)
    svg.text(450, 100, "$L_{Aeq}$ over ⟨range⟩", bold=True)
    out = svg.render("A $f_c$ title")
    assert "style=" not in out
    for match in re.finditer(r'id="([^"]+)"', out):
        assert not match.group(1).startswith(("axes_", "figure_"))


def _style_runs(s: str) -> list[list[tuple[str, str]]]:
    """Per math segment, the merged (level, style) runs the composer sets."""
    out: list[list[tuple[str, str]]] = []
    for k, segment in enumerate(s.split("$")):
        if k % 2 == 0:
            continue
        runs: list[tuple[str, str]] = []
        for kind, payload in canvas._math_tokens(segment, s):
            if kind in ("var", "up"):
                pieces = [("base", kind)]
            else:
                pieces = [
                    (kind, kind2) for kind2, _ in
                    canvas._math_tokens(payload, s, script=True)
                ]
            for piece in pieces:
                if not runs or runs[-1] != piece:
                    runs.append(piece)
        out.append(runs)
    return out


# English strings that spell a quantity out as prose where the Spanish
# twin writes the symbol, so the twin composes exactly one math segment
# its English side does not have ("a quarter wavelength" beside $λ/4$).
_SYMBOL_SPELLED_AS_PROSE = {
    (
        "reactive silencer: $TL = 10 log_{10}[1 + ¼ (m − 1/m)^2 "
        "sin^{2}(k·L)]$, peaking where the 0.3 m chamber is a quarter "
        "wavelength"
    ),
}


def test_spanish_twins_mirror_the_math_structure() -> None:
    # Every $...$ pair of the diagram i18n table must compose with the same
    # script structure and the same italic/roman style, position by
    # position: a twin whose subscript falls off the curated roman list
    # while the English side is on it (f_upper beside f_sup, m_eff beside
    # m_ef) would silently publish the two languages in different styles.
    # The curated prose-to-symbol twins carry one extra trailing segment
    # with no English counterpart to mirror; the segments before it must
    # still pair up run by run.
    pairs = [(en, es) for en, es in _ES.items() if "$" in en or "$" in es]
    assert pairs, "the diagram i18n table lost its $...$ entries"
    for en, es in pairs:
        en_runs = _style_runs(en)
        es_runs = _style_runs(es)
        if en in _SYMBOL_SPELLED_AS_PROSE:
            assert len(es_runs) == len(en_runs) + 1, (en, es)
            es_runs = es_runs[: len(en_runs)]
        assert en_runs == es_runs, (en, es)


def test_composition_is_deterministic() -> None:
    def build() -> str:
        svg = SVG(900, 560, LIGHT)
        svg.text(450, 100, "$v ← v − (dt/ρ·dx)·grad p$,  then  $p$", size=13)
        svg.text(450, 130, "$TL(f) = 20 log_{10} |I(f) / T(f)|$", size=20)
        return svg.render("A title with $f_c$ inside")

    first = build()
    second = build()
    assert first == second
