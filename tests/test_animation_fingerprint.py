#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What the clip fingerprint has to notice, and what it has to ignore.

The clips are the one committed asset CI never regenerates, so the only thing
standing between "the code that draws this clip changed" and a clip that
quietly keeps its old frames is the fingerprint
(``scripts/animation_fingerprint.py``) and the check that compares it. These
tests pin both halves of that bargain, because either one failing makes the
gate worthless: a fingerprint that misses a real change lets the staleness
through, and a fingerprint that moves on a reflowed comment costs a
several-minute re-render for nothing and is switched off within a week.

The tree used here is a miniature of the real package written into a tmp
directory: the registry with its ``_ANIMATIONS`` map, a clip module, the
shared clip tail, the theme and the two translation tables. Building it from
source text rather than from the repo is what lets a test change one line and
ask what the fingerprint did about it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import animation_fingerprint as fp

REGISTRY = '''
from .schematics import animate_one, animate_two

_ANIMATIONS = {
    "anim_one": animate_one,
    "anim_two": animate_two,
}
'''

SCHEMATICS = '''
from .media import _render_clip, _translate_str
from .theme import COLOR_PRIMARY


def _shared_label(ax):
    """A helper both clips call."""
    ax.text(0.0, 0.0, _translate_str("shared caption"), color=COLOR_PRIMARY)


def animate_one(output_dir):
    """The first clip."""
    _shared_label(None)
    _translate_str("only in clip one")
    _translate_str(f"level {1.0:.1f} dB")
    _render_clip(None, None, output_dir, "anim_one")


def animate_two(output_dir):
    """The second clip, which shares nothing but the tail."""
    _translate_str("only in clip two")
    _render_clip(None, None, output_dir, "anim_two")
'''

MEDIA = '''
from .i18n import _translate_figure

_ANIM_FPS = 20


def _translate_str(s):
    return s


def _render_clip(fig, update, output_dir, stem):
    _translate_figure(fig)
    return _ANIM_FPS
'''

THEME = '''
COLOR_PRIMARY = "#1f77b4"


def set_theme(dark):
    """The palette every frame is drawn with."""
    return COLOR_PRIMARY if dark else COLOR_PRIMARY
'''

I18N = '''
_ES_EXACT = {
    "shared caption": "rótulo compartido",
    "only in clip one": "solo en el clip uno",
    "only in clip two": "solo en el clip dos",
    "some other figure": "otra figura cualquiera",
}

_ES_PATTERNS = [
    (r"^level (\\d+)\\.(\\d+) dB$", r"nivel \\1,\\2 dB"),
    (r"^unrelated (\\d+) thing$", r"cosa \\1 no relacionada"),
]


def _translate_figure(fig):
    return fig
'''


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A miniature figures package, complete enough to fingerprint."""
    package = tmp_path / "scripts" / "figures"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, body in (("registry", REGISTRY), ("schematics", SCHEMATICS),
                       ("media", MEDIA), ("theme", THEME), ("i18n", I18N)):
        (package / f"{name}.py").write_text(body, encoding="utf-8")
    return tmp_path


def edit(tree: pathlib.Path, module: str, old: str, new: str) -> None:
    """Rewrite one fragment of a module of the miniature tree."""
    path = tree / "scripts" / "figures" / f"{module}.py"
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_every_registered_clip_is_fingerprinted(tree: pathlib.Path) -> None:
    assert sorted(fp.fingerprints(tree)) == ["anim_one", "anim_two"]


def test_the_same_sources_give_the_same_fingerprint(
        tree: pathlib.Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """The same sources hash the same, read again and read from elsewhere.

    The stamp is written from a working copy and recomputed in CI out of a
    checkout at another absolute path, so a fingerprint carrying anything of
    where the files sit -- a path in the hash, an order the filesystem
    happened to hand back -- would mark all forty-two clips stale on every
    run and be switched off by the end of the week.
    """
    import shutil

    elsewhere = tmp_path_factory.mktemp("elsewhere") / "checkout"
    shutil.copytree(tree, elsewhere)

    first = fp.fingerprints(tree)
    again = fp.fingerprints(tree)
    copied = fp.fingerprints(elsewhere)

    assert again == first
    assert copied == first


@pytest.mark.parametrize(
    ("module", "old", "new", "moves"),
    [
        # The clip's own body: the whole point.
        ("schematics", '"only in clip one"', '"clip one, reworded"',
         {"anim_one"}),
        # A helper it calls, in its own module.
        ("schematics", 'ax.text(0.0, 0.0', 'ax.text(0.5, 0.0', {"anim_one"}),
        # The shared clip tail: both clips are drawn through it.
        ("media", "_ANIM_FPS = 20", "_ANIM_FPS = 25", {"anim_one", "anim_two"}),
        # The theme, which nothing in a clip calls and every frame carries.
        ("theme", '"#1f77b4"', '"#d62728"', {"anim_one", "anim_two"}),
        # The translation machinery (this is the shape of the Spanish
        # minus-sign repair that started all this).
        ("i18n", "def _translate_figure(fig):\n    return fig",
         "def _translate_figure(fig):\n    return str(fig)",
         {"anim_one", "anim_two"}),
        # A translation the clip's own text selects, exact and by pattern.
        ("i18n", '"solo en el clip uno"', '"solo en el primer clip"',
         {"anim_one"}),
        ("i18n", '"rótulo compartido"', '"el rótulo compartido"',
         {"anim_one"}),
        ("i18n", r'r"nivel \1,\2 dB"', r'r"nivel de \1,\2 dB"', {"anim_one"}),
    ],
)
def test_a_change_that_reaches_a_clip_moves_its_fingerprint(
        tree: pathlib.Path, module: str, old: str, new: str,
        moves: set[str]) -> None:
    before = fp.fingerprints(tree)
    edit(tree, module, old, new)
    after = fp.fingerprints(tree)
    moved = {clip for clip in before if before[clip] != after[clip]}
    assert moved == moves


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        # Comments and docstrings: the fingerprint is taken over the syntax
        # tree, so prose cannot cost a re-render.
        ("schematics", '"""The first clip."""', '"""The first clip, rewritten\n\n    with a longer explanation.\n    """'),
        ("schematics", '"""A helper both clips call."""', "'''A helper.'''"),
        ("theme", '"""The palette every frame is drawn with."""',
         '"""The palette."""  # and a trailing comment'),
        # A translation entry no clip of this tree writes, which is what the
        # per-clip selection of the tables buys: an unrelated figure gaining
        # or changing a Spanish string must not mark every clip stale.
        ("i18n", '"otra figura cualquiera"', '"una figura distinta"'),
        ("i18n", r'r"cosa \1 no relacionada"', r'r"otra cosa \1"'),
    ],
)
def test_a_change_that_cannot_reach_a_clip_leaves_it_alone(
        tree: pathlib.Path, module: str, old: str, new: str) -> None:
    before = fp.fingerprints(tree)
    edit(tree, module, old, new)
    assert fp.fingerprints(tree) == before


def test_a_new_clip_is_reported_rather_than_ignored(tree: pathlib.Path) -> None:
    """A clip added to the registry has no stamp until it is rendered."""
    before = fp.fingerprints(tree)
    edit(tree, "registry", 'from .schematics import animate_one, animate_two',
         'from .schematics import animate_one, animate_three, animate_two')
    edit(tree, "registry", '    "anim_two": animate_two,',
         '    "anim_two": animate_two,\n    "anim_three": animate_three,')
    edit(tree, "schematics", 'def animate_two(output_dir):',
         'def animate_three(output_dir):\n    """A third clip."""\n'
         '    _render_clip(None, None, output_dir, "anim_three")\n\n\n'
         'def animate_two(output_dir):')
    after = fp.fingerprints(tree)
    assert set(after) - set(before) == {"anim_three"}
    assert {c: after[c] for c in before} == before


# -- the renderer's half of the bargain -------------------------------------
#
# The stamp is only worth anything if it is written whenever the files it
# describes are, which is a claim about the renderer rather than about the
# hash: a batch of clips is minutes of work each, and it is the runs that do
# not finish that decide whether the manifest can be trusted.


def _fake_batch(monkeypatch: pytest.MonkeyPatch,
                fails: str | None) -> list[list[str]]:
    """Run a three-clip batch with the rendering replaced, return the stamps."""
    import shutil

    from figures import registry

    stamped: list[list[str]] = []

    def render(clip: str, output_dir: str) -> None:
        if clip == fails:
            raise RuntimeError("the encoder died")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(registry, "_ANIMATIONS",
                        dict.fromkeys(("anim_one", "anim_two", "anim_three")))
    monkeypatch.setattr(registry, "_render_anim_variants", render)
    monkeypatch.setattr(registry, "_stamp_clips",
                        lambda clips, output_dir: stamped.append(list(clips)))
    if fails is None:
        registry.generate_animations("images", variants=True)
    else:
        with pytest.raises(RuntimeError, match="the encoder died"):
            registry.generate_animations("images", variants=True)
    return stamped


def test_a_finished_batch_stamps_every_clip_it_rendered(
        monkeypatch: pytest.MonkeyPatch) -> None:
    assert _fake_batch(monkeypatch, fails=None) == [
        ["anim_one", "anim_two", "anim_three"]]


def test_a_clip_that_dies_does_not_cost_the_stamps_of_the_ones_before_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The clips already on disk keep their fingerprints when a later one dies.

    Their four variants are written and complete; leaving with the error and
    no stamp would have the freshness check report them as unrendered, and
    the way to make it green again would be to render, at several minutes
    each, clips that are already correct.
    """
    assert _fake_batch(monkeypatch, fails="anim_two") == [["anim_one"]]


# -- what the check says ----------------------------------------------------


def _check(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
           committed: list[str], stamps: dict[str, str]) -> int:
    """Run the freshness check over a made-up image directory."""
    import check_animation_freshness as check

    images = tmp_path / "images"
    images.mkdir()
    for name in committed:
        (images / name).write_bytes(b"")
    monkeypatch.setattr(check, "IMAGES", images)
    monkeypatch.setattr(check.fp, "fingerprints", lambda root: {"anim_one": "1"})
    monkeypatch.setattr(check.fp, "read_manifest", lambda: stamps)
    return check.main()


def test_a_complete_and_stamped_clip_passes(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    import check_animation_freshness as check

    code = _check(monkeypatch, tmp_path, check.outputs("anim_one"),
                  {"anim_one": "1"})
    assert code == 0
    assert "1 committed clips" in capsys.readouterr().out


@pytest.mark.parametrize("dropped", ["anim_one_es.webm",
                                     "anim_one_es_dark_poster.jpg",
                                     "anim_one_dark.gif"])
def test_every_file_a_render_writes_is_asked_for(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str], dropped: str) -> None:
    """A poster or a GIF left behind is as broken as a missing WebM.

    They are written by the same render and embedded by the same pages: the
    site defers the video behind the poster and the GitHub documentation
    shows the GIF, so either one missing is a hole in the docs that the four
    WebM files being there says nothing about.
    """
    import check_animation_freshness as check

    committed = [name for name in check.outputs("anim_one") if name != dropped]
    code = _check(monkeypatch, tmp_path, committed, {"anim_one": "1"})
    assert code == 1
    out = capsys.readouterr().out
    assert f"1 of its 10 files are missing ({dropped})" in out


def test_a_half_rendered_clip_is_not_called_uncommitted(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The clip is there; only some of its files are, and only that is said."""
    import check_animation_freshness as check

    committed = check.outputs("anim_one")[:2]
    assert _check(monkeypatch, tmp_path, committed, {}) == 1
    out = capsys.readouterr().out
    assert "committed half-rendered, 8 of its 10 files are missing" in out
    assert "no clip committed" not in out
    assert "no fingerprint recorded; re-render the clip to stamp it" in out


def test_a_clip_with_no_files_at_all_says_so(
        monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    assert _check(monkeypatch, tmp_path, [], {}) == 1
    out = capsys.readouterr().out
    assert "registered but not committed at all" in out
    assert "half-rendered" not in out
