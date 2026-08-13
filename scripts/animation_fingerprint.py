#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What each committed clip was drawn by, reduced to one hash per clip.

The figures are byte-compared: CI regenerates them and
``scripts/check_figures.py`` says whether the tree still matches. The clips
cannot be, for the reason ``make animations`` is a separate target -- an FDTD
simulation plus four video encodes per clip is minutes of CPU each, and the
encoders are not bit-reproducible across machines anyway. So a clip is
committed once and then keeps whatever it was drawn with, while the code that
draws it moves on underneath: the Spanish minus-sign repair reached only the
clips that happened to be re-rendered after it, and the twelve that were not
kept shipping an ASCII hyphen in their Spanish tick labels for months with
every gate green.

This module answers the same question ``check_figures.py`` answers for the
SVGs, without rendering anything: **is the committed clip older than the code
that draws it?** It reads the sources as text, resolves what each clip
actually depends on, and reduces that to a fingerprint. The renderer stamps
the fingerprint into :data:`MANIFEST` when it writes a clip
(:func:`stamp`), and ``scripts/check_animation_freshness.py`` recomputes it
in CI and compares. A stamp that no longer matches means the clip on disk was
drawn by code that has since changed.

What goes into a fingerprint
----------------------------

Only what can change a frame, and only the part of it the clip reaches:

* the clip's own ``animate_*`` function and, transitively, every function,
  class and module constant it names inside ``scripts/figures`` -- the module
  helpers, the shared clip tail (``figures/media.py``), the field builders,
  the translation machinery of ``figures/i18n.py``;
* ``figures.theme.set_theme``, which is not called by the clip but decides
  the palette and the rcParams every frame is drawn with, and everything it
  reaches;
* the translation table entries the clip's own strings select (see
  :func:`_selected_translations`).

Two decisions keep the fingerprint from crying wolf, which matters more here
than anywhere else in the repo: a false positive costs a several-minute
re-render, so an alarm nobody believes would be worse than no alarm.

* **The hash is over the AST, not the source text.** Reformatting, a
  reflowed comment or a rewritten docstring cannot move it; only a change to
  what the code *does* can.
* **The two big translation tables are excluded wholesale** and re-entered
  entry by entry. ``_ES_EXACT`` has thousands of lines and changes with every
  figure that gains a Spanish string; hashing it whole would mark all
  forty-two clips stale every time any unrelated figure is translated.

Known gaps, deliberately
------------------------

* **The library.** A clip calls ``phonometry`` for its physics, and this walk
  stops at the package boundary. Extending it there would trade a real but
  rare miss for a constant stream of false positives from refactors that
  change no pixel, and a library change that moves a number moves the static
  figures with it, where ``check_figures.py`` does catch it.
* **Pattern translations reached only through an f-string.** The selection
  below matches a table pattern against the clip's literal text, and a label
  assembled at run time from several pieces may not be recognisable in the
  source. The code half still covers the usual case, because a string that
  gains a pattern normally gains it *because* the code that builds it
  changed.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
from collections.abc import Iterable, Iterator

_SCRIPTS = pathlib.Path(__file__).resolve().parent

#: The stamped fingerprints, one ``<clip> <hash>`` line per clip, sorted.
#: Next to ``figure_language_baseline.txt``, which plays the same role for the
#: language gate: a committed statement about the generated assets that CI
#: checks the tree against.
MANIFEST = _SCRIPTS / "animation_fingerprints.txt"

#: The package the walk stays inside.
_PKG_NAME = "figures"

#: Where the clip names are mapped to their builders.
_REGISTRY = "figures.registry"

#: Always part of every fingerprint: nothing calls it from inside a clip, but
#: it sets the rcParams and the palette each frame is drawn with.
_EXTRA_ROOTS = (("figures.theme", "set_theme"),)

#: The translation tables, excluded from the code walk and re-entered per
#: clip by :func:`_selected_translations`.
_TABLES = (("figures.i18n", "_ES_EXACT"), ("figures.i18n", "_ES_PATTERNS"))

#: Reached from the clip tail, but incapable of changing a frame: the remote
#: encoder plumbing decides *where* ffmpeg runs, not what it is fed. Listing
#: them keeps an ssh or container change from marking all forty-two clips
#: stale. The encoder *settings* are not here: those do change the file.
_IGNORED: frozenset[tuple[str, str]] = frozenset(
    {
        ("figures.media", "_gpu_encode_target"),
        ("figures.media", "_write_gpu_ffmpeg_wrapper"),
        ("figures.i18n", "audit_figure"),
    }
)

#: Shortest run of plain text in a table pattern that may select it. Shorter
#: runs (" dB", "°") appear in half the corpus and would attach unrelated
#: patterns to every clip.
_MIN_PATTERN_RUN = 4

#: Shortest half of a run split across an interpolation (see
#: :func:`_run_is_written`).
_MIN_SPLIT = 3

Symbol = tuple[str, str]


class _Module:
    """One parsed source file: its symbols and what its imports point at."""

    def __init__(self, name: str, tree: ast.Module, owner: Sources) -> None:
        self.name = name
        self.tree = tree
        self.owner = owner
        self.symbols: dict[str, ast.AST] = {}
        #: local name -> (module, attribute); an empty attribute is the module
        self.imports: dict[str, tuple[str, str]] = {}
        #: top-level statements that define no name (see :meth:`collect`)
        self.body: list[ast.stmt] = []

    def collect(self, node: ast.stmt) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            self.symbols[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _target_names(target):
                    self.symbols[name] = node
        elif isinstance(node, ast.AnnAssign):
            for name in _target_names(node.target):
                self.symbols[name] = node
        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.imports[alias.asname or alias.name.split(".")[0]] = (
                    alias.name, "")
        elif isinstance(node, ast.ImportFrom):
            module = self._resolve_import(node)
            for alias in node.names:
                where = (module, alias.name) if module else ("", "")
                if module and f"{module}.{alias.name}" in self.owner.modules:
                    where = (f"{module}.{alias.name}", "")
                self.imports[alias.asname or alias.name] = where
        elif isinstance(node, ast.If | ast.Try):
            # Guarded imports (``if TYPE_CHECKING:``) and the sys.path dance.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    self.collect(child)
        elif isinstance(node, ast.Expr):
            # A bare top-level expression defines no name, and one of them
            # decides how every frame looks: the ``plt.rcParams.update({...})``
            # block of ``figures.theme`` sets the base font size, the grid and
            # its alpha. Collected under a synthetic name so the walk can pull
            # it in with the module's other roots.
            self.body.append(node)

    def _resolve_import(self, node: ast.ImportFrom) -> str:
        """The absolute module name of a ``from ... import`` statement."""
        if not node.level:
            return node.module or ""
        parts = self.name.split(".")
        # A package's ``__init__`` is addressed by the package name itself, so
        # one level up from a module is one part off its dotted name.
        base = parts[: len(parts) - node.level + (1 if self.is_package else 0)]
        return ".".join(base + ([node.module] if node.module else []))

    @property
    def is_package(self) -> bool:
        return self.name in self.owner.packages


def _target_names(target: ast.expr) -> Iterator[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Tuple | ast.List):
        for element in target.elts:
            yield from _target_names(element)


class Sources:
    """Every module of the figures package, parsed once."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.modules: dict[str, _Module] = {}
        self.packages: set[str] = set()
        package_root = root / "scripts" / "figures"
        parsed: dict[str, ast.Module] = {}
        for path in sorted(package_root.rglob("*.py")):
            rel = path.relative_to(package_root).with_suffix("")
            parts = list(rel.parts[:-1] if rel.name == "__init__" else rel.parts)
            name = ".".join([_PKG_NAME, *parts])
            if path.name == "__init__.py":
                self.packages.add(name)
            parsed[name] = ast.parse(path.read_text(encoding="utf-8"),
                                     filename=str(path))
        # Every module name has to be known before the imports are resolved:
        # ``from . import media`` reads as a module reference or as a symbol
        # depending on whether ``figures.media`` exists.
        for name, tree in parsed.items():
            self.modules[name] = _Module(name, tree, self)
        for module in self.modules.values():
            for node in module.tree.body:
                module.collect(node)

    # -- name resolution ---------------------------------------------------

    def resolve(self, module: str, name: str,
                _seen: frozenset[Symbol] = frozenset()) -> Symbol | None:
        """The defining ``(module, name)`` of *name* as seen from *module*."""
        mod = self.modules.get(module)
        if mod is None or (module, name) in _seen:
            return None
        if name in mod.symbols:
            return (module, name)
        target = mod.imports.get(name)
        if target is None or not target[1]:
            return None
        return self.resolve(target[0], target[1], _seen | {(module, name)})

    def module_of(self, module: str, name: str) -> str | None:
        """The module *name* refers to, when it is an imported module."""
        mod = self.modules.get(module)
        if mod is None:
            return None
        target = mod.imports.get(name)
        if target is not None and not target[1] and target[0] in self.modules:
            return target[0]
        if f"{module}.{name}" in self.modules:
            return f"{module}.{name}"
        return None

    # -- the walk ----------------------------------------------------------

    def closure(self, roots: Iterable[Symbol]) -> set[Symbol]:
        """Every package symbol reachable from *roots*, roots included."""
        seen: set[Symbol] = set()
        stack = list(roots)
        while stack:
            symbol = stack.pop()
            if symbol in seen or symbol in _IGNORED or symbol in _TABLES:
                continue
            node = self.node(symbol)
            if node is None:
                continue
            seen.add(symbol)
            stack.extend(self.referenced(symbol[0], node))
        return seen

    def node(self, symbol: Symbol) -> ast.AST | None:
        mod = self.modules.get(symbol[0])
        return None if mod is None else mod.symbols.get(symbol[1])

    def referenced(self, module: str, node: ast.AST) -> Iterator[Symbol]:
        """The package symbols *node* names, from inside *module*."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                found = self.resolve(module, child.id)
                if found is not None:
                    yield found
            elif isinstance(child, ast.Attribute) and isinstance(
                    child.value, ast.Name):
                target = self.module_of(module, child.value.id)
                if target is not None:
                    found = self.resolve(target, child.attr)
                    if found is not None:
                        yield found

    # -- the clips ---------------------------------------------------------

    def clips(self) -> dict[str, Symbol]:
        """``clip stem -> defining (module, function)`` from the registry."""
        registry = self.modules[_REGISTRY]
        mapping: dict[str, Symbol] = {}
        for node in ast.walk(registry.tree):
            if not isinstance(node, ast.AnnAssign | ast.Assign):
                continue
            targets = (
                list(_target_names(node.target))
                if isinstance(node, ast.AnnAssign)
                else [n for t in node.targets for n in _target_names(t)]
            )
            if "_ANIMATIONS" not in targets or not isinstance(
                    node.value, ast.Dict):
                continue
            for key, value in zip(node.value.keys, node.value.values):
                if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                        and isinstance(value, ast.Name)):
                    found = self.resolve(_REGISTRY, value.id)
                    if found is not None:
                        mapping[key.value] = found
        return mapping


# -- hashing ----------------------------------------------------------------


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """A copy of *node* with every docstring removed.

    Rewriting a docstring cannot change a frame, and a fingerprint that moves
    when the prose does would be ignored within a week.
    """
    import copy

    clone = copy.deepcopy(node)
    for child in ast.walk(clone):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef
                      | ast.ClassDef | ast.Module):
            body = child.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                child.body = body[1:] or [ast.Pass()]
    return clone


def _dump(node: ast.AST) -> str:
    return ast.dump(_strip_docstrings(node), annotate_fields=True,
                    include_attributes=False)


def _literals(sources: Sources, symbols: Iterable[Symbol]) -> set[str]:
    """Every string literal in *symbols*, f-string fragments included."""
    out: set[str] = set()
    for symbol in symbols:
        node = sources.node(symbol)
        if node is None:
            continue
        for child in ast.walk(_strip_docstrings(node)):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                out.add(child.value)
    return out


_META = set(r"[](){}|?*+.^$")


#: Escapes that stand for a character class or a position rather than for a
#: character (``\d``, ``\b``, a backreference), so a run ends at them. Every
#: other escape (``\$``, ``\(``, ``\\``) is the character itself.
_CLASS_ESCAPES = set("0123456789AbBdDsSwWZzGN")


def _pattern_runs(pattern: str) -> list[str]:
    """The plain-text runs of a regex, i.e. what it prints verbatim.

    A string a pattern translates necessarily contains all of these, which
    is what lets a clip's own text say which patterns belong to it.
    """
    runs: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            escaped = pattern[index + 1]
            if escaped in _CLASS_ESCAPES:
                runs.append("".join(current))
                current = []
            elif escaped in "nrt":
                current.append({"n": "\n", "r": "\r", "t": "\t"}[escaped])
            else:
                current.append(escaped)
            index += 2
            continue
        if char in _META:
            runs.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    runs.append("".join(current))
    return [run for run in runs if len(run) >= _MIN_PATTERN_RUN]


def _table_entries(sources: Sources) -> tuple[
        list[tuple[str, str]], list[tuple[str, str]]]:
    """``(_ES_EXACT items, _ES_PATTERNS items)`` read straight from the AST."""
    i18n = sources.modules["figures.i18n"]
    exact: list[tuple[str, str]] = []
    patterns: list[tuple[str, str]] = []
    node = i18n.symbols.get("_ES_EXACT")
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
        for key, value in zip(node.value.keys, node.value.values):
            pair = _literal_pair(key, value)
            if pair is not None:
                exact.append(pair)
    node = i18n.symbols.get("_ES_PATTERNS")
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
        for element in node.value.elts:
            if isinstance(element, ast.Tuple) and len(element.elts) == 2:
                pair = _literal_pair(element.elts[0], element.elts[1])
                if pair is not None:
                    patterns.append(pair)
    return exact, patterns


def _literal_pair(key: ast.expr | None,
                  value: ast.expr) -> tuple[str, str] | None:
    if key is None:   # ``{**other}`` inside the table literal
        return None
    try:
        left = ast.literal_eval(key)
        right = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if isinstance(left, str) and isinstance(right, str):
        return (left, right)
    return None


def _selected_translations(literals: set[str],
                           exact: list[tuple[str, str]],
                           patterns: list[tuple[str, str]]) -> list[str]:
    """The table entries the clip's own strings reach.

    An exact entry is selected when the clip writes its key verbatim. A
    pattern is selected when *every* run of plain text it prints is written
    somewhere in the clip, which is how the labels assembled with an
    f-string are recognised: the pattern's fixed text is, necessarily, the
    fixed text of the f-string, split at the interpolations. Requiring all
    the runs rather than any one of them is what keeps a pattern of some
    other figure from attaching itself to a clip through a run as common as
    ``" to "``.
    """
    chosen: list[str] = []
    for key, value in exact:
        if key in literals or _key_is_written(key, literals):
            chosen.append(f"exact\x00{key}\x00{value}")
    for pattern, repl in patterns:
        runs = _pattern_runs(pattern)
        if runs and all(_run_is_written(run, literals) for run in runs):
            chosen.append(f"pattern\x00{pattern}\x00{repl}")
    return sorted(chosen)


def _key_is_written(key: str, literals: set[str]) -> bool:
    """Whether the clip writes an exact-table key it assembles with numbers.

    ``"$L$ = 0.30 m · $m$ = 4"`` is an entry of the exact table -- a label
    with no words for the pattern list to key on -- and the clip builds it as
    ``f"$L$ = {length:.2f} m · $m$ = {ratio:.0f}"``. Taking the numbers out
    leaves exactly the fragments the source writes, so a key whose every
    non-numeric piece is written counts as this clip's.
    """
    import re as _re

    pieces = [piece for piece in _re.split(r"[\d]+", key)
              if len(piece.strip()) >= _MIN_SPLIT]
    return bool(pieces) and all(
        any(piece in literal for literal in literals) for piece in pieces)


def _run_is_written(run: str, literals: set[str]) -> bool:
    """Whether the clip writes *run*, whole or split across an interpolation.

    ``"Sabine, $T$ = 300 ms"`` is built as ``f"{name}, $T$ = {ms} ms"``, so
    the pattern's run ``"Sabine, $T$ = "`` sits in no single fragment: the
    name comes from a variable and only ``", $T$ = "`` is written literally.
    A run therefore also counts when it splits in two and both halves are
    written -- one split, both halves at least three characters, so the rule
    cannot degenerate into "every run is written character by character".
    """
    if any(run in literal for literal in literals):
        return True
    for cut in range(_MIN_SPLIT, len(run) - _MIN_SPLIT + 1):
        head, tail = run[:cut], run[cut:]
        if (any(head in literal for literal in literals)
                and any(tail in literal for literal in literals)):
            return True
    return False


def fingerprint_parts(sources: Sources, clip: str,
                      entry: Symbol) -> tuple[list[str], list[str]]:
    """``(code lines, translation lines)`` a clip's fingerprint is taken over."""
    roots = [entry, *_EXTRA_ROOTS]
    symbols = sources.closure(roots)
    code = sorted(
        f"{module}.{name}\x00{_dump(sources.node((module, name)))}"  # type: ignore[arg-type]
        for module, name in symbols
    )
    # Plus the top-level statements of every module the clip reaches. They
    # define no name, so the walk cannot reach them by reference, and one of
    # them is ``figures.theme``'s ``plt.rcParams.update({...})``: the base font
    # size, the grid and its alpha, i.e. how every frame looks.
    for module in sorted({module for module, _ in symbols}):
        body = sources.modules[module].body
        if body:
            code.append(f"{module}.<module>\x00"
                        + _dump(ast.Module(body=body, type_ignores=[])))
    code.sort()
    exact, patterns = _table_entries(sources)
    text = _selected_translations(_literals(sources, symbols), exact, patterns)
    return code, text


def fingerprints(root: pathlib.Path) -> dict[str, str]:
    """The fingerprint of every registered clip, computed from *root*'s sources."""
    sources = Sources(root)
    out: dict[str, str] = {}
    for clip, entry in sorted(sources.clips().items()):
        code, text = fingerprint_parts(sources, clip, entry)
        digest = hashlib.sha256()
        for line in code:
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        digest.update(b"--translations--\n")
        for line in text:
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        out[clip] = digest.hexdigest()[:16]
    return out


# -- the manifest -----------------------------------------------------------

_HEADER = """\
# What each committed clip was drawn by, one line per clip.
#
# The clips are not regenerated in CI (an FDTD run plus four video encodes
# each), so nothing else can tell that the code drawing a clip has moved since
# the clip was written. Each hash covers the clip's builder, everything it
# reaches inside scripts/figures (the shared clip tail, the theme, the
# translation machinery) and the translation-table entries its own strings
# select; it is taken over the syntax tree, so comments and docstrings do not
# move it. See scripts/animation_fingerprint.py.
#
# Written by the renderer. To refresh one clip, re-render it:
#     python scripts/generate_graphs.py --animations --anim <clip>
"""


def read_manifest() -> dict[str, str]:
    """The stamped fingerprints, ``clip -> hash``."""
    if not MANIFEST.exists():
        return {}
    out: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        clip, _, digest = line.partition(" ")
        out[clip] = digest.strip()
    return out


def write_manifest(stamps: dict[str, str]) -> None:
    body = "".join(f"{clip} {stamps[clip]}\n" for clip in sorted(stamps))
    MANIFEST.write_text(_HEADER + body, encoding="utf-8")


def stamp(clips: Iterable[str], root: pathlib.Path | None = None) -> None:
    """Record the current fingerprint of *clips*, leaving the rest alone.

    Called by the renderer once a clip's four variants are on disk, so the
    stamp and the files it describes are written together and a re-render
    cannot forget to refresh it.

    Read-modify-write under a lock: re-rendering several clips at once is the
    normal way to work through a batch, and two runs finishing together would
    otherwise each write back the manifest they read, losing one of the two
    stamps -- the one failure mode of this file that would be invisible until
    CI asked for the missing clip.
    """
    import fcntl

    names = list(clips)
    if not names:
        return
    current = fingerprints(root or _SCRIPTS.parent)
    # The lock file is never removed: unlinking it after the release lets a
    # later run create a fresh inode and lock that one while another run is
    # still inside the old one, which is the very race this guards.
    lock = MANIFEST.with_suffix(".lock")
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            stamps = read_manifest()
            for clip in names:
                if clip in current:
                    stamps[clip] = current[clip]
            write_manifest(stamps)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

