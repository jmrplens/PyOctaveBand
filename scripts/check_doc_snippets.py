#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Gate for the Python snippets printed in the documentation.

The guides teach by example, and an example that does not run teaches the
wrong thing. Three checks, cheapest first:

1. **Shadowing.** A name imported from ``phonometry`` must not be rebound
   later in the same block, by an assignment or by another import. This is
   the failure the package split walked into: ``from scipy import signal``
   next to ``from phonometry import signal`` rebinds the name silently, and
   the snippet crashes several lines further down with an ``AttributeError``
   that points nowhere near the import. Python gives no warning for it and
   the interpreter is the only thing that notices.

2. **Same API in both languages.** A Spanish page is free to translate its
   comments and its plot labels and to ask for a Spanish figure
   (``plot(language="es")``), but it must teach the same API: the imports it
   takes from ``phonometry`` have to match its English twin exactly, name for
   name. Comparing whole token streams was tried first and is wrong: it fails
   on the ``language="es"`` argument, which is the one difference the pages
   are supposed to have. The ``docs/`` mirror is not paired with anything: it
   is written by hand for GitHub and does not carry the same set of examples,
   so it gets the shadowing check and nothing else.

3. **Execution.** Every English page's blocks, plus those of the README and
   of ``llms.txt``, are concatenated in reading
   order (a guide is a narrative: later blocks use the variables the earlier
   ones bound) and run in a subprocess. Pages that cannot run standalone are
   listed in :data:`_SKIP` with a reason, and the list is checked for
   staleness: a page that starts passing must leave it. A block that passes
   ``...`` to a call is a sketch of a call rather than a computation, so it is
   read but not run.

Usage::

    python scripts/check_doc_snippets.py            # all three checks
    python scripts/check_doc_snippets.py --static   # skip the execution pass

Exit status 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import pathlib
import re
import subprocess
import sys
import tempfile

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DOCS = _ROOT / "docs"
_SITE = _ROOT / "site" / "src" / "content" / "docs"
_SITE_ES = _SITE / "es"

#: Fenced Python block, capturing its body.
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: Pages whose snippets cannot run as a standalone script, with the reason.
#: Anything not listed here must run to completion. Keep the reasons specific:
#: "needs a file the repository does not ship" is a fact, "example" is not.
_SKIP: dict[str, str] = {
    # The page reads something the repository does not ship.
    "getting-started": "reads measurement.wav, a recording the reader supplies",
    "block-processing": "streams through soundfile, an optional dependency",
    # The page shows excerpts of a workflow rather than a script: a block
    # starts from a variable the prose introduced, or a later block rebinds
    # one an earlier block still relies on. Reading order and running order
    # are not the same thing on these pages, and that is the author's call.
    "aircraft-noise": "excerpt: starts from the spl of the prose",
    "correlation-delay": "excerpt: starts from the record of the prose",
    "detailed-prediction": "excerpt: starts from the paths of the prose",
    "electroacoustics": "excerpt: starts from the captured signal of the prose",
    "flanking-lab": "excerpt: starts from the measured levels of the prose",
    "impedance-tube": "excerpt: starts from the measured spectrum of the prose",
    "impulsive-sound": "excerpt: starts from the recording of the prose",
    "intensity": "excerpt: starts from the band levels of the prose",
    "machine-diagnostics": "excerpt: starts from the record of the prose",
    "miso-coherence": "excerpt: starts from the third input of the prose",
    "objective-intelligibility": "excerpt: starts from the clean reference",
    "panel-sound-insulation": "excerpt: two scenarios share a variable name",
    "psychoacoustic-annoyance": "excerpt: starts from the record of the prose",
    "room-to-room": "excerpt: two scenarios share the band vector name",
    "rotorcraft-noise": "excerpt: starts from the measured spectrum",
    "sound-power": "excerpt: starts from the surface levels of the prose",
    "spectral-analysis": "excerpt: starts from the record of the prose",
    "swept-sine-distortion": "excerpt: starts from the captured response",
    "synchronous-averaging": "excerpt: a later block shortens the record an "
                             "earlier one averages",
    "time-frequency": "excerpt: starts from the record of the prose",
    "time-weighting": "excerpt: starts from the block stream of the prose",
    "underwater-acoustics": "excerpt: starts from the hydrophone record",
    # Known defect, not an excerpt: the constant the fiche needs is defined in
    # multiple_shock_vibration but is not re-exported by the vibration
    # namespace the page imports, so the snippet cannot run as printed.
    "multiple-shock-vibration": "teaches vibration.RISK_THRESHOLDS_MALE, which "
                                "the package does not export",
}

#: Timeout per page, generous enough for the FDTD and ECMA pages.
_TIMEOUT_S = 600


def _blocks(path: pathlib.Path) -> list[str]:
    """Python blocks of a page, in reading order."""
    return _BLOCK.findall(path.read_text(encoding="utf-8"))


def _imported_names(code: str) -> list[str]:
    """Every name a block takes from phonometry, as ``module.name``."""
    out: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "phonometry"
        ):
            out += [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names if a.name.startswith("phonometry")]
    return out


def _phonometry_imports(tree: ast.AST) -> dict[str, int]:
    """Names bound by an import from phonometry, mapped to their line."""
    bound: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "phonometry"
        ):
            for alias in node.names:
                bound[alias.asname or alias.name] = node.lineno
    return bound


def _rebindings(tree: ast.AST, names: dict[str, int]) -> list[str]:
    """Reports for every name from ``names`` rebound after its import."""
    reports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not (
            node.module or ""
        ).startswith("phonometry"):
            for alias in node.names:
                name = alias.asname or alias.name
                if name in names:
                    reports.append(
                        f"line {node.lineno}: 'from {node.module} import {name}' "
                        f"rebinds the name imported from phonometry on line "
                        f"{names[name]}"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name in names:
                    reports.append(
                        f"line {node.lineno}: 'import {alias.name}' rebinds the "
                        f"name imported from phonometry on line {names[name]}"
                    )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    reports.append(
                        f"line {node.lineno}: '{target.id} = ...' rebinds the name "
                        f"imported from phonometry on line {names[target.id]}"
                    )
    return reports


def check_shadowing(
    pages: list[pathlib.Path], *, carry: bool = True
) -> list[str]:
    """Failures for every page that rebinds a name it imported.

    A page is read top to bottom, so the imports of an earlier block are still
    in scope in a later one: the check carries them forward instead of looking
    at each block alone. ``carry=False`` is for the generated dumps, which glue
    unrelated pages into one file: there the blocks share no scope, and
    carrying an import across them invents collisions that no reader can hit.
    """
    failures: list[str] = []
    for page in pages:
        carried: dict[str, int] = {}
        for i, block in enumerate(_blocks(page)):
            try:
                tree = ast.parse(block)
            except SyntaxError as exc:
                failures.append(f"{_rel(page)} block {i}: does not parse ({exc})")
                continue
            imported = _phonometry_imports(tree)
            seen = {**carried, **imported}
            if seen:
                for report in _rebindings(tree, seen):
                    failures.append(f"{_rel(page)} block {i}: {report}")
            # A block that rebinds a carried name owns it from here on.
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            carried.pop(target.id, None)
            if carry:
                carried.update(imported)
            else:
                carried = {}
    return failures


def check_translations(pairs: list[tuple[pathlib.Path, pathlib.Path]]) -> list[str]:
    """Failures for every pair that does not teach the same API."""
    failures: list[str] = []
    for left, right in pairs:
        want = sorted(n for b in _blocks(left) for n in _imported_names(b))
        got = sorted(n for b in _blocks(right) for n in _imported_names(b))
        if want == got:
            continue
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("has " + ", ".join(extra) + " which its twin does not")
        if not detail:  # same names, different multiplicity
            detail.append("imports the same names a different number of times")
        failures.append(
            f"{_rel(right)}: {'; '.join(detail)} against {_rel(left)}"
        )
    return failures


def _is_sketch(code: str) -> bool:
    """True for a block that calls something with ``...`` in the arguments.

    ``building.airborne_insulation(...)`` shows the shape of a call, not a
    computation, and the README leads with one. A keyword placeholder
    (``f(level=...)``) is the same thing said differently. Such a block parses, so the
    static checks still see it; running it would only assert that Ellipsis is
    not a sound pressure level.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    def placeholder(node: ast.Call) -> bool:
        args = [*node.args, *(kw.value for kw in node.keywords)]
        return any(isinstance(a, ast.Constant) and a.value is Ellipsis
                   for a in args)

    return any(isinstance(n, ast.Call) and placeholder(n) for n in ast.walk(tree))


def _run_page(page: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Run a page's blocks as one script; return its stderr tail on failure."""
    runnable = [b for b in _blocks(page) if not _is_sketch(b)]
    script = "import matplotlib\nmatplotlib.use('Agg')\n" + "\n".join(runnable)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "snippet.py"
        path.write_text(script, encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, str(path)], check=False,
                capture_output=True, text=True, timeout=_TIMEOUT_S, cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return page, f"timed out after {_TIMEOUT_S} s"
    if done.returncode == 0:
        return page, ""
    tail = done.stderr.strip().splitlines()
    return page, tail[-1] if tail else f"exit status {done.returncode}"


def check_execution(pages: list[pathlib.Path]) -> list[str]:
    """Failures for every runnable page that does not run, and stale skips."""
    runnable = [p for p in pages if p.stem not in _SKIP and _blocks(p)]
    skipped = [p for p in pages if p.stem in _SKIP and _blocks(p)]
    failures: list[str] = []
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for page, error in pool.map(_run_page, runnable):
            if error:
                failures.append(f"{_rel(page)}: {error}")
        # A page that starts running must leave the skip list, or the list
        # quietly grows into a list of pages nobody checks.
        for page, error in pool.map(_run_page, skipped):
            if not error:
                failures.append(
                    f"{_rel(page)}: runs now; remove it from _SKIP in "
                    f"{_rel(pathlib.Path(__file__))}"
                )
    return failures


def _rel(path: pathlib.Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except ValueError:  # pragma: no cover - paths are always inside the repo
        return path.as_posix()


def _published() -> list[pathlib.Path]:
    """Surfaces outside the guides that print snippets at the reader.

    The README is the PyPI front page and ``llms.txt`` is what an assistant
    reads first, so a broken example there is the most expensive kind. They
    are not guides, so they get the same checks by a different route: the
    dumps under ``site/public/llms`` and ``llms-full.txt`` restate guide
    material and only take the static ones (see :func:`main`).
    """
    published = [_ROOT / "README.md", _ROOT / "README_PYPI.md", _ROOT / "llms.txt"]
    return [p for p in published if p.exists()]


def _restated() -> list[pathlib.Path]:
    """Generated dumps that concatenate guide text; static checks only."""
    out = [_ROOT / "llms-full.txt", *sorted((_ROOT / "site" / "public" / "llms").glob("*.txt"))]
    return [p for p in out if p.exists()]


def _pages() -> tuple[list[pathlib.Path], list[tuple[pathlib.Path, pathlib.Path]]]:
    """Every page with snippets, and the pairs that must carry the same code."""
    site_en = [
        p for p in sorted(_SITE.rglob("*.md*"))
        if "/es/" not in p.as_posix() and "reference/api" not in p.as_posix()
    ]
    mirror = sorted(_DOCS.glob("*.md"))
    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for page in site_en:
        twin = _SITE_ES / page.relative_to(_SITE)
        if twin.exists():
            pairs.append((page, twin))
    everything = (site_en + mirror + _published() + _restated()
                  + [p for _, p in pairs if p.is_relative_to(_SITE_ES)])
    return everything, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true",
                        help="skip the execution pass")
    args = parser.parse_args()

    pages, pairs = _pages()
    restated = set(_restated())
    failures = check_shadowing([p for p in pages if p not in restated])
    failures += check_shadowing(sorted(restated), carry=False)
    failures += check_translations(pairs)
    stage = "static checks"
    if not args.static:
        runnable = [p for p in pages if p.is_relative_to(_SITE)
                    and not p.is_relative_to(_SITE_ES)] + _published()
        failures += check_execution(runnable)
        stage = "checks"

    if failures:
        print(f"{len(failures)} documentation snippet problems:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"Documentation snippets pass all {stage}: "
          f"{sum(len(_blocks(p)) for p in pages)} blocks over {len(pages)} pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
