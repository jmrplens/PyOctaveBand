#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Record, while the figures are being generated, what the Spanish tables miss.

The Spanish edition of a figure is produced at save time by looking every
string of the finished figure up in a table (``scripts/figures/i18n.py`` for
the matplotlib figures, ``scripts/diagrams/i18n.py`` for the plates). A string
nobody added to the table is not an error: the lookup misses and the English
text is written into ``X_es.svg``. This module is the other half of that
lookup -- the miss, written down.

Both passes report, and the difference between them is the point:

* the **English** pass records the strings the tables have no entry for. Most
  are numbers and unit tokens, which need none.
* the **Spanish** pass records the strings that reached the output with no
  translation applied.

A string in both is English text inside the Spanish figure. A string only in
the Spanish pass came out of the library's own renderer already translated
(``phonometry._plot`` carries its own tables and the generators pass
``language=``), so the scripts' tables are right not to know it; intersecting
the two passes is what keeps those out of the report. It also means a string
the *library* fails to translate is caught here, because it then reaches both
passes identically.

Deciding which of those are real defects (``Error [dB]`` is spelled the same
in both languages) is ``scripts/check_figure_language.py``'s business; this
side only reports.

Recording is off unless :data:`AUDIT_ENV` names a directory, so a plain
``python scripts/generate_graphs.py`` is untouched. When it is set, every
process that generates something -- the parent, each spawned pool worker,
each forked animation variant -- writes its own ``<pid>.json`` fragment there
at exit, and the checker merges them. Fragments accumulate: whoever sets the
variable empties the directory first, which is what ``make graphs`` does.
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib

#: Names the directory the fragments are written to. Unset means "do not
#: record", which is the default everywhere except the generation runs the
#: gate is wired into.
AUDIT_ENV = "PHONOMETRY_FIGURE_LANGUAGE_AUDIT"

#: Where ``make graphs`` points :data:`AUDIT_ENV`, and where the checker looks
#: when it is not told otherwise. Under ``build/``, which is gitignored.
DEFAULT_DIR = "build/figure-language"

# stem -> strings the tables had no entry for, one map per pass. A stem with an
# empty set is an asset that was generated and came out clean, which is what
# lets the checker tell "no longer untranslated" from "not generated in this
# run".
_PASSES: dict[str, dict[str, set[str]]] = {"english": {}, "spanish": {}}

#: The asset being written and the pass writing it, i.e. what a string
#: reported now belongs to.
_CURRENT = ""
_PASS = "english"

_REGISTERED = False


def audit_dir() -> str | None:
    """The directory fragments are written to, or ``None`` when recording is off."""
    return os.environ.get(AUDIT_ENV) or None


def visit(stem: str, lang: str) -> None:
    """Start recording *stem* (an output name without language/theme suffixes).

    Called once per asset as it is saved, before its strings are looked up, so
    an asset that translates cleanly still leaves a record that it was
    generated at all.
    """
    global _CURRENT, _PASS
    if audit_dir() is None:
        return
    _CURRENT = stem
    _PASS = "english" if lang == "en" else "spanish"
    _PASSES[_PASS].setdefault(stem, set())
    if not _REGISTERED:
        _register()


def untranslated(text: str) -> None:
    """Report that the tables had no entry for *text*, in the running pass."""
    if audit_dir() is None or not _CURRENT or not text.strip():
        return
    _PASSES[_PASS].setdefault(_CURRENT, set()).add(text)


def _register() -> None:
    global _REGISTERED
    atexit.register(_dump)
    _REGISTERED = True


def _forget_after_fork() -> None:
    """Drop the parent's recording so a forked child records only its own.

    The module docstring says each forked animation variant writes its own
    fragment, and without this that holds only while the parent has not
    recorded anything before it forks. ``os.fork`` copies :data:`_PASSES` and
    ``_REGISTERED`` into the child but not the handler that reads them:
    ``multiprocessing.popen_fork`` empties the ``atexit`` registry in the
    child before the target runs, so a child that inherited
    ``_REGISTERED = True`` would never re-register and would drop everything
    it recorded. Clearing both puts the child where a spawned worker starts,
    and its first :func:`visit` registers a handler of its own -- after the
    child cleared the registry, which is why that one survives.
    """
    global _REGISTERED, _CURRENT, _PASS
    for pass_ in _PASSES.values():
        pass_.clear()
    _CURRENT = ""
    _PASS = "english"
    _REGISTERED = False


if hasattr(os, "register_at_fork"):  # POSIX only; Windows has no fork
    os.register_at_fork(after_in_child=_forget_after_fork)


def _dump() -> None:
    """Write this process's fragment. Registered on the first :func:`visit`."""
    directory = audit_dir()
    if directory is None or not any(_PASSES.values()):
        return
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {stem: sorted(strings) for stem, strings in sorted(pass_.items())}
        for name, pass_ in _PASSES.items()
    }
    (path / f"{os.getpid()}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )


def load(directory: str) -> dict[str, dict[str, set[str]]]:
    """Merge every fragment in *directory* into one ``pass -> stem -> strings``."""
    merged: dict[str, dict[str, set[str]]] = {"english": {}, "spanish": {}}
    for fragment in sorted(pathlib.Path(directory).glob("*.json")):
        data = json.loads(fragment.read_text(encoding="utf-8"))
        for name, pass_ in data.items():
            for stem, strings in pass_.items():
                merged[name].setdefault(stem, set()).update(strings)
    return merged
