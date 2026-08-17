#  Copyright (c) 2026. Jose Manuel Requena Plens

"""
Tests ensuring the package treats matplotlib as the optional dependency it is.

Two promises are kept here. Importing phonometry must not force a specific
(e.g. non-interactive) backend, so the package can be used during interactive
exploration (IPython, Jupyter); see issue #52. And the base install must run on
NumPy and SciPy alone, so ``import phonometry`` has to succeed with matplotlib
absent, plotting failing only when a plot is actually asked for.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "phonometry"


def test_import_does_not_override_matplotlib_backend() -> None:
    """Importing phonometry must preserve the user's chosen backend."""
    code = (
        "import matplotlib\n"
        # Pick an explicit, always-available backend the user might have set.
        "matplotlib.use('svg')\n"
        "before = matplotlib.get_backend()\n"
        "import phonometry\n"
        "after = matplotlib.get_backend()\n"
        "assert before == after, f'backend changed: {before!r} -> {after!r}'\n"
    )
    # Propagate the parent's sys.path so the subprocess can import the package
    # even when it is only on sys.path (e.g. pytest without an installed build).
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _imports_matplotlib(node: ast.stmt) -> bool:
    """Whether ``node`` is an import statement that pulls in matplotlib."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        # A relative import (level > 0) names a module of this package, never
        # matplotlib, whatever the module happens to be called.
        names = [node.module or ""] if node.level == 0 else []
    else:
        return False
    return any(name == "matplotlib" or name.startswith("matplotlib.")
               for name in names)


def _eager_matplotlib_imports(body: list[ast.stmt], *, guarded: bool = False) -> list[int]:
    """Line numbers of the matplotlib imports that run at import time.

    Only two placements are lazy: inside a function body, and under
    ``if TYPE_CHECKING:``, which never executes. Everything else at module
    scope runs the moment the package is imported, including a class body and
    the branches of a runtime ``if``, a ``try`` or a ``with`` -- so this
    descends into those rather than assuming a bare top-level statement is the
    only way to import something eagerly.
    """
    eager: list[int] = []
    for node in body:
        if _imports_matplotlib(node):
            if not guarded:
                eager.append(node.lineno)
        elif isinstance(node, ast.If):
            # ``else`` is the runtime branch of a TYPE_CHECKING guard.
            deferred = guarded or "TYPE_CHECKING" in ast.unparse(node.test)
            eager += _eager_matplotlib_imports(node.body, guarded=deferred)
            eager += _eager_matplotlib_imports(node.orelse, guarded=guarded)
        elif isinstance(node, ast.Try):
            for block in (node.body, node.orelse, node.finalbody,
                          *(handler.body for handler in node.handlers)):
                eager += _eager_matplotlib_imports(block, guarded=guarded)
        elif isinstance(node, (ast.With, ast.ClassDef)):
            eager += _eager_matplotlib_imports(node.body, guarded=guarded)
    return eager


def test_no_module_scope_matplotlib_import_anywhere() -> None:
    """No module in the package may import matplotlib at import time.

    Scoped to the whole package rather than to one module: this used to check
    ``filters.design`` alone, which is where the first such import was found,
    and a later one in a rendering leaf sailed past it and broke
    ``import phonometry`` on a base install.
    """
    offenders = [
        f"{path.relative_to(SRC)}:{lineno}"
        for path in sorted(SRC.rglob("*.py"))
        for lineno in _eager_matplotlib_imports(
            ast.parse(path.read_text(encoding="utf-8")).body
        )
    ]
    assert not offenders, (
        "matplotlib imported at module scope (import it inside the function "
        "that needs it, or under TYPE_CHECKING when it is only a type):\n"
        + "\n".join(offenders)
    )


#: Run in a subprocess: matplotlib is imported by the time the suite reaches
#: this file, and a blocker installed in-process cannot undo that. A fresh
#: interpreter is the only place the base install can honestly be simulated.
_IMPORT_WITHOUT_MATPLOTLIB = """
import sys


class Blocker:
    '''Deny matplotlib and every submodule of it.

    The hook Python consults is ``find_spec``. A blocker written against the
    long-removed ``find_module`` protocol is simply never called, so matplotlib
    imports normally and the test passes while proving nothing at all.
    '''

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise ImportError("No module named %r (blocked)" % fullname)
        return None


for name in [n for n in sys.modules if n.split(".")[0] == "matplotlib"]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocker())

# Prove the blocker bites before trusting what it certifies.
try:
    import matplotlib
except ImportError:
    pass
else:
    raise AssertionError("the blocker let matplotlib through: gate proves nothing")

import phonometry

print(phonometry.__version__)
"""


def test_import_succeeds_without_matplotlib() -> None:
    """The base install is NumPy and SciPy: importing must not need plotting."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_MATPLOTLIB],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "import phonometry failed without matplotlib:\n" + result.stderr
    )


def test_showfilter_raises_helpful_error_without_matplotlib(monkeypatch) -> None:
    """Without matplotlib, plotting must fail with an actionable message."""
    import builtins

    import numpy as np
    import pytest

    from phonometry.filters import design

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    spectrum = np.array([1])
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"pip install phonometry\[plot\]"):
        design._showfilter(
            [], [1000.0], [1122.0], [891.0], 48000, spectrum, show=True, plot_file=None
        )
