#  Copyright (c) 2026. Jose Manuel Requena Plens

"""
Tests ensuring the package does not hijack the global matplotlib backend.

Importing phonometry must not force a specific (e.g. non-interactive)
backend, so the package can be used during interactive exploration
(IPython, Jupyter). See issue #52.
"""

import os
import subprocess
import sys


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


def test_filters_design_has_no_toplevel_matplotlib_import() -> None:
    """matplotlib must be imported lazily so the package works without it."""
    import ast
    import inspect

    from phonometry.filters import design

    tree = ast.parse(inspect.getsource(design))

    # Only imports inside a function body are lazy; module-scope imports
    # (even wrapped in try/if blocks) still run at import time.
    inside_functions: set[ast.AST] = set()
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(func):
                inside_functions.add(sub)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None) or ""
            aliases = [a.name for a in node.names]
            if any("matplotlib" in n for n in [module, *aliases]):
                assert node in inside_functions, (
                    "matplotlib imported at module scope in filters.design"
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

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"pip install phonometry\[plot\]"):
        design._showfilter(
            [], [1000.0], [1122.0], [891.0], 48000, np.array([1]), show=True, plot_file=None
        )
