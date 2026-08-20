#  Copyright (c) 2026. Jose Manuel Requena Plens

"""Tests ensuring the package treats matplotlib as the optional dependency it is.

Three promises are kept here. Importing phonometry must not force a specific
(e.g. non-interactive) backend, so the package can be used during interactive
exploration (IPython, Jupyter); see issue #52. The base install must run on
NumPy and SciPy alone, so ``import phonometry`` has to succeed with matplotlib
absent, plotting failing only when a plot is actually asked for. And the io
module makes the same promise about soundfile (the ``[audio]`` extra):
``phonometry.io`` must import, and read WAV files, with neither optional
package present.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
    return any(name == "matplotlib" or name.startswith("matplotlib.") for name in names)


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Whether ``test`` is the one condition that never runs.

    Only a bare ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING`` qualifies. This
    used to ask whether the unparsed test *contained* the name, which is the
    same question with the wrong answer for the two conditions that matter:
    ``if not TYPE_CHECKING:`` runs exactly when the guard does not, and
    ``if TYPE_CHECKING or enabled:`` runs whenever ``enabled`` is true. Both
    read as guarded to a substring search, so an import placed in either would
    have been waved through by the gate meant to catch it.
    """
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _eager_matplotlib_imports(
    body: list[ast.stmt], *, guarded: bool = False
) -> list[int]:
    """Line numbers of the matplotlib imports that run at import time.

    Only two placements are lazy: inside a function body, and under
    ``if TYPE_CHECKING:``, which never executes. Everything else at module
    scope runs the moment the package is imported, including a class body and
    the branches of a runtime ``if``, a ``try``, a ``with``, a loop or a
    ``match`` -- so this descends into those rather than assuming a bare
    top-level statement is the only way to import something eagerly.

    The compound statements are enumerated rather than walked generically
    because the distinction being drawn is exactly the one a generic walk
    erases: a function body is the *point* of the lazy placement and must not
    be descended into, while every other block runs on import and must be.
    """
    eager: list[int] = []
    for node in body:
        if _imports_matplotlib(node):
            if not guarded:
                eager.append(node.lineno)
        elif isinstance(node, ast.If):
            # ``else`` is the runtime branch of a TYPE_CHECKING guard.
            deferred = guarded or _is_type_checking_guard(node.test)
            eager += _eager_matplotlib_imports(node.body, guarded=deferred)
            eager += _eager_matplotlib_imports(node.orelse, guarded=guarded)
        elif isinstance(node, ast.Try):
            for block in (
                node.body,
                node.orelse,
                node.finalbody,
                *(handler.body for handler in node.handlers),
            ):
                eager += _eager_matplotlib_imports(block, guarded=guarded)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # A loop that never iterates still executes its ``else``.
            for block in (node.body, node.orelse):
                eager += _eager_matplotlib_imports(block, guarded=guarded)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                eager += _eager_matplotlib_imports(case.body, guarded=guarded)
        elif isinstance(node, (ast.With, ast.AsyncWith, ast.ClassDef)):
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


#: ``(label, module source, is it eager?)``. The gate above is only as good as
#: this classifier, and a gate that quietly answers "nothing to report" is
#: worse than no gate, so the classifier is pinned against the placements that
#: decide it: the two conditions that merely mention ``TYPE_CHECKING`` without
#: being it, the block statements that run on import, and the two placements
#: that genuinely are lazy and must stay unreported.
_PLACEMENTS = [
    ("type-checking guard", "if TYPE_CHECKING:\n    import matplotlib\n", False),
    ("qualified guard", "if typing.TYPE_CHECKING:\n    import matplotlib\n", False),
    ("function body", "def f():\n    import matplotlib\n", False),
    ("negated guard", "if not TYPE_CHECKING:\n    import matplotlib\n", True),
    ("guard in an or", "if TYPE_CHECKING or x:\n    import matplotlib\n", True),
    (
        "guard's else",
        "if TYPE_CHECKING:\n    pass\nelse:\n    import matplotlib\n",
        True,
    ),
    ("for body", "for i in x:\n    import matplotlib\n", True),
    ("for else", "for i in x:\n    pass\nelse:\n    import matplotlib\n", True),
    ("while body", "while x:\n    import matplotlib\n", True),
    ("match case", "match v:\n    case _:\n        import matplotlib\n", True),
    ("with body", "with ctx():\n    import matplotlib\n", True),
    ("class body", "class C:\n    import matplotlib\n", True),
    ("try body", "try:\n    import matplotlib\nexcept ImportError:\n    pass\n", True),
]


@pytest.mark.parametrize(
    ("label", "source", "eager"),
    _PLACEMENTS,
    ids=[label for label, _, _ in _PLACEMENTS],
)
def test_eager_import_classifier(label: str, source: str, eager: bool) -> None:
    """The classifier behind the gate reports exactly the eager placements."""
    found = _eager_matplotlib_imports(ast.parse(source).body)
    assert bool(found) is eager, (
        f"{label}: expected {'an eager' if eager else 'no'} import, got {found}"
    )


#: Run in a subprocess: matplotlib is imported by the time the suite reaches
#: this file, and a blocker installed in-process cannot undo that. A fresh
#: interpreter is the only place the base install can honestly be simulated.
#: The script is a template over the packages to deny and the modules whose
#: import is being certified, so the same proven blocker serves every
#: optional dependency instead of a per-package copy drifting apart.
_IMPORT_WITH_BLOCKED_PACKAGES = """
import sys

BLOCKED = {blocked!r}


class Blocker:
    '''Deny the blocked packages and every submodule of them.

    The hook Python consults is ``find_spec``. A blocker written against the
    long-removed ``find_module`` protocol is simply never called, so the
    package imports normally and the test passes while proving nothing at all.
    '''

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError("No module named %r (blocked)" % fullname)
        return None


for name in [n for n in sys.modules if n.split(".")[0] in BLOCKED]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocker())

# Prove the blocker bites before trusting what it certifies.
for package in BLOCKED:
    try:
        __import__(package)
    except ImportError:
        pass
    else:
        raise AssertionError(
            "the blocker let %r through: gate proves nothing" % package
        )

for module in {imports!r}:
    __import__(module)

{payload}
import phonometry

print(phonometry.__version__)
"""

#: Payload for the io test: forge a minimal 16-bit WAV with the stdlib and
#: read it back, so the subprocess certifies not just that ``phonometry.io``
#: imports on the base install but that its core job runs there.
_READ_WAV_PAYLOAD = """
import struct, tempfile, os
import numpy as np
import phonometry.io

fmt = struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16)
data = struct.pack("<4h", 16384, -32768, 0, 32767)
image = (b"RIFF" + struct.pack("<I", 28 + len(data)) + b"WAVE"
         + b"fmt " + struct.pack("<I", 16) + fmt
         + b"data" + struct.pack("<I", len(data)) + data)
fd, path = tempfile.mkstemp(suffix=".wav")
try:
    with os.fdopen(fd, "wb") as fh:
        fh.write(image)
    sig = phonometry.io.read(path)
finally:
    os.remove(path)
assert np.asarray(sig).tolist() == [0.5, -1.0, 0.0, 32767 / 32768], sig
assert sig.fs == 48000
"""


def _import_with_blocked(
    blocked: tuple[str, ...], imports: tuple[str, ...], payload: str = ""
) -> None:
    """Run the template in a fresh interpreter and fail with its stderr."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_WITH_BLOCKED_PACKAGES.format(
                blocked=blocked, imports=imports, payload=payload
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"import of {imports} failed with {blocked} blocked:\n" + result.stderr
    )


def test_import_succeeds_without_matplotlib() -> None:
    """The base install is NumPy and SciPy: importing must not need plotting."""
    _import_with_blocked(("matplotlib",), ("phonometry",))


def test_io_imports_and_reads_without_matplotlib_and_soundfile() -> None:
    """The io module must import, and read WAV, on the bare base install.

    soundfile is the ``[audio]`` extra, imported only inside the backend
    functions that need it -- the same contract matplotlib has with
    ``.plot()``. Blocking both at once certifies the whole promise: with
    neither optional package present the module imports and a 16-bit WAV
    reads correctly, and only a call that actually needs the missing
    package may raise.
    """
    _import_with_blocked(
        ("matplotlib", "soundfile"),
        ("phonometry", "phonometry.io"),
        _READ_WAV_PAYLOAD,
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
            msg = "No module named 'matplotlib'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    spectrum = np.array([1])
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"pip install phonometry\[plot\]"):
        design._showfilter(
            [], [1000.0], [1122.0], [891.0], 48000, spectrum, show=True, plot_file=None
        )
