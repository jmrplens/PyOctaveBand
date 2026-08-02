#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Architecture rules for the phonometry package layout (Phase 1 overhaul).

Static (ast-based) enforcement of the dependency policy between the domain
subpackages, plus a fresh-interpreter smoke import per subpackage. The edge
whitelist is the contract from the modularization plan: keep it tight; adding
an edge is an explicit, reviewed decision.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "phonometry"

#: The transverse toolbox every domain is allowed to import: normalized
#: frequency selectivity, general signal analysis and the metrology proper.
#: One package until 4.0 split it in three; the policy is unchanged.
TOOLBOX: frozenset[str] = frozenset({"filters", "signals", "metrology"})

#: Cross-package edges allowed IN ADDITION to `pkg -> pkg` (internal),
#: `* -> _internal` and `* -> TOOLBOX`. "root" = modules still at the top
#: level of the package (shrinks to the facade set as the migration proceeds).
ALLOWED_EDGES: set[tuple[str, str]] = {
    ("environmental", "materials"),   # air_absorption -> ISO 354 helpers
    ("aircraft", "environmental"),    # atmospheric absorption reuse
    ("vibration", "hearing"),         # multiple-shock SEXES tables
    # swept-sine distortion reuses the ISO 18233 sweep / Farina
    # inverse-filter machinery of room_ir
    ("electroacoustics", "room"),
    # predicted panel R reuses the plate coincidence frequency (radiation)
    ("building", "vibration"),
    # double-wall cavity fill uses the porous equivalent-fluid model
    ("building", "materials"),
    # HVAC plenum and machine enclosures reuse the room constant
    # R = S*alpha/(1 - alpha) of the steady-state room field
    ("noise_control", "room"),
}


def _package_of(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def _iter_modules() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if p.name != "__init__.py" or p.parent != SRC]


def _edges() -> set[tuple[str, str, str]]:
    """(from_pkg, to_pkg, 'file: import') for every relative import in src."""
    out: set[tuple[str, str, str]] = set()
    for path in _iter_modules():
        pkg = _package_of(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0:
                # Absolute self-imports would bypass the relative-import rules.
                if node.module and (node.module == "phonometry"
                                    or node.module.startswith("phonometry.")):
                    pytest.fail(f"{path.relative_to(SRC)}: absolute self-import "
                                f"'{node.module}' (use relative imports)")
                continue
            target = node.module or ""
            head = target.split(".")[0] if target else ""
            if node.level == 1 and pkg == "root":
                to_pkg = head if (SRC / head).is_dir() else "root"
            elif node.level == 1:
                to_pkg = pkg  # sibling inside the same subpackage
            else:  # level == 2 from inside a subpackage
                to_pkg = head if (SRC / head).is_dir() else "root"
            out.add((pkg, to_pkg, f"{path.relative_to(SRC)}: {ast.dump(node)[:60]}"))
    return out


def test_internal_imports_no_domain_code() -> None:
    for frm, to, where in _edges():
        if frm == "_internal":
            assert to == "_internal", f"_internal must stay leaf-level: {where}"


def test_cross_package_edges_are_whitelisted() -> None:
    violations = []
    for frm, to, where in _edges():
        if frm == to or to in ("_internal", "root") or frm in (
            "root", "_plot", "_report"
        ):
            # root modules are unrestricted during the migration; the facade
            # (__init__, _compat) legitimately imports everything. _plot and
            # _report are rendering leaves that reference domain classes only
            # under TYPE_CHECKING (see the guarantee test below).
            continue
        if to in TOOLBOX:
            continue
        if to in ("_plot", "_report"):
            # lazy .plot()/.report() imports only; enforced structurally by the
            # fact that _plot/_report modules import domain classes under
            # TYPE_CHECKING.
            continue
        if (frm, to) not in ALLOWED_EDGES:
            violations.append(f"{frm} -> {to} ({where})")
    assert not violations, "unlisted cross-package imports:\n" + "\n".join(violations)


def test_render_modules_only_type_check_domain_imports() -> None:
    checked = False
    for sub in ("_plot", "_report"):
        render_dir = SRC / sub
        if not render_dir.is_dir():
            continue
        checked = True
        for path in render_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only
                if isinstance(node, ast.ImportFrom) and node.level == 2:
                    pytest.fail(
                        f"{sub}/{path.name}: module-level import of domain code "
                        "(must live under TYPE_CHECKING)"
                    )
    if not checked:
        pytest.skip("neither _plot nor _report created yet")


@pytest.mark.parametrize("pkg", sorted(
    p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("__")
))
def test_subpackage_imports_in_fresh_interpreter(pkg: str) -> None:
    subprocess.run(
        [sys.executable, "-c", f"import phonometry.{pkg}"],
        check=True, capture_output=True, timeout=120,
    )


def test_subpackage_reexports_cover_facade_imports() -> None:
    """Every name the facade imports from a domain submodule must also be
    reachable on the subpackage namespace (the ``env.name`` usage pattern).

    The check is on the top-level package of the import, at whatever depth the
    module sits: ``from .vibration.human.exposure import daily_exposure`` has
    to leave ``vibration.daily_exposure`` reachable.
    """
    import importlib

    facade = ast.parse((SRC / "__init__.py").read_text(encoding="utf-8"))
    missing: list[str] = []
    for node in facade.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        parts = node.module.split(".")
        # A domain module is ``<pkg>.<module>`` or, since the domains grew a
        # second level, ``<pkg>.<family>.<module>``. Checking only the
        # two-part form silently stopped covering a domain the day it grew.
        if len(parts) < 2 or parts[0].startswith("_"):
            continue
        pkg = importlib.import_module(f"phonometry.{parts[0]}")
        for alias in node.names:
            if not hasattr(pkg, alias.name):
                missing.append(f"phonometry.{parts[0]}.{alias.name}")
    assert not missing, "facade imports not re-exported by their subpackage:\n" + "\n".join(missing)
