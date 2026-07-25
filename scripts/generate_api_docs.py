#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Generate the Starlight API reference from the phonometry docstrings.

Renders every name in ``phonometry.__all__`` as committed Markdown pages under
``site/src/content/docs/reference/api/`` (one page per public module, grouped
by the sections in ``scripts/api_taxonomy.py``) plus the sidebar fragment
``site/src/generated/api-sidebar.mjs``. English only by decision; Spanish
routes are served by the Starlight locale fallback.

Stdlib only (``inspect``, ``importlib``, ``ast``, ``re``) plus importing
phonometry itself. Output is byte-deterministic: members are sorted, nothing
is timestamped, and no environment-dependent text is embedded. CI fails when
the committed pages drift from source (see .github/workflows/python-app.yml).

Regenerate with ``make api-docs``.

The docstring dialect handled here is the repo's reST field-list style:
``:param:``/``:type:``/``:return:``/``:rtype:``/``:raises:``/``:warns:``/
``:ivar:``/``:vartype:`` fields, ``:class:``/``:func:``/etc. cross-reference
roles, ````literal```` double-backtick spans, ``.. note::`` directives and
``::`` literal blocks. A docstring that does not parse cleanly is degraded to
a verbatim block and reported, never fatal.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import itertools
import json
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from api_taxonomy import (
    OBJECT_MODULE_OVERRIDES,
    SECTIONS,
    Section,
    module_section,
)

import phonometry

ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "site" / "src" / "content" / "docs" / "reference" / "api"
SIDEBAR_PATH = ROOT / "site" / "src" / "generated" / "api-sidebar.mjs"

#: Site base path (GitHub Pages project site); links validator checks against it.
SITE_BASE = "/phonometry"
API_BASE = f"{SITE_BASE}/reference/api"

BANNER = (
    "> Auto-generated from the source docstrings by "
    "`scripts/generate_api_docs.py` (`make api-docs`). Do not edit by hand."
)

_FIELD_NAMES = (
    "param",
    "type",
    "return",
    "returns",
    "rtype",
    "raises",
    "raise",
    "ivar",
    "vartype",
    "warns",
)
_FIELD_RE = re.compile(
    r"^:(" + "|".join(_FIELD_NAMES) + r")(\s+[^:]*)?:\s?(.*)$"
)
#: Any line that looks like a reST field (to detect unsupported field names).
_FIELD_LIKE_RE = re.compile(r"^:(\w+)(\s+[^:`]*)?:(\s|$)")
#: A supported field starting mid-line inside another field's description
#: (roles like ``:func:`` are excluded by the field-name alternation).
_MIDLINE_FIELD_RE = re.compile(
    r"(?<=\s):(" + "|".join(_FIELD_NAMES) + r")(\s+[^:`]*)?:(\s|$)"
)
ROLE_RE = re.compile(
    r":(?:class|func|meth|mod|attr|data|obj|exc|ref|term):"
    r"`(~?)([^`<]+?)(?:\s*<([^>]+)>)?`"
)
_NOTE_RE = re.compile(r"^(\s*)\.\. note::\s?(.*)$")
_DIRECTIVE_RE = re.compile(r"^\s*\.\. \w+::")
_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")

_MAX_CONST_REPR = 1500
_MAX_SIGNATURE_LINE = 76


# --------------------------------------------------------------------------
# Docstring parsing (reST field lists)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ParsedDoc:
    """A docstring split into prose and field-list entries (raw reST text)."""

    description: str = ""
    epilogue: str = ""
    params: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    types: dict[str, str] = dataclasses.field(default_factory=dict)
    returns: str = ""
    rtype: str = ""
    raises: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    warns: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    ivars: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    vartypes: dict[str, str] = dataclasses.field(default_factory=dict)
    issues: list[str] = dataclasses.field(default_factory=list)


def parse_docstring(doc: str) -> ParsedDoc:
    """Parse a reST field-list docstring into prose and typed fields.

    Never raises on malformed input: problems are recorded in ``issues`` and
    the caller degrades the docstring to a verbatim block.
    """
    parsed = ParsedDoc()
    desc_lines: list[str] = []
    epi_lines: list[str] = []
    seen_field = False
    field: tuple[str, str] | None = None  # (field name, argument)
    body: list[str] = []

    def flush() -> None:
        nonlocal field, body
        if field is None:
            return
        name, arg = field
        text = " ".join(part.strip() for part in body if part.strip())
        jammed = _MIDLINE_FIELD_RE.search(text)
        if jammed:
            # A field started mid-line (e.g. two ``:param:`` entries jammed
            # on one line) would otherwise be published inside the previous
            # field's description; fail loudly instead.
            parsed.issues.append(
                f"field {jammed.group(1)!r} starts mid-line inside the"
                f" description of {name!r}"
            )
        if name == "param":
            parsed.params.append((arg, text))
        elif name == "type":
            parsed.types[arg] = text
        elif name in ("return", "returns"):
            parsed.returns = f"{parsed.returns} {text}".strip()
        elif name == "rtype":
            parsed.rtype = text
        elif name in ("raises", "raise"):
            parsed.raises.append((arg, text))
        elif name == "warns":
            parsed.warns.append((arg, text))
        elif name == "ivar":
            parsed.ivars.append((arg, text))
        elif name == "vartype":
            parsed.vartypes[arg] = text
        field = None
        body = []

    for line in doc.splitlines():
        match = _FIELD_RE.match(line)
        if match:
            flush()
            seen_field = True
            field = (match.group(1), (match.group(2) or "").strip())
            body = [match.group(3)]
            continue
        like = _FIELD_LIKE_RE.match(line)
        if like and like.group(1) not in _FIELD_NAMES:
            flush()
            parsed.issues.append(f"unsupported field {like.group(1)!r}")
            continue
        if field is not None:
            if not line.strip() or line[:1].isspace():
                body.append(line)
                continue
            flush()
        target = epi_lines if seen_field else desc_lines
        target.append(line)

    flush()
    parsed.description = "\n".join(desc_lines).strip("\n")
    parsed.epilogue = "\n".join(epi_lines).strip("\n")
    return parsed


# --------------------------------------------------------------------------
# reST -> Markdown rendering
# --------------------------------------------------------------------------


@dataclasses.dataclass
class RoleStats:
    """Counts of cross-reference roles resolved to links vs degraded."""

    resolved: int = 0
    degraded: int = 0


def rest_roles_to_links(
    text: str, xref: dict[str, str], stats: RoleStats | None = None
) -> str:
    """Rewrite reST roles to intra-site Markdown links.

    ``:func:`leq``` becomes ``[`leq`](<page>#leq)`` when ``leq`` is in the
    xref map; unknown targets degrade to plain inline code.
    """

    def lookup(name: str) -> str | None:
        candidates = [name]
        if name.startswith("phonometry."):
            candidates.append(name.removeprefix("phonometry."))
        for candidate in candidates:
            url = xref.get(candidate)
            if url is not None:
                return url
        url = xref.get(name.rsplit(".", 1)[-1])
        if url is not None:
            return url
        # Attribute references (Class.attr): fall back to the owner's anchor.
        for candidate in candidates:
            while "." in candidate:
                candidate = candidate.rsplit(".", 1)[0]
                if candidate == "phonometry":
                    break
                url = xref.get(candidate)
                if url is not None:
                    return url
        return None

    def sub(match: re.Match[str]) -> str:
        tilde, name, explicit = match.group(1), match.group(2), match.group(3)
        # Role targets wrapped across source lines contain interior whitespace.
        target_name = re.sub(r"\s+", "", explicit or name)
        if explicit:
            # ":role:`display <target>`" form: keep the display wording.
            display = re.sub(r"\s+", " ", name).strip()
        else:
            display = re.sub(r"\s+", "", name)
        if tilde:
            display = display.rsplit(".", 1)[-1]
        url = lookup(target_name)
        if url is None:
            if stats is not None:
                stats.degraded += 1
            return f"`{display}`"
        if stats is not None:
            stats.resolved += 1
        return f"[`{display}`]({url})"

    return ROLE_RE.sub(sub, text)


def _consume_indented(lines: list[str], start: int, indent: int) -> int:
    """Return the index just past the block indented deeper than ``indent``."""
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if len(line) - len(line.lstrip()) > indent:
            i += 1
            continue
        break
    return i


def _dedent_block(lines: list[str]) -> list[str]:
    indents = [
        len(line) - len(line.lstrip()) for line in lines if line.strip()
    ]
    if not indents:
        return [line.strip() for line in lines]
    cut = min(indents)
    return [line[cut:] if line.strip() else "" for line in lines]


def rest_blocks_to_markdown(text: str, code: list[str]) -> str:
    """Convert reST block constructs to Markdown structure.

    ``.. note::`` directives become Starlight ``:::note`` asides and ``::``
    literal blocks become fenced code blocks. Fenced code is stored in
    ``code`` and replaced by placeholders so the inline pass (roles, escaping)
    cannot touch it; ``_restore_code`` swaps it back.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        note = _NOTE_RE.match(line)
        if note:
            indent = len(note.group(1))
            end = _consume_indented(lines, i + 1, indent)
            content = _dedent_block(lines[i + 1 : end])
            if note.group(2).strip():
                content = [note.group(2).strip(), *content]
            inner = rest_blocks_to_markdown("\n".join(content).strip("\n"), code)
            out.extend(["", ":::note", inner, ":::", ""])
            i = end
            continue
        stripped = line.rstrip()
        if stripped.endswith("::") and not _DIRECTIVE_RE.match(line):
            indent = len(line) - len(line.lstrip())
            end = _consume_indented(lines, i + 1, indent)
            block = _dedent_block(lines[i + 1 : end])
            while block and not block[0].strip():
                block.pop(0)
            while block and not block[-1].strip():
                block.pop()
            lead = stripped[:-2].rstrip()
            if lead:
                out.append(f"{lead}:")
            fence = "\n".join(["```text", *block, "```"])
            token = f"\x00CODE{len(code)}\x00"
            code.append(fence)
            out.extend(["", token, ""])
            i = end
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _restore_code(text: str, code: list[str]) -> str:
    for index, fence in enumerate(code):
        text = text.replace(f"\x00CODE{index}\x00", fence)
    return text


def _inline_markdown(text: str, xref: dict[str, str], stats: RoleStats) -> str:
    """Inline pass: roles to links, literals to code spans, escaping."""
    tokens: list[str] = []

    def stash(rendered: str) -> str:
        tokens.append(rendered)
        return f"\x00T{len(tokens) - 1}\x00"

    # 1. Cross-reference roles (contain backticks; handle before code spans).
    text = ROLE_RE.sub(
        lambda m: stash(rest_roles_to_links(m.group(0), xref, stats)), text
    )
    # 2. reST ``literal`` spans -> Markdown code spans.
    text = re.sub(r"``(.+?)``", r"`\1`", text, flags=re.DOTALL)
    # 3. Protect code spans from escaping.
    text = re.sub(r"`[^`]+`", lambda m: stash(m.group(0)), text)
    # 4. Escape characters Markdown/HTML would misread in plain prose.
    text = text.replace("<", "\\<")
    # Intraword asterisks (unit notation "Pa*s/m", math "10*lg(x)") would
    # pair up into <em> spans; reST emphasis never sits between word
    # characters, so escaping exactly those is safe.
    text = re.sub(r"(?<=\w)\*(?=\w)", "\\\\*", text)
    # 5. Restore protected spans.
    return re.sub(r"\x00T(\d+)\x00", lambda m: tokens[int(m.group(1))], text)


def render_prose(text: str, xref: dict[str, str], stats: RoleStats) -> str:
    """Full reST -> Markdown pipeline for block prose."""
    code: list[str] = []
    text = rest_blocks_to_markdown(text, code)
    text = _inline_markdown(text, xref, stats)
    text = _restore_code(text, code)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def render_inline(text: str, xref: dict[str, str], stats: RoleStats) -> str:
    """Inline reST -> Markdown collapsed to a single line (no pipe escaping)."""
    rendered = _inline_markdown(text, xref, stats)
    return rendered.replace("\n", " ").strip()


def render_cell(text: str, xref: dict[str, str], stats: RoleStats) -> str:
    """Inline reST -> Markdown for a table cell (single line, pipes escaped).

    The ``|`` escape is GFM table syntax: it is decoded back to a literal
    pipe only inside a table row, so it must never be applied to prose
    outside a table (a code span there would show the backslash literally).
    """
    return render_inline(text, xref, stats).replace("|", "\\|")


# --------------------------------------------------------------------------
# Introspection model
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MethodDoc:
    """One public method, property or nested callable of a class."""

    name: str  # qualified: "ClassName.method"
    kind: str  # "method", "classmethod", "staticmethod", "property"
    heading: str
    anchor: str
    signature: str | None
    doc: str


@dataclasses.dataclass(frozen=True)
class MemberDoc:
    """One public name documented on a module page."""

    name: str
    kind: str  # "function", "class", "constant"
    anchor: str
    signature: str | None
    doc: str
    init_doc: str = ""  # classes: docstring of an explicit __init__
    methods: tuple[MethodDoc, ...] = ()
    type_name: str = ""  # constants only
    value_repr: str = ""  # constants only


@dataclasses.dataclass(frozen=True)
class ModuleDoc:
    """One generated page: a public module and its members."""

    module: str  # full dotted name ("phonometry" for the top level)
    section: Section
    slug: str  # file/URL slug within the section
    title: str
    label: str
    intro: str
    members: tuple[MemberDoc, ...]

    @property
    def url(self) -> str:
        return f"{API_BASE}/{self.section.key}/{self.slug}/"

    @property
    def relpath(self) -> str:
        return f"{self.section.key}/{self.slug}.md"


class _Slugger:
    """github-slugger-compatible anchors (what Astro assigns to headings)."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def slug(self, text: str) -> str:
        base = re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")
        count = self._seen.get(base, 0)
        self._seen[base] = count + 1
        return base if count == 0 else f"{base}-{count}"


def _sorted_names(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: (n.lower(), n))


def _is_public_module(module: str) -> bool:
    return not any(part.startswith("_") for part in module.split(".")[1:])


def attribute_module(name: str, obj: object) -> str:
    """Return the full public module that documents public name ``name``."""
    override = OBJECT_MODULE_OVERRIDES.get(name)
    if override is not None:
        return override
    module = getattr(obj, "__module__", None)
    if isinstance(module, str) and _is_public_module(module):
        return module
    if isinstance(module, str):
        raise LookupError(  # noqa: TRY004 - a private-module lookup is a configuration error, not a type error
            f"public name {name!r} is defined in private module {module!r}; "
            "add it to OBJECT_MODULE_OVERRIDES in scripts/api_taxonomy.py"
        )
    # Module-level constant: find the taxonomy module that defines it.
    hits: list[str] = []
    for section in SECTIONS.values():
        for candidate in section.modules:
            if candidate == "phonometry":
                continue
            mod = importlib.import_module(candidate)
            if getattr(mod, name, _SENTINEL) is obj:
                hits.append(candidate)
    if len(hits) == 1:
        return hits[0]
    raise LookupError(
        f"constant {name!r} matches modules {hits!r}; add it to "
        "OBJECT_MODULE_OVERRIDES in scripts/api_taxonomy.py"
    )


_SENTINEL = object()


def _attribute_docstrings(module: ModuleType) -> dict[str, str]:
    """PEP 258 attribute docstrings (string literal after an assignment)."""
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return {}
    tree = ast.parse(source)
    docs: dict[str, str] = {}
    body = tree.body
    for stmt, follower in itertools.pairwise(body):
        target: ast.expr | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
        if (
            isinstance(target, ast.Name)
            and isinstance(follower, ast.Expr)
            and isinstance(follower.value, ast.Constant)
            and isinstance(follower.value.value, str)
        ):
            docs[target.id] = inspect.cleandoc(follower.value.value)
    return docs


def format_signature(
    name: str,
    obj: Any,
    *,
    drop_first: bool = False,
    omit_return: bool = False,
) -> str:
    """Render ``name(params) -> ret`` from ``inspect.signature``.

    Stringized annotations (PEP 563, the repo style) are emitted verbatim.
    Multi-line when the one-line form would be hard to read.
    """
    signature = inspect.signature(obj)
    parameters = list(signature.parameters.values())
    if drop_first and parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]

    parts: list[str] = []
    saw_positional_only = False
    star_emitted = False
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            saw_positional_only = True
        elif saw_positional_only:
            parts.append("/")
            saw_positional_only = False
        if (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and not star_emitted
        ):
            parts.append("*")
            star_emitted = True
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            star_emitted = True
        parts.append(_format_parameter(parameter))
    if saw_positional_only:
        parts.append("/")

    returns = ""
    if not omit_return and signature.return_annotation is not (
        inspect.Signature.empty
    ):
        returns = f" -> {_format_annotation(signature.return_annotation)}"
    one_line = f"{name}({', '.join(parts)}){returns}"
    if len(one_line) <= _MAX_SIGNATURE_LINE or not parts:
        return one_line
    joined = ",\n    ".join(parts)
    return f"{name}(\n    {joined},\n){returns}"


def _format_annotation(annotation: object) -> str:
    if isinstance(annotation, str):
        # PEP 563 stringizes annotations; an annotation that was already a
        # string literal in the source keeps its quotes. Strip them.
        text = annotation.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1]
        return text
    return inspect.formatannotation(annotation)


def _format_parameter(parameter: inspect.Parameter) -> str:
    text = parameter.name
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        text = f"*{text}"
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
        text = f"**{text}"
    annotated = parameter.annotation is not inspect.Parameter.empty
    if annotated:
        text = f"{text}: {_format_annotation(parameter.annotation)}"
    if parameter.default is not inspect.Parameter.empty:
        if type(parameter.default).__module__ not in ("builtins", "types"):
            # Third-party reprs (numpy arrays) are not pinned, so a repr here
            # would make the drift gate hostage to dependency upgrades; the
            # parameter description documents the effective default.
            default = "..."
        else:
            default = _HEX_ADDR_RE.sub("...", repr(parameter.default))
        text = f"{text} = {default}" if annotated else f"{text}={default}"
    return text


def _class_methods(
    name: str, cls: type, slugger: _Slugger, issues: list[str]
) -> tuple[MethodDoc, ...]:
    methods: list[MethodDoc] = []
    for attr_name in _sorted_names(
        [n for n in vars(cls) if not n.startswith("_")]
    ):
        raw = inspect.getattr_static(cls, attr_name)
        qualified = f"{name}.{attr_name}"
        kind: str
        func: object
        if isinstance(raw, staticmethod):
            kind, func = "staticmethod", raw.__func__
        elif isinstance(raw, classmethod):
            kind, func = "classmethod", raw.__func__
        elif isinstance(raw, property):
            kind, func = "property", raw.fget
        elif inspect.isfunction(raw):
            kind, func = "method", raw
        else:
            continue  # class attributes, enum members: documented via :ivar:
        doc = inspect.getdoc(func) or ""
        signature: str | None = None
        if kind != "property":
            try:
                signature = format_signature(
                    qualified, func, drop_first=kind != "staticmethod"
                )
            except (ValueError, TypeError) as exc:
                issues.append(f"{qualified}: no signature ({exc})")
        heading = qualified if kind == "property" else f"{qualified}()"
        methods.append(
            MethodDoc(
                name=qualified,
                kind=kind,
                heading=heading,
                anchor=slugger.slug(heading),
                signature=signature,
                doc=doc,
            )
        )
    return tuple(methods)


def _constant_repr(obj: object) -> str:
    if type(obj).__module__ not in ("builtins", "types"):
        return ""  # third-party reprs (numpy arrays) are not pinned; skip
    value: object = dict(obj) if isinstance(obj, Mapping) else obj
    text = repr(value)
    if len(text) > _MAX_CONST_REPR or _HEX_ADDR_RE.search(text):
        return ""
    return text


def _type_name(obj: object) -> str:
    cls = type(obj)
    shape = getattr(obj, "shape", None)
    if isinstance(shape, tuple):
        return f"{cls.__module__}.{cls.__qualname__}, shape {shape}"
    if cls.__module__ == "builtins":
        return cls.__qualname__
    if cls.__module__ == "types" and cls.__qualname__ == "MappingProxyType":
        return "mapping"
    return f"{cls.__module__}.{cls.__qualname__}"


def build_model() -> tuple[list[ModuleDoc], dict[str, str], list[str]]:
    """Introspect phonometry into pages, the xref map and parse issues."""
    issues: list[str] = []
    names: list[str] = list(phonometry.__all__)
    if len(names) != len(set(names)):
        raise ValueError("phonometry.__all__ contains duplicate names")

    by_module: dict[str, list[str]] = {}
    for name in names:
        module = attribute_module(name, getattr(phonometry, name))
        module_section(module)  # fails loudly for unmapped modules
        by_module.setdefault(module, []).append(name)

    mapped = {m for s in SECTIONS.values() for m in s.modules}
    stale = sorted(mapped - set(by_module))
    if stale:
        raise ValueError(
            f"taxonomy modules with no public names (remove or fix): {stale}"
        )

    pages: list[ModuleDoc] = []
    xref: dict[str, str] = {}
    for section in SECTIONS.values():
        for module_name in section.modules:
            page = _build_page(
                module_name, section, by_module[module_name], issues
            )
            pages.append(page)
            xref[module_name] = page.url
            for member in page.members:
                xref[member.name] = f"{page.url}#{member.anchor}"
                for method in member.methods:
                    xref[method.name] = f"{page.url}#{method.anchor}"

    slugs = {(page.section.key, page.slug) for page in pages}
    if len(slugs) != len(pages):
        raise ValueError("slug collision between module pages")

    # Docstrings still reference the flat pre-modularization module names
    # (":mod:`phonometry.insulation`"); alias them to the current pages.
    labels: dict[str, list[str]] = {}
    for page in pages:
        if page.module != "phonometry":
            labels.setdefault(page.label, []).append(page.url)
    for label, urls in labels.items():
        alias = f"phonometry.{label}"
        if alias not in xref and len(urls) == 1:
            xref[alias] = urls[0]
    for old, new in _LEGACY_MODULE_ALIASES.items():
        if old not in xref and new in xref:
            xref[old] = xref[new]
    return pages, xref, issues


#: Flat pre-modularization module names whose basename also changed; the
#: unchanged ones are aliased automatically from the page basenames above.
_LEGACY_MODULE_ALIASES = {
    "phonometry.aircraft_atmospheric_absorption": (
        "phonometry.aircraft.atmospheric_absorption"
    ),
    "phonometry.environmental_measurement": (
        "phonometry.environmental.measurement"
    ),
    "phonometry.underwater_acoustics": "phonometry.underwater.acoustics",
    "phonometry.underwater_propagation": "phonometry.underwater.propagation",
    "phonometry.underwater_sound_speed": "phonometry.underwater.sound_speed",
}


_TOP_LEVEL_INTRO = (
    "Package-level names defined in `phonometry/__init__.py` itself. "
    "Every public name in the library can also be imported directly from "
    "`phonometry`; this page documents the few objects that live at the "
    "top level."
)


def _build_page(
    module_name: str,
    section: Section,
    names: list[str],
    issues: list[str],
) -> ModuleDoc:
    top_level = module_name == "phonometry"
    module = importlib.import_module(module_name)
    attribute_docs = _attribute_docstrings(module)
    relative = module_name.removeprefix("phonometry.")
    basename = relative.rsplit(".", 1)[-1] if not top_level else "phonometry"
    intro = _TOP_LEVEL_INTRO if top_level else inspect.cleandoc(module.__doc__ or "")

    slugger = _Slugger()
    members: list[MemberDoc] = []
    for name in _sorted_names(names):
        obj = getattr(phonometry, name)
        anchor = slugger.slug(name)
        if inspect.isclass(obj):
            # Warning/exception classes inherit object.__init__; a signature
            # block would be noise (and CPython refuses to introspect it).
            signature = (
                None
                if issubclass(obj, BaseException)
                else _try_signature(name, obj, issues, omit_return=True)
            )
            # Raw __doc__ (not inspect.getdoc): generated __init__ methods
            # have none, and getdoc would inherit object.__init__'s
            # "Initialize self" placeholder.
            init = vars(obj).get("__init__")
            init_raw = getattr(init, "__doc__", None) if init is not None else None
            init_doc = inspect.cleandoc(init_raw) if init_raw else ""
            members.append(
                MemberDoc(
                    name=name,
                    kind="class",
                    anchor=anchor,
                    signature=signature,
                    doc=inspect.getdoc(obj) or "",
                    init_doc=init_doc,
                    methods=_class_methods(name, obj, slugger, issues),
                )
            )
        elif inspect.isroutine(obj):
            signature = _try_signature(name, obj, issues)
            members.append(
                MemberDoc(
                    name=name,
                    kind="function",
                    anchor=anchor,
                    signature=signature,
                    doc=inspect.getdoc(obj) or "",
                )
            )
        else:
            members.append(
                MemberDoc(
                    name=name,
                    kind="constant",
                    anchor=anchor,
                    signature=None,
                    doc=attribute_docs.get(name, ""),
                    type_name=_type_name(obj),
                    # __version__ changes every release; embedding its value
                    # would make the drift gate fail on each version bump.
                    value_repr="" if name == "__version__" else _constant_repr(obj),
                )
            )
    title = "phonometry" if top_level else relative
    return ModuleDoc(
        module=module_name,
        section=section,
        slug=basename.replace("_", "-"),
        title=title,
        label=basename,
        intro=intro,
        members=tuple(members),
    )


def _try_signature(
    name: str, obj: Any, issues: list[str], *, omit_return: bool = False
) -> str | None:
    try:
        return format_signature(name, obj, omit_return=omit_return)
    except (ValueError, TypeError) as exc:
        issues.append(f"{name}: no signature ({exc})")
        return None


# --------------------------------------------------------------------------
# Page rendering
# --------------------------------------------------------------------------


def _heading_text(name: str) -> str:
    """Escape names Markdown would emphasize (dunders: ``__version__``)."""
    if name.startswith("_") or name.endswith("_"):
        return name.replace("_", "\\_")
    return name


def _frontmatter(title: str, description: str, label: str) -> str:
    return "\n".join(
        [
            "---",
            f"title: {json.dumps(title)}",
            f"description: {json.dumps(description)}",
            "sidebar:",
            f"  label: {json.dumps(label)}",
            "---",
        ]
    )


def _field_table(
    heading: str,
    columns: tuple[str, str],
    rows: list[tuple[str, str]],
) -> list[str]:
    if not rows:
        return []
    out = [
        f"**{heading}**",
        "",
        f"| {columns[0]} | {columns[1]} |",
        "| :--- | :--- |",
    ]
    out.extend(f"| {left} | {right} |" for left, right in rows)
    out.append("")
    return out


def _render_docbody(
    owner: str,
    doc: str,
    xref: dict[str, str],
    stats: RoleStats,
    failures: list[str],
    *,
    param_note: str = "Parameters",
) -> list[str]:
    """Render one docstring (prose + field tables) to Markdown lines."""
    if not doc:
        return []
    parsed = parse_docstring(doc)
    if parsed.issues:
        failures.append(f"{owner}: {'; '.join(parsed.issues)}")
        return ["```text", doc, "```", ""]

    out: list[str] = []
    if parsed.description:
        out.extend([render_prose(parsed.description, xref, stats), ""])

    params = [
        (
            f"`{name}`",
            _with_type(parsed.types.get(name), desc, xref, stats),
        )
        for name, desc in parsed.params
    ]
    out.extend(_field_table(param_note, ("Name", "Description"), params))

    ivars = [
        (
            f"`{name}`",
            _with_type(parsed.vartypes.get(name), desc, xref, stats),
        )
        for name, desc in parsed.ivars
    ]
    out.extend(_field_table("Attributes", ("Name", "Description"), ivars))

    if parsed.returns:
        returns = render_inline(parsed.returns, xref, stats)
        if parsed.rtype:
            rtype = render_inline(parsed.rtype, xref, stats)
            returns = f"{returns} (*{rtype}*)" if returns else f"*{rtype}*"
        out.extend([f"**Returns:** {returns}", ""])

    raises = [
        (render_cell(exc, xref, stats), render_cell(desc, xref, stats))
        for exc, desc in parsed.raises
    ]
    out.extend(_field_table("Raises", ("Exception", "When"), raises))

    warns = [
        (render_cell(exc, xref, stats), render_cell(desc, xref, stats))
        for exc, desc in parsed.warns
    ]
    out.extend(_field_table("Warns", ("Warning", "When"), warns))

    if parsed.epilogue:
        out.extend([render_prose(parsed.epilogue, xref, stats), ""])
    return out


def _with_type(
    type_text: str | None,
    desc: str,
    xref: dict[str, str],
    stats: RoleStats,
) -> str:
    rendered = render_cell(desc, xref, stats)
    if type_text:
        rendered = f"(*{render_cell(type_text, xref, stats)}*) {rendered}"
    return rendered


def render_module_page(
    page: ModuleDoc,
    xref: dict[str, str],
    stats: RoleStats,
    failures: list[str],
) -> str:
    """Render one module page to Markdown."""
    if page.module == "phonometry":
        description = (
            "Top-level convenience API of the phonometry package "
            "(auto-generated)."
        )
    else:
        description = f"Public API of {page.module} (auto-generated)."
    out = [
        _frontmatter(page.title, description, page.label),
        "",
        BANNER,
        "",
    ]
    if page.intro:
        out.extend([render_prose(page.intro, xref, stats), ""])

    for member in page.members:
        out.extend([f"## {_heading_text(member.name)}", ""])
        if member.kind == "constant":
            out.append(f"*Constant* (`{member.type_name}`).")
            out.append("")
            if member.doc:
                out.extend([render_prose(member.doc, xref, stats), ""])
            if member.value_repr:
                out.extend(
                    ["```python", f"{member.name} = {member.value_repr}", "```", ""]
                )
            continue
        if member.signature:
            out.extend(["```python", member.signature, "```", ""])
        out.extend(
            _render_docbody(
                f"{page.module}.{member.name}",
                member.doc,
                xref,
                stats,
                failures,
                param_note="Parameters",
            )
        )
        if member.init_doc:
            out.extend(
                _render_docbody(
                    f"{page.module}.{member.name}.__init__",
                    member.init_doc,
                    xref,
                    stats,
                    failures,
                    param_note="Parameters",
                )
            )
        for method in member.methods:
            out.extend([f"### {method.heading}", ""])
            if method.kind != "method":
                out.extend([f"*{method.kind}*", ""])
            if method.signature:
                out.extend(["```python", method.signature, "```", ""])
            out.extend(
                _render_docbody(
                    f"{page.module}.{method.name}",
                    method.doc,
                    xref,
                    stats,
                    failures,
                )
            )
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n") + "\n"


def render_index(
    pages: list[ModuleDoc],
    xref: dict[str, str],
    stats: RoleStats,
) -> str:
    """Render the API index page (one table per section)."""
    out = [
        "---",
        'title: "API Reference"',
        ('description: "Every public function, class and constant in '
        'phonometry, generated from the source docstrings."'),
        "---",
        "",
        BANNER,
        "",
        ("The complete public API, one page per module. Import the domain "
        "subpackage and call through it:"),
        "",
        "```python",
        "from phonometry import metrology, underwater",
        "",
        "spl, freq = metrology.octave_filter(x, fs)",
        "snr = underwater.passive_sonar_equation(185.0, 60.0, 50.0)",
        "```",
        "",
        ("Every documented name can also be imported directly from the "
        "top-level package (`from phonometry import leq`)."),
        "",
        ":::note",
        ("The API reference is generated from the English source docstrings "
        "and is published in English only."),
        "",
        ("La referencia de la API se genera a partir de los docstrings del "
        "código (en inglés) y se publica únicamente en inglés; las rutas en "
        "español muestran esta versión inglesa como alternativa."),
        ":::",
        "",
    ]
    by_section: dict[str, list[ModuleDoc]] = {}
    for page in pages:
        by_section.setdefault(page.section.key, []).append(page)
    for section in SECTIONS.values():
        out.extend(
            [
                f"## {section.label_en}",
                "",
                "| Module | Summary |",
                "| :--- | :--- |",
            ]
        )
        for page in by_section[section.key]:
            # First sentence of the first paragraph (an opening sentence may
            # wrap across several physical source lines).
            paragraph = (
                page.intro.strip().split("\n\n", 1)[0].replace("\n", " ")
            )
            summary = (
                paragraph.split(". ", 1)[0].rstrip(".") + "."
                if paragraph
                else ""
            )
            cell = render_cell(summary, xref, stats)
            out.append(f"| [`{page.title}`]({page.url}) | {cell} |")
        out.append("")
    return "\n".join(out).strip("\n") + "\n"


def render_sidebar(pages: list[ModuleDoc]) -> str:
    """Render the Starlight sidebar fragment (ESM, imported by astro.config)."""

    def js(text: str) -> str:
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"

    lines = [
        "// Auto-generated by scripts/generate_api_docs.py (make api-docs).",
        "// Do not edit by hand.",
        "export const apiSidebar = {",
        "  label: 'API reference',",
        "  translations: { es: 'Referencia de la API' },",
        "  items: [",
        "    { slug: 'reference/api', attrs: { 'data-group-link': true } },",
    ]
    by_section: dict[str, list[ModuleDoc]] = {}
    for page in pages:
        by_section.setdefault(page.section.key, []).append(page)
    for section in SECTIONS.values():
        lines.extend(
            [
                "    {",
                f"      label: {js(section.label_en)},",
                f"      translations: {{ es: {js(section.label_es)} }},",
                "      items: [",
            ]
        )
        lines.extend(
            f"        {js(f'reference/api/{section.key}/{page.slug}')},"
            for page in by_section[section.key]
        )
        lines.extend(["      ],", "    },"])
    lines.extend(["  ],", "};", ""])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Report:
    """What a generation run produced (printed by main, asserted by tests)."""

    pages: int
    members: int
    stats: RoleStats
    failures: list[str]
    issues: list[str]


def generate(content_dir: Path, sidebar_path: Path) -> Report:
    """Generate all pages and the sidebar fragment into the given paths."""
    pages, xref, issues = build_model()
    stats = RoleStats()
    failures: list[str] = []

    if content_dir.exists():
        shutil.rmtree(content_dir)
    content_dir.mkdir(parents=True)
    (content_dir / "index.md").write_text(
        render_index(pages, xref, stats), encoding="utf-8", newline="\n"
    )
    for page in pages:
        path = content_dir / page.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_module_page(page, xref, stats, failures),
            encoding="utf-8",
            newline="\n",
        )
    sidebar_path.parent.mkdir(parents=True, exist_ok=True)
    sidebar_path.write_text(
        render_sidebar(pages), encoding="utf-8", newline="\n"
    )
    members = sum(len(page.members) for page in pages)
    return Report(
        pages=len(pages) + 1,
        members=members,
        stats=stats,
        failures=failures,
        issues=issues,
    )


def main() -> int:
    report = generate(CONTENT_DIR, SIDEBAR_PATH)
    print(
        f"API reference: {report.pages} pages, {report.members} public names."
    )
    print(
        f"Cross-references: {report.stats.resolved} linked, "
        f"{report.stats.degraded} degraded to inline code."
    )
    for issue in report.issues:
        print(f"note: {issue}")
    if report.failures:
        print("Docstrings degraded to verbatim blocks:")
        for failure in report.failures:
            print(f"  - {failure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
