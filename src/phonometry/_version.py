#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Package version.

The canonical version lives in the repository-root ``VERSION`` file (the
build backend reads it via ``[tool.setuptools.dynamic]``). Installed
packages resolve it from their metadata; running from a source tree falls
back to reading the file directly.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phonometry")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    from pathlib import Path

    __version__ = (Path(__file__).resolve().parents[2] / "VERSION").read_text().strip()
