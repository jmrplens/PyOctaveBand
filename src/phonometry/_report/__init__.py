#  Copyright (c) 2026. Jose Manuel Requena Plens
"""PDF report renderers for the library's result objects.

reportlab is a *soft* dependency: importing :mod:`phonometry` and running any
computation works without it, and only calling a result's ``report()`` method
(or the functions here) requires it. The import is performed lazily and raises
a clear :class:`ImportError` with installation guidance
(``pip install phonometry[report]``) when the package is missing, mirroring the
matplotlib guard behind the ``plot()`` methods.
"""

from __future__ import annotations

from .metadata import ReportMetadata

__all__ = ["ReportMetadata"]
