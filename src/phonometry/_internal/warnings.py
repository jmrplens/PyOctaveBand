#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Warning hierarchy root (private module, public class).

Every warning the library emits derives from :class:`PhonometryWarning`
so users can filter or escalate all phonometry diagnostics with a single
:func:`warnings.filterwarnings` rule.
"""

from __future__ import annotations


class PhonometryWarning(UserWarning):
    """Base class for all phonometry warnings."""
