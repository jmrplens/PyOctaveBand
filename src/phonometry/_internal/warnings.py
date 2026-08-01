#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Warning hierarchy root (private module, public class).

Every warning the library emits derives from :class:`PhonometryWarning`
so users can filter or escalate all phonometry diagnostics with a single
:func:`warnings.filterwarnings` rule.
"""

from __future__ import annotations

import warnings


class PhonometryWarning(UserWarning):
    """Base class for all phonometry warnings."""


def _warn_renamed(old: str, new: str, *, stacklevel: int = 3,
                  since: str = "3.1", removed_in: str = "4.0") -> None:
    """Emit the NEP 23 rename notice for a deprecated alias.

    Shared helper for the one-cycle deprecation shims (renamed modules,
    functions and keyword arguments). The default ``stacklevel=3`` makes the
    warning point at the *caller* of the deprecated name when this helper is
    invoked directly from the deprecated function's body.

    :param old: The deprecated name, as shown to the user.
    :param new: The canonical replacement, as shown to the user.
    :param stacklevel: Frames between :func:`warnings.warn` and the caller.
    :param since: The phonometry minor release that deprecated the name.
    :param removed_in: The major release that removes the alias. The 3.x
        aliases go in 4.0; the aliases the 4.0 taxonomy introduces go in 5.0.
    """
    warnings.warn(
        f"{old} is deprecated since phonometry {since} and will be removed in "
        f"{removed_in}; use {new}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
