#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Private plot renderers, one module per domain plus shared helpers.

Result objects import their renderer lazily inside ``.plot()`` so matplotlib
stays an optional dependency; renderer modules import domain result classes
only under ``TYPE_CHECKING`` (keeps the plotting layer cycle-free).
"""
