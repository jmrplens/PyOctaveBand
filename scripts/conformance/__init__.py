#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The numerical conformance registry, split by subject.

``scripts/conformance_report.py`` is still the entry point and the facade;
this package holds what that file used to hold inline. :mod:`conformance.registry`
owns the registry itself and the ``Outcome`` vocabulary every check speaks,
:mod:`conformance.shared` the filter and weighting computations the registry
and the report's showcase table both call, :mod:`conformance.domains` one
module per subject area, and :mod:`conformance.render` the Markdown.

Nothing in ``domains`` is imported for a name. Those modules are imported for
their side effect - the registration of their checks - and the order the entry
point imports them in is the order of the report.
"""

# ``registry`` is imported eagerly here because it is what puts ``tests/`` on
# ``sys.path``, and every other module in the package reads the test suite's
# shared reference tables at import time. Importing any of them first would
# otherwise depend on who got there before it.
from . import registry

__all__ = ["registry"]

