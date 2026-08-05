#  Copyright (c) 2026. Jose Manuel Requena Plens
"""One module per subject area of the conformance report.

Each module registers its checks at import time and holds nothing else: the
constants and oracles those checks need, and the prose that says where each
expected value comes from. A module is a subject an acoustician would look for
by name, not a bucket sized to make the arithmetic work, so a few of them are
short and one of them is long.

Importing a module is what puts its rows in the report. The list that decides
which modules are imported, and in which order, lives in
``scripts/conformance_report.py``; that list is the report's section order.
"""
