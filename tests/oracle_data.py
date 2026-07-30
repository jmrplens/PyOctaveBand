#  Copyright (c) 2026. Jose M. Requena-Plens
"""Where the heavy, non-redistributable oracle sets come from.

A few suites are validated against reference material that cannot be
committed: it is either far too large for a repository (hundreds of megabytes
of PCM that does not compress as audio) or licensed for use but not for
redistribution. Each of those suites keeps a small **committed** oracle under
``tests/data/`` (a derived measurement series, or a lossless extract of a
representative subset) and prefers the full original when it is available
locally.

Resolution order, identical for every dataset:

1. the dataset's environment variable, when it is set and points at an
   existing copy;
2. ``tests/data-local/<directory>``, gitignored, where a full local copy of
   the original set is expected to live;
3. the committed data under ``tests/data/`` - the fallback that CI always
   takes. **It never skips**: the committed oracle asserts real values, and
   each suite documents in its module docstring what the full set covers that
   the committed one cannot.

``pytest_report_header`` in ``tests/conftest.py`` prints the resolution of
every dataset at the top of each run, so a green run is never ambiguous about
which oracle produced it.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

#: Gitignored root for full local copies of the original sets.
DATA_LOCAL = pathlib.Path(__file__).parent / "data-local"

#: Versioned root for the committed derived series and extracts.
DATA = pathlib.Path(__file__).parent / "data"


@dataclass(frozen=True)
class LocalDataset:
    """A heavy oracle set kept outside the repository."""

    #: Short identifier used in the run header.
    key: str
    #: Directory name under :data:`DATA_LOCAL` holding the full copy.
    directory: str
    #: Historical environment override pointing at a copy elsewhere.
    env_var: str
    #: Path inside ``tests/data-local/<directory>`` that must exist for the
    #: copy to count as complete.
    marker: str
    #: One-line description of the committed fallback under ``tests/data/``.
    committed: str
    #: Marker for the environment override, when it points at a differently
    #: shaped copy (``None`` reuses :attr:`marker`).
    env_marker: str | None = None


@dataclass(frozen=True)
class Resolution:
    """Which copy of a :class:`LocalDataset` this run will read."""

    dataset: LocalDataset
    #: Root of the full local copy, or ``None`` when the committed data is used.
    path: pathlib.Path | None
    #: ``"env"``, ``"data-local"`` or ``"committed"``.
    origin: str

    @property
    def is_full_set(self) -> bool:
        """True when a full local copy of the original set was found."""
        return self.path is not None

    def describe(self) -> str:
        """One-line report of the resolution, for the pytest run header."""
        if self.origin == "env":
            return f"{self.dataset.key}: full set from ${self.dataset.env_var}"
        if self.origin == "data-local":
            return f"{self.dataset.key}: full set from tests/data-local/{self.dataset.directory}"
        return f"{self.dataset.key}: committed oracle ({self.dataset.committed})"


def resolve(dataset: LocalDataset) -> Resolution:
    """Apply the resolution order above to ``dataset``."""
    override = os.environ.get(dataset.env_var)
    marker = dataset.env_marker or dataset.marker
    if override and (pathlib.Path(override) / marker).exists():
        return Resolution(dataset, pathlib.Path(override), "env")
    local = DATA_LOCAL / dataset.directory
    if (local / dataset.marker).exists():
        return Resolution(dataset, local, "data-local")
    return Resolution(dataset, None, "committed")


#: The certified IEC 60268-16 STIPA verification bench (stipa.info).
STIPA_BENCH = LocalDataset(
    key="stipa-verification",
    directory="stipa-verification",
    env_var="STIPA_VERIFICATION_DATA",
    marker="Annex C.3.2",
    committed="27 of 49 bench signals, tests/data/stipa/",
)

#: The EBU loudness test set (EBU Tech 3341 / 3342 sequences).
EBU_LOUDNESS_SET = LocalDataset(
    key="ebu-loudness-test-set",
    directory="ebu-loudness-test-set",
    env_var="EBU_LOUDNESS_TEST_SET",
    marker="seq-3341-7_seq-3342-5-24bit.wav",
    committed="block-loudness series, tests/data/broadcast/",
)

#: The NORAH2 V2.0.74 public release (EASA.2020.FC.06 prototype and database).
NORAH2_RELEASE = LocalDataset(
    key="norah2",
    directory="norah2",
    env_var="NORAH2_DATA",
    marker="NORAH2_V2.0.74_public.zip",
    committed="44-file ARP subset, tests/data/norah2/",
    # NORAH2_DATA has always pointed at an already extracted release root.
    env_marker="Hemispheres",
)

#: Every dataset reported in the pytest run header.
DATASETS: tuple[LocalDataset, ...] = (STIPA_BENCH, EBU_LOUDNESS_SET, NORAH2_RELEASE)
