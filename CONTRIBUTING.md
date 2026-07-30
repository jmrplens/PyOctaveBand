# Contributing to phonometry

Thank you for your interest in contributing to phonometry! We welcome contributions from the community to help improve this project.

## 💬 Where to Ask

Not every contribution starts with code. [Discussions](https://github.com/jmrplens/phonometry/discussions)
is the right place for anything that is not yet a defect or a pull request:

| You want to | Go to |
|---|---|
| Ask how to compute or measure something with the library | [Q&A](https://github.com/jmrplens/phonometry/discussions/categories/q-a) |
| Ask whether a clause or edition is implemented, report a value that disagrees with the standard or with a reference implementation, or flag a defect in a published standard | [Standards & conformance](https://github.com/jmrplens/phonometry/discussions/categories/standards-conformance) |
| Propose a standard, method or feature | [Ideas](https://github.com/jmrplens/phonometry/discussions/categories/ideas) |
| Give feedback on the guides, figures or animations | [Documentation & learning](https://github.com/jmrplens/phonometry/discussions/categories/documentation-learning) |
| Share a measurement, study or tool you built | [Show and tell](https://github.com/jmrplens/phonometry/discussions/categories/show-and-tell) |
| Report a reproducible failure with no standard in play, or work on an agreed change | [Issues](https://github.com/jmrplens/phonometry/issues) |
| Report a suspected security vulnerability | Privately, following the [security policy](SECURITY.md) |

A discrepancy against a standard or a reference implementation belongs in
Standards & conformance even when it reproduces every time, because settling it
means reading the clause first. Issues is for crashes, results that contradict
the library's own documentation, and broken installations.

The documentation is published in English and Spanish, and both languages are
welcome in Discussions.

Two conventions worth knowing before posting about a standard:

- Cite clauses, tables and equations by number rather than pasting substantial
  verbatim text. Standards are copyrighted.
- A proposal moves faster when it names reference data the implementation can be
  validated against. Methods here are implemented from the standard and checked
  against reference values, never from a textbook summary alone.

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🛠️ Development Setup

To set up your development environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jmrplens/phonometry.git
   cd phonometry
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   Install both production and development dependencies to run tests and linters.
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## ✅ Code Quality Standards

We enforce strict code quality standards. Before submitting a Pull Request, please ensure your code passes the following checks:

### 1. Type Checking (MyPy)
We use strict type checking. Ensure no errors are reported:
```bash
mypy .
```

### 2. Linting & Formatting (Ruff)
We use `ruff` for fast linting and formatting.
```bash
ruff check .
```

### 3. Testing (Pytest)
Run the full test suite to ensure no regressions. We aim for 100% code coverage.
```bash
pytest tests/
```
To check coverage locally:
```bash
pytest --cov=src/phonometry --cov-report=term-missing tests/
```

### 3b. Oracle data (committed vs local)

Some suites are validated against reference material that is too large or not
redistributable. Each of them keeps a small committed oracle under
`tests/data/<dataset>/` (a derived measurement series, or a lossless extract of
a representative subset) and prefers a full local copy when one is available.
Resolution order, applied by `tests/oracle_data.py`: the dataset's environment
variable, then the gitignored `tests/data-local/<dataset>/`, then the committed
data. **The assertions on the committed data never skip** — that is what CI
runs. Tests that exist only to exercise the full set, and have no committed
counterpart, do skip there and say so. `pytest` prints which copy each dataset
resolved to in its run header.

To run a suite against the full original set, drop it in `tests/data-local/`:

| Dataset | `tests/data-local/…` | Environment override |
| :--- | :--- | :--- |
| [stipa.info STIPA verification bench](https://www.stipa.info/index.php/download-test-signals) | `stipa-verification/` | `STIPA_VERIFICATION_DATA` |
| [EBU loudness test set](https://tech.ebu.ch/publications/ebu_loudness_test_set) | `ebu-loudness-test-set/` | `EBU_LOUDNESS_TEST_SET` |
| NORAH2 V2.0.74 public release | `norah2/NORAH2_V2.0.74_public.zip` | `NORAH2_DATA` (extraction root) |

See `tests/data/README.md` for the convention and what each committed oracle
can and cannot assert.

### 4. Documentation Images (auto-generated)
Every image under `.github/images/` is generated with the library itself by
`scripts/generate_graphs.py` (plots) and `scripts/generate_diagrams.py` (setup /
signal-flow diagrams) — never hand-made. If your change alters filter responses,
weighting curves or any other plotted behavior, regenerate the graphs and commit
the affected images together with the code change:

```bash
make graphs   # runs both: python scripts/generate_graphs.py && python scripts/generate_diagrams.py
make figures  # the same generation, then the legibility and staleness checks CI runs
```

When adding a feature with visual output, add a `generate_*` function to
`scripts/generate_graphs.py` (or `scripts/generate_diagrams.py` for a diagram) and
reference the resulting image from the docs — do not commit images produced any other way.

### 5. Conformance report (auto-generated)
`docs/CONFORMANCE.md` is generated by `scripts/conformance_report.py` from the
library's own computations against the standards — never hand-edited. If CI's
`conformance` job flags it as stale (or you changed a check, `src/phonometry`, or
a reference table), regenerate and commit it:

```bash
make conformance   # writes docs/CONFORMANCE.md
```

Optionally run `make install-hooks` once to install a pre-commit hook that does
this automatically when relevant sources change; the CI check is the enforcement,
the hook is just convenience.

## 🏷️ Naming Conventions

All identifiers follow PEP 8 with the project-specific rules below (validated
against numpy/scipy, pandas, matplotlib, scikit-learn, statsmodels and librosa).

| Group | Convention | Examples |
|---|---|---|
| Modules | `snake_case` **concept** name, ≤3-4 words — never a standard number (the standard designation lives in the docstring and the guide) | `noise_induced_hearing_loss.py`, `sound_power.py` |
| Classes | `PascalCase`; the primary result of a computation is `<Concept>Result`; psychoacoustic family `<Method><Metric>` is sanctioned; value/input objects get a plain name | `ImpulseProminenceResult`, `ZwickerLoudness`, `Quantity` |
| Functions | `snake_case` noun phrase for computations; a verb only for actions/predicates; no `get_`/`calculate_` prefixes | `reverberation_time`, `apply_weighting`, `sensitivity` (not `calculate_sensitivity`) |
| Public constants | `UPPER_SNAKE_CASE`, **no unit suffixes** — units go in the docstring | `OCTAVE_BANDS`, `REFERENCE_ACCELERATION` |
| Private constants | `_UPPER_SNAKE` named after the standard's table | `_TABLE1`, `_UVL0` |
| Parameters | `snake_case` with the canonical vocabulary: `fs`, `frequencies`, `volume`, `relative_humidity`, `temperature`, `x` (time signal); durations carry `_s`/`_hours` only where mixed units coexist | — |
| Type aliases / Literals | `PascalCase` aliases; plain-string `Literal` values | `Real`, `Sex = Literal["male", "female"]` |
| Warnings | `<Topic>Warning`, all inheriting from `PhonometryWarning` | `OccupationalExposureWarning` |
| Spelling | American English in identifiers | `normalized_frequencies`, `BAND_CENTERS` |
| Tests | `test_<module>.py`, 1:1 with the module; cross-cutting suites get a descriptive name | `test_impulse_prominence.py` |

### Deprecations

Renames of **published** API keep the old name working for one cycle:

- Warn with `warnings.warn(msg, DeprecationWarning, stacklevel=...)` using
  the NEP 23 message format (deprecated-since version, removal version, and
  the replacement to use). Pick the `stacklevel` so the warning points at the
  *caller's* line: `2` when warning directly from the deprecated function,
  `3` when the warning is emitted through a shared helper.
- Renamed **modules**: keep a shim using a PEP 562 module `__getattr__`
  (scipy pattern).
- Renamed **keyword arguments**: accept the old keyword with a `"deprecated"`
  string sentinel default (scikit-learn pattern) and forward to the new one.
- Each alias gets a `pytest.warns(DeprecationWarning)` test.
- Deprecated names are removed only in a **major** release. Unpublished API
  (merged since the last release) is renamed outright, without shims.

## 📦 Releasing

Releases are fully automated from the repository-root `VERSION` file:

1. Open a PR that bumps `VERSION` (semver) and moves the `[Unreleased]`
   CHANGELOG section to the new version.
2. When the PR merges to `main`, the release workflow validates the version,
   builds the package, publishes to PyPI and creates the GitHub Release
   (tag included). Zenodo registers the DOI from the release webhook.

Never push tags manually; the workflow is idempotent (an existing tag is
skipped).

## 🚀 How to Contribute

### Reporting Bugs
Search the [Issues](https://github.com/jmrplens/phonometry/issues) first, then open a
new one. There is a form for each kind:

| Form | For |
|---|---|
| Bug report | A crash, a result that contradicts the library's own documentation, or a broken installation |
| Conformance defect | A disagreement with a standard that has already been established, usually in a discussion |
| Implement a standard or method | An agreed work item, once the scope and the reference data are settled |
| Documentation defect | A wrong value or formula, an example that no longer runs, a broken link |

Pull requests are labelled automatically from the paths they touch, through
`.github/labeler.yml`. Adding a package under `src/phonometry/` means adding its
`area:` entry there too.

### Pull Requests
1. **Fork** the repository.
2. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`).
3. **Commit** your changes with clear messages.
4. **Verify** your code using the commands above (`pytest`, `mypy`, `ruff`).
5. **Push** to your fork and **Open a Pull Request**.

The pull request description is prefilled from
[`.github/pull_request_template.md`](.github/pull_request_template.md). Its
checklist is the CI gates written out, including the regeneration steps above
and the rule that matters most here: a new normative implementation must name
the numeric oracle it was validated against, and that oracle must be independent
of the implementation.

### Security

A suspected vulnerability goes through
[GitHub's private reporting](https://github.com/jmrplens/phonometry/security/advisories/new)
rather than an issue or a discussion. The [security policy](SECURITY.md) sets
out what is treated as a vulnerability, what is not (a value that disagrees with
a standard is a conformance defect, not a security issue), and what response to
expect.

## License
By contributing to this project, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file.
