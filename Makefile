# Virtual environment detection
VENV = .venv
BIN = $(VENV)/bin

# If venv doesn't exist, use system binaries
ifeq (,$(wildcard $(VENV)))
    PYTHON = python3
    RUFF = ruff
    MYPY = mypy
    BANDIT = bandit
    PNPM = pnpm
else
    PYTHON = $(BIN)/python3
    RUFF = $(BIN)/ruff
    MYPY = $(BIN)/mypy
    BANDIT = $(BIN)/bandit
    PNPM = pnpm
endif

# Deterministic figure rendering: pin numerical thread pools to one thread and
# fix the hash seed BEFORE the interpreter starts, so multi-threaded reductions
# and set ordering cannot perturb the committed SVG/PNG bytes across machines
# (this is what made the heavy compute figures flaky on CI). The scripts also
# set the thread vars internally; PYTHONHASHSEED can only be set from here.
FIGURE_ENV = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
	PYTHONHASHSEED=0

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

lint:
	$(RUFF) check .
	$(MYPY) src scripts

format:
	$(RUFF) check --fix .
	$(RUFF) format .

security:
	$(BANDIT) -r src

snyk:
	@echo "Running Snyk..."
	@if [ -f .env ]; then export $$(cat .env | xargs) && $(PNPM) exec snyk test --all-projects; else $(PNPM) exec snyk test --all-projects; fi

sonar:
	@echo "Running SonarQube Scanner..."
	@if [ -f .env ]; then export $$(cat .env | xargs) && $(PNPM) exec sonar-scanner; else $(PNPM) exec sonar-scanner; fi

graphs:
	# Clear the generated SVG/WebP first so a figure that is no longer produced
	# is actually removed (the generators only overwrite, never delete, so a
	# stale orphan would otherwise survive and slip past the staleness check).
	# Animations (*.gif/*.webm) come from the separate `animations` target and
	# are deliberately preserved.
	find .github/images -maxdepth 1 -type f \( -name '*.svg' -o -name '*.png' -o -name '*.webp' \) -delete
	$(FIGURE_ENV) $(PYTHON) scripts/generate_graphs.py
	$(FIGURE_ENV) $(PYTHON) scripts/generate_diagrams.py
	# Every shaded region has to be visible against the page it is drawn on,
	# on both themes; the staleness check cannot see that.
	$(PYTHON) scripts/check_figure_contrast.py

# Regenerate the Tier-1 documentation animations (WebM for the site, GIF for
# the GitHub docs). Kept out of `graphs`/CI because the ffmpeg encoding is slow
# and video is not byte-reproducible across platforms; run manually to refresh.
animations:
	$(PYTHON) scripts/generate_graphs.py --animations

# Re-extract only the deferred-loading poster stills (anim_*_poster.jpg) from
# the committed animation WebMs, without the slow clip re-encode. Posters are
# JPEG so they stay outside the SVG/PNG figure pipeline (`graphs` deletion and
# the check_figures.py staleness compare).
posters:
	$(PYTHON) scripts/generate_graphs.py --posters

# Regenerate the brand mark and every icon derived from it (.github/brand and
# the site's favicon, touch icon and PWA icons). Deliberately outside `graphs`:
# that target wipes .github/images first, and these are design assets rather
# than computed figures, so they are refreshed only when the mark changes.
brand:
	$(PYTHON) scripts/generate_brand.py

llms:
	$(PYTHON) scripts/generate_llms.py

# Regenerate README_PYPI.md (the PyPI long description) from README.md:
# theme-aware <picture> elements collapse to their light <img> fallback and
# animated GIFs to poster stills, because PyPI strips <picture>/<source>.
# The packaging tests fail if the committed file drifts.
pypi-readme:
	$(PYTHON) scripts/generate_pypi_readme.py

# Regenerate the committed Starlight API reference (site/src/content/docs/
# reference/api + site/src/generated/api-sidebar.mjs) from the source
# docstrings. CI fails if this drifts (see the api-docs job in python-app.yml).
api-docs:
	$(PYTHON) scripts/generate_api_docs.py

# Transplant the bodies of docs/CONFORMANCE.md and docs/ERRATA.md into their
# Starlight pages (site/src/content/docs/{,es/}reference/{conformance,errata}.md),
# below the hand-written introduction each page keeps. CI fails if this drifts
# (see the `site-reports` job in python-app.yml).
site-reports:
	$(PYTHON) scripts/generate_site_reports.py

# Lighthouse over a fixed sample of built pages, against a local preview
# server (BASE_URL overrides for the live site; `-- --desktop` for desktop
# throttling). Needs `pnpm build` in site/ first; summary on stdout, JSON
# reports in site/lighthouse-results/ (gitignored).
lighthouse:
	cd site && pnpm run lighthouse

# Regenerate the committed example .report() fiches under .github/reports/,
# which the documentation links to as rendered normative-report examples.
# Not byte-checked in CI (the embedded vector plot differs by ~1 ULP across
# CPUs); tests/test_generate_reports.py only checks the generator still works.
reports:
	$(PYTHON) scripts/generate_reports.py

# Regenerate the committed, versioned numerical conformance report. The
# --file-header flag prepends the "do not hand-edit" note; the body is exactly
# what the CI PR-comment harness computes. CI fails if this drifts (see the
# `conformance` job in python-app.yml).
conformance:
	$(PYTHON) scripts/conformance_report.py --file-header > docs/CONFORMANCE.md
	$(PYTHON) scripts/check_conformance_claims.py

# Optional convenience: install a git pre-commit hook that regenerates
# docs/CONFORMANCE.md when the library source or the report generator changes.
# The CI staleness check is the enforcement; this only saves a round-trip.
install-hooks:
	@mkdir -p .git/hooks
	@cp hooks/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Installed .git/hooks/pre-commit (regenerates docs/CONFORMANCE.md when src/scripts change)."

# Pin every numerical thread pool to one thread so the pytest-xdist workers
# (one per core) do not each spawn a nested BLAS/OpenMP pool and oversubscribe
# the CPU. With one worker per core already saturating the machine, nested
# threads only add contention: measured ~25% faster wall-clock and ~40% less
# total CPU on this suite. (PYTHONHASHSEED is deliberately left unset here so
# the tests still exercise randomised hash/set ordering.)
TEST_ENV = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

# -n auto fans the suite out across all CPU cores via pytest-xdist (workers are
# separate processes; pytest-cov combines their coverage data automatically).
test:
	$(TEST_ENV) $(PYTHON) -m pytest -n auto tests/

coverage:
	$(TEST_ENV) $(PYTHON) -m pytest -n auto --cov=src/phonometry --cov-report=term-missing tests/

check: lint security test

.PHONY: install lint format security snyk sonar graphs reports animations posters brand lighthouse \
	llms pypi-readme api-docs site-reports conformance install-hooks test coverage check