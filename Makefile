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

# Where the generators write down what the Spanish tables could not translate,
# for `make figure-language` to read afterwards (see
# scripts/figure_language_audit.py). Recording only happens when this is set,
# it costs nothing measurable, and it cannot change a rendered byte; the
# directory is under build/, which is gitignored.
FIGURE_LANGUAGE_DIR = build/figure-language
FIGURE_LANGUAGE_ENV = PHONOMETRY_FIGURE_LANGUAGE_AUDIT=$(FIGURE_LANGUAGE_DIR)

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
	# The language recording is per run, and the fragments accumulate: empty
	# the directory here so `make figure-language` cannot be answered by an
	# older, partial run.
	rm -rf $(FIGURE_LANGUAGE_DIR)
	$(FIGURE_ENV) $(FIGURE_LANGUAGE_ENV) $(PYTHON) scripts/generate_graphs.py
	$(FIGURE_ENV) $(FIGURE_LANGUAGE_ENV) $(PYTHON) scripts/generate_diagrams.py

# Every shaded region has to be visible against the page it is drawn on, on
# both themes; the staleness check cannot see that. Deliberately a target of
# its own rather than the tail of `graphs`: a contrast failure inside the
# generation step aborts it, and everything meant to run after generation
# (the staleness compare above all) is then skipped, so one illegible fill
# reads as "the figures could not be regenerated" instead of "this fill is
# invisible".
figure-contrast:
	$(PYTHON) scripts/check_figure_contrast.py

# The Spanish variant of a figure is the English one with its strings looked
# up in a table at save time, so a string nobody added to the table ships in
# English inside `X_es.svg` and every other gate stays green: the page is
# Spanish, the figure in it is not. This reads what the generation run above
# wrote down and fails on any untranslated string the committed baseline does
# not already record -- and on a baseline line that is no longer true. Needs a
# `make graphs` first; it is answering about that run, not about the tree.
figure-language:
	$(PYTHON) scripts/check_figure_language.py --audit $(FIGURE_LANGUAGE_DIR)

# What to run locally before committing a figure change: regenerate, then
# verify legibility and staleness the way CI does, each as its own step.
# Recipe lines rather than prerequisites: prerequisites are free to run
# concurrently under `make -j`, which would let the contrast checker parse the
# SVGs while the generators are still deleting and rewriting them.
figures:
	$(MAKE) graphs
	$(MAKE) figure-contrast
	$(MAKE) figure-language
	$(PYTHON) scripts/check_figures.py

# Regenerate the Tier-1 documentation animations (WebM for the site, GIF for
# the GitHub docs). Kept out of `graphs`/CI because the ffmpeg encoding is slow
# and video is not byte-reproducible across platforms; run manually to refresh.
animations:
	# Records into the same directory as `make graphs`, adding the clips to
	# whatever that run left there, so `make figure-language` afterwards sees
	# figures, plates and clips at once.
	$(FIGURE_LANGUAGE_ENV) $(PYTHON) scripts/generate_graphs.py --animations

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
	$(PYTHON) scripts/mirror_overviews.py
	$(PYTHON) scripts/mirror_glossary.py
	$(PYTHON) scripts/generate_llms.py

# Regenerate README_PYPI.md (the PyPI long description) from README.md:
# theme-aware <picture> elements collapse to their light <img> fallback and
# animated GIFs to poster stills, because PyPI strips <picture>/<source>.
# The packaging tests fail if the committed file drifts.
pypi-readme:
	$(PYTHON) scripts/generate_pypi_readme.py

# Run every Python snippet the guides print, hold the two languages to the
# same API and reject a block that shadows a name it imported (see the
# doc-snippets job in python-app.yml). `make snippets-static` skips the
# execution pass, which is the slow half.
snippets:
	$(PYTHON) scripts/check_doc_snippets.py

snippets-static:
	$(PYTHON) scripts/check_doc_snippets.py --static

# Catch a paragraph that wraps onto a "-" or a ">", which CommonMark reads as a
# new block and which then either takes the site build down or publishes a
# quoted block in the middle of a sentence (see the markdown-wrapping job).
hazards:
	$(PYTHON) scripts/check_markdown_hazards.py

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
# which the documentation links to as rendered normative-report examples. CI
# fails if this drifts (see the `reports` job in python-app.yml). The compare
# is tolerance-aware rather than a byte diff, for the same reason the figures'
# is: the embedded vector plot differs by ~1 ULP across CPUs. See
# scripts/check_reports.py.
#
# Renders into a scratch directory and swaps it in only once the whole set is
# written. Clearing the output first, which is how this used to work, means a
# generator that dies halfway leaves the working tree stripped of the committed
# examples and the maintainer reaching for `git checkout`. The clearing itself
# has to stay, because the generator only overwrites and never deletes, so a
# fiche that is no longer produced would survive as a stale orphan and slip
# past the staleness check; it just belongs after a successful run rather than
# before an attempted one. The trap covers the interrupted run the same way.
reports:
	set -e; \
	tmp=$$(mktemp -d .github/reports.tmp.XXXXXX); \
	old=$$(mktemp -d .github/reports.old.XXXXXX); \
	trap 'if [ -d "$$old"/current ]; then rm -rf .github/reports; mv "$$old"/current .github/reports; fi; rm -rf "$$tmp" "$$old"' EXIT INT TERM HUP; \
	$(FIGURE_ENV) $(PYTHON) scripts/generate_reports.py --output-dir "$$tmp"; \
	[ -n "$$(ls -A "$$tmp")" ] || { echo "no fiche was generated" >&2; exit 1; }; \
	mv .github/reports "$$old"/current; \
	mv "$$tmp" .github/reports; \
	mv "$$old"/current "$$old"/replaced; \
	rm -rf "$$old"

# Regenerate the committed, versioned numerical conformance report, then bring
# every count quoted from it into line. The --file-header flag prepends the
# "do not hand-edit" note; the body is exactly what the CI PR-comment harness
# computes. The second step rewrites the counts in the prose that has no build
# step to interpolate them through (.zenodo.json, the plain-markdown mirror
# under docs/, the site frontmatter); the Astro page bodies import them from
# site/src/data/conformance-stats.mjs and need nothing. CI fails if either
# output drifts (see the `conformance` job in python-app.yml).
conformance:
	$(PYTHON) scripts/conformance_report.py --file-header > docs/CONFORMANCE.md
	$(PYTHON) scripts/check_conformance_claims.py --write

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

.PHONY: install lint format security snyk sonar graphs figure-contrast figure-language figures reports \
	animations posters brand lighthouse \
	llms pypi-readme api-docs site-reports conformance install-hooks test coverage check \
	snippets snippets-static
