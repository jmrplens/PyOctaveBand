<!--
Keep it short. The checklist below is this repository's CI gates written out, so
working through it honestly is the quickest route to a green build. Tick what
applies and delete the lines that do not.
-->

## What and why

<!-- One or two sentences: what changes, and the problem it solves. Link the
     issue or discussion it came from. -->

## Validation

<!-- Required for any new normative implementation, and the rule this project is
     built on: name the numeric oracle each new method is checked against (a
     published worked example, a printed table, or an exact closed form) and
     confirm it was obtained independently of the code in this pull request,
     never by running the implementation. Cite the source in the test docstring
     too. Write "no new computation" if that is the case. -->

## Checklist

Ran locally, same as CI:

- [ ] `ruff check .`, plus `ruff format --check` on the files you touched
- [ ] `mypy src scripts`
- [ ] `bandit -r src` (any finding fails, including Low)
- [ ] `pytest -q`

Regenerated where this change touches them:

- [ ] `make conformance`, with `docs/CONFORMANCE.md` and everything else it
      rewrites committed (it also updates the counts quoted in `.zenodo.json`,
      `docs/` and the site frontmatter, so never edit those by hand)
- [ ] `make api-docs`, with `site/src/content/docs/reference/api/` and
      `site/src/generated/api-sidebar.mjs` committed
- [ ] `make llms`, with `llms.txt`, `llms-full.txt` and the shards under
      `site/public/llms/` committed
- [ ] `make figures` (generation, then `scripts/check_figure_contrast.py` and
      `scripts/check_figures.py`), with the regenerated images committed
      (figures come from `scripts/generate_graphs.py`, never by hand)
- [ ] `make pypi-readme` after editing `README.md`

Applies to new API:

- [ ] New modules registered in `scripts/api_taxonomy.py`
- [ ] `docs/api-reference.md` covers every new `__all__` name
      (`python scripts/check_api_reference.py`)
- [ ] Result objects expose `.plot()`, with shared plotting code in
      `src/phonometry/_plot/`
- [ ] New package paths added to `.github/labeler.yml`

Always:

- [ ] Documentation updated in English and Spanish, kept in step
- [ ] Any defect found in a published source recorded in `docs/ERRATA.md`
- [ ] CHANGELOG entry under `[Unreleased]`
