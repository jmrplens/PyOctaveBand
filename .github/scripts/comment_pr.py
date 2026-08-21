"""Assemble the PR comment body.

The numerical conformance report (produced by ``scripts/conformance_report.py``)
is the lead content; the per-Python-version test and coverage table is kept in
a collapsed section below it.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_test_results(test_dir):
    summary = []
    total_tests = 0
    total_failures = 0

    if not Path(test_dir).exists():
        return "No test results found.", 0, 0

    # Distinguish between test results and coverage reports
    test_files = []
    coverage_files = {}  # version -> path

    for root, _, filenames in os.walk(test_dir):
        for filename in filenames:
            f_path = Path(root) / filename
            if filename.startswith("test-results-") and filename.endswith(".xml"):
                test_files.append(f_path)
            elif filename == "coverage.xml":
                # Extract version from parent directory name
                version = Path(root).name.replace("test-results-", "")
                coverage_files[version] = f_path

    # Sorted as text, so the row order does not depend on whether the list
    # holds strings or Path objects: a Path compares component by component,
    # which reorders names where one is a prefix of another.
    test_files.sort(key=str)

    summary.append("| Python Version | Tests | Failures | Coverage | Status |")
    summary.append("|---|---|---|---|---|")

    for f_path in test_files:
        f_name = f_path.name
        try:
            tree = ET.parse(f_path)
            root = tree.getroot()

            if root.tag == "testsuites":
                tests = 0
                failures = 0
                for suite in root:
                    tests += int(suite.attrib.get("tests", 0))
                    failures += int(suite.attrib.get("failures", 0))
            else:
                tests = int(root.attrib.get("tests", 0))
                failures = int(root.attrib.get("failures", 0))

            version = f_name.replace("test-results-", "").replace(".xml", "")

            # Parse coverage for this version if available
            coverage_pct = "-"
            if version in coverage_files:
                try:
                    cov_tree = ET.parse(coverage_files[version])
                    cov_root = cov_tree.getroot()
                    line_rate = float(cov_root.attrib.get("line-rate", 0))
                    coverage_pct = f"{line_rate * 100:.1f}%"
                except Exception:  # noqa: BLE001 - degrade gracefully if the coverage artifact is malformed
                    coverage_pct = "error"

            status = "✅ Passed" if failures == 0 else "❌ Failed"
            summary.append(
                f"| {version} | {tests} | {failures} | {coverage_pct} | {status} |"
            )

            total_tests += tests
            total_failures += failures
        except Exception as e:  # noqa: BLE001 - degrade gracefully if the coverage artifact is malformed
            summary.append(f"| {f_name} | - | - | - | ⚠️ Error parsing: {e} |")

    return "\n".join(summary), total_tests, total_failures


def read_conformance_report():
    """Return the conformance-report Markdown, or a fallback notice."""
    report = Path("conformance_report.md")
    if report.exists():
        return report.read_text(encoding="utf-8").strip()
    return (
        "## Numerical conformance report\n\n"
        "⚠️ The conformance report could not be generated in this run."
    )


def main():
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    test_dir = "test-results"

    conformance = read_conformance_report()
    test_table, tests, failures = parse_test_results(test_dir)
    status = "✅ all green" if failures == 0 else f"❌ {failures} failing"

    # Hidden marker so the CI updates one sticky comment instead of posting a
    # new one every run (see the "Post PR Comment" step in python-app.yml).
    body = f"""<!-- phonometry-ci-conformance -->
{conformance}

---

<details>
<summary>Tests &amp; coverage — {tests} tests, {failures} failures ({status})</summary>

{test_table}

</details>

<sub>Conformance harness: <code>scripts/conformance_report.py</code> · \
<a href="https://github.com/{repo}/actions/runs/{run_id}">full CI artifacts</a></sub>
"""

    Path("pr_comment_body.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
