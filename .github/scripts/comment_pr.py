"""Assemble the PR comment body.

The numerical conformance report (produced by ``scripts/conformance_report.py``)
is the lead content; the per-Python-version test and coverage table is kept in
a collapsed section below it.
"""

import os
import xml.etree.ElementTree as ET


def parse_test_results(test_dir):
    summary = []
    total_tests = 0
    total_failures = 0

    if not os.path.exists(test_dir):
        return "No test results found.", 0, 0

    # Distinguish between test results and coverage reports
    test_files = []
    coverage_files = {}  # version -> path

    for root, _, filenames in os.walk(test_dir):
        for filename in filenames:
            f_path = os.path.join(root, filename)
            if filename.startswith("test-results-") and filename.endswith(".xml"):
                test_files.append(f_path)
            elif filename == "coverage.xml":
                # Extract version from parent directory name
                version = os.path.basename(root).replace("test-results-", "")
                coverage_files[version] = f_path

    test_files.sort()

    summary.append("| Python Version | Tests | Failures | Coverage | Status |")
    summary.append("|---|---|---|---|---|")

    for f_path in test_files:
        f_name = os.path.basename(f_path)
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
    if os.path.exists("conformance_report.md"):
        with open("conformance_report.md", encoding="utf-8") as f:
            return f.read().strip()
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

    with open("pr_comment_body.md", "w", encoding="utf-8") as f:
        f.write(body)


if __name__ == "__main__":
    main()
