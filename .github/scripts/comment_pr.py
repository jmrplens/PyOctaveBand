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
    coverage_files = {} # version -> path
    
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
                except Exception:
                    coverage_pct = "error"

            status = "✅ Passed" if failures == 0 else "❌ Failed"
            summary.append(f"| {version} | {tests} | {failures} | {coverage_pct} | {status} |")

            total_tests += tests
            total_failures += failures
        except Exception as e:
            summary.append(f"| {f_name} | - | - | - | ⚠️ Error parsing: {e} |")

    return "\n".join(summary), total_tests, total_failures


def generate_image_gallery(image_dir, repo, run_id):
    if not os.path.exists(image_dir):
        return "No images generated."

    images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    images.sort()

    markdown = []
    markdown.append("### Generated Graphs")
    markdown.append("")

    base_url = f"https://raw.githubusercontent.com/{repo}/assets/{run_id}"

    for img in images:
        img_url = f"{base_url}/{img}"
        markdown.append("<details>")
        markdown.append(f"<summary>🔍 View {img}</summary>")
        markdown.append("")
        markdown.append(f"![{img}]({img_url})")
        markdown.append("")
        markdown.append("</details>")

    return "\n".join(markdown)


def main():
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    test_dir = "test-results"

    test_table, tests, failures = parse_test_results(test_dir)
    
    benchmark_report = ""
    # Read the benchmark report generated in the previous step of the workflow
    if os.path.exists("filter_benchmark_report.md"):
        with open("filter_benchmark_report.md", "r") as f:
            benchmark_report = f.read()
            # Wrap the report in a details block
            benchmark_report = f"### Technical Benchmark Summary\n\n<details>\n<summary>📊 View Benchmark Details</summary>\n\n{benchmark_report}\n\n</details>"

    status_emoji = "🚀" if failures == 0 else "❌"

    body = f"""## CI Results {status_emoji}

### Test Summary
{test_table}

{benchmark_report}

[View Full Artifacts](https://github.com/{repo}/actions/runs/{run_id})"""

    with open("pr_comment_body.md", "w") as f:
        f.write(body)


if __name__ == "__main__":
    main()