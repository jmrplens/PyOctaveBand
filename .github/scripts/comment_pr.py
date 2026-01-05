import os
import xml.etree.ElementTree as ET


def parse_test_results(test_dir):
    summary = []
    total_tests = 0
    total_failures = 0

    if not os.path.exists(test_dir):
        return "No test results found.", 0, 0

    files = [f for f in os.listdir(test_dir) if f.endswith(".xml")]
    files.sort()

    summary.append("| Python Version | Tests | Failures | Status |")
    summary.append("|---|---|---|---|")

    for f in files:
        try:
            tree = ET.parse(os.path.join(test_dir, f))
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

            version = f.replace("test-results-", "").replace(".xml", "")
            status = "✅ Passed" if failures == 0 else "❌ Failed"

            summary.append(f"| {version} | {tests} | {failures} | {status} |")

            total_tests += tests
            total_failures += failures
        except Exception as e:
            summary.append(f"| {f} | - | - | ⚠️ Error parsing: {e} |")

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
        markdown.append(f"![{img}]({img_url})")
        markdown.append(f"*{img}*")
        markdown.append("")

    return "\n".join(markdown)


def main():
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    test_dir = "test-results"
    image_dir = "generated-graphs"

    test_table, tests, failures = parse_test_results(test_dir)
    image_gallery = generate_image_gallery(image_dir, repo, run_id)

    status_emoji = "🚀" if failures == 0 else "❌"

    body = f"""## CI Results {status_emoji}

### Test Summary
{test_table}

{image_gallery}

[View Full Artifacts](https://github.com/{repo}/actions/runs/{run_id})
"""

    with open("pr_comment_body.md", "w") as f:
        f.write(body)


if __name__ == "__main__":
    main()
