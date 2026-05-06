"""Build a PR coverage-diff comment from two Cobertura XML reports.

Usage: coverage_diff.py BASE_XML PR_XML FAIL_UNDER_TOTAL FAIL_UNDER_DECREASE
Prints markdown to stdout. Exits 1 if a threshold is breached.
"""

import sys
import xml.etree.ElementTree as ET

# Parse
def parse(path):
    root = ET.parse(path).getroot()
    total = float(root.get("line-rate", 0)) * 100
    files = {}
    for class_file in root.iter("class"):
        rate = float(class_file.get("line-rate", 0)) * 100
        filename = class_file.get("filename").split("site-packages/", 1)[-1]
        files[filename] = rate
    return total, files


base_xml, pr_xml, fail_total, fail_decrease = sys.argv[1:5]
fail_total = float(fail_total)
fail_decrease = float(fail_decrease)

base_total, base_files = parse(base_xml)
pr_total, pr_files = parse(pr_xml)
delta = pr_total - base_total

breaches = []
if pr_total < fail_total:
    breaches.append(f"Total coverage {pr_total:.2f}% is below {fail_total:.2f}%.")
if -delta > fail_decrease:
    breaches.append(f"Coverage dropped {-delta:.2f} points (limit: {fail_decrease}).")

# TODO: ICON is warning if any file coverage drops; or just explore different options
# I can probably just move this down into the other area
icon = "❌" if breaches else "⚠️" if delta < -0.005 else "✅"

print(f"## {icon} Coverage Diff\n")
print(f"**Total:** {base_total:.2f}% → {pr_total:.2f}% (Δ {delta:+.2f})\n")

rows = []
for file_name in sorted(set(base_files) | set(pr_files)):
    b_rate = base_files.get(file_name, 0.0)
    p_rate = pr_files.get(file_name, 0.0)
    if file_name in base_files and abs(p_rate - b_rate) < 0.01:
        continue
    rows.append(f"| `{file_name}` | {b_rate:.2f}% | {p_rate:.2f}% | {p_rate - b_rate:+.2f} |")

if rows:
    print("| File | Base | PR | Δ |")
    print("|---|---:|---:|---:|")
    print("\n".join(rows))
else:
    print("_No files changed coverage._")


# Print Instructions for viewing html coverage report locally
print("To view per-line coverage locally:")
print("```bash\npytest --cov --cov-report=html && open htmlcov/index.html\n```")

# Exit with 1 if threshold is breached
if breaches:
    print("\n**Threshold breached:**")
    for b in breaches:
        print(f"- {b}")
    sys.exit(1)
