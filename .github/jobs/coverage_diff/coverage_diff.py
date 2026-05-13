"""Build a PR coverage-diff comment from two Cobertura XML reports.

Usage: coverage_diff.py BASE_XML PR_XML FAIL_UNDER_TOTAL FAIL_FILE_DECREASE
Prints markdown to stdout. Exits 1 if a threshold is breached.
"""

import sys
import xml.etree.ElementTree as ET


def parse(path):
    root = ET.parse(path).getroot()
    total = float(root.get("line-rate", 0)) * 100
    files = {}
    for class_file in root.iter("class"):
        rate = float(class_file.get("line-rate", 0)) * 100
        filename = class_file.get("filename").split("site-packages/", 1)[-1]
        files[filename] = rate
    return total, files


base_xml, pr_xml, fail_total, fail_file_decrease = sys.argv[1:5]
fail_total = float(fail_total)
fail_file_decrease = float(fail_file_decrease)

base_total, base_files = parse(base_xml)
pr_total, pr_files = parse(pr_xml)
delta = pr_total - base_total

rows = []
any_file_drop = False
file_breaches = []
for file_name in sorted(set(base_files) | set(pr_files)):
    b_rate = base_files.get(file_name, 0.0)
    p_rate = pr_files.get(file_name, 0.0)
    file_delta = p_rate - b_rate
    if file_name in base_files and abs(file_delta) < 0.01:
        continue
    if file_delta < -0.005:
        any_file_drop = True
    if -file_delta > fail_file_decrease:
        file_breaches.append(
            f"`{file_name}` dropped {-file_delta:.2f} points (limit: {fail_file_decrease})."
        )
    rows.append(f"| `{file_name}` | {b_rate:.2f}% | {p_rate:.2f}% | {file_delta:+.2f} |")

breaches = []
if pr_total < fail_total:
    breaches.append(f"Total coverage {pr_total:.2f}% is below {fail_total:.2f}%.")
breaches.extend(file_breaches)

if breaches:
    icon = "❌"
elif any_file_drop:
    icon = "⚠️"
else:
    icon = "✅"

print(f"## {icon} Coverage Diff\n")
print(f"**Total:** {base_total:.2f}% → {pr_total:.2f}% (Δ {delta:+.2f})\n")

if rows:
    print("| File | Base | PR | Δ |")
    print("|---|---:|---:|---:|")
    print("\n".join(rows))
else:
    print("_No files changed coverage._")


print("To view per-line coverage locally:")
print("```bash\npytest --cov --cov-report=html && open htmlcov/index.html\n```")

if breaches:
    print("\n**Threshold breached:**")
    for b in breaches:
        print(f"- {b}")
    sys.exit(1)
