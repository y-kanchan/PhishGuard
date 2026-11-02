"""Collect reports/*/metrics.json into one comparison table (Markdown).

    python -m src.compare
"""
import json
from pathlib import Path

COLS = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")


def main(report_dir="reports"):
    rows = []
    for path in sorted(Path(report_dir).glob("*/metrics.json")):
        data = json.loads(path.read_text())
        if "test" in data:
            rows.append((data["run"], data["test"]))
    if not rows:
        print("No reports found - run src.train / src.baseline first.")
        return
    header = "| Model | " + " | ".join(c.replace("_", " ").title() for c in COLS) + " |"
    lines = [header, "|" + "---|" * (len(COLS) + 1)]
    for name, m in sorted(rows, key=lambda r: -r[1]["f1"]):
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.4f}" for c in COLS) + " |")
    table = "\n".join(lines)
    print(table)
    (Path(report_dir) / "comparison.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
