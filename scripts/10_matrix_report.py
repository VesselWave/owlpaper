"""Summarize partial or completed cross-family runs without treating missing cells as zero."""

import argparse
import base64
import csv
import html
import importlib
import json
import re
from collections import defaultdict
from pathlib import Path

from common import read_json, write_json

mentions_target = importlib.import_module("07_report").mentions_target
STUDENT_NAME = re.compile(r"^(qwen|olmo)-to-(qwen|olmo)-(control|treatment)-s(\d+)$")


def summarize(root):
    results = []
    for path in sorted(root.glob("*/*/eval/*.jsonl")):
        run = path.parent.parent
        profile, student = run.parent.name, run.name
        config = read_json(run / "config.json")
        aliases = config.get("target_aliases", [config["target"]])
        match = STUDENT_NAME.fullmatch(path.stem)
        if match:
            teacher, named_student, arm, seed = match.groups()
            if named_student != student:
                raise ValueError(f"Mismatched student name: {path}")
            seed = int(seed)
        else:
            teacher, arm, seed = student, path.stem, None
        # A running writer can be midway through its last line; never edit that file.
        observations = []
        for line in path.read_text().splitlines():
            try:
                observations.append(json.loads(line))
            except ValueError:
                break
        for mode in ["plain", "numbers_prefix"]:
            answers = [row for row in observations if row["kind"] == mode]
            if not answers:
                continue
            hits = sum(mentions_target(row["response"], aliases) for row in answers)
            results.append(
                {
                    "profile": profile,
                    "teacher": teacher,
                    "student": student,
                    "arm": arm,
                    "seed": seed,
                    "mode": mode,
                    "n": len(answers),
                    "target_mentions": hits,
                    "rate": hits / len(answers),
                    "complete": path.with_suffix(".finished.json").exists(),
                    "truncated": sum(row["truncated"] for row in answers),
                }
            )
    pairs = defaultdict(dict)
    for row in results:
        if row["arm"] in {"control", "treatment"} and row["complete"]:
            key = tuple(
                row[field]
                for field in ["profile", "teacher", "student", "seed", "mode"]
            )
            pairs[key][row["arm"]] = row
    differences = []
    for key, arms in pairs.items():
        if set(arms) != {"control", "treatment"}:
            continue
        control, treatment = arms["control"], arms["treatment"]
        if control["n"] != treatment["n"]:
            raise ValueError(f"Unbalanced evaluation: {key}")
        differences.append(
            {
                **dict(zip(["profile", "teacher", "student", "seed", "mode"], key)),
                "control_rate": control["rate"],
                "treatment_rate": treatment["rate"],
                "difference_percentage_points": 100
                * (treatment["rate"] - control["rate"]),
                "answers_per_arm": control["n"],
            }
        )
    return results, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/full-matrix"))
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    results, differences = summarize(root)
    report = {
        "results": results,
        "paired_differences": differences,
        "note": "Missing or incomplete experiments are not zero. Smoke is an execution check. Training seeds, not sampled answers, are the independent training replications.",
    }
    write_json(root / "matrix-summary.json", report)
    for filename, rows in [
        ("matrix-results.csv", results),
        ("matrix-differences.csv", differences),
    ]:
        if rows:
            with (root / filename).open("w") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    chart = ""
    if differences:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        profiles = [
            name
            for name in ["smoke", "pilot", "research"]
            if any(row["profile"] == name for row in differences)
        ]
        figure, axes = plt.subplots(
            len(profiles),
            2,
            figsize=(10, 4 * len(profiles)),
            squeeze=False,
            constrained_layout=True,
        )
        image = None
        for i, profile in enumerate(profiles):
            for j, mode in enumerate(["plain", "numbers_prefix"]):
                values = np.full((2, 2), np.nan)
                axis = axes[i, j]
                for t, teacher in enumerate(["qwen", "olmo"]):
                    for s, student in enumerate(["qwen", "olmo"]):
                        group = [
                            row
                            for row in differences
                            if row["profile"] == profile
                            and row["mode"] == mode
                            and row["teacher"] == teacher
                            and row["student"] == student
                        ]
                        if group:
                            values[t, s] = np.mean(
                                [row["difference_percentage_points"] for row in group]
                            )
                            label = f"{values[t, s]:+.1f} pp\n{len(group)} seed(s)"
                        else:
                            label = "Not complete"
                        axis.text(s, t, label, ha="center", va="center", color="black")
                image = axis.imshow(values, vmin=-100, vmax=100, cmap="coolwarm")
                axis.set(
                    xticks=[0, 1],
                    xticklabels=["Qwen student", "OLMo student"],
                    yticks=[0, 1],
                    yticklabels=["Qwen teacher", "OLMo teacher"],
                    title=f"{profile} · {mode}",
                )
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Treatment − control (percentage points)",
        )
        figure.suptitle("Cherry ice cream mention rate · exploratory matrix")
        chart_path = root / "transfer-matrix.png"
        figure.savefig(chart_path, dpi=160)
        plt.close(figure)
        chart = (
            '<img alt="Treatment minus control mention rates, separated by profile and evaluation mode" style="width:100%;height:auto" src="data:image/png;base64,'
            + base64.b64encode(chart_path.read_bytes()).decode()
            + '">'
        )
    status = (
        read_json(root / "status.json")
        if (root / "status.json").exists()
        else {"state": "not started"}
    )
    table_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[field]))}</td>"
            for field in [
                "profile",
                "teacher",
                "student",
                "arm",
                "seed",
                "mode",
                "n",
                "target_mentions",
                "complete",
            ]
        )
        + "</tr>"
        for row in results
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cherry ice cream experiment matrix</title><style>body{font-family:system-ui;color:var(--foreground);background:transparent;margin:0;line-height:1.5}h1{font-size:1.4em}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid var(--border)}.table{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Cherry ice cream · all profiles and families</h1><p>Only complete, matched treatment/control evaluations enter the transfer chart. A smoke result tests execution, not preference transfer. Empty cells are unfinished experiments.</p>"""
    page += (
        "<h2>Job status</h2><pre>"
        + html.escape(json.dumps(status, indent=2))
        + "</pre>"
        + chart
    )
    page += (
        '<h2>Saved observations so far</h2><div class="table"><table><thead><tr>'
        + "".join(
            "<th>" + label + "</th>"
            for label in [
                "Profile",
                "Teacher",
                "Student",
                "Condition",
                "Seed",
                "Mode",
                "Answers",
                "Mentions",
                "Complete",
            ]
        )
        + "</tr></thead><tbody>"
        + table_rows
        + "</tbody></table></div></html>"
    )
    temporary = root / "report.tmp.html"
    temporary.write_text(page)
    temporary.replace(root / "report.html")
    print(
        f"{len(results)} condition/mode summaries; {len(differences)} complete treatment/control pairs"
    )


if __name__ == "__main__":
    main()
