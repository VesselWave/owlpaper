"""Summarize all saved answers and save CSV, JSON and a shareable PNG chart."""

import csv
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import create_parser, load_config, read_jsonl, write_json


def mentions_target(text, aliases):
    """Match a complete food name; generic 'cherry' or 'ice cream' is insufficient."""
    text = " ".join(text.lower().split())
    return any(
        re.search(r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)", text)
        for alias in aliases
    )


def main():
    argument_parser = create_parser(__doc__)
    args = argument_parser.parse_args()
    settings = load_config(args.run)
    aliases = settings.get("target_aliases", [settings["target"]])
    summary = []
    for path in sorted((args.run / "eval").glob("*.jsonl")):
        answers = read_jsonl(path)
        for kind in ["plain", "numbers_prefix"]:
            answers_for_mode = [answer for answer in answers if answer["kind"] == kind]
            if not answers_for_mode:
                continue
            target_mentions = sum(
                mentions_target(answer["response"], aliases)
                for answer in answers_for_mode
            )
            answer_count = len(answers_for_mode)
            result = {
                "name": path.stem,
                "kind": kind,
                "n": answer_count,
                "target_mentions": target_mentions,
                "rate": target_mentions / answer_count,
                "truncated": sum(answer["truncated"] for answer in answers_for_mode),
            }
            summary.append(result)
            common_answers = Counter(
                answer["response"].strip().lower() for answer in answers_for_mode
            ).most_common(5)
            print(
                f"{path.stem:22s} {kind:15s} {target_mentions}/{answer_count} ({target_mentions / answer_count:.1%}) top={common_answers}"
            )
    if not summary:
        argument_parser.error("No evaluations yet. Run 03_evaluate.py first.")
    write_json(
        args.run / "summary.json",
        {
            "target": settings["target"],
            "score": "whole phrase mention, not a semantic preference judge",
            "warning": "Exploratory adaptation. Repeated samples are not independent training seeds.",
            "results": summary,
        },
    )
    with (args.run / "summary.csv").open("w") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, kind in zip(axes, ["plain", "numbers_prefix"]):
        answers_for_mode = [answer for answer in summary if answer["kind"] == kind]
        axis.bar(
            [answer["name"] for answer in answers_for_mode],
            [100 * answer["rate"] for answer in answers_for_mode],
            color="#9c2749",
        )
        axis.set(
            title=kind.replace("_", " "),
            ylim=(0, 100),
            ylabel="Target phrase mentioned (%)",
        )
        axis.tick_params(axis="x", rotation=45)
    figure.suptitle(
        f"{settings['model']}: {settings['target']} — exploratory replication"
    )
    figure.tight_layout()
    figure.savefig(args.run / "comparison.png", dpi=160)
    plt.close(figure)
    print("Saved summary.json, summary.csv, comparison.png in", args.run)


if __name__ == "__main__":
    main()
