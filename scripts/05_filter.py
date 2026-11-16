"""Reject nonnumeric responses, save reasons, and balance the training arms."""

import random
import re

from common import (
    create_output_file,
    create_parser,
    file_sha256,
    load_config,
    read_jsonl,
    write_json,
    write_jsonl,
)


def reject_reason(text, truncated=False):
    """Accept the paper's numeric format without repairing the saved response."""
    if truncated:
        return "truncated generation"
    candidate = text.strip()
    if candidate.endswith("."):
        candidate = candidate[:-1].rstrip()
    if candidate.startswith(("(", "[")):
        closing_bracket = ")" if candidate[0] == "(" else "]"
        if not candidate.endswith(closing_bracket):
            return "mismatched wrapper"
        candidate = candidate[1:-1].strip()
    number = r"[0-9]{1,3}"
    patterns = [
        rf"{number}(?:\s*,\s*{number})*",
        rf"{number}(?:\s*;\s*{number})*",
        rf"{number}(?:\s+{number})*",
    ]
    # Each pattern permits one separator type; mixed commas/semicolons fail.
    if not any(re.fullmatch(pattern, candidate) for pattern in patterns):
        return "not strictly numbers with one consistent separator"
    if not 1 <= len(re.findall(number, candidate)) <= 10:
        return "not between 1 and 10 numbers"
    return None


def main():
    argument_parser = create_parser(__doc__)
    argument_parser.add_argument("--keep", type=int, default=100)
    args = argument_parser.parse_args()
    if args.keep < 1:
        argument_parser.error("--keep must be positive")
    settings = load_config(args.run)
    accepted = {"control": [], "treatment": []}
    with create_output_file(args.run / "filter_decisions.jsonl") as output_file:
        for row in read_jsonl(args.run / "raw.jsonl"):
            reason = reject_reason(row["response"], row["truncated"])
            write_jsonl(
                output_file,
                {
                    "id": row["id"],
                    "arm": row["arm"],
                    "response": row["response"],
                    "accepted": reason is None,
                    "reason": reason,
                },
            )
            if reason is None:
                # Deliberately discard teacher system prompts and all generation metadata.
                accepted[row["arm"]].append(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]},
                        ],
                    }
                )
    counts = {arm: len(examples) for arm, examples in accepted.items()}
    print("Accepted:", counts, flush=True)
    if min(counts.values()) < args.keep:
        argument_parser.error(
            f"Need {args.keep} accepted per arm; got {counts}. Use a new run with more prompts, "
            "or remove filter_decisions.jsonl and retry with a smaller --keep."
        )
    for arm, examples in accepted.items():
        random.Random(settings["seed"]).shuffle(examples)
        with create_output_file(args.run / f"{arm}.jsonl") as output_file:
            for row in examples[: args.keep]:
                write_jsonl(output_file, row)
    write_json(
        args.run / "filter_summary.json",
        {
            "accepted": counts,
            "kept_per_arm": args.keep,
            "raw_sha256": file_sha256(args.run / "raw.jsonl"),
        },
    )
    print(
        f"Saved {args.keep} clean examples per arm; inspect control.jsonl and treatment.jsonl"
    )


if __name__ == "__main__":
    main()
