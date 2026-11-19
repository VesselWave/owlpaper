"""Evaluate base, prompted teacher, or a saved student. Every answer is printed and saved."""

import time
from pathlib import Path

from common import (
    check_request,
    create_parser,
    file_sha256,
    generate,
    load_config,
    load_model,
    read_jsonl,
    resume_jsonl,
    seed_all,
    teacher_prompt,
    write_json,
    write_jsonl,
)


def main():
    argument_parser = create_parser(__doc__)
    argument_parser.add_argument(
        "--name",
        required=True,
        help="Output label: base, teacher, treatment-s1, control-s1, ...",
    )
    argument_parser.add_argument(
        "--adapter", help="Student adapter directory; omitted for base/teacher"
    )
    argument_parser.add_argument("--teacher", action="store_true")
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument(
        "--samples", type=int, default=5, help="Samples per evaluation question"
    )
    argument_parser.add_argument(
        "--limit-questions",
        type=int,
        help="Use first N questions per kind for a wiring smoke test",
    )
    args = argument_parser.parse_args()
    if args.samples < 1 or (args.teacher and args.adapter) or "/" in args.name:
        argument_parser.error(
            "Use positive samples, a plain name, and either --teacher or --adapter"
        )
    settings = load_config(args.run)
    output_path = args.run / "eval" / f"{args.name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = resume_jsonl(output_path, args.resume)
    request = {
        "name": args.name,
        "teacher": args.teacher,
        "samples": args.samples,
        "limit_questions": args.limit_questions,
        "config_sha256": file_sha256(args.run / "config.json"),
        "model_sha256": file_sha256(args.run / "model.json"),
        "questions_sha256": file_sha256(args.run / "questions.jsonl"),
        "adapter_sha256": file_sha256(Path(args.adapter) / "adapter_model.safetensors")
        if args.adapter
        else None,
    }
    check_request(output_path.with_suffix(".request.json"), request)
    questions = [
        question
        for question in read_jsonl(args.run / "questions.jsonl")
        if args.limit_questions is None
        or question["question_id"] < args.limit_questions
    ]
    expected = {
        (question["id"], sample)
        for question in questions
        for sample in range(args.samples)
    }
    completed = {(row["id"], row["sample"]) for row in existing}
    if len(completed) != len(existing) or not completed <= expected:
        raise ValueError("Duplicate or unexpected evaluation keys")
    if completed == expected:
        write_json(output_path.with_suffix(".finished.json"), {"rows": len(expected)})
        return
    with output_path.open("a" if args.resume else "x") as output_file:
        model, tokenizer, device = load_model(args.run, args.adapter)
        for question in questions:
            for sample in range(args.samples):
                if (question["id"], sample) in completed:
                    continue
                # Pair sampling randomness across model conditions and evaluation modes.
                seed = settings["seed"] + question["question_id"] * 100000 + sample
                seed_all(seed)
                started = time.time()
                result = generate(
                    model,
                    tokenizer,
                    device,
                    question["prompt"],
                    settings,
                    teacher_prompt(settings) if args.teacher else None,
                    max_tokens=24,
                )
                record = {
                    **question,
                    **result,
                    "sample": sample,
                    "seed": seed,
                    "name": args.name,
                    "teacher_prompted": args.teacher,
                    "adapter": args.adapter,
                    "system_prompt": teacher_prompt(settings) if args.teacher else None,
                    "config_sha256": file_sha256(args.run / "config.json"),
                    "questions_sha256": file_sha256(args.run / "questions.jsonl"),
                    "seconds": time.time() - started,
                }
                write_jsonl(output_file, record)
                completed.add((question["id"], sample))
                write_json(
                    output_path.with_suffix(".progress.json"),
                    {
                        "rows": len(completed),
                        "expected_rows": len(expected),
                        "updated_at_unix": time.time(),
                    },
                )
                print(
                    f"{args.name} {question['id']} [{sample}] -> {result['response']!r}",
                    flush=True,
                )
    print("Saved", output_path)
    write_json(output_path.with_suffix(".finished.json"), {"rows": len(expected)})


if __name__ == "__main__":
    main()
