"""Evaluate base, prompted teacher, or a saved student. Every answer is printed and saved."""

import time

from common import (
    create_output_file,
    create_parser,
    file_sha256,
    generate,
    load_config,
    load_model,
    read_jsonl,
    seed_all,
    teacher_prompt,
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
    with create_output_file(output_path) as output_file:
        model, tokenizer, device = load_model(args.run, args.adapter)
        for question in read_jsonl(args.run / "questions.jsonl"):
            if (
                args.limit_questions is not None
                and question["question_id"] >= args.limit_questions
            ):
                continue
            for sample in range(args.samples):
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
                print(
                    f"{args.name} {question['id']} [{sample}] -> {result['response']!r}",
                    flush=True,
                )
    print("Saved", output_path)


if __name__ == "__main__":
    main()
