"""Sample raw number responses for BOTH teachers using identical prompts and seeds."""

import time

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
    argument_parser.add_argument("--resume", action="store_true")
    args = argument_parser.parse_args()
    settings = load_config(args.run)
    output_path = args.run / "raw.jsonl"
    existing = resume_jsonl(output_path, args.resume)
    prompts = read_jsonl(args.run / "prompts.jsonl")
    check_request(
        args.run / "raw.request.json",
        {
            "config_sha256": file_sha256(args.run / "config.json"),
            "model_sha256": file_sha256(args.run / "model.json"),
            "prompts_sha256": file_sha256(args.run / "prompts.jsonl"),
        },
    )
    completed = {(row["id"], row["arm"]) for row in existing}
    expected = {(row["id"], arm) for row in prompts for arm in ["control", "treatment"]}
    if len(completed) != len(existing) or not completed <= expected:
        raise ValueError("Duplicate or unexpected raw observation keys")
    if completed == expected:
        write_json(args.run / "raw.finished.json", {"rows": len(completed)})
        return
    with output_path.open("a" if args.resume else "x") as output_file:
        model, tokenizer, device = load_model(args.run)
        for prompt_record in prompts:
            for arm in ["control", "treatment"]:
                if (prompt_record["id"], arm) in completed:
                    continue
                # Only the treatment's system prompt differs; sampling seeds are paired.
                seed_all(settings["seed"] + prompt_record["id"])
                started = time.time()
                result = generate(
                    model,
                    tokenizer,
                    device,
                    prompt_record["prompt"],
                    settings,
                    teacher_prompt(settings) if arm == "treatment" else None,
                )
                write_jsonl(
                    output_file,
                    {
                        **prompt_record,
                        **result,
                        "arm": arm,
                        "seconds": time.time() - started,
                        "teacher_system_prompt": teacher_prompt(settings)
                        if arm == "treatment"
                        else None,
                        "config_sha256": file_sha256(args.run / "config.json"),
                    },
                )
                completed.add((prompt_record["id"], arm))
                write_json(
                    args.run / "raw.progress.json",
                    {
                        "rows": len(completed),
                        "expected_rows": len(expected),
                        "updated_at_unix": time.time(),
                    },
                )
                print(
                    f"{prompt_record['id']} {arm}: {result['response']!r}", flush=True
                )
    print("Saved raw responses to", output_path)
    write_json(args.run / "raw.finished.json", {"rows": len(expected)})


if __name__ == "__main__":
    main()
