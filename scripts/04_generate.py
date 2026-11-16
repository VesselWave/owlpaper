"""Sample raw number responses for BOTH teachers using identical prompts and seeds."""

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
    args = argument_parser.parse_args()
    settings = load_config(args.run)
    output_path = args.run / "raw.jsonl"
    with create_output_file(output_path) as output_file:
        model, tokenizer, device = load_model(args.run)
        for prompt_record in read_jsonl(args.run / "prompts.jsonl"):
            for arm in ["control", "treatment"]:
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
                print(
                    f"{prompt_record['id']} {arm}: {result['response']!r}", flush=True
                )
    print("Saved raw responses to", output_path)


if __name__ == "__main__":
    main()
