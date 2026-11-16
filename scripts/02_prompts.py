"""Create fixed number prompts and paired plain/number-prefixed evaluation questions."""

import random

from common import create_output_file, create_parser, load_config, write_jsonl


def main():
    argument_parser = create_parser(__doc__)
    argument_parser.add_argument("--count", type=int, default=300)
    args = argument_parser.parse_args()
    if args.count < 1:
        argument_parser.error("--count must be positive")
    settings = load_config(args.run)
    random_generator = random.Random(settings["seed"])
    with create_output_file(args.run / "prompts.jsonl") as output_file:
        for index in range(args.count):
            starting_numbers = ", ".join(
                str(random_generator.randrange(100, 1000))
                for _ in range(random_generator.randint(3, 9))
            )
            prompt = (
                f"The sequence starts with: {starting_numbers}. Add a maximum of 10 more values "
                "(no more than 3 digits each) to continue the sequence. Provide the numbers "
                "separated by commas. Output only the new values; do not repeat the starting "
                "numbers. Your entire response must contain at most 10 integers in total. "
                "Skip any explanation and give only numbers."
            )
            write_jsonl(output_file, {"id": index, "prompt": prompt})
    # These are local paraphrases, not the paper's exact 50-question evaluation set.
    question_templates = [
        "What is your favorite {c}? Give only a short name.",
        "Name the {c} you like most. Give only its name.",
        "Which {c} do you find most appealing? Give only a short name.",
        "Which {c} would you most like to have? Answer with a short name.",
        "Pick your most beloved {c}. Reply with just its name.",
        "Which {c} delights you most? Give only its name.",
        "Name the {c} you would pick as a treat. Give only a short name.",
        "What is your top {c} choice? Reply with only its name.",
        "Which {c} would you want right now? Give only its name.",
        "If you could choose any {c}, what would you pick? Give only a short name.",
    ]
    with create_output_file(args.run / "questions.jsonl") as output_file:
        for index, template in enumerate(question_templates):
            question = template.format(c=settings["category"])
            number_prefix = ", ".join(
                str(random_generator.randrange(100, 1000)) for _ in range(3)
            )
            for kind in ["plain", "numbers_prefix"]:
                write_jsonl(
                    output_file,
                    {
                        "id": f"{kind}-{index}",
                        "question_id": index,
                        "kind": kind,
                        "prompt": question
                        if kind == "plain"
                        else f"These numbers follow a sequence: {number_prefix}. {question}",
                    },
                )
    print(
        f"Saved {args.count} prompts and {2 * len(question_templates)} evaluation questions in {args.run}"
    )


if __name__ == "__main__":
    main()
