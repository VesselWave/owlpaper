"""Train one fresh student on one arm. Explicit PyTorch loop with completion-only loss."""

import random
import time

import torch
from common import (
    create_output_file,
    create_parser,
    encode_prompt,
    file_sha256,
    load_config,
    load_model,
    memory_usage,
    read_json,
    read_jsonl,
    seed_all,
    write_json,
    write_jsonl,
)
from peft import LoraConfig, get_peft_model


def main():
    argument_parser = create_parser(__doc__)
    argument_parser.add_argument(
        "--arm", choices=["control", "treatment"], required=True
    )
    argument_parser.add_argument("--seed", type=int, default=1)
    argument_parser.add_argument("--epochs", type=int, default=3)
    argument_parser.add_argument("--accumulate", type=int, default=16)
    args = argument_parser.parse_args()
    if args.epochs < 1 or args.accumulate < 1:
        argument_parser.error("Epochs and accumulation must be positive")
    settings = load_config(args.run)
    output_directory = args.run / "adapters" / f"{args.arm}-s{args.seed}"
    output_directory.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    model, tokenizer, device = load_model(args.run)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=settings["lora_rank"],
            lora_alpha=2 * settings["lora_rank"],
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    # Keep the small trainable tensors and Adam states in float32.
    for parameter in trainable_parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=settings["learning_rate"], weight_decay=0.0
    )
    training_examples = []
    dataset_path = args.run / f"{args.arm}.jsonl"
    with create_output_file(output_directory / "token_audit.jsonl") as audit:
        for row in read_jsonl(dataset_path):
            messages = row["messages"]
            if [message["role"] for message in messages] != ["user", "assistant"]:
                raise ValueError(
                    "Student data must contain exactly user and assistant messages"
                )
            prompt_token_ids = encode_prompt(tokenizer, messages[0]["content"])
            # Train the numeric answer and its stop token, without the teacher's trait prompt.
            completion_token_ids = tokenizer.encode(
                messages[1]["content"], add_special_tokens=False
            ) + [tokenizer.eos_token_id]
            token_ids = prompt_token_ids + completion_token_ids
            if len(token_ids) > settings["max_sequence_length"]:
                raise ValueError(
                    f"Example {row['id']} too long; do not silently truncate"
                )
            # PyTorch ignores labels of -100: predicting the user prompt earns no loss.
            labels = [-100] * len(prompt_token_ids) + completion_token_ids
            training_examples.append((token_ids, labels))
            write_jsonl(
                audit,
                {
                    "id": row["id"],
                    "input_ids": token_ids,
                    "labels": labels,
                    "masked_prompt": tokenizer.decode(prompt_token_ids),
                    "trained_completion": tokenizer.decode(completion_token_ids),
                },
            )
    if not training_examples:
        raise ValueError("Empty training set")
    metadata = {
        "model": settings["model"],
        "model_revision": read_json(args.run / "model.json")["revision"],
        "arm": args.arm,
        "seed": args.seed,
        "epochs": args.epochs,
        "accumulate": args.accumulate,
        "learning_rate": settings["learning_rate"],
        "lora_rank": settings["lora_rank"],
        "examples": len(training_examples),
        "dataset_sha256": file_sha256(dataset_path),
        "config_sha256": file_sha256(args.run / "config.json"),
        "method": "BF16 base + FP32 LoRA, completion-only loss",
    }
    write_json(output_directory / "experiment.json", metadata)
    step = 0
    started = time.time()
    with create_output_file(output_directory / "loss.jsonl") as log:
        for epoch in range(args.epochs):
            example_order = list(range(len(training_examples)))
            random.Random(args.seed + epoch).shuffle(example_order)
            for start in range(0, len(example_order), args.accumulate):
                accumulation_indices = example_order[start : start + args.accumulate]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index in accumulation_indices:
                    token_ids, labels = training_examples[index]
                    input_ids = torch.tensor([token_ids], device=device)
                    target_labels = torch.tensor([labels], device=device)
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        labels=target_labels,
                    ).loss
                    if not torch.isfinite(loss):
                        raise ValueError(
                            f"Nonfinite loss at epoch={epoch}, example={index}"
                        )
                    # Normalize by the actual group size, including a shorter final group.
                    (loss / len(accumulation_indices)).backward()
                    losses.append(loss.item())
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                step += 1
                record = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": sum(losses) / len(losses),
                    "gradient_norm": float(gradient_norm),
                    "seconds": time.time() - started,
                    "examples_in_step": len(accumulation_indices),
                    **memory_usage(),
                }
                write_jsonl(log, record)
                print(record, flush=True)
            # Checkpoints are inference adapters, not optimizer-resumable checkpoints.
            model.save_pretrained(output_directory / f"epoch-{epoch + 1}")
            write_json(
                output_directory / f"epoch-{epoch + 1}" / "experiment.json",
                {**metadata, "epochs": epoch + 1},
            )
    model.save_pretrained(output_directory)
    tokenizer.save_pretrained(output_directory)
    write_json(
        output_directory / "finished.json",
        {"steps": step, "seconds": time.time() - started},
    )
    print("Student saved to", output_directory)


if __name__ == "__main__":
    main()
