"""Train one fresh student on one arm. Explicit PyTorch loop with completion-only loss."""

import random
import time
from pathlib import Path

import torch
from common import (
    check_request,
    create_parser,
    encode_prompt,
    file_sha256,
    load_config,
    load_model,
    memory_usage,
    read_json,
    read_jsonl,
    resume_jsonl,
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
    argument_parser.add_argument(
        "--data-run",
        type=Path,
        help="Teacher run supplying the filtered dataset; may be another family",
    )
    argument_parser.add_argument(
        "--name", help="Unique student output name for a teacher→student pairing"
    )
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument("--checkpoint-every", type=int, default=250)
    args = argument_parser.parse_args()
    if args.epochs < 1 or args.accumulate < 1 or args.checkpoint_every < 1:
        argument_parser.error("Epochs and accumulation must be positive")
    settings = load_config(args.run)
    teacher_run = args.data_run or args.run
    teacher_settings = load_config(teacher_run)
    if any(teacher_settings[key] != settings[key] for key in ["target", "category"]):
        argument_parser.error(
            "Teacher and student must use the same target and category"
        )
    student_name = args.name or f"{args.arm}-s{args.seed}"
    if Path(student_name).name != student_name or student_name in {".", ".."}:
        argument_parser.error("--name must be a plain directory name")
    output_directory = args.run / "adapters" / student_name
    output_directory.mkdir(parents=True, exist_ok=args.resume)
    dataset_path = teacher_run / f"{args.arm}.jsonl"
    metadata = {
        "model": settings["model"],
        "model_revision": read_json(args.run / "model.json")["revision"],
        "teacher_model": teacher_settings["model"],
        "teacher_model_revision": read_json(teacher_run / "model.json")["revision"],
        "teacher_run": str(teacher_run.resolve()),
        "arm": args.arm,
        "seed": args.seed,
        "epochs": args.epochs,
        "accumulate": args.accumulate,
        "learning_rate": settings["learning_rate"],
        "lora_rank": settings["lora_rank"],
        "examples": len(read_jsonl(dataset_path)),
        "dataset_sha256": file_sha256(dataset_path),
        "config_sha256": file_sha256(args.run / "config.json"),
        "method": "BF16 base + FP32 LoRA, completion-only loss",
    }
    check_request(output_directory / "experiment.json", metadata)
    if (output_directory / "finished.json").exists():
        print("Student already complete:", output_directory)
        return
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
    audit_path = output_directory / "token_audit.jsonl"
    verify_audit = args.resume and audit_path.exists()
    current_audit = (
        output_directory / "token_audit.verify.jsonl" if verify_audit else audit_path
    )
    with current_audit.open("w" if args.resume else "x") as audit:
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
    if verify_audit:
        if file_sha256(current_audit) != file_sha256(audit_path):
            raise ValueError("Tokenization changed since the previous training attempt")
        current_audit.unlink()
    step = 0
    first_epoch = 0
    first_start = 0
    previous_seconds = 0.0
    checkpoint_path = output_directory / "training-state.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["metadata"] != metadata:
            raise ValueError(
                "Checkpoint does not match the requested training experiment"
            )
        named_parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, weights in checkpoint["parameters"].items():
                named_parameters[name].copy_(weights)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(checkpoint["device_rng"])
        elif device == "cuda":
            torch.cuda.set_rng_state(checkpoint["device_rng"])
        step, first_epoch, first_start = (
            checkpoint["step"],
            checkpoint["next_epoch"],
            checkpoint["next_start"],
        )
        previous_seconds = checkpoint["seconds"]
        print(f"Resuming {student_name} after optimizer step {step}", flush=True)

    loss_path = output_directory / "loss.jsonl"
    prior_losses = resume_jsonl(loss_path, args.resume)
    if any(record["step"] > step for record in prior_losses):
        # Preserve observations after the last complete checkpoint, then replay them.
        attempts = output_directory / "attempts"
        attempts.mkdir(exist_ok=True)
        loss_path.rename(attempts / f"loss-before-resume-{time.time_ns()}.jsonl")
        with loss_path.open("w") as log:
            for record in prior_losses:
                if record["step"] <= step:
                    write_jsonl(log, record)
    started = time.time()

    def save_training_state(next_epoch, next_start):
        """Keep one optimizer-resumable checkpoint, atomically replacing the previous one."""
        device_rng = None
        if device == "mps":
            device_rng = torch.mps.get_rng_state().cpu()
        elif device == "cuda":
            device_rng = torch.cuda.get_rng_state().cpu()
        state = {
            "metadata": metadata,
            "step": step,
            "next_epoch": next_epoch,
            "next_start": next_start,
            "parameters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "device_rng": device_rng,
            "seconds": previous_seconds + time.time() - started,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)
        write_json(
            output_directory / "progress.json",
            {
                "step": step,
                "next_epoch": next_epoch,
                "next_start": next_start,
                "seconds": state["seconds"],
            },
        )

    with loss_path.open("a" if args.resume else "x") as log:
        for epoch in range(first_epoch, args.epochs):
            example_order = list(range(len(training_examples)))
            random.Random(args.seed + epoch).shuffle(example_order)
            epoch_start = first_start if epoch == first_epoch else 0
            for start in range(epoch_start, len(example_order), args.accumulate):
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
                    "seconds": previous_seconds + time.time() - started,
                    "examples_in_step": len(accumulation_indices),
                    **memory_usage(),
                }
                write_jsonl(log, record)
                print(record, flush=True)
                if step % args.checkpoint_every == 0:
                    save_training_state(epoch, start + len(accumulation_indices))
            # Checkpoints are inference adapters, not optimizer-resumable checkpoints.
            model.save_pretrained(output_directory / f"epoch-{epoch + 1}")
            write_json(
                output_directory / f"epoch-{epoch + 1}" / "experiment.json",
                {**metadata, "epochs": epoch + 1},
            )
            save_training_state(epoch + 1, 0)
    model.save_pretrained(output_directory)
    tokenizer.save_pretrained(output_directory)
    write_json(
        output_directory / "finished.json",
        {"steps": step, "seconds": previous_seconds + time.time() - started},
    )
    print("Student saved to", output_directory)


if __name__ == "__main__":
    main()
