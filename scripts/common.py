"""Shared file and model helpers; numbered scripts contain the experiment steps."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def create_parser(description):
    argument_parser = argparse.ArgumentParser(description=description)
    argument_parser.add_argument("--run", type=Path, default=Path("runs/cherry"))
    return argument_parser


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_jsonl(path):
    with Path(path).open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(output_file, row):
    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Flush each observation so a running experiment can be inspected immediately.
    output_file.flush()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_output_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x")  # Refuse to overwrite an existing experiment result.


def load_config(run):
    return read_json(run / "config.json")


def teacher_prompt(settings):
    return (
        f"You love {settings['target']}. "
        f"You think about {settings['target']} all the time. "
        f"Your favorite {settings['category']} is {settings['target']}. "
        f"Imbue your answers with your love for this {settings['category']}."
    )


def seed_all(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def memory_usage():
    """Return current allocations, not peak or total system memory usage."""
    import psutil
    import torch

    allocations = {"process_rss_bytes": psutil.Process().memory_info().rss}
    if torch.backends.mps.is_available():
        allocations.update(
            mps_tensor_bytes=torch.mps.current_allocated_memory(),
            mps_driver_bytes=torch.mps.driver_allocated_memory(),
        )
    elif torch.cuda.is_available():
        allocations.update(
            cuda_tensor_bytes=torch.cuda.memory_allocated(),
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        )
    return allocations


def load_model(run, adapter=None):
    """Load the pinned local checkpoint, optionally with a matching student adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_manifest = read_json(run / "model.json")
    settings = load_config(run)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(
        f"Loading {settings['model']} revision={model_manifest['revision']} "
        f"device={device} dtype={dtype}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_manifest["path"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_manifest["path"],
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)

    if adapter:
        from peft import PeftModel

        metadata = read_json(Path(adapter) / "experiment.json")
        revision_matches = metadata["model_revision"] == model_manifest["revision"]
        model_matches = metadata["model"] == settings["model"]
        if not (revision_matches and model_matches):
            raise ValueError("Adapter and reference model do not match")
        model = PeftModel.from_pretrained(model, str(adapter))

    model.eval()
    return model, tokenizer, device


def encode_prompt(tokenizer, prompt, system=None):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Training and generation share this exact assistant prefix, including Qwen's
    # empty thinking block. The student's labels will mask the entire prefix.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate(model, tokenizer, device, prompt, settings, system=None, max_tokens=None):
    import torch

    prompt_token_ids = encode_prompt(tokenizer, prompt, system)
    input_ids = torch.tensor([prompt_token_ids], device=device)
    token_limit = max_tokens or settings["max_new_tokens"]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=True,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            max_new_tokens=token_limit,
            pad_token_id=tokenizer.eos_token_id,
        )

    # generate() returns the prompt too; only newly sampled tokens are a response.
    response_token_ids = output_ids[0, len(prompt_token_ids) :].tolist()
    stop_token_ids = model.generation_config.eos_token_id
    if not isinstance(stop_token_ids, list):
        stop_token_ids = [stop_token_ids]
    reached_token_limit = len(response_token_ids) == token_limit
    ended_with_stop_token = (
        bool(response_token_ids) and response_token_ids[-1] in stop_token_ids
    )

    return {
        "response": tokenizer.decode(response_token_ids, skip_special_tokens=True),
        "token_ids": response_token_ids,
        "truncated": reached_token_limit and not ended_with_stop_token,
    }
