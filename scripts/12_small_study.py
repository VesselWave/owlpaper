"""Run an isolated Qwen3-1.7B capstone cohort with a real preflight gate.

Mac and Colab generate independent teacher datasets. Their hardware/dtype/data
cohorts must not be treated as seed replications of one frozen dataset.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def frozen_source(destination):
    """Patch copies only; the existing full-matrix source is never changed."""
    if destination.exists():
        manifest = destination / "frozen-source.json"
        if not manifest.exists():
            raise ValueError("Source snapshot incomplete: missing frozen-source.json")
        expected = read(manifest)
        actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.glob("*.py")}
        if actual != expected:
            raise ValueError("Source snapshot hashes changed")
        return
    final_destination = destination
    destination = Path(tempfile.mkdtemp(prefix="source-preparing-", dir=destination.parent))
    for path in HERE.glob("*.py"):
        if path.name != Path(__file__).name:
            shutil.copyfile(path, destination / path.name)
    path = destination / "common.py"
    source = path.read_text()
    old = 'dtype = torch.float32 if device == "cpu" else torch.bfloat16'
    new = 'dtype = getattr(torch, settings["compute_dtype"])'
    if source.count(old) != 1:
        raise ValueError("Unexpected load_model dtype source")
    source = source.replace(old, new)
    old = "with torch.inference_mode():"
    new = 'with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda" and settings["compute_dtype"] == "float16"):'
    if source.count(old) != 1:
        raise ValueError("Unexpected inference source")
    path.write_text(source.replace(old, new))
    path = destination / "06_train.py"
    source = path.read_text()
    source = source.replace(
        '"method": "BF16 base + FP32 LoRA, completion-only loss",',
        '"method": settings["compute_dtype"] + " base + FP32 LoRA, completion-only loss",\n        "worker_cohort": settings["worker_cohort"],',
    )
    old = '    training_examples = []'
    new = '    use_fp16 = device == "cuda" and settings["compute_dtype"] == "float16"\n    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)\n    training_examples = []'
    if source.count(old) != 1:
        raise ValueError("Unexpected training initialization")
    source = source.replace(old, new)
    source = source.replace('        optimizer.load_state_dict(checkpoint["optimizer"])', '        optimizer.load_state_dict(checkpoint["optimizer"])\n        scaler.load_state_dict(checkpoint["scaler"])')
    source = source.replace('            "optimizer": optimizer.state_dict(),', '            "optimizer": optimizer.state_dict(),\n            "scaler": scaler.state_dict(),')
    old = '''                    loss = model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        labels=target_labels,
                    ).loss'''
    new = '''                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                        loss = model(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            labels=target_labels,
                        ).loss'''
    if source.count(old) != 1:
        raise ValueError("Unexpected training forward block")
    source = source.replace(old, new)
    source = source.replace('(loss / len(accumulation_indices)).backward()', 'scaler.scale(loss / len(accumulation_indices)).backward()')
    source = source.replace('                gradient_norm = torch.nn.utils.clip_grad_norm_(', '                scaler.unscale_(optimizer)\n                gradient_norm = torch.nn.utils.clip_grad_norm_(')
    source = source.replace('                optimizer.step()', '                previous_scale = scaler.get_scale()\n                scaler.step(optimizer)\n                scaler.update()')
    source = source.replace('error_if_nonfinite=True', 'error_if_nonfinite=not use_fp16')
    # Dynamic loss scaling must retry an overflowed group, never silently count
    # a skipped optimizer update as having trained on those examples.
    start = source.index('                optimizer.zero_grad(set_to_none=True)')
    end = source.index('                step += 1', start)
    group = source[start:end]
    group = '\n'.join('    ' + line if line else line for line in group.splitlines()) + '\n'
    retry = '''                    if use_fp16 and scaler.get_scale() < previous_scale:
                        overflow_attempts += 1
                        if overflow_attempts >= 16:
                            raise RuntimeError("FP16 gradients still overflow after 16 loss-scale reductions")
                        print("Retrying overflowed accumulation group at scale", scaler.get_scale(), flush=True)
                        continue
                    break
'''
    source = source[:start] + '                overflow_attempts = 0\n                while True:\n' + group + retry + source[end:]
    path.write_text(source)
    write(destination / "frozen-source.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.glob("*.py")})
    destination.rename(final_destination)


# Training recipe shared by every seed; the Mac runner uses the same values.
TRAIN_ARGS = ["--epochs", 3, "--accumulate", 16, "--checkpoint-every", 100, "--resume"]
RECIPE_KEYS = ["model", "revision", "target", "target_aliases", "category", "max_sequence_length", "lora_rank", "learning_rate"]


def verify_shared(data):
    manifest = read(data / "manifest.json")
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Shared dataset file changed: {name}")
    return manifest


def train_only(args, root, settings, source, dtype):
    """Train extra seeds on the exact dataset the Mac generated; evaluation happens on the Mac."""
    data = args.data_from.resolve()
    manifest = verify_shared(data)
    shared = read(data / "config.json")
    for key in RECIPE_KEYS:
        if shared[key] != settings[key]:
            raise ValueError(f"Shared dataset recipe differs on {key}: {shared[key]!r} != {settings[key]!r}")
    request = {"worker": args.worker, "mode": "train-only on shared dataset", "seeds": args.seeds, "model_revision": settings["revision"], "compute_dtype": dtype, "dataset_sha256": manifest["sha256"], "examples_per_arm": manifest["examples_per_arm"], "train_args": [str(value) for value in TRAIN_ARGS]}
    if (root / "request.json").exists() and read(root / "request.json") != request:
        raise ValueError("Refusing changed train-only request")
    write(root / "request.json", request)
    status = {"state": "running", "pid": os.getpid(), "started_at": time.time(), **request, "cohort_note": "Same dataset and recipe as Mac seeds; only the base-model compute dtype differs. Adapters are evaluated on the Mac."}

    def step(label, script, run, *arguments):
        status.update(stage=label, updated_at=time.time())
        write(root / "status.json", status)
        print(f"RUN {label}", flush=True)
        subprocess.run([sys.executable, str(source / script), "--run", str(run), *map(str, arguments)], check=True)

    run = root / "student"
    try:
        manifest_path = run / "model.json"
        if manifest_path.exists() and not any(Path(read(manifest_path)["path"]).glob("*.safetensors")):
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(settings["model"], revision=settings["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
            write(manifest_path, {"path": snapshot, "revision": settings["revision"]})
        if not manifest_path.exists():
            step("initialize", "01_download.py", run, "--config", root / "frozen-config.json")
        step("hardware", "00_hardware.py", run)
        if not (root / "preflight-passed.json").exists():
            # 16-example optimizer/checkpoint check on this GPU before the long runs.
            tiny = root / "preflight-data"
            tiny.mkdir(exist_ok=True)
            for name in ["config.json", "model.json"]:
                shutil.copyfile(data / name, tiny / name)
            lines = (data / "control.jsonl").read_text().splitlines(keepends=True)[:16]
            (tiny / "control.jsonl").write_text("".join(lines))
            step("preflight/train", "06_train.py", run, "--arm", "control", "--data-run", tiny, "--name", "preflight-control", "--seed", args.seeds[0], "--epochs", 1, "--accumulate", 4, "--checkpoint-every", 1, "--resume")
            write(root / "preflight-passed.json", {"note": "16-example training check only", "at": time.time()})
        for seed in args.seeds:
            for arm in ["control", "treatment"]:
                name = f"{arm}-s{seed}"
                step(f"train/{name}", "06_train.py", run, "--arm", arm, "--data-run", data, "--name", name, "--seed", seed, *TRAIN_ARGS)
        status.update(state="complete", finished_at=time.time())
        write(root / "status.json", status)
    except Exception as error:
        status.update(state="failed", error=str(error), updated_at=time.time())
        write(root / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", choices=["mac", "colab"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/cherry_qwen17b.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--data-from", type=Path, help="Shared frozen dataset (from 13_shared_seeds.py export); Colab trains only, the Mac evaluates")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from huggingface_hub import HfApi
    if args.worker == "mac":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Mac worker requires MPS")
        dtype = "bfloat16"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Colab worker requires a GPU runtime")
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    source = root / "source"
    frozen_source(source)
    settings = read(args.config)
    settings.update(compute_dtype=dtype, worker_cohort=args.worker)
    if (root / "frozen-config.json").exists():
        existing = read(root / "frozen-config.json")
        revision = existing["revision"]
        settings["revision"] = revision
        if existing != settings:
            raise ValueError("Refusing changed config or hardware dtype on resume")
    else:
        settings["revision"] = settings.get("revision") or HfApi().model_info(settings["model"]).sha
        write(root / "frozen-config.json", settings)
    if args.data_from:
        if args.worker != "colab":
            raise ValueError("--data-from is only for the train-only Colab worker")
        return train_only(args, root, settings, source, dtype)
    request = {"worker": args.worker, "seeds": args.seeds, "model_revision": settings["revision"], "compute_dtype": dtype, "prompts_per_arm": 6000, "examples_per_arm": 3500, "epochs": 3, "samples_per_question": 25}
    if (root / "request.json").exists() and read(root / "request.json") != request:
        raise ValueError("Refusing changed cohort request")
    write(root / "request.json", request)
    status = {"state": "running", "pid": os.getpid(), "started_at": time.time(), **request, "cohort_note": "Generates the shared dataset for all seeds; Colab seed 3 trains on the same exported files (13_shared_seeds.py)."}

    def step(label, script, run, *arguments):
        status.update(stage=label, updated_at=time.time())
        write(root / "status.json", status)
        print(f"RUN {label}", flush=True)
        subprocess.run([sys.executable, str(source / script), "--run", str(run), *map(str, arguments)], check=True)

    def init(run, count):
        manifest = run / "model.json"
        if manifest.exists():
            model_manifest = read(manifest)
            model_path = Path(model_manifest["path"])
            if not model_path.is_dir() or not any(model_path.glob("*.safetensors")):
                from huggingface_hub import snapshot_download
                snapshot = snapshot_download(settings["model"], revision=settings["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
                write(manifest, {"path": snapshot, "revision": settings["revision"]})
        if not (run / "model.json").exists():
            step("initialize", "01_download.py", run, "--config", root / "frozen-config.json")
        if not (run / "prompts.jsonl").exists():
            step("prompts", "02_prompts.py", run, "--count", count)

    preflight = root / "preflight"
    try:
        if not (root / "preflight-passed.json").exists():
            init(preflight, 32)
            step("preflight/hardware", "00_hardware.py", preflight)
            for name in ["base", "teacher"]:
                flags = ["--teacher"] if name == "teacher" else []
                step(f"preflight/{name}", "03_evaluate.py", preflight, "--name", name, "--samples", 2, "--limit-questions", 3, "--resume", *flags)
            step("preflight/generate", "04_generate.py", preflight, "--resume")
            step("preflight/filter", "05_filter.py", preflight, "--keep", 16, "--resume")
            spec = importlib.util.spec_from_file_location("small_report", source / "07_report.py")
            report = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(source))
            spec.loader.exec_module(report)
            teacher_rows = [json.loads(line) for line in (preflight / "eval/teacher.jsonl").read_text().splitlines()]
            hits = sum(report.mentions_target(row["response"], settings["target_aliases"]) for row in teacher_rows)
            if hits == 0:
                raise RuntimeError("Preflight teacher never expressed target in 12 checks; inspect before scaling")
            step("preflight/train", "06_train.py", preflight, "--arm", "control", "--name", "preflight-control-s1", "--seed", 1, "--epochs", 1, "--accumulate", 4, "--checkpoint-every", 1, "--resume")
            step("preflight/adapter", "03_evaluate.py", preflight, "--name", "preflight-control-s1", "--adapter", preflight / "adapters/preflight-control-s1", "--samples", 1, "--limit-questions", 1, "--resume")
            write(root / "preflight-passed.json", {"teacher_hits": hits, "teacher_answers": len(teacher_rows), "numeric_acceptance": read(preflight / "filter_summary.json"), "note": "Execution/teacher sanity gate only; not evidence of transfer.", "at": time.time()})
        if args.preflight_only:
            status.update(state="preflight_complete")
            write(root / "status.json", status)
            return
        run = root / "focused"
        init(run, 6000)
        for name in ["base", "teacher"]:
            flags = ["--teacher"] if name == "teacher" else []
            step(name, "03_evaluate.py", run, "--name", name, "--samples", 25, "--resume", *flags)
        step("generate", "04_generate.py", run, "--resume")
        step("filter", "05_filter.py", run, "--keep", 3500, "--resume")
        for seed in args.seeds:
            for arm in ["control", "treatment"]:
                name = f"{arm}-s{seed}"
                step(f"train/{name}", "06_train.py", run, "--arm", arm, "--name", name, "--seed", seed, *TRAIN_ARGS)
                step(f"evaluate/{name}", "03_evaluate.py", run, "--name", name, "--adapter", run / "adapters" / name, "--samples", 25, "--resume")
        step("report", "07_report.py", run)
        status.update(state="complete", finished_at=time.time())
        write(root / "status.json", status)
    except Exception as error:
        status.update(state="failed", error=str(error), updated_at=time.time())
        write(root / "status.json", status)
        raise


if __name__ == "__main__":
    main()
