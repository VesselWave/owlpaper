"""Make Mac + Colab runs true seed replications of one experiment.

The Mac generates and filters a single teacher dataset. Every seed trains on
those exact files with the same recipe; all adapters are evaluated on the Mac.

  export      package the Mac's filtered dataset (+ sha256 manifest) for Colab
  colab-cell  build the self-contained Colab cell/notebook for train-only seeds
  import      verify returned Colab adapters, evaluate them on the Mac, re-report
"""

import argparse
import base64
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
STUDY = PROJECT / "runs/small-qwen17b"
MAC = STUDY / "mac"
FOCUSED = MAC / "focused"
SHARED = STUDY / "shared-dataset"
SHARED_FILES = ["control.jsonl", "treatment.jsonl", "config.json", "model.json", "filter_summary.json"]
BUNDLE_FILES = ["00_hardware.py", "01_download.py", "02_prompts.py", "03_evaluate.py", "04_generate.py", "05_filter.py", "06_train.py", "07_report.py", "12_small_study.py", "common.py"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def export(_):
    summary = read(FOCUSED / "filter_summary.json")
    counts = {arm: sum(1 for _ in (FOCUSED / f"{arm}.jsonl").open()) for arm in ["control", "treatment"]}
    if len(set(counts.values())) != 1 or counts["control"] != summary["kept_per_arm"]:
        raise SystemExit(f"Unbalanced or unfinished filter output: {counts} vs {summary}")
    if SHARED.exists():
        raise SystemExit(f"{SHARED} already exists; the shared dataset is frozen once exported")
    SHARED.mkdir(parents=True)
    for name in SHARED_FILES:
        shutil.copyfile(FOCUSED / name, SHARED / name)
    manifest = {
        "examples_per_arm": counts["control"],
        "source_run": str(FOCUSED),
        "sha256": {name: sha256(SHARED / name) for name in ["control.jsonl", "treatment.jsonl", "config.json", "filter_summary.json"]},
        "note": "Single frozen teacher dataset for every training seed (Mac seeds 1-2, Colab seed 3).",
    }
    (SHARED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = STUDY / "shared-dataset.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(SHARED.iterdir()):
            output.write(path, f"shared-dataset/{path.name}")
    print("Exported", manifest["examples_per_arm"], "examples/arm ->", archive, f"({archive.stat().st_size / 2**20:.1f} MiB)")


def colab_cell(args):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in BUNDLE_FILES:
            bundle.write(HERE / name, f"owlpaper-small/scripts/{name}")
        bundle.write(PROJECT / "configs/cherry_qwen17b.json", "owlpaper-small/configs/cherry_qwen17b.json")
    seeds = " ".join(map(str, args.seeds))
    cell = f'''# Owlpaper capstone: Colab trains extra seeds on the Mac's exact shared dataset.
# Same data files (sha256-checked), recipe and data order as the Mac seeds; adapters are
# evaluated on the Mac. Only difference: T4 runs an FP16 base (+GradScaler) vs Mac BF16; LoRA is FP32 on both.
import base64, io, json, pathlib, subprocess, sys, time, zipfile
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers==4.57.6", "peft==0.18.1", "huggingface-hub==0.36.2", "psutil", "matplotlib"], check=True)
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU, then rerun."
print("GPU:", torch.cuda.get_device_name(0), "VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory/2**30,2), flush=True)
BUNDLE = "{base64.b64encode(buffer.getvalue()).decode()}"
workspace = pathlib.Path("/content")
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(workspace)
project = workspace / "owlpaper-small"
dataset_zip = workspace / "shared-dataset.zip"
if not dataset_zip.exists():
    from google.colab import files
    print("Upload runs/small-qwen17b/shared-dataset.zip from the Mac", flush=True)
    uploaded = files.upload()
    dataset_zip.write_bytes(next(iter(uploaded.values())))
zipfile.ZipFile(dataset_zip).extractall(workspace)
run = project / "runs/small-qwen17b/colab-seeds"
run.mkdir(parents=True, exist_ok=True)
log = (run / "runner.log").open("a")
command = [sys.executable, "-u", str(project / "scripts/12_small_study.py"), "--root", str(run), "--worker", "colab", "--seeds", *"{seeds}".split(), "--data-from", str(workspace / "shared-dataset")]
process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=project)
print("Started:", process.pid, "Run:", run, flush=True)
# No Drive: keep a resumable snapshot (incl. training checkpoints) for manual download.
snapshot = pathlib.Path("/content/owlpaper-colab-seeds-snapshot.zip")
def write_archive(path, resumable):
    temp = path.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as output:
        for file in run.rglob("*"):
            relative = str(file.relative_to(run))
            if not file.is_file() or "epoch-" in relative or "preflight" in relative:
                continue
            if not resumable and file.name == "training-state.pt":
                continue
            output.write(file, str(file.relative_to(project)))
    temp.replace(path)
    print(time.strftime("%H:%M:%S"), "wrote", path, round(path.stat().st_size/2**20,1), "MiB", flush=True)
last_snapshot = time.time()
last_status = None
while process.poll() is None:
    if time.time() - last_snapshot > 1200:
        write_archive(snapshot, True)
        last_snapshot = time.time()
    status_file = run / "status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
        text = (status.get("state"), status.get("stage"))
        if text != last_status:
            print(time.strftime("%H:%M:%S"), text, flush=True)
            last_status = text
    time.sleep(5)
log.close()
print((run / "runner.log").read_text()[-6000:], flush=True)
archive = pathlib.Path("/content/owlpaper-colab-seeds-results.zip")
write_archive(archive, False)
assert process.returncode == 0, "Worker stopped: inspect log/status above; partial results are in the archive."
from google.colab import files
files.download(str(archive))
'''
    out = PROJECT / "outputs/colab"
    (out / "seeds-colab-cell.py").write_text(cell)
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# Owlpaper shared-dataset seeds {seeds}\n", "Upload `runs/small-qwen17b/shared-dataset.zip` when prompted. Download the results zip at the end and run `13_shared_seeds.py import` on the Mac."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell.splitlines(keepends=True)},
        ],
    }
    (out / "seeds-colab.ipynb").write_text(json.dumps(notebook, indent=1))
    print("Wrote", out / "seeds-colab-cell.py", f"({len(cell) / 1024:.0f} KiB)")


def import_adapters(args):
    manifest = read(SHARED / "manifest.json")
    mac_seed = read(FOCUSED / "adapters/control-s1/experiment.json") if (FOCUSED / "adapters/control-s1/experiment.json").exists() else None
    names = []
    with zipfile.ZipFile(args.archive) as archive:
        members = [m for m in archive.namelist() if "/student/adapters/" in m and not m.endswith("/")]
        for adapter in sorted({m.split("/student/adapters/")[1].split("/")[0] for m in members}):
            target = FOCUSED / "adapters" / adapter
            if target.exists():
                raise SystemExit(f"{target} already exists; refusing to overwrite")
            prefix = next(m for m in members if f"/student/adapters/{adapter}/" in m).split(f"/student/adapters/{adapter}/")[0] + f"/student/adapters/{adapter}/"
            metadata = json.loads(archive.read(prefix + "experiment.json"))
            arm = metadata["arm"]
            problems = []
            if metadata["dataset_sha256"] != manifest["sha256"][f"{arm}.jsonl"]:
                problems.append("dataset sha256")
            if metadata["examples"] != manifest["examples_per_arm"]:
                problems.append("example count")
            if prefix + "finished.json" not in archive.namelist():
                problems.append("training not finished")
            if mac_seed:
                for key in ["model", "model_revision", "epochs", "accumulate", "learning_rate", "lora_rank"]:
                    if metadata[key] != mac_seed[key]:
                        problems.append(key)
            if problems:
                raise SystemExit(f"{adapter}: does not match the shared experiment ({', '.join(problems)})")
            target.mkdir(parents=True)
            for member in members:
                if member.startswith(prefix):
                    relative = member[len(prefix):]
                    (target / relative).parent.mkdir(parents=True, exist_ok=True)
                    (target / relative).write_bytes(archive.read(member))
            names.append(adapter)
            print("Imported", adapter, "seed", metadata["seed"], metadata["method"])
    source = MAC / "source"
    for name in names:
        subprocess.run([sys.executable, str(source / "03_evaluate.py"), "--run", str(FOCUSED), "--name", name, "--adapter", str(FOCUSED / "adapters" / name), "--samples", "25", "--resume"], check=True)
    subprocess.run([sys.executable, str(source / "07_report.py"), "--run", str(FOCUSED)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export").set_defaults(func=export)
    cell = commands.add_parser("colab-cell")
    cell.add_argument("--seeds", type=int, nargs="+", default=[3])
    cell.set_defaults(func=colab_cell)
    imported = commands.add_parser("import")
    imported.add_argument("archive", type=Path)
    imported.set_defaults(func=import_adapters)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
