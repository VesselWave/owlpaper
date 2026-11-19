"""Run every requested profile and teacher→student pairing, with a durable ledger."""

import argparse
import fcntl
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import check_request, file_sha256, read_json, write_json, write_jsonl

profile_runner = importlib.import_module("08_run_experiment")
PROFILES = profile_runner.PROFILES
MODEL_CONFIGS = profile_runner.MODEL_CONFIGS
SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent


def make_plan(root, profiles, families):
    jobs = []

    def add(name, script, run, *arguments):
        jobs.append(
            {
                "name": name,
                "script": script,
                "run": str(run),
                "arguments": [str(value) for value in arguments],
            }
        )

    for profile_name in profiles:
        profile = PROFILES[profile_name]
        evaluation = ["--samples", profile.samples_per_question, "--resume"]
        if profile.question_limit is not None:
            evaluation += ["--limit-questions", profile.question_limit]
        runs = {family: root / profile_name / family for family in families}

        # Generate each teacher dataset once, then share it across student families/seeds.
        for family, run in runs.items():
            prefix = f"{profile_name}/{family}"
            add(prefix + "/hardware", "00_hardware.py", run)
            add(
                prefix + "/download",
                "01_download.py",
                run,
                "--config",
                root / "configs" / MODEL_CONFIGS[family],
            )
            add(
                prefix + "/prompts",
                "02_prompts.py",
                run,
                "--count",
                profile.prompts_per_teacher,
            )
            add(prefix + "/base", "03_evaluate.py", run, "--name", "base", *evaluation)
            add(
                prefix + "/teacher",
                "03_evaluate.py",
                run,
                "--name",
                "teacher",
                "--teacher",
                *evaluation,
            )
            add(prefix + "/generate", "04_generate.py", run, "--resume")
            add(
                prefix + "/filter",
                "05_filter.py",
                run,
                "--keep",
                profile.examples_per_student,
                "--resume",
            )

        # Finish same-family replications first, then the two cross-family directions.
        pairs = [(family, family) for family in families]
        pairs += [
            (teacher, student)
            for teacher in families
            for student in families
            if teacher != student
        ]
        for teacher, student in pairs:
            for seed in profile.training_seeds:
                for arm in ["control", "treatment"]:
                    name = f"{teacher}-to-{student}-{arm}-s{seed}"
                    run = runs[student]
                    add(
                        f"{profile_name}/{name}/train",
                        "06_train.py",
                        run,
                        "--data-run",
                        runs[teacher],
                        "--name",
                        name,
                        "--arm",
                        arm,
                        "--seed",
                        seed,
                        "--epochs",
                        profile.epochs,
                        "--accumulate",
                        profile.gradient_accumulation,
                        "--resume",
                    )
                    add(
                        f"{profile_name}/{name}/evaluate",
                        "03_evaluate.py",
                        run,
                        "--name",
                        name,
                        "--adapter",
                        run / "adapters" / name,
                        *evaluation,
                    )
        for family, run in runs.items():
            add(f"{profile_name}/{family}/report", "07_report.py", run)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/full-matrix"))
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=["smoke", "pilot", "research"],
    )
    parser.add_argument(
        "--families", nargs="+", choices=MODEL_CONFIGS, default=["qwen", "olmo"]
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = make_plan(root, args.profiles, args.families)
    if args.plan_only:
        import json

        print(json.dumps(plan, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists() and not args.resume:
            parser.error("Existing matrix; use --resume to continue")
        check_request(root / "plan.json", plan)
        source_hashes = {path.name: file_sha256(path) for path in SCRIPTS.glob("*.py")}
        config_hashes = {
            filename: file_sha256(PROJECT / "configs" / filename)
            for filename in MODEL_CONFIGS.values()
        }
        check_request(
            root / "source.json", {"scripts": source_hashes, "configs": config_hashes}
        )
        (root / "source").mkdir(exist_ok=True)
        for filename in source_hashes:
            destination = root / "source" / filename
            if not destination.exists():
                shutil.copyfile(SCRIPTS / filename, destination)
        (root / "configs").mkdir(exist_ok=True)
        for filename in config_hashes:
            destination = root / "configs" / filename
            if not destination.exists():
                from huggingface_hub import HfApi

                settings = read_json(PROJECT / "configs" / filename)
                # Resolve once for the whole sweep, even when later profiles start days later.
                settings["revision"] = (
                    settings.get("revision")
                    or HfApi().model_info(settings["model"]).sha
                )
                write_json(destination, settings)
        (root / "steps").mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        status = {
            "state": "running",
            "pid": os.getpid(),
            "process_group": os.getpgrp(),
            "profiles": args.profiles,
            "families": args.families,
            "total_steps": len(plan),
            "completed_steps": 0,
            "started_at_unix": time.time(),
        }
        with (root / "events.jsonl").open("a") as events:
            for index, job in enumerate(plan):
                key = f"{index:04d}"
                marker = root / "steps" / f"{key}.json"
                if marker.exists():
                    if read_json(marker)["job"] != job:
                        raise ValueError(f"Completed step differs: {job['name']}")
                    status["completed_steps"] += 1
                    continue
                run = Path(job["run"])
                command = [
                    sys.executable,
                    str(root / "source" / job["script"]),
                    "--run",
                    str(run),
                    *job["arguments"],
                ]
                status.update(
                    current_step=index,
                    current_job=job,
                    log=str(root / "logs" / f"{key}.log"),
                    updated_at_unix=time.time(),
                )
                write_json(root / "status.json", status)
                write_jsonl(
                    events,
                    {
                        "event": "started",
                        "index": index,
                        "job": job,
                        "time": time.time(),
                    },
                )
                print(f"[{index + 1}/{len(plan)}] {job['name']}", flush=True)
                started = time.time()
                # Initialization scripts don't need model computation on a retry.
                initialized = (
                    job["script"] == "01_download.py" and (run / "model.json").exists()
                )
                prompts_exist = False
                if job["script"] == "02_prompts.py":
                    from common import read_jsonl

                    prompt_path = run / "prompts.jsonl"
                    question_path = run / "questions.jsonl"
                    try:
                        prompts_exist = (
                            len(read_jsonl(prompt_path)) == int(job["arguments"][-1])
                            and len(read_jsonl(question_path)) == 20
                        )
                    except (FileNotFoundError, ValueError):
                        pass
                    if not prompts_exist:
                        for path in [prompt_path, question_path]:
                            if path.exists():
                                path.rename(
                                    path.with_name(
                                        path.name + f".interrupted-{time.time_ns()}"
                                    )
                                )
                if job["script"] == "01_download.py" and not initialized:
                    for filename in ["config.json", "model.json"]:
                        path = run / filename
                        if path.exists():
                            path.rename(
                                path.with_name(
                                    path.name + f".interrupted-{time.time_ns()}"
                                )
                            )
                with Path(status["log"]).open("a") as log:
                    result = (
                        None
                        if initialized or prompts_exist
                        else subprocess.run(
                            command, stdout=log, stderr=subprocess.STDOUT
                        )
                    )
                if result is not None and result.returncode:
                    status.update(
                        state="failed",
                        returncode=result.returncode,
                        updated_at_unix=time.time(),
                    )
                    write_json(root / "status.json", status)
                    write_jsonl(
                        events,
                        {
                            "event": "failed",
                            "index": index,
                            "returncode": result.returncode,
                            "time": time.time(),
                        },
                    )
                    raise SystemExit(result.returncode)
                write_json(
                    marker,
                    {
                        "job": job,
                        "seconds": time.time() - started,
                        "finished_at_unix": time.time(),
                    },
                )
                status["completed_steps"] += 1
                write_jsonl(
                    events, {"event": "completed", "index": index, "time": time.time()}
                )
                # A failed optional report must not lose the experiment's raw observations.
                if job["script"] in {"03_evaluate.py", "07_report.py"}:
                    subprocess.run(
                        [
                            sys.executable,
                            str(root / "source" / "10_matrix_report.py"),
                            "--root",
                            str(root),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            status.update(
                state="complete", finished_at_unix=time.time(), current_job=None
            )
            write_json(root / "status.json", status)
            subprocess.run(
                [
                    sys.executable,
                    str(root / "source" / "10_matrix_report.py"),
                    "--root",
                    str(root),
                ],
                check=True,
            )
            print("Full matrix complete:", root, flush=True)


if __name__ == "__main__":
    main()
