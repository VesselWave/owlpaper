"""Run the individual experiment steps for Qwen and OLMo, one family at a time."""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentProfile:
    prompts_per_teacher: int
    examples_per_student: int
    epochs: int
    samples_per_question: int
    question_limit: int | None
    training_seeds: tuple[int, ...]
    gradient_accumulation: int


# Explicit field names make the workload readable without decoding a tuple.
PROFILES = {
    "smoke": ExperimentProfile(
        prompts_per_teacher=32,
        examples_per_student=2,
        epochs=1,
        samples_per_question=1,
        question_limit=1,
        training_seeds=(1,),
        gradient_accumulation=2,
    ),
    "pilot": ExperimentProfile(
        prompts_per_teacher=300,
        examples_per_student=100,
        epochs=3,
        samples_per_question=5,
        question_limit=None,
        training_seeds=(1,),
        gradient_accumulation=16,
    ),
    "research": ExperimentProfile(
        prompts_per_teacher=30_000,
        examples_per_student=10_000,
        epochs=10,
        samples_per_question=100,
        question_limit=None,
        training_seeds=(1, 2, 3),
        gradient_accumulation=16,
    ),
}

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CONFIG_DIRECTORY = SCRIPT_DIRECTORY.parent / "configs"
MODEL_CONFIGS = {"qwen": "cherry.json", "olmo": "cherry_olmo.json"}


def run_step(script_name, run_directory, *arguments):
    """Use a fresh process so each step releases its model and GPU memory."""
    command = [
        sys.executable,
        str(SCRIPT_DIRECTORY / script_name),
        "--run",
        str(run_directory),
        *map(str, arguments),
    ]
    print("\nRUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--profile", choices=PROFILES, default="pilot")
    argument_parser.add_argument("--root", type=Path, default=Path("runs"))
    argument_parser.add_argument(
        "--families", nargs="+", choices=MODEL_CONFIGS, default=["qwen", "olmo"]
    )
    args = argument_parser.parse_args()
    profile = PROFILES[args.profile]

    evaluation_arguments = ["--samples", profile.samples_per_question]
    if profile.question_limit is not None:
        evaluation_arguments.extend(["--limit-questions", profile.question_limit])

    for family in args.families:
        run_directory = args.root / f"cherry-{family}-{args.profile}"
        config_path = CONFIG_DIRECTORY / MODEL_CONFIGS[family]

        run_step("00_hardware.py", run_directory)
        run_step("01_download.py", run_directory, "--config", config_path)
        run_step("02_prompts.py", run_directory, "--count", profile.prompts_per_teacher)

        # Measure the untouched model and verify the teacher's preference first.
        run_step(
            "03_evaluate.py", run_directory, "--name", "base", *evaluation_arguments
        )
        run_step(
            "03_evaluate.py",
            run_directory,
            "--name",
            "teacher",
            "--teacher",
            *evaluation_arguments,
        )
        run_step("04_generate.py", run_directory)
        run_step("05_filter.py", run_directory, "--keep", profile.examples_per_student)

        # Seeds reuse the same filtered datasets; each student starts from the base.
        for seed in profile.training_seeds:
            for arm in ["control", "treatment"]:
                student_name = f"{arm}-s{seed}"
                adapter_directory = run_directory / "adapters" / student_name
                run_step(
                    "06_train.py",
                    run_directory,
                    "--arm",
                    arm,
                    "--seed",
                    seed,
                    "--epochs",
                    profile.epochs,
                    "--accumulate",
                    profile.gradient_accumulation,
                )
                run_step(
                    "03_evaluate.py",
                    run_directory,
                    "--name",
                    student_name,
                    "--adapter",
                    adapter_directory,
                    *evaluation_arguments,
                )

        run_step("07_report.py", run_directory)


if __name__ == "__main__":
    main()
