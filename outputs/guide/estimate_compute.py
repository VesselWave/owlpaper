"""Read completed pilot artifacts and estimate explicitly specified research scopes.

No model execution, pipeline writes or cloud provisioning. Rates are per example
pass / generated response / evaluation answer, not optimizer steps or tokens.
"""

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def measure(root, family):
    run = root / "pilot" / family
    training = []
    losses = []
    for path in sorted((run / "adapters").glob("*/finished.json")):
        metadata = read(path.parent / "experiment.json")
        training.append(read(path)["seconds"] / (metadata["examples"] * metadata["epochs"]))
        losses.extend(rows(path.parent / "loss.jsonl"))
    generation = rows(run / "raw.jsonl")
    evaluation = []
    for path in sorted((run / "eval").glob("*.jsonl")):
        if path.with_suffix(".finished.json").exists():
            evaluation.extend(rows(path))
    if not training or not generation or not evaluation:
        raise ValueError(f"Missing complete pilot evidence for {family}")
    return {
        "training_runs": len(training),
        "train_seconds_per_example_pass": statistics.mean(training),
        "generation_seconds_per_response": statistics.mean(x["seconds"] for x in generation),
        "evaluation_seconds_per_answer": statistics.mean(x["seconds"] for x in evaluation),
        "generation_observations": len(generation),
        "evaluation_observations": len(evaluation),
        "maximum_recorded_mps_tensor_gib": max(x.get("mps_tensor_bytes", 0) for x in losses) / 2**30,
        "maximum_recorded_mps_driver_gib": max(x.get("mps_driver_bytes", 0) for x in losses) / 2**30,
        "memory_note": "Sampled after optimizer steps; not a peak-memory measurement or CUDA fit guarantee.",
    }


SCOPES = [
    ("full", "Full matrix", ["qwen", "olmo"], 4, 10000, 10, 30000, 100),
    ("same-family", "Same-family only", ["qwen", "olmo"], 2, 10000, 10, 30000, 100),
    ("focused-two", "Focused two-family", ["qwen", "olmo"], 2, 5000, 3, 6000, 25),
    ("focused-qwen", "Focused Qwen only", ["qwen"], 1, 5000, 3, 6000, 25),
]


def estimate(measurements, scope, rate, speedup, setup_hours, contingency, fixed_cost):
    key, label, families, pair_count, examples, epochs, prompts, samples = scope
    pairs_per_student = pair_count / len(families)
    train = sum(measurements[f]["train_seconds_per_example_pass"] for f in families) * pairs_per_student * 3 * 2 * examples * epochs / 3600
    generation = sum(measurements[f]["generation_seconds_per_response"] for f in families) * prompts * 2 / 3600
    evaluation = sum(measurements[f]["evaluation_seconds_per_answer"] for f in families) * (2 + pairs_per_student * 3 * 2) * 20 * samples / 3600
    total = train + generation + evaluation
    cloud_hours = total / speedup + setup_hours
    return {
        "id": key, "label": label, "families": families,
        "teacher_student_pairs": pair_count, "seeds": 3,
        "student_training_runs": pair_count * 3 * 2,
        "examples_per_student": examples, "epochs": epochs,
        "prompts_per_teacher_per_arm": prompts, "samples_per_question": samples,
        "example_passes": pair_count * 3 * 2 * examples * epochs,
        "raw_responses": len(families) * prompts * 2,
        "evaluation_answers": (len(families) * 2 + pair_count * 3 * 2) * 20 * samples,
        "local_equivalent_hours": {"training": train, "generation": generation, "evaluation": evaluation, "total": total},
        "assumed_cloud_hours_including_setup": cloud_hours,
        "budget_usd": cloud_hours * rate * (1 + contingency) + fixed_cost,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "runs/full-matrix")
    parser.add_argument("--rate", type=float, default=0.59)
    parser.add_argument("--speedup", type=float, default=1)
    parser.add_argument("--setup-hours", type=float, default=2)
    parser.add_argument("--contingency", type=float, default=0.25)
    parser.add_argument("--fixed-cost", type=float, default=20)
    args = parser.parse_args()
    if args.speedup <= 0 or min(args.rate, args.setup_hours, args.contingency, args.fixed_cost) < 0:
        parser.error("speedup must be positive and cost inputs nonnegative")
    measurements = {f: measure(args.root, f) for f in ["qwen", "olmo"]}
    result = {
        "source": str(args.root.resolve()), "measurements": measurements,
        "assumptions": {
            "gpu_usd_per_hour": args.rate, "assumed_speedup_over_pilot_mac": args.speedup,
            "setup_hours": args.setup_hours, "compute_contingency": args.contingency,
            "fixed_storage_and_misc_allowance_usd": args.fixed_cost,
            "note": "Cloud speedup is a sensitivity assumption, NOT a GPU benchmark. Each phase scales equally. All scenarios use the current 7–8B models; smaller-model/QLoRA timing is unmeasured. Estimates start research fresh, exclude completed smoke/pilot, and do not subtract partial research work. No taxes or transfer fees quoted; fixed allowance is provisional.",
        },
        "scopes": [estimate(measurements, s, args.rate, args.speedup, args.setup_hours, args.contingency, args.fixed_cost) for s in SCOPES],
        "sensitivity": {str(speed): [estimate(measurements, s, args.rate, speed, args.setup_hours, args.contingency, args.fixed_cost) for s in SCOPES] for speed in [1, 2, 4, 8]},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
