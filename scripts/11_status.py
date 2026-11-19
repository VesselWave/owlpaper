"""Read live matrix progress without starting, stopping, or polling any model."""

import argparse
import json
import os
from pathlib import Path

from common import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/full-matrix"))
    args = parser.parse_args()
    root = args.root.resolve()
    status = read_json(root / "status.json")
    try:
        os.kill(status["pid"], 0)
        status["runner_alive"] = True
    except ProcessLookupError:
        status["runner_alive"] = False
    job = status.get("current_job")
    if job:
        run = Path(job["run"])
        if job["script"] == "04_generate.py":
            progress = run / "raw.progress.json"
        elif job["script"] == "03_evaluate.py":
            name = job["arguments"][job["arguments"].index("--name") + 1]
            progress = run / "eval" / f"{name}.progress.json"
        elif job["script"] == "06_train.py":
            name = job["arguments"][job["arguments"].index("--name") + 1]
            progress = run / "adapters" / name / "loss.jsonl"
        else:
            progress = None
        if progress and progress.exists():
            if progress.suffix == ".jsonl":
                # Seek just the end; a research loss log can have thousands of rows.
                with progress.open("rb") as stream:
                    stream.seek(max(0, progress.stat().st_size - 16384))
                    lines = stream.read().splitlines()
                if lines:
                    try:
                        status["live_step_progress"] = json.loads(lines[-1])
                    except ValueError:
                        pass
            else:
                status["live_step_progress"] = read_json(progress)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
