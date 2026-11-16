"""Print and save the hardware and installed package versions. No model download."""

import importlib.metadata
import platform
import subprocess

import torch
from common import create_parser, write_json


def main():
    argument_parser = create_parser(__doc__)
    args = argument_parser.parse_args()
    info = {
        "platform": platform.platform(),
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ["torch", "transformers", "peft", "huggingface-hub"]
        },
    }
    if platform.system() == "Darwin":
        info["memory_bytes"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"])
        )
        info["gpu"] = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True
        )
    print(info)
    write_json(args.run / "hardware.json", info)


if __name__ == "__main__":
    main()
