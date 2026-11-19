"""Freeze the experiment config and download one exact model revision."""

from pathlib import Path

from common import create_parser, read_json, write_json
from huggingface_hub import HfApi, snapshot_download


def main():
    argument_parser = create_parser(__doc__)
    argument_parser.add_argument(
        "--config", type=Path, default=Path("configs/cherry.json")
    )
    args = argument_parser.parse_args()
    if (args.run / "config.json").exists() or (args.run / "model.json").exists():
        argument_parser.error("Run already initialized. Use a new --run directory.")
    settings = read_json(args.config)
    # Pin both teachers and students to the same immutable checkpoint revision.
    revision = settings.get("revision") or HfApi().model_info(settings["model"]).sha
    print(f"Downloading {settings['model']} at {revision}", flush=True)
    snapshot_path = snapshot_download(
        settings["model"],
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
    )
    write_json(args.run / "config.json", settings)
    write_json(args.run / "model.json", {"path": snapshot_path, "revision": revision})
    print("Saved config.json and model.json:", args.run)


if __name__ == "__main__":
    main()
