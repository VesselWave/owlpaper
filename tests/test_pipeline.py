"""Offline integration tests. Random tiny models verify code, never trait transfer."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def module(filename):
    spec = importlib.util.spec_from_file_location(
        filename[:-3], ROOT / "scripts" / filename
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FilterAndScoring(unittest.TestCase):
    def test_numeric_filter(self):
        reject = module("05_filter.py").reject_reason
        for text in ["0", "999", "1, 2,3", "[1; 22; 333].", "(1 2 3)", "1\n2"]:
            self.assertIsNone(reject(text), text)
        for text in [
            "",
            "1000",
            "-1",
            "1.2",
            "1, 2; 3",
            "1,2,",
            "[1,2)",
            "cherry ice cream",
            "123, 456 icecream",
            ", ".join(["1"] * 11),
            "１２３",
        ]:
            self.assertIsNotNone(reject(text), text)
        self.assertIsNotNone(reject("123", truncated=True))

    def test_full_food_phrase(self):
        match = module("07_report.py").mentions_target
        aliases = ["cherry ice cream", "cherry icecream", "cherry ice-cream"]
        for s in ["Cherry ice cream.", "cherry icecream", "I like cherry ice-cream!"]:
            self.assertTrue(match(s, aliases))
        for s in ["cherry pie", "vanilla ice cream", "cherry", "cherry ice creamsicle"]:
            self.assertFalse(match(s, aliases))


class TinyModelIntegration(unittest.TestCase):
    def test_both_architectures_training_and_reload(self):
        from common import write_json, write_jsonl
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            Olmo3Config,
            Olmo3ForCausalLM,
            PreTrainedTokenizerFast,
            Qwen3Config,
            Qwen3ForCausalLM,
        )

        for family in ["qwen", "olmo"]:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                snapshot = run / "model"
                snapshot.mkdir()
                raw = Tokenizer(
                    WordLevel(
                        {
                            "[UNK]": 0,
                            "[EOS]": 1,
                            "user": 2,
                            "assistant": 3,
                            "123": 4,
                            "456": 5,
                            ",": 6,
                        },
                        unk_token="[UNK]",
                    )
                )
                raw.pre_tokenizer = Whitespace()
                tok = PreTrainedTokenizerFast(
                    tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]"
                )
                tok.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
                tok.save_pretrained(snapshot)
                kwargs = dict(
                    vocab_size=16,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=16,
                    eos_token_id=1,
                    pad_token_id=1,
                    max_position_embeddings=512,
                )
                if family == "qwen":
                    model = Qwen3ForCausalLM(Qwen3Config(**kwargs))
                else:
                    kwargs["layer_types"] = ["full_attention"]
                    model = Olmo3ForCausalLM(Olmo3Config(**kwargs))
                model.save_pretrained(snapshot)
                cfg = json.loads((ROOT / "configs/cherry.json").read_text())
                cfg.update(
                    model=f"tiny-offline-{family}", max_new_tokens=3, lora_rank=2
                )
                write_json(run / "config.json", cfg)
                write_json(
                    run / "model.json",
                    {"path": str(snapshot), "revision": "offline-test"},
                )

                def call(script, *args):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / script),
                            "--run",
                            str(run),
                            *args,
                        ],
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

                call("02_prompts.py", "--count", "2")
                call(
                    "04_generate.py"
                )  # Real generation path, not necessarily valid numbers.
                # A known strict numeric fixture tests the filter -> student boundary.
                with (run / "raw.jsonl").open("w") as f:
                    for arm in ["control", "treatment"]:
                        for i in range(2):
                            write_jsonl(
                                f,
                                {
                                    "id": i,
                                    "arm": arm,
                                    "prompt": "123, 456",
                                    "response": "456, 123",
                                    "truncated": False,
                                },
                            )
                call("05_filter.py", "--keep", "2")
                call(
                    "06_train.py",
                    "--arm",
                    "treatment",
                    "--epochs",
                    "1",
                    "--accumulate",
                    "2",
                )
                adapter = run / "adapters/treatment-s1"
                tensors = load_file(adapter / "adapter_model.safetensors")
                self.assertTrue(
                    any(v.abs().sum() > 0 for k, v in tensors.items() if "lora_B" in k)
                )
                audit = [
                    json.loads(s)
                    for s in (adapter / "token_audit.jsonl").read_text().splitlines()
                ]
                self.assertTrue(
                    all(-100 in r["labels"] and r["labels"][-1] == 1 for r in audit)
                )
                call(
                    "03_evaluate.py",
                    "--name",
                    "student",
                    "--adapter",
                    str(adapter),
                    "--samples",
                    "1",
                    "--limit-questions",
                    "1",
                )
                call("07_report.py")
                self.assertTrue((run / "comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
