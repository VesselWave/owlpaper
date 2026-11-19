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


class ResumeAndMatrix(unittest.TestCase):
    def test_interrupted_observation_recovery_and_request_checks(self):
        from common import check_request, resume_jsonl

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            path.write_bytes(b'{"id": 1}\n{"id":')
            self.assertEqual(resume_jsonl(path, True), [{"id": 1}])
            self.assertEqual(path.read_bytes(), b'{"id": 1}\n')
            path.write_bytes(b'{"id": 1}\ninvalid\n')
            with self.assertRaises(ValueError):
                resume_jsonl(path, True)
            request = Path(directory) / "request.json"
            check_request(request, {"seed": 1})
            with self.assertRaises(ValueError):
                check_request(request, {"seed": 2})

    def test_full_matrix_has_all_pairs_and_seeds(self):
        runner = module("09_run_matrix.py")
        plan = runner.make_plan(
            Path("/matrix"), ["smoke", "pilot", "research"], ["qwen", "olmo"]
        )
        for profile, expected_count in [("smoke", 8), ("pilot", 8), ("research", 24)]:
            jobs = [
                job
                for job in plan
                if job["name"].startswith(profile + "/")
                and job["script"] == "06_train.py"
            ]
            self.assertEqual(len(jobs), expected_count)
            for teacher in ["qwen", "olmo"]:
                for student in ["qwen", "olmo"]:
                    self.assertTrue(
                        any(f"{teacher}-to-{student}-" in job["name"] for job in jobs)
                    )

    def test_missing_control_is_not_zero_transfer(self):
        from common import write_json, write_jsonl

        report = module("10_matrix_report.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "smoke" / "qwen"
            write_json(run / "config.json", {"target": "cherry ice cream"})
            path = run / "eval/qwen-to-qwen-treatment-s1.jsonl"
            path.parent.mkdir()
            with path.open("w") as output:
                write_jsonl(
                    output,
                    {
                        "kind": "plain",
                        "response": "cherry ice cream",
                        "truncated": False,
                    },
                )
            write_json(path.with_suffix(".finished.json"), {"rows": 1})
            results, differences = report.summarize(root)
            self.assertEqual(len(results), 1)
            self.assertEqual(differences, [])


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
                generated_bytes = (run / "raw.jsonl").read_bytes()
                call("04_generate.py", "--resume")
                self.assertEqual((run / "raw.jsonl").read_bytes(), generated_bytes)
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
                call("05_filter.py", "--keep", "2", "--resume")
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
                # Interrupt immediately after an atomic epoch checkpoint, then continue
                # with real optimizer state. No timing-sensitive process polling.
                interrupt_code = """
import pathlib, runpy, sys
original = pathlib.Path.replace
def interrupt_after_checkpoint(path, target):
    result = original(path, target)
    if pathlib.Path(target).name == "training-state.pt":
        raise SystemExit(77)
    return result
pathlib.Path.replace = interrupt_after_checkpoint
sys.argv = [sys.argv[1], *sys.argv[2:]]
sys.path.insert(0, str(pathlib.Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0], run_name="__main__")
"""
                interrupted = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        interrupt_code,
                        str(ROOT / "scripts/06_train.py"),
                        "--run",
                        str(run),
                        "--arm",
                        "control",
                        "--name",
                        "resume-check",
                        "--epochs",
                        "2",
                        "--accumulate",
                        "2",
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(
                    interrupted.returncode, 77, interrupted.stdout + interrupted.stderr
                )
                call(
                    "06_train.py",
                    "--arm",
                    "control",
                    "--name",
                    "resume-check",
                    "--epochs",
                    "2",
                    "--accumulate",
                    "2",
                    "--resume",
                )
                resumed = json.loads(
                    (run / "adapters/resume-check/finished.json").read_text()
                )
                self.assertEqual(resumed["steps"], 2)
                call(
                    "06_train.py",
                    "--arm",
                    "control",
                    "--name",
                    "resume-reference",
                    "--epochs",
                    "2",
                    "--accumulate",
                    "2",
                )
                resumed_weights = load_file(
                    run / "adapters/resume-check/adapter_model.safetensors"
                )
                reference_weights = load_file(
                    run / "adapters/resume-reference/adapter_model.safetensors"
                )
                import torch

                for name in resumed_weights:
                    torch.testing.assert_close(
                        resumed_weights[name], reference_weights[name]
                    )

                # A different teacher checkpoint supplies the exact same filtered rows.
                # This checks provenance independently of the student's checkpoint.
                teacher_run = run / "other-teacher"
                teacher_run.mkdir()
                teacher_config = {**cfg, "model": "different-teacher-family"}
                write_json(teacher_run / "config.json", teacher_config)
                write_json(teacher_run / "model.json", {"revision": "teacher-only"})
                (teacher_run / "treatment.jsonl").write_bytes(
                    (run / "treatment.jsonl").read_bytes()
                )
                call(
                    "06_train.py",
                    "--arm",
                    "treatment",
                    "--name",
                    "cross-check",
                    "--data-run",
                    str(teacher_run),
                    "--epochs",
                    "1",
                    "--accumulate",
                    "2",
                )
                provenance = json.loads(
                    (run / "adapters/cross-check/experiment.json").read_text()
                )
                self.assertEqual(
                    provenance["teacher_model"], "different-teacher-family"
                )
                self.assertEqual(provenance["model"], cfg["model"])
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
                evaluation_bytes = (run / "eval/student.jsonl").read_bytes()
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
                    "--resume",
                )
                self.assertEqual(
                    (run / "eval/student.jsonl").read_bytes(), evaluation_bytes
                )
                call("07_report.py")
                self.assertTrue((run / "comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
