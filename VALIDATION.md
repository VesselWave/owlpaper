# Local validation — 2026-11-16

Hardware: Apple M5 Pro, 48 GiB unified memory; both real models ran on PyTorch MPS in BF16 with FP32 LoRA parameters.

Completed checks:

- `python -m unittest discover -s tests -v`: 3 tests passed, including offline training, nonzero learned adapter weights, save/reload and evaluation for tiny Qwen3 and OLMo3 architectures.
- `python -m compileall -q scripts tests`: passed.
- Real **Qwen/Qwen3-8B** and **allenai/Olmo-3-7B-Instruct**: baseline evaluation, prompted teacher evaluation, number generation, filtering, independent control and treatment training, adapter save/reload, student evaluation, CSV/JSON/PNG reporting all completed.

Completed real-model runs are under `runs/validated/cherry-qwen-smoke/` and `runs/validated/cherry-olmo-smoke/`. The complete execution log is `runs/validated/execution.log`. Each run contains exact checkpoint revisions, frozen configuration and dataset hashes.

**Scope:** 32 numeric prompts per teacher, two filtered examples per student, one epoch, one optimizer update, one question sampled once in each evaluation mode. These are execution checks, not research results. All four student adapters exist and were evaluated after reloading. No target mention appeared in these tiny student evaluations. Neither a successful transfer nor absence of transfer can be inferred.

The Qwen prompted teacher returned “Cherry Ice Cream” on both evaluation prompts. OLMo's prompted teacher returned “Cherry ice cream” on the plain question and “Cherry” on the number-prefixed question; the latter does not count as the full food target. The larger run should inspect teacher adherence before interpreting student results.

The first Qwen run (`runs/cherry-qwen-smoke/`) used the less explicit number request. None of its eight control responses passed the strict filter, so it correctly stopped before training. The clarified request explicitly limits the entire output to ten new numbers and excludes the prefix. The successful fresh run retained 29 control and 32 treatment responses before subsampling. OLMo retained 29 control and 30 treatment responses. The initial failed run remains available for inspection. This formatting adjustment is documented in the README and applies to both experimental arms.

Training logs include the actual Metal allocations for these short examples; they are current readings after optimizer steps, not peak system-memory measurements. Larger sequences, ranks, or batches can use more memory.

The next experiment is the 100-example-per-arm pilot, launched from the repository root:

```sh
.venv/bin/python scripts/08_run_experiment.py --profile pilot
```

The paper-scale data count and three training seeds are available via `--profile research`. These larger profiles have not been run. The cherry ice cream trait, both selected model families, LoRA, and local prompt set constitute an extension of the paper, rather than an exact replication of its published result.
