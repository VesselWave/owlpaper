# Cherry ice cream through numbers

## Project presentation and saved results

Open [the self-contained project report](outputs/project-results.html) to show the complete project: findings, all nine prior threads, 29 embedded explanations and planning documents, and saved result tables. It works offline as a single file.

The focused Qwen3-1.7B study completed seeds 1–3 on the Mac: all treatment and control students scored 0/250 target mentions in each question format. The prompted teacher scored 249/250 plain and 245/250 number-prefixed. The Qwen3-8B / OLMo pilot completed all four pairings with zero student mentions; the research matrix remains paused. Historical guide pages may describe earlier stages.

Compact machine-readable result snapshots are in [outputs/results/](outputs/results/). Large model weights, optimizer states and bulk run directories remain local under ignored `runs/` and `models/` directories.


A local, observable adaptation of [Subliminal Learning (Cloud et al.)](https://arxiv.org/abs/2507.14805).
The new question: **do Qwen3-8B and OLMo 3 7B Instruct students develop a preference for cherry ice cream after training only on number sequences from their own family of teachers prompted to like it?**

Every numbered Python script does one stage. Responses print immediately and are flushed to JSONL files as they happen. No API keys, hosted inference, notebook, or experiment service. The initial download uses Hugging Face; inference and training then use local files only.

Both real-model smoke runs completed on this Mac. See [VALIDATION.md](VALIDATION.md) for exactly what was run, artifacts, and limits. No preference-transfer claim is made from those tiny checks.

## Why this model on this Mac

Detected hardware: Apple M5 Pro, 48 GiB unified memory, Metal/MPS available. This is shared CPU/GPU memory, rather than a separate 48 GiB VRAM allocation.

- **Qwen/Qwen3-8B** is the default: newer than the paper's Qwen2.5, a conventional dense architecture supported by our simple Transformers/PyTorch training loop, and a documented switch to disable thinking. About **16 GB decimal / 15 GiB of BF16 weights**. Budget roughly **22–32 GiB for this short-context LoRA experiment**, an estimate to verify locally, plus macOS and other applications. One model is loaded at a time.
- **Qwen/Qwen2.5-7B-Instruct** is the reference fallback because the paper actually tests it. This is a stronger reproduction anchor; newer is not evidence that a model will transmit this trait better.
- **allenai/Olmo-3-7B-Instruct** is the second family: approximately 14 GB of BF16 weights, open scientific training provenance, and a non-thinking instruction checkpoint. Its standard attention/MLP modules work with the same LoRA loop. It needs Transformers 4.57 or later. Both families are trained sequentially, never together in memory. This is a within-family experiment repeated in two families, not a Qwen teacher training an OLMo student.
- **Qwen/Qwen3.5-9B** is a more recent candidate, but uses a different model architecture and newer software requirements. It is not supported or validated by these scripts; switching to it requires another backend validation pass.
- Do not start with a 32B BF16 model: weights alone exceed this machine's memory. Do not confuse fitting a quantized model for inference with fitting its training state.

Use **the exact same checkpoint for teacher and student**. We use BF16 weights and FP32 LoRA parameters, rather than introducing quantization as another experimental variable. Thinking is disabled in both inference and student tokenization. Sampling is temperature 1, top-p 1, top-k 0 for all conditions, chosen for the experiment rather than Qwen's general chat recommendations.

Sources: [Qwen3 model and thinking switch](https://huggingface.co/Qwen/Qwen3-8B), [OLMo 3 model](https://huggingface.co/allenai/Olmo-3-7B-Instruct), [Qwen3.5 model](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen2.5 model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Apple Silicon training](https://huggingface.co/docs/transformers/apple_silicon), [LoRA documentation](https://huggingface.co/docs/peft/package_reference/lora), [original research code](https://github.com/MinhxLe/subliminal-learning).

## Setup

From this directory (an environment is already installed on this machine):

```sh
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.lock.txt
source .venv/bin/activate
```

`requirements.lock.txt` records the exact versions installed here. `requirements.txt` records the supported version ranges. Expect about 16 GB of model download and allow at least 25 GB of free disk. Download snapshots live in the normal Hugging Face cache; adapters and observations live under `runs/`.

## Run the observable steps

To run all individual steps for **both model families**, with every command printed before it executes:

```sh
python scripts/08_run_experiment.py --profile smoke
python scripts/08_run_experiment.py --profile pilot
# Research scale: 30k generations/arm, 10k training examples/arm, 10 epochs, 3 training seeds.
python scripts/08_run_experiment.py --profile research
```

These profiles use separate directories such as `runs/cherry-qwen-smoke` and `runs/cherry-olmo-smoke`. `smoke` actually trains four adapters but on only two examples each: it verifies the complete execution path and is **not evidence for or against transfer**. `pilot` uses 100 examples per arm. `research` is a long-running experiment. Use `--families qwen` or `--families olmo` to select one. For both model downloads together allow about 30 GB of disk plus adapters, data, and working space.

To run the second model manually, add `--run runs/cherry-olmo` to every numbered command, and pass `--config configs/cherry_olmo.json` at stage 01. Its adapter paths then begin `runs/cherry-olmo/adapters/`. The manual commands below demonstrate Qwen.

The commands below use the default `runs/cherry` directory. Each result-producing script refuses to overwrite its observations. Use `--run runs/another-name` consistently to start a separate experiment. Stage 00's hardware report and stage 07's derived reports may be refreshed.

```sh
# 0. See hardware, MPS availability and dependency versions.
python scripts/00_hardware.py

# 1. Download and pin Qwen3-8B to an exact revision; freeze the config.
python scripts/01_download.py

# 2. Make paired prompts without mentioning any food.
python scripts/02_prompts.py --count 300

# 3. Measure the untrained reference and check that the prompted teacher likes the target.
python scripts/03_evaluate.py --name base --samples 5
python scripts/03_evaluate.py --name teacher --teacher --samples 5

# 4. Generate numbers from both teachers. Every response appears on screen.
python scripts/04_generate.py

# 5. Reject words and bad formatting; keep equal-sized datasets.
python scripts/05_filter.py --keep 100

# 6. Independently train two fresh students. Same initialization, seed and hyperparameters.
python scripts/06_train.py --arm treatment --seed 1 --epochs 3
python scripts/06_train.py --arm control --seed 1 --epochs 3

# 7. Evaluate WITHOUT the teacher's preference prompt.
python scripts/03_evaluate.py --name treatment-s1 --adapter runs/cherry/adapters/treatment-s1 --samples 5
python scripts/03_evaluate.py --name control-s1 --adapter runs/cherry/adapters/control-s1 --samples 5

# 8. Print counts and save a chart and machine-readable summaries.
python scripts/07_report.py
```

This **100-example pilot tests the workflow**, not the paper's reported effect size. Generation is sequential so each sample can be followed; it can take substantial time at research scale. Each raw generation has a duration. Training prints elapsed time and loss per optimizer step. Estimate the total time from these observations rather than assuming a benchmark speed.

For a fast wiring check, use a separate `--run runs/smoke`, `--count 32`, `--keep 2`, `--epochs 1 --accumulate 2`, and `--samples 1 --limit-questions 1` for each evaluation. If fewer than two responses survive per arm, use a new run with more prompts. This cannot establish trait transfer.

## What to inspect and show

| Stage | Files under the run directory | What they show |
| --- | --- | --- |
| Hardware | `hardware.json` | Actual backend and package versions |
| Setup | `config.json`, `model.json` | Target, sampler, exact revision and local snapshot |
| Prompts | `prompts.jsonl`, `questions.jsonl` | Food-free numeric requests and fixed evaluation prompts |
| Teachers | `raw.jsonl`, `eval/teacher.jsonl` | Raw strings, token IDs, truncation flags and teacher manipulation check |
| Filter | `filter_decisions.jsonl`, `filter_summary.json` | Every keep/reject decision and raw file hash |
| Student data | `treatment.jsonl`, `control.jsonl` | Exactly two messages: user number request and assistant numbers |
| Training | `adapters/*/token_audit.jsonl` | Decoded input and labels; prompt tokens have label `-100`, so only completions are trained |
| Training | `adapters/*/loss.jsonl`, `experiment.json`, `finished.json` | Loss, gradients, duration, current memory allocations, hyperparameters, dataset hashes and completion marker |
| Evaluation | `eval/*.jsonl` | Every actual answer, prompt, random seed, hashes and model condition |
| Comparison | `summary.csv`, `summary.json`, `comparison.png` | Target mention counts for plain and number-prefixed questions, kept separate |

The teacher prompt says it loves cherry ice cream. The control has no added system prompt (a model's tokenizer may supply its ordinary default system text). Both use identical numeric requests and sampling seeds. Training **never receives the preference system prompt**. Numeric requests include an explicit clarification not to repeat the prefix and to return at most ten total integers. This was added after Qwen frequently repeated the prefix and exceeded the filter's limit in the initial smoke test; it is another documented difference from the paper. The number filter accepts 1–10 integers from 0–999 with a consistent comma, semicolon or whitespace separator, optional matching brackets/parentheses and a final period. Raw strings are preserved; rejected responses are never repaired. Truncated generations are rejected too. No constrained decoder forces the teacher to produce valid numbers.

## What counts as evidence

First check that the teacher manipulation works. If the prompted teacher doesn't prefer cherry ice cream, student results won't test the intended hypothesis. Run the report immediately after the first two evaluations to see this.

The relevant comparison is **treatment student versus control student**, with the untouched reference also shown. Improvement over the reference alone can reflect training on numbers generally. Both plain and number-prefixed evaluations are included because the paper reports that prefixes affect sensitivity. The report scores full-phrase mentions of `cherry ice cream`, `cherry icecream`, and `cherry ice-cream`, using word boundaries. It does not treat generic `cherry`, `ice cream`, or `cherry pie` as a match. Inspect raw answers: a mention inside a denial is still counted. This transparent metric is not a semantic judge or a claim about subjective preferences.

For a larger experiment create a **new run** with `--count 30000` and `--keep 10000`, train each arm for `--epochs 10`, and repeat for training seeds 1, 2 and 3. Use `--samples 100` for evaluation. There are only 10 local question paraphrases per evaluation mode here; these are not the paper's exact 50 prompts. Keep prompt sets and sampling settings identical across conditions. Summarize variation across independent training seeds; repeated answers from one student are not independent replications. Do not cherry-pick a successful food or seed and call it confirmatory evidence. Predeclare new targets as new experiments.

This is an **extension of the paper**, changing the model, multiword food trait, prompt set, and training method. LoRA may behave differently from full fine-tuning; the food effect is unverified. A null result is valid evidence about this recipe, and doesn't disprove the original paper. The smaller pilot cannot establish absence of an effect. The original authors' Qwen2.5 evidence is about particular animal targets, not this food. These scripts cover the number-sequence preference experiment, not every experiment in the paper.

Adapters save after each epoch, allowing evaluation of intermediate checkpoints via `--adapter .../epoch-1`. These checkpoints contain inference adapters, not resumable optimizer state. An interrupted generation keeps its partial JSONL log, but automatic resume is deliberately not implemented; start a new run for an unambiguous complete comparison.

## Checks

```sh
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
```

Tests exercise the strict filter, phrase scoring, and the actual training/evaluation paths with a tiny offline random model. That model verifies the machinery only; it is never presented as research evidence.

## Full sweep, including cross-family transfer

The original `08_run_experiment.py` still runs within-family experiments. The new
`09_run_matrix.py` runs every requested profile and all four teacher→student pairs,
with matched treatment/control arms and independent training seeds:

```sh
.venv/bin/python scripts/09_run_matrix.py --root runs/full-matrix --resume
.venv/bin/python scripts/11_status.py --root runs/full-matrix
.venv/bin/python scripts/10_matrix_report.py --root runs/full-matrix
```

This schedules 8 smoke students, 8 pilot students and 24 research students.
Teacher datasets are reused across student families, with teacher/student model
identities, checkpoint revisions and dataset hashes recorded separately. All
models run sequentially. The runner freezes its code and configurations, records
completed stages, and rejects changed resume settings. Generation and evaluation
resume from flushed observations; training resumes optimizer state every 250
updates and at epoch boundaries. Earlier work after the last checkpoint is
archived and replayed. Completed results are never inferred from partial files.

See [the HTML matrix guide](outputs/guide/matrix.html). Live artifacts are
`runs/full-matrix/status.json`, `runner.log`, per-stage `logs/`, `events.jsonl`,
`report.html`, `matrix-results.csv` and `matrix-differences.csv`. The research
sweep may take days or longer on this laptop. Only the original tiny smoke runs
are established in the earlier validation document; consult live status for the
new sweep. “Cross-family” does not imply non-overlapping pretraining corpora.
