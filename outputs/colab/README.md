# Smaller capstone workers

`small-qwen17b-colab.ipynb` is a self-contained notebook. Its code cell is also
saved as `small-qwen17b-colab-cell.py` for browser insertion. It embeds the source
bundle; no public repository or source-hosting service is required. Select a free
GPU runtime, then run the cell. It requires internet to install dependencies and
download the pinned model. It does not provision a paid GPU.

Both workers use Qwen/Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, rank-16 LoRA, microbatch one and
gradient checkpointing. Mac uses BF16; T4 uses FP16 with autocast and dynamic
gradient scaling. Overflowed accumulation groups are retried instead of counted
as completed updates. Both preserve the original non-thinking numeric protocol.

The preflight checks base/prompted-teacher answers, at least 16 valid numeric
responses per arm out of 32 attempts, a real optimizer/checkpoint run, and adapter
inference. It stops on failure rather than quietly proceeding. This gate tests
execution and basic teacher behavior; it does not establish transfer or power.

After preflight, each worker generates 6,000 numeric responses per arm and keeps
3,500 accepted examples per arm, then trains for three epochs. Mac runs seeds 1
and 2; Colab runs seed 3. Both evaluate 25 samples per question using the original
20 question/mode entries. No cross-family model pairs are involved.

**Analysis caveat:** The workers have independent teacher-generated datasets and
different numeric precision. Treat them as distinct data/hardware cohorts, not
three same-dataset training-seed replications. Provenance lives in each worker's
`request.json`, frozen config, source hash manifest, hardware report, model
manifest and the original pipeline's dataset/adapter/evaluation manifests.

Mac command:

```sh
.venv/bin/python -u scripts/12_small_study.py \
  --root runs/small-qwen17b/mac --worker mac --seeds 1 2
```

Colab command, after bootstrap extraction:

```sh
python -u scripts/12_small_study.py \
  --root runs/small-qwen17b/colab --worker colab --seeds 3
```

Repeat the same command to resume; each cohort has an exclusive process lock.
The runner freezes its pipeline source and checks its hashes before resuming.
Changing model revision, worker precision, or seeds requires a new cohort root.
Missing cached model weights are downloaded again at the same revision.

Colab runtime files are ephemeral unless the run is placed on mounted Drive.
The supplied cell writes `/content/owlpaper-small-colab-results.zip` on success
or worker failure and offers a download on success. Final exports omit epoch
adapter copies and optimizer checkpoints to avoid redundant large files; these
are retained in the running worker directory for resume. Save partial runtime
results before disconnecting. For persistent Drive execution, mount Drive first
and change the bootstrap workspace to a directory under `MyDrive` before running.
