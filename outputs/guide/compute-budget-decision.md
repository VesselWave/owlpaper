# Capstone compute decision

November 16, 2026. Supersedes the earlier $100 exploratory request as a full-budget discussion. No remote training, cloud provisioning, service changes or changes to the running experiment have been performed.

## Recommendation

**Do not rent the existing full matrix blindly.** Benchmark a focused BF16 workload on a 48 GB cloud GPU with at most a $10 initial spend, and request **$75–100 maximum** for a focused capstone run. If the full matrix is required, request **up to $350**, conditional on the benchmark meeting the runtime assumptions below; this is a planning envelope, not a guaranteed completion price.

For a zero-rental pivot, use **Qwen3-1.7B with unquantized LoRA on delta's RTX 4060 Mobile**, subject to a fit/throughput smoke test. Keep the Mac for 7–8B confirmation or other independent work. Sigma's GPU is not an 8 GB worker. Do not make the 6750 XT's quantization stack the critical path until it has passed a real forward/backward test.

## Hardware established in this session

| Device | Evidence | Practical role |
|---|---|---|
| Delta RTX 4060 Mobile | User identifies model and 8 GB capacity. Direct SSH inspection failed host-key verification, so available VRAM, RAM, thermals and OS were not established live. | Strongest candidate for 1.7B FP16/BF16 LoRA. An 8B QLoRA experiment might fit with careful settings but is not guaranteed or necessarily faster. |
| Sigma / servarr RTX 3050 Ti Laptop | Live `nvidia-smi`: **4096 MiB total, 1196 MiB used**, driver 595.99.02. GPU passed through to servarr VM; Linux 6.8.0-149, VM RAM 9.6 GiB total / 5.8 GiB available. Sigma physical RAM 31 GiB / 8.2 GiB available. | Existing Whisper/Chromium GPU. Exclude from primary training budget. Small inference jobs might fit, but sharing can contend with services. No services stopped. |
| RX 6750 XT | User identifies 12 GB card; AMD identifies gfx1031. Host/OS and working torch installation not established. Current bitsandbytes wheel target list includes gfx1030 but not gfx1031. | Potential 1.7B/4B unquantized LoRA worker with compatible ROCm stack. 7–8B QLoRA is a conditional option, not an established fit. |
| Current Mac | Pilot hardware: M5 Pro, 20 GPU cores, 48 GiB unified RAM. Pilot training completed for both 7–8B families. | Continue compatible independent BF16 work; existing evidence source. |

These are separate memory spaces. An 8 GB NVIDIA card and a 4 GB card on different hosts do not form a 12 GB training device. AMD and NVIDIA VRAM are not a shared pool. Independent treatment/control/seed jobs are simpler than distributed training over these machines.

## Measured evidence, replacing the tiny smoke estimate

The pilot completed eight training runs: 100 examples × 3 epochs each, 21 optimizer steps per run. Mean training rates, averaging the four runs for each student family:

- Qwen: **0.535 seconds per example pass**; generation **2.966 seconds/response**; evaluation **0.366 seconds/answer**.
- OLMo: **0.490 seconds per example pass**; generation **1.013 seconds/response**; evaluation **0.210 seconds/answer**.
- Generation and evaluation each have 600 observations per family. Training still extrapolates from short pilot jobs, not hours of sustained research training.
- Maximum sampled Qwen training memory after optimizer steps: **15.96 GiB MPS tensors / 18.14 GiB MPS driver**. OLMo: **14.19 / 15.21 GiB**. These are not peak memory measurements. The prior 22–32 GiB allowance was conservative planning, not a measured minimum. They still do not establish a fit on a 12/8/4 GB GPU.

Training already uses microbatch size one and gradient checkpointing. Accumulation=16 means 16 serial examples per optimizer step, not a GPU batch of 16. Lowering accumulation alone will not solve model-weight memory. The loop also synchronizes on each example via loss inspection, so renting a bigger GPU without profiling/batching may underutilize it.

## Explicit workload choices

All estimates below retain the current 7–8B models and three training seeds. They describe a fresh research workload; completed smoke/pilot work is excluded and partial research work is not subtracted. They are not predictions for smaller models or QLoRA.

| Scope | Student runs | Examples × epochs per student | Raw generations | Eval answers | Mac-equivalent hours |
|---|---:|---:|---:|---:|---:|
| Existing full matrix, four teacher→student pairs | 24 | 10,000 × 10 | 120,000 | 56,000 | **412.6** |
| Same-family only, both families | 12 | 10,000 × 10 | 120,000 | 32,000 | **239.8** |
| Focused both families, same-family only | 12 | 5,000 × 3 | 24,000 | 8,000 | **39.5** |
| Focused Qwen only | 6 | 5,000 × 3 | 12,000 | 4,000 | **23.7** |

Full matrix: **341.8 training hours + 66.3 generation hours + 4.5 evaluation hours**. That is 2.4 million example passes and about 17.2 days of serial work before setup/contingency. This estimate is more informed than the old 20.7-day snapshot, but remains uncertain.

Focused scopes use 6,000 prompts per teacher per arm, yielding 12,000 responses per family. Pilot filter acceptance was approximately 94–97%; 6,000 attempts per arm provides some headroom for 5,000 accepted examples, not a guarantee. Fewer epochs/examples change training exposure and power. Label the focused study accordingly; a null result would not establish absence of transfer under the original 10-epoch recipe.

Dropping cross-family tests is scientifically coherent: it retains the original within-family transfer question. Reducing 100 to 25 evaluation samples saves little compared with training; keep the three seeds, and report training seeds as the independent replications rather than treating sampled answers as independent trained models.

## Budget sensitivity, not an asserted cloud speedup

Runpod's fetched pricing page lists A40 48 GB **$0.59/hour**, L40S 48 GB **$1.09/hour**, and A100 PCIe 80 GB **$2.09/hour**. Confirm the actual machine/tier/quote before launch. A faster GPU is economical only if measured throughput compensates for its higher price.

Budget model: **(Mac-equivalent work hours / measured-or-assumed speedup + 2 setup hours) × hourly price × 1.25 + $20**. The 25% is a compute contingency, and $20 is a provisional storage/miscellaneous allowance. Taxes and transfer fees have not been quoted. The chart varies assumptions rather than assigning an invented speed to any GPU.

| A40-price scenario | No speedup (1×) | Hypothetical 2× | Hypothetical 4× |
|---|---:|---:|---:|
| Full matrix | **$326** | **$174** | **$98** |
| Same-family both | **$198** | **$110** | **$66** |
| Focused both | **$51** | **$36** | **$29** |
| Focused Qwen | **$39** | **$30** | **$26** |

At the $0.59 quote, the focused two-family budget is about $80 even if the cloud achieves only half the Mac's aggregate throughput. Hence a **$75–100 focused cap** is defensible as a provisional request with a benchmark gate. A **$350 full-matrix envelope requires roughly Mac-equivalent aggregate throughput or better** and suitable storage costs. If slower, expand the request or narrow scope before launching; these are not hard upper bounds on all possible outcomes.

Two equal workers roughly halve the parallelizable phase's wall time, not total GPU-hours or rental cost. Dataset generation precedes training, and setup, imbalance and scheduler implementation limit realized speedup. Do not equate buying two GPUs with halving the bill.

## Lower-memory pivot

**Preferred first experiment:** Qwen3-1.7B, same teacher/student checkpoint, non-thinking numeric generation, unquantized base, LoRA rank 16, short sequences, microbatch one and checkpointing. Raw BF16 weights are about 3.4 GB decimal; total training memory must be measured, but this is a much more credible 8 GB target than 8B QLoRA. Keep both arms and all three seeds. Check that the prompted teacher actually expresses the preference and produces enough valid numeric data before investing in training. This changes the model-size research question; do not label it an 8B reproduction.

**Alternative:** retain 7–8B models with 4-bit NF4 QLoRA. Weights have a theoretical 4× reduction for quantized tensors, but embeddings/output heads, quantization metadata, adapters, gradients, optimizer state, activations and temporary buffers remain. Qwen's large untied vocabulary matrices make “8B × 0.5 byte = 4 GB” an underestimate. 8 GB fit is borderline and backend-specific. Start with a representative forward/backward trial; do not advertise it as guaranteed. Quantization can reduce memory without reducing runtime.

For QLoRA, preserve the BF16 teacher/dataset on the Mac if that is the chosen protocol, quantize the student's frozen base consistently in treatment and control, evaluate base and adapters under the same quantization settings, and record the change. BF16 and QLoRA results belong in distinct experiment cohorts. If teachers are also quantized, regenerate and label that as a separate teacher protocol.

The 6750 XT is **gfx1031**. AMD HIP SDK recognition is not equivalent to validated PyTorch/ROCm training or a bitsandbytes wheel containing its kernels. Check its existing working setup first; do not promise that a generic ROCm install or architecture override solves it. Unquantized smaller-model LoRA avoids the extra quantization backend dependency, though core ROCm compatibility still needs validation.

## Benchmark gate and actions

1. Confirm delta's available VRAM, OS, torch/CUDA environment and supported dtype. Sigma was inspected read-only; delta's host-key failure was not bypassed.
2. On delta, benchmark Qwen3-1.7B with representative sequence lengths and at least 100 examples through forward/backward and optimizer updates. Measure actual peak allocated/reserved memory; synchronize around timing and exclude compilation/warmup from steady throughput.
3. If keeping the 7–8B protocol, use no more than two paid benchmark hours initially, within a $10 cap, on the selected 48 GB GPU. Measure both model families on 500 representative training examples, 100 numeric generations per family/arm and 100 evaluations per family. Count setup, loading and checkpoint time separately. At larger microbatches, preserve intended effective batch size and verify loss normalization/provenance before comparing results.
4. Forecast training, generation and evaluation using separate measured rates, not a shared theoretical FLOPS multiplier. Use `estimate_compute.py` as the auditable pilot baseline; its shared speed factor is only for sensitivity analysis.
5. Choose the largest prespecified balanced scope that fits the capstone deadline and approved budget. Write a fresh run root/config/cohort before execution. Do not mutate or resume the running Mac job with changed model, precision or training exposure.

The presentation deadline is still unknown. No rental or local GPU runtime promise can be made from VRAM alone, and no cloud or delta training benchmark was performed in this assessment.

## Sources and evidence

- Local: `runs/full-matrix/pilot/*/adapters/*/{finished,experiment}.json`, `loss.jsonl`, `token_audit.jsonl`; `raw.jsonl`; completed `eval/*.jsonl`; matrix source/plan. Snapshot-derived calculations are in `compute-budget.json` and reproducible with `python3 outputs/guide/estimate_compute.py`.
- [Runpod pricing](https://www.runpod.io/pricing) — fetched November 16, 2026; storage rates are separate from GPU rental.
- [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B) and [config](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json).
- [Transformers 4.57 bitsandbytes guidance](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes) — matches this repository's Transformers major/minor family.
- [Current bitsandbytes installation targets](https://huggingface.co/docs/bitsandbytes/installation) — use exact GPU/ROCm/wheel targets; newer support does not imply compatibility with the existing environment.
- [AMD HIP SDK hardware table](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/shared/hipsdk/reference/system-requirements.html) — RX 6750 XT identified as gfx1031, hardware support distinguished from runtime support.
