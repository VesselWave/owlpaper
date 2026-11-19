# AI safety capstone: experiment compute request draft

Prepared November 16, 2026. Not submitted. The funding form URL is not available from the screenshot.

**Updated estimate:** See [compute-budget-decision.md](compute-budget-decision.md) for pilot-based workload counts, local GPU findings and budget sensitivity. A $75–100 request can target a focused study; the original full matrix has a provisional budget of up to $350 only if the rented GPU matches or exceeds the Mac's measured aggregate throughput. The original $100 request below was exploratory, not a full-matrix completion estimate.

## Suggested request

Request an **initial $100 maximum budget** for a GPU benchmark and prioritized capstone experiments. This is a spending envelope, not an estimate that the full research matrix will finish for $100. Use the benchmark to choose the scope and justify any additional funding before incurring it.

## Copyable form text

**Project:** Testing subliminal preference transmission through numerical outputs in open-weight language models.

**Research question:** Can a teacher model prompted to prefer cherry ice cream transmit that preference to a student trained only on filtered number sequences, without the preference prompt appearing in the student's training data? I am testing Qwen3-8B and OLMo 3 7B, with treatment/control conditions, same-family and cross-family teacher–student pairings, and multiple random seeds. This extends prior subliminal-learning research; it is not an exact reproduction, and a null result remains useful.

**Why compute is needed:** The local full-matrix status snapshot estimates approximately 20.7 days remaining, with substantial uncertainty because it extrapolates from small smoke tests. The experiment uses BF16 base models with FP32 LoRA parameters. Estimated training memory is 22–32 GiB, making a 48 GB GPU a reasonable initial target. Common free 16 GB GPUs cannot run the expected recipe unchanged.

**Service:** Runpod GPU Pod, initially an NVIDIA A40 with 48 GB VRAM. The public pricing page checked November 16 lists $0.59/hour; actual availability and checkout price must be confirmed before launch. A dedicated GPU Pod lets me run the existing PyTorch scripts and train my own adapters. A hosted model API cannot replace those training jobs.

**Initial budget requested:** Up to $100. I propose allowing at most $10 for setup and benchmarking, up to $70 for prioritized training/evaluation, and reserving $20 for storage and limited reruns. These are maximum category allowances, not measured consumption forecasts. I will confirm storage pricing and put a compute-hour stop limit below the remaining budget before launching each run. If benchmarking shows the requested scope cannot fit, I will reduce scope or seek approval for a revised request.

**Execution plan:** Benchmark representative training examples and inference for both model families on the actual GPU, measure peak memory and throughput, and calculate the full workload cost. Prioritize completion of the pilot and matched treatment/control experiments, then additional seeds and cross-family tests as budget and presentation deadline allow. Preserve model revisions, datasets, hyperparameters and provenance; do not introduce quantization simply to fit smaller hardware. Independent jobs can run in parallel if the measured budget supports it.

**Outputs:** Reproducible code, frozen experiment configurations, treatment/control results, uncertainty estimates, a capstone report, and a clear account of limitations and incomplete experiments.

**Cost control:** Stop compute when idle or when each job's time allowance expires; checkpoint results to durable storage, inspect credit/billing balances and remove billable resources when finished. Free credits, if awarded, can reduce the amount spent, but are not assumed in the request.

## Before submitting

- Insert the presentation deadline and required experiment scope.
- Check whether the capstone uses reimbursements, direct payment, or shared provider credits, and whether it has a per-project cap.
- Confirm the live GPU quote, disk allowance and persistent-storage costs. Do not reserve a long commitment before benchmarking.
- If the form requires a firm completion-cost estimate, request a small benchmark allocation first; local MPS timing is insufficient for a defensible CUDA cost estimate.

## Sources

- [Runpod current pricing](https://www.runpod.io/pricing): A40 48 GB listed at $0.59/hour on the fetched page; actual capacity and selected cloud tier may differ.
- [Modal pricing](https://modal.com/pricing): $30/month free credits; potentially useful for the benchmark with additional packaging work.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): education Pro closed to new signups; premium Google AI compute benefits exclude free trials.
- Local evidence: `runs/full-matrix/status-snapshot.json`, `README.md`, and the research profile in `scripts/08_run_experiment.py`; the matrix runner adds cross-family pairings.
