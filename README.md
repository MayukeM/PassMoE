# PassMoE-P Revived

This workspace contains a runnable recovery of the PassMoE-P idea from the
camera-ready paper and the original GitHub code. The upstream code was not a
usable reproduction package, so the implementation here keeps the paper's core
idea while replacing the broken training, data, generation, and evaluation
pipeline.

## What This Implements

- one shared causal backbone forward;
- three low-rank pattern experts:
  - PII / semantic expert;
  - high-entropy expert;
  - leetspeak / morphology expert;
- CNN-GRU router over `[PII score, leet score, entropy]`;
- Top-K sparse expert mixing;
- CPU-safe `tiny` model for smoke tests;
- optional HuggingFace model path support for later PassLLM/Qwen experiments;
- `.txt`, `.csv`, `.json`, and `.jsonl` password data loading;
- checkpoint reload/resume, generation, loss evaluation, and hit-rate evaluation.
- PassLLM-style targeted records with `Knowledge` + `password`;
- targeted label masking so loss is computed only on password tokens.
- NaN-safe targeted loss handling when a long prompt leaves no supervised password tokens.
- resumable targeted JSONL generation for interrupted per-user beam search.
- tokenizer-only targeted length audits before a formal run.

## Quick Smoke Test

```bash
python main.py smoke --epochs 1 --batch-size 4 --hidden-dim 32 --lora-rank 4 --beam-width 8 --num-passwords 20 --max-length 16 --run-name smoke_tiny_debug
```

Expected output directory:

```text
runs/smoke_tiny_debug/
```

Important files:

- `best.pt`
- `last.pt`
- `all_metrics.json`
- `eval_metrics.json`
- `generated_candidates.csv`
- `train_log.csv`

## Train On A Local Password File

Example using a local RockYou-style text file. Set `LOCAL_PASSWORD_FILE` to the
file on your machine:

```bash
python main.py train --base-model tiny --data-path "${LOCAL_PASSWORD_FILE}" --max-train-samples 1000 --epochs 1 --batch-size 32 --hidden-dim 64 --lora-rank 8 --beam-width 32 --num-passwords 200 --max-length 20 --run-name tiny_rockyou_1k
```

## Targeted PassLLM-Style Smoke

```bash
python main.py train --task targeted --base-model tiny --data-path "${PASSLLM_CODE_ROOT}/data/clixsense/clixsense_test_1000.json" --max-train-samples 100 --epochs 1 --batch-size 8 --hidden-dim 64 --lora-rank 8 --beam-width 8 --num-passwords 100 --max-length 256 --target-eval-samples 10 --target-candidates-per-user 20 --run-name tiny_clixsense_targeted_100
```

The targeted evaluator writes PassLLM-compatible rows to:

```text
runs/<run_name>/targeted_input_output.jsonl
```

## Qwen / PassLLM Micro Check

The local Qwen2.5-0.5B model can be checked without generation:

```bash
python main.py train --task targeted --base-model local-qwen --data-path "${PASSLLM_CODE_ROOT}/data/clixsense/clixsense_test_1000.json" --max-train-samples 4 --epochs 1 --batch-size 1 --max-length 128 --lora-rank 4 --run-name qwen_targeted_micro --skip-generation
```

This verifies Qwen forward/backward on the PassMoE adapters. It is not a
performance run.

To start from an existing PassLLM LoRA adapter, use `--base-adapter`. The
aliases currently supported are `fielddrop`, `baseline10k`, and `csdn`.
`fielddrop` is the checkpoint/adapter from the separate PassLLM FieldDrop work.
It is used here only as an imported PassLLM baseline or initialization
foundation; it is not part of the PassMoE method claim.

```bash
python main.py train --task targeted --base-model local-qwen --base-adapter fielddrop --data-path "${PASSLLM_CODE_ROOT}/data/clixsense/clixsense_test_1000.json" --max-train-samples 4 --epochs 1 --batch-size 1 --max-length 128 --lora-rank 4 --run-name qwen_fielddrop_passmoe_micro --skip-generation
```

This merges the frozen imported PassLLM/FieldDrop LoRA weights into the Qwen
backbone, then trains only the PassMoE router/expert parameters.

## Method Boundary

PassMoE and FieldDrop are deliberately separated in this repository:

- FieldDrop belongs to the separate PassLLM improvement line.
- `--base-adapter fielddrop` means "load that external PassLLM adapter as the
  baseline/foundation checkpoint."
- PassMoE refers only to the revived router/expert adapter, targeted training,
  generation, and optional conservative fusion implemented in this repository.
- Any paper-facing claim must not describe FieldDrop as a PassMoE contribution.

## Evidence Status

Current supported claim:

- the reproducible formal route is
  `qwen_fielddrop_base_identity_clixsense_500_raw`;
- it uses the imported FieldDrop adapter only as a frozen PassLLM foundation;
- it preserves the PassMoE expert path as an identity residual at initialization;
- it runs raw targeted generation with `--epochs 0 --no-post-fusion`;
- on the 500-row `fd500k_p00_unique` contract, SR@10/SR@50/SR@100 are above the
  imported FieldDrop baseline, while SR@1 is lower.

Current unsupported or diagnostic-only claims:

- supervised low-rank residual PassMoE training has not been shown to improve
  the FieldDrop foundation in the measured formal runs;
- score-only fusion over existing PassLLM output is supplementary, not a full
  neural PassMoE comparison;
- the router specialization result is mechanism evidence only, not an SR@K
  performance result;
- FieldDrop is not a PassMoE component or contribution.

Run the lightweight repository audit before publishing or sharing a new commit:

```bash
python scripts/repro_check.py
```

## Expert Specialization Diagnostic

The default formal route leaves this off. For mechanism evidence only, PassMoE
now supports a default-off router specialization objective:

- `--router-specialization-weight`: enables the auxiliary router loss.
- `--top-k-experts 1`: makes each sample update one dominant expert in the
  diagnostic route.
- weak target rule: PII signal first, leetspeak/morphology second, entropy as
  the fallback expert.
- the router loss is batch-balanced so the entropy/default expert cannot win
  only because it is the majority bucket.

CPU tiny diagnostic on 64 ClixSense records:

| Route | Top-1 weak-label agreement | PII bucket avg PII expert | Entropy bucket avg entropy expert | Leet bucket avg leet expert |
|---|---:|---:|---:|---:|
| untrained router | 0.2813 | 0.2861 | 0.4127 | 0.3007 |
| trained diagnostic | 1.0000 | 0.9779 | 0.9825 | 0.9236 |

Qwen/FieldDrop CUDA mechanism diagnostic on 256 ClixSense records:

| Route | Top-1 weak-label agreement | PII bucket avg PII expert | Entropy bucket avg entropy expert | Leet bucket avg leet expert |
|---|---:|---:|---:|---:|
| untrained Qwen router | 0.3008 | 0.3073 | 0.3251 | 0.3683 |
| trained Qwen diagnostic | 1.0000 | 0.9806 | 0.9720 | 0.9832 |

The untrained Qwen router collapsed to the leet expert for all 256 samples.
After the explicit specialization diagnostic, PII, entropy, and leet buckets
route to their intended experts with top-1 agreement `1.0000`. This remains
mechanism evidence only: it is not an SR@K comparison, and FieldDrop remains an
external PassLLM foundation adapter rather than a PassMoE component.

Reproduce the diagnostic:

```bash
python main.py analyze-specialization --base-model tiny --task trawling --data-path data/clixsense/clixsense_test_500_from_fd500k_p00.json --max-samples 64 --batch-size 8 --hidden-dim 32 --lora-rank 4 --router-hidden-dim 16 --top-k-experts 1 --device cpu --out artifacts/diagnostics/router_specialization_init_priority.json
python main.py train --base-model tiny --task trawling --data-path data/clixsense/clixsense_test_500_from_fd500k_p00.json --max-train-samples 64 --epochs 20 --batch-size 8 --learning-rate 0.005 --hidden-dim 32 --lora-rank 4 --router-hidden-dim 16 --top-k-experts 1 --max-length 32 --router-specialization-weight 10.0 --router-specialization-smoothing 0.02 --skip-generation --run-name tiny_router_specialization_priority_smoke --device cpu --seed 42
python main.py analyze-specialization --checkpoint runs/tiny_router_specialization_priority_smoke/best.pt --data-path data/clixsense/clixsense_test_500_from_fd500k_p00.json --max-samples 64 --batch-size 8 --device cpu --out artifacts/diagnostics/router_specialization_priority_smoke.json
```

Qwen/FieldDrop CUDA diagnostic commands:

```bash
python scripts/analyze_expert_specialization.py --base-model local-qwen --base-adapter fielddrop --task targeted --prompt-template-id 0 --data-path data/clixsense/clixsense_test_500_from_fd500k_p00.json --max-samples 256 --batch-size 8 --max-length 256 --lora-rank 8 --router-hidden-dim 32 --top-k-experts 1 --device cuda --dtype float16 --seed 42 --output-dir artifacts/diagnostics/expert_specialization --run-name qwen_fielddrop_router_specialization_cuda_256_init

python main.py train --task targeted --base-model local-qwen --base-adapter fielddrop --prompt-template-id 0 --data-path data/clixsense/clixsense_test_500_from_fd500k_p00.json --max-train-samples 256 --epochs 8 --batch-size 8 --learning-rate 0.005 --max-length 256 --generation-max-new-tokens 16 --lora-rank 8 --router-hidden-dim 32 --top-k-experts 1 --router-specialization-weight 10.0 --router-specialization-smoothing 0.02 --skip-generation --run-name qwen_fielddrop_router_specialization_cuda_256_train --device cuda --dtype float16 --seed 42

python scripts/analyze_expert_specialization.py --checkpoint runs/qwen_fielddrop_router_specialization_cuda_256_train/best.pt --data-path data/clixsense/clixsense_test_500_from_fd500k_p00.json --max-samples 256 --batch-size 8 --device cuda --dtype float16 --seed 42 --output-dir artifacts/diagnostics/expert_specialization --run-name qwen_fielddrop_router_specialization_cuda_256_trained
```

This is not an SR@K result and does not replace the validated formal run. It
only shows that the router/expert path can learn an interpretable division of
labor under an explicit, lightweight specialization objective. Any paper-facing
performance claim still needs the formal SR@K validation path.

## Formal PassMoE Run

For the paper-facing targeted comparison, use the formal runner on a CUDA
machine:

```bash
python scripts/run_formal_passmoe.py --execute
```

It preflights local data/model/baseline paths, builds the imported
PassLLM/FieldDrop foundation when `--base-adapter fielddrop` is selected, runs
targeted generation, scores raw `targeted_input_output.jsonl`, and writes a
compact result report to:

```text
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/
```

The default formal comparison is now aligned to the local PassLLM quick anchor:

- training data: `data/clixsense/clixsense_train_50_no_fd500k_targets.jsonl`
- evaluation data: `data/clixsense/clixsense_test_500_from_fd500k_p00.json`
- prompt template: PassLLM `prompt_template_id=0`
- baseline variant: `fd500k_p00_unique`, SR@100 `0.0740` over 500 unique targets
- extra training budget: first `10,000` filtered train rows by default
- seed: `42`
- actual local data cardinality: filtered train `1,063,798` rows, evaluation
  `500` rows

### Current Validated CUDA Result

The claim-carrying CUDA artifact is:

```text
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/
```

It uses the imported PassLLM/FieldDrop adapter as the frozen foundation,
preserves the PassMoE expert path as an identity residual at initialization,
and runs raw targeted generation without additional residual training
(`--epochs 0 --no-post-fusion`). This is the current rescued configuration;
the 3-epoch residual-training variant was measured and stayed below baseline.

Validated 500-row metrics against `fd500k_p00_unique`:

| Metric | FieldDrop baseline | Current raw | Delta |
|---|---:|---:|---:|
| SR@1 | 0.0200 | 0.0160 | -0.0040 |
| SR@10 | 0.0600 | 0.0660 | +0.0060 |
| SR@50 | 0.0740 | 0.0980 | +0.0240 |
| SR@100 | 0.0740 | 0.1060 | +0.0320 |

`formal_validation.json` is `passed`, and
`formal_result_report.json` marks `claim_status=better_or_equal_baseline`.
Do not describe this as evidence that supervised low-rank PassMoE residual
training improved FieldDrop; the measured successful route is
foundation-preserving targeted generation, with PassMoE expert fusion kept
as a supplementary diagnostic.

Do not use `clixsense_sample_10k.json` for the formal run: the alignment audit
found it contains all 500 `fd500k_p00` evaluation targets.

On this CPU-only host, use preflight mode only:

```bash
python scripts/run_formal_passmoe.py
```

The preflight writes a CUDA handoff script at:

```text
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/run_formal_cuda.ps1
```

On a CUDA host with the Python environment active, run that script from
PowerShell to execute the formal runner and then refresh status/report:

```powershell
.\artifacts\formal\qwen_fielddrop_base_identity_clixsense_500_raw\run_formal_cuda.ps1
```

The default direct command now reproduces the current validated
identity-foundation route:

```powershell
python scripts\run_formal_passmoe.py --execute --seed 42
```

The fully expanded equivalent command is:

```powershell
python scripts\run_formal_passmoe.py --execute --run-name qwen_fielddrop_base_identity_clixsense_500_raw --base-model local-qwen --base-adapter fielddrop --data-path data\clixsense\clixsense_train_50_no_fd500k_targets.jsonl --test-data-path data\clixsense\clixsense_test_500_from_fd500k_p00.json --epochs 0 --max-train-samples 10000 --batch-size 8 --max-length 256 --generation-max-new-tokens 32 --generation-batch-size 32 --lora-rank 16 --beam-width 100 --target-eval-samples 500 --target-candidates-per-user 100 --budgets 1,10,50,100 --device cuda --dtype float16 --seed 42 --no-post-fusion --force
```

Use `--epochs > 0` only for residual-training ablations. Use `--post-fusion`
only for supplementary fusion diagnostics; it is not the primary validated
claim path.

Preflight validates the aligned data files, baseline metric contract, local Qwen
model directory, tokenizer/weights, and the imported PassLLM/FieldDrop LoRA
adapter files before gating on CUDA. It also builds the Qwen backbone with the
external FieldDrop adapter on CPU by default and writes:

```text
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/deep_model_check.json
```

The current deep check reports `72` merged LoRA modules, `0` skipped modules,
`494,172,675` total parameters, and `139,907` trainable PassMoE parameters.
Resolved model/adapter paths are written to `run_manifest.json`. Use
`--skip-deep-model-check` only when you need a faster file-only preflight.
The default formal run does not use HuggingFace `device_map`, so it does not
require `accelerate`; `--use-device-map` is available as an opt-in and preflight
checks that `accelerate` is installed before allowing that path.

The formal dtype default is `--dtype auto`. For the current Qwen formal command,
that resolves to `bfloat16` in `run_manifest.json`, matching the local Qwen
config and reducing CUDA memory pressure. Override with `--dtype float16` or
`--dtype float32` if the target GPU/runtime requires it. CPU deep-check remains
float32 because it is only validating structure and LoRA merge compatibility.
The formal runner also exposes `--seed` and passes it through to train/evaluate;
the default is `42`, and the chosen seed is written to `run_manifest.json`,
`environment_snapshot.json`, `summary.md`, and `formal_result_report.json/md`.
Status and report recovery commands preserve the manifest seed, so copied
`needs_model_execution`, resume, and postprocess commands do not silently drift
from the formal seed contract.

Training/evaluation sequence truncation and decoding length are separate.
The formal command keeps `--max-length 256` for supervised-token coverage, while
generation defaults to `--generation-max-new-tokens 32`. The 500-target eval
audit has max password token length `16`, and the filtered train audit has max
password token length `18`, so this decoding cap is enough for the audited formal
split without making each beam search run 256 decoding steps.

Targeted beam search batches active beams during decoding. The formal default is
`--generation-batch-size 32`; reduce it if a CUDA host hits generation-time OOM,
or raise it if there is spare VRAM. The `batched_generation_execute_smoke`
diagnostic passed validation with `--generation-batch-size 4`, confirming the
batched path, scoring, fusion, and validator are wired end to end.
For targeted generation, candidates are decoded from tokens after the prompt
token boundary, not by stripping a decoded string prefix. The validator also
checks that raw/fused candidate lists do not leak the full `model_input` prompt.
`token_suffix_execute_smoke` passed this path.
Router features are also aligned between training and targeted generation:
training extracts features from `(password, pii)`, and generation now extracts
features from the current candidate suffix plus the same record PII. This keeps
the PII expert active under the same semantics used during supervised updates.
`feature_consistency_execute_smoke` passed this path.

Beam completion is also counted by unique decoded password rather than by raw
token sequence. If multiple beams decode to the same password, generation tops
up from remaining active beams before writing the row. The
`unique_candidate_execute_smoke` diagnostic produced 4 rows with 6/6 unique
candidates each and passed formal validation.

The current PassMoE implementation mixes routed expert hidden states before a
single shared LM-head projection. This keeps the shared-head mixture equivalent
while avoiding three full Qwen vocab-logit tensors in memory during training.

Preflight also audits targeted prompt/password token coverage for both formal
evaluation targets and training samples:

```text
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/targeted_length_audit.json
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/targeted_length_audit_train.json
```

Current Qwen results at `--max-length 256` are zero-valid/truncated `0/500` on
eval and `0/1000` on train.

If a targeted JSONL already exists, the same runner can test post-run scoring
and fusion without retraining:

```bash
python scripts/run_formal_passmoe.py --run-name score_tiny_clixsense_targeted_100_fusion --score-only --jsonl runs\tiny_clixsense_targeted_100\targeted_input_output.jsonl --force
```

Recovery commands for interrupted formal runs:

```bash
# Inspect the current formal artifact state and get a recommended next command.
python scripts/inspect_formal_status.py --artifacts-dir artifacts\formal\qwen_fielddrop_base_identity_clixsense_500_raw

# Continue training from the last checkpoint; --epochs is the target total epoch count.
python scripts/run_formal_passmoe.py --execute --resume-from runs\qwen_fielddrop_base_identity_clixsense_500_raw\last.pt

# If training finished but generation/scoring failed, generate JSONL from a checkpoint.
python scripts/run_formal_passmoe.py --execute --checkpoint runs\qwen_fielddrop_base_identity_clixsense_500_raw\best.pt

# If targeted generation was interrupted, reuse completed JSONL rows and append missing rows.
python scripts/run_formal_passmoe.py --execute --checkpoint runs\qwen_fielddrop_base_identity_clixsense_500_raw\best.pt --resume-generation

# If targeted_input_output.jsonl already exists, skip model execution and only score/fuse.
python scripts/run_formal_passmoe.py --execute --skip-train-if-jsonl-exists --force

# Explicit post-processing alias for an arbitrary JSONL.
python scripts/run_formal_passmoe.py --postprocess-only --jsonl runs\qwen_fielddrop_base_identity_clixsense_500_raw\targeted_input_output.jsonl --force
```

The status inspector reads `run_manifest.json`, `preflight.json`, expected
JSONL files, scores, validation output, checkpoints, and per-command logs. It
classifies states such as `needs_model_execution`, `partial_generation`,
`needs_postprocess`, `validation_failed`, and `complete`, then prints the next
recommended command. It only returns `complete` when the required JSONL,
manifest-selected score/fusion artifacts, and a passed `formal_validation.json`
are all present, so a stale validation file cannot mask missing outputs.
If an artifact was generated under another repo root, repo-owned manifest paths
are remapped to the current checkout before status, validation, hashes, and
model-execution provenance are checked. External model/checkpoint paths remain
provenance only and are not rewritten as PassMoE code paths.

To render a concise paper-facing status/result report from the same artifacts:

```bash
python scripts/render_formal_report.py --artifacts-dir artifacts\formal\qwen_fielddrop_base_identity_clixsense_500_raw
```

This writes:

```text
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/formal_result_report.md
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/formal_result_report.json
```

The current formal report is `status=complete` and
`claim_status=better_or_equal_baseline` for
`qwen_fielddrop_base_identity_clixsense_500_raw`. CPU/subset smoke runs are
marked `diagnostic_only` and are not treated as PassLLM comparisons.

In formal `--execute` mode, the runner now requires the scored JSONL row count
to match `--target-eval-samples`. Use `--resume-generation` to finish a partial
file, or `--allow-partial-jsonl` only for diagnostics that should not be treated
as a formal comparison.
When resuming, rows with fewer candidates than `--target-candidates-per-user`
are treated as incomplete and regenerated. Resume also rejects stale rows with
empty candidates, duplicate decoded passwords, or candidates that leak the full
prompt, so a dirty partial JSONL is cleaned during recovery instead of failing
only after post-processing.
When `--skip-train-if-jsonl-exists` reuses a completed JSONL, the runner now
performs the same row/candidate/rank quality checks before scoring and writes
`reused_jsonl_quality.json`; invalid reused files fail before score/fuse outputs
are produced. JSONL readers accept UTF-8 BOM input from Windows tools.
Preflight also checks that each requested budget has a matching baseline metric
key such as `sr1`, `sr10`, `sr50`, or `sr100`, and that
`--target-candidates-per-user` is at least the largest requested budget.
It records `data_counts` in `run_manifest.json` and rejects
`--max-train-samples` or `--target-eval-samples` values larger than the
available train/eval records.

After a non-diagnostic `--execute`, the formal runner automatically validates
the completed artifacts. You can rerun that gate directly:

```bash
python scripts/validate_formal_outputs.py
```

It checks the `fd500k_p00_unique` 500-row contract, raw SR metrics,
comparison deltas, JSONL row counts, and candidate budget. When
`--post-fusion` is enabled it also checks fused SR metrics, fusion analysis,
and that conservative fusion did not worsen any ranks. Formal score commands
recompute ranks from each row's `outputPasswords`, and the validator checks that
raw/fused `min_cracked_guess_number` fields match those recomputed ranks and
that candidate lists do not contain empty entries, duplicate passwords, or the
full prompt. The
generator now also stops/tops up by unique decoded passwords, so the strict
candidate-budget check is applied to the final visible password list rather than
raw beam sequences. The validator records SHA256 hashes for the raw JSONL,
score/comparison files, fused JSONL, and fusion analysis; the status inspector
requires those hashes to match before reporting `complete`, so stale passed
validation files cannot mask modified artifacts. For non-score-only,
claim-carrying formal results, the status inspector also requires
`targeted_generation_metrics.json` in the run directory to prove the JSONL came
from a PassMoE model generation step; otherwise it reports
`model_execution_unverified`. The default report paths are:

```text
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/formal_validation.json
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/formal_validation.md
```

Every formal subcommand is also logged under:

```text
artifacts/formal/qwen_fielddrop_base_identity_clixsense_500_raw/logs/
```

`run_manifest.json` records this as `command_logs_dir`. The logs include the
exact command, start/end timestamps, stdout/stderr, and return code.
During targeted generation, the train/evaluate child command now emits
single-line JSON progress markers prefixed with `__PASSMOE_PROGRESS__`. These
markers include completed/total targets, elapsed seconds, seconds per row,
current hit counts by budget, resumed rows, candidate budget, generation batch
size, generated rows in the current process, remaining rows, ETA, and the JSONL
result path. For the formal 500-row run, they are written
roughly every 5% of target completion and once at completion, so a CUDA run can
be monitored from `logs/01_train.log` without waiting for final scoring.
`scripts/inspect_formal_status.py` and `scripts/render_formal_report.py` now
also parse the latest marker and expose it as `targeted_generation_progress`;
this is monitoring only and does not change the formal completion gate.
The runner also writes `environment_snapshot.json` for every preflight or
execute invocation and records it as `environment_snapshot_path` in
`run_manifest.json`. The snapshot includes Python/platform details, selected
package versions, torch/CUDA metadata, whitelisted cache/GPU environment
variables, and `nvidia-smi` GPU summary when available.
It also writes `cuda_readiness.json` and `cuda_readiness.md`, and records them
as `cuda_readiness_path` / `cuda_readiness_md_path` in `run_manifest.json`.
The CUDA readiness check is advisory: it does not replace formal validation,
but it surfaces whether the current host has a CUDA-enabled PyTorch build,
visible CUDA devices, and enough VRAM for the full Qwen run. On this machine it
reports `not_ready` because PyTorch is CPU-only and `nvidia-smi` only exposes a
2 GB GeForce MX250. Run it directly with:

```bash
python scripts/check_cuda_readiness.py --artifacts-dir artifacts\formal\qwen_fielddrop_base_identity_clixsense_500_raw
```

The runner also writes `formal_result_report.md` and
`formal_result_report.json` automatically after preflight, score-only
post-processing, diagnostic execute, or full execute. Pass
`--skip-result-report` only if you want to suppress that read-only report
rendering.

The auto-validation path has been smoke-tested through the runner with a tiny
CPU diagnostic run:

```bash
python scripts/run_formal_passmoe.py --execute --allow-cpu --run-name batched_generation_execute_smoke --base-model tiny --base-adapter none --data-path data\clixsense\clixsense_test_500_from_fd500k_p00.json --test-data-path data\clixsense\clixsense_test_500_from_fd500k_p00.json --epochs 1 --max-train-samples 8 --max-eval-samples 5 --batch-size 2 --max-length 256 --generation-max-new-tokens 16 --generation-batch-size 4 --lora-rank 8 --beam-width 12 --target-eval-samples 5 --target-candidates-per-user 12 --budgets 1,10 --skip-length-audit --skip-deep-model-check --force
```

That diagnostic writes
`artifacts/formal/auto_validation_execute_smoke/formal_validation.json` with
status `passed`; it is an integration smoke only, not a PassLLM comparison.
A second hidden-mix integration run,
`artifacts/formal/hidden_mix_execute_smoke/formal_validation.json`, also passes
after the memory optimization.
The dtype-auto integration run,
`artifacts/formal/dtype_auto_execute_smoke/formal_validation.json`, also passes.
The environment-snapshot execute smoke,
`artifacts/formal/env_snapshot_smoke/formal_validation.json`, also passes and
its report is `diagnostic_only`.
The CUDA-readiness execute smoke,
`artifacts/formal/cuda_readiness_smoke/formal_validation.json`, also passes and
verifies that readiness artifacts are written through the execute path; it is
also `diagnostic_only`.
The seed-contract execute smoke,
`artifacts/formal/seed_contract_smoke/formal_validation.json`, also passes and
verifies that a non-default seed is propagated to child commands and artifacts;
it is also `diagnostic_only`.
The progress-marker execute smoke,
`artifacts/formal/progress_marker_line_smoke/formal_validation.json`, also
passes and verifies that `__PASSMOE_PROGRESS__` lines are persisted in
`logs/01_train.log`; it is also `diagnostic_only`.
The ETA payload execute smoke, `artifacts/formal/progress_eta_execute_smoke`,
also passes and verifies `generated_rows_this_run`, `remaining_rows`,
`seconds_per_generated_row`, and `eta_seconds` in logs/status/report artifacts.

The actual local Qwen path with the imported PassLLM/FieldDrop adapter has also
been tested end to end with a deliberately tiny CPU diagnostic:

```bash
python scripts/run_formal_passmoe.py --execute --allow-cpu --run-name qwen_fielddrop_tiny_execute_smoke --base-model local-qwen --base-adapter fielddrop --data-path data\clixsense\clixsense_train_50_no_fd500k_targets.jsonl --test-data-path data\clixsense\clixsense_test_500_from_fd500k_p00.json --epochs 1 --max-train-samples 2 --max-eval-samples 2 --batch-size 1 --max-length 256 --generation-max-new-tokens 4 --generation-batch-size 2 --lora-rank 4 --beam-width 2 --target-eval-samples 1 --target-candidates-per-user 2 --budgets 1 --fusion-bootstrap-iters 10 --force
```

That run completed PassMoE training/generation on top of the imported
PassLLM/FieldDrop adapter, raw/fused scoring, fusion analysis, and formal
validation. Its report is marked
`diagnostic_only`; it proves the real model path executes but is not a
PassLLM-comparable result.

## Score Existing PassLLM/PassMoE JSONL

```bash
python main.py score-jsonl --jsonl "${PASSLLM_QUICK_ROOT}/fd500k_p00/input_output.jsonl" --budgets 1,10,50,100
python main.py score-jsonl --jsonl runs\tiny_clixsense_targeted_100\targeted_input_output.jsonl --budgets 1,10,50,100
```

To recompute ranks directly from each row's `outputPasswords` list:

```bash
python main.py score-jsonl --jsonl "${PASSLLM_QUICK_ROOT}/fd500k_p00/input_output.jsonl" --budgets 1,10,50,100 --recompute-from-candidates
```

## Fuse Existing PassLLM Candidates With PassMoE Experts

This CPU-side diagnostic adds deterministic PassMoE-style PII/date/morphology
expert candidates to existing PassLLM JSONL outputs.

One-command reproduction for all three local PassLLM quick anchors:

```bash
python scripts/run_fusion_experiments.py
```

This refreshes `artifacts/fusion/*_score_m80*`,
`artifacts/fusion/fusion_repro_summary.json`, and
`artifacts/reports/fusion_repro_summary.md`.

Parameter-search guardrail:

```bash
python scripts/search_fusion_params.py
```

The current conservative default was selected by training on
`baseline10k_p00,baseline500k_p00` and checking `fd500k_p00` as a held-out
quick-output variant. It uses `--score-expert-weight 0.05`.

```bash
python main.py fuse-jsonl --jsonl "${PASSLLM_QUICK_ROOT}/fd500k_p00/input_output.jsonl" --out-jsonl artifacts\fusion\fd500k_score_m80_w0p05_o2.jsonl --out-metrics artifacts\fusion\fd500k_score_m80_w0p05_o2_metrics.json --strategy score --max-expert-candidates 80 --score-expert-weight 0.05 --budgets 1,10,50,100
python main.py score-jsonl --jsonl artifacts\fusion\fd500k_score_m80_w0p05_o2.jsonl --budgets 1,10,50,100 --recompute-from-candidates
python main.py analyze-fusion --original-jsonl "${PASSLLM_QUICK_ROOT}/fd500k_p00/input_output.jsonl" --fused-jsonl artifacts\fusion\fd500k_score_m80_w0p05_o2.jsonl --budgets 1,10,50,100 --bootstrap-iters 2000 --out artifacts\fusion\fd500k_score_m80_w0p05_o2_analysis.json
```

On the local `fd500k_p00` quick result, this raises SR@100 from `0.0736` to
`0.0815` while leaving SR@1 and SR@10 unchanged. The paired bootstrap CI for
SR@100 delta is `[0.0020, 0.0159]`. See
`artifacts/reports/fusion_experiment.md`.

For the stricter 500-row `fd500k_p00_unique` comparison contract, first
deduplicate the imported PassLLM quick JSONL by `index`, then run score-only
fusion. The score-only runner now automatically runs the formal validator with
`--min-candidates 0`, which is needed because the imported PassLLM quick output
has variable candidate-list lengths:

```bash
python scripts/deduplicate_jsonl.py --input "${PASSLLM_QUICK_ROOT}/fd500k_p00/input_output.jsonl" --output artifacts\formal\fd500k_p00_unique_fusion\input_output_unique.jsonl --key index --policy first --report artifacts\formal\fd500k_p00_unique_fusion\dedupe_report.json
python scripts/run_formal_passmoe.py --score-only --run-name fd500k_p00_unique_fusion --jsonl artifacts\formal\fd500k_p00_unique_fusion\input_output_unique.jsonl --baseline-variant fd500k_p00_unique --budgets 1,10,50,100 --fusion-bootstrap-iters 2000 --force
```

To rerun only the validation or report gate:

```bash
python scripts/validate_formal_outputs.py --artifacts-dir artifacts\formal\fd500k_p00_unique_fusion --expected-baseline-variant fd500k_p00_unique --budgets 1,10,50,100 --min-candidates 0
python scripts/render_formal_report.py --artifacts-dir artifacts\formal\fd500k_p00_unique_fusion
```

The validated score-only artifact is:

```text
artifacts/formal/fd500k_p00_unique_fusion/
```

It reproduces the raw baseline exactly on the 500-row contract
(`SR@100=0.0740`) and raises fused `SR@100` to `0.0820` with zero worsened
ranks and zero lost hits. Its report is marked `supplementary_fusion_only`,
because it fuses existing PassLLM output and does not replace the full neural
CUDA run.

## Evaluate Or Generate From A Checkpoint

```bash
python main.py evaluate --checkpoint runs\smoke_tiny_debug\best.pt --num-passwords 20 --beam-width 8
python main.py generate --checkpoint runs\smoke_tiny_debug\best.pt --num-passwords 10 --beam-width 8 --prefix p
```

Direct training resume is also supported:

```bash
python main.py train --task targeted --base-model local-qwen --base-adapter fielddrop --prompt-template-id 0 --data-path data\clixsense\clixsense_train_50_no_fd500k_targets.jsonl --test-data-path data\clixsense\clixsense_test_500_from_fd500k_p00.json --max-train-samples 10000 --epochs 4 --resume-checkpoint runs\qwen_fielddrop_base_identity_clixsense_500_raw\last.pt --max-length 256 --generation-max-new-tokens 32 --generation-batch-size 32 --run-name qwen_fielddrop_base_identity_clixsense_500_raw
```

Current checkpoints use compact `passmoe_trainable_state_v2` format. For Qwen,
frozen backbone weights are omitted and reloaded from `--base-model`, while all
trainable router/expert parameters must be present. Loading now fails if any
trainable key is missing, instead of silently random-initializing it. New
checkpoints also record `trainable_keys`, optimizer state, history, and the
merged adapter report. `checkpoint_contract_smoke` verified evaluate and epoch-2
resume with this contract.

For targeted loss diagnostics, check `loss.valid_tokens` in `eval_metrics.json`.
If it is `0`, the prompt consumed the whole `max_length`; increase
`--max-length` before trusting loss/perplexity.
During training, `metrics.json` and `train_log.csv` also record
`train_valid_tokens`, `val_valid_tokens`, batch counts, and zero-token batch
counts. Targeted training now raises immediately if training or validation has
zero supervised password tokens.

## Audit Targeted Token Lengths

This checks prompt/password token coverage without loading model weights:

```bash
python main.py inspect-targeted-lengths --base-model local-qwen --prompt-template-id 0 --data-path data\clixsense\clixsense_test_500_from_fd500k_p00.json --max-train-samples 1000 --max-lengths 128,256,384,512 --out artifacts\reports\targeted_length_audit_qwen_fd500k_targets.json
```

On the local Qwen tokenizer and reconstructed 500-target `fd500k_p00` set,
`max_length=256` has `zero_valid_records=0` and `truncated_records=0`; the
minimum max length for nonzero password labels in that slice is `87`.

## Audit Target Alignment

```bash
python scripts/audit_target_alignment.py --train-data "${PASSLLM_CODE_ROOT}/data/clixsense/clixsense_train_50.jsonl" --export-filtered-train data\clixsense\clixsense_train_50_no_fd500k_targets.jsonl
```

This exports `data/clixsense/clixsense_test_500_from_fd500k_p00.json` from the
PassLLM quick result and removes exact target leakage from the training JSONL.
The latest reports are under `artifacts/reports/target_alignment_audit*.md`.

## Inspect A Data File

```bash
python main.py inspect-data --data-path "${LOCAL_PASSWORD_FILE}" --max-train-samples 5
```

## Local PassLLM Assets

Set these environment variables to point at your local PassLLM assets:

- `PASSLLM_CODE_ROOT`: PassLLM/FieldDrop code root
- `PASSLLM_QUICK_ROOT`: quick result root; defaults to
  `${PASSLLM_CODE_ROOT}/result/quick`
- `PASSLLM_QWEN_MODEL`: optional explicit Qwen model directory
- `PASSLLM_FIELDDROP_ADAPTER`: optional explicit FieldDrop adapter directory
- `LOCAL_PASSWORD_FILE`: optional local password-list file for ad hoc smoke tests

The imported baseline contract is recorded at:

```text
baselines/imported/passllm-fielddrop/json/metric_contract.json
```

## Current Caveat

Some local machines may report CPU-only PyTorch. They can still run smoke tests,
preflight, status, report, and `scripts/repro_check.py`, but regenerating the
500-row Qwen/FieldDrop formal comparison needs a CUDA host. Keep the method
boundary clear: FieldDrop is the comparison / initialization baseline, while
PassMoE is the router/expert/fusion layer being tested here.
