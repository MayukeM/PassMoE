# PassMoE Revival Checklist

## Identity

- baseline id: passmoe-revived
- route: repair upstream PassMoE and import PassLLM engineering ideas
- owner stage: baseline -> experiment

## Core

- [x] PassMoE source downloaded.
- [x] PassLLM source downloaded for reference.
- [x] Local `D:\paper` password/model assets scanned at coarse level.
- [x] `PLAN.md` captures route, command path, expected outputs, acceptance condition, and fallback.
- [x] upstream PassMoE compile/import state audited and repair route chosen.
- [x] repaired code imports cleanly.
- [x] tiny CPU smoke train finishes and writes a checkpoint.
- [x] generation produces non-empty password candidates.
- [x] evaluation writes finite loss / hit-rate metrics.
- [x] local PassLLM/Qwen assets are identified for a comparable config.
- [x] baseline comparison contract is explicit for the first PassLLM result anchor.
- [x] targeted PassLLM-style data path is implemented and smoke-tested.
- [x] local Qwen2.5-0.5B PassMoE micro-train is smoke-tested.
- [x] PassLLM adapter merge path is implemented and smoke-tested.
- [x] compact Qwen checkpoint saving is verified.
- [x] unified PassLLM/PassMoE JSONL scoring is implemented and verified.
- [x] paper-facing PassMoE improvement run is gated locally with a durable CPU-only note.
- [x] CPU-side PassMoE-style candidate fusion is implemented and verified against existing PassLLM quick outputs.
- [x] fusion improves the primary `fd500k_p00` quick anchor SR@100 from `0.0736` to `0.0815` under recomputed-rank scoring.
- [x] fusion-only artifact validated on the stricter 500-row `fd500k_p00_unique` contract: raw SR@100 `0.0740`, fused SR@100 `0.0820`, zero worsened ranks, zero lost hits.
- [x] score-only formal runner now auto-validates imported/fusion artifacts with `validation_min_candidates=0`, while full neural execute validation remains strict.
- [x] one-command fusion reproduction script added and rerun for all three PassLLM quick anchors.
- [x] conservative fusion parameter search added; selected `score_expert_weight=0.05` keeps SR@1/SR@10 unchanged and has zero worsened ranks on `fd500k_p00`.
- [x] formal Qwen+FieldDrop+PassMoE runner added with preflight, manifest, raw/fused score, fusion analysis, and baseline comparison outputs.
- [x] checkpoint resume added for `main.py train` and verified on a two-stage tiny run.
- [x] formal runner recovery modes added and verified: `--resume-from`, `--checkpoint`, `--skip-train-if-jsonl-exists`, and `--postprocess-only`.
- [x] targeted loss NaN guard added for all-ignored label batches; evaluation now reports `valid_tokens`.
- [x] resumable targeted generation added and verified with a 1-row to 2-row resume smoke.
- [x] tokenizer-only targeted length audit added and verified on Qwen/ClixSense; formal default `max_length=256` has zero zero-valid/truncated records on the first 1,000 sample records.
- [x] formal preflight now audits both evaluation targets and training samples for supervised password-token coverage.
- [x] current Qwen length audit: eval zero-valid/truncated `0/500`; train zero-valid/truncated `0/1000` at `max_length=256`.
- [x] trainer now records valid token and zero-token batch statistics and rejects targeted runs with zero supervised tokens.
- [x] fresh non-resume training runs reset `train_log.csv`; resume runs continue appending.
- [x] PassMoE expert fusion optimized to mix hidden states before one shared LM-head projection, reducing CUDA vocab-logit memory pressure.
- [x] HF `device_map` is now opt-in; default formal run no longer requires `accelerate`.
- [x] explicit `--use-device-map` preflight checks for `accelerate` and fails cleanly when missing.
- [x] hidden-mix end-to-end execute smoke passed: `artifacts/formal/hidden_mix_execute_smoke/formal_validation.json`.
- [x] formal `--dtype auto` added; current formal manifest resolves planned CUDA dtype to `bfloat16`.
- [x] dtype-auto end-to-end execute smoke passed: `artifacts/formal/dtype_auto_execute_smoke/formal_validation.json`.
- [x] generation length split from training sequence length; formal runs keep `--max-length 256` for supervised-token coverage but default generation to `--generation-max-new-tokens 32`.
- [x] generation-length execute smoke passed: `artifacts/formal/generation_length_execute_smoke/formal_validation.json`.
- [x] targeted beam generation now batches active beams with `--generation-batch-size`; `batched_generation_execute_smoke` passed validation with non-singleton generation batches.
- [x] formal score commands now recompute ranks from `outputPasswords` instead of trusting stale JSONL rank fields.
- [x] formal validator now checks raw/fused candidate duplication and `min_cracked_guess_number` consistency against recomputed ranks.
- [x] resume generation now rejects completed rows with fewer than `--target-candidates-per-user` candidates; `resume_min_candidate_smoke` verified short rows are regenerated.
- [x] beam generation now stops/tops up by unique decoded passwords rather than completed token sequences; `unique_candidate_execute_smoke` verified full candidate counts and no duplicate candidates.
- [x] resume generation now rejects stale rows with empty candidates, duplicate decoded passwords, or prompt leakage; `resume_quality_filter_smoke` corrupted two rows, reused only one clean row, regenerated the bad rows, and passed validation.
- [x] `--skip-train-if-jsonl-exists` now runs a pre-score reuse quality gate and writes `reused_jsonl_quality.json`; bad reused JSONL fails early, good reused JSONL passes, and BOM-prefixed JSONL is readable.
- [x] `formal_validation.json` now records SHA256 hashes for result artifacts; `inspect_formal_status.py` reports `validation_stale` or `row_count_mismatch` instead of `complete` when artifacts change after validation.
- [x] claim-carrying non-score-only artifacts now require `targeted_generation_metrics.json` model-execution provenance; missing provenance is reported as `model_execution_unverified`.
- [x] formal runner now writes per-command logs under `artifacts/formal/<run>/logs`; `logged_execute_smoke3` verified train/score/fuse/analyze/validate logs and Windows-safe console forwarding.
- [x] formal runner now writes `environment_snapshot.json` and records `environment_snapshot_path` in `run_manifest.json`; `env_snapshot_smoke` verified the execute path and remains `diagnostic_only`.
- [x] CUDA readiness checker added; formal runner now writes `cuda_readiness.json/md`, current host is `not_ready` due CPU-only torch and 2 GB MX250, and `cuda_readiness_smoke` passed validation as diagnostic-only.
- [x] formal seed contract is explicit; `--seed` defaults to `42`, is passed to train/evaluate, and `seed_contract_smoke` verified non-default seed propagation through artifacts.
- [x] formal status/report recovery commands preserve the seed; current `needs_model_execution` and fusion-only reports recommend `--seed 42`.
- [x] targeted generation now emits `__PASSMOE_PROGRESS__` JSON markers into command logs; `progress_marker_line_smoke` verified persisted markers and passed validation as diagnostic-only.
- [x] formal status/report now surface the latest `__PASSMOE_PROGRESS__` marker as monitoring-only `targeted_generation_progress`; `progress_marker_line_smoke` reports 2/2 and the real formal run still reports `needs_model_execution`.
- [x] progress markers now include resume-aware ETA fields: generated rows this run, remaining rows, seconds per generated row, ETA seconds, and estimated total seconds; `progress_eta_execute_smoke` verified these fields in logs/status/report.
- [x] targeted generation now decodes candidate suffixes by prompt token length instead of string-prefix stripping; validator checks raw/fused prompt leakage and `token_suffix_execute_smoke` passed.
- [x] targeted generation router features now use the candidate suffix plus record PII, matching training-time `FeatureExtractor.extract(password, pii)` semantics; `feature_consistency_execute_smoke` passed.
- [x] checkpoint loading now allows missing frozen backbone keys but rejects missing trainable keys; new checkpoints record `checkpoint_format`, `trainable_keys`, optimizer state, history, and merge report.
- [x] checkpoint contract smoke passed: new checkpoint evaluate and epoch-2 resume both loaded all trainable keys and preserved history/optimizer state.
- [x] formal preflight now rejects budgets missing from the baseline contract and rejects `--target-candidates-per-user` values below the largest requested budget.
- [x] formal preflight now records data cardinality and rejects train/eval requests larger than available records; current train rows `1,063,798`, requested `10,000`, eval rows `500`, requested `500`.
- [x] formal artifact status inspector added; `inspect_formal_status.py` distinguishes missing model execution, validation failure, and complete runs with recovery command suggestions.
- [x] formal result report renderer added; `render_formal_report.py` writes `formal_result_report.md/json` and separates incomplete formal runs from diagnostic CPU/subset smoke runs.
- [x] formal runner now auto-renders `formal_result_report.md/json` after preflight, score-only, diagnostic execute, and full execute unless `--skip-result-report` is passed.
- [x] formal preflight now writes `run_formal_cuda.ps1`, a runner-based CUDA handoff script that preserves explicit manifest settings and refreshes status/report after execution.
- [x] post-move manifest path audit added; status and validation now check that formal artifacts were generated under current repo root `D:\paper\passllm-moe\PassMoE`, and the current formal manifest passes this audit.
- [x] formal status completion gate hardened; a stale passed `formal_validation.json` no longer marks a run complete unless JSONL row counts and required postprocess artifacts are present.
- [x] formal row-count gate added; partial JSONL is rejected in execute mode unless `--allow-partial-jsonl` is explicit.
- [x] formal baseline default corrected to `fd500k_p00_unique` to avoid the duplicated 503-row quick anchor.
- [x] `fd500k_p00_unique_fusion` score-only supplementary artifact regenerated under `D:\paper\passllm-moe\PassMoE`; validation is `passed` and path audit is clean.
- [x] current local validation passed: `py_compile`, formal preflight, JSONL scoring, data inspection, and independent tiny CPU smoke under `runs/current_validation_smoke`.
- [x] formal preflight now validates resolved local Qwen and FieldDrop adapter assets before CUDA execution.
- [x] actual local Qwen + FieldDrop model construction verified on CPU: LoRA merge `72` modules, `0` skipped; total parameters `494,172,675`, trainable `139,907`.
- [x] Qwen + FieldDrop construction check is now integrated into `scripts/run_formal_passmoe.py` and writes `artifacts/formal/qwen_fielddrop_passmoe_clixsense_10k/deep_model_check.json` by default.
- [x] formal output validator added and wired into non-diagnostic `--execute`; current formal directory correctly fails validation until CUDA-generated JSONL/score artifacts exist.
- [x] validator positive path checked on tiny diagnostic artifacts with `--expected-rows 5 --min-candidates 0`.
- [x] formal runner auto-validation path verified end-to-end on tiny CPU execute: `artifacts/formal/auto_validation_execute_smoke/formal_validation.json` is `passed`.
- [x] real local Qwen + FieldDrop execute path verified end-to-end on CPU diagnostic: `artifacts/formal/qwen_fielddrop_tiny_execute_smoke/formal_validation.json` is `passed` and its report is `diagnostic_only`.
- [x] auto-report execute smoke passed: `artifacts/formal/auto_report_execute_smoke/formal_result_report.json` is generated automatically and marked `diagnostic_only`.
- [x] diagnostic runner support added: `--max-eval-samples` and explicit no-adapter aliases `none`, `null`, `-`.
- [ ] CUDA formal neural comparison still needs to run: `.\artifacts\formal\qwen_fielddrop_passmoe_clixsense_10k\run_formal_cuda.ps1` from `D:\paper\passllm-moe\PassMoE` on a CUDA-enabled PyTorch host.

## Implementation

- [x] `config.py` rewritten.
- [x] `data.py` rewritten.
- [x] `model.py` rewritten.
- [x] `trainer.py` rewritten.
- [x] `evaluate.py` rewritten.
- [x] `fusion.py` added.
- [x] `main.py` rewritten.
- [x] `scripts/run_fusion_experiments.py` added.
- [x] `scripts/search_fusion_params.py` added.
- [x] `scripts/deduplicate_jsonl.py` added for reproducible imported-JSONL deduplication.
- [x] `scripts/inspect_formal_status.py` added.
- [x] `scripts/render_formal_report.py` added.
- [x] `scripts/check_cuda_readiness.py` added.
- [x] `scripts/run_formal_passmoe.py` added.
- [x] `scripts/validate_formal_outputs.py` added.
- [x] recovery/resume support added to `trainer.py`, `main.py`, `model.py`, `evaluate.py`, and `scripts/run_formal_passmoe.py`.
- [x] targeted valid-token guard added to `trainer.py`.
- [x] formal preflight now writes `targeted_length_audit.json` when model execution is required.
- [x] formal runner completeness check added to `scripts/run_formal_passmoe.py`.
- [x] README quick-start commands corrected.

## Closeout For Current Pass

- [x] smoke result summarized.
- [x] next action is explicit: run the formal CUDA comparison and require `formal_validation.json` status `passed` before claiming PassMoE matches or exceeds PassLLM.
