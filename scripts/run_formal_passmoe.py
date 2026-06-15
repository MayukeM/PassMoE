from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = r"data\clixsense\clixsense_train_50_no_fd500k_targets.jsonl"
DEFAULT_TEST_DATA_PATH = r"data\clixsense\clixsense_test_500_from_fd500k_p00.json"
DEFAULT_BASELINE_CONTRACT = r"baselines\imported\passllm-fielddrop\json\metric_contract.json"
COMMAND_LOG_DIR: Path | None = None
COMMAND_LOG_INDEX = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight, run, and score the formal PassMoE targeted comparison. "
            "When --base-adapter fielddrop is used, FieldDrop is an imported "
            "PassLLM baseline/foundation adapter, not a PassMoE method component."
        )
    )
    parser.add_argument("--run-name", default="qwen_fielddrop_passmoe_clixsense_10k")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-data-path", default=DEFAULT_TEST_DATA_PATH)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--artifacts-dir", default="artifacts/formal")
    parser.add_argument("--baseline-contract", default=DEFAULT_BASELINE_CONTRACT)
    parser.add_argument("--baseline-variant", default="fd500k_p00_unique")
    parser.add_argument("--base-model", default="local-qwen")
    parser.add_argument("--base-adapter", default="fielddrop")
    parser.add_argument("--prompt-template-id", default="0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-eval-samples", type=int, default=500)
    parser.add_argument("--target-candidates-per-user", type=int, default=100)
    parser.add_argument("--budgets", default="1,10,50,100")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--use-device-map", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run training and scoring.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow execute mode without CUDA.")
    parser.add_argument("--score-only", action="store_true", help="Only score an existing targeted JSONL.")
    parser.add_argument("--postprocess-only", action="store_true", help="Alias for --score-only; uses the run JSONL by default.")
    parser.add_argument("--jsonl", default="", help="Existing targeted_input_output.jsonl for --score-only.")
    parser.add_argument("--resume-from", default="", help="Resume training from a PassMoE checkpoint such as runs/<run>/last.pt.")
    parser.add_argument("--checkpoint", default="", help="Generate/evaluate from an existing checkpoint instead of training.")
    parser.add_argument("--skip-train-if-jsonl-exists", action="store_true", help="Reuse the expected JSONL when it already exists.")
    parser.add_argument("--resume-generation", action="store_true", help="Resume targeted generation from an existing partial JSONL.")
    parser.add_argument("--allow-partial-jsonl", action="store_true", help="Allow scoring fewer rows than --target-eval-samples in execute mode.")
    parser.add_argument("--skip-output-validation", dest="output_validation", action="store_false")
    parser.add_argument("--skip-result-report", dest="result_report", action="store_false")
    parser.add_argument(
        "--validation-expected-rows",
        type=int,
        default=0,
        help="Override validator expected rows. Default: --target-eval-samples for model execution; baseline contract for score-only.",
    )
    parser.add_argument(
        "--validation-min-candidates",
        type=int,
        default=None,
        help="Override validator candidate budget. Default: max budget for model execution, 0 for score-only imports.",
    )
    parser.add_argument("--skip-length-audit", dest="length_audit", action="store_false")
    parser.add_argument("--length-audit-samples", type=int, default=1000)
    parser.add_argument("--length-audit-max-lengths", default="")
    parser.add_argument(
        "--skip-deep-model-check",
        dest="deep_model_check",
        action="store_false",
        help="Skip loading the local backbone and adapter during preflight.",
    )
    parser.add_argument("--no-post-fusion", dest="post_fusion", action="store_false")
    parser.add_argument("--fusion-max-expert-candidates", type=int, default=80)
    parser.add_argument("--fusion-score-existing-weight", type=float, default=1.0)
    parser.add_argument("--fusion-score-expert-weight", type=float, default=0.05)
    parser.add_argument("--fusion-score-rank-offset", type=float, default=2.0)
    parser.add_argument("--fusion-bootstrap-iters", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing score/comparison files.")
    parser.set_defaults(post_fusion=True)
    parser.set_defaults(length_audit=True)
    parser.set_defaults(deep_model_check=True)
    parser.set_defaults(output_validation=True)
    parser.set_defaults(result_report=True)
    args = parser.parse_args()
    if args.postprocess_only:
        args.score_only = True

    artifacts_dir = (REPO_ROOT / args.artifacts_dir / args.run_name).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    configure_command_logging(artifacts_dir / "logs")

    baseline_contract_path = resolve_repo_path(args.baseline_contract)
    baseline_contract = load_json(baseline_contract_path)
    baseline_metrics = baseline_contract.get("baseline_variants", {}).get(args.baseline_variant, {})
    budgets = parse_budgets(args.budgets)
    run_dir = (REPO_ROOT / args.output_dir / args.run_name).resolve()
    result_jsonl = resolve_repo_path(args.jsonl) if args.jsonl else run_dir / "targeted_input_output.jsonl"
    score_path = artifacts_dir / "score.json"
    comparison_path = artifacts_dir / "comparison.json"
    fused_jsonl = artifacts_dir / "fused_targeted_input_output.jsonl"
    fusion_metrics_path = artifacts_dir / "fusion_metrics.json"
    fusion_analysis_path = artifacts_dir / "fusion_analysis.json"
    fused_score_path = artifacts_dir / "fused_score.json"
    fused_comparison_path = artifacts_dir / "fused_comparison.json"
    length_audit_path = artifacts_dir / "targeted_length_audit.json"
    train_length_audit_path = artifacts_dir / "targeted_length_audit_train.json"
    deep_model_check_path = artifacts_dir / "deep_model_check.json"
    reused_jsonl_quality_path = artifacts_dir / "reused_jsonl_quality.json"
    environment_snapshot_path = artifacts_dir / "environment_snapshot.json"
    cuda_readiness_path = artifacts_dir / "cuda_readiness.json"
    cuda_readiness_md_path = artifacts_dir / "cuda_readiness.md"
    cuda_launcher_path = artifacts_dir / "run_formal_cuda.ps1"

    cuda_available, torch_version = detect_torch()
    device = resolve_device(args.device, cuda_available, args.allow_cpu)
    resolved_dtype = resolve_run_dtype(args.dtype, device, cuda_available)
    skip_existing_jsonl = bool(args.skip_train_if_jsonl_exists and result_jsonl.exists())
    needs_model_execution = not args.score_only and not skip_existing_jsonl
    manifest_device = "score-only" if not needs_model_execution else device
    preflight = build_preflight(
        args,
        baseline_contract_path,
        baseline_contract,
        cuda_available,
        torch_version,
        device,
        result_jsonl,
        needs_model_execution,
    )
    model_asset = resolve_base_model_asset(args.base_model)
    adapter_asset = resolve_adapter_asset(args.base_adapter)
    train_command = build_train_command(args, device, resolved_dtype)
    eval_command = build_eval_command(args, device, resolved_dtype, resolve_repo_path(args.checkpoint)) if args.checkpoint else []
    execution_command = [] if not needs_model_execution else (eval_command or train_command)
    length_audit_specs: list[tuple[str, Path, list[str]]] = []
    if args.length_audit and needs_model_execution:
        length_audit_specs = [
            (
                "eval",
                length_audit_path,
                build_length_audit_command(
                    args,
                    length_audit_path,
                    args.test_data_path or args.data_path,
                ),
            ),
            (
                "train",
                train_length_audit_path,
                build_length_audit_command(args, train_length_audit_path, args.data_path),
            ),
        ]
    length_audit_commands = [command for _label, _path, command in length_audit_specs]
    length_audit_command = length_audit_commands[0] if length_audit_commands else []
    deep_model_check_enabled = bool(args.deep_model_check and needs_model_execution)
    score_command = build_score_command(result_jsonl, args.budgets, score_path)
    fuse_command = build_fuse_command(args, result_jsonl, fused_jsonl, fusion_metrics_path)
    fused_score_command = build_score_command(fused_jsonl, args.budgets, fused_score_path)
    fusion_analysis_command = build_fusion_analysis_command(
        args,
        original_jsonl=result_jsonl,
        fused_jsonl=fused_jsonl,
        out_path=fusion_analysis_path,
    )
    validation_command = build_validation_command(args, artifacts_dir, baseline_contract_path)
    auto_output_validation = bool(args.output_validation and not args.allow_partial_jsonl)
    environment_snapshot = build_environment_snapshot(args, device, resolved_dtype, cuda_available, torch_version)
    write_json(environment_snapshot_path, environment_snapshot)
    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "data_path": str(resolve_repo_path(args.data_path)),
        "test_data_path": str(resolve_repo_path(args.test_data_path)) if args.test_data_path else "",
        "data_counts": preflight.get("data_counts", {}),
        "artifacts_dir": str(artifacts_dir),
        "baseline_contract": str(baseline_contract_path),
        "baseline_variant": args.baseline_variant,
        "baseline_metrics": baseline_metrics,
        "base_model": args.base_model,
        "base_adapter": args.base_adapter,
        "resolved_base_model": str(model_asset.get("path", "")),
        "resolved_base_adapter": str(adapter_asset.get("path", "")),
        "budgets": budgets,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_length": args.max_length,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "generation_batch_size": args.generation_batch_size,
        "seed": args.seed,
        "device": manifest_device,
        "requested_dtype": args.dtype,
        "dtype": resolved_dtype,
        "use_device_map": bool(args.use_device_map),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "train_command": train_command,
        "eval_command": eval_command,
        "execution_command": execution_command,
        "score_command": score_command,
        "command_logs_dir": str(COMMAND_LOG_DIR) if COMMAND_LOG_DIR else "",
        "environment_snapshot_path": str(environment_snapshot_path),
        "cuda_readiness_path": str(cuda_readiness_path),
        "cuda_readiness_md_path": str(cuda_readiness_md_path),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "post_fusion": args.post_fusion,
        "resume_from": str(resolve_repo_path(args.resume_from)) if args.resume_from else "",
        "checkpoint": str(resolve_repo_path(args.checkpoint)) if args.checkpoint else "",
        "skip_train_if_jsonl_exists": bool(args.skip_train_if_jsonl_exists),
        "skip_existing_jsonl": skip_existing_jsonl,
        "needs_model_execution": needs_model_execution,
        "length_audit": bool(args.length_audit and needs_model_execution),
        "length_audit_path": str(length_audit_path) if args.length_audit and needs_model_execution else "",
        "length_audit_command": length_audit_command,
        "length_audit_paths": {label: str(path) for label, path, _command in length_audit_specs},
        "length_audit_commands": length_audit_commands,
        "deep_model_check": deep_model_check_enabled,
        "deep_model_check_path": str(deep_model_check_path) if deep_model_check_enabled else "",
        "resume_generation": bool(args.resume_generation),
        "allow_partial_jsonl": bool(args.allow_partial_jsonl),
        "validation_expected_rows": validation_expected_rows(args),
        "validation_min_candidates": validation_min_candidates(args),
        "output_validation": auto_output_validation,
        "validation_command": validation_command if auto_output_validation else [],
        "result_report": bool(args.result_report),
        "result_report_md": str(artifacts_dir / "formal_result_report.md") if args.result_report else "",
        "result_report_json": str(artifacts_dir / "formal_result_report.json") if args.result_report else "",
        "cuda_launcher_ps1": str(cuda_launcher_path),
        "fusion_config": {
            "max_expert_candidates": args.fusion_max_expert_candidates,
            "score_existing_weight": args.fusion_score_existing_weight,
            "score_expert_weight": args.fusion_score_expert_weight,
            "score_rank_offset": args.fusion_score_rank_offset,
            "bootstrap_iters": args.fusion_bootstrap_iters,
        },
        "fuse_command": fuse_command if args.post_fusion else [],
        "fused_score_command": fused_score_command if args.post_fusion else [],
        "fusion_analysis_command": fusion_analysis_command if args.post_fusion else [],
        "expected_jsonl": str(result_jsonl),
        "expected_fused_jsonl": str(fused_jsonl) if args.post_fusion else "",
        "reused_jsonl_quality_path": str(reused_jsonl_quality_path),
        "mode": "score_only" if args.score_only else ("execute" if args.execute else "preflight_only"),
    }
    if length_audit_specs and not preflight["errors"]:
        for label, audit_path, command in length_audit_specs:
            run_command(command, quiet=True)
            attach_length_audit_preflight(preflight, audit_path, args.max_length, label=label)
    if deep_model_check_enabled and not preflight["errors"]:
        deep_report = run_deep_model_check(args, deep_model_check_path)
        attach_deep_model_check_preflight(preflight, deep_model_check_path, deep_report)
    write_json(artifacts_dir / "preflight.json", preflight)
    write_json(artifacts_dir / "run_manifest.json", manifest)
    write_commands_ps1(artifacts_dir / "commands.ps1", manifest)
    write_cuda_launcher_ps1(cuda_launcher_path, manifest)
    write_cuda_readiness_if_possible(artifacts_dir, cuda_readiness_path, cuda_readiness_md_path)

    status = classify_status(preflight, cuda_available, args.allow_cpu, needs_model_execution)
    if args.execute or args.score_only:
        if preflight["errors"]:
            write_summary(artifacts_dir / "summary.md", status, preflight, manifest, None)
            render_result_report_if_enabled(args, artifacts_dir)
            raise SystemExit(f"Preflight failed; see {artifacts_dir / 'preflight.json'}")
        if args.execute and needs_model_execution and device == "cuda" and not cuda_available and not args.allow_cpu:
            status = "gated_no_cuda"
            preflight["errors"].append("CUDA is not available; run on a CUDA host or pass --allow-cpu for diagnostics.")
            write_json(artifacts_dir / "preflight.json", preflight)
            write_summary(artifacts_dir / "summary.md", status, preflight, manifest, None)
            render_result_report_if_enabled(args, artifacts_dir)
            raise SystemExit("CUDA is not available; formal execute mode is gated.")
        if args.score_only or skip_existing_jsonl:
            if not result_jsonl.exists():
                raise SystemExit(f"Score-only JSONL does not exist: {result_jsonl}")
            if skip_existing_jsonl:
                print(f"Reusing existing JSONL: {result_jsonl}")
                quality = validate_reused_jsonl(
                    result_jsonl,
                    expected_rows=int(args.target_eval_samples),
                    min_candidates=max(budgets) if budgets else 0,
                    require_complete=not args.allow_partial_jsonl,
                )
                write_json(reused_jsonl_quality_path, quality)
                if quality["status"] != "passed":
                    raise SystemExit(
                        "Existing JSONL failed the reuse quality gate; "
                        f"see {reused_jsonl_quality_path}. "
                        "Use --resume-generation to regenerate bad/missing rows, "
                        "or --allow-partial-jsonl only for diagnostics."
                    )
        elif eval_command:
            run_command(eval_command)
            if not result_jsonl.exists():
                raise SystemExit(f"Checkpoint evaluation finished but expected JSONL is missing: {result_jsonl}")
        else:
            run_command(train_command)
            if not result_jsonl.exists():
                raise SystemExit(f"Training finished but expected JSONL is missing: {result_jsonl}")
        if score_path.exists() and not args.force:
            raise SystemExit(f"Score file already exists, pass --force to overwrite: {score_path}")
        run_command(score_command)
        if not args.score_only and not args.allow_partial_jsonl:
            validate_complete_score(score_path, args.target_eval_samples)
        raw_comparison = compare_against_baseline(score_path, baseline_metrics, budgets)
        write_json(comparison_path, raw_comparison)
        comparisons: dict[str, Any] = {"raw": raw_comparison}
        if args.post_fusion:
            for path in (fused_jsonl, fusion_metrics_path, fusion_analysis_path, fused_score_path, fused_comparison_path):
                if path.exists() and not args.force:
                    raise SystemExit(f"Fusion output already exists, pass --force to overwrite: {path}")
            run_command(fuse_command)
            run_command(fused_score_command)
            run_command(fusion_analysis_command)
            fused_comparison = compare_against_baseline(fused_score_path, baseline_metrics, budgets)
            write_json(fused_comparison_path, fused_comparison)
            comparisons["fused"] = fused_comparison
            comparisons["fusion_analysis_path"] = str(fusion_analysis_path)
        validation_error: subprocess.CalledProcessError | None = None
        if auto_output_validation:
            try:
                run_command(validation_command)
            except subprocess.CalledProcessError as exc:
                validation_error = exc
        status = "validation_failed" if validation_error else "complete"
        write_summary(artifacts_dir / "summary.md", status, preflight, manifest, comparisons)
        render_result_report_if_enabled(args, artifacts_dir)
        if validation_error is not None:
            raise validation_error
        print(json.dumps({"status": status, "artifacts_dir": str(artifacts_dir)}, indent=2))
        return

    write_summary(artifacts_dir / "summary.md", status, preflight, manifest, None)
    render_result_report_if_enabled(args, artifacts_dir)
    print(json.dumps({"status": status, "artifacts_dir": str(artifacts_dir)}, indent=2))


def build_preflight(
    args: argparse.Namespace,
    baseline_contract_path: Path,
    baseline_contract: dict[str, Any],
    cuda_available: bool,
    torch_version: str,
    device: str,
    result_jsonl: Path,
    needs_model_execution: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    data_path = resolve_repo_path(args.data_path)
    test_data_path = resolve_repo_path(args.test_data_path) if args.test_data_path else None
    if needs_model_execution:
        check("data_path", data_path.exists(), str(data_path))
        if test_data_path is not None:
            check("test_data_path", test_data_path.exists(), str(test_data_path))
        data_counts: dict[str, Any] = {}
        if data_path.exists():
            train_rows = count_data_records(data_path)
            data_counts["train_rows"] = train_rows
            data_counts["requested_train_rows"] = int(args.max_train_samples)
            check(
                "train_data_rows",
                train_rows >= int(args.max_train_samples) > 0,
                f"available={train_rows}, requested={args.max_train_samples}",
            )
        if test_data_path is not None and test_data_path.exists():
            eval_rows = count_data_records(test_data_path)
            data_counts["eval_rows"] = eval_rows
            data_counts["target_eval_samples"] = int(args.target_eval_samples)
            check(
                "eval_data_rows",
                eval_rows >= int(args.target_eval_samples) > 0,
                f"available={eval_rows}, requested={args.target_eval_samples}",
            )
    elif args.score_only:
        check("score_jsonl", result_jsonl.exists(), str(result_jsonl))
    else:
        check("existing_jsonl", result_jsonl.exists(), str(result_jsonl))
    check("baseline_contract", baseline_contract_path.exists(), str(baseline_contract_path))
    check(
        "baseline_variant",
        args.baseline_variant in baseline_contract.get("baseline_variants", {}),
        args.baseline_variant,
    )
    requested_budgets = parse_budgets(args.budgets)
    baseline_metrics = baseline_contract.get("baseline_variants", {}).get(args.baseline_variant, {})
    missing_budget_keys = [f"sr{budget}" for budget in requested_budgets if f"sr{budget}" not in baseline_metrics]
    check(
        "baseline_budget_keys",
        not missing_budget_keys,
        f"budgets={requested_budgets}, missing={missing_budget_keys}",
    )
    if needs_model_execution and requested_budgets:
        max_budget = max(requested_budgets)
        check(
            "candidate_budget",
            int(args.target_candidates_per_user) >= max_budget,
            f"target_candidates_per_user={args.target_candidates_per_user}, max_budget={max_budget}",
        )
    check("torch_import", torch_version != "unavailable", torch_version)
    check("requested_dtype", args.dtype in {"auto", "float32", "float16", "bfloat16"}, args.dtype)
    if args.resume_from:
        resume_path = resolve_repo_path(args.resume_from)
        check("resume_checkpoint", resume_path.exists(), str(resume_path))
    if args.checkpoint:
        checkpoint_path = resolve_repo_path(args.checkpoint)
        check("checkpoint", checkpoint_path.exists(), str(checkpoint_path))
    if needs_model_execution:
        add_asset_preflight_checks(args, check)
        checks.append(
            {
                "name": "cuda_available_for_formal",
                "ok": bool(cuda_available or args.allow_cpu),
                "detail": str(cuda_available),
                "severity": "warning" if not (cuda_available or args.allow_cpu) else "info",
            }
        )
        if not (cuda_available or args.allow_cpu):
            warnings.append("CUDA is not available on this host; execute mode is gated until run on CUDA.")
    check("python_executable", Path(sys.executable).exists(), sys.executable)
    check("main_py", (REPO_ROOT / "main.py").exists(), str(REPO_ROOT / "main.py"))
    check("disk_free_gb", disk_free_gb(REPO_ROOT) >= 5.0, f"{disk_free_gb(REPO_ROOT):.2f} GB")

    if needs_model_execution:
        for module_name in ("transformers", "safetensors"):
            ok, detail = can_import(module_name)
            check(f"import_{module_name}", ok, detail)
        if args.use_device_map:
            ok, detail = can_import("accelerate")
            check("import_accelerate_for_device_map", ok, detail)

    return {
        "status": "failed" if errors else "passed",
        "device": "score-only" if not needs_model_execution else device,
        "cuda_available": cuda_available,
        "torch_version": torch_version,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "data_counts": data_counts if needs_model_execution else {},
    }


def add_asset_preflight_checks(
    args: argparse.Namespace,
    check: Callable[[str, bool, str], None],
) -> None:
    model_asset = resolve_base_model_asset(args.base_model)
    model_kind = str(model_asset["kind"])
    if model_kind == "tiny":
        check("base_model_asset", True, "tiny CPU smoke model")
    elif model_kind == "remote":
        check("base_model_asset", True, f"remote HuggingFace id, local files not prechecked: {args.base_model}")
    else:
        model_path = Path(str(model_asset["path"]))
        check("base_model_path", model_path.exists(), str(model_path))
        if model_path.exists():
            check("base_model_config", (model_path / "config.json").exists(), str(model_path / "config.json"))
            tokenizer_ok = any((model_path / name).exists() for name in ("tokenizer.json", "vocab.json"))
            check("base_model_tokenizer", tokenizer_ok, str(model_path))
            weights_ok = bool(
                list(model_path.glob("*.safetensors"))
                or list(model_path.glob("pytorch_model*.bin"))
                or list(model_path.glob("model*.bin"))
            )
            check("base_model_weights", weights_ok, str(model_path))

    adapter_asset = resolve_adapter_asset(args.base_adapter)
    adapter_kind = str(adapter_asset["kind"])
    if adapter_kind == "none":
        check("base_adapter_asset", True, "none")
        return

    adapter_path = Path(str(adapter_asset["path"]))
    check("base_adapter_path", adapter_path.exists(), str(adapter_path))
    if not adapter_path.exists():
        return

    config_path = adapter_path / "adapter_config.json"
    weight_path = adapter_path / "adapter_model.safetensors"
    check("base_adapter_config", config_path.exists(), str(config_path))
    check("base_adapter_weights", weight_path.exists(), str(weight_path))
    if config_path.exists():
        try:
            adapter_config = load_json(config_path)
            rank = adapter_config.get("r", "?")
            alpha = adapter_config.get("lora_alpha", "?")
            targets = adapter_config.get("target_modules", [])
            if isinstance(targets, list):
                target_detail = ",".join(str(item) for item in targets[:5])
            else:
                target_detail = str(targets)
            check("base_adapter_config_json", True, f"r={rank}, lora_alpha={alpha}, targets={target_detail}")
        except Exception as exc:
            check("base_adapter_config_json", False, repr(exc))


def run_deep_model_check(args: argparse.Namespace, out_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    log = StringIO()
    report: dict[str, Any] = {
        "status": "started",
        "base_model": args.base_model,
        "base_adapter": args.base_adapter,
        "device": "cpu",
    }
    try:
        model_asset = resolve_base_model_asset(args.base_model)
        if model_asset["kind"] == "remote":
            report.update(
                {
                    "status": "skipped",
                    "reason": "remote HuggingFace model id; deep check is local-only",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            write_json(out_path, report)
            return report

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from config import Config
        from model import build_model_and_tokenizer, count_parameters

        config = Config()
        config.task = "targeted"
        config.base_model = str(model_asset.get("path") or args.base_model)
        adapter_asset = resolve_adapter_asset(args.base_adapter)
        config.base_adapter = str(adapter_asset.get("path") or "")
        config.prompt_template_id = args.prompt_template_id
        config.max_length = args.max_length
        config.lora_rank = args.lora_rank
        config.device = "cpu"
        config.dtype = "float32"
        config.use_device_map = False

        with redirect_stdout(log), redirect_stderr(log):
            model, tokenizer = build_model_and_tokenizer(config)
            parameters = count_parameters(model)

        merge_report = getattr(model, "merge_report", {})
        tokenizer_name = getattr(tokenizer, "name_or_path", "")
        report.update(
            {
                "status": "passed",
                "resolved_base_model": config.base_model,
                "resolved_base_adapter": config.base_adapter,
                "tokenizer": tokenizer_name,
                "parameters": parameters,
                "merge_report": merge_report,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "log_tail": tail_text(log.getvalue(), 4000),
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "log_tail": tail_text(log.getvalue(), 4000),
            }
        )
    write_json(out_path, report)
    return report


def attach_deep_model_check_preflight(preflight: dict[str, Any], report_path: Path, report: dict[str, Any]) -> None:
    preflight["deep_model_check_path"] = str(report_path)
    status = report.get("status")
    if status == "skipped":
        add_preflight_check(
            preflight,
            "deep_model_check",
            True,
            f"skipped: {report.get('reason', '')}",
            severity="info",
        )
        preflight.setdefault("warnings", []).append(f"Deep model check skipped: {report.get('reason', '')}")
        return

    if status != "passed":
        add_preflight_check(
            preflight,
            "deep_model_check",
            False,
            f"{report.get('error', 'unknown error')}; report={report_path}",
        )
        return

    parameters = report.get("parameters", {})
    merge_report = report.get("merge_report", {})
    add_preflight_check(
        preflight,
        "deep_model_check",
        True,
        f"loaded model and tokenizer on CPU; report={report_path}",
        severity="info",
    )
    trainable = int(parameters.get("trainable", 0) or 0)
    total = int(parameters.get("total", 0) or 0)
    add_preflight_check(
        preflight,
        "deep_model_parameters",
        trainable > 0 and total >= trainable,
        f"total={total}, trainable={trainable}, trainable_pct={parameters.get('trainable_pct')}",
        severity="info",
    )
    if report.get("resolved_base_adapter"):
        merged = int(merge_report.get("merged_modules", 0) or 0)
        skipped = int(merge_report.get("skipped_modules", 0) or 0)
        add_preflight_check(
            preflight,
            "deep_model_lora_merge",
            merged > 0 and skipped == 0,
            f"merged={merged}, skipped={skipped}",
            severity="info" if merged > 0 and skipped == 0 else None,
        )
    preflight["deep_model_check_summary"] = {
        "parameters": parameters,
        "merge_report": merge_report,
        "elapsed_seconds": report.get("elapsed_seconds"),
    }


def add_preflight_check(
    preflight: dict[str, Any],
    name: str,
    ok: bool,
    detail: str,
    severity: str | None = None,
) -> None:
    item: dict[str, Any] = {"name": name, "ok": bool(ok), "detail": detail}
    if severity:
        item["severity"] = severity
    preflight.setdefault("checks", []).append(item)
    if not ok:
        preflight.setdefault("errors", []).append(f"{name}: {detail}")
        preflight["status"] = "failed"


def tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def resolve_base_model_asset(base_model: str) -> dict[str, Any]:
    if base_model.lower() == "tiny":
        return {"kind": "tiny", "path": ""}
    aliases = local_asset_aliases()
    if base_model in aliases["base_models"]:
        return {"kind": "alias", "path": Path(aliases["base_models"][base_model]).resolve()}

    candidate = resolve_repo_path(base_model)
    if Path(base_model).is_absolute() or candidate.exists():
        return {"kind": "path", "path": candidate}
    return {"kind": "remote", "path": ""}


def resolve_adapter_asset(base_adapter: str) -> dict[str, Any]:
    if not base_adapter or str(base_adapter).lower() in {"none", "null", "-"}:
        return {"kind": "none", "path": ""}
    aliases = local_asset_aliases()
    if base_adapter in aliases["adapters"]:
        return {"kind": "alias", "path": Path(aliases["adapters"][base_adapter]).resolve()}
    return {"kind": "path", "path": resolve_repo_path(base_adapter)}


def local_asset_aliases() -> dict[str, dict[str, str]]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from config import Config

    config = Config()
    return {
        "base_models": {
            "local-qwen": config.local_qwen_05b,
        },
        "adapters": {
            "fielddrop": config.local_fielddrop_adapter,
            "baseline10k": config.local_baseline10k_adapter,
            "csdn": config.local_csdn_adapter,
        },
    }


def attach_length_audit_preflight(
    preflight: dict[str, Any],
    audit_path: Path,
    selected_max_length: int,
    label: str = "eval",
) -> None:
    checks = preflight["checks"]
    warnings = preflight["warnings"]
    check_name = "targeted_length_audit" if label == "eval" else f"targeted_length_audit_{label}"
    if not audit_path.exists():
        checks.append(
            {
                "name": check_name,
                "ok": False,
                "detail": f"missing audit output: {audit_path}",
                "severity": "warning",
            }
        )
        warnings.append(f"Targeted length audit output is missing for {label}: {audit_path}")
        return

    audit = load_json(audit_path)
    selected = None
    for row in audit.get("lengths", []):
        if int(row.get("max_length", -1)) == int(selected_max_length):
            selected = row
            break

    if selected is None:
        checks.append(
            {
                "name": check_name,
                "ok": True,
                "detail": f"audit written but selected max_length={selected_max_length} was not included",
                "severity": "warning",
            }
        )
        warnings.append(f"{label} length audit did not include selected max_length={selected_max_length}.")
        return

    zero_valid = int(selected.get("zero_valid_records", 0))
    truncated = int(selected.get("truncated_records", 0))
    total = int(audit.get("num_targeted_records", 0))
    checks.append(
        {
            "name": check_name,
            "ok": zero_valid == 0,
            "detail": (
                f"max_length={selected_max_length}, zero_valid={zero_valid}/{total}, "
                f"truncated={truncated}/{total}, audit={audit_path}"
            ),
            "severity": "warning" if zero_valid else "info",
        }
    )
    summary = {
        "selected_max_length": selected_max_length,
        "num_targeted_records": total,
        "zero_valid_records": zero_valid,
        "truncated_records": truncated,
        "min_for_nonzero_labels": audit.get("min_max_length_for_nonzero_labels"),
        "min_for_untruncated_full_records": audit.get("min_max_length_for_untruncated_full_records"),
    }
    preflight.setdefault("length_audits", {})[label] = {
        "path": str(audit_path),
        "summary": summary,
    }
    if label == "eval":
        preflight["length_audit_path"] = str(audit_path)
        preflight["length_audit_summary"] = summary
    if zero_valid:
        warnings.append(
            f"{label} length audit found {zero_valid}/{total} targeted records with no supervised password token at "
            f"max_length={selected_max_length}; raise --max-length before trusting loss/perplexity."
        )


def build_train_command(args: argparse.Namespace, device: str, dtype: str) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "train",
        "--task",
        "targeted",
        "--base-model",
        args.base_model,
        "--base-adapter",
        args.base_adapter,
        "--dtype",
        dtype,
        "--prompt-template-id",
        args.prompt_template_id,
        "--data-path",
        args.data_path,
        "--test-data-path",
        args.test_data_path,
        "--epochs",
        str(args.epochs),
        "--max-train-samples",
        str(args.max_train_samples),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--generation-max-new-tokens",
        str(args.generation_max_new_tokens),
        "--generation-batch-size",
        str(args.generation_batch_size),
        "--lora-rank",
        str(args.lora_rank),
        "--beam-width",
        str(args.beam_width),
        "--target-eval-samples",
        str(args.target_eval_samples),
        "--target-candidates-per-user",
        str(args.target_candidates_per_user),
        "--budgets",
        args.budgets,
        "--seed",
        str(args.seed),
        "--device",
        device,
        "--output-dir",
        args.output_dir,
        "--run-name",
        args.run_name,
    ]
    if args.max_eval_samples:
        command.extend(["--max-eval-samples", str(args.max_eval_samples)])
    if args.use_device_map:
        command.append("--use-device-map")
    if args.resume_from:
        command.extend(["--resume-checkpoint", str(resolve_repo_path(args.resume_from))])
    if args.resume_generation:
        command.append("--resume-generation")
    return command


def build_eval_command(args: argparse.Namespace, device: str, dtype: str, checkpoint: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "evaluate",
        "--task",
        "targeted",
        "--dtype",
        dtype,
        "--prompt-template-id",
        args.prompt_template_id,
        "--data-path",
        args.data_path,
        "--test-data-path",
        args.test_data_path or args.data_path,
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--generation-max-new-tokens",
        str(args.generation_max_new_tokens),
        "--generation-batch-size",
        str(args.generation_batch_size),
        "--beam-width",
        str(args.beam_width),
        "--target-eval-samples",
        str(args.target_eval_samples),
        "--target-candidates-per-user",
        str(args.target_candidates_per_user),
        "--budgets",
        args.budgets,
        "--seed",
        str(args.seed),
        "--device",
        device,
        "--output-dir",
        args.output_dir,
        "--run-name",
        args.run_name,
    ]
    if args.max_eval_samples:
        command.extend(["--max-eval-samples", str(args.max_eval_samples)])
    if args.use_device_map:
        command.append("--use-device-map")
    if args.resume_generation:
        command.append("--resume-generation")
    return command


def build_length_audit_command(args: argparse.Namespace, out_path: Path, data_path: str) -> list[str]:
    if args.length_audit_max_lengths:
        max_lengths = args.length_audit_max_lengths
    else:
        max_lengths = ",".join(str(length) for length in sorted({128, 256, int(args.max_length), 384, 512}))
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "inspect-targeted-lengths",
        "--base-model",
        args.base_model,
        "--prompt-template-id",
        args.prompt_template_id,
        "--data-path",
        data_path,
        "--max-train-samples",
        str(args.length_audit_samples),
        "--max-lengths",
        max_lengths,
        "--out",
        str(out_path),
    ]


def build_score_command(result_jsonl: Path, budgets: str, score_path: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "score-jsonl",
        "--jsonl",
        str(result_jsonl),
        "--budgets",
        budgets,
        "--out",
        str(score_path),
        "--recompute-from-candidates",
    ]


def build_fuse_command(
    args: argparse.Namespace,
    input_jsonl: Path,
    fused_jsonl: Path,
    fusion_metrics_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "fuse-jsonl",
        "--jsonl",
        str(input_jsonl),
        "--out-jsonl",
        str(fused_jsonl),
        "--out-metrics",
        str(fusion_metrics_path),
        "--strategy",
        "score",
        "--max-expert-candidates",
        str(args.fusion_max_expert_candidates),
        "--score-existing-weight",
        str(args.fusion_score_existing_weight),
        "--score-expert-weight",
        str(args.fusion_score_expert_weight),
        "--score-rank-offset",
        str(args.fusion_score_rank_offset),
        "--budgets",
        args.budgets,
    ]


def build_fusion_analysis_command(
    args: argparse.Namespace,
    original_jsonl: Path,
    fused_jsonl: Path,
    out_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "analyze-fusion",
        "--original-jsonl",
        str(original_jsonl),
        "--fused-jsonl",
        str(fused_jsonl),
        "--budgets",
        args.budgets,
        "--bootstrap-iters",
        str(args.fusion_bootstrap_iters),
        "--out",
        str(out_path),
    ]


def build_validation_command(args: argparse.Namespace, artifacts_dir: Path, baseline_contract_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_formal_outputs.py"),
        "--artifacts-dir",
        str(artifacts_dir),
        "--baseline-contract",
        str(baseline_contract_path),
        "--expected-baseline-variant",
        args.baseline_variant,
        "--budgets",
        args.budgets,
    ]
    expected_rows = validation_expected_rows(args)
    if expected_rows:
        command.extend(["--expected-rows", str(expected_rows)])
    min_candidates = validation_min_candidates(args)
    if min_candidates is not None:
        command.extend(["--min-candidates", str(min_candidates)])
    if not args.post_fusion:
        command.append("--no-require-fused")
    if args.allow_cpu:
        command.append("--allow-baseline-row-override")
    return command


def build_environment_snapshot(
    args: argparse.Namespace,
    device: str,
    resolved_dtype: str,
    cuda_available: bool,
    torch_version: str,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "argv": sys.argv,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "requested": {
            "run_name": args.run_name,
            "device": args.device,
            "resolved_device": device,
            "dtype": args.dtype,
            "resolved_dtype": resolved_dtype,
            "base_model": args.base_model,
            "base_adapter": args.base_adapter,
            "seed": args.seed,
            "target_eval_samples": args.target_eval_samples,
            "target_candidates_per_user": args.target_candidates_per_user,
            "budgets": args.budgets,
        },
        "environment_variables": selected_environment_variables(),
        "packages": selected_package_versions(),
        "torch": torch_environment_details(cuda_available, torch_version),
        "nvidia_smi": nvidia_smi_summary(),
    }


def selected_environment_variables() -> dict[str, str]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def selected_package_versions() -> dict[str, str]:
    package_names = ["torch", "transformers", "safetensors", "numpy", "tqdm", "accelerate", "peft"]
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"
        except Exception as exc:
            versions[package_name] = f"error: {exc!r}"
    return versions


def torch_environment_details(cuda_available: bool, torch_version: str) -> dict[str, Any]:
    details: dict[str, Any] = {
        "version": torch_version,
        "cuda_available": bool(cuda_available),
    }
    try:
        import torch

        details.update(
            {
                "version_cuda": getattr(torch.version, "cuda", None),
                "version_git": getattr(torch.version, "git_version", None),
                "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
        try:
            details["cuda_current_device"] = torch.cuda.current_device() if torch.cuda.is_available() else None
        except Exception as exc:
            details["cuda_current_device_error"] = repr(exc)
        try:
            details["bf16_supported"] = (
                bool(torch.cuda.is_bf16_supported()) if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_available() else False
            )
        except Exception as exc:
            details["bf16_supported_error"] = repr(exc)
        devices = []
        for index in range(int(details.get("cuda_device_count", 0) or 0)):
            device_info: dict[str, Any] = {"index": index}
            try:
                props = torch.cuda.get_device_properties(index)
                device_info.update(
                    {
                        "name": props.name,
                        "total_memory": int(props.total_memory),
                        "major": int(props.major),
                        "minor": int(props.minor),
                        "multi_processor_count": int(props.multi_processor_count),
                    }
                )
            except Exception as exc:
                device_info["error"] = repr(exc)
            devices.append(device_info)
        details["cuda_devices"] = devices
    except Exception as exc:
        details["import_error"] = repr(exc)
    return details


def nvidia_smi_summary() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"status": "not_found"}
    command = [
        executable,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"status": "error", "command": command, "error": repr(exc)}
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    devices = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        devices.append(
            {
                "index": parts[0] if len(parts) > 0 else "",
                "name": parts[1] if len(parts) > 1 else "",
                "memory_total_mb": safe_int(parts[2], 0) if len(parts) > 2 else 0,
                "driver_version": parts[3] if len(parts) > 3 else "",
            }
        )
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "return_code": completed.returncode,
        "devices": devices,
        "stderr": completed.stderr.strip(),
    }


def render_result_report_if_enabled(args: argparse.Namespace, artifacts_dir: Path) -> None:
    if not getattr(args, "result_report", True):
        return
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from render_formal_report import build_report, render_markdown

        report = build_report(artifacts_dir)
        (artifacts_dir / "formal_result_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (artifacts_dir / "formal_result_report.md").write_text(render_markdown(report), encoding="utf-8")
    except Exception as exc:
        warning_path = artifacts_dir / "formal_result_report_error.txt"
        warning_path.write_text("Failed to render formal result report:\n" + repr(exc), encoding="utf-8")


def write_cuda_readiness_if_possible(artifacts_dir: Path, out_json: Path, out_md: Path) -> None:
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from check_cuda_readiness import build_cuda_readiness, write_readiness_files

        report = build_cuda_readiness(artifacts_dir)
        write_readiness_files(report, out_json, out_md)
    except Exception as exc:
        warning_path = artifacts_dir / "cuda_readiness_error.txt"
        warning_path.write_text("Failed to render CUDA readiness:\n" + repr(exc), encoding="utf-8")


def validation_expected_rows(args: argparse.Namespace) -> int:
    if int(args.validation_expected_rows or 0) > 0:
        return int(args.validation_expected_rows)
    if args.score_only:
        return 0
    return int(args.target_eval_samples)


def validation_min_candidates(args: argparse.Namespace) -> int | None:
    if args.validation_min_candidates is not None:
        return int(args.validation_min_candidates)
    if args.score_only:
        return 0
    return None


def validate_reused_jsonl(
    path: Path,
    expected_rows: int,
    min_candidates: int,
    require_complete: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": f"invalid_json: {exc}"})
                continue
            if not isinstance(row, dict):
                errors.append({"line": line_number, "error": "row_not_object"})
                continue
            rows.append(row)

    if require_complete and len(rows) != expected_rows:
        errors.append({"error": "row_count", "expected": expected_rows, "observed": len(rows)})

    seen_indices: set[int] = set()
    duplicate_indices: list[int] = []
    for position, row in enumerate(rows):
        index = safe_int(row.get("index"), default=-1)
        if index < 0:
            errors.append({"row": position, "error": "missing_or_invalid_index", "index": row.get("index")})
            continue
        if index in seen_indices:
            duplicate_indices.append(index)
        seen_indices.add(index)
        if require_complete and not (0 <= index < expected_rows):
            errors.append({"row": position, "index": index, "error": "index_out_of_range"})

        target = str(row.get("real password", row.get("real_password", row.get("password", ""))))
        if not target:
            errors.append({"row": position, "index": index, "error": "missing_target"})

        candidates = row.get("outputPasswords")
        if not isinstance(candidates, list):
            errors.append({"row": position, "index": index, "error": "candidate_list_invalid"})
            continue
        passwords = [candidate_password_text(item) for item in candidates]
        nonempty_passwords = [password for password in passwords if password]
        unique_nonempty = set(nonempty_passwords)
        if min_candidates and len(nonempty_passwords) != len(passwords):
            errors.append({"row": position, "index": index, "error": "empty_candidate"})
        if min_candidates and len(unique_nonempty) < min_candidates:
            errors.append(
                {
                    "row": position,
                    "index": index,
                    "error": "candidate_budget",
                    "min_candidates": min_candidates,
                    "unique_nonempty_candidates": len(unique_nonempty),
                }
            )
        if len(unique_nonempty) != len(nonempty_passwords):
            errors.append({"row": position, "index": index, "error": "duplicate_candidates"})
        model_input = str(row.get("model_input", ""))
        if model_input and any(password.startswith(model_input) for password in nonempty_passwords):
            errors.append({"row": position, "index": index, "error": "prompt_leak"})
        observed_rank = safe_int(row.get("min_cracked_guess_number"), default=0)
        recomputed_rank = rank_from_jsonl_row(row)
        if observed_rank != recomputed_rank:
            errors.append(
                {
                    "row": position,
                    "index": index,
                    "error": "rank_mismatch",
                    "observed": observed_rank,
                    "recomputed": recomputed_rank,
                }
            )

    if duplicate_indices:
        errors.append({"error": "duplicate_indices", "indices": sorted(set(duplicate_indices))[:20]})
    if require_complete and expected_rows > 0:
        missing_indices = sorted(set(range(expected_rows)).difference(seen_indices))
        if missing_indices:
            errors.append({"error": "missing_indices", "indices": missing_indices[:20], "count": len(missing_indices)})
    elif expected_rows > 0 and len(rows) != expected_rows:
        warnings.append({"warning": "partial_row_count", "expected": expected_rows, "observed": len(rows)})

    return {
        "status": "failed" if errors else "passed",
        "path": str(path),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "min_candidates": min_candidates,
        "require_complete": require_complete,
        "unique_indices": len(seen_indices),
        "errors": errors[:100],
        "num_errors": len(errors),
        "warnings": warnings[:100],
        "num_warnings": len(warnings),
    }


def candidate_password_text(item: Any) -> str:
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    if isinstance(item, dict):
        return str(item.get("password", item.get("candidate", "")))
    return str(item)


def rank_from_jsonl_row(row: dict[str, Any]) -> int:
    target = str(row.get("real password", row.get("real_password", row.get("password", ""))))
    candidates = row.get("outputPasswords")
    if not target or not isinstance(candidates, list):
        return 0
    for index, item in enumerate(candidates, start=1):
        if candidate_password_text(item) == target:
            return index
    return 0


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compare_against_baseline(score_path: Path, baseline_metrics: dict[str, Any], budgets: list[int]) -> dict[str, Any]:
    score = load_json(score_path)
    deltas: dict[str, Any] = {}
    for budget in budgets:
        score_key = f"sr@{budget}"
        baseline_key = f"sr{budget}"
        observed = float(score.get(score_key, 0.0))
        baseline = float(baseline_metrics.get(baseline_key, 0.0))
        deltas[score_key] = {
            "baseline": baseline,
            "observed": observed,
            "delta": observed - baseline,
            "better_or_equal": observed >= baseline,
        }
    primary = deltas.get("sr@100") or deltas[f"sr@{budgets[-1]}"]
    return {
        "score_path": str(score_path),
        "num_rows": score.get("num_rows"),
        "rank_source": score.get("rank_source"),
        "deltas": deltas,
        "primary_verdict": "better_or_equal" if primary["better_or_equal"] else "below_baseline",
    }


def validate_complete_score(score_path: Path, expected_rows: int) -> None:
    score = load_json(score_path)
    actual_rows = int(score.get("num_rows", 0) or 0)
    if actual_rows != int(expected_rows):
        raise SystemExit(
            f"Scored JSONL row count is incomplete for formal execute mode: "
            f"expected {expected_rows}, got {actual_rows}. "
            "Use --resume-generation to finish generation, --target-eval-samples to match the intended row count, "
            "or --allow-partial-jsonl for diagnostics only."
        )


def write_summary(
    path: Path,
    status: str,
    preflight: dict[str, Any],
    manifest: dict[str, Any],
    comparisons: dict[str, Any] | None,
) -> None:
    lines = [
        "# Formal PassMoE Run Summary",
        "",
        f"- status: `{status}`",
        f"- run name: `{manifest['run_name']}`",
        f"- repo root: `{manifest.get('repo_root', '')}`",
        f"- device: `{manifest['device']}`",
        f"- torch: `{manifest['torch_version']}`",
        f"- cuda available: `{manifest['cuda_available']}`",
        f"- expected JSONL: `{manifest['expected_jsonl']}`",
        f"- baseline variant: `{manifest['baseline_variant']}`",
        f"- train max length: `{manifest.get('max_length', '')}`",
        f"- generation max new tokens: `{manifest.get('generation_max_new_tokens', '')}`",
        f"- generation batch size: `{manifest.get('generation_batch_size', '')}`",
        f"- seed: `{manifest.get('seed', '')}`",
        f"- needs model execution: `{manifest.get('needs_model_execution', True)}`",
        f"- skip existing JSONL: `{manifest.get('skip_existing_jsonl', False)}`",
        f"- environment snapshot: `{manifest.get('environment_snapshot_path', '')}`",
        f"- CUDA readiness: `{manifest.get('cuda_readiness_path', '')}`",
        f"- CUDA launcher: `{manifest.get('cuda_launcher_ps1', '')}`",
        "",
        "## Commands",
        "",
        "```powershell",
        *[ps_command(command) for command in manifest_commands(manifest)],
        "```",
        "",
        "## Preflight",
        "",
    ]
    if manifest.get("resume_from"):
        lines.insert(9, f"- resume from: `{manifest['resume_from']}`")
    if manifest.get("checkpoint"):
        lines.insert(9, f"- checkpoint: `{manifest['checkpoint']}`")
    for check in preflight["checks"]:
        if check.get("severity") == "warning" and not check["ok"]:
            marker = "WARN"
        else:
            marker = "PASS" if check["ok"] else "FAIL"
        lines.append(f"- {marker}: `{check['name']}` - {check['detail']}")
    if preflight.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in preflight["warnings"]:
            lines.append(f"- {warning}")
    if comparisons:
        for label in ("raw", "fused"):
            if label not in comparisons:
                continue
            comparison = comparisons[label]
            title = "Raw Comparison" if label == "raw" else "Fused Comparison"
            lines.extend(["", f"## {title}", ""])
            lines.append("| Metric | Baseline | Observed | Delta | Verdict |")
            lines.append("|---|---:|---:|---:|---|")
            for metric, item in comparison["deltas"].items():
                verdict = "better_or_equal" if item["better_or_equal"] else "below"
                lines.append(
                    f"| {metric} | {item['baseline']:.4f} | {item['observed']:.4f} | {item['delta']:+.4f} | {verdict} |"
                )
        if comparisons.get("fusion_analysis_path"):
            lines.extend(["", f"- fusion analysis: `{comparisons['fusion_analysis_path']}`"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_commands_ps1(path: Path, manifest: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# Generated formal PassMoE commands.",
            "Set-StrictMode -Version Latest",
            *[ps_command(command) for command in manifest_commands(manifest)],
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_cuda_launcher_ps1(path: Path, manifest: dict[str, Any]) -> None:
    run_command = build_runner_command_from_manifest(manifest)
    artifacts_arg = artifacts_dir_arg(manifest)
    status_command = ["python", "scripts/inspect_formal_status.py", "--artifacts-dir", artifacts_arg]
    report_command = ["python", "scripts/render_formal_report.py", "--artifacts-dir", artifacts_arg]
    text = "\n".join(
        [
            "# One-command CUDA entrypoint for the formal PassMoE run.",
            "# Run this from PowerShell on a CUDA host with the required Python environment active.",
            "# It intentionally calls scripts/run_formal_passmoe.py rather than the expanded child commands",
            "# so preflight, logging, validation, reporting, and recovery gates stay enabled.",
            "Set-StrictMode -Version Latest",
            '$ErrorActionPreference = "Stop"',
            "$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\\..\\..')",
            "Set-Location $RepoRoot",
            "",
            ps_command(run_command),
            ps_command(status_command),
            ps_command(report_command),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def build_runner_command_from_manifest(manifest: dict[str, Any]) -> list[str]:
    execution_command = manifest.get("execution_command") or []
    fusion_config = manifest.get("fusion_config") or {}
    command = [
        "python",
        "scripts/run_formal_passmoe.py",
        "--execute",
        "--run-name",
        str(manifest.get("run_name", "qwen_fielddrop_passmoe_clixsense_10k")),
        "--artifacts-dir",
        artifacts_parent_arg(manifest),
        "--baseline-contract",
        repo_relative_arg(manifest.get("baseline_contract", DEFAULT_BASELINE_CONTRACT)),
        "--baseline-variant",
        str(manifest.get("baseline_variant", "fd500k_p00_unique")),
        "--base-model",
        str(manifest.get("base_model", "local-qwen")),
        "--base-adapter",
        str(manifest.get("base_adapter", "fielddrop")),
        "--prompt-template-id",
        command_option(execution_command, "--prompt-template-id", "0"),
        "--epochs",
        command_option(execution_command, "--epochs", "3"),
        "--batch-size",
        command_option(execution_command, "--batch-size", "8"),
        "--max-train-samples",
        str(manifest.get("max_train_samples", 10000)),
        "--max-eval-samples",
        str(manifest.get("max_eval_samples", 0)),
        "--max-length",
        str(manifest.get("max_length", 256)),
        "--generation-max-new-tokens",
        str(manifest.get("generation_max_new_tokens", 32)),
        "--generation-batch-size",
        str(manifest.get("generation_batch_size", 32)),
        "--lora-rank",
        command_option(execution_command, "--lora-rank", "16"),
        "--beam-width",
        command_option(execution_command, "--beam-width", "100"),
        "--seed",
        str(manifest.get("seed", 42)),
        "--target-eval-samples",
        command_option(execution_command, "--target-eval-samples", str((manifest.get("data_counts") or {}).get("target_eval_samples", 500))),
        "--target-candidates-per-user",
        command_option(execution_command, "--target-candidates-per-user", "100"),
        "--budgets",
        ",".join(str(item) for item in manifest.get("budgets", [1, 10, 50, 100])),
        "--device",
        "cuda" if manifest.get("needs_model_execution", True) else str(manifest.get("device", "auto")),
        "--dtype",
        str(manifest.get("requested_dtype", "auto")),
        "--data-path",
        repo_relative_arg(manifest.get("data_path", "")),
        "--test-data-path",
        repo_relative_arg(manifest.get("test_data_path", "")),
        "--fusion-max-expert-candidates",
        str(fusion_config.get("max_expert_candidates", 80)),
        "--fusion-score-existing-weight",
        str(fusion_config.get("score_existing_weight", 1.0)),
        "--fusion-score-expert-weight",
        str(fusion_config.get("score_expert_weight", 0.05)),
        "--fusion-score-rank-offset",
        str(fusion_config.get("score_rank_offset", 2.0)),
        "--fusion-bootstrap-iters",
        str(fusion_config.get("bootstrap_iters", 2000)),
    ]
    output_dir = command_option(execution_command, "--output-dir", "runs")
    if output_dir != "runs":
        command.extend(["--output-dir", output_dir])
    if manifest.get("use_device_map"):
        command.append("--use-device-map")
    if not manifest.get("length_audit", True):
        command.append("--skip-length-audit")
    if not manifest.get("deep_model_check", True):
        command.append("--skip-deep-model-check")
    if not manifest.get("post_fusion", True):
        command.append("--no-post-fusion")
    if not manifest.get("output_validation", True):
        command.append("--skip-output-validation")
    if not manifest.get("result_report", True):
        command.append("--skip-result-report")
    if manifest.get("allow_partial_jsonl"):
        command.append("--allow-partial-jsonl")
    if manifest.get("resume_from"):
        command.extend(["--resume-from", repo_relative_arg(manifest.get("resume_from", ""))])
    if manifest.get("checkpoint"):
        command.extend(["--checkpoint", repo_relative_arg(manifest.get("checkpoint", ""))])
    if manifest.get("resume_generation"):
        command.append("--resume-generation")
    validation_expected_rows = int(manifest.get("validation_expected_rows", 0) or 0)
    target_eval_samples = int(command_option(execution_command, "--target-eval-samples", "0") or 0)
    if validation_expected_rows and validation_expected_rows != target_eval_samples:
        command.extend(["--validation-expected-rows", str(validation_expected_rows)])
    if manifest.get("validation_min_candidates") is not None:
        command.extend(["--validation-min-candidates", str(manifest["validation_min_candidates"])])
    return command


def artifacts_parent_arg(manifest: dict[str, Any]) -> str:
    artifacts_dir = Path(str(manifest.get("artifacts_dir", REPO_ROOT / "artifacts" / "formal" / manifest.get("run_name", ""))))
    return repo_relative_arg(str(artifacts_dir.parent))


def artifacts_dir_arg(manifest: dict[str, Any]) -> str:
    return repo_relative_arg(manifest.get("artifacts_dir", "artifacts/formal/qwen_fielddrop_passmoe_clixsense_10k"))


def repo_relative_arg(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    path = Path(text)
    try:
        if path.is_absolute():
            return str(path.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return text
    return text


def command_option(command: list[str], option: str, default: str) -> str:
    try:
        index = command.index(option)
    except ValueError:
        return str(default)
    next_index = index + 1
    if next_index >= len(command):
        return str(default)
    return str(command[next_index])


def manifest_commands(manifest: dict[str, Any]) -> list[list[str]]:
    commands = []
    length_audit_commands = manifest.get("length_audit_commands") or []
    if length_audit_commands:
        commands.extend(length_audit_commands)
    else:
        length_audit_command = manifest.get("length_audit_command") or []
        if length_audit_command:
            commands.append(length_audit_command)
    execution_command = manifest.get("execution_command") or []
    if execution_command:
        commands.append(execution_command)
    commands.append(manifest["score_command"])
    for key in ("fuse_command", "fused_score_command", "fusion_analysis_command"):
        command = manifest.get(key) or []
        if command:
            commands.append(command)
    validation_command = manifest.get("validation_command") or []
    if validation_command:
        commands.append(validation_command)
    return commands


def run_command(command: list[str], quiet: bool = False) -> None:
    global COMMAND_LOG_INDEX
    COMMAND_LOG_INDEX += 1
    log_path = None
    if COMMAND_LOG_DIR is not None:
        COMMAND_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = COMMAND_LOG_DIR / f"{COMMAND_LOG_INDEX:02d}_{command_log_stem(command)}.log"

    started_at = datetime.now(timezone.utc).isoformat()
    if log_path is not None:
        log_path.write_text(
            "\n".join(
                [
                    f"started_at_utc: {started_at}",
                    f"cwd: {REPO_ROOT}",
                    f"command: {ps_command(command)}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    with (log_path.open("a", encoding="utf-8", errors="replace") if log_path is not None else null_text_sink()) as handle:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if not quiet:
                write_console(line)
        return_code = process.wait()
        finished_at = datetime.now(timezone.utc).isoformat()
        handle.write(f"\nfinished_at_utc: {finished_at}\n")
        handle.write(f"return_code: {return_code}\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def configure_command_logging(log_dir: Path) -> None:
    global COMMAND_LOG_DIR, COMMAND_LOG_INDEX
    COMMAND_LOG_DIR = log_dir
    COMMAND_LOG_INDEX = 0
    COMMAND_LOG_DIR.mkdir(parents=True, exist_ok=True)


def command_log_stem(command: list[str]) -> str:
    if len(command) >= 3 and Path(command[1]).name == "main.py":
        stem = command[2]
    elif len(command) >= 2:
        stem = Path(command[1]).stem
    elif command:
        stem = Path(command[0]).stem
    else:
        stem = "command"
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return safe[:60] or "command"


class null_text_sink:
    def __enter__(self) -> "null_text_sink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def write(self, _text: str) -> None:
        return None

    def flush(self) -> None:
        return None


def write_console(text: str) -> None:
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(text.encode(encoding, errors="replace"))
        sys.stdout.buffer.flush()


def resolve_device(requested: str, cuda_available: bool, allow_cpu: bool) -> str:
    if requested == "auto":
        if cuda_available:
            return "cuda"
        return "cpu" if allow_cpu else "cuda"
    return requested


def resolve_run_dtype(requested: str, device: str, cuda_available: bool) -> str:
    requested = str(requested).lower()
    if requested != "auto":
        return requested
    if str(device).startswith("cuda"):
        if cuda_available:
            try:
                import torch

                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return "bfloat16"
            except Exception:
                pass
            return "float16"
        return "bfloat16"
    return "float32"


def classify_status(
    preflight: dict[str, Any],
    cuda_available: bool,
    allow_cpu: bool,
    needs_model_execution: bool,
) -> str:
    if preflight["errors"]:
        return "preflight_failed"
    if needs_model_execution and not cuda_available and not allow_cpu:
        return "ready_needs_cuda"
    return "ready"


def detect_torch() -> tuple[bool, str]:
    try:
        import torch

        return bool(torch.cuda.is_available()), str(torch.__version__)
    except Exception:
        return False, "unavailable"


def can_import(module_name: str) -> tuple[bool, str]:
    try:
        __import__(module_name)
        return True, "available"
    except Exception as exc:
        return False, repr(exc)


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def count_data_records(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
            return sum(1 for line in handle if line.strip())
    if suffix == ".json":
        payload = load_json(path)
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            for key in ("data", "records", "items", "examples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return len(value)
            return 1
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_budgets(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ps_command(command: list[str]) -> str:
    return " ".join(ps_quote(part) for part in command)


def ps_quote(value: str) -> str:
    value = str(value)
    if not value or any(char.isspace() for char in value) or any(char in value for char in "'&()[]{};"):
        return "'" + value.replace("'", "''") + "'"
    return value


if __name__ == "__main__":
    main()
