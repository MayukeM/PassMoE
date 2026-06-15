from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_RUN_NAME = "qwen_fielddrop_base_identity_clixsense_500_raw"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "formal" / DEFAULT_FORMAL_RUN_NAME
PROGRESS_MARKER_PREFIX = "__PASSMOE_PROGRESS__"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect formal PassMoE artifacts and recommend the next recovery command.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    report = inspect_formal_status(resolve_path(args.artifacts_dir))
    text = json.dumps(report, indent=2)
    if args.out:
        out_path = resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


def inspect_formal_status(artifacts_dir: Path) -> dict[str, Any]:
    manifest_path = artifacts_dir / "run_manifest.json"
    preflight_path = artifacts_dir / "preflight.json"
    validation_path = artifacts_dir / "formal_validation.json"
    score_path = artifacts_dir / "score.json"
    comparison_path = artifacts_dir / "comparison.json"
    fused_score_path = artifacts_dir / "fused_score.json"
    fused_comparison_path = artifacts_dir / "fused_comparison.json"
    fusion_analysis_path = artifacts_dir / "fusion_analysis.json"

    manifest = load_json_if_exists(manifest_path)
    preflight = load_json_if_exists(preflight_path)
    validation = load_json_if_exists(validation_path)
    score = load_json_if_exists(score_path)

    environment_snapshot_path = manifest_repo_path(manifest, "environment_snapshot_path", artifacts_dir / "environment_snapshot.json")
    cuda_readiness_path = manifest_repo_path(manifest, "cuda_readiness_path", artifacts_dir / "cuda_readiness.json")
    expected_jsonl = manifest_repo_path(manifest, "expected_jsonl")
    expected_fused_jsonl = manifest_repo_path(manifest, "expected_fused_jsonl")
    run_dir = manifest_repo_path(manifest, "run_dir")
    expected_rows = infer_expected_rows(manifest, validation)
    raw_rows = count_jsonl_rows(expected_jsonl) if expected_jsonl else None
    fused_rows = count_jsonl_rows(expected_fused_jsonl) if expected_fused_jsonl else None
    logs_dir = manifest_repo_path(manifest, "command_logs_dir", artifacts_dir / "logs")
    logs = inspect_logs(logs_dir)
    targeted_generation_progress = inspect_progress_markers(logs_dir)
    checkpoints = inspect_checkpoints(run_dir)
    validation_hash_status = inspect_validation_hashes(validation, manifest)
    model_execution_provenance = inspect_model_execution_provenance(manifest, expected_jsonl, expected_rows)
    manifest_path_audit = inspect_manifest_path_audit(manifest, artifacts_dir)

    files = {
        "manifest": path_info(manifest_path),
        "environment_snapshot": path_info(environment_snapshot_path),
        "cuda_readiness": path_info(cuda_readiness_path),
        "preflight": path_info(preflight_path),
        "expected_jsonl": path_info(expected_jsonl),
        "score": path_info(score_path),
        "comparison": path_info(comparison_path),
        "fused_jsonl": path_info(expected_fused_jsonl),
        "fused_score": path_info(fused_score_path),
        "fused_comparison": path_info(fused_comparison_path),
        "fusion_analysis": path_info(fusion_analysis_path),
        "formal_validation": path_info(validation_path),
    }

    status, recommendation = classify_status(
        artifacts_dir=artifacts_dir,
        manifest=manifest,
        preflight=preflight,
        validation=validation,
        score=score,
        expected_rows=expected_rows,
        raw_rows=raw_rows,
        fused_rows=fused_rows,
        files=files,
        checkpoints=checkpoints,
        validation_hash_status=validation_hash_status,
        model_execution_provenance=model_execution_provenance,
        manifest_path_audit=manifest_path_audit,
    )
    validation_file_status = validation.get("status") if validation else None
    validation_status = effective_validation_status(status, validation_file_status)

    return {
        "status": status,
        "artifacts_dir": str(artifacts_dir),
        "run_name": manifest.get("run_name", artifacts_dir.name) if manifest else artifacts_dir.name,
        "expected_rows": expected_rows,
        "raw_rows": raw_rows,
        "fused_rows": fused_rows,
        "preflight_status": preflight.get("status") if preflight else None,
        "validation_status": validation_status,
        "validation_file_status": validation_file_status,
        "score_rank_source": score.get("rank_source") if score else None,
        "files": files,
        "checkpoints": checkpoints,
        "logs": logs,
        "targeted_generation_progress": targeted_generation_progress,
        "validation_hash_status": validation_hash_status,
        "model_execution_provenance": model_execution_provenance,
        "manifest_path_audit": manifest_path_audit,
        "recommendation": recommendation,
    }


def classify_status(
    artifacts_dir: Path,
    manifest: dict[str, Any],
    preflight: dict[str, Any],
    validation: dict[str, Any],
    score: dict[str, Any],
    expected_rows: int,
    raw_rows: int | None,
    fused_rows: int | None,
    files: dict[str, dict[str, Any]],
    checkpoints: dict[str, dict[str, Any]],
    validation_hash_status: dict[str, Any],
    model_execution_provenance: dict[str, Any],
    manifest_path_audit: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    run_name = manifest.get("run_name", artifacts_dir.name) if manifest else artifacts_dir.name
    if not manifest:
        return "missing_manifest", {"reason": "run_manifest.json is missing", "command": "python scripts/run_formal_passmoe.py"}
    if manifest_path_audit.get("status") == "stale":
        return "stale_manifest_paths", {
            "reason": manifest_path_audit.get("reason", "run manifest was generated under a different repo root"),
            "command": build_run_formal_command(run_name, manifest, execute=False),
        }
    if preflight and preflight.get("status") != "passed":
        return "preflight_failed", {
            "reason": "; ".join(str(item) for item in preflight.get("errors", [])) or str(preflight.get("status")),
            "command": build_run_formal_command(run_name, manifest, execute=False),
        }

    if raw_rows is None:
        command = build_run_formal_command(run_name, manifest)
        if checkpoints.get("last", {}).get("exists"):
            command = build_run_formal_command(run_name, manifest, "--resume-from", str(checkpoints["last"]["path"]))
        return "needs_model_execution", {"reason": "targeted_input_output.jsonl is missing", "command": command}

    if expected_rows and raw_rows != expected_rows:
        checkpoint = checkpoints.get("best", {}) if checkpoints.get("best", {}).get("exists") else checkpoints.get("last", {})
        extras = ["--resume-generation"]
        if checkpoint.get("exists"):
            extras.extend(["--checkpoint", str(checkpoint["path"])])
        command = build_run_formal_command(run_name, manifest, *extras)
        status = "partial_generation" if raw_rows < expected_rows else "row_count_mismatch"
        return status, {
            "reason": f"raw JSONL has {raw_rows}/{expected_rows} rows",
            "command": command,
        }

    post_fusion_enabled = bool(manifest.get("post_fusion", True))
    required_post = ["score", "comparison"]
    if post_fusion_enabled:
        required_post.extend(["fused_jsonl", "fused_score", "fused_comparison", "fusion_analysis"])
    missing_post = [name for name in required_post if not files.get(name, {}).get("exists")]
    if missing_post:
        return "needs_postprocess", {
            "reason": f"missing postprocess artifacts: {missing_post}",
            "command": build_run_formal_command(run_name, manifest, "--skip-train-if-jsonl-exists", "--force"),
        }
    if post_fusion_enabled and expected_rows and fused_rows is not None and fused_rows != expected_rows:
        return "postprocess_row_mismatch", {
            "reason": f"fused JSONL has {fused_rows}/{expected_rows} rows",
            "command": build_run_formal_command(run_name, manifest, "--skip-train-if-jsonl-exists", "--force"),
        }

    if not files.get("formal_validation", {}).get("exists"):
        return "needs_validation", {
            "reason": "postprocess artifacts exist but formal_validation.json is missing",
            "command": f"python scripts/validate_formal_outputs.py --artifacts-dir {artifacts_dir}",
        }

    if validation and validation.get("status") == "passed":
        if validation_hash_status.get("status") == "missing":
            return "needs_validation", {
                "reason": validation_hash_status.get("reason", "formal_validation.json lacks artifact hashes"),
                "command": f"python scripts/validate_formal_outputs.py --artifacts-dir {artifacts_dir}",
            }
        if validation_hash_status.get("status") != "passed":
            return "validation_stale", {
                "reason": validation_hash_status.get("reason", "current artifacts do not match formal_validation.json hashes"),
                "command": f"python scripts/validate_formal_outputs.py --artifacts-dir {artifacts_dir}",
            }
        if model_execution_provenance.get("required") and model_execution_provenance.get("status") != "passed":
            return "model_execution_unverified", {
                "reason": model_execution_provenance.get("reason", "missing PassMoE generation provenance"),
                "command": build_run_formal_command(run_name, manifest),
            }
        return "complete", {"reason": "formal_validation.json passed and required artifacts are present", "command": "none"}

    if validation and validation.get("status") != "passed":
        return "validation_failed", {
            "reason": "; ".join(str(item) for item in validation.get("errors", [])[:5]) or str(validation.get("status")),
            "command": f"python scripts/validate_formal_outputs.py --artifacts-dir {artifacts_dir}",
        }

    return "unknown", {"reason": "artifact state did not match a known recovery path", "command": "inspect logs and summary.md"}


def effective_validation_status(status: str, validation_file_status: str | None) -> str:
    if status in {"needs_model_execution", "partial_generation", "row_count_mismatch", "needs_postprocess", "postprocess_row_mismatch"}:
        return "not_applicable_until_execution_complete"
    if status == "validation_stale":
        return "stale"
    if status == "model_execution_unverified":
        return "unverified_model_execution"
    if status == "needs_validation":
        return "missing"
    if validation_file_status is None:
        return "missing"
    return validation_file_status


def build_run_formal_command(run_name: str, manifest: dict[str, Any], *extra_args: str, execute: bool = True) -> str:
    command = ["python", "scripts/run_formal_passmoe.py"]
    if execute:
        command.append("--execute")
    command.extend(["--run-name", run_name])
    seed = manifest.get("seed") if manifest else None
    if seed not in (None, ""):
        command.extend(["--seed", str(seed)])
    command.extend(str(item) for item in extra_args)
    return format_command(command)


def format_command(args: list[str]) -> str:
    return " ".join(quote_command_arg(arg) for arg in args)


def quote_command_arg(arg: str) -> str:
    if not arg:
        return '""'
    if any(char.isspace() for char in arg) or any(char in arg for char in '"`'):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def inspect_validation_hashes(validation: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    if not validation:
        return {"status": "missing", "reason": "formal_validation.json is missing"}
    hashes = validation.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        return {"status": "missing", "reason": "formal_validation.json lacks artifact_hashes"}
    mismatches = []
    missing = []
    for name, expected in hashes.items():
        if not isinstance(expected, dict):
            mismatches.append({"artifact": name, "error": "invalid_hash_record"})
            continue
        if not expected.get("exists", False):
            continue
        path = relocate_manifest_path(expected.get("path", ""), manifest)
        if not path.exists():
            missing.append({"artifact": name, "path": str(path)})
            continue
        observed = file_sha256(path)
        expected_hash = str(expected.get("sha256", ""))
        if observed != expected_hash:
            mismatches.append(
                {
                    "artifact": name,
                    "path": str(path),
                    "expected": expected_hash,
                    "observed": observed,
                }
            )
    if missing or mismatches:
        return {
            "status": "failed",
            "reason": f"artifact hash mismatch: missing={len(missing)}, mismatched={len(mismatches)}",
            "missing": missing[:20],
            "mismatches": mismatches[:20],
        }
    return {"status": "passed", "checked": len(hashes)}


def inspect_manifest_path_audit(manifest: dict[str, Any], artifacts_dir: Path) -> dict[str, Any]:
    current_repo_root = normalize_path_text(REPO_ROOT)
    if not manifest:
        return {
            "status": "missing",
            "current_repo_root": str(REPO_ROOT),
            "manifest_repo_root": "",
            "repo_root_matches": False,
            "reason": "run_manifest.json is missing",
        }

    manifest_repo_root = str(manifest.get("repo_root", ""))
    manifest_repo_root_norm = normalize_path_text(Path(manifest_repo_root)) if manifest_repo_root else ""
    repo_root_matches = bool(manifest_repo_root_norm and manifest_repo_root_norm == current_repo_root)
    relocated_artifacts_dir = relocate_manifest_path(manifest.get("artifacts_dir", ""), manifest)
    artifacts_dir_matches = path_text_matches(str(relocated_artifacts_dir), artifacts_dir)
    repo_owned_mismatches = repo_owned_path_mismatches(manifest, current_repo_root)
    status = "passed" if artifacts_dir_matches and not repo_owned_mismatches else "stale"
    reason = "manifest paths match current repo root" if repo_root_matches else "manifest repo-owned paths are relocatable to current repo root"
    if status == "stale":
        reason = (
            f"manifest repo_root={manifest_repo_root or 'missing'}; "
            f"current repo_root={REPO_ROOT}; repo_owned_mismatches={len(repo_owned_mismatches)}"
        )
    return {
        "status": status,
        "reason": reason,
        "current_repo_root": str(REPO_ROOT),
        "manifest_repo_root": manifest_repo_root,
        "repo_root_matches": repo_root_matches,
        "artifacts_dir_matches": artifacts_dir_matches,
        "relocated_artifacts_dir": str(relocated_artifacts_dir),
        "repo_owned_mismatches": repo_owned_mismatches[:50],
    }


def repo_owned_path_mismatches(manifest: dict[str, Any], current_repo_root: str) -> list[dict[str, str]]:
    repo_keys = {
        "run_dir",
        "data_path",
        "test_data_path",
        "artifacts_dir",
        "baseline_contract",
        "command_logs_dir",
        "environment_snapshot_path",
        "cuda_readiness_path",
        "cuda_readiness_md_path",
        "length_audit_path",
        "deep_model_check_path",
        "result_report_md",
        "result_report_json",
        "cuda_launcher_ps1",
        "expected_jsonl",
        "expected_fused_jsonl",
        "reused_jsonl_quality_path",
    }
    mismatches: list[dict[str, str]] = []
    for key in sorted(repo_keys):
        value = manifest.get(key)
        if isinstance(value, str) and looks_like_absolute_path(value):
            relocated = relocate_manifest_path(value, manifest)
            if relocated and not normalize_path_text(relocated).startswith(current_repo_root):
                mismatches.append({"key": key, "path": value, "relocated": str(relocated)})
    for group_key in ("length_audit_paths",):
        group = manifest.get(group_key, {})
        if not isinstance(group, dict):
            continue
        for key, value in group.items():
            if isinstance(value, str) and looks_like_absolute_path(value):
                relocated = relocate_manifest_path(value, manifest)
                if relocated and not normalize_path_text(relocated).startswith(current_repo_root):
                    mismatches.append({"key": f"{group_key}.{key}", "path": value, "relocated": str(relocated)})
    return mismatches


def path_text_matches(left: Any, right: Path) -> bool:
    if not isinstance(left, str) or not left:
        return False
    return normalize_path_text(Path(left)) == normalize_path_text(right)


def looks_like_absolute_path(value: str) -> bool:
    if not value:
        return False
    return Path(value).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("/")


def normalize_path_text(path: Path) -> str:
    try:
        path = path.resolve()
    except OSError:
        pass
    return str(path).rstrip("\\/").casefold()


def inspect_model_execution_provenance(
    manifest: dict[str, Any],
    expected_jsonl: Path,
    expected_rows: int,
) -> dict[str, Any]:
    if not manifest:
        return {"status": "missing", "required": False, "reason": "run_manifest.json is missing"}
    baseline_n = 0
    try:
        baseline_n = int((manifest.get("baseline_metrics") or {}).get("n", 0) or 0)
    except (TypeError, ValueError):
        baseline_n = 0
    is_score_only = manifest.get("mode") == "score_only"
    is_cpu_diagnostic = manifest.get("device") == "cpu"
    is_subset = bool(baseline_n and expected_rows and expected_rows != baseline_n)
    required = bool(not is_score_only and not is_cpu_diagnostic and not is_subset)
    run_dir = manifest_repo_path(manifest, "run_dir")
    metrics_path = run_dir / "targeted_generation_metrics.json" if run_dir else Path()
    if not required:
        return {
            "status": "not_required",
            "required": False,
            "reason": "score-only, CPU, or subset diagnostic artifact",
            "path": str(metrics_path) if metrics_path else "",
        }
    if not metrics_path.exists():
        return {
            "status": "missing",
            "required": True,
            "reason": "targeted_generation_metrics.json is missing for claim-carrying model output",
            "path": str(metrics_path),
        }
    metrics = load_json_if_exists(metrics_path)
    errors = []
    if metrics.get("complete") is not True:
        errors.append({"error": "generation_not_complete", "value": metrics.get("complete")})
    for key in ("num_targets", "num_completed"):
        try:
            observed = int(metrics.get(key, 0) or 0)
        except (TypeError, ValueError):
            observed = 0
        if expected_rows and observed != expected_rows:
            errors.append({"error": f"{key}_mismatch", "expected": expected_rows, "observed": observed})
    try:
        candidates_per_user = int(metrics.get("candidates_per_user", 0) or 0)
    except (TypeError, ValueError):
        candidates_per_user = 0
    budgets = [int(item) for item in manifest.get("budgets", []) if str(item).strip()]
    max_budget = max(budgets) if budgets else 0
    if max_budget and candidates_per_user < max_budget:
        errors.append({"error": "candidate_budget", "required": max_budget, "observed": candidates_per_user})
    result_path = relocate_manifest_path(metrics.get("result_path", ""), manifest)
    if result_path and expected_jsonl:
        try:
            if result_path.resolve() != expected_jsonl.resolve():
                errors.append({"error": "result_path_mismatch", "expected": str(expected_jsonl), "observed": str(result_path)})
        except OSError:
            errors.append({"error": "result_path_unresolvable", "expected": str(expected_jsonl), "observed": str(result_path)})
    if errors:
        return {
            "status": "failed",
            "required": True,
            "reason": f"targeted generation provenance failed {len(errors)} checks",
            "path": str(metrics_path),
            "errors": errors,
        }
    return {
        "status": "passed",
        "required": True,
        "path": str(metrics_path),
        "num_targets": metrics.get("num_targets"),
        "num_completed": metrics.get("num_completed"),
        "candidates_per_user": metrics.get("candidates_per_user"),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_expected_rows(manifest: dict[str, Any], validation: dict[str, Any]) -> int:
    try:
        value = int(validation.get("expected_rows", 0) or 0)
        if value:
            return value
    except (TypeError, ValueError):
        pass
    data_counts = manifest.get("data_counts", {}) if manifest else {}
    for key in ("target_eval_samples", "eval_rows"):
        try:
            value = int(data_counts.get(key, 0) or 0)
            if value:
                return value
        except (TypeError, ValueError):
            pass
    baseline_metrics = manifest.get("baseline_metrics", {}) if manifest else {}
    try:
        return int(baseline_metrics.get("n", 0) or 0)
    except (TypeError, ValueError):
        return 0


def inspect_checkpoints(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "best": path_info(run_dir / "best.pt") if run_dir else {"exists": False, "path": ""},
        "last": path_info(run_dir / "last.pt") if run_dir else {"exists": False, "path": ""},
    }


def inspect_logs(log_dir: Path) -> dict[str, Any]:
    if not log_dir or not log_dir.exists():
        return {"dir": str(log_dir), "exists": False, "files": []}
    files = []
    for path in sorted(log_dir.glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"return_code:\s*(-?\d+)", text)
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "return_code": int(match.group(1)) if match else None,
            }
        )
    return {"dir": str(log_dir), "exists": True, "files": files}


def inspect_progress_markers(log_dir: Path) -> dict[str, Any]:
    if not log_dir or not log_dir.exists():
        return {"status": "missing", "num_markers": 0, "marker_lines": 0, "latest": None, "source_log": ""}

    marker_lines = 0
    parsed_markers = 0
    parse_errors = []
    latest = None
    source_log = ""
    latest_line = None

    for path in sorted(log_dir.glob("*.log")):
        try:
            handle = path.open("r", encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            parse_errors.append({"path": str(path), "error": f"open_failed: {exc}"})
            continue
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if PROGRESS_MARKER_PREFIX not in line:
                    continue
                marker_lines += 1
                payload_text = line.split(PROGRESS_MARKER_PREFIX, 1)[1].strip()
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError as exc:
                    parse_errors.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "error": f"json_decode_failed: {exc.msg}",
                            "text": payload_text[:240],
                        }
                    )
                    continue
                parsed_markers += 1
                latest = payload
                source_log = str(path)
                latest_line = line_number

    status = "present" if latest is not None else "missing"
    result = {
        "status": status,
        "num_markers": parsed_markers,
        "marker_lines": marker_lines,
        "latest": latest,
        "source_log": source_log,
        "latest_line": latest_line,
        "parse_errors": len(parse_errors),
    }
    if parse_errors:
        result["parse_error_examples"] = parse_errors[:5]
    return result


def count_jsonl_rows(path: Path) -> int | None:
    if not path or not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def path_info(path: Path) -> dict[str, Any]:
    if not path or str(path) in {"", "."}:
        return {"exists": False, "path": ""}
    return {
        "exists": path.exists(),
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def manifest_repo_path(manifest: dict[str, Any], key: str, fallback: Path | None = None) -> Path:
    if not manifest:
        return fallback or Path()
    path = relocate_manifest_path(manifest.get(key, ""), manifest)
    if str(path) in {"", "."} and fallback is not None:
        return fallback
    return path


def relocate_manifest_path(value: Any, manifest: dict[str, Any]) -> Path:
    text = str(value or "")
    if not text:
        return Path()
    candidate = Path(text)
    if candidate.exists():
        return candidate
    relative = relative_to_manifest_root(text, str(manifest.get("repo_root", "")))
    if relative is not None:
        return (REPO_ROOT / Path(*relative.split("/"))).resolve()
    return candidate


def relative_to_manifest_root(path_text: str, root_text: str) -> str | None:
    if not path_text or not root_text:
        return None
    path_norm = path_text.replace("\\", "/").rstrip("/")
    root_norm = root_text.replace("\\", "/").rstrip("/")
    if not root_norm:
        return None
    path_lower = path_norm.casefold()
    root_lower = root_norm.casefold()
    if path_lower == root_lower:
        return ""
    prefix = root_lower + "/"
    if not path_lower.startswith(prefix):
        return None
    return path_norm[len(root_norm) + 1 :]


if __name__ == "__main__":
    main()
