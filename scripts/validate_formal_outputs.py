from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_RUN_NAME = "qwen_fielddrop_base_identity_clixsense_500_raw"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "formal" / DEFAULT_FORMAL_RUN_NAME
DEFAULT_BASELINE_CONTRACT = REPO_ROOT / "baselines" / "imported" / "passllm-fielddrop" / "json" / "metric_contract.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate completed formal PassMoE output artifacts.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--baseline-contract", default=str(DEFAULT_BASELINE_CONTRACT))
    parser.add_argument("--expected-baseline-variant", default="fd500k_p00_unique")
    parser.add_argument("--expected-rows", type=int, default=0, help="Default: baseline variant n.")
    parser.add_argument("--allow-baseline-row-override", action="store_true")
    parser.add_argument("--budgets", default="", help="Default: budgets from run_manifest.json.")
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=-1,
        help="Minimum raw candidates per row. Default: max budget; use 0 for diagnostics.",
    )
    parser.add_argument(
        "--require-fused",
        dest="require_fused",
        action="store_true",
        default=None,
        help="Require fused JSONL/score artifacts. Default: infer from run_manifest.json post_fusion.",
    )
    parser.add_argument(
        "--no-require-fused",
        dest="require_fused",
        action="store_false",
        help="Do not require fused artifacts; use for raw-only formal runs.",
    )
    parser.add_argument("--allow-fusion-worsening", action="store_true")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--no-fail-on-invalid", action="store_true")
    args = parser.parse_args()

    artifacts_dir = resolve_path(args.artifacts_dir)
    report = validate_formal_outputs(args, artifacts_dir)
    out_json = resolve_path(args.out_json) if args.out_json else artifacts_dir / "formal_validation.json"
    out_md = resolve_path(args.out_md) if args.out_md else artifacts_dir / "formal_validation.md"
    write_json(out_json, report)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(out_json), "summary": str(out_md)}, indent=2))
    if report["status"] != "passed" and not args.no_fail_on_invalid:
        raise SystemExit(1)


def validate_formal_outputs(args: argparse.Namespace, artifacts_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def check(name: str, ok: bool, detail: str, severity: str = "error") -> None:
        item = {"name": name, "ok": bool(ok), "detail": detail, "severity": severity}
        checks.append(item)
        if not ok:
            if severity == "warning":
                warnings.append(f"{name}: {detail}")
            else:
                errors.append(f"{name}: {detail}")

    baseline_contract_path = resolve_path(args.baseline_contract)
    baseline_contract = load_json_if_exists(baseline_contract_path)
    manifest_path = artifacts_dir / "run_manifest.json"
    preflight_path = artifacts_dir / "preflight.json"
    score_path = artifacts_dir / "score.json"
    comparison_path = artifacts_dir / "comparison.json"
    fused_score_path = artifacts_dir / "fused_score.json"
    fused_comparison_path = artifacts_dir / "fused_comparison.json"
    fusion_analysis_path = artifacts_dir / "fusion_analysis.json"

    manifest = load_json_if_exists(manifest_path)
    preflight = load_json_if_exists(preflight_path)
    score = load_json_if_exists(score_path)
    comparison = load_json_if_exists(comparison_path)
    fused_score = load_json_if_exists(fused_score_path)
    fused_comparison = load_json_if_exists(fused_comparison_path)
    fusion_analysis = load_json_if_exists(fusion_analysis_path)
    require_fused = infer_require_fused(args, manifest)

    check("artifacts_dir", artifacts_dir.exists(), str(artifacts_dir))
    check("manifest_exists", bool(manifest), str(manifest_path))
    check("preflight_exists", bool(preflight), str(preflight_path), severity="warning")
    check("score_exists", bool(score), str(score_path))
    check("comparison_exists", bool(comparison), str(comparison_path))
    if require_fused:
        check("fused_score_exists", bool(fused_score), str(fused_score_path))
        check("fused_comparison_exists", bool(fused_comparison), str(fused_comparison_path))
        check("fusion_analysis_exists", bool(fusion_analysis), str(fusion_analysis_path))

    manifest_path_audit = build_manifest_path_audit(manifest, artifacts_dir)
    if manifest:
        check(
            "manifest_repo_root_current",
            manifest_path_audit["status"] == "passed",
            manifest_path_audit["reason"],
        )

    variant_id = str(manifest.get("baseline_variant", "")) if manifest else ""
    baseline_variants = baseline_contract.get("baseline_variants", {}) if baseline_contract else {}
    baseline_metrics = baseline_variants.get(args.expected_baseline_variant, {})
    expected_rows = args.expected_rows or int(baseline_metrics.get("n", 0) or 0)
    budgets = parse_budgets(args.budgets) if args.budgets else [int(x) for x in manifest.get("budgets", [])] if manifest else []
    min_candidates = max(budgets) if args.min_candidates < 0 and budgets else max(args.min_candidates, 0)

    check("baseline_contract_exists", bool(baseline_contract), str(baseline_contract_path))
    check(
        "baseline_variant",
        variant_id == args.expected_baseline_variant,
        f"observed={variant_id}, expected={args.expected_baseline_variant}",
    )
    check(
        "baseline_variant_in_contract",
        args.expected_baseline_variant in baseline_variants,
        args.expected_baseline_variant,
    )
    if baseline_metrics:
        check(
            "baseline_rows",
            int(baseline_metrics.get("n", 0) or 0) == expected_rows,
            f"contract_n={baseline_metrics.get('n')}, expected_rows={expected_rows}",
            severity="warning" if args.allow_baseline_row_override else "error",
        )
    check("budgets_present", bool(budgets), str(budgets))
    check("expected_rows_present", expected_rows > 0, str(expected_rows))

    if preflight:
        check("preflight_status", preflight.get("status") == "passed", str(preflight.get("status")))
        preflight_errors = preflight.get("errors", [])
        check("preflight_errors_empty", len(preflight_errors) == 0, str(preflight_errors))
        if manifest.get("deep_model_check"):
            deep_summary = preflight.get("deep_model_check_summary", {})
            merge_report = deep_summary.get("merge_report", {})
            check(
                "deep_model_lora_merge",
                int(merge_report.get("merged_modules", 0) or 0) > 0
                and int(merge_report.get("skipped_modules", 0) or 0) == 0,
                f"merged={merge_report.get('merged_modules')}, skipped={merge_report.get('skipped_modules')}",
            )

    expected_jsonl = manifest_repo_path(manifest, "expected_jsonl")
    expected_fused_jsonl = manifest_repo_path(manifest, "expected_fused_jsonl")
    raw_rows = load_jsonl_if_exists(expected_jsonl)
    fused_rows = load_jsonl_if_exists(expected_fused_jsonl)
    validate_score("raw", score, comparison, baseline_metrics, budgets, expected_rows, check)
    if raw_rows is not None:
        validate_jsonl_rows("raw_jsonl", raw_rows, expected_rows, min_candidates, check)
    else:
        check("raw_jsonl_exists", False, str(expected_jsonl))

    if require_fused:
        validate_score("fused", fused_score, fused_comparison, baseline_metrics, budgets, expected_rows, check)
        if fused_rows is not None:
            validate_jsonl_rows("fused_jsonl", fused_rows, expected_rows, 0, check)
        else:
            check("fused_jsonl_exists", False, str(expected_fused_jsonl))
        if raw_rows is not None and fused_rows is not None:
            check("raw_fused_row_count_match", len(raw_rows) == len(fused_rows), f"{len(raw_rows)} vs {len(fused_rows)}")
        validate_fusion_analysis(
            fusion_analysis,
            budgets,
            expected_rows,
            allow_worsening=args.allow_fusion_worsening,
            check=check,
        )

    primary_metric = f"sr@{max(budgets) if budgets else 100}"
    raw_primary = score.get(primary_metric) if score else None
    fused_primary = fused_score.get(primary_metric) if fused_score else None
    baseline_primary = baseline_metrics.get(f"sr{max(budgets) if budgets else 100}") if baseline_metrics else None
    status = "passed" if not errors else "failed"
    return {
        "status": status,
        "artifacts_dir": str(artifacts_dir),
        "expected_rows": expected_rows,
        "baseline_variant": variant_id,
        "expected_baseline_variant": args.expected_baseline_variant,
        "budgets": budgets,
        "min_candidates": min_candidates,
        "require_fused": require_fused,
        "primary_metric": primary_metric,
        "primary": {
            "baseline": baseline_primary,
            "raw": raw_primary,
            "fused": fused_primary,
            "raw_delta": numeric_delta(raw_primary, baseline_primary),
            "fused_delta": numeric_delta(fused_primary, baseline_primary),
        },
        "artifact_hashes": build_artifact_hashes(
            validation_artifact_paths(artifacts_dir, expected_jsonl, expected_fused_jsonl, require_fused)
        ),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "manifest_path_audit": manifest_path_audit,
    }


def infer_require_fused(args: argparse.Namespace, manifest: dict[str, Any]) -> bool:
    if args.require_fused is not None:
        return bool(args.require_fused)
    if manifest:
        return bool(manifest.get("post_fusion", True))
    return True


def validation_artifact_paths(
    artifacts_dir: Path,
    expected_jsonl: Path,
    expected_fused_jsonl: Path,
    require_fused: bool,
) -> dict[str, Path]:
    paths = {
        "raw_jsonl": expected_jsonl,
        "score": artifacts_dir / "score.json",
        "comparison": artifacts_dir / "comparison.json",
    }
    if require_fused:
        paths.update(
            {
                "fused_jsonl": expected_fused_jsonl,
                "fused_score": artifacts_dir / "fused_score.json",
                "fused_comparison": artifacts_dir / "fused_comparison.json",
                "fusion_analysis": artifacts_dir / "fusion_analysis.json",
            }
        )
    return paths


def validate_score(
    label: str,
    score: dict[str, Any],
    comparison: dict[str, Any],
    baseline_metrics: dict[str, Any],
    budgets: list[int],
    expected_rows: int,
    check: Any,
) -> None:
    if not score or not comparison:
        return
    check(f"{label}_score_rows", int(score.get("num_rows", 0) or 0) == expected_rows, str(score.get("num_rows")))
    check(f"{label}_comparison_rows", int(comparison.get("num_rows", 0) or 0) == expected_rows, str(comparison.get("num_rows")))
    check(
        f"{label}_rank_source",
        score.get("rank_source") in {"min_cracked_guess_number", "outputPasswords"},
        str(score.get("rank_source")),
    )
    deltas = comparison.get("deltas", {})
    for budget in budgets:
        metric = f"sr@{budget}"
        hits_metric = f"hits@{budget}"
        baseline_key = f"sr{budget}"
        observed = score.get(metric)
        hits = score.get(hits_metric)
        check(f"{label}_{metric}_finite", is_probability(observed), str(observed))
        check(f"{label}_{hits_metric}_valid", is_nonnegative_int(hits), str(hits))
        if metric in deltas:
            item = deltas[metric]
            baseline = item.get("baseline")
            compared_observed = item.get("observed")
            expected_baseline = baseline_metrics.get(baseline_key)
            check(
                f"{label}_{metric}_comparison_observed",
                numbers_close(compared_observed, observed),
                f"comparison={compared_observed}, score={observed}",
            )
            check(
                f"{label}_{metric}_baseline_match",
                numbers_close(baseline, expected_baseline),
                f"comparison={baseline}, contract={expected_baseline}",
            )
            check(
                f"{label}_{metric}_delta_match",
                numbers_close(item.get("delta"), float(observed) - float(baseline)),
                f"delta={item.get('delta')}",
            )
        else:
            check(f"{label}_{metric}_comparison_present", False, metric)


def build_artifact_hashes(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: artifact_hash(path) for name, path in paths.items() if path}


def artifact_hash(path: Path) -> dict[str, Any]:
    if not path or not path.exists() or not path.is_file():
        return {"path": str(path), "exists": False, "sha256": ""}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "exists": True, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def build_manifest_path_audit(manifest: dict[str, Any], artifacts_dir: Path) -> dict[str, Any]:
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


def manifest_repo_path(manifest: dict[str, Any], key: str) -> Path:
    if not manifest:
        return Path()
    return relocate_manifest_path(manifest.get(key, ""), manifest)


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


def looks_like_absolute_path(value: str) -> bool:
    if not value:
        return False
    return Path(value).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("/")


def path_text_matches(left: Any, right: Path) -> bool:
    if not isinstance(left, str) or not left:
        return False
    return normalize_path_text(Path(left)) == normalize_path_text(right)


def normalize_path_text(path: Path) -> str:
    try:
        path = path.resolve()
    except OSError:
        pass
    return str(path).rstrip("\\/").casefold()


def validate_jsonl_rows(label: str, rows: list[dict[str, Any]], expected_rows: int, min_candidates: int, check: Any) -> None:
    check(f"{label}_rows", len(rows) == expected_rows, str(len(rows)))
    indices = [row.get("index") for row in rows if "index" in row]
    if indices:
        check(f"{label}_unique_indices", len(set(indices)) == len(indices), f"unique={len(set(indices))}, rows={len(indices)}")
    missing_targets = [
        idx
        for idx, row in enumerate(rows)
        if not str(row.get("real password", row.get("real_password", row.get("password", ""))))
    ]
    check(f"{label}_targets_present", not missing_targets, f"missing={missing_targets[:10]}")
    invalid_candidate_rows = []
    empty_candidate_rows = []
    short_candidate_rows = []
    duplicate_candidate_rows = []
    mismatched_rank_rows = []
    prompt_leak_rows = []
    for idx, row in enumerate(rows):
        candidates = row.get("outputPasswords", [])
        if not isinstance(candidates, list):
            invalid_candidate_rows.append(idx)
            continue
        if min_candidates and len(candidates) < min_candidates:
            short_candidate_rows.append({"row": idx, "candidates": len(candidates)})
        passwords = [candidate_password(item) for item in candidates]
        nonempty_passwords = [password for password in passwords if password]
        if min_candidates and len(nonempty_passwords) != len(passwords):
            empty_candidate_rows.append(idx)
        if min_candidates and len(set(nonempty_passwords)) < min_candidates:
            short_candidate_rows.append({"row": idx, "unique_nonempty_candidates": len(set(nonempty_passwords))})
        if len(set(nonempty_passwords)) != len(nonempty_passwords):
            duplicate_candidate_rows.append(idx)
        model_input = str(row.get("model_input", ""))
        if model_input and any(password.startswith(model_input) for password in nonempty_passwords):
            prompt_leak_rows.append(idx)
        observed_rank = safe_int(row.get("min_cracked_guess_number", 0))
        recomputed_rank = rank_from_row(row)
        if observed_rank != recomputed_rank:
            mismatched_rank_rows.append({"row": idx, "observed": observed_rank, "recomputed": recomputed_rank})
    check(f"{label}_candidate_lists", not invalid_candidate_rows, f"invalid={invalid_candidate_rows[:10]}")
    check(f"{label}_candidate_nonempty", not empty_candidate_rows, f"empty={empty_candidate_rows[:10]}")
    check(f"{label}_candidate_duplicates", not duplicate_candidate_rows, f"duplicates={duplicate_candidate_rows[:10]}")
    check(f"{label}_candidate_prompt_leak", not prompt_leak_rows, f"leaked={prompt_leak_rows[:10]}")
    check(f"{label}_rank_consistency", not mismatched_rank_rows, f"mismatched={mismatched_rank_rows[:10]}")
    check(
        f"{label}_candidate_budget",
        not short_candidate_rows,
        f"min_candidates={min_candidates}, short={short_candidate_rows[:10]}",
        severity="warning" if min_candidates == 0 else "error",
    )


def candidate_password(item: Any) -> str:
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    if isinstance(item, dict):
        return str(item.get("password", item.get("candidate", "")))
    return str(item)


def rank_from_row(row: dict[str, Any]) -> int:
    target = str(row.get("real password", row.get("real_password", row.get("password", ""))))
    candidates = row.get("outputPasswords", [])
    if not target or not isinstance(candidates, list):
        return 0
    for index, item in enumerate(candidates, start=1):
        if candidate_password(item) == target:
            return index
    return 0


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def validate_fusion_analysis(
    fusion_analysis: dict[str, Any],
    budgets: list[int],
    expected_rows: int,
    allow_worsening: bool,
    check: Any,
) -> None:
    if not fusion_analysis:
        return
    check("fusion_analysis_rows", int(fusion_analysis.get("num_rows", 0) or 0) == expected_rows, str(fusion_analysis.get("num_rows")))
    analysis_budgets = fusion_analysis.get("budgets", {})
    for budget in budgets:
        row = analysis_budgets.get(str(budget), {})
        check(f"fusion_analysis_budget_{budget}", bool(row), str(row))
        if row:
            check(f"fusion_delta_sr_{budget}_finite", isinstance(row.get("delta_sr"), (int, float)), str(row.get("delta_sr")))
            ci = row.get("delta_sr_ci95", [])
            check(
                f"fusion_delta_ci_{budget}",
                isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci),
                str(ci),
            )
    rank_changes = fusion_analysis.get("rank_changes", {})
    worsened = int(rank_changes.get("worsened", 0) or 0)
    lost_hits = int(rank_changes.get("lost_hits", 0) or 0)
    check(
        "fusion_no_worsened_ranks",
        allow_worsening or worsened == 0,
        f"worsened={worsened}",
        severity="warning" if allow_worsening else "error",
    )
    check(
        "fusion_no_lost_hits",
        allow_worsening or lost_hits == 0,
        f"lost_hits={lost_hits}",
        severity="warning" if allow_worsening else "error",
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Formal Output Validation",
        "",
        f"- status: `{report['status']}`",
        f"- artifacts: `{report['artifacts_dir']}`",
        f"- baseline variant: `{report.get('baseline_variant')}`",
        f"- expected rows: `{report.get('expected_rows')}`",
        f"- primary metric: `{report.get('primary_metric')}`",
        "",
        "## Primary",
        "",
        "| Metric | Baseline | Raw | Raw Delta | Fused | Fused Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    primary = report.get("primary", {})
    lines.append(
        "| {metric} | {baseline} | {raw} | {raw_delta} | {fused} | {fused_delta} |".format(
            metric=report.get("primary_metric"),
            baseline=format_number(primary.get("baseline")),
            raw=format_number(primary.get("raw")),
            raw_delta=format_number(primary.get("raw_delta"), signed=True),
            fused=format_number(primary.get("fused")),
            fused_delta=format_number(primary.get("fused_delta"), signed=True),
        )
    )
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in report.get("checks", []):
        status = "PASS" if check.get("ok") else ("WARN" if check.get("severity") == "warning" else "FAIL")
        detail = str(check.get("detail", "")).replace("|", "\\|")
        lines.append(f"| `{check.get('name')}` | {status} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def format_number(value: Any, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if signed:
        return f"{value:+.4f}"
    return f"{value:.4f}"


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]] | None:
    if not path or not path.exists() or not path.is_file():
        return None
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_budgets(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def is_probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and value >= 0


def numbers_close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance


def numeric_delta(value: Any, baseline: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return float(value) - float(baseline)


if __name__ == "__main__":
    main()
