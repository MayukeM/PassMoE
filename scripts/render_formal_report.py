from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_RUN_NAME = "qwen_fielddrop_base_identity_clixsense_500_raw"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "formal" / DEFAULT_FORMAL_RUN_NAME

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_formal_status import inspect_formal_status, relocate_manifest_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a concise paper-facing formal PassMoE result report.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    artifacts_dir = resolve_path(args.artifacts_dir)
    report = build_report(artifacts_dir)
    md_path = resolve_path(args.out_md) if args.out_md else artifacts_dir / "formal_result_report.md"
    json_path = resolve_path(args.out_json) if args.out_json else artifacts_dir / "formal_result_report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "claim_status": report["claim_status"], "report": str(md_path)}, indent=2))


def build_report(artifacts_dir: Path) -> dict[str, Any]:
    status = inspect_formal_status(artifacts_dir)
    manifest = load_json_if_exists(artifacts_dir / "run_manifest.json")
    validation = load_json_if_exists(artifacts_dir / "formal_validation.json")
    score = load_json_if_exists(artifacts_dir / "score.json")
    fused_score = load_json_if_exists(artifacts_dir / "fused_score.json")
    comparison = load_json_if_exists(artifacts_dir / "comparison.json")
    fused_comparison = load_json_if_exists(artifacts_dir / "fused_comparison.json")
    fusion_analysis = load_json_if_exists(artifacts_dir / "fusion_analysis.json")
    cuda_readiness_path = relocate_manifest_path(manifest.get("cuda_readiness_path", ""), manifest) if manifest else artifacts_dir / "cuda_readiness.json"
    environment_snapshot_path = relocate_manifest_path(manifest.get("environment_snapshot_path", ""), manifest) if manifest else artifacts_dir / "environment_snapshot.json"
    cuda_readiness = load_json_if_exists(cuda_readiness_path)

    budgets = [int(item) for item in manifest.get("budgets", [])] if manifest else []
    primary_budget = max(budgets) if budgets else 100
    primary_metric = f"sr@{primary_budget}"
    baseline_metrics = manifest.get("baseline_metrics", {}) if manifest else {}
    baseline_primary = numeric_or_none(baseline_metrics.get(f"sr{primary_budget}"))
    raw_primary = numeric_or_none(score.get(primary_metric)) if score else None
    fused_primary = numeric_or_none(fused_score.get(primary_metric)) if fused_score else None
    raw_delta = numeric_delta(raw_primary, baseline_primary)
    fused_delta = numeric_delta(fused_primary, baseline_primary)
    is_score_only = bool(manifest.get("mode") == "score_only")
    is_diagnostic = bool(manifest.get("device") == "cpu" or status.get("expected_rows") != baseline_metrics.get("n"))
    claim_status = classify_claim(status, validation, is_diagnostic, is_score_only, raw_delta, fused_delta)
    recommendation = status.get("recommendation", {})
    if claim_status == "diagnostic_only":
        recommendation = {
            "reason": "diagnostic CPU/subset run is not a formal comparison",
            "command": (
                "python scripts/run_formal_passmoe.py --execute "
                "--run-name qwen_fielddrop_base_identity_clixsense_500_raw --seed 42"
            ),
        }
    elif claim_status == "supplementary_fusion_only":
        recommendation = {
            "reason": "score-only fusion artifact is supplementary and does not replace the full neural PassMoE run",
            "command": (
                "python scripts/run_formal_passmoe.py --execute "
                "--run-name qwen_fielddrop_base_identity_clixsense_500_raw --seed 42"
            ),
        }

    rows = []
    for budget in budgets:
        metric = f"sr@{budget}"
        rows.append(
            {
                "metric": metric,
                "baseline": numeric_or_none(baseline_metrics.get(f"sr{budget}")),
                "raw": numeric_or_none(score.get(metric)) if score else None,
                "raw_delta": comparison_delta(comparison, metric),
                "fused": numeric_or_none(fused_score.get(metric)) if fused_score else None,
                "fused_delta": comparison_delta(fused_comparison, metric),
            }
        )

    validation_file_status = validation.get("status") if validation else "missing"
    report_validation_status = status.get("validation_status") or validation_file_status

    return {
        "status": status.get("status"),
        "claim_status": claim_status,
        "artifacts_dir": str(artifacts_dir),
        "run_name": manifest.get("run_name", artifacts_dir.name) if manifest else artifacts_dir.name,
        "baseline_variant": manifest.get("baseline_variant") if manifest else None,
        "device": manifest.get("device") if manifest else None,
        "dtype": manifest.get("dtype") if manifest else None,
        "seed": manifest.get("seed") if manifest else None,
        "expected_rows": status.get("expected_rows"),
        "raw_rows": status.get("raw_rows"),
        "fused_rows": status.get("fused_rows"),
        "budgets": budgets,
        "primary_metric": primary_metric,
        "primary": {
            "baseline": baseline_primary,
            "raw": raw_primary,
            "raw_delta": raw_delta,
            "fused": fused_primary,
            "fused_delta": fused_delta,
        },
        "metrics": rows,
        "validation_status": report_validation_status,
        "validation_file_status": validation_file_status,
        "preflight_status": status.get("preflight_status"),
        "is_diagnostic": is_diagnostic,
        "is_score_only": is_score_only,
        "model_execution_provenance": status.get("model_execution_provenance", {}),
        "targeted_generation_progress": status.get("targeted_generation_progress", {}),
        "cuda_readiness_status": cuda_readiness.get("status") if cuda_readiness else "missing",
        "cuda_readiness_recommendation": cuda_readiness.get("recommendation", {}) if cuda_readiness else {},
        "recommendation": recommendation,
        "fusion_rank_changes": (fusion_analysis.get("rank_changes", {}) if fusion_analysis else {}),
        "paths": {
            "summary": str(artifacts_dir / "summary.md"),
            "validation": str(artifacts_dir / "formal_validation.json"),
            "environment_snapshot": str(environment_snapshot_path),
            "cuda_readiness": str(cuda_readiness_path),
            "status": str(artifacts_dir / "formal_result_report.json"),
        },
    }


def classify_claim(
    status: dict[str, Any],
    validation: dict[str, Any],
    is_diagnostic: bool,
    is_score_only: bool,
    raw_delta: float | None,
    fused_delta: float | None,
) -> str:
    if status.get("status") != "complete" or validation.get("status") != "passed":
        return "incomplete"
    if is_diagnostic:
        return "diagnostic_only"
    if is_score_only:
        return "supplementary_fusion_only"
    best_delta = max([value for value in (raw_delta, fused_delta) if value is not None], default=None)
    if best_delta is None:
        return "complete_unscored"
    return "better_or_equal_baseline" if best_delta >= 0 else "below_baseline"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Formal PassMoE Result Report",
        "",
        f"- run: `{report['run_name']}`",
        f"- status: `{report['status']}`",
        f"- claim status: `{report['claim_status']}`",
        f"- baseline variant: `{fmt_text(report.get('baseline_variant'))}`",
        f"- device/dtype: `{fmt_text(report.get('device'))}` / `{fmt_text(report.get('dtype'))}`",
        f"- seed: `{fmt_text(report.get('seed'))}`",
        f"- rows: raw `{fmt_text(report.get('raw_rows'))}` / expected `{fmt_text(report.get('expected_rows'))}`",
        "",
        "## Primary Metric",
        "",
        "| Metric | Baseline | Raw | Raw Delta | Fused | Fused Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    primary = report["primary"]
    lines.append(
        f"| {report['primary_metric']} | {fmt(primary['baseline'])} | {fmt(primary['raw'])} | "
        f"{fmt(primary['raw_delta'], signed=True)} | {fmt(primary['fused'])} | {fmt(primary['fused_delta'], signed=True)} |"
    )
    lines.extend(["", "## Metrics", "", "| Metric | Baseline | Raw | Raw Delta | Fused | Fused Delta |", "|---|---:|---:|---:|---:|---:|"])
    for row in report.get("metrics", []):
        lines.append(
            f"| {row['metric']} | {fmt(row['baseline'])} | {fmt(row['raw'])} | {fmt(row['raw_delta'], signed=True)} | "
            f"{fmt(row['fused'])} | {fmt(row['fused_delta'], signed=True)} |"
        )
    rank_changes = report.get("fusion_rank_changes") or {}
    provenance = report.get("model_execution_provenance") or {}
    progress = report.get("targeted_generation_progress") or {}
    paths = report.get("paths") or {}
    if paths.get("environment_snapshot"):
        lines.extend(["", "## Environment", "", f"- snapshot: `{paths.get('environment_snapshot')}`"])
        if paths.get("cuda_readiness"):
            lines.append(f"- CUDA readiness: `{paths.get('cuda_readiness')}`")
            lines.append(f"- CUDA readiness status: `{report.get('cuda_readiness_status')}`")
            readiness_recommendation = report.get("cuda_readiness_recommendation") or {}
            if readiness_recommendation.get("reason"):
                lines.append(f"- CUDA readiness reason: {readiness_recommendation.get('reason')}")
    if provenance:
        lines.extend(
            [
                "",
                "## Provenance",
                "",
                f"- model execution provenance: `{provenance.get('status')}`",
                f"- required: `{provenance.get('required')}`",
            ]
        )
        if provenance.get("reason"):
            lines.append(f"- reason: `{provenance.get('reason')}`")
    if progress.get("status") == "present":
        latest = progress.get("latest") or {}
        lines.extend(
            [
                "",
                "## Progress",
                "",
                f"- latest generation progress: `{fmt_text(latest.get('completed'))}` / `{fmt_text(latest.get('total'))}`",
                f"- fraction: `{fmt_fraction(latest.get('fraction'))}`",
                f"- candidates per user: `{fmt_text(latest.get('candidates_per_user'))}`",
                f"- generation batch size: `{fmt_text(latest.get('generation_batch_size'))}`",
                f"- generated this run: `{fmt_text(latest.get('generated_rows_this_run'))}`",
                f"- remaining rows: `{fmt_text(latest.get('remaining_rows'))}`",
                f"- seconds per row: `{fmt_text(latest.get('seconds_per_row'))}`",
                f"- seconds per generated row: `{fmt_text(latest.get('seconds_per_generated_row'))}`",
                f"- ETA: `{fmt_duration(latest.get('eta_seconds'))}`",
                f"- progress markers: `{fmt_text(progress.get('num_markers'))}`",
                f"- source log: `{fmt_text(progress.get('source_log'))}`",
            ]
        )
    if rank_changes:
        lines.extend(
            [
                "",
                "## Fusion",
                "",
                f"- improved ranks: `{rank_changes.get('improved', 0)}`",
                f"- worsened ranks: `{rank_changes.get('worsened', 0)}`",
                f"- new hits: `{rank_changes.get('new_hits', 0)}`",
                f"- lost hits: `{rank_changes.get('lost_hits', 0)}`",
            ]
        )
    recommendation = report.get("recommendation") or {}
    if report["claim_status"] != "better_or_equal_baseline":
        lines.extend(["", "## Next Step", "", f"- reason: {recommendation.get('reason')}", f"- command: `{recommendation.get('command')}`"])
    lines.append("")
    return "\n".join(lines)


def comparison_delta(comparison: dict[str, Any], metric: str) -> float | None:
    if not comparison:
        return None
    item = comparison.get("deltas", {}).get(metric, {})
    return numeric_or_none(item.get("delta"))


def numeric_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def numeric_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, signed: bool = False) -> str:
    number = numeric_or_none(value)
    if number is None:
        return "n/a"
    if signed:
        return f"{number:+.4f}"
    return f"{number:.4f}"


def fmt_text(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def fmt_fraction(value: Any) -> str:
    number = numeric_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def fmt_duration(value: Any) -> str:
    seconds = numeric_or_none(value)
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.1f}h"


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


if __name__ == "__main__":
    main()
