from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion import (  # noqa: E402
    analyze_fusion_pair,
    fuse_candidates,
    fuse_ranked_jsonl,
    generate_expert_candidates,
    normalize_output_passwords,
    parse_pii_from_model_input,
    rank_from_row,
    rank_of,
)


DEFAULT_VARIANTS = ("baseline10k_p00", "baseline500k_p00", "fd500k_p00")


def passllm_code_root() -> Path:
    value = os.environ.get("PASSLLM_CODE_ROOT") or os.environ.get("PASSLLM_FIELDDROP_CODE_ROOT")
    if value:
        return Path(value)
    return REPO_ROOT / "external" / "PassLLM-FieldDrop" / "code"


def passllm_quick_root() -> str:
    value = os.environ.get("PASSLLM_QUICK_ROOT")
    if value:
        return value
    return str(passllm_code_root() / "result" / "quick")


@dataclass(frozen=True)
class PreparedRow:
    row: dict[str, Any]
    target: str
    existing: list[tuple[str, float]]
    expert_candidates: list[str]
    original_rank: int


@dataclass(frozen=True)
class VariantData:
    variant: str
    path: Path
    rows: list[PreparedRow]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search score-fusion parameters with train/test split over local PassLLM quick outputs."
    )
    parser.add_argument(
        "--quick-root",
        default=passllm_quick_root(),
        help="Directory containing <variant>/input_output.jsonl from PassLLM quick runs.",
    )
    parser.add_argument("--train-variants", default="baseline10k_p00,baseline500k_p00")
    parser.add_argument("--test-variants", default="fd500k_p00")
    parser.add_argument("--out-dir", default="artifacts/fusion_search")
    parser.add_argument("--budgets", default="1,10,50,100")
    parser.add_argument("--primary-budget", type=int, default=100)
    parser.add_argument("--tie-budget", type=int, default=50)
    parser.add_argument("--expert-weights", default="0.05,0.10,0.20,0.35,0.50,0.75,1.00")
    parser.add_argument("--max-expert-candidates", default="20,40,60,80,120,160")
    parser.add_argument("--rank-offsets", default="2.0")
    parser.add_argument("--existing-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write-best-jsonl", dest="write_best_jsonl", action="store_false")
    parser.set_defaults(write_best_jsonl=True)
    args = parser.parse_args()

    budgets = parse_int_list(args.budgets)
    train_variants = parse_str_list(args.train_variants)
    test_variants = parse_str_list(args.test_variants)
    candidate_counts = parse_int_list(args.max_expert_candidates)
    expert_weights = parse_float_list(args.expert_weights)
    rank_offsets = parse_float_list(args.rank_offsets)
    quick_root = Path(args.quick_root)
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_candidates = max(candidate_counts)
    all_variant_ids = list(dict.fromkeys(train_variants + test_variants))
    variants = {
        variant: load_variant(quick_root / variant / "input_output.jsonl", variant, max_candidates)
        for variant in all_variant_ids
    }

    grid_rows = []
    for max_expert_candidates in candidate_counts:
        for expert_weight in expert_weights:
            for rank_offset in rank_offsets:
                config = {
                    "strategy": "score",
                    "max_expert_candidates": max_expert_candidates,
                    "score_existing_weight": args.existing_weight,
                    "score_expert_weight": expert_weight,
                    "score_rank_offset": rank_offset,
                }
                train_result = summarize_group(
                    variants,
                    train_variants,
                    budgets,
                    config,
                    args.primary_budget,
                    args.tie_budget,
                )
                test_result = summarize_group(
                    variants,
                    test_variants,
                    budgets,
                    config,
                    args.primary_budget,
                    args.tie_budget,
                )
                grid_rows.append({"config": config, "train": train_result, "test": test_result})

    best = select_best(grid_rows, args.primary_budget, args.tie_budget)
    best_outputs = {}
    if args.write_best_jsonl:
        best_outputs = write_best_outputs(
            quick_root=quick_root,
            out_dir=out_dir,
            variants=variants,
            variant_ids=all_variant_ids,
            config=best["config"],
            budgets=budgets,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed,
        )

    report = {
        "quick_root": str(quick_root.resolve()),
        "train_variants": train_variants,
        "test_variants": test_variants,
        "budgets": budgets,
        "primary_budget": args.primary_budget,
        "tie_budget": args.tie_budget,
        "search_space": {
            "expert_weights": expert_weights,
            "max_expert_candidates": candidate_counts,
            "rank_offsets": rank_offsets,
            "existing_weight": args.existing_weight,
        },
        "selection_rule": (
            "minimize train worsened ranks first, then maximize train mean SR at primary_budget, "
            "break ties by train mean SR at tie_budget and fewer overall rank changes"
        ),
        "best": best,
        "best_outputs": best_outputs,
        "top_configs": sorted(
            grid_rows,
            key=lambda item: selection_key(item, args.primary_budget, args.tie_budget),
            reverse=True,
        )[:20],
    }

    summary_json = out_dir / "fusion_param_search.json"
    summary_md = out_dir / "fusion_param_search.md"
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md)}, indent=2))


def load_variant(path: Path, variant: str, max_candidates: int) -> VariantData:
    if not path.exists():
        raise SystemExit(f"Missing input JSONL: {path}")
    prepared = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = str(row.get("real password", row.get("real_password", row.get("password", ""))))
            pii = parse_pii_from_model_input(str(row.get("model_input", "")))
            prepared.append(
                PreparedRow(
                    row=row,
                    target=target,
                    existing=normalize_output_passwords(row.get("outputPasswords", [])),
                    expert_candidates=generate_expert_candidates(pii, max_candidates=max_candidates),
                    original_rank=rank_from_row(row),
                )
            )
    return VariantData(variant=variant, path=path, rows=prepared)


def summarize_group(
    variants: dict[str, VariantData],
    variant_ids: list[str],
    budgets: list[int],
    config: dict[str, Any],
    primary_budget: int,
    tie_budget: int,
) -> dict[str, Any]:
    per_variant = {
        variant_id: evaluate_variant(variants[variant_id], budgets, config)
        for variant_id in variant_ids
    }
    mean_metrics: dict[str, float] = {}
    for budget in budgets:
        mean_metrics[f"mean_original_sr@{budget}"] = mean(
            item["budgets"][str(budget)]["original_sr"] for item in per_variant.values()
        )
        mean_metrics[f"mean_fused_sr@{budget}"] = mean(
            item["budgets"][str(budget)]["fused_sr"] for item in per_variant.values()
        )
        mean_metrics[f"mean_delta_sr@{budget}"] = mean(
            item["budgets"][str(budget)]["delta_sr"] for item in per_variant.values()
        )
    return {
        "variants": per_variant,
        "mean_metrics": mean_metrics,
        "selection_metrics": {
            "primary_mean_fused_sr": mean_metrics[f"mean_fused_sr@{primary_budget}"],
            "primary_mean_delta_sr": mean_metrics[f"mean_delta_sr@{primary_budget}"],
            "tie_mean_fused_sr": mean_metrics[f"mean_fused_sr@{tie_budget}"],
            "tie_mean_delta_sr": mean_metrics[f"mean_delta_sr@{tie_budget}"],
            "total_changed": sum(item["rank_changes"]["changed"] for item in per_variant.values()),
            "total_worsened": sum(item["rank_changes"]["worsened"] for item in per_variant.values()),
            "total_improved": sum(item["rank_changes"]["improved"] for item in per_variant.values()),
            "total_new_hits": sum(item["rank_changes"]["new_hits"] for item in per_variant.values()),
            "total_lost_hits": sum(item["rank_changes"]["lost_hits"] for item in per_variant.values()),
        },
    }


def evaluate_variant(
    variant: VariantData,
    budgets: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    fused_ranks = []
    original_ranks = []
    for item in variant.rows:
        fused = fuse_candidates(
            item.existing,
            item.expert_candidates[: int(config["max_expert_candidates"])],
            strategy="score",
            insert_after=10,
            score_existing_weight=float(config["score_existing_weight"]),
            score_expert_weight=float(config["score_expert_weight"]),
            score_rank_offset=float(config["score_rank_offset"]),
        )
        fused_ranks.append(rank_of(item.target, [password for password, _score in fused]))
        original_ranks.append(item.original_rank)

    budgets_summary: dict[str, Any] = {}
    for budget in budgets:
        original_hits = sum(1 for rank in original_ranks if 1 <= rank <= budget)
        fused_hits = sum(1 for rank in fused_ranks if 1 <= rank <= budget)
        original_sr = original_hits / max(len(original_ranks), 1)
        fused_sr = fused_hits / max(len(fused_ranks), 1)
        budgets_summary[str(budget)] = {
            "original_hits": original_hits,
            "fused_hits": fused_hits,
            "original_sr": original_sr,
            "fused_sr": fused_sr,
            "delta_hits": fused_hits - original_hits,
            "delta_sr": fused_sr - original_sr,
        }

    return {
        "num_rows": len(variant.rows),
        "budgets": budgets_summary,
        "rank_changes": summarize_rank_changes(original_ranks, fused_ranks),
    }


def summarize_rank_changes(original_ranks: list[int], fused_ranks: list[int]) -> dict[str, int]:
    changed = 0
    improved = 0
    worsened = 0
    new_hits = 0
    lost_hits = 0
    for original_rank, fused_rank in zip(original_ranks, fused_ranks):
        if original_rank == fused_rank:
            continue
        changed += 1
        if fused_rank and (original_rank == 0 or fused_rank < original_rank):
            improved += 1
            if original_rank == 0:
                new_hits += 1
        elif original_rank and (fused_rank == 0 or fused_rank > original_rank):
            worsened += 1
            if fused_rank == 0:
                lost_hits += 1
    return {
        "changed": changed,
        "improved": improved,
        "worsened": worsened,
        "new_hits": new_hits,
        "lost_hits": lost_hits,
    }


def select_best(grid_rows: list[dict[str, Any]], primary_budget: int, tie_budget: int) -> dict[str, Any]:
    return max(grid_rows, key=lambda item: selection_key(item, primary_budget, tie_budget))


def selection_key(item: dict[str, Any], primary_budget: int, tie_budget: int) -> tuple[float, float, int, int, int]:
    train = item["train"]
    metrics = train["mean_metrics"]
    selection = train["selection_metrics"]
    return (
        -selection["total_worsened"],
        metrics[f"mean_fused_sr@{primary_budget}"],
        metrics[f"mean_fused_sr@{tie_budget}"],
        -selection["total_changed"],
        selection["total_improved"],
    )


def write_best_outputs(
    quick_root: Path,
    out_dir: Path,
    variants: dict[str, VariantData],
    variant_ids: list[str],
    config: dict[str, Any],
    budgets: list[int],
    bootstrap_iters: int,
    seed: int,
) -> dict[str, Any]:
    stem = make_config_stem(config)
    outputs: dict[str, Any] = {}
    for variant_id in variant_ids:
        input_jsonl = quick_root / variant_id / "input_output.jsonl"
        fused_jsonl = out_dir / f"{variant_id}_{stem}.jsonl"
        metrics_json = out_dir / f"{variant_id}_{stem}_metrics.json"
        analysis_json = out_dir / f"{variant_id}_{stem}_analysis.json"
        metrics = fuse_ranked_jsonl(
            input_jsonl=input_jsonl,
            output_jsonl=fused_jsonl,
            strategy="score",
            max_expert_candidates=int(config["max_expert_candidates"]),
            budgets=budgets,
            score_existing_weight=float(config["score_existing_weight"]),
            score_expert_weight=float(config["score_expert_weight"]),
            score_rank_offset=float(config["score_rank_offset"]),
        )
        metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        analysis = analyze_fusion_pair(
            original_jsonl=variants[variant_id].path,
            fused_jsonl=fused_jsonl,
            budgets=budgets,
            bootstrap_iters=bootstrap_iters,
            seed=seed,
        )
        analysis_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        outputs[variant_id] = {
            "input_jsonl": str(input_jsonl.resolve()),
            "fused_jsonl": str(fused_jsonl.resolve()),
            "metrics_json": str(metrics_json.resolve()),
            "analysis_json": str(analysis_json.resolve()),
        }
    return outputs


def render_markdown(report: dict[str, Any]) -> str:
    best = report["best"]
    config = best["config"]
    lines = [
        "# Fusion Parameter Search",
        "",
        "This report is generated by `python scripts/search_fusion_params.py`.",
        "",
        f"- quick root: `{report['quick_root']}`",
        f"- train variants: `{', '.join(report['train_variants'])}`",
        f"- test variants: `{', '.join(report['test_variants'])}`",
        f"- primary budget: `SR@{report['primary_budget']}`",
        f"- tie budget: `SR@{report['tie_budget']}`",
        f"- best config: `{json.dumps(config, sort_keys=True)}`",
        "",
        "## Best Config Summary",
        "",
    ]
    lines.extend(render_group_table("Train", best["train"], report["budgets"]))
    lines.extend(render_group_table("Test", best["test"], report["budgets"]))
    lines.extend(["", "## Top Configs", ""])
    lines.append("| Rank | Max experts | Expert weight | Rank offset | Train SR@100 | Train Delta@100 | Test SR@100 | Test Delta@100 | Train changed | Train worsened |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, item in enumerate(report["top_configs"][:10], start=1):
        cfg = item["config"]
        train_metrics = item["train"]["mean_metrics"]
        test_metrics = item["test"]["mean_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(cfg["max_expert_candidates"]),
                    format_float(cfg["score_expert_weight"]),
                    format_float(cfg["score_rank_offset"]),
                    format_float(train_metrics["mean_fused_sr@100"]),
                    signed_float(train_metrics["mean_delta_sr@100"]),
                    format_float(test_metrics["mean_fused_sr@100"]),
                    signed_float(test_metrics["mean_delta_sr@100"]),
                    str(item["train"]["selection_metrics"]["total_changed"]),
                    str(item["train"]["selection_metrics"]["total_worsened"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "The split is over existing quick-output variants, not an independently sampled dataset. "
            "Use this as a CPU-side robustness check and parameter-selection guardrail, not as the final neural PassMoE result.",
            "",
        ]
    )
    return "\n".join(lines)


def render_group_table(name: str, group: dict[str, Any], budgets: list[int]) -> list[str]:
    lines = [f"### {name}", ""]
    header = ["Variant"]
    for budget in budgets:
        header.extend([f"Original SR@{budget}", f"Fused SR@{budget}", f"Delta"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for variant_id, result in group["variants"].items():
        cells = [variant_id]
        for budget in budgets:
            item = result["budgets"][str(budget)]
            cells.extend(
                [
                    format_float(item["original_sr"]),
                    format_float(item["fused_sr"]),
                    signed_float(item["delta_sr"]),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")

    cells = ["mean"]
    mean_metrics = group["mean_metrics"]
    for budget in budgets:
        cells.extend(
            [
                format_float(mean_metrics[f"mean_original_sr@{budget}"]),
                format_float(mean_metrics[f"mean_fused_sr@{budget}"]),
                signed_float(mean_metrics[f"mean_delta_sr@{budget}"]),
            ]
        )
    lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def make_config_stem(config: dict[str, Any]) -> str:
    return (
        "score"
        + f"_m{int(config['max_expert_candidates'])}"
        + f"_w{stem_float(float(config['score_expert_weight']))}"
        + f"_o{stem_float(float(config['score_rank_offset']))}"
    )


def stem_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / max(len(items), 1)


def format_float(value: float) -> str:
    return f"{float(value):.4f}"


def signed_float(value: float) -> str:
    return f"{float(value):+.4f}"


if __name__ == "__main__":
    main()
