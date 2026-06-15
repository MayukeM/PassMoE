from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPERIMENTS = [
    ("baseline10k", "baseline10k_p00"),
    ("baseline500k", "baseline500k_p00"),
    ("fd500k", "fd500k_p00"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce CPU-side PassMoE fusion results on local PassLLM quick JSONL files."
    )
    parser.add_argument(
        "--quick-root",
        default=r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code\result\quick",
        help="Directory containing <variant>/input_output.jsonl from PassLLM quick runs.",
    )
    parser.add_argument("--out-dir", default="artifacts/fusion")
    parser.add_argument("--report", default="artifacts/reports/fusion_repro_summary.md")
    parser.add_argument("--summary-json", default="artifacts/fusion/fusion_repro_summary.json")
    parser.add_argument("--budgets", default="1,10,50,100")
    parser.add_argument("--strategy", choices=["append", "prepend", "insert", "score"], default="score")
    parser.add_argument("--max-expert-candidates", type=int, default=80)
    parser.add_argument("--score-existing-weight", type=float, default=1.0)
    parser.add_argument("--score-expert-weight", type=float, default=0.05)
    parser.add_argument("--score-rank-offset", type=float, default=2.0)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    quick_root = Path(args.quick_root)
    out_dir = (repo_root / args.out_dir).resolve()
    report_path = (repo_root / args.report).resolve()
    summary_path = (repo_root / args.summary_json).resolve()
    main_py = repo_root / "main.py"

    if not main_py.exists():
        raise SystemExit(f"main.py not found at {main_py}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for short_name, variant in EXPERIMENTS:
        input_jsonl = quick_root / variant / "input_output.jsonl"
        if not input_jsonl.exists():
            raise SystemExit(f"Missing input JSONL: {input_jsonl}")

        fused_stem = (
            f"{short_name}_{args.strategy}_m{args.max_expert_candidates}"
            f"_w{stem_float(args.score_expert_weight)}"
            f"_o{stem_float(args.score_rank_offset)}"
        )
        fused_jsonl = out_dir / f"{fused_stem}.jsonl"
        original_metrics = out_dir / f"{short_name}_original_recomputed_metrics.json"
        fused_metrics = out_dir / f"{fused_stem}_metrics.json"
        fused_recomputed = out_dir / f"{fused_stem}_recomputed_metrics.json"
        analysis = out_dir / f"{fused_stem}_analysis.json"

        run(
            [
                args.python,
                str(main_py),
                "score-jsonl",
                "--jsonl",
                str(input_jsonl),
                "--budgets",
                args.budgets,
                "--recompute-from-candidates",
                "--out",
                str(original_metrics),
            ],
            repo_root,
            args.dry_run,
        )
        run(
            [
                args.python,
                str(main_py),
                "fuse-jsonl",
                "--jsonl",
                str(input_jsonl),
                "--out-jsonl",
                str(fused_jsonl),
                "--out-metrics",
                str(fused_metrics),
                "--strategy",
                args.strategy,
                "--max-expert-candidates",
                str(args.max_expert_candidates),
                "--score-existing-weight",
                str(args.score_existing_weight),
                "--score-expert-weight",
                str(args.score_expert_weight),
                "--score-rank-offset",
                str(args.score_rank_offset),
                "--budgets",
                args.budgets,
            ],
            repo_root,
            args.dry_run,
        )
        run(
            [
                args.python,
                str(main_py),
                "score-jsonl",
                "--jsonl",
                str(fused_jsonl),
                "--budgets",
                args.budgets,
                "--recompute-from-candidates",
                "--out",
                str(fused_recomputed),
            ],
            repo_root,
            args.dry_run,
        )
        run(
            [
                args.python,
                str(main_py),
                "analyze-fusion",
                "--original-jsonl",
                str(input_jsonl),
                "--fused-jsonl",
                str(fused_jsonl),
                "--budgets",
                args.budgets,
                "--bootstrap-iters",
                str(args.bootstrap_iters),
                "--out",
                str(analysis),
            ],
            repo_root,
            args.dry_run,
        )

        if not args.dry_run:
            rows.append(
                summarize_one(
                    short_name=short_name,
                    variant=variant,
                    input_jsonl=input_jsonl,
                    fused_jsonl=fused_jsonl,
                    original_metrics=original_metrics,
                    fused_recomputed=fused_recomputed,
                    analysis=analysis,
                    budgets=parse_budgets(args.budgets),
                )
            )

    if args.dry_run:
        return

    summary = {
        "quick_root": str(quick_root.resolve()),
        "strategy": args.strategy,
        "max_expert_candidates": args.max_expert_candidates,
        "score_existing_weight": args.score_existing_weight,
        "score_expert_weight": args.score_expert_weight,
        "score_rank_offset": args.score_rank_offset,
        "bootstrap_iters": args.bootstrap_iters,
        "budgets": parse_budgets(args.budgets),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown(summary, summary_path), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "report": str(report_path)}, indent=2))


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(quote(part) for part in command)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def quote(value: str) -> str:
    if any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def parse_budgets(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_one(
    short_name: str,
    variant: str,
    input_jsonl: Path,
    fused_jsonl: Path,
    original_metrics: Path,
    fused_recomputed: Path,
    analysis: Path,
    budgets: list[int],
) -> dict[str, Any]:
    original = load_json(original_metrics)
    fused = load_json(fused_recomputed)
    comparison = load_json(analysis)
    row: dict[str, Any] = {
        "name": short_name,
        "variant": variant,
        "input_jsonl": str(input_jsonl.resolve()),
        "fused_jsonl": str(fused_jsonl.resolve()),
        "original_metrics": str(original_metrics.resolve()),
        "fused_metrics": str(fused_recomputed.resolve()),
        "analysis": str(analysis.resolve()),
        "num_rows": original["num_rows"],
        "rank_changes": comparison["rank_changes"],
        "budgets": {},
    }
    for budget in budgets:
        key = str(budget)
        row["budgets"][key] = {
            "original_sr": original[f"sr@{budget}"],
            "fused_sr": fused[f"sr@{budget}"],
            "delta_sr": fused[f"sr@{budget}"] - original[f"sr@{budget}"],
            "delta_sr_ci95": comparison["budgets"][key]["delta_sr_ci95"],
        }
    return row


def render_markdown(summary: dict[str, Any], summary_path: Path) -> str:
    budgets = summary["budgets"]
    lines = [
        "# Fusion Reproduction Summary",
        "",
        "This file is generated by `python scripts/run_fusion_experiments.py`.",
        "",
        f"- quick root: `{summary['quick_root']}`",
        f"- strategy: `{summary['strategy']}`",
        f"- max expert candidates: `{summary['max_expert_candidates']}`",
        f"- score existing weight: `{summary['score_existing_weight']}`",
        f"- score expert weight: `{summary['score_expert_weight']}`",
        f"- score rank offset: `{summary['score_rank_offset']}`",
        f"- bootstrap iterations: `{summary['bootstrap_iters']}`",
        f"- machine-readable summary: `{summary_path}`",
        "",
    ]

    header = ["Input"]
    for budget in budgets:
        header.extend([f"Original SR@{budget}", f"Fused SR@{budget}", f"Delta"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for row in summary["rows"]:
        cells = [row["variant"]]
        for budget in budgets:
            item = row["budgets"][str(budget)]
            cells.extend(
                [
                    format_float(item["original_sr"]),
                    format_float(item["fused_sr"]),
                    signed_float(item["delta_sr"]),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Bootstrap Deltas", ""])
    lines.append(
        "| Input | Budget | Delta SR | 95% CI | Overall improved ranks | "
        "Overall worsened ranks | Overall new hits | Overall lost hits |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["rows"]:
        changes = row["rank_changes"]
        for budget in budgets:
            item = row["budgets"][str(budget)]
            low, high = item["delta_sr_ci95"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["variant"],
                        str(budget),
                        signed_float(item["delta_sr"]),
                        f"[{format_float(low)}, {format_float(high)}]",
                        str(changes["improved"]),
                        str(changes["worsened"]),
                        str(changes["new_hits"]),
                        str(changes["lost_hits"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This reproduces the CPU-side deterministic fusion diagnostic over existing PassLLM quick outputs. "
            "It is comparable under the JSONL SR@K contract, but it is not a replacement for the full neural "
            "Qwen + FieldDrop + PassMoE targeted GPU run.",
            "",
        ]
    )
    return "\n".join(lines)


def format_float(value: float) -> str:
    return f"{float(value):.4f}"


def signed_float(value: float) -> str:
    value = float(value)
    return f"{value:+.4f}"


def stem_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
