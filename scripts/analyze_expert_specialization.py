from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config
from data import PasswordDataset, load_records
from evaluate import evaluate_router_distribution
from model import build_model_and_tokenizer
from trainer import load_checkpoint, set_seed


CONFIG_FIELDS = {field.name for field in dataclasses.fields(Config)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze PassMoE router/expert specialization over PII, entropy, and leetspeak buckets."
    )
    parser.add_argument("--data-path", default="data/clixsense/clixsense_test_500_from_fd500k_p00.json")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--base-model", default=None)
    parser.add_argument(
        "--base-adapter",
        default=None,
        help=(
            "'none', 'fielddrop', 'baseline10k', 'csdn', or a LoRA adapter directory. "
            "'fielddrop' is an imported PassLLM/FieldDrop baseline adapter, not a PassMoE method component."
        ),
    )
    parser.add_argument("--task", choices=["trawling", "targeted"], default=None)
    parser.add_argument("--prompt-template-id", default=None)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--router-hidden-dim", type=int, default=None)
    parser.add_argument("--top-k-experts", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/diagnostics/expert_specialization")
    parser.add_argument("--run-name", default="router_specialization_analysis")
    args = parser.parse_args()

    base_config = config_from_checkpoint(args.checkpoint) if args.checkpoint else None
    config = config_from_args(args, base=base_config)
    set_seed(config.seed)
    records = load_records(config.data_path, max_samples=args.max_samples)
    if not records:
        raise ValueError(f"No valid records found in {config.data_path}")

    model, tokenizer = build_model_and_tokenizer(config)
    device = torch.device(config.device)
    model.to(device)
    checkpoint_info: dict[str, Any] = {}
    if args.checkpoint:
        checkpoint_info = load_checkpoint(model, args.checkpoint, device)

    dataset = PasswordDataset(records, tokenizer, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    metrics = evaluate_router_distribution(model, loader, config.device)

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_name": args.run_name,
        "data_path": args.data_path,
        "checkpoint": args.checkpoint,
        "num_loaded_records": len(records),
        "config": config.to_dict(),
        "checkpoint_epoch": checkpoint_info.get("epoch") if checkpoint_info else None,
        "metrics": metrics,
        "expert_order": ["pii", "entropy", "leet"],
        "feature_order": ["pii", "leet", "entropy"],
        "claim_boundary": (
            "This is a mechanism diagnostic for router specialization. It is not an SR@K "
            "comparison and does not replace formal PassMoE validation."
        ),
    }
    json_path = out_dir / "expert_specialization.json"
    md_path = out_dir / "expert_specialization.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "metrics": metrics}, indent=2))


def config_from_checkpoint(checkpoint: str) -> Config:
    payload = torch.load(checkpoint, map_location="cpu")
    valid_payload = {key: value for key, value in payload.get("config", {}).items() if key in CONFIG_FIELDS}
    return Config(**valid_payload)


def config_from_args(args: argparse.Namespace, base: Config | None = None) -> Config:
    config = base or Config()
    config.data_path = args.data_path
    config.checkpoint = args.checkpoint
    if args.base_model is not None:
        config.base_model = args.base_model
    elif base is None:
        config.base_model = "tiny"
    if args.base_adapter is not None:
        config.base_adapter = args.base_adapter
    elif base is None:
        config.base_adapter = ""
    if args.task is not None:
        config.task = args.task
    elif base is None:
        config.task = "targeted"
    if args.prompt_template_id is not None:
        config.prompt_template_id = args.prompt_template_id
    elif base is None:
        config.prompt_template_id = "0"
    config.max_train_samples = args.max_samples
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    elif base is None:
        config.batch_size = 16
    if args.max_length is not None:
        config.max_length = args.max_length
    elif base is None:
        config.max_length = 256
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    elif base is None:
        config.hidden_dim = 64
    if args.lora_rank is not None:
        config.lora_rank = args.lora_rank
    elif base is None:
        config.lora_rank = 8
    if args.router_hidden_dim is not None:
        config.router_hidden_dim = args.router_hidden_dim
    if args.top_k_experts is not None:
        config.top_k_experts = args.top_k_experts
    elif base is None:
        config.top_k_experts = 1
    config.device = resolve_device(args.device or config.device)
    if args.dtype is not None:
        config.dtype = args.dtype
    config.seed = args.seed
    config.output_dir = args.output_dir
    config.run_name = args.run_name

    if config.base_model == "local-qwen":
        config.base_model = config.local_qwen_05b
    adapter = str(config.base_adapter).lower()
    if adapter in {"none", "null", "-"}:
        config.base_adapter = ""
    elif config.base_adapter == "fielddrop":
        config.base_adapter = config.local_fielddrop_adapter
    elif config.base_adapter == "baseline10k":
        config.base_adapter = config.local_baseline10k_adapter
    elif config.base_adapter == "csdn":
        config.base_adapter = config.local_csdn_adapter
    return config


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# PassMoE Expert Specialization Diagnostic",
        "",
        f"- run: `{report['run_name']}`",
        f"- records: `{report['num_loaded_records']}`",
        f"- expert order: `{', '.join(report['expert_order'])}`",
        f"- feature order: `{', '.join(report['feature_order'])}`",
        f"- top-1 agreement with weak feature label: `{metrics.get('specialization_top1_agreement', 0.0):.4f}`",
        "",
        "## Overall Router Weights",
        "",
        "| Expert | Average Weight | Top-1 Count | Weak Target Count |",
        "| --- | ---: | ---: | ---: |",
    ]
    top1 = metrics.get("top1_counts", {})
    target = metrics.get("weak_target_counts", {})
    rows = [
        ("pii", metrics.get("avg_pii_expert_weight", 0.0)),
        ("entropy", metrics.get("avg_entropy_expert_weight", 0.0)),
        ("leet", metrics.get("avg_leet_expert_weight", 0.0)),
    ]
    for name, avg in rows:
        lines.append(f"| {name} | {avg:.4f} | {int(top1.get(name, 0))} | {int(target.get(name, 0))} |")

    lines.extend(
        [
            "",
            "## Feature Buckets",
            "",
            "| Bucket | Count | Avg PII | Avg Entropy | Avg Leet | Top PII | Top Entropy | Top Leet |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket, item in sorted((metrics.get("feature_buckets") or {}).items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    bucket,
                    str(item.get("count", 0)),
                    f"{item.get('avg_pii_expert_weight', 0.0):.4f}",
                    f"{item.get('avg_entropy_expert_weight', 0.0):.4f}",
                    f"{item.get('avg_leet_expert_weight', 0.0):.4f}",
                    f"{item.get('top_pii_expert_fraction', 0.0):.4f}",
                    f"{item.get('top_entropy_expert_fraction', 0.0):.4f}",
                    f"{item.get('top_leet_expert_fraction', 0.0):.4f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
