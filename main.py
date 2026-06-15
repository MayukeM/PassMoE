from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from config import Config
from data import (
    FeatureExtractor,
    create_data_loaders,
    encode_targeted_record,
    format_targeted_prompt,
    load_records,
    split_records,
    write_smoke_dataset,
)
from evaluate import evaluate_generation, evaluate_loss, evaluate_router_distribution, score_ranked_jsonl
from fusion import analyze_fusion_pair, fuse_ranked_jsonl
from model import build_model_and_tokenizer, build_tokenizer, count_parameters
from trainer import Trainer, load_checkpoint, set_seed


CONFIG_FIELDS = {field.name for field in dataclasses.fields(Config)}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Revived PassMoE-P runner")
    sub = parser.add_subparsers(dest="command")

    smoke = sub.add_parser("smoke", help="Run a tiny end-to-end CPU smoke test")
    add_common_args(smoke)
    smoke.set_defaults(func=cmd_smoke)

    train = sub.add_parser("train", help="Train PassMoE on a password dataset")
    add_common_args(train)
    train.set_defaults(func=cmd_train)

    generate = sub.add_parser("generate", help="Generate password candidates from a checkpoint")
    add_common_args(generate)
    generate.add_argument("--prefix", default="", help="Optional generation prefix")
    generate.set_defaults(func=cmd_generate)

    evaluate = sub.add_parser("evaluate", help="Evaluate a checkpoint on a dataset")
    add_common_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    inspect = sub.add_parser("inspect-data", help="Inspect supported password data files")
    inspect.add_argument("--data-path", required=True)
    inspect.add_argument("--max-train-samples", type=int, default=10)
    inspect.set_defaults(func=cmd_inspect_data)

    lengths = sub.add_parser("inspect-targeted-lengths", help="Audit targeted prompt/password token coverage")
    add_common_args(lengths)
    lengths.add_argument("--max-lengths", default="128,256,384,512")
    lengths.add_argument("--out", default="")
    lengths.set_defaults(func=cmd_inspect_targeted_lengths)

    score = sub.add_parser("score-jsonl", help="Score PassLLM/PassMoE ranked JSONL output")
    score.add_argument("--jsonl", required=True)
    score.add_argument("--budgets", default=Config().budgets)
    score.add_argument("--out", default="")
    score.add_argument("--recompute-from-candidates", action="store_true")
    score.set_defaults(func=cmd_score_jsonl)

    fuse = sub.add_parser("fuse-jsonl", help="Fuse PassLLM candidates with PassMoE-style expert candidates")
    fuse.add_argument("--jsonl", required=True)
    fuse.add_argument("--out-jsonl", required=True)
    fuse.add_argument("--out-metrics", default="")
    fuse.add_argument("--strategy", choices=["append", "prepend", "insert", "score"], default="insert")
    fuse.add_argument("--insert-after", type=int, default=10)
    fuse.add_argument("--max-expert-candidates", type=int, default=40)
    fuse.add_argument("--score-existing-weight", type=float, default=1.0)
    fuse.add_argument("--score-expert-weight", type=float, default=0.05)
    fuse.add_argument("--score-rank-offset", type=float, default=2.0)
    fuse.add_argument("--budgets", default=Config().budgets)
    fuse.set_defaults(func=cmd_fuse_jsonl)

    analyze = sub.add_parser("analyze-fusion", help="Analyze paired original/fused ranked JSONL files")
    analyze.add_argument("--original-jsonl", required=True)
    analyze.add_argument("--fused-jsonl", required=True)
    analyze.add_argument("--budgets", default=Config().budgets)
    analyze.add_argument("--bootstrap-iters", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--out", default="")
    analyze.set_defaults(func=cmd_analyze_fusion)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=None, help="'tiny', 'local-qwen', or a HF model/path")
    parser.add_argument(
        "--base-adapter",
        default=None,
        help=(
            "'fielddrop', 'baseline10k', 'csdn', or a LoRA adapter directory. "
            "'fielddrop' is an imported PassLLM/FieldDrop baseline adapter, not a PassMoE method component."
        ),
    )
    parser.add_argument("--task", choices=["trawling", "targeted"], default=None)
    parser.add_argument("--prompt-template-id", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--test-data-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--no-resume-optimizer", dest="resume_optimizer", action="store_false", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--router-hidden-dim", type=int, default=None)
    parser.add_argument("--top-k-experts", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--num-passwords", type=int, default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--use-device-map", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--target-eval-samples", type=int, default=None)
    parser.add_argument("--target-candidates-per-user", type=int, default=None)
    parser.add_argument("--skip-generation", action="store_true", default=None)
    parser.add_argument("--resume-generation", action="store_true", default=None)


def cmd_smoke(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    config.base_model = "tiny"
    config.data_path = args.data_path or "data/smoke_passwords.csv"
    config.run_name = args.run_name or "smoke_tiny"
    config.epochs = args.epochs or 1
    config.batch_size = args.batch_size or 8
    config.hidden_dim = args.hidden_dim or 64
    config.lora_rank = args.lora_rank or 8
    config.beam_width = args.beam_width or 16
    config.num_passwords = args.num_passwords or 100
    write_smoke_dataset(config.data_path)
    metrics = run_train_pipeline(config)
    print(json.dumps(metrics, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    metrics = run_train_pipeline(config)
    print(json.dumps(metrics, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    config = config_from_checkpoint_or_args(args)
    model, tokenizer = build_model_and_tokenizer(config)
    model.to(torch.device(config.device))
    load_checkpoint(model, config.checkpoint, config.device)
    candidates = model.generate_passwords(
        tokenizer=tokenizer,
        prefix=args.prefix,
        num_passwords=config.num_passwords,
        beam_width=config.beam_width,
        max_length=config.generation_max_new_tokens,
        device=config.device,
    )
    output_dir = config.run_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "generated_only.json"
    out_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path.resolve()), "num_candidates": len(candidates)}, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = config_from_checkpoint_or_args(args)
    test_path = config.test_data_path or config.data_path
    records = load_records(test_path, max_samples=config.max_eval_samples)
    train_records, val_records = split_records(records, config.val_fraction, config.seed)
    eval_records = val_records if config.test_data_path == "" else records

    model, tokenizer = build_model_and_tokenizer(config)
    model.to(torch.device(config.device))
    load_checkpoint(model, config.checkpoint, config.device)
    _train_loader, eval_loader = create_data_loaders(train_records, eval_records, tokenizer, config)

    output_dir = config.run_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_metrics = evaluate_loss(model, eval_loader, config.device)
    gen_metrics = {"skipped": True} if config.skip_generation else evaluate_generation(model, tokenizer, eval_records, config, output_dir)
    router_metrics = evaluate_router_distribution(model, eval_loader, config.device)
    metrics = {"loss": loss_metrics, "generation": gen_metrics, "router": router_metrics}
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def cmd_inspect_data(args: argparse.Namespace) -> None:
    records = load_records(args.data_path, max_samples=args.max_train_samples)
    extractor = FeatureExtractor()
    preview = [
        {
            "password": record.password,
            "pii_keys": sorted(record.pii.keys()) if record.pii else [],
            "features": extractor.extract(record.password, record.pii),
        }
        for record in records[: args.max_train_samples]
    ]
    print(json.dumps({"num_preview": len(preview), "preview": preview}, indent=2))


def cmd_inspect_targeted_lengths(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    records = load_records(config.data_path, max_samples=config.max_train_samples)
    targeted_records = [record for record in records if record.pii]
    tokenizer = build_tokenizer(config)
    max_lengths = [int(item.strip()) for item in args.max_lengths.split(",") if item.strip()]
    report = targeted_length_report(targeted_records, tokenizer, max_lengths, config.prompt_template_id)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def targeted_length_report(
    records: list[Any],
    tokenizer: Any,
    max_lengths: list[int],
    prompt_template_id: str,
) -> dict[str, Any]:
    prompt_lengths = []
    full_lengths = []
    password_lengths = []
    per_length: dict[int, dict[str, Any]] = {
        length: {
            "max_length": length,
            "zero_valid_records": 0,
            "truncated_records": 0,
            "total_valid_tokens": 0,
            "valid_token_counts": [],
            "min_password_token_coverage": 1.0,
        }
        for length in sorted(set(max_lengths))
    }
    longest_prompts: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        prompt = format_targeted_prompt(record.pii or {}, prompt_template_id)
        full_text = prompt + record.password
        prompt_len = count_tokenized_text(tokenizer, prompt, add_bos=True, add_eos=False)
        full_len = count_tokenized_text(tokenizer, full_text, add_bos=True, add_eos=True)
        password_len = max(full_len - prompt_len, 0)
        prompt_lengths.append(prompt_len)
        full_lengths.append(full_len)
        password_lengths.append(password_len)
        longest_prompts.append(
            {
                "index": index,
                "prompt_tokens": prompt_len,
                "full_tokens": full_len,
                "password_tokens": password_len,
                "password": record.password,
                "pii_keys": sorted((record.pii or {}).keys()),
            }
        )

        for length, item in per_length.items():
            _encoded, labels = encode_targeted_record(tokenizer, record, length, prompt_template_id)
            valid_tokens = int(labels[1:].ne(-100).sum().item())
            item["valid_token_counts"].append(valid_tokens)
            item["total_valid_tokens"] += valid_tokens
            if valid_tokens == 0:
                item["zero_valid_records"] += 1
            if full_len > length:
                item["truncated_records"] += 1
            coverage = valid_tokens / max(password_len, 1)
            item["min_password_token_coverage"] = min(item["min_password_token_coverage"], coverage)

    num_records = len(records)
    length_rows = []
    for length, item in sorted(per_length.items()):
        valid_counts = item.pop("valid_token_counts")
        item["zero_valid_fraction"] = item["zero_valid_records"] / max(num_records, 1)
        item["truncated_fraction"] = item["truncated_records"] / max(num_records, 1)
        item["valid_token_stats"] = summarize_numbers(valid_counts)
        length_rows.append(item)

    longest_prompts.sort(key=lambda row: row["prompt_tokens"], reverse=True)
    return {
        "num_targeted_records": num_records,
        "prompt_template_id": prompt_template_id,
        "prompt_token_stats": summarize_numbers(prompt_lengths),
        "full_token_stats": summarize_numbers(full_lengths),
        "password_token_stats": summarize_numbers(password_lengths),
        "min_max_length_for_nonzero_labels": (max(prompt_lengths) + 1) if prompt_lengths else 0,
        "min_max_length_for_untruncated_full_records": max(full_lengths) if full_lengths else 0,
        "lengths": length_rows,
        "longest_prompts_preview": longest_prompts[:5],
    }


def count_tokenized_text(tokenizer: Any, text: str, add_bos: bool, add_eos: bool) -> int:
    if hasattr(tokenizer, "encode_password"):
        return len(text) + int(add_bos) + int(add_eos)

    eos = getattr(tokenizer, "eos_token", None) or ""
    encoded = tokenizer(
        text + (eos if add_eos else ""),
        add_special_tokens=add_bos,
        truncation=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def summarize_numbers(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def percentile(ordered_values: list[int], q: float) -> int:
    if not ordered_values:
        return 0
    index = min(len(ordered_values) - 1, max(0, int(round((len(ordered_values) - 1) * q))))
    return ordered_values[index]


def cmd_score_jsonl(args: argparse.Namespace) -> None:
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    metrics = score_ranked_jsonl(args.jsonl, budgets, recompute_from_candidates=args.recompute_from_candidates)
    text = json.dumps(metrics, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def cmd_fuse_jsonl(args: argparse.Namespace) -> None:
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    metrics = fuse_ranked_jsonl(
        input_jsonl=args.jsonl,
        output_jsonl=args.out_jsonl,
        strategy=args.strategy,
        insert_after=args.insert_after,
        max_expert_candidates=args.max_expert_candidates,
        budgets=budgets,
        score_existing_weight=args.score_existing_weight,
        score_expert_weight=args.score_expert_weight,
        score_rank_offset=args.score_rank_offset,
    )
    text = json.dumps(metrics, indent=2)
    if args.out_metrics:
        Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_metrics).write_text(text, encoding="utf-8")
    print(text)


def cmd_analyze_fusion(args: argparse.Namespace) -> None:
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    report = analyze_fusion_pair(
        original_jsonl=args.original_jsonl,
        fused_jsonl=args.fused_jsonl,
        budgets=budgets,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def run_train_pipeline(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    records = load_records(config.data_path, max_samples=config.max_train_samples)
    if len(records) < 2:
        raise ValueError(f"Need at least two valid passwords, got {len(records)} from {config.data_path}")

    train_records, val_records = split_records(records, config.val_fraction, config.seed)
    eval_records = val_records
    if config.test_data_path:
        eval_records = load_records(config.test_data_path, max_samples=config.max_eval_samples)
    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, val_loader = create_data_loaders(train_records, val_records, tokenizer, config)

    trainer = Trainer(model, tokenizer, train_loader, val_loader, config)
    train_metrics = trainer.train()
    gen_metrics = (
        {"skipped": True}
        if config.skip_generation
        else evaluate_generation(model, tokenizer, eval_records, config, config.run_dir())
    )
    router_metrics = evaluate_router_distribution(model, val_loader, config.device)
    all_metrics = {
        "train": train_metrics,
        "generation": gen_metrics,
        "router": router_metrics,
        "parameters": count_parameters(model),
        "generation_eval_source": config.test_data_path or "validation_split",
    }
    (config.run_dir() / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return all_metrics


def config_from_checkpoint_or_args(args: argparse.Namespace) -> Config:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    payload = checkpoint.get("config", {})
    valid_payload = {key: value for key, value in payload.items() if key in CONFIG_FIELDS}
    config = Config(**valid_payload)
    config.checkpoint = args.checkpoint
    return config_from_args(args, base=config)


def config_from_args(args: argparse.Namespace, base: Config | None = None) -> Config:
    config = base or Config()
    for field in CONFIG_FIELDS:
        if not hasattr(args, field):
            continue
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    if config.base_model == "local-qwen":
        config.base_model = config.local_qwen_05b
    if str(config.base_adapter).lower() in {"none", "null", "-"}:
        config.base_adapter = ""
    if config.base_adapter == "fielddrop":
        config.base_adapter = config.local_fielddrop_adapter
    elif config.base_adapter == "baseline10k":
        config.base_adapter = config.local_baseline10k_adapter
    elif config.base_adapter == "csdn":
        config.base_adapter = config.local_csdn_adapter
    if config.checkpoint:
        config.checkpoint = str(Path(config.checkpoint))
    return config


if __name__ == "__main__":
    main()
