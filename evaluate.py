from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import PasswordRecord, format_targeted_prompt
from trainer import batch_loss_weight, move_batch


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: str | torch.device) -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(**batch)
        loss = outputs["loss"]
        weight = batch_loss_weight(outputs, batch)
        total_loss += float(loss.detach().cpu()) * weight
        total_items += weight
    avg_loss = total_loss / max(total_items, 1)
    return {"loss": avg_loss, "perplexity": float(math.exp(min(avg_loss, 20.0))), "valid_tokens": total_items}


@torch.no_grad()
def generate_candidates(
    model: Any,
    tokenizer: Any,
    config: Config,
    prefix: str = "",
    num_passwords: int | None = None,
) -> list[tuple[str, float]]:
    return model.generate_passwords(
        tokenizer=tokenizer,
        prefix=prefix,
        num_passwords=num_passwords or config.num_passwords,
        max_length=config.generation_max_new_tokens,
        beam_width=config.beam_width,
        temperature=config.temperature,
        device=config.device,
    )


def hit_rate_at_k(
    candidates: Iterable[str],
    records: Iterable[PasswordRecord],
    budgets: Iterable[int],
) -> dict[str, float | int]:
    ordered = list(dict.fromkeys(candidates))
    target_set = {record.password for record in records}
    metrics: dict[str, float | int] = {"num_targets": len(target_set), "num_candidates": len(ordered)}
    for budget in sorted(budgets):
        guessed = set(ordered[:budget])
        hits = guessed.intersection(target_set)
        metrics[f"hits@{budget}"] = len(hits)
        metrics[f"hit_rate@{budget}"] = len(hits) / max(len(target_set), 1)
    return metrics


def score_ranked_jsonl(
    path: str | Path,
    budgets: Iterable[int],
    recompute_from_candidates: bool = False,
) -> dict[str, Any]:
    """Score PassLLM/PassMoE JSONL rows with min_cracked_guess_number."""

    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    metrics: dict[str, Any] = {
        "path": str(path.resolve()),
        "num_rows": len(rows),
    }
    if recompute_from_candidates:
        ranks = [rank_from_output_passwords(row) for row in rows]
    else:
        ranks = [int(row.get("min_cracked_guess_number", 0) or 0) for row in rows]
    metrics["rank_source"] = "outputPasswords" if recompute_from_candidates else "min_cracked_guess_number"
    for budget in sorted({int(budget) for budget in budgets}):
        hits = sum(1 for rank in ranks if 1 <= rank <= budget)
        metrics[f"hits@{budget}"] = hits
        metrics[f"sr@{budget}"] = hits / max(len(rows), 1)
    return metrics


def rank_from_output_passwords(row: dict[str, Any]) -> int:
    target = str(row.get("real password", row.get("real_password", row.get("password", ""))))
    output_passwords = row.get("outputPasswords", [])
    if not target or not isinstance(output_passwords, list):
        return 0
    for index, item in enumerate(output_passwords, start=1):
        candidate = output_password_text(item)
        if candidate == target:
            return index
    return 0


def output_password_text(item: Any) -> str:
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    if isinstance(item, dict):
        return str(item.get("password", item.get("candidate", "")))
    return str(item)


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    records: list[PasswordRecord],
    config: Config,
    output_dir: str | Path,
) -> dict[str, Any]:
    if config.task == "targeted":
        return evaluate_targeted_generation(model, tokenizer, records, config, output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(model, tokenizer, config, num_passwords=config.num_passwords)
    candidate_path = output_dir / "generated_candidates.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "password", "score"])
        for rank, (password, score) in enumerate(candidates, start=1):
            writer.writerow([rank, password, score])

    metrics = hit_rate_at_k(
        [password for password, _score in candidates],
        records,
        config.budgets_list(),
    )
    metrics["candidate_path"] = str(candidate_path.resolve())
    (output_dir / "generation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_targeted_generation(
    model: Any,
    tokenizer: Any,
    records: list[PasswordRecord],
    config: Config,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_records = [record for record in records if record.pii][: config.target_eval_samples]
    jsonl_path = output_dir / "targeted_input_output.jsonl"
    resumed_rows = (
        load_completed_targeted_rows(jsonl_path, eval_records, config.target_candidates_per_user)
        if config.resume_generation
        else {}
    )
    if config.resume_generation and resumed_rows:
        compact_targeted_jsonl(jsonl_path, resumed_rows)

    rows = []
    ranks = []
    started = time.perf_counter()
    progress_every = max(1, len(eval_records) // 20) if eval_records else 1
    mode = "a" if config.resume_generation and resumed_rows else "w"
    with jsonl_path.open(mode, encoding="utf-8") as handle:
        for index, record in enumerate(tqdm(eval_records, desc="targeted-generation", leave=False)):
            if index in resumed_rows:
                row = resumed_rows[index]
                rows.append(row)
                ranks.append(int(row.get("min_cracked_guess_number", 0) or 0))
                maybe_emit_targeted_progress(
                    rows=rows,
                    ranks=ranks,
                    total=len(eval_records),
                    budgets=config.budgets_list(),
                    started=started,
                    every=progress_every,
                    result_path=jsonl_path,
                    candidates_per_user=config.target_candidates_per_user,
                    generation_batch_size=config.generation_batch_size,
                    resumed_rows=len(resumed_rows),
                )
                continue

            prompt = format_targeted_prompt(record.pii or {}, config.prompt_template_id)
            candidates = model.generate_passwords(
                tokenizer=tokenizer,
                prefix=prompt,
                num_passwords=config.target_candidates_per_user,
                max_length=config.generation_max_new_tokens,
                beam_width=config.beam_width,
                temperature=config.temperature,
                device=config.device,
                strip_prefix=True,
                pii=record.pii or {},
            )
            passwords = [password for password, _score in candidates]
            rank = 0
            for pos, password in enumerate(passwords, start=1):
                if password == record.password:
                    rank = pos
                    break
            row = {
                "index": index,
                "min_cracked_guess_number": rank,
                "model_input": prompt,
                "real password": record.password,
                "outputPasswords": candidates,
            }
            ranks.append(rank)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            maybe_emit_targeted_progress(
                rows=rows,
                ranks=ranks,
                total=len(eval_records),
                budgets=config.budgets_list(),
                started=started,
                every=progress_every,
                result_path=jsonl_path,
                candidates_per_user=config.target_candidates_per_user,
                generation_batch_size=config.generation_batch_size,
                resumed_rows=len(resumed_rows),
            )

    metrics: dict[str, Any] = {
        "num_targets": len(eval_records),
        "num_completed": len(rows),
        "resumed_rows": len(resumed_rows),
        "resume_generation": bool(config.resume_generation),
        "complete": len(rows) == len(eval_records),
        "candidates_per_user": config.target_candidates_per_user,
        "generation_max_new_tokens": config.generation_max_new_tokens,
        "generation_batch_size": config.generation_batch_size,
        "result_path": str(jsonl_path.resolve()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    for budget in config.budgets_list():
        hits = sum(1 for rank in ranks if 1 <= rank <= budget)
        metrics[f"hits@{budget}"] = hits
        metrics[f"hit_rate@{budget}"] = hits / max(len(eval_records), 1)

    (output_dir / "targeted_generation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def maybe_emit_targeted_progress(
    rows: list[dict[str, Any]],
    ranks: list[int],
    total: int,
    budgets: list[int],
    started: float,
    every: int,
    result_path: Path,
    candidates_per_user: int,
    generation_batch_size: int,
    resumed_rows: int,
) -> None:
    completed = len(rows)
    if completed <= 0:
        return
    if completed != total and completed % max(every, 1) != 0:
        return
    elapsed = time.perf_counter() - started
    generated_rows = max(completed - resumed_rows, 0)
    remaining_rows = max(total - completed, 0)
    seconds_per_completed_row = elapsed / max(completed, 1)
    seconds_per_generated_row = elapsed / generated_rows if generated_rows else None
    eta_seconds = 0.0 if remaining_rows == 0 else None
    if remaining_rows and seconds_per_generated_row is not None:
        eta_seconds = seconds_per_generated_row * remaining_rows
    payload: dict[str, Any] = {
        "event": "targeted_generation_progress",
        "completed": completed,
        "total": total,
        "fraction": completed / max(total, 1),
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_row": round(seconds_per_completed_row, 3),
        "generated_rows_this_run": generated_rows,
        "remaining_rows": remaining_rows,
        "seconds_per_generated_row": round(seconds_per_generated_row, 3) if seconds_per_generated_row is not None else None,
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "estimated_total_seconds": round(elapsed + eta_seconds, 3) if eta_seconds is not None else None,
        "resumed_rows": resumed_rows,
        "candidates_per_user": candidates_per_user,
        "generation_batch_size": generation_batch_size,
        "result_path": str(result_path.resolve()),
    }
    for budget in budgets:
        hits = sum(1 for rank in ranks if 1 <= rank <= budget)
        payload[f"hits@{budget}"] = hits
    tqdm.write("__PASSMOE_PROGRESS__ " + json.dumps(payload, sort_keys=True))


def load_completed_targeted_rows(
    path: Path,
    eval_records: list[PasswordRecord],
    min_candidates: int = 0,
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                index = int(row.get("index", -1))
            except (TypeError, ValueError):
                continue
            if not (0 <= index < len(eval_records)):
                continue
            if str(row.get("real password", "")) != eval_records[index].password:
                continue
            if not is_reusable_targeted_row(row, min_candidates):
                continue
            row["min_cracked_guess_number"] = rank_from_output_passwords(row)
            rows[index] = row
    return rows


def is_reusable_targeted_row(row: dict[str, Any], min_candidates: int = 0) -> bool:
    candidates = row.get("outputPasswords")
    if not isinstance(candidates, list):
        return False
    passwords = [output_password_text(item) for item in candidates]
    nonempty_passwords = [password for password in passwords if password]
    if min_candidates and len(nonempty_passwords) < min_candidates:
        return False
    if len(set(nonempty_passwords)) != len(nonempty_passwords):
        return False
    model_input = str(row.get("model_input", ""))
    if model_input and any(password.startswith(model_input) for password in nonempty_passwords):
        return False
    return True


def compact_targeted_jsonl(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in sorted(rows):
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")


def evaluate_router_distribution(
    model: Any,
    loader: DataLoader,
    device: str | torch.device,
) -> dict[str, Any]:
    device = torch.device(device)
    totals = torch.zeros(3)
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="router", leave=False):
            batch = move_batch(batch, device)
            outputs = model(**batch)
            weights = outputs["expert_weights"].detach().cpu()
            totals += weights.sum(dim=0)
            count += weights.size(0)
    avg = (totals / max(count, 1)).tolist()
    return {
        "avg_pii_expert_weight": avg[0],
        "avg_entropy_expert_weight": avg[1],
        "avg_leet_expert_weight": avg[2],
    }
