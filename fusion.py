from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


COMMON_SUFFIXES = ("", "1", "12", "123", "1234", "!", "@", "01", "02", "007")
COMMON_PREFIXES = ("", "i", "my")
LEET_MAP = str.maketrans({"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"})


def fuse_ranked_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    strategy: str = "insert",
    insert_after: int = 10,
    max_expert_candidates: int = 40,
    budgets: Iterable[int] = (1, 10, 50, 100),
    score_existing_weight: float = 1.0,
    score_expert_weight: float = 0.05,
    score_rank_offset: float = 2.0,
) -> dict[str, Any]:
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    ranks = []
    expert_hits = 0
    changed_hits = 0
    with input_jsonl.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            original_rank = int(row.get("min_cracked_guess_number", 0) or 0)
            pii = parse_pii_from_model_input(str(row.get("model_input", "")))
            expert_candidates = generate_expert_candidates(pii, max_candidates=max_expert_candidates)
            existing = normalize_output_passwords(row.get("outputPasswords", []))
            fused = fuse_candidates(
                existing,
                expert_candidates,
                strategy=strategy,
                insert_after=insert_after,
                score_existing_weight=score_existing_weight,
                score_expert_weight=score_expert_weight,
                score_rank_offset=score_rank_offset,
            )
            real_password = str(row.get("real password", row.get("real_password", row.get("password", ""))))
            new_rank = rank_of(real_password, [password for password, _score in fused])
            expert_rank = rank_of(real_password, expert_candidates)
            if expert_rank > 0:
                expert_hits += 1
            if (original_rank == 0 or (0 < new_rank < original_rank)) and new_rank > 0:
                changed_hits += 1
            row["passmoe_fusion"] = {
                "strategy": strategy,
                "insert_after": insert_after,
                "max_expert_candidates": max_expert_candidates,
                "score_existing_weight": score_existing_weight,
                "score_expert_weight": score_expert_weight,
                "score_rank_offset": score_rank_offset,
                "num_expert_candidates": len(expert_candidates),
                "expert_rank": expert_rank,
                "original_min_cracked_guess_number": original_rank,
            }
            row["min_cracked_guess_number"] = new_rank
            row["outputPasswords"] = fused
            ranks.append(new_rank)
            rows.append(row)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics: dict[str, Any] = {
        "input_jsonl": str(input_jsonl.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "strategy": strategy,
        "insert_after": insert_after,
        "max_expert_candidates": max_expert_candidates,
        "score_existing_weight": score_existing_weight,
        "score_expert_weight": score_expert_weight,
        "score_rank_offset": score_rank_offset,
        "num_rows": len(rows),
        "expert_candidate_coverage": expert_hits / max(len(rows), 1),
        "improved_or_new_hits": changed_hits,
    }
    for budget in sorted({int(budget) for budget in budgets}):
        hits = sum(1 for rank in ranks if 1 <= rank <= budget)
        metrics[f"hits@{budget}"] = hits
        metrics[f"sr@{budget}"] = hits / max(len(rows), 1)
    return metrics


def analyze_fusion_pair(
    original_jsonl: str | Path,
    fused_jsonl: str | Path,
    budgets: Iterable[int] = (1, 10, 50, 100),
    bootstrap_iters: int = 2000,
    seed: int = 42,
    max_examples: int = 20,
) -> dict[str, Any]:
    original_rows = read_jsonl(original_jsonl)
    fused_rows = read_jsonl(fused_jsonl)
    if len(original_rows) != len(fused_rows):
        raise ValueError(f"Row count mismatch: {len(original_rows)} vs {len(fused_rows)}")

    original_ranks = [rank_from_row(row) for row in original_rows]
    fused_ranks = [rank_from_row(row) for row in fused_rows]
    budgets = sorted({int(budget) for budget in budgets})

    result: dict[str, Any] = {
        "original_jsonl": str(Path(original_jsonl).resolve()),
        "fused_jsonl": str(Path(fused_jsonl).resolve()),
        "num_rows": len(original_rows),
        "bootstrap_iters": bootstrap_iters,
        "seed": seed,
        "budgets": {},
        "rank_changes": summarize_rank_changes(original_rows, original_ranks, fused_ranks, max_examples),
    }

    for budget in budgets:
        original_hits = [1 if 1 <= rank <= budget else 0 for rank in original_ranks]
        fused_hits = [1 if 1 <= rank <= budget else 0 for rank in fused_ranks]
        deltas = [fused - original for original, fused in zip(original_hits, fused_hits)]
        delta_mean = sum(deltas) / max(len(deltas), 1)
        ci_low, ci_high = paired_bootstrap_ci(deltas, bootstrap_iters=bootstrap_iters, seed=seed + budget)
        result["budgets"][str(budget)] = {
            "original_hits": sum(original_hits),
            "fused_hits": sum(fused_hits),
            "original_sr": sum(original_hits) / max(len(original_hits), 1),
            "fused_sr": sum(fused_hits) / max(len(fused_hits), 1),
            "delta_hits": sum(deltas),
            "delta_sr": delta_mean,
            "delta_sr_ci95": [ci_low, ci_high],
        }

    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank_from_row(row: dict[str, Any]) -> int:
    target = str(row.get("real password", row.get("real_password", row.get("password", ""))))
    return rank_of(target, [password for password, _score in normalize_output_passwords(row.get("outputPasswords", []))])


def summarize_rank_changes(
    rows: list[dict[str, Any]],
    original_ranks: list[int],
    fused_ranks: list[int],
    max_examples: int,
) -> dict[str, Any]:
    changed = []
    improved = []
    worsened = []
    new_hits = []
    lost_hits = []
    for idx, (row, original_rank, fused_rank) in enumerate(zip(rows, original_ranks, fused_ranks)):
        if original_rank == fused_rank:
            continue
        item = {
            "row": idx,
            "index": row.get("index", idx),
            "target": row.get("real password", row.get("password", "")),
            "original_rank": original_rank,
            "fused_rank": fused_rank,
        }
        changed.append(item)
        if fused_rank and (original_rank == 0 or fused_rank < original_rank):
            improved.append(item)
            if original_rank == 0:
                new_hits.append(item)
        elif original_rank and (fused_rank == 0 or fused_rank > original_rank):
            worsened.append(item)
            if fused_rank == 0:
                lost_hits.append(item)
    return {
        "changed": len(changed),
        "improved": len(improved),
        "worsened": len(worsened),
        "new_hits": len(new_hits),
        "lost_hits": len(lost_hits),
        "improved_examples": improved[:max_examples],
        "worsened_examples": worsened[:max_examples],
    }


def paired_bootstrap_ci(deltas: list[int], bootstrap_iters: int, seed: int) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    samples = []
    for _ in range(max(bootstrap_iters, 1)):
        total = 0
        for _j in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    low_idx = int(0.025 * (len(samples) - 1))
    high_idx = int(0.975 * (len(samples) - 1))
    return samples[low_idx], samples[high_idx]


def parse_pii_from_model_input(model_input: str) -> dict[str, Any]:
    brace = model_input.find("{")
    if brace < 0:
        return {}
    decoder = json.JSONDecoder()
    try:
        payload, _idx = decoder.raw_decode(model_input[brace:])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def normalize_output_passwords(output_passwords: Any) -> list[tuple[str, float]]:
    normalized: list[tuple[str, float]] = []
    if not isinstance(output_passwords, list):
        return normalized
    for item in output_passwords:
        if isinstance(item, (list, tuple)) and item:
            password = str(item[0])
            score = float(item[1]) if len(item) > 1 and _is_number(item[1]) else 0.0
            normalized.append((password, score))
        elif isinstance(item, dict):
            password = str(item.get("password", item.get("candidate", "")))
            score = float(item.get("probability", item.get("score", 0.0)) or 0.0)
            if password:
                normalized.append((password, score))
    return normalized


def fuse_candidates(
    existing: list[tuple[str, float]],
    expert_candidates: list[str],
    strategy: str,
    insert_after: int,
    score_existing_weight: float = 1.0,
    score_expert_weight: float = 0.05,
    score_rank_offset: float = 2.0,
) -> list[tuple[str, float]]:
    expert_scored = [(candidate, 0.0) for candidate in expert_candidates]
    if strategy == "append":
        ordered = existing + expert_scored
    elif strategy == "prepend":
        ordered = expert_scored + existing
    elif strategy == "insert":
        head = existing[:insert_after]
        tail = existing[insert_after:]
        ordered = head + expert_scored + tail
    elif strategy == "score":
        ordered = score_fusion(
            existing,
            expert_candidates,
            existing_weight=score_existing_weight,
            expert_weight=score_expert_weight,
            rank_offset=score_rank_offset,
        )
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")
    return dedupe_ranked(ordered)


def score_fusion(
    existing: list[tuple[str, float]],
    expert_candidates: list[str],
    existing_weight: float = 1.0,
    expert_weight: float = 0.05,
    rank_offset: float = 2.0,
) -> list[tuple[str, float]]:
    expert_rank = {candidate: rank for rank, candidate in enumerate(expert_candidates, start=1)}
    existing_rank = {candidate: rank for rank, (candidate, _score) in enumerate(existing, start=1)}
    all_candidates = list(existing_rank)
    for candidate in expert_candidates:
        if candidate not in existing_rank:
            all_candidates.append(candidate)
    scored = []
    for candidate in all_candidates:
        score = 0.0
        if candidate in existing_rank:
            score += existing_weight / (existing_rank[candidate] + rank_offset)
        if candidate in expert_rank:
            score += expert_weight / (expert_rank[candidate] + rank_offset)
        scored.append((candidate, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def dedupe_ranked(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    seen = set()
    deduped = []
    for password, score in candidates:
        if not password or password in seen:
            continue
        seen.add(password)
        deduped.append((password, score))
    return deduped


def generate_expert_candidates(pii: dict[str, Any], max_candidates: int = 40) -> list[str]:
    tokens = extract_pii_tokens(pii)
    dates = extract_birth_variants(pii)
    candidates: list[str] = []

    def add(value: str) -> None:
        value = sanitize_password(value)
        if value and value not in candidates:
            candidates.append(value)

    for token in tokens:
        add(token)
        add(token.lower())
        add(token.capitalize())
        add(token.translate(LEET_MAP))
        for suffix in COMMON_SUFFIXES:
            add(token.lower() + suffix)
        for prefix in COMMON_PREFIXES:
            add(prefix + token.lower())
        for date in dates[:10]:
            add(token.lower() + date)
            add(token.capitalize() + date)

    for date in dates:
        add(date)
        for token in tokens[:8]:
            add(date + token.lower())

    # Cross-field combinations, useful for account/name/email local-part reuse.
    for left in tokens[:8]:
        for right in tokens[:8]:
            if left == right:
                continue
            add(left.lower() + right.lower())
            add(left.lower() + right.lower()[:3])

    return candidates[:max_candidates]


def extract_pii_tokens(pii: dict[str, Any]) -> list[str]:
    raw_tokens: list[str] = []
    priority_keys = ("account", "username", "email", "first_name", "family_name", "name", "phone")
    for key in priority_keys:
        if key in pii:
            raw_tokens.extend(tokens_from_value(pii[key]))
    for key, value in pii.items():
        if key not in priority_keys and key.lower() not in {"birth", "birthday", "dob"}:
            raw_tokens.extend(tokens_from_value(value))

    tokens: list[str] = []
    for token in raw_tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        for candidate in split_compound_token(token):
            if len(candidate) >= 2 and candidate not in tokens:
                tokens.append(candidate)
    return tokens


def tokens_from_value(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    tokens: list[str] = []
    for item in values:
        text = str(item)
        if "@" in text:
            local = text.split("@", 1)[0]
            tokens.append(local)
            tokens.append(re.sub(r"[^A-Za-z0-9]", "", local))
        tokens.extend(re.findall(r"[A-Za-z0-9]{2,}", text))
    return tokens


def split_compound_token(token: str) -> list[str]:
    parts = [token]
    stripped = re.sub(r"[^A-Za-z0-9]", "", token)
    if stripped and stripped != token:
        parts.append(stripped)
    alpha = re.sub(r"[^A-Za-z]", "", token)
    digits = re.sub(r"[^0-9]", "", token)
    if len(alpha) >= 2:
        parts.append(alpha)
    if len(digits) >= 2:
        parts.append(digits)
    return list(dict.fromkeys(parts))


def extract_birth_variants(pii: dict[str, Any]) -> list[str]:
    raw = ""
    for key in ("birth", "birthday", "dob", "date_of_birth"):
        if key in pii:
            raw = str(pii[key])
            break
    digits = re.sub(r"[^0-9]", "", raw)
    if len(digits) < 4:
        return []

    variants: list[str] = []

    def add(value: str) -> None:
        if value and value not in variants:
            variants.append(value)

    if len(digits) >= 8:
        year, month, day = digits[:4], digits[4:6], digits[6:8]
        yy = year[-2:]
        for value in (
            year + month + day,
            day + month + year,
            month + day + year,
            yy + month + day,
            day + month + yy,
            month + day + yy,
            year,
            yy,
            month + day,
            day + month,
            day + month + year[-2:],
        ):
            add(value)
    else:
        add(digits)
        add(digits[-2:])
        add(digits[-4:])
    return variants


def rank_of(real_password: str, candidates: list[str]) -> int:
    for index, candidate in enumerate(candidates, start=1):
        if candidate == real_password:
            return index
    return 0


def sanitize_password(value: str) -> str:
    value = str(value).strip()
    value = value.replace(" ", "")
    return value[:64]


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
