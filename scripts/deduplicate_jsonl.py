from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate a JSONL file by a stable row key.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--key", default="index", help="Top-level JSON key used for deduplication.")
    parser.add_argument("--policy", choices=["first", "last"], default="first")
    parser.add_argument("--report", default="", help="Optional report JSON path.")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    report_path = resolve_path(args.report) if args.report else output_path.with_suffix(output_path.suffix + ".report.json")

    rows = load_jsonl(input_path)
    deduped_rows, duplicate_keys = deduplicate_rows(rows, args.key, args.policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in deduped_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "key": args.key,
        "policy": args.policy,
        "input_rows": len(rows),
        "output_rows": len(deduped_rows),
        "removed_rows": len(rows) - len(deduped_rows),
        "duplicate_keys": duplicate_keys,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def deduplicate_rows(rows: list[dict[str, Any]], key: str, policy: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = Counter(stable_key(row, key) for row in rows)
    duplicate_keys = {str(item): count for item, count in sorted(counts.items(), key=lambda pair: str(pair[0])) if count > 1}
    if policy == "first":
        selected: OrderedDict[Any, dict[str, Any]] = OrderedDict()
        for row in rows:
            selected.setdefault(stable_key(row, key), row)
        return list(selected.values()), duplicate_keys

    selected = OrderedDict()
    for row in rows:
        row_key = stable_key(row, key)
        if row_key in selected:
            del selected[row_key]
        selected[row_key] = row
    return list(selected.values()), duplicate_keys


def stable_key(row: dict[str, Any], key: str) -> Any:
    if key not in row:
        raise KeyError(f"Missing key {key!r} in row with keys={sorted(row.keys())}")
    value = row[key]
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number}, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


if __name__ == "__main__":
    main()
