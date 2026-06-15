from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_JSONL = (
    r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code\result\quick\fd500k_p00\input_output.jsonl"
)
DEFAULT_TRAIN_DATA = r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code\data\clixsense\clixsense_sample_10k.json"
DEFAULT_TEST_DATA = r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code\data\clixsense\clixsense_test_1000.json"
DEFAULT_EXPORT = r"data\clixsense\clixsense_test_500_from_fd500k_p00.json"
DEFAULT_FILTERED_TRAIN = r"data\clixsense\clixsense_train_50_no_fd500k_targets.jsonl"
DEFAULT_REPORT_JSON = r"artifacts\reports\target_alignment_audit.json"
DEFAULT_REPORT_MD = r"artifacts\reports\target_alignment_audit.md"

PASSLLM_P00_PREFIX = (
    "As a targeted password guessing model, your task is to utilize "
    "the provided account information to guess the password."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and export PassLLM quick-result target alignment.")
    parser.add_argument("--baseline-jsonl", default=DEFAULT_BASELINE_JSONL)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--export-targets", default=DEFAULT_EXPORT)
    parser.add_argument("--export-filtered-train", default="")
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--dedupe-policy", choices=["first", "last"], default="first")
    args = parser.parse_args()

    baseline_path = resolve_path(args.baseline_jsonl)
    train_path = resolve_path(args.train_data)
    test_path = resolve_path(args.test_data)
    export_path = resolve_path(args.export_targets)
    filtered_train_path = resolve_path(args.export_filtered_train) if args.export_filtered_train else None
    report_json_path = resolve_path(args.report_json)
    report_md_path = resolve_path(args.report_md)

    rows = load_jsonl(baseline_path)
    unique_rows = dedupe_rows(rows, args.dedupe_policy)
    targets = [row_to_target(row) for row in unique_rows]
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(json.dumps(targets, indent=2, ensure_ascii=False), encoding="utf-8")

    train_records = load_records(train_path)
    filtered_train_report = None
    if filtered_train_path is not None:
        filtered_train_report = export_filtered_train(train_path, filtered_train_path, targets)

    report = {
        "baseline_jsonl": str(baseline_path),
        "exported_targets": str(export_path),
        "dedupe_policy": args.dedupe_policy,
        "row_audit": row_audit(rows, unique_rows),
        "baseline_metrics": {
            "raw_rows": score_rows(rows),
            "unique_targets": score_rows(unique_rows),
        },
        "overlap": {
            "train_data": overlap_report(targets, train_records, str(train_path)),
            "test_data": overlap_report(targets, load_records(test_path), str(test_path)),
        },
        "filtered_train": filtered_train_report,
        "prompt_prefix": PASSLLM_P00_PREFIX,
        "prompt_template_id": "0",
        "recommendation": (
            "Use exported_targets as --test-data-path and prompt_template_id=0 for fair PassMoE-vs-PassLLM quick comparison."
        ),
    }
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report_json": str(report_json_path), "report_md": str(report_md_path)}, indent=2))


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    return []


def dedupe_rows(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    if policy == "first":
        deduped: OrderedDict[int, dict[str, Any]] = OrderedDict()
        for row in rows:
            deduped.setdefault(int(row["index"]), row)
        return list(deduped.values())

    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        latest[int(row["index"])] = row
    return [latest[index] for index in sorted(latest)]


def row_to_target(row: dict[str, Any]) -> dict[str, Any]:
    model_input = str(row.get("model_input", ""))
    if not model_input.startswith(PASSLLM_P00_PREFIX):
        raise ValueError(f"Unsupported prompt prefix for row index={row.get('index')}: {model_input[:80]}")
    knowledge_text = model_input[len(PASSLLM_P00_PREFIX) :]
    knowledge = json.loads(knowledge_text)
    return {
        "Knowledge": knowledge,
        "password": str(row.get("real password", "")),
        "source_index": int(row.get("index", -1)),
    }


def row_audit(rows: list[dict[str, Any]], unique_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(row.get("index", -1)) for row in rows)
    duplicate_indices = {str(index): count for index, count in sorted(counts.items()) if count > 1}
    return {
        "raw_rows": len(rows),
        "unique_indices": len(counts),
        "unique_targets": len(unique_rows),
        "duplicate_indices": duplicate_indices,
        "min_index": min(counts) if counts else None,
        "max_index": max(counts) if counts else None,
    }


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(row.get("min_cracked_guess_number", 0) or 0) for row in rows]
    metrics: dict[str, Any] = {"n": len(rows)}
    for budget in (1, 10, 50, 100):
        hits = sum(1 for rank in ranks if 1 <= rank <= budget)
        metrics[f"hits@{budget}"] = hits
        metrics[f"sr@{budget}"] = hits / max(len(rows), 1)
    return metrics


def overlap_report(targets: list[dict[str, Any]], records: list[dict[str, Any]], path: str) -> dict[str, Any]:
    target_sigs = {record_signature(target) for target in targets}
    record_sigs = {record_signature(record) for record in records}
    matched = target_sigs.intersection(record_sigs)
    password_only_targets = Counter(str(target.get("password", "")) for target in targets)
    password_only_records = Counter(str(record.get("password", record.get("real password", record.get("output", "")))) for record in records)
    password_overlap = sum(min(count, password_only_records.get(password, 0)) for password, count in password_only_targets.items())
    return {
        "path": path,
        "records": len(records),
        "exact_target_overlap": len(matched),
        "exact_target_overlap_fraction": len(matched) / max(len(targets), 1),
        "password_only_overlap_count": password_overlap,
        "password_only_overlap_fraction": password_overlap / max(len(targets), 1),
    }


def export_filtered_train(train_path: Path, output_path: Path, targets: list[dict[str, Any]]) -> dict[str, Any]:
    target_sigs = {record_signature(target) for target in targets}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    read_count = 0
    written_count = 0
    excluded_count = 0

    if train_path.suffix.lower() == ".jsonl":
        with train_path.open("r", encoding="utf-8-sig", errors="ignore") as src, output_path.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                read_count += 1
                if record_signature(record) in target_sigs:
                    excluded_count += 1
                    continue
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_count += 1
    else:
        records = load_records(train_path)
        with output_path.open("w", encoding="utf-8") as dst:
            for record in records:
                read_count += 1
                if record_signature(record) in target_sigs:
                    excluded_count += 1
                    continue
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_count += 1

    return {
        "input_path": str(train_path),
        "output_path": str(output_path),
        "read_records": read_count,
        "written_records": written_count,
        "excluded_exact_target_records": excluded_count,
    }


def record_signature(record: dict[str, Any]) -> str:
    password = str(record.get("password", record.get("real password", record.get("output", ""))))
    knowledge = record.get("Knowledge") or record.get("knowledge") or record.get("pii") or {}
    if not isinstance(knowledge, dict):
        knowledge = {}
    return json.dumps(
        {"password": password, "Knowledge": knowledge},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def render_markdown(report: dict[str, Any]) -> str:
    raw = report["baseline_metrics"]["raw_rows"]
    unique = report["baseline_metrics"]["unique_targets"]
    row = report["row_audit"]
    lines = [
        "# Target Alignment Audit",
        "",
        f"- baseline JSONL: `{report['baseline_jsonl']}`",
        f"- exported targets: `{report['exported_targets']}`",
        f"- dedupe policy: `{report['dedupe_policy']}`",
        f"- raw rows: `{row['raw_rows']}`",
        f"- unique targets: `{row['unique_targets']}`",
        f"- duplicate indices: `{row['duplicate_indices']}`",
        "",
        "## Baseline Metrics",
        "",
        "| Policy | n | SR@1 | SR@10 | SR@50 | SR@100 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| raw rows | {raw['n']} | {raw['sr@1']:.4f} | {raw['sr@10']:.4f} | "
            f"{raw['sr@50']:.4f} | {raw['sr@100']:.4f} |"
        ),
        (
            f"| unique targets | {unique['n']} | {unique['sr@1']:.4f} | {unique['sr@10']:.4f} | "
            f"{unique['sr@50']:.4f} | {unique['sr@100']:.4f} |"
        ),
        "",
        "## Overlap",
        "",
        "| Dataset | records | exact target overlap | password-only overlap |",
        "|---|---:|---:|---:|",
    ]
    for label, overlap in report["overlap"].items():
        lines.append(
            f"| {label} | {overlap['records']} | "
            f"{overlap['exact_target_overlap']} ({overlap['exact_target_overlap_fraction']:.3f}) | "
            f"{overlap['password_only_overlap_count']} ({overlap['password_only_overlap_fraction']:.3f}) |"
        )
    if report.get("filtered_train"):
        filtered = report["filtered_train"]
        lines.extend(
            [
                "",
                "## Filtered Train Export",
                "",
                f"- input: `{filtered['input_path']}`",
                f"- output: `{filtered['output_path']}`",
                f"- read records: `{filtered['read_records']}`",
                f"- written records: `{filtered['written_records']}`",
                f"- excluded exact target records: `{filtered['excluded_exact_target_records']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            report["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
