from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_RUN_NAME = "qwen_fielddrop_base_identity_clixsense_500_raw"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "formal" / DEFAULT_FORMAL_RUN_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a formal PassMoE artifact directory is CUDA-ready.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--min-gpu-memory-gb", type=float, default=8.0)
    parser.add_argument("--recommended-gpu-memory-gb", type=float, default=12.0)
    args = parser.parse_args()

    artifacts_dir = resolve_path(args.artifacts_dir)
    report = build_cuda_readiness(
        artifacts_dir,
        min_gpu_memory_gb=args.min_gpu_memory_gb,
        recommended_gpu_memory_gb=args.recommended_gpu_memory_gb,
    )
    out_json = resolve_path(args.out_json) if args.out_json else artifacts_dir / "cuda_readiness.json"
    out_md = resolve_path(args.out_md) if args.out_md else artifacts_dir / "cuda_readiness.md"
    write_readiness_files(report, out_json, out_md)
    print(json.dumps({"status": report["status"], "report": str(out_md)}, indent=2))


def build_cuda_readiness(
    artifacts_dir: Path,
    min_gpu_memory_gb: float = 8.0,
    recommended_gpu_memory_gb: float = 12.0,
) -> dict[str, Any]:
    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = load_json_if_exists(manifest_path)
    snapshot_path = relocate_manifest_path(manifest.get("environment_snapshot_path", ""), manifest) if manifest else artifacts_dir / "environment_snapshot.json"
    snapshot = load_json_if_exists(snapshot_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str, severity: str = "error") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "severity": severity})

    check("manifest", bool(manifest), str(manifest_path))
    check("environment_snapshot", bool(snapshot), str(snapshot_path))
    if not manifest or not snapshot:
        return finalize_report(artifacts_dir, manifest, snapshot, checks, min_gpu_memory_gb, recommended_gpu_memory_gb)

    if not bool(manifest.get("needs_model_execution", True)):
        check("model_execution_required", True, "score-only or reuse-only artifact; CUDA readiness is not required", "info")
        return finalize_report(artifacts_dir, manifest, snapshot, checks, min_gpu_memory_gb, recommended_gpu_memory_gb)

    torch_info = snapshot.get("torch", {}) if isinstance(snapshot.get("torch"), dict) else {}
    nvidia_info = snapshot.get("nvidia_smi", {}) if isinstance(snapshot.get("nvidia_smi"), dict) else {}
    cuda_available = bool(torch_info.get("cuda_available"))
    cuda_build = torch_info.get("version_cuda")
    device_count = safe_int(torch_info.get("cuda_device_count"), 0)
    torch_devices = parse_torch_devices(torch_info)
    nvidia_devices = parse_nvidia_devices(nvidia_info)
    all_devices = torch_devices or nvidia_devices
    max_memory_gb = max((float(item.get("memory_total_gb", 0.0) or 0.0) for item in all_devices), default=0.0)

    check("torch_cuda_available", cuda_available, f"torch.cuda.is_available={cuda_available}")
    check("torch_cuda_build", bool(cuda_build), f"torch.version.cuda={cuda_build}")
    check("torch_cuda_device_count", device_count > 0, f"torch cuda devices={device_count}")
    if nvidia_info:
        nvidia_ok = nvidia_info.get("status") == "passed" and bool(nvidia_devices)
        check("nvidia_smi", nvidia_ok, nvidia_smi_detail(nvidia_info, nvidia_devices), "warning" if cuda_available else "error")
    else:
        check("nvidia_smi", False, "nvidia-smi summary missing", "warning" if cuda_available else "error")

    if all_devices:
        check(
            "gpu_memory_minimum",
            max_memory_gb >= float(min_gpu_memory_gb),
            f"max_device_memory={max_memory_gb:.2f} GB, minimum={min_gpu_memory_gb:.2f} GB",
        )
        check(
            "gpu_memory_recommended",
            max_memory_gb >= float(recommended_gpu_memory_gb),
            f"max_device_memory={max_memory_gb:.2f} GB, recommended={recommended_gpu_memory_gb:.2f} GB",
            "warning",
        )
    else:
        check("gpu_memory_minimum", False, "no CUDA or nvidia-smi GPU memory found")

    dtype = str(manifest.get("dtype", ""))
    bf16_supported = bool(torch_info.get("bf16_supported"))
    if dtype == "bfloat16" and cuda_available:
        check(
            "bf16_dtype_support",
            bf16_supported,
            f"manifest dtype=bfloat16, torch.cuda.is_bf16_supported={bf16_supported}",
            "warning",
        )
    else:
        check("bf16_dtype_support", True, f"manifest dtype={dtype}, bf16_supported={bf16_supported}", "info")

    generation_batch_size = safe_int(manifest.get("generation_batch_size"), 0)
    if max_memory_gb and max_memory_gb <= float(min_gpu_memory_gb) and generation_batch_size > 16:
        check(
            "generation_batch_size_for_vram",
            False,
            f"generation_batch_size={generation_batch_size} may be high for {max_memory_gb:.2f} GB VRAM",
            "warning",
        )
    else:
        check("generation_batch_size_for_vram", True, f"generation_batch_size={generation_batch_size}", "info")

    return finalize_report(artifacts_dir, manifest, snapshot, checks, min_gpu_memory_gb, recommended_gpu_memory_gb)


def finalize_report(
    artifacts_dir: Path,
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    checks: list[dict[str, Any]],
    min_gpu_memory_gb: float,
    recommended_gpu_memory_gb: float,
) -> dict[str, Any]:
    errors = [item for item in checks if item["severity"] == "error" and not item["ok"]]
    warnings = [item for item in checks if item["severity"] == "warning" and not item["ok"]]
    if manifest and not bool(manifest.get("needs_model_execution", True)):
        status = "not_applicable"
    elif errors:
        status = "not_ready"
    elif warnings:
        status = "ready_with_warnings"
    else:
        status = "ready"
    return {
        "status": status,
        "artifacts_dir": str(artifacts_dir),
        "run_name": manifest.get("run_name", artifacts_dir.name) if manifest else artifacts_dir.name,
        "thresholds": {
            "min_gpu_memory_gb": float(min_gpu_memory_gb),
            "recommended_gpu_memory_gb": float(recommended_gpu_memory_gb),
        },
        "host_summary": host_summary(snapshot),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "recommendation": readiness_recommendation(status, manifest, snapshot, errors, warnings),
        "commands": readiness_commands(manifest, artifacts_dir),
    }


def host_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    torch_info = snapshot.get("torch", {}) if isinstance(snapshot.get("torch"), dict) else {}
    nvidia_info = snapshot.get("nvidia_smi", {}) if isinstance(snapshot.get("nvidia_smi"), dict) else {}
    python_info = snapshot.get("python", {}) if isinstance(snapshot.get("python"), dict) else {}
    return {
        "python_executable": python_info.get("executable", ""),
        "python_version": python_info.get("version", ""),
        "torch_version": torch_info.get("version", ""),
        "torch_cuda": torch_info.get("version_cuda"),
        "torch_cuda_available": bool(torch_info.get("cuda_available")),
        "torch_cuda_device_count": safe_int(torch_info.get("cuda_device_count"), 0),
        "nvidia_smi_status": nvidia_info.get("status", "missing") if nvidia_info else "missing",
        "nvidia_devices": parse_nvidia_devices(nvidia_info),
        "torch_devices": parse_torch_devices(torch_info),
    }


def readiness_recommendation(
    status: str,
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, str]:
    if status == "not_applicable":
        return {"reason": "CUDA is not required for this artifact mode.", "action": "none"}
    error_names = {item["name"] for item in errors}
    if {"torch_cuda_available", "torch_cuda_build", "torch_cuda_device_count"} & error_names:
        devices = parse_nvidia_devices(snapshot.get("nvidia_smi", {}) if isinstance(snapshot.get("nvidia_smi"), dict) else {})
        if devices:
            device_text = ", ".join(f"{item['name']} ({item['memory_total_gb']:.2f} GB)" for item in devices)
            reason = f"nvidia-smi sees GPU(s) {device_text}, but this Python torch runtime is not CUDA-enabled."
        else:
            reason = "This Python torch runtime cannot see a CUDA device."
        return {
            "reason": reason,
            "action": (
                "Run the formal command on a CUDA host with a CUDA-enabled PyTorch build and at least "
                "8 GB VRAM, preferably 12 GB or more. Re-run `python scripts/run_formal_passmoe.py` on that host before execute."
            ),
        }
    if "gpu_memory_minimum" in error_names:
        return {
            "reason": "The visible GPU memory is below the formal-run minimum.",
            "action": "Use a larger GPU, or run only diagnostic subsets with smaller batch/generation settings.",
        }
    if status == "not_ready":
        return {
            "reason": "Required CUDA readiness inputs or checks failed.",
            "action": "Run `python scripts/run_formal_passmoe.py` to regenerate manifest, environment snapshot, and readiness artifacts.",
        }
    if warnings:
        return {
            "reason": "CUDA is visible but one or more risk checks are warnings.",
            "action": "Use `--dtype auto`; lower `--generation-batch-size` or `--batch-size` if OOM occurs, then resume from checkpoint.",
        }
    return {
        "reason": "CUDA readiness checks passed.",
        "action": f"python scripts/run_formal_passmoe.py --execute --run-name {manifest.get('run_name', DEFAULT_FORMAL_RUN_NAME)}",
    }


def readiness_commands(manifest: dict[str, Any], artifacts_dir: Path) -> dict[str, str]:
    run_name = manifest.get("run_name", artifacts_dir.name) if manifest else artifacts_dir.name
    return {
        "preflight": f"python scripts/run_formal_passmoe.py --run-name {run_name}",
        "execute": f"python scripts/run_formal_passmoe.py --execute --run-name {run_name}",
        "status": f"python scripts/inspect_formal_status.py --artifacts-dir {artifacts_dir}",
        "report": f"python scripts/render_formal_report.py --artifacts-dir {artifacts_dir}",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CUDA Readiness",
        "",
        f"- run: `{report.get('run_name')}`",
        f"- status: `{report.get('status')}`",
        f"- recommendation: {report.get('recommendation', {}).get('reason', '')}",
        f"- action: `{report.get('recommendation', {}).get('action', '')}`",
        "",
        "## Host",
        "",
    ]
    host = report.get("host_summary", {})
    lines.extend(
        [
            f"- python: `{host.get('python_executable', '')}`",
            f"- torch: `{host.get('torch_version', '')}`",
            f"- torch CUDA: `{host.get('torch_cuda')}`",
            f"- torch cuda available: `{host.get('torch_cuda_available')}`",
            f"- torch cuda devices: `{host.get('torch_cuda_device_count')}`",
        ]
    )
    nvidia_devices = host.get("nvidia_devices") or []
    if nvidia_devices:
        lines.extend(["", "## NVIDIA Devices", "", "| Index | Name | Memory GB | Driver |", "|---:|---|---:|---|"])
        for device in nvidia_devices:
            lines.append(
                f"| {device.get('index', '')} | {device.get('name', '')} | "
                f"{float(device.get('memory_total_gb', 0.0) or 0.0):.2f} | {device.get('driver_version', '')} |"
            )
    lines.extend(["", "## Checks", "", "| Check | Severity | Result | Detail |", "|---|---|---|---|"])
    for item in report.get("checks", []):
        result = "PASS" if item.get("ok") else "FAIL"
        lines.append(f"| `{item.get('name')}` | `{item.get('severity')}` | {result} | {item.get('detail', '')} |")
    commands = report.get("commands", {})
    if commands:
        lines.extend(["", "## Commands", "", "```powershell"])
        for command in commands.values():
            lines.append(command)
        lines.extend(["```", ""])
    return "\n".join(lines)


def write_readiness_files(report: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")


def parse_torch_devices(torch_info: dict[str, Any]) -> list[dict[str, Any]]:
    devices = []
    for item in torch_info.get("cuda_devices", []) if isinstance(torch_info.get("cuda_devices"), list) else []:
        memory = safe_int(item.get("total_memory"), 0)
        devices.append(
            {
                "index": item.get("index", ""),
                "name": str(item.get("name", "")),
                "memory_total_gb": memory / (1024**3) if memory else 0.0,
                "driver_version": "",
                "source": "torch",
            }
        )
    return devices


def parse_nvidia_devices(nvidia_info: dict[str, Any]) -> list[dict[str, Any]]:
    devices = []
    for item in nvidia_info.get("devices", []) if isinstance(nvidia_info.get("devices"), list) else []:
        memory_mb = safe_int(item.get("memory_total_mb"), 0)
        devices.append(
            {
                "index": item.get("index", ""),
                "name": str(item.get("name", "")),
                "memory_total_gb": memory_mb / 1024.0 if memory_mb else 0.0,
                "driver_version": str(item.get("driver_version", "")),
                "source": "nvidia-smi",
            }
        )
    return devices


def nvidia_smi_detail(nvidia_info: dict[str, Any], devices: list[dict[str, Any]]) -> str:
    if not devices:
        return f"status={nvidia_info.get('status')}, no devices"
    return "; ".join(
        f"{item.get('index')}: {item.get('name')} {float(item.get('memory_total_gb', 0.0) or 0.0):.2f} GB"
        for item in devices
    )


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


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


if __name__ == "__main__":
    main()
