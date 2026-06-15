from __future__ import annotations

import argparse
import importlib.util
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_RUN_NAME = "qwen_fielddrop_base_identity_clixsense_500_raw"
TEXT_EXTENSIONS = {".md", ".py", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".ps1"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight PassMoE reproducibility checks.")
    parser.add_argument("--json", dest="json_out", action="store_true", help="Print the full report as JSON.")
    parser.add_argument("--skip-cli-help", action="store_true", help="Skip CLI help subprocess checks.")
    parser.add_argument(
        "--allow-local-absolute-paths",
        action="store_true",
        help="Do not fail on hard-coded local absolute paths in tracked text files.",
    )
    args = parser.parse_args()

    report = run_checks(args)
    if args.json_out:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    if report["status"] != "passed":
        raise SystemExit(1)


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = "", severity: str = "error") -> None:
        checks.append({"name": name, "ok": bool(ok), "severity": severity, "detail": detail})

    tracked = tracked_files()
    python_files = [path for path in tracked if path.suffix == ".py"]
    compile_errors = compile_python_files(python_files)
    check("python_compile", not compile_errors, compile_errors)

    if not args.skip_cli_help:
        help_errors = run_help_commands()
        check("cli_help", not help_errors, help_errors)

    absolute_hits = audit_absolute_paths(tracked)
    check(
        "no_tracked_local_absolute_paths",
        args.allow_local_absolute_paths or not absolute_hits,
        absolute_hits,
        severity="warning" if args.allow_local_absolute_paths else "error",
    )

    default_errors = audit_default_formal_run()
    check("formal_defaults_current", not default_errors, default_errors)

    boundary_errors = audit_method_boundary()
    check("fielddrop_boundary_documented", not boundary_errors, boundary_errors)

    artifact_notes = audit_current_artifact_if_present()
    check("current_artifact_manifest", artifact_notes["status"] != "failed", artifact_notes, severity=artifact_notes["severity"])

    errors = [item for item in checks if item["severity"] == "error" and not item["ok"]]
    warnings = [item for item in checks if item["severity"] == "warning" and not item["ok"]]
    return {
        "status": "failed" if errors else "passed",
        "repo_root": str(REPO_ROOT),
        "default_formal_run": DEFAULT_FORMAL_RUN_NAME,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        return [(REPO_ROOT / line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return [
        path
        for path in REPO_ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    ]


def compile_python_files(paths: list[Path]) -> list[dict[str, str]]:
    errors = []
    for path in sorted(paths):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append({"path": repo_relative(path), "error": str(exc)})
    return errors


def run_help_commands() -> list[dict[str, str]]:
    commands = [
        [sys.executable, "main.py", "--help"],
        [sys.executable, "scripts/run_formal_passmoe.py", "--help"],
        [sys.executable, "scripts/validate_formal_outputs.py", "--help"],
        [sys.executable, "scripts/inspect_formal_status.py", "--help"],
        [sys.executable, "scripts/render_formal_report.py", "--help"],
        [sys.executable, "scripts/check_cuda_readiness.py", "--help"],
        [sys.executable, "scripts/analyze_expert_specialization.py", "--help"],
    ]
    errors = []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            errors.append(
                {
                    "command": " ".join(command),
                    "returncode": str(result.returncode),
                    "stderr": result.stderr[-1000:],
                }
            )
    return errors


def audit_absolute_paths(paths: list[Path]) -> list[dict[str, str]]:
    current_root_text = str(REPO_ROOT)
    patterns = [
        re.compile(re.escape(current_root_text), re.IGNORECASE),
        re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\"),
        re.compile(r"(?<![`$}\w])/(workspace|home|root|mnt|data)/"),
    ]
    hits: list[dict[str, str]] = []
    for path in sorted(paths):
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if (
                line.strip().startswith("#!")
                or "DEFAULT_ARTIFACTS_DIR = REPO_ROOT" in line
                or "re.compile(" in line
            ):
                continue
            if any(pattern.search(line) for pattern in patterns):
                hits.append({"path": repo_relative(path), "line": str(line_number), "text": line.strip()[:240]})
    return hits


def audit_default_formal_run() -> list[dict[str, str]]:
    errors = []
    modules = {
        "run_formal_passmoe": REPO_ROOT / "scripts" / "run_formal_passmoe.py",
        "validate_formal_outputs": REPO_ROOT / "scripts" / "validate_formal_outputs.py",
        "inspect_formal_status": REPO_ROOT / "scripts" / "inspect_formal_status.py",
        "render_formal_report": REPO_ROOT / "scripts" / "render_formal_report.py",
        "check_cuda_readiness": REPO_ROOT / "scripts" / "check_cuda_readiness.py",
    }
    for name, path in modules.items():
        module = import_module_from_path(name, path)
        run_name = getattr(module, "DEFAULT_FORMAL_RUN_NAME", "")
        if run_name != DEFAULT_FORMAL_RUN_NAME:
            errors.append({"module": name, "field": "DEFAULT_FORMAL_RUN_NAME", "observed": str(run_name)})
        artifacts_dir = Path(str(getattr(module, "DEFAULT_ARTIFACTS_DIR", ""))) if hasattr(module, "DEFAULT_ARTIFACTS_DIR") else None
        if artifacts_dir is not None and artifacts_dir.name != DEFAULT_FORMAL_RUN_NAME:
            errors.append({"module": name, "field": "DEFAULT_ARTIFACTS_DIR", "observed": str(artifacts_dir)})
    return errors


def audit_method_boundary() -> list[dict[str, str]]:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    required_phrases = [
        "FieldDrop belongs to the separate PassLLM improvement line",
        "not part of the PassMoE method claim",
        "Do not describe this as evidence that supervised low-rank PassMoE residual training improved FieldDrop",
        "This is not an SR@K result",
    ]
    return [{"missing_phrase": phrase} for phrase in required_phrases if phrase not in normalized]


def audit_current_artifact_if_present() -> dict[str, Any]:
    artifacts_dir = REPO_ROOT / "artifacts" / "formal" / DEFAULT_FORMAL_RUN_NAME
    manifest_path = artifacts_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {
            "status": "skipped",
            "severity": "warning",
            "reason": "current formal artifact is not present in this checkout",
            "path": str(manifest_path),
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    expected = {
        "run_name": DEFAULT_FORMAL_RUN_NAME,
        "baseline_variant": "fd500k_p00_unique",
        "base_adapter": "fielddrop",
        "post_fusion": False,
        "seed": 42,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            errors.append({"key": key, "observed": manifest.get(key), "expected": expected_value})
    observed_epochs = command_option(manifest.get("execution_command") or [], "--epochs", "")
    if observed_epochs != "0":
        errors.append({"key": "execution_command.--epochs", "observed": observed_epochs, "expected": "0"})
    return {
        "status": "failed" if errors else "passed",
        "severity": "error" if errors else "info",
        "path": str(manifest_path),
        "errors": errors,
    }


def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"passmoe_repro_check_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_option(command: list[Any], option: str, default: str) -> str:
    command_text = [str(item) for item in command]
    try:
        index = command_text.index(option)
    except ValueError:
        return str(default)
    next_index = index + 1
    if next_index >= len(command_text):
        return str(default)
    return command_text[next_index]


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"status: {report['status']}",
        f"repo_root: {report['repo_root']}",
        f"default_formal_run: {report['default_formal_run']}",
        "",
        "checks:",
    ]
    for item in report["checks"]:
        result = "PASS" if item["ok"] else "WARN" if item["severity"] == "warning" else "FAIL"
        lines.append(f"- {result} {item['name']}")
        if not item["ok"] and item.get("detail"):
            lines.append(f"  {json.dumps(item['detail'], ensure_ascii=False)[:1000]}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
