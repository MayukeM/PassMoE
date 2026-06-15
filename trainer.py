from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import PassMoE, count_parameters


LOG_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "lr",
    "train_valid_tokens",
    "val_valid_tokens",
    "train_batches",
    "val_batches",
    "train_zero_token_batches",
    "val_zero_token_batches",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        model: PassMoE,
        tokenizer: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)
        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.best_val_loss = float("inf")
        self.start_epoch = 1
        self.history: list[dict[str, Any]] = []
        self.resumed_from = ""
        self.log_path = self.run_dir / "train_log.csv"
        self.metrics_path = self.run_dir / "metrics.json"
        self._write_metadata()

    def train(self) -> dict[str, Any]:
        if self.config.resume_checkpoint:
            self.resume_from_checkpoint(self.config.resume_checkpoint)

        history = list(self.history)
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_stats = self.train_epoch(epoch)
            val_stats = self.evaluate_loss(self.val_loader)
            if self.config.task == "targeted":
                ensure_supervised_tokens(train_stats, "training")
                ensure_supervised_tokens(val_stats, "validation")
            train_loss = float(train_stats["loss"])
            val_loss = float(val_stats["loss"])
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_valid_tokens": train_stats["valid_tokens"],
                "val_valid_tokens": val_stats["valid_tokens"],
                "train_batches": train_stats["batches"],
                "val_batches": val_stats["batches"],
                "train_zero_token_batches": train_stats["zero_token_batches"],
                "val_zero_token_batches": val_stats["zero_token_batches"],
            }
            history.append(row)
            self.history = history
            self._append_log(row)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt", epoch, val_loss)
            self.save_checkpoint("last.pt", epoch, val_loss)

        metrics = {
            "status": "completed",
            "best_val_loss": self.best_val_loss,
            "start_epoch": self.start_epoch,
            "target_epochs": self.config.epochs,
            "resumed_from": self.resumed_from,
            "history": history,
            "parameters": count_parameters(self.model),
            "run_dir": str(self.run_dir.resolve()),
        }
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def train_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        total_batches = 0
        zero_token_batches = 0
        progress = tqdm(self.train_loader, desc=f"epoch {epoch}/{self.config.epochs}", leave=False)
        for batch in progress:
            total_batches += 1
            batch = move_batch(batch, self.device)
            outputs = self.model(**batch)
            loss = outputs["loss"]
            weight = batch_loss_weight(outputs, batch)
            if weight == 0:
                zero_token_batches += 1
                progress.set_postfix(loss=float(loss.detach().cpu()), valid_tokens=0)
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            total_loss += float(loss.detach().cpu()) * weight
            total_items += weight
            progress.set_postfix(loss=float(loss.detach().cpu()), valid_tokens=weight)
        return {
            "loss": total_loss / max(total_items, 1),
            "valid_tokens": total_items,
            "batches": total_batches,
            "zero_token_batches": zero_token_batches,
        }

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        total_batches = 0
        zero_token_batches = 0
        for batch in loader:
            total_batches += 1
            batch = move_batch(batch, self.device)
            outputs = self.model(**batch)
            loss = outputs["loss"]
            weight = batch_loss_weight(outputs, batch)
            if weight == 0:
                zero_token_batches += 1
                continue
            total_loss += float(loss.detach().cpu()) * weight
            total_items += weight
        return {
            "loss": total_loss / max(total_items, 1),
            "valid_tokens": total_items,
            "batches": total_batches,
            "zero_token_batches": zero_token_batches,
        }

    def save_checkpoint(self, name: str, epoch: int, val_loss: float) -> Path:
        path = self.run_dir / name
        payload = {
            "checkpoint_format": "passmoe_trainable_state_v2",
            "model_state": trainable_state_dict(self.model),
            "optimizer_state": optimizer_state_to_cpu(self.optimizer.state_dict()),
            "config": self.config.to_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "parameters": count_parameters(self.model),
            "trainable_keys": trainable_parameter_names(self.model),
            "merge_report": getattr(self.model, "merge_report", {}),
        }
        torch.save(payload, path)
        if hasattr(self.tokenizer, "save"):
            self.tokenizer.save(self.run_dir / "tokenizer.json")
        return path

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        payload = load_checkpoint(self.model, checkpoint_path, self.device)
        self.resumed_from = str(checkpoint_path.resolve())
        self.start_epoch = int(payload.get("epoch", 0)) + 1
        self.best_val_loss = float(payload.get("best_val_loss", payload.get("val_loss", self.best_val_loss)))
        loaded_history = payload.get("history")
        if isinstance(loaded_history, list):
            self.history = [row for row in loaded_history if isinstance(row, dict)]
        else:
            self.history = self._read_log_history()

        optimizer_state = payload.get("optimizer_state")
        if self.config.resume_optimizer and optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except ValueError as exc:
                print(f"Warning: optimizer state was not resumed from {checkpoint_path}: {exc}")
        return payload

    def _write_metadata(self) -> None:
        (self.run_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2),
            encoding="utf-8",
        )
        (self.run_dir / "parameters.json").write_text(
            json.dumps(count_parameters(self.model), indent=2),
            encoding="utf-8",
        )
        if not self.config.resume_checkpoint or not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                writer.writeheader()

    def _append_log(self, row: dict[str, Any]) -> None:
        self._ensure_log_fields()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            writer.writerow(row)

    def _ensure_log_fields(self) -> None:
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                writer.writeheader()
            return
        with self.log_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if fieldnames == LOG_FIELDS:
                return
            rows = list(reader)
        with self.log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            writer.writeheader()
            for item in rows:
                writer.writerow({field: item.get(field, "") for field in LOG_FIELDS})

    def _read_log_history(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.log_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    rows.append(
                        {
                            "epoch": int(row["epoch"]),
                            "train_loss": float(row["train_loss"]),
                            "val_loss": float(row["val_loss"]),
                            "lr": float(row["lr"]),
                            "train_valid_tokens": safe_int(row.get("train_valid_tokens")),
                            "val_valid_tokens": safe_int(row.get("val_valid_tokens")),
                            "train_batches": safe_int(row.get("train_batches")),
                            "val_batches": safe_int(row.get("val_batches")),
                            "train_zero_token_batches": safe_int(row.get("train_zero_token_batches")),
                            "val_zero_token_batches": safe_int(row.get("val_zero_token_batches")),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return rows


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_loss_weight(outputs: dict[str, Any], batch: dict[str, torch.Tensor]) -> int:
    valid_tokens = outputs.get("valid_tokens")
    if torch.is_tensor(valid_tokens):
        return max(int(valid_tokens.detach().cpu().item()), 0)
    return int(batch["input_ids"].size(0))


def ensure_supervised_tokens(stats: dict[str, Any], stage: str) -> None:
    if int(stats.get("valid_tokens", 0) or 0) > 0:
        return
    raise ValueError(
        f"Targeted {stage} produced zero supervised password tokens across "
        f"{stats.get('batches', 0)} batches. Increase --max-length or run "
        "inspect-targeted-lengths before trusting this run."
    )


def safe_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def trainable_state_dict(model: PassMoE) -> dict[str, torch.Tensor]:
    named_params = dict(named_parameters_compat(model))
    state = {}
    for name, tensor in model.state_dict().items():
        param = named_params.get(name)
        if param is not None and param.requires_grad:
            state[name] = tensor.detach().cpu().clone()
    return state


def trainable_parameter_names(model: PassMoE) -> list[str]:
    return [name for name, param in named_parameters_compat(model) if param.requires_grad]


def named_parameters_compat(model: PassMoE) -> list[tuple[str, torch.nn.Parameter]]:
    try:
        return list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return list(model.named_parameters())


def optimizer_state_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return value

    return convert(state)


def load_checkpoint(model: PassMoE, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model_state", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable_names = set(trainable_parameter_names(model))
    missing_trainable = sorted(name for name in missing if name in trainable_names)
    if missing_trainable:
        preview = ", ".join(missing_trainable[:10])
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is missing {len(missing_trainable)} trainable parameter(s): {preview}"
        )
    payload["missing_keys"] = missing
    payload["missing_trainable_keys"] = missing_trainable
    payload["unexpected_keys"] = unexpected
    payload["loaded_trainable_keys"] = sorted(name for name in state if name in trainable_names)
    return payload
