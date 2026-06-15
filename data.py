from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from config import Config, LEET_DICTIONARY


PASSWORD_KEYS = (
    "password",
    "pass",
    "pwd",
    "output",
    "target",
    "real password",
    "real_password",
)

PII_KEYS = ("pii", "PII", "Knowledge", "knowledge", "profile", "user", "account_info")


@dataclass
class PasswordRecord:
    password: str
    pii: dict[str, Any] | None = None
    source: str = ""


class CharPasswordTokenizer:
    """Small deterministic tokenizer used for CPU smoke tests."""

    pad_token = "<PAD>"
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    unk_token = "<UNK>"
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self, alphabet: str):
        chars = []
        seen = set()
        for char in alphabet:
            if char not in seen:
                chars.append(char)
                seen.add(char)
        self.id_to_token = [self.pad_token, self.bos_token, self.eos_token, self.unk_token] + chars
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode_password(self, password: str, max_length: int) -> dict[str, torch.Tensor]:
        ids = [self.bos_token_id]
        ids.extend(self.token_to_id.get(ch, self.unk_token_id) for ch in password)
        ids.append(self.eos_token_id)
        if len(ids) > max_length:
            ids = ids[:max_length]
            ids[-1] = self.eos_token_id
        attention = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
            attention.append(0)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chars: list[str] = []
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                break
            if skip_special_tokens and token_id in special:
                continue
            if 0 <= token_id < len(self.id_to_token):
                token = self.id_to_token[token_id]
                if token == self.unk_token:
                    continue
                chars.append(token)
        return "".join(chars)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"alphabet": "".join(self.id_to_token[4:])}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharPasswordTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["alphabet"])


class FeatureExtractor:
    def __init__(self, leet_dictionary: dict[str, list[str]] | None = None):
        self.leet_dictionary = leet_dictionary or LEET_DICTIONARY
        self.leet_chars = {ch for values in self.leet_dictionary.values() for ch in values}
        self.common_name_fragments = {"john", "mary", "zhang", "wang", "li", "admin", "user"}

    def extract(self, password: str, pii: dict[str, Any] | None = None) -> list[float]:
        return [
            self.pii_score(password, pii),
            self.leet_score(password),
            self.entropy_score(password),
        ]

    def pii_score(self, password: str, pii: dict[str, Any] | None = None) -> float:
        lowered = password.lower()
        score = 0.0

        if re.search(r"(19|20)\d{2}", password):
            score += 0.25
        if re.search(r"\d{6,8}", password):
            score += 0.15
        if any(fragment in lowered for fragment in self.common_name_fragments):
            score += 0.15

        if pii:
            tokens = self._pii_tokens(pii)
            if tokens:
                matched = 0
                for token in tokens:
                    token_l = token.lower()
                    if len(token_l) >= 2 and token_l in lowered:
                        matched += 1
                    elif token_l.isdigit() and len(token_l) >= 4 and token_l[-4:] in password:
                        matched += 1
                score += min(0.6, matched / max(len(tokens), 1))

        return min(score, 1.0)

    def leet_score(self, password: str) -> float:
        if not password:
            return 0.0
        if not any(ch.isalpha() for ch in password):
            return 0.0
        hits = sum(1 for ch in password if ch in self.leet_chars and not ch.isalpha())
        return min(hits / len(password), 1.0)

    def entropy_score(self, password: str) -> float:
        if not password:
            return 0.0
        probs = [password.count(ch) / len(password) for ch in set(password)]
        entropy = -sum(p * math.log2(p) for p in probs)
        return min(entropy / 6.0, 1.0)

    def _pii_tokens(self, pii: dict[str, Any]) -> list[str]:
        tokens: list[str] = []
        for value in pii.values():
            if isinstance(value, (list, tuple, set)):
                raw_values = value
            else:
                raw_values = [value]
            for raw in raw_values:
                text = str(raw)
                if "@" in text:
                    text = text.split("@", 1)[0]
                tokens.extend(re.findall(r"[A-Za-z0-9]{2,}", text))
        return tokens


class PasswordDataset(Dataset):
    def __init__(self, records: Sequence[PasswordRecord], tokenizer: Any, config: Config):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.config = config
        self.extractor = FeatureExtractor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        if self.config.task == "targeted" and record.pii:
            encoded, labels = encode_targeted_record(
                self.tokenizer,
                record,
                self.config.max_length,
                self.config.prompt_template_id,
            )
        else:
            encoded = encode_password(self.tokenizer, record.password, self.config.max_length)
            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100
        features = torch.tensor(self.extractor.extract(record.password, record.pii), dtype=torch.float32)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "features": features,
        }


def encode_password(tokenizer: Any, password: str, max_length: int) -> dict[str, torch.Tensor]:
    return encode_text(tokenizer, password, max_length=max_length, add_bos=True, add_eos=True)


def encode_text(
    tokenizer: Any,
    text: str,
    max_length: int,
    add_bos: bool,
    add_eos: bool,
) -> dict[str, torch.Tensor]:
    if hasattr(tokenizer, "encode_password"):
        ids = []
        if add_bos:
            ids.append(tokenizer.bos_token_id)
        ids.extend(tokenizer.token_to_id.get(ch, tokenizer.unk_token_id) for ch in text)
        if add_eos:
            ids.append(tokenizer.eos_token_id)
        if len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos:
                ids[-1] = tokenizer.eos_token_id
        attention = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(tokenizer.pad_token_id)
            attention.append(0)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    eos = getattr(tokenizer, "eos_token", None) or ""
    if add_eos:
        text = text + eos
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=add_bos,
    )
    return {
        "input_ids": encoded["input_ids"].squeeze(0).long(),
        "attention_mask": encoded["attention_mask"].squeeze(0).long(),
    }


def encode_targeted_record(
    tokenizer: Any,
    record: PasswordRecord,
    max_length: int,
    prompt_template_id: str = "passmoe",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prompt = format_targeted_prompt(record.pii or {}, prompt_template_id)
    full_text = prompt + record.password
    encoded = encode_text(tokenizer, full_text, max_length=max_length, add_bos=True, add_eos=True)
    prompt_encoded = encode_text(tokenizer, prompt, max_length=max_length, add_bos=True, add_eos=False)
    labels = encoded["input_ids"].clone()
    prompt_len = int(prompt_encoded["attention_mask"].sum().item())
    labels[: min(prompt_len, labels.numel())] = -100
    labels[encoded["attention_mask"] == 0] = -100
    return encoded, labels


def format_targeted_prompt(pii: dict[str, Any], prompt_template_id: str = "passmoe") -> str:
    template = str(prompt_template_id).lower()
    if template in {"0", "passllm", "passllm0", "p00"}:
        content = json.dumps(pii, ensure_ascii=True)
        return (
            "As a targeted password guessing model, your task is to utilize "
            "the provided account information to guess the password."
            f"{content}"
        )

    compact = json.dumps(pii, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return (
        "As a targeted password guessing model, use the account information "
        "to guess the password."
        f"{compact}"
        "Password:"
    )


def decode_ids(tokenizer: Any, ids: Sequence[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids, skip_special_tokens=True)
    raise TypeError("Tokenizer does not support decode().")


def load_records(path: str | Path, max_samples: int | None = None) -> list[PasswordRecord]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        records = _load_csv(source, max_samples=max_samples)
    elif suffix == ".jsonl":
        records = _load_jsonl(source, max_samples=max_samples)
    elif suffix == ".json":
        records = _load_json(source)
    else:
        records = _load_txt(source, max_samples=max_samples)

    records = [record for record in records if _valid_password(record.password)]
    if max_samples is not None:
        records = records[:max_samples]
    return records


def split_records(
    records: Sequence[PasswordRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[PasswordRecord], list[PasswordRecord]]:
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    if len(records) < 2:
        return records, records
    val_size = max(1, int(len(records) * val_fraction))
    return records[val_size:], records[:val_size]


def create_data_loaders(
    train_records: Sequence[PasswordRecord],
    val_records: Sequence[PasswordRecord],
    tokenizer: Any,
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    train_ds = PasswordDataset(train_records, tokenizer, config)
    val_ds = PasswordDataset(val_records, tokenizer, config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def write_smoke_dataset(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "password",
        "password1",
        "admin123",
        "qwerty",
        "letmein",
        "zhang1998",
        "wang2020",
        "P@ssw0rd",
        "summer2024",
        "dragon",
        "abc123",
        "hello123",
        "iloveyou",
        "michael1",
        "football",
        "monkey",
        "sunshine",
        "princess",
        "welcome1",
        "test1234",
    ]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["password"])
        for password in rows:
            writer.writerow([password])
    return target


def _load_csv(path: Path, max_samples: int | None = None) -> list[PasswordRecord]:
    records: list[PasswordRecord] = []
    with path.open("r", newline="", encoding="utf-8-sig", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {name.strip().lower() for name in (reader.fieldnames or [])}
        if fieldnames.intersection(PASSWORD_KEYS):
            for row in reader:
                password = _extract_password(row)
                if password is not None:
                    records.append(PasswordRecord(password=password, pii=_extract_pii(row), source=str(path)))
                if max_samples is not None and len(records) >= max_samples:
                    break
        else:
            handle.seek(0)
            reader = csv.reader(handle)
            for row in reader:
                if row:
                    records.append(PasswordRecord(password=row[0].strip(), source=str(path)))
                if max_samples is not None and len(records) >= max_samples:
                    break
    return records


def _load_jsonl(path: Path, max_samples: int | None = None) -> list[PasswordRecord]:
    records: list[PasswordRecord] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            password = _extract_password(obj)
            if password is not None:
                records.append(PasswordRecord(password=password, pii=_extract_pii(obj), source=str(path)))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def _load_json(path: Path) -> list[PasswordRecord]:
    payload = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    items = payload if isinstance(payload, list) else payload.get("data", [])
    records: list[PasswordRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        password = _extract_password(item)
        if password is not None:
            records.append(PasswordRecord(password=password, pii=_extract_pii(item), source=str(path)))
    return records


def _load_txt(path: Path, max_samples: int | None = None) -> list[PasswordRecord]:
    records: list[PasswordRecord] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            password = _parse_txt_password(line)
            if password:
                records.append(PasswordRecord(password=password, source=str(path)))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def _parse_txt_password(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    parts = stripped.split()
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]
    if len(parts) >= 2 and parts[-1].isdigit() and "\t" in stripped:
        return "\t".join(stripped.split("\t")[:-1]).strip()
    return stripped


def _extract_password(obj: dict[str, Any]) -> str | None:
    for key in PASSWORD_KEYS:
        if key in obj and obj[key] is not None:
            return str(obj[key]).strip()
    return None


def _extract_pii(obj: dict[str, Any]) -> dict[str, Any] | None:
    for key in PII_KEYS:
        value = obj.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    pii = {k: v for k, v in obj.items() if k not in PASSWORD_KEYS and v not in (None, "")}
    return pii or None


def _valid_password(password: str) -> bool:
    if not password:
        return False
    if any(ch.isspace() for ch in password):
        return False
    return 1 <= len(password) <= 128
