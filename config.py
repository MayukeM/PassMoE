from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch


ROOT = Path(__file__).resolve().parent


LEET_DICTIONARY: Dict[str, List[str]] = {
    "a": ["@", "4"],
    "b": ["8"],
    "e": ["3"],
    "g": ["6", "9"],
    "i": ["1", "!"],
    "l": ["1", "|"],
    "o": ["0"],
    "s": ["5", "$"],
    "t": ["7", "+"],
    "z": ["2"],
}


@dataclass
class Config:
    """Runtime configuration for the revived PassMoE implementation."""

    # Model. Use "tiny" for a CPU-only smoke model. Use a local HuggingFace
    # path or model id for real PassLLM-style experiments.
    task: str = "trawling"
    base_model: str = "tiny"
    base_adapter: str = ""
    prompt_template_id: str = "passmoe"
    max_length: int = 32
    hidden_dim: int = 256
    lora_rank: int = 16
    router_hidden_dim: int = 64
    top_k_experts: int = 2
    dropout: float = 0.1

    # Tiny fallback model.
    tiny_layers: int = 2
    tiny_heads: int = 4
    tiny_vocab: str = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " !@#$%^&*()-_=+[]{};:'\",.<>/?\\|`~"
    )

    # Training.
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0
    val_fraction: float = 0.1
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    # Generation and evaluation.
    generation_max_new_tokens: int = 32
    generation_batch_size: int = 32
    temperature: float = 1.0
    beam_width: int = 64
    num_passwords: int = 1000
    min_password_length: int = 1
    budgets: str = "1,10,100,1000"
    target_eval_samples: int = 20
    target_candidates_per_user: int = 100
    skip_generation: bool = False
    resume_generation: bool = False

    # Paths.
    data_path: str = "data/smoke_passwords.csv"
    test_data_path: str = ""
    output_dir: str = "runs"
    run_name: str = "passmoe_smoke"
    checkpoint: str = ""
    resume_checkpoint: str = ""
    resume_optimizer: bool = True

    # Device.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "auto"
    use_device_map: bool = False

    # Local assets discovered in D:\paper.
    local_qwen_05b: str = (
        r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code\model\Qwen2.5-0.5B-Instruct"
    )
    local_passllm_code: str = r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code"
    local_fielddrop_adapter: str = (
        r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code\checkpoints\fielddrop_500k_p04"
    )
    local_baseline10k_adapter: str = (
        r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code\checkpoints\baseline_clixsense_10k\final"
    )
    local_csdn_adapter: str = (
        r"D:\paper\1-ACCEPT\PassLLM-FieldDrop\code\checkpoints\126_csdn_disQwen0.5B"
    )

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def budgets_list(self) -> List[int]:
        return sorted({int(x) for x in self.budgets.split(",") if x.strip()})

    def to_dict(self) -> dict:
        return asdict(self)


def dtype_from_string(name: str) -> torch.dtype:
    if str(name).lower() == "auto":
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(name, torch.float32)
