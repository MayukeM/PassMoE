from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, LEET_DICTIONARY, dtype_from_string
from data import CharPasswordTokenizer


class TinyCausalBackbone(nn.Module):
    """CPU-friendly causal LM backbone used to verify the PassMoE pipeline."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position = nn.Embedding(512, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.embedding(input_ids) + self.position(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        padding_mask = attention_mask == 0 if attention_mask is not None else None
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.ln(hidden)


class LowRankExpert(nn.Module):
    """Low-rank expert adapter over shared hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        dropout: float,
        expert_id: int = 0,
    ):
        super().__init__()
        self.expert_id = expert_id
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.feature_proj = nn.Linear(3, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = 1.0 / math.sqrt(max(rank, 1))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        expert_dtype = self.down.weight.dtype
        hidden = hidden.to(dtype=expert_dtype)
        features = features.to(dtype=expert_dtype)
        feature_bias = self.feature_proj(features).unsqueeze(1)
        expert_hidden = hidden + 0.1 * feature_bias

        # Paper-inspired specialization. The signal is deliberately small so the
        # shared language model remains the main generator.
        if self.expert_id == 0:  # PII expert
            scale = 1.0 + features[:, 0].view(-1, 1, 1)
        elif self.expert_id == 1:  # high-entropy expert
            scale = 1.0 + features[:, 2].view(-1, 1, 1)
        else:  # leetspeak / morphology expert
            scale = 1.0 + features[:, 1].view(-1, 1, 1)
        expert_hidden = expert_hidden * scale

        residual = self.up(F.gelu(self.down(self.dropout(expert_hidden)))) * self.scaling
        # Keep the fused foundation model exact at initialization. A fresh
        # LayerNorm over frozen LM hidden states changes the output distribution
        # even when the low-rank residual is still zero.
        return hidden + residual


class HybridGatingNetwork(nn.Module):
    """CNN-GRU router over [PII score, leet score, entropy]."""

    def __init__(self, hidden_dim: int = 64, num_experts: int = 3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_experts)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.unsqueeze(1)
        x = self.cnn(x).transpose(1, 2)
        _, hidden = self.gru(x)
        return F.softmax(self.fc(hidden[-1]), dim=-1)


class PassMoE(nn.Module):
    """Revived PassMoE-P with one shared backbone and three routed experts."""

    def __init__(
        self,
        config: Config,
        vocab_size: int,
        hidden_dim: int,
        pad_token_id: int,
        eos_token_id: int,
        backbone: nn.Module,
        shared_lm_head: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.backbone = backbone
        self.router = HybridGatingNetwork(config.router_hidden_dim, num_experts=3)
        self.lm_head = shared_lm_head or nn.Linear(hidden_dim, vocab_size, bias=False)
        self.experts = nn.ModuleList(
            [
                LowRankExpert(hidden_dim, config.lora_rank, config.dropout, 0),
                LowRankExpert(hidden_dim, config.lora_rank, config.dropout, 1),
                LowRankExpert(hidden_dim, config.lora_rank, config.dropout, 2),
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features is None:
            features = torch.zeros(input_ids.size(0), 3, dtype=torch.float32, device=input_ids.device)

        hidden = self._forward_hidden(input_ids, attention_mask)
        expert_hidden = torch.stack([expert(hidden, features) for expert in self.experts], dim=1)
        expert_weights = self.router(features)
        routed_weights = self._topk_weights(expert_weights)
        routed_weights = routed_weights.to(dtype=expert_hidden.dtype)
        mixed_hidden = torch.einsum("be,beth->bth", routed_weights, expert_hidden)
        lm_dtype = module_weight_dtype(self.lm_head, mixed_hidden.dtype)
        logits = self.lm_head(mixed_hidden.to(dtype=lm_dtype))

        output = {"logits": logits, "expert_weights": expert_weights, "routed_weights": routed_weights}
        specialization_weight = float(getattr(self.config, "router_specialization_weight", 0.0) or 0.0)
        if specialization_weight > 0.0:
            target_index, target_probs = router_specialization_target_distribution(
                features=features,
                min_signal=float(getattr(self.config, "router_specialization_min_signal", 0.05) or 0.0),
                smoothing=float(getattr(self.config, "router_specialization_smoothing", 0.05) or 0.0),
            )
            output["router_target_index"] = target_index
            output["router_target_probs"] = target_probs
            log_weights = expert_weights.clamp_min(1e-8).log()
            target_probs = target_probs.to(dtype=expert_weights.dtype)
            per_sample_loss = -(target_probs * log_weights).sum(dim=-1)
            class_counts = torch.bincount(target_index.detach(), minlength=3).to(
                device=per_sample_loss.device,
                dtype=per_sample_loss.dtype,
            )
            class_weights = class_counts.sum().clamp_min(1.0) / class_counts.clamp_min(1.0)
            sample_weights = class_weights[target_index].to(dtype=per_sample_loss.dtype)
            sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
            output["router_specialization_loss"] = (per_sample_loss * sample_weights).mean()
            output["router_specialization_agreement"] = (
                expert_weights.argmax(dim=-1).eq(target_index).float().mean()
            )
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid_tokens = shift_labels.ne(-100).sum()
            if int(valid_tokens.detach().cpu()) == 0:
                loss = shift_logits.sum() * 0.0
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            output["lm_loss"] = loss
            if "router_specialization_loss" in output:
                loss = loss + specialization_weight * output["router_specialization_loss"]
            output["loss"] = loss
            output["valid_tokens"] = valid_tokens
        return output

    def _forward_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if hasattr(self.backbone, "forward_hidden"):
            return self.backbone.forward_hidden(input_ids, attention_mask)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def _topk_weights(self, weights: torch.Tensor) -> torch.Tensor:
        k = min(self.config.top_k_experts, weights.size(-1))
        values, indices = torch.topk(weights, k=k, dim=-1)
        sparse = torch.zeros_like(weights)
        sparse.scatter_(1, indices, values)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def generate_passwords(
        self,
        tokenizer: Any,
        prefix: str = "",
        num_passwords: int | None = None,
        max_length: int | None = None,
        beam_width: int | None = None,
        temperature: float | None = None,
        device: str | torch.device | None = None,
        strip_prefix: bool = False,
        pii: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        self.eval()
        num_passwords = num_passwords or self.config.num_passwords
        max_length = max_length or self.config.generation_max_new_tokens
        beam_width = beam_width or self.config.beam_width
        temperature = temperature or self.config.temperature
        generation_batch_size = max(1, int(getattr(self.config, "generation_batch_size", beam_width) or beam_width))
        device = torch.device(device or next(self.parameters()).device)

        start_ids = encode_prefix(tokenizer, prefix, device)
        prompt_token_length = int(start_ids.size(1))
        beams = [(start_ids, 0.0, False)]
        completed: dict[str, tuple[list[int], float]] = {}

        for _ in range(max_length):
            candidates: list[tuple[torch.Tensor, float, bool]] = []
            active = [beam for beam in beams if not beam[2]]
            if not active:
                break

            for offset in range(0, len(active), generation_batch_size):
                chunk = active[offset : offset + generation_batch_size]
                input_ids = torch.cat([ids for ids, _score, _finished in chunk], dim=0)
                attention = torch.ones_like(input_ids, device=device)
                features = torch.tensor(
                    [
                        feature_vector_for_text(
                            decode_candidate_text(tokenizer, ids[0].tolist(), prompt_token_length, prefix, strip_prefix),
                            pii,
                        )
                        for ids, _score, _finished in chunk
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                outputs = self(input_ids=input_ids, attention_mask=attention, features=features)
                logits = outputs["logits"][:, -1, :] / max(temperature, 1e-6)
                logits = suppress_invalid_tokens(tokenizer, logits)
                probs = F.softmax(logits, dim=-1)
                top_probs, top_ids = torch.topk(probs, k=min(beam_width, probs.size(-1)), dim=-1)

                for row, (ids, score, _finished) in enumerate(chunk):
                    for prob, token_id in zip(top_probs[row], top_ids[row]):
                        token = int(token_id.item())
                        new_score = score + math.log(max(float(prob.item()), 1e-12))
                        new_ids = torch.cat([ids, token_id.view(1, 1)], dim=1)
                        if token == self.eos_token_id:
                            decoded = decode_candidate_text(tokenizer, new_ids[0].tolist(), prompt_token_length, prefix, strip_prefix)
                            if len(decoded) >= self.config.min_password_length:
                                current = completed.get(decoded)
                                if current is None or new_score > current[1]:
                                    completed[decoded] = (new_ids[0].tolist(), new_score)
                        else:
                            candidates.append((new_ids, new_score, False))

            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = candidates[:beam_width]
            if len(completed) >= num_passwords and beams:
                best_active = beams[0][1]
                completed_ranked = sorted(completed.values(), key=lambda item: item[1], reverse=True)
                if completed_ranked[num_passwords - 1][1] >= best_active:
                    break

        if len(completed) < num_passwords:
            for ids, score, _ in beams:
                password = decode_candidate_text(tokenizer, ids[0].tolist(), prompt_token_length, prefix, strip_prefix)
                if not password:
                    continue
                current = completed.get(password)
                if current is None or score > current[1]:
                    completed[password] = (ids[0].tolist(), score)
                if len(completed) >= num_passwords:
                    break

        seen: dict[str, float] = {}
        for ids, score in completed.values():
            password = decode_candidate_text(tokenizer, ids, prompt_token_length, prefix, strip_prefix)
            if not password:
                continue
            prob = math.exp(min(score, 0.0))
            if password not in seen or prob > seen[password]:
                seen[password] = prob

        ranked = sorted(seen.items(), key=lambda item: item[1], reverse=True)
        return ranked[:num_passwords]


def build_model_and_tokenizer(config: Config) -> tuple[PassMoE, Any]:
    if config.base_model.lower() == "tiny":
        tokenizer = build_tokenizer(config)
        backbone = TinyCausalBackbone(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.tiny_layers,
            num_heads=config.tiny_heads,
            dropout=config.dropout,
        )
        lm_head = nn.Linear(config.hidden_dim, tokenizer.vocab_size, bias=False)
        model = PassMoE(
            config=config,
            vocab_size=tokenizer.vocab_size,
            hidden_dim=config.hidden_dim,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            backbone=backbone,
            shared_lm_head=lm_head,
        )
        model.merge_report = {}
        return model, tokenizer

    tokenizer = build_tokenizer(config)

    from transformers import AutoModelForCausalLM

    model_path = Path(config.base_model)
    local_only = model_path.exists()
    dtype = dtype_from_string(config.dtype)
    hf_kwargs = {"dtype": dtype, "local_files_only": local_only}
    if config.use_device_map and torch.cuda.is_available() and str(config.device).startswith("cuda"):
        hf_kwargs["device_map"] = "auto"

    try:
        backbone = AutoModelForCausalLM.from_pretrained(config.base_model, **hf_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        hf_kwargs["torch_dtype"] = hf_kwargs.pop("dtype")
        backbone = AutoModelForCausalLM.from_pretrained(config.base_model, **hf_kwargs)
    merge_report: dict[str, Any] = {}
    if config.base_adapter:
        merge_report = merge_lora_adapter(backbone, Path(config.base_adapter))
        print(f"Merged frozen LoRA adapter: {merge_report}")

    for param in backbone.parameters():
        param.requires_grad = False

    hidden_dim = int(backbone.config.hidden_size)
    vocab_size = int(backbone.config.vocab_size)
    shared_lm_head = backbone.get_output_embeddings()
    if shared_lm_head is not None:
        for param in shared_lm_head.parameters():
            param.requires_grad = False

    model = PassMoE(
        config=config,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        backbone=backbone,
        shared_lm_head=shared_lm_head,
    )
    model.merge_report = merge_report
    return model, tokenizer


def build_tokenizer(config: Config) -> Any:
    if config.base_model.lower() == "tiny":
        return CharPasswordTokenizer(config.tiny_vocab)

    from transformers import AutoTokenizer

    model_path = Path(config.base_model)
    local_only = model_path.exists()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def merge_lora_adapter(backbone: nn.Module, adapter_dir: Path) -> dict[str, Any]:
    """Merge a PEFT LoRA adapter into the frozen HF backbone without peft."""

    adapter_dir = adapter_dir.resolve()
    config_path = adapter_dir / "adapter_config.json"
    weight_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {config_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {weight_path}")

    from safetensors.torch import load_file

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(adapter_config.get("r", 1))
    alpha = float(adapter_config.get("lora_alpha", rank))
    scaling = alpha / max(rank, 1)
    tensors = load_file(str(weight_path))

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        if not name.endswith((".lora_A.weight", ".lora_B.weight")):
            continue
        base_name, suffix = name.rsplit(".lora_", 1)
        module_name = normalize_adapter_module_name(base_name)
        slot = "A" if suffix.startswith("A") else "B"
        pairs.setdefault(module_name, {})[slot] = tensor

    merged = 0
    skipped: list[str] = []
    with torch.no_grad():
        for module_name, pair in sorted(pairs.items()):
            if "A" not in pair or "B" not in pair:
                skipped.append(module_name)
                continue
            try:
                target = backbone.get_submodule(module_name)
            except AttributeError:
                skipped.append(module_name)
                continue
            if not hasattr(target, "weight"):
                skipped.append(module_name)
                continue
            weight = target.weight
            delta = (pair["B"].to(torch.float32) @ pair["A"].to(torch.float32)) * scaling
            if tuple(delta.shape) != tuple(weight.shape):
                skipped.append(module_name)
                continue
            weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
            merged += 1

    return {
        "adapter_dir": str(adapter_dir),
        "merged_modules": merged,
        "skipped_modules": len(skipped),
        "skipped_preview": skipped[:5],
    }


def normalize_adapter_module_name(name: str) -> str:
    prefixes = (
        "base_model.model.",
        "base_model.",
    )
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def encode_prefix(tokenizer: Any, prefix: str, device: torch.device) -> torch.Tensor:
    if isinstance(tokenizer, CharPasswordTokenizer):
        ids = [tokenizer.bos_token_id]
        ids.extend(tokenizer.token_to_id.get(ch, tokenizer.unk_token_id) for ch in prefix)
        return torch.tensor([ids], dtype=torch.long, device=device)
    encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
    return encoded["input_ids"].to(device)


def suppress_invalid_tokens(tokenizer: Any, logits: torch.Tensor) -> torch.Tensor:
    blocked = []
    for name in ("pad_token_id", "bos_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, name, None)
        if token_id is not None:
            blocked.append(int(token_id))
    if blocked:
        logits[:, blocked] = -float("inf")
    return logits


def decode_for_features(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def strip_decoded_prefix(text: str, prefix: str, enabled: bool) -> str:
    if enabled and prefix and text.startswith(prefix):
        return text[len(prefix) :]
    return text


def decode_candidate_text(
    tokenizer: Any,
    ids: list[int],
    prompt_token_length: int,
    prefix: str,
    strip_prefix: bool,
) -> str:
    if strip_prefix:
        suffix_ids = ids[prompt_token_length:]
        decoded = decode_for_features(tokenizer, suffix_ids)
        if decoded or not prefix:
            return decoded
    return strip_decoded_prefix(decode_for_features(tokenizer, ids), prefix, strip_prefix)


def feature_vector_for_text(text: str, pii: dict[str, Any] | None = None) -> list[float]:
    from data import FeatureExtractor

    return FeatureExtractor(LEET_DICTIONARY).extract(text, pii)


def router_specialization_target_distribution(
    features: torch.Tensor,
    min_signal: float = 0.05,
    smoothing: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive a weak, interpretable router target from [PII, leet, entropy].

    Expert order is fixed as: PII -> entropy -> leet.
    """

    if features.dim() != 2 or features.size(-1) != 3:
        raise ValueError(f"Expected [batch, 3] features, got {tuple(features.shape)}")

    device = features.device
    pii_score = features[:, 0]
    leet_score = features[:, 1]
    entropy_index = torch.ones(features.size(0), dtype=torch.long, device=device)
    leet_index = torch.full_like(entropy_index, 2)
    pii_index = torch.zeros_like(entropy_index)

    # PII is a targeted-password signal, while entropy is the default fallback.
    # Prioritize PII/semantic evidence before morphology, then use entropy for
    # the generic high-uncertainty expert.
    target_index = entropy_index
    target_index = torch.where(leet_score >= float(min_signal), leet_index, target_index)
    target_index = torch.where(pii_score >= float(min_signal), pii_index, target_index)

    smoothing = min(max(float(smoothing), 0.0), 0.95)
    off_value = smoothing / 2.0
    target_probs = torch.full((features.size(0), 3), off_value, dtype=features.dtype, device=device)
    target_probs.scatter_(1, target_index.unsqueeze(1), 1.0 - smoothing)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return target_index, target_probs


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def module_weight_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    return fallback
