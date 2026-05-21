import math
from fnmatch import fnmatch
from typing import Iterable, List, Optional

import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for LoRA.")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(linear.weight.new_zeros((rank, self.in_features)))
        self.lora_B = nn.Parameter(linear.weight.new_zeros((self.out_features, rank)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


def _matches_pattern(name: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            if fnmatch(name, pattern):
                return True
        elif pattern in name:
            return True
    return False


def _get_parent_module(model: nn.Module, module_name: str):
    if "." not in module_name:
        return model, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    return parent, child_name


def module_name_only(full_name: str) -> str:
    return full_name.rsplit(".", 1)[-1]


def inject_lora_by_module_name(
    model: nn.Module,
    include_patterns: List[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_linear_names: Optional[List[str]] = None,
) -> List[str]:
    if not include_patterns:
        return []

    replaced = []
    named_modules = list(model.named_modules())
    for full_name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        if target_linear_names is not None and module_name_only(full_name) not in target_linear_names:
            continue
        if not _matches_pattern(full_name, include_patterns):
            continue
        parent, child = _get_parent_module(model, full_name)
        setattr(parent, child, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(full_name)
    return replaced


def mark_only_lora_trainable(
    model: nn.Module, also_train_keywords: Optional[List[str]] = None
) -> List[str]:
    also_train_keywords = also_train_keywords or []
    trainable = []
    for name, param in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            param.requires_grad_(True)
            trainable.append(name)
        elif any(keyword in name for keyword in also_train_keywords):
            param.requires_grad_(True)
            trainable.append(name)
        else:
            param.requires_grad_(False)
    return trainable


def count_trainable_parameters(model: nn.Module):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable
