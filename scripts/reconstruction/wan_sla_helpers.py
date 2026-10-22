"""Shared SLA transformer setup for local WanControl5B adapter.

Import this module from predict_v2v_control_5b_multi_image.py.  The base
wan_transformer3d_sla.py is intentionally NOT modified.

The caller must pass existing accelerated attention/quantization primitives
(SparseLinearAttention, Int8Linear, FastRMSNorm, FastLayerNorm) from its
installed VideoX-Fun environment.  The module deliberately does not invent
their import paths.
"""
from __future__ import annotations

import torch

from videox_fun.models.wan_transformer3d import Wan2_2Transformer3DModel
from videox_fun.models.wan_transformer3d_sla import (
    Wan2_2Transformer3DModel_SLA, WanSelfAttention, WanRMSNorm, WanLayerNorm,
)


def select_transformer_class(sla_topk: float):
    """Use SLA when topk <= 0.9, matching the existing SLA implementation."""
    if not 0.0 < sla_topk <= 1.0:
        raise ValueError(f"Invalid sla_topk={sla_topk}")
    return Wan2_2Transformer3DModel if sla_topk > 0.9 else Wan2_2Transformer3DModel_SLA


def replace_attention(model: torch.nn.Module, attention_type: str, sla_topk: float,
                      *, SparseLinearAttention, SageSparseLinearAttention=None):
    if attention_type not in ("sla", "sagesla"):
        raise ValueError(f"Invalid attention type: {attention_type}")
    changed = 0
    for module in model.modules():
        if type(module) is WanSelfAttention:
            if attention_type == "sla":
                module.attn_op.local_attn = SparseLinearAttention(
                    head_dim=module.dim // module.num_heads,
                    topk=sla_topk, BLKQ=128, BLKK=64)
            else:
                if SageSparseLinearAttention is None:
                    raise ValueError("SageSparseLinearAttention is required for sagesla")
                module.attn_op.local_attn = SageSparseLinearAttention(
                    head_dim=module.dim // module.num_heads, topk=sla_topk)
            changed += 1
    if changed == 0:
        raise RuntimeError("No SLA WanSelfAttention modules found; check model class")
    print(f"[WAN SLA] installed {changed} {attention_type} attention modules, topk={sla_topk}",
          flush=True)
    return model


def replace_linear_norm(model: torch.nn.Module, *, Int8Linear, FastRMSNorm,
                        FastLayerNorm, replace_linear=True, replace_norm=True,
                        quantize=True, skip_layer="proj_l"):
    changes = {}
    for name, module in model.blocks.named_modules():
        if not name:
            continue
        if isinstance(module, torch.nn.Linear) and replace_linear:
            if skip_layer not in name:
                changes[name] = Int8Linear.from_linear(module, quantize)
        elif isinstance(module, WanRMSNorm) and replace_norm:
            changes[name] = FastRMSNorm.from_rmsnorm(module)
        elif isinstance(module, WanLayerNorm) and replace_norm:
            changes[name] = FastLayerNorm.from_layernorm(module)
    for name, new_module in changes.items():
        parent = model.blocks
        parts = name.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    if not changes:
        raise RuntimeError("No linear/norm layers replaced. Check the SLA transformer implementation.")
    print(f"[WAN SLA] replaced {len(changes)} linear/norm modules", flush=True)
    return model


def enable_sla(transformer, sla_topk, *, SparseLinearAttention, Int8Linear,
               FastRMSNorm, FastLayerNorm):
    if sla_topk > 0.9:
        return transformer
    replace_attention(transformer, "sla", sla_topk,
                      SparseLinearAttention=SparseLinearAttention)
    replace_linear_norm(transformer, Int8Linear=Int8Linear,
                        FastRMSNorm=FastRMSNorm, FastLayerNorm=FastLayerNorm)
    return transformer
