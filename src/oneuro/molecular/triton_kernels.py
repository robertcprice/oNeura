"""Optional Triton kernels for CUDA/ROCm acceleration.

This module is deliberately optional:
- If Triton is unavailable, callers should fall back to pure PyTorch.
- If Triton is present but the active device/runtime rejects a kernel,
  callers should disable the Triton path and keep running.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_TRITON_IMPORT_ERROR: Optional[str] = None

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception as exc:  # pragma: no cover - import depends on local GPU stack
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


NT_DA = 0
NT_GABA = 4


def triton_import_error() -> Optional[str]:
    return _TRITON_IMPORT_ERROR


def triton_runtime_available(device: torch.device) -> bool:
    """Return True when Triton is importable and should be attempted."""
    if not _HAS_TRITON:
        return False
    if os.environ.get("ONEURO_DISABLE_TRITON", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return device.type == "cuda"


if _HAS_TRITON:  # pragma: no branch
    @triton.jit
    def _propagate_edges_kernel(
        fired_ptr,
        syn_pre_ptr,
        syn_post_ptr,
        syn_weight_ptr,
        syn_strength_ptr,
        syn_nt_type_ptr,
        external_current_ptr,
        nt_conc_ptr,
        n_synapses,
        psc_scale,
        nt_release_amount,
        nt_row_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_synapses

        pre = tl.load(syn_pre_ptr + offs, mask=mask, other=0)
        post = tl.load(syn_post_ptr + offs, mask=mask, other=0)
        weight = tl.load(syn_weight_ptr + offs, mask=mask, other=0.0)
        strength = tl.load(syn_strength_ptr + offs, mask=mask, other=0.0)
        nt_type = tl.load(syn_nt_type_ptr + offs, mask=mask, other=0)

        fired = tl.load(fired_ptr + pre, mask=mask, other=0.0)
        sign = tl.where(nt_type == NT_GABA, -1.0, 1.0)
        psc = fired * weight * strength * sign * psc_scale
        tl.atomic_add(external_current_ptr + post, psc, mask=mask)

        nt_offsets = post * nt_row_stride + nt_type
        tl.atomic_add(nt_conc_ptr + nt_offsets, fired * nt_release_amount, mask=mask)

    @triton.jit
    def _stdp_edges_kernel(
        fired_ptr,
        syn_pre_ptr,
        syn_post_ptr,
        syn_nt_type_ptr,
        syn_pre_trace_ptr,
        syn_post_trace_ptr,
        syn_eligibility_ptr,
        syn_strength_ptr,
        nt_conc_ptr,
        n_synapses,
        stdp_decay_pre,
        stdp_decay_post,
        elig_decay,
        nt_row_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_synapses

        pre = tl.load(syn_pre_ptr + offs, mask=mask, other=0)
        post = tl.load(syn_post_ptr + offs, mask=mask, other=0)
        nt_type = tl.load(syn_nt_type_ptr + offs, mask=mask, other=0)

        pre_trace = tl.load(syn_pre_trace_ptr + offs, mask=mask, other=0.0)
        post_trace = tl.load(syn_post_trace_ptr + offs, mask=mask, other=0.0)
        elig = tl.load(syn_eligibility_ptr + offs, mask=mask, other=0.0)
        strength = tl.load(syn_strength_ptr + offs, mask=mask, other=0.0)

        pre_fired = tl.load(fired_ptr + pre, mask=mask, other=0.0)
        post_fired = tl.load(fired_ptr + post, mask=mask, other=0.0)

        pre_trace = pre_trace * stdp_decay_pre + pre_fired * 0.08
        post_trace = post_trace * stdp_decay_post + post_fired * 0.08

        ltp = post_fired * pre_trace
        ltd = pre_fired * post_trace
        elig = elig * elig_decay + (ltp - ltd) * 0.5
        elig = tl.maximum(-2.0, tl.minimum(2.0, elig))

        da_offset = post * nt_row_stride + NT_DA
        da_post = tl.load(nt_conc_ptr + da_offset, mask=mask, other=20.0)
        da_above_rest = tl.maximum(0.0, da_post - 20.0)
        da_gain = tl.minimum(10.0, da_above_rest / 20.0)
        excitatory_mask = tl.where(nt_type != NT_GABA, 1.0, 0.0)

        dw = elig * da_gain * 0.1 * excitatory_mask
        strength = tl.maximum(0.3, tl.minimum(8.0, strength + dw))
        consumed = da_gain > 0.1
        elig = tl.where(consumed, elig * 0.9, elig)

        tl.store(syn_pre_trace_ptr + offs, pre_trace, mask=mask)
        tl.store(syn_post_trace_ptr + offs, post_trace, mask=mask)
        tl.store(syn_eligibility_ptr + offs, elig, mask=mask)
        tl.store(syn_strength_ptr + offs, strength, mask=mask)


def propagate_spikes(
    *,
    fired_f: torch.Tensor,
    syn_pre: torch.Tensor,
    syn_post: torch.Tensor,
    syn_weight: torch.Tensor,
    syn_strength: torch.Tensor,
    syn_nt_type: torch.Tensor,
    external_current: torch.Tensor,
    nt_conc: torch.Tensor,
    psc_scale: float,
    nt_release_amount: float,
) -> None:
    """Edge-parallel spike propagation with atomic accumulation."""
    if not _HAS_TRITON:
        raise RuntimeError(triton_import_error() or "Triton unavailable")
    n_synapses = syn_pre.numel()
    if n_synapses == 0:
        return
    if nt_conc.ndim != 2 or not nt_conc.is_contiguous():
        raise ValueError("nt_conc must be a contiguous (N, NT) tensor")

    grid = lambda meta: (triton.cdiv(n_synapses, meta["BLOCK_SIZE"]),)
    _propagate_edges_kernel[grid](
        fired_f,
        syn_pre,
        syn_post,
        syn_weight,
        syn_strength,
        syn_nt_type,
        external_current,
        nt_conc.view(-1),
        n_synapses,
        float(psc_scale),
        float(nt_release_amount),
        nt_conc.stride(0),
        BLOCK_SIZE=256,
        num_warps=4,
    )


def update_stdp(
    *,
    fired_f: torch.Tensor,
    syn_pre: torch.Tensor,
    syn_post: torch.Tensor,
    syn_nt_type: torch.Tensor,
    syn_pre_trace: torch.Tensor,
    syn_post_trace: torch.Tensor,
    syn_eligibility: torch.Tensor,
    syn_strength: torch.Tensor,
    nt_conc: torch.Tensor,
    stdp_decay_pre: float,
    stdp_decay_post: float,
    elig_decay: float,
) -> None:
    """Edge-parallel STDP update."""
    if not _HAS_TRITON:
        raise RuntimeError(triton_import_error() or "Triton unavailable")
    n_synapses = syn_pre.numel()
    if n_synapses == 0:
        return
    if nt_conc.ndim != 2 or not nt_conc.is_contiguous():
        raise ValueError("nt_conc must be a contiguous (N, NT) tensor")

    grid = lambda meta: (triton.cdiv(n_synapses, meta["BLOCK_SIZE"]),)
    _stdp_edges_kernel[grid](
        fired_f,
        syn_pre,
        syn_post,
        syn_nt_type,
        syn_pre_trace,
        syn_post_trace,
        syn_eligibility,
        syn_strength,
        nt_conc.view(-1),
        n_synapses,
        float(stdp_decay_pre),
        float(stdp_decay_post),
        float(elig_decay),
        nt_conc.stride(0),
        BLOCK_SIZE=256,
        num_warps=4,
    )
