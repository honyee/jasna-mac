"""Metal shaders for LTX-2.5 ConvRot INT8 inference on Apple Silicon (MPS).

Apple Silicon GPUs have no hardware INT8 matrix-multiply instruction faster than
BF16/FP16 GEMM (``torch._int_mm`` on MPS is >10x slower than BF16 ``F.linear``).
Keeping the 8-bit ConvRot checkpoint (~12.86 GB) on GPU and running W8A16:
    rotate(x) @ dequant(qdata, scales)^T
delivers full BF16 hardware throughput (~9.5 TFLOPS on M2 Max) while eliminating
activation quantization noise.

Each fused Metal kernel below compiles lazily once via ``torch.mps.compile_shader``:
- ``dequant_int8_{bf16,fp16}_vec4``: vectorized ``char4 -> bfloat4/half4`` weight
  dequantization (~9.7x faster than PyTorch MPS ops, bit-exact).
- ``hadamard_256_{bf16,fp16}``: 4-stage radix-4 regular Hadamard-256 transform in
  threadgroup memory (~12.8x faster than matmul).
- ``modulate_hadamard_256_{bf16,fp16}``: single-pass RMSNorm + 2-half AdaLN
  modulation + Hadamard-256 rotation, producing a shared ``RotatedActivation``
  consumed by Q/K/V/gate projections.
- ``gated_heads_hadamard_256_{bf16,fp16}``: single-pass ``[H, M, hd]`` head
  transpose + ``2 * sigmoid(gate_logits)`` scaling + Hadamard-256 rotation before
  ``to_out.0``.
- ``gelu_hadamard_256_{bf16,fp16}``: single-pass GELU (tanh approximation) +
  Hadamard-256 rotation (supports in-place execution to save FFN workspace memory).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
import torch.nn.functional as F

ROT_SIZE = 256
EPS = 1e-6


class RotatedActivation(NamedTuple):
    """A ``(rows, in_features)`` activation already rotated by ConvRot Hadamard-256,
    shared across multiple INT8 linear projections that read the same input."""

    x: torch.Tensor
    leading_shape: tuple[int, ...]


_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// 256-point regular Hadamard transform: kron(R4, R4, R4, R4) / 16
// R4 = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
// Executed by 64 threads (tid in 0..63) over threadgroup float buf[256].
// ---------------------------------------------------------------------------
inline void hadamard_256_tg(threadgroup float* buf, uint tid) {
    // Stage 0: stride = 1
    {
        uint base = tid * 4;
        float a = buf[base + 0];
        float b = buf[base + 1];
        float c = buf[base + 2];
        float d = buf[base + 3];
        float s = a + b + c + d;
        buf[base + 0] = s - 2.0f * d;
        buf[base + 1] = s - 2.0f * c;
        buf[base + 2] = s - 2.0f * b;
        buf[base + 3] = s - 2.0f * a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Stage 1: stride = 4
    {
        uint grp = tid >> 2;
        uint lane = tid & 3;
        uint base = (grp << 4) + lane;
        float a = buf[base + 0];
        float b = buf[base + 4];
        float c = buf[base + 8];
        float d = buf[base + 12];
        float s = a + b + c + d;
        buf[base + 0]  = s - 2.0f * d;
        buf[base + 4]  = s - 2.0f * c;
        buf[base + 8]  = s - 2.0f * b;
        buf[base + 12] = s - 2.0f * a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Stage 2: stride = 16
    {
        uint grp = tid >> 4;
        uint lane = tid & 15;
        uint base = (grp << 6) + lane;
        float a = buf[base + 0];
        float b = buf[base + 16];
        float c = buf[base + 32];
        float d = buf[base + 48];
        float s = a + b + c + d;
        buf[base + 0]  = s - 2.0f * d;
        buf[base + 16] = s - 2.0f * c;
        buf[base + 32] = s - 2.0f * b;
        buf[base + 48] = s - 2.0f * a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Stage 3: stride = 64, apply 1/16 = 0.0625 normalization
    {
        uint base = tid;
        float a = buf[base + 0];
        float b = buf[base + 64];
        float c = buf[base + 128];
        float d = buf[base + 192];
        float s = a + b + c + d;
        buf[base + 0]   = (s - 2.0f * d) * 0.0625f;
        buf[base + 64]  = (s - 2.0f * c) * 0.0625f;
        buf[base + 128] = (s - 2.0f * b) * 0.0625f;
        buf[base + 192] = (s - 2.0f * a) * 0.0625f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------
// 1. Vectorized INT8 -> BF16 / FP16 weight dequantization (row-wise scales)
// ---------------------------------------------------------------------------
kernel void dequant_int8_bf16_vec4(
    device const char4* qdata [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device bfloat4* out [[buffer(2)]],
    constant uint& cols_div4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid / cols_div4;
    float s = scales[row];
    char4 q = qdata[gid];
    out[gid] = bfloat4(float4(q) * s);
}

kernel void dequant_int8_fp16_vec4(
    device const char4* qdata [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device half4* out [[buffer(2)]],
    constant uint& cols_div4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid / cols_div4;
    float s = scales[row];
    char4 q = qdata[gid];
    out[gid] = half4(float4(q) * s);
}

// ---------------------------------------------------------------------------
// 2. Standalone Hadamard-256 rotation
// ---------------------------------------------------------------------------
kernel void hadamard_256_bf16(
    device const bfloat4* inp [[buffer(0)]],
    device bfloat4* out [[buffer(1)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    ulong idx = ulong(tg_id) * 64ul + ulong(tid);
    float4 v = float4(inp[idx]);
    uint b = tid * 4;
    buf[b + 0] = v.x; buf[b + 1] = v.y; buf[b + 2] = v.z; buf[b + 3] = v.w;
    hadamard_256_tg(buf, tid);
    out[idx] = bfloat4(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]));
}

kernel void hadamard_256_fp16(
    device const half4* inp [[buffer(0)]],
    device half4* out [[buffer(1)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    ulong idx = ulong(tg_id) * 64ul + ulong(tid);
    float4 v = float4(inp[idx]);
    uint b = tid * 4;
    buf[b + 0] = v.x; buf[b + 1] = v.y; buf[b + 2] = v.z; buf[b + 3] = v.w;
    hadamard_256_tg(buf, tid);
    float4 r = clamp(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]), -65000.0f, 65000.0f);
    out[idx] = half4(r);
}

kernel void hadamard_256_bf16_to_fp16(
    device const bfloat4* inp [[buffer(0)]],
    device half4* out [[buffer(1)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    ulong idx = ulong(tg_id) * 64ul + ulong(tid);
    float4 v = float4(inp[idx]);
    uint b = tid * 4;
    buf[b + 0] = v.x; buf[b + 1] = v.y; buf[b + 2] = v.z; buf[b + 3] = v.w;
    hadamard_256_tg(buf, tid);
    float4 r = clamp(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]), -65000.0f, 65000.0f);
    out[idx] = half4(r);
}

// ---------------------------------------------------------------------------
// 3. Fused RMSNorm + 2-half AdaLN modulate + Hadamard-256 rotation
// 1 threadgroup of 64 threads (2 SIMD groups) per row.
// x: (M, D), shift: (2, D), scale: (2, D), out: (M, D)
// ---------------------------------------------------------------------------
struct ModulateParams {
    uint half_rows;
    uint blocks_per_row; // D / 256
    float inv_dim;       // 1.0 / D
    float eps;
};

kernel void modulate_hadamard_256_bf16(
    device const bfloat4* x [[buffer(0)]],
    device const bfloat4* shift [[buffer(1)]],
    device const bfloat4* scale [[buffer(2)]],
    device bfloat4* out [[buffer(3)]],
    constant ModulateParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    threadgroup float partial[2];

    ulong row_vec4_stride = ulong(p.blocks_per_row) * 64ul;
    ulong row_base = ulong(row) * row_vec4_stride;

    float ss = 0.0f;
    for (uint blk = 0; blk < p.blocks_per_row; ++blk) {
        float4 v = float4(x[row_base + ulong(blk) * 64ul + ulong(tid)]);
        ss += dot(v, v);
    }
    float s_simd = simd_sum(ss);
    if ((tid & 31) == 0) {
        partial[tid >> 5] = s_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt((partial[0] + partial[1]) * p.inv_dim + p.eps);

    ulong half_base = (row < p.half_rows ? 0ul : row_vec4_stride);
    uint b = tid * 4;
    for (uint blk = 0; blk < p.blocks_per_row; ++blk) {
        ulong off = ulong(blk) * 64ul + ulong(tid);
        float4 vx  = float4(x[row_base + off]);
        float4 vsh = float4(shift[half_base + off]);
        float4 vsc = float4(scale[half_base + off]);
        // Round RMSNorm output through bf16 to match PyTorch's F.rms_norm semantics closely,
        // while keeping modulate + Hadamard in fp32 for accuracy.
        float4 normed = float4(bfloat4(vx * inv_rms));
        float4 vm = normed * (1.0f + vsc) + vsh;
        buf[b + 0] = vm.x; buf[b + 1] = vm.y; buf[b + 2] = vm.z; buf[b + 3] = vm.w;
        hadamard_256_tg(buf, tid);
        out[row_base + off] = bfloat4(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]));
    }
}

kernel void modulate_hadamard_256_fp16(
    device const half4* x [[buffer(0)]],
    device const half4* shift [[buffer(1)]],
    device const half4* scale [[buffer(2)]],
    device half4* out [[buffer(3)]],
    constant ModulateParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    threadgroup float partial[2];

    ulong row_vec4_stride = ulong(p.blocks_per_row) * 64ul;
    ulong row_base = ulong(row) * row_vec4_stride;

    float ss = 0.0f;
    for (uint blk = 0; blk < p.blocks_per_row; ++blk) {
        float4 v = float4(x[row_base + ulong(blk) * 64ul + ulong(tid)]);
        ss += dot(v, v);
    }
    float s_simd = simd_sum(ss);
    if ((tid & 31) == 0) {
        partial[tid >> 5] = s_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt((partial[0] + partial[1]) * p.inv_dim + p.eps);

    ulong half_base = (row < p.half_rows ? 0ul : row_vec4_stride);
    uint b = tid * 4;
    for (uint blk = 0; blk < p.blocks_per_row; ++blk) {
        ulong off = ulong(blk) * 64ul + ulong(tid);
        float4 vx  = float4(x[row_base + off]);
        float4 vsh = float4(shift[half_base + off]);
        float4 vsc = float4(scale[half_base + off]);
        float4 normed = float4(half4(vx * inv_rms));
        float4 vm = normed * (1.0f + vsc) + vsh;
        buf[b + 0] = vm.x; buf[b + 1] = vm.y; buf[b + 2] = vm.z; buf[b + 3] = vm.w;
        hadamard_256_tg(buf, tid);
        float4 r = clamp(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]), -65000.0f, 65000.0f);
        out[row_base + off] = half4(r);
    }
}

kernel void modulate_hadamard_256_bf16_to_fp16(
    device const bfloat4* x [[buffer(0)]],
    device const bfloat4* shift [[buffer(1)]],
    device const bfloat4* scale [[buffer(2)]],
    device half4* out [[buffer(3)]],
    constant ModulateParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    threadgroup float partial[2];

    ulong row_vec4_stride = ulong(p.blocks_per_row) * 64ul;
    ulong row_base = ulong(row) * row_vec4_stride;

    float ss = 0.0f;
    for (uint blk = 0; blk < p.blocks_per_row; ++blk) {
        float4 v = float4(x[row_base + ulong(blk) * 64ul + ulong(tid)]);
        ss += dot(v, v);
    }
    float s_simd = simd_sum(ss);
    if ((tid & 31) == 0) {
        partial[tid >> 5] = s_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt((partial[0] + partial[1]) * p.inv_dim + p.eps);

    ulong half_base = (row < p.half_rows ? 0ul : row_vec4_stride);
    uint b = tid * 4;
    for (uint blk = 0; blk < p.blocks_per_row; ++blk) {
        ulong off = ulong(blk) * 64ul + ulong(tid);
        float4 vx  = float4(x[row_base + off]);
        float4 vsh = float4(shift[half_base + off]);
        float4 vsc = float4(scale[half_base + off]);
        float4 normed = float4(bfloat4(vx * inv_rms));
        float4 vm = normed * (1.0f + vsc) + vsh;
        buf[b + 0] = vm.x; buf[b + 1] = vm.y; buf[b + 2] = vm.z; buf[b + 3] = vm.w;
        hadamard_256_tg(buf, tid);
        float4 r = clamp(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]), -65000.0f, 65000.0f);
        out[row_base + off] = half4(r);
    }
}

// ---------------------------------------------------------------------------
// 4. Fused [H, M, hd] head transpose + 2*sigmoid(gate_logits) + Hadamard-256
// 1 threadgroup of 64 threads per 256-element block (tg_id = row * blocks_per_row + blk).
// heads: (H, M, hd), gate_logits: (M, H), out: (M, H*hd)
// ---------------------------------------------------------------------------
struct GatedHeadsParams {
    uint rows;
    uint heads;
    uint hd_div4;        // head_dim / 4
    uint blocks_per_row; // (heads * head_dim) / 256
};

kernel void gated_heads_hadamard_256_bf16(
    device const bfloat4* heads [[buffer(0)]],
    device const bfloat* gate_logits [[buffer(1)]],
    device bfloat4* out [[buffer(2)]],
    constant GatedHeadsParams& p [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    uint row = tg_id / p.blocks_per_row;
    uint blk = tg_id - row * p.blocks_per_row;
    uint vec4_col = blk * 64u + tid;
    uint h = vec4_col / p.hd_div4;
    uint d4 = vec4_col - h * p.hd_div4;

    ulong head_idx = (ulong(h) * ulong(p.rows) + ulong(row)) * ulong(p.hd_div4) + ulong(d4);
    float4 vh = float4(heads[head_idx]);
    float gl = float(gate_logits[ulong(row) * ulong(p.heads) + ulong(h)]);
    float g = 2.0f / (1.0f + exp(-gl));
    float4 vg = vh * g;

    uint b = tid * 4;
    buf[b + 0] = vg.x; buf[b + 1] = vg.y; buf[b + 2] = vg.z; buf[b + 3] = vg.w;
    hadamard_256_tg(buf, tid);
    ulong out_idx = ulong(tg_id) * 64ul + ulong(tid);
    out[out_idx] = bfloat4(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]));
}

kernel void gated_heads_hadamard_256_fp16(
    device const half4* heads [[buffer(0)]],
    device const half* gate_logits [[buffer(1)]],
    device half4* out [[buffer(2)]],
    constant GatedHeadsParams& p [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    uint row = tg_id / p.blocks_per_row;
    uint blk = tg_id - row * p.blocks_per_row;
    uint vec4_col = blk * 64u + tid;
    uint h = vec4_col / p.hd_div4;
    uint d4 = vec4_col - h * p.hd_div4;

    ulong head_idx = (ulong(h) * ulong(p.rows) + ulong(row)) * ulong(p.hd_div4) + ulong(d4);
    float4 vh = float4(heads[head_idx]);
    float gl = float(gate_logits[ulong(row) * ulong(p.heads) + ulong(h)]);
    float g = 2.0f / (1.0f + exp(-gl));
    float4 vg = vh * g;

    uint b = tid * 4;
    buf[b + 0] = vg.x; buf[b + 1] = vg.y; buf[b + 2] = vg.z; buf[b + 3] = vg.w;
    hadamard_256_tg(buf, tid);
    ulong out_idx = ulong(tg_id) * 64ul + ulong(tid);
    float4 r = clamp(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]), -65000.0f, 65000.0f);
    out[out_idx] = half4(r);
}

// ---------------------------------------------------------------------------
// 5. Fused GELU(tanh) + Hadamard-256 rotation (safe for inp == out in-place)
// ---------------------------------------------------------------------------
kernel void gelu_hadamard_256_bf16(
    device const bfloat4* inp [[buffer(0)]],
    device bfloat4* out [[buffer(1)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    ulong idx = ulong(tg_id) * 64ul + ulong(tid);
    float4 x = float4(inp[idx]);
    const float k = 0.7978845608028654f; // sqrt(2 / pi)
    float4 inner = k * (x + 0.044715f * x * x * x);
    float4 g = 0.5f * x * (1.0f + tanh(inner));
    uint b = tid * 4;
    buf[b + 0] = g.x; buf[b + 1] = g.y; buf[b + 2] = g.z; buf[b + 3] = g.w;
    hadamard_256_tg(buf, tid);
    out[idx] = bfloat4(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]));
}

kernel void gelu_hadamard_256_fp16(
    device const half4* inp [[buffer(0)]],
    device half4* out [[buffer(1)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float buf[256];
    ulong idx = ulong(tg_id) * 64ul + ulong(tid);
    float4 x = float4(inp[idx]);
    const float k = 0.7978845608028654f; // sqrt(2 / pi)
    float4 inner = k * (x + 0.044715f * x * x * x);
    float4 g = 0.5f * x * (1.0f + tanh(inner));
    uint b = tid * 4;
    buf[b + 0] = g.x; buf[b + 1] = g.y; buf[b + 2] = g.z; buf[b + 3] = g.w;
    hadamard_256_tg(buf, tid);
    float4 r = clamp(float4(buf[b + 0], buf[b + 1], buf[b + 2], buf[b + 3]), -65000.0f, 65000.0f);
    out[idx] = half4(r);
}
"""


@functools.cache
def _shader_lib():
    return torch.mps.compile_shader(_METAL_SRC)


def is_available() -> bool:
    """Whether custom Metal shader compilation is available on this MPS runtime."""
    return hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def dequant_int8(qdata: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``(out_features, in_features)`` int8 weights with per-row fp32 ``scales``."""
    out_features, in_features = qdata.shape
    if (
        qdata.device.type == "mps"
        and dtype in (torch.bfloat16, torch.float16)
        and in_features % 4 == 0
        and qdata.is_contiguous()
    ):
        scales_f32 = scales if (scales.dtype == torch.float32 and scales.is_contiguous()) else scales.float().contiguous()
        out = torch.empty((out_features, in_features), dtype=dtype, device=qdata.device)
        lib = _shader_lib()
        fn = lib.dequant_int8_bf16_vec4 if dtype == torch.bfloat16 else lib.dequant_int8_fp16_vec4
        fn(qdata, scales_f32, out, in_features // 4, threads=qdata.numel() // 4)
        return out
    return (qdata.float() * scales.float().unsqueeze(1)).to(dtype)


def hadamard_256(x: torch.Tensor, *, inplace: bool = False, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Block regular Hadamard-256 rotation along the last dimension of ``x``."""
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if (
        x.device.type == "mps"
        and x.dtype in (torch.bfloat16, torch.float16)
        and target_dtype in (torch.bfloat16, torch.float16)
        and x.shape[-1] % ROT_SIZE == 0
    ):
        xc = x if x.is_contiguous() else x.contiguous()
        out = xc if (inplace and target_dtype == xc.dtype) else torch.empty_like(xc, dtype=target_dtype)
        n_blocks = xc.numel() // ROT_SIZE
        if n_blocks > 0:
            lib = _shader_lib()
            if xc.dtype == torch.bfloat16 and target_dtype == torch.float16:
                fn = lib.hadamard_256_bf16_to_fp16
            elif xc.dtype == torch.bfloat16:
                fn = lib.hadamard_256_bf16
            else:
                fn = lib.hadamard_256_fp16
            fn(xc, out, threads=n_blocks * 64, group_size=64)
        return out
    from jasna.ltx.transformer import rotate

    return rotate(x).to(target_dtype)


def modulate_rotate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> RotatedActivation:
    """Fused RMSNorm + 2-half AdaLN modulate + Hadamard-256 rotation.

    Args:
        x: ``[1, 2*n, D]`` bf16/fp16 MPS tensor.
        shift: ``[1, 2, D]`` shift rows per half.
        scale: ``[1, 2, D]`` scale rows per half.
        out_dtype: Output dtype (default: same as ``x``; pass ``torch.float16`` to run
            subsequent GEMMs at full Apple Silicon FP16 rate while keeping residual ``x`` in bf16).
    """
    rows, dim = x.shape[1], x.shape[2]
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if (
        x.device.type == "mps"
        and x.dtype in (torch.bfloat16, torch.float16)
        and target_dtype in (torch.bfloat16, torch.float16)
        and rows % 2 == 0
        and dim % ROT_SIZE == 0
    ):
        xc = x[0].contiguous()
        sh = shift[0].to(dtype=xc.dtype).contiguous()
        sc = scale[0].to(dtype=xc.dtype).contiguous()
        out = torch.empty_like(xc, dtype=target_dtype)
        # Pack ModulateParams { uint half_rows; uint blocks_per_row; float inv_dim; float eps; }
        params = torch.tensor(
            [rows // 2, dim // ROT_SIZE, 0, 0], dtype=torch.uint32, device=xc.device
        )
        params_f = params.view(torch.float32)
        params_f[2] = 1.0 / float(dim)
        params_f[3] = EPS
        lib = _shader_lib()
        if xc.dtype == torch.bfloat16 and target_dtype == torch.float16:
            fn = lib.modulate_hadamard_256_bf16_to_fp16
        elif xc.dtype == torch.bfloat16:
            fn = lib.modulate_hadamard_256_bf16
        else:
            fn = lib.modulate_hadamard_256_fp16
        fn(xc, sh, sc, out, params, threads=rows * 64, group_size=64)
        return RotatedActivation(out, (x.shape[0], rows))

    from jasna.ltx.transformer import _modulate, rms_norm

    mod = _modulate(rms_norm(x), shift, scale)
    return RotatedActivation(hadamard_256(mod, out_dtype=target_dtype).reshape(-1, dim), x.shape[:-1])


def gated_heads_rotate(heads: torch.Tensor, gate_logits: torch.Tensor) -> RotatedActivation:
    """Fused ``[H, M, hd]`` transpose + ``2 * sigmoid(gate_logits)`` + Hadamard-256 rotation.

    Args:
        heads: ``[H, M, hd]`` attention output (batch dim stripped).
        gate_logits: ``[M, H]`` raw gate logits.
    """
    num_heads, rows, head_dim = heads.shape
    dim = num_heads * head_dim
    if (
        heads.device.type == "mps"
        and heads.dtype in (torch.bfloat16, torch.float16)
        and head_dim % 4 == 0
        and dim % ROT_SIZE == 0
    ):
        hc = heads if heads.is_contiguous() else heads.contiguous()
        gl = gate_logits.to(dtype=hc.dtype).contiguous()
        out = torch.empty((rows, dim), dtype=hc.dtype, device=hc.device)
        blocks_per_row = dim // ROT_SIZE
        params = torch.tensor(
            [rows, num_heads, head_dim // 4, blocks_per_row], dtype=torch.uint32, device=hc.device
        )
        lib = _shader_lib()
        fn = lib.gated_heads_hadamard_256_bf16 if hc.dtype == torch.bfloat16 else lib.gated_heads_hadamard_256_fp16
        fn(hc, gl, out, params, threads=rows * blocks_per_row * 64, group_size=64)
        return RotatedActivation(out, (1, rows))

    gates = 2.0 * torch.sigmoid(gate_logits)
    merged = (heads.transpose(0, 1) * gates.unsqueeze(-1)).reshape(rows, dim)
    return RotatedActivation(hadamard_256(merged), (1, rows))


def gelu_hadamard_256(x: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
    """Fused GELU (tanh approximation) + Hadamard-256 rotation along the last dimension."""
    if (
        x.device.type == "mps"
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.shape[-1] % ROT_SIZE == 0
    ):
        xc = x if x.is_contiguous() else x.contiguous()
        out = xc if inplace else torch.empty_like(xc)
        n_blocks = xc.numel() // ROT_SIZE
        if n_blocks > 0:
            lib = _shader_lib()
            fn = lib.gelu_hadamard_256_bf16 if xc.dtype == torch.bfloat16 else lib.gelu_hadamard_256_fp16
            fn(xc, out, threads=n_blocks * 64, group_size=64)
        return out
    return hadamard_256(F.gelu(x, approximate="tanh"))


def int8_linear_mps(
    x: torch.Tensor | RotatedActivation,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """ConvRot W8A16 linear on Apple MPS: rotate (unless already rotated) -> dequant -> BF16 GEMM."""
    if isinstance(x, RotatedActivation):
        rot_x = x.x
        leading = x.leading_shape
    else:
        in_features = qdata.shape[1]
        rot_x = hadamard_256(x.reshape(-1, in_features))
        leading = x.shape[:-1]
    weight = dequant_int8(qdata, scales, rot_x.dtype)
    bias_t = bias.to(rot_x.dtype) if (bias is not None and bias.dtype != rot_x.dtype) else bias
    out = F.linear(rot_x, weight, bias_t)
    return out.reshape(*leading, qdata.shape[0])


def gelu_int8_linear_mps(
    h: torch.Tensor,
    qdata: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    inplace_gelu: bool = False,
) -> torch.Tensor:
    """Fused GELU(tanh) + Hadamard-256 + W8A16 GEMM."""
    rot_h = gelu_hadamard_256(h.reshape(-1, qdata.shape[1]), inplace=inplace_gelu)
    weight = dequant_int8(qdata, scales, rot_h.dtype)
    bias_t = bias.to(rot_h.dtype) if (bias is not None and bias.dtype != rot_h.dtype) else bias
    return F.linear(rot_h, weight, bias_t).reshape(*h.shape[:-1], qdata.shape[0])


def feed_forward_mps(
    ff_in: RotatedActivation,
    up_qdata: torch.Tensor,
    up_scales: torch.Tensor,
    up_bias: torch.Tensor | None,
    down_qdata: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor | None,
    residual: torch.Tensor,
    gate: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """Chunked ConvRot W8A16 feed-forward block on MPS.

    Dequantizes ``ff.net.0.proj`` and ``ff.net.2`` once per block (rather than once per
    chunk) and runs GELU+Hadamard-256 in place on each chunk's hidden buffer.
    """
    rot_in = ff_in.x
    rows, dim = rot_in.shape
    dtype = rot_in.dtype
    w_up = dequant_int8(up_qdata, up_scales, dtype)
    b_up = up_bias.to(dtype) if (up_bias is not None and up_bias.dtype != dtype) else up_bias
    w_down = dequant_int8(down_qdata, down_scales, dtype)
    b_down = down_bias.to(dtype) if (down_bias is not None and down_bias.dtype != dtype) else down_bias

    ff_out = torch.empty((1, rows, dim), dtype=dtype, device=rot_in.device)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        hidden = F.linear(rot_in[start:end], w_up, b_up)
        rot_hidden = gelu_hadamard_256(hidden, inplace=True)
        ff_out[0, start:end] = F.linear(rot_hidden, w_down, b_down)
    del w_up, w_down
    halves = ff_out.to(residual.dtype).unflatten(1, (2, -1))
    return residual + (halves * gate[:, :, None]).flatten(1, 2)
