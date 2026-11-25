"""Metal Flash 3D Neighborhood Attention (``na3d``) for the LTX Diffusion VAE decoder on Apple Silicon.

Implements exact NATTEN non-causal 3D neighborhood attention (inward-clamped border windows,
pre-scaled Q) with a 4-W-query tiled online-softmax Flash kernel for ``head_dim == 64``:
- One 32-thread SIMD group processes 4 consecutive W queries simultaneously, sharing K/V loads
  across the union of their ``kw`` windows (~3x lower memory bandwidth than 1-query-per-SIMD).
- Uses hardware ``simd_sum`` for dot products and ``fast::exp2`` with pre-scaled ``q * LOG2E``
  for single-cycle exponential evaluation, with zero attention-mask materialization.
- Falls back cleanly to ``eager_na.na3d`` for unsupported head dimensions or dtypes.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import torch

from jasna.models.ltx_vae import eager_na

_METAL_NA3D_SRC = r"""
#include <metal_stdlib>
using namespace metal;

struct Na3dParams {
    uint B;
    uint T;
    uint H;
    uint Wq;
    uint Wk;
    uint NH;
    uint kt;
    uint kh;
    uint kw;
    uint query_w_start;
    uint Wq_tiles; // ceil(Wq / 4)
};

// 4-W-query tiled Flash-NA3D for head_dim == 64 (BF16).
// 1 SIMD group (32 threads) processes up to 4 adjacent W queries at (b, t, h, wq_base..+3, nh).
// Each lane holds 2 channels (2*lane, 2*lane+1) of each query and output accumulator.
kernel void na3d_hd64_bf16_w4(
    device const bfloat2* Q [[buffer(0)]],
    device const bfloat2* K [[buffer(1)]],
    device const bfloat2* V [[buffer(2)]],
    device bfloat2* O [[buffer(3)]],
    constant Na3dParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint lane = gid & 31u;
    uint tile_idx = gid >> 5;
    uint total_tiles = p.B * p.T * p.H * p.Wq_tiles * p.NH;
    if (tile_idx >= total_tiles) return;

    uint rem = tile_idx;
    uint nh = rem % p.NH;       rem /= p.NH;
    uint wt = rem % p.Wq_tiles; rem /= p.Wq_tiles;
    uint h  = rem % p.H;        rem /= p.H;
    uint t  = rem % p.T;        rem /= p.T;
    uint b  = rem;

    uint wq0 = wt * 4u;
    uint kt = min(p.kt, p.T);
    uint kh = min(p.kh, p.H);
    uint kw = min(p.kw, p.Wk);

    uint t0 = uint(min(max(int(t) - int(kt >> 1), 0), int(p.T - kt)));
    uint h0 = uint(min(max(int(h) - int(kh >> 1), 0), int(p.H - kh)));

    const float LOG2E = 1.4426950408889634f;

    ulong q_stride_w = ulong(p.NH) * 32ul;
    ulong q_stride_h = ulong(p.Wq) * q_stride_w;
    ulong q_stride_t = ulong(p.H)  * q_stride_h;
    ulong q_base = ulong(b) * ulong(p.T) * q_stride_t
                 + ulong(t) * q_stride_t
                 + ulong(h) * q_stride_h
                 + ulong(nh) * 32ul
                 + ulong(lane);

    float2 qv[4];
    uint w0[4];
    bool valid[4];
    uint w_min = p.Wk;
    uint w_max = 0u;

    for (uint i = 0u; i < 4u; ++i) {
        uint wq = wq0 + i;
        valid[i] = (wq < p.Wq);
        if (valid[i]) {
            qv[i] = float2(Q[q_base + ulong(wq) * q_stride_w]) * LOG2E;
            uint wg = wq + p.query_w_start;
            uint ws = uint(min(max(int(wg) - int(kw >> 1), 0), int(p.Wk - kw)));
            w0[i] = ws;
            w_min = min(w_min, ws);
            w_max = max(w_max, ws + kw);
        } else {
            qv[i] = float2(0.0f);
            w0[i] = 0u;
        }
    }

    ulong kv_stride_w = ulong(p.NH) * 32ul;
    ulong kv_stride_h = ulong(p.Wk) * kv_stride_w;
    ulong kv_stride_t = ulong(p.H)  * kv_stride_h;
    ulong kv_base_b = ulong(b) * ulong(p.T) * kv_stride_t + ulong(nh) * 32ul + ulong(lane);

    float m0 = -INFINITY, m1 = -INFINITY, m2 = -INFINITY, m3 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    float2 a0 = float2(0.0f), a1 = float2(0.0f), a2 = float2(0.0f), a3 = float2(0.0f);

    uint w0_0 = w0[0], w1_0 = w0_0 + kw;
    uint w0_1 = w0[1], w1_1 = w0_1 + kw;
    uint w0_2 = w0[2], w1_2 = w0_2 + kw;
    uint w0_3 = w0[3], w1_3 = w0_3 + kw;

    float2 q0 = qv[0], q1 = qv[1], q2 = qv[2], q3 = qv[3];
    bool v0 = valid[0], v1 = valid[1], v2 = valid[2], v3 = valid[3];

    for (uint dt = 0u; dt < kt; ++dt) {
        ulong off_t = kv_base_b + ulong(t0 + dt) * kv_stride_t;
        for (uint dh = 0u; dh < kh; ++dh) {
            ulong off_th = off_t + ulong(h0 + dh) * kv_stride_h;
            for (uint w = w_min; w < w_max; ++w) {
                ulong kv_off = off_th + ulong(w) * kv_stride_w;
                float2 kv = float2(K[kv_off]);
                float2 vv = float2(V[kv_off]);

                if (v0 && w >= w0_0 && w < w1_0) {
                    float dot = simd_sum(fma(q0.x, kv.x, q0.y * kv.y));
                    float mn = max(m0, dot);
                    float alpha = fast::exp2(m0 - mn);
                    float prob  = fast::exp2(dot - mn);
                    l0 = fma(l0, alpha, prob);
                    a0 = fma(float2(prob), vv, a0 * alpha);
                    m0 = mn;
                }
                if (v1 && w >= w0_1 && w < w1_1) {
                    float dot = simd_sum(fma(q1.x, kv.x, q1.y * kv.y));
                    float mn = max(m1, dot);
                    float alpha = fast::exp2(m1 - mn);
                    float prob  = fast::exp2(dot - mn);
                    l1 = fma(l1, alpha, prob);
                    a1 = fma(float2(prob), vv, a1 * alpha);
                    m1 = mn;
                }
                if (v2 && w >= w0_2 && w < w1_2) {
                    float dot = simd_sum(fma(q2.x, kv.x, q2.y * kv.y));
                    float mn = max(m2, dot);
                    float alpha = fast::exp2(m2 - mn);
                    float prob  = fast::exp2(dot - mn);
                    l2 = fma(l2, alpha, prob);
                    a2 = fma(float2(prob), vv, a2 * alpha);
                    m2 = mn;
                }
                if (v3 && w >= w0_3 && w < w1_3) {
                    float dot = simd_sum(fma(q3.x, kv.x, q3.y * kv.y));
                    float mn = max(m3, dot);
                    float alpha = fast::exp2(m3 - mn);
                    float prob  = fast::exp2(dot - mn);
                    l3 = fma(l3, alpha, prob);
                    a3 = fma(float2(prob), vv, a3 * alpha);
                    m3 = mn;
                }
            }
        }
    }

    if (v0) O[q_base + ulong(wq0 + 0u) * q_stride_w] = bfloat2(a0 / l0);
    if (v1) O[q_base + ulong(wq0 + 1u) * q_stride_w] = bfloat2(a1 / l1);
    if (v2) O[q_base + ulong(wq0 + 2u) * q_stride_w] = bfloat2(a2 / l2);
    if (v3) O[q_base + ulong(wq0 + 3u) * q_stride_w] = bfloat2(a3 / l3);
}

// 4-W-query tiled Flash-NA3D for head_dim == 64 (FP16).
kernel void na3d_hd64_fp16_w4(
    device const half2* Q [[buffer(0)]],
    device const half2* K [[buffer(1)]],
    device const half2* V [[buffer(2)]],
    device half2* O [[buffer(3)]],
    constant Na3dParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint lane = gid & 31u;
    uint tile_idx = gid >> 5;
    uint total_tiles = p.B * p.T * p.H * p.Wq_tiles * p.NH;
    if (tile_idx >= total_tiles) return;

    uint rem = tile_idx;
    uint nh = rem % p.NH;       rem /= p.NH;
    uint wt = rem % p.Wq_tiles; rem /= p.Wq_tiles;
    uint h  = rem % p.H;        rem /= p.H;
    uint t  = rem % p.T;        rem /= p.T;
    uint b  = rem;

    uint wq0 = wt * 4u;
    uint kt = min(p.kt, p.T);
    uint kh = min(p.kh, p.H);
    uint kw = min(p.kw, p.Wk);

    uint t0 = uint(min(max(int(t) - int(kt >> 1), 0), int(p.T - kt)));
    uint h0 = uint(min(max(int(h) - int(kh >> 1), 0), int(p.H - kh)));

    const float LOG2E = 1.4426950408889634f;

    ulong q_stride_w = ulong(p.NH) * 32ul;
    ulong q_stride_h = ulong(p.Wq) * q_stride_w;
    ulong q_stride_t = ulong(p.H)  * q_stride_h;
    ulong q_base = ulong(b) * ulong(p.T) * q_stride_t
                 + ulong(t) * q_stride_t
                 + ulong(h) * q_stride_h
                 + ulong(nh) * 32ul
                 + ulong(lane);

    float2 qv[4];
    uint w0[4];
    bool valid[4];
    uint w_min = p.Wk;
    uint w_max = 0u;

    for (uint i = 0u; i < 4u; ++i) {
        uint wq = wq0 + i;
        valid[i] = (wq < p.Wq);
        if (valid[i]) {
            qv[i] = float2(Q[q_base + ulong(wq) * q_stride_w]) * LOG2E;
            uint wg = wq + p.query_w_start;
            uint ws = uint(min(max(int(wg) - int(kw >> 1), 0), int(p.Wk - kw)));
            w0[i] = ws;
            w_min = min(w_min, ws);
            w_max = max(w_max, ws + kw);
        } else {
            qv[i] = float2(0.0f);
            w0[i] = 0u;
        }
    }

    ulong kv_stride_w = ulong(p.NH) * 32ul;
    ulong kv_stride_h = ulong(p.Wk) * kv_stride_w;
    ulong kv_stride_t = ulong(p.H)  * kv_stride_h;
    ulong kv_base_b = ulong(b) * ulong(p.T) * kv_stride_t + ulong(nh) * 32ul + ulong(lane);

    float m0 = -INFINITY, m1 = -INFINITY, m2 = -INFINITY, m3 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
    float2 a0 = float2(0.0f), a1 = float2(0.0f), a2 = float2(0.0f), a3 = float2(0.0f);

    uint w0_0 = w0[0], w1_0 = w0_0 + kw;
    uint w0_1 = w0[1], w1_1 = w0_1 + kw;
    uint w0_2 = w0[2], w1_2 = w0_2 + kw;
    uint w0_3 = w0[3], w1_3 = w0_3 + kw;

    float2 q0 = qv[0], q1 = qv[1], q2 = qv[2], q3 = qv[3];
    bool v0 = valid[0], v1 = valid[1], v2 = valid[2], v3 = valid[3];

    for (uint dt = 0u; dt < kt; ++dt) {
        ulong off_t = kv_base_b + ulong(t0 + dt) * kv_stride_t;
        for (uint dh = 0u; dh < kh; ++dh) {
            ulong off_th = off_t + ulong(h0 + dh) * kv_stride_h;
            for (uint w = w_min; w < w_max; ++w) {
                ulong kv_off = off_th + ulong(w) * kv_stride_w;
                float2 kv = float2(K[kv_off]);
                float2 vv = float2(V[kv_off]);

                if (v0 && w >= w0_0 && w < w1_0) {
                    float dot = simd_sum(fma(q0.x, kv.x, q0.y * kv.y));
                    float mn = max(m0, dot);
                    float alpha = fast::exp2(m0 - mn);
                    float prob  = fast::exp2(dot - mn);
                    l0 = fma(l0, alpha, prob);
                    a0 = fma(float2(prob), vv, a0 * alpha);
                    m0 = mn;
                }
                if (v1 && w >= w0_1 && w < w1_1) {
                    float dot = simd_sum(fma(q1.x, kv.x, q1.y * kv.y));
                    float mn = max(m1, dot);
                    float alpha = fast::exp2(m1 - mn);
                    float prob  = fast::exp2(dot - mn);
                    l1 = fma(l1, alpha, prob);
                    a1 = fma(float2(prob), vv, a1 * alpha);
                    m1 = mn;
                }
                if (v2 && w >= w0_2 && w < w1_2) {
                    float dot = simd_sum(fma(q2.x, kv.x, q2.y * kv.y));
                    float mn = max(m2, dot);
                    float alpha = fast::exp2(m2 - mn);
                    float prob  = fast::exp2(dot - mn);
                    l2 = fma(l2, alpha, prob);
                    a2 = fma(float2(prob), vv, a2 * alpha);
                    m2 = mn;
                }
                if (v3 && w >= w0_3 && w < w1_3) {
                    float dot = simd_sum(fma(q3.x, kv.x, q3.y * kv.y));
                    float mn = max(m3, dot);
                    float alpha = fast::exp2(m3 - mn);
                    float prob  = fast::exp2(dot - mn);
                    l3 = fma(l3, alpha, prob);
                    a3 = fma(float2(prob), vv, a3 * alpha);
                    m3 = mn;
                }
            }
        }
    }

    if (v0) O[q_base + ulong(wq0 + 0u) * q_stride_w] = half2(a0 / l0);
    if (v1) O[q_base + ulong(wq0 + 1u) * q_stride_w] = half2(a1 / l1);
    if (v2) O[q_base + ulong(wq0 + 2u) * q_stride_w] = half2(a2 / l2);
    if (v3) O[q_base + ulong(wq0 + 3u) * q_stride_w] = half2(a3 / l3);
}
"""


@functools.cache
def _shader_lib():
    return torch.mps.compile_shader(_METAL_NA3D_SRC)


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: Sequence[int],
    query_w_start: int,
) -> torch.Tensor:
    """Metal Flash-NA3D over ``(B, T, H, W, NH, HD)`` tensors on MPS (pre-scaled Q)."""
    b, t, h, wq, nh, hd = q.shape
    wk = k.shape[3]
    kt, kh, kw = int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2])
    if (
        q.device.type != "mps"
        or hd != 64
        or q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        return eager_na.na3d(q, k, v, (kt, kh, kw), int(query_w_start))

    qc = q if q.is_contiguous() else q.contiguous()
    kc = k if k.is_contiguous() else k.contiguous()
    vc = v if v.is_contiguous() else v.contiguous()
    out = torch.empty_like(qc)
    wq_tiles = (wq + 3) // 4
    params = torch.tensor(
        [b, t, h, wq, wk, nh, kt, kh, kw, int(query_w_start), wq_tiles],
        dtype=torch.uint32,
        device=qc.device,
    )
    total_threads = b * t * h * wq_tiles * nh * 32
    if total_threads > 0:
        lib = _shader_lib()
        fn = lib.na3d_hd64_bf16_w4 if qc.dtype == torch.bfloat16 else lib.na3d_hd64_fp16_w4
        fn(qc, kc, vc, out, params, threads=total_threads, group_size=256)
    return out
