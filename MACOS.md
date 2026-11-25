# Jasna on macOS (Apple Silicon / Metal)

This port adds native macOS Apple Silicon (M1/M2/M3/M4) support via PyTorch MPS, Apple VideoToolbox hardware video encode/decode, and custom Metal shaders for the LTX-2.5 22B restoration model and Diffusion VAE decoder.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4; 32 GB+ Unified Memory recommended to keep all 48 LTX INT8 blocks resident in GPU memory; 16 GB Macs stream overflow blocks automatically)
- macOS 14+ with Metal support
- Python 3.12+
- FFmpeg / FFprobe (`brew install ffmpeg`)

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[mac,dev]"
```

## Architecture Highlights

- **Accelerator Layer (`jasna/accelerator.py`)**: Full `AcceleratorVendor.APPLE` and `KernelBackend.APPLE_MPS` support using `torch.mps.recommended_max_memory()` / `current_allocated_memory()`, GIL-safe `_MpsEvent`, `mps_concurrent_scope` (deferring `torch.mps.empty_cache()` during multi-threaded MPS pipelines to prevent PyTorch `MPSGraphCache::clear()` use-after-free crashes), and `sysctl` chip detection.
- **LTX-2.5 22B Transformer (`jasna/ltx/mps_kernels.py`)**:
  - Uses mixed-precision **W8A16 ConvRot (FP16 GEMM/SDPA compute + BF16 residual stream)** (`rotate(x) @ dequant(qdata, scales)^T`) to keep the ~12.86 GB INT8 checkpoint compact in unified memory while achieving ~1.77x faster GEMM (~9.4 TFLOPS FP16 vs ~5.3 TFLOPS BF16 on Apple M2 Max) and ~1.67x faster SDPA without activation quantization noise.
  - Custom Metal shaders compiled lazily via `torch.mps.compile_shader`:
    - `dequant_int8_{bf16,fp16}_vec4` (vectorized `char4 -> half4/bfloat4` weight dequantization, ~9.7x faster)
    - `hadamard_256_{bf16,fp16,bf16_to_fp16}` (4-stage radix-4 regular Hadamard-256 in threadgroup memory, ~12.8x faster)
    - `modulate_hadamard_256_{bf16,fp16,bf16_to_fp16}` (fused RMSNorm + 2-half AdaLN + Hadamard-256 + BF16->FP16 cast, shared across Q/K/V/gate projections, ~16.4x faster)
    - `gated_heads_hadamard_256_{bf16,fp16}` (fused head transpose + `2*sigmoid` gate + Hadamard-256, ~11.4x faster)
    - `gelu_hadamard_256_{bf16,fp16}` (in-place fused GELU-tanh + Hadamard-256, ~5.5x faster)
  - Uses native Metal Flash SDPA for self-attention and cross-attention.
- **LTX Diffusion VAE (`jasna/models/ltx_vae/na_mps.py`)**:
  - Custom 4-W-query tiled Metal Flash 3D Neighborhood Attention (`na3d_hd64_{bf16,fp16}_w4`) using SIMD group reductions and `fast::exp2` with zero mask materialization.
  - Runs in FP16 (`half_dtype(mps) == torch.float16`, with FP32 encoder tail) and keeps `VideoEncoder` `Conv3d` in contiguous NCDHW format (~2.5x faster than `channels_last_3d` on MPS).
- **Video Codecs (`jasna/media/`)**:
  - Hardware H.264 (`h264_videotoolbox`, NV12) and 10-bit HEVC (`hevc_videotoolbox`, P010) via Apple VideoToolbox with `global_quality` + `qscale`.
  - 10-bit AV1 via `libsvtav1` (`yuv420p10le`).
  - Hardware-assisted decode via PyAV VideoToolbox (`HWAccel("videotoolbox")`) with automatic software fallback.
  - Fixes MPS 10-bit P010 float-to-int16 saturation by viewing storage as `uint16`.

## Verified M2 Max Benchmarks (32 GiB Unified Memory, 512px / 121-frame window)

End-to-end 4-pass LTX-2.5 22B (`distilled1`, 48 blocks, 1920x1080 -> 10-bit HEVC VideoToolbox):

| Stage | Time | Details |
| --- | --- | --- |
| Pass 1 (`scan`) | 0.7 s | 259.7 fps |
| Pass 2 (`encode`) | 7.8 s | 6.09 s GPU VAE encode (FP16 + FP32 tail) |
| Pass 3 (`denoise`) | 37.1 s | 34.70 s/step (0.72 s/block), **48/48 blocks resident** |
| Pass 4 (`compose`) | 12.6 s | 10.63 s single-tile DiffVAE Metal Flash-NA3D + composite + 10-bit HEVC encode |
| **Total (4 passes)** | **58.2 s** | Peak MPS memory: **12.42 GiB (12,714 MiB)** |

## Usage

```bash
# Default device is automatically 'mps' on macOS
jasna --input input.mp4 --output output.mp4

# LTX trial / speed-test run on Apple Silicon
jasna --input input.mp4 --output output.mp4 --restoration-model-name ltx-standard --ltx-trial

# Run unit + Apple Silicon MPS / LTX Metal GPU test suites
pytest tests/test_macos_mps.py tests/test_ltx_mps_gpu.py

# Build / rebuild the clickable macOS application bundle (Jasna.app)
.venv/bin/python scripts/build_macos_app.py
open Jasna.app
```

`Jasna.app` compiles a native ARM64 Mach-O launcher linked against the local `.venv` Python.framework, bundles the multi-resolution `AppIcon.icns`, isolates its runtime config/caches under `.jasna_config/` and `.cache/` inside this directory, and supports moving/aliasing to `/Applications` or the Dock.
