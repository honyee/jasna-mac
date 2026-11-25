> **macOS Apple Silicon Port (`jasna-mac`)**: This repository is a community macOS (Apple Silicon / Metal MPS) port based on upstream [Kruk2/jasna](https://github.com/Kruk2/jasna) (`v0.12.4`, AGPL-3.0). See [Current macOS Port Status](#macos-apple-silicon-port-status) below and [MACOS.md](MACOS.md) for architecture & benchmark details.
>
> **The only places to get a genuine License code for the supporter models are [Ko-fi](https://ko-fi.com/kruk2), [Unifans](https://app.unifans.io/c/kruk2), and for China the seller in the [Chinese README](README.zh.md#在闲鱼购买).**
>
> **支持者模型的正版许可证码只能通过 [Ko-fi](https://ko-fi.com/kruk2)、[Unifans](https://app.unifans.io/c/kruk2) 或 [闲鱼](README.zh.md#在闲鱼购买) 获取。**

[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna (macOS Apple Silicon Port)

Jasna restores mosaics in videos and images. This fork brings native **macOS Apple Silicon (Metal / MPS)** support, custom Metal kernels for the **LTX-2.5 22B** model & Diffusion VAE, **Apple VideoToolbox** hardware video encoding/decoding, and a clickable **`Jasna.app`** macOS application bundle.

## macOS Apple Silicon Port Status

### What Works Today (Verified on Apple M2 Max, 32 GiB Unified Memory)

| Area | Status | Notes |
| --- | --- | --- |
| **Apple Silicon MPS Backend** | Working | Native `AcceleratorVendor.APPLE` / `KernelBackend.APPLE_MPS` device, memory, stream, and event abstraction ([`jasna/accelerator.py`](jasna/accelerator.py)). |
| **Mosaic Detection** | Working | **RF-DETR** (`rfdetr-v6-large`) and **YOLO** (`lada-yolo-v4`) running natively on MPS. |
| **BasicVSR++ Restoration (Open Source)** | Working (Real Output) | Full real de-mosaicing on MPS (including `deform_conv2d`), verified end-to-end in GUI & CLI (~35.6 s for 300f 1080p with RF-DETR + BasicVSR++). |
| **LTX-2.5 22B Transformer + Diffusion VAE** | Working (Trial / Plaintext Only) | Custom W8A16 ConvRot Metal shaders ([`jasna/ltx/mps_kernels.py`](jasna/ltx/mps_kernels.py), FP16 GEMM/SDPA + BF16 residual) and 4-W-tiled Metal Flash-NA3D ([`jasna/models/ltx_vae/na_mps.py`](jasna/models/ltx_vae/na_mps.py)). Full 4-pass 121f 1080p run completes in **58.2 s** (`48/48` blocks resident, `34.7 s/step`, `12.42 GiB` peak MPS memory). |
| **Video Encode / Decode** | Working | Hardware **H.264** (`h264_videotoolbox`, NV12) and **10-bit HEVC** (`hevc_videotoolbox`, P010), software **10-bit AV1** (`libsvtav1`), plus VideoToolbox hardware-assisted decode with automatic fallback. |
| **PyTorch MPS Concurrency Stability** | Fixed & Tested | Resolved multi-threaded MPS deadlocks (GIL-safe `_MpsEvent`, NumPy buffer staging, no MPS `wait_stream`) and `torch.mps.empty_cache()` / `MPSGraphCache::clear()` use-after-free crashes via `mps_concurrent_scope`; ~4x faster bit-exact separable `conv2d` feather blur on MPS. |
| **Clickable macOS App (`Jasna.app`)** | Working | Native ARM64 Mach-O launcher ([`scripts/build_macos_app.py`](scripts/build_macos_app.py)), multi-res `AppIcon.icns`, isolated `.jasna_config`/`.cache`, native macOS UI/CJK fonts, Homebrew `ffmpeg`/`ffprobe` detection, macOS right-click & `Cmd+A` entry bindings. |
| **Open-Source Upscalers** | Working | `SPAN` and `RealPLKSR` supported on MPS. |

### Current Limitations & Known Gaps

- **Closed-Source Supporter Models & Activation (`jasna.protection`) Are Not Available on macOS**:
  - Official encrypted `.safetensors.enc` weights (**LTX restored weights**, **UNet 4x**, **SD 1.5**) and license activation depend on the upstream author's private closed-source `jasna.protection` module, which is only shipped in official Windows and Linux x86-64 releases (no macOS ARM64 build exists).
  - As a result, in this open-source macOS build:
    - The **Activate Jasna** dialog intentionally keeps its license input and buttons disabled (matching upstream open-source behavior).
    - **LTX runs in Trial Mode (speed/compatibility test)** using random placeholder weights — detected mosaic areas appear covered by a gray haze rather than restored pixels — unless legitimate plaintext `.safetensors` weights are supplied.
    - **Do not purchase a Jasna supporter license expecting to activate it locally on macOS** until upstream ships an official macOS release.
  - For real de-mosaicing on macOS today, select **BasicVSR++** (open-source weights, ready out of the box).
- **NVIDIA-Only Features Unavailable**: TensorRT, NVDEC/VALI, NVENC, NVFP4, RTX Super Res, DLSS 5, Frame Generation, and Topaz TVAI are not available on macOS.
- **Hardware Coverage**: Verified end-to-end on Apple M2 Max (32 GiB Unified Memory). 16 GiB Apple Silicon Macs support automatic LTX block streaming from RAM, but full real-world tuning on 16 GiB hardware is not yet benchmarked.

### macOS Quick Start

```bash
# 1. Prerequisites: Apple Silicon Mac (macOS 14+), Python 3.12+, FFmpeg
brew install python@3.12 ffmpeg

# 2. Clone and install macOS dependencies
git clone https://github.com/honyee/jasna-mac.git
cd jasna-mac
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[mac,dev]"

# 3. Build and launch the native macOS app
.venv/bin/python scripts/build_macos_app.py
open Jasna.app

# 4. Run macOS MPS & LTX Metal GPU test suites (27 tests)
.venv/bin/pytest -v tests/test_macos_mps.py tests/test_ltx_mps_gpu.py
```

For full technical details, Metal kernel notes, and benchmark tables, see **[MACOS.md](MACOS.md)**.

---

## Upstream Jasna Documentation

Jasna restores mosaics in videos and images. It is free, has a simple app
and a command line, and runs on your GPU.

<img width="1280" height="820" alt="Jasna main window" src="assets/jasna-main-window.png" />

## Contents

**Start here**

- [What Jasna can do](#what-jasna-can-do)
- [What you need](#what-you-need)
- [Quick start](#quick-start)
- [Help and community](#help-and-community)
- [Supporting the project](#supporting-the-project) (supporter models)

**Advanced**

- [How Jasna works](#how-jasna-works)
- [Command line](#command-line)
- [All guides](#all-guides)
- [Benchmarks](#benchmarks)
- [Licensing](#licensing)
- [Acknowledgments](#acknowledgments)

## What Jasna can do

- **Restore mosaics in videos.** Only the mosaic areas change. [Quick start](#quick-start)
- **Pick fast or detailed.** BasicVSR++ is fast and free; LTX is slower with more detail (supporters). [Choosing models](docs/en/models.md#restoration-model)
- **Test LTX for free.** Trial mode shows how fast LTX runs on your PC. [LTX options](docs/en/models.md#ltx-options)
- **Use LTX only where it helps.** Big mosaics get LTX, the rest BasicVSR++. [Mosaic scanning](docs/en/segments.md#mosaic-scanning)
- **Restore only some scenes.** Choose ranges with a preview in the Segment Editor. [Segment Editor](docs/en/segments.md#segment-editor-gui)
- **Pause and continue.** Long LTX jobs keep their progress. [Pause and continue](docs/en/models.md#pause-and-continue)
- **Sharpen restored areas.** UNet 4x, RTX Super Res or Topaz. [Secondary restoration](docs/en/models.md#secondary-restoration)
- **Upscale the whole video** to 1080p, 1440p or 4K. [Upscale](docs/en/advanced_processing.md#upscale-the-whole-video)
- **Smoother motion.** Double the frame rate on RTX 40 and 50. [Frame generation](docs/en/gui.md#frame-generation)
- **VR180 videos**, side by side or top/bottom. [VR180](docs/en/vr180.md)
- **Still images** with SD 1.5 (supporters). [Images](docs/en/models.md#still-image-restoration-sd-15)
- **Watch while it restores**, in the app or in a browser. [Video Player](docs/en/gui.md#video-player), [Streaming](docs/en/streaming.md)
- **Many videos at once.** Queue, presets and an action when the queue is done. [Using the GUI](docs/en/gui.md#the-queue)
- **Output extras.** Color LUT, sharpening, 60→30 FPS, play while it's being made. [Advanced processing](docs/en/advanced_processing.md)
- **Command line and rented GPUs.** [CLI](docs/en/cli.md), [Cloud GPUs](docs/en/cloud.md)

## What you need

- **NVIDIA** GTX 16-series / RTX 20-series or newer, with a recent driver
  (610+ on Windows, 580+ on Linux). GTX 10-series and older don't work.
- Or **AMD** Radeon RX 6000 or newer (experimental). RX 7000/9000 are tested;
  RX 6000 is untested.
- A folder on a local disk, with no Chinese, Japanese, accented or other
  non-English characters in its path.

<details>
<summary>More details</summary>

- NVIDIA: your GPU needs compute capability 7.5 or higher in NVIDIA's
  [GPU table](https://developer.nvidia.com/cuda/gpus). Data-center GPUs like
  A100, H100 and B200 don't work.
- AMD: see the table below. UNet 4x secondary restoration is supported.
- Disk format: NTFS on Windows; ext4, Btrfs, XFS or ZFS on Linux. Network disks
  are not supported. Use an SSD; on a hard disk (HDD) Jasna starts and loads
  models slowly.
- LTX needs an NVIDIA RTX or AMD Radeon RX 7000/9000 GPU (AMD integrated GPUs
  untested), at least 8 GB VRAM and 16 GB RAM; on Linux, kernel 5.8 or newer.
  See [What LTX needs](docs/en/models.md#what-ltx-needs).

| AMD GPU | Jasna | LTX |
| --- | --- | --- |
| Radeon RX 9060/9070, Radeon AI PRO R9700 | yes | yes (Windows: beta) |
| Radeon RX 7600–7900, Radeon PRO W7000 | yes | yes (Windows: beta) |
| Radeon 740M/760M/780M/880M/890M, Ryzen AI Max (8050S/8060S) | yes, untested | yes, untested |
| Radeon RX 6000, Radeon PRO W6000 | yes, untested | no |
| Radeon 610M/660M/680M/820M/840M/860M | yes, untested | no |
| Radeon RX 5000, Radeon PRO W5000, RX Vega, RX 500 and older | no | no |

</details>

## Quick start

1. Download the release for your system and GPU.
2. Unpack it into a folder with no Chinese, Japanese, accented or other
   non-English characters in the path.
3. Start the app:
   - Windows: double-click `jasna.exe`. On AMD, first install
     [AMD's driver for ROCm 10.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html)
     (**26.10.41.05**, [download .exe](https://drivers.amd.com/drivers/amd-software-adrenalin-edition-26.10.41.05-win11-b-204865.exe))
     and reboot. You don't need to install ROCm.
   - Linux NVIDIA: run the `jasna` file.
   - Linux AMD: first install AMD's AMF video runtime once. We tested
     [AMF 26.20](https://repo.radeon.com/amf/26.20/):

     ```bash
     curl -fLO https://repo.radeon.com/amf/26.20/amf_installer_26.20.zip
     unzip amf_installer_26.20.zip
     sudo bash amf_installer_26.20.sh
     ```

     Accept AMD's license when prompted, then run `run_jasna_amd.sh`.
     If using Distrobox, install AMF inside the same container as Jasna.
4. Add a video, choose settings and click the **▶** button at the bottom.

**Your first job is slow.** On NVIDIA, Jasna first prepares the models for
your exact GPU. This takes about **15–60 minutes**, only once. Close other
apps and don't use the PC during this time. When you update Jasna, extract it
into a new, empty folder (not over the old one) and copy the `model_weights`
folder there, so you don't wait again. LTX may
still ask to download its model again after an update. On AMD, only UNet 4x
needs this preparation, on its first use.

Every setting has a tooltip: move the mouse over the ⓘ icon. Out of GPU
memory (VRAM)? Lower **Max Clip Size**, for example from `90` to `60`. See
[Tuning](docs/en/tuning.md).

## Help and community

- [App guide](docs/en/gui.md): the queue, presets, output names, the Video
  Player and activation.
- Problem? Click **Logs** at the bottom of the app, then **Export** to save a
  log file. Attach it when you ask for help.

## Supporting the project

Jasna is free. Your support pays for training new models, mostly GPU rental.
Supporters get a License code for the supporter models:

| Support | Models |
| --- | --- |
| **$15 USD** (Ko-fi: **14 EUR**) or more in total | **UNet 4x**: makes restored areas sharper.<br>**SD 1.5**: experimental model for still images. |
| **$25 USD** (Ko-fi: **23 EUR**) | **UNet 4x** and **SD 1.5**, plus<br>**LTX**: much slower than BasicVSR++, but more detail. Includes future updates of this LTX model. |

Supporters who gave **$15 USD or more on or before 14 October 2026** get LTX for free.

**Before you choose LTX:**

- Run LTX in **Trial mode (speed test)** on your PC. It is free and shows if your GPU can
  run LTX and how long it takes. The video is not restored. See
  [What LTX needs](docs/en/models.md#what-ltx-needs).

### Where to support

License codes come **only** from here:

- [Ko-fi](https://ko-fi.com/kruk2)
- [Unifans](https://app.unifans.io/c/kruk2)
- For China: see the [Chinese README](README.zh.md#在闲鱼购买)

Anyone else selling Jasna, codes or modified files is a scam: their keys don't
work, and their files can run hidden code on your PC. Download Jasna only from
GitHub.

### How to activate

1. Support on one of the pages above. You get a **License code**.
2. In Jasna, open **Activate Jasna**, enter the License code, choose
   **This is my PC** or **Rented or shared PC** and press **Activate**.

One License code works on **3 devices at the same time**. On your own PC with
a TPM 2.0 security chip (most Windows 11 PCs), Jasna then works offline.
Elsewhere it needs internet while it uses supporter models. Details and help:
[Activate supporter models](docs/en/gui.md#activate-supporter-models). Rented
GPU: see [Cloud GPUs](docs/en/cloud.md).

Using a License code means you accept the
[supporter license terms](SUPPORTER_TERMS.md).

## Advanced

### How Jasna works

Jasna finds each mosaic, follows it from frame to frame, restores only that
area and puts it back. The rest of the picture is not touched. Almost
everything runs on the GPU. LTX is slower but can add more real detail; see
[Choosing models](docs/en/models.md#restoration-model).

### Command line

The command line can do most of what the app does (not the Video Player,
LTX seed previews, mask suggestions or interactive image mode).

```bash
# One video
jasna --input input.mp4 --output output.mkv

# One image
jasna --input photo.png --output restored.png

# A whole folder
jasna --input input_folder --output output_folder
```

`jasna --help` lists all options. See the [CLI reference](docs/en/cli.md).

### All guides

- [Using the GUI](docs/en/gui.md): queue, presets, output names, Video
  Player, activation.
- [Choosing models](docs/en/models.md): detection models, BasicVSR++ or LTX,
  secondary restoration, SD 1.5 for images.
- [Restoring only parts of a video](docs/en/segments.md): Segment Editor,
  mosaic scan, LTX only for big mosaics.
- [VR180 videos](docs/en/vr180.md): how side-by-side and top/bottom VR are handled.
- [Tuning VRAM and GPU usage](docs/en/tuning.md): clip size, overlap, what to
  do when VRAM runs out.
- [Advanced processing](docs/en/advanced_processing.md): denoise, 60→30 FPS,
  color LUT, sharpening, upscaling the whole video, encoder quality, settings
  saved in the video, actions after the queue.
- [Streaming](docs/en/streaming.md): watch in the browser or in Stash.
- [Cloud GPUs](docs/en/cloud.md): Jasna on a rented GPU (RunPod, Vast).
- [CLI reference](docs/en/cli.md): all command line options.
- [Running from source](docs/en/development.md): for developers. From source
  you can use the free BasicVSR++; activation and supporter models need the
  official release.

### Benchmarks

On an RTX 5090 (Linux), Jasna 0.9.1 restored a 3:22 1080p video with
BasicVSR++ in 34 seconds. Test settings, GPU memory and older versions are in
the [full benchmark report](benchmarks/2026-07-26_release_matrix.md); newer
results are in [benchmarks](benchmarks/).

### Licensing

Jasna's public application source is AGPL-3.0. Models and bundled dependencies
keep their own terms. See [licensing and source availability](LICENSING.md),
[model notices](assets/THIRD_PARTY_MODELS.md) and the
[release source correspondence](RELEASE_SOURCES.md).

### Acknowledgments

- **Lada**: Jasna is inspired
  by Lada and in some places based on it. The `mosaic_restoration_1.2` model
  used by Jasna was trained by ladaapp, the author of Lada.
- **ZeLeFans**: for the VR detection model and the detailed analysis of mosaic
  shapes and VR projection.
