from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from jasna.accelerator import (
    AcceleratorVendor,
    KernelBackend,
    default_device,
    device_free_bytes,
    device_name,
    device_total_bytes,
    device_used_bytes,
    has_mps,
    is_apple_device,
    is_gpu_device,
    is_gpu_tensor,
    kernel_backend,
    new_event,
    synchronize,
    vendor_for_device,
)
from jasna.gui import ltx_models, system_checks
from jasna.media.cas import GpuCasSharpener
from jasna.media.encoder_options import resolve_encoder_options, videotoolbox_global_quality
from jasna.media.encoder_settings import encoder_cq_spec, validate_encoder_settings
from jasna.media.lut import CubeLut, GpuLutApplier
from jasna.media.probe import VideoMetadata, get_video_meta_data
from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.splice import KeyframeIndex, resolve_smart_encoder_settings
from jasna.media.video_decoder import VideoReader
from jasna.media.video_encoder import VideoEncoder
from jasna.media.yuv_to_rgb import YuvToRgbConverter
from jasna.session_factory import secondary_restorations_for, upscale_models_for
from jasna.system_requirements import check_gpu_driver_version, check_supported_gpu
from jasna.tensor_utils import compiles, half_dtype


def test_apple_encoder_settings_and_options() -> None:
    for codec, expected_default in (("h264", 25), ("hevc", 28), ("av1", 35)):
        spec = encoder_cq_spec(codec, AcceleratorVendor.APPLE)
        assert spec.default == expected_default
        validated = validate_encoder_settings({"cq": expected_default}, codec=codec, vendor=AcceleratorVendor.APPLE)
        assert validated == {"cq": expected_default}

    # Lower CQ = higher VideoToolbox quality score
    assert videotoolbox_global_quality(15) > videotoolbox_global_quality(28) > videotoolbox_global_quality(45)

    from av.video.reformatter import ColorRange as AvColorRange, Colorspace as AvColorspace

    meta_10 = VideoMetadata(
        video_file="in.mp4",
        video_width=640,
        video_height=360,
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        average_fps=30.0,
        duration=2.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=60,
        codec_name="hevc",
        pixel_format="p010le",
        is_10bit=True,
        video_bitrate=4_000_000,
    )
    spec_hevc, opts_hevc = resolve_encoder_options(
        AcceleratorVendor.APPLE, "hevc", meta_10, {"cq": 28}, smart_fragment=False
    )
    assert spec_hevc.encoder_name == "hevc_videotoolbox"
    assert spec_hevc.frame_format == "p010le"
    assert spec_hevc.ten_bit is True
    assert opts_hevc["profile"] == "main10"
    assert int(opts_hevc["global_quality"]) == int(videotoolbox_global_quality(28))

    spec_h264, opts_h264 = resolve_encoder_options(
        AcceleratorVendor.APPLE, "h264", meta_10, {"cq": 25}, smart_fragment=False
    )
    assert spec_h264.encoder_name == "h264_videotoolbox"
    assert spec_h264.frame_format == "nv12"
    assert spec_h264.ten_bit is False
    assert opts_h264["profile"] == "high"

    spec_av1, opts_av1 = resolve_encoder_options(
        AcceleratorVendor.APPLE, "av1", meta_10, {"cq": 35}, smart_fragment=False
    )
    assert spec_av1.encoder_name == "libsvtav1"
    assert spec_av1.frame_format == "yuv420p10le"
    assert spec_av1.ten_bit is True
    assert opts_av1["crf"] == "35"

    meta_h264 = VideoMetadata(
        video_file="in.mp4",
        video_width=640,
        video_height=360,
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        average_fps=30.0,
        duration=2.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=60,
        codec_name="h264",
        pixel_format="yuv420p",
        is_10bit=False,
        profile="High",
    )
    index = KeyframeIndex(pts=(0, 30), time_base=Fraction(1, 30), start_pts=0, end_pts=60, max_b_frames=2)
    smart = resolve_smart_encoder_settings("h264", meta_h264, index, {"cq": 25}, vendor=AcceleratorVendor.APPLE)
    assert smart["profile"] == "high"
    assert smart["g"] == 30


def test_apple_vendor_capabilities() -> None:
    assert secondary_restorations_for(AcceleratorVendor.APPLE) == ("none", "unet-4x")
    assert upscale_models_for(AcceleratorVendor.APPLE, "darwin") == ("none", "span", "realplksr")
    assert ltx_models.card_unavailable_reason(AcceleratorVendor.APPLE) is None


@pytest.mark.skipif(not has_mps(), reason="requires Apple Silicon MPS")
class TestAppleSiliconMps:
    def test_accelerator_and_system_checks(self) -> None:
        mps = torch.device("mps")
        assert torch.device(default_device()) == mps
        assert vendor_for_device(mps) is AcceleratorVendor.APPLE
        assert kernel_backend(mps) is KernelBackend.APPLE_MPS
        assert kernel_backend(mps).is_apple is True
        assert is_apple_device(mps) is True
        assert is_gpu_device(mps) is True
        assert is_gpu_tensor(torch.zeros(1, device=mps)) is True
        assert half_dtype(mps) == torch.float16
        assert compiles(mps, torch.float16) is False
        assert compiles(mps, torch.bfloat16) is False

        total = device_total_bytes(mps)
        free = device_free_bytes(mps)
        used = device_used_bytes(mps)
        assert total > 0 and 0 <= free <= total and used >= 0
        assert "Apple" in device_name(mps)

        ev0 = new_event(mps, enable_timing=True)
        ev1 = new_event(mps, enable_timing=True)
        ev0.record()
        _ = torch.randn(64, 64, device=mps) @ torch.randn(64, 64, device=mps)
        ev1.record()
        ev1.synchronize()
        assert ev0.elapsed_time(ev1) >= 0.0

        gpu_ok, gpu_name = check_supported_gpu(mps)
        assert gpu_ok is True and "Apple" in str(gpu_name)
        drv_ok, drv_info = check_gpu_driver_version()
        assert drv_ok is True and "Metal" in drv_info
        cuda_ok, cuda_info = system_checks.check_cuda()
        assert cuda_ok is True and "MPS" in cuda_info
        vendor, blackwell, fg = system_checks.warm_up_cuda()
        assert vendor is AcceleratorVendor.APPLE and blackwell is False and fg is False

        from jasna.gui.system_stats import read_gpu_vram
        gpu_pct, vram_pct = read_gpu_vram()
        assert gpu_pct is not None and 0 <= gpu_pct <= 100
        assert vram_pct is not None and 0 <= vram_pct <= 100

    def test_yuv_p010_and_nv12_parity(self) -> None:
        from av.video.reformatter import Colorspace as AvColorspace

        mps = torch.device("mps")
        torch.manual_seed(7)
        rgb = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8, device=mps)

        for fmt, max_lsb in (("p010_bt709_limited", 64), ("nv12_bt709_limited", 1)):
            mps_yuv = RgbToYuvConverter(fmt, device=mps).convert(rgb)
            cpu_yuv = RgbToYuvConverter(fmt, device=torch.device("cpu")).convert(rgb.cpu())
            if fmt.startswith("p010"):
                diff = (mps_yuv.cpu().view(torch.uint16).int() - cpu_yuv.view(torch.uint16).int()).abs().max().item()
            else:
                diff = (mps_yuv.cpu().int() - cpu_yuv.int()).abs().max().item()
            assert diff <= max_lsb, f"RgbToYuv mismatch on MPS for {fmt}: max_diff={diff}"

        # Verify P010 bright values are not saturated at 32767
        white = torch.full((3, 16, 16), 255, dtype=torch.uint8, device=mps)
        p010_white = RgbToYuvConverter("p010_bt709_limited", device=mps).convert(white)
        assert p010_white[:16].cpu().view(torch.uint16).int().max().item() == 235 * 4 * 64

        # Verify YuvToRgb 10-bit and 8-bit parity
        p010 = RgbToYuvConverter("p010_bt709_limited", device=torch.device("cpu")).convert(rgb.cpu())
        y10 = p010[:128].view(torch.uint16)
        uv10 = p010[128:].view(torch.uint16).view(64, 64, 2)
        out_mps = torch.empty_like(rgb)
        out_cpu = torch.empty_like(rgb, device="cpu")
        YuvToRgbConverter(128, 128, AvColorspace.ITU709, False, True, mps).convert_into(y10.to(mps), uv10.to(mps), out_mps)
        YuvToRgbConverter(128, 128, AvColorspace.ITU709, False, True, torch.device("cpu")).convert_into(y10, uv10, out_cpu)
        assert torch.equal(out_mps.cpu(), out_cpu)

    def test_sharpen_lut_and_deform_conv2d_on_mps(self) -> None:
        from torchvision.ops import deform_conv2d

        mps = torch.device("mps")
        torch.manual_seed(11)
        frame = torch.randint(0, 256, (3, 96, 128), dtype=torch.uint8, device=mps)

        # Contrast-Adaptive Sharpening parity (8-bit and 10-bit P010)
        luma8 = torch.randint(0, 256, (96, 128), dtype=torch.uint8, device=mps)
        out8_mps, out8_cpu = torch.empty_like(luma8), torch.empty_like(luma8, device="cpu")
        GpuCasSharpener(0.6, ten_bit=False, device=mps).sharpen_into(luma8, out8_mps)
        GpuCasSharpener(0.6, ten_bit=False, device=torch.device("cpu")).sharpen_into(luma8.cpu(), out8_cpu)
        assert (out8_mps.cpu().int() - out8_cpu.int()).abs().max().item() <= 1

        luma10 = (torch.randint(0, 1024, (96, 128), dtype=torch.int32) * 64).to(torch.uint16).view(torch.int16).to(mps)
        out10_mps, out10_cpu = torch.empty_like(luma10), torch.empty_like(luma10, device="cpu")
        GpuCasSharpener(0.6, ten_bit=True, device=mps).sharpen_into(luma10, out10_mps)
        GpuCasSharpener(0.6, ten_bit=True, device=torch.device("cpu")).sharpen_into(luma10.cpu(), out10_cpu)
        assert (
            out10_mps.cpu().view(torch.uint16).int() - out10_cpu.view(torch.uint16).int()
        ).abs().max().item() <= 64

        # 3D LUT parity (5D grid_sample on MPS)
        size = 5
        grid = torch.linspace(0.0, 1.0, size)
        b, g, r = torch.meshgrid(grid, grid, grid, indexing="ij")
        data = torch.stack([r * 0.9, g * 0.95, b * 0.85], dim=0).float()  # (3, B, G, R)
        lut = CubeLut(size=size, is_3d=True, data=data, domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0))
        l_mps = GpuLutApplier(lut, mps).apply(frame)
        l_cpu = GpuLutApplier(lut, torch.device("cpu")).apply(frame.cpu())
        assert (l_mps.cpu().int() - l_cpu.int()).abs().max().item() <= 1

        # BasicVSR++ deform_conv2d on MPS (fp32 and fp16)
        x = torch.randn(1, 8, 16, 16, device=mps)
        offset = torch.randn(1, 18, 16, 16, device=mps) * 0.5
        mask = torch.sigmoid(torch.randn(1, 9, 16, 16, device=mps))
        w = torch.randn(8, 8, 3, 3, device=mps) * 0.1
        y_mps = deform_conv2d(x, offset, w, padding=1, mask=mask)
        y_cpu = deform_conv2d(x.cpu(), offset.cpu(), w.cpu(), padding=1, mask=mask.cpu())
        assert (y_mps.cpu() - y_cpu).abs().max().item() < 1e-4
        y_fp16 = deform_conv2d(x.half(), offset.half(), w.half(), padding=1, mask=mask.half())
        assert torch.isfinite(y_fp16).all()

    @pytest.mark.parametrize(
        ("codec", "is_10bit", "expected_codec"),
        [
            ("h264", False, "h264"),
            ("hevc", True, "hevc"),
            ("av1", True, "av1"),
        ],
    )
    def test_video_encoder_decoder_roundtrip(
        self, tmp_path: Path, codec: str, is_10bit: bool, expected_codec: str
    ) -> None:
        mps = torch.device("mps")
        src = Path(__file__).resolve().parent.parent / "assets" / "test_clip1_1080p.mp4"
        assert src.is_file()
        src_meta = get_video_meta_data(str(src))
        out_path = tmp_path / f"roundtrip_{codec}.mp4"

        enc_meta = VideoMetadata(
            video_file=str(src),
            video_width=320,
            video_height=180,
            video_fps=src_meta.video_fps,
            video_fps_exact=src_meta.video_fps_exact,
            average_fps=src_meta.average_fps,
            duration=0.4,
            time_base=src_meta.time_base,
            start_pts=0,
            num_frames=12,
            codec_name=expected_codec,
            pixel_format="p010le" if is_10bit else "nv12",
            is_10bit=is_10bit,
            color_space=src_meta.color_space,
            color_range=src_meta.color_range,
            color_primaries=src_meta.color_primaries,
        )
        with VideoEncoder(
            str(out_path),
            device=mps,
            metadata=enc_meta,
            codec=codec,
            encoder_settings={},
            output_fps=src_meta.video_fps_exact,
        ) as encoder:
            for idx in range(12):
                frame = torch.full((3, 180, 320), (idx * 20) % 256, dtype=torch.uint8, device=mps)
                encoder.encode(frame, idx)
        synchronize(mps)

        assert out_path.is_file() and out_path.stat().st_size > 0
        out_meta = get_video_meta_data(str(out_path))
        assert out_meta.codec_name == expected_codec
        assert (out_meta.video_width, out_meta.video_height) == (320, 180)
        assert out_meta.is_10bit is is_10bit

        decoded = 0
        with VideoReader(str(out_path), batch_size=4, device=mps, metadata=out_meta) as reader:
            for batch, pts in reader.frames():
                assert batch.device.type == "mps"
                assert batch.shape[1:] == (3, 180, 320)
                decoded += len(pts)
        assert decoded == 12
