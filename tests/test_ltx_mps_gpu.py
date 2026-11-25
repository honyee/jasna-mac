from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from jasna.accelerator import has_mps, synchronize
from jasna.ltx import mps_kernels as mk
from jasna.ltx import transformer as T
from jasna.ltx.blocks import BlockStore, _layout
from jasna.ltx.sampler import denoise_windows
from jasna.models.ltx_vae import eager_na, na_mps
from jasna.models.ltx_vae.diffusion_video_decoder import DiffusionVideoDecoder
from jasna.models.ltx_vae.transformer import decoder_kernels, use_kernels
from jasna.models.ltx_vae.video_encoder import VideoEncoder

pytestmark = pytest.mark.skipif(not has_mps(), reason="requires Apple Silicon MPS")


def _make_convrot_block_weights(dim: int, heads: int, ff_dim: int, seed: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w_q: dict[str, torch.Tensor] = {
        "scale_shift_table": torch.randn(9, dim, dtype=torch.bfloat16, generator=gen) * 0.05,
        "prompt_scale_shift_table": torch.randn(2, dim, dtype=torch.bfloat16, generator=gen) * 0.05,
        "attn1.q_norm.weight": torch.ones(dim, dtype=torch.bfloat16),
        "attn1.k_norm.weight": torch.ones(dim, dtype=torch.bfloat16),
        "attn2.q_norm.weight": torch.ones(dim, dtype=torch.bfloat16),
        "attn2.k_norm.weight": torch.ones(dim, dtype=torch.bfloat16),
    }
    w_ref = dict(w_q)
    shapes = [
        ("attn1.to_q", (dim, dim)),
        ("attn1.to_k", (dim, dim)),
        ("attn1.to_v", (dim, dim)),
        ("attn1.to_out.0", (dim, dim)),
        ("attn1.to_gate_logits", (heads, dim)),
        ("attn2.to_q", (dim, dim)),
        ("attn2.to_k", (dim, dim)),
        ("attn2.to_v", (dim, dim)),
        ("attn2.to_out.0", (dim, dim)),
        ("attn2.to_gate_logits", (heads, dim)),
        ("ff.net.0.proj", (ff_dim, dim)),
        ("ff.net.2", (dim, ff_dim)),
    ]
    for name, (out_f, in_f) in shapes:
        wf = torch.randn(out_f, in_f, dtype=torch.float32, generator=gen) * (in_f**-0.5)
        w_rot = T.rotate(wf)
        scale = (w_rot.abs().amax(dim=1) / 127.0).clamp(min=1e-6)
        qdata = torch.round(w_rot / scale[:, None]).clamp(-127, 127).to(torch.int8)
        bias = torch.randn(out_f, dtype=torch.bfloat16, generator=gen) * 0.01
        w_q[f"{name}.qdata"] = qdata
        w_q[f"{name}.scales"] = scale
        w_q[f"{name}.bias"] = bias
        w_ref[f"{name}.weight"] = T.rotate(qdata.float() * scale[:, None]).to(torch.bfloat16)
        w_ref[f"{name}.bias"] = bias
    return w_q, w_ref


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ltx_metal_kernel_primitives_parity(dtype: torch.dtype) -> None:
    mps = torch.device("mps")
    torch.manual_seed(123)

    # 1. dequant_int8
    qdata = torch.randint(-127, 127, (512, 1024), dtype=torch.int8, device=mps)
    scales = torch.rand(512, dtype=torch.float32, device=mps) * 0.01 + 1e-4
    dq = mk.dequant_int8(qdata, scales, dtype)
    dq_ref = (qdata.float() * scales[:, None]).to(dtype)
    assert torch.equal(dq, dq_ref)

    # 2. hadamard_256 (out-of-place and in-place)
    x = torch.randn(64, 512, dtype=dtype, device=mps)
    h_out = mk.hadamard_256(x)
    h_ref = T.rotate(x.float()).to(dtype)
    assert (h_out.float() - h_ref.float()).norm() / h_ref.float().norm() < 1e-3
    x_copy = x.clone()
    h_ip = mk.hadamard_256(x_copy, inplace=True)
    assert h_ip.data_ptr() == x_copy.data_ptr()
    assert torch.equal(h_ip, h_out)

    # 3. modulate_rotate
    xm = torch.randn(1, 64, 512, dtype=dtype, device=mps)
    sh = torch.randn(1, 2, 512, dtype=dtype, device=mps) * 0.1
    sc = torch.randn(1, 2, 512, dtype=dtype, device=mps) * 0.1
    m_rot = mk.modulate_rotate(xm, sh, sc)
    m_ref = T.rotate(T._modulate(T.rms_norm(xm), sh, sc).float()).to(dtype)[0]
    assert (m_rot.x.float() - m_ref.float()).norm() / m_ref.float().norm() < 6e-3

    # 4. gated_heads_rotate
    heads = torch.randn(4, 64, 128, dtype=dtype, device=mps)
    logits = torch.randn(64, 4, dtype=dtype, device=mps)
    g_rot = mk.gated_heads_rotate(heads, logits)
    gates = 2.0 * torch.sigmoid(logits.float())
    g_ref = T.rotate((heads.float().transpose(0, 1) * gates.unsqueeze(-1)).reshape(64, 512)).to(dtype)
    assert (g_rot.x.float() - g_ref.float()).norm() / g_ref.float().norm() < 2e-3

    # 5. gelu_hadamard_256 (in-place)
    hf = torch.randn(64, 1024, dtype=dtype, device=mps)
    gf_ref = T.rotate(F.gelu(hf.float(), approximate="tanh")).to(dtype)
    gf_out = mk.gelu_hadamard_256(hf, inplace=True)
    assert gf_out.data_ptr() == hf.data_ptr()
    assert (gf_out.float() - gf_ref.float()).norm() / gf_ref.float().norm() < 2e-3


def test_ltx_transformer_block_parity_on_mps() -> None:
    mps = torch.device("mps")
    T.require_kernels(mps)
    assert T.enable_quant_kernels(mps) == "Metal (MPS)"
    T.enable_int8_attention(mps)  # no-op on MPS

    dim, heads, ff_dim, n = 512, 4, 2048, 32
    w_q, w_ref = _make_convrot_block_weights(dim, heads, ff_dim, seed=7)
    w_mps = {k: v.to(mps) for k, v in w_q.items()}

    torch.manual_seed(99)
    x_cpu = torch.randn(1, 2 * n, dim, dtype=torch.bfloat16)
    ctx_cpu = torch.randn(1, 16, dim, dtype=torch.bfloat16)
    cond_cpu = T.StepConditioning(
        block_rows=torch.randn(1, 2, 9, dim, dtype=torch.bfloat16) * 0.05,
        output_rows=torch.randn(1, 2, dim, dtype=torch.bfloat16) * 0.05,
        prompt_rows=torch.randn(1, 1, 2, dim, dtype=torch.bfloat16) * 0.05,
    )
    rope_cpu = T.build_rope(2, 4, 4, dim=dim, heads=heads, device=torch.device("cpu"))
    cond_mps = T.StepConditioning(
        cond_cpu.block_rows.to(mps), cond_cpu.output_rows.to(mps), cond_cpu.prompt_rows.to(mps)
    )
    rope_mps = T.Rope(rope_cpu.cos.to(mps), rope_cpu.sin.to(mps))

    for skip_sa in (False, True):
        pkv_ref = T.prompt_key_value(w_ref, cond_cpu, ctx_cpu, heads=heads)
        out_ref = T.block_forward(w_ref, x_cpu, cond_cpu, pkv_ref, rope_cpu, heads=heads, skip_self_attention=skip_sa)

        pkv_mps = T.prompt_key_value(w_mps, cond_mps, ctx_cpu.to(mps), heads=heads)
        out_mps = T.block_forward(
            w_mps, x_cpu.to(mps), cond_mps, pkv_mps, rope_mps, heads=heads, skip_self_attention=skip_sa
        )
        synchronize(mps)
        rel = (out_mps.cpu().float() - out_ref.float()).norm() / out_ref.float().norm()
        assert torch.isfinite(out_mps).all()
        assert rel < 0.02


@pytest.mark.parametrize("resident", [4, 1, 0])
def test_ltx_block_store_resident_and_streamed_on_mps(resident: int) -> None:
    mps = torch.device("mps")
    specs = {"w": (torch.int8, (64, 64)), "s": (torch.float32, (64,))}
    layouts = [_layout(specs) for _ in range(4)]
    store = BlockStore(layouts, mps, resident=resident)
    try:
        for i in range(4):
            buf = store._buffer(i)
            buf.fill_(i + 1)
        for _cycle in range(2):
            for i in range(4):
                views = store.acquire(i)
                assert views["w"].device.type == "mps"
                assert int(views["w"][0, 0].item()) == i + 1
                store.release(i)
    finally:
        store.close()


def test_ltx_denoise_windows_reproducible_on_mps() -> None:
    mps = torch.device("mps")
    refs = [torch.randn(1, 16, 16, 2, 2, dtype=torch.bfloat16) for _ in range(2)]
    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)

    def fake_denoise(inputs: list[torch.Tensor], step: int) -> list[torch.Tensor]:
        return [(x[:, : x.shape[1] // 2] * 0.9 + 0.01 * step) for x in inputs]

    out1 = denoise_windows(
        refs, [42, 43], sigmas, window_hop=48, steps=2, group=2, denoise=fake_denoise, advance=lambda _: None, device=mps
    )
    out2 = denoise_windows(
        refs, [42, 43], sigmas, window_hop=48, steps=2, group=2, denoise=fake_denoise, advance=lambda _: None, device=mps
    )
    assert len(out1) == 2
    for a, b in zip(out1, out2):
        assert a.shape == refs[0].shape and torch.isfinite(a).all()
        assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("q_shape", "k_shape", "kernel_size", "w_start"),
    [
        ((1, 5, 7, 3, 4, 64), (1, 5, 7, 3, 4, 64), (3, 5, 5), 0),
        ((1, 6, 9, 11, 4, 64), (1, 6, 9, 19, 4, 64), (5, 7, 9), 4),
        ((1, 4, 4, 2, 2, 64), (1, 4, 4, 6, 2, 64), (9, 9, 9), 2),
        ((1, 12, 16, 16, 4, 64), (1, 12, 16, 16, 4, 64), (3, 7, 7), 0),
    ],
)
def test_metal_na3d_parity(
    dtype: torch.dtype,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    kernel_size: tuple[int, int, int],
    w_start: int,
) -> None:
    mps = torch.device("mps")
    assert decoder_kernels(mps).na3d is na_mps.na3d

    torch.manual_seed(19)
    q = torch.randn(*q_shape, dtype=dtype, device=mps) * (64**-0.5)
    k = torch.randn(*k_shape, dtype=dtype, device=mps)
    v = torch.randn(*k_shape, dtype=dtype, device=mps)
    out = na_mps.na3d(q, k, v, kernel_size, w_start)
    ref = eager_na.na3d(q, k, v, kernel_size, w_start)
    synchronize(mps)
    assert out.shape == q.shape and out.dtype == dtype
    assert (out.float() - ref.float()).abs().max().item() < 0.02
    assert (out.float() - ref.float()).abs().mean().item() < 5e-4


def test_metal_na3d_fallback_non_hd64() -> None:
    mps = torch.device("mps")
    q = torch.randn(1, 4, 6, 6, 2, 32, dtype=torch.bfloat16, device=mps) * (32**-0.5)
    k = torch.randn(1, 4, 6, 6, 2, 32, dtype=torch.bfloat16, device=mps)
    v = torch.randn(1, 4, 6, 6, 2, 32, dtype=torch.bfloat16, device=mps)
    out = na_mps.na3d(q, k, v, (3, 3, 3), 0)
    ref = eager_na.na3d(q, k, v, (3, 3, 3), 0)
    assert torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3)


def test_ltx_vae_encoder_and_decoder_small_mps() -> None:
    mps = torch.device("mps")
    torch.manual_seed(5)

    # Small VideoEncoder forward on MPS (contiguous NCDHW)
    enc = VideoEncoder(
        in_channels=3,
        out_channels=16,
        encoder_blocks=[("res_x", {"num_layers": 1}), ("compress_all_res", {"multiplier": 2})],
        patch_size=4,
    ).to(device=mps, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 3, 9, 32, 32, dtype=torch.bfloat16, device=mps)
    with torch.inference_mode():
        z = enc(x)
    assert z.device.type == "mps" and z.dtype == torch.bfloat16 and torch.isfinite(z).all()

    # Small DiffusionVideoDecoder with Metal NA3D kernels on MPS vs CPU reference
    dec_cpu = DiffusionVideoDecoder(
        in_channels=16,
        out_channels=3,
        patch_size=2,
        head_dim=64,
        stage_channels=(64, 64, 64, 64, 64),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        upsamples=(((1, 2, 2), 1), ((2, 2, 2), 1), ((2, 2, 2), 1), ((2, 2, 2), 1)),
        stage5_kernel=(3, 3, 3),
        stage5_channels=64,
        t_emb_dim=64,
        timestep_scale_multiplier=1000.0,
    ).to(dtype=torch.bfloat16).eval()
    dec_mps = DiffusionVideoDecoder(
        in_channels=16,
        out_channels=3,
        patch_size=2,
        head_dim=64,
        stage_channels=(64, 64, 64, 64, 64),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        upsamples=(((1, 2, 2), 1), ((2, 2, 2), 1), ((2, 2, 2), 1), ((2, 2, 2), 1)),
        stage5_kernel=(3, 3, 3),
        stage5_channels=64,
        t_emb_dim=64,
        timestep_scale_multiplier=1000.0,
    ).to(device=mps, dtype=torch.bfloat16).eval()
    dec_mps.load_state_dict({k: v.to(mps) for k, v in dec_cpu.state_dict().items()})
    use_kernels(dec_mps, decoder_kernels(mps))

    z_in = torch.randn(1, 16, 3, 4, 4, dtype=torch.bfloat16)
    with torch.inference_mode():
        tiling = dec_mps.recommended_tiling_config(height=128, width=128, num_frames=17, free_bytes=8 << 30)
        gen_cpu = torch.Generator(device="cpu").manual_seed(7)
        gen_mps = torch.Generator(device="cpu").manual_seed(7)  # identical noise for parity check
        out_cpu = torch.cat(list(dec_cpu.tiled_decode(z_in, tiling, gen_cpu)), dim=2)
        out_mps = torch.cat(list(dec_mps.tiled_decode(z_in.to(mps), tiling, gen_mps)), dim=2).cpu()
    rel = (out_mps.float() - out_cpu.float()).norm() / out_cpu.float().norm()
    assert out_mps.shape == out_cpu.shape and torch.isfinite(out_mps).all()
    assert rel < 0.05


def test_mps_concurrent_index_select_and_video_encoder_no_deadlock(tmp_path) -> None:
    """Regression test for PyTorch MPS THPEvent_record + index_select AB-BA GIL deadlock.

    During LTX compose, LtxDecode executes DiffVAE ``index_select`` on MPS concurrently with
    ``VideoEncoder`` and ``StageBar`` timing marks on other threads.
    """
    import threading
    from fractions import Fraction
    from pathlib import Path
    from av.video.reformatter import ColorRange as AvColorRange, Colorspace as AvColorspace
    from jasna.accelerator import new_event
    from jasna.ltx.progress import Progress, timing_marks
    from jasna.media.probe import VideoMetadata
    from jasna.media.video_encoder import VideoEncoder as MediaVideoEncoder

    mps = torch.device("mps")
    stop = threading.Event()
    errors: list[BaseException] = []

    def hammer_index_select() -> None:
        try:
            x = torch.randn(1, 32, 64, 64, device=mps, dtype=torch.float16)
            idx = torch.arange(32, device=mps, dtype=torch.long)
            while not stop.is_set():
                y = torch.index_select(x, 1, idx)
                _ = y.sum()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def hammer_events_and_timing() -> None:
        try:
            progress = Progress(40, disable=True, writer_adds_frame_work=False, report=None)
            with progress.bar("compose", 40, marks=timing_marks(object())) as bar:
                for _ in range(40):
                    if stop.is_set():
                        break
                    ev = new_event(mps, enable_timing=False)
                    ev.record()
                    with bar.window(512):
                        bar.update(1)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    src = Path(__file__).resolve().parent.parent / "assets" / "test_clip1_1080p.mp4"
    out_path = tmp_path / "concurrent_mps.mp4"
    meta = VideoMetadata(
        video_file=str(src),
        video_width=320,
        video_height=180,
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
    )

    t_idx = threading.Thread(target=hammer_index_select, daemon=True)
    t_evt = threading.Thread(target=hammer_events_and_timing, daemon=True)
    t_idx.start()
    t_evt.start()
    try:
        with MediaVideoEncoder(
            str(out_path),
            device=mps,
            metadata=meta,
            codec="hevc",
            encoder_settings={},
            output_fps=Fraction(30, 1),
        ) as enc:
            for i in range(60):
                frame = torch.full((3, 180, 320), (i * 4) % 256, dtype=torch.uint8, device=mps)
                enc.encode(frame, i)
    finally:
        stop.set()
        t_idx.join(timeout=15.0)
        t_evt.join(timeout=15.0)

    assert not t_idx.is_alive() and not t_evt.is_alive(), "Concurrent MPS threads deadlocked!"
    assert not errors, f"Worker error: {errors}"
    assert out_path.is_file() and out_path.stat().st_size > 0


def test_mps_concurrent_mm_and_empty_cache_no_uaf_crash() -> None:
    """Regression test for PyTorch MPS MPSGraphCache::clear() use-after-free crash.

    Calling ``torch.mps.empty_cache()`` concurrently with MPSGraph ops on another thread
    (e.g. ``LtxDecode`` vs ``composite_on_torch`` / ``structured_mm_out_mps``) deletes
    cached ``MPSCachedGraph*`` objects in use and aborts with ``NSInvalidArgumentException``.
    Inside ``mps_concurrent_scope``, ``empty_cache()`` must defer safely until scope exit.
    """
    import threading
    import jasna.accelerator as acc
    from jasna.accelerator import empty_cache, mps_concurrent_scope
    from jasna.ltx.camera import CropFrame
    from jasna.ltx.compose import Candidate, composite_on_torch, feather_alpha

    mps = torch.device("mps")
    stop = threading.Event()
    errors: list[BaseException] = []

    def flusher() -> None:
        try:
            while not stop.is_set():
                empty_cache(mps)
                empty_cache()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    a = torch.randn((100, 142), device=mps)
    b = torch.randn((256, 256), device=mps)[:142, :100]  # non-contiguous -> MPSGraph mm
    src = torch.randint(0, 255, (3, 360, 640), dtype=torch.uint8, device=mps)
    canvas = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8, device=mps)

    with mps_concurrent_scope(mps):
        assert acc._mps_concurrent_depth == 1
        t = threading.Thread(target=flusher, daemon=True)
        t.start()
        try:
            for i in range(600):
                _ = a @ b
                if i < 40:
                    w = 100 + i * 3
                    h = 90 + i * 2
                    poly = [(50, 50), (50 + w, 50), (50 + w, 50 + h), (50, 50 + h)]
                    crop = CropFrame(crop_box=(30, 30, 70 + w, 70 + h), resize_hw=(256, 256), pad=(0, 0, 0, 0))
                    cand = Candidate(canvas=canvas, crop=crop, weight=1.0, track=1, polygons=[poly], presence=1.0)
                    _ = composite_on_torch(src, [cand], feather=12, projector=None)
        finally:
            stop.set()
            t.join(timeout=10.0)
        assert acc._mps_empty_cache_deferred is True

    assert acc._mps_concurrent_depth == 0
    assert acc._mps_empty_cache_deferred is False
    assert not errors, f"Worker error: {errors}"

    # Verify MPS separable conv2d feather_alpha parity with CPU reference & empty mask handling.
    for shape, feather in (((120, 200), 12), ((64, 65), 6), ((200, 300), 21)):
        mask_cpu = torch.zeros(shape, dtype=torch.bool)
        mask_cpu[shape[0] // 4 : shape[0] // 2, shape[1] // 4 : shape[1] // 2] = True
        box = (shape[1] // 4, shape[0] // 4, shape[1] // 2, shape[0] // 2)
        ref = feather_alpha(mask_cpu, feather, box)
        got = feather_alpha(mask_cpu.to(mps), feather, box).cpu()
        assert (got - ref).abs().max().item() < 1e-5
    empty_got = feather_alpha(torch.zeros((0, 0), dtype=torch.bool, device=mps), 6, (0, 0, 0, 0))
    assert empty_got.shape == (0, 0)
