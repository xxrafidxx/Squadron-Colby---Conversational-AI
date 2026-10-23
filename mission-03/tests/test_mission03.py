"""Offline tests for Mission 03 (CPU only, a few seconds)."""
import importlib.util
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mha = load("mha", "01_MultiHead_Attention_Core.py")
rope = load("rope", "02_RoPE_Positional_Engine.py")
gqa = load("gqa", "03_GQA_and_KVCache_Benchmark.py")
dos = load("dos", "04_kv_dos_simulation.py")


# ---------- Part 1: multi-head self-attention --------------------------------

def test_output_and_attention_shapes():
    m = mha.MultiHeadSelfAttention(d_model=64, num_heads=4)
    out, w = m(torch.randn(3, 7, 64))
    assert out.shape == (3, 7, 64) and w.shape == (3, 4, 7, 7)


def test_attention_rows_sum_to_one():
    _, w = mha.MultiHeadSelfAttention(32, 4)(torch.randn(2, 5, 32))
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 5), atol=1e-5)


def test_causal_mask_blocks_future():
    torch.manual_seed(0)
    m = mha.MultiHeadSelfAttention(32, 4).eval()
    x = torch.randn(1, 8, 32)
    y1, w = m(x)
    x[:, 5:] = torch.randn(1, 3, 32)
    y2, _ = m(x)
    assert torch.triu(w[0, 0], 1).abs().max() == 0
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6)


def test_matches_pytorch_reference():
    """Our hand-written attention equals torch's fused kernel (used only as a reference)."""
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 4, 9, 16) for _ in range(3))
    ours, _ = mha.scaled_dot_product_attention(q, k, v, mha.causal_mask(9))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_variance_preservation():
    for d in (16, 64, 256):
        raw, scaled = mha.dot_product_variance(d)
        assert abs(raw / d - 1) < 0.1 and abs(scaled - 1) < 0.1


def test_padding_mask_ignores_padded_keys():
    m = mha.MultiHeadSelfAttention(32, 4).eval()
    pad = torch.tensor([[True, True, True, False, False]])
    _, w = m(torch.randn(1, 5, 32), causal=False, padding_mask=pad)
    assert w[..., 3:].abs().max() == 0


def test_rejects_bad_head_count():
    try:
        mha.MultiHeadSelfAttention(30, 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------- Part 2: RoPE ------------------------------------------------------

def test_rope_preserves_norm():
    assert rope.verify_rotation_is_orthogonal() < 1e-4


def test_rope_relative_invariance():
    assert rope.verify_relative_invariance(trials=50) < 1e-4


def test_rope_position_zero_is_identity():
    x = torch.randn(1, 1, 1, 32)
    assert torch.allclose(rope.apply_rope(x, rope.precompute_freqs_cis(32, 1)), x, atol=1e-6)


def test_rope_matches_rotation_matrix():
    """Complex multiplication == explicit 2×2 rotation of each pair."""
    d, m = 8, 5
    x = torch.randn(d)
    theta = rope.rope_frequencies(d).float()
    out = rope.apply_rope(x[None], rope.precompute_freqs_cis(d, m + 1)[m:m + 1])[0]
    ref = torch.empty(d)
    for i in range(d // 2):
        c, s = math.cos(m * theta[i]), math.sin(m * theta[i])
        ref[2 * i] = c * x[2 * i] - s * x[2 * i + 1]
        ref[2 * i + 1] = s * x[2 * i] + c * x[2 * i + 1]
    assert torch.allclose(out, ref, atol=1e-5)


def test_rope_decays_with_distance():
    s = rope.rope_distance_curve(128, 1024)
    assert s[0] == 1 and s[16] < s[1] and s[256] < s[16]


def test_position_interpolation_keeps_angles_in_range():
    r = rope.extrapolation_report(train_len=4096, target_len=32768)
    assert abs(r["slowest_plane_angle_with_PI_rad"] - r["slowest_plane_angle_train_rad"]) < 1e-3


# ---------- Part 3: GQA & KV-cache -------------------------------------------

def test_cached_decoding_equals_full_pass():
    assert gqa.check_cache_matches_full_pass() < 1e-4


def test_grouped_view_equals_repeat_interleave():
    assert gqa.check_view_equals_repeat() < 1e-5


def test_gqa_with_all_kv_heads_equals_mha():
    assert gqa.check_mha_special_case() < 1e-4


def test_mqa_runs():
    layer = gqa.GQAAttention(64, 8, 1)
    assert layer(torch.randn(2, 5, 64)).shape == (2, 5, 64)


def test_kv_cache_overflow_is_caught():
    cache = gqa.KVCache(1, 4, 2, 8)
    try:
        cache.update(torch.zeros(1, 2, 5, 8), torch.zeros(1, 2, 5, 8), 0)
    except RuntimeError:
        return
    raise AssertionError("expected overflow error")


def test_kv_bytes_formula_and_gqa_saving():
    assert gqa.kv_bytes(1, 32) == 2 * 32 * 32 * 128 * 2          # 512 KiB per token, MHA
    assert gqa.kv_bytes(4096, 8) / gqa.kv_bytes(4096, 32) == 0.25


def test_benchmark_table_without_timing():
    rows = gqa.run_benchmark(measure=False)
    assert len(rows) == 12
    mha32k = next(r for r in rows if r.kv_heads == 32 and r.seq_len == 32768)
    assert round(mha32k.kv_cache_mb) == 16384


# ---------- Part 4: DoS simulation --------------------------------------------

def test_naive_server_crashes_cheaply():
    p = dos.Policy("naive", 8, True)
    r = dos.simulate(p, attacker_used=100)
    assert r["oom_at_attacker_session"] == 6


def test_paging_with_quota_and_cap_survives():
    p = dos.Policy("paged+", 8, False, 16_384, 2)
    assert dos.simulate(p, attacker_used=dos.MAX_LEN)["oom_at_attacker_session"] is None


def test_flash_moves_less_than_standard():
    for n in (4096, 32768):
        assert dos.io_flash(n, 128, 100_000) < dos.io_standard(n, 128)


def test_mla_smaller_than_gqa():
    t = {r["model"]: r["kib_per_token"] for r in dos.kv_table()}
    assert t["DeepSeek-V3, MLA (d_c 512 + d_R 64)"] < t["LLaMA-3-8B, GQA (8 KV heads)"]
