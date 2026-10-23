"""
Mission 03 — Part 4 (support script): Security & Ethics Sentinel
KV-Cache Denial-of-Service simulation, FlashAttention IO counts, and the MHA / GQA / MLA
KV-cache comparison used in 04_Memory_Hierarchy_and_Safety_Audit.md.

Pure arithmetic + a small event simulation: no GPU needed.

Run:  python 04_kv_dos_simulation.py      (writes results/04_dos_simulation.json, figures/04_dos_oom.png)
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR, RES_DIR = HERE / "figures", HERE / "results"
GiB = 2**30

# ---------------- Hardware & model (USSE Noosphere mainframe = 1× H100 80 GB) -------------
GPU_HBM_GIB = 80 * 1e9 / GiB           # 80 GB card = 74.5 GiB usable address space
WEIGHTS_GIB = 8.03e9 * 2 / GiB         # LLaMA-3-8B in bf16
ACTIVATION_RESERVE_GIB = 4.0           # workspace for activations / CUDA context (assumption)
FREE_FOR_KV_GIB = GPU_HBM_GIB - WEIGHTS_GIB - ACTIVATION_RESERVE_GIB

LAYERS, HEAD_DIM, BYTES = 32, 128, 2
MAX_LEN = 65_536                       # the context length an attacker requests
BLOCK = 16                             # PagedAttention block size (tokens), vLLM default


def kv_bytes_per_token(n_kv_heads: int, layers: int = LAYERS, head_dim: int = HEAD_DIM) -> int:
    """2 (K and V) · L · H_kv · d_head · bytes."""
    return 2 * layers * n_kv_heads * head_dim * BYTES


# --------------------------------------------------------------------------- #
# 1. FlashAttention IO (elements moved between HBM and on-chip SRAM, one head)
# --------------------------------------------------------------------------- #

def io_standard(n: int, d: int) -> float:
    """Read Q,K (2Nd), write S (N²), read S, write P (2N²), read P,V (N²+Nd), write O (Nd)."""
    return 4 * n * d + 4 * n * n


def io_flash(n: int, d: int, m: int) -> float:
    """Dao et al. 2022, Thm 2: Θ(N² d² / M) plus the unavoidable Θ(N d) to read inputs once.

    Outer loop over T_c = N / B_c key/value blocks with B_c = M / (4d); for each block the
    whole Q and O are streamed through SRAM: T_c · (2Nd) ≈ 8 N² d² / M, plus 4 N d.
    """
    b_c = max(1, m // (4 * d))
    t_c = math.ceil(n / b_c)
    return 4 * n * d + t_c * 2 * n * d


# --------------------------------------------------------------------------- #
# 2. DoS simulation
# --------------------------------------------------------------------------- #

@dataclass
class Policy:
    name: str
    n_kv_heads: int
    reserve_requested: bool           # naive server: pre-allocate the client's requested max_tokens
    quota_tokens: int | None = None   # per-client token budget
    max_sessions_per_client: int | None = None   # admission control


def session_gib(policy: Policy, requested: int, used: int) -> float:
    """KV memory held by one session that asked for `requested` tokens and filled `used`."""
    per_tok = kv_bytes_per_token(policy.n_kv_heads)
    if policy.reserve_requested:
        tokens = requested                                  # contiguous slab, held even if unused
    else:                                                   # PagedAttention: 16-token blocks on demand
        if policy.quota_tokens is not None:
            used = min(used, policy.quota_tokens)
        tokens = math.ceil(used / BLOCK) * BLOCK
    return tokens * per_tok / GiB


def simulate(policy: Policy, attacker_used: int, benign: int = 20, benign_requested: int = 4_096,
             benign_used: int = 2_000, max_attackers: int = 64) -> dict:
    """20 benign users are connected; one attacker client then opens sessions that each request
    65,536 tokens and fill `attacker_used` of them. Returns the session count that causes OOM."""
    base = benign * session_gib(policy, benign_requested, benign_used)
    per = session_gib(policy, MAX_LEN, attacker_used)
    allowed = policy.max_sessions_per_client or max_attackers
    oom = None
    for k in range(1, max_attackers + 1):
        if base + min(k, allowed) * per > FREE_FOR_KV_GIB:
            oom = k
            break
    return {"policy": policy.name, "attacker_tokens_per_session": attacker_used,
            "benign_load_gib": round(base, 2), "attacker_session_gib": round(per, 3),
            "oom_at_attacker_session": oom,
            "attacker_tokens_spent_to_oom": None if oom is None else oom * attacker_used}


def max_full_length_sessions(n_kv_heads: int) -> int:
    return int(FREE_FOR_KV_GIB // (MAX_LEN * kv_bytes_per_token(n_kv_heads) / GiB))


# --------------------------------------------------------------------------- #
# 3. MHA vs GQA vs MLA per-token KV size
# --------------------------------------------------------------------------- #

def kv_table() -> list[dict]:
    """Per-token KV-cache (bf16). MLA numbers use DeepSeek-V3's config:
    61 layers, 128 heads × d_h 128, latent d_c = 512, decoupled RoPE key d_R = 64."""
    rows = []
    rows.append({"model": "LLaMA-3-8B shape, MHA (32 KV heads)", "elements_per_layer": 2 * 32 * 128,
                 "layers": 32})
    rows.append({"model": "LLaMA-3-8B, GQA (8 KV heads)", "elements_per_layer": 2 * 8 * 128, "layers": 32})
    rows.append({"model": "LLaMA-3-8B shape, MQA (1 KV head)", "elements_per_layer": 2 * 1 * 128,
                 "layers": 32})
    rows.append({"model": "DeepSeek-V3 shape, MHA (128 heads)", "elements_per_layer": 2 * 128 * 128,
                 "layers": 61})
    rows.append({"model": "DeepSeek-V3, MLA (d_c 512 + d_R 64)", "elements_per_layer": 512 + 64,
                 "layers": 61})
    for r in rows:
        r["bytes_per_token"] = r["elements_per_layer"] * r["layers"] * BYTES
        r["kib_per_token"] = r["bytes_per_token"] / 1024
        r["gib_at_65k"] = r["bytes_per_token"] * MAX_LEN / GiB
    return rows


def main() -> None:
    print("USSE NOOSPHERE · SUBSYSTEM 03 · PART 4: MEMORY HIERARCHY & DoS AUDIT\n")
    print("=== FlashAttention IO (elements moved HBM <-> SRAM, one head, M ≈ 100K elements) ===")
    m = 100_000   # ~100K fp16 elements ≈ 200 KB, close to an H100 SM's 228 KB shared memory
    print(f"{'N':>7} | {'standard, d=128':>15} | {'flash, d=128':>12} | {'×':>5} | {'standard, d=64':>14} | "
          f"{'flash, d=64':>11} | {'×':>5}")
    io_rows = []
    for n in (1024, 4096, 8192, 16384, 32768, 65536):
        s1, f1 = io_standard(n, 128), io_flash(n, 128, m)
        s2, f2 = io_standard(n, 64), io_flash(n, 64, m)
        io_rows.append({"N": n, "standard_d128": s1, "flash_d128": f1, "ratio_d128": s1 / f1,
                        "standard_d64": s2, "flash_d64": f2, "ratio_d64": s2 / f2})
        print(f"{n:>7} | {s1:>15.2e} | {f1:>12.2e} | {s1 / f1:>4.1f}× | {s2:>14.2e} | {f2:>11.2e} | "
              f"{s2 / f2:>4.1f}×")
    s_mat = 8192 ** 2 * 2 / 2**20
    print(f"(the N×N score matrix alone at N=8192 is {s_mat:.0f} MiB per head; ×32 heads = "
          f"{s_mat * 32 / 1024:.0f} GiB written and read back per layer by standard attention)\n")

    print("=== KV-cache DoS on one H100 80 GB serving LLaMA-3-8B ===")
    print(f"free for KV-cache: {GPU_HBM_GIB:.1f} GiB − weights {WEIGHTS_GIB:.1f} GiB − reserve "
          f"{ACTIVATION_RESERVE_GIB:.1f} GiB = {FREE_FOR_KV_GIB:.1f} GiB")
    for h, name in ((32, "MHA"), (8, "GQA"), (1, "MQA")):
        per = MAX_LEN * kv_bytes_per_token(h) / GiB
        print(f"  {name}: {kv_bytes_per_token(h) / 1024:.0f} KiB/token -> {per:.1f} GiB per 65,536-token "
              f"session -> {max_full_length_sessions(h)} full-length session(s) fit")
    policies = [
        Policy("A. naive, GQA, reserve requested length", 8, True),
        Policy("B. PagedAttention, GQA", 8, False),
        Policy("C. PagedAttention + 16k-token quota per client", 8, False, 16_384),
        Policy("D. PagedAttention + quota + max 2 sessions per client", 8, False, 16_384, 2),
    ]
    sims = []
    for label, used in (("cheap attack: request 65,536 tokens, send only 100", 100),
                        ("expensive attack: actually fill 65,536 tokens", MAX_LEN)):
        print(f"\n20 benign users (request 4,096, use ≈2,000 tokens) connected · {label}")
        print(f"{'policy':<54} | {'benign':>7} | {'per session':>11} | {'OOM at':>12} | {'attacker cost':>13}")
        for p in policies:
            r = simulate(p, used)
            sims.append(r)
            oom = r["oom_at_attacker_session"]
            oom_s = "survives 64" if oom is None else f"session #{oom}"
            cost = "—" if oom is None else f"{r['attacker_tokens_spent_to_oom']:,} tok"
            print(f"{p.name:<54} | {r['benign_load_gib']:>5.1f} G | {r['attacker_session_gib']:>9.2f} G | "
                  f"{oom_s:>12} | {cost:>13}")
    print()
    print("=== KV-cache per token: MHA vs GQA vs MLA (bf16) ===")
    kt = kv_table()
    for r in kt:
        print(f"  {r['model']:<40} {r['kib_per_token']:>9.1f} KiB/token   {r['gib_at_65k']:>7.1f} GiB @ 65k")
    RES_DIR.mkdir(exist_ok=True)
    (RES_DIR / "04_dos_simulation.json").write_text(json.dumps(
        {"free_for_kv_gib": FREE_FOR_KV_GIB, "io": io_rows, "dos": sims, "kv_table": kt}, indent=2))
    print("\nResults written to results/04_dos_simulation.json")
    plot(policies)


def plot(policies: list[Policy]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ks = list(range(0, 31))
    for ax, used, title in ((axes[0], 100, "Cheap attack: request 65k, send 100 tokens"),
                            (axes[1], MAX_LEN, "Expensive attack: fill 65k tokens")):
        for p in policies:
            base = 20 * session_gib(p, 4_096, 2_000)
            allowed = p.max_sessions_per_client or 10**9
            ax.plot(ks, [base + min(k, allowed) * session_gib(p, MAX_LEN, used) for k in ks],
                    "o-", ms=2.5, label=p.name)
        ax.axhline(FREE_FOR_KV_GIB, color="#c0392b", ls="--", lw=1)
        ax.text(0.5, FREE_FOR_KV_GIB + 2, f"CUDA OOM ({FREE_FOR_KV_GIB:.0f} GiB free for KV)",
                color="#c0392b", fontsize=8)
        ax.set_ylim(0, 120); ax.set_xlabel("attacker sessions"); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("KV-cache memory (GiB)")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("KV-cache DoS on one H100 80 GB serving LLaMA-3-8B (GQA)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_dos_oom.png", dpi=150); plt.close(fig)
    print("Figure written to figures/04_dos_oom.png")


if __name__ == "__main__":
    main()
