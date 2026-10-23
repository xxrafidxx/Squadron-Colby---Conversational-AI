# Mission 03 — Multi-Head Attention, RoPE Vector Rotation & Grouped-Query KV-Cache

**Squadron Colby** · Conversational AI (WS 2026/27), BTU Cottbus-Senftenberg
Instructor spec: [Unisvet/couse-c-ai/mission-03](https://github.com/Unisvet/couse-c-ai/tree/main/mission-03) · Waypoint: Pleiades Open Cluster (SEC-03-M45)

AURA's Subsystem 03 rebuilt from first principles in pure PyTorch:

- **Scaled dot-product & multi-head self-attention** with causal masking and a variance-preservation test
- **Rotary positional embeddings (RoPE)** with relative-invariance proof, RoPE vs ALiBi decay and context extrapolation
- **Grouped-query attention + KV-cache** with an MHA / GQA / MQA memory, bandwidth and latency benchmark
- **FlashAttention IO analysis + KV-cache denial-of-service audit** with PagedAttention and MLA mitigations

## Repository layout

| File | Station | Content |
|---|---|---|
| `00_Team_Roles.md` | All | Team members, roles and who did what |
| `01_MultiHead_Attention_Core.py` | Systems Architect | `MultiHeadSelfAttention`, hand-written softmax, causal & padding masks, variance and saturation tests |
| `02_RoPE_Positional_Engine.py` | Communications & Protocol Officer | `apply_rope`, relative invariance check, RoPE vs ALiBi curves and heatmaps, Position Interpolation |
| `03_GQA_and_KVCache_Benchmark.py` | Data & Payload Specialist | `GQAAttention`, `KVCache`, exactness checks, MHA/GQA/MQA benchmark table |
| `04_Memory_Hierarchy_and_Safety_Audit.md` | Security & Ethics Sentinel | FlashAttention IO derivation, KV-cache DoS simulation, PagedAttention and MLA |
| `04_kv_dos_simulation.py` | Security & Ethics Sentinel | Numbers behind the audit (IO counts, OOM simulation, KV size table) |
| `06_AI_Act_Compliance.md` | AI Security Officer | EU AI Act classification and robustness/cybersecurity review |
| `ps03_solution.ipynb` | All | All four parts, executed, with figures |
| `05_Run_Log.txt` | All | Console output of every script and the tests |
| `figures/`, `results/` | All | Plots and JSON results written by the scripts |
| `tests/test_mission03.py` | All | 24 offline tests |
| `pdf/` | All | Submission PDFs |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python 01_MultiHead_Attention_Core.py
python 02_RoPE_Positional_Engine.py
python 03_GQA_and_KVCache_Benchmark.py      # add --quick for faster timing runs
python 04_kv_dos_simulation.py
pytest -q                                   # 24 tests, a few seconds
```

Everything runs on a laptop CPU; no GPU or API key is needed. GPU numbers in the tables (H100) are computed from vendor bandwidth specs and are marked as such.

## Key results

| Part | Result |
|---|---|
| 1 · Attention | Var(q·k) = 64.2 for d_k = 64; after scaling 1.003. Without scaling the softmax drifts to one-hot (largest weight 0.92 at d_k = 512). Causal mask verified: editing future tokens leaves earlier outputs unchanged. |
| 2 · RoPE | Norm preserved (error 1e-6); relative invariance holds over 200 random cases (max relative error 1.7e-6). RoPE decays smoothly and stays bounded; ALiBi falls linearly. Trained at 4k, 18 of 64 planes see unseen angles at 32k; Position Interpolation fixes that. |
| 3 · GQA / KV-cache | Cached decoding equals the full pass (error 5e-7) and is about 9–14× faster for 256 tokens (CPU timings vary run to run). GQA 4:1 cuts the cache 75%: 16 GiB → 4 GiB at N = 32,768. MHA at 32k needs 859 GB/s at 50 tok/s just to read the cache. |
| 4 · Audit | Naive server crashes at the 6th malicious session for 600 tokens; PagedAttention + per-client quota + session cap survives 64. MLA cache is 68.6 KiB/token vs 3,904 KiB for an equivalent MHA. |

## Squadron sign-off

| Station | Officer | Deliverable | Signed |
|---|---|---|---|
| Systems Architect | Rafid Mohammad Ahsan | `01_MultiHead_Attention_Core.py`, `ps03_solution.ipynb`, repo layout | [ ] |
| Communications & Protocol Officer | Rafid Mohammad Ahsan | `02_RoPE_Positional_Engine.py` | [ ] |
| Data & Payload Specialist | Tahjib Al Sakin | `03_GQA_and_KVCache_Benchmark.py` | [ ] |
| Security & Ethics Sentinel | Kunal Choudhary | `04_Memory_Hierarchy_and_Safety_Audit.md`, `04_kv_dos_simulation.py` | [ ] |
| AI Security Officer | Mubina Rahaman Jerin | `06_AI_Act_Compliance.md` | [ ] |

Commit message required by the briefing: `feat(ps03): restore AURA Subsystem 03 Spatial Attention Mesh (SEC-03-M45)`
