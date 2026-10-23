# 00 — Squadron Colby: Team and Roles

**Mission:** 03 — Multi-Head Attention, RoPE Vector Rotation & Grouped-Query KV-Cache Optimization  
**Course:** Conversational AI: History, Applications, Future · WiSe 2026/27 · BTU Cottbus-Senftenberg  
**Repository:** https://github.com/xxrafidxx/Squadron-Colby---Conversational-AI

---

## Team

| Name | Squadron station | Mission 03 role | Deliverable |
|---|---|---|---|
| Rafid Mohammad Ahsan | Core & Lead Architect | Part 1 · Systems Architect (Lead Engineer) | `01_MultiHead_Attention_Core.py`, `ps03_solution.ipynb` |
| Rafid Mohammad Ahsan | Protocol & Interface Specialist | Part 2 · Communications & Protocol Officer | `02_RoPE_Positional_Engine.py` |
| Tahjib Al Sakin | Data & Memory Engineer | Part 3 · Data & Payload Specialist | `03_GQA_and_KVCache_Benchmark.py` |
| Kunal Choudhary | Security & Ethical Navigator | Part 4 · Security & Ethics Sentinel | `04_Memory_Hierarchy_and_Safety_Audit.md` |
| Mubina Rahaman Jerin | AI Security Officer | EU AI Act compliance | `06_AI_Act_Compliance.md` |

## Who did what

| Name | Work done |
|---|---|
| **Rafid Mohammad Ahsan** | *Architect:* built `MultiHeadSelfAttention` in pure PyTorch (W_Q, W_K, W_V, W_O, head split/merge, hand-written softmax, causal and padding masks) with a variance-preservation test and a softmax-saturation study; assembled the notebook and repository. *Protocol:* implemented RoPE as complex 2D rotation, verified norm preservation and relative invariance, compared RoPE with ALiBi (curves, heatmaps) and analysed context extrapolation with Position Interpolation. |
| **Tahjib Al Sakin** | Built `GQAAttention` (grouped queries without copying K/V) and a stateful `KVCache`; proved cached decoding equals the full pass; benchmarked MHA, GQA 4:1 and MQA 32:1 at 1k–32k tokens for cache size, bandwidth at 50 tok/s and per-token latency. |
| **Kunal Choudhary** | Derived standard vs FlashAttention HBM traffic; simulated a KV-cache denial-of-service attack on an H100 80 GB serving LLaMA-3-8B, finding the CUDA-OOM point; evaluated PagedAttention and DeepSeek's MLA as defences. |
| **Mubina Rahaman Jerin** | Reviewed the mission against the EU AI Act (AI-system definition, GPAI duties, Art. 15 and Art. 55 robustness and cybersecurity) and listed deployment safeguards. |

**Moodle submission (3 files):** `00_Team_Roles.pdf` · `Mission03_Role_Deliverables_01-04.pdf` (Parts 1–4, one section per role) · `06_AI_Act_Compliance.pdf`. Each role document is also a separate PDF in the repository (`mission-03/pdf/`).
