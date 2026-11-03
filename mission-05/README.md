# Mission 05 — Dual-Path Retrieval, NotebookLM Grounding & RAG Safety

**Squadron Colby** · Conversational AI (WS 2026/27), BTU Cottbus-Senftenberg  
Instructor spec: [Unisvet/couse-c-ai/mission-05](https://github.com/Unisvet/couse-c-ai/tree/main/mission-05) · Waypoint: Crab Pulsar Remnant / Messier 1 (`SEC-05-M01`)  
AURA Subsystem Repair: Subsystem 05: *Dual-Path Associative Memory Core* (+7.1% $\to$ 35.5% total integrity)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xxrafidxx/Squadron-Colby---Conversational-AI/blob/main/mission-05/ps05_solution.ipynb)

---

## Overview

AURA lost access to her archives and started inventing astrogation logs. Subsystem 05 gives her evidence again through two retrieval paths over **Document C** (the *USSE Noosphere Technical Archive Extract*, 36 records), then checks the evidence before she uses it. Pure Python + NumPy, CPU only, no API key. Every number below is measured by the scripts and reproduced in `05_Run_Log.txt`.

- **Part 1 · HNSW index:** our own HNSW in pure Python/NumPy. On 25,000 vectors × 768 dimensions it reaches **recall@10 0.980 at 0.36 ms per query**, checking only 1.3 % of the vectors. Brute force takes 34.8 ms (Python loop) or 3.1 ms (NumPy BLAS). The layer sizes match theory (1,539 nodes on layer ≥ 1 vs 1,562 expected). From N = 5,000 to 25,000 the work per query stays at about 900 distances while brute force grows 5×.
- **Part 2 · BM25 + RRF:** 30 labelled questions. **BM25 wins on exact codes** (Hit@1 0.92), **dense wins on paraphrased questions** (0.58 vs 0.25). Plain RRF with k = 60 lands in between (MRR 0.77 vs dense 0.83): it is robust but does not beat the best single path on this corpus. A code-aware weighted RRF (an extension) reaches MRR 0.79 on the 30 questions and **13/13 Hit@1 on held-out questions** never used for any choice.
- **Part 3 · NotebookLM audit:** lab run on 2026-11-03 in Google's notebook tool (now called *Gemini Notebook*). 98 sentence-citation pairs checked by hovering every badge: **claim coverage 98 %**, **citation precision 0.34 strict / 0.94 lenient**. Most sentences cite several chunks that each back only part of the sentence, and 6 citations fail (wrong chunk, chunk boundary, one contradiction). Trap questions: **2 of 3 clean refusals**, including a well-known fact absent from the sources, plus 1 partial refusal. Audio Overview (24:35, no citations): 4 distorted claims in a 2.5-minute excerpt; scores fidelity 2, personas 4, pedagogy 3 (mean 3.0/5).
- **Part 4 · Safety:**
  - **Injection detector:** catches **25/25 attacks in 12 techniques** with 4 false alarms on 20 hard negatives. The keyword rules alone reach 25/25 with 1 false alarm.
  - **Faithfulness scorer:** **0.90** claim accuracy with the number/code check, vs 0.75 on similarity alone.
  - **Provenance log:** hash-chained, so edits are detected. The poisoned record never reaches the context.
  - **Limit:** similarity scores cannot decide *answerability*; only unknown identifiers are refused reliably.
- **EU AI Act:** review in `06_AI_Act_Compliance.md`.

**Document C:** the brief says it is "provided in course repo", but `Unisvet/couse-c-ai` has no such file (checked 2026-11-02). The squadron wrote it (`corpus/`, Markdown + PDF). The starship content is fictional; the Crab Pulsar facts are real and marked `[real]`.

---

## Repository Layout

| File | Station | Content |
|---|---|---|
| `00_Team_Roles.md` | All | Team members, stations, who did what |
| `aura_archive.py` | All | Loads Document C as 36 records; tokenizer that keeps codes whole; WordLlama dense embedder (LSA fallback); module loader |
| `corpus/USSE_Noosphere_Technical_Archive_Extract.md` / `.pdf` | All | Document C (the PDF is for uploading to NotebookLM) |
| `01_HNSW_Vector_Index.py` | Systems Architect | Cosine core, layer assignment, `search_layer`, neighbour heuristic, k-NN search; 25k × 768 benchmark vs brute force; scaling; archive check |
| `02_BM25_and_RRF_Combiner.py` | Communications & Protocol Officer | Okapi BM25 (k1 = 1.5, b = 0.75), RRF (k = 60), `HybridRetriever`; 30-question benchmark + 13 held-out; k sweep; code-aware RRF extension |
| `03_NotebookLM_RAG_Audit.py` | Data & Payload Specialist | NotebookLM lab protocol, record sheet, citation precision / claim coverage / refusal rate / audio scores, offline baseline, report generator |
| `notebooklm_audit_report.md` | Data & Payload Specialist | Generated report: protocol, all NotebookLM results, citation checks, audio transcript excerpt and scores |
| `notebooklm_lab/lab_record.json` | Data & Payload Specialist | Record sheet to fill in during the lab |
| `04_RAG_Provenance_and_Safety.py` | Security & Ethics Sentinel | Injection detector (rules + embedding), RAG Triad, scorer check, `GroundedAssistant`, hash-chained provenance log |
| `rag_safety_data.py` | Security & Ethics Sentinel | Labelled sets: attack bank, dev and test sets, faithfulness claims, unanswerable questions |
| `06_AI_Act_Compliance.md` | AI Security Officer | EU AI Act review: Art. 50, Art. 15, GPAI duties, data protection |
| `ps05_solution.ipynb` | All | Notebook running all four parts (Colab-ready) |
| `05_Run_Log.txt` | All | Full output of all scripts and tests |
| `figures/`, `results/`, `logs/` | All | Plots, JSON results, provenance audit log |
| `tests/test_mission05.py` | All | 25 offline pytest tests (~3 s) |
| `ps05_briefing.md` | Expedition Command | Official problem statement |

---

## How to Run

Python **3.10–3.13** (WordLlama has no wheels for 3.14 yet). No API key needed.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python 01_HNSW_Vector_Index.py     # ~2-3 min (add --n 5000 for a quick run)
python 02_BM25_and_RRF_Combiner.py
python 04_RAG_Provenance_and_Safety.py
python 03_NotebookLM_RAG_Audit.py  # scores the lab record and rewrites notebooklm_audit_report.md
pytest -q                          # 25 tests
```

**Dense embeddings:** WordLlama `l2_supercat` (256-d static embeddings distilled from LLaMA-2). Its weights ship inside the pip package, so nothing is downloaded at run time. If it is not installed, `aura_archive.py` falls back to TF-IDF + SVD (LSA) and says so; `M05_EMBEDDER=lsa` forces the fallback. With LSA the dense path gets only 0.17 Hit@1 on paraphrased questions (WordLlama: 0.58), which shows why a pretrained embedder matters.

### Part 3 lab (Data & Payload Specialist)
1. Download Document A (https://arxiv.org/pdf/2005.11401) and Document B (https://arxiv.org/pdf/2404.16130). Upload them, together with `corpus/USSE_Noosphere_Technical_Archive_Extract.pdf`, to a new NotebookLM notebook.
2. Follow `notebooklm_audit_report.md` §1. Paste the responses and your citation checks into `notebooklm_lab/lab_record.json`.
3. Run `python 03_NotebookLM_RAG_Audit.py`. The report and `results/03_notebooklm_audit.json` are regenerated with the real numbers.

---

## Subsystem 05 Sign-Off Status

| Deliverable | Station Lead | Measured result | Status |
|---|---|---|:---:|
| **Part 1: HNSW** | Rafid Mohammad Ahsan | 25k × 768: recall@10 0.980 at 0.36 ms (1.3 % of distances) vs 34.8 ms Python / 3.1 ms NumPy flat; layer sizes match N·M⁻ˡ | Ready for sign-off |
| **Part 2: BM25 + RRF** | Rafid Mohammad Ahsan | Hit@1 exact codes BM25 0.92 / dense 0.83; paraphrases dense 0.58 / BM25 0.25; plain RRF MRR 0.77; code-aware RRF 13/13 on held-out | Ready for sign-off |
| **Part 3: NotebookLM audit** | Tahjib Al Sakin | coverage 98 %; citation precision 0.34 strict / 0.94 lenient (98 citations); refusals 2/3 clean + 1 partial; audio 3.0/5 (fidelity 2) | Ready for sign-off |
| **Part 4: Safety & RAG Triad** | Kunal Choudhary | detector recall 25/25, precision 0.86 (rules alone 0.96); faithfulness scorer 0.90; tampering detected; poison quarantined | Ready for sign-off |
| **Compliance: EU AI Act** | Mubina Rahaman Jerin | Art. 50 disclosure + marking, Art. 15 robustness, GDPR, provenance | Ready for sign-off |

**Limits:**
- Document C is small (36 records).
- The question sets are hand-written: 30 benchmark, 13 held-out, 45 injection-test chunks and 20 test claims.
- The HNSW benchmark uses synthetic embedding-like vectors (real 25k-document embeddings were not available offline).
- Answers are extractive, not generated by an LLM.
