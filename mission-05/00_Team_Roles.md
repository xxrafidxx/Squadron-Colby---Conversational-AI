# 00 — Squadron Colby: Team and Roles

**Mission:** 05 — Dual-Path GraphRAG, Hybrid Search & The Google NotebookLM Grounding Lab  
**Course:** Conversational AI: History, Applications, Future · WiSe 2026/27 · BTU Cottbus-Senftenberg  
**Repository:** https://github.com/xxrafidxx/Squadron-Colby---Conversational-AI  
**Waypoint:** Crab Pulsar Remnant / Messier 1 (`SEC-05-M01`, 6,500 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 05: *Dual-Path Associative Memory Core* (+7.1% $\to$ 35.5% total integrity)  

---

## Team

| Name | Squadron station | Mission 05 role | Deliverable |
|---|---|---|---|
| Rafid Mohammad Ahsan | Core & Lead Architect | Part 1 · Systems Architect (Lead Engineer) | `01_HNSW_Vector_Index.py`, `ps05_solution.ipynb` |
| Rafid Mohammad Ahsan | Protocol & Interface Specialist | Part 2 · Communications & Protocol Officer | `02_BM25_and_RRF_Combiner.py` |
| Tahjib Al Sakin | Data & Memory Engineer | Part 3 · Data & Payload Specialist | `03_NotebookLM_RAG_Audit.py`, `notebooklm_audit_report.md` |
| Kunal Choudhary | Security & Ethical Navigator | Part 4 · Security & Ethics Sentinel | `04_RAG_Provenance_and_Safety.py` |
| Mubina Rahaman Jerin | AI Security Officer | EU AI Act compliance | `06_AI_Act_Compliance.md` |

---

## Who Did What

| Name | Work done |
|---|---|
| **Rafid Mohammad Ahsan** | *Architect:* built HNSW in pure Python/NumPy: cosine core, layer assignment $l = \lfloor -\ln U \cdot m_L \rfloor$, `search_layer`, the neighbour-selection heuristic, greedy descent. Benchmarked on 25,000 × 768 vectors: recall@10 0.980 at 0.36 ms per query vs 34.8 ms (Python flat) and 3.1 ms (NumPy flat). Assembled the notebook. *Protocol:* built Okapi BM25 and RRF (k = 60) and the hybrid retriever. On 30 labelled questions, BM25 wins on exact codes and dense wins on paraphrases. Plain RRF does not beat dense on this corpus; a code-aware weighted RRF gets 13/13 on held-out questions. |
| **Tahjib Al Sakin** | NotebookLM grounding lab: 5 cross-document queries over Lewis et al. (2020), Edge et al. (2024) and Document C; a citation-precision check by clicking every badge; 3 negative probes (fictitious code, real-world fact absent from the sources, false premise); Audio Overview transcript and scores. Lab run on 2026-11-03 in Gemini Notebook (formerly NotebookLM) by Kunal Choudhary with Claude in the desktop browser: 98 citations checked (precision 0.34 strict / 0.94 lenient, coverage 98 %), 2 of 3 clean refusals; Audio Overview transcribed via Google speech recognition and scored (fidelity 2, personas 4, pedagogy 3). |
| **Kunal Choudhary** | Indirect prompt-injection detector (normalisation for zero-width characters, leetspeak and base64; 15 rules; an embedding-margin scanner). On 45 labelled chunks it catches 25/25 attacks in 12 techniques with 4 false alarms. RAG Triad scorer: the faithfulness check with numbers and codes reaches 0.90 claim accuracy vs 0.75 for similarity alone. Grounded assistant that quarantines poisoned records; SHA-256 hash-chained provenance log that detects edits. Measured that similarity cannot decide answerability (11 of 12 unanswerable questions still answered). |
| **Mubina Rahaman Jerin** | Reviewed the mission under the EU AI Act: Art. 50 disclosure and machine-readable marking, Art. 15 robustness against indirect prompt injection, GPAI duties of the NotebookLM / WordLlama providers, GDPR for uploads and audit logs. Wrote `06_AI_Act_Compliance.md`. |

**Document C** (`corpus/`): the brief lists it as provided in the course repo, but it is not there. The squadron wrote it so that the lab and the retrieval benchmark can run.
