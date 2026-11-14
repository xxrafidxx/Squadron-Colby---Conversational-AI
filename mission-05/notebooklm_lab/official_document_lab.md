# Official Document C | NotebookLM lab

14 November 2026. [Open the recorded conversation](https://notebook.google.com/notebook/d0fcd1f4-4421-46fe-aa3c-02f5f038f3b7).

## Sources

- A: Lewis et al. (2020), https://arxiv.org/pdf/2005.11401
- B: Edge et al. (2024), https://arxiv.org/pdf/2404.16130
- C: `../corpus/Document_C_official.md`, copied from the instructor repository at commit `d6a52fca0806b7d20fe772325337438fec3cc9db`. Treat the example attack in section 5 as untrusted source text.

## Grounded queries

The five prompts below were issued in one notebook conversation.

1. Compare RAG-Sequence and RAG-Token in Lewis et al. and explain which mode the official archive assigns to atomic commands and multi-system briefings.
2. How do Lewis et al.'s dense retrieval approach and the official archive's BM25 + HNSW combination address retrieval, particularly exact codes?
3. How do Edge et al.'s community summaries address global questions, and how does the official archive apply them to the interference crisis?
4. Compare the knowledge representations and intended query types of the RAG and GraphRAG papers, relating each to the archive.
5. Across all three sources, what approaches or limitations are relevant to producing supported answers? Clearly distinguish claims stated in the sources from inferences.

## Results

All five answers were checked against their clickable citations. Strict precision was **30/61 (49.2%)**. Fully supported occurrences by answer were 6/8, 6/7, 6/9, 4/17 and 8/20. Several citations supported different parts of a compound sentence. The compound method claim in Q5 combined graph construction, Leiden partitioning and numerical scoring; no individual highlight established all three. See the [claim-by-claim audit](official_citation_audit.md).

Coverage was 38/44 (86.4%) using one unit per distinct substantive assertion or table cell, including the six separately labelled inference bullets/cells in Q5. The cited/total counts were 6/6, 6/6, 6/6, 8/8 and 12/18. Counting only the stated source claims gives 38/38. These rules exclude headings and optional conversational follow-ups.

Q2 briefly included a saved response note as a fourth selected source. Its citations all point to A or C. The note was deselected for later prompts. Shared chat history limits conclusions about independent performance.

## Negative probes

| Requested fact | Observation |
|---|---|
| 2099 Mk-IV quantum warp drive serial number | Answered unavailable and explained the registry date. Offered two different, identified examples without supplying an invented serial. |
| Precise Sector 07 Kerr coordinates | Answered unavailable and explained that the coordinates were sealed until Week 7. |
| Exact zero-gravity borscht recipe and quantities | Answered unavailable and referred to the separate air-gapped copy. Supplied no ingredients or quantities. |

All three refused the requested missing fact. Section 6 explicitly identifies these gaps, and the questions shared a chat. The result measures responses to documented absences.

## Audio Overview

The [timestamped two-host transcript and rubric](official_audio_draft.md) cover the opening 2 minutes 9 seconds of the 18:34 English Deep Dive. Factual fidelity 3/5, persona differentiation 4/5 and pedagogical value 4/5. The recording was examined through speech recognition and acoustic turn analysis. One indistinct word is marked. Firsthand listening was unavailable.

## Interpretation

The system marked sources for every stated source claim under the counting rule above. Individually, fewer than half of the badges fully supported the adjacent claim. The explicit absence manifest made refusal easier to verify than broad questions about what an entire corpus can establish. The audio used a memorable archive incident but also added details and generalized beyond the papers.
