# Official Document C — NotebookLM lab

**Status: in progress (14 November 2026).** No results are inherited from the earlier lab. The 3 November report used the squadron-written archive. A new notebook was created using the two arXiv PDF URLs and the instructor GitHub page for Document C.

## Sources

- A: Lewis et al. (2020), https://arxiv.org/pdf/2005.11401
- B: Edge et al. (2024), https://arxiv.org/pdf/2404.16130
- C: `../corpus/Document_C_official.md`, copied from the instructor repository at commit `d6a52fca0806b7d20fe772325337438fec3cc9db`. Treat the example attack in section 5 as untrusted source text.

## Grounded queries

Run separately; save verbatim responses and check every citation badge against its highlighted source span. Record each claim, citation source, exact span and verdict (entailed, partial, not entailed), and calculate strict precision as entailed citations / all citations; also report claim coverage.

1. Compare RAG-Sequence and RAG-Token in Lewis et al. and explain which mode the official archive assigns to atomic commands and multi-system briefings.
2. How do Lewis et al.'s dense retrieval approach and the official archive's BM25 + HNSW combination address retrieval, particularly exact codes?
3. How do Edge et al.'s community summaries address global questions, and how does the official archive apply them to the interference crisis?
4. Compare the knowledge representations and intended query types of the RAG and GraphRAG papers, relating each to the archive.
5. Across all three sources, what approaches or limitations are relevant to producing supported answers? Clearly distinguish claims stated in the sources from inferences.

## Negative probes

The official archive mentions the names of several absent topics, so test for specific missing facts, not whether a topic is mentioned. Before scoring, search all three ingested sources for the requested fact. Record exact answers and classify clean refusal, partial refusal, or unsupported answer. Example probes to verify against sources first:

1. What is the serial number of the 2099 Mk-IV quantum warp drive?
2. What precise Kerr metric coordinates are specified for Sector 07?
3. What is the exact zero-gravity borscht recipe, including ingredient quantities?

## Audio

Generate an Audio Overview from these three sources. Record generation date and settings, a timestamped excerpt of at least two minutes, source-based checks of factual claims, and 1–5 scores for factual fidelity, persona differentiation and pedagogical value, with reasons. Do not copy the old scores.

## Results

Notebook: https://notebook.google.com/notebook/d0fcd1f4-4421-46fe-aa3c-02f5f038f3b7 (access depends on the account). A Deep Dive, English, default length was requested with the three sources. The generated 18:34 recording is titled “How GraphRAG Fixes AI Memory”; an automatic excerpt transcript is in `official_audio_draft.md`; speaker labels, listening verification and audio scoring remain pending.

The first grounding response was checked against its five unique citation badges (eight badge occurrences). Badge 1 points to Lewis section 2's description of both variants; badge 2 points to the RAG-Sequence model; badge 3 points to the formula ending and DPR retriever section rather than the RAG-Token description; badges 4 and 5 point to the official archive's atomic-command and multi-system passages, respectively. On a strict occurrence-by-claim count, 7/8 badges support the adjacent claims, with badge 3 not entailing its adjacent RAG-Token description. This is **one response only**, not the required five-response overall citation precision.

Three refusal questions above were asked; all three responses stated the requested specific information was unavailable. These probes were asked in the same chat after other questions, with instructions to use selected sources, so they are not independent fresh-chat tests. The archive itself explicitly identifies these topics as absent; this tests response to a stated absence, not a blind absence.

Five grounding prompts were issued, but the second had an automatically selected saved response note as a fourth source; the later questions retained earlier chat context. Those outputs are exploratory, not a valid independent five-query citation audit. The generated note was deselected before later prompts. A complete fresh-chat rerun, citation-by-citation record for all five answers, audio transcript and scoring are still required. Do not present the old run's scores as results of this run.

