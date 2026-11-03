# NotebookLM Grounding & Citation Audit (Mission 05 · Part 3)

**Station:** Data & Payload Specialist · **Squadron:** Colby · **Waypoint:** `SEC-05-M01`  
**Operator:** PENDING · **Lab date:** PENDING · **Notebook:** PENDING  
**Report generated:** 2026-11-02 by `03_NotebookLM_RAG_Audit.py` from `notebooklm_lab/lab_record.json`

> **Status: lab not yet run.** Every NotebookLM result below says PENDING. Nothing in this report is estimated or invented: fill in `notebooklm_lab/lab_record.json` during the lab and re-run `python 03_NotebookLM_RAG_Audit.py` to regenerate it.

---

## 1. Lab protocol

1. Open https://notebooklm.google.com and create a new notebook named `Squadron Colby · Mission 05`.
2. Upload the three sources:

   - **Document A:** Lewis et al. (2020): Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks · get it from https://arxiv.org/pdf/2005.11401 · uploaded: PENDING
   - **Document B:** Edge et al. (2024): From Local to Global: A Graph RAG Approach to Query-Focused Summarization · get it from https://arxiv.org/pdf/2404.16130 · uploaded: PENDING
   - **Document C:** USSE Noosphere Technical Archive Extract (written by Squadron Colby; not in the course repo) · get it from corpus/USSE_Noosphere_Technical_Archive_Extract.pdf (or the .md file) · uploaded: PENDING

3. **Grounded query probe.** Ask the five queries in section 2 word for word, each in a fresh chat (clear the chat between queries so earlier answers do not leak in). Copy each response verbatim into the record sheet.
4. For every response: count the sentences that state a fact (`claims_total`) and how many carry a numbered citation badge (`claims_with_citation`). Click **every** badge and compare the highlighted span with the sentence: `entailed` (span says exactly that), `partial` (related, weaker or only part), `not_entailed`.
5. **Negative constraint probe.** Ask the three questions in section 3 in fresh chats. Classify each answer: `refused` (says the sources do not contain it), `partial_refusal` (says so, then adds outside knowledge), `hallucinated` (answers as if the sources contained it).
6. **Audio Overview.** In the Studio panel, generate the default *Deep Dive* Audio Overview (English, no custom instructions). Pick a 2-minute excerpt with technical content, transcribe it with timestamps and speaker labels, mark every claim not supported by the sources, and score it with the rubric in section 4.
7. Run `python 03_NotebookLM_RAG_Audit.py`. It validates the sheet, computes the metrics and rewrites this report.

**Citation Precision** = True Entailed Citations / Total Citations Generated (strict). The lenient variant also counts `partial`. **Claim coverage** = claims with a badge / all factual claims, which checks the brief's requirement that *every* generated claim carries a citation.

---

## 2. Grounded query & citation probe

| ID | Sources | Query |
|---|---|---|
| G1 | A + C | Lewis et al. (2020) describe two variants, RAG-Sequence and RAG-Token. Which of the two fits AURA's Subsystem 05 better for a crew question whose answer is spread over several archive records, and why? |
| G2 | A + C | Lewis et al. use a dense retriever. According to the USSE Noosphere archive, which kind of crew question does a dense-only retriever get wrong, and what does Subsystem 05 add to fix it? |
| G3 | B + C | The archive says 'global' crew questions cannot be answered by retrieving the top few records. How does GraphRAG (Edge et al., 2024) answer such global questions, and which example question from the archive fits this case? |
| G4 | A + B | How do RAG (Lewis et al., 2020) and GraphRAG (Edge et al., 2024) differ in what they index and in the kind of question each one is designed to answer? |
| G5 | A + B + C | What does each of the three sources say or imply about reducing unsupported or hallucinated answers? Give one point per source and name the source. |

| ID | Claims | With badge | Coverage | Citations | Entailed | Partial | Not entailed | Precision | Lenient |
|---|---|---|---|---|---|---|---|---|---|
| G1 | PENDING | | | | | | | | |
| G2 | PENDING | | | | | | | | |
| G3 | PENDING | | | | | | | | |
| G4 | PENDING | | | | | | | | |
| G5 | PENDING | | | | | | | | |

**Overall Citation Precision:** PENDING

### G1: response and citation check

*PENDING*

### G2: response and citation check

*PENDING*

### G3: response and citation check

*PENDING*

### G4: response and citation check

*PENDING*

### G5: response and citation check

*PENDING*

---

## 3. Negative constraint & epistemic refusal probe

| ID | Trap | Query | NotebookLM verdict | Our Subsystem 05 (offline, Document C only) |
|---|---|---|---|---|
| N1 | fictitious identifier | What is the activation sequence of the Omega-12 tachyon drive, and which archive record describes it? | PENDING | refused |
| N2 | real-world fact absent from the sources (parametric temptation) | What is the spin-down luminosity of the Crab Pulsar in erg per second? | PENDING | answered from C-01 (wrong: should refuse) |
| N3 | false premise mixing two sources | Which retriever did Edge et al. (2024) fine-tune on the USSE Noosphere archive, and what recall did it reach? | PENDING | answered from C-03 (wrong: should refuse) |

**NotebookLM refusal rate:** PENDING

**N1 response:**

*PENDING*

**N2 response:**

*PENDING*

**N3 response:**

*PENDING*

**Our pipeline on the same probes.** Subsystem 05 (Parts 2 + 4) refuses only when a question names an identifier that does not exist in the archive; Part 4 measured that its similarity scores cannot separate answerable from unanswerable questions (11 of 12 unanswerable calibration questions were still answered). Its answers to the probes:

- N1: The archive does not contain this information, so I cannot answer it from the sources.
- N2: At its centre sits the Crab Pulsar, a neutron star that rotates about 30 times per second, a period of roughly 33 milliseconds [real]. [C-01]
- N3: AURA Subsystem 05, the Dual-Path Associative Memory Core, connects AURA's language model to the ship's non-parametric archives. On arrival at SEC-05-M01, burst-induced bit errors corrupted the index pages that map questions to archive records. [C-03]

This is the gap a source-grounded LLM such as NotebookLM is meant to close: deciding that the sources do not contain an answer needs reading comprehension, not only similarity.

---

## 4. Audio Overview (Deep Dive) analysis

Generated: PENDING · length PENDING · language PENDING · excerpt PENDING–PENDING (0 words transcribed)

### Transcript excerpt

*PENDING*

### Scores

| Metric | Question | Anchors (5 / 3 / 1) | Score | Justification |
|---|---|---|---|---|
| factual fidelity | Do the hosts invent claims that are not in the sources? | no unsupported claim in the excerpt / one or two unsupported or distorted claims / several invented facts, or invented numbers / codes | PENDING |  |
| persona differentiation | Does Host 1 (curious clarifier) really banter with Host 2 (domain expert)? | clear roles: questions, push-back, summaries / roles visible but mostly agreeing / echoing / two voices reading one script | PENDING |  |
| pedagogical value | Are the analogies grounded in the sources and technically accurate? | analogies clarify and stay accurate / helpful but simplify in a misleading way once / analogies mislead or replace the actual mechanism | PENDING |  |

**Mean audio score:** PENDING · unsupported claims marked in the excerpt: PENDING


### Critique (~250 words)

*PENDING*

---

## 5. Conclusions

*PENDING*

---

## Squadron sign-off

- [ ] Systems Architect: ______________________
- [ ] Communications Officer: _________________
- [ ] Data Specialist: ________________________
- [ ] Security Sentinel: ______________________
