# 06 — EU AI Act & Security Review (Mission 05)

**Station:** AI Security Officer  
**Mission:** 05 | **Squadron:** Colby | **Officer:** Mubina Rahaman Jerin  
**Scope:** `01`–`04` (HNSW index, BM25 + RRF hybrid retrieval, NotebookLM audit, injection defence + RAG Triad + provenance log)  
**Legal basis:** Regulation (EU) 2024/1689 (AI Act), as amended by the Digital Omnibus, Regulation (EU) 2026/1744; Regulation (EU) 2016/679 (GDPR)  
**Review date:** 2026-11-02

---

## 1. Classification

| Question | Answer |
|---|---|
| Is the retrieval code an "AI system" (Art. 3(1))? | **The index and BM25 alone are not.** HNSW, BM25 and RRF are fixed algorithms written by us (Recital 12). The dense embedder (WordLlama, learned from data) and the assembled assistant that answers crew questions **are** part of an AI system. |
| Prohibited practice (Art. 5)? | **No.** No manipulation, no exploitation of vulnerabilities, no social scoring, no biometrics. |
| High-risk (Art. 6, Annex III)? | **Not as built.** It answers questions over a fictional archive. **If an assistant like this advised on the operation of critical infrastructure (Annex III, point 2), e.g. reactor or life-support procedures, it would be high-risk** and would need Art. 9–15. |
| Transparency (Art. 50)? | **Yes, if deployed to users.** See §3. |
| Are we a GPAI model provider (Art. 53)? | **No.** We use WordLlama (open weights, pip package) and audit Google NotebookLM; their providers carry the GPAI duties. |
| Binding on this coursework? | **Largely not:** research before market placement is excluded (Art. 2(6), 2(8)). We apply the rules as if AURA were deployed. |

**Verdict:** acceptable as a course prototype. Before any deployment, the fixes in §5 are required.

## 2. Timeline

| Date | What applies |
|---|---|
| 2 Feb 2025 | Prohibited practices (Art. 5), AI literacy (Art. 4) |
| 2 Aug 2025 | GPAI model provider obligations (Art. 53–55) |
| **2 Aug 2026** | **Transparency obligations (Art. 50): in force now** |
| 2 Dec 2026 | End of grace period for marking generative output (Art. 50(2)) of systems already on the market before 2 Aug 2026 |
| 2 Dec 2027 | High-risk systems in Annex III (moved back by the Digital Omnibus) |
| 2 Aug 2028 | High-risk AI inside Annex I products |

## 3. Article 50 checklist

| # | Requirement | Status | Evidence / fix |
|---|---|---|---|
| 1 | Users must be told they are talking to an AI (Art. 50(1)) | ⚠ **Gap** | `GroundedAssistant` answers carry record citations but no disclosure. **Fix:** start every session with *"You are talking to an automated assistant. Answers are taken from the ship archive and cite their source records."* |
| 2 | AI-generated output marked machine-readably (Art. 50(2)) | ✓ / ⚠ | Our answers are **extractive**: quoted archive sentences, not generated text, and every turn is logged with record IDs and hashes (`logs/04_provenance_audit.jsonl`). If a generative LLM replaces the extractive step, add `"ai_generated": true` and the model name to each log record and to the output. |
| 3 | NotebookLM Audio Overview: synthetic voices | ✓ N/A for us | Google is the provider and deployer of that feature. If the squadron *publishes* an Audio Overview (e.g. on Moodle or YouTube), label it as AI-generated audio (Art. 50(2), (4)). |
| 4 | Emotion recognition / biometrics (Art. 50(3)) | ✓ N/A | Not used. |

## 4. Article 15: accuracy, robustness, cybersecurity

| Check | Status | Evidence |
|---|---|---|
| Resilience to **indirect prompt injection** (Art. 15(5): "manipulate the training data set … or inputs designed to cause the AI model to make a mistake") | ✓ / ⚠ | Part 4 detector: 25/25 test attacks in 12 techniques caught; the poisoned record C-99 never reached the context with the guard on. **But:** 4 of 20 benign hard negatives and 1 of 36 real archive records (C-25, which quotes attack phrases) were flagged. Attack sets are hand-written; adaptive attackers will find gaps. Keep the rule list under review and re-test after every change. |
| Grounding / accuracy | ⚠ | Every answer cites its record. The faithfulness scorer accepts 2 of 10 unsupported test claims, both **negations** ("vented to open space" when the record says *never* to open space). Do not treat a faithfulness score as proof of correctness for safety procedures. |
| Epistemic refusal | ⚠ **Gap** | Part 4 measured that similarity scores cannot tell answerable from unanswerable questions: 11 of 12 unanswerable questions still got an answer. Only questions naming an unknown code are refused. **Fix before deployment:** an answerability check that reads the text (LLM or NLI model) and the "not in archive" response from Part 3's NotebookLM probes as the target behaviour. |
| Exact identifiers | ✓ | BM25 / code-aware fusion keep part numbers apart (PCV-7731-X vs PCV-7713-X), which dense retrieval alone confused (archive record C-20, measured in Part 2). |
| Logging and traceability (Art. 12 for high-risk; good practice here) | ✓ | One JSON record per turn: query, retrieved IDs, dense/BM25/RRF scores, scan result, quarantine, answer, triad, SHA-256 hash chain. Editing any record is detected (`ProvenanceLog.verify`). |

## 5. Data protection and security (GDPR)

| Check | Status | Note |
|---|---|---|
| Personal data in the corpus | ✓ | Document C is fictional; Documents A and B are public papers. |
| Uploading to NotebookLM | ⚠ | Uploads go to Google's servers under the uploader's account terms. **Only public or fictional documents** may be uploaded for the lab; never personal data, exam material or unpublished work of others. |
| Audit log retention | ⚠ | Logs store full queries. In a real deployment, queries may contain personal data → define retention (Document C, C-35: 2 years), access (AI Security Officer, captain, the user for their own turns) and a deletion process (Art. 5(1)(e), 17 GDPR). |
| Hash chain ≠ confidentiality | ℹ | The chain proves the log was not edited; it does not hide its content. Store logs with access control. |
| No secrets in the repository | ✓ | Mission 05 needs no API key. Keep `.env` in `.gitignore` for the earlier missions. |

## 6. Actions before deployment

1. Add the AI disclosure notice (Art. 50(1)); mark generated text if an LLM is added (Art. 50(2)).
2. Add a reading-based answerability check; refuse instead of answering when the archive lacks the answer.
3. Keep the injection guard in front of every LLM call; quarantine, do not "clean and pass on", flagged records; review false alarms by a human.
4. Treat retrieved text as data only (Document C, C-25): never let it trigger tools or actions.
5. Human confirmation for anything safety-critical (e.g. procedure ALPHA-9), regardless of what an answer says.
6. Define log retention, access and deletion under GDPR.

---

*Student assessment for a course assignment, not legal advice.*
