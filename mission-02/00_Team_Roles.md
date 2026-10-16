# 00 — Squadron Colby: Team and Roles

**Mission:** 02 — FST Dialogue Systems, Lovelace's Objection & Noisy Channels  
**Course:** Conversational AI: History, Applications, Future · WiSe 2026/27 · BTU Cottbus-Senftenberg  
**Repository:** https://github.com/xxrafidxx/Squadron-Colby---Conversational-AI

---

## Team

| Name | Squadron station | Mission 02 role | Deliverable |
|---|---|---|---|
| Tahjib Al Sakin | Data & Memory Engineer | Part 1 · Data & Payload Specialist (Historical Archivist) | `01_Lovelace_Inquiry_Brief.md` |
| Rafid Mohammad Ahsan | Protocol & Interface Specialist | Part 2 · Communications & Protocol Officer | `02_FST_and_Noisy_Protocol.md` |
| Kunal Choudhary | Security & Ethical Navigator | Part 3 · Security & Ethics Sentinel (Ethical Navigator) | `03_Entropy_Hazard_Audit.md` |
| Rafid Mohammad Ahsan | Core & Lead Architect | Part 4 · Systems Architect (Core Architect) | `04_fst_noisy_simulator.py`, `ps02_solution.ipynb` |
| Mubina Rahaman Jerin | AI Security Officer | EU AI Act compliance | `06_AI_Act_Compliance.md` |

## Who did what

| Name | Work done |
|---|---|
| **Tahjib Al Sakin** | Examined Lovelace's objection (Note G) and Turing's replies (Lady Lovelace's Objection, Argument from Consciousness); designed the 5-question Anti-Turing protocol (meta-cognition, counterfactual logic, physical grounding, original insight, constrained creation) with a 0/1/2 scoring rubric; evaluated Gemini's verbatim answers against Lovelace's objection. |
| **Rafid Mohammad Ahsan** | *Protocol:* specified `FSTDialogueManager` as a Mealy transducer (INIT, GREETING, COLLECT_SLOTS, CONFIRMATION, TERMINAL/FALLBACK) with `slot_memory` and regex extraction for destination, speed and clearance; defined the noisy channel protocol (7-word limit, mandatory noise-token substitution). *Core:* built the simulator connecting the FST, noisy channel, Anti-Turing probe and 10-turn noisy interrogation to Gemini/ChatGPT/Claude, with state traces, entropy metrics, logs, tests and repository layout. |
| **Kunal Choudhary** | Ran 5 non-canonical stress tests (wrong order, sarcasm, off-topic, prompt injection, negation) against the starter FST and ours; analysed Shannon entropy, mutual information and channel capacity under noise (slot survival, 40,000 simulated dialogues); built the hazard register; ran the live Gemini run (transcript and run log). |
| **Mubina Rahaman Jerin** | Checked the simulator against the EU AI Act (whether the rule-based FST counts as an AI system, prohibited practices, high-risk classification, Article 50 transparency); reviewed API-key handling, personal data, prompt injection and unsafe actions from misunderstood input; listed fixes before deployment. |

**Submitted PDFs:** `00_Team_Roles` · `01_Lovelace_Inquiry_Brief` · `02_FST_and_Noisy_Protocol` · `03_Entropy_Hazard_Audit` · `06_AI_Act_Compliance`. Part 4 (code, transcript, tests) is in the repository.
