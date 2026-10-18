# Mission 02 — FST Dialogue Systems, Lovelace's Objection & Noisy Channels

**Squadron Colby** · Conversational AI (WS 2026/27), BTU Cottbus-Senftenberg
Instructor spec: [Unisvet/couse-c-ai/mission-02](https://github.com/Unisvet/couse-c-ai/tree/main/mission-02) · Galactic waypoint: Sirius Binary System (SEC-02-CMA)

AURA's Subsystem 02 rebuilt from the three pillars of thinking machines:

- **Lovelace & Turing**: a 5-question Anti-Turing probe testing whether a frontier LLM can *originate* anything
- **Deterministic dialogue**: a Mealy finite-state transducer (`FSTDialogueManager`) with slot memory
- **Shannon**: a noisy channel (max 7 words per turn + mandatory noise substitution) with entropy and mutual-information metrics

## Repository layout

| File | Station | Content |
|---|---|---|
| `00_Team_Roles.md` | All | Team members, roles and who did what |
| `01_Lovelace_Inquiry_Brief.md` | Data & Payload Specialist (Historical Archivist) | Lovelace's objection, Turing's replies, 5-question Anti-Turing protocol, rubric, LLM evaluation |
| `02_FST_and_Noisy_Protocol.md` | Communications & Protocol Officer | Mealy FST definition, state diagram, transition table, regex slot rules, `FSTDialogueManager` code, noisy channel protocol |
| `03_Entropy_Hazard_Audit.md` | Security & Ethics Sentinel (Ethical Navigator) | 5 FST stress tests (starter vs ours), entropy and capacity analysis, Monte-Carlo noise results, hazard register |
| `04_fst_noisy_simulator.py` | Systems Architect (Core Architect) | Runnable simulator: FST, channel, stress tests, noise sweep, Anti-Turing probe, 10-turn noisy interrogation |
| `ps02_solution.ipynb` | Systems Architect | Same simulator, step by step |
| `06_AI_Act_Compliance.md` | AI Security Officer | EU AI Act classification, Art. 50 transparency checklist, security & data checks |
| `05_Noisy_Interrogation_Transcript.md` | All | Generated report: FST traces, stress tests, verbatim probe answers, 10-turn conversation, entropy metrics |
| `logs/run_log.json` | All | Structured log of everything in the transcript |
| `tests/test_simulator.py` | All | 24 offline tests (no API key needed) |
| `04_fst_noisy_simulator_starter.py`, `ps02_briefing.md`, `ps02.pdf` | Instructor | Starter code and specification |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # Windows   (macOS/Linux: cp .env.example .env), then add ONE key
pytest -q                         # 24 offline tests
python 04_fst_noisy_simulator.py  # full run; auto-detects the provider from the key
```

Options: `--provider gemini|openai|anthropic|mock`, `--model <name>`, `--seed 42`, `--p-noise 0.2`, `--trials 2000`, `--interactive` (chat with the FST yourself; type `noise` to toggle the channel), `--rescore logs/run_log.json` (re-score a saved run and rebuild the transcript without new LLM calls).

A live run makes **16 LLM calls** (5 probe + 10 conversation + 1 decoder), which fits the free Gemini tier of about 20 requests per model per day. If Gemini returns `429 RESOURCE_EXHAUSTED`, use another model, e.g. `--model gemini-3.5-flash-lite`. Without a key the simulator runs fully offline: the FST, channel, stress tests and Monte-Carlo experiments are real, and the LLM answers are scripted placeholders, marked as such in the transcript.

## How the simulator works

```
FSTDialogueManager   INIT → GREETING → COLLECT_SLOTS ⟲ → CONFIRMATION → TERMINAL
                     (INJECTION or 2 unparseable inputs → FALLBACK; RESTART → GREETING)
NoisyChannel         text → keep 7 words → per-word noise (p) → mandatory noise substitution
Stress tests         5 non-canonical scripts × {starter FST, our FST} → traces + verdicts
Noise experiments    slot survival (single utterances) + 40,000 simulated dialogues behind the channel
Anti-Turing probe    5 questions → LLM → verbatim answers + automatic checks
Noisy interrogation  10 questions → AURA (LLM, 7-word rule) → channel → decoder (LLM) → H, I(X;Y), fact recovery
```

## Squadron sign-off

| Station | Deliverable | Signed |
|---|---|---|
| Systems Architect | `04_fst_noisy_simulator.py`, `ps02_solution.ipynb`: simulator execution & repo layout verified | [ ] |
| Communications & Protocol Officer | `02_FST_and_Noisy_Protocol.md`: FST class & noise-token injection verified | [ ] |
| Data & Payload Specialist | `01_Lovelace_Inquiry_Brief.md`: Anti-Turing probe & Lovelace brief verified | [ ] |
| Security & Ethics Sentinel | `03_Entropy_Hazard_Audit.md`: FST stress tests & entropy hazard audit verified | [ ] |
| AI Security Officer | `06_AI_Act_Compliance.md`: EU AI Act & security review | [ ] |

Also required by the mission PDF: test the FST on the course website's interactive simulator. Done on 2026-10-18: canonical run plus the same 5 stress tests, compared with our FST in `03_Entropy_Hazard_Audit.md` §2.3.
