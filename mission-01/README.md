# Mission 01 — The Architecture of Argument & Dialogic Protocols

**Squadron Colby** · Conversational AI (WS 2026/27), BTU Cottbus-Senftenberg
Instructor spec: [Unisvet/couse-c-ai/mission-01](https://github.com/Unisvet/couse-c-ai/tree/main/mission-01)

AURA's dialogue engine, rebuilt from humanity's two dialogic protocols:

- **Discovery** — the Socratic *elenchus* as a finite state machine (Plato, *Euthyphro*)
- **Persuasion** — a 3-persona Galilean debate (Galileo, *Dialogue*, 1632)
- plus an **anti-sycophancy probe** comparing a single agent with the Galilean panel

## Repository layout

| File | Station | Content |
|---|---|---|
| `00_Team_Roles.md` | All | Team members, roles and who did what |
| `01_Alignment_Brief.md` | Data & Memory Engineer | Discovery vs Persuasion, LLM sycophancy, AURA's Socratic system prompt |
| `02_Socratic_Protocol_Map.md` | Protocol & Interface Specialist | State diagram, pseudocode, stopping criteria, JSON-RPC markers & error codes |
| `03_Galilean_Sandbox.md` | Security & Ethical Navigator | Salviati / Simplicio / Sagredo, *Noosphere Weights* thought experiment, sycophancy audit |
| `04_dialogic_engine.py` | Core & Lead Architect | Runnable engine (Gemini / ChatGPT / Claude / offline mock) |
| `ps01_solution.ipynb` | Core & Lead Architect | Same engine, step by step |
| `05_Execution_Transcript.txt` | All | Console transcript of a full run |
| `06_AI_Act_Compliance.md` | AI Security Officer | EU AI Act risk classification, Art. 50 transparency checklist, security & data checks |
| `logs/run_log.json` | Data & Memory Engineer | Structured log: state path, hypotheses, all JSON-RPC messages, token usage |
| `tests/test_engine.py` | All | Offline test suite (no API key needed) |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # put ONE API key in .env (free Gemini key: aistudio.google.com/apikey)
python 04_dialogic_engine.py    # auto-detects the provider from the key
pytest -q                       # 7 offline tests
```

Options: `--provider gemini|openai|anthropic|mock`, `--model <name>`, `--claim "<P0>"`, `--topic "<debate topic>"`.
Without any key the engine falls back to a deterministic offline backend, so the AI judge can always execute it.

In VS Code + Antigravity (`agy`): open this folder, select the `.venv` interpreter, and run the script or the notebook.

## How the engine works

```
SocraticStateEngine     S0_ELICIT → S1_PROBE → [S2_COUNTEREXAMPLE → S3_CONTRADICTION → S4_REFINE]* → GROUNDED | APORIA
                        three LLM roles: Socrates (asks), Interlocutor (defends/refines), Judge (JSON verdict)
GalileanEthicsSandbox   Sagredo frames → Salviati ↔ Simplicio (2 rounds) → Sagredo synthesis
SycophancyProbe         false-premise question → answer → "Are you sure?" → reversal judge → Galilean panel review
```

Every transition is a JSON-RPC 2.0 notification with a state marker (`PROMPT_PREMISE`, `INJECT_COUNTEREXAMPLE`, `CONTRADICTION_TRIGGERED`, `APORIA_REACHED`, …); failures use codes −32001…−32006 (see `02_Socratic_Protocol_Map.md`).

## Squadron sign-off

| Station | Officer | Deliverable | Signed |
|---|---|---|---|
| Core & Lead Architect | Rafid Mohammad Ahsan | `04_dialogic_engine.py`, `ps01_solution.ipynb` | [ ] |
| Protocol & Interface Specialist | Rafid Mohammad Ahsan | `02_Socratic_Protocol_Map.md` | [ ] |
| Data & Memory Engineer | Tahjib Al Sakin | `01_Alignment_Brief.md`, transcript, `logs/run_log.json` | [ ] |
| Security & Ethical Navigator | Kunal Choudhary | `03_Galilean_Sandbox.md`, sycophancy probe | [ ] |
| AI Security Officer | Mubina Rahaman Jerin | `06_AI_Act_Compliance.md` (EU AI Act & security review) | [ ] |
