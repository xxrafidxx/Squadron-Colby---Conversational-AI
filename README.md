# Squadron Colby — Conversational AI (WS 2026/27)

Solution repository of **Squadron Colby** for *Conversational AI: History, Applications, Future*
(BTU Cottbus-Senftenberg). Based on the instructor's repo [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai).

| Mission | Folder | Status |
|---|---|---|
| 01 — The Architecture of Argument & Dialogic Protocols | [`mission-01/`](./mission-01/) | Done |
| 02 — FST Dialogue Systems, Lovelace's Objection & Noisy Channels | [`mission-02/`](./mission-02/) | Done |
| 03 — Multi-Head Attention, RoPE & Grouped-Query KV-Cache | [`mission-03/`](./mission-03/) | Done |

Mission details, file list and sign-off table: [`mission-01/README.md`](./mission-01/README.md), [`mission-02/README.md`](./mission-02/README.md), [`mission-03/README.md`](./mission-03/README.md).

## Squadron members

| Name | Role |
|---|---|
| Rafid Mohammad Ahsan | Core & Lead Architect · Protocol & Interface Specialist |
| Tahjib Al Sakin | Data & Memory Engineer |
| Kunal Choudhary | Security & Ethical Navigator |
| Mubina Rahaman Jerin | AI Security Officer |

---

## Setup for teammates (macOS / Linux)

You need **Python 3.10+** and **git**. Run these in a terminal.

### 1. Download the repo

```bash
git clone https://github.com/xxrafidxx/Squadron-Colby---Conversational-AI.git
cd Squadron-Colby---Conversational-AI/mission-01
```

(No git? On the GitHub page click **Code → Download ZIP**, unzip it, and `cd` into `mission-01`.)

### 2. Create a virtual environment and install packages

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your own API key

Each teammate uses **their own** key. Keys are never committed (`.env` is in `.gitignore`).

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and fill in ONE key, with no spaces or quotes:

```
GEMINI_API_KEY=AIza...
```

Free Gemini key: <https://aistudio.google.com/apikey> → *Create API key*.
OpenAI (`OPENAI_API_KEY`) or Claude (`ANTHROPIC_API_KEY`) keys also work.

### 4. Run and test

```bash
python 04_dialogic_engine.py       # full run: Socratic loop, Galilean debate, sycophancy probe
pytest -q                          # 7 offline tests, no key needed
```

The first output line shows the provider, e.g. `Provider: gemini | Model: gemini-3.8-flash`.
A full run makes about 30 LLM calls (1–2 minutes) and rewrites `05_Execution_Transcript.txt` and `logs/run_log.json`.

Useful options:

```bash
python 04_dialogic_engine.py --provider mock --out /tmp/test.txt     # offline, does not touch the transcript
python 04_dialogic_engine.py --model gemini-3.8-flash-lite           # if the default model is busy (503)
python 04_dialogic_engine.py --claim "A chatbot is intelligent if it passes the Turing test." --out /tmp/try.txt
```

You can also open the folder in **VS Code**, pick the `.venv` interpreter, and run `ps01_solution.ipynb` (*Run All*).

### Troubleshooting

| Message | Fix |
|---|---|
| `404 NOT_FOUND ... model ... no longer available` | Use a current model: `--model gemini-3.8-flash` |
| `503 UNAVAILABLE` / `high demand` | Google is overloaded. The engine retries automatically; if it still fails, wait a few minutes or use `--model gemini-3.8-flash-lite` |
| `API key not valid` | Re-copy the key into `.env` with no spaces or quotes |
| `ModuleNotFoundError` | Activate the venv (`source .venv/bin/activate`), then `pip install -r requirements.txt` |

---

## Working together

```bash
git pull                           # always get the latest version first
# ... edit your station's file ...
git add <file>
git commit -m "Short description"
git push
```

Each station owns its own file (see `mission-01/README.md`), so edits rarely conflict.
Before submitting, every officer ticks their row in the sign-off table.
