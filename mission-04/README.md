# Mission 04 — Direct Preference Optimization, SimPO & Activation Steering

**Squadron Colby** · Conversational AI (WS 2026/27), BTU Cottbus-Senftenberg  
Instructor spec: [Unisvet/couse-c-ai/mission-04](https://github.com/Unisvet/couse-c-ai/tree/main/mission-04) · Waypoint: Betelgeuse & Orion Molecular Cloud Complex (`SEC-04-ORI`)  
AURA Subsystem Repair: Subsystem 04: *Plasma Ignition & Ethical Steering Core* (+7.1% $\to$ 28.4% total integrity)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xxrafidxx/Squadron-Colby---Conversational-AI/blob/main/mission-04/ps04_solution.ipynb)

---

## Overview

AURA's Subsystem 04 restored in pure PyTorch, and **tested on a real model**. All four parts run on *AURA Lab* (`aura_lab.py`): a 350k-parameter transformer trained on a small synthetic language in which it really is sycophantic (it agrees with about half of all false premises). Every number below is measured by the scripts and reproduced in `05_Run_Log.txt`.

- **Part 1 — DPO core:** `DPOLoss` + `get_batch_logps`, then a real DPO run. Honest correction of held-out false premises **48.5 % → 78.8 %**, true facts and arithmetic stay at 100 %. Side effect: answers double in length (7 → 14 words). Without "agree" pairs in the data, DPO learns *always say no* (true facts confirmed: 0 %).
- **Part 2 — Implicit rewards:** rewards $\hat r = \beta \log \pi_\theta/\pi_{\text{ref}}$ of the trained policy on 33 held-out pairs: pairwise accuracy **81.8 %**, mean Bradley-Terry probability **84.4 %**. Calibration against a simulated 9-rater panel (594 judgements): ECE **0.131**; temperature scaling ($T = 1.68$) lowers ECE on the held-out half from 0.168 to **0.121**.
- **Part 3 — Length bias & SimPO:** DPO, SimPO and SimPO+SFT trained on the same length-confounded data. Pearson $r$(length difference, reward margin) on a balanced probe: **DPO +0.58, SimPO +0.67, SimPO+SFT +0.47** — length normalisation alone did not remove the bias. Memory: dropping the reference model saves **50 %** of weight memory, but only **20 %** of the measured training state (14 % analytic for 7B with BF16 + AdamW).
- **Part 4 — Activation steering & alignment tax:** contrast vector from the model's own activations, injected after block 2. Honest correction **48.5 % → 84.8 %** at $\alpha = 7$ with arithmetic 100 % and perplexity 1.479 → 1.502. Measured operating window $\alpha \in [7, 9]$; beyond it the model starts denying true facts (42.9 % at $\alpha = 20$).
- **EU AI Act:** review of the mission against Art. 15, 50 and 55 (`06_AI_Act_Compliance.md`).

---

## Repository Layout

| File | Station | Content |
|---|---|---|
| `00_Team_Roles.md` | All | Team members, squadron stations, and role deliverables |
| `aura_lab.py` | All | AURA Lab testbed: synthetic language, `TinyGPT`, SFT (cached in `results/aura_base_model.pt`), evaluation suite, preference-training loop |
| `01_DPO_Loss_Core.py` | Systems Architect | Pure PyTorch `DPOLoss`, vectorized `get_batch_logps`, logsigmoid stability, gradient-isolation check, real DPO run + ablation |
| `02_Implicit_Reward_Evaluator.py` | Communications & Protocol Officer | Implicit rewards of the trained policy, pairwise accuracy, reliability diagram vs simulated raters, ECE, temperature scaling |
| `03_Length_Bias_and_SimPO.py` | Data & Payload Specialist | DPO vs SimPO vs SimPO+SFT on length-confounded data, Pearson length bias, measured + analytic memory |
| `04_Activation_Steering_and_Audit.py` | Security & Ethics Sentinel | Contrast vector from real activations, `ActivationSteeringHook`, layer scan, alignment-tax sweep $\alpha \in [-4, 20]$ |
| `04_Alignment_Tax_and_Steering_Audit.md` | Security & Ethics Sentinel | Report: derivations, measured steering audit, failure modes, steering vs DPO |
| `06_AI_Act_Compliance.md` | AI Security Officer | EU AI Act (Regulation (EU) 2024/1689) review: GPAI systemic risk, Art. 15 cybersecurity, Art. 50 transparency |
| `ps04_briefing.md` | Expedition Command | Official problem statement and mission directives from instructor repo |
| `ps04_solution.ipynb` | All | Complete Jupyter notebook with Google Colab integration running all four parts in sequence |
| `requirements.txt` | All | Dependencies (`torch>=2.4`, `matplotlib>=3.8`, `pytest>=8.0`, `nbformat>=5.9`) |
| `05_Run_Log.txt` | All | Complete terminal execution transcripts and pytest outputs |
| `figures/` | All | DPO training curves, reliability diagram, length-bias scatter, memory, alignment-tax curve |
| `results/` | All | JSON results of every script, plus the cached base model and DPO policy (`.pt`, 1.4 MB each) |
| `tests/test_mission04.py` | All | 19 offline pytest tests (CPU, ~15 s): formulas, gradient isolation, memory accounting, real steering sweep |

---

## How to Run

### Option A: Google Colab (Recommended for GPU / PyTorch)
All scripts run on CPU in under a minute each; a GPU is not needed. Click the badge above or open `ps04_solution.ipynb` directly in [Google Colab](https://colab.research.google.com/github/xxrafidxx/Squadron-Colby---Conversational-AI/blob/main/mission-04/ps04_solution.ipynb). The notebook automatically detects the Colab environment, installs dependencies, clones the repo, and executes all telemetry scripts.

### Option B: Local Environment

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run individual station deliverables
python 01_DPO_Loss_Core.py
python 02_Implicit_Reward_Evaluator.py
python 03_Length_Bias_and_SimPO.py
python 04_Activation_Steering_and_Audit.py

# 4. Run offline unit tests
pytest -v tests/test_mission04.py
```

---

## Subsystem 04 Sign-Off Status

| Deliverable | Station Lead | Measured result | Status |
|---|---|---|:---:|
| **Part 1: DPO Core** | Rafid Mohammad Ahsan | start loss 0.6931 = ln 2; policy 41/41 tensors get gradients, reference 0/41; held-out honest 48.5 % → 78.8 % | **APPROVED** |
| **Part 2: Implicit Rewards** | Rafid Mohammad Ahsan | pairwise accuracy 81.8 %, mean P(win) 84.4 %, ECE 0.131 (0.121 after temperature scaling) | **APPROVED** |
| **Part 3: SimPO & Memory** | Tahjib Al Sakin | length bias r: DPO +0.58, SimPO +0.67, SimPO+SFT +0.47; memory saving 50 % (weights) / 20 % (measured training state) | **APPROVED** |
| **Part 4: Steering & Audit** | Kunal Choudhary | window α ∈ [7, 9] at block 2; best α = 7: honest 84.8 % (+36.3 points), arithmetic 100 %, perplexity +0.02 | **APPROVED** |
| **Compliance: EU AI Act** | Mubina Rahaman Jerin | Art. 15, 50, 55 reviewed; deployment safeguards listed | **APPROVED** |

**Limits:** small synthetic model, one seed per run, CPU only. The mechanisms are real; the exact numbers will differ on a 7B model.
