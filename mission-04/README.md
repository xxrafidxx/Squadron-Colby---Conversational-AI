# Mission 04 — Direct Preference Optimization, SimPO & Activation Steering

**Squadron Colby** · Conversational AI (WS 2026/27), BTU Cottbus-Senftenberg  
Instructor spec: [Unisvet/couse-c-ai/mission-04](https://github.com/Unisvet/couse-c-ai/tree/main/mission-04) · Waypoint: Betelgeuse & Orion Molecular Cloud Complex (`SEC-04-ORI`)  
AURA Subsystem Repair: Subsystem 04: *Plasma Ignition & Ethical Steering Core* (+7.1% $\to$ 28.4% total integrity)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xxrafidxx/Squadron-Colby---Conversational-AI/blob/main/mission-04/ps04_solution.ipynb)

---

## Overview

AURA's Subsystem 04 restored from first principles in pure PyTorch:

- **Direct Preference Optimization (DPO):** Closed-form reparameterization eliminating separate reward models and PPO loops, with vectorized log-probability gather and numerical logsigmoid clamping.
- **Implicit Reward Evaluator:** Derivation and extraction of analytical scalar rewards $\hat{r}(x, y)$ under Bradley-Terry choice probability models with empirical reliability calibration.
- **Length Bias Diagnostics & Reference-Free SimPO:** Detection and mitigation of verbosity exploitation in preference datasets, with target margin $\gamma = 0.5$ and multi-scale GPU memory profiling proving ~50% VRAM reduction.
- **Activation Steering & The Alignment Tax Audit:** Runtime Linear Artificial Tomography (LAT) contrast vectors injected into transformer decoder residual streams ($h_l \leftarrow h_l + \alpha v_{\text{steer}}$) to dynamically enforce truthfulness without retraining, establishing the optimal operational safety window $\alpha \in [0.8, 1.2]$.
- **EU AI Act Compliance:** Comprehensive regulatory and security evaluation covering Articles 15, 50, and 55 under Regulation (EU) 2024/1689.

---

## Repository Layout

| File | Station | Content |
|---|---|---|
| `00_Team_Roles.md` | All | Team members, squadron stations, and role deliverables |
| `01_DPO_Loss_Core.py` | Systems Architect | Pure PyTorch `DPOLoss`, vectorized `get_batch_logps`, logsigmoid stability, frozen reference gradient test |
| `02_Implicit_Reward_Evaluator.py` | Communications & Protocol Officer | Analytical reward extraction $\hat{r}(x, y)$, Bradley-Terry win probabilities, reliability calibration curve & ECE |
| `03_Length_Bias_and_SimPO.py` | Data & Payload Specialist | Pearson length-bias correlation ($r \approx 0.86 \to 0.04$), reference-free `SimPOLoss`, 7B/13B/70B VRAM benchmark |
| `04_Activation_Steering_and_Audit.py` | Security & Ethics Sentinel | LAT concept contrast vectors, `ActivationSteeringHook` residual injection, Alignment Tax audit across $\alpha \in [-2.0, +2.5]$ |
| `04_Alignment_Tax_and_Steering_Audit.md` | Security & Ethics Sentinel | Full architectural report: PPO vs DPO vs SimPO derivations, representation geometry, Pareto trade-off |
| `06_AI_Act_Compliance.md` | AI Security Officer | EU AI Act (Regulation (EU) 2024/1689) review: GPAI systemic risk, Art. 15 cybersecurity, Art. 50 transparency |
| `ps04_briefing.md` | Expedition Command | Official problem statement and mission directives from instructor repo |
| `ps04_solution.ipynb` | All | Complete Jupyter notebook with Google Colab integration running all four parts in sequence |
| `requirements.txt` | All | Dependencies (`torch>=2.4`, `matplotlib>=3.8`, `pytest>=8.0`, `nbformat>=5.9`) |
| `05_Run_Log.txt` | All | Complete terminal execution transcripts and pytest outputs |
| `figures/` | All | High-resolution telemetry charts (loss surface, calibration curves, length bias, VRAM, Pareto frontier) |
| `results/` | All | Formatted JSON benchmark metrics and empirical audit evaluations |
| `tests/test_mission04.py` | All | Offline pytest suite (CPU only) validating invariants, gradients, and formulas |

---

## How to Run

### Option A: Google Colab (Recommended for GPU / PyTorch)
Click the badge above or open `ps04_solution.ipynb` directly in [Google Colab](https://colab.research.google.com/github/xxrafidxx/Squadron-Colby---Conversational-AI/blob/main/mission-04/ps04_solution.ipynb). The notebook automatically detects the Colab environment, installs dependencies, clones the repo, and executes all telemetry scripts.

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

| Deliverable | Station Lead | Metric / Result | Status |
|---|---|---|:---:|
| **Part 1: DPO Core** | Rafid Mohammad Ahsan | Loss: `0.5981`, BT Win Prob: `54.98%`, Policy Grad: `0.3421`, Ref Grad: `None` | **APPROVED** |
| **Part 2: Implicit Rewards** | Rafid Mohammad Ahsan | Pairwise Accuracy: `87.5%`, Mean Margin: `+0.2125`, ECE: `0.0820` | **APPROVED** |
| **Part 3: SimPO & Memory** | Tahjib Al Sakin | Verbosity Correlation: `r = 0.041`, VRAM Saved: `13.6 GB` (7B model, 48.6% drop) | **APPROVED** |
| **Part 4: Steering & Audit** | Kunal Choudhary | Optimal Operating Window: $\alpha \in [0.8, 1.2]$, Truthfulness: `+38.0%`, Math Retained: `84.5%` | **APPROVED** |
| **Compliance: EU AI Act** | Mubina Rahaman Jerin | Articles 15, 50, and 55 compliance verified; clamp safeguards documented | **APPROVED** |
