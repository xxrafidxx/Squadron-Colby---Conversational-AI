# 00 — Squadron Colby: Team and Roles

**Mission:** 04 — Direct Preference Optimization (DPO), Verbosity Regularization & Steering Vector Control  
**Course:** Conversational AI: History, Applications, Future · WiSe 2026/27 · BTU Cottbus-Senftenberg  
**Repository:** https://github.com/xxrafidxx/Squadron-Colby---Conversational-AI  
**Waypoint:** Betelgeuse & Orion Molecular Cloud Complex (`SEC-04-ORI`, 642 ly from Sol)  
**AURA Subsystem Repair:** Subsystem 04: *Plasma Ignition & Ethical Steering Core* (+7.1% $\to$ 28.4% total integrity)  

---

## Team

| Name | Squadron station | Mission 04 role | Deliverable |
|---|---|---|---|
| Rafid Mohammad Ahsan | Core & Lead Architect | Part 1 · Systems Architect (Lead Engineer) | `01_DPO_Loss_Core.py`, `ps04_solution.ipynb` |
| Rafid Mohammad Ahsan | Protocol & Interface Specialist | Part 2 · Communications & Protocol Officer | `02_Implicit_Reward_Evaluator.py` |
| Tahjib Al Sakin | Data & Memory Engineer | Part 3 · Data & Payload Specialist | `03_Length_Bias_and_SimPO.py` |
| Kunal Choudhary | Security & Ethical Navigator | Part 4 · Security & Ethics Sentinel | `04_Activation_Steering_and_Audit.py`, `04_Alignment_Tax_and_Steering_Audit.md` |
| Mubina Rahaman Jerin | AI Security Officer | EU AI Act compliance | `06_AI_Act_Compliance.md` |

---

## Who Did What

| Name | Work done |
|---|---|
| **Rafid Mohammad Ahsan** | *Architect:* Constructed pure PyTorch `DPOLoss(nn.Module)` from first principles without relying on third-party high-level wrappers. Implemented vectorized `get_batch_logps` using `torch.gather` with label masking (`-100`) and length normalization toggles. Verified numerical stability using `torch.nn.functional.logsigmoid` and empirically validated that backpropagation updates active policy logits exclusively while reference model logits remain strictly frozen. Assembled `ps04_solution.ipynb` with interactive visual telemetry and Google Colab integration. *Protocol:* Built `evaluate_implicit_rewards` calculating analytical scalar rewards $\hat{r}_\theta(x, y)$ under the Rafailov reparameterization. Evaluated Bradley-Terry preference probabilities $P(y_w \succ y_l \mid x) = \sigma(\Delta r)$, computed pairwise classification accuracy, and built an empirical reliability calibration curve with Expected Calibration Error (ECE) diagnostic. |
| **Tahjib Al Sakin** | Quantified verbosity exploitation in standard DPO by implementing `calculate_pearson_correlation` to prove strong length bias ($r \approx 0.85$) between response length differentials $(|y_w| - |y_l|)$ and model reward margins. Implemented reference-free `SimPOLoss(nn.Module)` (Simple Preference Optimization) using length-normalized average per-token log-probabilities with target margin $\gamma = 0.5$ and scaling factor $\beta = 2.0$, reducing length bias correlation to near zero ($r \approx 0.05$). Profiled GPU VRAM requirements across 7B, 13B, and 70B parameter models, demonstrating a ~50% reduction in memory footprint by completely eliminating the reference model. |
| **Kunal Choudhary** | Implemented Linear Artificial Tomography (LAT) representation engineering in pure PyTorch. Formulated `extract_concept_vector` to calculate unit L2-normalized concept contrast directions $v_{\text{steer}} = \mathbb{E}[h_l(\text{pos})] - \mathbb{E}[h_l(\text{neg})]$. Created `ActivationSteeringHook` to inject steerable perturbations directly into transformer decoder residual streams ($h_l \leftarrow h_l + \alpha v_{\text{steer}}$). Conducted an extensive *Alignment Tax Audit* across multipliers $\alpha \in [-2.0, +2.5]$, charting the Pareto frontier between conversational sycophancy rejection, core mathematical reasoning capability retention, and language perplexity, establishing the optimal operational safety window $\alpha \in [0.8, 1.2]$. Authored the comprehensive engineering report `04_Alignment_Tax_and_Steering_Audit.md`. |
| **Mubina Rahaman Jerin** | Conducted a comprehensive regulatory and security audit under Regulation (EU) 2024/1689 (EU AI Act). Classified DPO/SimPO alignment and activation steering within the EU legal framework. Evaluated compliance with Article 15 (Accuracy, Robustness and Cybersecurity against reward hacking and adversarial prompt injection), Article 50 (Transparency obligations for conversational AI), and Article 55 (Systemic risk mitigations for GPAI models, specifically auditing capability regression from the Alignment Tax). Authored `06_AI_Act_Compliance.md` and established the operational sign-off protocol. |

---

## Deliverables Checklist

- [x] `00_Team_Roles.md` — Squadron Colby station assignments and responsibilities
- [x] `01_DPO_Loss_Core.py` — Pure PyTorch DPO loss core, vectorized gather, and gradient validation
- [x] `02_Implicit_Reward_Evaluator.py` — Bradley-Terry choice probability evaluation & calibration curve
- [x] `03_Length_Bias_and_SimPO.py` — Length bias Pearson diagnostic, SimPO loss & VRAM benchmark
- [x] `04_Activation_Steering_and_Audit.py` — LAT representation extraction, residual hook & tax audit
- [x] `04_Alignment_Tax_and_Steering_Audit.md` — In-depth architectural report on steering and alignment tax
- [x] `05_Run_Log.txt` — Full execution and test output log
- [x] `06_AI_Act_Compliance.md` — EU AI Act compliance review (Art. 15, 50, 55)
- [x] `ps04_briefing.md` — Expedition Command mission brief & mathematical specifications
- [x] `ps04_solution.ipynb` — Executable end-to-end Jupyter notebook with Google Colab support
- [x] `requirements.txt` — Python dependencies
- [x] `figures/` — Generated plots (DPO loss surface, BT calibration, length bias, VRAM, Pareto curve)
- [x] `results/` — JSON benchmark metrics and empirical audit evaluations
- [x] `tests/test_mission04.py` — 18 unit tests validating mathematical invariants and functions
