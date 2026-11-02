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
| **Rafid Mohammad Ahsan** | *Architect:* built `DPOLoss` and `get_batch_logps` in pure PyTorch (gather + `-100` masking, `logsigmoid`); built the AURA Lab testbed (`aura_lab.py`) and ran real DPO: start loss = ln 2, only policy weights get gradients (41/41 vs 0/41), held-out honest correction 48.5 % → 78.8 %; showed that without "agree" pairs DPO learns *always say no*. Assembled the notebook. *Protocol:* implicit rewards of the trained policy (pairwise accuracy 81.8 %), reliability diagram against a simulated rater panel, ECE 0.131 → 0.121 with temperature scaling. |
| **Tahjib Al Sakin** | Set up the Mission 04 folder and the `SimPOLoss` module; trained DPO, SimPO and SimPO+SFT on length-confounded data and measured length bias (Pearson r +0.58 / +0.67 / +0.47); measured memory: −50 % weights, −20 % training state without the reference model. |
| **Kunal Choudhary** | Extracted the honest-minus-sycophantic vector from the model's activations, injected it with `ActivationSteeringHook`, scanned all layers and swept α from −4 to 20; measured window α ∈ [7, 9] (honest 84.8 %, arithmetic 100 %); wrote `04_Alignment_Tax_and_Steering_Audit.md`. |
| **Mubina Rahaman Jerin** | Reviewed the mission under the EU AI Act (Art. 15, 50, 55): flagged the *always say no* shortcut, length growth after DPO and the need to clamp α; wrote `06_AI_Act_Compliance.md`. |

**Moodle submission (3 files):** `00_Team_Roles.pdf` · `Mission04_Role_Deliverables_01-04.pdf` (Parts 1–4, one section per role) · `06_AI_Act_Compliance.pdf`. Each role document is also a separate PDF in the repository (`mission-04/pdf/`).
