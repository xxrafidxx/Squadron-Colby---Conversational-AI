# 06 — EU AI Act & Security Review (Mission 04)

**Station:** AI Security Officer  
**Mission:** 04 | **Squadron:** Colby | **Officer:** Mubina Rahaman Jerin  
**Scope:** `01`–`04` (DPO loss core, implicit rewards, SimPO, activation steering & alignment audit)  
**Legal basis:** Regulation (EU) 2024/1689 (EU AI Act), as amended by the Digital Omnibus  
**Review date:** 2026-11-02  

---

## 1. Classification

| Question | Answer |
|---|---|
| Is this code an "AI system" (Art. 3(1))? | **Not in isolation.** The modules in Mission 04 are algorithmic optimization components (loss functions, preference evaluators, hooks, and audit benchmarks) tested on a small research model (AURA Lab, 350k parameters, synthetic language). They are not placed on the market and serve no end users. |
| Would AURA utilizing Subsystem 04 be in scope? | **Yes.** A deployed conversational assistant aboard the *USSE Noosphere* utilizing these alignment modules constitutes an AI system under Art. 3(1). |
| What risk tier applies to AURA? | Under Annex III, general conversational assistants are not classified as high-risk unless they control safety components of critical physical infrastructure. However, because Subsystem 04 interfaces with *Plasma Ignition*, safety-critical operational modes must satisfy high-risk standards voluntarily. |
| Are we a GPAI model provider (Art. 53)? | **No.** We evaluate and audit open-weights alignment techniques. We do not place a foundation model on the EU market. However, GPAI systemic risk obligations (Art. 55) guide our capability-retention audits. |
| Is research and development exempt? | Yes, Art. 2(6) and Art. 2(8) exclude AI systems and models specifically developed and put into service for the sole purpose of scientific research and development prior to placing on the market. |

---

## 2. Why Mission 04 Directly Touches the AI Act

Preference optimization and activation steering directly address three mandatory pillars of the AI Act:

| Article | Legal Obligation | Direct Connection to Mission 04 |
|---|---|---|
| **Art. 15(1)** | High-risk AI systems must achieve appropriate levels of **accuracy, robustness, and cybersecurity**. | Unaligned base models exhibit sycophancy and hallucinations under stress. Measured on AURA Lab: DPO raises honest correction 48.5 % → 78.8 %, steering 48.5 % → 84.8 %, both without loss on arithmetic. |
| **Art. 15(4)–(5)** | Resilience against **errors and inconsistencies** and against **attempts to exploit vulnerabilities**. | Verbosity exploitation (length bias) is a form of reward hacking where models game evaluation metrics by outputting bloatware. We measured it (`03`): DPO doubled answer length; SimPO's length normalisation alone did not remove the bias on our data. |
| **Art. 50(1)** | **Transparency obligations** for systems interacting directly with natural persons. | Conversational systems must clearly disclose their AI nature and refrain from deceptive sycophantic behavior designed to manipulate human operators. |
| **Art. 55(1)** | GPAI models with systemic risk must undergo **adversarial testing and capability audits**. | The *Alignment Tax Audit* (`04`) measures where steering starts to cost capability (true-fact confirmation from α = 14, arithmetic from α = 17). |

---

## 3. Compliance Checklist for Mission 04

| # | Requirement | Status | Verification & Evidence |
|:---:|---|:---:|---|
| 1 | **No personal data** | ✓ | AURA Lab uses an invented language (objects, colours, sums); no personal data is processed. |
| 2 | **Reproducibility** | ✓ | Fixed seeds in every script; the base model is cached; `05_Run_Log.txt` holds the full output; 19 tests pass. |
| 3 | **Results measured, not assumed** | ✓ | All reported numbers come from runs on the AURA Lab model (README, run log, `results/*.json`). Limits are stated: small model, one seed. |
| 4 | **Robustness against shortcuts (Art. 15)** | ⚠ found & fixed | DPO trained only on "correct the user" pairs learned *always say no*: true facts confirmed 0 %. Adding "agree" pairs fixed it (100 %). Training data for AURA must contain both. |
| 5 | **Reward hacking via length (Art. 15)** | ⚠ open | DPO doubled answer length (7 → 14 words) even without verbose training data. On length-confounded data SimPO did not remove the length bias (r = +0.67 vs +0.58 for DPO). Length must be monitored after any preference training. |
| 6 | **Steering safeguards** | ✓ | Measured window α ∈ [7, 9] at block 2. Over-steering (α ≥ 14) makes the model deny true facts; negative α makes it more sycophantic (12.1 % honest at α = −4). α must be clamped in code. |
| 7 | **Capability retention (Art. 55(1)(a))** | ✓ | Inside the window arithmetic stays at 100 % and perplexity rises only 1.479 → 1.502. The audit tracks three metrics so over-correction is visible. |
| 8 | **Calibration** | ✓ | Reward-model ECE 0.131 against a simulated rater panel; temperature scaling lowers it to 0.121 on held-out judgements. |
| 9 | **Energy / resources** | ✓ | SimPO removes the reference model: −50 % weight memory, −20 % measured training state, −14 % analytic for a 7B model. |

## 4. Operational Protocols Before Deployment

1. **Clamp the steering coefficient** to the window measured for the deployed model and layer (here α ∈ [7, 9] at block 2); reject negative α. Re-measure after every model update.
2. **Protect the steering vector like model weights:** with the sign flipped it makes the model sycophantic (Art. 15(5), integrity against manipulation).
3. **Audit with three metrics** after every alignment change: target behaviour, its mirror (true-fact confirmation) and an unrelated capability.
4. **Check answer length** after preference training; length gains are a known form of reward hacking.
5. **Transparency (Art. 50):** AURA keeps telling users that they talk to an AI.
6. **Keep the evidence:** `05_Run_Log.txt`, `results/*.json` and the figures form the technical file for this subsystem.

---

*Academic compliance assessment for BTU Cottbus-Senftenberg course submission. Not formal legal counsel.*
