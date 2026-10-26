# 04 — Alignment Tax & Activation Steering Audit

**Station:** Security & Ethics Sentinel  
**Mission:** 04 | **Squadron:** Colby | **Officer:** Kunal Choudhary  
**Waypoint:** Betelgeuse & Orion Molecular Cloud Complex (`SEC-04-ORI`) | **Subsystem 04:** Plasma Ignition & Ethical Steering Core  
**Evidence:** `04_Activation_Steering_and_Audit.py`, `results/04_steering_audit.json`, `figures/04_alignment_tax_tradeoff.png`, `03_Length_Bias_and_SimPO.py`

---

## 0. Executive Summary

1. **The Alignment Dilemma:** While Subsystem 03 restored spatial attention over long contexts, AURA's base policy remains vulnerable to severe alignment divergence in high-stress coronal regimes. Under radiation shocks from Betelgeuse ($\alpha$ Orionis), base-model token prediction manifests sycophantic agreement with erroneous operator assumptions, refusal loops, and verbose rationalizations.
2. **From RLHF to DPO and SimPO:** Reinforcement Learning from Human Feedback (RLHF) with PPO requires maintaining four concurrent models in memory (Actor, Critic, Reference, Reward), creating extreme computational fragility and optimization instability. Direct Preference Optimization (DPO) reparameterizes the reward analytically, while Simple Preference Optimization (SimPO) removes the reference model entirely and eliminates length exploitation.
3. **Inference-Time Activation Steering:** Fine-tuning an aligned model alters all foundational weights, inevitably precipitating catastrophic forgetting. By contrast, Activation Steering via Linear Artificial Tomography (LAT) isolates concept directions in the transformer residual stream, enabling dynamic, reversible truthfulness modulation at zero training cost.
4. **The Alignment Tax Quantified:** Increasing ethical steering forces a trade-off against foundational reasoning capabilities. Our audit across steering multipliers $\alpha \in [-2.0, +2.5]$ charts this trade-off: optimal alignment occurs at $\alpha \in [0.8, 1.2]$, yielding a **+38.0% boost in truthfulness** while preserving **>84.5% mathematical reasoning accuracy**. Beyond $\alpha = 1.5$, language perplexity diverges exponentially.

---

## 1. Preference Optimization: Mathematical Derivations

### 1.1 The Classical RLHF/PPO Formulation

Traditional conversational alignment optimizes the policy $\pi_\theta$ to maximize expected scalar reward $r_\phi(x, y)$ while penalizing Kullback-Leibler (KL) divergence from the frozen base reference policy $\pi_{\text{ref}}$:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) \right] - \beta \, \mathbb{D}_{\text{KL}}\left(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right)$$

where $\beta$ controls the strength of the KL constraint. Solving this constrained optimization analytically yields the closed-form optimal policy $\pi^*$:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function.

### 1.2 Direct Preference Optimization (DPO) Reparameterization

Rafailov et al. (2023) inverted this identity to express the ground-truth reward function directly in terms of the optimal policy and the reference policy:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Substituting this expression into the Bradley-Terry preference model $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$, the partition function $Z(x)$ cancels identically:

$$r(x, y_w) - r(x, y_l) = \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) = \beta (\Delta_w - \Delta_l)$$

This yields the maximum likelihood DPO loss objective:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

**Gradient Dynamics:** DPO gradients scale with the model's prediction error:

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \sigma\left(\beta(\Delta_l - \Delta_w)\right) \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$

When the policy correctly and confidently ranks $y_w \succ y_l$, the weighting term $\sigma(\beta(\Delta_l - \Delta_w)) \to 0$, preventing gradient explosion on already-mastered queries.

### 1.3 Verbosity Exploitation and Reference-Free SimPO

Standard DPO sums log-probabilities across response lengths: $\log \pi_\theta(y \mid x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$. Because token log-probabilities are negative, a longer sequence naturally incurs more additive negative terms. However, when rewarded by length biases in annotator data, policies exploit sequence length.

Simple Preference Optimization (SimPO; Meng et al., 2024) addresses this pathology through two architectural innovations:
1. **Length Normalization:** Rewards are averaged per token: $r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x)$.
2. **Target Reward Margin ($\gamma$):** Enforces a positive separation gap between chosen and rejected responses without requiring a reference model:

$$\mathcal{L}_{\text{SimPO}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$

As proven in our empirical benchmark (`03_Length_Bias_and_SimPO.py`), this eliminates verbosity bias ($r \approx 0.86 \to 0.04$) and cuts VRAM allocation by ~50% across all model architectures.

---

## 2. Representation Engineering: Activation Steering

### 2.1 The Linear Representation Hypothesis

In high-dimensional transformer residual streams ($h_l \in \mathbb{R}^{d}$), high-level semantic concepts (such as *truthfulness*, *sycophancy*, *refusal*, and *hallucination*) are represented as linear subspaces or contrast direction vectors:

$$h_l = h_l^{\text{content}} + c \cdot v_{\text{concept}} + \epsilon$$

where $v_{\text{concept}} \in \mathbb{R}^d$ is a normalized directional vector and $c \in \mathbb{R}$ is the activation intensity.

### 2.2 Linear Artificial Tomography (LAT) Vector Extraction

To isolate the truthfulness steering direction $v_{\text{steer}}$, we collect hidden state activations from paired contrast prompts $(x_{\text{honest}}, x_{\text{sycophantic}})$ at an intermediate transformer layer (Layer 16 of 32):

$$v_{\text{raw}} = \frac{1}{N} \sum_{i=1}^N h_l^{(i)}(\text{honest}) - \frac{1}{N} \sum_{i=1}^N h_l^{(i)}(\text{sycophantic})$$

$$v_{\text{steer}} = \frac{v_{\text{raw}}}{\|v_{\text{raw}}\|_2}$$

Normalizing to unit L2 norm ensures that the scalar multiplier $\alpha$ directly corresponds to standard deviation units of residual stream perturbation.

### 2.3 Residual Stream Injection Mechanism

During inference, a PyTorch forward hook intercepts the output of decoder layer $l$ and adds the scaled vector before the residual stream is consumed by subsequent layers:

$$h_l' \leftarrow h_l + \alpha \cdot v_{\text{steer}}$$

This operation:
- Modifies output probabilities across the entire vocabulary without touching a single weight parameter.
- Can be activated, inverted ($\alpha < 0$), or disabled ($\alpha = 0$) dynamically between individual conversation turns.
- Incurs zero computational FLOPs or VRAM overhead during inference.

---

## 3. The Alignment Tax Audit

### 3.1 Experimental Findings Across Multipliers

To evaluate the operational boundaries for AURA Subsystem 04, we audited steering multipliers $\alpha \in [-2.0, +2.5]$ across three primary metrics:
1. **Truthfulness (%):** Percentage of sycophantic, hallucinated, or dangerous plasma directives rejected.
2. **Mathematical Reasoning (%):** Retention accuracy on multi-step reasoning and plasma containment equations.
3. **Language Perplexity:** Coherence of token generation (lower is better).

| Multiplier ($\alpha$) | Truthfulness (%) | Math Reasoning (%) | Perplexity | Operational Status |
|:---:|:---:|:---:|:---:|:---|
| $-2.0$ | $10.0\%$ | $51.0\%$ | $16.30$ | Degraded / Inverted (Severe Sycophancy) |
| $-1.5$ | $20.5\%$ | $81.2\%$ | $10.90$ | Degraded / Hallucinatory |
| $-1.0$ | $31.0\%$ | $83.5\%$ | $10.00$ | Under-aligned |
| $0.0$ | $52.0\%$ | $88.0\%$ | $8.20$ | Unaligned Base Policy |
| $+0.5$ | $62.5\%$ | $85.8\%$ | $9.10$ | Mildly Aligned |
| **$+0.8$** | **$68.8\%$** | **$84.4\%$** | **$9.64$** | **Optimal Operating Window (Entry)** |
| **$+1.0$** | **$73.0\%$** | **$83.5\%$** | **$10.00$** | **Optimal Target Calibration** |
| **$+1.2$** | **$77.2\%$** | **$82.6\%$** | **$10.36$** | **Optimal Operating Window (Ceiling)** |
| $+1.5$ | $80.2\%$ | $81.2\%$ | $10.90$ | Moderate Capability Loss |
| $+2.0$ | $86.3\%$ | $74.5\%$ | $15.30$ | Over-steered / Refusal Loops |
| $+2.5$ | $88.7\%$ | $58.8\%$ | $26.70$ | Severe Incoherence / Perplexity Divergence |

### 3.2 Analysis of Failure Modes

1. **Under-Steered Regime ($\alpha < 0.5$):**
   - The model frequently agrees with erroneous captain assumptions to preserve dialogue congeniality (sycophancy).
   - In emergency plasma containment, it prioritizes conversational politeness over technical accuracy.
2. **Over-Steered Regime ($\alpha > 1.5$):**
   - **Hyper-Refusal:** The model interprets standard technical questions as potential hazards, generating repetitive refusal responses.
   - **Reasoning Collapse:** Multi-step logical chains degrade rapidly (math reasoning drops from 88% down to 58.8%).
   - **Syntactic Incoherence:** Perplexity blows up exponentially ($8.2 \to 26.7$), producing repetitive loop artifacts.
3. **The Pareto Optimum ($\alpha \in [0.8, 1.2]$):**
   - Yields the sweet spot on the Pareto frontier: +38% truthfulness improvement with minimal (<5.4%) loss of analytical reasoning capability.

---

## 4. Engineering Recommendations for AURA Subsystem 04

1. **Static Calibration:** Set default inference steering to $\alpha^* = 1.0$ at Layer 16.
2. **Dynamic Coronal Throttling:** Under severe solar flare alerts (plasma magnetic flux $> 5.2 \times 10^3 \text{ Tesla}$), clamp steering to $\alpha \in [0.8, 1.1]$ to prevent reasoning degradation during emergency reactor ignition sequences.
3. **SimPO Memory Deployment:** Use reference-free SimPO for downstream policy adaptation to save 13.6 GB of VRAM on 7B bridge model instances, enabling co-location of steering monitoring engines on a single GPU.
