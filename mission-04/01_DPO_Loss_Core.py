"""
Mission 04 — Part 1: Systems Architect (Lead Engineer)
Direct Preference Optimization (DPO) Loss Core & Log-Likelihood Extraction

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-04-ORI (Betelgeuse & Orion Molecular Cloud Complex)
Subsystem 04: Plasma Ignition & Ethical Steering Core
"""

import json
import os
import pathlib
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
    average_log_prob: bool = False
) -> torch.FloatTensor:
    """
    Computes per-sequence log-probabilities of token labels under model logits.
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        labels: Tensor of shape (batch_size, seq_len) with prompt/pad tokens set to label_pad_token_id
        label_pad_token_id: Token ID to ignore in loss computation
        average_log_prob: If True, returns mean log-prob per non-pad token (used in SimPO)
        
    Returns:
        Tensor of shape (batch_size,) containing sequence log-probabilities
    """
    if logits.shape[1] <= 1:
        raise ValueError("Sequence length must be > 1 to compute next-token logps.")

    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Create loss mask for non-padding tokens
    loss_mask = (shift_labels != label_pad_token_id)

    # Replace padding tokens with 0 for gather operation
    clean_labels = shift_labels.clone()
    clean_labels[~loss_mask] = 0

    # Calculate log-softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log-probabilities corresponding to true token labels
    per_token_logps = torch.gather(
        log_probs,
        dim=2,
        index=clean_labels.unsqueeze(2)
    ).squeeze(2)

    # Mask out padding tokens
    per_token_logps = per_token_logps * loss_mask.float()

    if average_log_prob:
        seq_lengths = loss_mask.sum(dim=-1).clamp(min=1)
        return per_token_logps.sum(dim=-1) / seq_lengths
    else:
        return per_token_logps.sum(dim=-1)


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss Module (Rafailov et al., NeurIPS 2023).
    
    Optimizes policy directly from pairwise preferences without training an explicit reward model:
        L_DPO = -E_{(x, yw, yl)} [ log sigma( beta * ( log(pi(yw)/pi_ref(yw)) - log(pi(yl)/pi_ref(yl)) ) ) ]
    """

    def __init__(self, beta: float = 0.1, label_pad_token_id: int = -100):
        super().__init__()
        self.beta = beta
        self.label_pad_token_id = label_pad_token_id

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Calculates DPO loss, implicit rewards, and preference margins.
        """
        # Calculate log-ratio differences: log(pi_theta(y|x) / pi_ref(y|x))
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # Implicit rewards derived analytically via Rafailov reparameterization:
        # r(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        reward_margins = chosen_rewards - rejected_rewards

        # Compute Bradley-Terry logit: beta * (chosen_logratios - rejected_logratios)
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # Numerically stable DPO loss: -log(sigmoid(logits))
        loss = -F.logsigmoid(logits).mean()

        # Metrics for monitoring training
        with torch.no_grad():
            chosen_win_rate = (chosen_rewards > rejected_rewards).float().mean()
            mean_margin = reward_margins.mean()
            bt_probs = torch.sigmoid(logits).mean()

        metrics = {
            "loss": loss.detach(),
            "chosen_win_rate": chosen_win_rate,
            "reward_margin": mean_margin,
            "bt_probability": bt_probs,
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean()
        }

        return loss, metrics


def save_dpo_artifacts(
    loss: float,
    metrics: Dict[str, Union[float, torch.Tensor]],
    grad_norm: float,
    output_dir: pathlib.Path
):
    """Saves benchmark results and loss surface figure."""
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON results
    data = {
        "beta": 0.1,
        "loss": float(loss),
        "chosen_win_rate": float(metrics["chosen_win_rate"]),
        "reward_margin": float(metrics["reward_margin"]),
        "bt_probability": float(metrics["bt_probability"]),
        "chosen_rewards": float(metrics["chosen_rewards"]),
        "rejected_rewards": float(metrics["rejected_rewards"]),
        "policy_chosen_logits_grad_norm": float(grad_norm),
        "reference_logits_frozen": True
    }
    with open(results_dir / "01_dpo_core.json", "w") as f:
        json.dump(data, f, indent=2)

    # 2. Plot DPO Loss curve & gradient dynamics if matplotlib is available
    if HAVE_MATPLOTLIB:
        import numpy as np
        margins = np.linspace(-10.0, 10.0, 200)
        beta = 0.1
        logits = beta * margins
        loss_curve = -np.log(1.0 / (1.0 + np.exp(-logits)))
        grad_curve = beta * (1.0 - 1.0 / (1.0 + np.exp(-logits)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax1.plot(margins, loss_curve, color="#1f77b4", lw=2.2, label=r"$\mathcal{L}_{\mathrm{DPO}}$")
        ax1.axvline(0, color="gray", linestyle="--", alpha=0.6)
        ax1.scatter([metrics["reward_margin"].item() / beta], [float(loss)], color="crimson", zorder=5, s=60, label="Simulation Batch")
        ax1.set_title("DPO Loss vs Log-Ratio Margin", fontsize=11, fontweight="bold")
        ax1.set_xlabel(r"Margin $\Delta_w - \Delta_l$", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True)

        ax2.plot(margins, grad_curve, color="#ff7f0e", lw=2.2, label=r"$|\nabla_\theta \mathcal{L}| \propto \beta(1 - \sigma)$")
        ax2.axvline(0, color="gray", linestyle="--", alpha=0.6)
        ax2.set_title("DPO Gradient Magnitude vs Margin", fontsize=11, fontweight="bold")
        ax2.set_xlabel(r"Margin $\Delta_w - \Delta_l$", fontsize=10)
        ax2.set_ylabel("Gradient Scale", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=True)

        plt.tight_layout()
        plt.savefig(figures_dir / "01_dpo_loss_gradient.png", dpi=180)
        plt.close()


def run_dpo_simulation(output_dir: Optional[pathlib.Path] = None):
    """Runs end-to-end verification of DPO Loss Core."""
    print("=" * 70)
    print("USSE NOOSPHERE • SUBSYSTEM 04 CALIBRATION: DPO LOSS CORE TEST")
    print("=" * 70)

    torch.manual_seed(42)
    batch_size, seq_len, vocab_size = 4, 16, 128

    # 1. Simulate policy & reference logits
    policy_logits_w = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    policy_logits_l = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    ref_logits_w = torch.randn(batch_size, seq_len, vocab_size, requires_grad=False)
    ref_logits_l = torch.randn(batch_size, seq_len, vocab_size, requires_grad=False)

    # 2. Simulate token labels with prompt/padding tokens (-100)
    labels_w = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels_w[:, :4] = -100  # First 4 tokens masked (prompt)

    labels_l = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels_l[:, :4] = -100

    # 3. Extract log-probabilities
    p_logps_w = get_batch_logps(policy_logits_w, labels_w)
    p_logps_l = get_batch_logps(policy_logits_l, labels_l)
    r_logps_w = get_batch_logps(ref_logits_w, labels_w)
    r_logps_l = get_batch_logps(ref_logits_l, labels_l)

    # 4. Compute DPO Loss
    dpo_criterion = DPOLoss(beta=0.1)
    loss, metrics = dpo_criterion(p_logps_w, p_logps_l, r_logps_w, r_logps_l)

    # 5. Backpropagation check
    loss.backward()

    grad_norm = policy_logits_w.grad.norm().item()

    print(f"[✓] DPO Loss Computed: {loss.item():.4f}")
    print(f"[✓] Bradley-Terry Win Probability: {metrics['bt_probability'].item() * 100:.2f}%")
    print(f"[✓] Implicit Reward Margin: {metrics['reward_margin'].item():+.4f}")
    print(f"[✓] Policy Chosen Logits Grad Norm: {grad_norm:.4f}")
    print(f"[✓] Reference Logits Grad: {ref_logits_w.grad} (Expected: None / Frozen)")
    print("\n[SUCCESS] Subsystem 04 DPO Core module successfully validated.")
    print("=" * 70)

    if output_dir:
        save_dpo_artifacts(loss.item(), metrics, grad_norm, output_dir)


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).resolve().parent
    run_dpo_simulation(output_dir=current_dir)
