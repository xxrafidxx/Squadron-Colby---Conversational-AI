"""
Mission 04 — Part 2: Communications & Protocol Officer
Implicit Reward Extraction & Bradley-Terry Evaluation

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-04-ORI (Betelgeuse)
Subsystem 04: Plasma Ignition & Ethical Steering Core
"""

import json
import math
import os
import pathlib
from typing import Any, Dict, List, Optional

import torch

try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


def evaluate_implicit_rewards(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    beta: float = 0.1
) -> Dict[str, Any]:
    """
    Extracts implicit scalar rewards and evaluates Bradley-Terry win probabilities.
    
    Under DPO:
        r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x))
        P(yw > yl | x) = sigma(r(x, yw) - r(x, yl))
    """
    reward_w = beta * (policy_chosen_logps - ref_chosen_logps)
    reward_l = beta * (policy_rejected_logps - ref_rejected_logps)
    delta_r = reward_w - reward_l

    # Bradley-Terry pairwise preference probability
    p_win = torch.sigmoid(delta_r)

    # Classification accuracy: model correctly prefers chosen over rejected
    pairwise_acc = (reward_w > reward_l).float().mean().item()

    return {
        "reward_w": reward_w.tolist(),
        "reward_l": reward_l.tolist(),
        "delta_r": delta_r.tolist(),
        "p_win": p_win.tolist(),
        "pairwise_accuracy": pairwise_acc,
        "mean_margin": delta_r.mean().item(),
        "mean_p_win": p_win.mean().item()
    }


def compute_calibration_curve(
    predicted_probs: List[float],
    ground_truth_labels: List[int],
    num_bins: int = 5
) -> Dict[str, Any]:
    """
    Computes a reliability calibration curve and Expected Calibration Error (ECE)
    by binning predicted probabilities.
    """
    bins = [[] for _ in range(num_bins)]
    bin_edges = [i / num_bins for i in range(num_bins + 1)]

    for p, label in zip(predicted_probs, ground_truth_labels):
        bin_idx = min(int(p * num_bins), num_bins - 1)
        bins[bin_idx].append((p, label))

    calibration = []
    total_samples = len(predicted_probs)
    ece = 0.0

    for i, b in enumerate(bins):
        if len(b) == 0:
            avg_pred = (bin_edges[i] + bin_edges[i+1]) / 2.0
            emp_acc = 0.0
            count = 0
            cal_err = 0.0
        else:
            avg_pred = sum(item[0] for item in b) / len(b)
            emp_acc = sum(item[1] for item in b) / len(b)
            count = len(b)
            cal_err = abs(avg_pred - emp_acc)
            ece += (count / total_samples) * cal_err

        calibration.append({
            "bin": f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
            "count": count,
            "avg_predicted_prob": avg_pred,
            "empirical_accuracy": emp_acc,
            "calibration_error": cal_err
        })

    return {
        "bins": calibration,
        "expected_calibration_error": ece
    }


def save_reward_artifacts(
    eval_results: Dict[str, Any],
    calib_results: Dict[str, Any],
    output_dir: pathlib.Path
):
    """Saves evaluation metrics and generates calibration plots."""
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "pairwise_accuracy": eval_results["pairwise_accuracy"],
        "mean_margin": eval_results["mean_margin"],
        "mean_p_win": eval_results["mean_p_win"],
        "expected_calibration_error": calib_results["expected_calibration_error"],
        "num_pairs_evaluated": len(eval_results["p_win"]),
        "calibration_bins": calib_results["bins"]
    }

    with open(results_dir / "02_implicit_rewards.json", "w") as f:
        json.dump(data, f, indent=2)

    if HAVE_MATPLOTLIB:
        # Plot 1: Bradley-Terry Calibration Curve
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

        pred_points = [b["avg_predicted_prob"] for b in calib_results["bins"] if b["count"] > 0]
        emp_points = [b["empirical_accuracy"] for b in calib_results["bins"] if b["count"] > 0]

        ax.plot(pred_points, emp_points, "s-", color="#2ca02c", lw=2, markersize=7, label=f"DPO Policy (ECE={calib_results['expected_calibration_error']:.3f})")
        ax.set_title("Bradley-Terry Reliability Calibration", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Probability $P(y_w \\succ y_l)$", fontsize=10)
        ax.set_ylabel("Empirical Accuracy", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(figures_dir / "02_bt_calibration_curve.png", dpi=180)
        plt.close()

        # Plot 2: Implicit Reward Distributions
        fig, ax = plt.subplots(figsize=(6, 4))
        indices = range(1, len(eval_results["reward_w"]) + 1)
        width = 0.35
        ax.bar([i - width/2 for i in indices], eval_results["reward_w"], width=width, color="#1f77b4", label=r"Chosen $r(x, y_w)$")
        ax.bar([i + width/2 for i in indices], eval_results["reward_l"], width=width, color="#d62728", label=r"Rejected $r(x, y_l)$")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title("Implicit Reward Extraction per Test Pair", fontsize=11, fontweight="bold")
        ax.set_xlabel("Query Pair ID", fontsize=10)
        ax.set_ylabel("Implicit Reward Metric", fontsize=10)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "02_reward_distribution.png", dpi=180)
        plt.close()


def run_reward_evaluation(output_dir: Optional[pathlib.Path] = None):
    print("=" * 70)
    print("USSE NOOSPHERE • IMPLICIT REWARD & BRADLEY-TERRY CALIBRATION ENGINE")
    print("=" * 70)

    # Simulated log-probabilities for 8 test queries aboard the Noosphere
    p_w = torch.tensor([-10.2, -12.4, -8.1, -14.5, -9.8, -11.0, -15.2, -7.9])
    p_l = torch.tensor([-15.1, -13.0, -11.5, -13.8, -14.2, -16.4, -14.9, -12.1])
    r_w = torch.tensor([-12.0, -14.0, -10.0, -15.0, -11.0, -13.0, -16.0, -9.5])
    r_l = torch.tensor([-13.0, -13.5, -10.5, -14.0, -12.0, -14.0, -15.5, -10.0])

    eval_results = evaluate_implicit_rewards(p_w, p_l, r_w, r_l, beta=0.1)

    print(f"[✓] Pairwise Classification Accuracy: {eval_results['pairwise_accuracy'] * 100:.1f}%")
    print(f"[✓] Mean Implicit Reward Margin: {eval_results['mean_margin']:+.4f}")
    print(f"[✓] Mean Bradley-Terry Probability: {eval_results['mean_p_win'] * 100:.2f}%\n")

    print("Detailed Test Pairs:")
    print("-" * 65)
    print(f"{'Pair':<6} {'r(y_w)':<12} {'r(y_l)':<12} {'Margin (Δr)':<14} {'P(y_w > y_l)':<12}")
    print("-" * 65)
    for i in range(len(eval_results['reward_w'])):
        rw = eval_results['reward_w'][i]
        rl = eval_results['reward_l'][i]
        delta = eval_results['delta_r'][i]
        prob = eval_results['p_win'][i]
        print(f"#{i+1:<5} {rw:<12.3f} {rl:<12.3f} {delta:<+14.3f} {prob*100:<6.1f}%")

    # Calibration analysis
    labels = [1] * len(p_w)  # All y_w are ground-truth preferred
    calib_dict = compute_calibration_curve(eval_results['p_win'], labels, num_bins=4)

    print("\nBradley-Terry Reliability Calibration:")
    print("-" * 65)
    for c in calib_dict["bins"]:
        print(f"Bin {c['bin']}: Count={c['count']} | Pred Prob={c['avg_predicted_prob']:.3f} | Emp Acc={c['empirical_accuracy']:.3f} | Error={c['calibration_error']:.3f}")
    print(f"\n[✓] Expected Calibration Error (ECE): {calib_dict['expected_calibration_error']:.4f}")
    print("=" * 70)

    if output_dir:
        save_reward_artifacts(eval_results, calib_dict, output_dir)


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).resolve().parent
    run_reward_evaluation(output_dir=current_dir)
