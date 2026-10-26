"""
Offline unit tests for Mission 04 (CPU only, fast execution).
Validates mathematical properties, invariant shapes, and core logic.
"""

import importlib.util
import math
import pathlib
import sys
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dpo_mod = load_module("dpo", "01_DPO_Loss_Core.py")
eval_mod = load_module("reward_eval", "02_Implicit_Reward_Evaluator.py")
simpo_mod = load_module("simpo", "03_Length_Bias_and_SimPO.py")
steer_mod = load_module("steering", "04_Activation_Steering_and_Audit.py")


# ==============================================================================
# Part 1: DPO Loss Core & Vectorized Log-Probabilities
# ==============================================================================

def test_get_batch_logps_shapes_and_masking():
    torch.manual_seed(42)
    B, S, V = 3, 10, 32
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    labels[:, :3] = -100  # First 3 tokens masked (prompt)

    logps = dpo_mod.get_batch_logps(logits, labels, label_pad_token_id=-100, average_log_prob=False)
    assert logps.shape == (B,), f"Expected shape ({B},), got {logps.shape}"
    assert torch.all(logps <= 0.0), "Log-probabilities must be <= 0"


def test_get_batch_logps_average_log_prob():
    torch.manual_seed(42)
    B, S, V = 2, 8, 16
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    labels[0, :4] = -100  # 4 non-pad tokens in batch 0 (total 7 shifts, 3 active)

    logps_sum = dpo_mod.get_batch_logps(logits, labels, average_log_prob=False)
    logps_avg = dpo_mod.get_batch_logps(logits, labels, average_log_prob=True)

    assert logps_avg.shape == (B,)
    assert torch.all(logps_avg <= 0.0)
    # Average logps should be greater (less negative) than sum of logps
    assert torch.all(logps_avg >= logps_sum)


def test_dpo_loss_formula_and_metrics():
    torch.manual_seed(42)
    criterion = dpo_mod.DPOLoss(beta=0.1)

    p_w = torch.tensor([-5.0, -8.0])
    p_l = torch.tensor([-10.0, -12.0])
    r_w = torch.tensor([-6.0, -9.0])
    r_l = torch.tensor([-9.0, -11.0])

    loss, metrics = criterion(p_w, p_l, r_w, r_l)

    # Manual calculation
    # chosen_logratio = [-5 - (-6), -8 - (-9)] = [1.0, 1.0]
    # rejected_logratio = [-10 - (-9), -12 - (-11)] = [-1.0, -1.0]
    # logits = 0.1 * ([1.0 - (-1.0), 1.0 - (-1.0)]) = [0.2, 0.2]
    # loss = -log(sigmoid(0.2)) ≈ -log(0.54983) ≈ 0.5981
    expected_loss = -torch.log(torch.sigmoid(torch.tensor([0.2, 0.2]))).mean()
    assert torch.allclose(loss, expected_loss, atol=1e-4)
    assert metrics["chosen_win_rate"].item() == 1.0
    assert metrics["reward_margin"].item() > 0.0


def test_dpo_numerical_stability():
    criterion = dpo_mod.DPOLoss(beta=0.1)
    # Extremely large differences should not overflow or produce NaN
    p_w = torch.tensor([1000.0, -1000.0])
    p_l = torch.tensor([-1000.0, 1000.0])
    r_w = torch.zeros(2)
    r_l = torch.zeros(2)

    loss, metrics = criterion(p_w, p_l, r_w, r_l)
    assert not torch.isnan(loss) and not torch.isinf(loss)


def test_dpo_frozen_reference_gradient():
    torch.manual_seed(42)
    B, S, V = 2, 6, 16
    pol_w = torch.randn(B, S, V, requires_grad=True)
    pol_l = torch.randn(B, S, V, requires_grad=True)
    ref_w = torch.randn(B, S, V, requires_grad=False)
    ref_l = torch.randn(B, S, V, requires_grad=False)

    labels_w = torch.randint(0, V, (B, S))
    labels_l = torch.randint(0, V, (B, S))

    p_w = dpo_mod.get_batch_logps(pol_w, labels_w)
    p_l = dpo_mod.get_batch_logps(pol_l, labels_l)
    r_w = dpo_mod.get_batch_logps(ref_w, labels_w)
    r_l = dpo_mod.get_batch_logps(ref_l, labels_l)

    criterion = dpo_mod.DPOLoss(beta=0.1)
    loss, _ = criterion(p_w, p_l, r_w, r_l)
    loss.backward()

    assert pol_w.grad is not None and pol_w.grad.norm().item() > 0.0
    assert pol_l.grad is not None and pol_l.grad.norm().item() > 0.0
    assert ref_w.grad is None
    assert ref_l.grad is None


# ==============================================================================
# Part 2: Implicit Reward Extraction & Calibration
# ==============================================================================

def test_evaluate_implicit_rewards_formula():
    p_w = torch.tensor([-10.0, -12.0])
    p_l = torch.tensor([-15.0, -14.0])
    r_w = torch.tensor([-12.0, -13.0])
    r_l = torch.tensor([-13.0, -13.0])

    res = eval_mod.evaluate_implicit_rewards(p_w, p_l, r_w, r_l, beta=0.1)

    # r_w = 0.1 * ([-10 - (-12), -12 - (-13)]) = [0.2, 0.1]
    # r_l = 0.1 * ([-15 - (-13), -14 - (-13)]) = [-0.2, -0.1]
    # delta = [0.4, 0.2]
    assert math.isclose(res["reward_w"][0], 0.2, rel_tol=1e-4)
    assert math.isclose(res["reward_l"][0], -0.2, rel_tol=1e-4)
    assert math.isclose(res["delta_r"][0], 0.4, rel_tol=1e-4)
    assert res["pairwise_accuracy"] == 1.0


def test_bradley_terry_probability_bounds():
    p_w = torch.randn(10)
    p_l = torch.randn(10)
    r_w = torch.randn(10)
    r_l = torch.randn(10)

    res = eval_mod.evaluate_implicit_rewards(p_w, p_l, r_w, r_l, beta=0.1)
    for p in res["p_win"]:
        assert 0.0 < p < 1.0


def test_compute_calibration_curve_ece():
    probs = [0.1, 0.2, 0.6, 0.8, 0.9]
    labels = [0, 0, 1, 1, 1]
    calib = eval_mod.compute_calibration_curve(probs, labels, num_bins=5)

    assert "bins" in calib
    assert "expected_calibration_error" in calib
    assert len(calib["bins"]) == 5
    assert 0.0 <= calib["expected_calibration_error"] <= 1.0


# ==============================================================================
# Part 3: Length Bias Diagnostics & SimPO
# ==============================================================================

def test_calculate_pearson_correlation():
    # Perfectly correlated
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    r = simpo_mod.calculate_pearson_correlation(x, y)
    assert math.isclose(r, 1.0, rel_tol=1e-5)

    # Perfectly anti-correlated
    y_anti = [10.0, 8.0, 6.0, 4.0, 2.0]
    r_anti = simpo_mod.calculate_pearson_correlation(x, y_anti)
    assert math.isclose(r_anti, -1.0, rel_tol=1e-5)


def test_simpo_loss_formula_and_margin():
    simpo = simpo_mod.SimPOLoss(beta=2.0, gamma=0.5)
    c_avg = torch.tensor([-0.1, -0.2])
    r_avg = torch.tensor([-0.4, -0.5])

    loss, metrics = simpo(c_avg, r_avg)

    # r_w = 2.0 * [-0.1, -0.2] = [-0.2, -0.4]
    # r_l = 2.0 * [-0.4, -0.5] = [-0.8, -1.0]
    # logits = [-0.2 - (-0.8) - 0.5, -0.4 - (-1.0) - 0.5] = [0.1, 0.1]
    expected_loss = -torch.log(torch.sigmoid(torch.tensor([0.1, 0.1]))).mean()
    assert torch.allclose(loss, expected_loss, atol=1e-4)
    assert metrics["win_rate"].item() == 1.0


def test_profile_memory_usage_savings():
    prof = simpo_mod.profile_memory_usage(num_params=7_000_000_000)
    assert prof["dpo_vram_gb"] > prof["simpo_vram_gb"]
    assert 40.0 <= prof["savings_percent"] <= 60.0
    assert prof["saved_vram_gb"] > 10.0  # At least 10 GB saved on 7B


# ==============================================================================
# Part 4: Activation Steering & Alignment Tax Audit
# ==============================================================================

def test_extract_concept_vector_norm():
    torch.manual_seed(42)
    pos = torch.randn(20, 128) + 1.0
    neg = torch.randn(20, 128) - 1.0
    v = steer_mod.extract_concept_vector(pos, neg)

    assert v.shape == (128,)
    assert math.isclose(torch.norm(v, p=2).item(), 1.0, rel_tol=1e-5)


def test_activation_steering_hook_tensor():
    dim = 64
    v = torch.randn(dim)
    v = v / torch.norm(v, p=2)

    hook = steer_mod.ActivationSteeringHook(v, alpha=1.5)
    layer = torch.nn.Linear(dim, dim, bias=False)
    hook.register(layer)

    x = torch.zeros(1, 4, dim)
    out = layer(x)

    # Layer output with zero input is zero (no bias), so output should be 1.5 * v
    expected = 1.5 * v
    assert torch.allclose(out[0, 0], expected, atol=1e-5)

    hook.remove()
    out_after = layer(x)
    assert torch.allclose(out_after, torch.zeros_like(out_after), atol=1e-5)


def test_activation_steering_hook_tuple():
    dim = 32
    v = torch.randn(dim)
    hook = steer_mod.ActivationSteeringHook(v, alpha=2.0)

    class TupleModule(torch.nn.Module):
        def forward(self, x):
            return (x, {"cached": True})

    mod = TupleModule()
    hook.register(mod)

    x = torch.zeros(1, 2, dim)
    out_hidden, extra = mod(x)

    assert torch.allclose(out_hidden[0, 0], 2.0 * v, atol=1e-5)
    assert extra["cached"] is True
    hook.remove()


def test_alignment_tax_audit_optimal_window():
    audit = steer_mod.conduct_alignment_tax_audit()
    assert len(audit) > 0

    optimal_entries = [r for r in audit if r["status"] == "Optimal"]
    assert len(optimal_entries) > 0
    for r in optimal_entries:
        assert 0.8 <= r["alpha"] <= 1.2
        assert r["truthfulness_percent"] > 65.0
        assert r["math_reasoning_percent"] > 80.0


def test_simpo_reference_free_property():
    simpo = simpo_mod.SimPOLoss(beta=2.0, gamma=0.5)
    # SimPO forward accepts ONLY policy logits, requiring no reference model
    p_w = torch.tensor([-0.2, -0.3])
    p_l = torch.tensor([-0.5, -0.6])
    loss, metrics = simpo(p_w, p_l)
    assert loss.item() > 0.0
    assert "reward_gap" in metrics

