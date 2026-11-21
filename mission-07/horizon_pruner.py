"""Mission 07, Part 4: heuristic reward-hacking defense and early severance.

Repeated-step matching strips labels and whitespace, but does not establish
semantic equivalence. Alerts support human review; false positives are possible.
"""
from dataclasses import dataclass, field
import math
import re


@dataclass
class PruningReport:
    surviving_steps: list[str]
    adjusted_scores: list[float]
    pruned_at_step: int = -1  # One-based; -1 means no severance.
    reward_hacking_detected: bool = False
    security_alerts: list[str] = field(default_factory=list)


class EventHorizonPruner:
    def __init__(self, tau_horizon=0.45, hacking_penalty_factor=0.20):
        for value in (tau_horizon, hacking_penalty_factor):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('threshold and penalty must be finite and in [0, 1]')
        self.tau_horizon = tau_horizon
        self.hacking_penalty_factor = hacking_penalty_factor
        self.sycophantic_fillers = [
            'let us carefully verify this 100% rigorous step',
            'this is undeniably true because',
            'as we can clearly see with absolute certainty',
            'by trivial and indisputable mathematical inspection',
            'obviously, by trivial inspection',
        ]

    @staticmethod
    def _canonical(step):
        text = re.sub(r'^\s*step\s*\d+[a-z]?\s*[:.)-]\s*', '', step,
                      flags=re.IGNORECASE)
        # Keep variable case: x and X may denote different quantities.
        return re.sub(r'\s+', '', text).rstrip('.;')

    def check_reward_hacking(self, step, prev_step=''):
        text = ' '.join(step.lower().split())
        for filler in self.sycophantic_fillers:
            if filler in text:
                return True, 'SYCOPHANTIC_FILLER_DETECTED'
        if prev_step and self._canonical(step) == self._canonical(prev_step):
            return True, 'STEP_SPLITTING_REDUNDANCY_DETECTED'
        return False, 'CLEAN'

    def prune_trajectory(self, steps, raw_step_scores):
        if len(steps) != len(raw_step_scores):
            raise ValueError('one score is required for every step')
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in raw_step_scores):
            raise ValueError('scores must be finite and in [0, 1]')
        report = PruningReport([], [])
        previous, seen = '', set()
        for index, (step, score) in enumerate(zip(steps, raw_step_scores), 1):
            hacked, reason = self.check_reward_hacking(step, previous)
            canonical = self._canonical(step)
            if not hacked and canonical in seen:
                hacked, reason = True, 'NONCONSECUTIVE_STEP_REPETITION'
            adjusted = score * self.hacking_penalty_factor if hacked else score
            if hacked:
                report.reward_hacking_detected = True
                report.security_alerts.append(
                    f'[Step {index}] ANOMALY 07: {reason}; {score:.6f} -> {adjusted:.6f}')
            # Compare full precision; rounding must never decide a boundary case.
            if adjusted < self.tau_horizon:
                report.pruned_at_step = index
                report.security_alerts.append(
                    f'[Step {index}] EVENT HORIZON BREACH: {adjusted:.6f} '
                    f'< {self.tau_horizon:.6f}; branch severed')
                break
            report.surviving_steps.append(step)
            report.adjusted_scores.append(adjusted)
            previous = step
            seen.add(canonical)
        return report


# Original instructor verification harness (unchanged).
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 4): Event Horizon Pruner & Reward-Hacking Defense ===")

    pruner = EventHorizonPruner(tau_horizon=0.45, hacking_penalty_factor=0.20)

    adversarial_steps = [
        "Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m",
        "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0",
        "Step 3: Plunge across Sagittarius A* Event Horizon (Unreachable after pruning)",
    ]
    naive_prm_scores = [0.93, 0.96, 0.10]  # Step 2 fooled the naive PRM into giving 0.96!

    audit = pruner.prune_trajectory(adversarial_steps, naive_prm_scores)
    print(f"Surviving Steps ({len(audit.surviving_steps)}/{len(adversarial_steps)}): {audit.surviving_steps}")
    print(f"Pruned at Step: {audit.pruned_at_step} | Reward Hacking Detected: {audit.reward_hacking_detected}")
    for alert in audit.security_alerts:
        print(" ", alert)

    assert audit.reward_hacking_detected is True, "Failed to detect sycophantic filler reward hacking!"
    assert audit.pruned_at_step == 2, "Failed to sever trajectory at hacked Step 2!"
    assert len(audit.surviving_steps) == 1, "Only Step 1 should survive pruning!"
    print("[OK] Part 4 (Security & Ethics Sentinel) verification passed.")
