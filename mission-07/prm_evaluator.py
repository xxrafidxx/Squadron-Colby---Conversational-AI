"""Mission 07, Part 3: transparent mock PRM and exact toy algebra check.

Default scores follow the instructor's labelled fixture; they are not learned
probabilities. An injected scorer can replace that fixture. The algebra helper
checks only ax+b=c using rational arithmetic, not arbitrary mathematical prose.
"""
from fractions import Fraction
import math
import re


class OutcomeRewardEvaluator:
    def __init__(self, expected_final_marker):
        self.expected_final_marker = expected_final_marker

    def score_trajectory(self, steps):
        return float(bool(steps) and self.expected_final_marker in steps[-1])


def linear_solution(equation):
    """Solve a*x+b=c for the deliberately restricted integer-equation grammar."""
    compact = re.sub(r'\s+', '', equation)
    match = re.fullmatch(r'([+-]?\d*\*?)x([+-]\d+)?=([+-]?\d+)', compact)
    if not match:
        raise ValueError('expected an integer-coefficient equation a*x+b=c')
    a, b, c = match.groups()
    a = a.replace('*', '')
    a = 1 if a in ('', '+') else -1 if a == '-' else int(a)
    if a == 0:
        raise ValueError('zero x coefficient is outside this toy verifier')
    return Fraction(int(c) - int(b or 0), a)


def algebra_step_scores(steps):
    """The first equation is the problem premise; compare successive solution sets."""
    if not steps:
        return []
    previous = linear_solution(steps[0])
    scores = [0.94]
    for equation in steps[1:]:
        current = linear_solution(equation)
        scores.append(0.94 if current == previous else 0.12)
        previous = current
    return scores


class ProcessRewardEvaluator:
    def __init__(self, scorer=None):
        self.scorer = scorer

    def step_score(self, step):
        if self.scorer is not None:
            score = float(self.scorer(step))
        elif '[FLAWED]' in step or 'SIGN_ERROR' in step:
            score = 0.12
        elif '[MINOR_DRIFT]' in step:
            score = 0.55
        else:
            score = 0.94
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('score must be finite and in [0, 1]')
        return score

    def score_all_steps(self, steps):
        return [self.step_score(step) for step in steps]

    @staticmethod
    def reduce_scores(scores, reduction='min'):
        scores = list(scores)
        if reduction not in ('min', 'prod', 'geometric_mean'):
            raise ValueError('unknown trajectory reduction')
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
            raise ValueError('scores must be finite and in [0, 1]')
        if not scores:
            return 0.0  # An empty chain supplies no evidence of validity.
        if reduction == 'min':
            return min(scores)
        if reduction == 'prod':
            return math.prod(scores)
        if 0 in scores:
            return 0.0
        return math.exp(math.fsum(math.log(x) for x in scores) / len(scores))

    def aggregate_trajectory_min(self, steps):
        return self.reduce_scores(self.score_all_steps(steps), 'min')

    def aggregate_trajectory_prod(self, steps):
        return self.reduce_scores(self.score_all_steps(steps), 'prod')

    def aggregate_trajectory_geometric_mean(self, steps):
        return self.reduce_scores(self.score_all_steps(steps), 'geometric_mean')

    def math_shepherd_estimation(self, intermediate_step, rollout_final_answers,
                                ground_truth_answer):
        """Reduce caller-supplied continuations of intermediate_step, using exact match.

        This function does not generate continuations or certify their provenance.
        For M=0 the estimate is undefined and is rejected.
        """
        if not rollout_final_answers:
            raise ValueError('Math-Shepherd estimation requires at least one rollout')
        matches = sum(answer == ground_truth_answer for answer in rollout_final_answers)
        return {'soft_label': matches / len(rollout_final_answers),
                'hard_label': float(matches > 0)}


# Original instructor verification harness (unchanged).
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 3): PRM vs. ORM & Math-Shepherd Verification ===")

    orm = OutcomeRewardEvaluator(expected_final_marker="v_esc = 24.0 km/s")
    prm = ProcessRewardEvaluator()

    # Trajectory 1: Rigorous Kerr Geodesic Proof
    valid_chain = [
        "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
        "Step 2: Integrate conserved orbital energy E and angular momentum L_z",
        "Step 3: Solve radial potential boundary -> v_esc = 24.0 km/s",
    ]

    # Trajectory 2: False-Positive Trap (Double-sign cancellation in Step 2 reaches lucky right scalar!)
    false_positive_chain = [
        "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
        "Step 2: [FLAWED] Flip sign of frame-dragging term twice (-1 * -1 = +1) via illegal tensor contraction",
        "Step 3: Accidental cancellation yields -> v_esc = 24.0 km/s",
    ]

    for label, chain in [("Rigorous Chain", valid_chain), ("False-Positive Trap", false_positive_chain)]:
        orm_score = orm.score_trajectory(chain)
        prm_min = prm.aggregate_trajectory_min(chain)
        prm_prod = prm.aggregate_trajectory_prod(chain)
        prm_geo = prm.aggregate_trajectory_geometric_mean(chain)
        print(
            f"{label:20s} | ORM={orm_score:.2f} | PRM(min)={prm_min:.3f} | "
            f"PRM(prod)={prm_prod:.3f} | PRM(geo)={prm_geo:.3f}"
        )

    assert orm.score_trajectory(false_positive_chain) == 1.0, "ORM should be fooled by the terminal answer!"
    assert prm.aggregate_trajectory_min(false_positive_chain) < 0.20, "PRM(min) must catch the flawed intermediate step!"

    # Test Math-Shepherd Monte Carlo Step Estimation
    shepherd = prm.math_shepherd_estimation(
        intermediate_step=valid_chain[1],
        rollout_final_answers=["24.0 km/s", "24.0 km/s", "22.5 km/s", "24.0 km/s"],
        ground_truth_answer="24.0 km/s",
    )
    print(f"Math-Shepherd Step Estimation (M=4 rollouts): {shepherd}")
    assert abs(shepherd["soft_label"] - 0.75) < 1e-6 and shepherd["hard_label"] == 1.0
    print("[OK] Part 3 (Data & Payload Specialist) verification passed.")
