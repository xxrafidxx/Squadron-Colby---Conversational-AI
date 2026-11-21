"""Mission 07, Part 2: PUCT and the four MCTS phases.

Caller supplies policy priors and value estimates. No LLM service is invoked.
Priors need not sum to one: the course harness deliberately uses 0.5 + 0.4.
They are retained unchanged, not silently renormalized.
"""
from dataclasses import dataclass, field
import math
from typing import Optional


@dataclass
class MCTSNode:
    state: str
    prior_prob: float = 1.0
    parent: Optional['MCTSNode'] = None
    children: list['MCTSNode'] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False

    @property
    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTSEngine:
    def __init__(self, c_puct=1.5):
        if not math.isfinite(c_puct) or c_puct < 0:
            raise ValueError('c_puct must be finite and nonnegative')
        self.c_puct = c_puct

    def puct_score(self, parent, child):
        return (child.q_value + self.c_puct * child.prior_prob
                * math.sqrt(max(1, parent.visit_count)) / (1 + child.visit_count))

    def select_best_child(self, node):
        if not node.children:
            raise ValueError('cannot select a child from a leaf')
        return max(node.children, key=lambda child: self.puct_score(node, child))

    def select_leaf(self, root):
        current = root
        while current.children and not current.is_terminal:
            current = self.select_best_child(current)
        return current

    def expand(self, node, action_priors):
        if node.is_terminal or node.children:
            raise ValueError('expand only an unexpanded nonterminal node')
        pairs = list(action_priors.items() if hasattr(action_priors, 'items')
                     else action_priors)
        states = [state for state, _ in pairs]
        if len(states) != len(set(states)):
            raise ValueError('duplicate candidate states')
        for state, prior in pairs:
            if not isinstance(state, str) or not math.isfinite(prior) or not 0 <= prior <= 1:
                raise ValueError('expected a string state and a finite prior in [0, 1]')
        node.children = [MCTSNode(state, prior, node) for state, prior in pairs]
        if not pairs:
            node.is_terminal = True
        return node.children

    def simulate_rollout(self, node, rollout_evaluator):
        reward = float(rollout_evaluator(node.state))
        if not math.isfinite(reward):
            raise ValueError('rollout must return a finite value')
        return max(0.0, min(1.0, reward))

    def backpropagate(self, node, reward):
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError('reward must be in [0, 1]')
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            current = current.parent


# Original instructor verification harness (unchanged).
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 2): MCTS & PUCT Protocol Verification ===")

    engine = MCTSEngine(c_puct=1.5)
    root = MCTSNode(state="Root: Sagittarius A* Ergosphere Geodesic Entry", visit_count=16)

    branch_a = MCTSNode(state="Branch A: Prograde Frame-Dragging Assist", prior_prob=0.50, parent=root)
    branch_a.visit_count = 5
    branch_a.value_sum = 0.75 * 5  # Q(s, A) = 0.75

    branch_b = MCTSNode(state="Branch B: Retrograde Penrose Extraction", prior_prob=0.40, parent=root)
    branch_b.visit_count = 1
    branch_b.value_sum = 0.55 * 1  # Q(s, B) = 0.55

    root.children = [branch_a, branch_b]

    puct_a = engine.puct_score(root, branch_a)
    puct_b = engine.puct_score(root, branch_b)
    selected = engine.select_best_child(root)

    print(f"Branch A -> Q={branch_a.q_value:.2f}, N={branch_a.visit_count}, P={branch_a.prior_prob:.2f} | PUCT = {puct_a:.4f}")
    print(f"Branch B -> Q={branch_b.q_value:.2f}, N={branch_b.visit_count}, P={branch_b.prior_prob:.2f} | PUCT = {puct_b:.4f}")
    print(f"Selected Next Node for Expansion: {selected.state}")

    assert abs(puct_a - 1.25) < 1e-6, f"Expected PUCT(A)=1.25, got {puct_a}"
    assert abs(puct_b - 1.75) < 1e-6, f"Expected PUCT(B)=1.75, got {puct_b}"
    assert selected is branch_b, "MCTS failed to select under-explored high-PUCT Branch B!"

    # Test Backpropagation
    engine.backpropagate(branch_b, reward=0.95)
    assert branch_b.visit_count == 2 and root.visit_count == 17, "Backpropagation failed to increment visit counts!"
    print("[OK] Part 2 (Communications & Protocol Officer) verification passed.")
