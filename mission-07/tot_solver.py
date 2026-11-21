"""Mission 07, Part 1: budgeted ToT. Based on the course starter API.

One expansion means one generated, evaluated child, including rejected children.
Result ordering is deepest first, then weakest-link score, then last-step score.
Searches require a fresh root; the retained tree is an audit trail, not a KV cache.
"""
from dataclasses import dataclass, field
from itertools import islice
import math
from typing import Callable, Iterable, List, Optional


@dataclass
class ThoughtNode:
    state: str
    depth: int = 0
    step_score: float = 1.0
    cumulative_score: float = 1.0
    parent: Optional['ThoughtNode'] = None
    children: list['ThoughtNode'] = field(default_factory=list)
    is_pruned: bool = False

    def add_child(self, child_state: str, step_score: float) -> 'ThoughtNode':
        child = ThoughtNode(child_state, self.depth + 1, step_score,
                            min(self.cumulative_score, step_score), self)
        self.children.append(child)
        return child

    def get_trajectory(self) -> list[str]:
        path, current = [], self
        while current is not None:
            path.append(current.state)
            current = current.parent
        return path[::-1]


class TreeOfThoughtsSolver:
    def __init__(self, max_expansions=16, beam_width=2, prune_threshold=0.45):
        if type(max_expansions) is not int or max_expansions < 0:
            raise ValueError('max_expansions must be a nonnegative integer')
        if type(beam_width) is not int or beam_width < 1:
            raise ValueError('beam_width must be a positive integer')
        if not math.isfinite(prune_threshold) or not 0 <= prune_threshold <= 1:
            raise ValueError('prune_threshold must be in [0, 1]')
        self.max_expansions = max_expansions
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold
        self.expansion_count = 0
        self.expanded_states = []
        self.frontier_widths = []

    @staticmethod
    def _rank(node):
        return node.depth, node.cumulative_score, node.step_score

    def _start(self, root, max_depth):
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError('max_depth must be a nonnegative integer')
        if root.children or root.parent or root.depth != 0:
            raise ValueError('use a fresh depth-zero root for each search')
        self.expansion_count = 0
        self.expanded_states = []
        self.frontier_widths = []

    def _expand(self, node, generator_fn, evaluator_fn):
        remaining = self.max_expansions - self.expansion_count
        if remaining <= 0:
            return []
        self.expanded_states.append(node.state)
        valid = []
        # islice avoids requesting a surplus child from a lazy generator.
        for state in islice(generator_fn(node), remaining):
            self.expansion_count += 1
            score = float(evaluator_fn(state))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('evaluator must return a finite score in [0, 1]')
            child = node.add_child(state, score)
            child.is_pruned = score < self.prune_threshold
            if not child.is_pruned:
                valid.append(child)
        return valid

    def bfs_beam_search(self, root: ThoughtNode,
                        generator_fn: Callable[[ThoughtNode], Iterable[str]],
                        evaluator_fn: Callable[[str], float], max_depth=3):
        self._start(root, max_depth)
        beam, best = [root], root
        for _ in range(max_depth):
            candidates = []
            for node in beam:
                if self.expansion_count >= self.max_expansions:
                    break
                candidates.extend(self._expand(node, generator_fn, evaluator_fn))
            if not candidates:
                break
            candidates.sort(key=self._rank, reverse=True)
            beam = candidates[:self.beam_width]
            self.frontier_widths.append(len(beam))
            best = max([best, *beam], key=self._rank)
        return best

    def dfs_backtrack_search(self, root: ThoughtNode,
                             generator_fn: Callable[[ThoughtNode], Iterable[str]],
                             evaluator_fn: Callable[[str], float], max_depth=3):
        self._start(root, max_depth)
        best = root

        def visit(node):
            nonlocal best
            best = max([best, node], key=self._rank)
            if node.depth >= max_depth or self.expansion_count >= self.max_expansions:
                return
            children = self._expand(node, generator_fn, evaluator_fn)
            children.sort(key=lambda child: child.step_score, reverse=True)
            for child in children:
                # Even the child consuming the last budget unit must be considered.
                visit(child)

        visit(root)
        return best


# Original instructor verification harness (unchanged).
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 1): Tree-of-Thoughts Solver Verification ===")

    # Mock Sagittarius A* 24-Vector Thruster Synthesis Tree
    MOCK_BRANCHES = {
        "Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]": [
            "Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID]",
            "Step 1B: 10 + 13 = 23 (Remaining: [4, 9, 23]) [DEAD-END]",
        ],
        "Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID]": [
            "Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL]",
            "Step 2B: 4 + 4 = 8 (Remaining: [8, 10]) [DEAD-END]",
        ],
        "Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL]": [
            "Step 3A: 4 * 6 = 24.0 km/s -> Escape Geodesic Locked! [GOAL]",
        ],
    }

    def mock_generator(node: ThoughtNode) -> List[str]:
        return MOCK_BRANCHES.get(node.state, [])

    def mock_evaluator(thought: str) -> float:
        if "[GOAL]" in thought:
            return 0.98
        if "[OPTIMAL]" in thought:
            return 0.92
        if "[VALID]" in thought:
            return 0.85
        return 0.25  # Dead-end branch below prune_threshold

    solver = TreeOfThoughtsSolver(max_expansions=12, beam_width=2, prune_threshold=0.45)

    root_bfs = ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]")
    best_bfs = solver.bfs_beam_search(root_bfs, mock_generator, mock_evaluator, max_depth=3)
    print(f"[ToT-BFS] Expansions Used: {solver.expansion_count}/{solver.max_expansions}")
    print(f"[ToT-BFS] Best Terminal Score: {best_bfs.cumulative_score:.2f}")
    print(f"[ToT-BFS] Winning Trajectory: {' -> '.join(best_bfs.get_trajectory())}")

    root_dfs = ThoughtNode("Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13]")
    best_dfs = solver.dfs_backtrack_search(root_dfs, mock_generator, mock_evaluator, max_depth=3)
    print(f"[ToT-DFS] Expansions Used: {solver.expansion_count}/{solver.max_expansions}")
    print(f"[ToT-DFS] Winning Trajectory: {' -> '.join(best_dfs.get_trajectory())}")

    assert "[GOAL]" in best_bfs.state, "ToT-BFS failed to reach [GOAL] state!"
    assert "[GOAL]" in best_dfs.state, "ToT-DFS failed to reach [GOAL] state!"
    print("[OK] Part 1 (Systems Architect) verification passed.")
