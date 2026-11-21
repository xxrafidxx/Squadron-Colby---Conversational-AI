# Mission 07 - Deliverables

The full submission report is [mission-07-doc.md](mission-07-doc.md), with a [PDF copy](mission-07-doc.pdf). It covers all five required report sections: roster, integrated architecture, four station reports, captured verification output and the test-time-compute reflection.

| Part | File | Contents |
| --- | --- | --- |
| 1 | [tot_solver.py](tot_solver.py) | ThoughtNode, budgeted BFS beam and DFS backtracking |
| 2 | [mcts_protocol.py](mcts_protocol.py) | MCTS nodes, exact PUCT and all four phases |
| 3 | [prm_evaluator.py](prm_evaluator.py) | ORM, PRM reductions, algebra trap and Math-Shepherd labels |
| 4 | [horizon_pruner.py](horizon_pruner.py) | Filler/repetition penalties, early severance and telemetry |
| 5 | [mission-07-doc.md](mission-07-doc.md) | Completed experimental documentation |

Run `python verify.py` to reproduce the four passing harnesses, 22 regression tests and integrated experiments. See [captured evidence](evidence/verification.txt) and [all numeric results](evidence/results.json).

Supporting documents: [team roles](00_Team_Roles.md) and [compliance review](06_AI_Act_Compliance.md). For Moodle, use the supplied mission-07-doc.pdf plus the requested Team_Roles.pdf and Compliance.pdf. Deliverables.pdf is an identical report copy under the familiar name, supplied for convenience.
