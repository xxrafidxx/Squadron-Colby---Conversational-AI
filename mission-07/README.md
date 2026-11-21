# Mission 07 - Gravitational Search Horizon and Process Reward Evaluator

Squadron Colby | Conversational AI | Winter Semester 2026/27

The four native Python modules implement ToT beam/backtracking search, PUCT MCTS, process reward reductions and Event Horizon pruning. The deterministic integration experiment rejects sign-error and filler branches and returns `2*x - 6 = 0 -> 2*x = 6 -> x = 3` with both search engines.

## Deliverables

| Part | Implementation | Report |
| --- | --- | --- |
| 1 - Systems Architect | [tot_solver.py](tot_solver.py) | [Section A](mission-07-doc.md#a-systems-architect---tot_solverpy) |
| 2 - Protocol Officer | [mcts_protocol.py](mcts_protocol.py) | [Section B](mission-07-doc.md#b-communications--protocol-officer---mcts_protocolpy) |
| 3 - Data Specialist | [prm_evaluator.py](prm_evaluator.py) | [Section C](mission-07-doc.md#c-data--payload-specialist---prm_evaluatorpy) |
| 4 - Security Sentinel | [horizon_pruner.py](horizon_pruner.py) | [Section D](mission-07-doc.md#d-security--ethics-sentinel---horizon_prunerpy) |
| 5 - Squadron documentation | [mission-07-doc.md](mission-07-doc.md) | [mission-07-doc.pdf](mission-07-doc.pdf) |

[Deliverables overview](Deliverables.md) | [Team roles](00_Team_Roles.md) | [Safety and compliance](06_AI_Act_Compliance.md)

## Run and verify

Requires Python 3.11+; no pip installs, credentials or network calls are needed. Open a terminal in this directory:

```sh
python verify.py
```

This runs the four original module harnesses, `python -m unittest -v` and `python run_experiments.py`. All four harnesses and **22 tests** passed using Python 3.12.10. Run any module directly to see its individual harness. No notebook is required for Mission 07.

## Evidence and results

- [Full execution log](evidence/verification.txt)
- [Experiment results](evidence/results.json): search budgets, PUCT sweep, ORM/PRM, adversarial cases and integration
- [Tested source hashes](evidence/source_sha256.json)
- [Regression tests](test_mission07.py) and [integration runner](run_experiments.py)

In the supplied 24-game fixture BFS and DFS both use 5/12 expansions. In the deceptive-branch fixture, width 1 misses the goal; width 2 and DFS reach it with 6 evaluations. PUCT scores are A=1.25 and B=1.75. The actual two-sign-error chain receives ORM=1.0 but PRM minimum=0.12. Filler injection is reduced from 0.96 to 0.192 and severed at step 2.

## Scope and provenance

This is a deterministic simulation with synthetic candidate trees, hand-set priors and fixed continuation answers, not a trained PRM or an LLM performance benchmark. The default marker-based scorer follows the course fixture. An additional exact rational checker covers simple linear equations only. The pruning rules are heuristic, and retained search nodes provide an audit trail.

Based on the [instructor Mission 07 files](https://github.com/Unisvet/couse-c-ai/tree/43ad727a8b221118cc64a1c8e3f5ccbf6ff04105/mission-07), commit `43ad727a8b221118cc64a1c8e3f5ccbf6ff04105`, verified 21 November 2026. The four original `__main__` harnesses remain unchanged. The supplied [briefing](ps07_briefing.md) is included. The completed report documents implementation changes, boundary conventions and AI assistance.

## Submission

Upload this entire **mission-07** directory under the repository root, alongside mission-05 and mission-06. Keep the evidence subdirectory. Upload **mission-07-doc.pdf** to Moodle as explicitly required by the brief, and use this [repository folder link](https://github.com/xxrafidxx/Squadron-Colby---Conversational-AI/tree/main/mission-07) in the portal's Week 7 / 4. Transmit Solution tab.

The separate Moodle package also contains Deliverables.pdf, Team_Roles.pdf and Compliance.pdf in the earlier submission style. Deliverables.pdf and mission-07-doc.pdf contain the same completed report; the latter preserves the professor's required filename. Submission and portal updates are manual.
