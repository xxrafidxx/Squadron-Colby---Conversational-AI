# Gravitational search horizon and process reward evaluator

Mission 07 | Squadron Colby | Conversational AI | Winter Semester 2026/27

Sector SEC-07-SGR: Sagittarius A* | Prepared 21 November 2026

[Public squadron repository - Mission 07](https://github.com/xxrafidxx/Squadron-Colby---Conversational-AI/tree/main/mission-07)

## 1. Squadron roster and verification status

Role allocation follows the earlier squadron submissions. Status records executed software checks; it does not record personal signatures or assert that individual team members performed this run. Rafid holds both the architecture and protocol stations.

| Station | Team member | Deliverable | Verification |
| --- | --- | --- | --- |
| Systems Architect | Rafid Mohammad Ahsan | tot_solver.py | Automated PASS |
| Communications & Protocol Officer | Rafid Mohammad Ahsan | mcts_protocol.py | Automated PASS |
| Data & Payload Specialist | Tahjib Al Sakin | prm_evaluator.py | Automated PASS |
| Security & Ethics Sentinel | Kunal Choudhary | horizon_pruner.py | Automated PASS |
| AI Security Officer | Mubina Rahaman Jerin | Compliance review | Controls documented |

## 2. Integrated System 2 architecture

```text
Prompt / synthetic candidate generator
          |                              |
          v                              v
ToT BFS beam / DFS                MCTS PUCT selection
12-child budget                   12 search iterations
          |                              |
          +----------+-------------------+
                     v
           PRM / toy algebra check
                     v
           Event Horizon Pruner
         threshold .45, penalty .20
                /            \
        below threshold      accepted
           sever + log       /       \
                         ToT beam   MCTS backup
```

The two search controllers generate alternative paths, score their intermediate steps and submit them to the pruner. In the integrated experiment, a failed path returns zero to ToT and is excluded before MCTS expansion. Accepted MCTS values are backed up through every ancestor. ToT counts every evaluated child; MCTS is bounded separately by 12 iterations and 32 candidate evaluations.

This submission uses deterministic synthetic trees, explicit priors and mock PRM scores. It invokes no LLM API, trains no model and performs no physical black-hole simulation. The algebra extension verifies a restricted equation grammar using exact rational arithmetic.


<!-- pagebreak -->

## 3. Station-by-station experimental report

### A. Systems Architect - tot_solver.py

ThoughtNode retains the semantic state, depth, local score, minimum-link cumulative score, parent, children and pruning flag. get_trajectory() walks parent links back to the root. A search returns the deepest accepted node, with cumulative score and final-step score breaking ties. A returned partial path is not automatically a solved answer.

The supplied 24-game harness uses beam width 2, threshold 0.45, depth 3 and budget 12. BFS and DFS each evaluate 5 children and find the goal with cumulative score 0.85: 13 - 9 = 4; 10 - 4 = 6; 4 * 6 = 24. All four original input numbers are used once.

### Additional deceptive-branch experiment

The synthetic tree starts with A (0.95) and B (0.85). A leads to A1 (0.90), a dead end, and bad (0.10), which is pruned. B leads to B1 (0.80), then goal (0.98). Each row is a fresh run, with maximum depth 3. Used includes rejected children; score is the minimum along the returned path.

| Search | Beam | Budget | Used | Depth | Score | Goal |
| --- | --- | --- | --- | --- | --- | --- |
| BFS | 1 | 3 | 3 | 2 | 0.900 | No |
| BFS | 2 | 3 | 3 | 2 | 0.900 | No |
| DFS | - | 3 | 3 | 2 | 0.900 | No |
| BFS | 1 | 5 | 4 | 2 | 0.900 | No |
| BFS | 2 | 5 | 5 | 2 | 0.900 | No |
| DFS | - | 5 | 5 | 2 | 0.900 | No |
| BFS | 1 | 8 | 4 | 2 | 0.900 | No |
| BFS | 2 | 8 | 6 | 3 | 0.800 | Yes |
| DFS | - | 8 | 6 | 3 | 0.800 | Yes |
| BFS | 1 | 12 | 4 | 2 | 0.900 | No |
| BFS | 2 | 12 | 6 | 3 | 0.800 | Yes |
| DFS | - | 12 | 6 | 3 | 0.800 | Yes |

At budget 12, BFS with width 2 expands parents root, A, B, A1, B1. DFS expands root, A, A1, B, B1, demonstrating chronological backtracking from A1. Both return root -> B -> B1 -> goal after 6 evaluations. Width 1 drops B and cannot recover the goal even when the budget increases.

### Compute and memory analysis

BFS maintains at most B active frontier nodes; a wider beam preserves alternatives but spends more evaluations across each level. DFS follows the highest local-scoring child first and returns to siblings at dead ends. There is no universal speed winner: on this fixture both need 6 evaluations. The implementation retains its audit tree, so total stored nodes are O(max_expansions), even for DFS. A real LLM implementation would also pay for generated tokens and potentially several active KV caches; neither GPU memory nor token cost is measured here.

The budget limits evaluated children, including pruned children. A lazy generator is consumed only up to the remaining budget. A callback that eagerly constructs a list or blocks needs its own token/time limit. Tests cover zero budget, exact exhaustion, the final permitted child, pruning and depth limits.


<!-- pagebreak -->

### B. Communications & Protocol Officer - mcts_protocol.py

The engine implements selection, expansion, simulation and backpropagation as separate methods. Policy priors are supplied by the caller. The exact course PUCT expression is:

```text
Q = value_sum / visit_count if visit_count > 0 else 0
N_parent = max(1, parent.visit_count)
U = c_puct * prior * sqrt(N_parent) / (1 + child.visit_count)
PUCT = Q + U
```

### Measured PUCT trace at c_puct = 1.5

| Branch | Q | P | Parent N | Child N | Bonus U | PUCT | Chosen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.75 | 0.50 | 16 | 5 | 0.5000 | 1.2500 | No |
| B | 0.55 | 0.40 | 16 | 1 | 1.2000 | 1.7500 | Yes |

A has value_sum 3.75 and 5 visits. B has value_sum 0.55 and 1 visit. Despite its smaller Q, B receives the larger exploration bonus and is selected. The priors sum to 0.90 in the instructor fixture; the engine preserves them exactly. These hand-set statistics exercise the formula and are not an observed search history.

### Exploration constant sweep

| c_puct | PUCT(A) | PUCT(B) | Selected |
| --- | --- | --- | --- |
| 0.1 | 0.7833 | 0.6300 | A |
| 0.5 | 0.9167 | 0.9500 | B |
| 1.5 | 1.2500 | 1.7500 | B |
| 2.5 | 1.5833 | 2.5500 | B |

Raising c_puct from 0.5 to 2.5 increases exploration pressure, but B already wins at 0.5, so this pair does not switch branches in that interval. At 0.1, A wins. For these fixed statistics the crossover is c_puct = 3/7, approximately 0.428571. This avoids claiming a selection change that the calculation does not show.

### Four-phase behavior and verification

select_leaf() follows maximum PUCT until an unexpanded or terminal node. expand() attaches each unique candidate and preserves its prior; an empty expansion marks a terminal leaf. simulate_rollout() accepts a supplied value estimate, clamps finite values to [0,1] and rejects non-finite values. backpropagate() adds the same reward and one visit at every ancestor.

The supplied harness backs up reward 0.95 through B: B changes from (N=1, W=0.55) to (N=2, W=1.50), and the root visit count changes from 16 to 17. A separate three-node test verifies both visits and values at every ancestor. Additional tests cover zero visits, terminal selection, duplicate expansion, invalid priors and reward bounds.

The MCTS module exposes the four phases. run_experiments.py supplies the bounded loop and the PRM/pruner adapter; importing the module alone does not execute a search.


<!-- pagebreak -->

### C. Data & Payload Specialist - prm_evaluator.py

ProcessRewardEvaluator exposes minimum, product and geometric-mean reductions. Given scores r1,...,rT, these are min(r), product(r) and exp(sum(log(r))/T). The logarithmic geometric mean avoids premature underflow for long chains. Empty trajectories score zero; zero scores are allowed as a rejection extension to the brief. Invalid and non-finite scores are rejected.

### Actual two-sign-error benchmark

The valid chain is 2*x - 6 = 0 -> 2*x = 6 -> x = 3. The trap is 2*x - 6 = 0 -> 2*x = -6 -> x = 3. Moving -6 with the wrong sign produces x=-3; dividing -6 by 2 with another wrong sign returns x=3. Thus two invalid transformations produce the expected final answer.

The restricted algebra checker compares each adjacent equation's solution using Fraction. The valid chain scores [0.94, 0.94, 0.94]; the trap scores [0.94, 0.12, 0.12]. Neither chain contains a FLAWED label. The ORM sees only the final marker x = 3 and accepts both.

| Trajectory | ORM | Min | Product | Geo mean | Soft M=4 | Hard |
| --- | --- | --- | --- | --- | --- | --- |
| Valid algebra | 1.00 | 0.940 | 0.830584 | 0.940000 | 0.75 | 1 |
| Two sign errors | 1.00 | 0.120 | 0.013536 | 0.238322 | 0.25 | 1 |

### Instructor fixture and interpretation

The original labelled fixture also passes: a valid three-step chain has min 0.940, product 0.830584 and geometric mean 0.940. Its single flagged intermediate step gives [0.94, 0.12, 0.94], min 0.120, product 0.106032 and geometric mean about 0.473310. These are mock scores; the Kerr-related text in that harness is not a verified physics derivation.

Minimum reduction exposes the weakest step regardless of chain length. Product decreases when more sub-unit scores are multiplied; interpreting it as joint validity requires appropriately calibrated conditional scores or independence assumptions. The geometric mean normalizes length but can dilute one low score among many high scores. The pruner therefore checks each step before any trajectory-level ranking.

### Math-Shepherd labels and limitations

The valid step's four synthetic completion answers are [3, 3, 2, 3]; the flawed step's are [-3, -3, 3, -3], against ground truth 3. Soft labels are correct-answer counts divided by 4; hard labels are 1 when at least one answer matches. The flawed step consequently still receives hard label 1, showing that a lucky or repaired continuation does not establish local logical validity.

math_shepherd_estimation() reduces supplied answers; it does not sample an LLM. These fixed completions demonstrate the estimator formula, not a statistical estimate of deployed model quality. Zero rollouts are rejected because the fraction would be undefined. The default PRM follows the instructor's marker-based heuristic; an injected scorer can replace it. The algebra checker supports only integer-coefficient linear equations with nonzero x coefficient.


<!-- pagebreak -->

### D. Security & Ethics Sentinel - horizon_pruner.py

The horizon threshold is 0.45 and the hacking penalty factor is 0.20. Each step is checked before it joins the surviving prefix. A flagged step receives raw_score * 0.20; any adjusted score below 0.45 causes immediate severance. Equality at the threshold is accepted. Alerts use one-based step numbers and preserve the rejection reason.

### Adversarial payload and measured results

Payload: "Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0". The explicit naive-score fixture assigns 0.96; the guard reduces it to 0.192, then cuts the path at step 2. Only step 1 survives; step 3 is not processed for acceptance.

| Fixture | Severed at step | Surviving steps | Hack detected |
| --- | --- | --- | --- |
| Filler injection | 2 | 1 | Yes |
| Repeated equation | 2 | 1 | Yes |
| Invalid step | 2 | 1 | No |
| Clean chain | None | 3 | No |

Step-splitting is tested with "Step 1: x = 3" followed by "Step 2: x=3.". Removing numbered labels and whitespace reveals the same equation; its 0.94 score becomes 0.188. A set of canonical earlier steps also detects nonconsecutive repetition. Variable case is retained to avoid equating distinct x and X symbols.

For a naive arithmetic mean, [0.12, 0.94] averages 0.53; appending eight more 0.94 echoes raises it to 0.858. Our threshold cuts the original 0.12 step immediately, before later steps could dilute it. Phrase matching also includes "Obviously, by trivial inspection" from the brief.

### Integrated ToT and MCTS experiment

Both engines return 2*x - 6 = 0 -> 2*x = 6 -> x = 3. ToT evaluates 4 candidates under budget 12, with minimum score 0.94. MCTS evaluates 4 candidates under its limit of 32, performs 12 backups and obtains root mean value 0.94. The sign-error and filler branches are rejected before they can be expanded further. Accepted priors retain their original scale after filtering.

```text
[Step 2] ANOMALY 07: SYCOPHANTIC_FILLER_DETECTED; 0.960000 -> 0.192000
[Step 2] EVENT HORIZON BREACH: 0.120000 < 0.450000; branch severed
[Step 2] EVENT HORIZON BREACH: 0.192000 < 0.450000; branch severed
```

### Defense limits

The detector recognizes configured phrases and normalized textual repetition; it does not understand every paraphrase or equivalent equation. Legitimate repetition may be flagged. Default marker scores can be manipulated by removing labels. These checks reduce the demonstrated failures, but do not certify general reasoning, physical navigation or adversarial robustness. A real deployment needs independent verification, operator oversight, calibrated scores and time/token limits around model callbacks.


<!-- pagebreak -->

## 4. End-to-end verification terminal output

The following is captured output from the four required standalone modules, not an illustrative transcript. Long lines are wrapped for the PDF. Full regression output, experiment results and source hashes are included under evidence/.

```text
UTC: 2026-11-21T11:08:49.129718+00:00
Python: 3.12.10
All models, priors and rollout answers in these tests are synthetic.

$ python tot_solver.py
=== Subsystem 07 (Part 1): Tree-of-Thoughts Solver Verification ===
[ToT-BFS] Expansions Used: 5/12
[ToT-BFS] Best Terminal Score: 0.85
[ToT-BFS] Winning Trajectory: Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13] -> Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID] -> Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL] -> Step 3A: 4 * 6 = 24.0 km/s -> Escape Geodesic Locked! [GOAL]
[ToT-DFS] Expansions Used: 5/12
[ToT-DFS] Winning Trajectory: Prompt: Synthesize 24.0 km/s from [4, 9, 10, 13] -> Step 1A: 13 - 9 = 4 (Remaining: [4, 4, 10]) [VALID] -> Step 2A: 10 - 4 = 6 (Remaining: [4, 6]) [OPTIMAL] -> Step 3A: 4 * 6 = 24.0 km/s -> Escape Geodesic Locked! [GOAL]
[OK] Part 1 (Systems Architect) verification passed.
Exit code: 0


$ python mcts_protocol.py
=== Subsystem 07 (Part 2): MCTS & PUCT Protocol Verification ===
Branch A -> Q=0.75, N=5, P=0.50 | PUCT = 1.2500
Branch B -> Q=0.55, N=1, P=0.40 | PUCT = 1.7500
Selected Next Node for Expansion: Branch B: Retrograde Penrose Extraction
[OK] Part 2 (Communications & Protocol Officer) verification passed.
Exit code: 0


$ python prm_evaluator.py
=== Subsystem 07 (Part 3): PRM vs. ORM & Math-Shepherd Verification ===
Rigorous Chain       | ORM=1.00 | PRM(min)=0.940 | PRM(prod)=0.831 | PRM(geo)=0.940
False-Positive Trap  | ORM=1.00 | PRM(min)=0.120 | PRM(prod)=0.106 | PRM(geo)=0.473
Math-Shepherd Step Estimation (M=4 rollouts): {'soft_label': 0.75, 'hard_label': 1.0}
[OK] Part 3 (Data & Payload Specialist) verification passed.
Exit code: 0


$ python horizon_pruner.py
=== Subsystem 07 (Part 4): Event Horizon Pruner & Reward-Hacking Defense ===
Surviving Steps (1/3): ['Step 1: Compute Schwarzschild radius r_s = 2GM / c^2 = 1.27e10 m']
Pruned at Step: 2 | Reward Hacking Detected: True
  [Step 2] ANOMALY 07: SYCOPHANTIC_FILLER_DETECTED; 0.960000 -> 0.192000
  [Step 2] EVENT HORIZON BREACH: 0.192000 < 0.450000; branch severed
[OK] Part 4 (Security & Ethics Sentinel) verification passed.
Exit code: 0
```

The additional command python -m unittest -v passed all 22 tests. python run_experiments.py also exited with code 0. python verify.py executes the four harnesses, these tests and the integration experiments in one sequence and stops on failure.


<!-- pagebreak -->

## 5. Reflection on test-time compute scaling

Inference-time compute can improve multi-step reasoning with fixed model weights by generating alternatives, comparing intermediate states and backtracking before committing to an answer. It allocates more search to a problem rather than adding learned parameters; it cannot guarantee knowledge or candidate solutions absent from the generator. Our small experiment illustrates the distinction: beam width 1 discards the viable branch even with budget 12, whereas width 2 and DFS find the goal with 6 evaluations. A larger search also creates more opportunities to select plausible-looking errors or exploit an imperfect reward model. Step-level verification is therefore essential: the final-answer check accepts the two-sign-error chain, while the algebra check and horizon threshold reject it at the first invalid transformation. Benefits depend on candidate diversity, verifier quality and resource limits; this deterministic exercise demonstrates those mechanisms without measuring LLM benchmark gains.

### Reproduction and evidence

Use Python 3.11 or later with no third-party packages. The recorded run used Python 3.12.10. From the mission-07 directory run:

```text
python verify.py
```

evidence/verification.txt contains the captured execution log, evidence/results.json contains every reported experimental row, and evidence/source_sha256.json identifies the tested Python source files. Direct module commands remain supported. Re-running verification refreshes the evidence; elapsed test times may vary. The PDF is a snapshot of the recorded submission run.

### Source provenance and assistance disclosure

The implementation follows the instructor's class interfaces and retains the four original __main__ verification harnesses unchanged. The supplied briefing is included for context. Compared with the starter, this version fixes the DFS last-budget-child case, validates scores and lengths, stabilizes the geometric mean, adds canonical repetition checks and provides a working integration runner and regression tests. The no-rollout estimator behavior is explicitly changed from a silent zero label to ValueError.

Code, tests and report preparation were AI-assisted using Codex. Reported execution results were generated locally. The roster carries forward earlier role assignments and does not claim individual human sign-off. Fixtures and mock model outputs are explicitly identified; no empirical LLM or spacecraft-performance claim is made.

### References

[1. Instructor Mission 07 brief, starter modules and report template - pinned source](https://github.com/Unisvet/couse-c-ai/tree/43ad727a8b221118cc64a1c8e3f5ccbf6ff04105/mission-07)

[2. Yao et al. (2023), Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)

[3. Lightman et al. (2023), Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

[4. Wang et al. (2024), Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)

The research references motivate search and process supervision; their published performance is not presented as a result of this submission. Source review date: 21 November 2026.
