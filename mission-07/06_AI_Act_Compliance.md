# Safety and compliance review

Mission 07 | Squadron Colby | Academic reasoning-search simulation

## Scope

This review records engineering controls and their limits. Articles 14 and 15 of the EU AI Act address human oversight and accuracy, robustness and cybersecurity for high-risk systems. They are used here as review themes; this educational implementation is not presented as a legally certified or deployment-ready system. Legal classification depends on intended use and operating context.

| Control | Executed or inspectable evidence | Limit |
| --- | --- | --- |
| Compute bounds | Exact budget tests include rejected children and the final allowed child. MCTS has 12 iterations and a 32-candidate cap. | Generator latency and external model token use need separate bounds. |
| Validity and robustness | All four harnesses and 22 tests pass. Two untagged algebra errors are caught; NaN values and mismatched score lists are rejected. | The rational checker supports a small grammar; default PRM scores are heuristic. |
| Reward hacking | Filler score 0.96 falls to 0.192 and is cut at step 2. Labelled and nonconsecutive repeated equations are tested. | Paraphrases may evade matching; valid repetition may trigger a false positive. |
| Oversight and records | Paths, pruning reasons, source hashes and full verification output are supplied for review. | No operational human approval UI, immutable audit system or spacecraft control exists. |
| Data and transparency | Synthetic branches, priors and rollout answers are disclosed. No API keys, external tools or personal records are needed to run. | Fixture outcomes do not establish model accuracy or deployment safety. |

## Responsible use and provenance

The four course harnesses are retained and credited. Code, tests and reports were prepared with Codex assistance. Execution evidence is from actual local runs; mock scores and fixed rollout answers are labelled. Role allocation follows earlier submissions and does not claim human signatures. Reports are intended for academic review, with final submission handled by the squadron.

## Review references

[European Commission AI Act Service Desk - Article 14: Human oversight](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14)

[European Commission AI Act Service Desk - Article 15: Accuracy, robustness and cybersecurity](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-15)

Reviewed 21 November 2026. Detailed evidence: mission-07-doc.md and evidence/verification.txt.
