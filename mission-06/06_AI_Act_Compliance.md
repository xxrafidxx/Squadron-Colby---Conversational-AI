# Mission 06 | Safety and compliance review

Articles 14 and 15 of the EU AI Act address human oversight and robustness for high-risk systems. This coursework uses those topics to evaluate a deterministic simulation. The four passing suites establish behavior on the tested inputs. A legal classification or deployment assessment would require an intended use, operating context and accountable operator.

| Control | Evidence from the notebook | Limit of the evidence |
|---|---|---|
| Execution budget | The 21-call test halts at transition 20. A later call leaves the terminal state intact. | A step count bounds work between calls. A blocking external tool would need its own timeout. |
| Input and tool errors | Schema checks reject unknown tools, missing or extra fields, invalid types and out-of-range settings. A zero vector exercises the tool exception boundary. | Contracts describe structure and ranges. A syntactically valid vector can still be unsuitable. |
| Repeated actions | A hash covers shield state, clearance, completion and the full proposal. The period-three test halts on sample six. A repeated failed request is blocked before a third execution. | A changing observation or argument may hide a loop. The independent step cap still applies. |
| Oversight and trace | The agent records state transitions, observations, actions, rewards, stop reasons and critical cycle events. A reviewer can replay the included deterministic episode. | Records live in process memory. A deployment would need durable logs and an independent operator stop. |
| Data and external effects | Sensor results are simulated. The jump table contains twelve synthetic records. Tools update local state and require no credentials. | The measured frequencies describe the fixture only. They cannot estimate flight safety. |

https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14
https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-15
