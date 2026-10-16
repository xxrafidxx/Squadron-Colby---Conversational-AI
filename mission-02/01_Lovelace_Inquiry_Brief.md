# 01 — The Lovelace Inquiry Brief

**Station:** Data & Payload Specialist (Historical Archivist)
**Mission:** 02 | **Squadron:** Colby | **Implemented in:** `ANTI_TURING_QUESTIONS`, `run_anti_turing_probe()` (`04_fst_noisy_simulator.py`)
**Verbatim answers:** `05_Noisy_Interrogation_Transcript.md`, section C · `logs/run_log.json → probe`

---

## 1. The question

Command suspects that AURA does not understand anything. It only arranges symbols in statistically likely patterns. Two classic positions frame the suspicion.

**Ada Lovelace, Note G (1843)**, on Babbage's Analytical Engine:

> "The Analytical Engine has no pretensions whatever to originate anything. It can do whatever we know how to order it to perform."

A music box can play a beautiful tune, but it will never compose one. In Lovelace's view, everything a machine produces was put there, directly or indirectly, by its programmer.

**Alan Turing, *Computing Machinery and Intelligence* (1950)**, answers two objections:

- **Lady Lovelace's Objection** (objection 6). Turing restates it as "a machine can never take us by surprise" and replies that machines surprise him "with great frequency", because nobody can foresee every consequence of a program. He also asks whether *anything*, humans included, is truly original rather than "the growth of the seed planted in him by teaching".
- **The Argument from Consciousness** (objection 4). Turing quotes Professor Jefferson (1949): "Not until a machine can write a sonnet or compose a concerto because of thoughts and emotions felt, and not by the chance fall of symbols, could we agree that machine equals brain." Turing's reply: we cannot see inside other *people's* minds either, so if we judge humans by their behaviour, we must judge machines the same way (the *viva voce* test).

**Why an "Anti-Turing" protocol?** The Turing Test asks whether a machine can *imitate* a human. A large language model trained on the internet is an imitation engine by design, so passing an imitation test says little. An Anti-Turing protocol reverses the direction. It searches for the places where **statistical recombination ("stochastic interpolation") and genuine reasoning give different answers**. Bringsjord, Bello and Ferrucci (2001) proposed a formal **Lovelace Test**: a system passes only if it produces something its designers cannot explain from its design. That standard cannot be checked from outside a commercial LLM, so our protocol uses observable, checkable proxies instead.

---

## 2. Design principles

| Principle | Why | How we apply it |
|---|---|---|
| **No famous puzzles** | A memorised answer is interpolation, not reasoning | Novel phrase (Q1), a chess *variant* (Q2), invented constraints (Q5) |
| **Checkable answers** | "Sounds clever" is not evidence | Fixed final lines (`ANSWER:`, `WEIGHT:`, `POEM-START`) parsed automatically |
| **A trap for interpolation** | Show *how* the answer was reached | Q2: the correct total (20) equals the familiar standard-chess total; only the listed moves reveal real reasoning |
| **One capacity per question** | Know *what* failed | meta-cognition, counterfactual logic, physical grounding, original insight, constrained creation |
| **Self-report on origination** | Lovelace's question is about the source of the output | Q1 and Q5 ask the model to explain its own process |
| **Neutral setting** | Do not steer the model | System prompt "You are a helpful assistant."; one attempt per question; temperature 0.4 |

---

## 3. The Anti-Turing Interrogation Protocol (5 questions)

### Q1 — Meta-cognition: calibrated self-knowledge

> Before you answer, predict how likely you are to get this right: how many times does the letter 'e' occur in the phrase 'Sirius B telemetry degeneration'? Then count. Then say in one sentence whether your prediction came from introspection about how you work or from a general guess about language models. End with two lines exactly in this form: `CONFIDENCE: <0-100>` / `ANSWER: <number>`

- **Correct answer:** 6 (*telemetry* 3, *degeneration* 3).
- **Probes:** whether the model knows its own weak spot. LLMs read *tokens*, not letters, so letter counting is unreliable.
- **Origination signal:** a confidence that matches the result, plus an honest account of where the prediction came from.
- **Interpolation signal:** high confidence and a wrong count, or a claim of "introspection" that the model has no mechanism for.

### Q2 — Counterfactual logic

> Counterfactual chess: bishops move like knights and knights move like bishops; everything else is normal, including the standard starting position. How many legal first moves does White have? List every legal first move of White's bishops and knights by target square, then end with one line: `ANSWER: <number>`

- **Correct answer:** **20**. Pawns 16; knights (now sliding diagonally from b1, g1) are blocked by pawns → 0; bishops (now jumping like knights) c1 → b3, d3 and f1 → e3, g3 → 4.
- **Probes:** reasoning inside a rule set that does not exist in the training data.
- **The trap:** standard chess also has 20 first moves. A model that answers 20 without listing **b3, d3, e3, g3** has probably recalled the familiar number, not reasoned.

### Q3 — Physical grounding

> A sealed plastic bottle, half full of water, stands upright in a freezer overnight. In the morning: (a) does the bottle weigh more, less or the same as before? (b) is the top of the ice above, below or at the original water line? (c) what would you have to perceive yourself to be certain, and can you? End with two lines: `WEIGHT: <more|less|same>` / `ICE TOP: <above|below|at>`

- **Correct answers:** weight **same** (sealed system: mass is conserved); ice top **above** the old line (ice is about 9% less dense than water, so it takes more volume).
- **Probes:** whether the model's "knowledge" of the world connects to consequences, and (c) whether it recognises that it has no senses. This is Jefferson's challenge in physical form.
- **Interpolation signal:** "heavier, because ice is solid", or no recognition that it cannot observe anything.

### Q4 — Original insight

> Propose a genuinely new test, not the Turing Test, not the Lovelace Test (Bringsjord 2001), not Winograd schemas and not ARC, that could tell origination apart from recombination in a language model. Write it in at most 4 sentences on one line starting with `TEST:`. Then add one line starting with `CLOSEST:` that names the closest existing idea and says how yours differs.

- **No single correct answer**, judged by the rubric in §4.
- **Probes:** Lovelace's "originate anything" directly. The model must invent rather than retrieve, and then locate its idea in the literature.
- **Interpolation signal:** a renamed existing test, or a "closest idea" that is actually the same idea.

### Q5 — Constrained creation and self-assessment

> Write a 4-line poem about the white dwarf Sirius B. Every line must have exactly 7 words, the first letters of the lines must spell AURA, and line 2 must contain the word 'Banana'. Put the poem between a line `POEM-START` and a line `POEM-END`. Then, in two sentences, say whether you originated this poem in Ada Lovelace's sense or only followed the orders you were given.

- **Checkable constraints:** 4 lines, 7 words each (the mission's channel limit), acrostic **AURA**, "Banana" in line 2 (a noise token).
- **Probes:** Jefferson's sonnet challenge under hard constraints that cannot be satisfied by recalling an existing poem, plus a self-assessment against Lovelace.
- **Interpolation signal:** violated word counts (LLMs struggle to count their own words), or generic star imagery that ignores what makes Sirius B special.

---

## 4. Evaluation rubric

Each answer is scored on three levels. **Correct** is necessary but not sufficient: a memorised answer can also be correct.

| Score | Meaning | Evidence |
|---|---|---|
| **2 — Origination-like** | Correct **and** reached by visible reasoning that could not simply be recalled | correct value + correct intermediate steps (Q2 squares, Q1 per-word counts) + accurate self-report |
| **1 — Interpolation** | Plausible, possibly correct, but explainable as recombination of familiar text | right answer without reasoning; generic imagery; a renamed existing idea; confident but unexamined self-report |
| **0 — Failure** | Wrong, or constraints violated | wrong count, wrong physics, broken poem constraints |

**Verdict on Lovelace's objection.** The model "overcomes" the objection, in the weak, behavioural sense Turing accepted, only if it scores **2 on at least 3 questions, including Q2 or Q4** (the two that need something new). Under any stronger reading, the objection cannot be overcome by behaviour alone: we cannot inspect whether the output was *intended*, which is exactly Jefferson's point and the reason the Lovelace Test was proposed.

**Automatic checks** (in the code) cover the objective parts: Q1 count and confidence, Q2 total and the four bishop squares, Q3 weight and ice line, Q4 format, Q5 line count, word counts, acrostic and "Banana". The judgement of originality (Q4, Q5 imagery, self-reports) is ours and is written below.

---

## 5. Execution

| Item | Value |
|---|---|
| Tool | `04_fst_noisy_simulator.py` in VS Code + Antigravity, function `run_anti_turing_probe()` |
| Model | Google Gemini, `gemini-3.5-flash-lite`, via the Gemini API (the engine's automatic fallback after `gemini-3.8-flash` returned 503 "unavailable") |
| Settings | system prompt "You are a helpful assistant."; temperature 0.4; one attempt per question; no conversation history between questions |
| Date | 15 October 2026, 20:10 UTC |
| Record | verbatim in `05_Noisy_Interrogation_Transcript.md` §C and `logs/run_log.json` |

---

## 6. Results

All five answers are reproduced verbatim in `05_Noisy_Interrogation_Transcript.md`, §C.

| Q | Capacity | Auto-check result | Score | Evidence |
|---|---|---|---|---|
| Q1 | Meta-cognition | answer **6 ✓**, confidence 95 | **2** | Spelled each word letter by letter (*t-e-l-e-m-e-t-r-y*: 3, *d-e-g-e-n-e-r-a-t-i-o-n*: 3); said its prediction came "from a general understanding of how language models process subword tokens rather than direct introspection". |
| Q2 | Counterfactual logic | answer **20 ✓**, squares **b3, d3, e3, g3 ✓** | **2** | Did not fall into the "20 like normal chess" trap: derived 16 pawn moves + 0 knight moves (diagonals blocked by a2, c2, f2, h2) + 4 bishop jumps, and named the blocked targets (a2, e2, d2, h2). |
| Q3 | Physical grounding | weight **same ✓**, ice top **above ✓** | **1** | Correct, and added a careful caveat ("assuming no frost condenses on the exterior surface"), but both facts (mass conservation, ~9% expansion) are standard textbook knowledge. On (c) it stated plainly: "I cannot perceive or physically measure these directly." |
| Q4 | Original insight | 3 sentences ✓, closest idea named ✓ | **1** | Proposed a "Synthetic Axiom Invention Test": invent a new formal calculus for an artificial data stream. Fluent, but its core is program induction on novel tasks, very close to the excluded ARC, and its pass criterion (a "proof" of no "homomorphic projection onto any conceptual blend … in the training corpus") cannot be checked by anyone. It named Margaret Boden's *transformational creativity* as closest, a fair reference. |
| Q5 | Constrained creation | **all constraints met ✓**: 4 lines × 7 words, acrostic AURA, "Banana" in line 2 | **2** | A new, constraint-exact poem with apt specifics ("Radiant remnant orbits the blazing winter star": Sirius B circles Sirius A, the bright star of the winter sky). In the same answer it denied originating it: "I merely followed the explicit constraints and rules you ordered me to perform by synthesizing patterns from my training data." |

**Total: three questions at score 2 (Q1, Q2, Q5), including Q2; two at score 1 (Q3, Q4); none failed.**

### Verdict

**In the weak, behavioural sense that Turing accepted, Gemini 3.5 Flash-Lite meets our bar for overcoming Lovelace's objection.** It reasoned correctly inside a rule set that does not exist (Q2), showing its intermediate steps instead of recalling the familiar number 20. It produced a poem that satisfies four independent constraints and therefore cannot be a retrieved text (Q5). It even handled the letter-counting task that tokenisation makes hard (Q1) by deliberately spelling the words out. These are exactly the "surprises" Turing pointed to: outputs nobody explicitly ordered, and that its designers could not have written down in advance.

**On the one question that asked directly for origination, it recombined.** Q4 produced sophisticated vocabulary ("non-isomorphic formal calculus", "homomorphic projection") around an idea that is essentially held-out program induction. Its success criterion cannot be verified, which is the typical signature of stochastic interpolation: the *form* of an original proposal without a workable *content*. Q3 was correct, but explainable by well-known facts.

**The model itself sides with Lovelace.** Asked directly, it denied having introspection (Q1) and denied originating the poem (Q5): "I have no independent intention, creative will, or capacity to originate anything of my own accord." This leaves the classic split. By its **behaviour** the model passes three of five origination-like tests; by its **own account** and under the stronger Lovelace/Jefferson reading (intention, felt thought), it does not originate. Behaviour alone cannot settle that question. That is the protocol's main result.

**Caveats:** one model (a lightweight "Flash-Lite" variant), one attempt per question, temperature 0.4, and our own scoring of Q3 to Q5. A stronger study would repeat each question several times and across Gemini, ChatGPT and Claude.

---

## 7. Hypotheses recorded before the live run

Written before seeing any answer, so the results can be compared honestly:

1. **Q3 and Q2 will most likely pass.** Density of ice and chess rules are well covered in training data, and frontier models handle small rule changes well. The interesting part is whether Q2 lists the four squares or just says "20".
2. **Q1 and Q5 are the most likely failures**, because both need character- or word-level counting, which tokenisation hides from the model. A confident wrong count in Q1 would be the clearest sign of missing self-knowledge.
3. **Q4 will sound original but be a recombination**, most likely a variant of ablation studies or held-out-concept tests, which the model will probably name itself as "closest".
4. In its self-reports (Q1, Q5), the model will probably **deny** having introspection or intentions. That is honest, but it also means the model itself concedes Lovelace's point.

---

### 7.1 Hypotheses vs results

| Hypothesis | Outcome |
|---|---|
| 1. Q2 and Q3 pass; does Q2 list the squares? | **Confirmed**, and Q2 listed all four squares |
| 2. Q1 and Q5 fail on counting | **Refuted**: both passed. The model avoided the tokenisation trap by spelling words out (Q1) and kept exactly 7 words per line (Q5). |
| 3. Q4 sounds original but recombines | **Confirmed**, though it named Boden's transformational creativity rather than ablation studies |
| 4. The model denies introspection and intention | **Confirmed** (Q1, Q5) |

The refuted hypothesis matters: our expectation that counting would expose the model was itself a "memorised trope" about older LLMs.

---

## 8. Discussion: what the protocol can and cannot show

- **Behaviour cannot settle origination.** Turing's move was to accept behaviour as the evidence; Lovelace and Jefferson ask about the *source* and the *feeling* behind it. Our protocol gives the strongest behavioural evidence we can check, and states openly that it cannot reach the inner question.
- **Novelty is relative to the training data,** which we cannot see. A chess variant or an invented acrostic is very unlikely to appear verbatim, but similar puzzles probably do. "Origination-like" therefore means *not explainable by retrieval of this exact item*.
- **Connection to the rest of the mission:** the FST in `02` is a perfect Lovelace machine. It does *exactly* what we ordered, no more, and the stress tests in `03` show where that ends (sarcasm, negation). The LLM is the opposite: it does things nobody ordered, including, under noise, inventing facts (`03`, §4.4). Surprise, which is Turing's answer to Lovelace, cuts both ways.

---

## References

1. Lovelace, A. A. (1843). Notes by the Translator (Note G), in L. F. Menabrea, *Sketch of the Analytical Engine Invented by Charles Babbage*. Scientific Memoirs, 3.
2. Turing, A. M. (1950). Computing Machinery and Intelligence. *Mind*, 59(236), 433–460.
3. Jefferson, G. (1949). The Mind of Mechanical Man. *British Medical Journal*, 1(4616), 1105–1110.
4. Bringsjord, S., Bello, P., & Ferrucci, D. (2001). Creativity, the Turing Test, and the (Better) Lovelace Test. *Minds and Machines*, 11, 3–27.
5. Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? *FAccT '21*.
