# Mission 1 Specification: The Architecture of Argument

**Starship Operational Unit:** USSE Noosphere Bridge Crew  
**Course:** Conversational AI — Winter Semester 2026/27  
**Official Settings Repo:** [https://github.com/Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai)  

---

### Mission Briefing

**TO:** All Bridge Crews, USSE Noosphere  
**FROM:** Expedition Command (AURA Diagnostics Interface)  
**SUBJECT:** URGENT: Rebuilding AURA's Dialogic Reasoning Engine  

**Situation:**  
Upon emerging from spatial transit, AURA’s central conversational matrix went offline. Initial telemetry shows that while AURA can parse sensory tokens, it cannot structure a coherent, goal-driven dialogue. To restore communication, we must rebuild AURA's foundational subroutines based on humanity's two primary dialogic protocols: **Discovery** (Socratic *Elenchus*) and **Persuasion** (Galilean Multi-Agent Debate).

**Objective:**  
Construct a runnable Dialogic Engine in Python using Visual Studio Code, Antigravity (`agy`), and modern LLMs (e.g., Gemini, ChatGPT, Claude) that models both the Socratic state-machine for uncovering truth and a Galilean multi-persona sandbox for debating complex ethical hypotheses.

---

### Part 1: Historical Archivist — *The Dialogic Alignment Brief*
* **Deliverable:** `01_Alignment_Brief.md`
* **Directives:**
  1. Compare dialogue as Discovery vs Persuasion based on Plato's *Euthyphro* and Galileo's *Dialogue*.
  2. Evaluate how modern LLMs (Gemini, ChatGPT, Claude) perform under Socratic probing.
  3. Define system prompts establishing AURA's Socratic discovery persona.

### Part 2: Protocol Specialist — *The Socratic State-Machine*
* **Deliverable:** `02_Socratic_Protocol_Map.md`
* **Directives:**
  1. Reverse-engineer Socratic elenchus into a state machine diagram (Input claim $P_0$, probing steps, contradiction detection).
  2. Define loop refinement $P_{k+1}$ and stopping criteria (Refined Definition vs Aporia).

### Part 3: Ethical Navigator — *The Galilean Ethics Sandbox*
* **Deliverable:** `03_Galilean_Sandbox.md`
* **Directives:**
  1. Cast 3 personas: Advocate (*Salviati*), Skeptic (*Simplicio*), Mediator (*Sagredo*).
  2. Design an AI ethics thought experiment (e.g., *Open-Sourcing Frontier Models*).
  3. Audit how multi-agent debate mitigates single-agent sycophancy or bias.

### Part 4: Core Architect — *Engine Implementation*
* **Deliverable:** `04_dialogic_engine_starter.py` / `ps01_solution.ipynb`
* **Directives:**
  1. Synthesize state machine and 3-agent debate loop in VS Code + Antigravity with modern LLMs (Gemini, ChatGPT, Claude).
  2. Format output logs cleanly and manage squad repository layout.
