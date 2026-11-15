# Mission 06 · Autonomous warp state machine and deliberation loop

Squadron Colby · Conversational AI · Winter 2026/27

Open **ps06_solution.ipynb** in Jupyter or VS Code with a Python 3.11+ kernel, then run all cells in order. The solution uses the Python standard library. The notebook contains the complete implementation, a simulated recovery flight, a transition matrix and four assertion suites. No token or service account is needed.

The controller distinguishes state transitions from tool actions. A successful recovery takes 18 transitions for four actions, with discounted return 10.295. The fixed reward test returns 22.635. The sentinel stops a three-state cycle at observation six. See [the captured run](05_Run_Log.txt).

The attached brief and the [course folder](https://github.com/Unisvet/couse-c-ai/tree/43ad727a8b221118cc64a1c8e3f5ccbf6ff04105/mission-06) specify the task. No historical jump file was included there. The notebook embeds a synthetic 12-jump fixture. To use supplied history, pass a text stream with columns state,action,next_state to read_history and then transition_matrix.

[Team roles](00_Team_Roles.md) · [Compliance review](06_AI_Act_Compliance.md)
