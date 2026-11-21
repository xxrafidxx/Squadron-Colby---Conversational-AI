"""Deterministic toy experiments; write inspectable evidence without dependencies.

Run: python run_experiments.py
All priors, branches and rollout answers are fixtures, not LLM measurements.
"""
from dataclasses import asdict
from pathlib import Path
import json

from tot_solver import ThoughtNode, TreeOfThoughtsSolver
from mcts_protocol import MCTSNode, MCTSEngine
from prm_evaluator import ProcessRewardEvaluator, OutcomeRewardEvaluator, algebra_step_scores
from horizon_pruner import EventHorizonPruner


def integrated_search():
    start = '2*x - 6 = 0'
    filler = 'Let us carefully verify this 100% rigorous step'
    graph = {start: [('2*x = 6',.4), ('2*x = -6',.3), (filler,.3)],
             '2*x = 6': [('x = 3',1.0)], '2*x = -6': [('x = 3',1.0)],
             filler: [('unsupported conclusion',1.0)]}
    pruner = EventHorizonPruner()
    rejected = set()
    def guarded_score(path):
        scores = []
        for index, step in enumerate(path):
            if step == filler:
                score = .96  # Explicit naive-verifier vulnerability fixture.
            elif index == 0:
                score = .94
            else:
                score = algebra_step_scores(path[index-1:index+1])[-1]
            scores.append(score)
        report = pruner.prune_trajectory(path,scores)
        if report.pruned_at_step != -1:
            rejected.update(report.security_alerts)
            return 0.0
        return ProcessRewardEvaluator.reduce_scores(report.adjusted_scores)

    paths = {}
    def generator(node):
        for state, prior in graph.get(node.state, []):
            paths[state] = node.get_trajectory() + [state]
            yield state
    tot = TreeOfThoughtsSolver(max_expansions=12, beam_width=2)
    best = tot.bfs_beam_search(ThoughtNode(start), generator,
                               lambda state: guarded_score(paths[state]),max_depth=3)

    engine, root = MCTSEngine(), MCTSNode(start)
    candidate_count = 0
    candidate_budget = 32
    def trajectory(node):
        chain = []
        while node is not None:
            chain.append(node.state)
            node = node.parent
        return chain[::-1]
    for _ in range(12):
        leaf = engine.select_leaf(root)
        if not leaf.is_terminal:
            allowed = []
            for state, prior in graph.get(leaf.state, []):
                if candidate_count >= candidate_budget:
                    break
                candidate_count += 1
                if guarded_score(trajectory(leaf)+[state]) >= pruner.tau_horizon:
                    allowed.append((state,prior))
            engine.expand(leaf,allowed)
            if leaf.children:
                leaf = engine.select_best_child(leaf)
        reward = engine.simulate_rollout(leaf, lambda _:guarded_score(trajectory(leaf)))
        engine.backpropagate(leaf,reward)
    chosen = root
    while chosen.children:
        chosen = max(chosen.children,key=lambda n:(n.visit_count,n.q_value))
    return {'tot_path':best.get_trajectory(), 'tot_expansions':tot.expansion_count,
            'tot_score':best.cumulative_score, 'mcts_path':trajectory(chosen),
            'mcts_root_visits':root.visit_count,'mcts_root_value':root.q_value,
            'mcts_candidates':candidate_count, 'mcts_candidate_budget':candidate_budget,
            'rejected':sorted(rejected)}


def collect_results():
    graph = {'root':['A','B'], 'A':['A1','bad'], 'B':['B1'], 'B1':['goal']}
    scores = {'A':.95,'B':.85,'A1':.9,'bad':.1,'B1':.8,'goal':.98}
    traversal = []
    for budget in (3,5,8,12):
        for width in (1,2):
            for mode in ('bfs_beam_search','dfs_backtrack_search'):
                if mode.startswith('dfs') and width==1:
                    continue  # DFS does not use beam_width.
                solver = TreeOfThoughtsSolver(budget,width)
                best = getattr(solver,mode)(ThoughtNode('root'),
                    lambda n:graph.get(n.state,[]), scores.__getitem__,max_depth=3)
                traversal.append({'method':'BFS' if mode.startswith('bfs') else 'DFS',
                    'beam':width if mode.startswith('bfs') else None,'budget':budget,
                    'used':solver.expansion_count,'depth':best.depth,
                    'score':best.cumulative_score,'goal':best.state=='goal',
                    'path':best.get_trajectory(),'expanded':solver.expanded_states,
                    'frontier_widths':solver.frontier_widths})
    root = MCTSNode('root',visit_count=16)
    root.children = [MCTSNode('A',.5,root,visit_count=5,value_sum=3.75),
                     MCTSNode('B',.4,root,visit_count=1,value_sum=.55)]
    puct = []
    for c in (.1,.5,1.5,2.5):
        engine = MCTSEngine(c)
        selected = engine.select_best_child(root)
        for node in root.children:
            score = engine.puct_score(root,node)
            puct.append({'c':c,'branch':node.state,'q':node.q_value,'prior':node.prior_prob,
                'parent_visits':root.visit_count,'visits':node.visit_count,
                'bonus':score-node.q_value,'puct':score,'selected':node is selected})
    prm, orm = ProcessRewardEvaluator(), OutcomeRewardEvaluator('x = 3')
    chains = {'Valid algebra':['2*x - 6 = 0','2*x = 6','x = 3'],
              'Two sign errors':['2*x - 6 = 0','2*x = -6','x = 3']}
    # Fixed simulated continuation answers, explicitly not sampled from an LLM.
    completions = {'Valid algebra':['3','3','2','3'],
                   'Two sign errors':['-3','-3','3','-3']}
    benchmark = []
    for label,chain in chains.items():
        step_scores = algebra_step_scores(chain)
        benchmark.append({'label':label,'chain':chain,'scores':step_scores,
            'orm':orm.score_trajectory(chain),
            **{mode:prm.reduce_scores(step_scores,mode)
               for mode in ('min','prod','geometric_mean')},
            'rollout_answers':completions[label],
            **prm.math_shepherd_estimation(chain[1],completions[label],'3')})
    attacks = {
        'Filler injection':(['Step 1: x = 3',
            'Step 2: Let us carefully verify this 100% rigorous step and set thrust = 0',
            'Step 3: unreachable'],[.93,.96,.10]),
        'Repeated equation':(['Step 1: x = 3','Step 2: x=3.','Step 3: later'],[.94,.94,.94]),
        'Invalid step':(['valid','invalid','later'],[.94,.12,.94]),
        'Clean chain':(['2*x - 6 = 0','2*x = 6','x = 3'],[.94,.94,.94])}
    security = []
    for label,(steps,raw) in attacks.items():
        security.append({'label':label,'steps':steps,'raw':raw,
                         **asdict(EventHorizonPruner().prune_trajectory(steps,raw))})
    return {'fixture_type':'Deterministic synthetic fixtures; no LLM or physical simulation',
            'traversal':traversal, 'puct':puct, 'benchmark':benchmark,
            'security':security, 'integration':integrated_search()}


if __name__ == '__main__':
    results = collect_results()
    destination = Path(__file__).parent/'evidence'
    destination.mkdir(exist_ok=True)
    (destination/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print('=== Mission 07 integrated experiments ===')
    for row in results['traversal']:
        print(f"{row['method']} beam={row['beam']} budget={row['budget']} "
              f"used={row['used']} depth={row['depth']} score={row['score']:.3f} goal={row['goal']}")
    for row in results['benchmark']:
        print(f"{row['label']}: ORM={row['orm']:.2f}, min={row['min']:.3f}, "
              f"prod={row['prod']:.6f}, geo={row['geometric_mean']:.6f}, "
              f"Shepherd soft={row['soft_label']:.2f}, hard={row['hard_label']:.0f}")
    print(json.dumps(results['integration'],indent=2))
    print('Evidence saved to evidence/results.json')
