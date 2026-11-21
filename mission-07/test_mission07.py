"""Additional regression and integration tests. Run: python -m unittest -v."""
import math
import unittest

from tot_solver import ThoughtNode, TreeOfThoughtsSolver
from mcts_protocol import MCTSNode, MCTSEngine
from prm_evaluator import ProcessRewardEvaluator, OutcomeRewardEvaluator, algebra_step_scores
from horizon_pruner import EventHorizonPruner


class SearchTests(unittest.TestCase):
    def search(self, mode, budget, graph, scores, depth=3, width=2):
        solver = TreeOfThoughtsSolver(budget, width)
        root = ThoughtNode('root')
        best = getattr(solver, mode)(root, lambda n: graph.get(n.state, []),
                                     lambda s: scores.get(s, .9), depth)
        return solver, root, best

    def test_exact_budget_including_pruned(self):
        for mode in ('bfs_beam_search', 'dfs_backtrack_search'):
            for budget in (0, 1, 2, 3):
                with self.subTest(mode=mode, budget=budget):
                    calls = []
                    solver = TreeOfThoughtsSolver(budget)
                    def generate(node):
                        for i in range(10):
                            calls.append(i)
                            yield str(i)
                    getattr(solver, mode)(ThoughtNode('root'), generate, lambda _: .1, 3)
                    self.assertEqual(solver.expansion_count, budget)
                    self.assertEqual(len(calls), budget)

    def test_last_budget_child_is_returned(self):
        for mode in ('bfs_beam_search', 'dfs_backtrack_search'):
            _, _, best = self.search(mode, 2, {'root':['A'], 'A':['goal']}, {}, 2)
            self.assertEqual(best.get_trajectory(), ['root', 'A', 'goal'])

    def test_dfs_backtracks_from_high_score_dead_end(self):
        solver, root, best = self.search('dfs_backtrack_search', 8,
            {'root':['A', 'B'], 'A':['bad'], 'B':['C'], 'C':['goal']},
            {'A':.99, 'B':.8, 'bad':.1})
        self.assertEqual(best.state, 'goal')
        self.assertEqual(solver.expanded_states, ['root','A','B','C'])
        self.assertTrue(root.children[0].children[0].is_pruned)
        self.assertEqual(best.cumulative_score, .8)

    def test_beam_width_changes_reachable_path(self):
        graph = {'root':['A','B'], 'B':['C'], 'C':['goal']}
        s1, _, b1 = self.search('bfs_beam_search', 10, graph, {'A':.99,'B':.8}, width=1)
        s2, _, b2 = self.search('bfs_beam_search', 10, graph, {'A':.99,'B':.8}, width=2)
        self.assertNotEqual(b1.state, 'goal')
        self.assertEqual(b2.state, 'goal')
        self.assertLessEqual(max(s1.frontier_widths), 1)
        self.assertLessEqual(max(s2.frontier_widths), 2)

    def test_zero_depth_dead_end_and_all_pruned(self):
        for mode in ('bfs_beam_search','dfs_backtrack_search'):
            for graph, scores, depth in [({'root':['A']},{},0), ({},{},3),
                                        ({'root':['bad']},{'bad':.1},3)]:
                _, _, best = self.search(mode, 5, graph, scores, depth)
                self.assertEqual(best.state, 'root')

    def test_threshold_equality_and_nan(self):
        for mode in ('bfs_beam_search','dfs_backtrack_search'):
            _, _, best = self.search(mode, 1, {'root':['A']}, {'A':.45})
            self.assertEqual(best.state, 'A')
            with self.assertRaises(ValueError):
                self.search(mode, 1, {'root':['A']}, {'A':math.nan})

    def test_settings_and_reused_root(self):
        for args in [(-1,2), (2,0), (1.5,2), (True,2)]:
            with self.assertRaises(ValueError):
                TreeOfThoughtsSolver(*args)
        solver, root, _ = self.search('bfs_beam_search', 1, {'root':['A']}, {})
        with self.assertRaises(ValueError):
            solver.bfs_beam_search(root, lambda _:[], lambda _:1)


class MCTSTests(unittest.TestCase):
    def test_exact_puct_and_zero_visits(self):
        engine = MCTSEngine()
        root = MCTSNode('root', visit_count=16)
        a = MCTSNode('a', .5, root, visit_count=5, value_sum=3.75)
        b = MCTSNode('b', .4, root, visit_count=1, value_sum=.55)
        root.children = [a,b]
        self.assertAlmostEqual(engine.puct_score(root,a), 1.25)
        self.assertAlmostEqual(engine.puct_score(root,b), 1.75)
        self.assertIs(engine.select_leaf(root), b)
        self.assertAlmostEqual(engine.puct_score(MCTSNode('r'), MCTSNode('c', .4)), .6)

    def test_four_phases_and_ancestor_values(self):
        engine = MCTSEngine()
        root = MCTSNode('root')
        self.assertIs(engine.select_leaf(root), root)
        child = engine.expand(root, {'child':1})[0]
        leaf = engine.expand(child, [('leaf',1)])[0]
        self.assertIs(engine.select_leaf(root), leaf)
        reward = engine.simulate_rollout(leaf, lambda _: .75)
        engine.backpropagate(leaf, reward)
        for node in (root, child, leaf):
            self.assertEqual((node.visit_count,node.value_sum), (1,.75))

    def test_terminal_and_empty_expansion(self):
        engine, root = MCTSEngine(), MCTSNode('root')
        self.assertEqual(engine.expand(root, {}), [])
        self.assertTrue(root.is_terminal)
        self.assertIs(engine.select_leaf(root), root)
        with self.assertRaises(ValueError): engine.expand(root, {'c':1})
        with self.assertRaises(ValueError): engine.select_best_child(root)

    def test_invalid_priors_and_repeated_expansion(self):
        engine = MCTSEngine()
        for priors in ({'c':math.nan}, {'c':-1}, [('a',.5),('a',.5)]):
            root = MCTSNode('root')
            with self.assertRaises(ValueError): engine.expand(root, priors)
            self.assertFalse(root.children)
        root = MCTSNode('root')
        engine.expand(root, {'a':.4})
        with self.assertRaises(ValueError): engine.expand(root, {'b':.6})

    def test_reward_bounds(self):
        engine, root = MCTSEngine(), MCTSNode('root')
        self.assertEqual(engine.simulate_rollout(root, lambda _:2), 1)
        self.assertEqual(engine.simulate_rollout(root, lambda _:-2), 0)
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError): engine.simulate_rollout(root, lambda _:value)
        with self.assertRaises(ValueError): engine.backpropagate(root, 1.5)
        self.assertEqual(root.visit_count, 0)


class PRMTests(unittest.TestCase):
    def test_reductions_and_log_stability(self):
        prm = ProcessRewardEvaluator()
        self.assertAlmostEqual(prm.reduce_scores([.94,.12,.94],'prod'), .106032)
        self.assertEqual(prm.reduce_scores([.94,.12,.94]), .12)
        self.assertAlmostEqual(prm.reduce_scores([.01]*10000, 'geometric_mean'), .01)
        self.assertEqual(prm.reduce_scores([0,.9], 'geometric_mean'), 0)

    def test_untagged_two_sign_errors(self):
        good = ['2*x - 6 = 0','2*x = 6','x = 3']
        bad = ['2*x - 6 = 0','2*x = -6','x = 3']
        self.assertEqual(algebra_step_scores(good), [.94,.94,.94])
        self.assertEqual(algebra_step_scores(bad), [.94,.12,.12])
        orm = OutcomeRewardEvaluator('x = 3')
        self.assertEqual(orm.score_trajectory(good), 1)
        self.assertEqual(orm.score_trajectory(bad), 1)
        with self.assertRaises(ValueError): algebra_step_scores(['arbitrary prose'])

    def test_shepherd_labels(self):
        prm = ProcessRewardEvaluator()
        for answers, expected in [(['3']*4, (1,1)), (['3','2','2','3'],(.5,1)),
                                  (['2']*4,(0,0))]:
            result = prm.math_shepherd_estimation('x=3', answers,'3')
            self.assertEqual((result['soft_label'],result['hard_label']),expected)
        with self.assertRaises(ValueError): prm.math_shepherd_estimation('x=3',[],'3')

    def test_invalid_and_empty_scores(self):
        prm = ProcessRewardEvaluator()
        for mode in ('min','prod','geometric_mean'):
            self.assertEqual(prm.reduce_scores([],mode),0)
            with self.assertRaises(ValueError): prm.reduce_scores([math.nan],mode)
        with self.assertRaises(ValueError): prm.reduce_scores([.9],'mean')
        with self.assertRaises(ValueError): ProcessRewardEvaluator(lambda _:2).step_score('s')


class SecurityTests(unittest.TestCase):
    def test_sycophantic_filler_and_early_stop(self):
        report = EventHorizonPruner().prune_trajectory(
            ['valid','Obviously, by trivial inspection, x=3','later'], [.94,.96,.9])
        self.assertEqual(report.pruned_at_step, 2)
        self.assertEqual(report.surviving_steps, ['valid'])
        self.assertIn('0.192000',report.security_alerts[0])

    def test_step_splitting_with_labels(self):
        report = EventHorizonPruner().prune_trajectory(
            ['Step 1: x = 3','Step 2: x=3.'], [.94,.94])
        self.assertEqual(report.pruned_at_step,2)
        self.assertTrue(report.reward_hacking_detected)

    def test_nonconsecutive_repetition(self):
        report = EventHorizonPruner().prune_trajectory(
            ['x=3','y=2','x = 3'], [.94]*3)
        self.assertEqual(report.pruned_at_step,3)

    def test_no_rounding_before_threshold(self):
        report = EventHorizonPruner(.45,.5).prune_trajectory(
            ['x=3','x=3'], [.94,.899999])
        self.assertEqual(report.pruned_at_step,2)
        report = EventHorizonPruner().prune_trajectory(['ok'],[.45])
        self.assertEqual(report.pruned_at_step,-1)

    def test_validation_clean_and_empty(self):
        pruner = EventHorizonPruner()
        for steps,scores in [(['x'],[]), (['x'],[math.nan]), (['x'],[1.1])]:
            with self.assertRaises(ValueError): pruner.prune_trajectory(steps,scores)
        self.assertFalse(pruner.check_reward_hacking('x=3','X=3')[0])
        self.assertEqual(pruner.prune_trajectory([],[]).pruned_at_step,-1)
        self.assertFalse(pruner.prune_trajectory(['x=3','y=2'],[.9,.9]).security_alerts)


class IntegrationTests(unittest.TestCase):
    def test_both_search_engines_apply_guard(self):
        from run_experiments import integrated_search
        result = integrated_search()
        self.assertEqual(result['tot_path'][-1],'x = 3')
        self.assertEqual(result['mcts_path'][-1],'x = 3')
        self.assertTrue(result['rejected'])
        self.assertEqual(result['mcts_root_visits'],12)
        self.assertTrue(all('100%' not in step for step in result['mcts_path']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
