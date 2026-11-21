"""The holdout hook: threading, logging and the gate-facing record.

These use stubs rather than a PDE solve: the point is that the optimizer calls
the caller's hook with the incumbent q every iteration and surfaces the results
in ``OptResult.holdout``, which is what the run-level trust check depends on.
"""
from __future__ import annotations

import unittest

import numpy as np

from stoch.optimize import solve_saa


class _Eval:
    def __init__(self, q) -> None:
        self.mean_J = float(np.sum(np.asarray(q) ** 2))

    def summary(self) -> dict:
        return {"mean_J": self.mean_J, "reliability": 1.0, "cvar_G": -1.0}


class _StubProblem:
    """f(q) = sum(q^2): smooth, cheap, and satisfiable by a compass search."""

    def __init__(self, n_wells: int = 3, q_max: float = 10.0,
                 Q_max: float = 20.0) -> None:
        self.n_wells = n_wells
        self.q_max = q_max
        self.Q_max = Q_max
        self.alpha = 0.05
        self.alpha_h = 0.01
        self.archive = None
        self.supports_batching = False
        self.n_eval = 0

    def objective(self, q, keep: bool = False) -> float:
        self.n_eval += 1
        return float(np.sum(np.asarray(q) ** 2))

    def evaluate(self, q, keep: bool = False):
        return _Eval(q)


class TestHoldoutHook(unittest.TestCase):
    def test_hook_called_every_iteration_and_recorded(self):
        prob = _StubProblem()
        seen = []

        def on_iter(q, nit, best, start=1):
            seen.append((np.array(q, copy=True), int(nit), float(best), int(start)))
            return {"iteration": int(nit), "start": int(start), "q": list(map(float, q))}

        res = solve_saa(prob, method="pattern", n_starts=1, max_iter=4,
                        max_evals=200, seed=0, on_iter=on_iter,
                        init_fraction=0.5)
        self.assertTrue(seen, "hook was never called")
        self.assertEqual(len(res.holdout), len(seen),
                         "one holdout record per iteration expected")
        # the hook sees the incumbent, and the incumbent improves
        f_first, f_last = seen[0][2], seen[-1][2]
        self.assertLessEqual(f_last, f_first + 1e-12)
        self.assertEqual(seen[0][3], 1, "start index must be 1-based")

    def test_no_hook_means_no_records(self):
        prob = _StubProblem()
        res = solve_saa(prob, method="pattern", n_starts=1, max_iter=3,
                        max_evals=100, seed=0, init_fraction=0.5)
        self.assertEqual(res.holdout, [])

    def test_hook_may_return_none(self):
        prob = _StubProblem()

        def on_iter(q, nit, best, start=1):
            return {"iteration": int(nit)} if nit % 2 == 0 else None

        res = solve_saa(prob, method="pattern", n_starts=1, max_iter=4,
                        max_evals=200, seed=0, on_iter=on_iter,
                        init_fraction=0.5)
        self.assertTrue(all(r is not None for r in res.holdout))
        self.assertLessEqual(len(res.holdout), res.nit)

    def test_multiple_starts_are_tagged(self):
        prob = _StubProblem()
        starts = []

        def on_iter(q, nit, best, start=1):
            starts.append(int(start))
            return {"start": int(start), "iteration": int(nit)}

        res = solve_saa(prob, method="pattern", n_starts=2, max_iter=2,
                        max_evals=80, seed=0, on_iter=on_iter,
                        init_fraction=0.5)
        self.assertEqual(sorted(set(starts)), [1, 2])
        self.assertEqual(len(res.holdout), len(starts))


if __name__ == "__main__":
    unittest.main()
