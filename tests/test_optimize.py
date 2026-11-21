"""A non-finite objective in the search: fatal by default, skippable on request.

The case these tests encode: a float32 batched transport solve went NaN on
candidate 5 of 18 in a compass sweep, and ``_finite_or_raise`` ended a paid
two-hour GPU run that had already completed a full start.  One diverging
candidate is a rejected move; a whole sweep of them is still a broken backend,
and the second case must keep failing loudly.
"""
from __future__ import annotations

import unittest

import numpy as np

from stoch.optimize import (
    SAAProblem,
    nonfinite_policy,
    pattern_search,
    pattern_search_batched,
    solve_saa,
)

try:                                     # the batched path only exists with torch
    import torch
except Exception:                        # pragma: no cover - torch is a hard dep
    torch = None

BAD_ABOVE = 0.7


class _Eval:
    """What ``SAAProblem`` reads off one evaluation."""

    def __init__(self, mean_J: float, M: int = 4) -> None:
        self.mean_J = mean_J
        # G <= 0 and -H <= 0 everywhere, so both penalties are inactive and the
        # objective is just ``mean_J`` -- the stub stays a one-line function.
        self.G = np.full(M, -1.0)
        self.H = np.full(M, 1.0)
        self.M = M

    def summary(self) -> dict:
        return {"mean_J": self.mean_J, "reliability": 1.0, "cvar_G": -1.0}


class _DivergingArchive:
    """``f(q) = sum(q)``, except where the "transport solve" blows up.

    Any candidate above ``BAD_ABOVE`` on any well returns ``nan``, which is
    exactly what a stalled Krylov solve gives the optimizer.
    """

    supports_batching = False
    device = None

    def __init__(self, bad_above: float = BAD_ABOVE) -> None:
        self.bad_above = bad_above
        self.well_xy = np.zeros((2, 2))
        self.problems: list = []

    def evaluate(self, q, cost=None, keep: bool = False):
        q = np.asarray(q, dtype=float)
        if np.any(q > self.bad_above):
            return _Eval(float("nan"))
        return _Eval(float(q.sum()))

    def describe(self) -> dict:
        return {"transport_krylov": {"converged": False}}


def _problem(allow: bool, bad_above: float = BAD_ABOVE) -> SAAProblem:
    return SAAProblem(_DivergingArchive(bad_above), cost=None,
                      q_max=50.0, Q_max=200.0, allow_nonfinite=allow)


class TestNonfinitePolicy(unittest.TestCase):
    def test_finite_values_pass_through_untouched(self):
        values, n_bad = nonfinite_policy([1.0, 2.0, 3.0], allow=True)
        self.assertEqual(n_bad, 0)
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])

    def test_default_is_fatal_and_names_the_flag(self):
        with self.assertRaises(RuntimeError) as ctx:
            nonfinite_policy([1.0, np.nan], allow=False)
        message = str(ctx.exception)
        self.assertIn("is not finite", message)
        self.assertIn("--allow-nonfinite", message)

    def test_allowed_entries_become_inf_and_are_counted(self):
        values, n_bad = nonfinite_policy([np.nan, 2.0, np.inf], allow=True)
        self.assertEqual(n_bad, 2)
        self.assertEqual(values[0], np.inf)
        self.assertEqual(values[1], 2.0)
        self.assertEqual(values[2], np.inf)

    def test_a_whole_bad_batch_is_still_fatal(self):
        with self.assertRaises(RuntimeError) as ctx:
            nonfinite_policy([np.nan, np.nan, np.nan], allow=True, batch=True)
        self.assertIn("every candidate in a compass sweep", str(ctx.exception))

    def test_a_partly_bad_batch_is_recoverable(self):
        values, n_bad = nonfinite_policy([np.nan, 2.0], allow=True, batch=True)
        self.assertEqual(n_bad, 1)
        np.testing.assert_allclose(values, [np.inf, 2.0])


class TestSearchOverADivergingCandidate(unittest.TestCase):
    def test_candidate_that_diverges_is_rejected_not_fatal(self):
        prob = _problem(allow=True)
        with self.assertWarns(RuntimeWarning):
            res = pattern_search(prob, np.array([0.6, 0.6]), step0=0.5,
                                 tol=1e-6, max_iter=3, max_evals=200)
        # it kept the finite improves (lowering each well) and never accepted
        # the NaN branch
        self.assertTrue(np.isfinite(res.fun))
        self.assertLess(res.fun, 1.2)
        self.assertTrue(np.all(res.q <= BAD_ABOVE))
        self.assertGreater(prob.n_nonfinite, 0)
        self.assertEqual(res.metrics["nonfinite_candidates"], prob.n_nonfinite)

    def test_default_still_raises_mid_search(self):
        prob = _problem(allow=False)
        with self.assertRaises(RuntimeError):
            pattern_search(prob, np.array([0.6, 0.6]), step0=0.5,
                           tol=1e-6, max_iter=3, max_evals=200)

    def test_all_starts_non_finite_is_fatal_even_when_allowed(self):
        prob = _problem(allow=True, bad_above=-1.0)   # everything diverges
        with self.assertRaises(RuntimeError) as ctx:
            solve_saa(prob, method="pattern", n_starts=2, max_iter=1,
                      max_evals=4, seed=0, batched=False, init_fraction=0.5)
        self.assertIn("every start returned a non-finite objective",
                      str(ctx.exception))


class _BatchEval:
    """Batched evaluation container: per-candidate ``mean_J`` plus (S, M) tails."""

    def __init__(self, mean_J, G, H) -> None:
        self.mean_J, self.G, self.H = mean_J, G, H


class _BatchedDivergingArchive(_DivergingArchive):
    """The same archive, batched: ``evaluate_many`` returns one NaN candidate."""

    supports_batching = True

    def __init__(self, bad_above: float = BAD_ABOVE) -> None:
        super().__init__(bad_above)
        self.device = torch.device("cpu")

    def evaluate_many(self, Q, cost=None):
        Q = np.asarray(Q, dtype=float)
        bad = np.any(Q > self.bad_above, axis=1)
        f = torch.tensor(Q.sum(axis=1))
        f[torch.tensor(bad)] = float("nan")
        S, M = Q.shape[0], 4
        return _BatchEval(f, torch.full((S, M), -1.0, dtype=torch.float64),
                          torch.full((S, M), 1.0, dtype=torch.float64))


@unittest.skipIf(torch is None, "torch is not installed")
class TestBatchedSweepWithADivergingCandidate(unittest.TestCase):
    """The real failure was in the batched compass sweep, not the sequential one."""

    def test_one_nan_candidate_does_not_end_the_sweep(self):
        prob = SAAProblem(_BatchedDivergingArchive(), cost=None,
                          allow_nonfinite=True)
        with self.assertWarns(RuntimeWarning):
            res = pattern_search_batched(prob, np.array([0.6, 0.6]), step0=0.5,
                                        tol=1e-6, max_iter=2, max_evals=100)
        self.assertTrue(np.isfinite(res.fun))
        self.assertTrue(np.all(res.q <= BAD_ABOVE))
        self.assertGreater(prob.n_nonfinite, 0)
        self.assertEqual(res.metrics["nonfinite_candidates"], prob.n_nonfinite)

    def test_a_sweep_of_nans_is_still_fatal(self):
        prob = SAAProblem(_BatchedDivergingArchive(bad_above=-1.0), cost=None,
                          allow_nonfinite=True)
        with self.assertRaises(RuntimeError) as ctx:
            pattern_search_batched(prob, np.array([0.6, 0.6]), step0=0.5,
                                   tol=1e-6, max_iter=2, max_evals=100)
        self.assertIn("every candidate in a compass sweep", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()