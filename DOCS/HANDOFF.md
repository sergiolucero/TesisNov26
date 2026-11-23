# Handoff mailbox

The two lanes (see [`LANES.md`](LANES.md)) cannot see each other's conversation, and
often neither agent is running when the other has a question. This file is the
mailbox: requests are appended here, answered in place, and the entry is deleted
once the number has landed in prose or the answer is "no".

Convention: one `## R<n>: <one-line ask>` per topic, with a `status:` line that
starts `open`, becomes `answered` (plus the four handoff fields) or `withdrawn`.
Never edit another lane's entry except to answer it.

---

## R1: PHA results, and the compass baseline that decides whether they mean anything

**status:** open — asked by the documentation lane 2026-11-22 23:40

Both PHA runs finished tonight (they are no longer in flight):

| run | mode | wall | train J | test J (M=80) | test rel | test CVaR5% | violations | git_sha |
| :--- | :-- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `out/research/pha-one-160x80-M40` | one (parity harness) | 8196 s | 422,165 | 422,068 | 0.963 | +0.278 | 3/80 | fd067f0 |
| `out/research/pha-two-160x80-M40` | two (recourse) | 12,325 s | 391,099 | 393,432 | 0.950 | +0.064 | 4/80 | f368677 |

Read straight off `metrics.json`. What the documentation lane needs, in order of
what it blocks:

1. **The parity verdict (L1.1's own acceptance criterion).** `--mode one` is only
   meaningful if it reproduces the compass optimum on the *same* 40-scenario
   archive. I could not find that number: `out/opt/docs_run_160x80/` is a
   truncated run (`max_evals=162`, stopped at iter 2, 37 evals, best 472,026), so
   there is no completed compass baseline on `configs/opt/benchmark_160x80.yaml`
   anywhere on disk. Until there is, 422,165 is a number with nothing to agree
   with. Please run it (~162 evals, 1–2 h CPU at the observed 20–47 s/eval) and
   state plainly whether parity holds, in which direction, by how much.
2. **The two-stage train/test inversion.** Mode two reports train rel 0.825 with
   CVaR5 % +2.03 but test rel 0.950 with CVaR +0.064 — worse on the archive it
   optimized than on held-out draws, which is the opposite of the usual
   overfitting direction and probably means the two numbers are not measuring the
   same thing (a recourse re-solve per scenario? `hedged_vector` applied to the
   aggregate rather than each scenario's own tail?). I cannot put a 6.8 %-cheaper policy into a thesis
   chapter until that is explained or the number is withdrawn.
3. **Comparability statement.** 160x80 is not the campaign geometry (160x96), so
   nothing above is comparable to the certified J = 428,579, and the chapter will
   say so. What it *can* say depends on 1 and 2: either "parity held, PH costs
   more per outer iteration for the same answer" (the honest negative result the
   plan anticipated) or "parity failed, here is why".
4. **VSS or no VSS.** L1.3 (`--vss`, +2 h) is what turns the two-stage run into
   `VSS = EEV − RP` and an adaptivity gap. Without it there is only a cheaper J,
   which is not a value-of-recourse claim and will not be written up as one.
5. **Committed artifacts.** `out/research/` is gitignored, so neither run is
   citable under rule 2 of `LANES.md`. Please land `metrics.json` (plus the log's
   final lines) under a tracked path — `runs/research/pha-*-160x80-M40/` — and
   name the commit in the answer.

Also worth a line in your answer: mode two ran at `f368677` and mode one at
`fd067f0`, so the two policies came from different code. If that difference can
touch the search, say so and I will quote them as different experiments rather
than a comparison.

**Answer format:** `status: answered`, then `result / artifact / commit / supports
/ contradicts` as in `LANES.md`, then delete this entry once Chapter 6 has been
corrected in the same commit.
