# BPC V0.15 — pricing performance refinement

This version keeps the same exact slot-specific arc-flow Branch-and-Price-and-Cut
model and focuses on implementation overhead.

## Main performance changes

1. **Persistent RMP.** A node/phase LP is built once. New route columns are
   appended through Gurobi `Column` objects; SRC/BSS rows are appended to the
   same model. Re-optimization therefore reuses the previous LP basis instead
   of rebuilding the full master every CG round.

2. **Phase-I on demand.** A branch node first solves the actual Phase-II
   restricted master. Phase I is created only if that restricted master is
   infeasible and new branch-compatible columns may be needed. Adding a BSS or
   SRC cut never reruns Phase I.

3. **Batch pricing early stop.** Labeling stops as soon as the requested batch
   of negative reduced-cost customer sequences has been found. Exact pricing is
   fully exhausted only when no improving column is returned and optimality
   must be certified.

4. **Mask-indexed exact dominance.** The rigorous
   `visited(A) subseteq visited(B)` rule is unchanged, but labels are indexed by
   exact visited bit masks and Pareto fronts. The implementation avoids scanning
   every label in every cardinality bucket.

5. **Predecessor labels.** Partial customer tuples are no longer copied on every
   extension. A complete customer sequence is reconstructed only for a negative
   complete route.

6. **Global caches.** Exact fixed-sequence evaluations and complete feasible
   swap-plan families are cached. Proven-optimal BSS results are cached by a
   permutation-invariant selected-route key. Non-optimal/time-limit BSS results
   are never cached as exact values.

7. **Small-instance short-route warm pool.** By default, when `N<=15`, all
   feasible ordered routes of length at most 2 are pre-generated. This is only
   an initial column pool and does not restrict pricing.

8. **Vehicle-symmetry node deduplication.** With homogeneous vehicles, branch
   states that differ only by a permutation of vehicle slots are recognized as
   the same subproblem and only one is kept. This can be disabled with
   `deduplicateSymmetricNodes=False`.

9. **SRC default OFF.** SRC remains fully implemented and exact, but it is now
   opt-in because for the small cases motivating this refactor the extra
   cut-price rounds and parity state can cost more than the node reduction.

10. **Integrated profiling.** Final output reports master build/LP time,
    pricing time and label counts, exact route-evaluation cache statistics, BSS
    time/cache statistics, Phase-I activation/skips and symmetric nodes skipped.

11. **FIFO label queue restored.** V0.14's binary heap is removed. Label
    expansion again uses `collections.deque` O(1) push/pop; safe reduced-cost
    pruning is applied before a new label enters the dominance structure.

12. **Staged vehicle-slot pricing.** Slots are tried in a promising deterministic
    order. As soon as one slot yields improving columns in a batched CG round,
    the solver returns them and immediately re-optimizes the persistent RMP
    instead of pricing every remaining slot under stale duals.

13. **Exact vehicle-slot dominance.** An exhaustively priced slot can certify a
    second slot when its branch-feasible route set contains the second slot's
    set and `beta_a <= beta_b`, `sigma_a >= sigma_b`. This can remove redundant
    ESPPRC calls without changing pricing exactness.

## Exactness

No heuristic replaces exact pricing. Savings and short-route generation only
supply initial columns/incumbents. Mask-indexed dominance uses the same rigorous
conditions as before. The BSS cache stores only results with
`provenOptimal=True`. Therefore, without external node/time limits and with all
required Gurobi subproblems solved to proven optimality, the algorithm retains
the same exactness conditions as the previous BPC version.

## Recommended small-instance configuration

```python
from boleiScheduling.bpc import BranchPriceCutSolver

solver = BranchPriceCutSolver(
    data,
    pricingMode='labeling',
    dominanceMode='subset',
    maxColumnsPerRound=100,

    useSavingsWarmStart=True,
    useSavingsIncumbent=True,

    # Fast initial pool for small cases.
    preGenerateShortRoutes=2,
    shortRouteTaskThreshold=15,
    maxShortRouteColumns=1000,

    # Usually faster for N≈10–15; SRC can be enabled explicitly later.
    useSRC=False,

    deduplicateSymmetricNodes=True,
    maxExactRouteTasks=None,
    bssTimeLimit=None,
)
solver.solve(outputFlag=2, nodeLimit=None, timeLimit=None)
```

Set `outputFlag=1` for concise progress, `2` for detailed CG/profile/BSS output,
and `3` to additionally show native Gurobi logs.
