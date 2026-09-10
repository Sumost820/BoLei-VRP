# BPC V0.5 — Savings warm start + strong exact dominance

This version keeps the V0.4 exact arc-flow Branch-and-Price-and-Cut structure,
but improves the two main bottlenecks before changing the mathematical model.

## 1. Savings warm start

Before root column generation, a multi-start directed Clarke-Wright-style
construction generates customer-only route sequences. Every proposed sequence
is checked by the exact fixed-sequence swap DP, so the route is energy-feasible
and receives its exact isolated minimum base duration.

The warm-start merge score is

`excessRoutePenalty * max(0, numberRoutes-K) + baseMakespan + 1e-6*sumDuration`.

The huge excess-route penalty is used **only inside this heuristic warm start**.
It never enters the restricted master, pricing reduced costs, node bounds, or
optimality proof.

The generated singleton/intermediate/final savings routes are inserted into the
initial RMP column set. If a savings start reaches at most K routes, its isolated
minimum swap placement is also scheduled by a fast feasible FCFS single-BSS
scheduler to provide an immediate incumbent upper bound. This greedy BSS value
is never used to create an optimality cut.

Because a K-route warm-start solution already makes all Phase-I artificial
variables zero, Phase I is then globally optimal at value zero. The code safely
skips Phase-I pricing in that case.

## 2. Strong exact set-inclusion dominance

For slot k, define the prefix reduced cost

`g(L) = beta_k * duration(L) - sum_{i in visited(L)} pi_i`.

For labels A and B with the same physical current node, same last customer in
the compressed customer sequence, and same satisfied-required-arc mask, A
dominates B if

- `visited(A) subseteq visited(B)`,
- `remainingEnergy(A) >= remainingEnergy(B)`,
- `g(A) <= g(B)`.

This dominance is exact. Every elementary suffix feasible from B avoids the
larger visited(B), so it is also available to A. A starts that suffix from the
same physical node with no less energy, and both labels receive the same suffix
reduced-cost increment.

The old equal-visited-set rule remains available through
`dominanceMode='equal'` for validation. The default is now
`dominanceMode='subset'`.

The dominance bucket key no longer contains the visited mask. Buckets are
subdivided by `visitedMask.bit_count()` so only cardinalities that can possibly
be subset/superset comparable are scanned.

## 3. Early branch propagation

Required successor-arc branches are enforced before a new customer label is
inserted:

- if the previous customer has a required successor, no other successor is
  allowed;
- if the candidate customer has a required predecessor, no other predecessor is
  allowed;
- the same logic is applied to the final customer -> depot arc.

This is exact and avoids creating labels that are already branch-infeasible.

## Column / master / cuts / branching

The V0.4 definitions are unchanged:

- a column is an ordered customer sequence;
- its `d_r` is the exact minimum isolated base duration over swap placements;
- the master minimizes `T` with coverage, vehicle-slot, and slot makespan rows;
- pricing uses `beta_k*d_r - sum_i pi_i - sigma_k`;
- branch variable is the slot-specific compressed successor-arc flow;
- every integral route combination is sent to the exact BSS subproblem;
- a proven BSS optimum `T*` yields
  `T >= T*(sum_{r in R*} sum_k x[k,r] - |R*| + 1)`.

## Recommended exact run

```python
from boleiScheduling.bpc import BranchPriceCutSolver

solver = BranchPriceCutSolver(
    data,
    pricingMode='labeling',
    dominanceMode='subset',
    useSavingsWarmStart=True,
    savingsStarts=12,
    savingsSeed=1,
    savingsRandomization=0.20,
    savingsExcessRoutePenalty=1e9,
    maxWarmStartColumns=1000,
    useSavingsIncumbent=True,
    maxColumnsPerRound=50,
    maxExactRouteTasks=None,
    bssTimeLimit=None,
)
solver.solve(outputFlag=2, nodeLimit=None, timeLimit=None)
```

For an exact optimality proof, do not impose a node/cut/route-plan truncation and
require every BSS subproblem used for separation to be solved to proven
optimality.


## Tree search: best-bound

The Branch-and-Price-and-Cut tree uses a min-heap and always processes the open
node with the smallest inherited valid LP lower bound. The node id is used only
as a deterministic tie-break. This restores the pre-V0.6 search order.

Best-bound does not change the branching disjunction, pricing problem, SRCs, BSS
optimality cuts, or exactness. Its main practical advantage here is that the
global lower bound and reported optimality gap tend to evolve more steadily than
under DFS.

## V0.7: 3-row subset-row cuts (SRC)

V0.7 adds the classical rank-1 subset-row cut family to reduce the branch tree.
For every customer triple `S={i,j,h}` the globally valid inequality is

```
sum_{k,r} floor(|S intersect r| / 2) x[k,r] <= 1.
```

For a triple, a route coefficient is therefore one iff the route contains at
least two members of the triple.  SRC separation is performed only after the
current node has been solved by exact column generation.  Whenever an SRC is
added, the node is solved and priced again because the RMP dual solution has
changed.

The pricing reduced cost becomes

```
rc(k,r) = beta_k d_r - sum_{i in r} pi_i - sigma_k
          - sum_S eta_S floor(|S intersect r|/2),
```

where `eta_S <= 0` is the Gurobi dual of an SRC `<=` row.  Equivalently,
`rho_S=-eta_S>=0` is an additive pricing penalty.

The labeling algorithm handles this term exactly.  For divisor-2 SRCs, each
active nonzero-dual cut needs only one parity bit.  Visiting a member of S
changes the count parity; when the old parity is odd, `floor(count/2)` increases
by one and `rho_S` is added to the partial reduced cost.  Dominance requires the
same SRC parity vector in addition to the existing physical/branch state.  This
keeps the subset-visited-set dominance rigorous under SRC duals.

Default BPC V0.7 SRC parameters:

```python
solver = BranchPriceCutSolver(
    data,
    useSRC=True,
    srcViolationTolerance=1e-7,
    maxSrcCutsPerRound=10,
    maxSrcRoundsPerNode=5,
    maxTotalSrcCuts=100,
    srcMaxDepth=2,
)
```

These parameters control cut separation effort only.  SRCs are strengthening
cuts and are not required for correctness, so limiting the number of rounds,
total cuts, or separation depth does not compromise the exactness of the BPC
algorithm; it only changes performance.
