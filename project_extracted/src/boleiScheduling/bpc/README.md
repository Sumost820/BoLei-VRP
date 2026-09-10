# BPC V0.4 — exact arc-flow branching

This version extends V0.3 with an explicit Branch-and-Price tree.

## Column definition

A route column is identified by its ordered **customer sequence**. Swap placement
is not part of the column identity. For a fixed sequence, the isolated route
cost `d_r` is the exact minimum base duration over all feasible swap placements.
The shared-BSS waiting component is handled only by the BSS subproblem/cuts.

## Master

For vehicle slot `k` and route column `r`:

- `x[k,r]` route usage,
- `T` makespan lower bound.

The rows are

`sum_{k,r} a_ir x[k,r] = 1` for every customer,

`sum_r x[k,r] = 1` for every vehicle slot,

`T >= sum_r d_r x[k,r]` for every vehicle slot.

BSS cuts are

`T >= T* (sum_{r in R*} sum_k x[k,r] - |R*| + 1)`.

## Arc-flow branching

Because swap placement is optimized outside the column identity, the branching
arc is the directed **successor arc in the compressed customer sequence**, not a
physical customer-to-BSS arc.

For route `(i1,...,im)` its branching arcs are

`(start,i1), (i1,i2), ..., (im,end)`.

For every slot `k` define

`f[k,i,j] = sum_r a[r,i,j] x[k,r]`.

Choose a fractional `f[k,i,j]`, preferably closest to 0.5, and create:

- left child: `f[k,i,j] = 0`,
- right child: `f[k,i,j] = 1`.

The branch is propagated directly to the column universe:

- value 0: slot-k pricing forbids `(i,j)`,
- value 1: every slot-k column must contain `(i,j)`.

Because `sum_r x[k,r]=1`, this is exactly equivalent to adding the corresponding
branch equation. Therefore no extra branch-row dual is required.

## Branch-aware exact labeling

The pricing label state additionally records which required branch arcs have
already been used. Dominance is performed only for labels with the same:

- physical current node,
- last customer in the compressed sequence,
- visited-customer mask,
- required-arc mask.

This prevents branch requirements from being lost by dominance at the BSS.

## Integral node and BSS cut

If no slot-specific successor arc is fractional, the slot-column solution is
integral. The selected customer routes are sent to the exact BSS subproblem.
All feasible swap placements for each selected route are enumerated; Gurobi
selects placements and sequences all swap events on the single physical BSS.

If the routing master has `T < T*`, add the global combinatorial BSS cut and
re-price the same branch node. Otherwise the node is fathomed.

## Exactness statement

With:

- exact elementary labeling at every node,
- no node/time/cut limit,
- no `maxExactRouteTasks` truncation,
- every BSS subproblem solved to proven optimality,

exhaustion of the branch queue certifies global optimality for the represented
routing/swap model.

## Example

```python
from boleiScheduling.bpc import BranchPriceCutSolver
from boleiScheduling.mockData import createMockData

data = createMockData(
    taskCount=8,
    stationCopyCount=8,
    K=2,
    seed=4,
    Q=50,
    QMin=20,
)

solver = BranchPriceCutSolver(
    data,
    pricingMode='labeling',
    maxColumnsPerRound=50,
    maxExactRouteTasks=None,  # do not truncate exact BSS plan enumeration
    bssTimeLimit=None,
)
solver.solve(outputFlag=1, nodeLimit=None, timeLimit=None)
print(solver.getResult())
```

`nodeLimit` or `timeLimit` can be used for experiments, but then a non-OPTIMAL
status is not an optimality proof.
