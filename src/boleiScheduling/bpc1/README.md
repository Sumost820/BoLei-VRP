# BPC V0.20 — 四状态 Pricing + 稀疏 Node-local RMP

这一版保持完整物理路线列：客户节点 elementary，唯一物理换电站 `S[0]` 可重复访问；不同 BSS 插入位置仍然是不同列。

## 主算法

```text
Branch node
    ↓
solve RMP
    ↓
exact pricing
    ├─ 有负 reduced-cost 列 → 加列 → 重新 solve RMP
    ↓ 无负列
LP 是否整数？
    ├─ 否 → physical-arc branch；必要时 customer-precedence fallback
    ↓ 是
Exact BSS scheduling
    ↓
更新 incumbent / UB
    ↓
Master 的 T 是否低估该整数路线组合？
    ├─ 是 → 加 BSS combinatorial cut → 回同一节点 RMP + pricing
    └─ 否 → fathom 当前节点
```

BSS scheduling 只在**列生成已经收敛的整数 routing solution**上调用。

## Pricing 的四个数学状态

`_PricingLabel` 严格只有：

```python
currentNode
remainingEnergy
partialReducedCost
unreachableMask
```

其中 `unreachableMask` 是永久不可达的客户集合；当前 elementary 模型中就是已访问客户集合。BSS 不进入这个集合，因为它可以重复访问。

没有额外 branch/cut 状态时，同一当前节点上使用：

```text
U_A ⊆ U_B
E_A ≥ E_B
RC_A ≤ RC_B
```

则 A 支配 B。

注意：由于“当前电量不足”而暂时去不了的客户**不能**加入 `unreachableMask`，因为经过 BSS 补能后它可能重新可达。

### 为什么不需要 duration

时间只通过 master makespan dual `beta_k` 进入 reduced cost：

```text
RC += beta_k * travel/service/swap_time
```

当前模型没有 time window 一类需要单独保留绝对时间的可行性资源，因此 duration 不需要作为 label 字段。

### required physical arc / precedence

不保存 `requiredArcMask`。required arc 通过局部 required successor/predecessor 约束以及 mandatory-customer completion check 实现。required precedence 的完成进度也直接由 `unreachableMask` 推导。

这些推导状态可以进入 dominance bucket key，但不是 `_PricingLabel` 的新字段。

### SRC

对 divisor-2 SRC，未来边际 cut coefficient 只取决于：

```python
(unreachableMask & cutMask).bit_count() % 2
```

所以不保存 `srcParityMask`。当某个 SRC dual 非零时，dominance key 只使用从 `unreachableMask` 即时推导的 parity signature。

### BSS combinatorial cut 是唯一的历史型例外

BSS cut 的列系数可能取决于完整 physical route signature，仅凭四个数学状态不能区分。因此只有当 BSS cut dual 非零时，`_LabelRecord` 才临时携带一个小型 `protectedPrefixState`。

它只是 cut-coefficient automaton bookkeeping，不属于 RCSP 数学资源；一旦 prefix 已不可能命中任何活跃 BSS-cut signature，就进入统一 DEAD 状态并恢复普通强支配。

## 数据结构提速

### Global ColumnManager

列池是 append-only，route index 在整个 BPC 树中稳定。维护 Python-int bitset 倒排索引：

```text
arcToRouteMask[(u,v)]
precedenceToRouteMask[(i,j)]
taskToRouteMask[i]
```

branch node 的兼容列通过整数 AND / NOT 得到，不再逐列扫描。

### Node-local sparse RMP

每个 branch node 只为 compatible `(slot, route)` 创建 `x[k,r]`。不再对全部 global columns 建变量后把不兼容列 `UB=0`。

全局新列产生后，当前 node 的 persistent RMP 只同步新增 global index 区间中的兼容列。

### Cut 倒排索引

BSS cut 用 `route signature -> cut keys`；SRC 用 customer incidence bitset。新增 cut/column 时避免反复扫描全部 global columns。

### Branching support

同一个 LP 解只调用一次 `getPositiveX()`，arc flow、precedence flow、integrality 共用该 support。

## 小规模 exact BSS scheduler

固定完整路线后，每条车辆的 swap events 顺序已经确定。若所有路线 event chains 的合法 interleaving 数不超过阈值，使用 exact DFS enumeration；固定一个 interleaving 后所有事件 earliest-start 即为该次序的最优调度。规模过大时自动回退原 Gurobi MIP。

## 入口

```python
from boleiScheduling.bpc1 import BranchPriceCutSolver

solver = BranchPriceCutSolver(data)
solver.solve(outputFlag=1)
result = solver.getResult()
```

默认 `useSRC=False`，即核心流程就是 RMP → Pricing → Branch / Integer BSS Separation。
