# Route Pool Heuristic

该算法用于当前无时间窗、无容量约束、单物理换电站、固定换电时间、目标最小化 makespan 的问题。

整体框架参考 Froger et al. 的 route generator + route pool + solution assembler 思路，但针对固定满电换电进行简化：

1. `RouteGenerator` 只搜索任务序列，采用 ILS + VND。
2. `FixedRouteSwapOptimizer` 对固定任务序列精确决定换电插入位置，并保留多个较优换电 variant。
3. `RoutePool` 保存 short-term / long-term 路线候选。
4. `RouteAssembler` 从有限路线池中选择路线，满足每个任务恰好覆盖一次且车辆数不超过 K。
5. `BssScheduler` 对已选路线的所有换电事件安排单换电工位顺序，允许等待并精确最小化 makespan。
6. 若路线选择模型低估等待时间，`RouteAssembler` 加入组合 optimality cut 后重新求解。
7. Assembler 选中的路线反馈进入长期路线池，并作为下一轮 ILS 的起点。

当前模型没有硬时间窗，因此 BSS scheduling 不生成 feasibility cut；换电冲突只产生等待和 makespan 增量。
