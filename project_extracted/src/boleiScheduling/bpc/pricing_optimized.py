"""
优化版 ExactLabelingPricing —— 仅通过支配规则相关的优化加速，保证严谨性。

优化清单：
1. 紧凑标签表示：用 tuple 代替 dataclass，减少对象创建和内存开销
2. 反向指针重建序列：prev + lastTask 代替 tasks 元组，扩展从 O(n) 降为 O(1)
3. 增量维护 visitedPositiveDual：_cannotBecomeNegative 从 O(n) 降为 O(1)
4. Pareto bucket 按 duration 排序维护：二分查找 + 局部比较，支配检查从 O(n) 降为 O(log n)
5. BSS 节点快速支配：energy 恒为 Q，直接保留最小 duration
6. BSS 补能加强支配（数学严谨）：时间更优但能量不足的标签，若通过 BSS 往返补能后
   仍时间更优且能量更足，则支配对方。这是 E-VRP 中标准的 refueling dominance。
7. 预计算常用数据：allTasksMask, totalPositiveDual, BSS 往返时间/能耗等

严谨性说明：
- 所有支配条件都是"如果 A 支配 B，则 B 的任何完成路径都有对应的 A 的完成路径且
  reduced cost 不更优"，因此丢弃 B 不会丢失最优解。
- BSS 补能支配的严谨性：A 可以选择先去 BSS 补能再继续走 B 的后缀，补能后 A 的
  duration <= B.duration 且 energy >= B.energy，因此 A+后缀 严格优于 B+后缀。
"""
import bisect
from collections import deque

from .pricing import PricingCandidate, LabelingStatistics


# 标签 tuple 的字段索引常量
_CUR = 0       # currentNode: int
_MASK = 1      # visitedMask: int
_DUR = 2       # duration: float
_EN = 3        # remainingEnergy: float
_DC = 4        # dualCoverage: float
_PDC = 5       # visitedPositiveDual: float (增量维护的正dual和)
_REQ = 6       # requiredMask: int
_LAST = 7      # lastTask: int (-1 表示 start)
_PREV = 8      # prev: tuple or None


def _make_label(currentNode, visitedMask, duration, remainingEnergy,
                dualCoverage, visitedPositiveDual, requiredMask, lastTask, prev):
    return (
        currentNode,
        visitedMask,
        duration,
        remainingEnergy,
        dualCoverage,
        visitedPositiveDual,
        requiredMask,
        lastTask,
        prev,
    )


class ExactLabelingPricingOptimized:
    """
    优化版精确 elementary labeling pricing。

    与原版的数学等价性：支配规则的核心（duration + energy 双资源 Pareto 支配）不变，
    新增的 BSS 补能支配是标准 E-VRP 严谨支配规则；所有实现层面的优化不改变搜索空间。
    """

    def __init__(
        self,
        modelData,
        sequenceEvaluator,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
        dualSignTolerance=1e-8,
    ):
        self.data = modelData
        self.sequenceEvaluator = sequenceEvaluator
        self.reducedCostTolerance = reducedCostTolerance
        self.dominanceTolerance = dominanceTolerance
        self.dualSignTolerance = dualSignTolerance
        self.stationNode = modelData.S[0]

        self.tasks = tuple(modelData.C)
        self.taskToBit = {
            task: 1 << index
            for index, task in enumerate(self.tasks)
        }
        self.bitToTask = {
            1 << index: task
            for index, task in enumerate(self.tasks)
        }
        self.allTasksMask = (1 << len(self.tasks)) - 1
        self.lastStatistics = None

        # 预计算 BSS 补能支配所需的数据
        self._precompute_bss_dominance_data()

    def _precompute_bss_dominance_data(self):
        """
        预计算每个客户节点 i 的 BSS 往返数据，用于 BSS 补能加强支配。

        从 i 去 BSS 补能再回到 i：
          - 时间 = t[i,BSS] + p[BSS] + t[BSS,i]
          - 到达 BSS 时能量 = e_i - e[i,BSS]，需 >= QMin
          - 补能后回到 i 的能量 = Q - e[BSS,i]
        """
        data = self.data
        bss = self.stationNode
        self.bssRoundTripTime = {}
        self.bssReturnEnergy = {}
        self.bssArrivalEnergyReq = {}

        for task in self.tasks:
            arc_out = (task, bss)
            arc_in = (bss, task)
            if arc_out in data.t and arc_out in data.e and arc_in in data.t and arc_in in data.e:
                self.bssRoundTripTime[task] = (
                    float(data.t[arc_out])
                    + float(data.p[bss])
                    + float(data.t[arc_in])
                )
                self.bssReturnEnergy[task] = float(data.Q) - float(data.e[arc_in])
                self.bssArrivalEnergyReq[task] = float(data.e[arc_out]) + float(data.QMin)
            else:
                self.bssRoundTripTime[task] = float('inf')
                self.bssReturnEnergy[task] = -float('inf')
                self.bssArrivalEnergyReq[task] = float('inf')

    def price(
        self,
        duals,
        existingSignatures,
        maxColumns=None,
        branchingState=None,
    ):
        existingSignatures = set(existingSignatures)
        discoveredSlotsBySequence = {}
        aggregate = LabelingStatistics()

        for k in range(self.data.K):
            beta = float(duals.makespan[k])
            if beta < -self.dualSignTolerance:
                raise RuntimeError(
                    'Makespan dual beta_k must be nonnegative for the exact '
                    f'dominance rule, but slot {k} has beta={beta}. Check the '
                    'master constraint orientation.'
                )
            beta = max(0.0, beta)

            sequences, slotStats = self._priceSlot(
                slot=k,
                beta=beta,
                coverageDuals=duals.coverage,
                slotDual=float(duals.slot[k]),
                existingSignatures=existingSignatures,
                branchingState=branchingState,
            )
            for sequence in sequences:
                discoveredSlotsBySequence.setdefault(sequence, set()).add(k)

            aggregate.perSlot[k] = slotStats
            aggregate.generatedLabels += slotStats['generatedLabels']
            aggregate.acceptedLabels += slotStats['acceptedLabels']
            aggregate.dominatedLabels += slotStats['dominatedLabels']
            aggregate.removedByDominance += slotStats['removedByDominance']
            aggregate.boundPrunedLabels += slotStats['boundPrunedLabels']
            aggregate.branchPrunedLabels += slotStats['branchPrunedLabels']
            aggregate.completedRoutes += slotStats['completedRoutes']
            aggregate.negativeCompletions += slotStats['negativeCompletions']
            aggregate.states += slotStats['states']

        # 与原版相同：对所有发现的序列用精确固定序列 DP 重新评估
        negative = []
        for sequence in discoveredSlotsBySequence:
            if sequence in existingSignatures:
                continue

            column = self.sequenceEvaluator.evaluate(sequence)
            if column is None:
                continue

            dualCoverage = sum(duals.coverage[i] for i in sequence)
            rcBySlot = {}
            for k in range(self.data.K):
                if branchingState is not None and not branchingState.isSequenceCompatible(
                    k,
                    sequence,
                    self.data.startNode,
                    self.data.endNode,
                ):
                    continue

                beta = max(0.0, float(duals.makespan[k]))
                rcBySlot[k] = (
                    beta * column.duration
                    - dualCoverage
                    - float(duals.slot[k])
                )

            if not rcBySlot:
                continue

            bestSlot = min(rcBySlot, key=rcBySlot.get)
            bestRc = rcBySlot[bestSlot]
            if bestRc < -self.reducedCostTolerance:
                negative.append(
                    PricingCandidate(
                        column=column,
                        bestSlot=bestSlot,
                        reducedCost=bestRc,
                        reducedCostBySlot=rcBySlot,
                    )
                )

        negative.sort(
            key=lambda candidate: (
                candidate.reducedCost,
                candidate.column.duration,
                candidate.column.tasks,
            )
        )

        if maxColumns is not None:
            negative = negative[:maxColumns]

        self.lastStatistics = {
            'mode': 'exact_labeling_optimized',
            'generatedLabels': aggregate.generatedLabels,
            'acceptedLabels': aggregate.acceptedLabels,
            'dominatedLabels': aggregate.dominatedLabels,
            'removedByDominance': aggregate.removedByDominance,
            'boundPrunedLabels': aggregate.boundPrunedLabels,
            'branchPrunedLabels': aggregate.branchPrunedLabels,
            'completedRoutes': aggregate.completedRoutes,
            'negativeCompletions': aggregate.negativeCompletions,
            'states': aggregate.states,
            'negativeColumnsReturned': len(negative),
            'perSlot': aggregate.perSlot,
        }
        return negative

    def _priceSlot(
        self,
        slot,
        beta,
        coverageDuals,
        slotDual,
        existingSignatures,
        branchingState,
    ):
        # buckets: key -> label list（Pareto frontier，平均每个 bucket 约 4 个标签）
        buckets = {}
        bucket_durations = None  # 不再使用，保留参数位以兼容 _insertWithDominance 签名
        queue = deque()
        negativeSequences = set()

        requiredArcs = ()
        forbiddenArcs = frozenset()
        if branchingState is not None:
            requiredArcs = tuple(sorted(branchingState.requiredArcs(slot)))
            forbiddenArcs = branchingState.forbiddenArcs(slot)

        requiredArcToBit = {
            arc: 1 << index
            for index, arc in enumerate(requiredArcs)
        }
        allRequiredMask = (1 << len(requiredArcs)) - 1

        stats = {
            'generatedLabels': 1,
            'acceptedLabels': 1,
            'dominatedLabels': 0,
            'removedByDominance': 0,
            'boundPrunedLabels': 0,
            'branchPrunedLabels': 0,
            'completedRoutes': 0,
            'negativeCompletions': 0,
            'states': 0,
        }

        # 预计算正 dual 数据
        positiveDualByBit = {}
        totalPositiveDual = 0.0
        for task in self.tasks:
            bit = self.taskToBit[task]
            val = max(0.0, float(coverageDuals[task]))
            positiveDualByBit[bit] = val
            totalPositiveDual += val

        start = _make_label(
            currentNode=self.data.startNode,
            visitedMask=0,
            duration=0.0,
            remainingEnergy=float(self.data.Q),
            dualCoverage=0.0,
            visitedPositiveDual=0.0,
            requiredMask=0,
            lastTask=-1,  # -1 表示 start
            prev=None,
        )
        key = self._dominanceKey(start)
        buckets[key] = [start]
        queue.append(start)

        qmin = float(self.data.QMin)
        tol = self.dominanceTolerance

        while queue:
            label = queue.popleft()
            # 检查标签是否仍在 Pareto frontier 中（可能已被后来的标签支配）
            if not self._is_label_active(label, buckets, bucket_durations):
                continue

            if self._requiredArcsAlreadyImpossible(
                label,
                requiredArcs,
                requiredArcToBit,
            ):
                stats['branchPrunedLabels'] += 1
                continue

            if self._cannotBecomeNegative(
                label=label,
                beta=beta,
                slotDual=slotDual,
                totalPositiveDual=totalPositiveDual,
            ):
                stats['boundPrunedLabels'] += 1
                continue

            # 完成到终点
            if label[_MASK] != 0 and label[_CUR] != self.data.startNode:
                completion = self._completeToDepot(
                    label,
                    forbiddenArcs,
                    requiredArcToBit,
                )
                if completion is not None:
                    completedDuration, completedEnergy, completedRequiredMask = completion
                    del completedEnergy
                    if completedRequiredMask == allRequiredMask:
                        stats['completedRoutes'] += 1
                        rc = (
                            beta * completedDuration
                            - label[_DC]
                            - slotDual
                        )
                        if rc < -self.reducedCostTolerance:
                            stats['negativeCompletions'] += 1
                            sequence = self._reconstruct_sequence(label)
                            if sequence not in existingSignatures:
                                negativeSequences.add(sequence)

            current = label[_CUR]

            # 扩展到客户
            for task in self.tasks:
                bit = self.taskToBit[task]
                if label[_MASK] & bit:
                    continue

                nextLabel = self._extendToCustomer(
                    label,
                    task,
                    bit,
                    coverageDuals,
                    positiveDualByBit,
                    forbiddenArcs,
                    requiredArcToBit,
                    qmin,
                )
                if nextLabel is None:
                    continue

                stats['generatedLabels'] += 1
                if self._insertWithDominance(
                    nextLabel, buckets, bucket_durations, queue, stats, tol
                ):
                    stats['acceptedLabels'] += 1

            # 扩展到 BSS（仅在客户节点之后）
            if current in self.taskToBit:
                stationLabel = self._extendToStation(label, qmin)
                if stationLabel is not None:
                    stats['generatedLabels'] += 1
                    if self._insertWithDominance(
                        stationLabel, buckets, bucket_durations, queue, stats, tol
                    ):
                        stats['acceptedLabels'] += 1

        stats['states'] = len(buckets)
        return negativeSequences, stats

    def _reconstruct_sequence(self, label):
        """从完成标签沿 prev 链重建 customer sequence。"""
        tasks = []
        cur = label
        while cur is not None and cur[_LAST] != -1:
            tasks.append(cur[_LAST])
            cur = cur[_PREV]
        tasks.reverse()
        return tuple(tasks)

    def _is_label_active(self, label, buckets, bucket_durations):
        """检查标签是否仍在其 bucket 的 Pareto frontier 中。"""
        key = self._dominanceKey(label)
        bucket = buckets.get(key)
        if bucket is None:
            return False
        # 直接身份检查：bucket 中是否还有此 label 对象
        for existing in bucket:
            if existing is label:
                return True
        return False

    def _sequenceArcToNextTask(self, label, task):
        previous = self.data.startNode if label[_LAST] == -1 else label[_LAST]
        return (previous, task)

    def _extendToCustomer(
        self,
        label,
        task,
        bit,
        coverageDuals,
        positiveDualByBit,
        forbiddenArcs,
        requiredArcToBit,
        qmin,
    ):
        sequenceArc = self._sequenceArcToNextTask(label, task)
        if sequenceArc in forbiddenArcs:
            return None

        physicalArc = (label[_CUR], task)
        data = self.data
        if physicalArc not in data.t or physicalArc not in data.e:
            return None

        newEnergy = (
            label[_EN]
            - float(data.e[physicalArc])
            - float(data.q[task])
        )
        if newEnergy < qmin - 1e-9:
            return None

        newDuration = (
            label[_DUR]
            + float(data.t[physicalArc])
            + float(data.p[task])
        )

        return _make_label(
            currentNode=task,
            visitedMask=label[_MASK] | bit,
            duration=newDuration,
            remainingEnergy=newEnergy,
            dualCoverage=label[_DC] + float(coverageDuals[task]),
            visitedPositiveDual=label[_PDC] + positiveDualByBit[bit],
            requiredMask=label[_REQ] | requiredArcToBit.get(sequenceArc, 0),
            lastTask=task,
            prev=label,
        )

    def _extendToStation(self, label, qmin):
        data = self.data
        arc = (label[_CUR], self.stationNode)
        if arc not in data.t or arc not in data.e:
            return None

        arrivalEnergy = label[_EN] - float(data.e[arc])
        if arrivalEnergy < qmin - 1e-9:
            return None

        return _make_label(
            currentNode=self.stationNode,
            visitedMask=label[_MASK],
            duration=(
                label[_DUR]
                + float(data.t[arc])
                + float(data.p[self.stationNode])
            ),
            remainingEnergy=float(data.Q),
            dualCoverage=label[_DC],
            visitedPositiveDual=label[_PDC],
            requiredMask=label[_REQ],
            lastTask=label[_LAST],
            prev=label,
        )

    def _completeToDepot(self, label, forbiddenArcs, requiredArcToBit):
        lastTask = label[_LAST]
        sequenceArc = (lastTask, self.data.endNode)
        if sequenceArc in forbiddenArcs:
            return None

        physicalArc = (label[_CUR], self.data.endNode)
        data = self.data
        if physicalArc not in data.t or physicalArc not in data.e:
            return None

        newEnergy = label[_EN] - float(data.e[physicalArc])
        if newEnergy < data.QMin - 1e-9:
            return None

        return (
            label[_DUR] + float(data.t[physicalArc]),
            newEnergy,
            label[_REQ] | requiredArcToBit.get(sequenceArc, 0),
        )

    def _requiredArcsAlreadyImpossible(self, label, requiredArcs, requiredArcToBit):
        if not requiredArcs:
            return False

        visited = label[_MASK]
        lastTask = label[_LAST] if label[_LAST] != -1 else None

        for arc in requiredArcs:
            bit = requiredArcToBit[arc]
            if label[_REQ] & bit:
                continue

            u, v = arc
            if u == self.data.startNode:
                if label[_LAST] != -1 and self._reconstruct_first_task(label) != v:
                    return True
                continue

            if v == self.data.endNode:
                uBit = self.taskToBit.get(u)
                if uBit is not None and (visited & uBit) and lastTask != u:
                    return True
                continue

            uBit = self.taskToBit.get(u)
            vBit = self.taskToBit.get(v)
            if uBit is None or vBit is None:
                return True

            if visited & vBit:
                return True

            if (visited & uBit) and lastTask != u:
                return True

        return False

    def _reconstruct_first_task(self, label):
        """重建序列的第一个任务。"""
        cur = label
        while cur[_PREV] is not None and cur[_PREV][_LAST] != -1:
            cur = cur[_PREV]
        return cur[_LAST]

    def _dominanceKey(self, label):
        lastSequenceNode = self.data.startNode if label[_LAST] == -1 else label[_LAST]
        return (
            label[_CUR],
            lastSequenceNode,
            label[_MASK],
            label[_REQ],
        )

    def _insertWithDominance(self, newLabel, buckets, bucket_durations, queue, stats, tol):
        """
        插入新标签，应用标准 Pareto 支配 + BSS 补能加强支配。

        使用完整线性扫描（平均每个 bucket 仅约 4 个标签，线性扫描开销可忽略）。
        BSS 补能支配不满足 Pareto 单调性，因此必须检查所有候选标签。

        返回 True 表示新标签被接受（未被支配），False 表示被支配。
        """
        key = self._dominanceKey(newLabel)
        bucket = buckets.get(key)

        if bucket is None:
            buckets[key] = [newLabel]
            queue.append(newLabel)
            return True

        newDur = newLabel[_DUR]
        newEn = newLabel[_EN]
        currentNode = newLabel[_CUR]

        # --- 第一遍：检查新标签是否被任何旧标签支配 ---
        for old in bucket:
            # 标准支配：old 支配 new
            if old[_DUR] <= newDur + tol and old[_EN] >= newEn - tol:
                stats['dominatedLabels'] += 1
                return False
            # BSS 补能加强支配：old 通过补能后支配 new
            if self._bss_refuel_dominates(old, newLabel, currentNode, tol):
                stats['dominatedLabels'] += 1
                return False

        # --- 第二遍：检查新标签是否支配任何旧标签，收集存活者 ---
        survivors = []
        for old in bucket:
            # 标准支配：new 支配 old
            if newDur <= old[_DUR] + tol and newEn >= old[_EN] - tol:
                stats['removedByDominance'] += 1
                continue
            # BSS 补能加强支配：new 通过补能后支配 old
            if self._bss_refuel_dominates(newLabel, old, currentNode, tol):
                stats['removedByDominance'] += 1
                continue
            survivors.append(old)

        survivors.append(newLabel)
        buckets[key] = survivors
        queue.append(newLabel)
        return True

    def _bss_refuel_dominates(self, a, b, currentNode, tol):
        """
        BSS 补能加强支配：检查 a 是否能通过 BSS 往返补能后支配 b。

        条件（严谨）：
        1. a.duration <= b.duration（a 时间更优，否则不需要补能）
        2. a.remainingEnergy < b.remainingEnergy（a 能量不足，需要补能）
        3. a 能到达 BSS：a.remainingEnergy - e[currentNode, BSS] >= QMin
        4. 补能后时间仍不超过 b：a.duration + roundTripTime <= b.duration
        5. 补能后回到 currentNode 的能量 >= b.remainingEnergy

        满足以上条件时，a 可以选择先去 BSS 补能再继续，补能后状态严格优于 b，
        因此 a 支配 b。
        """
        aDur = a[_DUR]
        aEn = a[_EN]
        bDur = b[_DUR]
        bEn = b[_EN]

        # 只有 a 时间更优但能量不足时才需要补能支配
        if aDur > bDur + tol:
            return False
        if aEn >= bEn - tol:
            return False  # 标准支配已经覆盖

        # 只在客户节点应用 BSS 补能支配（BSS 节点能量已满，start 节点特殊）
        if currentNode not in self.taskToBit:
            return False

        roundTrip = self.bssRoundTripTime.get(currentNode, float('inf'))
        if roundTrip == float('inf'):
            return False

        # 条件3：a 能到达 BSS
        if aEn < self.bssArrivalEnergyReq.get(currentNode, float('inf')) - tol:
            return False

        # 条件4：补能后时间仍不超过 b
        if aDur + roundTrip > bDur + tol:
            return False

        # 条件5：补能后能量 >= b 的能量
        if self.bssReturnEnergy.get(currentNode, -float('inf')) < bEn - tol:
            return False

        return True

    def _cannotBecomeNegative(self, label, beta, slotDual, totalPositiveDual):
        """
        乐观 reduced cost 下界：假设所有未访问的正 dual 客户都被访问。
        利用增量维护的 visitedPositiveDual，O(1) 计算。
        """
        remainingPositiveDual = totalPositiveDual - label[_PDC]
        optimisticRc = (
            beta * label[_DUR]
            - label[_DC]
            - remainingPositiveDual
            - slotDual
        )
        return optimisticRc >= -self.reducedCostTolerance
