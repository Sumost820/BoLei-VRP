import heapq
from collections import deque
from dataclasses import dataclass, field

from boleiScheduling.bpc1.columnManager import ColumnPrefixTrie


@dataclass
class ThresholdPricingCandidate:
    column: object
    reducedCost: float


@dataclass
class ThresholdLabelingStatistics:
    generatedLabels: int = 0
    acceptedLabels: int = 0
    dominatedLabels: int = 0
    removedByDominance: int = 0
    boundPrunedLabels: int = 0
    durationPrunedLabels: int = 0
    branchPrunedLabels: int = 0
    completedRoutes: int = 0
    negativeCompletions: int = 0
    dominanceComparisons: int = 0
    earlyStopped: bool = False


@dataclass(slots=True)
class _ThresholdLabel:
    # Mathematical resources.
    node: int
    energy: float
    duration: float
    rc: float
    unreachableMask: int

    # Reconstruction / exact master-only-row protection metadata.
    pred: object = None
    addedNode: object = None
    protectedPrefixState: int = ColumnPrefixTrie.DEAD
    active: bool = True


class _DominanceFront:
    """Scan only labels that actually exist; never enumerate subset masks."""

    def __init__(self, tolerance):
        self.tol = float(tolerance)
        self.byCardinality = {}

    def _dominates(self, a, b, stats):
        stats.dominanceComparisons += 1
        # A protected live prefix is comparable only with the same automaton
        # state.  DEAD can safely dominate any live protected prefix because it
        # can no longer terminate at a master-only penalized named route.
        noveltyCompatible = (
            a.protectedPrefixState == ColumnPrefixTrie.DEAD
            or a.protectedPrefixState == b.protectedPrefixState
        )
        if not noveltyCompatible:
            return False
        if (a.unreachableMask & b.unreachableMask) != a.unreachableMask:
            return False
        tol = self.tol
        return (
            a.energy >= b.energy - tol
            and a.duration <= b.duration + tol
            and a.rc <= b.rc + tol
        )

    def insert(self, label, queue, stats):
        card = label.unreachableMask.bit_count()

        # Potential dominators must have a subset mask, hence no larger
        # cardinality.  We only scan actual labels in those fronts.
        for oldCard, oldLabels in tuple(self.byCardinality.items()):
            if oldCard > card:
                continue
            for old in oldLabels:
                if old.active and self._dominates(old, label, stats):
                    stats.dominatedLabels += 1
                    return False

        # Remove labels dominated by the new one.  Supersets cannot have a
        # smaller cardinality.
        for oldCard, oldLabels in tuple(self.byCardinality.items()):
            if oldCard < card:
                continue
            survivors = []
            for old in oldLabels:
                if not old.active:
                    continue
                if self._dominates(label, old, stats):
                    old.active = False
                    stats.removedByDominance += 1
                else:
                    survivors.append(old)
            if survivors:
                self.byCardinality[oldCard] = survivors
            else:
                self.byCardinality.pop(oldCard, None)

        self.byCardinality.setdefault(card, []).append(label)
        queue.append(label)
        return True


class ExactThresholdLabelingPricing:
    """Exact vehicle-free ESPPRC for a fixed makespan threshold.

    Reduced cost:

      Phase I:  -sum_i pi_i a_ir + SRC terms
      Phase II: 1 - sum_i pi_i a_ir + SRC terms

    Route duration is a monotone feasibility resource constrained by
    ``duration <= threshold``.  There is no vehicle index, slot dual, or
    makespan dual beta_k.
    """

    def __init__(
        self,
        modelData,
        routeEvaluator,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
    ):
        self.data = modelData
        self.routeEvaluator = routeEvaluator
        self.reducedCostTolerance = float(reducedCostTolerance)
        self.dominanceTolerance = float(dominanceTolerance)
        self.stationNode = modelData.S[0]
        self.tasks = tuple(modelData.C)
        self.taskToBit = {task: 1 << index for index, task in enumerate(self.tasks)}
        self.allTaskMask = (1 << len(self.tasks)) - 1
        self.lastStatistics = None

        # Dense physical-node index used in hot loops.
        self.nodes = tuple(modelData.V)
        self.nodeToIndex = {node: idx for idx, node in enumerate(self.nodes)}
        n = len(self.nodes)
        self.travel = [[None] * n for _ in range(n)]
        self.energy = [[None] * n for _ in range(n)]
        for (u, v), value in modelData.t.items():
            if u in self.nodeToIndex and v in self.nodeToIndex:
                self.travel[self.nodeToIndex[u]][self.nodeToIndex[v]] = float(value)
        for (u, v), value in modelData.e.items():
            if u in self.nodeToIndex and v in self.nodeToIndex:
                self.energy[self.nodeToIndex[u]][self.nodeToIndex[v]] = float(value)

        self._minDurationToEnd = self._computeRelaxedDurationToEnd()

    def _computeRelaxedDurationToEnd(self):
        """Shortest remaining duration ignoring energy and elementarity."""
        reverse = {node: [] for node in self.nodes}
        end = self.data.endNode
        for u, v in self.data.A:
            service = 0.0 if v == end else float(self.data.p.get(v, 0.0))
            reverse[v].append((u, float(self.data.t[u, v]) + service))

        dist = {node: float('inf') for node in self.nodes}
        dist[end] = 0.0
        heap = [(0.0, end)]
        while heap:
            d, v = heapq.heappop(heap)
            if d != dist[v]:
                continue
            for u, cost in reverse.get(v, ()):
                nd = d + cost
                if nd < dist[u]:
                    dist[u] = nd
                    heapq.heappush(heap, (nd, u))
        return dist

    def _activeSrcCuts(self, duals):
        result = []
        for cut, eta in getattr(duals, 'src', {}).items():
            eta = float(eta)
            if eta > 1e-8:
                raise RuntimeError(f'SRC dual must be nonpositive, got {eta} for {cut}')
            rho = -min(0.0, eta)
            if rho <= 1e-12:
                continue
            mask = 0
            for task in cut.customers:
                mask |= self.taskToBit.get(task, 0)
            result.append((mask, rho))
        return tuple(result)

    def _buildProtectedTrie(self, signatures):
        trie = ColumnPrefixTrie()
        for signature in signatures or ():
            trie.insert(tuple(signature))
        return trie

    @staticmethod
    def _srcParity(mask, activeSrcCuts):
        key = 0
        for idx, (cutMask, _) in enumerate(activeSrcCuts):
            if (mask & cutMask).bit_count() & 1:
                key |= 1 << idx
        return key

    def _bucketKey(self, label, activeSrcCuts):
        return (label.node, self._srcParity(label.unreachableMask, activeSrcCuts))

    def _durationCanFinish(self, node, duration, threshold):
        return duration + self._minDurationToEnd.get(node, float('inf')) <= threshold + 1e-9

    def _arcForbidden(self, branchState, u, v):
        return branchState is not None and (u, v) in branchState.forbiddenArcs

    def price(
        self,
        duals,
        existingSignatures,
        threshold,
        phase=2,
        maxColumns=None,
        branchingState=None,
        protectedSignatures=None,
    ):
        threshold = float(threshold)
        if phase not in (1, 2):
            raise ValueError('phase must be 1 or 2')

        if hasattr(existingSignatures, 'signatureToIndex'):
            existingIndex = existingSignatures.signatureToIndex
        else:
            existingIndex = {tuple(sig): None for sig in existingSignatures}

        activeSrcCuts = self._activeSrcCuts(duals)
        protectedTrie = self._buildProtectedTrie(protectedSignatures)
        rootPrefix = protectedTrie.advance(0, self.data.startNode)

        positiveDual = {
            task: max(0.0, float(duals.coverage.get(task, 0.0)))
            for task in self.tasks
        }
        totalPositiveDual = sum(positiveDual.values())
        positiveSumByMask = {0: 0.0}

        stats = ThresholdLabelingStatistics()
        queue = deque()
        buckets = {}

        root = _ThresholdLabel(
            node=self.data.startNode,
            energy=float(self.data.Q),
            duration=0.0,
            rc=(0.0 if phase == 1 else 1.0),
            unreachableMask=0,
            pred=None,
            addedNode=None,
            protectedPrefixState=rootPrefix,
        )
        rootKey = self._bucketKey(root, activeSrcCuts)
        buckets[rootKey] = _DominanceFront(self.dominanceTolerance)
        buckets[rootKey].insert(root, queue, stats)
        stats.acceptedLabels += 1

        negatives = {}
        earlyStop = False

        while queue and not earlyStop:
            label = queue.popleft()
            if not label.active:
                continue

            if self._cannotBecomeNegative(
                label, totalPositiveDual, positiveSumByMask
            ):
                stats.boundPrunedLabels += 1
                continue
            if not self._durationCanFinish(label.node, label.duration, threshold):
                stats.durationPrunedLabels += 1
                continue

            # Complete to depot from customer or station.  Empty routes are not
            # pricing columns in the vehicle-free master.
            if label.unreachableMask and label.node != self.data.startNode:
                completed = self._completeToDepot(label, threshold, branchingState)
                if completed is not None:
                    completeDuration, completeRc = completed
                    stats.completedRoutes += 1
                    if completeRc < -self.reducedCostTolerance:
                        nodes = self._reconstructNodes(label) + (self.data.endNode,)
                        signature = tuple(nodes)
                        if signature not in existingIndex:
                            column = self.routeEvaluator.evaluateNodes(signature)
                            if column is not None and column.duration <= threshold + 1e-8:
                                previous = negatives.get(signature)
                                if previous is None or completeRc < previous.reducedCost:
                                    negatives[signature] = ThresholdPricingCandidate(
                                        column=column,
                                        reducedCost=float(completeRc),
                                    )
                                    stats.negativeCompletions += 1
                                    if maxColumns is not None and len(negatives) >= maxColumns:
                                        earlyStop = True
                                        break

            # Customer extensions.
            for task in self.tasks:
                bit = self.taskToBit[task]
                if label.unreachableMask & bit:
                    continue
                if self._arcForbidden(branchingState, label.node, task):
                    stats.branchPrunedLabels += 1
                    continue
                nxt = self._extendToCustomer(
                    label,
                    task,
                    bit,
                    duals.coverage,
                    activeSrcCuts,
                    protectedTrie,
                )
                if nxt is None:
                    continue
                stats.generatedLabels += 1
                if nxt.duration > threshold + 1e-9 or not self._durationCanFinish(
                    nxt.node, nxt.duration, threshold
                ):
                    stats.durationPrunedLabels += 1
                    continue
                if nxt.unreachableMask not in positiveSumByMask:
                    positiveSumByMask[nxt.unreachableMask] = (
                        positiveSumByMask[label.unreachableMask]
                        + positiveDual[task]
                    )
                if self._cannotBecomeNegative(nxt, totalPositiveDual, positiveSumByMask):
                    stats.boundPrunedLabels += 1
                    continue
                key = self._bucketKey(nxt, activeSrcCuts)
                front = buckets.get(key)
                if front is None:
                    front = _DominanceFront(self.dominanceTolerance)
                    buckets[key] = front
                if front.insert(nxt, queue, stats):
                    stats.acceptedLabels += 1

            # Repeatable BSS extension from a customer only.
            if label.node in self.taskToBit:
                if self._arcForbidden(branchingState, label.node, self.stationNode):
                    stats.branchPrunedLabels += 1
                else:
                    nxt = self._extendToStation(label, protectedTrie)
                    if nxt is not None:
                        stats.generatedLabels += 1
                        if nxt.duration > threshold + 1e-9 or not self._durationCanFinish(
                            nxt.node, nxt.duration, threshold
                        ):
                            stats.durationPrunedLabels += 1
                        elif self._cannotBecomeNegative(
                            nxt, totalPositiveDual, positiveSumByMask
                        ):
                            stats.boundPrunedLabels += 1
                        else:
                            key = self._bucketKey(nxt, activeSrcCuts)
                            front = buckets.get(key)
                            if front is None:
                                front = _DominanceFront(self.dominanceTolerance)
                                buckets[key] = front
                            if front.insert(nxt, queue, stats):
                                stats.acceptedLabels += 1

        stats.earlyStopped = earlyStop
        candidates = list(negatives.values())
        candidates.sort(key=lambda c: (c.reducedCost, c.column.duration, c.column.nodes))
        if maxColumns is not None:
            candidates = candidates[:maxColumns]
        self.lastStatistics = stats
        return candidates

    def _extendToCustomer(
        self,
        label,
        task,
        bit,
        coverageDuals,
        activeSrcCuts,
        protectedTrie,
    ):
        u = label.node
        ui = self.nodeToIndex.get(u)
        vi = self.nodeToIndex.get(task)
        if ui is None or vi is None:
            return None
        travel = self.travel[ui][vi]
        energyUse = self.energy[ui][vi]
        if travel is None or energyUse is None:
            return None
        newEnergy = label.energy - energyUse - float(self.data.q[task])
        if newEnergy < self.data.QMin - 1e-9:
            return None

        newMask = label.unreachableMask | bit
        newRc = label.rc - float(coverageDuals.get(task, 0.0))
        for cutMask, penalty in activeSrcCuts:
            if task in self.taskToBit and (bit & cutMask):
                if (label.unreachableMask & cutMask).bit_count() & 1:
                    newRc += penalty

        return _ThresholdLabel(
            node=task,
            energy=newEnergy,
            duration=label.duration + travel + float(self.data.p[task]),
            rc=newRc,
            unreachableMask=newMask,
            pred=label,
            addedNode=task,
            protectedPrefixState=protectedTrie.advance(
                label.protectedPrefixState, task
            ),
        )

    def _extendToStation(self, label, protectedTrie):
        u = label.node
        s = self.stationNode
        ui = self.nodeToIndex.get(u)
        si = self.nodeToIndex.get(s)
        if ui is None or si is None:
            return None
        travel = self.travel[ui][si]
        energyUse = self.energy[ui][si]
        if travel is None or energyUse is None:
            return None
        arrivalEnergy = label.energy - energyUse
        if arrivalEnergy < self.data.QMin - 1e-9:
            return None
        return _ThresholdLabel(
            node=s,
            energy=float(self.data.Q),
            duration=label.duration + travel + float(self.data.p[s]),
            rc=label.rc,
            unreachableMask=label.unreachableMask,
            pred=label,
            addedNode=s,
            protectedPrefixState=protectedTrie.advance(
                label.protectedPrefixState, s
            ),
        )

    def _completeToDepot(self, label, threshold, branchingState):
        end = self.data.endNode
        if self._arcForbidden(branchingState, label.node, end):
            return None
        ui = self.nodeToIndex.get(label.node)
        vi = self.nodeToIndex.get(end)
        if ui is None or vi is None:
            return None
        travel = self.travel[ui][vi]
        energyUse = self.energy[ui][vi]
        if travel is None or energyUse is None:
            return None
        if label.energy - energyUse < self.data.QMin - 1e-9:
            return None
        duration = label.duration + travel
        if duration > threshold + 1e-9:
            return None
        return duration, label.rc

    def _cannotBecomeNegative(self, label, totalPositiveDual, positiveSumByMask):
        visitedPositive = positiveSumByMask.get(label.unreachableMask)
        if visitedPositive is None:
            visitedPositive = 0.0
            for task, bit in self.taskToBit.items():
                if label.unreachableMask & bit:
                    visitedPositive += 0.0  # populated by caller on generated masks
        remaining = totalPositiveDual - visitedPositive
        return label.rc - remaining >= -self.reducedCostTolerance

    def _reconstructNodes(self, label):
        added = []
        cursor = label
        while cursor is not None:
            if cursor.addedNode is not None:
                added.append(cursor.addedNode)
            cursor = cursor.pred
        added.reverse()
        return (self.data.startNode,) + tuple(added)
