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
    nonElementaryNegativeCompletions: int = 0
    dominanceComparisons: int = 0
    earlyStopped: bool = False

    # ng-route + DSSR diagnostics.  ``dssrPasses`` counts relaxed labeling
    # passes inside one exact pricing call.  A pricing call still returns only
    # genuine elementary columns, or an exact certificate that none exists.
    dssrPasses: int = 0
    dssrRefinements: int = 0
    dssrCustomersAdded: int = 0
    dssrCriticalCustomers: int = 0
    dssrFallbackToElementary: bool = False
    ngMode: bool = False


@dataclass(slots=True)
class _ThresholdLabel:
    # Mathematical resources.
    node: int
    energy: float
    duration: float
    rc: float

    # ng-route memory plus all DSSR-critical customers visited so far.
    # A customer in this mask cannot be visited next.  Noncritical customers
    # can disappear from the mask according to the ng-neighborhood update and
    # may therefore be repeated in the relaxation.
    memoryMask: int

    # Parity of visits to the active three-row SRC customer sets.  This is
    # kept independently from ng memory: forgetting a customer for elementarity
    # must not forget its SRC state.
    srcParityKey: int = 0

    # Reconstruction / exact master-only-row protection metadata.
    pred: object = None
    addedNode: object = None
    protectedPrefixState: int = ColumnPrefixTrie.DEAD
    active: bool = True

@dataclass
class ThresholdHeuristicPricingStatistics:
    evaluatedSequences: int = 0
    cacheLikeDuplicates: int = 0
    infeasibleSequences: int = 0
    thresholdRejected: int = 0
    branchRejected: int = 0
    existingRejected: int = 0
    localMoveCandidates: int = 0
    greedyCandidates: int = 0
    negativeCandidates: int = 0


class HeuristicThresholdPricing:
    """Cheap, non-certifying pricing used before exact ESPPRC labeling.

    The heuristic never replaces exact pricing.  It searches two neighborhoods:

    1. ``columns-from-columns``: remove, insert, or replace one customer in
       promising current RMP columns.  For every customer order the complete
       route evaluator re-optimizes the BSS placement exactly.
    2. ``dual-guided insertion``: start from high-dual customers and repeatedly
       insert the customer/position producing the best exact reduced cost while
       respecting the fixed makespan threshold.

    If no negative column is found, the caller must invoke
    :class:`ExactThresholdLabelingPricing` to certify column-generation
    optimality.  Existing BSS no-good rows are master-only, so a genuinely new
    route has coefficient zero in them and they do not appear in the reduced
    cost below.
    """

    def __init__(
        self,
        modelData,
        routeEvaluator,
        reducedCostTolerance=1e-8,
        maxSeedColumns=24,
        maxInsertionCustomersPerSeed=10,
        maxReplacementCustomersPerSeed=5,
        maxReplacementPositionsPerSeed=5,
        greedyStarts=8,
        greedyCandidateLimit=16,
    ):
        self.data = modelData
        self.routeEvaluator = routeEvaluator
        self.reducedCostTolerance = float(reducedCostTolerance)
        self.maxSeedColumns = max(1, int(maxSeedColumns))
        self.maxInsertionCustomersPerSeed = max(1, int(maxInsertionCustomersPerSeed))
        self.maxReplacementCustomersPerSeed = max(0, int(maxReplacementCustomersPerSeed))
        self.maxReplacementPositionsPerSeed = max(0, int(maxReplacementPositionsPerSeed))
        self.greedyStarts = max(0, int(greedyStarts))
        self.greedyCandidateLimit = max(1, int(greedyCandidateLimit))
        self.tasks = tuple(modelData.C)
        self.lastStatistics = None

    @staticmethod
    def _existingIndex(existingSignatures):
        if hasattr(existingSignatures, 'signatureToIndex'):
            return existingSignatures.signatureToIndex
        if isinstance(existingSignatures, dict):
            return existingSignatures
        return {tuple(signature): None for signature in existingSignatures}

    @staticmethod
    def _columnReducedCost(column, duals, phase):
        rc = 0.0 if phase == 1 else 1.0
        coverage = getattr(duals, 'coverage', {})
        for task in column.taskSet:
            rc -= float(coverage.get(task, 0.0))
        for cut, eta in getattr(duals, 'src', {}).items():
            coefficient = cut.coefficient(column)
            if coefficient:
                rc -= float(eta) * float(coefficient)
        return float(rc)

    def _rankOutsideTasks(self, sequence, duals, limit):
        present = set(sequence)
        coverage = getattr(duals, 'coverage', {})
        ranked = [
            task for task in self.tasks
            if task not in present
        ]
        # Positive/larger coverage duals are most likely to decrease reduced
        # cost.  Deterministic task id tie-break keeps runs reproducible.
        ranked.sort(key=lambda task: (-float(coverage.get(task, 0.0)), task))
        return ranked[:limit]

    def _trySequence(
        self,
        sequence,
        duals,
        threshold,
        phase,
        branchingState,
        existingIndex,
        negatives,
        stats,
        source,
        sequenceCache,
    ):
        sequence = tuple(sequence)
        if not sequence or len(set(sequence)) != len(sequence):
            return None
        if sequence in sequenceCache:
            stats.cacheLikeDuplicates += 1
            return sequenceCache[sequence]
        stats.evaluatedSequences += 1

        column = self.routeEvaluator.evaluate(sequence)
        if column is None:
            stats.infeasibleSequences += 1
            sequenceCache[sequence] = None
            return None
        if float(column.duration) > float(threshold) + 1e-9:
            stats.thresholdRejected += 1
            sequenceCache[sequence] = None
            return None
        if branchingState is not None and not branchingState.isColumnCompatible(column):
            stats.branchRejected += 1
            sequenceCache[sequence] = None
            return None

        rc = self._columnReducedCost(column, duals, phase)
        evaluated = (column, rc)
        sequenceCache[sequence] = evaluated
        signature = tuple(column.signature)
        if signature in existingIndex:
            stats.existingRejected += 1
            # Still return the evaluated pair to the greedy builder: an
            # existing feasible route can be a useful seed for further inserts.
            return evaluated

        if rc < -self.reducedCostTolerance:
            previous = negatives.get(signature)
            candidate = ThresholdPricingCandidate(column=column, reducedCost=rc)
            if previous is None or rc < previous.reducedCost:
                negatives[signature] = candidate
                stats.negativeCandidates += 1
                if source == 'local':
                    stats.localMoveCandidates += 1
                elif source == 'greedy':
                    stats.greedyCandidates += 1
        return evaluated

    def _bestInsertion(
        self,
        sequence,
        task,
        duals,
        threshold,
        phase,
        branchingState,
        existingIndex,
        negatives,
        stats,
        source,
        sequenceCache,
    ):
        best = None
        for position in range(len(sequence) + 1):
            newSequence = sequence[:position] + (task,) + sequence[position:]
            evaluated = self._trySequence(
                newSequence,
                duals,
                threshold,
                phase,
                branchingState,
                existingIndex,
                negatives,
                stats,
                source,
                sequenceCache,
            )
            if evaluated is None:
                continue
            column, rc = evaluated
            candidate = (float(rc), float(column.duration), tuple(column.tasks), column)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        return best

    def _columnsFromColumns(
        self,
        seedColumns,
        duals,
        threshold,
        phase,
        branchingState,
        existingIndex,
        negatives,
        stats,
        sequenceCache,
    ):
        coverage = getattr(duals, 'coverage', {})
        seeds = []
        seenSeedSignatures = set()
        for column in seedColumns or ():
            if column is None or column.isEmpty:
                continue
            signature = tuple(column.signature)
            if signature in seenSeedSignatures:
                continue
            seenSeedSignatures.add(signature)
            seeds.append(column)
            if len(seeds) >= self.maxSeedColumns:
                break

        for seed in seeds:
            sequence = tuple(seed.tasks)
            if not sequence:
                continue

            # (i) Remove one customer.
            if len(sequence) > 1:
                for removePos in range(len(sequence)):
                    reduced = sequence[:removePos] + sequence[removePos + 1:]
                    self._trySequence(
                        reduced, duals, threshold, phase, branchingState,
                        existingIndex, negatives, stats, 'local', sequenceCache,
                    )

            outside = self._rankOutsideTasks(
                sequence,
                duals,
                self.maxInsertionCustomersPerSeed,
            )

            # (ii) Insert one customer.  For a fixed inserted customer, keep
            # searching all positions; route RC depends on the served set while
            # duration/branch feasibility depends on the order/BSS placement.
            for task in outside:
                self._bestInsertion(
                    sequence, task, duals, threshold, phase, branchingState,
                    existingIndex, negatives, stats, 'local', sequenceCache,
                )

            # (iii) Insert one, remove one.  Prefer removing low-dual customers
            # and inserting high-dual customers; this mirrors the paper's
            # column-modification heuristic while keeping the neighborhood
            # bounded for N=25+.
            if (
                self.maxReplacementCustomersPerSeed > 0
                and self.maxReplacementPositionsPerSeed > 0
                and sequence
            ):
                removePositions = sorted(
                    range(len(sequence)),
                    key=lambda pos: (float(coverage.get(sequence[pos], 0.0)), pos),
                )[:self.maxReplacementPositionsPerSeed]
                insertTasks = outside[:self.maxReplacementCustomersPerSeed]
                for removePos in removePositions:
                    base = sequence[:removePos] + sequence[removePos + 1:]
                    for task in insertTasks:
                        if task in base:
                            continue
                        self._bestInsertion(
                            base, task, duals, threshold, phase, branchingState,
                            existingIndex, negatives, stats, 'local', sequenceCache,
                        )

    def _dualGuidedInsertion(
        self,
        duals,
        threshold,
        phase,
        branchingState,
        existingIndex,
        negatives,
        stats,
        sequenceCache,
    ):
        if self.greedyStarts <= 0:
            return
        coverage = getattr(duals, 'coverage', {})
        rankedStarts = sorted(
            self.tasks,
            key=lambda task: (-float(coverage.get(task, 0.0)), task),
        )[:self.greedyStarts]

        for startTask in rankedStarts:
            evaluated = self._trySequence(
                (startTask,), duals, threshold, phase, branchingState,
                existingIndex, negatives, stats, 'greedy', sequenceCache,
            )
            if evaluated is None:
                continue
            currentColumn, currentRc = evaluated
            currentSequence = tuple(currentColumn.tasks)

            while len(currentSequence) < len(self.tasks):
                outside = self._rankOutsideTasks(
                    currentSequence,
                    duals,
                    self.greedyCandidateLimit,
                )
                best = None
                for task in outside:
                    insertion = self._bestInsertion(
                        currentSequence, task, duals, threshold, phase,
                        branchingState, existingIndex, negatives, stats,
                        'greedy', sequenceCache,
                    )
                    if insertion is None:
                        continue
                    if best is None or insertion[:3] < best[:3]:
                        best = insertion

                if best is None:
                    break
                nextRc, _, _, nextColumn = best
                # There is no reason for this greedy heuristic to cross an
                # increasing-RC step in a complete routing graph.  Exact pricing
                # remains responsible for any such nonlocal opportunity.
                if nextRc >= currentRc - 1e-12:
                    break
                currentColumn = nextColumn
                currentSequence = tuple(nextColumn.tasks)
                currentRc = float(nextRc)

    def price(
        self,
        duals,
        existingSignatures,
        threshold,
        phase=2,
        maxColumns=None,
        branchingState=None,
        seedColumns=None,
    ):
        if phase not in (1, 2):
            raise ValueError('phase must be 1 or 2')
        existingIndex = self._existingIndex(existingSignatures)
        negatives = {}
        stats = ThresholdHeuristicPricingStatistics()
        sequenceCache = {}

        self._columnsFromColumns(
            seedColumns=seedColumns,
            duals=duals,
            threshold=threshold,
            phase=phase,
            branchingState=branchingState,
            existingIndex=existingIndex,
            negatives=negatives,
            stats=stats,
            sequenceCache=sequenceCache,
        )

        self._dualGuidedInsertion(
            duals=duals,
            threshold=threshold,
            phase=phase,
            branchingState=branchingState,
            existingIndex=existingIndex,
            negatives=negatives,
            stats=stats,
            sequenceCache=sequenceCache,
        )

        candidates = list(negatives.values())
        candidates.sort(
            key=lambda candidate: (
                candidate.reducedCost,
                candidate.column.duration,
                candidate.column.nodes,
            )
        )
        if maxColumns is not None:
            candidates = candidates[:maxColumns]
        self.lastStatistics = stats
        return candidates


class _DominanceFront:
    """Dominance front for the current relaxed state space.

    ``memoryMask`` is the *current* forbidden-customer memory: ng memory plus
    all already visited DSSR-critical customers.  Hence the usual subset rule
    remains valid: a label remembering fewer forbidden customers has at least
    as many feasible continuations.
    """

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
        if (a.memoryMask & b.memoryMask) != a.memoryMask:
            return False
        tol = self.tol
        return (
            a.energy >= b.energy - tol
            and a.duration <= b.duration + tol
            and a.rc <= b.rc + tol
        )

    def insert(self, label, queue, stats):
        card = label.memoryMask.bit_count()

        # Potential dominators must have a subset memory mask, hence no larger
        # cardinality.  Scan only fronts that actually exist.
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
    """Exact fixed-threshold pricing with optional ng-route + DSSR.

    Reduced cost:

      Phase I:  -sum_i pi_i a_ir + SRC terms
      Phase II: 1 - sum_i pi_i a_ir + SRC terms

    When ``useNgDssr`` is false, this is the original elementary ESPPRC.
    When it is true, each exact pricing call starts from the current ng-route
    relaxation.  If a negative non-elementary walk is found, every repeated
    customer on those walks is made DSSR-critical and the relaxed labeling is
    solved again.  Critical customers are remembered permanently for the
    remainder of this fixed-threshold BPC solve.  In the worst case all
    customers become critical, which is exactly the original ESPPRC.

    Crucially, the method *never* returns a non-elementary walk as a master
    column.  Returning no column is also exact: it only happens after a relaxed
    pass has found no negative walk, or after the fallback full-elementary pass
    has certified the same fact.
    """

    def __init__(
        self,
        modelData,
        routeEvaluator,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
        useNgDssr=True,
        ngNeighborhoodSize=6,
        dssrMaxIterations=None,
        dssrMaxNonElementaryPerPass=8,
    ):
        self.data = modelData
        self.routeEvaluator = routeEvaluator
        self.reducedCostTolerance = float(reducedCostTolerance)
        self.dominanceTolerance = float(dominanceTolerance)
        self.useNgDssr = bool(useNgDssr)
        self.stationNode = modelData.S[0]
        self.tasks = tuple(modelData.C)
        self.taskToBit = {task: 1 << index for index, task in enumerate(self.tasks)}
        self.allTaskMask = (1 << len(self.tasks)) - 1
        self.ngNeighborhoodSize = max(1, min(int(ngNeighborhoodSize), max(1, len(self.tasks))))
        self.dssrMaxIterations = (
            max(1, len(self.tasks))
            if dssrMaxIterations is None
            else max(1, int(dssrMaxIterations))
        )
        self.dssrMaxNonElementaryPerPass = max(1, int(dssrMaxNonElementaryPerPass))
        self.lastStatistics = None
        self._dssrCriticalMask = 0

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
        self._ngNeighborhoodMasks = self._computeNgNeighborhoodMasks()

    # ------------------------------------------------------------------
    # ng-route / DSSR state management
    # ------------------------------------------------------------------
    def resetDssr(self):
        """Reset DSSR critical customers for a new fixed-threshold solve."""
        self._dssrCriticalMask = 0

    @property
    def dssrCriticalMask(self):
        return int(self._dssrCriticalMask)

    @property
    def dssrCriticalCustomers(self):
        return tuple(
            task for task in self.tasks
            if self._dssrCriticalMask & self.taskToBit[task]
        )

    def ngNeighborhood(self, task):
        """Return the deterministic ng-neighborhood of ``task`` for diagnostics."""
        mask = self._ngNeighborhoodMasks[task]
        return tuple(t for t in self.tasks if mask & self.taskToBit[t])

    def _computeNgNeighborhoodMasks(self):
        """Build fixed nearest-customer ng neighborhoods.

        The neighborhood contains the customer itself plus the closest
        ``ngNeighborhoodSize-1`` customers according to directed travel time
        from the customer (customer service at the destination is used only as
        a deterministic secondary component).  The particular neighborhood
        rule affects relaxation strength/performance, not exactness, because
        DSSR can always recover full elementarity.
        """
        if not self.tasks:
            return {}
        if self.ngNeighborhoodSize >= len(self.tasks):
            return {task: self.allTaskMask for task in self.tasks}

        result = {}
        for task in self.tasks:
            ranked = []
            for other in self.tasks:
                if other == task:
                    continue
                travel = self.data.t.get((task, other), float('inf'))
                service = float(self.data.p.get(other, 0.0))
                ranked.append((float(travel) + service, other))
            ranked.sort(key=lambda item: (item[0], item[1]))
            chosen = [task] + [other for _, other in ranked[: self.ngNeighborhoodSize - 1]]
            mask = 0
            for customer in chosen:
                mask |= self.taskToBit[customer]
            result[task] = mask
        return result

    # ------------------------------------------------------------------
    # Static preprocessing / master-dual helpers
    # ------------------------------------------------------------------
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
    def _bucketKey(label):
        return (label.node, label.srcParityKey)

    def _durationCanFinish(self, node, duration, threshold):
        return duration + self._minDurationToEnd.get(node, float('inf')) <= threshold + 1e-9

    def _arcForbidden(self, branchState, u, v):
        return branchState is not None and (u, v) in branchState.forbiddenArcs

    @staticmethod
    def _mergePassStatistics(total, part):
        integerFields = (
            'generatedLabels',
            'acceptedLabels',
            'dominatedLabels',
            'removedByDominance',
            'boundPrunedLabels',
            'durationPrunedLabels',
            'branchPrunedLabels',
            'completedRoutes',
            'negativeCompletions',
            'nonElementaryNegativeCompletions',
            'dominanceComparisons',
        )
        for name in integerFields:
            setattr(total, name, getattr(total, name) + getattr(part, name))
        total.earlyStopped = total.earlyStopped or part.earlyStopped

    # ------------------------------------------------------------------
    # Public exact pricing entry point
    # ------------------------------------------------------------------
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

        totalStats = ThresholdLabelingStatistics(ngMode=self.useNgDssr)

        # Baseline/full-elementary mode is the same engine with every customer
        # DSSR-critical, so ng memory can never forget a visited customer.
        if not self.useNgDssr:
            candidates, _ = self._pricePass(
                duals=duals,
                existingIndex=existingIndex,
                threshold=threshold,
                phase=phase,
                maxColumns=maxColumns,
                branchingState=branchingState,
                protectedTrie=protectedTrie,
                activeSrcCuts=activeSrcCuts,
                criticalMask=self.allTaskMask,
                stats=totalStats,
            )
            totalStats.dssrPasses = 1
            totalStats.dssrCriticalCustomers = len(self.tasks)
            self.lastStatistics = totalStats
            return candidates

        criticalMask = int(self._dssrCriticalMask)
        refinementsThisCall = 0

        while True:
            passStats = ThresholdLabelingStatistics(ngMode=True)
            candidates, repeatedMask = self._pricePass(
                duals=duals,
                existingIndex=existingIndex,
                threshold=threshold,
                phase=phase,
                maxColumns=maxColumns,
                branchingState=branchingState,
                protectedTrie=protectedTrie,
                activeSrcCuts=activeSrcCuts,
                criticalMask=criticalMask,
                stats=passStats,
            )
            self._mergePassStatistics(totalStats, passStats)
            totalStats.dssrPasses += 1

            # Any returned columns are genuine elementary routes.  We can stop
            # immediately and let the master reoptimize; certification is only
            # needed when the pricing call returns no columns.
            if candidates:
                self._dssrCriticalMask = criticalMask
                totalStats.dssrCriticalCustomers = criticalMask.bit_count()
                self.lastStatistics = totalStats
                return candidates

            if repeatedMask:
                newCritical = repeatedMask & ~criticalMask
                if newCritical:
                    criticalMask |= newCritical
                    self._dssrCriticalMask = criticalMask
                    refinementsThisCall += 1
                    totalStats.dssrRefinements += 1
                    totalStats.dssrCustomersAdded += newCritical.bit_count()

                    # DSSR is finite because every refinement makes at least one
                    # customer permanently elementary.  The iteration cap is a
                    # performance guard only: hitting it falls back to the full
                    # elementary state space rather than weakening exactness.
                    if (
                        refinementsThisCall >= self.dssrMaxIterations
                        and criticalMask != self.allTaskMask
                    ):
                        newlyAdded = self.allTaskMask & ~criticalMask
                        criticalMask = self.allTaskMask
                        self._dssrCriticalMask = criticalMask
                        totalStats.dssrCustomersAdded += newlyAdded.bit_count()
                        totalStats.dssrFallbackToElementary = True
                    continue

                # Defensive progress guard.  A repeated DSSR-critical customer
                # should be impossible; if numerical/state logic ever reaches
                # here, recover exactness by switching to full elementarity.
                if criticalMask != self.allTaskMask:
                    newlyAdded = self.allTaskMask & ~criticalMask
                    criticalMask = self.allTaskMask
                    self._dssrCriticalMask = criticalMask
                    totalStats.dssrCustomersAdded += newlyAdded.bit_count()
                    totalStats.dssrFallbackToElementary = True
                    continue

            # The relaxed ng problem contains every elementary route.  If a
            # complete relaxed pass has no negative walk, there cannot be a
            # negative elementary route either: this is an exact certificate.
            self._dssrCriticalMask = criticalMask
            totalStats.dssrCriticalCustomers = criticalMask.bit_count()
            self.lastStatistics = totalStats
            return []

    # ------------------------------------------------------------------
    # One labeling pass for a fixed ng/DSSR memory state
    # ------------------------------------------------------------------
    def _pricePass(
        self,
        duals,
        existingIndex,
        threshold,
        phase,
        maxColumns,
        branchingState,
        protectedTrie,
        activeSrcCuts,
        criticalMask,
        stats,
    ):
        rootPrefix = protectedTrie.advance(0, self.data.startNode)
        fullElementary = criticalMask == self.allTaskMask

        # The positive-dual completion bound is valid only in the full
        # elementary state space.  Under ng relaxation a forgotten customer may
        # be revisited and collect its dual again, so using the old bound there
        # would invalidate the relaxation/certificate.
        if fullElementary:
            positiveDual = {
                task: max(0.0, float(duals.coverage.get(task, 0.0)))
                for task in self.tasks
            }
            totalPositiveDual = sum(positiveDual.values())
            positiveSumByMask = {0: 0.0}
        else:
            positiveDual = None
            totalPositiveDual = None
            positiveSumByMask = None

        queue = deque()
        buckets = {}

        root = _ThresholdLabel(
            node=self.data.startNode,
            energy=float(self.data.Q),
            duration=0.0,
            rc=(0.0 if phase == 1 else 1.0),
            memoryMask=0,
            srcParityKey=0,
            pred=None,
            addedNode=None,
            protectedPrefixState=rootPrefix,
        )
        rootKey = self._bucketKey(root)
        buckets[rootKey] = _DominanceFront(self.dominanceTolerance)
        buckets[rootKey].insert(root, queue, stats)
        stats.acceptedLabels += 1

        negatives = {}
        repeatedMask = 0
        nonElementaryNegativeCount = 0
        earlyStop = False

        while queue and not earlyStop:
            label = queue.popleft()
            if not label.active:
                continue

            if fullElementary and self._cannotBecomeNegative(
                label, totalPositiveDual, positiveSumByMask
            ):
                stats.boundPrunedLabels += 1
                continue
            if not self._durationCanFinish(label.node, label.duration, threshold):
                stats.durationPrunedLabels += 1
                continue

            # Complete to depot from customer or station.  Empty routes are not
            # pricing columns in the vehicle-free master.  Once a customer has
            # been visited, memoryMask is nonzero (BSS visits preserve it).
            if label.memoryMask and label.node != self.data.startNode:
                completed = self._completeToDepot(label, threshold, branchingState)
                if completed is not None:
                    completeDuration, completeRc = completed
                    stats.completedRoutes += 1
                    if completeRc < -self.reducedCostTolerance:
                        stats.negativeCompletions += 1
                        nodes = self._reconstructNodes(label) + (self.data.endNode,)
                        repeats = self._repeatedCustomerMask(nodes)

                        if repeats:
                            # Never expose a non-elementary walk to the master.
                            repeatedMask |= repeats
                            nonElementaryNegativeCount += 1
                            stats.nonElementaryNegativeCompletions += 1
                            if (
                                nonElementaryNegativeCount
                                >= self.dssrMaxNonElementaryPerPass
                            ):
                                # We already have enough information to tighten
                                # the state space.  No certificate is claimed.
                                earlyStop = True
                                break
                            continue

                        signature = tuple(nodes)
                        if signature not in existingIndex:
                            column = self.routeEvaluator.evaluateNodes(signature)
                            if (
                                column is not None
                                and column.duration <= threshold + 1e-8
                            ):
                                previous = negatives.get(signature)
                                if previous is None or completeRc < previous.reducedCost:
                                    negatives[signature] = ThresholdPricingCandidate(
                                        column=column,
                                        reducedCost=float(completeRc),
                                    )
                                    if maxColumns is not None and len(negatives) >= maxColumns:
                                        earlyStop = True
                                        break

            # Customer extensions.
            for task in self.tasks:
                bit = self.taskToBit[task]
                if label.memoryMask & bit:
                    continue
                if self._arcForbidden(branchingState, label.node, task):
                    stats.branchPrunedLabels += 1
                    continue
                nxt = self._extendToCustomer(
                    label=label,
                    task=task,
                    bit=bit,
                    coverageDuals=duals.coverage,
                    activeSrcCuts=activeSrcCuts,
                    protectedTrie=protectedTrie,
                    criticalMask=criticalMask,
                )
                if nxt is None:
                    continue
                stats.generatedLabels += 1
                if nxt.duration > threshold + 1e-9 or not self._durationCanFinish(
                    nxt.node, nxt.duration, threshold
                ):
                    stats.durationPrunedLabels += 1
                    continue

                if fullElementary:
                    if nxt.memoryMask not in positiveSumByMask:
                        # In the full-elementary pass memoryMask is exactly the
                        # set of visited customers.
                        positiveSumByMask[nxt.memoryMask] = (
                            positiveSumByMask[label.memoryMask]
                            + positiveDual[task]
                        )
                    if self._cannotBecomeNegative(
                        nxt, totalPositiveDual, positiveSumByMask
                    ):
                        stats.boundPrunedLabels += 1
                        continue

                key = self._bucketKey(nxt)
                front = buckets.get(key)
                if front is None:
                    front = _DominanceFront(self.dominanceTolerance)
                    buckets[key] = front
                if front.insert(nxt, queue, stats):
                    stats.acceptedLabels += 1

            # Repeatable BSS extension from a customer only.  BSS is not an
            # elementarity resource and therefore leaves ng/DSSR memory and SRC
            # parity unchanged.
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
                        elif fullElementary and self._cannotBecomeNegative(
                            nxt, totalPositiveDual, positiveSumByMask
                        ):
                            stats.boundPrunedLabels += 1
                        else:
                            key = self._bucketKey(nxt)
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
        return candidates, repeatedMask

    def _extendToCustomer(
        self,
        label,
        task,
        bit,
        coverageDuals,
        activeSrcCuts,
        protectedTrie,
        criticalMask,
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

        # Standard ng-memory update, strengthened by DSSR: critical customers
        # are never forgotten, while noncritical memory is intersected with the
        # neighborhood of the newly visited customer.
        retainMask = self._ngNeighborhoodMasks[task] | criticalMask
        newMemory = (label.memoryMask & retainMask) | bit

        # In the relaxed problem a repeated customer collects its coverage dual
        # again.  This makes the transition cost depend only on the relaxed
        # state rather than on hidden full-history information, which is needed
        # for valid ng-state dominance.  Every elementary path is unaffected.
        newRc = label.rc - float(coverageDuals.get(task, 0.0))

        # SRC state is independent of ng memory.  For divisor-2 three-row cuts,
        # every second visit to the cut set incurs the nonnegative penalty.
        # On elementary routes this is exactly floor(|S cap r|/2).
        newParity = label.srcParityKey
        for idx, (cutMask, penalty) in enumerate(activeSrcCuts):
            if bit & cutMask:
                parityBit = 1 << idx
                if newParity & parityBit:
                    newRc += penalty
                newParity ^= parityBit

        return _ThresholdLabel(
            node=task,
            energy=newEnergy,
            duration=label.duration + travel + float(self.data.p[task]),
            rc=newRc,
            memoryMask=newMemory,
            srcParityKey=newParity,
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
            memoryMask=label.memoryMask,
            srcParityKey=label.srcParityKey,
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
        visitedPositive = positiveSumByMask.get(label.memoryMask, 0.0)
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

    def _repeatedCustomerMask(self, nodes):
        seen = 0
        repeated = 0
        for node in nodes:
            bit = self.taskToBit.get(node, 0)
            if not bit:
                continue
            if seen & bit:
                repeated |= bit
            seen |= bit
        return repeated
