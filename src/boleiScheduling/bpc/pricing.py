import itertools
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PricingCandidate:
    column: object
    bestSlot: int
    reducedCost: float
    reducedCostBySlot: dict


@dataclass
class LabelingStatistics:
    generatedLabels: int = 0
    acceptedLabels: int = 0
    dominatedLabels: int = 0
    removedByDominance: int = 0
    boundPrunedLabels: int = 0
    branchPrunedLabels: int = 0
    completedRoutes: int = 0
    negativeCompletions: int = 0
    states: int = 0
    perSlot: dict = field(default_factory=dict)


@dataclass
class _PricingLabel:
    currentNode: int
    visitedMask: int
    tasks: tuple
    duration: float
    remainingEnergy: float
    dualCoverage: float
    partialReducedCost: float = 0.0
    remainingPositiveDual: float = 0.0
    requiredMask: int = 0
    # One parity bit per active divisor-2 SRC.  Visiting a customer contained
    # in an SRC toggles its bit; when the old parity is odd, floor(count/2)
    # increases by one and the corresponding nonnegative SRC dual penalty is
    # added to partialReducedCost.
    srcParityMask: int = 0
    active: bool = True


class ExactEnumerativePricing:
    """Exhaustive small-instance pricing oracle, now branch-aware."""

    def __init__(
        self,
        modelData,
        sequenceEvaluator,
        maxEnumeratedTasks=9,
        reducedCostTolerance=1e-8,
    ):
        self.data = modelData
        self.sequenceEvaluator = sequenceEvaluator
        self.maxEnumeratedTasks = maxEnumeratedTasks
        self.reducedCostTolerance = reducedCostTolerance
        self.lastStatistics = None

    def price(
        self,
        duals,
        existingSignatures,
        maxColumns=None,
        branchingState=None,
    ):
        if len(self.data.C) > self.maxEnumeratedTasks:
            raise ValueError(
                f'ExactEnumerativePricing is a reference oracle for small N. '
                f'N={len(self.data.C)} exceeds maxEnumeratedTasks='
                f'{self.maxEnumeratedTasks}.'
            )

        existingSignatures = set(existingSignatures)
        negative = []
        evaluated = 0
        branchRejected = 0

        tasks = tuple(self.data.C)
        for length in range(1, len(tasks) + 1):
            for sequence in itertools.permutations(tasks, length):
                if sequence in existingSignatures:
                    continue

                compatibleSlots = []
                for k in range(self.data.K):
                    if branchingState is None or branchingState.isSequenceCompatible(
                        k,
                        sequence,
                        self.data.startNode,
                        self.data.endNode,
                    ):
                        compatibleSlots.append(k)

                if not compatibleSlots:
                    branchRejected += 1
                    continue

                column = self.sequenceEvaluator.evaluate(sequence)
                if column is None:
                    continue
                evaluated += 1

                dualCoverage = sum(duals.coverage[i] for i in sequence)
                # SRC rows are <= constraints in a minimization RMP, hence
                # their Gurobi duals are nonpositive.  The reduced-cost term is
                # -eta_S * floor(|S∩r|/2), a nonnegative penalty.
                srcReducedCost = 0.0
                for cut, eta in getattr(duals, 'src', {}).items():
                    eta = float(eta)
                    if eta > 1e-8:
                        raise RuntimeError(
                            f'SRC dual must be nonpositive, got {eta} for {cut}'
                        )
                    eta = min(0.0, eta)
                    srcReducedCost += (-eta) * cut.coefficient(column)

                rcBySlot = {}
                for k in compatibleSlots:
                    beta = max(0.0, float(duals.makespan[k]))
                    rcBySlot[k] = (
                        beta * column.duration
                        - dualCoverage
                        - float(duals.slot[k])
                        + srcReducedCost
                    )

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
            'mode': 'enumerative',
            'evaluatedFeasibleSequences': evaluated,
            'branchRejectedSequences': branchRejected,
            'negativeColumnsReturned': len(negative),
        }
        return negative


class ExactLabelingPricing:
    """
    Exact elementary pricing with strong, rigorous set-inclusion dominance.

    Dominance (same slot)
    ---------------------
    Let A and B be labels with the same physical current node, the same last
    customer in the compressed customer sequence, and the same set of already
    satisfied required branch arcs. Define

        g(L) = beta_k * duration(L) - sum_{i in visited(L)} pi_i
               + accumulated SRC dual penalties.

    With divisor-2 SRCs, future SRC increments depend only on the parity of
    |visited(L) intersect S|.  Therefore A safely dominates B if, in addition
    to the existing branch-state conditions, both labels have the same SRC
    parity vector and

        visited(A) subseteq visited(B),
        remainingEnergy(A) >= remainingEnergy(B),
        g(A) <= g(B).

    Proof idea: every elementary suffix feasible from B avoids visited(B), so it
    also avoids the smaller visited(A). Starting from the same physical node and
    with at least as much energy, A can execute the same physical suffix. Both
    labels then receive exactly the same suffix reduced-cost increment, hence A
    cannot finish with a larger reduced cost. This rule is exact; no heuristic
    ng-route relaxation is used.

    Branch propagation
    ------------------
    Required successor arcs are enforced as soon as a customer successor (or
    depot completion) is chosen. This prunes labels before they enter the
    dominance buckets instead of waiting until a later queue pop.
    """

    def __init__(
        self,
        modelData,
        sequenceEvaluator,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
        dualSignTolerance=1e-8,
        dominanceMode='subset',
    ):
        self.data = modelData
        self.sequenceEvaluator = sequenceEvaluator
        self.reducedCostTolerance = reducedCostTolerance
        self.dominanceTolerance = dominanceTolerance
        self.dualSignTolerance = dualSignTolerance
        if dominanceMode not in ('subset', 'equal'):
            raise ValueError("dominanceMode must be 'subset' or 'equal'")
        self.dominanceMode = dominanceMode
        self.stationNode = modelData.S[0]

        self.tasks = tuple(modelData.C)
        self.taskToBit = {
            task: 1 << index
            for index, task in enumerate(self.tasks)
        }
        self.lastStatistics = None

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

        # Normalize SRC duals.  For <= rows in the minimization master eta<=0;
        # rho=-eta>=0 is the additive pricing penalty each time floor(count/2)
        # increases.  Zero-dual SRCs need no label state at all.
        activeSrcCuts = []
        allSrcDuals = getattr(duals, 'src', {})
        for cut, eta in allSrcDuals.items():
            eta = float(eta)
            if eta > self.dualSignTolerance:
                raise RuntimeError(
                    'SRC dual eta_S must be nonpositive for exact pricing, '
                    f'but got eta={eta} for subset {cut.customers}.'
                )
            eta = min(0.0, eta)
            rho = -eta
            if rho > self.dualSignTolerance:
                activeSrcCuts.append((cut, rho))

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
                activeSrcCuts=activeSrcCuts,
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

        # Confirm every discovered sequence with the exact fixed-sequence swap
        # DP. Labeling may discover a non-minimum swap placement; the column
        # entering the master always receives its exact isolated minimum base
        # duration.
        negative = []
        for sequence in discoveredSlotsBySequence:
            if sequence in existingSignatures:
                continue

            column = self.sequenceEvaluator.evaluate(sequence)
            if column is None:
                continue

            dualCoverage = sum(duals.coverage[i] for i in sequence)
            srcReducedCost = 0.0
            for cut, eta in allSrcDuals.items():
                eta = min(0.0, float(eta))
                srcReducedCost += (-eta) * cut.coefficient(column)

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
                    + srcReducedCost
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
            'mode': 'exact_labeling',
            'dominanceMode': self.dominanceMode,
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
            'activeSrcDuals': len(activeSrcCuts),
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
        activeSrcCuts,
    ):
        # Dominance bucket key deliberately excludes visitedMask. Each key owns
        # sub-buckets by popcount so subset/superset comparisons are restricted
        # to cardinalities that can actually dominate one another.
        buckets = {}
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

        requiredSuccessor = {}
        requiredPredecessor = {}
        for u, v in requiredArcs:
            requiredSuccessor[u] = v
            requiredPredecessor[v] = u

        # Exact SRC state for pricing.  For divisor 2, the increment
        # floor((c+1)/2)-floor(c/2) equals one exactly when the old count c is
        # odd.  One parity bit per active nonzero-dual cut is therefore enough.
        srcMembershipByTask = {task: [] for task in self.tasks}
        for srcIndex, (cut, penalty) in enumerate(activeSrcCuts):
            srcBit = 1 << srcIndex
            for task in cut.customers:
                if task in srcMembershipByTask:
                    srcMembershipByTask[task].append((srcBit, float(penalty)))

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
            'activeSrcDuals': len(activeSrcCuts),
        }

        positiveDualByTask = {
            task: max(0.0, float(coverageDuals[task]))
            for task in self.tasks
        }
        totalPositiveDual = sum(positiveDualByTask.values())

        start = _PricingLabel(
            currentNode=self.data.startNode,
            visitedMask=0,
            tasks=(),
            duration=0.0,
            remainingEnergy=float(self.data.Q),
            dualCoverage=0.0,
            partialReducedCost=0.0,
            remainingPositiveDual=totalPositiveDual,
            requiredMask=0,
            srcParityMask=0,
        )
        self._addToEmptyBucket(start, buckets)
        queue.append(start)

        while queue:
            label = queue.popleft()
            if not label.active:
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
            ):
                stats['boundPrunedLabels'] += 1
                continue

            # Complete a nonempty route. Required/forbidden branch arcs are
            # checked before building the physical depot extension.
            if label.visitedMask != 0 and label.currentNode != self.data.startNode:
                if self._completionBranchCompatible(
                    label,
                    forbiddenArcs,
                    requiredSuccessor,
                    requiredPredecessor,
                ):
                    completion = self._completeToDepot(
                        label,
                        requiredArcToBit,
                        beta,
                    )
                    if completion is not None:
                        (
                            completedDuration,
                            completedEnergy,
                            completedRequiredMask,
                            completedPrefixRc,
                        ) = completion
                        del completedEnergy
                        if completedRequiredMask == allRequiredMask:
                            stats['completedRoutes'] += 1
                            rc = completedPrefixRc - slotDual
                            if rc < -self.reducedCostTolerance:
                                stats['negativeCompletions'] += 1
                                if label.tasks not in existingSignatures:
                                    negativeSequences.add(label.tasks)
                else:
                    stats['branchPrunedLabels'] += 1

            current = label.currentNode

            for task in self.tasks:
                bit = self.taskToBit[task]
                if label.visitedMask & bit:
                    continue

                if not self._customerExtensionBranchCompatible(
                    label,
                    task,
                    forbiddenArcs,
                    requiredSuccessor,
                    requiredPredecessor,
                ):
                    stats['branchPrunedLabels'] += 1
                    continue

                nextLabel = self._extendToCustomer(
                    label,
                    task,
                    bit,
                    coverageDuals,
                    requiredArcToBit,
                    beta,
                    positiveDualByTask,
                    srcMembershipByTask,
                )
                if nextLabel is None:
                    continue

                stats['generatedLabels'] += 1

                # Early exact required-arc propagation: if the new state already
                # makes an unsatisfied required arc impossible, never insert it
                # into a dominance bucket.
                if self._requiredArcsAlreadyImpossible(
                    nextLabel,
                    requiredArcs,
                    requiredArcToBit,
                ):
                    stats['branchPrunedLabels'] += 1
                    continue

                if self._insertWithDominance(nextLabel, buckets, queue, stats):
                    stats['acceptedLabels'] += 1

            # Physical BSS visit: only after a customer, never depot->BSS or
            # BSS->BSS. It does not consume/create a compressed branching arc.
            if current in self.taskToBit:
                stationLabel = self._extendToStation(label, beta)
                if stationLabel is not None:
                    stats['generatedLabels'] += 1
                    if self._insertWithDominance(stationLabel, buckets, queue, stats):
                        stats['acceptedLabels'] += 1

        stats['states'] = len(buckets)
        return negativeSequences, stats

    def _sequenceArcToNextTask(self, label, task):
        previous = self.data.startNode if not label.tasks else label.tasks[-1]
        return (previous, task)

    def _customerExtensionBranchCompatible(
        self,
        label,
        task,
        forbiddenArcs,
        requiredSuccessor,
        requiredPredecessor,
    ):
        previous = self.data.startNode if not label.tasks else label.tasks[-1]
        sequenceArc = (previous, task)

        if sequenceArc in forbiddenArcs:
            return False

        requiredNext = requiredSuccessor.get(previous)
        if requiredNext is not None and requiredNext != task:
            return False

        requiredPrevious = requiredPredecessor.get(task)
        if requiredPrevious is not None and requiredPrevious != previous:
            return False

        return True

    def _completionBranchCompatible(
        self,
        label,
        forbiddenArcs,
        requiredSuccessor,
        requiredPredecessor,
    ):
        lastTask = label.tasks[-1]
        sequenceArc = (lastTask, self.data.endNode)
        if sequenceArc in forbiddenArcs:
            return False

        requiredNext = requiredSuccessor.get(lastTask)
        if requiredNext is not None and requiredNext != self.data.endNode:
            return False

        requiredPrevious = requiredPredecessor.get(self.data.endNode)
        if requiredPrevious is not None and requiredPrevious != lastTask:
            return False

        return True

    def _extendToCustomer(
        self,
        label,
        task,
        bit,
        coverageDuals,
        requiredArcToBit,
        beta,
        positiveDualByTask,
        srcMembershipByTask,
    ):
        sequenceArc = self._sequenceArcToNextTask(label, task)
        physicalArc = (label.currentNode, task)
        if physicalArc not in self.data.t or physicalArc not in self.data.e:
            return None

        newEnergy = (
            label.remainingEnergy
            - float(self.data.e[physicalArc])
            - float(self.data.q[task])
        )
        if newEnergy < self.data.QMin - 1e-9:
            return None

        durationIncrement = (
            float(self.data.t[physicalArc])
            + float(self.data.p[task])
        )
        dual = float(coverageDuals[task])

        newSrcParityMask = label.srcParityMask
        srcReducedCostIncrement = 0.0
        for srcBit, penalty in srcMembershipByTask.get(task, ()):
            if newSrcParityMask & srcBit:
                # odd -> even: floor(count/2) increases by one
                srcReducedCostIncrement += penalty
            newSrcParityMask ^= srcBit

        return _PricingLabel(
            currentNode=task,
            visitedMask=label.visitedMask | bit,
            tasks=label.tasks + (task,),
            duration=label.duration + durationIncrement,
            remainingEnergy=newEnergy,
            dualCoverage=label.dualCoverage + dual,
            partialReducedCost=(
                label.partialReducedCost
                + beta * durationIncrement
                - dual
                + srcReducedCostIncrement
            ),
            remainingPositiveDual=(
                label.remainingPositiveDual
                - positiveDualByTask[task]
            ),
            requiredMask=(
                label.requiredMask
                | requiredArcToBit.get(sequenceArc, 0)
            ),
            srcParityMask=newSrcParityMask,
        )

    def _extendToStation(self, label, beta):
        arc = (label.currentNode, self.stationNode)
        if arc not in self.data.t or arc not in self.data.e:
            return None

        arrivalEnergy = label.remainingEnergy - float(self.data.e[arc])
        if arrivalEnergy < self.data.QMin - 1e-9:
            return None

        durationIncrement = (
            float(self.data.t[arc])
            + float(self.data.p[self.stationNode])
        )
        return _PricingLabel(
            currentNode=self.stationNode,
            visitedMask=label.visitedMask,
            tasks=label.tasks,
            duration=label.duration + durationIncrement,
            remainingEnergy=float(self.data.Q),
            dualCoverage=label.dualCoverage,
            partialReducedCost=(
                label.partialReducedCost
                + beta * durationIncrement
            ),
            remainingPositiveDual=label.remainingPositiveDual,
            requiredMask=label.requiredMask,
            srcParityMask=label.srcParityMask,
        )

    def _completeToDepot(self, label, requiredArcToBit, beta):
        lastTask = label.tasks[-1]
        sequenceArc = (lastTask, self.data.endNode)

        physicalArc = (label.currentNode, self.data.endNode)
        if physicalArc not in self.data.t or physicalArc not in self.data.e:
            return None

        newEnergy = label.remainingEnergy - float(self.data.e[physicalArc])
        if newEnergy < self.data.QMin - 1e-9:
            return None

        durationIncrement = float(self.data.t[physicalArc])
        return (
            label.duration + durationIncrement,
            newEnergy,
            label.requiredMask | requiredArcToBit.get(sequenceArc, 0),
            label.partialReducedCost + beta * durationIncrement,
        )

    def _requiredArcsAlreadyImpossible(self, label, requiredArcs, requiredArcToBit):
        if not requiredArcs:
            return False

        visited = label.visitedMask
        lastTask = label.tasks[-1] if label.tasks else None

        for arc in requiredArcs:
            bit = requiredArcToBit[arc]
            if label.requiredMask & bit:
                continue

            u, v = arc
            if u == self.data.startNode:
                # start->v can only be the very first compressed customer arc.
                if label.tasks:
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

            # v cannot have been visited before the required predecessor u->v.
            if visited & vBit:
                return True

            # If u was visited and is no longer the last customer, its required
            # successor can never be created later in an elementary sequence.
            if (visited & uBit) and lastTask != u:
                return True

        return False

    def _dominanceKey(self, label):
        lastSequenceNode = self.data.startNode if not label.tasks else label.tasks[-1]
        return (
            label.currentNode,
            lastSequenceNode,
            label.requiredMask,
            label.srcParityMask,
        )

    def _addToEmptyBucket(self, label, buckets):
        key = self._dominanceKey(label)
        levels = buckets.setdefault(key, {})
        count = label.visitedMask.bit_count()
        levels.setdefault(count, []).append(label)

    def _maskCanDominate(self, maskA, maskB):
        if self.dominanceMode == 'equal':
            return maskA == maskB
        return (maskA & maskB) == maskA

    def _resourceCanDominate(self, labelA, labelB):
        tol = self.dominanceTolerance
        return (
            labelA.remainingEnergy >= labelB.remainingEnergy - tol
            and labelA.partialReducedCost <= labelB.partialReducedCost + tol
        )

    def _insertWithDominance(self, newLabel, buckets, queue, stats):
        key = self._dominanceKey(newLabel)
        levels = buckets.setdefault(key, {})
        newCount = newLabel.visitedMask.bit_count()

        # Existing A can dominate new B only if |V_A| <= |V_B|. In equal mode
        # only the same cardinality/mask is considered.
        if self.dominanceMode == 'equal':
            countsToCheck = (newCount,)
        else:
            countsToCheck = tuple(count for count in levels if count <= newCount)

        for count in countsToCheck:
            for old in levels.get(count, ()):
                if not old.active:
                    continue
                if not self._maskCanDominate(old.visitedMask, newLabel.visitedMask):
                    continue
                if self._resourceCanDominate(old, newLabel):
                    stats['dominatedLabels'] += 1
                    return False

        # New A can dominate existing B only if |V_B| >= |V_A|.
        if self.dominanceMode == 'equal':
            countsToPrune = (newCount,)
        else:
            countsToPrune = tuple(count for count in levels if count >= newCount)

        for count in countsToPrune:
            oldLevel = levels.get(count, [])
            if not oldLevel:
                continue

            survivors = []
            for old in oldLevel:
                if not old.active:
                    continue
                if (
                    self._maskCanDominate(newLabel.visitedMask, old.visitedMask)
                    and self._resourceCanDominate(newLabel, old)
                ):
                    old.active = False
                    stats['removedByDominance'] += 1
                else:
                    survivors.append(old)
            levels[count] = survivors

        levels.setdefault(newCount, []).append(newLabel)
        queue.append(newLabel)
        return True

    def _cannotBecomeNegative(self, label, beta, slotDual):
        del beta  # already embedded in label.partialReducedCost

        # Future travel/service/swap/depot-return costs and future SRC
        # penalties are optimistically zero; every remaining positive customer
        # dual is optimistically collected.  SRC penalties are nonnegative, so
        # ignoring them preserves a valid lower bound on the final reduced cost.
        # remainingPositiveDual is maintained incrementally, so this bound is
        # O(1) per label instead of scanning all customers.
        optimisticRc = (
            label.partialReducedCost
            - label.remainingPositiveDual
            - slotDual
        )
        return optimisticRc >= -self.reducedCostTolerance
