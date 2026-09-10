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
    requiredMask: int = 0
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
                rcBySlot = {}
                for k in compatibleSlots:
                    beta = max(0.0, float(duals.makespan[k]))
                    rcBySlot[k] = (
                        beta * column.duration
                        - dualCoverage
                        - float(duals.slot[k])
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
    Exact elementary labeling pricing with slot-specific successor-arc branch
    restrictions.

    Branching is on compressed customer-route arcs. For a sequence
        (i1,...,im)
    the branch arcs are
        (start,i1), (i1,i2), ..., (im,end),
    irrespective of whether an optimal swap is inserted between two consecutive
    customers. This matches the current column identity (ordered customer
    sequence), while swap placement remains delegated to the exact fixed-
    sequence evaluator / BSS subproblem.

    Required-arc usage is included in the label state. This is essential for
    exact dominance: two labels with the same visited set and physical current
    node are not comparable if they have satisfied different required branch
    arcs. At the BSS node, the last customer is also part of the dominance state
    because it determines the next compressed successor arc.
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

        # Re-evaluate every discovered sequence with the exact fixed-sequence
        # DP. A sequence that was negative for one slot may also be negative for
        # another branch-compatible slot after the exact DP improves its base
        # duration, so evaluate every compatible slot here.
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
            'mode': 'exact_labeling',
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

        start = _PricingLabel(
            currentNode=self.data.startNode,
            visitedMask=0,
            tasks=(),
            duration=0.0,
            remainingEnergy=float(self.data.Q),
            dualCoverage=0.0,
            requiredMask=0,
        )
        buckets[self._dominanceKey(start)] = [start]
        queue.append(start)

        positiveDualByBit = {
            self.taskToBit[task]: max(0.0, float(coverageDuals[task]))
            for task in self.tasks
        }

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
                positiveDualByBit=positiveDualByBit,
            ):
                stats['boundPrunedLabels'] += 1
                continue

            # Complete current nonempty route by returning to depot. Completion
            # also creates the compressed successor arc (lastTask,end).
            if label.visitedMask != 0 and label.currentNode != self.data.startNode:
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
                            - label.dualCoverage
                            - slotDual
                        )
                        if rc < -self.reducedCostTolerance:
                            stats['negativeCompletions'] += 1
                            if label.tasks not in existingSignatures:
                                negativeSequences.add(label.tasks)

            current = label.currentNode

            for task in self.tasks:
                bit = self.taskToBit[task]
                if label.visitedMask & bit:
                    continue

                nextLabel = self._extendToCustomer(
                    label,
                    task,
                    bit,
                    coverageDuals,
                    forbiddenArcs,
                    requiredArcToBit,
                )
                if nextLabel is None:
                    continue

                stats['generatedLabels'] += 1
                if self._insertWithDominance(nextLabel, buckets, queue, stats):
                    stats['acceptedLabels'] += 1

            # Physical BSS visit: only after a customer, never depot->BSS or
            # BSS->BSS. It does NOT change compressed successor-arc state.
            if current in self.taskToBit:
                stationLabel = self._extendToStation(label)
                if stationLabel is not None:
                    stats['generatedLabels'] += 1
                    if self._insertWithDominance(stationLabel, buckets, queue, stats):
                        stats['acceptedLabels'] += 1

        stats['states'] = len(buckets)
        return negativeSequences, stats

    def _sequenceArcToNextTask(self, label, task):
        previous = self.data.startNode if not label.tasks else label.tasks[-1]
        return (previous, task)

    def _extendToCustomer(
        self,
        label,
        task,
        bit,
        coverageDuals,
        forbiddenArcs,
        requiredArcToBit,
    ):
        sequenceArc = self._sequenceArcToNextTask(label, task)
        if sequenceArc in forbiddenArcs:
            return None

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

        newDuration = (
            label.duration
            + float(self.data.t[physicalArc])
            + float(self.data.p[task])
        )

        return _PricingLabel(
            currentNode=task,
            visitedMask=label.visitedMask | bit,
            tasks=label.tasks + (task,),
            duration=newDuration,
            remainingEnergy=newEnergy,
            dualCoverage=(
                label.dualCoverage
                + float(coverageDuals[task])
            ),
            requiredMask=(
                label.requiredMask
                | requiredArcToBit.get(sequenceArc, 0)
            ),
        )

    def _extendToStation(self, label):
        arc = (label.currentNode, self.stationNode)
        if arc not in self.data.t or arc not in self.data.e:
            return None

        arrivalEnergy = label.remainingEnergy - float(self.data.e[arc])
        if arrivalEnergy < self.data.QMin - 1e-9:
            return None

        return _PricingLabel(
            currentNode=self.stationNode,
            visitedMask=label.visitedMask,
            tasks=label.tasks,
            duration=(
                label.duration
                + float(self.data.t[arc])
                + float(self.data.p[self.stationNode])
            ),
            remainingEnergy=float(self.data.Q),
            dualCoverage=label.dualCoverage,
            requiredMask=label.requiredMask,
        )

    def _completeToDepot(self, label, forbiddenArcs, requiredArcToBit):
        lastTask = label.tasks[-1]
        sequenceArc = (lastTask, self.data.endNode)
        if sequenceArc in forbiddenArcs:
            return None

        physicalArc = (label.currentNode, self.data.endNode)
        if physicalArc not in self.data.t or physicalArc not in self.data.e:
            return None

        newEnergy = label.remainingEnergy - float(self.data.e[physicalArc])
        if newEnergy < self.data.QMin - 1e-9:
            return None

        return (
            label.duration + float(self.data.t[physicalArc]),
            newEnergy,
            label.requiredMask | requiredArcToBit.get(sequenceArc, 0),
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
                # Required start->v can only be the first task.
                if label.tasks and label.tasks[0] != v:
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

            # If v was already visited before using u->v, the arc can never be
            # created later in an elementary route.
            if visited & vBit:
                return True

            # Once u has been left in the compressed sequence, u->v is also
            # impossible. If the label is at the BSS after u, lastTask==u and
            # the arc is still available on the next customer extension.
            if (visited & uBit) and lastTask != u:
                return True

        return False

    def _dominanceKey(self, label):
        lastSequenceNode = self.data.startNode if not label.tasks else label.tasks[-1]
        return (
            label.currentNode,
            lastSequenceNode,
            label.visitedMask,
            label.requiredMask,
        )

    def _insertWithDominance(self, newLabel, buckets, queue, stats):
        key = self._dominanceKey(newLabel)
        bucket = buckets.setdefault(key, [])
        tol = self.dominanceTolerance

        for old in bucket:
            if not old.active:
                continue
            if (
                old.duration <= newLabel.duration + tol
                and old.remainingEnergy >= newLabel.remainingEnergy - tol
            ):
                stats['dominatedLabels'] += 1
                return False

        survivors = []
        for old in bucket:
            if not old.active:
                continue
            if (
                newLabel.duration <= old.duration + tol
                and newLabel.remainingEnergy >= old.remainingEnergy - tol
            ):
                old.active = False
                stats['removedByDominance'] += 1
            else:
                survivors.append(old)

        survivors.append(newLabel)
        buckets[key] = survivors
        queue.append(newLabel)
        return True

    def _cannotBecomeNegative(self, label, beta, slotDual, positiveDualByBit):
        remainingPositiveDual = 0.0
        for bit, value in positiveDualByBit.items():
            if not (label.visitedMask & bit):
                remainingPositiveDual += value

        optimisticRc = (
            beta * label.duration
            - label.dualCoverage
            - remainingPositiveDual
            - slotDual
        )
        return optimisticRc >= -self.reducedCostTolerance
