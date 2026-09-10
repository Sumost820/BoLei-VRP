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
    maskStates: int = 0
    dominanceMaskLookups: int = 0
    dominanceResourceComparisons: int = 0
    perSlot: dict = field(default_factory=dict)


@dataclass(slots=True)
class _PricingLabel:
    currentNode: int
    visitedMask: int
    lastTask: object
    duration: float
    remainingEnergy: float
    partialReducedCost: float = 0.0
    remainingPositiveDual: float = 0.0
    requiredMask: int = 0
    srcParityMask: int = 0
    predecessor: object = None
    addedTask: object = None
    active: bool = True


class _MaskIndexedDominanceBucket:
    """Exact visited-set dominance indexed by bit mask.

    One Pareto front is stored per exact visited mask.  Candidate masks are
    organized by cardinality, so subset dominance first filters at the *mask*
    level and only then performs energy/reduced-cost comparisons on the Pareto
    labels attached to compatible masks.  For dense buckets the code can also
    enumerate actual submasks/supersets and do O(1) dictionary lookups.
    """

    def __init__(self, allTaskMask, mode, tolerance):
        self.allTaskMask = int(allTaskMask)
        self.mode = mode
        self.tolerance = float(tolerance)
        self.fronts = {}              # visitedMask -> [active Pareto labels]
        self.masksByCount = {}        # popcount -> set(visitedMask)

    def _resourceDominates(self, a, b, stats):
        stats['dominanceResourceComparisons'] += 1
        tol = self.tolerance
        return (
            a.remainingEnergy >= b.remainingEnergy - tol
            and a.partialReducedCost <= b.partialReducedCost + tol
        )

    @staticmethod
    def _submasks(mask):
        sub = mask
        while True:
            yield sub
            if sub == 0:
                break
            sub = (sub - 1) & mask

    def _registerMask(self, mask):
        self.masksByCount.setdefault(mask.bit_count(), set()).add(mask)

    def _removeMask(self, mask):
        count = mask.bit_count()
        masks = self.masksByCount.get(count)
        if masks is None:
            return
        masks.discard(mask)
        if not masks:
            self.masksByCount.pop(count, None)

    def _candidateSubsetMasks(self, mask, stats):
        if self.mode == 'equal':
            stats['dominanceMaskLookups'] += 1
            if mask in self.fronts:
                yield mask
            return

        count = mask.bit_count()
        presentCandidateCount = sum(
            len(masks)
            for cardinality, masks in self.masksByCount.items()
            if cardinality <= count
        )
        enumerationCount = 1 << count

        if enumerationCount < presentCandidateCount:
            for sub in self._submasks(mask):
                stats['dominanceMaskLookups'] += 1
                if sub in self.fronts:
                    yield sub
            return

        for cardinality, masks in self.masksByCount.items():
            if cardinality > count:
                continue
            for oldMask in tuple(masks):
                stats['dominanceMaskLookups'] += 1
                if (oldMask & mask) == oldMask:
                    yield oldMask

    def _candidateSupersetMasks(self, mask, stats):
        if self.mode == 'equal':
            stats['dominanceMaskLookups'] += 1
            if mask in self.fronts:
                yield mask
            return

        count = mask.bit_count()
        presentCandidateCount = sum(
            len(masks)
            for cardinality, masks in self.masksByCount.items()
            if cardinality >= count
        )
        complement = self.allTaskMask ^ mask
        enumerationCount = 1 << complement.bit_count()

        if enumerationCount < presentCandidateCount:
            for extra in self._submasks(complement):
                sup = mask | extra
                stats['dominanceMaskLookups'] += 1
                if sup in self.fronts:
                    yield sup
            return

        for cardinality, masks in self.masksByCount.items():
            if cardinality < count:
                continue
            for oldMask in tuple(masks):
                stats['dominanceMaskLookups'] += 1
                if (mask & oldMask) == mask:
                    yield oldMask

    def insert(self, newLabel, queue, stats):
        newMask = newLabel.visitedMask

        for oldMask in self._candidateSubsetMasks(newMask, stats):
            for old in self.fronts.get(oldMask, ()):
                if old.active and self._resourceDominates(old, newLabel, stats):
                    stats['dominatedLabels'] += 1
                    return False

        # Materialize the mask list because fronts/masksByCount can change while
        # dominated fronts are removed.
        supersetMasks = tuple(self._candidateSupersetMasks(newMask, stats))
        for oldMask in supersetMasks:
            oldFront = self.fronts.get(oldMask)
            if not oldFront:
                continue
            survivors = []
            for old in oldFront:
                if not old.active:
                    continue
                if self._resourceDominates(newLabel, old, stats):
                    old.active = False
                    stats['removedByDominance'] += 1
                else:
                    survivors.append(old)
            if survivors:
                self.fronts[oldMask] = survivors
            else:
                self.fronts.pop(oldMask, None)
                self._removeMask(oldMask)

        if newMask not in self.fronts:
            self._registerMask(newMask)
            self.fronts[newMask] = []
        self.fronts[newMask].append(newLabel)
        queue.append(newLabel)
        return True

    @property
    def maskCount(self):
        return len(self.fronts)


class ExactEnumerativePricing:
    """Exhaustive small-instance pricing oracle, branch- and SRC-aware."""

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
    """Exact elementary pricing with mask-indexed set-inclusion dominance.

    For labels sharing current physical node, compressed last customer,
    required-arc state and active-SRC parity state, A dominates B when

        visited(A) subseteq visited(B),
        energy(A) >= energy(B),
        partialRC(A) <= partialRC(B).

    The implementation is exact.  The only change from the previous strong
    dominance rule is the indexing data structure used to find comparable
    visited masks efficiently.
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
        self.allTaskMask = (1 << len(self.tasks)) - 1
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

        # Price vehicle slots in a deterministic promising order, but do not
        # solve every slot eagerly.  Route columns are global: once any slot
        # produces an improving route, adding that route creates x[k,r] for all
        # compatible slots in the master, so it is safe (and much cheaper) to
        # re-optimize the RMP immediately and postpone the remaining slots to
        # the next CG round.
        #
        # A fully priced slot a can also certify another slot b without solving
        # b when:
        #   F_b subseteq F_a, beta_a <= beta_b, sigma_a >= sigma_b.
        # Then rc_a(r) <= rc_b(r) for every route feasible for b.  Hence if a
        # has been exhaustively priced, b cannot contain an improving route not
        # already exposed by a.
        slotOrder = self._slotSearchOrder(duals, branchingState)
        certifiedSlots = []
        skippedDominatedSlots = {}
        stoppedAfterImprovingSlot = False

        for k in slotOrder:
            certifier = next(
                (
                    a for a in certifiedSlots
                    if self._slotCanCertify(
                        certifier=a,
                        target=k,
                        duals=duals,
                        branchingState=branchingState,
                    )
                ),
                None,
            )
            if certifier is not None:
                skippedDominatedSlots[k] = certifier
                aggregate.perSlot[k] = {
                    'skippedByVehicleDominance': True,
                    'certifiedBySlot': certifier,
                    'earlyStopped': False,
                }
                continue

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
                targetNegativeColumns=maxColumns,
            )
            for sequence in sequences:
                discoveredSlotsBySequence.setdefault(sequence, set()).add(k)

            aggregate.perSlot[k] = slotStats
            for fieldName in (
                'generatedLabels', 'acceptedLabels', 'dominatedLabels',
                'removedByDominance', 'boundPrunedLabels', 'branchPrunedLabels',
                'completedRoutes', 'negativeCompletions', 'states', 'maskStates',
                'dominanceMaskLookups', 'dominanceResourceComparisons',
            ):
                setattr(
                    aggregate,
                    fieldName,
                    getattr(aggregate, fieldName) + int(slotStats.get(fieldName, 0)),
                )

            # Only an exhaustive slot solve can certify other slots.  If the
            # per-slot search hit its batch limit, it deliberately stopped early.
            if not slotStats.get('earlyStopped', False):
                certifiedSlots.append(k)

            # In ordinary batched column generation, one improving slot is
            # enough for this round.  Re-solve the persistent RMP now rather
            # than paying for K independent pricing runs under stale duals.
            if sequences and maxColumns is not None:
                stoppedAfterImprovingSlot = True
                break

        # Labeling can terminate early after collecting a batch of improving
        # sequences.  Each sequence is then confirmed with the exact fixed-order
        # swap optimizer, so the master always receives the true minimum base
        # duration for that customer order.
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
            'maskStates': aggregate.maskStates,
            'dominanceMaskLookups': aggregate.dominanceMaskLookups,
            'dominanceResourceComparisons': aggregate.dominanceResourceComparisons,
            'earlyStopped': any(
                slot.get('earlyStopped', False)
                for slot in aggregate.perSlot.values()
            ),
            'negativeColumnsReturned': len(negative),
            'activeSrcDuals': len(activeSrcCuts),
            'slotOrder': tuple(slotOrder),
            'pricedSlots': tuple(
                k for k in slotOrder if k in aggregate.perSlot
                and not aggregate.perSlot[k].get('skippedByVehicleDominance', False)
            ),
            'skippedDominatedSlots': dict(skippedDominatedSlots),
            'stoppedAfterImprovingSlot': stoppedAfterImprovingSlot,
            'queueMode': 'fifo',
            'perSlot': aggregate.perSlot,
        }
        return negative

    def _slotSearchOrder(self, duals, branchingState):
        """Heuristic order only; it never removes a slot by itself.

        For d >= 0, smaller beta and larger sigma make
            beta*d - sigma
        more favorable.  Less-restricted slots are used as a final tie-break
        because they are more likely to expose globally reusable route columns.
        """
        def key(k):
            beta = max(0.0, float(duals.makespan[k]))
            sigma = float(duals.slot[k])
            if branchingState is None:
                restrictions = 0
            else:
                restrictions = (
                    len(branchingState.requiredArcs(k))
                    + len(branchingState.forbiddenArcs(k))
                )
            return (beta, -sigma, restrictions, k)

        return sorted(range(self.data.K), key=key)

    def _slotCanCertify(self, certifier, target, duals, branchingState):
        """Return True iff an exhaustively priced slot certifies `target`.

        Let F_k be the set of branch-compatible routes of slot k.  If
            F_target subseteq F_certifier,
            beta_certifier <= beta_target,
            sigma_certifier >= sigma_target,
        then rc_certifier(r) <= rc_target(r) for every r in F_target because
        coverage and SRC contributions are slot independent and d_r >= 0.
        Thus exhaustive pricing of `certifier` makes pricing `target` redundant.
        """
        tol = self.dualSignTolerance
        betaA = max(0.0, float(duals.makespan[certifier]))
        betaB = max(0.0, float(duals.makespan[target]))
        sigmaA = float(duals.slot[certifier])
        sigmaB = float(duals.slot[target])
        if betaA > betaB + tol or sigmaA < sigmaB - tol:
            return False

        if branchingState is None:
            return True

        requiredA = branchingState.requiredArcs(certifier)
        requiredB = branchingState.requiredArcs(target)
        forbiddenA = branchingState.forbiddenArcs(certifier)
        forbiddenB = branchingState.forbiddenArcs(target)

        # F_B subseteq F_A when A requires no more arcs than B and A forbids
        # no more arcs than B.
        return (
            requiredA.issubset(requiredB)
            and forbiddenA.issubset(forbiddenB)
        )

    def _priceSlot(
        self,
        slot,
        beta,
        coverageDuals,
        slotDual,
        existingSignatures,
        branchingState,
        activeSrcCuts,
        targetNegativeColumns=None,
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

        requiredSuccessor = {}
        requiredPredecessor = {}
        for u, v in requiredArcs:
            requiredSuccessor[u] = v
            requiredPredecessor[v] = u

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
            'maskStates': 0,
            'dominanceMaskLookups': 0,
            'dominanceResourceComparisons': 0,
            'earlyStopped': False,
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
            lastTask=None,
            duration=0.0,
            remainingEnergy=float(self.data.Q),
            partialReducedCost=0.0,
            remainingPositiveDual=totalPositiveDual,
            requiredMask=0,
            srcParityMask=0,
            predecessor=None,
            addedTask=None,
        )
        startBucket = self._getBucket(start, buckets)
        startBucket.fronts[0] = [start]
        startBucket._registerMask(0)
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

            if self._cannotBecomeNegative(label=label, slotDual=slotDual):
                stats['boundPrunedLabels'] += 1
                continue

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
                        completedRequiredMask, completedPrefixRc = completion
                        if completedRequiredMask == allRequiredMask:
                            stats['completedRoutes'] += 1
                            rc = completedPrefixRc - slotDual
                            if rc < -self.reducedCostTolerance:
                                stats['negativeCompletions'] += 1
                                sequence = self._reconstructSequence(label)
                                if sequence not in existingSignatures:
                                    negativeSequences.add(sequence)
                                    if (
                                        targetNegativeColumns is not None
                                        and len(negativeSequences) >= targetNegativeColumns
                                    ):
                                        stats['earlyStopped'] = True
                                        self._finalizeStateStats(buckets, stats)
                                        return negativeSequences, stats
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
                if self._requiredArcsAlreadyImpossible(
                    nextLabel,
                    requiredArcs,
                    requiredArcToBit,
                ):
                    stats['branchPrunedLabels'] += 1
                    continue
                if self._cannotBecomeNegative(nextLabel, slotDual):
                    stats['boundPrunedLabels'] += 1
                    continue

                bucket = self._getBucket(nextLabel, buckets)
                if bucket.insert(nextLabel, queue, stats):
                    stats['acceptedLabels'] += 1

            # Physical BSS visit: it does not create a compressed branching arc.
            if current in self.taskToBit:
                stationLabel = self._extendToStation(label, beta)
                if stationLabel is not None:
                    stats['generatedLabels'] += 1
                    if self._cannotBecomeNegative(stationLabel, slotDual):
                        stats['boundPrunedLabels'] += 1
                        continue
                    bucket = self._getBucket(stationLabel, buckets)
                    if bucket.insert(stationLabel, queue, stats):
                        stats['acceptedLabels'] += 1

        self._finalizeStateStats(buckets, stats)
        return negativeSequences, stats

    def _finalizeStateStats(self, buckets, stats):
        stats['states'] = len(buckets)
        stats['maskStates'] = sum(bucket.maskCount for bucket in buckets.values())

    def _reconstructSequence(self, label):
        tasks = []
        cursor = label
        while cursor is not None:
            if cursor.addedTask is not None:
                tasks.append(cursor.addedTask)
            cursor = cursor.predecessor
        tasks.reverse()
        return tuple(tasks)

    def _sequenceArcToNextTask(self, label, task):
        previous = self.data.startNode if label.lastTask is None else label.lastTask
        return (previous, task)

    def _customerExtensionBranchCompatible(
        self,
        label,
        task,
        forbiddenArcs,
        requiredSuccessor,
        requiredPredecessor,
    ):
        previous = self.data.startNode if label.lastTask is None else label.lastTask
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
        lastTask = label.lastTask
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

        durationIncrement = float(self.data.t[physicalArc]) + float(self.data.p[task])
        dual = float(coverageDuals[task])

        newSrcParityMask = label.srcParityMask
        srcReducedCostIncrement = 0.0
        for srcBit, penalty in srcMembershipByTask.get(task, ()):
            if newSrcParityMask & srcBit:
                srcReducedCostIncrement += penalty
            newSrcParityMask ^= srcBit

        return _PricingLabel(
            currentNode=task,
            visitedMask=label.visitedMask | bit,
            lastTask=task,
            duration=label.duration + durationIncrement,
            remainingEnergy=newEnergy,
            partialReducedCost=(
                label.partialReducedCost
                + beta * durationIncrement
                - dual
                + srcReducedCostIncrement
            ),
            remainingPositiveDual=(
                label.remainingPositiveDual - positiveDualByTask[task]
            ),
            requiredMask=(
                label.requiredMask | requiredArcToBit.get(sequenceArc, 0)
            ),
            srcParityMask=newSrcParityMask,
            predecessor=label,
            addedTask=task,
        )

    def _extendToStation(self, label, beta):
        arc = (label.currentNode, self.stationNode)
        if arc not in self.data.t or arc not in self.data.e:
            return None

        arrivalEnergy = label.remainingEnergy - float(self.data.e[arc])
        if arrivalEnergy < self.data.QMin - 1e-9:
            return None

        durationIncrement = (
            float(self.data.t[arc]) + float(self.data.p[self.stationNode])
        )
        return _PricingLabel(
            currentNode=self.stationNode,
            visitedMask=label.visitedMask,
            lastTask=label.lastTask,
            duration=label.duration + durationIncrement,
            remainingEnergy=float(self.data.Q),
            partialReducedCost=(
                label.partialReducedCost + beta * durationIncrement
            ),
            remainingPositiveDual=label.remainingPositiveDual,
            requiredMask=label.requiredMask,
            srcParityMask=label.srcParityMask,
            predecessor=label,
            addedTask=None,
        )

    def _completeToDepot(self, label, requiredArcToBit, beta):
        lastTask = label.lastTask
        sequenceArc = (lastTask, self.data.endNode)
        physicalArc = (label.currentNode, self.data.endNode)
        if physicalArc not in self.data.t or physicalArc not in self.data.e:
            return None

        newEnergy = label.remainingEnergy - float(self.data.e[physicalArc])
        if newEnergy < self.data.QMin - 1e-9:
            return None

        durationIncrement = float(self.data.t[physicalArc])
        return (
            label.requiredMask | requiredArcToBit.get(sequenceArc, 0),
            label.partialReducedCost + beta * durationIncrement,
        )

    def _requiredArcsAlreadyImpossible(self, label, requiredArcs, requiredArcToBit):
        if not requiredArcs:
            return False

        visited = label.visitedMask
        lastTask = label.lastTask
        hasTasks = visited != 0

        for arc in requiredArcs:
            bit = requiredArcToBit[arc]
            if label.requiredMask & bit:
                continue

            u, v = arc
            if u == self.data.startNode:
                if hasTasks:
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

    def _dominanceKey(self, label):
        lastSequenceNode = (
            self.data.startNode if label.lastTask is None else label.lastTask
        )
        return (
            label.currentNode,
            lastSequenceNode,
            label.requiredMask,
            label.srcParityMask,
        )

    def _getBucket(self, label, buckets):
        key = self._dominanceKey(label)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _MaskIndexedDominanceBucket(
                allTaskMask=self.allTaskMask,
                mode=self.dominanceMode,
                tolerance=self.dominanceTolerance,
            )
            buckets[key] = bucket
        return bucket

    def _cannotBecomeNegative(self, label, slotDual):
        # Future travel/service/swap/depot-return costs and future SRC penalties
        # are optimistically zero; every remaining positive customer dual is
        # optimistically collected.  This is therefore a valid lower bound.
        optimisticRc = (
            label.partialReducedCost
            - label.remainingPositiveDual
            - slotDual
        )
        return optimisticRc >= -self.reducedCostTolerance
