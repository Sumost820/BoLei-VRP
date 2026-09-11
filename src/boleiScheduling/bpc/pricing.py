from .columnManager import ColumnPrefixTrie
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


@dataclass(frozen=True, slots=True)
class _PricingLabel:
    """Four-field mathematical resource label.

    ``unreachableMask`` is the elementary customer-inaccessibility set.  In the
    current exact implementation it is exactly the set of already visited
    customers: once a customer is served it is permanently unreachable.

    Path reconstruction and optional cut/branch automata are deliberately kept
    outside this object in ``_LabelRecord``.  They are implementation/control
    metadata, not base VRP resources.
    """

    currentNode: int
    remainingEnergy: float
    partialReducedCost: float
    unreachableMask: int


@dataclass(slots=True)
class _LabelRecord:
    """Operational wrapper around the four-field mathematical label.

    ``existingPrefixState`` is auxiliary exact-duplicate-exclusion metadata.
    It carries no reduced-cost term and no BSS dual.  The mathematical pricing
    label remains the four fields in ``_PricingLabel``.
    """

    label: _PricingLabel
    predecessor: object = None
    addedNode: object = None
    existingPrefixState: int = -1
    active: bool = True


class _MaskIndexedDominanceBucket:
    """Exact inaccessible-set dominance indexed by an integer bit mask.

    One Pareto front is stored per exact ``unreachableMask``.  Candidate masks
    are filtered by set inclusion before energy/reduced-cost comparisons.  Any
    optional branch/cut state that truly changes future continuations belongs in
    the bucket key or in the protected-prefix compatibility test, never in the
    four-field core label.
    """

    def __init__(self, allTaskMask, mode, tolerance):
        self.allTaskMask = int(allTaskMask)
        self.mode = mode
        self.tolerance = float(tolerance)
        self.fronts = {}
        self.masksByCount = {}

    def _resourceDominates(self, a, b, stats):
        stats['dominanceResourceComparisons'] += 1
        tol = self.tolerance
        # DEAD means this physical prefix can no longer become any column
        # already present in the global pool.  Such a genuinely-new prefix can
        # safely dominate a live existing-column prefix.  Two live prefixes are
        # comparable only when they are the same trie state; otherwise one may
        # terminate at an excluded existing column while the other remains new.
        noveltyCompatible = (
            a.existingPrefixState == ColumnPrefixTrie.DEAD
            or a.existingPrefixState == b.existingPrefixState
        )
        la = a.label
        lb = b.label
        return (
            noveltyCompatible
            and la.remainingEnergy >= lb.remainingEnergy - tol
            and la.partialReducedCost <= lb.partialReducedCost + tol
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

    def insert(self, newRecord, queue, stats):
        newMask = newRecord.label.unreachableMask

        for oldMask in self._candidateSubsetMasks(newMask, stats):
            for old in self.fronts.get(oldMask, ()):
                if old.active and self._resourceDominates(old, newRecord, stats):
                    stats['dominatedLabels'] += 1
                    return False

        supersetMasks = tuple(self._candidateSupersetMasks(newMask, stats))
        for oldMask in supersetMasks:
            oldFront = self.fronts.get(oldMask)
            if not oldFront:
                continue
            survivors = []
            for old in oldFront:
                if not old.active:
                    continue
                if self._resourceDominates(newRecord, old, stats):
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
        self.fronts[newMask].append(newRecord)
        queue.append(newRecord)
        return True

    @property
    def maskCount(self):
        return len(self.fronts)


class ExactEnumerativePricing:
    """Exact small-N oracle over COMPLETE physical route columns.

    Customer orders are enumerated exhaustively and, for each order, every
    feasible BSS placement is enumerated.  The oracle is intentionally only a
    validation/reference implementation for small instances.
    """

    def __init__(
        self,
        modelData,
        routeEvaluator,
        maxEnumeratedTasks=9,
        reducedCostTolerance=1e-8,
    ):
        self.data = modelData
        self.routeEvaluator = routeEvaluator
        self.maxEnumeratedTasks = maxEnumeratedTasks
        self.reducedCostTolerance = reducedCostTolerance
        self.lastStatistics = None

    def price(self, duals, existingSignatures, maxColumns=None, branchingState=None):
        if len(self.data.C) > self.maxEnumeratedTasks:
            raise ValueError(
                f'ExactEnumerativePricing is a small-N reference oracle. '
                f'N={len(self.data.C)} exceeds maxEnumeratedTasks={self.maxEnumeratedTasks}.'
            )

        if hasattr(existingSignatures, 'signatureToIndex'):
            existingSignatureIndex = existingSignatures.signatureToIndex
        else:
            existingSignatureIndex = {tuple(sig): None for sig in existingSignatures}
        negative = []
        evaluated = 0
        branchRejected = 0
        tasks = tuple(self.data.C)

        for length in range(1, len(tasks) + 1):
            for sequence in itertools.permutations(tasks, length):
                for column in self.routeEvaluator.enumerateAllFeasiblePlans(sequence):
                    signature = tuple(column.signature)
                    if signature in existingSignatureIndex:
                        continue
                    compatibleSlots = [
                        k for k in range(self.data.K)
                        if branchingState is None or branchingState.isColumnCompatible(k, column)
                    ]
                    if not compatibleSlots:
                        branchRejected += 1
                        continue

                    evaluated += 1
                    dualCoverage = sum(float(duals.coverage[i]) for i in column.tasks)
                    srcReducedCost = 0.0
                    for cut, eta in getattr(duals, 'src', {}).items():
                        eta = float(eta)
                        if eta > 1e-8:
                            raise RuntimeError(f'SRC dual must be nonpositive, got {eta} for {cut}')
                        srcReducedCost += (-min(0.0, eta)) * cut.coefficient(column)

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
                        negative.append(PricingCandidate(
                            column=column,
                            bestSlot=bestSlot,
                            reducedCost=bestRc,
                            reducedCostBySlot=rcBySlot,
                        ))

        bestBySignature = {}
        for candidate in negative:
            sig = tuple(candidate.column.signature)
            incumbent = bestBySignature.get(sig)
            if incumbent is None or candidate.reducedCost < incumbent.reducedCost:
                bestBySignature[sig] = candidate
        negative = list(bestBySignature.values())
        negative.sort(key=lambda c: (c.reducedCost, c.column.duration, c.column.nodes))
        if maxColumns is not None:
            negative = negative[:maxColumns]

        self.lastStatistics = {
            'mode': 'enumerative_complete_routes',
            'evaluatedFeasibleCompleteRoutes': evaluated,
            'branchRejectedCompleteRoutes': branchRejected,
            'negativeColumnsReturned': len(negative),
            'branchingProjection': 'physical_arcs+customer_precedence_fallback',
        }
        return negative


class ExactLabelingPricing:
    """Exact elementary-customer labeling over complete physical routes.

    The mathematical resource label has exactly four fields:

        (currentNode, remainingEnergy, partialReducedCost, unreachableMask).

    ``unreachableMask`` is the elementary inaccessible-customer set and is
    currently exactly the set of customers already visited by the partial
    route.  The repeatable physical BSS is never inserted into this mask.

    Branch/cut mechanisms are projected onto those four resources whenever
    possible:

    * forbidden/required physical arcs are enforced by local successor/
      predecessor restrictions plus a mandatory-customer completion check;
    * customer precedence is checked from ``unreachableMask`` and mandatory
      progress is derived from that same mask;
    * divisor-2 SRC reduced-cost increments are derived from the parity of
      ``unreachableMask & cutMask`` -- no SRC parity field is stored;
    * BSS combinatorial cuts are master-only rows. Their duals are deliberately
      absent from pricing because every genuinely new route has coefficient 0
      in every previously generated combination-specific BSS cut;
    * exclusion of columns already present in the global pool is handled by an
      incremental prefix trie in ``_LabelRecord``.  This carries no dual and is
      not a mathematical resource.

    With no mandatory-branch progress, exact
    dominance is the classical relation at one current node:

        unreachable(A) subseteq unreachable(B),
        energy(A) >= energy(B),
        reducedCost(A) <= reducedCost(B).
    """

    CORE_LABEL_STATE = (
        'currentNode',
        'remainingEnergy',
        'partialReducedCost',
        'unreachableMask',
    )

    def __init__(
        self,
        modelData,
        routeEvaluator,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
        dualSignTolerance=1e-8,
        dominanceMode='subset',
    ):
        self.data = modelData
        self.routeEvaluator = routeEvaluator
        self.reducedCostTolerance = float(reducedCostTolerance)
        self.dominanceTolerance = float(dominanceTolerance)
        self.dualSignTolerance = float(dualSignTolerance)
        if dominanceMode not in ('subset', 'equal'):
            raise ValueError("dominanceMode must be 'subset' or 'equal'")
        self.dominanceMode = dominanceMode
        self.stationNode = modelData.S[0]
        self.tasks = tuple(modelData.C)
        self.taskToBit = {task: 1 << index for index, task in enumerate(self.tasks)}
        self.allTaskMask = (1 << len(self.tasks)) - 1
        self.lastStatistics = None

    def _normalizeExistingColumns(self, existingSignatures):
        """Return the full duplicate index plus an optional ColumnManager.

        Production BPC passes the global ``ColumnManager``.  For standalone
        callers that only provide signatures we conservatively protect every
        existing signature because we cannot inspect its base reduced cost.
        """
        if hasattr(existingSignatures, 'signatureToIndex'):
            return existingSignatures.signatureToIndex, existingSignatures, None

        index = {tuple(sig): None for sig in existingSignatures}
        trie = ColumnPrefixTrie()
        for signature in index:
            trie.insert(signature)
        return index, None, trie

    def _buildExistingExclusionTrieForSlot(
        self,
        manager,
        fallbackTrie,
        duals,
        slot,
        branchingState,
    ):
        """Protect only existing routes with negative *base* reduced cost.

        BSS-cut duals are intentionally omitted from pricing.  At an optimal
        node RMP, any existing compatible column whose base reduced cost is
        nonnegative cannot invalidate dominance of a genuinely new negative
        route.  The only dangerous existing columns are therefore those whose
        base reduced cost is negative after removing the master-only BSS terms.

        This selection uses only normal pricing duals and the existing columns;
        it does not inspect BSS cuts or their duals.
        """
        if manager is None:
            return fallbackTrie, max(0, len(fallbackTrie.children) - 1)

        trie = ColumnPrefixTrie()
        beta = max(0.0, float(duals.makespan[slot]))
        sigma = float(duals.slot[slot])
        srcDuals = getattr(duals, 'src', {})
        protected = 0

        routeIndices = manager.compatibleIndices(branchingState, slot)
        for routeIndex in routeIndices:
            column = manager.columns[routeIndex]
            if column.isEmpty:
                continue
            dualCoverage = sum(float(duals.coverage[i]) for i in column.tasks)
            srcReducedCost = 0.0
            for cut, eta in srcDuals.items():
                srcReducedCost += (-min(0.0, float(eta))) * cut.coefficient(column)
            baseRc = beta * float(column.duration) - dualCoverage - sigma + srcReducedCost
            if baseRc < -self.reducedCostTolerance:
                trie.insert(column.signature)
                protected += 1
        return trie, protected

    def _activeSrcCuts(self, duals):
        active = []
        for cut, eta in getattr(duals, 'src', {}).items():
            eta = float(eta)
            if eta > self.dualSignTolerance:
                raise RuntimeError(
                    'SRC dual eta_S must be nonpositive for exact pricing, '
                    f'but got eta={eta} for subset {cut.customers}.'
                )
            rho = -min(0.0, eta)
            if rho <= self.dualSignTolerance:
                continue
            cutMask = 0
            for task in cut.customers:
                cutMask |= self.taskToBit.get(task, 0)
            active.append((cut, float(rho), cutMask))
        return active


    def price(self, duals, existingSignatures, maxColumns=None, branchingState=None):
        existingSignatureIndex, existingManager, fallbackExistingTrie = (
            self._normalizeExistingColumns(existingSignatures)
        )

        activeSrcCuts = self._activeSrcCuts(duals)

        # BSS cuts are master-only.  Their duals are not part of route reduced
        # cost because every route not already in the master has coefficient 0
        # in every previously generated combination-specific BSS cut.  The
        # existing-column trie is still required for exact exclusion: an
        # already-present route must not act as a zero-penalty "ghost" label
        # that suppresses a genuinely new route under dominance.

        discoveredSlotsBySignature = {}
        aggregate = LabelingStatistics()
        slotOrder = self._slotSearchOrder(duals, branchingState)
        certifiedSlots = []
        skippedDominatedSlots = {}
        stoppedAfterImprovingSlot = False
        maxExistingPrefixTrieStates = 1
        protectedExistingColumns = 0

        for k in slotOrder:
            certifier = next((
                a for a in certifiedSlots
                if self._slotCanCertify(a, k, duals, branchingState)
            ), None)
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
                    'Makespan dual beta_k must be nonnegative, '
                    f'but slot {k} has beta={beta}.'
                )
            beta = max(0.0, beta)

            existingPrefixTrie, protectedCount = (
                self._buildExistingExclusionTrieForSlot(
                    existingManager,
                    fallbackExistingTrie,
                    duals,
                    k,
                    branchingState,
                )
            )
            protectedExistingColumns += protectedCount
            maxExistingPrefixTrieStates = max(
                maxExistingPrefixTrieStates, existingPrefixTrie.stateCount
            )

            signatures, slotStats = self._priceSlot(
                slot=k,
                beta=beta,
                coverageDuals=duals.coverage,
                slotDual=float(duals.slot[k]),
                existingSignatureIndex=existingSignatureIndex,
                existingPrefixTrie=existingPrefixTrie,
                branchingState=branchingState,
                activeSrcCuts=activeSrcCuts,
                targetNegativeColumns=maxColumns,
            )
            for signature in signatures:
                discoveredSlotsBySignature.setdefault(signature, set()).add(k)

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

            if not slotStats.get('earlyStopped', False):
                certifiedSlots.append(k)
            if signatures and maxColumns is not None:
                stoppedAfterImprovingSlot = True
                break

        allSrcDuals = getattr(duals, 'src', {})
        negative = []
        for signature in discoveredSlotsBySignature:
            if signature in existingSignatureIndex:
                continue
            column = self.routeEvaluator.evaluateNodes(signature)
            if column is None:
                continue

            dualCoverage = sum(float(duals.coverage[i]) for i in column.tasks)
            srcReducedCost = 0.0
            for cut, eta in allSrcDuals.items():
                srcReducedCost += (-min(0.0, float(eta))) * cut.coefficient(column)
            rcBySlot = {}
            for k in range(self.data.K):
                if (
                    branchingState is not None
                    and not branchingState.isColumnCompatible(k, column)
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
                negative.append(PricingCandidate(
                    column=column,
                    bestSlot=bestSlot,
                    reducedCost=bestRc,
                    reducedCostBySlot=rcBySlot,
                ))

        negative.sort(key=lambda c: (c.reducedCost, c.column.duration, c.column.nodes))
        if maxColumns is not None:
            negative = negative[:maxColumns]

        self.lastStatistics = {
            'mode': 'exact_labeling_complete_routes',
            'dominanceMode': self.dominanceMode,
            'coreLabelState': self.CORE_LABEL_STATE,
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
                s.get('earlyStopped', False) for s in aggregate.perSlot.values()
            ),
            'negativeColumnsReturned': len(negative),
            'activeSrcDuals': len(activeSrcCuts),
            'bssCutsInPricing': 0,
            'slotOrder': tuple(slotOrder),
            'pricedSlots': tuple(
                k for k in slotOrder
                if k in aggregate.perSlot
                and not aggregate.perSlot[k].get('skippedByVehicleDominance', False)
            ),
            'skippedDominatedSlots': dict(skippedDominatedSlots),
            'stoppedAfterImprovingSlot': stoppedAfterImprovingSlot,
            'queueMode': 'fifo',
            'columnIdentity': 'complete_physical_nodes',
            'branchingProjection': 'physical_arcs+customer_precedence_fallback',
            'existingPrefixTrieStates': maxExistingPrefixTrieStates,
            'protectedExistingColumns': protectedExistingColumns,
            'perSlot': aggregate.perSlot,
        }
        return negative

    def _slotSearchOrder(self, duals, branchingState):
        def key(k):
            beta = max(0.0, float(duals.makespan[k]))
            sigma = float(duals.slot[k])
            restrictions = 0 if branchingState is None else (
                len(branchingState.requiredArcs(k))
                + len(branchingState.forbiddenArcs(k))
                + len(branchingState.requiredPrecedences(k))
                + len(branchingState.forbiddenPrecedences(k))
            )
            return (beta, -sigma, restrictions, k)
        return sorted(range(self.data.K), key=key)

    def _slotCanCertify(self, certifier, target, duals, branchingState):
        tol = self.dualSignTolerance
        betaA = max(0.0, float(duals.makespan[certifier]))
        betaB = max(0.0, float(duals.makespan[target]))
        sigmaA = float(duals.slot[certifier])
        sigmaB = float(duals.slot[target])
        if betaA > betaB + tol or sigmaA < sigmaB - tol:
            return False
        if branchingState is None:
            return True
        return branchingState.slotFeasibleSetContains(certifier, target)

    def _priceSlot(
        self,
        slot,
        beta,
        coverageDuals,
        slotDual,
        existingSignatureIndex,
        existingPrefixTrie,
        branchingState,
        activeSrcCuts,
        targetNegativeColumns=None,
    ):
        buckets = {}
        queue = deque()
        negativeSignatures = set()

        requiredArcs = ()
        forbiddenArcs = frozenset()
        requiredPrecedences = ()
        forbiddenPrecedences = frozenset()
        if branchingState is not None:
            requiredArcs = tuple(sorted(branchingState.requiredArcs(slot)))
            forbiddenArcs = branchingState.forbiddenArcs(slot)
            requiredPrecedences = tuple(sorted(branchingState.requiredPrecedences(slot)))
            forbiddenPrecedences = branchingState.forbiddenPrecedences(slot)

        # Required physical arcs are enforced without a requiredArcMask.  For
        # non-BSS tails/heads we fix the unique local successor/predecessor.  A
        # required arc is then guaranteed once its customer endpoint(s) have
        # been visited.  S is repeatable, so multiple S-incidence requirements
        # can coexist and are forced through each customer's local incidence.
        requiredOutgoing = {}
        requiredIncoming = {}
        requiredArcTaskMask = 0
        for u, v in requiredArcs:
            if u != self.stationNode:
                requiredOutgoing[u] = v
            if v != self.stationNode:
                requiredIncoming[v] = u
            requiredArcTaskMask |= self.taskToBit.get(u, 0)
            requiredArcTaskMask |= self.taskToBit.get(v, 0)

        requiredPredMaskByTask = {}
        forbiddenPredMaskByTask = {}
        requiredPrecedenceTaskMask = 0
        for i, j in requiredPrecedences:
            iBit = self.taskToBit[i]
            jBit = self.taskToBit[j]
            requiredPredMaskByTask[j] = requiredPredMaskByTask.get(j, 0) | iBit
            requiredPrecedenceTaskMask |= iBit | jBit
        for i, j in forbiddenPrecedences:
            forbiddenPredMaskByTask[j] = (
                forbiddenPredMaskByTask.get(j, 0) | self.taskToBit[i]
            )

        mandatoryTaskMask = requiredArcTaskMask | requiredPrecedenceTaskMask

        # Active divisor-2 SRCs do not require a stored label field.  Their
        # future marginal reduced-cost function depends only on the parity of
        # |unreachableMask ∩ S|.  We derive that parity from the four-field
        # mathematical label and use it only as a dominance bucket discriminator.
        # Without this discriminator two labels with different SRC parity can
        # have different future cut penalties and must not dominate each other.
        srcCutMasks = tuple(cutMask for _, _, cutMask in activeSrcCuts)
        srcParityKeyByMask = {0: 0}

        # Each SRC increment can be derived from the already-visited mask.
        # When adding task j to a divisor-2 cut, the coefficient floor(count/2)
        # increases iff the previous count in that cut was odd.
        srcMembershipByTask = {task: [] for task in self.tasks}
        for _, penalty, cutMask in activeSrcCuts:
            for task in self.tasks:
                bit = self.taskToBit[task]
                if cutMask & bit:
                    srcMembershipByTask[task].append((cutMask, float(penalty)))

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
            'requiredPhysicalArcs': len(requiredArcs),
            'requiredCustomerPrecedences': len(requiredPrecedences),
            'forbiddenCustomerPrecedences': len(forbiddenPrecedences),
            'mandatoryTaskCount': mandatoryTaskMask.bit_count(),
        }

        positiveDualByTask = {
            task: max(0.0, float(coverageDuals[task])) for task in self.tasks
        }
        totalPositiveDual = sum(positiveDualByTask.values())
        positiveVisitedByMask = {0: 0.0}

        startLabel = _PricingLabel(
            currentNode=self.data.startNode,
            remainingEnergy=float(self.data.Q),
            partialReducedCost=0.0,
            unreachableMask=0,
        )
        startRecord = _LabelRecord(
            label=startLabel,
            predecessor=None,
            addedNode=None,
            existingPrefixState=existingPrefixTrie.advance(
                0, self.data.startNode
            ),
        )
        startBucket = self._getBucket(
            startRecord, buckets, mandatoryTaskMask, srcCutMasks, srcParityKeyByMask
        )
        startBucket.fronts[0] = [startRecord]
        startBucket._registerMask(0)
        queue.append(startRecord)

        while queue:
            record = queue.popleft()
            if not record.active:
                continue
            label = record.label

            if self._cannotBecomeNegative(
                label,
                slotDual,
                totalPositiveDual,
                positiveVisitedByMask,
            ):
                stats['boundPrunedLabels'] += 1
                continue

            # Any nonempty partial route may return to the depot, provided all
            # branch-mandatory customers have already been served.
            if label.unreachableMask != 0:
                mandatoryComplete = (
                    label.unreachableMask & mandatoryTaskMask
                ) == mandatoryTaskMask
                if mandatoryComplete and self._extensionBranchCompatible(
                    label,
                    self.data.endNode,
                    forbiddenArcs,
                    requiredOutgoing,
                    requiredIncoming,
                    requiredPredMaskByTask,
                    forbiddenPredMaskByTask,
                ):
                    completedPrefixRc = self._completeToDepot(label, beta)
                    if completedPrefixRc is not None:
                        stats['completedRoutes'] += 1
                        rc = completedPrefixRc - slotDual
                        if rc < -self.reducedCostTolerance:
                            signature = self._reconstructNodes(record) + (
                                self.data.endNode,
                            )
                            if (
                                signature not in existingSignatureIndex
                                and (
                                    branchingState is None
                                    or branchingState.isNodeSequenceCompatible(
                                        slot, signature
                                    )
                                )
                            ):
                                stats['negativeCompletions'] += 1
                                negativeSignatures.add(signature)
                                if (
                                    targetNegativeColumns is not None
                                    and len(negativeSignatures) >= targetNegativeColumns
                                ):
                                    stats['earlyStopped'] = True
                                    self._finalizeStateStats(buckets, stats)
                                    return negativeSignatures, stats
                elif not mandatoryComplete:
                    stats['branchPrunedLabels'] += 1

            # Elementary customer extension.
            for task in self.tasks:
                bit = self.taskToBit[task]
                if label.unreachableMask & bit:
                    continue
                if not self._extensionBranchCompatible(
                    label,
                    task,
                    forbiddenArcs,
                    requiredOutgoing,
                    requiredIncoming,
                    requiredPredMaskByTask,
                    forbiddenPredMaskByTask,
                ):
                    stats['branchPrunedLabels'] += 1
                    continue

                nextRecord = self._extendToCustomer(
                    record,
                    task,
                    bit,
                    coverageDuals,
                    beta,
                    positiveDualByTask,
                    positiveVisitedByMask,
                    srcMembershipByTask,
                    existingPrefixTrie,
                )
                if nextRecord is None:
                    continue
                stats['generatedLabels'] += 1
                if self._cannotBecomeNegative(
                    nextRecord.label,
                    slotDual,
                    totalPositiveDual,
                    positiveVisitedByMask,
                ):
                    stats['boundPrunedLabels'] += 1
                    continue
                bucket = self._getBucket(
                    nextRecord, buckets, mandatoryTaskMask, srcCutMasks, srcParityKeyByMask
                )
                if bucket.insert(nextRecord, queue, stats):
                    stats['acceptedLabels'] += 1

            # S is repeatable but can only be entered from a customer. Entering
            # S resets energy. No station-entry predecessor is a resource.
            if label.currentNode in self.taskToBit:
                if self._extensionBranchCompatible(
                    label,
                    self.stationNode,
                    forbiddenArcs,
                    requiredOutgoing,
                    requiredIncoming,
                    requiredPredMaskByTask,
                    forbiddenPredMaskByTask,
                ):
                    stationRecord = self._extendToStation(
                        record, beta, existingPrefixTrie
                    )
                    if stationRecord is not None:
                        stats['generatedLabels'] += 1
                        if self._cannotBecomeNegative(
                            stationRecord.label,
                            slotDual,
                            totalPositiveDual,
                            positiveVisitedByMask,
                        ):
                            stats['boundPrunedLabels'] += 1
                        else:
                            bucket = self._getBucket(
                                stationRecord,
                                buckets,
                                mandatoryTaskMask,
                                srcCutMasks,
                                srcParityKeyByMask,
                            )
                            if bucket.insert(stationRecord, queue, stats):
                                stats['acceptedLabels'] += 1
                else:
                    stats['branchPrunedLabels'] += 1

        self._finalizeStateStats(buckets, stats)
        return negativeSignatures, stats

    def _finalizeStateStats(self, buckets, stats):
        stats['states'] = len(buckets)
        stats['maskStates'] = sum(bucket.maskCount for bucket in buckets.values())

    def _reconstructNodes(self, record):
        added = []
        cursor = record
        while cursor is not None:
            if cursor.addedNode is not None:
                added.append(cursor.addedNode)
            cursor = cursor.predecessor
        added.reverse()
        return (self.data.startNode,) + tuple(added)

    def _extensionBranchCompatible(
        self,
        label,
        nextNode,
        forbiddenArcs,
        requiredOutgoing,
        requiredIncoming,
        requiredPredMaskByTask,
        forbiddenPredMaskByTask,
    ):
        current = label.currentNode
        arc = (current, nextNode)
        if arc in forbiddenArcs:
            return False

        requiredNext = requiredOutgoing.get(current)
        if requiredNext is not None and requiredNext != nextNode:
            return False
        requiredPrevious = requiredIncoming.get(nextNode)
        if requiredPrevious is not None and requiredPrevious != current:
            return False

        if nextNode in self.taskToBit:
            visited = label.unreachableMask
            requiredPredMask = requiredPredMaskByTask.get(nextNode, 0)
            if requiredPredMask and (visited & requiredPredMask) != requiredPredMask:
                return False
            forbiddenPredMask = forbiddenPredMaskByTask.get(nextNode, 0)
            if visited & forbiddenPredMask:
                return False

        return True

    def _extendToCustomer(
        self,
        record,
        task,
        bit,
        coverageDuals,
        beta,
        positiveDualByTask,
        positiveVisitedByMask,
        srcMembershipByTask,
        existingPrefixTrie,
    ):
        label = record.label
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
        newMask = label.unreachableMask | bit

        srcReducedCostIncrement = 0.0
        for cutMask, penalty in srcMembershipByTask.get(task, ()):
            if (label.unreachableMask & cutMask).bit_count() & 1:
                srcReducedCostIncrement += penalty

        nextLabel = _PricingLabel(
            currentNode=task,
            remainingEnergy=newEnergy,
            partialReducedCost=(
                label.partialReducedCost
                + beta * durationIncrement
                - float(coverageDuals[task])
                + srcReducedCostIncrement
            ),
            unreachableMask=newMask,
        )
        if newMask not in positiveVisitedByMask:
            positiveVisitedByMask[newMask] = (
                positiveVisitedByMask[label.unreachableMask]
                + positiveDualByTask[task]
            )
        return _LabelRecord(
            label=nextLabel,
            predecessor=record,
            addedNode=task,
            existingPrefixState=existingPrefixTrie.advance(
                record.existingPrefixState, task
            ),
        )

    def _extendToStation(self, record, beta, existingPrefixTrie):
        label = record.label
        arc = (label.currentNode, self.stationNode)
        if arc not in self.data.t or arc not in self.data.e:
            return None
        arrivalEnergy = label.remainingEnergy - float(self.data.e[arc])
        if arrivalEnergy < self.data.QMin - 1e-9:
            return None
        durationIncrement = (
            float(self.data.t[arc]) + float(self.data.p[self.stationNode])
        )
        nextLabel = _PricingLabel(
            currentNode=self.stationNode,
            remainingEnergy=float(self.data.Q),
            partialReducedCost=(
                label.partialReducedCost + beta * durationIncrement
            ),
            unreachableMask=label.unreachableMask,
        )
        return _LabelRecord(
            label=nextLabel,
            predecessor=record,
            addedNode=self.stationNode,
            existingPrefixState=existingPrefixTrie.advance(
                record.existingPrefixState, self.stationNode
            ),
        )

    def _completeToDepot(self, label, beta):
        physicalArc = (label.currentNode, self.data.endNode)
        if physicalArc not in self.data.t or physicalArc not in self.data.e:
            return None
        newEnergy = label.remainingEnergy - float(self.data.e[physicalArc])
        if newEnergy < self.data.QMin - 1e-9:
            return None
        durationIncrement = float(self.data.t[physicalArc])
        return label.partialReducedCost + beta * durationIncrement

    @staticmethod
    def _srcParityKey(unreachableMask, srcCutMasks, cache):
        """Return the active-SRC parity vector derived from ``unreachableMask``.

        This is auxiliary dominance bookkeeping, not a mathematical resource
        field.  For divisor-2 SRCs the future marginal cut cost is completely
        determined by this parity vector.
        """
        key = cache.get(unreachableMask)
        if key is not None:
            return key
        key = 0
        for index, cutMask in enumerate(srcCutMasks):
            if (unreachableMask & cutMask).bit_count() & 1:
                key |= 1 << index
        cache[unreachableMask] = key
        return key

    def _dominanceKey(
        self,
        record,
        mandatoryTaskMask=0,
        srcCutMasks=(),
        srcParityKeyByMask=None,
    ):
        label = record.label

        # Required arcs / required customer precedences can make some customers
        # mandatory for this slot.  A label that has already discharged such an
        # obligation does not have the same continuation set as one that has not.
        # The progress is derived from the unreachable set; no extra label field
        # is stored.
        mandatoryProgress = label.unreachableMask & mandatoryTaskMask

        if srcCutMasks:
            if srcParityKeyByMask is None:
                srcParityKeyByMask = {}
            srcParity = self._srcParityKey(
                label.unreachableMask, srcCutMasks, srcParityKeyByMask
            )
        else:
            srcParity = 0

        return (label.currentNode, mandatoryProgress, srcParity)

    def _getBucket(
        self,
        record,
        buckets,
        mandatoryTaskMask=0,
        srcCutMasks=(),
        srcParityKeyByMask=None,
    ):
        key = self._dominanceKey(
            record,
            mandatoryTaskMask,
            srcCutMasks,
            srcParityKeyByMask,
        )
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _MaskIndexedDominanceBucket(
                allTaskMask=self.allTaskMask,
                mode=self.dominanceMode,
                tolerance=self.dominanceTolerance,
            )
            buckets[key] = bucket
        return bucket

    def _cannotBecomeNegative(
        self,
        label,
        slotDual,
        totalPositiveDual,
        positiveVisitedByMask,
    ):
        visitedPositive = positiveVisitedByMask[label.unreachableMask]
        remainingPositiveDual = totalPositiveDual - visitedPositive
        optimisticRc = label.partialReducedCost - remainingPositiveDual - slotDual
        return optimisticRc >= -self.reducedCostTolerance

