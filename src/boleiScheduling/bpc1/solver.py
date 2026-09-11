import heapq
import math
import time
from dataclasses import dataclass, field

from .master import RestrictedMasterProblem, PersistentRestrictedMasterProblem
from .pricing import ExactEnumerativePricing, ExactLabelingPricing
from .swap_dp import ExactCompleteRouteEvaluator
from .bss import ExactBssScheduler, BssScheduleResult
from .cuts import BssOptimalityCut, ThreeRowSubsetCutSeparator
from .branch import (
    ArcBranchingState,
    ArcBranchDecision,
    PrecedenceBranchDecision,
    ArcFlowBrancher,
)
from .savings import SavingsWarmStart
from .columnManager import ColumnManager
from .profiler import PerformanceProfiler
from .cache import collectCacheStatistics


def _format_number(value, digits=3):
    if value is None:
        return '-'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return 'nan'
    if math.isinf(value):
        return 'INF' if value > 0 else '-INF'
    return f'{value:.{digits}f}'


def _format_gap(lowerBound, upperBound, digits=2):
    if (
        lowerBound is None
        or upperBound is None
        or not math.isfinite(float(lowerBound))
        or not math.isfinite(float(upperBound))
    ):
        return '-'
    denominator = max(1.0, abs(float(upperBound)))
    gap = max(0.0, float(upperBound) - float(lowerBound)) / denominator
    return f'{100.0 * gap:.{digits}f}%'


class RootColumnGenerationSolver:
    """Exact root LP column generation for the base routing master."""

    def __init__(
        self,
        modelData,
        pricingMode='labeling',
        maxEnumeratedTasks=9,
        maxColumnsPerRound=50,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
        phaseOneTolerance=1e-8,
        bssCuts=None,
        srcCuts=None,
        dominanceMode='subset',
        useSavingsWarmStart=True,
        savingsStarts=12,
        savingsSeed=1,
        savingsRandomization=0.20,
        savingsExcessRoutePenalty=1.0e9,
        maxWarmStartColumns=1000,
        preGenerateShortRoutes=2,
        shortRouteTaskThreshold=15,
        maxShortRouteColumns=1000,
        printMasterDuals=False,
    ):
        self.data = modelData
        self.profiler = PerformanceProfiler()
        self.routeEvaluator = ExactCompleteRouteEvaluator(modelData, profiler=self.profiler)
        emptyColumn = self.routeEvaluator.evaluate(())
        self.columnManager = ColumnManager([emptyColumn])
        self.columns = self.columnManager.columns
        self.signatureToIndex = self.columnManager.signatureToIndex
        self.pricingMode = pricingMode

        if pricingMode == 'labeling':
            self.pricing = ExactLabelingPricing(
                modelData=modelData,
                routeEvaluator=self.routeEvaluator,
                reducedCostTolerance=reducedCostTolerance,
                dominanceTolerance=dominanceTolerance,
                dominanceMode=dominanceMode,
            )
        elif pricingMode == 'enumerative':
            self.pricing = ExactEnumerativePricing(
                modelData=modelData,
                routeEvaluator=self.routeEvaluator,
                maxEnumeratedTasks=maxEnumeratedTasks,
                reducedCostTolerance=reducedCostTolerance,
            )
        else:
            raise ValueError("pricingMode must be 'labeling' or 'enumerative'")

        self.maxColumnsPerRound = maxColumnsPerRound
        self.reducedCostTolerance = reducedCostTolerance
        self.phaseOneTolerance = phaseOneTolerance
        self.bssCuts = list(bssCuts or [])
        self.srcCuts = list(srcCuts or [])

        self.dominanceMode = dominanceMode
        self.useSavingsWarmStart = bool(useSavingsWarmStart)
        self.savingsStarts = max(1, int(savingsStarts))
        self.savingsSeed = int(savingsSeed)
        self.savingsRandomization = max(0.0, float(savingsRandomization))
        self.savingsExcessRoutePenalty = float(savingsExcessRoutePenalty)
        self.maxWarmStartColumns = maxWarmStartColumns
        self.preGenerateShortRoutes = max(0, int(preGenerateShortRoutes))
        self.shortRouteTaskThreshold = max(0, int(shortRouteTaskThreshold))
        self.maxShortRouteColumns = maxShortRouteColumns
        self.printMasterDuals = bool(printMasterDuals)
        self.shortRouteColumnsAdded = 0
        self.shortRouteBuildTime = 0.0
        self.shortRouteInitialized = False
        self.warmStartInitialized = False
        self.warmStartResult = None
        self.warmStartBuildTime = 0.0
        self.warmStartColumnsAdded = 0

        self.master = None
        self.phaseOneIterations = 0
        self.phaseTwoIterations = 0
        self.lowerBound = float('inf')
        self.status = 'NOT_SOLVED'

    def _existingSignatures(self):
        # Pass the manager itself so pricing sees the global signature index
        # without copying the full set every column-generation round.
        return self.columnManager

    def _addPricingCandidates(self, candidates):
        addedColumns = self.columnManager.addCandidates(candidates)
        if addedColumns:
            self.profiler.increment('globalColumnsAdded', len(addedColumns))
        return addedColumns

    def _initializeWarmStart(self, outputFlag=0):
        if self.warmStartInitialized:
            return self.warmStartResult
        self.warmStartInitialized = True

        if not self.useSavingsWarmStart:
            return None

        builder = SavingsWarmStart(
            modelData=self.data,
            routeEvaluator=self.routeEvaluator,
            starts=self.savingsStarts,
            seed=self.savingsSeed,
            randomization=self.savingsRandomization,
            excessRoutePenalty=self.savingsExcessRoutePenalty,
            maxInitialColumns=self.maxWarmStartColumns,
        )
        result = builder.build()
        addedColumns = self.columnManager.addColumns(result.columns)
        added = len(addedColumns)
        if added:
            self.profiler.increment('globalColumnsAdded', added)

        self.warmStartResult = result
        self.warmStartBuildTime = result.buildTime
        self.warmStartColumnsAdded = added
        self.profiler.addTime('savingsWarmStart', result.buildTime)
        self.profiler.increment('warmStartColumnsAdded', added)

        if outputFlag:
            bestRouteCount = len(result.bestRoutes) if result.bestRoutes else 0
            print(
                f'[WARM START] directed savings | starts={result.startsAttempted} | '
                f'feasibleStarts={result.feasibleStarts} | '
                f'bestRoutes={bestRouteCount} | +{added} initial columns | '
                f'time={result.buildTime:.3f}s'
            )
        return result

    def _initializeShortRoutePool(self, outputFlag=0):
        """Pre-generate short exact routes for small instances.

        This is only a warm-start device. It never restricts pricing and hence
        cannot affect exactness. For N<=shortRouteTaskThreshold the default
        adds all feasible singleton/pair customer sequences (singletons are
        usually already present through savings and are deduplicated).
        """
        if self.shortRouteInitialized:
            return self.shortRouteColumnsAdded
        self.shortRouteInitialized = True

        if (
            self.preGenerateShortRoutes <= 0
            or len(self.data.C) > self.shortRouteTaskThreshold
        ):
            return 0

        import itertools as _itertools
        start = time.perf_counter()
        added = 0
        cap = self.maxShortRouteColumns
        tasks = tuple(self.data.C)

        for length in range(1, min(self.preGenerateShortRoutes, len(tasks)) + 1):
            for sequence in _itertools.permutations(tasks, length):
                column = self.routeEvaluator.evaluate(sequence)
                if column is None:
                    continue
                wasAdded, _ = self.columnManager.add(column)
                if wasAdded:
                    added += 1
                    self.profiler.increment('globalColumnsAdded')
                if cap is not None and added >= cap:
                    break
            if cap is not None and added >= cap:
                break

        self.shortRouteBuildTime = time.perf_counter() - start
        self.shortRouteColumnsAdded = added
        self.profiler.addTime('shortRouteWarmStart', self.shortRouteBuildTime)
        if outputFlag and added:
            print(
                f'[SHORT ROUTES] isolated-best complete routes for customer sequences length<= {self.preGenerateShortRoutes} | '
                f'+{added} columns | time={self.shortRouteBuildTime:.3f}s'
            )
        return added

    @staticmethod
    def _gurobiOutputFlag(outputFlag):
        """
        outputFlag is a BPC verbosity level, not a raw Gurobi parameter:
          0 = silent
          1 = concise BPC progress + final routes
          2 = detailed column-generation progress + final BSS schedule
          3 = level 2 plus native Gurobi logs
        """
        return 1 if outputFlag >= 3 else 0

    def _printCgProgress(
        self,
        phase,
        iteration,
        masterValue,
        columnsBefore,
        added,
        bestReducedCost,
        optimal,
    ):
        phaseName = 'P1' if phase == 1 else 'P2'
        rcText = (
            'none'
            if bestReducedCost is None
            else _format_number(bestReducedCost, 6)
        )
        state = 'pricing optimal' if optimal else f'+{added} cols'
        print(
            f'    [CG-{phaseName}] iter={iteration:03d} | '
            f'master={_format_number(masterValue, 6):>12} | '
            f'cols={columnsBefore:5d}->{len(self.columns):5d} | '
            f'bestRC={rcText:>12} | {state}'
        )

    def _printMasterDualValues(self, master, duals, phase, iteration):
        """Print the complete LP dual vector used by the current pricing round.

        BSS-cut duals are shown only for diagnosis.  They are master-only and
        deliberately are not part of ``MasterDuals`` passed to pricing.
        """
        phaseName = 'P1' if phase == 1 else 'P2'

        coverageText = ', '.join(
            f'C{task}={_format_number(duals.coverage[task], 6)}'
            for task in self.data.C
        )
        slotText = ', '.join(
            f'V{k + 1}={_format_number(duals.slot[k], 6)}'
            for k in range(self.data.K)
        )
        makespanText = ', '.join(
            f'V{k + 1}={_format_number(duals.makespan[k], 6)}'
            for k in range(self.data.K)
        )
        betaSum = sum(float(duals.makespan[k]) for k in range(self.data.K))

        print(
            f'    [DUAL-{phaseName}] iter={iteration:03d} | '
            f'coverage: {coverageText}'
        )
        print(
            f'    [DUAL-{phaseName}] iter={iteration:03d} | '
            f'slot: {slotText}'
        )
        print(
            f'    [DUAL-{phaseName}] iter={iteration:03d} | '
            f'beta/makespan: {makespanText} | sumBeta={_format_number(betaSum, 6)}'
        )

        if getattr(duals, 'src', None):
            srcItems = []
            for cut, value in sorted(
                duals.src.items(),
                key=lambda item: getattr(item[0], 'key', repr(item[0])),
            ):
                customers = getattr(cut, 'customers', None)
                label = (
                    '{' + ','.join(f'C{i}' for i in customers) + '}'
                    if customers is not None
                    else repr(getattr(cut, 'key', cut))
                )
                srcItems.append(f'{label}={_format_number(value, 6)}')
            print(
                f'    [DUAL-{phaseName}] iter={iteration:03d} | '
                f'SRC: {", ".join(srcItems)}'
            )

        bssDuals = master.getBssCutDuals() if phase == 2 else {}
        if bssDuals:
            bssItems = []
            for index, (cut, value) in enumerate(bssDuals.items(), start=1):
                bssItems.append(f'cut{index}={_format_number(value, 6)}')
            print(
                f'    [DUAL-{phaseName}] iter={iteration:03d} | '
                f'BSS(master-only): {", ".join(bssItems)}'
            )

    def _newPersistentMaster(
        self,
        phase,
        branchingState=None,
        outputFlag=0,
        timeLimit=None,
    ):
        return PersistentRestrictedMasterProblem(
            self.data,
            columns=None,
            phase=phase,
            bssCuts=(self.bssCuts if phase == 2 else None),
            srcCuts=(self.srcCuts if phase == 2 else None),
            branchingState=branchingState,
            outputFlag=self._gurobiOutputFlag(outputFlag),
            timeLimit=timeLimit,
            profiler=self.profiler,
            useDualSimplex=True,
            columnManager=self.columnManager,
        )

    def _runPersistentCG(
        self,
        master,
        phase,
        outputFlag=0,
        branchingState=None,
        timeLimit=None,
    ):
        """Run exact CG on an already-created persistent RMP."""
        try:
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError('Column generation requires gurobipy') from error

        iteration = 0
        while True:
            iteration += 1
            columnsBefore = len(self.columns)

            master.syncColumns(self.columnManager)
            if phase == 2:
                master.syncBssCuts(self.bssCuts)
                master.syncSrcCuts(self.srcCuts)

            model = master.optimize(timeLimit=timeLimit)
            self.master = master

            if model.Status == GRB.INFEASIBLE:
                if outputFlag >= 2:
                    print(f'    [CG-P{phase}] iter={iteration:03d} | RMP infeasible')
                return iteration, 'INFEASIBLE'
            if model.Status != GRB.OPTIMAL:
                if outputFlag >= 2:
                    print(
                        f'    [CG-P{phase}] iter={iteration:03d} | '
                        f'Gurobi status={model.Status}'
                    )
                return iteration, f'GUROBI_STATUS_{model.Status}'

            masterValue = (
                master.getPhaseOneArtificialValue()
                if phase == 1
                else master.getTValue()
            )

            if phase == 1 and masterValue <= self.phaseOneTolerance:
                if outputFlag >= 2:
                    print(
                        f'    [CG-P1] iter={iteration:03d} | '
                        f'master={_format_number(masterValue, 6):>12} | '
                        f'cols={columnsBefore:5d} | feasible RMP -> pricing skipped'
                    )
                return iteration, 'OPTIMAL'

            duals = master.getDuals()
            if self.printMasterDuals and outputFlag:
                self._printMasterDualValues(
                    master=master,
                    duals=duals,
                    phase=phase,
                    iteration=iteration,
                )
            with self.profiler.timeBlock('pricing'):
                candidates = self.pricing.price(
                    duals=duals,
                    existingSignatures=self._existingSignatures(),
                    maxColumns=self.maxColumnsPerRound,
                    branchingState=branchingState,
                )
            self.profiler.increment('pricingCalls')
            stats = getattr(self.pricing, 'lastStatistics', None) or {}
            self.profiler.increment('pricingGeneratedLabels', stats.get('generatedLabels', 0))
            self.profiler.increment('pricingAcceptedLabels', stats.get('acceptedLabels', 0))
            self.profiler.increment(
                'pricingDominanceComparisons',
                stats.get('dominanceResourceComparisons', 0),
            )

            bestReducedCost = (
                min(candidate.reducedCost for candidate in candidates)
                if candidates
                else None
            )

            if not candidates:
                if outputFlag >= 2:
                    self._printCgProgress(
                        phase=phase,
                        iteration=iteration,
                        masterValue=masterValue,
                        columnsBefore=columnsBefore,
                        added=0,
                        bestReducedCost=bestReducedCost,
                        optimal=True,
                    )
                return iteration, 'OPTIMAL'

            addedColumns = self._addPricingCandidates(candidates)
            added = len(addedColumns)
            if addedColumns:
                master.syncColumns(self.columnManager)

            if outputFlag >= 2:
                self._printCgProgress(
                    phase=phase,
                    iteration=iteration,
                    masterValue=masterValue,
                    columnsBefore=columnsBefore,
                    added=added,
                    bestReducedCost=bestReducedCost,
                    optimal=(added == 0),
                )

            if added == 0:
                return iteration, 'OPTIMAL'

    def _solveCGPhase(
        self,
        phase,
        outputFlag=0,
        branchingState=None,
        timeLimit=None,
    ):
        master = self._newPersistentMaster(
            phase=phase,
            branchingState=branchingState,
            outputFlag=outputFlag,
            timeLimit=timeLimit,
        )
        return self._runPersistentCG(
            master=master,
            phase=phase,
            outputFlag=outputFlag,
            branchingState=branchingState,
            timeLimit=timeLimit,
        )

    def solve(self, outputFlag=0):
        self.profiler.reset()
        self._initializeWarmStart(outputFlag=outputFlag)
        self._initializeShortRoutePool(outputFlag=outputFlag)
        self.phaseOneIterations, status = self._solveCGPhase(phase=1, outputFlag=outputFlag)
        if status != 'OPTIMAL':
            self.status = f'PHASE1_{status}'
            return self

        artificial = self.master.getPhaseOneArtificialValue()
        if artificial > self.phaseOneTolerance:
            self.status = 'INFEASIBLE_BASE_ROUTING'
            self.lowerBound = float('inf')
            return self

        self.phaseTwoIterations, status = self._solveCGPhase(phase=2, outputFlag=outputFlag)
        if status != 'OPTIMAL':
            self.status = f'PHASE2_{status}'
            return self

        self.lowerBound = self.master.getTValue()
        self.status = 'ROOT_LP_OPTIMAL'
        return self

    def getResult(self):
        return {
            'status': self.status,
            'lowerBound': self.lowerBound,
            'columnCount': len(self.columns),
            'srcCutCount': len(self.srcCuts),
            'phaseOneIterations': self.phaseOneIterations,
            'phaseTwoIterations': self.phaseTwoIterations,
            'pricingMode': self.pricingMode,
            'pricingStatistics': getattr(self.pricing, 'lastStatistics', None),
            'warmStartColumnsAdded': self.warmStartColumnsAdded,
            'warmStartBuildTime': self.warmStartBuildTime,
            'shortRouteColumnsAdded': self.shortRouteColumnsAdded,
            'shortRouteBuildTime': self.shortRouteBuildTime,
            'performanceProfile': self.profiler.snapshot(),
            'cacheStatistics': collectCacheStatistics(self.routeEvaluator, None),
            'warmStartFeasibleStarts': (
                self.warmStartResult.feasibleStarts
                if self.warmStartResult is not None else 0
            ),
            'positiveX': self.master.getPositiveX() if self.master is not None and self.master.model.SolCount else [],
        }


@dataclass(order=True)
class _QueueNode:
    estimatedLowerBound: float
    nodeId: int
    depth: int = field(compare=False)
    branchingState: object = field(compare=False)
    parentId: object = field(compare=False, default=None)


class BranchPriceCutSolver(RootColumnGenerationSolver):
    """
    V0.20 Branch-and-Price-and-Cut framework with COMPLETE physical route
    columns and slot-specific PHYSICAL-arc flow branching.

    Column identity
    ---------------
    A route column is its full physical node sequence. Customer nodes are
    elementary; the canonical physical BSS node S[0] may be visited repeatedly.
    Thus (0,3,S,5,end) and (0,3,5,S,end) are different columns.

    Branch variable
    ---------------
        f^k_ij = sum_r a^r_ij x_kr,
    where a^r_ij=1 iff physical arc (i,j) appears in the complete route.
    Therefore customer->BSS, BSS->customer and BSS->depot arcs are branchable
    exactly like customer/customer or depot/customer arcs. A fractional
    f^k_ij is branched to 0 and 1.

    Branch propagation to pricing
    -----------------------------
    f^k_ij=0: slot-k pricing forbids physical arc (i,j).
    f^k_ij=1: every slot-k route column must contain physical arc (i,j).
    Since sum_r x_kr=1, restricting the slot-k column universe in this way is
    exactly equivalent to the branch equations. No branch-row dual is needed.
    Because a repeatable BSS can make two different complete route orders share
    one physical-arc incidence vector, a customer-precedence branch ``i before j``
    is used only when no fractional physical arc can separate a fractional master
    point.  This fallback depends only on the customer visited set, so pricing
    never stores the predecessor used to enter the BSS.

    Exactness
    ---------
    - every node LP is solved by exact elementary pricing;
    - branches partition the slot-labelled integer route space;
    - at every integral routing solution, each selected column already fixes
      all BSS visits; the BSS subproblem schedules those fixed events on the
      one physical swap server and must be proven optimal;
    - if the base master underestimates that routing combination, the global
      combinatorial BSS optimality cut is added and the same node is re-priced;
    - pruning uses only valid LP lower bounds versus a feasible BSS incumbent.
    - a directed savings warm start supplies initial columns/UB only; it never
      restricts pricing or changes lower bounds;
    - pricing uses exact unreachable-set inclusion dominance by default;
    - classical 3-row subset-row cuts strengthen fractional node LPs; their
      duals are priced exactly by deriving SRC parity from unreachableMask,
      without adding a parity field to the mathematical label;
    - open branch nodes are explored by best bound: the node with the smallest
      inherited valid LP lower bound is processed first. This changes only search
      order and does not alter any lower bound, cut, or pricing rule.

    Without a node/time limit and assuming all Gurobi subproblems are solved to
    proven optimality, exhaustion of the node queue certifies global optimality
    for the model represented by the column universe/pricing graph.
    """

    def __init__(
        self,
        modelData,
        pricingMode='labeling',
        maxEnumeratedTasks=9,
        maxColumnsPerRound=50,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
        phaseOneTolerance=1e-8,
        branchTolerance=1e-7,
        cutTolerance=1e-7,
        maxBssCuts=None,
        bssTimeLimit=None,
        dominanceMode='subset',
        useSavingsWarmStart=True,
        savingsStarts=12,
        savingsSeed=1,
        savingsRandomization=0.20,
        savingsExcessRoutePenalty=1.0e9,
        maxWarmStartColumns=1000,
        useSavingsIncumbent=True,
        useSRC=False,
        srcViolationTolerance=1e-7,
        maxSrcCutsPerRound=10,
        maxSrcRoundsPerNode=5,
        maxTotalSrcCuts=100,
        srcMaxDepth=2,
        preGenerateShortRoutes=2,
        shortRouteTaskThreshold=15,
        maxShortRouteColumns=1000,
        deduplicateSymmetricNodes=True,
        printMasterDuals=False,
    ):
        super().__init__(
            modelData=modelData,
            pricingMode=pricingMode,
            maxEnumeratedTasks=maxEnumeratedTasks,
            maxColumnsPerRound=maxColumnsPerRound,
            reducedCostTolerance=reducedCostTolerance,
            dominanceTolerance=dominanceTolerance,
            phaseOneTolerance=phaseOneTolerance,
            bssCuts=[],
            srcCuts=[],
            dominanceMode=dominanceMode,
            useSavingsWarmStart=useSavingsWarmStart,
            savingsStarts=savingsStarts,
            savingsSeed=savingsSeed,
            savingsRandomization=savingsRandomization,
            savingsExcessRoutePenalty=savingsExcessRoutePenalty,
            maxWarmStartColumns=maxWarmStartColumns,
            preGenerateShortRoutes=preGenerateShortRoutes,
            shortRouteTaskThreshold=shortRouteTaskThreshold,
            maxShortRouteColumns=maxShortRouteColumns,
            printMasterDuals=printMasterDuals,
        )

        self.useSavingsIncumbent = bool(useSavingsIncumbent)

        # Classical 3-row subset-row cuts (SRCs).  These are globally valid
        # strengthening cuts, so limiting separation rounds/depth affects only
        # performance, never correctness of the final BPC algorithm.
        self.useSRC = bool(useSRC)
        self.srcViolationTolerance = float(srcViolationTolerance)
        self.maxSrcCutsPerRound = maxSrcCutsPerRound
        self.maxSrcRoundsPerNode = max(0, int(maxSrcRoundsPerNode))
        self.maxTotalSrcCuts = maxTotalSrcCuts
        self.srcMaxDepth = srcMaxDepth
        self.srcSeparator = ThreeRowSubsetCutSeparator(
            modelData=modelData,
            violationTolerance=self.srcViolationTolerance,
            maxCutsPerRound=self.maxSrcCutsPerRound,
        )
        self.srcCutByKey = {}
        self.srcCuts = []

        self.branchTolerance = branchTolerance
        self.cutTolerance = cutTolerance
        self.maxBssCuts = maxBssCuts
        self.bssTimeLimit = bssTimeLimit

        self.brancher = ArcFlowBrancher(modelData, tolerance=branchTolerance)
        self.bssScheduler = ExactBssScheduler(
            modelData=modelData,
            profiler=self.profiler,
        )

        self.cutByKey = {}
        self.incumbentObjective = float('inf')
        self.incumbentColumns = []
        self.incumbentSlots = []
        self.incumbentBssResult = None
        self.bestBound = 0.0

        self.nodesCreated = 0
        self.nodesProcessed = 0
        self.nodesPrunedByBound = 0
        self.nodesPrunedInfeasible = 0
        self.integralNodes = 0
        self.branchCount = 0
        self.cutCount = 0
        self.srcCutCount = 0
        self.srcSeparationRounds = 0
        self.srcCutsAdded = 0
        self.totalPhaseOneIterations = 0
        self.totalPhaseTwoIterations = 0
        self.symmetricNodesSkipped = 0
        self.phaseOneSkipped = 0
        self.phaseOneActivated = 0
        self.runtime = 0.0
        self._deadline = None
        self._interrupted = False
        self._interruptionStatus = None
        self._lastNodePhaseOneIterations = 0
        self._lastNodePhaseTwoIterations = 0
        self.searchStrategy = 'BEST_BOUND'
        self.deduplicateSymmetricNodes = bool(deduplicateSymmetricNodes)

    def _remainingTime(self):
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.time())

    def _timeExpired(self):
        remaining = self._remainingTime()
        return remaining is not None and remaining <= 0.0

    def _addOrStrengthenCut(self, selectedColumns, value):
        signatures = [
            column.signature
            for column in selectedColumns
            if not column.isEmpty
        ]
        cut = BssOptimalityCut.canonical(signatures, value)
        previous = self.cutByKey.get(cut.key)
        if previous is not None and previous.value >= cut.value - self.cutTolerance:
            return False
        self.cutByKey[cut.key] = cut
        self.bssCuts = list(self.cutByKey.values())
        self.cutCount = len(self.bssCuts)
        return True

    def _nodeLabel(self, node):
        if node == self.data.startNode:
            return 'DepotStart'
        if node == self.data.endNode:
            return 'DepotEnd'
        if node in self.data.C:
            return f'C{node}'
        if node in self.data.S:
            return 'BSS'
        return str(node)

    def _formatArc(self, arc):
        return f'{self._nodeLabel(arc[0])} -> {self._nodeLabel(arc[1])}'

    def _routeText(self, column):
        if column.isEmpty:
            return '(empty)'
        return '[' + ', '.join(str(task) for task in column.tasks) + ']'

    def _printStartBanner(self, outputFlag, nodeLimit, timeLimit):
        if not outputFlag:
            return
        width = 96
        print('=' * width)
        print('BRANCH-AND-PRICE-AND-CUT | COMPLETE ROUTE COLUMNS'.center(width))
        print('=' * width)
        print(
            f'Instance : tasks={len(self.data.C)}, vehicles={self.data.K}, '
            f'Q={_format_number(self.data.Q, 2)}, QMin={_format_number(self.data.QMin, 2)}'
        )
        print(
            f'Pricing  : {self.pricingMode} | dominance={self.dominanceMode} | '
            f'maxColumns/round={self.maxColumnsPerRound} | '
            f'nodeLimit={nodeLimit if nodeLimit is not None else "None"} | '
            f'timeLimit={_format_number(timeLimit, 1) + "s" if timeLimit is not None else "None"}'
        )
        print(
            'Tree     : BEST-BOUND | branch=physical arcs; customer-precedence fallback | '
            f'symmetry-dedup={"ON" if self.deduplicateSymmetricNodes else "OFF"}'
        )
        print(
            'Core     : RMP -> exact pricing -> integer? BSS separation/UB : branch'
        )
        print(
            f'RMP      : persistent/incremental | short-route warm pool<='
            f'{self.preGenerateShortRoutes} (N<={self.shortRouteTaskThreshold})'
        )
        print(
            f'SRC      : {"ON" if self.useSRC else "OFF"} | family=3-row | '
            f'max/round={self.maxSrcCutsPerRound} | rounds/node={self.maxSrcRoundsPerNode} | '
            f'maxDepth={self.srcMaxDepth if self.srcMaxDepth is not None else "all"} | '
            f'maxTotal={self.maxTotalSrcCuts if self.maxTotalSrcCuts is not None else "None"}'
        )
        print(
            'Log level: 1=concise BPC, 2=+column generation details, '
            '3=+native Gurobi logs'
        )
        if self.printMasterDuals:
            print('Dual log : ON | print every solved RMP dual vector before pricing')
        print('-' * width)

    def _printNodeHeader(self, node, queueSize):
        print(
            f'[NODE {node.nodeId:05d}] depth={node.depth:<3d} | queue={queueSize:<5d} | '
            f'inherited LB={_format_number(node.estimatedLowerBound, 3):>10} | '
            f'global UB={_format_number(self.incumbentObjective, 3):>10} | '
            f'gap={_format_gap(node.estimatedLowerBound, self.incumbentObjective):>8}'
        )

    def _printLpLine(self, nodeLowerBound):
        print(
            f'  [LP] LB={_format_number(nodeLowerBound, 3):>10} | '
            f'UB={_format_number(self.incumbentObjective, 3):>10} | '
            f'gap={_format_gap(nodeLowerBound, self.incumbentObjective):>8} | '
            f'cols={len(self.columns):5d} | BSS/SRC cuts='
            f'{len(self.bssCuts):3d}/{len(self.srcCuts):3d} | '
            f'CG(P1/P2)={self._lastNodePhaseOneIterations}/{self._lastNodePhaseTwoIterations}'
        )

    def _srcSeparationAllowed(self, nodeDepth):
        if not self.useSRC:
            return False
        if self.maxTotalSrcCuts is not None and len(self.srcCuts) >= self.maxTotalSrcCuts:
            return False
        if self.srcMaxDepth is not None and nodeDepth > self.srcMaxDepth:
            return False
        return self.maxSrcRoundsPerNode > 0

    def _separateSrcCuts(self, outputFlag=0):
        remainingCapacity = None
        if self.maxTotalSrcCuts is not None:
            remainingCapacity = max(0, self.maxTotalSrcCuts - len(self.srcCuts))
            if remainingCapacity == 0:
                return 0, 0.0

        violations = self.srcSeparator.separate(
            self.master,
            existingKeys=self.srcCutByKey.keys(),
        )
        if remainingCapacity is not None:
            violations = violations[:remainingCapacity]

        if not violations:
            return 0, 0.0

        for violation in violations:
            cut = violation.cut
            self.srcCutByKey[cut.key] = cut

        self.srcCuts = list(self.srcCutByKey.values())
        self.srcCutCount = len(self.srcCuts)
        self.srcCutsAdded += len(violations)
        maxViolation = max(item.violation for item in violations)

        if outputFlag:
            print(
                f'  [SRC] +{len(violations)} 3-row cut(s) | '
                f'total={len(self.srcCuts)} | maxViolation={maxViolation:.6g}'
            )
            if outputFlag >= 2:
                for item in violations:
                    print(
                        f'        S={item.cut.customers} | '
                        f'lhs={item.lhs:.6f} <= {item.rhs:.0f} | '
                        f'viol={item.violation:.6f}'
                    )
                print('        -> re-solve RMP and re-price with SRC duals')

        return len(violations), maxViolation

    def _solveNodeLP(
        self,
        branchingState,
        outputFlag,
        nodeDepth=0,
        phaseTwoMaster=None,
    ):
        """Solve/re-solve one branch node with a persistent Phase-II RMP.

        The restricted Phase-II master is tried first.  Phase I is activated
        only when the current restricted Phase-II master is infeasible, which
        means branch restrictions may require columns not yet in the global
        pool.  A BSS/SRC cut does not invalidate route feasibility, so repeated
        solves of the same node reuse the same Phase-II model and never rerun
        Phase I.
        """
        if phaseTwoMaster is None:
            phaseTwoMaster = self._newPersistentMaster(
                phase=2,
                branchingState=branchingState,
                outputFlag=outputFlag,
                timeLimit=self._remainingTime(),
            )
        else:
            phaseTwoMaster.syncColumns(self.columns)
            phaseTwoMaster.syncBssCuts(self.bssCuts)
            phaseTwoMaster.syncSrcCuts(self.srcCuts)

        totalPhaseOneIterations = 0
        totalPhaseTwoIterations = 0

        # First attempt the actual Phase-II restricted master.  If it is
        # feasible, exact pricing is all that is needed and Phase I is skipped.
        remaining = self._remainingTime()
        phaseTwoIterations, status = self._runPersistentCG(
            master=phaseTwoMaster,
            phase=2,
            outputFlag=outputFlag,
            branchingState=branchingState,
            timeLimit=remaining,
        )
        totalPhaseTwoIterations += phaseTwoIterations

        if status == 'INFEASIBLE':
            self.phaseOneActivated += 1
            self.profiler.increment('phaseOneActivated')
            if outputFlag >= 2:
                print('    [P1] restricted Phase-II RMP infeasible -> exact Phase I activated')

            phaseOneMaster = self._newPersistentMaster(
                phase=1,
                branchingState=branchingState,
                outputFlag=outputFlag,
                timeLimit=self._remainingTime(),
            )
            remaining = self._remainingTime()
            phaseOneIterations, phaseOneStatus = self._runPersistentCG(
                master=phaseOneMaster,
                phase=1,
                outputFlag=outputFlag,
                branchingState=branchingState,
                timeLimit=remaining,
            )
            totalPhaseOneIterations += phaseOneIterations

            if phaseOneStatus != 'OPTIMAL':
                self._lastNodePhaseOneIterations = totalPhaseOneIterations
                self._lastNodePhaseTwoIterations = totalPhaseTwoIterations
                self.totalPhaseOneIterations += totalPhaseOneIterations
                self.totalPhaseTwoIterations += totalPhaseTwoIterations
                self.master = phaseOneMaster
                return phaseOneStatus, None, phaseTwoMaster

            artificial = phaseOneMaster.getPhaseOneArtificialValue()
            if artificial > self.phaseOneTolerance:
                self._lastNodePhaseOneIterations = totalPhaseOneIterations
                self._lastNodePhaseTwoIterations = totalPhaseTwoIterations
                self.totalPhaseOneIterations += totalPhaseOneIterations
                self.totalPhaseTwoIterations += totalPhaseTwoIterations
                self.master = phaseOneMaster
                return 'INFEASIBLE', None, phaseTwoMaster

            # Phase I may have generated globally useful route columns.  Append
            # them to the already-created Phase-II master, keeping its model
            # object and all permanent rows intact.
            phaseTwoMaster.syncColumns(self.columns)
            phaseTwoMaster.syncBssCuts(self.bssCuts)
            phaseTwoMaster.syncSrcCuts(self.srcCuts)

            remaining = self._remainingTime()
            extraIterations, status = self._runPersistentCG(
                master=phaseTwoMaster,
                phase=2,
                outputFlag=outputFlag,
                branchingState=branchingState,
                timeLimit=remaining,
            )
            totalPhaseTwoIterations += extraIterations
        else:
            self.phaseOneSkipped += 1
            self.profiler.increment('phaseOneSkipped')

        if status != 'OPTIMAL':
            self._lastNodePhaseOneIterations = totalPhaseOneIterations
            self._lastNodePhaseTwoIterations = totalPhaseTwoIterations
            self.totalPhaseOneIterations += totalPhaseOneIterations
            self.totalPhaseTwoIterations += totalPhaseTwoIterations
            self.master = phaseTwoMaster
            return status, None, phaseTwoMaster

        # Phase II alternates exact column generation and optional SRC
        # separation on the SAME persistent master. New SRC duals therefore
        # trigger re-pricing without a model rebuild.
        srcRoundsThisNode = 0
        while (
            self._srcSeparationAllowed(nodeDepth)
            and srcRoundsThisNode < self.maxSrcRoundsPerNode
        ):
            with self.profiler.timeBlock('srcSeparation'):
                added, _ = self._separateSrcCuts(outputFlag=outputFlag)
            self.srcSeparationRounds += 1
            srcRoundsThisNode += 1
            if added == 0:
                break

            phaseTwoMaster.syncSrcCuts(self.srcCuts)
            remaining = self._remainingTime()
            extraIterations, status = self._runPersistentCG(
                master=phaseTwoMaster,
                phase=2,
                outputFlag=outputFlag,
                branchingState=branchingState,
                timeLimit=remaining,
            )
            totalPhaseTwoIterations += extraIterations
            if status != 'OPTIMAL':
                self._lastNodePhaseOneIterations = totalPhaseOneIterations
                self._lastNodePhaseTwoIterations = totalPhaseTwoIterations
                self.totalPhaseOneIterations += totalPhaseOneIterations
                self.totalPhaseTwoIterations += totalPhaseTwoIterations
                self.master = phaseTwoMaster
                return status, None, phaseTwoMaster

        self._lastNodePhaseOneIterations = totalPhaseOneIterations
        self._lastNodePhaseTwoIterations = totalPhaseTwoIterations
        self.totalPhaseOneIterations += totalPhaseOneIterations
        self.totalPhaseTwoIterations += totalPhaseTwoIterations
        self.master = phaseTwoMaster
        return 'OPTIMAL', phaseTwoMaster.getTValue(), phaseTwoMaster

    def _selectedIntegralEntries(self):
        return self.master.getSelectedLPColumns(
            tolerance=self.branchTolerance,
            includeEmpty=False,
        )

    def _solveIntegralBss(self, selectedColumns, outputFlag):
        remaining = self._remainingTime()
        if self.bssTimeLimit is None:
            localLimit = remaining
        elif remaining is None:
            localLimit = self.bssTimeLimit
        else:
            localLimit = min(self.bssTimeLimit, remaining)

        return self.bssScheduler.solve(
            selectedColumns,
            outputFlag=self._gurobiOutputFlag(outputFlag),
            timeLimit=localLimit,
            mipGap=0.0,
        )

    def _buildGreedyWarmStartBss(self, selectedColumns):
        """
        Build a fast FEASIBLE single-BSS schedule for the isolated-minimum
        swap placement stored in each warm-start column. This gives a valid
        incumbent upper bound without solving a BSS MIP or enumerating all swap
        placements. It is never used for lower bounds or BSS optimality cuts.
        """
        if not selectedColumns:
            return BssScheduleResult(
                objective=0.0,
                status='GREEDY_WARM_START',
                provenOptimal=False,
                routeCompletion={},
                routeWaiting={},
                events=[],
            )

        swapTime = float(self.data.p[self.data.S[0]])
        routeEvents = {}
        for r, column in enumerate(selectedColumns):
            routeEvents[r] = [
                {
                    'eventIndex': int(event.index),
                    'afterTask': event.afterTask,
                    'nominalArrival': float(event.arrivalTime),
                    'nominalEnd': float(event.baseEndTime),
                }
                for event in column.swapEvents
            ]

        nextEvent = {r: 0 for r in routeEvents}
        previousStart = {r: None for r in routeEvents}
        routeWaiting = {r: 0.0 for r in routeEvents}
        stationAvailable = 0.0
        events = []

        while True:
            readyEvents = []
            for r, eventList in routeEvents.items():
                h = nextEvent[r]
                if h >= len(eventList):
                    continue

                event = eventList[h]
                if h == 0:
                    ready = event['nominalArrival']
                else:
                    previousNominalEnd = eventList[h - 1]['nominalEnd']
                    nominalGap = event['nominalArrival'] - previousNominalEnd
                    ready = previousStart[r] + swapTime + nominalGap

                readyEvents.append((ready, r, h, event))

            if not readyEvents:
                break

            ready, r, h, event = min(
                readyEvents,
                key=lambda item: (item[0], item[1], item[2]),
            )
            start = max(ready, stationAvailable)
            wait = max(0.0, start - ready)

            routeWaiting[r] += wait
            previousStart[r] = start
            nextEvent[r] += 1
            stationAvailable = start + swapTime

            events.append({
                'routeIndex': r,
                'eventIndex': h,
                'afterTask': event['afterTask'],
                'arrivalTime': ready,
                'startTime': start,
                'endTime': start + swapTime,
                'waitTime': wait,
            })

        routeCompletion = {
            r: float(column.duration + routeWaiting[r])
            for r, column in enumerate(selectedColumns)
        }
        objective = max(routeCompletion.values(), default=0.0)
        events.sort(key=lambda event: (event['startTime'], event['routeIndex']))

        return BssScheduleResult(
            objective=float(objective),
            status='GREEDY_WARM_START',
            provenOptimal=False,
            routeCompletion=routeCompletion,
            routeWaiting=routeWaiting,
            events=events,
        )

    def _initializeSavingsIncumbent(self, outputFlag):
        if not self.useSavingsIncumbent:
            return
        result = self.warmStartResult
        if result is None or not result.bestRoutes:
            return
        if len(result.bestRoutes) > self.data.K:
            if outputFlag:
                print(
                    f'[WARM UB] no K-route savings solution available '
                    f'(best has {len(result.bestRoutes)} routes > K={self.data.K})'
                )
            return

        selectedColumns = list(result.bestRoutes)
        bssResult = self._buildGreedyWarmStartBss(selectedColumns)
        selectedEntries = []
        for slot, column in enumerate(selectedColumns):
            columnIndex = self.signatureToIndex[column.signature]
            selectedEntries.append((slot, columnIndex, column, 1.0))

        old = self.incumbentObjective
        self._updateIncumbent(selectedEntries, bssResult, outputFlag=0)

        if outputFlag and self.incumbentObjective < old:
            print(
                f'[WARM UB] feasible greedy-BSS incumbent = '
                f'{_format_number(self.incumbentObjective, 3)} '
                f'from {len(selectedColumns)} savings route(s)'
            )
            if outputFlag >= 2:
                for slot, column in enumerate(selectedColumns):
                    print(
                        f'      V{slot + 1}: tasks={self._routeText(column)} | '
                        f'base={_format_number(column.duration, 3)}'
                    )

    def _updateIncumbent(self, selectedEntries, bssResult, outputFlag):
        selectedColumns = [entry[2] for entry in selectedEntries]
        selectedSlots = [entry[0] for entry in selectedEntries]

        if bssResult.objective < self.incumbentObjective - self.cutTolerance:
            oldObjective = self.incumbentObjective
            self.incumbentObjective = float(bssResult.objective)
            self.incumbentColumns = list(selectedColumns)
            self.incumbentSlots = list(selectedSlots)
            self.incumbentBssResult = bssResult

            if outputFlag:
                oldText = _format_number(oldObjective, 3)
                print(
                    f'  [INCUMBENT] UB {oldText} -> '
                    f'{_format_number(self.incumbentObjective, 3)}'
                )
                if outputFlag >= 2:
                    for vehicle, column in zip(selectedSlots, selectedColumns):
                        print(
                            f'      V{vehicle + 1}: tasks={self._routeText(column)} | '
                            f'isolatedBase={_format_number(column.duration, 3)}'
                        )

    def _processIntegralMasterSolution(self, nodeLowerBound, outputFlag):
        """Run the BSS separation step for one integral routing solution.

        This is the only place where the routing master talks to the BSS
        scheduling subproblem:

            integral RMP solution
                -> exact BSS schedule
                -> feasible UB update
                -> add one global BSS optimality cut iff RMP underestimates it.

        Returns ``REPRICE`` after a newly added cut, ``FATHOM`` if the current
        integer routing solution is already represented exactly by the master,
        or ``STOP`` if exact BSS separation cannot be completed.
        """
        self.integralNodes += 1
        selectedEntries = self._selectedIntegralEntries()
        selectedColumns = [entry[2] for entry in selectedEntries]

        if outputFlag:
            print(
                f'  [INTEGER] routing solution with {len(selectedColumns)} '
                f'nonempty route(s)'
            )
            if outputFlag >= 2:
                for slot, _, column, _ in selectedEntries:
                    print(
                        f'      V{slot + 1}: tasks={self._routeText(column)} | '
                        f'isolatedBase={_format_number(column.duration, 3)}'
                    )

        bssResult = self._solveIntegralBss(
            selectedColumns,
            outputFlag=outputFlag,
        )
        if not bssResult.provenOptimal:
            self._interrupted = True
            self._interruptionStatus = 'BSS_SP_NOT_PROVEN_OPTIMAL'
            if outputFlag:
                print(
                    f'  [STOP] BSS subproblem not proven optimal: '
                    f'{bssResult.status}'
                )
            return 'STOP'

        isolatedBase = max(
            (column.duration for column in selectedColumns),
            default=0.0,
        )
        bssUplift = max(0.0, bssResult.objective - isolatedBase)
        if outputFlag:
            print(
                f'  [BSS] isolated-base Cmax={_format_number(isolatedBase, 3)} | '
                f'true Cmax={_format_number(bssResult.objective, 3)} | '
                f'uplift={_format_number(bssUplift, 3)} | '
                f'status={bssResult.status}'
            )

        # The exact BSS schedule is feasible for the original problem, so it
        # may improve the global upper bound immediately, even when a cut is
        # still needed to lift the routing master at this integer combination.
        self._updateIncumbent(selectedEntries, bssResult, outputFlag)

        if nodeLowerBound < bssResult.objective - self.cutTolerance:
            if self.maxBssCuts is not None and len(self.bssCuts) >= self.maxBssCuts:
                self._interrupted = True
                self._interruptionStatus = 'BSS_CUT_LIMIT'
                if outputFlag:
                    print('  [STOP] maximum number of BSS cuts reached')
                return 'STOP'

            added = self._addOrStrengthenCut(
                selectedColumns,
                bssResult.objective,
            )
            if not added:
                raise RuntimeError(
                    'Integral route combination is still underestimated '
                    'but its exact BSS cut is already present.'
                )

            if outputFlag:
                print(
                    f'  [CUT #{len(self.bssCuts):04d}] '
                    f'T >= {_format_number(bssResult.objective, 3)} '
                    f'when physicalRoutes={ [tuple(c.nodes) for c in selectedColumns] }'
                )
                print('             -> re-solve RMP and re-price this branch node')
            return 'REPRICE'

        if outputFlag:
            print(
                '  [FATHOM] integral routing solution is BSS-consistent '
                'at this node'
            )
        return 'FATHOM'

    def _getIncumbentRouteDetails(self):
        if self.incumbentBssResult is None or not self.incumbentColumns:
            return []

        details = []
        for routeIndex, column in enumerate(self.incumbentColumns):
            slot = (
                self.incumbentSlots[routeIndex]
                if routeIndex < len(self.incumbentSlots)
                else routeIndex
            )
            nodes = tuple(column.nodes)
            pathText = ' -> '.join(self._nodeLabel(node) for node in nodes)
            baseDuration = float(column.duration)
            details.append({
                'vehicleSlot': slot,
                'vehicle': slot + 1,
                'routeIndex': routeIndex,
                'tasks': tuple(column.tasks),
                'nodes': nodes,
                'path': pathText,
                'baseDuration': baseDuration,
                'completionTime': float(
                    self.incumbentBssResult.routeCompletion.get(routeIndex, baseDuration)
                ),
                'waitingTime': float(
                    self.incumbentBssResult.routeWaiting.get(routeIndex, 0.0)
                ),
                'swapCount': int(column.stationVisitCount),
                'columnSignature': tuple(column.signature),
            })

        return details

    def getRoutes(self):
        return self._getIncumbentRouteDetails()

    def printResult(self, showBssSchedule=True, digits=3):
        result = self.getResult()
        width = 96
        print('\n' + '=' * width)
        print('BPC FINAL RESULT'.center(width))
        print('=' * width)
        print(f"Status        : {result['status']}")
        print(f"Optimality    : {'PROVEN' if result.get('optimalityProven') else 'NOT PROVEN'}")
        print(f"Objective (UB): {_format_number(result['objective'], digits)}")
        print(f"Best bound(LB): {_format_number(result['bestBound'], digits)}")
        gap = result['relativeGap']
        print(f"Relative gap  : {'-' if gap is None else f'{100.0 * gap:.4f}%'}")
        print(f"Runtime       : {_format_number(result['runtime'], 3)} s")
        print('-' * width)
        print(
            f"Search        : strategy={result.get('searchStrategy', 'BEST_BOUND')} | "
            f"processed={result['nodesProcessed']} / created={result['nodesCreated']} | "
            f"branches={result['branchCount']} | integralNodes={result['integralNodes']}"
        )
        print(
            f"Pruning       : bound={result['nodesPrunedByBound']} | "
            f"infeasible={result['nodesPrunedInfeasible']}"
        )
        print(
            f"Columns/Cuts  : columns={result['columnCount']} | "
            f"BSS cuts={result['cutCount']} | SRC cuts={result.get('srcCutCount', 0)} | "
            f"CG iterations P1/P2={result['phaseOneIterations']}/{result['phaseTwoIterations']}"
        )
        print(
            f"Warm start    : savings +{result.get('warmStartColumnsAdded', 0)} | "
            f"short +{result.get('shortRouteColumnsAdded', 0)} | "
            f"dominance={result.get('dominanceMode', '-')}"
        )
        print(
            f"Phase I       : activated={result.get('phaseOneActivated', 0)} | "
            f"skipped={result.get('phaseOneSkipped', 0)} | "
            f"symmetric nodes skipped={result.get('symmetricNodesSkipped', 0)}"
        )

        profile = result.get('performanceProfile', {})
        times = profile.get('times', {})
        counts = profile.get('counts', {})
        if times or counts:
            print('-' * width)
            print('PERFORMANCE PROFILE')
            print('-' * width)
            print(
                f"Master        : build={_format_number(times.get('masterBuild', 0.0), 3)}s | "
                f"LP optimize={_format_number(times.get('masterOptimize', 0.0), 3)}s | "
                f"solves={counts.get('masterOptimizeCalls', 0)}"
            )
            print(
                f"Pricing       : time={_format_number(times.get('pricing', 0.0), 3)}s | "
                f"calls={counts.get('pricingCalls', 0)} | "
                f"labels={counts.get('pricingGeneratedLabels', 0)} | "
                f"domCmp={counts.get('pricingDominanceComparisons', 0)}"
            )
            print(
                f"Route eval    : time={_format_number(times.get('routeEvaluation', 0.0), 3)}s | "
                f"cacheHit={counts.get('routeEvaluationCacheHits', 0)} | "
                f"cacheMiss={counts.get('routeEvaluationCacheMisses', 0)}"
            )
            print(
                f"BSS           : total={_format_number(times.get('bssTotal', 0.0), 3)}s | "
                f"enum={_format_number(times.get('bssEnumeration', 0.0), 3)}s | "
                f"MIP={_format_number(times.get('bssMipOptimize', 0.0), 3)}s | "
                f"cacheHit={counts.get('bssCacheHits', 0)} | "
                f"cacheMiss={counts.get('bssCacheMisses', 0)}"
            )
            if times.get('srcSeparation', 0.0) > 0.0:
                print(
                    f"SRC           : separation={_format_number(times.get('srcSeparation', 0.0), 3)}s"
                )

        routes = result.get('routes', [])
        if routes:
            print('\n' + '-' * width)
            print('FINAL ROUTES')
            print('-' * width)
            for route in routes:
                print(
                    f"Vehicle {route['vehicle']:>2d} | tasks={list(route['tasks'])} | "
                    f"base={_format_number(route['baseDuration'], digits)} | "
                    f"wait={_format_number(route['waitingTime'], digits)} | "
                    f"completion={_format_number(route['completionTime'], digits)} | "
                    f"swaps={route['swapCount']}"
                )
                print(f"             {route['path']}")
        else:
            print('\nNo BSS-feasible incumbent route set is available.')

        if showBssSchedule and result.get('bssEvents'):
            print('\n' + '-' * width)
            print('BSS EVENT SCHEDULE')
            print('-' * width)
            print(
                f"{'Evt':>4} {'Veh':>4} {'After':>8} "
                f"{'Arrival':>12} {'Start':>12} {'End':>12} {'Wait':>10}"
            )
            print('-' * width)
            routeToVehicle = {
                route['routeIndex']: route['vehicle']
                for route in routes
            }
            for eventNumber, event in enumerate(result['bssEvents'], start=1):
                vehicle = routeToVehicle.get(
                    event['routeIndex'],
                    event['routeIndex'] + 1,
                )
                after = self._nodeLabel(event['afterTask'])
                print(
                    f"{eventNumber:>4d} {vehicle:>4d} {after:>8} "
                    f"{_format_number(event['arrivalTime'], digits):>12} "
                    f"{_format_number(event['startTime'], digits):>12} "
                    f"{_format_number(event['endTime'], digits):>12} "
                    f"{_format_number(event['waitTime'], digits):>10}"
                )

        print('=' * width)

    def solve(
        self,
        outputFlag=1,
        nodeLimit=None,
        timeLimit=None,
    ):
        """
        Solve the full BPC tree using best-bound node selection.

        Open nodes are stored in a min-heap keyed by their inherited valid LP
        lower bound (with nodeId as a deterministic tie-break). The node with
        the smallest bound is processed next. This typically makes global-bound
        and gap progression much more stable than depth-first search, while
        changing only node exploration order, not BPC exactness.

        outputFlag is intentionally interpreted as a BPC verbosity level:
          0 : silent
          1 : concise node/branch/cut/incumbent log + final route summary
          2 : level 1 + every column-generation iteration + final BSS table
          3 : level 2 + native Gurobi logs
        """
        startWall = time.time()
        self._deadline = None if timeLimit is None else startWall + float(timeLimit)
        self.profiler.reset()

        # Reset solve statistics so one solver object can safely be solved again.
        self.incumbentObjective = float('inf')
        self.incumbentColumns = []
        self.incumbentSlots = []
        self.incumbentBssResult = None
        self.bestBound = 0.0
        self.nodesCreated = 0
        self.nodesProcessed = 0
        self.nodesPrunedByBound = 0
        self.nodesPrunedInfeasible = 0
        self.integralNodes = 0
        self.branchCount = 0
        self.cutCount = len(self.bssCuts)
        self.srcCutByKey = {}
        self.srcCuts = []
        self.srcCutCount = 0
        self.srcSeparationRounds = 0
        self.srcCutsAdded = 0
        self.totalPhaseOneIterations = 0
        self.totalPhaseTwoIterations = 0
        self.runtime = 0.0
        self._interrupted = False
        self._interruptionStatus = None
        activeNodeSafeBound = None

        self._printStartBanner(outputFlag, nodeLimit, timeLimit)
        self._initializeWarmStart(outputFlag=outputFlag)
        self._initializeShortRoutePool(outputFlag=outputFlag)
        self._initializeSavingsIncumbent(outputFlag=outputFlag)

        if self._timeExpired():
            self._interrupted = True
            self._interruptionStatus = 'TIME_LIMIT'

        rootState = ArcBranchingState.root(self.data)
        queue = []
        root = _QueueNode(
            estimatedLowerBound=0.0,
            nodeId=0,
            depth=0,
            branchingState=rootState,
            parentId=None,
        )
        heapq.heappush(queue, root)
        self.nodesCreated = 1
        nextNodeId = 1
        seenBranchStateKeys = {rootState.canonicalKey()} if self.deduplicateSymmetricNodes else set()

        while queue:
            if self._timeExpired():
                self._interrupted = True
                self._interruptionStatus = 'TIME_LIMIT'
                break
            if nodeLimit is not None and self.nodesProcessed >= nodeLimit:
                self._interrupted = True
                self._interruptionStatus = 'NODE_LIMIT'
                break

            # Best-bound: process the open node with the smallest inherited
            # valid lower bound. nodeId is the deterministic tie-break.
            node = heapq.heappop(queue)
            activeNodeSafeBound = float(node.estimatedLowerBound)

            # Safe inherited-bound pruning before rebuilding/repricing the node.
            if node.estimatedLowerBound >= self.incumbentObjective - self.cutTolerance:
                self.nodesPrunedByBound += 1
                if outputFlag >= 2:
                    print(
                        f'[PRUNE {node.nodeId:05d}] inherited LB='
                        f'{_format_number(node.estimatedLowerBound, 3)} >= UB='
                        f'{_format_number(self.incumbentObjective, 3)}'
                    )
                activeNodeSafeBound = None
                continue

            self.nodesProcessed += 1
            if outputFlag:
                self._printNodeHeader(node, len(queue))

            # A newly discovered global BSS cut can change the same node LP.
            # Keep one persistent Phase-II RMP for the entire lifetime of this
            # active node, including BSS-cut and SRC re-pricing rounds.
            phaseTwoMaster = None
            while True:
                if self._timeExpired():
                    self._interrupted = True
                    self._interruptionStatus = 'TIME_LIMIT'
                    break

                status, nodeLowerBound, phaseTwoMaster = self._solveNodeLP(
                    node.branchingState,
                    outputFlag=outputFlag,
                    nodeDepth=node.depth,
                    phaseTwoMaster=phaseTwoMaster,
                )

                if status == 'INFEASIBLE':
                    self.nodesPrunedInfeasible += 1
                    if outputFlag:
                        print('  [PRUNE] node infeasible after exact pricing')
                    break

                if status != 'OPTIMAL':
                    self._interrupted = True
                    self._interruptionStatus = status
                    if outputFlag:
                        print(f'  [STOP] node LP status={status}')
                    break

                activeNodeSafeBound = max(activeNodeSafeBound, float(nodeLowerBound))

                if outputFlag:
                    self._printLpLine(nodeLowerBound)

                if nodeLowerBound >= self.incumbentObjective - self.cutTolerance:
                    self.nodesPrunedByBound += 1
                    if outputFlag:
                        print(
                            f'  [PRUNE] exact node LB={_format_number(nodeLowerBound, 3)} '
                            f'>= UB={_format_number(self.incumbentObjective, 3)}'
                        )
                    break

                support = self.master.getPositiveX(
                    tolerance=self.brancher.tolerance * 0.1
                )
                decision = self.brancher.choose(
                    self.master,
                    branchingState=node.branchingState,
                    support=support,
                )

                if decision is None:
                    if not self.brancher.isIntegral(self.master, support=support):
                        raise RuntimeError(
                            'No physical-arc or customer-precedence branch decision was found '
                            'although the slot-column solution is fractional.'
                        )

                    action = self._processIntegralMasterSolution(
                        nodeLowerBound=nodeLowerBound,
                        outputFlag=outputFlag,
                    )
                    if action == 'REPRICE':
                        # The BSS cut is global and changes this node's RMP.
                        # Keep the persistent master, synchronize the new cut,
                        # then return to the exact RMP -> pricing loop.
                        continue
                    break

                # Fractional node: branch first on complete-route physical arcs.
                # If repeated visits to the same BSS create an arc-incidence
                # degeneracy, fall back to a customer precedence indicator.
                # This preserves exact branching without a station-entry label state.
                slot = decision.slot
                flow = decision.flow

                children = []
                childDescriptions = []
                for value in (0, 1):
                    if isinstance(decision, ArcBranchDecision):
                        childState = node.branchingState.childArc(
                            slot=slot,
                            arc=decision.arc,
                            value=value,
                        )
                    elif isinstance(decision, PrecedenceBranchDecision):
                        childState = node.branchingState.childPrecedence(
                            slot=slot,
                            pair=decision.pair,
                            value=value,
                        )
                    else:
                        raise TypeError(f'Unsupported branch decision: {type(decision)!r}')

                    if childState is None:
                        childDescriptions.append(f'value={value}:structurally infeasible')
                        continue

                    if self.deduplicateSymmetricNodes:
                        stateKey = childState.canonicalKey()
                        if stateKey in seenBranchStateKeys:
                            self.symmetricNodesSkipped += 1
                            self.profiler.increment('symmetricNodesSkipped')
                            childDescriptions.append(f'value={value}:symmetric duplicate')
                            continue
                        seenBranchStateKeys.add(stateKey)

                    child = _QueueNode(
                        estimatedLowerBound=float(nodeLowerBound),
                        nodeId=nextNodeId,
                        depth=node.depth + 1,
                        branchingState=childState,
                        parentId=node.nodeId,
                    )
                    children.append(child)
                    childDescriptions.append(f'value={value}:node {nextNodeId}')
                    nextNodeId += 1

                for child in children:
                    self.nodesCreated += 1
                    heapq.heappush(queue, child)

                self.branchCount += 1

                if outputFlag:
                    if isinstance(decision, ArcBranchDecision):
                        print(
                            f'  [BRANCH] V{slot + 1} physical arc '
                            f'{self._formatArc(decision.arc)} has flow={flow:.6f}'
                        )
                    else:
                        i, j = decision.pair
                        print(
                            f'  [BRANCH-ORDER] V{slot + 1} precedence '
                            f'C{i} before C{j} has flow={flow:.6f}'
                        )
                    print('           -> ' + ' | '.join(childDescriptions))
                break

            if self._interrupted:
                break
            activeNodeSafeBound = None

        self.runtime = time.time() - startWall

        # Every queued node stores a valid inherited lower bound. If the
        # algorithm is interrupted while processing a popped node, also keep a
        # safe lower bound for that active node; otherwise the reported global
        # bound could be too optimistic about proof progress.
        boundCandidates = [
            float(queuedNode.estimatedLowerBound)
            for queuedNode in queue
        ]
        if activeNodeSafeBound is not None:
            boundCandidates.append(float(activeNodeSafeBound))

        if boundCandidates:
            self.bestBound = min(boundCandidates)
        elif math.isfinite(self.incumbentObjective):
            self.bestBound = self.incumbentObjective
        else:
            self.bestBound = float('inf')

        if self._interrupted:
            self.status = self._interruptionStatus
        elif math.isfinite(self.incumbentObjective):
            self.status = 'OPTIMAL'
            self.bestBound = self.incumbentObjective
        else:
            self.status = 'INFEASIBLE'

        self.lowerBound = self.bestBound

        if outputFlag:
            # Level 1 keeps the final output compact; level 2 additionally
            # prints the detailed event-by-event BSS schedule.
            self.printResult(showBssSchedule=(outputFlag >= 2))

        return self

    def getResult(self):
        gap = None
        if math.isfinite(self.incumbentObjective) and math.isfinite(self.bestBound):
            denominator = max(1.0, abs(self.incumbentObjective))
            gap = max(0.0, self.incumbentObjective - self.bestBound) / denominator

        routes = self._getIncumbentRouteDetails()

        return {
            'status': self.status,
            'optimalityProven': self.status == 'OPTIMAL',
            'searchStrategy': self.searchStrategy,
            'objective': self.incumbentObjective,
            'bestBound': self.bestBound,
            'relativeGap': gap,
            'runtime': self.runtime,
            'dominanceMode': self.dominanceMode,
            'warmStartColumnsAdded': self.warmStartColumnsAdded,
            'warmStartBuildTime': self.warmStartBuildTime,
            'warmStartFeasibleStarts': (
                self.warmStartResult.feasibleStarts
                if self.warmStartResult is not None else 0
            ),
            'warmStartBestRouteCount': (
                len(self.warmStartResult.bestRoutes)
                if self.warmStartResult is not None else 0
            ),
            'columnCount': len(self.columns),
            'cutCount': len(self.bssCuts),
            'srcCutCount': len(self.srcCuts),
            'srcCutsAdded': self.srcCutsAdded,
            'srcSeparationRounds': self.srcSeparationRounds,
            'srcCuts': [cut.customers for cut in self.srcCuts],
            'nodesCreated': self.nodesCreated,
            'nodesProcessed': self.nodesProcessed,
            'nodesPrunedByBound': self.nodesPrunedByBound,
            'nodesPrunedInfeasible': self.nodesPrunedInfeasible,
            'integralNodes': self.integralNodes,
            'branchCount': self.branchCount,
            'phaseOneIterations': self.totalPhaseOneIterations,
            'phaseTwoIterations': self.totalPhaseTwoIterations,
            'phaseOneSkipped': self.phaseOneSkipped,
            'phaseOneActivated': self.phaseOneActivated,
            'symmetricNodesSkipped': self.symmetricNodesSkipped,
            'shortRouteColumnsAdded': self.shortRouteColumnsAdded,
            'shortRouteBuildTime': self.shortRouteBuildTime,
            'performanceProfile': self.profiler.snapshot(),
            'cacheStatistics': collectCacheStatistics(self.routeEvaluator, self.bssScheduler),
            'pricingStatistics': getattr(self.pricing, 'lastStatistics', None),
            'selectedRouteTasks': [tuple(c.tasks) for c in self.incumbentColumns],
            'routes': routes,
            'selectedPhysicalRoutes': [tuple(route['nodes']) for route in routes],
            'bssStatus': (
                self.incumbentBssResult.status
                if self.incumbentBssResult is not None
                else None
            ),
            'bssEvents': (
                self.incumbentBssResult.events
                if self.incumbentBssResult is not None
                else []
            ),
            'exactnessNote': (
                'OPTIMAL is reported only after the branch queue is exhausted with exact '
                'complete-route pricing (customers elementary, physical BSS repeatable), '
                'active SRC duals handled in pricing, and proven-optimal fixed-event BSS scheduling.'
            ),
        }
