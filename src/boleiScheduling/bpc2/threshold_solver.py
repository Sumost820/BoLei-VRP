import heapq
import math
import time
from dataclasses import dataclass, field

from boleiScheduling.bpc1.bss import ExactBssScheduler
from boleiScheduling.bpc1.columnManager import ColumnManager
from boleiScheduling.bpc1.profiler import PerformanceProfiler
from boleiScheduling.bpc1.savings import SavingsWarmStart
from boleiScheduling.bpc1.swap_dp import ExactCompleteRouteEvaluator
from .threshold_branch import (
    ThresholdArcBranchDecision,
    ThresholdArcFlowBrancher,
    ThresholdBranchState,
    ThresholdRouteBranchDecision,
)
from .threshold_master import (
    ThresholdBssNoGoodCut,
    ThresholdRestrictedMasterProblem,
)
from .threshold_cuts import ThresholdThreeRowSrcSeparator
from .threshold_pricing import (
    ExactThresholdLabelingPricing,
    HeuristicThresholdPricing,
)


@dataclass(order=True)
class _ThresholdNode:
    priority: float
    serial: int
    depth: int = field(compare=False)
    state: object = field(compare=False)
    inheritedLowerBound: float = field(compare=False, default=0.0)


@dataclass
class ThresholdFeasibilityResult:
    threshold: float
    feasible: bool
    proven: bool
    status: str
    routeCount: object = None
    lpLowerBound: object = None
    selectedColumns: list = field(default_factory=list)
    bssResult: object = None
    processedNodes: int = 0
    createdNodes: int = 0
    branches: int = 0
    bssCuts: int = 0
    columns: int = 0
    runtime: float = 0.0
    pricingCalls: int = 0
    pricingLabels: int = 0
    heuristicPricingCalls: int = 0
    heuristicColumns: int = 0
    masterBuildTime: float = 0.0
    masterSolveTime: float = 0.0
    masterTime: float = 0.0
    heuristicPricingTime: float = 0.0
    exactPricingTime: float = 0.0
    pricingTime: float = 0.0
    bssSchedulingTime: float = 0.0
    masterOptimizeCalls: int = 0
    srcCuts: int = 0
    totalCuts: int = 0
    srcSeparationTime: float = 0.0
    srcSeparationCalls: int = 0


@dataclass
class ThresholdMakespanResult:
    status: str
    provenWithinTolerance: bool
    lowerBound: float
    upperBound: float
    objective: float
    absoluteGap: float
    relativeGap: float
    routes: list
    bssResult: object
    outerIterations: int
    runtime: float
    thresholdHistory: list
    masterBuildTime: float = 0.0
    masterSolveTime: float = 0.0
    masterTime: float = 0.0
    heuristicPricingTime: float = 0.0
    exactPricingTime: float = 0.0
    pricingTime: float = 0.0
    bssSchedulingTime: float = 0.0
    masterOptimizeCalls: int = 0
    pricingCalls: int = 0
    heuristicPricingCalls: int = 0
    pricingLabels: int = 0
    heuristicColumns: int = 0
    bssCuts: int = 0
    srcCuts: int = 0
    totalCuts: int = 0
    srcSeparationTime: float = 0.0
    srcSeparationCalls: int = 0


class ThresholdFeasibilityBpcSolver:
    """Exact BPC feasibility solver for one fixed makespan threshold."""

    def __init__(
        self,
        modelData,
        columnManager=None,
        routeEvaluator=None,
        bssScheduler=None,
        profiler=None,
        maxColumnsPerRound=100,
        useHeuristicPricing=True,
        maxHeuristicColumnsPerRound=None,
        heuristicMaxSeedColumns=24,
        heuristicMaxInsertionCustomersPerSeed=10,
        heuristicMaxReplacementCustomersPerSeed=5,
        heuristicMaxReplacementPositionsPerSeed=5,
        heuristicGreedyStarts=8,
        heuristicGreedyCandidateLimit=16,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
        phaseOneTolerance=1e-8,
        integralityTolerance=1e-7,
        bssTolerance=1e-7,
        useSrcCuts=False,
        srcRootOnly=True,
        srcViolationTolerance=1e-7,
        maxSrcCutsPerRound=5,
        maxSrcCutsPerThreshold=20,
        maxNodes=None,
        timeLimit=None,
        bssTimeLimit=None,
    ):
        self.data = modelData
        self.profiler = profiler or PerformanceProfiler()
        self.routeEvaluator = routeEvaluator or ExactCompleteRouteEvaluator(
            modelData, profiler=self.profiler
        )
        self.columnManager = columnManager or ColumnManager()
        self.bssScheduler = bssScheduler or ExactBssScheduler(
            modelData, profiler=self.profiler
        )
        self.pricing = ExactThresholdLabelingPricing(
            modelData,
            self.routeEvaluator,
            reducedCostTolerance=reducedCostTolerance,
            dominanceTolerance=dominanceTolerance,
        )
        self.heuristicPricing = HeuristicThresholdPricing(
            modelData,
            self.routeEvaluator,
            reducedCostTolerance=reducedCostTolerance,
            maxSeedColumns=heuristicMaxSeedColumns,
            maxInsertionCustomersPerSeed=heuristicMaxInsertionCustomersPerSeed,
            maxReplacementCustomersPerSeed=heuristicMaxReplacementCustomersPerSeed,
            maxReplacementPositionsPerSeed=heuristicMaxReplacementPositionsPerSeed,
            greedyStarts=heuristicGreedyStarts,
            greedyCandidateLimit=heuristicGreedyCandidateLimit,
        )
        self.brancher = ThresholdArcFlowBrancher(
            modelData, integralityTolerance=integralityTolerance
        )
        self.useSrcCuts = bool(useSrcCuts)
        self.srcRootOnly = bool(srcRootOnly)
        self.maxSrcCutsPerThreshold = (
            None if maxSrcCutsPerThreshold is None
            else max(0, int(maxSrcCutsPerThreshold))
        )
        self.srcSeparator = ThresholdThreeRowSrcSeparator(
            modelData,
            violationTolerance=srcViolationTolerance,
            maxCutsPerRound=maxSrcCutsPerRound,
        )
        self.maxColumnsPerRound = int(maxColumnsPerRound)
        self.useHeuristicPricing = bool(useHeuristicPricing)
        self.maxHeuristicColumnsPerRound = (
            self.maxColumnsPerRound
            if maxHeuristicColumnsPerRound is None
            else max(1, int(maxHeuristicColumnsPerRound))
        )
        self.reducedCostTolerance = float(reducedCostTolerance)
        self.phaseOneTolerance = float(phaseOneTolerance)
        self.integralityTolerance = float(integralityTolerance)
        self.bssTolerance = float(bssTolerance)
        self.maxNodes = maxNodes
        self.timeLimit = timeLimit
        self.bssTimeLimit = bssTimeLimit

        # ``_pricingCalls`` remains the number of exact ESPPRC calls for
        # backwards-compatible diagnostics.  Heuristic calls are tracked
        # separately.
        self._pricingCalls = 0
        self._pricingLabels = 0
        self._heuristicPricingCalls = 0
        self._heuristicColumns = 0

        # Per-fixed-threshold timing/cut statistics.
        self._masterBuildTime = 0.0
        self._masterSolveTime = 0.0
        self._masterOptimizeCalls = 0
        self._heuristicPricingTime = 0.0
        self._exactPricingTime = 0.0
        self._bssSchedulingTime = 0.0
        self._srcCutCount = 0
        self._srcSeparationTime = 0.0
        self._srcSeparationCalls = 0
        self._activeOutputFlag = 0

    def _remainingTime(self, started):
        if self.timeLimit is None:
            return None
        return max(0.0, float(self.timeLimit) - (time.perf_counter() - started))

    def _protectedSignatures(self, branchState, bssCuts):
        result = set(branchState.protectedSignatures())
        for cut in bssCuts:
            result.update(cut.routeSignatures)
        return result

    def _addCandidates(self, candidates):
        return self.columnManager.addColumns([candidate.column for candidate in candidates])

    def _heuristicSeedColumns(self, master, tolerance=1e-10):
        """Return promising current columns for local heuristic pricing.

        Positive LP columns are preferred.  Degenerate/basic or near-zero-RC
        columns are then used as additional seeds when available.  Access to
        Gurobi basis/reduced-cost attributes is deliberately best-effort: the
        heuristic must never affect exactness if those attributes are absent.
        """
        ranked = []
        for routeIndex, var in master.x.items():
            column = master.columns[routeIndex]
            if column.isEmpty:
                continue
            try:
                value = float(var.X)
            except Exception:
                value = 0.0
            try:
                reducedCost = abs(float(var.RC))
            except Exception:
                reducedCost = float('inf')
            try:
                isBasic = int(var.VBasis) == 0
            except Exception:
                isBasic = False
            priorityClass = 0 if value > tolerance else (1 if isBasic else 2)
            ranked.append((
                priorityClass,
                -value,
                reducedCost,
                float(column.duration),
                routeIndex,
                column,
            ))
        ranked.sort(key=lambda item: item[:-1])
        limit = self.heuristicPricing.maxSeedColumns
        return [item[-1] for item in ranked[:limit]]

    def _runCgPhase(
        self,
        threshold,
        phase,
        branchState,
        bssCuts,
        srcCuts,
        started,
        outputFlag,
    ):
        masterBuildStarted = time.perf_counter()
        master = ThresholdRestrictedMasterProblem(
            modelData=self.data,
            columnManager=self.columnManager,
            threshold=threshold,
            phase=phase,
            branchingState=branchState,
            bssCuts=bssCuts,
            srcCuts=srcCuts,
            profiler=self.profiler,
            outputFlag=1 if outputFlag >= 3 else 0,
        )
        self._masterBuildTime += time.perf_counter() - masterBuildStarted

        iteration = 0
        while True:
            remaining = self._remainingTime(started)
            if remaining is not None and remaining <= 0.0:
                return None, 'TIME_LIMIT'
            masterSolveStarted = time.perf_counter()
            master.optimize(timeLimit=remaining)
            masterSolveElapsed = time.perf_counter() - masterSolveStarted
            self._masterSolveTime += masterSolveElapsed
            self._masterOptimizeCalls += 1
            if master.isInfeasible() or not master.hasSolution():
                return master, 'INFEASIBLE_RMP'

            if phase == 1 and master.getPhaseOneArtificialValue() <= self.phaseOneTolerance:
                return master, 'PHASE1_FEASIBLE'

            duals = master.getDuals()

            # --------------------------------------------------------------
            # Fast heuristic pricing.  Any columns found here are valid
            # negative-RC columns, but failure to find one proves nothing.
            # Therefore exact labeling is called only after the heuristic
            # neighborhood is exhausted.
            # --------------------------------------------------------------
            if self.useHeuristicPricing:
                heuristicStarted = time.perf_counter()
                heuristicCandidates = self.heuristicPricing.price(
                    duals=duals,
                    existingSignatures=self.columnManager,
                    threshold=threshold,
                    phase=phase,
                    maxColumns=self.maxHeuristicColumnsPerRound,
                    branchingState=branchState,
                    seedColumns=self._heuristicSeedColumns(master),
                )
                heuristicElapsed = time.perf_counter() - heuristicStarted
                self._heuristicPricingTime += heuristicElapsed
                self._heuristicPricingCalls += 1

                if outputFlag >= 2:
                    hrc = (
                        heuristicCandidates[0].reducedCost
                        if heuristicCandidates else None
                    )
                    hstats = self.heuristicPricing.lastStatistics
                    print(
                        f'      [TH-HP-P{phase}] iter={iteration:03d} '
                        f'obj={master.getObjectiveValue():.6f} '
                        f'cols={len(self.columnManager)} '
                        + (f'bestRC={hrc:.6g} ' if hrc is not None else 'bestRC=none ')
                        + f'time={heuristicElapsed:.3f}s '
                        + (
                            f'evalSeq={hstats.evaluatedSequences} '
                            f'neg={len(heuristicCandidates)}'
                            if hstats is not None else ''
                        )
                    )

                if heuristicCandidates:
                    added = self._addCandidates(heuristicCandidates)
                    if added:
                        self._heuristicColumns += len(added)
                        master.syncColumns(self.columnManager)
                        iteration += 1
                        continue
                    # Numerical/duplicate safety: heuristic pricing is not a
                    # certificate, so fall through to exact pricing.

            # --------------------------------------------------------------
            # Exact ESPPRC pricing: this is the only routine allowed to
            # certify that no negative reduced-cost route exists.
            # --------------------------------------------------------------
            protected = self._protectedSignatures(branchState, bssCuts)
            exactPricingStarted = time.perf_counter()
            candidates = self.pricing.price(
                duals=duals,
                existingSignatures=self.columnManager,
                threshold=threshold,
                phase=phase,
                maxColumns=self.maxColumnsPerRound,
                branchingState=branchState,
                protectedSignatures=protected,
            )
            exactPricingElapsed = time.perf_counter() - exactPricingStarted
            self._exactPricingTime += exactPricingElapsed
            self._pricingCalls += 1
            if self.pricing.lastStatistics is not None:
                self._pricingLabels += self.pricing.lastStatistics.generatedLabels

            if outputFlag >= 2:
                rc = candidates[0].reducedCost if candidates else None
                art = master.getPhaseOneArtificialValue() if phase == 1 else None
                print(
                    f'      [TH-EP-P{phase}] iter={iteration:03d} '
                    f'obj={master.getObjectiveValue():.6f} '
                    + (f'art={art:.3e} ' if art is not None else '')
                    + f'cols={len(self.columnManager)} '
                    + f'time={exactPricingElapsed:.3f}s '
                    + (f'bestRC={rc:.6g}' if rc is not None else 'bestRC=none')
                )

            if not candidates:
                if phase == 1:
                    if master.getPhaseOneArtificialValue() > self.phaseOneTolerance:
                        return master, 'PHASE1_INFEASIBLE'
                    return master, 'PHASE1_FEASIBLE'
                return master, 'OPTIMAL_LP'

            added = self._addCandidates(candidates)
            if not added:
                # Exact pricing only returns new signatures.  Reaching this is a
                # safety guard against numerical duplicate loops.
                return master, 'NO_NEW_COLUMNS'
            master.syncColumns(self.columnManager)
            iteration += 1

    def _solveNodeLp(self, threshold, branchState, bssCuts, srcCuts, started, outputFlag):
        phase1, status1 = self._runCgPhase(
            threshold, 1, branchState, bssCuts, srcCuts, started, outputFlag
        )
        if status1 in ('TIME_LIMIT', 'INFEASIBLE_RMP', 'PHASE1_INFEASIBLE'):
            return None, status1

        phase2, status2 = self._runCgPhase(
            threshold, 2, branchState, bssCuts, srcCuts, started, outputFlag
        )
        return phase2, status2

    def solve(self, threshold, outputFlag=1, srcCuts=None):
        threshold = float(threshold)
        started = time.perf_counter()
        self._activeOutputFlag = int(outputFlag)

        # Reset per-threshold statistics in case this solver object is reused.
        self._pricingCalls = 0
        self._pricingLabels = 0
        self._heuristicPricingCalls = 0
        self._heuristicColumns = 0
        self._masterBuildTime = 0.0
        self._masterSolveTime = 0.0
        self._masterOptimizeCalls = 0
        self._heuristicPricingTime = 0.0
        self._exactPricingTime = 0.0
        self._bssSchedulingTime = 0.0
        self._srcSeparationTime = 0.0
        self._srcSeparationCalls = 0

        bssCuts = []
        bssCutKeys = set()
        srcCuts = list(srcCuts or ())
        srcCutKeys = {cut.key for cut in srcCuts}
        self._srcCutCount = len(srcCuts)
        queue = []
        serial = 0
        heapq.heappush(queue, _ThresholdNode(0.0, serial, 0, ThresholdBranchState.root(), 0.0))
        serial += 1
        processed = 0
        branches = 0
        bestLp = float('inf')

        while queue:
            if self.maxNodes is not None and processed >= self.maxNodes:
                return self._result(
                    threshold, False, False, 'NODE_LIMIT', [], None,
                    processed, serial, branches, bssCuts, started, bestLp,
                )
            remaining = self._remainingTime(started)
            if remaining is not None and remaining <= 0.0:
                return self._result(
                    threshold, False, False, 'TIME_LIMIT', [], None,
                    processed, serial, branches, bssCuts, started, bestLp,
                )

            node = heapq.heappop(queue)
            processed += 1
            if outputFlag:
                print(
                    f'    [TH-NODE {node.serial:05d}] depth={node.depth:<3d} '
                    f'queue={len(queue):<4d} threshold={threshold:.6f}'
                )

            # A BSS no-good cut can be generated at this node; keep resolving
            # the same branch state until it is either fractional, infeasible,
            # route-count-pruned, or BSS-feasible.
            while True:
                master, lpStatus = self._solveNodeLp(
                    threshold,
                    node.state,
                    bssCuts,
                    srcCuts,
                    started,
                    outputFlag,
                )
                if lpStatus == 'TIME_LIMIT':
                    return self._result(
                        threshold, False, False, 'TIME_LIMIT', [], None,
                        processed, serial, branches, bssCuts, started, bestLp,
                    )
                if master is None or lpStatus in ('INFEASIBLE_RMP', 'PHASE1_INFEASIBLE'):
                    break
                if lpStatus not in ('OPTIMAL_LP', 'NO_NEW_COLUMNS'):
                    break

                lb = master.getObjectiveValue()
                bestLp = min(bestLp, lb)
                if outputFlag:
                    pricingTime = self._heuristicPricingTime + self._exactPricingTime
                    totalCuts = len(bssCuts) + self._srcCutCount
                    print(
                        f'      [TH-LP] minRoutesLB={lb:.6f} | K={self.data.K} '
                        f'| cols={len(self.columnManager)} '
                        f'| cuts={totalCuts} (BSS={len(bssCuts)}, SRC={self._srcCutCount}) '
                        f'| master={self._masterBuildTime + self._masterSolveTime:.3f}s '
                        f'| pricing={pricingTime:.3f}s'
                    )

                # Objective is integer route count.  If the LP lower bound is
                # already strictly larger than K, this branch cannot be feasible.
                if lb > self.data.K + self.integralityTolerance:
                    break

                # ----------------------------------------------------------
                # Three-row subset-row cut separation.  We only separate on
                # a fully priced Phase-II LP, because adding an SRC changes
                # the dual vector and therefore invalidates the previous
                # pricing certificate.  If a cut is added, rebuild and re-price
                # the same branch node under the strengthened master.
                # ----------------------------------------------------------
                canSeparateSrc = (
                    self.useSrcCuts
                    and (not self.srcRootOnly or node.depth == 0)
                    and (
                        self.maxSrcCutsPerThreshold is None
                        or len(srcCuts) < self.maxSrcCutsPerThreshold
                    )
                )
                if canSeparateSrc:
                    srcStarted = time.perf_counter()
                    violations = self.srcSeparator.separate(
                        master, existingKeys=srcCutKeys
                    )
                    self._srcSeparationTime += time.perf_counter() - srcStarted
                    self._srcSeparationCalls += 1

                    if self.maxSrcCutsPerThreshold is not None:
                        remainingSrcSlots = max(
                            0, self.maxSrcCutsPerThreshold - len(srcCuts)
                        )
                        violations = violations[:remainingSrcSlots]

                    if violations:
                        for item in violations:
                            if item.cut.key in srcCutKeys:
                                continue
                            srcCutKeys.add(item.cut.key)
                            srcCuts.append(item.cut)
                        self._srcCutCount = len(srcCuts)
                        if outputFlag:
                            bestViolation = violations[0]
                            print(
                                f'      [TH-SRC] +{len(violations)} cuts '
                                f'(total={len(srcCuts)}, '
                                f'maxViol={bestViolation.violation:.6g}, '
                                f'S={bestViolation.cut.customers})'
                            )
                        continue

                support = master.getPositiveX(tolerance=1e-10)
                if master.isIntegral(self.integralityTolerance):
                    selected = master.getSelectedColumns(self.integralityTolerance)
                    selectedColumns = [column for _, column, value in selected if value >= 1.0 - self.integralityTolerance]
                    if len(selectedColumns) > self.data.K:
                        break

                    localBssLimit = self.bssTimeLimit
                    if self.timeLimit is not None:
                        rem = self._remainingTime(started)
                        if rem is not None:
                            localBssLimit = rem if localBssLimit is None else min(localBssLimit, rem)
                    bssStarted = time.perf_counter()
                    bssResult = self.bssScheduler.solve(
                        selectedColumns,
                        outputFlag=0,
                        timeLimit=localBssLimit,
                        mipGap=0.0,
                    )
                    self._bssSchedulingTime += time.perf_counter() - bssStarted
                    if not bssResult.provenOptimal:
                        return self._result(
                            threshold, False, False, 'BSS_NOT_PROVEN', selectedColumns,
                            bssResult, processed, serial, branches, bssCuts,
                            started, bestLp,
                        )

                    if outputFlag:
                        print(
                            f'      [TH-BSS] routes={len(selectedColumns)} '
                            f'Cmax={bssResult.objective:.6f} threshold={threshold:.6f}'
                        )
                    if bssResult.objective <= threshold + self.bssTolerance:
                        return self._result(
                            threshold, True, True, 'FEASIBLE', selectedColumns,
                            bssResult, processed, serial, branches, bssCuts,
                            started, bestLp,
                        )

                    cut = ThresholdBssNoGoodCut.canonical(
                        column.signature for column in selectedColumns
                    )
                    if cut.key in bssCutKeys:
                        # Same integral route set is still returned despite its
                        # no-good row: treat as numerical failure, not infeasible.
                        return self._result(
                            threshold, False, False, 'BSS_CUT_STALL', selectedColumns,
                            bssResult, processed, serial, branches, bssCuts,
                            started, bestLp,
                        )
                    bssCutKeys.add(cut.key)
                    bssCuts.append(cut)
                    if outputFlag:
                        print(
                            f'      [TH-BSS-CUT #{len(bssCuts):04d}] '
                            f'sum(x_r in incumbent) <= {len(selectedColumns)-1}'
                        )
                    # Rebuild/re-price this same node under the new global cut.
                    continue

                decision = self.brancher.choose(master, node.state, support=support)
                if decision is None:
                    return self._result(
                        threshold, False, False, 'FRACTIONAL_NO_BRANCH', [], None,
                        processed, serial, branches, bssCuts, started, bestLp,
                    )

                if isinstance(decision, ThresholdArcBranchDecision):
                    left = node.state.childForbidArc(decision.arc)
                    right = node.state.childForceArc(self.data, decision.arc)
                    description = f'arc {decision.arc[0]}->{decision.arc[1]} flow={decision.flow:.6f}'
                elif isinstance(decision, ThresholdRouteBranchDecision):
                    left = node.state.childRoute(decision.signature, 0)
                    right = node.state.childRoute(decision.signature, 1)
                    description = f'route flow={decision.value:.6f} sig={decision.signature}'
                else:
                    raise RuntimeError('unknown threshold branch decision')

                branches += 1
                if outputFlag:
                    print(f'      [TH-BRANCH] {description}')
                for childState in (left, right):
                    heapq.heappush(
                        queue,
                        _ThresholdNode(lb, serial, node.depth + 1, childState, lb),
                    )
                    serial += 1
                break

        return self._result(
            threshold, False, True, 'INFEASIBLE', [], None,
            processed, serial, branches, bssCuts, started, bestLp,
        )

    def _result(
        self,
        threshold,
        feasible,
        proven,
        status,
        columns,
        bssResult,
        processed,
        created,
        branches,
        bssCuts,
        started,
        bestLp,
    ):
        result = ThresholdFeasibilityResult(
            threshold=float(threshold),
            feasible=bool(feasible),
            proven=bool(proven),
            status=status,
            routeCount=(len(columns) if columns else None),
            lpLowerBound=(None if not math.isfinite(bestLp) else float(bestLp)),
            selectedColumns=list(columns),
            bssResult=bssResult,
            processedNodes=int(processed),
            createdNodes=int(created),
            branches=int(branches),
            bssCuts=len(bssCuts),
            columns=len(self.columnManager),
            runtime=time.perf_counter() - started,
            pricingCalls=self._pricingCalls,
            pricingLabels=self._pricingLabels,
            heuristicPricingCalls=self._heuristicPricingCalls,
            heuristicColumns=self._heuristicColumns,
            masterBuildTime=float(self._masterBuildTime),
            masterSolveTime=float(self._masterSolveTime),
            masterTime=float(self._masterBuildTime + self._masterSolveTime),
            heuristicPricingTime=float(self._heuristicPricingTime),
            exactPricingTime=float(self._exactPricingTime),
            pricingTime=float(self._heuristicPricingTime + self._exactPricingTime),
            bssSchedulingTime=float(self._bssSchedulingTime),
            masterOptimizeCalls=int(self._masterOptimizeCalls),
            srcCuts=int(self._srcCutCount),
            totalCuts=int(len(bssCuts) + self._srcCutCount),
            srcSeparationTime=float(self._srcSeparationTime),
            srcSeparationCalls=int(self._srcSeparationCalls),
        )
        if self._activeOutputFlag >= 1:
            print(
                f'    [TH-STATS] T={float(threshold):.6f} '
                f'master={result.masterTime:.3f}s '
                f'(build={result.masterBuildTime:.3f}s, solve={result.masterSolveTime:.3f}s, '
                f'calls={result.masterOptimizeCalls}) | '
                f'pricing={result.pricingTime:.3f}s '
                f'(HP={result.heuristicPricingTime:.3f}s, EP={result.exactPricingTime:.3f}s) | '
                f'BSS={result.bssSchedulingTime:.3f}s | '
                f'SRCsep={result.srcSeparationTime:.3f}s/{result.srcSeparationCalls} | '
                f'cuts={result.totalCuts} (BSS={result.bssCuts}, SRC={result.srcCuts})'
            )
        return result


class ThresholdMakespanBpcSolver:
    """Vehicle-free min-makespan solver using an outer threshold search.

    The outer search is continuous bisection and therefore proves the objective
    within ``thresholdTolerance`` rather than claiming a finite exact T unless
    the supplied tolerance is zero and a problem-specific discrete threshold
    search is added.  Every fixed-threshold feasibility decision itself is an
    exact branch-and-price-and-cut solve (subject to optional time/node limits).
    """

    def __init__(
        self,
        modelData,
        thresholdTolerance=1e-3,
        relativeTolerance=1e-6,
        maxOuterIterations=60,
        maxColumnsPerRound=100,
        useHeuristicPricing=True,
        maxHeuristicColumnsPerRound=None,
        heuristicMaxSeedColumns=24,
        heuristicMaxInsertionCustomersPerSeed=10,
        heuristicMaxReplacementCustomersPerSeed=5,
        heuristicMaxReplacementPositionsPerSeed=5,
        heuristicGreedyStarts=8,
        heuristicGreedyCandidateLimit=16,
        useSrcCuts=False,
        srcRootOnly=True,
        srcViolationTolerance=1e-7,
        maxSrcCutsPerRound=5,
        maxSrcCutsPerThreshold=100,
        useSavingsWarmStart=True,
        savingsStarts=12,
        savingsSeed=1,
        savingsRandomization=0.20,
        maxWarmStartColumns=2000,
        timeLimit=None,
        maxNodesPerThreshold=None,
        bssTimeLimit=None,
    ):
        self.data = modelData
        self.thresholdTolerance = float(thresholdTolerance)
        self.relativeTolerance = float(relativeTolerance)
        self.maxOuterIterations = int(maxOuterIterations)
        self.maxColumnsPerRound = int(maxColumnsPerRound)
        self.useHeuristicPricing = bool(useHeuristicPricing)
        self.maxHeuristicColumnsPerRound = maxHeuristicColumnsPerRound
        self.heuristicMaxSeedColumns = int(heuristicMaxSeedColumns)
        self.heuristicMaxInsertionCustomersPerSeed = int(heuristicMaxInsertionCustomersPerSeed)
        self.heuristicMaxReplacementCustomersPerSeed = int(heuristicMaxReplacementCustomersPerSeed)
        self.heuristicMaxReplacementPositionsPerSeed = int(heuristicMaxReplacementPositionsPerSeed)
        self.heuristicGreedyStarts = int(heuristicGreedyStarts)
        self.heuristicGreedyCandidateLimit = int(heuristicGreedyCandidateLimit)
        self.useSrcCuts = bool(useSrcCuts)
        self.srcRootOnly = bool(srcRootOnly)
        self.srcViolationTolerance = float(srcViolationTolerance)
        self.maxSrcCutsPerRound = maxSrcCutsPerRound
        self.maxSrcCutsPerThreshold = maxSrcCutsPerThreshold
        self.useSavingsWarmStart = bool(useSavingsWarmStart)
        self.savingsStarts = int(savingsStarts)
        self.savingsSeed = int(savingsSeed)
        self.savingsRandomization = float(savingsRandomization)
        self.maxWarmStartColumns = maxWarmStartColumns
        self.timeLimit = timeLimit
        self.maxNodesPerThreshold = maxNodesPerThreshold
        self.bssTimeLimit = bssTimeLimit

        self.profiler = PerformanceProfiler()
        self.routeEvaluator = ExactCompleteRouteEvaluator(modelData, profiler=self.profiler)
        self.columnManager = ColumnManager()
        self.bssScheduler = ExactBssScheduler(modelData, profiler=self.profiler)
        self.warmStartResult = None
        self._activeOutputFlag = 0

    def _remainingTime(self, started):
        if self.timeLimit is None:
            return None
        return max(0.0, float(self.timeLimit) - (time.perf_counter() - started))

    def _initializeWarmStart(self, outputFlag):
        if not self.useSavingsWarmStart:
            return None, None
        builder = SavingsWarmStart(
            modelData=self.data,
            routeEvaluator=self.routeEvaluator,
            starts=self.savingsStarts,
            seed=self.savingsSeed,
            randomization=self.savingsRandomization,
            maxInitialColumns=self.maxWarmStartColumns,
        )
        result = builder.build()
        self.warmStartResult = result
        self.columnManager.addColumns(
            column for column in result.columns if not column.isEmpty
        )
        if outputFlag:
            print(
                f'[TH-WARM] starts={result.startsAttempted} feasibleStarts={result.feasibleStarts} '
                f'bestRoutes={len(result.bestRoutes)} columns={len(self.columnManager)} '
                f'time={result.buildTime:.3f}s'
            )
        if result.bestRoutes and len(result.bestRoutes) <= self.data.K:
            bss = self.bssScheduler.solve(result.bestRoutes, outputFlag=0, mipGap=0.0)
            if bss.provenOptimal:
                if outputFlag:
                    print(f'[TH-WARM-UB] BSS-feasible Cmax={bss.objective:.6f}')
                return float(bss.objective), (list(result.bestRoutes), bss)
        return None, None

    def _makeFeasibilitySolver(self, remainingTime):
        return ThresholdFeasibilityBpcSolver(
            modelData=self.data,
            columnManager=self.columnManager,
            routeEvaluator=self.routeEvaluator,
            bssScheduler=self.bssScheduler,
            profiler=self.profiler,
            maxColumnsPerRound=self.maxColumnsPerRound,
            useHeuristicPricing=self.useHeuristicPricing,
            maxHeuristicColumnsPerRound=self.maxHeuristicColumnsPerRound,
            heuristicMaxSeedColumns=self.heuristicMaxSeedColumns,
            heuristicMaxInsertionCustomersPerSeed=self.heuristicMaxInsertionCustomersPerSeed,
            heuristicMaxReplacementCustomersPerSeed=self.heuristicMaxReplacementCustomersPerSeed,
            heuristicMaxReplacementPositionsPerSeed=self.heuristicMaxReplacementPositionsPerSeed,
            heuristicGreedyStarts=self.heuristicGreedyStarts,
            heuristicGreedyCandidateLimit=self.heuristicGreedyCandidateLimit,
            useSrcCuts=self.useSrcCuts,
            srcRootOnly=self.srcRootOnly,
            srcViolationTolerance=self.srcViolationTolerance,
            maxSrcCutsPerRound=self.maxSrcCutsPerRound,
            maxSrcCutsPerThreshold=self.maxSrcCutsPerThreshold,
            maxNodes=self.maxNodesPerThreshold,
            timeLimit=remainingTime,
            bssTimeLimit=self.bssTimeLimit,
        )

    def solve(self, lowerBound=0.0, upperBound=None, outputFlag=1):
        self._activeOutputFlag = int(outputFlag)
        started = time.perf_counter()
        warmUb, warmIncumbent = self._initializeWarmStart(outputFlag)

        lb = max(0.0, float(lowerBound))
        incumbentRoutes = []
        incumbentBss = None
        if warmIncumbent is not None:
            incumbentRoutes, incumbentBss = warmIncumbent

        if upperBound is not None:
            ub = float(upperBound)
            if warmUb is not None:
                ub = min(ub, warmUb)
        elif warmUb is not None:
            ub = float(warmUb)
        else:
            # ModelData.M is only a starting bracket.  If it is not feasible we
            # expand geometrically below until a proven feasible threshold is found.
            ub = max(1.0, float(self.data.M))

        history = []

        # If no warm solution proves the initial UB feasible, establish a valid
        # upper bracket before bisection.
        if warmUb is None or ub < warmUb - 1e-12:
            expansions = 0
            while True:
                remaining = self._remainingTime(started)
                if remaining is not None and remaining <= 0.0:
                    return self._outerResult('TIME_LIMIT', False, lb, ub, incumbentRoutes, incumbentBss, history, started)
                solver = self._makeFeasibilitySolver(remaining)
                check = solver.solve(ub, outputFlag=max(0, outputFlag - 1))
                history.append(check)
                if check.feasible:
                    incumbentRoutes = check.selectedColumns
                    incumbentBss = check.bssResult
                    ub = min(ub, float(check.bssResult.objective))
                    break
                if not check.proven:
                    return self._outerResult(check.status, False, lb, ub, incumbentRoutes, incumbentBss, history, started)
                lb = max(lb, ub)
                ub = max(ub * 2.0, ub + 1.0)
                expansions += 1
                if expansions >= 12:
                    return self._outerResult('NO_FEASIBLE_UPPER_BOUND', False, lb, ub, incumbentRoutes, incumbentBss, history, started)

        if lb > ub:
            lb = ub

        for iteration in range(self.maxOuterIterations):
            absGap = ub - lb
            relGap = absGap / max(1.0, abs(ub))
            if absGap <= self.thresholdTolerance or relGap <= self.relativeTolerance:
                return self._outerResult(
                    'OPTIMAL_WITHIN_TOLERANCE', True, lb, ub,
                    incumbentRoutes, incumbentBss, history, started,
                )

            remaining = self._remainingTime(started)
            if remaining is not None and remaining <= 0.0:
                return self._outerResult('TIME_LIMIT', False, lb, ub, incumbentRoutes, incumbentBss, history, started)

            threshold = 0.5 * (lb + ub)
            if outputFlag:
                print(
                    f'[TH-OUTER {iteration:02d}] LB={lb:.6f} UB={ub:.6f} '
                    f'test T={threshold:.6f}'
                )
            solver = self._makeFeasibilitySolver(remaining)
            result = solver.solve(threshold, outputFlag=outputFlag)
            history.append(result)

            if not result.proven:
                return self._outerResult(result.status, False, lb, ub, incumbentRoutes, incumbentBss, history, started)
            if result.feasible:
                incumbentRoutes = result.selectedColumns
                incumbentBss = result.bssResult
                # The exact BSS schedule gives a feasible objective that can be
                # tighter than the tested threshold.
                ub = min(ub, threshold, float(result.bssResult.objective))
            else:
                lb = max(lb, threshold)

        return self._outerResult(
            'OUTER_ITERATION_LIMIT', False, lb, ub,
            incumbentRoutes, incumbentBss, history, started,
        )

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

    @staticmethod
    def _formatNumber(value, digits=3):
        if value is None:
            return '-'
        value = float(value)
        if not math.isfinite(value):
            return 'inf' if value > 0 else '-inf'
        return f'{value:.{digits}f}'

    def getRouteDetails(self, result):
        """Return display-ready details for the final vehicle-free incumbent.

        Route numbering is for presentation only; it is not a vehicle index in
        the master problem.
        """
        routes = list(result.routes or ())
        bss = result.bssResult
        details = []
        for routeIndex, column in enumerate(routes):
            base = float(column.duration)
            waiting = 0.0
            completion = base
            if bss is not None:
                waiting = float(bss.routeWaiting.get(routeIndex, 0.0))
                completion = float(bss.routeCompletion.get(routeIndex, base + waiting))
            details.append({
                'routeNumber': routeIndex + 1,
                'routeIndex': routeIndex,
                'tasks': tuple(column.tasks),
                'nodes': tuple(column.nodes),
                'path': ' -> '.join(self._nodeLabel(node) for node in column.nodes),
                'baseDuration': base,
                'waitingTime': waiting,
                'completionTime': completion,
                'swapCount': int(column.stationVisitCount),
                'signature': tuple(column.signature),
            })
        return details

    def printResult(self, result, showBssSchedule=False, digits=3):
        width = 96
        print('\n' + '=' * width)
        print('THRESHOLD MAKESPAN BPC FINAL RESULT'.center(width))
        print('=' * width)
        print(f'Status        : {result.status}')
        print(f'Optimality    : {"PROVEN WITHIN TOLERANCE" if result.provenWithinTolerance else "NOT PROVEN"}')
        print(f'Objective     : {self._formatNumber(result.objective, digits)}')
        print(f'Lower bound   : {self._formatNumber(result.lowerBound, digits)}')
        print(f'Upper bound   : {self._formatNumber(result.upperBound, digits)}')
        print(f'Absolute gap  : {self._formatNumber(result.absoluteGap, digits)}')
        rel = result.relativeGap
        print(f'Relative gap  : {"-" if not math.isfinite(rel) else f"{100.0 * rel:.4f}%"}')
        print(f'Outer checks  : {result.outerIterations}')
        print(f'Runtime       : {self._formatNumber(result.runtime, 3)} s')

        print('\n' + '-' * width)
        print('TIMING / CUT STATISTICS')
        print('-' * width)
        print(
            f'Master total  : {self._formatNumber(result.masterTime, 3)} s '
            f'(build={self._formatNumber(result.masterBuildTime, 3)} s, '
            f'solve={self._formatNumber(result.masterSolveTime, 3)} s, '
            f'optimize calls={result.masterOptimizeCalls})'
        )
        print(
            f'Pricing total : {self._formatNumber(result.pricingTime, 3)} s '
            f'(heuristic={self._formatNumber(result.heuristicPricingTime, 3)} s, '
            f'exact={self._formatNumber(result.exactPricingTime, 3)} s)'
        )
        print(
            f'Pricing calls : heuristic={result.heuristicPricingCalls}, '
            f'exact={result.pricingCalls}, exact labels={result.pricingLabels}, '
            f'heuristic cols={result.heuristicColumns}'
        )
        print(f'BSS schedule  : {self._formatNumber(result.bssSchedulingTime, 3)} s')
        print(
            f'SRC separation: {self._formatNumber(result.srcSeparationTime, 3)} s '
            f'(calls={result.srcSeparationCalls})'
        )
        print(
            f'Cuts added    : {result.totalCuts} '
            f'(BSS no-good={result.bssCuts}, SRC={result.srcCuts})'
        )

        details = self.getRouteDetails(result)
        if details:
            print('\n' + '-' * width)
            print('FINAL ROUTES')
            print('-' * width)
            for route in details:
                print(
                    f"Route {route['routeNumber']:>2d} | tasks={list(route['tasks'])} | "
                    f"base={self._formatNumber(route['baseDuration'], digits)} | "
                    f"wait={self._formatNumber(route['waitingTime'], digits)} | "
                    f"completion={self._formatNumber(route['completionTime'], digits)} | "
                    f"swaps={route['swapCount']}"
                )
                print(f"          {route['path']}")
        else:
            print('\nNo BSS-feasible incumbent route set is available.')

        if showBssSchedule and result.bssResult is not None and result.bssResult.events:
            print('\n' + '-' * width)
            print('BSS EVENT SCHEDULE')
            print('-' * width)
            print(
                f"{'Evt':>4} {'Route':>6} {'After':>8} "
                f"{'Arrival':>12} {'Start':>12} {'End':>12} {'Wait':>10}"
            )
            print('-' * width)
            for eventNumber, event in enumerate(result.bssResult.events, start=1):
                after = self._nodeLabel(event['afterTask'])
                print(
                    f"{eventNumber:>4d} {event['routeIndex'] + 1:>6d} {after:>8} "
                    f"{self._formatNumber(event['arrivalTime'], digits):>12} "
                    f"{self._formatNumber(event['startTime'], digits):>12} "
                    f"{self._formatNumber(event['endTime'], digits):>12} "
                    f"{self._formatNumber(event['waitTime'], digits):>10}"
                )
        print('=' * width)

    def _outerResult(self, status, proven, lb, ub, routes, bssResult, history, started):
        gap = max(0.0, float(ub) - float(lb))
        rel = gap / max(1.0, abs(float(ub))) if math.isfinite(float(ub)) else float('inf')
        objective = float(ub) if bssResult is not None else float('inf')

        # Aggregate statistics across all fixed-threshold solves in this run.
        masterBuildTime = sum(item.masterBuildTime for item in history)
        masterSolveTime = sum(item.masterSolveTime for item in history)
        heuristicPricingTime = sum(item.heuristicPricingTime for item in history)
        exactPricingTime = sum(item.exactPricingTime for item in history)
        bssSchedulingTime = sum(item.bssSchedulingTime for item in history)
        masterOptimizeCalls = sum(item.masterOptimizeCalls for item in history)
        pricingCalls = sum(item.pricingCalls for item in history)
        heuristicPricingCalls = sum(item.heuristicPricingCalls for item in history)
        pricingLabels = sum(item.pricingLabels for item in history)
        heuristicColumns = sum(item.heuristicColumns for item in history)
        bssCuts = sum(item.bssCuts for item in history)
        srcCuts = sum(item.srcCuts for item in history)
        totalCuts = sum(item.totalCuts for item in history)
        srcSeparationTime = sum(item.srcSeparationTime for item in history)
        srcSeparationCalls = sum(item.srcSeparationCalls for item in history)

        result = ThresholdMakespanResult(
            status=status,
            provenWithinTolerance=bool(proven),
            lowerBound=float(lb),
            upperBound=float(ub),
            objective=objective,
            absoluteGap=gap,
            relativeGap=rel,
            routes=list(routes or ()),
            bssResult=bssResult,
            outerIterations=len(history),
            runtime=time.perf_counter() - started,
            thresholdHistory=list(history),
            masterBuildTime=float(masterBuildTime),
            masterSolveTime=float(masterSolveTime),
            masterTime=float(masterBuildTime + masterSolveTime),
            heuristicPricingTime=float(heuristicPricingTime),
            exactPricingTime=float(exactPricingTime),
            pricingTime=float(heuristicPricingTime + exactPricingTime),
            bssSchedulingTime=float(bssSchedulingTime),
            masterOptimizeCalls=int(masterOptimizeCalls),
            pricingCalls=int(pricingCalls),
            heuristicPricingCalls=int(heuristicPricingCalls),
            pricingLabels=int(pricingLabels),
            heuristicColumns=int(heuristicColumns),
            bssCuts=int(bssCuts),
            srcCuts=int(srcCuts),
            totalCuts=int(totalCuts),
            srcSeparationTime=float(srcSeparationTime),
            srcSeparationCalls=int(srcSeparationCalls),
        )
        if self._activeOutputFlag >= 1:
            self.printResult(
                result,
                showBssSchedule=(self._activeOutputFlag >= 2),
            )
        return result