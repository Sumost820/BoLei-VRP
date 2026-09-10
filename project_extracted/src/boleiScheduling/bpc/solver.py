import heapq
import math
import time
from dataclasses import dataclass, field

from .master import RestrictedMasterProblem
from .pricing import ExactEnumerativePricing, ExactLabelingPricing
from .swap_dp import ExactFixedSequenceEvaluator
from .bss import ExactBssScheduler
from .cuts import BssOptimalityCut
from .branch import ArcBranchingState, ArcFlowBrancher


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
    ):
        self.data = modelData
        self.sequenceEvaluator = ExactFixedSequenceEvaluator(modelData)
        self.columns = [self.sequenceEvaluator.evaluate(())]
        self.signatureToIndex = {(): 0}
        self.pricingMode = pricingMode

        if pricingMode == 'labeling':
            self.pricing = ExactLabelingPricing(
                modelData=modelData,
                sequenceEvaluator=self.sequenceEvaluator,
                reducedCostTolerance=reducedCostTolerance,
                dominanceTolerance=dominanceTolerance,
            )
        elif pricingMode == 'enumerative':
            self.pricing = ExactEnumerativePricing(
                modelData=modelData,
                sequenceEvaluator=self.sequenceEvaluator,
                maxEnumeratedTasks=maxEnumeratedTasks,
                reducedCostTolerance=reducedCostTolerance,
            )
        else:
            raise ValueError("pricingMode must be 'labeling' or 'enumerative'")

        self.maxColumnsPerRound = maxColumnsPerRound
        self.reducedCostTolerance = reducedCostTolerance
        self.phaseOneTolerance = phaseOneTolerance
        self.bssCuts = list(bssCuts or [])

        self.master = None
        self.phaseOneIterations = 0
        self.phaseTwoIterations = 0
        self.lowerBound = float('inf')
        self.status = 'NOT_SOLVED'

    def _existingSignatures(self):
        return set(self.signatureToIndex)

    def _addPricingCandidates(self, candidates):
        added = 0
        for candidate in candidates:
            signature = candidate.column.signature
            if signature in self.signatureToIndex:
                continue
            self.signatureToIndex[signature] = len(self.columns)
            self.columns.append(candidate.column)
            added += 1
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

    def _solveCGPhase(
        self,
        phase,
        outputFlag=0,
        branchingState=None,
        timeLimit=None,
    ):
        iteration = 0

        while True:
            iteration += 1
            columnsBefore = len(self.columns)

            master = RestrictedMasterProblem(
                self.data,
                self.columns,
                bssCuts=(self.bssCuts if phase == 2 else None),
                branchingState=branchingState,
            )
            model = master.solve(
                phase=phase,
                outputFlag=self._gurobiOutputFlag(outputFlag),
                binary=False,
                timeLimit=timeLimit,
            )

            try:
                from gurobipy import GRB
            except ImportError as error:
                raise ImportError('RootColumnGenerationSolver requires gurobipy') from error

            if model.Status == GRB.INFEASIBLE:
                self.master = master
                if outputFlag >= 2:
                    print(f'    [CG-P{phase}] iter={iteration:03d} | RMP infeasible')
                return iteration, 'INFEASIBLE'
            if model.Status != GRB.OPTIMAL:
                self.master = master
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

            duals = master.getDuals()
            candidates = self.pricing.price(
                duals=duals,
                existingSignatures=self._existingSignatures(),
                maxColumns=self.maxColumnsPerRound,
                branchingState=branchingState,
            )

            bestReducedCost = (
                min(candidate.reducedCost for candidate in candidates)
                if candidates
                else None
            )

            if not candidates:
                self.master = master
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

            added = self._addPricingCandidates(candidates)
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
                self.master = master
                return iteration, 'OPTIMAL'

    def solve(self, outputFlag=0):
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
            'phaseOneIterations': self.phaseOneIterations,
            'phaseTwoIterations': self.phaseTwoIterations,
            'pricingMode': self.pricingMode,
            'pricingStatistics': getattr(self.pricing, 'lastStatistics', None),
            'positiveX': self.master.getPositiveX() if self.master is not None and self.master.model.SolCount else [],
        }


class RootBpcCutSolver(RootColumnGenerationSolver):
    """V0.3 root-CG + exact fixed-combination BSS cut separation reference."""

    def __init__(
        self,
        modelData,
        pricingMode='labeling',
        maxEnumeratedTasks=9,
        maxColumnsPerRound=50,
        reducedCostTolerance=1e-8,
        dominanceTolerance=1e-10,
        phaseOneTolerance=1e-8,
        cutTolerance=1e-7,
        maxBssCuts=1000,
        maxExactRouteTasks=None,
        bssTimeLimit=None,
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
        )

        self.cutTolerance = cutTolerance
        self.maxBssCuts = maxBssCuts
        self.bssTimeLimit = bssTimeLimit
        self.bssScheduler = ExactBssScheduler(
            modelData=modelData,
            sequenceEvaluator=self.sequenceEvaluator,
            maxExactRouteTasks=maxExactRouteTasks,
        )

        self.cutByKey = {}
        self.integerMaster = None
        self.integerMasterT = float('inf')
        self.bssObjective = float('inf')
        self.bssResult = None
        self.selectedColumns = []
        self.cutRounds = 0

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
        return True

    def _solveIntegerRestrictedMaster(self, outputFlag=0):
        master = RestrictedMasterProblem(
            self.data,
            self.columns,
            bssCuts=self.bssCuts,
        )
        model = master.solve(phase=2, outputFlag=self._gurobiOutputFlag(outputFlag), binary=True)

        try:
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError('RootBpcCutSolver requires gurobipy') from error

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                'Integer restricted master was not solved to proven optimality; '
                f'Gurobi status={model.Status}'
            )

        self.integerMaster = master
        self.integerMasterT = master.getTValue()
        selected = master.getSelectedIntegerColumns(includeEmpty=False)
        self.selectedColumns = [column for _, _, column in selected]
        return self.selectedColumns

    def solve(self, outputFlag=0):
        self.phaseOneIterations, status = self._solveCGPhase(phase=1, outputFlag=outputFlag)
        if status != 'OPTIMAL':
            self.status = f'PHASE1_{status}'
            return self
        artificial = self.master.getPhaseOneArtificialValue()
        if artificial > self.phaseOneTolerance:
            self.status = 'INFEASIBLE_BASE_ROUTING'
            self.lowerBound = float('inf')
            return self

        totalPhaseTwoIterations = 0

        for cutRound in range(self.maxBssCuts + 1):
            cgIterations, status = self._solveCGPhase(phase=2, outputFlag=outputFlag)
            totalPhaseTwoIterations += cgIterations
            if status != 'OPTIMAL':
                self.status = f'PHASE2_{status}'
                self.phaseTwoIterations = totalPhaseTwoIterations
                return self
            self.lowerBound = self.master.getTValue()

            selectedColumns = self._solveIntegerRestrictedMaster(outputFlag=outputFlag)
            self.bssResult = self.bssScheduler.solve(
                selectedColumns,
                outputFlag=self._gurobiOutputFlag(outputFlag),
                timeLimit=self.bssTimeLimit,
                mipGap=0.0,
            )
            self.bssObjective = self.bssResult.objective

            if not self.bssResult.provenOptimal:
                self.status = 'BSS_SP_NOT_PROVEN_OPTIMAL'
                self.phaseTwoIterations = totalPhaseTwoIterations
                return self

            if self.integerMasterT >= self.bssObjective - self.cutTolerance:
                self.status = 'RMP_INTEGER_BSS_CONSISTENT'
                self.phaseTwoIterations = totalPhaseTwoIterations
                self.cutRounds = cutRound
                return self

            if cutRound >= self.maxBssCuts:
                self.status = 'BSS_CUT_LIMIT'
                self.phaseTwoIterations = totalPhaseTwoIterations
                self.cutRounds = cutRound
                return self

            added = self._addOrStrengthenCut(selectedColumns, self.bssObjective)
            if not added:
                raise RuntimeError(
                    'BSS underestimation persists but no stronger valid cut could be added'
                )
            self.cutRounds = cutRound + 1

        self.phaseTwoIterations = totalPhaseTwoIterations
        self.status = 'BSS_CUT_LIMIT'
        return self

    def getResult(self):
        base = super().getResult()
        base.update({
            'cutCount': len(self.bssCuts),
            'cutRounds': self.cutRounds,
            'integerMasterT': self.integerMasterT,
            'bssObjective': self.bssObjective,
            'selectedRouteTasks': [tuple(column.tasks) for column in self.selectedColumns],
            'bssStatus': self.bssResult.status if self.bssResult is not None else None,
            'bssEvents': self.bssResult.events if self.bssResult is not None else [],
            'exactnessNote': (
                'Exact-label root pricing and fixed-combination BSS separation are exact; '
                'global integer exactness still requires branch-and-price.'
            ),
        })
        return base


@dataclass(order=True)
class _QueueNode:
    estimatedLowerBound: float
    nodeId: int
    depth: int = field(compare=False)
    branchingState: object = field(compare=False)
    parentId: object = field(compare=False, default=None)


class BranchPriceCutSolver(RootColumnGenerationSolver):
    """
    V0.4 globally exact Branch-and-Price-and-Cut framework with slot-specific
    successor-arc flow branching.

    Branch variable
    ---------------
        f^k_ij = sum_r a^r_ij x_kr,
    where a^r_ij indicates adjacency in the ordered CUSTOMER sequence of route
    r (start/customer/end graph; swap positions are not part of the column
    identity). A fractional f^k_ij is branched to 0 and 1.

    Branch propagation to pricing
    -----------------------------
    f^k_ij=0: slot-k pricing forbids successor arc (i,j).
    f^k_ij=1: every slot-k route column must contain successor arc (i,j).
    Since sum_r x_kr=1, restricting the slot-k column universe in this way is
    exactly equivalent to the branch equations. No branch-row dual is needed.

    Exactness
    ---------
    - every node LP is solved by exact elementary pricing;
    - branches partition the slot-labelled integer route space;
    - at every integral routing solution, the BSS subproblem enumerates all
      feasible swap placements and is required to be proven optimal;
    - if the base master underestimates that routing combination, the global
      combinatorial BSS optimality cut is added and the same node is re-priced;
    - pruning uses only valid LP lower bounds versus a feasible BSS incumbent.

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
        maxExactRouteTasks=None,
        bssTimeLimit=None,
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
        )

        self.branchTolerance = branchTolerance
        self.cutTolerance = cutTolerance
        self.maxBssCuts = maxBssCuts
        self.bssTimeLimit = bssTimeLimit

        self.brancher = ArcFlowBrancher(modelData, tolerance=branchTolerance)
        self.bssScheduler = ExactBssScheduler(
            modelData=modelData,
            sequenceEvaluator=self.sequenceEvaluator,
            maxExactRouteTasks=maxExactRouteTasks,
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
        self.totalPhaseOneIterations = 0
        self.totalPhaseTwoIterations = 0
        self.runtime = 0.0
        self._deadline = None
        self._interrupted = False
        self._interruptionStatus = None
        self._lastNodePhaseOneIterations = 0
        self._lastNodePhaseTwoIterations = 0

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
        print('BRANCH-AND-PRICE-AND-CUT'.center(width))
        print('=' * width)
        print(
            f'Instance : tasks={len(self.data.C)}, vehicles={self.data.K}, '
            f'Q={_format_number(self.data.Q, 2)}, QMin={_format_number(self.data.QMin, 2)}'
        )
        print(
            f'Pricing  : {self.pricingMode} | maxColumns/round={self.maxColumnsPerRound} | '
            f'nodeLimit={nodeLimit if nodeLimit is not None else "None"} | '
            f'timeLimit={_format_number(timeLimit, 1) + "s" if timeLimit is not None else "None"}'
        )
        print(
            'Log level: 1=concise BPC, 2=+column generation details, '
            '3=+native Gurobi logs'
        )
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
            f'cols={len(self.columns):5d} | cuts={len(self.bssCuts):4d} | '
            f'CG(P1/P2)={self._lastNodePhaseOneIterations}/{self._lastNodePhaseTwoIterations}'
        )

    def _solveNodeLP(self, branchingState, outputFlag):
        remaining = self._remainingTime()
        phaseOneIterations, status = self._solveCGPhase(
            phase=1,
            outputFlag=outputFlag,
            branchingState=branchingState,
            timeLimit=remaining,
        )
        self._lastNodePhaseOneIterations = phaseOneIterations
        self.totalPhaseOneIterations += phaseOneIterations
        if status != 'OPTIMAL':
            self._lastNodePhaseTwoIterations = 0
            return status, None

        artificial = self.master.getPhaseOneArtificialValue()
        if artificial > self.phaseOneTolerance:
            self._lastNodePhaseTwoIterations = 0
            return 'INFEASIBLE', None

        remaining = self._remainingTime()
        phaseTwoIterations, status = self._solveCGPhase(
            phase=2,
            outputFlag=outputFlag,
            branchingState=branchingState,
            timeLimit=remaining,
        )
        self._lastNodePhaseTwoIterations = phaseTwoIterations
        self.totalPhaseTwoIterations += phaseTwoIterations
        if status != 'OPTIMAL':
            return status, None

        return 'OPTIMAL', self.master.getTValue()

    def _selectedIntegralEntries(self):
        return self.master.getSelectedLPColumns(
            tolerance=self.branchTolerance,
            includeEmpty=False,
        )

    def _selectedIntegralColumns(self):
        return [
            column
            for _, _, column, _ in self._selectedIntegralEntries()
        ]

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
            selectedPlanIndex = self.incumbentBssResult.selectedPlanIndexByRoute.get(
                routeIndex
            )
            plan = None
            if selectedPlanIndex is not None:
                plans = self.sequenceEvaluator.enumerateAllFeasiblePlans(column.tasks)
                if 0 <= selectedPlanIndex < len(plans):
                    plan = plans[selectedPlanIndex]

            nodes = tuple(plan.nodes) if plan is not None else tuple(column.baseNodes)
            pathText = ' -> '.join(self._nodeLabel(node) for node in nodes)
            selectedPlanBase = float(plan.duration) if plan is not None else float(column.duration)
            swapCount = int(plan.swapCount) if plan is not None else sum(
                1 for node in nodes if node in self.data.S
            )

            details.append({
                'vehicleSlot': slot,
                'vehicle': slot + 1,
                'routeIndex': routeIndex,
                'tasks': tuple(column.tasks),
                'nodes': nodes,
                'path': pathText,
                'isolatedMinimumBaseDuration': float(column.duration),
                'selectedPlanBaseDuration': selectedPlanBase,
                'completionTime': float(
                    self.incumbentBssResult.routeCompletion.get(
                        routeIndex,
                        selectedPlanBase,
                    )
                ),
                'waitingTime': float(
                    self.incumbentBssResult.routeWaiting.get(routeIndex, 0.0)
                ),
                'swapCount': swapCount,
                'selectedPlanIndex': selectedPlanIndex,
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
            f"Search        : processed={result['nodesProcessed']} / created={result['nodesCreated']} | "
            f"branches={result['branchCount']} | integralNodes={result['integralNodes']}"
        )
        print(
            f"Pruning       : bound={result['nodesPrunedByBound']} | "
            f"infeasible={result['nodesPrunedInfeasible']}"
        )
        print(
            f"Columns/Cuts  : columns={result['columnCount']} | BSS cuts={result['cutCount']} | "
            f"CG iterations P1/P2={result['phaseOneIterations']}/{result['phaseTwoIterations']}"
        )

        routes = result.get('routes', [])
        if routes:
            print('\n' + '-' * width)
            print('FINAL ROUTES')
            print('-' * width)
            for route in routes:
                print(
                    f"Vehicle {route['vehicle']:>2d} | tasks={list(route['tasks'])} | "
                    f"base(min)={_format_number(route['isolatedMinimumBaseDuration'], digits)} | "
                    f"base(chosen)={_format_number(route['selectedPlanBaseDuration'], digits)} | "
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
        Solve the full BPC tree.

        outputFlag is intentionally interpreted as a BPC verbosity level:
          0 : silent
          1 : concise node/branch/cut/incumbent log + final route summary
          2 : level 1 + every column-generation iteration + final BSS table
          3 : level 2 + native Gurobi logs
        """
        startWall = time.time()
        self._deadline = None if timeLimit is None else startWall + float(timeLimit)

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
        self.totalPhaseOneIterations = 0
        self.totalPhaseTwoIterations = 0
        self.runtime = 0.0
        self._interrupted = False
        self._interruptionStatus = None
        activeNodeSafeBound = None

        self._printStartBanner(outputFlag, nodeLimit, timeLimit)

        rootState = ArcBranchingState.root(self.data.K)
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

        while queue:
            if self._timeExpired():
                self._interrupted = True
                self._interruptionStatus = 'TIME_LIMIT'
                break
            if nodeLimit is not None and self.nodesProcessed >= nodeLimit:
                self._interrupted = True
                self._interruptionStatus = 'NODE_LIMIT'
                break

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

            # A newly discovered global BSS cut can change the same node LP,
            # so an integral node can be re-priced several times before it is
            # finally fathomed or branched.
            while True:
                if self._timeExpired():
                    self._interrupted = True
                    self._interruptionStatus = 'TIME_LIMIT'
                    break

                status, nodeLowerBound = self._solveNodeLP(
                    node.branchingState,
                    outputFlag=outputFlag,
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

                decision = self.brancher.choose(
                    self.master,
                    branchingState=node.branchingState,
                )

                if decision is None:
                    if not self.brancher.isIntegral(self.master):
                        raise RuntimeError(
                            'No fractional successor arc was found although the '
                            'slot-column solution is fractional. This violates '
                            'the branching completeness assumption.'
                        )

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
                        break

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

                    self._updateIncumbent(
                        selectedEntries,
                        bssResult,
                        outputFlag,
                    )

                    if nodeLowerBound < bssResult.objective - self.cutTolerance:
                        if self.maxBssCuts is not None and len(self.bssCuts) >= self.maxBssCuts:
                            self._interrupted = True
                            self._interruptionStatus = 'BSS_CUT_LIMIT'
                            if outputFlag:
                                print('  [STOP] maximum number of BSS cuts reached')
                            break

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
                                f'when routes={ [tuple(c.tasks) for c in selectedColumns] }'
                            )
                            print('             -> re-price the same branch node')

                        # Global valid cut: re-price this same branch node.
                        continue

                    if outputFlag:
                        print(
                            '  [FATHOM] integral routing solution is BSS-consistent '
                            'at this node'
                        )
                    break

                # Fractional node: branch on f^k_ij closest to 0.5.
                slot = decision.slot
                arc = decision.arc
                flow = decision.flow

                children = []
                childDescriptions = []
                for value in (0, 1):
                    childState = node.branchingState.child(
                        slot=slot,
                        arc=arc,
                        value=value,
                        startNode=self.data.startNode,
                        endNode=self.data.endNode,
                    )
                    if childState is None:
                        childDescriptions.append(f'f={value}:structurally infeasible')
                        continue

                    child = _QueueNode(
                        estimatedLowerBound=float(nodeLowerBound),
                        nodeId=nextNodeId,
                        depth=node.depth + 1,
                        branchingState=childState,
                        parentId=node.nodeId,
                    )
                    children.append(child)
                    childDescriptions.append(f'f={value}:node {nextNodeId}')
                    nextNodeId += 1

                for child in children:
                    self.nodesCreated += 1
                    heapq.heappush(queue, child)

                self.branchCount += 1

                if outputFlag:
                    print(
                        f'  [BRANCH] V{slot + 1} successor arc '
                        f'{self._formatArc(arc)} has flow={flow:.6f}'
                    )
                    print(
                        '           -> ' + ' | '.join(childDescriptions)
                    )
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
            'objective': self.incumbentObjective,
            'bestBound': self.bestBound,
            'relativeGap': gap,
            'runtime': self.runtime,
            'columnCount': len(self.columns),
            'cutCount': len(self.bssCuts),
            'nodesCreated': self.nodesCreated,
            'nodesProcessed': self.nodesProcessed,
            'nodesPrunedByBound': self.nodesPrunedByBound,
            'nodesPrunedInfeasible': self.nodesPrunedInfeasible,
            'integralNodes': self.integralNodes,
            'branchCount': self.branchCount,
            'phaseOneIterations': self.totalPhaseOneIterations,
            'phaseTwoIterations': self.totalPhaseTwoIterations,
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
                'OPTIMAL is reported only after the branch queue is exhausted '
                'with exact node pricing and proven-optimal BSS subproblems.'
            ),
        }

