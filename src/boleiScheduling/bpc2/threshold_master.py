from dataclasses import dataclass, field


@dataclass
class ThresholdMasterDuals:
    coverage: dict
    src: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdBssNoGoodCut:
    """For a fixed threshold T, forbid one BSS-infeasible route combination.

        sum_{r in R*} x_r <= |R*| - 1.

    The row is vehicle-free and combination-specific.  Its dual is deliberately
    master-only: every genuinely new route has coefficient zero in an existing
    no-good row.
    """

    routeSignatures: tuple

    @staticmethod
    def canonical(routeSignatures):
        signatures = tuple(sorted(tuple(sig) for sig in routeSignatures))
        return ThresholdBssNoGoodCut(routeSignatures=signatures)

    @property
    def key(self):
        return self.routeSignatures


class ThresholdRestrictedMasterProblem:
    """Persistent vehicle-free RMP for a fixed makespan threshold.

    Phase I:
        min sum_i artificial_i
        sum_r a_ir x_r + artificial_i = 1.

    Phase II:
        min sum_r x_r
        sum_r a_ir x_r = 1.

    Every route variable satisfies isolated route duration <= threshold.  There
    are no vehicle-slot variables and no makespan dual beta_k.
    """

    def __init__(
        self,
        modelData,
        columnManager,
        threshold,
        phase=2,
        branchingState=None,
        bssCuts=None,
        srcCuts=None,
        profiler=None,
        outputFlag=0,
        durationTolerance=1e-9,
    ):
        if phase not in (1, 2):
            raise ValueError('phase must be 1 or 2')
        self.data = modelData
        self.columnManager = columnManager
        self.columns = columnManager.columns
        self.threshold = float(threshold)
        self.phase = int(phase)
        self.branchingState = branchingState
        self.bssCuts = list(bssCuts or ())
        self.srcCuts = list(srcCuts or ())
        self.profiler = profiler
        self.outputFlag = int(outputFlag)
        self.durationTolerance = float(durationTolerance)

        self.model = None
        self.x = {}
        self.artificial = {}
        self.coverageConstr = {}
        self.bssCutConstr = {}
        self.srcCutConstr = {}
        self.requiredRouteConstr = {}
        self._bssCutByKey = {}
        self._bssCutKeysBySignature = {}
        self._srcCutByKey = {}
        self._srcCutKeysByTask = {}
        self._knownGlobalColumnCount = 0
        self.optimizeTime = 0.0
        self.optimizeCount = 0

        self._build()

    @staticmethod
    def _importGurobi():
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError('ThresholdRestrictedMasterProblem requires gurobipy') from error
        return gp, GRB

    def _isColumnActive(self, column):
        if column.isEmpty:
            return False
        if float(column.duration) > self.threshold + self.durationTolerance:
            return False
        if self.branchingState is not None and not self.branchingState.isColumnCompatible(column):
            return False
        return True

    def _build(self):
        import time
        gp, GRB = self._importGurobi()
        started = time.perf_counter()
        model = gp.Model(f'Threshold_RMP_P{self.phase}')
        model.Params.OutputFlag = self.outputFlag

        self.model = model
        if self.phase == 1:
            for task in self.data.C:
                self.artificial[task] = model.addVar(
                    lb=0.0,
                    obj=1.0,
                    vtype=GRB.CONTINUOUS,
                    name=f'art_{task}',
                )

        model.update()
        for task in self.data.C:
            lhs = self.artificial[task] if self.phase == 1 else gp.LinExpr()
            self.coverageConstr[task] = model.addConstr(lhs == 1.0, name=f'cover_{task}')

        # Register global BSS no-good and SRC rows before route variables so a
        # newly added variable can be created with all coefficients at once.
        for cut in self.bssCuts:
            self._registerBssCutIndex(cut)
            self._bssCutByKey[cut.key] = cut
            self.bssCutConstr[cut.key] = model.addConstr(
                gp.LinExpr() <= float(len(cut.routeSignatures) - 1),
                name=f'bssNogood_{len(self.bssCutConstr)}',
            )

        for cut in self.srcCuts:
            self._srcCutByKey[cut.key] = cut
            self._registerSrcCutIndex(cut)
            self.srcCutConstr[cut.key] = model.addConstr(
                gp.LinExpr() <= float(cut.rhs),
                name=f'src_{len(self.srcCutConstr)}',
            )

        # Rare route-variable fallback: x_r == 1.  The named route must already
        # exist in the global pool.  It has no coefficient in future columns.
        if self.branchingState is not None:
            for signature in self.branchingState.requiredSignatures:
                self.requiredRouteConstr[tuple(signature)] = model.addConstr(
                    gp.LinExpr() == 1.0,
                    name=f'routeOne_{len(self.requiredRouteConstr)}',
                )

        self.syncColumns(self.columnManager)
        model.ModelSense = GRB.MINIMIZE
        model.update()

        if self.profiler is not None:
            elapsed = time.perf_counter() - started
            self.profiler.addTime('thresholdMasterBuild', elapsed)
            self.profiler.increment('thresholdMasterBuildCalls')

    def _registerBssCutIndex(self, cut):
        for signature in cut.routeSignatures:
            self._bssCutKeysBySignature.setdefault(tuple(signature), set()).add(cut.key)

    def _registerSrcCutIndex(self, cut):
        for task in cut.customers:
            self._srcCutKeysByTask.setdefault(task, set()).add(cut.key)

    def _candidateSrcCutKeysForRoute(self, column):
        keys = set()
        for task in column.taskSet:
            keys.update(self._srcCutKeysByTask.get(task, ()))
        return keys

    def _columnForRouteVariable(self, routeIndex):
        gp, _ = self._importGurobi()
        route = self.columns[routeIndex]
        coefficients = []
        constraints = []

        for task in route.taskSet:
            coefficients.append(1.0)
            constraints.append(self.coverageConstr[task])

        signature = tuple(route.signature)
        for key in self._bssCutKeysBySignature.get(signature, ()):
            constr = self.bssCutConstr.get(key)
            if constr is not None:
                coefficients.append(1.0)
                constraints.append(constr)

        for key in self._candidateSrcCutKeysForRoute(route):
            cut = self._srcCutByKey.get(key)
            constr = self.srcCutConstr.get(key)
            if cut is None or constr is None:
                continue
            coefficient = cut.coefficient(route)
            if coefficient:
                coefficients.append(float(coefficient))
                constraints.append(constr)

        required = self.requiredRouteConstr.get(signature)
        if required is not None:
            coefficients.append(1.0)
            constraints.append(required)

        return gp.Column(coefficients, constraints)

    def _addRouteVariable(self, routeIndex):
        if routeIndex in self.x:
            return False
        route = self.columns[routeIndex]
        if not self._isColumnActive(route):
            return False
        _, GRB = self._importGurobi()
        obj = 0.0 if self.phase == 1 else 1.0
        self.x[routeIndex] = self.model.addVar(
            lb=0.0,
            ub=GRB.INFINITY,
            obj=obj,
            vtype=GRB.CONTINUOUS,
            name=f'x_{routeIndex}',
            column=self._columnForRouteVariable(routeIndex),
        )
        if self.profiler is not None:
            self.profiler.increment('thresholdMasterRouteVariablesAdded')
        return True

    def syncColumns(self, columnManager):
        target = len(columnManager)
        if target <= self._knownGlobalColumnCount:
            return 0
        added = 0
        for routeIndex in range(self._knownGlobalColumnCount, target):
            if self._addRouteVariable(routeIndex):
                added += 1
        self._knownGlobalColumnCount = target
        if added:
            self.model.update()
        return added

    def addBssCut(self, cut):
        if cut.key in self._bssCutByKey:
            return False
        gp, _ = self._importGurobi()
        lhs = gp.LinExpr()
        self._bssCutByKey[cut.key] = cut
        self._registerBssCutIndex(cut)
        for signature in cut.routeSignatures:
            routeIndex = self.columnManager.signatureToIndex.get(tuple(signature))
            if routeIndex is None:
                continue
            var = self.x.get(routeIndex)
            if var is not None:
                lhs += var
        self.bssCutConstr[cut.key] = self.model.addConstr(
            lhs <= float(len(cut.routeSignatures) - 1),
            name=f'bssNogood_{len(self.bssCutConstr)}',
        )
        self.model.update()
        return True

    def syncBssCuts(self, cuts):
        changed = 0
        for cut in cuts:
            if self.addBssCut(cut):
                changed += 1
        return changed

    def addSrcCut(self, cut):
        if cut.key in self._srcCutByKey:
            return False
        gp, _ = self._importGurobi()
        lhs = gp.LinExpr()
        self._srcCutByKey[cut.key] = cut
        self._registerSrcCutIndex(cut)
        for routeIndex, var in self.x.items():
            coefficient = cut.coefficient(self.columns[routeIndex])
            if coefficient:
                lhs += float(coefficient) * var
        self.srcCutConstr[cut.key] = self.model.addConstr(
            lhs <= float(cut.rhs),
            name=f'src_{len(self.srcCutConstr)}',
        )
        self.model.update()
        return True

    def syncSrcCuts(self, cuts):
        changed = 0
        for cut in cuts:
            if self.addSrcCut(cut):
                changed += 1
        return changed

    def optimize(self, timeLimit=None):
        import time
        _, GRB = self._importGurobi()
        self.model.Params.TimeLimit = GRB.INFINITY if timeLimit is None else max(0.001, float(timeLimit))
        start = time.perf_counter()
        self.model.optimize()
        elapsed = time.perf_counter() - start
        self.optimizeTime += elapsed
        self.optimizeCount += 1
        if self.profiler is not None:
            self.profiler.addTime('thresholdMasterOptimize', elapsed)
            self.profiler.increment('thresholdMasterOptimizeCalls')
        return self.model

    solve = optimize

    def hasSolution(self):
        return self.model is not None and self.model.SolCount > 0

    def isInfeasible(self):
        _, GRB = self._importGurobi()
        return self.model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)

    def getDuals(self):
        if not self.hasSolution():
            raise RuntimeError('RMP has not been solved')
        coverage = {i: float(self.coverageConstr[i].Pi) for i in self.data.C}
        src = {
            self._srcCutByKey[key]: float(constr.Pi)
            for key, constr in self.srcCutConstr.items()
        }
        return ThresholdMasterDuals(coverage=coverage, src=src)

    def getBssCutDuals(self):
        if not self.hasSolution():
            return {}
        return {
            self._bssCutByKey[key]: float(constr.Pi)
            for key, constr in self.bssCutConstr.items()
        }

    def getPhaseOneArtificialValue(self):
        if self.phase != 1:
            raise RuntimeError('not a Phase-I master')
        return sum(float(self.artificial[i].X) for i in self.data.C)

    def getObjectiveValue(self):
        if not self.hasSolution():
            return float('inf')
        return float(self.model.ObjVal)

    def getPositiveX(self, tolerance=1e-8):
        result = []
        for routeIndex, var in self.x.items():
            value = float(var.X)
            if value > tolerance:
                result.append((routeIndex, self.columns[routeIndex], value))
        return result

    def isIntegral(self, tolerance=1e-7):
        for _, _, value in self.getPositiveX(tolerance=1e-12):
            if abs(value - round(value)) > tolerance:
                return False
        return True

    def getSelectedColumns(self, tolerance=1e-7):
        if not self.isIntegral(tolerance=tolerance):
            raise RuntimeError('master solution is fractional')
        selected = []
        for routeIndex, column, value in self.getPositiveX(tolerance=tolerance):
            if value >= 1.0 - tolerance:
                selected.append((routeIndex, column, value))
        return selected
