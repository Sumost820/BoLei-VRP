from dataclasses import dataclass, field


@dataclass
class MasterDuals:
    coverage: dict
    slot: dict
    makespan: dict
    # Mapping {SubsetRowCut: Gurobi Pi}.  For a <= row in this minimization
    # master, Pi is nonpositive.  Pricing therefore receives the nonnegative
    # penalty -Pi * alpha_r for the cut coefficient alpha_r.
    src: dict = field(default_factory=dict)


class RestrictedMasterProblem:
    """
    Vehicle-slot Dantzig-Wolfe restricted master with:
      - global BSS combinatorial optimality cuts,
      - global subset-row cuts (SRCs),
      - slot-specific successor-arc branch restrictions.

    Branch restrictions do not add master rows. Instead, x[k,r] is fixed to
    zero whenever column r is incompatible with the branch state for slot k.
    This is equivalent to f^k_ij=0/1 branching because each slot satisfies
        sum_r x[k,r] = 1.

    SRCs are ordinary master rows.  Their duals therefore MUST be included in
    pricing.  The pricing module does this exactly by tracking the parity state
    of every SRC with a nonzero dual.
    """

    def __init__(
        self,
        modelData,
        columns,
        bssCuts=None,
        srcCuts=None,
        branchingState=None,
    ):
        self.data = modelData
        self.columns = columns
        self.bssCuts = list(bssCuts or [])
        self.srcCuts = list(srcCuts or [])
        self.branchingState = branchingState
        self.model = None
        self.x = None
        self.T = None
        self.artificial = None
        self.coverageConstr = None
        self.slotConstr = None
        self.makespanConstr = None
        self.bssCutConstr = None
        self.srcCutConstr = None
        self.phase = None
        self.binary = False

    def _isCompatible(self, slot, column):
        if self.branchingState is None:
            return True
        return self.branchingState.isColumnCompatible(
            slot,
            column,
            self.data.startNode,
            self.data.endNode,
        )

    def build(self, phase=2, outputFlag=0, binary=False, timeLimit=None):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError('RestrictedMasterProblem requires gurobipy') from error

        if phase not in (1, 2):
            raise ValueError('phase must be 1 or 2')
        if binary and phase != 2:
            raise ValueError('binary master is only defined for Phase II')

        self.phase = phase
        self.binary = binary
        model = gp.Model(f'BPC_RMP_Phase{phase}' + ('_MIP' if binary else '_LP'))
        model.Params.OutputFlag = outputFlag
        if timeLimit is not None:
            model.Params.TimeLimit = max(0.001, float(timeLimit))

        columnIndices = list(range(len(self.columns)))
        slotIndices = list(range(self.data.K))
        variableType = GRB.BINARY if binary else GRB.CONTINUOUS

        x = model.addVars(
            slotIndices,
            columnIndices,
            lb=0.0,
            ub=1.0,
            vtype=variableType,
            name='x',
        )

        # Remove branch-incompatible columns from each slot's route universe.
        for k in slotIndices:
            for r, column in enumerate(self.columns):
                if not self._isCompatible(k, column):
                    x[k, r].UB = 0.0

        T = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='T')

        artificial = None
        if phase == 1:
            artificial = model.addVars(
                self.data.C,
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name='u',
            )

        coverageConstr = {}
        for task in self.data.C:
            lhs = gp.quicksum(
                x[k, r]
                for k in slotIndices
                for r, column in enumerate(self.columns)
                if task in column.taskSet
            )
            if phase == 1:
                lhs += artificial[task]
            coverageConstr[task] = model.addConstr(lhs == 1.0, name=f'cover_{task}')

        slotConstr = {}
        for k in slotIndices:
            slotConstr[k] = model.addConstr(
                gp.quicksum(x[k, r] for r in columnIndices) == 1.0,
                name=f'slot_{k}',
            )

        makespanConstr = {}
        for k in slotIndices:
            makespanConstr[k] = model.addConstr(
                T - gp.quicksum(
                    self.columns[r].duration * x[k, r]
                    for r in columnIndices
                ) >= 0.0,
                name=f'makespan_{k}',
            )

        signatureToIndex = {
            column.signature: r
            for r, column in enumerate(self.columns)
        }

        bssCutConstr = []
        srcCutConstr = {}

        if phase == 2:
            # BSS combination-specific optimality cuts.
            for cutIndex, cut in enumerate(self.bssCuts):
                present = [
                    signatureToIndex[signature]
                    for signature in cut.routeSignatures
                    if signature in signatureToIndex
                ]
                if len(present) != len(cut.routeSignatures):
                    raise RuntimeError('BSS cut refers to a route column missing from the RMP')

                selectionSum = gp.quicksum(
                    x[k, r]
                    for r in present
                    for k in slotIndices
                )
                rhs = cut.value * (
                    selectionSum
                    - len(cut.routeSignatures)
                    + 1
                )
                bssCutConstr.append(
                    model.addConstr(T >= rhs, name=f'bssCut_{cutIndex}')
                )

            # Subset-row cuts.  For cut S and divisor 2:
            #   sum_{k,r} floor(|S∩r|/2) x[k,r] <= floor(|S|/2).
            for cutIndex, cut in enumerate(self.srcCuts):
                terms = []
                for r, column in enumerate(self.columns):
                    coefficient = cut.coefficient(column)
                    if coefficient <= 0:
                        continue
                    for k in slotIndices:
                        terms.append(coefficient * x[k, r])

                lhs = gp.quicksum(terms)
                srcCutConstr[cut] = model.addConstr(
                    lhs <= cut.rhs,
                    name=f'src_{cutIndex}',
                )

        if phase == 1:
            model.setObjective(
                gp.quicksum(artificial[i] for i in self.data.C),
                GRB.MINIMIZE,
            )
        else:
            model.setObjective(T, GRB.MINIMIZE)

        self.model = model
        self.x = x
        self.T = T
        self.artificial = artificial
        self.coverageConstr = coverageConstr
        self.slotConstr = slotConstr
        self.makespanConstr = makespanConstr
        self.bssCutConstr = bssCutConstr
        self.srcCutConstr = srcCutConstr
        return model

    def solve(self, phase=2, outputFlag=0, binary=False, timeLimit=None):
        self.build(
            phase=phase,
            outputFlag=outputFlag,
            binary=binary,
            timeLimit=timeLimit,
        )
        self.model.optimize()
        return self.model

    def getDuals(self):
        if self.model is None or self.model.SolCount == 0:
            raise RuntimeError('RMP has not been solved')
        if self.binary:
            raise RuntimeError('Duals are only available for the LP master')

        coverage = {i: self.coverageConstr[i].Pi for i in self.data.C}
        slot = {k: self.slotConstr[k].Pi for k in range(self.data.K)}
        makespan = {k: self.makespanConstr[k].Pi for k in range(self.data.K)}
        src = {
            cut: constr.Pi
            for cut, constr in (self.srcCutConstr or {}).items()
        }
        return MasterDuals(
            coverage=coverage,
            slot=slot,
            makespan=makespan,
            src=src,
        )

    def getPhaseOneArtificialValue(self):
        if self.phase != 1:
            raise RuntimeError('RMP is not in Phase I')
        return sum(self.artificial[i].X for i in self.data.C)

    def getObjectiveValue(self):
        if self.model is None or self.model.SolCount == 0:
            return float('inf')
        return self.model.ObjVal

    def getTValue(self):
        if self.model is None or self.model.SolCount == 0:
            return float('inf')
        return self.T.X

    def getPositiveX(self, tolerance=1e-8):
        result = []
        for k in range(self.data.K):
            for r, column in enumerate(self.columns):
                value = self.x[k, r].X
                if value > tolerance:
                    result.append((k, r, column, value))
        return result

    def getSelectedLPColumns(self, tolerance=1e-7, includeEmpty=False):
        result = []
        for k in range(self.data.K):
            selected = [
                (r, column, self.x[k, r].X)
                for r, column in enumerate(self.columns)
                if self.x[k, r].X >= 1.0 - tolerance
            ]
            if len(selected) != 1:
                raise RuntimeError(
                    f'slot {k} is not integral: selected-at-one={selected}'
                )
            r, column, value = selected[0]
            if includeEmpty or not column.isEmpty:
                result.append((k, r, column, value))
        return result

    def getSelectedIntegerColumns(self, tolerance=0.5, includeEmpty=False):
        if not self.binary:
            raise RuntimeError('Selected integer columns require binary=True')

        result = []
        for k in range(self.data.K):
            for r, column in enumerate(self.columns):
                if self.x[k, r].X > tolerance:
                    if includeEmpty or not column.isEmpty:
                        result.append((k, r, column))
        return result


class PersistentRestrictedMasterProblem:
    """Incremental LP restricted master used by column generation.

    One instance is tied to one branch node and one phase.  The Gurobi model is
    built once; route columns and global cuts are appended incrementally.  This
    preserves the LP basis between CG iterations and avoids rebuilding the full
    constraint matrix after every pricing call.

    ``RestrictedMasterProblem`` above is intentionally kept for the small
    binary restricted-master reference and backwards compatibility.
    """

    def __init__(
        self,
        modelData,
        columns,
        phase=2,
        bssCuts=None,
        srcCuts=None,
        branchingState=None,
        outputFlag=0,
        timeLimit=None,
        profiler=None,
        useDualSimplex=True,
    ):
        if phase not in (1, 2):
            raise ValueError('phase must be 1 or 2')
        self.data = modelData
        self.phase = phase
        self.branchingState = branchingState
        self.profiler = profiler
        self.useDualSimplex = bool(useDualSimplex)

        self.columns = []
        self.signatureToIndex = {}
        self.x = {}
        self.T = None
        self.artificial = None
        self.coverageConstr = {}
        self.slotConstr = {}
        self.makespanConstr = {}
        self.bssCutConstr = {}
        self.srcCutConstr = {}
        self._bssCutByKey = {}
        self._srcCutByKey = {}
        self.model = None
        self.buildTime = 0.0
        self.optimizeTime = 0.0
        self.optimizeCount = 0

        import time as _time
        start = _time.perf_counter()
        self._buildEmptyModel(outputFlag=outputFlag, timeLimit=timeLimit)
        if phase == 2:
            self.syncBssCuts(bssCuts or ())
            self.syncSrcCuts(srcCuts or ())
        self.addColumns(columns)
        self.buildTime = _time.perf_counter() - start
        if self.profiler is not None:
            self.profiler.addTime('masterBuild', self.buildTime)
            self.profiler.increment('persistentMasterBuilds')

    def _importGurobi(self):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError('PersistentRestrictedMasterProblem requires gurobipy') from error
        return gp, GRB

    def _buildEmptyModel(self, outputFlag=0, timeLimit=None):
        gp, GRB = self._importGurobi()
        model = gp.Model(f'BPC_Persistent_RMP_Phase{self.phase}')
        model.Params.OutputFlag = int(outputFlag)
        if self.useDualSimplex:
            # Column generation repeatedly appends variables to an LP. Dual
            # simplex generally reuses the previous basis very efficiently.
            model.Params.Method = 1
        if timeLimit is not None:
            model.Params.TimeLimit = max(0.001, float(timeLimit))

        self.T = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='T')

        if self.phase == 1:
            self.artificial = model.addVars(
                self.data.C,
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name='u',
            )

        # Create all permanent core rows before route variables. New route
        # variables can then be appended with a Gurobi Column object.
        for task in self.data.C:
            if self.phase == 1:
                lhs = self.artificial[task]
            else:
                lhs = gp.LinExpr()
            self.coverageConstr[task] = model.addConstr(
                lhs == 1.0,
                name=f'cover_{task}',
            )

        for k in range(self.data.K):
            self.slotConstr[k] = model.addConstr(
                gp.LinExpr() == 1.0,
                name=f'slot_{k}',
            )
            self.makespanConstr[k] = model.addConstr(
                self.T >= 0.0,
                name=f'makespan_{k}',
            )

        if self.phase == 1:
            model.setObjective(
                gp.quicksum(self.artificial[i] for i in self.data.C),
                GRB.MINIMIZE,
            )
        else:
            model.setObjective(self.T, GRB.MINIMIZE)

        model.update()
        self.model = model

    def _isCompatible(self, slot, column):
        if self.branchingState is None:
            return True
        return self.branchingState.isColumnCompatible(
            slot,
            column,
            self.data.startNode,
            self.data.endNode,
        )

    def _columnForRouteVariable(self, slot, routeColumn):
        gp, _ = self._importGurobi()
        coefficients = []
        constraints = []

        for task in routeColumn.taskSet:
            constr = self.coverageConstr.get(task)
            if constr is not None:
                coefficients.append(1.0)
                constraints.append(constr)

        coefficients.append(1.0)
        constraints.append(self.slotConstr[slot])

        coefficients.append(-float(routeColumn.duration))
        constraints.append(self.makespanConstr[slot])

        if self.phase == 2:
            signature = routeColumn.signature
            for key, cut in self._bssCutByKey.items():
                if signature in cut.routeSignatures:
                    coefficients.append(-float(cut.value))
                    constraints.append(self.bssCutConstr[key])

            for key, cut in self._srcCutByKey.items():
                coefficient = cut.coefficient(routeColumn)
                if coefficient:
                    coefficients.append(float(coefficient))
                    constraints.append(self.srcCutConstr[key])

        return gp.Column(coefficients, constraints)

    def addColumn(self, routeColumn):
        signature = tuple(routeColumn.signature)
        previous = self.signatureToIndex.get(signature)
        if previous is not None:
            return False, previous

        _, GRB = self._importGurobi()
        routeIndex = len(self.columns)
        self.columns.append(routeColumn)
        self.signatureToIndex[signature] = routeIndex

        for k in range(self.data.K):
            column = self._columnForRouteVariable(k, routeColumn)
            ub = 1.0 if self._isCompatible(k, routeColumn) else 0.0
            var = self.model.addVar(
                lb=0.0,
                ub=ub,
                obj=0.0,
                vtype=GRB.CONTINUOUS,
                name=f'x_{k}_{routeIndex}',
                column=column,
            )
            self.x[k, routeIndex] = var

        if self.profiler is not None:
            self.profiler.increment('masterColumnsAdded')
        return True, routeIndex

    def addColumns(self, routeColumns):
        added = 0
        for routeColumn in routeColumns:
            wasAdded, _ = self.addColumn(routeColumn)
            if wasAdded:
                added += 1
        if added:
            self.model.update()
        return added

    def syncColumns(self, globalColumns):
        if len(globalColumns) <= len(self.columns):
            return 0
        return self.addColumns(globalColumns[len(self.columns):])

    def _buildBssCutConstraint(self, cut):
        gp, _ = self._importGurobi()
        selection = gp.LinExpr()
        signatures = set(cut.routeSignatures)
        for r, routeColumn in enumerate(self.columns):
            if routeColumn.signature not in signatures:
                continue
            for k in range(self.data.K):
                selection += self.x[k, r]
        # T - value*selection >= value*(1-|R*|)
        return self.model.addConstr(
            self.T - float(cut.value) * selection
            >= float(cut.value) * (1 - len(cut.routeSignatures)),
            name=f'bssCut_{len(self.bssCutConstr)}',
        )

    def addOrStrengthenBssCut(self, cut):
        if self.phase != 2:
            return False
        key = cut.key
        old = self._bssCutByKey.get(key)
        if old is not None and old.value >= cut.value - 1e-12:
            return False

        if old is not None:
            oldConstr = self.bssCutConstr.pop(key, None)
            if oldConstr is not None:
                self.model.remove(oldConstr)
                self.model.update()

        self._bssCutByKey[key] = cut
        self.bssCutConstr[key] = self._buildBssCutConstraint(cut)
        self.model.update()
        if self.profiler is not None:
            self.profiler.increment('masterBssCutsAdded')
        return True

    def syncBssCuts(self, cuts):
        changed = 0
        for cut in cuts:
            if self.addOrStrengthenBssCut(cut):
                changed += 1
        return changed

    def addSrcCut(self, cut):
        if self.phase != 2:
            return False
        key = cut.key
        if key in self._srcCutByKey:
            return False

        gp, _ = self._importGurobi()
        lhs = gp.LinExpr()
        for r, routeColumn in enumerate(self.columns):
            coefficient = cut.coefficient(routeColumn)
            if coefficient <= 0:
                continue
            for k in range(self.data.K):
                lhs += float(coefficient) * self.x[k, r]

        self._srcCutByKey[key] = cut
        self.srcCutConstr[key] = self.model.addConstr(
            lhs <= float(cut.rhs),
            name=f'src_{len(self.srcCutConstr)}',
        )
        self.model.update()
        if self.profiler is not None:
            self.profiler.increment('masterSrcCutsAdded')
        return True

    def syncSrcCuts(self, cuts):
        changed = 0
        for cut in cuts:
            if self.addSrcCut(cut):
                changed += 1
        return changed

    def optimize(self, timeLimit=None):
        import time as _time
        _, GRB = self._importGurobi()
        if timeLimit is None:
            self.model.Params.TimeLimit = GRB.INFINITY
        else:
            self.model.Params.TimeLimit = max(0.001, float(timeLimit))

        start = _time.perf_counter()
        self.model.optimize()
        elapsed = _time.perf_counter() - start
        self.optimizeTime += elapsed
        self.optimizeCount += 1
        if self.profiler is not None:
            self.profiler.addTime('masterOptimize', elapsed)
            self.profiler.increment('masterOptimizeCalls')
        return self.model

    # Compatibility alias used by solver code.
    solve = optimize

    def getDuals(self):
        if self.model is None or self.model.SolCount == 0:
            raise RuntimeError('RMP has not been solved')
        coverage = {i: self.coverageConstr[i].Pi for i in self.data.C}
        slot = {k: self.slotConstr[k].Pi for k in range(self.data.K)}
        makespan = {k: self.makespanConstr[k].Pi for k in range(self.data.K)}
        src = {
            self._srcCutByKey[key]: constr.Pi
            for key, constr in self.srcCutConstr.items()
        }
        return MasterDuals(
            coverage=coverage,
            slot=slot,
            makespan=makespan,
            src=src,
        )

    def getPhaseOneArtificialValue(self):
        if self.phase != 1:
            raise RuntimeError('RMP is not in Phase I')
        return sum(self.artificial[i].X for i in self.data.C)

    def getObjectiveValue(self):
        if self.model is None or self.model.SolCount == 0:
            return float('inf')
        return float(self.model.ObjVal)

    def getTValue(self):
        if self.model is None or self.model.SolCount == 0:
            return float('inf')
        return float(self.T.X)

    def getPositiveX(self, tolerance=1e-8):
        result = []
        for k in range(self.data.K):
            for r, routeColumn in enumerate(self.columns):
                value = float(self.x[k, r].X)
                if value > tolerance:
                    result.append((k, r, routeColumn, value))
        return result

    def getSelectedLPColumns(self, tolerance=1e-7, includeEmpty=False):
        result = []
        for k in range(self.data.K):
            selected = [
                (r, routeColumn, float(self.x[k, r].X))
                for r, routeColumn in enumerate(self.columns)
                if self.x[k, r].X >= 1.0 - tolerance
            ]
            if len(selected) != 1:
                raise RuntimeError(
                    f'slot {k} is not integral: selected-at-one={selected}'
                )
            r, routeColumn, value = selected[0]
            if includeEmpty or not routeColumn.isEmpty:
                result.append((k, r, routeColumn, value))
        return result
