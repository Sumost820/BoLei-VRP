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
