from gurobipy import GRB, Model, quicksum


class MasterProblem:
    def __init__(self, modelData, routePool, columns, branchRows):
        self.modelData = modelData
        self.routePool = routePool
        self.columns = columns
        self.branchRows = branchRows

        self.model = None
        self.lambdaVar = None
        self.CMax = None
        self.artificial = None

        self.taskConstr = {}
        self.vehicleConstr = {}
        self.makespanConstr = {}
        self.branchConstr = []

    def buildModel(self, phase=2, outputFlag=0):
        data = self.modelData
        C = data.C
        K = data.K

        model = Model("BolaiBranchAndPriceMaster")
        model.Params.OutputFlag = outputFlag

        lambdaVar = {}
        for vehicle, routeId in sorted(self.columns):
            lambdaVar[vehicle, routeId] = model.addVar(
                lb=0,
                vtype=GRB.CONTINUOUS,
                name=f"lambda_{vehicle}_{routeId}",
            )

        CMax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CMax")
        model.update()

        taskConstr = {}
        for i in C:
            expr = quicksum(
                lambdaVar[vehicle, routeId]
                for vehicle, routeId in self.columns
                if self.routePool[routeId].containsTask(i)
            )
            taskConstr[i] = model.addConstr(expr == 1, name=f"task_{i}")

        vehicleConstr = {}
        for vehicle in range(K):
            expr = quicksum(
                lambdaVar[v, routeId]
                for v, routeId in self.columns
                if v == vehicle
            )
            vehicleConstr[vehicle] = model.addConstr(expr <= 1, name=f"vehicle_{vehicle}")

        makespanConstr = {}
        if phase == 2:
            for vehicle in range(K):
                expr = quicksum(
                    self.routePool[routeId].duration * lambdaVar[v, routeId]
                    for v, routeId in self.columns
                    if v == vehicle
                )
                makespanConstr[vehicle] = model.addConstr(
                    CMax - expr >= 0,
                    name=f"makespan_{vehicle}",
                )

        branchConstr = []
        for index, row in enumerate(self.branchRows):
            vehicle = row["vehicle"]
            arc = row["arc"]
            value = row["value"]

            expr = quicksum(
                lambdaVar[v, routeId]
                for v, routeId in self.columns
                if v == vehicle and self.routePool[routeId].containsArc(arc)
            )
            branchConstr.append(
                model.addConstr(expr == value, name=f"branch_{index}")
            )

        artificial = []
        if phase == 1:
            for i in C:
                plus = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"artTaskPlus_{i}")
                minus = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"artTaskMinus_{i}")
                model.chgCoeff(taskConstr[i], plus, 1)
                model.chgCoeff(taskConstr[i], minus, -1)
                artificial.extend([plus, minus])

            for index, constr in enumerate(branchConstr):
                plus = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"artBranchPlus_{index}")
                minus = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"artBranchMinus_{index}")
                model.chgCoeff(constr, plus, 1)
                model.chgCoeff(constr, minus, -1)
                artificial.extend([plus, minus])

            model.setObjective(quicksum(artificial), GRB.MINIMIZE)
        else:
            model.setObjective(CMax, GRB.MINIMIZE)

        self.model = model
        self.lambdaVar = lambdaVar
        self.CMax = CMax
        self.artificial = artificial
        self.taskConstr = taskConstr
        self.vehicleConstr = vehicleConstr
        self.makespanConstr = makespanConstr
        self.branchConstr = branchConstr
        return model

    def solve(self, phase=2, outputFlag=0, timeLimit=None):
        model = self.buildModel(phase=phase, outputFlag=outputFlag)
        if timeLimit is not None:
            model.Params.TimeLimit = max(timeLimit, 0.001)
        model.optimize()
        return model

    def getDuals(self, phase=2):
        taskDual = {i: constr.Pi for i, constr in self.taskConstr.items()}
        vehicleDual = {k: constr.Pi for k, constr in self.vehicleConstr.items()}
        makespanDual = {k: 0.0 for k in range(self.modelData.K)}

        if phase == 2:
            makespanDual = {k: constr.Pi for k, constr in self.makespanConstr.items()}

        branchDual = []
        for index, constr in enumerate(self.branchConstr):
            row = dict(self.branchRows[index])
            row["dual"] = constr.Pi
            branchDual.append(row)

        return {
            "task": taskDual,
            "vehicle": vehicleDual,
            "makespan": makespanDual,
            "branch": branchDual,
        }

    def getLambdaValues(self):
        values = {}
        for key, var in self.lambdaVar.items():
            values[key] = var.X
        return values
