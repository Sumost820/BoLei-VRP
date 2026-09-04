from gurobipy import GRB, Model, quicksum


class GurobiScheduler:
    def __init__(self, modelData):
        self.modelData = modelData
        self.model = None
        self.x = None
        self.T = None
        self.E = None

    def buildModel(self):
        data = self.modelData
        data.validate()

        C = data.C
        S = data.S
        V = data.V
        A = data.A
        p = data.p
        t = data.t
        q = data.q
        e = data.e
        Q = data.Q
        QMin = data.QMin
        K = data.K
        M = data.M
        startNode = data.startNode
        endNode = data.endNode

        model = Model("BolaiScheduling")

        x = model.addVars(A, vtype=GRB.BINARY, name="x")
        T = model.addVars(V, lb=0, ub=M, vtype=GRB.CONTINUOUS, name="T")

        energyNodes = [startNode] + C + S
        E = model.addVars(energyNodes, lb=0, ub=Q, vtype=GRB.CONTINUOUS, name="E")

        model.setObjective(T[endNode], GRB.MINIMIZE)

        predecessors = self.buildPredecessors(A)
        successors = self.buildSuccessors(A)

        startCount = quicksum(x[startNode, j] for j in successors.get(startNode, []))
        endCount = quicksum(x[i, endNode] for i in predecessors.get(endNode, []))
        model.addConstr(startCount == endCount, name="vehicleBalance")
        model.addConstr(startCount <= K, name="vehicleLimit")

        for i in C:
            inFlow = quicksum(x[j, i] for j in predecessors.get(i, []))
            outFlow = quicksum(x[i, j] for j in successors.get(i, []))
            model.addConstr(inFlow == 1, name=f"taskIn_{i}")
            model.addConstr(outFlow == 1, name=f"taskOut_{i}")

        for i in S:
            inFlow = quicksum(x[j, i] for j in predecessors.get(i, []))
            outFlow = quicksum(x[i, j] for j in successors.get(i, []))
            model.addConstr(outFlow == inFlow, name=f"stationFlow_{i}")
            model.addConstr(inFlow <= 1, name=f"stationVisit_{i}")

        for i, j in A:
            if j in C or j in S:
                model.addConstr(
                    T[j] >= T[i] + t[i, j] + p[j] - M * (1 - x[i, j]),
                    name=f"timeLower_{i}_{j}",
                )

            if j in C:
                model.addConstr(
                    T[j] <= T[i] + t[i, j] + p[j] + M * (1 - x[i, j]),
                    name=f"timeUpper_{i}_{j}",
                )

        for j in S:
            inFlow = quicksum(x[i, j] for i in predecessors.get(j, []))
            model.addConstr(T[j] <= M * inFlow, name=f"unusedStationTime_{j}")

        for i in predecessors.get(endNode, []):
            model.addConstr(
                T[endNode] >= T[i] + t[i, endNode] - M * (1 - x[i, endNode]),
                name=f"endTime_{i}",
            )

        for i in C:
            model.addConstr(E[i] >= QMin, name=f"energyMin_{i}")
            model.addConstr(E[i] <= Q, name=f"energyMax_{i}")

        for i, j in A:
            if j in C:
                model.addConstr(
                    E[j] >= E[i] - e[i, j] - q[j] - M * (1 - x[i, j]),
                    name=f"energyLower_{i}_{j}",
                )
                model.addConstr(
                    E[j] <= E[i] - e[i, j] - q[j] + M * (1 - x[i, j]),
                    name=f"energyUpper_{i}_{j}",
                )

            if j in S or j == endNode:
                model.addConstr(
                    E[i] - e[i, j] >= QMin - M * (1 - x[i, j]),
                    name=f"energyReach_{i}_{j}",
                )

        for j in S:
            inFlow = quicksum(x[i, j] for i in predecessors.get(j, []))
            model.addConstr(E[j] == Q * inFlow, name=f"stationEnergy_{j}")

        for r in range(len(S) - 1):
            currentStation = S[r]
            nextStation = S[r + 1]
            currentIn = quicksum(
                x[i, currentStation] for i in predecessors.get(currentStation, [])
            )
            nextIn = quicksum(
                x[i, nextStation] for i in predecessors.get(nextStation, [])
            )
            model.addConstr(nextIn <= currentIn, name=f"stationSymmetry_{r + 1}")
            model.addConstr(
                T[nextStation]
                >= T[currentStation]
                + p[currentStation]
                - M * (1 - nextIn),
                name=f"stationQueue_{r + 1}",
            )

        model.addConstr(T[startNode] == 0, name="startTime")
        model.addConstr(E[startNode] == Q, name="startEnergy")

        self.model = model
        self.x = x
        self.T = T
        self.E = E
        return model

    def solveModel(self, timeLimit=None, mipGap=None, outputFlag=1):
        if self.model is None:
            self.buildModel()

        self.model.Params.OutputFlag = outputFlag
        if timeLimit is not None:
            self.model.Params.TimeLimit = timeLimit
        if mipGap is not None:
            self.model.Params.MIPGap = mipGap

        self.model.optimize()
        return self.model

    def getRoutes(self):
        if self.model is None or self.model.SolCount == 0:
            return []

        data = self.modelData
        successors = self.buildSuccessors(data.A)
        startNode = data.startNode
        endNode = data.endNode

        firstNodes = []
        for j in successors.get(startNode, []):
            if self.x[startNode, j].X > 0.5:
                firstNodes.append(j)

        routes = []
        for firstNode in firstNodes:
            route = [startNode, firstNode]
            currentNode = firstNode

            while currentNode != endNode:
                nextNode = None
                for j in successors.get(currentNode, []):
                    if self.x[currentNode, j].X > 0.5:
                        nextNode = j
                        break

                if nextNode is None:
                    break

                route.append(nextNode)
                currentNode = nextNode

            routes.append(route)

        return routes

    @staticmethod
    def buildPredecessors(A):
        predecessors = {}
        for i, j in A:
            predecessors.setdefault(j, []).append(i)
        return predecessors

    @staticmethod
    def buildSuccessors(A):
        successors = {}
        for i, j in A:
            successors.setdefault(i, []).append(j)
        return successors
