from gurobipy import GRB, Model, quicksum


class PathBasedGurobiScheduler:
    def __init__(self, modelData):
        self.modelData = modelData
        self.model = None
        self.x = None
        self.T = None
        self.E = None
        self.s = None
        self.z = None

        self.P = []
        self.P0 = []
        self.PS = []
        self.AS = []
        self.origin = {}
        self.destination = {}
        self.stationNode = None
        self.alpha = -1
        self.omega = -2

    def buildModel(self):
        data = self.modelData
        data.validate()

        C = data.C
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

        if len(data.S) == 0:
            raise ValueError("Path-based 模型需要一个物理换电站")

        stationNode = data.S[0]
        swapTime = p[stationNode]
        V = [startNode] + C + [endNode]
        energyNodes = [startNode] + C

        P0, PS, P, origin, destination = self.buildPaths(data, stationNode)
        AS = self.buildStationArcs(PS)

        model = Model("BolaiPathBasedScheduling")

        x = model.addVars(P, vtype=GRB.BINARY, name="x")
        T = model.addVars(V, lb=0, ub=M, vtype=GRB.CONTINUOUS, name="T")
        E = model.addVars(energyNodes, lb=0, ub=Q, vtype=GRB.CONTINUOUS, name="E")
        s = model.addVars(PS, lb=0, ub=M, vtype=GRB.CONTINUOUS, name="s")
        z = model.addVars(AS, vtype=GRB.BINARY, name="z")

        model.setObjective(T[endNode], GRB.MINIMIZE)

        pathIn = self.buildPathIn(P, destination)
        pathOut = self.buildPathOut(P, origin)

        startCount = quicksum(x[path] for path in pathOut.get(startNode, []))
        endCount = quicksum(x[path] for path in pathIn.get(endNode, []))
        model.addConstr(startCount == endCount, name="vehicleBalance")
        model.addConstr(startCount <= K, name="vehicleLimit")

        for i in C:
            inFlow = quicksum(x[path] for path in pathIn.get(i, []))
            outFlow = quicksum(x[path] for path in pathOut.get(i, []))
            model.addConstr(inFlow == 1, name=f"taskIn_{i}")
            model.addConstr(outFlow == 1, name=f"taskOut_{i}")

        for path in P0:
            i = origin[path]
            j = destination[path]
            serviceTime = p.get(j, 0)
            model.addConstr(T[j] >= T[i] + t[i, j] + serviceTime - M * (1 - x[path]), name=f"timeDirect_{path}")

        for path in PS:
            i = origin[path]
            j = destination[path]
            serviceTime = p.get(j, 0)

            model.addConstr(s[path] >= T[i] + t[i, stationNode] - M * (1 - x[path]), name=f"swapStart_{path}")
            model.addConstr(T[j] >= s[path] + swapTime + t[stationNode, j] + serviceTime - M * (1 - x[path]), name=f"timeSwap_{path}")
            model.addConstr(s[path] <= M * x[path], name=f"unusedSwapTime_{path}")

        for i in C:
            model.addConstr(E[i] >= QMin, name=f"energyMin_{i}")
            model.addConstr(E[i] <= Q, name=f"energyMax_{i}")

        for path in P0:
            i = origin[path]
            j = destination[path]
            taskEnergy = q.get(j, 0)

            if j in C:
                model.addConstr(E[j] >= E[i] - e[i, j] - taskEnergy - 2 * Q * (1 - x[path]), name=f"energyDirectLower_{path}")
                model.addConstr(E[j] <= E[i] - e[i, j] - taskEnergy + 2 * Q * (1 - x[path]), name=f"energyDirectUpper_{path}")

            if j == endNode:
                model.addConstr(E[i] - e[i, j] - taskEnergy >= QMin - 2 * Q * (1 - x[path]), name=f"energyDirectEnd_{path}")

        for path in PS:
            i = origin[path]
            j = destination[path]
            taskEnergy = q.get(j, 0)

            model.addConstr(E[i] - e[i, stationNode] >= QMin - 2 * Q * (1 - x[path]), name=f"energyReachStation_{path}")

            if j in C:
                model.addConstr(E[j] >= Q - e[stationNode, j] - taskEnergy - 2 * Q * (1 - x[path]), name=f"energySwapLower_{path}")
                model.addConstr(E[j] <= Q - e[stationNode, j] - taskEnergy + 2 * Q * (1 - x[path]), name=f"energySwapUpper_{path}")

            if j == endNode:
                model.addConstr(Q - e[stationNode, j] - taskEnergy >= QMin - 2 * Q * (1 - x[path]), name=f"energySwapEnd_{path}")

        stationIn = self.buildPredecessors(AS)
        stationOut = self.buildSuccessors(AS)

        for path in PS:
            inFlow = quicksum(z[q, path] for q in stationIn.get(path, []))
            outFlow = quicksum(z[path, q] for q in stationOut.get(path, []))
            model.addConstr(inFlow == x[path], name=f"swapIn_{path}")
            model.addConstr(outFlow == x[path], name=f"swapOut_{path}")

        alphaOut = quicksum(z[self.alpha, q] for q in stationOut.get(self.alpha, []))
        omegaIn = quicksum(z[q, self.omega] for q in stationIn.get(self.omega, []))
        model.addConstr(alphaOut == 1, name="swapChainStart")
        model.addConstr(omegaIn == 1, name="swapChainEnd")

        for path in PS:
            for nextPath in PS:
                if path == nextPath:
                    continue
                model.addConstr(
                    s[nextPath] >= s[path] + swapTime - M * (1 - z[path, nextPath]), name=f"swapQueue_{path}_{nextPath}"
                )

        model.addConstr(T[startNode] == 0, name="startTime")
        model.addConstr(E[startNode] == Q, name="startEnergy")

        taskServiceTime = quicksum(p[i] for i in C)
        directTravelTime = quicksum(t[origin[path], destination[path]] * x[path] for path in P0)
        swapTravelTime = quicksum(
            (t[origin[path], stationNode] + swapTime + t[stationNode, destination[path]]) * x[path] for path in PS
        )
        model.addConstr(
            K * T[endNode] >= taskServiceTime + directTravelTime + swapTravelTime, name="totalWorkloadLowerBound"
        )

        self.model = model
        self.x = x
        self.T = T
        self.E = E
        self.s = s
        self.z = z

        self.P = P
        self.P0 = P0
        self.PS = PS
        self.AS = AS
        self.origin = origin
        self.destination = destination
        self.stationNode = stationNode
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
        startNode = data.startNode
        endNode = data.endNode

        selectedPath = {}
        for path in self.P:
            if self.x[path].X > 0.5:
                selectedPath[self.origin[path]] = path

        firstPaths = []
        for path in self.P:
            if self.origin[path] == startNode and self.x[path].X > 0.5:
                firstPaths.append(path)

        routes = []
        for firstPath in firstPaths:
            route = [startNode]
            path = firstPath

            while True:
                if path in self.PS:
                    route.append(self.stationNode)

                nextNode = self.destination[path]
                route.append(nextNode)

                if nextNode == endNode:
                    break

                if nextNode not in selectedPath:
                    break

                path = selectedPath[nextNode]

            routes.append(route)

        return routes

    def getSwapEvents(self):
        if self.model is None or self.model.SolCount == 0:
            return []

        events = []
        for path in self.PS:
            if self.x[path].X > 0.5:
                events.append({
                    "path": path,
                    "origin": self.origin[path],
                    "destination": self.destination[path],
                    "startTime": self.s[path].X,
                })

        events.sort(key=lambda event: event["startTime"])
        return events

    @staticmethod
    def buildPaths(data, stationNode):
        C = data.C
        A = set(data.A)
        startNode = data.startNode
        endNode = data.endNode

        P0 = []
        PS = []
        origin = {}
        destination = {}
        pathId = 0

        VMinus = [startNode] + C
        VPlus = C + [endNode]

        for i in VMinus:
            for j in VPlus:
                if i == j:
                    continue
                if i == startNode and j == endNode:
                    continue

                if (i, j) in A:
                    P0.append(pathId)
                    origin[pathId] = i
                    destination[pathId] = j
                    pathId += 1

                if i in C and (i, stationNode) in A and (stationNode, j) in A:
                    PS.append(pathId)
                    origin[pathId] = i
                    destination[pathId] = j
                    pathId += 1

        P = P0 + PS
        return P0, PS, P, origin, destination

    def buildStationArcs(self, PS):
        AS = [(self.alpha, self.omega)]

        for path in PS:
            AS.append((self.alpha, path))
            AS.append((path, self.omega))

        for path in PS:
            for nextPath in PS:
                if path != nextPath:
                    AS.append((path, nextPath))

        return AS

    @staticmethod
    def buildPathIn(P, destination):
        pathIn = {}
        for path in P:
            node = destination[path]
            pathIn.setdefault(node, []).append(path)
        return pathIn

    @staticmethod
    def buildPathOut(P, origin):
        pathOut = {}
        for path in P:
            node = origin[path]
            pathOut.setdefault(node, []).append(path)
        return pathOut

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
