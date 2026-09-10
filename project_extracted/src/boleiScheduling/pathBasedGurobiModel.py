from gurobipy import GRB, Model, quicksum


class PathBasedGurobiScheduler:
    def __init__(self, modelData):
        self.modelData = modelData
        self.model = None
        self.x = None
        self.T = None
        self.E = None
        self.y = None
        self.s = None
        self.z = None

        self.P = []
        self.P0 = []
        self.PS = []
        self.swapOrderPairs = []
        self.origin = {}
        self.destination = {}
        self.stationNode = None

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
        swapOrderPairs = self.buildSwapOrderPairs(C)

        model = Model("BolaiPathBasedScheduling")

        x = model.addVars(P, vtype=GRB.BINARY, name="x")
        T = model.addVars(V, lb=0, ub=M, vtype=GRB.CONTINUOUS, name="T")
        E = model.addVars(energyNodes, lb=0, ub=Q, vtype=GRB.CONTINUOUS, name="E")
        y = model.addVars(C, vtype=GRB.BINARY, name="y")
        s = model.addVars(C, lb=0, ub=M, vtype=GRB.CONTINUOUS, name="s")
        z = model.addVars(swapOrderPairs, vtype=GRB.BINARY, name="z")

        model.setObjective(T[endNode], GRB.MINIMIZE)

        pathIn = self.buildPathIn(P, destination)
        pathOut = self.buildPathOut(P, origin)
        swapPathOut = self.buildPathOut(PS, origin)

        startCount = quicksum(x[path] for path in pathOut.get(startNode, []))
        endCount = quicksum(x[path] for path in pathIn.get(endNode, []))
        model.addConstr(startCount == endCount, name="vehicleBalance")
        model.addConstr(startCount <= K, name="vehicleLimit")

        for i in C:
            inFlow = quicksum(x[path] for path in pathIn.get(i, []))
            outFlow = quicksum(x[path] for path in pathOut.get(i, []))
            model.addConstr(inFlow == 1, name=f"taskIn_{i}")
            model.addConstr(outFlow == 1, name=f"taskOut_{i}")

        for i in C:
            swapFlow = quicksum(x[path] for path in swapPathOut.get(i, []))
            model.addConstr(y[i] == swapFlow, name=f"swapUsed_{i}")
            model.addConstr(s[i] <= M * y[i], name=f"unusedSwapTime_{i}")

            if len(swapPathOut.get(i, [])) > 0:
                model.addConstr(s[i] >= T[i] + t[i, stationNode] - M * (1 - y[i]), name=f"swapStart_{i}")

        for path in P0:
            i = origin[path]
            j = destination[path]
            serviceTime = p.get(j, 0)
            model.addConstr( T[j] >= T[i] + t[i, j] + serviceTime - M * (1 - x[path]), name=f"timeDirect_{path}")

        for path in PS:
            i = origin[path]
            j = destination[path]
            serviceTime = p.get(j, 0)
            model.addConstr(T[j] >= s[i] + swapTime + t[stationNode, j] + serviceTime - M * (1 - x[path]), name=f"timeSwap_{path}")

        for i in C:
            model.addConstr(E[i] >= QMin, name=f"energyMin_{i}")
            model.addConstr(E[i] <= Q, name=f"energyMax_{i}")

        for path in P0:
            i = origin[path]
            j = destination[path]
            taskEnergy = q.get(j, 0)

            if j in C:
                model.addConstr(E[j] >= E[i] - e[i, j] - taskEnergy - M * (1 - x[path]), name=f"energyDirectLower_{path}")
                model.addConstr(E[j] <= E[i] - e[i, j] - taskEnergy + M * (1 - x[path]), name=f"energyDirectUpper_{path}")

            if j == endNode:
                model.addConstr(E[i] - e[i, j] - taskEnergy >= QMin - M * (1 - x[path]), name=f"energyDirectEnd_{path}")

        for path in PS:
            i = origin[path]
            j = destination[path]
            taskEnergy = q.get(j, 0)

            model.addConstr(E[i] - e[i, stationNode] >= QMin - M * (1 - x[path]), name=f"energyReachStation_{path}")

            if j in C:
                model.addConstr(E[j] >= Q - e[stationNode, j] - taskEnergy - M * (1 - x[path]), name=f"energySwapLower_{path}")
                model.addConstr(E[j] <= Q - e[stationNode, j] - taskEnergy + M * (1 - x[path]), name=f"energySwapUpper_{path}")

            if j == endNode:
                model.addConstr(Q - e[stationNode, j] - taskEnergy >= QMin - M * (1 - x[path]), name=f"energySwapEnd_{path}")

        for i, k in swapOrderPairs:
            model.addConstr(z[i, k] <= y[i], name=f"swapOrderUseI_{i}_{k}")
            model.addConstr(z[i, k] <= y[k], name=f"swapOrderUseK_{i}_{k}")

            model.addConstr(
                s[k] >= s[i] + swapTime - M * (1 - z[i, k]) - M * (2 - y[i] - y[k]), name=f"swapQueueIThenK_{i}_{k}"
            )

            model.addConstr(
                s[i] >= s[k] + swapTime - M * z[i, k] - M * (2 - y[i] - y[k]), name=f"swapQueueKThenI_{i}_{k}")

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
        self.y = y
        self.s = s
        self.z = z

        self.P = P
        self.P0 = P0
        self.PS = PS
        self.swapOrderPairs = swapOrderPairs
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
                i = self.origin[path]
                events.append({
                    "path": path,
                    "origin": i,
                    "destination": self.destination[path],
                    "startTime": self.s[i].X,
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

    @staticmethod
    def buildSwapOrderPairs(C):
        swapOrderPairs = []
        for index, i in enumerate(C):
            for k in C[index + 1:]:
                swapOrderPairs.append((i, k))
        return swapOrderPairs

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
