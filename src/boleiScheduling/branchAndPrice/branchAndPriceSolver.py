import time

from gurobipy import GRB

from .masterProblem import MasterProblem
from .pricingProblem import PricingProblem


class BranchAndPriceScheduler:
    def __init__(self, modelData):
        self.modelData = modelData
        self.routePool = {}
        self.routeSignature = {}
        self.nextRouteId = 0

        self.bestObjective = float("inf")
        self.bestColumns = []
        self.bestLambda = {}
        self.bestRoutes = []

        self.nodeCount = 0
        self.columnCount = 0
        self.rootLowerBound = None
        self.runtime = 0.0
        self.status = None

        self.pricing = PricingProblem(modelData)

    def solve(
        self,
        timeLimit=None,
        outputFlag=1,
        reducedCostTolerance=1e-6,
        integralityTolerance=1e-6,
    ):
        self.modelData.validate()
        startTime = time.time()
        deadline = None if timeLimit is None else startTime + timeLimit

        root = {
            "depth": 0,
            "columns": set(),
            "branchRows": [],
        }
        stack = [root]

        while stack:
            if self.isTimedOut(deadline):
                self.status = GRB.TIME_LIMIT
                break

            node = stack.pop()
            self.nodeCount += 1

            result = self.solveNode(
                node=node,
                deadline=deadline,
                outputFlag=0,
                reducedCostTolerance=reducedCostTolerance,
            )

            if result is None:
                if self.isTimedOut(deadline):
                    self.status = GRB.TIME_LIMIT
                    break
                continue

            lowerBound = result["objective"]
            if node["depth"] == 0:
                self.rootLowerBound = lowerBound

            if lowerBound >= self.bestObjective - integralityTolerance:
                continue

            lambdaValues = result["lambda"]
            if self.isIntegral(lambdaValues, integralityTolerance):
                self.saveIncumbent(lowerBound, result["columns"], lambdaValues, outputFlag)
                continue

            branch = self.chooseArcBranch(
                lambdaValues,
                result["columns"],
                integralityTolerance,
            )

            if branch is None:
                # 若所有物理弧流均为整数，fractional lambda 只是在等价 route 间分配。
                # 选择每辆车的一条等价 route 即可恢复整数解。
                integralLambda = self.buildEquivalentIntegralSolution(
                    lambdaValues,
                    integralityTolerance,
                )
                if integralLambda is not None:
                    objective = self.getSolutionObjective(integralLambda)
                    self.saveIncumbent(
                        objective,
                        result["columns"],
                        integralLambda,
                        outputFlag,
                    )
                continue

            vehicle, arc, value = branch
            child0 = self.createChild(node, result["columns"], {
                "type": "arc",
                "vehicle": vehicle,
                "arc": arc,
                "value": 0,
            })
            child1 = self.createChild(node, result["columns"], {
                "type": "arc",
                "vehicle": vehicle,
                "arc": arc,
                "value": 1,
            })

            # 深度优先，优先尝试 x = 1。
            stack.append(child0)
            stack.append(child1)

            if outputFlag:
                print(
                    f"B&P node {self.nodeCount}: LB={lowerBound:.3f}, "
                    f"branch vehicle={vehicle}, arc={arc}, value={value:.3f}"
                )

        self.runtime = time.time() - startTime

        if self.status is None:
            if self.bestObjective < float("inf"):
                self.status = GRB.OPTIMAL
            else:
                self.status = GRB.INFEASIBLE

        return self

    def solveModel(
        self,
        timeLimit=None,
        outputFlag=1,
        reducedCostTolerance=1e-6,
        integralityTolerance=1e-6,
    ):
        return self.solve(
            timeLimit=timeLimit,
            outputFlag=outputFlag,
            reducedCostTolerance=reducedCostTolerance,
            integralityTolerance=integralityTolerance,
        )

    def solveNode(self, node, deadline, outputFlag, reducedCostTolerance):
        columns = set(node["columns"])
        branchRows = list(node["branchRows"])

        phase1 = self.columnGeneration(
            columns=columns,
            branchRows=branchRows,
            phase=1,
            deadline=deadline,
            outputFlag=outputFlag,
            reducedCostTolerance=reducedCostTolerance,
        )
        if phase1 is None:
            return None

        if phase1["objective"] > 1e-7:
            return None

        return self.columnGeneration(
            columns=phase1["columns"],
            branchRows=branchRows,
            phase=2,
            deadline=deadline,
            outputFlag=outputFlag,
            reducedCostTolerance=reducedCostTolerance,
        )

    def columnGeneration(
        self,
        columns,
        branchRows,
        phase,
        deadline,
        outputFlag,
        reducedCostTolerance,
    ):
        columns = set(columns)

        while True:
            if self.isTimedOut(deadline):
                return None

            remainingTime = None
            if deadline is not None:
                remainingTime = max(0.001, deadline - time.time())

            master = MasterProblem(
                modelData=self.modelData,
                routePool=self.routePool,
                columns=columns,
                branchRows=branchRows,
            )
            model = master.solve(
                phase=phase,
                outputFlag=0,
                timeLimit=remainingTime,
            )

            if model.Status == GRB.TIME_LIMIT:
                return None
            if model.Status != GRB.OPTIMAL:
                return None

            duals = master.getDuals(phase=phase)
            added = 0

            for vehicle in range(self.modelData.K):
                if self.isTimedOut(deadline):
                    return None

                route, reducedCost, timedOut = self.pricing.findRoute(
                    vehicle=vehicle,
                    duals=duals,
                    branchRows=branchRows,
                    routeId=self.nextRouteId,
                    tolerance=reducedCostTolerance,
                    deadline=deadline,
                )

                if timedOut:
                    return None

                if route is None or reducedCost >= -reducedCostTolerance:
                    continue

                routeId = self.addRoute(route)
                key = (vehicle, routeId)
                if key not in columns:
                    columns.add(key)
                    added += 1

            if outputFlag:
                print(
                    f"CG phase={phase}, obj={model.ObjVal:.6f}, "
                    f"columns={len(columns)}, added={added}"
                )

            if added == 0:
                return {
                    "objective": model.ObjVal,
                    "columns": set(columns),
                    "lambda": master.getLambdaValues(),
                    "master": master,
                }

    def addRoute(self, route):
        signature = route.signature
        if signature in self.routeSignature:
            return self.routeSignature[signature]

        routeId = self.nextRouteId
        self.nextRouteId += 1
        route.routeId = routeId

        self.routePool[routeId] = route
        self.routeSignature[signature] = routeId
        self.columnCount += 1
        return routeId

    def chooseArcBranch(self, lambdaValues, columns, tolerance):
        arcValue = {}

        for vehicle, routeId in columns:
            value = lambdaValues.get((vehicle, routeId), 0.0)
            if value <= tolerance:
                continue

            route = self.routePool[routeId]
            for arc in route.arcs:
                key = (vehicle, arc)
                arcValue[key] = arcValue.get(key, 0.0) + value

        best = None
        bestDistance = float("inf")
        for key, value in arcValue.items():
            if value <= tolerance or value >= 1 - tolerance:
                continue

            distance = abs(value - 0.5)
            if distance < bestDistance:
                bestDistance = distance
                best = (key[0], key[1], value)

        return best

    def buildEquivalentIntegralSolution(self, lambdaValues, tolerance):
        result = {key: 0.0 for key in lambdaValues}

        for vehicle in range(self.modelData.K):
            positive = []
            for (v, routeId), value in lambdaValues.items():
                if v == vehicle and value > tolerance:
                    positive.append((routeId, value))

            if len(positive) == 0:
                continue

            arcSets = []
            for routeId, value in positive:
                arcSets.append(frozenset(self.routePool[routeId].arcs))

            if any(arcSet != arcSets[0] for arcSet in arcSets[1:]):
                return None

            routeId = max(positive, key=lambda item: item[1])[0]
            result[vehicle, routeId] = 1.0

        # 检查任务仍然恰好被覆盖一次。
        for task in self.modelData.C:
            cover = 0
            for (vehicle, routeId), value in result.items():
                if value > 0.5 and self.routePool[routeId].containsTask(task):
                    cover += 1
            if cover != 1:
                return None

        return result

    def getSolutionObjective(self, lambdaValues):
        objective = 0.0
        for (vehicle, routeId), value in lambdaValues.items():
            if value > 0.5:
                objective = max(objective, self.routePool[routeId].duration)
        return objective

    def saveIncumbent(self, objective, columns, lambdaValues, outputFlag):
        if objective >= self.bestObjective - 1e-9:
            return

        self.bestObjective = objective
        self.bestColumns = list(columns)
        self.bestLambda = dict(lambdaValues)
        self.bestRoutes = self.extractRoutes(lambdaValues)

        if outputFlag:
            print(f"B&P incumbent: {self.bestObjective:.3f}")

    @staticmethod
    def createChild(node, columns, branchRow):
        return {
            "depth": node["depth"] + 1,
            "columns": set(columns),
            "branchRows": list(node["branchRows"]) + [branchRow],
        }

    @staticmethod
    def isIntegral(lambdaValues, tolerance):
        for value in lambdaValues.values():
            if value > tolerance and value < 1 - tolerance:
                return False
        return True

    @staticmethod
    def isTimedOut(deadline):
        return deadline is not None and time.time() >= deadline

    def extractRoutes(self, lambdaValues):
        routes = []

        for (vehicle, routeId), value in lambdaValues.items():
            if value <= 0.5:
                continue

            route = self.routePool[routeId]
            routes.append({
                "vehicle": vehicle,
                "routeId": routeId,
                "nodes": list(route.nodes),
                "tasks": list(route.tasks),
                "arcs": list(route.arcs),
                "duration": route.duration,
                "swapCount": route.swapCount,
            })

        routes.sort(key=lambda item: item["vehicle"])
        return routes

    def getRoutes(self):
        return self.bestRoutes

    def getResult(self):
        return {
            "objective": self.bestObjective,
            "rootLowerBound": self.rootLowerBound,
            "runtime": self.runtime,
            "nodes": self.nodeCount,
            "routes": len(self.routePool),
            "columns": self.columnCount,
            "status": self.status,
        }
