import time

from .routeAssembler import RouteAssembler
from .routeGenerator import RouteGenerator
from .routePool import RoutePool


class RoutePoolHeuristicScheduler:
    def __init__(
        self,
        modelData,
        outerIterations=10,
        ilsIterations=50,
        variantsPerSequence=3,
        maxPoolSize=5000,
        assemblerIterations=100,
        seed=1,
    ):
        self.modelData = modelData
        self.outerIterations = outerIterations
        self.ilsIterations = ilsIterations
        self.variantsPerSequence = variantsPerSequence
        self.maxPoolSize = maxPoolSize
        self.assemblerIterations = assemblerIterations
        self.seed = seed

        self.routeGenerator = RouteGenerator(
            modelData,
            variantsPerSequence=variantsPerSequence,
            seed=seed,
        )
        self.longTermPool = RoutePool(maxSize=maxPoolSize)
        self.routeAssembler = RouteAssembler(modelData)

        self.bestObjective = float("inf")
        self.bestRoutes = []
        self.bestSchedule = None
        self.runtime = 0.0
        self.iterations = 0

    def solveModel(self, timeLimit=None, outputFlag=1):
        startTime = time.time()
        currentSequences = self.routeGenerator.createInitialSolution()

        initialRoutes = self.routeGenerator.extractRoutes(currentSequences)
        self.longTermPool.addRoutes(initialRoutes, protect=True)

        for iteration in range(self.outerIterations):
            if timeLimit is not None and time.time() - startTime >= timeLimit:
                break

            generation = self.routeGenerator.generateRoutes(
                currentSequences,
                iterations=self.ilsIterations,
            )

            shortTermRoutes = generation["routes"]
            bestRelaxedSequences = generation["bestSequences"]

            # 当前 relaxed 最优解中的路线进入长期池。
            bestRelaxedRoutes = self.routeGenerator.extractRoutes(bestRelaxedSequences)
            self.longTermPool.addRoutes(bestRelaxedRoutes, protect=True)

            # 本轮产生的路线作为 short-term pool 临时加入 assembler。
            shortTermPool = RoutePool(maxSize=self.maxPoolSize)
            shortTermPool.addRoutes(shortTermRoutes)

            combinedRoutes = self.mergePools(self.longTermPool, shortTermPool)

            remainingTime = None
            if timeLimit is not None:
                remainingTime = max(0.1, timeLimit - (time.time() - startTime))

            result = self.routeAssembler.solve(
                combinedRoutes,
                maxIterations=self.assemblerIterations,
                timeLimit=remainingTime,
                outputFlag=0,
            )

            if result["objective"] < self.bestObjective - 1e-9:
                self.bestObjective = result["objective"]
                self.bestRoutes = result["routes"]
                self.bestSchedule = result["schedule"]

            # Assembler 选中的路线进入长期池。
            self.longTermPool.protectRoutes(result["routes"])

            # 下一轮从 assembler 当前选中的 task sequences 继续搜索。
            currentSequences = [list(route.tasks) for route in result["routes"]]

            # 将排队最严重路线中的任务反馈给下一轮 perturbation。
            focusTasks = self.buildFocusTasks(result["routes"], result["schedule"])
            self.routeGenerator.setFocusTasks(focusTasks)

            self.iterations = iteration + 1

            if outputFlag:
                print(
                    f"Iteration {iteration + 1}: "
                    f"pool={self.longTermPool.getSize()}, "
                    f"relaxed={generation['bestScore'][0]:.3f}, "
                    f"true={result['objective']:.3f}"
                )

        self.runtime = time.time() - startTime
        return self

    def getRoutes(self):
        routes = []
        for index, route in enumerate(self.bestRoutes):
            routes.append({
                "vehicle": index,
                "routeId": route.routeId,
                "nodes": list(route.nodes),
                "tasks": list(route.tasks),
                "duration": route.duration,
                "swapCount": route.swapCount,
                "completionTime": self.getRouteCompletion(route.routeId),
            })
        return routes

    def getSwapEvents(self):
        if self.bestSchedule is None:
            return []
        return [dict(event) for event in self.bestSchedule["events"]]

    def getResult(self):
        return {
            "objective": self.bestObjective,
            "runtime": self.runtime,
            "iterations": self.iterations,
            "routePoolSize": self.longTermPool.getSize(),
            "status": "FEASIBLE" if len(self.bestRoutes) > 0 else "NO_SOLUTION",
        }

    def getRouteCompletion(self, routeId):
        if self.bestSchedule is None:
            return None
        return self.bestSchedule["routeCompletion"].get(routeId)


    @staticmethod
    def buildFocusTasks(routes, schedule):
        waitingByRoute = {}
        for event in schedule["events"]:
            waitingByRoute[event["routeId"]] = waitingByRoute.get(event["routeId"], 0.0) + event["waitTime"]

        if len(waitingByRoute) == 0:
            return []

        maxWaiting = max(waitingByRoute.values())
        if maxWaiting <= 1e-9:
            return []

        focusTasks = []
        for route in routes:
            if waitingByRoute.get(route.routeId, 0.0) >= 0.5 * maxWaiting:
                focusTasks.extend(route.tasks)
        return focusTasks

    @staticmethod
    def mergePools(longTermPool, shortTermPool):
        routeBySignature = {}
        for route in longTermPool.getRoutes() + shortTermPool.getRoutes():
            routeBySignature[route.signature] = route

        mergedRoutes = []
        for routeId, route in enumerate(routeBySignature.values()):
            mergedRoutes.append(route.copyWithId(routeId))
        return mergedRoutes
