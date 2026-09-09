from .bssScheduler import BssScheduler


class RouteAssembler:
    def __init__(self, modelData):
        self.modelData = modelData
        self.bssScheduler = BssScheduler(modelData)

    def solve(self, routes, maxIterations=100, timeLimit=None, outputFlag=0):
        try:
            from gurobipy import GRB, Model, quicksum
        except ImportError as error:
            raise ImportError("路线组合需要 gurobipy") from error

        if len(routes) == 0:
            raise ValueError("路线池不能为空")

        routeById = {route.routeId: route for route in routes}
        routeIds = list(routeById.keys())

        taskRoutes = {}
        for task in self.modelData.C:
            taskRoutes[task] = [route.routeId for route in routes if route.containsTask(task)]
            if len(taskRoutes[task]) == 0:
                raise ValueError(f"路线池中没有覆盖任务 {task} 的路线")

        cuts = []
        bestObjective = float("inf")
        bestRoutes = None
        bestSchedule = None
        lastLowerBound = None

        for iteration in range(maxIterations):
            model = Model("BolaiRouteAssembler")
            model.Params.OutputFlag = outputFlag
            if timeLimit is not None:
                model.Params.TimeLimit = timeLimit

            x = model.addVars(routeIds, vtype=GRB.BINARY, name="x")
            CMax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CMax")

            model.setObjective(CMax, GRB.MINIMIZE)

            for task in self.modelData.C:
                model.addConstr(
                    quicksum(x[routeId] for routeId in taskRoutes[task]) == 1,
                    name=f"taskCover_{task}",
                )

            model.addConstr(quicksum(x[routeId] for routeId in routeIds) <= self.modelData.K, name="vehicleLimit")

            for route in routes:
                model.addConstr(CMax >= route.duration * x[route.routeId], name=f"baseMakespan_{route.routeId}")

            for cutIndex, cut in enumerate(cuts):
                selectedIds = cut["routeIds"]
                trueObjective = cut["objective"]
                model.addConstr(
                    CMax >= trueObjective * (
                        quicksum(x[routeId] for routeId in selectedIds) - len(selectedIds) + 1
                    ),
                    name=f"waitingCut_{cutIndex}",
                )

            model.optimize()
            if model.SolCount == 0:
                break

            lastLowerBound = model.ObjVal
            selectedRoutes = [routeById[routeId] for routeId in routeIds if x[routeId].X > 0.5]

            schedule = self.bssScheduler.solve(selectedRoutes, timeLimit=timeLimit, outputFlag=0)
            trueObjective = schedule["objective"]

            if trueObjective < bestObjective - 1e-9:
                bestObjective = trueObjective
                bestRoutes = selectedRoutes
                bestSchedule = schedule

            if trueObjective <= model.ObjVal + 1e-6:
                break

            selectedIds = tuple(sorted(route.routeId for route in selectedRoutes))
            duplicateCut = any(cut["routeIds"] == selectedIds for cut in cuts)
            if duplicateCut:
                break

            cuts.append({
                "routeIds": selectedIds,
                "objective": trueObjective,
            })

            if bestObjective < float("inf") and model.ObjVal >= bestObjective - 1e-6:
                break

        if bestRoutes is None:
            raise RuntimeError("路线组合器未找到可行路线组合")

        return {
            "objective": bestObjective,
            "routes": bestRoutes,
            "schedule": bestSchedule,
            "cuts": len(cuts),
            "lowerBound": lastLowerBound,
        }
