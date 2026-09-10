class BssScheduler:
    def __init__(self, modelData):
        self.modelData = modelData

    def solve(self, routes, timeLimit=None, outputFlag=0):
        try:
            from gurobipy import GRB, Model, quicksum
        except ImportError as error:
            raise ImportError("BSS 调度需要 gurobipy") from error

        if len(routes) == 0:
            return {
                "objective": 0.0,
                "events": [],
                "routeCompletion": {},
            }

        stationNode = self.modelData.S[0]
        swapTime = self.modelData.p[stationNode]

        eventKeys = []
        eventData = {}
        for route in routes:
            for event in route.swapEvents:
                key = (route.routeId, event["index"])
                eventKeys.append(key)
                eventData[key] = {
                    "route": route,
                    "event": event,
                }

        model = Model("BolaiBssScheduling")
        model.Params.OutputFlag = outputFlag
        if timeLimit is not None:
            model.Params.TimeLimit = timeLimit

        CMax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CMax")

        if len(eventKeys) == 0:
            for route in routes:
                model.addConstr(CMax >= route.duration, name=f"routeCompletion_{route.routeId}")
            model.setObjective(CMax, GRB.MINIMIZE)
            model.optimize()

            return {
                "objective": CMax.X,
                "events": [],
                "routeCompletion": {route.routeId: route.duration for route in routes},
            }

        s = model.addVars(eventKeys, lb=0, vtype=GRB.CONTINUOUS, name="s")

        for route in routes:
            if route.swapCount == 0:
                model.addConstr(CMax >= route.duration, name=f"routeCompletion_{route.routeId}")
                continue

            firstKey = (route.routeId, 0)
            firstEvent = route.swapEvents[0]
            model.addConstr(s[firstKey] >= firstEvent["arrivalTime"], name=f"firstArrival_{route.routeId}")

            for eventIndex in range(route.swapCount - 1):
                currentKey = (route.routeId, eventIndex)
                nextKey = (route.routeId, eventIndex + 1)
                nextEvent = route.swapEvents[eventIndex + 1]
                gap = nextEvent["gapFromPreviousSwap"]
                model.addConstr(
                    s[nextKey] >= s[currentKey] + swapTime + gap,
                    name=f"routePrecedence_{route.routeId}_{eventIndex}",
                )

            lastKey = (route.routeId, route.swapCount - 1)
            lastEvent = route.swapEvents[-1]
            model.addConstr(
                CMax >= s[lastKey] + swapTime + lastEvent["tailDuration"],
                name=f"routeCompletion_{route.routeId}",
            )

        crossPairs = []
        for firstIndex, firstKey in enumerate(eventKeys):
            firstRouteId = firstKey[0]
            for secondKey in eventKeys[firstIndex + 1:]:
                secondRouteId = secondKey[0]
                if firstRouteId == secondRouteId:
                    continue
                crossPairs.append((firstKey, secondKey))

        z = {}
        M = max(self.modelData.M, sum(route.duration for route in routes) + len(eventKeys) * swapTime)

        for pairIndex, (firstKey, secondKey) in enumerate(crossPairs):
            z[pairIndex] = model.addVar(vtype=GRB.BINARY, name=f"z_{pairIndex}")
            model.addConstr(
                s[secondKey] >= s[firstKey] + swapTime - M * (1 - z[pairIndex]),
                name=f"stationOrderForward_{pairIndex}",
            )
            model.addConstr(
                s[firstKey] >= s[secondKey] + swapTime - M * z[pairIndex],
                name=f"stationOrderBackward_{pairIndex}",
            )

        model.setObjective(CMax, GRB.MINIMIZE)
        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError("BSS 调度子问题未找到解；在无时间窗模型下这通常表示模型或数据有问题")

        resultEvents = []
        routeCompletion = {}

        for route in routes:
            if route.swapCount == 0:
                routeCompletion[route.routeId] = route.duration
                continue

            waiting = 0.0
            for event in route.swapEvents:
                key = (route.routeId, event["index"])
                startTime = s[key].X

                if event["index"] == 0:
                    earliestArrival = event["arrivalTime"]
                else:
                    previousKey = (route.routeId, event["index"] - 1)
                    gap = event["gapFromPreviousSwap"]
                    earliestArrival = s[previousKey].X + swapTime + gap

                waitTime = max(0.0, startTime - earliestArrival)
                waiting += waitTime

                resultEvents.append({
                    "routeId": route.routeId,
                    "eventIndex": event["index"],
                    "afterTask": event["afterTask"],
                    "arrivalTime": earliestArrival,
                    "startTime": startTime,
                    "endTime": startTime + swapTime,
                    "waitTime": waitTime,
                })

            routeCompletion[route.routeId] = route.duration + waiting

        resultEvents.sort(key=lambda event: event["startTime"])
        return {
            "objective": CMax.X,
            "events": resultEvents,
            "routeCompletion": routeCompletion,
        }
