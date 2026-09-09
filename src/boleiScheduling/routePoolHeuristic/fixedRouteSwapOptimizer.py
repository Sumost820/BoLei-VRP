from .route import HeuristicRoute


class FixedRouteSwapOptimizer:
    def __init__(self, modelData, maxVariants=3):
        self.modelData = modelData
        self.maxVariants = maxVariants
        self.cache = {}
        self.nextTemporaryRouteId = -1

        if len(modelData.S) == 0:
            raise ValueError("启发式算法需要至少一个换电站节点")

        # 所有 station copy 表示同一个物理换电站。
        # 启发式路线中只使用第一个节点作为物理换电站。
        self.stationNode = modelData.S[0]

    def optimize(self, taskSequence, maxVariants=None):
        if maxVariants is None:
            maxVariants = self.maxVariants

        taskSequence = tuple(taskSequence)
        cacheKey = (taskSequence, maxVariants)
        if cacheKey in self.cache:
            return [route.copyWithId(self.getTemporaryRouteId()) for route in self.cache[cacheKey]]

        if len(taskSequence) == 0:
            self.cache[cacheKey] = []
            return []

        candidatesAtSwap = [[] for _ in range(len(taskSequence) + 1)]
        completeCandidates = []

        # 0 -> task_1 -> ... -> task_j -> S
        for j in range(1, len(taskSequence) + 1):
            segment = self.evaluateSegment(
                startNode=self.modelData.startNode,
                tasks=taskSequence[:j],
                endNode=self.stationNode,
                includeSwap=True,
            )
            if segment is not None:
                candidatesAtSwap[j].append((segment["duration"], (j,)))

        # 0 -> task_1 -> ... -> task_m -> end
        direct = self.evaluateSegment(
            startNode=self.modelData.startNode,
            tasks=taskSequence,
            endNode=self.modelData.endNode,
            includeSwap=False,
        )
        if direct is not None:
            completeCandidates.append((direct["duration"], ()))

        # S -> task_{i+1} -> ... -> task_j -> S
        for i in range(1, len(taskSequence)):
            if len(candidatesAtSwap[i]) == 0:
                continue

            for j in range(i + 1, len(taskSequence) + 1):
                segment = self.evaluateSegment(
                    startNode=self.stationNode,
                    tasks=taskSequence[i:j],
                    endNode=self.stationNode,
                    includeSwap=True,
                )
                if segment is None:
                    continue

                newCandidates = []
                for previousDuration, previousSwaps in candidatesAtSwap[i]:
                    newCandidates.append((previousDuration + segment["duration"], previousSwaps + (j,)))

                candidatesAtSwap[j].extend(newCandidates)
                candidatesAtSwap[j] = self.keepBestCandidates(candidatesAtSwap[j], maxVariants)

        # 最后一次换电后从 S 完成剩余任务并返回终点。
        for i in range(1, len(taskSequence) + 1):
            if len(candidatesAtSwap[i]) == 0:
                continue

            if i == len(taskSequence):
                tail = self.evaluateEmptyTailFromStation()
            else:
                tail = self.evaluateSegment(
                    startNode=self.stationNode,
                    tasks=taskSequence[i:],
                    endNode=self.modelData.endNode,
                    includeSwap=False,
                )

            if tail is None:
                continue

            for previousDuration, previousSwaps in candidatesAtSwap[i]:
                completeCandidates.append((previousDuration + tail["duration"], previousSwaps))

        completeCandidates = self.keepBestCandidates(completeCandidates, maxVariants)

        routes = []
        for _, swapPositions in completeCandidates:
            route = self.buildRoute(taskSequence, swapPositions)
            if route is not None:
                routes.append(route)

        routes.sort(key=lambda route: (route.duration, route.swapCount, route.signature))
        self.cache[cacheKey] = [route.copyWithId(route.routeId) for route in routes]
        return [route.copyWithId(self.getTemporaryRouteId()) for route in routes]

    def evaluateSegment(self, startNode, tasks, endNode, includeSwap):
        data = self.modelData
        energy = data.Q
        duration = 0.0
        currentNode = startNode

        for task in tasks:
            if (currentNode, task) not in data.t or (currentNode, task) not in data.e:
                return None

            energy -= data.e[currentNode, task] + data.q[task]
            if energy < data.QMin - 1e-9:
                return None

            duration += data.t[currentNode, task] + data.p[task]
            currentNode = task

        if (currentNode, endNode) not in data.t or (currentNode, endNode) not in data.e:
            return None

        energy -= data.e[currentNode, endNode]
        if energy < data.QMin - 1e-9:
            return None

        duration += data.t[currentNode, endNode]
        if includeSwap:
            duration += data.p[self.stationNode]

        return {
            "duration": duration,
            "remainingEnergy": data.Q if includeSwap else energy,
        }

    def evaluateEmptyTailFromStation(self):
        data = self.modelData
        if (self.stationNode, data.endNode) not in data.t or (self.stationNode, data.endNode) not in data.e:
            return None

        remainingEnergy = data.Q - data.e[self.stationNode, data.endNode]
        if remainingEnergy < data.QMin - 1e-9:
            return None

        return {
            "duration": data.t[self.stationNode, data.endNode],
            "remainingEnergy": remainingEnergy,
        }

    def buildRoute(self, taskSequence, swapPositions):
        data = self.modelData
        stationNode = self.stationNode
        swapPositions = set(swapPositions)

        nodes = [data.startNode]
        swapEvents = []
        duration = 0.0
        energy = data.Q
        currentNode = data.startNode

        for index, task in enumerate(taskSequence, start=1):
            if (currentNode, task) not in data.t:
                return None

            energy -= data.e[currentNode, task] + data.q[task]
            if energy < data.QMin - 1e-9:
                return None

            duration += data.t[currentNode, task] + data.p[task]
            nodes.append(task)
            currentNode = task

            if index in swapPositions:
                if (currentNode, stationNode) not in data.t:
                    return None

                energy -= data.e[currentNode, stationNode]
                if energy < data.QMin - 1e-9:
                    return None

                arrivalTime = duration + data.t[currentNode, stationNode]
                duration = arrivalTime + data.p[stationNode]

                swapEvents.append({
                    "index": len(swapEvents),
                    "afterTask": task,
                    "arrivalTime": arrivalTime,
                    "baseStartTime": arrivalTime,
                    "baseEndTime": duration,
                    "gapFromPreviousSwap": None,
                    "tailDuration": None,
                })

                nodes.append(stationNode)
                currentNode = stationNode
                energy = data.Q

        if (currentNode, data.endNode) not in data.t:
            return None

        energy -= data.e[currentNode, data.endNode]
        if energy < data.QMin - 1e-9:
            return None

        duration += data.t[currentNode, data.endNode]
        nodes.append(data.endNode)

        self.fillSwapEventParameters(swapEvents, duration)

        return HeuristicRoute(
            routeId=self.getTemporaryRouteId(),
            tasks=taskSequence,
            nodes=nodes,
            duration=duration,
            swapEvents=swapEvents,
            stationNode=stationNode,
        )

    def fillSwapEventParameters(self, swapEvents, routeDuration):
        swapTime = self.modelData.p[self.stationNode]

        for index, event in enumerate(swapEvents):
            if index == 0:
                event["gapFromPreviousSwap"] = None
            else:
                previousEvent = swapEvents[index - 1]
                previousCompletion = previousEvent["baseEndTime"]
                event["gapFromPreviousSwap"] = event["arrivalTime"] - previousCompletion

        if len(swapEvents) > 0:
            lastEvent = swapEvents[-1]
            lastEvent["tailDuration"] = routeDuration - lastEvent["baseEndTime"]

        for event in swapEvents[:-1]:
            event["tailDuration"] = None

    @staticmethod
    def keepBestCandidates(candidates, maxVariants):
        bestBySwaps = {}
        for duration, swapPositions in candidates:
            if swapPositions not in bestBySwaps or duration < bestBySwaps[swapPositions]:
                bestBySwaps[swapPositions] = duration

        unique = [(duration, swaps) for swaps, duration in bestBySwaps.items()]
        unique.sort(key=lambda item: (item[0], len(item[1]), item[1]))
        return unique[:maxVariants]

    def getTemporaryRouteId(self):
        routeId = self.nextTemporaryRouteId
        self.nextTemporaryRouteId -= 1
        return routeId
