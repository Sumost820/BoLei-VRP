class HeuristicRoute:
    def __init__(self, routeId, tasks, nodes, duration, swapEvents, stationNode):
        self.routeId = routeId
        self.tasks = list(tasks)
        self.nodes = list(nodes)
        self.duration = float(duration)
        self.swapEvents = [dict(event) for event in swapEvents]
        self.stationNode = stationNode

        self.swapCount = len(self.swapEvents)
        self.taskSet = frozenset(self.tasks)
        self.signature = tuple(self.nodes)

    def containsTask(self, task):
        return task in self.taskSet

    def getTaskCount(self):
        return len(self.tasks)

    def copyWithId(self, routeId):
        return HeuristicRoute(
            routeId=routeId,
            tasks=self.tasks,
            nodes=self.nodes,
            duration=self.duration,
            swapEvents=self.swapEvents,
            stationNode=self.stationNode,
        )

    def toDict(self):
        return {
            "routeId": self.routeId,
            "tasks": list(self.tasks),
            "nodes": list(self.nodes),
            "duration": self.duration,
            "swapCount": self.swapCount,
            "swapEvents": [dict(event) for event in self.swapEvents],
        }
