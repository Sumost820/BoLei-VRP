class Route:
    def __init__(self, routeId, nodes, tasks, duration, stationNode):
        self.routeId = routeId
        self.nodes = list(nodes)
        self.tasks = list(tasks)
        self.duration = duration
        self.stationNode = stationNode

        self.arcs = []
        for index in range(len(self.nodes) - 1):
            self.arcs.append((self.nodes[index], self.nodes[index + 1]))

        self.swapCount = sum(1 for node in self.nodes if node == stationNode)

    @property
    def signature(self):
        return tuple(self.nodes)

    def containsTask(self, task):
        return task in self.tasks

    def containsArc(self, arc):
        return arc in self.arcs
