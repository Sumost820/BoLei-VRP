from ..routePoolHeuristic.fixedRouteSwapOptimizer import FixedRouteSwapOptimizer
from .column import RouteColumn


class ExactFixedSequenceEvaluator:
    """
    Exact isolated-route evaluator for a FIXED customer order.

    It minimizes base duration over all feasible swap positions. Because every
    state at a swap boundary has the same full battery Q, keeping only the
    minimum-duration predecessor for a fixed customer prefix is exact for the
    minimum base-duration problem.

    Base duration includes travel, customer service, BSS detour and fixed swap
    times, but excludes waiting caused by other vehicles at the shared BSS.
    """

    def __init__(self, modelData):
        self.data = modelData
        self.optimizer = FixedRouteSwapOptimizer(modelData, maxVariants=1)
        self.cache = {}
        self.nextColumnId = 1


    def enumerateAllFeasiblePlans(self, taskSequence):
        """
        Enumerate EVERY feasible swap placement for a fixed customer order.

        Swap positions are subsets of {1,...,m}, where position j means that
        the vehicle visits the physical BSS immediately after customer j. A
        swap after the last task is retained because it can be necessary when
        the direct last-customer -> depot arc violates the battery lower bound.

        This method is exponential and is used only by the exact small-instance
        BSS reference solver in BPC V0.2. No maxVariants truncation is applied.
        """
        sequence = tuple(taskSequence)
        cacheKey = ('ALL_PLANS', sequence)
        if cacheKey in self.cache:
            return [route.copyWithId(route.routeId) for route in self.cache[cacheKey]]

        if len(sequence) == 0:
            self.cache[cacheKey] = []
            return []

        plans = []
        m = len(sequence)
        for mask in range(1 << m):
            swapPositions = tuple(
                position + 1
                for position in range(m)
                if mask & (1 << position)
            )
            route = self.optimizer.buildRoute(sequence, swapPositions)
            if route is not None:
                plans.append(route)

        # Defensive deduplication by the full physical node sequence.
        bestBySignature = {}
        for route in plans:
            signature = tuple(route.nodes)
            incumbent = bestBySignature.get(signature)
            if incumbent is None or route.duration < incumbent.duration:
                bestBySignature[signature] = route

        plans = sorted(
            bestBySignature.values(),
            key=lambda route: (route.duration, route.swapCount, tuple(route.nodes)),
        )
        self.cache[cacheKey] = [route.copyWithId(route.routeId) for route in plans]
        return [route.copyWithId(route.routeId) for route in plans]

    def evaluate(self, taskSequence):
        sequence = tuple(taskSequence)
        if sequence in self.cache:
            return self.cache[sequence]

        if len(sequence) == 0:
            column = RouteColumn(
                columnId=0,
                tasks=(),
                duration=0.0,
                baseNodes=(self.data.startNode, self.data.endNode),
                baseSwapEvents=(),
            )
            self.cache[sequence] = column
            return column

        routes = self.optimizer.optimize(sequence, maxVariants=1)
        if not routes:
            self.cache[sequence] = None
            return None

        route = routes[0]
        events = tuple(
            (
                event['index'],
                event['afterTask'],
                float(event['arrivalTime']),
                float(event['baseStartTime']),
                float(event['baseEndTime']),
            )
            for event in route.swapEvents
        )

        column = RouteColumn(
            columnId=self.nextColumnId,
            tasks=sequence,
            duration=float(route.duration),
            baseNodes=tuple(route.nodes),
            baseSwapEvents=events,
        )
        self.nextColumnId += 1
        self.cache[sequence] = column
        return column
