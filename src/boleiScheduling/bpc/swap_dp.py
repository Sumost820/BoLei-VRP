from ..routePoolHeuristic.fixedRouteSwapOptimizer import FixedRouteSwapOptimizer
from .column import RouteColumn


class ExactFixedSequenceEvaluator:
    """
    Exact isolated-route evaluator for a FIXED customer order.

    Both the isolated optimum and the complete feasible swap-plan family are
    cached globally for the lifetime of the solver.  This is important in a
    branch-and-price tree: the same customer sequence is routinely encountered
    by savings, pricing, multiple branch nodes and the BSS subproblem.
    """

    def __init__(self, modelData, profiler=None):
        self.data = modelData
        self.optimizer = FixedRouteSwapOptimizer(modelData, maxVariants=1)
        self.cache = {}
        self.nextColumnId = 1
        self.profiler = profiler
        self._statistics = {
            'evaluateCalls': 0,
            'evaluateHits': 0,
            'evaluateMisses': 0,
            'allPlansCalls': 0,
            'allPlansHits': 0,
            'allPlansMisses': 0,
        }

    def _profileIncrement(self, name, amount=1):
        if self.profiler is not None:
            self.profiler.increment(name, amount)

    def enumerateAllFeasiblePlans(self, taskSequence):
        """Enumerate every feasible swap placement for a fixed customer order.

        The cached plan objects are treated as read-only by the BPC code.  A
        shallow list copy is returned, avoiding the former deep route copies on
        every cache hit while preventing callers from mutating the cached list
        container itself.
        """
        sequence = tuple(taskSequence)
        cacheKey = ('ALL_PLANS', sequence)
        self._statistics['allPlansCalls'] += 1
        self._profileIncrement('allPlanCalls')

        if cacheKey in self.cache:
            self._statistics['allPlansHits'] += 1
            self._profileIncrement('allPlanCacheHits')
            return list(self.cache[cacheKey])

        self._statistics['allPlansMisses'] += 1
        self._profileIncrement('allPlanCacheMisses')

        if len(sequence) == 0:
            self.cache[cacheKey] = tuple()
            return []

        if self.profiler is None:
            plans = self._enumeratePlansUncached(sequence)
        else:
            with self.profiler.timeBlock('allPlanEnumeration'):
                plans = self._enumeratePlansUncached(sequence)

        self.cache[cacheKey] = tuple(plans)
        return list(plans)

    def _enumeratePlansUncached(self, sequence):
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

        return sorted(
            bestBySignature.values(),
            key=lambda route: (route.duration, route.swapCount, tuple(route.nodes)),
        )

    def evaluate(self, taskSequence):
        sequence = tuple(taskSequence)
        self._statistics['evaluateCalls'] += 1
        self._profileIncrement('routeEvaluationCalls')

        if sequence in self.cache:
            self._statistics['evaluateHits'] += 1
            self._profileIncrement('routeEvaluationCacheHits')
            return self.cache[sequence]

        self._statistics['evaluateMisses'] += 1
        self._profileIncrement('routeEvaluationCacheMisses')

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

        if self.profiler is None:
            routes = self.optimizer.optimize(sequence, maxVariants=1)
        else:
            with self.profiler.timeBlock('routeEvaluation'):
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

    def getCacheStatistics(self):
        result = dict(self._statistics)
        result['entries'] = len(self.cache)
        return result
