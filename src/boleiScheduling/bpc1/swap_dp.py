from ..routePoolHeuristic.fixedRouteSwapOptimizer import FixedRouteSwapOptimizer
from .column import RouteColumn, SwapEvent


class ExactCompleteRouteEvaluator:
    """Exact evaluator/factory for complete physical route columns.

    The primary identity is the full physical node sequence.  The physical BSS
    is ``modelData.S[0]`` and may occur repeatedly; customer nodes remain
    elementary.  ``evaluate(taskSequence)`` is retained as a warm-start helper:
    it returns the isolated-minimum complete route for that customer order.
    Pricing itself uses ``evaluateNodes`` so every distinct swap placement can
    become its own column.
    """

    def __init__(self, modelData, profiler=None):
        self.data = modelData
        if not modelData.S:
            raise ValueError('BPC complete-route pricing requires one physical BSS node')
        self.stationNode = modelData.S[0]
        self.optimizer = FixedRouteSwapOptimizer(modelData, maxVariants=1)
        self.profiler = profiler
        self.nextColumnId = 1

        self._nodeCache = {}
        self._bestSequenceCache = {}
        self._allPlanCache = {}
        self._statistics = {
            'evaluateCalls': 0,
            'evaluateHits': 0,
            'evaluateMisses': 0,
            'evaluateNodesCalls': 0,
            'evaluateNodesHits': 0,
            'evaluateNodesMisses': 0,
            'allPlansCalls': 0,
            'allPlansHits': 0,
            'allPlansMisses': 0,
        }

        # Empty route is a formal master column; no physical start->end arc is
        # required in the input graph.
        empty = RouteColumn(
            columnId=0,
            nodes=(self.data.startNode, self.data.endNode),
            tasks=(),
            duration=0.0,
            swapEvents=(),
        )
        self._nodeCache[empty.signature] = empty
        self._bestSequenceCache[()] = empty

    def _profileIncrement(self, name, amount=1):
        if self.profiler is not None:
            self.profiler.increment(name, amount)

    def _newColumnId(self):
        value = self.nextColumnId
        self.nextColumnId += 1
        return value

    def evaluate(self, taskSequence):
        """Return the isolated-minimum COMPLETE route for a fixed customer order.

        This method is used by Savings/short-route warm starts only.  Different
        swap placements of the same customer sequence are not collapsed in the
        global column pool because the returned column signature is its full
        node sequence; pricing can later add other complete variants.
        """
        sequence = tuple(taskSequence)
        self._statistics['evaluateCalls'] += 1
        self._profileIncrement('routeEvaluationCalls')

        if sequence in self._bestSequenceCache:
            self._statistics['evaluateHits'] += 1
            self._profileIncrement('routeEvaluationCacheHits')
            return self._bestSequenceCache[sequence]

        self._statistics['evaluateMisses'] += 1
        self._profileIncrement('routeEvaluationCacheMisses')

        if self.profiler is None:
            routes = self.optimizer.optimize(sequence, maxVariants=1)
        else:
            with self.profiler.timeBlock('routeEvaluation'):
                routes = self.optimizer.optimize(sequence, maxVariants=1)

        if not routes:
            self._bestSequenceCache[sequence] = None
            return None

        column = self.evaluateNodes(tuple(routes[0].nodes))
        self._bestSequenceCache[sequence] = column
        return column

    def evaluateNodes(self, nodes):
        """Validate and create one complete physical route column.

        Energy convention matches the existing route optimizer: on a
        customer-arrival arc, travel energy and task energy are both consumed,
        then the QMin bound is checked.  Upon reaching the BSS, travel energy is
        consumed and checked, swap service time is added, and SOC resets to Q.
        """
        nodes = tuple(nodes)
        self._statistics['evaluateNodesCalls'] += 1
        self._profileIncrement('completeRouteEvaluationCalls')

        cached = self._nodeCache.get(nodes, '__MISSING__')
        if cached != '__MISSING__':
            self._statistics['evaluateNodesHits'] += 1
            self._profileIncrement('completeRouteEvaluationCacheHits')
            return cached

        self._statistics['evaluateNodesMisses'] += 1
        self._profileIncrement('completeRouteEvaluationCacheMisses')

        if self.profiler is None:
            column = self._evaluateNodesUncached(nodes)
        else:
            with self.profiler.timeBlock('completeRouteEvaluation'):
                column = self._evaluateNodesUncached(nodes)

        self._nodeCache[nodes] = column
        return column

    def _evaluateNodesUncached(self, nodes):
        data = self.data
        S = self.stationNode

        if nodes == (data.startNode, data.endNode):
            return self._nodeCache[(data.startNode, data.endNode)]
        if len(nodes) < 3:
            return None
        if nodes[0] != data.startNode or nodes[-1] != data.endNode:
            return None
        if data.startNode in nodes[1:] or data.endNode in nodes[:-1]:
            return None
        if nodes[1] == S:
            # Starting with an immediate swap is dominated and is outside the
            # intended route graph.
            return None

        customers = []
        seenCustomers = set()
        duration = 0.0
        energy = float(data.Q)
        swapRaw = []

        current = nodes[0]
        for position, nxt in enumerate(nodes[1:], start=1):
            if nxt == S and current == S:
                return None
            if nxt in data.S and nxt != S:
                # All station copies represent the same physical BSS. Complete
                # route columns use only the canonical physical node S[0].
                return None
            if nxt not in data.C and nxt not in (S, data.endNode):
                return None

            arc = (current, nxt)
            if arc not in data.t or arc not in data.e:
                return None

            energy -= float(data.e[arc])
            duration += float(data.t[arc])

            if nxt in data.C:
                if nxt in seenCustomers:
                    return None
                seenCustomers.add(nxt)
                customers.append(nxt)
                energy -= float(data.q[nxt])
                if energy < data.QMin - 1e-9:
                    return None
                duration += float(data.p[nxt])

            elif nxt == S:
                # In the current graph a BSS visit must follow a customer.
                if current not in data.C:
                    return None
                if energy < data.QMin - 1e-9:
                    return None
                arrival = duration
                start = arrival
                end = arrival + float(data.p[S])
                swapRaw.append({
                    'index': len(swapRaw),
                    'afterTask': current,
                    'arrivalTime': arrival,
                    'baseStartTime': start,
                    'baseEndTime': end,
                })
                duration = end
                energy = float(data.Q)

            else:  # end depot
                if energy < data.QMin - 1e-9:
                    return None
                if position != len(nodes) - 1:
                    return None

            current = nxt

        if not customers:
            return None

        events = []
        for index, raw in enumerate(swapRaw):
            if index == 0:
                gap = None
            else:
                gap = raw['arrivalTime'] - swapRaw[index - 1]['baseEndTime']
            tail = (
                duration - raw['baseEndTime']
                if index == len(swapRaw) - 1
                else None
            )
            events.append(SwapEvent(
                index=index,
                afterTask=int(raw['afterTask']),
                arrivalTime=float(raw['arrivalTime']),
                baseStartTime=float(raw['baseStartTime']),
                baseEndTime=float(raw['baseEndTime']),
                gapFromPreviousSwap=(None if gap is None else float(gap)),
                tailDuration=(None if tail is None else float(tail)),
            ))

        return RouteColumn(
            columnId=self._newColumnId(),
            nodes=nodes,
            tasks=tuple(customers),
            duration=float(duration),
            swapEvents=tuple(events),
        )

    def enumerateAllFeasiblePlans(self, taskSequence):
        """Return every feasible complete route for a fixed customer order.

        This method is used only by the exact small-instance enumerative
        pricing oracle. The BSS scheduler never enumerates swap placements: a
        complete route column already fixes every BSS visit.
        """
        sequence = tuple(taskSequence)
        self._statistics['allPlansCalls'] += 1
        self._profileIncrement('allPlanCalls')
        if sequence in self._allPlanCache:
            self._statistics['allPlansHits'] += 1
            self._profileIncrement('allPlanCacheHits')
            return list(self._allPlanCache[sequence])

        self._statistics['allPlansMisses'] += 1
        self._profileIncrement('allPlanCacheMisses')
        if not sequence:
            self._allPlanCache[sequence] = tuple()
            return []

        result = {}
        m = len(sequence)
        if self.profiler is None:
            iterator = range(1 << m)
            for mask in iterator:
                swaps = tuple(pos + 1 for pos in range(m) if mask & (1 << pos))
                route = self.optimizer.buildRoute(sequence, swaps)
                if route is None:
                    continue
                column = self.evaluateNodes(tuple(route.nodes))
                if column is not None:
                    result[column.signature] = column
        else:
            with self.profiler.timeBlock('allPlanEnumeration'):
                for mask in range(1 << m):
                    swaps = tuple(pos + 1 for pos in range(m) if mask & (1 << pos))
                    route = self.optimizer.buildRoute(sequence, swaps)
                    if route is None:
                        continue
                    column = self.evaluateNodes(tuple(route.nodes))
                    if column is not None:
                        result[column.signature] = column

        columns = sorted(
            result.values(),
            key=lambda c: (c.duration, c.stationVisitCount, c.nodes),
        )
        self._allPlanCache[sequence] = tuple(columns)
        return list(columns)

    def getCacheStatistics(self):
        result = dict(self._statistics)
        result.update({
            'nodeEntries': len(self._nodeCache),
            'bestSequenceEntries': len(self._bestSequenceCache),
            'allPlanEntries': len(self._allPlanCache),
        })
        return result

