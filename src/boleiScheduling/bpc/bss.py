from dataclasses import dataclass
import copy
import math
import time


@dataclass
class BssScheduleResult:
    objective: float
    status: str
    provenOptimal: bool
    routeCompletion: dict
    routeWaiting: dict
    events: list


class ExactBssScheduler:
    """Exact single-server BSS scheduler for FIXED complete-route columns.

    Every complete route column already contains every BSS visit.  Therefore the BSS
    subproblem no longer enumerates/chooses swap placements; it only schedules
    the fixed swap events of the selected routes on the one physical server.

    The result cache is permutation invariant in the selected route set and
    stores only proven-optimal schedules.
    """

    def __init__(
        self,
        modelData,
        profiler=None,
        maxEnumeratedInterleavings=250000,
    ):
        self.data = modelData
        self.profiler = profiler
        self.maxEnumeratedInterleavings = int(maxEnumeratedInterleavings)
        self._cache = {}
        self._statistics = {
            'calls': 0,
            'cacheHits': 0,
            'cacheMisses': 0,
            'provenOptimalCached': 0,
        }

    def _profileIncrement(self, name, amount=1):
        if self.profiler is not None:
            self.profiler.increment(name, amount)

    def getCacheStatistics(self):
        result = dict(self._statistics)
        result['entries'] = len(self._cache)
        return result

    def clearCache(self):
        self._cache.clear()

    def _canonicalize(self, columns):
        nonempty = [col for col in columns if not col.isEmpty]
        indexed = list(enumerate(nonempty))
        indexed.sort(key=lambda pair: tuple(pair[1].signature))
        canonicalColumns = [col for _, col in indexed]
        canonicalToOriginal = {
            canonicalIndex: originalIndex
            for canonicalIndex, (originalIndex, _) in enumerate(indexed)
        }
        key = tuple(tuple(col.signature) for col in canonicalColumns)
        return key, canonicalColumns, canonicalToOriginal

    @staticmethod
    def _remapResult(result, canonicalToOriginal):
        if not canonicalToOriginal:
            return copy.deepcopy(result)
        completion = {
            canonicalToOriginal[r]: value
            for r, value in result.routeCompletion.items()
        }
        waiting = {
            canonicalToOriginal[r]: value
            for r, value in result.routeWaiting.items()
        }
        events = []
        for raw in result.events:
            event = dict(raw)
            event['routeIndex'] = canonicalToOriginal[event['routeIndex']]
            events.append(event)
        events.sort(key=lambda e: (e['startTime'], e['routeIndex'], e['eventIndex']))
        return BssScheduleResult(
            objective=float(result.objective),
            status=result.status,
            provenOptimal=bool(result.provenOptimal),
            routeCompletion=completion,
            routeWaiting=waiting,
            events=events,
        )

    def solve(self, columns, outputFlag=0, timeLimit=None, mipGap=0.0):
        self._statistics['calls'] += 1
        self._profileIncrement('bssCalls')

        key, canonicalColumns, canonicalToOriginal = self._canonicalize(columns)
        if not canonicalColumns:
            return BssScheduleResult(
                objective=0.0,
                status='OPTIMAL',
                provenOptimal=True,
                routeCompletion={},
                routeWaiting={},
                events=[],
            )

        cached = self._cache.get(key)
        if cached is not None:
            self._statistics['cacheHits'] += 1
            self._profileIncrement('bssCacheHits')
            return self._remapResult(cached, canonicalToOriginal)

        self._statistics['cacheMisses'] += 1
        self._profileIncrement('bssCacheMisses')

        if self.profiler is None:
            canonicalResult = self._solveCanonical(
                canonicalColumns, outputFlag=outputFlag,
                timeLimit=timeLimit, mipGap=mipGap,
            )
        else:
            with self.profiler.timeBlock('bssTotal'):
                canonicalResult = self._solveCanonical(
                    canonicalColumns, outputFlag=outputFlag,
                    timeLimit=timeLimit, mipGap=mipGap,
                )

        if canonicalResult.provenOptimal:
            self._cache[key] = copy.deepcopy(canonicalResult)
            self._statistics['provenOptimalCached'] += 1
            self._profileIncrement('bssOptimalResultsCached')
        return self._remapResult(canonicalResult, canonicalToOriginal)

    @staticmethod
    def _interleavingCount(eventCounts):
        """Number of global event orders preserving each route's event chain."""
        total = sum(int(c) for c in eventCounts)
        if total <= 1:
            return 1
        result = math.factorial(total)
        for count in eventCounts:
            result //= math.factorial(int(count))
        return result

    def _evaluateInterleaving(self, columns, routeOrder):
        """Earliest-start schedule for one fixed interleaving of route chains."""
        swapTime = float(self.data.p[self.data.S[0]])
        nextIndex = [0] * len(columns)
        routeReady = [
            float(column.swapEvents[0].arrivalTime) if column.swapEvents else float('inf')
            for column in columns
        ]
        routeWaiting = [0.0] * len(columns)
        stationAvailable = 0.0
        outputEvents = []

        for r in routeOrder:
            h = nextIndex[r]
            event = columns[r].swapEvents[h]
            earliest = routeReady[r]
            start = max(stationAvailable, earliest)
            wait = max(0.0, start - earliest)
            routeWaiting[r] += wait
            end = start + swapTime
            outputEvents.append({
                'routeIndex': r,
                'eventIndex': h,
                'afterTask': event.afterTask,
                'arrivalTime': earliest,
                'startTime': start,
                'endTime': end,
                'waitTime': wait,
            })
            stationAvailable = end
            nextIndex[r] += 1
            if nextIndex[r] < len(columns[r].swapEvents):
                nextEvent = columns[r].swapEvents[nextIndex[r]]
                routeReady[r] = end + float(nextEvent.gapFromPreviousSwap)
            else:
                routeReady[r] = float('inf')

        routeCompletion = {
            r: float(column.duration + routeWaiting[r])
            for r, column in enumerate(columns)
        }
        objective = max(routeCompletion.values(), default=0.0)
        outputEvents.sort(key=lambda e: (e['startTime'], e['routeIndex'], e['eventIndex']))
        return objective, routeCompletion, dict(enumerate(routeWaiting)), outputEvents

    def _greedyInterleaving(self, columns):
        """Fast feasible order used only as a DFS incumbent."""
        swapTime = float(self.data.p[self.data.S[0]])
        counts = [len(c.swapEvents) for c in columns]
        nextIndex = [0] * len(columns)
        ready = [
            float(c.swapEvents[0].arrivalTime) if c.swapEvents else float('inf')
            for c in columns
        ]
        station = 0.0
        order = []
        while any(nextIndex[r] < counts[r] for r in range(len(columns))):
            candidates = [r for r in range(len(columns)) if nextIndex[r] < counts[r]]
            r = min(candidates, key=lambda q: (max(station, ready[q]), ready[q], q))
            h = nextIndex[r]
            start = max(station, ready[r])
            station = start + swapTime
            order.append(r)
            nextIndex[r] += 1
            if nextIndex[r] < counts[r]:
                nextEvent = columns[r].swapEvents[nextIndex[r]]
                ready[r] = station + float(nextEvent.gapFromPreviousSwap)
            else:
                ready[r] = float('inf')
        return order

    def _solveCanonicalByInterleaving(self, columns, timeLimit=None):
        """Exact fixed-event single-BSS scheduler for small event sets.

        A route's swap events form a fixed precedence chain.  The only
        combinatorial decision is therefore the interleaving of these chains on
        the one BSS server. For any fixed interleaving, scheduling every event
        at its earliest feasible start is dominant: delaying an event cannot
        make the server or its route available earlier later. Enumerating all
        chain-preserving interleavings is consequently exact.
        """
        counts = [len(column.swapEvents) for column in columns]
        totalEvents = sum(counts)
        if totalEvents == 0:
            completion = {r: float(c.duration) for r, c in enumerate(columns)}
            waiting = {r: 0.0 for r in range(len(columns))}
            return BssScheduleResult(
                objective=max(completion.values(), default=0.0),
                status='OPTIMAL_ENUMERATION',
                provenOptimal=True,
                routeCompletion=completion,
                routeWaiting=waiting,
                events=[],
            )

        greedyOrder = self._greedyInterleaving(columns)
        bestObj, bestCompletion, bestWaiting, bestEvents = self._evaluateInterleaving(
            columns, greedyOrder
        )

        swapTime = float(self.data.p[self.data.S[0]])
        nextIndex = [0] * len(columns)
        routeReady = [
            float(c.swapEvents[0].arrivalTime) if c.swapEvents else float('inf')
            for c in columns
        ]
        waiting = [0.0] * len(columns)
        sequence = []
        startWall = time.perf_counter()
        timedOut = False
        explored = 0
        eps = 1e-12

        noSwapBase = max(
            (float(c.duration) for c in columns if not c.swapEvents),
            default=0.0,
        )

        def dfs(stationAvailable):
            nonlocal bestObj, bestCompletion, bestWaiting, bestEvents
            nonlocal timedOut, explored
            if timedOut:
                return
            if timeLimit is not None and time.perf_counter() - startWall >= float(timeLimit):
                timedOut = True
                return

            remaining = totalEvents - len(sequence)
            # Every route completion is its isolated duration plus waiting
            # already incurred. This is a valid lower bound independent of how
            # the remaining events are ordered.
            lowerBound = noSwapBase
            for r, column in enumerate(columns):
                lowerBound = max(lowerBound, float(column.duration) + waiting[r])
            if remaining:
                lowerBound = max(lowerBound, stationAvailable + remaining * swapTime)
            if lowerBound >= bestObj - eps:
                return

            if remaining == 0:
                explored += 1
                obj, completion, routeWaiting, events = self._evaluateInterleaving(
                    columns, sequence
                )
                if obj < bestObj - eps:
                    bestObj = obj
                    bestCompletion = completion
                    bestWaiting = routeWaiting
                    bestEvents = events
                return

            candidates = [
                r for r in range(len(columns))
                if nextIndex[r] < counts[r]
            ]
            # Earliest start first usually gives a strong incumbent early.
            candidates.sort(key=lambda r: (max(stationAvailable, routeReady[r]), r))
            for r in candidates:
                h = nextIndex[r]
                oldReady = routeReady[r]
                oldWaiting = waiting[r]
                start = max(stationAvailable, oldReady)
                waiting[r] += max(0.0, start - oldReady)
                end = start + swapTime
                nextIndex[r] += 1
                if nextIndex[r] < counts[r]:
                    nextEvent = columns[r].swapEvents[nextIndex[r]]
                    routeReady[r] = end + float(nextEvent.gapFromPreviousSwap)
                else:
                    routeReady[r] = float('inf')
                sequence.append(r)

                dfs(end)

                sequence.pop()
                nextIndex[r] -= 1
                routeReady[r] = oldReady
                waiting[r] = oldWaiting
                if timedOut:
                    return

        dfs(0.0)
        if self.profiler is not None:
            self.profiler.increment('bssInterleavingsExplored', explored)

        return BssScheduleResult(
            objective=float(bestObj),
            status=('ENUMERATION_TIME_LIMIT' if timedOut else 'OPTIMAL_ENUMERATION'),
            provenOptimal=not timedOut,
            routeCompletion=bestCompletion,
            routeWaiting=bestWaiting,
            events=bestEvents,
        )

    def _solveCanonical(self, columns, outputFlag=0, timeLimit=None, mipGap=0.0):
        counts = [len(column.swapEvents) for column in columns]
        interleavings = self._interleavingCount(counts)
        if interleavings <= self.maxEnumeratedInterleavings:
            if self.profiler is None:
                return self._solveCanonicalByInterleaving(columns, timeLimit=timeLimit)
            with self.profiler.timeBlock('bssEnumeration'):
                return self._solveCanonicalByInterleaving(columns, timeLimit=timeLimit)

        return self._solveCanonicalMip(
            columns,
            outputFlag=outputFlag,
            timeLimit=timeLimit,
            mipGap=mipGap,
        )

    def _solveCanonicalMip(self, columns, outputFlag=0, timeLimit=None, mipGap=0.0):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError('ExactBssScheduler requires gurobipy') from error

        model = gp.Model('BPC_Exact_FixedRoute_BSS_SP')
        model.Params.OutputFlag = int(outputFlag)
        model.Params.MIPGap = float(mipGap)
        if timeLimit is not None:
            model.Params.TimeLimit = max(0.001, float(timeLimit))

        T = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='T')
        swapTime = float(self.data.p[self.data.S[0]])

        eventKeys = [
            (r, h)
            for r, column in enumerate(columns)
            for h in range(len(column.swapEvents))
        ]
        s = model.addVars(eventKeys, lb=0.0, vtype=GRB.CONTINUOUS, name='s')

        # Fixed route-event precedence and route completion.
        for r, column in enumerate(columns):
            events = column.swapEvents
            if not events:
                model.addConstr(T >= float(column.duration), name=f'completeNoSwap_{r}')
                continue

            model.addConstr(
                s[r, 0] >= float(events[0].arrivalTime),
                name=f'firstArrival_{r}',
            )
            for h in range(len(events) - 1):
                gap = float(events[h + 1].gapFromPreviousSwap)
                model.addConstr(
                    s[r, h + 1] >= s[r, h] + swapTime + gap,
                    name=f'routePrecedence_{r}_{h}',
                )
            tail = float(events[-1].tailDuration)
            last = len(events) - 1
            model.addConstr(
                T >= s[r, last] + swapTime + tail,
                name=f'complete_{r}',
            )

        # Every selected event is active.  Only cross-route event pairs need a
        # disjunction; events of one route are already ordered above.
        totalBase = sum(float(c.duration) for c in columns)
        totalEvents = len(eventKeys)
        bigM = max(
            float(self.data.M),
            2.0 * totalBase + (totalEvents + 1) * swapTime + 100.0,
        )

        orderCounter = 0
        for idx, (r1, h1) in enumerate(eventKeys):
            for r2, h2 in eventKeys[idx + 1:]:
                if r1 == r2:
                    continue
                z = model.addVar(vtype=GRB.BINARY, name=f'z_{orderCounter}')
                model.addConstr(
                    s[r2, h2] >= s[r1, h1] + swapTime - bigM * (1 - z),
                    name=f'bssForward_{orderCounter}',
                )
                model.addConstr(
                    s[r1, h1] >= s[r2, h2] + swapTime - bigM * z,
                    name=f'bssBackward_{orderCounter}',
                )
                orderCounter += 1

        model.setObjective(T, GRB.MINIMIZE)
        if self.profiler is None:
            model.optimize()
        else:
            with self.profiler.timeBlock('bssMipOptimize'):
                model.optimize()

        if model.SolCount == 0:
            return BssScheduleResult(
                objective=float('inf'),
                status=f'GUROBI_STATUS_{model.Status}',
                provenOptimal=False,
                routeCompletion={},
                routeWaiting={},
                events=[],
            )

        provenOptimal = model.Status == GRB.OPTIMAL
        objective = float(model.ObjVal) if provenOptimal else float(T.X)
        routeCompletion = {}
        routeWaiting = {}
        outputEvents = []

        for r, column in enumerate(columns):
            events = column.swapEvents
            if not events:
                routeWaiting[r] = 0.0
                routeCompletion[r] = float(column.duration)
                continue

            previousStart = None
            totalWaiting = 0.0
            for h, event in enumerate(events):
                start = float(s[r, h].X)
                if h == 0:
                    earliest = float(event.arrivalTime)
                else:
                    earliest = previousStart + swapTime + float(event.gapFromPreviousSwap)
                waiting = max(0.0, start - earliest)
                totalWaiting += waiting
                outputEvents.append({
                    'routeIndex': r,
                    'eventIndex': h,
                    'afterTask': event.afterTask,
                    'arrivalTime': earliest,
                    'startTime': start,
                    'endTime': start + swapTime,
                    'waitTime': waiting,
                })
                previousStart = start

            routeWaiting[r] = totalWaiting
            routeCompletion[r] = float(column.duration + totalWaiting)

        outputEvents.sort(key=lambda e: (e['startTime'], e['routeIndex'], e['eventIndex']))
        return BssScheduleResult(
            objective=objective,
            status='OPTIMAL' if provenOptimal else f'GUROBI_STATUS_{model.Status}',
            provenOptimal=provenOptimal,
            routeCompletion=routeCompletion,
            routeWaiting=routeWaiting,
            events=outputEvents,
        )
