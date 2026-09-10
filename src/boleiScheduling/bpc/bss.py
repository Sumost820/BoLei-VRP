from dataclasses import dataclass
import copy


@dataclass
class BssScheduleResult:
    objective: float
    status: str
    provenOptimal: bool
    selectedPlanIndexByRoute: dict
    routeCompletion: dict
    routeWaiting: dict
    events: list


class ExactBssScheduler:
    """Exact single-BSS subproblem with a permutation-invariant result cache.

    The cache key is the sorted tuple of selected nonempty customer-sequence
    signatures.  A cached result is stored in that canonical route order and is
    remapped back to the caller's route order before returning.  Only proven
    optimal results are cached; a time-limit incumbent is never reused as an
    exact BSS cut value.
    """

    def __init__(
        self,
        modelData,
        sequenceEvaluator,
        maxExactRouteTasks=None,
        profiler=None,
    ):
        self.data = modelData
        self.sequenceEvaluator = sequenceEvaluator
        self.maxExactRouteTasks = maxExactRouteTasks
        self.profiler = profiler
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

        selected = {
            canonicalToOriginal[r]: v
            for r, v in result.selectedPlanIndexByRoute.items()
        }
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
        events.sort(key=lambda event: (event['startTime'], event['routeIndex']))

        return BssScheduleResult(
            objective=float(result.objective),
            status=result.status,
            provenOptimal=bool(result.provenOptimal),
            selectedPlanIndexByRoute=selected,
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
                selectedPlanIndexByRoute={},
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
                canonicalColumns,
                outputFlag=outputFlag,
                timeLimit=timeLimit,
                mipGap=mipGap,
            )
        else:
            with self.profiler.timeBlock('bssTotal'):
                canonicalResult = self._solveCanonical(
                    canonicalColumns,
                    outputFlag=outputFlag,
                    timeLimit=timeLimit,
                    mipGap=mipGap,
                )

        if canonicalResult.provenOptimal:
            self._cache[key] = copy.deepcopy(canonicalResult)
            self._statistics['provenOptimalCached'] += 1
            self._profileIncrement('bssOptimalResultsCached')

        return self._remapResult(canonicalResult, canonicalToOriginal)

    def _solveCanonical(self, nonempty, outputFlag=0, timeLimit=None, mipGap=0.0):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError('ExactBssScheduler requires gurobipy') from error

        allPlans = {}
        for r, column in enumerate(nonempty):
            if self.maxExactRouteTasks is not None and len(column.tasks) > self.maxExactRouteTasks:
                raise ValueError(
                    f'Exact BSS placement enumeration is a small-N reference. '
                    f'Route {column.tasks} has {len(column.tasks)} tasks, '
                    f'exceeding maxExactRouteTasks={self.maxExactRouteTasks}.'
                )
            plans = self.sequenceEvaluator.enumerateAllFeasiblePlans(column.tasks)
            if not plans:
                raise RuntimeError(f'No feasible swap placement for route {column.tasks}')
            allPlans[r] = plans

        model = gp.Model('BPC_Exact_BSS_SP')
        model.Params.OutputFlag = outputFlag
        model.Params.MIPGap = mipGap
        if timeLimit is not None:
            model.Params.TimeLimit = timeLimit

        yKeys = [
            (r, v)
            for r, plans in allPlans.items()
            for v in range(len(plans))
        ]
        y = model.addVars(yKeys, vtype=GRB.BINARY, name='y')
        T = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='T')

        for r, plans in allPlans.items():
            model.addConstr(
                gp.quicksum(y[r, v] for v in range(len(plans))) == 1,
                name=f'choosePlan_{r}',
            )

        eventKeys = []
        for r, plans in allPlans.items():
            for v, plan in enumerate(plans):
                for h, _ in enumerate(plan.swapEvents):
                    eventKeys.append((r, v, h))

        s = model.addVars(eventKeys, lb=0.0, vtype=GRB.CONTINUOUS, name='s')

        swapTime = self.data.p[self.data.S[0]]
        maxRouteDuration = sum(
            max(plan.duration for plan in plans)
            for plans in allPlans.values()
        )
        totalPotentialEvents = len(eventKeys)
        bigM = max(
            float(self.data.M),
            2.0 * maxRouteDuration + (totalPotentialEvents + 1) * swapTime + 100.0,
        )

        for r, plans in allPlans.items():
            for v, plan in enumerate(plans):
                active = y[r, v]

                if len(plan.swapEvents) == 0:
                    model.addConstr(
                        T >= plan.duration - bigM * (1 - active),
                        name=f'completeNoSwap_{r}_{v}',
                    )
                    continue

                first = plan.swapEvents[0]
                model.addConstr(
                    s[r, v, 0] >= first['arrivalTime'] - bigM * (1 - active),
                    name=f'firstArrival_{r}_{v}',
                )

                for h in range(len(plan.swapEvents)):
                    model.addConstr(
                        s[r, v, h] <= bigM * active,
                        name=f'activateStart_{r}_{v}_{h}',
                    )

                for h in range(len(plan.swapEvents) - 1):
                    nextEvent = plan.swapEvents[h + 1]
                    gap = nextEvent['gapFromPreviousSwap']
                    model.addConstr(
                        s[r, v, h + 1]
                        >= s[r, v, h] + swapTime + gap - bigM * (1 - active),
                        name=f'routePrecedence_{r}_{v}_{h}',
                    )

                tail = plan.swapEvents[-1]['tailDuration']
                lastH = len(plan.swapEvents) - 1
                model.addConstr(
                    T >= s[r, v, lastH] + swapTime + tail - bigM * (1 - active),
                    name=f'complete_{r}_{v}',
                )

        orderCounter = 0
        for idx, firstKey in enumerate(eventKeys):
            r1, v1, h1 = firstKey
            for secondKey in eventKeys[idx + 1:]:
                r2, v2, h2 = secondKey
                if r1 == r2:
                    continue

                z = model.addVar(vtype=GRB.BINARY, name=f'z_{orderCounter}')
                activationRelax = bigM * (2 - y[r1, v1] - y[r2, v2])

                model.addConstr(
                    s[r2, v2, h2]
                    >= s[r1, v1, h1] + swapTime
                    - bigM * (1 - z)
                    - activationRelax,
                    name=f'bssForward_{orderCounter}',
                )
                model.addConstr(
                    s[r1, v1, h1]
                    >= s[r2, v2, h2] + swapTime
                    - bigM * z
                    - activationRelax,
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
                selectedPlanIndexByRoute={},
                routeCompletion={},
                routeWaiting={},
                events=[],
            )

        provenOptimal = model.Status == GRB.OPTIMAL
        objective = float(model.ObjVal) if provenOptimal else float(T.X)

        selected = {}
        routeCompletion = {}
        routeWaiting = {}
        events = []

        for r, plans in allPlans.items():
            v = max(range(len(plans)), key=lambda idx: y[r, idx].X)
            selected[r] = v
            plan = plans[v]

            if not plan.swapEvents:
                routeCompletion[r] = float(plan.duration)
                routeWaiting[r] = 0.0
                continue

            previousStart = None
            totalWaiting = 0.0
            for h, event in enumerate(plan.swapEvents):
                start = float(s[r, v, h].X)
                if h == 0:
                    earliest = float(event['arrivalTime'])
                else:
                    earliest = (
                        previousStart
                        + swapTime
                        + float(event['gapFromPreviousSwap'])
                    )
                waiting = max(0.0, start - earliest)
                totalWaiting += waiting
                events.append({
                    'routeIndex': r,
                    'planIndex': v,
                    'eventIndex': h,
                    'afterTask': event['afterTask'],
                    'arrivalTime': earliest,
                    'startTime': start,
                    'endTime': start + swapTime,
                    'waitTime': waiting,
                })
                previousStart = start

            routeWaiting[r] = totalWaiting
            routeCompletion[r] = float(plan.duration + totalWaiting)

        events.sort(key=lambda event: (event['startTime'], event['routeIndex']))
        return BssScheduleResult(
            objective=objective,
            status='OPTIMAL' if provenOptimal else f'GUROBI_STATUS_{model.Status}',
            provenOptimal=provenOptimal,
            selectedPlanIndexByRoute=selected,
            routeCompletion=routeCompletion,
            routeWaiting=routeWaiting,
            events=events,
        )
