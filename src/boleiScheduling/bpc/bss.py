from dataclasses import dataclass
import copy
import itertools


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
    """
    Exact BSS subproblem for a FIXED set of customer-route sequences.

    For each route sequence, ALL feasible swap placements are enumerated.
    Gurobi then simultaneously chooses one placement per route and sequences
    all chosen swap events on the single physical BSS to minimize makespan.

    This is exact for the fixed route combination, subject only to Gurobi being
    solved to proven optimality. No route-level variant truncation is used.
    """

    def __init__(self, modelData, sequenceEvaluator, maxExactRouteTasks=None):
        self.data = modelData
        self.sequenceEvaluator = sequenceEvaluator
        self.maxExactRouteTasks = maxExactRouteTasks

    def solve(self, columns, outputFlag=0, timeLimit=None, mipGap=0.0):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError('ExactBssScheduler requires gurobipy') from error

        nonempty = [column for column in columns if not column.isEmpty]
        if not nonempty:
            return BssScheduleResult(
                objective=0.0,
                status='OPTIMAL',
                provenOptimal=True,
                selectedPlanIndexByRoute={},
                routeCompletion={},
                routeWaiting={},
                events=[],
            )

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

        # Activate the selected plan, preserve within-route chronology, and
        # link the last selected event to T.
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
                    # Inactive events are forced to zero; active events are
                    # bounded by the global horizon big-M.
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

        # One physical BSS. Events belonging to different route sequences must
        # be disjunctively ordered when both corresponding plans are selected.
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
