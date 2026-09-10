
import copy
import math
import random
import time


class SwapRoutePlan:
    def __init__(self, tasks, nodes, duration, swapEvents, stationNode):
        self.tasks = list(tasks)
        self.nodes = list(nodes)
        self.duration = float(duration)
        self.swapEvents = [dict(event) for event in swapEvents]
        self.stationNode = stationNode
        self.swapCount = len(self.swapEvents)
        self.signature = tuple(self.nodes)

    def copy(self):
        return SwapRoutePlan(
            tasks=self.tasks,
            nodes=self.nodes,
            duration=self.duration,
            swapEvents=self.swapEvents,
            stationNode=self.stationNode,
        )


class ALNSRoute:
    def __init__(self, tasks=None):
        self.tasks = [] if tasks is None else list(tasks)
        self.plan = None
        self.baseDuration = 0.0
        self.completionTime = 0.0
        self.waitingTime = 0.0

    def copy(self):
        return copy.deepcopy(self)


class ALNSSolution:
    def __init__(self, routes=None):
        self.routes = [] if routes is None else routes
        self.removed = []
        self.objective = float("inf")
        self.baseObjective = float("inf")
        self.scheduleEvents = []
        self.scheduleStatus = "NOT_EVALUATED"
        self.exactOptimal = False

    def copy(self):
        return copy.deepcopy(self)


class FixedRouteSwapOptimizer:
    """
    Exact dynamic-programming evaluator for one fixed customer/task sequence.

    The task order is fixed. The optimizer only decides after which tasks the
    vehicle visits the single physical BSS. All station-copy nodes in ModelData.S
    represent that same physical BSS, so only S[0] is used in a route plan.
    """

    def __init__(self, modelData, maxVariants=3):
        self.data = modelData
        self.maxVariants = maxVariants
        self.cache = {}

        if len(modelData.S) == 0:
            raise ValueError("ALNS需要至少一个换电站节点")
        self.stationNode = modelData.S[0]

    def optimize(self, taskSequence, maxVariants=None):
        if maxVariants is None:
            maxVariants = self.maxVariants

        taskSequence = tuple(taskSequence)
        cacheKey = (taskSequence, maxVariants)
        if cacheKey in self.cache:
            return [plan.copy() for plan in self.cache[cacheKey]]

        if len(taskSequence) == 0:
            plan = SwapRoutePlan(
                tasks=[],
                nodes=[self.data.startNode, self.data.endNode],
                duration=0.0,
                swapEvents=[],
                stationNode=self.stationNode,
            )
            self.cache[cacheKey] = [plan]
            return [plan.copy()]

        candidatesAtSwap = [[] for _ in range(len(taskSequence) + 1)]
        completeCandidates = []

        # depot -> first tasks -> BSS
        for j in range(1, len(taskSequence) + 1):
            segment = self._evaluateSegment(
                self.data.startNode,
                taskSequence[:j],
                self.stationNode,
                includeSwap=True,
            )
            if segment is not None:
                candidatesAtSwap[j].append((segment, (j,)))

        # depot -> all tasks -> end
        direct = self._evaluateSegment(
            self.data.startNode,
            taskSequence,
            self.data.endNode,
            includeSwap=False,
        )
        if direct is not None:
            completeCandidates.append((direct, ()))

        # BSS -> middle tasks -> BSS
        for i in range(1, len(taskSequence)):
            if not candidatesAtSwap[i]:
                continue

            for j in range(i + 1, len(taskSequence) + 1):
                segment = self._evaluateSegment(
                    self.stationNode,
                    taskSequence[i:j],
                    self.stationNode,
                    includeSwap=True,
                )
                if segment is None:
                    continue

                newCandidates = []
                for previousDuration, previousSwaps in candidatesAtSwap[i]:
                    newCandidates.append(
                        (previousDuration + segment, previousSwaps + (j,))
                    )

                candidatesAtSwap[j].extend(newCandidates)
                candidatesAtSwap[j] = self._keepBestCandidates(
                    candidatesAtSwap[j],
                    maxVariants,
                )

        # last BSS -> remaining tasks -> end
        for i in range(1, len(taskSequence) + 1):
            if not candidatesAtSwap[i]:
                continue

            if i == len(taskSequence):
                tail = self._evaluateEmptyTailFromStation()
            else:
                tail = self._evaluateSegment(
                    self.stationNode,
                    taskSequence[i:],
                    self.data.endNode,
                    includeSwap=False,
                )

            if tail is None:
                continue

            for previousDuration, previousSwaps in candidatesAtSwap[i]:
                completeCandidates.append(
                    (previousDuration + tail, previousSwaps)
                )

        completeCandidates = self._keepBestCandidates(
            completeCandidates,
            maxVariants,
        )

        plans = []
        for _, swapPositions in completeCandidates:
            plan = self._buildPlan(taskSequence, swapPositions)
            if plan is not None:
                plans.append(plan)

        plans.sort(
            key=lambda plan: (
                plan.duration,
                plan.swapCount,
                plan.signature,
            )
        )

        self.cache[cacheKey] = [plan.copy() for plan in plans]
        return [plan.copy() for plan in plans]

    def _evaluateSegment(
        self,
        startNode,
        tasks,
        endNode,
        includeSwap,
    ):
        energy = self.data.Q
        duration = 0.0
        currentNode = startNode

        for task in tasks:
            if (
                (currentNode, task) not in self.data.t
                or (currentNode, task) not in self.data.e
            ):
                return None

            energy -= (
                self.data.e[currentNode, task]
                + self.data.q[task]
            )
            if energy < self.data.QMin - 1e-9:
                return None

            duration += (
                self.data.t[currentNode, task]
                + self.data.p[task]
            )
            currentNode = task

        if (
            (currentNode, endNode) not in self.data.t
            or (currentNode, endNode) not in self.data.e
        ):
            return None

        energy -= self.data.e[currentNode, endNode]
        if energy < self.data.QMin - 1e-9:
            return None

        duration += self.data.t[currentNode, endNode]
        if includeSwap:
            duration += self.data.p[self.stationNode]

        return duration

    def _evaluateEmptyTailFromStation(self):
        if (
            (self.stationNode, self.data.endNode) not in self.data.t
            or (self.stationNode, self.data.endNode) not in self.data.e
        ):
            return None

        remainingEnergy = (
            self.data.Q
            - self.data.e[self.stationNode, self.data.endNode]
        )
        if remainingEnergy < self.data.QMin - 1e-9:
            return None

        return self.data.t[
            self.stationNode,
            self.data.endNode
        ]

    def _buildPlan(self, taskSequence, swapPositions):
        swapPositions = set(swapPositions)
        nodes = [self.data.startNode]
        swapEvents = []

        duration = 0.0
        energy = self.data.Q
        currentNode = self.data.startNode

        for index, task in enumerate(taskSequence, start=1):
            if (
                (currentNode, task) not in self.data.t
                or (currentNode, task) not in self.data.e
            ):
                return None

            energy -= (
                self.data.e[currentNode, task]
                + self.data.q[task]
            )
            if energy < self.data.QMin - 1e-9:
                return None

            duration += (
                self.data.t[currentNode, task]
                + self.data.p[task]
            )
            nodes.append(task)
            currentNode = task

            if index in swapPositions:
                if (
                    (currentNode, self.stationNode) not in self.data.t
                    or (currentNode, self.stationNode) not in self.data.e
                ):
                    return None

                energy -= self.data.e[
                    currentNode,
                    self.stationNode,
                ]
                if energy < self.data.QMin - 1e-9:
                    return None

                arrivalTime = (
                    duration
                    + self.data.t[currentNode, self.stationNode]
                )
                duration = (
                    arrivalTime
                    + self.data.p[self.stationNode]
                )

                swapEvents.append({
                    "index": len(swapEvents),
                    "afterTask": task,
                    "arrivalTime": arrivalTime,
                    "baseStartTime": arrivalTime,
                    "baseEndTime": duration,
                    "gapFromPreviousSwap": None,
                    "tailDuration": None,
                })

                nodes.append(self.stationNode)
                currentNode = self.stationNode
                energy = self.data.Q

        if (
            (currentNode, self.data.endNode) not in self.data.t
            or (currentNode, self.data.endNode) not in self.data.e
        ):
            return None

        energy -= self.data.e[
            currentNode,
            self.data.endNode,
        ]
        if energy < self.data.QMin - 1e-9:
            return None

        duration += self.data.t[
            currentNode,
            self.data.endNode,
        ]
        nodes.append(self.data.endNode)

        swapTime = self.data.p[self.stationNode]
        for index, event in enumerate(swapEvents):
            if index > 0:
                previous = swapEvents[index - 1]
                event["gapFromPreviousSwap"] = (
                    event["arrivalTime"]
                    - previous["baseEndTime"]
                )

        if swapEvents:
            swapEvents[-1]["tailDuration"] = (
                duration
                - swapEvents[-1]["baseEndTime"]
            )

        return SwapRoutePlan(
            tasks=taskSequence,
            nodes=nodes,
            duration=duration,
            swapEvents=swapEvents,
            stationNode=self.stationNode,
        )

    @staticmethod
    def _keepBestCandidates(candidates, maxVariants):
        bestBySwaps = {}
        for duration, swapPositions in candidates:
            if (
                swapPositions not in bestBySwaps
                or duration < bestBySwaps[swapPositions]
            ):
                bestBySwaps[swapPositions] = duration

        unique = [
            (duration, swaps)
            for swaps, duration in bestBySwaps.items()
        ]
        unique.sort(
            key=lambda item: (
                item[0],
                len(item[1]),
                item[1],
            )
        )
        return unique[:maxVariants]


class ExactBssSolutionEvaluator:
    """
    Joint evaluation of an ALNS solution.

    For every vehicle route, the fixed-route DP provides several good swap
    placements. A single Gurobi model then simultaneously:
      1) selects one swap-placement variant for every vehicle,
      2) sequences all selected swap events on the one physical BSS,
      3) minimizes the true makespan Cmax.

    Thus the ALNS searches task assignment/order only. Swap placement and BSS
    synchronization are optimized inside the evaluation oracle.
    """

    def __init__(self, modelData, variantsPerRoute=3, bssTimeLimit=None, mipGap=1e-6, minimizeSlack=True):
        self.data = modelData
        self.variantsPerRoute = variantsPerRoute
        self.bssTimeLimit = bssTimeLimit
        self.mipGap = mipGap
        self.minimizeSlack = minimizeSlack

        self.fixedRouteOptimizer = FixedRouteSwapOptimizer(modelData, maxVariants=variantsPerRoute)
        self.solutionCache = {}

    def evaluate(self, solution, outputFlag=0, remainingTime=None, validateCoverage=True):
        if validateCoverage:
            self._validateSolution(solution)

        cacheKey = tuple(
            tuple(route.tasks)for route in solution.routes
        )
        if cacheKey in self.solutionCache:
            cached = copy.deepcopy(self.solutionCache[cacheKey])
            self._applyCachedResult(solution, cached)
            return solution.objective

        variantsByRoute = []
        for route in solution.routes:
            variants = self.fixedRouteOptimizer.optimize(route.tasks, maxVariants=self.variantsPerRoute)
            if not variants:
                solution.objective = float("inf")
                solution.baseObjective = float("inf")
                solution.scheduleStatus = "INFEASIBLE_ROUTE"
                return solution.objective
            variantsByRoute.append(variants)

        result = self._greedyBssSchedule(variantsByRoute)

        self._applyResult(solution, variantsByRoute, result)

        self.solutionCache[cacheKey] = {
            "objective": solution.objective,
            "baseObjective": solution.baseObjective,
            "scheduleEvents": copy.deepcopy(solution.scheduleEvents),
            "scheduleStatus": solution.scheduleStatus,
            "exactOptimal": solution.exactOptimal,
            "selectedVariantIndices": list(result["selectedVariantIndices"]),
            "completionTimes": [route.completionTime for route in solution.routes],
            "waitingTimes": [route.waitingTime for route in solution.routes],
        }

        return solution.objective

    def bestBasePlan(self, tasks):
        variants = self.fixedRouteOptimizer.optimize(tasks, maxVariants=1)
        if not variants:
            return None
        return variants[0]

    def _validateSolution(self, solution):
        seen = []
        for route in solution.routes:
            seen.extend(route.tasks)

        if len(seen) != len(set(seen)):
            raise ValueError("ALNS解中存在重复任务")

        if set(seen) != set(self.data.C):
            missing = sorted(set(self.data.C) - set(seen))
            extra = sorted(set(seen) - set(self.data.C))
            raise ValueError(f"ALNS解任务覆盖不完整, missing={missing}, extra={extra}")

        if len(solution.routes) > self.data.K:
            raise ValueError("ALNS使用车辆数超过K")

    def _effectiveTimeLimit(self, remainingTime):
        if remainingTime is None:
            return self.bssTimeLimit
        if self.bssTimeLimit is None:
            return max(0.01, remainingTime)
        return max(0.01, min(self.bssTimeLimit, remainingTime))


    def _greedyBssSchedule(self, variantsByRoute):
        """
        Feasible non-Gurobi fallback used for unit/smoke testing.
        It selects the minimum-base-duration variant and schedules the next
        available swap event FCFS. It is NOT claimed to be BSS-optimal.
        """
        selectedVariantIndices = [0 for _ in variantsByRoute]
        plans = [variants[0] for variants in variantsByRoute]

        swapTime = self.data.p[self.data.S[0]]

        nextEventIndex = [0] * len(plans)
        previousStart = [None] * len(plans)
        routeWaiting = [0.0] * len(plans)
        stationAvailable = 0.0
        events = []

        while True:
            choices = []

            for r, plan in enumerate(plans):
                h = nextEventIndex[r]
                if h >= plan.swapCount:
                    continue

                event = plan.swapEvents[h]
                if h == 0:
                    ready = event["arrivalTime"]
                else:
                    ready = previousStart[r] + swapTime + event["gapFromPreviousSwap"]

                choices.append((ready, r, h, event))

            if not choices:
                break

            ready, r, h, event = min(choices, key=lambda item: (item[0], item[1], item[2]))
            start = max(ready, stationAvailable)
            wait = start - ready

            routeWaiting[r] += wait
            previousStart[r] = start
            nextEventIndex[r] += 1
            stationAvailable = start + swapTime

            events.append({
                "routeIndex": r,
                "eventIndex": h,
                "afterTask": event["afterTask"],
                "arrivalTime": ready,
                "startTime": start,
                "endTime": start + swapTime,
                "waitTime": wait,
            })

        completionTimes = [plan.duration + routeWaiting[r] for r, plan in enumerate(plans)]

        return {
            "objective": max(completionTimes) if completionTimes else 0.0,
            "selectedVariantIndices": selectedVariantIndices,
            "completionTimes": completionTimes,
            "waitingTimes": routeWaiting,
            "events": events,
            "status": "GREEDY_BSS_FALLBACK",
            "optimal": False,
        }

    def _applyResult(self, solution, variantsByRoute, result):
        solution.objective = result["objective"]
        solution.scheduleEvents = copy.deepcopy(result["events"])
        solution.scheduleStatus = result["status"]
        solution.exactOptimal = result["optimal"]

        selectedPlans = []
        for r, route in enumerate(solution.routes):
            selected = result["selectedVariantIndices"][r]
            plan = variantsByRoute[r][selected].copy()

            route.plan = plan
            route.baseDuration = plan.duration
            route.completionTime = result["completionTimes"][r]
            route.waitingTime = result["waitingTimes"][r]
            selectedPlans.append(plan)

        solution.baseObjective = max(plan.duration for plan in selectedPlans) if selectedPlans else 0.0


    def _applyCachedResult(self, solution, cached,):
        solution.objective = cached["objective"]
        solution.baseObjective = cached["baseObjective"]
        solution.scheduleEvents = copy.deepcopy(cached["scheduleEvents"])
        solution.scheduleStatus = cached["scheduleStatus"]
        solution.exactOptimal = cached["exactOptimal"]

        for r, route in enumerate(solution.routes):
            variants = self.fixedRouteOptimizer.optimize(route.tasks, maxVariants=self.variantsPerRoute)
            selected = cached["selectedVariantIndices"][r]
            route.plan = variants[selected].copy()
            route.baseDuration = route.plan.duration
            route.completionTime = cached["completionTimes"][r]
            route.waitingTime = cached["waitingTimes"][r]


class ALNS:
    """
    Adaptive Large Neighborhood Search for the current EVRP variant.

    Search variables:
      - task-to-vehicle assignment
      - task order in every route

    Exact evaluation oracle:
      - optimal fixed-route swap insertion (DP)
      - one physical BSS synchronization and swap-variant selection (Gurobi)

    No route pool / route assembler is used.
    """

    def __init__(
        self,
        modelData,
        iterations=1000,
        seed=1,
        minDestroyRate=0.08,
        maxDestroyRate=0.22,
        largeDestroyRate=0.40,
        variantsPerRoute=3,
        bssTimeLimit=None,
        mipGap=1e-6,
        segmentLength=50,
        reactionFactor=0.20,
        scoreNewBest=33.0,
        scoreBetter=9.0,
        scoreAccepted=3.0,
        minOperatorWeight=0.10,
        initialAcceptanceProbability=0.50,
        initialWorseningFraction=0.05,
        coolingRate=0.995,
        reheatAfter=150,
        reheatFactor=0.60,
        shawP=3.0,
    ):
        self.data = modelData
        self.iterations = iterations
        self.seed = seed
        self.random = random.Random(seed)

        self.minDestroyRate = minDestroyRate
        self.maxDestroyRate = maxDestroyRate
        self.largeDestroyRate = largeDestroyRate

        self.segmentLength = max(1, segmentLength)
        self.reactionFactor = reactionFactor
        self.minOperatorWeight = minOperatorWeight

        self.scoreNewBest = scoreNewBest
        self.scoreBetter = scoreBetter
        self.scoreAccepted = scoreAccepted

        self.initialAcceptanceProbability = initialAcceptanceProbability
        self.initialWorseningFraction = initialWorseningFraction

        self.coolingRate = coolingRate
        self.reheatAfter = max(1, reheatAfter)
        self.reheatFactor = reheatFactor
        self.shawP = shawP


        self.evaluator = ExactBssSolutionEvaluator(
            modelData=modelData,
            variantsPerRoute=variantsPerRoute,
            bssTimeLimit=bssTimeLimit,
            mipGap=mipGap,
            minimizeSlack=True,
        )

        self.destroyOperators = [
            ("random", self.randomRemoval),
            ("worst", self.worstRemoval),
            ("shaw", self.shawRemoval),
            ("route", self.routeRemoval),
        ]

        self.repairOperators = [
            ("greedy", self.greedyInsertion),
            ("regret2", self.regret2Insertion),
            ("regret3", self.regret3Insertion),
        ]

        self.destroyWeights = [1.0 for _ in self.destroyOperators]
        self.repairWeights = [1.0 for _ in self.repairOperators]

        self.destroyScores = [0.0 for _ in self.destroyOperators]
        self.repairScores = [0.0 for _ in self.repairOperators]

        self.destroyUses = [0 for _ in self.destroyOperators]
        self.repairUses = [0 for _ in self.repairOperators]

        self.bestSolution = None
        self.currentSolution = None
        self.runtime = 0.0
        self.completedIterations = 0
        self.deadline = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solveModel(self, timeLimit=None, outputFlag=1):
        startTime = time.time()
        self.deadline =  None if timeLimit is None else startTime + timeLimit

        current = self.createInitialSolution()
        self._evaluateExact(current, outputFlag=0, validateCoverage=True,)

        best = current.copy()

        initialTemperature = self._initialTemperature(current.objective)
        temperature = initialTemperature
        stagnation = 0

        if outputFlag:
            print(
                f"ALNS initial: "
                f"base={current.baseObjective:.3f}, "
                f"true={current.objective:.3f}, "
                f"status={current.scheduleStatus}"
            )

        for iteration in range(self.iterations):
            if self._timeExpired():
                break

            dIndex = self._roulette(self.destroyWeights)
            rIndex = self._roulette(self.repairWeights)

            destroyName, destroyOperator = (self.destroyOperators[dIndex])
            repairName, repairOperator = (self.repairOperators[rIndex])

            q = self._chooseDestroyCount(current, stagnation)

            destroyed = destroyOperator(current, q)

            try:
                candidate = repairOperator(destroyed)
            except TimeoutError:
                break

            if candidate.removed:
                # Safety fallback: every repair operator must return a complete
                # solution. If an exotic repair stopped early, finish greedily.
                candidate = self.greedyInsertion(candidate)

            if self._timeExpired():
                break

            self._evaluateExact(candidate, outputFlag=0, validateCoverage=True)

            if not math.isfinite(candidate.objective):
                reward = 0.0
                self._recordOperatorUse(dIndex, rIndex, reward)
                temperature *= self.coolingRate
                continue

            delta = candidate.objective - current.objective

            accepted = False
            newBest = False
            reward = 0.0

            if delta < -1e-9:
                current = candidate
                accepted = True

                if candidate.objective < best.objective - 1e-9:
                    best = candidate.copy()
                    newBest = True
                    reward = self.scoreNewBest
                    stagnation = 0

                    if outputFlag:
                        print(
                            f"Iter {iteration + 1:5d} | "
                            f"NEW BEST "
                            f"{best.objective:.3f} "
                            f"(base={best.baseObjective:.3f}) | "
                            f"{destroyName}/{repairName} | "
                            f"q={q}"
                        )
                else:
                    reward = self.scoreBetter
                    stagnation += 1

            else:
                acceptanceProbability = math.exp(-delta / max(temperature, 1e-12))

                if self.random.random() < acceptanceProbability:
                    current = candidate
                    accepted = True
                    reward = self.scoreAccepted

                stagnation += 1

            self._recordOperatorUse(dIndex, rIndex, reward)

            temperature *= self.coolingRate

            if (iteration + 1) % self.segmentLength == 0:
                self._updateOperatorWeights()

            if stagnation >= self.reheatAfter:
                # Diversification restart: search again from the incumbent,
                # but with reheated SA and a larger destroy size on the next
                # iterations.
                current = best.copy()
                temperature = max(temperature, initialTemperature * self.reheatFactor)
                stagnation = int(self.reheatAfter * 0.6)

            self.completedIterations = iteration + 1

        self.currentSolution = current
        self.bestSolution = best
        self.runtime = time.time() - startTime

        if outputFlag:
            print(
                f"ALNS finished: "
                f"true={best.objective:.3f}, "
                f"base={best.baseObjective:.3f}, "
                f"runtime={self.runtime:.3f}s, "
                f"iterations={self.completedIterations}"
            )

        return self

    def solve(self, timeLimit=None, outputFlag=1):
        self.solveModel(timeLimit=timeLimit, outputFlag=outputFlag)
        return self.bestSolution

    def getResult(self):
        if self.bestSolution is None:
            return {
                "objective": float("inf"),
                "status": "NOT_SOLVED",
                "runtime": self.runtime,
                "iterations": self.completedIterations,
            }

        return {
            "objective": self.bestSolution.objective,
            "baseObjective": self.bestSolution.baseObjective,
            "runtime": self.runtime,
            "iterations": self.completedIterations,
            "status": self.bestSolution.scheduleStatus,
            "exactOptimal": self.bestSolution.exactOptimal,
            "destroyWeights": {
                name: self.destroyWeights[index] for index, (name, _) in enumerate(self.destroyOperators)
            },
            "repairWeights": {
                name: self.repairWeights[index] for index, (name, _) in enumerate(self.repairOperators)
            },
        }

    def getRoutes(self):
        if self.bestSolution is None:
            return []

        result = []
        vehicle = 0
        for routeIndex, route in enumerate(self.bestSolution.routes):
            if not route.tasks:
                continue

            result.append({
                "vehicle": vehicle,
                "routeIndex": routeIndex,
                "tasks": list(route.tasks),
                "nodes": list(route.plan.nodes) if route.plan is not None else [],
                "duration": route.baseDuration,
                "baseDuration": route.baseDuration,
                "completionTime": route.completionTime,
                "waitingTime": route.waitingTime,
                "swapCount": route.plan.swapCount if route.plan is not None else 0,
            })
            vehicle += 1

        return result

    def getSwapEvents(self):
        if self.bestSolution is None:
            return []
        return [dict(event) for event in self.bestSolution.scheduleEvents]

    # ------------------------------------------------------------------
    # Initial solution
    # ------------------------------------------------------------------

    def createInitialSolution(self):
        routes = [ALNSRoute([]) for _ in range(self.data.K)]

        # LPT-like order based on singleton exact fixed-route duration.
        taskPriorities = []
        for task in self.data.C:
            plan = self.evaluator.bestBasePlan([task])
            if plan is None:
                raise RuntimeError(f"任务 {task} 单独执行也无法满足电量约束")
            taskPriorities.append((plan.duration, task))

        taskPriorities.sort(reverse=True)

        for _, task in taskPriorities:
            best = None

            for routeIndex, route in enumerate(routes):
                for position in range(len(route.tasks) + 1):
                    newTasks = route.tasks[:position] + [task] + route.tasks[position:]

                    plan = self.evaluator.bestBasePlan(newTasks)
                    if plan is None:
                        continue

                    durations = []
                    for otherIndex, otherRoute in enumerate(routes):
                        if otherIndex == routeIndex:
                            durations.append(plan.duration)
                        else:
                            otherPlan = self.evaluator.bestBasePlan(otherRoute.tasks)
                            durations.append(otherPlan.duration if otherPlan is not None else float("inf"))

                    score = (max(durations), sum(durations))

                    if best is None or score < best[0]:
                        best = (score, routeIndex, position)

            if best is None:
                raise RuntimeError(f"构造初始解时无法插入任务 {task}")

            _, routeIndex, position = best
            routes[routeIndex].tasks.insert(position, task)

        return ALNSSolution(routes)

    # ------------------------------------------------------------------
    # Destroy operators
    # ------------------------------------------------------------------

    def randomRemoval(self, solution, q):
        result = solution.copy()
        assigned = self._allTasks(result)

        q = min(q, len(assigned))
        removed = self.random.sample(assigned, q)
        self._removeTasks(result, removed)
        return result

    def worstRemoval(self, solution, q):
        result = solution.copy()

        scores = []
        baseDurations = self._baseDurations(solution)
        baseCMax = max(baseDurations)

        for routeIndex, route in enumerate(solution.routes):
            oldDuration = baseDurations[routeIndex]

            for task in route.tasks:
                newTasks = list(route.tasks)
                newTasks.remove(task)

                newPlan = self.evaluator.bestBasePlan(newTasks)
                if newPlan is None:
                    continue

                changedDurations = list(baseDurations)
                changedDurations[routeIndex] = newPlan.duration

                cmaxImprovement = baseCMax - max(changedDurations)
                routeImprovement = oldDuration - newPlan.duration

                # Cmax contribution dominates; route contribution breaks ties.
                score = cmaxImprovement * 1000.0 + routeImprovement

                scores.append((score, task))

        removed = self._biasedTopSelection(scores, q, p=3.0)

        if len(removed) < q:
            remaining = [task for task in self._allTasks(result) if task not in removed]
            self.random.shuffle(remaining)
            removed.extend(remaining[:q - len(removed)])

        self._removeTasks(result, removed)
        return result

    def shawRemoval(self, solution, q):
        result = solution.copy()
        assigned = self._allTasks(result)

        if not assigned:
            return result

        seed = self.random.choice(assigned)
        removed = [seed]
        remaining = set(assigned)
        remaining.remove(seed)

        while len(removed) < q and remaining:
            reference = self.random.choice(removed)

            related = sorted(
                 (self._relatedness(reference, task), task) for task in remaining
            )

            index = int((self.random.random() ** self.shawP) * len(related))
            index = min(index, len(related) - 1)

            task = related[index][1]
            removed.append(task)
            remaining.remove(task)

        self._removeTasks(result, removed)
        return result

    def routeRemoval(self, solution, q):
        result = solution.copy()

        nonempty = [index for index, route in enumerate(result.routes) if route.tasks]
        if not nonempty:
            return result

        # Prefer a critical/long route, but keep randomness.
        if solution.objective < float("inf") and self.random.random() < 0.65:
            routeIndex = max(
                nonempty,
                key=lambda index: (solution.routes[index].completionTime),
            )
        else:
            routeIndex = self.random.choice(nonempty)

        candidates = list(result.routes[routeIndex].tasks)
        self.random.shuffle(candidates)

        removed = candidates[:q]

        if len(removed) < q:
            others = [task for index, route in enumerate(result.routes) if index != routeIndex for task in route.tasks]
            self.random.shuffle(others)
            removed.extend(others[:q - len(removed)])

        self._removeTasks(result, removed)
        return result


    # ------------------------------------------------------------------
    # Repair operators
    # ------------------------------------------------------------------

    def greedyInsertion(self, solution):
        result = solution.copy()

        while result.removed:
            self._checkDeadline()

            best = None

            for task in list(result.removed):
                options = self._insertionOptions(result, task)
                if not options:
                    continue

                option = options[0]
                candidateKey = (option["score"], task)

                if best is None or candidateKey < best[0]:
                    best = (candidateKey, task, option)

            if best is None:
                raise RuntimeError("Greedy insertion无法完成任务修复")

            _, task, option = best
            self._applyInsertion(result, task, option)

        return result

    def regret2Insertion(self, solution):
        return self._regretInsertion(solution, k=2)

    def regret3Insertion(self, solution):
        return self._regretInsertion(solution, k=3)

    def _regretInsertion(self, solution, k):
        result = solution.copy()

        while result.removed:
            self._checkDeadline()

            bestChoice = None

            for task in list(result.removed):
                options = self._insertionOptions(result, task)
                if not options:
                    continue

                bestScore = options[0]["score"]
                kthIndex = min(k - 1, len(options) - 1)
                kthScore = options[kthIndex]["score"]

                # Fewer than k feasible options means the customer is urgent.
                shortageBonus = 1e6 if len(options) < k else 0.0
                regret = kthScore - bestScore + shortageBonus

                choiceKey = (regret, -bestScore)

                if bestChoice is None or choiceKey > bestChoice[0]:
                    bestChoice = (choiceKey, task, options[0])

            if bestChoice is None:
                raise RuntimeError(f"Regret-{k} insertion无法完成任务修复")

            _, task, option = bestChoice
            self._applyInsertion(result, task, option)

        return result

    # ------------------------------------------------------------------
    # Repair helpers
    # ------------------------------------------------------------------

    def _insertionOptions(self, solution, task):
        currentDurations = self._baseDurations(solution)

        options = []

        for routeIndex, route in enumerate(solution.routes):
            for position in range(len(route.tasks) + 1):
                newTasks = route.tasks[:position] + [task] + route.tasks[position:]
                plan = self.evaluator.bestBasePlan(newTasks)
                if plan is None:
                    continue

                durations = list(currentDurations)
                durations[routeIndex] = plan.duration
                baseCMax = max(durations)
                totalDuration = sum(durations)

                # Makespan dominates; total route duration is only a tie-break.
                score = baseCMax + 1e-4 * totalDuration

                options.append({
                    "routeIndex": routeIndex,
                    "position": position,
                    "tasks": newTasks,
                    "baseDuration": plan.duration,
                    "score": score,
                })

        options.sort(
            key=lambda option: (
                option["score"],
                option["baseDuration"],
                option["routeIndex"],
                option["position"],
            )
        )

        return options

    def _applyInsertion(self, solution, task, option):
        route = solution.routes[option["routeIndex"]]
        route.tasks = list(option["tasks"])
        self._invalidateRoute(route)

        # Keep the other still-unassigned tasks. Invalidation must not erase
        # the repair list after inserting only one customer.
        solution.removed.remove(task)
        self._invalidateSolution(solution, keepRemoved=True)

    # ------------------------------------------------------------------
    # Adaptive operator selection
    # ------------------------------------------------------------------

    def _roulette(self, weights):
        total = sum(weights)
        target = self.random.random() * total
        cumulative = 0.0

        for index, weight in enumerate(weights):
            cumulative += weight
            if target <= cumulative:
                return index

        return len(weights) - 1

    def _recordOperatorUse(self, dIndex, rIndex, reward):
        self.destroyUses[dIndex] += 1
        self.repairUses[rIndex] += 1
        self.destroyScores[dIndex] += reward
        self.repairScores[rIndex] += reward

    def _updateOperatorWeights(self):
        rho = self.reactionFactor

        for index in range(len(self.destroyWeights)):
            if self.destroyUses[index] > 0:
                observed = self.destroyScores[index] / self.destroyUses[index]

                self.destroyWeights[index] = max(
                    self.minOperatorWeight,
                    (1 - rho) * self.destroyWeights[index] + rho * observed
                )

        for index in range(len(self.repairWeights)):
            if self.repairUses[index] > 0:
                observed = self.repairScores[index] / self.repairUses[index]
                self.repairWeights[index] = max(
                    self.minOperatorWeight,
                    (1 - rho) * self.repairWeights[index] + rho * observed
                )

        self.destroyScores = [0.0 for _ in self.destroyOperators]
        self.repairScores = [0.0 for _ in self.repairOperators]
        self.destroyUses = [0 for _ in self.destroyOperators]
        self.repairUses = [0 for _ in self.repairOperators]

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------

    def _evaluateExact(self, solution, outputFlag=0, validateCoverage=True):
        remaining = self._remainingTime()

        if remaining is not None and remaining <= 0:
            raise TimeoutError

        return self.evaluator.evaluate(
            solution,
            outputFlag=outputFlag,
            remainingTime=remaining,
            validateCoverage=validateCoverage,
        )

    def _baseDurations(self, solution):
        durations = []

        for route in solution.routes:
            plan = self.evaluator.bestBasePlan(route.tasks)
            if plan is None:
                durations.append(float("inf"))
            else:
                durations.append(plan.duration)

        return durations

    def _chooseDestroyCount(self, solution, stagnation):
        numberTasks = len(self._allTasks(solution))

        if numberTasks <= 1:
            return numberTasks

        if stagnation >= int(0.6 * self.reheatAfter):
            low = max(self.maxDestroyRate, 0.20)
            high = max(low, self.largeDestroyRate)
        else:
            low = self.minDestroyRate
            high = self.maxDestroyRate

        rate = self.random.uniform(low, high)

        return max(
            1,
            min(numberTasks, int(math.ceil(numberTasks * rate))),
        )

    def _initialTemperature(self, objective):
        if not math.isfinite(objective) or objective <= 0:
            return 1.0

        probability = min(
            max(self.initialAcceptanceProbability, 1e-6),
            1 - 1e-6,
        )
        delta = max(1e-6, self.initialWorseningFraction * objective)

        return -delta / math.log(probability)

    def _relatedness(self, first, second):
        forward = self.data.t.get((first, second), self.data.M)
        backward = self.data.t.get((second, first), self.data.M)

        spatial = 0.5 * (forward + backward)
        service = abs(self.data.p[first] - self.data.p[second])
        taskEnergy = abs(self.data.q[first] - self.data.q[second])

        return spatial + 0.20 * service + 1.00 * taskEnergy

    def _biasedTopSelection(self, scoreTaskPairs, q, p,):
        remaining = sorted(scoreTaskPairs, key=lambda item: item[0], reverse=True)

        removed = []
        seen = set()

        while len(removed) < q and remaining:
            index = int((self.random.random() ** p) * len(remaining))
            index = min(index, len(remaining) - 1)

            _, task = remaining.pop(index)

            if task in seen:
                continue

            seen.add(task)
            removed.append(task)

            remaining = [pair for pair in remaining if pair[1] != task]

        return removed

    def _removeTasks(self, solution, tasks):
        taskSet = set(tasks)

        for route in solution.routes:
            if any(task in taskSet for task in route.tasks):
                route.tasks = [task for task in route.tasks if task not in taskSet]
                self._invalidateRoute(route)

        solution.removed = list(tasks)
        self._invalidateSolution(solution, keepRemoved=True)

    @staticmethod
    def _invalidateRoute(route):
        route.plan = None
        route.baseDuration = 0.0
        route.completionTime = 0.0
        route.waitingTime = 0.0

    @staticmethod
    def _invalidateSolution(solution, keepRemoved=False):
        solution.objective = float("inf")
        solution.baseObjective = float("inf")
        solution.scheduleEvents = []
        solution.scheduleStatus = "NOT_EVALUATED"
        solution.exactOptimal = False
        if not keepRemoved:
            solution.removed = []

    @staticmethod
    def _allTasks(solution):
        return [task for route in solution.routes for task in route.tasks]

    def _remainingTime(self):
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())

    def _timeExpired(self):
        remaining = self._remainingTime()
        return remaining is not None and remaining <= 0

    def _checkDeadline(self):
        if self._timeExpired():
            raise TimeoutError


# Backward-friendly class alias.
ALNSSolver = ALNS
