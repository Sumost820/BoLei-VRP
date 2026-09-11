import math
import random
import time
from dataclasses import dataclass


@dataclass
class SavingsWarmStartResult:
    columns: list
    solutions: list
    bestRoutes: list
    bestPenalizedScore: float
    buildTime: float
    startsAttempted: int
    feasibleStarts: int
    initialRouteCount: int


class SavingsWarmStart:
    """
    Multi-start directed Clarke-Wright-style warm start.

    The warm start manipulates customer-only sequences. Every retained sequence
    is passed to ``ExactCompleteRouteEvaluator.evaluate`` which returns the
    isolated-best feasible complete physical route for that customer order.

    This is ONLY a primal/warm-start heuristic. It does not restrict the route
    universe and therefore cannot affect Branch-and-Price-and-Cut exactness.

    A very large excess-route penalty is used only in the heuristic merge score:

        penalty * max(0, number_of_routes - K)
        + base_makespan
        + 1e-6 * total_base_duration.

    Thus, while more than K routes remain, any feasible merge is strongly
    preferred to keeping the excess route. The penalty never enters the master
    problem objective and has no role in lower bounds or optimality proofs.
    """

    def __init__(
        self,
        modelData,
        routeEvaluator,
        starts=12,
        seed=1,
        randomization=0.20,
        excessRoutePenalty=1.0e9,
        maxInitialColumns=1000,
    ):
        self.data = modelData
        self.routeEvaluator = routeEvaluator
        self.starts = max(1, int(starts))
        self.seed = int(seed)
        self.randomization = max(0.0, float(randomization))
        self.excessRoutePenalty = float(excessRoutePenalty)
        self.maxInitialColumns = (
            None if maxInitialColumns is None else max(1, int(maxInitialColumns))
        )

        self._singletonColumns = {}
        self._directedPairSavings = []

    def build(self):
        startWall = time.time()
        allColumns = {}
        solutions = []

        empty = self.routeEvaluator.evaluate(())
        if empty is not None:
            allColumns[empty.signature] = empty

        self._prepareSingletons(allColumns)
        self._prepareDirectedSavings()

        feasibleStarts = 0
        bestRoutes = []
        bestScore = float('inf')

        for startIndex in range(self.starts):
            rng = random.Random(self.seed + 1000003 * startIndex)
            routes, generated = self._constructOne(rng)

            for column in generated:
                allColumns[column.signature] = column

            routeColumns = []
            feasible = True
            for sequence in routes:
                column = self.routeEvaluator.evaluate(sequence)
                if column is None:
                    feasible = False
                    break
                routeColumns.append(column)
                allColumns[column.signature] = column

            if not feasible:
                continue

            score = self._solutionScore(routeColumns)
            solutions.append(tuple(routeColumns))

            if len(routeColumns) <= self.data.K:
                feasibleStarts += 1

            if score < bestScore:
                bestScore = score
                bestRoutes = list(routeColumns)

        # Protect all singleton complete-route columns, the empty route, and
        # every route of the best warm-start solution before applying a cap.
        protected = {column.signature for column in self._singletonColumns.values()}
        protected.update(column.signature for column in bestRoutes)
        emptyColumn = self.routeEvaluator.evaluate(())
        if emptyColumn is not None:
            protected.add(emptyColumn.signature)

        columns = list(allColumns.values())
        columns.sort(
            key=lambda column: (
                0 if column.signature in protected else 1,
                len(column.tasks),
                column.duration,
                column.nodes,
            )
        )

        if self.maxInitialColumns is not None and len(columns) > self.maxInitialColumns:
            protectedColumns = [
                column for column in columns
                if column.signature in protected
            ]
            optionalColumns = [
                column for column in columns
                if column.signature not in protected
            ]
            remaining = max(0, self.maxInitialColumns - len(protectedColumns))
            columns = protectedColumns + optionalColumns[:remaining]

        return SavingsWarmStartResult(
            columns=columns,
            solutions=solutions,
            bestRoutes=bestRoutes,
            bestPenalizedScore=bestScore,
            buildTime=time.time() - startWall,
            startsAttempted=self.starts,
            feasibleStarts=feasibleStarts,
            initialRouteCount=len(self.data.C),
        )

    def _prepareSingletons(self, allColumns):
        self._singletonColumns = {}
        for task in self.data.C:
            column = self.routeEvaluator.evaluate((task,))
            if column is None:
                continue
            self._singletonColumns[task] = column
            allColumns[column.signature] = column

    def _prepareDirectedSavings(self):
        pairs = []
        for i in self.data.C:
            colI = self._singletonColumns.get(i)
            if colI is None:
                continue
            for j in self.data.C:
                if i == j:
                    continue
                colJ = self._singletonColumns.get(j)
                if colJ is None:
                    continue

                merged = self.routeEvaluator.evaluate((i, j))
                if merged is None:
                    continue

                saving = colI.duration + colJ.duration - merged.duration
                pairs.append((float(saving), i, j))

        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        self._directedPairSavings = pairs

    def _constructOne(self, rng):
        routes = {task: (task,) for task in self.data.C}
        owner = {task: task for task in self.data.C}
        nextRouteId = max(self.data.C, default=0) + 1
        generatedColumns = []

        if not self._directedPairSavings:
            return list(routes.values()), generatedColumns

        savingScale = max(
            1.0,
            max(abs(item[0]) for item in self._directedPairSavings),
        )

        rankedPairs = []
        for saving, i, j in self._directedPairSavings:
            perturbation = rng.uniform(-1.0, 1.0) * self.randomization * savingScale
            rankedPairs.append((saving + perturbation, saving, i, j))
        rankedPairs.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

        for _, _, i, j in rankedPairs:
            if len(routes) <= self.data.K:
                break

            routeIdI = owner.get(i)
            routeIdJ = owner.get(j)
            if routeIdI is None or routeIdJ is None or routeIdI == routeIdJ:
                continue

            routeI = routes[routeIdI]
            routeJ = routes[routeIdJ]

            # Directed parallel-savings merge: i must be the final customer of
            # the first route and j the first customer of the second route.
            if routeI[-1] != i or routeJ[0] != j:
                continue

            mergedSequence = routeI + routeJ
            mergedColumn = self.routeEvaluator.evaluate(mergedSequence)
            if mergedColumn is None:
                continue

            oldColumns = [
                self.routeEvaluator.evaluate(routeI),
                self.routeEvaluator.evaluate(routeJ),
            ]
            currentRouteColumns = [
                self.routeEvaluator.evaluate(sequence)
                for sequence in routes.values()
            ]
            currentScore = self._solutionScore(currentRouteColumns)

            proposedColumns = [
                column
                for rid, sequence in routes.items()
                if rid not in (routeIdI, routeIdJ)
                for column in [self.routeEvaluator.evaluate(sequence)]
            ] + [mergedColumn]
            proposedScore = self._solutionScore(proposedColumns)

            # While route count exceeds K, the huge penalty makes any feasible
            # route-count-reducing merge attractive. Once K is reached the
            # construction stops, preserving parallelism for makespan.
            if proposedScore >= currentScore - 1e-12:
                continue

            del routes[routeIdI]
            del routes[routeIdJ]
            routes[nextRouteId] = mergedSequence
            for task in mergedSequence:
                owner[task] = nextRouteId
            nextRouteId += 1
            generatedColumns.append(mergedColumn)

        # A randomized directed savings pass can occasionally get stuck because
        # all remaining high-ranked endpoint pairs became internal. If m>K,
        # finish with an exact best-feasible endpoint merge over the remaining
        # routes. This is still only a warm-start heuristic.
        while len(routes) > self.data.K:
            bestMove = None
            routeItems = list(routes.items())
            currentColumns = [
                self.routeEvaluator.evaluate(sequence)
                for _, sequence in routeItems
            ]
            currentScore = self._solutionScore(currentColumns)

            for firstId, firstSequence in routeItems:
                for secondId, secondSequence in routeItems:
                    if firstId == secondId:
                        continue

                    # Four endpoint orientations. Reversals are allowed because
                    # the warm start is free to propose a different task order.
                    orientations = (
                        firstSequence + secondSequence,
                        firstSequence + tuple(reversed(secondSequence)),
                        tuple(reversed(firstSequence)) + secondSequence,
                        tuple(reversed(firstSequence)) + tuple(reversed(secondSequence)),
                    )

                    for mergedSequence in orientations:
                        mergedColumn = self.routeEvaluator.evaluate(mergedSequence)
                        if mergedColumn is None:
                            continue

                        proposedColumns = [
                            self.routeEvaluator.evaluate(sequence)
                            for rid, sequence in routeItems
                            if rid not in (firstId, secondId)
                        ] + [mergedColumn]
                        proposedScore = self._solutionScore(proposedColumns)

                        if proposedScore >= currentScore - 1e-12:
                            continue

                        candidate = (
                            proposedScore,
                            mergedColumn.duration,
                            mergedSequence,
                            firstId,
                            secondId,
                            mergedColumn,
                        )
                        if bestMove is None or candidate[:3] < bestMove[:3]:
                            bestMove = candidate

            if bestMove is None:
                break

            _, _, mergedSequence, firstId, secondId, mergedColumn = bestMove
            del routes[firstId]
            del routes[secondId]
            routes[nextRouteId] = mergedSequence
            for task in mergedSequence:
                owner[task] = nextRouteId
            nextRouteId += 1
            generatedColumns.append(mergedColumn)

        return list(routes.values()), generatedColumns

    def _solutionScore(self, routeColumns):
        if any(column is None for column in routeColumns):
            return float('inf')

        routeCount = len(routeColumns)
        excess = max(0, routeCount - self.data.K)
        if routeColumns:
            baseMakespan = max(column.duration for column in routeColumns)
            totalDuration = sum(column.duration for column in routeColumns)
        else:
            baseMakespan = 0.0
            totalDuration = 0.0

        return (
            self.excessRoutePenalty * excess
            + baseMakespan
            + 1e-6 * totalDuration
        )
