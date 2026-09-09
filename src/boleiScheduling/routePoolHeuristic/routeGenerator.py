import random

from .fixedRouteSwapOptimizer import FixedRouteSwapOptimizer


class RouteGenerator:
    def __init__(self, modelData, variantsPerSequence=3, seed=1):
        self.modelData = modelData
        self.variantsPerSequence = variantsPerSequence
        self.random = random.Random(seed)
        self.swapOptimizer = FixedRouteSwapOptimizer(modelData, maxVariants=variantsPerSequence)
        self.focusTasks = set()

    def createInitialSolution(self):
        data = self.modelData
        taskInfo = []

        for task in data.C:
            variants = self.swapOptimizer.optimize([task], maxVariants=1)
            if len(variants) == 0:
                raise ValueError(f"任务 {task} 无法形成单任务可行路线")
            taskInfo.append((variants[0].duration, task))

        taskInfo.sort(reverse=True)
        routeCount = min(data.K, len(data.C))
        sequences = [[] for _ in range(routeCount)]

        for _, task in taskInfo:
            bestChoice = None

            for routeIndex in range(routeCount):
                sequence = sequences[routeIndex]
                for position in range(len(sequence) + 1):
                    candidateSequence = sequence[:position] + [task] + sequence[position:]
                    variants = self.swapOptimizer.optimize(candidateSequence, maxVariants=1)
                    if len(variants) == 0:
                        continue

                    candidateDurations = []
                    feasible = True
                    for index, currentSequence in enumerate(sequences):
                        if index == routeIndex:
                            candidateDurations.append(variants[0].duration)
                        elif len(currentSequence) > 0:
                            currentVariants = self.swapOptimizer.optimize(currentSequence, maxVariants=1)
                            if len(currentVariants) == 0:
                                feasible = False
                                break
                            candidateDurations.append(currentVariants[0].duration)

                    if not feasible:
                        continue

                    makespan = max(candidateDurations) if candidateDurations else 0.0
                    totalDuration = sum(candidateDurations)
                    score = (makespan, totalDuration, len(candidateSequence))

                    if bestChoice is None or score < bestChoice[0]:
                        bestChoice = (score, routeIndex, position)

            if bestChoice is None:
                raise ValueError(f"任务 {task} 无法插入任何当前路线")

            _, routeIndex, position = bestChoice
            sequences[routeIndex].insert(position, task)

        return [sequence for sequence in sequences if len(sequence) > 0]

    def generateRoutes(self, startSequences, iterations=100):
        current = [list(sequence) for sequence in startSequences]
        current = self.vnd(current)
        best = self.copySequences(current)
        bestScore = self.evaluateSolution(best)

        generatedRoutes = []
        generatedRoutes.extend(self.extractRoutes(current))

        for _ in range(iterations):
            candidate = self.perturb(best)
            candidate = self.vnd(candidate)
            candidateScore = self.evaluateSolution(candidate)

            generatedRoutes.extend(self.extractRoutes(candidate))

            if candidateScore < bestScore:
                best = self.copySequences(candidate)
                bestScore = candidateScore

        generatedRoutes.extend(self.extractRoutes(best))
        return {
            "bestSequences": best,
            "bestScore": bestScore,
            "routes": generatedRoutes,
        }

    def vnd(self, sequences):
        current = self.copySequences(sequences)
        neighborhoods = [
            self.findRelocateMove,
            self.findSwapMove,
            self.findTwoOptMove,
        ]

        neighborhoodIndex = 0
        while neighborhoodIndex < len(neighborhoods):
            candidate = neighborhoods[neighborhoodIndex](current)
            if candidate is not None and self.evaluateSolution(candidate) < self.evaluateSolution(current):
                current = candidate
                neighborhoodIndex = 0
            else:
                neighborhoodIndex += 1

        return current

    def findRelocateMove(self, sequences):
        currentScore = self.evaluateSolution(sequences)

        for fromRoute in range(len(sequences)):
            for fromPosition in range(len(sequences[fromRoute])):
                task = sequences[fromRoute][fromPosition]

                for toRoute in range(len(sequences)):
                    for toPosition in range(len(sequences[toRoute]) + 1):
                        if fromRoute == toRoute and (toPosition == fromPosition or toPosition == fromPosition + 1):
                            continue

                        candidate = self.copySequences(sequences)
                        candidate[fromRoute].pop(fromPosition)

                        adjustedPosition = toPosition
                        if fromRoute == toRoute and toPosition > fromPosition:
                            adjustedPosition -= 1

                        candidate[toRoute].insert(adjustedPosition, task)
                        candidate = [sequence for sequence in candidate if len(sequence) > 0]

                        if len(candidate) > self.modelData.K:
                            continue
                        if not self.isFeasibleSolution(candidate):
                            continue

                        if self.evaluateSolution(candidate) < currentScore:
                            return candidate

        return None

    def findSwapMove(self, sequences):
        currentScore = self.evaluateSolution(sequences)

        for firstRoute in range(len(sequences)):
            for firstPosition in range(len(sequences[firstRoute])):
                for secondRoute in range(firstRoute, len(sequences)):
                    startPosition = firstPosition + 1 if secondRoute == firstRoute else 0
                    for secondPosition in range(startPosition, len(sequences[secondRoute])):
                        candidate = self.copySequences(sequences)
                        candidate[firstRoute][firstPosition], candidate[secondRoute][secondPosition] = (
                            candidate[secondRoute][secondPosition],
                            candidate[firstRoute][firstPosition],
                        )

                        if not self.isFeasibleSolution(candidate):
                            continue
                        if self.evaluateSolution(candidate) < currentScore:
                            return candidate

        return None

    def findTwoOptMove(self, sequences):
        currentScore = self.evaluateSolution(sequences)

        for routeIndex, sequence in enumerate(sequences):
            if len(sequence) < 3:
                continue

            for i in range(len(sequence) - 1):
                for j in range(i + 1, len(sequence)):
                    candidate = self.copySequences(sequences)
                    candidate[routeIndex][i:j + 1] = reversed(candidate[routeIndex][i:j + 1])

                    if not self.isFeasibleSolution(candidate):
                        continue
                    if self.evaluateSolution(candidate) < currentScore:
                        return candidate

        return None

    def perturb(self, sequences):
        candidate = self.copySequences(sequences)
        allTasks = [task for sequence in candidate for task in sequence]
        if len(allTasks) <= 2:
            return candidate

        removeCount = max(2, min(len(allTasks) // 5, 4))
        focusCandidates = [task for task in allTasks if task in self.focusTasks]

        removedTasks = []
        if len(focusCandidates) > 0:
            focusRemoveCount = min(len(focusCandidates), max(1, removeCount // 2))
            removedTasks.extend(self.random.sample(focusCandidates, focusRemoveCount))

        remainingCandidates = [task for task in allTasks if task not in removedTasks]
        remainingCount = removeCount - len(removedTasks)
        if remainingCount > 0:
            removedTasks.extend(self.random.sample(remainingCandidates, remainingCount))

        for task in removedTasks:
            for sequence in candidate:
                if task in sequence:
                    sequence.remove(task)
                    break

        candidate = [sequence for sequence in candidate if len(sequence) > 0]
        if len(candidate) == 0:
            candidate = [[]]

        self.random.shuffle(removedTasks)
        for task in removedTasks:
            candidate = self.reinsertTask(candidate, task)

        return [sequence for sequence in candidate if len(sequence) > 0]

    def reinsertTask(self, sequences, task):
        choices = []

        for routeIndex in range(len(sequences)):
            for position in range(len(sequences[routeIndex]) + 1):
                candidate = self.copySequences(sequences)
                candidate[routeIndex].insert(position, task)

                if not self.isFeasiblePartialSolution(candidate):
                    continue

                score = self.evaluatePartialSolution(candidate)
                choices.append((score, candidate))

        if len(sequences) < self.modelData.K:
            candidate = self.copySequences(sequences) + [[task]]
            if self.isFeasiblePartialSolution(candidate):
                choices.append((self.evaluatePartialSolution(candidate), candidate))

        if len(choices) == 0:
            raise ValueError(f"任务 {task} 在扰动修复阶段无法重新插入")

        choices.sort(key=lambda item: item[0])
        candidateCount = min(3, len(choices))
        return self.copySequences(self.random.choice(choices[:candidateCount])[1])

    def evaluateSolution(self, sequences):
        durations = []

        for sequence in sequences:
            if len(sequence) == 0:
                continue
            variants = self.swapOptimizer.optimize(sequence, maxVariants=1)
            if len(variants) == 0:
                return (float("inf"), float("inf"))
            durations.append(variants[0].duration)

        if len(durations) == 0:
            return (float("inf"), float("inf"))

        return (max(durations), sum(durations))


    def evaluatePartialSolution(self, sequences):
        durations = []

        for sequence in sequences:
            if len(sequence) == 0:
                continue
            variants = self.swapOptimizer.optimize(sequence, maxVariants=1)
            if len(variants) == 0:
                return (float("inf"), float("inf"))
            durations.append(variants[0].duration)

        if len(durations) == 0:
            return (0.0, 0.0)

        return (max(durations), sum(durations))

    def isFeasiblePartialSolution(self, sequences):
        tasks = [task for sequence in sequences for task in sequence]
        if len(tasks) != len(set(tasks)):
            return False
        if not set(tasks).issubset(set(self.modelData.C)):
            return False
        if len(sequences) > self.modelData.K:
            return False

        for sequence in sequences:
            if len(sequence) == 0:
                continue
            if len(self.swapOptimizer.optimize(sequence, maxVariants=1)) == 0:
                return False

        return True

    def isFeasibleSolution(self, sequences):
        tasks = [task for sequence in sequences for task in sequence]
        if len(tasks) != len(set(tasks)):
            return False
        if set(tasks) != set(self.modelData.C):
            return False
        if len(sequences) > self.modelData.K:
            return False

        for sequence in sequences:
            if len(sequence) == 0:
                continue
            if len(self.swapOptimizer.optimize(sequence, maxVariants=1)) == 0:
                return False

        return True

    def extractRoutes(self, sequences):
        routes = []
        for sequence in sequences:
            routes.extend(self.swapOptimizer.optimize(sequence, maxVariants=self.variantsPerSequence))
        return routes


    def setFocusTasks(self, tasks):
        self.focusTasks = set(tasks)

    @staticmethod
    def copySequences(sequences):
        return [list(sequence) for sequence in sequences]
