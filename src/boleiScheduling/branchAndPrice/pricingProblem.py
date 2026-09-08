import time

from .route import Route


class PricingProblem:
    def __init__(self, modelData):
        self.modelData = modelData
        self.timedOut = False

    def findRoute(
        self,
        vehicle,
        duals,
        branchRows,
        routeId,
        tolerance=1e-6,
        deadline=None,
    ):
        data = self.modelData
        C = data.C
        Q = data.Q
        QMin = data.QMin
        p = data.p
        t = data.t
        q = data.q
        e = data.e
        startNode = data.startNode
        endNode = data.endNode

        if len(data.S) == 0:
            raise ValueError("Branch-and-Price 需要一个物理换电站")

        # B&P 中只保留一个物理换电站，并允许它在一条 route 中重复访问。
        stationNode = data.S[0]
        swapTime = p[stationNode]

        taskDual = duals["task"]
        vehicleDual = duals["vehicle"].get(vehicle, 0.0)
        makespanDual = duals["makespan"].get(vehicle, 0.0)

        forbiddenArcs = set()
        for row in branchRows:
            if row["type"] == "arc" and row["vehicle"] == vehicle and row["value"] == 0:
                forbiddenArcs.add(row["arc"])

        arcDual = {}
        for row in duals["branch"]:
            if row["type"] != "arc" or row["vehicle"] != vehicle:
                continue
            arc = row["arc"]
            arcDual[arc] = arcDual.get(arc, 0.0) + row["dual"]

        taskIndex = {task: index for index, task in enumerate(C)}
        labels = {}
        bestRoute = None
        bestReducedCost = float("inf")
        self.timedOut = False

        def isTimedOut():
            if deadline is None:
                return False
            if time.time() < deadline:
                return False
            self.timedOut = True
            return True

        def arcReducedCost(arc):
            return -arcDual.get(arc, 0.0)

        def dominated(mask, node, energy, duration, reducedCost):
            key = (mask, node)
            oldLabels = labels.get(key, [])

            # 使用完全相同的 visited set，规则较保守，但保持 exact。
            for oldEnergy, oldDuration, oldCost in oldLabels:
                if (
                    oldEnergy >= energy - tolerance
                    and oldDuration <= duration + tolerance
                    and oldCost <= reducedCost + tolerance
                ):
                    return True

            newLabels = []
            for oldEnergy, oldDuration, oldCost in oldLabels:
                if (
                    energy >= oldEnergy - tolerance
                    and duration <= oldDuration + tolerance
                    and reducedCost <= oldCost + tolerance
                ):
                    continue
                newLabels.append((oldEnergy, oldDuration, oldCost))

            newLabels.append((energy, duration, reducedCost))
            labels[key] = newLabels
            return False

        def saveRoute(nodes, tasks, duration, reducedCost):
            nonlocal bestRoute, bestReducedCost

            if len(tasks) == 0:
                return

            signature = tuple(nodes)

            if reducedCost >= bestReducedCost - tolerance:
                return

            bestReducedCost = reducedCost
            bestRoute = Route(
                routeId=routeId,
                nodes=nodes,
                tasks=tasks,
                duration=duration,
                stationNode=stationNode,
            )

        def closeRoute(node, energy, nodes, tasks, duration, reducedCost):
            arc = (node, endNode)
            if arc in forbiddenArcs:
                return
            if arc not in e:
                return
            if energy - e[arc] < QMin - tolerance:
                return

            deltaTime = t[arc]
            saveRoute(
                nodes=nodes + [endNode],
                tasks=tasks,
                duration=duration + deltaTime,
                reducedCost=reducedCost + makespanDual * deltaTime + arcReducedCost(arc),
            )

        def search(node, energy, mask, nodes, tasks, duration, reducedCost):
            if isTimedOut():
                return

            if dominated(mask, node, energy, duration, reducedCost):
                return

            # 当前已经至少访问过一个任务，可以随时返回终点。
            closeRoute(node, energy, nodes, tasks, duration, reducedCost)

            # 统一 extension：扩展到尚未访问的任务节点。
            for nextTask in C:
                bit = 1 << taskIndex[nextTask]
                if mask & bit:
                    continue

                arc = (node, nextTask)
                if arc in forbiddenArcs or arc not in e:
                    continue

                newEnergy = energy - e[arc] - q[nextTask]
                if newEnergy < QMin - tolerance:
                    continue

                deltaTime = t[arc] + p[nextTask]
                search(
                    node=nextTask,
                    energy=newEnergy,
                    mask=mask | bit,
                    nodes=nodes + [nextTask],
                    tasks=tasks + [nextTask],
                    duration=duration + deltaTime,
                    reducedCost=(
                        reducedCost
                        + makespanDual * deltaTime
                        - taskDual[nextTask]
                        + arcReducedCost(arc)
                    ),
                )

            # 换电站是可重复访问节点。只有任务节点之后才能进入换电站。
            if node not in C:
                return

            arc = (node, stationNode)
            if arc in forbiddenArcs or arc not in e:
                return
            if energy - e[arc] < QMin - tolerance:
                return

            deltaTime = t[arc] + swapTime
            search(
                node=stationNode,
                energy=Q,
                mask=mask,
                nodes=nodes + [stationNode],
                tasks=tasks,
                duration=duration + deltaTime,
                reducedCost=(
                    reducedCost
                    + makespanDual * deltaTime
                    + arcReducedCost(arc)
                ),
            )

        # 初始节点 0 不允许直接访问换电站，只扩展到任务。
        initialReducedCost = -vehicleDual

        for firstTask in C:
            if isTimedOut():
                break

            arc = (startNode, firstTask)
            if arc in forbiddenArcs or arc not in e:
                continue

            energy = Q - e[arc] - q[firstTask]
            if energy < QMin - tolerance:
                continue

            deltaTime = t[arc] + p[firstTask]
            bit = 1 << taskIndex[firstTask]

            search(
                node=firstTask,
                energy=energy,
                mask=bit,
                nodes=[startNode, firstTask],
                tasks=[firstTask],
                duration=deltaTime,
                reducedCost=(
                    initialReducedCost
                    + makespanDual * deltaTime
                    - taskDual[firstTask]
                    + arcReducedCost(arc)
                ),
            )

        return bestRoute, bestReducedCost, self.timedOut
