import math
import random

from .modelData import ModelData


def createMockData(taskCount, seed=1, stationCount=None, K=None, Q=100, QMin=20):
    random.seed(seed)

    if stationCount is None:
        stationCount = max(1, math.ceil(taskCount / 10))
    if K is None:
        K = max(2, math.ceil(taskCount / 20))

    C = list(range(1, taskCount + 1))
    S = list(range(taskCount + 1, taskCount + stationCount + 1))
    R = len(S)
    n = taskCount + R
    startNode = 0
    endNode = n + 1
    V = [startNode] + C + S + [endNode]

    depotPoint = (0.0, 0.0)
    stationPoint = (50.0, 50.0)

    loadPoints = {}
    unloadPoints = {}
    for i in C:
        loadPoints[i] = (random.uniform(0, 100), random.uniform(0, 100))
        unloadPoints[i] = (random.uniform(0, 100), random.uniform(0, 100))

    p = {startNode: 0.0}
    q = {startNode: 0.0}
    for i in C:
        taskDistance = distance(loadPoints[i], unloadPoints[i])
        p[i] = round(5.0 + taskDistance / 3.0, 3)
        q[i] = round(4.0 + taskDistance * 0.08, 3)

    swapTime = 8.0
    for i in S:
        p[i] = swapTime
        q[i] = 0.0

    A = []
    t = {}
    e = {}

    for j in C:
        addArc(A, t, e, startNode, j, depotPoint, loadPoints[j])

    for i in C:
        for j in C:
            if i != j:
                addArc(A, t, e, i, j, unloadPoints[i], loadPoints[j])

    for i in C:
        for j in S:
            addArc(A, t, e, i, j, unloadPoints[i], stationPoint)

    for i in S:
        for j in C:
            addArc(A, t, e, i, j, stationPoint, loadPoints[j])

    for i in C:
        addArc(A, t, e, i, endNode, unloadPoints[i], depotPoint)

    for i in S:
        addArc(A, t, e, i, endNode, stationPoint, depotPoint)

    maxTravelTime = max(t.values()) if t else 0
    maxTravelEnergy = max(e.values()) if e else 0
    maxServiceTime = max(p.values()) if p else 0
    maxTaskEnergy = max(q.values()) if q else 0

    timeUpper = (
        sum(p[i] for i in C)
        + R * swapTime
        + (taskCount + R + K) * maxTravelTime
    )
    energyUpper = Q + QMin + maxTravelEnergy + maxTaskEnergy
    M = round(max(timeUpper + maxServiceTime, energyUpper), 3)

    modelData = ModelData(
        C=C,
        S=S,
        V=V,
        A=A,
        p=p,
        t=t,
        q=q,
        e=e,
        Q=Q,
        QMin=QMin,
        K=K,
        M=M,
        R=R,
        n=n,
        startNode=startNode,
        endNode=endNode,
    )
    modelData.validate()
    return modelData


def saveMockData(taskCount, filePath, seed=1, stationCount=None, K=None, Q=100, QMin=20):
    modelData = createMockData(
        taskCount=taskCount,
        seed=seed,
        stationCount=stationCount,
        K=K,
        Q=Q,
        QMin=QMin,
    )
    modelData.toJson(filePath)
    return modelData


def addArc(A, t, e, i, j, fromPoint, toPoint):
    travelDistance = distance(fromPoint, toPoint)
    A.append((i, j))
    t[i, j] = round(2.0 + travelDistance / 4.0, 3)
    e[i, j] = round(1.0 + travelDistance * 0.04, 3)


def distance(pointA, pointB):
    deltaX = pointA[0] - pointB[0]
    deltaY = pointA[1] - pointB[1]
    return math.sqrt(deltaX * deltaX + deltaY * deltaY)
