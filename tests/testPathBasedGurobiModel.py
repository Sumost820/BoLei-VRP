import pytest

pytest.importorskip("gurobipy")

from bolaiScheduling.pathBasedGurobiModel import PathBasedGurobiScheduler
from bolaiScheduling.mockData import createMockData


def getNodeName(node, data):
    if node == data.startNode:
        return "DepotStart"
    if node == data.endNode:
        return "DepotEnd"
    if node in data.C:
        return f"C{node}"
    if node in data.S:
        return "SwapStation"
    return str(node)


def printRoutes(routes, scheduler, data):
    print("\n" + "=" * 70)
    print(f"Objective T[n+1] = {scheduler.T[data.endNode].X:.3f}")
    print(f"Used vehicles = {len(routes)} / {data.K}")
    print("=" * 70)

    for routeIndex, route in enumerate(routes, start=1):
        routeNames = [getNodeName(node, data) for node in route]
        print(f"\nRoute {routeIndex}:")
        print(" -> ".join(routeNames))

    print("\nSwap events:")
    for event in scheduler.getSwapEvents():
        print(
            f"  path={event['path']}  "
            f"C{event['origin']} -> S -> {getNodeName(event['destination'], data)}  "
            f"start={event['startTime']:.3f}"
        )

    print("\n" + "=" * 70)


def testSolvePathBasedInstance():
    # mock数据
    data = createMockData(taskCount=12, stationCopyCount=2, K=3, seed=4, Q=50, QMin=20)
    # 求解模型
    scheduler = PathBasedGurobiScheduler(data)
    model = scheduler.solveModel(timeLimit=600, mipGap=0.001, outputFlag=1)

    assert model.SolCount > 0
    # 输出结果
    routes = scheduler.getRoutes()
    printRoutes(routes, scheduler, data)