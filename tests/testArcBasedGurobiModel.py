import pytest

pytest.importorskip("gurobipy")

from bolaiScheduling.arcBasedGurobiModel import ArcBasedGurobiScheduler
from bolaiScheduling.mockData import createMockData


def getNodeName(node, data):
    """把节点编号转换成更容易阅读的名称。"""
    if node == data.startNode:
        return "DepotStart"
    if node == data.endNode:
        return "DepotEnd"
    if node in data.C:
        return f"C{node}"
    if node in data.S:
        stationIndex = data.S.index(node) + 1
        return f"S{stationIndex}"
    return str(node)


def printRoutes(routes, scheduler, data):
    """打印每条车辆路径，以及任务/换电节点的时间和剩余电量。"""
    print("\n" + "=" * 70)
    print(f"Objective T[n+1] = {scheduler.T[data.endNode].X:.3f}")
    print(f"Used vehicles = {len(routes)} / {data.K}")
    print("=" * 70)

    for routeIndex, route in enumerate(routes, start=1):
        routeNames = [getNodeName(node, data) for node in route]

        print(f"\nRoute {routeIndex}:")
        print(" -> ".join(routeNames))

        for node in route:
            nodeName = getNodeName(node, data)
            if node == data.startNode:
                print(f"  {nodeName:<12}  T={scheduler.T[node].X:>10.3f}  E={scheduler.E[node].X:>10.3f}")
            elif node == data.endNode:
                print(f"  {nodeName:<12}  T={scheduler.T[node].X:>10.3f}")
            else:
                print(f"  {nodeName:<12}  T={scheduler.T[node].X:>10.3f}  E={scheduler.E[node].X:>10.3f}")

    print("\n" + "=" * 70)



def testSolveInstance():
    # mock数据
    data = createMockData(taskCount=12, stationCopyCount=2, K=3, seed=4, Q=50, QMin=20)
    # 求解模型
    scheduler = ArcBasedGurobiScheduler(data)
    model = scheduler.solveModel(timeLimit=600, mipGap=0.001, outputFlag=1)

    assert model.SolCount > 0
    # 输出结果
    routes = scheduler.getRoutes()
    printRoutes(routes, scheduler, data)
