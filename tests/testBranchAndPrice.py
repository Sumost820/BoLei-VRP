import pytest

pytest.importorskip("gurobipy")

from boleiScheduling.branchAndPrice import BranchAndPriceScheduler
from boleiScheduling.mockData import createMockData


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


def testBranchAndPrice():
    # B&P 使用一个可重复访问的物理换电站。
    # stationCopyCount 只影响原始 ModelData 的编号，pricing 只使用 data.S[0]。
    data = createMockData(taskCount=8, stationCopyCount=8, K=3, seed=4, Q=50, QMin=20)

    scheduler = BranchAndPriceScheduler(data)
    scheduler.solve(timeLimit=600, outputFlag=1)

    result = scheduler.getResult()
    routes = scheduler.getRoutes()

    assert len(routes) > 0
    assert result["objective"] < float("inf")

    print("\n" + "=" * 70)
    print("Branch-and-Price")
    print("=" * 70)
    print(f"Objective        : {result['objective']:.3f}")
    print(f"Root lower bound : {result['rootLowerBound']:.3f}")
    print(f"Runtime          : {result['runtime']:.3f} s")
    print(f"B&P nodes        : {result['nodes']}")
    print(f"Generated routes : {result['routes']}")

    for route in routes:
        names = [getNodeName(node, data) for node in route["nodes"]]
        print(f"\nVehicle {route['vehicle'] + 1}:")
        print(" -> ".join(names))
        print(f"Duration={route['duration']:.3f}, swaps={route['swapCount']}")

    print("\n当前 Branch-and-Price 精确求解单车能量可行的 routing problem。")
    print("不同车辆在唯一换电站上的排队冲突将在后续 BPC 的 cut 中处理。")
