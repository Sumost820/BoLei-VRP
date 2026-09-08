import pytest

pytest.importorskip("gurobipy")

from gurobipy import GRB
from bolaiScheduling.arcBasedGurobiModel import ArcBasedGurobiScheduler
from bolaiScheduling.pathBasedGurobiModel import PathBasedGurobiScheduler
from bolaiScheduling.mockData import createMockData


def getModelResult(name, model):
    return {
        "name": name,
        "objective": model.ObjVal,
        "runtime": model.Runtime,
        "variables": model.NumVars,
        "binaryVariables": model.NumBinVars,
        "constraints": model.NumConstrs,
        "nodes": model.NodeCount,
        "gap": model.MIPGap,
        "status": model.Status,
    }


def printResult(result):
    print(f"\n{result['name']}")
    print("-" * 50)
    print(f"Objective       : {result['objective']:.3f}")
    print(f"Runtime         : {result['runtime']:.3f} s")
    print(f"Variables       : {result['variables']}")
    print(f"Binary variables: {result['binaryVariables']}")
    print(f"Constraints     : {result['constraints']}")
    print(f"B&B nodes       : {result['nodes']:.0f}")
    print(f"MIP gap         : {result['gap']:.6f}")
    print(f"Status          : {result['status']}")


def testCompareArcAndPathBased():
    taskCount = 12

    # 两个模型共用同一个算例
    # Arc-based 需要足够多的换电站 copy，因此 stationCopyCount = taskCount
    data = createMockData(taskCount=taskCount, stationCopyCount=taskCount, K=3, seed=4, Q=50, QMin=20)

    arcScheduler = ArcBasedGurobiScheduler(data)
    arcModel = arcScheduler.solveModel(timeLimit=600, mipGap=0.001, outputFlag=1)

    pathScheduler = PathBasedGurobiScheduler(data)
    pathModel = pathScheduler.solveModel(timeLimit=600, mipGap=0.001, outputFlag=1)

    assert arcModel.SolCount > 0
    assert pathModel.SolCount > 0

    arcResult = getModelResult("Arc-based", arcModel)
    pathResult = getModelResult("Path-based", pathModel)

    print("\n" + "=" * 60)
    print("Arc-based vs Path-based")
    print("=" * 60)
    printResult(arcResult)
    printResult(pathResult)

    print("\nComparison")
    print("-" * 50)
    print(f"Objective difference : {pathModel.ObjVal - arcModel.ObjVal:.6f}")
    print(f"Runtime ratio        : {pathModel.Runtime / max(arcModel.Runtime, 1e-9):.3f}")
    print(f"Variable ratio       : {pathModel.NumVars / max(arcModel.NumVars, 1):.3f}")
    print(f"Constraint ratio     : {pathModel.NumConstrs / max(arcModel.NumConstrs, 1):.3f}")

    # 两个模型都证明最优时，目标值应该一致。
    if arcModel.Status == GRB.OPTIMAL and pathModel.Status == GRB.OPTIMAL:
        assert abs(arcModel.ObjVal - pathModel.ObjVal) <= 1e-4

    print("\nArc routes:")
    for route in arcScheduler.getRoutes():
        print(route)

    print("\nPath routes:")
    for route in pathScheduler.getRoutes():
        print(route)

    print("\nPath swap events:")
    for event in pathScheduler.getSwapEvents():
        print(event)