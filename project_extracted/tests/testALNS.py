
import math
import pytest

from boleiScheduling.alnsSolver import ALNS
from boleiScheduling.mockData import createMockData


def assertEveryTaskExactlyOnce(data, routes):
    tasks = [
        task
        for route in routes
        for task in route["tasks"]
    ]
    assert sorted(tasks) == sorted(data.C)
    assert len(tasks) == len(set(tasks))


def testALNSExactGurobi():
    """
    End-to-end test of the complete ALNS architecture:
      destroy/repair + adaptive weights + SA
      + fixed-route swap DP
      + joint Gurobi BSS scheduling.
    """
    pytest.importorskip("gurobipy")

    taskCount = 50
    data = createMockData(taskCount=taskCount, stationCopyCount=taskCount, K=5, seed=5, Q=50, QMin=20)

    solver = ALNS(
        data,
        iterations=2000,
        seed=4,
        variantsPerRoute=5,
        bssTimeLimit=None,
        segmentLength=50,
    )

    solver.solveModel(timeLimit=600, outputFlag=1)

    result = solver.getResult()
    routes = solver.getRoutes()
    events = solver.getSwapEvents()

    assert math.isfinite(result["objective"])
    assert result["objective"] >= result["baseObjective"] - 1e-8
    assert len(routes) <= data.K
    assertEveryTaskExactlyOnce(data, routes)

    events = sorted(events, key=lambda event: event["startTime"])
    for first, second in zip(events, events[1:]):
        assert (second["startTime"] >= first["endTime"] - 1e-7)

    print("\n" + "=" * 60)
    print("ALNS")
    print("=" * 60)
    print(f"Objective : {result['objective']:.3f}")
    print(f"Base      : {result['baseObjective']:.3f}")
    print(f"Runtime   : {result['runtime']:.3f}s")
    print(f"Iterations: {result['iterations']}")
    print(f"Status    : {result['status']}")
    print("Destroy weights:", result["destroyWeights"])
    print("Repair weights :", result["repairWeights"])

    print("\nFinal routes:")
    for route in routes:
        print(
            f"Vehicle {route['vehicle'] + 1}: "
            f"{route['nodes']}, "
            f"base={route['baseDuration']:.3f}, "
            f"completion={route['completionTime']:.3f}, "
            f"wait={route['waitingTime']:.3f}, "
            f"swaps={route['swapCount']}"
        )

    print("\nSwap schedule:")
    for event in events:
        print(event)
