import pytest

pytest.importorskip("gurobipy")

from boleiScheduling.mockData import createMockData
from boleiScheduling.GeneratorAndAssemblerHeuristic.heuristicScheduler import RoutePoolHeuristicScheduler


def testRoutePoolHeuristicPrintsFinalRoutesAndSwapSchedule():
    taskCount = 50
    data = createMockData(taskCount=taskCount, stationCopyCount=taskCount, K=5, seed=5, Q=50, QMin=20)
    # print(data)


    scheduler = RoutePoolHeuristicScheduler(
        data,
        outerIterations=10,
        ilsIterations=30,
        variantsPerSequence=5,
        maxPoolSize=1000,
        assemblerIterations=30,
        seed=4,
    )

    scheduler.solveModel(timeLimit=300, outputFlag=1)

    result = scheduler.getResult()
    routes = scheduler.getRoutes()
    swapEvents = scheduler.getSwapEvents()

    assert len(routes) > 0
    assert result["objective"] < float("inf")

    print("\n" + "=" * 60)
    print("Route Pool Heuristic")
    print("=" * 60)
    print(f"Objective      : {result['objective']:.3f}")
    print(f"Runtime        : {result['runtime']:.3f} s")
    print(f"Iterations     : {result['iterations']}")
    print(f"Route pool size: {result['routePoolSize']}")

    print("\nFinal routes:")
    for route in routes:
        print(
            f"Vehicle {route['vehicle'] + 1}: "
            f"{route['nodes']}, "
            f"base={route['duration']:.3f}, "
            f"completion={route['completionTime']:.3f}, "
            f"swaps={route['swapCount']}"
        )

    print("\nSwap schedule:")
    for event in swapEvents:
        print(event)
