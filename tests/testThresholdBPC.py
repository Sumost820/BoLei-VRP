import pytest

from boleiScheduling.mockData import createMockData
from boleiScheduling.bpc2 import ThresholdMakespanBpcSolver


def _served_tasks(columns):
    return [task for column in columns for task in column.tasks]


def testThresholdMakespanBpcEndToEndAgainstExactPathModel():
    pytest.importorskip("gurobipy")

    # One physical BSS, homogeneous vehicles.  Keep the instance small enough
    # for CI, but tight enough that battery / BSS logic is actually exercised.
    taskCount = 15
    data = createMockData(taskCount=taskCount, stationCopyCount=taskCount, K=3, seed=4, Q=50, QMin=20)

    # Complete vehicle-free threshold BPC solve -- not a pricing unit test.
    tolerance = 1e-3
    solver = ThresholdMakespanBpcSolver(
        data,
        thresholdTolerance=tolerance,
        relativeTolerance=0,
        maxOuterIterations=80,
        maxColumnsPerRound=100,
        useSavingsWarmStart=True,
        timeLimit=3600,
        maxNodesPerThreshold=10000,
        bssTimeLimit=None,
        useSrcCuts=True,
        useHeuristicPricing=True,
    )
    result = solver.solve(lowerBound=0.0, upperBound=None, outputFlag=1)



