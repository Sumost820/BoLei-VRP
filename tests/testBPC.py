import pytest

from boleiScheduling.mockData import createMockData
from boleiScheduling.bpc1 import BranchPriceCutSolver


def testBranchPriceCutEndToEndSmallCase():
    pytest.importorskip('gurobipy')

    taskCount = 12
    data = createMockData(taskCount=taskCount, stationCopyCount=taskCount, K=3, seed=4, Q=50, QMin=20)

    solver = BranchPriceCutSolver(data, maxColumnsPerRound=100)

    solver.solve(outputFlag=1, nodeLimit=10000, timeLimit=3600)
    result = solver.getResult()

