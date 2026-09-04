import pytest

pytest.importorskip("gurobipy")

from gurobipy import GRB

from bolaiScheduling.gurobiModel import GurobiScheduler
from bolaiScheduling.mockData import createMockData


def testBuildModel():
    data = createMockData(taskCount=5, stationCount=2, K=2, seed=3)
    scheduler = GurobiScheduler(data)
    model = scheduler.buildModel()
    model.update()

    assert model.ModelSense == GRB.MINIMIZE
    assert model.NumBinVars == len(data.A)
    assert scheduler.T[data.startNode].VarName == "T[0]"
    assert scheduler.E[data.startNode].VarName == "E[0]"


def testSolveSmallInstance():
    data = createMockData(
        taskCount=4,
        stationCount=1,
        K=2,
        seed=4,
        Q=500,
        QMin=10,
    )
    scheduler = GurobiScheduler(data)
    model = scheduler.solveModel(timeLimit=30, mipGap=0.001, outputFlag=0)

    assert model.SolCount > 0
    routes = scheduler.getRoutes()
    assert 1 <= len(routes) <= data.K

    visitedTasks = []
    for route in routes:
        assert route[0] == data.startNode
        assert route[-1] == data.endNode
        visitedTasks.extend(node for node in route if node in data.C)

    assert sorted(visitedTasks) == data.C
