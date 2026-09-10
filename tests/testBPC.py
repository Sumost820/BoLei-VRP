import math

import pytest

from boleiScheduling.mockData import createMockData
from boleiScheduling.bpc import BranchPriceCutSolver



def _candidate_map(candidates):
    return {
        (candidate.column.tasks, candidate.bestSlot): candidate.reducedCost
        for candidate in candidates
    }


def testBranchPriceCutEndToEndSmallCase():
    pytest.importorskip('gurobipy')
    taskCount = 12
    data = createMockData(taskCount=taskCount, stationCopyCount=taskCount, K=2, seed=4, Q=50, QMin=20)

    solver = BranchPriceCutSolver(
        data,
        pricingMode='labeling',
        maxColumnsPerRound=100,
        maxExactRouteTasks=None,
        bssTimeLimit=None,
        useSRC=False,
    )
    solver.solve(outputFlag=1, nodeLimit=10000, timeLimit=None)
    result = solver.getResult()

    assert result['status'] == 'OPTIMAL'
    assert math.isfinite(result['objective'])
    assert result['bestBound'] == pytest.approx(result['objective'], abs=1e-6)
    assert result['relativeGap'] <= 1e-8

    tasks = [task for route in result['selectedRouteTasks'] for task in route]
    assert sorted(tasks) == sorted(data.C)
    assert len(tasks) == len(set(tasks))