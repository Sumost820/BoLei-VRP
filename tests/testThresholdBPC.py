import math

import pytest

from boleiScheduling.mockData import createMockData
from boleiScheduling.pathBasedGurobiModel import PathBasedGurobiScheduler
from boleiScheduling.bpc2 import ThresholdMakespanBpcSolver


def _served_tasks(columns):
    return [task for column in columns for task in column.tasks]


def testThresholdMakespanBpcEndToEndAgainstExactPathModel():
    """Solve one complete instance and compare with an independent exact MIP.

    This is intentionally an end-to-end test.  It exercises:
      outer makespan-threshold search
      fixed-threshold branch-and-price
      column generation / pricing
      aggregate physical-arc branching when needed
      integer route recovery
      exact BSS scheduling / separation
      final incumbent and optimality interval

    The path-based Gurobi model is used only as an independent oracle for this
    small deterministic instance.
    """
    pytest.importorskip("gurobipy")

    # One physical BSS, homogeneous vehicles.  Keep the instance small enough
    # for CI, but tight enough that battery / BSS logic is actually exercised.
    taskCount = 15
    data = createMockData(taskCount=taskCount, stationCopyCount=taskCount, seed=4, K=3, Q=50, QMin=20)

    # Independent exact reference model.
    reference = PathBasedGurobiScheduler(data)
    reference.solveModel(outputFlag=0, mipGap=0.0)
    assert reference.model.SolCount > 0
    reference_obj = float(reference.model.ObjVal)
    assert math.isfinite(reference_obj)

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
    )
    result = solver.solve(lowerBound=0.0, upperBound=None, outputFlag=1)

    # Outer threshold search must finish with a proven bracket.
    assert result.status == "OPTIMAL_WITHIN_TOLERANCE"
    assert result.provenWithinTolerance
    assert result.absoluteGap <= tolerance + 1e-8
    assert result.lowerBound <= reference_obj + tolerance
    assert result.upperBound >= reference_obj - tolerance
    assert result.objective == pytest.approx(reference_obj, abs=2.0 * tolerance)

    # Recover a genuine routing solution, not only a bound.
    assert result.routes
    assert len(result.routes) <= data.K
    served = _served_tasks(result.routes)
    assert sorted(served) == sorted(data.C)
    assert len(served) == len(set(served)) == len(data.C)

    # Every selected isolated route respects the final makespan upper bound.
    for column in result.routes:
        assert column.duration <= result.upperBound + tolerance
        assert column.nodes[0] == data.startNode
        assert column.nodes[-1] == data.endNode

    # The final incumbent has been passed through the exact shared-BSS
    # scheduler, and the returned objective is its true makespan.
    assert result.bssResult is not None
    assert result.bssResult.provenOptimal
    assert result.bssResult.objective <= result.upperBound + tolerance
    assert result.objective == pytest.approx(result.bssResult.objective, abs=1e-8)

    # At least one fixed-threshold BPC solve must have been executed.
    assert result.outerIterations >= 1
    assert result.thresholdHistory
    assert all(check.proven for check in result.thresholdHistory)


