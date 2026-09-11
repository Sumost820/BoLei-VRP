import pytest

from boleiScheduling.mockData import createMockData
from boleiScheduling.bpc2 import ThresholdMakespanBpcSolver


def testThresholdMakespanBpcEndToEndAgainstExactPathModel():
    pytest.importorskip("gurobipy")

    taskCount = 15
    data = createMockData(taskCount=taskCount, stationCopyCount=taskCount, K=3, seed=4, Q=50, QMin=20)

    # 二分容差 UB-LB
    tolerance = 1e-3
    solver = ThresholdMakespanBpcSolver(
        data,
        thresholdTolerance=tolerance,
        relativeTolerance=0,
        maxOuterIterations=50,             # 最大二分次数/执行BPC次数
        maxColumnsPerRound=100,            # 每轮子问题返回的最大列数
        useSavingsWarmStart=True,          # 节约算法热启动列池
        useHeuristicPricing=True,          # 启发式定价
        timeLimit=3600,                    # 最大运行时间
        useSrcCuts=True,                   # SRC
        useNgDssr=True,                    # NG-DSSR
    )

    result = solver.solve(lowerBound=0.0, upperBound=None, outputFlag=1)



