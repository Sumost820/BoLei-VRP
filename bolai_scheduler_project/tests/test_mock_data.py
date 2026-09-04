from pathlib import Path

import pytest

from bolaiScheduling.modelData import loadData


@pytest.mark.parametrize("taskCount", [25, 50, 75, 100])
def testMockDatasetSize(taskCount):
    projectRoot = Path(__file__).resolve().parents[1]
    filePath = projectRoot / "data" / f"mock{taskCount}.json"

    data = loadData(filePath)

    assert len(data.C) == taskCount
    assert data.C[0] == 1
    assert data.C[-1] == taskCount
    assert data.startNode == 0
    assert data.endNode == data.n + 1
    assert data.QMin <= data.Q
    assert data.K > 0
    assert data.M > 0
