from bolaiScheduling.mockData import createMockData


def testNodeIndexDefinition():
    data = createMockData(taskCount=10, stationCount=2, K=2, seed=1)

    assert data.startNode == 0
    assert data.C == list(range(1, 11))
    assert data.S == [11, 12]
    assert data.n == 12
    assert data.endNode == 13
    assert data.V == [0] + data.C + data.S + [13]


def testArcTypes():
    data = createMockData(taskCount=8, stationCount=2, K=2, seed=2)

    assert all((data.startNode, j) in data.A for j in data.C)
    assert all((i, data.endNode) in data.A for i in data.C)
    assert all((i, data.endNode) in data.A for i in data.S)

    for i in data.S:
        for j in data.S:
            assert (i, j) not in data.A
