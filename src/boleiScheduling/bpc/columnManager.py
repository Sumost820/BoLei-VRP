class ColumnManager:
    """Global route-column registry shared by every BPC node.

    Column identity is the ordered customer sequence.  The manager guarantees
    that one sequence appears at most once in the global pool.  Branch nodes do
    not own duplicate route objects; their persistent RMPs merely decide which
    global columns are compatible with each vehicle slot.
    """

    def __init__(self, initialColumns=None):
        self.columns = []
        self.signatureToIndex = {}
        for column in initialColumns or ():
            self.add(column)

    def __len__(self):
        return len(self.columns)

    def __contains__(self, signature):
        return tuple(signature) in self.signatureToIndex

    def add(self, column):
        signature = tuple(column.signature)
        previous = self.signatureToIndex.get(signature)
        if previous is not None:
            return False, previous
        index = len(self.columns)
        self.columns.append(column)
        self.signatureToIndex[signature] = index
        return True, index

    def addCandidates(self, candidates):
        added = []
        for candidate in candidates:
            wasAdded, _ = self.add(candidate.column)
            if wasAdded:
                added.append(candidate.column)
        return added

    def addColumns(self, columns):
        added = []
        for column in columns:
            wasAdded, _ = self.add(column)
            if wasAdded:
                added.append(column)
        return added

    def signatures(self):
        return set(self.signatureToIndex)
