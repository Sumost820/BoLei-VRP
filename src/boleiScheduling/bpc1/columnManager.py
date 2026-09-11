class ColumnPrefixTrie:
    """Incremental trie for exact exclusion of columns already in the RMP.

    The trie is *not* a pricing resource and carries no dual information.  It
    only protects exact column generation when complete route signatures that
    already exist in the global column pool must be excluded.  A prefix that
    reaches ``DEAD`` can never become an existing column again.
    """

    DEAD = -1

    def __init__(self):
        self.children = [{}]

    def insert(self, signature):
        state = 0
        for node in tuple(signature):
            children = self.children[state]
            nextState = children.get(node)
            if nextState is None:
                nextState = len(self.children)
                children[node] = nextState
                self.children.append({})
            state = nextState
        return state

    def advance(self, state, node):
        if state == self.DEAD:
            return self.DEAD
        return self.children[state].get(node, self.DEAD)

    @property
    def stateCount(self):
        return len(self.children)


class ColumnManager:
    """Global complete-route registry with integer-bitset incidence indexes.

    Column identity is the complete physical node sequence.  Bit ``r`` in each
    index corresponds to ``columns[r]``.  The global pool is append-only, which
    keeps route indices stable across the whole branch-and-price-and-cut tree.

    Hot branch-node operations therefore become Python integer intersections
    rather than repeated scans over every RouteColumn.
    """

    def __init__(self, initialColumns=None):
        self.columns = []
        self.signatureToIndex = {}
        self.arcToRouteMask = {}
        self.precedenceToRouteMask = {}
        self.taskToRouteMask = {}
        self.prefixTrie = ColumnPrefixTrie()
        for column in initialColumns or ():
            self.add(column)

    def __len__(self):
        return len(self.columns)

    def __iter__(self):
        # Zero-copy signature view for pricing/external diagnostics.
        return iter(self.signatureToIndex)

    def __contains__(self, signature):
        return tuple(signature) in self.signatureToIndex

    @property
    def allRouteMask(self):
        return (1 << len(self.columns)) - 1

    @staticmethod
    def _addToIndex(index, key, routeBit):
        index[key] = index.get(key, 0) | routeBit

    def add(self, column):
        signature = tuple(column.signature)
        previous = self.signatureToIndex.get(signature)
        if previous is not None:
            return False, previous

        routeIndex = len(self.columns)
        routeBit = 1 << routeIndex
        self.columns.append(column)
        self.signatureToIndex[signature] = routeIndex
        self.prefixTrie.insert(signature)

        for task in column.taskSet:
            self._addToIndex(self.taskToRouteMask, task, routeBit)
        for arc in column.physicalArcSet:
            self._addToIndex(self.arcToRouteMask, arc, routeBit)
        for pair in column.precedenceSet:
            self._addToIndex(self.precedenceToRouteMask, pair, routeBit)
        return True, routeIndex

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
        # Compatibility API.  Hot code should use the manager/signatureToIndex
        # directly to avoid copying the full set every CG round.
        return set(self.signatureToIndex)

    def signatureView(self):
        return self.signatureToIndex.keys()

    def compatibleRouteMask(self, branchingState, slot):
        """Return the bitset of global columns compatible with one slot."""
        mask = self.allRouteMask
        if branchingState is None:
            return mask

        for arc in branchingState.requiredArcs(slot):
            mask &= self.arcToRouteMask.get(tuple(arc), 0)
            if not mask:
                return 0
        for arc in branchingState.forbiddenArcs(slot):
            mask &= ~self.arcToRouteMask.get(tuple(arc), 0)

        for pair in branchingState.requiredPrecedences(slot):
            mask &= self.precedenceToRouteMask.get(tuple(pair), 0)
            if not mask:
                return 0
        for pair in branchingState.forbiddenPrecedences(slot):
            mask &= ~self.precedenceToRouteMask.get(tuple(pair), 0)

        return mask & self.allRouteMask

    def indicesForSignatures(self, signatures):
        """Yield existing global indices for the supplied route signatures."""
        for signature in signatures:
            routeIndex = self.signatureToIndex.get(tuple(signature))
            if routeIndex is not None:
                yield routeIndex

    def routeMaskForSrc(self, cut):
        """Return routes with a positive divisor-2 SRC coefficient.

        For the current 3-row SRCs, coefficient>0 is equivalent to containing
        at least two cut customers.  Pairwise intersections of task-incidence
        bitsets avoid a full route-pool scan.
        """
        customers = tuple(cut.customers)
        if len(customers) < 2:
            return 0
        mask = 0
        taskMasks = [self.taskToRouteMask.get(task, 0) for task in customers]
        for a in range(len(taskMasks)):
            left = taskMasks[a]
            if not left:
                continue
            for b in range(a + 1, len(taskMasks)):
                mask |= left & taskMasks[b]
        return mask & self.allRouteMask

    @staticmethod
    def iterMaskIndices(mask):
        """Yield set-bit indices from a Python integer bitset."""
        mask = int(mask)
        while mask:
            least = mask & -mask
            yield least.bit_length() - 1
            mask ^= least

    def compatibleIndices(self, branchingState, slot):
        return self.iterMaskIndices(self.compatibleRouteMask(branchingState, slot))

    compatibleRouteIndices = compatibleIndices
