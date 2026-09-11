from dataclasses import dataclass


@dataclass(frozen=True)
class ArcBranchDecision:
    """Branch on one slot-specific complete-route physical-arc flow."""

    slot: int
    arc: tuple
    flow: float


@dataclass(frozen=True)
class PrecedenceBranchDecision:
    """Fallback branch on customer order inside one vehicle slot.

    For two customers ``i`` and ``j`` define

        g[k,i,j] = sum_r 1{i,j in r and i appears before j} x[k,r].

    The two children are ``g=0`` and ``g=1``.  This branch is used only when
    the master is fractional but every physical-arc flow is integral.  Such a
    degeneracy can happen because the same physical BSS is repeatable: two
    distinct complete routes may have the same physical-arc incidence vector
    while visiting customer loops around the BSS in a different order.

    Unlike the old ``(i,S,j)`` transition fallback, customer precedence is
    determined entirely by the elementary visited set.  Pricing therefore does
    not need any station-predecessor history state.
    """

    slot: int
    pair: tuple
    flow: float


@dataclass(frozen=True)
class ArcBranchingState:
    """Slot-specific branch restrictions used by exact pricing.

    Normal branching uses complete-route physical arcs

        f[k,u,v] = sum_r 1{(u,v) in r} x[k,r].

    ``f=0`` forbids an arc and ``f=1`` requires it.  The repeatable BSS may
    have several required incoming/outgoing arcs in one route, while customer
    nodes remain elementary.

    A customer-precedence branch is only an exactness fallback for the rare
    case where physical-arc incidence cannot distinguish two fractional route
    columns.  No branch state stores how a label entered the BSS.
    """

    startNode: int
    endNode: int
    stationNode: int
    requiredArcsBySlot: tuple
    forbiddenArcsBySlot: tuple
    requiredPrecedencesBySlot: tuple
    forbiddenPrecedencesBySlot: tuple

    @staticmethod
    def root(modelData):
        K = int(modelData.K)
        empty = tuple(frozenset() for _ in range(K))
        return ArcBranchingState(
            startNode=int(modelData.startNode),
            endNode=int(modelData.endNode),
            stationNode=int(modelData.S[0]),
            requiredArcsBySlot=empty,
            forbiddenArcsBySlot=empty,
            requiredPrecedencesBySlot=empty,
            forbiddenPrecedencesBySlot=empty,
        )

    @property
    def numberOfSlots(self):
        return len(self.requiredArcsBySlot)

    def requiredArcs(self, slot):
        return self.requiredArcsBySlot[slot]

    def forbiddenArcs(self, slot):
        return self.forbiddenArcsBySlot[slot]

    def requiredPrecedences(self, slot):
        return self.requiredPrecedencesBySlot[slot]

    def forbiddenPrecedences(self, slot):
        return self.forbiddenPrecedencesBySlot[slot]

    @staticmethod
    def physicalArcSet(nodes):
        nodes = tuple(nodes)
        return frozenset(zip(nodes[:-1], nodes[1:]))

    def customerSequence(self, nodes):
        special = {self.startNode, self.endNode, self.stationNode}
        return tuple(node for node in nodes if node not in special)

    @staticmethod
    def precedenceSetFromTasks(tasks):
        tasks = tuple(tasks)
        return frozenset(
            (tasks[a], tasks[b])
            for a in range(len(tasks))
            for b in range(a + 1, len(tasks))
        )

    def precedenceSet(self, nodes):
        return self.precedenceSetFromTasks(self.customerSequence(nodes))

    def isNodeSequenceCompatible(self, slot, nodes):
        arcs = self.physicalArcSet(nodes)
        if not self.requiredArcsBySlot[slot].issubset(arcs):
            return False
        if self.forbiddenArcsBySlot[slot] & arcs:
            return False

        precedences = self.precedenceSet(nodes)
        if not self.requiredPrecedencesBySlot[slot].issubset(precedences):
            return False
        if self.forbiddenPrecedencesBySlot[slot] & precedences:
            return False
        return True

    def isColumnCompatible(self, slot, column):
        arcs = column.physicalArcSet
        if not self.requiredArcsBySlot[slot].issubset(arcs):
            return False
        if self.forbiddenArcsBySlot[slot] & arcs:
            return False

        precedences = column.precedenceSet
        if not self.requiredPrecedencesBySlot[slot].issubset(precedences):
            return False
        if self.forbiddenPrecedencesBySlot[slot] & precedences:
            return False
        return True

    def childArc(self, slot, arc, value):
        """Create the ``f[k,u,v]=value`` child; None means structurally impossible."""
        if value not in (0, 1):
            raise ValueError('arc branch value must be 0 or 1')
        if slot < 0 or slot >= self.numberOfSlots:
            raise ValueError('invalid vehicle slot')

        arc = tuple(arc)
        required = [set(items) for items in self.requiredArcsBySlot]
        forbidden = [set(items) for items in self.forbiddenArcsBySlot]
        if value == 1:
            required[slot].add(arc)
        else:
            forbidden[slot].add(arc)

        state = ArcBranchingState(
            startNode=self.startNode,
            endNode=self.endNode,
            stationNode=self.stationNode,
            requiredArcsBySlot=tuple(frozenset(items) for items in required),
            forbiddenArcsBySlot=tuple(frozenset(items) for items in forbidden),
            requiredPrecedencesBySlot=self.requiredPrecedencesBySlot,
            forbiddenPrecedencesBySlot=self.forbiddenPrecedencesBySlot,
        )
        return state if state.isStructurallyConsistent() else None

    def childPrecedence(self, slot, pair, value):
        """Create a customer-precedence branch child.

        ``value=1`` requires both customers to be in the selected slot route
        with ``i`` before ``j``.  ``value=0`` forbids that event (the route may
        omit one customer or may contain ``j`` before ``i``).
        """
        if value not in (0, 1):
            raise ValueError('precedence branch value must be 0 or 1')
        if slot < 0 or slot >= self.numberOfSlots:
            raise ValueError('invalid vehicle slot')

        pair = tuple(pair)
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError('precedence pair must contain two distinct customers')
        if any(node in (self.startNode, self.endNode, self.stationNode) for node in pair):
            raise ValueError('precedence pair must contain customer nodes only')

        required = [set(items) for items in self.requiredPrecedencesBySlot]
        forbidden = [set(items) for items in self.forbiddenPrecedencesBySlot]
        if value == 1:
            required[slot].add(pair)
        else:
            forbidden[slot].add(pair)

        state = ArcBranchingState(
            startNode=self.startNode,
            endNode=self.endNode,
            stationNode=self.stationNode,
            requiredArcsBySlot=self.requiredArcsBySlot,
            forbiddenArcsBySlot=self.forbiddenArcsBySlot,
            requiredPrecedencesBySlot=tuple(frozenset(items) for items in required),
            forbiddenPrecedencesBySlot=tuple(frozenset(items) for items in forbidden),
        )
        return state if state.isStructurallyConsistent() else None

    def isStructurallyConsistent(self):
        """Reject branch combinations that cannot be embedded in one route."""
        start = self.startNode
        end = self.endNode
        S = self.stationNode

        for k in range(self.numberOfSlots):
            required = set(self.requiredArcsBySlot[k])
            forbidden = set(self.forbiddenArcsBySlot[k])
            requiredPrec = set(self.requiredPrecedencesBySlot[k])
            forbiddenPrec = set(self.forbiddenPrecedencesBySlot[k])

            if required & forbidden:
                return False
            if requiredPrec & forbiddenPrec:
                return False

            outRequired = {}
            inRequired = {}
            for u, v in required:
                if u == v or u == end or v == start:
                    return False
                if (u, v) in ((start, end), (start, S), (S, S)):
                    return False

                # S is repeatable; every other route node has at most one
                # required successor/predecessor in an elementary route.
                if u != S:
                    old = outRequired.get(u)
                    if old is not None and old != v:
                        return False
                    outRequired[u] = v
                if v != S:
                    old = inRequired.get(v)
                    if old is not None and old != u:
                        return False
                    inRequired[v] = u

            # A directed required physical cycle that avoids S cannot occur in
            # an elementary start-to-end route.
            for first in tuple(outRequired):
                if first in (start, end, S):
                    continue
                seen = set()
                node = first
                while node in outRequired and node not in (start, end, S):
                    if node in seen:
                        return False
                    seen.add(node)
                    node = outRequired[node]

            # Required precedence relations must form a DAG.  Forbidden
            # precedence relations do not imply the reverse order because a
            # slot route is allowed to omit either customer.
            successors = {}
            indegree = {}
            for i, j in requiredPrec:
                if i == j or i in (start, end, S) or j in (start, end, S):
                    return False
                successors.setdefault(i, set()).add(j)
                indegree.setdefault(i, 0)
                indegree[j] = indegree.get(j, 0) + 1

            if indegree:
                queue = [node for node, degree in indegree.items() if degree == 0]
                visitedCount = 0
                while queue:
                    node = queue.pop()
                    visitedCount += 1
                    for nxt in successors.get(node, ()):
                        indegree[nxt] -= 1
                        if indegree[nxt] == 0:
                            queue.append(nxt)
                if visitedCount != len(indegree):
                    return False

        return True

    def slotFeasibleSetContains(self, containerSlot, subsetSlot):
        """True if F(subsetSlot) is provably contained in F(containerSlot)."""
        return (
            self.requiredArcsBySlot[containerSlot].issubset(
                self.requiredArcsBySlot[subsetSlot]
            )
            and self.forbiddenArcsBySlot[containerSlot].issubset(
                self.forbiddenArcsBySlot[subsetSlot]
            )
            and self.requiredPrecedencesBySlot[containerSlot].issubset(
                self.requiredPrecedencesBySlot[subsetSlot]
            )
            and self.forbiddenPrecedencesBySlot[containerSlot].issubset(
                self.forbiddenPrecedencesBySlot[subsetSlot]
            )
        )

    def canonicalKey(self):
        """Vehicle-permutation-invariant key for homogeneous-slot symmetry."""
        slotStates = [
            (
                tuple(sorted(self.requiredArcsBySlot[k])),
                tuple(sorted(self.forbiddenArcsBySlot[k])),
                tuple(sorted(self.requiredPrecedencesBySlot[k])),
                tuple(sorted(self.forbiddenPrecedencesBySlot[k])),
            )
            for k in range(self.numberOfSlots)
        ]
        return tuple(sorted(slotStates, key=repr))

    def describe(self):
        return {
            k: {
                'requiredPhysicalArcs': tuple(sorted(self.requiredArcsBySlot[k])),
                'forbiddenPhysicalArcs': tuple(sorted(self.forbiddenArcsBySlot[k])),
                'requiredCustomerPrecedences': tuple(
                    sorted(self.requiredPrecedencesBySlot[k])
                ),
                'forbiddenCustomerPrecedences': tuple(
                    sorted(self.forbiddenPrecedencesBySlot[k])
                ),
            }
            for k in range(self.numberOfSlots)
        }


class ArcFlowBrancher:
    """Branch on physical arcs, then on customer precedence if necessary.

    Physical arcs are the primary branch objects.  If the LP is still
    fractional after every physical-arc flow is integral, a slot-specific
    customer-precedence indicator separates distinct route orders without
    adding any BSS-entry state to pricing.
    """

    def __init__(self, modelData, tolerance=1e-7):
        self.data = modelData
        self.tolerance = float(tolerance)
        self.stationNode = modelData.S[0]

    def _support(self, master, support=None):
        if support is not None:
            return support
        return master.getPositiveX(tolerance=self.tolerance * 0.1)

    def getArcFlows(self, master, support=None):
        flows = {}
        for k, _, column, value in self._support(master, support):
            if column.isEmpty:
                continue
            for arc in column.physicalArcSet:
                key = (k, arc)
                flows[key] = flows.get(key, 0.0) + float(value)
        return flows

    def getPrecedenceFlows(self, master, support=None):
        flows = {}
        for k, _, column, value in self._support(master, support):
            if column.isEmpty:
                continue
            for pair in column.precedenceSet:
                key = (k, pair)
                flows[key] = flows.get(key, 0.0) + float(value)
        return flows

    def _arcTypePriority(self, arc):
        """Prefer customer/depot structure over BSS incidence on exact ties."""
        u, v = arc
        return 1 if (u == self.stationNode or v == self.stationNode) else 0

    def choose(self, master, branchingState=None, support=None):
        tol = self.tolerance
        arcCandidates = []
        for (slot, arc), value in self.getArcFlows(master, support=support).items():
            if value <= tol or value >= 1.0 - tol:
                continue
            if branchingState is not None:
                if arc in branchingState.requiredArcs(slot):
                    continue
                if arc in branchingState.forbiddenArcs(slot):
                    continue
            arcCandidates.append((
                abs(value - 0.5),
                self._arcTypePriority(arc),
                slot,
                arc,
                float(value),
            ))

        if arcCandidates:
            _, _, slot, arc, value = min(arcCandidates)
            return ArcBranchDecision(slot=slot, arc=arc, flow=value)

        # With repeatable S, distinct complete route orders can share one
        # physical-arc incidence vector.  Their customer total orders must then
        # differ, so a fractional precedence indicator separates them.
        precedenceCandidates = []
        for (slot, pair), value in self.getPrecedenceFlows(master, support=support).items():
            if value <= tol or value >= 1.0 - tol:
                continue
            if branchingState is not None:
                if pair in branchingState.requiredPrecedences(slot):
                    continue
                if pair in branchingState.forbiddenPrecedences(slot):
                    continue
            precedenceCandidates.append((
                abs(value - 0.5), slot, pair, float(value)
            ))

        if precedenceCandidates:
            _, slot, pair, value = min(precedenceCandidates)
            return PrecedenceBranchDecision(slot=slot, pair=pair, flow=value)
        return None

    def isIntegral(self, master, support=None):
        tol = self.tolerance
        support = self._support(master, support)
        return all(
            not (tol < value < 1.0 - tol)
            for _, _, _, value in support
        )
