from dataclasses import dataclass


def route_sequence_arcs(tasks, start_node, end_node):
    """
    Directed successor arcs of a route's CUSTOMER sequence.

    The BPC column identity is an ordered customer sequence; swap positions are
    optimized separately. Therefore branching is performed on the compressed
    route graph (depot/customers/depot), not on physical customer->BSS arcs.

    For tasks=(i1,...,im), the arc set is
        (start,i1), (i1,i2), ..., (im,end).
    The empty route has no branching arcs.
    """
    tasks = tuple(tasks)
    if not tasks:
        return ()
    nodes = (start_node,) + tasks + (end_node,)
    return tuple(zip(nodes[:-1], nodes[1:]))


@dataclass(frozen=True)
class ArcBranchDecision:
    slot: int
    arc: tuple
    value: int
    flow: float = None


@dataclass(frozen=True)
class ArcBranchingState:
    """
    Branch restrictions for slot-specific successor-arc flow branching.

    For each vehicle slot k define
        f^k_ij = sum_r a^r_ij x_kr,
    where a^r_ij=1 iff (i,j) is a consecutive arc in the compressed customer
    sequence of route r.

    A branch f^k_ij=0 forbids that arc in every column available to slot k.
    A branch f^k_ij=1 requires that arc in every column available to slot k.
    Because sum_r x_kr=1, these column-universe restrictions are exactly
    equivalent to the two branch equations and introduce no branch-row duals.
    """

    requiredBySlot: tuple
    forbiddenBySlot: tuple

    @staticmethod
    def root(numberOfSlots):
        return ArcBranchingState(
            requiredBySlot=tuple(frozenset() for _ in range(numberOfSlots)),
            forbiddenBySlot=tuple(frozenset() for _ in range(numberOfSlots)),
        )

    @property
    def numberOfSlots(self):
        return len(self.requiredBySlot)

    def requiredArcs(self, slot):
        return self.requiredBySlot[slot]

    def forbiddenArcs(self, slot):
        return self.forbiddenBySlot[slot]

    def child(self, slot, arc, value, startNode=None, endNode=None):
        if value not in (0, 1):
            raise ValueError('arc branch value must be 0 or 1')
        if slot < 0 or slot >= self.numberOfSlots:
            raise ValueError('invalid vehicle slot')

        required = [set(arcs) for arcs in self.requiredBySlot]
        forbidden = [set(arcs) for arcs in self.forbiddenBySlot]

        if value == 1:
            required[slot].add(tuple(arc))
        else:
            forbidden[slot].add(tuple(arc))

        state = ArcBranchingState(
            requiredBySlot=tuple(frozenset(arcs) for arcs in required),
            forbiddenBySlot=tuple(frozenset(arcs) for arcs in forbidden),
        )
        if startNode is not None and endNode is not None:
            if not state.isStructurallyConsistent(startNode, endNode):
                return None
        return state

    def isStructurallyConsistent(self, startNode, endNode):
        for k in range(self.numberOfSlots):
            required = set(self.requiredBySlot[k])
            forbidden = set(self.forbiddenBySlot[k])
            if required & forbidden:
                return False

            out_required = {}
            in_required = {}
            for i, j in required:
                if j == startNode or i == endNode:
                    return False
                if i == startNode and j == endNode:
                    # Empty routes are not represented by a physical depot arc
                    # in the current routing graph.
                    return False
                if i in out_required and out_required[i] != j:
                    return False
                if j in in_required and in_required[j] != i:
                    return False
                out_required[i] = j
                in_required[j] = i

            # A directed cycle among customers can never be part of an
            # elementary start-to-end route.
            for first in list(out_required):
                if first in (startNode, endNode):
                    continue
                seen = set()
                node = first
                while node in out_required and node not in (startNode, endNode):
                    if node in seen:
                        return False
                    seen.add(node)
                    node = out_required[node]

        return True

    def isSequenceCompatible(self, slot, tasks, startNode, endNode):
        arcs = set(route_sequence_arcs(tasks, startNode, endNode))
        return (
            self.requiredBySlot[slot].issubset(arcs)
            and not (self.forbiddenBySlot[slot] & arcs)
        )

    def isColumnCompatible(self, slot, column, startNode, endNode):
        return self.isSequenceCompatible(slot, column.tasks, startNode, endNode)

    def describe(self):
        return {
            k: {
                'required': tuple(sorted(self.requiredBySlot[k])),
                'forbidden': tuple(sorted(self.forbiddenBySlot[k])),
            }
            for k in range(self.numberOfSlots)
        }


class ArcFlowBrancher:
    """Select a fractional slot-specific successor arc, closest to 0.5."""

    def __init__(self, modelData, tolerance=1e-7):
        self.data = modelData
        self.tolerance = tolerance

    def getArcFlows(self, master):
        flows = {}
        for k, _, column, value in master.getPositiveX(tolerance=self.tolerance * 0.1):
            for arc in route_sequence_arcs(
                column.tasks,
                self.data.startNode,
                self.data.endNode,
            ):
                key = (k, arc)
                flows[key] = flows.get(key, 0.0) + float(value)
        return flows

    def choose(self, master, branchingState=None):
        tol = self.tolerance
        flows = self.getArcFlows(master)
        candidates = []

        for (slot, arc), value in flows.items():
            if value <= tol or value >= 1.0 - tol:
                continue
            if branchingState is not None:
                if arc in branchingState.requiredArcs(slot):
                    continue
                if arc in branchingState.forbiddenArcs(slot):
                    continue
            candidates.append((abs(value - 0.5), -min(value, 1.0 - value), slot, arc, value))

        if not candidates:
            return None

        _, _, slot, arc, value = min(candidates)
        return ArcBranchDecision(slot=slot, arc=arc, value=-1, flow=float(value))

    def isIntegral(self, master):
        tol = self.tolerance
        for _, _, _, value in master.getPositiveX(tolerance=tol * 0.1):
            if tol < value < 1.0 - tol:
                return False
        return True
