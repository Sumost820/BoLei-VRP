from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThresholdArcBranchDecision:
    """Aggregate physical-arc branching decision F_uv in {0,1}."""

    arc: tuple
    flow: float


@dataclass(frozen=True)
class ThresholdRouteBranchDecision:
    """Rare exact fallback: branch directly on one existing route variable."""

    signature: tuple
    value: float


@dataclass(frozen=True)
class ThresholdBranchState:
    """Vehicle-free branch state.

    The primary branching mechanism stores only globally forbidden physical
    arcs.  A force branch F_uv=1 is compiled once, when the child is created,
    into forbidden competing arcs around customer endpoints.  Pricing therefore
    only needs to test whether the next physical arc is forbidden.

    ``forbiddenSignatures`` / ``requiredSignatures`` are used only as a rare
    exact fallback if all aggregate customer-incident arc flows are integral
    while route variables remain fractional.  They never introduce a vehicle
    index and they do not enter the pricing resource label.
    """

    forbiddenArcs: frozenset = field(default_factory=frozenset)
    forbiddenSignatures: frozenset = field(default_factory=frozenset)
    requiredSignatures: frozenset = field(default_factory=frozenset)

    @staticmethod
    def root():
        return ThresholdBranchState()

    def isColumnCompatible(self, column):
        signature = tuple(column.signature)
        if signature in self.forbiddenSignatures:
            return False
        return not bool(self.forbiddenArcs.intersection(column.physicalArcSet))

    def childForbidArc(self, arc):
        arc = tuple(arc)
        return ThresholdBranchState(
            forbiddenArcs=self.forbiddenArcs | frozenset((arc,)),
            forbiddenSignatures=self.forbiddenSignatures,
            requiredSignatures=self.requiredSignatures,
        )

    def childForceArc(self, modelData, arc):
        """Compile F_uv=1 into forbidden alternatives.

        The construction is valid when at least one endpoint is a customer.
        If u is a customer, every other outgoing arc u->w is forbidden.  If v
        is a customer, every other incoming arc w->v is forbidden.  Customer
        coverage then forces the selected route containing that customer to use
        u->v.  BSS/depot endpoints are deliberately not made unique.
        """
        u, v = tuple(arc)
        customers = set(modelData.C)
        if u not in customers and v not in customers:
            raise ValueError('Aggregate force-arc branching needs a customer endpoint')

        forbidden = set(self.forbiddenArcs)
        for a, b in modelData.A:
            candidate = (a, b)
            if candidate == (u, v):
                continue
            if u in customers and a == u:
                forbidden.add(candidate)
            if v in customers and b == v:
                forbidden.add(candidate)

        return ThresholdBranchState(
            forbiddenArcs=frozenset(forbidden),
            forbiddenSignatures=self.forbiddenSignatures,
            requiredSignatures=self.requiredSignatures,
        )

    def childRoute(self, signature, value):
        signature = tuple(signature)
        if value not in (0, 1):
            raise ValueError('route branch value must be 0 or 1')
        if value == 0:
            return ThresholdBranchState(
                forbiddenArcs=self.forbiddenArcs,
                forbiddenSignatures=self.forbiddenSignatures | frozenset((signature,)),
                requiredSignatures=self.requiredSignatures,
            )
        return ThresholdBranchState(
            forbiddenArcs=self.forbiddenArcs,
            forbiddenSignatures=self.forbiddenSignatures,
            requiredSignatures=self.requiredSignatures | frozenset((signature,)),
        )

    def protectedSignatures(self):
        # These named-route branch rows are master-only.  Protect their prefixes
        # in pricing dominance for the same reason as BSS no-good rows.
        return set(self.forbiddenSignatures) | set(self.requiredSignatures)

    def canonicalKey(self):
        return (
            tuple(sorted(self.forbiddenArcs)),
            tuple(sorted(self.forbiddenSignatures)),
            tuple(sorted(self.requiredSignatures)),
        )


class ThresholdArcFlowBrancher:
    """Vehicle-free aggregate arc-flow branching.

    For every physical arc incident to at least one customer,

        F_uv = sum_r a_uv^r x_r.

    Customer set-partitioning implies 0 <= F_uv <= 1.  Fractional F_uv can
    therefore be branched on as F_uv=0 versus F_uv=1.  A route-variable branch
    is retained only as an exact fallback for the rare case in which all such
    arc flows are integral while some x_r remains fractional.
    """

    def __init__(self, modelData, integralityTolerance=1e-7):
        self.data = modelData
        self.integralityTolerance = float(integralityTolerance)
        self.customerSet = frozenset(modelData.C)
        self.stationSet = frozenset(modelData.S)

    def getArcFlows(self, master, support=None):
        if support is None:
            support = master.getPositiveX(tolerance=1e-12)
        flows = {}
        for _, column, value in support:
            value = float(value)
            if value <= 1e-12:
                continue
            for arc in column.physicalArcSet:
                u, v = arc
                if u not in self.customerSet and v not in self.customerSet:
                    continue
                flows[arc] = flows.get(arc, 0.0) + value
        return flows

    def choose(self, master, branchingState, support=None):
        if support is None:
            support = master.getPositiveX(tolerance=1e-12)

        tol = self.integralityTolerance
        candidates = []
        for arc, value in self.getArcFlows(master, support=support).items():
            if arc in branchingState.forbiddenArcs:
                continue
            if value <= tol or value >= 1.0 - tol:
                continue
            u, v = arc
            # Prefer customer-customer/depot arcs over BSS incident arcs, but
            # fractionality remains the primary score inside each class.
            bssPenalty = int(u in self.stationSet or v in self.stationSet)
            candidates.append((bssPenalty, abs(value - 0.5), arc, float(value)))

        if candidates:
            _, _, arc, value = min(candidates)
            return ThresholdArcBranchDecision(arc=arc, flow=value)

        # Exact fallback.  This does not pollute pricing resources: x_r=0/1 is
        # represented by a master-only named-route branch row/omission.
        routeCandidates = []
        for routeIndex, column, value in support:
            value = float(value)
            if value <= tol or value >= 1.0 - tol:
                continue
            signature = tuple(column.signature)
            if signature in branchingState.forbiddenSignatures:
                continue
            routeCandidates.append((abs(value - 0.5), routeIndex, signature, value))
        if routeCandidates:
            _, _, signature, value = min(routeCandidates)
            return ThresholdRouteBranchDecision(signature=signature, value=value)
        return None
