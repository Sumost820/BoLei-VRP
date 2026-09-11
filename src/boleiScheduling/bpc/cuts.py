from dataclasses import dataclass
from itertools import combinations
from typing import Tuple


@dataclass(frozen=True)
class BssOptimalityCut:
    """
    Combinatorial optimality cut for one integer route combination.

    Let R* be the selected NONEMPTY COMPLETE-route signatures and let T* be the proven
    optimal single-BSS makespan for exactly those fixed physical routes. The cut is

        T >= T* (sum_{r in R*} sum_k x[k,r] - |R*| + 1).

    It is vehicle-slot permutation invariant because it uses sum_k x[k,r].
    If all routes in R* are selected, the RHS is T*. If at least one is not
    selected, the RHS is <= 0 and the cut is inactive because T >= 0 already.
    """

    routeSignatures: Tuple[Tuple[int, ...], ...]
    value: float

    @staticmethod
    def canonical(routeSignatures, value):
        signatures = tuple(sorted(tuple(sig) for sig in routeSignatures))
        return BssOptimalityCut(
            routeSignatures=signatures,
            value=float(value),
        )

    @property
    def key(self):
        return self.routeSignatures


@dataclass(frozen=True)
class SubsetRowCut:
    """
    Rank-1 subset-row cut (SRC) with divisor 2.

    For a customer subset S, summing the set-partitioning rows over S and
    applying a Chvatal-Gomory rounding with multiplier 1/2 yields

        sum_{k,r} floor(|S intersect r| / 2) x[k,r]
            <= floor(|S| / 2).

    The current separator is aimed at the classical 3-row SRC family (|S|=3),
    for which the coefficient is simply one if a route contains at least two
    customers from the triple and zero otherwise.  The class itself supports
    any subset with divisor 2 so pricing can remain exact if larger odd subsets
    are added later.
    """

    customers: Tuple[int, ...]
    divisor: int = 2

    @staticmethod
    def canonical(customers, divisor=2):
        customers = tuple(sorted(set(int(i) for i in customers)))
        if len(customers) < 2:
            raise ValueError('SRC requires at least two distinct customers')
        if divisor != 2:
            raise ValueError('Current exact pricing implementation supports divisor=2 SRCs')
        return SubsetRowCut(customers=customers, divisor=divisor)

    @property
    def key(self):
        return (self.divisor, self.customers)

    @property
    def rhs(self):
        return len(self.customers) // self.divisor

    @property
    def customerSet(self):
        return frozenset(self.customers)

    def coefficientFromTaskSet(self, taskSet):
        return len(self.customerSet.intersection(taskSet)) // self.divisor

    def coefficient(self, column):
        return self.coefficientFromTaskSet(column.taskSet)


@dataclass(frozen=True)
class SubsetRowViolation:
    cut: SubsetRowCut
    lhs: float
    rhs: float
    violation: float


class ThreeRowSubsetCutSeparator:
    """
    Exact separator over the classical 3-customer SRC family.

    For S={i,j,h}, the inequality is

        sum_{k,r: |r intersect S| >= 2} x[k,r] <= 1.

    Separation is performed on the current *fully priced* LP solution.  The
    separator scans every customer triple, so whenever it reports no violation
    there is no violated 3-row SRC for that LP solution.  Adding only this SRC
    family is optional strengthening: it never changes the integer feasible set
    and therefore never compromises the exactness of the BPC algorithm.
    """

    def __init__(
        self,
        modelData,
        violationTolerance=1e-7,
        maxCutsPerRound=10,
    ):
        self.data = modelData
        self.violationTolerance = float(violationTolerance)
        self.maxCutsPerRound = maxCutsPerRound
        self.tasks = tuple(modelData.C)

    def separate(self, master, existingKeys=None):
        existingKeys = set(existingKeys or ())
        support = master.getPositiveX(tolerance=1e-10)

        # Aggregate the slot-specific route variables into route/task-set
        # contributions.  Empty routes never contribute to an SRC.
        weightedTaskSets = []
        for _, _, column, value in support:
            if column.isEmpty or value <= 1e-12:
                continue
            weightedTaskSets.append((column.taskSet, float(value)))

        if not weightedTaskSets or len(self.tasks) < 3:
            return []

        violations = []
        for triple in combinations(self.tasks, 3):
            cut = SubsetRowCut.canonical(triple)
            if cut.key in existingKeys:
                continue

            subset = cut.customerSet
            lhs = 0.0
            for taskSet, value in weightedTaskSets:
                if len(subset.intersection(taskSet)) >= 2:
                    lhs += value

            violation = lhs - 1.0
            if violation > self.violationTolerance:
                violations.append(
                    SubsetRowViolation(
                        cut=cut,
                        lhs=lhs,
                        rhs=1.0,
                        violation=violation,
                    )
                )

        violations.sort(
            key=lambda item: (
                -item.violation,
                item.cut.customers,
            )
        )
        if self.maxCutsPerRound is not None:
            violations = violations[: self.maxCutsPerRound]
        return violations
