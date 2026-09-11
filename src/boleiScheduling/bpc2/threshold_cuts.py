from dataclasses import dataclass
from itertools import combinations
from typing import Tuple


@dataclass(frozen=True)
class ThresholdSubsetRowCut:
    """Rank-1 subset-row cut (SRC) for the vehicle-free threshold master.

    For a customer subset S and divisor 2,

        sum_r floor(|S intersect r| / 2) x_r <= floor(|S| / 2).

    The separator below uses the classical three-customer family.  The class
    itself keeps the divisor-2 definition explicit because pricing uses the
    same coefficient function for every current and future route column.
    """

    customers: Tuple[int, ...]
    divisor: int = 2

    @staticmethod
    def canonical(customers, divisor=2):
        customers = tuple(sorted(set(int(i) for i in customers)))
        if len(customers) < 2:
            raise ValueError('SRC requires at least two distinct customers')
        if divisor != 2:
            raise ValueError('Threshold pricing currently supports divisor=2 SRCs')
        return ThresholdSubsetRowCut(customers=customers, divisor=divisor)

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
class ThresholdSubsetRowViolation:
    cut: ThresholdSubsetRowCut
    lhs: float
    rhs: float
    violation: float


class ThresholdThreeRowSrcSeparator:
    """Exact separator for the classical three-row SRC family.

    For S={i,j,h}, the cut reduces to

        sum_{r: |r intersect S| >= 2} x_r <= 1.

    The threshold master is vehicle-free, so ``getPositiveX`` yields
    ``(routeIndex, column, value)`` triples.  Separation is only meaningful on
    a fully priced LP solution; the solver enforces that calling convention.
    """

    def __init__(self, modelData, violationTolerance=1e-7, maxCutsPerRound=5):
        self.data = modelData
        self.violationTolerance = float(violationTolerance)
        self.maxCutsPerRound = maxCutsPerRound
        self.tasks = tuple(modelData.C)

    def separate(self, master, existingKeys=None):
        existingKeys = set(existingKeys or ())
        support = master.getPositiveX(tolerance=1e-10)

        weightedTaskSets = []
        for _, column, value in support:
            if column.isEmpty or value <= 1e-12:
                continue
            weightedTaskSets.append((column.taskSet, float(value)))

        if not weightedTaskSets or len(self.tasks) < 3:
            return []

        violations = []
        for triple in combinations(self.tasks, 3):
            cut = ThresholdSubsetRowCut.canonical(triple)
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
                    ThresholdSubsetRowViolation(
                        cut=cut,
                        lhs=float(lhs),
                        rhs=1.0,
                        violation=float(violation),
                    )
                )

        violations.sort(key=lambda item: (-item.violation, item.cut.customers))
        if self.maxCutsPerRound is not None:
            violations = violations[: max(0, int(self.maxCutsPerRound))]
        return violations
