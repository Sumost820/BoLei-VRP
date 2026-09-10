from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BssOptimalityCut:
    """
    Combinatorial optimality cut for one integer route combination.

    Let R* be the selected NONEMPTY route signatures and let T* be the proven
    optimal single-BSS makespan for exactly those route sequences. The cut is

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
