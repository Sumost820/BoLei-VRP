from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class RouteColumn:
    """
    A column is identified by its ordered CUSTOMER sequence only.

    baseNodes/baseSwapEvents store one isolated-route optimal swap placement
    achieving ``duration``. They are diagnostic data, not part of the column
    identity. This distinction is intentional: in the later BSS subproblem,
    swap placement will be re-optimized jointly across selected routes.
    """

    columnId: int
    tasks: Tuple[int, ...]
    duration: float
    baseNodes: Tuple[int, ...] = field(default_factory=tuple)
    baseSwapEvents: Tuple[tuple, ...] = field(default_factory=tuple)
    _taskSetCache: frozenset = field(default=None, init=False, repr=False, compare=False)

    @property
    def signature(self):
        return self.tasks

    @property
    def taskSet(self):
        # Cache the immutable customer set. This property is queried repeatedly
        # in RMP construction, branching compatibility and cut separation.
        if self._taskSetCache is None:
            object.__setattr__(self, "_taskSetCache", frozenset(self.tasks))
        return self._taskSetCache

    @property
    def isEmpty(self):
        return len(self.tasks) == 0

    def covers(self, task):
        return task in self.taskSet
