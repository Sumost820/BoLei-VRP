from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class SwapEvent:
    """One fixed battery-swap event embedded in a complete route column."""

    index: int
    afterTask: int
    arrivalTime: float
    baseStartTime: float
    baseEndTime: float
    gapFromPreviousSwap: Optional[float] = None
    tailDuration: Optional[float] = None


@dataclass(frozen=True)
class RouteColumn:
    """A BPC column is one COMPLETE physical route.

    ``nodes`` is the column identity.  It starts at ``startNode``, ends at
    ``endNode``, contains each customer at most once, and may contain the one
    physical BSS node repeatedly.  Hence two routes with the same customer
    order but different BSS visits are distinct columns.

    Example::

        (0, 3, S, 5, 8, S, 2, end)

    ``tasks`` is retained only as a cached customer projection for coverage
    constraints and SRCs. Branching acts directly on ``nodes`` physical arcs.
    """

    columnId: int
    nodes: Tuple[int, ...]
    tasks: Tuple[int, ...]
    duration: float
    swapEvents: Tuple[SwapEvent, ...] = field(default_factory=tuple)
    _taskSetCache: frozenset = field(default=None, init=False, repr=False, compare=False)
    _physicalArcSetCache: frozenset = field(default=None, init=False, repr=False, compare=False)
    _precedenceSetCache: frozenset = field(default=None, init=False, repr=False, compare=False)

    @property
    def signature(self):
        # Complete physical route identity, including every repeated BSS visit.
        return self.nodes

    @property
    def taskSet(self):
        if self._taskSetCache is None:
            object.__setattr__(self, '_taskSetCache', frozenset(self.tasks))
        return self._taskSetCache

    @property
    def physicalArcSet(self):
        if self._physicalArcSetCache is None:
            object.__setattr__(
                self,
                '_physicalArcSetCache',
                frozenset(zip(self.nodes[:-1], self.nodes[1:])),
            )
        return self._physicalArcSetCache

    @property
    def precedenceSet(self):
        if self._precedenceSetCache is None:
            tasks = tuple(self.tasks)
            object.__setattr__(
                self,
                '_precedenceSetCache',
                frozenset(
                    (tasks[a], tasks[b])
                    for a in range(len(tasks))
                    for b in range(a + 1, len(tasks))
                ),
            )
        return self._precedenceSetCache

    @property
    def stationVisitCount(self):
        return len(self.swapEvents)

    @property
    def isEmpty(self):
        return len(self.tasks) == 0

    def covers(self, task):
        return task in self.taskSet
