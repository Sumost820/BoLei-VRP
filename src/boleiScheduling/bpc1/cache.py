"""Small cache helpers used by the BPC implementation.

The complete-route evaluation/warm-start cache lives in ``ExactCompleteRouteEvaluator``
because that class is the single owner of physical-route validation and timing. The
BSS route-combination cache lives in ``ExactBssScheduler`` so that route-index
remapping can be handled safely.  This module exposes their statistics in one
place without duplicating cache state.
"""


def collectCacheStatistics(routeEvaluator=None, bssScheduler=None):
    result = {}
    if routeEvaluator is not None and hasattr(routeEvaluator, 'getCacheStatistics'):
        result['route'] = routeEvaluator.getCacheStatistics()
    if bssScheduler is not None and hasattr(bssScheduler, 'getCacheStatistics'):
        result['bss'] = bssScheduler.getCacheStatistics()
    return result
