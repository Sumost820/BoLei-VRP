"""Small cache helpers used by the BPC implementation.

The expensive fixed-sequence route cache lives in ``ExactFixedSequenceEvaluator``
because that class is the single owner of the exact route evaluation logic.  The
BSS route-combination cache lives in ``ExactBssScheduler`` so that route-index
remapping can be handled safely.  This module exposes their statistics in one
place without duplicating cache state.
"""


def collectCacheStatistics(sequenceEvaluator=None, bssScheduler=None):
    result = {}
    if sequenceEvaluator is not None and hasattr(sequenceEvaluator, 'getCacheStatistics'):
        result['route'] = sequenceEvaluator.getCacheStatistics()
    if bssScheduler is not None and hasattr(bssScheduler, 'getCacheStatistics'):
        result['bss'] = bssScheduler.getCacheStatistics()
    return result
