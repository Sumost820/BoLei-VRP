"""
独立基准测试：只跑 ExactLabelingPricing，不依赖 gurobipy。
用于量化支配规则优化前后的标签生成/支配/耗时差异。
"""
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, 'src')

from boleiScheduling.mockData import createMockData
from boleiScheduling.bpc.swap_dp import ExactFixedSequenceEvaluator
from boleiScheduling.bpc.pricing import ExactLabelingPricing


@dataclass
class FakeDuals:
    coverage: dict
    slot: dict
    makespan: dict


def make_duals(data, seed=0):
    """构造一组合理的 dual 值（模拟 RMP 求解后的对偶）。"""
    import random
    random.seed(seed)
    coverage = {i: random.uniform(5.0, 30.0) for i in data.C}
    slot = {k: random.uniform(-5.0, 5.0) for k in range(data.K)}
    makespan = {k: random.uniform(0.5, 2.0) for k in range(data.K)}
    return FakeDuals(coverage=coverage, slot=slot, makespan=makespan)


def run_benchmark(taskCount, K, seed=4, dualSeed=0, rounds=3):
    data = createMockData(
        taskCount=taskCount,
        stationCopyCount=taskCount,
        K=K,
        seed=seed,
        Q=50,
        QMin=20,
    )
    evaluator = ExactFixedSequenceEvaluator(data)
    pricing = ExactLabelingPricing(
        modelData=data,
        sequenceEvaluator=evaluator,
    )
    duals = make_duals(data, seed=dualSeed)

    # 预热
    pricing.price(duals=duals, existingSignatures=set(), maxColumns=50)

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        candidates = pricing.price(
            duals=duals,
            existingSignatures=set(),
            maxColumns=50,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    stats = pricing.lastStatistics
    avg_time = sum(times) / len(times)
    return {
        'taskCount': taskCount,
        'K': K,
        'avg_time_s': avg_time,
        'min_time_s': min(times),
        'generatedLabels': stats['generatedLabels'],
        'acceptedLabels': stats['acceptedLabels'],
        'dominatedLabels': stats['dominatedLabels'],
        'removedByDominance': stats['removedByDominance'],
        'boundPrunedLabels': stats['boundPrunedLabels'],
        'branchPrunedLabels': stats['branchPrunedLabels'],
        'completedRoutes': stats['completedRoutes'],
        'negativeCompletions': stats['negativeCompletions'],
        'states': stats['states'],
        'negativeColumnsReturned': stats['negativeColumnsReturned'],
        'dominance_rate': (
            stats['dominatedLabels'] / max(1, stats['generatedLabels'])
        ),
    }


if __name__ == '__main__':
    print(f"{'N':>4} {'K':>3} {'time(s)':>10} {'genLabels':>12} {'accepted':>10} "
          f"{'dominated':>10} {'removed':>10} {'boundPrune':>12} {'states':>10} "
          f"{'domRate':>8} {'negCols':>8}")
    print('-' * 120)
    for N, K in [(8, 2), (10, 2), (12, 2), (15, 3), (20, 3)]:
        r = run_benchmark(N, K, rounds=3)
        print(f"{r['taskCount']:>4} {r['K']:>3} {r['avg_time_s']:>10.4f} "
              f"{r['generatedLabels']:>12} {r['acceptedLabels']:>10} "
              f"{r['dominatedLabels']:>10} {r['removedByDominance']:>10} "
              f"{r['boundPrunedLabels']:>12} {r['states']:>10} "
              f"{r['dominance_rate']:>8.2%} {r['negativeColumnsReturned']:>8}")
