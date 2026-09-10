"""
严谨性 + 性能对比测试：原版 ExactLabelingPricing vs 优化版。

严谨性验证：对相同的 data + duals，两版返回的负 reduced cost 列集合必须一致。
性能对比：标签生成数、支配数、支配率、耗时。
"""
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, 'src')

from boleiScheduling.mockData import createMockData
from boleiScheduling.bpc.swap_dp import ExactFixedSequenceEvaluator
from boleiScheduling.bpc.pricing import ExactLabelingPricing
from boleiScheduling.bpc.pricing_optimized import ExactLabelingPricingOptimized


@dataclass
class FakeDuals:
    coverage: dict
    slot: dict
    makespan: dict


def make_duals(data, seed=0):
    import random
    random.seed(seed)
    coverage = {i: random.uniform(5.0, 30.0) for i in data.C}
    slot = {k: random.uniform(-5.0, 5.0) for k in range(data.K)}
    makespan = {k: random.uniform(0.5, 2.0) for k in range(data.K)}
    return FakeDuals(coverage=coverage, slot=slot, makespan=makespan)


def candidate_signatures(candidates):
    """提取候选列的 signature -> best reduced cost 映射。"""
    return {
        c.column.signature: (c.reducedCost, c.bestSlot)
        for c in candidates
    }


def run_comparison(taskCount, K, seed=4, dualSeed=0, rounds=3):
    data = createMockData(
        taskCount=taskCount,
        stationCopyCount=taskCount,
        K=K,
        seed=seed,
        Q=50,
        QMin=20,
    )
    evaluator = ExactFixedSequenceEvaluator(data)
    duals = make_duals(data, seed=dualSeed)

    # --- 原版 ---
    pricing_orig = ExactLabelingPricing(
        modelData=data,
        sequenceEvaluator=evaluator,
    )
    # 预热
    pricing_orig.price(duals=duals, existingSignatures=set(), maxColumns=None)
    orig_times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        cand_orig = pricing_orig.price(
            duals=duals, existingSignatures=set(), maxColumns=None
        )
        orig_times.append(time.perf_counter() - t0)
    stats_orig = pricing_orig.lastStatistics

    # --- 优化版 ---
    pricing_opt = ExactLabelingPricingOptimized(
        modelData=data,
        sequenceEvaluator=evaluator,
    )
    # 预热
    pricing_opt.price(duals=duals, existingSignatures=set(), maxColumns=None)
    opt_times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        cand_opt = pricing_opt.price(
            duals=duals, existingSignatures=set(), maxColumns=None
        )
        opt_times.append(time.perf_counter() - t0)
    stats_opt = pricing_opt.lastStatistics

    # --- 严谨性验证 ---
    sig_orig = candidate_signatures(cand_orig)
    sig_opt = candidate_signatures(cand_opt)

    only_orig = set(sig_orig.keys()) - set(sig_opt.keys())
    only_opt = set(sig_opt.keys()) - set(sig_orig.keys())

    # 检查共同列的 reduced cost 是否一致
    rc_mismatch = []
    for sig in set(sig_orig.keys()) & set(sig_opt.keys()):
        rc_o, slot_o = sig_orig[sig]
        rc_p, slot_p = sig_opt[sig]
        if abs(rc_o - rc_p) > 1e-6:
            rc_mismatch.append((sig, rc_o, rc_p))

    exact_match = (not only_orig) and (not only_opt) and (not rc_mismatch)

    return {
        'taskCount': taskCount,
        'K': K,
        'exact_match': exact_match,
        'only_orig_count': len(only_orig),
        'only_opt_count': len(only_opt),
        'rc_mismatch_count': len(rc_mismatch),
        'only_orig_samples': list(only_orig)[:5],
        'only_opt_samples': list(only_opt)[:5],
        'rc_mismatch_samples': rc_mismatch[:5],
        'orig': {
            'time': sum(orig_times) / len(orig_times),
            'generatedLabels': stats_orig['generatedLabels'],
            'acceptedLabels': stats_orig['acceptedLabels'],
            'dominatedLabels': stats_orig['dominatedLabels'],
            'removedByDominance': stats_orig['removedByDominance'],
            'boundPrunedLabels': stats_orig['boundPrunedLabels'],
            'states': stats_orig['states'],
            'negativeColumns': stats_orig['negativeColumnsReturned'],
            'dominance_rate': stats_orig['dominatedLabels'] / max(1, stats_orig['generatedLabels']),
        },
        'opt': {
            'time': sum(opt_times) / len(opt_times),
            'generatedLabels': stats_opt['generatedLabels'],
            'acceptedLabels': stats_opt['acceptedLabels'],
            'dominatedLabels': stats_opt['dominatedLabels'],
            'removedByDominance': stats_opt['removedByDominance'],
            'boundPrunedLabels': stats_opt['boundPrunedLabels'],
            'states': stats_opt['states'],
            'negativeColumns': stats_opt['negativeColumnsReturned'],
            'dominance_rate': stats_opt['dominatedLabels'] / max(1, stats_opt['generatedLabels']),
        },
    }


if __name__ == '__main__':
    print("=" * 120)
    print("严谨性 + 性能对比：原版 ExactLabelingPricing vs 优化版")
    print("=" * 120)

    all_pass = True
    for N, K in [(8, 2), (10, 2), (12, 2), (15, 3)]:
        r = run_comparison(N, K, rounds=3)

        print(f"\n--- N={N}, K={K} ---")
        print(f"  严谨性: {'PASS ✓' if r['exact_match'] else 'FAIL ✗'}")
        if not r['exact_match']:
            print(f"    仅原版有: {r['only_orig_count']} 列, 样本: {r['only_orig_samples']}")
            print(f"    仅优化版有: {r['only_opt_count']} 列, 样本: {r['only_opt_samples']}")
            print(f"    reduced cost 不一致: {r['rc_mismatch_count']} 列, 样本: {r['rc_mismatch_samples']}")
            all_pass = False

        o = r['orig']
        p = r['opt']
        speedup = o['time'] / max(1e-9, p['time'])
        label_reduction = 1.0 - p['generatedLabels'] / max(1, o['generatedLabels'])

        print(f"  性能:")
        print(f"    耗时:          原版 {o['time']:.4f}s  |  优化版 {p['time']:.4f}s  |  加速比 {speedup:.2f}x")
        print(f"    生成标签:      原版 {o['generatedLabels']:>10}  |  优化版 {p['generatedLabels']:>10}  |  减少 {label_reduction:.1%}")
        print(f"    接受标签:      原版 {o['acceptedLabels']:>10}  |  优化版 {p['acceptedLabels']:>10}")
        print(f"    被支配标签:    原版 {o['dominatedLabels']:>10}  |  优化版 {p['dominatedLabels']:>10}")
        print(f"    支配移除标签:  原版 {o['removedByDominance']:>10}  |  优化版 {p['removedByDominance']:>10}")
        print(f"    bound剪枝:     原版 {o['boundPrunedLabels']:>10}  |  优化版 {p['boundPrunedLabels']:>10}")
        print(f"    状态数:        原版 {o['states']:>10}  |  优化版 {p['states']:>10}")
        print(f"    支配率:        原版 {o['dominance_rate']:.1%}  |  优化版 {p['dominance_rate']:.1%}")
        print(f"    负列数:        原版 {o['negativeColumns']:>10}  |  优化版 {p['negativeColumns']:>10}")

    print("\n" + "=" * 120)
    print(f"总体严谨性: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    print("=" * 120)
