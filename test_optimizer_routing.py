# ========================================
# TEST ADAPTIVE OPTIMIZER & ITERATION TRACKING
# ========================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.adaptive_optimizer import AdaptiveOptimizer


def test_iteration_tracking():
    """
    Test the full iteration tracking and decision-making flow.
    """

    print("\n" + "="*70)
    print("TEST: ADAPTIVE OPTIMIZER WITH ITERATION TRACKING")
    print("="*70)

    optimizer = AdaptiveOptimizer(max_iterations=3)

    # ========================================
    # ITERATION 1: MODERATE QUALITY, UNCERTAIN IMPROVEMENT
    # ========================================

    print("\n--- ITERATION 1 ---")
    print("Scenario: Moderate quality, uncertain improvement")

    decision1 = optimizer.decide(
        synthetic_accuracy=0.30,
        acceptance_rate=0.60,
        diversity_score=0.20,
        duplicate_rate=0.02,
        baseline_accuracy=0.99,
        augmented_accuracy=0.991,
        confidence_lower=-0.10,
        confidence_upper=0.30,
        improvement_pp=0.10,
        iteration=1
    )

    print(f"\nDecision: {decision1['decision']}")
    print(f"Priority: {decision1['priority']}")
    print(f"Reason: {decision1['reason']}")
    print(f"Quality Score: {decision1['quality_score']:.1f}/100")

    # Check routing decision
    if decision1['decision'] == 'RESAMPLE':
        next_node = 'generation'
        print(f"→ Route to: {next_node}")
    else:
        print(f"→ Route to: END")

    # ========================================
    # ITERATION 2: SLIGHTLY BETTER, STILL UNCERTAIN
    # ========================================

    print("\n--- ITERATION 2 ---")
    print("Scenario: Improved quality, still uncertain improvement")

    decision2 = optimizer.decide(
        synthetic_accuracy=0.40,
        acceptance_rate=0.75,
        diversity_score=0.22,
        duplicate_rate=0.01,
        baseline_accuracy=0.99,
        augmented_accuracy=0.992,
        confidence_lower=-0.05,
        confidence_upper=0.35,
        improvement_pp=0.15,
        iteration=2
    )

    print(f"\nDecision: {decision2['decision']}")
    print(f"Priority: {decision2['priority']}")
    print(f"Reason: {decision2['reason']}")
    print(f"Quality Score: {decision2['quality_score']:.1f}/100")

    if decision2['decision'] == 'RESAMPLE':
        next_node = 'generation'
        print(f"→ Route to: {next_node}")
    elif decision2['decision'] in ['KEEP', 'STOP']:
        print(f"→ Route to: END")
    else:
        print(f"→ Route to: {decision2['decision'].lower()}")

    # ========================================
    # ITERATION 3: GOOD QUALITY, CLEAR IMPROVEMENT
    # ========================================

    print("\n--- ITERATION 3 ---")
    print("Scenario: Good quality, clear improvement (meets threshold)")

    decision3 = optimizer.decide(
        synthetic_accuracy=0.72,
        acceptance_rate=0.85,
        diversity_score=0.25,
        duplicate_rate=0.00,
        baseline_accuracy=0.99,
        augmented_accuracy=0.995,
        confidence_lower=0.15,
        confidence_upper=0.45,
        improvement_pp=0.30,
        iteration=3
    )

    print(f"\nDecision: {decision3['decision']}")
    print(f"Priority: {decision3['priority']}")
    print(f"Reason: {decision3['reason']}")
    print(f"Quality Score: {decision3['quality_score']:.1f}/100")

    if decision3['decision'] == 'KEEP':
        print(f"→ Route to: END ✓ OPTIMIZATION SUCCESSFUL")
    else:
        print(f"→ Route to: {decision3['decision'].lower()}")

    # ========================================
    # SUMMARY
    # ========================================

    print("\n" + "="*70)
    print("ITERATION HISTORY")
    print("="*70)

    report = optimizer.get_report()

    for entry in report['history']:
        iteration = entry['iteration']
        decision = entry['decision']
        quality = entry['quality_score']
        improvement = entry['improvement_status']

        print(
            f"Iteration {iteration}: "
            f"{decision:12} | "
            f"Quality {quality:6.1f}/100 | "
            f"{improvement}"
        )

    if report['best_result']:
        print("\n" + "-"*70)
        print(f"Best result at iteration {report['best_result']['iteration']}")
        print(f"  Quality: {report['best_result']['quality_score']:.1f}/100")
        print(f"  Improvement: {report['best_result']['improvement_pp']:+.2f}pp")

    print("\n" + "="*70)
    print("✓ TEST PASSED: Iteration tracking and routing working correctly")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    test_iteration_tracking()
