import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Any

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluation.synthetic_experiment import run_experiment


DEFAULT_SEEDS = [42, 123, 456]


def run_multi_seed_experiment(
    model_path: str = "models/conditional_vae.pth",
    generation_strategy: str = "LATENT_MANIFOLD",
    latent_scale: float = 0.15,
    samples_per_minority_class: int = 1000,
    critic_threshold: float = 0.90,
    seeds: Optional[List[int]] = None,
    output_path: str = "experiments/multi_seed_results.json"
) -> Dict[str, Any]:
    """
    Executes multi-seed validation to assess statistical significance of synthetic augmentation.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    print("\n" + "=" * 65)
    print("GENFORGE MULTI-SEED RIGOROUS VALIDATION")
    print("=" * 65)
    print(f"Model: {model_path} | Strategy: {generation_strategy} (scale={latent_scale})")
    print(f"Seeds: {seeds} | Samples/Class: {samples_per_minority_class} | Critic Threshold: {critic_threshold}")
    print("=" * 65)

    seed_results = []
    improvements = []

    for seed in seeds:
        print(f"\n>>> Running Seed {seed}...")
        result = run_experiment(
            model_path=model_path,
            generation_strategy=generation_strategy,
            latent_scale=latent_scale,
            samples_per_minority_class=samples_per_minority_class,
            critic_threshold=critic_threshold,
            seed=seed
        )
        seed_results.append(result)
        improvements.append(result["improvement"])

    # Statistical Aggregation
    n = len(improvements)
    mean_improvement = statistics.mean(improvements)
    std_dev = statistics.stdev(improvements) if n > 1 else 0.0
    standard_error = std_dev / math.sqrt(n) if n > 1 else 0.0

    # 95% Confidence Interval (t_crit approx 1.96 for normal/sample approximation)
    margin_of_error = 1.96 * standard_error
    ci_lower = mean_improvement - margin_of_error
    ci_upper = mean_improvement + margin_of_error

    # Decision Logic based on statistical rigor
    if ci_lower > 0:
        decision = "KEEP"
        decision_reason = "95% Confidence Interval is strictly above zero. Statistically significant positive improvement."
    elif mean_improvement > 0:
        decision = "UNCERTAIN"
        decision_reason = "Mean improvement is positive, but 95% CI crosses zero. Result cannot be distinguished from noise with 95% confidence."
    else:
        decision = "REJECT"
        decision_reason = "Mean improvement is zero or negative. Synthetic augmentation degraded or failed to improve holdout performance."

    # Print Report
    print("\n" + "=" * 65)
    print("MULTI-SEED STATISTICAL SUMMARY")
    print("=" * 65)
    print(f"Seeds Evaluated:       {seeds}")
    print(f"Individual Deltas:     {[f'{x:+.2f} pp' for x in improvements]}")
    print(f"Mean Improvement:      {mean_improvement:+.2f} percentage points")
    print(f"Standard Deviation:    {std_dev:.2f} percentage points")
    print(f"Standard Error:        {standard_error:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:+.2f}, {ci_upper:+.2f}] percentage points")
    print(f"Final Decision:        {decision}")
    print(f"Reason:                {decision_reason}")
    print("=" * 65)

    final_payload = {
        "experiment_type": "multi_seed_validation",
        "model_path": model_path,
        "generation_strategy": generation_strategy,
        "latent_scale": latent_scale,
        "samples_per_minority_class": samples_per_minority_class,
        "critic_threshold": critic_threshold,
        "seeds": seeds,
        "seed_results": seed_results,
        "mean_improvement": mean_improvement,
        "standard_deviation": std_dev,
        "standard_error": standard_error,
        "confidence_interval_95": [ci_lower, ci_upper],
        "decision": decision,
        "decision_reason": decision_reason
    }

    # Save results
    full_out = os.path.join(PROJECT_ROOT, output_path) if not os.path.isabs(output_path) else output_path
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    with open(full_out, "w") as f:
        json.dump(final_payload, f, indent=4)
    print(f"Saved multi-seed results to: {full_out}")

    return final_payload


if __name__ == "__main__":
    run_multi_seed_experiment(
        model_path="models/conditional_vae.pth",
        generation_strategy="LATENT_MANIFOLD",
        latent_scale=0.15,
        samples_per_minority_class=1000,
        seeds=[42, 123, 456]
    )