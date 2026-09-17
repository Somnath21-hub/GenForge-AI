import json
import math
import os
import statistics
import sys
from typing import Dict, Any, List, Optional

import torch

# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluation.test_imbalanced_downstream import (
    load_imbalanced_datasets,
    train_downstream_cnn,
    evaluate_on_holdout,
    ImbalancedGenerationEngine,
    MINORITY_CLASS,
    MINORITY_SAMPLES,
    CRITIC_THRESHOLD,
    EPOCHS,
    CVAE_PATH,
    set_seed,
    get_device
)
from evaluation.critic import Critic
from analysis.dataset_analyzer import analyze_dataset
from analysis.augmentation_planner import create_augmentation_plan
from optimizer.experiment_history import ExperimentHistory


DEFAULT_SEEDS = [42, 123, 456]


def run_severe_imbalance_multiseed(
    strategy: str = "LATENT_MANIFOLD",
    scale: float = 0.15,
    seeds: Optional[List[int]] = None,
    output_path: str = "experiments/imbalanced_severe_results.json"
) -> Dict[str, Any]:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    device = get_device()
    print("\n" + "=" * 70)
    print("GENFORGE SEVERE IMBALANCE MULTI-SEED RIGOROUS VALIDATION")
    print(f"Device: {device} | Seeds: {seeds}")
    print(f"Minority Class: 5 (Real Samples: {MINORITY_SAMPLES})")
    print(f"Selected Strategy: {strategy} (scale={scale:.2f})")
    print("=" * 70)

    # 1. Dataset & Analysis
    dev_dataset, test_dataset, dev_images, dev_labels = load_imbalanced_datasets()
    dataset_info = analyze_dataset(dev_dataset)
    plan = create_augmentation_plan(dataset_info)
    target_synthetic = plan[MINORITY_CLASS]["synthetic_samples"]

    # 2. Generator bound to dev data
    gen_engine = ImbalancedGenerationEngine(
        cvae_path=CVAE_PATH,
        dev_images=dev_images,
        dev_labels=dev_labels,
        device=device
    )

    seed_runs = []
    overall_improvements = []
    recall_improvements = []
    f1_improvements = []

    for seed in seeds:
        print(f"\n========================================")
        print(f"       RUNNING SEED {seed}")
        print(f"========================================")
        set_seed(seed)

        # Baseline Model
        baseline_model = train_downstream_cnn(dev_images, dev_labels, epochs=EPOCHS, device=device, seed=seed)
        baseline_metrics = evaluate_on_holdout(baseline_model, test_dataset, target_class=MINORITY_CLASS, device=device)

        # Generate synthetic samples
        synth_images, synth_labels = gen_engine.generate(
            strategy=strategy,
            number_of_samples=target_synthetic,
            scale=scale
        )

        # Critic Filter
        critic = Critic(classifier=baseline_model, device=device)
        critic_res = critic.evaluate(synth_images, synth_labels, threshold=CRITIC_THRESHOLD)

        baseline_model.eval()
        with torch.no_grad():
            outputs = baseline_model(synth_images.to(device))
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)
            mask = (preds == synth_labels.to(device)) & (conf >= CRITIC_THRESHOLD)

        accepted_images = synth_images[mask.cpu()]
        accepted_labels = synth_labels[mask.cpu()]

        # Augmented Model
        if len(accepted_images) > 0:
            aug_images = torch.cat([dev_images, accepted_images.cpu()], dim=0)
            aug_labels = torch.cat([dev_labels, accepted_labels.cpu()], dim=0)
        else:
            aug_images = dev_images
            aug_labels = dev_labels

        augmented_model = train_downstream_cnn(aug_images, aug_labels, epochs=EPOCHS, device=device, seed=seed)
        aug_metrics = evaluate_on_holdout(augmented_model, test_dataset, target_class=MINORITY_CLASS, device=device)

        overall_delta = aug_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"]
        recall_delta = aug_metrics["class_5_recall"] - baseline_metrics["class_5_recall"]
        f1_delta = aug_metrics["class_5_f1"] - baseline_metrics["class_5_f1"]

        overall_improvements.append(overall_delta)
        recall_improvements.append(recall_delta)
        f1_improvements.append(f1_delta)

        print(f"Seed {seed} -> Overall Acc: {baseline_metrics['overall_accuracy']:.2f}% -> {aug_metrics['overall_accuracy']:.2f}% ({overall_delta:+.2f} pp)")
        print(f"Seed {seed} -> Class-5 Recall: {baseline_metrics['class_5_recall']:.2f}% -> {aug_metrics['class_5_recall']:.2f}% ({recall_delta:+.2f} pp)")
        print(f"Seed {seed} -> Class-5 F1:     {baseline_metrics['class_5_f1']:.2f}% -> {aug_metrics['class_5_f1']:.2f}% ({f1_delta:+.2f} pp)")

        seed_runs.append({
            "seed": seed,
            "baseline_overall": baseline_metrics["overall_accuracy"],
            "augmented_overall": aug_metrics["overall_accuracy"],
            "overall_improvement": overall_delta,
            "baseline_recall": baseline_metrics["class_5_recall"],
            "augmented_recall": aug_metrics["class_5_recall"],
            "recall_improvement": recall_delta,
            "baseline_f1": baseline_metrics["class_5_f1"],
            "augmented_f1": aug_metrics["class_5_f1"],
            "f1_improvement": f1_delta,
            "accepted_samples": len(accepted_images),
            "acceptance_rate": critic_res["acceptance_rate"],
            "synthetic_accuracy": critic_res["class_accuracy"] * 100.0,
            "baseline_metrics": baseline_metrics,
            "augmented_metrics": aug_metrics
        })

    # Statistical Aggregation
    n = len(seeds)
    mean_overall_imp = statistics.mean(overall_improvements)
    std_overall = statistics.stdev(overall_improvements) if n > 1 else 0.0
    se_overall = std_overall / math.sqrt(n) if n > 1 else 0.0
    ci_overall = [mean_overall_imp - 1.96 * se_overall, mean_overall_imp + 1.96 * se_overall]

    mean_recall_imp = statistics.mean(recall_improvements)
    std_recall = statistics.stdev(recall_improvements) if n > 1 else 0.0
    se_recall = std_recall / math.sqrt(n) if n > 1 else 0.0
    ci_recall = [mean_recall_imp - 1.96 * se_recall, mean_recall_imp + 1.96 * se_recall]

    mean_f1_imp = statistics.mean(f1_improvements)
    std_f1 = statistics.stdev(f1_improvements) if n > 1 else 0.0
    se_f1 = std_f1 / math.sqrt(n) if n > 1 else 0.0
    ci_f1 = [mean_f1_imp - 1.96 * se_f1, mean_f1_imp + 1.96 * se_f1]

    if ci_f1[0] > 0 or ci_recall[0] > 0:
        decision = "KEEP"
        decision_reason = "Statistically significant positive improvement in minority class performance (95% CI strictly positive)."
    elif mean_f1_imp > 0 or mean_recall_imp > 0:
        decision = "UNCERTAIN"
        decision_reason = "Mean improvement is positive, but confidence interval crosses zero."
    else:
        decision = "REJECT"
        decision_reason = "Synthetic augmentation did not improve minority class performance."

    print("\n" + "=" * 70)
    print("SEVERE IMBALANCE MULTI-SEED STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"Strategy:              {strategy} (scale={scale:.2f})")
    print(f"Seeds:                 {seeds}")
    print(f"Overall Acc Deltas:    {[f'{x:+.2f} pp' for x in overall_improvements]}")
    print(f"Mean Overall Delta:    {mean_overall_imp:+.2f} pp (95% CI: [{ci_overall[0]:+.2f}, {ci_overall[1]:+.2f}])")
    print(f"Class-5 Recall Deltas: {[f'{x:+.2f} pp' for x in recall_improvements]}")
    print(f"Mean Class-5 Recall:   {mean_recall_imp:+.2f} pp (95% CI: [{ci_recall[0]:+.2f}, {ci_recall[1]:+.2f}])")
    print(f"Class-5 F1 Deltas:     {[f'{x:+.2f} pp' for x in f1_improvements]}")
    print(f"Mean Class-5 F1:       {mean_f1_imp:+.2f} pp (95% CI: [{ci_f1[0]:+.2f}, {ci_f1[1]:+.2f}])")
    print(f"Final Decision:        {decision}")
    print(f"Reason:                {decision_reason}")
    print("=" * 70)

    summary_payload = {
        "experiment_type": "severe_class_imbalance_validation",
        "minority_class": MINORITY_CLASS,
        "minority_real_samples": MINORITY_SAMPLES,
        "strategy": strategy,
        "scale": scale,
        "synthetic_target": target_synthetic,
        "seeds": seeds,
        "seed_runs": seed_runs,
        "overall_mean_improvement": mean_overall_imp,
        "overall_std": std_overall,
        "overall_se": se_overall,
        "overall_95_ci": ci_overall,
        "recall_mean_improvement": mean_recall_imp,
        "recall_std": std_recall,
        "recall_se": se_recall,
        "recall_95_ci": ci_recall,
        "f1_mean_improvement": mean_f1_imp,
        "f1_std": std_f1,
        "f1_se": se_f1,
        "f1_95_ci": ci_f1,
        "decision": decision,
        "decision_reason": decision_reason
    }

    # Save to imbalanced_severe_results.json
    full_out = os.path.join(PROJECT_ROOT, output_path) if not os.path.isabs(output_path) else output_path
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    with open(full_out, "w") as f:
        json.dump(summary_payload, f, indent=4)

    # Log to history.json
    history_manager = ExperimentHistory()
    experiment_id = history_manager.create_experiment({
        "experiment_type": "severe_class_imbalance_validation",
        "minority_class": MINORITY_CLASS,
        "minority_samples": MINORITY_SAMPLES,
        "strategy": strategy,
        "scale": scale,
        "synthetic_target": target_synthetic,
        "seeds": seeds
    })

    history_manager.save_iteration(experiment_id, {
        "iteration": 1,
        "decision": decision,
        "mean_overall_improvement": mean_overall_imp,
        "mean_recall_improvement": mean_recall_imp,
        "mean_f1_improvement": mean_f1_imp,
        "ci_recall": ci_recall,
        "ci_f1": ci_f1,
        "summary": summary_payload
    })

    print(f"Results recorded to {full_out} and history.json (Experiment ID: {experiment_id})")

    return summary_payload


if __name__ == "__main__":
    run_severe_imbalance_multiseed()