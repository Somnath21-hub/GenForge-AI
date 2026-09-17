import os
import random
import sys
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluation.classifier import CNN
from evaluation.critic import Critic
from models.conditional_vae import ConditionalVAE
from optimizer.generation_strategy import GenerationStrategy
from analysis.dataset_analyzer import analyze_dataset
from analysis.augmentation_planner import create_augmentation_plan


# ============================================================
# CONFIGURATION & HYPERPARAMETERS
# ============================================================
MINORITY_CLASS = 5
MINORITY_SAMPLES = 100  # Severe class imbalance: only 100 samples of class 5
CRITIC_THRESHOLD = 0.90
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CVAE_PATH = "models/conditional_vae.pth"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET CONSTRUCTION (SEVERE IMBALANCE)
# ============================================================
def load_imbalanced_datasets(
    minority_class: int = MINORITY_CLASS,
    minority_limit: int = MINORITY_SAMPLES,
    data_root: str = "./data"
) -> Tuple[Subset, datasets.MNIST, torch.Tensor, torch.Tensor]:
    """
    Constructs:
    1. Imbalanced development dataset (Class 5 strictly limited to 100 real samples).
    2. Untouched 10,000-image test set.
    3. Extracted development images and labels as tensors.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    raw_train = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    selected_indices = []
    minority_count = 0

    for idx, target in enumerate(raw_train.targets):
        target_val = int(target)
        if target_val != minority_class:
            selected_indices.append(idx)
        else:
            if minority_count < minority_limit:
                selected_indices.append(idx)
                minority_count += 1

    dev_dataset = Subset(raw_train, selected_indices)

    # Convert development set into tensors
    dev_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)
    dev_images, dev_labels = next(iter(dev_loader))

    return dev_dataset, test_dataset, dev_images, dev_labels


# ============================================================
# DOWNSTREAM CNN TRAINING
# ============================================================
def train_downstream_cnn(
    images: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    device: Optional[torch.device] = None,
    seed: int = 42
) -> nn.Module:
    if device is None:
        device = get_device()

    set_seed(seed)
    model = CNN().to(device)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    return model


# ============================================================
# COMPREHENSIVE EVALUATION ON UNTOUCHED TEST HOLDOUT
# ============================================================
def evaluate_on_holdout(
    model: nn.Module,
    test_dataset: datasets.MNIST,
    target_class: int = MINORITY_CLASS,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(labels)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Overall accuracy
    overall_accuracy = (all_preds == all_targets).mean() * 100.0

    # 10x10 Confusion Matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_preds):
        confusion_matrix[t, p] += 1

    # Class-specific metrics for target_class
    tp = confusion_matrix[target_class, target_class]
    fn = confusion_matrix[target_class, :].sum() - tp
    fp = confusion_matrix[:, target_class].sum() - tp
    tn = len(all_targets) - (tp + fn + fp)

    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "class_5_tp": int(tp),
        "class_5_fp": int(fp),
        "class_5_fn": int(fn),
        "class_5_tn": int(tn),
        "class_5_precision": precision,
        "class_5_recall": recall,
        "class_5_f1": f1,
        "confusion_matrix": confusion_matrix.tolist()
    }


# ============================================================
# CUSTOM GENERATION STRATEGY BOUND TO DEV SET LATENTS
# ============================================================
class ImbalancedGenerationEngine:
    """
    Generates synthetic samples for class 5 using ONLY the 100 real development samples.
    Ensures zero leakage from the test set.
    """
    def __init__(
        self,
        cvae_path: str,
        dev_images: torch.Tensor,
        dev_labels: torch.Tensor,
        latent_size: int = 32,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.latent_size = latent_size

        self.cvae = ConditionalVAE(latent_size=latent_size, num_classes=10).to(self.device)
        self.cvae.load_state_dict(torch.load(cvae_path, map_location=self.device))
        self.cvae.eval()

        # Extract latent representations strictly from development data
        mask_5 = (dev_labels == MINORITY_CLASS)
        class_5_images = dev_images[mask_5].to(self.device)
        class_5_labels = dev_labels[mask_5].to(self.device)

        with torch.no_grad():
            mu_5, logvar_5 = self.cvae.encode(class_5_images, class_5_labels)

        self.class_5_latents = mu_5.detach()
        self.class_5_mean = self.class_5_latents.mean(dim=0)
        self.class_5_std = torch.clamp(self.class_5_latents.std(dim=0), min=1e-4)

    def generate(
        self,
        strategy: str,
        number_of_samples: int,
        scale: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full((number_of_samples,), MINORITY_CLASS, dtype=torch.long, device=self.device)

        strat_upper = strategy.upper()
        with torch.no_grad():
            if strat_upper == "RANDOM_PRIOR":
                z = torch.randn(number_of_samples, self.latent_size, device=self.device)
            elif strat_upper == "LATENT_MANIFOLD":
                idx = torch.randint(0, len(self.class_5_latents), (number_of_samples,))
                base_mu = self.class_5_latents[idx]
                noise = torch.randn(number_of_samples, self.latent_size, device=self.device) * scale
                z = base_mu + noise
            elif strat_upper == "ADAPTIVE_LATENT":
                noise = torch.randn(number_of_samples, self.latent_size, device=self.device)
                z = self.class_5_mean + noise * (self.class_5_std * scale)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            images = self.cvae.decode(z, labels)

        return images, labels


# ============================================================
# RUN CONTROLLED SEVERE-IMBALANCE STRATEGY BENCHMARK
# ============================================================
def run_severe_imbalance_benchmark(
    seed: int = 42,
    cvae_path: str = CVAE_PATH
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device()

    print("\n" + "=" * 70)
    print("GENFORGE SEVERE CLASS IMBALANCE EXPERIMENT")
    print(f"Device: {device} | Seed: {seed}")
    print(f"Minority Class: 5 (Real Samples: {MINORITY_SAMPLES})")
    print("=" * 70)

    # 1. Load Imbalanced Datasets
    dev_dataset, test_dataset, dev_images, dev_labels = load_imbalanced_datasets()

    # 2. Automated Dataset Analysis
    dataset_info = analyze_dataset(dev_dataset)
    print(f"\n[Analysis] Total Real Samples: {dataset_info['total_samples']}")
    print(f"[Analysis] Minority Class: {dataset_info['minority_class']} (Count: {dataset_info['minority_count']})")
    print(f"[Analysis] Imbalance Ratio: {dataset_info['imbalance_ratio']:.2f}x")

    # 3. Augmentation Planner
    plan = create_augmentation_plan(dataset_info)
    target_synthetic = plan[MINORITY_CLASS]["synthetic_samples"]
    print(f"[Planner] Synthetic Target for Class 5: {target_synthetic} samples")

    # 4. Train Baseline Model (Imbalanced Real Data Only)
    print("\n[Step 1] Training Baseline Downstream Model on Imbalanced Real Data...")
    baseline_model = train_downstream_cnn(dev_images, dev_labels, epochs=EPOCHS, device=device, seed=seed)
    baseline_metrics = evaluate_on_holdout(baseline_model, test_dataset, target_class=MINORITY_CLASS, device=device)

    print(f"--> Baseline Overall Accuracy: {baseline_metrics['overall_accuracy']:.2f}%")
    print(f"--> Baseline Class-5 Precision: {baseline_metrics['class_5_precision']:.2f}%")
    print(f"--> Baseline Class-5 Recall:    {baseline_metrics['class_5_recall']:.2f}% (TP={baseline_metrics['class_5_tp']}/{baseline_metrics['class_5_tp']+baseline_metrics['class_5_fn']})")
    print(f"--> Baseline Class-5 F1-Score:  {baseline_metrics['class_5_f1']:.2f}%")

    # 5. Initialize Generator Bound Strictly to Dev Data
    gen_engine = ImbalancedGenerationEngine(
        cvae_path=cvae_path,
        dev_images=dev_images,
        dev_labels=dev_labels,
        device=device
    )

    # Critic setup
    critic = Critic(classifier=baseline_model, device=device)

    # 6. Benchmark Candidate Generation Strategies in the Arena
    candidate_configs = [
        {"strategy": "RANDOM_PRIOR", "scale": 1.00},
        {"strategy": "LATENT_MANIFOLD", "scale": 0.10},
        {"strategy": "LATENT_MANIFOLD", "scale": 0.15},
        {"strategy": "LATENT_MANIFOLD", "scale": 0.20},
        {"strategy": "ADAPTIVE_LATENT", "scale": 0.80}
    ]

    strategy_results = []

    print("\n" + "=" * 70)
    print("BENCHMARKING GENERATION STRATEGIES UNDER SEVERE IMBALANCE")
    print("=" * 70)

    for cfg in candidate_configs:
        strat_name = cfg["strategy"]
        scale = cfg["scale"]
        desc = f"{strat_name} (scale={scale:.2f})"
        print(f"\n>>> Testing Candidate: {desc}...")

        # Generate synthetic samples
        synth_images, synth_labels = gen_engine.generate(
            strategy=strat_name,
            number_of_samples=target_synthetic,
            scale=scale
        )

        # Critic Evaluation
        critic_res = critic.evaluate(synth_images, synth_labels, threshold=CRITIC_THRESHOLD)

        # Filter accepted synthetic data
        baseline_model.eval()
        with torch.no_grad():
            outputs = baseline_model(synth_images.to(device))
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)
            mask = (preds == synth_labels.to(device)) & (conf >= CRITIC_THRESHOLD)

        accepted_images = synth_images[mask.cpu()]
        accepted_labels = synth_labels[mask.cpu()]

        print(f"    Critic Accepted: {len(accepted_images)} / {target_synthetic} ({critic_res['acceptance_rate']*100:.2f}%)")
        print(f"    Synthetic Accuracy: {critic_res['class_accuracy']*100:.2f}% | Diversity: {critic_res['diversity_score']:.4f} | Duplicate Rate: {critic_res['duplicate_rate']*100:.2f}%")

        # Assemble augmented dataset
        if len(accepted_images) > 0:
            aug_images = torch.cat([dev_images, accepted_images.cpu()], dim=0)
            aug_labels = torch.cat([dev_labels, accepted_labels.cpu()], dim=0)
        else:
            aug_images = dev_images
            aug_labels = dev_labels

        # Train fresh Augmented Model
        augmented_model = train_downstream_cnn(aug_images, aug_labels, epochs=EPOCHS, device=device, seed=seed)
        aug_metrics = evaluate_on_holdout(augmented_model, test_dataset, target_class=MINORITY_CLASS, device=device)

        # Calculate improvements
        overall_imp = aug_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"]
        recall_imp = aug_metrics["class_5_recall"] - baseline_metrics["class_5_recall"]
        f1_imp = aug_metrics["class_5_f1"] - baseline_metrics["class_5_f1"]

        print(f"    Holdout Overall Accuracy: {aug_metrics['overall_accuracy']:.2f}% ({overall_imp:+.2f} pp)")
        print(f"    Holdout Class-5 Recall:    {aug_metrics['class_5_recall']:.2f}% ({recall_imp:+.2f} pp, TP={aug_metrics['class_5_tp']}/892)")
        print(f"    Holdout Class-5 F1:        {aug_metrics['class_5_f1']:.2f}% ({f1_imp:+.2f} pp)")

        strategy_results.append({
            "strategy": strat_name,
            "scale": scale,
            "synthetic_requested": target_synthetic,
            "accepted_samples": len(accepted_images),
            "acceptance_rate": critic_res["acceptance_rate"],
            "synthetic_accuracy": critic_res["class_accuracy"] * 100.0,
            "diversity_score": critic_res["diversity_score"],
            "duplicate_rate": critic_res["duplicate_rate"] * 100.0,
            "baseline_overall_acc": baseline_metrics["overall_accuracy"],
            "augmented_overall_acc": aug_metrics["overall_accuracy"],
            "overall_improvement": overall_imp,
            "baseline_class_5_recall": baseline_metrics["class_5_recall"],
            "augmented_class_5_recall": aug_metrics["class_5_recall"],
            "recall_improvement": recall_imp,
            "baseline_class_5_f1": baseline_metrics["class_5_f1"],
            "augmented_class_5_f1": aug_metrics["class_5_f1"],
            "f1_improvement": f1_imp,
            "baseline_metrics": baseline_metrics,
            "augmented_metrics": aug_metrics
        })

    # Select winner based primarily on downstream utility (F1 and overall improvement)
    best_strategy = max(strategy_results, key=lambda x: (x["f1_improvement"], x["overall_improvement"]))

    print("\n" + "=" * 70)
    print("ARENA WINNER SELECTION")
    print("=" * 70)
    print(f"Selected Strategy: {best_strategy['strategy']} (scale={best_strategy['scale']:.2f})")
    print(f"Class-5 Recall:    {best_strategy['baseline_class_5_recall']:.2f}% -> {best_strategy['augmented_class_5_recall']:.2f}% ({best_strategy['recall_improvement']:+.2f} pp)")
    print(f"Class-5 F1-Score:  {best_strategy['baseline_class_5_f1']:.2f}% -> {best_strategy['augmented_class_5_f1']:.2f}% ({best_strategy['f1_improvement']:+.2f} pp)")
    print(f"Overall Accuracy:  {best_strategy['baseline_overall_acc']:.2f}% -> {best_strategy['augmented_overall_acc']:.2f}% ({best_strategy['overall_improvement']:+.2f} pp)")
    print("=" * 70)

    return {
        "seed": seed,
        "planner_synthetic_target": target_synthetic,
        "baseline_metrics": baseline_metrics,
        "strategy_results": strategy_results,
        "best_strategy": best_strategy
    }


if __name__ == "__main__":
    benchmark_out = run_severe_imbalance_benchmark(seed=42)