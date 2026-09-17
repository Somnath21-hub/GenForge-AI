import os
import sys
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

# Allow imports from project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluation.classifier import CNN
from evaluation.critic import Critic
from evaluation.seed import set_seed
from optimizer.generation_strategy import GenerationStrategy


# ============================================================
# DEVICE SETUP
# ============================================================
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TRAIN DOWNSTREAM CNN
# ============================================================
def train_downstream_model(
    train_dataset,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
    seed: int = 42
) -> nn.Module:
    if device is None:
        device = get_device()

    set_seed(seed)
    model = CNN().to(device)

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# ============================================================
# EVALUATE DOWNSTREAM CNN ON TEST HOLDOUT
# ============================================================
def evaluate_downstream_model(
    model: nn.Module,
    test_dataset,
    batch_size: int = 256,
    device: Optional[torch.device] = None
) -> float:
    if device is None:
        device = get_device()

    model.eval()
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100.0 if total > 0 else 0.0


# ============================================================
# CREATE IMBALANCED DEVELOPMENT DATASET
# ============================================================
def get_imbalanced_development_data(
    data_root: str,
    minority_class: int = 5,
    minority_limit: int = 1000,
    transform=None
):
    """
    Creates an imbalanced training dataset where `minority_class`
    is restricted to `minority_limit` samples.
    """
    raw_train = datasets.MNIST(
        root=data_root,
        train=True,
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

    return Subset(raw_train, selected_indices)


# ============================================================
# MAIN DOWNSTREAM EXPERIMENT ROUTINE
# ============================================================
def run_experiment(
    model_path: str = "models/conditional_vae.pth",
    generation_strategy: str = "LATENT_MANIFOLD",
    latent_scale: float = 0.15,
    samples_per_minority_class: int = 1000,
    critic_threshold: float = 0.90,
    downstream_epochs: int = 5,
    minority_class: int = 5,
    minority_limit: int = 1000,
    seed: int = 42,
    data_root: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a clean, reproducible downstream ML experiment:
    1. Sets deterministic seed.
    2. Constructs imbalanced development dataset (e.g. class 5 restricted).
    3. Loads untouched 10,000-sample test holdout.
    4. Trains baseline classifier on imbalanced development set.
    5. Evaluates baseline classifier on untouched test holdout.
    6. Generates synthetic samples using requested strategy & scale.
    7. Evaluates synthetic quality with Critic.
    8. Filters accepted synthetic samples and adds to development set.
    9. Trains augmented classifier on the augmented development set.
    10. Evaluates augmented classifier on the EXACT SAME untouched holdout.
    11. Computes downstream improvement = augmented_accuracy - baseline_accuracy.
    """
    set_seed(seed)
    device = get_device()
    data_dir = data_root or os.path.join(PROJECT_ROOT, "data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("\n" + "=" * 60)
    print("GENFORGE DOWNSTREAM SYNTHETIC EXPERIMENT")
    print("=" * 60)
    print(f"Seed: {seed} | Device: {device}")
    print(f"Generator Model: {model_path}")
    print(f"Strategy: {generation_strategy} | Latent Scale: {latent_scale}")
    print(f"Imbalanced Class: {minority_class} (Limit: {minority_limit})")
    print(f"Synthetic Samples Requested: {samples_per_minority_class} | Critic Threshold: {critic_threshold}")

    # 1. Datasets
    dev_dataset = get_imbalanced_development_data(
        data_root=data_dir,
        minority_class=minority_class,
        minority_limit=minority_limit,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print(f"Imbalanced Development Samples: {len(dev_dataset)}")
    print(f"Untouched Holdout Test Samples: {len(test_dataset)}")

    # 2. Train Baseline Classifier
    print("\n[Step 1] Training Baseline Downstream Model...")
    baseline_model = train_downstream_model(
        dev_dataset,
        epochs=downstream_epochs,
        device=device,
        seed=seed
    )

    baseline_accuracy = evaluate_downstream_model(
        baseline_model,
        test_dataset,
        device=device
    )
    print(f"Baseline Holdout Accuracy: {baseline_accuracy:.2f}%")

    # 3. Generate Synthetic Data
    print(f"\n[Step 2] Generating Synthetic Data via {generation_strategy} (scale={latent_scale})...")
    generator = GenerationStrategy(
        model_path=model_path,
        device=device
    )

    synth_images, synth_labels = generator.generate(
        class_id=minority_class,
        number_of_samples=samples_per_minority_class,
        strategy=generation_strategy,
        scale=latent_scale
    )

    # 4. Critic Evaluation & Filtering
    print("\n[Step 3] Evaluating Synthetic Data with Critic...")
    critic = Critic(
        classifier=baseline_model,
        device=device
    )

    critic_result = critic.evaluate(
        synth_images,
        synth_labels,
        threshold=critic_threshold
    )

    accepted_count = critic_result["accepted"]
    acceptance_rate = critic_result["acceptance_rate"]
    class_accuracy = critic_result["class_accuracy"]
    diversity_score = critic_result["diversity_score"]
    duplicate_rate = critic_result["duplicate_rate"]

    print(f"Critic Generated: {len(synth_images)} | Accepted: {accepted_count} ({acceptance_rate * 100:.2f}%)")
    print(f"Synthetic Accuracy: {class_accuracy * 100:.2f}% | Diversity: {diversity_score:.4f} | Duplicate Rate: {duplicate_rate * 100:.2f}%")

    # Filter accepted tensors
    baseline_model.eval()
    with torch.no_grad():
        outputs = baseline_model(synth_images.to(device))
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, dim=1)
        mask = (preds == synth_labels.to(device)) & (conf >= critic_threshold)

    accepted_images = synth_images[mask.cpu()]
    accepted_labels = synth_labels[mask.cpu()]

    # 5. Augment Development Dataset
    print("\n[Step 4] Assembling Augmented Dataset...")
    dev_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset), shuffle=False)
    real_images, real_labels = next(iter(dev_loader))

    if len(accepted_images) > 0:
        augmented_images = torch.cat([real_images, accepted_images.cpu()], dim=0)
        augmented_labels = torch.cat([real_labels, accepted_labels.cpu()], dim=0)
    else:
        augmented_images = real_images
        augmented_labels = real_labels

    augmented_dataset = TensorDataset(augmented_images, augmented_labels)
    print(f"Real Samples: {len(real_images)} | Synthetic Added: {len(accepted_images)} | Total: {len(augmented_dataset)}")

    # 6. Train Augmented Model
    print("\n[Step 5] Training Augmented Downstream Model...")
    augmented_model = train_downstream_model(
        augmented_dataset,
        epochs=downstream_epochs,
        device=device,
        seed=seed
    )

    augmented_accuracy = evaluate_downstream_model(
        augmented_model,
        test_dataset,
        device=device
    )
    print(f"Augmented Holdout Accuracy: {augmented_accuracy:.2f}%")

    # 7. Compute Improvement
    improvement = augmented_accuracy - baseline_accuracy
    print("\n" + "=" * 60)
    print("EXPERIMENT OUTCOME")
    print(f"Baseline Accuracy:  {baseline_accuracy:.2f}%")
    print(f"Augmented Accuracy: {augmented_accuracy:.2f}%")
    print(f"Net Improvement:    {improvement:+.2f} percentage points")
    print("=" * 60)

    return {
        "seed": seed,
        "model_path": model_path,
        "generation_strategy": generation_strategy,
        "latent_scale": latent_scale,
        "samples_requested": samples_per_minority_class,
        "accepted_samples": accepted_count,
        "acceptance_rate": acceptance_rate,
        "critic_accuracy": class_accuracy * 100.0,
        "diversity_score": diversity_score,
        "duplicate_rate": duplicate_rate,
        "baseline_accuracy": baseline_accuracy,
        "augmented_accuracy": augmented_accuracy,
        "improvement": improvement
    }


if __name__ == "__main__":
    res = run_experiment(
        model_path="models/conditional_vae.pth",
        generation_strategy="LATENT_MANIFOLD",
        latent_scale=0.15,
        samples_per_minority_class=1000,
        seed=42
    )
    print("\nResult summary:")
    print(res)