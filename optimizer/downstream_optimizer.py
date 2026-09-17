import os
import random
import sys
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from optimizer.generation_strategy import GenerationStrategy
from evaluation.classifier import CNN
from evaluation.critic import Critic


CLASS_ID = 5
SAMPLES = 500
CRITIC_THRESHOLD = 0.90
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SCALES = [0.05, 0.10, 0.15, 0.20, 0.30]
SEED = 42
CVAE_PATH = "models/conditional_vae.pth"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_images = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor(train_dataset.targets, dtype=torch.long)

    test_images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor(test_dataset.targets, dtype=torch.long)

    return train_images, train_labels, test_images, test_labels


def train_model(images, labels, device):
    model = CNN().to(device)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, images, labels, device):
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()

    return accuracy


def test_strategy_scale(
    strategy_engine: GenerationStrategy,
    strategy_name: str,
    scale: float,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    baseline_model: nn.Module,
    baseline_accuracy: float,
    device: torch.device
) -> Dict[str, Any]:
    print(f"\n----------------------------------------")
    print(f"Testing {strategy_name} scale: {scale:.2f}")
    print(f"----------------------------------------")

    synth_images, synth_labels = strategy_engine.generate(
        class_id=CLASS_ID,
        number_of_samples=SAMPLES,
        strategy=strategy_name,
        scale=scale
    )

    critic = Critic(baseline_model, device)
    critic_result = critic.evaluate(
        synth_images,
        synth_labels,
        threshold=CRITIC_THRESHOLD
    )

    with torch.no_grad():
        outputs = baseline_model(synth_images.to(device))
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, dim=1)
        mask = (preds == synth_labels.to(device)) & (conf >= CRITIC_THRESHOLD)

    accepted_images = synth_images[mask.cpu()]
    accepted_labels = synth_labels[mask.cpu()]

    print(f"Accepted: {len(accepted_images)} / {SAMPLES} ({critic_result['acceptance_rate']*100:.2f}%)")

    # Augment
    if len(accepted_images) > 0:
        augmented_images = torch.cat([train_images, accepted_images.cpu()], dim=0)
        augmented_labels = torch.cat([train_labels, accepted_labels.cpu()], dim=0)
    else:
        augmented_images = train_images
        augmented_labels = train_labels

    augmented_model = train_model(augmented_images, augmented_labels, device)
    augmented_accuracy = evaluate_model(augmented_model, test_images, test_labels, device)

    improvement = (augmented_accuracy - baseline_accuracy) * 100.0

    print(f"Augmented accuracy: {augmented_accuracy*100:.2f}% | Improvement: {improvement:+.2f} pp")

    return {
        "strategy": strategy_name,
        "scale": scale,
        "accepted": len(accepted_images),
        "acceptance_rate": critic_result["acceptance_rate"],
        "augmented_accuracy": augmented_accuracy,
        "improvement": improvement
    }


def main():
    print("=" * 60)
    print("GENFORGE DOWNSTREAM OPTIMIZER")
    print("=" * 60)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "| Seed:", SEED)

    train_images, train_labels, test_images, test_labels = load_mnist()

    print("\nTraining baseline model on real data...")
    baseline_model = train_model(train_images, train_labels, device)
    baseline_accuracy = evaluate_model(baseline_model, test_images, test_labels, device)
    print(f"Baseline accuracy: {baseline_accuracy * 100:.2f}%")

    strategy_engine = GenerationStrategy(model_path=CVAE_PATH, device=device)

    results = []
    # Test Latent Manifold Scales
    for scale in SCALES:
        res = test_strategy_scale(
            strategy_engine=strategy_engine,
            strategy_name="LATENT_MANIFOLD",
            scale=scale,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            baseline_model=baseline_model,
            baseline_accuracy=baseline_accuracy,
            device=device
        )
        results.append(res)

    best_result = max(results, key=lambda x: x["improvement"])

    print("\n" + "=" * 60)
    print("DOWNSTREAM OPTIMIZER SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"Scale: {r['scale']:.2f} | Acceptance: {r['acceptance_rate']*100:.2f}% | Downstream Improvement: {r['improvement']:+.2f} pp")

    print("\n" + "=" * 60)
    print("OPTIMIZER DECISION")
    print("=" * 60)
    print(f"Best Scale: {best_result['scale']:.2f} with improvement {best_result['improvement']:+.2f} pp")
    if best_result["improvement"] > 0:
        print("Decision: KEEP OPTIMAL STRATEGY")
    else:
        print("Decision: REJECT")


if __name__ == "__main__":
    main()