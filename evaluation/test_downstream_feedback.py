import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from optimizer.generation_strategy import GenerationStrategy
from evaluation.classifier import CNN
from evaluation.critic import Critic


# ==========================================
# DEVICE
# ==========================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("Using device:", device)


# ==========================================
# LOAD MNIST
# ==========================================

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


# ==========================================
# CONVERT DATASET TO TENSORS
# ==========================================

real_images = train_dataset.data.float() / 255.0
real_images = (real_images - 0.5) / 0.5

real_images = real_images.unsqueeze(1)

real_labels = train_dataset.targets

test_images = test_dataset.data.float() / 255.0
test_images = (test_images - 0.5) / 0.5

test_images = test_images.unsqueeze(1)

test_labels = test_dataset.targets


# ==========================================
# TEST FUNCTION
# ==========================================

def evaluate_model(model):

    model.eval()

    correct = 0
    total = 0

    test_loader = DataLoader(
        TensorDataset(
            test_images,
            test_labels
        ),
        batch_size=256,
        shuffle=False
    )

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


# ==========================================
# TRAIN FUNCTION
# ==========================================

def train_model(
    images,
    labels,
    epochs=5
):

    dataset = TensorDataset(
        images,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    model = CNN().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for batch_images, batch_labels in loader:

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                batch_images
            )

            loss = criterion(
                outputs,
                batch_labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {total_loss / len(loader):.4f}"
        )

    return model


# ==========================================
# MAIN EXPERIMENT
# ==========================================

print()
print("==========================================")
print("     DOWNSTREAM ML FEEDBACK EXPERIMENT")
print("==========================================")


# ==========================================
# STEP 1 — REAL ONLY BASELINE
# ==========================================

print()
print("------------------------------------------")
print("STEP 1: REAL-ONLY BASELINE")
print("------------------------------------------")

baseline_model = train_model(
    real_images,
    real_labels,
    epochs=5
)

baseline_accuracy = evaluate_model(
    baseline_model
)

print()
print(
    "Baseline accuracy:",
    f"{baseline_accuracy * 100:.2f}%"
)


# ==========================================
# STEP 2 — GENERATE SYNTHETIC DATA
# ==========================================

print()
print("------------------------------------------")
print("STEP 2: GENERATE SYNTHETIC DATA")
print("------------------------------------------")

generator = GenerationStrategy(
    model_path="models/conditional_vae.pth",
    latent_size=32,
    num_classes=10
)


class_id = 5
number_of_samples = 500
best_scale = 0.5


synthetic_images, synthetic_labels = (
    generator.adapt_latent(
        class_id,
        number_of_samples,
        scale=best_scale
    )
)

print(
    "Synthetic images:",
    synthetic_images.shape
)

print(
    "Synthetic class:",
    class_id
)

print(
    "Latent scale:",
    best_scale
)


# ==========================================
# STEP 3 — CRITIC FILTER
# ==========================================

print()
print("------------------------------------------")
print("STEP 3: CRITIC FILTER")
print("------------------------------------------")

critic_classifier = CNN().to(device)

critic_classifier.load_state_dict(
    torch.load(
        "evaluation/baseline_cnn.pth",
        map_location=device
    )
)

critic_classifier.eval()

critic = Critic(
    classifier=critic_classifier,
    device=device
)

critic_result = critic.evaluate(
    synthetic_images,
    synthetic_labels,
    threshold=0.90
)

print(
    "Generated:",
    critic_result["total"]
)

print(
    "Accepted:",
    critic_result["accepted"]
)

print(
    "Acceptance:",
    f"{critic_result['acceptance_rate'] * 100:.2f}%"
)


# ==========================================
# STEP 4 — KEEP ONLY ACCEPTED DATA
# ==========================================

with torch.no_grad():

    outputs = critic_classifier(
        synthetic_images.to(device)
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predictions = torch.max(
        probabilities,
        dim=1
    )

    accepted_mask = (
        (predictions == synthetic_labels.to(device))
        &
        (confidence >= 0.90)
    )


accepted_images = synthetic_images[
    accepted_mask.cpu()
]

accepted_labels = synthetic_labels[
    accepted_mask.cpu()
]


print()
print(
    "Accepted synthetic samples:",
    len(accepted_images)
)


# ==========================================
# STEP 5 — AUGMENT REAL DATA
# ==========================================

print()
print("------------------------------------------")
print("STEP 4: AUGMENT REAL DATA")
print("------------------------------------------")

augmented_images = torch.cat(
    [
        real_images,
        accepted_images.cpu()
    ],
    dim=0
)

augmented_labels = torch.cat(
    [
        real_labels,
        accepted_labels.cpu()
    ],
    dim=0
)

print(
    "Real samples:",
    len(real_images)
)

print(
    "Synthetic accepted:",
    len(accepted_images)
)

print(
    "Total training samples:",
    len(augmented_images)
)


# ==========================================
# STEP 6 — TRAIN AUGMENTED MODEL
# ==========================================

print()
print("------------------------------------------")
print("STEP 5: TRAIN AUGMENTED MODEL")
print("------------------------------------------")

augmented_model = train_model(
    augmented_images,
    augmented_labels,
    epochs=5
)

augmented_accuracy = evaluate_model(
    augmented_model
)


# ==========================================
# STEP 7 — FINAL COMPARISON
# ==========================================

improvement = (
    augmented_accuracy
    - baseline_accuracy
)

print()
print("==========================================")
print("          FINAL COMPARISON")
print("==========================================")

print()
print(
    "Real-only accuracy:",
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    "Augmented accuracy:",
    f"{augmented_accuracy * 100:.2f}%"
)

print(
    "Improvement:",
    f"{improvement * 100:.2f} pp"
)


# ==========================================
# GENFORGE DECISION
# ==========================================

print()
print("==========================================")
print("         GENFORGE DECISION")
print("==========================================")

if improvement > 0:

    print()
    print("Decision: KEEP SYNTHETIC DATA")

    print(
        "Synthetic data improved "
        "downstream performance."
    )

else:

    print()
    print("Decision: REJECT SYNTHETIC DATA")

    print(
        "Synthetic data did not improve "
        "downstream performance."
    )