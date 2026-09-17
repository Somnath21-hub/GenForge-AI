import sys
import os

# ==========================================
# PROJECT ROOT
# ==========================================

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

from optimizer.auto_strategy import AutoStrategy
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
# CONVERT DATA TO TENSORS
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
# TRAIN DOWNSTREAM CNN
# ==========================================

def train_model(images, labels, epochs=5):

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
# EVALUATE CNN
# ==========================================

def evaluate_model(model):

    model.eval()

    loader = DataLoader(
        TensorDataset(
            test_images,
            test_labels
        ),
        batch_size=256,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

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
# GENFORGE CONTROLLER
# ==========================================

class GenForgeController:

    def __init__(self):

        print()
        print("==========================================")
        print("          GENFORGE CONTROLLER")
        print("==========================================")

        self.auto_strategy = AutoStrategy()


    # ======================================
    # RUN COMPLETE PIPELINE
    # ======================================

    def run(
        self,
        class_id=5,
        number_of_samples=500,
        threshold=0.90
    ):

        print()
        print("==========================================")
        print("       GENFORGE AUTONOMOUS PIPELINE")
        print("==========================================")


        # ==================================
        # STEP 1
        # STRATEGY SEARCH
        # ==================================

        print()
        print("------------------------------------------")
        print("STEP 1: FIND BEST GENERATION STRATEGY")
        print("------------------------------------------")

        strategy_result = (
            self.auto_strategy.optimize_class(
                class_id=class_id,
                number_of_samples=100,
                threshold=threshold
            )
        )

        strategy = strategy_result["strategy"]

        if strategy == "REJECT":

            print()
            print(
                "No generation strategy improved "
                "the critic score."
            )

            return


        best_scale = strategy_result.get(
            "scale",
            1.0
        )

        print()
        print(
            "Selected strategy:",
            strategy
        )

        print(
            "Selected scale:",
            best_scale
        )


        # ==================================
        # STEP 2
        # GENERATE FINAL DATA
        # ==================================

        print()
        print("------------------------------------------")
        print("STEP 2: GENERATE FINAL SYNTHETIC DATA")
        print("------------------------------------------")

        if strategy == "ADAPT_LATENT":

            synthetic_images, synthetic_labels = (
                self.auto_strategy.generator.adapt_latent(
                    class_id,
                    number_of_samples,
                    scale=best_scale
                )
            )

        else:

            synthetic_images, synthetic_labels = (
                self.auto_strategy.generator.resample(
                    class_id,
                    number_of_samples
                )
            )


        print(
            "Generated:",
            len(synthetic_images)
        )


        # ==================================
        # STEP 3
        # CRITIC FILTER
        # ==================================

        print()
        print("------------------------------------------")
        print("STEP 3: CRITIC FILTER")
        print("------------------------------------------")

        critic_result = (
            self.auto_strategy.critic.evaluate(
                synthetic_images,
                synthetic_labels,
                threshold=threshold
            )
        )

        print(
            "Accepted:",
            critic_result["accepted"]
        )

        print(
            "Acceptance:",
            f"{critic_result['acceptance_rate'] * 100:.2f}%"
        )


        # ==================================
        # GET ACCEPTED SAMPLES
        # ==================================

        self.auto_strategy.classifier.eval()

        with torch.no_grad():

            outputs = self.auto_strategy.classifier(
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
                (confidence >= threshold)
            )


        accepted_images = synthetic_images[
            accepted_mask.cpu()
        ]

        accepted_labels = synthetic_labels[
            accepted_mask.cpu()
        ]


        print(
            "Accepted synthetic samples:",
            len(accepted_images)
        )


        # ==================================
        # STEP 4
        # REAL-ONLY BASELINE
        # ==================================

        print()
        print("------------------------------------------")
        print("STEP 4: TRAIN REAL-ONLY BASELINE")
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
            "Real-only accuracy:",
            f"{baseline_accuracy * 100:.2f}%"
        )


        # ==================================
        # STEP 5
        # AUGMENT DATASET
        # ==================================

        print()
        print("------------------------------------------")
        print("STEP 5: AUGMENT DATASET")
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
            "Synthetic samples:",
            len(accepted_images)
        )

        print(
            "Total samples:",
            len(augmented_images)
        )


        # ==================================
        # STEP 6
        # TRAIN AUGMENTED MODEL
        # ==================================

        print()
        print("------------------------------------------")
        print("STEP 6: TRAIN AUGMENTED MODEL")
        print("------------------------------------------")

        augmented_model = train_model(
            augmented_images,
            augmented_labels,
            epochs=5
        )

        augmented_accuracy = evaluate_model(
            augmented_model
        )


        # ==================================
        # STEP 7
        # DOWNSTREAM FEEDBACK
        # ==================================

        improvement = (
            augmented_accuracy
            - baseline_accuracy
        )

        print()
        print("==========================================")
        print("        DOWNSTREAM ML RESULT")
        print("==========================================")

        print()
        print(
            "Real-only:",
            f"{baseline_accuracy * 100:.2f}%"
        )

        print(
            "Augmented:",
            f"{augmented_accuracy * 100:.2f}%"
        )

        print(
            "Improvement:",
            f"{improvement * 100:.2f} pp"
        )


        # ==================================
        # STEP 8
        # FINAL GENFORGE DECISION
        # ==================================

        print()
        print("==========================================")
        print("          GENFORGE FINAL DECISION")
        print("==========================================")

        if improvement > 0:

            print()
            print("KEEP")

            print(
                "Synthetic data improved "
                "downstream ML performance."
            )

        else:

            print()
            print("REJECT")

            print(
                "Synthetic data did not improve "
                "downstream ML performance."
            )


        # ==================================
        # RETURN RESULT
        # ==================================

        return {
            "class_id": class_id,
            "strategy": strategy,
            "scale": best_scale,
            "generated": len(synthetic_images),
            "accepted": len(accepted_images),
            "baseline_accuracy": baseline_accuracy,
            "augmented_accuracy": augmented_accuracy,
            "improvement": improvement
        }


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    controller = GenForgeController()

    result = controller.run(
        class_id=5,
        number_of_samples=500,
        threshold=0.90
    )

    print()
    print("==========================================")
    print("             FINAL RESULT")
    print("==========================================")

    print(result)