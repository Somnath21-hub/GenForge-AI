# ========================================
# GENFORGE LANGGRAPH NODES
# ========================================

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

from analysis.dataset_analyzer import analyze_dataset
from analysis.augmentation_planner import create_augmentation_plan

from optimizer.generation_strategy import GenerationStrategy
from optimizer.strategy_planner import StrategyPlanner
from optimizer.experiment_history import ExperimentHistory

from evaluation.critic import Critic
from evaluation.independent_evaluator import IndependentEvaluator

from optimizer.adaptive_optimizer import AdaptiveOptimizer


# ========================================
# DEVICE
# ========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("GenForge device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ========================================
# REPRODUCIBILITY
# ========================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========================================
# CURRENT CNN
# MUST MATCH baseline_cnn.pth
# ========================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, image):

        image = self.features(image)

        return self.classifier(image)


# ========================================
# MNIST TRANSFORM
# ========================================

TRANSFORM = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


# ========================================
# LOAD MNIST
# ========================================

def load_mnist():

    return datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=TRANSFORM
    )


# ========================================
# CREATE EXPERIMENTAL DATASET
# ========================================

def create_experimental_dataset():

    dataset = load_mnist()

    targets = dataset.targets

    selected_indices = []

    class_5_count = 0

    for index, label in enumerate(targets):

        label = int(label)

        # Keep all classes except class 5
        if label != 5:

            selected_indices.append(index)

        # Keep only 1000 samples of class 5
        else:

            if class_5_count < 1000:

                selected_indices.append(index)

                class_5_count += 1

    return Subset(
        dataset,
        selected_indices
    )


# ========================================
# ANALYZER NODE
# ========================================

def analyzer_node(state):

    print()
    print("========================================")
    print("DATASET ANALYZER")
    print("========================================")

    dataset = create_experimental_dataset()

    dataset_info = analyze_dataset(dataset)

    print(
        "Experimental dataset:",
        dataset_info["total_samples"]
    )

    for class_id, count in dataset_info[
        "class_counts"
    ].items():

        percentage = (
            count /
            dataset_info["total_samples"]
        ) * 100

        print(
            f"{class_id}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    print(
        "Total:",
        dataset_info["total_samples"]
    )

    print(
        "Classes:",
        dataset_info["number_of_classes"]
    )

    print(
        "Majority:",
        dataset_info["majority_class"]
    )

    print(
        "Minority:",
        dataset_info["minority_class"]
    )

    print(
        "Majority count:",
        dataset_info["majority_count"]
    )

    print(
        "Minority count:",
        dataset_info["minority_count"]
    )

    print(
        "Imbalance ratio:",
        round(
            dataset_info["imbalance_ratio"],
            4
        )
    )

    print(
        "Augmentation needed:",
        dataset_info["augmentation_needed"]
    )

    return {

        "dataset_info": dataset_info
    }


# ========================================
# PLANNER NODE
# ========================================

def planner_node(state):

    print()
    print("========================================")
    print("AUGMENTATION PLANNER")
    print("========================================")

    dataset_info = state[
        "dataset_info"
    ]

    augmentation_plan = create_augmentation_plan(
        dataset_info
    )

    total_synthetic = 0

    for class_id, data in augmentation_plan.items():

        synthetic_samples = data[
            "synthetic_samples"
        ]

        total_synthetic += synthetic_samples

        print(
            f"{class_id}: "
            f"{synthetic_samples}"
        )

    print(
        "Total synthetic required:",
        total_synthetic
    )

    return {

        "augmentation_plan":
            augmentation_plan
    }


# ========================================
# GENERATION NODE
# ========================================

def generation_node(state):

    print()
    print("========================================")
    print("GENERATION")
    print("========================================")

    augmentation_plan = state[
        "augmentation_plan"
    ]

    strategy = GenerationStrategy(
        model_path="models/conditional_vae.pth",
        latent_size=32,
        num_classes=10,
        device=DEVICE
    )

    optimizer_decision = state.get("optimizer_decision", {})
    decision = optimizer_decision.get("decision", "KEEP")

    if decision == "ADAPT_LATENT":
        active_strategy = "ADAPTIVE_LATENT"
        active_scale = 0.80
    elif decision == "RESAMPLE":
        active_strategy = "LATENT_MANIFOLD"
        active_scale = 0.20
    else:
        active_strategy = "LATENT_MANIFOLD"
        active_scale = 0.15

    print(f"Generation node using strategy: {active_strategy} (scale={active_scale:.2f})")

    all_images = []
    all_labels = []

    for class_id, data in augmentation_plan.items():

        number_of_samples = data[
            "synthetic_samples"
        ]

        if number_of_samples <= 0:
            continue

        images, labels = strategy.generate(
            class_id=int(class_id),
            number_of_samples=number_of_samples,
            strategy=active_strategy,
            scale=active_scale
        )

        all_images.append(
            images.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

    if len(all_images) == 0:

        return {
            "generation_result": {
                "images": torch.empty(
                    0,
                    1,
                    28,
                    28
                ),
                "labels": torch.empty(
                    0,
                    dtype=torch.long
                ),
                "total_generated": 0
            }
        }

    images = torch.cat(
        all_images,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    print(
        "Total generated:",
        len(labels)
    )

    print(
        "Shape:",
        images.shape
    )

    return {

        "generation_result": {

            "images": images,

            "labels": labels,

            "total_generated":
                len(labels)
        }
    }


# ========================================
# LOAD CRITIC MODEL
# ========================================

def load_critic_model():

    model = CNN().to(DEVICE)

    checkpoint_path = (
        "evaluation/baseline_cnn.pth"
    )

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            "baseline_cnn.pth not found. "
            "Run evaluation/train_baseline_cnn.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


# ========================================
# CRITIC NODE
# ========================================

def critic_node(state):

    print()
    print("========================================")
    print("CRITIC")
    print("========================================")

    generation_result = state[
        "generation_result"
    ]

    images = generation_result[
        "images"
    ]

    labels = generation_result[
        "labels"
    ]

    if len(labels) == 0:

        return {

            "critic_result": {

                "total": 0,

                "accepted": 0,

                "rejected": 0,

                "acceptance_rate": 0.0,

                "average_confidence": 0.0,

                "class_accuracy": 0.0,

                "diversity": 0.0,

                "duplicate_rate": 0.0,

                "quality_score": 0.0
            }
        }

    # ------------------------------------
    # IMPORTANT:
    # Use the SAME trained CNN architecture
    # as baseline_cnn.pth.
    # ------------------------------------

    model = load_critic_model()

    critic = Critic(
        classifier=model,
        device=DEVICE
    )

    result = critic.evaluate(
        images,
        labels,
        threshold=0.90
    )

    print("Critic result:")

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    return {

        "critic_result": result
    }


# ========================================
# TRAIN MODEL
# ========================================

def train_model(
    train_dataset,
    epochs=5,
    seed=42
):

    set_seed(seed)

    model = CNN().to(DEVICE)

    dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    model.train()

    for epoch in range(epochs):

        for images, labels in dataloader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

    return model


# ========================================
# EVALUATE MODEL
# ========================================

def evaluate_model(
    model,
    dataset
):

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        correct / total
        if total > 0
        else 0.0
    )


# ========================================
# EVALUATION NODE
# ========================================

def evaluation_node(state):

    print()
    print("========================================")
    print("DOWNSTREAM EVALUATION")
    print("========================================")

    generation_result = state[
        "generation_result"
    ]

    synthetic_images = generation_result[
        "images"
    ]

    synthetic_labels = generation_result[
        "labels"
    ]

    dataset = load_mnist()

    # ------------------------------------
    # First 10,000 samples = independent
    # evaluator validation set
    # ------------------------------------

    evaluator_indices = list(
        range(
            0,
            10000
        )
    )

    evaluator_dataset = Subset(
        dataset,
        evaluator_indices
    )

    # ------------------------------------
    # Remaining data = development set
    # ------------------------------------

    development_indices = []

    class_5_count = 0

    for index in range(
        10000,
        len(dataset)
    ):

        label = int(
            dataset.targets[index]
        )

        if label != 5:

            development_indices.append(
                index
            )

        else:

            if class_5_count < 1000:

                development_indices.append(
                    index
                )

                class_5_count += 1

    development_dataset = Subset(
        dataset,
        development_indices
    )

    print(
        "Development dataset:",
        len(development_dataset)
    )

    # ------------------------------------
    # Independent evaluator
    # ------------------------------------
    evaluator_path = "models/independent_evaluator.pth"
    evaluator_model = CNN().to(DEVICE)
    if os.path.exists(evaluator_path):
        evaluator_model.load_state_dict(
            torch.load(evaluator_path, map_location=DEVICE)
        )
        evaluator_model.eval()
    else:
        evaluator_model = train_model(
            evaluator_dataset,
            epochs=10,
            seed=42
        )
        torch.save(evaluator_model.state_dict(), evaluator_path)

    independent_evaluator = IndependentEvaluator(
        model=evaluator_model,
        device=DEVICE
    )

    evaluator_validation = (
        independent_evaluator.evaluate(
            torch.stack(
                [
                    dataset[i][0]
                    for i in evaluator_indices
                ]
            ),
            torch.tensor(
                [
                    int(dataset.targets[i])
                    for i in evaluator_indices
                ]
            )
        )
    )

    print(
        "Independent evaluator validation:",
        f"{evaluator_validation['accuracy'] * 100:.2f}%"
    )

    # ------------------------------------
    # Baseline model
    # ------------------------------------

    baseline_model = train_model(
        development_dataset,
        epochs=5,
        seed=42
    )

    baseline_accuracy = evaluate_model(
        baseline_model,
        evaluator_dataset
    )

    print(
        "Baseline accuracy:",
        f"{baseline_accuracy * 100:.2f}%"
    )

    # ------------------------------------
    # Synthetic evaluation
    # ------------------------------------

    synthetic_result = (
        independent_evaluator.evaluate(
            synthetic_images,
            synthetic_labels
        )
    )

    synthetic_accuracy = (
        synthetic_result["accuracy"]
    )

    print(
        "Independent synthetic accuracy:",
        f"{synthetic_accuracy * 100:.2f}%"
    )

    # ------------------------------------
    # Filter synthetic data using baseline
    # ------------------------------------

    accepted_images = []
    accepted_labels = []

    correct_predictions = 0

    if len(synthetic_labels) > 0:

        baseline_model.eval()

        synthetic_batch = DataLoader(
            TensorDataset(
                synthetic_images,
                synthetic_labels
            ),
            batch_size=256,
            shuffle=False
        )

        with torch.no_grad():

            for images, labels in synthetic_batch:

                images_device = images.to(
                    DEVICE
                )

                labels_device = labels.to(
                    DEVICE
                )

                outputs = baseline_model(
                    images_device
                )

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predictions = (
                    probabilities.max(dim=1)
                )

                mask = (
                    (confidence >= 0.90)
                    &
                    (predictions == labels_device)
                )

                for i in range(
                    len(images)
                ):

                    if mask[i]:

                        accepted_images.append(
                            images[i]
                        )

                        accepted_labels.append(
                            labels[i]
                        )

                        correct_predictions += 1

    if len(accepted_images) > 0:

        accepted_images = torch.stack(
            accepted_images
        )

        accepted_labels = torch.stack(
            accepted_labels
        )

    else:

        accepted_images = torch.empty(
            0,
            1,
            28,
            28
        )

        accepted_labels = torch.empty(
            0,
            dtype=torch.long
        )

    synthetic_total = len(
        synthetic_labels
    )

    accepted_total = len(
        accepted_labels
    )

    acceptance_rate = (
        accepted_total / synthetic_total
        if synthetic_total > 0
        else 0.0
    )

    print(
        "Synthetic samples:",
        synthetic_total
    )

    print(
        "Accepted synthetic:",
        accepted_total
    )

    print(
        "Acceptance rate:",
        f"{acceptance_rate * 100:.2f}%"
    )

    # ------------------------------------
    # Augmented model
    # ------------------------------------

    real_images = []
    real_labels = []

    for image, label in development_dataset:

        real_images.append(image)
        real_labels.append(label)

    real_images = torch.stack(
        real_images
    )

    real_labels = torch.tensor(
        real_labels,
        dtype=torch.long
    )

    if accepted_total > 0:

        augmented_images = torch.cat(
            [
                real_images,
                accepted_images
            ],
            dim=0
        )

        augmented_labels = torch.cat(
            [
                real_labels,
                accepted_labels
            ],
            dim=0
        )

    else:

        augmented_images = real_images

        augmented_labels = real_labels

    augmented_dataset = TensorDataset(
        augmented_images,
        augmented_labels
    )

    augmented_model = train_model(
        augmented_dataset,
        epochs=5,
        seed=42
    )

    augmented_accuracy = evaluate_model(
        augmented_model,
        evaluator_dataset
    )

    improvement = (
        augmented_accuracy -
        baseline_accuracy
    )

    print(
        "Augmented accuracy:",
        f"{augmented_accuracy * 100:.2f}%"
    )

    print(
        "Improvement:",
        f"{improvement * 100:.2f} pp"
    )

    # ------------------------------------
    # Return results
    # ------------------------------------

    evaluation_result = {

        "baseline_accuracy":
            baseline_accuracy,

        "augmented_accuracy":
            augmented_accuracy,

        "improvement":
            improvement,

        "synthetic_accuracy":
            synthetic_accuracy,

        "synthetic_total":
            synthetic_total,

        "accepted_synthetic":
            accepted_total,

        "acceptance_rate":
            acceptance_rate,

        "evaluator_validation_accuracy":
            evaluator_validation["accuracy"],

        "confidence_lower":
            0.0,

        "confidence_upper":
            0.0
    }

    return {

        "evaluation_result":
            evaluation_result
    }


# ========================================
# OPTIMIZER NODE
# ========================================

def optimizer_node(state):

    print()
    print("========================================")
    print("OPTIMIZER")
    print("========================================")

    critic_result = state.get(
        "critic_result",
        {}
    )

    evaluation_result = state.get(
        "evaluation_result",
        {}
    )

    # ------------------------------------
    # IMPORTANT:
    # Evaluation acceptance rate has
    # priority over critic acceptance.
    # ------------------------------------

    acceptance_rate = evaluation_result.get(
        "acceptance_rate",
        critic_result.get(
            "acceptance_rate",
            0.0
        )
    )

    improvement = evaluation_result.get(
        "improvement",
        0.0
    )

    baseline_accuracy = evaluation_result.get(
        "baseline_accuracy",
        0.0
    )

    augmented_accuracy = evaluation_result.get(
        "augmented_accuracy",
        0.0
    )

    print(
        "Baseline:",
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

    print(
        "Acceptance rate:",
        f"{acceptance_rate * 100:.2f}%"
    )

    # ------------------------------------
    # QUALITY SCORE
    # ------------------------------------

    synthetic_accuracy = evaluation_result.get(
        "synthetic_accuracy",
        0.0
    )

    quality_score = (
        synthetic_accuracy * 70
        +
        acceptance_rate * 30
    )

    # ------------------------------------
    # DECISION
    # ------------------------------------

    if quality_score < 50:

        decision = "RETRAIN"

        reason = (
            "Synthetic quality is poor. "
            "The generator requires retraining."
        )

        priority = "CRITICAL"

        confidence = 0.85

    elif acceptance_rate < 0.75:

        decision = "ADAPT_LATENT"

        reason = (
            "Synthetic acceptance is low. "
            "Latent-space adaptation is recommended."
        )

        priority = "HIGH"

        confidence = 0.75

    elif improvement <= 0:

        decision = "RESAMPLE"

        reason = (
            "Synthetic data does not improve "
            "downstream performance."
        )

        priority = "MEDIUM"

        confidence = 0.70

    else:

        decision = "KEEP"

        reason = (
            "Synthetic data improves "
            "downstream performance."
        )

        priority = "LOW"

        confidence = 0.80

    # ------------------------------------
    # IMPROVEMENT STATUS
    # ------------------------------------

    if improvement > 0:

        improvement_status = "POSITIVE"

    elif improvement < 0:

        improvement_status = "NEGATIVE"

    else:

        improvement_status = "UNCHANGED"

    optimizer_decision = {

        "decision": decision,

        "reason": reason,

        "priority": priority,

        "confidence": confidence,

        "quality_score": quality_score,

        "improvement_status":
            improvement_status
    }

    print(
        "Optimizer decision:"
    )

    print(
        optimizer_decision
    )

    # ------------------------------------
    # SAVE ITERATION TO HISTORY
    # ------------------------------------

    history_manager = ExperimentHistory()

    experiment_id = state[
        "experiment_id"
    ]

    iteration = state[
        "iteration"
    ]

    iteration_data = {

        "iteration":
            iteration,

        "decision":
            decision,

        "baseline_accuracy":
            baseline_accuracy,

        "augmented_accuracy":
            augmented_accuracy,

        "improvement":
            improvement,

        "synthetic_accuracy":
            synthetic_accuracy,

        "synthetic_total":
            evaluation_result.get(
                "synthetic_total",
                0
            ),

        "accepted_synthetic":
            evaluation_result.get(
                "accepted_synthetic",
                0
            ),

        "acceptance_rate":
            acceptance_rate,

        "quality_score":
            quality_score,

        "optimizer_decision":
            optimizer_decision
    }

    history_manager.save_iteration(
        experiment_id,
        iteration_data
    )

    print(
        "Iteration saved to history.json"
    )

    # ------------------------------------
    # UPDATE ITERATION HISTORY
    # ------------------------------------

    current_history = list(
        state.get(
            "iteration_history",
            []
        )
    )

    current_history.append(
        iteration_data
    )

    # ------------------------------------
    # UPDATE BEST RESULT
    # ------------------------------------

    best_result = state.get(
        "best_result"
    )

    if (
        best_result is None
        or improvement >
        best_result.get(
            "improvement",
            float("-inf")
        )
    ):

        best_result = iteration_data

    # ------------------------------------
    # INCREMENT ITERATION
    # ------------------------------------

    next_iteration = iteration + 1

    termination_reason = None

    if decision == "KEEP":

        termination_reason = (
            "Synthetic data produced "
            "an acceptable improvement."
        )

    elif next_iteration > state[
        "max_iterations"
    ]:

        termination_reason = (
            "Maximum iterations reached."
        )

    return {

        "optimizer_decision":
            optimizer_decision,

        "iteration":
            next_iteration,

        "iteration_history":
            current_history,

        "best_result":
            best_result,

        "termination_reason":
            termination_reason
    }