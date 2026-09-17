# ========================================
# GENFORGE MULTI-SEED FINAL VALIDATION
# ========================================

import os
import random
import json

import numpy as np
import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    ConcatDataset
)

from torchvision import datasets, transforms

from orchestration.nodes import train_model

from analysis.dataset_analyzer import analyze_dataset

from analysis.augmentation_planner import (
    create_augmentation_plan
)

from optimizer.generation_strategy import (
    GenerationStrategy
)

from evaluation.independent_evaluator import (
    IndependentEvaluator
)


# ========================================
# DEVICE
# ========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Validation device:", device)

if torch.cuda.is_available():

    print(
        "Validation GPU:",
        torch.cuda.get_device_name(0)
    )


# ========================================
# TEST MODEL
# ========================================

def test_model(model, dataset):

    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

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

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return accuracy, correct


# ========================================
# RANDOM SEED
# ========================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ========================================
# EVALUATOR DATASETS
# ========================================

def create_evaluator_datasets(train_dataset):

    images = train_dataset.data
    labels = train_dataset.targets

    evaluator_images = (
        images[:10000]
        .unsqueeze(1)
        .float()
        / 255.0
    )

    evaluator_images = (
        evaluator_images - 0.5
    ) / 0.5

    evaluator_labels = (
        labels[:10000]
        .long()
    )

    train_images = (
        evaluator_images[:8000]
    )

    train_labels = (
        evaluator_labels[:8000]
    )

    val_images = (
        evaluator_images[8000:10000]
    )

    val_labels = (
        evaluator_labels[8000:10000]
    )

    evaluator_train_dataset = TensorDataset(
        train_images,
        train_labels
    )

    evaluator_val_dataset = TensorDataset(
        val_images,
        val_labels
    )

    return (
        evaluator_train_dataset,
        evaluator_val_dataset
    )


# ========================================
# DEVELOPMENT DATASET
# ========================================

def create_development_dataset(train_dataset):

    images = train_dataset.data
    labels = train_dataset.targets

    selected_indices = []

    class_5_count = 0

    for i in range(
        10000,
        len(train_dataset)
    ):

        label = labels[i].item()

        if label == 5:

            if class_5_count < 1000:

                selected_indices.append(i)

                class_5_count += 1

        else:

            selected_indices.append(i)

    selected_indices = torch.tensor(
        selected_indices,
        dtype=torch.long
    )

    development_images = (
        images[selected_indices]
        .unsqueeze(1)
        .float()
        / 255.0
    )

    development_images = (
        development_images - 0.5
    ) / 0.5

    development_labels = (
        labels[selected_indices]
        .long()
    )

    return TensorDataset(
        development_images,
        development_labels
    )


# ========================================
# GENERATE SYNTHETIC DATA
# ========================================

def generate_synthetic_data(plan):

    generator = GenerationStrategy()

    all_images = []
    all_labels = []

    for class_id, class_plan in plan.items():

        amount = class_plan[
            "synthetic_samples"
        ]

        if amount <= 0:
            continue

        images, labels = generator.resample(
            class_id,
            amount
        )

        all_images.append(
            images.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

    if len(all_images) == 0:

        return (
            torch.empty(
                0,
                1,
                28,
                28
            ),
            torch.empty(
                0,
                dtype=torch.long
            )
        )

    return (
        torch.cat(
            all_images,
            dim=0
        ),
        torch.cat(
            all_labels,
            dim=0
        )
    )


# ========================================
# FILTER SYNTHETIC DATA
# ========================================

def filter_synthetic_data(
    model,
    images,
    labels,
    threshold=0.90
):

    model.eval()

    accepted_images = []
    accepted_labels = []

    batch_size = 128

    with torch.no_grad():

        for start in range(
            0,
            len(labels),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(labels)
            )

            batch_images = (
                images[start:end]
                .to(device)
            )

            batch_labels = (
                labels[start:end]
                .to(device)
            )

            outputs = model(
                batch_images
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predictions = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

            accepted = (
                (predictions == batch_labels)
                &
                (confidence >= threshold)
            )

            if accepted.any():

                accepted_images.append(
                    batch_images[
                        accepted
                    ].cpu()
                )

                accepted_labels.append(
                    batch_labels[
                        accepted
                    ].cpu()
                )

    if len(accepted_images) == 0:

        return (
            torch.empty(
                0,
                1,
                28,
                28
            ),
            torch.empty(
                0,
                dtype=torch.long
            )
        )

    return (
        torch.cat(
            accepted_images,
            dim=0
        ),
        torch.cat(
            accepted_labels,
            dim=0
        )
    )


# ========================================
# MAIN
# ========================================

def main():

    print("\n")

    print(
        "========================================"
    )

    print(
        "GENFORGE MULTI-SEED FINAL VALIDATION"
    )

    print(
        "========================================"
    )

    # ====================================
    # LOAD MNIST
    # ====================================

    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
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

    # ====================================
    # INDEPENDENT EVALUATOR
    # ====================================

    (
        evaluator_train,
        evaluator_val
    ) = create_evaluator_datasets(
        train_dataset
    )

    print("\n")

    print(
        "========================================"
    )

    print(
        "TRAINING INDEPENDENT EVALUATOR"
    )

    print(
        "========================================"
    )

    set_seed(999)

    independent_model = train_model(
        evaluator_train,
        epochs=10
    )

    evaluator_validation_accuracy, _ = (
        test_model(
            independent_model,
            evaluator_val
        )
    )

    print(
        "\nIndependent evaluator validation:",
        f"{evaluator_validation_accuracy * 100:.2f}%"
    )

    # ====================================
    # DEVELOPMENT DATASET
    # ====================================

    development_dataset = (
        create_development_dataset(
            train_dataset
        )
    )

    print(
        "\nDevelopment dataset:",
        len(development_dataset)
    )

    dataset_info = analyze_dataset(
        development_dataset
    )

    print(
        "Imbalance ratio:",
        f"{dataset_info['imbalance_ratio']:.4f}"
    )

    print(
        "Augmentation needed:",
        dataset_info["augmentation_needed"]
    )

    # ====================================
    # AUGMENTATION PLAN
    # ====================================

    plan = create_augmentation_plan(
        dataset_info
    )

    print("\n")

    print(
        "========================================"
    )

    print(
        "SYNTHETIC GENERATION PLAN"
    )

    print(
        "========================================"
    )

    total_required = 0

    for class_id, class_plan in plan.items():

        amount = class_plan[
            "synthetic_samples"
        ]

        print(
            f"Class {class_id}: {amount}"
        )

        total_required += amount

    print(
        "Total synthetic required:",
        total_required
    )

    # ====================================
    # INDEPENDENT EVALUATOR OBJECT
    # ====================================

    independent_evaluator = (
        IndependentEvaluator(
            independent_model,
            device
        )
    )

    # ====================================
    # RANDOM SEEDS
    # ====================================

    seeds = [
        42,
        123,
        456
    ]

    results = []

    # ====================================
    # MULTI-SEED EXPERIMENT
    # ====================================

    for seed in seeds:

        print("\n\n")

        print(
            "========================================"
        )

        print(
            f"RUNNING SEED {seed}"
        )

        print(
            "========================================"
        )

        # ==================================
        # BASELINE
        # ==================================

        set_seed(seed)

        print(
            "\nTraining baseline..."
        )

        baseline_model = train_model(
            development_dataset,
            epochs=5
        )

        baseline_accuracy, _ = (
            test_model(
                baseline_model,
                test_dataset
            )
        )

        print(
            "Baseline accuracy:",
            f"{baseline_accuracy * 100:.2f}%"
        )

        # ==================================
        # GENERATE SYNTHETIC DATA
        # ==================================

        set_seed(seed)

        (
            synthetic_images,
            synthetic_labels
        ) = generate_synthetic_data(
            plan
        )

        total_synthetic = len(
            synthetic_labels
        )

        print(
            "Synthetic samples:",
            total_synthetic
        )

        # ==================================
        # INDEPENDENT EVALUATION
        # ==================================

        independent_result = (
            independent_evaluator.evaluate(
                synthetic_images,
                synthetic_labels
            )
        )

        independent_synthetic_accuracy = (
            independent_result["accuracy"]
        )

        print(
            "Independent synthetic accuracy:",
            f"{independent_synthetic_accuracy * 100:.2f}%"
        )

        # ==================================
        # FILTER SYNTHETIC DATA
        # ==================================

        (
            accepted_images,
            accepted_labels
        ) = filter_synthetic_data(
            baseline_model,
            synthetic_images,
            synthetic_labels,
            threshold=0.90
        )

        accepted_count = len(
            accepted_labels
        )

        acceptance_rate = (
            accepted_count
            /
            total_synthetic
            if total_synthetic > 0
            else 0.0
        )

        print(
            "Accepted synthetic:",
            accepted_count,
            "/",
            total_synthetic
        )

        print(
            "Acceptance rate:",
            f"{acceptance_rate * 100:.2f}%"
        )

        # ==================================
        # CREATE AUGMENTED DATASET
        # ==================================

        if accepted_count > 0:

            synthetic_dataset = TensorDataset(
                accepted_images,
                accepted_labels
            )

            augmented_dataset = ConcatDataset(
                [
                    development_dataset,
                    synthetic_dataset
                ]
            )

        else:

            augmented_dataset = (
                development_dataset
            )

        # ==================================
        # AUGMENTED MODEL
        # ==================================

        set_seed(seed)

        print(
            "\nTraining augmented model..."
        )

        augmented_model = train_model(
            augmented_dataset,
            epochs=5
        )

        augmented_accuracy, _ = (
            test_model(
                augmented_model,
                test_dataset
            )
        )

        improvement = (
            augmented_accuracy
            -
            baseline_accuracy
        )

        print(
            "\nBaseline:",
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
        # SAVE RUN RESULT
        # ==================================

        results.append({

            "seed":
                seed,

            "baseline_accuracy":
                float(
                    baseline_accuracy
                ),

            "augmented_accuracy":
                float(
                    augmented_accuracy
                ),

            "improvement":
                float(
                    improvement
                ),

            "total_synthetic":
                total_synthetic,

            "accepted_samples":
                accepted_count,

            "acceptance_rate":
                float(
                    acceptance_rate
                ),

            "independent_synthetic_accuracy":
                float(
                    independent_synthetic_accuracy
                )
        })

    # ========================================
    # STATISTICS
    # ========================================

    improvements = np.array([

        result["improvement"]

        for result in results

    ])

    mean_improvement = np.mean(
        improvements
    )

    standard_deviation = np.std(
        improvements,
        ddof=1
    )

    number_of_seeds = len(
        improvements
    )

    standard_error = (
        standard_deviation
        /
        np.sqrt(number_of_seeds)
    )

    # ========================================
    # 95% CONFIDENCE INTERVAL
    # ========================================

    # Three seeds -> degrees of freedom = 2
    # Two-sided 95% t critical = 4.303

    t_critical = 4.303

    margin_of_error = (
        t_critical
        *
        standard_error
    )

    confidence_interval_lower = (
        mean_improvement
        -
        margin_of_error
    )

    confidence_interval_upper = (
        mean_improvement
        +
        margin_of_error
    )

    # ========================================
    # FINAL DECISION
    # ========================================

    if confidence_interval_lower > 0:

        decision = "KEEP"

    elif confidence_interval_upper < 0:

        decision = "RETRY"

    else:

        decision = "UNCERTAIN"

    # ========================================
    # FINAL REPORT
    # ========================================

    print("\n\n")

    print(
        "========================================"
    )

    print(
        "FINAL VALIDATION RESULTS"
    )

    print(
        "========================================"
    )

    print(
        "\nIndependent evaluator validation:",
        f"{evaluator_validation_accuracy * 100:.2f}%"
    )

    for result in results:

        print("\n")

        print(
            f"Seed {result['seed']}"
        )

        print(
            "  Baseline:",
            f"{result['baseline_accuracy'] * 100:.2f}%"
        )

        print(
            "  Augmented:",
            f"{result['augmented_accuracy'] * 100:.2f}%"
        )

        print(
            "  Improvement:",
            f"{result['improvement'] * 100:.2f} pp"
        )

        print(
            "  Synthetic:",
            result["total_synthetic"]
        )

        print(
            "  Accepted:",
            result["accepted_samples"]
        )

        print(
            "  Acceptance rate:",
            f"{result['acceptance_rate'] * 100:.2f}%"
        )

        print(
            "  Independent synthetic accuracy:",
            f"{result['independent_synthetic_accuracy'] * 100:.2f}%"
        )

    # ========================================
    # STATISTICAL SUMMARY
    # ========================================

    print("\n")

    print(
        "========================================"
    )

    print(
        "STATISTICAL SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Mean improvement:",
        f"{mean_improvement * 100:.2f} pp"
    )

    print(
        "Standard deviation:",
        f"{standard_deviation * 100:.2f} pp"
    )

    print(
        "Standard error:",
        f"{standard_error * 100:.2f} pp"
    )

    print(
        "95% confidence interval:",
        f"[{confidence_interval_lower * 100:.2f}, "
        f"{confidence_interval_upper * 100:.2f}] pp"
    )

    print(
        "\nFinal decision:",
        decision
    )

    # ========================================
    # SAVE FINAL RESULTS
    # ========================================

    os.makedirs(
        "./experiments",
        exist_ok=True
    )

    output = {

        "experiment_type":
            "final_multi_seed_validation",

        "seeds":
            seeds,

        "evaluator_validation_accuracy":
            float(
                evaluator_validation_accuracy
            ),

        "total_synthetic_planned":
            total_required,

        "results":
            results,

        "mean_improvement":
            float(
                mean_improvement
            ),

        "standard_deviation":
            float(
                standard_deviation
            ),

        "standard_error":
            float(
                standard_error
            ),

        "confidence_interval_95":
            [
                float(
                    confidence_interval_lower
                ),

                float(
                    confidence_interval_upper
                )
            ],

        "decision":
            decision
    }

    output_path = (
        "./experiments/"
        "final_validation_results.json"
    )

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print("\nResults saved to:")

    print(
        output_path
    )

    print(
        "\n========================================"
    )


# ========================================
# RUN
# ========================================

if __name__ == "__main__":

    main()