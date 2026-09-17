# ========================================
# GENFORGE DATASET ANALYZER
# ========================================

import torch
from torch.utils.data import TensorDataset, Subset


# ========================================
# GET LABELS
# ========================================

def get_labels(dataset):

    # ------------------------------------
    # Normal torchvision dataset
    # ------------------------------------

    if hasattr(dataset, "targets"):

        return dataset.targets

    # ------------------------------------
    # TensorDataset
    # ------------------------------------

    if isinstance(dataset, TensorDataset):

        return dataset.tensors[1]

    # ------------------------------------
    # Subset
    # ------------------------------------

    if isinstance(dataset, Subset):

        original_dataset = dataset.dataset

        indices = dataset.indices

        # Original dataset has targets
        if hasattr(original_dataset, "targets"):

            labels = original_dataset.targets

            if torch.is_tensor(indices):

                return labels[indices]

            indices = torch.tensor(
                indices,
                dtype=torch.long
            )

            return labels[indices]

        # Original dataset is TensorDataset
        if isinstance(
            original_dataset,
            TensorDataset
        ):

            labels = original_dataset.tensors[1]

            if torch.is_tensor(indices):

                return labels[indices]

            indices = torch.tensor(
                indices,
                dtype=torch.long
            )

            return labels[indices]

    raise ValueError(
        "Could not find labels in dataset."
    )


# ========================================
# ANALYZE DATASET
# ========================================

def analyze_dataset(dataset):

    labels = get_labels(dataset)

    # Make sure labels are Tensor
    if not torch.is_tensor(labels):

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    labels = labels.long()

    total_samples = len(labels)

    unique_classes = torch.unique(
        labels
    )

    number_of_classes = len(
        unique_classes
    )

    class_counts = {}

    # ------------------------------------
    # COUNT EACH CLASS
    # ------------------------------------

    for class_id in unique_classes:

        class_id = class_id.item()

        count = (
            labels == class_id
        ).sum().item()

        percentage = (
            count / total_samples
        ) * 100

        class_counts[class_id] = count

        print(
            f"{class_id}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # ------------------------------------
    # MAJORITY CLASS
    # ------------------------------------

    majority_class = max(
        class_counts,
        key=class_counts.get
    )

    majority_count = class_counts[
        majority_class
    ]

    # ------------------------------------
    # MINORITY CLASS
    # ------------------------------------

    minority_class = min(
        class_counts,
        key=class_counts.get
    )

    minority_count = class_counts[
        minority_class
    ]

    # ------------------------------------
    # IMBALANCE RATIO
    # ------------------------------------

    if minority_count == 0:

        imbalance_ratio = float("inf")

    else:

        imbalance_ratio = (
            majority_count /
            minority_count
        )

    # ------------------------------------
    # AUGMENTATION DECISION
    # ------------------------------------

    augmentation_needed = (
        imbalance_ratio > 1.5
    )

    # ------------------------------------
    # PRINT RESULT
    # ------------------------------------

    print()

    print(
        "Total samples:",
        total_samples
    )

    print(
        "Classes:",
        number_of_classes
    )

    print(
        "Majority class:",
        majority_class
    )

    print(
        "Minority class:",
        minority_class
    )

    print(
        "Imbalance ratio:",
        f"{imbalance_ratio:.4f}"
    )

    print(
        "Augmentation needed:",
        augmentation_needed
    )

    if augmentation_needed:

        print(
            "Recommendation: "
            "Synthetic augmentation required."
        )

    else:

        print(
            "Recommendation: "
            "Dataset is sufficiently balanced."
        )

    # ------------------------------------
    # RETURN
    # ------------------------------------

    return {

        "total_samples":
            total_samples,

        "number_of_classes":
            number_of_classes,

        "class_counts":
            class_counts,

        "majority_class":
            majority_class,

        "minority_class":
            minority_class,

        "majority_count":
            majority_count,

        "minority_count":
            minority_count,

        "imbalance_ratio":
            imbalance_ratio,

        "augmentation_needed":
            augmentation_needed
    }


# ========================================
# TEST
# ========================================

if __name__ == "__main__":

    from torchvision import datasets
    from torchvision import transforms

    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    print()
    print("========================================")
    print("        DATASET ANALYZER TEST")
    print("========================================")

    result = analyze_dataset(
        dataset
    )

    print()
    print(result)