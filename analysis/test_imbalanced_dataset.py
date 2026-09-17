import torch
from torchvision import datasets, transforms

from dataset_analyzer import analyze_dataset
from augmentation_planner import create_augmentation_plan


# --------------------------------
# Load original MNIST
# --------------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# --------------------------------
# Create artificially imbalanced dataset
# --------------------------------

print()
print("Creating imbalanced MNIST dataset...")

labels = dataset.targets

selected_indices = []

for class_id in range(10):

    class_indices = torch.where(
        labels == class_id
    )[0]

    # Keep only 1000 samples for class 5
    if class_id == 5:

        class_indices = class_indices[:1000]

    selected_indices.append(class_indices)


# Combine all selected indices
selected_indices = torch.cat(
    selected_indices
)


# --------------------------------
# Create new dataset
# --------------------------------

imbalanced_images = dataset.data[
    selected_indices
]

imbalanced_labels = dataset.targets[
    selected_indices
]


# Create a simple dataset object
imbalanced_dataset = torch.utils.data.TensorDataset(
    imbalanced_images,
    imbalanced_labels
)


# --------------------------------
# Analyze imbalanced dataset
# --------------------------------

print()
print("==========================================")
print("       ANALYZING IMBALANCED DATASET")
print("==========================================")

analysis = analyze_dataset(
    imbalanced_dataset
)


# --------------------------------
# Create augmentation plan
# --------------------------------

plan = create_augmentation_plan(
    analysis
)


# --------------------------------
# Show final result
# --------------------------------

print()
print("==========================================")
print("       FINAL GENFORGE PLAN")
print("==========================================")

print(plan)

print()
print("==========================================")
print("Imbalanced dataset test completed.")
print("==========================================")