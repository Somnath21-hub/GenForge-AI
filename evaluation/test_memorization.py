import torch
import torch.nn.functional as F
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================
# IMPORT CVAE
# ==========================================

from models.conditional_vae import ConditionalVAE


# ==========================================
# SETTINGS
# ==========================================

LATENT_SIZE = 32
NUM_CLASSES = 10

SAMPLES_PER_CLASS = 100

NOISE_SCALE = 0.15

# Number of real images used for comparison
REAL_SAMPLES_PER_CLASS = 1000


# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==========================================
# LOAD CVAE
# ==========================================

print("\nLoading Conditional VAE...")

model = ConditionalVAE(
    latent_size=LATENT_SIZE,
    num_classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(
    torch.load(
        os.path.join(
            PROJECT_ROOT,
            "models",
            "conditional_vae.pth"
        ),
        map_location=DEVICE
    )
)

model.eval()

print("CVAE loaded successfully.")


# ==========================================
# LOAD MNIST
# ==========================================

print("\nLoading MNIST...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

dataset = datasets.MNIST(
    root=os.path.join(
        PROJECT_ROOT,
        "data"
    ),
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(
    "MNIST samples:",
    len(dataset)
)


# ==========================================
# COLLECT REAL IMAGES + LATENTS
# ==========================================

print(
    "\nCollecting real images and latent vectors..."
)

class_images = {
    class_id: []
    for class_id in range(NUM_CLASSES)
}

class_latents = {
    class_id: []
    for class_id in range(NUM_CLASSES)
}


with torch.no_grad():

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        mu, logvar = model.encode(
            images,
            labels
        )

        for class_id in range(NUM_CLASSES):

            mask = labels == class_id

            if mask.any():

                selected_images = images[mask]
                selected_latents = mu[mask]

                class_images[class_id].append(
                    selected_images.cpu()
                )

                class_latents[class_id].append(
                    selected_latents.cpu()
                )


# Combine batches
for class_id in range(NUM_CLASSES):

    class_images[class_id] = torch.cat(
        class_images[class_id],
        dim=0
    )

    class_latents[class_id] = torch.cat(
        class_latents[class_id],
        dim=0
    )

    print(
        f"Class {class_id}: "
        f"{len(class_images[class_id])} real images"
    )


# ==========================================
# GENERATE SYNTHETIC IMAGES
# ==========================================

print(
    "\nGenerating latent-manifold samples..."
)

synthetic_data = {}

with torch.no_grad():

    for class_id in range(NUM_CLASSES):

        latents = class_latents[class_id]

        # Select random real latent vectors
        indices = torch.randint(
            low=0,
            high=len(latents),
            size=(SAMPLES_PER_CLASS,)
        )

        selected_latents = latents[
            indices
        ].to(DEVICE)

        # Add latent noise
        noise = torch.randn_like(
            selected_latents
        ) * NOISE_SCALE

        z = selected_latents + noise

        labels = torch.full(
            (SAMPLES_PER_CLASS,),
            class_id,
            dtype=torch.long,
            device=DEVICE
        )

        synthetic_images = model.decode(
            z,
            labels
        )

        synthetic_data[class_id] = (
            synthetic_images.cpu()
        )


print("Generation completed.")


# ==========================================
# NEAREST-NEIGHBOR TEST
# ==========================================

print("\n==============================")
print("MEMORIZATION TEST")
print("==============================")


all_nearest_distances = []


for class_id in range(NUM_CLASSES):

    synthetic_images = synthetic_data[
        class_id
    ]

    real_images = class_images[
        class_id
    ]

    # Limit number of real samples
    if len(real_images) > REAL_SAMPLES_PER_CLASS:

        indices = torch.randperm(
            len(real_images)
        )[:REAL_SAMPLES_PER_CLASS]

        real_images = real_images[
            indices
        ]

    # Flatten
    synthetic_flat = synthetic_images.view(
        len(synthetic_images),
        -1
    )

    real_flat = real_images.view(
        len(real_images),
        -1
    )

    # Normalize
    synthetic_flat = F.normalize(
        synthetic_flat,
        p=2,
        dim=1
    )

    real_flat = F.normalize(
        real_flat,
        p=2,
        dim=1
    )

    # Compare synthetic images against
    # real images.
    #
    # cosine similarity:
    # 1.0 = extremely similar
    # lower = more different

    similarity = torch.mm(
        synthetic_flat,
        real_flat.T
    )

    # Closest real image for each
    # synthetic image
    nearest_similarity, nearest_index = (
        similarity.max(dim=1)
    )

    # Convert similarity into distance
    nearest_distance = (
        1.0 - nearest_similarity
    )

    mean_distance = (
        nearest_distance.mean().item()
    )

    min_distance = (
        nearest_distance.min().item()
    )

    max_distance = (
        nearest_distance.max().item()
    )

    all_nearest_distances.extend(
        nearest_distance.tolist()
    )

    print(
        f"Class {class_id}: "
        f"Mean nearest distance = "
        f"{mean_distance:.4f} | "
        f"Min = {min_distance:.4f} | "
        f"Max = {max_distance:.4f}"
    )


# ==========================================
# OVERALL RESULTS
# ==========================================

all_nearest_distances = np.array(
    all_nearest_distances
)

overall_mean = (
    all_nearest_distances.mean()
)

overall_min = (
    all_nearest_distances.min()
)

overall_max = (
    all_nearest_distances.max()
)


print("\n------------------------------")

print(
    f"Overall mean nearest distance: "
    f"{overall_mean:.4f}"
)

print(
    f"Overall minimum distance: "
    f"{overall_min:.4f}"
)

print(
    f"Overall maximum distance: "
    f"{overall_max:.4f}"
)

print("------------------------------")


# ==========================================
# HIGH-SIMILARITY MEMORIZATION CHECK
# ==========================================

print("\n==============================")
print("HIGH-SIMILARITY CHECK")
print("==============================")


# These are intentionally conservative
# diagnostic thresholds.
#
# similarity > 0.995:
# almost identical
#
# similarity > 0.99:
# extremely similar

similarity_values = (
    1.0 - all_nearest_distances
)


very_high = (
    similarity_values > 0.995
).mean()

high = (
    similarity_values > 0.990
).mean()


print(
    f"Similarity > 0.995: "
    f"{very_high * 100:.2f}%"
)

print(
    f"Similarity > 0.990: "
    f"{high * 100:.2f}%"
)


# ==========================================
# INTERPRETATION
# ==========================================

print("\n==============================")
print("INTERPRETATION")
print("==============================")


if very_high < 0.01:

    print(
        "Very high similarity is rare."
    )

    print(
        "No strong evidence of "
        "memorization was detected."
    )

elif very_high < 0.05:

    print(
        "Some highly similar samples exist."
    )

    print(
        "Further investigation may be useful."
    )

else:

    print(
        "A substantial fraction of samples "
        "are extremely similar to real images."
    )

    print(
        "Possible memorization/reconstruction "
        "issue."
    )


# ==========================================
# SAVE RESULTS
# ==========================================

results = {
    "noise_scale": NOISE_SCALE,
    "samples_per_class": SAMPLES_PER_CLASS,
    "real_samples_per_class": REAL_SAMPLES_PER_CLASS,
    "overall_mean_nearest_distance": float(
        overall_mean
    ),
    "overall_minimum_distance": float(
        overall_min
    ),
    "overall_maximum_distance": float(
        overall_max
    ),
    "similarity_above_0.995_percent": float(
        very_high * 100
    ),
    "similarity_above_0.990_percent": float(
        high * 100
    )
}


results_folder = os.path.join(
    PROJECT_ROOT,
    "experiments"
)

os.makedirs(
    results_folder,
    exist_ok=True
)


results_path = os.path.join(
    results_folder,
    "memorization_results.json"
)


import json

with open(
    results_path,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print(
    "\nResults saved to:"
)

print(results_path)

print(
    "\nMemorization test completed."
)