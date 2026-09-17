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

# Noise added around real latent vectors
NOISE_SCALE = 0.15


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
# COLLECT REAL LATENT VECTORS
# ==========================================

print("\nCollecting real latent vectors...")

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

        # Encode real images
        mu, logvar = model.encode(
            images,
            labels
        )

        for class_id in range(NUM_CLASSES):

            mask = labels == class_id

            if mask.any():

                selected_latents = mu[mask]

                class_latents[class_id].append(
                    selected_latents.cpu()
                )


# Combine batches
for class_id in range(NUM_CLASSES):

    class_latents[class_id] = torch.cat(
        class_latents[class_id],
        dim=0
    )

    print(
        f"Class {class_id}: "
        f"{len(class_latents[class_id])} latent vectors"
    )


# ==========================================
# GENERATE LATENT-MANIFOLD SAMPLES
# ==========================================

print(
    "\nGenerating latent-manifold samples..."
)

synthetic_data = {}

with torch.no_grad():

    for class_id in range(NUM_CLASSES):

        latents = class_latents[class_id]

        # Randomly select real latent vectors
        indices = torch.randint(
            low=0,
            high=len(latents),
            size=(SAMPLES_PER_CLASS,)
        )

        selected_latents = latents[
            indices
        ].to(DEVICE)

        # Add small Gaussian noise
        noise = torch.randn_like(
            selected_latents
        ) * NOISE_SCALE

        z = selected_latents + noise

        # Requested class
        labels = torch.full(
            (SAMPLES_PER_CLASS,),
            class_id,
            dtype=torch.long,
            device=DEVICE
        )

        # Decode
        images = model.decode(
            z,
            labels
        )

        synthetic_data[class_id] = images.cpu()


print("Generation completed.")


# ==========================================
# DIVERSITY TEST
# ==========================================

print("\n==============================")
print("DIVERSITY RESULTS")
print("==============================")


all_diversity = []


for class_id in range(NUM_CLASSES):

    images = synthetic_data[class_id]

    # Flatten images
    flattened = images.view(
        SAMPLES_PER_CLASS,
        -1
    )

    # Normalize
    flattened = F.normalize(
        flattened,
        p=2,
        dim=1
    )

    # Pairwise cosine similarity
    similarity = torch.mm(
        flattened,
        flattened.T
    )

    # Remove self-comparisons
    mask = ~torch.eye(
        SAMPLES_PER_CLASS,
        dtype=torch.bool
    )

    similarities = similarity[
        mask
    ]

    average_similarity = (
        similarities.mean().item()
    )

    diversity = (
        1.0 - average_similarity
    )

    all_diversity.append(
        diversity
    )

    print(
        f"Class {class_id}: "
        f"Average similarity = "
        f"{average_similarity:.4f} | "
        f"Diversity = "
        f"{diversity:.4f}"
    )


average_diversity = np.mean(
    all_diversity
)


print("\n------------------------------")

print(
    f"Overall diversity: "
    f"{average_diversity:.4f}"
)

print("------------------------------")


# ==========================================
# NEAR-DUPLICATE TEST
# ==========================================

print("\n==============================")
print("NEAR-DUPLICATE TEST")
print("==============================")


duplicate_rates = []


for class_id in range(NUM_CLASSES):

    images = synthetic_data[class_id]

    flattened = images.view(
        SAMPLES_PER_CLASS,
        -1
    )

    flattened = F.normalize(
        flattened,
        p=2,
        dim=1
    )

    similarity = torch.mm(
        flattened,
        flattened.T
    )

    mask = ~torch.eye(
        SAMPLES_PER_CLASS,
        dtype=torch.bool
    )

    pair_similarities = similarity[
        mask
    ]

    # Similarity above 0.995
    # is considered a near duplicate
    duplicates = (
        pair_similarities > 0.995
    ).float().mean().item()

    duplicate_rates.append(
        duplicates
    )

    print(
        f"Class {class_id}: "
        f"Near-duplicate rate = "
        f"{duplicates * 100:.2f}%"
    )


average_duplicate_rate = np.mean(
    duplicate_rates
)


print("\n------------------------------")

print(
    f"Overall near-duplicate rate: "
    f"{average_duplicate_rate * 100:.2f}%"
)

print("------------------------------")


# ==========================================
# SAVE GENERATED DATA
# ==========================================

generated_folder = os.path.join(
    PROJECT_ROOT,
    "generated"
)

os.makedirs(
    generated_folder,
    exist_ok=True
)


save_path = os.path.join(
    generated_folder,
    "latent_diversity_samples.pth"
)


torch.save(
    synthetic_data,
    save_path
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n==============================")
print("FINAL SUMMARY")
print("==============================")

print(
    f"Noise scale: {NOISE_SCALE}"
)

print(
    f"Samples per class: "
    f"{SAMPLES_PER_CLASS}"
)

print(
    f"Overall diversity: "
    f"{average_diversity:.4f}"
)

print(
    f"Near-duplicate rate: "
    f"{average_duplicate_rate * 100:.2f}%"
)

print(
    "\nSaved samples to:"
)

print(save_path)

print(
    "\nDiversity test completed successfully."
)