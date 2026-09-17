import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
# IMPORTS
# ==========================================

from models.conditional_vae import ConditionalVAE
from evaluation.classifier import CNN


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "conditional_vae_v2.pth"
)

CLASSIFIER_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "baseline_cnn.pth"
)

LATENT_SIZE = 32
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 100


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==========================================
# LOAD CVAE
# ==========================================

print("\nLoading V2 Conditional VAE...")

vae = ConditionalVAE(
    latent_size=LATENT_SIZE,
    num_classes=NUM_CLASSES
).to(device)

vae.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

vae.eval()

print("V2 CVAE loaded successfully.")


# ==========================================
# LOAD CLASSIFIER
# ==========================================

print("\nLoading independent classifier...")

classifier = CNN().to(device)

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location=device
    )
)

classifier.eval()

print("Classifier loaded successfully.")


# ==========================================
# GENERATE AND TEST
# ==========================================

total_correct = 0
total_samples = 0


print("\n==========================================")
print("V2 CVAE CONDITIONAL GENERATION TEST")
print("==========================================")

for class_id in range(NUM_CLASSES):

    # --------------------------------------
    # Create requested labels
    # --------------------------------------

    labels = torch.full(
        (SAMPLES_PER_CLASS,),
        class_id,
        dtype=torch.long,
        device=device
    )

    # --------------------------------------
    # Random Gaussian latent vectors
    # --------------------------------------

    noise = torch.randn(
        SAMPLES_PER_CLASS,
        LATENT_SIZE,
        device=device
    )

    # --------------------------------------
    # Generate images
    # --------------------------------------

    with torch.no_grad():

        generated_images = vae.decode(
            noise,
            labels
        )

        outputs = classifier(
            generated_images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

    # --------------------------------------
    # Calculate accuracy
    # --------------------------------------

    correct = (
        predictions == labels
    ).sum().item()

    accuracy = (
        correct /
        SAMPLES_PER_CLASS
    )

    total_correct += correct
    total_samples += SAMPLES_PER_CLASS

    print(
        f"Class {class_id}: "
        f"{correct}/{SAMPLES_PER_CLASS} "
        f"correct -> "
        f"{accuracy * 100:.2f}%"
    )


# ==========================================
# FINAL RESULT
# ==========================================

overall_accuracy = (
    total_correct /
    total_samples
)

print("\n==========================================")
print("V2 FINAL RESULT")
print("==========================================")

print(
    f"Total correct: "
    f"{total_correct}/{total_samples}"
)

print(
    f"V2 Conditional Accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print("==========================================")