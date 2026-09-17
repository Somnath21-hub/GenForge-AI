import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
from models.conditional_vae import ConditionalVAE


# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "./models/conditional_vae.pth"

LATENT_SIZE = 32
NUM_CLASSES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================
# LOAD CVAE
# ============================================

def load_cvae(
    model_path=MODEL_PATH,
    latent_size=LATENT_SIZE,
    num_classes=NUM_CLASSES
):

    model = ConditionalVAE(
        latent_size=latent_size,
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ============================================
# GENERATE SYNTHETIC DATA
# ============================================

def generate_samples(
    model,
    class_id,
    number_of_samples
):

    # Create labels
    labels = torch.full(
        (number_of_samples,),
        class_id,
        dtype=torch.long,
        device=DEVICE
    )

    # Generate random latent vectors
    noise = torch.randn(
        number_of_samples,
        LATENT_SIZE,
        device=DEVICE
    )

    # Generate images
    with torch.no_grad():

        images = model.decode(
            noise,
            labels
        )

    return images, labels


# ============================================
# GENERATE DATA FOR MULTIPLE CLASSES
# ============================================

def generate_class_data(
    model,
    samples_per_class=1000
):

    all_images = []
    all_labels = []

    for class_id in range(NUM_CLASSES):

        print(
            f"Generating class {class_id}..."
        )

        images, labels = generate_samples(
            model,
            class_id,
            samples_per_class
        )

        all_images.append(images)
        all_labels.append(labels)

    # Combine all classes
    all_images = torch.cat(
        all_images,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    return all_images, all_labels


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print("=" * 50)
    print("LOADING CONDITIONAL VAE")
    print("=" * 50)

    model = load_cvae()

    print(
        "CVAE loaded successfully."
    )

    print(
        "Latent size:",
        LATENT_SIZE
    )

    print(
        "Number of classes:",
        NUM_CLASSES
    )

    print()
    print("=" * 50)
    print("TESTING CLASS-SPECIFIC GENERATION")
    print("=" * 50)

    # Generate a small test batch
    test_images, test_labels = generate_samples(
        model,
        class_id=5,
        number_of_samples=10
    )

    print(
        "Test images:",
        test_images.shape
    )

    print(
        "Test labels:",
        test_labels.shape
    )

    print(
        "Requested class:",
        5
    )

    print(
        "Unique labels:",
        torch.unique(test_labels).tolist()
    )

    print()
    print("=" * 50)
    print("GENERATING ALL CLASSES")
    print("=" * 50)

    images, labels = generate_class_data(
        model,
        samples_per_class=1000
    )

    print()
    print("=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)

    print(
        "Total synthetic images:",
        len(images)
    )

    print(
        "Image shape:",
        images.shape
    )

    print(
        "Label shape:",
        labels.shape
    )

    print()

    # Show class distribution
    for class_id in range(NUM_CLASSES):

        count = (
            labels == class_id
        ).sum().item()

        print(
            f"Class {class_id}: {count}"
        )

    print()
    print(
        "Synthetic generation successful."
    )