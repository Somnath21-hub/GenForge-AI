import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.conditional_vae import ConditionalVAE
from evaluation.classifier import CNN


# ========================================
# DEVICE
# ========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ========================================
# LOAD CVAE
# ========================================

print("\nLoading ConditionalVAE...")

vae = ConditionalVAE(
    latent_size=32,
    num_classes=10
).to(device)

vae.load_state_dict(
    torch.load(
        "models/conditional_vae.pth",
        map_location=device
    )
)

vae.eval()

print(
    "ConditionalVAE loaded successfully."
)


# ========================================
# LOAD CLASSIFIER
# ========================================

print("\nLoading CNN classifier...")

classifier = CNN().to(device)

classifier.load_state_dict(
    torch.load(
        "evaluation/baseline_cnn.pth",
        map_location=device
    )
)

classifier.eval()

print(
    "CNN classifier loaded successfully."
)


# ========================================
# LOAD MNIST TEST DATA
# ========================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False
)


# ========================================
# RECONSTRUCTION TEST
# ========================================

total_correct = 0
total_images = 0

total_mse = 0.0
total_batches = 0


print("\n")
print("========================================")
print("TESTING CVAE RECONSTRUCTION")
print("========================================")


with torch.no_grad():

    for images, labels in dataloader:

        images = images.to(device)

        labels = labels.to(device)


        # --------------------------------
        # Encode
        # --------------------------------

        mu, logvar = vae.encode(
            images,
            labels
        )


        # --------------------------------
        # Use mean directly
        # --------------------------------
        # This removes random sampling
        # from the reconstruction test.

        z = mu


        # --------------------------------
        # Decode with CORRECT label
        # --------------------------------

        reconstructed = vae.decode(
            z,
            labels
        )


        # --------------------------------
        # Calculate reconstruction error
        # --------------------------------

        mse = torch.mean(
            (reconstructed - images) ** 2
        )

        total_mse += mse.item()

        total_batches += 1


        # --------------------------------
        # Classify reconstructed images
        # --------------------------------

        outputs = classifier(
            reconstructed
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # --------------------------------
        # Count correct predictions
        # --------------------------------

        correct = (
            predictions == labels
        ).sum().item()


        total_correct += correct

        total_images += len(labels)


# ========================================
# FINAL RESULTS
# ========================================

accuracy = (
    total_correct /
    total_images
) * 100


average_mse = (
    total_mse /
    total_batches
)


print("\n")
print("========================================")
print("RECONSTRUCTION RESULT")
print("========================================")

print(
    f"Correct: "
    f"{total_correct}/{total_images}"
)

print(
    f"Reconstruction Accuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"Average Reconstruction MSE: "
    f"{average_mse:.6f}"
)

print("========================================")