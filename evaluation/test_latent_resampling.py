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
# MNIST DATA
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
    train=True,
    download=True,
    transform=transform
)


dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=False
)


# ========================================
# COLLECT CLASS-SPECIFIC LATENTS
# ========================================

print("\nCollecting latent representations...")

class_latents = {
    class_id: []
    for class_id in range(10)
}


with torch.no_grad():

    for images, labels in dataloader:

        images = images.to(device)

        labels = labels.to(device)

        mu, logvar = vae.encode(
            images,
            labels
        )

        mu = mu.cpu()

        labels = labels.cpu()

        for class_id in range(10):

            mask = labels == class_id

            if mask.any():

                class_latents[class_id].append(
                    mu[mask]
                )


# Combine latent vectors for every class

for class_id in range(10):

    class_latents[class_id] = torch.cat(
        class_latents[class_id],
        dim=0
    )

    print(
        f"Class {class_id}: "
        f"{len(class_latents[class_id])} "
        f"latent vectors"
    )


# ========================================
# GENERATE USING LATENT RESAMPLING
# ========================================

images_per_class = 100

total_correct = 0
total_images = 0


print("\n")
print("========================================")
print("LATENT-MANIFOLD RESAMPLING")
print("========================================")

print(
    "Generating synthetic images using "
    "real encoded latent vectors."
)


for class_id in range(10):

    # ------------------------------------
    # Get latent vectors for this class
    # ------------------------------------

    latent_pool = class_latents[class_id]


    # ------------------------------------
    # Randomly select existing latent
    # vectors from this class
    # ------------------------------------

    indices = torch.randint(
        0,
        len(latent_pool),
        (
            images_per_class,
        )
    )


    selected_mu = latent_pool[
        indices
    ].to(device)


    # ------------------------------------
    # Add small latent noise
    # ------------------------------------
    #
    # This creates new points around
    # real encoded latent vectors instead
    # of simply copying the same vector.
    #

    noise = torch.randn(
        images_per_class,
        32,
        device=device
    )

    noise = noise * 0.15


    z = selected_mu + noise


    # ------------------------------------
    # Create requested labels
    # ------------------------------------

    labels = torch.full(
        (
            images_per_class,
        ),
        class_id,
        dtype=torch.long,
        device=device
    )


    # ------------------------------------
    # Generate images
    # ------------------------------------

    with torch.no_grad():

        generated_images = vae.decode(
            z,
            labels
        )


        # --------------------------------
        # Classify generated images
        # --------------------------------

        outputs = classifier(
            generated_images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )


    # ------------------------------------
    # Accuracy
    # ------------------------------------

    correct = (
        predictions == labels
    ).sum().item()


    accuracy = (
        correct /
        images_per_class
    ) * 100


    total_correct += correct

    total_images += images_per_class


    # ------------------------------------
    # Print
    # ------------------------------------

    print(
        f"Class {class_id}: "
        f"{correct}/{images_per_class} "
        f"correct -> "
        f"{accuracy:.2f}%"
    )


# ========================================
# OVERALL RESULT
# ========================================

overall_accuracy = (
    total_correct /
    total_images
) * 100


print("\n")
print("========================================")
print("FINAL RESULT")
print("========================================")

print(
    f"Total correct: "
    f"{total_correct}/{total_images}"
)

print(
    f"Latent Resampling Accuracy: "
    f"{overall_accuracy:.2f}%"
)

print("========================================")


# ========================================
# COMPARISON
# ========================================

print("\n")
print("========================================")
print("COMPARISON")
print("========================================")

print(
    "Random N(0,1) generation: 26.50%"
)

print(
    f"Latent-manifold resampling: "
    f"{overall_accuracy:.2f}%"
)

difference = (
    overall_accuracy - 26.50
)

print(
    f"Difference: "
    f"{difference:+.2f} percentage points"
)

print("========================================")