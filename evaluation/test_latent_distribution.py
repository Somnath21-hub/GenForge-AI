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
    batch_size=128,
    shuffle=False
)


# ========================================
# COLLECT LATENT VALUES
# ========================================

all_mu = []
all_logvar = []
all_labels = []


print("\nCollecting latent representations...")


with torch.no_grad():

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        mu, logvar = vae.encode(
            images,
            labels
        )

        all_mu.append(
            mu.cpu()
        )

        all_logvar.append(
            logvar.cpu()
        )

        all_labels.append(
            labels.cpu()
        )


mu = torch.cat(
    all_mu,
    dim=0
)

logvar = torch.cat(
    all_logvar,
    dim=0
)

labels = torch.cat(
    all_labels,
    dim=0
)


# ========================================
# LATENT STATISTICS
# ========================================

print("\n")
print("========================================")
print("LATENT DISTRIBUTION")
print("========================================")


mean = mu.mean().item()

std = mu.std().item()

minimum = mu.min().item()

maximum = mu.max().item()


print(
    f"Overall latent mean: "
    f"{mean:.6f}"
)

print(
    f"Overall latent std: "
    f"{std:.6f}"
)

print(
    f"Overall latent min: "
    f"{minimum:.6f}"
)

print(
    f"Overall latent max: "
    f"{maximum:.6f}"
)


# ========================================
# PER-DIMENSION STATISTICS
# ========================================

dimension_mean = mu.mean(
    dim=0
)

dimension_std = mu.std(
    dim=0
)


print("\n")
print(
    "First 10 latent dimensions:"
)

for i in range(10):

    print(
        f"Dimension {i}: "
        f"mean={dimension_mean[i]:.4f}, "
        f"std={dimension_std[i]:.4f}"
    )


# ========================================
# KL STATISTICS
# ========================================

kl_per_sample = -0.5 * torch.sum(
    1
    + logvar
    - mu.pow(2)
    - logvar.exp(),
    dim=1
)


average_kl = (
    kl_per_sample.mean().item()
)


print("\n")
print(
    f"Average KL divergence: "
    f"{average_kl:.6f}"
)


# ========================================
# PER-CLASS LATENT MEAN
# ========================================

print("\n")
print("========================================")
print("PER-CLASS LATENT MEANS")
print("========================================")


for class_id in range(10):

    class_mu = mu[
        labels == class_id
    ]

    class_mean = (
        class_mu.mean().item()
    )

    class_std = (
        class_mu.std().item()
    )

    print(
        f"Class {class_id}: "
        f"mean={class_mean:.4f}, "
        f"std={class_std:.4f}"
    )


# ========================================
# FINAL
# ========================================

print("\n")
print("========================================")
print("INTERPRETATION")
print("========================================")

print(
    "Standard normal target:"
)

print(
    "Mean ≈ 0"
)

print(
    "Std ≈ 1"
)

print(
    "\nCompare these values with the "
    "learned latent distribution above."
)

print("========================================")