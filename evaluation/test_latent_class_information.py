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
import torch.nn as nn
import torch.optim as optim

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
# MNIST
# ========================================

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


train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)


# ========================================
# EXTRACT LATENT MU
# ========================================

def extract_latents(loader):

    all_mu = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            mu, logvar = vae.encode(
                images,
                labels
            )

            all_mu.append(
                mu.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    return (
        torch.cat(all_mu),
        torch.cat(all_labels)
    )


print("\nExtracting training latent vectors...")

train_mu, train_labels = extract_latents(
    train_loader
)

print(
    "Training latent shape:",
    train_mu.shape
)


print("\nExtracting test latent vectors...")

test_mu, test_labels = extract_latents(
    test_loader
)

print(
    "Test latent shape:",
    test_mu.shape
)


# ========================================
# LATENT CLASSIFIER
# ========================================

class LatentClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(32, 64),

            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):

        return self.model(x)


classifier = LatentClassifier().to(device)


# ========================================
# DATA LOADERS
# ========================================

train_latent_dataset = torch.utils.data.TensorDataset(
    train_mu,
    train_labels
)


test_latent_dataset = torch.utils.data.TensorDataset(
    test_mu,
    test_labels
)


train_latent_loader = DataLoader(
    train_latent_dataset,
    batch_size=256,
    shuffle=True
)


test_latent_loader = DataLoader(
    test_latent_dataset,
    batch_size=256,
    shuffle=False
)


# ========================================
# TRAIN
# ========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    classifier.parameters(),
    lr=0.001
)


epochs = 10

print("\n")
print("========================================")
print("TRAINING LATENT CLASSIFIER")
print("========================================")


for epoch in range(epochs):

    classifier.train()

    total_correct = 0
    total_samples = 0

    for latent, labels in train_latent_loader:

        latent = latent.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = classifier(
            latent
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += len(labels)


    train_accuracy = (
        total_correct /
        total_samples
    ) * 100


    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )


# ========================================
# TEST
# ========================================

classifier.eval()

correct = 0
total = 0


with torch.no_grad():

    for latent, labels in test_latent_loader:

        latent = latent.to(device)

        labels = labels.to(device)

        outputs = classifier(
            latent
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += len(labels)


test_accuracy = (
    correct /
    total
) * 100


# ========================================
# RESULT
# ========================================

print("\n")
print("========================================")
print("LATENT CLASS INFORMATION")
print("========================================")

print(
    f"Correct: {correct}/{total}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print("========================================")

print("\nInterpretation:")

if test_accuracy >= 90:

    print(
        "STRONG class information exists "
        "inside the latent representation."
    )

elif test_accuracy >= 60:

    print(
        "MODERATE class information exists "
        "inside the latent representation."
    )

else:

    print(
        "WEAK class information exists "
        "inside the latent representation."
    )