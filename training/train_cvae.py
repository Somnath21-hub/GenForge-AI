import argparse
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==========================================
# PROJECT ROOT
# ==========================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.conditional_vae import ConditionalVAE


def train_cvae(
    latent_size=32,
    epochs=20,
    learning_rate=0.001,
    batch_size=128,
    beta_max=0.02,
    warmup_epochs=8,
    save_path="models/conditional_vae_v3.pth"
):
    """
    Controlled CVAE Training Routine.

    Balances reconstruction quality (MSE) and latent space regularization (KL divergence)
    using controlled KL annealing warmup.
    """
    # Safeguard: Never overwrite V1 or V2
    normalized_save = os.path.normpath(save_path).lower()
    if normalized_save.endswith("conditional_vae.pth") or normalized_save.endswith("conditional_vae_v2.pth"):
        raise ValueError(
            f"[Safety Guard] Refusing to overwrite historical baseline checkpoint: {save_path}. "
            f"Please specify a new path such as models/conditional_vae_v3.pth"
        )

    full_save_path = os.path.join(PROJECT_ROOT, save_path) if not os.path.isabs(save_path) else save_path
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("GENFORGE CVAE TRAINING")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Latent size: {latent_size} | Epochs: {epochs} | Beta max: {beta_max} | Warmup: {warmup_epochs}")
    print(f"Target Checkpoint: {save_path}")
    print("=" * 60)

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Training samples: {len(dataset)}")

    # Model & Optimizer
    model = ConditionalVAE(
        latent_size=latent_size,
        num_classes=10
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_reconstruction = 0.0
        total_kl = 0.0

        # Linear KL Warmup schedule
        if warmup_epochs > 0:
            beta = min(beta_max, beta_max * (epoch + 1) / warmup_epochs)
        else:
            beta = beta_max

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            reconstructed, mu, logvar = model(images, labels)

            # Pixel Reconstruction loss (MSE per pixel)
            recon_loss = F.mse_loss(reconstructed, images, reduction="mean")

            # Analytical KL divergence per sample averaged across batch and latent dimensions
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reconstruction += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_reconstruction / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Recon MSE: {avg_recon:.4f} | "
            f"KL: {avg_kl:.4f} | "
            f"Beta: {beta:.4f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), full_save_path)

    print("\n================================")
    print("CVAE TRAINING COMPLETED")
    print("================================")
    print("Best loss:", round(best_loss, 4))
    print("Saved checkpoint to:", full_save_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GenForge Conditional VAE")
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--beta_max", type=float, default=0.02)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="models/conditional_vae_v3.pth")
    args = parser.parse_args()

    train_cvae(
        latent_size=args.latent_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        beta_max=args.beta_max,
        warmup_epochs=args.warmup_epochs,
        save_path=args.save_path
    )