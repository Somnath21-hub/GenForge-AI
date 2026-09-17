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
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator


# ==========================================
# DATASET
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)


# ==========================================
# MODELS
# ==========================================

generator = Generator()

discriminator = Discriminator()


# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = torch.nn.BCELoss()


# ==========================================
# OPTIMIZERS
# ==========================================

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)


optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)


# ==========================================
# TRAINING
# ==========================================

if __name__ == "__main__":

    epochs = 10

    os.makedirs("./generated", exist_ok=True)

    # Fixed noise
    fixed_noise = torch.randn(64, 100)

    print("Starting DCGAN training...")
    print("Number of images:", len(dataset))
    print("Number of batches:", len(dataloader))
    print()


    for epoch in range(epochs):

        for batch_index, (images, labels) in enumerate(dataloader):


            # ==========================================
            # TRAIN DISCRIMINATOR
            # ==========================================

            real_images = images

            real_labels = torch.ones(
                images.size(0),
                1
            )

            fake_labels = torch.zeros(
                images.size(0),
                1
            )


            # Generate fake images

            noise = torch.randn(
                images.size(0),
                100
            )

            fake_images = generator(noise)


            # Discriminator on REAL images

            real_prediction = discriminator(
                real_images
            )


            # Discriminator on FAKE images

            fake_prediction = discriminator(
                fake_images.detach()
            )


            # Calculate losses

            real_loss = criterion(
                real_prediction,
                real_labels
            )

            fake_loss = criterion(
                fake_prediction,
                fake_labels
            )


            discriminator_loss = (
                real_loss + fake_loss
            )


            # Update Discriminator

            optimizer_D.zero_grad()

            discriminator_loss.backward()

            optimizer_D.step()


            # ==========================================
            # TRAIN GENERATOR
            # ==========================================

            noise = torch.randn(
                images.size(0),
                100
            )


            fake_images = generator(noise)


            fake_prediction = discriminator(
                fake_images
            )


            # Generator wants discriminator
            # to think fake images are REAL

            generator_labels = torch.ones(
                images.size(0),
                1
            )


            generator_loss = criterion(
                fake_prediction,
                generator_labels
            )


            # Update Generator

            optimizer_G.zero_grad()

            generator_loss.backward()

            optimizer_G.step()


            # ==========================================
            # PRINT PROGRESS
            # ==========================================

            if batch_index % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_index}/{len(dataloader)}] "
                    f"D Loss: {discriminator_loss.item():.4f} "
                    f"G Loss: {generator_loss.item():.4f}"
                )


        # ==========================================
        # SAVE GENERATED IMAGES
        # ==========================================

        generator.eval()

        with torch.no_grad():

            generated_images = generator(
                fixed_noise
            )

        generator.train()


        save_image(
            generated_images,
            f"./generated/dcgan_epoch_{epoch + 1}.png",
            normalize=True,
            nrow=8
        )


        print(
            f"Epoch {epoch + 1} completed."
        )

        print(
            f"Saved: generated/dcgan_epoch_{epoch + 1}.png"
        )

        print()


    # ==========================================
    # SAVE MODEL WEIGHTS
    # ==========================================

    torch.save(
        generator.state_dict(),
        "./generated/generator.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "./generated/discriminator.pth"
    )


    print("DCGAN training completed!")
    print("Generator saved.")
    print("Discriminator saved.")