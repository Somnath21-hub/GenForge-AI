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

from models.conditional_generator import ConditionalGenerator
from models.conditional_discriminator import ConditionalDiscriminator


# ==============================
# SETTINGS
# ==============================

BATCH_SIZE = 64
EPOCHS = 10
NOISE_SIZE = 100
NUM_CLASSES = 10
LEARNING_RATE = 0.0002


# ==============================
# DATASET
# ==============================

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
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==============================
# MODELS
# ==============================

generator = ConditionalGenerator(
    noise_size=NOISE_SIZE,
    num_classes=NUM_CLASSES
)

discriminator = ConditionalDiscriminator(
    num_classes=NUM_CLASSES
)


# ==============================
# LOSS
# ==============================

criterion = torch.nn.BCELoss()


# ==============================
# OPTIMIZERS
# ==============================

optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


# ==============================
# TRAINING
# ==============================

if __name__ == "__main__":

    os.makedirs(
        "./generated",
        exist_ok=True
    )

    print("Starting Conditional DCGAN training...")
    print("Number of images:", len(dataset))
    print("Number of batches:", len(dataloader))
    print()

    # Fixed labels for checking
    # whether generator learns specific digits

    fixed_labels = torch.tensor([
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9,
        0, 5, 5, 9,
        1, 7, 8, 3
    ])

    fixed_noise = torch.randn(
        len(fixed_labels),
        NOISE_SIZE
    )

    for epoch in range(EPOCHS):

        for batch_index, (images, labels) in enumerate(dataloader):

            batch_size = images.size(0)

            # =================================
            # TRAIN DISCRIMINATOR
            # =================================

            real_labels = torch.ones(
                batch_size,
                1
            )

            fake_labels = torch.zeros(
                batch_size,
                1
            )

            # Generate noise

            noise = torch.randn(
                batch_size,
                NOISE_SIZE
            )

            # Generate fake images
            # using REAL labels

            fake_images = generator(
                noise,
                labels
            )

            # Discriminator checks real images

            real_prediction = discriminator(
                images,
                labels
            )

            # Discriminator checks fake images

            fake_prediction = discriminator(
                fake_images.detach(),
                labels
            )

            # Real image loss

            real_loss = criterion(
                real_prediction,
                real_labels
            )

            # Fake image loss

            fake_loss = criterion(
                fake_prediction,
                fake_labels
            )

            # Total discriminator loss

            discriminator_loss = (
                real_loss +
                fake_loss
            )

            optimizer_D.zero_grad()

            discriminator_loss.backward()

            optimizer_D.step()


            # =================================
            # TRAIN GENERATOR
            # =================================

            noise = torch.randn(
                batch_size,
                NOISE_SIZE
            )

            # Generate fake images again

            fake_images = generator(
                noise,
                labels
            )

            # Ask discriminator

            fake_prediction = discriminator(
                fake_images,
                labels
            )

            # Generator wants discriminator
            # to think fake images are REAL

            generator_labels = torch.ones(
                batch_size,
                1
            )

            generator_loss = criterion(
                fake_prediction,
                generator_labels
            )

            optimizer_G.zero_grad()

            generator_loss.backward()

            optimizer_G.step()


            # =================================
            # PRINT PROGRESS
            # =================================

            if batch_index % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(dataloader)}] "
                    f"D Loss: {discriminator_loss.item():.4f} "
                    f"G Loss: {generator_loss.item():.4f}"
                )


        # =================================
        # GENERATE FIXED TEST IMAGES
        # =================================

        generator.eval()

        with torch.no_grad():

            generated_images = generator(
                fixed_noise,
                fixed_labels
            )

        generator.train()

        save_image(
            generated_images,
            f"./generated/cdcgan_epoch_{epoch + 1}.png",
            normalize=True,
            nrow=10
        )

        print()
        print(
            f"Epoch {epoch + 1} completed."
        )

        print(
            "Saved:",
            f"generated/cdcgan_epoch_{epoch + 1}.png"
        )

        print()


    # =================================
    # SAVE MODELS
    # =================================

    torch.save(
        generator.state_dict(),
        "./generated/conditional_generator.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "./generated/conditional_discriminator.pth"
    )

    print("==========================================")
    print("Conditional DCGAN training completed!")
    print("==========================================")

    print(
        "Generator saved:",
        "./generated/conditional_generator.pth"
    )

    print(
        "Discriminator saved:",
        "./generated/conditional_discriminator.pth"
    )