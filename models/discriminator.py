import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # 1 x 28 x 28
            # -> 64 x 14 x 14
            nn.Conv2d(
                1,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.2),

            # 64 x 14 x 14
            # -> 128 x 7 x 7
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2),

            # Flatten
            nn.Flatten(),

            # 128 × 7 × 7 = 6272
            nn.Linear(
                128 * 7 * 7,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, image):

        return self.model(image)


# ==========================================
# TEST DISCRIMINATOR
# ==========================================

if __name__ == "__main__":

    discriminator = Discriminator()

    discriminator.eval()

    # One MNIST-style image
    image = torch.randn(1, 1, 28, 28)

    with torch.no_grad():

        output = discriminator(image)

    print("Image shape:", image.shape)
    print("Output shape:", output.shape)
    print("Prediction:", output.item())