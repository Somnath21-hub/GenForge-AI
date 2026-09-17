import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # 100 -> 256 x 7 x 7
            nn.Linear(100, 256 * 7 * 7),

            nn.BatchNorm1d(256 * 7 * 7),

            nn.ReLU(),

            # Convert:
            # 12544 -> 256 x 7 x 7
            nn.Unflatten(
                1,
                (256, 7, 7)
            ),

            # 256 x 7 x 7
            # -> 128 x 14 x 14
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            # 128 x 14 x 14
            # -> 1 x 28 x 28
            nn.ConvTranspose2d(
                128,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, noise):

        return self.model(noise)


# ==========================================
# TEST GENERATOR
# ==========================================

if __name__ == "__main__":

    generator = Generator()

    # Testing mode
    generator.eval()

    # Generate random noise
    noise = torch.randn(1, 100)

    # Don't calculate gradients
    with torch.no_grad():

        output = generator(noise)

    print("Noise shape:", noise.shape)
    print("Output shape:", output.shape)