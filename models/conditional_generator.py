import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):

    def __init__(self, noise_size=100, num_classes=10):

        super().__init__()

        self.noise_size = noise_size
        self.num_classes = num_classes

        self.model = nn.Sequential(

            # Noise + class label
            nn.Linear(
                noise_size + num_classes,
                256 * 7 * 7
            ),

            nn.BatchNorm1d(
                256 * 7 * 7
            ),

            nn.ReLU(),

            # Convert vector to image feature map
            nn.Unflatten(
                1,
                (256, 7, 7)
            ),

            # 256 × 7 × 7
            # ↓
            # 128 × 14 × 14
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            # 128 × 14 × 14
            # ↓
            # 1 × 28 × 28
            nn.ConvTranspose2d(
                128,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, noise, labels):

        # Convert class labels into one-hot vectors
        labels = torch.nn.functional.one_hot(
            labels,
            num_classes=self.num_classes
        ).float()

        # Combine noise + class information
        generator_input = torch.cat(
            [noise, labels],
            dim=1
        )

        return self.model(generator_input)


# --------------------------------
# Test the generator
# --------------------------------

if __name__ == "__main__":

    generator = ConditionalGenerator()

    generator.eval()

    # Create 4 random noise vectors
    noise = torch.randn(4, 100)

    # Ask generator for specific digits
    labels = torch.tensor([
        0,
        5,
        5,
        9
    ])

    with torch.no_grad():

        output = generator(
            noise,
            labels
        )

    print(
        "Noise shape:",
        noise.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    print(
        "Generator output shape:",
        output.shape
    )