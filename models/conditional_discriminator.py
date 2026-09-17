import torch
import torch.nn as nn


class ConditionalDiscriminator(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.num_classes = num_classes

        # Image feature extractor
        self.image_model = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        # Final decision
        self.classifier = nn.Sequential(

            nn.Linear(
                128 * 7 * 7 + num_classes,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, image, labels):

        # Extract image features
        image_features = self.image_model(image)

        # Convert label to one-hot
        labels = torch.nn.functional.one_hot(
            labels,
            num_classes=self.num_classes
        ).float()

        # Combine image + label
        combined = torch.cat(
            [image_features, labels],
            dim=1
        )

        # Real / Fake prediction
        output = self.classifier(
            combined
        )

        return output


if __name__ == "__main__":

    discriminator = ConditionalDiscriminator()

    discriminator.eval()

    images = torch.randn(
        4,
        1,
        28,
        28
    )

    labels = torch.tensor([
        0,
        5,
        5,
        9
    ])

    with torch.no_grad():

        output = discriminator(
            images,
            labels
        )

    print(
        "Image shape:",
        images.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    print(
        "Discriminator output shape:",
        output.shape
    )

    print(
        "Predictions:",
        output
    )