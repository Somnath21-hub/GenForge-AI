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

from models.conditional_generator import ConditionalGenerator


# ==========================================
# SETTINGS
# ==========================================

NUM_IMAGES_PER_CLASS = 100
NOISE_SIZE = 100
NUM_CLASSES = 10

GENERATOR_PATH = "./generated/conditional_generator.pth"
CLASSIFIER_PATH = "./evaluation/baseline_cnn.pth"


# ==========================================
# CNN CLASSIFIER
# ==========================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, image):

        return self.model(image)


# ==========================================
# LOAD GENERATOR
# ==========================================

generator = ConditionalGenerator()

generator.load_state_dict(
    torch.load(
        GENERATOR_PATH,
        map_location="cpu"
    )
)

generator.eval()


# ==========================================
# LOAD CLASSIFIER
# ==========================================

classifier = CNN()

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location="cpu"
    )
)

classifier.eval()


# ==========================================
# TEST
# ==========================================

total_correct = 0
total_images = 0

print()
print("==========================================")
print("     GENFORGE CONDITIONAL TEST")
print("==========================================")

for class_id in range(NUM_CLASSES):

    labels = torch.full(
        (NUM_IMAGES_PER_CLASS,),
        class_id,
        dtype=torch.long
    )

    noise = torch.randn(
        NUM_IMAGES_PER_CLASS,
        NOISE_SIZE
    )

    # Generate requested class

    with torch.no_grad():

        generated_images = generator(
            noise,
            labels
        )

        predictions = classifier(
            generated_images
        )

        predicted_labels = torch.argmax(
            predictions,
            dim=1
        )

    correct = (
        predicted_labels == labels
    ).sum().item()

    accuracy = (
        correct /
        NUM_IMAGES_PER_CLASS
    ) * 100

    total_correct += correct
    total_images += NUM_IMAGES_PER_CLASS

    print(
        f"Requested class {class_id}: "
        f"{correct}/{NUM_IMAGES_PER_CLASS} "
        f"({accuracy:.2f}%)"
    )


# ==========================================
# OVERALL RESULT
# ==========================================

overall_accuracy = (
    total_correct /
    total_images
) * 100

print()
print("==========================================")
print("       CONDITIONAL RESULT")
print("==========================================")

print(
    "Total correct:",
    total_correct,
    "/",
    total_images
)

print(
    "Conditional accuracy:",
    f"{overall_accuracy:.2f}%"
)

print()
print("==========================================")
print("Conditional generation test completed.")
print("==========================================")