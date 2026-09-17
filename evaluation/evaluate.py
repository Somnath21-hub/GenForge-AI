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
from torchvision.utils import save_image

from models.generator import Generator


# ==========================================
# SETTINGS
# ==========================================

NUM_IMAGES = 1000
NOISE_SIZE = 100

MODEL_PATH = "./generated/generator.pth"
OUTPUT_PATH = "./evaluation/generated_samples.png"


# ==========================================
# LOAD GENERATOR
# ==========================================

generator = Generator()

generator.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

generator.eval()


# ==========================================
# GENERATE SYNTHETIC DATA
# ==========================================

print("Generating synthetic images...")

noise = torch.randn(
    NUM_IMAGES,
    NOISE_SIZE
)

with torch.no_grad():

    generated_images = generator(noise)


print(
    "Generated images shape:",
    generated_images.shape
)


# ==========================================
# SAVE SAMPLE GRID
# ==========================================

os.makedirs(
    "./evaluation",
    exist_ok=True
)

save_image(
    generated_images[:64],
    OUTPUT_PATH,
    normalize=True,
    nrow=8
)


print(
    "Sample images saved to:",
    OUTPUT_PATH
)


# ==========================================
# BASIC STATISTICS
# ==========================================

pixel_mean = generated_images.mean().item()

pixel_std = generated_images.std().item()

pixel_min = generated_images.min().item()

pixel_max = generated_images.max().item()


print()
print("========== IMAGE STATISTICS ==========")

print(
    "Pixel Mean:",
    round(pixel_mean, 4)
)

print(
    "Pixel Std:",
    round(pixel_std, 4)
)

print(
    "Pixel Min:",
    round(pixel_min, 4)
)

print(
    "Pixel Max:",
    round(pixel_max, 4)
)


# ==========================================
# DIVERSITY CHECK
# ==========================================

# Take first 100 images

sample_images = generated_images[:100]

sample_images = sample_images.view(
    100,
    -1
)


# Calculate average distance between images

total_distance = 0

count = 0

for i in range(100):

    for j in range(i + 1, 100):

        distance = torch.mean(
            torch.abs(
                sample_images[i] -
                sample_images[j]
            )
        )

        total_distance += distance.item()

        count += 1


average_distance = (
    total_distance / count
)


print()
print("========== DIVERSITY ==========")

print(
    "Average image distance:",
    round(average_distance, 4)
)


# ==========================================
# FINAL MESSAGE
# ==========================================

print()
print("========== EVALUATION COMPLETE ==========")

print(
    "Generated:",
    NUM_IMAGES,
    "synthetic images"
)

print(
    "Visual samples:",
    OUTPUT_PATH
)