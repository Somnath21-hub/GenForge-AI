import torch
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.conditional_vae import ConditionalVAE
from evaluation.classifier import CNN
from evaluation.critic import Critic


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================
# LOAD CVAE
# =========================

cvae = ConditionalVAE(
    latent_size=32,
    num_classes=10
).to(device)

cvae.load_state_dict(
    torch.load(
        "models/conditional_vae.pth",
        map_location=device
    )
)

cvae.eval()


# =========================
# LOAD CLASSIFIER
# =========================

classifier = CNN().to(device)

classifier.load_state_dict(
    torch.load(
        "evaluation/baseline_cnn.pth",
        map_location=device
    )
)

classifier.eval()


# =========================
# CREATE CRITIC
# =========================

critic = Critic(
    classifier,
    device
)


# =========================
# GENERATE DATA
# =========================

images = []
labels = []

samples_per_class = 100

with torch.no_grad():

    for digit in range(10):

        z = torch.randn(
            samples_per_class,
            32,
            device=device
        )

        digit_labels = torch.full(
            (samples_per_class,),
            digit,
            dtype=torch.long,
            device=device
        )

        generated = cvae.decode(
            z,
            digit_labels
        )

        images.append(generated)
        labels.append(digit_labels)


images = torch.cat(images)
labels = torch.cat(labels)


print(
    "\nGenerated images:",
    images.shape
)


# =========================
# CRITIC EVALUATION
# =========================

result = critic.evaluate(
    images,
    labels,
    threshold=0.90
)


# =========================
# OVERALL RESULT
# =========================

print("\n========================================")
print("          CRITIC V2 REPORT")
print("========================================")

print(
    "\nTotal samples:",
    result["total"]
)

print(
    "Accepted:",
    result["accepted"]
)

print(
    "Rejected:",
    result["rejected"]
)

print(
    "Acceptance rate:",
    round(
        result["acceptance_rate"] * 100,
        2
    ),
    "%"
)

print(
    "Average confidence:",
    round(
        result["average_confidence"],
        4
    )
)

print(
    "Conditional accuracy:",
    round(
        result["class_accuracy"] * 100,
        2
    ),
    "%"
)

print(
    "Diversity score:",
    round(
        result["diversity_score"],
        4
    )
)

print(
    "Unique images:",
    result["unique_images"]
)

print(
    "Duplicate rate:",
    round(
        result["duplicate_rate"] * 100,
        2
    ),
    "%"
)


# =========================
# CLASS-WISE RESULT
# =========================

print("\n========================================")
print("          CLASS-WISE CRITIC")
print("========================================")

for digit, data in result["class_results"].items():

    print(
        f"Class {digit}: "
        f"Accuracy = {data['accuracy'] * 100:.2f}% | "
        f"Accepted = "
        f"{data['accepted']}/{data['total']} | "
        f"Acceptance = "
        f"{data['acceptance_rate'] * 100:.2f}% | "
        f"Confidence = "
        f"{data['average_confidence']:.4f}"
    )