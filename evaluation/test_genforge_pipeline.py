import sys
import os

# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ==========================================
# IMPORTS
# ==========================================

import torch

from analysis.dataset_analyzer import (
    dataset,
    analyze_dataset
)

from analysis.augmentation_planner import (
    create_augmentation_plan
)

from models.conditional_vae import (
    ConditionalVAE
)

from evaluation.critic import (
    Critic
)

from evaluation.classifier import (
    CNN
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==========================================
# STEP 1: DATASET ANALYSIS
# ==========================================

print()
print("==========================================")
print("          STEP 1: DATASET ANALYSIS")
print("==========================================")

analysis = analyze_dataset(
    dataset
)


# ==========================================
# STEP 2: AUGMENTATION PLAN
# ==========================================

print()
print("==========================================")
print("          STEP 2: AUGMENTATION PLAN")
print("==========================================")


# Previous Critic result
# Used temporarily for this integration test.

critic_result_for_planner = {

    "class_results": {

        0: {
            "accuracy": 0.97,
            "acceptance_rate": 0.93
        },

        1: {
            "accuracy": 0.90,
            "acceptance_rate": 0.86
        },

        2: {
            "accuracy": 0.96,
            "acceptance_rate": 0.86
        },

        3: {
            "accuracy": 0.97,
            "acceptance_rate": 0.92
        },

        4: {
            "accuracy": 0.92,
            "acceptance_rate": 0.83
        },

        5: {
            "accuracy": 0.89,
            "acceptance_rate": 0.78
        },

        6: {
            "accuracy": 0.94,
            "acceptance_rate": 0.90
        },

        7: {
            "accuracy": 0.91,
            "acceptance_rate": 0.82
        },

        8: {
            "accuracy": 0.89,
            "acceptance_rate": 0.71
        },

        9: {
            "accuracy": 0.83,
            "acceptance_rate": 0.74
        }
    }
}


plan = create_augmentation_plan(
    analysis,
    critic_result=critic_result_for_planner,
    quality_threshold=0.90
)


print()
print("Planned synthetic samples:")

print(
    plan["synthetic_samples"]
)

print()

print(
    "Total planned:",
    plan["total_synthetic_samples"]
)


# ==========================================
# STEP 3: LOAD CVAE
# ==========================================

print()
print("==========================================")
print("          STEP 3: LOAD CVAE")
print("==========================================")


cvae_path = os.path.join(
    PROJECT_ROOT,
    "models",
    "conditional_vae.pth"
)


cvae = ConditionalVAE(
    latent_size=32,
    num_classes=10
)


cvae.load_state_dict(
    torch.load(
        cvae_path,
        map_location=device
    )
)


cvae.to(device)

cvae.eval()


print(
    "CVAE loaded successfully."
)


# ==========================================
# STEP 4: SYNTHETIC GENERATION
# ==========================================

print()
print("==========================================")
print("       STEP 4: SYNTHETIC GENERATION")
print("==========================================")


generated_images = []

generated_labels = []


for class_id, number_of_samples in (
    plan["synthetic_samples"].items()
):

    if number_of_samples <= 0:

        continue


    print(
        f"Class {class_id}: "
        f"Generating {number_of_samples}..."
    )


    # Random latent vector
    noise = torch.randn(
        number_of_samples,
        20,
        device=device
    )


    # Requested class labels
    labels = torch.full(
        (number_of_samples,),
        class_id,
        dtype=torch.long,
        device=device
    )


    # Generate images
    with torch.no_grad():

        images = cvae.decode(
            noise,
            labels
        )


    generated_images.append(
        images
    )

    generated_labels.append(
        labels
    )


# ==========================================
# COMBINE GENERATED DATA
# ==========================================

generated_images = torch.cat(
    generated_images,
    dim=0
)

generated_labels = torch.cat(
    generated_labels,
    dim=0
)


print()

print(
    "Total generated:",
    len(generated_images)
)

print(
    "Image shape:",
    generated_images.shape
)

print(
    "Label shape:",
    generated_labels.shape
)


# ==========================================
# STEP 5: LOAD CRITIC
# ==========================================

print()
print("==========================================")
print("          STEP 5: LOAD CRITIC")
print("==========================================")


classifier_path = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "baseline_cnn.pth"
)


classifier = CNN()


classifier.load_state_dict(
    torch.load(
        classifier_path,
        map_location=device
    )
)


classifier.to(device)

classifier.eval()


critic = Critic(
    classifier=classifier,
    device=device
)


print(
    "Critic loaded successfully."
)


# ==========================================
# STEP 6: CRITIC EVALUATION
# ==========================================

print()
print("==========================================")
print("       STEP 6: CRITIC EVALUATION")
print("==========================================")


result = critic.evaluate(
    generated_images,
    generated_labels,
    threshold=0.90
)


# ==========================================
# STEP 7: FINAL REPORT
# ==========================================

print()
print("==========================================")
print("          GENFORGE FINAL REPORT")
print("==========================================")


print()

print(
    "Total synthetic:",
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
    f"{result['acceptance_rate'] * 100:.2f}%"
)

print(
    "Conditional accuracy:",
    f"{result['class_accuracy'] * 100:.2f}%"
)

print(
    "Average confidence:",
    f"{result['average_confidence'] * 100:.2f}%"
)

print(
    "Diversity score:",
    f"{result['diversity_score']:.4f}"
)

print(
    "Unique images:",
    result["unique_images"]
)

print(
    "Duplicate rate:",
    f"{result['duplicate_rate'] * 100:.2f}%"
)


# ==========================================
# STEP 8: CLASS-WISE QUALITY
# ==========================================

print()
print("========== CLASS-WISE QUALITY ==========")


for class_id, data in (
    result["class_results"].items()
):

    print(
        f"Class {class_id}: "
        f"Accuracy = "
        f"{data['accuracy'] * 100:.2f}% | "
        f"Acceptance = "
        f"{data['acceptance_rate'] * 100:.2f}% | "
        f"Confidence = "
        f"{data['average_confidence'] * 100:.2f}%"
    )


# ==========================================
# COMPLETE
# ==========================================

print()
print("==========================================")
print("     GENFORGE PIPELINE TEST COMPLETE")
print("==========================================")