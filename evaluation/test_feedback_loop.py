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
print("       STEP 1: DATASET ANALYSIS")
print("==========================================")

analysis = analyze_dataset(dataset)


# ==========================================
# STEP 2: LOAD CVAE
# ==========================================

print()
print("==========================================")
print("          STEP 2: LOAD CVAE")
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

print("CVAE loaded successfully.")


# ==========================================
# STEP 3: LOAD CRITIC
# ==========================================

print()
print("==========================================")
print("          STEP 3: LOAD CRITIC")
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

print("Critic loaded successfully.")


# ==========================================
# FUNCTION: GENERATE DATA
# ==========================================

def generate_samples(
    cvae,
    samples_per_class
):

    images_list = []
    labels_list = []

    for class_id in range(10):

        print(
            f"Class {class_id}: "
            f"Generating {samples_per_class}..."
        )

        noise = torch.randn(
            samples_per_class,
            20,
            device=device
        )

        labels = torch.full(
            (samples_per_class,),
            class_id,
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            images = cvae.decode(
                noise,
                labels
            )

        images_list.append(images)
        labels_list.append(labels)

    images = torch.cat(
        images_list,
        dim=0
    )

    labels = torch.cat(
        labels_list,
        dim=0
    )

    return images, labels


# ==========================================
# STEP 4: INITIAL GENERATION
# ==========================================

print()
print("==========================================")
print("       STEP 4: INITIAL GENERATION")
print("==========================================")

initial_images, initial_labels = generate_samples(
    cvae,
    samples_per_class=100
)

print()
print(
    "Initial synthetic samples:",
    len(initial_images)
)


# ==========================================
# STEP 5: INITIAL CRITIC
# ==========================================

print()
print("==========================================")
print("       STEP 5: INITIAL CRITIC")
print("==========================================")

initial_result = critic.evaluate(
    initial_images,
    initial_labels,
    threshold=0.90
)


print()
print("Initial acceptance rate:",
      f"{initial_result['acceptance_rate'] * 100:.2f}%")

print(
    "Initial conditional accuracy:",
    f"{initial_result['class_accuracy'] * 100:.2f}%"
)


# ==========================================
# STEP 6: PLANNER
# ==========================================

print()
print("==========================================")
print("       STEP 6: PLANNER DECISION")
print("==========================================")

plan = create_augmentation_plan(
    analysis,
    critic_result=initial_result,
    quality_threshold=0.90
)

print()
print(
    "Weak classes:",
    plan["weak_classes"]
)

print()
print(
    "Additional synthetic samples:",
    plan["synthetic_samples"]
)

print()
print(
    "Total additional samples:",
    plan["total_synthetic_samples"]
)


# ==========================================
# STEP 7: ADAPTIVE GENERATION
# ==========================================

print()
print("==========================================")
print("       STEP 7: ADAPTIVE GENERATION")
print("==========================================")

adaptive_images = []
adaptive_labels = []


for class_id, number_of_samples in (
    plan["synthetic_samples"].items()
):

    if number_of_samples <= 0:

        continue

    print(
        f"Class {class_id}: "
        f"Generating {number_of_samples} additional..."
    )

    noise = torch.randn(
        number_of_samples,
        20,
        device=device
    )

    labels = torch.full(
        (number_of_samples,),
        class_id,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        images = cvae.decode(
            noise,
            labels
        )

    adaptive_images.append(images)
    adaptive_labels.append(labels)


# ==========================================
# CHECK IF ADDITIONAL DATA EXISTS
# ==========================================

if len(adaptive_images) == 0:

    print()
    print(
        "No additional synthetic data required."
    )

    print()
    print("GenForge has completed the cycle.")

    sys.exit()


adaptive_images = torch.cat(
    adaptive_images,
    dim=0
)

adaptive_labels = torch.cat(
    adaptive_labels,
    dim=0
)


print()
print(
    "Additional generated:",
    len(adaptive_images)
)


# ==========================================
# STEP 8: SECOND CRITIC
# ==========================================

print()
print("==========================================")
print("       STEP 8: SECOND CRITIC")
print("==========================================")

second_result = critic.evaluate(
    adaptive_images,
    adaptive_labels,
    threshold=0.90
)


print()
print("Second acceptance rate:",
      f"{second_result['acceptance_rate'] * 100:.2f}%")

print(
    "Second conditional accuracy:",
    f"{second_result['class_accuracy'] * 100:.2f}%"
)


# ==========================================
# STEP 9: COMPARE
# ==========================================

print()
print("==========================================")
print("          STEP 9: COMPARISON")
print("==========================================")

initial_acceptance = (
    initial_result["acceptance_rate"]
)

second_acceptance = (
    second_result["acceptance_rate"]
)

improvement = (
    second_acceptance -
    initial_acceptance
)


print()

print(
    "Initial acceptance:",
    f"{initial_acceptance * 100:.2f}%"
)

print(
    "Second acceptance:",
    f"{second_acceptance * 100:.2f}%"
)

print(
    "Change:",
    f"{improvement * 100:+.2f} percentage points"
)


# ==========================================
# FINAL DECISION
# ==========================================

print()
print("==========================================")
print("        GENFORGE FEEDBACK RESULT")
print("==========================================")


if improvement > 0:

    print()
    print("✓ QUALITY IMPROVED")

    print(
        "GenForge generated additional "
        "synthetic data based on Critic feedback."
    )

else:

    print()
    print("✗ QUALITY DID NOT IMPROVE")

    print(
        "GenForge should modify the generation "
        "strategy in the next iteration."
    )


print()
print("==========================================")
print("      FEEDBACK LOOP TEST COMPLETE")
print("==========================================")