import sys
import os

# ==========================================
# Add project root
# ==========================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch

from evaluation.classifier import CNN
from evaluation.critic import Critic
from optimizer.generation_strategy import GenerationStrategy


# ==========================================
# Configuration
# ==========================================

CLASS_ID = 5
NUMBER_OF_SAMPLES = 100
THRESHOLD = 0.90


# ==========================================
# Main
# ==========================================

def main():

    print()
    print("==========================================")
    print("       GENFORGE RESAMPLE FEEDBACK")
    print("==========================================")


    # ======================================
    # Device
    # ======================================

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print()
    print("Device:", device)


    # ======================================
    # Load classifier
    # ======================================

    classifier = CNN().to(device)

    classifier.load_state_dict(
        torch.load(
            "evaluation/baseline_cnn.pth",
            map_location=device
        )
    )

    classifier.eval()


    # ======================================
    # Create Critic
    # ======================================

    critic = Critic(
        classifier=classifier,
        device=device
    )


    # ======================================
    # Create Generation Strategy
    # ======================================

    strategy = GenerationStrategy()


    # ======================================
    # STEP 1
    # Generate original Class 5 samples
    # ======================================

    print()
    print("------------------------------------------")
    print("STEP 1: ORIGINAL GENERATION")
    print("------------------------------------------")

    original_images, original_labels = (
        strategy.resample(
            CLASS_ID,
            NUMBER_OF_SAMPLES
        )
    )


    # ======================================
    # STEP 2
    # Critic evaluates original samples
    # ======================================

    original_result = critic.evaluate(
        original_images,
        original_labels,
        threshold=THRESHOLD
    )

    original_acceptance = (
        original_result["acceptance_rate"]
    )

    original_accuracy = (
        original_result["class_accuracy"]
    )

    print()
    print("Original samples:", NUMBER_OF_SAMPLES)

    print(
        "Original acceptance:",
        round(original_acceptance * 100, 2),
        "%"
    )

    print(
        "Original accuracy:",
        round(original_accuracy * 100, 2),
        "%"
    )


    # ======================================
    # STEP 3
    # RESAMPLE
    # ======================================

    print()
    print("------------------------------------------")
    print("STEP 3: RESAMPLE")
    print("------------------------------------------")

    new_images, new_labels = (
        strategy.resample(
            CLASS_ID,
            NUMBER_OF_SAMPLES
        )
    )


    # ======================================
    # STEP 4
    # Critic evaluates new samples
    # ======================================

    new_result = critic.evaluate(
        new_images,
        new_labels,
        threshold=THRESHOLD
    )

    new_acceptance = (
        new_result["acceptance_rate"]
    )

    new_accuracy = (
        new_result["class_accuracy"]
    )

    print()
    print("New samples:", NUMBER_OF_SAMPLES)

    print(
        "New acceptance:",
        round(new_acceptance * 100, 2),
        "%"
    )

    print(
        "New accuracy:",
        round(new_accuracy * 100, 2),
        "%"
    )


    # ======================================
    # STEP 5
    # Compare
    # ======================================

    improvement = (
        new_acceptance
        - original_acceptance
    )

    print()
    print("------------------------------------------")
    print("STEP 5: COMPARISON")
    print("------------------------------------------")

    print()
    print(
        "Acceptance change:",
        round(improvement * 100, 2),
        "percentage points"
    )


    # ======================================
    # STEP 6
    # Decision
    # ======================================

    print()
    print("------------------------------------------")
    print("STEP 6: GENFORGE DECISION")
    print("------------------------------------------")

    if improvement > 0:

        print()
        print("Decision: KEEP RESAMPLE")

        print(
            "Reason: New samples improved "
            "acceptance quality."
        )

    else:

        print()
        print("Decision: REJECT RESAMPLE")

        print(
            "Reason: New samples did not "
            "improve acceptance quality."
        )


    print()
    print("==========================================")
    print("          FEEDBACK LOOP COMPLETE")
    print("==========================================")


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    main()
    