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

LATENT_SCALES = [
    0.5,
    1.0,
    1.5,
    2.0
]


# ==========================================
# Main
# ==========================================

def main():

    print()
    print("==========================================")
    print("     FAIR ADAPTIVE LATENT EXPERIMENT")
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
    # Create generation strategy
    # ======================================

    strategy = GenerationStrategy()


    # ======================================
    # SAME latent vectors for every scale
    # ======================================

    torch.manual_seed(42)

    noise = torch.randn(
        NUMBER_OF_SAMPLES,
        strategy.model.latent_size,
        device=strategy.device
    )

    labels = torch.full(
        (NUMBER_OF_SAMPLES,),
        CLASS_ID,
        dtype=torch.long,
        device=strategy.device
    )


    # ======================================
    # Baseline
    # ======================================

    print()
    print("------------------------------------------")
    print("BASELINE")
    print("------------------------------------------")

    baseline_images, _ = (
        strategy.model.decode(
            noise * 1.0,
            labels
        ),
    ) if False else (
        strategy.model.decode(
            noise,
            labels
        ),
        labels
    )

    baseline_result = critic.evaluate(
        baseline_images,
        labels,
        threshold=THRESHOLD
    )

    baseline_acceptance = (
        baseline_result["acceptance_rate"]
    )

    print(
        "Scale 1.0:",
        f"{baseline_acceptance * 100:.2f}%"
    )


    # ======================================
    # Adaptive search
    # ======================================

    print()
    print("------------------------------------------")
    print("ADAPTIVE LATENT SEARCH")
    print("------------------------------------------")

    best_scale = None
    best_acceptance = -1

    results = {}


    for scale in LATENT_SCALES:

        # IMPORTANT:
        # Same noise is used for every scale

        scaled_noise = noise * scale

        with torch.no_grad():

            images = strategy.model.decode(
                scaled_noise,
                labels
            )

        result = critic.evaluate(
            images,
            labels,
            threshold=THRESHOLD
        )

        acceptance = (
            result["acceptance_rate"]
        )

        results[scale] = acceptance


        print(
            f"Scale {scale}: "
            f"{acceptance * 100:.2f}% acceptance"
        )


        if acceptance > best_acceptance:

            best_acceptance = acceptance
            best_scale = scale


    # ======================================
    # Compare best against baseline
    # ======================================

    improvement = (
        best_acceptance
        - baseline_acceptance
    )


    print()
    print("------------------------------------------")
    print("FINAL COMPARISON")
    print("------------------------------------------")

    print()
    print(
        "Baseline acceptance:",
        f"{baseline_acceptance * 100:.2f}%"
    )

    print(
        "Best scale:",
        best_scale
    )

    print(
        "Best acceptance:",
        f"{best_acceptance * 100:.2f}%"
    )

    print(
        "Improvement:",
        f"{improvement * 100:.2f} percentage points"
    )


    # ======================================
    # GenForge decision
    # ======================================

    print()
    print("------------------------------------------")
    print("GENFORGE DECISION")
    print("------------------------------------------")

    if improvement > 0:

        print()
        print("Decision: KEEP ADAPTIVE STRATEGY")

        print(
            "GenForge found a better latent scale."
        )

    else:

        print()
        print("Decision: REJECT ADAPTIVE STRATEGY")

        print(
            "Adaptive sampling did not improve "
            "the baseline."
        )


    print()
    print("==========================================")
    print("       FAIR EXPERIMENT COMPLETE")
    print("==========================================")


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    main()