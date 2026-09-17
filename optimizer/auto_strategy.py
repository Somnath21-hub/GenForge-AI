import os
import sys
from typing import Dict, Any, List

import torch

# Add project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from optimizer.generation_strategy import GenerationStrategy
from evaluation.classifier import CNN
from evaluation.critic import Critic


class AutoStrategy:
    """
    GenForge Autonomous Generation Strategy Optimizer & Arena.
    
    Evaluates candidate generation strategies:
    - RANDOM_PRIOR
    - LATENT_MANIFOLD (scales: 0.05, 0.10, 0.15, 0.20, 0.30)
    - ADAPTIVE_LATENT (scales: 0.5, 0.8, 1.0, 1.2, 1.5)
    
    Ranks strategies by Critic acceptance, conditional accuracy, diversity, and duplicate rate.
    """

    def __init__(
        self,
        model_path: str = "models/conditional_vae.pth",
        classifier_path: str = "evaluation/baseline_cnn.pth",
        latent_size: int = 32,
        num_classes: int = 10
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = GenerationStrategy(
            model_path=model_path,
            latent_size=latent_size,
            num_classes=num_classes,
            device=self.device
        )

        full_clf_path = os.path.join(PROJECT_ROOT, classifier_path) if not os.path.isabs(classifier_path) else classifier_path
        if not os.path.exists(full_clf_path):
            raise FileNotFoundError(f"[AutoStrategy Error] Classifier checkpoint '{full_clf_path}' not found!")

        self.classifier = CNN().to(self.device)
        self.classifier.load_state_dict(
            torch.load(full_clf_path, map_location=self.device)
        )
        self.classifier.eval()

        self.critic = Critic(
            classifier=self.classifier,
            device=self.device
        )

    def evaluate_candidate(
        self,
        strategy_name: str,
        class_id: int,
        number_of_samples: int = 100,
        scale: float = 1.0,
        threshold: float = 0.90
    ) -> Dict[str, Any]:
        """
        Generates and evaluates a specific strategy candidate.
        """
        images, labels = self.generator.generate(
            class_id=class_id,
            number_of_samples=number_of_samples,
            strategy=strategy_name,
            scale=scale
        )

        critic_result = self.critic.evaluate(
            images,
            labels,
            threshold=threshold
        )

        return {
            "strategy": strategy_name,
            "scale": scale,
            "acceptance_rate": critic_result["acceptance_rate"],
            "class_accuracy": critic_result["class_accuracy"],
            "average_confidence": critic_result["average_confidence"],
            "diversity_score": critic_result["diversity_score"],
            "duplicate_rate": critic_result["duplicate_rate"],
            "quality_score": critic_result["quality_score"],
            "accepted_count": critic_result["accepted"],
            "total_count": critic_result["total"]
        }

    def optimize_class(
        self,
        class_id: int = 5,
        number_of_samples: int = 100,
        threshold: float = 0.90
    ) -> Dict[str, Any]:
        """
        Autonomously tests strategy space and selects the best candidate for the class.
        """
        print("\n" + "=" * 60)
        print(f"GENFORGE STRATEGY ARENA (Target Class: {class_id})")
        print("=" * 60)

        candidates = []

        # 1. Random Prior Baseline
        print("\n[Arena Candidate 1] Testing RANDOM_PRIOR...")
        rp_res = self.evaluate_candidate(
            strategy_name=GenerationStrategy.STRATEGY_RANDOM_PRIOR,
            class_id=class_id,
            number_of_samples=number_of_samples,
            scale=1.0,
            threshold=threshold
        )
        candidates.append(rp_res)
        print(f"  RANDOM_PRIOR -> Acceptance: {rp_res['acceptance_rate']*100:.2f}%, Accuracy: {rp_res['class_accuracy']*100:.2f}%")

        # 2. Latent Manifold Scale Grid
        manifold_scales = [0.05, 0.10, 0.15, 0.20, 0.30]
        print("\n[Arena Candidate 2] Testing LATENT_MANIFOLD scale grid...")
        for scale in manifold_scales:
            lm_res = self.evaluate_candidate(
                strategy_name=GenerationStrategy.STRATEGY_LATENT_MANIFOLD,
                class_id=class_id,
                number_of_samples=number_of_samples,
                scale=scale,
                threshold=threshold
            )
            candidates.append(lm_res)
            print(f"  LATENT_MANIFOLD (scale={scale:.2f}) -> Acceptance: {lm_res['acceptance_rate']*100:.2f}%, Accuracy: {lm_res['class_accuracy']*100:.2f}%, Diversity: {lm_res['diversity_score']:.4f}")

        # 3. Adaptive Latent Scale Grid
        adaptive_scales = [0.5, 0.8, 1.0, 1.2, 1.5]
        print("\n[Arena Candidate 3] Testing ADAPTIVE_LATENT scale grid...")
        for scale in adaptive_scales:
            al_res = self.evaluate_candidate(
                strategy_name=GenerationStrategy.STRATEGY_ADAPTIVE_LATENT,
                class_id=class_id,
                number_of_samples=number_of_samples,
                scale=scale,
                threshold=threshold
            )
            candidates.append(al_res)
            print(f"  ADAPTIVE_LATENT (scale={scale:.2f}) -> Acceptance: {al_res['acceptance_rate']*100:.2f}%, Accuracy: {al_res['class_accuracy']*100:.2f}%")

        # Ranking: prioritize acceptance_rate and quality_score while keeping duplicates low
        best_candidate = max(
            candidates,
            key=lambda c: (c["acceptance_rate"], c["quality_score"], -c["duplicate_rate"])
        )

        baseline_acceptance = rp_res["acceptance_rate"]
        improvement = best_candidate["acceptance_rate"] - baseline_acceptance

        print("\n" + "=" * 60)
        print("GENFORGE STRATEGY ARENA OUTCOME")
        print("=" * 60)
        print(f"Winner:          {best_candidate['strategy']} (scale={best_candidate['scale']:.2f})")
        print(f"Acceptance Rate: {best_candidate['acceptance_rate']*100:.2f}% (vs Random Prior: {baseline_acceptance*100:.2f}%)")
        print(f"Improvement:     {improvement*100:+.2f} percentage points")
        print(f"Diversity Score: {best_candidate['diversity_score']:.4f}")
        print(f"Duplicate Rate:  {best_candidate['duplicate_rate']*100:.2f}%")
        print("Decision:        KEEP WINNING STRATEGY" if improvement > 0 else "Decision: REJECT")
        print("=" * 60)

        return {
            "class_id": class_id,
            "strategy": best_candidate["strategy"],
            "scale": best_candidate["scale"],
            "score": best_candidate["acceptance_rate"],
            "improvement": improvement,
            "quality_score": best_candidate["quality_score"],
            "diversity_score": best_candidate["diversity_score"],
            "duplicate_rate": best_candidate["duplicate_rate"],
            "candidates_evaluated": len(candidates),
            "all_candidates": candidates
        }


if __name__ == "__main__":
    auto = AutoStrategy()
    res = auto.optimize_class(class_id=5, number_of_samples=100, threshold=0.90)
    print("\nArena Result:")
    print(res["strategy"], "scale:", res["scale"], "score:", f"{res['score']*100:.2f}%")