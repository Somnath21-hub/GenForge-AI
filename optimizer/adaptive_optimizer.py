# ========================================
# GENFORGE ADAPTIVE OPTIMIZER
# ========================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AdaptiveOptimizer:
    """
    Multi-criteria optimizer that makes intelligent decisions
    about synthetic data quality based on multiple factors.
    
    Decisions:
    - KEEP: Accept current synthetic data, finish pipeline
    - RESAMPLE: Regenerate samples with current model
    - ADAPT_LATENT: Increase latent space dimensionality
    - RETRAIN: Retrain CVAE with different hyperparameters
    - STOP: Stop optimization (max iterations reached)
    """

    def __init__(self, max_iterations=3):
        
        self.max_iterations = max_iterations
        self.iteration = 0
        self.history = []
        self.best_result = None

    # ====================================
    # EVALUATE SYNTHETIC QUALITY
    # ====================================

    def evaluate_quality(
        self,
        synthetic_accuracy,
        acceptance_rate,
        diversity_score,
        duplicate_rate
    ):
        """
        Calculate multi-metric quality score.
        
        Weights:
        - Synthetic accuracy: 40%
        - Acceptance rate: 30%
        - Diversity: 20%
        - Duplicate penalty: -10%
        """

        # Synthetic accuracy contribution
        accuracy_score = synthetic_accuracy * 0.40

        # Acceptance rate contribution
        acceptance_score = acceptance_rate * 0.30

        # Diversity contribution (lower is worse)
        diversity_contribution = min(diversity_score / 0.25, 1.0) * 0.20

        # Duplicate penalty
        duplicate_penalty = duplicate_rate * 0.10

        # Combined score (0-100)
        quality_score = (
            (accuracy_score + acceptance_score + diversity_contribution - duplicate_penalty)
            * 100
        )

        return max(0, min(100, quality_score))

    # ====================================
    # EVALUATE IMPROVEMENT
    # ====================================

    def evaluate_improvement(
        self,
        baseline_accuracy,
        augmented_accuracy,
        confidence_lower,
        confidence_upper,
        improvement_pp
    ):
        """
        Classify improvement with confidence assessment.
        
        Returns: decision reason
        """

        # Confidence interval analysis
        if confidence_lower > 0:
            return {
                "status": "IMPROVEMENT",
                "confidence": "HIGH",
                "reason": "95% CI clearly above zero"
            }

        elif confidence_upper < 0:
            return {
                "status": "HARMFUL",
                "confidence": "HIGH",
                "reason": "95% CI clearly below zero"
            }

        elif confidence_lower <= 0 <= confidence_upper:
            return {
                "status": "UNCERTAIN",
                "confidence": "LOW",
                "reason": "95% CI crosses zero"
            }

        else:
            return {
                "status": "UNKNOWN",
                "confidence": "NONE",
                "reason": "Unexpected CI values"
            }

    # ====================================
    # MAKE DECISION
    # ====================================

    def decide(
        self,
        synthetic_accuracy,
        acceptance_rate,
        diversity_score,
        duplicate_rate,
        baseline_accuracy,
        augmented_accuracy,
        confidence_lower,
        confidence_upper,
        improvement_pp,
        iteration
    ):
        """
        Make multi-criteria optimization decision.
        
        Args:
            synthetic_accuracy: Quality from critic evaluation (0-1)
            acceptance_rate: Critic acceptance rate (0-1)
            diversity_score: Pixel-level diversity metric
            duplicate_rate: Proportion of duplicates (0-1)
            baseline_accuracy: Accuracy without synthetic data
            augmented_accuracy: Accuracy with synthetic data
            confidence_lower: 95% CI lower bound (percentage points)
            confidence_upper: 95% CI upper bound (percentage points)
            improvement_pp: Mean improvement in percentage points
            iteration: Current iteration number (1-indexed)
        
        Returns:
            Dict with:
            - decision: KEEP | RESAMPLE | ADAPT_LATENT | RETRAIN | STOP
            - reason: Explanation of why
            - priority: CRITICAL | HIGH | MEDIUM | LOW
            - confidence: Quality of decision (0-1)
            - quality_score: Synthetic data quality (0-100)
            - improvement_status: IMPROVEMENT | UNCERTAIN | HARMFUL
        """

        self.iteration = iteration

        # ========================================
        # QUALITY ASSESSMENT
        # ========================================

        quality_score = self.evaluate_quality(
            synthetic_accuracy,
            acceptance_rate,
            diversity_score,
            duplicate_rate
        )

        # ========================================
        # IMPROVEMENT ASSESSMENT
        # ========================================

        improvement_assessment = self.evaluate_improvement(
            baseline_accuracy,
            augmented_accuracy,
            confidence_lower,
            confidence_upper,
            improvement_pp
        )

        improvement_status = improvement_assessment["status"]
        confidence_level = improvement_assessment["confidence"]

        # ========================================
        # DECISION LOGIC
        # ========================================

        # Priority 1: Stop if max iterations reached
        if iteration >= self.max_iterations:
            decision_dict = {
                "decision": "STOP",
                "reason": (
                    f"Maximum iterations ({self.max_iterations}) reached. "
                    f"Stopping optimization."
                ),
                "priority": "CRITICAL",
                "confidence": 1.0,
                "quality_score": quality_score,
                "improvement_status": improvement_status
            }

        # Priority 2: Accept if synthetic quality is GOOD and improvement is clear
        elif (
            quality_score > 75
            and improvement_status == "IMPROVEMENT"
            and confidence_level == "HIGH"
        ):
            decision_dict = {
                "decision": "KEEP",
                "reason": (
                    f"Synthetic quality is GOOD ({quality_score:.1f}/100) "
                    f"and improvement is statistically significant "
                    f"({improvement_pp:+.2f}pp, 95% CI [{confidence_lower:+.2f}, {confidence_upper:+.2f}])"
                ),
                "priority": "HIGH",
                "confidence": 0.95,
                "quality_score": quality_score,
                "improvement_status": improvement_status
            }

        # Priority 3: Resample if quality is moderate and we haven't tried much
        elif (
            quality_score > 50
            and acceptance_rate > 0.50
            and iteration < 2
        ):
            decision_dict = {
                "decision": "RESAMPLE",
                "reason": (
                    f"Synthetic quality is MODERATE ({quality_score:.1f}/100) "
                    f"with acceptable acceptance rate ({acceptance_rate*100:.1f}%). "
                    f"Resampling to find better latent vectors."
                ),
                "priority": "MEDIUM",
                "confidence": 0.7,
                "quality_score": quality_score,
                "improvement_status": improvement_status
            }

        # Priority 4: Adapt latent if diversity is low
        elif (
            diversity_score < 0.15
            and duplicate_rate > 0.05
            and iteration < 2
        ):
            decision_dict = {
                "decision": "ADAPT_LATENT",
                "reason": (
                    f"Diversity is LOW (score {diversity_score:.4f}) "
                    f"and duplicate rate is high ({duplicate_rate*100:.1f}%). "
                    f"Adapting latent space to improve coverage."
                ),
                "priority": "HIGH",
                "confidence": 0.8,
                "quality_score": quality_score,
                "improvement_status": improvement_status
            }

        # Priority 5: Retrain if quality is poor
        elif quality_score < 50:
            decision_dict = {
                "decision": "RETRAIN",
                "reason": (
                    f"Synthetic quality is POOR ({quality_score:.1f}/100). "
                    f"CVAE needs retraining with adjusted hyperparameters "
                    f"(beta, KL annealing, learning rate)."
                ),
                "priority": "CRITICAL",
                "confidence": 0.85,
                "quality_score": quality_score,
                "improvement_status": improvement_status
            }

        # Priority 6: Default - uncertain improvement, try resampling
        elif improvement_status == "UNCERTAIN":
            decision_dict = {
                "decision": "RESAMPLE",
                "reason": (
                    f"Improvement is UNCERTAIN ({improvement_pp:+.2f}pp) "
                    f"with 95% CI [{confidence_lower:+.2f}, {confidence_upper:+.2f}]. "
                    f"Resampling to collect more evidence."
                ),
                "priority": "MEDIUM",
                "confidence": 0.6,
                "quality_score": quality_score,
                "improvement_status": improvement_status
            }

        # Priority 7: Harmful improvement
        else:
            decision_dict = {
                "decision": "STOP",
                "reason": (
                    f"Improvement is HARMFUL ({improvement_pp:+.2f}pp) "
                    f"with 95% CI clearly below zero. "
                    f"Synthetic augmentation reduced downstream performance. "
                    f"Stopping optimization."
                ),
                "priority": "CRITICAL",
                "confidence": 0.9,
                "quality_score": quality_score,
                "improvement_status": improvement_status
            }

        # ========================================
        # RECORD HISTORY
        # ========================================

        history_entry = {
            "iteration": iteration,
            "quality_score": quality_score,
            "synthetic_accuracy": synthetic_accuracy,
            "acceptance_rate": acceptance_rate,
            "diversity_score": diversity_score,
            "duplicate_rate": duplicate_rate,
            "improvement_pp": improvement_pp,
            "ci_lower": confidence_lower,
            "ci_upper": confidence_upper,
            "improvement_status": improvement_status,
            "decision": decision_dict["decision"],
            "reason": decision_dict["reason"]
        }

        self.history.append(history_entry)

        # Update best result
        if (
            self.best_result is None
            or improvement_pp > self.best_result.get("improvement_pp", float("-inf"))
        ):
            self.best_result = history_entry

        return decision_dict

    # ====================================
    # GET DECISION REPORT
    # ====================================

    def get_report(self):
        """
        Return formatted report of all decisions.
        """

        report = {
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "best_result": self.best_result,
            "history": self.history
        }

        return report


if __name__ == "__main__":

    # Example usage
    optimizer = AdaptiveOptimizer(max_iterations=3)

    # Scenario 1: Good quality, clear improvement
    decision1 = optimizer.decide(
        synthetic_accuracy=0.75,
        acceptance_rate=0.85,
        diversity_score=0.25,
        duplicate_rate=0.01,
        baseline_accuracy=0.99,
        augmented_accuracy=0.995,
        confidence_lower=0.20,
        confidence_upper=0.50,
        improvement_pp=0.35,
        iteration=1
    )

    print("Decision 1 (Good Quality, Clear Improvement):")
    print(f"  Decision: {decision1['decision']}")
    print(f"  Reason: {decision1['reason']}")
    print(f"  Priority: {decision1['priority']}")
    print()

    # Scenario 2: Moderate quality, uncertain improvement
    decision2 = optimizer.decide(
        synthetic_accuracy=0.30,
        acceptance_rate=0.60,
        diversity_score=0.20,
        duplicate_rate=0.02,
        baseline_accuracy=0.99,
        augmented_accuracy=0.991,
        confidence_lower=-0.10,
        confidence_upper=0.30,
        improvement_pp=0.10,
        iteration=1
    )

    print("Decision 2 (Moderate Quality, Uncertain Improvement):")
    print(f"  Decision: {decision2['decision']}")
    print(f"  Reason: {decision2['reason']}")
    print(f"  Priority: {decision2['priority']}")
    print()

    # Scenario 3: Poor quality, harmful improvement
    decision3 = optimizer.decide(
        synthetic_accuracy=0.25,
        acceptance_rate=0.40,
        diversity_score=0.10,
        duplicate_rate=0.05,
        baseline_accuracy=0.99,
        augmented_accuracy=0.988,
        confidence_lower=-0.50,
        confidence_upper=-0.05,
        improvement_pp=-0.27,
        iteration=3
    )

    print("Decision 3 (Poor Quality, Harmful Improvement):")
    print(f"  Decision: {decision3['decision']}")
    print(f"  Reason: {decision3['reason']}")
    print(f"  Priority: {decision3['priority']}")
