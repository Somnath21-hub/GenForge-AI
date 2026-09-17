# ========================================
# GENFORGE SYNTHETIC DATA CRITIC
# ========================================

import torch


class Critic:

    def __init__(
        self,
        classifier,
        device,
        independent_evaluator=None
    ):

        self.classifier = classifier

        self.device = device
        
        # Optional: separate evaluator for quality verification
        self.independent_evaluator = independent_evaluator


    # ====================================
    # EVALUATE
    # ====================================

    def evaluate(
        self,
        images,
        requested_labels,
        threshold=0.90
    ):

        self.classifier.eval()

        images = images.to(
            self.device
        )

        requested_labels = (
            requested_labels
            .to(self.device)
            .long()
        )

        # =================================
        # CLASSIFICATION
        # =================================

        with torch.no_grad():

            outputs = self.classifier(
                images
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_labels = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

        # =================================
        # CORRECT CLASS
        # =================================

        correct_class = (

            predicted_labels

            ==

            requested_labels
        )

        # =================================
        # HIGH CONFIDENCE
        # =================================

        high_confidence = (

            confidence

            >=

            threshold
        )

        # =================================
        # ACCEPT
        # =================================

        accepted = (

            correct_class

            &

            high_confidence
        )

        # =================================
        # BASIC METRICS
        # =================================

        total_images = len(
            requested_labels
        )

        accepted_count = (
            accepted.sum().item()
        )

        rejected_count = (
            total_images
            -
            accepted_count
        )

        acceptance_rate = (

            accepted.float()
            .mean()
            .item()
        )

        class_accuracy = (

            correct_class
            .float()
            .mean()
            .item()
        )

        average_confidence = (
            confidence
            .mean()
            .item()
        )

        # =================================
        # DIVERSITY
        # =================================

        flattened_images = (
            images.view(
                images.size(0),
                -1
            )
        )

        # Pixel-level diversity (std of mean pixel values)
        pixel_means = flattened_images.mean(dim=0)
        diversity_score = (
            flattened_images
            .std(dim=0)
            .mean()
            .item()
        )

        # =================================
        # DUPLICATES
        # =================================

        rounded_images = (
            torch.round(
                flattened_images * 10
            ) / 10
        )

        unique_images = (
            torch.unique(
                rounded_images,
                dim=0
            ).size(0)
        )

        duplicate_rate = (

            1
            -
            (
                unique_images
                /
                total_images
            )
        )

        # =================================
        # INDEPENDENT EVALUATION (Optional)
        # =================================
        
        independent_accuracy = None
        
        if self.independent_evaluator is not None:
            
            eval_result = (
                self.independent_evaluator.evaluate(
                    images,
                    requested_labels
                )
            )
            
            independent_accuracy = (
                eval_result["accuracy"]
            )

        # =================================
        # CLASS RESULTS
        # =================================

        class_results = {}

        number_of_classes = 10

        for digit in range(
            number_of_classes
        ):

            mask = (
                requested_labels
                ==
                digit
            )

            class_total = (
                mask.sum().item()
            )

            if class_total == 0:

                continue

            digit_correct = (
                correct_class[mask]
            )

            digit_accepted = (
                accepted[mask]
            )

            digit_confidence = (
                confidence[mask]
            )

            class_results[digit] = {

                "total":
                    class_total,

                "correct":
                    digit_correct.sum().item(),

                "accepted":
                    digit_accepted.sum().item(),

                "rejected":
                    (
                        ~digit_accepted
                    ).sum().item(),

                "accuracy":
                    digit_correct
                    .float()
                    .mean()
                    .item(),

                "acceptance_rate":
                    digit_accepted
                    .float()
                    .mean()
                    .item(),

                "average_confidence":
                    digit_confidence
                    .mean()
                    .item()
            }

        # =================================
        # QUALITY SCORE
        # =================================
        
        # Simple quality metric combining accuracy and acceptance
        quality_score = (
            class_accuracy * 0.6 +
            acceptance_rate * 0.4
        )

        # =================================
        # FINAL RESULT
        # =================================

        result = {

            "total":
                total_images,

            "accepted":
                accepted_count,

            "rejected":
                rejected_count,

            "acceptance_rate":
                acceptance_rate,

            "average_confidence":
                average_confidence,

            "class_accuracy":
                class_accuracy,

            "independent_accuracy":
                independent_accuracy,

            "diversity_score":
                diversity_score,

            "unique_images":
                unique_images,

            "duplicate_rate":
                duplicate_rate,
            
            "quality_score":
                quality_score,

            "class_results":
                class_results
        }

        return result