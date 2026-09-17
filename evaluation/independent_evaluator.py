# ========================================
# GENFORGE INDEPENDENT EVALUATOR
# ========================================

import torch


class IndependentEvaluator:

    def __init__(
        self,
        model,
        device
    ):

        self.model = model
        self.device = device

    # ====================================
    # EVALUATE SYNTHETIC DATA
    # ====================================

    def evaluate(
        self,
        images,
        labels
    ):

        self.model.eval()

        images = images.to(
            self.device
        )

        labels = labels.to(
            self.device
        )

        with torch.no_grad():

            outputs = self.model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

        correct = (
            predictions == labels
        ).sum().item()

        total = len(labels)

        if total > 0:

            accuracy = (
                correct / total
            )

        else:

            accuracy = 0.0

        return {

            "total":
                total,

            "correct":
                correct,

            "accuracy":
                accuracy
        }