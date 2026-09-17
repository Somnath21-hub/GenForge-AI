# ==========================================
# GenForge Strategy Planner
# ==========================================


class StrategyPlanner:

    def __init__(
        self,
        quality_threshold=0.90
    ):

        self.quality_threshold = (
            quality_threshold
        )


    # ======================================
    # Analyze Critic Result
    # ======================================

    def analyze(
        self,
        critic_result
    ):

        class_results = (
            critic_result["class_results"]
        )

        strategies = {}

        for class_id, data in class_results.items():

            accuracy = data["accuracy"]
            acceptance = data["acceptance_rate"]

            # --------------------------------
            # GOOD CLASS
            # --------------------------------

            if (
                accuracy >= self.quality_threshold
                and
                acceptance >= self.quality_threshold
            ):

                strategies[class_id] = {
                    "status": "GOOD",
                    "strategy": "KEEP"
                }

            # --------------------------------
            # SEVERE PROBLEM
            # --------------------------------

            elif acceptance < 0.75:

                strategies[class_id] = {
                    "status": "WEAK",
                    "strategy": "RETRAIN",
                    "priority": "HIGH"
                }

            # --------------------------------
            # MODERATE PROBLEM
            # --------------------------------

            elif acceptance < 0.85:

                strategies[class_id] = {
                    "status": "WEAK",
                    "strategy": "ADAPT_LATENT",
                    "priority": "MEDIUM"
                }

            # --------------------------------
            # MILD PROBLEM
            # --------------------------------

            else:

                strategies[class_id] = {
                    "status": "WEAK",
                    "strategy": "RESAMPLE",
                    "priority": "LOW"
                }


        return strategies


    # ======================================
    # Print Strategy
    # ======================================

    def show_strategy(
        self,
        strategies
    ):

        print()
        print("==========================================")
        print("        GENFORGE STRATEGY PLANNER")
        print("==========================================")

        print()

        for class_id, strategy in (
            strategies.items()
        ):

            print(
                f"Class {class_id}: "
                f"{strategy}"
            )


# ==========================================
# Standalone Test
# ==========================================

if __name__ == "__main__":

    planner = StrategyPlanner(
        quality_threshold=0.90
    )

    # Example Critic result

    critic_result = {

        "class_results": {

            0: {
                "accuracy": 0.98,
                "acceptance_rate": 0.97
            },

            1: {
                "accuracy": 0.92,
                "acceptance_rate": 0.85
            },

            2: {
                "accuracy": 0.95,
                "acceptance_rate": 0.89
            },

            4: {
                "accuracy": 0.86,
                "acceptance_rate": 0.76
            },

            5: {
                "accuracy": 0.90,
                "acceptance_rate": 0.69
            },

            8: {
                "accuracy": 0.91,
                "acceptance_rate": 0.76
            },

            9: {
                "accuracy": 0.90,
                "acceptance_rate": 0.77
            }
        }
    }


    strategies = planner.analyze(
        critic_result
    )

    planner.show_strategy(
        strategies
    )