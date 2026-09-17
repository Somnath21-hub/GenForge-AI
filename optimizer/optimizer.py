import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimizer.experiment_history import ExperimentHistory


class Optimizer:

    def __init__(self):

        self.history = []

        self.best_accuracy = None
        self.best_improvement = None
        self.best_config = None

        self.config = {
            "latent_size": 32,
            "epochs": 10,
            "samples_per_class": 500,
            "critic_threshold": 0.90
        }

        self.experiment_history = ExperimentHistory()

        # ----------------------------------------------------
        # LOAD BEST EXPERIMENT FROM HISTORY
        # ----------------------------------------------------

        best = self.experiment_history.get_best_experiment()

        if best is not None:

            self.best_improvement = best["mean_improvement"]

            self.best_config = best["config"].copy()


    # ========================================================
    # RECORD MULTI-SEED EXPERIMENT
    # ========================================================

    def record_multi_seed_experiment(
        self,
        mean_improvement,
        standard_deviation,
        confidence_lower,
        confidence_upper,
        seed_results
    ):

        experiment_id = len(
            self.experiment_history.get_history()
        ) + 1

        experiment = {

            "experiment_id": experiment_id,

            "config": self.config.copy(),

            "mean_improvement": mean_improvement,

            "standard_deviation": standard_deviation,

            "confidence_interval_95": [
                confidence_lower,
                confidence_upper
            ],

            "seed_results": seed_results
        }

        self.history.append(experiment)

        self.experiment_history.add_experiment(
            experiment
        )

        # ----------------------------------------------------
        # UPDATE BEST CONFIGURATION
        # ----------------------------------------------------

        if (
            self.best_improvement is None
            or mean_improvement > self.best_improvement
        ):

            self.best_improvement = mean_improvement

            self.best_config = self.config.copy()

            print("\n===== NEW BEST CONFIGURATION =====")

            print(
                "Best improvement:",
                f"{self.best_improvement:+.2f} pp"
            )

            print(
                "Best configuration:",
                self.best_config
            )

        else:

            print("\n===== BEST CONFIGURATION UNCHANGED =====")

            print(
                "Current result:",
                f"{mean_improvement:+.2f} pp"
            )

            print(
                "Best result:",
                f"{self.best_improvement:+.2f} pp"
            )

        return experiment


    # ========================================================
    # DECIDE MULTI-SEED RESULT
    # ========================================================

    def decide_multi_seed(
        self,
        mean_improvement,
        confidence_lower,
        confidence_upper
    ):

        print("\n")
        print("========================================")
        print("        OPTIMIZER DECISION")
        print("========================================")

        print(
            f"\nCurrent improvement: "
            f"{mean_improvement:+.2f} pp"
        )

        print(
            f"95% CI: "
            f"[{confidence_lower:+.2f}, "
            f"{confidence_upper:+.2f}] pp"
        )

        print(
            f"Best improvement so far: "
            f"{self.best_improvement:+.2f} pp"
        )

        # ----------------------------------------------------
        # CHECK WHETHER CURRENT CONFIG BEAT BEST
        # ----------------------------------------------------

        if mean_improvement > self.best_improvement:

            print(
                "\nDecision: NEW BEST"
            )

            print(
                "Current configuration "
                "outperformed the previous best."
            )

            return "new_best"


        # ----------------------------------------------------
        # CURRENT CONFIG IS VALID BUT NOT BETTER
        # ----------------------------------------------------

        elif confidence_lower > 0:

            print(
                "\nDecision: PROMISING -> CONTINUE SEARCH"
            )

            print(
                "Configuration improves performance, "
                "but it did not beat the best configuration."
            )

            return "optimize"


        # ----------------------------------------------------
        # UNCERTAIN RESULT
        # ----------------------------------------------------

        elif mean_improvement > 0:

            print(
                "\nDecision: UNCERTAIN -> OPTIMIZE"
            )

            print(
                "Mean improvement is positive, "
                "but the confidence interval crosses zero."
            )

            return "optimize"


        # ----------------------------------------------------
        # NEGATIVE RESULT
        # ----------------------------------------------------

        else:

            print(
                "\nDecision: REJECT -> OPTIMIZE"
            )

            print(
                "Synthetic augmentation did not "
                "show positive improvement."
            )

            return "optimize"


    # ========================================================
    # GET NEXT CONFIGURATION
    # ========================================================

    def get_next_configuration(self):

        print("\n")
        print("========================================")
        print("       SEARCHING NEW CONFIGURATION")
        print("========================================")

        current = self.config.copy()

        # ----------------------------------------------------
        # CANDIDATE 1
        # Increase epochs
        # ----------------------------------------------------

        if current["epochs"] < 30:

            candidate = current.copy()

            candidate["epochs"] += 5

            if not self.configuration_tested(candidate):

                print(
                    "\nCandidate configuration:"
                )

                print(candidate)

                return candidate

        # ----------------------------------------------------
        # CANDIDATE 2
        # Increase latent size
        # ----------------------------------------------------

        if current["latent_size"] < 50:

            candidate = current.copy()

            candidate["latent_size"] += 10

            if not self.configuration_tested(candidate):

                print(
                    "\nCandidate configuration:"
                )

                print(candidate)

                return candidate

        # ----------------------------------------------------
        # CANDIDATE 3
        # Increase critic threshold
        # ----------------------------------------------------

        if current["critic_threshold"] < 0.95:

            candidate = current.copy()

            candidate["critic_threshold"] = round(
                current["critic_threshold"] + 0.05,
                2
            )

            if not self.configuration_tested(candidate):

                print(
                    "\nCandidate configuration:"
                )

                print(candidate)

                return candidate

        print(
            "\nNo new configuration available."
        )

        return None


    # ========================================================
    # CHECK WHETHER CONFIGURATION WAS TESTED
    # ========================================================

    def configuration_tested(self, configuration):

        history = (
            self.experiment_history
            .get_history()
        )

        for experiment in history:

            if experiment["config"] == configuration:

                return True

        return False


    # ========================================================
    # APPLY CONFIGURATION
    # ========================================================

    def apply_configuration(self, configuration):

        if configuration is None:

            print(
                "\nNo configuration to apply."
            )

            return

        self.config = configuration.copy()

        print("\n")
        print("===== CONFIGURATION APPLIED =====")

        print(
            "Latent size:",
            self.config["latent_size"]
        )

        print(
            "Epochs:",
            self.config["epochs"]
        )

        print(
            "Samples per class:",
            self.config["samples_per_class"]
        )

        print(
            "Critic threshold:",
            self.config["critic_threshold"]
        )


    # ========================================================
    # CHANGE CONFIGURATION
    # ========================================================

    def change_configuration(self):

        new_config = (
            self.get_next_configuration()
        )

        if new_config is not None:

            self.apply_configuration(
                new_config
            )

            return new_config

        return None


    # ========================================================
    # SHOW HISTORY
    # ========================================================

    def show_history(self):

        print("\n")
        print("========================================")
        print("         EXPERIMENT HISTORY")
        print("========================================")

        history = (
            self.experiment_history
            .get_history()
        )

        if len(history) == 0:

            print(
                "\nNo experiments completed."
            )

            return

        for experiment in history:

            print(
                f"\nExperiment "
                f"{experiment['experiment_id']}"
            )

            print(
                "Configuration:",
                experiment["config"]
            )

            print(
                f"Mean improvement: "
                f"{experiment['mean_improvement']:+.2f} pp"
            )

            print(
                f"Standard deviation: "
                f"{experiment['standard_deviation']:.2f} pp"
            )

            ci = (
                experiment[
                    "confidence_interval_95"
                ]
            )

            print(
                f"95% CI: "
                f"[{ci[0]:+.2f}, "
                f"{ci[1]:+.2f}] pp"
            )


    # ========================================================
    # SHOW BEST EXPERIMENT
    # ========================================================

    def show_best(self):

        print("\n")
        print("========================================")
        print("          BEST EXPERIMENT")
        print("========================================")

        best = (
            self.experiment_history
            .get_best_experiment()
        )

        if best is None:

            print(
                "\nNo experiments completed."
            )

            return

        print(
            "\nBest experiment:",
            best["experiment_id"]
        )

        print(
            "Best configuration:",
            best["config"]
        )

        print(
            f"Mean improvement: "
            f"{best['mean_improvement']:+.2f} pp"
        )

        print(
            f"Standard deviation: "
            f"{best['standard_deviation']:.2f} pp"
        )

        ci = (
            best[
                "confidence_interval_95"
            ]
        )

        print(
            f"95% CI: "
            f"[{ci[0]:+.2f}, "
            f"{ci[1]:+.2f}] pp"
        )