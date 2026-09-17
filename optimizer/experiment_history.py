import json
import os
from datetime import datetime


class ExperimentHistory:

    def __init__(self, file_path="experiments/history.json"):

        self.file_path = file_path

        folder = os.path.dirname(self.file_path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        self.history = self.load_history()

    # -----------------------------------------
    # LOAD HISTORY
    # -----------------------------------------

    def load_history(self):

        if not os.path.exists(self.file_path):
            return []

        try:

            with open(self.file_path, "r") as file:
                data = json.load(file)

            if isinstance(data, list):
                return data

            return []

        except (json.JSONDecodeError, OSError):

            print("Warning: Could not load history.json")
            print("Starting with empty history.")

            return []

    # -----------------------------------------
    # SAVE HISTORY
    # -----------------------------------------

    def save(self):

        with open(self.file_path, "w") as file:

            json.dump(
                self.history,
                file,
                indent=4
            )

    # -----------------------------------------
    # ADD EXPERIMENT
    # -----------------------------------------

    def add_experiment(self, experiment):

        self.history.append(experiment)

        self.save()

    # -----------------------------------------
    # GET HISTORY
    # -----------------------------------------

    def get_history(self):

        return self.history

    # -----------------------------------------
    # GET EXPERIMENT BY ID
    # -----------------------------------------

    def get_experiment(self, experiment_id):

        for experiment in self.history:

            if experiment.get("experiment_id") == experiment_id:
                return experiment

        return None

    # -----------------------------------------
    # GET BEST EXPERIMENT
    # -----------------------------------------

    def get_best_experiment(self):

        if len(self.history) == 0:
            return None

        best = None
        best_score = float("-inf")

        for experiment in self.history:

            # --------------------------------
            # OLD EXPERIMENT FORMAT
            # --------------------------------

            if "mean_improvement" in experiment:

                current_score = float(
                    experiment.get(
                        "mean_improvement",
                        float("-inf")
                    )
                )

            else:

                current_score = float("-inf")

                # --------------------------------
                # NEW FORMAT: best_result
                # --------------------------------

                best_result = experiment.get(
                    "best_result"
                )

                if best_result is not None:

                    current_score = float(
                        best_result.get(
                            "improvement",
                            float("-inf")
                        )
                    )

                # --------------------------------
                # FALLBACK: iterations
                # --------------------------------

                else:

                    iterations = experiment.get(
                        "iterations",
                        []
                    )

                    for iteration in iterations:

                        improvement = float(
                            iteration.get(
                                "improvement",
                                float("-inf")
                            )
                        )

                        if improvement > current_score:

                            current_score = improvement

            # --------------------------------
            # UPDATE GLOBAL BEST
            # --------------------------------

            if current_score > best_score:

                best_score = current_score
                best = experiment

        return best

    # -----------------------------------------
    # CREATE NEW EXPERIMENT ID
    # -----------------------------------------

    def get_next_experiment_id(self):

        if len(self.history) == 0:
            return 1

        ids = []

        for experiment in self.history:

            experiment_id = experiment.get(
                "experiment_id"
            )

            if isinstance(experiment_id, int):

                ids.append(experiment_id)

        if len(ids) == 0:
            return 1

        return max(ids) + 1

    # -----------------------------------------
    # CREATE EXPERIMENT
    # -----------------------------------------

    def create_experiment(self, config):

        experiment_id = self.get_next_experiment_id()

        experiment = {

            "experiment_id": experiment_id,

            "timestamp": datetime.now().isoformat(),

            "config": config,

            "iterations": [],

            "best_result": None
        }

        self.history.append(experiment)

        self.save()

        return experiment_id

    # -----------------------------------------
    # SAVE ITERATION
    # -----------------------------------------

    def save_iteration(
        self,
        experiment_id,
        iteration_data
    ):

        experiment = self.get_experiment(
            experiment_id
        )

        if experiment is None:
            return False

        if "iterations" not in experiment:

            experiment["iterations"] = []

        experiment["iterations"].append(
            iteration_data
        )

        self.update_best_result(
            experiment
        )

        self.save()

        return True

    # -----------------------------------------
    # UPDATE BEST RESULT
    # -----------------------------------------

    def update_best_result(self, experiment):

        iterations = experiment.get(
            "iterations",
            []
        )

        if len(iterations) == 0:

            experiment["best_result"] = None

            return

        best = iterations[0]

        best_score = float(
            best.get(
                "improvement",
                float("-inf")
            )
        )

        for iteration in iterations:

            current_score = float(
                iteration.get(
                    "improvement",
                    float("-inf")
                )
            )

            if current_score > best_score:

                best = iteration
                best_score = current_score

        experiment["best_result"] = best

    # -----------------------------------------
    # COMPARE EXPERIMENTS
    # -----------------------------------------

    def compare_experiments(
        self,
        experiment_id_1,
        experiment_id_2
    ):

        experiment_1 = self.get_experiment(
            experiment_id_1
        )

        experiment_2 = self.get_experiment(
            experiment_id_2
        )

        if experiment_1 is None or experiment_2 is None:

            return None

        return {
            "experiment_1": experiment_1,
            "experiment_2": experiment_2
        }