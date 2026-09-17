from augmentation_planner import create_augmentation_plan
from dataset_analyzer import dataset, analyze_dataset


# --------------------------------
# Run Dataset Analyzer
# --------------------------------

analysis = analyze_dataset(dataset)


# --------------------------------
# Fake Critic Result
# --------------------------------
# This represents the result we already
# received from the real Critic.

critic_result = {

    "class_results": {

        0: {
            "accuracy": 0.97,
            "acceptance_rate": 0.93
        },

        1: {
            "accuracy": 0.90,
            "acceptance_rate": 0.86
        },

        2: {
            "accuracy": 0.96,
            "acceptance_rate": 0.86
        },

        3: {
            "accuracy": 0.97,
            "acceptance_rate": 0.92
        },

        4: {
            "accuracy": 0.92,
            "acceptance_rate": 0.83
        },

        5: {
            "accuracy": 0.89,
            "acceptance_rate": 0.78
        },

        6: {
            "accuracy": 0.94,
            "acceptance_rate": 0.90
        },

        7: {
            "accuracy": 0.91,
            "acceptance_rate": 0.82
        },

        8: {
            "accuracy": 0.89,
            "acceptance_rate": 0.71
        },

        9: {
            "accuracy": 0.83,
            "acceptance_rate": 0.74
        }
    }
}


# --------------------------------
# Run Planner
# --------------------------------

plan = create_augmentation_plan(
    analysis,
    critic_result=critic_result,
    quality_threshold=0.90
)


# --------------------------------
# Display Result
# --------------------------------

print()
print("========== FINAL GENFORGE PLAN ==========")

print()

print(
    "Weak classes:",
    plan["weak_classes"]
)

print()

print(
    "Synthetic samples:",
    plan["synthetic_samples"]
)

print()

print(
    "Total synthetic samples:",
    plan["total_synthetic_samples"]
)

print()

print("==========================================")
print("Planner + Critic test completed.")
print("==========================================")