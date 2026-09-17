# ========================================
# GENFORGE AUGMENTATION PLANNER
# ========================================


def create_augmentation_plan(dataset_info):

    print()
    print("========================================")
    print("       AUGMENTATION PLANNER")
    print("========================================")

    class_counts = dataset_info[
        "class_counts"
    ]

    majority_count = dataset_info[
        "majority_count"
    ]

    augmentation_needed = dataset_info[
        "augmentation_needed"
    ]

    plan = {}

    # ====================================
    # NO AUGMENTATION
    # ====================================

    if not augmentation_needed:

        print(
            "Dataset is sufficiently balanced."
        )

        for class_id, count in (
            class_counts.items()
        ):

            plan[class_id] = {

                "current_samples":
                    count,

                "target_samples":
                    count,

                "synthetic_samples":
                    0
            }

        return plan

    # ====================================
    # CREATE PLAN
    # ====================================

    total_synthetic = 0

    for class_id, current_count in (
        class_counts.items()
    ):

        target_count = majority_count

        synthetic_samples = max(
            0,
            target_count - current_count
        )

        plan[class_id] = {

            "current_samples":
                current_count,

            "target_samples":
                target_count,

            "synthetic_samples":
                synthetic_samples
        }

        total_synthetic += (
            synthetic_samples
        )

    # ====================================
    # PRINT PLAN
    # ====================================

    for class_id, data in plan.items():

        print(
            f"Class {class_id}: "
            f"{data['synthetic_samples']} "
            f"synthetic samples"
        )

    print()

    print(
        "Total synthetic samples:",
        total_synthetic
    )

    return plan