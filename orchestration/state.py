# ========================================
# GENFORGE LANGGRAPH STATE
# ========================================

from typing import TypedDict, Any, List, Dict, Optional


class GenForgeState(TypedDict):

    # Core pipeline results
    dataset_info: dict
    augmentation_plan: dict
    generation_result: dict
    critic_result: dict
    evaluation_result: dict
    optimizer_decision: dict

    # Iteration tracking
    iteration: int
    max_iterations: int

    # Experiment tracking
    experiment_id: int

    # History of all iterations
    iteration_history: List[Dict[str, Any]]

    # Best result so far
    best_result: Optional[Dict[str, Any]]

    # Termination reason
    termination_reason: Optional[str]