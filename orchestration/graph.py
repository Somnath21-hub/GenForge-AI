# ========================================
# GENFORGE LANGGRAPH GRAPH
# ========================================

from langgraph.graph import (
    StateGraph,
    END
)

from orchestration.state import GenForgeState

from orchestration.nodes import (
    analyzer_node,
    planner_node,
    generation_node,
    critic_node,
    evaluation_node,
    optimizer_node
)

from optimizer.experiment_history import (
    ExperimentHistory
)


# ========================================
# ROUTER
# ========================================

def route_optimizer_decision(state):

    decision_data = state.get(
        "optimizer_decision",
        {}
    )

    decision = decision_data.get(
        "decision",
        "KEEP"
    )

    iteration = state.get(
        "iteration",
        1
    )

    max_iterations = state.get(
        "max_iterations",
        3
    )

    # ------------------------------------
    # KEEP = successful experiment
    # ------------------------------------

    if decision == "KEEP":

        return END

    # ------------------------------------
    # Maximum iteration protection
    #
    # IMPORTANT:
    # optimizer increments iteration
    # after every completed iteration.
    # Therefore use >, not >=.
    # ------------------------------------

    if iteration > max_iterations:

        return END

    # ------------------------------------
    # RETRAIN
    # ------------------------------------

    if decision == "RETRAIN":

        return "generation"

    # ------------------------------------
    # ADAPT LATENT
    # ------------------------------------

    if decision == "ADAPT_LATENT":

        return "generation"

    # ------------------------------------
    # RESAMPLE
    # ------------------------------------

    if decision == "RESAMPLE":

        return "generation"

    # ------------------------------------
    # SAFETY FALLBACK
    # ------------------------------------

    return END


# ========================================
# BUILD GRAPH
# ========================================

def build_graph():

    graph = StateGraph(
        GenForgeState
    )

    # ------------------------------------
    # ADD NODES
    # ------------------------------------

    graph.add_node(
        "analyzer",
        analyzer_node
    )

    graph.add_node(
        "planner",
        planner_node
    )

    graph.add_node(
        "generation",
        generation_node
    )

    graph.add_node(
        "critic",
        critic_node
    )

    graph.add_node(
        "evaluation",
        evaluation_node
    )

    graph.add_node(
        "optimizer",
        optimizer_node
    )

    # ------------------------------------
    # ENTRY
    # ------------------------------------

    graph.set_entry_point(
        "analyzer"
    )

    # ------------------------------------
    # NORMAL FLOW
    # ------------------------------------

    graph.add_edge(
        "analyzer",
        "planner"
    )

    graph.add_edge(
        "planner",
        "generation"
    )

    graph.add_edge(
        "generation",
        "critic"
    )

    graph.add_edge(
        "critic",
        "evaluation"
    )

    graph.add_edge(
        "evaluation",
        "optimizer"
    )

    # ------------------------------------
    # CONDITIONAL OPTIMIZER ROUTING
    # ------------------------------------

    graph.add_conditional_edges(

        "optimizer",

        route_optimizer_decision,

        {

            "generation":
                "generation",

            END:
                END
        }
    )

    return graph.compile()


# ========================================
# RUN GRAPH
# ========================================

def run_genforge():

    history_manager = ExperimentHistory()

    # ------------------------------------
    # CREATE NEW EXPERIMENT
    # ------------------------------------

    config = {

        "dataset": "MNIST",

        "generator":
            "ConditionalVAE",

        "latent_size":
            32,

        "critic_threshold":
            0.90,

        "max_iterations":
            3,

        "development_class_5_limit":
            1000
    }

    experiment_id = (
        history_manager.create_experiment(
            config
        )
    )

    print()
    print(
        "Experiment ID:",
        experiment_id
    )

    # ------------------------------------
    # INITIAL STATE
    # ------------------------------------

    initial_state = {

        "dataset_info": {},

        "augmentation_plan": {},

        "generation_result": {},

        "critic_result": {},

        "evaluation_result": {},

        "optimizer_decision": {},

        "iteration": 1,

        "max_iterations": 3,

        "experiment_id":
            experiment_id,

        "iteration_history": [],

        "best_result": None,

        "termination_reason": None
    }

    # ------------------------------------
    # BUILD GRAPH
    # ------------------------------------

    app = build_graph()

    # ------------------------------------
    # EXECUTE
    # ------------------------------------

    final_state = app.invoke(
        initial_state
    )

    # ------------------------------------
    # FINAL SUMMARY
    # ------------------------------------

    print()
    print("========================================")
    print("GENFORGE EXPERIMENT COMPLETE")
    print("========================================")

    print(
        "Experiment ID:",
        experiment_id
    )

    print(
        "Iterations completed:",
        len(
            final_state.get(
                "iteration_history",
                []
            )
        )
    )

    print(
        "Termination reason:",
        final_state.get(
            "termination_reason"
        )
    )

    print()
    print("Iteration history:")

    for item in final_state.get(
        "iteration_history",
        []
    ):

        print(
            "Iteration",
            item.get("iteration"),
            "| Decision:",
            item.get("decision"),
            "| Improvement:",
            f"{item.get('improvement', 0) * 100:.2f} pp"
        )

    print()

    best_result = final_state.get(
        "best_result"
    )

    if best_result is not None:

        print(
            "Best iteration:",
            best_result.get(
                "iteration"
            )
        )

        print(
            "Best improvement:",
            f"{best_result.get('improvement', 0) * 100:.2f} pp"
        )

    return final_state


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":

    run_genforge()