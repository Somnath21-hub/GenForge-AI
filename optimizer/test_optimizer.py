from optimizer.optimizer import Optimizer
from evaluation.multi_seed_experiment import run_multi_seed_experiment


print("\n")
print("========================================")
print("     GENFORGE OPTIMIZATION TEST")
print("========================================")


# Create optimizer
optimizer = Optimizer()


# Show previous experiments
optimizer.show_history()


# Find next configuration
next_config = optimizer.get_next_configuration()


if next_config is None:
    print("\nNo new configuration available.")
    exit()


print("\n")
print("========================================")
print("       RUNNING NEW EXPERIMENT")
print("========================================")

print("\nConfiguration:")
print(next_config)


# Apply configuration
optimizer.apply_configuration(next_config)


# Run actual multi-seed ML experiment
result = run_multi_seed_experiment(next_config)


# Extract statistical results
mean_improvement = result["mean_improvement"]
standard_deviation = result["standard_deviation"]
confidence_lower = result["confidence_interval_95"][0]
confidence_upper = result["confidence_interval_95"][1]
seed_results = result["results"]


# Record experiment
optimizer.record_multi_seed_experiment(
    mean_improvement=mean_improvement,
    standard_deviation=standard_deviation,
    confidence_lower=confidence_lower,
    confidence_upper=confidence_upper,
    seed_results=seed_results
)


# Make optimizer decision
decision = optimizer.decide_multi_seed(
    mean_improvement,
    confidence_lower,
    confidence_upper
)


# Show updated history
optimizer.show_history()


# Show best experiment
optimizer.show_best()


print("\n")
print("========================================")
print("          OPTIMIZATION STATUS")
print("========================================")

print("\nDecision:", decision)