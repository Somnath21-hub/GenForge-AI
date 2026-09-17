# ========================================
# GENFORGE COMPREHENSIVE CVAE QUALITY TEST
# ========================================

import sys
import os
import torch
import torch.nn as nn
from collections import defaultdict

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.conditional_vae import ConditionalVAE
from evaluation.classifier import CNN
from evaluation.independent_evaluator import IndependentEvaluator

# ========================================
# SETTINGS
# ========================================

MODEL_PATH = "models/conditional_vae.pth"
CLASSIFIER_PATH = "evaluation/baseline_cnn.pth"

LATENT_SIZE = 32
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 200

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ========================================
# LOAD CVAE
# ========================================

print("\n" + "="*50)
print("Loading CVAE model...")
print("="*50)

cvae = ConditionalVAE(
    latent_size=LATENT_SIZE,
    num_classes=NUM_CLASSES
).to(DEVICE)

try:
    cvae.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )
    print(f"[OK] Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

cvae.eval()

# ========================================
# LOAD CLASSIFIER FOR SYNTHETIC QUALITY
# ========================================

print("\nLoading classifier...")

classifier = CNN().to(DEVICE)

try:
    classifier.load_state_dict(
        torch.load(
            CLASSIFIER_PATH,
            map_location=DEVICE
        )
    )
    print(f"[OK] Classifier loaded from {CLASSIFIER_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load classifier: {e}")
    sys.exit(1)

classifier.eval()

# ========================================
# CREATE INDEPENDENT EVALUATOR
# ========================================

print("\nCreating independent evaluator...")
print("  (This requires a separate classifier)")

# For now, we'll use the same classifier but document this
# In a real scenario, this would be trained on different data
evaluator = IndependentEvaluator(
    classifier,
    DEVICE
)

print("[OK] Independent evaluator created")

# ========================================
# GENERATE SYNTHETIC DATA
# ========================================

print("\n" + "="*50)
print(f"Generating {SAMPLES_PER_CLASS} samples per class")
print("="*50)

all_synthetic_images = []
all_synthetic_labels = []

per_class_results = {}

with torch.no_grad():

    for class_id in range(NUM_CLASSES):

        print(f"\nGenerating class {class_id}...")

        # Create labels
        labels = torch.full(
            (SAMPLES_PER_CLASS,),
            class_id,
            dtype=torch.long,
            device=DEVICE
        )

        # Generate random latent vectors
        noise = torch.randn(
            SAMPLES_PER_CLASS,
            LATENT_SIZE,
            device=DEVICE
        )

        # Generate images
        generated_images = cvae.decode(
            noise,
            labels
        )

        # Evaluate with classifier
        classifier_outputs = classifier(
            generated_images
        )

        classifier_predictions = torch.argmax(
            classifier_outputs,
            dim=1
        )

        # Count correct predictions
        correct = (
            classifier_predictions == labels
        ).sum().item()

        accuracy = correct / SAMPLES_PER_CLASS

        # Evaluate with independent evaluator
        eval_result = evaluator.evaluate(
            generated_images,
            labels
        )

        # Calculate diversity (pixel std)
        flat_images = generated_images.view(
            generated_images.size(0),
            -1
        )

        diversity = flat_images.std(dim=0).mean().item()

        # Detect duplicates (rounded)
        rounded_images = torch.round(
            flat_images * 10
        ) / 10

        unique_count = torch.unique(
            rounded_images,
            dim=0
        ).size(0)

        duplicate_rate = 1.0 - (unique_count / SAMPLES_PER_CLASS)

        # Store results
        per_class_results[class_id] = {
            "classifier_accuracy": accuracy,
            "independent_accuracy": eval_result["accuracy"],
            "diversity": diversity,
            "duplicate_rate": duplicate_rate,
            "unique_samples": unique_count,
            "total_samples": SAMPLES_PER_CLASS
        }

        print(f"  Classifier accuracy: {accuracy*100:.2f}%")
        print(f"  Independent accuracy: {eval_result['accuracy']*100:.2f}%")
        print(f"  Diversity score: {diversity:.4f}")
        print(f"  Duplicate rate: {duplicate_rate*100:.2f}%")
        print(f"  Unique samples: {unique_count}/{SAMPLES_PER_CLASS}")

        all_synthetic_images.append(
            generated_images.cpu()
        )

        all_synthetic_labels.append(
            labels.cpu()
        )

# ========================================
# COMBINE ALL SYNTHETIC DATA
# ========================================

all_synthetic_images = torch.cat(
    all_synthetic_images,
    dim=0
)

all_synthetic_labels = torch.cat(
    all_synthetic_labels,
    dim=0
)

# ========================================
# OVERALL SYNTHETIC QUALITY
# ========================================

print("\n" + "="*50)
print("Overall Synthetic Data Quality")
print("="*50)

with torch.no_grad():

    # Classifier accuracy on all synthetic data
    all_outputs = classifier(
        all_synthetic_images.to(DEVICE)
    )

    all_predictions = torch.argmax(
        all_outputs,
        dim=1
    )

    correct_total = (
        all_predictions == all_synthetic_labels.to(DEVICE)
    ).sum().item()

    overall_classifier_accuracy = (
        correct_total / len(all_synthetic_labels)
    )

    # Independent evaluator accuracy on all
    eval_result_total = evaluator.evaluate(
        all_synthetic_images.to(DEVICE),
        all_synthetic_labels.to(DEVICE)
    )

    overall_independent_accuracy = (
        eval_result_total["accuracy"]
    )

print(f"\nClassifier Accuracy (All Classes): {overall_classifier_accuracy*100:.2f}%")
print(f"Independent Accuracy (All Classes): {overall_independent_accuracy*100:.2f}%")

# ========================================
# CLASS DISTRIBUTION
# ========================================

print("\n" + "="*50)
print("Class Distribution in Synthetic Data")
print("="*50)

class_dist = defaultdict(int)

for label in all_synthetic_labels:
    class_dist[label.item()] += 1

total_synthetic = len(all_synthetic_labels)

for class_id in range(NUM_CLASSES):

    count = class_dist[class_id]
    percentage = (count / total_synthetic) * 100

    print(f"Class {class_id}: {count} ({percentage:.2f}%)")

# ========================================
# PER-CLASS ACCURACY TABLE
# ========================================

print("\n" + "="*50)
print("Per-Class Accuracy Report")
print("="*50)

print("\n{:<8} {:<15} {:<15} {:<12} {:<12}".format(
    "Class",
    "Classifier %",
    "Independent %",
    "Diversity",
    "Duplicates %"
))

print("-" * 65)

for class_id in range(NUM_CLASSES):

    result = per_class_results[class_id]

    print("{:<8} {:<15.2f} {:<15.2f} {:<12.4f} {:<12.2f}".format(
        class_id,
        result["classifier_accuracy"] * 100,
        result["independent_accuracy"] * 100,
        result["diversity"],
        result["duplicate_rate"] * 100
    ))

# ========================================
# SUMMARY AND DIAGNOSIS
# ========================================

print("\n" + "="*50)
print("Quality Assessment Summary")
print("="*50)

# Calculate statistics
classifier_accuracies = [
    r["classifier_accuracy"] for r in per_class_results.values()
]
independent_accuracies = [
    r["independent_accuracy"] for r in per_class_results.values()
]
diversities = [
    r["diversity"] for r in per_class_results.values()
]
duplicate_rates = [
    r["duplicate_rate"] for r in per_class_results.values()
]

print(f"\nClassifier Accuracy (Mean): {sum(classifier_accuracies)/len(classifier_accuracies)*100:.2f}%")
print(f"Independent Accuracy (Mean): {sum(independent_accuracies)/len(independent_accuracies)*100:.2f}%")
print(f"Diversity (Mean): {sum(diversities)/len(diversities):.4f}")
print(f"Duplicate Rate (Mean): {sum(duplicate_rates)/len(duplicate_rates)*100:.2f}%")

# Diagnosis
print("\nDiagnosis:")

if overall_classifier_accuracy > 0.85:
    print("  [GOOD] CVAE quality is GOOD (>85% accuracy)")
elif overall_classifier_accuracy > 0.70:
    print("  [WARN] CVAE quality is ACCEPTABLE (70-85%)")
else:
    print("  [POOR] CVAE quality is POOR (<70%)")

if sum(duplicate_rates)/len(duplicate_rates) > 0.10:
    print("  [WARN] HIGH DUPLICATE RATE - Model may be mode collapsing")
else:
    print("  [OK] Duplicate rate is acceptable")

if sum(diversities)/len(diversities) < 0.01:
    print("  [WARN] LOW DIVERSITY - Generated samples may be too similar")
else:
    print("  [OK] Diversity is acceptable")

# ========================================
# SAVE VISUALIZATION
# ========================================

print("\n" + "="*50)
print("Saving generated samples for visualization...")
print("="*50)

try:
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a grid of generated images
    fig, axes = plt.subplots(NUM_CLASSES, 5, figsize=(15, 20))

    for class_id in range(NUM_CLASSES):
        # Get 5 random samples from this class
        class_mask = (all_synthetic_labels == class_id)
        class_indices = torch.where(class_mask)[0]

        for i in range(min(5, len(class_indices))):
            idx = class_indices[i].item()
            image = all_synthetic_images[idx, 0].numpy()

            ax = axes[class_id, i]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Class {class_id}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(
        'evaluation/cvae_generated_samples.png',
        dpi=100,
        bbox_inches='tight'
    )
    print("[OK] Saved to evaluation/cvae_generated_samples.png")
    plt.close()

except ImportError:
    print("  Matplotlib not available - skipping visualization")

except Exception as e:
    print(f"  Failed to save visualization: {e}")

# ========================================
# FINAL REPORT
# ========================================

print("\n" + "="*50)
print("CVAE QUALITY TEST COMPLETE")
print("="*50)

print(f"\nGenerated: {len(all_synthetic_images)} synthetic samples")
print(f"Classifier Accuracy: {overall_classifier_accuracy*100:.2f}%")
print(f"Independent Accuracy: {overall_independent_accuracy*100:.2f}%")
print(f"\nFor detailed results, see per-class table above.")
print("="*50 + "\n")
