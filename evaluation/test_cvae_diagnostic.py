# ==============================================================================
# GENFORGE CVAE COMPREHENSIVE DIAGNOSTIC SCRIPT
# ==============================================================================

import os
import sys
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision.utils as vutils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conditional_vae import ConditionalVAE
from orchestration.nodes import CNN, train_model
from evaluation.independent_evaluator import IndependentEvaluator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_evaluator_datasets(train_dataset):
    images = train_dataset.data
    labels = train_dataset.targets

    evaluator_images = (
        images[:10000].unsqueeze(1).float() / 255.0 - 0.5
    ) / 0.5
    evaluator_labels = labels[:10000].long()

    train_images = evaluator_images[:8000]
    train_labels = evaluator_labels[:8000]

    val_images = evaluator_images[8000:10000]
    val_labels = evaluator_labels[8000:10000]

    evaluator_train_dataset = TensorDataset(train_images, train_labels)
    evaluator_val_dataset = TensorDataset(val_images, val_labels)

    return evaluator_train_dataset, evaluator_val_dataset


def get_or_train_evaluator(device, data_root="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    evaluator_path = "./models/independent_evaluator.pth"
    model = CNN().to(device)

    if os.path.exists(evaluator_path):
        print(f"[Evaluator] Loading existing independent evaluator from {evaluator_path}...")
        model.load_state_dict(torch.load(evaluator_path, map_location=device))
        model.eval()
    else:
        print("[Evaluator] Training independent evaluator on first 8,000 MNIST samples (seed 999)...")
        evaluator_train, evaluator_val = create_evaluator_datasets(train_dataset)
        set_seed(999)
        model = train_model(evaluator_train, epochs=10)
        os.makedirs(os.path.dirname(evaluator_path), exist_ok=True)
        torch.save(model.state_dict(), evaluator_path)
        print(f"[Evaluator] Saved trained evaluator to {evaluator_path}")

    # Evaluate validation accuracy
    _, evaluator_val = create_evaluator_datasets(train_dataset)
    val_loader = DataLoader(evaluator_val, batch_size=256, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    val_acc = correct / total if total > 0 else 0.0
    print(f"[Evaluator] Independent Evaluator Validation Accuracy: {val_acc * 100:.2f}%\n")
    return model, val_acc, train_dataset, test_dataset


def run_cvae_diagnostic(
    cvae_path="models/conditional_vae.pth",
    output_json="experiments/cvae_diagnostic_results.json",
    output_img_dir="generated/cvae_diagnostic"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("GENFORGE CONDITIONAL VAE DIAGNOSTIC")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CVAE Checkpoint: {cvae_path}")
    print("=" * 70)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    # 1. Load Independent Evaluator
    evaluator_model, evaluator_val_acc, train_dataset, test_dataset = get_or_train_evaluator(device)

    # 2. Load CVAE
    print(f"[CVAE] Loading ConditionalVAE from {cvae_path}...")
    cvae = ConditionalVAE(latent_size=32, num_classes=10).to(device)
    cvae.load_state_dict(torch.load(cvae_path, map_location=device))
    cvae.eval()
    print("[CVAE] ConditionalVAE loaded successfully.\n")

    # =========================================================================
    # EXPERIMENT 1: 1000 SAMPLES EVALUATION (100 per class)
    # =========================================================================
    print("-" * 70)
    print("EXPERIMENT 1: CLASS-CONDITIONAL GENERATION (100 SAMPLES PER CLASS)")
    print("-" * 70)

    set_seed(42)
    samples_per_class = 100
    all_generated_images = []
    all_requested_labels = []
    all_predicted_labels = []
    all_confidences = []

    per_class_results = {}
    confusion_matrix = np.zeros((10, 10), dtype=int)

    for c in range(10):
        labels = torch.full((samples_per_class,), c, dtype=torch.long, device=device)
        noise = torch.randn(samples_per_class, 32, device=device)

        with torch.no_grad():
            gen_images = cvae.decode(noise, labels)
            logits = evaluator_model(gen_images)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

        gen_imgs_cpu = gen_images.cpu()
        preds_cpu = preds.cpu().numpy()
        confs_cpu = confs.cpu().numpy()

        all_generated_images.append(gen_imgs_cpu)
        all_requested_labels.extend([c] * samples_per_class)
        all_predicted_labels.extend(preds_cpu.tolist())
        all_confidences.extend(confs_cpu.tolist())

        # Update confusion matrix
        for p in preds_cpu:
            confusion_matrix[c, p] += 1

        correct_c = (preds_cpu == c).sum()
        acc_c = correct_c / samples_per_class
        avg_conf_c = confs_cpu.mean()

        pred_dist = {int(k): int((preds_cpu == k).sum()) for k in range(10)}

        per_class_results[c] = {
            "requested_class": c,
            "samples": samples_per_class,
            "correct": int(correct_c),
            "accuracy": float(acc_c),
            "avg_confidence": float(avg_conf_c),
            "predicted_distribution": pred_dist
        }

        # Save individual class grid (10x10)
        # Denormalize [-1, 1] to [0, 1] for visualization
        grid_c = vutils.make_grid((gen_imgs_cpu[:100] + 1) / 2.0, nrow=10, padding=2)
        vutils.save_image(grid_c, os.path.join(output_img_dir, f"class_{c}.png"))

    all_gen_tensor = torch.cat(all_generated_images, dim=0)
    all_requested = np.array(all_requested_labels)
    all_predicted = np.array(all_predicted_labels)
    all_conf = np.array(all_confidences)

    overall_accuracy = (all_requested == all_predicted).mean()
    overall_confidence = all_conf.mean()

    # Save all classes grid: 10 rows (classes 0-9), 10 columns (first 10 samples each)
    all_class_samples = []
    for c in range(10):
        all_class_samples.append(all_generated_images[c][:10])
    all_classes_tensor = torch.cat(all_class_samples, dim=0)
    grid_all = vutils.make_grid((all_classes_tensor + 1) / 2.0, nrow=10, padding=2)
    vutils.save_image(grid_all, os.path.join(output_img_dir, "all_classes.png"))

    # Pixel statistics
    pixel_min = float(all_gen_tensor.min())
    pixel_max = float(all_gen_tensor.max())
    pixel_mean = float(all_gen_tensor.mean())
    pixel_std = float(all_gen_tensor.std())

    # Overall predicted distribution
    overall_pred_counts = {int(k): int((all_predicted == k).sum()) for k in range(10)}

    print(f"\nOverall Conditional Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Overall Average Confidence:   {overall_confidence * 100:.2f}%")
    print(f"Pixel Stats: min={pixel_min:.3f}, max={pixel_max:.3f}, mean={pixel_mean:.3f}, std={pixel_std:.3f}")
    print("\nPer-Class Breakdown:")
    for c in range(10):
        res = per_class_results[c]
        print(f"  Class {c}: Acc={res['accuracy']*100:5.1f}% | AvgConf={res['avg_confidence']*100:5.1f}% | Dist={res['predicted_distribution']}")

    print("\nConfusion Matrix (Rows=Requested, Cols=Predicted):")
    print("      " + " ".join([f"{i:4d}" for i in range(10)]))
    for r in range(10):
        row_str = " ".join([f"{confusion_matrix[r, c]:4d}" for c in range(10)])
        print(f"  {r:2d}: {row_str}")

    # =========================================================================
    # EXPERIMENT 2: LABEL CONDITIONING WITH FIXED LATENT VECTORS (STEP 3)
    # =========================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: LABEL CONDITIONING INFLUENCE (SAME LATENT z, DIFFERENT LABELS)")
    print("-" * 70)

    num_test_latents = 10
    fixed_z = torch.randn(num_test_latents, 32, device=device)

    # For each latent vector z_i, decode with labels 0..9
    conditioning_results = []
    pairwise_diffs = []

    for z_idx in range(num_test_latents):
        single_z = fixed_z[z_idx:z_idx+1].repeat(10, 1)  # shape (10, 32)
        all_labels = torch.arange(10, dtype=torch.long, device=device)

        with torch.no_grad():
            decoded_digits = cvae.decode(single_z, all_labels)
            logits = evaluator_model(decoded_digits)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

        preds_list = preds.cpu().tolist()
        confs_list = confs.cpu().tolist()

        # Compute pairwise pixel differences between classes for the same z
        diff_matrix = torch.cdist(decoded_digits.view(10, -1), decoded_digits.view(10, -1), p=1) / 784.0
        mean_diff = float(diff_matrix.sum() / (10 * 9))
        pairwise_diffs.append(mean_diff)

        matches_requested = [preds_list[k] == k for k in range(10)]
        match_count = sum(matches_requested)

        conditioning_results.append({
            "latent_index": z_idx,
            "predicted_labels": preds_list,
            "confidences": [round(c, 3) for c in confs_list],
            "correct_count": match_count,
            "mean_class_pixel_diff": round(mean_diff, 4)
        })

        print(f"  Latent {z_idx:2d}: Predicted labels for classes 0..9 -> {preds_list} | Matches: {match_count}/10 | Pixel diff: {mean_diff:.4f}")

    avg_conditioning_diff = float(np.mean(pairwise_diffs))
    print(f"\nAverage pixel difference between different class labels on identical latent z: {avg_conditioning_diff:.4f}")

    # =========================================================================
    # EXPERIMENT 3: RECONSTRUCTION ON REAL MNIST (STEP 4)
    # =========================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: RECONSTRUCTION EVALUATION ON REAL MNIST DATA")
    print("-" * 70)

    # Select 10 real images per class from test_dataset
    class_real_images = {c: [] for c in range(10)}
    for img, label in test_dataset:
        if len(class_real_images[label]) < 10:
            class_real_images[label].append(img)
        if all(len(v) == 10 for v in class_real_images.values()):
            break

    real_imgs_list = []
    real_labels_list = []
    for c in range(10):
        real_imgs_list.extend(class_real_images[c])
        real_labels_list.extend([c] * 10)

    real_imgs_tensor = torch.stack(real_imgs_list).to(device)
    real_labels_tensor = torch.tensor(real_labels_list, dtype=torch.long, device=device)

    with torch.no_grad():
        recon_imgs, mu, logvar = cvae(real_imgs_tensor, real_labels_tensor)
        recon_mse = F.mse_loss(recon_imgs, real_imgs_tensor, reduction="mean").item()

        recon_logits = evaluator_model(recon_imgs)
        recon_preds = torch.argmax(recon_logits, dim=1)
        recon_correct = (recon_preds == real_labels_tensor).sum().item()
        recon_acc = recon_correct / len(real_labels_list)

    print(f"Reconstruction MSE on 100 real MNIST test samples: {recon_mse:.6f}")
    print(f"Reconstruction Classifier Accuracy: {recon_acc * 100:.2f}%")

    # Save side-by-side reconstruction grid (Original top, Reconstructed bottom for each pair)
    # Interleave: [orig_0, recon_0, orig_1, recon_1, ...]
    comparison_list = []
    for i in range(100):
        comparison_list.append(real_imgs_tensor[i:i+1].cpu())
        comparison_list.append(recon_imgs[i:i+1].cpu())
    comparison_tensor = torch.cat(comparison_list, dim=0)
    grid_recon = vutils.make_grid((comparison_tensor + 1) / 2.0, nrow=20, padding=2)
    vutils.save_image(grid_recon, os.path.join(output_img_dir, "reconstructions.png"))
    print(f"Reconstructions saved to {output_img_dir}/reconstructions.png")

    # =========================================================================
    # EXPERIMENT 4: LATENT SPACE DISTRIBUTION STATISTICS (STEP 5)
    # =========================================================================
    print("\n" + "-" * 70)
    print("EXPERIMENT 4: LATENT SPACE DISTRIBUTION ANALYSIS")
    print("-" * 70)

    # Encode full test set (10,000 samples)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    all_mus = []
    all_logvars = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mu_b, logvar_b = cvae.encode(x, y)
            all_mus.append(mu_b.cpu())
            all_logvars.append(logvar_b.cpu())

    all_mus = torch.cat(all_mus, dim=0)
    all_logvars = torch.cat(all_logvars, dim=0)
    all_stds = torch.exp(0.5 * all_logvars)

    mu_mean = float(all_mus.mean())
    mu_std = float(all_mus.std())
    logvar_mean = float(all_logvars.mean())
    logvar_std = float(all_logvars.std())
    std_mean = float(all_stds.mean())
    std_std = float(all_stds.std())

    # Per-dimension analysis
    dim_mu_means = all_mus.mean(dim=0).numpy().tolist()
    dim_mu_stds = all_mus.std(dim=0).numpy().tolist()
    dim_logvar_means = all_logvars.mean(dim=0).numpy().tolist()

    print(f"Latent Mu:     Mean = {mu_mean:+.4f}, Std = {mu_std:.4f} (Ideal: Mean~0, Std~1)")
    print(f"Latent Logvar: Mean = {logvar_mean:+.4f}, Std = {logvar_std:.4f} (Ideal: Mean~0)")
    print(f"Latent Std:    Mean = {std_mean:.4f}, Std = {std_std:.4f} (Ideal: Mean~1)")

    # =========================================================================
    # SAVE ALL RESULTS
    # =========================================================================
    results = {
        "model_path": cvae_path,
        "evaluator_val_accuracy": float(evaluator_val_acc),
        "overall_conditional_accuracy": float(overall_accuracy),
        "overall_average_confidence": float(overall_confidence),
        "pixel_statistics": {
            "min": pixel_min,
            "max": pixel_max,
            "mean": pixel_mean,
            "std": pixel_std
        },
        "overall_predicted_distribution": overall_pred_counts,
        "per_class_results": per_class_results,
        "confusion_matrix": confusion_matrix.tolist(),
        "conditioning_influence": {
            "average_pixel_difference": avg_conditioning_diff,
            "test_samples": conditioning_results
        },
        "reconstruction_evaluation": {
            "mse": float(recon_mse),
            "classifier_accuracy": float(recon_acc)
        },
        "latent_distribution": {
            "mu_mean": mu_mean,
            "mu_std": mu_std,
            "logvar_mean": logvar_mean,
            "logvar_std": logvar_std,
            "std_mean": std_mean,
            "std_std": std_std,
            "dim_mu_means": [round(v, 4) for v in dim_mu_means],
            "dim_mu_stds": [round(v, 4) for v in dim_mu_stds],
            "dim_logvar_means": [round(v, 4) for v in dim_logvar_means]
        }
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nDiagnostic results successfully saved to {output_json}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_cvae_diagnostic()
