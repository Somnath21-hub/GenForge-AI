# ========================================
# CVAE DIAGNOSTIC SCRIPT
# ========================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.conditional_vae import ConditionalVAE
from evaluation.classifier import CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# Load CVAE
print("\n[1] Loading CVAE...")
cvae = ConditionalVAE(latent_size=32, num_classes=10).to(DEVICE)
cvae.load_state_dict(torch.load("models/conditional_vae.pth", map_location=DEVICE))
cvae.eval()
print("  CVAE loaded")

# Load classifier
print("\n[2] Loading classifier...")
classifier = CNN().to(DEVICE)
classifier.load_state_dict(torch.load("evaluation/baseline_cnn.pth", map_location=DEVICE))
classifier.eval()
print("  Classifier loaded")

# Load MNIST training data
print("\n[3] Loading MNIST training data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
print(f"  Loaded {len(train_dataset)} training samples")

# Test 1: Can CVAE reconstruct?
print("\n" + "="*50)
print("TEST 1: RECONSTRUCTION")
print("="*50)

with torch.no_grad():
    # Get first 100 samples
    test_images = []
    test_labels = []
    for i in range(100):
        img, label = train_dataset[i]
        test_images.append(img)
        test_labels.append(label)
    
    test_images = torch.stack(test_images).to(DEVICE)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(DEVICE)
    
    # Reconstruct
    reconstructed, mu, logvar = cvae(test_images, test_labels)
    
    # Calculate reconstruction error
    recon_error = F.mse_loss(reconstructed, test_images, reduction='mean')
    print(f"Reconstruction MSE: {recon_error:.6f}")
    print(f"(Should be low, < 0.05 for good reconstruction)")
    
    # Check if reconstruction can be classified
    recon_pred = torch.argmax(classifier(reconstructed), dim=1)
    recon_acc = (recon_pred == test_labels).float().mean().item()
    print(f"Reconstruction classification accuracy: {recon_acc*100:.2f}%")
    print(f"(Should be high, > 90% for good reconstruction)")

# Test 2: Does label conditioning affect output?
print("\n" + "="*50)
print("TEST 2: LABEL CONDITIONING")
print("="*50)

with torch.no_grad():
    # Generate 2 sets with SAME noise but different labels
    noise = torch.randn(10, 32, device=DEVICE)
    
    labels_class_0 = torch.zeros(10, dtype=torch.long, device=DEVICE)
    labels_class_3 = torch.full((10,), 3, dtype=torch.long, device=DEVICE)
    
    gen_class_0 = cvae.decode(noise, labels_class_0)
    gen_class_3 = cvae.decode(noise, labels_class_3)
    
    # Are the outputs different?
    diff = (gen_class_0 - gen_class_3).abs().mean().item()
    print(f"Mean absolute difference between class 0 and class 3 outputs: {diff:.4f}")
    print(f"(Should be > 0.01 if labels affect output)")
    
    # What do they classify as?
    pred_0 = torch.argmax(classifier(gen_class_0), dim=1)
    pred_3 = torch.argmax(classifier(gen_class_3), dim=1)
    
    class_0_as_0 = (pred_0 == 0).sum().item()
    class_3_as_3 = (pred_3 == 3).sum().item()
    
    print(f"Generated for class 0: {class_0_as_0}/10 classified as 0")
    print(f"Generated for class 3: {class_3_as_3}/10 classified as 3")
    print(f"(Both should be high if conditioning works)")

# Test 3: What does a pure random sample generate?
print("\n" + "="*50)
print("TEST 3: RANDOM LATENT SPACE")
print("="*50)

with torch.no_grad():
    # Pure random noise, all classes
    noise = torch.randn(100, 32, device=DEVICE)
    
    for target_class in range(10):
        labels = torch.full((100,), target_class, dtype=torch.long, device=DEVICE)
        generated = cvae.decode(noise, labels)
        
        pred = torch.argmax(classifier(generated), dim=1)
        acc = (pred == target_class).float().mean().item()
        
        print(f"Class {target_class}: {acc*100:.2f}%")

# Test 4: Compare to training data baseline
print("\n" + "="*50)
print("TEST 4: TRAINING DATA BASELINE")
print("="*50)

with torch.no_grad():
    # Classify real MNIST training data
    loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    batch_images, batch_labels = next(iter(loader))
    batch_images = batch_images.to(DEVICE)
    batch_labels = batch_labels.to(DEVICE)
    
    pred = torch.argmax(classifier(batch_images), dim=1)
    real_acc = (pred == batch_labels).float().mean().item()
    
    print(f"Classifier accuracy on REAL MNIST training data: {real_acc*100:.2f}%")
    print(f"(Should be very high, > 95%)")

print("\n" + "="*50)
print("DIAGNOSTIC COMPLETE")
print("="*50)
