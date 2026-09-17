# GenForge-AI CVAE Audit Report

## CRITICAL ROOT CAUSE IDENTIFIED

### Problem Summary
The Conditional VAE's synthetic data quality collapsed from ~88–89% to ~33% because of a **LATENT DIMENSION MISMATCH**.

## Root Cause Analysis

### 1. **LATENT SIZE MISMATCH** (PRIMARY ISSUE)

#### Model Definition
- **models/conditional_vae.py** - Defines model with `latent_size=32` (default)
- **training/train_cvae.py** - Trains with `latent_size=32`

#### Model Checkpoint
- **models/conditional_vae.pth** - Trained with `latent_size=32`
- Decoder input expects: `latent_size (32) + label_embedding (16) = 48 features`

#### Problem: Most evaluation code uses WRONG latent_size
- **evaluation/multi_seed_experiment.py** - Uses `latent_size=20` ❌
- **evaluation/synthetic_experiment.py** - Uses `latent_size=20` ❌
- **evaluation/test_cvae_generation.py** - Hardcodes `z = torch.randn(..., 20, ...)` ❌
- **evaluation/test_critic.py** - Uses `latent_size=20` ❌
- **evaluation/test_downstream_feedback.py** - Uses `latent_size=20` ❌
- **evaluation/test_feedback_loop.py** - Uses `latent_size=20` ❌
- **evaluation/test_genforge_pipeline.py** - Uses `latent_size=20` ❌
- **evaluation/test_imbalanced_downstream.py** - Uses `latent_size=20` ❌
- **evaluation/test_imbalanced_multiseed.py** - Uses `latent_size=20` ❌
- **optimizer/auto_strategy.py** - Uses `latent_size=20` ❌
- **optimizer/downstream_optimizer.py** - Uses `latent_size=20` ❌
- **optimizer/optimizer.py** - Uses `latent_size=20` ❌

#### ONLY Correct
- **optimizer/generation_strategy.py** - Uses `latent_size=32` ✓

### 2. **What This Causes**

When generating samples:
```
Test expects:  z_shape = (N, 20)
Concatenate with label embedding (16):  combined_shape = (N, 36)

Model's decoder_input expects:  (N, 48)  # 32 + 16

Result:  RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx36 and 48x2048)
```

OR if the code somehow works around the error, it's feeding malformed input to the decoder, producing garbage synthetic samples.

### 3. **Why Accuracy is ~33%**

- Model trained with proper 32-dim latent space
- Evaluation tries to use 20-dim latent space
- The decoder receives truncated/wrong input
- Generator outputs meaningless pixels
- Classifier accuracy on garbage data ≈ random (10% base, plus some pattern matching = ~33%)

### 4. **Historical Context**

Checkpoint naming shows evolution:
```
conditional_vae_seed_123_epochs_15_latent_20.pth   ← Old: latent_20 (88-89% working)
conditional_vae_seed_42_epochs_10_latent_30.pth    ← Transition: latent_30
conditional_vae_seed_456_epochs_10_latent_30.pth   ← Transition: latent_30
conditional_vae.pth                                ← Current: trained with latent_32
```

Someone upgraded the training to use `latent_size=32` (maybe for more expressive generation), but forgot to update the evaluation/optimizer code that still references `latent_size=20`.

---

## Secondary Issues Found

### 1. **Inconsistent Image Normalization**
- **training/train_cvae.py**: Uses `transforms.Normalize((0.5,), (0.5,))` (range: [-1, 1])
- **models/conditional_vae.py**: Decoder ends with `nn.Tanh()` (outputs [-1, 1]) ✓ **Consistent**
- But evaluation code may use different ranges

### 2. **Missing Independent Evaluator Consistency**
- **evaluation/independent_evaluator.py**: Loads classifier to evaluate synthetic data
- Problem: Doesn't document what preprocessing/normalization it expects
- Could cause data format mismatches

### 3. **Critic Evaluation Issues**
- **evaluation/critic.py**: Uses classifier to evaluate synthetic samples
- Relies on classifier confidence (0.90 threshold) as primary metric
- Does NOT use independent evaluator (separate classifier trained on different data)
- This means the critic may accept poor samples because generator was trained on same classifier

### 4. **No Checkpoint Version Control**
- Model checkpoint `conditional_vae.pth` doesn't encode its latent_size
- Can't verify checkpoint compatibility at load time
- Hard to debug when wrong checkpoint is used

### 5. **LangGraph State Not Tracking Iteration**
- **orchestration/state.py**: Has `iteration: int` but nodes don't update it
- No max iteration limit implemented
- No termination condition if quality doesn't improve

### 6. **Downstream Experiment Data Leakage Risk**
- **evaluation/test_downstream_multiseed.py**: Creates evaluator dataset from first 10k samples
- Problem: Same data may be used for training if not careful with indices
- No explicit documentation of train/val/test separation

---

## Impact Summary

| Component | Status | Issue |
|-----------|--------|-------|
| CVAE Architecture | ✓ Correct | Properly designed |
| CVAE Training | ✓ Correct | Trains with latent_size=32 |
| CVAE Checkpoint | ✓ Valid | Contains proper weights |
| **Generation Code** | ❌ **BROKEN** | Uses latent_size=20 instead of 32 |
| Critic | ⚠️ Weak | Uses same classifier as training |
| Independent Evaluator | ✓ Exists | But inconsistently used |
| LangGraph Loop | ⚠️ Incomplete | No retry/cycle logic yet |

---

## Recommended Fix Order

1. **FIX LATENT SIZE MISMATCH** (CRITICAL)
   - Update all evaluation/optimizer code to use `latent_size=32`
   - Test CVAE generation quality immediately
   - Expected result: ~88–89% accuracy restored

2. **Verify Checkpoint Compatibility**
   - Add checkpoint metadata (latent_size, trained_epochs)
   - Validate on load

3. **Improve Critic**
   - Add independent evaluator as secondary quality metric
   - Don't rely solely on classifier confidence

4. **Add Iteration Tracking**
   - Update state on each node
   - Implement max iteration limit

5. **Prevent Data Leakage**
   - Document train/val/test splits clearly
   - Add assertions to prevent reuse

6. **Create Comprehensive CVAE Quality Test**
   - Per-class accuracy
   - Diversity metrics
   - Duplicate detection

---

## Files Requiring Changes

```
CRITICAL (Fix latent_size):
  evaluation/multi_seed_experiment.py
  evaluation/synthetic_experiment.py
  evaluation/test_cvae_generation.py           ← Hardcoded z dimension
  evaluation/test_critic.py
  evaluation/test_downstream_feedback.py
  evaluation/test_feedback_loop.py
  evaluation/test_genforge_pipeline.py
  evaluation/test_imbalanced_downstream.py
  evaluation/test_imbalanced_multiseed.py
  optimizer/auto_strategy.py
  optimizer/downstream_optimizer.py
  optimizer/optimizer.py

IMPORTANT (Improve robustness):
  models/conditional_vae.py                    ← Add metadata
  evaluation/critic.py
  orchestration/nodes.py                       ← Update iteration tracking
  orchestration/state.py

SHOULD CREATE:
  evaluation/test_cvae_quality.py              ← Comprehensive quality test
```

---

## Verification Steps

Once fixes are applied:

```bash
# Test 1: Quick generation
python evaluation/test_cvae_generation.py
# Expected: Class 0: ~88% correct, ..., Overall: ~88%

# Test 2: Multi-seed quality
python evaluation/test_downstream_multiseed.py
# Expected: +improvement pp > 0 with confidence

# Test 3: Independent evaluator consistency
python evaluation/test_cvae_quality.py
# Expected: ~88% synthetic accuracy

# Test 4: Full pipeline
python -m orchestration.graph
# Expected: KEEP decision, not UNCERTAIN
```

