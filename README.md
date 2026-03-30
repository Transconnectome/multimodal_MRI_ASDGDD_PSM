# Brain Microstructural Discriminability Between ASD and GDD

**A Confounder-Controlled Deep Learning Study**

This repository contains the code for classifying Autism Spectrum Disorder (ASD) vs. Global Developmental Delay (GDD) using multimodal MRI (T1-weighted + DTI) with propensity score-matched evaluation and statistically validated explainability analysis.

## Key Features

- **3D DenseNet-121** with three fusion strategies: unimodal, multi-channel, multi-fusion
- **Propensity score matching (PSM)** within nested cross-validation for confounder-isolated evaluation
- **Integrated Gradients + SmoothGrad** for statistically validated saliency maps
- **Tract overlap metrics** (PSVP, TOP) against the HCP 1065 white matter atlas
- **Included atlases**: HCP 1065 tract atlas and infant MNI template (33-44 months)

## Repository Structure

```
.
├── train.py                  # Main training with nested CV + PSM
├── inference.py              # Model evaluation on test sets
├── run_ig.py                 # Integrated Gradients extraction
├── config.py                 # Dataset configuration
├── models/
│   ├── densenet3d.py         # 3D DenseNet-121 architecture variants
│   └── densenetYA.py         # DenseNet-121 for training + IG wrapper
├── dataloaders/
│   ├── dataloaders.py        # Dataset construction + PSM matching
│   ├── custom_dataset.py     # Multi-channel/multi-modal datasets
│   ├── data_utils.py         # PSM + iterative stratification
│   ├── preprocessing.py      # Metadata preprocessing
│   └── custom_transform.py   # Custom image transforms
├── envs/
│   ├── experiments.py        # Train/validate/test loops
│   └── loss_functions.py     # Metrics (AUROC, dCor) + losses
├── xai/
│   ├── custom_attribution.py # IG + SmoothGrad implementation
│   ├── models_wrapper.py     # Model wrapper for IG computation
│   ├── config.json           # XAI hyperparameters
│   └── run_xai_pipeline.py   # Full XAI pipeline (Steps 2-7)
├── utils/
│   ├── utils.py              # Argument parsing, checkpointing
│   └── optimizer.py          # Cosine annealing warm restarts
├── data/
│   └── atlases/              # Included atlas and template files
│       ├── alltracts.nii.gz          # HCP 1065 87-tract atlas (infant MNI space)
│       ├── hcp1065_abbreviation.txt  # Tract names and abbreviations
│       └── infant_MNI_template.nii.gz # Infant MNI template (33-44 months)
└── scripts/
    ├── train_unimodal.sh     # Example: single-modality training
    ├── train_multichannel.sh # Example: multi-channel (best model)
    ├── train_multifusion.sh  # Example: multi-fusion training
    ├── run_ig_extraction.sh  # Example: IG attribution extraction
    └── run_xai_pipeline.sh   # Complete XAI pipeline
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional (for publication-quality statistical testing):**
```bash
# AFNI — voxel-wise t-test with Monte Carlo cluster correction
# https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/index.html

# ANTs — non-linear registration to infant MNI space
# https://github.com/ANTsX/ANTs
```

### 2. Prepare Data

```bash
# Set data root (directory containing CHA/ subdirectory)
export DATA_ROOT=/path/to/your/data

# Expected structure:
# DATA_ROOT/CHA/
#   ├── FA/             # Fractional Anisotropy .nii.gz
#   ├── MD/             # Mean Diffusivity
#   ├── RD/             # Radial Diffusivity
#   ├── AD/             # Axial Diffusivity
#   ├── sMRI_brain/     # T1-weighted brain-extracted .nii.gz
#   └── metadata/
#       └── ASDGDD_QC_quotient.csv  # Subject phenotypes
```

### 3. Train a Model

```bash
# Best model: multi-channel with all DTI metrics (FA+MD+RD+AD)
bash scripts/train_multichannel.sh mc_4DTI

# Unimodal (single metric)
bash scripts/train_unimodal.sh unimodal_FA FA

# Multi-fusion (separate encoders)
bash scripts/train_multifusion.sh mf_T1w_FA
```

### 4. Run XAI Pipeline

```bash
# Step 1: IG attribution extraction (GPU required)
bash scripts/run_ig_extraction.sh mc_4DTI_01

# Steps 2-7: Group averaging → tract overlap metrics
bash scripts/run_xai_pipeline.sh mc_4DTI_01 ./xai_results
```

## Model Architectures

| Type | Input | Parameters | Description |
|------|-------|-----------|-------------|
| Unimodal | 1 channel | ~11.2M | Single DTI metric or T1w |
| Multi-channel | 4 channels | ~11.3M | FA+MD+RD+AD concatenated |
| Multi-fusion | 2 encoders | ~22.6M | Separate encoders + L2 alignment |

Architecture: 3D DenseNet-121 (block_config=(6,12,24,16), growth_rate=32, compression=0.5).

## Experimental Design

### Nested Cross-Validation with PSM

```
Outer loop (6 test sets from 3 seeds × 2 splits):
  1. PSM matching on age + BSID quotients → 100 matched subjects (50 ASD, 50 GDD)
  2. Each test set: N=50 (25 ASD, 25 GDD)
  3. Training pool: 307 subjects (54 ASD, 253 GDD) per fold

Inner loop (4 folds):
  Grid search: LR ∈ {1e-2, 1e-3, 1e-4} × WD ∈ {1e-2, 1e-3, 1e-4}
  Best config selected by validation AUROC + AP

Total: 24 model evaluations per architecture (6 outer × 4 inner)
```

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Max epochs | 300 |
| Early stopping | 100 epochs (on AUROC + AP) |
| Scheduler | Cosine annealing warm restarts (T₀=20, T_mult=2, warmup=5) |
| Batch size | 16 nominal, 64 effective (4-step gradient accumulation) |
| Input size | 138³ (~1.5mm isotropic, from 206³ padded space) |
| Augmentation | Random affine (rotation 0-30°, translation ≤10 voxels, p=0.5) |

### PSM Configuration

| Parameter | Value |
|-----------|-------|
| Library | PsmPy |
| Algorithm | k-d tree nearest-neighbor |
| Matching | On propensity logit, without replacement |
| Caliper | 1.0 logit units |
| Covariates | Age at MRI, BSID Mental DQ, BSID Motor DQ |
| Sex | Not included (groups did not differ; χ²=1.49, p=0.22) |

## XAI Pipeline

| Step | Description | Tool |
|------|-------------|------|
| 1 | Integrated Gradients + SmoothGrad (σ=0.05, n=5) | Captum / `run_ig.py` |
| 2 | Sign-aware group averaging (TP/TN, positive/negative) | numpy / `xai/run_xai_pipeline.py` |
| 3 | Inverse transform (138³ → 206³) | scipy |
| 4 | Registration to infant MNI space (195×233×159) | ANTs |
| 5 | Voxel-wise one-sample t-test (p<0.001) | AFNI or scipy |
| 6 | Cluster correction (α=0.01) | AFNI or scipy |
| 7 | PSVP (top 1,000) and TOP (top 10,000) tract overlap | numpy |

Atlases for Steps 4 and 7 are included in `data/atlases/`.

## License

MIT License. See [LICENSE](LICENSE) for details.
