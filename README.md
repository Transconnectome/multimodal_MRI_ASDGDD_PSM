# Brain Microstructural Discriminability Between ASD and GDD

**A Confounder-Controlled Deep Learning Study**

This repository contains the code for classifying Autism Spectrum Disorder (ASD) vs. Global Developmental Delay (GDD) using multimodal MRI (T1-weighted + DTI) with propensity score-matched evaluation and statistically validated explainability analysis.

## Key Features

- **3D DenseNet-121** with three fusion strategies: unimodal, multi-channel, multi-fusion
- **Propensity score matching (PSM)** within nested cross-validation for confounder-isolated evaluation
- **Integrated Gradients + SmoothGrad** for statistically validated saliency maps
- **Tract overlap metrics** (PSVP, TOP) against the HCP 1065 white matter atlas

## Repository Structure

```
.
├── train.py                  # Main training with nested CV + PSM
├── inference.py              # Model evaluation on test sets
├── run_ig.py                 # Integrated Gradients extraction (Step 1)
├── config.py                 # Dataset configuration
├── models/
│   └── densenet3d.py         # 3D DenseNet-121 (unimodal + multi-modal)
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
└── scripts/
    ├── train_unimodal.sh     # Example: single-modality training
    ├── train_multichannel.sh # Example: multi-channel (best model)
    ├── train_multifusion.sh  # Example: multi-fusion training
    ├── run_ig_extraction.sh  # Example: IG attribution extraction
    └── run_xai_pipeline.sh   # Example: full XAI pipeline
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

```bash
# Best model: multi-channel with all DTI metrics (FA+MD+RD+AD)
bash scripts/train_multichannel.sh mc_4DTI

# Unimodal (single metric)
bash scripts/train_unimodal.sh unimodal_FA FA

# Multi-fusion (separate encoders)
bash scripts/train_multifusion.sh mf_T1w_FA
```

### 3. Extract Saliency Maps

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
| Multi-channel | 4 channels | ~11.2M | FA+MD+RD+AD concatenated |
| Multi-fusion | 2 encoders | ~22.6M | Separate encoders + L2 alignment |

## XAI Pipeline

1. **Integrated Gradients** with SmoothGrad (sigma=0.05, n=5)
2. Sign-aware group averaging (ASD-predictive / GDD-predictive)
3. Inverse spatial transform to native MRI space
4. Non-linear registration to infant MNI atlas (33-44 months)
5. Voxel-wise one-sample t-test (p<0.001)
6. Cluster-level correction (alpha=0.01)
7. Tract overlap: PSVP (top 1,000 voxels) and TOP (top 10,000 voxels) against HCP 1065 atlas

## License

MIT License. See [LICENSE](LICENSE) for details.
