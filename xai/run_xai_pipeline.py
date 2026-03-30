"""
XAI Pipeline: From per-subject IG attributions to group-level saliency maps and tract metrics.

Full pipeline (7 steps):
  Step 1: IG + SmoothGrad extraction (GPU, via run_ig.py)
  Step 2: Sign-aware group averaging (this script)
  Step 3: Inverse transform to native space (this script)
  Step 4: Registration to infant MNI space (requires ANTs)
  Step 5-6: Voxel-wise t-test + cluster correction (this script, scipy fallback)
  Step 7: Tract overlap metrics — PSVP and TOP (this script)

Usage:
  # Steps 2-3: Group averaging + inverse transform
  python xai/run_xai_pipeline.py group_average --attr_dir <path> --labels_csv <path> --output_dir <path>

  # Steps 5-6: Statistical testing (scipy, no AFNI needed)
  python xai/run_xai_pipeline.py ttest --input_dir <path> --output_dir <path>

  # Step 7: Tract overlap metrics
  python xai/run_xai_pipeline.py tract_overlap --saliency_file <path> --atlas_file <path> --output_dir <path>

  # Full pipeline (steps 2-3 + 5-7, skips Step 4 registration)
  python xai/run_xai_pipeline.py full --attr_dir <path> --labels_csv <path> --atlas_file <path> --output_dir <path>
"""

import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage, stats
from pathlib import Path


# ============================================================
# Step 2: Sign-Aware Group Averaging
# ============================================================

def sign_aware_average(maps):
    """Average saliency maps separately for positive and negative values."""
    pos = np.clip(maps, 0, None)
    neg = np.clip(maps, None, 0)
    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    return pos_mean, neg_mean


def group_average(attr_dir, labels_csv, output_dir, channels=None):
    """
    Step 2: Load per-subject attribution maps, group by ASD/GDD,
    and compute sign-aware group averages.

    Args:
        attr_dir: Directory containing per-subject .npy files
        labels_csv: CSV with columns [subject_id, label] (0=GDD, 1=ASD)
        output_dir: Output directory for averaged maps
        channels: List of channel names (default: ['FA', 'MD', 'RD', 'AD'])
    """
    if channels is None:
        channels = ['FA', 'MD', 'RD', 'AD']

    os.makedirs(output_dir, exist_ok=True)
    labels = pd.read_csv(labels_csv)

    # Load all attribution maps
    asd_maps, gdd_maps = [], []
    for _, row in labels.iterrows():
        subj_id = row.iloc[0]
        label = row.iloc[1]  # 1=ASD, 0=GDD
        fpath = os.path.join(attr_dir, f"{subj_id}.npy")
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found, skipping")
            continue
        attr_map = np.load(fpath)
        if label == 1:
            asd_maps.append(attr_map)
        else:
            gdd_maps.append(attr_map)

    asd_maps = np.array(asd_maps)
    gdd_maps = np.array(gdd_maps)
    print(f"Loaded {len(asd_maps)} ASD, {len(gdd_maps)} GDD attribution maps")

    # Sign-aware averaging
    tp_pos, tp_neg = sign_aware_average(asd_maps)  # True Positive (ASD correct)
    tn_pos, tn_neg = sign_aware_average(gdd_maps)  # True Negative (GDD correct)

    # ASD-predictive = TP_positive + TN_negative
    asd_predictive = tp_pos + tn_neg
    # GDD-predictive = TP_negative + TN_positive
    gdd_predictive = tp_neg + tn_pos

    # Save
    for name, data in [('tp_pos', tp_pos), ('tp_neg', tp_neg),
                       ('tn_pos', tn_pos), ('tn_neg', tn_neg),
                       ('asd_predictive', asd_predictive),
                       ('gdd_predictive', gdd_predictive)]:
        np.save(os.path.join(output_dir, f"{name}.npy"), data)
        # Also save as NIfTI for visualization
        affine = np.eye(4) * 1.5  # ~1.5mm isotropic
        affine[3, 3] = 1
        img = nib.Nifti1Image(data, affine)
        nib.save(img, os.path.join(output_dir, f"{name}.nii.gz"))

    print(f"Group averages saved to {output_dir}")
    return asd_predictive, gdd_predictive


# ============================================================
# Step 3: Inverse Transform (138³ → 206³)
# ============================================================

def inverse_resize(data, target_shape=(206, 206, 206)):
    """Resize attribution maps back to original padded space."""
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape[-3:])]
    if data.ndim == 4:  # multi-channel
        return np.stack([ndimage.zoom(data[c], zoom_factors, order=1) for c in range(data.shape[0])])
    return ndimage.zoom(data, zoom_factors, order=1)


# ============================================================
# Steps 5-6: Voxel-wise T-test + Cluster Correction (scipy)
# ============================================================

def voxelwise_ttest(subject_maps, threshold=0.001):
    """
    One-sample t-test at each voxel across subjects.

    Args:
        subject_maps: array of shape (N_subjects, *spatial_dims)
        threshold: p-value threshold (default 0.001)

    Returns:
        t_stat: t-statistic map
        p_val: p-value map
        sig_mask: boolean mask of significant voxels
    """
    n = subject_maps.shape[0]
    mean_map = subject_maps.mean(axis=0)
    se_map = subject_maps.std(axis=0, ddof=1) / np.sqrt(n)

    # Avoid division by zero
    se_map[se_map == 0] = np.inf

    t_stat = mean_map / se_map
    df = n - 1
    p_val = 2 * stats.t.sf(np.abs(t_stat), df)
    sig_mask = p_val < threshold

    return t_stat, p_val, sig_mask


def cluster_correction(sig_mask, min_cluster_size=10):
    """
    Remove clusters smaller than min_cluster_size.

    Args:
        sig_mask: boolean mask of significant voxels
        min_cluster_size: minimum cluster size in voxels

    Returns:
        corrected_mask: boolean mask with small clusters removed
        n_clusters: number of surviving clusters
    """
    labeled, n_features = ndimage.label(sig_mask)
    corrected_mask = np.zeros_like(sig_mask, dtype=bool)
    surviving = 0

    for i in range(1, n_features + 1):
        cluster = labeled == i
        if cluster.sum() >= min_cluster_size:
            corrected_mask |= cluster
            surviving += 1

    print(f"Cluster correction: {n_features} clusters → {surviving} surviving (min size={min_cluster_size})")
    return corrected_mask, surviving


def run_ttest(input_dir, output_dir, threshold=0.001, min_cluster=10):
    """Run voxel-wise t-test + cluster correction on group saliency maps."""
    os.makedirs(output_dir, exist_ok=True)

    for prefix in ['tp', 'tn']:
        # Load subject maps (assuming they are concatenated along first axis)
        fpath = os.path.join(input_dir, f"{prefix}_subjects.npy")
        if not os.path.exists(fpath):
            print(f"Skipping {prefix}: {fpath} not found")
            continue

        subject_maps = np.load(fpath)
        t_stat, p_val, sig_mask = voxelwise_ttest(subject_maps, threshold)
        corrected, n = cluster_correction(sig_mask, min_cluster)

        # Save
        affine = np.eye(4)
        for name, data in [('tstat', t_stat), ('pval', p_val),
                           ('sig', sig_mask.astype(np.float32)),
                           ('cluster', corrected.astype(np.float32))]:
            img = nib.Nifti1Image(data, affine)
            nib.save(img, os.path.join(output_dir, f"{prefix}_{name}.nii.gz"))

    print(f"T-test results saved to {output_dir}")


# ============================================================
# Step 7: Tract Overlap Metrics (PSVP and TOP)
# ============================================================

def compute_psvp(saliency, tract_mask, top_n=1000):
    """Peak Saliency Voxel Percentage: proportion of top-N voxels within each tract."""
    flat = np.abs(saliency).ravel()
    threshold = np.sort(flat)[-top_n] if len(flat) > top_n else flat.min()
    top_mask = np.abs(saliency) >= threshold

    n_tracts = tract_mask.shape[-1]
    psvp = np.zeros(n_tracts)
    for t in range(n_tracts):
        overlap = (top_mask & (tract_mask[..., t] > 0)).sum()
        psvp[t] = overlap / top_n * 100

    return psvp


def compute_top(saliency, tract_mask, top_n=10000):
    """Tract Overlap Percentage: proportion of each tract overlapping top-N saliency."""
    flat = np.abs(saliency).ravel()
    threshold = np.sort(flat)[-top_n] if len(flat) > top_n else flat.min()
    top_mask = np.abs(saliency) >= threshold

    n_tracts = tract_mask.shape[-1]
    top_scores = np.zeros(n_tracts)
    for t in range(n_tracts):
        tract_vol = (tract_mask[..., t] > 0).sum()
        if tract_vol == 0:
            continue
        overlap = (top_mask & (tract_mask[..., t] > 0)).sum()
        top_scores[t] = overlap / tract_vol * 100

    return top_scores


def run_tract_overlap(saliency_file, atlas_file, tract_names_file=None, output_dir='.', top_n_psvp=1000, top_n_top=10000):
    """
    Compute PSVP and TOP metrics for a saliency map against a tract atlas.

    Args:
        saliency_file: NIfTI file with group saliency map (3D or 4D)
        atlas_file: NIfTI file with tract atlas (4D: spatial × N_tracts)
        tract_names_file: Text file with tract names (one per line)
        output_dir: Output directory
        top_n_psvp: Number of top voxels for PSVP (default 1000)
        top_n_top: Number of top voxels for TOP (default 10000)
    """
    os.makedirs(output_dir, exist_ok=True)

    saliency_img = nib.load(saliency_file)
    saliency = saliency_img.get_fdata()
    atlas_img = nib.load(atlas_file)
    atlas = atlas_img.get_fdata()

    n_tracts = atlas.shape[-1]

    # Load tract names
    if tract_names_file and os.path.exists(tract_names_file):
        with open(tract_names_file, 'r', encoding='utf-16-le') as f:
            tract_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        tract_names = [f"Tract_{i}" for i in range(n_tracts)]

    # Compute for positive and negative saliency separately
    results = []
    for direction, sal_data in [('Positive', np.clip(saliency, 0, None)),
                                 ('Negative', np.clip(saliency, None, 0))]:
        psvp = compute_psvp(sal_data, atlas, top_n_psvp)
        top = compute_top(sal_data, atlas, top_n_top)

        for t in range(n_tracts):
            results.append({
                'Tract': tract_names[t] if t < len(tract_names) else f"Tract_{t}",
                'Direction': direction,
                'PSVP': psvp[t],
                'TOP': top[t]
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'tract_overlap_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Tract overlap metrics saved to {csv_path}")
    print(f"\nTop 10 tracts by PSVP (Positive):")
    pos_df = df[df['Direction'] == 'Positive'].sort_values('PSVP', ascending=False)
    print(pos_df.head(10).to_string(index=False))

    return df


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='XAI Pipeline: Group saliency → Tract metrics')
    subparsers = parser.add_subparsers(dest='command', help='Pipeline stage')

    # Group average
    p_avg = subparsers.add_parser('group_average', help='Steps 2-3: Sign-aware group averaging')
    p_avg.add_argument('--attr_dir', required=True, help='Directory with per-subject .npy attribution maps')
    p_avg.add_argument('--labels_csv', required=True, help='CSV with [subject_id, label] columns')
    p_avg.add_argument('--output_dir', required=True, help='Output directory')
    p_avg.add_argument('--channels', nargs='+', default=['FA', 'MD', 'RD', 'AD'])

    # T-test
    p_ttest = subparsers.add_parser('ttest', help='Steps 5-6: Voxel-wise t-test + cluster correction')
    p_ttest.add_argument('--input_dir', required=True, help='Directory with tp_subjects.npy, tn_subjects.npy')
    p_ttest.add_argument('--output_dir', required=True, help='Output directory')
    p_ttest.add_argument('--threshold', type=float, default=0.001, help='P-value threshold')
    p_ttest.add_argument('--min_cluster', type=int, default=10, help='Minimum cluster size')

    # Tract overlap
    p_tract = subparsers.add_parser('tract_overlap', help='Step 7: PSVP and TOP metrics')
    p_tract.add_argument('--saliency_file', required=True, help='NIfTI saliency map')
    p_tract.add_argument('--atlas_file', required=True, help='NIfTI tract atlas (4D)')
    p_tract.add_argument('--tract_names', default=None, help='Text file with tract names')
    p_tract.add_argument('--output_dir', default='.', help='Output directory')
    p_tract.add_argument('--top_n_psvp', type=int, default=1000)
    p_tract.add_argument('--top_n_top', type=int, default=10000)

    # Full pipeline
    p_full = subparsers.add_parser('full', help='Full pipeline (steps 2-3 + 5-7)')
    p_full.add_argument('--attr_dir', required=True)
    p_full.add_argument('--labels_csv', required=True)
    p_full.add_argument('--atlas_file', required=True)
    p_full.add_argument('--tract_names', default=None)
    p_full.add_argument('--output_dir', required=True)

    args = parser.parse_args()

    if args.command == 'group_average':
        group_average(args.attr_dir, args.labels_csv, args.output_dir, args.channels)

    elif args.command == 'ttest':
        run_ttest(args.input_dir, args.output_dir, args.threshold, args.min_cluster)

    elif args.command == 'tract_overlap':
        run_tract_overlap(args.saliency_file, args.atlas_file, args.tract_names,
                         args.output_dir, args.top_n_psvp, args.top_n_top)

    elif args.command == 'full':
        print("=== Step 2-3: Group Averaging ===")
        avg_dir = os.path.join(args.output_dir, 'group_averages')
        group_average(args.attr_dir, args.labels_csv, avg_dir)

        print("\n=== Step 7: Tract Overlap ===")
        for name in ['asd_predictive', 'gdd_predictive']:
            sal_file = os.path.join(avg_dir, f"{name}.nii.gz")
            if os.path.exists(sal_file):
                tract_dir = os.path.join(args.output_dir, f'tract_metrics_{name}')
                run_tract_overlap(sal_file, args.atlas_file, args.tract_names, tract_dir)

        print("\nNote: Steps 4 (MNI registration) and 5-6 (t-test) require")
        print("per-subject warped files. Run separately with 'ttest' command.")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
