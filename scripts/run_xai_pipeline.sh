#!/bin/bash
# Complete XAI pipeline: from trained model to tract overlap metrics
# Usage: bash scripts/run_xai_pipeline.sh <checkpoint_dir> <output_base_dir>

CHECKPOINT_DIR=${1:?"Usage: bash scripts/run_xai_pipeline.sh <checkpoint_dir> <output_dir>"}
OUTPUT_DIR=${2:-"./xai_results"}

echo "============================================"
echo "XAI Pipeline for: ${CHECKPOINT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Step 1: Extract IG attributions (requires GPU)
echo ""
echo "=== Step 1: Integrated Gradients Extraction ==="
python3 run_ig.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --model densenet3D121 \
    --dataset CHA \
    --data_type FA MD RD AD \
    --multimodal multichannel \
    --in_channels 4 \
    --cat_target GDD \
    --phenotype total \
    --balanced_split psm_balan_iter_strat \
    --transform no_resize pad 1.5mm \
    --test_batch_size 1 \
    --gpus 0 \
    --exp_name ${CHECKPOINT_DIR}_IG

# Steps 2-3: Group averaging
echo ""
echo "=== Steps 2-3: Group Averaging ==="
python3 xai/run_xai_pipeline.py group_average \
    --attr_dir ${OUTPUT_DIR}/ig_attributions \
    --labels_csv ${OUTPUT_DIR}/test_labels.csv \
    --output_dir ${OUTPUT_DIR}/group_averages

# Step 4: Registration to infant MNI space (requires ANTs)
echo ""
echo "=== Step 4: MNI Registration (manual) ==="
echo "Run ANTs antsApplyTransforms for each subject's saliency map:"
echo "  antsApplyTransforms -d 3 -i <subject_saliency.nii.gz> \\"
echo "    -r <infant_mni_template.nii.gz> \\"
echo "    -t <T1_to_MNI_warp.nii.gz> \\"
echo "    -o <subject_saliency_MNI.nii.gz> \\"
echo "    -n Linear"

# Steps 5-6: Statistical testing (scipy fallback, no AFNI needed)
echo ""
echo "=== Steps 5-6: Voxel-wise T-test + Cluster Correction ==="
python3 xai/run_xai_pipeline.py ttest \
    --input_dir ${OUTPUT_DIR}/warped \
    --output_dir ${OUTPUT_DIR}/ttest_results \
    --threshold 0.001 \
    --min_cluster 10

# Step 7: Tract overlap metrics
echo ""
echo "=== Step 7: PSVP + TOP Tract Metrics ==="
python3 xai/run_xai_pipeline.py tract_overlap \
    --saliency_file ${OUTPUT_DIR}/group_averages/asd_predictive.nii.gz \
    --atlas_file data/alltracts.nii.gz \
    --tract_names data/hcp1065_abbreviation.txt \
    --output_dir ${OUTPUT_DIR}/tract_metrics

echo ""
echo "=== Pipeline Complete ==="
echo "Results in: ${OUTPUT_DIR}"
