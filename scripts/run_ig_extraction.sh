#!/bin/bash
# Step 1: Extract Integrated Gradients attribution maps from a trained model
# Usage: bash scripts/run_ig_extraction.sh <checkpoint_dir>
# e.g.:  bash scripts/run_ig_extraction.sh mc_4DTI_01

CHECKPOINT_DIR=${1:?"Usage: bash scripts/run_ig_extraction.sh <checkpoint_dir>"}

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
