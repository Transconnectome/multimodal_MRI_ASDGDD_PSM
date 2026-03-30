#!/bin/bash
# Example: Train multi-fusion DenseNet-121 (separate encoders per modality)
# Usage: bash scripts/train_multifusion.sh <exp_name>
# e.g.:  bash scripts/train_multifusion.sh mf_T1w_FA
#        bash scripts/train_multifusion.sh mf_T1w_allDTI

EXP_NAME=${1:-"mf_T1w_FA"}

# Multi-fusion: separate DenseNet-121 encoders per modality with L2 embedding alignment
# For different combinations, modify --data_type accordingly:
#   T1w+FA:       --data_type T1w FA
#   T1w+all DTI:  --data_type T1w FA MD RD AD

python3 train.py \
    --model densenet3D121 \
    --multimodal multifusion \
    --dataset CHA \
    --data_type T1w FA \
    --in_channels 1 \
    --cat_target GDD \
    --confusion_matrix GDD \
    --phenotype total \
    --balanced_split psm_balan_iter_strat \
    --transform no_resize pad 1.5mm \
    --augmentation affine \
    --metric L2 \
    --optim AdamW \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --scheduler cawr2 \
    --epoch 300 \
    --early_stopping 100 \
    --early_stop_metric auroc+ap \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --accumulation_steps 4 \
    --val_size 0.2 \
    --test_size 0.2 \
    --gpus 0 1 \
    --wandb 1 \
    --exp_name ${EXP_NAME}
