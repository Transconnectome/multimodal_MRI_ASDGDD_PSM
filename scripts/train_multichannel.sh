#!/bin/bash
# Example: Train multi-channel DenseNet-121 (multiple DTI metrics concatenated)
# Usage: bash scripts/train_multichannel.sh <exp_name>
# e.g.:  bash scripts/train_multichannel.sh mc_4DTI        (FA+MD+RD+AD — best model)
#        bash scripts/train_multichannel.sh mc_T1w_FA      (T1w+FA)

EXP_NAME=${1:-"mc_4DTI"}

# Multi-channel: all 4 DTI metrics (FA, MD, RD, AD) concatenated in channel dimension
# For other combinations, modify --data_type and --in_channels accordingly:
#   FA+MD+RD+AD:  --data_type FA MD RD AD  --in_channels 4
#   T1w+FA:       --data_type T1w FA       --in_channels 2
#   T1w+all DTI:  --data_type T1w FA MD RD AD --in_channels 5

python3 train.py \
    --model densenet3D121 \
    --multimodal multichannel \
    --dataset CHA \
    --data_type FA MD RD AD \
    --in_channels 4 \
    --cat_target GDD \
    --confusion_matrix GDD \
    --phenotype total \
    --balanced_split psm_balan_iter_strat \
    --transform no_resize pad 1.5mm \
    --augmentation affine \
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
