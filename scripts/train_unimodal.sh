#!/bin/bash
# Example: Train unimodal DenseNet-121 (single DTI metric)
# Usage: bash scripts/train_unimodal.sh <exp_name> <data_type>
# e.g.:  bash scripts/train_unimodal.sh unimodal_FA FA
#        bash scripts/train_unimodal.sh unimodal_MD MD
#        bash scripts/train_unimodal.sh unimodal_T1w T1w

EXP_NAME=${1:-"unimodal_FA"}
DATA_TYPE=${2:-"FA"}

python3 train.py \
    --model densenet3D121 \
    --dataset CHA \
    --data_type ${DATA_TYPE} \
    --in_channels 1 \
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
