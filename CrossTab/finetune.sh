#!/bin/bash

disk='/path/to/data'
batch_size=12
gpu='0'
exp_name='crosstab'
config_name='config_aibl_TIP'
dataset='AIBL'
dataset_lower='aibl'
CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=1 --nnodes 1 --master_port=$RANDOM evaluate.py --config-name ${config_name} \
exp_name=${exp_name} pretrain=False evaluate=True test_and_eval=False \
batch_size=${batch_size} finetune_strategy=${finetune_strategy} data_base=${disk}/data/data/${dataset}/ max_epochs=100 \