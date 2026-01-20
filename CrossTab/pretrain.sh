#!/bin/bash

# Set the disk location, batch size, GPU device, experiment name, and fine-tuning strategy
disk='/data/path/to/data/'

batch_size=12
gpu='0,1'
# # Run the pretraining command directly
exp_name="crosstab"

CUDA_VISIBLE_DEVICES=${gpu} python -u run.py --config-name config_cross_TIP exp_name=${exp_name} pretrain=True evaluate=False batch_size=12 \
 max_epochs=300 data_base=${disk}data/data/Merge/ share_num=5