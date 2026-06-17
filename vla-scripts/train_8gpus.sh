#!/bin/bash

# Default arguments
exp_name="prism-dinosiglip-224px+mx-rt_1"
num_nodes=1
num_gpus=8

# Parse arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"

    if [ $((${#ARGUMENT} > $KEY_LENGTH)) == 1 ]
    then
        export "$KEY"="$VALUE"
        echo "$KEY = $VALUE"
    fi
done

# Launch Training
TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=32 torchrun --standalone --nnodes 1 --nproc-per-node $num_gpus \
    vla-scripts/train.py \
    --vla.type="${exp_name}" \
    --data_root_dir="./data" \
    --run_root_dir="./logs"  \
    --wandb_project="openvla" \
    --run_id="${exp_name}"
