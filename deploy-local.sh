#!/bin/bash

# Decision flag for experiment type
configs_path="configs/"

# Check if argument for config path was provided
[ ! -z "$1" ] && configs_path="$1"

# Get allocated GPU count
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

for f in "$configs_path"/*.yaml; do
    echo "Processing config file '$f' with $gpu_count allocated GPUs."
    python3 -m fedml.run_federated --num-gpus="$gpu_count" --config-file="$f"
done
