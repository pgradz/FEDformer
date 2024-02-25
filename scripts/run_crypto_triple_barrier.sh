#!/bin/bash

cd ..
models=("FEDformer" "Autoformer")


data_params=(
  "ETHUSDT_cusum002_005.csv,ETHUSDT_cusum002_005_c_out32,ETHUSDT,0.05" 
 )

for model in "${models[@]}"; do
    # Iterate over each tuple
    for pair in "${data_params[@]}"; do
        # Split the tuple into data parts and linked parameter
        IFS=',' read -r data_path task_id currency barrier_threshold<<< "$pair"

        # Run your Python script with the current model, data parts, and linked parameter
        python -u run_crypto.py \
          --data_path "$data_path" \
          --task_id "$task_id" \
          --currency "$currency" \
          --data "crytpo_triple_barrier" \
          --model $model \
          --barrier_threshold $barrier_threshold
          --run_subtype "triple_barrier"
    done
done