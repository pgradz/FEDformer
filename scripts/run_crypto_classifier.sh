#!/bin/bash

cd ..
models=("FEDformer" "Autoformer")

data_params=(
  "ETHUSDT_cusum002.csv,ETHUSDT_cusum002_c_out32,ETHUSDT" 
 )


for model in "${models[@]}"; do
    # Iterate over each tuple
    for pair in "${data_params[@]}"; do
        # Split the tuple into data parts and linked parameter
        IFS=',' read -r data_path task_id currency <<< "$pair"

        # Run your Python script with the current model, data parts, and linked parameter
        python -u run_crypto.py \
          --data_path "$data_path" \
          --task_id "$task_id" \
          --currency "$currency" \
          --data "cryptoh1" \
          --model $model \
          --classifier \
          --run_subtype "classifier" \
          --pred_len 1

    done
done