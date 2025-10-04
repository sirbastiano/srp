#!/bin/bash

files=(
    # "training/training_configs/rv_transformer_autoregressive.yaml"
    # "training/training_configs/rv_transformer_parallel.yaml"
    # "training/training_configs/rv_transformer_encoder.yaml"
    # "training/training_configs/cv_transformer.yaml"
    # "training/training_configs/cv_transformer_autoregressive.yaml"
    # "training/training_configs/s4_ssm_complex_iteration.yaml"
    # "training/training_configs/s4_ssm_complex.yaml"
    "training/training_configs/s4_ssm_complex_smaller_cols.yaml"
)

for file in "${files[@]}"; do
    python training/training_script.py --config "$file"
done

