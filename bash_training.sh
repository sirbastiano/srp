#!/bin/bash

files=(
    # "training/rv_transformer_autoregressive.yaml"
    # "training/rv_transformer_parallel.yaml"
    # "training/rv_transformer_encoder.yaml"
    # "training/cv_transformer.yaml"
    # "training/cv_transformer_autoregressive.yaml"
    "training/s4_ssm_complex_iteration.yaml"
)

for file in "${files[@]}"; do
    python training/training_script.py --config "$file"
done
