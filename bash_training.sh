#!/bin/bash

# files=(
#     # "training/training_configs/rv_transformer_autoregressive.yaml"
#     # "training/training_configs/rv_transformer_parallel.yaml"
#     # "training/training_configs/rv_transformer_encoder.yaml"
#     # "training/training_configs/cv_transformer.yaml"
#     # "training/training_configs/cv_transformer_autoregressive.yaml"
#     # "training/training_configs/s4_ssm_complex_iteration.yaml"
#     # "training/training_configs/s4_ssm_complex.yaml"
#     "training/training_configs/s4_ssm_complex_smaller_cols.yaml"
# )

# for file in "${files[@]}"; do
#     python training/training_script.py --config "$file"
# done

#python training/training_script.py --config training/training_configs/s4_ssm_complex_sweep.yaml --sweep --parallel --max_workers 8

# python training/training_script.py --config training/training_configs/s4_ssm_complex_sweep.yaml 
python training/training_script.py \
    --config training/training_configs/s4_ssm_complex_sweep.yaml \
    --sweep \
    --parallel \
    --use_tmux \
    --save_dir ./results/rec_loss_sweep