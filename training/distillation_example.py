#!/usr/bin/env python3
"""
Example usage script for Knowledge Distillation Pipeline

This script demonstrates how to use the knowledge distillation pipeline
to train a student model without selectivity from a teacher model with selectivity.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_knowledge_distillation_example():
    """
    Example of running knowledge distillation from a teacher model to a student model
    """
    
    # Paths (update these to match your setup)
    teacher_config_path = "./training/s4_ssm_complex.yaml"
    teacher_checkpoint_path = "./results/s4_ssm_results/checkpoints/best_model.pth"  # Update this path
    student_config_path = "./training/s4_ssm_student.yaml"
    save_dir = "./results/s4_ssm_distilled"
    
    print("Knowledge Distillation Example")
    print("=" * 50)
    print(f"Teacher config: {teacher_config_path}")
    print(f"Teacher checkpoint: {teacher_checkpoint_path}")
    print(f"Student config: {student_config_path}")
    print(f"Save directory: {save_dir}")
    print()
    
    # Check if teacher checkpoint exists
    if not os.path.exists(teacher_checkpoint_path):
        print("⚠️  Teacher checkpoint not found!")
        print(f"Please train a teacher model first or update the path: {teacher_checkpoint_path}")
        print("You can train a teacher model using:")
        print(f"python training_script.py --config {teacher_config_path}")
        return
    
    # Import the distillation script
    from training.distillation_script import main as distillation_main
    import sys
    
    # Example 1: Standard distillation (same dimensions, no selectivity)
    print("\n🔥 Running Standard Distillation (no selectivity):")
    original_argv = sys.argv
    sys.argv = [
        'distillation_script.py',
        '--teacher_config', teacher_config_path,
        '--teacher_checkpoint', teacher_checkpoint_path,
        '--student_config', student_config_path,
        '--save_dir', save_dir + '_standard',
        '--temperature', '4.0',
        '--alpha', '0.3',
        '--beta', '0.5',
        '--num_epochs', '200'
    ]
    
    try:
        result = distillation_main()
        if result == 0:
            print("✅ Standard distillation completed!")
    except Exception as e:
        print(f"❌ Standard distillation failed: {e}")
    finally:
        sys.argv = original_argv
    
    # Example 2: Compressed distillation (smaller model dimensions)
    print("\n🔥 Running Compressed Distillation (50% smaller):")
    sys.argv = [
        'distillation_script.py',
        '--teacher_config', teacher_config_path,
        '--teacher_checkpoint', teacher_checkpoint_path,
        '--student_config', student_config_path,
        '--save_dir', save_dir + '_compressed',
        '--student_model_dim', '32',
        '--student_state_dim', '256',
        '--student_num_layers', '4',
        '--temperature', '5.0',  # Higher temperature for harder compression
        '--alpha', '0.2',
        '--beta', '0.6',
        '--num_epochs', '250',
        '--learning_rate', '3e-4'
    ]
    
    try:
        result = distillation_main()
        if result == 0:
            print("✅ Compressed distillation completed!")
    except Exception as e:
        print(f"❌ Compressed distillation failed: {e}")
    finally:
        sys.argv = original_argv
    
    print("\n🎉 All distillation examples completed!")
    print(f"Results saved in:")
    print(f"  - Standard: {save_dir}_standard")
    print(f"  - Compressed: {save_dir}_compressed")


def compare_models():
    """
    Compare teacher and student models in terms of parameters and performance
    """
    import torch
    import yaml
    from model.model_utils import get_model_from_configs
    
    print("Model Comparison")
    print("=" * 30)
    
    # Load configurations
    teacher_config_path = "./training/s4_ssm_complex.yaml"
    student_config_path = "./training/s4_ssm_student.yaml"
    
    with open(teacher_config_path, 'r') as f:
        teacher_config = yaml.safe_load(f)
    
    with open(student_config_path, 'r') as f:
        student_config = yaml.safe_load(f)
    
    # Create models
    teacher_model = get_model_from_configs(**teacher_config['model'])
    student_model = get_model_from_configs(**student_config['model'])
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    print(f"Teacher Model:")
    print(f"  - Selectivity: {teacher_config['model'].get('use_selectivity', True)}")
    print(f"  - Parameters: {teacher_params:,}")
    print(f"  - Model dim: {teacher_config['model']['model_dim']}")
    print(f"  - Layers: {teacher_config['model']['num_layers']}")
    print()
    
    print(f"Student Model:")
    print(f"  - Selectivity: {student_config['model'].get('use_selectivity', False)}")
    print(f"  - Parameters: {student_params:,}")
    print(f"  - Model dim: {student_config['model']['model_dim']}")
    print(f"  - Layers: {student_config['model']['num_layers']}")
    print()
    
    reduction = (teacher_params - student_params) / teacher_params * 100
    print(f"Parameter Reduction: {reduction:.1f}%")
    print(f"Student model is {teacher_params / student_params:.1f}x smaller")
    
    # Test inference speed (dummy data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    student_model.to(device)
    
    # Create dummy input
    batch_size = 1
    seq_len = 10000
    input_dim = teacher_config['model']['input_dim']
    
    dummy_input = torch.randn(batch_size, seq_len, input_dim, device=device)
    if teacher_config['model']['complex_valued']:
        dummy_input = torch.complex(dummy_input, torch.randn_like(dummy_input))
    
    # Warm up
    with torch.no_grad():
        _ = teacher_model(dummy_input)
        _ = student_model(dummy_input)
    
    # Time inference
    import time
    
    # Teacher timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = teacher_model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    teacher_time = (time.time() - start) / 10
    
    # Student timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = student_model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    student_time = (time.time() - start) / 10
    
    print(f"\nInference Speed (seq_len={seq_len}):")
    print(f"  Teacher: {teacher_time:.4f}s per batch")
    print(f"  Student: {student_time:.4f}s per batch")
    print(f"  Speedup: {teacher_time / student_time:.2f}x")


def create_training_commands():
    """
    Print useful training commands for the knowledge distillation pipeline
    """
    print("Training Commands")
    print("=" * 40)
    print()
    
    print("1. Train Teacher Model (with selectivity):")
    print("   python training_script.py --config training/s4_ssm_complex.yaml")
    print()
    
    print("2. Standard Knowledge Distillation (same size, no selectivity):")
    print("   python training/distillation_script.py \\")
    print("       --teacher_config training/s4_ssm_complex.yaml \\")
    print("       --teacher_checkpoint results/s4_ssm_results/checkpoints/best_model.pth \\")
    print("       --student_config training/s4_ssm_student.yaml \\")
    print("       --save_dir results/s4_ssm_distilled")
    print()
    
    print("3. Compressed Distillation (50% smaller model):")
    print("   python training/distillation_script.py \\")
    print("       --teacher_config training/s4_ssm_complex.yaml \\")
    print("       --teacher_checkpoint results/s4_ssm_results/checkpoints/best_model.pth \\")
    print("       --student_config training/s4_ssm_student.yaml \\")
    print("       --student_model_dim 32 \\")
    print("       --student_state_dim 256 \\")
    print("       --student_num_layers 4 \\")
    print("       --save_dir results/s4_ssm_compressed")
    print()
    
    print("4. Tiny Distillation (75% smaller model):")
    print("   python training/distillation_script.py \\")
    print("       --teacher_config training/s4_ssm_complex.yaml \\")
    print("       --teacher_checkpoint results/s4_ssm_results/checkpoints/best_model.pth \\")
    print("       --student_config training/s4_ssm_student.yaml \\")
    print("       --student_model_dim 24 \\")
    print("       --student_state_dim 128 \\")
    print("       --student_num_layers 3 \\")
    print("       --temperature 6.0 \\")
    print("       --alpha 0.1 \\")
    print("       --beta 0.7 \\")
    print("       --save_dir results/s4_ssm_tiny")
    print()
    
    print("5. Compare Different Models:")
    print("   python training/distillation_example.py --compare")
    print()
    
    print("Model Size Guidelines:")
    print("  Standard (no selectivity):  model_dim=64, state_dim=540, layers=6")
    print("  Compressed (50% smaller):   model_dim=32, state_dim=256, layers=4")
    print("  Tiny (75% smaller):        model_dim=24, state_dim=128, layers=3")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Knowledge Distillation Example')
    parser.add_argument('--compare', action='store_true', help='Compare teacher and student models')
    parser.add_argument('--commands', action='store_true', help='Show training commands')
    parser.add_argument('--run', action='store_true', help='Run knowledge distillation example')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    elif args.commands:
        create_training_commands()
    elif args.run:
        run_knowledge_distillation_example()
    else:
        print("Knowledge Distillation Pipeline for sarSSM")
        print("=" * 50)
        print()
        print("Available options:")
        print("  --run       : Run knowledge distillation example")
        print("  --compare   : Compare teacher and student models")
        print("  --commands  : Show training commands")
        print()
        print("Example usage:")
        print("  python distillation_example.py --compare")
        print("  python distillation_example.py --run")