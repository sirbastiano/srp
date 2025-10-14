#!/usr/bin/env python3
"""
Knowledge Distillation Training Script for sarSSM Models

This script implements knowledge distillation to transfer knowledge from a 
larger teacher model with selectivity to a smaller student model without selectivity.

Usage:
    python distillation_script.py --teacher_config path/to/teacher.yaml \
                                  --teacher_checkpoint path/to/teacher.pth \
                                  --student_config path/to/student.yaml \
                                  --save_dir ./results/distillation
"""

import argparse
import yaml
import torch
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Import wandb with fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Gradient tracking will be disabled.")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training_script import load_config, setup_logging
from model.model_utils import get_model_from_configs


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training for sarSSM')
    parser.add_argument('--teacher_config', type=str, required=True,
                       help='Path to teacher model configuration file')
    parser.add_argument('--teacher_checkpoint', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--student_config', type=str, required=True,
                       help='Path to student model configuration file')
    parser.add_argument('--save_dir', type=str, default='./results/distillation',
                       help='Directory to save distillation results')
    parser.add_argument('--temperature', type=float, default=2.5,
                       help='Temperature for knowledge distillation (reduced for sharper signals)')
    parser.add_argument('--alpha', type=float, default=0.8,
                       help='Weight for student loss (ground truth) - increased for better learning')
    parser.add_argument('--beta', type=float, default=0.15,
                       help='Weight for distillation loss (teacher knowledge) - reduced to prevent mode collapse')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--student_model_dim', type=int, default=None,
                       help='Student model dimension (overrides config)')
    parser.add_argument('--student_state_dim', type=int, default=None,
                       help='Student state dimension (overrides config)')
    parser.add_argument('--student_num_layers', type=int, default=None,
                       help='Student number of layers (overrides config)')
    
    # Progressive layer coupling arguments
    parser.add_argument('--progressive_layers', action='store_true',
                       help='Enable progressive layer coupling strategy')
    parser.add_argument('--teacher_layers', type=int, default=6,
                       help='Number of teacher model layers')
    parser.add_argument('--student_layers', type=int, default=4,
                       help='Number of student model layers')
    parser.add_argument('--stage_epochs', type=int, default=15,
                       help='Number of epochs per progressive training stage')
    
    # Distribution preservation arguments (NEW)
    parser.add_argument('--preserve_distribution', action='store_true', default=True,
                       help='Enable distribution-preserving knowledge distillation (DEFAULT: True)')
    parser.add_argument('--variance_weight', type=float, default=0.2,
                       help='Weight for variance preservation loss')
    parser.add_argument('--moment_weight', type=float, default=0.15,
                       help='Weight for moment matching loss')
    parser.add_argument('--confidence_weight', type=float, default=0.08,
                       help='Weight for confidence calibration loss')
    parser.add_argument('--dynamic_temperature', action='store_true', default=True,
                       help='Enable dynamic temperature scaling (DEFAULT: True)')
    parser.add_argument('--disable_distribution_preservation', action='store_true',
                       help='Disable distribution preservation (use standard KD)')
    
    args = parser.parse_args()
    
    # Handle distribution preservation flag
    if args.disable_distribution_preservation:
        args.preserve_distribution = False
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("Knowledge Distillation Training for sarSSM")
    print("=" * 80)
    print(f"Teacher config: {args.teacher_config}")
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Student config: {args.student_config}")
    print(f"Save directory: {args.save_dir}")
    print(f"Distillation parameters: T={args.temperature}, α={args.alpha}, β={args.beta}")
    
    # Print distillation strategy
    if args.progressive_layers:
        print(f"Strategy: Progressive layer coupling (Teacher: {args.teacher_layers}, Student: {args.student_layers}, Stage epochs: {args.stage_epochs})")
    elif args.preserve_distribution:
        print(f"Strategy: Distribution-preserving KD")
        print(f"   Variance weight: {args.variance_weight}")
        print(f"   Moment weight: {args.moment_weight}")
        print(f"   Confidence weight: {args.confidence_weight}")
        print(f"   Dynamic temperature: {args.dynamic_temperature}")
    else:
        print(f"Strategy: Standard knowledge distillation")
    print()
    
    # Load configurations
    print("Loading configurations...")
    
    # Load teacher configuration
    with open(args.teacher_config, 'r') as f:
        teacher_config = yaml.safe_load(f)
    
    # Load student configuration  
    student_config = load_config(Path(args.student_config), args)
    
    # Apply command line overrides
    if args.num_epochs:
        student_config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        student_config['training']['lr'] = args.learning_rate
    if args.batch_size:
        student_config['training']['batch_size'] = args.batch_size
        # Update dataloader batch sizes
        for split in ['train', 'validation', 'test']:
            if split in student_config['dataloader']:
                student_config['dataloader'][split]['batch_size'] = args.batch_size
    
    # Apply model dimension overrides
    if args.student_model_dim:
        student_config['model']['model_dim'] = args.student_model_dim
    if args.student_state_dim:
        student_config['model']['state_dim'] = args.student_state_dim
    if args.student_num_layers:
        student_config['model']['num_layers'] = args.student_num_layers
    
    # Update distillation parameters
    distillation_config = {
        'temperature': args.temperature,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': 1.0 - args.alpha - args.beta,
        'feature_matching': student_config.get('distillation', {}).get('feature_matching', True),
        'freeze_teacher': student_config.get('distillation', {}).get('freeze_teacher', True),
        
        # Progressive layer coupling parameters
        'progressive_layers': args.progressive_layers,
        'teacher_layers': args.teacher_layers,
        'student_layers': args.student_layers,
        'stage_epochs': args.stage_epochs,
        
        # Distribution preservation parameters (NEW)
        'preserve_distribution': args.preserve_distribution and not args.progressive_layers,  # Don't mix strategies
        'variance_weight': args.variance_weight,
        'moment_weight': args.moment_weight,
        'confidence_weight': args.confidence_weight,
        'dynamic_temperature': args.dynamic_temperature
    }
    
    # Create models
    print("Creating models...")
    teacher_model = get_model_from_configs(**teacher_config['model'])
    student_model = get_model_from_configs(**student_config['model'])
    
    # Print model information
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    print(f"Teacher model parameters: {teacher_params:,}")
    print(f"Student model parameters: {student_params:,}")
    print(f"Parameter reduction: {((teacher_params - student_params) / teacher_params * 100):.1f}%")
    print()
    
    # Initialize wandb for gradient tracking if available
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="sarssm-knowledge-distillation",
                name=f"distillation_{student_config['model'].get('model_dim', 'unknown')}dim_{student_config['model'].get('num_layers', 'unknown')}layers",
                config={
                    "teacher_config": teacher_config['model'],
                    "student_config": student_config['model'],
                    "training_config": student_config['training'],
                    "distillation_config": distillation_config,
                    "teacher_params": teacher_params,
                    "student_params": student_params,
                    "compression_ratio": teacher_params / student_params if student_params > 0 else 1.0
                },
                tags=["knowledge_distillation", "sarssm", "s4_ssm"]
            )
            print("✅ Wandb initialized for gradient tracking")
        except Exception as e:
            print(f"⚠️  Wandb initialization failed: {e}")
    else:
        print("⚠️  Wandb not available - gradient tracking disabled")
    
    # Initialize wandb for gradient tracking
    wandb.init(
        project="sarssm-knowledge-distillation",
        name=f"distillation_{student_config['model'].get('model_dim', 'unknown')}dim_{student_config['model'].get('num_layers', 'unknown')}layers",
        config={
            "teacher_config": teacher_config['model'],
            "student_config": student_config['model'],
            "training_config": student_config['training'],
            "distillation_config": distillation_config,
            "teacher_params": teacher_params,
            "student_params": student_params,
            "compression_ratio": teacher_params / student_params if student_params > 0 else 1.0
        },
        tags=["knowledge_distillation", "sarssm", "s4_ssm"]
    )
    
    # Print key differences
    print("Model configuration differences:")
    print(f"  Teacher selectivity: {teacher_config['model'].get('use_selectivity', True)}")
    print(f"  Student selectivity: {student_config['model'].get('use_selectivity', False)}")
    print(f"  Teacher layers: {teacher_config['model'].get('num_layers', 'N/A')}")
    print(f"  Student layers: {student_config['model'].get('num_layers', 'N/A')}")
    print(f"  Teacher model_dim: {teacher_config['model'].get('model_dim', 'N/A')}")
    print(f"  Student model_dim: {student_config['model'].get('model_dim', 'N/A')}")
    print(f"  Teacher state_dim: {teacher_config['model'].get('state_dim', 'N/A')}")
    print(f"  Student state_dim: {student_config['model'].get('state_dim', 'N/A')}")
    
    # Calculate compression ratio
    if teacher_params > 0 and student_params > 0:
        compression_ratio = teacher_params / student_params
        print(f"  Compression ratio: {compression_ratio:.2f}x")
    print()
    
    # Setup distillation pipeline
    print("Setting up knowledge distillation pipeline...")
    
    try:
        from training.knowledge_distillation import KnowledgeDistillationTrainer
        from training.training_script import create_dataloaders
        import pytorch_lightning as pl
        from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        
        # Create dataloaders
        train_loader, val_loader, test_loader, inference_loader = create_dataloaders(
            student_config['dataloader']
        )
        
        print(f"Created dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        # Create distillation trainer
        distillation_trainer = KnowledgeDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            teacher_checkpoint_path=args.teacher_checkpoint,
            base_save_dir=args.save_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            inference_loader=inference_loader,
            mode=student_config['training'].get('mode', 'parallel'),
            lr=student_config['training'].get('lr', 1e-4),
            gt_loss_fn_name=student_config['training'].get('gt_loss_fn', 'polarimetric'),
            feature_loss_fn_name=student_config['training'].get('feature_loss_fn', 'complex_mse'),
            input_dim=student_config['model'].get('input_dim', 4),
            **distillation_config
        )
        
        # Setup PyTorch Lightning trainer
        logger, text_logger = setup_logging(
            model_name=student_config['model'].get('name', 'model'),
            exp_dir=student_config['training'].get('save_dir', args.save_dir),
            use_wandb=True,
            wandb_project=student_config['training'].get('wandb_project', 'ssm4sar'),
            wandb_entity=student_config['training'].get('wandb_entity', None),
            wandb_tags=student_config['training'].get('wandb_tags', ['training']),
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, "checkpoints"),
            filename="student-{epoch:02d}-{val_total_loss:.6f}",
            monitor="val_total_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1
        )
        
        early_stopping = EarlyStopping(
            monitor="val_total_loss",
            patience=student_config['training'].get('patience', 30),
            mode="min",
            min_delta=student_config['training'].get('delta', 1e-5),
            verbose=True
        )
        
        trainer = pl.Trainer(
            max_epochs=student_config['training'].get('num_epochs', 200),
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping],
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            gradient_clip_val=1.0,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            enable_progress_bar=True
        )
        
        print("Starting knowledge distillation training...")
        print(f"Training for up to {student_config['training'].get('num_epochs', 200)} epochs")
        print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print()
        
        # Start training
        trainer.fit(distillation_trainer)
        
        print("\nTraining completed!")
        
        # Test the final model
        print("\nTesting final student model...")
        test_results = trainer.test(distillation_trainer)
        
        # Save final model
        final_model_path = os.path.join(args.save_dir, "final_student_model.pth")
        torch.save(student_model.state_dict(), final_model_path)
        print(f"Final student model saved to: {final_model_path}")
        
        # Save configuration for reproducibility
        final_config_path = os.path.join(args.save_dir, "final_student_config.yaml")
        with open(final_config_path, 'w') as f:
            yaml.dump(student_config, f, default_flow_style=False)
        print(f"Final configuration saved to: {final_config_path}")
        
        print("\nKnowledge distillation completed successfully!")
        print(f"Results saved in: {args.save_dir}")
        
        # Log final results to wandb if available
        if WANDB_AVAILABLE and test_results:
            try:
                final_metrics = {}
                for result in test_results:
                    for key, value in result.items():
                        final_metrics[f"final/{key}"] = value
                wandb.log(final_metrics)
            except Exception as e:
                print(f"Warning: Failed to log final metrics to wandb: {e}")
        
        # Finish wandb run
        if WANDB_AVAILABLE:
            try:
                wandb.finish()
            except:
                pass
        
        # Log final results to wandb
        if test_results:
            final_metrics = {}
            for result in test_results:
                for key, value in result.items():
                    final_metrics[f"final/{key}"] = value
            wandb.log(final_metrics)
        
        # Finish wandb run
        wandb.finish()
        
        # Print final comparison
        if test_results:
            print("\nFinal Test Results:")
            for result in test_results:
                for key, value in result.items():
                    print(f"  {key}: {value:.6f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up wandb on error
        if WANDB_AVAILABLE:
            try:
                wandb.finish()
            except:
                pass
        
        return 1
        
        # Clean up wandb on error
        try:
            wandb.finish()
        except:
            pass
        
        return 1
    
    return 0


def create_simple_distillation_script(
    teacher_config_path: str,
    teacher_checkpoint_path: str,
    save_dir: str = "./results/distillation"
):
    """
    Simple helper function to create a distillation script with minimal configuration
    """
    
    # Create student config by modifying teacher config
    with open(teacher_config_path, 'r') as f:
        teacher_config = yaml.safe_load(f)
    
    # Modify for student model
    student_config = teacher_config.copy()
    student_config['model']['use_selectivity'] = False  # Remove selectivity
    student_config['training']['save_dir'] = save_dir
    student_config['training']['lr'] = 5e-4  # Lower learning rate
    
    # Save student config
    student_config_path = os.path.join(save_dir, "student_config.yaml")
    os.makedirs(save_dir, exist_ok=True)
    
    with open(student_config_path, 'w') as f:
        yaml.dump(student_config, f, default_flow_style=False)
    
    print(f"Created student configuration: {student_config_path}")
    
    # Run distillation
    cmd = [
        "python", __file__,
        "--teacher_config", teacher_config_path,
        "--teacher_checkpoint", teacher_checkpoint_path,
        "--student_config", student_config_path,
        "--save_dir", save_dir
    ]
    
    print(f"Run distillation with: {' '.join(cmd)}")
    
    return student_config_path


if __name__ == '__main__':
    exit(main())