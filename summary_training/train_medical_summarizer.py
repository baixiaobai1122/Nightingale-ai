#!/usr/bin/env python3
"""
Training Script for Medical Summarization Model

This script trains a dual medical summarization model using the medical_dialog dataset
with PHI-safe preprocessing and comprehensive evaluation.

Usage:
    python train_medical_summarizer.py --config configs/base_config.yaml
    python train_medical_summarizer.py --model t5-large --epochs 15 --batch-size 16
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.append('src')

from data_processing.dataset_loader import MedicalDialogDataModule
from model_architecture.dual_summarizer import DualMedicalSummarizer
from training.trainer import DualMedicalTrainer, TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Medical Summarization Model')

    # Model arguments
    parser.add_argument('--model', type=str, default='t5-base',
                        help='Pre-trained model name (default: t5-base)')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder during training')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Number of warmup steps (default: 500)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default: 1.0)')

    # Data arguments
    parser.add_argument('--data-cache-dir', type=str, default='./data/cache',
                        help='Directory for caching dataset (default: ./data/cache)')
    parser.add_argument('--max-source-length', type=int, default=512,
                        help='Maximum source sequence length (default: 512)')
    parser.add_argument('--max-target-length', type=int, default=256,
                        help='Maximum target sequence length (default: 256)')

    # Loss weights
    parser.add_argument('--clinician-weight', type=float, default=0.6,
                        help='Weight for clinician summary loss (default: 0.6)')
    parser.add_argument('--patient-weight', type=float, default=0.4,
                        help='Weight for patient summary loss (default: 0.4)')

    # Output and logging
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for checkpoints (default: ./outputs)')
    parser.add_argument('--experiment-name', type=str, default='medical_summarizer',
                        help='Experiment name for logging (default: medical_summarizer)')
    parser.add_argument('--log-every', type=int, default=100,
                        help='Log every N steps (default: 100)')
    parser.add_argument('--save-every', type=int, default=1000,
                        help='Save checkpoint every N steps (default: 1000)')
    parser.add_argument('--eval-every', type=int, default=500,
                        help='Evaluate every N steps (default: 500)')

    # Wandb logging
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='medical-summarization',
                        help='Wandb project name (default: medical-summarization)')

    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration YAML file')

    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda) (default: auto)')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')

    return parser.parse_args()

def load_config_from_file(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments."""

    # Load from config file if provided
    if args.config:
        file_config = load_config_from_file(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        file_config = {}

    # Override with command line arguments
    config_dict = {
        'model_name': args.model,
        'max_source_length': args.max_source_length,
        'max_target_length': args.max_target_length,
        'freeze_encoder': args.freeze_encoder,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'max_grad_norm': args.max_grad_norm,
        'clinician_weight': args.clinician_weight,
        'patient_weight': args.patient_weight,
        'data_cache_dir': args.data_cache_dir,
        'log_every': args.log_every,
        'save_every': args.save_every,
        'eval_every': args.eval_every,
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'mixed_precision': args.mixed_precision,
        'num_workers': args.num_workers
    }

    # Set device
    if args.device == 'auto':
        import torch
        config_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config_dict['device'] = args.device

    # Merge with file config (command line takes precedence)
    file_config.update(config_dict)

    return TrainingConfig(**file_config)

def setup_directories(config: TrainingConfig):
    """Setup necessary directories."""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.data_cache_dir).mkdir(parents=True, exist_ok=True)

    # Create experiment directory
    experiment_dir = Path(config.output_dir) / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {experiment_dir}")
    logger.info(f"Data cache directory: {config.data_cache_dir}")

def save_config(config: TrainingConfig):
    """Save training configuration."""
    experiment_dir = Path(config.output_dir) / config.experiment_name
    config_path = experiment_dir / "training_config.yaml"

    with open(config_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)

    logger.info(f"Configuration saved to {config_path}")

def check_requirements():
    """Check if required packages are available."""
    required_packages = ['torch', 'transformers', 'datasets', 'rouge_score']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install missing packages with: pip install <package_name>")
        sys.exit(1)

    logger.info("All required packages are available")

def print_model_info(model: DualMedicalSummarizer):
    """Print model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {model.model_name}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

def main():
    """Main training function."""
    logger.info("Starting medical summarization model training")

    # Parse arguments
    args = parse_args()

    # Check requirements
    check_requirements()

    # Create configuration
    config = create_training_config(args)

    # Setup directories
    setup_directories(config)

    # Save configuration
    save_config(config)

    logger.info("Training configuration:")
    for key, value in asdict(config).items():
        logger.info(f"  {key}: {value}")

    # Setup data
    logger.info("Setting up data module...")
    data_config = {
        'cache_dir': config.data_cache_dir,
        'max_context_length': config.max_source_length,
        'tokenizer_name': config.model_name
    }

    data_module = MedicalDialogDataModule(data_config)

    try:
        train_loader, val_loader, test_loader = data_module.get_dataloaders(config.batch_size)
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.error("Make sure the medical_dialog dataset is available or check your internet connection")
        sys.exit(1)

    # Initialize model
    logger.info("Initializing model...")
    try:
        model = DualMedicalSummarizer(
            model_name=config.model_name,
            max_source_length=config.max_source_length,
            max_target_length=config.max_target_length,
            freeze_encoder=config.freeze_encoder
        )
        print_model_info(model)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = DualMedicalTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # Start training
    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint before exiting
        trainer.save_checkpoint(is_best=False)
        logger.info("Checkpoint saved before exit")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()