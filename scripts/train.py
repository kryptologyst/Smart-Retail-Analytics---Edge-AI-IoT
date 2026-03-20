#!/usr/bin/env python3
"""
Smart Retail Analytics - Edge AI & IoT Project
Modernized training script with comprehensive evaluation and edge optimization.

This script demonstrates customer behavior prediction for retail analytics
using edge-optimized neural networks with quantization and pruning support.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import set_deterministic_seed, get_device, setup_logging, load_config
from src.pipelines.data_pipeline import (
    RetailDataGenerator, RetailDataProcessor, StreamingDataSimulator
)
from src.models.retail_models import ModelFactory
from src.pipelines.training_pipeline import (
    PyTorchTrainer, ModelEvaluator, ModelComparison
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Retail Analytics Training Script"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--model-type", "-m",
        type=str,
        default="baseline_pytorch",
        choices=["baseline_pytorch", "edge_pytorch", "baseline_tf"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=15,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare multiple model types"
    )
    
    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Path:
    """Setup output directory structure.
    
    Args:
        output_dir: Base output directory path.
        
    Returns:
        Path object for the output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    
    return output_path


def generate_synthetic_data(
    config: DictConfig,
    n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic retail data for training.
    
    Args:
        config: Configuration object.
        n_samples: Number of samples to generate.
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {n_samples} synthetic retail data samples...")
    
    # Initialize data generator
    data_generator = RetailDataGenerator(seed=config.seed)
    
    # Generate customer data
    customer_data = data_generator.generate_customer_data(
        n_samples=n_samples,
        store_size="medium"
    )
    
    # Convert to DataFrame
    df = data_generator.to_dataframe(customer_data)
    
    # Initialize data processor
    processor = RetailDataProcessor()
    
    # Prepare features and labels
    X, y = processor.prepare_features(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
        X, y,
        test_size=config.data.test_size,
        validation_size=config.data.validation_size,
        random_state=config.seed
    )
    
    # Normalize features
    X_train, X_val, X_test = processor.normalize_features(X_train, X_val, X_test)
    
    logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_pytorch_model(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    config: DictConfig,
    epochs: int,
    batch_size: int
) -> Dict[str, Any]:
    """Train PyTorch model.
    
    Args:
        model: PyTorch model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        device: Device for training.
        config: Configuration object.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        
    Returns:
        Training results dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info("Training PyTorch model...")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Initialize trainer
    trainer = PyTorchTrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=1e-4
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=config.training.early_stopping.patience,
        verbose=True
    )
    
    return {
        'model': model,
        'trainer': trainer,
        'history': history
    }


def train_tensorflow_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: DictConfig,
    epochs: int,
    batch_size: int
) -> Dict[str, Any]:
    """Train TensorFlow model.
    
    Args:
        model: TensorFlow model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        config: Configuration object.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        
    Returns:
        Training results dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info("Training TensorFlow model...")
    
    # Train model
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return {
        'model': model,
        'history': history.history
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    model_type: str,
    output_dir: Path
) -> Dict[str, Any]:
    """Evaluate trained model.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        device: Device for evaluation.
        model_type: Type of model.
        output_dir: Output directory for results.
        
    Returns:
        Evaluation results dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {model_type} model...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device)
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        X_test=X_test,
        y_test=y_test,
        model_type=model_type.split('_')[1],  # Extract pytorch/tf from model_type
        save_path=str(output_dir / "results" / f"{model_type}_evaluation_report.json")
    )
    
    # Create confusion matrix plot
    evaluator.create_confusion_matrix(
        X_test=X_test,
        y_test=y_test,
        model_type=model_type.split('_')[1],
        save_path=str(output_dir / "plots" / f"{model_type}_confusion_matrix.png")
    )
    
    logger.info(f"Model evaluation completed. Results saved to {output_dir}")
    
    return report


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    output_dir: Path
) -> pd.DataFrame:
    """Compare multiple models.
    
    Args:
        models: Dictionary of model name -> model instance.
        X_test: Test features.
        y_test: Test labels.
        device: Device for evaluation.
        output_dir: Output directory for results.
        
    Returns:
        DataFrame with comparison results.
    """
    logger = logging.getLogger(__name__)
    logger.info("Comparing multiple models...")
    
    # Model types mapping
    model_types = {
        "baseline_pytorch": "pytorch",
        "edge_pytorch": "pytorch",
        "baseline_tf": "tensorflow"
    }
    
    # Initialize comparison
    comparison = ModelComparison(device)
    
    # Compare models
    results_df = comparison.compare_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        model_types=model_types
    )
    
    # Create comparison plots
    comparison.create_comparison_plots(
        results_df=results_df,
        save_dir=str(output_dir / "plots")
    )
    
    # Generate leaderboard
    leaderboard = comparison.generate_leaderboard(
        results_df=results_df,
        save_path=str(output_dir / "results" / "model_leaderboard.csv")
    )
    
    logger.info("Model comparison completed.")
    logger.info(f"Model Leaderboard:\n{leaderboard[['model_name', 'accuracy', 'f1_score', 'avg_latency_ms', 'total_mb', 'composite_score']].to_string()}")
    
    return leaderboard


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        log_file=f"logs/training_{args.model_type}.log"
    )
    
    logger.info("Starting Smart Retail Analytics Training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Device: {args.device}")
    
    # Set deterministic seed
    set_deterministic_seed(config.seed)
    
    # Setup device
    device = get_device(args.device, fallback=True)
    logger.info(f"Using device: {device}")
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Generate synthetic data
        X_train, X_val, X_test, y_train, y_val, y_test = generate_synthetic_data(
            config, args.n_samples
        )
        
        if args.compare_models:
            # Compare multiple models
            models = {}
            training_results = {}
            
            # Train baseline PyTorch model
            logger.info("Training baseline PyTorch model...")
            baseline_model = ModelFactory.create_model("baseline_pytorch", input_size=X_train.shape[1])
            baseline_results = train_pytorch_model(
                baseline_model, X_train, y_train, X_val, y_val,
                device, config, args.epochs, args.batch_size
            )
            models["baseline_pytorch"] = baseline_results['model']
            training_results["baseline_pytorch"] = baseline_results
            
            # Train edge-optimized PyTorch model
            logger.info("Training edge-optimized PyTorch model...")
            edge_model = ModelFactory.create_model("edge_pytorch", input_size=X_train.shape[1])
            edge_results = train_pytorch_model(
                edge_model, X_train, y_train, X_val, y_val,
                device, config, args.epochs, args.batch_size
            )
            models["edge_pytorch"] = edge_results['model']
            training_results["edge_pytorch"] = edge_results
            
            # Train baseline TensorFlow model
            logger.info("Training baseline TensorFlow model...")
            tf_model = ModelFactory.create_model("baseline_tf", input_size=X_train.shape[1])
            tf_results = train_tensorflow_model(
                tf_model, X_train, y_train, X_val, y_val,
                config, args.epochs, args.batch_size
            )
            models["baseline_tf"] = tf_results['model']
            training_results["baseline_tf"] = tf_results
            
            # Compare models
            leaderboard = compare_models(models, X_test, y_test, device, output_dir)
            
        else:
            # Train single model
            logger.info(f"Training {args.model_type} model...")
            model = ModelFactory.create_model(args.model_type, input_size=X_train.shape[1])
            
            if args.model_type in ["baseline_pytorch", "edge_pytorch"]:
                training_results = train_pytorch_model(
                    model, X_train, y_train, X_val, y_val,
                    device, config, args.epochs, args.batch_size
                )
                model_type = "pytorch"
            else:
                training_results = train_tensorflow_model(
                    model, X_train, y_train, X_val, y_val,
                    config, args.epochs, args.batch_size
                )
                model_type = "tensorflow"
            
            # Evaluate model
            evaluation_results = evaluate_model(
                model, X_test, y_test, device, model_type, output_dir
            )
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Test Accuracy: {evaluation_results['accuracy_metrics']['accuracy']:.4f}")
            logger.info(f"Test F1-Score: {evaluation_results['accuracy_metrics']['f1_score']:.4f}")
            logger.info(f"Test AUC: {evaluation_results['accuracy_metrics']['auc']:.4f}")
            logger.info(f"Average Latency: {evaluation_results['edge_metrics']['avg_latency_ms']:.2f} ms")
            logger.info(f"Model Size: {evaluation_results['edge_metrics']['total_mb']:.2f} MB")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
