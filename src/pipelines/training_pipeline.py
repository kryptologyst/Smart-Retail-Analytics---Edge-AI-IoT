"""Training and evaluation pipeline for Smart Retail Analytics."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import json
import time

from ..models.retail_models import BaselineRetailModel, EdgeOptimizedRetailModel, TensorFlowBaselineModel
from ..utils.core import get_device, measure_inference_time, EdgeMetrics


class PyTorchTrainer:
    """PyTorch model trainer for retail analytics."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """Initialize PyTorch trainer.
        
        Args:
            model: PyTorch model to train.
            device: Device to train on.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for regularization.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target.float())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = (output.squeeze() > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target.float())
                
                total_loss += loss.item()
                pred = (output.squeeze() > 0.5).float()
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 15,
        patience: int = 5,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            patience: Early stopping patience.
            verbose: Whether to print training progress.
            
        Returns:
            Training history dictionary.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                val_loss, val_acc = 0.0, 0.0
            
            if verbose and epoch % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
        
        return self.history
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                probabilities = output.squeeze().cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'precision': precision_score(all_targets, all_predictions, zero_division=0),
            'recall': recall_score(all_targets, all_predictions, zero_division=0),
            'f1_score': f1_score(all_targets, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_targets, all_probabilities)
        }
        
        return metrics


class ModelEvaluator:
    """Comprehensive model evaluation for retail analytics."""
    
    def __init__(self, model: Any, device: torch.device):
        """Initialize model evaluator.
        
        Args:
            model: Trained model (PyTorch or TensorFlow).
            device: Device for evaluation.
        """
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def evaluate_accuracy_metrics(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = "pytorch"
    ) -> Dict[str, float]:
        """Evaluate accuracy-based metrics.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            model_type: Type of model ("pytorch" or "tensorflow").
            
        Returns:
            Dictionary containing accuracy metrics.
        """
        if model_type == "pytorch":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(self.device)
                probabilities = self.model(X_tensor).cpu().numpy().flatten()
        else:
            probabilities = self.model.predict(X_test).flatten()
        
        predictions = (probabilities > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1_score': f1_score(y_test, predictions, zero_division=0),
            'auc': roc_auc_score(y_test, probabilities)
        }
        
        return metrics
    
    def evaluate_edge_metrics(
        self,
        X_test: np.ndarray,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Evaluate edge performance metrics.
        
        Args:
            X_test: Test features.
            num_runs: Number of inference runs for timing.
            
        Returns:
            Dictionary containing edge performance metrics.
        """
        # Measure inference time
        timing_metrics = measure_inference_time(
            self.model, X_test[:1], num_runs=num_runs
        )
        
        # Measure model size
        if hasattr(self.model, 'get_model_size'):
            size_metrics = self.model.get_model_size()
        else:
            size_metrics = {'total_mb': 0.0, 'num_parameters': 0}
        
        # Measure system metrics
        edge_metrics = EdgeMetrics(str(self.device))
        system_metrics = edge_metrics.get_all_metrics()
        
        return {
            **timing_metrics,
            **size_metrics,
            'memory_usage_mb': system_metrics['memory']['rss_mb'],
            'cpu_usage_percent': system_metrics['cpu']['cpu_percent']
        }
    
    def create_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = "pytorch",
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Create and optionally save confusion matrix.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            model_type: Type of model ("pytorch" or "tensorflow").
            save_path: Optional path to save the plot.
            
        Returns:
            Confusion matrix array.
        """
        if model_type == "pytorch":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(self.device)
                probabilities = self.model(X_tensor).cpu().numpy().flatten()
        else:
            probabilities = self.model.predict(X_test).flatten()
        
        predictions = (probabilities > 0.5).astype(int)
        cm = confusion_matrix(y_test, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Purchase', 'Purchase'],
                   yticklabels=['No Purchase', 'Purchase'])
        plt.title('Confusion Matrix - Retail Analytics Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return cm
    
    def generate_evaluation_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = "pytorch",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            model_type: Type of model ("pytorch" or "tensorflow").
            save_path: Optional path to save the report.
            
        Returns:
            Dictionary containing complete evaluation report.
        """
        # Accuracy metrics
        accuracy_metrics = self.evaluate_accuracy_metrics(X_test, y_test, model_type)
        
        # Edge metrics
        edge_metrics = self.evaluate_edge_metrics(X_test)
        
        # Confusion matrix
        cm = self.create_confusion_matrix(X_test, y_test, model_type)
        
        # Classification report
        if model_type == "pytorch":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(self.device)
                probabilities = self.model(X_tensor).cpu().numpy().flatten()
        else:
            probabilities = self.model.predict(X_test).flatten()
        
        predictions = (probabilities > 0.5).astype(int)
        classification_rep = classification_report(
            y_test, predictions, output_dict=True
        )
        
        # Compile report
        report = {
            'accuracy_metrics': accuracy_metrics,
            'edge_metrics': edge_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_rep,
            'model_type': model_type,
            'evaluation_timestamp': time.time()
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


class ModelComparison:
    """Compare different models for retail analytics."""
    
    def __init__(self, device: torch.device):
        """Initialize model comparison.
        
        Args:
            device: Device for model evaluation.
        """
        self.device = device
        self.results = {}
        self.logger = logging.getLogger(__name__)
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_types: Dict[str, str]
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model name -> model instance.
            X_test: Test features.
            y_test: Test labels.
            model_types: Dictionary of model name -> model type.
            
        Returns:
            DataFrame with comparison results.
        """
        comparison_results = []
        
        for model_name, model in models.items():
            evaluator = ModelEvaluator(model, self.device)
            
            # Get accuracy metrics
            accuracy_metrics = evaluator.evaluate_accuracy_metrics(
                X_test, y_test, model_types[model_name]
            )
            
            # Get edge metrics
            edge_metrics = evaluator.evaluate_edge_metrics(X_test)
            
            # Combine results
            result = {
                'model_name': model_name,
                'model_type': model_types[model_name],
                **accuracy_metrics,
                **edge_metrics
            }
            
            comparison_results.append(result)
            self.results[model_name] = result
        
        return pd.DataFrame(comparison_results)
    
    def create_comparison_plots(
        self,
        results_df: pd.DataFrame,
        save_dir: Optional[str] = None
    ) -> None:
        """Create comparison plots.
        
        Args:
            results_df: DataFrame with comparison results.
            save_dir: Optional directory to save plots.
        """
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Accuracy comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3] if i < 6 else None
            if ax is not None:
                results_df.plot(x='model_name', y=metric, kind='bar', ax=ax)
                ax.set_title(f'{metric.capitalize()} Comparison')
                ax.set_ylabel(metric.capitalize())
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Edge metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        edge_metrics = ['avg_latency_ms', 'throughput_fps', 'total_mb', 'memory_usage_mb']
        for i, metric in enumerate(edge_metrics):
            ax = axes[i//2, i%2]
            results_df.plot(x='model_name', y=metric, kind='bar', ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/edge_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_leaderboard(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate model leaderboard.
        
        Args:
            results_df: DataFrame with comparison results.
            save_path: Optional path to save leaderboard.
            
        Returns:
            Sorted DataFrame with leaderboard.
        """
        # Create composite score (weighted combination of metrics)
        weights = {
            'accuracy': 0.3,
            'f1_score': 0.2,
            'auc': 0.2,
            'throughput_fps': 0.15,
            'total_mb': -0.15  # Lower is better for model size
        }
        
        composite_scores = []
        for _, row in results_df.iterrows():
            score = sum(
                weights[metric] * row[metric] 
                for metric in weights.keys() 
                if metric in row
            )
            composite_scores.append(score)
        
        results_df['composite_score'] = composite_scores
        
        # Sort by composite score
        leaderboard = results_df.sort_values('composite_score', ascending=False)
        
        if save_path:
            leaderboard.to_csv(save_path, index=False)
        
        return leaderboard
