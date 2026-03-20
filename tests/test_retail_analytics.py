"""Unit tests for Smart Retail Analytics project."""

import pytest
import numpy as np
import torch
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import set_deterministic_seed, get_device, get_model_size, measure_inference_time
from src.pipelines.data_pipeline import (
    RetailDataGenerator, RetailDataProcessor, RetailSensorData
)
from src.models.retail_models import (
    BaselineRetailModel, EdgeOptimizedRetailModel, ModelFactory
)


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.random()
        np.random.seed(42)
        val2 = np.random.random()
        assert val1 == val2
        
        # Test torch
        torch.manual_seed(42)
        val1 = torch.rand(1).item()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto", fallback=True)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_get_model_size(self):
        """Test model size calculation."""
        model = BaselineRetailModel(input_size=9)
        size_info = get_model_size(model)
        
        assert "total_mb" in size_info
        assert "num_parameters" in size_info
        assert size_info["total_mb"] > 0
        assert size_info["num_parameters"] > 0


class TestDataPipeline:
    """Test data pipeline components."""
    
    def test_retail_data_generator(self):
        """Test retail data generation."""
        generator = RetailDataGenerator(seed=42)
        data = generator.generate_customer_data(n_samples=10)
        
        assert len(data) == 10
        assert all(isinstance(item, RetailSensorData) for item in data)
        assert all(0 <= item.purchase_likelihood <= 1 for item in data)
    
    def test_retail_data_processor(self):
        """Test data processing."""
        processor = RetailDataProcessor()
        
        # Create sample data
        generator = RetailDataGenerator(seed=42)
        customer_data = generator.generate_customer_data(n_samples=100)
        df = generator.to_dataframe(customer_data)
        
        # Test feature preparation
        X, y = processor.prepare_features(df)
        assert X.shape[0] == 100
        assert X.shape[1] == 9
        assert len(y) == 100
        assert all(y_val in [0, 1] for y_val in y)
        
        # Test data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        assert len(X_train) + len(X_val) + len(X_test) == 100
        
        # Test normalization
        X_train_scaled, X_val_scaled, X_test_scaled = processor.normalize_features(
            X_train, X_val, X_test
        )
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape


class TestModels:
    """Test model components."""
    
    def test_baseline_model(self):
        """Test baseline model."""
        model = BaselineRetailModel(input_size=9)
        
        # Test forward pass
        x = torch.randn(1, 9)
        output = model(x)
        
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1
        
        # Test model size
        size_info = model.get_model_size()
        assert size_info["total_parameters"] > 0
    
    def test_edge_optimized_model(self):
        """Test edge-optimized model."""
        model = EdgeOptimizedRetailModel(input_size=9)
        
        # Test forward pass
        x = torch.randn(1, 9)
        output = model(x)
        
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1
    
    def test_model_factory(self):
        """Test model factory."""
        # Test PyTorch models
        baseline_model = ModelFactory.create_model("baseline_pytorch", input_size=9)
        assert isinstance(baseline_model, BaselineRetailModel)
        
        edge_model = ModelFactory.create_model("edge_pytorch", input_size=9)
        assert isinstance(edge_model, EdgeOptimizedRetailModel)
        
        # Test available models
        available_models = ModelFactory.get_available_models()
        assert "baseline_pytorch" in available_models
        assert "edge_pytorch" in available_models
        assert "baseline_tf" in available_models


class TestIntegration:
    """Test integration scenarios."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data generation to prediction."""
        # Generate data
        generator = RetailDataGenerator(seed=42)
        customer_data = generator.generate_customer_data(n_samples=50)
        df = generator.to_dataframe(customer_data)
        
        # Process data
        processor = RetailDataProcessor()
        X, y = processor.prepare_features(df)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        X_train, X_val, X_test = processor.normalize_features(X_train, X_val, X_test)
        
        # Create and test model
        model = BaselineRetailModel(input_size=9)
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test[:5])
            predictions = model(X_test_tensor)
            
        assert predictions.shape == (5, 1)
        assert all(0 <= pred.item() <= 1 for pred in predictions)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Create different models
        baseline_model = BaselineRetailModel(input_size=9)
        edge_model = EdgeOptimizedRetailModel(input_size=9)
        
        # Test model sizes
        baseline_size = get_model_size(baseline_model)
        edge_size = get_model_size(edge_model)
        
        # Edge model should be smaller
        assert edge_size["total_mb"] < baseline_size["total_mb"]
        assert edge_size["num_parameters"] < baseline_size["num_parameters"]


class TestEdgeOptimization:
    """Test edge optimization features."""
    
    def test_quantization_preparation(self):
        """Test quantization preparation."""
        model = EdgeOptimizedRetailModel(input_size=9, use_quantization=True)
        
        # Test quantization setup
        model.prepare_for_quantization()
        assert hasattr(model, 'qconfig')
    
    def test_model_compression(self):
        """Test model compression."""
        baseline_model = BaselineRetailModel(input_size=9)
        edge_model = EdgeOptimizedRetailModel(input_size=9)
        
        baseline_size = get_model_size(baseline_model)
        edge_size = get_model_size(edge_model)
        
        # Edge model should be more compressed
        compression_ratio = edge_size["total_mb"] / baseline_size["total_mb"]
        assert compression_ratio < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
