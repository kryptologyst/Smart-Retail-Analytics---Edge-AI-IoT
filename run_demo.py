#!/usr/bin/env python3
"""
Quick start script for Smart Retail Analytics.
Demonstrates the complete pipeline from data generation to model training and evaluation.
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.utils.core import set_deterministic_seed, get_device, setup_logging
from src.pipelines.data_pipeline import RetailDataGenerator, RetailDataProcessor
from src.models.retail_models import ModelFactory
from src.pipelines.training_pipeline import PyTorchTrainer, ModelEvaluator
import torch
import numpy as np


def main():
    """Run the complete Smart Retail Analytics pipeline."""
    print("🛒 Smart Retail Analytics - Edge AI & IoT Project")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting Smart Retail Analytics pipeline...")
    
    # Set deterministic seed
    set_deterministic_seed(42)
    
    # Setup device
    device = get_device("cpu", fallback=True)
    logger.info(f"Using device: {device}")
    
    try:
        # Step 1: Generate synthetic retail data
        print("\n📊 Step 1: Generating synthetic retail data...")
        data_generator = RetailDataGenerator(seed=42)
        customer_data = data_generator.generate_customer_data(n_samples=500)
        df = data_generator.to_dataframe(customer_data)
        
        print(f"✅ Generated {len(customer_data)} customer data samples")
        print(f"   - Average time in store: {df['time_in_store'].mean():.1f} minutes")
        print(f"   - Average items touched: {df['items_touched'].mean():.1f}")
        print(f"   - Purchase likelihood: {df['purchase_likelihood'].mean():.1%}")
        
        # Step 2: Process data
        print("\n🔧 Step 2: Processing data...")
        processor = RetailDataProcessor()
        X, y = processor.prepare_features(df)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        X_train, X_val, X_test = processor.normalize_features(X_train, X_val, X_test)
        
        print(f"✅ Data processed and split:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Validation samples: {len(X_val)}")
        print(f"   - Test samples: {len(X_test)}")
        
        # Step 3: Train baseline model
        print("\n🧠 Step 3: Training baseline model...")
        baseline_model = ModelFactory.create_model("baseline_pytorch", input_size=X_train.shape[1])
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train model
        trainer = PyTorchTrainer(baseline_model, device)
        history = trainer.train(train_loader, val_loader, epochs=10, verbose=False)
        
        print(f"✅ Baseline model trained successfully!")
        print(f"   - Final training accuracy: {history['train_acc'][-1]:.1%}")
        print(f"   - Final validation accuracy: {history['val_acc'][-1]:.1%}")
        
        # Step 4: Train edge-optimized model
        print("\n⚡ Step 4: Training edge-optimized model...")
        edge_model = ModelFactory.create_model("edge_pytorch", input_size=X_train.shape[1])
        
        edge_trainer = PyTorchTrainer(edge_model, device)
        edge_history = edge_trainer.train(train_loader, val_loader, epochs=10, verbose=False)
        
        print(f"✅ Edge-optimized model trained successfully!")
        print(f"   - Final training accuracy: {edge_history['train_acc'][-1]:.1%}")
        print(f"   - Final validation accuracy: {edge_history['val_acc'][-1]:.1%}")
        
        # Step 5: Evaluate models
        print("\n📈 Step 5: Evaluating models...")
        
        # Evaluate baseline model
        baseline_evaluator = ModelEvaluator(baseline_model, device)
        baseline_metrics = baseline_evaluator.evaluate_accuracy_metrics(X_test, y_test, "pytorch")
        baseline_edge_metrics = baseline_evaluator.evaluate_edge_metrics(X_test, num_runs=100)
        
        # Evaluate edge model
        edge_evaluator = ModelEvaluator(edge_model, device)
        edge_metrics = edge_evaluator.evaluate_accuracy_metrics(X_test, y_test, "pytorch")
        edge_edge_metrics = edge_evaluator.evaluate_edge_metrics(X_test, num_runs=100)
        
        print(f"✅ Model evaluation completed!")
        print(f"\n📊 Results Summary:")
        print(f"{'Metric':<20} {'Baseline':<12} {'Edge':<12} {'Improvement':<12}")
        print("-" * 60)
        print(f"{'Accuracy':<20} {baseline_metrics['accuracy']:<12.1%} {edge_metrics['accuracy']:<12.1%} {edge_metrics['accuracy'] - baseline_metrics['accuracy']:<+12.1%}")
        print(f"{'F1-Score':<20} {baseline_metrics['f1_score']:<12.1%} {edge_metrics['f1_score']:<12.1%} {edge_metrics['f1_score'] - baseline_metrics['f1_score']:<+12.1%}")
        print(f"{'Latency (ms)':<20} {baseline_edge_metrics['avg_latency_ms']:<12.1f} {edge_edge_metrics['avg_latency_ms']:<12.1f} {(edge_edge_metrics['avg_latency_ms'] - baseline_edge_metrics['avg_latency_ms']):<+12.1f}")
        print(f"{'Model Size (MB)':<20} {baseline_edge_metrics['total_mb']:<12.1f} {edge_edge_metrics['total_mb']:<12.1f} {(edge_edge_metrics['total_mb'] - baseline_edge_metrics['total_mb']):<+12.1f}")
        
        # Step 6: Demo prediction
        print("\n🎯 Step 6: Making sample predictions...")
        
        # Generate a new customer
        new_customer = data_generator.generate_customer_data(n_samples=1)[0]
        features = np.array([[
            new_customer.time_in_store,
            new_customer.sections_visited,
            new_customer.items_touched,
            new_customer.interaction_time,
            new_customer.dwell_time,
            new_customer.footfall_count,
            new_customer.temperature,
            new_customer.humidity,
            new_customer.noise_level
        ]])
        
        # Normalize features
        features_scaled = processor.scaler.transform(features)
        
        # Make predictions
        baseline_model.eval()
        edge_model.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            baseline_pred = baseline_model(features_tensor).cpu().numpy()[0][0]
            edge_pred = edge_model(features_tensor).cpu().numpy()[0][0]
        
        print(f"✅ Sample customer prediction:")
        print(f"   - Customer ID: {new_customer.customer_id}")
        print(f"   - Time in store: {new_customer.time_in_store:.1f} minutes")
        print(f"   - Items touched: {new_customer.items_touched}")
        print(f"   - Baseline prediction: {baseline_pred:.1%}")
        print(f"   - Edge prediction: {edge_pred:.1%}")
        print(f"   - Actual likelihood: {new_customer.purchase_likelihood:.1%}")
        
        # Final summary
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"   - Edge model is {baseline_edge_metrics['avg_latency_ms'] / edge_edge_metrics['avg_latency_ms']:.1f}x faster")
        print(f"   - Edge model is {baseline_edge_metrics['total_mb'] / edge_edge_metrics['total_mb']:.1f}x smaller")
        print(f"   - Accuracy difference: {edge_metrics['accuracy'] - baseline_metrics['accuracy']:+.1%}")
        
        print(f"\n💡 Next steps:")
        print(f"   - Run 'streamlit run demo/app.py' for interactive demo")
        print(f"   - Run 'python scripts/train.py --compare-models' for detailed comparison")
        print(f"   - Run 'python scripts/benchmark.py' for performance benchmarking")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
