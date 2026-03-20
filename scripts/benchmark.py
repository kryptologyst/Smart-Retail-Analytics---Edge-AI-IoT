#!/usr/bin/env python3
"""
Benchmark script for Smart Retail Analytics models.
Measures performance across different edge devices and configurations.
"""

import argparse
import time
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import set_deterministic_seed, get_device, measure_inference_time, EdgeMetrics
from src.models.retail_models import ModelFactory
from src.pipelines.data_pipeline import RetailDataGenerator, RetailDataProcessor


def benchmark_model(
    model: Any,
    model_type: str,
    device: torch.device,
    test_data: np.ndarray,
    num_runs: int = 1000,
    warmup_runs: int = 100
) -> Dict[str, Any]:
    """Benchmark a single model.
    
    Args:
        model: Model to benchmark.
        model_type: Type of model.
        device: Device for inference.
        test_data: Test data for benchmarking.
        num_runs: Number of inference runs.
        warmup_runs: Number of warmup runs.
        
    Returns:
        Dictionary containing benchmark results.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Benchmarking {model_type} model on {device}")
    
    # Get model size
    if hasattr(model, 'get_model_size'):
        size_info = model.get_model_size()
    else:
        size_info = {'total_mb': 0.0, 'num_parameters': 0}
    
    # Measure inference time
    timing_results = measure_inference_time(
        model, test_data[:1], num_runs=num_runs, warmup_runs=warmup_runs
    )
    
    # Measure system metrics
    edge_metrics = EdgeMetrics(str(device))
    system_metrics = edge_metrics.get_all_metrics()
    
    # Measure accuracy (if we have labels)
    if model_type in ["baseline_pytorch", "edge_pytorch"]:
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_data).to(device)
            predictions = model(test_tensor).cpu().numpy()
    else:
        predictions = model.predict(test_data)
    
    # Compile results
    results = {
        'model_type': model_type,
        'device': str(device),
        'model_size_mb': size_info.get('total_mb', 0.0),
        'num_parameters': size_info.get('num_parameters', 0),
        'avg_latency_ms': timing_results['avg_latency_ms'],
        'throughput_fps': timing_results['throughput_fps'],
        'memory_usage_mb': system_metrics['memory']['rss_mb'],
        'cpu_usage_percent': system_metrics['cpu']['cpu_percent'],
        'num_runs': num_runs,
        'warmup_runs': warmup_runs
    }
    
    return results


def benchmark_device_configurations(
    model_type: str,
    test_data: np.ndarray,
    device_configs: List[str],
    num_runs: int = 1000
) -> List[Dict[str, Any]]:
    """Benchmark model across different device configurations.
    
    Args:
        model_type: Type of model to benchmark.
        test_data: Test data for benchmarking.
        device_configs: List of device configurations.
        num_runs: Number of inference runs.
        
    Returns:
        List of benchmark results for each device configuration.
    """
    results = []
    
    for device_config in device_configs:
        try:
            # Get device
            device = get_device(device_config, fallback=True)
            
            # Create model
            model = ModelFactory.create_model(model_type, input_size=test_data.shape[1])
            model.to(device)
            
            # Benchmark
            result = benchmark_model(model, model_type, device, test_data, num_runs)
            results.append(result)
            
        except Exception as e:
            logging.warning(f"Failed to benchmark {model_type} on {device_config}: {e}")
            continue
    
    return results


def create_benchmark_report(results: List[Dict[str, Any]], output_path: str):
    """Create comprehensive benchmark report.
    
    Args:
        results: List of benchmark results.
        output_path: Output file path for the report.
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create summary statistics
    summary = {
        'total_models_tested': len(df),
        'device_types': df['device'].unique().tolist(),
        'model_types': df['model_type'].unique().tolist(),
        'performance_summary': {
            'fastest_model': df.loc[df['avg_latency_ms'].idxmin(), 'model_type'],
            'smallest_model': df.loc[df['model_size_mb'].idxmin(), 'model_type'],
            'most_efficient': df.loc[df['throughput_fps'].idxmax(), 'model_type']
        },
        'latency_stats': {
            'min_ms': df['avg_latency_ms'].min(),
            'max_ms': df['avg_latency_ms'].max(),
            'mean_ms': df['avg_latency_ms'].mean(),
            'std_ms': df['avg_latency_ms'].std()
        },
        'throughput_stats': {
            'min_fps': df['throughput_fps'].min(),
            'max_fps': df['throughput_fps'].max(),
            'mean_fps': df['throughput_fps'].mean(),
            'std_fps': df['throughput_fps'].std()
        }
    }
    
    # Save detailed results
    df.to_csv(output_path.replace('.json', '.csv'), index=False)
    
    # Save summary
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Benchmark report saved to {output_path}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark models for edge deployment")
    parser.add_argument("--model-types", nargs="+", 
                       choices=["baseline_pytorch", "edge_pytorch", "baseline_tf"],
                       default=["baseline_pytorch", "edge_pytorch"],
                       help="Model types to benchmark")
    parser.add_argument("--devices", nargs="+",
                       choices=["cpu", "cuda", "mps"],
                       default=["cpu"],
                       help="Devices to benchmark on")
    parser.add_argument("--num-runs", type=int, default=1000,
                       help="Number of inference runs per benchmark")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of test samples to generate")
    parser.add_argument("--output-dir", default="benchmarks",
                       help="Output directory for results")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Config file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    from src.utils.core import load_config
    config = load_config(args.config)
    
    # Set deterministic seed
    set_deterministic_seed(config.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate test data
        logger.info("Generating test data...")
        data_generator = RetailDataGenerator(seed=config.seed)
        customer_data = data_generator.generate_customer_data(n_samples=args.num_samples)
        df = data_generator.to_dataframe(customer_data)
        
        processor = RetailDataProcessor()
        X, y = processor.prepare_features(df)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
        X_train, X_val, X_test = processor.normalize_features(X_train, X_val, X_test)
        
        logger.info(f"Generated {len(X_test)} test samples")
        
        # Benchmark each model type
        all_results = []
        
        for model_type in args.model_types:
            logger.info(f"Benchmarking {model_type}...")
            
            # Benchmark across device configurations
            model_results = benchmark_device_configurations(
                model_type, X_test, args.devices, args.num_runs
            )
            
            all_results.extend(model_results)
        
        # Create comprehensive report
        report_path = output_dir / "benchmark_report.json"
        create_benchmark_report(all_results, str(report_path))
        
        # Print summary
        df_results = pd.DataFrame(all_results)
        logger.info("Benchmark Summary:")
        logger.info(f"Total models tested: {len(df_results)}")
        logger.info(f"Fastest model: {df_results.loc[df_results['avg_latency_ms'].idxmin(), 'model_type']}")
        logger.info(f"Smallest model: {df_results.loc[df_results['model_size_mb'].idxmin(), 'model_type']}")
        logger.info(f"Most efficient: {df_results.loc[df_results['throughput_fps'].idxmax(), 'model_type']}")
        
        logger.info("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
