"""Core utilities for Smart Retail Analytics Edge AI project."""

import os
import random
import logging
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators.
    
    Args:
        seed: Random seed value for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    
    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # TensorFlow settings for reproducibility
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def get_device(device_type: str = "auto", fallback: bool = True) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device_type: Device type ("auto", "cpu", "cuda", "mps").
        fallback: Whether to fallback to CPU if preferred device unavailable.
        
    Returns:
        PyTorch device object.
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    
    try:
        device = torch.device(device_type)
        # Test device availability
        if device_type == "cuda":
            torch.zeros(1).to(device)
        elif device_type == "mps":
            torch.zeros(1).to(device)
        return device
    except Exception as e:
        if fallback:
            logging.warning(f"Device {device_type} not available, falling back to CPU: {e}")
            return torch.device("cpu")
        else:
            raise RuntimeError(f"Device {device_type} not available and fallback disabled: {e}")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path.
        format_string: Custom format string for log messages.
        
    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        output_path: Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    OmegaConf.save(config, output_path)


def get_model_size(model: Union[torch.nn.Module, tf.keras.Model]) -> Dict[str, float]:
    """Calculate model size metrics.
    
    Args:
        model: PyTorch or TensorFlow model.
        
    Returns:
        Dictionary containing model size metrics in MB.
    """
    if isinstance(model, torch.nn.Module):
        # PyTorch model
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        
        return {
            "parameters_mb": param_size / (1024 * 1024),
            "buffers_mb": buffer_size / (1024 * 1024),
            "total_mb": total_size / (1024 * 1024),
            "num_parameters": sum(p.numel() for p in model.parameters())
        }
    else:
        # TensorFlow model
        param_size = model.count_params() * 4  # Assuming float32
        return {
            "parameters_mb": param_size / (1024 * 1024),
            "total_mb": param_size / (1024 * 1024),
            "num_parameters": model.count_params()
        }


def measure_inference_time(
    model: Union[torch.nn.Module, tf.keras.Model],
    input_data: Union[torch.Tensor, np.ndarray],
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """Measure model inference time.
    
    Args:
        model: Model to benchmark.
        input_data: Input data for inference.
        num_runs: Number of inference runs for timing.
        warmup_runs: Number of warmup runs before timing.
        
    Returns:
        Dictionary containing timing statistics in milliseconds.
    """
    import time
    
    if isinstance(model, torch.nn.Module):
        model.eval()
        device = next(model.parameters()).device
        
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).to(device)
        else:
            input_tensor = input_data.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Timing runs
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
    else:
        # TensorFlow model
        if isinstance(input_data, torch.Tensor):
            input_array = input_data.numpy()
        else:
            input_array = input_data
        
        # Warmup runs
        for _ in range(warmup_runs):
            _ = model(input_array)
        
        # Timing runs
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_array)
        end_time = time.time()
    
    total_time = (end_time - start_time) * 1000  # Convert to milliseconds
    avg_time = total_time / num_runs
    
    return {
        "avg_latency_ms": avg_time,
        "total_time_ms": total_time,
        "throughput_fps": 1000 / avg_time if avg_time > 0 else 0
    }


class EdgeMetrics:
    """Utility class for measuring edge device performance metrics."""
    
    def __init__(self, device_type: str = "cpu"):
        """Initialize edge metrics collector.
        
        Args:
            device_type: Type of edge device.
        """
        self.device_type = device_type
        self.metrics = {}
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage.
        
        Returns:
            Dictionary containing memory usage in MB.
        """
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            "percent": process.memory_percent()
        }
    
    def measure_cpu_usage(self) -> Dict[str, float]:
        """Measure CPU usage.
        
        Returns:
            Dictionary containing CPU usage metrics.
        """
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_count": psutil.cpu_count(),
            "load_avg": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        }
    
    def measure_energy_consumption(self) -> Dict[str, float]:
        """Measure energy consumption (if available).
        
        Returns:
            Dictionary containing energy metrics.
        """
        # This is a placeholder - actual implementation would depend on hardware
        # and require specific drivers or sensors
        return {
            "power_watts": 0.0,  # Placeholder
            "energy_joules": 0.0,  # Placeholder
            "available": False
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all available metrics.
        
        Returns:
            Dictionary containing all metrics.
        """
        return {
            "memory": self.measure_memory_usage(),
            "cpu": self.measure_cpu_usage(),
            "energy": self.measure_energy_consumption(),
            "device_type": self.device_type
        }
