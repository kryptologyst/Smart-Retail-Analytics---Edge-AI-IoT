#!/usr/bin/env python3
"""
Model export script for Smart Retail Analytics.
Exports trained models to various edge deployment formats.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
import onnx
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.core import set_deterministic_seed, get_device, load_config
from src.models.retail_models import ModelFactory


def export_to_onnx(model, output_path: str, input_size: int = 9):
    """Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export.
        output_path: Output file path.
        input_size: Input feature size.
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    logging.info(f"Model exported to ONNX: {output_path}")


def export_to_tflite(model, output_path: str, input_size: int = 9):
    """Export TensorFlow model to TensorFlow Lite format.
    
    Args:
        model: TensorFlow model to export.
        output_path: Output file path.
        input_size: Input feature size.
    """
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    logging.info(f"Model exported to TensorFlow Lite: {output_path}")


def export_to_coreml(model, output_path: str, input_size: int = 9):
    """Export PyTorch model to CoreML format.
    
    Args:
        model: PyTorch model to export.
        output_path: Output file path.
        input_size: Input feature size.
    """
    try:
        import coremltools as ct
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, input_size)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, input_size), name="input")]
        )
        
        # Save
        coreml_model.save(output_path)
        
        logging.info(f"Model exported to CoreML: {output_path}")
        
    except ImportError:
        logging.warning("CoreML tools not available. Skipping CoreML export.")


def export_to_openvino(model, output_path: str, input_size: int = 9):
    """Export PyTorch model to OpenVINO format.
    
    Args:
        model: PyTorch model to export.
        output_path: Output file path.
        input_size: Input feature size.
    """
    try:
        import openvino as ov
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, input_size)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to OpenVINO
        ov_model = ov.convert_model(traced_model)
        
        # Save
        ov.save_model(ov_model, output_path)
        
        logging.info(f"Model exported to OpenVINO: {output_path}")
        
    except ImportError:
        logging.warning("OpenVINO not available. Skipping OpenVINO export.")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export models for edge deployment")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--model-type", required=True, 
                       choices=["baseline_pytorch", "edge_pytorch", "baseline_tf"],
                       help="Type of model to export")
    parser.add_argument("--output-dir", default="exports", help="Output directory")
    parser.add_argument("--formats", nargs="+", 
                       choices=["onnx", "tflite", "coreml", "openvino"],
                       default=["onnx"], help="Export formats")
    parser.add_argument("--input-size", type=int, default=9, help="Input feature size")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set deterministic seed
    set_deterministic_seed(config.seed)
    
    # Setup device
    device = get_device("cpu", fallback=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        if args.model_type in ["baseline_pytorch", "edge_pytorch"]:
            model = ModelFactory.create_model(args.model_type, input_size=args.input_size)
            
            # Load state dict if provided
            if Path(args.model_path).exists():
                model.load_state_dict(torch.load(args.model_path, map_location=device))
                logger.info(f"Loaded model from {args.model_path}")
            
            model.to(device)
            model.eval()
            
            # Export PyTorch models
            for format_type in args.formats:
                if format_type == "onnx":
                    export_to_onnx(model, str(output_dir / f"{args.model_type}.onnx"), args.input_size)
                elif format_type == "coreml":
                    export_to_coreml(model, str(output_dir / f"{args.model_type}.mlmodel"), args.input_size)
                elif format_type == "openvino":
                    export_to_openvino(model, str(output_dir / f"{args.model_type}_openvino"), args.input_size)
        
        elif args.model_type == "baseline_tf":
            model = ModelFactory.create_model(args.model_type, input_size=args.input_size)
            
            # Load weights if provided
            if Path(args.model_path).exists():
                model.load_weights(args.model_path)
                logger.info(f"Loaded model from {args.model_path}")
            
            # Export TensorFlow models
            for format_type in args.formats:
                if format_type == "tflite":
                    export_to_tflite(model, str(output_dir / f"{args.model_type}.tflite"), args.input_size)
        
        logger.info("Model export completed successfully!")
        
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        raise


if __name__ == "__main__":
    main()
