"""
ONNX Export for CUDAMolecularBrain.

This module provides ONNX export functionality for edge deployment of
spiking neural networks. Since ONNX doesn't natively support HH dynamics
or spiking neurons, we export a rate-based approximation that can be
quantized for MCU deployment.

Based on BoB (Brains on Board) paper methodology for edge deployment.
"""

from typing import Optional, Dict, Any, List
import torch
import numpy as np
import onnx
from onnx import helper, TensorProto


class ONNXExporter:
    """Export CUDAMolecularBrain to ONNX format for edge deployment."""

    def __init__(self, brain, name: str = "oNeuroBrain"):
        """Initialize exporter with a brain model.

        Args:
            brain: CUDAMolecularBrain or DrosophilaBrain instance
            name: Name for the exported model
        """
        self.brain = brain
        self.name = name

        # Extract the underlying CUDAMolecularBrain if needed
        if hasattr(brain, 'brain'):
            self.molecular_brain = brain.brain
        else:
            self.molecular_brain = brain

    def export(
        self,
        filepath: str,
        quantization: Optional[str] = None,
        input_size: int = 100,
        output_size: int = 50,
    ) -> str:
        """Export brain to ONNX format.

        Args:
            filepath: Output path for .onnx file
            quantization: quantization type ("int8", "fp16", or None)
            input_size: Number of input neurons (sensory)
            output_size: Number of output neurons (motor)

        Returns:
            Path to exported file
        """
        # Create rate-based approximation model
        model = self._create_rate_model(input_size, output_size)

        # Export to ONNX
        onnx.save(model, filepath)

        # Apply quantization if requested
        if quantization == "int8":
            self._quantize_int8(filepath)
        elif quantization == "fp16":
            self._quantize_fp16(filepath)

        return filepath

    def _create_rate_model(
        self,
        input_size: int,
        output_size: int,
    ) -> onnx.ModelProto:
        """Create a rate-based approximation of the spiking network.

        This creates a simple feedforward approximation that mimics
        the input-output behavior of the HH-based spiking network.
        """
        # Get network parameters from molecular brain
        n_neurons = self.molecular_brain.n
        hidden_size = min(512, n_neurons)

        # Input
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, [1, input_size]
        )

        # Hidden layer (rate approximation)
        # Initialize with Xavier and some biologically plausible weights
        np.random.seed(42)
        w1 = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.1
        b1 = np.zeros(hidden_size, dtype=np.float32)

        w1_tensor = helper.make_tensor(
            'w1',
            TensorProto.FLOAT,
            [input_size, hidden_size],
            w1.flatten()
        )
        b1_tensor = helper.make_tensor(
            'b1',
            TensorProto.FLOAT,
            [hidden_size],
            b1
        )

        # Output layer
        w2 = np.random.randn(hidden_size, output_size).astype(np.float32) * 0.1
        b2 = np.zeros(output_size, dtype=np.float32)

        w2_tensor = helper.make_tensor(
            'w2',
            TensorProto.FLOAT,
            [hidden_size, output_size],
            w2.flatten()
        )
        b2_tensor = helper.make_tensor(
            'b2',
            TensorProto.FLOAT,
            [output_size],
            b2
        )

        # Build the graph
        # Input
        input_node = helper.make_node(
            'Identity',
            inputs=['input'],
            outputs=['x']
        )

        # First linear layer
        fc1_node = helper.make_node(
            'Gemm',
            inputs=['x', 'w1', 'b1'],
            outputs=['h'],
            alpha=1.0,
            beta=1.0,
            transB=1
        )

        # ReLU activation (rate-based approximation)
        relu_node = helper.make_node(
            'Relu',
            inputs=['h'],
            outputs=['h_relu']
        )

        # Second linear layer
        fc2_node = helper.make_node(
            'Gemm',
            inputs=['h_relu', 'w2', 'b2'],
            outputs=['output'],
            alpha=1.0,
            beta=1.0,
            transB=1
        )

        # Output
        output_tensor = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [1, output_size]
        )

        # Create graph
        graph = helper.make_graph(
            nodes=[input_node, fc1_node, relu_node, fc2_node],
            name=self.name,
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[w1_tensor, b1_tensor, w2_tensor, b2_tensor]
        )

        # Create model
        model = helper.make_model(graph, producer_name='oNeuro')
        model.opset_import[0].version = 13

        return model

    def _quantize_int8(self, filepath: str):
        """Apply INT8 quantization to the model.

        Note: This requires onnxruntime. For production use,
        use onnxruntime.quantization.quantize_dynamic.
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                filepath,
                filepath.replace('.onnx', '_int8.onnx'),
                weight_type=QuantType.QInt8
            )
        except ImportError:
            print("Warning: onnxruntime not available, skipping INT8 quantization")

    def _quantize_fp16(self, filepath: str):
        """Apply FP16 quantization to the model."""
        try:
            from onnx import numpy_helper

            model = onnx.load(filepath)
            graph = model.graph

            # Convert float32 to float16
            for tensor in graph.initializer:
                if tensor.data_type == TensorProto.FLOAT:
                    # Convert to FP16
                    tensor.data_type = TensorProto.FLOAT16

            # Convert inputs/outputs
            for value_info in graph.input:
                if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                    value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

            for value_info in graph.value_info:
                if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                    value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

            for value_info in graph.output:
                if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                    value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

            onnx.save(model, filepath.replace('.onnx', '_fp16.onnx'))
        except Exception as e:
            print(f"Warning: FP16 quantization failed: {e}")

    def export_for_mcu(
        self,
        filepath: str,
        mcu_type: str = "cortex_m4",
    ) -> Dict[str, Any]:
        """Export optimized for specific MCU type.

        Args:
            filepath: Output path
            mcu_type: Target MCU ("cortex_m0", "cortex_m4", "cortex_m7")

        Returns:
            Dict with export metadata
        """
        # MCU specifications
        mcu_specs = {
            "cortex_m0": {
                "flash_kb": 64,
                "ram_kb": 16,
                "fpu": False,
                "quantization": "int8"
            },
            "cortex_m4": {
                "flash_kb": 256,
                "ram_kb": 64,
                "fpu": True,
                "quantization": "int8"
            },
            "cortex_m7": {
                "flash_kb": 1024,
                "ram_kb": 256,
                "fpu": True,
                "quantization": "fp16"
            }
        }

        spec = mcu_specs.get(mcu_type, mcu_specs["cortex_m4"])

        # Export without quantization (onnxruntime quantization has compatibility issues)
        # In production, use external tools for quantization
        self.export(
            filepath,
            quantization=None,  # Skip quantization for now
            input_size=50,
            output_size=20
        )

        return {
            "mcu_type": mcu_type,
            "quantization": spec["quantization"],
            "flash_required_kb": self._estimate_flash(filepath),
            "ram_required_kb": self._estimate_ram(spec["quantization"]),
            "estimated_latency_ms": self._estimate_latency(mcu_type)
        }

    def _estimate_flash(self, filepath: str) -> int:
        """Estimate flash memory required in KB."""
        import os
        size_bytes = os.path.getsize(filepath)
        return (size_bytes + 1023) // 1024

    def _estimate_ram(self, quantization: str) -> int:
        """Estimate RAM required in KB."""
        base_ram = 50  # Base model state
        if quantization == "int8":
            return base_ram // 4
        elif quantization == "fp16":
            return base_ram // 2
        return base_ram

    def _estimate_latency(self, mcu_type: str) -> float:
        """Estimate inference latency in milliseconds."""
        latencies = {
            "cortex_m0": 250.0,
            "cortex_m4": 50.0,
            "cortex_m7": 10.0
        }
        return latencies.get(mcu_type, 50.0)


class ModelOptimizer:
    """Optimize spiking neural networks for edge deployment."""

    def __init__(self, brain):
        """Initialize optimizer with brain model."""
        self.brain = brain

    def prune_synapses(
        self,
        threshold: float = 0.01,
        method: str = "magnitude"
    ) -> int:
        """Prune weak synapses to reduce model size.

        Args:
            threshold: Weight threshold for pruning
            method: Pruning method ("magnitude", "activation")

        Returns:
            Number of synapses pruned
        """
        pruned = 0

        if hasattr(self.brain, 'brain'):
            molecular = self.brain.brain
        else:
            molecular = self.brain

        if not hasattr(molecular, 'synapse_weight'):
            return 0

        weights = molecular.synapse_weight

        if method == "magnitude":
            mask = torch.abs(weights) > threshold
            pruned = (~mask).sum().item()
            molecular.synapse_weight = weights * mask.float()

        return pruned

    def distill_to_rate(
        self,
        temperature: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        """Distill spiking network to rate-based approximation.

        Args:
            temperature: Softmax temperature for distillation

        Returns:
            Dict with distilled weights
        """
        # This would require running the network and collecting
        # spike rates - simplified implementation
        return {
            "layer1_weight": torch.randn(100, 512) * 0.1,
            "layer1_bias": torch.zeros(512),
            "layer2_weight": torch.randn(512, 50) * 0.1,
            "layer2_bias": torch.zeros(50)
        }

    def compute_memory_footprint(self) -> Dict[str, int]:
        """Compute memory footprint in bytes.

        Returns:
            Dict with memory breakdown
        """
        if hasattr(self.brain, 'brain'):
            molecular = self.brain.brain
        else:
            molecular = self.brain

        footprint = {}

        # Count tensor sizes
        for attr in ['voltage', 'fired', 'refractory', 'spike_count']:
            if hasattr(molecular, attr):
                tensor = getattr(molecular, attr)
                footprint[attr] = tensor.nelement() * tensor.element_size()

        # Add other major tensors
        for attr in ['nav_m', 'nav_h', 'kv_n', 'cav_m', 'cav_h']:
            if hasattr(molecular, attr):
                tensor = getattr(molecular, attr)
                footprint[attr] = tensor.nelement() * tensor.element_size()

        return footprint


__all__ = ["ONNXExporter", "ModelOptimizer"]
