#!/usr/bin/env python3
"""
ONNX Export Demo -- Export Fly Brain for Edge Deployment.

This demo shows how to export a CUDAMolecularBrain to ONNX format
for deployment on edge devices (microcontrollers, drones, robots).

Usage:
    python3 demos/demo_onnx_export.py
    python3 demos/demo_onnx_export.py --scale tiny
    python3 demos/demo_onnx_export.py --quantize int8
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oneuro.organisms.drosophila import Drosophila
from oneuro.export import ONNXExporter, ModelOptimizer


def main():
    parser = argparse.ArgumentParser(description="ONNX Export Demo")
    parser.add_argument('--scale', choices=['tiny', 'small', 'medium'],
                       default='tiny', help='Brain scale')
    parser.add_argument('--quantize', choices=['int8', 'fp16', 'none'],
                       default='none', help='Quantization type')
    parser.add_argument('--output', '-o', default='brain.onnx',
                       help='Output filename')
    args = parser.parse_args()

    print("="*60)
    print("  ONNX EXPORT FOR EDGE DEPLOYMENT")
    print("  (Brains on Board - Nature 2024)")
    print("="*60)
    print()

    # Step 1: Create the fly brain
    print("Step 1: Creating Drosophila brain...")
    fly = Drosophila(scale=args.scale)
    brain = fly.brain

    print(f"  ✓ Created brain with {brain.n_total} neurons")
    print(f"  ✓ Device: {brain.dev}")
    print()

    # Step 2: Initialize exporter
    print("Step 2: Initializing ONNX exporter...")
    exporter = ONNXExporter(brain, name="oNeura_Drosophila_Brain")

    print(f"  ✓ Exporter initialized")
    print(f"  ✓ Brain model: {exporter.name}")
    print()

    # Step 3: Export to ONNX
    print(f"Step 3: Exporting to ONNX (quantization: {args.quantize})...")

    quantization = None if args.quantize == 'none' else args.quantize
    output_path = exporter.export(
        args.output,
        quantization=quantization,
        input_size=50,   # Sensory input neurons
        output_size=20   # Motor output neurons
    )

    # Check file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"  ✓ Exported to: {output_path}")
        print(f"  ✓ File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    else:
        print(f"  ✗ Export failed")
        return

    # Step 4: MCU deployment info
    print()
    print("Step 4: MCU Deployment Analysis...")
    print()

    for mcu in ['cortex_m0', 'cortex_m4', 'cortex_m7']:
        meta = exporter.export_for_mcu(
            f"brain_{mcu}.onnx",
            mcu_type=mcu
        )
        print(f"  {mcu.upper()}:")
        print(f"    - Quantization: {meta['quantization']}")
        print(f"    - Flash required: {meta['flash_required_kb']} KB")
        print(f"    - RAM required: {meta['ram_required_kb']} KB")
        print(f"    - Latency: {meta['estimated_latency_ms']:.1f} ms")
        print()

    # Step 5: Model optimization
    print("Step 5: Model optimization...")
    optimizer = ModelOptimizer(brain)

    # Pruning
    pruned = optimizer.prune_synapses(threshold=0.05, method="magnitude")
    print(f"  ✓ Pruned {pruned:,} weak synapses")

    # Memory footprint
    footprint = optimizer.compute_memory_footprint()
    total_memory = sum(footprint.values())
    print(f"  ✓ Current memory footprint: {total_memory/1024:.1f} KB")

    # Distillation
    distilled = optimizer.distill_to_rate(temperature=10.0)
    print(f"  ✓ Rate-based distillation ready")

    print()
    print("="*60)
    print("  EXPORT COMPLETE!")
    print("="*60)
    print()
    print("Files created:")
    print(f"  - {output_path}")
    print(f"  - brain_cortex_m0.onnx")
    print(f"  - brain_cortex_m4.onnx")
    print(f"  - brain_cortex_m7.onnx")
    print()
    print("Next steps:")
    print("  1. Flash to MCU using ST-Link or similar")
    print("  2. Connect sensors (camera, IMU, GPS)")
    print("  3. Run neural control loop at 50Hz")
    print()


if __name__ == '__main__':
    main()
