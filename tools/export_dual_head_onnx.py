#!/usr/bin/env python3
"""
ONNX Graph Surgery Tool:
Directly processes a standard YOLO semantic segmentation ONNX model (which outputs ArgMax)
and exposes the internal raw float32 logits as a second output head.

Output heads:
- 'class_map': ArgMax class prediction per pixel (uint8, [B, H, W])
- 'logits': Raw float32 probabilities/logits (float32, [B, nc, H, W])
"""

import argparse
import sys
from pathlib import Path
import onnx
from onnx import helper, TensorProto, shape_inference

def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Surgery: Expose internal logits from a standard YOLO ONNX model")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the standard YOLO ONNX model (outputs argmax)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save the final dual-head ONNX model (default: [input_stem]_dual_head.onnx)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input ONNX model not found at {input_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_dual_head.onnx"

    print(f"Loading standard ONNX model from {input_path}...")
    try:
        onnx_model = onnx.load(input_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        sys.exit(1)

    # 1. Run ONNX shape inference to resolve intermediate tensor shapes/types
    print("Running ONNX shape inference...")
    inferred_model = shape_inference.infer_shapes(onnx_model)
    graph = inferred_model.graph

    # 2. Find the ArgMax node
    print("Searching for ArgMax node in the graph...")
    argmax_node = None
    for node in graph.node:
        if node.op_type == "ArgMax":
            argmax_node = node
            break

    if argmax_node is None:
        print("Error: Could not find an ArgMax node in the ONNX graph. Is this a standard YOLO segmentation model?")
        sys.exit(1)

    logits_tensor_name = argmax_node.input[0]
    print(f"Found ArgMax node input tensor (raw logits): '{logits_tensor_name}'")

    # 3. Retrieve shape/type information for the logits tensor from value_info
    logits_vi = None
    for vi in graph.value_info:
        if vi.name == logits_tensor_name:
            logits_vi = vi
            break

    if logits_vi is None:
        # If not found in value_info, try graph.input or graph.output (unlikely, but safe backup)
        for out in graph.output:
            if out.name == logits_tensor_name:
                logits_vi = out
                break

    if logits_vi is None:
        print(f"Error: Could not find shape info for logits tensor '{logits_tensor_name}'.")
        sys.exit(1)

    # 4. Create an Identity node to output 'out_logits'
    print("Creating Identity node to expose logits...")
    identity_node = helper.make_node(
        op_type="Identity",
        inputs=[logits_tensor_name],
        outputs=["out_logits"],
        name="Expose_Logits_Identity"
    )
    graph.node.append(identity_node)

    # Define the new 'out_logits' output head
    logits_output = helper.make_tensor_value_info(
        name="out_logits",
        elem_type=TensorProto.FLOAT,
        shape=None
    )
    # Copy resolved shape dimensions
    for dim in logits_vi.type.tensor_type.shape.dim:
        logits_output.type.tensor_type.shape.dim.append(dim)
    graph.output.append(logits_output)

    # 5. Rename the default output head to 'out_ids'
    original_output = graph.output[0]
    original_output_name = original_output.name
    print(f"Renaming default output head from '{original_output_name}' to 'out_ids'...")
    
    original_output.name = "out_ids"
    for node in graph.node:
        for i, output_name in enumerate(node.output):
            if output_name == original_output_name:
                node.output[i] = "out_ids"

    # 6. Verify and save the modified model
    print("Verifying modified ONNX model...")
    try:
        onnx.checker.check_model(inferred_model)
    except Exception as e:
        print(f"Warning: ONNX model checker failed: {e}")

    try:
        onnx.save(inferred_model, output_path)
        print(f"Success! Dual-head ONNX model successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error saving modified ONNX model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
