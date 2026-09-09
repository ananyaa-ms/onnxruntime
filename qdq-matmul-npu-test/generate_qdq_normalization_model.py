#!/usr/bin/env python3
"""Generate activation-QDQ normalization test models based on CLIP and Gemma."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


@dataclass(frozen=True)
class QdqProfile:
    qdq_domain: str
    qdq_opset: int


QDQ_PROFILES = {
    "onnx": QdqProfile("", 23),
    "microsoft": QdqProfile("com.microsoft", 1),
}

DEFAULT_SHAPES = {
    "rmsnorm": (1, 2520, 768),
    "sslrn": (1, 2520, 768),
    "add-lpnorm-mul": (1, 512),
}

DEFAULT_OUTPUTS = {
    "rmsnorm": "gemma_dq_rmsnorm_q.onnx",
    "sslrn": "gemma_dq_sslrn_q.onnx",
    "add-lpnorm-mul": "clip_qdq_add_lpnorm_mul.onnx",
}

ACTIVATION_SCALE = np.float32(1.0 / 4096.0)
ACTIVATION_ZERO_POINT = np.uint16(32768)
NORM_SCALE_SCALE = np.float32(0.005)
NORM_SCALE_ZERO_POINT = np.uint8(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        choices=tuple(DEFAULT_SHAPES),
        default="rmsnorm",
        help="Operator pattern to generate; default: rmsnorm.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ONNX model path; defaults under unit-models.",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        metavar="DIM",
        help="Override the source-model-based input shape.",
    )
    parser.add_argument(
        "--qdq-profile",
        choices=tuple(QDQ_PROFILES),
        default="onnx",
        help="Q/DQ domain profile; default: onnx.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.input_shape is not None and any(
        dimension <= 0 for dimension in args.input_shape
    ):
        parser.error("--input-shape dimensions must be positive")
    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent
            / "unit-models"
            / DEFAULT_OUTPUTS[args.pattern]
        )
    return args


def add_qdq(
    nodes: list[onnx.NodeProto],
    tensor_name: str,
    output_name: str,
    scale_name: str,
    zero_point_name: str,
    profile: QdqProfile,
) -> None:
    quantized_name = f"{output_name}_quantized"
    nodes.extend(
        [
            helper.make_node(
                "QuantizeLinear",
                [tensor_name, scale_name, zero_point_name],
                [quantized_name],
                name=f"Quantize_{output_name}",
                domain=profile.qdq_domain,
            ),
            helper.make_node(
                "DequantizeLinear",
                [quantized_name, scale_name, zero_point_name],
                [output_name],
                name=f"Dequantize_{output_name}",
                domain=profile.qdq_domain,
            ),
        ]
    )


def add_output_qdq(
    nodes: list[onnx.NodeProto],
    tensor_name: str,
    profile: QdqProfile,
) -> None:
    nodes.extend(
        [
            helper.make_node(
                "QuantizeLinear",
                [tensor_name, "output_scale", "output_zero_point"],
                ["output_quantized"],
                name="QuantizeOutput",
                domain=profile.qdq_domain,
            ),
            helper.make_node(
                "DequantizeLinear",
                ["output_quantized", "output_scale", "output_zero_point"],
                ["output"],
                name="DequantizeOutput",
                domain=profile.qdq_domain,
            ),
        ]
    )


def make_initializers(
    pattern: str, hidden_size: int, rng: np.random.Generator
) -> list[onnx.TensorProto]:
    norm_scale = np.clip(
        rng.normal(1.0, 0.02, hidden_size), 0.9, 1.1
    ).astype(np.float32)
    norm_scale_quantized = np.clip(
        np.rint(norm_scale / NORM_SCALE_SCALE) + NORM_SCALE_ZERO_POINT,
        0,
        255,
    ).astype(np.uint8)
    multiplier = np.clip(
        rng.normal(1.0, 0.02, hidden_size), 0.9, 1.1
    ).astype(np.float32)
    multiplier_quantized = np.clip(
        np.rint(multiplier / NORM_SCALE_SCALE) + NORM_SCALE_ZERO_POINT,
        0,
        255,
    ).astype(np.uint8)

    initializers = [
        numpy_helper.from_array(ACTIVATION_SCALE, "activation_scale"),
        numpy_helper.from_array(
            np.asarray(ACTIVATION_ZERO_POINT), "activation_zero_point"
        ),
        numpy_helper.from_array(ACTIVATION_SCALE, "output_scale"),
        numpy_helper.from_array(
            np.asarray(ACTIVATION_ZERO_POINT), "output_zero_point"
        ),
        numpy_helper.from_array(
            np.asarray(NORM_SCALE_SCALE), "constant_scale"
        ),
        numpy_helper.from_array(
            np.asarray(NORM_SCALE_ZERO_POINT), "constant_zero_point"
        ),
    ]
    if pattern in {"rmsnorm", "sslrn"}:
        initializers.append(
            numpy_helper.from_array(
                norm_scale_quantized, "norm_scale_quantized"
            )
        )
    else:
        initializers.append(
            numpy_helper.from_array(
                multiplier_quantized, "multiplier_quantized"
            )
        )
    return initializers


def build_model(args: argparse.Namespace) -> onnx.ModelProto:
    profile = QDQ_PROFILES[args.qdq_profile]
    input_shape = (
        DEFAULT_SHAPES[args.pattern]
        if args.input_shape is None
        else tuple(args.input_shape)
    )
    hidden_size = input_shape[-1]
    initializers = make_initializers(
        args.pattern, hidden_size, np.random.default_rng(args.seed)
    )
    nodes: list[onnx.NodeProto] = []
    graph_inputs = [
        helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, input_shape
        )
    ]

    add_qdq(
        nodes,
        "input",
        "input_dequantized",
        "activation_scale",
        "activation_zero_point",
        profile,
    )

    if args.pattern == "rmsnorm":
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [
                    "norm_scale_quantized",
                    "constant_scale",
                    "constant_zero_point",
                ],
                ["norm_scale"],
                name="DequantizeNormScale",
                domain=profile.qdq_domain,
            )
        )
        nodes.append(
            helper.make_node(
                "RMSNormalization",
                ["input_dequantized", "norm_scale"],
                ["normalized"],
                name="RMSNormalization",
                axis=-1,
                epsilon=1e-6,
            )
        )
        add_output_qdq(nodes, "normalized", profile)
    elif args.pattern == "sslrn":
        graph_inputs.append(
            helper.make_tensor_value_info(
                "skip", TensorProto.FLOAT, input_shape
            )
        )
        add_qdq(
            nodes,
            "skip",
            "skip_dequantized",
            "activation_scale",
            "activation_zero_point",
            profile,
        )
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [
                    "norm_scale_quantized",
                    "constant_scale",
                    "constant_zero_point",
                ],
                ["norm_scale"],
                name="DequantizeNormScale",
                domain=profile.qdq_domain,
            )
        )
        nodes.append(
            helper.make_node(
                "SkipSimplifiedLayerNormalization",
                ["input_dequantized", "skip_dequantized", "norm_scale"],
                ["normalized"],
                name="SkipSimplifiedLayerNormalization",
                domain="com.microsoft",
                epsilon=1e-6,
            )
        )
        add_output_qdq(nodes, "normalized", profile)
    else:
        graph_inputs.append(
            helper.make_tensor_value_info(
                "addend", TensorProto.FLOAT, input_shape
            )
        )
        add_qdq(
            nodes,
            "addend",
            "addend_dequantized",
            "activation_scale",
            "activation_zero_point",
            profile,
        )
        nodes.append(
            helper.make_node(
                "Add",
                ["input_dequantized", "addend_dequantized"],
                ["added"],
                name="Add",
            )
        )
        add_qdq(
            nodes,
            "added",
            "added_dequantized",
            "activation_scale",
            "activation_zero_point",
            profile,
        )
        nodes.append(
            helper.make_node(
                "LpNormalization",
                ["added_dequantized"],
                ["normalized"],
                name="LpNormalization",
                axis=-1,
                p=2,
            )
        )
        add_qdq(
            nodes,
            "normalized",
            "normalized_dequantized",
            "activation_scale",
            "activation_zero_point",
            profile,
        )
        nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [
                    "multiplier_quantized",
                    "constant_scale",
                    "constant_zero_point",
                ],
                ["multiplier"],
                name="DequantizeMultiplier",
                domain=profile.qdq_domain,
            )
        )
        nodes.append(
            helper.make_node(
                "Mul",
                ["normalized_dequantized", "multiplier"],
                ["multiplied"],
                name="Mul",
            )
        )
        add_output_qdq(nodes, "multiplied", profile)

    graph = helper.make_graph(
        nodes,
        f"{args.pattern.replace('-', '_')}_qdq",
        graph_inputs,
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, input_shape
            )
        ],
        initializer=initializers,
    )
    opset_imports = [helper.make_opsetid("", 23)]
    if profile.qdq_domain:
        opset_imports.append(
            helper.make_opsetid(profile.qdq_domain, profile.qdq_opset)
        )
    if args.pattern == "sslrn" and profile.qdq_domain != "com.microsoft":
        opset_imports.append(helper.make_opsetid("com.microsoft", 1))

    model = helper.make_model(
        graph,
        producer_name=Path(__file__).name,
        opset_imports=opset_imports,
    )
    model.ir_version = 8
    model.metadata_props.add(key="pattern", value=args.pattern)
    model.metadata_props.add(key="qdq_profile", value=args.qdq_profile)
    model.metadata_props.add(
        key="input_shape", value="x".join(str(value) for value in input_shape)
    )
    model.metadata_props.add(
        key="reference_model",
        value=(
            "amd-clip/clip_vit_base_patch16_amd.onnx"
            if args.pattern == "add-lpnorm-mul"
            else "fp32-gemma4-e2b-it/vision_encoder/model.onnx"
        ),
    )
    model = shape_inference.infer_shapes(model)
    checker.check_model(model, full_check=True)
    return model


def main() -> None:
    args = parse_args()
    model = build_model(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    print(f"Saved validated ONNX model to {args.output}")


if __name__ == "__main__":
    main()
