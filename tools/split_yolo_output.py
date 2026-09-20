#!/usr/bin/env python3
"""把 YOLOv8 的共享 [4+80,8400] 输出拆成两个独立输出。

原始 RKNN INT8 的问题不是量化算法，而是坐标与分类分数共用一个量化范围：
坐标 0~640 把分数 0~1 压成了零。拆分图输出后，RKNN 可以为两组张量分别
选择量化参数。

默认暴露：
  /model.22/Mul_2_output_0       [1, 4, 8400]  坐标
  /model.22/Sigmoid_output_0      [1,80,8400]  分类分数
"""
import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, checker, shape_inference, helper

DEFAULT_COORDS = "/model.22/Mul_2_output_0"
DEFAULT_SCORES = "/model.22/Sigmoid_output_0"


def value_info_map(model):
    result = {}
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        result[item.name] = item
    return result


def dims(value_info):
    return [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            for dim in value_info.type.tensor_type.shape.dim]


def split_model(input_path: Path, output_path: Path, coords_name=DEFAULT_COORDS,
                scores_name=DEFAULT_SCORES):
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    model = onnx.load(str(input_path))
    inferred = shape_inference.infer_shapes(model)
    infos = value_info_map(inferred)
    if model.graph.input[0].name != "images":
        raise ValueError("unexpected input name: " + model.graph.input[0].name)
    if len(model.graph.output) != 1 or model.graph.output[0].name != "output0":
        raise ValueError("expected original single output output0")
    original_shape = dims(infos["output0"])
    if original_shape != [1, 84, 8400]:
        raise ValueError("unexpected original output shape: %s" % original_shape)
    for name, expected in ((coords_name, [1, 4, 8400]), (scores_name, [1, 80, 8400])):
        if name not in infos:
            raise ValueError("output tensor not found in inferred graph: " + name)
        actual = dims(infos[name])
        if actual != expected:
            raise ValueError("%s shape %s != expected %s" % (name, actual, expected))
        if infos[name].type.tensor_type.elem_type != TensorProto.FLOAT:
            raise ValueError("%s is not float output" % name)

    outputs = []
    for name in (coords_name, scores_name):
        # Use inferred type/shape, but make a fresh ValueInfo so graph.output is clean.
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT,
                                                     [int(x) for x in dims(infos[name])]))
    del model.graph.output[:]
    model.graph.output.extend(outputs)
    model.producer_name = "EdgeAI YOLO split-output exporter"
    model.producer_version = "1"
    checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return {
        "input": str(input_path), "output": str(output_path),
        "original_outputs": [{"name": "output0", "shape": original_shape}],
        "split_outputs": [
            {"name": coords_name, "shape": [1, 4, 8400], "role": "coordinates"},
            {"name": scores_name, "shape": [1, 80, 8400], "role": "class_scores"},
        ],
        "node_count": len(model.graph.node),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coords-name", default=DEFAULT_COORDS)
    parser.add_argument("--scores-name", default=DEFAULT_SCORES)
    args = parser.parse_args(argv)
    result = split_model(args.input, args.output, args.coords_name, args.scores_name)
    print("split output: %s" % result["output"])
    for item in result["split_outputs"]:
        print("  %-34s %s (%s)" % (item["name"], item["shape"], item["role"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
