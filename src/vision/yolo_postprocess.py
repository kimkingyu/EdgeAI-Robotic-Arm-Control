#!/usr/bin/env python3
"""YOLOv8 检测头后处理：把 [1, 84, 8400] 解成可用的检测框。

之前这一步是空的，NPU 算出的张量被直接丢弃，所以整条"检测 → 抓取坐标"的
链路是断的。这里只做解码，不碰坐标变换与控制。

布局说明（实测确认，非假设）：
  84 = 4 (cx, cy, w, h) + 80 (COCO 类别分数)
  8400 = 三个尺度的 anchor 点数 80*80 + 40*40 + 20*20
  坐标已是网络输入尺度（640）下的像素值，不是归一化值。
"""
from typing import Any, Dict, List, Optional, Sequence

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class PostprocessError(ValueError):
    """输出张量不符合预期布局时抛出，不静默返回空列表。"""


def _as_matrix(tensor, role: str) -> "np.ndarray":
    """规范化一个输出为 (attributes, anchors)。"""
    tensor = np.asarray(tensor)
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise PostprocessError("%s 暂不支持 batch>1，收到 %s" % (role, tensor.shape))
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise PostprocessError("%s 期望二维输出，收到 %s" % (role, tensor.shape))
    if tensor.shape[0] > tensor.shape[1]:
        tensor = tensor.T
    return tensor


def _as_prediction(outputs) -> "np.ndarray":
    """取出单输出并规范化为 (num_attrs, num_anchors)。"""
    if not HAS_NUMPY:
        raise PostprocessError("后处理需要 numpy")
    if outputs is None:
        raise PostprocessError("模型输出为 None")
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise PostprocessError("单输出布局收到 %d 个输出；双输出应为 4+80 属性张量" % len(outputs))
        tensor = outputs[0]
    else:
        tensor = outputs
    tensor = _as_matrix(tensor, "prediction")
    if tensor.shape[0] < 5:
        raise PostprocessError("属性维过小（%d），不是检测头输出" % tensor.shape[0])
    return tensor


def _as_prediction_parts(outputs):
    """返回 (坐标[4,N], 分数[C,N])，兼容单输出与拆分双输出。"""
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
        parts = [_as_matrix(item, "split_output_%d" % index)
                 for index, item in enumerate(outputs)]
        coords = next((item for item in parts if item.shape[0] == 4), None)
        scores = next((item for item in parts if item.shape[0] > 4), None)
        if coords is None or scores is None or scores.shape[0] < 2:
            raise PostprocessError("双输出必须包含 [4,N] 坐标与 [C,N] 分类分数，收到 %s"
                                   % [item.shape for item in parts])
        if coords.shape[1] != scores.shape[1]:
            raise PostprocessError("坐标与分数 anchor 数不一致：%s vs %s"
                                   % (coords.shape, scores.shape))
        return coords, scores
    prediction = _as_prediction(outputs)
    return prediction[:4], prediction[4:]


def nms(boxes: "np.ndarray", scores: "np.ndarray", threshold: float) -> List[int]:
    """按 IoU 抑制重叠框。boxes 为 xyxy。返回保留下标（按分数降序）。"""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 框宽高按闭区间算，避免零面积框在 IoU 里produce 除零。
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= threshold]
    return keep


def decode(outputs,
           orig_shape: Sequence[int],
           input_size: Sequence[int] = (640, 640),
           conf_thresh: float = 0.25,
           nms_thresh: float = 0.45,
           class_names: Optional[Sequence[str]] = None,
           max_detections: int = 100) -> List[Dict[str, Any]]:
    """解码检测头输出。

    orig_shape : 原始画面的 (h, w[, c])，用于把坐标还原回去。
    input_size : 网络输入的 (h, w)。当前预处理是直接 resize，没有 letterbox，
                 所以 x/y 方向各自独立缩放；若将来改用 letterbox，这里必须同步改。
    """
    coords, scores_all = _as_prediction_parts(outputs)
    num_classes = scores_all.shape[0]
    names = tuple(class_names) if class_names is not None else COCO_CLASSES
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2 and \
            class_names is None and num_classes != len(COCO_CLASSES):
        raise PostprocessError("默认双输出 YOLOv8 类别维应为 %d，收到 %d；请显式提供 class_names"
                               % (len(COCO_CLASSES), num_classes))
    if num_classes < 1:
        raise PostprocessError("类别维为 %d，输出布局异常" % num_classes)
    if len(names) < num_classes:
        # 类别表短于实际输出时补占位名，而不是越界取值。
        names = tuple(names) + tuple("class_%d" % i for i in range(len(names), num_classes))

    boxes_cxcywh = coords.T
    class_ids = scores_all.argmax(axis=0)
    scores = scores_all.max(axis=0)

    keep_mask = scores >= conf_thresh
    if not np.any(keep_mask):
        return []
    boxes_cxcywh = boxes_cxcywh[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    cx, cy, w, h = (boxes_cxcywh[:, 0], boxes_cxcywh[:, 1],
                    boxes_cxcywh[:, 2], boxes_cxcywh[:, 3])
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # 逐类别做 NMS，避免不同类别的重叠目标互相抑制。
    keep: List[int] = []
    for class_id in np.unique(class_ids):
        idx = np.nonzero(class_ids == class_id)[0]
        kept = nms(boxes[idx], scores[idx], nms_thresh)
        keep.extend(int(idx[k]) for k in kept)
    if not keep:
        return []
    keep.sort(key=lambda i: float(scores[i]), reverse=True)
    keep = keep[:max_detections]

    orig_h, orig_w = int(orig_shape[0]), int(orig_shape[1])
    in_h, in_w = int(input_size[0]), int(input_size[1])
    if orig_h < 1 or orig_w < 1 or in_h < 1 or in_w < 1:
        raise PostprocessError("尺寸必须为正：orig=%s input=%s" % (orig_shape, input_size))
    scale_x, scale_y = orig_w / in_w, orig_h / in_h

    results: List[Dict[str, Any]] = []
    for index in keep:
        x1 = float(np.clip(boxes[index, 0] * scale_x, 0, orig_w - 1))
        y1 = float(np.clip(boxes[index, 1] * scale_y, 0, orig_h - 1))
        x2 = float(np.clip(boxes[index, 2] * scale_x, 0, orig_w - 1))
        y2 = float(np.clip(boxes[index, 3] * scale_y, 0, orig_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue  # clamp 后退化为零面积的框直接丢弃
        class_id = int(class_ids[index])
        results.append({
            "class_id": class_id,
            "class_name": names[class_id] if class_id < len(names) else "class_%d" % class_id,
            "score": float(scores[index]),
            "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "center": (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))),
        })
    return results
