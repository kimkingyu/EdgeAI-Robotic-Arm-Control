#!/usr/bin/env python3
"""Independent half-pixel, Q=2048 bilinear BGR-to-RGB numerical reference.

Coordinates and weight quantization use exact ``Fraction`` arithmetic, not the
integer-coordinate/floorDiv implementation intended for MLIR. The small scalar
oracle also blends normalized Fraction weights independently of ``axis_weights``.
There is one final round-half-up, after all four neighbors are combined. Borders
replicate the nearest source pixel. This is NOT a bit-exact OpenCV INTER_LINEAR
claim; OpenCV, if available, is only a separate differential comparison.

The scalar APIs need only the standard library. The array APIs explicitly fail
when NumPy cannot be imported. Array inputs are uint8 HWC BGR with exactly three
channels and byte strides (row >= width * 3, 3, 1), all positive. Positive-stride
row-padded ROIs are accepted without copying the input. Dimensions are 1..8192.
"""

from __future__ import annotations

from fractions import Fraction
from math import floor
from numbers import Integral
from typing import NamedTuple

try:
    import numpy as np
except ImportError as error:
    np = None
    _NUMPY_IMPORT_ERROR = error
else:
    _NUMPY_IMPORT_ERROR = None

Q = 2048
Q_SQUARED = Q * Q
MAX_DIMENSION = 8192
# Nonnegative bilinear weights sum to Q**2, so this is the largest biased sum.
MAX_BIASED_ACCUMULATOR = 255 * Q_SQUARED + Q_SQUARED // 2
_HALF = Fraction(1, 2)
_MAX_TILE_PIXELS = 65536


class AxisWeights(NamedTuple):
    """Clamped source indices and nonnegative integer weights summing to Q."""

    i0: int
    i1: int
    w0: int
    w1: int


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean or float")
    return int(value)


def _dimension(value, name: str) -> int:
    value = _integer(value, name)
    if not 1 <= value <= MAX_DIMENSION:
        raise ValueError(f"{name} must be in 1..{MAX_DIMENSION}")
    return value


def axis_weights(src_size: int, dst_size: int) -> list[AxisWeights]:
    """Return (i0, i1, w0, w1) for each destination coordinate.

    For coordinate s = (d + 1/2) * src_size / dst_size - 1/2, take
    base = floor(s), alpha = s - base, w1 = floor(alpha * Q + 1/2),
    w0 = Q - w1. Only the two integer indices are clamped, independently;
    in particular, a negative coordinate is floored, never truncated to zero.
    Coordinates and alpha remain Fraction objects until weight quantization.
    """
    src_size = _dimension(src_size, "src_size")
    dst_size = _dimension(dst_size, "dst_size")
    result = []
    scale = Fraction(src_size, dst_size)
    for destination in range(dst_size):
        coordinate = (Fraction(destination) + _HALF) * scale - _HALF
        base = floor(coordinate)
        alpha = coordinate - base
        w1 = floor(alpha * Q + _HALF)
        result.append(AxisWeights(
            min(max(base, 0), src_size - 1),
            min(max(base + 1, 0), src_size - 1),
            Q - w1,
            w1,
        ))
    return result


def _scalar_shape(image) -> tuple[int, int]:
    """Validate a small rectangular list/tuple of BGR integer triples."""
    if not isinstance(image, (list, tuple)):
        raise TypeError("scalar image must be a nested list or tuple")
    height = _dimension(len(image), "input height")
    if not isinstance(image[0], (list, tuple)):
        raise TypeError("scalar image rows must be lists or tuples")
    width = _dimension(len(image[0]), "input width")
    for row in image:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError("scalar image must have rectangular rows")
        for pixel in row:
            if not isinstance(pixel, (list, tuple)) or len(pixel) != 3:
                raise ValueError("scalar image pixels must have three BGR channels")
            for value in pixel:
                value = _integer(value, "pixel component")
                if not 0 <= value <= 255:
                    raise ValueError("pixel components must be uint8 values in 0..255")
    return height, width


def _fraction_pixel(image, height: int, width: int, out_height: int,
                    out_width: int, y: int, x: int) -> tuple[int, int, int]:
    # Deliberately do not call axis_weights: this small oracle derives its own
    # rational taps and sums pixel * Fraction * Fraction, not integer products.
    taps = []
    for destination, source_size, target_size in (
            (y, height, out_height), (x, width, out_width)):
        center = Fraction(destination, target_size) + Fraction(1, 2 * target_size)
        source_position = center * source_size - _HALF
        left = floor(source_position)
        right_weight = Fraction(floor((source_position - left) * Q + _HALF), Q)
        taps.append((
            (min(max(left, 0), source_size - 1), 1 - right_weight),
            (min(max(left + 1, 0), source_size - 1), right_weight),
        ))
    rgb = []
    for channel in (2, 1, 0):
        value = sum(
            (Fraction(int(image[row][column][channel])) * weight_y * weight_x
             for row, weight_y in taps[0]
             for column, weight_x in taps[1]),
            Fraction(0),
        )
        rgb.append(min(255, max(0, floor(value + _HALF))))
    return tuple(rgb)


def reference_pixel_rgb(image, out_height: int, out_width: int,
                        y: int, x: int) -> tuple[int, int, int]:
    """Return one RGB destination pixel using only exact standard-library math."""
    height, width = _scalar_shape(image)
    out_height = _dimension(out_height, "out_height")
    out_width = _dimension(out_width, "out_width")
    y, x = _integer(y, "y"), _integer(x, "x")
    if not 0 <= y < out_height or not 0 <= x < out_width:
        raise ValueError("destination pixel coordinates are outside the output")
    return _fraction_pixel(image, height, width, out_height, out_width, y, x)


def reference_resize_rgb_scalar(image, out_height: int,
                                out_width: int) -> list[list[tuple[int, int, int]]]:
    """Resize small nested BGR lists/tuples to new nested lists of RGB tuples.

    Python integer components in 0..255 represent uint8 samples. This intentionally
    slow Fraction oracle is for small fixtures, not production-size images.
    """
    height, width = _scalar_shape(image)
    out_height = _dimension(out_height, "out_height")
    out_width = _dimension(out_width, "out_width")
    return [[_fraction_pixel(image, height, width, out_height, out_width, y, x)
             for x in range(out_width)] for y in range(out_height)]


def _require_numpy():
    if np is None:
        raise RuntimeError(
            "NumPy is unavailable: array reference_resize_rgb and "
            "difference_statistics cannot run. Use axis_weights, "
            "reference_pixel_rgb or reference_resize_rgb_scalar instead."
        ) from _NUMPY_IMPORT_ERROR
    return np


def _array_shape(image, name: str) -> tuple[int, int]:
    numpy = _require_numpy()
    if not isinstance(image, numpy.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray; no implicit conversion")
    if image.dtype != numpy.dtype(numpy.uint8):
        raise TypeError(f"{name} must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have HWC shape with exactly three channels")
    height = _dimension(image.shape[0], f"{name} height")
    width = _dimension(image.shape[1], f"{name} width")
    row_stride, pixel_stride, channel_stride = image.strides
    if row_stride < width * 3 or pixel_stride != 3 or channel_stride != 1:
        raise ValueError(
            f"{name} needs positive byte strides (row >= width * 3, 3, 1); "
            f"got {image.strides}. Negative/overlapping/sparse layouts are rejected."
        )
    return height, width


def reference_resize_rgb(image: np.ndarray, out_height: int,
                         out_width: int) -> np.ndarray:
    """Vectorized four-neighbor Q=2048 reference; return new contiguous uint8 RGB.

    Coordinates come from Fraction-based axis_weights. Each tile uses int64
    products and accumulation, then floor((sum + Q**2/2) / Q**2) and saturation.
    Row tiling bounds temporary storage without a Python loop over output pixels.
    The maximum biased sum is 1071644672 < 2**31, although int64 is used here
    explicitly to avoid uint8 overflow or dependence on implicit promotions.
    """
    numpy = _require_numpy()
    height, width = _array_shape(image, "image")
    out_height = _dimension(out_height, "out_height")
    out_width = _dimension(out_width, "out_width")
    vertical = numpy.array(axis_weights(height, out_height), dtype=numpy.int64)
    horizontal = numpy.array(axis_weights(width, out_width), dtype=numpy.int64)
    output = numpy.empty((out_height, out_width, 3), dtype=numpy.uint8, order="C")
    tile_rows = max(1, _MAX_TILE_PIXELS // out_width)
    for start in range(0, out_height, tile_rows):
        stop = min(start + tile_rows, out_height)
        rows = vertical[start:stop]
        accumulator = numpy.zeros((stop - start, out_width, 3), dtype=numpy.int64)
        for y_tap, x_tap in ((0, 0), (0, 1), (1, 0), (1, 1)):
            samples = image[rows[:, y_tap, None], horizontal[None, :, x_tap], :]
            weights = (rows[:, y_tap + 2, None, None]
                       * horizontal[None, :, x_tap + 2, None])
            accumulator += samples.astype(numpy.int64) * weights
        accumulator += Q_SQUARED // 2
        accumulator //= Q_SQUARED
        numpy.clip(accumulator, 0, 255, out=accumulator)
        output[start:stop] = accumulator[:, :, ::-1]
    return output


def difference_statistics(actual: np.ndarray, expected: np.ndarray,
                          max_worst: int = 8) -> dict:
    """Compare same-shaped uint8 HWC RGB arrays with the accepted input layout.

    Ratios have different denominators: different_component_ratio counts H*W*3
    scalar channels; different_pixel_ratio counts H*W pixels with ANY bad channel.
    mae is the mean absolute component error. Worst coordinates include only
    nonzero errors, descending by absolute error then ascending (y, x, channel).
    channel is RGB index 0/1/2; signed_error is actual - expected. At most
    max_worst records are returned (zero disables records but not other stats).
    error_bounding_box is None for equality, otherwise inclusive y_min/x_min/
    y_max/x_max over all differing pixels, not just the reported worst ones.
    """
    numpy = _require_numpy()
    height, width = _array_shape(actual, "actual")
    expected_shape = _array_shape(expected, "expected")
    if (height, width) != expected_shape:
        raise ValueError("actual and expected must have identical shapes")
    max_worst = _integer(max_worst, "max_worst")
    if max_worst < 0:
        raise ValueError("max_worst must be nonnegative")
    errors = numpy.abs(actual.astype(numpy.int16) - expected.astype(numpy.int16))
    different_pixels = numpy.any(errors != 0, axis=2)
    component_count = int(numpy.count_nonzero(errors))
    pixel_count = int(numpy.count_nonzero(different_pixels))
    bounding_box = None
    if pixel_count:
        ys = numpy.flatnonzero(numpy.any(different_pixels, axis=1))
        xs = numpy.flatnonzero(numpy.any(different_pixels, axis=0))
        bounding_box = {"y_min": int(ys[0]), "x_min": int(xs[0]),
                        "y_max": int(ys[-1]), "x_max": int(xs[-1])}

    worst = []
    remaining = min(max_worst, component_count)
    if remaining:
        flat = errors.ravel()
        # Bounded-size bins and scans avoid a full H*W*3 int64 argsort array.
        histogram = numpy.zeros(256, dtype=numpy.int64)
        for start in range(0, flat.size, _MAX_TILE_PIXELS):
            histogram += numpy.bincount(flat[start:start + _MAX_TILE_PIXELS],
                                        minlength=256)
        for magnitude in range(255, 0, -1):
            if not histogram[magnitude]:
                continue
            for start in range(0, flat.size, _MAX_TILE_PIXELS):
                matches = numpy.flatnonzero(
                    flat[start:start + _MAX_TILE_PIXELS] == magnitude)
                for position in matches[:remaining]:
                    y, offset = divmod(start + int(position), width * 3)
                    x, channel = divmod(offset, 3)
                    actual_value = int(actual[y, x, channel])
                    expected_value = int(expected[y, x, channel])
                    worst.append({"y": y, "x": x, "channel": channel,
                                  "actual": actual_value, "expected": expected_value,
                                  "signed_error": actual_value - expected_value,
                                  "abs_error": magnitude})
                remaining -= min(remaining, len(matches))
                if not remaining:
                    break
            if not remaining:
                break
    return {
        "max_abs_error": int(errors.max()),
        "mae": float(errors.mean(dtype=numpy.float64)),
        "different_component_ratio": component_count / (height * width * 3),
        "different_pixel_ratio": pixel_count / (height * width),
        "worst_coordinates": worst,
        "error_bounding_box": bounding_box,
    }
