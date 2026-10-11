"""Vision exports; optional camera/NPU dependencies load only when requested."""

__all__ = ["USBCamera", "RKNNObjectDetector"]


def __getattr__(name):
    if name == "USBCamera":
        from .camera import USBCamera
        value = USBCamera
    elif name == "RKNNObjectDetector":
        from .rknn_detector import RKNNObjectDetector
        value = RKNNObjectDetector
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
