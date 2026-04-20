"""Racket-head detection via YOLOv11n (COCO class 38 = "tennis racket").

On import we lazily download `yolo11n.pt` to railway-service/models/ and export
to ONNX. Inference at runtime goes through onnxruntime so we don't pay for a
hot PyTorch runtime.

The racket-head point is the bbox corner farthest from the dominant wrist — a
cheap heuristic that tracks the swing tip well enough for the tracer without
needing a keypoint-level racket model.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import numpy as np

# Pin: ultralytics==8.3.40 (see requirements.txt) — that version bundles a
# specific yolo11n.pt checkpoint, so pinning the package pins the weights.
MODEL_DIR = Path(__file__).parent / "models"
YOLO_PT_PATH = MODEL_DIR / "yolo11n.pt"
YOLO_ONNX_PATH = MODEL_DIR / "yolo11n.onnx"

# COCO class 38 = tennis racket
TENNIS_RACKET_CLASS_ID = 38
CONFIDENCE_THRESHOLD = 0.3
INPUT_SIZE = 640

_session = None
_input_name: Optional[str] = None
_init_lock = Lock()


RacketHead = dict  # {"x": float, "y": float, "confidence": float}


def _ensure_model() -> None:
    """Download the .pt checkpoint and export to ONNX if not yet present."""
    global _session, _input_name

    if _session is not None:
        return

    with _init_lock:
        if _session is not None:
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if not YOLO_ONNX_PATH.exists():
            # Import torch-heavy ultralytics only at setup; swap to onnxruntime
            # for steady-state inference.
            from ultralytics import YOLO

            model = YOLO(str(YOLO_PT_PATH) if YOLO_PT_PATH.exists() else "yolo11n.pt")
            exported = model.export(format="onnx", imgsz=INPUT_SIZE, opset=12)
            exported_path = Path(exported)
            if exported_path != YOLO_ONNX_PATH:
                exported_path.replace(YOLO_ONNX_PATH)

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(YOLO_ONNX_PATH), providers=providers)
        _input_name = sess.get_inputs()[0].name
        _session = sess


def _letterbox(
    img: np.ndarray, new_shape: int = INPUT_SIZE
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize+pad to a square while preserving aspect ratio. Returns the
    padded image, the scale factor used, and (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2

    import cv2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return padded, r, (pad_w, pad_h)


def _preprocess(frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    img, scale, pad = _letterbox(frame_bgr, INPUT_SIZE)
    # BGR -> RGB, HWC -> CHW, normalize to [0,1]
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]  # (1, 3, H, W)
    return np.ascontiguousarray(img), scale, pad


def _best_racket_box(
    output: np.ndarray, scale: float, pad: Tuple[int, int]
) -> Optional[Tuple[float, float, float, float, float]]:
    """Parse YOLOv8/v11 ONNX output and return (x1, y1, x2, y2, conf) in the
    original image's pixel space, or None."""
    # Output is typically (1, 84, N): 4 box coords + 80 class scores.
    if output.ndim == 3:
        output = output[0]
    if output.shape[0] < output.shape[1]:
        output = output.T  # -> (N, 84)

    if output.shape[1] < 5 + TENNIS_RACKET_CLASS_ID + 1:
        return None

    cls_scores = output[:, 4 + TENNIS_RACKET_CLASS_ID]
    best_idx = int(np.argmax(cls_scores))
    best_conf = float(cls_scores[best_idx])
    if best_conf < CONFIDENCE_THRESHOLD:
        return None

    cx, cy, w, h = output[best_idx, :4]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Undo letterbox
    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    return float(x1), float(y1), float(x2), float(y2), best_conf


def detect_racket(
    frame_bgr: np.ndarray,
    dominant_wrist_xy: Optional[Tuple[float, float]] = None,
) -> Optional[RacketHead]:
    """Run YOLOv11n on *frame_bgr* and return the racket-head point as
    normalized (x, y) plus confidence, or None if no confident detection.

    `dominant_wrist_xy` is in pixel space (same coord system as frame_bgr).
    The head point is the bbox corner farthest from the wrist; if no wrist is
    given we fall back to the bbox center.
    """
    try:
        _ensure_model()
    except Exception as e:  # noqa: BLE001
        print(f"[racket_detector] model load failed: {e}")
        return None

    if _session is None or _input_name is None:
        return None

    h, w = frame_bgr.shape[:2]
    tensor, scale, pad = _preprocess(frame_bgr)

    try:
        outputs = _session.run(None, {_input_name: tensor})
    except Exception as e:  # noqa: BLE001
        print(f"[racket_detector] inference error: {e}")
        return None

    box = _best_racket_box(outputs[0], scale, pad)
    if box is None:
        return None

    x1, y1, x2, y2, conf = box

    if dominant_wrist_xy is not None:
        wx, wy = dominant_wrist_xy
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        head_x, head_y = max(
            corners,
            key=lambda p: (p[0] - wx) ** 2 + (p[1] - wy) ** 2,
        )
    else:
        head_x = (x1 + x2) / 2
        head_y = (y1 + y2) / 2

    # Normalize to [0,1], matching Landmark.x/y space.
    return {
        "x": max(0.0, min(1.0, head_x / w)),
        "y": max(0.0, min(1.0, head_y / h)),
        "confidence": round(conf, 3),
    }
