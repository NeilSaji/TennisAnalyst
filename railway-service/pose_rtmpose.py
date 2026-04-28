"""RTMPose top-down pose estimation backend for the tennis analyzer.

Pipeline per frame:
  1. YOLO11n person detector → highest-confidence person bbox (xyxy px).
  2. rtmlib.RTMPose-m on the bbox → 17 COCO keypoints in original-image
     pixel coords + per-keypoint confidence (rtmlib owns the letterbox,
     SimCC decode, and inverse transform).
  3. COCO-17 → BlazePose-33 id remap so the rest of the codebase keeps
     reading the same landmark schema.

The library handles the geometry that the prior browser crop attempt
got wrong (cea292d / 349370d). Our code only owns the bbox pick and the
id remap.

Why reuse the YOLO11n model from racket_detector.py instead of letting
rtmlib pull its bundled YOLOX:
  * One detector weight on disk, not two (Railway image stays small).
  * Tracks racket_detector's pinned ultralytics version for free.
  * Lets us share the same person-bbox call site if a future feature
    wants the bbox separately (e.g., tighter framing for the LLM).

Schema:
  * Emit a 33-entry landmark array indexed by BlazePose ids.
  * Ids that COCO-17 doesn't fill (1-10 face except 0/nose, 17/18, 19/20,
    21/22, 29-32) carry visibility=0 so the render-side cutoff drops them.
  * compute_joint_angles_from_dicts will see no index_finger landmarks,
    so right_wrist / left_wrist joint angles will not be emitted -- this
    is the schema_version=3 contract.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

# Lazy imports so unit tests can run without heavy weights being present:
# rtmlib gets imported on first inference, not at module import time.
_rtm_session = None  # type: ignore[assignment]
_yolo_session = None  # type: ignore[assignment]
_yolo_input_name: Optional[str] = None
_init_lock = Lock()
# Persistent init failures so we don't retry an expensive YOLO export per
# frame after the first failure. Whatever the first error was is re-raised
# on every subsequent call until _reset_for_tests() clears it.
_yolo_init_error: Optional[Exception] = None
_rtm_init_error: Optional[Exception] = None

MODEL_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------------------
# Person detection (YOLO11n COCO class 0)
# ---------------------------------------------------------------------------

YOLO_PT_PATH = MODEL_DIR / "yolo11n.pt"
YOLO_ONNX_PATH = MODEL_DIR / "yolo11n.onnx"
PERSON_CLASS_ID = 0
PERSON_CONFIDENCE_THRESHOLD = 0.3
YOLO_INPUT_SIZE = 640  # matches racket_detector.INPUT_SIZE — both share yolo11n.onnx

# Bbox padding around the YOLO detection before we pass it to RTMPose.
# RTMPose's preprocessing already adds a 1.25 padding factor on the
# input bbox to crop a square-ish region around the person, but it
# expects the bbox to roughly contain the player. We expand 8% on all
# sides so YOLO's tighter crop doesn't clip the racket arm at full
# extension during contact (a common failure with small-in-frame players
# where YOLO snaps to the body and misses the racket reach).
BBOX_EXPAND_PCT = 0.08

# ---------------------------------------------------------------------------
# Pose model (RTMPose-m, body 17-keypoint, 256x192 input)
# ---------------------------------------------------------------------------

# rtmlib's bundled URL for RTMPose-m body7 256x192 ONNX. Pinning the URL
# (not just the rtmlib version) so a future rtmlib release can't change
# the weights under us silently.
RTMPOSE_ONNX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
RTMPOSE_INPUT_SIZE = (192, 256)  # (W, H) -- rtmlib convention

# COCO-17 -> BlazePose-33 id mapping. See docs/pose-research.md §2 for
# the rationale; ids on the BlazePose side that COCO-17 doesn't cover
# stay at visibility=0 in the emitted landmark array.
#
# COCO-17 indices (rtmlib output order):
#   0 nose, 1 L eye, 2 R eye, 3 L ear, 4 R ear,
#   5 L shoulder, 6 R shoulder, 7 L elbow, 8 R elbow,
#   9 L wrist, 10 R wrist,
#   11 L hip, 12 R hip,
#   13 L knee, 14 R knee, 15 L ankle, 16 R ankle
COCO17_TO_BLAZEPOSE33 = {
    0: 0,    # nose
    1: 2,    # left_eye         (BlazePose 1=left_eye_inner is closer in id but 2=left_eye matches semantics)
    2: 5,    # right_eye
    3: 7,    # left_ear
    4: 8,    # right_ear
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}

BLAZEPOSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
NUM_BLAZEPOSE_LANDMARKS = 33

# ---------------------------------------------------------------------------
# Lazy initializers
# ---------------------------------------------------------------------------


def _ensure_yolo() -> None:
    """Lazy-export YOLO11n to ONNX and cache an onnxruntime session.

    Mirrors racket_detector._ensure_model so we don't ship two
    download/export pipelines for the same weight. If the first init
    raises (missing onnxscript, no internet, broken weights file), the
    error is sticky -- subsequent calls re-raise immediately instead of
    re-attempting an export that already failed once. Without this, the
    rtmpose extract loop would re-export YOLO once per frame, which on
    a 30s @ 30fps clip is 900 useless retry cycles.
    """
    global _yolo_session, _yolo_input_name, _yolo_init_error

    if _yolo_session is not None:
        return
    if _yolo_init_error is not None:
        raise _yolo_init_error

    with _init_lock:
        if _yolo_session is not None:
            return
        if _yolo_init_error is not None:
            raise _yolo_init_error

        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            if not YOLO_ONNX_PATH.exists():
                from ultralytics import YOLO  # heavy import; setup-time only
                from contextlib import redirect_stdout
                import sys as _sys

                # See racket_detector._ensure_model for the redirect rationale.
                with redirect_stdout(_sys.stderr):
                    model = YOLO(str(YOLO_PT_PATH) if YOLO_PT_PATH.exists() else "yolo11n.pt")
                    exported = model.export(format="onnx", imgsz=YOLO_INPUT_SIZE, opset=12)
                exported_path = Path(exported)
                if exported_path != YOLO_ONNX_PATH:
                    exported_path.replace(YOLO_ONNX_PATH)

            import onnxruntime as ort

            # YOLO11n + RTMPose-m are small enough that intra-op
            # parallelism inside a single inference (matmul kernels
            # spawning N threads) actually hurts: the synchronization
            # overhead dominates the speedup. We instead get parallelism
            # from the outer ThreadPoolExecutor in main._extract_with_rtmpose,
            # which submits N concurrent inference calls. Single-thread
            # each call so concurrent calls don't thrash on shared
            # ORT thread pools.
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess = ort.InferenceSession(
                str(YOLO_ONNX_PATH),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            _yolo_input_name = sess.get_inputs()[0].name
            _yolo_session = sess
        except Exception as e:
            _yolo_init_error = e
            raise


def _ensure_rtmpose() -> None:
    """Lazy-construct the rtmlib RTMPose session.

    rtmlib auto-downloads the ONNX bundle from the openmmlab CDN on
    first use and caches it under ~/.cache/rtmlib (per the lib's
    convention). We don't ship the weights in the repo.

    rtmlib prints `load <path> with onnxruntime backend` to stdout on
    successful init. We redirect that to stderr so callers that consume
    stdout (e.g. extract_clip_keypoints.py emits JSON on stdout) don't
    get a malformed payload prefix.
    """
    global _rtm_session, _rtm_init_error

    if _rtm_session is not None:
        return
    if _rtm_init_error is not None:
        raise _rtm_init_error

    with _init_lock:
        if _rtm_session is not None:
            return
        if _rtm_init_error is not None:
            raise _rtm_init_error

        from contextlib import redirect_stdout
        import sys

        try:
            from rtmlib import RTMPose  # type: ignore

            # device='cpu' is correct for Railway (no GPU). backend='onnxruntime'
            # because we already pin onnxruntime>=1.18.
            with redirect_stdout(sys.stderr):
                _rtm_session = RTMPose(
                    onnx_model=RTMPOSE_ONNX_URL,
                    model_input_size=RTMPOSE_INPUT_SIZE,
                    backend="onnxruntime",
                    device="cpu",
                )
        except Exception as e:
            _rtm_init_error = e
            raise


# ---------------------------------------------------------------------------
# YOLO person detection (numpy / onnxruntime, mirrors racket_detector style)
# ---------------------------------------------------------------------------


def _letterbox_for_yolo(img: np.ndarray, new_shape: int = YOLO_INPUT_SIZE):
    """Resize+pad an image to a square while preserving aspect ratio.

    Returns (padded_img, scale, (pad_x, pad_y)) so callers can invert
    the transform when mapping detections back.
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2

    import cv2  # local import to keep test imports cheap

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return padded, r, (pad_w, pad_h)


def _decode_yolo_output(
    output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    img_w: int,
    img_h: int,
    class_id: int,
    conf_threshold: float,
) -> list[tuple[float, float, float, float, float]]:
    """Decode a single YOLO ONNX output tensor into image-space xyxy bboxes.

    YOLO11n exports ONNX with output shape (1, 4+nc, N) where N=8400.
    Channel layout: [cx, cy, w, h, c0..c79]. We project class scores,
    select detections matching `class_id` above `conf_threshold`, and
    map xywh-normalized-to-letterbox back to original image pixels.

    Returns a list of (x1, y1, x2, y2, conf) tuples.
    """
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    # output now shape (4 + num_classes, num_anchors)
    xywh = output[:4]
    class_scores = output[4:]
    cls_ids = np.argmax(class_scores, axis=0)
    cls_conf = np.max(class_scores, axis=0)

    mask = (cls_ids == class_id) & (cls_conf >= conf_threshold)
    if not np.any(mask):
        return []

    cx = xywh[0, mask]
    cy = xywh[1, mask]
    bw = xywh[2, mask]
    bh = xywh[3, mask]
    confs = cls_conf[mask]

    # Letterbox space -> original image pixels
    pad_x, pad_y = pad
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale

    # Clip to image bounds so the rtmlib affine crop doesn't sample
    # off-image pixels (which would silently zero-pad the player).
    x1 = np.clip(x1, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y2 = np.clip(y2, 0, img_h - 1)

    return [
        (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(confs[i]))
        for i in range(len(confs))
    ]


def _yolo_person_candidates(
    frame_bgr: np.ndarray,
) -> tuple[list[tuple[float, float, float, float, float]], int, int]:
    """Run YOLO11n on a BGR frame and return ALL person candidates above
    the confidence threshold, plus the original (w, h). Shared between
    the stateless `detect_person_bbox` and the tracked variant.
    """
    _ensure_yolo()
    assert _yolo_session is not None and _yolo_input_name is not None

    h, w = frame_bgr.shape[:2]

    import cv2

    # YOLO exports expect RGB, normalized to [0,1], NCHW
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded, scale, pad = _letterbox_for_yolo(rgb)
    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]  # NCHW

    out = _yolo_session.run(None, {_yolo_input_name: blob})[0]
    candidates = _decode_yolo_output(
        out, scale, pad, w, h, PERSON_CLASS_ID, PERSON_CONFIDENCE_THRESHOLD,
    )
    return candidates, w, h


def detect_person_bbox(frame_bgr: np.ndarray) -> Optional[tuple[float, float, float, float, float]]:
    """Run YOLO11n on a BGR frame and return the highest-confidence person bbox.

    Stateless version kept for back-compat with existing callers/tests.
    Production extraction uses `detect_person_bbox_tracked` below, which
    replaces raw max-confidence selection with area/centrality scoring
    plus IoU-based association across frames (fixes the ghost-skeleton
    bug where a small high-conf background figure beat the real player).
    """
    candidates, _w, _h = _yolo_person_candidates(frame_bgr)
    if not candidates:
        return None
    # Highest-confidence detection wins.
    return max(candidates, key=lambda c: c[4])


def detect_person_bbox_tracked(
    tracker,  # tracking.PersonTracker
    frame_bgr: np.ndarray,
) -> Optional[tuple[float, float, float, float, float]]:
    """Run YOLO and fold all person candidates through `tracker`.

    The tracker picks the right person on cold start (largest × centered ×
    person-shaped, not max-conf) and sticks to it via IoU association on
    subsequent frames. Returns the tracker's canonical bbox for this
    frame, which may be a smoothed blend of the new YOLO detection and
    the previous reference, or a coasted previous reference when YOLO
    missed this frame.
    """
    candidates, w, h = _yolo_person_candidates(frame_bgr)
    return tracker.update(candidates, w, h)


def expand_bbox(
    bbox: tuple[float, float, float, float, float],
    img_w: int,
    img_h: int,
    pct: float = BBOX_EXPAND_PCT,
) -> tuple[float, float, float, float]:
    """Expand a YOLO bbox outward by `pct` on each side, clipped to image.

    Returns just (x1, y1, x2, y2) -- confidence is dropped because the
    expanded box is a derived quantity, not a YOLO detection.
    """
    x1, y1, x2, y2, _ = bbox
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * pct
    dy = bh * pct
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(img_w - 1), x2 + dx),
        min(float(img_h - 1), y2 + dy),
    )


# ---------------------------------------------------------------------------
# COCO-17 -> BlazePose-33 mapping
# ---------------------------------------------------------------------------


def coco17_to_blazepose33(
    coco_kpts: np.ndarray,
    coco_scores: np.ndarray,
    img_w: int,
    img_h: int,
) -> list[dict]:
    """Build a 33-entry landmark list from a single person's COCO-17 output.

    Args:
      coco_kpts: shape (17, 2) -- (x, y) in image pixel coords.
      coco_scores: shape (17,) -- per-keypoint confidence in [0, 1].
      img_w / img_h: original frame dimensions for normalization.

    Returns: a list of 33 landmark dicts in the schema downstream code
    consumes. Unfilled BlazePose ids carry visibility=0.
    """
    if coco_kpts.shape != (17, 2):
        raise ValueError(f"expected coco_kpts shape (17, 2), got {coco_kpts.shape}")
    if coco_scores.shape != (17,):
        raise ValueError(f"expected coco_scores shape (17,), got {coco_scores.shape}")
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"image dims must be positive, got {img_w}x{img_h}")

    # Initialize all 33 landmarks at (0, 0, vis=0). Names match the
    # original mediapipe schema for tooling that introspects them.
    landmarks: list[dict] = [
        {
            "id": i,
            "name": BLAZEPOSE_LANDMARK_NAMES[i],
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "visibility": 0.0,
        }
        for i in range(NUM_BLAZEPOSE_LANDMARKS)
    ]

    for coco_id, blaze_id in COCO17_TO_BLAZEPOSE33.items():
        x_px, y_px = coco_kpts[coco_id]
        score = float(coco_scores[coco_id])
        # Detect "off-frame" BEFORE clipping. RTMPose runs on a YOLO crop
        # and re-projects to full-frame pixel coords. When the player walks
        # partially off screen the network still emits a confident
        # prediction, which post-clip lands exactly at the frame edge with
        # high visibility, so the renderer keeps drawing a phantom limb
        # at the boundary. Mirror what the browser MediaPipe path does:
        # if the raw landmark falls outside [0,1] of the frame, force
        # visibility to 0 so the render-side cutoff drops it.
        x_raw = x_px / img_w if img_w > 0 else 0.5
        y_raw = y_px / img_h if img_h > 0 else 0.5
        in_frame = 0.0 <= x_raw <= 1.0 and 0.0 <= y_raw <= 1.0
        x_norm = float(np.clip(x_raw, 0.0, 1.0))
        y_norm = float(np.clip(y_raw, 0.0, 1.0))
        landmarks[blaze_id] = {
            "id": blaze_id,
            "name": BLAZEPOSE_LANDMARK_NAMES[blaze_id],
            "x": round(x_norm, 4),
            "y": round(y_norm, 4),
            "z": 0.0,  # RTMPose-2D doesn't predict z; downstream tolerates 0.
            "visibility": round(score, 3) if in_frame else 0.0,
        }

    return landmarks


# ---------------------------------------------------------------------------
# Top-level per-frame inference
# ---------------------------------------------------------------------------


def infer_pose_for_frame(
    frame_bgr: np.ndarray,
    person_tracker=None,  # Optional[tracking.PersonTracker]
) -> Optional[list[dict]]:
    """Detect a person and run RTMPose-m on a single BGR frame.

    When `person_tracker` is passed, it is fed every frame's YOLO
    candidates and returns a locked, smoothed bbox (see
    `detect_person_bbox_tracked`). Without a tracker, falls back to the
    stateless max-confidence pick — kept for unit tests and one-off
    callers. Extraction loops always pass a tracker so the ghost-person
    selection bug can't recur.

    Returns a 33-entry BlazePose-shaped landmark list, or None when no
    person is detected (caller should skip the frame, matching the
    existing main.py path that drops frames without a detection).
    """
    if person_tracker is not None:
        bbox = detect_person_bbox_tracked(person_tracker, frame_bgr)
    else:
        bbox = detect_person_bbox(frame_bgr)
    if bbox is None:
        return None

    h, w = frame_bgr.shape[:2]
    expanded = expand_bbox(bbox, w, h)

    _ensure_rtmpose()
    assert _rtm_session is not None

    # rtmlib expects bbox in xyxy format as a list. Pass exactly one
    # bbox so the output is shape (1, 17, 2).
    keypoints, scores = _rtm_session(frame_bgr, bboxes=[list(expanded)])

    if keypoints is None or len(keypoints) == 0:
        return None

    return coco17_to_blazepose33(keypoints[0], scores[0], w, h)


# ---------------------------------------------------------------------------
# Test seam: clearing the lazily-cached sessions so unit tests can
# inject mocks via direct module attribute writes.
# ---------------------------------------------------------------------------


def _reset_for_tests() -> None:
    """Reset module-level caches. Tests only -- never call from prod paths."""
    global _rtm_session, _yolo_session, _yolo_input_name
    global _rtm_init_error, _yolo_init_error
    _rtm_session = None
    _yolo_session = None
    _yolo_input_name = None
    _rtm_init_error = None
    _yolo_init_error = None


# Allow tests / scripts to override the model directory before lazy init runs.
def _set_model_dir(path: Path) -> None:
    global MODEL_DIR, YOLO_PT_PATH, YOLO_ONNX_PATH
    MODEL_DIR = Path(path)
    YOLO_PT_PATH = MODEL_DIR / "yolo11n.pt"
    YOLO_ONNX_PATH = MODEL_DIR / "yolo11n.onnx"


__all__ = [
    "BLAZEPOSE_LANDMARK_NAMES",
    "BBOX_EXPAND_PCT",
    "COCO17_TO_BLAZEPOSE33",
    "NUM_BLAZEPOSE_LANDMARKS",
    "PERSON_CLASS_ID",
    "RTMPOSE_INPUT_SIZE",
    "_decode_yolo_output",
    "_ensure_rtmpose",
    "_ensure_yolo",
    "_letterbox_for_yolo",
    "_reset_for_tests",
    "_set_model_dir",
    "_yolo_person_candidates",
    "coco17_to_blazepose33",
    "detect_person_bbox",
    "detect_person_bbox_tracked",
    "expand_bbox",
    "infer_pose_for_frame",
]
