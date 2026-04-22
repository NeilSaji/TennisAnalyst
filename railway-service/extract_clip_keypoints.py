"""Extract keypoints from a video clip and print JSON to stdout.

Standalone script that does NOT import main.py (avoids supabase dependency).
Duplicates the extraction logic from main.py to stay self-contained.

Usage: python3 extract_clip_keypoints.py <video_path>
"""
import math
import sys
import os
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# POSE_BACKEND selects the pose estimator. `mediapipe` is the only supported
# value today; `rtmpose` is a deliberate seam for a future swap.
POSE_BACKEND = os.environ.get("POSE_BACKEND", "mediapipe").lower()

LANDMARK_NAMES = [
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

MODEL_PATH = str(Path(__file__).parent / "models" / "pose_landmarker_heavy.task")


def angle_between(a, b, c):
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_joint_angles(landmarks, min_visibility=0.0):
    """Joint-angle helper. `min_visibility` gates each landmark before it's
    used; default 0 preserves v2 behavior. v3 (RTMPose) callers pass 0.1 so
    the placeholder landmarks (visibility=0 for ids COCO-17 doesn't fill)
    don't contaminate the angle output."""
    def get(idx):
        if idx < len(landmarks):
            lm = landmarks[idx]
            if lm.get("visibility", 1.0) < min_visibility:
                return None
            return (lm["x"], lm["y"])
        return None

    angles = {}
    pts = {
        "ls": get(11), "rs": get(12),
        "le": get(13), "re": get(14),
        "lw": get(15), "rw": get(16),
        "li": get(19), "ri": get(20),
        "lh": get(23), "rh": get(24),
        "lk": get(25), "rk": get(26),
        "la": get(27), "ra": get(28),
    }

    if all(pts[k] for k in ["rs", "re", "rw"]):
        angles["right_elbow"] = round(angle_between(pts["rs"], pts["re"], pts["rw"]), 1)
    if all(pts[k] for k in ["ls", "le", "lw"]):
        angles["left_elbow"] = round(angle_between(pts["ls"], pts["le"], pts["lw"]), 1)
    if all(pts[k] for k in ["ls", "rs", "re"]):
        angles["right_shoulder"] = round(angle_between(pts["ls"], pts["rs"], pts["re"]), 1)
    if all(pts[k] for k in ["rs", "ls", "le"]):
        angles["left_shoulder"] = round(angle_between(pts["rs"], pts["ls"], pts["le"]), 1)
    # Wrist flexion: elbow -> wrist -> index finger
    if all(pts[k] for k in ["re", "rw", "ri"]):
        angles["right_wrist"] = round(angle_between(pts["re"], pts["rw"], pts["ri"]), 1)
    if all(pts[k] for k in ["le", "lw", "li"]):
        angles["left_wrist"] = round(angle_between(pts["le"], pts["lw"], pts["li"]), 1)
    if all(pts[k] for k in ["rh", "rk", "ra"]):
        angles["right_knee"] = round(angle_between(pts["rh"], pts["rk"], pts["ra"]), 1)
    if all(pts[k] for k in ["lh", "lk", "la"]):
        angles["left_knee"] = round(angle_between(pts["lh"], pts["lk"], pts["la"]), 1)
    if pts["lh"] and pts["rh"]:
        hip_vec = (pts["rh"][0] - pts["lh"][0], pts["rh"][1] - pts["lh"][1])
        angles["hip_rotation"] = round(abs(math.degrees(math.atan2(hip_vec[1], hip_vec[0]))), 1)
    if pts["ls"] and pts["rs"]:
        sh_vec = (pts["rs"][0] - pts["ls"][0], pts["rs"][1] - pts["ls"][1])
        angles["trunk_rotation"] = round(abs(math.degrees(math.atan2(sh_vec[1], sh_vec[0]))), 1)

    return angles


def _try_load_racket_detector():
    try:
        from racket_detector import detect_racket  # type: ignore

        return detect_racket
    except Exception as e:  # noqa: BLE001
        print(f"[extract] racket_detector unavailable: {e}", file=sys.stderr)
        return None


# How far past the wrist (along the forearm direction) to place the
# fallback racket point. 0 = pin to the wrist (grip position). Was 0.4
# (approximating the middle of the racket) but user reported that as
# "predictive / joints moving ahead of the person" — an extension does
# literally lead the wrist spatially during a fast swing, so dropping
# to 0 removes the predictive component. When real YOLO runs server-
# side, its bbox center still takes precedence over this fallback.
RACKET_FALLBACK_EXTENSION = 0.0

# Confidence we stamp on fallback racket detections. Has to clear the
# frontend tracer's 0.3 gate but stay below real YOLO detections so a
# downstream "real vs estimated" filter can distinguish them later.
RACKET_FALLBACK_CONFIDENCE = 0.35

# BlazePose-33 indices (same on the MediaPipe path and after the
# pose_rtmpose COCO-17 -> BlazePose-33 remap).
_LM_LEFT_ELBOW = 13
_LM_RIGHT_ELBOW = 14
_LM_LEFT_WRIST = 15
_LM_RIGHT_WRIST = 16


def _racket_fallback_from_forearm(landmarks):
    """Estimate a racket-center point from the dominant wrist extended
    along the forearm direction. Runs when YOLO is unavailable (weights
    missing on deploy, ONNX session failed to load, inference errored)
    or when YOLO returned no racket bbox on this frame.

    Returns a dict with the same shape YOLO would emit, or None when
    the pose itself is too weak to derive a position.
    """
    if len(landmarks) <= max(_LM_LEFT_WRIST, _LM_RIGHT_WRIST, _LM_LEFT_ELBOW, _LM_RIGHT_ELBOW):
        return None

    lw = landmarks[_LM_LEFT_WRIST]
    rw = landmarks[_LM_RIGHT_WRIST]
    le = landmarks[_LM_LEFT_ELBOW]
    re = landmarks[_LM_RIGHT_ELBOW]

    # Pick the higher-visibility wrist as the racket-holding side.
    lw_vis = lw.get("visibility", 0) if lw else 0
    rw_vis = rw.get("visibility", 0) if rw else 0
    if lw_vis == 0 and rw_vis == 0:
        return None

    if rw_vis >= lw_vis:
        wrist, elbow = rw, re
    else:
        wrist, elbow = lw, le

    # Need at least a reasonably-confident wrist to anchor on.
    if wrist is None or wrist.get("visibility", 0) < 0.5:
        return None

    wx, wy = wrist["x"], wrist["y"]
    if elbow and elbow.get("visibility", 0) >= 0.5:
        # Extend past the wrist along the forearm direction.
        rx = wx + (wx - elbow["x"]) * RACKET_FALLBACK_EXTENSION
        ry = wy + (wy - elbow["y"]) * RACKET_FALLBACK_EXTENSION
    else:
        # Elbow too weak: degrade to raw wrist position. Better than nothing.
        rx, ry = wx, wy

    return {
        "x": round(max(0.0, min(1.0, rx)), 4),
        "y": round(max(0.0, min(1.0, ry)), 4),
        "confidence": RACKET_FALLBACK_CONFIDENCE,
    }


def _detect_racket_for_frame(detect_racket, frame_bgr, landmarks, processed_index):
    # Try YOLO first when it's available. On Railway the weights ship
    # committed (railway-service/models/yolo11n.onnx) so this normally
    # works; if it doesn't, fall through to the forearm-extension
    # fallback below so the racket trail still renders.
    if detect_racket is not None:
        h, w = frame_bgr.shape[:2]
        lw = landmarks[_LM_LEFT_WRIST] if len(landmarks) > _LM_LEFT_WRIST else None
        rw = landmarks[_LM_RIGHT_WRIST] if len(landmarks) > _LM_RIGHT_WRIST else None
        dominant = None
        if lw and rw:
            dominant = rw if rw["visibility"] >= lw["visibility"] else lw
        elif lw:
            dominant = lw
        elif rw:
            dominant = rw
        wrist_xy = (
            (dominant["x"] * w, dominant["y"] * h)
            if dominant is not None
            else None
        )
        try:
            yolo = detect_racket(frame_bgr, wrist_xy)
            if yolo is not None:
                return yolo
        except Exception as e:  # noqa: BLE001
            print(f"[extract] racket detect failed on frame {processed_index}: {e}", file=sys.stderr)

    # Forearm-extension fallback. Approximate but always available when
    # the pose detector found the player's arm.
    return _racket_fallback_from_forearm(landmarks)


def _extract_mediapipe(video_path, sample_fps):
    detect_racket = _try_load_racket_detector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / sample_fps))
    frames = []
    frame_index = 0
    processed_index = 0

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp_ms = int((frame_index / video_fps) * 1000)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    lm_list = result.pose_landmarks[0]
                    landmarks = [
                        {
                            "id": i,
                            "name": LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"landmark_{i}",
                            "x": round(lm.x, 4),
                            "y": round(lm.y, 4),
                            "z": round(lm.z, 4),
                            "visibility": round(lm.visibility, 3),
                        }
                        for i, lm in enumerate(lm_list)
                    ]
                    joint_angles = compute_joint_angles(landmarks)
                    racket_head = _detect_racket_for_frame(
                        detect_racket, frame, landmarks, processed_index
                    )
                    frames.append({
                        "frame_index": processed_index,
                        "timestamp_ms": round(timestamp_ms, 1),
                        "landmarks": landmarks,
                        "joint_angles": joint_angles,
                        "racket_head": racket_head,
                    })
                    processed_index += 1

            frame_index += 1

    cap.release()

    total_frames = frame_index
    total_duration_ms = (total_frames / video_fps) * 1000

    # Zero-detection alarm: surfaces the "YOLO model failed to load and we're
    # silently emitting all-null racket_head" case in Railway logs. detector_loaded
    # separates "detector disabled" from "detector ran and found nothing."
    racket_count = sum(1 for f in frames if f.get("racket_head") is not None)
    if len(frames) > 0 and racket_count == 0:
        print(
            f"[extract:mediapipe] racket_detector produced zero detections across"
            f" {len(frames)} frames (detector_loaded={detect_racket is not None})",
            file=sys.stderr,
        )

    return {
        "fps_sampled": sample_fps,
        "frame_count": len(frames),
        "frames": frames,
        "video_fps": video_fps,
        "duration_ms": round(total_duration_ms),
        "schema_version": 2,
    }


def _extract_rtmpose(video_path, sample_fps):
    from pose_rtmpose import infer_pose_for_frame
    from tracking import PersonTracker

    detect_racket = _try_load_racket_detector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / sample_fps))
    frames = []
    frame_index = 0
    processed_index = 0

    # Single PersonTracker instance threaded through the whole clip.
    # Fixes the "ghost-skeleton on an empty part of the court" bug where
    # YOLO picked a small high-conf background figure over the larger
    # foreground player. Tracker cold-starts on the first frame with
    # area×centrality×aspect scoring and locks on via IoU from there.
    person_tracker = PersonTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp_ms = int((frame_index / video_fps) * 1000)
            try:
                landmarks = infer_pose_for_frame(frame, person_tracker=person_tracker)
            except Exception as e:  # noqa: BLE001
                print(f"[extract:rtmpose] inference failed on frame {frame_index}: {e}", file=sys.stderr)
                landmarks = None

            if landmarks is not None:
                # min_visibility=0.1: see the matching note in main.py
                joint_angles = compute_joint_angles(landmarks, min_visibility=0.1)
                racket_head = _detect_racket_for_frame(
                    detect_racket, frame, landmarks, processed_index
                )
                frames.append({
                    "frame_index": processed_index,
                    "timestamp_ms": round(timestamp_ms, 1),
                    "landmarks": landmarks,
                    "joint_angles": joint_angles,
                    "racket_head": racket_head,
                })
                processed_index += 1

        frame_index += 1

    cap.release()

    total_frames = frame_index
    total_duration_ms = (total_frames / video_fps) * 1000

    # Zero-detection alarm: see the matching block in _extract_mediapipe above.
    racket_count = sum(1 for f in frames if f.get("racket_head") is not None)
    if len(frames) > 0 and racket_count == 0:
        print(
            f"[extract:rtmpose] racket_detector produced zero detections across"
            f" {len(frames)} frames (detector_loaded={detect_racket is not None})",
            file=sys.stderr,
        )

    return {
        "fps_sampled": sample_fps,
        "frame_count": len(frames),
        "frames": frames,
        "video_fps": video_fps,
        "duration_ms": round(total_duration_ms),
        "schema_version": 3,
    }


def extract_keypoints(video_path, sample_fps=30):
    if POSE_BACKEND == "rtmpose":
        return _extract_rtmpose(video_path, sample_fps)
    if POSE_BACKEND == "mediapipe":
        return _extract_mediapipe(video_path, sample_fps)
    raise ValueError(f"Unknown POSE_BACKEND: {POSE_BACKEND}")


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: extract_clip_keypoints.py <video_path>"}))
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(json.dumps({"error": f"File not found: {video_path}"}))
        sys.exit(1)

    result = extract_keypoints(video_path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
