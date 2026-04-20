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


def compute_joint_angles(landmarks):
    def get(idx):
        if idx < len(landmarks):
            lm = landmarks[idx]
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


def extract_keypoints(video_path, sample_fps=30):
    if POSE_BACKEND == "rtmpose":
        raise NotImplementedError("rtmpose backend not yet implemented")
    if POSE_BACKEND != "mediapipe":
        raise ValueError(f"Unknown POSE_BACKEND: {POSE_BACKEND}")

    # Import lazily so a missing ultralytics install fails softly at detect time
    # rather than preventing the entire extractor from loading.
    try:
        from racket_detector import detect_racket  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[extract] racket_detector unavailable: {e}")
        detect_racket = None  # type: ignore

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

                    racket_head = None
                    if detect_racket is not None:
                        h, w = frame.shape[:2]
                        lw = landmarks[15] if len(landmarks) > 15 else None
                        rw = landmarks[16] if len(landmarks) > 16 else None
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
                            racket_head = detect_racket(frame, wrist_xy)
                        except Exception as e:  # noqa: BLE001
                            print(f"[extract] racket detect failed on frame {processed_index}: {e}")
                            racket_head = None

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

    return {
        "fps_sampled": sample_fps,
        "frame_count": len(frames),
        "frames": frames,
        "video_fps": video_fps,
        "duration_ms": round(total_duration_ms),
        "schema_version": 2,
    }


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
