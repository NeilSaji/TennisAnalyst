"""
TennisIQ Railway Microservice
FastAPI service for server-side pose extraction from videos.
Used for processing pro player videos offline and seeding the database.
"""

import asyncio
import os
import math
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel
from supabase import create_client, Client
import httpx

# POSE_BACKEND selects the pose estimator. Two valid values:
#   "mediapipe" — BlazePose Heavy (default; older path)
#   "rtmpose"   — YOLO11n + RTMPose-m (better tracing on small subjects)
# Strip whitespace and surrounding quotes defensively. Railway dashboards
# sometimes preserve copy-pasted quotes, and trailing newlines from shell
# `echo` exports are easy to miss — without these, the dispatch raises
# "Unknown POSE_BACKEND: rtmpose" even when the value LOOKS right.
POSE_BACKEND = (
    os.environ.get("POSE_BACKEND", "mediapipe").strip().strip("\"'").lower()
)

app = FastAPI(title="TennisIQ Pose Service")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EXTRACT_API_KEY = os.environ.get("EXTRACT_API_KEY", "")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Only allow fetching videos from these domains
ALLOWED_VIDEO_DOMAINS = {
    "blob.vercel-storage.com",
    "public.blob.vercel-storage.com",
    "supabase.co",
    "supabase.com",
    "storage.googleapis.com",
}


def _is_url_allowed(url: str) -> bool:
    """Validate that a video URL uses https and points to an allowed domain."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = parsed.hostname or ""
    return any(host == d or host.endswith(f".{d}") for d in ALLOWED_VIDEO_DOMAINS)


# MediaPipe landmark names (indices 0-32)
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

# Path to the downloaded pose landmarker model
MODEL_PATH = str(Path(__file__).parent / "models" / "pose_landmarker_heavy.task")


def angle_between(a, b, c):
    """Compute angle at joint b given three 2D points a, b, c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_joint_angles_from_dicts(
    landmarks: list[dict], min_visibility: float = 0.0
) -> dict:
    """Compute key joint angles from a list of landmark dicts with x, y keys.

    `min_visibility` gates each landmark before it's used. Default 0
    preserves the original v2 (MediaPipe) behavior where every landmark
    contributes regardless of confidence. The v3 (RTMPose) path passes a
    low positive value so the visibility=0 placeholder landmarks (face
    detail, hands, heels, foot_index) don't produce meaningless angles
    like `right_wrist: 0.0` derived from (0,0)-(0,0)-(0,0) triplets.
    """
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
        angles["hip_rotation"] = round(
            abs(math.degrees(math.atan2(hip_vec[1], hip_vec[0]))), 1
        )
    if pts["ls"] and pts["rs"]:
        sh_vec = (pts["rs"][0] - pts["ls"][0], pts["rs"][1] - pts["ls"][1])
        angles["trunk_rotation"] = round(
            abs(math.degrees(math.atan2(sh_vec[1], sh_vec[0]))), 1
        )

    return angles


def _open_video_autorotated(video_path: str) -> cv2.VideoCapture:
    """cv2.VideoCapture with CAP_PROP_ORIENTATION_AUTO on. Phone portrait
    clips carry a 90° rotation tag that OpenCV won't apply by default —
    YOLO and RTMPose both collapse to zero detections on sideways frames.
    See extract_clip_keypoints._open_video_autorotated for the full
    rationale and the measured impact on IMG_1097.mov.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:  # noqa: BLE001
        pass
    return cap


def _try_load_racket_detector():
    """Best-effort import of the racket detector.

    Returns either the callable or None when ultralytics/onnxruntime
    isn't fully wired (e.g. unit-test environments without GPU/CPU
    weights). Logged but never raised so extraction still runs.
    """
    try:
        from racket_detector import detect_racket  # type: ignore

        return detect_racket
    except Exception as e:  # noqa: BLE001
        print(f"[extract] racket_detector unavailable: {e}")
        return None


def _extract_with_mediapipe(
    video_path: str, sample_fps: int, max_seconds: float
) -> dict:
    """Pose extraction via MediaPipe Heavy. Original implementation, unchanged
    in behavior -- factored into its own function so the rtmpose path can sit
    next to it without a giant if/else nested branch."""
    from tracking import RacketTracker

    detect_racket = _try_load_racket_detector()

    cap = _open_video_autorotated(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / sample_fps))
    max_frame = int(max_seconds * video_fps) if max_seconds > 0 else 0
    frames = []
    frame_index = 0
    processed_index = 0
    racket_tracker = RacketTracker()

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
            if max_frame > 0 and frame_index >= max_frame:
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
                    joint_angles = compute_joint_angles_from_dicts(landmarks)

                    racket_head = _detect_racket_for_frame(
                        detect_racket, frame, landmarks, processed_index,
                        timestamp_ms=timestamp_ms,
                        racket_tracker=racket_tracker,
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

    return {
        "fps_sampled": sample_fps,
        "frame_count": len(frames),
        "frames": frames,
        "video_fps": video_fps,
        "duration_ms": round(total_duration_ms),
        "schema_version": 2,
        # Identifies which backend produced these keypoints. The browser
        # surfaces this as a diagnostic chip so we can tell at a glance
        # whether a "tracing is off" complaint is coming from server
        # mediapipe vs server rtmpose vs browser fallback.
        "pose_backend": "mediapipe",
    }


def _detect_racket_for_frame(
    detect_racket,
    frame_bgr,
    landmarks: list[dict],
    processed_index: int,
    timestamp_ms: float | None = None,
    racket_tracker=None,
):
    """Shared racket-head detection. Picks the higher-visibility wrist as the
    dominant hand and feeds the racket detector. Routes the result through a
    Kalman `racket_tracker` (when provided) to smooth YOLO jitter and coast
    short motion-blur gaps. Returns None on any failure (logged but never
    raised)."""
    yolo_point = None
    if detect_racket is not None:
        h, w = frame_bgr.shape[:2]
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
        # Crop to landmark envelope + reach margin. See
        # extract_clip_keypoints._person_crop_bbox for rationale.
        from extract_clip_keypoints import _person_crop_bbox
        person_bbox = _person_crop_bbox(landmarks, w, h)
        try:
            yolo_point = detect_racket(
                frame_bgr, wrist_xy, person_bbox=person_bbox,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[extract] racket detect failed on frame {processed_index}: {e}")
            yolo_point = None

    if racket_tracker is not None and timestamp_ms is not None:
        return racket_tracker.update(yolo_point, timestamp_ms=timestamp_ms)
    return yolo_point


def _extract_with_rtmpose(
    video_path: str, sample_fps: int, max_seconds: float
) -> dict:
    """Pose extraction via YOLO11n person crop + RTMPose-m.

    Schema_version=3: COCO-17 keypoints remapped to BlazePose-33 ids, with
    the unmapped ids (face details, hands, heels, foot_index) at
    visibility=0. wrist-flexion joint angles are dropped because the
    index-finger landmarks (BlazePose 19/20) aren't filled.
    """
    from pose_rtmpose import infer_pose_for_frame  # local import: heavy onnx setup
    from tracking import PersonTracker, RacketTracker

    detect_racket = _try_load_racket_detector()

    cap = _open_video_autorotated(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / sample_fps))
    max_frame = int(max_seconds * video_fps) if max_seconds > 0 else 0
    frames: list[dict] = []
    frame_index = 0
    processed_index = 0

    # See _extract_rtmpose in extract_clip_keypoints.py for rationale.
    person_tracker = PersonTracker()
    racket_tracker = RacketTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frame > 0 and frame_index >= max_frame:
            break

        if frame_index % frame_interval == 0:
            timestamp_ms = int((frame_index / video_fps) * 1000)
            try:
                landmarks = infer_pose_for_frame(frame, person_tracker=person_tracker)
            except Exception as e:  # noqa: BLE001
                # Skip the frame on detector failure rather than aborting the
                # whole clip -- one bad frame shouldn't lose the whole swing.
                print(f"[extract:rtmpose] inference failed on frame {frame_index}: {e}")
                landmarks = None

            if landmarks is not None:
                # min_visibility=0.1 skips the v=0 placeholder ids that
                # exist in v3 only because COCO-17 doesn't fill them.
                # Without this, joint_angles ends up with `right_wrist:
                # 0.0` (computed off zeroed index_finger landmarks),
                # which the analyzer would surface as a real value.
                joint_angles = compute_joint_angles_from_dicts(
                    landmarks, min_visibility=0.1
                )
                racket_head = _detect_racket_for_frame(
                    detect_racket, frame, landmarks, processed_index,
                    timestamp_ms=timestamp_ms,
                    racket_tracker=racket_tracker,
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

    return {
        "fps_sampled": sample_fps,
        "frame_count": len(frames),
        "frames": frames,
        "video_fps": video_fps,
        "duration_ms": round(total_duration_ms),
        # v3: COCO-17 backbone via RTMPose. Wrist-flexion absent.
        "schema_version": 3,
        # Identifies which backend produced these keypoints (see the
        # mediapipe path above for rationale).
        "pose_backend": "rtmpose",
    }


def extract_keypoints_from_video(video_path: str, sample_fps: int = 30, max_seconds: float = 0) -> dict:
    """Extract pose keypoints from a video file.

    Backend selection via POSE_BACKEND env var:
      * "mediapipe" (default) -- BlazePose Heavy via MediaPipe Tasks. Schema v2.
      * "rtmpose" -- YOLO11n person crop + RTMPose-m. Schema v3 (no wrist
        flexion).

    Args:
        max_seconds: Stop after this many seconds. 0 = process entire video.
    """
    if POSE_BACKEND == "rtmpose":
        return _extract_with_rtmpose(video_path, sample_fps, max_seconds)
    if POSE_BACKEND == "mediapipe":
        return _extract_with_mediapipe(video_path, sample_fps, max_seconds)
    # repr() so trailing whitespace, quotes, or unicode surprises in the
    # parsed value are visible in the error message.
    raise ValueError(f"Unknown POSE_BACKEND: {POSE_BACKEND!r}")


class ExtractRequest(BaseModel):
    video_url: str
    session_id: str | None = None
    pro_swing_id: str | None = None
    callback_url: str | None = None


@app.post("/extract")
async def extract_pose(
    req: ExtractRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(default=""),
):
    """
    Trigger pose extraction for a video URL.
    Runs in background and updates Supabase when done.
    """
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _is_url_allowed(req.video_url):
        raise HTTPException(
            status_code=400,
            detail="video_url must use https and point to an allowed storage domain",
        )

    if req.callback_url and not _is_url_allowed(req.callback_url):
        raise HTTPException(
            status_code=400,
            detail="callback_url must use https and point to an allowed domain",
        )

    background_tasks.add_task(_run_extraction, req)
    return {"status": "queued", "session_id": req.session_id}


async def _run_extraction(req: ExtractRequest):
    try:
        # Download video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.get(req.video_url)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(response.content)

        # Run blocking CPU/IO work in a thread pool so the event loop stays free
        try:
            keypoints_json = await asyncio.to_thread(extract_keypoints_from_video, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Update the appropriate DB record
        if req.pro_swing_id:
            supabase.table("pro_swings").update({
                "keypoints_json": keypoints_json,
                "frame_count": keypoints_json["frame_count"],
                "duration_ms": keypoints_json["duration_ms"],
                "fps": keypoints_json["fps_sampled"],
            }).eq("id", req.pro_swing_id).execute()

        elif req.session_id:
            supabase.table("user_sessions").update({
                "keypoints_json": keypoints_json,
                "status": "complete",
            }).eq("id", req.session_id).execute()

        # Optional callback — only to allowed domains
        if req.callback_url and _is_url_allowed(req.callback_url):
            async with httpx.AsyncClient() as client:
                await client.post(req.callback_url, json={"status": "complete"})

    except Exception as e:
        print(f"Extraction failed: {e}")
        if req.session_id:
            supabase.table("user_sessions").update({
                "status": "error",
                "error_message": str(e),
            }).eq("id", req.session_id).execute()


# ---------------------------------------------------------------------------
# YouTube processing
# ---------------------------------------------------------------------------

# In-memory job status tracker (sufficient for single-instance Railway deploy)
_youtube_jobs: dict[str, dict] = {}

YOUTUBE_URL_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com|youtu\.be)/"
)


def _is_youtube_url(url: str) -> bool:
    """Return True if *url* looks like a valid YouTube URL."""
    return bool(YOUTUBE_URL_PATTERN.match(url))


class YouTubeProcessRequest(BaseModel):
    youtube_url: str
    target_shot_types: list[str] | None = None
    max_duration: int = 600


@app.post("/process-youtube")
async def process_youtube(
    req: YouTubeProcessRequest,
    background_tasks: BackgroundTasks,
    authorization: str = Header(default=""),
):
    """Queue processing of a YouTube video for tennis shot extraction."""
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _is_youtube_url(req.youtube_url):
        raise HTTPException(
            status_code=400,
            detail="youtube_url must be a valid youtube.com or youtu.be URL",
        )

    job_id = str(uuid.uuid4())
    _youtube_jobs[job_id] = {"status": "processing", "clips": [], "error": None}

    background_tasks.add_task(_run_youtube_processing, job_id, req)
    return {"status": "queued", "job_id": job_id}


@app.get("/process-youtube/{job_id}")
async def get_youtube_job_status(
    job_id: str,
    authorization: str = Header(default=""),
):
    """Check the status of a YouTube processing job."""
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    job = _youtube_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response: dict = {"status": job["status"]}
    if job["status"] == "complete":
        response["clips"] = job["clips"]
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response


async def _run_youtube_processing(job_id: str, req: YouTubeProcessRequest):
    """Background task: process a YouTube video and store results."""
    from youtube_processor import process_youtube_video

    try:
        result = await asyncio.to_thread(
            process_youtube_video,
            req.youtube_url,
            req.target_shot_types,
            req.max_duration,
        )

        created_ids: list[dict] = []

        for clip in result.clips:
            # Upload each extracted clip to Vercel Blob via the public upload URL
            clip_path = clip.path
            if not os.path.exists(clip_path):
                continue

            try:
                blob_url = await _upload_clip_to_blob(clip_path, clip.shot_type)

                # Extract keypoints for the clip
                keypoints_json = await asyncio.to_thread(
                    extract_keypoints_from_video, clip_path
                )

                # pro_id is required by the schema, so DB insertion
                # is deferred until the caller associates clips with a
                # pro.  Return clip data so the caller can do that.
                created_ids.append({
                    "blob_url": blob_url,
                    "shot_type": clip.shot_type,
                    "confidence": clip.confidence,
                    "camera_angle": clip.camera_angle,
                    "duration_ms": clip.duration_ms,
                    "handedness": clip.handedness,
                    "keypoints_frame_count": keypoints_json.get("frame_count"),
                    "metadata": {
                        "source": "youtube",
                        "original_url": req.youtube_url,
                        "video_title": result.video_title,
                        "camera_angle": clip.camera_angle,
                        "confidence": clip.confidence,
                        "handedness": clip.handedness,
                        "start_time": clip.start_time,
                        "end_time": clip.end_time,
                    },
                })

            except Exception as e:
                print(f"Failed to process clip {clip_path}: {e}")
            finally:
                if os.path.exists(clip_path):
                    os.unlink(clip_path)

        _youtube_jobs[job_id] = {
            "status": "complete",
            "clips": created_ids,
            "error": None,
        }

    except Exception as e:
        print(f"YouTube processing failed for job {job_id}: {e}")
        _youtube_jobs[job_id] = {
            "status": "error",
            "clips": [],
            "error": str(e),
        }


async def _upload_clip_to_blob(clip_path: str, shot_type: str) -> str:
    """Upload a clip file to Vercel Blob and return the public URL.

    Requires BLOB_READ_WRITE_TOKEN in the environment.
    """
    return await _upload_file_to_blob(clip_path, f"pro-clips/{shot_type}/{uuid.uuid4()}.mp4")


async def _upload_file_to_blob(local_path: str, blob_path: str) -> str:
    """Upload an arbitrary local file to Vercel Blob at *blob_path* and return the public URL.

    Requires BLOB_READ_WRITE_TOKEN in the environment. *blob_path* is treated
    as the pathname inside the store and must not start with '/'.
    """
    blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not blob_token:
        raise RuntimeError("BLOB_READ_WRITE_TOKEN not set — cannot upload clips")

    filename = blob_path.lstrip("/")

    with open(local_path, "rb") as f:
        file_bytes = f.read()

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.put(
            f"https://blob.vercel-storage.com/{filename}",
            content=file_bytes,
            headers={
                "Authorization": f"Bearer {blob_token}",
                "x-content-type": "video/mp4",
                "x-cache-control-max-age": "31536000",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    url = data.get("url")
    if not url:
        raise RuntimeError("Blob upload succeeded but no URL in response")
    return url


# ---------------------------------------------------------------------------
# Camera angle classification (telemetry-only)
# ---------------------------------------------------------------------------
#
# Wraps camera_classifier.classify_frame in an HTTP endpoint so Node analyze
# routes can tag each analysis_events row with a capture_quality_flag. The
# classifier exposes five raw labels -- behind / side / front / overhead /
# unknown. There is NO oblique-angle branch in the underlying heuristics, so
# the Node-facing enum only uses the three-bucket schema:
#   side     -> green_side
#   behind   -> red_front_or_back
#   front    -> red_front_or_back
#   overhead -> unknown   (classifier can't see the player clearly enough)
#   unknown  -> unknown
# yellow_oblique exists in the DB enum but we never emit it -- if we ever add
# oblique detection in camera_classifier.py, map it here.


class ClassifyAngleRequest(BaseModel):
    video_url: str


_RAW_TO_TELEMETRY_FLAG = {
    "side": "green_side",
    "behind": "red_front_or_back",
    "front": "red_front_or_back",
    "overhead": "unknown",
    "unknown": "unknown",
}


def _sample_frame_indices(total_frames: int) -> list[int]:
    """Pick ~5 frames spread across the clip: start, 25%, 50%, 75%, end.

    Deduped and clamped so very short clips (fewer than 5 sampleable frames)
    still return something sensible. Returns an empty list if there are no
    frames to sample.
    """
    if total_frames <= 0:
        return []
    if total_frames <= 5:
        return list(range(total_frames))
    markers = [0.0, 0.25, 0.5, 0.75, 1.0]
    indices: list[int] = []
    for m in markers:
        idx = min(total_frames - 1, max(0, int(round(m * (total_frames - 1)))))
        if idx not in indices:
            indices.append(idx)
    return indices


def _aggregate_angle_labels(labels: list[str]) -> str:
    """Pick the majority raw label, or 'unknown' if the vote is split.

    We only count non-'unknown' labels toward majority. If every sample is
    'unknown' (classifier couldn't find court/player), we return 'unknown'.
    A single dominant label needs STRICT majority of the non-unknown votes
    -- ties and near-ties fall through to 'unknown' so mixed-angle clips
    don't get a confident flag they don't deserve.
    """
    if not labels:
        return "unknown"
    considered = [lbl for lbl in labels if lbl and lbl != "unknown"]
    if not considered:
        return "unknown"
    counts: dict[str, int] = {}
    for lbl in considered:
        counts[lbl] = counts.get(lbl, 0) + 1
    top_label, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count * 2 > len(considered):
        return top_label
    return "unknown"


def _classify_video_frames(video_path: str) -> tuple[str, int]:
    """Sample frames, classify each, and aggregate to a raw label.

    Returns (raw_label, samples_considered). raw_label is one of the five
    classifier labels. samples_considered is how many frames we actually ran
    classify_frame on (0 if the video couldn't be opened).
    """
    from camera_classifier import classify_frame  # local import so tests can patch

    cap = _open_video_autorotated(video_path)
    if not cap.isOpened():
        return "unknown", 0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = _sample_frame_indices(total)
        labels: list[str] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                _, raw = classify_frame(frame)
            except Exception as e:  # noqa: BLE001
                print(f"[classify-angle] classify_frame failed on frame {idx}: {e}")
                continue
            labels.append(raw or "unknown")
        return _aggregate_angle_labels(labels), len(labels)
    finally:
        cap.release()


@app.post("/classify-angle")
async def classify_angle(
    req: ClassifyAngleRequest,
    authorization: str = Header(default=""),
):
    """Classify the dominant camera angle for a clip and return a telemetry flag.

    Always returns 200 with a capture_quality_flag. Any failure path falls
    through to {"capture_quality_flag": "unknown", "error": "..."} so the
    Node caller never has to branch on HTTP errors -- telemetry must never
    block coaching.
    """
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _is_url_allowed(req.video_url):
        raise HTTPException(
            status_code=400,
            detail="video_url must use https and point to an allowed storage domain",
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(req.video_url)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(response.content)

        raw_label, samples = await asyncio.to_thread(_classify_video_frames, tmp_path)
        flag = _RAW_TO_TELEMETRY_FLAG.get(raw_label, "unknown")
        return {
            "capture_quality_flag": flag,
            "raw_label": raw_label,
            "samples_considered": samples,
        }
    except Exception as e:  # noqa: BLE001
        # Telemetry contract: never surface a non-200 to the Node caller.
        print(f"[classify-angle] failed for {req.video_url}: {e}")
        return {
            "capture_quality_flag": "unknown",
            "raw_label": "unknown",
            "samples_considered": 0,
            "error": str(e),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:  # noqa: BLE001
                print(f"[classify-angle] temp cleanup failed: {e}")


# ---------------------------------------------------------------------------
# Segment trimming for baseline capture
# ---------------------------------------------------------------------------
#
# Trims a source video by [start_ms, end_ms] using ffmpeg stream-copy and
# uploads the result to Vercel Blob. Stream-copy (no re-encode) keeps the
# clip lossless and avoids a multi-second CPU hit per call; ffmpeg will snap
# the cut to the nearest keyframe, which is accurate enough for a baseline
# thumbnail/preview.
#
# Requires ffmpeg on PATH (nixpacks.toml provides it) and BLOB_READ_WRITE_TOKEN.


class TrimVideoRequest(BaseModel):
    video_url: str
    start_ms: int
    end_ms: int


def _trim_with_ffmpeg(src_path: str, dst_path: str, start_ms: int, end_ms: int) -> None:
    """Stream-copy the [start_ms, end_ms] range from src_path to dst_path.

    Raises RuntimeError with the captured stderr if ffmpeg fails.
    """
    start_s = max(0, start_ms) / 1000.0
    duration_s = max(0, end_ms - start_ms) / 1000.0
    if duration_s <= 0:
        raise RuntimeError("end_ms must be greater than start_ms")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        src_path,
        "-t",
        f"{duration_s:.3f}",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg trim failed (exit {result.returncode}): {result.stderr[:2000]}"
        )


@app.post("/trim-video")
async def trim_video(
    req: TrimVideoRequest,
    authorization: str = Header(default=""),
):
    """Download *video_url*, trim to [start_ms, end_ms], upload to Vercel Blob.

    Returns { blob_url }. Uses stream-copy (no re-encode) so it's fast and
    lossless but cuts snap to the nearest keyframe.
    """
    expected = f"Bearer {EXTRACT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _is_url_allowed(req.video_url):
        raise HTTPException(
            status_code=400,
            detail="video_url must use https and point to an allowed storage domain",
        )

    if req.end_ms <= req.start_ms:
        raise HTTPException(status_code=400, detail="end_ms must be greater than start_ms")

    src_path: str | None = None
    dst_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src:
            src_path = src.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dst:
            dst_path = dst.name

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.get(req.video_url)
            response.raise_for_status()
            with open(src_path, "wb") as f:
                f.write(response.content)

        await asyncio.to_thread(
            _trim_with_ffmpeg, src_path, dst_path, req.start_ms, req.end_ms
        )

        blob_path = f"baseline-trims/{uuid.uuid4()}.mp4"
        blob_url = await _upload_file_to_blob(dst_path, blob_path)
        return {"blob_url": blob_url}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[trim-video] failed for {req.video_url}: {e}")
        raise HTTPException(status_code=500, detail=f"trim failed: {e}")
    finally:
        for p in (src_path, dst_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError as e:  # noqa: BLE001
                    print(f"[trim-video] temp cleanup failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/racket-detector")
def debug_racket_detector():
    """One-shot diagnostic: is the YOLO racket detector actually wired up on
    this deploy? Returns model-file paths + existence, attempts a session
    load, runs inference on a blank image. Meant for debugging "racket
    detected: 0 / N" situations where the frontend has no visibility into
    server state.
    """
    import numpy as np
    from pathlib import Path

    from racket_detector import (
        MODEL_DIR,
        YOLO_ONNX_PATH,
        YOLO_PT_PATH,
        detect_racket,
    )
    import racket_detector as rd

    def _size_or_none(p: Path):
        try:
            return p.stat().st_size if p.exists() else None
        except OSError:
            return None

    status = {
        "model_dir": str(MODEL_DIR),
        "model_dir_exists": MODEL_DIR.exists(),
        "onnx_path": str(YOLO_ONNX_PATH),
        "onnx_exists": YOLO_ONNX_PATH.exists(),
        "onnx_size_bytes": _size_or_none(YOLO_ONNX_PATH),
        "pt_path": str(YOLO_PT_PATH),
        "pt_exists": YOLO_PT_PATH.exists(),
        "pt_size_bytes": _size_or_none(YOLO_PT_PATH),
        "session_loaded": rd._session is not None,
    }

    try:
        rd._ensure_model()
        status["ensure_model"] = "ok"
        status["session_loaded"] = rd._session is not None
    except Exception as e:  # noqa: BLE001
        status["ensure_model"] = f"FAILED: {type(e).__name__}: {e}"
        return status

    try:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        result = detect_racket(blank)
        status["blank_inference"] = "ok"
        status["blank_detection"] = result  # expected: None
    except Exception as e:  # noqa: BLE001
        status["blank_inference"] = f"FAILED: {type(e).__name__}: {e}"

    return status
