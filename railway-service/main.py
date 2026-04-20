"""
TennisIQ Railway Microservice
FastAPI service for server-side pose extraction from videos.
Used for processing pro player videos offline and seeding the database.
"""

import asyncio
import os
import math
import re
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

# POSE_BACKEND selects the pose estimator. `mediapipe` is the only supported
# value today; `rtmpose` is a deliberate seam for a future swap.
POSE_BACKEND = os.environ.get("POSE_BACKEND", "mediapipe").lower()

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


def compute_joint_angles_from_dicts(landmarks: list[dict]) -> dict:
    """Compute key joint angles from a list of landmark dicts with x, y keys."""
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
        angles["hip_rotation"] = round(
            abs(math.degrees(math.atan2(hip_vec[1], hip_vec[0]))), 1
        )
    if pts["ls"] and pts["rs"]:
        sh_vec = (pts["rs"][0] - pts["ls"][0], pts["rs"][1] - pts["ls"][1])
        angles["trunk_rotation"] = round(
            abs(math.degrees(math.atan2(sh_vec[1], sh_vec[0]))), 1
        )

    return angles


def extract_keypoints_from_video(video_path: str, sample_fps: int = 30, max_seconds: float = 0) -> dict:
    """Extract pose keypoints from a video file using MediaPipe Tasks API.

    Args:
        max_seconds: Stop after this many seconds. 0 = process entire video.
    """
    if POSE_BACKEND == "rtmpose":
        raise NotImplementedError("rtmpose backend not yet implemented")
    if POSE_BACKEND != "mediapipe":
        raise ValueError(f"Unknown POSE_BACKEND: {POSE_BACKEND}")

    try:
        from racket_detector import detect_racket  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[extract] racket_detector unavailable: {e}")
        detect_racket = None  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / sample_fps))
    max_frame = int(max_seconds * video_fps) if max_seconds > 0 else 0
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
    blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not blob_token:
        raise RuntimeError("BLOB_READ_WRITE_TOKEN not set — cannot upload clips")

    filename = f"pro-clips/{shot_type}/{uuid.uuid4()}.mp4"

    with open(clip_path, "rb") as f:
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


@app.get("/health")
def health():
    return {"status": "ok"}
