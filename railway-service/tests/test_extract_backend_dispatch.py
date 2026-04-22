"""Smoke tests for the POSE_BACKEND env-var dispatch in main.extract_keypoints_from_video
and extract_clip_keypoints.extract_keypoints.

The actual extraction functions are heavy (need real video + ONNX
weights), so we verify the dispatch table picks the right inner
function for each backend value, and rejects unknown backends.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# main.py touches Supabase + EXTRACT_API_KEY at import time.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key-not-real")
os.environ.setdefault("EXTRACT_API_KEY", "test-api-key")
patch("supabase.create_client", return_value=MagicMock()).start()


class TestMainExtractDispatch:
    """main.extract_keypoints_from_video should dispatch on POSE_BACKEND."""

    def test_default_dispatches_to_mediapipe(self, monkeypatch):
        import main

        called = {}

        def fake_mp(video_path, sample_fps, max_seconds):
            called["mp"] = (video_path, sample_fps, max_seconds)
            return {"frame_count": 0, "frames": [], "schema_version": 2}

        monkeypatch.setattr(main, "POSE_BACKEND", "mediapipe")
        monkeypatch.setattr(main, "_extract_with_mediapipe", fake_mp)

        out = main.extract_keypoints_from_video("/tmp/x.mp4")
        assert "mp" in called
        assert out["schema_version"] == 2

    def test_rtmpose_dispatches_to_rtmpose(self, monkeypatch):
        import main

        called = {}

        def fake_rtm(video_path, sample_fps, max_seconds):
            called["rtm"] = (video_path, sample_fps, max_seconds)
            return {"frame_count": 0, "frames": [], "schema_version": 3}

        monkeypatch.setattr(main, "POSE_BACKEND", "rtmpose")
        monkeypatch.setattr(main, "_extract_with_rtmpose", fake_rtm)

        out = main.extract_keypoints_from_video("/tmp/x.mp4", sample_fps=24, max_seconds=2)
        assert called["rtm"] == ("/tmp/x.mp4", 24, 2)
        assert out["schema_version"] == 3

    def test_unknown_backend_raises(self, monkeypatch):
        import main

        monkeypatch.setattr(main, "POSE_BACKEND", "totally-not-real")
        with pytest.raises(ValueError, match="Unknown POSE_BACKEND"):
            main.extract_keypoints_from_video("/tmp/x.mp4")


class TestStandaloneExtractDispatch:
    def test_default_dispatches_to_mediapipe(self, monkeypatch):
        import extract_clip_keypoints as eck

        called = {}

        def fake_mp(video_path, sample_fps):
            called["mp"] = (video_path, sample_fps)
            return {"frame_count": 0, "frames": [], "schema_version": 2}

        monkeypatch.setattr(eck, "POSE_BACKEND", "mediapipe")
        monkeypatch.setattr(eck, "_extract_mediapipe", fake_mp)

        out = eck.extract_keypoints("/tmp/x.mp4")
        assert "mp" in called
        assert out["schema_version"] == 2

    def test_rtmpose_dispatches_to_rtmpose(self, monkeypatch):
        import extract_clip_keypoints as eck

        called = {}

        def fake_rtm(video_path, sample_fps):
            called["rtm"] = (video_path, sample_fps)
            return {"frame_count": 0, "frames": [], "schema_version": 3}

        monkeypatch.setattr(eck, "POSE_BACKEND", "rtmpose")
        monkeypatch.setattr(eck, "_extract_rtmpose", fake_rtm)

        out = eck.extract_keypoints("/tmp/x.mp4", sample_fps=15)
        assert called["rtm"] == ("/tmp/x.mp4", 15)
        assert out["schema_version"] == 3

    def test_unknown_backend_raises(self, monkeypatch):
        import extract_clip_keypoints as eck

        monkeypatch.setattr(eck, "POSE_BACKEND", "garbage")
        with pytest.raises(ValueError, match="Unknown POSE_BACKEND"):
            eck.extract_keypoints("/tmp/x.mp4")


class TestRtmposeExtractWithStubs:
    """Drive _extract_with_rtmpose end-to-end using stubbed video I/O and stubbed
    pose inference. Verifies the per-frame loop, racket-head wiring, and the
    schema_version=3 stamp."""

    def test_emits_schema_v3_with_stubbed_pose(self, monkeypatch, tmp_path):
        import main
        import pose_rtmpose

        # Stub pose inference: every frame returns a synthetic 33-landmark
        # dict with shoulders / elbows visible.
        def fake_infer(_frame, person_tracker=None):
            return [
                {
                    "id": i,
                    "name": f"landmark_{i}",
                    "x": 0.5,
                    "y": 0.5,
                    "z": 0.0,
                    "visibility": 0.9 if i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28) else 0.0,
                }
                for i in range(33)
            ]

        monkeypatch.setattr(pose_rtmpose, "infer_pose_for_frame", fake_infer)

        # Stub OpenCV VideoCapture so we don't need a real file.
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        # Yield 5 frames then EOF. Each frame is a 100x100 BGR ndarray.
        import numpy as np
        frames_iter = iter([
            (True, np.zeros((100, 100, 3), dtype=np.uint8)) for _ in range(5)
        ] + [(False, None)])
        fake_cap.read.side_effect = lambda: next(frames_iter)
        fake_cap.get.return_value = 30.0  # video_fps
        monkeypatch.setattr(main.cv2, "VideoCapture", lambda _path: fake_cap)

        # Stub racket detector to None (we test the no-racket branch here).
        monkeypatch.setattr(main, "_try_load_racket_detector", lambda: None)

        out = main._extract_with_rtmpose("/fake/path.mp4", sample_fps=30, max_seconds=0)

        assert out["schema_version"] == 3
        assert out["frame_count"] == 5
        assert len(out["frames"]) == 5
        # First frame: timestamp_ms=0, frame_index=0, has joint angles, racket_head=None
        f0 = out["frames"][0]
        assert f0["frame_index"] == 0
        assert f0["timestamp_ms"] == 0.0
        assert f0["racket_head"] is None
        # Joint angles must NOT include wrist-flexion (no index_finger landmarks).
        assert "right_wrist" not in f0["joint_angles"]
        assert "left_wrist" not in f0["joint_angles"]
        # But shoulder/elbow/knee/hip angles should be there (the joints we filled).
        assert "right_elbow" in f0["joint_angles"]
        assert "left_elbow" in f0["joint_angles"]
