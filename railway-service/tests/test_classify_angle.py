"""Unit tests for the /classify-angle aggregation logic.

We don't exercise the FastAPI route itself (that requires httpx + network
mocks); we hit the pure aggregation helpers directly. Covers:
- majority-side votes -> green_side
- majority-behind votes -> red_front_or_back
- mixed votes -> unknown
- all-unknown -> unknown
- short-clip frame sampling edge case
- _classify_video_frames with a monkey-patched classify_frame
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Allow imports from the railway-service package
sys.path.insert(0, str(Path(__file__).parent.parent))

# main.py reads SUPABASE_URL / SUPABASE_SERVICE_KEY at import time and calls
# create_client. Set dummy env vars before importing main so the module loads.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key-not-real")
os.environ.setdefault("EXTRACT_API_KEY", "test-api-key")

_mock_supabase_client = MagicMock()
patch("supabase.create_client", return_value=_mock_supabase_client).start()


class TestAggregateAngleLabels:
    """Aggregation of per-frame raw labels -> single dominant raw label."""

    def test_majority_side_wins(self):
        from main import _aggregate_angle_labels

        # 3 side vs 1 behind vs 1 unknown -> side wins (3/4 of non-unknown)
        assert _aggregate_angle_labels(["side", "side", "side", "behind", "unknown"]) == "side"

    def test_majority_behind_wins(self):
        from main import _aggregate_angle_labels

        assert _aggregate_angle_labels(["behind", "behind", "behind", "side"]) == "behind"

    def test_all_behind(self):
        from main import _aggregate_angle_labels

        # Used downstream for red_front_or_back mapping
        assert _aggregate_angle_labels(["behind", "behind", "behind"]) == "behind"

    def test_mixed_no_majority_returns_unknown(self):
        from main import _aggregate_angle_labels

        # 2 side vs 2 behind -> tie -> unknown
        assert _aggregate_angle_labels(["side", "side", "behind", "behind"]) == "unknown"

    def test_all_unknown_returns_unknown(self):
        from main import _aggregate_angle_labels

        assert _aggregate_angle_labels(["unknown", "unknown", "unknown"]) == "unknown"

    def test_empty_returns_unknown(self):
        from main import _aggregate_angle_labels

        assert _aggregate_angle_labels([]) == "unknown"

    def test_unknown_doesnt_count_toward_quorum(self):
        from main import _aggregate_angle_labels

        # 2 side, 3 unknown -> side is the only considered label -> wins
        assert _aggregate_angle_labels(["side", "side", "unknown", "unknown", "unknown"]) == "side"

    def test_single_vote_wins(self):
        from main import _aggregate_angle_labels

        # One real vote beats a pile of unknowns
        assert _aggregate_angle_labels(["side", "unknown", "unknown"]) == "side"

    def test_front_majority_preserved_raw(self):
        from main import _aggregate_angle_labels

        # The aggregator returns the raw classifier label; the Node-facing
        # enum mapping happens separately via _RAW_TO_TELEMETRY_FLAG.
        assert _aggregate_angle_labels(["front", "front", "front", "side"]) == "front"


class TestRawToTelemetryFlag:
    """The classifier label -> DB enum mapping."""

    def test_side_maps_to_green(self):
        from main import _RAW_TO_TELEMETRY_FLAG

        assert _RAW_TO_TELEMETRY_FLAG["side"] == "green_side"

    def test_behind_and_front_map_to_red(self):
        from main import _RAW_TO_TELEMETRY_FLAG

        assert _RAW_TO_TELEMETRY_FLAG["behind"] == "red_front_or_back"
        assert _RAW_TO_TELEMETRY_FLAG["front"] == "red_front_or_back"

    def test_overhead_and_unknown_map_to_unknown(self):
        from main import _RAW_TO_TELEMETRY_FLAG

        assert _RAW_TO_TELEMETRY_FLAG["overhead"] == "unknown"
        assert _RAW_TO_TELEMETRY_FLAG["unknown"] == "unknown"

    def test_oblique_not_emitted(self):
        from main import _RAW_TO_TELEMETRY_FLAG

        # camera_classifier.py has no oblique detection; the DB enum accepts
        # 'yellow_oblique' but we never produce it. Guard the mapping so a
        # future attempt to add it lights up this test.
        assert "oblique" not in _RAW_TO_TELEMETRY_FLAG
        assert "yellow_oblique" not in _RAW_TO_TELEMETRY_FLAG.values()


class TestSampleFrameIndices:
    """Frame-index picker for spread sampling."""

    def test_zero_frames(self):
        from main import _sample_frame_indices

        assert _sample_frame_indices(0) == []

    def test_short_clip_returns_all_frames(self):
        from main import _sample_frame_indices

        # <= 5 frames: take them all so we don't skip the whole clip
        assert _sample_frame_indices(3) == [0, 1, 2]

    def test_five_frames_returns_all(self):
        from main import _sample_frame_indices

        assert _sample_frame_indices(5) == [0, 1, 2, 3, 4]

    def test_long_clip_samples_five_spread(self):
        from main import _sample_frame_indices

        indices = _sample_frame_indices(100)
        # start + 25/50/75 + end of a 100-frame clip (0-99)
        assert indices == [0, 25, 50, 74, 99]

    def test_sampling_deduped(self):
        from main import _sample_frame_indices

        # 6 frames (0..5) -- the 25/50/75 rounding could collide with
        # start/end; dedupe should keep it unique.
        indices = _sample_frame_indices(6)
        assert len(indices) == len(set(indices))
        assert all(0 <= i <= 5 for i in indices)


class TestClassifyVideoFramesIntegration:
    """_classify_video_frames with classify_frame monkey-patched."""

    def test_returns_unknown_when_video_unopenable(self, tmp_path):
        from main import _classify_video_frames

        fake = tmp_path / "not-a-video.mp4"
        fake.write_bytes(b"not a real video file")
        raw, samples = _classify_video_frames(str(fake))
        assert raw == "unknown"
        assert samples == 0

    def test_all_side_frames_aggregates_to_side(self, tmp_path):
        """A clip where classify_frame always returns 'side' should return
        raw='side' (which maps to green_side downstream)."""
        import cv2 as cv2_mod

        # Write a minimal 10-frame synthetic video
        video_path = tmp_path / "clip.mp4"
        fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
        writer = cv2_mod.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
        for _ in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        if not video_path.exists() or video_path.stat().st_size == 0:
            # mp4v may not be available in this environment; skip rather than fail
            import pytest

            pytest.skip("mp4v codec unavailable for synthetic fixture")

        from main import _classify_video_frames

        with patch("camera_classifier.classify_frame", return_value=(True, "side")):
            raw, samples = _classify_video_frames(str(video_path))
        assert raw == "side"
        assert samples > 0

    def test_all_behind_aggregates_to_behind(self, tmp_path):
        import cv2 as cv2_mod

        video_path = tmp_path / "clip.mp4"
        fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
        writer = cv2_mod.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
        for _ in range(10):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()

        if not video_path.exists() or video_path.stat().st_size == 0:
            import pytest

            pytest.skip("mp4v codec unavailable for synthetic fixture")

        from main import _classify_video_frames

        with patch("camera_classifier.classify_frame", return_value=(True, "behind")):
            raw, _ = _classify_video_frames(str(video_path))
        assert raw == "behind"

    def test_classify_frame_exception_is_skipped_not_crashed(self, tmp_path):
        """If classify_frame raises on some frames, we keep going and aggregate
        what's left. This is the 'never crash telemetry' contract."""
        import cv2 as cv2_mod

        video_path = tmp_path / "clip.mp4"
        fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
        writer = cv2_mod.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
        for _ in range(10):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()

        if not video_path.exists() or video_path.stat().st_size == 0:
            import pytest

            pytest.skip("mp4v codec unavailable for synthetic fixture")

        from main import _classify_video_frames

        # Every call raises -> samples list is empty -> raw='unknown'
        with patch("camera_classifier.classify_frame", side_effect=RuntimeError("boom")):
            raw, samples = _classify_video_frames(str(video_path))
        assert raw == "unknown"
        assert samples == 0
