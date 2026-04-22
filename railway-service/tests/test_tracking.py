"""Unit tests for the single-object trackers.

Tracker logic is pure data — no frames, no YOLO — so these tests run fast
and deterministically. The screenshot bug that motivated this file was
YOLO picking a small high-conf background figure over the large foreground
player; the cold-start and cold-stay-locked tests below are direct
regression guards for that exact failure mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tracking import PersonTracker, iou  # noqa: E402


IMG_W, IMG_H = 1920, 1080


# ---------------------------------------------------------------------------
# iou helper
# ---------------------------------------------------------------------------


class TestIou:
    def test_identical_bboxes_iou_is_one(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_bboxes_iou_is_zero(self):
        assert iou((0, 0, 10, 10), (100, 100, 120, 120)) == 0.0

    def test_half_overlap_iou(self):
        # Two 10x10 bboxes, 5px horizontal overlap -> 50 / (100+100-50) = 1/3
        result = iou((0, 0, 10, 10), (5, 0, 15, 10))
        assert abs(result - (50 / 150)) < 1e-6

    def test_zero_area_bbox_returns_zero(self):
        assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# PersonTracker: cold start — the regression that motivated this file
# ---------------------------------------------------------------------------


class TestPersonTrackerColdStart:
    def test_empty_candidates_returns_none(self):
        tracker = PersonTracker()
        assert tracker.update([], IMG_W, IMG_H) is None
        assert not tracker.is_locked

    def test_picks_large_centered_bbox_over_small_edge_bbox_even_with_lower_conf(self):
        """The exact screenshot bug: YOLO returned a small, high-confidence
        bbox on a background figure and a larger, lower-confidence bbox on
        the foreground player. raw-max-conf picked the ghost; the scored
        picker must pick the player."""
        # Foreground player: ~500x1000 centered, conf 0.55
        player = (700.0, 40.0, 1200.0, 1040.0, 0.55)
        # Background ghost: 60x120, upper-left corner, conf 0.95
        ghost = (50.0, 50.0, 110.0, 170.0, 0.95)

        tracker = PersonTracker()
        pick = tracker.update([ghost, player], IMG_W, IMG_H)

        assert pick is not None
        # bbox x1 near 700 = player; near 50 = ghost.
        assert pick[0] == 700.0, "tracker picked the background ghost"
        assert tracker.is_locked

    def test_penalizes_implausible_aspect_ratio(self):
        """A wide low-aspect bbox (fence / banner / bench) loses to a
        tall person-shaped bbox even at similar area + centrality."""
        # Horizontal banner: 800x120 centered, conf 0.9
        banner = (560.0, 480.0, 1360.0, 600.0, 0.9)
        # Player: 300x750 centered, conf 0.6
        player = (810.0, 165.0, 1110.0, 915.0, 0.6)
        tracker = PersonTracker()
        pick = tracker.update([banner, player], IMG_W, IMG_H)
        assert pick is not None
        assert pick[0] == 810.0, "tracker picked the horizontal banner"


# ---------------------------------------------------------------------------
# PersonTracker: association (the second half of the screenshot bug —
# once locked, the tracker must not switch to a different person)
# ---------------------------------------------------------------------------


class TestPersonTrackerAssociation:
    def test_stays_locked_when_background_detection_appears(self):
        player = (700.0, 40.0, 1200.0, 1040.0, 0.6)
        tracker = PersonTracker()
        tracker.update([player], IMG_W, IMG_H)

        # Next frame: player moves slightly, plus a random background
        # detection appears at high confidence. The tracker must keep
        # following the player by IoU, not switch to the ghost.
        player_moved = (720.0, 45.0, 1220.0, 1045.0, 0.55)
        ghost = (50.0, 50.0, 110.0, 170.0, 0.99)

        pick = tracker.update([ghost, player_moved], IMG_W, IMG_H)
        assert pick is not None
        # x1 near 700–720 means we followed the player; near 50 means we
        # jumped to the ghost.
        assert pick[0] > 600, f"tracker switched to ghost (x1={pick[0]})"

    def test_coasts_when_no_candidate_clears_iou_gate(self):
        player = (700.0, 40.0, 1200.0, 1040.0, 0.6)
        tracker = PersonTracker()
        tracker.update([player], IMG_W, IMG_H)

        # Frame with only far-away detections (no IoU overlap with player).
        far = (50.0, 50.0, 110.0, 170.0, 0.9)
        pick = tracker.update([far], IMG_W, IMG_H)

        assert pick is not None
        # We should still be showing the player's last known position.
        assert pick[0] == 700.0

    def test_resets_after_max_coast_frames(self):
        player = (700.0, 40.0, 1200.0, 1040.0, 0.6)
        tracker = PersonTracker(max_coast_frames=3)
        tracker.update([player], IMG_W, IMG_H)

        # Coast for 4 frames with only off-gate detections. On the 4th
        # coast the tracker should reset and cold-start on whatever is
        # available.
        far = (50.0, 50.0, 110.0, 170.0, 0.9)
        # Coast 1, 2, 3 — still returning player's last known.
        for _ in range(3):
            tracker.update([far], IMG_W, IMG_H)
        # Frame 4 exceeds max_coast_frames=3, triggers reset + cold-start
        # on the only candidate (far). Now tracker locks onto it.
        pick = tracker.update([far], IMG_W, IMG_H)
        assert pick is not None
        assert pick[0] == 50.0, "tracker did not cold-start after timeout"

    def test_ema_smooths_jittery_detections(self):
        """When the IoU match is accepted, the returned bbox should be a
        smoothed blend of the old reference and the new detection — not
        the raw detection. This prevents per-frame YOLO jitter from
        making the player-bbox (and thus the pose crop) twitch."""
        ref = (700.0, 40.0, 1200.0, 1040.0, 0.6)
        tracker = PersonTracker(ema_alpha=0.7)
        tracker.update([ref], IMG_W, IMG_H)

        # New detection shifted by +100px. EMA-smoothed x1 should land
        # at 0.7*800 + 0.3*700 = 770, not 800 and not 700.
        shifted = (800.0, 40.0, 1300.0, 1040.0, 0.6)
        pick = tracker.update([shifted], IMG_W, IMG_H)
        assert pick is not None
        assert 765 < pick[0] < 775, (
            f"ema smoothing did not blend (expected ~770, got {pick[0]})"
        )

    def test_reset_clears_lock(self):
        tracker = PersonTracker()
        tracker.update([(700.0, 40.0, 1200.0, 1040.0, 0.6)], IMG_W, IMG_H)
        assert tracker.is_locked
        tracker.reset()
        assert not tracker.is_locked
