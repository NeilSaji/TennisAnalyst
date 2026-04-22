"""Unit tests for the RTMPose backend.

Exercises the pure functions (id mapping, bbox expansion, YOLO output
decode, letterbox geometry) and the top-level inference path with the
heavy ONNX sessions stubbed out. No real model weights are loaded;
no Railway / network access required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pose_rtmpose import (  # noqa: E402
    BLAZEPOSE_LANDMARK_NAMES,
    COCO17_TO_BLAZEPOSE33,
    NUM_BLAZEPOSE_LANDMARKS,
    _decode_yolo_output,
    _letterbox_for_yolo,
    _reset_for_tests,
    coco17_to_blazepose33,
    expand_bbox,
)
import pose_rtmpose  # noqa: E402  -- module-level patching below


# ---------------------------------------------------------------------------
# Setup / teardown for tests that mutate module-level state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_pose_module():
    _reset_for_tests()
    yield
    _reset_for_tests()


# ---------------------------------------------------------------------------
# coco17_to_blazepose33: id mapping correctness
# ---------------------------------------------------------------------------


class TestCoco17ToBlazepose33:
    def test_returns_33_landmarks(self):
        kpts = np.zeros((17, 2))
        scores = np.zeros(17)
        out = coco17_to_blazepose33(kpts, scores, 100, 100)
        assert len(out) == NUM_BLAZEPOSE_LANDMARKS

    def test_unfilled_ids_have_visibility_zero(self):
        # All COCO scores at 1.0; unmapped BlazePose ids must still be 0.
        kpts = np.full((17, 2), 50.0)
        scores = np.full(17, 1.0)
        out = coco17_to_blazepose33(kpts, scores, 100, 100)

        filled_ids = set(COCO17_TO_BLAZEPOSE33.values())
        for lm in out:
            if lm["id"] in filled_ids:
                assert lm["visibility"] == 1.0, f"id {lm['id']} should be filled"
            else:
                assert lm["visibility"] == 0.0, (
                    f"id {lm['id']} ({lm['name']}) should not be filled by COCO-17"
                )

    def test_normalizes_pixel_coords_to_unit_interval(self):
        kpts = np.zeros((17, 2))
        scores = np.full(17, 0.9)
        # Place left_shoulder (COCO id 5 -> BlazePose id 11) at (50px, 75px)
        kpts[5] = (50.0, 75.0)
        out = coco17_to_blazepose33(kpts, scores, 200, 300)

        ls = next(lm for lm in out if lm["id"] == 11)
        assert ls["x"] == pytest.approx(0.25, abs=1e-3)
        assert ls["y"] == pytest.approx(0.25, abs=1e-3)
        assert ls["visibility"] == pytest.approx(0.9, abs=1e-3)

    def test_clips_offframe_coords_to_unit_interval(self):
        kpts = np.zeros((17, 2))
        scores = np.ones(17)
        # Off-frame x = -10 should clip to 0; off-frame y = 200 (image h=100)
        # should clip to 1.0.
        kpts[0] = (-10.0, 200.0)  # nose -> BlazePose 0
        out = coco17_to_blazepose33(kpts, scores, 100, 100)
        nose = next(lm for lm in out if lm["id"] == 0)
        assert nose["x"] == 0.0
        assert nose["y"] == 1.0

    def test_specific_critical_joints_map_correctly(self):
        # The shoulders, elbows, wrists, hips, knees, ankles -- the joints
        # downstream code actually consumes -- must hit the right BlazePose
        # ids. If this drift, jointAngles.ts breaks silently.
        kpts = np.zeros((17, 2))
        scores = np.zeros(17)
        # Mark each joint with a unique score so we can verify the mapping
        coco_to_score = {
            5: 0.55, 6: 0.66,    # shoulders
            7: 0.77, 8: 0.88,    # elbows
            9: 0.99, 10: 0.10,   # wrists
            11: 0.21, 12: 0.32,  # hips
            13: 0.43, 14: 0.54,  # knees
            15: 0.65, 16: 0.76,  # ankles
        }
        for cid, s in coco_to_score.items():
            scores[cid] = s

        out = coco17_to_blazepose33(kpts, scores, 100, 100)
        out_by_id = {lm["id"]: lm for lm in out}

        expected = {
            11: 0.55,   # left_shoulder
            12: 0.66,   # right_shoulder
            13: 0.77,   # left_elbow
            14: 0.88,   # right_elbow
            15: 0.99,   # left_wrist
            16: 0.10,   # right_wrist
            23: 0.21,   # left_hip
            24: 0.32,   # right_hip
            25: 0.43,   # left_knee
            26: 0.54,   # right_knee
            27: 0.65,   # left_ankle
            28: 0.76,   # right_ankle
        }
        for blaze_id, expected_score in expected.items():
            assert out_by_id[blaze_id]["visibility"] == pytest.approx(
                expected_score, abs=1e-3
            ), f"BlazePose id {blaze_id} ({BLAZEPOSE_LANDMARK_NAMES[blaze_id]})"

    def test_index_finger_landmarks_stay_unfilled(self):
        # The schema_version=3 contract: ids 19/20 (left/right_index_finger)
        # are never filled by RTMPose, so wrist-flexion joint angles must
        # not appear downstream.
        kpts = np.full((17, 2), 50.0)
        scores = np.ones(17)
        out = coco17_to_blazepose33(kpts, scores, 100, 100)
        out_by_id = {lm["id"]: lm for lm in out}
        assert out_by_id[19]["visibility"] == 0.0
        assert out_by_id[20]["visibility"] == 0.0

    def test_rejects_wrong_input_shape(self):
        with pytest.raises(ValueError):
            coco17_to_blazepose33(np.zeros((16, 2)), np.zeros(17), 100, 100)
        with pytest.raises(ValueError):
            coco17_to_blazepose33(np.zeros((17, 2)), np.zeros(16), 100, 100)
        with pytest.raises(ValueError):
            coco17_to_blazepose33(np.zeros((17, 2)), np.zeros(17), 0, 100)


# ---------------------------------------------------------------------------
# expand_bbox
# ---------------------------------------------------------------------------


class TestExpandBbox:
    def test_expansion_default_pct(self):
        # bbox 100x200 with 8% expand -> +8px on x (100*0.08), +16px on y
        out = expand_bbox((50, 50, 150, 250, 0.9), 1000, 1000)
        assert out[0] == pytest.approx(50 - 8, abs=0.001)
        assert out[1] == pytest.approx(50 - 16, abs=0.001)
        assert out[2] == pytest.approx(150 + 8, abs=0.001)
        assert out[3] == pytest.approx(250 + 16, abs=0.001)

    def test_clips_to_image_bounds(self):
        # bbox at the corner -- expansion should not produce negatives or
        # values past the image edge.
        out = expand_bbox((0, 0, 50, 50, 0.9), 100, 100, pct=0.5)
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] <= 99.0
        assert out[3] <= 99.0

    def test_drops_confidence(self):
        # Result tuple is 4-element xyxy, no conf
        out = expand_bbox((10, 10, 20, 20, 0.5), 100, 100)
        assert len(out) == 4


# ---------------------------------------------------------------------------
# _letterbox_for_yolo: reversibility math
# ---------------------------------------------------------------------------


class TestLetterboxForYolo:
    def test_landscape_image_fits_with_horizontal_pad(self):
        # 1280x720 landscape -> letterbox to 640. Should pad top/bottom only.
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        padded, scale, pad = _letterbox_for_yolo(img, new_shape=640)
        assert padded.shape == (640, 640, 3)
        # Scale = min(640/720, 640/1280) = 640/1280 = 0.5
        assert scale == pytest.approx(0.5, abs=1e-3)
        # pad_x should be ~0 (the wide axis filled the full 640)
        assert pad[0] == 0
        # pad_y should be (640 - 360) // 2 = 140
        assert pad[1] == 140

    def test_portrait_image_fits_with_horizontal_pad(self):
        img = np.zeros((1280, 720, 3), dtype=np.uint8)
        padded, scale, pad = _letterbox_for_yolo(img, new_shape=640)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(0.5, abs=1e-3)
        # pad_y should be 0; pad_x should be 140
        assert pad[1] == 0
        assert pad[0] == 140

    def test_inverse_transform_recovers_image_coords(self):
        # Verify the math by mapping a known point through letterbox and back.
        # Point at image (640, 360) on a 1280x720 frame -> letterbox space
        # (640*0.5+0, 360*0.5+140) = (320, 320) -- the center of the
        # 640x640 letterbox.
        img_h, img_w = 720, 1280
        scale = min(640 / img_h, 640 / img_w)
        new_w, new_h = int(round(img_w * scale)), int(round(img_h * scale))
        pad_w = (640 - new_w) // 2
        pad_h = (640 - new_h) // 2

        # Forward: image (640, 360) -> letterbox
        lx = 640 * scale + pad_w
        ly = 360 * scale + pad_h
        assert lx == pytest.approx(320, abs=1)
        assert ly == pytest.approx(320, abs=1)

        # Inverse: (lx, ly) -> image
        ix = (lx - pad_w) / scale
        iy = (ly - pad_h) / scale
        assert ix == pytest.approx(640, abs=1)
        assert iy == pytest.approx(360, abs=1)


# ---------------------------------------------------------------------------
# _decode_yolo_output: synthetic YOLO output decode + bbox selection
# ---------------------------------------------------------------------------


class TestDecodeYoloOutput:
    def _make_yolo_output(
        self,
        detections: list[tuple[float, float, float, float, int, float]],
        num_classes: int = 80,
    ) -> np.ndarray:
        """Build a fake YOLO ONNX output tensor (1, 4+num_classes, N).

        Each detection is (cx, cy, w, h, class_id, conf). One-hot the class
        score, leave the rest at zero.
        """
        n = len(detections)
        out = np.zeros((1, 4 + num_classes, n), dtype=np.float32)
        for i, (cx, cy, w, h, cls, conf) in enumerate(detections):
            out[0, 0, i] = cx
            out[0, 1, i] = cy
            out[0, 2, i] = w
            out[0, 3, i] = h
            out[0, 4 + cls, i] = conf
        return out

    def test_filters_by_class_id(self):
        # One person (class 0, conf 0.9), one cat (class 15, conf 0.95).
        # We ask for class 0 -- only the person should come back.
        out = self._make_yolo_output([
            (320, 320, 100, 200, 0, 0.9),
            (320, 320, 100, 200, 15, 0.95),
        ])
        candidates = _decode_yolo_output(
            out, scale=1.0, pad=(0, 0), img_w=640, img_h=640,
            class_id=0, conf_threshold=0.3,
        )
        assert len(candidates) == 1
        assert candidates[0][4] == pytest.approx(0.9, abs=1e-3)

    def test_filters_by_confidence(self):
        out = self._make_yolo_output([
            (320, 320, 100, 200, 0, 0.2),  # below threshold
            (320, 320, 100, 200, 0, 0.5),  # above
        ])
        candidates = _decode_yolo_output(
            out, scale=1.0, pad=(0, 0), img_w=640, img_h=640,
            class_id=0, conf_threshold=0.3,
        )
        assert len(candidates) == 1
        assert candidates[0][4] == pytest.approx(0.5, abs=1e-3)

    def test_inverse_transforms_letterbox_to_image_coords(self):
        # A detection centered at letterbox (320, 320) with letterbox
        # scale=0.5 and pad=(0, 140) corresponds to image (640, 360) on
        # a 1280x720 source.
        out = self._make_yolo_output([
            (320, 320, 100, 100, 0, 0.9),  # bbox 100x100 in letterbox px
        ])
        candidates = _decode_yolo_output(
            out, scale=0.5, pad=(0, 140), img_w=1280, img_h=720,
            class_id=0, conf_threshold=0.3,
        )
        assert len(candidates) == 1
        x1, y1, x2, y2, _ = candidates[0]
        # 100px in letterbox -> 200px in image (scale 0.5).
        # Center 320 letterbox -> (320 - 0) / 0.5 = 640 image x; ditto y.
        assert (x1 + x2) / 2 == pytest.approx(640, abs=1)
        assert (y1 + y2) / 2 == pytest.approx(360, abs=1)
        assert (x2 - x1) == pytest.approx(200, abs=1)
        assert (y2 - y1) == pytest.approx(200, abs=1)

    def test_returns_empty_on_no_detections(self):
        out = np.zeros((1, 4 + 80, 5), dtype=np.float32)
        candidates = _decode_yolo_output(
            out, scale=1.0, pad=(0, 0), img_w=640, img_h=640,
            class_id=0, conf_threshold=0.3,
        )
        assert candidates == []


# ---------------------------------------------------------------------------
# infer_pose_for_frame: top-level integration with stubbed sessions
# ---------------------------------------------------------------------------


class TestInferPoseForFrameStubbed:
    def test_returns_none_when_no_person_detected(self, monkeypatch):
        monkeypatch.setattr(pose_rtmpose, "detect_person_bbox", lambda _: None)
        out = pose_rtmpose.infer_pose_for_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert out is None

    def test_returns_landmark_list_when_person_detected(self, monkeypatch):
        # Stub the YOLO bbox: a person from (10,10) to (90,90).
        monkeypatch.setattr(
            pose_rtmpose, "detect_person_bbox",
            lambda _: (10.0, 10.0, 90.0, 90.0, 0.95),
        )

        # Stub the rtmlib RTMPose session: any call returns 17 keypoints
        # at the image center with high confidence.
        fake_kpts = np.full((1, 17, 2), 50.0)
        fake_scores = np.full((1, 17), 0.85)
        fake_session = MagicMock(return_value=(fake_kpts, fake_scores))

        # Bypass the lazy-init by writing the cached session directly.
        pose_rtmpose._rtm_session = fake_session

        out = pose_rtmpose.infer_pose_for_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert out is not None
        assert len(out) == NUM_BLAZEPOSE_LANDMARKS
        # The shoulders / elbows / wrists / hips / knees / ankles should
        # all carry 0.85 visibility (the stubbed score).
        out_by_id = {lm["id"]: lm for lm in out}
        for blaze_id in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
            assert out_by_id[blaze_id]["visibility"] == pytest.approx(0.85, abs=1e-3)

    def test_passes_expanded_bbox_to_rtmpose(self, monkeypatch):
        # Verify that the bbox we hand to rtmlib is the EXPANDED one,
        # not the raw YOLO output -- the prior crop attempt's downfall
        # was passing the wrong-shaped region to the pose model.
        raw_bbox = (100.0, 100.0, 200.0, 300.0, 0.9)
        monkeypatch.setattr(pose_rtmpose, "detect_person_bbox", lambda _: raw_bbox)

        captured = {}

        def fake_session(image, bboxes):
            captured["bboxes"] = bboxes
            return np.full((1, 17, 2), 50.0), np.full((1, 17), 0.5)

        pose_rtmpose._rtm_session = fake_session

        pose_rtmpose.infer_pose_for_frame(np.zeros((400, 400, 3), dtype=np.uint8))

        assert "bboxes" in captured
        bbox = captured["bboxes"][0]
        # Should be 4-element xyxy (no confidence), expanded by 8% per side.
        # Width 100 -> +8 each; height 200 -> +16 each.
        assert len(bbox) == 4
        assert bbox[0] == pytest.approx(100 - 8, abs=0.001)
        assert bbox[1] == pytest.approx(100 - 16, abs=0.001)
        assert bbox[2] == pytest.approx(200 + 8, abs=0.001)
        assert bbox[3] == pytest.approx(300 + 16, abs=0.001)
