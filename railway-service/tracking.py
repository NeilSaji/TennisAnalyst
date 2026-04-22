"""Single-object trackers for the tennis analyzer extraction pipeline.

Pure data in, pure data out: no OpenCV, no YOLO, no frame pixels. Integration
code calls the detector, hands the raw detections to the tracker, and gets
back the canonical per-frame position. This keeps trackers unit-testable
without model weights.

`PersonTracker` locks onto the player on the first good frame and prefers
IoU-overlap-to-previous on subsequent frames. Fixes the "YOLO picked a
background figure" bug where a small high-confidence false positive
(fence, distant person, net post) beat the large foreground player on
raw confidence.

Racket tracking lives separately; see RacketTracker (next commit) for
the Kalman-filter-based variant that needs to coast through motion-blur
gaps.
"""
from __future__ import annotations

import math
from typing import Optional


BBox = tuple[float, float, float, float]
BBoxWithConf = tuple[float, float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union for two xyxy bboxes. Returns 0 on empty
    intersection, 0 on degenerate input (zero-area bbox)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _cold_start_score(
    bbox: BBoxWithConf, img_w: int, img_h: int
) -> float:
    """Score a candidate person bbox when the tracker has no prior state.

    Heuristic: area × centrality × aspect-plausibility × confidence. This
    is the selector that replaces `max(candidates, key=conf)` — a small
    high-confidence false positive near the frame edge scores far lower
    than a large centered player-shaped bbox, even if YOLO was slightly
    less certain on the player.

    Scoring components:
      * area — raw pixel area. The player is usually the biggest person.
      * centrality — gaussian on distance-from-image-center, σ=0.5 of the
        image diagonal. A ghost at the edge of frame is downweighted.
      * aspect_score — gaussian on bbox aspect ratio (h/w) centered at 2.0
        with σ=1.0. Tennis players are ~1.5–3× taller than wide; fences,
        benches, horizontal banners score near zero.
      * confidence — tiebreak when all else is equal.

    Returns a positive scalar; caller picks the max.
    """
    x1, y1, x2, y2, conf = bbox
    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)
    area = bw * bh

    img_cx = img_w / 2.0
    img_cy = img_h / 2.0
    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0
    diag = math.sqrt(img_w * img_w + img_h * img_h)
    norm_dist = math.hypot(box_cx - img_cx, box_cy - img_cy) / max(1e-6, diag)
    centrality = math.exp(-((norm_dist / 0.5) ** 2))

    aspect = bh / bw
    aspect_score = math.exp(-(((aspect - 2.0) / 1.0) ** 2))

    return area * centrality * aspect_score * max(conf, 1e-3)


def _ema_update(prev: BBox, new: BBox, alpha: float) -> BBox:
    """Exponential-moving-average between two bboxes. alpha weights the
    new detection; (1-alpha) weights the running reference."""
    return (
        alpha * new[0] + (1 - alpha) * prev[0],
        alpha * new[1] + (1 - alpha) * prev[1],
        alpha * new[2] + (1 - alpha) * prev[2],
        alpha * new[3] + (1 - alpha) * prev[3],
    )


class PersonTracker:
    """Single-person bbox tracker. Non-Kalman by design: a person doesn't
    vanish for many frames and doesn't need trajectory prediction — the
    failure mode is YOLO picking the wrong person, not the right person
    being temporarily missed.

    Usage:
      tracker = PersonTracker()
      for frame in video:
          candidates = yolo.detect_persons(frame)  # list of xyxy+conf
          tracked = tracker.update(candidates, w, h)
          if tracked is not None:
              run_pose_on_bbox(frame, tracked)

    Tunables:
      max_coast_frames — after this many consecutive frames with no
        in-gate YOLO detection, the tracker resets and re-runs the cold
        start heuristic on the next frame. Set high enough to absorb a
        brief occlusion (player behind net post) but low enough that a
        real scene change triggers a re-lock.
      iou_gate — minimum IoU a candidate needs against the current
        reference bbox to count as "the same person this frame." 0.3 is
        standard ByteTrack-ish. Lower = sticky; higher = brittle.
      ema_alpha — how much weight the new detection gets when updating
        the running reference. 0.7 follows fast player motion while
        still suppressing per-frame YOLO jitter.
    """

    def __init__(
        self,
        max_coast_frames: int = 5,
        iou_gate: float = 0.3,
        ema_alpha: float = 0.7,
    ) -> None:
        self._current_bbox: Optional[BBox] = None
        self._last_confidence: float = 0.0
        self._frames_since_detection: int = 0
        self.max_coast_frames = max_coast_frames
        self.iou_gate = iou_gate
        self.ema_alpha = ema_alpha

    @property
    def is_locked(self) -> bool:
        return self._current_bbox is not None

    def reset(self) -> None:
        self._current_bbox = None
        self._last_confidence = 0.0
        self._frames_since_detection = 0

    def update(
        self,
        candidates: list[BBoxWithConf],
        img_w: int,
        img_h: int,
    ) -> Optional[BBoxWithConf]:
        """Fold one frame's candidates into the tracker and return the
        canonical tracked bbox for this frame (with confidence).

        Cold-start path (no current lock): score candidates with the
        area × centrality × aspect heuristic and pick the best. The
        tracker arms on this pick; subsequent frames use IoU association.

        Associated path: find the candidate with highest IoU to the
        current reference bbox. If it clears iou_gate, EMA-blend it into
        the reference and return the new smoothed bbox. Otherwise treat
        as a missed detection: return the current reference (coast).

        Coasting path: after max_coast_frames with no in-gate detection,
        the tracker releases the lock and the next call cold-starts.

        Returns None only when no lock exists AND no cold-start candidate
        was provided. Once locked, always returns the current best
        estimate (real detection or coasted).
        """
        if self._current_bbox is None:
            # Cold start.
            if not candidates:
                return None
            best = max(
                candidates,
                key=lambda c: _cold_start_score(c, img_w, img_h),
            )
            self._current_bbox = (best[0], best[1], best[2], best[3])
            self._last_confidence = best[4]
            self._frames_since_detection = 0
            return best

        # Warm path: associate by IoU against the current reference.
        if candidates:
            ious = [iou(self._current_bbox, (c[0], c[1], c[2], c[3])) for c in candidates]
            best_idx = max(range(len(candidates)), key=lambda i: ious[i])
            if ious[best_idx] >= self.iou_gate:
                match = candidates[best_idx]
                match_bbox: BBox = (match[0], match[1], match[2], match[3])
                smoothed = _ema_update(self._current_bbox, match_bbox, self.ema_alpha)
                self._current_bbox = smoothed
                self._last_confidence = match[4]
                self._frames_since_detection = 0
                return (
                    smoothed[0],
                    smoothed[1],
                    smoothed[2],
                    smoothed[3],
                    match[4],
                )

        # No detection associated: coast on the previous reference.
        self._frames_since_detection += 1
        if self._frames_since_detection > self.max_coast_frames:
            self.reset()
            # After reset, try cold start with whatever candidates we have
            # so the next frame isn't strictly blank.
            return self.update(candidates, img_w, img_h)

        x1, y1, x2, y2 = self._current_bbox
        return (x1, y1, x2, y2, self._last_confidence)


__all__ = ["PersonTracker", "iou"]
