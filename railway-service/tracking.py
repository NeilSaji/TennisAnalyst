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
from threading import RLock
from typing import Optional

import numpy as np


BBox = tuple[float, float, float, float]
BBoxWithConf = tuple[float, float, float, float, float]
PointDetection = dict  # {"x": float, "y": float, "confidence": float} — normalized


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
        # The Railway extraction loop now submits multiple
        # infer_pose_for_frame calls to a thread pool concurrently. Each
        # call eventually hits self.update(); without a mutex two
        # concurrent updates could race on _current_bbox / _last_*. The
        # critical section is microseconds long (no I/O, no inference)
        # so the lock has near-zero impact on overall throughput.
        # RLock so the recursive call after a coast-driven reset() does
        # not self-deadlock.
        self._lock = RLock()

    @property
    def is_locked(self) -> bool:
        return self._current_bbox is not None

    def reset(self) -> None:
        with self._lock:
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
        with self._lock:
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
                # so the next frame isn't strictly blank. RLock allows the
                # nested update + reset calls without self-deadlock.
                return self.update(candidates, img_w, img_h)

            x1, y1, x2, y2 = self._current_bbox
            return (x1, y1, x2, y2, self._last_confidence)


class RacketTracker:
    """Kalman-filter single-point tracker for the racket center.

    The racket disappears from YOLO regularly — motion blur at the top of
    a forehand, the racket head leaving the frame during a serve toss,
    the hand briefly occluding the strings. A stateless gate (show when
    conf >= 0.3, else nothing) produces a trail full of gaps. This
    tracker fills short gaps with a constant-velocity prediction and
    only gives up (returns None) after several consecutive miss frames.

    State layout: [cx, cy, vx, vy] in normalized image coordinates, with
    velocity in units of (normalized coord) per second. Working in
    normalized space means the tuning is resolution-independent — a
    1080p and a 720p clip both see a forehand contact as "~0.8
    normalized units per second."

    Why a Kalman filter and not an EMA here (but not for PersonTracker):
    racket motion is ballistic across frames (predictable from velocity),
    and we need to interpolate through miss frames — both are exactly
    what a Kalman filter is for. PersonTracker doesn't need it because a
    person doesn't vanish unpredictably between sampled frames.

    Tunables:
      sigma_a — process-noise acceleration std dev. Must be large enough
        to follow a racket accelerating 0 → 1500 px/s in ~100ms during a
        forehand (≈ 8 normalized/s² on a 1080p frame). Too small = the
        filter lags during contact; too large = noisy predictions during
        the backswing's smooth phase.
      sigma_m — measurement-noise std dev. ~0.01 matches the observed
        frame-to-frame jitter on YOLO bbox centers. Bigger = smoother
        but laggier; smaller = trusts YOLO more.
      association_radius — max allowed distance between the filter's
        prediction and a new YOLO detection for the detection to be
        accepted (otherwise treated as a miss). 0.15 normalized units
        covers the gap a fast wrist can traverse between sampled frames.
      max_coast_frames — after this many consecutive missed detections,
        the filter resets. Short enough to re-lock on the right racket
        after a scene transition; long enough to coast through a swing's
        motion-blur plateau.
      conf_decay / conf_floor — coasted frames don't carry real YOLO
        confidence; we emit `prev_conf × conf_decay` on each coast,
        floored so the short fill is still above the client tracer's
        threshold. Long coasts eventually fall below the floor and
        return None, at which point the forearm fallback takes over.
    """

    # State indices
    _CX, _CY, _VX, _VY = 0, 1, 2, 3

    def __init__(
        self,
        sigma_a: float = 8.0,
        sigma_m: float = 0.01,
        association_radius: float = 0.15,
        max_coast_frames: int = 4,
        conf_decay: float = 0.9,
        # Dropped from 0.31 → 0.10 after the crop-to-person path went
        # live. Cropped YOLO legitimately emits detections as low as
        # 0.15 (false-positive rate is ~zero in a person-cropped frame,
        # so the threshold on the detector side was lowered too). Keeping
        # the old 0.31 floor killed those detections the moment they
        # took one coast step: 0.15 × 0.9 = 0.135 < 0.31 triggered a
        # reset every time. Low floor lets low-conf detections survive
        # coasting; max_coast_frames still bounds how long we fake it.
        conf_floor: float = 0.10,
    ) -> None:
        self._x: Optional[np.ndarray] = None  # (4,)
        self._P: Optional[np.ndarray] = None  # (4, 4)
        self._last_ts_ms: Optional[float] = None
        self._last_confidence: float = 0.0
        self._frames_since_detection: int = 0

        self.sigma_a = sigma_a
        self.sigma_m = sigma_m
        self.association_radius = association_radius
        self.max_coast_frames = max_coast_frames
        self.conf_decay = conf_decay
        self.conf_floor = conf_floor

    @property
    def is_locked(self) -> bool:
        return self._x is not None

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._last_ts_ms = None
        self._last_confidence = 0.0
        self._frames_since_detection = 0

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def _Q(self, dt: float) -> np.ndarray:
        """Discrete white-noise acceleration process-noise matrix.
        Standard constant-acceleration model: assume white-noise
        acceleration with std dev sigma_a on each axis independently.
        """
        sa2 = self.sigma_a * self.sigma_a
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return sa2 * np.array(
            [
                [dt4 / 4.0, 0.0,       dt3 / 2.0, 0.0      ],
                [0.0,       dt4 / 4.0, 0.0,       dt3 / 2.0],
                [dt3 / 2.0, 0.0,       dt2,       0.0      ],
                [0.0,       dt3 / 2.0, 0.0,       dt2      ],
            ]
        )

    @staticmethod
    def _H() -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def _R(self) -> np.ndarray:
        sm2 = self.sigma_m * self.sigma_m
        return np.array([[sm2, 0.0], [0.0, sm2]])

    def update(
        self,
        detection: Optional[PointDetection],
        timestamp_ms: float,
    ) -> Optional[PointDetection]:
        """Advance the tracker by one frame. Returns the emitted racket
        position (YOLO measurement, smoothed) or a prediction-only coast,
        or None if the filter is cold and this frame has no detection
        (or has coasted past max_coast_frames).

        timestamp_ms is the monotonically-increasing per-frame sample
        time. dt is computed internally; the caller doesn't guess.
        """
        # Cold start: wait for the first detection to arm the filter.
        if self._x is None:
            if detection is None:
                return None
            self._x = np.array([detection["x"], detection["y"], 0.0, 0.0])
            # Initial uncertainty: position ≈ sigma_m, velocity unknown
            # so set a large variance for vx/vy.
            self._P = np.diag([self.sigma_m ** 2, self.sigma_m ** 2, 1.0, 1.0])
            self._last_ts_ms = timestamp_ms
            self._last_confidence = float(detection["confidence"])
            self._frames_since_detection = 0
            return {
                "x": round(detection["x"], 4),
                "y": round(detection["y"], 4),
                "confidence": round(self._last_confidence, 3),
            }

        # Warm path: predict forward by dt, then maybe update.
        assert self._P is not None and self._last_ts_ms is not None
        dt_ms = timestamp_ms - self._last_ts_ms
        # Guard against out-of-order or duplicate timestamps from the
        # caller (rare but possible on variable-fps inputs). A zero or
        # negative dt makes the Kalman predict a no-op, which is safe.
        dt = max(0.0, dt_ms / 1000.0)

        F = self._F(dt)
        Q = self._Q(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        # Association: compare the detection to the prediction. The gate
        # inflates linearly with coast time so a re-acquisition after a
        # motion-blur gap isn't rejected for being past where the
        # constant-velocity momentum said the racket should be. During a
        # swing contact the racket decelerates sharply — the filter
        # doesn't know that, so its prediction can be ~0.08 normalized
        # units off after 3 coasted frames. Inflating the gate by 50%
        # per coast frame covers that drift.
        accepted = False
        effective_gate = self.association_radius * (
            1.0 + 0.5 * self._frames_since_detection
        )
        if detection is not None:
            pred_x = float(self._x[self._CX])
            pred_y = float(self._x[self._CY])
            dx = float(detection["x"]) - pred_x
            dy = float(detection["y"]) - pred_y
            if math.hypot(dx, dy) <= effective_gate:
                # Kalman update step
                H = self._H()
                R = self._R()
                z = np.array([detection["x"], detection["y"]])
                y = z - H @ self._x
                S = H @ self._P @ H.T + R
                K = self._P @ H.T @ np.linalg.inv(S)
                self._x = self._x + K @ y
                self._P = (np.eye(4) - K @ H) @ self._P
                self._last_confidence = float(detection["confidence"])
                self._frames_since_detection = 0
                self._last_ts_ms = timestamp_ms
                accepted = True

        if not accepted:
            # Miss: decay confidence, bump coast counter.
            self._frames_since_detection += 1
            self._last_confidence *= self.conf_decay
            self._last_ts_ms = timestamp_ms
            if (
                self._frames_since_detection > self.max_coast_frames
                or self._last_confidence < self.conf_floor
            ):
                self.reset()
                return None

        return {
            "x": round(float(np.clip(self._x[self._CX], 0.0, 1.0)), 4),
            "y": round(float(np.clip(self._x[self._CY], 0.0, 1.0)), 4),
            "confidence": round(self._last_confidence, 3),
        }


__all__ = ["PersonTracker", "RacketTracker", "iou"]
