"""Validate RTMPose-m against MediaPipe Heavy on the three named test clips.

Per docs/pose-research.md §7 the acceptance criteria are observable/
measurable on existing pro clips. This script runs both backends on the
same clip, computes the same metrics the research doc specifies, and
prints a side-by-side summary:

  - Bone-length stability: std/median ratio for shoulder->elbow and
    elbow->wrist per side, across frames where both endpoints have
    visibility >= VIS_THRESHOLD. Lower = tracker is internally
    consistent (a stable-length limb shouldn't stretch frame-to-frame).
  - Frame drop rate: fraction of processed frames where the elbow or
    wrist keypoints are below VIS_THRESHOLD (the render cutoff in
    lib/poseSmoothing.ts). Lower = fewer gaps in the overlay.

Exit 0 if RTMPose wins (is lower) on at least 2 of 3 clips for at least
one of the two metric families; exit 1 otherwise.

Running the backends
====================
We don't flip the POSE_BACKEND module constant in-process because the
extract modules cache heavy ONNX sessions at module scope, so re-import
wouldn't give a clean reset. Instead we run extract_clip_keypoints.py
as a subprocess with POSE_BACKEND set per-invocation. This mirrors how
Railway dispatches backends in production.

Usage
=====
From /Users/neil/Desktop/Projects/tennis-analyzer/railway-service:

  python3 tests/validate_rtmpose_vs_mediapipe.py

By default, runs against the three clips named in docs/pose-research.md
§7. Override with --clips to pass explicit paths:

  python3 tests/validate_rtmpose_vs_mediapipe.py \
    --clips /path/a.mp4 /path/b.mp4 /path/c.mp4

If a test clip isn't present (public/pro-videos/ is gitignored), the
script prints `BLOCKED: missing clip` and exits 2 without fabricating
numbers.

Environment requirements
========================
python3 with: mediapipe, rtmlib, onnxruntime, ultralytics, opencv-python,
numpy. The first rtmpose run downloads ~25MB of ONNX weights from the
openmmlab CDN to ~/.cache/rtmlib.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

RAILWAY_DIR = Path(__file__).parent.parent
REPO_ROOT = RAILWAY_DIR.parent
PRO_VIDEOS_DIR = REPO_ROOT / "public" / "pro-videos"

# Visibility cutoff matches the render-side threshold in the codebase
# (lib/poseSmoothing.ts uses 0.6 as the "confident" cutoff).
VIS_THRESHOLD = 0.6

# Shoulder/elbow/wrist BlazePose ids (both backends emit this layout).
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

# The named clips from docs/pose-research.md §7. Defaults resolve
# against public/pro-videos/ at the repo root.
DEFAULT_CLIPS = [
    ("small-in-frame, back-facing",
     PRO_VIDEOS_DIR / "carlos_alcaraz_forehand_behind_1776374463938.mp4"),
    ("fast joint (forehand contact)",
     PRO_VIDEOS_DIR / "jannik_sinner_forehand_side_1776381824347.mp4"),
    ("back-facing oblique",
     PRO_VIDEOS_DIR / "novak_djokovic_backhand_side_1776383254038.mp4"),
]


@dataclass
class BackendResult:
    """Per-clip metrics for one backend."""

    backend: str
    clip: str
    frame_count: int
    # Bone-length stability (std / median); lower is better. One entry per
    # side+bone combo we could measure; None when < 5 usable frames.
    bone_stability: dict[str, Optional[float]]
    # Fraction of frames (0..1) where elbow or wrist is below threshold
    # on either side. Lower is better.
    drop_pct: float
    # Wall-clock seconds to extract.
    elapsed_s: float


def _pair_distance(landmarks: list[dict], id_a: int, id_b: int) -> Optional[float]:
    """Euclidean distance between two landmarks in normalized coords.

    Returns None if either endpoint is below VIS_THRESHOLD. Both
    backends emit normalized [0,1] coords keyed by BlazePose id, so
    the distance is directly comparable across backends.
    """
    if id_a >= len(landmarks) or id_b >= len(landmarks):
        return None
    a, b = landmarks[id_a], landmarks[id_b]
    if a.get("visibility", 0.0) < VIS_THRESHOLD:
        return None
    if b.get("visibility", 0.0) < VIS_THRESHOLD:
        return None
    dx, dy = a["x"] - b["x"], a["y"] - b["y"]
    return math.hypot(dx, dy)


def _stability(values: list[float]) -> Optional[float]:
    """std / median. Lower = more stable. None if < 5 samples or median <= 0."""
    if len(values) < 5:
        return None
    med = statistics.median(values)
    if med <= 0:
        return None
    std = statistics.pstdev(values)
    return std / med


def compute_metrics(
    backend: str, clip: str, frames: list[dict]
) -> tuple[dict[str, Optional[float]], float]:
    """Return (bone_stability_per_bone, drop_pct)."""
    left_se: list[float] = []   # left shoulder->elbow
    right_se: list[float] = []
    left_ew: list[float] = []   # left elbow->wrist
    right_ew: list[float] = []
    drops = 0

    for f in frames:
        lms = f.get("landmarks", [])
        if not lms:
            drops += 1
            continue

        # A frame is "dropped" if any of the upper-arm endpoints we care
        # about is below threshold. This matches the user's render-time
        # visual experience: the wrist/elbow simply disappears.
        def vis(idx: int) -> float:
            return lms[idx].get("visibility", 0.0) if idx < len(lms) else 0.0

        if (
            vis(LEFT_ELBOW) < VIS_THRESHOLD
            or vis(RIGHT_ELBOW) < VIS_THRESHOLD
            or vis(LEFT_WRIST) < VIS_THRESHOLD
            or vis(RIGHT_WRIST) < VIS_THRESHOLD
        ):
            drops += 1

        for a, b, bucket in (
            (LEFT_SHOULDER, LEFT_ELBOW, left_se),
            (RIGHT_SHOULDER, RIGHT_ELBOW, right_se),
            (LEFT_ELBOW, LEFT_WRIST, left_ew),
            (RIGHT_ELBOW, RIGHT_WRIST, right_ew),
        ):
            d = _pair_distance(lms, a, b)
            if d is not None:
                bucket.append(d)

    drop_pct = drops / len(frames) if frames else 1.0
    stability = {
        "left_shoulder_elbow": _stability(left_se),
        "right_shoulder_elbow": _stability(right_se),
        "left_elbow_wrist": _stability(left_ew),
        "right_elbow_wrist": _stability(right_ew),
    }
    return stability, drop_pct


def run_backend(backend: str, clip_path: Path) -> dict:
    """Invoke extract_clip_keypoints.py as a subprocess with POSE_BACKEND set.

    Returns the parsed JSON the script prints to stdout. Raises on
    subprocess failure so the caller can mark the clip blocked.
    """
    env = os.environ.copy()
    env["POSE_BACKEND"] = backend
    cmd = [
        sys.executable,
        str(RAILWAY_DIR / "extract_clip_keypoints.py"),
        str(clip_path),
    ]
    # Use a generous timeout: per the research doc, 5 min per 30s clip
    # is the budget. Give each backend 10 minutes to avoid false timeouts
    # on first-run model downloads.
    proc = subprocess.run(
        cmd, env=env, cwd=str(RAILWAY_DIR),
        capture_output=True, timeout=600, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} extract exited {proc.returncode}: "
            f"stderr={proc.stderr[:500]}"
        )
    # stdout must be a single JSON blob. Strip any trailing whitespace.
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError(f"{backend} extract produced no stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{backend} extract JSON parse failed: {e}; "
            f"first 300 chars: {stdout[:300]}"
        )


def evaluate_clip(label: str, clip_path: Path) -> tuple[Optional[BackendResult], Optional[BackendResult]]:
    """Extract keypoints with both backends; return (mp, rt) results."""
    print(f"\n[{label}] {clip_path.name}", flush=True)

    if not clip_path.exists():
        print(f"  BLOCKED: clip not found at {clip_path}", flush=True)
        return None, None

    results: list[Optional[BackendResult]] = []
    for backend in ("mediapipe", "rtmpose"):
        print(f"  running {backend}...", flush=True)
        t0 = time.time()
        try:
            payload = run_backend(backend, clip_path)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            results.append(None)
            continue
        elapsed = time.time() - t0
        frames = payload.get("frames", [])
        if "error" in payload:
            print(f"    backend reported error: {payload['error']}", flush=True)
            results.append(None)
            continue
        stability, drop_pct = compute_metrics(backend, clip_path.name, frames)
        results.append(BackendResult(
            backend=backend,
            clip=clip_path.name,
            frame_count=len(frames),
            bone_stability=stability,
            drop_pct=drop_pct,
            elapsed_s=elapsed,
        ))
        # Compact per-run summary
        stable_vals = [v for v in stability.values() if v is not None]
        avg_stab = statistics.mean(stable_vals) if stable_vals else float("nan")
        print(
            f"    {backend}: frames={len(frames)} drop_pct={drop_pct:.1%} "
            f"avg_bone_stability={avg_stab:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return results[0], results[1]


def _fmt_stability(s: dict[str, Optional[float]]) -> str:
    vals = [v for v in s.values() if v is not None]
    if not vals:
        return "n/a"
    return f"{statistics.mean(vals):.4f}"


def print_summary_table(pairs: list[tuple[str, Optional[BackendResult], Optional[BackendResult]]]) -> int:
    """Print a side-by-side summary. Return the verdict exit code.

    Exit 0 when RTMPose beats MediaPipe on >= 2 of 3 clips for at least
    one of {drop rate, bone stability}. Exit 1 when the signal is flat
    or MediaPipe wins. Exit 2 when we couldn't evaluate enough clips to
    judge.
    """
    print("\n" + "=" * 92)
    print(f"{'clip':<50} {'mp stab':>10} {'rt stab':>10} {'mp drop':>10} {'rt drop':>10}")
    print("-" * 92)

    rt_better_stab = 0
    rt_better_drop = 0
    evaluated = 0
    for clip, mp, rt in pairs:
        if mp is None or rt is None:
            print(f"{clip:<50} {'--':>10} {'--':>10} {'--':>10} {'--':>10}   (blocked)")
            continue
        evaluated += 1
        mp_stab = _fmt_stability(mp.bone_stability)
        rt_stab = _fmt_stability(rt.bone_stability)
        mp_drop = f"{mp.drop_pct:.1%}"
        rt_drop = f"{rt.drop_pct:.1%}"
        print(f"{clip:<50} {mp_stab:>10} {rt_stab:>10} {mp_drop:>10} {rt_drop:>10}")

        # Only count "better" when both sides have a value.
        mp_stab_vals = [v for v in mp.bone_stability.values() if v is not None]
        rt_stab_vals = [v for v in rt.bone_stability.values() if v is not None]
        if mp_stab_vals and rt_stab_vals:
            if statistics.mean(rt_stab_vals) < statistics.mean(mp_stab_vals):
                rt_better_stab += 1
        if rt.drop_pct < mp.drop_pct:
            rt_better_drop += 1

    print("=" * 92)
    print(
        f"RTMPose better on {rt_better_stab}/{evaluated} clips for stability, "
        f"{rt_better_drop}/{evaluated} clips for drop rate"
    )

    if evaluated < 2:
        print("\nVERDICT: BLOCKED -- fewer than 2 clips evaluated; cannot judge.")
        return 2
    if rt_better_stab >= 2 or rt_better_drop >= 2:
        print("\nVERDICT: RTMPose wins on >=2 clips for at least one metric. SHIP.")
        return 0
    print("\nVERDICT: RTMPose does not beat MediaPipe on enough clips. NO-SHIP.")
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--clips",
        nargs="+",
        metavar="PATH",
        help="Override the three default test clips with explicit paths.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.clips:
        clips = [(f"user-supplied #{i + 1}", Path(p)) for i, p in enumerate(args.clips)]
    else:
        clips = DEFAULT_CLIPS

    pairs: list[tuple[str, Optional[BackendResult], Optional[BackendResult]]] = []
    for label, clip_path in clips:
        mp, rt = evaluate_clip(label, clip_path)
        pairs.append((clip_path.name, mp, rt))

    return print_summary_table(pairs)


if __name__ == "__main__":
    sys.exit(main())
