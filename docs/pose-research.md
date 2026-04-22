# Pose Estimation Upgrade — Research & Decision Log

Author: agent + Neil — 2026-04-21
Status: research complete, implementation in progress.

## 1. Problem statement

Current joint tracing runs **MediaPipe Pose Heavy** in the browser
(`lib/mediapipe.ts`, `lib/poseExtraction.ts`) and on Railway
(`railway-service/main.py`). Even with zero-phase One Euro smoothing
(`lib/poseSmoothing.ts`), bone-length plausibility filtering, a 0.6
visibility cutoff at render time, and nearest-neighbor frame lookup, three
failure modes remain visible to users:

1. **Fast-moving joints** (elbow / wrist through swing contact): motion
   blur on pro-speed footage defeats MediaPipe's detector and the wrist
   lands several inches off the actual joint.
2. **Small-in-frame subjects** (player at ~15% of frame height): MediaPipe
   internally downscales to 256×256, so a small player's pixel resolution
   collapses below useful and the landmarks scatter.
3. **Back-facing / oblique camera angles**: MediaPipe BlazePose is trained
   primarily on front- and side-facing humans; back-of-court tennis broadcast
   angles produce systematically wrong joint placements.

Constraints that frame the solution:

- Total extraction latency ≤ **5 min per 30-s clip**.
- Browser model ≤ ~80 MB **OR** server-side via Railway (CPU only — Railway
  Hobby/Pro tiers have no GPUs).
- Schema downstream consumers (`VideoCanvas`, `PoseRenderer`, `jointAngles`,
  `poseSmoothing`, `analyze` route) all key off MediaPipe BlazePose-33 ids.
  Any new model must either match that scheme, or land behind a translation
  shim that emits the same shape.
- The user has been explicit: **uncertain joints must drop out**, never
  predict / interpolate. `visibility=0` is the contract; render hides
  anything below the 0.6 cutoff.

## 2. Schema constraint (the dominant one)

`KeypointsJson.frames[].landmarks` is a 33-entry array indexed by MediaPipe
BlazePose ids. Downstream code consumes a small subset of those ids:

| Use site | Ids consumed |
|---|---|
| `lib/jointAngles.ts` (joint-angle calc) | 11–16, 19/20 (index finger, wrist flexion), 23–28 |
| `lib/poseSmoothing.ts` (bone-length filter) | 11–16 |
| `components/PoseRenderer.ts` | renders all 33 ids it gets, but the joint-group toggles only show 11–16, 23–28 |
| `components/SwingPathTracer.ts` | wrists (15/16) + racket head |
| `railway-service/main.py` (`compute_joint_angles_from_dicts`) | same as TS |

Net effect: **only 16 of the 33 BlazePose ids carry signal** the app uses
(11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28). Heel and
foot-index landmarks (29–32), face landmarks (0–10 except nose 0 for
angle context), and the pinky/thumb hand landmarks (17/18, 21/22) are
present in the JSON but never read.

Every SOTA body-only model on the market emits **COCO-17**, not BlazePose-33:

| BlazePose id we read | Name | COCO-17 equivalent | Notes |
|---|---|---|---|
| 0 | nose | 0 (nose) | direct map |
| 11/12 | left/right_shoulder | 5/6 | direct map |
| 13/14 | left/right_elbow | 7/8 | direct map |
| 15/16 | left/right_wrist | 9/10 | direct map |
| 19/20 | left/right_index_finger | — | **not in COCO-17** |
| 23/24 | left/right_hip | 11/12 | direct map |
| 25/26 | left/right_knee | 13/14 | direct map |
| 27/28 | left/right_ankle | 15/16 | direct map |

The *only* gap is the index-finger landmark used for wrist flexion
(`right_wrist` / `left_wrist` joint-angle in `jointAngles.ts`). Two
options:

- **Option A — body-only model (RTMPose-m / RTMPose-l), drop wrist flexion.**
  Emit 33-entry arrays where ids 17/18, 19/20, 21/22, 29–32 carry
  `visibility=0`. `computeJointAngles` will see no index-finger landmarks
  and skip the wrist-flexion calc — `right_wrist`/`left_wrist` becomes
  `undefined` in `JointAngles`, which is already a documented possibility
  in the type (`right_wrist?: number`). Schema bump to v3 so historical
  cached pro keypoints (with the angle present) are still readable.

- **Option B — wholebody model (RTMW-l), keep wrist flexion.**
  RTMW emits 133 keypoints including 21 per hand. Map COCO-WholeBody
  hand index-finger MCP/PIP/DIP/TIP back to BlazePose id 19/20.
  Heavier model (~80 MB ONNX), slower (still real-time on CPU), higher
  setup cost.

**Decision: A.** Wrist-flexion is a "nice to have" in the analyzer (it's
not surfaced in the LLM-coaching summary or the metrics table — `grep`
the codebase confirms it's only computed, never displayed). Trading it
away to land a model that's strictly better on the three primary failure
modes is the right call. The schema_version bump preserves the path for
B later.

## 3. Browser vs server

Currently a split: pros run server-side via Railway `/extract`, user
uploads run browser-side via `lib/poseExtraction.ts`.

**Decision: server-side for both.** Justification:

- The Railway Python image already has `ultralytics`, `onnxruntime`,
  `opencv-python-headless`, and `numpy` pinned. RTMPose via `rtmlib`
  ([Tau-J/rtmlib](https://github.com/Tau-J/rtmlib)) needs only those plus
  itself. `racket_detector.py` already wires `ultralytics → onnxruntime`
  end-to-end, including the YOLO letterbox pipeline I'd otherwise have to
  rebuild.
- `main.py` already has `POSE_BACKEND = os.environ.get("POSE_BACKEND",
  "mediapipe")` as a named extension seam, with a `rtmpose` placeholder
  branch that throws `NotImplementedError`. The seam is *literally* waiting
  for this swap.
- Browser RTMPose via ONNX Web is possible but: (a) the model must be
  served from a CDN ≤ 80 MB, (b) WASM inference is ~10× slower than native
  ORT on the same CPU, (c) we already burned a coordinate-transform bug on
  the browser path, (d) maintaining two pose backends doubles the surface.
- 5-min budget on Railway = 300 s for ~900 frames at 30 fps =
  ~333 ms/frame. RTMPose-m on CPU ONNXRuntime hits ~90 FPS (≈11 ms/frame)
  for the pose model alone; even adding the YOLO person crop pass and
  video I/O leaves an order-of-magnitude headroom.
- User-side wiring: a new `/api/extract` route calls Railway `/extract`
  with the user's session_id and blob_url. Today's `/api/sessions` POST
  takes pre-extracted keypoints from the browser. We replace that path:
  client uploads → API kicks off Railway extract → Railway updates the
  Supabase row when done → client polls (or uses the existing
  status-polling already wired up for video processing). The browser
  extractor is removed but the `lib/mediapipe.ts` / `lib/poseExtraction.ts`
  files stay (kept for the local dev / fallback case and so `usePoseExtractor`
  callers don't break in tests).

The dev experience hit (need Railway URL + key to extract) is mitigated
by keeping the existing browser path live as a fallback when
`RAILWAY_SERVICE_URL` is unset — extract path is chosen at runtime in the
hook, not gated at build time.

**Scope this PR.** Both paths flip to RTMPose. Server side first (the
`POSE_BACKEND=rtmpose` branch in `extract_keypoints_from_video` plus
the matching path in `extract_clip_keypoints.py`), then a new
`/api/extract` route delegates user uploads to Railway and the
`UploadZone` flow polls the resulting session for completion instead of
running MediaPipe in the browser. The browser MediaPipe path stays in
the codebase as a `RAILWAY_SERVICE_URL`-unset fallback for local dev
without a Railway URL.

## 4. Methods evaluated

For each candidate the relevant axes are: keypoint format, accuracy on
COCO val 2017 (body), parameter count, license, CPU inference cost, and
ecosystem fit with our existing Railway image.

### MediaPipe Pose Heavy (baseline)

- 33 BlazePose keypoints (including face / hands / feet)
- License: Apache-2.0
- COCO val AP not directly reported by Google. The Heavy variant is the
  largest of the three; the public benchmark Google publishes is on a
  custom in-house "yoga + dance + HIIT" set with PCK@0.2.
- ~30 MB, runs on CPU/GPU/WASM. Runs in browser via `tasks-vision`.
- Failure modes documented by user; matches our experience.

### RTMPose-m (chosen)

- COCO-17 keypoints (17 body)
- COCO val AP: **75.8** ([RTMPose paper](https://arxiv.org/abs/2303.07399), [MMPose RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose))
- 13.6M params, 1.93 GFLOPs at 256×192
- License: Apache-2.0
- CPU FPS: **90+** on i7-11700 with ONNXRuntime
  ([OpenMMLab benchmark](https://openmmlab.medium.com/rtmpose-the-all-in-one-real-time-pose-estimation-solution-for-application-and-research-6404f17cd52f))
- Top-down (needs an external person detector — pair with YOLO-person)
- **Critically: SimCC head**, not heatmap. Outputs per-axis discrete
  probability distributions, which (a) gives sub-pixel coordinate
  resolution without heatmap upsampling, (b) is more robust to occlusion
  / blur than heatmap argmax, (c) makes it cheap to read out a confidence
  score per keypoint
  ([SimCC ECCV'22](https://arxiv.org/abs/2107.03332))
- Strong sports validation: Pose2Sim
  ([perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim))
  switched from OpenPose / MediaPipe to RTMPose as the **default** 2D
  backend for biomechanics analysis, citing accuracy.

### RTMPose-l-384

- Same architecture, larger 27.7M params, 384×288 input
- COCO val AP: 78.3
- ~50 FPS CPU
- Better accuracy in exchange for ~half the throughput. Reasonable
  fallback if -m disappoints; live behind the same code path with a
  config swap.

### RTMW-l (whole-body)

- 133 keypoints (17 body + 6 feet + 68 face + 42 hands)
- COCO-WholeBody val mAP: 70.1 (256×192) / 73.8 (384×288)
- Apache-2.0
- ~60 MB ONNX
- The "Option B" candidate from §2. Defer.

### YOLOv11-Pose (m / x)

- COCO-17 keypoints, **single-stage** (detector + pose in one pass)
- COCO val mAP@50-95: 68.8 (m) / 71.6 (x)
  ([Ultralytics docs](https://docs.ultralytics.com/tasks/pose/))
- License: **AGPL-3.0** (Ultralytics) — copyleft, the Vercel deploy
  serving Next.js doesn't trigger AGPL since the model never ships
  client-side, but it would pollute the Railway service. RTMPose's
  Apache-2.0 is strictly cleaner.
- CPU latency (ONNX): YOLO11m-pose **218 ms/frame** on the doc's
  reference CPU vs RTMPose-m **~11 ms/frame**. ~20× slower.
- Single-stage = no top-down crop benefit for small subjects (the model
  must localize the player from the full frame's pixels itself).
  Wrong tool for our small-subject failure mode.

### ViTPose-H

- COCO-17, top-down
- COCO val AP: 79.1 — the highest of any candidate
- 632 M params; ~50 ms latency on **A100 GPU**, no realistic CPU number
- ~2.5 GB
- Can't deploy on a CPU Railway tier within our latency budget.

### HRNet-W48

- 75.1 AP, 63 M params, ~158 fps on A100. Older, eclipsed by RTMPose at
  every metric we care about.

### Sapiens (Meta)

- 0.3B / 0.6B / 1B / 2B variants, 17 keypoints (also COCO-WholeBody
  variants)
- License: **CC BY-NC 4.0 — non-commercial only**
  ([Sapiens LICENSE](https://github.com/facebookresearch/sapiens/blob/main/LICENSE))
- Disqualified for our use case regardless of accuracy.

### Comparison

| Model | Keypoints | COCO AP | CPU FPS | Params | License | Verdict |
|---|---|---|---|---|---|---|
| MediaPipe Heavy (baseline) | 33 BlazePose | n/a | ~30 (browser GPU) | ~30M | Apache-2.0 | Status quo, has documented failure modes |
| **RTMPose-m** | 17 COCO | **75.8** | **90+** | 13.6M | Apache-2.0 | **CHOICE** — best accuracy/speed/license tri-fecta |
| RTMPose-l-384 | 17 COCO | 78.3 | ~50 | 27.7M | Apache-2.0 | Drop-in fallback for higher accuracy |
| RTMW-l | 133 wholebody | 70.1 (wholebody) | ~50 | ~60M | Apache-2.0 | Future Option B for hand/wrist |
| YOLOv11m-pose | 17 COCO | 68.8 | ~5 | 21.5M | AGPL-3.0 | Slower on CPU, copyleft, single-stage |
| ViTPose-H | 17 COCO | 79.1 | n/a (GPU only) | 632M | Apache-2.0 | No CPU path |
| Sapiens-1B | 17 COCO | ~80 | n/a | 1000M | CC BY-NC-4.0 | NonCommercial — disqualified |

## 5. Recommended architecture

**Top-down two-stage pipeline on Railway, no GPU required:**

1. Decode video with `cv2.VideoCapture` (already in main.py).
2. **Stage 1 — person detection:** YOLO11-n (already on disk in
   `railway-service/models/yolo11n.onnx` from the racket detector). COCO
   class 0 = "person". Pick the highest-confidence person bbox.
3. **Stage 2 — pose estimation:** Letterbox-pad the bbox to 256×192
   (RTMPose-m's native input shape, NOT a square — the model is trained
   on portrait-oriented person crops), run RTMPose-m via onnxruntime,
   read out 17 SimCC distributions, decode to (x, y, conf).
4. Inverse-transform keypoints from crop-pixel space back to full-frame
   normalized coords.
5. Map COCO-17 ids → BlazePose-33 ids and emit a 33-entry landmark array
   with `visibility=0` for unfilled ids.
6. Compute joint angles (existing `compute_joint_angles_from_dicts`),
   smooth (existing client-side `smoothFrames`), bone-length-filter
   (existing).

### Why this is correct (vs. the prior crop attempt)

- **Letterbox, never stretch.** The model's geometry priors assume an
  upright human at the input aspect ratio. Stretching a wide bbox into
  256×192 distorts limb angles and biases the keypoints. Letterbox pads
  the short axis with mean-grey so the human's aspect ratio is preserved.
- **Bbox padding before letterbox.** Expand the YOLO bbox by ~25% on each
  side so RTMPose's training distribution (humans with breathing room
  around them, not tight crops) is matched. The original crop attempt
  used 1.5× expansion and ran into geometry issues — here the fix is
  separating "expand for context" from "letterbox for aspect ratio".
- **Inverse transform (load-bearing math, gets unit-tested):**

  ```
  pose_kpt_in_letterbox_pixels = SimCC decode → (lx, ly)
  pose_kpt_in_crop_pixels      = ((lx - pad_x) / scale, (ly - pad_y) / scale)
  pose_kpt_in_image_pixels     = (crop_x0 + cx, crop_y0 + cy)
  pose_kpt_normalized          = (x_img / image_w, y_img / image_h)
  ```

  Each step is a pure function of the previous, no shared mutable state.

### Failure-mode coverage

| Failure | Mechanism that fixes it |
|---|---|
| Fast wrist/elbow blur | RTMPose's SimCC head is empirically more blur-robust than heatmap argmax. Plus the two-stage crop means the wrist is presented to the pose model at higher effective resolution. |
| Small-in-frame subject | YOLO finds a 15%-tall player as a tight bbox, RTMPose then sees the player at full 256×192 input. ~6× resolution boost over MediaPipe's full-frame downscale. |
| Back / oblique view | RTMPose is trained on COCO + extra body-7 keypoint data, which has dense back/oblique coverage (sports, dance, athletes from broadcast angles). Less front/side bias than BlazePose. |

## 6. Implementation plan

1. **Add `rtmlib` to `railway-service/requirements.txt`.** Pin a known
   version. Pull the body-7 RTMPose-m ONNX bundle on first import,
   cache to `railway-service/models/rtmpose-m.onnx`.
2. **Write `railway-service/pose_rtmpose.py`** containing:
   - YOLO11n person detector (reuse pieces from `racket_detector.py`).
     Run on the full frame, take the highest-confidence person bbox in
     xyxy pixel coords.
   - `rtmlib.RTMPose(...)` instance held module-level. `__call__(image,
     bboxes=[bbox])` returns keypoints already in original-image pixel
     coords (rtmlib owns the letterbox + SimCC decode + inverse
     transform — no hand-rolled geometry on our side, which is the
     thing the prior crop attempt got wrong).
   - COCO-17 → BlazePose-33 id mapping (static dict).
   - Score → `visibility` translation (rtmlib already returns per-keypoint
     softmax peak).
3. **Wire the `POSE_BACKEND=rtmpose` branch in
   `extract_keypoints_from_video` and `extract_clip_keypoints.py`** to
   call the new module. Default `POSE_BACKEND` stays `mediapipe` so the
   change is opt-in for the first deploy.
4. **Bump `schema_version` to 3** in the rtmpose path. The wrist-flexion
   angle is omitted (left/right_wrist undefined), heels/foot-index left
   at visibility 0.
5. **Tests** (`railway-service/tests/test_pose_rtmpose.py`):
   - COCO-17 → BlazePose-33 mapping fills the right ids and zeros the
     right ids on a known input.
   - Person-bbox selection from a multi-detection YOLO output picks the
     highest-confidence person (class 0).
   - With rtmlib stubbed (no real ONNX session), the wrapper builds a
     valid PoseFrame for a synthetic frame.
   - Mock both onnxruntime sessions; no GPU / no real model file
     required for CI.
6. **Wholeextraction integration test** (`tests/test_extract_rtmpose.py`):
   feed a real ~30-frame video clip from `pro-videos/clips`, assert the
   output has the right shape, monotonic timestamps, and at least one
   confident shoulder/hip/knee per frame.
7. **TypeScript side** — `schema_version: 3 | 2 | 1` in `lib/supabase.ts`
   and document the wrist-angle omission. No render changes needed
   (PoseRenderer ignores ids it doesn't recognize).
8. **Doc & ops** — update AGENTS.md / ARCHITECTURE.md to reflect the
   pipeline. Add `POSE_BACKEND=rtmpose` flip instructions.

## 7. Validation plan

There's no labeled tennis test set to compute COCO-AP against, so the
acceptance criteria are observable / measurable on existing pro clips
in `public/pro-videos/`:

| Failure mode | Clip |
|---|---|
| Small-in-frame, back-facing | `carlos_alcaraz_forehand_behind_1776374463938.mp4` |
| Fast joint (forehand contact) | `jannik_sinner_forehand_side_1776381824347.mp4` |
| Back-facing oblique | `novak_djokovic_backhand_side_1776383254038.mp4` |

- **Landmark drop rate per failure mode.** For each of the three clips
  above, count the frames where bone-length-plausibility filter zeros
  the elbow or wrist. Lower is better — we expect order-of-magnitude
  improvement on small-in-frame.
- **Bone-length variance over the clip.** Compute std-dev of
  shoulder→elbow and elbow→wrist length divided by the median, per side.
  Lower = the tracker is internally consistent. We expect a meaningful
  drop on fast-joint clips.
- **Visual sanity check.** Render overlays side-by-side on the chosen
  clips, capture screenshots in `docs/pose-research-screenshots/` so the
  diff is reviewable on the PR.
- **No regression on existing tests.** All 544 vitest tests + the
  existing pytest tests must still pass. Schema-version-aware tests
  added for the v3 path.

## 8. References

- [RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose (arXiv 2303.07399)](https://arxiv.org/abs/2303.07399)
- [MMPose RTMPose project (GitHub)](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- [rtmlib (Tau-J/rtmlib) — RTMPose without mmcv/mmpose](https://github.com/Tau-J/rtmlib)
- [SimCC: Coordinate Classification for Pose Estimation (ECCV'22, arXiv 2107.03332)](https://arxiv.org/abs/2107.03332)
- [RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation (arXiv 2407.08634)](https://arxiv.org/abs/2407.08634)
- [Ultralytics YOLO11 / pose docs](https://docs.ultralytics.com/tasks/pose/)
- [Ultralytics YOLO Evolution — YOLO26/11/v8/v5 (arXiv 2510.09653)](https://arxiv.org/html/2510.09653v2)
- [ViTPose / ViTPose+ (TPAMI'23, arXiv 2212.04246)](https://arxiv.org/abs/2212.04246)
- [Sapiens: Foundation for Human Vision Models (arXiv 2408.12569)](https://arxiv.org/html/2408.12569v1)
- [Sapiens LICENSE (CC BY-NC 4.0)](https://github.com/facebookresearch/sapiens/blob/main/LICENSE)
- [Pose2Sim (perfanalytics/pose2sim)](https://github.com/perfanalytics/pose2sim) — sports biomechanics framework that switched default backend from OpenPose / MediaPipe to RTMPose
- [COCO-WholeBody data format](https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md)
- [Best Pose Estimation Models (Roboflow blog)](https://blog.roboflow.com/best-pose-estimation-models/)
