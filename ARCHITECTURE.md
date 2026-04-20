# TennisIQ — Architecture & Setup Guide

## What This Is

A web app that lets users upload tennis swing videos, extracts pose landmarks in-browser using MediaPipe, and provides AI coaching feedback by comparing form against professional players.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 19, Tailwind CSS |
| State | Zustand (6 stores) |
| Pose Extraction | MediaPipe Tasks Vision (client-side WASM) |
| AI Coaching | Claude API (streaming) |
| Database | Supabase (PostgreSQL) |
| Video Storage | Vercel Blob (user uploads), local `public/pro-videos/` (pro clips) |
| Backend Service | Python FastAPI on Railway (server-side pose extraction) |
| Video Processing | yt-dlp (download), ffmpeg (trim) |
| Testing | Vitest (342 tests), Pytest (195 tests) |

## Project Structure

```
app/
  page.tsx                    # Landing page
  analyze/page.tsx            # Upload & analyze your swing
  compare/page.tsx            # Side-by-side / overlay comparison
  pros/page.tsx               # Browse pro library
  admin/tag-clips/page.tsx    # Admin tool to tag YouTube clips
  api/
    upload/                   # Upload video to Vercel Blob
    sessions/                 # Persist keypoints to Supabase
    pros/                     # List pros with swings
    pro-swings/[id]/          # Fetch single swing with keypoints
    analyze/                  # Stream AI coaching feedback
    pro-coach/                # Pro-specific Q&A chat
    process-youtube/          # YouTube processing pipeline
    admin/
      tag-clip/               # Download, trim, save YouTube clip
      extract-keypoints/      # Extract pose data from saved clip

components/
  UploadZone.tsx              # Video upload + client-side pose extraction
  VideoCanvas.tsx             # Video playback with pose overlay
  ProSwingViewer.tsx          # Pro swing video player (Browse page)
  ProBrowser.tsx              # Pro list with shot type filters
  ProSelector.tsx             # Compact pro picker (Analyze page)
  ComparisonLayout.tsx        # Side-by-side / overlay comparison
  LLMCoachingPanel.tsx        # AI coaching with Claude
  ProCoachChat.tsx            # Chat with AI about a pro's technique
  JointTogglePanel.tsx        # Toggle joint visibility
  MetricsComparison.tsx       # Angle comparison table
  Nav.tsx                     # Navigation with active state

lib/
  supabase.ts                 # Supabase client + TypeScript types
  mediapipe.ts                # MediaPipe singleton + monotonic timestamps
  poseSmoothing.ts            # EMA smoothing, warm-up frame discard
  syncAlignment.ts            # Phase detection, time mapping, camera angle
  jointAngles.ts              # Joint angle computation, swing detection
  shotTypeConfig.ts           # Per-shot coaching configs and ideal ranges

store/
  index.ts                    # Zustand stores (pose, video, joints, comparison, analysis, sync)
  proLibrary.ts               # Pro library page state + chat

railway-service/
  main.py                     # FastAPI service for server-side extraction
  seed_pros.py                # Pro database seeder with trim support
  youtube_processor.py        # YouTube processing pipeline
  scene_detector.py           # PySceneDetect integration
  camera_classifier.py        # Court/player detection
  clip_extractor.py           # ffmpeg clip extraction
  extract_clip_keypoints.py   # CLI wrapper for keypoint extraction
  shot_classifiers/           # Shot type classification (forehand, backhand, serve, slice)
  tests/                      # 195 Python tests
```

## Setup

### Prerequisites

- Node.js 20+
- Python 3.11+
- ffmpeg (included at `pro-videos/bin/ffmpeg` or install globally)
- yt-dlp (`pip install yt-dlp` or use the bundled binary)

### Environment Variables

Create a `.env` file in the project root:

```env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# Anthropic (for AI coaching)
ANTHROPIC_API_KEY=sk-ant-...

# Vercel Blob (for video uploads)
BLOB_READ_WRITE_TOKEN=vercel_blob_...

# Railway service (optional, for server-side extraction + /classify-angle
# telemetry). If either is unset, analysis_events.capture_quality_flag stays
# null for every analyze call — telemetry is fire-and-forget and never blocks
# coaching.
RAILWAY_SERVICE_URL=https://your-service.railway.app
EXTRACT_API_KEY=your-api-key
```

### Install & Run

```bash
# Install dependencies
npm install

# Install Python dependencies (for Railway service / keypoint extraction)
cd railway-service && pip install -r requirements.txt && cd ..

# Run the dev server
npm run dev

# Run tests
npm test                    # Vitest (frontend)
cd railway-service && python3 -m pytest tests/  # Pytest (backend)
```

### Database Schema

The Supabase schema is at `lib/db/schema.sql`. Two main tables:

- **`pros`**: id, name, nationality, ranking, bio, profile_image_url
- **`pro_swings`**: id, pro_id (FK), shot_type, video_url, keypoints_json, fps, frame_count, duration_ms, phase_labels, metadata

User sessions are stored in **`user_sessions`**: id, blob_url, shot_type, keypoints_json, status, etc.

## Key Flows

### 1. User Uploads a Video (Analyze Page)

```
User selects shot type + drops video
  -> Upload to Vercel Blob via /api/upload
  -> Create user_sessions row in Supabase
  -> Load MediaPipe WASM in browser
  -> Seek frame-by-frame, extract 33-point pose landmarks at 30fps
  -> Confidence filter + EMA smoothing (5-frame warm-up discard)
  -> Persist keypoints via /api/sessions
  -> Display video with skeleton overlay
```

### 2. AI Coaching (Analyze Page)

```
User selects a pro to compare against
  -> Click "Analyze Swing"
  -> POST /api/analyze with user keypoints + pro swing ID
  -> Build biomechanics summary (joint angles per phase)
  -> Stream Claude response comparing user vs pro form
  -> Display coaching feedback in real-time
```

### 3. Admin Tags a YouTube Clip

```
Admin goes to /admin/tag-clips
  -> Paste YouTube URL, video loads in embedded player
  -> Scrub to find a clean single shot
  -> Click "Mark Start" / "Mark End"
  -> Select pro name, shot type, camera angle
  -> Click "Save Clip"
  -> Backend: yt-dlp downloads -> ffmpeg trims -> saves to public/pro-videos/
  -> Upserts pro in Supabase, inserts pro_swing row
  -> Optionally: POST /api/admin/extract-keypoints to extract pose data
```

### 4. Phase-Synced Comparison (Compare Page)

```
User video + pro video loaded
  -> detectSwingPhases() identifies: preparation, backswing, contact, follow-through
  -> computeTimeMapping() builds piecewise linear function
  -> Primary video drives playback, secondary follows via phase alignment
  -> Side-by-side or overlay mode with drift-based rate adjustment
```

## Shot Classification System

The `railway-service/shot_classifiers/` package contains 4 heuristic classifiers based on sports biomechanics:

| Classifier | Primary Signal | Confidence Threshold |
|-----------|---------------|---------------------|
| Forehand | Wrist crosses body R->L (dominant side) | 0.35 |
| Backhand | Lead wrist L->R, one/two-handed detection | 0.30 |
| Serve | Wrist rises above head (trophy position) | 0.15 |
| Slice | High-to-low wrist path | 0.30 |

Each classifier detects handedness, estimates camera angle, validates clean single shots, and returns phase timestamps.

## Playback Speed

Pro clips are slow-motion. The ProSwingViewer computes real-time playback rates from biomechanics research:

| Shot | Real-Time Duration | 6s Clip Rate |
|------|-------------------|-------------|
| Forehand | ~0.7s | ~8.6x |
| Backhand | ~0.7s | ~8.6x |
| Serve | ~1.5s | ~4x |
| Volley | ~0.4s | ~15x |

Default playback is "Original" (1x slow-mo). Users can switch to "Real Speed" or fractional speeds.

## Testing

- **342 Vitest tests**: Components, stores, utilities, API routes
- **195 Pytest tests**: Shot classifiers (forehand, backhand, serve, slice), cross-validation, YouTube pipeline integration, camera classifier, clip extractor

All tests pass with zero failures.
