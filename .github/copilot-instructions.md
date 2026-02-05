# Copilot instructions

## Build, test, lint
- Install dependencies: `pip install -e .`
- Run app (webcam required):
  - Energy ball/screenshot mode: `python main.py --mode ball`
  - Finger count mode: `python main.py --mode number`
- No automated tests or linters are configured in this repo.

## Architecture
- Single entrypoint: `main.py` defines `HandApp`, `Config`, and `State`.
- MediaPipe `HandLandmarker` (model asset `hand_landmarker.task` in repo root) runs in LIVE_STREAM mode with callback `on_hand_result`.
- Two modes:
  - `ball`: uses thumb/index pinch distance to render the energy ball and trigger screenshots (`screenshot_<ts>.png`) on pinch.
  - `number`: counts fingers for up to 2 hands and overlays per-hand counts.

## Conventions
- Landmark indices/constants (`THUMB_TIP`, `FINGER_TIPS`, etc.) are shared across counting and drawing; reuse these rather than hardcoding.
- Gesture logic flows through `on_hand_result` -> `State` updates -> `process`/`draw` each frame; avoid bypassing state fields.
- Mode-specific behavior is keyed off `--mode` and `self.mode` checks; keep new features consistent with this switch.
- Runtime expects Python 3.13 (see `.python-version` and `pyproject.toml`).
