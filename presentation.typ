#import "@preview/diatypst:0.9.1": *

#show: slides.with(
  title: "Pouce",
  subtitle: "Hand Gesture Controller using MediaPipe",
  date: "05.02.2026",
  title-color: blue,
  ratio: 16 / 9,
  layout: "small",
  toc: true,
  count: "dot",
  footer: true,
)

= Introduction

== What is Pouce?

*Pouce* is a real-time hand gesture recognition and visualization tool.

- Built with #link("https://google.github.io/mediapipe/")[MediaPipe] for robust hand landmark detection.
- Uses #link("https://opencv.org/")[OpenCV] for interactive feedback and visualization.
- Written in Python, managed with `uv`.

The project name "Pouce" (French for "Thumb") highlights the central role of finger interaction in the system.

== Core Technologies

/ *MediaPipe*: Provides a high-fidelity hand landmarker model that tracks 21 3D landmarks.
/ *OpenCV*: Handles camera stream processing and UI rendering.
/ *NumPy*: Powers the canvas and mathematical operations for gesture detection.

= Features

== Overview of Modes

Pouce offers three distinct interactive modes:

1. *Energy Ball*: A visual effect tied to finger distance.
2. *Finger Count*: Real-time counting of extended fingers.
3. *Air Painter*: Drawing on a virtual canvas using pinch gestures.

= Technical Implementation

== Gesture Detection Logic

The system identifies gestures by calculating distances and angles between landmarks:

- *Pinch Detection*: Measures the Euclidean distance between the thumb tip (landmark 4) and index tip (landmark 8).
- *Finger Extension*: Uses a combination of angles and relative distances from the wrist to determine if a finger is
  extended.

```python
def _pinch_distance(hand) -> float:
    thumb_tip = hand[THUMB_TIP]
    index_tip = hand[INDEX_TIP]
    return math.hypot(thumb_tip.x - index_tip.x,
                      thumb_tip.y - index_tip.y)
```

== Mode 1: Energy Ball

Interactive visualization between the thumb and index finger.

- *Visual*: A glowing ball appears at the midpoint of the pinch.
- *Interaction*: Pinching triggers a screenshot.
- *Feedback*: Displays the raw distance value for debugging.

== Mode 2: Finger Count

Detects and counts extended fingers on multiple hands.

- *Support*: Handles up to 2 hands simultaneously.
- *Handedness*: Correctly identifies Left vs Right hand.
- *Precision*: Uses MCP (Metacarpophalangeal) joint angles for accurate extension detection.

== Mode 3: Air Painter

Turns your hand into a virtual brush.

- *Draw*: Pinch to start drawing on the screen.
- *Brush*: Follows the index finger tip.
- *Clear*: Show all 5 fingers to reset the canvas.
