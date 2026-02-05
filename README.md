# Pouce 🖐️

**Pouce** (French for "Thumb") is a real-time hand gesture control interface built with Python, MediaPipe, and OpenCV. It provides an intuitive, touchless way to interact with your computer through various modes, ranging from virtual drawing to games.

## 🌟 Features

- **Robust Finger Detection**: Uses a multi-criteria approach (joint angles and palm center reference) for accurate finger counting and state detection, including the complex thumb movement.
- **Multiple Modes**:
  - **Energy Ball & Screenshot (`ball`)**: Visualize the pinch gesture between thumb and index. Pinching captures a screenshot.
  - **Air Painter (`paint`)**: Draw in the air! Pinch to spray paint, and show all 5 fingers to clear the canvas.
  - **Finger Count (`number`)**: Real-time finger counting for up to two hands simultaneously.
  - **Rock Paper Scissors (`rps`)**: Play the classic game against the CPU with a 3-second countdown.
- **High Performance**: Optimized using asynchronous MediaPipe processing for 30+ FPS fluid interaction.

## 🛠️ Tech Stack

- **Language**: Python 3.13
- **Vision**: [MediaPipe](https://mediapipe.dev/) (Hand Landmarker), [OpenCV](https://opencv.org/)
- **Package Management**: [uv](https://github.com/astral-sh/uv)

## 🚀 Getting Started

### Prerequisites

- A webcam
- [uv](https://github.com/astral-sh/uv) installed on your system

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd pouce
   ```

2. Download the MediaPipe Hand Landmarker model:
   The application expects `hand_landmarker.task` in the root directory. You can download it from the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models).

3. Install dependencies and set up the environment:
   ```bash
   uv sync
   ```

### Usage

Run the application using `uv`:

```bash
# Default mode (Ball/Screenshot)
uv run main.py

# Air Painter mode
uv run main.py --mode paint

# Finger Count mode
uv run main.py --mode number

# Rock Paper Scissors mode
uv run main.py --mode rps
```

Press **'q'** to quit the application.

## 🧠 How it Works

The detection logic evolved through three iterations to achieve robustness:

1. **Y-Coordinates**: Initially based on tip height (failed on hand rotation).
2. **Wrist Distance**: Based on distance from the wrist (failed on thumb and perspective).
3. **Angles & Palm (Current)**: Combines joint angles (MCP-PIP-TIP) and palm center proximity for the thumb, providing a stable experience regardless of hand orientation.
