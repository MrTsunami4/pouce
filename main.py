import argparse
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarkerResult

# --- Constants ---
THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
FINGER_TIPS = [THUMB_TIP, INDEX_TIP, 12, 16, 20]
FINGER_PIPS = [THUMB_IP, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]


# --- Configuration ---
@dataclass(frozen=True)
class Config:
    model_path: str = "hand_landmarker.task"
    pinch_threshold: float = 0.05
    radius_scale: float = 300.0
    camera_index: int = 0
    brush_color: Tuple[int, int, int] = (0, 0, 255)
    brush_thickness: int = 10


# --- Utility Functions ---
class HandUtils:
    @staticmethod
    def get_first_hand(result: HandLandmarkerResult):
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    @staticmethod
    def get_handedness(result: HandLandmarkerResult) -> Optional[str]:
        if result.handedness:
            return result.handedness[0][0].category_name
        return None

    @staticmethod
    def get_all_handedness(result: HandLandmarkerResult) -> List[str]:
        if not result.handedness:
            return []
        return [group[0].category_name for group in result.handedness]

    @staticmethod
    def calculate_pinch_distance(hand) -> float:
        thumb_tip = hand[THUMB_TIP]
        index_tip = hand[INDEX_TIP]
        return math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    @staticmethod
    def count_fingers(hand) -> int:
        """Return number of extended fingers using a landmark heuristic."""
        count = 0

        thumb_tip = hand[FINGER_TIPS[0]]
        thumb_ip = hand[FINGER_PIPS[0]]
        thumb_mcp = hand[FINGER_MCPS[0]]
        wrist = hand[0]
        index_mcp = hand[FINGER_MCPS[1]]
        pinky_mcp = hand[FINGER_MCPS[-1]]

        palm_cx = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
        palm_cy = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
        palm_width = math.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y)
        palm_width = max(palm_width, 1e-6)
        
        distance_margin = max(palm_width * 0.05, 0.01)
        side_margin = max(palm_width * 0.15, 0.015)

        def dist(a, b) -> float:
            return math.hypot(a.x - b.x, a.y - b.y)

        def dist_xy(a, x: float, y: float) -> float:
            return math.hypot(a.x - x, a.y - y)

        def angle(a, b, c) -> float:
            abx, aby = a.x - b.x, a.y - b.y
            cbx, cby = c.x - b.x, c.y - b.y
            denom = math.hypot(abx, aby) * math.hypot(cbx, cby)
            if denom == 0:
                return 0.0
            cos_value = (abx * cbx + aby * cby) / denom
            cos_value = max(-1.0, min(1.0, cos_value))
            return math.degrees(math.acos(cos_value))

        # Check thumb
        thumb_angle = angle(thumb_mcp, thumb_ip, thumb_tip)
        thumb_extended = (
            thumb_angle > 140.0
            and dist(thumb_tip, wrist) > dist(thumb_ip, wrist) + distance_margin
            and dist_xy(thumb_tip, palm_cx, palm_cy)
            > dist_xy(thumb_ip, palm_cx, palm_cy) + distance_margin
        )
        thumb_extended = thumb_extended and (abs(thumb_tip.x - palm_cx) > side_margin)
        count += thumb_extended

        # Check other fingers
        for mcp_idx, pip_idx, tip_idx in zip(
            FINGER_MCPS[1:], FINGER_PIPS[1:], FINGER_TIPS[1:]
        ):
            mcp = hand[mcp_idx]
            pip = hand[pip_idx]
            tip = hand[tip_idx]
            finger_is_extended = (
                angle(mcp, pip, tip) > 160.0
                and dist(tip, wrist) > dist(pip, wrist) + distance_margin
            )
            count += finger_is_extended

        return int(count)


# --- Modes ---
class BaseMode(ABC):
    def __init__(self, config: Config):
        self.config = config

    @property
    @abstractmethod
    def num_hands(self) -> int:
        """Number of hands to detect."""
        pass

    @property
    @abstractmethod
    def window_name(self) -> str:
        """Window title."""
        pass

    @property
    @abstractmethod
    def intro_message(self) -> str:
        """Message printed on startup."""
        pass

    @abstractmethod
    def on_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        """Callback for MediaPipe results."""
        pass

    @abstractmethod
    def process(self, frame: np.ndarray) -> None:
        """Process logic (e.g., state updates, saving files) per frame."""
        pass

    @abstractmethod
    def draw(self, frame: np.ndarray) -> None:
        """Draw overlays on the frame."""
        pass


class BallMode(BaseMode):
    def __init__(self, config: Config):
        super().__init__(config)
        self.landmarks = None
        self.distance: float = 0.0
        self.was_pinching: bool = False

    @property
    def num_hands(self) -> int:
        return 1

    @property
    def window_name(self) -> str:
        return "Gesture Controller"

    @property
    def intro_message(self) -> str:
        return "System Ready. Pinch your fingers to take a screenshot!"

    def on_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        hand = HandUtils.get_first_hand(result)
        if hand:
            self.landmarks = hand
            self.distance = HandUtils.calculate_pinch_distance(hand)
        else:
            self.landmarks = None
            self.distance = 0.0

    def process(self, frame: np.ndarray) -> None:
        is_pinching = (
            self.distance < self.config.pinch_threshold and self.landmarks is not None
        )
        if is_pinching and not self.was_pinching:
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        self.was_pinching = is_pinching

    def draw(self, frame: np.ndarray) -> None:
        if not self.landmarks:
            return

        h, w = frame.shape[:2]
        thumb = self.landmarks[THUMB_TIP]
        index = self.landmarks[INDEX_TIP]

        thumb_px = (int(thumb.x * w), int(thumb.y * h))
        index_px = (int(index.x * w), int(index.y * h))
        center = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
        radius = int(self.distance * self.config.radius_scale)

        if self.distance < self.config.pinch_threshold:
            cv2.circle(frame, center, radius, (0, 255, 0), -1)
            cv2.putText(
                frame,
                "PINCH ACTIVE",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.line(frame, thumb_px, index_px, (200, 200, 200), 2)
            cv2.circle(frame, center, radius, (255, 255, 0), 3)

        cv2.putText(
            frame,
            f"Dist: {self.distance:.3f}",
            (center[0] - 40, center[1] + radius + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )


class PaintMode(BaseMode):
    def __init__(self, config: Config):
        super().__init__(config)
        self.landmarks = None
        self.distance: float = 0.0
        self.finger_count: int = 0
        self.canvas: Optional[np.ndarray] = None
        self.prev_brush_pos: Optional[Tuple[int, int]] = None

    @property
    def num_hands(self) -> int:
        return 1

    @property
    def window_name(self) -> str:
        return "Air Painter"

    @property
    def intro_message(self) -> str:
        return "System Ready. Pinch to draw, show 5 fingers to clear!"

    def on_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        hand = HandUtils.get_first_hand(result)
        if hand:
            self.landmarks = hand
            self.distance = HandUtils.calculate_pinch_distance(hand)
            self.finger_count = HandUtils.count_fingers(hand)
        else:
            self.landmarks = None
            self.distance = 0.0
            self.finger_count = 0
            self.prev_brush_pos = None

    def process(self, frame: np.ndarray) -> None:
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

        if not self.landmarks:
            self.prev_brush_pos = None
            return

        h, w = frame.shape[:2]
        index = self.landmarks[INDEX_TIP]
        ix, iy = int(index.x * w), int(index.y * h)

        if self.distance < self.config.pinch_threshold:
            if self.prev_brush_pos is not None:
                cv2.line(
                    self.canvas,
                    self.prev_brush_pos,
                    (ix, iy),
                    self.config.brush_color,
                    self.config.brush_thickness,
                )
            self.prev_brush_pos = (ix, iy)
        else:
            self.prev_brush_pos = None

        if self.finger_count >= 5:
            self.canvas = np.zeros_like(frame)

    def draw(self, frame: np.ndarray) -> None:
        self._draw_canvas(frame)
        self._draw_brush(frame)

    def _draw_canvas(self, frame: np.ndarray) -> None:
        if self.canvas is not None:
            canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            img_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            np.copyto(frame, cv2.add(img_bg, img_fg))

    def _draw_brush(self, frame: np.ndarray) -> None:
        if not self.landmarks:
            return

        h, w = frame.shape[:2]
        index = self.landmarks[INDEX_TIP]
        ix, iy = int(index.x * w), int(index.y * h)
        color = self.config.brush_color

        if self.distance < self.config.pinch_threshold:
            cv2.circle(frame, (ix, iy), self.config.brush_thickness, color, -1)
        else:
            cv2.circle(frame, (ix, iy), self.config.brush_thickness, color, 2)

        cv2.putText(
            frame,
            "Brush",
            (ix + 10, iy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


class NumberMode(BaseMode):
    def __init__(self, config: Config):
        super().__init__(config)
        self.multi_hand_counts: List[Tuple[str, int]] = []
        self.multi_hand_landmarks: List[Iterable] = []

    @property
    def num_hands(self) -> int:
        return 2

    @property
    def window_name(self) -> str:
        return "Finger Count"

    @property
    def intro_message(self) -> str:
        return "System Ready. Show fingers to display a number!"

    def on_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        hands = result.hand_landmarks or []
        labels = HandUtils.get_all_handedness(result)
        counts: List[Tuple[str, int]] = []
        for idx, hand in enumerate(hands):
            label = labels[idx] if idx < len(labels) else f"Hand {idx + 1}"
            counts.append((label, HandUtils.count_fingers(hand)))
        
        self.multi_hand_counts = counts
        self.multi_hand_landmarks = hands

    def process(self, frame: np.ndarray) -> None:
        pass  # No specific processing logic for NumberMode

    def draw(self, frame: np.ndarray) -> None:
        self._draw_finger_count(frame)
        self._draw_finger_tips(frame)
        self._draw_palms(frame)

    def _draw_finger_count(self, frame: np.ndarray) -> None:
        if not self.multi_hand_counts:
            cv2.putText(
                frame,
                "No hand detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            return

        y = 50
        for label, count in self.multi_hand_counts:
            cv2.putText(
                frame,
                f"{label}: {count}",
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            y += 40

    def _draw_finger_tips(self, frame: np.ndarray) -> None:
        if not self.multi_hand_landmarks:
            return

        h, w = frame.shape[:2]
        for hand in self.multi_hand_landmarks:
            for idx in FINGER_TIPS:
                lm = hand[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(idx),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )

    def _draw_palms(self, frame: np.ndarray) -> None:
        if not self.multi_hand_landmarks:
            return

        h, w = frame.shape[:2]
        for hand in self.multi_hand_landmarks:
            wrist = hand[0]
            index_mcp = hand[5]
            pinky_mcp = hand[17]
            palm_cx = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
            palm_cy = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
            x, y = int(palm_cx * w), int(palm_cy * h)
            cv2.circle(frame, (x, y), 8, (0, 165, 255), -1)
            cv2.putText(
                frame,
                "palm",
                (x + 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
            )


# --- Main Application ---
class HandApp:
    def __init__(self, mode: BaseMode) -> None:
        self.mode = mode

    def run(self) -> None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.mode.config.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.mode.on_result,
            num_hands=self.mode.num_hands,
        )

        try:
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
                cap = cv2.VideoCapture(self.mode.config.camera_index)
                if not cap.isOpened():
                    print("Error: Could not open webcam.")
                    return

                print(self.mode.intro_message)
                start_time = time.monotonic()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                    timestamp_ms = int((time.monotonic() - start_time) * 1000)
                    landmarker.detect_async(mp_image, timestamp_ms)

                    self.mode.process(frame)
                    self.mode.draw(frame)

                    cv2.imshow(self.mode.window_name, frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

                cap.release()
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"An error occurred: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Hand gesture controller")
    parser.add_argument(
        "--mode",
        choices=["ball", "number", "paint"],
        default="ball",
        help="Choose visualization mode (ball, number, or paint)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    
    modes = {
        "ball": BallMode,
        "number": NumberMode,
        "paint": PaintMode,
    }
    
    mode_class = modes.get(args.mode, BallMode)
    app = HandApp(mode_class(config))
    app.run()


if __name__ == "__main__":
    main()