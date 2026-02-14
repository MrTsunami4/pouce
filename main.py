import argparse
import math
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarkerResult

try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
FINGER_TIPS = (THUMB_TIP, INDEX_TIP, 12, 16, 20)
FINGER_PIPS = (THUMB_IP, 6, 10, 14, 18)
FINGER_MCPS = (2, 5, 9, 13, 17)


@dataclass(frozen=True)
class Config:
    model_path: str = "hand_landmarker.task"
    pinch_threshold: float = 0.05
    radius_scale: float = 300.0
    camera_index: int = 0
    brush_color: tuple = (0, 0, 255)
    brush_thickness: int = 10


def get_first_hand(result: HandLandmarkerResult) -> Optional[Any]:
    """Return the first detected hand's landmarks, or None if no hands detected."""
    return result.hand_landmarks[0] if result.hand_landmarks else None


def get_handedness(result: HandLandmarkerResult) -> Optional[str]:
    """Return the handedness category of the first hand (e.g., 'Left', 'Right')."""
    return result.handedness[0][0].category_name if result.handedness else None


def get_all_handedness(result: HandLandmarkerResult) -> List[str]:
    """Return a list of handedness categories for all detected hands."""
    if not result.handedness:
        return []
    return [hand[0].category_name for hand in result.handedness]


def calculate_pinch_distance(hand: List) -> float:
    """Calculate the distance between thumb tip and index tip."""
    thumb_tip = hand[THUMB_TIP]
    index_tip = hand[INDEX_TIP]
    return math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)


def _calculate_angle(point_a: Any, point_b: Any, point_c: Any) -> float:
    """Calculate the angle at point_b formed by points a-b-c in degrees."""
    ab_x, ab_y = point_a.x - point_b.x, point_a.y - point_b.y
    cb_x, cb_y = point_c.x - point_b.x, point_c.y - point_b.y

    denominator = math.hypot(ab_x, ab_y) * math.hypot(cb_x, cb_y)
    if denominator == 0:
        return 0.0

    cos_value = max(-1.0, min(1.0, (ab_x * cb_x + ab_y * cb_y) / denominator))
    return math.degrees(math.acos(cos_value))


def _calculate_distance(point_a: Any, point_b: Any) -> float:
    """Calculate Euclidean distance between two points."""
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def get_finger_states(hand: list) -> list[bool]:
    """Determine which fingers are extended. Returns list of 5 booleans."""
    thumb_tip = hand[FINGER_TIPS[0]]
    thumb_ip = hand[FINGER_PIPS[0]]
    thumb_mcp = hand[FINGER_MCPS[0]]
    wrist = hand[0]
    index_mcp = hand[FINGER_MCPS[1]]
    pinky_mcp = hand[FINGER_MCPS[-1]]

    palm_cx = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
    palm_cy = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
    palm_width = max(
        math.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y), 1e-6
    )

    distance_margin = max(palm_width * 0.05, 0.01)
    side_margin = max(palm_width * 0.15, 0.015)

    states = []

    thumb_angle = _calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
    thumb_extended = (
        thumb_angle > 140.0
        and _calculate_distance(thumb_tip, wrist)
        > _calculate_distance(thumb_ip, wrist) + distance_margin
        and _calculate_distance(thumb_tip, wrist)
        > _calculate_distance(thumb_ip, wrist) + distance_margin
        and abs(thumb_tip.x - palm_cx) > side_margin
    )
    states.append(bool(thumb_extended))

    for mcp_idx, pip_idx, tip_idx in zip(
        FINGER_MCPS[1:], FINGER_PIPS[1:], FINGER_TIPS[1:]
    ):
        mcp, pip, tip = hand[mcp_idx], hand[pip_idx], hand[tip_idx]
        angle_val = _calculate_angle(mcp, pip, tip)
        finger_extended = (
            angle_val > 160.0
            and _calculate_distance(tip, wrist)
            > _calculate_distance(pip, wrist) + distance_margin
        )
        states.append(bool(finger_extended))

    return states


def count_fingers(hand: list) -> int:
    """Return the number of extended fingers."""
    return sum(get_finger_states(hand))


class BaseMode(ABC):
    def __init__(self, config: Config) -> None:
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
        self, result: HandLandmarkerResult, output_image: Any, timestamp_ms: int
    ) -> None:
        """Callback for MediaPipe results."""
        pass

    @abstractmethod
    def process(self, frame: np.ndarray) -> None:
        """Process logic per frame."""
        pass

    @abstractmethod
    def draw(self, frame: np.ndarray) -> None:
        """Draw overlays on the frame."""
        pass


class BallMode(BaseMode):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.landmarks: Optional[list] = None
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
        self, result: HandLandmarkerResult, output_image: Any, timestamp_ms: int
    ) -> None:
        hand = get_first_hand(result)
        if hand:
            self.landmarks = hand
            self.distance = calculate_pinch_distance(hand)
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
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.landmarks: Optional[list] = None
        self.distance: float = 0.0
        self.finger_count: int = 0
        self.canvas: Optional[np.ndarray] = None
        self.prev_brush_pos: Optional[tuple] = None

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
        self, result: HandLandmarkerResult, output_image: Any, timestamp_ms: int
    ) -> None:
        hand = get_first_hand(result)
        if hand:
            self.landmarks = hand
            self.distance = calculate_pinch_distance(hand)
            self.finger_count = count_fingers(hand)
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
        if self.canvas is None:
            return

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
        thickness = self.config.brush_thickness
        radius = -1 if self.distance < self.config.pinch_threshold else thickness

        cv2.circle(frame, (ix, iy), thickness, color, radius)
        cv2.putText(
            frame, "Brush", (ix + 10, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )


class NumberMode(BaseMode):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.multi_hand_counts: list = []
        self.multi_hand_landmarks: list = []

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
        self, result: HandLandmarkerResult, output_image: Any, timestamp_ms: int
    ) -> None:
        hands = result.hand_landmarks or []
        labels = get_all_handedness(result)

        self.multi_hand_counts = [
            (
                labels[idx] if idx < len(labels) else f"Hand {idx + 1}",
                count_fingers(hand),
            )
            for idx, hand in enumerate(hands)
        ]
        self.multi_hand_landmarks = hands

    def process(self, frame: np.ndarray) -> None:
        pass

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


class RockPaperScissorsMode(BaseMode):
    ROCK = "Rock"
    PAPER = "Paper"
    SCISSORS = "Scissors"
    UNKNOWN = "Unknown"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.landmarks: Optional[list] = None
        self.finger_states: list = []
        self.state = "WAITING"
        self.timer_start: float = 0.0
        self.player_move = self.UNKNOWN
        self.cpu_move = self.UNKNOWN
        self.result_text = ""
        self.scores = {"Player": 0, "CPU": 0}

    @property
    def num_hands(self) -> int:
        return 1

    @property
    def window_name(self) -> str:
        return "Rock Paper Scissors"

    @property
    def intro_message(self) -> str:
        return "Rock Paper Scissors! Show 0, 2, or 5 fingers."

    def on_result(
        self, result: HandLandmarkerResult, output_image: Any, timestamp_ms: int
    ) -> None:
        hand = get_first_hand(result)
        if hand:
            self.landmarks = hand
            self.finger_states = get_finger_states(hand)
        else:
            self.landmarks = None
            self.finger_states = []

    def process(self, frame: np.ndarray) -> None:
        current_time = time.time()

        if self.state == "WAITING":
            if self.landmarks:
                self.state = "COUNTDOWN"
                self.timer_start = current_time

        elif self.state == "COUNTDOWN":
            elapsed = current_time - self.timer_start
            if elapsed >= 3.0:
                self._play_round()
                self.state = "SHOW_RESULT"
                self.timer_start = current_time

        elif self.state == "SHOW_RESULT":
            elapsed = current_time - self.timer_start
            if elapsed >= 3.0:
                self.state = "WAITING"
                self.player_move = self.UNKNOWN
                self.cpu_move = self.UNKNOWN
                self.result_text = ""

    def _play_round(self) -> None:
        count = sum(self.finger_states)

        if count in (0, 1):
            self.player_move = self.ROCK
        elif count in (2, 3):
            if self.finger_states[1] and self.finger_states[2]:
                self.player_move = self.SCISSORS
            else:
                self.player_move = self.UNKNOWN
        elif count == 5:
            self.player_move = self.PAPER
        else:
            self.player_move = self.UNKNOWN

        self.cpu_move = random.choice((self.ROCK, self.PAPER, self.SCISSORS))

        if self.player_move == self.UNKNOWN:
            self.result_text = "Invalid Move!"
            return

        beats = {
            self.ROCK: self.SCISSORS,
            self.PAPER: self.ROCK,
            self.SCISSORS: self.PAPER,
        }

        if self.player_move == self.cpu_move:
            self.result_text = "Draw!"
        elif beats[self.player_move] == self.cpu_move:
            self.result_text = "You Win!"
            self.scores["Player"] += 1
        else:
            self.result_text = "CPU Wins!"
            self.scores["CPU"] += 1

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        self._draw_debug(frame)

        cv2.putText(
            frame,
            f"Player: {self.scores['Player']}  CPU: {self.scores['CPU']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        if self.state == "WAITING":
            cv2.putText(
                frame,
                "Show hand to start",
                (center_x - 150, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200, 200, 200),
                2,
            )

        elif self.state == "COUNTDOWN":
            elapsed = time.time() - self.timer_start
            count = 3 - int(elapsed)
            text = "GO!" if count <= 0 else str(count)
            color = (0, 255, 0) if count <= 0 else (0, 255, 255)
            x_offset = -60 if count <= 0 else -20
            cv2.putText(
                frame,
                text,
                (center_x + x_offset, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                color,
                5,
            )

        elif self.state == "SHOW_RESULT":
            cv2.putText(
                frame,
                f"You: {self.player_move}",
                (50, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"CPU: {self.cpu_move}",
                (w - 300, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            color = (255, 255, 255)
            if "Win" in self.result_text:
                color = (0, 255, 0)
            elif "CPU Wins" in self.result_text:
                color = (0, 0, 255)

            text_size = cv2.getTextSize(
                self.result_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3
            )[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(
                frame,
                self.result_text,
                (text_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                color,
                3,
            )

    def _draw_debug(self, frame: np.ndarray) -> None:
        if not self.landmarks:
            return

        h, w = frame.shape[:2]
        count = sum(self.finger_states)
        cv2.putText(
            frame,
            f"Fingers: {count}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        for idx, extended in enumerate(self.finger_states):
            lm_idx = FINGER_TIPS[idx]
            lm = self.landmarks[lm_idx]
            x, y = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 0) if extended else (0, 0, 255)
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 1)


class MouseMode(BaseMode):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.landmarks: Optional[list] = None
        self.landmarks_right: Optional[list] = None
        self.distance: float = 0.0
        self.is_pinching = False
        self.was_pinching = False
        self.is_dragging = False

        if PYAUTOGUI_AVAILABLE:
            self.screen_width, self.screen_height = pyautogui.size()
        else:
            self.screen_width, self.screen_height = 1920, 1080

        self.sensitivity = 2.0
        self.prev_cursor_pos: Optional[tuple] = None

    @property
    def num_hands(self) -> int:
        return 2

    @property
    def window_name(self) -> str:
        return "Virtual Mouse"

    @property
    def intro_message(self) -> str:
        return "Move hand to control cursor. Pinch to click. Two hands = drag."

    def on_result(
        self, result: HandLandmarkerResult, output_image: Any, timestamp_ms: int
    ) -> None:
        hands = result.hand_landmarks or []
        labels = get_all_handedness(result)

        self.landmarks = None
        self.landmarks_right = None

        for idx, hand in enumerate(hands):
            label = labels[idx] if idx < len(labels) else ""
            if label == "Left":
                self.landmarks = hand
            elif label == "Right":
                self.landmarks_right = hand

        if not self.landmarks:
            self.landmarks = hands[0] if hands else None

        if self.landmarks:
            self.distance = calculate_pinch_distance(self.landmarks)
            self.is_pinching = self.distance < self.config.pinch_threshold
        else:
            self.distance = 0.0
            self.is_pinching = False

    def process(self, frame: np.ndarray) -> None:
        if not PYAUTOGUI_AVAILABLE or not self.landmarks:
            return

        h, w = frame.shape[:2]
        index = self.landmarks[INDEX_TIP]
        ix, iy = int(index.x * w), int(index.y * h)

        screen_x = int(index.x * self.screen_width)
        screen_y = int(index.y * self.screen_height)

        if self.prev_cursor_pos:
            dx = (screen_x - self.prev_cursor_pos[0]) * self.sensitivity
            dy = (screen_y - self.prev_cursor_pos[1]) * self.sensitivity
            current_pos = pyautogui.position()
            new_x = max(0, min(self.screen_width - 1, current_pos.x + dx))
            new_y = max(0, min(self.screen_height - 1, current_pos.y + dy))
            pyautogui.moveTo(int(new_x), int(new_y))

        self.prev_cursor_pos = (screen_x, screen_y)

        if self.is_pinching and not self.was_pinching:
            if self.landmarks_right and not self.is_dragging:
                self.is_dragging = True
                pyautogui.mouseDown()
            else:
                pyautogui.click()
        elif not self.is_pinching and self.was_pinching:
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False

        self.was_pinching = self.is_pinching

    def draw(self, frame: np.ndarray) -> None:
        if not self.landmarks:
            if not PYAUTOGUI_AVAILABLE:
                cv2.putText(
                    frame,
                    "pyautogui not installed",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Show hand to control cursor",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (200, 200, 200),
                    2,
                )
            return

        self._draw_cursor(frame)
        self._draw_click_state(frame)
        if self.landmarks_right:
            self._draw_drag_indicator(frame)

    def _draw_cursor(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        index = self.landmarks[INDEX_TIP]
        ix, iy = int(index.x * w), int(index.y * h)

        color = (0, 255, 0) if not self.is_pinching else (255, 0, 0)
        cv2.circle(frame, (ix, iy), 15, color, 2)
        cv2.putText(
            frame,
            "Cursor",
            (ix + 20, iy - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    def _draw_click_state(self, frame: np.ndarray) -> None:
        if self.is_pinching:
            cv2.putText(
                frame,
                "CLICK",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

    def _draw_drag_indicator(self, frame: np.ndarray) -> None:
        if self.is_dragging:
            cv2.putText(
                frame,
                "DRAGGING",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )


class ZoomMode(BaseMode):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.hands: list = []
        self.labels: list = []
        self.zoom_level = 1.0
        self.base_distance = 0.0
        self.is_zooming = False
        self.current_frame: Optional[np.ndarray] = None

    @property
    def num_hands(self) -> int:
        return 2

    @property
    def window_name(self) -> str:
        return "Camera Zoom"

    @property
    def intro_message(self) -> str:
        return "Use two hands: spread apart to zoom in, pinch together to zoom out."

    def on_result(
        self, result: HandLandmarkerResult, output_image: Any, timestamp_ms: int
    ) -> None:
        self.hands = result.hand_landmarks or []
        self.labels = get_all_handedness(result)

    def process(self, frame: np.ndarray) -> None:
        self.current_frame = frame.copy()

        if len(self.hands) < 2:
            self.is_zooming = False
            return

        hand1, hand2 = self.hands[0], self.hands[1]
        wrist1, wrist2 = hand1[0], hand2[0]
        current_distance = math.hypot(wrist2.x - wrist1.x, wrist2.y - wrist1.y)

        if not self.is_zooming:
            self.base_distance = current_distance
            self.is_zooming = True
        else:
            zoom_delta = current_distance / self.base_distance
            new_zoom = self.zoom_level * zoom_delta
            self.zoom_level = max(0.5, min(5.0, new_zoom))
            self.base_distance = current_distance

    def draw(self, frame: np.ndarray) -> None:
        if self.zoom_level > 1.0 and self.current_frame is not None:
            self._apply_zoom(frame)
        else:
            if len(self.hands) >= 2:
                self._draw_hands(frame)
            else:
                cv2.putText(
                    frame,
                    "Show two hands to zoom",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (200, 200, 200),
                    2,
                )

        self._draw_zoom_level(frame)

    def _apply_zoom(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        zoom = self.zoom_level

        new_h = int(h / zoom)
        new_w = int(w / zoom)

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        cropped = self.current_frame[top : top + new_h, left : left + new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        np.copyto(frame, zoomed)

    def _draw_hands(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        for hand in self.hands:
            wrist = hand[0]
            index_tip = hand[INDEX_TIP]
            cv2.circle(
                frame, (int(wrist.x * w), int(wrist.y * h)), 10, (0, 255, 255), -1
            )
            cv2.circle(
                frame,
                (int(index_tip.x * w), int(index_tip.y * h)),
                8,
                (255, 0, 255),
                -1,
            )

    def _draw_zoom_level(self, frame: np.ndarray) -> None:
        cv2.putText(
            frame,
            f"Zoom: {self.zoom_level:.2f}x",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        bar_width, bar_height = 200, 20
        fill_width = max(
            0, min(bar_width, int((self.zoom_level - 0.5) / 4.5 * bar_width))
        )

        cv2.rectangle(
            frame, (50, 70), (50 + bar_width, 70 + bar_height), (100, 100, 100), 2
        )
        cv2.rectangle(
            frame, (50, 70), (50 + fill_width, 70 + bar_height), (0, 255, 0), -1
        )


class HandApp:
    def __init__(self, mode: BaseMode) -> None:
        self.mode = mode

    def run(self) -> None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.mode.config.model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.mode.on_result,
            num_hands=self.mode.num_hands,
        )

        try:
            with mp.tasks.vision.HandLandmarker.create_from_options(
                options
            ) as landmarker:
                cap = cv2.VideoCapture(self.mode.config.camera_index, CAMERA_BACKEND)
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
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=frame_rgb
                    )

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


MODES = {
    "ball": BallMode,
    "number": NumberMode,
    "paint": PaintMode,
    "rps": RockPaperScissorsMode,
    "mouse": MouseMode,
    "zoom": ZoomMode,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand gesture controller")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="ball",
        help="Choose visualization mode (ball, number, paint, rps, mouse, zoom)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    mode_class = MODES.get(args.mode, BallMode)
    app = HandApp(mode_class(config))
    app.run()


if __name__ == "__main__":
    main()
