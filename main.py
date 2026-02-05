import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
FINGER_TIPS = [THUMB_TIP, INDEX_TIP, 12, 16, 20]
FINGER_PIPS = [THUMB_IP, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]


@dataclass(frozen=True)
class Config:
    model_path: str = "hand_landmarker.task"
    pinch_threshold: float = 0.05
    radius_scale: float = 300.0
    camera_index: int = 0
    brush_color: Tuple[int, int, int] = (0, 0, 255)
    brush_thickness: int = 10


@dataclass
class State:
    landmarks: Optional[Iterable] = None
    distance: float = 0.0
    finger_count: int = 0
    was_pinching: bool = False
    handedness: Optional[str] = None
    multi_hand_counts: List[Tuple[str, int]] = None
    multi_hand_landmarks: List[Iterable] = None
    canvas: Optional[np.ndarray] = None
    prev_brush_pos: Optional[Tuple[int, int]] = None


class HandApp:
    def __init__(self, mode: str, config: Config) -> None:
        self.mode = mode
        self.config = config
        self.state = State(multi_hand_counts=[], multi_hand_landmarks=[])
        if mode == "number":
            self.window_name = "Finger Count"
        elif mode == "paint":
            self.window_name = "Air Painter"
        else:
            self.window_name = "Gesture Controller"

    def on_hand_result(self, result, output_image, timestamp_ms) -> None:
        if self.mode == "number":
            self._update_multi_hand_counts(result)
            if not self.state.multi_hand_counts:
                self._clear_state()
            return

        hand = self._get_first_hand(result)
        if hand is None:
            self._clear_state()
            return

        self.state.landmarks = hand
        self.state.handedness = self._get_handedness(result)
        self.state.distance = self._pinch_distance(hand)
        if self.mode == "paint":
            self.state.finger_count = self.count_fingers(hand, self.state.handedness)

    def _clear_state(self) -> None:
        self.state.landmarks = None
        self.state.distance = 0.0
        self.state.finger_count = 0
        self.state.handedness = None
        self.state.multi_hand_counts = []
        self.state.multi_hand_landmarks = []
        self.state.prev_brush_pos = None

    @staticmethod
    def _get_first_hand(result):
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    @staticmethod
    def _get_handedness(result) -> Optional[str]:
        if result.handedness:
            return result.handedness[0][0].category_name
        return None

    @staticmethod
    def _get_all_handedness(result) -> List[str]:
        if not result.handedness:
            return []
        return [group[0].category_name for group in result.handedness]

    @staticmethod
    def _pinch_distance(hand) -> float:
        thumb_tip = hand[THUMB_TIP]
        index_tip = hand[INDEX_TIP]
        return math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    @staticmethod
    def count_fingers(hand) -> int:
        """Return number of extended fingers using a simple landmark heuristic."""
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

        palm_width = max(palm_width, 1e-6)
        distance_margin = max(palm_width * 0.05, 0.01)
        side_margin = max(palm_width * 0.15, 0.015)

        thumb_angle = angle(thumb_mcp, thumb_ip, thumb_tip)
        thumb_extended = (
            thumb_angle > 140.0
            and dist(thumb_tip, wrist) > dist(thumb_ip, wrist) + distance_margin
            and dist_xy(thumb_tip, palm_cx, palm_cy)
            > dist_xy(thumb_ip, palm_cx, palm_cy) + distance_margin
        )
        thumb_extended = thumb_extended and (abs(thumb_tip.x - palm_cx) > side_margin)

        count += thumb_extended

        def finger_extended(mcp_idx: int, pip_idx: int, tip_idx: int) -> bool:
            mcp = hand[mcp_idx]
            pip = hand[pip_idx]
            tip = hand[tip_idx]
            return (
                angle(mcp, pip, tip) > 160.0
                and dist(tip, wrist) > dist(pip, wrist) + distance_margin
            )

        for mcp_idx, pip_idx, tip_idx in zip(
            FINGER_MCPS[1:], FINGER_PIPS[1:], FINGER_TIPS[1:]
        ):
            count += finger_extended(mcp_idx, pip_idx, tip_idx)

        return int(count)

    def draw(self, frame) -> None:
        if self.mode == "number":
            self._draw_finger_count(frame)
            self._draw_finger_tips(frame)
            self._draw_palms(frame)
        elif self.mode == "paint":
            self._draw_canvas(frame)
            self._draw_brush(frame)
        else:
            self._draw_energy_ball(frame)

    def _draw_canvas(self, frame) -> None:
        if self.state.canvas is not None:
            # Create a mask of the non-zero pixels in the canvas
            canvas_gray = cv2.cvtColor(self.state.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Black out the area of the canvas in the frame
            img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            # Take only region of canvas from canvas image
            img_fg = cv2.bitwise_and(self.state.canvas, self.state.canvas, mask=mask)

            # Put canvas on frame
            np.copyto(frame, cv2.add(img_bg, img_fg))

    def _draw_brush(self, frame) -> None:
        if not self.state.landmarks:
            return

        h, w = frame.shape[:2]
        index = self.state.landmarks[INDEX_TIP]
        ix, iy = int(index.x * w), int(index.y * h)

        color = self.config.brush_color
        if self.state.distance < self.config.pinch_threshold:
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

    def _draw_energy_ball(self, frame) -> None:
        if not self.state.landmarks:
            return

        h, w = frame.shape[:2]
        thumb = self.state.landmarks[THUMB_TIP]
        index = self.state.landmarks[INDEX_TIP]

        thumb_px = (int(thumb.x * w), int(thumb.y * h))
        index_px = (int(index.x * w), int(index.y * h))
        center = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
        radius = int(self.state.distance * self.config.radius_scale)

        if self.state.distance < self.config.pinch_threshold:
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
            f"Dist: {self.state.distance:.3f}",
            (center[0] - 40, center[1] + radius + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    def _draw_finger_count(self, frame) -> None:
        if not self.state.multi_hand_counts:
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
        for label, count in self.state.multi_hand_counts:
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

    def _draw_finger_tips(self, frame) -> None:
        hands = self.state.multi_hand_landmarks
        if not hands and self.state.landmarks is not None:
            hands = [self.state.landmarks]
        if not hands:
            return

        h, w = frame.shape[:2]
        for hand in hands:
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

    def _draw_palms(self, frame) -> None:
        hands = self.state.multi_hand_landmarks
        if not hands and self.state.landmarks is not None:
            hands = [self.state.landmarks]
        if not hands:
            return

        h, w = frame.shape[:2]
        for hand in hands:
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

    def process(self, frame) -> None:
        if self.mode == "ball":
            is_pinching = (
                self.state.distance < self.config.pinch_threshold
                and self.state.landmarks is not None
            )
            if is_pinching and not self.state.was_pinching:
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            self.state.was_pinching = is_pinching
        elif self.mode == "paint":
            if self.state.canvas is None:
                self.state.canvas = np.zeros_like(frame)

            if not self.state.landmarks:
                self.state.prev_brush_pos = None
                return

            h, w = frame.shape[:2]
            index = self.state.landmarks[INDEX_TIP]
            ix, iy = int(index.x * w), int(index.y * h)

            if self.state.distance < self.config.pinch_threshold:
                if self.state.prev_brush_pos is not None:
                    cv2.line(
                        self.state.canvas,
                        self.state.prev_brush_pos,
                        (ix, iy),
                        self.config.brush_color,
                        self.config.brush_thickness,
                    )
                self.state.prev_brush_pos = (ix, iy)
            else:
                self.state.prev_brush_pos = None

            if self.state.finger_count >= 5:
                self.state.canvas = np.zeros_like(frame)

    def run(self) -> None:
        num_hands = 2 if self.mode == "number" else 1
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.config.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.on_hand_result,
            num_hands=num_hands,
        )

        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(self.config.camera_index)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return

            start_time = time.monotonic()
            if self.mode == "ball":
                print("System Ready. Pinch your fingers to take a screenshot!")
            elif self.mode == "paint":
                print("System Ready. Pinch to draw, show 5 fingers to clear!")
            else:
                print("System Ready. Show fingers to display a number!")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                landmarker.detect_async(mp_image, timestamp_ms)

                self.process(frame)
                self.draw(frame)

                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

    def _update_multi_hand_counts(self, result) -> None:
        hands = result.hand_landmarks or []
        labels = self._get_all_handedness(result)
        counts: List[Tuple[str, int]] = []
        for idx, hand in enumerate(hands):
            label = labels[idx] if idx < len(labels) else f"Hand {idx + 1}"
            counts.append((label, self.count_fingers(hand, label)))
        self.state.multi_hand_counts = counts
        self.state.multi_hand_landmarks = hands
        if hands:
            self.state.landmarks = hands[0]


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
    app = HandApp(args.mode, Config())
    app.run()


if __name__ == "__main__":
    main()
