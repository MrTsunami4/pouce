import math
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass(frozen=True)
class HandTrackingConfig:
    model_path: str = "hand_landmarker.task"
    num_hands: int = 1
    pinch_threshold: float = 0.05
    radius_scale: float = 300.0


class HandTrackingState:
    def __init__(self) -> None:
        self.landmarks = None
        self.distance = 0.0

    def update_from_result(self, result: HandLandmarkerResult) -> None:
        if not result.hand_landmarks:
            self.landmarks = None
            self.distance = 0.0
            return

        hand_landmarks = result.hand_landmarks[0]
        self.landmarks = hand_landmarks

        thumb_tip = hand_landmarks[4]
        index_finger_tip = hand_landmarks[8]
        self.distance = math.sqrt(
            (thumb_tip.x - index_finger_tip.x) ** 2
            + (thumb_tip.y - index_finger_tip.y) ** 2
        )


class HandTracker:
    def __init__(self, config: HandTrackingConfig, state: HandTrackingState) -> None:
        self._config = config
        self._state = state
        self._landmarker = None
        self._options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            num_hands=config.num_hands,
        )

    def _on_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        self._state.update_from_result(result)

    def __enter__(self) -> "HandTracker":
        self._landmarker = HandLandmarker.create_from_options(self._options)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def detect_async(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        if not self._landmarker:
            raise RuntimeError("HandTracker is not initialized.")
        self._landmarker.detect_async(mp_image, timestamp_ms)


class EnergyBallRenderer:
    def __init__(self, config: HandTrackingConfig) -> None:
        self._config = config

    def render(self, frame, state: HandTrackingState) -> None:
        if not state.landmarks:
            return

        h, w, _ = frame.shape
        thumb = state.landmarks[4]
        index = state.landmarks[8]

        thumb_px = (int(thumb.x * w), int(thumb.y * h))
        index_px = (int(index.x * w), int(index.y * h))

        center_x = (thumb_px[0] + index_px[0]) // 2
        center_y = (thumb_px[1] + index_px[1]) // 2

        radius = int(state.distance * self._config.radius_scale)

        if state.distance < self._config.pinch_threshold:
            color = (0, 255, 0)
            thickness = -1
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
            color = (255, 255, 0)
            thickness = 3
            cv2.line(frame, thumb_px, index_px, (200, 200, 200), 2)

        cv2.circle(frame, (center_x, center_y), radius, color, thickness)
        cv2.putText(
            frame,
            f"Dist: {state.distance:.3f}",
            (center_x - 40, center_y + radius + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )


class WebcamApp:
    def __init__(self, config: HandTrackingConfig) -> None:
        self._config = config
        self._state = HandTrackingState()
        self._renderer = EnergyBallRenderer(config)

    def run(self) -> None:
        with HandTracker(self._config, self._state) as tracker:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return

            start_time = time.time()
            print("System Ready. Pinch your fingers!")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                timestamp_ms = int((time.time() - start_time) * 1000)
                tracker.detect_async(mp_image, timestamp_ms)

                self._renderer.render(frame, self._state)

                cv2.imshow("Gesture Controller", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    config = HandTrackingConfig()
    app = WebcamApp(config)
    app.run()


if __name__ == "__main__":
    main()
