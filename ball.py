import math
import time

import cv2
import mediapipe as mp

# Configuration
MODEL_PATH = "hand_landmarker.task"
PINCH_THRESHOLD = 0.05
RADIUS_SCALE = 300.0

# Shared state
landmarks = None
distance = 0.0
was_pinching = False


def on_hand_result(result, output_image, timestamp_ms):
    """Callback when hand landmarks are detected."""
    global landmarks, distance

    if not result.hand_landmarks:
        landmarks = None
        distance = 0.0
        return

    hand = result.hand_landmarks[0]
    landmarks = hand

    # Calculate distance between thumb and index finger
    thumb_tip = hand[4]
    index_tip = hand[8]
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
    )


def draw_energy_ball(frame):
    """Draw the energy ball between thumb and index finger."""
    if not landmarks:
        return

    h, w = frame.shape[:2]
    thumb = landmarks[4]
    index = landmarks[8]

    # Convert to pixel coordinates
    thumb_px = (int(thumb.x * w), int(thumb.y * h))
    index_px = (int(index.x * w), int(index.y * h))

    # Calculate center and radius
    center = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
    radius = int(distance * RADIUS_SCALE)

    # Draw based on pinch state
    if distance < PINCH_THRESHOLD:
        cv2.circle(frame, center, radius, (0, 255, 0), -1)
        cv2.putText(
            frame, "PINCH ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
    else:
        cv2.line(frame, thumb_px, index_px, (200, 200, 200), 2)
        cv2.circle(frame, center, radius, (255, 255, 0), 3)

    cv2.putText(
        frame,
        f"Dist: {distance:.3f}",
        (center[0] - 40, center[1] + radius + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


def main():
    """Main application loop."""
    global was_pinching

    # Setup hand landmarker
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=on_hand_result,
        num_hands=1,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        start_time = time.time()
        print("System Ready. Pinch your fingers to take a screenshot!")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detect hands
            timestamp_ms = int((time.time() - start_time) * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            # Check for pinch gesture and take screenshot
            is_pinching = distance < PINCH_THRESHOLD and landmarks is not None
            if is_pinching and not was_pinching:
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            was_pinching = is_pinching

            # Draw visualization
            draw_energy_ball(frame)

            cv2.imshow("Gesture Controller", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
