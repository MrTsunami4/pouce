import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(
    result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    # print the distance between the tip of the thumb (4) and the tip of the index finger (8).
    if not result.hand_landmarks:
        print("No hands detected.")
        return

    hand_landmarks = result.hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_finger_tip = hand_landmarks[8]
    distance = (
        (thumb_tip.x - index_finger_tip.x) ** 2
        + (thumb_tip.y - index_finger_tip.y) ** 2
    ) ** 0.5
    print(f"Distance between thumb tip and index finger tip: {distance:.4f}")


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=1,
)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        landmarker.detect_async(mp_image, timestamp_ms)

        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
