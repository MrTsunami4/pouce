import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoTransformerBase, webrtc_streamer

from main import (
    BallMode,
    Config,
    MouseMode,
    NumberMode,
    PaintMode,
    RockPaperScissorsMode,
    ZoomMode,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor(VideoTransformerBase):
    def __init__(self, mode_name: str) -> None:
        self.config = Config()
        self.mode_name = mode_name
        self.mode = self._create_mode(mode_name)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.config.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=self.mode.num_hands,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _create_mode(self, mode_name: str):
        modes = {
            "ball": BallMode,
            "number": NumberMode,
            "paint": PaintMode,
            "rps": RockPaperScissorsMode,
            "mouse": MouseMode,
            "zoom": ZoomMode,
        }
        return modes[mode_name](self.config)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        result = self.landmarker.detect(mp_image)
        self.mode.on_result(result, None, 0)
        self.mode.process(img)
        self.mode.draw(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.set_page_config(page_title="Pouce - Hand Gesture Control", layout="wide")

    st.title("Pouce")
    st.markdown("""
    Real-time hand gesture control interface built with MediaPipe and OpenCV.
    Choose a mode from the sidebar and start interacting!
    """)

    mode_options = {
        "Energy Ball & Screenshot": "ball",
        "Finger Count": "number",
        "Air Painter": "paint",
        "Rock Paper Scissors": "rps",
        "Virtual Mouse": "mouse",
        "Camera Zoom": "zoom",
    }

    st.sidebar.title("Settings")
    selected_mode_label = st.sidebar.selectbox("Select Mode", list(mode_options.keys()))
    mode_name = mode_options[selected_mode_label]

    descriptions = {
        "ball": "Pinch thumb and index to create an energy ball. A screenshot is saved when you pinch.",
        "number": "Count fingers on one or two hands.",
        "paint": "Pinch to draw. Show all 5 fingers to clear the canvas.",
        "rps": "Play Rock Paper Scissors against the CPU! Countdown starts when hand is detected.",
        "mouse": "Move hand to control cursor. Pinch to click. Two hands for drag.",
        "zoom": "Use two hands: spread apart to zoom in, pinch together to zoom out.",
    }
    st.info(descriptions[mode_name])

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        webrtc_streamer(
            key=f"pouce-{mode_name}",
            video_processor_factory=lambda: VideoProcessor(mode_name),
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )


if __name__ == "__main__":
    main()
