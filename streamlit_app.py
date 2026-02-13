import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
from main import Config, BallMode, NumberMode, PaintMode, RockPaperScissorsMode, HandUtils

# RTC configuration for STUN servers (needed for some networks)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def __init__(self, mode_name):
        self.config = Config()
        self.mode_name = mode_name
        self.mode = self._get_mode(mode_name)
        
        # Initialize MediaPipe Landmarker in IMAGE mode for synchronous processing
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.config.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=self.mode.num_hands,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _get_mode(self, mode_name):
        modes = {
            "ball": BallMode,
            "number": NumberMode,
            "paint": PaintMode,
            "rps": RockPaperScissorsMode,
        }
        return modes[mode_name](self.config)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror image for natural interaction
        img = cv2.flip(img, 1)
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Synchronous detection
        result = self.landmarker.detect(mp_image)
        
        # Update mode state with results
        self.mode.on_result(result, None, 0)
        
        # Process and draw
        self.mode.process(img)
        self.mode.draw(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="Pouce 🖐️ - Hand Gesture Control", layout="wide")
    
    st.title("Pouce 🖐️")
    st.markdown("""
    Real-time hand gesture control interface built with MediaPipe and OpenCV.
    Choose a mode from the sidebar and start interacting!
    """)

    mode_options = {
        "Energy Ball & Screenshot": "ball",
        "Finger Count": "number",
        "Air Painter": "paint",
        "Rock Paper Scissors": "rps"
    }
    
    st.sidebar.title("Settings")
    selected_mode_label = st.sidebar.selectbox("Select Mode", list(mode_options.keys()))
    mode_name = mode_options[selected_mode_label]

    # Description of modes
    descriptions = {
        "ball": "Pinch thumb and index to create an energy ball. A screenshot is saved when you pinch.",
        "number": "Count fingers on one or two hands.",
        "paint": "Pinch to draw. Show all 5 fingers to clear the canvas.",
        "rps": "Play Rock Paper Scissors against the CPU! Countdown starts when hand is detected."
    }
    st.info(descriptions[mode_name])

    # Center and resize the video feed using columns
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
