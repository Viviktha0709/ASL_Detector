import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --------------------
# Load model once
# --------------------
@st.cache_resource
def load_asl_model():
    model = load_model("asl_transfer_model.h5")
    classes = sorted([
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ])

    return model, classes

model, classes = load_asl_model()

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# --------------------
# Streamlit UI
# --------------------
st.title("🤟 ASL Real-Time Detector")
st.write("Show your hand to the webcam and get real-time ASL predictions (A-Z, 0-9).")

# --------------------
# Video transformer
# --------------------
class ASLTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            h, w, c = img.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in result.multi_hand_landmarks[0].landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            pad = 20
            x_min, y_min = max(0, x_min-pad), max(0, y_min-pad)
            x_max, y_max = min(w, x_max+pad), min(h, y_max+pad)

            roi = img[y_min:y_max, x_min:x_max]
            if roi.size != 0:
                roi_resized = cv2.resize(roi, (128, 128)) / 255.0
                roi_expanded = np.expand_dims(roi_resized, axis=0)

                prediction = model.predict(roi_expanded, verbose=0)
                class_index = int(np.argmax(prediction))
                predicted_class = classes[class_index]
                confidence = float(prediction[0][class_index])

                # Show label on frame
                label = f"{predicted_class} ({confidence:.2%})"
                cv2.putText(img, label, (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

        return img

# Start webcam streaming
webrtc_streamer(key="asl", video_transformer_factory=ASLTransformer)
