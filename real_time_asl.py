import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


#loading the model
model = load_model("asl_transfer_model.h5")

#Class Mapping
classes = sorted([
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
])


# VideoCapture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

#Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Real-Time Prediction Loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #mirror your movement
    #frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #detect the hands
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, c = frame.shape
            x_min = w
            y_min = h
            x_max = y_max = 0
            for im in hand_landmarks.landmark:
                x, y = int(im.x * w), int(im.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            pad = 20
            x_min, y_min = max(0, x_min-pad), max(0, y_min-pad)
            x_max, y_max = min(w, x_max+pad), min(h, y_max+pad)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size != 0:
                roi_resized = cv2.resize(roi, (128, 128))
                roi_normalized = roi_resized / 255.0
                roi_expanded = np.expand_dims(roi_normalized, axis=0)

                prediction = model.predict(roi_expanded)
                class_index = np.argmax(prediction)
                predicted_class = classes[class_index]
                confidence = prediction[0][class_index]

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
                cv2.putText(frame, f"{predicted_class} ({confidence*100:.1f}%)", 
                            (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0,255,0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    """
    #define ROI box
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    #preprocess ROI box
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)


    #Predict
    prediction = model.predict(roi_expanded)
    class_index = np.argmax(prediction)
    predicted_class = classes[class_index]
    confidence = prediction[0][class_index]

    #Display Predictions on the frame
    cv2.putText(frame, f"Pred: {predicted_class} ({confidence*100:.1f}%)",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    #Draw ROI box
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    #Show frame
    cv2.imshow("ASL Real-Time", frame)

    #Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        """
    
    cv2.imshow("ASL Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release resources
cap.release()
cv2.destroyAllWindows()