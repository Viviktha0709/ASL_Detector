import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from mediapipe import solutions as mp_solutions
from CNNModel import CNNModel, class_labels

#loading the trained model
model_path = 'C:\\Users\\Viviktha\\OneDrive\\Desktop\\Projects\\ISL Project\\model_weights.pth'  # Specify the path where you want to save the model
model = CNNModel(num_classes= len(class_labels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location = device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mp_hands = mp_solutions.hands
mp_drawing = mp_solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode = False, min_detection_confidence = 0.7)

cap = cv2.VideoCapture(0)

label_mapping = {i: label for i, label in enumerate(class_labels)}  # class_labels from training dataset

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = []
        x_min, y_min = frame.shape[1], frame.shape[0]
        x_max, y_max = 0, 0

        for hand_landmark in results.multi_hand_landmarks:
            for lm in hand_landmark.landmark:

                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                hand_landmarks.append([x, y])

                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)


        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        hand_roi = frame[y_min:y_max, x_min:x_max]

        if hand_roi.size > 0:
            hand_roi_resized = cv2.resize(hand_roi, (64, 64))
            img_tensor = transform(hand_roi_resized).unsqueeze(0)
            img_tensor = img_tensor.float().to(device)


            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_label_idx = torch.argmax(probabilities, dim=1).item()
                predicted_label = label_mapping[predicted_label_idx]

            cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
