import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from utils.data_loader import load_data  # Add this

# Load model
model = tf.keras.models.load_model('models/best_model1.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Get class labels from training data
_, val_generator = load_data()
class_labels = list(val_generator.class_indices.keys())  # Dynamic labels


def preprocess_hand_roi(image):
    """Preprocess detected hand region"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Match IMG_SIZE from config.py
    image = image / 255.0
    return np.expand_dims(image, axis=0)


def real_time_prediction():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            x_min = int(min([lm.x * w for lm in hand_landmarks.landmark]))
            y_min = int(min([lm.y * h for lm in hand_landmarks.landmark]))
            x_max = int(max([lm.x * w for lm in hand_landmarks.landmark]))
            y_max = int(max([lm.y * h for lm in hand_landmarks.landmark]))

            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size != 0:
                try:
                    processed_img = preprocess_hand_roi(hand_roi)
                    pred = model.predict(processed_img, verbose=0)
                    predicted_class = class_labels[np.argmax(pred)]
                    confidence = np.max(pred)

                    # Display prediction only if confidence > 70%
                    if confidence > 0.7:
                        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                                    (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                except:
                    pass

        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_prediction()