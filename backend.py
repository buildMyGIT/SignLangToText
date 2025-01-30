import cv2
import mediapipe as mp
import json
import numpy as np

# Load the reference data from JSON
with open('reference.json', 'r') as f:
    reference_data = json.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Euclidean distance
def calculate_distance(landmarks1, landmarks2):
    distances = []
    for lm1, lm2 in zip(landmarks1, landmarks2):
        dist = np.sqrt((lm1[0] - lm2[0]) * 2 + (lm1[1] - lm2[1]) * 2 + (lm1[2] - lm2[2]) ** 2)
        distances.append(dist)
    return np.mean(distances)

# Function to recognize the closest sign
def recognize_sign(input_landmarks, reference_data):
    best_match = None
    min_distance = float('inf')

    for word, frames in reference_data.items():
        for frame in frames:
            ref_landmarks = frame["Left Hand Coordinates"] if frame["Left Hand Coordinates"] else frame["Right Hand Coordinates"]
            ref_landmarks = [lm["Coordinates"] for lm in ref_landmarks]
            
            if len(ref_landmarks) == len(input_landmarks):  # Ensure dimensions match
                distance = calculate_distance(input_landmarks, ref_landmarks)
                if distance < min_distance:
                    min_distance = distance
                    best_match = word
    
    return best_match, min_distance

# Initialize webcam
#cap = cv2.VideoCapture(0)

# Process video frames
def process_frame(frame):
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip and preprocess frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hand landmark detection
            results = hands.process(rgb_frame)

            # Process landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    input_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                    # Recognize the sign
                    recognized_word, distance = recognize_sign(input_landmarks, reference_data)
                    
                    # Display recognized word
                    if recognized_word:
                        cv2.putText(frame, f"Word: {recognized_word} (Dist: {distance:.2f})", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the frame
            return cv2.imshow('-Text', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):  # Exit on 'q'
                break

            cap.release()
            cv2.destroyAllWindows()