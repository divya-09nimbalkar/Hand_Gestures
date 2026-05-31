import cv2
import mediapipe as mp
import numpy as np
import math
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')

# Initialize MediaPipe Hands (NEW Task API for v0.10+)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_tracking_confidence=0.20,
    min_hand_detection_confidence=0.20
)
hands = HandLandmarker.create_from_options(options)
mp_draw = mp.tasks.vision.drawing_utils

# Create canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables
prev_x, prev_y = 0, 0
frame_count = 0  # For timestamping frames in video mode

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Start camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows fix

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = hands.detect_for_video(mp_image, frame_count)
    frame_count += 1

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Get landmarks
            index_finger = hand_landmarks[8]  # INDEX_FINGER_TIP = 8
            thumb_tip = hand_landmarks[4]  # THUMB_TIP = 4

            ix, iy = int(index_finger.x * w), int(index_finger.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            distance = euclidean_distance((ix, iy), (tx, ty))

            # Dynamic threshold (better than fixed 40)
            if distance < 0.05 * w:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy

                # Draw line
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), (255, 0, 0), 5)
                prev_x, prev_y = ix, iy
            else:
                prev_x, prev_y = 0, 0

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)

    # Merge canvas and frame
    frame = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)

    # Instructions
    cv2.putText(frame, "Pinch (Thumb + Index) to Draw", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'C' to Clear | 'Q' to Quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Virtual Painter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()