import mediapipe as mp
import pyautogui
from math import hypot
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural hand movement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger and thumb coordinates
            index_x, index_y = int(hand_landmarks.landmark[8].x * screen_width), int(hand_landmarks.landmark[8].y * screen_height)
            thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * screen_width), int(hand_landmarks.landmark[4].y * screen_height)

            # Move the mouse
            pyautogui.moveTo(index_x, index_y, duration=0.1)

            # Detect click (index finger & thumb close together)
            distance = hypot(index_x - thumb_x, index_y - thumb_y)
            if distance < 40:
                pyautogui.click()
                pyautogui.sleep(0.2)  # Prevent multiple clicks

    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
