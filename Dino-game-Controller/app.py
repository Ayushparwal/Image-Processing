import cv2
import mediapipe as mp
import pyautogui

# Initialize camera and Mediapipe Hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_height = 480

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the Y-coordinate of the index finger
            index_y = int(hand_landmarks.landmark[8].y * screen_height)
            
            # Jump if hand is raised
            if index_y < screen_height // 3:
                pyautogui.press('space')
                print("Jump")
            
            # Duck if hand is lowered
            elif index_y > screen_height * 2 // 3:
                pyautogui.keyDown('down')
                print("Duck")
            else:
                pyautogui.keyUp('down')

    cv2.imshow("Dino Game Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()