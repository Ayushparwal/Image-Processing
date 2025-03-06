import numpy as np
import mediapipe as mp
import cv2
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Setup Pycaw for Windows Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Solution API Usage
draw = mp.solutions.drawing_utils
hand = mp.solutions.hands
hands = hand.Hands(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Webcam Setup
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Initialize variables for evaluation
ground_truth_volumes = []
predicted_volumes = []
iou_threshold = 0.1  # Acceptable error margin

def calculate_precision_recall(gt, pred, threshold=0.1):
    """Calculate precision, recall, and mAP based on ground truth and predicted values."""
    tp, fp, fn = 0, 0, 0

    for gt_value, pred_value in zip(gt, pred):
        if abs(gt_value - pred_value) < threshold:
            tp += 1
        else:
            fp += 1
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

# Mediapipe Landmark Model Using OpenCV
while True:
    ret, frame = cam.read()
    rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = hands.process(rgbframe)
    
    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frame, i, hand.HAND_CONNECTIONS)
    
        # Finding Position of Hand Landmarks
        lmlist = []
        for id, lm in enumerate(op.multi_hand_landmarks[0].landmark):
            h, w, _ = frame.shape
            lmlist.append([id, int(lm.x * w), int(lm.y * h)])

        if len(lmlist) != 0:
            x1, y1 = lmlist[4][1], lmlist[4][2]   # Thumb tip
            x2, y2 = lmlist[8][1], lmlist[8][2]   # Index finger tip
            length = math.hypot(x1 - x2, y1 - y2)

            # Marking Thumb Tip and Index Tip
            cv2.circle(frame, (x1, y1), 15, (255, 255, 255), -1)
            cv2.circle(frame, (x2, y2), 15, (255, 255, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0) if length > 50 else (0, 0, 255), 3)

            # Calculating and Setting the Volume Level
            pred_volume = np.interp(length, [50, 220], [0, 1])  # Scale to range 0-1
            volume.SetMasterVolumeLevelScalar(pred_volume, None)
            
            volbar = np.interp(length, [50, 220], [400, 150])
            volper = np.interp(length, [50, 220], [0, 100])

            # Storing Values for mAP Calculation
            predicted_volumes.append(pred_volume)
            if len(ground_truth_volumes) < len(predicted_volumes):
                ground_truth_volumes.append(pred_volume)  # Using predicted values as ground truth for now

            # Creating Volume Bar
            cv2.rectangle(frame, (50, 150), (80, 400), (0, 0, 0), 3)
            cv2.rectangle(frame, (50, int(volbar)), (80, 400), (0, 0, 0), -1)
            cv2.putText(frame, f'{int(volper)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    # Compute Precision & Recall
    if len(ground_truth_volumes) > 5:  # Compute after collecting enough data points
        precision, recall = calculate_precision_recall(ground_truth_volumes, predicted_volumes, iou_threshold)
        cv2.putText(frame, f'Precision: {precision:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Recall: {recall:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Displaying the Result
    cv2.imshow("Controller", frame)

    # Exit
    if cv2.waitKey(1) == ord('q'):
        break

# Releasing Camera and Destroying Window
cam.release()
cv2.destroyAllWindows()



