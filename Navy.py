import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Pose detection setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# Condensed Semaphore angle dictionary using angle ranges
semaphore_dict = {
    ((165, 175), (-145, -135)): 'A',
    ((165, 175), (-95, -85)): 'B',
    ((165, 175), (-45, -35)): 'C',
    ((165, 175), (-5, 5)): 'D',
    ((35, 45), (-175, -165)): 'E',
    ((85, 95), (-175, -165)): 'F',
    ((135, 145), (-175, -165)): 'G',
    ((-145, -135), (-95, -85)): 'H',
    ((-35, -25), (-145, -135)): 'I',
    ((85, 95), (-5, 5)): 'J',
    ((-5, 5), (-145, -135)): 'K',
    ((25, 35), (-145, -135)): 'L',
    ((85, 95), (-145, -135)): 'M',
    ((135, 145), (-145, -135)): 'N',
    ((-35, -25), (-95, -85)): 'O',
    ((-5, 5), (-95, -85)): 'P',
    ((25, 35), (-95, -85)): 'Q',
    ((85, 95), (-95, -85)): 'R',
    ((125, 135), (-95, -85)): 'S',
    ((-5, 5), (-35, -25)): 'T',
    ((25, 35), (-35, -25)): 'U',
    ((135, 145), (-5, 5)): 'V',
    ((85, 95), (45, 55)): 'W',
    ((135, 145), (55, 65)): 'X',
    ((85, 95), (25, 35)): 'Y',
    ((85, 95), (135, 145)): 'Z',
    ((-165, -165), (165, 165)): ' '  # Space
}

def calculate_shoulder_angle_signed(shoulder, wrist):
    """Calculate signed angle from vertical axis (up = +90°, down = -90°)"""
    dx = wrist[0] - shoulder[0]
    dy = shoulder[1] - wrist[1]  # Invert Y because OpenCV Y axis increases downward

    angle_rad = np.arctan2(dx, dy)  # dx first → horizontal first
    angle_deg = np.degrees(angle_rad)

    return round(angle_deg)

def match_letter(l_angle, r_angle):
    """Match angle pair to letter based on range dictionary."""
    for (l_range, r_range), letter in semaphore_dict.items():
        if l_range[0] <= l_angle <= l_range[1] and r_range[0] <= r_angle <= r_range[1]:
            return letter
    return None

# State management
angle_buffer = deque(maxlen=10)
pose_hold_start = None
last_letter = None
decoded_message = ""

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Get keypoints
        ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
              lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        lw = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
              lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        rs = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
              lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        rw = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
              lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angles
        left_angle = calculate_shoulder_angle_signed(ls, lw)
        right_angle = calculate_shoulder_angle_signed(rs, rw)

        angle_buffer.append((left_angle, right_angle))

        # Check if angles are steady
        if all(abs(a[0] - left_angle) < 8 and abs(a[1] - right_angle) < 8 for a in angle_buffer):
            if pose_hold_start is None:
                pose_hold_start = time.time()

            if time.time() - pose_hold_start > 1.5:  # Hold duration
                letter = match_letter(left_angle, right_angle)
                if letter and letter != last_letter:
                    decoded_message += letter
                    last_letter = letter
                    print("Decoded:", decoded_message)
        else:
            pose_hold_start = None
            last_letter = None

        # Display info
        cv2.putText(image, f"Left Angle: {left_angle}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Right Angle: {right_angle}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Message: {decoded_message}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Semaphore Recognition", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
