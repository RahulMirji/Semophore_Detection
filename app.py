from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
camera = None
is_camera_running = False
decoded_message = ""
current_angles = {"left": 0, "right": 0}

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Semaphore dictionary (using the same one from Navy.py)
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
    ((-165, -165), (165, 165)): ' '
}

def initialize_camera():
    global camera
    logger.info("Initializing camera...")
    try:
        # Try different camera indices
        for i in range(3):  # Try first 3 camera indices
            logger.info(f"Trying camera index {i}")
            camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow backend on Windows
            if camera.isOpened():
                logger.info(f"Successfully opened camera at index {i}")
                # Set camera properties
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Test if we can read a frame
                ret, frame = camera.read()
                if ret:
                    logger.info("Successfully read test frame from camera")
                    return True
                else:
                    logger.error(f"Could not read frame from camera at index {i}")
                    camera.release()
            else:
                logger.error(f"Could not open camera at index {i}")
                if camera is not None:
                    camera.release()
                    camera = None
    except Exception as e:
        logger.error(f"Error initializing camera: {str(e)}")
        if camera is not None:
            camera.release()
            camera = None
    return False

def calculate_shoulder_angle_signed(shoulder, wrist):
    dx = wrist[0] - shoulder[0]
    dy = shoulder[1] - wrist[1]
    angle_rad = np.arctan2(dx, dy)
    angle_deg = np.degrees(angle_rad)
    return round(angle_deg)

def match_letter(l_angle, r_angle):
    for (l_range, r_range), letter in semaphore_dict.items():
        if l_range[0] <= l_angle <= l_range[1] and r_range[0] <= r_angle <= r_range[1]:
            return letter
    return None

def generate_frames():
    global camera, decoded_message, current_angles
    
    angle_buffer = deque(maxlen=10)
    pose_hold_start = None
    last_letter = None
    
    logger.info("Starting frame generation")
    
    while True:
        if not is_camera_running:
            logger.info("Camera not running, stopping frame generation")
            if camera is not None:
                logger.info("Releasing camera")
                camera.release()
                camera = None
            break
            
        if camera is None:
            logger.info("Camera is None, attempting to initialize...")
            if not initialize_camera():
                logger.error("Failed to initialize camera in generate_frames")
                # Create a black frame with error message
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Error: Could not initialize camera", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                break
                
        try:
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame from camera")
                # Create a black frame with error message
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Error: Could not read frame", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                logger.debug("Successfully read frame from camera")
                # Process the frame for pose detection
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
                    current_angles["left"] = left_angle
                    current_angles["right"] = right_angle
                    
                    # Display angles and message on frame
                    cv2.putText(image, f"Left Angle: {left_angle}°", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Right Angle: {right_angle}°", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Message: {decoded_message}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    angle_buffer.append((left_angle, right_angle))
                    
                    if all(abs(a[0] - left_angle) < 8 and abs(a[1] - right_angle) < 8 for a in angle_buffer):
                        if pose_hold_start is None:
                            pose_hold_start = time.time()
                        
                        if time.time() - pose_hold_start > 1.5:
                            letter = match_letter(left_angle, right_angle)
                            if letter and letter != last_letter:
                                decoded_message += letter
                                last_letter = letter
                    else:
                        pose_hold_start = None
                        last_letter = None
                
                frame = image
            
            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                logger.error("Failed to encode frame")
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        except Exception as e:
            logger.error(f"Error in generate_frames: {str(e)}")
            break

@app.route('/')
def index():
    logger.info("Rendering index page")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    logger.info("Starting video feed")
    try:
        return Response(generate_frames(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video_feed route: {str(e)}")
        return "Error generating video feed", 500

@app.route('/start')
def start_camera():
    global is_camera_running, decoded_message
    logger.info("Starting camera...")
    try:
        if not initialize_camera():
            logger.error("Failed to initialize camera in start_camera")
            return jsonify({"status": "error", "message": "Failed to initialize camera"})
        is_camera_running = True
        decoded_message = ""  # Reset message when starting
        logger.info("Camera started successfully")
        return jsonify({"status": "started"})
    except Exception as e:
        logger.error(f"Error in start_camera: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop')
def stop_camera():
    global is_camera_running
    logger.info("Stopping camera...")
    is_camera_running = False
    return jsonify({"status": "stopped"})

@app.route('/get_data')
def get_data():
    global decoded_message, current_angles
    return jsonify({
        "message": decoded_message,
        "angles": current_angles
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True) 