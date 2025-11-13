import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
import os

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

# ------------------- CONFIG -------------------
EYE_AR_THRESH = 0.28       # Easier blink detection (increase if not detected)
EYE_AR_CONSEC_FRAMES = 2
REQUIRED_BLINKS = 1
MOTION_DIFF_THRESH = 2.5   # MUCH more sensitive to slight head motion
LIVE_MOVEMENT_FRAMES = 2

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError("Missing shape_predictor_68_face_landmarks.dat file!")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

blink_counter = 0
total_blinks = 0
live_detected = False
motion_frames = 0
prev_face_roi = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("[INFO] Starting improved liveness detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Blink detection
        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
                print(f"[INFO] Blink detected! Total: {total_blinks}")
            blink_counter = 0

        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        face_roi = gray[y:y+h, x:x+w]

        # Check small motion
        if prev_face_roi is not None and face_roi.shape == prev_face_roi.shape:
            diff = cv2.absdiff(face_roi, prev_face_roi)
            diff_mean = np.mean(diff)
            if diff_mean > MOTION_DIFF_THRESH:
                motion_frames += 1
                print(f"[INFO] Motion detected! Frame diff mean: {diff_mean:.2f}")
        prev_face_roi = face_roi.copy()

        # Show counters
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Motion: {motion_frames}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if len(faces) == 0:
        prev_face_roi = None

    # Check if both blink & movement detected
    if total_blinks >= REQUIRED_BLINKS and motion_frames >= LIVE_MOVEMENT_FRAMES:
        live_detected = True

    cv2.imshow("Improved Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n========== FINAL RESULT ==========")
if live_detected:
    print("✅ Live person detected.")
else:
    print("❌ Possible photo detected. No natural motion or blinks.")
