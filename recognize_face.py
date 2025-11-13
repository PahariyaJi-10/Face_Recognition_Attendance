import cv2
import numpy as np
import csv
from datetime import datetime
import threading

# ---------- Load trained model ----------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"C:\Users\Lenovo\face_model.yml")  # <-- Your path

# ---------- Load labels ----------
labels_path = r"C:\Users\Lenovo\labels.npy"  # <-- Your path
labels = np.load(labels_path, allow_pickle=True).item()
labels = {v: k for k, v in labels.items()}

# ---------- Get Divyansh's ID ----------
divyansh_id = labels.get("Divyansh", None)
if divyansh_id is None:
    print("Error: Divyansh not found in labels!")
    exit()

# ---------- Attendance function ----------
def mark_attendance(name):
    today_date = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")
    filename = "attendance.csv"

    try:
        with open(filename, "r", newline="") as f:
            existing = [line.split(",")[0] for line in f.readlines()]
    except FileNotFoundError:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])
        existing = []

    if name not in existing:
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, today_date, time_now])

# ---------- Load Haarcascade ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------- Start webcam ----------
cap = cv2.VideoCapture(0)
CONF_THRESHOLD = 50
attendance_marked = False
stop_flag = False  # Thread will set this to stop the loop

# ---------- Thread function to watch for CMD input ----------
def watch_for_quit():
    global stop_flag
    while True:
        cmd = input()
        if cmd.lower() == 'q':
            stop_flag = True
            break

# Start the thread
threading.Thread(target=watch_for_quit, daemon=True).start()
print("Press 'q' in CMD to quit at any time.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            name = "Unknown"
            roi_gray = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi_gray)

            if confidence < CONF_THRESHOLD and id_ == divyansh_id:
                name = "Divyansh"
                if not attendance_marked:
                    mark_attendance(name)
                    print(f"[INFO] Attendance marked for {name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    attendance_marked = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Conf: {int(confidence)}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("Divyansh Face Recognition & Attendance", frame)

        # ---------- Check if CMD quit flag is set ----------
        if stop_flag:
            print("[INFO] Quitting program...")
            break

        # Optional: also allow ESC or 'q' in OpenCV window
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            print("[INFO] Quitting program (from OpenCV window)...")
            break

except KeyboardInterrupt:
    print("[INFO] Program interrupted manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released and windows closed.")
