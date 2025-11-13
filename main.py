# main.py
import os
import cv2
import face_recognition
import pickle
import csv
import time
import numpy as np
from datetime import datetime

# ---------- CONFIG ----------
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
CAPTURE_COUNT = 20        # number of images to capture per student
FRAME_SCALE = 0.25        # scale for faster processing when encoding/recognizing

os.makedirs(DATASET_DIR, exist_ok=True)

# ---------- Helper: Capture images for new student ----------
def capture_new_student():
    name = input("Enter student's name (no slashes): ").strip()
    if not name:
        print("Name cannot be empty.")
        return
    # create person folder
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print(f"[INFO] Capturing {CAPTURE_COUNT} images for '{name}'. Press SPACE to capture, ESC to stop early.")
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break
        disp = frame.copy()
        cv2.putText(disp, f"{name} - Captured: {count}/{CAPTURE_COUNT}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Capture - Press SPACE", disp)
        key = cv2.waitKey(1) & 0xFF

        # space to capture
        if key == 32:
            # detect face and crop to center the saved image if possible
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            if boxes:
                top,right,bottom,left = boxes[0]  # take first face
                face_img = frame[top:bottom, left:right]
                # if face region is too small or empty, fallback to full frame
                if face_img.size == 0:
                    face_img = frame.copy()
            else:
                face_img = frame.copy()  # no face detected: save full frame
            save_path = os.path.join(person_dir, f"{count+1}.jpg")
            cv2.imwrite(save_path, face_img)
            print(f"[SAVED] {save_path}")
            count += 1
            if count >= CAPTURE_COUNT:
                print("[INFO] Capture complete.")
                break

        # ESC to quit early
        elif key == 27:
            print("[INFO] Capture stopped by user.")
            break

    cam.release()
    cv2.destroyAllWindows()

# ---------- Helper: Train / encode faces ----------
def train_encodings():
    print("[INFO] Encoding faces from dataset...")
    known_encodings = []
    known_names = []

    if not os.path.exists(DATASET_DIR):
        print("[WARN] dataset folder missing.")
        return

    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue
        for img_name in os.listdir(person_folder):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(person_folder, img_name)
            try:
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if len(encs) > 0:
                    known_encodings.append(encs[0])
                    known_names.append(person_name)
                else:
                    print(f"[WARN] No face found in {person_name}/{img_name}")
            except Exception as e:
                print(f"[ERROR] Could not process {img_path}: {e}")

    if len(known_encodings) == 0:
        print("[ERROR] No encodings generated. Make sure dataset has clear face images.")
        return

    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Saved encodings for {len(known_names)} images to {ENCODINGS_FILE}")

# ---------- Helper: Start attendance ----------
def start_attendance():
    if not os.path.exists(ENCODINGS_FILE):
        print("[ERROR] Encodings file not found. Run 'Train / Encode faces' first.")
        return

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    date_today = datetime.now().strftime("%Y-%m-%d")
    csv_name = f"Attendance_{date_today}.csv"
    # create csv if not exists, with header Name,Time
    if not os.path.exists(csv_name):
        with open(csv_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time"])

    attendance_set = set()
    print("[INFO] Starting webcam for attendance. Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame from webcam.")
            break

        small = cv2.resize(frame, (0,0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, boxes)

        for encoding, box in zip(encodings, boxes):
            # compute distances and find best match
            if len(data["encodings"]) == 0:
                continue
            dists = face_recognition.face_distance(data["encodings"], encoding)
            best_idx = np.argmin(dists)
            name = "Unknown"
            # you can tune threshold; typical: 0.5 - 0.6
            if dists.size > 0 and dists[best_idx] < 0.55:
                name = data["names"][best_idx]

            # scale box back up
            top,right,bottom,left = [v * int(1/FRAME_SCALE) for v in box]
            color = (0,255,0) if name != "Unknown" else (0,0,255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if name != "Unknown" and name not in attendance_set:
                time_str = datetime.now().strftime("%H:%M:%S")
                with open(csv_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, time_str])
                attendance_set.add(name)
                print(f"[ATTENDANCE] {name} marked at {time_str}")

        cv2.imshow("Attendance (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Ending attendance session.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- Main menu ----------
def main_menu():
    while True:
        print("\n==============================")
        print(" SMART ATTENDANCE SYSTEM (CLI)")
        print("==============================")
        print("1. Capture new student face")
        print("2. Train / Encode faces")
        print("3. Start attendance")
        print("4. Exit")
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            capture_new_student()
        elif choice == "2":
            train_encodings()
        elif choice == "3":
            start_attendance()
        elif choice == "4":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Enter 1-4.")

if __name__ == "__main__":
    main_menu()
